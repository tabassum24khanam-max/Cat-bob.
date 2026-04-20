"""
ULTRAMAX Backend — FastAPI
All prediction logic, database queries, API orchestration
"""
import asyncio
import json
import os
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from config import BINANCE_SYMBOLS, ASSET_NAMES, WORKER_URL, ALL_ASSETS, get_asset_type, set_setting, is_configured
from database import (init_db, get_predictions, save_prediction, update_prediction_outcome,
                      similarity_search, get_news_history, get_macro_history, DB_PATH,
                      get_accuracy_stats, update_accuracy_stats, get_upcoming_events)
from data_fetcher import fetch_candles, fetch_candles_before, fetch_macro, fetch_fear_greed, fetch_onchain, fetch_current_price
from indicators import compute_indicators, monte_carlo
from ml_engine import predict_ensemble, train_ensemble, bayesian_confidence, extract_features
from agents.quant_agent import run_quant_agent, build_quant_prompt
from agents.news_agent import fetch_asset_news, filter_headlines_ai, run_news_agent
from agents.decision_agent import run_decision_agent
from sentiment import get_sentiment_snapshot
from macro_engine import get_macro_context
from correlation_engine import get_correlation_summary
from cluster_engine import assign_cluster
from alert_engine import scan_for_alerts, get_latest_alerts

app = FastAPI(title="ULTRAMAX Backend", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
frontend_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend'))
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
    print(f"✓ Frontend mounted from: {frontend_path}")


# ─── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    await init_db()
    print(f"✓ ULTRAMAX Backend started")
    print(f"  Database: {DB_PATH}")
    print(f"  Worker: {WORKER_URL}")
    # Pre-train ML ensemble from existing history
    asyncio.create_task(warm_ml_models())
    # Initial calendar refresh
    asyncio.create_task(_safe_refresh_calendar())


# ─── Models ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    asset: str
    horizon: int
    api_key: str
    ds_key: Optional[str] = None
    worker_url: Optional[str] = None
    use_r1: Optional[bool] = True

class OutcomeRequest(BaseModel):
    pred_id: str
    asset: str
    entry_price: float
    target_price: Optional[float] = None
    predicted_price: Optional[float] = None
    original_decision: Optional[str] = None
    decision: Optional[str] = None
    horizon: int
    api_key: Optional[str] = None
    worker_url: Optional[str] = None

class SavePredRequest(BaseModel):
    prediction: dict


# ─── Helpers ────────────────────────────────────────────────────────────────

def _score_prediction(decision: str, original_decision: str, entry_price: float,
                      outcome_price: float, predicted_price: float = None) -> str:
    """Score a prediction's accuracy.
    BUY/SELL: simple direction check.
    NO_TRADE with original_decision (gate-blocked): check if original direction was right.
    NO_TRADE pure abstention: SKIPPED — the system admitted uncertainty, don't penalize.
    """
    if not entry_price:
        return 'skipped'
    moved = outcome_price - entry_price
    moved_pct = moved / entry_price * 100

    if decision == 'BUY':
        return 'correct' if moved > 0 else 'wrong'
    if decision == 'SELL':
        return 'correct' if moved < 0 else 'wrong'

    # NO_TRADE — only score if there was a gate-blocked original decision
    direction = original_decision
    if not direction or direction == 'NO_TRADE':
        return 'skipped'

    # Gate-blocked trade: was the original direction right?
    if direction == 'BUY':
        return 'correct' if moved_pct > 0.3 else 'wrong'
    elif direction == 'SELL':
        return 'correct' if moved_pct < -0.3 else 'wrong'
    return 'skipped'


async def _safe_refresh_calendar():
    """Safely refresh economic calendar on startup."""
    await asyncio.sleep(3)
    try:
        from macro_engine import refresh_calendar
        await refresh_calendar()
    except Exception:
        pass


async def _safe_assign_cluster(asset: str, ind: dict) -> dict:
    """Safely assign cluster — returns {available: False} on error."""
    try:
        return await assign_cluster(asset, ind)
    except Exception:
        return {'available': False}


def build_state_vector(ind: dict) -> list:
    """Build 13-dimension feature vector for similarity search."""
    return [
        ind.get('rsi14', 50) / 100,
        max(-1, min(1, ind.get('macd_hist', 0) / (abs(ind.get('cur', 1)) * 0.001 + 0.0001))),
        max(-1, min(1, ind.get('dist_vwap', 0) / 5)),
        max(-1, min(1, ind.get('trend_slope', 0) / 0.3)),
        ind.get('trend_stability', 0),
        ind.get('vol_percentile', 50) / 100,
        max(-1, min(1, ind.get('momentum_score', 0) / 5)),
        ind.get('hurst_exp', 0.5),
        ind.get('entropy_ratio', 0.5),
        max(-1, min(1, ind.get('autocorr', 0))),
        ind.get('hmm_probs', {}).get('TRENDING', 0.33),
        ind.get('hmm_probs', {}).get('RANGING', 0.33),
        ind.get('hmm_probs', {}).get('VOLATILE', 0.33),
    ]


_ml_cache = {}  # asset -> trained ensemble model artifact


async def _async_retrain(asset_name: str):
    """Retrain ML ensemble in background after new feedback."""
    await asyncio.sleep(1)
    try:
        preds = await get_predictions(asset_name, 500)
        model = await train_ensemble(preds)
        if model:
            _ml_cache[asset_name] = model
            print(f"✓ ML retrained for {asset_name}: {model['n_samples']} samples")
    except Exception:
        pass


async def warm_ml_models():
    """Pre-train ML ensemble models from existing prediction history."""
    await asyncio.sleep(2)  # Let DB init complete
    try:
        all_preds = await get_predictions(limit=500)
        assets = list(set(p['asset'] for p in all_preds))
        for asset_name in assets:
            asset_preds = [p for p in all_preds if p['asset'] == asset_name]
            model = await train_ensemble(asset_preds)
            if model:
                _ml_cache[asset_name] = model
                print(f"✓ ML ensemble for {asset_name}: {model['n_samples']} samples, acc={model.get('train_accuracy', 0):.0%}")
    except Exception as e:
        print(f"⚠ ML warm-up skipped: {e}")


@app.post("/retrain")
async def retrain(asset_name: str = None):
    """Retrain ML ensemble after new feedback arrives."""
    try:
        preds = await get_predictions(asset_name, 500)
        model = await train_ensemble(preds)
        if model:
            key = asset_name or 'all'
            _ml_cache[key] = model
            return {"ok": True, "samples": model['n_samples'], "accuracy": model.get('train_accuracy', 0)}
        return {"ok": False, "reason": "Not enough rated predictions (need 20+)"}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    fp = os.path.join(frontend_path, 'index.html')
    if os.path.exists(fp):
        return FileResponse(fp)
    return {"status": "ULTRAMAX Backend v3.0", "endpoints": ["/predict", "/history", "/health"]}


@app.get("/app.js")
async def serve_appjs():
    fp = os.path.join(frontend_path, 'app.js')
    if os.path.exists(fp):
        return FileResponse(fp, media_type="application/javascript")
    raise HTTPException(404, "app.js not found")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0", "ts": int(time.time())}


@app.post("/predict")
async def predict(req: PredictRequest):
    """Main prediction endpoint — orchestrates all 3 agents."""
    from config import OPENAI_API_KEY, DEEPSEEK_API_KEY
    # Use server-side keys as fallback if frontend didn't provide them
    if not req.api_key or len(req.api_key) < 10:
        req.api_key = OPENAI_API_KEY
    if not req.ds_key or len(req.ds_key) < 10:
        req.ds_key = DEEPSEEK_API_KEY
    if not req.api_key or len(req.api_key) < 10:
        raise HTTPException(400, "No OpenAI API key configured — set OPENAI_API_KEY in Railway Variables")
    start_time = time.time()
    worker = req.worker_url or WORKER_URL
    logs = []

    def slog(msg: str):
        logs.append({"ts": int((time.time() - start_time) * 1000), "msg": msg})
        print(f"  [{logs[-1]['ts']}ms] {msg}")

    try:
        return await _run_prediction(req, worker, start_time, logs, slog)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        total_ms = int((time.time() - start_time) * 1000)
        raise HTTPException(500, f"Prediction failed after {total_ms}ms: {str(e)[:200]}")


async def _run_prediction(req, worker, start_time, logs, slog):
    """Internal prediction logic — separated so top-level can catch errors."""

    slog(f"🚀 PREDICT {req.asset} {req.horizon}H")

    # ── Fetch candles ────────────────────────────────────────────────────
    iv = '15m' if req.horizon <= 1 else '1h' if req.horizon <= 8 else '4h' if req.horizon <= 72 else '1d'
    slog(f"📊 Fetching candles ({iv})...")
    candles = await fetch_candles(req.asset, iv, 300)
    if not candles:
        raise HTTPException(400, "Failed to fetch candle data")
    slog(f"✓ {len(candles)} candles loaded")

    # ── Compute indicators ───────────────────────────────────────────────
    slog("⚙️ Computing 30+ indicators...")
    ind = compute_indicators(candles)
    if not ind:
        raise HTTPException(400, "Insufficient candle data for indicators")

    # Add recent price list for display
    recent_prices = [c['close'] for c in candles[-8:]]
    slog(f"✓ Indicators: RSI={ind['rsi14']:.1f} MACD={'▲' if ind['macd_hist']>0 else '▼'} Regime={ind['regime']}")

    # ── Monte Carlo ──────────────────────────────────────────────────────
    is_crypto = req.asset in BINANCE_SYMBOLS
    mc = monte_carlo(ind['cur'], ind['atr'], req.horizon, is_crypto)
    slog(f"✓ Monte Carlo: median={mc['median']:.4f} probUp={mc['prob_up']*100:.0f}%")

    # ── Historical similarity ────────────────────────────────────────────
    state_vec = build_state_vector(ind)
    slog("🔍 Running similarity search...")
    similar = await similarity_search(req.asset, state_vec, 50)
    if similar:
        wins = sum(1 for s in similar if
                   (s.get('fwd_4h') or 0) > 0.2)
        slog(f"✓ Found {len(similar)} similar periods — {wins/len(similar)*100:.0f}% were up")

    # ── Fetch parallel data ──────────────────────────────────────────────
    slog("🌐 Fetching macro, on-chain, sentiment, correlation, cluster in parallel...")
    asset_type = 'crypto' if is_crypto else 'macro' if req.asset in ['GC=F','CL=F','SI=F'] else 'stock'
    macro_data, fg_data, onchain_data, sentiment_data, macro_context, correlation_data, cluster_data = await asyncio.gather(
        fetch_macro(), fetch_fear_greed(), fetch_onchain(req.asset),
        get_sentiment_snapshot(req.asset),
        get_macro_context(req.asset, req.horizon),
        get_correlation_summary(req.asset),
        _safe_assign_cluster(req.asset, ind),
    )
    if sentiment_data.get('available'):
        slog(f"✓ Sentiment: score={sentiment_data.get('composite', 0):.2f}")
    if macro_context.get('warnings'):
        slog(f"⚠ Macro: {'; '.join(macro_context['warnings'][:2])}")
    if correlation_data.get('available'):
        slog(f"✓ Correlations loaded")
    if cluster_data.get('available'):
        slog(f"✓ Cluster #{cluster_data.get('cluster_id', '?')} ({cluster_data.get('members', 0)} members)")

    # ── Fetch news ───────────────────────────────────────────────────────
    slog("📰 Fetching news from 10+ sources...")
    articles = await fetch_asset_news(req.asset, ASSET_NAMES.get(req.asset, req.asset), asset_type)
    slog(f"✓ {len(articles)} headlines collected")

    if len(articles) > 8:
        slog("🤖 AI filtering headlines...")
        articles = await filter_headlines_ai(articles, req.asset, ASSET_NAMES.get(req.asset, req.asset),
                                              req.api_key, req.ds_key)
    slog(f"✓ {len(articles)} headlines after AI filter")

    # Get DB sentiment memory
    db_news = await get_news_history(req.asset, 24)
    db_sentiment = None
    if db_news:
        avg_24h = sum(r.get('avg_sentiment', 0) for r in db_news) / len(db_news)
        if len(db_news) >= 2:
            trend = db_news[-1].get('avg_sentiment', 0) - db_news[0].get('avg_sentiment', 0)
        else:
            trend = 0
        db_sentiment = {'hours': len(db_news), 'avg_24h': avg_24h, 'trend': trend}
        slog(f"✓ Sentiment memory: {len(db_news)}h history, avg={avg_24h:+.2f}, trend={trend:+.3f}")

    # ── MTF daily context ────────────────────────────────────────────────
    slog("📅 Fetching daily trend context...")
    daily_candles = await fetch_candles(req.asset, '1d', 32)
    mtf_data = {}
    if daily_candles and len(daily_candles) >= 20:
        # Use closed candles only (drop last)
        dc = daily_candles[:-1]
        daily_ind = compute_indicators(dc)
        if daily_ind:
            mtf_data = {
                'daily_macd_hist': daily_ind['macd_hist'],
                'daily_dist_e20': daily_ind['dist_e20'],
                'daily_bull': daily_ind['dist_e20'] > 1 and daily_ind['macd_hist'] > 0,
                'daily_bear': daily_ind['dist_e20'] < -1 and daily_ind['macd_hist'] < 0,
            }
            slog(f"✓ Daily: {'BULL' if mtf_data['daily_bull'] else 'BEAR' if mtf_data['daily_bear'] else 'NEUTRAL'}")

    # ── ML Ensemble ──────────────────────────────────────────────────────
    ml_result = {'confidence': 50, 'available': False}
    model_artifact = _ml_cache.get(req.asset)
    if model_artifact:
        ml_result = predict_ensemble(model_artifact, ind)
        slog(f"✓ ML ensemble: score={ml_result.get('score', 0):.2f} agree={ml_result.get('agreement', False)}")
    else:
        slog("⚠ ML ensemble: no trained model yet")

    # ── Agent 1: Quant ───────────────────────────────────────────────────
    slog("📐 Agent 1 (Quant) analyzing...")
    quant_prompt = build_quant_prompt(req.asset, ind, mc, req.horizon,
                                       cluster_data=cluster_data, correlation_data=correlation_data)
    quant_result = await run_quant_agent(req.asset, ind, mc, req.horizon, quant_prompt, req.api_key)
    slog(f"✓ Quant: {quant_result.get('direction')} {quant_result.get('confidence')}% — {quant_result.get('reasoning','')[:60]}")

    # ── Bayesian + 4-way confidence blending ────────────────────────────
    rated_preds = await get_predictions(req.asset, 200)
    raw_ai_conf = quant_result.get('confidence', 50)
    bayes_conf = bayesian_confidence(rated_preds, req.asset, req.horizon, raw_ai_conf)
    ml_conf = ml_result.get('score', 50) if ml_result['available'] else None
    cluster_conf = cluster_data.get('win_rate_4h') if cluster_data.get('available') else None

    if ml_conf is not None and cluster_conf is not None:
        # 4-way: ML leads — trained on 22K real samples at 70% accuracy
        blended_conf = raw_ai_conf * 0.15 + bayes_conf * 0.20 + ml_conf * 0.45 + cluster_conf * 0.20
    elif ml_conf is not None:
        # 3-way: ML dominant
        blended_conf = raw_ai_conf * 0.20 + bayes_conf * 0.25 + ml_conf * 0.55
    elif cluster_conf is not None:
        blended_conf = raw_ai_conf * 0.35 + bayes_conf * 0.35 + cluster_conf * 0.30
    else:
        blended_conf = raw_ai_conf * 0.55 + bayes_conf * 0.45
    quant_result['confidence'] = round(blended_conf)
    slog(f"✓ Blended: AI={raw_ai_conf}% Bayes={bayes_conf:.0f}% ML={ml_conf or 'N/A'} Cluster={cluster_conf or 'N/A'} → {quant_result['confidence']}%")

    # ── Agent 2: News ────────────────────────────────────────────────────
    slog(f"📰 Agent 2 (News/{'DeepSeek V3' if req.ds_key else 'GPT-4o-mini'}) analyzing...")
    news_result = await run_news_agent(
        req.asset, ASSET_NAMES.get(req.asset, req.asset), asset_type,
        articles, macro_data, onchain_data, fg_data, {},
        req.horizon, req.api_key, req.ds_key, db_sentiment
    )
    slog(f"✓ News: {news_result.get('sentiment')} ({news_result.get('sentiment_score',0):+d}) — {news_result.get('reasoning','')[:60]}")

    # ── Agent 3: Decision (R1) ───────────────────────────────────────────
    use_r1 = bool(req.ds_key) and (req.use_r1 is not False)
    model_name = 'R1' if use_r1 else ('V3' if req.ds_key else 'GPT-4o')
    slog(f"🧠 Agent 3 ({model_name}) making final decision...")
    decision = await run_decision_agent(
        req.asset, ind, req.horizon, quant_result, news_result,
        mtf_data, mc, similar, req.ds_key or '', req.api_key, use_r1
    )
    model_used = decision.get('_model', 'unknown')
    slog(f"✓ Decision: {decision.get('decision')} {decision.get('confidence')}% [{model_used}]")
    if model_used == 'gpt-4o' and decision.get('_r1_error'):
        slog(f"⚠ R1 fallback reason: {decision['_r1_error']}")

    # ── Post-processing gates ────────────────────────────────────────────
    gate_reason = None

    # WEEKEND + MARKET HOURS GATE (stocks only)
    from datetime import datetime, timedelta, timezone
    riyadh_tz = timezone(timedelta(hours=3))
    now_riyadh = datetime.now(riyadh_tz)
    if get_asset_type(req.asset) == 'stock':
        # Weekend gate: Saturday=5, Sunday=6
        if now_riyadh.weekday() in (5, 6):
            old_conf = decision.get('confidence', 50)
            decision['confidence'] = max(0, old_conf - 20)
            slog(f"Gate: Weekend — stocks confidence -20%")
        # Market hours gate: NYSE open 4:30 PM - 11:00 PM Riyadh (9:30 AM - 4:00 PM ET)
        riyadh_hour = now_riyadh.hour + now_riyadh.minute / 60.0
        if riyadh_hour < 16.5 or riyadh_hour >= 23.0:
            old_conf = decision.get('confidence', 50)
            decision['confidence'] = max(0, old_conf - 10)
            slog(f"Gate: NYSE closed — stocks confidence -10%")

    # MACRO BEAR/BULL VIX+DXY CAP GATES
    if decision.get('decision') != 'NO_TRADE':
        macro = macro_data or {}
        if macro.get('vix', 0) > 30 and macro.get('dxy', 0) > 107 and decision.get('decision') == 'BUY':
            if decision.get('confidence', 0) > 55:
                decision['confidence'] = 55
            slog(f"Gate: Macro Bear (VIX>30 + DXY>107) → BUY capped 55%")
        if macro.get('vix', 0) < 15 and macro.get('dxy', 0) < 100 and decision.get('decision') == 'SELL':
            if decision.get('confidence', 0) > 55:
                decision['confidence'] = 55
            slog(f"Gate: Macro Bull (VIX<15 + DXY<100) → SELL capped 55%")

    # MACRO BEAR/BULL GATE
    if decision.get('decision') != 'NO_TRADE':
        fg_val = fg_data.get('value', 50)
        news_score = news_result.get('sentiment_score', 0)
        bear_signals = [
            ind['macd_hist'] < 0,
            ind['dist_e20'] < 0,
            (macro_data.get('vix') or 0) > 18,
            news_score < 0,
            fg_val < 40
        ]
        bull_signals = [
            ind['macd_hist'] > 0,
            ind['dist_e20'] > 0,
            (macro_data.get('vix') or 0) < 15,
            news_score > 0,
            fg_val > 60
        ]
        bear_count = sum(bear_signals)
        bull_count = sum(bull_signals)

        if bear_count >= 4 and decision.get('decision') == 'BUY':
            original_decision = decision['decision']
            decision['_original_decision'] = original_decision
            decision['decision'] = 'NO_TRADE'
            gate_reason = f"🛡 MACRO BEAR GATE — {bear_count}/5 bearish signals"
            slog(f"🛡 MACRO BEAR GATE fired ({bear_count}/5)")
        elif bull_count >= 4 and decision.get('decision') == 'SELL':
            original_decision = decision['decision']
            decision['_original_decision'] = original_decision
            decision['decision'] = 'NO_TRADE'
            gate_reason = f"🛡 MACRO BULL GATE — {bull_count}/5 bullish signals"
            slog(f"🛡 MACRO BULL GATE fired ({bull_count}/5)")

    # VWAP HARD RULE: BUY below VWAP = -10 confidence
    if decision.get('decision') == 'BUY' and ind.get('dist_vwap', 0) < 0:
        old_conf = decision.get('confidence', 50)
        decision['confidence'] = max(0, old_conf - 10)
        slog(f"⚠ VWAP rule: BUY below VWAP — confidence {old_conf}% → {decision['confidence']}%")

    # COUNTER-TREND PENALTY: -15 confidence when signal opposes daily trend
    if decision.get('decision') != 'NO_TRADE' and mtf_data:
        daily_bull = mtf_data.get('daily_bull', False)
        daily_bear = mtf_data.get('daily_bear', False)
        if (daily_bear and decision.get('decision') == 'BUY') or \
           (daily_bull and decision.get('decision') == 'SELL'):
            old_conf = decision.get('confidence', 50)
            decision['confidence'] = max(0, old_conf - 15)
            slog(f"⚠ Counter-trend penalty: {decision['decision']} vs daily {'BEAR' if daily_bear else 'BULL'} — confidence {old_conf}% → {decision['confidence']}%")

    # ICHIMOKU INSIDE CLOUD GATE: price inside cloud + very low confluence = NO_TRADE
    if decision.get('decision') != 'NO_TRADE':
        inside_cloud = not ind.get('ich_bull') and not ind.get('ich_bear')
        very_low_confluence = decision.get('confidence', 0) < 42
        if inside_cloud and very_low_confluence:
            decision['_original_decision'] = decision.get('decision')
            decision['decision'] = 'NO_TRADE'
            gate_reason = f"Ichimoku inside cloud + low confluence ({decision.get('confidence', 0)}% < 42%)"
            slog(f"⚠ Ichimoku cloud gate: inside cloud + confidence {decision.get('confidence', 0)}%")

    # Confidence cap: neutral daily + bearish MACD
    if decision.get('decision') != 'NO_TRADE':
        daily_neutral = not mtf_data.get('daily_bull') and not mtf_data.get('daily_bear')
        daily_macd_bearish = mtf_data.get('daily_macd_hist', 0) < 0
        if daily_neutral and daily_macd_bearish and decision.get('confidence', 0) > 65:
            slog(f"⚠ Confidence capped: {decision['confidence']}% → 65% (neutral daily + bearish MACD)")
            decision['confidence'] = 65

    # Hurst gate — random walk penalty (cap, not NO_TRADE)
    if decision.get('decision') != 'NO_TRADE':
        hurst = ind.get('hurst_exp', 0.5)
        if 0.45 <= hurst <= 0.55 and decision.get('confidence', 0) > 60:
            decision['confidence'] = 60
            slog(f"⚠ Hurst gate: {hurst:.3f} random walk — capped 60%")

    # GATE 7: FUNDING RATE EXTREME — BUY + funding > 0.08% = -15 confidence
    if decision.get('decision') == 'BUY' and is_crypto:
        funding = onchain_data.get('funding_rate', 0) if onchain_data else 0
        if funding > 0.0008:
            old_conf = decision.get('confidence', 50)
            decision['confidence'] = max(0, old_conf - 15)
            slog(f"⚠ Funding rate gate: rate={funding:.4%} — confidence {old_conf}% → {decision['confidence']}%")

    # GATE 8: AGENT CONFLICT — Quant vs News disagree = -10 confidence (not NO_TRADE)
    if decision.get('decision') != 'NO_TRADE':
        q_dir = quant_result.get('direction', '').upper()
        n_sent = news_result.get('sentiment', '').upper()
        agents_disagree = (q_dir == 'BUY' and n_sent == 'BEARISH') or (q_dir == 'SELL' and n_sent == 'BULLISH')
        if agents_disagree:
            old_conf = decision.get('confidence', 50)
            decision['confidence'] = max(0, old_conf - 10)
            slog(f"⚠ Agent conflict: Quant={q_dir} vs News={n_sent} — confidence {old_conf}% → {decision['confidence']}%")

    # GATE 9: CMF CONTRADICTION — BUY + CMF<-0.1 or SELL + CMF>0.1 = -15 confidence
    if decision.get('decision') != 'NO_TRADE':
        cmf = ind.get('cmf', 0)
        if (decision['decision'] == 'BUY' and cmf < -0.1) or (decision['decision'] == 'SELL' and cmf > 0.1):
            old_conf = decision.get('confidence', 50)
            decision['confidence'] = max(0, old_conf - 15)
            slog(f"⚠ CMF contradiction: {decision['decision']} but CMF={cmf:.3f} — confidence {old_conf}% → {decision['confidence']}%")

    # GATE 10: OBV DIVERGENCE — BUY + OBV falling or SELL + OBV rising = -15 confidence
    if decision.get('decision') != 'NO_TRADE':
        obv_slope = ind.get('obv_slope', 0)
        if (decision['decision'] == 'BUY' and obv_slope < -0.1) or (decision['decision'] == 'SELL' and obv_slope > 0.1):
            old_conf = decision.get('confidence', 50)
            decision['confidence'] = max(0, old_conf - 15)
            slog(f"⚠ OBV divergence: {decision['decision']} but OBV slope={obv_slope:.3f} — confidence {old_conf}% → {decision['confidence']}%")

    # GATE 11: CLUSTER BOUNDARY — <10 members in cluster = cap 60%
    if decision.get('decision') != 'NO_TRADE' and cluster_data.get('available'):
        if cluster_data.get('members', 0) < 10 and decision.get('confidence', 0) > 60:
            slog(f"⚠ Cluster boundary: only {cluster_data['members']} members — capped 60%")
            decision['confidence'] = 60

    # GATE 12: ML AGREEMENT — ensemble disagrees with decision = cap 55%
    if decision.get('decision') != 'NO_TRADE' and ml_result.get('available'):
        ml_score = ml_result.get('score', 50)
        ml_says_up = ml_score > 55
        ml_says_down = ml_score < 45
        if (decision['decision'] == 'BUY' and ml_says_down) or (decision['decision'] == 'SELL' and ml_says_up):
            if decision.get('confidence', 0) > 55:
                slog(f"⚠ ML disagreement: ML score={ml_score:.1f} vs {decision['decision']} — capped 55%")
                decision['confidence'] = 55

    # GATE 13: CONFIDENCE FLOOR — <38% = NO_TRADE
    if decision.get('decision') != 'NO_TRADE' and decision.get('confidence', 0) < 38:
        decision['_original_decision'] = decision.get('decision')
        decision['decision'] = 'NO_TRADE'
        gate_reason = f"Confidence floor: {decision.get('confidence', 0)}% < 38%"
        slog(f"⚠ Confidence floor gate: {decision.get('confidence', 0)}%")

    # GATE 14: UPCOMING EVENT — high-impact event within horizon = -15 confidence
    if decision.get('decision') != 'NO_TRADE' and macro_context.get('upcoming_events'):
        high_impact = [e for e in macro_context['upcoming_events'] if e.get('impact') == 'high']
        if high_impact:
            old_conf = decision.get('confidence', 50)
            decision['confidence'] = max(0, old_conf - 15)
            slog(f"⚠ Event gate: {len(high_impact)} high-impact event(s) ahead — confidence {old_conf}% → {decision['confidence']}%")

    # GATE 15: ENTROPY GATE — Shannon entropy > 0.9 = cap 55%
    if decision.get('decision') != 'NO_TRADE':
        entropy = ind.get('entropy_ratio', 0.5)
        if entropy > 0.9 and decision.get('confidence', 0) > 55:
            slog(f"⚠ Entropy gate: ratio={entropy:.3f} (noise) — capped 55%")
            decision['confidence'] = 55

    # ML CONFIDENCE BOOST — when ML strongly agrees, boost confidence
    if decision.get('decision') != 'NO_TRADE' and ml_result.get('available'):
        ml_score = ml_result.get('score', 50)
        ml_agrees = (decision['decision'] == 'BUY' and ml_score > 62) or \
                    (decision['decision'] == 'SELL' and ml_score < 38)
        if ml_agrees:
            old_conf = decision.get('confidence', 50)
            boost = min(12, int((abs(ml_score - 50) - 12) * 0.8))
            decision['confidence'] = min(85, old_conf + boost)
            slog(f"✓ ML boost: ML={ml_score:.1f} agrees with {decision['decision']} → +{boost}% (confidence {old_conf}% → {decision['confidence']}%)")

    # Save predicted price BEFORE nulling target for NO_TRADE
    predicted_price = None
    if decision.get('decision') == 'NO_TRADE':
        predicted_price = decision.get('price_target') or mc.get('median')
        decision['price_target'] = None
        decision['price_target_bull'] = None
        decision['price_target_bear'] = None

    total_ms = int((time.time() - start_time) * 1000)
    slog(f"✅ Complete in {total_ms}ms")

    return {
        "decision": decision.get('decision', 'NO_TRADE'),
        "confidence": decision.get('confidence', 50),
        "prob_up": decision.get('prob_up', mc['prob_up'] * 100),
        "prob_down": decision.get('prob_down', (1 - mc['prob_up']) * 100),
        "price_target": decision.get('price_target'),
        "price_target_bull": decision.get('price_target_bull', mc['bull']),
        "price_target_bear": decision.get('price_target_bear', mc['bear']),
        "predicted_price": predicted_price,
        "predicted_path": decision.get('predicted_path', []),
        "insight": decision.get('insight', ''),
        "primary_reason": decision.get('primary_reason', ''),
        "agent_model": decision.get('_model', 'gpt-4o'),
        "original_decision": decision.get('_original_decision'),
        "gate_reason": gate_reason,
        "agent_agreement": decision.get('agent_agreement', 'partial'),
        "volatility": decision.get('volatility', 'moderate'),
        # Agent details
        "quant": {
            "direction": quant_result.get('direction'),
            "confidence": quant_result.get('confidence'),
            "prob_up": quant_result.get('prob_up'),
            "reasoning": quant_result.get('reasoning'),
            "key_levels": quant_result.get('key_levels', {})
        },
        "news": {
            "sentiment": news_result.get('sentiment'),
            "sentiment_score": news_result.get('sentiment_score'),
            "confidence": news_result.get('confidence'),
            "market_regime": news_result.get('market_regime'),
            "reasoning": news_result.get('reasoning'),
            "key_catalysts": news_result.get('key_catalysts', []),
            "macro_warning": news_result.get('macro_warning'),
        },
        # Indicator snapshot (full, for ML training)
        "ind": {
            "cur": ind['cur'], "atr": ind['atr'],
            "rsi14": ind['rsi14'], "macd_hist": ind['macd_hist'],
            "regime": ind['regime'], "hmm_probs": ind['hmm_probs'],
            "hurst_exp": ind['hurst_exp'], "entropy_ratio": ind['entropy_ratio'],
            "dist_vwap": ind['dist_vwap'], "vwap": ind['vwap'],
            "kalman_trend": ind['kalman_trend'], "kalman_uncertainty": ind['kalman_uncertainty'],
            "ich_bull": ind['ich_bull'], "ich_bear": ind['ich_bear'],
            "supertrend_bull": ind['supertrend_bull'],
            "cmf": ind['cmf'], "price_zscore": ind['price_zscore'],
            "dist_e20": ind['dist_e20'], "dist_e50": ind['dist_e50'],
            "will_r14": ind['will_r14'], "momentum_score": ind['momentum_score'],
            "autocorr": ind['autocorr'], "trend_slope": ind['trend_slope'],
            "trend_stability": ind['trend_stability'], "vol_percentile": ind['vol_percentile'],
            "vol_r": ind['vol_r'], "stoch_k": ind['stoch_k'],
            "bb_pos": ind['bb_pos'], "poc": ind['poc'], "dist_poc": ind['dist_poc'],
        },
        "ind_snapshot": json.dumps(ind),  # full snapshot for ML training
        # ML ensemble
        "ml": ml_result,
        # New engines
        "sentiment": sentiment_data,
        "macro_context": macro_context,
        "correlation": correlation_data,
        "cluster": cluster_data,
        # MC results
        "monte_carlo": mc,
        # Similarity
        "similarity": {
            "count": len(similar),
            "win_rate": sum(1 for s in similar if (s.get('fwd_4h') or 0) > 0) / len(similar) * 100 if similar else 0,
            "avg_fwd_4h": sum(s.get('fwd_4h') or 0 for s in similar) / len(similar) if similar else 0,
        },
        "bayesian_conf": bayes_conf,
        "raw_ai_conf": raw_ai_conf,
        # DB sentiment memory
        "db_sentiment": db_sentiment,
        # Candle data for chart (last 100)
        "candles": candles[-100:],
        "recent_prices": recent_prices,
        # System log
        "logs": logs,
        "duration_ms": total_ms,
    }


@app.get("/history")
async def get_history(asset: str = None, limit: int = 200):
    preds = await get_predictions(asset, limit)
    return {"predictions": preds, "total": len(preds)}


@app.post("/ml/sync-and-retrain")
async def sync_and_retrain(req: SavePredRequest):
    """Receive all predictions from frontend, save them, backfill indicators, retrain.
    Does everything in one shot — no separate button clicks needed.
    """
    import aiosqlite
    predictions = req.prediction  # actually a list sent as the 'prediction' field
    if isinstance(predictions, dict):
        predictions = [predictions]

    # Step 1: Save all predictions
    saved = 0
    for p in predictions:
        try:
            await save_prediction(p)
            saved += 1
        except Exception:
            pass

    # Step 2: Backfill missing indicators
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT id, asset, saved_at FROM predictions "
            "WHERE ind_snapshot IS NULL OR ind_snapshot = '' OR ind_snapshot = 'null'"
        )
        missing = [dict(r) for r in await cur.fetchall()]

    backfilled = 0
    for p in missing:
        try:
            a = p['asset']
            ts = int(float(p.get('saved_at', 0))) // 1000
            if not ts:
                continue
            candles = await fetch_candles_before(a, ts, '1h', 300)
            candles = [c for c in candles if c.get('time', 0) <= ts]
            if len(candles) < 60:
                continue
            ind = compute_indicators(candles[-300:])
            if not ind:
                continue
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("UPDATE predictions SET ind_snapshot=? WHERE id=?",
                                 (json.dumps(ind), p['id']))
                await db.commit()
            backfilled += 1
        except Exception:
            pass

    # Step 3: Retrain
    try:
        preds = await get_predictions(limit=2000)
        model = await train_ensemble(preds)
        if not model or model.get('ok') is False:
            error = model.get('error', 'Training failed') if model else 'No result'
            return {"ok": False, "reason": error, "saved": saved, "backfilled": backfilled}
        _ml_cache['all'] = model
        cv = model.get('cv_accuracy', {})
        avg_cv = (cv.get('xgb', 0) + cv.get('rf', 0)) / 2 if cv else 0
        return {
            "ok": True, "saved": saved, "backfilled": backfilled,
            "samples": model.get('n_train', 0), "accuracy": round(avg_cv, 1),
        }
    except Exception as e:
        return {"ok": False, "reason": str(e), "saved": saved, "backfilled": backfilled}


@app.post("/history/save")
async def save_history(req: SavePredRequest):
    await save_prediction(req.prediction)
    return {"ok": True}


@app.get("/predictions/export")
async def export_predictions(asset: str = None):
    """Export all predictions as formatted text log."""
    from datetime import datetime

    def _ts(ms):
        try:
            return datetime.utcfromtimestamp(float(ms) / 1000).strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception:
            return 'N/A'

    def _f(v):
        try:
            return float(v)
        except Exception:
            return 0

    try:
        preds = await get_predictions(asset, limit=5000)
        preds.sort(key=lambda p: p.get('saved_at') or 0)

        lines = ["=" * 80, "ULTRAMAX PREDICTION EXPORT",
                 f"Total predictions: {len(preds)}",
                 f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                 "=" * 80]

        for i, p in enumerate(preds, 1):
            entry = _f(p.get('entry_price'))
            target = _f(p.get('target_price') or p.get('predicted_price'))
            outcome = _f(p.get('outcome_price'))
            conf = _f(p.get('confidence'))
            decision = p.get('decision') or '?'
            orig = p.get('original_decision') or ''
            feedback = p.get('feedback') or 'pending'

            pct = f"{((outcome - entry) / entry * 100):+.2f}%" if entry and outcome else 'N/A'

            lines.append("")
            lines.append("-" * 80)
            lines.append(f"#{i}  {p.get('asset','?')} | {decision} | {p.get('horizon','?')}H")
            lines.append("-" * 80)
            lines.append(f"  Predicted at:     {_ts(p.get('saved_at'))}")
            lines.append(f"  Outcome at:       {_ts(p.get('outcome_at'))}")
            lines.append(f"  Model:            {p.get('agent_model') or 'N/A'}")
            lines.append(f"  Decision:         {decision}" + (f"  (original: {orig})" if orig and orig != decision else ""))
            lines.append(f"  Confidence:       {conf}%")
            lines.append(f"  Entry price:      ${entry:,.4f}" if entry else "  Entry price:      N/A")
            lines.append(f"  Predicted target: ${target:,.4f}" if target else "  Predicted target: N/A")
            lines.append(f"  Bull target:      ${_f(p.get('target_bull')):,.4f}" if p.get('target_bull') else "  Bull target:      N/A")
            lines.append(f"  Bear target:      ${_f(p.get('target_bear')):,.4f}" if p.get('target_bear') else "  Bear target:      N/A")
            lines.append(f"  Actual at expiry: ${outcome:,.4f}" if outcome else "  Actual at expiry: N/A")
            lines.append(f"  Price change:     {pct}")
            lines.append(f"  Prob up/down:     {p.get('prob_up','N/A')} / {p.get('prob_down','N/A')}")
            lines.append(f"  Outcome:          {str(feedback).upper()}")
            lines.append(f"  Target hit:       {p.get('target_hit', 'N/A')}")
            lines.append(f"  Gate reason:      {p.get('gate_reason') or 'none'}")
            lines.append(f"  ML score:         {p.get('ml_score', 'N/A')}")
            lines.append(f"  Quant verdict:    {p.get('quant_verdict') or 'N/A'}")
            lines.append(f"  News verdict:     {p.get('news_verdict') or 'N/A'}")
            lines.append(f"  Primary reason:   {p.get('primary_reason') or 'N/A'}")

            insight = p.get('insight')
            if insight:
                lines.append(f"  R1 Reasoning:")
                for ln in str(insight).split('\n'):
                    lines.append(f"    {ln.strip()}")

            note = p.get('feedback_note')
            if note:
                lines.append(f"  Feedback note:    {note}")

            ind_snap = p.get('ind_snapshot')
            if ind_snap:
                try:
                    ind = json.loads(ind_snap) if isinstance(ind_snap, str) else ind_snap
                    lines.append(f"  --- Indicators ---")
                    lines.append(f"  RSI14={ind.get('rsi14')}  StochK={ind.get('stoch_k')}  MACD={ind.get('macd_hist')}")
                    lines.append(f"  BB={ind.get('bb_pos')}  ATR={ind.get('atr')}  VolR={ind.get('vol_r')}")
                    lines.append(f"  Trend={ind.get('trend_slope')}  Hurst={ind.get('hurst_exp')}  Entropy={ind.get('entropy_ratio')}")
                    lines.append(f"  Regime={ind.get('regime')}  VWAP={ind.get('dist_vwap')}  CMF={ind.get('cmf')}")
                except Exception:
                    lines.append(f"  Indicators: (unparseable)")
            else:
                lines.append(f"  Indicators:       MISSING")

        lines.extend(["", "=" * 80, "END OF EXPORT", "=" * 80])
        return PlainTextResponse("\n".join(lines), headers={
            "Content-Disposition": "attachment; filename=ultramax_predictions_export.txt"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return PlainTextResponse(f"Export error: {str(e)}", status_code=500)


@app.post("/ml/retrain")
async def ml_retrain():
    """Retrain ML ensemble on rated prediction history."""
    try:
        preds = await get_predictions(limit=2000)
        model = await train_ensemble(preds)
        if not model:
            return {"ok": False, "reason": "No result from train_ensemble"}
        if model.get('ok') is False:
            return {"ok": False, "reason": model.get('error', 'Training failed')}
        _ml_cache['all'] = model
        cv = model.get('cv_accuracy', {})
        avg_cv = (cv.get('xgb', 0) + cv.get('rf', 0)) / 2 if cv else 0
        return {
            "ok": True,
            "samples": model.get('n_train', 0),
            "accuracy": round(avg_cv, 1),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "reason": str(e)}


@app.post("/ml/backfill")
async def ml_backfill():
    """Reconstruct ind_snapshot for existing predictions by re-fetching historical
    candles at each prediction's saved_at timestamp and recomputing indicators.
    """
    try:
        import aiosqlite

        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT id, asset, saved_at, horizon FROM predictions "
                "WHERE ind_snapshot IS NULL OR ind_snapshot = '' OR ind_snapshot = 'null' "
                "ORDER BY saved_at DESC"
            )
            rows = [dict(r) for r in await cur.fetchall()]

        total = len(rows)
        if total == 0:
            return {"ok": True, "total": 0, "success": 0, "failed": 0, "errors": ["All predictions already have indicator data"]}

        success = 0
        failed = 0
        errors = []

        for p in rows:
            try:
                a = p['asset']
                saved_at_ms = p.get('saved_at', 0)
                if not saved_at_ms:
                    failed += 1
                    continue
                saved_at_sec = int(float(saved_at_ms)) // 1000

                candles = await fetch_candles_before(a, saved_at_sec, '1h', 300)
                candles = [c for c in candles if c.get('time', 0) <= saved_at_sec]

                if len(candles) < 60:
                    failed += 1
                    if len(errors) < 15:
                        errors.append(f"{a}: only {len(candles)} candles")
                    continue

                ind = compute_indicators(candles[-300:])
                if not ind:
                    failed += 1
                    if len(errors) < 15:
                        errors.append(f"{a}: indicators returned None")
                    continue

                async with aiosqlite.connect(DB_PATH) as db:
                    await db.execute(
                        "UPDATE predictions SET ind_snapshot=? WHERE id=?",
                        (json.dumps(ind), p['id']),
                    )
                    await db.commit()
                success += 1
            except Exception as e:
                failed += 1
                if len(errors) < 15:
                    errors.append(f"{p.get('asset')}: {str(e)}")

        return {"ok": True, "total": total, "success": success, "failed": failed, "errors": errors}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "reason": str(e)}


@app.get("/history/check-all")
async def check_all_outcomes(asset: str = None):
    """Check outcomes for all expired unresolved predictions."""
    preds = await get_predictions(asset, 500)
    now = int(time.time())
    resolved = []

    for p in preds:
        if p.get('feedback'):
            continue  # already resolved
        expires_at = (p.get('saved_at', 0) // 1000) + (p.get('horizon', 4) * 3600)
        if now < expires_at:
            continue  # not expired yet

        # Fetch outcome price
        try:
            asset_sym = p['asset']
            result = await fetch_current_price(asset_sym)
            price = result['price']

            entry_price = p.get('entry_price', 0)
            orig = p.get('original_decision')
            decision = p.get('decision', 'NO_TRADE')
            pred_price = p.get('predicted_price') or p.get('target_price')
            feedback = _score_prediction(decision, orig, entry_price, price, pred_price)

            target_hit = None
            if p.get('target_price') and entry_price:
                tp = (p['target_price'] - entry_price) / entry_price * 100
                ap = moved / entry_price * 100
                target_hit = (ap >= tp * 0.8) if tp > 0 else (ap <= tp * 0.8)

            note = f"{entry_price:.4f} → {price:.4f} ({moved/entry_price*100:+.2f}%)"
            await update_prediction_outcome(p['id'], price, now, feedback, target_hit, note)
            resolved.append({'id': p['id'], 'asset': asset_sym, 'feedback': feedback, 'price': price})

        except Exception as e:
            continue

    return {"resolved": len(resolved), "items": resolved}


@app.post("/history/outcome")
async def check_outcome(req: OutcomeRequest):
    """Fetch price at horizon expiry and score prediction."""
    try:
        # Fetch current price
        result = await fetch_current_price(req.asset)
        price = result['price']

        moved = price - req.entry_price
        moved_pct = moved / req.entry_price * 100

        decision = req.decision or 'NO_TRADE'
        orig = req.original_decision
        pred_price = req.predicted_price or req.target_price
        feedback = _score_prediction(decision, orig, req.entry_price, price, pred_price)

        # Check target hit
        target_hit = None
        if req.target_price:
            target_pct = (req.target_price - req.entry_price) / req.entry_price * 100
            actual_pct = moved_pct
            if target_pct > 0:
                target_hit = actual_pct >= target_pct * 0.8
            elif target_pct < 0:
                target_hit = actual_pct <= target_pct * 0.8

        note = f"{req.entry_price:.4f} → {price:.4f} ({moved_pct:+.2f}%)"

        await update_prediction_outcome(
            req.pred_id, price, int(time.time()),
            feedback, target_hit, note
        )

        # Trigger ML retrain asynchronously
        asyncio.create_task(_async_retrain(req.asset))

        return {
            "price": price,
            "moved_pct": moved_pct,
            "feedback": feedback,
            "target_hit": target_hit,
            "note": note
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/price")
async def get_price(asset: str):
    """Get current price for an asset."""
    try:
        result = await fetch_current_price(asset)
        return {"price": result['price'], "chg": result.get('chg'), "asset": asset}
    except Exception as e:
        print(f"⚠ /price error for {asset}: {e}")
        return {"price": None, "chg": None, "asset": asset, "error": str(e)[:100]}


@app.get("/candles")
async def get_candles(asset: str, interval: str = '1h', limit: int = 100):
    try:
        candles = await fetch_candles(asset, interval, limit)
        return {"candles": candles, "asset": asset}
    except Exception as e:
        print(f"⚠ /candles error for {asset}: {e}")
        return {"candles": [], "asset": asset, "error": str(e)[:100]}


@app.get("/macro")
async def get_macro():
    data = await fetch_macro()
    fg = await fetch_fear_greed()
    data['fear_greed'] = fg
    return data


@app.get("/accuracy")
async def get_accuracy(asset: str = None):
    """Get prediction accuracy statistics."""
    try:
        stats = await get_accuracy_stats(asset)
        return {"stats": stats, "asset": asset}
    except Exception as e:
        return {"stats": [], "error": str(e)}


@app.get("/market_data/{asset}")
async def get_market_data(asset: str):
    """Get comprehensive market data for an asset."""
    try:
        price_result, sentiment_data, correlation_data, macro_ctx = await asyncio.gather(
            fetch_current_price(asset),
            get_sentiment_snapshot(asset),
            get_correlation_summary(asset),
            get_macro_context(asset, 4),
        )
        return {
            "asset": asset,
            "price": price_result.get('price'),
            "sentiment": sentiment_data,
            "correlation": correlation_data,
            "macro": macro_ctx,
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/alerts")
async def get_alerts():
    """Get latest confluence alerts."""
    try:
        alerts = get_latest_alerts()
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        return {"alerts": [], "error": str(e)}


class SettingsRequest(BaseModel):
    settings: dict


@app.get("/settings")
async def get_settings():
    """Get current backend settings (which optional keys are configured)."""
    return {
        "fred_configured": is_configured('FRED_API_KEY'),
        "reddit_configured": is_configured('REDDIT_CLIENT_ID'),
        "alpaca_configured": is_configured('ALPACA_KEY'),
        "openai_configured": is_configured('OPENAI_API_KEY'),
        "deepseek_configured": is_configured('DEEPSEEK_API_KEY'),
    }


@app.post("/settings")
async def save_settings(req: SettingsRequest):
    """Save API key settings from frontend."""
    saved = []
    for key, value in req.settings.items():
        if key in ('FRED_API_KEY', 'REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET',
                    'ALPACA_KEY', 'ALPACA_SECRET') and value:
            os.environ[key] = value
            set_setting(key, value)
            saved.append(key)
    return {"ok": True, "saved": saved}


@app.get("/db/status")
async def db_status():
    """Database statistics."""
    import aiosqlite
    async with aiosqlite.connect(DB_PATH) as db:
        stats = {}
        tables = ['price_data', 'news_sentiment', 'articles', 'predictions',
                  'sentiment_snapshots', 'macro_events', 'accuracy_stats', 'pattern_memory', 'clusters']
        for table in tables:
            try:
                cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
                row = await cursor.fetchone()
                stats[table] = row[0]
            except Exception:
                stats[table] = 0
        # Latest data per asset
        cursor = await db.execute(
            "SELECT asset, MAX(ts) as latest, COUNT(*) as cnt FROM price_data GROUP BY asset"
        )
        assets = await cursor.fetchall()
        stats['assets'] = [{'asset': r[0], 'latest': r[1], 'hours': r[2]} for r in assets]
    return stats
