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
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from config import BINANCE_SYMBOLS, ASSET_NAMES, WORKER_URL, ALL_ASSETS, get_asset_type, set_setting, is_configured
from database import (init_db, get_predictions, save_prediction, update_prediction_outcome,
                      similarity_search, get_news_history, get_macro_history, DB_PATH,
                      get_accuracy_stats, update_accuracy_stats, get_upcoming_events)
from data_fetcher import fetch_candles, fetch_macro, fetch_fear_greed, fetch_onchain, fetch_current_price
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
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


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

class OutcomeRequest(BaseModel):
    pred_id: str
    asset: str
    entry_price: float
    target_price: Optional[float] = None
    original_decision: Optional[str] = None
    horizon: int
    api_key: Optional[str] = None
    worker_url: Optional[str] = None

class SavePredRequest(BaseModel):
    prediction: dict


# ─── Helpers ────────────────────────────────────────────────────────────────

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
        return assign_cluster(asset, ind)
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
        model = train_ensemble(preds)
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
            model = train_ensemble(asset_preds)
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
        model = train_ensemble(preds)
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


@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0", "ts": int(time.time())}


@app.post("/predict")
async def predict(req: PredictRequest):
    """Main prediction endpoint — orchestrates all 3 agents."""
    start_time = time.time()
    worker = req.worker_url or WORKER_URL
    logs = []

    def slog(msg: str):
        logs.append({"ts": int((time.time() - start_time) * 1000), "msg": msg})
        print(f"  [{logs[-1]['ts']}ms] {msg}")

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
    ml_conf = ml_result.get('score', 0.5) * 100 if ml_result['available'] else None
    cluster_conf = cluster_data.get('win_rate_4h') if cluster_data.get('available') else None

    if ml_conf is not None and cluster_conf is not None:
        # 4-way: AI 25% + Bayes 30% + ML 25% + Cluster 20%
        blended_conf = raw_ai_conf * 0.25 + bayes_conf * 0.30 + ml_conf * 0.25 + cluster_conf * 0.20
    elif ml_conf is not None:
        # 3-way: AI 35% + Bayes 35% + ML 30%
        blended_conf = raw_ai_conf * 0.35 + bayes_conf * 0.35 + ml_conf * 0.30
    elif cluster_conf is not None:
        # 3-way: AI 35% + Bayes 35% + Cluster 30%
        blended_conf = raw_ai_conf * 0.35 + bayes_conf * 0.35 + cluster_conf * 0.30
    else:
        # 2-way: AI 55% + Bayes 45%
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
    use_r1 = bool(req.ds_key)
    slog(f"🧠 Agent 3 ({'R1' if use_r1 else 'GPT-4o'}) making final decision...")
    decision = await run_decision_agent(
        req.asset, ind, req.horizon, quant_result, news_result,
        mtf_data, mc, similar, req.ds_key or '', req.api_key, use_r1
    )
    slog(f"✓ Decision: {decision.get('decision')} {decision.get('confidence')}% [{decision.get('_model')}]")

    # ── Post-processing gates ────────────────────────────────────────────
    gate_reason = None

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

    # ICHIMOKU INSIDE CLOUD GATE: price inside cloud + low confluence = NO_TRADE
    if decision.get('decision') != 'NO_TRADE':
        inside_cloud = not ind.get('ich_bull') and not ind.get('ich_bear')
        low_confluence = decision.get('confidence', 0) < 60
        if inside_cloud and low_confluence:
            decision['_original_decision'] = decision.get('decision')
            decision['decision'] = 'NO_TRADE'
            gate_reason = f"Ichimoku inside cloud + low confluence ({decision.get('confidence', 0)}% < 60%)"
            slog(f"⚠ Ichimoku cloud gate: inside cloud + confidence {decision.get('confidence', 0)}%")

    # Confidence cap: neutral daily + bearish MACD
    if decision.get('decision') != 'NO_TRADE':
        daily_neutral = not mtf_data.get('daily_bull') and not mtf_data.get('daily_bear')
        daily_macd_bearish = mtf_data.get('daily_macd_hist', 0) < 0
        if daily_neutral and daily_macd_bearish and decision.get('confidence', 0) > 65:
            slog(f"⚠ Confidence capped: {decision['confidence']}% → 65% (neutral daily + bearish MACD)")
            decision['confidence'] = 65

    # Hurst gate
    if decision.get('decision') != 'NO_TRADE':
        hurst = ind.get('hurst_exp', 0.5)
        if 0.45 <= hurst <= 0.55 and decision.get('confidence', 0) < 65:
            decision['_original_decision'] = decision.get('decision')
            decision['decision'] = 'NO_TRADE'
            gate_reason = f"Hurst={hurst:.3f} random walk — need 65%+ confidence"
            slog(f"⚠ Hurst gate: {hurst:.3f}")

    # GATE 7: FUNDING RATE EXTREME — BUY + funding > 0.08% = -15 confidence
    if decision.get('decision') == 'BUY' and is_crypto:
        funding = onchain_data.get('funding_rate', 0) if onchain_data else 0
        if funding > 0.0008:
            old_conf = decision.get('confidence', 50)
            decision['confidence'] = max(0, old_conf - 15)
            slog(f"⚠ Funding rate gate: rate={funding:.4%} — confidence {old_conf}% → {decision['confidence']}%")

    # GATE 8: AGENT CONFLICT — Quant vs News disagree + <72% = NO_TRADE
    if decision.get('decision') != 'NO_TRADE':
        q_dir = quant_result.get('direction', '').upper()
        n_sent = news_result.get('sentiment', '').upper()
        agents_disagree = (q_dir == 'BUY' and n_sent == 'BEARISH') or (q_dir == 'SELL' and n_sent == 'BULLISH')
        if agents_disagree and decision.get('confidence', 0) < 72:
            decision['_original_decision'] = decision.get('decision')
            decision['decision'] = 'NO_TRADE'
            gate_reason = f"Agent conflict: Quant={q_dir} vs News={n_sent} + confidence < 72%"
            slog(f"⚠ Agent conflict gate: {q_dir} vs {n_sent}")

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

    # GATE 12: ML AGREEMENT — ensemble disagrees with decision = cap 60%
    if decision.get('decision') != 'NO_TRADE' and ml_result.get('available'):
        ml_says_up = ml_result.get('score', 0.5) > 0.55
        ml_says_down = ml_result.get('score', 0.5) < 0.45
        if (decision['decision'] == 'BUY' and ml_says_down) or (decision['decision'] == 'SELL' and ml_says_up):
            if decision.get('confidence', 0) > 60:
                slog(f"⚠ ML disagreement: ML score={ml_result['score']:.2f} vs {decision['decision']} — capped 60%")
                decision['confidence'] = 60

    # GATE 13: CONFIDENCE FLOOR — <45% = NO_TRADE
    if decision.get('decision') != 'NO_TRADE' and decision.get('confidence', 0) < 45:
        decision['_original_decision'] = decision.get('decision')
        decision['decision'] = 'NO_TRADE'
        gate_reason = f"Confidence floor: {decision.get('confidence', 0)}% < 45%"
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


@app.post("/history/save")
async def save_history(req: SavePredRequest):
    await save_prediction(req.prediction)
    return {"ok": True}


@app.post("/ml/retrain")
async def ml_retrain():
    """Retrain ML ensemble on rated prediction history."""
    try:
        preds = await get_predictions(limit=2000)
        model = train_ensemble(preds)
        if model:
            _ml_cache['all'] = model
            return {"ok": True, "samples": model['n_samples'], "accuracy": model.get('train_accuracy', 0)}
        return {"ok": False, "reason": "Not enough rated predictions"}
    except Exception as e:
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
            moved = price - entry_price
            orig = p.get('original_decision')
            decision = p.get('decision', 'NO_TRADE')

            if orig:
                feedback = 'correct' if (orig == 'BUY' and moved < 0) or (orig == 'SELL' and moved > 0) else 'wrong'
            elif decision == 'BUY':
                feedback = 'correct' if moved > 0 else 'wrong'
            elif decision == 'SELL':
                feedback = 'correct' if moved < 0 else 'wrong'
            else:
                feedback = 'skipped'

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

        # Score based on original_decision (for gated NO_TRADE) or the decision itself
        orig = req.original_decision
        feedback = None
        if orig == 'BUY':
            feedback = 'correct' if moved < 0 else 'wrong'  # avoided a bad BUY
        elif orig == 'SELL':
            feedback = 'correct' if moved > 0 else 'wrong'  # avoided a bad SELL
        else:
            feedback = 'skipped'  # pure abstention

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
        raise HTTPException(400, str(e))


@app.get("/candles")
async def get_candles(asset: str, interval: str = '1h', limit: int = 100):
    candles = await fetch_candles(asset, interval, limit)
    return {"candles": candles, "asset": asset}


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
