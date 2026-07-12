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
from agents.quant_agent import run_quant_agent, build_quant_prompt, build_quant_prompt_v3
from agents.news_agent import fetch_asset_news, filter_headlines_ai, run_news_agent
from agents.decision_agent import run_decision_agent
from sentiment import get_sentiment_snapshot
from macro_engine import get_macro_context
from correlation_engine import get_correlation_summary
from cluster_engine import assign_cluster
from alert_engine import scan_for_alerts, get_latest_alerts
from telegram_bot import send_prediction, send_scanner_summary
from trading_engine import get_engine as get_trading_engine, TradingEngine
from portfolio import scan_portfolio
from calibration import calibrate_confidence
from equity_tracker import EquityTracker
from ws_manager import ws_manager
from smc_engine import detect_smc
from orderbook import get_orderbook_imbalance
from whale_monitor import get_whale_activity
from options_flow import get_options_sentiment
from smart_money import analyze_smart_money
import forensic_log
# V5: New intelligence modules
from funding_oi import get_funding_oi_combined
from liquidations import get_liquidation_intel
from volume_profile import compute_volume_profile, volume_profile_signal
from order_flow import get_order_flow
from walkforward import get_walkforward_tester
from feature_pruner import get_feature_pruner
from model_retrainer import get_retrainer
from regime_strategies import apply_regime_adjustments, regime_confidence_adjustment, get_regime_from_hmm
from pre_event import get_pre_event_adjustments, should_skip_trade
from disagreement_signal import compute_disagreement, apply_disagreement
from rl_lite import get_rl_lite
from tick_engine import get_tick_engine, start_tick_stream
from execution_optimizer import get_execution_optimizer

app = FastAPI(title="ULTRAMAX Backend", version="4.0")

# Singletons
_equity_tracker = EquityTracker()

# A6: Session Intelligence — track recent predictions per asset
_session_predictions: dict = {}  # {asset: [{'ts': int, 'direction': str, 'confidence': int}]}

# ─── Autonomous Trader State ────────────────────────────────────────────────
_autotrader = {
    'enabled': False,
    'assets': [],
    'interval_minutes': 60,
    'last_cycle': 0,
    'total_cycles': 0,
    'trades_opened': 0,
    'trades_closed': 0,
    'status': 'stopped',
    'cycle_log': [],
    'trade_size': 0,
    'starting_equity': 10000,
    'force_trade': True,
    'force_size_scale': 0.5,
    '_loop_running': False,
    'use_local': False,
    'local_url': 'http://localhost:11434',
    'local_model': 'qwen2.5:7b',
    'hard_stop_pct': 5.0,  # emergency close if loss exceeds this %
    'pipeline_version': 3,   # 3 = ungagged-judge pipeline, 2 = classic (pre-2026-07)
    'compare_mode': False,   # run BOTH pipelines each cycle; trade active one, score the other as shadow
    'time_decay_hours': 7.0, # force-close losers held longer than this
}

# Shadow book: in COMPARE mode the inactive brain trades its OWN paper ledger,
# viewable on its own dashboard (bot.html?brain=shadow). Exits are evaluated at
# cycle boundaries only (no intra-cycle trailing).
_shadow_engine_inst: Optional[TradingEngine] = None

def get_shadow_engine() -> TradingEngine:
    global _shadow_engine_inst
    if _shadow_engine_inst is None:
        _shadow_engine_inst = TradingEngine()
    return _shadow_engine_inst


def _is_ai_dead(result: dict) -> bool:
    """True when every LLM agent failed (bad/missing API keys, no credit, outage):
    the judge errored AND the quant analyst produced no direction. In that state
    any trade would be a blind indicator coin flip — refuse it, loudly."""
    quant_dir = (result.get('quant', {}).get('direction') or '').upper()
    return result.get('agent_model') == 'error' and quant_dir not in ('BUY', 'SELL')


async def _manage_shadow_book(asset: str, direction: str, price: float):
    """Cash-out cycle rules applied to the shadow brain's own paper ledger:
    profit → close + reopen fresh direction; small loss → hold; hard stop or
    time decay → close. Exits evaluated only at cycle boundaries."""
    sh = get_shadow_engine()
    hard_stop_pct = _autotrader.get('hard_stop_pct', 5.0)
    max_hold = int(float(_autotrader.get('time_decay_hours', 7.0)) * 3600)
    size = _autotrader.get('trade_size') or min(10000.0, sh._equity * 0.05)
    open_pos = [p for p in sh.positions.values() if p.status == 'open' and p.asset == asset]
    if open_pos:
        pos = open_pos[0]
        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
        if pos.direction == 'SELL':
            pnl_pct = -pnl_pct
        age = int(time.time()) - pos.entry_time
        if pnl_pct > 0:
            await sh.close_position(pos.id, price, 'shadow_cashout')
        elif pnl_pct <= -hard_stop_pct:
            await sh.close_position(pos.id, price, 'shadow_hard_stop')
        elif age > max_hold:
            await sh.close_position(pos.id, price, 'shadow_time_decay')
        else:
            return  # holding the small loser — no reopen this cycle
    await sh.open_position(asset, direction, price, size,
                           stop_loss_pct=2.0, take_profit_pct=6.0, trailing=0.8)


# V3 vs V2 comparison scoreboard — each side tracks its own calls, resolved
# against actual price movement one cycle later.
_version_scoreboard = {
    'v3': {'decisions': 0, 'resolved': 0, 'wins': 0, 'losses': 0, 'sum_ret': 0.0, 'open': {}},
    'v2': {'decisions': 0, 'resolved': 0, 'wins': 0, 'losses': 0, 'sum_ret': 0.0, 'open': {}},
}


def _scoreboard_record(vkey: str, asset: str, direction: str, confidence: float, price: float):
    """Resolve this asset's previous call for `vkey` against the current price,
    then store the new call. Direction-correctness at next-cycle price."""
    sb = _version_scoreboard.get(vkey)
    if sb is None or not price or direction not in ('BUY', 'SELL'):
        return
    prev = sb['open'].get(asset)
    if prev and prev.get('price'):
        ret = (price - prev['price']) / prev['price'] * 100
        if prev['direction'] == 'SELL':
            ret = -ret
        sb['resolved'] += 1
        sb['sum_ret'] += ret
        if ret > 0:
            sb['wins'] += 1
        elif ret < 0:
            sb['losses'] += 1
    sb['decisions'] += 1
    sb['open'][asset] = {'direction': direction, 'price': price, 'ts': int(time.time()), 'confidence': confidence}


def _bot_direction_from_result(result: dict) -> tuple:
    """Resolve the tradeable direction+confidence from a prediction result,
    mirroring the bot's always-trade chain: decision → original → quant → indicators."""
    direction = result.get('decision', 'NO_TRADE')
    confidence = result.get('confidence', 0)
    if direction == 'NO_TRADE':
        orig = result.get('original_decision')
        quant_dir = (result.get('quant', {}).get('direction') or '').upper()
        if orig and orig in ('BUY', 'SELL'):
            direction = orig
            confidence = max(50, min(60, confidence))
        elif quant_dir in ('BUY', 'SELL'):
            direction = quant_dir
            confidence = max(50, min(58, result.get('quant', {}).get('confidence', 55) or 55))
        else:
            ind_data = result.get('ind', {})
            score_buy = sum([
                1 if ind_data.get('macd_hist', 0) > 0 else -1,
                1 if ind_data.get('dist_e20', 0) > 0 else -1,
                1 if ind_data.get('trend_slope', 0) > 0 else -1,
                1 if ind_data.get('supertrend_bull', False) else -1,
                1 if ind_data.get('kalman_trend', 0) > 0 else -1,
            ])
            direction = 'BUY' if score_buy > 0 else 'SELL'
            confidence = 52
    return direction, confidence

# Self-correction: per-asset accuracy tracker
_asset_accuracy: dict[str, dict] = {}

def _update_asset_accuracy(asset: str, was_correct: bool, pnl_pct: float):
    if asset not in _asset_accuracy:
        _asset_accuracy[asset] = {'wins': 0, 'losses': 0, 'total': 0, 'streak': 0, 'pnl_sum': 0.0}
    a = _asset_accuracy[asset]
    a['total'] += 1
    a['pnl_sum'] += pnl_pct
    if was_correct:
        a['wins'] += 1
        a['streak'] = max(0, a['streak']) + 1
    else:
        a['losses'] += 1
        a['streak'] = min(0, a['streak']) - 1

def _get_asset_size_factor(asset: str) -> float:
    """Self-correction: reduce size for assets that keep losing, increase for winners."""
    a = _asset_accuracy.get(asset)
    if not a or a['total'] < 3:
        return 1.0
    win_rate = a['wins'] / a['total']
    if win_rate >= 0.6:
        return 1.2
    if win_rate >= 0.45:
        return 1.0
    if win_rate >= 0.3:
        return 0.6
    return 0.35

# ─── Feature Registry ──────────────────────────────────────────────────────
# Single source of truth for toggleable features. Frontend fetches this list
# via GET /api/features, persists user choices in localStorage, and sends them
# back with every /predict call. Server applies exactly what was sent, echoes
# back which features actually ran. No drift possible.
FEATURES_REGISTRY = {
    "trade_history": {
        "default": True,
        "label": "Trade history in AI prompt",
        "description": "Inject recent W/L track record per asset into the Decision agent prompt so it can see its own performance. Risk: confirmation bias on streaks — toggle off to A/B test.",
    },
    "volume_confirm": {
        "default": True,
        "label": "Volume confirmation gate",
        "description": "Penalise confidence when OBV slope or volume z-score disagrees with the trade direction. Risk: hoax-driven volume can look like real conviction — toggle off if news-driven moves are being filtered out.",
    },
}

def feature_enabled(req_features, name: str) -> bool:
    """Returns True if the feature is enabled for this request.
    Falls back to the registry default if the request didn't send a flag
    (forward-compatible with old clients)."""
    if req_features is None:
        return FEATURES_REGISTRY.get(name, {}).get("default", True)
    if name in req_features:
        return bool(req_features[name])
    return FEATURES_REGISTRY.get(name, {}).get("default", True)


# ─── Learning Memory ────────────────────────────────────────────────────────
from typing import Dict
_trade_lessons: Dict[str, list] = {}

def record_lesson(asset: str, direction: str, entry: float, exit_price: float,
                   pnl: float, reason: str):
    if asset not in _trade_lessons:
        _trade_lessons[asset] = []
    was_correct = pnl > 0
    pnl_pct = (exit_price - entry) / entry * 100 if direction == 'BUY' else (entry - exit_price) / entry * 100
    if was_correct:
        lesson = f"{direction} was correct ({pnl_pct:+.2f}%). Closed via {reason}."
    else:
        lesson = f"{direction} was WRONG ({pnl_pct:+.2f}%). Lost ${abs(pnl):.2f}."
    _trade_lessons[asset].append({
        'ts': int(time.time()),
        'direction': direction,
        'entry': entry,
        'exit': exit_price,
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl_pct, 2),
        'reason': reason,
        'was_correct': was_correct,
        'lesson': lesson,
    })
    _trade_lessons[asset] = _trade_lessons[asset][-20:]
    _update_asset_accuracy(asset, was_correct, pnl_pct)

    # V5: RL-Lite — update trust scores from trade outcome
    try:
        rl = get_rl_lite()
        trade_signals = {
            'funding_oi': direction == 'BUY',
            'order_flow': direction == 'BUY',
            'volume_profile': direction == 'BUY',
            'tick_structure': direction == 'BUY',
            'model_disagreement': False,
        }
        rl.record_outcome(trade_signals, was_correct, pnl_pct)
    except Exception:
        pass

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
    asyncio.create_task(warm_ml_models())
    # V5: Start tick stream for crypto assets
    asyncio.create_task(start_tick_stream(['BTC', 'ETH', 'SOL']))
    # V5: Schedule periodic model retraining check
    asyncio.create_task(_periodic_retrain_check())
    asyncio.create_task(_safe_refresh_calendar())
    asyncio.create_task(_continuous_scanner())
    asyncio.create_task(_trading_position_monitor())
    asyncio.create_task(ws_manager.price_feed(fetch_current_price, ALL_ASSETS[:6]))


# ─── Models ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    asset: str
    horizon: int
    api_key: str
    ds_key: Optional[str] = None
    worker_url: Optional[str] = None
    use_r1: Optional[bool] = True
    bot_mode: Optional[bool] = False
    use_local: Optional[bool] = False
    local_url: Optional[str] = "http://localhost:11434"
    local_model: Optional[str] = "qwen2.5:7b"
    features: Optional[dict] = None
    pipeline_version: Optional[int] = 2  # 2 = classic, 3 = ungagged-judge pipeline
    shadow: Optional[bool] = False       # compare-mode shadow run: no telegram, no session memory

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
    """Score a prediction's accuracy based on direction + target proximity.
    For NO_TRADE with original_decision: evaluates the AI's original prediction.
    For NO_TRADE with predicted_price: checks if direction was right.
    """
    if not entry_price:
        return 'skipped'
    moved = outcome_price - entry_price
    moved_pct = moved / entry_price * 100

    if decision == 'BUY':
        return 'correct' if moved > 0 else 'wrong'
    if decision == 'SELL':
        return 'correct' if moved < 0 else 'wrong'

    # NO_TRADE — evaluate based on AI's original intent + target accuracy
    direction = original_decision
    target = predicted_price

    if not direction and not target:
        return 'skipped'

    if direction and direction != 'NO_TRADE':
        if direction == 'BUY':
            return 'correct' if moved_pct > 0.3 else 'wrong'
        elif direction == 'SELL':
            return 'correct' if moved_pct < -0.3 else 'wrong'

    if target:
        predicted_move_pct = (target - entry_price) / entry_price * 100
        if abs(predicted_move_pct) < 0.01:
            return 'skipped'
        direction_correct = (predicted_move_pct > 0 and moved_pct > 0) or \
                            (predicted_move_pct < 0 and moved_pct < 0)
        return 'correct' if direction_correct else 'wrong'

    return 'skipped'


def _postmortem_analysis(req, feedback: str, moved_pct: float, outcome_price: float) -> str:
    """A10: Simple post-mortem — compare decision vs actual move and note what went wrong/right."""
    decision = req.decision or 'NO_TRADE'
    orig = req.original_decision or decision
    parts = []

    if feedback == 'correct':
        parts.append(f"CORRECT — {orig} was right, price moved {moved_pct:+.2f}%.")
    elif feedback == 'wrong':
        parts.append(f"WRONG — {orig} predicted but price moved {moved_pct:+.2f}% (opposite).")
    else:
        parts.append(f"SKIPPED — no clear direction to evaluate.")

    # Compare predicted target vs actual
    pred_price = req.predicted_price or req.target_price
    if pred_price and req.entry_price:
        expected_pct = (pred_price - req.entry_price) / req.entry_price * 100
        parts.append(f"Expected {expected_pct:+.2f}%, got {moved_pct:+.2f}%.")
        if abs(expected_pct) > 0.01:
            accuracy_ratio = moved_pct / expected_pct if expected_pct != 0 else 0
            if accuracy_ratio > 0.8:
                parts.append("Target accuracy: GOOD (>80% of predicted move).")
            elif accuracy_ratio > 0.3:
                parts.append("Target accuracy: PARTIAL (direction right, magnitude off).")
            else:
                parts.append("Target accuracy: POOR.")

    # Gate override analysis
    if decision == 'NO_TRADE' and orig and orig != 'NO_TRADE':
        if feedback == 'correct':
            parts.append(f"Gate overrode {orig} to NO_TRADE — original call was correct, gate was too cautious.")
        elif feedback == 'wrong':
            parts.append(f"Gate overrode {orig} to NO_TRADE — gate correctly prevented a bad trade.")

    return ' '.join(parts) if parts else ''


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


def compute_pqs(quant_result: dict, news_result: dict, ml_result: dict,
                 ind: dict, decision: dict, mtf_data: dict = None,
                 funding_oi: dict = None, order_flow: dict = None,
                 vp_signal: dict = None, tick_data: dict = None) -> dict:
    """Prediction Quality Score 0-10 — measures signal confluence.
    Redesigned: uses indicators available for ALL assets (stocks, crypto, futures)
    so PQS is never stuck at 0-2 just because V5 crypto modules return unavailable."""
    score = 0
    reasons = []
    d_dir = decision.get('decision', '')

    # 1. Agent agreement (0-2) — quant, news, ML all agree on direction
    q_dir = quant_result.get('direction', '').upper()
    n_sent = news_result.get('sentiment', '').upper()
    ml_dir = None
    if ml_result and ml_result.get('available'):
        ms = ml_result.get('score', 50)
        ml_dir = 'BUY' if ms > 55 else 'SELL' if ms < 45 else None
    agree = sum([
        q_dir == d_dir,
        (n_sent == 'BULLISH' and d_dir == 'BUY') or (n_sent == 'BEARISH' and d_dir == 'SELL'),
        ml_dir == d_dir if ml_dir else False,
    ])
    if agree >= 3:
        score += 2; reasons.append('All agents agree')
    elif agree >= 2:
        score += 1; reasons.append('2/3 agents agree')

    # 2. Trend alignment — MACD, EMA20, EMA50 all agree with direction (0-1)
    macd_agrees = (d_dir == 'BUY' and ind.get('macd_hist', 0) > 0) or \
                  (d_dir == 'SELL' and ind.get('macd_hist', 0) < 0)
    ema20_agrees = (d_dir == 'BUY' and ind.get('dist_e20', 0) > 0) or \
                   (d_dir == 'SELL' and ind.get('dist_e20', 0) < 0)
    ema50_agrees = (d_dir == 'BUY' and ind.get('dist_e50', 0) > 0) or \
                   (d_dir == 'SELL' and ind.get('dist_e50', 0) < 0)
    trend_count = sum([macd_agrees, ema20_agrees, ema50_agrees])
    if trend_count >= 2:
        score += 1; reasons.append(f'Trend aligned ({trend_count}/3)')

    # 3. RSI confirms (not overbought for BUY, not oversold for SELL) (0-1)
    rsi = ind.get('rsi14', 50)
    rsi_ok = (d_dir == 'BUY' and 30 < rsi < 68) or (d_dir == 'SELL' and 32 < rsi < 70)
    if rsi_ok:
        score += 1; reasons.append(f'RSI confirms ({rsi:.0f})')

    # 4. Momentum score agrees with direction (0-1)
    mom = ind.get('momentum_score', 0)
    if (d_dir == 'BUY' and mom > 0) or (d_dir == 'SELL' and mom < 0):
        score += 1; reasons.append(f'Momentum aligned ({mom:+.1f})')

    # 5. Supertrend + Ichimoku agree (0-1)
    st_agrees = (d_dir == 'BUY' and ind.get('supertrend_bull', False)) or \
                (d_dir == 'SELL' and not ind.get('supertrend_bull', True))
    ich_agrees = (d_dir == 'BUY' and ind.get('ich_bull', False)) or \
                 (d_dir == 'SELL' and ind.get('ich_bear', False))
    if st_agrees or ich_agrees:
        score += 1; reasons.append('Supertrend/Ichimoku agrees')

    # 6. Volume above average (0-1)
    if ind.get('vol_percentile', 50) > 45:
        score += 1; reasons.append('Volume active')

    # 7. Regime clarity (0-1) — HMM has a clear dominant state
    hmm = ind.get('hmm_probs', {})
    max_prob = max(hmm.values()) if hmm else 0
    if max_prob > 0.45:
        score += 1; reasons.append(f'Clear regime ({max_prob:.0%})')

    # 8. Daily timeframe alignment (0-1)
    if mtf_data:
        if (mtf_data.get('daily_bull') and d_dir == 'BUY') or \
           (mtf_data.get('daily_bear') and d_dir == 'SELL'):
            score += 1; reasons.append('Daily aligned')

    # 9. V5 bonus: if crypto modules available and confirm, add a point (0-1)
    v5_confirms = 0
    if funding_oi and funding_oi.get('available'):
        foi_bias = funding_oi.get('bias', 0)
        if (d_dir == 'BUY' and foi_bias >= 2) or (d_dir == 'SELL' and foi_bias <= -2):
            v5_confirms += 1
    if order_flow and order_flow.get('available'):
        of_bias = order_flow.get('bias', 0)
        if (d_dir == 'BUY' and of_bias >= 2) or (d_dir == 'SELL' and of_bias <= -2):
            v5_confirms += 1
    if tick_data and tick_data.get('available'):
        tick_bias = tick_data.get('bias', 0)
        if (d_dir == 'BUY' and tick_bias >= 2) or (d_dir == 'SELL' and tick_bias <= -2):
            v5_confirms += 1
    if v5_confirms >= 1:
        score += 1; reasons.append(f'V5 confirms ({v5_confirms} sources)')

    # 10. Kalman trend agrees (0-1)
    kalman = ind.get('kalman_trend', 0)
    if (d_dir == 'BUY' and kalman > 0) or (d_dir == 'SELL' and kalman < 0):
        score += 1; reasons.append('Kalman trend agrees')

    return {'score': min(10, score), 'reasons': reasons, 'max': 10}


# ─── Adaptive Gate Thresholds (A5) ────────────────────────────────────────
_gate_stats = {'trades': 0, 'wins': 0}


def get_adaptive_floor():
    """Confidence floor adapts — with capped gates, floor can be lower since
    confidence values are more meaningful (not death-by-a-thousand-cuts)."""
    if _gate_stats['trades'] < 20:
        return 48
    win_rate = _gate_stats['wins'] / _gate_stats['trades']
    if win_rate > 0.60:
        return 45
    if win_rate > 0.50:
        return 48
    return 52


async def _continuous_scanner():
    """Background scanner — runs every 60s, sends Telegram alerts."""
    await asyncio.sleep(30)
    while True:
        try:
            alerts = await scan_for_alerts()
            if alerts:
                print(f"Scanner: {len(alerts)} high-confluence alert(s)")
                asyncio.create_task(send_scanner_summary(alerts))
        except Exception as e:
            print(f"Scanner error: {e}")
        await asyncio.sleep(60)


async def _autotrader_loop():
    """The autonomous trading brain. Wakes up every interval, runs predictions, opens trades."""
    if _autotrader['_loop_running']:
        print("AUTOTRADER: Loop already running, refusing duplicate spawn")
        return
    _autotrader['_loop_running'] = True
    from config import OPENAI_API_KEY, DEEPSEEK_API_KEY
    from telegram_bot import send_message

    await asyncio.sleep(10)
    engine = get_trading_engine()

    while True:
        if not _autotrader['enabled'] or not _autotrader['assets']:
            _autotrader['_loop_running'] = False
            return

        interval = max(10, _autotrader['interval_minutes']) * 60
        _autotrader['status'] = 'running'
        _autotrader['last_cycle'] = int(time.time())
        _autotrader['total_cycles'] += 1
        cycle = _autotrader['total_cycles']

        print(f"\n{'='*60}")
        print(f"AUTOTRADER CYCLE #{cycle} — {len(_autotrader['assets'])} assets")
        print(f"{'='*60}")

        cycle_id = forensic_log.log_cycle_start(
            cycle, _autotrader['assets'], _autotrader['interval_minutes'],
            engine._equity, _autotrader.get('trade_size', 0), engine.paper_mode,
        )

        cycle_summary = []

        for asset_name in _autotrader['assets']:
            try:
                # Run the full prediction — ALWAYS, even if holding
                api_key = OPENAI_API_KEY or ''
                ds_key = DEEPSEEK_API_KEY or ''

                if not api_key and not ds_key:
                    print(f"  {asset_name}: SKIP — no API keys")
                    continue

                interval_min = _autotrader.get('interval_minutes', 30)
                bot_horizon = max(1, round(interval_min * 1.5 / 60))
                pipe_version = int(_autotrader.get('pipeline_version', 3) or 3)
                req = PredictRequest(
                    asset=asset_name, horizon=bot_horizon,
                    api_key=api_key, ds_key=ds_key,
                    # V3 thinks with V4 (fast). Classic V2 keeps R1 for authentic comparison.
                    use_r1=(pipe_version < 3) and not _autotrader.get('use_local', False),
                    bot_mode=True,
                    use_local=_autotrader.get('use_local', False),
                    local_url=_autotrader.get('local_url', 'http://localhost:11434'),
                    local_model=_autotrader.get('local_model', 'qwen2.5:7b'),
                    features=_autotrader.get('features'),
                    pipeline_version=pipe_version,
                )

                start_time = time.time()
                logs = []
                def slog(msg):
                    logs.append({"ts": int((time.time() - start_time) * 1000), "msg": msg})

                if _autotrader.get('compare_mode'):
                    # COMPARE MODE: run both brains — trade the active one, shadow-score the other
                    other_v = 2 if pipe_version >= 3 else 3
                    shadow_req = req.copy(update={
                        'pipeline_version': other_v,
                        'use_r1': (other_v < 3) and not _autotrader.get('use_local', False),
                        'shadow': True,
                    })
                    shadow_logs = []
                    def shadow_slog(msg):
                        shadow_logs.append({"ts": int((time.time() - start_time) * 1000), "msg": msg})
                    result, shadow_result = await asyncio.gather(
                        _run_prediction(req, WORKER_URL, start_time, logs, slog),
                        _run_prediction(shadow_req, WORKER_URL, start_time, shadow_logs, shadow_slog),
                        return_exceptions=True,
                    )
                    if isinstance(result, Exception):
                        raise result
                    if not isinstance(shadow_result, Exception):
                        if _is_ai_dead(shadow_result):
                            forensic_log.log_error(asset_name, 'shadow_ai_offline',
                                                   f"V{other_v} shadow: all LLM calls failed — not scored, not traded")
                        else:
                            s_dir, s_conf = _bot_direction_from_result(shadow_result)
                            s_price = shadow_result.get('ind', {}).get('cur', 0)
                            _scoreboard_record(f'v{other_v}', asset_name, s_dir, s_conf, s_price)
                            forensic_log.log_event_simple(
                                'shadow_decision', asset_name,
                                f"V{other_v} (shadow) says {s_dir} {s_conf}% — trading its own shadow book",
                                direction=s_dir, confidence=s_conf, version=other_v, price=s_price,
                            )
                            if s_price and s_dir in ('BUY', 'SELL'):
                                try:
                                    await _manage_shadow_book(asset_name, s_dir, s_price)
                                except Exception as she:
                                    forensic_log.log_error(asset_name, 'shadow_book', str(she)[:150])
                else:
                    result = await _run_prediction(req, WORKER_URL, start_time, logs, slog)

                # ── AI-DEAD GUARD: if every LLM call failed (bad keys / no credit),
                # refuse to trade a blind coin flip. SL/TP still protect open positions.
                if _is_ai_dead(result):
                    forensic_log.log_error(asset_name, 'ai_offline',
                        "ALL LLM calls failed — check API keys / credit in Railway Variables. "
                        "Refusing to trade blind. "
                        f"Judge said: {(result.get('insight') or '')[:150]}")
                    cycle_summary.append(f"{asset_name}: ⚠ AI OFFLINE — no trade (fix API keys)")
                    print(f"  {asset_name}: ⚠ AI OFFLINE — skipping (all LLM calls failing)")
                    continue

                direction = result.get('decision', 'NO_TRADE')
                confidence = result.get('confidence', 0)
                pqs_score = result.get('pqs', {}).get('score', 0)
                price = result.get('ind', {}).get('cur', 0)
                stop_loss = result.get('quant', {}).get('stop_loss_pct', 2.0) or 2.0

                print(f"  {asset_name}: {direction} {confidence}% PQS:{pqs_score} @ {price}")

                # ── ADX Sizing Factor — weak/no trend = smaller position, never skip
                adx = result.get('ind', {}).get('adx', 0)
                if adx >= 25:
                    adx_scale = 1.0
                elif adx >= 15:
                    adx_scale = 0.6
                    print(f"  {asset_name}: ADX {adx:.1f} weak trend — 60% size")
                else:
                    adx_scale = 0.4
                    print(f"  {asset_name}: ADX {adx:.1f} very weak — 40% size")

                # ── PQS Sizing Factor — low quality = smaller position, never skip
                if pqs_score >= 7:
                    pqs_scale = 1.0
                elif pqs_score >= 5:
                    pqs_scale = 0.75
                elif pqs_score >= 3:
                    pqs_scale = 0.5
                else:
                    pqs_scale = 0.35
                    print(f"  {asset_name}: PQS {pqs_score}/10 low — 35% size")

                # ── ALWAYS TRADE: If AI says NO_TRADE, use AI's original direction
                if direction == 'NO_TRADE':
                    orig = result.get('original_decision')
                    quant_dir = result.get('quant', {}).get('direction', '').upper()
                    # Priority: 1) AI's original direction (pre-gate), 2) Quant agent direction
                    if orig and orig in ('BUY', 'SELL'):
                        direction = orig
                        confidence = max(50, min(60, confidence))
                        print(f"  {asset_name}: RESTORED AI direction {direction} (was gated to NO_TRADE)")
                    elif quant_dir in ('BUY', 'SELL'):
                        direction = quant_dir
                        confidence = max(50, min(58, result.get('quant', {}).get('confidence', 55)))
                        print(f"  {asset_name}: USING quant direction {direction}")
                    else:
                        # Last resort: simple indicator consensus
                        ind_data = result.get('ind', {})
                        score_buy = sum([
                            1 if ind_data.get('macd_hist', 0) > 0 else -1,
                            1 if ind_data.get('dist_e20', 0) > 0 else -1,
                            1 if ind_data.get('trend_slope', 0) > 0 else -1,
                            1 if ind_data.get('supertrend_bull', False) else -1,
                            1 if ind_data.get('kalman_trend', 0) > 0 else -1,
                        ])
                        direction = 'BUY' if score_buy > 0 else 'SELL'
                        confidence = 52
                        print(f"  {asset_name}: DERIVED {direction} (indicator score={score_buy}/5)")

                if not price or price <= 0:
                    cycle_summary.append(f"{asset_name}: no price data")
                    continue

                # Record the active version's call on the comparison scoreboard
                _scoreboard_record(f'v{pipe_version}', asset_name, direction, confidence, price)

                # ── Cash-out Cycles: profitable → cash out + reopen; loss → hold
                open_for_asset = [p for p in engine.positions.values()
                                  if p.status == 'open' and p.asset == asset_name]

                if open_for_asset:
                    current_pos = open_for_asset[0]
                    if current_pos.direction == 'BUY':
                        pnl_pct = (price - current_pos.entry_price) / current_pos.entry_price * 100
                    else:
                        pnl_pct = (current_pos.entry_price - price) / current_pos.entry_price * 100

                    hard_stop_pct = _autotrader.get('hard_stop_pct', 5.0)

                    if pnl_pct > 0:
                        # Profitable → cash out, lock in gains, reopen fresh position below
                        print(f"  {asset_name}: CASHOUT {current_pos.direction} +{pnl_pct:.2f}% profit, reopening {direction}")
                        close_result = await engine.close_position(current_pos.id, price, 'cycle_cashout')
                        cashed_pnl = close_result.get('pnl', 0)
                        _autotrader['trades_closed'] += 1
                        record_lesson(asset_name, current_pos.direction, current_pos.entry_price, price, cashed_pnl, 'cycle_cashout')
                        forensic_log.log_trade_flip(
                            asset_name, current_pos.direction, direction,
                            current_pos.entry_price, price, cashed_pnl, 0,
                            parent_id=cycle_id,
                        )
                        asyncio.create_task(send_message(
                            f"💰 CASHOUT {asset_name}: {current_pos.direction} +{pnl_pct:.2f}% | +${cashed_pnl:.2f} | Reopening {direction}"
                        ))
                        cycle_summary.append(f"{asset_name}: CASHOUT +{pnl_pct:.1f}% → {direction}")

                    elif pnl_pct > -hard_stop_pct:
                        # Time-decay: force-close losers held longer than the configured limit (default 7h)
                        age_seconds = int(time.time()) - current_pos.entry_time
                        max_hold_seconds = int(float(_autotrader.get('time_decay_hours', 7.0)) * 3600)
                        if pnl_pct < 0 and age_seconds > max_hold_seconds:
                            age_hrs = round(age_seconds / 3600, 1)
                            print(f"  {asset_name}: TIME DECAY {current_pos.direction} {pnl_pct:+.2f}% held {age_hrs}h > {max_hold_seconds/3600:.0f}h limit → force close")
                            close_result = await engine.close_position(current_pos.id, price, 'time_decay')
                            decay_pnl = close_result.get('pnl', 0)
                            _autotrader['trades_closed'] += 1
                            record_lesson(asset_name, current_pos.direction, current_pos.entry_price, price, decay_pnl, 'time_decay')
                            forensic_log.log_trade_flip(
                                asset_name, current_pos.direction, direction,
                                current_pos.entry_price, price, decay_pnl, 0,
                                parent_id=cycle_id,
                            )
                            asyncio.create_task(send_message(
                                f"⏰ TIME DECAY {asset_name}: {pnl_pct:+.2f}% after {age_hrs}h | ${decay_pnl:+.2f} | Reopening {direction}"
                            ))
                            cycle_summary.append(f"{asset_name}: TIME DECAY {pnl_pct:+.1f}% ({age_hrs}h) → {direction}")
                        else:
                            print(f"  {asset_name}: HOLD {current_pos.direction} {pnl_pct:+.2f}% (waiting for recovery, hard stop at -{hard_stop_pct:.1f}%)")
                            cycle_summary.append(f"{asset_name}: HOLD {current_pos.direction} ({pnl_pct:+.1f}% loss)")
                            continue

                    else:
                        # Loss exceeded hard stop → emergency close, accept the loss
                        print(f"  {asset_name}: HARD STOP {current_pos.direction} {pnl_pct:+.2f}% exceeds -{hard_stop_pct:.1f}% limit")
                        close_result = await engine.close_position(current_pos.id, price, 'hard_stop')
                        hard_pnl = close_result.get('pnl', 0)
                        _autotrader['trades_closed'] += 1
                        record_lesson(asset_name, current_pos.direction, current_pos.entry_price, price, hard_pnl, 'hard_stop')
                        forensic_log.log_trade_flip(
                            asset_name, current_pos.direction, direction,
                            current_pos.entry_price, price, hard_pnl, 0,
                            parent_id=cycle_id,
                        )
                        asyncio.create_task(send_message(
                            f"🛑 HARD STOP {asset_name}: {pnl_pct:+.2f}% loss | -${abs(hard_pnl):.2f} | Reopening {direction}"
                        ))
                        cycle_summary.append(f"{asset_name}: HARD STOP {pnl_pct:+.1f}% → {direction}")

                # ── Daily loss limit: only hard stop
                if engine.daily_pnl <= -(engine._equity * 0.05):
                    cycle_summary.append(f"{asset_name}: daily loss limit (5%)")
                    continue

                # ── R:R always 3:1 -- ATR-based stops with realistic floors
                atr = result.get('ind', {}).get('atr', 0)
                if atr and price:
                    atr_pct = (atr / price) * 100
                    sl_pct = max(1.0, min(3.0, atr_pct * 1.5))
                    tp_pct = sl_pct * 3.0
                    trail_pct = max(0.8, min(2.5, atr_pct * 1.2))
                else:
                    sl_pct = 1.5
                    tp_pct = sl_pct * 3.0
                    trail_pct = 1.2

                # ── Position sizing
                custom_size = _autotrader.get('trade_size', 0)
                if custom_size > 0:
                    size = round(min(custom_size, engine._equity * 0.25), 2)
                else:
                    base_size = max(min(engine._equity * 0.02, 50000), 500)
                    stats = _equity_tracker.get_stats()
                    if stats.get('n_trades', 0) >= 10:
                        win_rate = 0.5
                        rated = await get_predictions(asset_name, 100)
                        wins = sum(1 for p in rated if p.get('feedback') == 'correct')
                        total_rated = sum(1 for p in rated if p.get('feedback') in ('correct', 'wrong'))
                        if total_rated >= 5:
                            win_rate = wins / total_rated
                        kelly = engine.kelly_size(win_rate, 2.0, 1.0, engine._equity)
                        if kelly > 100:
                            base_size = kelly
                    if base_size < 10:
                        base_size = max(engine._equity * 0.02, 500)
                    asset_factor = _get_asset_size_factor(asset_name)
                    size = round(base_size * pqs_scale * asset_factor * adx_scale, 2)

                if pqs_score < 5:
                    sl_pct = min(sl_pct, 1.2)
                    tp_pct = sl_pct * 3.0

                print(f"  {asset_name}: size=${size:.0f} PQS={pqs_score} ADX={adx:.0f} SL={sl_pct:.1f}% TP={tp_pct:.1f}%")

                # Open the trade
                trade_result = await engine.open_position(
                    asset_name, direction, price, size,
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct,
                    trailing=trail_pct,
                )

                if trade_result.get('ok'):
                    _autotrader['trades_opened'] += 1
                    pos_data = trade_result.get('position', {})
                    # Store entry confidence for anti-whipsaw Rule 3
                    pos_id = pos_data.get('id', '')
                    if pos_id and pos_id in engine.positions:
                        engine.positions[pos_id].entry_confidence = confidence
                    forensic_log.log_trade_open(
                        asset_name, direction, price, size,
                        pos_data.get('stop_loss', 0), pos_data.get('take_profit', 0),
                        pos_data.get('trailing_stop_pct', 1.5),
                        f"autotrader_cycle_{cycle}", confidence, pqs_score,
                        parent_id=cycle_id,
                    )
                    msg = f"OPENED {direction} {asset_name} @ ${price:,.2f} size=${size:.0f} conf={confidence}% PQS={pqs_score}"
                    print(f"  >>> {msg}")
                    cycle_summary.append(f"{asset_name}: {direction} ${size:.0f}")
                    asyncio.create_task(send_message(f"🤖 AUTO-TRADE: {msg}"))
                else:
                    cycle_summary.append(f"{asset_name}: trade failed ({trade_result.get('error', '?')})")

            except Exception as e:
                print(f"  {asset_name}: ERROR — {str(e)[:100]}")
                forensic_log.log_error(asset_name, "autotrader_loop", str(e))
                cycle_summary.append(f"{asset_name}: error")

        # Save cycle log for frontend
        _autotrader['cycle_log'].append({
            'ts': int(time.time()),
            'cycle': cycle,
            'summaries': cycle_summary,
        })
        _autotrader['cycle_log'] = _autotrader['cycle_log'][-50:]

        forensic_log.log_cycle_end(
            cycle, engine._equity, cycle_summary,
            _autotrader['trades_opened'], _autotrader['trades_closed'],
            parent_id=cycle_id,
        )

        # Send cycle summary to Telegram
        hb = engine.heartbeat()
        summary = (
            f"🤖 AUTOTRADER CYCLE #{cycle}\n"
            f"Assets: {', '.join(cycle_summary)}\n"
            f"Equity: ${hb['equity']:,.2f} | Open: {hb['positions']} | Daily P&L: ${hb['daily_pnl']:+,.2f}\n"
            f"Next scan in {_autotrader['interval_minutes']}min"
        )
        asyncio.create_task(send_message(summary))

        # Should I keep going? Check equity health
        if hb['equity'] < 5000:  # lost 50% of starting capital
            asyncio.create_task(send_message(
                f"⚠️ AUTOTRADER WARNING: Equity ${hb['equity']:,.2f} is below $5,000. "
                f"Consider stopping. Daily P&L: ${hb['daily_pnl']:+,.2f}"
            ))
        elif hb['equity'] > 15000:  # up 50%
            asyncio.create_task(send_message(
                f"🎯 AUTOTRADER: Equity ${hb['equity']:,.2f} — up {((hb['equity']-10000)/10000*100):.1f}% from start. "
                f"Consider taking profits."
            ))

        _autotrader['status'] = 'sleeping'
        print(f"Autotrader sleeping {_autotrader['interval_minutes']}min until next cycle...")
        await asyncio.sleep(interval)


async def _trading_position_monitor():
    """Background task: check open trading positions every 30s + stale position timeout."""
    await asyncio.sleep(15)
    engine = get_trading_engine()
    while True:
        try:
            open_positions = [p for p in engine.positions.values() if p.status == 'open']
            if open_positions:
                actions = await engine.check_all_positions(fetch_current_price)
                for act in actions:
                    print(f"Trading: auto-closed {act.get('asset')} ({act.get('reason')}) P&L: {act.get('pnl')}")

                # V5: Stale position timeout — close positions stuck for 6+ hours not making progress
                now = int(time.time())
                for pos in open_positions:
                    hold_hours = (now - pos.entry_time) / 3600
                    if hold_hours < 6:
                        continue
                    try:
                        result = await fetch_current_price(pos.asset)
                        cur_price = result.get('price', 0)
                        if not cur_price:
                            continue
                        if pos.direction == 'BUY':
                            pnl_pct = (cur_price - pos.entry_price) / pos.entry_price * 100
                        else:
                            pnl_pct = (pos.entry_price - cur_price) / pos.entry_price * 100
                        # Close stale at 8h if losing — frees the slot
                        # Close stale at 12h regardless — slot must rotate
                        should_close = False
                        reason = ''
                        if 8 <= hold_hours < 12 and pnl_pct < -0.5:
                            should_close = True
                            reason = f"stale losing {pnl_pct:+.2f}% after {hold_hours:.1f}h"
                        elif hold_hours >= 12:
                            should_close = True
                            reason = f"hard timeout {hold_hours:.1f}h ({pnl_pct:+.2f}%)"
                        if should_close:
                            close_result = await engine.close_position(pos.id, cur_price, 'stale_timeout')
                            stale_pnl = close_result.get('pnl', 0)
                            record_lesson(pos.asset, pos.direction, pos.entry_price, cur_price, stale_pnl, 'stale_timeout')
                            print(f"Trading: stale timeout {pos.asset} — {reason} — P&L: {stale_pnl}")
                    except Exception:
                        continue
        except Exception as e:
            print(f"Position monitor error: {e}")
        await asyncio.sleep(30)


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


@app.get("/bot")
async def serve_bot():
    fp = os.path.join(frontend_path, 'bot.html')
    if os.path.exists(fp):
        return FileResponse(fp)
    raise HTTPException(404, "bot.html not found")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "5.0", "ts": int(time.time())}


@app.get("/api/features")
async def get_features():
    """Returns the feature registry so the frontend can render toggles.
    Single source of truth — frontend never hardcodes feature names."""
    return {"features": FEATURES_REGISTRY}


@app.get("/ollama/status")
async def ollama_status(url: str = "http://localhost:11434"):
    """Check if Ollama is running and list available models."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{url}/api/tags")
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                return {"ok": True, "models": models, "url": url}
            return {"ok": False, "error": f"HTTP {resp.status_code}"}
    except httpx.ConnectError:
        return {"ok": False, "error": "Ollama not running. Start it with: ollama serve"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:100]}


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


def build_risk_evidence(asset: str, ind: dict, mtf_data: dict, macro_data: dict,
                        fg_data: dict, onchain_data: dict, macro_context: dict,
                        is_crypto: bool) -> list:
    """V3: the Risk Officer's notes — direction-free FACTS handed to the judge
    BEFORE it rules. These replace the old confidence-subtracting gates."""
    ev = []
    try:
        from datetime import datetime, timedelta, timezone as _tz
        riyadh = datetime.now(_tz(timedelta(hours=3)))
        if get_asset_type(asset) == 'stock':
            if riyadh.weekday() in (5, 6):
                ev.append("US stock market is CLOSED (weekend) — any position will freeze until open")
            else:
                rh = riyadh.hour + riyadh.minute / 60.0
                if rh < 16.5 or rh >= 23.0:
                    ev.append("US regular market hours are CLOSED right now — thinner/no liquidity")
        macro = macro_data or {}
        vix = macro.get('vix', 0) or 0
        dxy = macro.get('dxy', 0) or 0
        if vix > 30:
            ev.append(f"VIX {vix:.1f} — high fear regime")
        elif vix > 0 and vix < 15:
            ev.append(f"VIX {vix:.1f} — complacent/calm regime")
        if dxy > 107:
            ev.append(f"DXY {dxy:.1f} — very strong dollar (risk-asset headwind)")
        fg_val = (fg_data or {}).get('value', 50)
        if fg_val < 25:
            ev.append(f"Fear&Greed {fg_val}/100 — extreme fear")
        elif fg_val > 75:
            ev.append(f"Fear&Greed {fg_val}/100 — extreme greed")
        if mtf_data:
            d = 'BULL' if mtf_data.get('daily_bull') else 'BEAR' if mtf_data.get('daily_bear') else 'NEUTRAL'
            h4 = 'BULL' if mtf_data.get('h4_bull') else 'BEAR' if mtf_data.get('h4_bear') else 'NEUTRAL'
            h1 = 'BULL' if mtf_data.get('h1_bull') else 'BEAR' if mtf_data.get('h1_bear') else 'NEUTRAL'
            ev.append(f"Higher timeframes: Daily {d}, 4H {h4}, 1H {h1} — trading against 2+ of these has historically failed")
        if not ind.get('ich_bull') and not ind.get('ich_bear'):
            ev.append("Price INSIDE the Ichimoku cloud — structurally indecisive zone")
        hurst = ind.get('hurst_exp', 0.5)
        entropy = ind.get('entropy_ratio', 0.5)
        if entropy > 0.87 and 0.45 <= hurst <= 0.55:
            ev.append(f"Hurst {hurst:.2f} + entropy {entropy:.2f} — market statistically indistinguishable from noise right now")
        cmf = ind.get('cmf', 0)
        if abs(cmf) > 0.15:
            ev.append(f"CMF {cmf:+.2f} — strong money {'inflow' if cmf > 0 else 'outflow'}")
        obv = ind.get('obv_slope', 0)
        if abs(obv) > 0.15:
            ev.append(f"OBV slope {'rising' if obv > 0 else 'falling'} strongly")
        if is_crypto and onchain_data:
            funding = onchain_data.get('funding_rate', 0) or 0
            if funding > 0.0008:
                ev.append(f"Funding rate {funding:.4f}% — longs are crowded (squeeze risk on BUY)")
            elif funding < -0.0008:
                ev.append(f"Funding rate {funding:.4f}% — shorts are crowded (squeeze risk on SELL)")
        upcoming = (macro_context or {}).get('upcoming_events', [])
        high_impact = [e for e in upcoming if e.get('impact') == 'high']
        if high_impact:
            names = ', '.join(str(e.get('name', 'event')) for e in high_impact[:2])
            ev.append(f"HIGH-IMPACT event ahead: {names} — volatility spike likely")
        # Correlated assets' recent calls this session
        rel_map = {'BTC': ['ETH', 'SOL'], 'ETH': ['BTC', 'SOL'], 'SOL': ['BTC', 'ETH'],
                   'AAPL': ['MSFT', 'SPY'], 'MSFT': ['AAPL', 'SPY'], 'NVDA': ['SPY', 'MSFT'],
                   'GOOGL': ['MSFT', 'SPY'], 'SPY': ['AAPL', 'MSFT', 'NVDA']}
        for rel in rel_map.get(asset, []):
            sess = _session_predictions.get(rel, [])
            if sess and (time.time() - sess[-1]['ts']) < 7200 and sess[-1]['direction'] in ('BUY', 'SELL'):
                ago = int((time.time() - sess[-1]['ts']) / 60)
                ev.append(f"Correlated asset {rel} was called {sess[-1]['direction']} {ago}min ago")
    except Exception as e:
        ev.append(f"(risk evidence partially unavailable: {str(e)[:60]})")
    return ev


async def _run_prediction(req, worker, start_time, logs, slog):
    """Internal prediction logic — separated so top-level can catch errors."""

    version = int(getattr(req, 'pipeline_version', 2) or 2)
    slog(f"🚀 PREDICT {req.asset} {req.horizon}H [pipeline V{version}]")

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
    slog("🌐 Fetching macro, on-chain, sentiment, correlation, cluster, SMC, orderbook, whales, funding, liquidations, order flow...")
    asset_type = 'crypto' if is_crypto else 'macro' if req.asset in ['GC=F','CL=F','SI=F'] else 'stock'
    (macro_data, fg_data, onchain_data, sentiment_data, macro_context, correlation_data,
     cluster_data, smc_data, orderbook_data, whale_data, options_data,
     funding_oi_data, liquidation_data, order_flow_data) = await asyncio.gather(
        fetch_macro(), fetch_fear_greed(), fetch_onchain(req.asset),
        get_sentiment_snapshot(req.asset),
        get_macro_context(req.asset, req.horizon),
        get_correlation_summary(req.asset),
        _safe_assign_cluster(req.asset, ind),
        detect_smc(candles),
        get_orderbook_imbalance(req.asset),
        get_whale_activity(req.asset),
        get_options_sentiment(req.asset),
        get_funding_oi_combined(req.asset),
        get_liquidation_intel(req.asset, ind.get('cur', 0)),
        get_order_flow(req.asset),
    )
    smart_money_data = analyze_smart_money(candles)
    # V5: Volume Profile
    vp_data = compute_volume_profile(candles)
    vp_signal = volume_profile_signal(vp_data, ind.get('cur', 0)) if vp_data.get('available') else {}
    # V5: Tick engine micro-structure
    tick_data = get_tick_engine().get_micro_structure(req.asset)

    if funding_oi_data.get('available'):
        slog(f"✓ Funding/OI: bias={funding_oi_data.get('bias',0):+d} ({funding_oi_data.get('signal')})")
    if liquidation_data.get('available'):
        slog(f"✓ Liquidations: bias={liquidation_data.get('bias',0):+d} ({liquidation_data.get('signal')})")
    if order_flow_data.get('available'):
        slog(f"✓ Order flow: bias={order_flow_data.get('bias',0):+d} ({order_flow_data.get('signal')})")
    if vp_data.get('available'):
        slog(f"✓ Volume Profile: VPOC={vp_data.get('vpoc',0):.2f} VAH={vp_data.get('vah',0):.2f} VAL={vp_data.get('val',0):.2f}")
    if tick_data.get('available'):
        slog(f"✓ Tick engine: CVD={tick_data.get('cvd_pct',0):+.1f}% ({tick_data.get('signal')})")
    if sentiment_data.get('available'):
        slog(f"✓ Sentiment: score={sentiment_data.get('composite', 0):.2f}")
    if macro_context.get('warnings'):
        slog(f"⚠ Macro: {'; '.join(macro_context['warnings'][:2])}")
    if correlation_data.get('available'):
        slog(f"✓ Correlations loaded")
    if cluster_data.get('available'):
        slog(f"✓ Cluster #{cluster_data.get('cluster_id', '?')} ({cluster_data.get('members', 0)} members)")
    if smc_data.get('available'):
        slog(f"✓ SMC: bias={smc_data.get('bias')} OBs={len(smc_data.get('order_blocks',[]))} FVGs={len(smc_data.get('fair_value_gaps',[]))}")
    if orderbook_data.get('available'):
        slog(f"✓ Orderbook: imbalance={orderbook_data.get('imbalance_ratio',0):+.3f} ({orderbook_data.get('bias')})")
    if whale_data.get('available'):
        slog(f"✓ Whale flow: net={whale_data.get('net_flow',0):,.0f} USD ({whale_data.get('bias')})")
    if options_data.get('available'):
        slog(f"✓ Options: P/C={options_data.get('put_call_ratio',0):.2f} max_pain={options_data.get('max_pain',0):.2f}")
    if smart_money_data.get('available'):
        slog(f"✓ Smart Money: SMI={smart_money_data.get('smart_money_index',0):+.4f} ({smart_money_data.get('bias')})")

    # ── Fetch news ───────────────────────────────────────────────────────
    slog("📰 Fetching news from 10+ sources...")
    articles = await fetch_asset_news(req.asset, ASSET_NAMES.get(req.asset, req.asset), asset_type)
    slog(f"✓ {len(articles)} headlines collected")

    headline_keep = 20 if version >= 3 else 8
    if len(articles) > headline_keep:
        slog("🤖 AI filtering headlines...")
        articles = await filter_headlines_ai(articles, req.asset, ASSET_NAMES.get(req.asset, req.asset),
                                              req.api_key, req.ds_key, keep=headline_keep)
    slog(f"✓ {len(articles)} headlines after AI filter")

    # V3: raw news brief for the quant analyst (cross-visibility, pre-analysis)
    news_brief = ""
    if version >= 3 and articles:
        top = articles[:5]
        avg_sent = sum(a.get('sentiment', 0) for a in articles) / len(articles)
        news_brief = (f"avg sentiment {avg_sent:+.2f} across {len(articles)} headlines; top: "
                      + " | ".join(a['headline'][:80] for a in top))

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

    # ── MTF: 1H + 4H + Daily context ──────────────────────────────────
    slog("📅 Fetching MTF trend context (1H + 4H + Daily)...")
    daily_candles, h4_candles, h1_candles = await asyncio.gather(
        fetch_candles(req.asset, '1d', 32),
        fetch_candles(req.asset, '4h', 60),
        fetch_candles(req.asset, '1h', 60),
    )
    mtf_data = {}
    if daily_candles and len(daily_candles) >= 20:
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
    if h4_candles and len(h4_candles) >= 20:
        h4c = h4_candles[:-1]
        h4_ind = compute_indicators(h4c)
        if h4_ind:
            mtf_data['h4_macd_hist'] = h4_ind['macd_hist']
            mtf_data['h4_dist_e20'] = h4_ind['dist_e20']
            mtf_data['h4_bull'] = h4_ind['dist_e20'] > 0.5 and h4_ind['macd_hist'] > 0
            mtf_data['h4_bear'] = h4_ind['dist_e20'] < -0.5 and h4_ind['macd_hist'] < 0
            mtf_data['h4_rsi'] = h4_ind['rsi14']
            slog(f"✓ 4H: {'BULL' if mtf_data['h4_bull'] else 'BEAR' if mtf_data['h4_bear'] else 'NEUTRAL'} RSI={h4_ind['rsi14']:.1f}")
    if h1_candles and len(h1_candles) >= 20:
        h1c = h1_candles[:-1]
        h1_ind = compute_indicators(h1c)
        if h1_ind:
            mtf_data['h1_macd_hist'] = h1_ind['macd_hist']
            mtf_data['h1_dist_e20'] = h1_ind['dist_e20']
            mtf_data['h1_bull'] = h1_ind['dist_e20'] > 0.3 and h1_ind['macd_hist'] > 0
            mtf_data['h1_bear'] = h1_ind['dist_e20'] < -0.3 and h1_ind['macd_hist'] < 0
            mtf_data['h1_rsi'] = h1_ind['rsi14']
            slog(f"✓ 1H: {'BULL' if mtf_data['h1_bull'] else 'BEAR' if mtf_data['h1_bear'] else 'NEUTRAL'} RSI={h1_ind['rsi14']:.1f}")

    # ── ML Ensemble ──────────────────────────────────────────────────────
    ml_result = {'confidence': 50, 'available': False}
    model_artifact = _ml_cache.get(req.asset)
    if model_artifact:
        ml_result = predict_ensemble(ind)
        slog(f"✓ ML ensemble: score={ml_result.get('score', 0):.2f} agree={ml_result.get('agreement', False)}")
    else:
        slog("⚠ ML ensemble: no trained model yet")

    # ── Agent 1: Quant (DeepSeek V4 primary, GPT-4o-mini fallback) ─────
    quant_model_label = 'DeepSeek V4' if req.ds_key else 'GPT-4o-mini'
    slog(f"📐 Agent 1 (Quant/{quant_model_label}) analyzing...")
    if version >= 3:
        quant_prompt = build_quant_prompt_v3(req.asset, ind, mc, req.horizon,
                                              cluster_data=cluster_data, correlation_data=correlation_data,
                                              bot_mode=getattr(req, 'bot_mode', False),
                                              news_brief=news_brief)
    else:
        quant_prompt = build_quant_prompt(req.asset, ind, mc, req.horizon,
                                           cluster_data=cluster_data, correlation_data=correlation_data,
                                           bot_mode=getattr(req, 'bot_mode', False))
    quant_result = await run_quant_agent(req.asset, ind, mc, req.horizon, quant_prompt, req.api_key, ds_key=req.ds_key or '')
    slog(f"✓ Quant[{quant_result.get('_quant_model','?')}]: {quant_result.get('direction')} {quant_result.get('confidence')}% — {quant_result.get('reasoning','')[:60]}")
    if not quant_result.get('_quant_model'):
        slog(f"⚠⚠ QUANT AGENT FAILED — no LLM reachable (bad key / no credit?): {quant_result.get('reasoning','')[:90]}")

    # ── Bayesian + 4-way confidence blending ────────────────────────────
    rated_preds = await get_predictions(req.asset, 200)
    raw_ai_conf = quant_result.get('confidence', 50)
    bayes_conf = bayesian_confidence(rated_preds, req.asset, req.horizon, raw_ai_conf)
    ml_conf = ml_result.get('score', 50) if ml_result['available'] else None
    cluster_conf = cluster_data.get('win_rate_4h') if cluster_data.get('available') else None

    # A3: Regime-adaptive weights — adjust blending based on market regime
    regime = ind.get('regime', 'NEUTRAL')
    hurst = ind.get('hurst_exp', 0.5)
    regime_ml_boost = 0.0
    if regime == 'TRENDING' and hurst > 0.55:
        regime_ml_boost = 0.05
    elif regime in ('RANGING', 'LOW_VOLATILITY') and hurst < 0.45:
        regime_ml_boost = 0.03
    elif regime == 'HIGH_VOLATILITY':
        regime_ml_boost = -0.05

    if version >= 3:
        # V3: the analyst's own confidence goes UNTOUCHED to the judge.
        # The numbers (ML/cluster/Bayes) join AFTER the ruling — 50% judge, 50% numbers.
        slog(f"✓ V3: quant keeps raw {raw_ai_conf}% (Bayes={bayes_conf:.0f} ML={ml_conf or 'N/A'} Cluster={cluster_conf or 'N/A'} join after the ruling)")
    else:
        if ml_conf is not None and cluster_conf is not None:
            ml_w = 0.45 + regime_ml_boost
            blended_conf = raw_ai_conf * 0.15 + bayes_conf * 0.20 + ml_conf * ml_w + cluster_conf * (0.20 - regime_ml_boost)
        elif ml_conf is not None:
            ml_w = 0.55 + regime_ml_boost
            blended_conf = raw_ai_conf * (0.20 - regime_ml_boost) + bayes_conf * 0.25 + ml_conf * ml_w
        elif cluster_conf is not None:
            blended_conf = raw_ai_conf * 0.35 + bayes_conf * 0.35 + cluster_conf * 0.30
        else:
            blended_conf = raw_ai_conf * 0.55 + bayes_conf * 0.45
        quant_result['confidence'] = round(blended_conf)
        slog(f"✓ Blended: AI={raw_ai_conf}% Bayes={bayes_conf:.0f}% ML={ml_conf or 'N/A'} Cluster={cluster_conf or 'N/A'} → {quant_result['confidence']}%")

    # ── Agent 2: News ────────────────────────────────────────────────────
    slog(f"📰 Agent 2 (News/{'DeepSeek V4' if req.ds_key else 'GPT-4o-mini'}) analyzing...")
    quant_brief = ""
    if version >= 3:
        quant_brief = (f"{quant_result.get('direction')} at {quant_result.get('confidence')}% — "
                       f"{(quant_result.get('reasoning') or '')[:160]}")
    news_result = await run_news_agent(
        req.asset, ASSET_NAMES.get(req.asset, req.asset), asset_type,
        articles, macro_data, onchain_data, fg_data, {},
        req.horizon, req.api_key, req.ds_key, db_sentiment,
        quant_brief=quant_brief, version=version,
    )
    slog(f"✓ News: {news_result.get('sentiment')} ({news_result.get('sentiment_score',0):+d}) — {news_result.get('reasoning','')[:60]}")

    # ── Agent 3: Decision (R1 / V4 / LOCAL) ────────────────────────────
    use_local = bool(req.use_local)
    use_r1 = bool(req.ds_key) and (req.use_r1 is not False) and not use_local
    model_name = 'LOCAL' if use_local else ('R1' if use_r1 else ('V4' if req.ds_key else 'GPT-4o'))
    slog(f"🧠 Agent 3 ({model_name}) making final decision...")

    # Feature: trade_history — inject recent W/L track record into AI prompt
    features_applied = []
    features_skipped = []
    recent_trades_ctx = ""
    if feature_enabled(req.features, 'trade_history'):
        lessons = _trade_lessons.get(req.asset, [])
        if lessons:
            recent = lessons[-5:]
            wins = sum(1 for l in recent if l['was_correct'])
            losses = len(recent) - wins
            lines = []
            for l in recent:
                ago = int((time.time() - l['ts']) / 60)
                status = "WIN" if l['was_correct'] else "LOSS"
                lines.append(f"  {l['direction']} {ago}min ago → {status} ({l['pnl_pct']:+.2f}%) closed via {l['reason']}")
            recent_trades_ctx = f"""
YOUR RECENT TRACK RECORD ON {req.asset} (last {len(recent)} trades):
Record: {wins}W {losses}L
{chr(10).join(lines)}
NOTE: This is factual history. Do NOT blindly repeat your last direction — evaluate the CURRENT signals independently. If your last 3 calls were all BUY but indicators now say SELL, follow the indicators."""
            slog(f"📜 [trade_history] Loaded {len(recent)} lessons for {req.asset} ({wins}W {losses}L)")
        else:
            slog(f"📜 [trade_history] No lessons yet for {req.asset}")
        features_applied.append('trade_history')
    else:
        slog("⏭ [trade_history] Skipped (disabled)")
        features_skipped.append('trade_history')

    # V3: the Risk Officer's notes — facts handed to the judge BEFORE it rules
    risk_evidence = None
    if version >= 3:
        risk_evidence = build_risk_evidence(req.asset, ind, mtf_data, macro_data,
                                            fg_data, onchain_data, macro_context,
                                            is_crypto)
        slog(f"📋 Risk Officer's notes: {len(risk_evidence)} facts for the judge")

    decision = await run_decision_agent(
        req.asset, ind, req.horizon, quant_result, news_result,
        mtf_data, mc, similar, req.ds_key or '', req.api_key, use_r1,
        ml_result=ml_result,
        use_local=use_local,
        local_url=req.local_url or "http://localhost:11434",
        local_model=req.local_model or "qwen2.5:7b",
        recent_trades_ctx=recent_trades_ctx,
        risk_evidence=risk_evidence,
        bot_mode=getattr(req, 'bot_mode', False),
        version=version,
    )
    model_used = decision.get('_model', 'unknown')
    slog(f"✓ Decision: {decision.get('decision')} {decision.get('confidence')}% [{model_used}]")
    if model_used == 'gpt-4o' and decision.get('_r1_error'):
        slog(f"⚠ R1 fallback reason: {decision['_r1_error']}")
    if model_used == 'error':
        slog(f"⚠⚠ DECISION AGENT FAILED — every LLM errored (check API keys/credit): {(decision.get('insight') or '')[:120]}")

    # ── V4 ML OVERRIDE: Only when ML is VERY confident ──────────────────
    # (V3 pipeline: the judge's ruling is FINAL — ML is a witness, never an override)
    if version < 3 and ml_result.get('available'):
        ml_score = ml_result.get('score', 50)
        ml_dir = 'BUY' if ml_score > 62 else 'SELL' if ml_score < 38 else None
        ai_dir = decision.get('decision')

        if ml_dir and ai_dir != ml_dir:
            if ml_score > 68 or ml_score < 32:
                slog(f"🔄 ML OVERRIDE: ML says {ml_dir} ({ml_score:.1f}%) but AI said {ai_dir} — forcing ML direction")
                decision['_original_decision'] = ai_dir
                decision['decision'] = ml_dir
                decision['confidence'] = max(decision.get('confidence', 50), int(ml_score if ml_dir == 'BUY' else 100 - ml_score))
            elif ai_dir == 'NO_TRADE' and ml_dir and (ml_score > 70 or ml_score < 30):
                slog(f"🔄 ML activates trade: ML says {ml_dir} ({ml_score:.1f}%) — overriding NO_TRADE (very high ML confidence)")
                decision['_original_decision'] = 'NO_TRADE'
                decision['decision'] = ml_dir
                decision['confidence'] = max(decision.get('confidence', 55), int(ml_score if ml_dir == 'BUY' else 100 - ml_score))
            elif ai_dir == 'NO_TRADE':
                slog(f"ML suggests {ml_dir} ({ml_score:.1f}%) but not strong enough to override NO_TRADE (need >70%)")

    # ── A6: Session Intelligence — track recent predictions per asset ────
    recent_session = _session_predictions.get(req.asset, [])
    if recent_session:
        last = recent_session[-1]
        if last['direction'] != 'NO_TRADE' and decision.get('decision') != 'NO_TRADE':
            if last['direction'] != decision.get('decision') and (time.time() - last['ts']) < 3600:
                slog(f"⚠ Session flip: was {last['direction']} {int(time.time()-last['ts'])}s ago, now {decision['decision']}")

    # ── A8: Cross-prediction contradiction check ─────────────────────────
    corr_assets = {'BTC': ['ETH', 'SOL'], 'ETH': ['BTC', 'SOL'], 'SOL': ['BTC', 'ETH'],
                   'AAPL': ['MSFT', 'GOOGL', 'SPY'], 'MSFT': ['AAPL', 'GOOGL', 'SPY'],
                   'GOOGL': ['AAPL', 'MSFT', 'SPY'], 'NVDA': ['SPY', 'MSFT'],
                   'SPY': ['AAPL', 'MSFT', 'NVDA', 'GOOGL']}
    related = corr_assets.get(req.asset, [])
    contradictions = 0
    for rel_asset in related:
        rel_session = _session_predictions.get(rel_asset, [])
        if rel_session and (time.time() - rel_session[-1]['ts']) < 7200:
            rel_dir = rel_session[-1]['direction']
            cur_dir = decision.get('decision')
            if (rel_dir == 'BUY' and cur_dir == 'SELL') or (rel_dir == 'SELL' and cur_dir == 'BUY'):
                contradictions += 1
    if contradictions >= 2 and decision.get('decision') != 'NO_TRADE':
        if version >= 3:
            slog(f"⚠ Cross-prediction contradiction noted: {contradictions} correlated assets disagree (V3: fact only, judge already saw correlations)")
        else:
            old_conf = decision.get('confidence', 50)
            decision['confidence'] = max(0, old_conf - 10)
            slog(f"⚠ Cross-prediction contradiction: {contradictions} correlated assets disagree — confidence {old_conf}% → {decision['confidence']}%")

    # ── CONSOLIDATED GATE SYSTEM ──────────────────────────────────────────
    # Instead of 24 sequential penalties that stack to -80%, collect all
    # signals and apply a SINGLE capped adjustment. Max total penalty: -25%.
    # This prevents confidence death-by-a-thousand-cuts.
    gate_reason = None
    gate_penalties = []   # (name, penalty_amount)
    gate_boosts = []      # (name, boost_amount)
    gate_kill = None      # if set, force NO_TRADE with this reason

    from datetime import datetime, timedelta, timezone
    riyadh_tz = timezone(timedelta(hours=3))
    now_riyadh = datetime.now(riyadh_tz)

    d_dir = decision.get('decision', '')

    # --- Collect gate signals (don't apply yet) ---

    # WEEKEND + MARKET HOURS (stocks only)
    if get_asset_type(req.asset) == 'stock':
        if now_riyadh.weekday() in (5, 6):
            gate_penalties.append(('weekend', 12))
        riyadh_hour = now_riyadh.hour + now_riyadh.minute / 60.0
        if riyadh_hour < 16.5 or riyadh_hour >= 23.0:
            gate_penalties.append(('market_closed', 8))

    # MACRO VIX+DXY extreme
    if d_dir != 'NO_TRADE':
        macro = macro_data or {}
        if macro.get('vix', 0) > 30 and macro.get('dxy', 0) > 107 and d_dir == 'BUY':
            gate_penalties.append(('macro_bear_extreme', 15))
        if macro.get('vix', 0) < 15 and macro.get('dxy', 0) < 100 and d_dir == 'SELL':
            gate_penalties.append(('macro_bull_extreme', 15))

    # MACRO BEAR/BULL CONFLUENCE — only kills trade if 5/5 signals agree
    if d_dir != 'NO_TRADE':
        fg_val = fg_data.get('value', 50)
        news_score = news_result.get('sentiment_score', 0)
        bear_signals = [ind['macd_hist'] < 0, ind['dist_e20'] < 0,
                        (macro_data.get('vix') or 0) > 18, news_score < 0, fg_val < 40]
        bull_signals = [ind['macd_hist'] > 0, ind['dist_e20'] > 0,
                        (macro_data.get('vix') or 0) < 15, news_score > 0, fg_val > 60]
        bear_count = sum(bear_signals)
        bull_count = sum(bull_signals)
        if bear_count >= 5 and d_dir == 'BUY':
            gate_kill = f"MACRO BEAR GATE — {bear_count}/5 bearish signals vs BUY"
        elif bull_count >= 5 and d_dir == 'SELL':
            gate_kill = f"MACRO BULL GATE — {bull_count}/5 bullish signals vs SELL"
        elif bear_count >= 4 and d_dir == 'BUY':
            gate_penalties.append(('macro_bear', 10))
        elif bull_count >= 4 and d_dir == 'SELL':
            gate_penalties.append(('macro_bull', 10))

    # MTF COUNTER-TREND — 1H + 4H + Daily alignment check
    if d_dir != 'NO_TRADE' and mtf_data:
        h1_opposes = (mtf_data.get('h1_bear') and d_dir == 'BUY') or \
                     (mtf_data.get('h1_bull') and d_dir == 'SELL')
        h4_opposes = (mtf_data.get('h4_bear') and d_dir == 'BUY') or \
                     (mtf_data.get('h4_bull') and d_dir == 'SELL')
        daily_opposes = (mtf_data.get('daily_bear') and d_dir == 'BUY') or \
                        (mtf_data.get('daily_bull') and d_dir == 'SELL')
        oppose_count = sum([h1_opposes, h4_opposes, daily_opposes])
        if oppose_count >= 2:
            gate_kill = f"MTF rejection — {oppose_count}/3 timeframes (1H+4H+Daily) oppose {d_dir}"
        elif daily_opposes:
            gate_penalties.append(('counter_trend_daily', 8))
        elif h4_opposes:
            gate_penalties.append(('counter_trend_4h', 5))
        elif h1_opposes:
            gate_penalties.append(('counter_trend_1h', 3))

    # ICHIMOKU CLOUD — only kill if confidence already very low
    if d_dir != 'NO_TRADE':
        inside_cloud = not ind.get('ich_bull') and not ind.get('ich_bear')
        if inside_cloud and decision.get('confidence', 0) < 52:
            gate_kill = f"Ichimoku inside cloud + very low confluence ({decision.get('confidence', 0)}%)"
        elif inside_cloud:
            gate_penalties.append(('ichimoku_cloud', 5))

    # VWAP
    if d_dir == 'BUY' and ind.get('dist_vwap', 0) < 0:
        gate_penalties.append(('vwap_below', 5))

    # HURST RANDOM WALK
    hurst = ind.get('hurst_exp', 0.5)
    if 0.45 <= hurst <= 0.55:
        gate_penalties.append(('hurst_random', 5))

    # ENTROPY
    entropy = ind.get('entropy_ratio', 0.5)
    if entropy > 0.9:
        gate_penalties.append(('entropy_noise', 5))

    # FUNDING RATE EXTREME (crypto only)
    if d_dir == 'BUY' and is_crypto:
        funding = onchain_data.get('funding_rate', 0) if onchain_data else 0
        if funding > 0.0008:
            gate_penalties.append(('funding_extreme', 8))

    # AGENT CONFLICT — quant vs news disagree
    q_dir = quant_result.get('direction', '').upper()
    n_sent = news_result.get('sentiment', '').upper()
    agents_disagree = (q_dir == 'BUY' and n_sent == 'BEARISH') or (q_dir == 'SELL' and n_sent == 'BULLISH')
    if agents_disagree and d_dir != 'NO_TRADE':
        gate_penalties.append(('agent_conflict', 5))

    # CMF CONTRADICTION
    cmf = ind.get('cmf', 0)
    if d_dir != 'NO_TRADE':
        if (d_dir == 'BUY' and cmf < -0.15) or (d_dir == 'SELL' and cmf > 0.15):
            gate_penalties.append(('cmf_contradiction', 5))

    # OBV DIVERGENCE
    obv_slope = ind.get('obv_slope', 0)
    if d_dir != 'NO_TRADE':
        if (d_dir == 'BUY' and obv_slope < -0.15) or (d_dir == 'SELL' and obv_slope > 0.15):
            gate_penalties.append(('obv_divergence', 5))

    # VOLUME CONFIRMATION (feature-gated)
    if feature_enabled(req.features, 'volume_confirm'):
        if d_dir != 'NO_TRADE':
            vol_z = ind.get('vol_zscore', 0)
            vol_pct = ind.get('vol_percentile', 50)
            obv_s = ind.get('obv_slope', 0)
            vol_problems = []
            if vol_pct < 25:
                vol_problems.append('low_volume')
            if (d_dir == 'BUY' and obv_s < -0.05) or (d_dir == 'SELL' and obv_s > 0.05):
                vol_problems.append('obv_opposes')
            if vol_z < -0.5:
                vol_problems.append('below_avg')
            if len(vol_problems) >= 2:
                gate_penalties.append(('volume_unconfirmed', 7))
                slog(f"📉 [volume_confirm] Volume does NOT confirm {d_dir}: {', '.join(vol_problems)} → -7 penalty")
            elif vol_problems:
                gate_penalties.append(('volume_weak', 3))
                slog(f"📉 [volume_confirm] Weak volume for {d_dir}: {', '.join(vol_problems)} → -3 penalty")
            else:
                slog(f"📈 [volume_confirm] Volume confirms {d_dir}: z={vol_z:.1f} pct={vol_pct:.0f} obv={obv_s:+.2f}")
        features_applied.append('volume_confirm')
    else:
        slog("⏭ [volume_confirm] Skipped (disabled)")
        features_skipped.append('volume_confirm')

    # ML DISAGREEMENT
    if d_dir != 'NO_TRADE' and ml_result.get('available'):
        ml_score = ml_result.get('score', 50)
        ml_says_up = ml_score > 58
        ml_says_down = ml_score < 42
        if (d_dir == 'BUY' and ml_says_down) or (d_dir == 'SELL' and ml_says_up):
            gate_penalties.append(('ml_disagree', 8))
        elif (d_dir == 'BUY' and ml_score > 62) or (d_dir == 'SELL' and ml_score < 38):
            gate_boosts.append(('ml_confirms', min(10, int(abs(ml_score - 50) * 0.5))))

    # V5 GATES — lighter touch, capped in aggregate
    if d_dir != 'NO_TRADE' and funding_oi_data.get('available'):
        foi_bias = funding_oi_data.get('bias', 0)
        if (d_dir == 'BUY' and foi_bias <= -3) or (d_dir == 'SELL' and foi_bias >= 3):
            gate_penalties.append(('funding_oi_oppose', 5))
        elif (d_dir == 'BUY' and foi_bias >= 3) or (d_dir == 'SELL' and foi_bias <= -3):
            gate_boosts.append(('funding_oi_confirm', 3))

    if d_dir != 'NO_TRADE' and order_flow_data.get('available'):
        of_bias = order_flow_data.get('bias', 0)
        if (d_dir == 'BUY' and of_bias <= -3) or (d_dir == 'SELL' and of_bias >= 3):
            gate_penalties.append(('order_flow_oppose', 5))

    if d_dir != 'NO_TRADE' and vp_signal.get('signal'):
        if vp_signal['signal'] != 'NEUTRAL' and vp_signal['signal'] != d_dir:
            gate_penalties.append(('vp_oppose', 4))
        elif vp_signal['signal'] == d_dir:
            gate_boosts.append(('vp_confirm', 3))

    if d_dir != 'NO_TRADE' and liquidation_data.get('available'):
        liq_bias = liquidation_data.get('bias', 0)
        if (d_dir == 'BUY' and liq_bias >= 2) or (d_dir == 'SELL' and liq_bias <= -2):
            gate_boosts.append(('liq_confirm', 3))
        elif (d_dir == 'BUY' and liq_bias <= -2) or (d_dir == 'SELL' and liq_bias >= 2):
            gate_penalties.append(('liq_oppose', 4))

    if d_dir != 'NO_TRADE' and tick_data.get('available'):
        tick_bias = tick_data.get('bias', 0)
        if (d_dir == 'BUY' and tick_bias <= -3) or (d_dir == 'SELL' and tick_bias >= 3):
            gate_penalties.append(('tick_oppose', 4))

    # UPCOMING EVENT
    if d_dir != 'NO_TRADE' and macro_context.get('upcoming_events'):
        high_impact = [e for e in macro_context['upcoming_events'] if e.get('impact') == 'high']
        if high_impact:
            gate_penalties.append(('upcoming_event', 8))

    # MODEL DISAGREEMENT (V5)
    model_preds = {
        'quant_agent': {'direction': quant_result.get('direction', ''), 'confidence': quant_result.get('confidence', 50)},
        'news_agent': {'direction': 'BUY' if news_result.get('sentiment','').upper() == 'BULLISH' else 'SELL' if news_result.get('sentiment','').upper() == 'BEARISH' else '', 'confidence': news_result.get('confidence', 50)},
        'ml_ensemble': {'direction': 'BUY' if ml_result.get('score', 50) > 55 else 'SELL' if ml_result.get('score', 50) < 45 else '', 'confidence': ml_result.get('score', 50)},
        'decision_agent': {'direction': decision.get('decision', ''), 'confidence': decision.get('confidence', 50)},
    }
    disagreement = compute_disagreement(model_preds)
    if disagreement.get('available') and disagreement.get('disagreement_score', 0) >= 6:
        gate_penalties.append(('model_disagreement', min(8, disagreement['disagreement_score'])))

    # PRE-EVENT (V5)
    upcoming = macro_context.get('upcoming_events', [])
    pre_event_adj = get_pre_event_adjustments(upcoming, hours_ahead=4.0)
    if pre_event_adj.get('active') and d_dir != 'NO_TRADE':
        skip, skip_reason = should_skip_trade(pre_event_adj, decision.get('confidence', 50))
        if skip and not getattr(req, 'bot_mode', False):
            gate_kill = f"Pre-event: {skip_reason}"
        elif pre_event_adj.get('confidence_penalty', 0) > 0:
            gate_penalties.append(('pre_event', min(8, pre_event_adj['confidence_penalty'])))

    # REGIME ADJUSTMENT (V5)
    hmm_probs = ind.get('hmm_probs', {})
    regime_name = get_regime_from_hmm(hmm_probs)
    if d_dir != 'NO_TRADE':
        regime_adj_conf = regime_confidence_adjustment(
            regime_name, decision.get('confidence', 50),
            d_dir, ind.get('trend_slope', 0)
        )
        diff = regime_adj_conf - decision.get('confidence', 50)
        if diff < 0:
            gate_penalties.append(('regime', abs(diff)))
        elif diff > 0:
            gate_boosts.append(('regime', diff))

    # RL-LITE
    rl = get_rl_lite()
    active_gates_list = []
    if funding_oi_data.get('available') and abs(funding_oi_data.get('bias',0)) >= 2:
        active_gates_list.append('funding_oi')
    if order_flow_data.get('available') and abs(order_flow_data.get('bias',0)) >= 2:
        active_gates_list.append('order_flow')
    if vp_signal.get('signal') and vp_signal['signal'] != 'NEUTRAL':
        active_gates_list.append('volume_profile')
    if tick_data.get('available') and abs(tick_data.get('bias',0)) >= 2:
        active_gates_list.append('tick_structure')
    if disagreement.get('disagreement_score', 0) >= 3:
        active_gates_list.append('model_disagreement')
    rl_adj = rl.get_confidence_adjustment(active_gates_list)
    if rl_adj < 0:
        gate_penalties.append(('rl_lite', abs(rl_adj)))
    elif rl_adj > 0:
        gate_boosts.append(('rl_lite', rl_adj))

    # --- APPLY: single capped adjustment ---
    total_penalty = sum(p for _, p in gate_penalties)
    total_boost = sum(b for _, b in gate_boosts)
    MAX_PENALTY = 25
    MAX_BOOST = 15
    capped_penalty = min(total_penalty, MAX_PENALTY)
    capped_boost = min(total_boost, MAX_BOOST)

    # Log all fired gates
    if gate_penalties:
        names = [f"{n}(-{p})" for n, p in gate_penalties]
        slog(f"Gates fired: {', '.join(names)} | raw=-{total_penalty} capped=-{capped_penalty}")
    if gate_boosts:
        names = [f"{n}(+{b})" for n, b in gate_boosts]
        slog(f"Boosts: {', '.join(names)} | raw=+{total_boost} capped=+{capped_boost}")

    # Apply kill gate first
    # V3: gates NEVER veto — they were already handed to the judge as the Risk
    # Officer's notes. The only V3 kill that survives is the pre-event blackout
    # in predict mode. Everything else is logged as post-decision critique.
    v3_kill_allowed = version >= 3 and gate_kill and gate_kill.startswith('Pre-event')
    if gate_kill and d_dir != 'NO_TRADE' and (version < 3 or v3_kill_allowed):
        decision['_original_decision'] = d_dir
        decision['decision'] = 'NO_TRADE'
        gate_reason = gate_kill
        slog(f"GATE KILL: {gate_kill}")
    elif version >= 3 and d_dir != 'NO_TRADE':
        # V3: pure critique — record what the old gates WOULD have done, touch nothing
        if gate_kill:
            slog(f"📋 V3 critique: old pipeline would have KILLED this trade ({gate_kill}) — judge already weighed the facts")
        if gate_penalties or gate_boosts:
            slog(f"📋 V3 critique: old gates would have adjusted {capped_boost - capped_penalty:+d} — not applied, judge's ruling stands")
    elif d_dir != 'NO_TRADE':
        old_conf = decision.get('confidence', 50)
        net_adj = capped_boost - capped_penalty
        new_conf = max(40, min(90, old_conf + net_adj))
        decision['confidence'] = new_conf
        if net_adj != 0:
            slog(f"Gate net adjustment: {old_conf}% → {new_conf}% (net {net_adj:+d})")

    # CONFIDENCE FLOOR — only after consolidated gates (V2 only; V3 has no floor kill)
    conf_floor = get_adaptive_floor()
    if version < 3 and decision.get('decision') != 'NO_TRADE' and decision.get('confidence', 0) < conf_floor:
        decision['_original_decision'] = decision.get('decision')
        decision['decision'] = 'NO_TRADE'
        gate_reason = f"Confidence floor: {decision.get('confidence', 0)}% < {conf_floor}%"
        slog(f"Confidence floor: {decision.get('confidence', 0)}% (floor={conf_floor}%)")

    # ── V3 FINAL: guarantee a direction in bot mode, then blend judge + numbers ──
    if version >= 3:
        if getattr(req, 'bot_mode', False) and decision.get('decision') not in ('BUY', 'SELL'):
            q_dir_v3 = (quant_result.get('direction') or '').upper()
            if q_dir_v3 in ('BUY', 'SELL'):
                decision['_original_decision'] = decision.get('decision')
                decision['decision'] = q_dir_v3
                decision['confidence'] = max(45, min(60, int(quant_result.get('confidence', 55) or 55)))
                slog(f"V3 bot mode: judge gave no direction — using analyst's {q_dir_v3} at {decision['confidence']}%")
        if decision.get('decision') in ('BUY', 'SELL'):
            judge_conf = float(decision.get('confidence', 55) or 55)
            d3 = decision['decision']
            parts = [(judge_conf, 0.50)]
            if ml_conf is not None:
                parts.append((ml_conf if d3 == 'BUY' else 100 - ml_conf, 0.25))
            if cluster_conf is not None:
                parts.append((cluster_conf if d3 == 'BUY' else 100 - cluster_conf, 0.15))
            parts.append((bayes_conf, 0.10))
            total_w = sum(w for _, w in parts)
            final_conf = sum(v * w for v, w in parts) / total_w if total_w else judge_conf
            decision['_judge_confidence'] = round(judge_conf)
            decision['confidence'] = int(round(max(40, min(95, final_conf))))
            slog(f"✓ V3 final: judge {judge_conf:.0f}% (50%) + ML/cluster/Bayes numbers (50%) → {decision['confidence']}%")

    # Save predicted price BEFORE nulling target for NO_TRADE
    # Use ML-informed direction for target instead of bland MC median
    predicted_price = None
    if decision.get('decision') == 'NO_TRADE':
        ml_score_final = ml_result.get('score', 50) if ml_result.get('available') else 50
        if ml_score_final > 55:
            predicted_price = decision.get('price_target') or mc.get('bull')
        elif ml_score_final < 45:
            predicted_price = decision.get('price_target') or mc.get('bear')
        else:
            predicted_price = decision.get('price_target') or mc.get('median')
        decision['price_target'] = None
        decision['price_target_bull'] = None
        decision['price_target_bear'] = None

    # ── PQS (A2) + Update gate stats (A5) ─────────────────────────────────
    pqs = compute_pqs(quant_result, news_result, ml_result, ind, decision, mtf_data,
                       funding_oi=funding_oi_data, order_flow=order_flow_data,
                       vp_signal=vp_signal, tick_data=tick_data)
    slog(f"✓ PQS: {pqs['score']}/10 [{', '.join(pqs['reasons'][:3])}]")

    # Update adaptive gate stats from recent outcomes
    try:
        recent = await get_predictions(req.asset, 50)
        rated = [p for p in recent if p.get('feedback') in ('correct', 'wrong')]
        _gate_stats['trades'] = len(rated)
        _gate_stats['wins'] = sum(1 for p in rated if p['feedback'] == 'correct')
    except Exception:
        pass

    total_ms = int((time.time() - start_time) * 1000)
    slog(f"✅ Complete in {total_ms}ms")

    response = {
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
        "pipeline_version": version,
        "risk_evidence": risk_evidence,
        "judge_confidence": decision.get('_judge_confidence'),
        "flip_trigger": decision.get('flip_trigger'),
        "catalyst_override": news_result.get('catalyst_override', False),
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
            "rsi_div_bull": ind.get('rsi_div_bull', False), "rsi_div_bear": ind.get('rsi_div_bear', False),
        },
        "ind_snapshot": json.dumps(ind),
        "ml": ml_result,
        "pqs": pqs,
        "rsi_divergence": {
            "bullish": ind.get('rsi_div_bull', False),
            "bearish": ind.get('rsi_div_bear', False),
        },
        "sentiment": sentiment_data,
        "macro_context": macro_context,
        "correlation": correlation_data,
        "cluster": cluster_data,
        "smc": smc_data,
        "orderbook": orderbook_data,
        "whale_activity": whale_data,
        "options_flow": options_data,
        "smart_money": smart_money_data,
        "monte_carlo": mc,
        # V5: New intelligence data
        "funding_oi": funding_oi_data,
        "liquidations": liquidation_data,
        "order_flow": order_flow_data,
        "volume_profile": vp_data,
        "vp_signal": vp_signal,
        "tick_structure": tick_data,
        "disagreement": disagreement,
        "pre_event": pre_event_adj if pre_event_adj.get('active') else None,
        "rl_trust": rl.get_report(),
        "regime_strategy": regime_name,
        "similarity": {
            "count": len(similar),
            "win_rate": sum(1 for s in similar if (s.get('fwd_4h') or 0) > 0) / len(similar) * 100 if similar else 0,
            "avg_fwd_4h": sum(s.get('fwd_4h') or 0 for s in similar) / len(similar) if similar else 0,
        },
        "bayesian_conf": bayes_conf,
        "raw_ai_conf": raw_ai_conf,
        "db_sentiment": db_sentiment,
        "candles": candles[-100:],
        "recent_prices": recent_prices,
        "logs": logs,
        "duration_ms": total_ms,
        "features_applied": features_applied,
        "features_skipped": features_skipped,
    }

    # Calibration (E3)
    try:
        cal = await calibrate_confidence(response['confidence'], req.asset, req.horizon, rated_preds)
        response['calibration'] = cal
        slog(f"✓ Calibration: raw={cal['raw']} calibrated={cal['calibrated']} ({cal['reliability']})")
    except Exception:
        response['calibration'] = None

    # A6: Record session prediction (skip for compare-mode shadow runs)
    if not getattr(req, 'shadow', False):
        if req.asset not in _session_predictions:
            _session_predictions[req.asset] = []
        _session_predictions[req.asset].append({
            'ts': int(time.time()),
            'direction': response['decision'],
            'confidence': response['confidence'],
        })
        _session_predictions[req.asset] = _session_predictions[req.asset][-10:]

        # Send Telegram notification (non-blocking)
        asyncio.create_task(send_prediction(response, req.asset, req.horizon))

    return response


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

        # A10: Post-mortem Learning — log what indicators said vs what happened
        postmortem = _postmortem_analysis(req, feedback, moved_pct, price)
        if postmortem:
            print(f"[POST-MORTEM] {req.asset} {req.horizon}H: {postmortem}")

        # Track equity curve
        _equity_tracker.record_outcome({
            'feedback': feedback, 'moved_pct': moved_pct,
            'asset': req.asset, 'decision': decision,
            'rated_at': int(time.time()),
        })

        # Trigger ML retrain asynchronously
        asyncio.create_task(_async_retrain(req.asset))

        return {
            "price": price,
            "moved_pct": moved_pct,
            "feedback": feedback,
            "target_hit": target_hit,
            "note": note,
            "postmortem": postmortem,
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
        data = get_latest_alerts()
        return data
    except Exception as e:
        return {"alerts": [], "count": 0, "error": str(e)}


@app.get("/scanner")
async def run_scanner():
    """Run a full scanner pass on demand and return results."""
    try:
        alerts = await scan_for_alerts()
        return {"alerts": alerts, "count": len(alerts), "ts": int(time.time())}
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
        "telegram_configured": is_configured('TELEGRAM_BOT_TOKEN'),
    }


@app.post("/settings")
async def save_settings(req: SettingsRequest):
    """Save API key settings from frontend."""
    saved = []
    for key, value in req.settings.items():
        if key in ('FRED_API_KEY', 'REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET',
                    'ALPACA_KEY', 'ALPACA_SECRET', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID') and value:
            os.environ[key] = value
            set_setting(key, value)
            saved.append(key)
    return {"ok": True, "saved": saved}


# ─── E1: Backtester Endpoint ───────────────────────────────────────────────

class BacktestRequest(BaseModel):
    asset: str
    horizon: int = 4
    periods: int = 50


@app.post("/backtest")
async def backtest(req: BacktestRequest):
    """E1: Walk-forward backtest over the last N candle periods.

    For each period, compute indicators on the history available at that point,
    derive a simple rule-based directional signal, then compare against
    what actually happened over the next *horizon* candles.
    Returns win rate and per-period stats.
    """
    try:
        iv = '1h' if req.horizon <= 8 else '4h' if req.horizon <= 72 else '1d'
        # Need enough candles: warmup (60) + periods + horizon lookahead
        need = 60 + req.periods + req.horizon
        candles = await fetch_candles(req.asset, iv, min(need + 50, 500))
        if not candles or len(candles) < 60 + req.periods + req.horizon:
            raise HTTPException(400, f"Not enough candle data (got {len(candles) if candles else 0}, need {60 + req.periods + req.horizon})")

        results = []
        wins = 0
        losses = 0
        skipped = 0

        # Walk forward: for each period i, use candles[:end_idx] to compute
        # indicators and candles[end_idx : end_idx + horizon] as the outcome.
        total_candles = len(candles)
        start_offset = total_candles - req.periods - req.horizon

        for i in range(req.periods):
            end_idx = start_offset + i  # last candle available for indicators
            if end_idx < 60:
                skipped += 1
                continue

            hist = candles[:end_idx + 1]
            ind = compute_indicators(hist[-300:])
            if not ind:
                skipped += 1
                continue

            # Simple rule-based signal from indicators
            bull_signals = 0
            bear_signals = 0

            # RSI
            if ind['rsi14'] < 35:
                bull_signals += 1
            elif ind['rsi14'] > 65:
                bear_signals += 1

            # MACD
            if ind['macd_hist'] > 0:
                bull_signals += 1
            else:
                bear_signals += 1

            # EMA alignment
            if ind['ema_align_bull'] >= 3:
                bull_signals += 1
            if ind['ema_align_bear'] >= 3:
                bear_signals += 1

            # Supertrend
            if ind['supertrend_bull']:
                bull_signals += 1
            else:
                bear_signals += 1

            # VWAP
            if ind['dist_vwap'] > 0:
                bull_signals += 1
            else:
                bear_signals += 1

            # CMF
            if ind['cmf'] > 0.05:
                bull_signals += 1
            elif ind['cmf'] < -0.05:
                bear_signals += 1

            # Ichimoku
            if ind['ich_bull']:
                bull_signals += 1
            elif ind['ich_bear']:
                bear_signals += 1

            # Determine signal
            if bull_signals > bear_signals + 1:
                signal = 'BUY'
            elif bear_signals > bull_signals + 1:
                signal = 'SELL'
            else:
                signal = 'NO_TRADE'
                skipped += 1
                continue

            # Outcome: price change over next horizon candles
            outcome_idx = min(end_idx + req.horizon, total_candles - 1)
            entry_price = candles[end_idx]['close']
            outcome_price = candles[outcome_idx]['close']
            moved_pct = (outcome_price - entry_price) / entry_price * 100

            correct = (signal == 'BUY' and moved_pct > 0) or (signal == 'SELL' and moved_pct < 0)
            if correct:
                wins += 1
            else:
                losses += 1

            results.append({
                'period': i + 1,
                'signal': signal,
                'entry': round(entry_price, 6),
                'outcome': round(outcome_price, 6),
                'moved_pct': round(moved_pct, 3),
                'correct': correct,
                'rsi': round(ind['rsi14'], 1),
                'macd_hist': round(ind['macd_hist'], 6),
                'regime': ind['regime'],
            })

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        avg_move = sum(r['moved_pct'] for r in results) / len(results) if results else 0
        avg_win = sum(r['moved_pct'] for r in results if r['correct']) / wins if wins > 0 else 0
        avg_loss = sum(r['moved_pct'] for r in results if not r['correct']) / losses if losses > 0 else 0

        return {
            'asset': req.asset,
            'horizon': req.horizon,
            'periods': req.periods,
            'interval': iv,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'skipped': skipped,
            'win_rate': round(win_rate, 1),
            'avg_move_pct': round(avg_move, 3),
            'avg_win_pct': round(avg_win, 3),
            'avg_loss_pct': round(avg_loss, 3),
            'profit_factor': round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
            'results': results,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Backtest failed: {str(e)[:200]}")


# ─── Trading Endpoints (D1-D11) ──────────────────────────────────────────

class TradeOpenRequest(BaseModel):
    asset: str
    direction: str
    price: float
    size_usd: float = 100.0
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    trailing_pct: float = 1.5
    confidence: int = 55
    pqs_score: int = 5


@app.post("/trade/open")
async def trade_open(req: TradeOpenRequest):
    engine = get_trading_engine()
    allowed, reason = engine.can_trade(req.confidence, req.pqs_score)
    if not allowed:
        return {"ok": False, "error": reason}
    result = await engine.open_position(
        req.asset, req.direction, req.price, req.size_usd,
        req.stop_loss_pct, req.take_profit_pct, req.trailing_pct
    )
    return result


class TradeCloseRequest(BaseModel):
    position_id: str
    price: float
    reason: str = "manual"


@app.post("/trade/close")
async def trade_close(req: TradeCloseRequest):
    engine = get_trading_engine()
    return await engine.close_position(req.position_id, req.price, req.reason)


@app.get("/trade/positions")
async def trade_positions(status: str = None):
    engine = get_trading_engine()
    return {"positions": engine.get_positions(status), "equity": engine._equity}


@app.get("/trade/heartbeat")
async def trade_heartbeat():
    engine = get_trading_engine()
    return engine.heartbeat()


@app.get("/trade/log")
async def trade_log(limit: int = 50):
    engine = get_trading_engine()
    return {"trades": engine.get_trade_log(limit)}


class PaperModeRequest(BaseModel):
    enabled: bool


@app.post("/trade/paper")
async def trade_paper(req: PaperModeRequest):
    engine = get_trading_engine()
    return engine.set_paper_mode(req.enabled)


class WebhookPayload(BaseModel):
    ticker: str
    action: str
    price: float


@app.post("/trade/webhook")
async def trade_webhook(payload: WebhookPayload):
    engine = get_trading_engine()
    return await engine.handle_webhook(payload.dict())


# ─── Autonomous Trader Control ──────────────────────────────────────────

@app.get("/keys/health")
async def keys_health():
    """Live-test the LLM API keys with a 1-token call each. Catches invalid keys,
    empty credit, and outages that otherwise fail SILENTLY and leave the bot blind."""
    import httpx as _httpx
    from config import OPENAI_API_KEY, DEEPSEEK_API_KEY

    async def probe(url: str, key: str, model: str) -> dict:
        if not key or len(key) < 10:
            return {'configured': False, 'ok': False, 'detail': 'not set'}
        try:
            async with _httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "max_tokens": 1,
                          "messages": [{"role": "user", "content": "ping"}]},
                )
                if resp.status_code == 200:
                    return {'configured': True, 'ok': True, 'detail': 'OK'}
                return {'configured': True, 'ok': False,
                        'detail': f"HTTP {resp.status_code}: {resp.text[:140]}"}
        except Exception as e:
            return {'configured': True, 'ok': False, 'detail': str(e)[:140]}

    ds, oa = await asyncio.gather(
        probe("https://api.deepseek.com/v1/chat/completions", DEEPSEEK_API_KEY or '', "deepseek-chat"),
        probe("https://api.openai.com/v1/chat/completions", OPENAI_API_KEY or '', "gpt-4o-mini"),
    )
    return {'deepseek': ds, 'openai': oa, 'any_ok': bool(ds['ok'] or oa['ok'])}


class AutotraderStartRequest(BaseModel):
    assets: list
    interval_minutes: int = 60
    trade_size: float = 0
    starting_equity: float = 10000
    force_trade: bool = True
    use_local: bool = False
    local_url: str = "http://localhost:11434"
    local_model: str = "qwen2.5:7b"
    hard_stop_pct: float = 5.0
    features: Optional[dict] = None
    pipeline_version: int = 3    # 3 = new brain (ungagged judge), 2 = classic
    compare_mode: bool = False   # run both brains, shadow-score the inactive one
    time_decay_hours: float = 7.0


@app.post("/autotrader/start")
async def autotrader_start(req: AutotraderStartRequest):
    valid_assets = [a for a in req.assets if a in ALL_ASSETS]
    if not valid_assets:
        return {"ok": False, "error": f"No valid assets. Choose from: {ALL_ASSETS}"}
    # Stop any existing loop before starting a new one
    if _autotrader['_loop_running']:
        _autotrader['enabled'] = False
        await asyncio.sleep(1)
        _autotrader['_loop_running'] = False
    _autotrader['enabled'] = True
    _autotrader['assets'] = valid_assets
    _autotrader['interval_minutes'] = max(10, req.interval_minutes)
    _autotrader['trade_size'] = max(0, req.trade_size)
    _autotrader['starting_equity'] = max(100, req.starting_equity)
    _autotrader['force_trade'] = req.force_trade
    _autotrader['use_local'] = req.use_local
    _autotrader['local_url'] = req.local_url
    _autotrader['local_model'] = req.local_model
    _autotrader['hard_stop_pct'] = max(1.0, min(10.0, req.hard_stop_pct))
    _autotrader['features'] = req.features
    _autotrader['pipeline_version'] = 3 if req.pipeline_version >= 3 else 2
    _autotrader['compare_mode'] = bool(req.compare_mode)
    _autotrader['time_decay_hours'] = max(1.0, min(24.0, req.time_decay_hours))
    # Fresh shadow book every start — same starting equity as the active book
    global _shadow_engine_inst
    _shadow_engine_inst = TradingEngine()
    _shadow_engine_inst._equity = max(100, req.starting_equity)
    _autotrader['status'] = 'starting'
    engine = get_trading_engine()
    engine._equity = _autotrader['starting_equity']
    asyncio.create_task(_autotrader_loop())
    from telegram_bot import send_message
    size_label = f"${_autotrader['trade_size']:.0f}/trade" if _autotrader['trade_size'] > 0 else "Auto (Kelly)"
    asyncio.create_task(send_message(
        f"AUTOTRADER STARTED\nAssets: {', '.join(valid_assets)}\n"
        f"Interval: every {_autotrader['interval_minutes']} min\n"
        f"Mode: {'PAPER' if engine.paper_mode else 'LIVE'}\n"
        f"Starting equity: ${engine._equity:,.2f}\nTrade size: {size_label}"
    ))
    return {
        "ok": True, "assets": valid_assets,
        "interval_minutes": _autotrader['interval_minutes'],
        "trade_size": _autotrader['trade_size'],
        "starting_equity": _autotrader['starting_equity'],
        "mode": "paper" if engine.paper_mode else "live",
        "pipeline_version": _autotrader['pipeline_version'],
        "compare_mode": _autotrader['compare_mode'],
    }


@app.post("/autotrader/stop")
async def autotrader_stop():
    _autotrader['enabled'] = False
    _autotrader['status'] = 'stopped'
    _autotrader['_loop_running'] = False
    engine = get_trading_engine()
    open_positions = engine.get_positions('open')
    from telegram_bot import send_message
    asyncio.create_task(send_message(
        f"AUTOTRADER STOPPED\nOpen positions: {len(open_positions)}\n"
        f"Equity: ${engine._equity:,.2f}\nTotal cycles: {_autotrader['total_cycles']}"
    ))
    return {"ok": True, "open_positions": len(open_positions), "equity": engine._equity,
            "total_cycles": _autotrader['total_cycles']}


@app.post("/autotrader/cashout")
async def autotrader_cashout():
    _autotrader['enabled'] = False
    _autotrader['status'] = 'stopped'
    engine = get_trading_engine()
    closed = []
    total_pnl = 0.0
    for pos in list(engine.positions.values()):
        if pos.status != 'open':
            continue
        try:
            result = await fetch_current_price(pos.asset)
            price = result.get('price', 0)
            if price:
                close_result = await engine.close_position(pos.id, price, 'cashout')
                pnl = close_result.get('pnl', 0)
                total_pnl += pnl
                closed.append({'asset': pos.asset, 'direction': pos.direction,
                               'entry': pos.entry_price, 'exit': price, 'pnl': pnl})
        except Exception:
            continue
    return {"ok": True, "closed": closed, "total_pnl": round(total_pnl, 2),
            "final_equity": round(engine._equity, 2)}


@app.post("/autotrader/cashout-winners")
async def autotrader_cashout_winners():
    """Close only positions currently in profit; leave losers open to recover.
    Does NOT stop the bot — losers keep being managed by the loop."""
    from telegram_bot import send_message
    engine = get_trading_engine()
    closed = []
    held = []
    total_pnl = 0.0
    for pos in list(engine.positions.values()):
        if pos.status != 'open':
            continue
        try:
            result = await fetch_current_price(pos.asset)
            price = result.get('price', 0)
            if not price:
                continue
            if pos.direction == 'BUY':
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
            else:
                pnl_pct = (pos.entry_price - price) / pos.entry_price * 100
            if pnl_pct > 0:
                close_result = await engine.close_position(pos.id, price, 'cashout_winners')
                pnl = close_result.get('pnl', 0)
                total_pnl += pnl
                closed.append({'asset': pos.asset, 'direction': pos.direction,
                               'entry': pos.entry_price, 'exit': price, 'pnl': pnl})
            else:
                held.append({'asset': pos.asset, 'direction': pos.direction,
                             'entry': pos.entry_price, 'pnl_pct': round(pnl_pct, 2)})
        except Exception:
            continue
    asyncio.create_task(send_message(
        f"💰 CASHOUT WINNERS: closed {len(closed)} winning positions for "
        f"${total_pnl:+,.2f} | {len(held)} losers held open"
    ))
    return {"ok": True, "closed": closed, "held": held,
            "total_pnl": round(total_pnl, 2),
            "winners_closed": len(closed), "losers_held": len(held),
            "final_equity": round(engine._equity, 2)}


@app.get("/autotrader/status-shadow")
async def autotrader_status_shadow():
    """Status of the COMPARE-mode shadow book — the inactive brain's own ledger.
    Same shape as /autotrader/status so the dashboard can render it directly."""
    sh = get_shadow_engine()
    active_v = _autotrader.get('pipeline_version', 3)
    shadow_v = 2 if active_v >= 3 else 3
    open_positions = sh.get_positions('open')
    trade_log = sh.get_trade_log(20)
    wins = sum(1 for t in trade_log if t.get('pnl', 0) > 0)
    total = len(trade_log)
    now = int(time.time())
    unrealized_total = 0.0
    open_winners = 0
    open_losers = 0
    position_health = []
    for pos_data in open_positions:
        try:
            resp = await fetch_current_price(pos_data['asset'])
            price = resp.get('price', 0)
        except Exception:
            price = 0
        if price and pos_data.get('entry_price'):
            if pos_data['direction'] == 'BUY':
                pnl_pct = (price - pos_data['entry_price']) / pos_data['entry_price'] * 100
            else:
                pnl_pct = (pos_data['entry_price'] - price) / pos_data['entry_price'] * 100
            pnl_usd = pnl_pct / 100 * pos_data.get('size', 0)
            unrealized_total += pnl_usd
            if pnl_pct >= 0:
                open_winners += 1
            else:
                open_losers += 1
            position_health.append({
                'asset': pos_data['asset'], 'direction': pos_data['direction'],
                'entry_price': pos_data['entry_price'], 'current_price': round(price, 4),
                'unrealized_pnl_pct': round(pnl_pct, 2), 'unrealized_pnl_usd': round(pnl_usd, 2),
                'if_closed_now': round(pnl_usd, 2),
                'age_minutes': round((now - pos_data.get('entry_time', now)) / 60),
                'sl_distance_pct': 0, 'tp_distance_pct': 0, 'entry_confidence': 0,
                'status': 'winning' if pnl_pct >= 0 else 'losing',
            })
    all_recent = trade_log + [{'pnl': h['unrealized_pnl_usd']} for h in position_health]
    true_wins = sum(1 for t in all_recent if t.get('pnl', 0) > 0)
    true_total = len(all_recent)
    return {
        "enabled": _autotrader['enabled'], "status": _autotrader['status'],
        "brain": f"V{shadow_v}", "is_shadow": True,
        "compare_mode": _autotrader.get('compare_mode', False),
        "assets": _autotrader['assets'], "interval_minutes": _autotrader['interval_minutes'],
        "trade_size": _autotrader.get('trade_size', 0),
        "total_cycles": _autotrader['total_cycles'],
        "trades_opened": len(sh.positions),
        "last_cycle": _autotrader['last_cycle'], "seconds_until_next": 0,
        "heartbeat": sh.heartbeat(),
        "open_positions": open_positions, "recent_trades": trade_log,
        "win_rate": round((wins / total * 100) if total > 0 else 0, 1),
        "true_win_rate": round((true_wins / true_total * 100) if true_total > 0 else 0, 1),
        "mtm_equity": round(sh._equity + unrealized_total, 2),
        "unrealized_pnl": round(unrealized_total, 2),
        "cashout_impact": round(unrealized_total, 2),
        "open_winners": open_winners, "open_losers": open_losers,
        "position_health": position_health,
        "pipeline_version": shadow_v,
        "version_scoreboard": {
            vkey: {
                'decisions': sb['decisions'], 'resolved': sb['resolved'],
                'wins': sb['wins'], 'losses': sb['losses'],
                'win_rate': round(sb['wins'] / sb['resolved'] * 100, 1) if sb['resolved'] else 0,
                'avg_return_pct': round(sb['sum_ret'] / sb['resolved'], 3) if sb['resolved'] else 0,
            } for vkey, sb in _version_scoreboard.items()
        },
        "cycle_log": [], "lessons": {},
        "all_positions": sh.get_positions(),
    }


@app.get("/autotrader/status")
async def autotrader_status():
    engine = get_trading_engine()
    hb = engine.heartbeat()
    open_positions = engine.get_positions('open')
    trade_log = engine.get_trade_log(20)
    wins = sum(1 for t in trade_log if t.get('pnl', 0) > 0)
    total = len(trade_log)
    win_rate = (wins / total * 100) if total > 0 else 0
    now = int(time.time())
    last = _autotrader['last_cycle']
    interval_sec = _autotrader['interval_minutes'] * 60
    if _autotrader['enabled'] and last > 0:
        next_at = last + interval_sec
        secs_left = max(0, next_at - now)
    else:
        secs_left = 0

    # ── Mark-to-market: compute unrealized P&L for every open position
    unrealized_total = 0.0
    open_winners = 0
    open_losers = 0
    position_health = []
    for pos_data in open_positions:
        try:
            result = await fetch_current_price(pos_data['asset'])
            price = result.get('price', 0)
        except Exception:
            price = 0
        if price and pos_data.get('entry_price'):
            if pos_data['direction'] == 'BUY':
                pnl_pct = (price - pos_data['entry_price']) / pos_data['entry_price'] * 100
            else:
                pnl_pct = (pos_data['entry_price'] - price) / pos_data['entry_price'] * 100
            pnl_usd = pnl_pct / 100 * pos_data.get('size', 0)
            unrealized_total += pnl_usd
            if pnl_pct >= 0:
                open_winners += 1
            else:
                open_losers += 1
            age_min = round((now - pos_data.get('entry_time', now)) / 60)
            sl_dist = abs(price - pos_data.get('stop_loss', price)) / price * 100 if price else 0
            tp_dist = abs(pos_data.get('take_profit', price) - price) / price * 100 if price else 0
            position_health.append({
                'asset': pos_data['asset'],
                'direction': pos_data['direction'],
                'entry_price': pos_data['entry_price'],
                'current_price': round(price, 4),
                'unrealized_pnl_pct': round(pnl_pct, 2),
                'unrealized_pnl_usd': round(pnl_usd, 2),
                'if_closed_now': round(pnl_usd, 2),
                'age_minutes': age_min,
                'sl_distance_pct': round(sl_dist, 2),
                'tp_distance_pct': round(tp_dist, 2),
                'entry_confidence': pos_data.get('entry_confidence', 0),
                'status': 'winning' if pnl_pct >= 0 else 'losing',
            })

    mtm_equity = round(engine._equity + unrealized_total, 2)
    all_recent = trade_log + [{'pnl': h['unrealized_pnl_usd']} for h in position_health]
    true_wins = sum(1 for t in all_recent if t.get('pnl', 0) > 0)
    true_total = len(all_recent)
    true_win_rate = round((true_wins / true_total * 100) if true_total > 0 else 0, 1)
    cashout_impact = round(unrealized_total, 2)

    return {
        "enabled": _autotrader['enabled'], "status": _autotrader['status'],
        "assets": _autotrader['assets'], "interval_minutes": _autotrader['interval_minutes'],
        "trade_size": _autotrader.get('trade_size', 0),
        "total_cycles": _autotrader['total_cycles'],
        "trades_opened": _autotrader['trades_opened'],
        "last_cycle": last,
        "seconds_until_next": secs_left,
        "heartbeat": hb, "open_positions": open_positions,
        "recent_trades": trade_log, "win_rate": round(win_rate, 1),
        "true_win_rate": true_win_rate,
        "mtm_equity": mtm_equity,
        "unrealized_pnl": round(unrealized_total, 2),
        "cashout_impact": cashout_impact,
        "open_winners": open_winners,
        "open_losers": open_losers,
        "position_health": position_health,
        "pipeline_version": _autotrader.get('pipeline_version', 3),
        "compare_mode": _autotrader.get('compare_mode', False),
        "version_scoreboard": {
            vkey: {
                'decisions': sb['decisions'], 'resolved': sb['resolved'],
                'wins': sb['wins'], 'losses': sb['losses'],
                'win_rate': round(sb['wins'] / sb['resolved'] * 100, 1) if sb['resolved'] else 0,
                'avg_return_pct': round(sb['sum_ret'] / sb['resolved'], 3) if sb['resolved'] else 0,
            } for vkey, sb in _version_scoreboard.items()
        },
        "cycle_log": _autotrader['cycle_log'][-20:],
        "lessons": {asset: lessons[-5:] for asset, lessons in _trade_lessons.items()},
        "all_positions": engine.get_positions(),
        "self_correction": {asset: {
            'wins': a['wins'], 'losses': a['losses'], 'total': a['total'],
            'win_rate': round(a['wins'] / a['total'] * 100, 1) if a['total'] > 0 else 0,
            'streak': a['streak'], 'size_factor': _get_asset_size_factor(asset),
        } for asset, a in _asset_accuracy.items()},
    }


# ─── Forensic Log Endpoints ─────────────────────────────────────────────

@app.get("/forensic/stats")
async def forensic_stats():
    return forensic_log.stats()


@app.get("/forensic/log")
async def forensic_log_get(limit: int = 200, asset: str = None, type: str = None):
    events = forensic_log.get_events(limit=limit, asset_filter=asset, type_filter=type)
    return {"events": events, "count": len(events)}


@app.get("/forensic/export")
async def forensic_export(asset: str = None):
    text = forensic_log.export_text(asset_filter=asset)
    return PlainTextResponse(content=text, media_type="text/plain")


@app.get("/forensic/export.json")
async def forensic_export_json(asset: str = None):
    events = forensic_log.get_events(limit=5000, asset_filter=asset)
    return {
        "generated": int(time.time()),
        "total_events": len(events),
        "events": events,
    }


@app.post("/forensic/clear")
async def forensic_clear():
    forensic_log.clear()
    return {"ok": True}


@app.post("/autotrader/reset")
async def autotrader_reset():
    """Full factory reset — wipes all state back to zero."""
    global _trade_lessons, _asset_accuracy, _session_predictions, _ml_cache, _gate_stats

    _autotrader['enabled'] = False
    _autotrader['status'] = 'stopped'
    _autotrader['_loop_running'] = False
    _autotrader['last_cycle'] = 0
    _autotrader['total_cycles'] = 0
    _autotrader['trades_opened'] = 0
    _autotrader['trades_closed'] = 0
    _autotrader['cycle_log'] = []

    engine = get_trading_engine()
    engine.positions.clear()
    engine.daily_pnl = 0.0
    engine.daily_trades = 0
    engine._equity = _autotrader.get('starting_equity', 10000)
    engine._trade_log.clear()

    _trade_lessons.clear()
    _asset_accuracy.clear()
    _session_predictions.clear()
    _ml_cache.clear()
    _gate_stats['trades'] = 0
    _gate_stats['wins'] = 0

    _equity_tracker._curve.clear()

    forensic_log.clear()

    return {"ok": True, "message": "Full reset complete — all history, trades, and AI learning wiped"}


# ─── Portfolio Scanner ───────────────────────────────────────────────────

@app.get("/portfolio/scan")
async def portfolio_scan():
    try:
        return await scan_portfolio()
    except Exception as e:
        return {"assets": [], "error": str(e)}


# ─── Equity Tracker ─────────────────────────────────────────────────────

@app.get("/equity")
async def get_equity():
    return {
        "curve": _equity_tracker.get_curve(),
        "stats": _equity_tracker.get_stats(),
    }


# ─── Calibration ────────────────────────────────────────────────────────

@app.get("/calibrate")
async def get_calibration(asset: str = 'BTC', horizon: int = 4):
    preds = await get_predictions(asset, 500)
    cal = await calibrate_confidence(50, asset, horizon, preds)
    return cal


# ─── WebSocket ──────────────────────────────────────────────────────────

from fastapi import WebSocket as WSType


@app.websocket("/ws")
async def ws_endpoint(websocket: WSType):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "pong", "ts": int(time.time())})
    except Exception:
        ws_manager.disconnect(websocket)


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


# ─── V5: Periodic Model Retrain Check ──────────────────────────────────────

async def _periodic_retrain_check():
    """Check every 6 hours if any models need retraining."""
    await asyncio.sleep(60)  # wait for startup
    while True:
        try:
            retrainer = get_retrainer()
            for asset in list(BINANCE_SYMBOLS.keys())[:5]:
                if retrainer.needs_retrain(asset):
                    print(f"[V5] Auto-retrain triggered for {asset}")
                    preds = await get_predictions(asset, 500)
                    retrainer.retrain_if_needed(asset, preds)
            # Decay RL trust scores toward neutral
            get_rl_lite().decay_unused()
        except Exception as e:
            print(f"[V5] Retrain check error: {e}")
        await asyncio.sleep(6 * 3600)  # every 6 hours


# ─── V5: New Intelligence Endpoints ────────────────────────────────────────

@app.get("/v5/funding-oi/{asset}")
async def v5_funding_oi(asset: str):
    return await get_funding_oi_combined(asset)


@app.get("/v5/liquidations/{asset}")
async def v5_liquidations(asset: str):
    from indicators import compute_indicators
    candles = await fetch_candles(asset, '1h', 50)
    price = candles[-1]['close'] if candles else 0
    return await get_liquidation_intel(asset, price)


@app.get("/v5/order-flow/{asset}")
async def v5_order_flow(asset: str):
    return await get_order_flow(asset)


@app.get("/v5/volume-profile/{asset}")
async def v5_volume_profile(asset: str):
    candles = await fetch_candles(asset, '1h', 200)
    if not candles:
        return {"available": False}
    vp = compute_volume_profile(candles)
    signal = volume_profile_signal(vp, candles[-1]['close']) if vp.get('available') else {}
    return {"profile": vp, "signal": signal}


@app.get("/v5/tick-structure/{asset}")
async def v5_tick_structure(asset: str):
    return get_tick_engine().get_micro_structure(asset)


@app.get("/v5/rl-report")
async def v5_rl_report():
    return get_rl_lite().get_report()


@app.get("/v5/execution-report")
async def v5_execution_report():
    return get_execution_optimizer().get_report()


@app.get("/v5/walkforward/{asset}")
async def v5_walkforward(asset: str, horizon: int = 4):
    tester = get_walkforward_tester()
    preds = await get_predictions(asset, 500)
    if len(preds) < 30:
        return {"available": False, "reason": "Need 30+ predictions for walk-forward"}
    result = tester.run(preds, asset, horizon)
    return result


@app.get("/v5/feature-importance/{asset}")
async def v5_feature_importance(asset: str):
    pruner = get_feature_pruner()
    preds = await get_predictions(asset, 500)
    if len(preds) < 50:
        return {"available": False, "reason": "Need 50+ predictions with indicators"}
    return pruner.analyze(preds)


@app.get("/v5/regime-strategy")
async def v5_regime_strategy(asset: str = 'BTC'):
    candles = await fetch_candles(asset, '1h', 100)
    if not candles:
        return {"available": False}
    ind = compute_indicators(candles)
    if not ind:
        return {"available": False}
    hmm_probs = ind.get('hmm_probs', {})
    regime = get_regime_from_hmm(hmm_probs)
    adjustments = apply_regime_adjustments(regime, {
        'confidence': 60, 'position_size_pct': 100,
        'stop_loss_pct': 2.0, 'take_profit_pct': 4.0,
    })
    return {
        "regime": regime,
        "hmm_probs": hmm_probs,
        "adjustments": adjustments,
    }


@app.get("/v5/disagreement")
async def v5_disagreement():
    """Show current model disagreement analysis for last prediction."""
    return {"info": "Disagreement computed per-prediction. Check prediction response 'disagreement' field."}


@app.get("/v5/intelligence-summary/{asset}")
async def v5_intelligence_summary(asset: str):
    """One-shot summary of all V5 intelligence for an asset."""
    is_crypto = asset in BINANCE_SYMBOLS
    results = {}
    try:
        if is_crypto:
            funding_oi, liq, of = await asyncio.gather(
                get_funding_oi_combined(asset),
                get_liquidation_intel(asset, 0),
                get_order_flow(asset),
            )
            results['funding_oi'] = funding_oi
            results['liquidations'] = liq
            results['order_flow'] = of
        results['tick'] = get_tick_engine().get_micro_structure(asset)
        results['rl'] = get_rl_lite().get_report()
        results['execution'] = get_execution_optimizer().get_report()
    except Exception as e:
        results['error'] = str(e)
    return results
