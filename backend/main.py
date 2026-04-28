"""
ULTRAMAX Backend — FastAPI
All prediction logic, database queries, API orchestration
"""
import asyncio
import json
import os
import time
from typing import Optional, Dict
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
from telegram_bot import send_prediction, send_scanner_summary, send_message
from trading_engine import get_engine as get_trading_engine
from portfolio import scan_portfolio
from calibration import calibrate_confidence
from equity_tracker import EquityTracker
from ws_manager import ws_manager
from smc_engine import detect_smc
from orderbook import get_orderbook_imbalance
from whale_monitor import get_whale_activity
from options_flow import get_options_sentiment
from smart_money import analyze_smart_money
from smart_money_intel import get_smart_money_score, refresh_all_smart_money, get_source_leaderboard
import forensic_log

app = FastAPI(title="ULTRAMAX Backend", version="3.0")

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
    'cycle_log': [],  # [{ts, cycle, summaries: [{asset, action, detail}]}]
    'trade_size': 0,  # 0 = auto (Kelly), >0 = fixed USD per trade
    'starting_equity': 10000,  # paper trading starting balance
}

# ─── Learning Memory ────────────────────────────────────────────────────────
# Every closed trade gets logged as a "lesson". Before each cycle, the bot
# reviews its last few trades per asset to avoid repeating mistakes.
_trade_lessons: Dict[str, list] = {}  # {asset: [{ts, direction, entry, exit, pnl, reason, was_correct, lesson}]}

def record_lesson(asset: str, direction: str, entry: float, exit_price: float,
                   pnl: float, reason: str):
    if asset not in _trade_lessons:
        _trade_lessons[asset] = []
    was_correct = pnl > 0
    pnl_pct = (exit_price - entry) / entry * 100 if direction == 'BUY' else (entry - exit_price) / entry * 100
    if was_correct:
        lesson = f"{direction} was correct ({pnl_pct:+.2f}%). Closed via {reason}."
    else:
        lesson = f"{direction} was WRONG ({pnl_pct:+.2f}%). Should have gone {'SELL' if direction == 'BUY' else 'BUY'}. Lost ${abs(pnl):.2f}."
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

def get_lessons_context(asset: str) -> str:
    lessons = _trade_lessons.get(asset, [])
    if not lessons:
        return ""
    recent = lessons[-5:]
    wins = sum(1 for l in lessons if l['was_correct'])
    total = len(lessons)
    lines = [f"BOT LEARNING MEMORY ({wins}/{total} correct):"]
    for l in recent:
        lines.append(f"  [{l['direction']}] entry ${l['entry']:.2f} → exit ${l['exit']:.2f} | {l['pnl_pct']:+.2f}% | {l['lesson']}")
    if total >= 3:
        wrong = [l for l in lessons if not l['was_correct']]
        if len(wrong) >= 2:
            last_wrong_dirs = [l['direction'] for l in wrong[-3:]]
            if all(d == last_wrong_dirs[0] for d in last_wrong_dirs):
                lines.append(f"  ⚠ PATTERN: Last {len(last_wrong_dirs)} losing trades were all {last_wrong_dirs[0]} — consider bias toward {('SELL' if last_wrong_dirs[0] == 'BUY' else 'BUY')}")
    return "\n".join(lines)

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
    asyncio.create_task(_safe_refresh_calendar())
    asyncio.create_task(_continuous_scanner())
    asyncio.create_task(_trading_position_monitor())
    asyncio.create_task(ws_manager.price_feed(fetch_current_price, ALL_ASSETS[:6]))
    asyncio.create_task(_smart_money_background_refresh())


async def _smart_money_background_refresh():
    """Refresh Smart Money Intelligence data every 6 hours for all assets."""
    await asyncio.sleep(60)
    while True:
        try:
            print("Smart Money Intel: background refresh starting...")
            await refresh_all_smart_money(ALL_ASSETS)
            print("Smart Money Intel: refresh complete")
        except Exception as e:
            print(f"Smart Money Intel refresh error: {e}")
        await asyncio.sleep(6 * 3600)


# ─── Models ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    asset: str
    horizon: int
    api_key: str
    ds_key: Optional[str] = None
    worker_url: Optional[str] = None
    use_r1: Optional[bool] = True
    bot_mode: Optional[bool] = False  # True = "maintain position" framing, not "predict future"

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
                 ind: dict, decision: dict, mtf_data: dict = None) -> dict:
    """Prediction Quality Score 0-10 — measures signal confluence."""
    score = 0
    reasons = []
    d_dir = decision.get('decision', '')

    # 1. Agent agreement (0-3)
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
    score += agree
    if agree == 3: reasons.append('All agents agree')
    elif agree == 2: reasons.append('2/3 agents agree')

    # 2. Regime clarity (0-2)
    hmm = ind.get('hmm_probs', {})
    max_prob = max(hmm.values()) if hmm else 0
    if max_prob > 0.6:
        score += 2; reasons.append(f'Clear regime ({max_prob:.0%})')
    elif max_prob > 0.45:
        score += 1; reasons.append('Moderate regime')

    # 3. Hurst non-random (0-1)
    hurst = ind.get('hurst_exp', 0.5)
    if hurst > 0.6 or hurst < 0.4:
        score += 1; reasons.append('Hurst decisive')

    # 4. Low entropy (0-1)
    if ind.get('entropy_ratio', 0.5) < 0.4:
        score += 1; reasons.append('Low noise')

    # 5. Volume confirmation (0-1)
    if ind.get('vol_percentile', 50) > 60:
        score += 1; reasons.append('Volume confirms')

    # 6. Daily trend alignment (0-1)
    if mtf_data:
        if (mtf_data.get('daily_bull') and d_dir == 'BUY') or \
           (mtf_data.get('daily_bear') and d_dir == 'SELL'):
            score += 1; reasons.append('Daily aligned')

    # 7. RSI divergence support (0-1)
    if (d_dir == 'BUY' and ind.get('rsi_div_bull')) or \
       (d_dir == 'SELL' and ind.get('rsi_div_bear')):
        score += 1; reasons.append('RSI divergence confirms')

    return {'score': min(10, score), 'reasons': reasons, 'max': 10}


# ─── Adaptive Gate Thresholds (A5) ────────────────────────────────────────
_gate_stats = {'trades': 0, 'wins': 0}


def get_adaptive_floor():
    """Confidence floor adapts: if recent accuracy is high, lower the floor."""
    if _gate_stats['trades'] < 20:
        return 38
    win_rate = _gate_stats['wins'] / _gate_stats['trades']
    if win_rate > 0.6:
        return 32
    if win_rate > 0.5:
        return 35
    return 42


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
    from config import OPENAI_API_KEY, DEEPSEEK_API_KEY
    from telegram_bot import send_message

    await asyncio.sleep(10)  # let everything initialize
    engine = get_trading_engine()

    while True:
        if not _autotrader['enabled'] or not _autotrader['assets']:
            await asyncio.sleep(10)
            continue

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

                # Bot mode: AI's thinking shifts from "predict 4h ahead" to
                # "what's the right side to be on RIGHT NOW to maintain profit?"
                req = PredictRequest(
                    asset=asset_name, horizon=1,
                    api_key=api_key, ds_key=ds_key,
                    use_r1=True, bot_mode=True,
                )

                start_time = time.time()
                logs = []
                def slog(msg):
                    logs.append({"ts": int((time.time() - start_time) * 1000), "msg": msg})

                result = await _run_prediction(req, WORKER_URL, start_time, logs, slog)

                direction = result.get('decision', 'NO_TRADE')
                confidence = result.get('confidence', 0)
                pqs_score = result.get('pqs', {}).get('score', 0)
                price = result.get('ind', {}).get('cur', 0)
                stop_loss = result.get('quant', {}).get('stop_loss_pct', 2.0) or 2.0

                print(f"  {asset_name}: {direction} {confidence}% PQS:{pqs_score} @ {price}")

                # Force a direction even on NO_TRADE — AI shifts to "what should we do anyway?"
                if direction == 'NO_TRADE':
                    ind_data = result.get('ind', {})
                    bull_score = 0
                    bear_score = 0
                    if ind_data.get('rsi14', 50) < 45: bull_score += 1
                    if ind_data.get('rsi14', 50) > 55: bear_score += 1
                    if ind_data.get('macd_hist', 0) > 0: bull_score += 1
                    else: bear_score += 1
                    if ind_data.get('supertrend_bull'): bull_score += 1
                    else: bear_score += 1
                    if result.get('quant', {}).get('direction') == 'BUY': bull_score += 2
                    elif result.get('quant', {}).get('direction') == 'SELL': bear_score += 2
                    mc_prob = result.get('monte_carlo', {}).get('prob_up', 0.5)
                    if mc_prob > 0.55: bull_score += 1
                    elif mc_prob < 0.45: bear_score += 1

                    if bull_score > bear_score:
                        direction = 'BUY'
                        confidence = max(confidence, 42)
                    elif bear_score > bull_score:
                        direction = 'SELL'
                        confidence = max(confidence, 42)
                    else:
                        direction = 'BUY' if mc_prob >= 0.5 else 'SELL'
                        confidence = max(confidence, 40)
                    pqs_score = max(pqs_score, 3)
                    print(f"  {asset_name}: FORCED {direction} (bull={bull_score} bear={bear_score})")

                # Check if already holding this asset
                open_for_asset = [p for p in engine.positions.values()
                                  if p.status == 'open' and p.asset == asset_name]

                if open_for_asset:
                    current_pos = open_for_asset[0]
                    # Same direction — STAY, do nothing
                    if current_pos.direction == direction:
                        pnl_pct = ((price - current_pos.entry_price) / current_pos.entry_price * 100) if current_pos.direction == 'BUY' else ((current_pos.entry_price - price) / current_pos.entry_price * 100)
                        print(f"  {asset_name}: HOLD {current_pos.direction} (P&L: {pnl_pct:+.2f}%)")
                        cycle_summary.append(f"{asset_name}: HOLD {current_pos.direction} ({pnl_pct:+.1f}%)")
                        continue
                    else:
                        # FLIP — close old position, open opposite
                        close_result = await engine.close_position(current_pos.id, price, 'flip')
                        old_pnl = close_result.get('pnl', 0)
                        _autotrader['trades_closed'] += 1
                        record_lesson(asset_name, current_pos.direction, current_pos.entry_price, price, old_pnl, 'flip')
                        forensic_log.log_trade_flip(
                            asset_name, current_pos.direction, direction,
                            current_pos.entry_price, price, old_pnl, 0,
                            parent_id=cycle_id,
                        )
                        print(f"  {asset_name}: FLIP {current_pos.direction}→{direction} (closed P&L: ${old_pnl:+.2f})")
                        asyncio.create_task(send_message(
                            f"🔄 FLIP {asset_name}: {current_pos.direction}→{direction} | Closed P&L: ${old_pnl:+.2f} | New signal: {confidence}%"
                        ))

                # Safety checks
                open_count = len([p for p in engine.positions.values() if p.status == 'open'])
                if open_count >= 10:
                    cycle_summary.append(f"{asset_name}: max positions (10)")
                    continue
                if engine.daily_pnl <= -(engine._equity * 0.05):
                    cycle_summary.append(f"{asset_name}: daily loss limit")
                    continue

                # Position size — use custom size if set, otherwise Kelly/default
                custom_size = _autotrader.get('trade_size', 0)
                if custom_size > 0:
                    size = min(custom_size, engine._equity * 0.25)  # never more than 25% of equity
                else:
                    stats = _equity_tracker.get_stats()
                    n_trades = stats.get('n_trades', 0)
                    if n_trades >= 10:
                        win_rate = 0.5
                        rated = await get_predictions(asset_name, 100)
                        wins = sum(1 for p in rated if p.get('feedback') == 'correct')
                        total_rated = sum(1 for p in rated if p.get('feedback') in ('correct', 'wrong'))
                        if total_rated >= 5:
                            win_rate = wins / total_rated
                        size = engine.kelly_size(win_rate, 2.0, 1.0, engine._equity)
                    else:
                        size = min(200, engine._equity * 0.05)

                if size < 10:
                    size = min(100, engine._equity * 0.05)

                # Open the trade
                trade_result = await engine.open_position(
                    asset_name, direction, price, size,
                    stop_loss_pct=min(stop_loss, 3.0),
                    take_profit_pct=min(stop_loss * 2, 6.0),
                    trailing=1.5,
                )

                if trade_result.get('ok'):
                    _autotrader['trades_opened'] += 1
                    pos_data = trade_result.get('position', {})
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
    """Background task: check open trading positions every 30s."""
    await asyncio.sleep(15)
    engine = get_trading_engine()
    while True:
        try:
            if any(p.status == 'open' for p in engine.positions.values()):
                actions = await engine.check_all_positions(fetch_current_price)
                for act in actions:
                    a = act.get('asset', '?')
                    pnl = act.get('pnl', 0)
                    reason = act.get('reason', 'unknown')
                    print(f"Trading: auto-closed {a} ({reason}) P&L: {pnl}")
                    pos_data = act.get('position', {})
                    if pos_data:
                        record_lesson(a, pos_data.get('direction', '?'),
                                      pos_data.get('entry_price', 0),
                                      pos_data.get('exit_price', 0), pnl, reason)
                        forensic_log.log_trade_close(
                            a, pos_data.get('direction', '?'),
                            pos_data.get('entry_price', 0),
                            pos_data.get('exit_price', 0),
                            pnl, reason, was_correct=(pnl > 0),
                        )
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
    slog("🌐 Fetching macro, on-chain, sentiment, correlation, cluster, SMC, orderbook, whales in parallel...")
    asset_type = 'crypto' if is_crypto else 'macro' if req.asset in ['GC=F','CL=F','SI=F'] else 'stock'
    (macro_data, fg_data, onchain_data, sentiment_data, macro_context, correlation_data,
     cluster_data, smc_data, orderbook_data, whale_data, options_data) = await asyncio.gather(
        fetch_macro(), fetch_fear_greed(), fetch_onchain(req.asset),
        get_sentiment_snapshot(req.asset),
        get_macro_context(req.asset, req.horizon),
        get_correlation_summary(req.asset),
        _safe_assign_cluster(req.asset, ind),
        detect_smc(candles),
        get_orderbook_imbalance(req.asset),
        get_whale_activity(req.asset),
        get_options_sentiment(req.asset),
    )
    smart_money_data = analyze_smart_money(candles)

    # ── Smart Money Intelligence (politicians, insiders, funds, options, dark pool, top traders) ──
    slog("🕵️ Fetching Smart Money Intelligence...")
    try:
        smi_data = await get_smart_money_score(req.asset)
        smi_score = smi_data.get('score', 0)
        smi_dir = smi_data.get('direction', 'neutral')
        smi_top = smi_data.get('top_signal', '')
        slog(f"✓ Smart Money Intel: score={smi_score}/100 direction={smi_dir} | {smi_top[:80]}")
        if smi_data.get('high_quality_flags'):
            for flag in smi_data['high_quality_flags'][:2]:
                slog(f"  ★ {flag}")
    except Exception as smi_err:
        smi_data = {'score': 0, 'direction': 'neutral', 'data_completeness': 0, 'components': {}}
        slog(f"⚠ Smart Money Intel: {str(smi_err)[:60]}")

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

    # ── Agent 1: Quant (DeepSeek V4 primary, GPT-4o-mini fallback) ─────
    quant_model_label = 'DeepSeek V4' if req.ds_key else 'GPT-4o-mini'
    slog(f"📐 Agent 1 (Quant/{quant_model_label}) analyzing...")
    quant_prompt = build_quant_prompt(req.asset, ind, mc, req.horizon,
                                       cluster_data=cluster_data, correlation_data=correlation_data,
                                       bot_mode=getattr(req, 'bot_mode', False))
    lessons_ctx = get_lessons_context(req.asset)
    if lessons_ctx:
        quant_prompt += f"\n\n{lessons_ctx}\nUse these past trades to avoid repeating mistakes. If a direction kept losing, weigh the opposite more heavily."
    quant_result = await run_quant_agent(req.asset, ind, mc, req.horizon, quant_prompt, req.api_key, ds_key=req.ds_key or '')
    slog(f"✓ Quant[{quant_result.get('_quant_model','?')}]: {quant_result.get('direction')} {quant_result.get('confidence')}% — {quant_result.get('reasoning','')[:60]}")

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
    news_result = await run_news_agent(
        req.asset, ASSET_NAMES.get(req.asset, req.asset), asset_type,
        articles, macro_data, onchain_data, fg_data, {},
        req.horizon, req.api_key, req.ds_key, db_sentiment
    )
    slog(f"✓ News: {news_result.get('sentiment')} ({news_result.get('sentiment_score',0):+d}) — {news_result.get('reasoning','')[:60]}")

    # ── Agent 3: Decision (R1) ───────────────────────────────────────────
    use_r1 = bool(req.ds_key) and (req.use_r1 is not False)
    model_name = 'R1' if use_r1 else ('V4' if req.ds_key else 'GPT-4o')
    slog(f"🧠 Agent 3 ({model_name}) making final decision...")
    decision = await run_decision_agent(
        req.asset, ind, req.horizon, quant_result, news_result,
        mtf_data, mc, similar, req.ds_key or '', req.api_key, use_r1,
        ml_result=ml_result
    )
    model_used = decision.get('_model', 'unknown')
    slog(f"✓ Decision: {decision.get('decision')} {decision.get('confidence')}% [{model_used}]")
    if model_used == 'gpt-4o' and decision.get('_r1_error'):
        slog(f"⚠ R1 fallback reason: {decision['_r1_error']}")

    # ── ML OVERRIDE: When ML is confident, enforce its direction ─────────
    if ml_result.get('available'):
        ml_score = ml_result.get('score', 50)
        ml_dir = 'BUY' if ml_score > 58 else 'SELL' if ml_score < 42 else None
        ai_dir = decision.get('decision')

        if ml_dir and ai_dir != ml_dir:
            if ml_score > 62 or ml_score < 38:
                slog(f"🔄 ML OVERRIDE: ML says {ml_dir} ({ml_score:.1f}%) but AI said {ai_dir} — forcing ML direction")
                decision['_original_decision'] = ai_dir
                decision['decision'] = ml_dir
                decision['confidence'] = max(decision.get('confidence', 50), int(ml_score if ml_dir == 'BUY' else 100 - ml_score))
            elif ai_dir == 'NO_TRADE' and ml_dir:
                slog(f"🔄 ML activates trade: ML says {ml_dir} ({ml_score:.1f}%) — overriding NO_TRADE")
                decision['_original_decision'] = 'NO_TRADE'
                decision['decision'] = ml_dir
                decision['confidence'] = max(decision.get('confidence', 40), int(ml_score if ml_dir == 'BUY' else 100 - ml_score))

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
        old_conf = decision.get('confidence', 50)
        decision['confidence'] = max(0, old_conf - 10)
        slog(f"⚠ Cross-prediction contradiction: {contradictions} correlated assets disagree — confidence {old_conf}% → {decision['confidence']}%")

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

    # GATE 13: ADAPTIVE CONFIDENCE FLOOR (A5)
    conf_floor = get_adaptive_floor()
    if decision.get('decision') != 'NO_TRADE' and decision.get('confidence', 0) < conf_floor:
        decision['_original_decision'] = decision.get('decision')
        decision['decision'] = 'NO_TRADE'
        gate_reason = f"Confidence floor: {decision.get('confidence', 0)}% < {conf_floor}%"
        slog(f"⚠ Confidence floor gate: {decision.get('confidence', 0)}% (floor={conf_floor}%)")

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

    # SMART MONEY INTEL GATE — boost or reduce based on institutional consensus
    if decision.get('decision') != 'NO_TRADE' and smi_data.get('score', 0) > 0:
        smi_score = smi_data.get('score', 0)
        smi_dir = smi_data.get('direction', 'neutral')
        smi_completeness = smi_data.get('data_completeness', 0)
        if smi_completeness >= 30 and smi_score >= 60:
            decision_dir = decision['decision']
            smi_matches = (decision_dir == 'BUY' and smi_dir == 'bullish') or \
                          (decision_dir == 'SELL' and smi_dir == 'bearish')
            smi_contradicts = (decision_dir == 'BUY' and smi_dir == 'bearish') or \
                              (decision_dir == 'SELL' and smi_dir == 'bullish')
            old_conf = decision.get('confidence', 50)
            if smi_matches:
                boost = min(10, smi_score // 10)
                decision['confidence'] = min(90, old_conf + boost)
                slog(f"✓ Smart Money boost: score={smi_score} {smi_dir} confirms {decision_dir} → +{boost}% (conf {old_conf}→{decision['confidence']}%)")
            elif smi_contradicts:
                penalty = min(12, smi_score // 8)
                decision['confidence'] = max(40, old_conf - penalty)
                slog(f"⚠ Smart Money conflict: score={smi_score} {smi_dir} contradicts {decision_dir} → -{penalty}% (conf {old_conf}→{decision['confidence']}%)")

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
    pqs = compute_pqs(quant_result, news_result, ml_result, ind, decision, mtf_data)
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

    # ── FORENSIC LOG: capture EVERYTHING ─────────────────────────────────
    try:
        gates_extracted = []
        for entry in (logs or []):
            msg = entry.get("msg", "") if isinstance(entry, dict) else str(entry)
            if "Gate" in msg or "gate" in msg or "GATE" in msg or "capped" in msg or "→" in msg and "%" in msg:
                gates_extracted.append({
                    "ts_ms": entry.get("ts", 0) if isinstance(entry, dict) else 0,
                    "gate": msg[:200],
                    "action": "see_message",
                    "conf_before": None, "conf_after": None,
                    "reason": msg[:300],
                })
        forensic_log.log_prediction(
            asset=req.asset,
            request_data={
                "horizon": req.horizon,
                "bot_mode": getattr(req, 'bot_mode', False),
                "use_r1": req.use_r1,
                "quant_model": quant_result.get('_quant_model', '?'),
                "decision_model": decision.get('_model', '?'),
            },
            indicators=ind,
            monte_carlo=mc,
            news={
                "count": len(articles) if articles else 0,
                "headlines": [{"title": a.get('title', '')[:200],
                               "source": a.get('source', '?'),
                               "sentiment": a.get('sentiment', None),
                               "url": a.get('url', '')[:200]} for a in (articles or [])[:30]],
                "sentiment_score": news_result.get('sentiment_score'),
                "sources": list(set(a.get('source', '?') for a in (articles or []))),
            },
            macro=macro_context if isinstance(macro_context, dict) else {"raw": str(macro_context)[:500]},
            sentiment=sentiment_data if isinstance(sentiment_data, dict) else {},
            correlation=correlation_data if isinstance(correlation_data, dict) else {},
            cluster=cluster_data if isinstance(cluster_data, dict) else {},
            smc=smc_data if isinstance(smc_data, dict) else {},
            orderbook=orderbook_data if isinstance(orderbook_data, dict) else {},
            whales=whale_data if isinstance(whale_data, dict) else {},
            options=options_data if isinstance(options_data, dict) else {},
            smart_money=smart_money_data if isinstance(smart_money_data, dict) else {},
            ml_result=ml_result if isinstance(ml_result, dict) else {},
            quant_agent={
                "model": quant_result.get('_quant_model', '?'),
                "prompt": quant_prompt[:8000] if isinstance(quant_prompt, str) else "",
                "response": {k: v for k, v in quant_result.items() if not k.startswith('_')},
                "latency_ms": quant_result.get('_latency_ms', 0),
            },
            news_agent={
                "model": news_result.get('_model', '?'),
                "prompt": f"Asset: {req.asset} | {len(articles or [])} articles fed",
                "response": {k: v for k, v in news_result.items() if not k.startswith('_')},
                "latency_ms": news_result.get('_latency_ms', 0),
            },
            decision_agent={
                "model": decision.get('_model', '?'),
                "prompt": f"Quant: {quant_result.get('direction')}/{quant_result.get('confidence')}% | News: {news_result.get('sentiment')}/{news_result.get('sentiment_score')} | ML: {ml_result.get('score') if ml_result else 'N/A'}",
                "response": {k: v for k, v in decision.items() if not k.startswith('_')},
                "latency_ms": decision.get('_latency_ms', 0),
                "r1_error": decision.get('_r1_error'),
            },
            gates=gates_extracted,
            confidence_blend={
                "ai_raw": raw_ai_conf,
                "bayes": round(bayes_conf, 1),
                "ml": ml_result.get('score') if ml_result else None,
                "cluster": cluster_data.get('confidence') if isinstance(cluster_data, dict) else None,
                "weights": "AI 25% + Bayes 30% + ML 25% + Cluster 20%",
                "final": quant_result.get('confidence'),
            },
            pqs=pqs,
            final_decision={
                "direction": decision.get('decision'),
                "confidence": decision.get('confidence'),
                "reasoning": decision.get('insight', '') or decision.get('primary_reason', ''),
                "counter_argument": quant_result.get('counter_argument', ''),
                "gate_reason": gate_reason,
                "original_decision": decision.get('_original_decision'),
            },
            timing_ms=total_ms,
        )
    except Exception as fe:
        forensic_log.log_error(req.asset, "_run_prediction.forensic_log", str(fe))

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
        "smart_money_intel": {
            "score": smi_data.get('score', 0),
            "direction": smi_data.get('direction', 'neutral'),
            "confirmed": smi_data.get('confirmed', False),
            "data_completeness": smi_data.get('data_completeness', 0),
            "top_signal": smi_data.get('top_signal', ''),
            "high_quality_flags": smi_data.get('high_quality_flags', []),
            "components": {k: {"score": v.get("score", 0), "direction": v.get("direction", "neutral"),
                               "detail": v.get("detail", "")}
                           for k, v in smi_data.get('components', {}).items()},
            "macro_context": smi_data.get('macro_context', ''),
        },
        "monte_carlo": mc,
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
    }

    # Calibration (E3)
    try:
        cal = await calibrate_confidence(response['confidence'], req.asset, req.horizon, rated_preds)
        response['calibration'] = cal
        slog(f"✓ Calibration: raw={cal['raw']} calibrated={cal['calibrated']} ({cal['reliability']})")
    except Exception:
        response['calibration'] = None

    # A6: Record session prediction
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


# ─── Autonomous Trader Control ──────────────────────────────────────────

class AutotraderStartRequest(BaseModel):
    assets: list  # e.g. ["BTC", "ETH", "SOL"]
    interval_minutes: int = 60  # how often to trade (default 1hr)
    trade_size: float = 0  # 0 = auto (Kelly sizing), >0 = fixed USD per trade
    starting_equity: float = 10000  # paper trading starting balance


@app.post("/autotrader/start")
async def autotrader_start(req: AutotraderStartRequest):
    """Start the autonomous trader on specified assets."""
    valid_assets = [a for a in req.assets if a in ALL_ASSETS]
    if not valid_assets:
        return {"ok": False, "error": f"No valid assets. Choose from: {ALL_ASSETS}"}

    _autotrader['enabled'] = True
    _autotrader['assets'] = valid_assets
    _autotrader['interval_minutes'] = max(10, req.interval_minutes)
    _autotrader['trade_size'] = max(0, req.trade_size)
    _autotrader['starting_equity'] = max(100, req.starting_equity)
    _autotrader['status'] = 'starting'

    engine = get_trading_engine()
    engine._equity = _autotrader['starting_equity']

    # Start the loop if not already running
    asyncio.create_task(_autotrader_loop())

    from telegram_bot import send_message
    size_label = f"${_autotrader['trade_size']:.0f}/trade" if _autotrader['trade_size'] > 0 else "Auto (Kelly)"
    asyncio.create_task(send_message(
        f"🤖 AUTOTRADER STARTED\n"
        f"Assets: {', '.join(valid_assets)}\n"
        f"Interval: every {_autotrader['interval_minutes']} min\n"
        f"Mode: {'PAPER' if engine.paper_mode else 'LIVE'}\n"
        f"Starting equity: ${engine._equity:,.2f}\n"
        f"Trade size: {size_label}"
    ))

    return {
        "ok": True,
        "assets": valid_assets,
        "interval_minutes": _autotrader['interval_minutes'],
        "trade_size": _autotrader['trade_size'],
        "starting_equity": _autotrader['starting_equity'],
        "mode": "paper" if engine.paper_mode else "live",
    }


@app.post("/autotrader/stop")
async def autotrader_stop():
    """Stop the autonomous trader. Keeps positions open."""
    _autotrader['enabled'] = False
    _autotrader['status'] = 'stopped'

    engine = get_trading_engine()
    open_positions = engine.get_positions('open')

    from telegram_bot import send_message
    asyncio.create_task(send_message(
        f"🛑 AUTOTRADER STOPPED\n"
        f"Open positions: {len(open_positions)}\n"
        f"Equity: ${engine._equity:,.2f}\n"
        f"Total cycles: {_autotrader['total_cycles']}\n"
        f"Trades opened: {_autotrader['trades_opened']}"
    ))

    return {
        "ok": True,
        "open_positions": len(open_positions),
        "equity": engine._equity,
        "total_cycles": _autotrader['total_cycles'],
    }


@app.post("/autotrader/cashout")
async def autotrader_cashout():
    """Stop trading and close ALL open positions at current prices."""
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
                closed.append({
                    'asset': pos.asset,
                    'direction': pos.direction,
                    'entry': pos.entry_price,
                    'exit': price,
                    'pnl': pnl,
                })
        except Exception:
            continue

    from telegram_bot import send_message
    lines = [f"  {c['asset']}: {c['direction']} ${c['pnl']:+.2f}" for c in closed]
    asyncio.create_task(send_message(
        f"💰 AUTOTRADER CASHOUT\n"
        f"Closed {len(closed)} positions\n"
        + '\n'.join(lines) + '\n'
        f"Total P&L: ${total_pnl:+,.2f}\n"
        f"Final equity: ${engine._equity:,.2f}"
    ))

    # Should you keep going?
    stats = _equity_tracker.get_stats()
    recommendation = "KEEP GOING" if stats.get('total_return', 0) > 0 and stats.get('sharpe_ratio', 0) > 0.3 else "STOP — system is not profitable yet"

    return {
        "ok": True,
        "closed": closed,
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(engine._equity, 2),
        "total_return_pct": stats.get('total_return', 0),
        "sharpe_ratio": stats.get('sharpe_ratio', 0),
        "max_drawdown_pct": stats.get('max_drawdown', 0),
        "recommendation": recommendation,
    }


@app.get("/autotrader/status")
async def autotrader_status():
    """Get current autotrader status, equity, open positions, and recommendation."""
    engine = get_trading_engine()
    hb = engine.heartbeat()
    open_positions = engine.get_positions('open')
    trade_log = engine.get_trade_log(20)
    equity_stats = _equity_tracker.get_stats()

    # Calculate win rate from trade log
    wins = sum(1 for t in trade_log if t.get('pnl', 0) > 0)
    total = len(trade_log)
    win_rate = (wins / total * 100) if total > 0 else 0

    # Recommendation
    if not _autotrader['enabled']:
        rec = "Bot is stopped."
    elif hb['equity'] < 7000:
        rec = "WARNING: Down significantly. Consider stopping."
    elif hb['equity'] > 12000 and win_rate > 55:
        rec = "Profitable and consistent. Keep running."
    elif hb['equity'] > 10500:
        rec = "Slightly profitable. Keep running but monitor."
    elif total < 10:
        rec = "Still learning. Need more trades for reliable assessment."
    else:
        rec = "Mixed results. Monitor closely."

    return {
        "enabled": _autotrader['enabled'],
        "status": _autotrader['status'],
        "assets": _autotrader['assets'],
        "interval_minutes": _autotrader['interval_minutes'],
        "trade_size": _autotrader.get('trade_size', 0),
        "starting_equity": _autotrader.get('starting_equity', 10000),
        "total_cycles": _autotrader['total_cycles'],
        "trades_opened": _autotrader['trades_opened'],
        "last_cycle": _autotrader['last_cycle'],
        "seconds_until_next": max(0, (_autotrader['last_cycle'] + _autotrader['interval_minutes'] * 60) - int(time.time())) if _autotrader['enabled'] else 0,
        "heartbeat": hb,
        "open_positions": open_positions,
        "recent_trades": trade_log,
        "equity_stats": equity_stats,
        "win_rate": round(win_rate, 1),
        "recommendation": rec,
        "cycle_log": _autotrader['cycle_log'][-20:],
        "lessons": {asset: lessons[-5:] for asset, lessons in _trade_lessons.items()},
        "all_positions": engine.get_positions(),
    }


# ─── Smart Money Intel Endpoints ─────────────────────────────────────────

@app.get("/smart_money/{asset}")
async def smart_money_endpoint(asset: str):
    try:
        data = await get_smart_money_score(asset)
        return data
    except Exception as e:
        return {"score": 0, "error": str(e)[:200]}


@app.get("/smart_money_leaderboard")
async def smart_money_leaderboard():
    return {"leaderboard": get_source_leaderboard()}


# ─── Forensic Log Endpoints ─────────────────────────────────────────────

@app.get("/forensic/stats")
async def forensic_stats():
    """Quick stats about the forensic log buffer."""
    return forensic_log.stats()


@app.get("/forensic/log")
async def forensic_log_get(limit: int = 200, asset: str = None, type: str = None):
    """Recent forensic events as JSON (for dashboard display)."""
    events = forensic_log.get_events(limit=limit, asset_filter=asset, type_filter=type)
    return {"count": len(events), "events": events}


@app.get("/forensic/export")
async def forensic_export(asset: str = None):
    """
    Big fat human-readable log file. Click EXPORT in the dashboard to download.
    Paste into Claude/ChatGPT for post-mortem analysis.
    """
    text = forensic_log.export_text(asset_filter=asset)
    fname = f"ultramax_forensic_{int(time.time())}.txt"
    return PlainTextResponse(
        text,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/forensic/export.json")
async def forensic_export_json(asset: str = None):
    """Same data as JSON for programmatic analysis."""
    text = forensic_log.export_json(asset_filter=asset)
    fname = f"ultramax_forensic_{int(time.time())}.json"
    return PlainTextResponse(
        text,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.post("/forensic/clear")
async def forensic_clear():
    """Clear the forensic log buffer."""
    forensic_log.clear()
    return {"ok": True, "cleared": True}


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
