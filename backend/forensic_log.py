"""
Forensic Log — captures every detail of every decision the bot makes.

Designed to produce a "big fat file" that can be pasted into an LLM
(Anthropic, OpenAI) for post-mortem analysis. Every event is structured
JSON and rendered as readable Markdown on export.

What gets logged (per prediction/cycle):
  - All ~30 indicators with values and interpretations
  - Monte Carlo paths (median, bull, bear, prob_up)
  - Macro context (VIX, DXY, fear&greed, on-chain)
  - News headlines fetched (title, source, sentiment per article)
  - Smart Money Concepts (order blocks, FVGs, BOS)
  - Orderbook imbalance, whale flow, options P/C
  - Cluster analysis, correlations
  - ML ensemble features + per-model predictions
  - Quant agent: full prompt + full response + model + latency
  - News agent: same
  - Decision agent (R1): same
  - Each of 15 gates that fired, with conf before/after
  - 4-way confidence blending (AI / Bayes / ML / Cluster)
  - PQS breakdown
  - Final decision + reasoning + counter-argument
  - Trade open/flip/close with prices and P&L
  - Outcome when known (was the decision correct?)
"""
import time
import json
from typing import Dict, List, Any, Optional
from collections import deque

# In-memory ring buffer — capped to avoid OOM on Railway
_MAX_EVENTS = 10000
_events: deque = deque(maxlen=_MAX_EVENTS)


def _now() -> int:
    return int(time.time() * 1000)


def log(event_type: str, asset: str, data: Dict[str, Any], parent_id: Optional[str] = None) -> str:
    """Record a forensic event. Returns event_id for chaining child events."""
    event_id = f"{event_type}_{_now()}_{asset}"
    event = {
        "id": event_id,
        "ts": _now(),
        "ts_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "type": event_type,
        "asset": asset,
        "parent_id": parent_id,
        "data": data,
    }
    _events.append(event)
    return event_id


def log_cycle_start(cycle: int, assets: List[str], interval_min: int,
                    equity: float, trade_size: float, paper: bool) -> str:
    return log("cycle_start", "*", {
        "cycle_number": cycle,
        "assets": assets,
        "interval_minutes": interval_min,
        "equity_before": equity,
        "trade_size_usd": trade_size,
        "mode": "paper" if paper else "live",
    })


def log_cycle_end(cycle: int, equity_after: float, summaries: List[str],
                   trades_opened: int, trades_closed: int, parent_id: str = None):
    return log("cycle_end", "*", {
        "cycle_number": cycle,
        "equity_after": equity_after,
        "summaries": summaries,
        "trades_opened_total": trades_opened,
        "trades_closed_total": trades_closed,
    }, parent_id=parent_id)


def log_prediction(asset: str, request_data: Dict, indicators: Dict,
                   monte_carlo: Dict, news: Dict, macro: Dict, sentiment: Dict,
                   correlation: Dict, cluster: Dict, smc: Dict, orderbook: Dict,
                   whales: Dict, options: Dict, smart_money: Dict, ml_result: Dict,
                   quant_agent: Dict, news_agent: Dict, decision_agent: Dict,
                   gates: List[Dict], confidence_blend: Dict, pqs: Dict,
                   final_decision: Dict, timing_ms: int, parent_id: str = None) -> str:
    """Log the complete record of a single prediction."""
    return log("prediction", asset, {
        "request": request_data,
        "price_at_decision": indicators.get("cur"),
        "previous_close": indicators.get("prev_close"),
        "price_change_24h_pct": indicators.get("change_24h_pct"),
        "indicators": indicators,
        "monte_carlo": monte_carlo,
        "news": {
            "article_count": news.get("count", 0),
            "headlines": news.get("headlines", [])[:30],
            "sources": news.get("sources", []),
            "sentiment_score": news.get("sentiment_score"),
        },
        "macro": macro,
        "sentiment_aggregate": sentiment,
        "correlation": correlation,
        "cluster": cluster,
        "smart_money_concepts": smc,
        "orderbook": orderbook,
        "whale_activity": whales,
        "options_flow": options,
        "smart_money_flow": smart_money,
        "ml_ensemble": ml_result,
        "agents": {
            "quant": quant_agent,
            "news": news_agent,
            "decision_r1": decision_agent,
        },
        "gates_triggered": gates,
        "confidence_blend": confidence_blend,
        "pqs": pqs,
        "final_decision": final_decision,
        "timing_ms": timing_ms,
    }, parent_id=parent_id)


def log_trade_open(asset: str, direction: str, price: float, size: float,
                    stop_loss: float, take_profit: float, trailing: float,
                    source: str, confidence: int = 0, pqs_score: int = 0,
                    parent_id: str = None):
    return log("trade_open", asset, {
        "direction": direction,
        "entry_price": price,
        "size_usd": size,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "trailing_pct": trailing,
        "source": source,  # "autotrader_cycle_N" or "webhook" or "manual"
        "confidence": confidence,
        "pqs_score": pqs_score,
    }, parent_id=parent_id)


def log_trade_flip(asset: str, old_direction: str, new_direction: str,
                    entry_price: float, exit_price: float, old_pnl: float,
                    new_size: float, parent_id: str = None):
    return log("trade_flip", asset, {
        "old_direction": old_direction,
        "new_direction": new_direction,
        "old_entry": entry_price,
        "flip_at_price": exit_price,
        "closed_pnl": old_pnl,
        "new_position_size": new_size,
        "lesson": f"{old_direction} {'WAS WRONG' if old_pnl < 0 else 'profitable'} — flipped to {new_direction}",
    }, parent_id=parent_id)


def log_trade_close(asset: str, direction: str, entry: float, exit_price: float,
                     pnl: float, reason: str, hold_seconds: int = 0,
                     was_correct: bool = False, parent_id: str = None):
    pnl_pct = ((exit_price - entry) / entry * 100) if direction == "BUY" else ((entry - exit_price) / entry * 100)
    return log("trade_close", asset, {
        "direction": direction,
        "entry_price": entry,
        "exit_price": exit_price,
        "pnl_usd": pnl,
        "pnl_pct": round(pnl_pct, 4),
        "exit_reason": reason,
        "hold_seconds": hold_seconds,
        "was_correct": was_correct,
        "lesson": f"{direction} was {'CORRECT' if pnl > 0 else 'WRONG'} ({pnl_pct:+.2f}%) — exit via {reason}",
    }, parent_id=parent_id)


def log_event_simple(event_type: str, asset: str, message: str, **extra):
    return log(event_type, asset, {"message": message, **extra})


def log_error(asset: str, where: str, error: str):
    return log("error", asset, {"location": where, "error": str(error)[:500]})


def get_events(limit: int = 1000, asset_filter: Optional[str] = None,
               type_filter: Optional[str] = None) -> List[Dict]:
    events = list(_events)
    if asset_filter:
        events = [e for e in events if e["asset"] == asset_filter or e["asset"] == "*"]
    if type_filter:
        events = [e for e in events if e["type"] == type_filter]
    return events[-limit:]


def export_text(asset_filter: Optional[str] = None, since_ts: Optional[int] = None) -> str:
    """Render the entire forensic log as a big readable Markdown file."""
    events = list(_events)
    if asset_filter:
        events = [e for e in events if e["asset"] == asset_filter or e["asset"] == "*"]
    if since_ts:
        events = [e for e in events if e["ts"] >= since_ts]

    lines = []
    lines.append("=" * 80)
    lines.append("ULTRAMAX FORENSIC LOG EXPORT")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"Total events: {len(events)}")
    if asset_filter:
        lines.append(f"Filtered by asset: {asset_filter}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("This file contains the complete decision history of the ULTRAMAX bot.")
    lines.append("Paste into Claude or GPT and ask: 'Why did the bot lose money on these trades?'")
    lines.append("or 'Which indicator was misleading?' for post-mortem analysis.")
    lines.append("")

    for ev in events:
        lines.append("")
        lines.append("─" * 80)
        lines.append(f"[{ev['ts_iso']}]  {ev['type'].upper()}  ({ev['asset']})")
        lines.append("─" * 80)
        lines.extend(_render_event(ev))

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF LOG")
    lines.append("=" * 80)
    return "\n".join(lines)


def _render_event(ev: Dict) -> List[str]:
    out = []
    et = ev["type"]
    d = ev.get("data", {})

    if et == "cycle_start":
        out.append(f"  Cycle #{d.get('cycle_number')} starting")
        out.append(f"  Assets to scan: {', '.join(d.get('assets', []))}")
        out.append(f"  Interval: every {d.get('interval_minutes')} minutes")
        out.append(f"  Equity before: ${d.get('equity_before', 0):,.2f}")
        out.append(f"  Trade size: ${d.get('trade_size_usd', 0)} (0=Kelly auto)")
        out.append(f"  Mode: {d.get('mode')}")

    elif et == "cycle_end":
        out.append(f"  Cycle #{d.get('cycle_number')} complete")
        out.append(f"  Equity after: ${d.get('equity_after', 0):,.2f}")
        out.append(f"  Per-asset summary: {' | '.join(d.get('summaries', []))}")

    elif et == "prediction":
        out.append(f"  ┌─ REQUEST ─")
        req = d.get("request", {})
        out.append(f"  │  Horizon: {req.get('horizon')}h")
        out.append(f"  │  Bot mode: {req.get('bot_mode', False)}")
        out.append(f"  │  Models: quant={req.get('quant_model','?')} decision={req.get('decision_model','?')}")
        out.append(f"  │  Price: ${d.get('price_at_decision', 0):,.4f}")
        out.append(f"  ")
        out.append(f"  ┌─ INDICATORS (all values at decision time) ─")
        ind = d.get("indicators", {})
        for k in sorted(ind.keys()):
            v = ind[k]
            if isinstance(v, (int, float)):
                out.append(f"  │  {k}: {v}")
            elif isinstance(v, dict):
                out.append(f"  │  {k}: {json.dumps(v)[:200]}")
            elif isinstance(v, list):
                out.append(f"  │  {k}: [{len(v)} items]")
            else:
                out.append(f"  │  {k}: {str(v)[:200]}")
        out.append(f"  ")

        mc = d.get("monte_carlo", {})
        if mc:
            out.append(f"  ┌─ MONTE CARLO (1000 paths) ─")
            for k, v in mc.items():
                out.append(f"  │  {k}: {v}")
            out.append(f"  ")

        news = d.get("news", {})
        if news.get("article_count"):
            out.append(f"  ┌─ NEWS ({news['article_count']} articles, sentiment={news.get('sentiment_score')}) ─")
            for h in news.get("headlines", [])[:15]:
                src = h.get("source", "?")
                title = h.get("title", "")[:120]
                sent = h.get("sentiment", "")
                out.append(f"  │  [{src}] {title}  → {sent}")
            out.append(f"  ")

        for section_key, label in [
            ("macro", "MACRO CONTEXT (VIX, DXY, fear&greed)"),
            ("sentiment_aggregate", "SENTIMENT AGGREGATE"),
            ("correlation", "CORRELATIONS"),
            ("cluster", "CLUSTER ANALYSIS"),
            ("smart_money_concepts", "SMART MONEY CONCEPTS"),
            ("orderbook", "ORDERBOOK"),
            ("whale_activity", "WHALE ACTIVITY"),
            ("options_flow", "OPTIONS FLOW"),
            ("smart_money_flow", "SMART MONEY FLOW"),
            ("ml_ensemble", "ML ENSEMBLE"),
        ]:
            sd = d.get(section_key, {}) or {}
            if sd:
                out.append(f"  ┌─ {label} ─")
                for k, v in sd.items():
                    if isinstance(v, (dict, list)):
                        out.append(f"  │  {k}: {json.dumps(v)[:300]}")
                    else:
                        out.append(f"  │  {k}: {v}")
                out.append(f"  ")

        agents = d.get("agents", {})
        for agent_key, label in [("quant", "QUANT AGENT"), ("news", "NEWS AGENT"), ("decision_r1", "DECISION AGENT (R1)")]:
            ag = agents.get(agent_key, {}) or {}
            if ag:
                out.append(f"  ┌─ {label} (model={ag.get('model','?')}, latency={ag.get('latency_ms','?')}ms) ─")
                if ag.get("prompt"):
                    out.append(f"  │  PROMPT:")
                    for line in str(ag.get("prompt", ""))[:3000].split("\n"):
                        out.append(f"  │    {line}")
                if ag.get("response"):
                    out.append(f"  │  RESPONSE:")
                    if isinstance(ag["response"], dict):
                        out.append(f"  │    {json.dumps(ag['response'], indent=2)[:2000]}")
                    else:
                        out.append(f"  │    {str(ag['response'])[:2000]}")
                out.append(f"  ")

        gates = d.get("gates_triggered", [])
        if gates:
            out.append(f"  ┌─ GATES TRIGGERED ({len(gates)}) ─")
            for g in gates:
                out.append(f"  │  Gate: {g.get('gate')}  Action: {g.get('action')}  Conf: {g.get('conf_before')}% → {g.get('conf_after')}%  Reason: {g.get('reason')}")
            out.append(f"  ")

        cb = d.get("confidence_blend", {})
        if cb:
            out.append(f"  ┌─ CONFIDENCE BLEND ─")
            out.append(f"  │  AI raw: {cb.get('ai_raw')}%")
            out.append(f"  │  Bayesian: {cb.get('bayes')}%")
            out.append(f"  │  ML calibrated: {cb.get('ml')}%")
            out.append(f"  │  Cluster: {cb.get('cluster')}%")
            out.append(f"  │  Weights: {cb.get('weights')}")
            out.append(f"  │  → BLENDED: {cb.get('final')}%")
            out.append(f"  ")

        pqs = d.get("pqs", {})
        if pqs:
            out.append(f"  ┌─ PQS (Prediction Quality Score) ─")
            out.append(f"  │  Score: {pqs.get('score')}/10")
            out.append(f"  │  Breakdown: {pqs.get('reasons', [])}")
            out.append(f"  ")

        fd = d.get("final_decision", {})
        if fd:
            out.append(f"  ┌─ FINAL DECISION ─")
            out.append(f"  │  Direction: {fd.get('direction')}")
            out.append(f"  │  Confidence: {fd.get('confidence')}%")
            out.append(f"  │  Reasoning: {fd.get('reasoning', '')[:400]}")
            out.append(f"  │  Counter-argument: {fd.get('counter_argument', '')[:300]}")
            out.append(f"  ")
        out.append(f"  Total time: {d.get('timing_ms', 0)}ms")

    elif et == "trade_open":
        out.append(f"  OPENED {d.get('direction')} @ ${d.get('entry_price', 0):,.4f}")
        out.append(f"  Size: ${d.get('size_usd', 0):.2f}")
        out.append(f"  Stop-loss: ${d.get('stop_loss', 0):,.4f}  Take-profit: ${d.get('take_profit', 0):,.4f}  Trailing: {d.get('trailing_pct', 0)}%")
        out.append(f"  Source: {d.get('source')}")
        out.append(f"  Confidence: {d.get('confidence')}%  PQS: {d.get('pqs_score')}/10")

    elif et == "trade_flip":
        out.append(f"  FLIP {d.get('old_direction')} → {d.get('new_direction')}")
        out.append(f"  Old entry: ${d.get('old_entry', 0):,.4f}  Flip at: ${d.get('flip_at_price', 0):,.4f}")
        out.append(f"  Closed P&L: ${d.get('closed_pnl', 0):+.2f}")
        out.append(f"  New size: ${d.get('new_position_size', 0):.2f}")
        out.append(f"  Lesson: {d.get('lesson')}")

    elif et == "trade_close":
        was = "✓ CORRECT" if d.get("was_correct") else "✗ WRONG"
        out.append(f"  CLOSED {d.get('direction')} ({was})")
        out.append(f"  Entry: ${d.get('entry_price', 0):,.4f}  Exit: ${d.get('exit_price', 0):,.4f}")
        out.append(f"  P&L: ${d.get('pnl_usd', 0):+.2f}  ({d.get('pnl_pct', 0):+.2f}%)")
        out.append(f"  Reason: {d.get('exit_reason')}  Held: {d.get('hold_seconds', 0)}s")
        out.append(f"  Lesson: {d.get('lesson')}")

    elif et == "error":
        out.append(f"  ERROR @ {d.get('location')}")
        out.append(f"  {d.get('error')}")

    else:
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                out.append(f"  {k}: {json.dumps(v)[:300]}")
            else:
                out.append(f"  {k}: {v}")
    return out


def export_json(asset_filter: Optional[str] = None) -> str:
    events = list(_events)
    if asset_filter:
        events = [e for e in events if e["asset"] == asset_filter or e["asset"] == "*"]
    return json.dumps(events, indent=2, default=str)


def clear():
    _events.clear()


def stats() -> Dict:
    by_type = {}
    by_asset = {}
    for e in _events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        by_asset[e["asset"]] = by_asset.get(e["asset"], 0) + 1
    return {
        "total_events": len(_events),
        "max_capacity": _MAX_EVENTS,
        "by_type": by_type,
        "by_asset": by_asset,
        "oldest_ts": _events[0]["ts"] if _events else None,
        "newest_ts": _events[-1]["ts"] if _events else None,
    }
