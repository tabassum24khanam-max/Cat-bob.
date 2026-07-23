"""
ULTRAMAX Earnings Calendar -- per-asset earnings dates feeding the same
upcoming_events pipeline the macro calendar already uses.

Why this exists: the macro calendar (macro_engine.py) only knows about
economy-wide events (FOMC, CPI, NFP). It has no idea AAPL reports earnings
Thursday after close. An earnings print is one of the few genuine, scheduled,
high-impact catalysts a stock has -- exactly the kind of "real surprise vs
priced-in" event the V3 news brain is built to weigh. This fills that gap.

Uses Finnhub's free-tier earnings calendar endpoint. Degrades to an empty
list with no error if FINNHUB_API_KEY is unset -- nothing breaks without it,
it just contributes nothing (same pattern as smart_money_intel.py).
"""
import time
from datetime import datetime, timedelta, timezone

import httpx

from config import FINNHUB_API_KEY, YAHOO_SYMBOLS


async def fetch_upcoming_earnings(days_ahead: int = 10) -> list:
    """Returns macro_event-shaped dicts for stocks with earnings in the window.
    Empty list (not an error) if no Finnhub key is configured."""
    if not FINNHUB_API_KEY:
        return []
    today = datetime.now(timezone.utc).date()
    frm = today.isoformat()
    to = (today + timedelta(days=days_ahead)).isoformat()
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get("https://finnhub.io/api/v1/calendar/earnings",
                            params={"from": frm, "to": to, "token": FINNHUB_API_KEY})
            if r.status_code != 200:
                return []
            rows = (r.json() or {}).get("earningsCalendar", [])
    except Exception:
        return []

    watchlist = set(YAHOO_SYMBOLS.keys())
    events = []
    for row in rows:
        sym = row.get("symbol")
        if sym not in watchlist:
            continue
        date_str = row.get("date")
        if not date_str:
            continue
        try:
            # Finnhub gives the report date; treat as end-of-day for scheduling
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
                hour=20, minute=0, tzinfo=timezone.utc)
        except Exception:
            continue
        hour = row.get("hour", "")  # 'bmo' before market open, 'amc' after close
        when = "before market open" if hour == "bmo" else "after market close" if hour == "amc" else "during session"
        eps_est = row.get("epsEstimate")
        rev_est = row.get("revenueEstimate")
        events.append({
            "event_type": f"EARNINGS_{sym}",
            "event_ts": int(dt.timestamp()),
            "description": (f"{sym} earnings report ({when})"
                            + (f", EPS est {eps_est}" if eps_est is not None else "")
                            + (f", rev est ${rev_est/1e9:.1f}B" if rev_est else "")),
            "impact_level": "high",
            "currency": None,
            "historical_btc_reaction": None,
            "historical_spy_reaction": None,
            "historical_gold_reaction": None,
        })
    return events


async def refresh_earnings_calendar():
    """Fetch and persist upcoming earnings into the shared macro_events table
    so they flow through the existing upcoming_events -> risk_evidence pipeline
    with zero changes needed elsewhere."""
    from database import save_macro_event
    events = await fetch_upcoming_earnings()
    saved = 0
    for e in events:
        try:
            await save_macro_event(e)
            saved += 1
        except Exception:
            continue
    return saved
