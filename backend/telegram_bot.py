"""
ULTRAMAX Telegram Bot — Alert & prediction notifications
Uses HTTP API directly (no extra dependency needed)
"""
import asyncio
import httpx
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

_BASE = "https://api.telegram.org/bot"


def _configured() -> bool:
    return bool(TELEGRAM_BOT_TOKEN) and bool(TELEGRAM_CHAT_ID)


async def send_message(text: str, parse_mode: str = "HTML") -> bool:
    if not _configured():
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{_BASE}{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                }
            )
            return resp.status_code == 200
    except Exception:
        return False


async def send_prediction(result: dict, asset: str, horizon: int):
    dec = result.get('decision', 'NO_TRADE')
    conf = result.get('confidence', 0)
    entry = result.get('ind', {}).get('cur', 0)
    target = result.get('price_target')
    ml = result.get('ml', {})
    pqs = result.get('pqs', {})

    emoji = {"BUY": "\U0001f7e2", "SELL": "\U0001f534"}.get(dec, "\U0001f7e1")

    lines = [
        f"{emoji} <b>ULTRAMAX {dec}</b> — {asset} {horizon}H",
        f"Confidence: <b>{conf}%</b>",
        f"Entry: <code>{entry:.4f}</code>",
    ]
    if target:
        pct = (target - entry) / entry * 100 if entry else 0
        lines.append(f"Target: <code>{target:.4f}</code> ({pct:+.2f}%)")
    if ml.get('available'):
        lines.append(f"ML: {ml.get('score', 50):.0f}% | Agree: {'Yes' if ml.get('agreement') else 'No'}")
    if pqs:
        lines.append(f"PQS: {pqs.get('score', 0)}/10")
    reason = result.get('primary_reason', '')
    if reason:
        lines.append(f"\n{reason}")

    await send_message("\n".join(lines))


async def send_alert(alert: dict):
    emoji = {"BUY": "\U0001f7e2", "SELL": "\U0001f534"}.get(alert.get('direction'), "\U0001f7e1")
    text = (
        f"{emoji} <b>ALERT: {alert['asset']}</b> — {alert['direction']} ({alert['score']}/10)\n"
        f"Price: <code>{alert.get('price', 0):.4f}</code> | Regime: {alert.get('regime', '?')}\n"
        f"Signals: {' · '.join(alert.get('signals', []))}"
    )
    await send_message(text)


async def send_scanner_summary(alerts: list):
    if not alerts:
        return
    lines = [f"\U0001f4e1 <b>Scanner: {len(alerts)} alert(s)</b>"]
    for a in alerts[:5]:
        emoji = {"BUY": "\U0001f7e2", "SELL": "\U0001f534"}.get(a.get('direction'), "\U0001f7e1")
        lines.append(f"{emoji} {a['asset']} {a['direction']} ({a['score']}/10) @ {a.get('price', 0):.4f}")
    await send_message("\n".join(lines))
