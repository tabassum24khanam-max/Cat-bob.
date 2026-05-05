# TradingView Integration Plan

How to wire ULTRAMAX's autonomous bot into TradingView so that TradingView alerts
become real (paper or live) trades on your bot — and the bot can execute on a
real broker through TradingView's broker integrations.

---

## Goal
Run your bot's brain (DeepSeek V4 + R1 + ML ensemble + 15 gates) inside ULTRAMAX,
but use TradingView for charting, alerting, and (optionally) trade execution.

Three modes, in increasing complexity:

1. **TV Alerts → ULTRAMAX (already plumbed)** — A Pine Script strategy on TradingView fires alerts; ULTRAMAX receives them and trades.
2. **ULTRAMAX → TV Webhook → Broker** — ULTRAMAX decides; TradingView's broker integration executes (Alpaca, OANDA, Tradovate, etc.).
3. **Pine Script that mirrors ULTRAMAX's brain** — A custom Pine indicator that reads the same data and shows the same signals on the chart, with two-way sync.

---

## Mode 1 — TV → ULTRAMAX (incoming alerts) ✅ already exists

Endpoint: `POST /trade/webhook` in `backend/main.py` → `engine.handle_webhook()` in `backend/trading_engine.py`.

### Steps

1. **Pine Script strategy** on TradingView:
   ```pinescript
   //@version=5
   strategy("ULTRAMAX Bridge", overlay=true)
   fastEMA = ta.ema(close, 12)
   slowEMA = ta.ema(close, 26)
   buy  = ta.crossover(fastEMA, slowEMA)
   sell = ta.crossunder(fastEMA, slowEMA)
   if buy
       strategy.entry("Long", strategy.long, alert_message='{"ticker":"{{ticker}}","action":"BUY","price":{{close}}}')
   if sell
       strategy.entry("Short", strategy.short, alert_message='{"ticker":"{{ticker}}","action":"SELL","price":{{close}}}')
   ```

2. **Create alert** on TradingView (right-click chart → Add Alert):
   - Condition: "ULTRAMAX Bridge → Any alert() function call"
   - Webhook URL: `https://cat-bob-production.up.railway.app/trade/webhook`
   - Message: `{{strategy.order.alert_message}}`

3. **TradingView Pro+ required** — webhooks are not on the free plan.

4. **Auth (recommended, not yet built):** Add `WEBHOOK_SECRET` env var on Railway.
   Have Pine Script include `"secret":"abc123"` in the JSON; have `/trade/webhook`
   reject any payload missing/wrong. ~10 lines of code.

### What ULTRAMAX does on receipt
- Parses `{ticker, action, price}`.
- Runs Kelly sizing against current equity.
- Calls `engine.open_position()` — paper or live based on `paper_mode`.
- Returns 200/JSON to TV.

### Limits
- TV alerts are one-shot fire-and-forget. They won't tell ULTRAMAX to close.
- Solution: send `{"action":"CLOSE","ticker":"..."}` from a separate alert when your strategy exits.

---

## Mode 2 — ULTRAMAX → TV Webhook → Broker (outgoing)

Use case: you trust the bot's brain but want a real broker (Alpaca for stocks,
OANDA for forex) and want TradingView to be the execution engine and audit trail.

### Architecture
```
ULTRAMAX bot decides BUY BTC 0.05
        │
        ▼
POST → TradingView webhook (or TV's "Trading Panel" via broker integration)
        │
        ▼
Broker (Alpaca/OANDA) fills the order
        │
        ▼
TradingView shows the position on chart, P&L updates
```

### How
TradingView itself doesn't accept incoming webhooks (only outgoing). So we need
either:

**Option 2A — Direct broker API.** Skip TradingView for execution. ULTRAMAX
already has Alpaca wired (`backend/trading_engine.py::_alpaca_order`). Add OANDA
and Binance live with HMAC signing.

**Option 2B — Bridge service (3Commas / TradersPost / PickMyTrade).** These
services accept JSON webhooks and convert them to broker orders, with TV
charting overlay.

  - Create account on TradersPost or 3Commas.
  - Connect your broker (Alpaca, IBKR, Tradovate).
  - Get a webhook URL like `https://traderspost.io/trading/webhook?key=xxx`.
  - In ULTRAMAX, on every `engine.open_position()` success, also send a POST to that URL with the trade payload.

  **Code change:** `backend/trading_engine.py::open_position()` — add 5 lines after
  successful position open:
  ```python
  if os.getenv("TRADERSPOST_WEBHOOK"):
      asyncio.create_task(httpx.AsyncClient().post(
          os.getenv("TRADERSPOST_WEBHOOK"),
          json={"ticker": asset, "action": direction.lower(), "quantity": qty}
      ))
  ```

---

## Mode 3 — Pine Script mirroring ULTRAMAX brain (advanced)

Use case: see ULTRAMAX's signals as colored arrows / boxes / labels directly on
the TradingView chart, while the bot keeps deciding.

### Architecture
```
TradingView chart loads
        │
        ▼ Pine `request.security()` with TV's URL fetcher
ULTRAMAX `/predict?asset=BTC&horizon=4` returns JSON
        │
        ▼
Pine renders BUY/SELL arrows, confidence labels, gate status, cluster info
```

### Reality check
Pine Script cannot do arbitrary HTTP calls. It can:
- Read other ticker symbols (`request.security`).
- Read TradingView Economic Calendar.
- Render shapes/labels.

It **cannot** call your Railway backend. So Mode 3 has two sub-options:

**3A — One-way: ULTRAMAX → TradingView via custom symbol feed.**
Publish a private `ULTRAMAX:BTC_SIGNAL` ticker via TradingView's
[UDF protocol](https://github.com/tradingview/charting_library/wiki/UDF). Each bar
emits signal=+1/-1/0 and confidence as the OHLC values. Pine reads it via
`request.security("ULTRAMAX:BTC_SIGNAL", "60", close)`. Heavy: needs a
Charting Library license + UDF server.

**3B — Two-way via Pine + Webhook + Bridge.** Mode 1 + Mode 2 chained: TV alert
fires ULTRAMAX, ULTRAMAX runs full brain, ULTRAMAX fires back through
TradersPost-style webhook which triggers a TV alert on a private symbol that
Pine reads. Triple round-trip. Not worth the latency for sub-hour decisions.

### Recommendation
Skip Mode 3. Keep TradingView for charting only, use ULTRAMAX's bot dashboard
(`/bot`) as the source of truth for signals.

---

## Recommended path

1. **Now (zero code change):** Test Mode 1 with paper trading. Set up the Pine
   strategy above, point alerts at `/trade/webhook`. Verify trades open in `/bot`.
2. **Soon (10 lines of code):** Add `WEBHOOK_SECRET` env var + signature check.
3. **When ready for live:** Mode 2A — fund Alpaca account, set
   `ALPACA_KEY`/`ALPACA_SECRET` on Railway, flip `paper_mode=False` via `POST /trade/paper`.
4. **For multi-broker / forex / futures:** Mode 2B — TradersPost subscription
   (~$10/mo) acts as universal broker bridge, ULTRAMAX just POSTs JSON.

---

## Security checklist (before going live)
- [ ] `WEBHOOK_SECRET` env var; reject unsigned payloads.
- [ ] Whitelist source IPs (TradingView publishes IP ranges).
- [ ] Rate-limit `/trade/webhook` (5 req/sec per IP).
- [ ] Daily loss limit gate (already enforced: `MAX_DAILY_LOSS_PCT=5%`).
- [ ] Email/Telegram on every live trade (already wired via `telegram_bot.py`).
- [ ] Two-step live trading toggle: env var `TRADING_LIVE_ENABLED=false` by default; require manual flip.
- [ ] Manual kill-switch: keep the `CASHOUT ALL` button on `/bot` reachable.

---

## Files that change for each mode

| Mode | Files modified | Estimated effort |
|------|---------------|------------------|
| 1 | none (already built) | 0h |
| 1 + auth | `backend/main.py` (~10 lines), `backend/config.py` (1 line) | 0.5h |
| 2A live broker | `backend/trading_engine.py::_binance_order` (HMAC signing) | 2h |
| 2B TradersPost bridge | `backend/trading_engine.py::open_position` (+5 lines) | 0.5h |
| 3 Pine mirror | `backend/main.py` UDF endpoints, separate Pine script | 8-16h |

---

## Verification plan

For Mode 1:
1. Set `paper_mode=True` (default).
2. Create Pine strategy with EMA cross.
3. Manually fire a test alert from TV → check `/bot` shows new paper position.
4. Fire opposite alert → verify position flip works.
5. Run for 7 days on paper, compare bot equity curve vs. TV strategy backtest equity.
6. Only after equity curves match within 2% should you flip to live.
