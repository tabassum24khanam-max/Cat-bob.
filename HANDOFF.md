# ULTRAMAX — PROJECT HANDOFF PROMPT

> **Paste this entire document into a fresh Claude/Opus session to continue the project with zero context loss.**
> It contains the full system architecture, the complete history of what was built and decided, the current state, and the roadmap. Treat it as the single source of truth.

---

## 0. WHO YOU ARE (instructions to the new assistant)

You are taking over an in-flight project called **ULTRAMAX** — an autonomous multi-agent crypto/stock trading bot with a web dashboard, deployed on Railway. The owner is non-academic about code but extremely sharp about the *product* and the *trading logic*; talk to them in plain language, show them results, and **never** silently change trading behavior. They run the bot in **paper mode** today and intend to move to **real money via webhook**. Be honest about risk — but their goal is to ship and to go public, so help them do that safely, not to talk them out of it.

Hard rules:
- **Don't fabricate results.** When you analyze forensic logs or P&L, compute the real numbers and show your work.
- **Paper trading is the default and the safety net.** Never flip to live trading without an explicit instruction.
- **Develop on a feature branch, then merge to `main`** (Railway auto-deploys from `main`). Commit with clear messages, push, and tell the owner what deployed.
- The repo is `tabassum24khanam-max/Cat-bob.` and the working branch has been `claude/review-ultramax-system-oldim`.

---

## 1. WHAT ULTRAMAX IS (one paragraph)

ULTRAMAX is a FastAPI backend + static HTML/JS frontend that runs a **3-agent prediction pipeline** (Quant → News → Decision) blended with ML, Bayesian, and cluster signals, gated by a consolidated risk system, and scored with a 0–10 "PQS" confluence score. On top of prediction sits an **autonomous trader loop** that opens/manages/flips paper (or live) positions on a fixed interval across ~9–18 assets. Every decision is written to a **forensic log** that can be exported and pasted into an LLM for post-mortem analysis. It's deployed as a Docker container on Railway.

---

## 2. REPOSITORY MAP

```
/Cat-bob.
├── Dockerfile                 # python:3.11-slim, runs uvicorn on :8000
├── railway.json               # Railway build (Dockerfile) + restart policy
├── requirements-deploy.txt    # Production deps (NO torch/transformers — saves RAM)
├── start.sh                   # Local dev: venv + uvicorn --reload on :8000
├── start_data_agent.sh        # Optional background data collector
├── ultramax-worker.js         # OPTIONAL Cloudflare Worker (Yahoo/macro/news proxy)
├── README.md / README-1.md    # README-1 is the real docs
├── docs/
│   └── TRADINGVIEW_INTEGRATION_PLAN.md   # The webhook/live-trading plan (READ THIS)
├── backend/
│   ├── main.py                # ★ CORE: FastAPI app, all ~40+ endpoints, autotrader loop, gate logic
│   ├── trading_engine.py      # ★ Position dataclass, TradingEngine, paper/live, Alpaca/Binance, webhook handler
│   ├── config.py              # Env vars / API keys / asset lists / get_asset_type()
│   ├── database.py            # Async SQLite (predictions, accuracy, news, macro, clusters)
│   ├── indicators.py          # 30+ TA indicators
│   ├── data_fetcher.py        # Candles/macro/price fetchers
│   ├── ml_engine.py           # XGBoost + RandomForest ensemble
│   ├── forensic_log.py        # ★ In-memory ring buffer (10k events) → export text/JSON
│   ├── equity_tracker.py      # P&L curve
│   ├── agents/
│   │   ├── quant_agent.py     # Technical-analysis LLM agent
│   │   ├── news_agent.py      # News/sentiment LLM agent
│   │   └── decision_agent.py  # Synthesis agent (R1 / GPT-4o / local Ollama)
│   ├── macro_engine.py, correlation_engine.py, cluster_engine.py,
│   ├── smc_engine.py, orderbook.py, whale_monitor.py, options_flow.py,
│   ├── smart_money*.py, calibration.py, alert_engine.py, telegram_bot.py,
│   └── (V5 stack) funding_oi.py, liquidations.py, order_flow.py,
│       volume_profile.py, tick_engine.py, rl_lite.py, regime_strategies.py,
│       pre_event.py, disagreement_signal.py, walkforward.py, feature_pruner.py
└── frontend/
    ├── index.html             # "Predict Mode" dashboard
    ├── bot.html               # Autonomous trader UI
    └── app.js                 # ~1500 lines of shared logic
```

---

## 3. HOW IT'S DEPLOYED

- **Platform:** Railway, building from the root `Dockerfile`.
- **Container:** `python:3.11-slim`, installs `requirements-deploy.txt`, copies `backend/` + `frontend/`, exposes **port 8000**, runs:
  `cd backend && python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}`
- **Auto-deploy:** Railway redeploys on every push to **`main`**. (Workflow used this session: develop on `claude/review-ultramax-system-oldim` → merge to `main` → push → Railway rebuilds.)
- **DB:** SQLite at `/app/backend/data/ultramax.db`.
- **Public URL:** a `*.up.railway.app` domain (the screenshots show `…-09ea.up.railway.app`).
- **Frontend** is served by the backend; `bot.html` is the autonomous-trading page.

### Environment variables (set in Railway)
| Var | Purpose |
|---|---|
| `OPENAI_API_KEY` | GPT-4o / 4o-mini |
| `DEEPSEEK_API_KEY` | DeepSeek V4 + R1 reasoning |
| `FRED_API_KEY` | Macro data (optional) |
| `FINNHUB_API_KEY` | News/insider (optional) |
| `ALPACA_KEY` / `ALPACA_SECRET` | Live **stock** trading (optional) |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Trade notifications (optional) |
| `WORKER_URL` | Cloudflare Worker proxy for candles (optional) |
| `WEBHOOK_SECRET` | **Recommended before going live** — not yet enforced |
| `TRADING_LIVE_ENABLED` | Defaults false; flip to enable live mode |

---

## 4. THE TRADING LOGIC (read carefully — this is the heart of it)

### 4.1 Prediction pipeline (`POST /predict`)
Fetch candles → 30+ indicators → Monte Carlo (1000 paths) → historical similarity search → parallel fetch (macro, sentiment, correlation, cluster, SMC, orderbook, whale, options, funding/OI, liquidations, order-flow, volume-profile, tick) → multi-timeframe (1H/4H/Daily) → ML ensemble → **Quant agent** → **News agent** → **4-way confidence blend** (AI 15–55% + Bayesian 20–45% + ML 45–55% + Cluster 20–30%, regime-adaptive) → **Decision agent** → consolidated gate (**max −25% penalty / +15% boost**, single capped adjustment) → kill-gates (force NO_TRADE) → adaptive confidence floor (45–52%) → **PQS score (0–10)** → calibration → response.

### 4.2 Autonomous loop (`_autotrader_loop` in main.py)
Runs every `interval_minutes`. For each asset:
1. Auto-close positions hitting SL/TP/trailing.
2. Run `/predict` in `bot_mode`.
3. **Force-trade:** if gated to NO_TRADE, fall back to original/quant/consensus direction (confidence floored 50–60%).
4. **Size** = base (Kelly if ≥10 trades, else ~2% equity) × PQS-scale × asset-accuracy-factor × ADX-scale, capped at 25% equity.
5. **Cash-out cycle states** (this is the key behavior the owner cares about):
   - **Profit (PnL > 0):** CASHOUT + REOPEN in fresh direction (logged as `trade_flip`).
   - **Small loss (−hard_stop < PnL ≤ 0):** HOLD until recovery — does not realize the loss this cycle.
   - **Loss beyond hard stop (PnL ≤ −hard_stop_pct, default 5%):** HARD STOP (realize loss) + reopen.
6. New positions: SL = 1.5×ATR (floor 1%, cap 3%), TP = 3× the SL (≈3:1 R:R), trailing = 1.2×ATR.
7. Record a "lesson" per close → updates per-asset accuracy → feeds sizing.

### 4.3 ⚠️ The win-rate caveat (must understand to read the dashboard honestly)
Because of the "small loss → hold until recovery" rule, **the live win-rate % overstates performance while positions are open** — losers sit open and unrealized, winners get flipped and realized. The honest metric is **equity after a full CASHOUT ALL** (which realizes everything). The owner already knows this and tests by comparing *before* vs *after* cashout. Always evaluate intervals on **post-cashout net equity**, not the headline win rate.

### 4.4 Safety controls (`trading_engine.py`)
`MAX_POSITIONS=10`, `MAX_DAILY_LOSS_PCT=5`, `MAX_POSITION_SIZE_PCT=20`, `MIN_CONFIDENCE=55`, `MIN_PQS=5`. Paper mode default. Live mode uses Alpaca (stocks, wired) / Binance (crypto, stubbed — needs HMAC signing + keys).

---

## 5. KEY ENDPOINTS (quick reference)

**Prediction/data:** `POST /predict`, `GET /candles`, `GET /price`, `GET /macro`, `GET /alerts`, `GET /api/features`
**History/ML:** `GET/POST /history*`, `POST /ml/retrain`, `POST /ml/backfill`, `POST /ml/sync-and-retrain`
**Manual trades:** `POST /trade/open`, `POST /trade/close`, `GET /trade/positions`, `GET /trade/heartbeat`, `GET /trade/log`, `POST /trade/paper`, **`POST /trade/webhook`** (TradingView)
**Autotrader:** `POST /autotrader/start`, `/stop`, `/cashout`, **`/reset`**, `GET /autotrader/status`
**Forensic:** `GET /forensic/stats`, `/log`, `/export`, `/export.json`, `POST /forensic/clear`
**V5 intel:** `GET /v5/{funding-oi|liquidations|order-flow|volume-profile|tick-structure|rl-report|regime-strategy|intelligence-summary}/{asset}`

---

## 6. FRONTEND BEHAVIOR

- **`index.html` (Predict Mode):** asset tabs (CRYPTO/STOCKS/MACRO), candlestick chart, horizon buttons, PREDICT button → result panel (decision, confidence, prob bars, agent scores, metric grid, PQS, reasoning, gate reason). Buttons: HISTORY, ALERTS, FEATURES, BOT, EXPORT LOG, ⚙ KEYS, **CASHOUT**, **RESET ALL**.
- **`bot.html` (Autonomous):** stat bar (EQUITY / DAY P&L / POSITIONS / WIN RATE / CYCLES), asset multi-select, interval (30m/1H/2H), inputs (starting equity, trade size, hard stop %), and the button row: **START BOT, CASHOUT ALL, RESET ALL, 📥 EXPORT LOG**. Tabs: POSITIONS / TRADES / LEARNING / SYSTEM LOG. Polls `/autotrader/status` every 10s.
- **Settings live in `localStorage`** (`um_backend`, `um_key`, `um_dskey`, `um_at_*`, `um_features`, `um_history_v3`, etc). Clearing the browser wipes them.

---

## 7. SESSION HISTORY — WHAT WE DID & DECIDED (chronological)

1. **Permissions setup.** Owner wanted to stop clicking "allow" on every action. Created `.claude/settings.json` allowing all Bash/Edit/Write/Read/Agent/GitHub-MCP tools. Committed + pushed. (An earlier accidental commit of a different settings file was reverted first.)

2. **Interval forensic analysis (30m vs 1h vs 2h).** Owner ran the bot on $100,000,000 paper equity, $10k/trade, across 9 assets, on a 30-minute interval, for **under 24 hours**, and compared intervals via screenshots taken **before and after CASHOUT ALL**:
   - **30m:** 95% win rate before cashout → **85% after**, net **≈ +$203**.
   - **2h:** 65% before → **20% after**, net **≈ +$8**.
   - **Conclusion: the 30-minute interval is the clear winner.** It holds an ~85% win rate even after all losers are realized, and earns ≈ **$203/day** at $10k/trade because it cycles ~9 assets many times per day. The 2h interval's edge collapses post-cashout (positions sit underwater).
   - I initially mis-read the forensic export as a "100% win-rate illusion" — that was **wrong**; the export was a mid-run snapshot taken *before* the final cashout, so it only contained the realized winners + still-open positions. The owner corrected this. **Lesson for the next assistant: the forensic export is a point-in-time snapshot; reconcile it with the post-cashout equity, and don't assume losers are never realized.**

3. **Built the RESET ALL feature.** Added `POST /autotrader/reset` in `backend/main.py` — wipes: autotrader counters/cycle_log/status, all engine positions, daily P&L, trade log, equity back to starting equity, `_trade_lessons`, `_asset_accuracy`, `_session_predictions`, `_ml_cache`, `_gate_stats`, equity-tracker curve, and the forensic log. Added a red **RESET ALL** button (with confirm dialog, also clears `um_history_v3`) to `bot.html` and `index.html`, plus `bReset()` / `resetAutotrader()` in `bot.html`/`app.js`. Merged to `main`, deployed.

4. **Fixed button visibility.** On mobile the four buttons crammed into one flex row hid RESET ALL behind EXPORT LOG. Moved CASHOUT ALL / RESET ALL / EXPORT LOG onto their **own row** below START BOT. Merged to `main`, deployed.

**Current state:** Bot runs clean in paper mode. RESET ALL works and is visible. 30-minute interval is the chosen strategy. Ready to (a) validate over more days, and (b) plan the path to real-money trading via webhook + going public.

---

## 8. ROADMAP / FUTURE GOALS

### A. Validate the edge (before risking real money)
- Run 30m interval for **2–4 weeks** of paper across multiple market regimes (trending, choppy, a volatility spike).
- Track **post-cashout net equity per day**, max drawdown, and what happens to "held" losers in a sustained adverse trend (the hold-until-recovery rule is the main blow-up risk — stress-test it).
- Consider making the dashboard show **equity *including* open-position mark-to-market** and a **true win rate** that counts open losers, so the headline number stops being optimistic.

### B. Go live with real money via webhook
The plan lives in `docs/TRADINGVIEW_INTEGRATION_PLAN.md`. Summary:
- **Mode 1 (incoming):** TradingView Pine alert → `POST /trade/webhook` → `engine.handle_webhook()` opens a position. Already implemented. Requires TradingView Pro+ (free has no webhooks).
- **Before going live:**
  1. Implement and **enforce `WEBHOOK_SECRET`** (signature/token check on `/trade/webhook`) — currently accepted but not validated.
  2. Add `stop_loss` / `take_profit` to the webhook payload.
  3. Rate-limit `/trade/webhook`; optionally whitelist TradingView IPs.
  4. Wire a real broker for execution: Alpaca for stocks is wired; for crypto either finish the **Binance HMAC-signed order** path in `trading_engine.py::_binance_order()` or use a **TradersPost/3Commas** bridge (~$10/mo).
  5. Flip `TRADING_LIVE_ENABLED=true` / `paper_mode=False` **only after** paper validation, and keep `MAX_DAILY_LOSS_PCT` + the CASHOUT ALL kill-switch as guards.
  6. Turn on Telegram notifications for every live fill.
- Start live with **tiny size** (e.g., $50–100/trade) and scale only after the live win rate matches paper.

### C. Make it public
- Decide the model: hosted multi-user SaaS vs open-source self-host. Today it's effectively single-tenant with `localStorage`-stored keys.
- If multi-user: move secrets server-side, add auth, per-user isolation of positions/equity/forensics, and rate limiting. **Never** trade real money on behalf of users without explicit, per-user authorization and clear risk disclosure.
- Write honest docs about the win-rate caveat and the hold-until-recovery risk so users aren't misled.

---

## 9. FIRST THINGS TO DO WHEN YOU TAKE OVER
1. `git fetch origin && git checkout claude/review-ultramax-system-oldim` (or branch fresh from `main`).
2. Read `backend/main.py` (autotrader loop + gates), `backend/trading_engine.py`, and `docs/TRADINGVIEW_INTEGRATION_PLAN.md`.
3. Ask the owner whether the immediate goal is **(a) longer paper validation**, **(b) wiring the live webhook safely**, or **(c) prepping for public release** — then proceed on that track.
4. Keep developing on a feature branch, merge to `main` to deploy, and report what went live.

---

*End of handoff. This document reflects the project as of 2026-06-02.*
