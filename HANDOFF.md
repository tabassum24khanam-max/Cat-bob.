# ULTRAMAX — PROJECT HANDOFF (v2.1, 2026-07-03)

> **Paste this entire document into a fresh Claude session to continue with zero context loss.**
> It contains the architecture, the completed forensic diagnosis, the agreed (NOT yet built) redesign, the owner's decisions on the previously open questions, and file/line pointers for the build. Treat it as the single source of truth. Supersedes the v1 and v2 handoffs.

---

## 0. WHO YOU ARE (instructions to the new assistant)

You are taking over **ULTRAMAX** — an autonomous multi-agent crypto/stock trading bot (FastAPI + static frontend) deployed on Railway, paper-trading today, aiming for real money via webhook later. The owner is non-technical about code but sharp about product and trading logic. Plain language. Show real numbers.

Hard rules:
- **Never fabricate results.** Compute real numbers from forensic logs; show your work.
- **Paper mode is the default.** Never enable live trading without explicit instruction.
- **Never silently change trading behavior.** If a change alters HOW the bot trades (vs. just displaying more), flag it loudly and get consent. (This rule was violated once with a 6h time-decay rule — owner kept it, but was rightly annoyed it wasn't flagged upfront.)
- **"Discussing" ≠ "build."** The owner explicitly says GO when they want code written. If they say "we are just discussing," do not edit files.
- Develop on a `claude/*` feature branch (current: `claude/ultramax-handoff-v2-2lmq5m`), merge to `main` to deploy (Railway auto-deploys main). Repo: `tabassum24khanam-max/Cat-bob.`

---

## 1. WHAT ULTRAMAX IS

FastAPI backend + HTML/JS frontend. A 3-agent pipeline (Quant → News → Decision) blended with ML/Bayesian/cluster signals, gated by a risk system, scored 0–10 ("PQS"). An autotrader loop re-runs the full pipeline per asset every interval (30m/1H/2H) and manages positions via "cash-out cycles." Every event goes to an in-memory forensic log exportable as text/JSON. Docker on Railway, SQLite, port 8000. `bot.html` is the autonomous UI; `index.html` is manual predict mode.

**Required env (Railway Variables):** at least one of `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` — with neither set, the autotrader silently skips every asset (this bit us once: 13 cycles, 0 trades, after the owner made a new Railway account). Optional: `FRED_API_KEY`, `FINNHUB_API_KEY`, `TELEGRAM_BOT_TOKEN`+`TELEGRAM_CHAT_ID`, `WORKER_URL` (has hardcoded default). Live-only (do NOT set yet): `ALPACA_KEY`/`SECRET`, `TRADING_LIVE_ENABLED`, `WEBHOOK_SECRET`.

---

## 2. WHAT IS DEPLOYED AND WORKING TODAY

- **Cash-out cycle autotrader** (mechanical version — see §4 for the redesign). Current live behavior per asset per interval (`main.py` ~659-729): profit → cash out + reopen in the fresh prediction's direction; loss within −5% → HOLD (waiting for recovery); loss beyond −5% (hard stop) → emergency close + reopen; loser held >6h → time-decay force-close + reopen.
- **Mark-to-market dashboard row** on bot.html (yellow): REAL EQUITY, UNREALIZED, IF CASHOUT, TRUE WIN%, W/L OPEN — exposes the gap between headline win rate (realized-only) and reality.
- `/autotrader/status` returns `mtm_equity`, `unrealized_pnl`, `cashout_impact`, `true_win_rate`, `open_winners/losers`, `position_health` (per-position: current price, unrealized %/USD, age, SL/TP distance, entry confidence).
- **CASHOUT WINNERS** button + `POST /autotrader/cashout-winners` — closes only profitable positions, keeps losers open, bot keeps running.
- **CASHOUT ALL** (`/autotrader/cashout` — closes everything, stops bot), **RESET ALL** (`/autotrader/reset` — wipes all state to zero), **EXPORT LOG** (forensic text+JSON).
- **6h time-decay**: losers held >6h are force-closed and reopened (in `_autotrader_loop`). ⚠ Conflicts with the owner's 2026-07-03 decision to hold small losses until −5% — see §4 Open Question.
- **Known cosmetic bug (diagnosed, unfixed):** W/L tile counts pnl ≥ 0 as "winning," TRUE WIN% counts only pnl > 0 — so market-closed stocks frozen at exactly 0.00% show 7W/0L next to a 22–28% true win rate. Also WIN RATE shows red "0%" instead of "—" when the trade log is empty.

---

## 3. THE FORENSIC DIAGNOSIS (completed — this is WHY the bot underperforms)

Owner exported forensic logs; analysis found almost every trade opens at **Confidence: 52% PQS: 1/10** — the signature of the force-trade fallback. The AI pipeline works but its signal is destroyed before the trade:

```
Quant agent: "BUY 75%"
  → 4-way blend (AI weight only 15–55%) dilutes to ~63%     [main.py ~1173-1202]
  → ~26 penalty gates fire on NORMAL conditions, cap −25    [main.py ~1308-1551]
    (hurst random −5, entropy −5, OBV −5, CMF −5, daily counter-trend −8…)
  → 63 − 25 = 38%
  → confidence floor (52 default) kills it → NO_TRADE       [main.py ~507-517, 1582-1588]
  → force-trade fallback resurrects a direction at hardcoded 52%, PQS 1  [main.py ~628-653]
```

Result: the Ferrari engine idles while a coin flip does the trading.

More discoveries (all verified against code 2026-07-03):
- **The Decision Agent is handcuffed** (`agents/decision_agent.py` ~82-121): its prompt says "YOUR PRIMARY JOB: Follow the ML model's prediction… If ML says BUY with >55%, your decision MUST be BUY. You may only adjust confidence, not direction." The "judge" the owner designed is a rubber stamp for XGBoost. There's also a post-hoc ML override at `main.py` ~1261-1279.
- **The bot still uses R1** (deepseek-reasoner) every cycle — `main.py:583` hardcodes `use_r1=not use_local` in the autotrader. The owner's switch to V4 only applied to manual Predict Mode. Owner's own A/B observation: R1 "overthinks" and results drop. Agreement: whichever is BETTER should be used (owner says latency doesn't matter); plan is to A/B R1 vs V4 in shadow mode once inputs are clean — V4 as working default.
- **`smart_money_intel.py` exists but was never wired in** — politicians (Capitol Trades, no key needed), SEC Form 4 insiders (needs free Finnhub key), institutions, options flow, dark pool. Entry point `get_smart_money_score(asset)`. `main.py` imports only `smart_money` (a different, price-action module). The owner explicitly wants this plugged in.
- **News agent** = Google News/CoinDesk/CoinTelegraph/WSJ RSS once per cycle. A market-moving headline can be up to a full interval late. No insider/politician/defense feeds reach it.
- **No market-hours awareness:** stock positions open at midnight, freeze at 0.00% for hours, distort all metrics.
- **Interval finding** from earlier testing (screenshots, before/after CASHOUT ALL): **30m interval is the winner** (~85% win rate post-cashout vs 2h collapsing to ~20%). Judge intervals ONLY on post-cashout equity, never the headline win rate.
- **Metrics are windowed to the LAST 20 CLOSED TRADES** (verified 2026-07-03): `/autotrader/status` computes both WIN RATE and TRUE WIN% from `engine.get_trade_log(20)` (`main.py` ~2605-2608, 2659-2662). Nothing on the dashboard is all-time. Explains why screenshots taken at different times show different percentages for "the same" run.
- **Forensic log root cause found (2026-07-03):** the log is an IN-MEMORY ring buffer capped at 10,000 events (`forensic_log.py:33-34`). Consequences: (a) every Railway restart/redeploy wipes it to zero; (b) past 10k events the OLDEST silently fall off. With ~9-18 assets logging full predictions every 30m, a multi-day export is guaranteed to be missing its beginning. This is why the owner's mid-period screenshots showed data the 35MB export didn't contain — the export wasn't wrong, it was amputated. Fix is in Phase 3.

---

## 4. THE AGREED TRADING PHILOSOPHY (owner-confirmed; replaces the mechanical rules WHEN built)

Every interval, per asset: re-run the full AI, then act — the AI's fresh prediction (not the P&L sign) decides:

| Situation | Action |
|---|---|
| In profit + AI says same direction | Cash out, reopen SAME direction ("profit ratchet" — bank green every interval; the direction continues, the position restarts) |
| In profit + AI predicts reversal before next interval | Cash out NOW, reopen FLIPPED (grab the green before it turns) |
| In loss + AI sees strong reversal | Realize the loss, reopen FLIPPED (earn it back on the reversal) — ONLY with high conviction (see guard) |
| In loss (small, within −5%) + AI says same direction | **HOLD, don't realize** — owner decided 2026-07-03: cash-out-and-reopen is for PROFIT only; small losses are held until recovery or the −5% hard stop |
| Loss beyond hard stop (default 5%) | Emergency close regardless |

**Whipsaw guard (agreed):** flipping is asymmetric by P&L. In profit, flip freely (you bank green either way). In loss, a flip realizes the loss permanently, so it must clear a high conviction bar (e.g., PQS ≥ 4 / confidence threshold / independent sources agreeing) — otherwise noise flip-flops (SELL→BUY→SELL on chop) would stamp three real losses in a row. A weak "maybe" is never enough to book a loss.

**Coupling constraint (agreed):** do NOT ship this prediction-driven flip logic while the signal is still the 52% coin flip — it would just churn faster. §5 Phase 1 (fix the signal) comes first.

**News philosophy (agreed):** the News Agent must do causal second-order reasoning, not sentiment tagging — e.g., "tariffs → container volumes drop → shipping names down, domestic steel up." A major catalyst outranks the math: the Decision Agent must be allowed to override technicals on a genuine market-mover (tariff, Fed surprise, contract award, hack, earnings shock). Impossible under the current "obey the ML" prompt — another reason to ungag it.

### ✅ DECISIONS (owner answered 2026-07-03 — the formerly open questions)

1. **Small loss + AI still says same direction → HOLD, don't realize.** Hold until recovery or the −5% hard stop. Cash-out-and-reopen-same is for profitable positions only.
2. **Major catalyst mid-interval → flip IMMEDIATELY, but VERIFIED.** The cheap 3-min headline poller (no LLM) never trades on its own — a high-impact hit triggers an immediate full 3-agent cycle for that asset, and the flip only executes if the full pipeline confirms the reversal with high conviction (whipsaw guard applies; in-loss flips need the stricter bar). No verification → the catalyst becomes evidence for the next scheduled cycle instead.
3. **Finnhub key → owner will create one (it IS free).** Free tier at finnhub.io (60 calls/min) is plenty. Owner signs up, adds `FINNHUB_API_KEY` to Railway Variables. Design the smart-money wiring to degrade gracefully if the key is absent (politicians + defense feeds still work; insiders become a bonus).
4. **Time-decay rule → KEEP it, extend 6h → 7-8h (default 7, configurable).** Owner's rationale: some losers hover at −1% forever and never reach −5%; the clock clears stale positions. Final loser rule: close at −5% OR after ~7-8h, whichever comes first.
5. **AI weight over ML → re-confirmed explicitly (2026-07-03).** History: earlier the blend deliberately leaned ML because the AI side wasn't tuned yet. Owner now says AI should outweigh ML in the blend. That reasoning is part of Phase 1.
6. **End goal stated: commercial product.** Sequence the owner wants: fix the bot → prove it on paper → owner tests with own real money (small) → productize for others. Currently zero real money at risk.

---

## 5. THE APPROVED BUILD PLAN (designed in full, ZERO code written — awaiting GO)

**Phase 1 — Ungag the judge (fix the signal; biggest impact):**
- Rewrite `decision_agent.py` prompt: ML becomes ONE witness with stated accuracy, not the boss. Judge weighs quant verdict + news verdict + ML + evidence and RULES. Remove/limit the post-hoc ML override in `main.py`.
- Gates become testimony, not executioners: the ~26 checks stop subtracting confidence after the verdict; their FACTS get written into the judge's prompt as an evidence block BEFORE it rules ("daily trend opposes; funding extreme; OBV disagrees"). Keep only hard kills: pre-event blackout, daily loss limit (market-closed becomes a skip, see Phase 3).
- Raise AI weight in the 4-way blend (15% → ~40%); confidence floor 52 → 45; remove hurst+entropy penalties.
- Force-trade fallback only fires with PQS ≥ 3; otherwise skip opening this cycle (existing position still managed).
- Bot default model → V4 (`use_r1` becomes a start-request flag, default false); later A/B R1 vs V4 in shadow mode.
- Quant prompt: restructure the 50-indicator dump into ~6 categorized signal blocks (trend/momentum/volume/volatility/structure), each pre-summarized to a one-line read. Summarize, don't add indicators.

**Phase 2 — News Agent becomes the Catalyst & Intelligence Analyst:**
- Wire `smart_money_intel.get_smart_money_score()` into the pipeline + decision evidence (graceful without Finnhub key, per decision #3).
- News interrupt: cheap 3-min headline poller (no LLM); high-impact hit → immediate full cycle for that asset (catches "Pentagon approves $9B to Dell" in ~3 min, not 30). Immediate verified flip per decision #2.
- Add defense.gov daily contracts RSS (owner trades LMT/RTX). Broaden sources (general news / Reddit-style social if feasible).
- Prompt for causal knock-on reasoning + catalyst-override authority.

**Phase 3 — Structure & honesty:**
- Market hours awareness incl. pre-market (4:00–9:30 ET) and after-hours (16:00–20:00 ET) — owner explicitly wants extended hours traded; skip only the dead zone and weekends; crypto 24/7. Skip predict entirely for closed markets (saves tokens). Frozen positions counted as FLAT, excluded from win metrics.
- Prediction autopsy logging: per prediction, log the chain (AI said X% → blend → evidence → final → fallback?) so exports show WHY, not just what.
- Rebuild the forensic export (owner: "way too confusing and I don't think this is actually correct"): summary header (totals, per-asset W/L/net P&L) + compact per-event lines + autopsy lines. `forensic_log.py::export_text`.
- **Persist forensic events to SQLite** (DB layer already exists in `database.py`) so restarts stop erasing history and the 10k cap stops amputating exports; keep the RAM buffer for the live UI only. Note: full durability across Railway REDEPLOYS additionally needs a Railway volume mounted (one setting in their dashboard) — flag to owner.
- **Per-cycle dashboard snapshot (owner-requested 2026-07-03):** every cycle, save the exact `/autotrader/status` JSON (equity, win rates, every position with live P&L) as a forensic event — the UI is just a rendering of that JSON, so this IS a "screenshot" of what the dashboard would have shown, capturable even when no browser is open. Export includes per-cycle snapshot lines so the owner's real screenshots can be cross-referenced number-for-number by timestamp. (A literal PNG screenshot would require running a headless browser in the container every cycle — heavy, fragile, adds nothing the JSON lacks; optionally render snapshots back into a visual HTML report instead.)
- One metric standard: win = pnl > 0, flat = 0, loss < 0, same everywhere; "—" not red 0% when the log is empty. **Add all-time counters** alongside the current last-20-trades window — dashboard shows both.

**Phase 4 — New cash-out matrix (§4) + exits:** prediction-driven flip logic with whipsaw guard; breakeven stop at +0.3% (SL→entry, losers become scratches); partial cashout at +0.5%; thesis-flip exit (holding loser + AI now opposes at ≥60% → close). Resolve the 6h time-decay question here.

**Phase 5 — Prove it:** shadow-mode A/B (old vs new pipeline side-by-side on paper, both logged) — the numbers pick the winner. Also planned: per-voter scoreboard (grade quant/news/ML/cluster on every closed trade, auto-adjust blend weights weekly) + judge memory (its last 5 calls+outcomes in prompt; a basic trade_history feature already exists).

**REJECTED by owner (do not build):** higher-timeframe trend filter ("we are not getting the fourth trade"); SQLite state persistence across redeploys (owner: not needed, they export history manually).

**Also agreed philosophy points:** the 3-agent shape is correct — do NOT add a fourth LLM voice (more voices = mush). Hedge funds' real LLM use = news synthesis, which validates the News Agent design; their edge is sizing/execution/selectivity, which is what Phases 1+4 give the bot.

---

## 6. REPO MAP (condensed)

```
backend/main.py            ★ everything: ~40 endpoints, _autotrader_loop (~520+),
                             force-trade fallback (~628-653), cash-out block (~659-729),
                             blend (~1173-1202), gates (~1308-1588), floor (507-517),
                             status endpoint w/ MTM (~2558+), cashout-winners (~2538+)
backend/trading_engine.py  ★ Position dataclass, open/close/check_exit, paper/live, Alpaca wired/Binance stub
backend/agents/quant_agent.py     50+ indicator prompt, 3/5 confluence rule, returns direction/conf JSON
backend/agents/news_agent.py      RSS feed list at top, returns sentiment JSON, ±15 max adjustment
backend/agents/decision_agent.py  ★ THE HANDCUFFED JUDGE — prompt ~82-123; R1/V4/GPT-4o/local fallback chain
backend/smart_money_intel.py      ★ BUILT BUT DISCONNECTED — get_smart_money_score(asset)
backend/forensic_log.py           ring buffer (10k), log_*() helpers, export_text() to rebuild
backend/ml_engine.py, indicators.py, data_fetcher.py, config.py (env+assets), database.py
frontend/bot.html          autonomous UI (stats rows, buttons, bRefresh polls status every 10s)
frontend/index.html + app.js      predict mode; keys live in localStorage (um_key, um_dskey…)
docs/TRADINGVIEW_INTEGRATION_PLAN.md   the go-live-via-webhook plan
Dockerfile / railway.json  Railway builds Dockerfile from main, uvicorn :8000
```

---

## 7. SESSION HISTORY (chronological, compressed)

1. Permissions setup; `.claude/settings.json` allows tools.
2. Interval testing (screenshots, before/after CASHOUT ALL): 30m >> 2h on post-cashout equity. Lesson: the forensic export is a point-in-time snapshot; always reconcile with post-cashout equity.
3. Built RESET ALL (endpoint + buttons). Fixed mobile button row. Wrote HANDOFF v1.
4. Explained cash-out mechanics + the win-rate illusion (losers hidden unrealized). Built mark-to-market row + position health + 6h time-decay (the one behavior change; owner kept it after it was flagged). Built CASHOUT WINNERS.
5. New Railway account → bot opened 0 positions in 13 cycles → root cause: no API keys in env (autotrader reads env only, NOT browser localStorage). A fix to pass browser keys was started, owner aborted it ("reverse everything") — reverted cleanly; owner set env keys instead. Env keys list is in §1.
6. Owner reported "50-50" results on new screenshots → full forensic analysis → THE DIAGNOSIS (§3): 52%/PQS-1 fallback signature, handcuffed judge, R1 still active in bot, smart_money_intel disconnected, market-closed stocks frozen at 0.00%, metric-definition clash (100% headline vs 22.2% true on the same screen — both explained exactly; the machine was NOT guessing wrong).
7. Answered "why don't hedge funds just use ChatGPT" (selection bias — the owner picks high-signal moments to ask; funds DO use LLMs for news synthesis; their edge is sizing/execution). Verified the 3-agent architecture against the owner's original tree vision. Designed the full rebuild (§5). Owner approved: market hours WITH extended hours, autopsy logging, export rebuild, insider wiring, exit intelligence. Rejected: trend filter, persistence.
8. Owner said GO; assistant began editing; owner immediately retracted ("we are just discussing") → reverted, tree clean. Then the trading philosophy was refined to the final matrix in §4 (profit ratchet + prediction-driven flips + whipsaw guard + catalyst override). Session ended with open questions unanswered.
9. (2026-07) Owner returned after a month; handoff v2 written.
10. (2026-07-03) New session verified the entire diagnosis against the code (all line pointers confirmed). **Owner answered the 3 open questions** — recorded in §4 DECISIONS.
11. (2026-07-04) Owner re-explained the trading philosophy in their own words (matches §4 matrix: slice a long trade into interval-sized re-decisions; full re-analysis each interval; stay→cash out+reopen same, reversal→flip; prediction credibility decays with horizon). Resolved the time-decay question (keep, extend to 7-8h), re-confirmed AI>ML weighting, stated the commercial-product end goal (§4 decisions 4-6). Reported the forensic-export-vs-screenshot mismatch → root cause found and verified (in-memory 10k ring buffer + restart wipe + last-20-trade metric window; see §3). Requested the per-cycle dashboard snapshot feature → agreed, JSON-snapshot design added to Phase 3. Also asked for the "realized vs actual realized" logic explanation → it's WIN RATE (last 20 closed only) vs TRUE WIN% (closed + open marked-to-market), built in session 4. Still awaiting GO; no build code written.

---

## 8. FIRST THINGS TO DO WHEN YOU TAKE OVER

1. Read §3–§5. Then skim `backend/main.py` (autotrader loop + gates), `agents/decision_agent.py`, `smart_money_intel.py`.
2. Ask the owner the remaining open question in §4 (6h time-decay vs hold-until-−5%) if still unresolved.
3. Wait for an explicit GO, then build **Phase 1 first, alone**, and deploy — it's the fix that makes everything else meaningful. One phase per deploy; measure by post-cashout equity + true win rate before starting the next phase.
4. Branch → commit → push → merge to `main` (deploys) → tell the owner exactly what changed and what difference to expect on the dashboard.

---

*End of handoff v2.1. Reflects the project as of 2026-07-03 (last code change: CASHOUT WINNERS, June; last decision update: owner answered the 3 open questions).*
