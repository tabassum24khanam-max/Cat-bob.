# ULTRAMAX — PROJECT HANDOFF (v2, 2026-07)

> **Paste this entire document into a fresh Claude session to continue with zero context loss.**
> It contains the architecture, the completed forensic diagnosis, the agreed (NOT yet built) redesign, the exact open questions, and file/line pointers for the build. Treat it as the single source of truth. Supersedes the v1 handoff.

---

## 0. WHO YOU ARE (instructions to the new assistant)

You are taking over **ULTRAMAX** — an autonomous multi-agent crypto/stock trading bot (FastAPI + static frontend) deployed on Railway, paper-trading today, aiming for real money via webhook later. The owner is non-technical about code but sharp about product and trading logic. Plain language. Show real numbers.

Hard rules:
- **Never fabricate results.** Compute real numbers from forensic logs; show your work.
- **Paper mode is the default.** Never enable live trading without explicit instruction.
- **Never silently change trading behavior.** If a change alters HOW the bot trades (vs. just displaying more), flag it loudly and get consent. (This rule was violated once with a 6h time-decay rule — owner kept it, but was rightly annoyed it wasn't flagged upfront.)
- **"Discussing" ≠ "build."** The owner explicitly says GO when they want code written. If they say "we are just discussing," do not edit files.
- Develop on branch `claude/review-ultramax-system-oldim`, merge to `main` to deploy (Railway auto-deploys `main`). Repo: `tabassum24khanam-max/Cat-bob.`

---

## 1. WHAT ULTRAMAX IS

FastAPI backend + HTML/JS frontend. A **3-agent pipeline** (Quant → News → Decision) blended with ML/Bayesian/cluster signals, gated by a risk system, scored 0–10 ("PQS"). An **autotrader loop** re-runs the full pipeline per asset every interval (30m/1H/2H) and manages positions via "cash-out cycles." Every event goes to an in-memory **forensic log** exportable as text/JSON. Docker on Railway, SQLite, port 8000. `bot.html` is the autonomous UI; `index.html` is manual predict mode.

**Required env (Railway Variables):** at least one of `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` — with neither set, the autotrader silently skips every asset (this bit us once: 13 cycles, 0 trades, after the owner made a new Railway account). Optional: `FRED_API_KEY`, `FINNHUB_API_KEY`, `TELEGRAM_BOT_TOKEN`+`TELEGRAM_CHAT_ID`, `WORKER_URL` (has hardcoded default). Live-only (do NOT set yet): `ALPACA_KEY/SECRET`, `TRADING_LIVE_ENABLED`, `WEBHOOK_SECRET`.

---

## 2. WHAT IS DEPLOYED AND WORKING TODAY

- Cash-out cycle autotrader (mechanical version — see §4 for the redesign).
- **Mark-to-market dashboard row** on bot.html (yellow): REAL EQUITY, UNREALIZED, IF CASHOUT, TRUE WIN%, W/L OPEN — exposes the gap between headline win rate (realized-only) and reality.
- **`/autotrader/status`** returns `mtm_equity`, `unrealized_pnl`, `cashout_impact`, `true_win_rate`, `open_winners/losers`, `position_health` (per-position: current price, unrealized %/USD, age, SL/TP distance, entry confidence).
- **CASHOUT WINNERS** button + `POST /autotrader/cashout-winners` — closes only profitable positions, keeps losers open, bot keeps running.
- **CASHOUT ALL** (`/autotrader/cashout` — closes everything, stops bot), **RESET ALL** (`/autotrader/reset` — wipes all state to zero), **EXPORT LOG** (forensic text+JSON).
- **6h time-decay**: losers held >6h are force-closed and reopened (in `_autotrader_loop`).
- Known cosmetic bug (diagnosed, unfixed): W/L tile counts pnl ≥ 0 as "winning," TRUE WIN% counts only pnl > 0 — so market-closed stocks frozen at exactly 0.00% show 7W/0L next to a 22–28% true win rate. Also WIN RATE shows red "0%" instead of "—" when the trade log is empty.

---

## 3. THE FORENSIC DIAGNOSIS (completed — this is WHY the bot underperforms)

Owner exported forensic logs; analysis found **almost every trade opens at `Confidence: 52% PQS: 1/10`** — the signature of the force-trade fallback. The AI pipeline works but its signal is destroyed before the trade:

```
Quant agent: "BUY 75%"
  → 4-way blend (AI weight only 15–55%) dilutes to ~63%     [main.py ~1173-1202]
  → ~26 penalty gates fire on NORMAL conditions, cap −25    [main.py ~1308-1551]
    (hurst random −5, entropy −5, OBV −5, CMF −5, daily counter-trend −8…)
  → 63 − 25 = 38%
  → confidence floor (52 default) kills it → NO_TRADE       [main.py ~507-517, 1582-1588]
  → force-trade fallback resurrects a direction at hardcoded 52%, PQS 1  [main.py ~628-653]
Result: the Ferrari engine idles while a coin flip does the trading.
```

More discoveries:
1. **The Decision Agent is handcuffed** (`agents/decision_agent.py` lines ~79-84, 113-117): its prompt says *"YOUR PRIMARY JOB: Follow the ML model's prediction… If ML says BUY with >55%, your decision MUST be BUY. You may only adjust confidence ±10."* The "judge" the owner designed is a rubber stamp for XGBoost. There's also a post-hoc ML override at main.py ~1261-1279.
2. **The bot still uses R1 (deepseek-reasoner) every cycle** — main.py:583 hardcodes `use_r1=not use_local` in the autotrader. The owner's switch to V4 only applied to manual Predict Mode. Owner's own A/B observation: R1 "overthinks" and results drop. Agreement: whichever is BETTER should be used (owner says latency doesn't matter); plan is to A/B R1 vs V4 in shadow mode once inputs are clean — V4 as working default.
3. **`smart_money_intel.py` exists but was never wired in** — politicians (Capitol Trades, no key needed), SEC Form 4 insiders (needs free Finnhub key), institutions, options flow, dark pool. Entry point `get_smart_money_score(asset)`. main.py imports only `smart_money` (a different, price-action module). The owner explicitly wants this plugged in.
4. News agent = Google News/CoinDesk/CoinTelegraph/WSJ **RSS once per cycle**. A market-moving headline can be up to a full interval late. No insider/politician/defense feeds reach it.
5. **No market-hours awareness**: stock positions open at midnight, freeze at 0.00% for hours, distort all metrics.

Interval finding from earlier testing (screenshots, before/after CASHOUT ALL): **30m interval is the winner** (~85% win rate post-cashout vs 2h collapsing to ~20%). Judge intervals ONLY on post-cashout equity, never the headline win rate.

---

## 4. THE AGREED TRADING PHILOSOPHY (owner-confirmed; replaces the mechanical rules WHEN built)

Every interval, per asset: **re-run the full AI, then cash out and reopen — the AI's fresh prediction (not the P&L sign) decides the reopen direction:**

| Situation | Action |
|---|---|
| **In profit + AI says same direction** | Cash out, reopen SAME direction ("profit ratchet" — bank green every interval; the direction continues, the position restarts) |
| **In profit + AI predicts reversal before next interval** | Cash out NOW, reopen FLIPPED (grab the green before it turns) |
| **In loss + AI sees strong reversal** | Realize the loss, reopen FLIPPED (earn it back on the reversal) — ONLY with high conviction (see guard) |
| **In loss + AI says same direction** | ⚠ OPEN QUESTION #1 (below) |
| **Loss beyond hard stop (default 5%)** | Emergency close regardless |

**Whipsaw guard (agreed):** flipping is asymmetric by P&L. In profit, flip freely (you bank green either way). In loss, a flip **realizes the loss permanently**, so it must clear a high conviction bar (e.g., PQS ≥ 4 / confidence threshold / independent sources agreeing) — otherwise noise flip-flops (SELL→BUY→SELL on chop) would stamp three real losses in a row. A weak "maybe" is never enough to book a loss.

**Coupling constraint (agreed):** do NOT ship this prediction-driven flip logic while the signal is still the 52% coin flip — it would just churn faster. §5 Phase 1 (fix the signal) comes first.

**News philosophy (agreed):** the News Agent must do causal second-order reasoning, not sentiment tagging — e.g., *"tariffs → container volumes drop → shipping names down, domestic steel up."* A **major catalyst outranks the math**: the Decision Agent must be allowed to override technicals on a genuine market-mover (tariff, Fed surprise, contract award, hack, earnings shock). Impossible under the current "obey the ML" prompt — another reason to ungag it.

### ⚠ OPEN QUESTIONS THE OWNER NEVER ANSWERED (ask before building the flip logic):
1. **Small loss + AI still says same direction:** (a) HOLD, don't realize (assistant's recommendation — booking a loss to reopen the identical position is pure churn), or (b) cash out + reopen same anyway (pure ratchet, books red)?
2. **Major catalyst mid-interval:** may it flip an open position IMMEDIATELY, or only steer the next cycle's decision?
3. (Minor) **Finnhub free key** for insider data: will the owner create one, or design no-key (politicians/defense feeds only) with insiders as a bonus?

---

## 5. THE APPROVED BUILD PLAN (designed in full, ZERO code written — owner retracted GO to keep discussing)

**Phase 1 — Ungag the judge (fix the signal; biggest impact):**
- Rewrite `decision_agent.py` prompt: ML becomes ONE witness with stated accuracy, not the boss. Judge weighs quant verdict + news verdict + ML + evidence and RULES. Remove/limit the post-hoc ML override in main.py.
- **Gates become testimony, not executioners:** the ~26 checks stop subtracting confidence after the verdict; their FACTS get written into the judge's prompt as an evidence block BEFORE it rules ("daily trend opposes; funding extreme; OBV disagrees"). Keep only hard kills: pre-event blackout, daily loss limit (market-closed becomes a skip, see Phase 3).
- Raise AI weight in the 4-way blend (15% → ~40%); confidence floor 52 → 45; remove hurst+entropy penalties.
- Force-trade fallback only fires with PQS ≥ 3; otherwise skip opening this cycle (existing position still managed).
- Bot default model → V4 (`use_r1` becomes a start-request flag, default false); later A/B R1 vs V4 in shadow mode.
- Quant prompt: restructure the 50-indicator dump into ~6 categorized signal blocks (trend/momentum/volume/volatility/structure), each pre-summarized to a one-line read. Summarize, don't add indicators.

**Phase 2 — News Agent becomes the Catalyst & Intelligence Analyst:**
- Wire `smart_money_intel.get_smart_money_score()` into the pipeline + decision evidence.
- **News interrupt:** cheap 3-min headline poller (no LLM); high-impact hit → immediate full cycle for that asset (catches "Pentagon approves $9B to Dell" in ~3 min, not 30).
- Add defense.gov daily contracts RSS (owner trades LMT/RTX). Broaden sources (general news / Reddit-style social if feasible).
- Prompt for causal knock-on reasoning + catalyst-override authority.

**Phase 3 — Structure & honesty:**
- **Market hours awareness incl. pre-market (4:00–9:30 ET) and after-hours (16:00–20:00 ET)** — owner explicitly wants extended hours traded; skip only the dead zone and weekends; crypto 24/7. Skip predict entirely for closed markets (saves tokens). Frozen positions counted as FLAT, excluded from win metrics.
- **Prediction autopsy logging:** per prediction, log the chain (AI said X% → blend → evidence → final → fallback?) so exports show WHY, not just what.
- **Rebuild the forensic export** (owner: "way too confusing and I don't think this is actually correct"): summary header (totals, per-asset W/L/net P&L) + compact per-event lines + autopsy lines. `forensic_log.py::export_text`.
- One metric standard: win = pnl > 0, flat = 0, loss < 0, same everywhere; "—" not red 0% when the log is empty.

**Phase 4 — New cash-out matrix (§4) + exits:** prediction-driven flip logic with whipsaw guard; breakeven stop at +0.3% (SL→entry, losers become scratches); partial cashout at +0.5%; thesis-flip exit (holding loser + AI now opposes at ≥60% → close).

**Phase 5 — Prove it:** shadow-mode A/B (old vs new pipeline side-by-side on paper, both logged) — the numbers pick the winner. Also planned: per-voter scoreboard (grade quant/news/ML/cluster on every closed trade, auto-adjust blend weights weekly) + judge memory (its last 5 calls+outcomes in prompt; a basic `trade_history` feature already exists).

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
8. Owner said GO; assistant began editing; owner immediately retracted ("we are just discussing") → reverted, tree clean. Then the trading philosophy was refined to the final matrix in §4 (profit ratchet + prediction-driven flips + whipsaw guard + catalyst override). Session ended with open questions #1–3 unanswered.
9. (2026-07) Owner returned after a month; this handoff v2 written. **No build has started. Working tree clean.**

---

## 8. FIRST THINGS TO DO WHEN YOU TAKE OVER

1. Read §3–§5. Then skim `backend/main.py` (autotrader loop + gates), `agents/decision_agent.py`, `smart_money_intel.py`.
2. **Ask the owner the 3 open questions in §4** (loss+same-direction cell; catalyst immediate-flip vs next-cycle; Finnhub key).
3. Wait for an explicit **GO**, then build **Phase 1 first, alone**, and deploy — it's the fix that makes everything else meaningful. One phase per deploy; measure by post-cashout equity + true win rate before starting the next phase.
4. Branch → commit → push → merge to `main` (deploys) → tell the owner exactly what changed and what difference to expect on the dashboard.

*End of handoff v2. Reflects the project as of 2026-07 (last code change: CASHOUT WINNERS, June).*
