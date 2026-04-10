# ULTRAMAX v3.0 — Ultimate Trading AI
## Multi-file Architecture: Python Backend + JS Frontend

---

## Quick Start (2 minutes)

```bash
# 1. Clone / download this folder

# 2. Start backend
./start.sh

# 3. Open frontend in Chrome
# Open: frontend/index.html
# Or visit: http://localhost:8000

# 4. Set API keys in UI (⚙ KEYS button)
#    - Backend URL: http://localhost:8000
#    - OpenAI key: sk-...
#    - DeepSeek key: sk-...
```

---

## Architecture

```
ultramax/
├── start.sh              ← One-command start
├── start_data_agent.sh   ← 24/7 data collection (run separately)
├── backend/
│   ├── main.py           ← FastAPI server (all API routes)
│   ├── database.py       ← SQLite 7-category schema
│   ├── data_agent.py     ← Background data pipeline
│   ├── agents/
│   │   ├── quant_agent.py    ← All math (30+ indicators, Monte Carlo, Kalman, HMM)
│   │   ├── news_agent.py     ← News sentiment, FinBERT, multi-source
│   │   └── decision_agent.py ← DeepSeek R1 final decision
│   ├── requirements.txt
│   └── .env              ← API keys (auto-created)
└── frontend/
    ├── index.html        ← UI
    └── app.js            ← Connects to backend API
```

---

## What Each Agent Does

### Quant Agent (Python, GPT-4o-mini)
Computes 30+ indicators locally:
- RSI, MACD, EMAs, Bollinger, ATR, Stochastic
- Williams %R, OBV, CMF, Supertrend, Parabolic SAR
- Ichimoku Cloud (full 5 lines), Pivot Points
- VWAP, Z-Score, Multi-Period Momentum
- Shannon Entropy, Autocorrelation, Hurst Exponent
- Market Profile POC
- Kalman Filter (noise-filtered trend + uncertainty)
- HMM Regime Detection (TRENDING/RANGING/VOLATILE with probabilities)
- Monte Carlo (1000 path simulations → real price targets)
- Historical similarity search (cosine distance, 5-year DB)

### News Agent (Python, DeepSeek V3.2)
- Fetches 10+ RSS feeds per asset in parallel
- Trust tier system (Fed=1.0, Reuters=0.8, CoinTelegraph=0.6, Reddit=0.3)
- Two-stage filtering: JS keyword relevance → AI filter to 8 headlines
- FinBERT sentiment scoring (real NLP, not rule-based)
- Persistent sentiment memory from database (24h history)
- Macro context: VIX, DXY, Fed stance, Fear & Greed

### Decision Agent (Python, DeepSeek R1)
- Receives full quant + news outputs
- Historical similarity win rates from 5-year database
- Resolves agent conflicts with historical evidence
- 10-minute timeout (R1 takes time to reason properly)
- Falls back to GPT-4o if R1 unavailable

---

## Gates (Hard-Coded, Not Prompts)

1. **Macro Bear Gate** — If 4+/5 bearish signals (MACD, EMA, VIX>18, news negative, F&G<40) → block BUY
2. **Macro Bull Gate** — Reverse for SELL
3. **Confidence cap** — Neutral daily trend + bearish MACD → cap at 65%
4. **Hurst random walk** — Hurst 0.45-0.55 + confidence < 65% → NO_TRADE
5. **Funding extreme** — Funding > 0.08% → penalize BUY -15 confidence
6. **Ichimoku cloud** — Price inside cloud + low confluence → NO_TRADE
7. **Weekend gate** — Stocks blocked on weekends (Saudi time)
8. **Market hours** — Stocks only during NYSE hours (4:30PM-11PM Saudi)

---

## Data Agent (Optional but Recommended)

Run `./start_data_agent.sh` in a separate terminal to:
- Update prices every 15 minutes for all assets
- Scrape news every 30 minutes
- Build 5-year historical database over time
- Compute forward returns (for similarity search)

The more data in the database, the more accurate similarity search becomes.

---

## API Keys Required

| Key | Where to get | Cost |
|-----|-------------|------|
| OpenAI | platform.openai.com | ~$0.01-0.05 per prediction |
| DeepSeek | platform.deepseek.com | ~$0.002-0.01 per prediction |

DeepSeek keys: both R1 and V3 use the same key.

---

## Database

SQLite at `backend/data/ultramax.db` — 7 tables:
- `price_data` — OHLCV + 30 computed indicators per asset per hour
- `news_sentiment` — Hourly aggregated news sentiment
- `social_sentiment` — Reddit, Twitter sentiment
- `macro_data` — VIX, DXY, Fed rate, SPY daily
- `derivatives_data` — Funding rates, OI, liquidations
- `labels` — ML targets (up/flat/down)
- `predictions` — Full prediction history
- `articles` — Individual headlines with FinBERT scores

Check DB status: `http://localhost:8000/db/status`

---

## Upgrading From Single HTML File

Your existing predictions in localStorage are NOT automatically migrated.
To keep your history, either:
- Continue using ULTRAMAX_V3.html for history reference
- Or export from the old file and import to the backend

---

## Next Steps

1. Start data agent and let it run for 24h — builds price history
2. After 24h, similarity search starts working (finds real historical analogs)
3. After 50+ rated predictions, ML classifier learns your specific patterns
4. Weekly: data agent auto-recomputes forward returns for all historical hours

The system gets meaningfully better every week as the database grows.
