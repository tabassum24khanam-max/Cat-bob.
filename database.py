"""
ULTRAMAX Database — SQLite, 7-category schema
"""
import aiosqlite
import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "ultramax.db"
DB_PATH.parent.mkdir(exist_ok=True)

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS price_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL, ts INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    rsi14 REAL, rsi7 REAL, macd_hist REAL, macd_val REAL,
    ema9 REAL, ema20 REAL, ema50 REAL, ema200 REAL,
    bb_pos REAL, bb_width REAL, bb_upper REAL, bb_lower REAL,
    atr REAL, vol_ratio REAL, stoch_k REAL,
    williams_r14 REAL, williams_r28 REAL, obv REAL, obv_slope REAL, cmf REAL,
    supertrend_bull INTEGER, psar_bull INTEGER, ich_bull INTEGER, ich_bear INTEGER,
    pivot_p REAL, pivot_r1 REAL, pivot_s1 REAL,
    price_zscore REAL, momentum_score REAL, entropy_ratio REAL,
    autocorr REAL, hurst_exp REAL, poc REAL, dist_poc REAL,
    kalman_estimate REAL, kalman_uncertainty REAL, kalman_trend REAL,
    vwap REAL, dist_vwap REAL,
    trend_slope REAL, trend_stability REAL, vol_percentile REAL, vol_zscore REAL,
    regime TEXT, hmm_trending REAL, hmm_ranging REAL, hmm_volatile REAL,
    fwd_1h REAL, fwd_4h REAL, fwd_8h REAL, fwd_1d REAL, fwd_3d REAL,
    direction_1h TEXT, direction_4h TEXT, direction_8h TEXT,
    UNIQUE(asset, ts)
);
CREATE TABLE IF NOT EXISTS macro_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL,
    vix REAL, dxy REAL, ten_year_yield REAL, fed_rate REAL,
    spy REAL, spy_chg REAL, gold REAL, oil REAL,
    fear_greed INTEGER, gpr_index REAL, fed_stance TEXT, fomc_days_away INTEGER,
    UNIQUE(ts)
);
CREATE TABLE IF NOT EXISTS news_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT, asset TEXT NOT NULL, ts INTEGER NOT NULL,
    headline_count INTEGER DEFAULT 0, avg_sentiment REAL DEFAULT 0,
    positive_count INTEGER DEFAULT 0, negative_count INTEGER DEFAULT 0,
    neutral_count INTEGER DEFAULT 0, avg_impact REAL DEFAULT 0, max_impact REAL DEFAULT 0,
    macro_count INTEGER DEFAULT 0, regulatory_count INTEGER DEFAULT 0,
    earnings_count INTEGER DEFAULT 0, geopolitical_count INTEGER DEFAULT 0,
    top_headlines TEXT, sentiment_24h_change REAL, sentiment_velocity REAL,
    UNIQUE(asset, ts)
);
CREATE TABLE IF NOT EXISTS derivatives_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT, asset TEXT NOT NULL, ts INTEGER NOT NULL,
    funding_rate REAL, open_interest REAL, oi_delta REAL, long_short_ratio REAL,
    liquidation_long REAL, liquidation_short REAL,
    ob_imbalance REAL, cvd REAL, cvd_pct REAL,
    UNIQUE(asset, ts)
);
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY, saved_at INTEGER NOT NULL,
    asset TEXT NOT NULL, horizon INTEGER NOT NULL,
    decision TEXT NOT NULL, confidence REAL,
    entry_price REAL, target_price REAL, target_bull REAL, target_bear REAL,
    prob_up REAL, prob_down REAL, predicted_price REAL, original_decision TEXT,
    insight TEXT, primary_reason TEXT, agent_model TEXT,
    quant_verdict TEXT, news_verdict TEXT,
    feedback TEXT, outcome_price REAL, outcome_at INTEGER,
    target_hit INTEGER, feedback_note TEXT, gate_reason TEXT,
    ind_snapshot TEXT, ml_score REAL, unscored INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL, ts INTEGER NOT NULL,
    source TEXT, tier REAL DEFAULT 0.5, headline TEXT NOT NULL,
    sentiment REAL, impact REAL DEFAULT 0.5, category TEXT,
    UNIQUE(asset, ts, headline)
);
CREATE TABLE IF NOT EXISTS agent_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job TEXT NOT NULL, asset TEXT, ts INTEGER NOT NULL,
    status TEXT, rows_processed INTEGER DEFAULT 0, error TEXT, duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_price_asset_ts ON price_data(asset, ts);
CREATE INDEX IF NOT EXISTS idx_news_asset_ts ON news_sentiment(asset, ts);
CREATE INDEX IF NOT EXISTS idx_articles_asset_ts ON articles(asset, ts);
CREATE INDEX IF NOT EXISTS idx_predictions_asset ON predictions(asset, saved_at);
"""

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(CREATE_TABLES)
        await db.commit()
    print(f"✓ Database: {DB_PATH}")

async def save_prediction(pred: dict):
    async with aiosqlite.connect(DB_PATH) as db:
        cols = list(pred.keys())
        vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in pred.values()]
        ph = ",".join(["?" for _ in vals])
        await db.execute(f"INSERT OR REPLACE INTO predictions ({','.join(cols)}) VALUES ({ph})", vals)
        await db.commit()

async def get_predictions(asset: str = None, limit: int = 200) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if asset:
            cur = await db.execute(
                "SELECT * FROM predictions WHERE asset=? ORDER BY saved_at DESC LIMIT ?", (asset, limit))
        else:
            cur = await db.execute(
                "SELECT * FROM predictions ORDER BY saved_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in await cur.fetchall()]

async def update_prediction_outcome(pred_id: str, outcome_price: float, outcome_at: int,
                                     feedback: str, target_hit, feedback_note: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""UPDATE predictions
            SET outcome_price=?, outcome_at=?, feedback=?, target_hit=?, feedback_note=?
            WHERE id=?""",
            (outcome_price, outcome_at, feedback,
             int(target_hit) if target_hit is not None else None, feedback_note, pred_id))
        await db.commit()

async def get_news_history(asset: str, hours: int = 24) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM news_sentiment WHERE asset=? ORDER BY ts DESC LIMIT ?", (asset, hours))
        return [dict(r) for r in reversed(await cur.fetchall())]

async def similarity_search(asset: str, current_vec: list, n: int = 50) -> list:
    import numpy as np
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT ts, rsi14, macd_hist, dist_vwap, trend_slope, trend_stability,
                   vol_percentile, momentum_score, hurst_exp, entropy_ratio, autocorr,
                   hmm_trending, hmm_ranging, hmm_volatile,
                   fwd_1h, fwd_4h, fwd_8h, fwd_1d, direction_1h, direction_4h
            FROM price_data WHERE asset=? AND fwd_4h IS NOT NULL
            ORDER BY ts DESC LIMIT 43800""", (asset,))
        rows = await cur.fetchall()
    if len(rows) < 20:
        return []
    feature_cols = ['rsi14','macd_hist','dist_vwap','trend_slope','trend_stability',
                    'vol_percentile','momentum_score','hurst_exp','entropy_ratio',
                    'autocorr','hmm_trending','hmm_ranging','hmm_volatile']
    vecs = [[dict(r).get(c) or 0 for c in feature_cols] for r in rows]
    meta = [dict(r) for r in rows]
    vecs_np = np.array(vecs, dtype=float)
    cur_np = np.array(current_vec[:len(feature_cols)], dtype=float)
    norms = np.linalg.norm(vecs_np, axis=1, keepdims=True)
    norms[norms == 0] = 1
    sims = (vecs_np / norms) @ (cur_np / (np.linalg.norm(cur_np) or 1))
    top_idx = np.argsort(sims)[-n:][::-1]
    return [{'ts': meta[i]['ts'], 'similarity': float(sims[i]),
             'fwd_1h': meta[i].get('fwd_1h'), 'fwd_4h': meta[i].get('fwd_4h'),
             'fwd_8h': meta[i].get('fwd_8h'), 'fwd_1d': meta[i].get('fwd_1d'),
             'direction_4h': meta[i].get('direction_4h')} for i in top_idx]
