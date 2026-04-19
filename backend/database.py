"""
ULTRAMAX Database — SQLite, 12-table schema (7 original + 5 new)
"""
import aiosqlite
import json
import time
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
    ml_confidence REAL, cluster_id INTEGER,
    macro_snapshot TEXT, news_snapshot TEXT,
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

-- New tables for v3.0 expansion
CREATE TABLE IF NOT EXISTS sentiment_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL, ts INTEGER NOT NULL,
    reddit_score REAL, reddit_volume INTEGER DEFAULT 0,
    stocktwits_score REAL, stocktwits_volume INTEGER DEFAULT 0,
    google_trends_score REAL,
    combined_score REAL,
    UNIQUE(asset, ts)
);
CREATE TABLE IF NOT EXISTS macro_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    event_ts INTEGER NOT NULL,
    description TEXT,
    impact_level TEXT DEFAULT 'medium',
    currency TEXT,
    historical_btc_reaction REAL,
    historical_spy_reaction REAL,
    historical_gold_reaction REAL,
    actual_reaction REAL,
    UNIQUE(event_type, event_ts)
);
CREATE TABLE IF NOT EXISTS accuracy_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL, horizon INTEGER NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    correct_count INTEGER DEFAULT 0,
    win_rate REAL, avg_confidence REAL,
    calibration_error REAL,
    updated_at INTEGER,
    UNIQUE(asset, horizon)
);
CREATE TABLE IF NOT EXISTS pattern_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    indicator_signature TEXT NOT NULL,
    outcome TEXT, outcome_count INTEGER DEFAULT 0,
    avg_return REAL, win_rate REAL,
    UNIQUE(asset, indicator_signature)
);
CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,
    centroid TEXT,
    n_members INTEGER DEFAULT 0,
    avg_fwd_1h REAL, avg_fwd_4h REAL, avg_fwd_8h REAL, avg_fwd_1d REAL,
    win_rate_4h REAL,
    updated_at INTEGER,
    UNIQUE(asset, cluster_id)
);

CREATE INDEX IF NOT EXISTS idx_price_asset_ts ON price_data(asset, ts);
CREATE INDEX IF NOT EXISTS idx_news_asset_ts ON news_sentiment(asset, ts);
CREATE INDEX IF NOT EXISTS idx_articles_asset_ts ON articles(asset, ts);
CREATE INDEX IF NOT EXISTS idx_predictions_asset ON predictions(asset, saved_at);
CREATE INDEX IF NOT EXISTS idx_sentiment_asset_ts ON sentiment_snapshots(asset, ts);
CREATE INDEX IF NOT EXISTS idx_macro_events_ts ON macro_events(event_ts);
CREATE INDEX IF NOT EXISTS idx_clusters_asset ON clusters(asset, cluster_id);
"""

# Columns to add to existing tables via migration
_MIGRATIONS = [
    ("price_data", "stoch_d", "REAL"),
    ("price_data", "pivot_r2", "REAL"),
    ("price_data", "pivot_s2", "REAL"),
    ("price_data", "engulfing", "INTEGER"),
    ("price_data", "doji", "INTEGER"),
    ("price_data", "hammer", "INTEGER"),
    ("price_data", "shooting_star", "INTEGER"),
    ("price_data", "cross_btc_corr", "REAL"),
    ("price_data", "cross_spy_corr", "REAL"),
    ("price_data", "event_proximity", "INTEGER"),
    ("price_data", "cluster_id", "INTEGER"),
]


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(CREATE_TABLES)
        # Run column migrations
        for table, col, col_type in _MIGRATIONS:
            try:
                await db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            except Exception:
                pass  # Column already exists
        await db.commit()
    print(f"✓ Database: {DB_PATH}")


_PREDICTION_COLS = {
    'id', 'saved_at', 'asset', 'horizon', 'decision', 'confidence',
    'entry_price', 'target_price', 'target_bull', 'target_bear',
    'prob_up', 'prob_down', 'predicted_price', 'original_decision',
    'insight', 'primary_reason', 'agent_model',
    'quant_verdict', 'news_verdict',
    'feedback', 'outcome_price', 'outcome_at',
    'target_hit', 'feedback_note', 'gate_reason',
    'ind_snapshot', 'ml_score', 'unscored',
    'ml_confidence', 'cluster_id',
    'macro_snapshot', 'news_snapshot', 'created_at',
}


async def save_prediction(pred: dict):
    filtered = {k: v for k, v in pred.items() if k in _PREDICTION_COLS}
    if not filtered.get('id'):
        return
    async with aiosqlite.connect(DB_PATH) as db:
        cols = list(filtered.keys())
        vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in filtered.values()]
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


async def get_macro_history(hours: int = 24) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM macro_data ORDER BY ts DESC LIMIT ?", (hours,))
        return [dict(r) for r in reversed(await cur.fetchall())]


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


# ─── New helper functions for v3.0 expansion ────────────────────────────────

async def save_sentiment_snapshot(asset: str, ts: int, data: dict):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO sentiment_snapshots
            (asset, ts, reddit_score, reddit_volume, stocktwits_score, stocktwits_volume,
             google_trends_score, combined_score)
            VALUES (?,?,?,?,?,?,?,?)
        """, (asset, ts,
              data.get('reddit_score'), data.get('reddit_volume', 0),
              data.get('stocktwits_score'), data.get('stocktwits_volume', 0),
              data.get('google_trends_score'), data.get('combined_score')))
        await db.commit()


async def get_sentiment_history(asset: str, hours: int = 24) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM sentiment_snapshots WHERE asset=? ORDER BY ts DESC LIMIT ?",
            (asset, hours))
        return [dict(r) for r in reversed(await cur.fetchall())]


async def save_macro_event(event: dict):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO macro_events
            (event_type, event_ts, description, impact_level, currency,
             historical_btc_reaction, historical_spy_reaction, historical_gold_reaction)
            VALUES (?,?,?,?,?,?,?,?)
        """, (event['event_type'], event['event_ts'], event.get('description'),
              event.get('impact_level', 'medium'), event.get('currency'),
              event.get('historical_btc_reaction'), event.get('historical_spy_reaction'),
              event.get('historical_gold_reaction')))
        await db.commit()


async def get_upcoming_events(hours_ahead: int = 72) -> list:
    now = int(time.time())
    future = now + hours_ahead * 3600
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM macro_events WHERE event_ts BETWEEN ? AND ? ORDER BY event_ts ASC",
            (now, future))
        rows = [dict(r) for r in await cur.fetchall()]
        for r in rows:
            r['hours_until'] = (r['event_ts'] - now) / 3600
        return rows


async def get_accuracy_stats(asset: str = None) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if asset:
            cur = await db.execute(
                "SELECT * FROM accuracy_stats WHERE asset=? ORDER BY horizon", (asset,))
        else:
            cur = await db.execute("SELECT * FROM accuracy_stats ORDER BY asset, horizon")
        return [dict(r) for r in await cur.fetchall()]


async def update_accuracy_stats(asset: str, horizon: int, feedback: str):
    """Update accuracy stats after a prediction outcome."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM accuracy_stats WHERE asset=? AND horizon=?", (asset, horizon))
        row = await cur.fetchone()

        now = int(time.time())
        if row:
            total = (row['total_predictions'] or 0) + 1
            correct = (row['correct_count'] or 0) + (1 if feedback == 'correct' else 0)
            win_rate = correct / total * 100 if total > 0 else 0
            await db.execute("""
                UPDATE accuracy_stats SET total_predictions=?, correct_count=?, win_rate=?, updated_at=?
                WHERE asset=? AND horizon=?
            """, (total, correct, win_rate, now, asset, horizon))
        else:
            correct = 1 if feedback == 'correct' else 0
            await db.execute("""
                INSERT INTO accuracy_stats (asset, horizon, total_predictions, correct_count, win_rate, updated_at)
                VALUES (?,?,1,?,?,?)
            """, (asset, horizon, correct, correct * 100, now))
        await db.commit()


async def save_clusters(asset: str, cluster_data: list):
    """Save cluster centroids and stats."""
    now = int(time.time())
    async with aiosqlite.connect(DB_PATH) as db:
        # Clear old clusters for this asset
        await db.execute("DELETE FROM clusters WHERE asset=?", (asset,))
        for c in cluster_data:
            await db.execute("""
                INSERT INTO clusters (asset, cluster_id, centroid, n_members,
                    avg_fwd_1h, avg_fwd_4h, avg_fwd_8h, avg_fwd_1d, win_rate_4h, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (asset, c['cluster_id'], json.dumps(c['centroid']), c['n_members'],
                  c.get('avg_fwd_1h'), c.get('avg_fwd_4h'), c.get('avg_fwd_8h'),
                  c.get('avg_fwd_1d'), c.get('win_rate_4h'), now))
        await db.commit()


async def get_clusters(asset: str) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM clusters WHERE asset=? ORDER BY cluster_id", (asset,))
        rows = [dict(r) for r in await cur.fetchall()]
        for r in rows:
            if r.get('centroid'):
                r['centroid'] = json.loads(r['centroid'])
        return rows


async def get_price_history(asset: str, limit: int = 1000) -> list:
    """Get raw price data for correlation/cluster computation."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM price_data WHERE asset=? ORDER BY ts DESC LIMIT ?",
            (asset, limit))
        return [dict(r) for r in reversed(await cur.fetchall())]
