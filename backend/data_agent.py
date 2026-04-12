"""
ULTRAMAX Data Agent — 24/7 Background Pipeline
Continuous scraping, database updates, weekly recompute
Now uses centralized config and data_fetcher
"""
import asyncio
import json
import time
import aiosqlite
from datetime import datetime, timezone
from config import ASSETS, BINANCE_SYMBOLS, YAHOO_SYMBOLS, ASSET_NAMES, ALL_ASSETS
from data_fetcher import fetch_binance_candles, fetch_yahoo_candles
from database import DB_PATH, init_db
from agents.news_agent import fetch_asset_news

log_queue = asyncio.Queue()

def now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())

def hour_ts(ts: int = None) -> int:
    t = ts or now_ts()
    return (t // 3600) * 3600


async def log(job: str, asset: str = None, status: str = 'ok', rows: int = 0, error: str = None, duration_ms: int = 0):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO agent_log (job, asset, ts, status, rows_processed, error, duration_ms) VALUES (?,?,?,?,?,?,?)",
            (job, asset, now_ts(), status, rows, error, duration_ms)
        )
        await db.commit()


async def compute_and_store_indicators(asset: str, candles: list):
    """Compute all indicators and store in database."""
    if len(candles) < 60:
        return

    from indicators import compute_indicators, monte_carlo
    ind = compute_indicators(candles)
    if not ind:
        return

    is_crypto = asset in BINANCE_SYMBOLS
    mc = monte_carlo(ind['cur'], ind['atr'], 4, is_crypto)

    ts = hour_ts(candles[-1]['time'])

    data = {
        'open': candles[-1]['open'], 'high': candles[-1]['high'],
        'low': candles[-1]['low'], 'close': candles[-1]['close'],
        'volume': candles[-1].get('volume', 0),
        'rsi14': ind['rsi14'], 'rsi7': ind['rsi7'],
        'macd_hist': ind['macd_hist'], 'macd_val': ind['macd_val'],
        'ema9': ind['e9'], 'ema20': ind['e20'], 'ema50': ind['e50'], 'ema200': ind['e200'],
        'bb_pos': ind['bb_pos'], 'bb_width': ind['bb_width'],
        'bb_upper': ind['bb_upper'], 'bb_lower': ind['bb_lower'],
        'atr': ind['atr'], 'vol_ratio': ind['vol_r'], 'stoch_k': ind['stoch_k'],
        'williams_r14': ind['will_r14'], 'williams_r28': ind['will_r28'],
        'obv': ind['obv'], 'obv_slope': ind['obv_slope'],
        'cmf': ind['cmf'],
        'supertrend_bull': 1 if ind['supertrend_bull'] else 0,
        'psar_bull': 1 if ind['psar_bull'] else 0,
        'ich_bull': 1 if ind['ich_bull'] else 0, 'ich_bear': 1 if ind['ich_bear'] else 0,
        'pivot_p': ind['pivot_p'], 'pivot_r1': ind['pivot_r1'], 'pivot_s1': ind['pivot_s1'],
        'price_zscore': ind['price_zscore'],
        'momentum_score': ind['momentum_score'],
        'entropy_ratio': ind['entropy_ratio'],
        'autocorr': ind['autocorr'],
        'hurst_exp': ind['hurst_exp'],
        'poc': ind['poc'], 'dist_poc': ind['dist_poc'],
        'kalman_estimate': ind['kalman_estimate'],
        'kalman_uncertainty': ind['kalman_uncertainty'],
        'kalman_trend': ind['kalman_trend'],
        'vwap': ind['vwap'], 'dist_vwap': ind['dist_vwap'],
        'trend_slope': ind['trend_slope'], 'trend_stability': ind['trend_stability'],
        'vol_percentile': ind['vol_percentile'], 'vol_zscore': ind['vol_zscore'],
        'regime': ind['regime'],
        'hmm_trending': ind['hmm_probs'].get('TRENDING', 0),
        'hmm_ranging': ind['hmm_probs'].get('RANGING', 0),
        'hmm_volatile': ind['hmm_probs'].get('VOLATILE', 0),
        # New indicator columns
        'stoch_d': ind.get('stoch_d'),
        'pivot_r2': ind.get('pivot_r2'),
        'pivot_s2': ind.get('pivot_s2'),
        'engulfing': ind.get('engulfing', 0),
        'doji': ind.get('doji', 0),
        'hammer': ind.get('hammer', 0),
        'shooting_star': ind.get('shooting_star', 0),
    }

    async with aiosqlite.connect(DB_PATH) as db:
        cols = ['asset', 'ts'] + list(data.keys())
        vals = [asset, ts] + list(data.values())
        placeholders = ','.join(['?' for _ in vals])
        await db.execute(
            f"INSERT OR REPLACE INTO price_data ({','.join(cols)}) VALUES ({placeholders})",
            vals
        )
        await db.commit()


async def compute_forward_returns():
    """Update forward returns for past hours where price data is available."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT asset, ts, close FROM price_data WHERE fwd_1h IS NULL ORDER BY ts DESC LIMIT 10000"
        )
        rows = await cursor.fetchall()

        for row in rows:
            asset, ts, close_price = row['asset'], row['ts'], row['close']
            updates = {}
            for hours, col in [(1, 'fwd_1h'), (4, 'fwd_4h'), (8, 'fwd_8h'), (24, 'fwd_1d'), (72, 'fwd_3d')]:
                future_ts = ts + hours * 3600
                future_cursor = await db.execute(
                    "SELECT close FROM price_data WHERE asset=? AND ts=?",
                    (asset, hour_ts(future_ts))
                )
                future_row = await future_cursor.fetchone()
                if future_row and close_price:
                    fwd = (future_row['close'] - close_price) / close_price * 100
                    updates[col] = fwd

            if updates:
                set_clause = ', '.join(f"{k}=?" for k in updates)
                await db.execute(
                    f"UPDATE price_data SET {set_clause} WHERE asset=? AND ts=?",
                    list(updates.values()) + [asset, ts]
                )

        await db.commit()


async def update_news_sentiment(asset: str, asset_type: str, articles: list):
    """Store aggregated hourly news sentiment."""
    if not articles:
        return

    ts = hour_ts()
    avg_sent = sum(a['sentiment'] for a in articles) / len(articles)
    pos = sum(1 for a in articles if a['sentiment'] > 0.2)
    neg = sum(1 for a in articles if a['sentiment'] < -0.2)
    neu = len(articles) - pos - neg
    avg_imp = sum(a['impact'] for a in articles) / len(articles)
    top_headlines = json.dumps([a['headline'] for a in articles[:3]])
    categories = [a['category'] for a in articles]

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO news_sentiment
            (asset, ts, headline_count, avg_sentiment, positive_count, negative_count,
             neutral_count, avg_impact, max_impact, macro_count, regulatory_count,
             earnings_count, geopolitical_count, top_headlines)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            asset, ts, len(articles), avg_sent, pos, neg, neu, avg_imp,
            max(a['impact'] for a in articles),
            categories.count('macro'), categories.count('regulatory'),
            categories.count('earnings'), categories.count('geopolitical'),
            top_headlines
        ))
        for a in articles:
            try:
                await db.execute("""
                    INSERT OR IGNORE INTO articles (asset, ts, source, tier, headline, sentiment, impact, category)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (asset, ts, a['source'], a['tier'], a['headline'], a['sentiment'], a['impact'], a['category']))
            except Exception:
                pass
        await db.commit()


async def update_macro_data():
    """Fetch macro data (VIX, DXY, etc.) and store."""
    from data_fetcher import fetch_macro
    import httpx

    ts = hour_ts()
    data = {'ts': ts}

    try:
        macro = await fetch_macro()
        if macro:
            data['vix'] = macro.get('vix')
            data['dxy'] = macro.get('dxy')
            data['ten_year_yield'] = macro.get('tenYear')
            data['fed_rate'] = macro.get('fedRate')
            data['spy'] = macro.get('spy')
    except Exception:
        pass

    if 'vix' in data:
        async with aiosqlite.connect(DB_PATH) as db:
            cols = list(data.keys())
            vals = list(data.values())
            await db.execute(
                f"INSERT OR REPLACE INTO macro_data ({','.join(cols)}) VALUES ({','.join(['?' for _ in vals])})",
                vals
            )
            await db.commit()


# ─── Main loop tasks ────────────────────────────────────────────────────────

async def task_update_prices():
    """Update price data for all assets every 15 minutes."""
    while True:
        start = time.time()
        for asset in ASSETS['crypto']:
            try:
                candles = await fetch_binance_candles(BINANCE_SYMBOLS[asset], '1h', 300)
                if candles:
                    await compute_and_store_indicators(asset, candles)
            except Exception as e:
                await log('price_update', asset, 'error', 0, str(e))

        for asset in list(ASSETS['stock']) + list(ASSETS['macro']):
            try:
                candles = await fetch_yahoo_candles(YAHOO_SYMBOLS.get(asset, asset), '1h', 300)
                if candles:
                    await compute_and_store_indicators(asset, candles)
            except Exception as e:
                await log('price_update', asset, 'error', 0, str(e))

        duration = int((time.time() - start) * 1000)
        await log('price_update', None, 'ok', len(ALL_ASSETS), duration_ms=duration)
        print(f"✓ Price update complete in {duration}ms")
        await asyncio.sleep(900)


async def task_update_news():
    """Update news sentiment for all assets every 30 minutes."""
    while True:
        start = time.time()
        for asset in ALL_ASSETS:
            try:
                asset_type = 'crypto' if asset in ASSETS['crypto'] else \
                             'stock' if asset in ASSETS['stock'] else 'macro'
                articles = await fetch_asset_news(asset, ASSET_NAMES.get(asset, asset), asset_type)
                if articles:
                    await update_news_sentiment(asset, asset_type, articles)
            except Exception as e:
                await log('news_update', asset, 'error', 0, str(e))

        duration = int((time.time() - start) * 1000)
        await log('news_update', None, 'ok', len(ALL_ASSETS), duration_ms=duration)
        print(f"✓ News update complete in {duration}ms")
        await asyncio.sleep(1800)


async def task_compute_forward_returns():
    """Compute forward returns every hour."""
    while True:
        await asyncio.sleep(3600)
        try:
            await compute_forward_returns()
            print("✓ Forward returns updated")
        except Exception as e:
            await log('forward_returns', None, 'error', 0, str(e))


async def task_update_macro():
    """Update macro data every hour."""
    while True:
        try:
            await update_macro_data()
        except Exception:
            pass
        await asyncio.sleep(3600)


async def task_update_sentiment():
    """Update social sentiment every 15 minutes."""
    while True:
        try:
            from sentiment import get_sentiment_snapshot
            for asset in ALL_ASSETS[:6]:  # Top 6 assets only (rate limits)
                try:
                    await get_sentiment_snapshot(asset)
                except Exception:
                    pass
                await asyncio.sleep(2)  # Rate limit spacing
        except ImportError:
            pass
        await asyncio.sleep(900)


async def task_refresh_calendar():
    """Refresh economic calendar every 6 hours."""
    while True:
        try:
            from macro_engine import refresh_calendar
            count = await refresh_calendar()
            print(f"✓ Calendar refreshed: {count} events")
        except Exception:
            pass
        await asyncio.sleep(21600)


async def run_data_agent(worker_url: str = None):
    """Main entry point for the Data Agent."""
    from config import WORKER_URL

    await init_db()
    print("🚀 ULTRAMAX Data Agent starting...")
    print(f"   Database: {DB_PATH}")
    print(f"   Worker URL: {worker_url or WORKER_URL}")
    print(f"   Tracking {len(ALL_ASSETS)} assets")
    print()

    # Run initial update
    print("📊 Running initial data collection...")
    for asset in ASSETS['crypto']:
        try:
            candles = await fetch_binance_candles(BINANCE_SYMBOLS[asset], '1h', 300)
            if candles:
                await compute_and_store_indicators(asset, candles)
                print(f'  ✓ {asset}')
        except Exception as e:
            print(f'  ✗ {asset}: {e}')

    # Start all background tasks
    await asyncio.gather(
        task_update_prices(),
        task_update_news(),
        task_compute_forward_returns(),
        task_update_macro(),
        task_update_sentiment(),
        task_refresh_calendar(),
    )


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(run_data_agent(os.getenv('WORKER_URL')))
