"""ULTRAMAX Portfolio Scanner — Scan all assets for opportunities"""
import asyncio
from config import ALL_ASSETS, ASSET_NAMES, get_asset_type
from data_fetcher import fetch_candles
from indicators import compute_indicators


async def scan_portfolio() -> dict:
    """Scan all 18 assets and rank by opportunity strength.
    Returns: {assets: [{asset, score, direction, regime, rsi, key_signal}], ts}
    """
    import time

    async def _scan_one(asset: str) -> dict:
        try:
            candles = await fetch_candles(asset, interval='1h', limit=200)
            if not candles or len(candles) < 60:
                return None
            ind = compute_indicators(candles)
            if not ind:
                return None

            score = 0.0
            signals = []

            # RSI extremes (0-2 pts)
            rsi_val = ind.get('rsi14', 50)
            if rsi_val < 30:
                score += 2.0
                signals.append(f'RSI oversold ({rsi_val:.0f})')
            elif rsi_val > 70:
                score += 2.0
                signals.append(f'RSI overbought ({rsi_val:.0f})')
            elif rsi_val < 40 or rsi_val > 60:
                score += 1.0

            # Trend strength via ADX (0-2 pts)
            adx = ind.get('adx', 25)
            if adx > 40:
                score += 2.0
                signals.append(f'Strong trend (ADX {adx:.0f})')
            elif adx > 25:
                score += 1.0

            # Volume surge (0-2 pts)
            vol_r = ind.get('vol_r', 1)
            if vol_r > 2.0:
                score += 2.0
                signals.append(f'Volume surge ({vol_r:.1f}x)')
            elif vol_r > 1.5:
                score += 1.0

            # MACD signal (0-2 pts)
            macd_hist = ind.get('macd_hist', 0)
            cur = ind.get('cur', 1)
            macd_norm = abs(macd_hist) / (cur * 0.001 + 1e-9)
            if macd_norm > 0.5:
                score += 2.0
                signals.append('MACD strong signal')
            elif macd_norm > 0.2:
                score += 1.0

            # Divergence (0-2 pts)
            if ind.get('rsi_div_bull'):
                score += 2.0
                signals.append('Bullish RSI divergence')
            elif ind.get('rsi_div_bear'):
                score += 2.0
                signals.append('Bearish RSI divergence')

            # Determine direction by majority vote
            bull_votes = 0
            bear_votes = 0

            if rsi_val < 50:
                bear_votes += 1
            else:
                bull_votes += 1

            if macd_hist > 0:
                bull_votes += 1
            else:
                bear_votes += 1

            if ind.get('supertrend_bull'):
                bull_votes += 1
            else:
                bear_votes += 1

            if ind.get('ich_bull'):
                bull_votes += 1
            elif ind.get('ich_bear'):
                bear_votes += 1

            if ind.get('trend_slope', 0) > 0:
                bull_votes += 1
            else:
                bear_votes += 1

            direction = 'BUY' if bull_votes > bear_votes else 'SELL'

            score = min(10.0, round(score, 1))
            key_signal = signals[0] if signals else 'No strong signal'

            return {
                'asset': asset,
                'name': ASSET_NAMES.get(asset, asset),
                'type': get_asset_type(asset),
                'score': score,
                'direction': direction,
                'regime': ind.get('regime', 'UNKNOWN'),
                'rsi': round(rsi_val, 1),
                'key_signal': key_signal,
                'price': cur,
            }
        except Exception:
            return None

    tasks = [_scan_one(asset) for asset in ALL_ASSETS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assets = []
    for r in results:
        if isinstance(r, dict) and r is not None:
            assets.append(r)

    # Sort by score descending
    assets.sort(key=lambda x: x['score'], reverse=True)

    return {
        'assets': assets,
        'ts': int(time.time()),
    }
