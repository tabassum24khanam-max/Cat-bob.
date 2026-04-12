"""
ULTRAMAX Alert Engine — Smart confluence alerts
Scans all assets for high-confluence trading signals
"""
import time
from config import ALL_ASSETS, BINANCE_SYMBOLS, ASSET_NAMES, get_asset_type
from data_fetcher import fetch_candles
from indicators import compute_indicators

# In-memory alert store (latest scan results)
_latest_alerts = []
_last_scan_ts = 0


async def scan_for_alerts() -> list:
    """Scan all assets and return high-confluence alerts."""
    global _latest_alerts, _last_scan_ts

    alerts = []

    for asset in ALL_ASSETS:
        try:
            candles = await fetch_candles(asset, '1h', 300)
            if not candles or len(candles) < 60:
                continue

            ind = compute_indicators(candles)
            if not ind:
                continue

            # Compute confluence score (0-10)
            score = 0
            signals = []

            # 1. RSI extreme
            if ind['rsi14'] < 30:
                score += 1
                signals.append('RSI oversold')
            elif ind['rsi14'] > 70:
                score += 1
                signals.append('RSI overbought')

            # 2. MACD histogram strong
            macd_threshold = abs(ind['cur']) * 0.0005
            if abs(ind['macd_hist']) > macd_threshold:
                score += 1
                signals.append(f"MACD {'bullish' if ind['macd_hist'] > 0 else 'bearish'}")

            # 3. Ichimoku clear signal
            if ind['ich_bull']:
                score += 1
                signals.append('Above Ichimoku cloud')
            elif ind['ich_bear']:
                score += 1
                signals.append('Below Ichimoku cloud')

            # 4. Supertrend
            if ind['supertrend_bull']:
                score += 1
                signals.append('Supertrend bullish')

            # 5. CMF strong money flow
            if abs(ind['cmf']) > 0.15:
                score += 1
                signals.append(f"CMF {'inflow' if ind['cmf'] > 0 else 'outflow'}")

            # 6. Trend stability high
            if ind['trend_stability'] > 0.7:
                score += 1
                signals.append('Strong trend')

            # 7. Hurst non-random
            if ind['hurst_exp'] > 0.6:
                score += 1
                signals.append('Hurst trending')
            elif ind['hurst_exp'] < 0.4:
                score += 1
                signals.append('Hurst mean-reverting')

            # 8. Z-score extreme
            if abs(ind['price_zscore']) > 2:
                score += 1
                signals.append(f"Z-score extreme ({ind['price_zscore']:+.1f})")

            # 9. VWAP extreme distance
            if abs(ind['dist_vwap']) > 2:
                score += 1
                signals.append(f"VWAP dist {ind['dist_vwap']:+.1f}%")

            # 10. Volume spike
            if ind['vol_percentile'] > 80:
                score += 1
                signals.append('High volume')

            # Only alert on 7+ confluence
            if score >= 7:
                # Determine direction
                bull_signals = sum([
                    ind['rsi14'] < 30,
                    ind['macd_hist'] > 0,
                    ind['ich_bull'],
                    ind['supertrend_bull'],
                    ind['cmf'] > 0.1,
                    ind['dist_vwap'] < -1,
                ])
                bear_signals = sum([
                    ind['rsi14'] > 70,
                    ind['macd_hist'] < 0,
                    ind['ich_bear'],
                    not ind['supertrend_bull'],
                    ind['cmf'] < -0.1,
                    ind['dist_vwap'] > 1,
                ])

                if bull_signals > bear_signals:
                    direction = 'BUY'
                elif bear_signals > bull_signals:
                    direction = 'SELL'
                else:
                    direction = 'NEUTRAL'

                alerts.append({
                    'asset': asset,
                    'asset_name': ASSET_NAMES.get(asset, asset),
                    'asset_type': get_asset_type(asset),
                    'score': score,
                    'direction': direction,
                    'signals': signals,
                    'price': ind['cur'],
                    'rsi': round(ind['rsi14'], 1),
                    'regime': ind['regime'],
                    'ts': int(time.time()),
                })

        except Exception:
            continue

    # Sort by score descending
    alerts.sort(key=lambda a: a['score'], reverse=True)

    _latest_alerts = alerts
    _last_scan_ts = int(time.time())

    return alerts


def get_latest_alerts() -> dict:
    """Return the most recent alert scan results."""
    return {
        'alerts': _latest_alerts,
        'last_scan': _last_scan_ts,
        'count': len(_latest_alerts),
    }
