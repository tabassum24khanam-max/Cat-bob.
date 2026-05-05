"""
Tier 1.2: Liquidation Heatmap — where leveraged traders will get stopped out
Price gravitates toward liquidity clusters. These are free edge.
"""
import asyncio
import time
import httpx
from typing import Dict, List
from config import BINANCE_SYMBOLS

BINANCE_FAPI = "https://fapi.binance.com"
_cache: Dict[str, dict] = {}
_cache_ttl = 120  # 2 minutes


async def get_recent_liquidations(asset: str, limit: int = 50) -> dict:
    """Get recent forced liquidations from Binance futures."""
    symbol = BINANCE_SYMBOLS.get(asset)
    if not symbol:
        return {'available': False, 'reason': 'not crypto'}

    cache_key = f"liq_{symbol}"
    if cache_key in _cache and time.time() - _cache[cache_key]['ts'] < _cache_ttl:
        return _cache[cache_key]['data']

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{BINANCE_FAPI}/fapi/v1/allForceOrders",
                                    params={'symbol': symbol, 'limit': limit})
            if resp.status_code != 200:
                return {'available': False, 'reason': f'HTTP {resp.status_code}'}

            orders = resp.json()
            if not orders:
                return {'available': True, 'liquidations': [], 'summary': _empty_summary()}

            parsed = []
            for o in orders:
                parsed.append({
                    'price': float(o.get('price', 0)),
                    'qty': float(o.get('origQty', 0)),
                    'side': o.get('side', ''),  # BUY = short liquidated, SELL = long liquidated
                    'time': o.get('time', 0),
                    'usd': float(o.get('price', 0)) * float(o.get('origQty', 0)),
                })

            summary = _build_summary(parsed)
            result = {
                'available': True,
                'liquidations': parsed[-20:],
                'summary': summary,
                'ts': int(time.time()),
            }
            _cache[cache_key] = {'data': result, 'ts': time.time()}
            return result
    except Exception as e:
        return {'available': False, 'reason': str(e)[:80]}


async def get_liquidation_levels(asset: str, current_price: float) -> dict:
    """Estimate where liquidation clusters sit based on OI and leverage assumptions."""
    symbol = BINANCE_SYMBOLS.get(asset)
    if not symbol or not current_price:
        return {'available': False}

    levels = []
    for leverage in [5, 10, 20, 25, 50, 100]:
        long_liq = current_price * (1 - 1/leverage)
        short_liq = current_price * (1 + 1/leverage)
        levels.append({
            'leverage': leverage,
            'long_liq_price': round(long_liq, 2),
            'short_liq_price': round(short_liq, 2),
            'long_dist_pct': round(-100/leverage, 2),
            'short_dist_pct': round(100/leverage, 2),
        })

    nearest_long = min(levels, key=lambda x: abs(x['long_liq_price'] - current_price * 0.97))
    nearest_short = min(levels, key=lambda x: abs(x['short_liq_price'] - current_price * 1.03))

    return {
        'available': True,
        'levels': levels,
        'nearest_long_cluster': nearest_long['long_liq_price'],
        'nearest_short_cluster': nearest_short['short_liq_price'],
        'magnetic_down': nearest_long['long_liq_price'],
        'magnetic_up': nearest_short['short_liq_price'],
    }


async def get_liquidation_intel(asset: str, current_price: float) -> dict:
    """Combined liquidation intelligence."""
    recent, levels = await asyncio.gather(
        get_recent_liquidations(asset),
        get_liquidation_levels(asset, current_price),
    )

    bias = 0
    reasons = []

    if recent.get('available') and recent.get('summary'):
        s = recent['summary']
        if s['long_liq_usd'] > s['short_liq_usd'] * 2:
            bias -= 2
            reasons.append(f"Longs getting liquidated (${s['long_liq_usd']:,.0f} vs ${s['short_liq_usd']:,.0f})")
        elif s['short_liq_usd'] > s['long_liq_usd'] * 2:
            bias += 2
            reasons.append(f"Shorts getting squeezed (${s['short_liq_usd']:,.0f})")
        if s['total_usd'] > 1_000_000:
            reasons.append(f"High liquidation volume (${s['total_usd']:,.0f})")

    if levels.get('available') and current_price:
        down_dist = abs(current_price - levels['magnetic_down']) / current_price * 100
        up_dist = abs(levels['magnetic_up'] - current_price) / current_price * 100
        if down_dist < up_dist * 0.7:
            bias -= 1
            reasons.append(f"Liq cluster below closer ({down_dist:.1f}% vs {up_dist:.1f}% up)")
        elif up_dist < down_dist * 0.7:
            bias += 1
            reasons.append(f"Liq cluster above closer ({up_dist:.1f}% vs {down_dist:.1f}% down)")

    return {
        'recent': recent,
        'levels': levels,
        'bias': max(-5, min(5, bias)),
        'signal': 'BUY' if bias >= 2 else 'SELL' if bias <= -2 else 'NEUTRAL',
        'reasons': reasons,
        'available': recent.get('available', False),
    }


def _build_summary(liquidations: List[dict]) -> dict:
    long_liqs = [l for l in liquidations if l['side'] == 'SELL']  # SELL = long liquidated
    short_liqs = [l for l in liquidations if l['side'] == 'BUY']  # BUY = short liquidated

    long_usd = sum(l['usd'] for l in long_liqs)
    short_usd = sum(l['usd'] for l in short_liqs)

    return {
        'total_count': len(liquidations),
        'long_liq_count': len(long_liqs),
        'short_liq_count': len(short_liqs),
        'long_liq_usd': round(long_usd, 2),
        'short_liq_usd': round(short_usd, 2),
        'total_usd': round(long_usd + short_usd, 2),
        'dominant': 'LONG_PAIN' if long_usd > short_usd * 1.5 else 'SHORT_SQUEEZE' if short_usd > long_usd * 1.5 else 'BALANCED',
    }


def _empty_summary() -> dict:
    return {
        'total_count': 0, 'long_liq_count': 0, 'short_liq_count': 0,
        'long_liq_usd': 0, 'short_liq_usd': 0, 'total_usd': 0,
        'dominant': 'NONE',
    }
