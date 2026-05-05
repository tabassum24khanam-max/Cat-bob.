"""
Tier 1.1: Funding Rate + Open Interest from Binance
Shows leveraged positioning. High funding + BUY = crowded trade (likely reversal).
"""
import asyncio
import time
import httpx
from typing import Dict, Optional
from config import BINANCE_SYMBOLS

_cache: Dict[str, dict] = {}
_cache_ttl = 300  # 5 minutes

BINANCE_FAPI = "https://fapi.binance.com"


async def get_funding_rate(asset: str) -> dict:
    """Get current funding rate for a crypto asset."""
    symbol = BINANCE_SYMBOLS.get(asset)
    if not symbol:
        return {'available': False, 'reason': 'not crypto'}

    cache_key = f"funding_{symbol}"
    if cache_key in _cache and time.time() - _cache[cache_key]['ts'] < _cache_ttl:
        return _cache[cache_key]['data']

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{BINANCE_FAPI}/fapi/v1/fundingRate",
                                    params={'symbol': symbol, 'limit': 10})
            if resp.status_code != 200:
                return {'available': False, 'reason': f'HTTP {resp.status_code}'}

            data = resp.json()
            if not data:
                return {'available': False, 'reason': 'no data'}

            rates = [float(d['fundingRate']) for d in data]
            current = rates[-1]
            avg_8h = sum(rates[-3:]) / min(3, len(rates))

            result = {
                'available': True,
                'current_rate': current,
                'current_pct': current * 100,
                'avg_8h': avg_8h,
                'avg_8h_pct': avg_8h * 100,
                'extreme_long': current > 0.0008,  # >0.08%
                'extreme_short': current < -0.0008,
                'signal': _funding_signal(current),
                'ts': int(time.time()),
            }
            _cache[cache_key] = {'data': result, 'ts': time.time()}
            return result
    except Exception as e:
        return {'available': False, 'reason': str(e)[:80]}


async def get_open_interest(asset: str) -> dict:
    """Get open interest and OI change for a crypto asset."""
    symbol = BINANCE_SYMBOLS.get(asset)
    if not symbol:
        return {'available': False, 'reason': 'not crypto'}

    cache_key = f"oi_{symbol}"
    if cache_key in _cache and time.time() - _cache[cache_key]['ts'] < _cache_ttl:
        return _cache[cache_key]['data']

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Current OI
            resp = await client.get(f"{BINANCE_FAPI}/fapi/v1/openInterest",
                                    params={'symbol': symbol})
            if resp.status_code != 200:
                return {'available': False, 'reason': f'HTTP {resp.status_code}'}

            current_oi = float(resp.json().get('openInterest', 0))

            # Historical OI for change calculation
            resp2 = await client.get(f"{BINANCE_FAPI}/futures/data/openInterestHist",
                                     params={'symbol': symbol, 'period': '1h', 'limit': 24})
            oi_history = []
            if resp2.status_code == 200:
                oi_history = [float(d['sumOpenInterest']) for d in resp2.json()]

            oi_change_1h = 0
            oi_change_24h = 0
            if oi_history:
                if len(oi_history) >= 2:
                    oi_change_1h = (current_oi - oi_history[-2]) / oi_history[-2] * 100 if oi_history[-2] else 0
                if len(oi_history) >= 24:
                    oi_change_24h = (current_oi - oi_history[0]) / oi_history[0] * 100 if oi_history[0] else 0

            result = {
                'available': True,
                'oi': current_oi,
                'oi_change_1h_pct': round(oi_change_1h, 2),
                'oi_change_24h_pct': round(oi_change_24h, 2),
                'rising': oi_change_1h > 2,
                'falling': oi_change_1h < -2,
                'signal': _oi_signal(oi_change_1h, oi_change_24h),
                'ts': int(time.time()),
            }
            _cache[cache_key] = {'data': result, 'ts': time.time()}
            return result
    except Exception as e:
        return {'available': False, 'reason': str(e)[:80]}


async def get_long_short_ratio(asset: str) -> dict:
    """Get top trader long/short account ratio."""
    symbol = BINANCE_SYMBOLS.get(asset)
    if not symbol:
        return {'available': False}

    cache_key = f"lsr_{symbol}"
    if cache_key in _cache and time.time() - _cache[cache_key]['ts'] < _cache_ttl:
        return _cache[cache_key]['data']

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{BINANCE_FAPI}/futures/data/topLongShortAccountRatio",
                                    params={'symbol': symbol, 'period': '1h', 'limit': 12})
            if resp.status_code != 200:
                return {'available': False}

            data = resp.json()
            if not data:
                return {'available': False}

            ratios = [float(d['longShortRatio']) for d in data]
            current = ratios[-1]
            avg = sum(ratios) / len(ratios)

            result = {
                'available': True,
                'current_ratio': round(current, 3),
                'avg_12h': round(avg, 3),
                'longs_pct': round(current / (1 + current) * 100, 1),
                'shorts_pct': round(1 / (1 + current) * 100, 1),
                'crowded_long': current > 2.5,
                'crowded_short': current < 0.5,
                'signal': 'SELL' if current > 2.5 else 'BUY' if current < 0.5 else 'NEUTRAL',
            }
            _cache[cache_key] = {'data': result, 'ts': time.time()}
            return result
    except Exception as e:
        return {'available': False}


async def get_funding_oi_combined(asset: str) -> dict:
    """Combined funding + OI + long/short — the full positioning picture."""
    funding, oi, lsr = await asyncio.gather(
        get_funding_rate(asset),
        get_open_interest(asset),
        get_long_short_ratio(asset),
    )

    bias = 0  # -10 to +10, negative = bearish
    reasons = []

    if funding.get('available'):
        if funding['extreme_long']:
            bias -= 3
            reasons.append(f"Funding extreme long ({funding['current_pct']:.3f}%)")
        elif funding['extreme_short']:
            bias += 3
            reasons.append(f"Funding extreme short ({funding['current_pct']:.3f}%)")
        elif funding['current_rate'] > 0.0003:
            bias -= 1
        elif funding['current_rate'] < -0.0003:
            bias += 1

    if oi.get('available'):
        if oi['rising'] and oi['oi_change_1h_pct'] > 5:
            reasons.append(f"OI surging +{oi['oi_change_1h_pct']:.1f}%/1h")
            bias += 1 if funding.get('current_rate', 0) < 0 else -1
        elif oi['falling'] and oi['oi_change_1h_pct'] < -5:
            reasons.append(f"OI dropping {oi['oi_change_1h_pct']:.1f}%/1h (liquidations)")

    if lsr.get('available'):
        if lsr['crowded_long']:
            bias -= 2
            reasons.append(f"Crowd long ({lsr['longs_pct']:.0f}%)")
        elif lsr['crowded_short']:
            bias += 2
            reasons.append(f"Crowd short ({lsr['shorts_pct']:.0f}%)")

    return {
        'funding': funding,
        'oi': oi,
        'long_short': lsr,
        'bias': max(-10, min(10, bias)),
        'signal': 'BUY' if bias >= 3 else 'SELL' if bias <= -3 else 'NEUTRAL',
        'reasons': reasons,
        'available': funding.get('available', False) or oi.get('available', False),
    }


def _funding_signal(rate: float) -> str:
    if rate > 0.0008:
        return 'SELL'  # too many longs, expect correction
    if rate < -0.0008:
        return 'BUY'  # too many shorts, expect squeeze
    return 'NEUTRAL'


def _oi_signal(change_1h: float, change_24h: float) -> str:
    if change_1h > 5 and change_24h > 10:
        return 'MOMENTUM'  # new money entering
    if change_1h < -5:
        return 'LIQUIDATION'  # forced exits
    return 'NEUTRAL'
