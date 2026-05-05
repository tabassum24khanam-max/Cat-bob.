"""
Tier 1.3: Volume Profile (VPOC / VAH / VAL)
Shows WHERE volume traded, not just how much. Price revisits VPOC ~70% of the time.
"""
import numpy as np
from typing import List, Dict, Optional


def compute_volume_profile(candles: List[dict], num_bins: int = 50) -> dict:
    """
    Compute volume profile from OHLCV candles.
    Returns VPOC (Point of Control), VAH (Value Area High), VAL (Value Area Low).
    """
    if not candles or len(candles) < 10:
        return {'available': False, 'reason': 'insufficient data'}

    highs = [c['high'] for c in candles if 'high' in c]
    lows = [c['low'] for c in candles if 'low' in c]
    volumes = [c.get('volume', 0) for c in candles]

    if not highs or not lows:
        return {'available': False, 'reason': 'no OHLC data'}

    price_min = min(lows)
    price_max = max(highs)

    if price_max <= price_min:
        return {'available': False, 'reason': 'flat price'}

    bin_size = (price_max - price_min) / num_bins
    profile = np.zeros(num_bins)

    for i, candle in enumerate(candles):
        h = candle.get('high', 0)
        l = candle.get('low', 0)
        v = volumes[i] if i < len(volumes) else 0
        if h <= l or v <= 0:
            continue

        low_bin = max(0, int((l - price_min) / bin_size))
        high_bin = min(num_bins - 1, int((h - price_min) / bin_size))
        bins_in_candle = high_bin - low_bin + 1
        vol_per_bin = v / bins_in_candle if bins_in_candle > 0 else 0

        for b in range(low_bin, high_bin + 1):
            if 0 <= b < num_bins:
                profile[b] += vol_per_bin

    vpoc_bin = int(np.argmax(profile))
    vpoc = price_min + (vpoc_bin + 0.5) * bin_size

    # Value Area: 70% of total volume centered on VPOC
    total_vol = profile.sum()
    if total_vol == 0:
        return {'available': False, 'reason': 'zero volume'}

    target_vol = total_vol * 0.70
    accumulated = profile[vpoc_bin]
    low_idx = vpoc_bin
    high_idx = vpoc_bin

    while accumulated < target_vol:
        expand_low = profile[low_idx - 1] if low_idx > 0 else 0
        expand_high = profile[high_idx + 1] if high_idx < num_bins - 1 else 0

        if expand_high >= expand_low and high_idx < num_bins - 1:
            high_idx += 1
            accumulated += profile[high_idx]
        elif low_idx > 0:
            low_idx -= 1
            accumulated += profile[low_idx]
        else:
            break

    val = price_min + low_idx * bin_size
    vah = price_min + (high_idx + 1) * bin_size

    # Find high-volume nodes (HVN) and low-volume nodes (LVN)
    mean_vol = profile.mean()
    hvn = []
    lvn = []
    for b in range(num_bins):
        price_level = price_min + (b + 0.5) * bin_size
        if profile[b] > mean_vol * 1.5:
            hvn.append(round(price_level, 4))
        elif profile[b] < mean_vol * 0.3 and profile[b] > 0:
            lvn.append(round(price_level, 4))

    return {
        'available': True,
        'vpoc': round(vpoc, 4),
        'vah': round(vah, 4),
        'val': round(val, 4),
        'value_area_pct': 70,
        'hvn': hvn[:5],
        'lvn': lvn[:5],
        'num_bins': num_bins,
        'price_range': [round(price_min, 4), round(price_max, 4)],
    }


def volume_profile_signal(vp: dict, current_price: float) -> dict:
    """Generate trading signal from volume profile relative to current price."""
    if not vp.get('available') or not current_price:
        return {'signal': 'NEUTRAL', 'confidence_adj': 0, 'reason': 'no VP data'}

    vpoc = vp['vpoc']
    vah = vp['vah']
    val = vp['val']

    dist_vpoc_pct = (current_price - vpoc) / vpoc * 100
    in_value_area = val <= current_price <= vah

    signal = 'NEUTRAL'
    confidence_adj = 0
    reason = ''

    if current_price < val:
        dist_below = (val - current_price) / val * 100
        if dist_below > 1.5:
            signal = 'BUY'
            confidence_adj = 5
            reason = f'Below value area ({dist_below:.1f}%), mean reversion likely → VPOC {vpoc:.2f}'
        else:
            signal = 'BUY'
            confidence_adj = 3
            reason = f'Just below VAL, expect return to {vpoc:.2f}'
    elif current_price > vah:
        dist_above = (current_price - vah) / vah * 100
        if dist_above > 1.5:
            signal = 'SELL'
            confidence_adj = 5
            reason = f'Above value area ({dist_above:.1f}%), gravity pulls to VPOC {vpoc:.2f}'
        else:
            signal = 'SELL'
            confidence_adj = 3
            reason = f'Just above VAH, expect pullback to {vpoc:.2f}'
    elif in_value_area:
        if abs(dist_vpoc_pct) < 0.3:
            signal = 'NEUTRAL'
            confidence_adj = -5
            reason = f'At VPOC — choppy zone, no edge'
        elif current_price < vpoc:
            signal = 'BUY'
            confidence_adj = 2
            reason = f'Below VPOC in value area, drift up likely'
        else:
            signal = 'SELL'
            confidence_adj = 2
            reason = f'Above VPOC in value area, drift down likely'

    # LVN as support/resistance
    lvn_nearby = [l for l in vp.get('lvn', []) if abs(l - current_price) / current_price < 0.02]
    if lvn_nearby:
        reason += f' | LVN gap near {lvn_nearby[0]:.2f} (fast move expected)'
        confidence_adj += 2

    return {
        'signal': signal,
        'confidence_adj': confidence_adj,
        'reason': reason,
        'dist_vpoc_pct': round(dist_vpoc_pct, 2),
        'in_value_area': in_value_area,
        'vpoc': vpoc,
        'vah': vah,
        'val': val,
    }
