"""ULTRAMAX Smart Money — Volume-based institutional flow estimation"""
import numpy as np


def _group_candles_by_day(candles: list) -> dict:
    """Group candles into trading days based on their timestamps.
    Returns a dict of {day_key: [candles_in_that_day]}.
    """
    days = {}
    for c in candles:
        # Convert unix timestamp to day key (integer division by 86400)
        day_key = c['time'] // 86400
        if day_key not in days:
            days[day_key] = []
        days[day_key].append(c)
    return days


def _calc_volume_weighted_return(candles_subset: list) -> float:
    """Calculate volume-weighted return for a subset of candles."""
    if not candles_subset:
        return 0.0

    total_volume = sum(c.get('volume', 0) for c in candles_subset)
    if total_volume == 0:
        return 0.0

    weighted_return = 0.0
    for c in candles_subset:
        if c['open'] != 0:
            ret = (c['close'] - c['open']) / c['open']
        else:
            ret = 0.0
        vol = c.get('volume', 0)
        weighted_return += ret * vol

    return weighted_return / total_volume


def analyze_smart_money(candles: list) -> dict:
    """Estimate smart vs dumb money flow from volume patterns.
    Returns: {available, smart_money_index, dumb_money_index,
              divergence: bool, bias: 'bullish'|'bearish'|'neutral'}

    Smart Money Index (SMI): first 30 min + last 60 min of trading session.
    Since we use hourly candles, approximate as first candle + last 2 candles of each day.

    Dumb Money Index (DMI): middle of session volume.
    Approximated as all candles except first and last 2 of each day.

    Divergence = SMI direction differs from DMI direction.
    """
    try:
        if not candles or len(candles) < 10:
            return {'available': False}

        days = _group_candles_by_day(candles)

        # Need at least a few full days to be meaningful
        full_days = {k: v for k, v in days.items() if len(v) >= 5}
        if len(full_days) < 2:
            return {'available': False}

        smi_returns = []
        dmi_returns = []

        for day_key in sorted(full_days.keys()):
            day_candles = full_days[day_key]
            # Sort by time within the day
            day_candles.sort(key=lambda c: c['time'])

            # Smart money: first candle + last 2 candles
            smart_candles = [day_candles[0]] + day_candles[-2:]
            # Dumb money: everything in between
            dumb_candles = day_candles[1:-2] if len(day_candles) > 3 else []

            smi_ret = _calc_volume_weighted_return(smart_candles)
            smi_returns.append(smi_ret)

            if dumb_candles:
                dmi_ret = _calc_volume_weighted_return(dumb_candles)
                dmi_returns.append(dmi_ret)

        if not smi_returns:
            return {'available': False}

        # Aggregate SMI and DMI as cumulative sums of recent days
        # Use last 5 days for the index
        recent_smi = smi_returns[-5:]
        recent_dmi = dmi_returns[-5:] if dmi_returns else [0.0]

        smart_money_index = round(float(np.sum(recent_smi)) * 100, 4)
        dumb_money_index = round(float(np.sum(recent_dmi)) * 100, 4)

        # Determine directions
        smi_direction = 'bullish' if smart_money_index > 0 else 'bearish' if smart_money_index < 0 else 'neutral'
        dmi_direction = 'bullish' if dumb_money_index > 0 else 'bearish' if dumb_money_index < 0 else 'neutral'

        divergence = (smi_direction != dmi_direction) and (smi_direction != 'neutral') and (dmi_direction != 'neutral')

        # Bias follows smart money
        if smart_money_index > 0.05:
            bias = 'bullish'
        elif smart_money_index < -0.05:
            bias = 'bearish'
        else:
            bias = 'neutral'

        return {
            'available': True,
            'smart_money_index': smart_money_index,
            'dumb_money_index': dumb_money_index,
            'divergence': divergence,
            'bias': bias,
        }

    except Exception:
        return {'available': False}
