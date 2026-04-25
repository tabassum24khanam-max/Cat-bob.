"""ULTRAMAX SMC Engine — Smart Money Concepts"""
import numpy as np


def _calc_atr(candles: list, period: int = 14) -> float:
    """Calculate Average True Range over the given period."""
    if len(candles) < 2:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        high = candles[i]['high']
        low = candles[i]['low']
        prev_close = candles[i - 1]['close']
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if not trs:
        return 0.0
    return float(np.mean(trs[-period:]))


def _find_swing_points(candles: list, lookback: int = 20):
    """Find recent swing highs and swing lows from the last `lookback` candles.
    A swing high is a candle whose high is greater than the high of its neighbours.
    A swing low is a candle whose low is less than the low of its neighbours.
    Returns (swing_highs, swing_lows) as lists of (index, price).
    """
    subset = candles[-lookback:]
    swing_highs = []
    swing_lows = []
    for i in range(1, len(subset) - 1):
        if subset[i]['high'] > subset[i - 1]['high'] and subset[i]['high'] > subset[i + 1]['high']:
            swing_highs.append((len(candles) - lookback + i, subset[i]['high']))
        if subset[i]['low'] < subset[i - 1]['low'] and subset[i]['low'] < subset[i + 1]['low']:
            swing_lows.append((len(candles) - lookback + i, subset[i]['low']))
    return swing_highs, swing_lows


def _detect_order_blocks(candles: list, atr: float) -> list:
    """Detect order blocks.
    Bullish OB: the last bearish candle before a strong bullish move (3+ consecutive
    bullish candles covering > 1 ATR).
    Bearish OB: the last bullish candle before a strong bearish move.
    """
    if len(candles) < 5 or atr <= 0:
        return []

    order_blocks = []

    for i in range(len(candles) - 3):
        # Check for strong bullish move starting at i+1
        bullish_run = 0
        move_start = candles[i + 1]['open']
        move_end = candles[i + 1]['close']
        for j in range(i + 1, len(candles)):
            if candles[j]['close'] > candles[j]['open']:
                bullish_run += 1
                move_end = candles[j]['close']
            else:
                break
        if bullish_run >= 3 and (move_end - move_start) > atr:
            # The candle at index i should be bearish (last bearish before the move)
            if candles[i]['close'] < candles[i]['open']:
                strength = min(1.0, (move_end - move_start) / (atr * 3))
                order_blocks.append({
                    'type': 'bullish',
                    'price': round(candles[i]['low'], 6),
                    'strength': round(strength, 3),
                })

        # Check for strong bearish move starting at i+1
        bearish_run = 0
        move_start_b = candles[i + 1]['open']
        move_end_b = candles[i + 1]['close']
        for j in range(i + 1, len(candles)):
            if candles[j]['close'] < candles[j]['open']:
                bearish_run += 1
                move_end_b = candles[j]['close']
            else:
                break
        if bearish_run >= 3 and (move_start_b - move_end_b) > atr:
            # The candle at index i should be bullish (last bullish before the move)
            if candles[i]['close'] > candles[i]['open']:
                strength = min(1.0, (move_start_b - move_end_b) / (atr * 3))
                order_blocks.append({
                    'type': 'bearish',
                    'price': round(candles[i]['high'], 6),
                    'strength': round(strength, 3),
                })

    return order_blocks


def _detect_fair_value_gaps(candles: list) -> list:
    """Detect fair value gaps (FVGs).
    Bullish FVG: gap between candle[i] high and candle[i+2] low (candle[i+2].low > candle[i].high).
    Bearish FVG: gap between candle[i] low and candle[i+2] high (candle[i+2].high < candle[i].low).
    """
    if len(candles) < 3:
        return []

    fvgs = []
    for i in range(len(candles) - 2):
        # Bullish FVG: gap up
        if candles[i + 2]['low'] > candles[i]['high']:
            fvgs.append({
                'type': 'bullish',
                'top': round(candles[i + 2]['low'], 6),
                'bottom': round(candles[i]['high'], 6),
            })
        # Bearish FVG: gap down
        if candles[i + 2]['high'] < candles[i]['low']:
            fvgs.append({
                'type': 'bearish',
                'top': round(candles[i]['low'], 6),
                'bottom': round(candles[i + 2]['high'], 6),
            })

    return fvgs


def _detect_bos(candles: list, lookback: int = 20) -> dict:
    """Detect Break of Structure.
    Bullish BOS: current price breaks above the most recent swing high.
    Bearish BOS: current price breaks below the most recent swing low.
    """
    if len(candles) < 5:
        return {'bullish': False, 'bearish': False}

    swing_highs, swing_lows = _find_swing_points(candles, lookback)
    current_close = candles[-1]['close']

    bullish_bos = False
    bearish_bos = False

    if swing_highs:
        # Exclude the very last candle from swing detection (it is current)
        recent_highs = [sh for sh in swing_highs if sh[0] < len(candles) - 1]
        if recent_highs:
            highest_swing = max(recent_highs, key=lambda x: x[1])
            if current_close > highest_swing[1]:
                bullish_bos = True

    if swing_lows:
        recent_lows = [sl for sl in swing_lows if sl[0] < len(candles) - 1]
        if recent_lows:
            lowest_swing = min(recent_lows, key=lambda x: x[1])
            if current_close < lowest_swing[1]:
                bearish_bos = True

    return {'bullish': bullish_bos, 'bearish': bearish_bos}


async def detect_smc(candles: list) -> dict:
    """Detect SMC patterns from candles.
    Returns: {available, order_blocks: [{type, price, strength}],
              fair_value_gaps: [{type, top, bottom}],
              bos: {bullish: bool, bearish: bool},
              bias: 'bullish'|'bearish'|'neutral'}
    """
    try:
        if not candles or len(candles) < 10:
            return {'available': False}

        atr = _calc_atr(candles)
        order_blocks = _detect_order_blocks(candles, atr)
        fair_value_gaps = _detect_fair_value_gaps(candles)
        bos = _detect_bos(candles, lookback=20)

        # Determine overall bias
        bullish_signals = 0
        bearish_signals = 0

        # Count order block types (recent ones matter more)
        for ob in order_blocks[-5:]:
            if ob['type'] == 'bullish':
                bullish_signals += 1
            else:
                bearish_signals += 1

        # Count FVG types (recent ones)
        for fvg in fair_value_gaps[-5:]:
            if fvg['type'] == 'bullish':
                bullish_signals += 1
            else:
                bearish_signals += 1

        # BOS is a strong signal
        if bos['bullish']:
            bullish_signals += 3
        if bos['bearish']:
            bearish_signals += 3

        if bullish_signals > bearish_signals + 1:
            bias = 'bullish'
        elif bearish_signals > bullish_signals + 1:
            bias = 'bearish'
        else:
            bias = 'neutral'

        return {
            'available': True,
            'order_blocks': order_blocks[-10:],
            'fair_value_gaps': fair_value_gaps[-10:],
            'bos': bos,
            'bias': bias,
        }

    except Exception:
        return {'available': False}
