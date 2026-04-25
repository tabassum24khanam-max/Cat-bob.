"""
ULTRAMAX Indicators — All pure mathematical computations
Extracted from quant_agent.py + new candle patterns, Stochastic %D, R2/S2 pivots
"""
import math
import numpy as np
from typing import Optional, Dict, Any


# ─── EMA / RSI helpers ──────────────────────────────────────────────────────

def ema(closes: list, period: int) -> float:
    if len(closes) < period:
        return closes[-1]
    k = 2 / (period + 1)
    val = sum(closes[:period]) / period
    for c in closes[period:]:
        val = c * k + val * (1 - k)
    return val


def ema_arr(closes: list, period: int) -> list:
    if len(closes) < period:
        return closes[:]
    k = 2 / (period + 1)
    result = [sum(closes[:period]) / period]
    for c in closes[period:]:
        result.append(c * k + result[-1] * (1 - k))
    return result


def rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    return 100 - 100 / (1 + ag / (al or 0.0001))


# ─── Candle pattern detection ───────────────────────────────────────────────

def detect_engulfing(candles: list) -> int:
    """Bullish engulfing = +1, Bearish engulfing = -1, None = 0."""
    if len(candles) < 2:
        return 0
    prev, cur = candles[-2], candles[-1]
    prev_body = prev['close'] - prev['open']
    cur_body = cur['close'] - cur['open']
    # Bullish: prev red, cur green, cur body engulfs prev body
    if prev_body < 0 and cur_body > 0:
        if cur['open'] <= prev['close'] and cur['close'] >= prev['open']:
            return 1
    # Bearish: prev green, cur red, cur body engulfs prev body
    if prev_body > 0 and cur_body < 0:
        if cur['open'] >= prev['close'] and cur['close'] <= prev['open']:
            return -1
    return 0


def detect_doji(candles: list) -> int:
    """Doji = 1 if body < 10% of total range, else 0."""
    if not candles:
        return 0
    c = candles[-1]
    total_range = c['high'] - c['low']
    if total_range == 0:
        return 1
    body = abs(c['close'] - c['open'])
    return 1 if body < total_range * 0.1 else 0


def detect_hammer(candles: list) -> int:
    """Hammer (bullish reversal) = 1 in downtrend, else 0."""
    if len(candles) < 6:
        return 0
    c = candles[-1]
    body = abs(c['close'] - c['open'])
    total_range = c['high'] - c['low']
    if total_range == 0 or body == 0:
        return 0
    lower_wick = min(c['open'], c['close']) - c['low']
    upper_wick = c['high'] - max(c['open'], c['close'])
    # Hammer: lower wick > 2x body, upper wick < 0.3x body
    if lower_wick > 2 * body and upper_wick < 0.3 * body:
        # Check downtrend (last 5 closes declining)
        recent = [x['close'] for x in candles[-6:-1]]
        if recent[-1] < recent[0]:
            return 1
    return 0


def detect_shooting_star(candles: list) -> int:
    """Shooting star (bearish reversal) = -1 in uptrend, else 0."""
    if len(candles) < 6:
        return 0
    c = candles[-1]
    body = abs(c['close'] - c['open'])
    total_range = c['high'] - c['low']
    if total_range == 0 or body == 0:
        return 0
    lower_wick = min(c['open'], c['close']) - c['low']
    upper_wick = c['high'] - max(c['open'], c['close'])
    # Shooting star: upper wick > 2x body, lower wick < 0.3x body
    if upper_wick > 2 * body and lower_wick < 0.3 * body:
        # Check uptrend (last 5 closes rising)
        recent = [x['close'] for x in candles[-6:-1]]
        if recent[-1] > recent[0]:
            return -1
    return 0


# B9: Additional candle patterns

def detect_morning_star(candles: list) -> int:
    """Morning star (bullish reversal) = +1 if detected, else 0."""
    if len(candles) < 3:
        return 0
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    b1 = c1['close'] - c1['open']
    b2 = abs(c2['close'] - c2['open'])
    b3 = c3['close'] - c3['open']
    r1 = c1['high'] - c1['low']
    if r1 == 0:
        return 0
    if b1 < 0 and abs(b1) > 0.5 * r1:
        if b2 < 0.3 * r1:
            if b3 > 0 and b3 > 0.5 * abs(b1):
                return 1
    return 0


def detect_evening_star(candles: list) -> int:
    """Evening star (bearish reversal) = -1 if detected, else 0."""
    if len(candles) < 3:
        return 0
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    b1 = c1['close'] - c1['open']
    b2 = abs(c2['close'] - c2['open'])
    b3 = c3['close'] - c3['open']
    r1 = c1['high'] - c1['low']
    if r1 == 0:
        return 0
    if b1 > 0 and b1 > 0.5 * r1:
        if b2 < 0.3 * r1:
            if b3 < 0 and abs(b3) > 0.5 * b1:
                return -1
    return 0


def detect_three_white_soldiers(candles: list) -> int:
    """Three white soldiers (strong bullish) = +1 if detected, else 0."""
    if len(candles) < 3:
        return 0
    for i in range(-3, 0):
        c = candles[i]
        if c['close'] <= c['open']:
            return 0
    for i in range(-2, 0):
        if candles[i]['close'] <= candles[i-1]['close']:
            return 0
    return 1


def detect_three_black_crows(candles: list) -> int:
    """Three black crows (strong bearish) = -1 if detected, else 0."""
    if len(candles) < 3:
        return 0
    for i in range(-3, 0):
        c = candles[i]
        if c['close'] >= c['open']:
            return 0
    for i in range(-2, 0):
        if candles[i]['close'] >= candles[i-1]['close']:
            return 0
    return -1


# ─── Support / Resistance Detection (B6) ──────────────────────────────────

def detect_support_resistance(highs: list, lows: list, closes: list, n_levels: int = 3) -> dict:
    """Find key support/resistance levels using pivot-point clustering.

    1. Identify local highs (resistance candidates) and local lows (support
       candidates) over sliding windows of size 5.
    2. Cluster nearby levels (within 0.5% of each other) by averaging.
    3. Return the top *n_levels* support and resistance levels, sorted.
    """
    n = len(closes)
    if n < 10:
        return {'support': [], 'resistance': []}

    window = 5
    half = window // 2

    # Collect local extremes
    local_highs = []
    local_lows = []
    for i in range(half, n - half):
        if highs[i] == max(highs[i - half:i + half + 1]):
            local_highs.append(highs[i])
        if lows[i] == min(lows[i - half:i + half + 1]):
            local_lows.append(lows[i])

    def cluster_levels(levels: list, threshold_pct: float = 0.5) -> list:
        """Merge nearby price levels and return (avg_price, count) pairs."""
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for lv in levels[1:]:
            if abs(lv - clusters[-1][-1]) / (clusters[-1][-1] + 1e-12) * 100 < threshold_pct:
                clusters[-1].append(lv)
            else:
                clusters.append([lv])
        # Sort by cluster size (most-touched first), return averages
        clusters.sort(key=lambda cl: len(cl), reverse=True)
        return [round(sum(cl) / len(cl), 8) for cl in clusters]

    resistance = cluster_levels(local_highs)[:n_levels]
    support = cluster_levels(local_lows)[:n_levels]
    return {'support': sorted(support), 'resistance': sorted(resistance)}


# ─── RSI Divergence Detection ──────────────────────────────────────────────

def detect_rsi_divergence(closes: list, period: int = 14, lookback: int = 20) -> dict:
    """Detect bullish/bearish RSI divergence by comparing price vs RSI extremes."""
    if len(closes) < period + lookback:
        return {'bullish': False, 'bearish': False, 'bull_strength': 0, 'bear_strength': 0}

    rsi_vals = []
    for i in range(len(closes) - lookback, len(closes) + 1):
        rsi_vals.append(rsi(closes[:i], period))

    recent_prices = closes[-lookback:]
    recent_rsi = rsi_vals[-lookback:]
    half = lookback // 2

    first_prices = recent_prices[:half]
    second_prices = recent_prices[half:]
    first_rsi = recent_rsi[:half]
    second_rsi = recent_rsi[half:]

    price_low_drop = (min(second_prices) - min(first_prices)) / (abs(min(first_prices)) + 1e-9) * 100
    rsi_low_rise = min(second_rsi) - min(first_rsi)
    bullish = price_low_drop < -0.5 and rsi_low_rise > 3

    price_high_rise = (max(second_prices) - max(first_prices)) / (abs(max(first_prices)) + 1e-9) * 100
    rsi_high_drop = max(second_rsi) - max(first_rsi)
    bearish = price_high_rise > 0.5 and rsi_high_drop < -3

    return {
        'bullish': bullish,
        'bearish': bearish,
        'bull_strength': round(abs(rsi_low_rise), 1) if bullish else 0,
        'bear_strength': round(abs(rsi_high_drop), 1) if bearish else 0,
    }


# ─── Kalman Filter ──────────────────────────────────────────────────────────

def kalman_filter(prices: list):
    if len(prices) < 5:
        return prices[-1] if prices else 0, 1, 0
    x, p = prices[0], 1.0
    q, r = 0.01, 0.1
    estimates = []
    for price in prices:
        p_pred = p + q
        k = p_pred / (p_pred + r)
        x = x + k * (price - x)
        p = (1 - k) * p_pred
        estimates.append(x)
    n = len(estimates)
    trend = (estimates[-1] - estimates[-5]) / estimates[-5] * 100 if n >= 5 else 0
    return estimates[-1], p**0.5, trend


# ─── HMM Regime Detection ──────────────────────────────────────────────────

def hmm_regime(closes: list, ranges: list):
    if len(closes) < 20:
        return {'state': 'UNKNOWN', 'probs': {'TRENDING': 0.33, 'RANGING': 0.33, 'VOLATILE': 0.33}}
    rets = [(closes[i] - closes[i-1]) / closes[i-1] * 100 for i in range(1, len(closes))]
    recent_vol = (sum(r**2 for r in rets[-20:]) / 20) ** 0.5

    n = min(20, len(closes))
    xs = list(range(n))
    ys = closes[-n:]
    mx, my = sum(xs)/n, sum(ys)/n
    ssxy = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    ssxx = sum((xs[i]-mx)**2 for i in range(n)) or 0.0001
    slope = ssxy / ssxx
    trend_strength = abs(slope) / (recent_vol + 0.001)

    trend_p = min(0.95, trend_strength * 0.5)
    range_p = min(0.95, 0.8 if recent_vol < 0.3 else 0.5 if recent_vol < 0.6 else 0.2)
    vol_p   = min(0.95, 0.85 if recent_vol > 1.0 else 0.5 if recent_vol > 0.6 else 0.15)
    total = trend_p + range_p + vol_p

    probs = {
        'TRENDING': trend_p / total,
        'RANGING': range_p / total,
        'VOLATILE': vol_p / total
    }
    state = max(probs, key=probs.get)
    return {'state': state, 'probs': probs}


# ─── Monte Carlo ────────────────────────────────────────────────────────────

def monte_carlo(cur_price: float, atr: float, horizon_h: int, is_crypto: bool = True, fat_tails: bool = True) -> dict:
    """1000 Monte Carlo price path simulations with fat-tail support."""
    import random
    n_sims = 1000
    steps = max(1, round(horizon_h))
    step_vol = (atr / cur_price) * (steps ** 0.5) * 0.5

    finals = []
    for _ in range(n_sims):
        price = cur_price
        for _ in range(steps):
            u1 = max(1e-10, random.random())
            u2 = random.random()
            z = (-2 * math.log(u1)) ** 0.5 * math.cos(2 * math.pi * u2)
            if fat_tails and random.random() < 0.05:
                z *= random.uniform(2.0, 4.0)
            price *= (1 + step_vol * z)
        finals.append(price)

    finals.sort()
    median = finals[n_sims // 2]
    bull   = finals[int(n_sims * 0.8)]
    bear   = finals[int(n_sims * 0.2)]
    prob_up = sum(1 for f in finals if f > cur_price) / n_sims

    max_pct = (0.4 if is_crypto else 0.2) if horizon_h <= 1 else \
              (1.2 if is_crypto else 0.6) if horizon_h <= 4 else \
              (2.0 if is_crypto else 1.0) if horizon_h <= 8 else \
              (4.0 if is_crypto else 2.5) if horizon_h <= 24 else \
              (12.0 if is_crypto else 6.0)

    def clamp(p):
        pct = (p - cur_price) / cur_price * 100
        clamped = max(-max_pct, min(max_pct, pct))
        return cur_price * (1 + clamped / 100)

    return {
        'median': clamp(median),
        'bull': clamp(bull),
        'bear': clamp(bear),
        'prob_up': prob_up,
        'max_pct': max_pct
    }


# ─── Main Indicator Computation ─────────────────────────────────────────────

def compute_indicators(candles: list) -> Optional[Dict[str, Any]]:
    """Compute 50+ technical indicators from OHLCV candles."""
    if not candles or len(candles) < 60:
        return None

    c = [x['close'] for x in candles]
    h = [x['high'] for x in candles]
    l = [x['low'] for x in candles]
    v = [x.get('volume', 0) for x in candles]
    n = len(c)
    cur = c[-1]

    # RSI
    rsi14 = rsi(c, 14)
    rsi7 = rsi(c, 7)

    # MACD
    ea12 = ema_arr(c, 12)
    ea26 = ema_arr(c, 26)
    offset = len(ea12) - len(ea26)
    macd_line = [ea12[i + offset] - ea26[i] for i in range(len(ea26))]
    sig_line = ema_arr(macd_line, 9)
    macd_hist = macd_line[-1] - sig_line[-1]
    macd_val = macd_line[-1]

    # EMAs
    e9  = ema(c, 9)
    e20 = ema(c, min(20, n-1))
    e50 = ema(c, min(50, n-1))
    e200= ema(c, min(200, n-1))

    # Bollinger
    sl = c[-20:]
    m20 = sum(sl) / 20
    std20 = (sum((x - m20)**2 for x in sl) / 20) ** 0.5
    bb_u = m20 + 2*std20
    bb_l = m20 - 2*std20
    bb_pos = (cur - bb_l) / (bb_u - bb_l + 0.0001)
    bb_w = (bb_u - bb_l) / m20

    # ATR
    trs = [max(candles[i]['high'] - candles[i]['low'],
               abs(candles[i]['high'] - c[i-1]),
               abs(candles[i]['low'] - c[i-1]))
           for i in range(1, n)]
    atr = sum(trs[-14:]) / 14

    # Volume ratio
    avg_v = sum(v[-20:]) / 20 if v[-1] else 1
    vol_r = v[-1] / (avg_v or 1)

    # Stochastic %K
    hh14 = max(h[-14:])
    ll14 = min(l[-14:])
    stoch_k = 100 * (cur - ll14) / (hh14 - ll14 + 0.0001)

    # Stochastic %D (3-period SMA of %K) — NEW
    stoch_k_vals = []
    for j in range(max(0, n-3), n):
        hh = max(h[max(0,j-13):j+1])
        ll = min(l[max(0,j-13):j+1])
        stoch_k_vals.append(100 * (c[j] - ll) / (hh - ll + 0.0001))
    stoch_d = sum(stoch_k_vals) / len(stoch_k_vals) if stoch_k_vals else stoch_k

    # Williams %R
    hh28 = max(h[-28:]) if n >= 28 else hh14
    ll28 = min(l[-28:]) if n >= 28 else ll14
    will_r14 = -100 * (hh14 - cur) / (hh14 - ll14 + 0.0001)
    will_r28 = -100 * (hh28 - cur) / (hh28 - ll28 + 0.0001)

    # OBV
    obv = 0.0
    for i in range(1, n):
        if c[i] > c[i-1]: obv += v[i]
        elif c[i] < c[i-1]: obv -= v[i]
    obv_prev = 0.0
    for i in range(1, max(1, n-9)):
        if c[i] > c[i-1]: obv_prev += v[i]
        elif c[i] < c[i-1]: obv_prev -= v[i]
    obv_slope = obv - obv_prev

    # CMF
    cmf_num = cmf_den = 0.0
    for x in candles[-20:]:
        rng = x['high'] - x['low']
        mfm = ((x['close'] - x['low']) - (x['high'] - x['close'])) / rng if rng > 0 else 0
        cmf_num += mfm * (x.get('volume', 0) or 0)
        cmf_den += x.get('volume', 0) or 0
    cmf = cmf_num / cmf_den if cmf_den > 0 else 0

    # Supertrend
    st_atr = sum([max(candles[-i-1]['high'] - candles[-i-1]['low'],
                      abs(candles[-i-1]['high'] - c[-i-2]),
                      abs(candles[-i-1]['low'] - c[-i-2]))
                  for i in range(10)]) / 10
    st_lower = (h[-1] + l[-1]) / 2 - 3 * st_atr
    supertrend_bull = cur > st_lower

    # Parabolic SAR (simplified)
    psar_bull = c[-1] > c[-2] if n >= 2 else True

    # Ichimoku
    ich_tenkan = (max(h[-9:]) + min(l[-9:])) / 2
    ich_kijun  = (max(h[-26:]) + min(l[-26:])) / 2 if n >= 26 else ich_tenkan
    senkou_a   = (ich_tenkan + ich_kijun) / 2
    senkou_b   = (max(h[-52:]) + min(l[-52:])) / 2 if n >= 52 else senkou_a
    kumo_top   = max(senkou_a, senkou_b)
    kumo_bot   = min(senkou_a, senkou_b)
    ich_bull   = cur > kumo_top
    ich_bear   = cur < kumo_bot

    # Pivot points
    if n >= 2:
        prev_h, prev_l, prev_c = h[-2], l[-2], c[-2]
    else:
        prev_h, prev_l, prev_c = h[-1], l[-1], c[-1]
    pivot_p  = (prev_h + prev_l + prev_c) / 3
    pivot_r1 = 2 * pivot_p - prev_l
    pivot_s1 = 2 * pivot_p - prev_h
    # R2/S2 pivots — NEW
    pivot_r2 = pivot_p + (prev_h - prev_l)
    pivot_s2 = pivot_p - (prev_h - prev_l)

    # Z-score
    price_zscore = (cur - m20) / (std20 or 0.0001)

    # Multi-period momentum (added mom2, mom72)
    mom2  = (cur - c[-3]) / c[-3] * 100 if n >= 3 else 0
    mom4  = (cur - c[-5]) / c[-5] * 100 if n >= 5 else 0
    mom10 = (cur - c[-11]) / c[-11] * 100 if n >= 11 else 0
    mom20 = (cur - c[-21]) / c[-21] * 100 if n >= 21 else 0
    mom50 = (cur - c[-51]) / c[-51] * 100 if n >= 51 else 0
    mom24 = (cur - c[-25]) / c[-25] * 100 if n >= 25 else 0
    mom72 = (cur - c[-73]) / c[-73] * 100 if n >= 73 else 0
    momentum_score = mom4 * 0.4 + mom10 * 0.3 + mom20 * 0.2 + mom50 * 0.1

    # Shannon Entropy
    rets = [f"{(c[i] - c[i-1]) / c[i-1] * 100:.1f}" for i in range(max(1, n-30), n)]
    counts = {}
    for r in rets:
        counts[r] = counts.get(r, 0) + 1
    total_r = len(rets)
    entropy = -sum((cnt/total_r) * math.log2(cnt/total_r) for cnt in counts.values())
    max_ent = math.log2(max(1, len(counts)))
    entropy_ratio = entropy / max_ent if max_ent > 0 else 0.5

    # Autocorrelation lag-1
    ac_rets = [(c[i] - c[i-1]) / c[i-1] for i in range(max(1, n-20), n)]
    ac_mean = sum(ac_rets) / len(ac_rets) if ac_rets else 0
    ac_num = sum((ac_rets[i] - ac_mean) * (ac_rets[i+1] - ac_mean) for i in range(len(ac_rets)-1))
    ac_den = sum((r - ac_mean)**2 for r in ac_rets) or 0.0001
    autocorr = ac_num / ac_den

    # Hurst exponent
    hurst_rets = [math.log(c[i]/c[i-1]) for i in range(max(1, n-64), n) if c[i-1] > 0]
    if len(hurst_rets) >= 10:
        hm = sum(hurst_rets) / len(hurst_rets)
        cum_dev = []
        acc = 0
        for r in hurst_rets:
            acc += r - hm
            cum_dev.append(acc)
        hr = max(cum_dev) - min(cum_dev)
        hs = (sum((r - hm)**2 for r in hurst_rets) / len(hurst_rets)) ** 0.5
        hurst_exp = math.log(hr / (hs or 0.0001)) / math.log(len(hurst_rets)) if hs > 0 else 0.5
    else:
        hurst_exp = 0.5

    # VWAP
    vwap_num = vwap_den = 0.0
    for x in candles[-50:]:
        tp = (x['high'] + x['low'] + x['close']) / 3
        vol = x.get('volume', 1) or 1
        vwap_num += tp * vol
        vwap_den += vol
    vwap = vwap_num / vwap_den if vwap_den > 0 else cur
    dist_vwap = (cur - vwap) / vwap * 100

    # Trend slope (linear regression)
    sw = c[-20:]
    sn = len(sw)
    smx = (sn - 1) / 2
    smy = sum(sw) / sn
    ssxy = sum((i - smx) * (sw[i] - smy) for i in range(sn))
    ssxx = sum((i - smx)**2 for i in range(sn)) or 0.0001
    slope = ssxy / ssxx
    trend_slope = slope / (cur or 0.0001)

    # R-squared (trend stability)
    y_pred = [smy + slope * (i - smx) for i in range(sn)]
    ss_res = sum((sw[i] - y_pred[i])**2 for i in range(sn))
    ss_tot = sum((sw[i] - smy)**2 for i in range(sn)) or 0.0001
    trend_stability = max(0, 1 - ss_res / ss_tot)

    # Volume percentile
    v50 = sorted(v[-50:])
    cur_v = v[-1]
    vol_percentile = sum(1 for x in v50 if x <= cur_v) / len(v50) * 100

    # Vol z-score
    vm = sum(v[-20:]) / 20
    vs = (sum((x - vm)**2 for x in v[-20:]) / 20) ** 0.5
    vol_zscore = (cur_v - vm) / (vs or 0.0001)

    # Market regime
    if trend_stability > 0.6 and abs(trend_slope) > 0.0002:
        regime = 'TRENDING'
    elif vol_percentile > 80:
        regime = 'HIGH_VOLATILITY'
    elif vol_percentile < 20 and bb_w < 0.02:
        regime = 'LOW_VOLATILITY'
    elif bb_w < 0.015:
        regime = 'RANGING'
    else:
        regime = 'NEUTRAL'

    # EMA alignment
    ema_align_bull = sum([e9 > e20, e20 > e50, e50 > e200])
    ema_align_bear = sum([e9 < e20, e20 < e50, e50 < e200])

    # RSI thresholds (adaptive)
    rsi_overbought = 80 if regime == 'TRENDING' else 70
    rsi_oversold = 20 if regime == 'TRENDING' else 30

    # POC (Market Profile)
    price_step = max(atr * 0.1, cur * 0.0001)
    poc_map = {}
    for x in candles[-20:]:
        lo = (x['low'] // price_step) * price_step
        hi = (x['high'] // price_step + 1) * price_step
        p = lo
        while p <= hi:
            k = round(p, 8)
            poc_map[k] = poc_map.get(k, 0) + 1
            p += price_step
    poc = max(poc_map, key=poc_map.get) if poc_map else cur
    dist_poc = (cur - poc) / poc * 100 if poc else 0

    # Elder Ray
    e13 = ema(c, 13)
    bull_power = h[-1] - e13
    bear_power = l[-1] - e13

    # Kalman filter
    kalman_estimate, kalman_uncertainty, kalman_trend = kalman_filter(c[-50:])

    # HMM regime
    hmm = hmm_regime(c[-50:], [x['high'] - x['low'] for x in candles[-50:]])

    # Candle patterns
    engulfing = detect_engulfing(candles)
    doji = detect_doji(candles)
    hammer = detect_hammer(candles)
    shooting_star = detect_shooting_star(candles)
    morning_star = detect_morning_star(candles)
    evening_star = detect_evening_star(candles)
    three_white = detect_three_white_soldiers(candles)
    three_black = detect_three_black_crows(candles)

    # RSI divergence (B5)
    rsi_div = detect_rsi_divergence(c, 14, 20)

    # Support / Resistance (B6)
    sr = detect_support_resistance(h, l, c)

    # VWAP Bands (B7) — 1 std-dev above/below VWAP
    vwap_sq_num = 0.0
    for x in candles[-50:]:
        tp = (x['high'] + x['low'] + x['close']) / 3
        vol_x = x.get('volume', 1) or 1
        vwap_sq_num += (tp ** 2) * vol_x
    vwap_var = vwap_sq_num / (vwap_den if vwap_den > 0 else 1) - vwap ** 2
    vwap_std = max(0, vwap_var) ** 0.5
    vwap_upper = vwap + vwap_std
    vwap_lower = vwap - vwap_std

    # Enhanced Volume Profile — VAH / VAL (B8)
    # Value Area = price range containing 70% of total volume
    vol_profile = {}
    total_vol_profile = 0.0
    for x in candles[-20:]:
        lo_edge = (x['low'] // price_step) * price_step
        hi_edge = (x['high'] // price_step + 1) * price_step
        p_iter = lo_edge
        bar_vol = x.get('volume', 1) or 1
        n_steps_bar = max(1, int((hi_edge - lo_edge) / price_step))
        vol_per_step = bar_vol / n_steps_bar
        while p_iter <= hi_edge:
            k_vp = round(p_iter, 8)
            vol_profile[k_vp] = vol_profile.get(k_vp, 0) + vol_per_step
            total_vol_profile += vol_per_step
            p_iter += price_step
    # Sort by volume descending, accumulate until 70%
    sorted_vp = sorted(vol_profile.items(), key=lambda kv: kv[1], reverse=True)
    va_target = total_vol_profile * 0.70
    va_accum = 0.0
    va_prices = []
    for price_lv, vol_lv in sorted_vp:
        va_accum += vol_lv
        va_prices.append(price_lv)
        if va_accum >= va_target:
            break
    vah = max(va_prices) if va_prices else cur
    val_ = min(va_prices) if va_prices else cur

    # GARCH-lite volatility forecast (C5) — EWMA variance
    ewma_lambda = 0.94
    log_rets = [(c[i] - c[i-1]) / c[i-1] for i in range(max(1, n - 30), n)]
    if len(log_rets) >= 2:
        ewma_var = log_rets[0] ** 2
        for r in log_rets[1:]:
            ewma_var = ewma_lambda * ewma_var + (1 - ewma_lambda) * r ** 2
        forecast_vol = ewma_var ** 0.5 * 100  # percentage
    else:
        forecast_vol = 0.0

    dist_e9   = (cur - e9)  / e9  * 100
    dist_e20  = (cur - e20) / e20 * 100
    dist_e50  = (cur - e50) / e50 * 100
    dist_e200 = (cur - e200)/ e200* 100

    # ADX (Average Directional Index, 14-period)
    plus_dm_list = []
    minus_dm_list = []
    tr_list = []
    for i in range(1, n):
        up_move = h[i] - h[i - 1]
        down_move = l[i - 1] - l[i]
        plus_dm_list.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm_list.append(down_move if down_move > up_move and down_move > 0 else 0)
        tr_list.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    smoothed_tr = ema_arr(tr_list, 14)
    smoothed_plus_dm = ema_arr(plus_dm_list, 14)
    smoothed_minus_dm = ema_arr(minus_dm_list, 14)
    min_len = min(len(smoothed_tr), len(smoothed_plus_dm), len(smoothed_minus_dm))
    dx_list = []
    for i in range(min_len):
        plus_di = (smoothed_plus_dm[i] / (smoothed_tr[i] or 0.0001)) * 100
        minus_di = (smoothed_minus_dm[i] / (smoothed_tr[i] or 0.0001)) * 100
        di_sum = plus_di + minus_di
        dx_list.append(abs(plus_di - minus_di) / (di_sum or 0.0001) * 100)
    adx = ema(dx_list, 14) if len(dx_list) >= 14 else (sum(dx_list) / len(dx_list) if dx_list else 0)

    # CCI (Commodity Channel Index, 20-period)
    tp_list = [(h[i] + l[i] + c[i]) / 3 for i in range(n)]
    tp_window = tp_list[-20:]
    tp_sma = sum(tp_window) / 20
    tp_mean_dev = sum(abs(tp - tp_sma) for tp in tp_window) / 20
    cci = (tp_list[-1] - tp_sma) / (0.015 * (tp_mean_dev or 0.0001))

    # MFI (Money Flow Index, 14-period)
    tp_prices = [(h[i] + l[i] + c[i]) / 3 for i in range(n)]
    pos_flow = 0.0
    neg_flow = 0.0
    for i in range(n - 14, n):
        raw_mf = tp_prices[i] * v[i]
        if i > 0 and tp_prices[i] > tp_prices[i - 1]:
            pos_flow += raw_mf
        elif i > 0 and tp_prices[i] < tp_prices[i - 1]:
            neg_flow += raw_mf
    money_ratio = pos_flow / (neg_flow or 0.0001)
    mfi = 100 - 100 / (1 + money_ratio)

    # StochRSI (Stochastic RSI, 14-period)
    rsi_vals = []
    for i in range(max(15, n - 28), n + 1):
        rsi_vals.append(rsi(c[:i], 14))
    rsi_window = rsi_vals[-14:] if len(rsi_vals) >= 14 else rsi_vals
    rsi_min = min(rsi_window)
    rsi_max = max(rsi_window)
    stoch_rsi = (rsi_vals[-1] - rsi_min) / (rsi_max - rsi_min + 0.0001) * 100

    return {
        'cur': cur, 'atr': atr, 'vol_r': vol_r,
        'rsi14': rsi14, 'rsi7': rsi7,
        'macd_hist': macd_hist, 'macd_val': macd_val,
        'e9': e9, 'e20': e20, 'e50': e50, 'e200': e200,
        'dist_e9': dist_e9, 'dist_e20': dist_e20, 'dist_e50': dist_e50, 'dist_e200': dist_e200,
        'bb_pos': bb_pos, 'bb_width': bb_w, 'bb_upper': bb_u, 'bb_lower': bb_l,
        'stoch_k': stoch_k, 'stoch_d': stoch_d,
        'will_r14': will_r14, 'will_r28': will_r28,
        'obv': obv, 'obv_slope': obv_slope,
        'cmf': cmf,
        'supertrend_bull': supertrend_bull, 'psar_bull': psar_bull,
        'ich_bull': ich_bull, 'ich_bear': ich_bear,
        'kumo_top': kumo_top, 'kumo_bot': kumo_bot,
        'pivot_p': pivot_p, 'pivot_r1': pivot_r1, 'pivot_s1': pivot_s1,
        'pivot_r2': pivot_r2, 'pivot_s2': pivot_s2,
        'price_zscore': price_zscore,
        'momentum_score': momentum_score,
        'mom2': mom2, 'mom4': mom4, 'mom24': mom24, 'mom72': mom72,
        'entropy_ratio': entropy_ratio,
        'autocorr': autocorr,
        'hurst_exp': hurst_exp,
        'poc': poc, 'dist_poc': dist_poc,
        'bull_power': bull_power, 'bear_power': bear_power,
        'vwap': vwap, 'dist_vwap': dist_vwap,
        'trend_slope': trend_slope, 'trend_stability': trend_stability,
        'vol_percentile': vol_percentile, 'vol_zscore': vol_zscore,
        'ema_align_bull': ema_align_bull, 'ema_align_bear': ema_align_bear,
        'regime': hmm['state'] if hmm['state'] != 'UNKNOWN' else regime,
        'hmm_probs': hmm['probs'],
        'rsi_overbought': rsi_overbought, 'rsi_oversold': rsi_oversold,
        'kalman_estimate': kalman_estimate, 'kalman_uncertainty': kalman_uncertainty,
        'kalman_trend': kalman_trend,
        'compression': bb_w < 0.02,
        # New fields
        'engulfing': engulfing, 'doji': doji,
        'hammer': hammer, 'shooting_star': shooting_star,
        'morning_star': morning_star, 'evening_star': evening_star,
        'three_white_soldiers': three_white, 'three_black_crows': three_black,
        'adx': adx, 'cci': cci, 'mfi': mfi, 'stoch_rsi': stoch_rsi,
        'rsi_div_bull': rsi_div['bullish'], 'rsi_div_bear': rsi_div['bearish'],
        'rsi_div_bull_str': rsi_div['bull_strength'], 'rsi_div_bear_str': rsi_div['bear_strength'],
        # B6: Support / Resistance
        'support_levels': sr['support'], 'resistance_levels': sr['resistance'],
        # B7: VWAP Bands
        'vwap_upper': vwap_upper, 'vwap_lower': vwap_lower,
        # B8: Enhanced Volume Profile
        'vah': vah, 'val': val_,
        # C5: Volatility forecast (GARCH-lite)
        'forecast_vol': forecast_vol,
    }
