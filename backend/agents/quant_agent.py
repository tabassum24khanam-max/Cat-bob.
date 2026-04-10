"""
ULTRAMAX Quant Agent
All mathematical computations: indicators, Monte Carlo, Kalman, HMM, ML classifier
"""
import numpy as np
import json
import asyncio
import httpx
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone


# ─── EMA / RSI / MACD helpers ──────────────────────────────────────────────

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

def compute_indicators(candles: list) -> Optional[Dict[str, Any]]:
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

    # Stochastic
    hh14 = max(h[-14:])
    ll14 = min(l[-14:])
    stoch_k = 100 * (cur - ll14) / (hh14 - ll14 + 0.0001)

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

    # Z-score
    price_zscore = (cur - m20) / (std20 or 0.0001)

    # Multi-period momentum
    mom4  = (cur - c[-5]) / c[-5] * 100 if n >= 5 else 0
    mom10 = (cur - c[-11]) / c[-11] * 100 if n >= 11 else 0
    mom20 = (cur - c[-21]) / c[-21] * 100 if n >= 21 else 0
    mom50 = (cur - c[-51]) / c[-51] * 100 if n >= 51 else 0
    mom24 = (cur - c[-25]) / c[-25] * 100 if n >= 25 else 0
    momentum_score = mom4 * 0.4 + mom10 * 0.3 + mom20 * 0.2 + mom50 * 0.1

    # Shannon Entropy
    rets = [f"{(c[i] - c[i-1]) / c[i-1] * 100:.1f}" for i in range(max(1, n-30), n)]
    counts = {}
    for r in rets:
        counts[r] = counts.get(r, 0) + 1
    total_r = len(rets)
    import math
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

    dist_e9   = (cur - e9)  / e9  * 100
    dist_e20  = (cur - e20) / e20 * 100
    dist_e50  = (cur - e50) / e50 * 100
    dist_e200 = (cur - e200)/ e200* 100

    return {
        'cur': cur, 'atr': atr, 'vol_r': vol_r,
        'rsi14': rsi14, 'rsi7': rsi7,
        'macd_hist': macd_hist, 'macd_val': macd_val,
        'e9': e9, 'e20': e20, 'e50': e50, 'e200': e200,
        'dist_e9': dist_e9, 'dist_e20': dist_e20, 'dist_e50': dist_e50, 'dist_e200': dist_e200,
        'bb_pos': bb_pos, 'bb_width': bb_w, 'bb_upper': bb_u, 'bb_lower': bb_l,
        'stoch_k': stoch_k,
        'will_r14': will_r14, 'will_r28': will_r28,
        'obv': obv, 'obv_slope': obv_slope,
        'cmf': cmf,
        'supertrend_bull': supertrend_bull, 'psar_bull': psar_bull,
        'ich_bull': ich_bull, 'ich_bear': ich_bear,
        'kumo_top': kumo_top, 'kumo_bot': kumo_bot,
        'pivot_p': pivot_p, 'pivot_r1': pivot_r1, 'pivot_s1': pivot_s1,
        'price_zscore': price_zscore,
        'momentum_score': momentum_score,
        'mom4': mom4, 'mom24': mom24,
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
    }


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


def hmm_regime(closes: list, ranges: list):
    if len(closes) < 20:
        return {'state': 'UNKNOWN', 'probs': {'TRENDING': 0.33, 'RANGING': 0.33, 'VOLATILE': 0.33}}
    import math
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


def monte_carlo(cur_price: float, atr: float, horizon_h: int, is_crypto: bool = True) -> dict:
    """1000 Monte Carlo price path simulations."""
    import math, random
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


async def run_quant_agent(asset: str, ind: dict, sim: dict, horizon: int,
                           quant_prompt: str, api_key: str) -> dict:
    """Call GPT-4o-mini with full quant context."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 600,
                "messages": [{"role": "user", "content": quant_prompt}]
            }
        )
        resp.raise_for_status()
        data = resp.json()
        text = data['choices'][0]['message']['content']
        import re
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            return json.loads(m.group())
        return {"direction": "NO_TRADE", "prob_up": 50, "prob_down": 50, "confidence": 40, "reasoning": "Parse error"}


def build_quant_prompt(asset: str, ind: dict, sim: dict, horizon: int,
                        hist_stats: dict = None, pattern_mem: dict = None) -> str:
    mc = sim if sim else {}
    return f"""You are a quantitative trading analyst. Respond ONLY with valid JSON.

ASSET: {asset} | PRICE: {ind['cur']:.4f} | HORIZON: {horizon}h

MARKET REGIME: {ind['regime']} (HMM: TREND={ind['hmm_probs'].get('TRENDING',0):.0%} RANGE={ind['hmm_probs'].get('RANGING',0):.0%} VOL={ind['hmm_probs'].get('VOLATILE',0):.0%})
KALMAN TREND: {ind['kalman_trend']:+.3f}% | Uncertainty: {ind['kalman_uncertainty']:.4f} {'⚠ HIGH' if ind['kalman_uncertainty'] > 0.1 else '✓ clean'}

CORE INDICATORS:
RSI(14): {ind['rsi14']:.1f} {'OVERBOUGHT' if ind['rsi14'] > ind['rsi_overbought'] else 'OVERSOLD' if ind['rsi14'] < ind['rsi_oversold'] else 'neutral'}
MACD Histogram: {ind['macd_hist']:+.4f} {'BULLISH' if ind['macd_hist'] > 0 else 'BEARISH'}
EMA Stack: {ind['ema_align_bull']}/4 bull | {ind['ema_align_bear']}/4 bear
Stochastic K: {ind['stoch_k']:.1f} | Williams %R(14): {ind['will_r14']:.1f}
BB Position: {ind['bb_pos']:.2f} | Width: {ind['bb_width']:.4f} {'COMPRESSION' if ind.get('compression') else ''}

ADVANCED:
CMF: {ind['cmf']:+.3f} {'(money IN)' if ind['cmf'] > 0.1 else '(money OUT)' if ind['cmf'] < -0.1 else ''}
OBV Slope: {'RISING' if ind['obv_slope'] > 0 else 'FALLING'}
Ichimoku: {'ABOVE cloud' if ind['ich_bull'] else 'BELOW cloud' if ind['ich_bear'] else 'INSIDE cloud'}
Supertrend: {'BULL' if ind['supertrend_bull'] else 'BEAR'}
Z-Score: {ind['price_zscore']:+.2f} {'EXTREME' if abs(ind['price_zscore']) > 2 else ''}
Hurst: {ind['hurst_exp']:.3f} {'(trending)' if ind['hurst_exp'] > 0.55 else '(mean-reverting)' if ind['hurst_exp'] < 0.45 else '(random walk)'}
Entropy: {ind['entropy_ratio']:.3f} {'(predictable)' if ind['entropy_ratio'] < 0.4 else '(noisy)'}
Autocorr: {ind['autocorr']:+.3f} {'(momentum)' if ind['autocorr'] > 0.1 else '(mean-rev)' if ind['autocorr'] < -0.1 else ''}
VWAP dist: {ind['dist_vwap']:+.2f}% {'ABOVE' if ind['dist_vwap'] > 0 else 'BELOW'}
POC dist: {ind['dist_poc']:+.2f}%

MONTE CARLO (1000 paths):
Median target: {mc.get('median', ind['cur']):.4f}
Bull (80th pct): {mc.get('bull', ind['cur']):.4f}
Bear (20th pct): {mc.get('bear', ind['cur']):.4f}
Prob up: {mc.get('prob_up', 0.5)*100:.0f}%

{f"HISTORICAL STATS (last {hist_stats['n']} trades): Win rate {hist_stats['win_rate']:.0f}% | Avg return {hist_stats['avg_return']:.2f}%" if hist_stats else ""}
{f"PATTERN MEMORY: {pattern_mem}" if pattern_mem else ""}

RULES:
- Hurst > 0.55: use momentum signals. Hurst < 0.45: use RSI extremes.
- High entropy (>0.6): require strong confirmation before trading.
- Ichimoku inside cloud + low confluence = NO_TRADE.
- VWAP RULE: BUY below VWAP = -10 confidence.

Respond ONLY with this JSON:
{{"direction":"<BUY|SELL|NO_TRADE>","prob_up":<0-100>,"prob_down":<0-100>,"confidence":<0-100>,"reasoning":"<2 sentences: regime + primary signal>","stop_loss_pct":<recommended %>,"mtf_alignment":"<aligned|counter-trend|neutral>","key_levels":{{"support1":<price>,"resistance1":<price>}}}}"""


# ─── ML Classifier (XGBoost trained on prediction history) ─────────────────

def train_ml_classifier(predictions: list) -> dict:
    """Train XGBoost on rated prediction history. Returns model artifact."""
    try:
        import xgboost as xgb
        import numpy as np

        rated = [p for p in predictions if p.get('feedback') in ('correct', 'wrong')
                 and p.get('ind_snapshot')]

        if len(rated) < 20:
            return None  # Not enough data

        import json
        X, y = [], []
        for p in rated:
            snap = p['ind_snapshot'] if isinstance(p['ind_snapshot'], dict) else json.loads(p['ind_snapshot'] or '{}')
            if not snap:
                continue
            features = [
                snap.get('rsi14', 50) / 100,
                snap.get('macd_hist', 0),
                snap.get('dist_vwap', 0),
                snap.get('trend_slope', 0),
                snap.get('trend_stability', 0),
                snap.get('vol_percentile', 50) / 100,
                snap.get('momentum_score', 0),
                snap.get('hurst_exp', 0.5),
                snap.get('entropy_ratio', 0.5),
                snap.get('autocorr', 0),
                1 if snap.get('ich_bull') else 0,
                1 if snap.get('ich_bear') else 0,
                1 if snap.get('supertrend_bull') else 0,
                snap.get('cmf', 0),
                snap.get('price_zscore', 0),
            ]
            X.append(features)
            y.append(1 if p['feedback'] == 'correct' else 0)

        if len(X) < 20:
            return None

        X_np = np.array(X)
        y_np = np.array(y)

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            eval_metric='logloss', use_label_encoder=False, verbosity=0
        )
        model.fit(X_np, y_np)

        # Feature importance
        importance = model.feature_importances_
        feat_names = ['rsi14','macd_hist','dist_vwap','trend_slope','trend_stability',
                      'vol_pct','momentum','hurst','entropy','autocorr',
                      'ich_bull','ich_bear','supertrend','cmf','zscore']
        top_features = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:3]

        return {
            'model': model,
            'n_samples': len(X),
            'top_features': top_features,
            'train_accuracy': float(model.score(X_np, y_np))
        }
    except ImportError:
        return None
    except Exception as e:
        return None


def ml_predict(model_artifact: dict, ind: dict) -> dict:
    """Run ML classifier on current indicators."""
    if not model_artifact or 'model' not in model_artifact:
        return {'confidence': 50, 'available': False}

    try:
        import numpy as np
        features = [[
            ind.get('rsi14', 50) / 100,
            ind.get('macd_hist', 0),
            ind.get('dist_vwap', 0),
            ind.get('trend_slope', 0),
            ind.get('trend_stability', 0),
            ind.get('vol_percentile', 50) / 100,
            ind.get('momentum_score', 0),
            ind.get('hurst_exp', 0.5),
            ind.get('entropy_ratio', 0.5),
            ind.get('autocorr', 0),
            1 if ind.get('ich_bull') else 0,
            1 if ind.get('ich_bear') else 0,
            1 if ind.get('supertrend_bull') else 0,
            ind.get('cmf', 0),
            ind.get('price_zscore', 0),
        ]]
        prob = model_artifact['model'].predict_proba(features)[0]
        return {
            'confidence': float(prob[1]) * 100,
            'available': True,
            'n_samples': model_artifact['n_samples'],
            'top_features': model_artifact['top_features']
        }
    except Exception as e:
        return {'confidence': 50, 'available': False}


def bayesian_confidence(rated_history: list, asset: str, horizon: int, ai_conf: float) -> float:
    """Bayesian posterior: blend AI confidence with historical win rate."""
    asset_rated = [p for p in rated_history
                   if p.get('asset') == asset and p.get('horizon') == horizon
                   and p.get('feedback') in ('correct', 'wrong')]

    if len(asset_rated) < 5:
        return ai_conf  # Not enough data, trust AI

    wins = sum(1 for p in asset_rated if p['feedback'] == 'correct')
    n = len(asset_rated)
    historical_rate = wins / n

    # Beta distribution: alpha=wins+1, beta=losses+1
    # Posterior mean = (wins+1) / (n+2)
    posterior = (wins + 1) / (n + 2)

    # Blend: weight historical more as n grows, cap at 40 samples
    hist_weight = min(0.6, n / 40)
    ai_weight = 1 - hist_weight

    blended = ai_conf * ai_weight + posterior * 100 * hist_weight
    return round(blended, 1)
