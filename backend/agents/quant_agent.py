"""
ULTRAMAX Quant Agent — LLM-based quantitative analysis
Pure math extracted to indicators.py; ML extracted to ml_engine.py
"""
import json
import asyncio
import httpx
import re
from typing import Optional, Dict, Any
from indicators import compute_indicators, monte_carlo, kalman_filter, hmm_regime
from ml_engine import bayesian_confidence


async def run_quant_agent(asset: str, ind: dict, sim: dict, horizon: int,
                           quant_prompt: str, api_key: str) -> dict:
    """Call GPT-4o-mini with full quant context."""
    if not api_key or len(api_key) < 10:
        return {"direction": "NO_TRADE", "prob_up": 50, "prob_down": 50,
                "confidence": 40, "reasoning": "No OpenAI API key — add it in KEYS settings"}
    try:
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
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                return json.loads(m.group())
            return {"direction": "NO_TRADE", "prob_up": 50, "prob_down": 50, "confidence": 40, "reasoning": "Parse error"}
    except Exception as e:
        return {"direction": "NO_TRADE", "prob_up": 50, "prob_down": 50,
                "confidence": 40, "reasoning": f"Quant agent error: {str(e)[:100]}"}


def build_quant_prompt(asset: str, ind: dict, sim: dict, horizon: int,
                        hist_stats: dict = None, pattern_mem: dict = None,
                        cluster_data: dict = None, correlation_data: dict = None) -> str:
    mc = sim if sim else {}

    # Cluster context
    cluster_ctx = ""
    if cluster_data and cluster_data.get('available'):
        cluster_ctx = f"""
CLUSTER ANALYSIS:
Current cluster: #{cluster_data['cluster_id']} ({cluster_data['n_members']} historical members)
Cluster avg 4h return: {cluster_data.get('avg_fwd_4h', 0):.2f}%
Cluster 4h win rate: {cluster_data.get('win_rate_4h', 50):.0f}%
Distance to centroid: {cluster_data.get('distance', 0):.4f}
"""

    # Correlation context
    corr_ctx = ""
    if correlation_data and correlation_data.get('available'):
        corr_ctx = f"""
CROSS-ASSET CORRELATIONS:
BTC: {correlation_data.get('btc_corr', 0):.2f} | SPY: {correlation_data.get('spy_corr', 0):.2f} | Gold: {correlation_data.get('gold_corr', 0):.2f}
{correlation_data.get('risk_note', '')}
"""
        if correlation_data.get('lead_lag'):
            ll = correlation_data['lead_lag'][0]
            corr_ctx += f"Lead/lag: {ll['leader']} leads {ll['follower']} by {ll['lag_hours']}h (r={ll['correlation']:.2f})\n"

    return f"""You are a quantitative trading analyst. Respond ONLY with valid JSON.

ASSET: {asset} | PRICE: {ind['cur']:.4f} | HORIZON: {horizon}h

MARKET REGIME: {ind['regime']} (HMM: TREND={ind['hmm_probs'].get('TRENDING',0):.0%} RANGE={ind['hmm_probs'].get('RANGING',0):.0%} VOL={ind['hmm_probs'].get('VOLATILE',0):.0%})
KALMAN TREND: {ind['kalman_trend']:+.3f}% | Uncertainty: {ind['kalman_uncertainty']:.4f} {'⚠ HIGH' if ind['kalman_uncertainty'] > 0.1 else '✓ clean'}

CORE INDICATORS:
RSI(14): {ind['rsi14']:.1f} {'OVERBOUGHT' if ind['rsi14'] > ind['rsi_overbought'] else 'OVERSOLD' if ind['rsi14'] < ind['rsi_oversold'] else 'neutral'}
MACD Histogram: {ind['macd_hist']:+.4f} {'BULLISH' if ind['macd_hist'] > 0 else 'BEARISH'}
EMA Stack: {ind['ema_align_bull']}/4 bull | {ind['ema_align_bear']}/4 bear
Stochastic K: {ind['stoch_k']:.1f} | D: {ind.get('stoch_d', 0):.1f} | Williams %R(14): {ind['will_r14']:.1f}
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
Pivots: R2={ind.get('pivot_r2', 0):.4f} R1={ind['pivot_r1']:.4f} P={ind['pivot_p']:.4f} S1={ind['pivot_s1']:.4f} S2={ind.get('pivot_s2', 0):.4f}

CANDLE PATTERNS:
{'ENGULFING(' + ('bull' if ind.get('engulfing', 0) > 0 else 'bear') + ')' if ind.get('engulfing') else ''} {'DOJI' if ind.get('doji') else ''} {'HAMMER' if ind.get('hammer') else ''} {'SHOOTING_STAR' if ind.get('shooting_star') else ''}

MONTE CARLO (1000 paths):
Median target: {mc.get('median', ind['cur']):.4f}
Bull (80th pct): {mc.get('bull', ind['cur']):.4f}
Bear (20th pct): {mc.get('bear', ind['cur']):.4f}
Prob up: {mc.get('prob_up', 0.5)*100:.0f}%
{cluster_ctx}{corr_ctx}
{f"HISTORICAL STATS (last {hist_stats['n']} trades): Win rate {hist_stats['win_rate']:.0f}% | Avg return {hist_stats['avg_return']:.2f}%" if hist_stats else ""}
{f"PATTERN MEMORY: {pattern_mem}" if pattern_mem else ""}

RULES:
- Hurst > 0.55: use momentum signals. Hurst < 0.45: use RSI extremes.
- High entropy (>0.6): require strong confirmation before trading.
- Ichimoku inside cloud + low confluence = NO_TRADE.
- VWAP RULE: BUY below VWAP = -10 confidence.
- Candle patterns add confirmation only — never trade on pattern alone.

Respond ONLY with this JSON:
{{"direction":"<BUY|SELL|NO_TRADE>","prob_up":<0-100>,"prob_down":<0-100>,"confidence":<0-100>,"reasoning":"<2 sentences: regime + primary signal>","stop_loss_pct":<recommended %>,"mtf_alignment":"<aligned|counter-trend|neutral>","key_levels":{{"support1":<price>,"resistance1":<price>}}}}"""
