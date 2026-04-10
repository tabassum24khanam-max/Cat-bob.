"""
ULTRAMAX Decision Agent — DeepSeek R1
Resolves conflict between Quant and News with historical evidence
"""
import httpx
import json
import re
from typing import Dict, Any, Optional


async def run_decision_agent(
    asset: str,
    ind: dict,
    horizon: int,
    quant: dict,
    news: dict,
    mtf_data: dict,
    mc: dict,
    similarity_results: list,
    ds_key: str,
    api_key: str,
    use_r1: bool = True
) -> dict:
    """Call R1 (or GPT-4o as fallback) for final decision."""

    max_pct = mc.get('max_pct', 3 if horizon <= 4 else 8 if horizon <= 24 else 15)
    price_min = ind['cur'] * (1 - max_pct / 100)
    price_max = ind['cur'] * (1 + max_pct / 100)

    # Build daily trend context
    daily_ctx = ""
    if mtf_data:
        daily_bull = mtf_data.get('daily_bull', False)
        daily_bear = mtf_data.get('daily_bear', False)
        daily_ctx = f"""
DAILY TREND CONTEXT:
Daily MACD: {'POSITIVE (bullish)' if mtf_data.get('daily_macd_hist', 0) > 0 else 'NEGATIVE (bearish)'}
Daily EMA20 dist: {mtf_data.get('daily_dist_e20', 0):+.2f}%
Daily direction: {'BULL' if daily_bull else 'BEAR' if daily_bear else 'NEUTRAL'}
{'⚠ COUNTER-TREND: Quant wants ' + quant['direction'] + ' but daily is ' + ('BULL' if daily_bull else 'BEAR') + ' — reduce confidence 15pts' if (daily_bull and quant['direction'] == 'SELL') or (daily_bear and quant['direction'] == 'BUY') else '✓ ALIGNED: quant matches daily trend'}"""

    # Build similarity context
    sim_ctx = ""
    if similarity_results and len(similarity_results) >= 10:
        sim_wins = sum(1 for s in similarity_results if
                       (quant['direction'] == 'BUY' and (s.get('fwd_4h') or 0) > 0) or
                       (quant['direction'] == 'SELL' and (s.get('fwd_4h') or 0) < 0))
        sim_win_rate = sim_wins / len(similarity_results) * 100
        avg_fwd = sum(s.get('fwd_4h') or 0 for s in similarity_results) / len(similarity_results)
        sim_ctx = f"""
HISTORICAL SIMILARITY ({len(similarity_results)} similar periods found):
Win rate for {quant['direction']} in similar conditions: {sim_win_rate:.0f}%
Average 4H forward return in similar conditions: {avg_fwd:+.2f}%
Strongest match similarity: {similarity_results[0]['similarity']:.3f}
Note: This is ACTUAL historical evidence — weight it heavily."""

    prompt = f"""You are DeepSeek R1, the final decision agent for ULTRAMAX trading AI. You resolve conflicts between the Quant Agent (math) and News Agent (sentiment) using historical evidence.

ASSET: {asset} | PRICE: {ind['cur']:.4f} | HORIZON: {horizon}h
VALID PRICE RANGE: {price_min:.4f} to {price_max:.4f} (max {max_pct}% move)

QUANT AGENT VERDICT:
Direction: {quant['direction']} | Confidence: {quant['confidence']}%
Prob up: {quant['prob_up']}% | Prob down: {quant['prob_down']}%
Reasoning: {quant.get('reasoning', 'N/A')}
MACD: {'BULL' if ind['macd_hist'] > 0 else 'BEAR'} | RSI: {ind['rsi14']:.1f} | Hurst: {ind['hurst_exp']:.3f}
Ichimoku: {'ABOVE cloud' if ind['ich_bull'] else 'BELOW cloud' if ind['ich_bear'] else 'INSIDE cloud'}
CMF: {ind['cmf']:+.3f} | Z-Score: {ind['price_zscore']:+.2f} | Entropy: {ind['entropy_ratio']:.3f}
VWAP dist: {ind['dist_vwap']:+.2f}% | Autocorr: {ind['autocorr']:+.3f}
{daily_ctx}

NEWS AGENT VERDICT:
Sentiment: {news['sentiment']} ({news['sentiment_score']:+d}/100)
Market regime: {news['market_regime']}
Key catalysts: {', '.join(news.get('key_catalysts', ['none'])[:3])}
Reasoning: {news.get('reasoning', 'N/A')}
Macro warning: {news.get('macro_warning') or 'none'}
{f"AGENT CONFLICT: Quant says {quant['direction']} but news says {'bearish' if quant['direction'] == 'BUY' else 'bullish'}. Resolve this conflict using the historical evidence below." if (quant['direction'] == 'BUY' and news['sentiment'] == 'bearish') or (quant['direction'] == 'SELL' and news['sentiment'] == 'bullish') else ''}

MONTE CARLO (1000 simulations):
Median: {mc['median']:.4f} | Bull(80th): {mc['bull']:.4f} | Bear(20th): {mc['bear']:.4f}
Probability up: {mc['prob_up']*100:.0f}%
{sim_ctx}

DECISION RULES:
1. Use ALL evidence — quant math, news sentiment, AND historical similarity.
2. Explain your step-by-step reasoning: "Quant said X because [indicators]. News said Y because [headlines]. Historical evidence shows Z. Therefore..."
3. When quant and news conflict, historical similarity is the tiebreaker.
4. Hurst < 0.45 = mean reverting market — trust RSI extremes over momentum.
5. High entropy (>0.6) = noisy market — require extra confidence.
6. News macro_warning present = reduce confidence 10pts and widen targets.
7. Output NO_TRADE if: genuine signal conflict with no historical resolution, OR confidence < 45%.
8. Price targets MUST be within {price_min:.4f} to {price_max:.4f}.
9. Use ATR ({ind['atr']:.4f}) for realistic path — not a straight line.

Respond with ONLY this JSON:
{{"decision":"<BUY|SELL|NO_TRADE>","prob_up":<0-100>,"prob_down":<0-100>,"confidence":<final 0-100>,"agent_agreement":"<agree|partial|conflict>","price_target":<realistic target>,"price_target_bull":<optimistic>,"price_target_bear":<pessimistic>,"predicted_path":[<5 prices zigzag to target>],"volatility":"<low|moderate|high>","insight":"<3-4 sentences: what quant found + what news found + historical evidence + final reasoning>","primary_reason":"<one clear sentence: the single most decisive factor>"}}"""

    # Try R1 first for long horizons or when explicitly requested
    if use_r1 and ds_key:
        try:
            async with httpx.AsyncClient(timeout=600) as client:  # 10 min timeout
                resp = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                    json={
                        "model": "deepseek-reasoner",
                        "max_tokens": 3000,
                        "stream": False,
                        "messages": [
                            {"role": "system", "content": "You are an expert trading AI. Respond ONLY with valid JSON."},
                            {"role": "user", "content": prompt}
                        ]
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                msg = data['choices'][0]['message']
                text = msg.get('content') or msg.get('reasoning_content') or ''
                text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()
                m = re.search(r'\{[\s\S]*\}', text)
                if m:
                    result = json.loads(m.group())
                    result['_model'] = 'deepseek-r1'
                    return result
        except Exception as e:
            print(f"R1 failed: {e}")

    # Fallback: GPT-4o
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o",
                    "max_tokens": 1200,
                    "messages": [
                        {"role": "system", "content": "You are an expert trading AI. Respond ONLY with valid JSON."},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            resp.raise_for_status()
            text = resp.json()['choices'][0]['message']['content']
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                result = json.loads(m.group())
                result['_model'] = 'gpt-4o'
                return result
    except Exception as e:
        print(f"GPT-4o failed: {e}")

    return {
        "decision": "NO_TRADE",
        "prob_up": 50, "prob_down": 50, "confidence": 0,
        "agent_agreement": "conflict",
        "price_target": None, "price_target_bull": None, "price_target_bear": None,
        "predicted_path": [],
        "volatility": "high",
        "insight": "All agents failed — cannot make prediction",
        "primary_reason": "API error",
        "_model": "error"
    }
