"""
ULTRAMAX Decision Agent — DeepSeek R1
Resolves conflict between Quant and News with historical evidence
"""
import asyncio
import httpx
import json
import re
import traceback
from typing import Dict, Any, Optional

# Track R1 failures for logging
_last_r1_error = None


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

    # Try R1 with retry — R1 is the primary decision maker
    global _last_r1_error
    if use_r1 and ds_key:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                print(f"R1 attempt {attempt}/{max_retries} for {asset}...")
                async with httpx.AsyncClient(timeout=httpx.Timeout(
                    connect=30.0, read=600.0, write=30.0, pool=30.0
                )) as client:
                    resp = await client.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                        json={
                            "model": "deepseek-reasoner",
                            "max_tokens": 8000,
                            "stream": False,
                            "messages": [
                                {"role": "user", "content": prompt}
                            ]
                        }
                    )

                    # Handle rate limiting with backoff
                    if resp.status_code == 429:
                        wait = min(30, 5 * attempt)
                        print(f"R1 rate limited, waiting {wait}s before retry...")
                        await asyncio.sleep(wait)
                        continue

                    if resp.status_code == 503 or resp.status_code == 502:
                        wait = 10 * attempt
                        print(f"R1 server error {resp.status_code}, waiting {wait}s before retry...")
                        await asyncio.sleep(wait)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    msg = data['choices'][0]['message']

                    # R1 returns answer in 'content', reasoning chain in 'reasoning_content'
                    # Try content first (final answer), then reasoning_content
                    text = msg.get('content', '') or ''
                    reasoning = msg.get('reasoning_content', '') or ''

                    # Extract JSON from content first
                    result = _extract_json(text)
                    if not result and reasoning:
                        # Sometimes R1 puts the JSON inside reasoning_content
                        result = _extract_json(reasoning)

                    if result:
                        result['_model'] = 'deepseek-r1'
                        _last_r1_error = None
                        print(f"R1 success on attempt {attempt}: {result.get('decision')} {result.get('confidence')}%")
                        return result
                    else:
                        # R1 responded but no valid JSON found
                        preview = (text or reasoning)[:200]
                        print(f"R1 attempt {attempt}: no JSON found in response. Preview: {preview}")
                        _last_r1_error = f"No JSON in R1 response (attempt {attempt})"
                        if attempt < max_retries:
                            await asyncio.sleep(3)
                            continue

            except httpx.ReadTimeout:
                _last_r1_error = f"R1 read timeout on attempt {attempt}"
                print(f"R1 timeout attempt {attempt}/{max_retries}")
                if attempt < max_retries:
                    await asyncio.sleep(5)
                    continue
            except Exception as e:
                _last_r1_error = f"R1 error attempt {attempt}: {str(e)[:100]}"
                print(f"R1 failed attempt {attempt}/{max_retries}: {e}")
                traceback.print_exc()
                if attempt < max_retries:
                    await asyncio.sleep(5 * attempt)
                    continue

        print(f"R1 exhausted all {max_retries} retries. Last error: {_last_r1_error}. Falling back.")

    # DeepSeek V3 (fast mode) — used when R1 is off or R1 failed
    if ds_key:
        try:
            print(f"Using DeepSeek V3 (fast) for {asset}...")
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=15.0, read=60.0, write=15.0, pool=15.0)) as client:
                resp = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                    json={
                        "model": "deepseek-chat",
                        "max_tokens": 3000,
                        "messages": [
                            {"role": "system", "content": "You are an expert trading AI. Respond ONLY with valid JSON."},
                            {"role": "user", "content": prompt}
                        ]
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data['choices'][0]['message'].get('content', '')
                    result = _extract_json(text)
                    if result:
                        result['_model'] = 'deepseek-v3'
                        print(f"V3 success for {asset}: {result.get('decision')} {result.get('confidence')}%")
                        return result
                else:
                    print(f"V3 HTTP {resp.status_code}")
        except Exception as e:
            print(f"V3 failed: {e}")

    # Last fallback: GPT-4o
    print(f"⚠ Using GPT-4o fallback for {asset}")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o",
                    "max_tokens": 2000,
                    "messages": [
                        {"role": "system", "content": "You are an expert trading AI. Respond ONLY with valid JSON."},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            resp.raise_for_status()
            text = resp.json()['choices'][0]['message']['content']
            result = _extract_json(text)
            if result:
                result['_model'] = 'gpt-4o'
                result['_r1_error'] = _last_r1_error or 'R1 not attempted'
                return result
    except Exception as e:
        print(f"GPT-4o also failed: {e}")

    return {
        "decision": "NO_TRADE",
        "prob_up": 50, "prob_down": 50, "confidence": 0,
        "agent_agreement": "conflict",
        "price_target": None, "price_target_bull": None, "price_target_bear": None,
        "predicted_path": [],
        "volatility": "high",
        "insight": f"All agents failed — R1: {_last_r1_error or 'no key'}, GPT-4o: also failed",
        "primary_reason": "API error",
        "_model": "error"
    }


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON object from text, handling markdown code blocks and think tags."""
    if not text:
        return None
    # Strip think tags
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()
    # Strip markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Find JSON object in text
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None
