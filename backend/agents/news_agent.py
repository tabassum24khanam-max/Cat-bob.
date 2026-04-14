"""
ULTRAMAX News Agent
Persistent sentiment memory, multi-source news, FinBERT scoring
"""
import asyncio
import httpx
import json
import re
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
import feedparser


FEEDS = {
    'crypto': [
        {'url': 'https://cointelegraph.com/rss', 'tier': 0.6, 'name': 'CoinTelegraph'},
        {'url': 'https://coindesk.com/arc/outboundfeeds/rss/', 'tier': 0.6, 'name': 'CoinDesk'},
        {'url': 'https://decrypt.co/feed', 'tier': 0.6, 'name': 'Decrypt'},
        {'url': 'https://blockworks.co/feed', 'tier': 0.7, 'name': 'Blockworks'},
    ],
    'stock': [
        {'url': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml', 'tier': 0.8, 'name': 'WSJ'},
        {'url': 'https://feeds.reuters.com/reuters/businessNews', 'tier': 0.8, 'name': 'Reuters'},
    ],
    'macro': [
        {'url': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml', 'tier': 0.8, 'name': 'WSJ'},
        {'url': 'https://feeds.reuters.com/reuters/businessNews', 'tier': 0.8, 'name': 'Reuters'},
    ],
    'universal': [
        {'url': 'https://news.google.com/rss/search?q=Federal+Reserve+FOMC+rates&hl=en-US&gl=US', 'tier': 1.0, 'name': 'FedWatch'},
        {'url': 'https://news.google.com/rss/search?q=CPI+inflation+PCE+employment&hl=en-US&gl=US', 'tier': 0.9, 'name': 'EconData'},
    ]
}


# Simple rule-based sentiment scorer (fallback when FinBERT not available)
BULLISH_WORDS = {'bullish', 'surge', 'rally', 'gain', 'soar', 'jump', 'rise', 'buy', 'bull',
                  'upgrade', 'beat', 'exceed', 'strong', 'positive', 'growth', 'up', 'high',
                  'record', 'boost', 'recover', 'breakout', 'approval'}
BEARISH_WORDS = {'bearish', 'drop', 'fall', 'crash', 'decline', 'sell', 'bear', 'down',
                  'downgrade', 'miss', 'weak', 'negative', 'loss', 'low', 'cut', 'ban',
                  'sanction', 'warn', 'risk', 'concern', 'plunge', 'tumble', 'fear'}

def rule_based_sentiment(text: str) -> float:
    words = set(text.lower().split())
    bull = len(words & BULLISH_WORDS)
    bear = len(words & BEARISH_WORDS)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


# Try to use FinBERT for proper financial NLP
_finbert = None
def get_finbert():
    global _finbert
    if _finbert is None:
        try:
            from transformers import pipeline
            _finbert = pipeline("sentiment-analysis",
                                 model="ProsusAI/finbert",
                                 max_length=512, truncation=True)
            print("✓ FinBERT loaded")
        except Exception as e:
            print(f"⚠ FinBERT unavailable ({e}), using rule-based sentiment")
            _finbert = False
    return _finbert if _finbert else None


def score_sentiment(text: str) -> float:
    """Returns -1.0 to +1.0"""
    model = get_finbert()
    if model:
        try:
            result = model(text[:512])[0]
            if result['label'] == 'positive': return result['score']
            if result['label'] == 'negative': return -result['score']
            return 0.0
        except:
            pass
    return rule_based_sentiment(text)


# Impact scoring heuristics
IMPACT_KEYWORDS = {
    'high': ['fed', 'fomc', 'sec', 'cftc', 'ban', 'regulation', 'earnings', 'cpi', 'inflation',
             'sanctions', 'opec', 'iran', 'war', 'bankruptcy', 'hack', 'exploit'],
    'medium': ['analyst', 'upgrade', 'downgrade', 'partnership', 'launch', 'acquisition'],
    'low': ['price', 'market', 'trading', 'investor', 'crypto']
}

def score_impact(headline: str) -> float:
    h = headline.lower()
    if any(w in h for w in IMPACT_KEYWORDS['high']): return 0.8
    if any(w in h for w in IMPACT_KEYWORDS['medium']): return 0.5
    return 0.3


CATEGORY_KEYWORDS = {
    'macro': ['fed', 'fomc', 'inflation', 'cpi', 'gdp', 'jobs', 'yields', 'treasury', 'powell'],
    'regulatory': ['sec', 'cftc', 'ban', 'regulation', 'compliance', 'lawsuit', 'settlement'],
    'earnings': ['earnings', 'revenue', 'profit', 'eps', 'guidance', 'quarter', 'results'],
    'geopolitical': ['iran', 'russia', 'china', 'war', 'sanctions', 'opec', 'nato', 'conflict'],
    'sentiment': ['fear', 'greed', 'bullish', 'bearish', 'retail', 'whale', 'institutional'],
}

def classify_category(headline: str) -> str:
    h = headline.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(w in h for w in keywords):
            return cat
    return 'general'


async def fetch_feed(session: httpx.AsyncClient, feed_info: dict) -> list:
    """Fetch a single RSS feed and return list of articles."""
    try:
        resp = await session.get(feed_info['url'], timeout=8.0, follow_redirects=True)
        if resp.status_code != 200:
            return []
        parsed = feedparser.parse(resp.text)
        articles = []
        for entry in parsed.entries[:8]:
            headline = entry.get('title', '').strip()
            if len(headline) < 15:
                continue
            articles.append({
                'headline': headline,
                'source': feed_info['name'],
                'tier': feed_info['tier'],
                'sentiment': score_sentiment(headline),
                'impact': score_impact(headline),
                'category': classify_category(headline),
            })
        return articles
    except Exception:
        return []


async def fetch_asset_news(asset: str, asset_name: str, asset_type: str) -> list:
    """Fetch news from all relevant sources for an asset."""
    is_crypto = asset_type == 'crypto'
    
    feed_list = []
    feed_list.extend(FEEDS.get(asset_type, FEEDS['stock']))
    feed_list.extend(FEEDS['universal'])
    
    # Asset-specific Google News feeds
    encoded = asset_name.replace(' ', '+')
    feed_list.append({
        'url': f'https://news.google.com/rss/search?q={encoded}+price+market&hl=en-US&gl=US',
        'tier': 0.5, 'name': 'GoogleNews-Asset'
    })
    if is_crypto:
        feed_list.append({
            'url': 'https://news.google.com/rss/search?q=Bitcoin+crypto+regulation+SEC&hl=en-US&gl=US',
            'tier': 0.6, 'name': 'GoogleNews-Crypto'
        })
        feed_list.append({
            'url': 'https://news.google.com/rss/search?q=crypto+macro+interest+rates+liquidity&hl=en-US&gl=US',
            'tier': 0.5, 'name': 'GoogleNews-CryptoMacro'
        })
        feed_list.append({
            'url': f'https://news.google.com/rss/search?q={encoded}+whale+sentiment+social&hl=en-US&gl=US',
            'tier': 0.3, 'name': 'GoogleNews-Social'
        })
    elif asset_type == 'macro':
        # Oil/Gold/Commodities — geopolitical feeds
        feed_list.append({
            'url': 'https://news.google.com/rss/search?q=OPEC+Iran+oil+supply+geopolitical&hl=en-US&gl=US',
            'tier': 0.7, 'name': 'GoogleNews-Geopolitical'
        })
        feed_list.append({
            'url': f'https://news.google.com/rss/search?q={encoded}+supply+demand+inventory&hl=en-US&gl=US',
            'tier': 0.6, 'name': 'GoogleNews-Commodity'
        })
    else:
        feed_list.append({
            'url': f'https://news.google.com/rss/search?q={asset}+earnings+analyst+insider&hl=en-US&gl=US',
            'tier': 0.7, 'name': 'GoogleNews-Stock'
        })
        feed_list.append({
            'url': f'https://news.google.com/rss/search?q={asset}+unusual+options+activity+upgrade&hl=en-US&gl=US',
            'tier': 0.6, 'name': 'GoogleNews-StockFlow'
        })

    async with httpx.AsyncClient() as client:
        tasks = [fetch_feed(client, f) for f in feed_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    articles = []
    seen = set()
    for result in results:
        if isinstance(result, list):
            for a in result:
                if a['headline'] not in seen:
                    seen.add(a['headline'])
                    articles.append(a)

    # Sort by tier * impact descending
    articles.sort(key=lambda a: a['tier'] * a['impact'], reverse=True)
    return articles[:30]


async def filter_headlines_ai(articles: list, asset: str, asset_name: str,
                               api_key: str, ds_key: str = None) -> list:
    """Use AI to filter to 8 most impactful headlines."""
    if len(articles) <= 8:
        return articles

    headlines_text = "\n".join(
        f"{i+1}. [{a['source']}|tier{a['tier']}] {a['headline']}"
        for i, a in enumerate(articles)
    )

    prompt = f"""Asset: {asset} ({asset_name}).
Filter to 8 most market-moving headlines for price prediction.
Tiers: 1.0=Fed/SEC official, 0.8=Reuters/WSJ, 0.7=specialist, 0.6=media, 0.3=social.
Keep: earnings surprises, regulatory action, insider buying, Fed statements, supply shocks, analyst calls.
Drop: minor commentary, duplicates, unrelated.
Return ONLY JSON array of indices (0-based): [0,3,5,...]
Headlines:
{headlines_text}"""

    try:
        if ds_key:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                    json={
                        "model": "deepseek-chat",
                        "max_tokens": 100,
                        "temperature": 0.1,
                        "messages": [
                            {"role": "system", "content": "Return ONLY a JSON array of integers."},
                            {"role": "user", "content": prompt}
                        ]
                    }
                )
                text = resp.json()['choices'][0]['message']['content']
                m = re.search(r'\[[\d,\s]+\]', text)
                if m:
                    indices = json.loads(m.group())
                    return [articles[i] for i in indices if i < len(articles)][:8]
    except:
        pass

    # Fallback: top 8 by tier * impact
    return articles[:8]


async def run_news_agent(asset: str, asset_name: str, asset_type: str,
                          articles: list, macro_data: dict, onchain_data: dict,
                          fg_data: dict, fr_data: dict, horizon: int,
                          api_key: str, ds_key: str = None,
                          db_sentiment: dict = None) -> dict:
    """Call DeepSeek V3 News Agent with full context."""

    # Aggregate article metrics
    avg_sentiment = sum(a['sentiment'] for a in articles) / len(articles) if articles else 0
    avg_impact = sum(a['impact'] for a in articles) / len(articles) if articles else 0
    headlines_str = "\n".join(f"- [{a['source']}] {a['headline']}" for a in articles[:8])

    # Build sentiment history context from database
    db_ctx = ""
    if db_sentiment and db_sentiment.get('hours'):
        avg_24h = db_sentiment.get('avg_24h', 0)
        trend = db_sentiment.get('trend', 0)
        historical_analog = db_sentiment.get('analog', {})
        db_ctx = f"""
SENTIMENT MEMORY (from database):
24h average sentiment: {avg_24h:.2f} ({'improving' if trend > 0 else 'worsening' if trend < 0 else 'stable'})
Sentiment momentum: {trend:+.3f}/hour
Historical analog: {historical_analog.get('description', 'insufficient data')}
"""

    macro_ctx = ""
    if macro_data:
        vix = macro_data.get('vix', 0)
        dxy = macro_data.get('dxy', 0)
        macro_ctx = f"""
MACRO ENVIRONMENT:
VIX: {vix:.1f} {'⚠ HIGH FEAR' if vix > 25 else '(normal)'}
DXY: {dxy:.2f} {'(strong dollar)' if dxy > 105 else '(weak dollar)' if dxy < 100 else ''}
Fear & Greed: {fg_data.get('value', 50)}/100 ({fg_data.get('label', 'neutral')})
"""

    onchain_ctx = ""
    if onchain_data and asset_type == 'crypto':
        onchain_ctx = f"""
ON-CHAIN ({asset}):
Funding Rate: {onchain_data.get('funding_rate', 0):.4f}% {'⚠ CROWDED LONGS' if (onchain_data.get('funding_rate', 0) or 0) > 0.05 else ''}
Long/Short Ratio: {onchain_data.get('long_short_ratio', 1):.2f}
"""

    prompt = f"""You are a professional financial news analyst with expertise in geopolitics, macro economics, and market microstructure. Respond ONLY with valid JSON.

ASSET: {asset} ({asset_name}) | HORIZON: {horizon}h
{db_ctx}
HEADLINES (tier-weighted, most impactful first):
{headlines_str}

Average headline sentiment: {avg_sentiment:+.2f} | Average impact: {avg_impact:.2f}
{macro_ctx}{onchain_ctx}

RULES:
- Tier 1.0 sources (Fed/SEC) dominate if present.
- Social/Reddit (tier 0.3) only matters if velocity is extreme.
- Breaking regulatory news overrides all technical analysis.
- Neutral news = no adjustment. Do NOT force sentiment from noise.
- Max confidence adjustment from news: ±15 points.

Respond ONLY with this JSON:
{{"sentiment":"<bullish|neutral|bearish>","sentiment_score":<-100 to 100>,"confidence":<0-100>,"market_regime":"<risk_on|risk_off|neutral>","key_catalysts":["<event1>","<event2>"],"time_bias":"<positive|negative|neutral>","reasoning":"<1 sentence: what news says and why>","macro_warning":"<high-impact upcoming event or null>","event_impact_score":<0-1>}}"""

    try:
        # Try DeepSeek V3 first
        if ds_key:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                    json={
                        "model": "deepseek-chat",
                        "max_tokens": 500,
                        "temperature": 0.1,
                        "messages": [
                            {"role": "system", "content": "You are a professional financial news analyst. Respond ONLY with valid JSON."},
                            {"role": "user", "content": prompt}
                        ]
                    }
                )
                resp.raise_for_status()
                text = resp.json()['choices'][0]['message']['content']
                m = re.search(r'\{[\s\S]*\}', text)
                if m:
                    return json.loads(m.group())

    except Exception as e:
        pass

    # Fallback to GPT-4o-mini
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            resp.raise_for_status()
            text = resp.json()['choices'][0]['message']['content']
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                return json.loads(m.group())
    except Exception as e:
        print(f"⚠ News agent error: {e}")

    return {
        "sentiment": "neutral", "sentiment_score": 0, "confidence": 50,
        "market_regime": "neutral", "key_catalysts": [], "time_bias": "neutral",
        "reasoning": f"News analysis unavailable", "macro_warning": None, "event_impact_score": 0.3
    }
