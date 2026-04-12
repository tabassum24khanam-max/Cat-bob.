"""
ULTRAMAX Sentiment Engine — Reddit, StockTwits, Google Trends
Each source degrades gracefully when API keys are missing
"""
import asyncio
import time
import httpx
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, is_configured, ASSET_NAMES
from database import save_sentiment_snapshot

# Asset-to-subreddit mapping
SUBREDDIT_MAP = {
    'BTC': ['cryptocurrency', 'bitcoin'],
    'ETH': ['cryptocurrency', 'ethereum'],
    'SOL': ['cryptocurrency', 'solana'],
    'BNB': ['cryptocurrency', 'binance'],
    'XRP': ['cryptocurrency', 'xrp'],
    'DOGE': ['cryptocurrency', 'dogecoin'],
    'AAPL': ['stocks', 'wallstreetbets', 'apple'],
    'TSLA': ['stocks', 'wallstreetbets', 'teslainvestorsclub'],
    'NVDA': ['stocks', 'wallstreetbets', 'nvidia_stock'],
    'MSFT': ['stocks', 'wallstreetbets'],
    'GOOGL': ['stocks', 'wallstreetbets'],
    'SPY': ['stocks', 'wallstreetbets', 'options'],
    'GC=F': ['stocks', 'gold', 'commodities'],
    'CL=F': ['stocks', 'commodities', 'oil'],
}

# StockTwits symbol mapping
STOCKTWITS_MAP = {
    'BTC': 'BTC.X', 'ETH': 'ETH.X', 'SOL': 'SOL.X',
    'BNB': 'BNB.X', 'XRP': 'XRP.X', 'DOGE': 'DOGE.X',
    'AAPL': 'AAPL', 'TSLA': 'TSLA', 'NVDA': 'NVDA',
    'MSFT': 'MSFT', 'GOOGL': 'GOOGL', 'SPY': 'SPY',
    'GC=F': 'GLD', 'CL=F': 'USO',
}

# Simple sentiment words (reuse from news_agent)
BULLISH_WORDS = {'bullish', 'surge', 'rally', 'gain', 'soar', 'jump', 'rise', 'buy', 'bull',
                  'upgrade', 'beat', 'exceed', 'strong', 'positive', 'growth', 'up', 'high',
                  'record', 'boost', 'recover', 'breakout', 'moon', 'pump', 'calls', 'long'}
BEARISH_WORDS = {'bearish', 'drop', 'fall', 'crash', 'decline', 'sell', 'bear', 'down',
                  'downgrade', 'miss', 'weak', 'negative', 'loss', 'low', 'cut', 'ban',
                  'sanction', 'warn', 'risk', 'plunge', 'tumble', 'fear', 'puts', 'short', 'dump'}


def simple_sentiment(text: str) -> float:
    """Rule-based sentiment scoring: -1.0 to +1.0."""
    words = set(text.lower().split())
    bull = len(words & BULLISH_WORDS)
    bear = len(words & BEARISH_WORDS)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


async def fetch_reddit_sentiment(asset: str) -> dict:
    """Fetch sentiment from Reddit via PRAW-style API. Requires Reddit API keys."""
    if not is_configured('REDDIT_CLIENT_ID') or not is_configured('REDDIT_CLIENT_SECRET'):
        return {'score': None, 'volume': 0, 'available': False}

    subreddits = SUBREDDIT_MAP.get(asset, ['stocks'])
    asset_name = ASSET_NAMES.get(asset, asset).lower()
    all_scores = []

    try:
        # Get Reddit access token
        async with httpx.AsyncClient(timeout=10) as client:
            auth_resp = await client.post(
                "https://www.reddit.com/api/v1/access_token",
                data={'grant_type': 'client_credentials'},
                auth=(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
                headers={'User-Agent': 'ULTRAMAX/3.0'}
            )
            if auth_resp.status_code != 200:
                return {'score': None, 'volume': 0, 'available': False}
            token = auth_resp.json().get('access_token')

        headers = {'Authorization': f'Bearer {token}', 'User-Agent': 'ULTRAMAX/3.0'}

        async with httpx.AsyncClient(timeout=10) as client:
            for sub in subreddits[:2]:  # Limit to 2 subreddits to avoid rate limits
                try:
                    resp = await client.get(
                        f"https://oauth.reddit.com/r/{sub}/hot",
                        headers=headers,
                        params={'limit': 25}
                    )
                    if resp.status_code != 200:
                        continue
                    posts = resp.json().get('data', {}).get('children', [])
                    for post in posts:
                        title = post.get('data', {}).get('title', '')
                        # Only score posts mentioning the asset
                        if asset.lower() in title.lower() or asset_name in title.lower():
                            score = simple_sentiment(title)
                            ups = post.get('data', {}).get('ups', 0)
                            all_scores.append(score * min(3, 1 + ups / 100))
                except Exception:
                    continue
                await asyncio.sleep(0.5)  # Rate limit respect

        if all_scores:
            avg = sum(all_scores) / len(all_scores)
            return {'score': round(avg, 3), 'volume': len(all_scores), 'available': True}
        return {'score': 0.0, 'volume': 0, 'available': True}

    except Exception:
        return {'score': None, 'volume': 0, 'available': False}


async def fetch_stocktwits_sentiment(asset: str) -> dict:
    """Fetch sentiment from StockTwits (free, no auth)."""
    symbol = STOCKTWITS_MAP.get(asset)
    if not symbol:
        return {'score': None, 'volume': 0, 'available': False}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json",
                headers={'User-Agent': 'ULTRAMAX/3.0'}
            )
            if resp.status_code != 200:
                return {'score': None, 'volume': 0, 'available': False}

            data = resp.json()
            messages = data.get('messages', [])
            if not messages:
                return {'score': 0.0, 'volume': 0, 'available': True}

            bull_count = 0
            bear_count = 0
            for msg in messages:
                sentiment = msg.get('entities', {}).get('sentiment', {})
                if sentiment:
                    if sentiment.get('basic') == 'Bullish':
                        bull_count += 1
                    elif sentiment.get('basic') == 'Bearish':
                        bear_count += 1

            total = bull_count + bear_count
            if total == 0:
                # Fall back to text analysis
                text_scores = [simple_sentiment(msg.get('body', '')) for msg in messages]
                avg = sum(text_scores) / len(text_scores) if text_scores else 0
                return {'score': round(avg, 3), 'volume': len(messages), 'available': True,
                        'bull_count': 0, 'bear_count': 0}

            score = (bull_count - bear_count) / total
            return {
                'score': round(score, 3),
                'volume': total,
                'available': True,
                'bull_count': bull_count,
                'bear_count': bear_count,
            }

    except Exception:
        return {'score': None, 'volume': 0, 'available': False}


async def fetch_google_trends(asset: str) -> dict:
    """Fetch Google Trends interest. Uses pytrends with caching."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        return {'score': None, 'spike_detected': False, 'available': False}

    try:
        keyword = ASSET_NAMES.get(asset, asset)
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(5, 10))
        pytrends.build_payload([keyword], timeframe='now 7-d')
        df = pytrends.interest_over_time()

        if df.empty:
            return {'score': 0.0, 'spike_detected': False, 'available': True}

        values = df[keyword].tolist()
        if not values:
            return {'score': 0.0, 'spike_detected': False, 'available': True}

        current = values[-1]
        avg_7d = sum(values) / len(values) if values else 1
        normalized = (current - avg_7d) / (avg_7d or 1)
        spike = current > avg_7d * 2  # Spike if 2x average

        return {
            'score': round(normalized, 3),
            'current_interest': current,
            'avg_7d': round(avg_7d, 1),
            'spike_detected': spike,
            'available': True,
        }

    except Exception:
        return {'score': None, 'spike_detected': False, 'available': False}


async def get_sentiment_snapshot(asset: str) -> dict:
    """Aggregate sentiment from all available sources."""
    reddit, stocktwits, trends = await asyncio.gather(
        fetch_reddit_sentiment(asset),
        fetch_stocktwits_sentiment(asset),
        fetch_google_trends(asset),
    )

    # Weighted average of available scores
    scores = []
    weights = []

    if reddit.get('available') and reddit.get('score') is not None:
        scores.append(reddit['score'])
        weights.append(0.3)

    if stocktwits.get('available') and stocktwits.get('score') is not None:
        scores.append(stocktwits['score'])
        weights.append(0.4)

    if trends.get('available') and trends.get('score') is not None:
        scores.append(trends['score'])
        weights.append(0.3)

    if scores and weights:
        total_weight = sum(weights)
        combined = sum(s * w for s, w in zip(scores, weights)) / total_weight
    else:
        combined = 0.0

    result = {
        'reddit': reddit,
        'stocktwits': stocktwits,
        'google_trends': trends,
        'combined_score': round(combined, 3),
        'sources_available': sum(1 for s in [reddit, stocktwits, trends] if s.get('available')),
    }

    # Save to database
    try:
        ts = int(time.time())
        ts = (ts // 3600) * 3600  # Round to hour
        await save_sentiment_snapshot(asset, ts, {
            'reddit_score': reddit.get('score'),
            'reddit_volume': reddit.get('volume', 0),
            'stocktwits_score': stocktwits.get('score'),
            'stocktwits_volume': stocktwits.get('volume', 0),
            'google_trends_score': trends.get('score'),
            'combined_score': combined,
        })
    except Exception:
        pass

    return result
