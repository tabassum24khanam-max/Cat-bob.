"""
ULTRAMAX Data Fetcher — Consolidated data fetching from all sources
Binance, Yahoo, Worker proxy, FRED, CoinGecko
"""
import asyncio
import httpx
from config import BINANCE_SYMBOLS, YAHOO_SYMBOLS, WORKER_URL, FRED_API_KEY, is_configured


async def fetch_binance_candles(symbol: str, interval: str = '1h', limit: int = 300) -> list:
    """Fetch candles from Binance API."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://api.binance.com/api/v3/klines",
            params={'symbol': symbol, 'interval': interval, 'limit': limit}
        )
        data = resp.json()
        return [{'time': int(k[0]) // 1000, 'open': float(k[1]), 'high': float(k[2]),
                 'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])} for k in data]


async def fetch_yahoo_candles(symbol: str, interval: str = '1h', limit: int = 300,
                               worker_url: str = None) -> list:
    """Fetch candles via Cloudflare Worker or direct Yahoo."""
    wurl = worker_url or WORKER_URL
    if wurl:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{wurl}/candles", params={'sym': symbol, 'iv': interval})
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get('candles', [])[-limit:]
        except Exception:
            pass

    # Direct Yahoo fallback
    try:
        range_map = {'1h': '60d', '4h': '3mo', '1d': '2y'}
        yrange = range_map.get(interval, '60d')
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={'range': yrange, 'interval': interval},
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            d = resp.json()
            res = d.get('chart', {}).get('result', [{}])[0]
            q = res.get('indicators', {}).get('quote', [{}])[0]
            ts_list = res.get('timestamp', [])
            candles = []
            for i, t in enumerate(ts_list):
                if q.get('close', [None])[i] is not None:
                    candles.append({'time': t, 'open': q['open'][i], 'high': q['high'][i],
                                   'low': q['low'][i], 'close': q['close'][i],
                                   'volume': (q.get('volume') or [0])[i] or 0})
            return candles[-limit:]
    except Exception:
        return []


async def fetch_candles(asset: str, interval: str = '1h', limit: int = 300,
                         worker_url: str = None) -> list:
    """Unified candle fetcher — picks Binance or Yahoo based on asset."""
    if asset in BINANCE_SYMBOLS:
        return await fetch_binance_candles(BINANCE_SYMBOLS[asset], interval, limit)

    symbol = YAHOO_SYMBOLS.get(asset, asset)
    return await fetch_yahoo_candles(symbol, interval, limit, worker_url)


async def fetch_macro(worker_url: str = None) -> dict:
    """Fetch macro data (VIX, DXY, etc.) from Worker."""
    wurl = worker_url or WORKER_URL
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{wurl}/macro")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return {}


async def fetch_fear_greed() -> dict:
    """Fetch Crypto Fear & Greed Index."""
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get("https://api.alternative.me/fng/?limit=2")
            d = resp.json()
            cur = d['data'][0]
            return {'value': int(cur['value']), 'label': cur['value_classification']}
    except Exception:
        return {'value': 50, 'label': 'Neutral'}


async def fetch_onchain(asset: str) -> dict:
    """Fetch on-chain data (funding rate, OI, long/short) from Binance Futures."""
    if asset not in BINANCE_SYMBOLS:
        return {}
    try:
        sym = BINANCE_SYMBOLS[asset]
        async with httpx.AsyncClient(timeout=10) as client:
            tasks = [
                client.get(f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={sym}"),
                client.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={sym}"),
                client.get(f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={sym}&period=1h&limit=1"),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        if not isinstance(results[0], Exception):
            fr_data = results[0].json()
            data['funding_rate'] = float(fr_data.get('lastFundingRate', 0)) * 100
        if not isinstance(results[1], Exception):
            oi_data = results[1].json()
            data['open_interest'] = float(oi_data.get('openInterest', 0))
        if not isinstance(results[2], Exception):
            ls_data = results[2].json()
            if ls_data:
                data['long_short_ratio'] = float(ls_data[0].get('longShortRatio', 1))
        return data
    except Exception:
        return {}


async def fetch_fred_data(series_id: str) -> float:
    """Fetch latest value from FRED API. Returns None if key not set."""
    if not is_configured('FRED_API_KEY'):
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    'series_id': series_id,
                    'api_key': FRED_API_KEY,
                    'file_type': 'json',
                    'sort_order': 'desc',
                    'limit': 1
                }
            )
            if resp.status_code == 200:
                obs = resp.json().get('observations', [])
                if obs and obs[0].get('value') != '.':
                    return float(obs[0]['value'])
    except Exception:
        pass
    return None


async def fetch_fred_macro() -> dict:
    """Fetch key macro indicators from FRED."""
    if not is_configured('FRED_API_KEY'):
        return {}

    series = {
        'VIXCLS': 'vix',
        'DTWEXBGS': 'dxy',
        'DGS10': 'ten_year_yield',
        'FEDFUNDS': 'fed_rate',
        'CPIAUCSL': 'cpi',
    }
    results = {}
    tasks = {name: fetch_fred_data(sid) for sid, name in series.items()}
    for sid, name in series.items():
        val = await fetch_fred_data(sid)
        if val is not None:
            results[name] = val
    return results


async def fetch_coingecko_dominance() -> dict:
    """Fetch BTC dominance and total market cap from CoinGecko."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.coingecko.com/api/v3/global")
            if resp.status_code == 200:
                data = resp.json().get('data', {})
                return {
                    'btc_dominance': data.get('market_cap_percentage', {}).get('btc', 0),
                    'total_market_cap': data.get('total_market_cap', {}).get('usd', 0),
                    'total_volume': data.get('total_volume', {}).get('usd', 0),
                }
    except Exception:
        pass
    return {}


async def fetch_current_price(asset: str, worker_url: str = None) -> float:
    """Fetch current price for a single asset."""
    if asset in BINANCE_SYMBOLS:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(
                f"https://api.binance.com/api/v3/ticker/price?symbol={BINANCE_SYMBOLS[asset]}"
            )
            return float(resp.json()['price'])

    wurl = worker_url or WORKER_URL
    async with httpx.AsyncClient(timeout=8) as client:
        resp = await client.get(f"{wurl}/price?sym={asset}")
        return resp.json()['price']
