"""ULTRAMAX Order Book — Free Binance depth data"""
import httpx
import numpy as np

BINANCE_SYMBOLS = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT',
    'BNB': 'BNBUSDT', 'XRP': 'XRPUSDT', 'DOGE': 'DOGEUSDT',
}


async def get_orderbook_imbalance(asset: str) -> dict:
    """Fetch order book and compute bid/ask imbalance.
    Returns: {available, bid_volume, ask_volume, imbalance_ratio,
              large_walls: [{side, price, size}], bias: 'bullish'|'bearish'|'neutral'}
    """
    if asset not in BINANCE_SYMBOLS:
        return {'available': False}

    symbol = BINANCE_SYMBOLS[asset]

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=100"
            )
            if resp.status_code != 200:
                return {'available': False}
            data = resp.json()

        bids = data.get('bids', [])
        asks = data.get('asks', [])

        if not bids or not asks:
            return {'available': False}

        # Parse bid/ask arrays: [[price, quantity], ...]
        bid_prices = [float(b[0]) for b in bids]
        bid_sizes = [float(b[1]) for b in bids]
        ask_prices = [float(a[0]) for a in asks]
        ask_sizes = [float(a[1]) for a in asks]

        bid_volume = float(np.sum(bid_sizes))
        ask_volume = float(np.sum(ask_sizes))

        total = bid_volume + ask_volume
        if total == 0:
            return {'available': False}

        imbalance_ratio = round((bid_volume - ask_volume) / total, 4)

        # Detect large walls: orders > 3x the average order size
        all_sizes = bid_sizes + ask_sizes
        avg_size = float(np.mean(all_sizes)) if all_sizes else 0
        wall_threshold = avg_size * 3

        large_walls = []
        for price, size in zip(bid_prices, bid_sizes):
            if size > wall_threshold:
                large_walls.append({
                    'side': 'bid',
                    'price': round(price, 8),
                    'size': round(size, 8),
                })
        for price, size in zip(ask_prices, ask_sizes):
            if size > wall_threshold:
                large_walls.append({
                    'side': 'ask',
                    'price': round(price, 8),
                    'size': round(size, 8),
                })

        # Sort walls by size descending, keep top 10
        large_walls.sort(key=lambda w: w['size'], reverse=True)
        large_walls = large_walls[:10]

        # Determine bias
        if imbalance_ratio > 0.1:
            bias = 'bullish'
        elif imbalance_ratio < -0.1:
            bias = 'bearish'
        else:
            bias = 'neutral'

        return {
            'available': True,
            'bid_volume': round(bid_volume, 4),
            'ask_volume': round(ask_volume, 4),
            'imbalance_ratio': imbalance_ratio,
            'large_walls': large_walls,
            'bias': bias,
        }

    except Exception:
        return {'available': False}
