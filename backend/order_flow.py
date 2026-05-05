"""
Tier 1.5: Order Flow — bid-ask imbalance and trade aggression from Binance depth
Who is buying vs selling RIGHT NOW. Not candles (summary), but live intent.
"""
import asyncio
import time
import httpx
from typing import Dict, List
from config import BINANCE_SYMBOLS

BINANCE_API = "https://api.binance.com"
_cache: Dict[str, dict] = {}
_cache_ttl = 30  # 30 seconds — order flow is fast-moving


async def get_order_flow(asset: str) -> dict:
    """Analyze order book depth imbalance + recent aggressive trades."""
    symbol = BINANCE_SYMBOLS.get(asset)
    if not symbol:
        return {'available': False, 'reason': 'not crypto'}

    cache_key = f"flow_{symbol}"
    if cache_key in _cache and time.time() - _cache[cache_key]['ts'] < _cache_ttl:
        return _cache[cache_key]['data']

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            depth_resp, trades_resp = await asyncio.gather(
                client.get(f"{BINANCE_API}/api/v3/depth",
                           params={'symbol': symbol, 'limit': 100}),
                client.get(f"{BINANCE_API}/api/v3/trades",
                           params={'symbol': symbol, 'limit': 500}),
            )

            depth = _analyze_depth(depth_resp.json() if depth_resp.status_code == 200 else {})
            aggression = _analyze_aggression(trades_resp.json() if trades_resp.status_code == 200 else [])

            # Combined signal
            bias = 0
            reasons = []

            if depth['available']:
                if depth['bid_wall']:
                    bias += 2
                    reasons.append(f"Bid wall at {depth['bid_wall_price']:.2f} ({depth['bid_wall_size']:.0f} units)")
                if depth['ask_wall']:
                    bias -= 2
                    reasons.append(f"Ask wall at {depth['ask_wall_price']:.2f} ({depth['ask_wall_size']:.0f} units)")
                if depth['imbalance_ratio'] > 1.5:
                    bias += 1
                    reasons.append(f"Bid-heavy orderbook ({depth['imbalance_ratio']:.2f}x)")
                elif depth['imbalance_ratio'] < 0.67:
                    bias -= 1
                    reasons.append(f"Ask-heavy orderbook ({depth['imbalance_ratio']:.2f}x)")

            if aggression['available']:
                if aggression['buy_aggression_pct'] > 60:
                    bias += 2
                    reasons.append(f"Buy aggression {aggression['buy_aggression_pct']:.0f}%")
                elif aggression['sell_aggression_pct'] > 60:
                    bias -= 2
                    reasons.append(f"Sell aggression {aggression['sell_aggression_pct']:.0f}%")
                if aggression['large_buy_volume'] > aggression['large_sell_volume'] * 2:
                    bias += 1
                    reasons.append("Large buyers dominating")
                elif aggression['large_sell_volume'] > aggression['large_buy_volume'] * 2:
                    bias -= 1
                    reasons.append("Large sellers dominating")

            result = {
                'available': True,
                'depth': depth,
                'aggression': aggression,
                'bias': max(-5, min(5, bias)),
                'signal': 'BUY' if bias >= 2 else 'SELL' if bias <= -2 else 'NEUTRAL',
                'reasons': reasons,
                'ts': int(time.time()),
            }
            _cache[cache_key] = {'data': result, 'ts': time.time()}
            return result
    except Exception as e:
        return {'available': False, 'reason': str(e)[:80]}


def _analyze_depth(depth_data: dict) -> dict:
    """Analyze orderbook depth for walls and imbalance."""
    bids = depth_data.get('bids', [])
    asks = depth_data.get('asks', [])

    if not bids or not asks:
        return {'available': False}

    bid_volumes = [(float(b[0]), float(b[1])) for b in bids[:50]]
    ask_volumes = [(float(a[0]), float(a[1])) for a in asks[:50]]

    total_bid = sum(v for _, v in bid_volumes)
    total_ask = sum(v for _, v in ask_volumes)
    imbalance = total_bid / total_ask if total_ask > 0 else 1.0

    # Detect walls (single level > 3x average)
    avg_bid = total_bid / len(bid_volumes) if bid_volumes else 0
    avg_ask = total_ask / len(ask_volumes) if ask_volumes else 0

    bid_wall = None
    bid_wall_price = 0
    bid_wall_size = 0
    for price, vol in bid_volumes:
        if vol > avg_bid * 3:
            bid_wall = True
            bid_wall_price = price
            bid_wall_size = vol
            break

    ask_wall = None
    ask_wall_price = 0
    ask_wall_size = 0
    for price, vol in ask_volumes:
        if vol > avg_ask * 3:
            ask_wall = True
            ask_wall_price = price
            ask_wall_size = vol
            break

    spread = ask_volumes[0][0] - bid_volumes[0][0] if bid_volumes and ask_volumes else 0
    spread_pct = spread / bid_volumes[0][0] * 100 if bid_volumes and bid_volumes[0][0] else 0

    return {
        'available': True,
        'total_bid_vol': round(total_bid, 2),
        'total_ask_vol': round(total_ask, 2),
        'imbalance_ratio': round(imbalance, 3),
        'bid_wall': bool(bid_wall),
        'bid_wall_price': bid_wall_price,
        'bid_wall_size': bid_wall_size,
        'ask_wall': bool(ask_wall),
        'ask_wall_price': ask_wall_price,
        'ask_wall_size': ask_wall_size,
        'spread_pct': round(spread_pct, 4),
    }


def _analyze_aggression(trades: list) -> dict:
    """Analyze recent trades for buy/sell aggression and large orders."""
    if not trades:
        return {'available': False}

    buy_vol = 0
    sell_vol = 0
    large_buy = 0
    large_sell = 0

    volumes = [float(t.get('qty', 0)) * float(t.get('price', 0)) for t in trades]
    avg_trade_usd = sum(volumes) / len(volumes) if volumes else 0
    large_threshold = avg_trade_usd * 5

    for t in trades:
        qty = float(t.get('qty', 0))
        price = float(t.get('price', 0))
        usd = qty * price
        is_buyer_maker = t.get('isBuyerMaker', False)

        if is_buyer_maker:
            sell_vol += usd  # taker sold into bid
            if usd > large_threshold:
                large_sell += usd
        else:
            buy_vol += usd  # taker bought into ask
            if usd > large_threshold:
                large_buy += usd

    total = buy_vol + sell_vol
    buy_pct = buy_vol / total * 100 if total else 50
    sell_pct = sell_vol / total * 100 if total else 50

    # CVD (Cumulative Volume Delta)
    cvd = buy_vol - sell_vol

    return {
        'available': True,
        'buy_volume_usd': round(buy_vol, 2),
        'sell_volume_usd': round(sell_vol, 2),
        'buy_aggression_pct': round(buy_pct, 1),
        'sell_aggression_pct': round(sell_pct, 1),
        'cvd': round(cvd, 2),
        'cvd_signal': 'BUY' if cvd > total * 0.1 else 'SELL' if cvd < -total * 0.1 else 'NEUTRAL',
        'large_buy_volume': round(large_buy, 2),
        'large_sell_volume': round(large_sell, 2),
        'n_trades': len(trades),
    }
