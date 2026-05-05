"""
Tier 3.2: Tick Engine — Binance real-time trade stream analysis
Aggregates individual trades into micro-structure signals that candles miss.
Detects: large order clusters, momentum bursts, absorption patterns.
"""
import asyncio
import time
from typing import Dict, List, Optional
from collections import deque
from config import BINANCE_SYMBOLS

BINANCE_WS = "wss://stream.binance.com:9443/ws"


class TickAggregator:
    def __init__(self, window_seconds: int = 300):
        self.window = window_seconds
        self.trades: Dict[str, deque] = {}
        self._running = False

    def add_trade(self, asset: str, price: float, qty: float,
                  is_buyer_maker: bool, ts_ms: int):
        if asset not in self.trades:
            self.trades[asset] = deque(maxlen=10000)

        self.trades[asset].append({
            'price': price,
            'qty': qty,
            'usd': price * qty,
            'side': 'sell' if is_buyer_maker else 'buy',
            'ts': ts_ms / 1000,
        })

        # Trim old trades outside window
        cutoff = time.time() - self.window
        while self.trades[asset] and self.trades[asset][0]['ts'] < cutoff:
            self.trades[asset].popleft()

    def get_micro_structure(self, asset: str) -> dict:
        if asset not in self.trades or len(self.trades[asset]) < 10:
            return {'available': False, 'reason': 'insufficient tick data'}

        trades = list(self.trades[asset])
        now = time.time()
        recent = [t for t in trades if t['ts'] > now - 60]  # last 60s

        if len(recent) < 5:
            return {'available': False, 'reason': 'too few recent ticks'}

        # Buy vs sell volume
        buy_vol = sum(t['usd'] for t in recent if t['side'] == 'buy')
        sell_vol = sum(t['usd'] for t in recent if t['side'] == 'sell')
        total_vol = buy_vol + sell_vol

        # Large trades (>5x average)
        avg_size = total_vol / len(recent) if recent else 0
        large_threshold = avg_size * 5
        large_buys = [t for t in recent if t['side'] == 'buy' and t['usd'] > large_threshold]
        large_sells = [t for t in recent if t['side'] == 'sell' and t['usd'] > large_threshold]

        # Momentum burst: 10+ trades in same direction within 5 seconds
        momentum_burst = self._detect_momentum_burst(recent)

        # Absorption: large sell volume but price not dropping (bid absorption)
        absorption = self._detect_absorption(recent)

        # Trade rate acceleration
        rate_60s = len(recent)
        rate_300s = len(trades) / (self.window / 60)
        rate_accel = rate_60s / rate_300s if rate_300s > 0 else 1

        # CVD (Cumulative Volume Delta)
        cvd = buy_vol - sell_vol
        cvd_pct = cvd / total_vol * 100 if total_vol else 0

        bias = 0
        reasons = []

        if cvd_pct > 20:
            bias += 2
            reasons.append(f"Strong buy CVD ({cvd_pct:.0f}%)")
        elif cvd_pct < -20:
            bias -= 2
            reasons.append(f"Strong sell CVD ({cvd_pct:.0f}%)")

        if large_buys and not large_sells:
            bias += 1
            reasons.append(f"{len(large_buys)} large buy(s)")
        elif large_sells and not large_buys:
            bias -= 1
            reasons.append(f"{len(large_sells)} large sell(s)")

        if momentum_burst:
            if momentum_burst['direction'] == 'buy':
                bias += 2
                reasons.append(f"Buy momentum burst ({momentum_burst['count']} trades)")
            else:
                bias -= 2
                reasons.append(f"Sell momentum burst ({momentum_burst['count']} trades)")

        if absorption:
            if absorption['type'] == 'bid_absorption':
                bias += 2
                reasons.append("Bid absorption detected (buyers absorbing sells)")
            else:
                bias -= 2
                reasons.append("Ask absorption detected (sellers absorbing buys)")

        if rate_accel > 3:
            reasons.append(f"Trade rate accelerating ({rate_accel:.1f}x)")

        return {
            'available': True,
            'buy_volume_usd': round(buy_vol, 2),
            'sell_volume_usd': round(sell_vol, 2),
            'cvd': round(cvd, 2),
            'cvd_pct': round(cvd_pct, 1),
            'large_buys': len(large_buys),
            'large_sells': len(large_sells),
            'momentum_burst': momentum_burst,
            'absorption': absorption,
            'trade_rate_60s': rate_60s,
            'rate_acceleration': round(rate_accel, 2),
            'bias': max(-5, min(5, bias)),
            'signal': 'BUY' if bias >= 2 else 'SELL' if bias <= -2 else 'NEUTRAL',
            'reasons': reasons,
            'n_trades_window': len(trades),
        }

    def _detect_momentum_burst(self, trades: list) -> Optional[dict]:
        if len(trades) < 10:
            return None

        # Look for 10+ consecutive same-direction trades within 5s
        for i in range(len(trades) - 9):
            window = trades[i:i+10]
            time_span = window[-1]['ts'] - window[0]['ts']
            if time_span > 5:
                continue
            sides = [t['side'] for t in window]
            buy_count = sides.count('buy')
            sell_count = sides.count('sell')
            if buy_count >= 8:
                return {'direction': 'buy', 'count': buy_count, 'span_s': round(time_span, 2)}
            elif sell_count >= 8:
                return {'direction': 'sell', 'count': sell_count, 'span_s': round(time_span, 2)}
        return None

    def _detect_absorption(self, trades: list) -> Optional[dict]:
        if len(trades) < 20:
            return None

        # Split into first half and second half
        mid = len(trades) // 2
        first_half = trades[:mid]
        second_half = trades[mid:]

        first_price = sum(t['price'] for t in first_half) / len(first_half)
        second_price = sum(t['price'] for t in second_half) / len(second_half)
        price_change_pct = (second_price - first_price) / first_price * 100

        sell_vol = sum(t['usd'] for t in trades if t['side'] == 'sell')
        buy_vol = sum(t['usd'] for t in trades if t['side'] == 'buy')

        # Bid absorption: heavy selling but price barely moves down
        if sell_vol > buy_vol * 1.5 and price_change_pct > -0.05:
            return {
                'type': 'bid_absorption',
                'sell_pressure': round(sell_vol, 0),
                'price_held': True,
            }

        # Ask absorption: heavy buying but price barely moves up
        if buy_vol > sell_vol * 1.5 and price_change_pct < 0.05:
            return {
                'type': 'ask_absorption',
                'buy_pressure': round(buy_vol, 0),
                'price_held': True,
            }

        return None


# Singleton
_tick_engine = TickAggregator()

def get_tick_engine() -> TickAggregator:
    return _tick_engine


async def start_tick_stream(assets: List[str]):
    """Start websocket connection to Binance trade stream."""
    try:
        import websockets
    except ImportError:
        print("websockets not installed — tick engine disabled")
        return

    symbols = [BINANCE_SYMBOLS[a].lower() for a in assets if a in BINANCE_SYMBOLS]
    if not symbols:
        return

    streams = '/'.join(f"{s}@trade" for s in symbols[:5])
    url = f"{BINANCE_WS}/{streams}"

    engine = get_tick_engine()
    engine._running = True

    while engine._running:
        try:
            async with websockets.connect(url) as ws:
                async for msg in ws:
                    import json
                    data = json.loads(msg)
                    if 'e' in data and data['e'] == 'trade':
                        symbol = data.get('s', '')
                        asset = next((a for a, s in BINANCE_SYMBOLS.items() if s == symbol), None)
                        if asset:
                            engine.add_trade(
                                asset,
                                float(data['p']),
                                float(data['q']),
                                data.get('m', False),
                                data.get('T', int(time.time() * 1000)),
                            )
        except Exception:
            await asyncio.sleep(5)
