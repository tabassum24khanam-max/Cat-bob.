"""
ULTRAMAX Trading Engine — Autonomous trading with safety controls
Supports: Alpaca (stocks), Binance (crypto), Paper trading (all)
"""
import asyncio
import time
import json
import os
import httpx
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict
from config import ALPACA_KEY, ALPACA_SECRET, BINANCE_SYMBOLS, get_asset_type

# D10: Safety Controls
MAX_POSITIONS = 10
MAX_DAILY_LOSS_PCT = 5.0
MAX_POSITION_SIZE_PCT = 20.0
MIN_CONFIDENCE = 40
MIN_PQS = 3

@dataclass
class Position:
    id: str
    asset: str
    direction: str  # BUY or SELL
    entry_price: float
    size: float  # in USD
    entry_time: int
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop_pct: float = 0.0
    highest_price: float = 0.0  # for trailing stop
    lowest_price: float = 0.0
    status: str = 'open'  # open, closed, stopped
    pnl: float = 0.0
    exit_price: float = 0.0
    exit_time: int = 0
    paper: bool = True

class TradingEngine:
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.paper_mode = True
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_heartbeat = 0
        self._equity = 10000.0  # paper trading starting equity
        self._trade_log = []

    # D1: Heartbeat
    def heartbeat(self) -> dict:
        self.last_heartbeat = int(time.time())
        return {
            'status': 'alive',
            'ts': self.last_heartbeat,
            'positions': len([p for p in self.positions.values() if p.status == 'open']),
            'daily_pnl': round(self.daily_pnl, 2),
            'paper_mode': self.paper_mode,
            'equity': round(self._equity, 2),
        }

    # D8: Kelly Criterion
    def kelly_size(self, win_rate: float, avg_win: float, avg_loss: float, equity: float) -> float:
        """Calculate Kelly criterion position size."""
        if avg_loss == 0 or win_rate <= 0:
            return 0
        b = avg_win / abs(avg_loss)  # win/loss ratio
        kelly = (win_rate * b - (1 - win_rate)) / b
        kelly = max(0, min(kelly, 0.25))  # cap at 25% (half-Kelly)
        return round(equity * kelly, 2)

    # D10: Safety checks
    def can_trade(self, confidence: int, pqs_score: int = 0) -> tuple:
        """Check all safety controls. Returns (allowed, reason)."""
        open_count = len([p for p in self.positions.values() if p.status == 'open'])
        if open_count >= MAX_POSITIONS:
            return False, f'Max positions reached ({MAX_POSITIONS})'
        if self.daily_pnl <= -(self._equity * MAX_DAILY_LOSS_PCT / 100):
            return False, f'Daily loss limit hit ({MAX_DAILY_LOSS_PCT}%)'
        if confidence < MIN_CONFIDENCE:
            return False, f'Confidence too low ({confidence}% < {MIN_CONFIDENCE}%)'
        if pqs_score < MIN_PQS:
            return False, f'PQS too low ({pqs_score} < {MIN_PQS})'
        return True, 'OK'

    # D2: Open position
    async def open_position(self, asset: str, direction: str, price: float,
                           size_usd: float, stop_loss_pct: float = 2.0,
                           take_profit_pct: float = 4.0, trailing: float = 1.5) -> dict:
        """Open a new position (paper or live)."""
        pos_id = f"pos_{int(time.time())}_{asset}"

        sl = price * (1 - stop_loss_pct/100) if direction == 'BUY' else price * (1 + stop_loss_pct/100)
        tp = price * (1 + take_profit_pct/100) if direction == 'BUY' else price * (1 - take_profit_pct/100)

        pos = Position(
            id=pos_id, asset=asset, direction=direction,
            entry_price=price, size=size_usd,
            entry_time=int(time.time()),
            stop_loss=sl, take_profit=tp,
            trailing_stop_pct=trailing,
            highest_price=price, lowest_price=price,
            paper=self.paper_mode,
        )

        # D3/D4: Execute on exchange if live mode
        if not self.paper_mode:
            asset_type = get_asset_type(asset)
            if asset_type == 'crypto':
                result = await self._binance_order(asset, direction, size_usd, price)
            else:
                result = await self._alpaca_order(asset, direction, size_usd, price)
            if not result.get('ok'):
                return {'ok': False, 'error': result.get('error', 'Exchange order failed')}

        self.positions[pos_id] = pos
        self.daily_trades += 1
        return {'ok': True, 'position': asdict(pos)}

    # D7: Stay/Exit decision
    def check_exit(self, pos: Position, current_price: float) -> dict:
        """Decide whether to stay or exit a position."""
        if pos.direction == 'BUY':
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            pos.highest_price = max(pos.highest_price, current_price)
            # D6: Trailing stop
            trailing_stop = pos.highest_price * (1 - pos.trailing_stop_pct / 100)
            if current_price <= pos.stop_loss:
                return {'action': 'exit', 'reason': 'stop_loss', 'pnl_pct': pnl_pct}
            if current_price >= pos.take_profit:
                return {'action': 'exit', 'reason': 'take_profit', 'pnl_pct': pnl_pct}
            if current_price <= trailing_stop and pos.highest_price > pos.entry_price * 1.005:
                return {'action': 'exit', 'reason': 'trailing_stop', 'pnl_pct': pnl_pct}
        else:  # SELL
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100
            pos.lowest_price = min(pos.lowest_price, current_price)
            trailing_stop = pos.lowest_price * (1 + pos.trailing_stop_pct / 100)
            if current_price >= pos.stop_loss:
                return {'action': 'exit', 'reason': 'stop_loss', 'pnl_pct': pnl_pct}
            if current_price <= pos.take_profit:
                return {'action': 'exit', 'reason': 'take_profit', 'pnl_pct': pnl_pct}
            if current_price >= trailing_stop and pos.lowest_price < pos.entry_price * 0.995:
                return {'action': 'exit', 'reason': 'trailing_stop', 'pnl_pct': pnl_pct}

        return {'action': 'hold', 'pnl_pct': pnl_pct}

    async def close_position(self, pos_id: str, current_price: float, reason: str = 'manual') -> dict:
        """Close a position."""
        pos = self.positions.get(pos_id)
        if not pos or pos.status != 'open':
            return {'ok': False, 'error': 'Position not found or already closed'}

        if pos.direction == 'BUY':
            pnl = (current_price - pos.entry_price) / pos.entry_price * pos.size
        else:
            pnl = (pos.entry_price - current_price) / pos.entry_price * pos.size

        pos.status = 'closed'
        pos.exit_price = current_price
        pos.exit_time = int(time.time())
        pos.pnl = round(pnl, 2)
        self.daily_pnl += pnl
        self._equity += pnl

        self._trade_log.append({
            'id': pos_id, 'asset': pos.asset, 'direction': pos.direction,
            'entry': pos.entry_price, 'exit': current_price,
            'pnl': pos.pnl, 'reason': reason, 'ts': pos.exit_time,
        })

        return {'ok': True, 'pnl': pos.pnl, 'reason': reason}

    # D9: Paper trading toggle
    def set_paper_mode(self, enabled: bool):
        self.paper_mode = enabled
        return {'paper_mode': enabled}

    def get_positions(self, status: str = None) -> list:
        positions = list(self.positions.values())
        if status:
            positions = [p for p in positions if p.status == status]
        return [asdict(p) for p in positions]

    def get_trade_log(self, limit: int = 50) -> list:
        return self._trade_log[-limit:]

    def get_equity_curve(self) -> list:
        """Build equity curve from trade log."""
        curve = [{'ts': 0, 'equity': 10000.0}]
        equity = 10000.0
        for trade in self._trade_log:
            equity += trade['pnl']
            curve.append({'ts': trade['ts'], 'equity': round(equity, 2)})
        return curve

    # D5: TradingView webhook handler
    async def handle_webhook(self, payload: dict) -> dict:
        """Process TradingView webhook alert."""
        asset = payload.get('ticker', '').replace('USDT', '')
        action = payload.get('action', '').upper()  # BUY or SELL
        price = float(payload.get('price', 0))

        if not asset or action not in ('BUY', 'SELL') or not price:
            return {'ok': False, 'error': 'Invalid webhook payload'}

        size = self.kelly_size(0.55, 2.0, 1.0, self._equity)
        if size < 10:
            size = min(100, self._equity * 0.05)

        return await self.open_position(asset, action, price, size)

    # D3: Alpaca order
    async def _alpaca_order(self, asset: str, direction: str, size_usd: float, price: float) -> dict:
        if not ALPACA_KEY or not ALPACA_SECRET:
            return {'ok': False, 'error': 'Alpaca keys not configured'}
        try:
            qty = max(1, int(size_usd / price))
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{base_url}/v2/orders",
                    headers={
                        'APCA-API-KEY-ID': ALPACA_KEY,
                        'APCA-API-SECRET-KEY': ALPACA_SECRET,
                    },
                    json={
                        'symbol': asset, 'qty': str(qty),
                        'side': 'buy' if direction == 'BUY' else 'sell',
                        'type': 'market', 'time_in_force': 'day',
                    }
                )
                if resp.status_code in (200, 201):
                    return {'ok': True, 'order': resp.json()}
                return {'ok': False, 'error': f'Alpaca HTTP {resp.status_code}: {resp.text[:100]}'}
        except Exception as e:
            return {'ok': False, 'error': str(e)[:100]}

    # D4: Binance order
    async def _binance_order(self, asset: str, direction: str, size_usd: float, price: float) -> dict:
        # Binance requires API keys and HMAC signing — stubbed for paper mode
        return {'ok': False, 'error': 'Binance live trading not yet configured — use paper mode'}

    async def check_all_positions(self, fetch_price_fn) -> list:
        """Check all open positions against current prices."""
        actions = []
        for pos in list(self.positions.values()):
            if pos.status != 'open':
                continue
            try:
                result = await fetch_price_fn(pos.asset)
                current_price = result.get('price', 0)
                if not current_price:
                    continue
                check = self.check_exit(pos, current_price)
                if check['action'] == 'exit':
                    close_result = await self.close_position(pos.id, current_price, check['reason'])
                    actions.append({**close_result, 'asset': pos.asset, 'reason': check['reason'],
                                    'position': {'direction': pos.direction, 'entry_price': pos.entry_price, 'exit_price': current_price}})
            except Exception:
                continue
        return actions


# Singleton
_engine = TradingEngine()

def get_engine() -> TradingEngine:
    return _engine
