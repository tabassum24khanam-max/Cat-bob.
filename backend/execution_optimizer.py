"""
Tier 3.3: Execution Optimizer — TWAP, limit orders, slippage estimation
Market orders lose 0.05-0.2% per trade on slippage. Over 1000 trades that's 50-200%.
Smart execution splits orders, uses limits, and estimates real fill quality.
"""
import time
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from config import BINANCE_SYMBOLS


@dataclass
class ExecutionPlan:
    asset: str
    direction: str
    total_size_usd: float
    strategy: str  # 'market', 'limit', 'twap', 'iceberg'
    n_slices: int
    slice_interval_s: float
    limit_offset_pct: float  # how far from mid-price to place limit
    max_slippage_pct: float
    urgency: str  # 'low', 'medium', 'high'


class ExecutionOptimizer:
    def __init__(self):
        self.execution_log: list = []
        self.total_slippage_saved = 0.0
        self.total_trades_optimized = 0

    def plan_execution(self, asset: str, direction: str, size_usd: float,
                       current_price: float, atr_pct: float = 1.0,
                       urgency: str = 'medium') -> ExecutionPlan:
        """
        Determine optimal execution strategy based on order size and market conditions.
        """
        # For small orders (<$500), market order is fine
        if size_usd < 500:
            return ExecutionPlan(
                asset=asset, direction=direction, total_size_usd=size_usd,
                strategy='market', n_slices=1, slice_interval_s=0,
                limit_offset_pct=0, max_slippage_pct=0.1, urgency=urgency,
            )

        # For medium orders ($500-$5000), use limit with timeout
        if size_usd < 5000:
            offset = min(0.05, atr_pct * 0.02)  # place limit slightly inside spread
            return ExecutionPlan(
                asset=asset, direction=direction, total_size_usd=size_usd,
                strategy='limit', n_slices=1, slice_interval_s=0,
                limit_offset_pct=offset, max_slippage_pct=0.05, urgency=urgency,
            )

        # For large orders (>$5000), use TWAP (Time-Weighted Average Price)
        n_slices = min(10, max(3, int(size_usd / 1000)))
        interval = 30 if urgency == 'high' else 60 if urgency == 'medium' else 120

        return ExecutionPlan(
            asset=asset, direction=direction, total_size_usd=size_usd,
            strategy='twap', n_slices=n_slices, slice_interval_s=interval,
            limit_offset_pct=0.03, max_slippage_pct=0.03, urgency=urgency,
        )

    def estimate_slippage(self, asset: str, size_usd: float,
                          orderbook_depth: dict = None) -> dict:
        """
        Estimate expected slippage for a given order size.
        Uses orderbook depth if available, otherwise uses heuristic.
        """
        if orderbook_depth and orderbook_depth.get('available'):
            # Walk the book to estimate fill price
            total_bid = orderbook_depth.get('total_bid_vol', 0)
            total_ask = orderbook_depth.get('total_ask_vol', 0)
            spread = orderbook_depth.get('spread_pct', 0.01)

            # Rough model: slippage = spread/2 + size_impact
            depth_usd = min(total_bid, total_ask) * orderbook_depth.get('mid_price', 1)
            if depth_usd > 0:
                size_impact = (size_usd / depth_usd) * 0.1  # 10% of depth ratio
            else:
                size_impact = 0.05

            estimated = spread / 2 + size_impact
        else:
            # Heuristic based on asset type
            is_crypto = asset in BINANCE_SYMBOLS
            base_spread = 0.02 if is_crypto else 0.01  # crypto has tighter spreads
            size_impact = (size_usd / 50000) * 0.05
            estimated = base_spread + size_impact

        estimated = max(0.01, min(0.5, estimated))

        return {
            'estimated_slippage_pct': round(estimated, 4),
            'estimated_slippage_usd': round(size_usd * estimated / 100, 2),
            'market_order_cost': round(size_usd * estimated / 100, 2),
            'limit_order_savings': round(size_usd * estimated / 100 * 0.7, 2),
            'recommendation': 'limit' if estimated > 0.03 else 'market',
        }

    def record_execution(self, asset: str, planned_price: float,
                         actual_price: float, size_usd: float, strategy: str):
        """Record actual execution for slippage tracking."""
        slippage_pct = abs(actual_price - planned_price) / planned_price * 100
        savings_vs_market = max(0, 0.05 - slippage_pct)  # assume 0.05% for market order

        self.execution_log.append({
            'ts': int(time.time()),
            'asset': asset,
            'planned_price': planned_price,
            'actual_price': actual_price,
            'slippage_pct': round(slippage_pct, 4),
            'size_usd': size_usd,
            'strategy': strategy,
            'savings_usd': round(savings_vs_market * size_usd / 100, 2),
        })
        self.execution_log = self.execution_log[-500:]
        self.total_trades_optimized += 1

        if savings_vs_market > 0:
            self.total_slippage_saved += savings_vs_market * size_usd / 100

    def get_optimal_limit_price(self, direction: str, current_price: float,
                                 bid: float = 0, ask: float = 0,
                                 offset_pct: float = 0.02) -> float:
        """Calculate optimal limit order price."""
        if direction == 'BUY':
            if bid > 0:
                return round(bid + (ask - bid) * 0.3, 6)  # 30% into the spread from bid
            return round(current_price * (1 - offset_pct / 100), 6)
        else:
            if ask > 0:
                return round(ask - (ask - bid) * 0.3, 6)  # 30% into the spread from ask
            return round(current_price * (1 + offset_pct / 100), 6)

    def get_report(self) -> dict:
        if not self.execution_log:
            return {
                'available': False,
                'total_trades': 0,
                'total_slippage_saved_usd': 0,
            }

        avg_slippage = sum(e['slippage_pct'] for e in self.execution_log) / len(self.execution_log)
        total_volume = sum(e['size_usd'] for e in self.execution_log)

        return {
            'available': True,
            'total_trades': self.total_trades_optimized,
            'avg_slippage_pct': round(avg_slippage, 4),
            'total_volume_usd': round(total_volume, 2),
            'total_slippage_saved_usd': round(self.total_slippage_saved, 2),
            'recent': self.execution_log[-10:],
        }


# Singleton
_optimizer = ExecutionOptimizer()

def get_execution_optimizer() -> ExecutionOptimizer:
    return _optimizer
