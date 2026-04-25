"""ULTRAMAX Equity Tracker — Track prediction P&L over time"""
import time


class EquityTracker:
    def __init__(self):
        self._curve = []
        self._initial = 10000.0

    def record_outcome(self, prediction: dict):
        """Record a prediction outcome to the equity curve.

        Assumes $100 per trade.
        BUY correct = +moved_pct * 100, wrong = -moved_pct * 100
        SELL correct = +moved_pct * 100, wrong = -moved_pct * 100
        """
        feedback = prediction.get('feedback')
        if feedback not in ('correct', 'wrong'):
            return

        moved_pct = abs(prediction.get('moved_pct', 0) or 0)
        trade_size = 100.0

        if feedback == 'correct':
            pnl = moved_pct / 100.0 * trade_size
        else:
            pnl = -(moved_pct / 100.0 * trade_size)

        prev_equity = self._curve[-1]['equity'] if self._curve else self._initial

        self._curve.append({
            'ts': prediction.get('rated_at', int(time.time())),
            'asset': prediction.get('asset', ''),
            'direction': prediction.get('decision', ''),
            'pnl': round(pnl, 2),
            'equity': round(prev_equity + pnl, 2),
        })

    def get_curve(self) -> list:
        """Return equity curve data points."""
        if not self._curve:
            return [{'ts': int(time.time()), 'equity': self._initial, 'pnl': 0}]
        return self._curve

    def get_stats(self) -> dict:
        """Return summary stats: total_return, max_drawdown, sharpe_ratio, win_streak."""
        if not self._curve:
            return {
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_streak': 0,
                'current_equity': self._initial,
                'n_trades': 0,
            }

        current_equity = self._curve[-1]['equity']
        total_return = (current_equity - self._initial) / self._initial * 100

        # Max drawdown
        peak = self._initial
        max_dd = 0.0
        for point in self._curve:
            eq = point['equity']
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (using per-trade returns)
        pnls = [p['pnl'] for p in self._curve]
        if len(pnls) >= 2:
            mean_pnl = sum(pnls) / len(pnls)
            std_pnl = (sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)) ** 0.5
            sharpe = mean_pnl / (std_pnl or 0.0001)
        else:
            sharpe = 0.0

        # Win streak (current and max)
        max_streak = 0
        current_streak = 0
        for p in self._curve:
            if p['pnl'] > 0:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
            else:
                current_streak = 0

        return {
            'total_return': round(total_return, 2),
            'max_drawdown': round(max_dd, 2),
            'sharpe_ratio': round(sharpe, 3),
            'win_streak': max_streak,
            'current_equity': round(current_equity, 2),
            'n_trades': len(self._curve),
        }
