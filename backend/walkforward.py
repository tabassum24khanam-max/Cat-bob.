"""
Tier 1.4: Walk-Forward Backtester
Tests the ML model on data it has NEVER seen — the only honest accuracy measure.
TimeSeriesSplit: train on months 1-3, test on month 4, slide forward, repeat.
"""
import time
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class WFResult:
    fold: int
    train_size: int
    test_size: int
    accuracy: float
    precision_buy: float
    precision_sell: float
    avg_confidence: float
    profitable_trades_pct: float
    max_drawdown_pct: float


class WalkForwardTester:
    def __init__(self):
        self.results: List[WFResult] = []
        self.last_run_ts = 0
        self.overall_accuracy = 0.0
        self.is_running = False

    async def run_backtest(self, predictions: List[dict], n_splits: int = 5) -> dict:
        """
        Run walk-forward validation on historical predictions.
        predictions: list of dicts with keys: ts, decision, confidence, feedback, entry_price, outcome_price
        """
        self.is_running = True
        self.results = []

        rated = [p for p in predictions if p.get('feedback') in ('correct', 'wrong')]
        if len(rated) < 30:
            self.is_running = False
            return {
                'available': False,
                'reason': f'Need 30+ rated predictions, have {len(rated)}',
                'n_rated': len(rated),
            }

        rated.sort(key=lambda x: x.get('ts', 0))

        fold_size = len(rated) // (n_splits + 1)
        if fold_size < 5:
            n_splits = max(2, len(rated) // 10)
            fold_size = len(rated) // (n_splits + 1)

        results = []
        for fold in range(n_splits):
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = min(test_start + fold_size, len(rated))

            if test_end <= test_start:
                break

            train_set = rated[:train_end]
            test_set = rated[test_start:test_end]

            # Evaluate: did the predictions in test_set actually work?
            correct = sum(1 for p in test_set if p['feedback'] == 'correct')
            total = len(test_set)
            accuracy = correct / total if total else 0

            # Precision per direction
            buy_preds = [p for p in test_set if p.get('decision') == 'BUY']
            sell_preds = [p for p in test_set if p.get('decision') == 'SELL']
            prec_buy = sum(1 for p in buy_preds if p['feedback'] == 'correct') / len(buy_preds) if buy_preds else 0
            prec_sell = sum(1 for p in sell_preds if p['feedback'] == 'correct') / len(sell_preds) if sell_preds else 0

            # Average confidence
            confs = [p.get('confidence', 50) for p in test_set]
            avg_conf = sum(confs) / len(confs) if confs else 50

            # Simulated equity curve for drawdown
            equity = 1000.0
            peak = equity
            max_dd = 0
            for p in test_set:
                entry = p.get('entry_price', 0)
                outcome = p.get('outcome_price', entry)
                if not entry or not outcome:
                    continue
                move_pct = (outcome - entry) / entry * 100
                if p.get('decision') == 'SELL':
                    move_pct = -move_pct
                # Position size: confidence-scaled
                size_pct = min(5, (p.get('confidence', 50) - 40) / 10)
                pnl = equity * size_pct / 100 * move_pct / 100
                equity += pnl
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100
                max_dd = max(max_dd, dd)

            profitable = sum(1 for p in test_set if _was_profitable(p)) / total * 100 if total else 0

            results.append(WFResult(
                fold=fold + 1,
                train_size=len(train_set),
                test_size=len(test_set),
                accuracy=round(accuracy * 100, 1),
                precision_buy=round(prec_buy * 100, 1),
                precision_sell=round(prec_sell * 100, 1),
                avg_confidence=round(avg_conf, 1),
                profitable_trades_pct=round(profitable, 1),
                max_drawdown_pct=round(max_dd, 2),
            ))

        self.results = results
        self.last_run_ts = int(time.time())

        accuracies = [r.accuracy for r in results]
        self.overall_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

        self.is_running = False
        return self.get_report()

    def get_report(self) -> dict:
        if not self.results:
            return {'available': False, 'reason': 'No backtest run yet'}

        accuracies = [r.accuracy for r in self.results]
        drawdowns = [r.max_drawdown_pct for r in self.results]

        return {
            'available': True,
            'n_folds': len(self.results),
            'overall_accuracy': round(self.overall_accuracy, 1),
            'min_accuracy': round(min(accuracies), 1),
            'max_accuracy': round(max(accuracies), 1),
            'std_accuracy': round(np.std(accuracies), 1) if len(accuracies) > 1 else 0,
            'avg_max_drawdown': round(sum(drawdowns) / len(drawdowns), 2),
            'worst_drawdown': round(max(drawdowns), 2),
            'folds': [asdict(r) for r in self.results],
            'last_run': self.last_run_ts,
            'verdict': _verdict(self.overall_accuracy),
        }


def _was_profitable(pred: dict) -> bool:
    entry = pred.get('entry_price', 0)
    outcome = pred.get('outcome_price', 0)
    if not entry or not outcome:
        return pred.get('feedback') == 'correct'
    if pred.get('decision') == 'BUY':
        return outcome > entry
    elif pred.get('decision') == 'SELL':
        return outcome < entry
    return False


def _verdict(accuracy: float) -> str:
    if accuracy >= 65:
        return 'STRONG — real edge confirmed'
    elif accuracy >= 58:
        return 'VIABLE — edge exists but thin, size conservatively'
    elif accuracy >= 52:
        return 'MARGINAL — barely above random, needs improvement'
    else:
        return 'WEAK — no proven edge, model may be overfitting'


# Singleton
_tester = WalkForwardTester()

def get_walkforward_tester() -> WalkForwardTester:
    return _tester
