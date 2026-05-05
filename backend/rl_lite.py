"""
Tier 3.1: RL-Lite — Outcome-Weighted Gate & Feature Adjustment
Not full PPO/SAC (needs millions of samples). Instead: a lightweight reward system
that learns which gates/features/signals are actually useful from real trade outcomes.

Each gate/signal gets a "trust score" 0-100. After each trade:
  - Win: boost trust of signals that agreed with the trade
  - Loss: reduce trust of signals that agreed with the trade
Over time, useless gates decay to 0, useful ones climb to 100.
"""
import time
import json
import os
from typing import Dict, List
from config import DATA_DIR

TRUST_FILE = DATA_DIR / "rl_trust_scores.json"
DECAY_RATE = 0.99
LEARN_RATE_WIN = 3.0
LEARN_RATE_LOSS = 4.0
INITIAL_TRUST = 50.0


class RLLite:
    def __init__(self):
        self.trust_scores: Dict[str, float] = {}
        self.update_count = 0
        self.trade_history: List[dict] = []
        self._load()

    def _load(self):
        if TRUST_FILE.exists():
            try:
                data = json.loads(TRUST_FILE.read_text())
                self.trust_scores = data.get('scores', {})
                self.update_count = data.get('update_count', 0)
            except Exception:
                self.trust_scores = {}

    def _save(self):
        try:
            TRUST_FILE.write_text(json.dumps({
                'scores': self.trust_scores,
                'update_count': self.update_count,
                'last_save': int(time.time()),
            }, indent=2))
        except Exception:
            pass

    def get_trust(self, signal_name: str) -> float:
        return self.trust_scores.get(signal_name, INITIAL_TRUST)

    def record_outcome(self, trade_signals: Dict[str, bool], won: bool, pnl_pct: float):
        magnitude = min(3.0, abs(pnl_pct) / 1.0)

        for signal_name, agreed in trade_signals.items():
            current = self.trust_scores.get(signal_name, INITIAL_TRUST)

            if agreed:
                if won:
                    delta = LEARN_RATE_WIN * magnitude
                    new_score = min(100, current + delta)
                else:
                    delta = LEARN_RATE_LOSS * magnitude
                    new_score = max(0, current - delta)
            else:
                if won:
                    delta = LEARN_RATE_WIN * magnitude * 0.5
                    new_score = max(0, current - delta)
                else:
                    delta = LEARN_RATE_LOSS * magnitude * 0.5
                    new_score = min(100, current + delta)

            self.trust_scores[signal_name] = round(new_score, 2)

        self.update_count += 1
        self.trade_history.append({
            'ts': int(time.time()),
            'won': won,
            'pnl_pct': round(pnl_pct, 2),
            'signals_count': len(trade_signals),
        })
        self.trade_history = self.trade_history[-200:]

        if self.update_count % 5 == 0:
            self._save()

    def apply_trust_weights(self, signals: Dict[str, float]) -> Dict[str, float]:
        weighted = {}
        for name, value in signals.items():
            trust = self.get_trust(name) / 100.0
            weight = max(0.1, trust * 1.5)
            weighted[name] = value * weight
        return weighted

    def get_confidence_adjustment(self, active_gates: List[str]) -> int:
        if not active_gates:
            return 0

        trust_sum = sum(self.get_trust(g) for g in active_gates)
        avg_trust = trust_sum / len(active_gates)

        if avg_trust > 70:
            return 5
        elif avg_trust > 55:
            return 2
        elif avg_trust < 30:
            return -10
        elif avg_trust < 40:
            return -5
        return 0

    def decay_unused(self):
        for name in list(self.trust_scores.keys()):
            score = self.trust_scores[name]
            if score > INITIAL_TRUST:
                self.trust_scores[name] = max(INITIAL_TRUST, score * DECAY_RATE)
            elif score < INITIAL_TRUST:
                self.trust_scores[name] = min(INITIAL_TRUST, score + (INITIAL_TRUST - score) * (1 - DECAY_RATE))

    def get_report(self) -> dict:
        if not self.trust_scores:
            return {'available': False, 'reason': 'No trades recorded yet'}

        sorted_scores = sorted(self.trust_scores.items(), key=lambda x: x[1], reverse=True)
        trusted = [(n, s) for n, s in sorted_scores if s > 65]
        distrusted = [(n, s) for n, s in sorted_scores if s < 35]
        neutral = [(n, s) for n, s in sorted_scores if 35 <= s <= 65]

        total_trades = len(self.trade_history)
        wins = sum(1 for t in self.trade_history if t['won'])

        return {
            'available': True,
            'total_signals_tracked': len(self.trust_scores),
            'update_count': self.update_count,
            'total_trades_learned': total_trades,
            'win_rate': round(wins / total_trades * 100, 1) if total_trades else 0,
            'trusted_signals': [(n, round(s, 1)) for n, s in trusted[:10]],
            'distrusted_signals': [(n, round(s, 1)) for n, s in distrusted[:10]],
            'neutral_signals': len(neutral),
            'top_3': [n for n, _ in sorted_scores[:3]],
            'bottom_3': [n for n, _ in sorted_scores[-3:]],
        }


_rl = RLLite()

def get_rl_lite() -> RLLite:
    return _rl
