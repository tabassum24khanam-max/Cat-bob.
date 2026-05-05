"""
Tier 2.3: Regime-Specific Strategy Parameters
The HMM detects trending/ranging/volatile, but the bot used the same settings for all.
This module provides different stop-loss, take-profit, trailing, and sizing per regime.
"""
from typing import Dict


# Each regime has optimal parameters discovered through backtesting intuition
REGIME_PARAMS = {
    'TRENDING': {
        'sl_multiplier': 2.5,      # wider stops — let trends run
        'tp_multiplier': 4.0,      # bigger targets
        'trail_multiplier': 1.8,   # trail loosely
        'size_multiplier': 1.2,    # slightly larger (higher win rate in trends)
        'min_confidence': 52,      # lower bar (trends are forgiving)
        'max_hold_hours': 24,      # hold longer
        'description': 'Trend-following: wide stops, big targets, let winners run',
    },
    'RANGING': {
        'sl_multiplier': 1.2,      # tight stops — mean reversion
        'tp_multiplier': 1.8,      # small targets (trade the range)
        'trail_multiplier': 0.8,   # trail tightly
        'size_multiplier': 0.8,    # smaller (lower win rate in chop)
        'min_confidence': 58,      # higher bar (more false signals in ranges)
        'max_hold_hours': 6,       # quick in and out
        'description': 'Mean-reversion: tight stops, small targets, quick exits',
    },
    'VOLATILE': {
        'sl_multiplier': 3.0,      # very wide (or you get stopped constantly)
        'tp_multiplier': 5.0,      # if volatile, moves are bigger
        'trail_multiplier': 2.5,   # wide trail
        'size_multiplier': 0.5,    # half-size (uncertainty is high)
        'min_confidence': 60,      # need stronger signal
        'max_hold_hours': 12,      # medium hold
        'description': 'Volatile: wide stops, half-size, need strong conviction',
    },
    'UNKNOWN': {
        'sl_multiplier': 2.0,
        'tp_multiplier': 3.0,
        'trail_multiplier': 1.5,
        'size_multiplier': 0.7,
        'min_confidence': 55,
        'max_hold_hours': 8,
        'description': 'Unknown regime: conservative defaults',
    },
}


def get_regime_from_hmm(hmm_probs: dict) -> str:
    """Determine dominant regime from HMM probabilities."""
    if not hmm_probs:
        return 'UNKNOWN'
    max_regime = max(hmm_probs.items(), key=lambda x: x[1])
    if max_regime[1] < 0.4:
        return 'UNKNOWN'
    return max_regime[0].upper()


def get_regime_params(regime: str) -> dict:
    """Get strategy parameters for a given regime."""
    return REGIME_PARAMS.get(regime, REGIME_PARAMS['UNKNOWN'])


def apply_regime_adjustments(base_sl_pct: float, base_tp_pct: float,
                              base_trail_pct: float, base_size: float,
                              hmm_probs: dict) -> dict:
    """
    Adjust trading parameters based on detected regime.
    Returns adjusted values + regime info.
    """
    regime = get_regime_from_hmm(hmm_probs)
    params = get_regime_params(regime)

    # Scale base values by regime multipliers
    # But keep within safety bounds
    adj_sl = base_sl_pct * params['sl_multiplier'] / 2.0  # normalize (base multiplier was 2.0)
    adj_tp = base_tp_pct * params['tp_multiplier'] / 3.0
    adj_trail = base_trail_pct * params['trail_multiplier'] / 1.5
    adj_size = base_size * params['size_multiplier']

    # Safety clamps
    adj_sl = max(0.5, min(5.0, adj_sl))
    adj_tp = max(1.0, min(10.0, adj_tp))
    adj_trail = max(0.3, min(4.0, adj_trail))
    adj_size = max(base_size * 0.3, min(base_size * 1.5, adj_size))

    return {
        'regime': regime,
        'regime_confidence': hmm_probs.get(regime, hmm_probs.get(regime.upper(), 0)),
        'sl_pct': round(adj_sl, 2),
        'tp_pct': round(adj_tp, 2),
        'trail_pct': round(adj_trail, 2),
        'size': round(adj_size, 2),
        'min_confidence': params['min_confidence'],
        'max_hold_hours': params['max_hold_hours'],
        'description': params['description'],
    }


def regime_confidence_adjustment(regime: str, base_confidence: int, direction: str,
                                  trend_slope: float = 0) -> int:
    """
    Adjust confidence based on regime + direction alignment.
    Trend-following in a trend = bonus. Counter-trend in a trend = penalty.
    """
    adj = 0

    if regime == 'TRENDING':
        if trend_slope > 0.1 and direction == 'BUY':
            adj = 5  # buying in uptrend
        elif trend_slope < -0.1 and direction == 'SELL':
            adj = 5  # selling in downtrend
        elif trend_slope > 0.1 and direction == 'SELL':
            adj = -8  # counter-trend penalty
        elif trend_slope < -0.1 and direction == 'BUY':
            adj = -8

    elif regime == 'RANGING':
        # In range, both directions are ok (mean reversion)
        adj = 0

    elif regime == 'VOLATILE':
        # Volatile = uncertain, slight penalty to all
        adj = -3

    return max(40, min(95, base_confidence + adj))


def get_all_regimes_info() -> dict:
    """Return info about all regime configurations (for frontend display)."""
    return {
        regime: {
            'sl_mult': p['sl_multiplier'],
            'tp_mult': p['tp_multiplier'],
            'size_mult': p['size_multiplier'],
            'min_conf': p['min_confidence'],
            'description': p['description'],
        }
        for regime, p in REGIME_PARAMS.items()
    }
