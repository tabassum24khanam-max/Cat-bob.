"""
Tier 2.5: Model Disagreement Filter
When XGBoost says BUY 72% but RF says SELL 60%, that IS information.
Disagreement = high uncertainty = reduce size or skip. Agreement = go bigger.
"""
from typing import Dict, List, Optional


def compute_disagreement(model_predictions: Dict[str, dict]) -> dict:
    """
    Analyze agreement/disagreement across multiple prediction sources.

    model_predictions: {
        'quant_agent': {'direction': 'BUY', 'confidence': 70},
        'news_agent': {'direction': 'SELL', 'confidence': 55},
        'ml_ensemble': {'direction': 'BUY', 'confidence': 65},
        'decision_agent': {'direction': 'BUY', 'confidence': 72},
    }
    """
    if not model_predictions or len(model_predictions) < 2:
        return {'available': False, 'reason': 'need 2+ models'}

    directions = {}
    confidences = []

    for name, pred in model_predictions.items():
        d = pred.get('direction', 'NO_TRADE').upper()
        c = pred.get('confidence', 50)
        if d in ('BUY', 'SELL'):
            directions[name] = d
            confidences.append(c)

    if len(directions) < 2:
        return {'available': False, 'reason': 'need 2+ directional predictions'}

    # Count votes
    buy_votes = sum(1 for d in directions.values() if d == 'BUY')
    sell_votes = sum(1 for d in directions.values() if d == 'SELL')
    total_votes = buy_votes + sell_votes

    # Unanimous agreement
    unanimous = buy_votes == total_votes or sell_votes == total_votes
    majority_direction = 'BUY' if buy_votes > sell_votes else 'SELL' if sell_votes > buy_votes else 'SPLIT'
    agreement_pct = max(buy_votes, sell_votes) / total_votes * 100

    # Confidence spread (high spread = uncertainty)
    conf_spread = max(confidences) - min(confidences) if confidences else 0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 50

    # Disagreement score: 0 (full agreement) to 10 (max disagreement)
    disagreement_score = 0
    reasons = []

    if not unanimous:
        disagreement_score += 3
        minority = [n for n, d in directions.items() if d != majority_direction]
        reasons.append(f"Split vote: {minority} disagree")

    if conf_spread > 20:
        disagreement_score += 2
        reasons.append(f"Confidence spread: {conf_spread:.0f}%")
    elif conf_spread > 10:
        disagreement_score += 1

    if agreement_pct < 60:
        disagreement_score += 3
        reasons.append(f"Near-even split ({agreement_pct:.0f}% vs {100-agreement_pct:.0f}%)")

    # Special: ML vs AI disagreement is the strongest signal
    ml_dir = model_predictions.get('ml_ensemble', {}).get('direction', '').upper()
    ai_dir = model_predictions.get('decision_agent', {}).get('direction', '').upper()
    if ml_dir and ai_dir and ml_dir != ai_dir and ml_dir in ('BUY', 'SELL') and ai_dir in ('BUY', 'SELL'):
        disagreement_score += 2
        reasons.append(f"ML ({ml_dir}) vs AI ({ai_dir}) conflict")

    disagreement_score = min(10, disagreement_score)

    # Determine action
    action = _determine_action(disagreement_score, agreement_pct, avg_confidence)

    return {
        'available': True,
        'disagreement_score': disagreement_score,
        'agreement_pct': round(agreement_pct, 1),
        'majority_direction': majority_direction,
        'unanimous': unanimous,
        'buy_votes': buy_votes,
        'sell_votes': sell_votes,
        'confidence_spread': round(conf_spread, 1),
        'avg_confidence': round(avg_confidence, 1),
        'action': action,
        'reasons': reasons,
        'details': {name: d for name, d in directions.items()},
    }


def apply_disagreement(base_confidence: int, base_size: float,
                       disagreement: dict) -> dict:
    """Apply disagreement adjustments to confidence and size."""
    if not disagreement.get('available'):
        return {
            'confidence': base_confidence,
            'size': base_size,
            'adjusted': False,
        }

    score = disagreement.get('disagreement_score', 0)
    action = disagreement.get('action', {})

    conf_adj = 0
    size_mult = 1.0

    if score >= 7:
        conf_adj = -15
        size_mult = 0.3
    elif score >= 5:
        conf_adj = -10
        size_mult = 0.5
    elif score >= 3:
        conf_adj = -5
        size_mult = 0.7
    elif score == 0 and disagreement.get('unanimous'):
        conf_adj = 5
        size_mult = 1.2

    return {
        'confidence': max(40, base_confidence + conf_adj),
        'size': round(base_size * size_mult, 2),
        'adjusted': conf_adj != 0,
        'conf_adjustment': conf_adj,
        'size_multiplier': size_mult,
        'disagreement_score': score,
    }


def _determine_action(score: int, agreement_pct: float, avg_conf: float) -> dict:
    if score >= 7:
        return {
            'recommendation': 'SKIP',
            'reason': 'High disagreement — models fundamentally conflict',
            'force_no_trade': True,
        }
    elif score >= 5:
        return {
            'recommendation': 'REDUCE',
            'reason': 'Moderate disagreement — trade with reduced conviction',
            'force_no_trade': False,
        }
    elif score >= 3:
        return {
            'recommendation': 'CAUTION',
            'reason': 'Some disagreement — slightly reduce exposure',
            'force_no_trade': False,
        }
    elif score == 0:
        return {
            'recommendation': 'STRONG',
            'reason': 'Full agreement across all models — high conviction trade',
            'force_no_trade': False,
        }
    return {
        'recommendation': 'NORMAL',
        'reason': 'Minor disagreement within tolerance',
        'force_no_trade': False,
    }
