"""
Tier 2.4: Pre-Event Sizing & Stop Adjustment
Before FOMC, CPI, NFP — volatility explodes. Reduce size, widen stops, or skip.
"""
import time
from typing import Dict, Optional


# Event impact classification
HIGH_IMPACT_EVENTS = {
    'FOMC', 'Fed Rate Decision', 'Interest Rate Decision',
    'CPI', 'Consumer Price Index', 'Inflation Rate',
    'NFP', 'Non-Farm Payrolls', 'Nonfarm Payrolls',
    'GDP', 'Gross Domestic Product',
    'PCE', 'Personal Consumption', 'Core PCE',
    'PPI', 'Producer Price Index',
    'Unemployment Rate', 'Jobs Report',
    'ECB Rate Decision', 'BOJ Rate Decision', 'BOE Rate Decision',
    'OPEC Meeting', 'OPEC+',
    'Retail Sales',
    'ISM Manufacturing', 'ISM Services',
}

MEDIUM_IMPACT_EVENTS = {
    'Durable Goods', 'Housing Starts', 'Building Permits',
    'Consumer Confidence', 'Michigan Sentiment',
    'Trade Balance', 'Current Account',
    'Industrial Production', 'Capacity Utilization',
    'Jobless Claims', 'Initial Claims',
    'Existing Home Sales', 'New Home Sales',
    'PMI', 'Flash PMI',
}


def classify_event(event_name: str) -> str:
    """Classify an economic event by impact level."""
    name_upper = event_name.upper()
    for high in HIGH_IMPACT_EVENTS:
        if high.upper() in name_upper:
            return 'HIGH'
    for med in MEDIUM_IMPACT_EVENTS:
        if med.upper() in name_upper:
            return 'MEDIUM'
    return 'LOW'


def get_pre_event_adjustments(upcoming_events: list, hours_ahead: float = 4.0) -> dict:
    """
    Given a list of upcoming events, determine how to adjust trading parameters.
    Events closer in time have more impact.

    Returns adjustments to apply to current trade decisions.
    """
    if not upcoming_events:
        return {
            'active': False,
            'size_multiplier': 1.0,
            'sl_multiplier': 1.0,
            'confidence_penalty': 0,
            'events': [],
        }

    now = time.time()
    active_events = []
    max_impact = 0

    for event in upcoming_events:
        event_ts = event.get('ts', 0)
        event_name = event.get('name', event.get('event', ''))
        if not event_ts or not event_name:
            continue

        hours_until = (event_ts - now) / 3600
        if hours_until < 0 or hours_until > hours_ahead:
            continue

        impact = classify_event(event_name)
        impact_score = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(impact, 0)

        # Closer events have more effect
        proximity_mult = 1.0
        if hours_until < 0.5:
            proximity_mult = 2.0  # within 30 min = max caution
        elif hours_until < 1:
            proximity_mult = 1.5
        elif hours_until < 2:
            proximity_mult = 1.2

        effective_impact = impact_score * proximity_mult
        max_impact = max(max_impact, effective_impact)

        active_events.append({
            'name': event_name,
            'hours_until': round(hours_until, 1),
            'impact': impact,
            'effective_score': round(effective_impact, 1),
        })

    if not active_events:
        return {
            'active': False,
            'size_multiplier': 1.0,
            'sl_multiplier': 1.0,
            'confidence_penalty': 0,
            'events': [],
        }

    # Calculate adjustments based on max impact
    if max_impact >= 5:
        # Very high impact event very close — extreme caution
        size_mult = 0.25
        sl_mult = 2.0
        conf_penalty = 20
    elif max_impact >= 3:
        # High impact event approaching
        size_mult = 0.5
        sl_mult = 1.5
        conf_penalty = 15
    elif max_impact >= 2:
        # Medium impact
        size_mult = 0.7
        sl_mult = 1.3
        conf_penalty = 10
    else:
        # Low impact
        size_mult = 0.85
        sl_mult = 1.1
        conf_penalty = 5

    return {
        'active': True,
        'size_multiplier': size_mult,
        'sl_multiplier': sl_mult,
        'confidence_penalty': conf_penalty,
        'max_impact_score': round(max_impact, 1),
        'events': active_events,
        'recommendation': _recommend(max_impact, active_events),
    }


def should_skip_trade(adjustments: dict, current_confidence: int) -> tuple:
    """
    Determine if we should skip this trade entirely due to upcoming event.
    Returns (should_skip, reason).
    """
    if not adjustments.get('active'):
        return False, ''

    # If confidence after penalty drops below 45, skip
    effective_conf = current_confidence - adjustments.get('confidence_penalty', 0)
    if effective_conf < 45:
        events_str = ', '.join(e['name'] for e in adjustments.get('events', [])[:2])
        return True, f"Pre-event skip: {events_str} (conf would be {effective_conf}%)"

    # If maximum impact and within 30 min, always skip
    if adjustments.get('max_impact_score', 0) >= 6:
        events_str = ', '.join(e['name'] for e in adjustments.get('events', [])[:2])
        return True, f"Too close to high-impact event: {events_str}"

    return False, ''


def adjust_for_post_event(event_ts: int, current_time: int = None) -> dict:
    """
    After a high-impact event, volatility is elevated for ~30-60 min.
    Widen stops but also targets (momentum moves are large post-event).
    """
    if current_time is None:
        current_time = int(time.time())

    minutes_after = (current_time - event_ts) / 60
    if minutes_after < 0 or minutes_after > 60:
        return {'active': False}

    # First 15 min: most volatile
    if minutes_after < 15:
        return {
            'active': True,
            'phase': 'immediate',
            'sl_multiplier': 2.5,
            'tp_multiplier': 2.0,
            'size_multiplier': 0.5,
        }
    # 15-30 min: settling
    elif minutes_after < 30:
        return {
            'active': True,
            'phase': 'settling',
            'sl_multiplier': 1.8,
            'tp_multiplier': 1.5,
            'size_multiplier': 0.7,
        }
    # 30-60 min: new trend may emerge
    else:
        return {
            'active': True,
            'phase': 'new_trend',
            'sl_multiplier': 1.3,
            'tp_multiplier': 1.3,
            'size_multiplier': 0.9,
        }


def _recommend(max_impact: float, events: list) -> str:
    if max_impact >= 5:
        return "SKIP TRADING — high-impact event imminent. Wait for post-event clarity."
    elif max_impact >= 3:
        return "CAUTION — reduce size 50%, widen stops. Post-event reversal likely."
    elif max_impact >= 2:
        return "MODERATE RISK — reduce size 30%. Tighten take-profit to lock gains before event."
    return "LOW RISK — minor event approaching. Normal trading with slight caution."
