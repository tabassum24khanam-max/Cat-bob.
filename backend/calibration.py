"""ULTRAMAX Calibration — Adjust confidence based on historical accuracy"""


async def calibrate_confidence(raw_conf: float, asset: str, horizon: int,
                                predictions: list) -> dict:
    """Platt scaling: map raw confidence to calibrated probability.
    Returns: {calibrated: float, reliability: str, n_samples: int,
              bucket_accuracy: dict}
    """
    # Filter relevant predictions that have been rated
    relevant = [
        p for p in predictions
        if p.get('asset') == asset
        and p.get('horizon') == horizon
        and p.get('feedback') in ('correct', 'wrong')
        and p.get('confidence') is not None
    ]

    n_samples = len(relevant)

    # Define confidence buckets
    bucket_ranges = {
        '40-50': (40, 50),
        '50-60': (50, 60),
        '60-70': (60, 70),
        '70-80': (70, 80),
        '80+': (80, 100),
    }

    bucket_accuracy = {}
    for label, (lo, hi) in bucket_ranges.items():
        bucket_preds = [
            p for p in relevant
            if lo <= (p.get('confidence') or 0) < (hi if hi < 100 else 101)
        ]
        if len(bucket_preds) > 0:
            correct_count = sum(1 for p in bucket_preds if p['feedback'] == 'correct')
            bucket_accuracy[label] = {
                'accuracy': round(correct_count / len(bucket_preds) * 100, 1),
                'count': len(bucket_preds),
            }
        else:
            bucket_accuracy[label] = {'accuracy': None, 'count': 0}

    # Find which bucket the raw confidence falls into
    calibrated = raw_conf
    matched_bucket = None
    for label, (lo, hi) in bucket_ranges.items():
        if lo <= raw_conf < (hi if hi < 100 else 101):
            matched_bucket = label
            break

    if matched_bucket and bucket_accuracy[matched_bucket]['count'] >= 10:
        # Use actual bucket accuracy as calibrated confidence
        calibrated = bucket_accuracy[matched_bucket]['accuracy']
    # Otherwise keep raw_conf as calibrated

    # Determine reliability label
    if n_samples < 10:
        reliability = 'insufficient_data'
    elif n_samples < 30:
        reliability = 'low'
    elif n_samples < 100:
        reliability = 'moderate'
    else:
        reliability = 'high'

    return {
        'calibrated': round(calibrated, 1),
        'raw': round(raw_conf, 1),
        'reliability': reliability,
        'n_samples': n_samples,
        'bucket_accuracy': bucket_accuracy,
    }
