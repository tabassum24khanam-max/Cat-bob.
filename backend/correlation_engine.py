"""
ULTRAMAX Correlation Engine — Cross-asset correlations and lead/lag detection
"""
import numpy as np
from database import get_price_history


# Reference assets for correlation computation
REFERENCE_ASSETS = ['BTC', 'SPY', 'GC=F']


async def compute_correlations(asset: str, window_days: int = 30) -> dict:
    """Compute rolling correlations against reference assets and detect lead/lag."""
    limit = window_days * 24  # hourly data points

    # Fetch price history for the target asset
    asset_data = await get_price_history(asset, limit)
    if len(asset_data) < 48:  # Need at least 2 days
        return {'available': False, 'reason': 'Insufficient data'}

    # Get asset returns
    asset_ts = {r['ts']: r['close'] for r in asset_data if r.get('close')}
    asset_times = sorted(asset_ts.keys())
    asset_returns = {}
    for i in range(1, len(asset_times)):
        t = asset_times[i]
        prev_t = asset_times[i-1]
        if asset_ts[prev_t] and asset_ts[prev_t] > 0:
            asset_returns[t] = (asset_ts[t] - asset_ts[prev_t]) / asset_ts[prev_t]

    results = {
        'available': True,
        'correlations': {},
        'lead_lag': [],
    }

    for ref_asset in REFERENCE_ASSETS:
        if ref_asset == asset:
            results['correlations'][ref_asset] = 1.0
            continue

        ref_data = await get_price_history(ref_asset, limit)
        if len(ref_data) < 48:
            continue

        ref_ts = {r['ts']: r['close'] for r in ref_data if r.get('close')}
        ref_times = sorted(ref_ts.keys())
        ref_returns = {}
        for i in range(1, len(ref_times)):
            t = ref_times[i]
            prev_t = ref_times[i-1]
            if ref_ts[prev_t] and ref_ts[prev_t] > 0:
                ref_returns[t] = (ref_ts[t] - ref_ts[prev_t]) / ref_ts[prev_t]

        # Find overlapping timestamps
        common_times = sorted(set(asset_returns.keys()) & set(ref_returns.keys()))
        if len(common_times) < 20:
            continue

        a_vals = np.array([asset_returns[t] for t in common_times])
        r_vals = np.array([ref_returns[t] for t in common_times])

        # Pearson correlation (lag 0)
        corr = _pearson(a_vals, r_vals)
        results['correlations'][ref_asset] = round(corr, 3)

        # Lead/lag detection: cross-correlate at lags -12h to +12h
        best_lag = 0
        best_corr = abs(corr)

        for lag in range(-12, 13):
            if lag == 0:
                continue
            if lag > 0:
                # ref leads asset by 'lag' hours
                a_shifted = a_vals[lag:]
                r_shifted = r_vals[:-lag]
            else:
                # asset leads ref by |lag| hours
                a_shifted = a_vals[:lag]
                r_shifted = r_vals[-lag:]

            if len(a_shifted) < 20:
                continue

            lag_corr = _pearson(a_shifted, r_shifted)
            if abs(lag_corr) > best_corr:
                best_corr = abs(lag_corr)
                best_lag = lag

        if best_lag != 0 and best_corr > 0.3:
            if best_lag > 0:
                leader = ref_asset
                follower = asset
                lag_h = best_lag
            else:
                leader = asset
                follower = ref_asset
                lag_h = -best_lag

            results['lead_lag'].append({
                'leader': leader,
                'follower': follower,
                'lag_hours': lag_h,
                'correlation': round(best_corr, 3),
            })

    return results


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    if len(a) < 2:
        return 0.0
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    a_diff = a - a_mean
    b_diff = b - b_mean
    numerator = np.sum(a_diff * b_diff)
    denominator = np.sqrt(np.sum(a_diff**2) * np.sum(b_diff**2))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


async def get_correlation_summary(asset: str) -> dict:
    """Get a concise correlation summary for display."""
    corr = await compute_correlations(asset)
    if not corr.get('available'):
        return corr

    summary = {
        'available': True,
        'btc_corr': corr['correlations'].get('BTC', 0),
        'spy_corr': corr['correlations'].get('SPY', 0),
        'gold_corr': corr['correlations'].get('GC=F', 0),
        'lead_lag': corr.get('lead_lag', []),
    }

    # Risk assessment based on correlations
    btc_corr = abs(summary['btc_corr'])
    spy_corr = abs(summary['spy_corr'])
    if btc_corr > 0.7:
        summary['risk_note'] = f"High BTC correlation ({summary['btc_corr']:.2f}) — crypto risk dominates"
    elif spy_corr > 0.7:
        summary['risk_note'] = f"High SPY correlation ({summary['spy_corr']:.2f}) — equity risk dominates"
    else:
        summary['risk_note'] = "Low cross-asset correlation — independent price action"

    return summary
