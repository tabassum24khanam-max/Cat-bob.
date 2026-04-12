"""
ULTRAMAX Cluster Engine — K-means clustering of historical market states
Groups similar market conditions and tracks forward return distributions
"""
import json
import time
import numpy as np
from database import get_price_history, save_clusters, get_clusters


# Same 13-dimension state vector used in similarity search
STATE_FEATURES = [
    'rsi14', 'macd_hist', 'dist_vwap', 'trend_slope', 'trend_stability',
    'vol_percentile', 'momentum_score', 'hurst_exp', 'entropy_ratio',
    'autocorr', 'hmm_trending', 'hmm_ranging', 'hmm_volatile',
]

N_CLUSTERS = 50  # 50 clusters per asset
MIN_DATA_POINTS = 500  # Minimum history needed


def _extract_state_vector(row: dict) -> list:
    """Extract normalized 13-dim state vector from a price_data row."""
    cur = row.get('close', 1) or 1
    return [
        (row.get('rsi14', 50) or 50) / 100,
        max(-1, min(1, (row.get('macd_hist', 0) or 0) / (abs(cur) * 0.001 + 0.0001))),
        max(-1, min(1, (row.get('dist_vwap', 0) or 0) / 5)),
        max(-1, min(1, (row.get('trend_slope', 0) or 0) / 0.3)),
        row.get('trend_stability', 0) or 0,
        (row.get('vol_percentile', 50) or 50) / 100,
        max(-1, min(1, (row.get('momentum_score', 0) or 0) / 5)),
        row.get('hurst_exp', 0.5) or 0.5,
        row.get('entropy_ratio', 0.5) or 0.5,
        max(-1, min(1, row.get('autocorr', 0) or 0)),
        row.get('hmm_trending', 0.33) or 0.33,
        row.get('hmm_ranging', 0.33) or 0.33,
        row.get('hmm_volatile', 0.33) or 0.33,
    ]


async def rebuild_clusters(asset: str) -> dict:
    """Rebuild K-means clusters for an asset from historical data."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return {'ok': False, 'error': 'scikit-learn not installed'}

    # Fetch all historical data
    data = await get_price_history(asset, limit=50000)
    if len(data) < MIN_DATA_POINTS:
        return {'ok': False, 'error': f'Need {MIN_DATA_POINTS}+ data points, have {len(data)}'}

    # Build feature matrix
    valid_rows = []
    vectors = []
    for row in data:
        vec = _extract_state_vector(row)
        # Skip rows with all zeros (missing data)
        if any(v != 0 for v in vec):
            valid_rows.append(row)
            vectors.append(vec)

    if len(vectors) < MIN_DATA_POINTS:
        return {'ok': False, 'error': f'Insufficient valid data points: {len(vectors)}'}

    X = np.array(vectors, dtype=float)

    # Fit K-means
    n_clusters = min(N_CLUSTERS, len(X) // 10)  # At least 10 members per cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X)

    # Compute forward return stats per cluster
    cluster_data = []
    for cid in range(n_clusters):
        mask = labels == cid
        cluster_rows = [valid_rows[i] for i in range(len(valid_rows)) if mask[i]]
        n_members = len(cluster_rows)

        if n_members == 0:
            continue

        # Forward returns
        fwd_1h = [r.get('fwd_1h') for r in cluster_rows if r.get('fwd_1h') is not None]
        fwd_4h = [r.get('fwd_4h') for r in cluster_rows if r.get('fwd_4h') is not None]
        fwd_8h = [r.get('fwd_8h') for r in cluster_rows if r.get('fwd_8h') is not None]
        fwd_1d = [r.get('fwd_1d') for r in cluster_rows if r.get('fwd_1d') is not None]

        # Win rate (4h: % of times price went up > 0.2%)
        wins_4h = sum(1 for f in fwd_4h if f and f > 0.2) if fwd_4h else 0
        win_rate = (wins_4h / len(fwd_4h) * 100) if fwd_4h else 50

        cluster_data.append({
            'cluster_id': cid,
            'centroid': kmeans.cluster_centers_[cid].tolist(),
            'n_members': n_members,
            'avg_fwd_1h': sum(fwd_1h) / len(fwd_1h) if fwd_1h else None,
            'avg_fwd_4h': sum(fwd_4h) / len(fwd_4h) if fwd_4h else None,
            'avg_fwd_8h': sum(fwd_8h) / len(fwd_8h) if fwd_8h else None,
            'avg_fwd_1d': sum(fwd_1d) / len(fwd_1d) if fwd_1d else None,
            'win_rate_4h': round(win_rate, 1),
        })

    # Save to database
    await save_clusters(asset, cluster_data)

    return {
        'ok': True,
        'asset': asset,
        'n_clusters': len(cluster_data),
        'total_points': len(vectors),
        'avg_members': len(vectors) // max(1, len(cluster_data)),
    }


async def assign_cluster(asset: str, indicators: dict) -> dict:
    """Find the nearest cluster for current market state."""
    clusters = await get_clusters(asset)
    if not clusters:
        return {'available': False, 'reason': 'No clusters built yet'}

    # Build current state vector
    current_vec = _extract_state_vector_from_indicators(indicators)
    current_np = np.array(current_vec, dtype=float)

    # Find nearest cluster by Euclidean distance
    best_cluster = None
    best_dist = float('inf')

    for cluster in clusters:
        centroid = np.array(cluster['centroid'], dtype=float)
        dist = np.linalg.norm(current_np - centroid)
        if dist < best_dist:
            best_dist = dist
            best_cluster = cluster

    if best_cluster is None:
        return {'available': False, 'reason': 'No matching cluster'}

    return {
        'available': True,
        'cluster_id': best_cluster['cluster_id'],
        'n_members': best_cluster['n_members'],
        'avg_fwd_1h': best_cluster.get('avg_fwd_1h'),
        'avg_fwd_4h': best_cluster.get('avg_fwd_4h'),
        'avg_fwd_8h': best_cluster.get('avg_fwd_8h'),
        'avg_fwd_1d': best_cluster.get('avg_fwd_1d'),
        'win_rate_4h': best_cluster.get('win_rate_4h', 50),
        'distance': round(float(best_dist), 4),
    }


def _extract_state_vector_from_indicators(ind: dict) -> list:
    """Extract state vector from compute_indicators() output format."""
    cur = ind.get('cur', 1) or 1
    return [
        ind.get('rsi14', 50) / 100,
        max(-1, min(1, (ind.get('macd_hist', 0) or 0) / (abs(cur) * 0.001 + 0.0001))),
        max(-1, min(1, (ind.get('dist_vwap', 0) or 0) / 5)),
        max(-1, min(1, (ind.get('trend_slope', 0) or 0) / 0.3)),
        ind.get('trend_stability', 0),
        ind.get('vol_percentile', 50) / 100,
        max(-1, min(1, (ind.get('momentum_score', 0) or 0) / 5)),
        ind.get('hurst_exp', 0.5),
        ind.get('entropy_ratio', 0.5),
        max(-1, min(1, ind.get('autocorr', 0))),
        ind.get('hmm_probs', {}).get('TRENDING', 0.33),
        ind.get('hmm_probs', {}).get('RANGING', 0.33),
        ind.get('hmm_probs', {}).get('VOLATILE', 0.33),
    ]
