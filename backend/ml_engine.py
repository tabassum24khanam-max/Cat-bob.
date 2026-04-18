"""
ULTRAMAX ML Engine — Random Forest + XGBoost Ensemble with Calibration
Replaces both the quant_agent.py embedded ML and ml_classifier.py
"""
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

MODEL_PATH = Path(__file__).parent / "data" / "ml_ensemble.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)

# Expanded feature set (~30 features)
FEATURE_COLS = [
    'rsi14', 'macd_hist', 'bb_pos', 'stoch_k', 'stoch_d',
    'dist_e20', 'dist_e50', 'vol_ratio', 'vol_percentile',
    'trend_slope', 'trend_stability', 'momentum_score',
    'hurst_exp', 'entropy_ratio', 'autocorr',
    'dist_vwap', 'cmf', 'price_zscore', 'will_r14',
    'hmm_trending', 'hmm_ranging', 'hmm_volatile',
    'ich_bull', 'ich_bear', 'supertrend_bull',
    'mom2', 'mom72', 'engulfing', 'doji',
    'pivot_dist_r1', 'pivot_dist_s1',
]

# Cache
_ensemble_cache = None
_cache_mtime = 0


def extract_features(ind: dict) -> list:
    """Extract normalized ML features from indicator dict."""
    hmm = ind.get('hmm_probs', {})
    cur = abs(ind.get('cur', 1)) or 1

    # Pivot distances (normalized)
    pivot_r1 = ind.get('pivot_r1', cur)
    pivot_s1 = ind.get('pivot_s1', cur)
    pivot_dist_r1 = (cur - pivot_r1) / cur * 100 if pivot_r1 else 0
    pivot_dist_s1 = (cur - pivot_s1) / cur * 100 if pivot_s1 else 0

    return [
        ind.get('rsi14', 50) / 100,
        max(-1, min(1, (ind.get('macd_hist', 0) or 0) / (cur * 0.001 + 1e-9))),
        ind.get('bb_pos', 0.5),
        ind.get('stoch_k', 50) / 100,
        ind.get('stoch_d', 50) / 100,
        max(-1, min(1, (ind.get('dist_e20', 0) or 0) / 5)),
        max(-1, min(1, (ind.get('dist_e50', 0) or 0) / 10)),
        min(3, (ind.get('vol_r', 1) or 1)),
        (ind.get('vol_percentile', 50) or 50) / 100,
        max(-1, min(1, (ind.get('trend_slope', 0) or 0) / 0.01)),
        ind.get('trend_stability', 0.5) or 0.5,
        max(-1, min(1, (ind.get('momentum_score', 0) or 0) / 5)),
        ind.get('hurst_exp', 0.5) or 0.5,
        ind.get('entropy_ratio', 0.5) or 0.5,
        max(-1, min(1, ind.get('autocorr', 0) or 0)),
        max(-1, min(1, (ind.get('dist_vwap', 0) or 0) / 3)),
        max(-1, min(1, ind.get('cmf', 0) or 0)),
        max(-3, min(3, ind.get('price_zscore', 0) or 0)) / 3,
        (ind.get('will_r14', -50) or -50) / -100,
        hmm.get('TRENDING', 0.33),
        hmm.get('RANGING', 0.33),
        hmm.get('VOLATILE', 0.33),
        1.0 if ind.get('ich_bull') else 0.0,
        1.0 if ind.get('ich_bear') else 0.0,
        1.0 if ind.get('supertrend_bull') else 0.0,
        max(-1, min(1, (ind.get('mom2', 0) or 0) / 3)),
        max(-1, min(1, (ind.get('mom72', 0) or 0) / 10)),
        float(ind.get('engulfing', 0) or 0),
        float(ind.get('doji', 0) or 0),
        max(-1, min(1, pivot_dist_r1 / 3)),
        max(-1, min(1, pivot_dist_s1 / 3)),
    ]


def load_ensemble():
    """Load ensemble model from disk."""
    global _ensemble_cache, _cache_mtime
    try:
        mtime = MODEL_PATH.stat().st_mtime
        if _ensemble_cache is None or mtime != _cache_mtime:
            with open(MODEL_PATH, 'rb') as f:
                _ensemble_cache = pickle.load(f)
            _cache_mtime = mtime
        return _ensemble_cache
    except Exception:
        return None


async def train_ensemble(predictions: list) -> dict:
    """Train XGBoost + Random Forest ensemble on rated prediction history."""
    try:
        from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import TimeSeriesSplit
    except ImportError as e:
        return {"ok": False, "error": f"Missing dependency: {e}"}

    # Filter rated trades with indicator data
    # Include NO_TRADE predictions that have original_decision (gate-blocked trades still have signal value)
    rated = []
    for p in predictions:
        if p.get('feedback') not in ('correct', 'wrong'):
            continue
        if not p.get('ind_snapshot'):
            continue
        direction = p.get('decision')
        if direction == 'NO_TRADE':
            direction = p.get('original_decision')
        if direction not in ('BUY', 'SELL'):
            continue
        rated.append({**p, '_effective_direction': direction})

    if len(rated) < 15:
        return {"ok": False, "error": f"Need 15+ rated trades with indicators, have {len(rated)}"}

    X, y = [], []
    for p in rated:
        try:
            ind = json.loads(p['ind_snapshot']) if isinstance(p['ind_snapshot'], str) else p['ind_snapshot']
            if not ind:
                continue
            features = extract_features(ind)
            direction = p['_effective_direction']
            moved_up = (p.get('feedback') == 'correct' and direction == 'BUY') or \
                       (p.get('feedback') == 'wrong' and direction == 'SELL')
            X.append(features)
            y.append(1 if moved_up else 0)
        except Exception:
            continue

    if len(X) < 15:
        return {"ok": False, "error": f"Insufficient valid samples after parsing ({len(X)})"}

    X = np.array(X)
    y = np.array(y)

    # XGBoost with calibration
    xgb_base = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, verbosity=0,
    )

    # Random Forest
    rf_base = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )

    # Cross-validation with TimeSeriesSplit
    cv_scores = {'xgb': [], 'rf': []}
    tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 10))

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        xgb_base.fit(X_tr, y_tr)
        rf_base.fit(X_tr, y_tr)

        cv_scores['xgb'].append(float(xgb_base.score(X_val, y_val)))
        cv_scores['rf'].append(float(rf_base.score(X_val, y_val)))

    # Final training on all data with calibration
    try:
        xgb_calibrated = CalibratedClassifierCV(xgb_base, cv=3, method='isotonic')
        xgb_calibrated.fit(X, y)
    except Exception:
        xgb_base.fit(X, y)
        xgb_calibrated = xgb_base

    try:
        rf_calibrated = CalibratedClassifierCV(rf_base, cv=3, method='isotonic')
        rf_calibrated.fit(X, y)
    except Exception:
        rf_base.fit(X, y)
        rf_calibrated = rf_base

    # Feature importance from uncalibrated models
    xgb_base.fit(X, y)
    rf_base.fit(X, y)
    xgb_importance = dict(zip(FEATURE_COLS, xgb_base.feature_importances_.tolist()))
    rf_importance = dict(zip(FEATURE_COLS, rf_base.feature_importances_.tolist()))
    combined_importance = {k: xgb_importance.get(k, 0) * 0.6 + rf_importance.get(k, 0) * 0.4
                          for k in FEATURE_COLS}
    top_features = sorted(combined_importance.items(), key=lambda x: -x[1])[:5]

    # Save ensemble
    model_data = {
        'xgb': xgb_calibrated,
        'rf': rf_calibrated,
        'n_train': len(X),
        'feature_cols': FEATURE_COLS,
        'up_rate': float(y.mean()),
        'cv_scores': cv_scores,
        'top_features': top_features,
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)

    # Update global cache
    global _ensemble_cache, _cache_mtime
    _ensemble_cache = model_data
    _cache_mtime = MODEL_PATH.stat().st_mtime

    avg_xgb_cv = sum(cv_scores['xgb']) / len(cv_scores['xgb']) if cv_scores['xgb'] else 0
    avg_rf_cv = sum(cv_scores['rf']) / len(cv_scores['rf']) if cv_scores['rf'] else 0

    return {
        "ok": True,
        "n_train": len(X),
        "up_rate": float(y.mean()),
        "top_features": top_features,
        "cv_accuracy": {"xgb": round(avg_xgb_cv * 100, 1), "rf": round(avg_rf_cv * 100, 1)},
    }


def predict_ensemble(ind: dict, direction: str = 'BUY') -> dict:
    """Run ensemble prediction on current indicators."""
    model_data = load_ensemble()
    if not model_data:
        return {"score": 50, "available": False, "n_train": 0}

    try:
        features = extract_features(ind)
        X = np.array([features])

        xgb_model = model_data['xgb']
        rf_model = model_data['rf']

        xgb_proba = xgb_model.predict_proba(X)[0]
        rf_proba = rf_model.predict_proba(X)[0]

        # Ensemble: XGBoost 60%, Random Forest 40%
        ensemble_prob_up = xgb_proba[1] * 0.6 + rf_proba[1] * 0.4

        # Adjust score based on direction
        score = ensemble_prob_up * 100 if direction == 'BUY' else (1 - ensemble_prob_up) * 100

        # Check agreement between models
        xgb_up = xgb_proba[1] > 0.5
        rf_up = rf_proba[1] > 0.5
        agreement = xgb_up == rf_up

        return {
            "score": round(score, 1),
            "available": True,
            "n_train": model_data.get('n_train', 0),
            "xgb_score": round(xgb_proba[1] * 100, 1),
            "rf_score": round(rf_proba[1] * 100, 1),
            "agreement": agreement,
            "top_features": model_data.get('top_features', []),
        }
    except Exception:
        return {"score": 50, "available": False, "n_train": 0}


def check_model_agreement(ind: dict, decision_direction: str) -> bool:
    """Check if ML ensemble agrees with the proposed decision direction."""
    result = predict_ensemble(ind, decision_direction)
    if not result['available']:
        return True  # No model = no disagreement
    return result['score'] > 50


def bayesian_confidence(rated_history: list, asset: str, horizon: int, ai_conf: float) -> float:
    """Bayesian posterior: blend AI confidence with historical win rate."""
    asset_rated = [p for p in rated_history
                   if p.get('asset') == asset and p.get('horizon') == horizon
                   and p.get('feedback') in ('correct', 'wrong')]

    if len(asset_rated) < 5:
        return ai_conf

    wins = sum(1 for p in asset_rated if p['feedback'] == 'correct')
    n = len(asset_rated)

    # Beta distribution posterior mean
    posterior = (wins + 1) / (n + 2)

    # Blend: weight historical more as n grows, cap at 40 samples
    hist_weight = min(0.6, n / 40)
    ai_weight = 1 - hist_weight

    blended = ai_conf * ai_weight + posterior * 100 * hist_weight
    return round(blended, 1)
