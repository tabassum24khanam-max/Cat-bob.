"""
ULTRAMAX ML Classifier — XGBoost
Trains on your rated prediction history, improves over time
"""
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any

MODEL_PATH = Path(__file__).parent.parent / "data" / "ml_model.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)

FEATURE_COLS = [
    'rsi14', 'macd_hist', 'bb_pos', 'stoch_k',
    'dist_e20', 'dist_e50', 'vol_ratio', 'vol_percentile',
    'trend_slope', 'trend_stability', 'momentum_score',
    'hurst_exp', 'entropy_ratio', 'autocorr',
    'dist_vwap', 'cmf', 'price_zscore', 'will_r14',
    'hmm_trending', 'hmm_ranging', 'hmm_volatile',
    'ich_bull', 'ich_bear', 'supertrend_bull',
]

_model_cache = None
_model_mtime = 0

def load_model():
    global _model_cache, _model_mtime
    try:
        mtime = MODEL_PATH.stat().st_mtime
        if _model_cache is None or mtime != _model_mtime:
            with open(MODEL_PATH, 'rb') as f:
                _model_cache = pickle.load(f)
            _model_mtime = mtime
        return _model_cache
    except:
        return None

def extract_features(ind: dict) -> list:
    """Extract ML features from indicator dict."""
    hmm = ind.get('hmm_probs', {})
    return [
        ind.get('rsi14', 50) / 100,
        max(-1, min(1, (ind.get('macd_hist', 0) or 0) / (abs(ind.get('cur', 1) or 1) * 0.001 + 1e-9))),
        ind.get('bb_pos', 0.5),
        ind.get('stoch_k', 50) / 100,
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
    ]

def predict_ml(ind: dict, direction: str) -> dict:
    """Return ML classifier confidence for a given direction."""
    model_data = load_model()
    if not model_data:
        return {"score": 50, "available": False, "n_train": 0}

    try:
        model = model_data['model']
        n_train = model_data.get('n_train', 0)
        features = extract_features(ind)
        X = np.array([features])
        proba = model.predict_proba(X)[0]
        # proba[1] = prob of going up
        score = proba[1] * 100 if direction == 'BUY' else (1 - proba[1]) * 100
        return {"score": round(score, 1), "available": True, "n_train": n_train}
    except Exception as e:
        return {"score": 50, "available": False, "n_train": 0, "error": str(e)}

async def retrain_model(predictions: list) -> dict:
    """Retrain XGBoost on rated predictions."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"ok": False, "error": "xgboost not installed"}

    # Filter: only rated directional trades (BUY/SELL correct/wrong)
    rated = [p for p in predictions
             if p.get('feedback') in ('correct', 'wrong')
             and p.get('decision') in ('BUY', 'SELL')
             and p.get('ind_snapshot')]

    if len(rated) < 20:
        return {"ok": False, "error": f"Need 20+ rated trades, have {len(rated)}"}

    X, y = [], []
    for p in rated:
        try:
            ind = json.loads(p['ind_snapshot']) if isinstance(p['ind_snapshot'], str) else p['ind_snapshot']
            features = extract_features(ind)
            # Label: 1 if price went up, 0 if down
            moved_up = p.get('feedback') == 'correct' and p.get('decision') == 'BUY' or \
                       p.get('feedback') == 'wrong' and p.get('decision') == 'SELL'
            X.append(features)
            y.append(1 if moved_up else 0)
        except:
            continue

    if len(X) < 20:
        return {"ok": False, "error": "Insufficient valid samples after parsing"}

    X = np.array(X)
    y = np.array(y)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    # Save
    model_data = {
        'model': model,
        'n_train': len(X),
        'feature_cols': FEATURE_COLS,
        'up_rate': float(y.mean()),
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)

    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    top5 = sorted(importance.items(), key=lambda x: -x[1])[:5]

    return {
        "ok": True,
        "n_train": len(X),
        "up_rate": float(y.mean()),
        "top_features": top5,
    }
