"""
ULTRAMAX Meta-Labeling Model — learns when the PRIMARY system is right.

Standard quant-fund practice (Lopez de Prado): the primary system stays at its
~51-55% directional accuracy, and a second model is trained on the primary's
OWN track record to answer one question — "given the machine just said BUY/SELL
in these conditions, what is the probability it's correct?" That probability
becomes the position size. The meta-model never predicts markets; it predicts
the machine. It finds the slice of conditions where the machine is actually
58%+ and bets there, and starves the coin-flip slices of capital.

Trains on the predictions DB (rated rows with ind_snapshot), logistic
regression — small data, needs calibrated probabilities, must not overfit.
Gracefully returns None (=> neutral sizing) until >=40 rated samples exist.
"""
import gzip
import json
import math
import pickle
import time

import numpy as np

from ml_engine import _DATA_DIR

META_PATH = _DATA_DIR / "meta_model.pkl.gz"
MIN_SAMPLES = 40

_cache = {"model": None, "mtime": 0.0}


def _features_from_row(confidence, ml_score, ind: dict, decision: str,
                        asset_type: str, ts_ms) -> list:
    hour = time.gmtime((ts_ms or 0) / 1000).tm_hour if ts_ms else 12
    return [
        (confidence or 50) / 100.0,
        (ml_score or 50) / 100.0,
        float(ind.get('hurst_exp', 0.5) or 0.5),
        float(ind.get('entropy_ratio', 0.5) or 0.5),
        (ind.get('adx', 20) or 20) / 100.0,
        (ind.get('rsi14', 50) or 50) / 100.0,
        (ind.get('vol_percentile', 50) or 50) / 100.0,
        1.0 if asset_type == 'crypto' else 0.0,
        1.0 if asset_type == 'stock' else 0.0,
        math.sin(2 * math.pi * hour / 24.0),
        math.cos(2 * math.pi * hour / 24.0),
        1.0 if decision == 'BUY' else 0.0,
    ]


def build_training_set(rated_predictions: list):
    """rated_predictions: rows from the predictions table with feedback set."""
    from config import get_asset_type
    X, y = [], []
    for p in rated_predictions:
        if p.get('feedback') not in ('correct', 'wrong'):
            continue
        snap = p.get('ind_snapshot')
        if not snap:
            continue
        try:
            ind = json.loads(snap) if isinstance(snap, str) else snap
            decision = p.get('decision')
            if decision == 'NO_TRADE':
                decision = p.get('original_decision')
            if decision not in ('BUY', 'SELL'):
                continue
            X.append(_features_from_row(p.get('confidence'), p.get('ml_score'),
                                         ind or {}, decision,
                                         get_asset_type(p.get('asset', '')),
                                         p.get('saved_at')))
            y.append(1 if p['feedback'] == 'correct' else 0)
        except Exception:
            continue
    return X, y


def train_meta_model(rated_predictions: list) -> dict:
    """Train + persist. Call from a thread (sklearn fit is CPU-bound)."""
    X, y = build_training_set(rated_predictions)
    if len(X) < MIN_SAMPLES:
        return {"ok": False, "reason": f"need {MIN_SAMPLES}+ rated samples, have {len(X)}"}
    if len(set(y)) < 2:
        return {"ok": False, "reason": "labels single-class"}
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    Xa, ya = np.array(X, float), np.array(y, int)
    mask = np.isfinite(Xa).all(axis=1)
    Xa, ya = Xa[mask], ya[mask]
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5)
    try:
        cv = cross_val_score(clf, Xa, ya, cv=min(5, max(2, len(Xa) // 15)))
        cv_acc = round(float(cv.mean()) * 100, 1)
    except Exception:
        cv_acc = None
    clf.fit(Xa, ya)
    data = {"model": clf, "n_train": int(len(Xa)), "base_rate": float(ya.mean()),
            "cv_accuracy": cv_acc, "trained_at": int(time.time())}
    with gzip.open(META_PATH, 'wb') as f:
        pickle.dump(data, f)
    _cache["model"] = data
    _cache["mtime"] = META_PATH.stat().st_mtime
    return {"ok": True, "n_train": len(Xa), "base_rate": round(float(ya.mean()), 3),
            "cv_accuracy": cv_acc}


def _load():
    try:
        if not META_PATH.exists():
            return None
        mt = META_PATH.stat().st_mtime
        if _cache["model"] is None or _cache["mtime"] != mt:
            with gzip.open(META_PATH, 'rb') as f:
                _cache["model"] = pickle.load(f)
            _cache["mtime"] = mt
        return _cache["model"]
    except Exception:
        return None


def meta_probability(confidence, ml_score, ind: dict, decision: str,
                      asset_type: str) -> float | None:
    """P(this trade is correct | conditions), or None if no model yet."""
    data = _load()
    if not data:
        return None
    try:
        feats = _features_from_row(confidence, ml_score, ind, decision,
                                    asset_type, int(time.time() * 1000))
        return float(data["model"].predict_proba(np.array([feats]))[0][1])
    except Exception:
        return None


def meta_status() -> dict:
    data = _load()
    if not data:
        return {"available": False, "reason": f"not trained (needs {MIN_SAMPLES}+ rated trades)"}
    return {"available": True, "n_train": data.get("n_train"),
            "base_rate": round(data.get("base_rate", 0), 3),
            "cv_accuracy": data.get("cv_accuracy"),
            "trained_at": data.get("trained_at")}
