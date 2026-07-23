"""
ULTRAMAX ML Trainer — trains the ensemble on REAL historical market data
(2 years of hourly candles from Yahoo) PLUS the bot's own rated trades.

Why this exists: the old path (ml_engine.train_ensemble) trained only on the
bot's own trades — often as few as 15 samples with 35 features, i.e. massive
overfitting, and it saved every asset into one shared file. This module builds
a proper per-asset dataset of thousands of real market moments, labels each by
what price actually did next, and trains an honest, out-of-sample-validated
model. The bot's own trades are folded in when available so the model keeps
learning from our live results too.

Data source note: Binance is geo-blocked from some hosts (HTTP 451), so we use
Yahoo for everything here — it serves both stocks (market-hours bars) and crypto
(24/7 via the `-USD` tickers). In production the live fetch still uses whatever
data_fetcher provides; this trainer only needs a reliable history source.
"""
import asyncio
import gzip
import json
import pickle
import time
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

from indicators import compute_indicators
from ml_engine import extract_features, FEATURE_COLS, _asset_model_path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ── Yahoo history fetch ──────────────────────────────────────────────────────
def _yahoo_symbol(asset: str) -> str:
    """Map an ULTRAMAX asset code to a Yahoo ticker."""
    from config import BINANCE_SYMBOLS, YAHOO_SYMBOLS
    if asset in BINANCE_SYMBOLS:          # crypto → Yahoo 24/7 ticker
        return f"{asset}-USD"
    return YAHOO_SYMBOLS.get(asset, asset)  # stocks/futures already Yahoo-style


async def fetch_history(asset: str, interval: str = "1h", rng: str = "2y") -> list:
    """Fetch deep candle history from Yahoo. Returns list of OHLCV dicts."""
    sym = _yahoo_symbol(asset)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.get(url, params={"range": rng, "interval": interval},
                                headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code != 200:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                res = r.json().get("chart", {}).get("result", [{}])[0]
                q = res.get("indicators", {}).get("quote", [{}])[0]
                ts = res.get("timestamp", []) or []
                out = []
                closes = q.get("close", [])
                for i, t in enumerate(ts):
                    if i < len(closes) and closes[i] is not None:
                        out.append({
                            "time": t,
                            "open": q["open"][i], "high": q["high"][i],
                            "low": q["low"][i], "close": q["close"][i],
                            "volume": (q.get("volume") or [0])[i] or 0,
                        })
                return out
        except Exception:
            await asyncio.sleep(1.5 * (attempt + 1))
    return []


# ── Dataset construction ─────────────────────────────────────────────────────
def build_dataset(candles: list, horizon_bars: int = 1, warmup: int = 300,
                  max_samples: int = 8000, step: int = 1, deadband_pct: float = 0.0):
    """Roll through history: at each bar compute the same indicators the live bot
    sees, then label 1 if price is higher `horizon_bars` ahead, else 0.

    deadband_pct: if >0, bars whose forward move is within ±deadband are dropped
    (removes pure-noise flat bars so the model learns real up/down, not coin-flips)."""
    X, y = [], []
    n = len(candles)
    last_usable = n - horizon_bars
    # Only walk the most recent window needed to hit max_samples (keeps it bounded)
    start = max(warmup, last_usable - max_samples * step)
    for i in range(start, last_usable, step):
        window = candles[max(0, i - warmup):i + 1]
        if len(window) < 60:
            continue
        try:
            ind = compute_indicators(window)
            if not ind:
                continue
            cur = candles[i]["close"]
            fut = candles[i + horizon_bars]["close"]
            move_pct = (fut - cur) / cur * 100 if cur else 0
            if deadband_pct > 0 and abs(move_pct) < deadband_pct:
                continue
            X.append(extract_features(ind))
            y.append(1 if fut > cur else 0)
        except Exception:
            continue
    return X, y


def _our_data_samples(rated_preds: list):
    """Turn the bot's own rated trades into (features, label) pairs."""
    X, y = [], []
    for p in rated_preds:
        if p.get("feedback") not in ("correct", "wrong"):
            continue
        snap = p.get("ind_snapshot")
        if not snap:
            continue
        direction = p.get("decision")
        if direction == "NO_TRADE":
            direction = p.get("original_decision")
        if direction not in ("BUY", "SELL"):
            continue
        try:
            ind = json.loads(snap) if isinstance(snap, str) else snap
            if not ind:
                continue
            moved_up = (p["feedback"] == "correct" and direction == "BUY") or \
                       (p["feedback"] == "wrong" and direction == "SELL")
            X.append(extract_features(ind))
            y.append(1 if moved_up else 0)
        except Exception:
            continue
    return X, y


# ── Training ─────────────────────────────────────────────────────────────────
def _train_models(X: np.ndarray, y: np.ndarray) -> dict:
    """Train XGB + RF with isotonic calibration and honest TimeSeriesSplit CV."""
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit

    xgb_base = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    rf_base = RandomForestClassifier(
        n_estimators=200, max_depth=7, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    )

    # Honest out-of-sample accuracy: always train on the PAST, test on the FUTURE
    cv = {"xgb": [], "rf": [], "ens": []}
    n_splits = min(5, max(2, len(X) // 400))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tscv.split(X):
        if len(np.unique(y[tr])) < 2:
            continue
        xgb_base.fit(X[tr], y[tr])
        rf_base.fit(X[tr], y[tr])
        cv["xgb"].append(float(xgb_base.score(X[va], y[va])))
        cv["rf"].append(float(rf_base.score(X[va], y[va])))
        ens = (xgb_base.predict_proba(X[va])[:, 1] * 0.6 +
               rf_base.predict_proba(X[va])[:, 1] * 0.4) > 0.5
        cv["ens"].append(float((ens.astype(int) == y[va]).mean()))

    # Final fit on all data, with calibration
    try:
        xgb = CalibratedClassifierCV(xgb_base, cv=3, method="isotonic")
        xgb.fit(X, y)
    except Exception:
        xgb_base.fit(X, y); xgb = xgb_base
    try:
        rf = CalibratedClassifierCV(rf_base, cv=3, method="isotonic")
        rf.fit(X, y)
    except Exception:
        rf_base.fit(X, y); rf = rf_base

    xgb_base.fit(X, y); rf_base.fit(X, y)
    imp = {k: xgb_base.feature_importances_[i] * 0.6 + rf_base.feature_importances_[i] * 0.4
           for i, k in enumerate(FEATURE_COLS)}
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:6]

    def _avg(a): return round(sum(a) / len(a) * 100, 1) if a else 0.0
    return {
        "xgb": xgb, "rf": rf,
        "cv_xgb": _avg(cv["xgb"]), "cv_rf": _avg(cv["rf"]),
        "cv_accuracy": _avg(cv["ens"]),
        "top_features": top,
    }


async def train_asset(asset: str, horizon_bars: int = 1, max_samples: int = 8000,
                      deadband_pct: float = 0.0, include_our_data: bool = True) -> dict:
    """Full pipeline for one asset: fetch history → build dataset → fold in our
    trades → train → save per-asset model. Returns a summary dict."""
    t0 = time.time()
    candles = await fetch_history(asset)
    if len(candles) < 500:
        return {"asset": asset, "ok": False, "error": f"only {len(candles)} history bars"}

    Xh, yh = build_dataset(candles, horizon_bars=horizon_bars,
                           max_samples=max_samples, deadband_pct=deadband_pct)
    n_hist = len(Xh)

    n_our = 0
    if include_our_data:
        try:
            from database import get_predictions
            rated = await get_predictions(asset, 500)
            Xo, yo = _our_data_samples(rated)
            n_our = len(Xo)
            Xh += Xo; yh += yo
        except Exception:
            pass

    if len(Xh) < 200:
        return {"asset": asset, "ok": False, "error": f"only {len(Xh)} usable samples"}

    X = np.array(Xh, dtype=float)
    y = np.array(yh, dtype=int)
    # guard: drop rows with nan/inf
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]
    if len(np.unique(y)) < 2:
        return {"asset": asset, "ok": False, "error": "labels are single-class"}

    trained = _train_models(X, y)

    model_data = {
        "xgb": trained["xgb"], "rf": trained["rf"],
        "n_train": int(len(X)),
        "n_historical": int(n_hist),
        "n_our_data": int(n_our),
        "feature_cols": FEATURE_COLS,
        "up_rate": float(y.mean()),
        "cv_accuracy": trained["cv_accuracy"],
        "cv_xgb": trained["cv_xgb"],
        "cv_rf": trained["cv_rf"],
        "top_features": trained["top_features"],
        "horizon_bars": horizon_bars,
        "trained_at": int(time.time()),
        "source": "yahoo_2y_1h+our_trades",
    }
    path = _asset_model_path(asset)
    with gzip.open(path, "wb", compresslevel=6) as f:
        pickle.dump(model_data, f)

    return {
        "asset": asset, "ok": True,
        "n_train": int(len(X)), "n_historical": int(n_hist), "n_our_data": int(n_our),
        "cv_accuracy": trained["cv_accuracy"],
        "cv_xgb": trained["cv_xgb"], "cv_rf": trained["cv_rf"],
        "up_rate": round(float(y.mean()) * 100, 1),
        "top_features": [t[0] for t in trained["top_features"]],
        "secs": round(time.time() - t0, 1),
        "path": str(path.name),
    }


async def train_all(assets: Optional[list] = None, **kw) -> list:
    """Train every asset (or a given subset). Sequential to bound memory."""
    from config import ALL_ASSETS
    assets = assets or ALL_ASSETS
    results = []
    for a in assets:
        try:
            r = await train_asset(a, **kw)
        except Exception as e:
            r = {"asset": a, "ok": False, "error": str(e)[:150]}
        results.append(r)
        tag = "✓" if r.get("ok") else "✗"
        print(f"{tag} {a}: {r}")
    return results


if __name__ == "__main__":
    import sys
    subset = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    out = asyncio.run(train_all(subset))
    ok = [r for r in out if r.get("ok")]
    print(f"\nTrained {len(ok)}/{len(out)} assets.")
