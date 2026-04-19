"""
ULTRAMAX ML Training Script
Fetches historical data from yfinance, computes all indicators, trains XGBoost + RF ensemble.
Generates 2000+ training samples from real market data across all supported assets.
"""
import sys
import json
import pickle
import math
import time
import numpy as np
from pathlib import Path

# Add parent to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from indicators import compute_indicators
from ml_engine import FEATURE_COLS, extract_features, MODEL_PATH

CRYPTO_TICKERS = {
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD',
    'BNB': 'BNB-USD', 'XRP': 'XRP-USD', 'DOGE': 'DOGE-USD',
}

STOCK_TICKERS = {
    'AAPL': 'AAPL', 'TSLA': 'TSLA', 'NVDA': 'NVDA', 'MSFT': 'MSFT',
    'GOOGL': 'GOOGL', 'SPY': 'SPY',
}

MACRO_TICKERS = {
    'GC=F': 'GC=F', 'CL=F': 'CL=F', 'SI=F': 'SI=F',
    'XOM': 'XOM', 'LMT': 'LMT', 'RTX': 'RTX',
}


def fetch_hourly_crypto(ticker: str, days: int = 59) -> list:
    """Fetch hourly data and aggregate to 4H candles."""
    import yfinance as yf
    df = yf.download(ticker, period=f'{days}d', interval='1h', progress=False)
    if df.empty:
        return []

    # Handle MultiIndex columns from yfinance
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)

    rows = []
    for idx, row in df.iterrows():
        rows.append({
            'time': idx,
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'volume': float(row['Volume']),
        })

    # Aggregate to 4H candles
    candles_4h = []
    for i in range(0, len(rows) - 3, 4):
        batch = rows[i:i+4]
        if len(batch) < 4:
            break
        candles_4h.append({
            'open': batch[0]['open'],
            'high': max(b['high'] for b in batch),
            'low': min(b['low'] for b in batch),
            'close': batch[-1]['close'],
            'volume': sum(b['volume'] for b in batch),
        })

    return candles_4h


def fetch_daily(ticker: str, years: int = 2) -> list:
    """Fetch daily candles."""
    import yfinance as yf
    df = yf.download(ticker, period=f'{years}y', interval='1d', progress=False)
    if df.empty:
        return []

    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)

    candles = []
    for idx, row in df.iterrows():
        candles.append({
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'volume': float(row['Volume']),
        })
    return candles


def generate_samples(candles: list, forward_periods: list = [1, 3, 6]) -> list:
    """
    Generate labeled training samples from candle data.
    For each candle (after warmup), compute indicators and label by future return.
    Uses multiple forward-look periods to multiply training data.
    """
    if len(candles) < 250:
        print(f"    Skipping: only {len(candles)} candles (need 250+)")
        return []

    samples = []
    warmup = 200

    for i in range(warmup, len(candles)):
        window = candles[:i+1]
        ind = compute_indicators(window)
        if not ind:
            continue

        features = extract_features(ind)
        if len(features) != len(FEATURE_COLS):
            continue

        for fwd in forward_periods:
            if i + fwd >= len(candles):
                break
            future_price = candles[i + fwd]['close']
            current_price = candles[i]['close']
            pct_change = (future_price - current_price) / current_price * 100

            # Label: 1 = went up, 0 = went down
            # Skip tiny moves (noise) — require at least 0.05% move
            if abs(pct_change) < 0.05:
                continue

            label = 1 if pct_change > 0 else 0
            samples.append({
                'features': features,
                'label': label,
                'pct_change': pct_change,
                'fwd_periods': fwd,
            })

    return samples


def parse_export_file() -> list:
    """Parse the user's export file for additional training data."""
    export_path = Path(__file__).parent.parent / "ultramax_predictions_export (3).txt"
    if not export_path.exists():
        print("  Export file not found, skipping")
        return []

    samples = []
    text = export_path.read_text()
    blocks = text.split('---' * 20 + '---' * 6 + '---' * 1)

    # Parse each prediction block
    current = {}
    indicators = {}
    in_indicators = False

    for line in text.split('\n'):
        line = line.strip()

        if line.startswith('#') and '|' in line:
            # Save previous
            if current and indicators and current.get('entry_price') and current.get('actual_price'):
                entry = current['entry_price']
                actual = current['actual_price']
                if entry > 0:
                    pct = (actual - entry) / entry * 100
                    if abs(pct) >= 0.05:
                        ind = {
                            'rsi14': indicators.get('RSI14', 50),
                            'stoch_k': indicators.get('StochK', 50),
                            'macd_hist': indicators.get('MACD', 0),
                            'bb_pos': indicators.get('BB', 0.5),
                            'cur': entry,
                            'atr': indicators.get('ATR', 0),
                            'vol_r': indicators.get('VolR', 1),
                            'trend_slope': indicators.get('Trend', 0),
                            'hurst_exp': indicators.get('Hurst', 0.5),
                            'entropy_ratio': indicators.get('Entropy', 0.5),
                            'dist_vwap': indicators.get('VWAP', 0),
                            'cmf': indicators.get('CMF', 0),
                            'momentum_score': indicators.get('Momentum', 0),
                            'autocorr': indicators.get('Autocorr', 0),
                            'price_zscore': indicators.get('Zscore', 0),
                        }
                        features = extract_features(ind)
                        label = 1 if pct > 0 else 0
                        samples.append({
                            'features': features,
                            'label': label,
                            'pct_change': pct,
                            'fwd_periods': 1,
                        })

            current = {}
            indicators = {}
            in_indicators = False

        elif line.startswith('Entry price:'):
            try:
                current['entry_price'] = float(line.split('$')[1].replace(',', ''))
            except:
                pass
        elif line.startswith('Actual at expiry:'):
            try:
                current['actual_price'] = float(line.split('$')[1].replace(',', ''))
            except:
                pass
        elif line.startswith('--- Indicators ---'):
            in_indicators = True
        elif in_indicators and '=' in line:
            for pair in line.split('  '):
                pair = pair.strip()
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    try:
                        indicators[k] = float(v)
                    except:
                        indicators[k] = v

    # Save last block
    if current and indicators and current.get('entry_price') and current.get('actual_price'):
        entry = current['entry_price']
        actual = current['actual_price']
        if entry > 0:
            pct = (actual - entry) / entry * 100
            if abs(pct) >= 0.05:
                ind = {
                    'rsi14': indicators.get('RSI14', 50),
                    'stoch_k': indicators.get('StochK', 50),
                    'macd_hist': indicators.get('MACD', 0),
                    'bb_pos': indicators.get('BB', 0.5),
                    'cur': entry,
                    'atr': indicators.get('ATR', 0),
                    'vol_r': indicators.get('VolR', 1),
                    'trend_slope': indicators.get('Trend', 0),
                    'hurst_exp': indicators.get('Hurst', 0.5),
                    'entropy_ratio': indicators.get('Entropy', 0.5),
                    'dist_vwap': indicators.get('VWAP', 0),
                    'cmf': indicators.get('CMF', 0),
                    'momentum_score': indicators.get('Momentum', 0),
                    'autocorr': indicators.get('Autocorr', 0),
                    'price_zscore': indicators.get('Zscore', 0),
                }
                features = extract_features(ind)
                label = 1 if pct > 0 else 0
                samples.append({
                    'features': features,
                    'label': label,
                    'pct_change': pct,
                    'fwd_periods': 1,
                })

    return samples


def train():
    """Main training function."""
    print("=" * 60)
    print("ULTRAMAX ML TRAINING")
    print("=" * 60)

    all_samples = []

    # 1. Fetch crypto 4H data
    print("\n[1/4] Fetching crypto hourly data (aggregating to 4H)...")
    for asset, ticker in CRYPTO_TICKERS.items():
        print(f"  {asset} ({ticker})...", end=" ", flush=True)
        candles = fetch_hourly_crypto(ticker)
        print(f"{len(candles)} 4H candles")
        if candles:
            samples = generate_samples(candles, forward_periods=[1, 3, 6])
            print(f"    → {len(samples)} training samples")
            all_samples.extend(samples)
        time.sleep(0.5)

    # 2. Fetch daily data for all assets (2 years)
    print("\n[2/4] Fetching 2-year daily data for all assets...")
    all_tickers = {**CRYPTO_TICKERS, **STOCK_TICKERS, **MACRO_TICKERS}
    for asset, ticker in all_tickers.items():
        print(f"  {asset} ({ticker})...", end=" ", flush=True)
        candles = fetch_daily(ticker)
        print(f"{len(candles)} daily candles")
        if candles:
            samples = generate_samples(candles, forward_periods=[1, 3, 5])
            print(f"    → {len(samples)} training samples")
            all_samples.extend(samples)
        time.sleep(0.3)

    # 3. Parse export file for user's actual predictions
    print("\n[3/4] Parsing user prediction export file...")
    export_samples = parse_export_file()
    print(f"  → {len(export_samples)} samples from export file")
    all_samples.extend(export_samples)

    print(f"\nTotal raw samples: {len(all_samples)}")

    if len(all_samples) < 100:
        print("ERROR: Not enough training data. Need at least 100 samples.")
        return False

    # 4. Train ensemble
    print("\n[4/4] Training XGBoost + Random Forest ensemble...")

    X = np.array([s['features'] for s in all_samples])
    y = np.array([s['label'] for s in all_samples])

    print(f"  Features shape: {X.shape}")
    print(f"  Up rate: {y.mean():.1%}")
    print(f"  Down rate: {(1-y.mean()):.1%}")

    # Replace any NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report

    # Shuffle data (we have mixed assets/timeframes so time-series split doesn't make sense)
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Hold out 15% for final validation
    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"  Train: {len(X_train)}, Validation: {len(X_val)}")

    # XGBoost — tuned for financial data
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.0,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )

    # Random Forest — robust against overfitting
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )

    # Gradient Boosting — additional ensemble member
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )

    # Train all three
    print("  Training XGBoost...", flush=True)
    xgb.fit(X_train, y_train)
    xgb_val_acc = accuracy_score(y_val, xgb.predict(X_val))
    print(f"    XGBoost validation accuracy: {xgb_val_acc:.1%}")

    print("  Training Random Forest...", flush=True)
    rf.fit(X_train, y_train)
    rf_val_acc = accuracy_score(y_val, rf.predict(X_val))
    print(f"    RF validation accuracy: {rf_val_acc:.1%}")

    print("  Training Gradient Boosting...", flush=True)
    gb.fit(X_train, y_train)
    gb_val_acc = accuracy_score(y_val, gb.predict(X_val))
    print(f"    GB validation accuracy: {gb_val_acc:.1%}")

    # Ensemble prediction on validation set
    xgb_proba = xgb.predict_proba(X_val)[:, 1]
    rf_proba = rf.predict_proba(X_val)[:, 1]
    gb_proba = gb.predict_proba(X_val)[:, 1]
    ensemble_proba = xgb_proba * 0.4 + rf_proba * 0.3 + gb_proba * 0.3
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    ensemble_acc = accuracy_score(y_val, ensemble_pred)
    print(f"\n  ENSEMBLE validation accuracy: {ensemble_acc:.1%}")

    print("\n  Classification Report (Ensemble):")
    print(classification_report(y_val, ensemble_pred, target_names=['DOWN', 'UP']))

    # Calibrate on full training data
    print("  Calibrating models...", flush=True)
    try:
        xgb_cal = CalibratedClassifierCV(xgb, cv=5, method='isotonic')
        xgb_cal.fit(X_train, y_train)
    except:
        xgb_cal = xgb

    try:
        rf_cal = CalibratedClassifierCV(rf, cv=5, method='isotonic')
        rf_cal.fit(X_train, y_train)
    except:
        rf_cal = rf

    # Feature importance
    xgb_imp = dict(zip(FEATURE_COLS, xgb.feature_importances_.tolist()))
    rf_imp = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))
    combined_imp = {k: xgb_imp.get(k, 0) * 0.5 + rf_imp.get(k, 0) * 0.5 for k in FEATURE_COLS}
    top_features = sorted(combined_imp.items(), key=lambda x: -x[1])[:10]

    print("\n  Top 10 most important features:")
    for feat, imp in top_features:
        print(f"    {feat}: {imp:.4f}")

    # Save model
    model_data = {
        'xgb': xgb_cal,
        'rf': rf_cal,
        'gb': gb,
        'n_train': len(X_train),
        'feature_cols': FEATURE_COLS,
        'up_rate': float(y.mean()),
        'cv_scores': {
            'xgb': xgb_val_acc,
            'rf': rf_val_acc,
            'gb': gb_val_acc,
            'ensemble': ensemble_acc,
        },
        'top_features': top_features[:5],
        'trained_on': 'historical_yfinance_2y_daily+60d_4h',
        'n_assets': len(all_tickers),
        'n_total_samples': len(X),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)

    model_size = MODEL_PATH.stat().st_size
    print(f"\n  Model saved to: {MODEL_PATH}")
    print(f"  Model size: {model_size / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"  Samples: {len(X)} ({len(X_train)} train + {len(X_val)} val)")
    print(f"  XGBoost accuracy: {xgb_val_acc:.1%}")
    print(f"  Random Forest accuracy: {rf_val_acc:.1%}")
    print(f"  Gradient Boosting accuracy: {gb_val_acc:.1%}")
    print(f"  ENSEMBLE accuracy: {ensemble_acc:.1%}")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = train()
    sys.exit(0 if success else 1)
