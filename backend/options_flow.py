"""ULTRAMAX Options Flow — Free delayed options data via yfinance"""
import asyncio

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from config import ASSETS, BINANCE_SYMBOLS

# Assets that support options (stocks only)
STOCK_ASSETS = set(ASSETS.get('stock', []))


def _fetch_options(symbol: str) -> dict:
    """Synchronous helper to fetch options data from yfinance.
    Must be run in an executor since yfinance is blocking.
    """
    try:
        ticker = yf.Ticker(symbol)
        expiry_dates = ticker.options
        if not expiry_dates:
            return {'available': False}

        # Use nearest expiry
        chain = ticker.option_chain(expiry_dates[0])
        calls = chain.calls
        puts = chain.puts

        if calls.empty and puts.empty:
            return {'available': False}

        total_call_vol = int(calls['volume'].sum()) if 'volume' in calls.columns else 0
        total_put_vol = int(puts['volume'].sum()) if 'volume' in puts.columns else 0
        total_call_oi = int(calls['openInterest'].sum()) if 'openInterest' in calls.columns else 0
        total_put_oi = int(puts['openInterest'].sum()) if 'openInterest' in puts.columns else 0

        # Put/Call ratio based on volume
        if total_call_vol > 0:
            put_call_ratio = round(total_put_vol / total_call_vol, 4)
        elif total_call_oi > 0:
            put_call_ratio = round(total_put_oi / total_call_oi, 4)
        else:
            put_call_ratio = 1.0

        # Max pain: strike with highest combined open interest
        all_strikes = set()
        call_oi_map = {}
        put_oi_map = {}

        if 'strike' in calls.columns and 'openInterest' in calls.columns:
            for _, row in calls.iterrows():
                strike = float(row['strike'])
                oi = float(row['openInterest']) if not _is_nan(row['openInterest']) else 0
                all_strikes.add(strike)
                call_oi_map[strike] = oi

        if 'strike' in puts.columns and 'openInterest' in puts.columns:
            for _, row in puts.iterrows():
                strike = float(row['strike'])
                oi = float(row['openInterest']) if not _is_nan(row['openInterest']) else 0
                all_strikes.add(strike)
                put_oi_map[strike] = oi

        max_pain = 0.0
        max_pain_oi = 0.0
        for strike in all_strikes:
            combined_oi = call_oi_map.get(strike, 0) + put_oi_map.get(strike, 0)
            if combined_oi > max_pain_oi:
                max_pain_oi = combined_oi
                max_pain = strike

        # Unusual activity: options with volume > 3x open interest
        unusual_calls = 0
        unusual_puts = 0

        if 'volume' in calls.columns and 'openInterest' in calls.columns:
            for _, row in calls.iterrows():
                vol = float(row['volume']) if not _is_nan(row['volume']) else 0
                oi = float(row['openInterest']) if not _is_nan(row['openInterest']) else 0
                if oi > 0 and vol > 3 * oi:
                    unusual_calls += 1

        if 'volume' in puts.columns and 'openInterest' in puts.columns:
            for _, row in puts.iterrows():
                vol = float(row['volume']) if not _is_nan(row['volume']) else 0
                oi = float(row['openInterest']) if not _is_nan(row['openInterest']) else 0
                if oi > 0 and vol > 3 * oi:
                    unusual_puts += 1

        # Bias determination
        if put_call_ratio < 0.7:
            bias = 'bullish'
        elif put_call_ratio > 1.3:
            bias = 'bearish'
        else:
            bias = 'neutral'

        return {
            'available': True,
            'put_call_ratio': put_call_ratio,
            'max_pain': round(max_pain, 2),
            'unusual_calls': unusual_calls,
            'unusual_puts': unusual_puts,
            'bias': bias,
        }

    except Exception:
        return {'available': False}


def _is_nan(value) -> bool:
    """Check if a value is NaN."""
    try:
        import math
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


async def get_options_sentiment(asset: str) -> dict:
    """Get put/call ratio and unusual activity from free options data.
    Returns: {available, put_call_ratio, max_pain,
              unusual_calls: int, unusual_puts: int,
              bias: 'bullish'|'bearish'|'neutral'}
    """
    try:
        # Only works for stocks — not crypto or commodities
        if asset in BINANCE_SYMBOLS:
            return {'available': False}

        if asset not in STOCK_ASSETS:
            return {'available': False}

        if not HAS_YFINANCE:
            return {'available': False}

        # yfinance is blocking, run in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _fetch_options, asset)
        return result

    except Exception:
        return {'available': False}
