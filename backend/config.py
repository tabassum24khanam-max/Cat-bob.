"""
ULTRAMAX Config — Centralized constants, env vars, API key management
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─── API Keys (set via Railway Variables or .env file) ───────────────────
WORKER_URL = os.getenv("WORKER_URL", "https://winter-sunset-e359.azk40772corp.workers.dev")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
ALPACA_KEY = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def is_configured(key_name: str) -> bool:
    """Check if an API key is set."""
    return bool(globals().get(key_name, ""))


# ─── Asset Definitions ──────────────────────────────────────────────────────
ASSETS = {
    'crypto': ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE'],
    'stock':  ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'SPY'],
    'macro':  ['GC=F', 'CL=F', 'SI=F', 'XOM', 'LMT', 'RTX'],
}

ALL_ASSETS = ASSETS['crypto'] + ASSETS['stock'] + ASSETS['macro']

BINANCE_SYMBOLS = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT',
    'BNB': 'BNBUSDT', 'XRP': 'XRPUSDT', 'DOGE': 'DOGEUSDT',
}

YAHOO_SYMBOLS = {
    'AAPL': 'AAPL', 'TSLA': 'TSLA', 'NVDA': 'NVDA', 'MSFT': 'MSFT',
    'GOOGL': 'GOOGL', 'SPY': 'SPY', 'GC=F': 'GC=F', 'CL=F': 'CL=F',
    'SI=F': 'SI=F', 'XOM': 'XOM', 'LMT': 'LMT', 'RTX': 'RTX',
}

ASSET_NAMES = {
    'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana', 'BNB': 'BNB',
    'XRP': 'XRP', 'DOGE': 'Dogecoin', 'AAPL': 'Apple Inc', 'TSLA': 'Tesla',
    'NVDA': 'Nvidia', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet Google',
    'SPY': 'S&P 500 ETF', 'GC=F': 'Gold Futures', 'CL=F': 'WTI Crude Oil',
    'SI=F': 'Silver Futures', 'XOM': 'ExxonMobil', 'LMT': 'Lockheed Martin',
    'RTX': 'Raytheon Technologies', 'NOC': 'Northrop Grumman', 'GD': 'General Dynamics',
}

def get_asset_type(asset: str) -> str:
    """Return 'crypto', 'stock', or 'macro' for a given asset."""
    if asset in ASSETS['crypto']:
        return 'crypto'
    if asset in ASSETS['stock']:
        return 'stock'
    return 'macro'


# ─── Runtime settings (overridable from frontend) ──────────────────────────
_runtime_settings = {}

def get_setting(key: str, default=None):
    return _runtime_settings.get(key, globals().get(key, default))

def set_setting(key: str, value):
    _runtime_settings[key] = value
    # Also update module-level var if it exists
    if key in globals():
        globals()[key] = value
