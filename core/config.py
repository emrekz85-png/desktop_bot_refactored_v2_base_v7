"""
Configuration and constants for the trading bot.
Centralized settings management for all modules.
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional

# ==========================================
# GOOGLE COLAB / HEADLESS MODE SUPPORT
# ==========================================
IS_COLAB = 'google.colab' in sys.modules or os.environ.get('COLAB_RELEASE_TAG') is not None
IS_HEADLESS = os.environ.get('HEADLESS_MODE', '').lower() in ('1', 'true', 'yes') or IS_COLAB
IS_NOTEBOOK = 'ipykernel' in sys.modules

# Try to import tqdm for progress bars
try:
    if IS_NOTEBOOK:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

# ==========================================
# SYMBOLS AND TIMEFRAMES
# ==========================================
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "LINKUSDT",
    "BNBUSDT", "XRPUSDT", "LTCUSDT", "DOGEUSDT", "SUIUSDT", "FARTCOINUSDT"
]

# HTF Only Mode (Optional)
# True = Only use 1h+ timeframes (fewer trades, less noise)
# False = Use all timeframes (5m, 15m, 1h, 4h, 12h, 1d)
HTF_ONLY_MODE = False

_ALL_LOWER_TIMEFRAMES = ["5m", "15m", "30m", "1h"]
LOWER_TIMEFRAMES = ["1h"] if HTF_ONLY_MODE else _ALL_LOWER_TIMEFRAMES
HTF_TIMEFRAMES = ["4h", "12h", "1d"]
TIMEFRAMES = LOWER_TIMEFRAMES + HTF_TIMEFRAMES

# ==========================================
# DATA DIRECTORY SETUP
# ==========================================
# All CSV, JSON and temporary files are stored here
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "trades.csv")
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
BEST_CONFIGS_FILE = os.path.join(DATA_DIR, "best_configs.json")
BACKTEST_META_FILE = os.path.join(DATA_DIR, "backtest_meta.json")
POT_LOG_FILE = os.path.join(DATA_DIR, "potential_trades.json")
DYNAMIC_BLACKLIST_FILE = os.path.join(DATA_DIR, "dynamic_blacklist.json")

# ==========================================
# CANDLE LIMITS
# ==========================================
BACKTEST_CANDLE_LIMITS = {
    "1m": 100000,
    "5m": 100000,
    "15m": 100000,
    "30m": 100000,
    "1h": 100000,
    "4h": 50000,
    "12h": 30000,
    "1d": 20000,
}

DAILY_REPORT_CANDLE_LIMITS = {
    "1m": 15000,
    "5m": 15000,
    "15m": 15000,
    "30m": 15000,
    "1h": 15000,
    "4h": 8000,
    "12h": 5000,
    "1d": 4000,
}

# Day -> Candle conversion constants (candles per day for each TF)
CANDLES_PER_DAY = {
    "1m": 1440,   # 24 * 60
    "5m": 288,    # 24 * 60 / 5
    "15m": 96,    # 24 * 60 / 15
    "30m": 48,    # 24 * 60 / 30
    "1h": 24,     # 24
    "4h": 6,      # 24 / 4
    "12h": 2,     # 24 / 12
    "1d": 1,      # 1
}

# Minutes per candle for each timeframe (for funding calculation)
MINUTES_PER_CANDLE = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "12h": 720,
    "1d": 1440,
}


def days_to_candles(days: int, timeframe: str) -> int:
    """Convert number of days to candle count for a given timeframe."""
    cpd = CANDLES_PER_DAY.get(timeframe, 24)  # default 1h
    return days * cpd


def days_to_candles_map(days: int, timeframes: list = None) -> dict:
    """Convert number of days to candle count dict for all timeframes."""
    if timeframes is None:
        timeframes = TIMEFRAMES
    return {tf: days_to_candles(days, tf) for tf in timeframes}


# ==========================================
# TRADING CONFIGURATION (ECONOMIC MODEL)
# ==========================================
TRADING_CONFIG = {
    "initial_balance": 2000.0,
    "leverage": 10,
    "usable_balance_pct": 0.20,  # 20% of balance
    "risk_per_trade_pct": 0.0175,  # 1.75% risk per trade
    "max_portfolio_risk_pct": 0.05,  # 5% total portfolio risk
    "slippage_rate": 0.0005,     # 0.05% slippage
    "funding_rate_8h": 0.0001,   # 0.01% funding (8 hour period)
    "maker_fee": 0.0002,         # 0.02% maker fee
    "taker_fee": 0.0005,         # 0.05% taker fee
    "total_fee": 0.0007          # 0.07% (entry + exit estimate) - safety margin
}

# ==========================================
# R-MULTIPLE BASED OPTIMIZER GATING
# ==========================================
# Multi-layer gating to prevent weak-edge configs from trading:
# 1. Minimum E[R] threshold (account-size independent)
# 2. Minimum score threshold (varies by timeframe)
# 3. Confidence-based risk multiplier
# 4. Walk-forward out-of-sample validation

MIN_EXPECTANCY_R_MULTIPLE = {
    "1m": 0.10,   # Very noisy - need strong edge
    "5m": 0.06,   # Noisy - need decent edge per trade
    "15m": 0.05,  # Moderate noise
    "30m": 0.04,
    "1h": 0.04,   # Cleaner signals
    "4h": 0.03,   # Low noise
    "1d": 0.02,   # Very clean
}

# DEPRECATED: Old $/trade thresholds (kept for backward compatibility)
MIN_EXPECTANCY_PER_TRADE = {
    "1m": 6.0,
    "5m": 4.0,
    "15m": 3.0,
    "30m": 2.5,
    "1h": 2.0,
    "4h": 1.5,
    "1d": 1.0,
}

MIN_SCORE_THRESHOLD = {
    "1m": 80.0,
    "5m": 40.0,
    "15m": 15.0,
    "30m": 10.0,
    "1h": 8.0,
    "4h": 5.0,
    "1d": 3.0,
}

CONFIDENCE_RISK_MULTIPLIER = {
    "high": 1.0,    # Full risk
    "medium": 0.5,  # Half risk - protects against optimizer overfitting
    "low": 0.0,     # No trades (effectively disabled)
}

# Post-portfolio pruning blacklist
POST_PORTFOLIO_BLACKLIST = {}

# ==========================================
# DYNAMIC BLACKLIST SYSTEM
# ==========================================
DYNAMIC_BLACKLIST_CONFIG = {
    "enabled": True,
    "negative_threshold": -20.0,
    "positive_threshold": 10.0,
    "min_trades_required": 3,
    "consecutive_losses_required": 1,
}

DYNAMIC_BLACKLIST_CACHE = {}


def load_dynamic_blacklist() -> dict:
    """Load dynamic blacklist from file. Returns dict of {(symbol, timeframe): info}"""
    global DYNAMIC_BLACKLIST_CACHE
    try:
        if os.path.exists(DYNAMIC_BLACKLIST_FILE):
            with open(DYNAMIC_BLACKLIST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                DYNAMIC_BLACKLIST_CACHE = {}
                for key, val in data.get("blacklist", {}).items():
                    if "|" in key:
                        sym, tf = key.split("|")
                        DYNAMIC_BLACKLIST_CACHE[(sym, tf)] = val
                return DYNAMIC_BLACKLIST_CACHE
    except Exception as e:
        print(f"[DYNAMIC_BLACKLIST] Load error: {e}")
    return {}


def save_dynamic_blacklist(blacklist: dict, summary_info: dict = None):
    """Save dynamic blacklist to file."""
    global DYNAMIC_BLACKLIST_CACHE
    DYNAMIC_BLACKLIST_CACHE = blacklist
    try:
        serializable = {}
        for (sym, tf), info in blacklist.items():
            serializable[f"{sym}|{tf}"] = info

        data = {
            "blacklist": serializable,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "config": DYNAMIC_BLACKLIST_CONFIG,
        }
        if summary_info:
            data["last_backtest_summary"] = summary_info

        with open(DYNAMIC_BLACKLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[DYNAMIC_BLACKLIST] Save error: {e}")


def update_dynamic_blacklist(summary_rows: list) -> dict:
    """Update dynamic blacklist based on backtest results."""
    if not DYNAMIC_BLACKLIST_CONFIG.get("enabled", True):
        return DYNAMIC_BLACKLIST_CACHE

    neg_thresh = DYNAMIC_BLACKLIST_CONFIG.get("negative_threshold", -20.0)
    pos_thresh = DYNAMIC_BLACKLIST_CONFIG.get("positive_threshold", 10.0)
    min_trades = DYNAMIC_BLACKLIST_CONFIG.get("min_trades_required", 3)

    current_blacklist = load_dynamic_blacklist()
    added = []
    removed = []

    for row in summary_rows:
        sym = row.get("symbol")
        tf = row.get("timeframe")
        pnl = row.get("net_pnl", 0)
        trades = row.get("trades", 0)
        win_rate = row.get("win_rate_pct", 0)

        if not sym or not tf:
            continue

        key = (sym, tf)

        if trades < min_trades:
            continue

        if pnl < neg_thresh:
            if key not in current_blacklist:
                added.append(f"{sym}-{tf} (PnL=${pnl:.2f}, {trades} trades)")
            current_blacklist[key] = {
                "reason": "negative_pnl",
                "pnl": pnl,
                "trades": trades,
                "win_rate": win_rate,
                "blacklisted_at": datetime.utcnow().isoformat() + "Z",
            }
        elif pnl > pos_thresh:
            if key in current_blacklist:
                removed.append(f"{sym}-{tf} (PnL=${pnl:.2f}, {trades} trades)")
                del current_blacklist[key]

    if added:
        print(f"\n[DYNAMIC_BLACKLIST] Added to blacklist:")
        for item in added:
            print(f"   - {item}")

    if removed:
        print(f"\n[DYNAMIC_BLACKLIST] Removed from blacklist:")
        for item in removed:
            print(f"   - {item}")

    if not added and not removed:
        print(f"\n[DYNAMIC_BLACKLIST] No changes. Current blacklist: {len(current_blacklist)} streams")

    summary_info = {
        "total_streams": len(summary_rows),
        "blacklisted_streams": len(current_blacklist),
        "added_count": len(added),
        "removed_count": len(removed),
    }
    save_dynamic_blacklist(current_blacklist, summary_info)

    return current_blacklist


def is_stream_blacklisted(symbol: str, timeframe: str) -> bool:
    """Check if a stream is in any blacklist (static or dynamic)."""
    key = (symbol, timeframe)

    # Check static blacklist first
    if key in POST_PORTFOLIO_BLACKLIST:
        return True

    # Check dynamic blacklist
    if not DYNAMIC_BLACKLIST_CACHE:
        load_dynamic_blacklist()
    return key in DYNAMIC_BLACKLIST_CACHE


# ==========================================
# WALK-FORWARD CONFIGURATION
# ==========================================
WALK_FORWARD_CONFIG = {
    "enabled": True,
    "train_ratio": 0.70,
    "oos_min_trades": 3,
    "overfit_ratio_threshold": 0.30,
}

# ==========================================
# OTHER SETTINGS
# ==========================================
REFRESH_RATE = 3
ENABLE_CHARTS = False
AUTO_RESTART_DELAY_SECONDS = 5

# Best config cache
BEST_CONFIG_CACHE = {}
BEST_CONFIG_WARNING_FLAGS = {
    "missing_signature": False,
    "signature_mismatch": False,
    "json_error": False,  # Bozuk JSON dosyası hatası için flag
    "load_error": False,  # Genel yükleme hatası için flag
}
