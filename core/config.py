"""
Configuration and constants for the trading bot.
Centralized settings management for all modules.

Thread Safety Notes:
- BEST_CONFIG_CACHE and DYNAMIC_BLACKLIST_CACHE are protected by locks
- Use the provided accessor functions for thread-safe access
"""

# ==========================================
# VERSION INFORMATION
# ==========================================
# Format: "vMAJOR.MINOR"
# - Major: Significant changes (new features, breaking changes)
# - Minor: Bug fixes, parameter tweaks, small improvements
#
# Changelog:
# v1.8.1 - P5 High Priority: AT flat threshold, PBEMA distance, optimizer scoring, WF validation fix
# v1.8.0 - P5 Critical Fixes: ADX filter skip, body position skip, SSL never lost off, realistic slippage
# v1.7.2 - Grid Search Optimizer: Configurable regime gating params for optimization
# v1.7.1 - TF-Adaptive SSL Lookback: 5m=HMA75, 15m=HMA60, 1h=HMA45
# v1.7.1-dyntp - REVERTED: Dynamic Partial TP by Regime slightly hurt H2 (-$0.50)
# v1.7.1-atmomentum - REVERTED: AlphaTrend Momentum filter too strict (0 trades in both H1 and H2)
# v1.7.0 - Window-Level Regime Gating: ADX_avg < 20 = skip trade (RANGING markets)
# v1.7-adxmax - REVERTED: ADX max filter caused -$316 loss (affected optimizer, eliminated best trends)
# v1.6.2-restored - Baseline restored after v1.7-adxmax failure
# v1.6.3 - REVERTED: Relaxed good config criteria caused -$106 loss (carry-forward in changed regime)
# v1.6.2 - Carry-forward max age extended (2 → 4 windows) for more active windows
# v1.6.1 - SSL touch tolerance relaxed (0.002 → 0.003) for more signals
# v1.6 - Relaxed optimizer thresholds for higher trade frequency
# v1.5 - Momentum TP Extension uses EMA15 instead of AlphaTrend (faster, lower lag)
# v1.4 - Momentum TP Extension fixed (AlphaTrend data now passed to trade updates)
# v1.3 - Progressive Partial TP (3-tranche system for profit locking)
# v1.8.2 - P5.2 REVERTED: Aggressive filter relaxation worse than baseline (-$124)
# v1.2 - Momentum TP Extension (let winners run when momentum strong)
# v1.1 - Tighter swing SL (20->10 candles) - REVERTED (worse performance)
# v1.0 - Initial versioning system, baseline strategy
VERSION = "v1.8.2"

import os
import sys
import json
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple


def _utcnow():
    """Helper to get current UTC time as naive datetime (replaces deprecated _utcnow())."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

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
    "BTCUSDT",  # Only BTC - other symbols removed for testing
]

# HTF Only Mode (Optional)
# True = Only use 1h+ timeframes (fewer trades, less noise)
# False = Use all timeframes (5m, 15m, 1h, 4h, 12h, 1d)
HTF_ONLY_MODE = False

# ==========================================
# M3 APPLE SILICON PERFORMANCE SETTINGS
# ==========================================
# Optimized for M3 MacBook Air (4 P-cores + 4 E-cores, fansiz)
# P-cores: High performance, CPU-intensive tasks
# E-cores: ~40% slower, background/IO tasks only
# Fansiz tasarim: Sureli yuk = thermal throttling riski
#
# Worker Settings:
# - CPU_WORKERS: 3 (P-cores - 1 main thread)
# - IO_WORKERS: 6 (network-bound, doesnt stress CPU)
#
# Memory: UMA (Unified Memory Architecture)
# - 8GB/16GB paylaşımlı (CPU + GPU)
# - DataFrame copies minimize edilmeli
# - PyArrow backend + downcast onerilir
M3_PERFORMANCE_CONFIG = {
    "cpu_workers": 3,  # P-cores only (E-cores yavaş)
    "io_workers": 6,   # Network/disk bound
    "thermal_burst_seconds": 180,  # 3 dk burst, sonra throttle riski
    "min_warmup_bars": 250,  # Indicator warmup
}

_ALL_LOWER_TIMEFRAMES = ["15m"]  # Only 15m - other TFs removed for testing
LOWER_TIMEFRAMES = _ALL_LOWER_TIMEFRAMES
HTF_TIMEFRAMES = []  # HTF disabled for testing
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
    # v1.7.3: Position sizing tested - REVERTED
    # 1.75% → $157.10 | 2.0% → $78.75 | 2.5% → $86.75 (ALL WORSE)
    # Higher risk changes optimizer config selection → worse configs chosen
    "risk_per_trade_pct": 0.0175,  # 1.75% risk per trade (OPTIMAL)
    "max_portfolio_risk_pct": 0.05,  # 5% total portfolio risk (OPTIMAL)
    "slippage_rate": 0.0005,     # BASELINE: 0.05% slippage
    "sl_slippage_multiplier": 1.0, # BASELINE: No extra SL slippage
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
    "5m": 0.02,   # Relaxed from 0.06 (v1.6)
    "15m": 0.03,  # Relaxed from 0.05 (v1.6)
    "30m": 0.03,  # Relaxed from 0.04 (v1.6)
    "1h": 0.05,   # Relaxed from 0.08 (v1.6)
    "4h": 0.07,   # Relaxed from 0.10 (v1.6)
    "12h": 0.08,  # Relaxed from 0.12 (v1.6)
    "1d": 0.10,   # Relaxed from 0.15 (v1.6)
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

# Thread-safe cache with lock
DYNAMIC_BLACKLIST_CACHE: Dict[Tuple[str, str], Any] = {}
_DYNAMIC_BLACKLIST_LOCK = threading.RLock()  # RLock for recursive access


def load_dynamic_blacklist() -> dict:
    """Load dynamic blacklist from file. Returns dict of {(symbol, timeframe): info}

    Thread-safe: Uses lock to prevent race conditions.
    """
    global DYNAMIC_BLACKLIST_CACHE
    with _DYNAMIC_BLACKLIST_LOCK:
        try:
            if os.path.exists(DYNAMIC_BLACKLIST_FILE):
                with open(DYNAMIC_BLACKLIST_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    DYNAMIC_BLACKLIST_CACHE = {}
                    for key, val in data.get("blacklist", {}).items():
                        if "|" in key:
                            sym, tf = key.split("|")
                            DYNAMIC_BLACKLIST_CACHE[(sym, tf)] = val
                    return DYNAMIC_BLACKLIST_CACHE.copy()
        except json.JSONDecodeError as e:
            from .logging_config import get_logger
            get_logger(__name__).error(f"[DYNAMIC_BLACKLIST] JSON parse error: {e}")
        except OSError as e:
            from .logging_config import get_logger
            get_logger(__name__).error(f"[DYNAMIC_BLACKLIST] File read error: {e}")
    return {}


def save_dynamic_blacklist(blacklist: dict, summary_info: dict = None):
    """Save dynamic blacklist to file.

    Thread-safe: Uses lock to prevent race conditions.
    """
    global DYNAMIC_BLACKLIST_CACHE
    with _DYNAMIC_BLACKLIST_LOCK:
        DYNAMIC_BLACKLIST_CACHE = blacklist.copy()
        try:
            serializable = {}
            for (sym, tf), info in blacklist.items():
                serializable[f"{sym}|{tf}"] = info

            data = {
                "blacklist": serializable,
                "updated_at": _utcnow().isoformat() + "Z",
                "config": DYNAMIC_BLACKLIST_CONFIG,
            }
            if summary_info:
                data["last_backtest_summary"] = summary_info

            with open(DYNAMIC_BLACKLIST_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            from .logging_config import get_logger
            get_logger(__name__).error(f"[DYNAMIC_BLACKLIST] Save error: {e}")


def update_dynamic_blacklist(summary_rows: list) -> dict:
    """Update dynamic blacklist based on backtest results.

    Thread-safe: Uses lock for blacklist modifications.
    """
    from .logging_config import get_logger
    logger = get_logger(__name__)

    with _DYNAMIC_BLACKLIST_LOCK:
        if not DYNAMIC_BLACKLIST_CONFIG.get("enabled", True):
            return DYNAMIC_BLACKLIST_CACHE.copy()

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
                    "blacklisted_at": _utcnow().isoformat() + "Z",
                }
            elif pnl > pos_thresh:
                if key in current_blacklist:
                    removed.append(f"{sym}-{tf} (PnL=${pnl:.2f}, {trades} trades)")
                    del current_blacklist[key]

        if added:
            logger.info("[DYNAMIC_BLACKLIST] Added to blacklist: %s", ", ".join(added))

        if removed:
            logger.info("[DYNAMIC_BLACKLIST] Removed from blacklist: %s", ", ".join(removed))

        if not added and not removed:
            logger.debug("[DYNAMIC_BLACKLIST] No changes. Current blacklist: %d streams", len(current_blacklist))

        summary_info = {
            "total_streams": len(summary_rows),
            "blacklisted_streams": len(current_blacklist),
            "added_count": len(added),
            "removed_count": len(removed),
        }
        save_dynamic_blacklist(current_blacklist, summary_info)

        return current_blacklist


def is_stream_blacklisted(symbol: str, timeframe: str) -> bool:
    """Check if a stream is in any blacklist (static or dynamic).

    Thread-safe: Uses lock for dynamic blacklist access.
    """
    key = (symbol, timeframe)

    # Check static blacklist first
    if key in POST_PORTFOLIO_BLACKLIST:
        return True

    # Check dynamic blacklist (thread-safe)
    with _DYNAMIC_BLACKLIST_LOCK:
        if not DYNAMIC_BLACKLIST_CACHE:
            load_dynamic_blacklist()
        return key in DYNAMIC_BLACKLIST_CACHE


# ==========================================
# WALK-FORWARD CONFIGURATION
# ==========================================
# Overfitting'i önlemek için:
# 1. Veriyi train (in-sample) ve test (out-of-sample) olarak böl
# 2. Train verisinde optimize et
# 3. Test verisinde sadece test et (tekrar optimize etme)
# 4. Test E[R] / Train E[R] oranı "overfit ratio"
# 5. Overfit ratio < min_overfit_ratio ise config reddedilir
WALK_FORWARD_CONFIG = {
    "enabled": True,            # Walk-forward testi etkinleştir
    "train_ratio": 0.70,        # %70 train, %30 test
    "min_test_trades": 3,       # Test için minimum trade sayısı (base)
    "min_overfit_ratio": 0.60,  # Test E[R] / Train E[R] minimum oranı (0.70→0.60 for Optuna)
}

# Timeframe-based minimum OOS trades (quant trader recommendation)
# Lower timeframes need more OOS trades to be statistically significant
# v1.6: Relaxed to increase trade frequency (was causing 64% zero-trade windows)
MIN_OOS_TRADES_BY_TF = {
    "1m": 20,   # Unchanged (rarely used)
    "5m": 10,   # Relaxed from 15 (v1.6)
    "15m": 6,   # Relaxed from 12 (v1.6)
    "30m": 5,   # Relaxed from 10 (v1.6)
    "1h": 4,    # Relaxed from 8 (v1.6)
    "4h": 3,    # Relaxed from 8 (v1.6)
    "12h": 2,   # Added missing timeframe (v1.6)
    "1d": 2,    # Relaxed from 5 (v1.6)
}

# ==========================================
# CIRCUIT BREAKER CONFIGURATION
# ==========================================
# 2-Level kill switch to prevent catastrophic losses:
# Level 1: Stream-level (individual symbol/timeframe)
# Level 2: Global (entire portfolio)
CIRCUIT_BREAKER_CONFIG = {
    "enabled": True,

    # --- LEVEL 1: STREAM CIRCUIT BREAKER ---
    # Shuts down individual symbol/timeframe when loss limit reached
    "stream_max_loss": -200.0,              # Max $ loss per stream before kill
    "stream_max_drawdown_dollars": 100.0,   # Max $ drawdown from stream peak PnL
    "stream_min_trades_before_kill": 5,     # Minimum trades before circuit breaker activates
    # Note: We use DOLLAR-based drawdown for streams (not percentage) because
    # percentage-based gives absurd results when initial balance is shared across streams

    # --- LEVEL 2: GLOBAL CIRCUIT BREAKER ---
    # Shuts down entire bot when portfolio-level limits reached
    "global_daily_max_loss": -400.0,    # Max daily $ loss across all streams
    "global_weekly_max_loss": -800.0,   # Max weekly $ loss across all streams
    "global_max_drawdown_pct": 0.20,    # Max 20% drawdown from portfolio peak equity
}

# ==========================================
# ROLLING E[R] CHECK CONFIGURATION
# ==========================================
# Monitors recent trade performance to detect regime shifts
# Shuts down stream when edge disappears (rolling E[R] goes negative)
ROLLING_ER_CONFIG = {
    "enabled": True,

    # Window sizes by timeframe (more trades needed for noisy TFs)
    "window_by_tf": {
        "1m": 30,
        "5m": 25,
        "15m": 20,
        "30m": 15,
        "1h": 12,
        "4h": 8,
        "1d": 5,
    },

    # EMA smoothing factor (0 = no smoothing, 1 = full smoothing)
    "ema_alpha": 0.3,

    # Minimum trades before rolling check activates
    "min_trades_before_check": 10,

    # Threshold: E[R] below this triggers warning, below negative triggers kill
    "warning_threshold": 0.02,  # Warn when rolling E[R] < 0.02
    "kill_threshold": -0.05,    # Kill stream when rolling E[R] < -0.05 (allowing some noise)

    # Confidence band: mean - (stdev * factor) < 0 triggers kill
    "use_confidence_band": True,
    "confidence_band_factor": 0.5,  # More conservative than 1.0 stdev
}

# ==========================================
# TF-ADAPTIVE THRESHOLDS SYSTEM
# ==========================================
# Percentage-based thresholds that scale appropriately across timeframes.
# 15m is the baseline (proven working) - other TFs adapt from this.
#
# Design rationale:
# - Lower TFs (5m): Need TIGHTER thresholds due to more noise
# - Higher TFs (1h+): Need LOOSER thresholds as moves are larger
#
# Thresholds explained:
# - flat_threshold: AlphaTrend flat detection (higher = more signals, line must move more)
# - ssl_touch_tolerance: Baseline touch detection (higher = more lenient touch)
# - min_pbema_distance: Minimum distance to TP target (lower on HTF = smaller % moves)
# - overlap_threshold: SSL-PBEMA overlap detection (higher = allow closer overlap)
# - lookback_candles: Candles to check for baseline touch (fewer on HTF = same time)

TF_THRESHOLDS = {
    # 5m: Tighter thresholds (more noise filtering)
    "5m": {
        "flat_threshold": 0.0025,       # 0.25% - slightly tighter than 15m
        "ssl_touch_tolerance": 0.0025,  # 0.25% - tighter touch detection
        "min_pbema_distance": 0.005,    # 0.5% - need more room for noise
        "overlap_threshold": 0.004,     # 0.4% - tighter overlap check
        "lookback_candles": 6,          # More candles (6 * 5m = 30min)
    },
    # 15m: BASELINE - proven working, DO NOT CHANGE
    "15m": {
        "flat_threshold": 0.002,        # 0.2% - baseline
        "ssl_touch_tolerance": 0.003,   # 0.3% - baseline
        "min_pbema_distance": 0.004,    # 0.4% - baseline
        "overlap_threshold": 0.005,     # 0.5% - baseline
        "lookback_candles": 5,          # 5 candles (5 * 15m = 75min)
    },
    # 30m: Slightly looser
    "30m": {
        "flat_threshold": 0.0018,       # 0.18% - slightly looser
        "ssl_touch_tolerance": 0.004,   # 0.4% - looser touch
        "min_pbema_distance": 0.0035,   # 0.35% - slightly tighter (larger moves)
        "overlap_threshold": 0.006,     # 0.6% - allow more overlap
        "lookback_candles": 4,          # 4 candles (4 * 30m = 2h)
    },
    # 1h: Looser thresholds for larger moves
    "1h": {
        "flat_threshold": 0.0015,       # 0.15% - looser (bigger moves needed)
        "ssl_touch_tolerance": 0.005,   # 0.5% - looser touch detection
        "min_pbema_distance": 0.003,    # 0.3% - smaller % for same $ move
        "overlap_threshold": 0.007,     # 0.7% - allow more overlap
        "lookback_candles": 4,          # 4 candles (4 * 1h = 4h)
    },
    # 4h: Much looser for swing trading
    "4h": {
        "flat_threshold": 0.001,        # 0.1% - very loose
        "ssl_touch_tolerance": 0.008,   # 0.8% - very loose touch
        "min_pbema_distance": 0.002,    # 0.2% - smaller % target
        "overlap_threshold": 0.01,      # 1.0% - generous overlap allowance
        "lookback_candles": 3,          # 3 candles (3 * 4h = 12h)
    },
    # 12h: Position trading thresholds
    "12h": {
        "flat_threshold": 0.0008,       # 0.08% - very loose
        "ssl_touch_tolerance": 0.01,    # 1.0% - very loose touch
        "min_pbema_distance": 0.0015,   # 0.15% - small % target
        "overlap_threshold": 0.012,     # 1.2% - generous overlap
        "lookback_candles": 2,          # 2 candles (2 * 12h = 24h)
    },
    # 1d: Daily timeframe (very loose)
    "1d": {
        "flat_threshold": 0.0006,       # 0.06% - minimal movement detection
        "ssl_touch_tolerance": 0.012,   # 1.2% - very loose touch
        "min_pbema_distance": 0.001,    # 0.1% - small % target
        "overlap_threshold": 0.015,     # 1.5% - very generous overlap
        "lookback_candles": 2,          # 2 candles (2 * 1d = 2 days)
    },
}

# Base thresholds (15m values) - used as fallback
BASE_TF_THRESHOLDS = TF_THRESHOLDS["15m"].copy()


# ==========================================
# ALPHATREND 3-LAYER VALIDATION THRESHOLDS
# ==========================================
# Enhanced AlphaTrend filtering to prevent fake signals in ranging markets.
#
# Based on mathematical analysis of AT behavior across timeframes:
# - recent_cross_lookback: How recently the AT cross (blue crosses red) must have occurred
# - momentum_lookback: How many candles to confirm AT line continues moving in trade direction
# - min_line_separation: Minimum % gap between blue/red lines to confirm real momentum
#
# Signal is VALID only when ALL 3 conditions are met:
# 1. AT cross occurred within recent_cross_lookback candles
# 2. Winning line (blue for LONG, red for SHORT) has been rising/falling for momentum_lookback candles
# 3. Gap between lines >= min_line_separation
#
# If ANY condition fails -> signal is FAKE (ranging market, no momentum)

AT_VALIDATION_THRESHOLDS = {
    # TF-ADAPTIVE LINE SEPARATION (Quant Agent Recommendation 2026-01-02)
    # Lower TF = more noise = tighter threshold
    # Higher TF = cleaner moves = looser threshold (allow more signals)

    # 1m: Most noise, need strict filtering
    "1m": {
        "recent_cross_lookback": 20,       # Cross within 20 minutes (relaxed)
        "momentum_lookback": 3,            # 3 candles of momentum
        "min_line_separation": 0.0015,     # 0.15% - tighter for noisy TF
    },
    # 5m: Still noisy, moderately strict
    "5m": {
        "recent_cross_lookback": 16,       # Cross within ~1.5 hours
        "momentum_lookback": 3,            # 3 candles = 15 min of momentum
        "min_line_separation": 0.0012,     # 0.12% - tighter for noise
    },
    # 15m: BASELINE - loosened from 0.1% to 0.08% to reduce overfit
    "15m": {
        "recent_cross_lookback": 16,       # Cross within 4 hours (16 * 15m)
        "momentum_lookback": 3,            # 3 candles = 45 min of momentum
        "min_line_separation": 0.0008,     # 0.08% - LOOSER to allow more signals
    },
    # 30m: Slightly looser
    "30m": {
        "recent_cross_lookback": 12,       # Cross within 6 hours
        "momentum_lookback": 3,            # 3 candles = 1.5 hours of momentum
        "min_line_separation": 0.0007,     # 0.07%
    },
    # 1h: HTF - looser thresholds for bigger moves
    "1h": {
        "recent_cross_lookback": 10,       # Cross within 10 hours
        "momentum_lookback": 3,            # 3 candles = 3 hours of momentum
        "min_line_separation": 0.0006,     # 0.06% - looser for HTF
    },
    # 4h: Swing trading - loose thresholds
    "4h": {
        "recent_cross_lookback": 8,        # Cross within 32 hours
        "momentum_lookback": 3,            # 3 candles = 12 hours of momentum
        "min_line_separation": 0.0004,     # 0.04% - loosest for swing trading
    },
    # 12h: Position trading
    "12h": {
        "recent_cross_lookback": 6,        # Cross within 3 days
        "momentum_lookback": 2,            # 2 candles = 1 day of momentum
        "min_line_separation": 0.0003,     # 0.03%
    },
    # 1d: Daily timeframe - loosest thresholds
    "1d": {
        "recent_cross_lookback": 5,        # Cross within 5 days
        "momentum_lookback": 2,            # 2 days of consistent momentum
        "min_line_separation": 0.0002,     # 0.02% - loosest for daily
    },
}

# Base AT validation thresholds (15m values) - used as fallback
BASE_AT_VALIDATION_THRESHOLDS = AT_VALIDATION_THRESHOLDS["15m"].copy()


def get_at_validation_thresholds(timeframe: str) -> dict:
    """
    Get AlphaTrend 3-layer validation thresholds for a timeframe.

    These thresholds are used to filter fake AT signals in ranging markets.
    A signal is only valid when:
    1. Cross occurred within recent_cross_lookback candles
    2. Winning line moved in signal direction for momentum_lookback candles
    3. Line separation >= min_line_separation

    Args:
        timeframe: Timeframe string (e.g., "15m", "1h", "4h")

    Returns:
        Dict with AT validation thresholds:
        - recent_cross_lookback: int (candles since last cross)
        - momentum_lookback: int (candles to check momentum)
        - min_line_separation: float (minimum % gap between lines)

    Example:
        >>> get_at_validation_thresholds("4h")
        {'recent_cross_lookback': 5, 'momentum_lookback': 4, 'min_line_separation': 0.005}
    """
    return AT_VALIDATION_THRESHOLDS.get(timeframe, BASE_AT_VALIDATION_THRESHOLDS).copy()


def get_tf_threshold(threshold_name: str, timeframe: str) -> float:
    """
    Get TF-adaptive threshold value.

    This provides timeframe-appropriate thresholds for strategy filters.
    Uses 15m as baseline (proven working) and adapts for other timeframes.

    Args:
        threshold_name: One of:
            - "flat_threshold": AlphaTrend flat detection
            - "ssl_touch_tolerance": Baseline touch detection
            - "min_pbema_distance": Minimum distance to TP target
            - "overlap_threshold": SSL-PBEMA overlap detection
            - "lookback_candles": Candles for baseline touch check
        timeframe: Timeframe string (e.g., "5m", "15m", "1h", "4h")

    Returns:
        Threshold value for the given timeframe.
        Falls back to 15m (baseline) if timeframe not found.

    Example:
        >>> get_tf_threshold("flat_threshold", "1h")
        0.0015
        >>> get_tf_threshold("ssl_touch_tolerance", "15m")
        0.003
    """
    tf_config = TF_THRESHOLDS.get(timeframe, BASE_TF_THRESHOLDS)
    return tf_config.get(threshold_name, BASE_TF_THRESHOLDS.get(threshold_name, 0.002))


def get_tf_thresholds(timeframe: str) -> dict:
    """
    Get all TF-adaptive thresholds for a timeframe.

    Args:
        timeframe: Timeframe string (e.g., "5m", "15m", "1h")

    Returns:
        Dict with all threshold values for the timeframe.
        Falls back to 15m (baseline) if timeframe not found.

    Example:
        >>> get_tf_thresholds("1h")
        {'flat_threshold': 0.0015, 'ssl_touch_tolerance': 0.005, ...}
    """
    return TF_THRESHOLDS.get(timeframe, BASE_TF_THRESHOLDS).copy()


# ==========================================
# ALPHATREND DUAL-LINE CONFIG
# ==========================================
# AlphaTrend indicator now uses dual lines (buyers vs sellers)
# instead of single line for more accurate trend detection.
# See: core/indicators.py::calculate_alphatrend()
#
# NOTE: flat_threshold is now TF-adaptive. Use get_tf_threshold() or
# pass timeframe to calculate_alphatrend() for proper scaling.
ALPHATREND_CONFIG = {
    "coeff": 1.0,              # ATR multiplier
    "ap": 14,                  # ATR/MFI period
    "flat_lookback": 5,        # Yatay hareket kontrolu icin bakilacak mum sayisi
    "flat_threshold": 0.002,   # DEFAULT: 15m baseline (use get_tf_threshold for TF-adaptive)
    # Note: at_is_flat = True when change < threshold, so HIGHER threshold = MORE signals
}

# ==========================================
# OTHER SETTINGS
# ==========================================
REFRESH_RATE = 3
ENABLE_CHARTS = False
AUTO_RESTART_DELAY_SECONDS = 5

# Best config cache (thread-safe)
BEST_CONFIG_CACHE: Dict[Tuple[str, str], Any] = {}
_BEST_CONFIG_LOCK = threading.RLock()  # RLock for recursive access

BEST_CONFIG_WARNING_FLAGS = {
    "missing_signature": False,
    "signature_mismatch": False,
    "json_error": False,  # Bozuk JSON dosyası hatası için flag
    "load_error": False,  # Genel yükleme hatası için flag
}


def get_best_config(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
    """Thread-safe getter for best config cache.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe

    Returns:
        Config dict or None if not found
    """
    with _BEST_CONFIG_LOCK:
        return BEST_CONFIG_CACHE.get((symbol, timeframe))


def set_best_config(symbol: str, timeframe: str, config: Dict[str, Any]):
    """Thread-safe setter for best config cache.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        config: Configuration dict to store
    """
    with _BEST_CONFIG_LOCK:
        BEST_CONFIG_CACHE[(symbol, timeframe)] = config


def clear_best_config_cache():
    """Thread-safe clear of best config cache."""
    with _BEST_CONFIG_LOCK:
        BEST_CONFIG_CACHE.clear()

# ==========================================
# DEFAULT STRATEGY CONFIGURATION
# ==========================================
# Single source of truth for strategy parameters
# Used by both main file and config_loader for signature generation
#
# Active strategy:
# - "ssl_flow": Trend following with SSL HYBRID baseline (AlphaTrend confirmation, TP at PBEMA)
DEFAULT_STRATEGY_CONFIG = {
    "rr": 2.0,
    "rsi": 70,
    "at_active": True,  # AlphaTrend is essential for SSL Flow
    "use_trailing": False,
    "use_partial": True,  # Partial TP aktif
    "use_dynamic_pbema_tp": True,
    "strategy_mode": "ssl_flow",

    # === SSL Flow Strategy Parameters ===
    "ssl_touch_tolerance": 0.003,    # 0.3% tolerance for SSL baseline touch detection
    "ssl_body_tolerance": 0.003,     # 0.3% tolerance for candle body position
    "min_pbema_distance": 0.004,     # Baseline: Min distance to PBEMA target
    "lookback_candles": 5,           # Baseline: SSL touch lookback
    "skip_body_position": False,     # Baseline: Check body position
    "skip_adx_filter": False,        # Baseline: Apply ADX filter
    "skip_overlap_check": False,     # Baseline: Check SSL-PBEMA overlap
    "skip_at_flat_filter": False,    # Baseline: Apply AT flat filter
    "skip_wick_rejection": True,     # FIX: Skip wick rejection filter (proven +$30 in P3 test)

    # === SSL Flip Grace Period ===
    # When SSL baseline is crossed (flip), allow signals for N bars even if AT hasn't confirmed yet
    # This addresses AT lag (2-5 bars behind SSL) that causes missed entries at trend start
    # Condition: SSL just flipped + AT is NOT opposing (not sellers_dominant for long)
    "use_ssl_flip_grace": False,     # Disabled by default - enable in Optuna for testing
    "ssl_flip_grace_bars": 3,        # Number of bars after SSL flip to allow grace

    # === THREE-TIER AT ARCHITECTURE ===
    # Mode controls how AlphaTrend is used in signal generation:
    # - "binary": Original per-bar check (AT buyers/sellers dominant required) [RECOMMENDED]
    # - "regime": Tier 1 - AT as regime filter (blocks only if strongly opposing)
    # - "score": Tier 3 - AT contributes to score (+2 aligned, +0.5 neutral, -1 opposing)
    # - "off": Disable AT entirely (useful for comparison)
    "at_mode": "binary",             # Default: binary mode (proven baseline)
    "at_regime_lookback": 20,        # Bars to calculate regime (used in regime/score modes)
    "at_score_weight": 2.0,          # AT contribution weight when at_mode="score"

    # === REGIME FILTER (Based on AT Scenario Analysis 2026-01-03) ===
    # Controls regime-based trade filtering. Key finding: Neutral regime has 19.7% win rate!
    # - "off": No regime filtering (default, backward compatible)
    # - "skip_neutral": Skip trades in neutral regime (RECOMMENDED - adds +$17 value)
    # - "aligned": Only trade when signal matches regime (LONG in bullish, SHORT in bearish)
    # - "veto": Block only opposing signals (LONG in bearish, SHORT in bullish)
    "regime_filter": "skip_neutral",  # NEW: Skip neutral regime (proven +$17 improvement)

    "regime_adx_threshold": 20.0,    # Baseline: Standard regime threshold
    "use_ssl_never_lost_filter": True,   # P1: Enabled - prevents counter-trend trades (+$6.44 in A/B test)
    "ssl_never_lost_lookback": 20,       # P1: Lookback period for SSL never lost check

    # === Confirmation Candle (P2 - Entry Timing) ===
    # Addresses "entry too early" issue from trade annotations
    # When enabled: Signal must form, then NEXT candle must confirm direction before entry
    # TEST RESULT (01 Jan 2025): Too aggressive - filtered 16/26 trades, PnL -$28 worse
    # Conclusion: Disabled - the filter blocks profitable entries along with bad ones
    "use_confirmation_candle": False,    # P2: DISABLED after A/B test showed -$28 impact
    "confirmation_candle_mode": "close", # "close" = next candle closes in signal direction
                                         # "body" = next candle body in signal direction

    # === Scoring System (Alternative to AND Logic) ===
    # use_scoring=False (default): Binary AND logic - all filters must pass (strict, fewer trades)
    # use_scoring=True: Weighted scoring - filters contribute points, signal if score >= threshold
    "use_scoring": False,            # Default: AND logic (backward compatible)
    "score_threshold": 6.0,          # Minimum score out of 10.0 for signal acceptance

    # === Hour Filter (v1.7.4 - Execution Timing Optimization) ===
    # Skip trading during historically losing hours (UTC)
    # Based on quant analyst analysis of trade PnL by hour
    # TEST RESULT: Hour filter FAILED - blocked profitable trades, PnL dropped $29.39
    "avoid_hours": None,  # DISABLED - did not improve PnL
    # Score breakdown (max 10.0):
    #   ADX strength: 2.0
    #   Regime trending: 1.0
    #   Baseline touch: 2.0
    #   AlphaTrend confirmation: 2.0
    #   PBEMA distance: 1.0
    #   Wick rejection: 1.0
    #   Body position: 0.5
    #   No overlap: 0.5

    # === MARKET STRUCTURE + FVG BONUS (v2.2.0) ===
    # Market Structure: Required filter for trend alignment (HH/HL = bullish, LH/LL = bearish)
    # FVG (Fair Value Gap): Bonus for entry timing - tighter SL when price returns to FVG
    #
    # TEST RESULTS (02 Jan 2026 - Full Year BTC+ETH+LINK):
    #   WITHOUT MS+FVG: -$4.56, 1521 trades
    #   WITH MS+FVG BONUS: +$4.83, 2062 trades (BEST)
    #   Improvement: +$9.39 PnL, +541 trades
    "use_market_structure": True,     # Required: Filter trades by MS trend alignment
    "min_ms_score": 1.0,              # Minimum MS score (0=no filter, 1=trend-aligned, 2=strong trend)
    "ms_swing_length": 5,             # Swing detection lookback
    "use_fvg_bonus": True,            # Bonus: Use FVG for tighter SL when available
    "fvg_min_gap_percent": 0.08,      # Minimum FVG size (0.08%)
    "fvg_max_age_bars": 50,           # Maximum age of FVG to consider
    "fvg_mitigation_lookback": 5,     # Bars to check for mitigation

    # === Shared Parameters ===
    "tp_min_dist_ratio": 0.0008,     # Min TP distance ratio
    "tp_max_dist_ratio": 0.050,      # Max TP distance ratio
    "adx_min": 15.0,                 # ADX minimum (trend strength)
    "adx_max": 100.0,                # ADX maximum DISABLED (v1.7 showed filtering hurts)

    # === EXIT PROFILE SYSTEM ===
    # Two profiles: CLIP (higher hit-rate, earlier exits) vs RUNNER (let winners run)
    "exit_profile": "clip",          # Default profile: "clip" or "runner"

    # Profile defaults by rolling mode (used when building baseline config)
    "default_profile_weekly": "clip",
    "default_profile_triday": "runner",
    "default_profile_monthly": "clip",

    # === CLIP Profile Parameters (weekly-friendly: higher hit-rate) ===
    "partial_trigger_clip": 0.45,           # Earlier partial (45%)
    "partial_fraction_clip": 0.50,          # Larger partial (50%)
    "dynamic_tp_only_after_partial_clip": False,  # Dynamic TP can help pre-partial
    "dynamic_tp_clamp_mode_clip": "tighten_only",  # Clamp TP tighter only

    # === RUNNER Profile Parameters (triday-friendly: let winners run) ===
    "partial_trigger_runner": 0.70,         # Later partial (70%)
    "partial_fraction_runner": 0.33,        # Smaller partial (33%)
    "dynamic_tp_only_after_partial_runner": True,  # Dynamic TP only after partial
    "dynamic_tp_clamp_mode_runner": "none",  # No clamping, let winners run

    # === Shared Partial TP Parameters (effective values derived from profile) ===
    "partial_trigger": 0.45,         # Effective trigger (derived from profile)
    "partial_fraction": 0.50,        # Effective fraction (derived from profile)
    "partial_rr_adjustment": True,   # RR-based trigger adjustment

    # RR thresholds (shared across profiles)
    "partial_rr_high_threshold": 1.8,
    "partial_rr_low_threshold": 1.2,

    # RR-based triggers per profile
    "partial_rr_high_trigger_clip": 0.55,    # CLIP: earlier at high RR
    "partial_rr_high_trigger_runner": 0.75,  # RUNNER: later at high RR
    "partial_rr_low_trigger_clip": 0.40,     # CLIP: earliest at low RR
    "partial_rr_low_trigger_runner": 0.55,   # RUNNER: moderate at low RR

    # Backward compat: old single-value fields (used if profile not set)
    "partial_rr_high_trigger": 0.55,
    "partial_rr_low_trigger": 0.40,

    # === Dynamic TP Parameters ===
    "dynamic_tp_only_after_partial": False,  # Effective value (derived from profile)
    "dynamic_tp_clamp_mode": "tighten_only", # Effective clamp mode (derived from profile)
    "dynamic_tp_min_distance": 0.004,        # Min distance safety (0.4%)
    "dynamic_tp_direction_check": True,      # Direction sanity check

    # === SL VALIDATION MODE (PR-1 baseline control) ===
    # "off": No SL distance validation (BASELINE - recommended for PR-1)
    # "reject": Reject trades with SL too tight
    # "widen": Widen SL to minimum distance (causes PnL regression)
    "sl_validation_mode": "off",             # PR-1: Default OFF for baseline

    # === MINIMUM SL DISTANCE (only active when sl_validation_mode != "off") ===
    "majors_symbols": ["BTCUSDT", "ETHUSDT"],
    "min_sl_distance_btc_eth": 0.010,        # 1.0% for majors
    "min_sl_distance_alts": 0.015,           # 1.5% for alts
    "min_sl_distance_action": "widen",       # "reject" or "widen" (legacy, use sl_validation_mode)
    "maintain_rr_on_sl_widen": True,         # Scale TP when SL widened to maintain RR

    # === MOMENTUM TP EXTENSION (v1.2) ===
    # When price reaches high progress with strong momentum, extend TP to let winners run
    "momentum_tp_extension": True,           # Enable momentum-based TP extension
    "momentum_extension_threshold": 0.80,    # Progress level to check momentum (80%)
    "momentum_extension_multiplier": 1.5,    # Extend TP by 50% if momentum strong

    # === PROGRESSIVE PARTIAL TP (v1.3) ===
    # Multi-tranche partial TP system - lock profits at intermediate levels
    "use_progressive_partial": True,         # Enable 3-tranche partial system
    "partial_tranches": [
        {"trigger": 0.40, "fraction": 0.33},  # Tranche 1: 33% at 40% progress
        {"trigger": 0.70, "fraction": 0.50},  # Tranche 2: 50% of remaining at 70%
        # Remaining ~33% rides to full TP or extended TP
    ],
    "progressive_be_after_tranche": 1,       # Move to BE after tranche 1 (0-indexed)

    # === ATR-BASED BREAKEVEN BUFFER (v46.x - BE Fix) ===
    # Dynamic BE buffer based on ATR instead of fixed 0.2%
    # This prevents getting stopped out by normal market noise after moving to breakeven
    # User annotation complaints: "Breakeven too early", "SL moved to breakeven too fast"
    "be_atr_multiplier": 0.5,                # BE buffer = ATR * multiplier (default: 0.5x ATR)
    "be_min_buffer_pct": 0.002,              # Minimum BE buffer (0.2% - fallback)
    "be_max_buffer_pct": 0.01,               # Maximum BE buffer (1.0% - cap to prevent too wide)

    # === SSL NEVER LOST FILTER (v46.x) ===
    # Skip counter-trend trades if SSL baseline was never broken in lookback
    # P1: ENABLED - prevents counter-trend trades (+$6.44 improvement in A/B test)
    "use_ssl_never_lost_filter": True,       # P1: Enabled - prevents counter-trend trades
    "ssl_never_lost_lookback": 20,           # Lookback period to check if baseline was ever crossed

    # === CONFIRMATION CANDLE (P2 - Entry Timing) ===
    # Addresses "entry too early" issue from trade annotations
    # TEST RESULT: Too aggressive - filtered 16/26 trades, PnL -$28 worse
    "use_confirmation_candle": False,        # P2: DISABLED after A/B test
    "confirmation_candle_mode": "close",     # "close" = next candle closes in signal direction

    # === TIME-BASED TRADE INVALIDATION (v46.x) ===
    # Exit stale trades at breakeven if no meaningful movement after X candles
    "use_time_invalidation": False,          # BASELINE: Disabled
    "time_invalidation_candles": 8,          # Max candles before checking invalidation (8 = 2h for 15m)
    "time_invalidation_min_move_pct": 0.003, # Min movement % required (0.3%)
    "time_invalidation_exit_mode": "breakeven",  # "breakeven" or "market" (exit at BE or current price)

    # === CIRCUIT BREAKER (DD control) ===
    "circuit_breaker_max_full_stops": 2,     # Max consecutive full STOPs before disable

    # === SMART RE-ENTRY SYSTEM (v1.9.0 / v46.x update) ===
    # Post-SL recovery system for liquidity grab scenarios
    # NOT a signal filter - it's a trade MANAGEMENT system
    # When SL is hit, if price recovers near entry and AlphaTrend confirms,
    # bypass cooldown and allow re-entry to catch post-sweep moves
    # v46.x: Wider parameters to actually trigger (was 2h/0.3%, now 4h/1.0%)
    "use_smart_reentry": False,              # BASELINE: Disabled
    "reentry_window_hours": 4.0,             # Time window after SL (4 hours - wider than original 2h)
    "reentry_price_threshold_pct": 0.01,     # Price must be within 1.0% of entry (wider than 0.3%)
    "reentry_require_at_confirm": True,      # Require AlphaTrend to confirm same direction

    # === ROC MOMENTUM FILTER (v1.10.0) ===
    # Prevents counter-trend entries by checking Rate of Change
    # When ROC > threshold, don't SHORT (uptrend too strong)
    # When ROC < -threshold, don't LONG (downtrend too strong)
    "use_roc_filter": False,                 # DISABLED: Testing baseline recovery
    "roc_period": 10,                        # Lookback period for ROC calculation
    "roc_threshold": 2.0,                    # ROC threshold (2.0% = balanced filter - TEST)

    # === HTF TREND FILTER (v1.14.0 - Counter-trend Prevention) ===
    # Uses 4H timeframe to determine overall trend direction
    # Only allows trades that align with HTF trend:
    # - HTF UP = only LONG allowed
    # - HTF DOWN = only SHORT allowed
    # - HTF NEUTRAL = both directions allowed
    "use_htf_filter": True,                  # Enable 4H trend filter
    "htf_timeframe": "4h",                   # Higher timeframe for trend detection
    "htf_trend_method": "baseline",          # "baseline" (SSL HMA60) or "ema" (EMA50/200)
    "htf_trend_lookback": 3,                 # Candles to check for consistent trend

    # === VOLATILITY REGIME (Expert Panel - Sinclair's 3-Tier System) ===
    # Adapts trading rules based on volatility regime:
    # - LOW_VOL: Conservative mode (strict AT confirmation, smaller positions)
    # - NORMAL_VOL: Standard mode (baseline parameters)
    # - HIGH_VOL: Aggressive mode (allow AT lag, larger positions)
    "use_volatility_regime": True,           # Enable volatility-based regime adaptation
    "vol_regime_lookback": 50,               # Lookback for ATR percentile calculation
    "vol_low_threshold": 40.0,               # Below this = LOW_VOL (conservative)
    "vol_high_threshold": 75.0,              # Above this = HIGH_VOL (aggressive)

    # === FILTER HIERARCHY (Expert Panel - Clenow's Tier System) ===
    # Controls which filters are required vs optional:
    # - Tier 1 (Core): SSL direction, AT aligned, PBEMA path - ALWAYS required
    # - Tier 2 (Quality): Baseline touch, PBEMA distance, ADX/RSI
    # - Tier 3 (Risk): Wick rejection, body position, overlap
    # Lower tier = more trades, lower quality
    # Higher tier = fewer trades, higher quality
    #
    # FULL YEAR BACKTEST RESULTS (2025):
    # - Tier 2: 149 trades, 31.5% WR, E[R]=0.283, ~$737 profit
    # - Recommended: Tier 2 for optimal trade count vs quality balance
    "filter_tier_level": 2,                  # UPDATED: Tier 2 (Expert Panel recommendation)

    # === VOLATILITY-NORMALIZED PBEMA DISTANCE ===
    # Ensures PBEMA target is within realistic ATR range
    # Relaxed from 4.0 to 20.0 based on backtest analysis
    "vol_norm_max_atr": 20.0,                # Maximum ATR distance (relaxed for trend-following)

    # === RECOMMENDED PORTFOLIO (Expert Panel + Full Year Backtest) ===
    # Based on 2025 full year backtest results:
    # - BTCUSDT-1h: Best performer (48.6% WR, 0.895 E[R])
    # - BTCUSDT-15m: High volume (48 trades, 0.046 E[R])
    # - ETHUSDT-15m: Good volume (56 trades, 0.191 E[R])
    # - 4h timeframes: Insufficient signals, excluded
    # Recommended: symbols=["BTCUSDT", "ETHUSDT"], timeframes=["15m", "1h"]
}

# ==========================================
# PATTERN 1: MOMENTUM EXHAUSTION EXIT CONFIGURATION
# ==========================================
# Dynamic TP based on momentum instead of fixed PBEMA target
#
# Core Concept: "Momentum yavaşlayan dek trade tutup TP oluyoruz"
# Exit when momentum slows down, not at fixed price level
#
# Evidence: Real trade analysis shows 5+ trades use momentum-based exits
#
# Examples from Real Trades:
# - NO 7: "momentum yavaşlayan dek takip edip TP oluyoruz"
# - NO 9: "momentum azalana dek fiyatı takip edip TP aldım"
# - NO 12: "momentum bitene dek trade devam ediyor"
# - NO 15: "momentum yavaşlayan dek trade tutup tp oluyorum"
# - NO 18: "momentum yavaşlayan dek trade tuttum"

MOMENTUM_EXIT_CONFIG = {
    # === Core Parameters ===
    "enabled": False,                     # Enable momentum-based exits (default: disabled)
    "lookback": 3,                        # Candles to analyze for momentum
    "min_conditions": 2,                  # Conditions required (2 of 3)

    # === Detection Thresholds ===
    "slope_threshold": 0.5,               # AlphaTrend slope threshold (0-1)
    "range_threshold": 0.7,               # Candle range threshold (0-1)
    "atr_threshold": 0.5,                 # ATR movement threshold (0-1)

    # === Dynamic TP Extension ===
    "min_profit_pct": 0.005,              # Min profit before extending (0.5%)
    "max_extension": 3.0,                 # Max TP extension multiplier
    "extension_multiplier": 1.2,          # TP extension when momentum strong

    # === Trade Management ===
    "lock_profit_on_exhaustion": True,    # Exit at current price when exhausted
    "use_lenient_mode": False,            # Require only 1 condition (more exits)
}

# ==========================================
# PBEMA RETEST STRATEGY CONFIGURATION
# ==========================================
# Trade PBEMA cloud as support/resistance after breakout
#
# Strategy: After price breaks through PBEMA, enter on retest
# Evidence: Real trade analysis shows 33% of profitable setups use this pattern
#
# Examples from Real Trades:
# - NO 7: "PBEMADAN güçlü fiyat sekmesi yaşanabileceği icin long entry"
# - NO 11: "Fiyat PBEMA bandını kazanıyor ve retest ediyor, entry aldım"
# - NO 12-13: "PBEMA retestinde entry aldım"
# - NO 18: "Fiyat PBEMA üzerinde yer edinmiş, bir çok kez retest edip entry aldım"

PBEMA_RETEST_CONFIG = {
    # === Core Parameters ===
    "min_rr": 1.5,                        # Minimum risk/reward ratio
    "breakout_lookback": 20,              # Candles to search for PBEMA breakout
    "min_breakout_strength": 0.005,       # Min distance beyond PBEMA (0.5%)
    "retest_tolerance": 0.003,            # Touch tolerance for retest (0.3%)
    "min_wick_ratio": 0.15,               # Minimum wick for rejection (15%)

    # === TP/SL Configuration ===
    "tp_target": "baseline",              # "baseline" (SSL) or "percentage"
    "tp_percentage": 0.015,               # TP % if using percentage mode (1.5%)
    "sl_buffer": 0.003,                   # SL buffer beyond PBEMA (0.3%)

    # === Optional Filters ===
    "require_at_confirmation": False,     # Require AlphaTrend confirmation
    "require_multiple_retests": False,    # Require 2+ retests (stronger setup)
    "min_retests": 2,                     # Minimum retest count if required
    "use_volume_confirmation": False,     # Volume spike on breakout

    # === Strategy Mode ===
    "strategy_mode": "pbema_retest",      # Strategy identifier
}

# ==========================================
# BASELINE CONFIG (PR-1: Known-good parameters)
# ==========================================
# When sl_validation_mode="off", these baseline values are used.
# This bypasses exit_profile system and uses simple, proven parameters.
BASELINE_CONFIG = {
    # Partial TP - simple baseline
    "partial_trigger": 0.40,                 # 40% progress triggers partial
    "partial_fraction": 0.50,                # Take 50% at partial
    "partial_rr_adjustment": False,          # NO RR-based trigger adjustment

    # Dynamic TP - free movement (no clamp)
    "dynamic_tp_only_after_partial": False,  # Dynamic TP active pre-partial too
    "dynamic_tp_clamp_mode": "none",         # NO tighten_only clamp
    "dynamic_tp_min_distance": 0.004,        # Keep min distance safety

    # SL validation - OFF (no widening)
    "sl_validation_mode": "off",             # No SL distance checks
    "sl_widened": False,                     # Never widened in baseline

    # Exit profile - IGNORED in baseline mode
    "exit_profile_active": False,            # Profile system bypassed
}

# ==========================================
# PR-2 CONFIG: Risk Management & Config Continuity
# ==========================================
# These settings apply to WEEKLY rolling walk-forward only.
# They do NOT modify PR-1 baseline entry/exit logic.

PR2_CONFIG = {
    # === WEEKLY PORTFOLIO LOSS LIMIT (window breaker) ===
    "weekly_max_loss_usd": 50.0,             # Max loss per weekly window (restored to baseline)
    "weekly_stop_trading_on_loss": True,     # Block new entries when limit hit

    # === STREAM FULL-STOP STREAK BREAKER ===
    "stream_fullstop_limit": 2,              # Disable stream after N full stops
    "stream_fullstop_cooldown_windows": 1,   # Cooldown windows before re-enabling (v1.7)
    # Full stop = STOP where partial_taken == False

    # === CARRY-FORWARD CONFIGS FOR EMPTY WEEKS ===
    "carry_forward_enabled": True,           # Use last good config when optimizer empty
    "carry_forward_max_age_windows": 4,      # Max windows to carry forward
    "carry_forward_risk_multiplier": 0.75,   # Reduce risk by 25% when carry-forward

    # Criteria for "good" config (all must be met)
    "min_trades_for_good": 5,                # Min trades to qualify as good
    "min_pnl_for_good": 10.0,                # Min PnL ($) to qualify as good
    "min_win_rate_for_good": 0.55,           # Min win rate to qualify as good

    # === BASELINE FALLBACK ===
    "baseline_fallback_enabled": False,      # DISABLED: Use baseline config for disabled_no_history streams
    "baseline_fallback_risk_multiplier": 0.50,  # Risk multiplier for baseline fallback (50%)

}

# ==========================================
# SYMBOL-SPECIFIC PARAMETERS
# ==========================================
# Default params per symbol/timeframe - can be overridden by optimizer results
# All symbols now use ssl_flow strategy by default with at_active=True

# Base configuration for all symbol/timeframe combinations
_BASE_SYMBOL_TF_CONFIG = {
    "rr": 2.0,
    "rsi": 70,
    "at_active": True,
    "strategy_mode": "ssl_flow",
}

# Generate SYMBOL_PARAMS programmatically (reduces 100+ lines to ~10)
SYMBOL_PARAMS = {
    symbol: {tf: _BASE_SYMBOL_TF_CONFIG.copy() for tf in TIMEFRAMES}
    for symbol in SYMBOLS
}
