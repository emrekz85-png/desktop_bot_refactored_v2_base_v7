# core/optuna_optimizer.py
# Bayesian Optimization using Optuna for SSL Flow Strategy
# Replaces grid search with efficient hyperparameter search
#
# Features:
# - Optimizes 20+ parameters simultaneously
# - Bayesian optimization (learns from previous trials)
# - Multi-objective: PnL + Sharpe + Drawdown
# - Walk-forward validation built-in
# - CPU optimized for MacBook Air M-series
#
# Created: 2025-01-02

import os
import sys
import math
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd

# Optuna import with fallback
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARNING] Optuna not installed. Run: pip install optuna")

from core.config import (
    TRADING_CONFIG, WALK_FORWARD_CONFIG,
    MIN_EXPECTANCY_R_MULTIPLE, DEFAULT_STRATEGY_CONFIG,
    BASELINE_CONFIG,
)
from core.trade_manager import SimTradeManager
from core.utils import tf_to_timedelta
from core.trading_engine import TradingEngine

# Market Structure import with fallback
try:
    from core.market_structure import get_structure_score, TrendType
    HAS_MARKET_STRUCTURE = True
except ImportError:
    HAS_MARKET_STRUCTURE = False
    get_structure_score = None
    TrendType = None

# FVG Detector import with fallback
try:
    from core.fvg_detector import FVGDetector, FVGType
    HAS_FVG_DETECTOR = True
except ImportError:
    HAS_FVG_DETECTOR = False
    FVGDetector = None
    FVGType = None

# NOTE: check_signal_fast() below implements optimized signal check for backtesting.
# It respects at_mode (binary/score/off) and matches ssl_flow.py logic.
# For live trading, use strategies.check_signal() which goes through router.py


# ==========================================
# CONSTANTS
# ==========================================

# Seed for reproducibility
RANDOM_SEED = 42

# Default optimization settings
# INCREASED from 150 to 500 - better coverage of simplified search space
# With 8 params instead of 25+, 500 trials provides ~25% coverage vs 0.3% before
DEFAULT_N_TRIALS = 500  # Number of optimization trials
DEFAULT_N_JOBS = -1     # -1 = use all cores

# MacBook Air M-series optimized settings
# M1/M2/M3 Air has 8 cores (4 performance + 4 efficiency)
# Using 6 workers leaves 2 cores for system + I/O
MACBOOK_AIR_WORKERS = 6


# ==========================================
# PARAMETER SPACE DEFINITION
# ==========================================

@dataclass
class ParameterSpace:
    """Defines all optimizable parameters for SSL Flow strategy."""

    # Core Parameters
    rr_min: float = 1.0
    rr_max: float = 3.0
    rsi_min: int = 30
    rsi_max: int = 80

    # Filter Thresholds
    regime_adx_min: float = 15.0
    regime_adx_max: float = 30.0
    min_pbema_distance_min: float = 0.001
    min_pbema_distance_max: float = 0.01
    lookback_candles_min: int = 3
    lookback_candles_max: int = 10
    ssl_never_lost_lookback_min: int = 10
    ssl_never_lost_lookback_max: int = 30

    # Market Structure Parameters
    min_ms_score_min: float = 0.0
    min_ms_score_max: float = 2.0
    ms_swing_length_min: int = 3
    ms_swing_length_max: int = 10

    # FVG Parameters
    fvg_min_gap_percent_min: float = 0.05
    fvg_min_gap_percent_max: float = 0.15
    fvg_max_age_bars_min: int = 20
    fvg_max_age_bars_max: int = 100
    fvg_mitigation_lookback_min: int = 3
    fvg_mitigation_lookback_max: int = 10

    # Exit Profile Parameters
    partial_trigger_min: float = 0.30
    partial_trigger_max: float = 0.60
    partial_fraction_min: float = 0.30
    partial_fraction_max: float = 0.70
    momentum_extension_threshold_min: float = 0.70
    momentum_extension_threshold_max: float = 0.90
    momentum_extension_multiplier_min: float = 1.2
    momentum_extension_multiplier_max: float = 2.0


# Default parameter space
DEFAULT_PARAM_SPACE = ParameterSpace()


# ==========================================
# MINIMAL BASE CONFIG
# ==========================================

# MINIMAL config: SSL + PBEMA + AlphaTrend only (100+ trades baseline)
# All extra filters are OFF by default - Optuna decides which to ADD
MINIMAL_BASE_CONFIG = {
    # Core (always present)
    "strategy_mode": "ssl_flow",
    "slope": 0.5,

    # SKIP all restrictive filters (True = skip/disable the check)
    "skip_body_position": True,
    "skip_adx_filter": True,
    "skip_overlap_check": True,
    "skip_at_flat_filter": True,
    "skip_wick_rejection": True,

    # DISABLE all optional filters (False = don't use)
    "use_ssl_never_lost_filter": False,
    "use_confirmation_candle": False,
    "use_market_structure": False,
    "use_fvg_bonus": False,
    "use_htf_filter": False,
    "use_roc_filter": False,
    "use_scoring": False,
    "use_smart_reentry": False,
    "use_time_invalidation": False,

    # Basic exit management
    "use_partial": True,
    "use_progressive_partial": False,
    "use_trailing": False,
    "use_dynamic_pbema_tp": False,

    # Sensible defaults for thresholds (when filters are enabled)
    "lookback_candles": 5,
    "min_pbema_distance": 0.004,
    "regime_adx_threshold": 20.0,
    "ssl_never_lost_lookback": 20,
    "min_ms_score": 1.0,
    "ms_swing_length": 5,
    "fvg_min_gap_percent": 0.08,
    "fvg_max_age_bars": 50,
    "fvg_mitigation_lookback": 5,

    # Exit profile defaults
    "partial_trigger": 0.4,
    "partial_fraction": 0.5,
    "partial_rr_adjustment": False,
    "dynamic_tp_only_after_partial": False,
    "dynamic_tp_clamp_mode": "none",
    "sl_validation_mode": "off",
    "momentum_tp_extension": False,
    "momentum_extension_threshold": 0.8,
    "momentum_extension_multiplier": 1.5,
}


# ==========================================
# OBJECTIVE FUNCTION
# ==========================================

def create_config_from_trial(trial: 'optuna.Trial', param_space: ParameterSpace = None) -> Dict:
    """Create a config dict from an Optuna trial.

    SIMPLIFIED APPROACH (v2.0 - Post-Analysis):
    Based on CHANGELOG failed experiments, we now FIX parameters that have been
    proven ineffective and only OPTIMIZE the ~8 parameters that actually matter.

    FIXED PARAMETERS (from CHANGELOG evidence):
    - skip_body_position=True: 99.9% pass rate = useless filter
    - skip_wick_rejection=True: CHANGELOG confirmed +$30 improvement when removed
    - at_mode="binary": Proven best, regime mode failed
    - use_ssl_flip_grace=False: Didn't help
    - use_market_structure=False: Adds complexity without benefit
    - use_fvg_bonus=False: Same
    - use_confirmation_candle=False: Reduces opportunities

    SEARCH SPACE REDUCTION: 25+ params -> 8 params (68% reduction)
    EXPECTED IMPROVEMENT: Better coverage with same trial count
    """
    if param_space is None:
        param_space = DEFAULT_PARAM_SPACE

    # Start from MINIMAL base config
    config = MINIMAL_BASE_CONFIG.copy()

    # =========================================================================
    # TIER 1: FIXED PARAMETERS (Proven by CHANGELOG experiments - DO NOT OPTIMIZE)
    # =========================================================================

    # AlphaTrend: ALWAYS ON with binary mode (proven best)
    config["at_active"] = True
    config["at_mode"] = "binary"  # FIXED: regime mode failed, score mode no better

    # Filters to SKIP (proven ineffective or harmful)
    config["skip_body_position"] = True   # FIXED: 99.9% pass rate = useless
    config["skip_wick_rejection"] = True  # FIXED: +$30 improvement when removed
    config["skip_adx_filter"] = True      # FIXED: Regime gating handles this

    # Features to DISABLE (add complexity without benefit)
    config["use_ssl_flip_grace"] = False      # FIXED: Didn't help
    config["use_market_structure"] = False    # FIXED: No OOS benefit
    config["use_fvg_bonus"] = False           # FIXED: No OOS benefit
    config["use_confirmation_candle"] = False # FIXED: Reduces opportunities
    config["use_btc_regime_filter"] = False   # FIXED: Blocks profitable trades
    config["use_scoring"] = False             # FIXED: AND logic works better

    # =========================================================================
    # TIER 2: OPTIMIZABLE PARAMETERS (~8 params - the ones that actually matter)
    # =========================================================================

    # --- CORE TRADE PARAMETERS ---
    config["rr"] = round(trial.suggest_float("rr", 1.2, 2.5, step=0.1), 2)
    config["rsi"] = trial.suggest_int("rsi", 50, 75, step=5)  # Narrowed from 30-80

    # --- REGIME DETECTION ---
    config["regime_adx_threshold"] = trial.suggest_float(
        "regime_adx_threshold", 17.0, 25.0, step=1.0  # Narrowed from 15-30
    )

    # --- SIGNAL TIMING ---
    config["lookback_candles"] = trial.suggest_int(
        "lookback_candles", 4, 7  # Narrowed from 3-10
    )

    # --- DISTANCE THRESHOLDS ---
    # Overlap check - binary choice
    use_overlap = trial.suggest_categorical("use_overlap_check", [True, False])
    if use_overlap:
        config["skip_overlap_check"] = False
        config["min_pbema_distance"] = trial.suggest_float(
            "min_pbema_distance", 0.003, 0.006, step=0.001  # Narrowed
        )

    # AT Flat filter - binary choice
    use_at_flat = trial.suggest_categorical("use_at_flat_filter", [True, False])
    if use_at_flat:
        config["skip_at_flat_filter"] = False

    # --- OPTIONAL FILTER ---
    # SSL Never Lost - only filter still worth testing
    config["use_ssl_never_lost_filter"] = trial.suggest_categorical("use_ssl_never_lost", [True, False])
    if config["use_ssl_never_lost_filter"]:
        config["ssl_never_lost_lookback"] = trial.suggest_int(
            "ssl_never_lost_lookback", 15, 25, step=5  # Narrowed
        )

    # =========================================================================
    # TIER 3: EXIT MANAGEMENT (Simplified)
    # =========================================================================

    # Basic exit options
    config["use_trailing"] = trial.suggest_categorical("use_trailing", [True, False])
    config["use_dynamic_pbema_tp"] = trial.suggest_categorical("use_dynamic_pbema_tp", [True, False])

    # Partial take profit settings
    config["partial_trigger"] = trial.suggest_float(
        "partial_trigger", 0.30, 0.60, step=0.05
    )
    config["partial_fraction"] = trial.suggest_float(
        "partial_fraction", 0.30, 0.70, step=0.10  # Coarser step
    )

    # Progressive partial - simplified
    config["use_progressive_partial"] = trial.suggest_categorical("use_progressive_partial", [True, False])
    if config["use_progressive_partial"]:
        config["partial_tranches"] = [
            {"trigger": config["partial_trigger"], "fraction": 0.33},
            {"trigger": min(config["partial_trigger"] + 0.30, 0.90), "fraction": 0.50},
        ]
        config["progressive_be_after_tranche"] = 1

    return config


def check_signal_fast(
    df: pd.DataFrame,
    config: Dict,
    index: int,
    # Pre-resolved values (passed from evaluate_config)
    ssl_touch_tolerance: float,
    min_pbema_distance: float,
    lookback_candles: int,
) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], str]:
    """
    PERFORMANCE-OPTIMIZED signal check for Optuna backtest.

    Key optimizations:
    1. NO debug_info dict creation (saves ~50% overhead)
    2. Short-circuits disabled features early
    3. Pre-resolved thresholds (no per-call imports)
    4. Respects at_mode parameter (binary/score/off)

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CRITICAL SYNC WARNING (See docs/AT_VALIDATION_CHANGES.md #12)  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  This function is a PERFORMANCE COPY of ssl_flow.py logic.      ║
    ║                                                                  ║
    ║  If you change strategies/ssl_flow.py, you MUST:                ║
    ║  1. Update this function to match                               ║
    ║  2. Run comparison test to verify 0 mismatches                  ║
    ║                                                                  ║
    ║  Verified: 2026-01-02 - 100% match (200/200 candles)           ║
    ╚══════════════════════════════════════════════════════════════════╝

    Alternative: Use check_signal_real() for guaranteed correctness
    but ~2x slower performance.
    """
    if df is None or df.empty or len(df) < 60:
        return None, None, None, None, "No Data"

    try:
        curr = df.iloc[index]
    except (IndexError, KeyError):
        return None, None, None, None, "Index Error"

    abs_index = index if index >= 0 else (len(df) + index)
    if abs_index < 60:
        return None, None, None, None, "Not Enough Data"

    # Extract values (no debug_info dict creation)
    close = float(curr["close"])
    open_ = float(curr["open"])
    high = float(curr["high"])
    low = float(curr["low"])
    baseline = float(curr.get("baseline", close))
    pb_top = float(curr.get("pb_ema_top", close))
    pb_bot = float(curr.get("pb_ema_bot", close))
    adx_val = float(curr.get("adx", 0))
    rsi_val = float(curr.get("rsi", 50))

    if any(pd.isna([close, baseline, pb_top, pb_bot])):
        return None, None, None, None, "NaN Values"

    min_rr = config.get("rr", 1.5)
    rsi_limit = config.get("rsi", 70)
    at_mode = config.get("at_mode", "binary")

    # Config flags for optional filters
    skip_body_position = config.get("skip_body_position", True)
    skip_overlap_check = config.get("skip_overlap_check", True)
    skip_wick_rejection = config.get("skip_wick_rejection", True)
    skip_at_flat_filter = config.get("skip_at_flat_filter", True)

    # === DIRECTION ===
    price_above_baseline = close > baseline
    price_below_baseline = close < baseline

    if not (price_above_baseline or price_below_baseline):
        return None, None, None, None, "No Direction"

    # === REGIME GATING (fast path - most common rejection) ===
    regime_adx_threshold = config.get("regime_adx_threshold", 20.0)
    regime_lookback = config.get("regime_lookback", 50)
    regime_start = max(0, abs_index - regime_lookback)
    adx_window = df["adx"].iloc[regime_start:abs_index]
    adx_avg = float(adx_window.mean()) if len(adx_window) > 0 else adx_val

    if adx_avg < regime_adx_threshold:
        return None, None, None, None, f"RANGING Regime (ADX_avg={adx_avg:.1f})"

    # === ALPHATREND VALUES ===
    at_buyers_dominant = bool(curr.get("at_buyers_dominant", False))
    at_sellers_dominant = bool(curr.get("at_sellers_dominant", False))
    at_is_flat = bool(curr.get("at_is_flat", False))

    # === SSL FLIP GRACE PERIOD ===
    # When SSL flips, allow N bars grace even if AT hasn't confirmed yet
    ssl_flip_grace_long = False
    ssl_flip_grace_short = False
    use_ssl_flip_grace = config.get("use_ssl_flip_grace", False)

    if use_ssl_flip_grace and abs_index >= 1:
        ssl_flip_grace_bars = config.get("ssl_flip_grace_bars", 3)
        _close_arr = df["close"].values
        _baseline_arr = df["baseline"].values

        for lookback in range(1, min(ssl_flip_grace_bars + 1, abs_index + 1)):
            prev_close = float(_close_arr[abs_index - lookback])
            prev_baseline = float(_baseline_arr[abs_index - lookback])

            # Bullish flip: was below, now above
            if prev_close < prev_baseline and close > baseline:
                if not at_sellers_dominant:  # Not opposing
                    ssl_flip_grace_long = True
                break

            # Bearish flip: was above, now below
            if prev_close > prev_baseline and close < baseline:
                if not at_buyers_dominant:  # Not opposing
                    ssl_flip_grace_short = True
                break

    # === THREE-TIER AT ARCHITECTURE ===
    # Determine if AT allows long/short based on mode
    if at_mode == "binary":
        at_allows_long = at_buyers_dominant or ssl_flip_grace_long
        at_allows_short = at_sellers_dominant or ssl_flip_grace_short

        # Flat filter only in binary mode
        if not skip_at_flat_filter and at_is_flat:
            return None, None, None, None, "AT Flat (No Flow)"

    elif at_mode == "score":
        # Score mode: AT doesn't hard-reject, contributes to scoring
        at_allows_long = True
        at_allows_short = True

    else:  # "off"
        at_allows_long = True
        at_allows_short = True

    # === BASELINE TOUCH DETECTION ===
    lookback_start = max(0, abs_index - lookback_candles)

    baseline_touch_long = False
    baseline_touch_short = False

    if price_above_baseline:
        lookback_lows = df["low"].iloc[lookback_start:abs_index + 1].values
        lookback_baselines = df["baseline"].iloc[lookback_start:abs_index + 1].values
        baseline_touch_long = np.any(lookback_lows <= lookback_baselines * (1 + ssl_touch_tolerance))

    if price_below_baseline:
        lookback_highs = df["high"].iloc[lookback_start:abs_index + 1].values
        lookback_baselines = df["baseline"].iloc[lookback_start:abs_index + 1].values
        baseline_touch_short = np.any(lookback_highs >= lookback_baselines * (1 - ssl_touch_tolerance))

    # === BODY POSITION CHECK ===
    ssl_body_tolerance = config.get("ssl_body_tolerance", 0.003)
    candle_body_min = min(open_, close)
    candle_body_max = max(open_, close)
    body_above_baseline = candle_body_min > baseline * (1 - ssl_body_tolerance)
    body_below_baseline = candle_body_max < baseline * (1 + ssl_body_tolerance)

    # === WICK REJECTION CHECK ===
    candle_range = high - low
    if candle_range > 0:
        lower_wick_ratio = (min(open_, close) - low) / candle_range
        upper_wick_ratio = (high - max(open_, close)) / candle_range
    else:
        lower_wick_ratio = 0.0
        upper_wick_ratio = 0.0

    min_wick_ratio = 0.10
    long_rejection = lower_wick_ratio >= min_wick_ratio
    short_rejection = upper_wick_ratio >= min_wick_ratio

    # === PBEMA PATH AND OVERLAP CHECK ===
    pbema_mid = (pb_top + pb_bot) / 2
    pbema_above_baseline = pbema_mid > baseline
    pbema_below_baseline = pbema_mid < baseline

    # PBEMA distances
    long_pbema_distance = (pb_bot - close) / close if close > 0 else 0
    short_pbema_distance = (close - pb_top) / close if close > 0 else 0

    # Overlap check (TF-adaptive threshold ~0.5% for 15m)
    overlap_threshold = config.get("overlap_threshold", 0.005)
    baseline_pbema_distance = abs(baseline - pbema_mid) / pbema_mid if pbema_mid > 0 else 0
    is_overlapping = baseline_pbema_distance < overlap_threshold

    if not skip_overlap_check and is_overlapping:
        return None, None, None, None, "SSL-PBEMA Overlap (No Flow)"

    # === SSL NEVER LOST FILTER (if enabled) ===
    baseline_ever_lost_bullish = True  # Default: allow
    baseline_ever_lost_bearish = True

    if config.get("use_ssl_never_lost_filter", False):
        ssl_never_lost_lookback = config.get("ssl_never_lost_lookback", 20)
        never_lost_start = max(0, abs_index - ssl_never_lost_lookback)

        lookback_lows_nl = df["low"].iloc[never_lost_start:abs_index].values
        lookback_highs_nl = df["high"].iloc[never_lost_start:abs_index].values
        lookback_baselines_nl = df["baseline"].iloc[never_lost_start:abs_index].values

        if len(lookback_lows_nl) > 0:
            baseline_ever_lost_bullish = np.any(lookback_highs_nl > lookback_baselines_nl)
            baseline_ever_lost_bearish = np.any(lookback_lows_nl < lookback_baselines_nl)

    # === RSI FILTER ===
    long_rsi_ok = pd.isna(rsi_val) or rsi_val <= rsi_limit
    short_rsi_ok = pd.isna(rsi_val) or rsi_val >= (100 - rsi_limit)

    # === ADX FILTER (if not skipped) ===
    if not config.get("skip_adx_filter", True):
        adx_min = config.get("adx_min", 15.0)
        if adx_val < adx_min:
            return None, None, None, None, f"ADX Too Low ({adx_val:.0f})"

    # === LONG SIGNAL CHECK ===
    # Matches ssl_flow.py line 894-903
    is_long = (
        price_above_baseline and
        at_allows_long and
        baseline_touch_long and
        (skip_body_position or body_above_baseline) and
        long_pbema_distance >= min_pbema_distance and
        (skip_wick_rejection or long_rejection) and
        pbema_above_baseline and
        baseline_ever_lost_bullish and
        long_rsi_ok
    )

    # === SHORT SIGNAL CHECK ===
    # Matches ssl_flow.py line 951-960
    is_short = (
        price_below_baseline and
        at_allows_short and
        baseline_touch_short and
        (skip_body_position or body_below_baseline) and
        short_pbema_distance >= min_pbema_distance and
        (skip_wick_rejection or short_rejection) and
        pbema_below_baseline and
        baseline_ever_lost_bearish and
        short_rsi_ok
    )

    # === COMPUTE ENTRY/TP/SL FOR LONG ===
    if is_long:
        entry = close
        tp = pb_bot

        # SL: below baseline or recent swing low
        swing_low = float(df["low"].iloc[max(0, abs_index - 20):abs_index].min())
        sl = min(swing_low * 0.998, baseline * 0.998)

        if tp <= entry or sl >= entry:
            return None, None, None, None, "Invalid Levels (LONG)"

        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "LONG", entry, tp, sl, f"ACCEPTED(R:{rr:.2f})"

    # === COMPUTE ENTRY/TP/SL FOR SHORT ===
    if is_short:
        entry = close
        tp = pb_top

        # SL: above baseline or recent swing high
        swing_high = float(df["high"].iloc[max(0, abs_index - 20):abs_index].max())
        sl = max(swing_high * 1.002, baseline * 1.002)

        if tp >= entry or sl <= entry:
            return None, None, None, None, "Invalid Levels (SHORT)"

        risk = sl - entry
        reward = entry - tp
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "SHORT", entry, tp, sl, f"ACCEPTED(R:{rr:.2f})"

    # === REJECTION REASONS ===
    if price_above_baseline:
        if not at_allows_long:
            return None, None, None, None, "AT: No Buyers"
        if not baseline_touch_long:
            return None, None, None, None, "No Baseline Touch (LONG)"
        if not (skip_body_position or body_above_baseline):
            return None, None, None, None, "Body Below Baseline (LONG)"
        if long_pbema_distance < min_pbema_distance:
            return None, None, None, None, f"PBEMA Too Close ({long_pbema_distance:.3f})"
        if not (skip_wick_rejection or long_rejection):
            return None, None, None, None, "No Wick Rejection (LONG)"
        if not pbema_above_baseline:
            return None, None, None, None, "No PBEMA Path (LONG)"
        if not baseline_ever_lost_bullish:
            return None, None, None, None, "SSL Never Lost (LONG blocked)"
        if not long_rsi_ok:
            return None, None, None, None, f"RSI Too High ({rsi_val:.0f})"

    if price_below_baseline:
        if not at_allows_short:
            return None, None, None, None, "AT: No Sellers"
        if not baseline_touch_short:
            return None, None, None, None, "No Baseline Touch (SHORT)"
        if not (skip_body_position or body_below_baseline):
            return None, None, None, None, "Body Above Baseline (SHORT)"
        if short_pbema_distance < min_pbema_distance:
            return None, None, None, None, f"PBEMA Too Close ({short_pbema_distance:.3f})"
        if not (skip_wick_rejection or short_rejection):
            return None, None, None, None, "No Wick Rejection (SHORT)"
        if not pbema_below_baseline:
            return None, None, None, None, "No PBEMA Path (SHORT)"
        if not baseline_ever_lost_bearish:
            return None, None, None, None, "SSL Never Lost (SHORT blocked)"
        if not short_rsi_ok:
            return None, None, None, None, f"RSI Too Low ({rsi_val:.0f})"

    return None, None, None, None, "No Signal"


def evaluate_config(
    df: pd.DataFrame,
    sym: str,
    tf: str,
    config: Dict,
) -> Tuple[float, int, float, float, float]:
    """Evaluate a config and return metrics.

    PERFORMANCE OPTIMIZED:
    - Pre-resolves TF-adaptive thresholds ONCE (not per-candle)
    - Calls ssl_flow directly (bypasses router overhead)
    - All ssl_flow.py changes (like three-tier AT) still work correctly

    Returns:
        (net_pnl, trades, expected_r, win_rate, max_drawdown)
    """
    from core.config import get_tf_thresholds

    tm = SimTradeManager(initial_balance=TRADING_CONFIG["initial_balance"])
    warmup = 250
    end = len(df) - 2

    if end <= warmup:
        return 0.0, 0, 0.0, 0.0, 0.0

    # OPTIMIZATION: Pre-resolve TF-adaptive thresholds ONCE
    tf_thresholds = get_tf_thresholds(tf)

    # Merge config with TF-adaptive defaults (config values override if present)
    ssl_touch_tolerance = config.get("ssl_touch_tolerance") or tf_thresholds.get("ssl_touch_tolerance", 0.003)
    min_pbema_distance = config.get("min_pbema_distance") or tf_thresholds.get("min_pbema_distance", 0.004)
    lookback_candles = config.get("lookback_candles") or int(tf_thresholds.get("lookback_candles", 5))

    # Extract arrays for performance
    timestamps = pd.to_datetime(df["timestamp"]).values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values

    pb_tops = df.get("pb_ema_top", df["close"]).values if "pb_ema_top" in df.columns else closes
    pb_bots = df.get("pb_ema_bot", df["close"]).values if "pb_ema_bot" in df.columns else closes

    # Optional arrays
    at_buyers_arr = df["at_buyers_dominant"].values if "at_buyers_dominant" in df.columns else None
    at_sellers_arr = df["at_sellers_dominant"].values if "at_sellers_dominant" in df.columns else None
    atr_arr = df["atr"].values if "atr" in df.columns else None

    for i in range(warmup, end):
        event_time = pd.Timestamp(timestamps[i]) + tf_to_timedelta(tf)

        candle_data = {}
        if at_buyers_arr is not None and at_sellers_arr is not None:
            candle_data["at_buyers_dominant"] = bool(at_buyers_arr[i])
            candle_data["at_sellers_dominant"] = bool(at_sellers_arr[i])
        if atr_arr is not None:
            candle_data["atr"] = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else None

        tm.update_trades(
            sym, tf,
            candle_high=float(highs[i]),
            candle_low=float(lows[i]),
            candle_close=float(closes[i]),
            candle_time_utc=event_time,
            pb_top=float(pb_tops[i]),
            pb_bot=float(pb_bots[i]),
            candle_data=candle_data,
        )

        # PERFORMANCE-OPTIMIZED: Use check_signal_fast
        # Key improvements over full ssl_flow.py:
        # 1. NO debug_info dict creation
        # 2. Short-circuits disabled features early
        # 3. Pre-resolved thresholds passed in
        # 4. Respects at_mode parameter (fixes original bug)
        s_type, s_entry, s_tp, s_sl, s_reason = check_signal_fast(
            df,
            config=config,
            index=i,
            ssl_touch_tolerance=ssl_touch_tolerance,
            min_pbema_distance=min_pbema_distance,
            lookback_candles=lookback_candles,
        )

        if not s_type:
            continue

        has_open = any(
            t.get("symbol") == sym and t.get("timeframe") == tf
            for t in tm.open_trades
        )
        if has_open or tm.check_cooldown(sym, tf, event_time):
            continue

        entry_open = float(opens[i + 1])
        open_ts = timestamps[i + 1]
        ts_str = (pd.Timestamp(open_ts) + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")

        min_rr = config.get("rr", 1.5)
        if s_type == "LONG":
            actual_risk = entry_open - s_sl
            actual_reward = s_tp - entry_open
        else:
            actual_risk = s_sl - entry_open
            actual_reward = entry_open - s_tp

        if actual_risk <= 0 or actual_reward <= 0:
            continue

        actual_rr = actual_reward / actual_risk
        if actual_rr < min_rr * 0.9:
            continue

        tm.open_trade({
            "symbol": sym,
            "timeframe": tf,
            "type": s_type,
            "setup": s_reason,
            "entry": entry_open,
            "tp": s_tp,
            "sl": s_sl,
            "timestamp": ts_str,
            "open_time_utc": open_ts,
            "use_trailing": config.get("use_trailing", False),
            "use_dynamic_pbema_tp": config.get("use_dynamic_pbema_tp", False),
        })

    # Calculate metrics
    # NOTE: tm.history may have duplicate entries per trade (partial exits)
    # We need to aggregate by trade ID to get correct metrics

    if not tm.history:
        return 0.0, 0, 0.0, 0.0, 0.0

    # Aggregate PnL by trade ID (sum partials for same trade)
    trade_pnl_by_id = {}
    for t in tm.history:
        tid = t.get("id")
        pnl = t.get("pnl", 0.0)
        if tid in trade_pnl_by_id:
            trade_pnl_by_id[tid] += pnl
        else:
            trade_pnl_by_id[tid] = pnl

    trades = len(trade_pnl_by_id)
    if trades == 0:
        return 0.0, 0, 0.0, 0.0, 0.0

    # Use aggregated PnLs (one per trade)
    trade_pnls = list(trade_pnl_by_id.values())

    net_pnl = sum(trade_pnls)  # More accurate than tm.total_pnl
    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = wins / trades

    # Calculate E[R] directly from PnL (most accurate method)
    # trade_r_multiples from trade manager doesn't properly account for partials
    risk_per_trade = 43.75  # 1.75% of $2500
    expected_r = (net_pnl / trades) / risk_per_trade if trades > 0 else 0.0

    # Calculate max drawdown (use aggregated trade PnLs)
    cumulative = 0
    peak = 0
    max_dd = 0
    for pnl in trade_pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    return net_pnl, trades, expected_r, win_rate, max_dd


def count_active_filters(config: Dict) -> int:
    """Count number of active (non-skipped) filters for complexity penalty."""
    count = 0
    # Filters that are active when skip_* is False
    if not config.get("skip_body_position", True):
        count += 1
    if not config.get("skip_wick_rejection", True):
        count += 1
    if not config.get("skip_overlap_check", True):
        count += 1
    if not config.get("skip_at_flat_filter", True):
        count += 1
    if not config.get("skip_adx_filter", True):
        count += 1
    # Filters that are active when use_* is True
    if config.get("use_ssl_never_lost_filter", False):
        count += 1
    if config.get("use_confirmation_candle", False):
        count += 1
    if config.get("use_market_structure", False):
        count += 1
    if config.get("use_fvg_bonus", False):
        count += 1
    return count


def compute_composite_score(
    net_pnl: float,
    trades: int,
    expected_r: float,
    win_rate: float,
    max_dd: float,
    tf: str = "15m",
    min_trades: int = 50,  # INCREASED: Expert Panel (Kaufman) - 50 min for statistical significance
    config: Dict = None,  # NEW: For complexity penalty
) -> float:
    """Compute composite optimization score v2.1 (Expert Panel Update).

    IMPROVEMENTS over v1.0:
    1. Minimum trades INCREASED to 50 (Kaufman: 13 trades = random noise)
    2. Trade count sweet spot: 50-100 (more trades = more confidence)
    3. Complexity penalty for active filters
    4. E[R] contribution capped to prevent gaming
    5. Better win rate handling
    6. Sample size bonus (log scale) - more trades = higher score

    Expert Panel Insight (Perry Kaufman):
    "13 trade ile optimization yapmak random noise fit etmek demek.
     Minimum 50 trade gerekli - 100+ trades ideal."

    Score Components (100 points max):
    - E[R]: 25 points max (capped to prevent gaming)
    - Trade count: 25 points max (optimal at 50-100, log scale bonus)
    - Win rate: 15 points max
    - Drawdown: 15 points max
    - Complexity bonus: 5 points max (fewer filters = better)
    - Reserve: 15 points for OOS bonus (applied in validate_oos)

    Returns -inf for rejected configs.
    """
    # === HARD REJECTS ===
    # Expert Panel: 50 trades minimum for statistical significance
    if trades < min_trades:
        return float("-inf")
    if net_pnl <= 0:
        return float("-inf")

    # TF-specific minimum E[R] (relaxed since we require more trades)
    min_expected_r = {"15m": 0.05, "1h": 0.04, "4h": 0.03}.get(tf, 0.05)
    if expected_r < min_expected_r:
        return float("-inf")

    # === COMPONENT SCORES ===

    # 1. E[R] COMPONENT (25 points max) - CAPPED to prevent gaming
    # An E[R] of 0.25 is excellent, cap at that level
    er_score = min(25, expected_r * 100)

    # 2. TRADE COUNT COMPONENT (25 points max) - SAMPLE SIZE BONUS
    # Expert Panel: More trades = more confidence in results
    # Sweet spot: 50-100 trades (statistical significance + not overtrading)
    # Uses log scale to reward higher trade counts without penalizing moderate counts
    if trades >= 100:
        # Excellent: 100+ trades gives full score
        trade_score = 25.0
    elif trades >= 50:
        # Good: 50-100 trades, linear scale
        trade_score = 15.0 + (trades - 50) * 0.2  # 15 at 50, 25 at 100
    else:
        # Below minimum (should not reach here due to hard reject)
        trade_score = max(0.0, trades * 0.3)

    # 3. WIN RATE COMPONENT (15 points max)
    # SSL Flow targets 70%+ win rate
    if win_rate >= 0.70:
        wr_score = 15.0
    elif win_rate >= 0.60:
        wr_score = 12.0
    elif win_rate >= 0.50:
        wr_score = 9.0
    elif win_rate >= 0.40:
        wr_score = 6.0
    else:
        wr_score = 3.0

    # 4. DRAWDOWN COMPONENT (15 points max)
    # Penalty if drawdown exceeds portion of profit
    if net_pnl > 0:
        dd_ratio = max_dd / net_pnl
        if dd_ratio <= 0.3:
            dd_score = 15.0  # Excellent: DD < 30% of profit
        elif dd_ratio <= 0.5:
            dd_score = 12.0  # Good: DD < 50% of profit
        elif dd_ratio <= 0.7:
            dd_score = 8.0   # Acceptable: DD < 70% of profit
        elif dd_ratio <= 1.0:
            dd_score = 4.0   # Poor: DD = profit
        else:
            dd_score = 0.0   # Bad: DD > profit
    else:
        dd_score = 0.0

    # 5. COMPLEXITY BONUS (5 points max) - NEW!
    # Prefer simpler configs (fewer active filters = less overfitting risk)
    if config is not None:
        active_filters = count_active_filters(config)
        # 0-1 filters: 5 points, 2-3 filters: 3 points, 4+ filters: 1 point
        if active_filters <= 1:
            complexity_score = 5.0
        elif active_filters <= 3:
            complexity_score = 3.0
        else:
            complexity_score = 1.0
    else:
        complexity_score = 3.0  # Default middle value

    # === FINAL SCORE ===
    score = er_score + trade_score + wr_score + dd_score + complexity_score

    return score


def compute_robust_score_with_oos(
    train_metrics: Dict,
    oos_metrics: Dict,
    config: Dict,
    tf: str = "15m",
) -> Tuple[float, bool, str]:
    """
    Compute final score incorporating OOS performance.

    This is called AFTER validation to determine if config should be enabled.

    Returns:
        (final_score, is_valid, rejection_reason)
    """
    train_pnl = train_metrics.get("net_pnl", 0)
    train_trades = train_metrics.get("trades", 0)
    train_er = train_metrics.get("expected_r", 0)
    train_wr = train_metrics.get("win_rate", 0)
    train_dd = train_metrics.get("max_dd", 0)

    oos_pnl = oos_metrics.get("net_pnl", 0)
    oos_trades = oos_metrics.get("trades", 0)
    oos_er = oos_metrics.get("expected_r", 0)

    # Get base score from training
    base_score = compute_composite_score(
        train_pnl, train_trades, train_er, train_wr, train_dd,
        tf=tf, config=config
    )

    if base_score == float("-inf"):
        return base_score, False, "Train metrics below threshold"

    # === OOS VALIDATION ===

    # Reject if 0 OOS trades
    if oos_trades < 3:
        return float("-inf"), False, f"OOS trades too low ({oos_trades})"

    # Reject if OOS E[R] is negative
    if oos_er < 0:
        return float("-inf"), False, f"OOS E[R] negative ({oos_er:.3f})"

    # Calculate overfit ratio
    if train_er > 0:
        overfit_ratio = oos_er / train_er
    else:
        overfit_ratio = 0.0

    # Reject if severe overfitting (OOS < 40% of train)
    # EXCEPTION: If OOS E[R] is still good (>= 0.08), accept despite low ratio
    # Rationale: A config with Train E[R]=0.80, OOS E[R]=0.12 is still profitable
    # even though ratio is only 0.15. The absolute OOS performance matters.
    MIN_ACCEPTABLE_OOS_ER = 0.08  # Minimum OOS E[R] to override ratio check

    if overfit_ratio < 0.40:
        # Check if absolute OOS performance is acceptable
        if oos_er >= MIN_ACCEPTABLE_OOS_ER:
            # Accept despite low ratio - absolute OOS is good enough
            pass  # Continue to OOS bonus calculation
        else:
            return float("-inf"), False, f"Overfit ratio too low ({overfit_ratio:.2f}) and OOS E[R] insufficient ({oos_er:.3f})"

    # === OOS BONUS (15 points max) ===
    # Reward configs that generalize well
    if overfit_ratio >= 0.80:
        oos_bonus = 15.0  # Excellent: OOS >= 80% of train
    elif overfit_ratio >= 0.60:
        oos_bonus = 10.0  # Good: OOS >= 60% of train
    elif overfit_ratio >= 0.50:
        oos_bonus = 5.0   # Acceptable: OOS >= 50% of train
    else:
        oos_bonus = 0.0   # Marginal: OOS 40-50% of train

    final_score = base_score + oos_bonus

    return final_score, True, f"Valid (overfit_ratio={overfit_ratio:.2f})"


# ==========================================
# OPTUNA STUDY CREATION
# ==========================================

class SSLFlowOptimizer:
    """Optuna-based optimizer for SSL Flow strategy.

    Features:
    - Bayesian optimization (TPE sampler)
    - Walk-forward validation
    - Multi-objective scoring
    - CPU optimized for MacBook Air
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sym: str,
        tf: str,
        n_trials: int = DEFAULT_N_TRIALS,
        n_jobs: int = DEFAULT_N_JOBS,
        train_ratio: float = 0.70,
        param_space: ParameterSpace = None,
        verbose: bool = True,
    ):
        self.df_full = df
        self.sym = sym
        self.tf = tf
        self.n_trials = n_trials
        self.train_ratio = train_ratio
        self.param_space = param_space or DEFAULT_PARAM_SPACE
        self.verbose = verbose

        # Determine optimal worker count
        if n_jobs == -1:
            cpu_count = os.cpu_count() or 8
            # MacBook Air optimization: use 6 out of 8 cores
            if cpu_count <= 8:
                self.n_jobs = min(6, cpu_count - 2)
            else:
                self.n_jobs = cpu_count - 2
        else:
            self.n_jobs = n_jobs

        # Split data for walk-forward
        self._split_data()

        # Results storage
        self.best_config = None
        self.best_score = float("-inf")
        self.study = None
        self.all_trials = []

    def _split_data(self):
        """Split data into train and test sets."""
        n = len(self.df_full)
        train_end = int(n * self.train_ratio)

        if train_end < 300 or (n - train_end) < 100:
            self.df_train = self.df_full
            self.df_test = None
            if self.verbose:
                print(f"[OPTUNA] Walk-forward: Insufficient data, using full dataset")
        else:
            self.df_train = self.df_full.iloc[:train_end].reset_index(drop=True)
            self.df_test = self.df_full.iloc[train_end:].reset_index(drop=True)
            if self.verbose:
                print(f"[OPTUNA] Walk-forward: Train={len(self.df_train)} candles, Test={len(self.df_test)} candles")

    def _objective(self, trial: 'optuna.Trial') -> float:
        """Optuna objective function with robust scoring v2.0."""
        # Create config from trial
        config = create_config_from_trial(trial, self.param_space)

        # Evaluate on training data
        net_pnl, trades, expected_r, win_rate, max_dd = evaluate_config(
            self.df_train, self.sym, self.tf, config
        )

        # Compute score with complexity penalty
        # Expert Panel (Kaufman): min_trades=50 for statistical significance
        # Note: This may reduce trade count but improves reliability
        score = compute_composite_score(
            net_pnl, trades, expected_r, win_rate, max_dd,
            tf=self.tf,
            min_trades=50,  # Expert Panel recommendation (was 15)
            config=config   # For complexity penalty
        )

        # Store metrics as user attributes
        trial.set_user_attr("net_pnl", net_pnl)
        trial.set_user_attr("trades", trades)
        trial.set_user_attr("expected_r", expected_r)
        trial.set_user_attr("win_rate", win_rate)
        trial.set_user_attr("max_dd", max_dd)
        trial.set_user_attr("active_filters", count_active_filters(config))

        return score

    def optimize(self) -> Tuple[Dict, float]:
        """Run the optimization.

        Returns:
            (best_config, best_score)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")

        if self.verbose:
            print(f"\n[OPTUNA] Starting optimization for {self.sym}-{self.tf}")
            print(f"[OPTUNA] Trials: {self.n_trials}, Workers: {self.n_jobs}")
            print(f"[OPTUNA] Optimizing 25+ parameters with Bayesian search")

        # Create study with TPE sampler (Bayesian optimization)
        # Note: warn_independent_sampling=False because we use dynamic search space
        # (conditional parameters like regime_adx_threshold only when use_adx_filter=True)
        sampler = TPESampler(
            seed=RANDOM_SEED,
            multivariate=True,
            warn_independent_sampling=False,
        )
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"ssl_flow_{self.sym}_{self.tf}",
        )

        # Suppress Optuna logging if not verbose
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        # Note: Optuna's n_jobs for study.optimize() is for parallel trial evaluation
        # For CPU-bound tasks, use n_jobs=1 to avoid multiprocessing overhead
        # The parallelism benefit comes from running multiple streams concurrently
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=1,  # Sequential trials (parallelism at stream level)
            show_progress_bar=self.verbose,
        )

        # Get best trial
        best_trial = self.study.best_trial
        self.best_config = create_config_from_trial(best_trial, self.param_space)
        self.best_score = best_trial.value

        # Add trial metrics to config
        self.best_config["_net_pnl"] = best_trial.user_attrs.get("net_pnl", 0)
        self.best_config["_trades"] = best_trial.user_attrs.get("trades", 0)
        self.best_config["_expected_r"] = best_trial.user_attrs.get("expected_r", 0)
        self.best_config["_win_rate"] = best_trial.user_attrs.get("win_rate", 0)
        self.best_config["_max_dd"] = best_trial.user_attrs.get("max_dd", 0)
        self.best_config["_score"] = best_trial.value

        if self.verbose:
            print(f"\n[OPTUNA] Best trial: #{best_trial.number}")
            print(f"[OPTUNA] Score: {best_trial.value:.2f}")
            print(f"[OPTUNA] PnL: ${best_trial.user_attrs.get('net_pnl', 0):.2f}")
            print(f"[OPTUNA] Trades: {best_trial.user_attrs.get('trades', 0)}")
            print(f"[OPTUNA] E[R]: {best_trial.user_attrs.get('expected_r', 0):.3f}")
            print(f"[OPTUNA] Win Rate: {best_trial.user_attrs.get('win_rate', 0):.1%}")

        return self.best_config, self.best_score

    def validate_oos(self) -> Optional[Dict]:
        """Validate best config on out-of-sample data using robust scoring v2.0.

        IMPROVEMENTS:
        - Uses compute_robust_score_with_oos for integrated OOS validation
        - Stricter overfit detection (OOS trades >= 3, E[R] > 0)
        - OOS bonus applied to final score

        Returns:
            OOS metrics dict or None if no test data
        """
        if self.df_test is None or self.best_config is None:
            return None

        # Evaluate on OOS data
        net_pnl, trades, expected_r, win_rate, max_dd = evaluate_config(
            self.df_test, self.sym, self.tf, self.best_config
        )

        oos_result = {
            "oos_pnl": net_pnl,
            "oos_trades": trades,
            "oos_expected_r": expected_r,
            "oos_win_rate": win_rate,
            "oos_max_dd": max_dd,
        }

        # Build train metrics dict
        train_metrics = {
            "net_pnl": self.best_config.get("_net_pnl", 0),
            "trades": self.best_config.get("_trades", 0),
            "expected_r": self.best_config.get("_expected_r", 0),
            "win_rate": self.best_config.get("_win_rate", 0),
            "max_dd": self.best_config.get("_max_dd", 0),
        }

        oos_metrics = {
            "net_pnl": net_pnl,
            "trades": trades,
            "expected_r": expected_r,
        }

        # Use new robust OOS scoring
        final_score, is_valid, reason = compute_robust_score_with_oos(
            train_metrics, oos_metrics, self.best_config, tf=self.tf
        )

        # Calculate overfit ratio for logging
        train_expected_r = self.best_config.get("_expected_r", 0)
        if train_expected_r > 0 and expected_r > 0:
            overfit_ratio = expected_r / train_expected_r
        else:
            overfit_ratio = 0.0

        oos_result["overfit_ratio"] = overfit_ratio
        oos_result["is_overfit"] = not is_valid
        oos_result["rejection_reason"] = reason if not is_valid else None
        oos_result["final_score"] = final_score

        if self.verbose:
            print(f"\n[OPTUNA] OOS Validation (Robust v2.0):")
            print(f"[OPTUNA] OOS PnL: ${net_pnl:.2f}")
            print(f"[OPTUNA] OOS Trades: {trades}")
            print(f"[OPTUNA] OOS E[R]: {expected_r:.3f}")
            print(f"[OPTUNA] Overfit Ratio: {overfit_ratio:.2f}")
            print(f"[OPTUNA] Final Score: {final_score:.2f}")
            if not is_valid:
                print(f"[OPTUNA] ❌ REJECTED: {reason}")
            else:
                print(f"[OPTUNA] ✓ {reason}")

        # Add to config
        self.best_config["_oos_pnl"] = net_pnl
        self.best_config["_oos_trades"] = trades
        self.best_config["_oos_expected_r"] = expected_r
        self.best_config["_oos_win_rate"] = win_rate
        self.best_config["_overfit_ratio"] = overfit_ratio
        self.best_config["_final_score"] = final_score
        self.best_config["_walk_forward_validated"] = is_valid

        return oos_result

    def get_parameter_importances(self) -> Dict[str, float]:
        """Get parameter importance scores from the study.

        Returns dict mapping parameter name to importance score.
        Requires completed study with enough trials.
        """
        if self.study is None or len(self.study.trials) < 10:
            return {}

        try:
            from optuna.importance import get_param_importances
            importances = get_param_importances(self.study)
            return dict(importances)
        except Exception:
            return {}

    def get_top_configs(self, n: int = 10) -> List[Dict]:
        """Get top N configs from all trials.

        Returns list of (config, score, metrics) tuples.
        """
        if self.study is None:
            return []

        # Sort trials by value (score)
        sorted_trials = sorted(
            self.study.trials,
            key=lambda t: t.value if t.value is not None else float("-inf"),
            reverse=True
        )

        top_configs = []
        for trial in sorted_trials[:n]:
            config = create_config_from_trial(trial, self.param_space)
            metrics = {
                "score": trial.value,
                "net_pnl": trial.user_attrs.get("net_pnl", 0),
                "trades": trial.user_attrs.get("trades", 0),
                "expected_r": trial.user_attrs.get("expected_r", 0),
                "win_rate": trial.user_attrs.get("win_rate", 0),
            }
            top_configs.append({"config": config, "metrics": metrics})

        return top_configs


# ==========================================
# K-FOLD TIME SERIES CROSS VALIDATION
# ==========================================

def create_time_series_folds(
    df: pd.DataFrame,
    n_folds: int = 3,
    purge_bars: int = 672,  # 7 days for 15m (96 bars/day * 7)
    min_train_size: int = 5000,
    min_test_size: int = 1000,
) -> List[Dict[str, pd.DataFrame]]:
    """
    Create K time-series folds with purging for walk-forward validation.

    PREVENTS: Overfitting to a single train/test boundary

    Structure (for 3 folds):
        Fold 1: Train [0 : 50%], Purge [50% : 50%+gap], Test [50%+gap : 70%]
        Fold 2: Train [0 : 65%], Purge [65% : 65%+gap], Test [65%+gap : 85%]
        Fold 3: Train [0 : 80%], Purge [80% : 80%+gap], Test [80%+gap : 100%]

    Args:
        df: Full dataset
        n_folds: Number of validation folds
        purge_bars: Number of bars to skip between train/test (prevents leakage)
        min_train_size: Minimum training set size
        min_test_size: Minimum test set size

    Returns:
        List of dicts with 'train' and 'test' DataFrames
    """
    n = len(df)
    folds = []

    # Calculate fold boundaries
    # Each fold uses expanding training window
    for i in range(n_folds):
        # Train end progresses: 50%, 65%, 80% for 3 folds
        train_pct = 0.50 + (i * 0.15)
        train_end = int(n * train_pct)

        # Test window after purge
        test_start = train_end + purge_bars
        test_pct = train_pct + 0.20  # 20% test window
        test_end = min(int(n * test_pct), n)

        # Validate sizes
        if train_end < min_train_size:
            continue
        if (test_end - test_start) < min_test_size:
            continue

        train_df = df.iloc[:train_end].reset_index(drop=True)
        test_df = df.iloc[test_start:test_end].reset_index(drop=True)

        folds.append({
            "fold": i + 1,
            "train": train_df,
            "test": test_df,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_pct": train_pct,
        })

    return folds


def validate_config_kfold(
    df: pd.DataFrame,
    sym: str,
    tf: str,
    config: Dict,
    n_folds: int = 3,
    verbose: bool = False,
) -> Dict:
    """
    Validate a config across multiple time-series folds.

    RETURNS: Aggregated metrics across all folds

    Validation Criteria:
    - All folds must have positive E[R]
    - Average overfit ratio must be >= 0.50
    - Total trades across folds >= 10
    """
    # Determine purge bars based on timeframe
    purge_bars_map = {
        "15m": 672,   # 7 days
        "1h": 168,    # 7 days
        "4h": 42,     # 7 days
    }
    purge_bars = purge_bars_map.get(tf, 672)

    folds = create_time_series_folds(df, n_folds=n_folds, purge_bars=purge_bars)

    if not folds:
        return {
            "is_valid": False,
            "reason": "Insufficient data for K-fold validation",
            "folds_evaluated": 0,
        }

    fold_results = []
    total_oos_trades = 0
    total_oos_pnl = 0.0
    all_positive_er = True

    for fold in folds:
        # Evaluate on training set
        train_pnl, train_trades, train_er, train_wr, train_dd = evaluate_config(
            fold["train"], sym, tf, config
        )

        # Evaluate on test set
        oos_pnl, oos_trades, oos_er, oos_wr, oos_dd = evaluate_config(
            fold["test"], sym, tf, config
        )

        # Calculate overfit ratio for this fold
        if train_er > 0 and oos_er > 0:
            overfit_ratio = oos_er / train_er
        elif train_er > 0 and oos_er <= 0:
            overfit_ratio = 0.0
            all_positive_er = False
        else:
            overfit_ratio = 0.0

        fold_results.append({
            "fold": fold["fold"],
            "train_trades": train_trades,
            "train_er": train_er,
            "oos_trades": oos_trades,
            "oos_er": oos_er,
            "oos_pnl": oos_pnl,
            "overfit_ratio": overfit_ratio,
        })

        total_oos_trades += oos_trades
        total_oos_pnl += oos_pnl

        if oos_er <= 0:
            all_positive_er = False

        if verbose:
            print(f"  Fold {fold['fold']}: Train E[R]={train_er:.3f}, "
                  f"OOS E[R]={oos_er:.3f}, Ratio={overfit_ratio:.2f}")

    # Calculate aggregate metrics
    avg_overfit_ratio = sum(f["overfit_ratio"] for f in fold_results) / len(fold_results)
    avg_oos_er = sum(f["oos_er"] for f in fold_results) / len(fold_results)

    # Validation criteria
    is_valid = (
        all_positive_er and
        avg_overfit_ratio >= 0.50 and
        total_oos_trades >= 10
    )

    reason = "Valid" if is_valid else (
        "Not all folds have positive E[R]" if not all_positive_er else
        f"Avg overfit ratio {avg_overfit_ratio:.2f} < 0.50" if avg_overfit_ratio < 0.50 else
        f"Total OOS trades {total_oos_trades} < 10"
    )

    return {
        "is_valid": is_valid,
        "reason": reason,
        "folds_evaluated": len(fold_results),
        "fold_results": fold_results,
        "total_oos_trades": total_oos_trades,
        "total_oos_pnl": total_oos_pnl,
        "avg_overfit_ratio": avg_overfit_ratio,
        "avg_oos_er": avg_oos_er,
        "all_positive_er": all_positive_er,
    }


# ==========================================
# BATCH OPTIMIZATION (MULTIPLE STREAMS)
# ==========================================

def optimize_multiple_streams(
    streams: Dict[Tuple[str, str], pd.DataFrame],
    n_trials: int = DEFAULT_N_TRIALS,
    n_jobs: int = DEFAULT_N_JOBS,
    train_ratio: float = 0.70,
    verbose: bool = True,
    progress_callback: Callable[[str], None] = None,
) -> Dict[Tuple[str, str], Dict]:
    """Optimize multiple symbol-timeframe streams in parallel.

    Args:
        streams: Dict mapping (symbol, tf) to DataFrame
        n_trials: Number of Optuna trials per stream
        n_jobs: Number of parallel workers (-1 = auto)
        train_ratio: Train/test split ratio
        verbose: Print progress
        progress_callback: Optional callback for progress updates

    Returns:
        Dict mapping (symbol, tf) to best config
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required. Install with: pip install optuna")

    def log(msg: str):
        if verbose:
            print(msg)
        if progress_callback:
            progress_callback(msg)

    # Determine worker count for stream-level parallelism
    cpu_count = os.cpu_count() or 8
    if n_jobs == -1:
        stream_workers = min(len(streams), max(1, cpu_count // 2))
    else:
        stream_workers = min(len(streams), n_jobs)

    log(f"\n[OPTUNA] Batch optimization: {len(streams)} streams")
    log(f"[OPTUNA] Trials per stream: {n_trials}")
    log(f"[OPTUNA] Parallel streams: {stream_workers}")

    results = {}

    # Sequential for now (Optuna has its own parallelism)
    # Stream-level parallelism can cause memory issues on MacBook Air
    for (sym, tf), df in streams.items():
        log(f"\n[OPTUNA] Optimizing {sym}-{tf}...")

        try:
            optimizer = SSLFlowOptimizer(
                df=df,
                sym=sym,
                tf=tf,
                n_trials=n_trials,
                n_jobs=1,  # Sequential within stream
                train_ratio=train_ratio,
                verbose=verbose,
            )

            best_config, best_score = optimizer.optimize()

            # Validate OOS
            oos_result = optimizer.validate_oos()

            # Always include detailed metrics in results
            result_config = best_config.copy() if best_config else {}

            # Add OOS metrics
            if oos_result:
                result_config["_oos_pnl"] = oos_result.get("oos_pnl", 0)
                result_config["_oos_trades"] = oos_result.get("oos_trades", 0)
                result_config["_oos_expected_r"] = oos_result.get("oos_expected_r", 0)
                result_config["_overfit_ratio"] = oos_result.get("overfit_ratio", 0)

            if oos_result and oos_result.get("is_overfit"):
                log(f"[OPTUNA] {sym}-{tf}: OVERFIT - Disabled")
                result_config["disabled"] = True
                result_config["_reason"] = "overfit"
                result_config["_rejection_details"] = {
                    "train_er": best_config.get("_expected_r", 0),
                    "oos_er": oos_result.get("oos_expected_r", 0),
                    "overfit_ratio": oos_result.get("overfit_ratio", 0),
                    "ratio_threshold": 0.40,
                    "min_oos_er_override": 0.08,
                    "rejection_reason": oos_result.get("rejection_reason", "unknown"),
                }
            elif best_score <= 0:
                log(f"[OPTUNA] {sym}-{tf}: No positive edge - Disabled")
                result_config["disabled"] = True
                result_config["_reason"] = "no_edge"
                result_config["_rejection_details"] = {
                    "best_score": best_score,
                    "train_pnl": best_config.get("_net_pnl", 0) if best_config else 0,
                    "train_trades": best_config.get("_trades", 0) if best_config else 0,
                }
            else:
                result_config["disabled"] = False
                # Log best parameters
                log(f"[OPTUNA] {sym}-{tf}: Best config found")
                log(f"  RR={best_config['rr']}, RSI={best_config['rsi']}")
                log(f"  PnL=${best_config.get('_net_pnl', 0):.2f}, E[R]={best_config.get('_expected_r', 0):.3f}")

                # Log parameter importances
                importances = optimizer.get_parameter_importances()
                if importances:
                    log(f"  Top 5 important params:")
                    for param, importance in list(importances.items())[:5]:
                        log(f"    {param}: {importance:.3f}")

            results[(sym, tf)] = result_config

        except Exception as e:
            log(f"[OPTUNA] {sym}-{tf}: Error - {e}")
            results[(sym, tf)] = {"disabled": True, "_reason": f"error: {e}"}

    log(f"\n[OPTUNA] Batch optimization complete")
    enabled = sum(1 for cfg in results.values() if not cfg.get("disabled"))
    log(f"[OPTUNA] Enabled streams: {enabled}/{len(streams)}")

    return results


# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    'OPTUNA_AVAILABLE',
    'ParameterSpace',
    'DEFAULT_PARAM_SPACE',
    'create_config_from_trial',
    'evaluate_config',
    'compute_composite_score',
    'compute_robust_score_with_oos',
    'count_active_filters',
    'create_time_series_folds',
    'validate_config_kfold',
    'SSLFlowOptimizer',
    'optimize_multiple_streams',
]
