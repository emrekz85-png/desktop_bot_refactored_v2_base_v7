# core/optimizer.py
# Optimizer functions for strategy parameter optimization
# Moved from main file for modularity (v40.5)
#
# This module contains:
# - Config grid generators (_generate_candidate_configs, _generate_quick_candidate_configs)
# - Scoring functions (_compute_optimizer_score, _score_config_for_stream)
# - Walk-forward validation (_split_data_walk_forward, _validate_config_oos, _check_overfit)
# - Main optimizer (_optimize_backtest_configs)

import os
import itertools
import random
import json
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd

from core.config import (
    TRADING_CONFIG, SYMBOL_PARAMS,
    WALK_FORWARD_CONFIG, MIN_OOS_TRADES_BY_TF,
    MIN_EXPECTANCY_R_MULTIPLE, MIN_SCORE_THRESHOLD,
    CONFIDENCE_RISK_MULTIPLIER, BASELINE_CONFIG,
    DEFAULT_STRATEGY_CONFIG,  # P5.2: Import for filter skip settings
)
from core.trade_manager import SimTradeManager
from core.utils import tf_to_timedelta
from core.trading_engine import TradingEngine

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)


# ==========================================
# CONSTANTS
# ==========================================

# Epsilon for floating point comparison (deterministic tie-breaking)
SCORE_EPSILON = 1e-10

# Minimum expectancy per trade thresholds ($ based, legacy fallback)
MIN_EXPECTANCY_PER_TRADE = {
    "1m": 6.0,   # Very noisy - need strong edge
    "5m": 4.0,   # Noisy - need decent edge per trade
    "15m": 3.0,  # Moderate noise
    "30m": 2.5,
    "1h": 2.0,   # Cleaner signals
    "4h": 1.5,   # Low noise
    "1d": 1.0,   # Very clean
}

# Strategy blacklist (empty by default, can be modified at runtime)
STRATEGY_BLACKLIST = {}


# ==========================================
# HELPER FUNCTIONS (DETERMINISM)
# ==========================================

def _config_hash(cfg: dict) -> str:
    """Deterministic hash for config tie-breaking.

    Used to ensure consistent config selection when scores are equal.
    Returns a deterministic JSON string representation of the config.
    """
    return json.dumps(cfg, sort_keys=True)


# ==========================================
# CONFIG GRID GENERATORS
# ==========================================

def _generate_candidate_configs():
    """Create a compact grid of configs to search for higher trade density.

    NOT: Slope parametresi artık taranmıyor çünkü:
    - PBEMA (200 EMA) çok yavaş hareket eder
    - Trend following stratejisinde slope filter kullanilmiyor

    Sadece SSL_Flow stratejisi test ediliyor.
    SSL Flow: SSL HYBRID baseline + AlphaTrend flow confirmation + PBEMA TP

    PR-1: Simplified grid with BASELINE parameters only.
    - sl_validation_mode="off" (no SL widening)
    - exit_profile removed (baseline mode ignores profiles)
    - Fixed baseline partial/dynamic TP params
    """

    rr_vals = np.arange(1.2, 2.6, 0.3)
    rsi_vals = np.arange(35, 76, 10)
    # AlphaTrend is now MANDATORY for SSL_Flow - only True option
    at_vals = [True]
    # Include both dynamic TP options to ensure optimizer matches what live will use
    dyn_tp_vals = [True, False]

    candidates = []

    # SSL Flow strategy configs (trend following with SSL HYBRID baseline)
    # PR-1: BASELINE MODE - no exit_profile variations
    # Read partial TP values from BASELINE_CONFIG for easy comparison testing
    partial_trigger = BASELINE_CONFIG.get("partial_trigger", 0.40)
    partial_fraction = BASELINE_CONFIG.get("partial_fraction", 0.50)

    for rr, rsi, at_active, dyn_tp in itertools.product(
        rr_vals, rsi_vals, at_vals, dyn_tp_vals
    ):
        candidates.append(
            {
                "rr": round(float(rr), 2),
                "rsi": int(rsi),
                "slope": 0.5,  # Sabit deger - artik kullanilmiyor
                "at_active": bool(at_active),
                "use_trailing": False,
                "use_partial": True,
                "use_dynamic_pbema_tp": bool(dyn_tp),
                "strategy_mode": "ssl_flow",
                # PR-1 BASELINE: Read from BASELINE_CONFIG
                "sl_validation_mode": "off",           # NO SL widening
                "partial_trigger": partial_trigger,    # From BASELINE_CONFIG
                "partial_fraction": partial_fraction,  # From BASELINE_CONFIG
                "partial_rr_adjustment": False,        # NO RR adjustment
                "dynamic_tp_only_after_partial": False,# Pre-partial too
                "dynamic_tp_clamp_mode": "none",       # NO clamp
                # === MOMENTUM TP EXTENSION (v1.2) ===
                "momentum_tp_extension": True,
                "momentum_extension_threshold": 0.80,
                "momentum_extension_multiplier": 1.5,
                # === PROGRESSIVE PARTIAL TP (v1.3) ===
                "use_progressive_partial": True,
                "partial_tranches": [
                    {"trigger": 0.40, "fraction": 0.33},
                    {"trigger": 0.70, "fraction": 0.50},
                ],
                "progressive_be_after_tranche": 1,
                # === FILTER SETTINGS (MUST MATCH config.py DEFAULTS) ===
                # CRITICAL FIX: Defaults must match DEFAULT_STRATEGY_CONFIG exactly
                # Previous bug: Optimizer tested with True, live ran with False
                "skip_body_position": DEFAULT_STRATEGY_CONFIG.get("skip_body_position", False),
                "skip_adx_filter": DEFAULT_STRATEGY_CONFIG.get("skip_adx_filter", False),
                "skip_overlap_check": DEFAULT_STRATEGY_CONFIG.get("skip_overlap_check", False),
                "skip_at_flat_filter": DEFAULT_STRATEGY_CONFIG.get("skip_at_flat_filter", False),
                "regime_adx_threshold": DEFAULT_STRATEGY_CONFIG.get("regime_adx_threshold", 20.0),
                "use_ssl_never_lost_filter": DEFAULT_STRATEGY_CONFIG.get("use_ssl_never_lost_filter", True),
                "ssl_never_lost_lookback": DEFAULT_STRATEGY_CONFIG.get("ssl_never_lost_lookback", 20),
                "use_confirmation_candle": DEFAULT_STRATEGY_CONFIG.get("use_confirmation_candle", True),
                "confirmation_candle_mode": DEFAULT_STRATEGY_CONFIG.get("confirmation_candle_mode", "close"),
                "min_pbema_distance": DEFAULT_STRATEGY_CONFIG.get("min_pbema_distance", 0.004),
                "lookback_candles": DEFAULT_STRATEGY_CONFIG.get("lookback_candles", 5),
            }
        )

    # Birkac agresif trailing secenegi ekle
    trailing_extras = []
    for base in candidates[:: max(1, len(candidates) // 20)]:
        cfg = dict(base)
        cfg["use_trailing"] = True
        trailing_extras.append(cfg)

    return candidates + trailing_extras


def _generate_quick_candidate_configs():
    """Create a minimal config grid for quick testing.

    Used when quick_mode=True for faster backtest iterations.
    Covers key combinations without exhaustive search.
    Only SSL_Flow strategy is tested.

    PR-1: BASELINE MODE - no exit_profile variations.
    """
    # Sadece en önemli RR ve RSI değerlerini kullan
    rr_vals = [1.2, 1.8, 2.4]  # 3 değer (vs 5)
    rsi_vals = [35, 55]        # 2 değer (vs 5)
    # AlphaTrend is now MANDATORY for SSL_Flow - only True option
    at_vals = [True]           # Sadece 1 değer (AlphaTrend zorunlu)
    dyn_tp_vals = [True]       # Dynamic TP usually helps, keep it simple

    candidates = []

    # SSL Flow strategy configs (trend following with SSL HYBRID baseline)
    # PR-1: BASELINE MODE - no exit_profile variations
    # Read partial TP values from BASELINE_CONFIG for easy comparison testing
    partial_trigger = BASELINE_CONFIG.get("partial_trigger", 0.40)
    partial_fraction = BASELINE_CONFIG.get("partial_fraction", 0.50)

    for rr, rsi, at_active, dyn_tp in itertools.product(
        rr_vals, rsi_vals, at_vals, dyn_tp_vals
    ):
        candidates.append({
            "rr": round(float(rr), 2),
            "rsi": int(rsi),
            "slope": 0.5,
            "at_active": bool(at_active),
            "use_trailing": False,
            "use_partial": True,
            "use_dynamic_pbema_tp": bool(dyn_tp),
            "strategy_mode": "ssl_flow",
            # PR-1 BASELINE: Read from BASELINE_CONFIG
            "sl_validation_mode": "off",           # NO SL widening
            "partial_trigger": partial_trigger,    # From BASELINE_CONFIG
            "partial_fraction": partial_fraction,  # From BASELINE_CONFIG
            "partial_rr_adjustment": False,        # NO RR adjustment
            "dynamic_tp_only_after_partial": False,# Pre-partial too
            "dynamic_tp_clamp_mode": "none",       # NO clamp
            "dynamic_tp_min_distance": 0.004,
            # === MOMENTUM TP EXTENSION (v1.2) ===
            "momentum_tp_extension": True,
            "momentum_extension_threshold": 0.80,
            "momentum_extension_multiplier": 1.5,
            # === PROGRESSIVE PARTIAL TP (v1.3) ===
            "use_progressive_partial": True,
            "partial_tranches": [
                {"trigger": 0.40, "fraction": 0.33},
                {"trigger": 0.70, "fraction": 0.50},
            ],
            "progressive_be_after_tranche": 1,
            # === FILTER SETTINGS (MUST MATCH config.py DEFAULTS) ===
            # CRITICAL FIX: Defaults must match DEFAULT_STRATEGY_CONFIG exactly
            "skip_body_position": DEFAULT_STRATEGY_CONFIG.get("skip_body_position", False),
            "skip_adx_filter": DEFAULT_STRATEGY_CONFIG.get("skip_adx_filter", False),
            "skip_overlap_check": DEFAULT_STRATEGY_CONFIG.get("skip_overlap_check", False),
            "skip_at_flat_filter": DEFAULT_STRATEGY_CONFIG.get("skip_at_flat_filter", False),
            "regime_adx_threshold": DEFAULT_STRATEGY_CONFIG.get("regime_adx_threshold", 20.0),
            "use_ssl_never_lost_filter": DEFAULT_STRATEGY_CONFIG.get("use_ssl_never_lost_filter", True),
            "ssl_never_lost_lookback": DEFAULT_STRATEGY_CONFIG.get("ssl_never_lost_lookback", 20),
            "use_confirmation_candle": DEFAULT_STRATEGY_CONFIG.get("use_confirmation_candle", True),
            "confirmation_candle_mode": DEFAULT_STRATEGY_CONFIG.get("confirmation_candle_mode", "close"),
            "min_pbema_distance": DEFAULT_STRATEGY_CONFIG.get("min_pbema_distance", 0.004),
            "lookback_candles": DEFAULT_STRATEGY_CONFIG.get("lookback_candles", 5),
        })

    # 1 trailing config ekle
    trailing_cfg = dict(candidates[0])
    trailing_cfg["use_trailing"] = True
    candidates.append(trailing_cfg)

    return candidates  # ~6-7 configs for quick mode (was ~12-15 with profiles)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _get_min_trades_for_timeframe(tf: str, num_candles: int = 20000) -> int:
    """Return minimum trade count for statistical significance based on timeframe AND data size.

    The key insight: min_trades must scale with available data.
    - 20000 candles on 15m = ~208 days → can expect ~40 trades
    - 3000 candles on 15m = ~31 days → can only expect ~6 trades
    - 3000 candles on 5m = ~10 days → can only expect ~3 trades

    Formula: min_trades = base_rate * (num_candles / base_candles)
    Where base_rate is calibrated for 20000 candles.
    """
    # Base rates for 20000 candles (these are the "ideal" minimums)
    base_candles = 20000
    tf_base_rates = {
        "1m": 150,   # Very noisy
        "5m": 80,    # Noisy
        "15m": 35,   # Moderate
        "30m": 25,   # Less noise
        "1h": 20,    # Even less noise
        "4h": 15,    # Low noise
        "1d": 10,    # Very low noise
    }

    base_rate = tf_base_rates.get(tf, 30)

    # Scale min_trades proportionally to data size
    # But enforce absolute minimum of 5 trades (below this is pure noise)
    scaled_min = max(5, int(base_rate * (num_candles / base_candles)))

    # Cap at base_rate (don't require MORE trades for small data)
    return min(scaled_min, base_rate)


# ==========================================
# WALK-FORWARD VALIDATION
# ==========================================

def _split_data_walk_forward(df: pd.DataFrame, train_ratio: float = 0.70) -> tuple:
    """Split data into train (in-sample) and test (out-of-sample) periods.

    Args:
        df: DataFrame with candle data (sorted by timestamp ascending)
        train_ratio: Ratio of data to use for training (default 0.70 = 70%)

    Returns:
        (train_df, test_df, oos_start_time): Tuple of DataFrames and OOS start timestamp
        oos_start_time is None if walk-forward is skipped
    """
    n = len(df)
    train_end = int(n * train_ratio)

    # Ensure we have enough data for both periods
    if train_end < 300 or (n - train_end) < 100:
        # Not enough data, return full df for both (skip walk-forward)
        return df, None, None

    train_df = df.iloc[:train_end].reset_index(drop=True)
    test_df = df.iloc[train_end:].reset_index(drop=True)

    # Get the OOS start timestamp for filtering backtest results
    oos_start_time = test_df['timestamp'].iloc[0] if 'timestamp' in test_df.columns else None

    return train_df, test_df, oos_start_time


def _validate_config_oos(df_test: pd.DataFrame, sym: str, tf: str, config: dict) -> dict:
    """Validate a config on out-of-sample test data.

    Args:
        df_test: Test DataFrame (out-of-sample data)
        sym: Symbol
        tf: Timeframe
        config: The optimized config to test

    Returns:
        dict with OOS metrics: {
            'oos_pnl': float,
            'oos_trades': int,
            'oos_expected_r': float,
            'oos_win_rate': float
        }
    """
    if df_test is None or len(df_test) < 100:
        return None

    try:
        net_pnl, trades, trade_pnls, trade_r_multiples = _score_config_for_stream(
            df_test, sym, tf, config
        )
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        return None

    if trades == 0:
        return {
            'oos_pnl': 0.0,
            'oos_trades': 0,
            'oos_expected_r': 0.0,
            'oos_win_rate': 0.0
        }

    expected_r = sum(trade_r_multiples) / len(trade_r_multiples) if trade_r_multiples else 0.0
    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = wins / trades if trades > 0 else 0.0

    return {
        'oos_pnl': net_pnl,
        'oos_trades': trades,
        'oos_expected_r': expected_r,
        'oos_win_rate': win_rate
    }


def _check_overfit(train_expected_r: float, oos_result: dict, tf: str) -> tuple:
    """Check if a config is overfitted by comparing train and OOS performance.

    Args:
        train_expected_r: E[R] from training data (in-sample)
        oos_result: Dict with OOS metrics from _validate_config_oos
        tf: Timeframe

    Returns:
        (is_overfit: bool, overfit_ratio: float, reason: str)
    """
    if oos_result is None:
        return False, 1.0, "no_oos_data"

    # Use timeframe-specific min OOS trades (quant trader recommendation)
    # Lower timeframes need more OOS trades to be statistically significant
    min_test_trades = MIN_OOS_TRADES_BY_TF.get(tf, WALK_FORWARD_CONFIG.get("min_test_trades", 3))
    min_overfit_ratio = WALK_FORWARD_CONFIG.get("min_overfit_ratio", 0.50)

    oos_trades = oos_result.get('oos_trades', 0)
    oos_expected_r = oos_result.get('oos_expected_r', 0.0)

    # P5 FIX: Zero OOS trades = treat as OVERFIT (can't validate)
    # Previously this returned is_overfit=False which let unvalidated configs through
    if oos_trades == 0:
        return True, 0.0, "zero_oos_trades_overfit"

    # Not enough OOS trades for statistical significance (but some trades exist)
    if oos_trades < min_test_trades:
        return False, 1.0, f"insufficient_oos_trades ({oos_trades}<{min_test_trades})"

    # OOS is negative = clear overfit
    if oos_expected_r < 0:
        return True, 0.0, "negative_oos_expected_r"

    # Calculate overfit ratio (how much of IS performance carries to OOS)
    if train_expected_r > 0:
        overfit_ratio = oos_expected_r / train_expected_r
    else:
        # Train was also bad, can't calculate ratio
        return False, 1.0, "train_not_positive"

    # Check if OOS degradation is too severe
    if overfit_ratio < min_overfit_ratio:
        return True, overfit_ratio, f"oos_degradation ({overfit_ratio:.2f}<{min_overfit_ratio})"

    return False, overfit_ratio, "ok"


# ==========================================
# SCORING FUNCTIONS
# ==========================================

def _compute_optimizer_score(net_pnl: float, trades: int, trade_pnls: list,
                              min_trades: int = 40, hard_min_trades: int = 5,
                              reject_negative_pnl: bool = True, tf: str = "15m",
                              trade_r_multiples: list = None) -> float:
    """Compute a robust optimizer score that prioritizes EDGE QUALITY over trade count.

    P5 REWRITE: Changed from trade-count-rewarding to edge-quality-focused scoring.
    Previous formula: score = net_pnl * trade_confidence (rewarded more trades)
    New formula: score = expected_r * 100 * trade_factor * consistency * dd_penalty

    Key changes:
    - E[R] is the main score driver, not raw PnL
    - Trade count is a PENALTY for low counts, NOT a reward for high counts
    - This prevents optimizer from gaming the system with many low-quality trades

    Args:
        net_pnl: Total net profit/loss
        trades: Number of trades
        trade_pnls: List of individual trade PnLs
        min_trades: Trades needed for full confidence (default 40)
        hard_min_trades: Absolute minimum trades, below this = reject (default 5)
        reject_negative_pnl: If True, reject configs with net_pnl <= 0 (default True)
        tf: Timeframe for looking up expectancy threshold
        trade_r_multiples: List of R-multiples per trade (PnL/Risk for each trade)

    Returns:
        Composite score (higher is better, -inf for rejected configs)
    """
    # HARD REJECT: Negative PnL = no edge, don't use this config
    if reject_negative_pnl and net_pnl <= 0:
        return -float("inf")

    # HARD REJECT: Too few trades = statistically meaningless
    if trades < hard_min_trades:
        return -float("inf")

    if trades == 0:
        return -float("inf")

    # Calculate E[R] - this is the PRIMARY score driver now
    if trade_r_multiples and len(trade_r_multiples) > 0:
        expected_r = sum(trade_r_multiples) / len(trade_r_multiples)
    else:
        # Fallback: estimate E[R] from PnL (assume average risk of $35)
        expected_r = (net_pnl / trades) / 35.0 if trades > 0 else 0

    # HARD REJECT: E[R] too low = barely positive, not a real edge
    min_expected_r = MIN_EXPECTANCY_R_MULTIPLE.get(tf, 0.08)
    if expected_r < min_expected_r:
        return -float("inf")

    # P5: Trade count PENALTY (not reward!)
    # Punish < 10 trades heavily, neutral 10-30, slight penalty > 50 (overtrading)
    if trades < 5:
        trade_factor = 0.3  # Heavy penalty
    elif trades < 10:
        trade_factor = 0.5 + (trades - 5) * 0.1  # 0.5 - 1.0
    elif trades <= 30:
        trade_factor = 1.0  # Neutral zone
    elif trades <= 50:
        trade_factor = 1.0 - (trades - 30) * 0.005  # Slight penalty 1.0 - 0.9
    else:
        trade_factor = 0.9 - min(0.2, (trades - 50) * 0.005)  # 0.9 - 0.7

    # Average PnL per trade (expectancy)
    avg_pnl = net_pnl / trades

    # Downside deviation (Sortino-style) - only penalize negative variance
    if len(trade_pnls) >= 2:
        negative_pnls = [p for p in trade_pnls if p < 0]
        if negative_pnls:
            downside_std = (sum(p**2 for p in negative_pnls) / len(negative_pnls)) ** 0.5
        else:
            downside_std = 0.0

        # Consistency bonus: lower variance = more reliable edge
        if downside_std > 0 and avg_pnl != 0:
            consistency_ratio = abs(avg_pnl) / (downside_std + 1e-6)
            consistency_factor = min(1.5, 0.5 + consistency_ratio * 0.25)
        else:
            consistency_factor = 1.0 if avg_pnl > 0 else 0.5

        # NEW: Max drawdown penalty
        # Calculate running max drawdown from trade PnLs
        cumulative = 0
        peak = 0
        max_dd = 0
        for pnl in trade_pnls:
            cumulative += pnl
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        # Penalize if max drawdown exceeds 50% of total profit (or is large absolute)
        if net_pnl > 0 and max_dd > net_pnl * 0.5:
            dd_penalty = 0.7  # 30% penalty for excessive drawdown
        elif max_dd > 100:  # $100 absolute drawdown threshold
            dd_penalty = 0.85
        else:
            dd_penalty = 1.0
    else:
        consistency_factor = 0.5  # Single trade = very unreliable
        dd_penalty = 0.5  # Heavy penalty for single trade

    # Win rate bonus (slight preference for higher win rates at equal PnL)
    if trade_pnls:
        win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)
        # Penalize very low win rates even with good RR
        if win_rate < 0.25:
            win_rate_factor = 0.7  # Too low WR = high variance
        elif win_rate < 0.35:
            win_rate_factor = 0.85
        else:
            win_rate_factor = 0.9 + win_rate * 0.2  # 0.9 to 1.1
    else:
        win_rate_factor = 1.0

    # Note: Negative PnL configs are already hard-rejected above if reject_negative_pnl=True
    # This branch only executes for positive PnL configs

    # P5: Final score = E[R] * 100 * trade_factor * consistency * win_rate * dd_penalty
    # Using E[R] * 100 to normalize to comparable scale with old PnL-based scoring
    score = expected_r * 100 * trade_factor * consistency_factor * win_rate_factor * dd_penalty

    return score


def _score_config_for_stream(df: pd.DataFrame, sym: str, tf: str, config: dict) -> Tuple[float, int, list, list]:
    """Simulate a single timeframe with the given config and return (net_pnl, trades, trade_pnls, trade_r_multiples).

    Not: Öndeki uyumsuzluk, tarayıcıdaki skorlamanın mum kapanışından
    aynı mumda giriş yapıp cooldown/açık trade kontrollerini atlaması yüzünden
    backtestten daha fazla trade ve PnL raporlamasından kaynaklanıyordu. Burada
    backtest ile birebir aynı kuralları (cooldown, açık trade engeli ve bir
    sonraki mum açılışından giriş) uygularız ki skor ve backtest sonuçları
    tutarlı kalsın.
    """

    tm = SimTradeManager(initial_balance=TRADING_CONFIG["initial_balance"])
    warmup = 250
    end = len(df) - 2
    if end <= warmup:
        return 0.0, 0, [], []

    # PERFORMANCE: Extract NumPy arrays once before the loop (10-50x faster than df.iloc[i])
    # Ensure timestamp is datetime64 (may become object after multiprocessing serialization)
    timestamps = pd.to_datetime(df["timestamp"]).values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values

    # SSL Flow ve Keltner Bounce icin EMA200 kullan
    strategy_mode = config.get("strategy_mode", "ssl_flow")
    if strategy_mode == "ssl_flow_DISABLED":  # EMA150 artik kullanilmiyor
        pb_top_col = "pb_ema_top_150"
        pb_bot_col = "pb_ema_bot_150"
    else:
        pb_top_col = "pb_ema_top"
        pb_bot_col = "pb_ema_bot"

    pb_tops = df.get(pb_top_col, df["close"]).values if pb_top_col in df.columns else closes
    pb_bots = df.get(pb_bot_col, df["close"]).values if pb_bot_col in df.columns else closes

    # Extract AlphaTrend arrays for momentum TP extension
    at_buyers_arr = df["at_buyers_dominant"].values if "at_buyers_dominant" in df.columns else None
    at_sellers_arr = df["at_sellers_dominant"].values if "at_sellers_dominant" in df.columns else None

    # v46.x: Extract ATR array for ATR-based BE buffer
    atr_arr = df["atr"].values if "atr" in df.columns else None

    for i in range(warmup, end):
        event_time = pd.Timestamp(timestamps[i]) + tf_to_timedelta(tf)

        # Build candle_data dict for this candle
        candle_data = {}
        if at_buyers_arr is not None and at_sellers_arr is not None:
            candle_data["at_buyers_dominant"] = bool(at_buyers_arr[i])
            candle_data["at_sellers_dominant"] = bool(at_sellers_arr[i])
        # v46.x: Add ATR for ATR-based BE buffer
        if atr_arr is not None:
            candle_data["atr"] = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else None

        tm.update_trades(
            sym,
            tf,
            candle_high=float(highs[i]),
            candle_low=float(lows[i]),
            candle_close=float(closes[i]),
            candle_time_utc=event_time,
            pb_top=float(pb_tops[i]),
            pb_bot=float(pb_bots[i]),
            candle_data=candle_data,
        )

        # Use wrapper function to support both ssl_flow and keltner_bounce strategies
        s_type, s_entry, s_tp, s_sl, s_reason = TradingEngine.check_signal(
            df,
            config=config,
            index=i,
            return_debug=False,
        )

        if not (s_type and "ACCEPTED" in s_reason):
            continue

        has_open = any(
            t.get("symbol") == sym and t.get("timeframe") == tf for t in tm.open_trades
        )
        if has_open or tm.check_cooldown(sym, tf, event_time):
            continue

        # Access next candle for entry price (more realistic simulation)
        entry_open = float(opens[i + 1])
        open_ts = timestamps[i + 1]
        ts_str = (pd.Timestamp(open_ts) + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")

        # RR re-validation with actual entry price (next candle open)
        # Signal calculated RR based on close, but actual entry is at next open
        # This prevents edge degradation from slippage
        min_rr = config["rr"]
        if s_type == "LONG":
            actual_risk = entry_open - s_sl
            actual_reward = s_tp - entry_open
        else:  # SHORT
            actual_risk = s_sl - entry_open
            actual_reward = entry_open - s_tp

        if actual_risk <= 0 or actual_reward <= 0:
            continue  # Invalid RR after slippage

        actual_rr = actual_reward / actual_risk
        if actual_rr < min_rr * 0.9:  # 10% tolerance for slippage
            continue  # RR degraded too much after slippage

        tm.open_trade(
            {
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
            }
        )

    unique_trades = len({t.get("id") for t in tm.history}) if tm.history else 0
    trade_pnls = [t.get("pnl", 0.0) for t in tm.history] if tm.history else []
    # R-Multiple listesi (E[R] hesabı için)
    trade_r_multiples = tm.trade_r_multiples if hasattr(tm, 'trade_r_multiples') else []
    return tm.total_pnl, unique_trades, trade_pnls, trade_r_multiples


# ==========================================
# MAIN OPTIMIZER
# ==========================================

def _optimize_backtest_configs(
    streams: dict,
    requested_pairs: list,
    progress_callback=None,
    log_to_stdout: bool = True,
    use_walk_forward: bool = None,  # None = WALK_FORWARD_CONFIG["enabled"] kullan
    quick_mode: bool = False,  # True = azaltılmış config grid (daha hızlı)
):
    """Brute-force search to find the best config (by net pnl) per symbol/timeframe.

    Walk-forward validation enabled by default:
    - Splits data 70% train, 30% test
    - Optimizes on train data
    - Validates best config on test data
    - Rejects overfitted configs (OOS E[R] < 50% of train E[R])

    Quick mode (quick_mode=True):
    - Uses reduced config grid (24 instead of 120 configs)
    - ~5x faster optimization
    """

    def log(msg: str):
        if log_to_stdout:
            print(msg)
        if progress_callback:
            progress_callback(msg)

    # Walk-forward ayarı
    walk_forward_enabled = use_walk_forward if use_walk_forward is not None else WALK_FORWARD_CONFIG.get("enabled", True)

    # Quick mode: Azaltılmış config grid kullan
    if quick_mode:
        candidates = _generate_quick_candidate_configs()
        mode_str = "HIZLI"
    else:
        candidates = _generate_candidate_configs()
        mode_str = "TAM"

    total_jobs = len([1 for pair in requested_pairs if pair in streams]) * len(candidates)
    if total_jobs == 0:
        return {}

    best_by_pair = {}
    completed = 0
    next_progress = 5

    # M3 Performance: Use 3 workers (P-cores only, avoid E-cores for CPU-bound work)
    # M3 has 4 Performance + 4 Efficiency cores
    # Use all available cores minus one for the main thread
    cpu_count = os.cpu_count() or 4
    max_workers = max(1, cpu_count - 1)
    wf_status = "Açık" if walk_forward_enabled else "Kapalı"
    quick_icon = "⚡" if quick_mode else ""

    # Windows multiprocessing uyumluluğu: Windows'ta ProcessPoolExecutor yerine
    # ThreadPoolExecutor kullan (DLL import ve bellek sorunlarını önler)
    import sys
    use_threads = sys.platform == "win32"
    executor_type = "Thread" if use_threads else "Process"

    log(
        f"[OPT]{quick_icon} {len(candidates)} farklı ayar taranacak ({mode_str}). "
        f"Paralel: {max_workers} ({executor_type}), Walk-Forward: {wf_status}"
    )

    for sym, tf in requested_pairs:
        if (sym, tf) not in streams:
            continue

        # Skip disabled symbol/timeframe combinations
        sym_cfg = SYMBOL_PARAMS.get(sym, {})
        tf_cfg = sym_cfg.get(tf, {}) if isinstance(sym_cfg, dict) else {}
        if tf_cfg.get("disabled", False):
            log(f"[OPT][{sym}-{tf}] Atlandı (disabled)")
            continue

        df_full = streams[(sym, tf)]
        num_candles_full = len(df_full)

        # Walk-forward: veriyi train/test olarak böl
        oos_start_time = None  # OOS başlangıç zamanı (backtest filtreleme için)
        if walk_forward_enabled:
            train_ratio = WALK_FORWARD_CONFIG.get("train_ratio", 0.70)
            df_train, df_test, oos_start_time = _split_data_walk_forward(df_full, train_ratio)
            if df_test is not None:
                log(f"[OPT][{sym}-{tf}] Walk-Forward: Train={len(df_train)} mum, Test={len(df_test)} mum")
            else:
                log(f"[OPT][{sym}-{tf}] Walk-Forward: Yetersiz veri, tam veri kullanılacak")
        else:
            df_train = df_full
            df_test = None

        df = df_train  # Optimizasyon için train verisini kullan
        num_candles = len(df)
        best_cfg = None
        best_score = -float("inf")
        best_pnl = -float("inf")
        best_trades = 0
        best_expected_r = 0.0  # E[R] değeri

        # Strategy blacklist check (currently disabled)
        if STRATEGY_BLACKLIST.get((sym, tf), False):
            stream_candidates = candidates
            log(f"[OPT][{sym}-{tf}] Blacklist'te - devam ediliyor")
        else:
            stream_candidates = candidates

        def handle_result(cfg, net_pnl, trades, trade_pnls, trade_r_multiples=None):
            nonlocal completed, next_progress, best_cfg, best_score, best_pnl, best_trades, best_expected_r

            completed += 1
            progress = (completed / total_jobs) * 100
            if progress >= next_progress:
                log(f"[OPT] %{progress:.1f} tamamlandı...")
                next_progress += 5

            if trades == 0:
                return

            # Use timeframe AND data-size aware min_trades for statistical significance
            tf_min_trades = _get_min_trades_for_timeframe(tf, num_candles)
            # Hard minimum: at least 3 trades (absolute floor)
            hard_min = max(3, tf_min_trades // 3)

            # Use the new composite score with anti-overfit measures
            # reject_negative_pnl=True means configs with net_pnl <= 0 are rejected
            # trade_r_multiples enables R-multiple based E[R] threshold
            score = _compute_optimizer_score(
                net_pnl, trades, trade_pnls,
                min_trades=tf_min_trades,
                hard_min_trades=hard_min,
                reject_negative_pnl=True,
                tf=tf,
                trade_r_multiples=trade_r_multiples
            )

            # Deterministic score comparison with epsilon tolerance
            # If scores are equal (within epsilon), use config hash for tie-breaking
            if score > best_score + SCORE_EPSILON or (
                abs(score - best_score) <= SCORE_EPSILON and
                best_cfg is not None and
                _config_hash(cfg) < _config_hash(best_cfg)
            ):
                best_score = score
                best_pnl = net_pnl
                best_cfg = cfg
                best_trades = trades
                # E[R] hesapla
                if trade_r_multiples and len(trade_r_multiples) > 0:
                    best_expected_r = sum(trade_r_multiples) / len(trade_r_multiples)
                else:
                    best_expected_r = 0.0

        try:
            # Windows: ThreadPoolExecutor kullan (DLL import ve bellek sorunlarını önler)
            # Linux/Mac: ProcessPoolExecutor kullan (daha hızlı)
            ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

            with ExecutorClass(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_score_config_for_stream, df, sym, tf, cfg): cfg
                    for cfg in stream_candidates
                }

                # Collect all results first (non-deterministic order from as_completed)
                all_results = []
                for future in as_completed(futures):
                    cfg = futures[future]
                    try:
                        result = future.result()
                        all_results.append((cfg, result))
                    except BrokenProcessPool:
                        # Havuza ait iş parçacığı çöktüyse seri moda düş.
                        raise
                    except Exception as exc:
                        log(f"[OPT][{sym}-{tf}] Skorlama hatası (cfg={cfg}): {exc}")
                        all_results.append((cfg, None))

                # Sort by config hash for deterministic processing order
                all_results.sort(key=lambda x: _config_hash(x[0]))

                # Process in deterministic order
                for cfg, result in all_results:
                    if result is None:
                        continue
                    net_pnl, trades, trade_pnls, trade_r_multiples = result
                    handle_result(cfg, net_pnl, trades, trade_pnls, trade_r_multiples)

        except BrokenProcessPool as exc:
            log(
                f"[OPT][{sym}-{tf}] Paralel işleme havuzu durdu (neden: {exc}). "
                "Seri moda düşülüyor."
            )

            for cfg in stream_candidates:
                try:
                    net_pnl, trades, trade_pnls, trade_r_multiples = _score_config_for_stream(df, sym, tf, cfg)
                except Exception as exc:
                    log(f"[OPT][{sym}-{tf}] Seri skorlama hatası (cfg={cfg}): {exc}")
                    continue

                handle_result(cfg, net_pnl, trades, trade_pnls, trade_r_multiples)

        # Get min_trades for this timeframe for logging (using actual num_candles)
        tf_min_trades = _get_min_trades_for_timeframe(tf, num_candles)
        hard_min = max(3, tf_min_trades // 3)
        min_score = MIN_SCORE_THRESHOLD.get(tf, 10.0)
        # Walk-Forward kullanılıyorsa, score eşiğini train_ratio ile ayarla
        # (Training seti daha küçük olduğu için daha az trade ve daha düşük score olur)
        if walk_forward_enabled:
            train_ratio = WALK_FORWARD_CONFIG.get("train_ratio", 0.70)
            min_score = min_score * train_ratio
        min_expected_r = MIN_EXPECTANCY_R_MULTIPLE.get(tf, 0.08)

        # Additional gate: score must meet minimum threshold for this timeframe
        if best_cfg and best_score < min_score:
            log(
                f"[OPT][{sym}-{tf}] Score ({best_score:.2f}) < min threshold ({min_score:.2f}) "
                f"→ Weak edge, DEVRE DIŞI"
            )
            best_cfg = None  # Reject weak-edge config

        # Walk-Forward OOS Validation
        oos_result = None
        overfit_ratio = 1.0
        if best_cfg and walk_forward_enabled and df_test is not None:
            log(f"[OPT][{sym}-{tf}] Walk-Forward: Out-of-Sample test yapılıyor...")
            oos_result = _validate_config_oos(df_test, sym, tf, best_cfg)

            if oos_result:
                is_overfit, overfit_ratio, overfit_reason = _check_overfit(
                    best_expected_r, oos_result, tf
                )

                oos_expected_r = oos_result.get('oos_expected_r', 0)
                oos_trades = oos_result.get('oos_trades', 0)
                oos_pnl = oos_result.get('oos_pnl', 0)

                if is_overfit:
                    log(
                        f"[OPT][{sym}-{tf}] ❌ OVERFIT TESPİT EDİLDİ! "
                        f"Train E[R]={best_expected_r:.3f}, OOS E[R]={oos_expected_r:.3f}, "
                        f"Ratio={overfit_ratio:.2f} | Sebep: {overfit_reason} → DEVRE DIŞI"
                    )
                    best_cfg = None  # Reject overfitted config
                else:
                    log(
                        f"[OPT][{sym}-{tf}] ✓ OOS Doğrulandı: "
                        f"Train E[R]={best_expected_r:.3f}, OOS E[R]={oos_expected_r:.3f}, "
                        f"OOS PnL=${oos_pnl:.2f}, OOS Trades={oos_trades}, "
                        f"Ratio={overfit_ratio:.2f}"
                    )

        if best_cfg:
            # Confidence level for risk multiplier (stored as string key)
            if best_trades >= tf_min_trades:
                confidence_level = "high"
                confidence_display = "✓ Yüksek"
            elif best_trades >= tf_min_trades * 0.6:
                confidence_level = "medium"
                confidence_display = "~ Orta"
            else:
                confidence_level = "low"
                confidence_display = "⚠ Düşük"

            # Low confidence = no trades (risk multiplier = 0)
            if confidence_level == "low":
                log(
                    f"[OPT][{sym}-{tf}] Düşük güven ({best_trades}/{tf_min_trades} trade) "
                    f"→ Risk çarpanı=0, DEVRE DIŞI"
                )
                best_by_pair[(sym, tf)] = {"disabled": True, "_reason": "low_confidence"}
            else:
                # Explicit disabled=False ensures stream is enabled even if previously disabled
                config_result = {
                    **best_cfg,
                    "disabled": False,  # Explicitly enable - overwrites any previous disabled state
                    "_net_pnl": best_pnl,
                    "_trades": best_trades,
                    "_score": best_score,
                    "confidence": confidence_level,  # For risk multiplier (underscore yok - save'de korunur)
                    "_expectancy": best_pnl / best_trades if best_trades > 0 else 0,
                    "_expected_r": best_expected_r,  # E[R] - R-multiple bazlı expectancy (train)
                }

                # Walk-Forward OOS metrikleri ekle (varsa)
                if oos_result:
                    config_result["_oos_expected_r"] = oos_result.get('oos_expected_r', 0)
                    config_result["_oos_pnl"] = oos_result.get('oos_pnl', 0)
                    config_result["_oos_trades"] = oos_result.get('oos_trades', 0)
                    config_result["_overfit_ratio"] = overfit_ratio
                    config_result["_walk_forward_validated"] = True
                    # OOS başlangıç zamanını kaydet - backtest sonuçlarını filtrelemek için
                    if oos_start_time is not None:
                        config_result["_oos_start_time"] = oos_start_time
                else:
                    config_result["_walk_forward_validated"] = False

                best_by_pair[(sym, tf)] = config_result

                dyn_tp_str = "Açık" if best_cfg.get('use_dynamic_pbema_tp') else "Kapalı"
                risk_mult = CONFIDENCE_RISK_MULTIPLIER.get(confidence_level, 1.0)

                # OOS bilgisi log'a ekle (varsa)
                oos_info = ""
                if oos_result and overfit_ratio > 0:
                    oos_info = f", OOS Ratio={overfit_ratio:.2f}"

                log(
                    f"[OPT][{sym}-{tf}] En iyi ayar: RR={best_cfg['rr']}, RSI={best_cfg['rsi']}, "
                    f"Slope={best_cfg['slope']}, AT={'Açık' if best_cfg['at_active'] else 'Kapalı'}, "
                    f"DynTP={dyn_tp_str} | Net PnL=${best_pnl:.2f}, E[R]={best_expected_r:.3f}{oos_info}, "
                    f"Trades={best_trades}/{tf_min_trades}, Score={best_score:.2f}, "
                    f"Güven={confidence_display}, Risk={risk_mult:.0%}"
                )
        else:
            log(
                f"[OPT][{sym}-{tf}] Geçerli config bulunamadı - trade<{hard_min} veya PnL<=0 veya "
                f"E[R]<{min_expected_r:.2f} (min {tf_min_trades} trade) → DEVRE DIŞI"
            )
            # Mark this stream as disabled so backtest skips it
            best_by_pair[(sym, tf)] = {"disabled": True, "_reason": "no_positive_config"}

    log("[OPT] Tarama tamamlandı. Bulunan ayarlar backtest'e uygulanacak.")
    return best_by_pair


# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    # Constants
    'MIN_EXPECTANCY_PER_TRADE',
    'STRATEGY_BLACKLIST',
    # Config generators
    '_generate_candidate_configs',
    '_generate_quick_candidate_configs',
    # Helper functions
    '_get_min_trades_for_timeframe',
    # Walk-forward
    '_split_data_walk_forward',
    '_validate_config_oos',
    '_check_overfit',
    # Scoring
    '_compute_optimizer_score',
    '_score_config_for_stream',
    # Main optimizer
    '_optimize_backtest_configs',
]
