# strategies/pbema_retest.py
# PBEMA Strategy - Based on Real Trade Analysis (2026-01-04)
#
# Core Concept:
# "PBEMA bulutundan güçlü fiyat sekmesi yaşanabilir!"
#
# ============================================================================
# KEY INSIGHTS FROM REAL TRADES:
# ============================================================================
#
# 1. LEVEL MUST BE PROVEN:
#    - NO 8: "Fiyat bir çok kez PBEMA bulutuna değip aşağıya düşüyor"
#    - NO 17: "ard arda Resistance olarak calisiyor"
#    - Minimum 3+ rejections before entry
#
# 2. APPROACH-BASED ROLE:
#    - Fiyat AŞAĞIDAN gelip PBEMA'ya değdi → RESISTANCE → SHORT
#    - Fiyat YUKARIDAN gelip PBEMA'ya değdi → SUPPORT → LONG
#
# 3. MOMENTUM EXIT:
#    - "momentum yavaşlayan dek takip edip TP"
#
# ============================================================================

from typing import Tuple, Union, Dict, Optional
import pandas as pd
import numpy as np

from .base import SignalResult, SignalResultWithDebug


def _count_recent_rejections(
    df: pd.DataFrame,
    current_idx: int,
    direction: str,
    lookback: int = 20,
    min_wick_ratio: float = 0.20,
    touch_tolerance: float = 0.003,
) -> Tuple[int, float]:
    """
    Count recent rejections from PBEMA.

    From real trades: "Fiyat bir çok kez PBEMA bulutuna değip aşağıya düşüyor"

    Args:
        df: DataFrame
        current_idx: Current index
        direction: "RESISTANCE" or "SUPPORT"
        lookback: Candles to look back
        min_wick_ratio: Minimum wick for rejection
        touch_tolerance: Touch tolerance

    Returns:
        Tuple[int, float]: (rejection_count, avg_wick_ratio)
    """
    rejections = 0
    wick_ratios = []

    for i in range(current_idx - lookback, current_idx):
        if i < 0 or i >= len(df):
            continue

        candle = df.iloc[i]
        high = float(candle["high"])
        low = float(candle["low"])
        open_ = float(candle["open"])
        close = float(candle["close"])
        pb_top = float(candle["pb_ema_top"])
        pb_bot = float(candle["pb_ema_bot"])

        candle_range = high - low
        if candle_range <= 0:
            continue

        body_high = max(open_, close)
        body_low = min(open_, close)
        upper_wick = high - body_high
        lower_wick = body_low - low

        if direction == "RESISTANCE":
            # Price touching PBEMA from below
            touched = high >= pb_bot * (1 - touch_tolerance)
            below_pbema = close < pb_bot
            wick_ratio = upper_wick / candle_range

            if touched and below_pbema and wick_ratio >= min_wick_ratio:
                rejections += 1
                wick_ratios.append(wick_ratio)

        else:  # SUPPORT
            # Price touching PBEMA from above
            touched = low <= pb_top * (1 + touch_tolerance)
            above_pbema = close > pb_top
            wick_ratio = lower_wick / candle_range

            if touched and above_pbema and wick_ratio >= min_wick_ratio:
                rejections += 1
                wick_ratios.append(wick_ratio)

    avg_wick = np.mean(wick_ratios) if wick_ratios else 0.0
    return rejections, avg_wick


def check_pbema_retest_signal(
    df: pd.DataFrame,
    index: int = -2,
    min_rr: float = 1.0,
    # === Approach Detection ===
    approach_lookback: int = 10,
    approach_threshold: float = 0.70,
    # === Rejection Requirements ===
    rejection_lookback: int = 20,
    min_rejections: int = 3,  # "bir çok kez"
    min_wick_ratio: float = 0.20,  # 20% wick
    touch_tolerance: float = 0.003,
    # === TP/SL ===
    tp_target: str = "percentage",
    tp_percentage: float = 0.015,
    sl_buffer: float = 0.005,
    min_sl_distance: float = 0.01,
    use_atr_sl: bool = True,
    atr_sl_multiplier: float = 1.5,
    # === Filters ===
    require_trend_alignment: bool = False,
    # === Debug ===
    return_debug: bool = False,
) -> Union[SignalResult, SignalResultWithDebug]:
    """
    PBEMA Strategy - Based on Real Trade Analysis.

    This strategy trades PBEMA as support/resistance based on:
    1. Approach direction (where price is coming from)
    2. Level strength (multiple rejections required)
    3. Current candle rejection (wick)

    Entry Criteria:
    1. Clear approach direction (70%+ candles on one side)
    2. PBEMA has acted as S/R multiple times (3+ rejections)
    3. Current candle shows rejection (wick >= 20%)

    Args:
        df: DataFrame with OHLCV + indicators
        index: Candle index
        approach_lookback: Candles to determine approach
        approach_threshold: % threshold for approach direction
        min_rejections: Minimum prior rejections
        min_wick_ratio: Minimum wick for rejection
        return_debug: Return debug info

    Returns:
        SignalResult or SignalResultWithDebug
    """
    debug_info = {
        "approach_direction": None,
        "pbema_role": None,
        "prior_rejections": 0,
        "avg_wick_ratio": 0.0,
        "current_wick_ratio": 0.0,
        "touch_detected": False,
        "rejection_confirmed": False,
    }

    def _ret(s_type, entry, tp, sl, reason):
        if return_debug:
            return s_type, entry, tp, sl, reason, debug_info
        return s_type, entry, tp, sl, reason

    # Validate input
    if df is None or df.empty:
        return _ret(None, None, None, None, "No Data")

    required_cols = ["open", "high", "low", "close", "pb_ema_top", "pb_ema_bot"]
    for col in required_cols:
        if col not in df.columns:
            return _ret(None, None, None, None, f"Missing {col}")

    try:
        curr = df.iloc[index]
    except (IndexError, KeyError):
        return _ret(None, None, None, None, "Index Error")

    abs_index = index if index >= 0 else (len(df) + index)
    if abs_index < max(approach_lookback, rejection_lookback) + 20:
        return _ret(None, None, None, None, "Not Enough Data")

    # Extract values
    open_ = float(curr["open"])
    high = float(curr["high"])
    low = float(curr["low"])
    close = float(curr["close"])
    pb_top = float(curr["pb_ema_top"])
    pb_bot = float(curr["pb_ema_bot"])
    pb_mid = (pb_top + pb_bot) / 2

    if any(pd.isna([open_, high, low, close, pb_top, pb_bot])):
        return _ret(None, None, None, None, "NaN Values")

    # Get baseline
    baseline = None
    if "baseline" in df.columns:
        baseline = float(curr["baseline"]) if not pd.isna(curr.get("baseline")) else None

    # ==========================================================================
    # STEP 1: Determine Approach Direction
    # ==========================================================================

    candles_below = 0
    candles_above = 0

    for i in range(abs_index - approach_lookback, abs_index):
        if i < 0:
            continue
        candle = df.iloc[i]
        c_close = float(candle["close"])
        c_pb_mid = (float(candle["pb_ema_top"]) + float(candle["pb_ema_bot"])) / 2

        if c_close < c_pb_mid:
            candles_below += 1
        else:
            candles_above += 1

    total = candles_below + candles_above
    if total == 0:
        return _ret(None, None, None, None, "No Valid Candles")

    below_ratio = candles_below / total
    above_ratio = candles_above / total

    if below_ratio >= approach_threshold:
        approach_direction = "FROM_BELOW"
        pbema_role = "RESISTANCE"
    elif above_ratio >= approach_threshold:
        approach_direction = "FROM_ABOVE"
        pbema_role = "SUPPORT"
    else:
        return _ret(None, None, None, None, "No Clear Approach Direction")

    debug_info["approach_direction"] = approach_direction
    debug_info["pbema_role"] = pbema_role

    # ==========================================================================
    # STEP 2: Count Prior Rejections
    # ==========================================================================

    prior_rejections, avg_wick = _count_recent_rejections(
        df, abs_index, pbema_role, rejection_lookback, min_wick_ratio, touch_tolerance
    )

    debug_info["prior_rejections"] = prior_rejections
    debug_info["avg_wick_ratio"] = avg_wick

    if prior_rejections < min_rejections:
        return _ret(None, None, None, None,
                   f"Insufficient Rejections ({prior_rejections}/{min_rejections})")

    # ==========================================================================
    # STEP 3: Check Current Candle Touch + Rejection
    # ==========================================================================

    candle_range = high - low
    if candle_range <= 0:
        return _ret(None, None, None, None, "Zero Range Candle")

    body_high = max(open_, close)
    body_low = min(open_, close)
    upper_wick = high - body_high
    lower_wick = body_low - low
    upper_wick_ratio = upper_wick / candle_range
    lower_wick_ratio = lower_wick / candle_range

    # ==========================================================================
    # SCENARIO: PBEMA as RESISTANCE (Price from below)
    # ==========================================================================

    if pbema_role == "RESISTANCE":
        # Check touch and rejection
        touched = high >= pb_bot * (1 - touch_tolerance)
        below_pbema = close < pb_bot
        has_rejection = upper_wick_ratio >= min_wick_ratio

        debug_info["touch_detected"] = touched
        debug_info["current_wick_ratio"] = upper_wick_ratio
        debug_info["rejection_confirmed"] = touched and below_pbema and has_rejection

        if not (touched and below_pbema and has_rejection):
            return _ret(None, None, None, None, "No Valid RESISTANCE Rejection")

        # Optional trend alignment: SHORT only if below baseline (downtrend)
        if require_trend_alignment and baseline is not None:
            if close > baseline:
                return _ret(None, None, None, None, "Counter-trend SHORT (above baseline)")

        entry = close
        tp = entry * (1 - tp_percentage)

        if use_atr_sl and "atr" in df.columns:
            atr = float(curr.get("atr", 0))
            if atr > 0:
                sl = entry + (atr * atr_sl_multiplier)
            else:
                sl = pb_top * (1 + sl_buffer)
        else:
            sl = pb_top * (1 + sl_buffer)

        sl_dist = (sl - entry) / entry
        if sl_dist < min_sl_distance:
            sl = entry * (1 + min_sl_distance)

        if tp >= entry or sl <= entry:
            return _ret(None, None, None, None, "Invalid TP/SL")

        risk = sl - entry
        reward = entry - tp
        if risk <= 0 or reward <= 0:
            return _ret(None, None, None, None, "Invalid RR")

        rr = reward / risk
        if rr < min_rr:
            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

        reason = f"PBEMA_RESIST_SHORT(R:{rr:.2f},Rej:{prior_rejections})"
        return _ret("SHORT", entry, tp, sl, reason)

    # ==========================================================================
    # SCENARIO: PBEMA as SUPPORT (Price from above)
    # ==========================================================================

    if pbema_role == "SUPPORT":
        # Check touch and rejection
        touched = low <= pb_top * (1 + touch_tolerance)
        above_pbema = close > pb_top
        has_rejection = lower_wick_ratio >= min_wick_ratio

        debug_info["touch_detected"] = touched
        debug_info["current_wick_ratio"] = lower_wick_ratio
        debug_info["rejection_confirmed"] = touched and above_pbema and has_rejection

        if not (touched and above_pbema and has_rejection):
            return _ret(None, None, None, None, "No Valid SUPPORT Rejection")

        # Optional trend alignment: LONG only if above baseline (uptrend)
        if require_trend_alignment and baseline is not None:
            if close < baseline:
                return _ret(None, None, None, None, "Counter-trend LONG (below baseline)")

        entry = close
        tp = entry * (1 + tp_percentage)

        if use_atr_sl and "atr" in df.columns:
            atr = float(curr.get("atr", 0))
            if atr > 0:
                sl = entry - (atr * atr_sl_multiplier)
            else:
                sl = pb_bot * (1 - sl_buffer)
        else:
            sl = pb_bot * (1 - sl_buffer)

        sl_dist = (entry - sl) / entry
        if sl_dist < min_sl_distance:
            sl = entry * (1 - min_sl_distance)

        if tp <= entry or sl >= entry:
            return _ret(None, None, None, None, "Invalid TP/SL")

        risk = entry - sl
        reward = tp - entry
        if risk <= 0 or reward <= 0:
            return _ret(None, None, None, None, "Invalid RR")

        rr = reward / risk
        if rr < min_rr:
            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

        reason = f"PBEMA_SUPPORT_LONG(R:{rr:.2f},Rej:{prior_rejections})"
        return _ret("LONG", entry, tp, sl, reason)

    return _ret(None, None, None, None, "No Signal")
