# strategies/pbema_retest.py
# PBEMA Retest Strategy - Trade PBEMA as support/resistance after breakout
#
# Core Concept:
# "PBEMA bulutundan güçlü fiyat sekmesi yaşanabilir!"
#
# Strategy Logic:
# After PBEMA is broken (price crosses above/below), enter on the RETEST
# of PBEMA as support/resistance.
#
# Entry Patterns:
# 1. LONG: Price broke above PBEMA → retest PBEMA as support → bounce → entry
# 2. SHORT: Price broke below PBEMA → retest PBEMA as resistance → rejection → entry
#
# Key Components:
# 1. PBEMA Breakout Detection: Price crossed PBEMA cloud in last N candles
# 2. Retest Confirmation: Price returns to touch PBEMA (within tolerance)
# 3. Bounce/Rejection: Wick rejection shows PBEMA is holding
# 4. Momentum Validation: Optional AlphaTrend confirmation
#
# Examples from Real Trades:
# - NO 7: "PBEMADAN güçlü fiyat sekmesi yaşanabileceği icin long entry"
# - NO 11: "Fiyat PBEMA bandını kazanıyor ve retest ediyor, entry aldım"
# - NO 12-13: "PBEMA retestinde entry aldım"
# - NO 18: "Fiyat PBEMA üzerinde yer edinmiş, bir çok kez retest edip entry aldım"

from typing import Tuple, Union, Dict, Optional
import pandas as pd
import numpy as np

from .base import SignalResult, SignalResultWithDebug


def check_pbema_retest_signal(
        df: pd.DataFrame,
        index: int = -2,
        min_rr: float = 1.0,  # Lowered from 1.5 - more signals
        # === PBEMA Breakout Detection ===
        breakout_lookback: int = 30,  # Increased from 20 - more breakout detection time
        min_breakout_strength: float = 0.002,  # Lowered from 0.5% to 0.2% - more realistic
        # === Retest Detection ===
        retest_tolerance: float = 0.003,  # Tolerance for PBEMA touch (0.3%)
        min_wick_ratio: float = 0.15,  # Minimum wick for rejection (15% of candle)
        # === TP/SL Configuration ===
        tp_target: str = "baseline",  # "baseline" or "percentage"
        tp_percentage: float = 0.015,  # TP % if using percentage mode (1.5%)
        sl_buffer: float = 0.003,  # SL buffer beyond PBEMA (0.3%)
        min_sl_distance: float = 0.015,  # Minimum SL distance (1.5%)
        use_atr_sl: bool = True,  # Use ATR-based SL instead of fixed buffer
        atr_sl_multiplier: float = 1.5,  # ATR multiplier for SL
        # === Optional Filters ===
        require_at_confirmation: bool = False,  # Require AlphaTrend confirmation
        require_multiple_retests: bool = False,  # Require 2+ retests (stronger setup)
        min_retests: int = 2,  # Minimum retest count if required
        # === Advanced ===
        use_volume_confirmation: bool = False,  # Volume spike on breakout (if available)
        timeframe: str = "15m",  # Timeframe for TF-adaptive thresholds
        return_debug: bool = False,
) -> Union[SignalResult, SignalResultWithDebug]:
    """
    PBEMA Retest Strategy - Enter on PBEMA support/resistance retest after breakout.

    This strategy trades PBEMA cloud as a dynamic support/resistance level.
    After price breaks through PBEMA, it often returns to retest the level.
    A successful retest (bounce/rejection) provides a high-probability entry.

    Signal Flow:
    1. Detect PBEMA breakout in recent history (price crossed PBEMA cloud)
    2. Check if price is currently retesting PBEMA (touching cloud boundary)
    3. Confirm bounce/rejection via wick analysis
    4. Optional: Validate with AlphaTrend, multiple retests, or volume
    5. Set TP (baseline or percentage) and SL (beyond PBEMA)

    Args:
        df: DataFrame with OHLCV + indicators
        index: Candle index for signal check (default: -2)
        min_rr: Minimum risk/reward ratio
        breakout_lookback: Candles to search for PBEMA breakout
        min_breakout_strength: Minimum breakout distance beyond PBEMA
        retest_tolerance: Touch tolerance for PBEMA retest detection
        min_wick_ratio: Minimum wick size for rejection confirmation
        tp_target: TP mode - "baseline" (SSL) or "percentage" (fixed %)
        tp_percentage: TP percentage if using percentage mode
        sl_buffer: SL buffer distance beyond PBEMA
        require_at_confirmation: Require AlphaTrend to confirm direction
        require_multiple_retests: Require multiple successful retests
        min_retests: Minimum retest count if required
        use_volume_confirmation: Require volume spike on breakout
        timeframe: Timeframe for adaptive parameters
        return_debug: Return debug info

    Returns:
        SignalResult: (signal_type, entry, tp, sl, reason)
        SignalResultWithDebug: Above + debug_info dict (if return_debug=True)

    Example:
        >>> signal_type, entry, tp, sl, reason = check_pbema_retest_signal(df, index=-2)
        >>> if signal_type == "LONG":
        ...     print(f"PBEMA retest LONG at {entry}, TP {tp}, SL {sl}")
    """

    debug_info = {
        "breakout_detected": None,
        "breakout_direction": None,
        "breakout_candle_idx": None,
        "breakout_distance": None,
        "currently_retesting": None,
        "retest_count": 0,
        "wick_rejection": None,
        "wick_ratio": None,
        "at_confirmed": None,
        "volume_confirmed": None,
        "timeframe": timeframe,
    }

    def _ret(s_type, entry, tp, sl, reason):
        if return_debug:
            return s_type, entry, tp, sl, reason, debug_info
        return s_type, entry, tp, sl, reason

    # === Validate Input ===
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
    if abs_index < breakout_lookback + 10:
        return _ret(None, None, None, None, "Not Enough Data")

    # === Extract Current Values ===
    open_ = float(curr["open"])
    high = float(curr["high"])
    low = float(curr["low"])
    close = float(curr["close"])
    pb_top = float(curr["pb_ema_top"])
    pb_bot = float(curr["pb_ema_bot"])
    pb_mid = (pb_top + pb_bot) / 2

    # Check for NaN
    if any(pd.isna([open_, high, low, close, pb_top, pb_bot])):
        return _ret(None, None, None, None, "NaN Values")

    # === Optional: Get Baseline for TP ===
    baseline = None
    if tp_target == "baseline" and "baseline" in df.columns:
        baseline = float(curr["baseline"])
        if pd.isna(baseline):
            baseline = None

    # === STEP 1: Detect PBEMA Breakout in Recent History ===
    # Search for a candle where price crossed PBEMA cloud

    pb_broken_above = False
    pb_broken_below = False
    breakout_candle_idx = None
    breakout_distance = 0.0

    for i in range(abs_index - breakout_lookback, abs_index):
        if i < 10:
            continue

        candle = df.iloc[i]
        candle_close = float(candle["close"])
        candle_pb_top = float(candle["pb_ema_top"])
        candle_pb_bot = float(candle["pb_ema_bot"])

        # Check previous candle to detect crossover
        if i > 0:
            prev_candle = df.iloc[i - 1]
            prev_close = float(prev_candle["close"])
            prev_pb_top = float(prev_candle["pb_ema_top"])
            prev_pb_bot = float(prev_candle["pb_ema_bot"])

            # === BULLISH BREAKOUT: Price crossed above PBEMA top ===
            # OLD (too strict): prev_close < prev_pb_bot and candle_close > candle_pb_top
            # NEW (realistic): prev_close was not above cloud, now above cloud top
            if prev_close <= prev_pb_top and candle_close > candle_pb_top:
                # Check breakout strength (distance moved beyond PBEMA)
                breakout_dist = (candle_close - candle_pb_top) / candle_pb_top
                if breakout_dist >= min_breakout_strength:
                    pb_broken_above = True
                    breakout_candle_idx = i
                    breakout_distance = breakout_dist
                    break  # Use first (earliest) breakout

            # === BEARISH BREAKOUT: Price crossed below PBEMA bottom ===
            # OLD (too strict): prev_close > prev_pb_top and candle_close < candle_pb_bot
            # NEW (realistic): prev_close was not below cloud, now below cloud bottom
            if prev_close >= prev_pb_bot and candle_close < candle_pb_bot:
                # Check breakout strength
                breakout_dist = (candle_pb_bot - candle_close) / candle_pb_bot
                if breakout_dist >= min_breakout_strength:
                    pb_broken_below = True
                    breakout_candle_idx = i
                    breakout_distance = breakout_dist
                    break  # Use first (earliest) breakout

    debug_info["breakout_detected"] = pb_broken_above or pb_broken_below
    debug_info["breakout_candle_idx"] = breakout_candle_idx
    debug_info["breakout_distance"] = breakout_distance

    if pb_broken_above:
        debug_info["breakout_direction"] = "BULLISH"
    elif pb_broken_below:
        debug_info["breakout_direction"] = "BEARISH"

    # No breakout detected - no retest opportunity
    if not (pb_broken_above or pb_broken_below):
        return _ret(None, None, None, None, "No PBEMA Breakout Detected")

    # === STEP 2: Check if Currently Retesting PBEMA ===

    currently_retesting_long = False
    currently_retesting_short = False

    # LONG Setup: Price broke above PBEMA, now retesting from above (support test)
    if pb_broken_above:
        # Price should be touching PBEMA top from above
        touching_pbema = low <= pb_top * (1 + retest_tolerance)
        price_still_above = close > pb_mid  # Price should still be above cloud mid

        currently_retesting_long = touching_pbema and price_still_above

    # SHORT Setup: Price broke below PBEMA, now retesting from below (resistance test)
    if pb_broken_below:
        # Price should be touching PBEMA bottom from below
        touching_pbema = high >= pb_bot * (1 - retest_tolerance)
        price_still_below = close < pb_mid  # Price should still be below cloud mid

        currently_retesting_short = touching_pbema and price_still_below

    debug_info["currently_retesting"] = currently_retesting_long or currently_retesting_short

    if not (currently_retesting_long or currently_retesting_short):
        return _ret(None, None, None, None, "Not Retesting PBEMA")

    # === STEP 3: Confirm Bounce/Rejection via Wick Analysis ===

    candle_range = high - low
    if candle_range <= 0:
        return _ret(None, None, None, None, "Zero Range Candle")

    # Calculate wicks
    candle_body_high = max(open_, close)
    candle_body_low = min(open_, close)
    upper_wick = high - candle_body_high
    lower_wick = candle_body_low - low

    upper_wick_ratio = upper_wick / candle_range
    lower_wick_ratio = lower_wick / candle_range

    debug_info["upper_wick_ratio"] = upper_wick_ratio
    debug_info["lower_wick_ratio"] = lower_wick_ratio

    # LONG: Need lower wick rejection (bounce from PBEMA support)
    long_rejection = lower_wick_ratio >= min_wick_ratio

    # SHORT: Need upper wick rejection (rejection from PBEMA resistance)
    short_rejection = upper_wick_ratio >= min_wick_ratio

    debug_info["wick_rejection"] = long_rejection if currently_retesting_long else short_rejection
    debug_info["wick_ratio"] = lower_wick_ratio if currently_retesting_long else upper_wick_ratio

    # === STEP 4: Optional - Count Multiple Retests ===

    if require_multiple_retests:
        retest_count = 0

        # Count how many times price retested PBEMA since breakout
        for i in range(breakout_candle_idx + 1, abs_index + 1):
            candle = df.iloc[i]
            candle_low = float(candle["low"])
            candle_high = float(candle["high"])
            candle_pb_top = float(candle["pb_ema_top"])
            candle_pb_bot = float(candle["pb_ema_bot"])

            if pb_broken_above and candle_low <= candle_pb_top * (1 + retest_tolerance):
                retest_count += 1
            elif pb_broken_below and candle_high >= candle_pb_bot * (1 - retest_tolerance):
                retest_count += 1

        debug_info["retest_count"] = retest_count

        if retest_count < min_retests:
            return _ret(None, None, None, None,
                       f"Insufficient Retests ({retest_count}/{min_retests})")

    # === STEP 5: Optional - AlphaTrend Confirmation ===

    at_confirmed_long = True
    at_confirmed_short = True

    if require_at_confirmation:
        required_at_cols = ['at_buyers_dominant', 'at_sellers_dominant']
        has_at_cols = all(col in df.columns for col in required_at_cols)

        if not has_at_cols:
            return _ret(None, None, None, None, "AlphaTrend columns missing")

        at_buyers_dominant = bool(curr.get("at_buyers_dominant", False))
        at_sellers_dominant = bool(curr.get("at_sellers_dominant", False))

        at_confirmed_long = at_buyers_dominant
        at_confirmed_short = at_sellers_dominant

        debug_info["at_confirmed"] = at_confirmed_long if currently_retesting_long else at_confirmed_short

    # === STEP 6: Optional - Volume Confirmation ===

    if use_volume_confirmation and "volume" in df.columns:
        # Check if breakout candle had volume spike
        if breakout_candle_idx is not None:
            breakout_volume = float(df.iloc[breakout_candle_idx]["volume"])
            avg_volume = float(df["volume"].iloc[breakout_candle_idx - 20:breakout_candle_idx].mean())

            volume_spike = breakout_volume > avg_volume * 1.5  # 50% above average
            debug_info["volume_confirmed"] = volume_spike

            if not volume_spike:
                return _ret(None, None, None, None, "No Volume Confirmation")

    # === EXECUTE LONG SIGNAL ===

    if currently_retesting_long and long_rejection and at_confirmed_long:
        entry = close

        # === TP: Baseline or Percentage ===
        if tp_target == "baseline" and baseline is not None and baseline > close:
            tp = baseline
        else:
            # Percentage TP
            tp = close * (1 + tp_percentage)

        # === SL: ATR-based or PBEMA-based with minimum distance ===
        if use_atr_sl and 'atr' in df.columns:
            atr = float(curr.get('atr', 0))
            if atr > 0:
                sl = entry - (atr * atr_sl_multiplier)
            else:
                sl = pb_bot * (1 - sl_buffer)
        else:
            sl = pb_bot * (1 - sl_buffer)

        # Ensure minimum SL distance
        sl_distance_pct = (entry - sl) / entry
        if sl_distance_pct < min_sl_distance:
            sl = entry * (1 - min_sl_distance)

        # Validate TP/SL
        if tp <= entry:
            return _ret(None, None, None, None, "TP Below Entry (LONG)")
        if sl >= entry:
            sl = entry * (1 - min_sl_distance)  # Force minimum SL

        risk = entry - sl
        reward = tp - entry

        if risk <= 0 or reward <= 0:
            return _ret(None, None, None, None, "Invalid RR (LONG)")

        rr = reward / risk

        if rr < min_rr:
            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

        reason = f"PBEMA_RETEST_LONG(R:{rr:.2f})"
        return _ret("LONG", entry, tp, sl, reason)

    # === EXECUTE SHORT SIGNAL ===

    if currently_retesting_short and short_rejection and at_confirmed_short:
        entry = close

        # === TP: Baseline or Percentage ===
        if tp_target == "baseline" and baseline is not None and baseline < close:
            tp = baseline
        else:
            # Percentage TP
            tp = close * (1 - tp_percentage)

        # === SL: ATR-based or PBEMA-based with minimum distance ===
        if use_atr_sl and 'atr' in df.columns:
            atr = float(curr.get('atr', 0))
            if atr > 0:
                sl = entry + (atr * atr_sl_multiplier)
            else:
                sl = pb_top * (1 + sl_buffer)
        else:
            sl = pb_top * (1 + sl_buffer)

        # Ensure minimum SL distance
        sl_distance_pct = (sl - entry) / entry
        if sl_distance_pct < min_sl_distance:
            sl = entry * (1 + min_sl_distance)

        # Validate TP/SL
        if tp >= entry:
            return _ret(None, None, None, None, "TP Above Entry (SHORT)")
        if sl <= entry:
            sl = entry * (1 + min_sl_distance)  # Force minimum SL

        risk = sl - entry
        reward = entry - tp

        if risk <= 0 or reward <= 0:
            return _ret(None, None, None, None, "Invalid RR (SHORT)")

        rr = reward / risk

        if rr < min_rr:
            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

        reason = f"PBEMA_RETEST_SHORT(R:{rr:.2f})"
        return _ret("SHORT", entry, tp, sl, reason)

    # === No Signal ===

    if currently_retesting_long and not long_rejection:
        return _ret(None, None, None, None, "LONG: No Wick Rejection")
    if currently_retesting_short and not short_rejection:
        return _ret(None, None, None, None, "SHORT: No Wick Rejection")
    if currently_retesting_long and not at_confirmed_long:
        return _ret(None, None, None, None, "LONG: AT Not Confirmed")
    if currently_retesting_short and not at_confirmed_short:
        return _ret(None, None, None, None, "SHORT: AT Not Confirmed")

    return _ret(None, None, None, None, "No Signal")
