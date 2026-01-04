# core/momentum_exit.py
# Momentum Exhaustion Exit Detection
#
# Pattern 1: Exit trades when momentum slows down instead of fixed TP
#
# Core Concept:
# "Momentum yavaşlayan dek trade tutup TP oluyoruz" - Hold until momentum slows
#
# Real Trade Evidence:
# - NO 7: "momentum yavaşlayan dek takip edip TP oluyoruz"
# - NO 9: "momentum azalana dek fiyatı takip edip TP aldım"
# - NO 12: "momentum bitene dek trade devam ediyor"
# - NO 15: "momentum yavaşlayan dek trade tutup tp oluyorum"
# - NO 18: "momentum yavaşlayan dek trade tuttum"
#
# Detection Methods:
# 1. AlphaTrend slope flattening (buyers/sellers line losing angle)
# 2. Candle range shrinking (last N candles < average range)
# 3. ATR deviation (price moving < threshold * ATR per candle)
# 4. Volume decline (optional - if volume data available)

from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np


def detect_momentum_exhaustion(
        df: pd.DataFrame,
        index: int,
        signal_type: str,
        lookback: int = 3,
        slope_threshold: float = 0.5,
        range_threshold: float = 0.7,
        atr_threshold: float = 0.5,
        min_conditions: int = 2,
        return_debug: bool = False,
) -> Tuple[bool, Optional[Dict]]:
    """
    Detect momentum exhaustion for dynamic TP exit.

    Returns True when momentum is slowing down (time to exit trade).

    This function is called during an ACTIVE trade to determine if momentum
    has exhausted and the trade should be closed.

    Args:
        df: DataFrame with OHLCV + indicators
        index: Current candle index
        signal_type: "LONG" or "SHORT" (direction of active trade)
        lookback: Candles to analyze for momentum change (default: 3)
        slope_threshold: AlphaTrend slope threshold (0-1, lower=stricter)
        range_threshold: Candle range threshold (0-1, lower=stricter)
        atr_threshold: ATR movement threshold (0-1, lower=stricter)
        min_conditions: Minimum conditions to trigger (default: 2 of 3)
        return_debug: Return debug information

    Returns:
        (exhausted: bool, debug_info: dict)
        - exhausted: True if momentum has slowed (exit recommended)
        - debug_info: Diagnostic information (if return_debug=True)

    Criteria for Momentum Exhaustion (2+ must be met):
    1. AlphaTrend slope flattening (line angle decreasing)
    2. Candle ranges shrinking (recent < historical average)
    3. ATR deviation low (price barely moving relative to volatility)

    Example:
        >>> exhausted, debug = detect_momentum_exhaustion(df, -1, "LONG", return_debug=True)
        >>> if exhausted:
        ...     print("Exit trade - momentum has slowed")
        ...     print(f"Conditions met: {debug['conditions_met']}")
    """

    debug_info = {
        "slope_flattening": False,
        "range_shrinking": False,
        "low_movement": False,
        "conditions_met": 0,
        "slope_ratio": None,
        "range_ratio": None,
        "atr_ratio": None,
    }

    # Validate inputs
    abs_idx = index if index >= 0 else len(df) + index

    if abs_idx < lookback + 10:
        if return_debug:
            return False, debug_info
        return False, None

    required_cols = ['close', 'high', 'low', 'atr']
    if signal_type in ["LONG", "SHORT"]:
        # Need AlphaTrend for slope analysis
        if signal_type == "LONG":
            required_cols.append('alphatrend')  # Buyers line
        else:
            required_cols.append('alphatrend_2')  # Sellers line

    for col in required_cols:
        if col not in df.columns:
            if return_debug:
                return False, debug_info
            return False, None

    # === CRITERION 1: AlphaTrend Slope Flattening ===
    # Check if the winning line (buyers for LONG, sellers for SHORT) is losing angle

    try:
        if signal_type == "LONG":
            at_line = df['alphatrend'].values
        else:
            at_line = df['alphatrend_2'].values

        # Get recent AlphaTrend values
        recent_at = at_line[abs_idx - lookback:abs_idx + 1]

        # Calculate percentage changes
        at_changes = np.diff(recent_at) / recent_at[:-1]

        # Check if most recent change is significantly less than average
        if len(at_changes) > 1:
            recent_change = abs(at_changes[-1])
            avg_change = np.mean(np.abs(at_changes[:-1]))

            slope_ratio = recent_change / avg_change if avg_change > 0 else 1.0
            debug_info["slope_ratio"] = slope_ratio

            # Slope is flattening if recent change < threshold * average change
            slope_flattening = slope_ratio < slope_threshold
            debug_info["slope_flattening"] = slope_flattening
        else:
            slope_flattening = False

    except (IndexError, ValueError, ZeroDivisionError):
        slope_flattening = False

    # === CRITERION 2: Candle Range Shrinking ===
    # Check if recent candles are smaller than historical average

    try:
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        # Recent candle ranges
        recent_ranges = highs[abs_idx - lookback:abs_idx + 1] - lows[abs_idx - lookback:abs_idx + 1]
        recent_avg_range = np.mean(recent_ranges)

        # Historical average (before recent period)
        hist_start = max(0, abs_idx - lookback * 3)
        hist_end = abs_idx - lookback
        hist_ranges = highs[hist_start:hist_end] - lows[hist_start:hist_end]
        hist_avg_range = np.mean(hist_ranges)

        range_ratio = recent_avg_range / hist_avg_range if hist_avg_range > 0 else 1.0
        debug_info["range_ratio"] = range_ratio

        # Range is shrinking if recent average < threshold * historical average
        range_shrinking = range_ratio < range_threshold
        debug_info["range_shrinking"] = range_shrinking

    except (IndexError, ValueError, ZeroDivisionError):
        range_shrinking = False

    # === CRITERION 3: ATR Deviation (Low Movement) ===
    # Check if price is barely moving relative to volatility

    try:
        atr_values = df['atr'].values
        current_atr = atr_values[abs_idx]

        # Calculate recent price movement
        recent_moves = np.abs(np.diff(closes[abs_idx - lookback:abs_idx + 1]))
        avg_move = np.mean(recent_moves)

        atr_ratio = avg_move / current_atr if current_atr > 0 else 1.0
        debug_info["atr_ratio"] = atr_ratio

        # Movement is low if average move < threshold * ATR
        low_movement = atr_ratio < atr_threshold
        debug_info["low_movement"] = low_movement

    except (IndexError, ValueError, ZeroDivisionError):
        low_movement = False

    # === DECISION: Momentum Exhausted? ===
    # Require min_conditions (default: 2 of 3) to be met

    conditions_met = sum([slope_flattening, range_shrinking, low_movement])
    debug_info["conditions_met"] = conditions_met

    exhausted = conditions_met >= min_conditions

    if return_debug:
        return exhausted, debug_info
    return exhausted, None


def calculate_dynamic_tp_from_momentum(
        df: pd.DataFrame,
        entry_price: float,
        signal_type: str,
        original_tp: float,
        current_index: int,
        min_profit_pct: float = 0.005,
        max_extension: float = 3.0,
) -> float:
    """
    Calculate dynamic TP based on momentum continuation.

    This function extends TP when momentum is strong, tightens when weak.

    Args:
        df: DataFrame with indicators
        entry_price: Entry price of trade
        signal_type: "LONG" or "SHORT"
        original_tp: Original TP target (e.g., PBEMA)
        current_index: Current candle index
        min_profit_pct: Minimum profit % before checking momentum (0.5%)
        max_extension: Maximum TP extension multiplier (3.0x original distance)

    Returns:
        new_tp: Adjusted TP based on momentum

    Logic:
        - If profitable and momentum strong: Extend TP
        - If momentum exhausted: Lock in current price (trailing stop)
        - If not profitable yet: Keep original TP
    """

    abs_idx = current_index if current_index >= 0 else len(df) + current_index

    try:
        current_price = float(df.iloc[abs_idx]['close'])
    except (IndexError, KeyError):
        return original_tp

    # Calculate current profit
    if signal_type == "LONG":
        profit_pct = (current_price - entry_price) / entry_price
        original_distance = original_tp - entry_price
    else:
        profit_pct = (entry_price - current_price) / entry_price
        original_distance = entry_price - original_tp

    # If not profitable yet, keep original TP
    if profit_pct < min_profit_pct:
        return original_tp

    # Check if momentum is exhausted
    exhausted, _ = detect_momentum_exhaustion(df, abs_idx, signal_type)

    if exhausted:
        # Momentum slowing - lock in current price (exit signal)
        # Return current price as "TP" to trigger exit
        return current_price

    # Momentum still strong - extend TP slightly
    # (Conservative: don't extend too far, original TP was chosen for a reason)
    extension_multiplier = 1.2  # Extend by 20%
    max_distance = original_distance * max_extension

    if signal_type == "LONG":
        extended_tp = entry_price + (original_distance * extension_multiplier)
        # Cap at max extension
        extended_tp = min(extended_tp, entry_price + max_distance)
        return extended_tp
    else:
        extended_tp = entry_price - (original_distance * extension_multiplier)
        # Cap at max extension
        extended_tp = max(extended_tp, entry_price - max_distance)
        return extended_tp


# Convenience function for integration
def should_exit_on_momentum(
        df: pd.DataFrame,
        index: int,
        signal_type: str,
        lenient: bool = False,
) -> bool:
    """
    Simple yes/no check: Should we exit this trade due to momentum exhaustion?

    Args:
        df: DataFrame with indicators
        index: Current candle index
        signal_type: "LONG" or "SHORT"
        lenient: If True, require only 1 condition instead of 2 (more exits)

    Returns:
        True if should exit, False otherwise

    Example:
        >>> if should_exit_on_momentum(df, -1, "LONG"):
        ...     # Close trade at market
        ...     pass
    """
    min_conditions = 1 if lenient else 2
    exhausted, _ = detect_momentum_exhaustion(
        df, index, signal_type, min_conditions=min_conditions
    )
    return exhausted
