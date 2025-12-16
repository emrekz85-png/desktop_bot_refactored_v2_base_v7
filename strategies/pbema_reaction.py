# strategies/pbema_reaction.py
# PBEMA Reaction Strategy - Trade when price approaches/touches PBEMA cloud
#
# Concept:
# - PBEMA cloud acts as strong support/resistance
# - Expect reaction when price approaches PBEMA
# - SHORT: Price approaches PBEMA from below -> sell pressure expected
# - LONG: Price approaches PBEMA from above -> buy pressure expected
#
# Uses EMA150 instead of EMA200 for PBEMA cloud.

from typing import Tuple, Union
import pandas as pd

from .base import SignalResult, SignalResultWithDebug


def check_pbema_reaction_signal(
        df: pd.DataFrame,
        index: int = -2,
        min_rr: float = 2.0,
        rsi_limit: float = 60.0,
        slope_thresh: float = 0.5,
        use_alphatrend: bool = False,
        pbema_approach_tolerance: float = 0.003,
        pbema_frontrun_margin: float = 0.002,
        tp_min_dist_ratio: float = 0.0015,
        tp_max_dist_ratio: float = 0.04,
        adx_min: float = 12.0,
        return_debug: bool = False,
) -> Union[SignalResult, SignalResultWithDebug]:
    """
    PBEMA Reaction Strategy - Trade when price approaches/touches PBEMA cloud.

    Args:
        df: DataFrame with OHLCV + indicators (rsi, adx, pb_ema_top_150, pb_ema_bot_150, keltner_upper, keltner_lower)
        index: Candle index for signal check (default: -2)
        min_rr: Minimum risk/reward ratio
        rsi_limit: RSI threshold for filtering
        slope_thresh: Slope threshold for direction filtering
        use_alphatrend: Whether to use AlphaTrend filter
        pbema_approach_tolerance: How close to PBEMA to generate signal (e.g., 0.003 = 0.3%)
        pbema_frontrun_margin: Frontrun margin (SL = PBEMA + this margin)
        tp_min_dist_ratio: Minimum TP distance ratio
        tp_max_dist_ratio: Maximum TP distance ratio
        adx_min: Minimum ADX value
        return_debug: Whether to return debug info

    Returns:
        SignalResult or SignalResultWithDebug tuple
    """

    debug_info = {
        "adx_ok": None,
        "price_near_pbema_top": None,
        "price_near_pbema_bot": None,
        "approaching_from_below": None,
        "approaching_from_above": None,
        "short_rsi_ok": None,
        "long_rsi_ok": None,
        "tp_dist_ratio": None,
        "rr_value": None,
        "min_wick_ratio_pbema": None,
        "upper_wick_ratio": None,
        "lower_wick_ratio": None,
        "wick_quality_short": None,
        "wick_quality_long": None,
    }

    def _ret(s_type, entry, tp, sl, reason):
        if return_debug:
            return s_type, entry, tp, sl, reason, debug_info
        return s_type, entry, tp, sl, reason

    if df is None or df.empty:
        return _ret(None, None, None, None, "No Data")

    required_cols = [
        "open", "high", "low", "close",
        "rsi", "adx",
        "pb_ema_top_150", "pb_ema_bot_150",
        "keltner_upper", "keltner_lower",
    ]
    for col in required_cols:
        if col not in df.columns:
            return _ret(None, None, None, None, f"Missing {col}")

    try:
        curr = df.iloc[index]
    except Exception:
        return _ret(None, None, None, None, "Index Error")

    abs_index = index if index >= 0 else (len(df) + index)
    if abs_index < 30:  # Need enough history for swing detection
        return _ret(None, None, None, None, "Not Enough Data")

    # Extract current values
    open_ = float(curr["open"])
    high = float(curr["high"])
    low = float(curr["low"])
    close = float(curr["close"])
    pb_top = float(curr["pb_ema_top_150"])
    pb_bot = float(curr["pb_ema_bot_150"])
    lower_band = float(curr["keltner_lower"])
    upper_band = float(curr["keltner_upper"])
    adx_val = float(curr["adx"])
    rsi_val = float(curr["rsi"])

    # Check for NaN values
    if any(pd.isna([open_, high, low, close, pb_top, pb_bot, lower_band, upper_band, adx_val, rsi_val])):
        return _ret(None, None, None, None, "NaN Values")

    # Wick-quality filter calculations
    min_wick_ratio_pbema = 0.12
    candle_range = high - low
    if candle_range <= 0:
        upper_wick_ratio = 0.0
        lower_wick_ratio = 0.0
    else:
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range

    debug_info["min_wick_ratio_pbema"] = min_wick_ratio_pbema
    debug_info["upper_wick_ratio"] = upper_wick_ratio
    debug_info["lower_wick_ratio"] = lower_wick_ratio

    # ADX filter
    adx_ok = adx_val >= adx_min
    debug_info["adx_ok"] = adx_ok
    if not adx_ok:
        return _ret(None, None, None, None, f"ADX Too Low ({adx_val:.1f})")

    # Calculate distances to PBEMA cloud
    dist_to_pb_top = abs(high - pb_top) / pb_top if pb_top > 0 else 1.0
    dist_to_pb_bot = abs(low - pb_bot) / pb_bot if pb_bot > 0 else 1.0

    # Check if price is approaching PBEMA from below (for SHORT)
    price_below_pbema = close < pb_bot
    price_near_pbema_top = (high >= pb_top * (1 - pbema_approach_tolerance)) and (high <= pb_top * (1 + pbema_frontrun_margin))
    approaching_from_below = (
        not price_below_pbema and
        (dist_to_pb_top <= pbema_approach_tolerance or high >= pb_top)
    )

    # Check if price is approaching PBEMA from above (for LONG)
    price_above_pbema = close > pb_top
    price_near_pbema_bot = (low <= pb_bot * (1 + pbema_approach_tolerance)) and (low >= pb_bot * (1 - pbema_frontrun_margin))
    approaching_from_above = (
        not price_above_pbema and
        (dist_to_pb_bot <= pbema_approach_tolerance or low <= pb_bot)
    )

    debug_info.update({
        "price_near_pbema_top": price_near_pbema_top,
        "price_near_pbema_bot": price_near_pbema_bot,
        "approaching_from_below": approaching_from_below,
        "approaching_from_above": approaching_from_above,
    })

    # ================= SHORT (PBEMA Resistance) =================
    is_short = price_near_pbema_top and close < pb_top

    # Rejection candle check
    if is_short:
        rejection_wick_short = (high >= pb_top * (1 - pbema_approach_tolerance)) and (close < pb_top)
        candle_body_below = max(open_, close) < pb_top
        wick_quality_short = (upper_wick_ratio >= min_wick_ratio_pbema)
        debug_info["wick_quality_short"] = wick_quality_short
        is_short = rejection_wick_short and candle_body_below and wick_quality_short

    # RSI filter for SHORT
    short_rsi_limit = 100.0 - (rsi_limit + 10.0)
    short_rsi_ok = rsi_val >= short_rsi_limit
    debug_info["short_rsi_ok"] = short_rsi_ok
    if is_short and not short_rsi_ok:
        is_short = False

    # Slope filter
    if is_short and "slope_top_150" in curr.index:
        slope_val = float(curr["slope_top_150"])
        if slope_val > slope_thresh:
            is_short = False

    if is_short:
        swing_n = 20
        start = max(0, abs_index - swing_n)
        swing_low = float(df["low"].iloc[start:abs_index].min())

        tp = swing_low * 0.998
        entry = close
        sl = pb_top * (1 + pbema_frontrun_margin + 0.002)

        if tp >= entry:
            return _ret(None, None, None, None, "TP Above Entry (SHORT)")
        if sl <= entry:
            sl = pb_top * (1 + pbema_frontrun_margin + 0.005)

        risk = sl - entry
        reward = entry - tp
        if risk <= 0 or reward <= 0:
            return _ret(None, None, None, None, "Invalid RR (SHORT)")

        rr = reward / risk
        tp_dist_ratio = reward / entry

        debug_info.update({
            "tp_dist_ratio": tp_dist_ratio,
            "rr_value": rr,
        })

        if tp_dist_ratio < tp_min_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
        if tp_dist_ratio > tp_max_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
        if rr < min_rr:
            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

        reason = f"ACCEPTED(PBEMA_Reaction,R:{rr:.2f})"
        return _ret("SHORT", entry, tp, sl, reason)

    # ================= LONG (PBEMA Support) =================
    is_long = price_near_pbema_bot and close > pb_bot

    # Rejection candle check
    if is_long:
        rejection_wick_long = (low <= pb_bot * (1 + pbema_approach_tolerance)) and (close > pb_bot)
        candle_body_above = min(open_, close) > pb_bot
        wick_quality_long = (lower_wick_ratio >= min_wick_ratio_pbema)
        debug_info["wick_quality_long"] = wick_quality_long
        is_long = rejection_wick_long and candle_body_above and wick_quality_long

    # RSI filter for LONG
    long_rsi_limit = rsi_limit + 10.0
    long_rsi_ok = rsi_val <= long_rsi_limit
    debug_info["long_rsi_ok"] = long_rsi_ok
    if is_long and not long_rsi_ok:
        is_long = False

    # Slope filter
    if is_long and "slope_bot_150" in curr.index:
        slope_val = float(curr["slope_bot_150"])
        if slope_val < -slope_thresh:
            is_long = False

    if is_long:
        swing_n = 20
        start = max(0, abs_index - swing_n)
        swing_high = float(df["high"].iloc[start:abs_index].max())

        tp = swing_high * 1.002
        entry = close
        sl = pb_bot * (1 - pbema_frontrun_margin - 0.002)

        if tp <= entry:
            return _ret(None, None, None, None, "TP Below Entry (LONG)")
        if sl >= entry:
            sl = pb_bot * (1 - pbema_frontrun_margin - 0.005)

        risk = entry - sl
        reward = tp - entry
        if risk <= 0 or reward <= 0:
            return _ret(None, None, None, None, "Invalid RR (LONG)")

        rr = reward / risk
        tp_dist_ratio = reward / entry

        debug_info.update({
            "tp_dist_ratio": tp_dist_ratio,
            "rr_value": rr,
        })

        if tp_dist_ratio < tp_min_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
        if tp_dist_ratio > tp_max_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
        if rr < min_rr:
            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

        reason = f"ACCEPTED(PBEMA_Reaction,R:{rr:.2f})"
        return _ret("LONG", entry, tp, sl, reason)

    return _ret(None, None, None, None, "No Signal")
