# strategies/keltner_bounce.py
# Keltner Bounce Strategy - Mean reversion using Keltner bands
#
# Entry: Price touches Keltner band and rejects
# Target: PBEMA cloud (EMA200)
# Stop Loss: Beyond Keltner band
#
# This is a mean reversion approach using PBEMA cloud as magnet;
# Keltner touches trigger both SHORT from top and LONG from bottom.

from typing import Tuple, Union
import pandas as pd

from .base import SignalResult, SignalResultWithDebug


def check_keltner_bounce_signal(
        df: pd.DataFrame,
        index: int = -2,
        min_rr: float = 2.0,
        rsi_limit: float = 60.0,
        slope_thresh: float = 0.5,
        use_alphatrend: bool = True,
        hold_n: int = 5,
        min_hold_frac: float = 0.8,
        pb_touch_tolerance: float = 0.0012,
        body_tolerance: float = 0.0015,
        cloud_keltner_gap_min: float = 0.003,
        tp_min_dist_ratio: float = 0.0015,
        tp_max_dist_ratio: float = 0.03,
        adx_min: float = 12.0,
        return_debug: bool = False,
) -> Union[SignalResult, SignalResultWithDebug]:
    """
    Keltner Bounce signal detection for LONG / SHORT.

    Filters:
    - ADX minimum threshold
    - Keltner holding + retest
    - PBEMA cloud alignment
    - Minimum distance between Keltner band and PBEMA TP target
    - TP not too close / too far
    - RR >= min_rr (RR = reward / risk)

    Args:
        df: DataFrame with OHLCV + indicators (rsi, adx, pb_ema_top, pb_ema_bot, keltner_upper, keltner_lower)
        index: Candle index for signal check (default: -2, second to last candle)
        min_rr: Minimum risk/reward ratio
        rsi_limit: RSI threshold for filtering
        slope_thresh: Slope threshold (currently disabled for mean reversion)
        use_alphatrend: Whether to use AlphaTrend filter
        hold_n: Number of candles for holding pattern
        min_hold_frac: Minimum fraction of candles holding pattern
        pb_touch_tolerance: Tolerance for Keltner band touch
        body_tolerance: Tolerance for candle body position
        cloud_keltner_gap_min: Minimum gap between Keltner and PBEMA
        tp_min_dist_ratio: Minimum TP distance ratio
        tp_max_dist_ratio: Maximum TP distance ratio
        adx_min: Minimum ADX value
        return_debug: Whether to return debug info

    Returns:
        SignalResult or SignalResultWithDebug tuple
    """

    debug_info = {
        "adx_ok": None,
        "trend_up_strong": None,
        "trend_down_strong": None,
        "holding_long": None,
        "retest_long": None,
        "pb_target_long": None,
        "long_rsi_ok": None,
        "holding_short": None,
        "retest_short": None,
        "pb_target_short": None,
        "short_rsi_ok": None,
        "tp_dist_ratio": None,
        "rr_value": None,
        "long_rr_ok": None,
        "short_rr_ok": None,
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
        "pb_ema_top", "pb_ema_bot",
        "keltner_upper", "keltner_lower",
    ]
    for col in required_cols:
        if col not in df.columns:
            return _ret(None, None, None, None, f"Missing {col}")

    try:
        curr = df.iloc[index]
    except (IndexError, KeyError):
        return _ret(None, None, None, None, "Index Error")

    for c in required_cols:
        v = curr.get(c)
        if pd.isna(v):
            return _ret(None, None, None, None, f"NaN in {c}")

    abs_index = index if index >= 0 else (len(df) + index)
    if abs_index < 0 or abs_index >= len(df):
        return _ret(None, None, None, None, "Index Out of Range")

    # --- Parameters ---
    hold_n = int(max(1, hold_n or 1))
    min_hold_frac = float(min_hold_frac if min_hold_frac is not None else 0.8)
    touch_tol = float(pb_touch_tolerance if pb_touch_tolerance is not None else 0.0012)
    body_tol = float(body_tolerance if body_tolerance is not None else 0.0015)
    cloud_keltner_gap_min = float(cloud_keltner_gap_min if cloud_keltner_gap_min is not None else 0.003)
    tp_min_dist_ratio = float(tp_min_dist_ratio if tp_min_dist_ratio is not None else 0.0015)
    tp_max_dist_ratio = float(tp_max_dist_ratio if tp_max_dist_ratio is not None else 0.03)
    adx_min = float(adx_min if adx_min is not None else 12.0)

    # ADX filter
    debug_info["adx_ok"] = float(curr["adx"]) >= adx_min
    if not debug_info["adx_ok"]:
        return _ret(None, None, None, None, "ADX Low")

    if abs_index < hold_n + 1:
        return _ret(None, None, None, None, "Warmup")

    slc = slice(abs_index - hold_n, abs_index)
    closes_slice = df["close"].iloc[slc]
    upper_slice = df["keltner_upper"].iloc[slc]
    lower_slice = df["keltner_lower"].iloc[slc]

    close = float(curr["close"])
    open_ = float(curr["open"])
    high = float(curr["high"])
    low = float(curr["low"])
    upper_band = float(curr["keltner_upper"])
    lower_band = float(curr["keltner_lower"])
    pb_top = float(curr["pb_ema_top"])
    pb_bot = float(curr["pb_ema_bot"])

    # --- Mean reversion: Slope filter DISABLED ---
    # PBEMA (200 EMA) moves too slowly - slope filter is wrong for mean reversion
    slope_top = float(curr.get("slope_top", 0.0) or 0.0)
    slope_bot = float(curr.get("slope_bot", 0.0) or 0.0)

    # Keep for debug, but NO filtering
    debug_info["slope_top"] = slope_top
    debug_info["slope_bot"] = slope_bot

    # Mean reversion = no direction restriction
    long_direction_ok = True
    short_direction_ok = True
    debug_info["long_direction_ok"] = long_direction_ok
    debug_info["short_direction_ok"] = short_direction_ok

    # ================= WICK REJECTION QUALITY =================
    candle_range = high - low
    if candle_range > 0:
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range
    else:
        upper_wick_ratio = 0.0
        lower_wick_ratio = 0.0

    min_wick_ratio = 0.15
    long_rejection_quality = lower_wick_ratio >= min_wick_ratio
    short_rejection_quality = upper_wick_ratio >= min_wick_ratio

    debug_info.update({
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "long_rejection_quality": long_rejection_quality,
        "short_rejection_quality": short_rejection_quality,
    })

    # ================= PRICE-PBEMA POSITION CHECK =================
    price_below_pbema = close < pb_bot
    price_above_pbema = close > pb_top

    debug_info.update({
        "price_below_pbema": price_below_pbema,
        "price_above_pbema": price_above_pbema,
    })

    # ================= KELTNER PENETRATION (TRAP) DETECTION =================
    penetration_lookback = min(3, len(df) - abs_index - 1) if abs_index < len(df) - 1 else 0

    # Long: check if any recent candle broke below lower Keltner
    long_penetration = False
    if penetration_lookback > 0:
        for i in range(1, penetration_lookback + 1):
            if abs_index - i >= 0:
                past_low = float(df["low"].iloc[abs_index - i])
                past_lower_band = float(df["keltner_lower"].iloc[abs_index - i])
                if past_low < past_lower_band:
                    long_penetration = True
                    break

    # Short: check if any recent candle broke above upper Keltner
    short_penetration = False
    if penetration_lookback > 0:
        for i in range(1, penetration_lookback + 1):
            if abs_index - i >= 0:
                past_high = float(df["high"].iloc[abs_index - i])
                past_upper_band = float(df["keltner_upper"].iloc[abs_index - i])
                if past_high > past_upper_band:
                    short_penetration = True
                    break

    debug_info.update({
        "long_penetration": long_penetration,
        "short_penetration": short_penetration,
    })

    # ================= LONG =================
    holding_long = (closes_slice > lower_slice).mean() >= min_hold_frac

    retest_long = (
            (low <= lower_band * (1 + touch_tol))
            and (close > lower_band)
            and (min(open_, close) > lower_band * (1 - body_tol))
    )

    keltner_pb_gap_long = (pb_bot - lower_band) / lower_band if lower_band != 0 else 0.0

    pb_target_long = (
            long_direction_ok and
            (keltner_pb_gap_long >= cloud_keltner_gap_min)
    )

    long_quality_ok = long_rejection_quality or long_penetration

    # SOFT VERSION: Only core filters are mandatory
    is_long = holding_long and retest_long and pb_target_long
    debug_info.update({
        "holding_long": holding_long,
        "retest_long": retest_long,
        "pb_target_long": pb_target_long,
        "long_quality_ok": long_quality_ok,
        "price_below_pbema": price_below_pbema,
    })

    # ================= SHORT =================
    holding_short = (closes_slice < upper_slice).mean() >= min_hold_frac

    retest_short = (
            (high >= upper_band * (1 - touch_tol))
            and (close < upper_band)
            and (max(open_, close) < upper_band * (1 + body_tol))
    )

    keltner_pb_gap_short = (upper_band - pb_top) / upper_band if upper_band != 0 else 0.0

    pb_target_short = (
            short_direction_ok and
            (keltner_pb_gap_short >= cloud_keltner_gap_min)
    )

    short_quality_ok = short_rejection_quality or short_penetration

    # SOFT VERSION: Only core filters are mandatory
    is_short = holding_short and retest_short and pb_target_short
    debug_info.update({
        "holding_short": holding_short,
        "retest_short": retest_short,
        "pb_target_short": pb_target_short,
        "short_quality_ok": short_quality_ok,
        "price_above_pbema": price_above_pbema,
    })

    # --- RSI filters (symmetric for LONG and SHORT) ---
    rsi_val = float(curr["rsi"])
    debug_info["rsi_value"] = rsi_val

    # LONG: RSI should not be too high (overbought territory)
    long_rsi_limit = rsi_limit + 10.0
    long_rsi_ok = rsi_val <= long_rsi_limit
    debug_info["long_rsi_ok"] = long_rsi_ok
    debug_info["long_rsi_limit"] = long_rsi_limit
    if is_long and not long_rsi_ok:
        is_long = False

    # SHORT: RSI should not be too low (oversold territory)
    short_rsi_limit = 100.0 - long_rsi_limit
    short_rsi_ok = rsi_val >= short_rsi_limit
    debug_info["short_rsi_ok"] = short_rsi_ok
    debug_info["short_rsi_limit"] = short_rsi_limit
    if is_short and not short_rsi_ok:
        is_short = False

    # --- AlphaTrend (optional) ---
    # Uses LINE DIRECTION to determine dominance (matches TradingView behavior):
    # - BUYERS dominant: AlphaTrend line is RISING (blue in TradingView)
    # - SELLERS dominant: AlphaTrend line is FALLING (red in TradingView)
    # NO TRADE if: at_is_flat = True (sideways market, no flow)
    if use_alphatrend:
        # Check for required AlphaTrend columns
        required_at_cols = ['alphatrend', 'alphatrend_2', 'at_buyers_dominant', 'at_sellers_dominant', 'at_is_flat']
        has_at_cols = all(col in df.columns for col in required_at_cols)

        if has_at_cols:
            # Get values for logging
            at_buyers = float(curr.get("at_buyers", 0))
            at_sellers = float(curr.get("at_sellers", 0))
            at_is_flat = bool(curr.get("at_is_flat", False))

            # USE PRE-CALCULATED DOMINANCE based on LINE DIRECTION
            at_buyers_dominant = bool(curr.get("at_buyers_dominant", False))
            at_sellers_dominant = bool(curr.get("at_sellers_dominant", False))

            # Add debug info for AlphaTrend
            debug_info["at_buyers"] = at_buyers
            debug_info["at_sellers"] = at_sellers
            debug_info["at_is_flat"] = at_is_flat
            debug_info["at_buyers_dominant"] = at_buyers_dominant
            debug_info["at_sellers_dominant"] = at_sellers_dominant

            # FLOW CHECK: If AlphaTrend line is flat, no trade (sideways market)
            if at_is_flat:
                if is_long:
                    is_long = False
                    debug_info["long_rejected_flat_at"] = True
                if is_short:
                    is_short = False
                    debug_info["short_rejected_flat_at"] = True

            # LONG filter: Buyers must be dominant (line rising = blue in TV)
            if is_long and not at_buyers_dominant:
                is_long = False
                debug_info["long_rejected_at_filter"] = True

            # SHORT filter: Sellers must be dominant (line falling = red in TV)
            if is_short and not at_sellers_dominant:
                is_short = False
                debug_info["short_rejected_at_filter"] = True

        elif "alphatrend" in df.columns:
            # Fallback to old single-line system (backward compatibility)
            at_val = float(curr["alphatrend"])
            debug_info["alphatrend_legacy"] = at_val
            if is_long and close < at_val:
                is_long = False
                debug_info["long_rejected_at_legacy"] = True
            if is_short and close > at_val:
                is_short = False
                debug_info["short_rejected_at_legacy"] = True

    # ---------- LONG ----------
    debug_info["long_candidate"] = is_long
    if is_long:
        swing_n = 20
        start = max(0, abs_index - swing_n)
        swing_low = float(df["low"].iloc[start:abs_index].min())
        if swing_low <= 0:
            return _ret(None, None, None, None, "Invalid Swing Low")

        sl_candidate = swing_low * 0.997
        band_sl = lower_band * 0.998
        sl = min(sl_candidate, band_sl)

        entry = close
        tp = pb_bot

        if tp <= entry:
            return _ret(None, None, None, None, "TP Below Entry")
        if sl >= entry:
            sl = min(swing_low * 0.995, entry * 0.997)

        risk = entry - sl
        reward = tp - entry
        if risk <= 0 or reward <= 0:
            return _ret(None, None, None, None, "Invalid RR")

        rr = reward / risk
        tp_dist_ratio = reward / entry

        debug_info.update({
            "tp_dist_ratio": tp_dist_ratio,
            "rr_value": rr,
            "long_rr_ok": rr >= min_rr,
        })

        if tp_dist_ratio < tp_min_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
        if tp_dist_ratio > tp_max_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
        if rr < min_rr:
            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

        reason = f"ACCEPTED(Base,R:{rr:.2f})"
        return _ret("LONG", entry, tp, sl, reason)

    # ---------- SHORT ----------
    debug_info["short_candidate"] = is_short
    if is_short:
        swing_n = 20
        start = max(0, abs_index - swing_n)
        swing_high = float(df["high"].iloc[start:abs_index].max())
        if swing_high <= 0:
            return _ret(None, None, None, None, "Invalid Swing High")

        sl_candidate = swing_high * 1.003
        band_sl = upper_band * 1.002
        sl = max(sl_candidate, band_sl)

        entry = close
        tp = pb_top

        if tp >= entry:
            return _ret(None, None, None, None, "TP Above Entry")
        if sl <= entry:
            sl = max(swing_high * 1.005, entry * 1.003)

        risk = sl - entry
        reward = entry - tp
        if risk <= 0 or reward <= 0:
            return _ret(None, None, None, None, "Invalid RR")

        rr = reward / risk
        tp_dist_ratio = reward / entry

        debug_info.update({
            "tp_dist_ratio": tp_dist_ratio,
            "rr_value": rr,
            "short_rr_ok": rr >= min_rr,
        })

        if tp_dist_ratio < tp_min_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
        if tp_dist_ratio > tp_max_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
        if rr < min_rr:
            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

        reason = f"ACCEPTED(Base,R:{rr:.2f})"
        return _ret("SHORT", entry, tp, sl, reason)

    return _ret(None, None, None, None, "No Signal")
