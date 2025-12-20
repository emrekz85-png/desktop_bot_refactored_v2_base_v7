# strategies/ssl_flow.py
# SSL Flow Strategy - Trend following with SSL HYBRID baseline
#
# Core Concept:
# "SSL HYBRID'den PBEMA bulutuna bir yol vardir!"
#
# Entry Logic:
# - LONG: Price above SSL baseline (HMA60) + AlphaTrend buyers > sellers
# - SHORT: Price below SSL baseline (HMA60) + AlphaTrend sellers > buyers
#
# Key Components:
# 1. SSL HYBRID (HMA60): Determines flow direction (support/resistance)
# 2. AlphaTrend: Confirms buyer/seller dominance (filters fake SSL signals)
# 3. PBEMA Cloud (EMA200): Take profit target
#
# Flow Detection:
# - SSL HYBRID alone can give fake signals in sideways markets
# - AlphaTrend dual-line system (buyers vs sellers) confirms real flow
# - If SSL turns bullish BUT AlphaTrend doesn't confirm -> NO TRADE
#
# Avoidance:
# - Don't trade when PBEMA and SSL baseline are too close (no room for profit)
# - Don't trade when AlphaTrend is flat (at_is_flat = True)

from typing import Tuple, Union
import pandas as pd
import numpy as np

from .base import SignalResult, SignalResultWithDebug


def check_ssl_flow_signal(
        df: pd.DataFrame,
        index: int = -2,
        min_rr: float = 2.0,
        rsi_limit: float = 70.0,
        # use_alphatrend REMOVED - AlphaTrend is now MANDATORY for SSL_Flow strategy
        # This prevents LONG trades when SELLERS are dominant (and vice versa)
        ssl_touch_tolerance: float = 0.002,
        ssl_body_tolerance: float = 0.003,
        min_pbema_distance: float = 0.004,
        tp_min_dist_ratio: float = 0.0015,
        tp_max_dist_ratio: float = 0.05,
        adx_min: float = 15.0,
        lookback_candles: int = 5,
        return_debug: bool = False,
) -> Union[SignalResult, SignalResultWithDebug]:
    """
    SSL Flow Strategy - Trend following with SSL HYBRID baseline direction.

    This strategy follows the flow/trend direction using:
    1. SSL HYBRID (baseline = HMA60) for trend direction
    2. AlphaTrend dual-lines for flow confirmation
    3. PBEMA cloud (EMA200) as TP target

    Entry Logic:
    - LONG: Price above SSL baseline + AlphaTrend buyers dominant + retest/bounce from baseline
    - SHORT: Price below SSL baseline + AlphaTrend sellers dominant + retest/bounce from baseline

    Args:
        df: DataFrame with OHLCV + indicators
        index: Candle index for signal check (default: -2, second to last candle)
        min_rr: Minimum risk/reward ratio
        rsi_limit: RSI threshold (LONG: not overbought, SHORT: not oversold)
        ssl_touch_tolerance: Tolerance for SSL baseline touch detection (0.002 = 0.2%)
        ssl_body_tolerance: Tolerance for candle body position relative to baseline
        min_pbema_distance: Minimum distance between price and PBEMA for valid TP
        tp_min_dist_ratio: Minimum TP distance ratio
        tp_max_dist_ratio: Maximum TP distance ratio
        adx_min: Minimum ADX value (trend strength filter)
        lookback_candles: Number of candles to check for baseline interaction
        return_debug: Whether to return debug info

    Returns:
        SignalResult or SignalResultWithDebug tuple
    """

    debug_info = {
        "adx_ok": None,
        "price_above_baseline": None,
        "price_below_baseline": None,
        "baseline_touch_long": None,
        "baseline_touch_short": None,
        "at_buyers_dominant": None,
        "at_sellers_dominant": None,
        "at_is_flat": None,
        "pbema_distance_ok": None,
        "long_rsi_ok": None,
        "short_rsi_ok": None,
        "rr_value": None,
        "tp_dist_ratio": None,
    }

    def _ret(s_type, entry, tp, sl, reason):
        if return_debug:
            return s_type, entry, tp, sl, reason, debug_info
        return s_type, entry, tp, sl, reason

    # Validate input
    if df is None or df.empty:
        return _ret(None, None, None, None, "No Data")

    required_cols = [
        "open", "high", "low", "close",
        "rsi", "adx",
        "baseline",  # SSL HYBRID (HMA60)
        "pb_ema_top", "pb_ema_bot",  # PBEMA cloud (EMA200)
    ]
    for col in required_cols:
        if col not in df.columns:
            return _ret(None, None, None, None, f"Missing {col}")

    try:
        curr = df.iloc[index]
    except Exception:
        return _ret(None, None, None, None, "Index Error")

    abs_index = index if index >= 0 else (len(df) + index)
    if abs_index < 60:  # Need enough history for HMA60 and other indicators
        return _ret(None, None, None, None, "Not Enough Data")

    # Extract current values
    open_ = float(curr["open"])
    high = float(curr["high"])
    low = float(curr["low"])
    close = float(curr["close"])
    baseline = float(curr["baseline"])  # SSL HYBRID (HMA60)
    pb_top = float(curr["pb_ema_top"])
    pb_bot = float(curr["pb_ema_bot"])
    adx_val = float(curr["adx"])
    rsi_val = float(curr["rsi"])

    # Check for NaN values
    if any(pd.isna([open_, high, low, close, baseline, pb_top, pb_bot, adx_val, rsi_val])):
        return _ret(None, None, None, None, "NaN Values")

    # ================= ADX FILTER =================
    debug_info["adx_ok"] = adx_val >= adx_min
    if not debug_info["adx_ok"]:
        return _ret(None, None, None, None, f"ADX Too Low ({adx_val:.1f})")

    # ================= SSL BASELINE DIRECTION =================
    # Price position relative to SSL baseline determines flow direction
    price_above_baseline = close > baseline
    price_below_baseline = close < baseline

    debug_info["price_above_baseline"] = price_above_baseline
    debug_info["price_below_baseline"] = price_below_baseline
    debug_info["baseline"] = baseline
    debug_info["close"] = close

    # ================= BASELINE TOUCH/RETEST DETECTION =================
    # Check if price has touched or come close to baseline in recent candles
    # This ensures we're entering on a retest, not chasing

    lookback_start = max(0, abs_index - lookback_candles)

    # For LONG: Check if low touched baseline (retest from above)
    baseline_touch_long = False
    for i in range(lookback_start, abs_index + 1):
        row_low = float(df["low"].iloc[i])
        row_baseline = float(df["baseline"].iloc[i])
        # Touch: low came within tolerance of baseline
        if row_low <= row_baseline * (1 + ssl_touch_tolerance):
            baseline_touch_long = True
            break

    # For SHORT: Check if high touched baseline (retest from below)
    baseline_touch_short = False
    for i in range(lookback_start, abs_index + 1):
        row_high = float(df["high"].iloc[i])
        row_baseline = float(df["baseline"].iloc[i])
        # Touch: high came within tolerance of baseline
        if row_high >= row_baseline * (1 - ssl_touch_tolerance):
            baseline_touch_short = True
            break

    debug_info["baseline_touch_long"] = baseline_touch_long
    debug_info["baseline_touch_short"] = baseline_touch_short

    # ================= CANDLE BODY POSITION =================
    # For LONG: Body should be above baseline (confirmation of support)
    # For SHORT: Body should be below baseline (confirmation of resistance)
    candle_body_min = min(open_, close)
    candle_body_max = max(open_, close)

    body_above_baseline = candle_body_min > baseline * (1 - ssl_body_tolerance)
    body_below_baseline = candle_body_max < baseline * (1 + ssl_body_tolerance)

    debug_info["body_above_baseline"] = body_above_baseline
    debug_info["body_below_baseline"] = body_below_baseline

    # ================= ALPHATREND FLOW CONFIRMATION =================
    # CRITICAL: AlphaTrend is MANDATORY for SSL_Flow strategy
    # This prevents LONG trades when SELLERS are dominant (and vice versa)
    # Without this, fake SSL signals lead to wrong-direction trades

    has_at_dual = all(col in df.columns for col in ['at_buyers', 'at_sellers', 'at_is_flat'])

    if not has_at_dual:
        return _ret(None, None, None, None, "AlphaTrend columns missing (REQUIRED)")

    at_buyers = float(curr.get("at_buyers", 0))
    at_sellers = float(curr.get("at_sellers", 0))
    at_is_flat = bool(curr.get("at_is_flat", False))

    at_buyers_dominant = at_buyers > at_sellers
    at_sellers_dominant = at_sellers > at_buyers

    debug_info["at_buyers"] = at_buyers
    debug_info["at_sellers"] = at_sellers
    debug_info["at_buyers_dominant"] = at_buyers_dominant
    debug_info["at_sellers_dominant"] = at_sellers_dominant
    debug_info["at_is_flat"] = at_is_flat

    # NO TRADE if AlphaTrend is flat (sideways market, no flow)
    if at_is_flat:
        return _ret(None, None, None, None, "AlphaTrend Flat (No Flow)")

    # ================= PBEMA DISTANCE CHECK =================
    # Ensure there's enough room between price and PBEMA for profitable trade
    pbema_mid = (pb_top + pb_bot) / 2

    # For LONG: PBEMA should be above price (TP target)
    long_pbema_distance = (pb_bot - close) / close if close > 0 else 0
    # For SHORT: PBEMA should be below price (TP target)
    short_pbema_distance = (close - pb_top) / close if close > 0 else 0

    debug_info["long_pbema_distance"] = long_pbema_distance
    debug_info["short_pbema_distance"] = short_pbema_distance

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

    min_wick_ratio = 0.10  # At least 10% wick for rejection signal
    long_rejection = lower_wick_ratio >= min_wick_ratio
    short_rejection = upper_wick_ratio >= min_wick_ratio

    debug_info["lower_wick_ratio"] = lower_wick_ratio
    debug_info["upper_wick_ratio"] = upper_wick_ratio
    debug_info["long_rejection"] = long_rejection
    debug_info["short_rejection"] = short_rejection

    # ================= LONG SIGNAL =================
    # Conditions:
    # 1. Price above SSL baseline (bullish flow)
    # 2. AlphaTrend buyers dominant (flow confirmation)
    # 3. Recent baseline touch/retest (entry opportunity)
    # 4. Candle body above baseline (support confirmation)
    # 5. PBEMA above price (room for TP)
    # 6. Rejection wick (bounce confirmation)

    is_long = (
        price_above_baseline and
        at_buyers_dominant and
        baseline_touch_long and
        body_above_baseline and
        long_pbema_distance >= min_pbema_distance and
        long_rejection
    )

    # RSI filter for LONG: not overbought
    long_rsi_ok = rsi_val <= rsi_limit
    debug_info["long_rsi_ok"] = long_rsi_ok
    debug_info["rsi_value"] = rsi_val

    if is_long and not long_rsi_ok:
        is_long = False
        debug_info["long_rejected_rsi"] = True

    debug_info["is_long_candidate"] = is_long

    # ================= SHORT SIGNAL =================
    # Conditions:
    # 1. Price below SSL baseline (bearish flow)
    # 2. AlphaTrend sellers dominant (flow confirmation)
    # 3. Recent baseline touch/retest (entry opportunity)
    # 4. Candle body below baseline (resistance confirmation)
    # 5. PBEMA below price (room for TP)
    # 6. Rejection wick (bounce confirmation)

    is_short = (
        price_below_baseline and
        at_sellers_dominant and
        baseline_touch_short and
        body_below_baseline and
        short_pbema_distance >= min_pbema_distance and
        short_rejection
    )

    # RSI filter for SHORT: not oversold
    short_rsi_limit = 100.0 - rsi_limit
    short_rsi_ok = rsi_val >= short_rsi_limit
    debug_info["short_rsi_ok"] = short_rsi_ok
    debug_info["short_rsi_limit"] = short_rsi_limit

    if is_short and not short_rsi_ok:
        is_short = False
        debug_info["short_rejected_rsi"] = True

    debug_info["is_short_candidate"] = is_short

    # ================= EXECUTE LONG =================
    if is_long:
        # Entry: current close
        entry = close

        # TP: PBEMA cloud bottom (pb_ema_bot)
        tp = pb_bot

        # SL: Below recent swing low or below baseline
        swing_n = 20
        start = max(0, abs_index - swing_n)
        swing_low = float(df["low"].iloc[start:abs_index].min())

        # SL candidates: swing low or baseline
        sl_swing = swing_low * 0.998
        sl_baseline = baseline * 0.998
        sl = min(sl_swing, sl_baseline)

        if tp <= entry:
            return _ret(None, None, None, None, "TP Below Entry (LONG)")
        if sl >= entry:
            sl = min(swing_low * 0.995, baseline * 0.995)

        risk = entry - sl
        reward = tp - entry

        if risk <= 0 or reward <= 0:
            return _ret(None, None, None, None, "Invalid RR (LONG)")

        rr = reward / risk
        tp_dist_ratio = reward / entry

        debug_info.update({
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "rr_value": rr,
            "tp_dist_ratio": tp_dist_ratio,
        })

        if tp_dist_ratio < tp_min_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
        if tp_dist_ratio > tp_max_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
        if rr < min_rr:
            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

        reason = f"ACCEPTED(SSL_Flow,R:{rr:.2f})"
        return _ret("LONG", entry, tp, sl, reason)

    # ================= EXECUTE SHORT =================
    if is_short:
        # Entry: current close
        entry = close

        # TP: PBEMA cloud top (pb_ema_top)
        tp = pb_top

        # SL: Above recent swing high or above baseline
        swing_n = 20
        start = max(0, abs_index - swing_n)
        swing_high = float(df["high"].iloc[start:abs_index].max())

        # SL candidates: swing high or baseline
        sl_swing = swing_high * 1.002
        sl_baseline = baseline * 1.002
        sl = max(sl_swing, sl_baseline)

        if tp >= entry:
            return _ret(None, None, None, None, "TP Above Entry (SHORT)")
        if sl <= entry:
            sl = max(swing_high * 1.005, baseline * 1.005)

        risk = sl - entry
        reward = entry - tp

        if risk <= 0 or reward <= 0:
            return _ret(None, None, None, None, "Invalid RR (SHORT)")

        rr = reward / risk
        tp_dist_ratio = reward / entry

        debug_info.update({
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "rr_value": rr,
            "tp_dist_ratio": tp_dist_ratio,
        })

        if tp_dist_ratio < tp_min_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
        if tp_dist_ratio > tp_max_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
        if rr < min_rr:
            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

        reason = f"ACCEPTED(SSL_Flow,R:{rr:.2f})"
        return _ret("SHORT", entry, tp, sl, reason)

    return _ret(None, None, None, None, "No Signal")
