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

from typing import Tuple, Union, Dict
import pandas as pd
import numpy as np

from .base import SignalResult, SignalResultWithDebug


def calculate_signal_score(
    adx: float,
    baseline_touch: bool,
    at_dominant: bool,
    at_is_flat: bool,
    pbema_distance: float,
    wick_ratio: float,
    no_overlap: bool,
    body_position_ok: bool,
    regime_ok: bool,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Calculate composite signal score for SSL Flow strategy.

    Converts binary AND logic to weighted scoring system:
    - Each filter contributes points based on quality
    - Total score compared against threshold for signal

    Args:
        adx: ADX value (trend strength)
        baseline_touch: Whether baseline was touched/retested
        at_dominant: AlphaTrend dominance in signal direction
        at_is_flat: Whether AlphaTrend is flat (no flow)
        pbema_distance: Distance to PBEMA target (ratio)
        wick_ratio: Rejection wick size (ratio)
        no_overlap: SSL-PBEMA bands don't overlap
        body_position_ok: Candle body on correct side of baseline
        regime_ok: Regime is trending (not ranging)

    Returns:
        (score, max_score, breakdown_dict)
    """
    score = 0.0
    max_score = 10.0
    breakdown = {}

    # 1. ADX Trend Strength (max 2.0)
    # Strong trends = best entries for trend-following
    if adx > 30:
        s = 2.0
    elif adx > 25:
        s = 1.5
    elif adx > 20:
        s = 1.0
    elif adx > 15:
        s = 0.5
    else:
        s = 0.0
    score += s
    breakdown['adx'] = s

    # 2. Regime Gating (max 1.0)
    # Bonus if regime is trending (filters choppy markets)
    if regime_ok:
        s = 1.0
    else:
        s = 0.0
    score += s
    breakdown['regime'] = s

    # 3. Baseline Touch (max 2.0)
    # Critical: ensures we're entering on retest, not chasing
    if baseline_touch:
        s = 2.0
    else:
        s = 0.0
    score += s
    breakdown['baseline_touch'] = s

    # 4. AlphaTrend Confirmation (max 2.0)
    # Confirms real flow (not fake SSL signals)
    if at_dominant and not at_is_flat:
        s = 2.0
    elif at_dominant:
        s = 1.5
    elif not at_is_flat:
        s = 1.0
    else:
        s = 0.0
    score += s
    breakdown['alphatrend'] = s

    # 5. PBEMA Distance (max 1.0)
    # Room for profit - more distance = better
    if pbema_distance >= 0.006:
        s = 1.0
    elif pbema_distance >= 0.004:
        s = 0.75
    elif pbema_distance >= 0.003:
        s = 0.5
    else:
        s = 0.25
    score += s
    breakdown['pbema_distance'] = s

    # 6. Wick Rejection (max 1.0)
    # Strong rejection = higher quality setup
    if wick_ratio >= 0.15:
        s = 1.0
    elif wick_ratio >= 0.10:
        s = 0.75
    elif wick_ratio >= 0.05:
        s = 0.5
    else:
        s = 0.0
    score += s
    breakdown['wick_rejection'] = s

    # 7. Body Position (max 0.5)
    # Body on correct side confirms support/resistance
    if body_position_ok:
        s = 0.5
    else:
        s = 0.0
    score += s
    breakdown['body_position'] = s

    # 8. No Overlap (max 0.5)
    # SSL-PBEMA overlap = no room for flow
    if no_overlap:
        s = 0.5
    else:
        s = 0.0
    score += s
    breakdown['no_overlap'] = s

    return score, max_score, breakdown


def check_ssl_flow_signal(
        df: pd.DataFrame,
        index: int = -2,
        min_rr: float = 2.0,
        rsi_limit: float = 70.0,
        # use_alphatrend REMOVED - AlphaTrend is now MANDATORY for SSL_Flow strategy
        # This prevents LONG trades when SELLERS are dominant (and vice versa)
        ssl_touch_tolerance: float = 0.003,
        ssl_body_tolerance: float = 0.003,
        min_pbema_distance: float = 0.004,
        tp_min_dist_ratio: float = 0.0015,
        tp_max_dist_ratio: float = 0.05,
        adx_min: float = 15.0,
        adx_max: float = 40.0,
        lookback_candles: int = 5,
        regime_adx_threshold: float = 20.0,  # v1.7.2: Now configurable for grid search
        regime_lookback: int = 50,  # v1.7.2: Now configurable
        skip_overlap_check: bool = False,  # Filter Discovery: skip SSL-PBEMA overlap check
        skip_wick_rejection: bool = False,  # Filter Discovery: skip wick rejection check
        use_scoring: bool = False,  # NEW: Enable scoring system (vs AND logic)
        score_threshold: float = 6.0,  # NEW: Minimum score (out of 10.0) for signal
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

    Mode Selection (use_scoring parameter):
    - use_scoring=False (default): Binary AND logic - all filters must pass (strict, fewer trades)
    - use_scoring=True: Weighted scoring - filters contribute points, signal if score >= threshold

    Args:
        df: DataFrame with OHLCV + indicators
        index: Candle index for signal check (default: -2, second to last candle)
        min_rr: Minimum risk/reward ratio
        rsi_limit: RSI threshold (LONG: not overbought, SHORT: not oversold)
        ssl_touch_tolerance: Tolerance for SSL baseline touch detection (0.003 = 0.3%)
        ssl_body_tolerance: Tolerance for candle body position relative to baseline
        min_pbema_distance: Minimum distance between price and PBEMA for valid TP
        tp_min_dist_ratio: Minimum TP distance ratio
        tp_max_dist_ratio: Maximum TP distance ratio
        adx_min: Minimum ADX value (trend strength filter)
        adx_max: Maximum ADX value (filters overly strong trends that may reverse)
        lookback_candles: Number of candles to check for baseline interaction
        regime_adx_threshold: Average ADX threshold for regime detection
        regime_lookback: Number of candles for regime detection
        skip_overlap_check: Skip SSL-PBEMA overlap check (filter discovery)
        skip_wick_rejection: Skip wick rejection check (filter discovery)
        use_scoring: Enable weighted scoring system (vs binary AND logic)
        score_threshold: Minimum score required for signal (out of 10.0)
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

    # OPTIMIZATION 4: Cache column arrays for vectorized operations
    _open_arr = df["open"].values
    _high_arr = df["high"].values
    _low_arr = df["low"].values
    _close_arr = df["close"].values
    _baseline_arr = df["baseline"].values
    _pb_top_arr = df["pb_ema_top"].values
    _pb_bot_arr = df["pb_ema_bot"].values
    _adx_arr = df["adx"].values
    _rsi_arr = df["rsi"].values

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
    # Note: ADX max filter REMOVED in v1.6.2-restored
    # Reason: ADX max affects optimizer, causes "0 configs found"
    # SSL Flow is trend-following - strong trends (ADX>40) are BEST opportunities
    debug_info["adx_ok"] = adx_val >= adx_min
    if not debug_info["adx_ok"]:
        return _ret(None, None, None, None, f"ADX Too Low ({adx_val:.1f})")

    # ================= REGIME GATING (v1.7.0, v1.7.2 configurable) =================
    # Window-level regime detection using ADX average over lookback period
    # RANGING markets (ADX_avg < threshold) = skip trade entirely
    # This prevents trades during sideways/choppy markets where SSL Flow struggles
    # v1.7.2: regime_adx_threshold and regime_lookback are now function parameters

    regime_start = max(0, abs_index - regime_lookback)
    # FIX: Look-ahead bias - exclude current bar from regime calculation
    # At signal time, we don't know the current bar's final ADX yet
    adx_window = df["adx"].iloc[regime_start:abs_index]
    adx_avg = float(adx_window.mean()) if len(adx_window) > 0 else adx_val

    regime = "TRENDING" if adx_avg >= regime_adx_threshold else "RANGING"
    debug_info["adx_avg"] = adx_avg
    debug_info["regime"] = regime
    debug_info["regime_lookback"] = regime_lookback
    debug_info["regime_adx_threshold"] = regime_adx_threshold

    if regime == "RANGING":
        return _ret(None, None, None, None, f"RANGING Regime (ADX_avg={adx_avg:.1f})")

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

    # OPTIMIZATION 3: Vectorize baseline touch detection with NumPy
    # For LONG: Check if low touched baseline (retest from above)
    lookback_lows = _low_arr[lookback_start:abs_index + 1]
    lookback_baselines_long = _baseline_arr[lookback_start:abs_index + 1]
    baseline_touch_long = np.any(lookback_lows <= lookback_baselines_long * (1 + ssl_touch_tolerance))

    # For SHORT: Check if high touched baseline (retest from below)
    lookback_highs = _high_arr[lookback_start:abs_index + 1]
    lookback_baselines_short = _baseline_arr[lookback_start:abs_index + 1]
    baseline_touch_short = np.any(lookback_highs >= lookback_baselines_short * (1 - ssl_touch_tolerance))

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

    # Check for required AlphaTrend columns
    required_at_cols = ['alphatrend', 'alphatrend_2', 'at_buyers_dominant', 'at_sellers_dominant', 'at_is_flat']
    has_at_cols = all(col in df.columns for col in required_at_cols)

    if not has_at_cols:
        return _ret(None, None, None, None, "AlphaTrend columns missing (REQUIRED)")

    # Get AlphaTrend values for logging
    alphatrend_val = float(curr.get("alphatrend", 0))
    alphatrend_2_val = float(curr.get("alphatrend_2", 0))
    at_is_flat = bool(curr.get("at_is_flat", False))

    # USE PRE-CALCULATED DOMINANCE based on LINE DIRECTION
    # Buyers dominant = AlphaTrend line is RISING (blue in TradingView)
    # Sellers dominant = AlphaTrend line is FALLING (red in TradingView)
    at_buyers_dominant = bool(curr.get("at_buyers_dominant", False))
    at_sellers_dominant = bool(curr.get("at_sellers_dominant", False))

    # Also get legacy at_buyers/at_sellers for backward compat logging
    at_buyers = float(curr.get("at_buyers", 0))
    at_sellers = float(curr.get("at_sellers", 0))

    debug_info["alphatrend"] = alphatrend_val
    debug_info["alphatrend_2"] = alphatrend_2_val
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

    # ================= PBEMA-SSL BASELINE OVERLAP CHECK =================
    # "PBEMA ve SSL Hybrid bantları İÇ İÇE olduğunda işlem ALINMAZ"
    # LONG için: PBEMA baseline'ın ÜSTÜNDE olmalı (yukarıya gidecek yol var)
    # SHORT için: PBEMA baseline'ın ALTINDA olmalı (aşağıya gidecek yol var)

    OVERLAP_THRESHOLD = 0.005  # %0.5 eşik değeri

    baseline_pbema_distance = abs(baseline - pbema_mid) / pbema_mid if pbema_mid > 0 else 0
    is_overlapping = baseline_pbema_distance < OVERLAP_THRESHOLD

    # LONG için: PBEMA hedefi baseline'ın üstünde olmalı
    pbema_above_baseline = pbema_mid > baseline
    # SHORT için: PBEMA hedefi baseline'ın altında olmalı
    pbema_below_baseline = pbema_mid < baseline

    debug_info["baseline_pbema_distance"] = baseline_pbema_distance
    debug_info["is_overlapping"] = is_overlapping
    debug_info["pbema_above_baseline"] = pbema_above_baseline
    debug_info["pbema_below_baseline"] = pbema_below_baseline

    # İç içe durumunda işlem alma - flow yok
    # Filter Discovery: skip_overlap_check allows disabling this filter
    if not skip_overlap_check and is_overlapping:
        return _ret(None, None, None, None, "SSL-PBEMA Overlap (No Flow)")

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
    # 7. PBEMA above baseline (target reachable - "yol var")

    is_long = (
        price_above_baseline and
        at_buyers_dominant and
        baseline_touch_long and
        body_above_baseline and
        long_pbema_distance >= min_pbema_distance and
        (skip_wick_rejection or long_rejection) and  # Filter Discovery: can skip wick check
        pbema_above_baseline  # PBEMA hedefine gidecek yol var (baseline üstünde)
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
    # 7. PBEMA below baseline (target reachable - "yol var")

    is_short = (
        price_below_baseline and
        at_sellers_dominant and
        baseline_touch_short and
        body_below_baseline and
        short_pbema_distance >= min_pbema_distance and
        (skip_wick_rejection or short_rejection) and  # Filter Discovery: can skip wick check
        pbema_below_baseline  # PBEMA hedefine gidecek yol var (baseline altında)
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

    # ================= SCORING MODE (ALTERNATIVE TO AND LOGIC) =================
    # If use_scoring=True, override AND logic with weighted scoring system
    if use_scoring:
        # Calculate scores for LONG and SHORT separately
        long_score, long_max, long_breakdown = calculate_signal_score(
            adx=adx_val,
            baseline_touch=baseline_touch_long,
            at_dominant=at_buyers_dominant,
            at_is_flat=at_is_flat,
            pbema_distance=long_pbema_distance,
            wick_ratio=lower_wick_ratio,
            no_overlap=(not is_overlapping or skip_overlap_check),
            body_position_ok=body_above_baseline,
            regime_ok=(regime == "TRENDING"),
        )

        short_score, short_max, short_breakdown = calculate_signal_score(
            adx=adx_val,
            baseline_touch=baseline_touch_short,
            at_dominant=at_sellers_dominant,
            at_is_flat=at_is_flat,
            pbema_distance=short_pbema_distance,
            wick_ratio=upper_wick_ratio,
            no_overlap=(not is_overlapping or skip_overlap_check),
            body_position_ok=body_below_baseline,
            regime_ok=(regime == "TRENDING"),
        )

        # Store in debug info
        debug_info["use_scoring"] = True
        debug_info["score_threshold"] = score_threshold
        debug_info["long_score"] = long_score
        debug_info["long_score_breakdown"] = long_breakdown
        debug_info["short_score"] = short_score
        debug_info["short_score_breakdown"] = short_breakdown

        # CRITICAL FILTERS (always checked even in scoring mode):
        # 1. Price position relative to baseline (determines direction)
        # 2. AlphaTrend direction (confirms buyers vs sellers)
        # 3. RSI bounds (avoid extreme overbought/oversold)

        # Override is_long/is_short based on score + critical filters
        is_long_scoring = (
            price_above_baseline and  # CORE: price direction
            at_buyers_dominant and  # CORE: AlphaTrend confirms buyers
            not at_is_flat and  # CORE: flow exists
            long_rsi_ok and  # CORE: not overbought
            long_score >= score_threshold  # SCORING: composite quality
        )

        is_short_scoring = (
            price_below_baseline and  # CORE: price direction
            at_sellers_dominant and  # CORE: AlphaTrend confirms sellers
            not at_is_flat and  # CORE: flow exists
            short_rsi_ok and  # CORE: not oversold
            short_score >= score_threshold  # SCORING: composite quality
        )

        # Override AND logic candidates with scoring results
        is_long = is_long_scoring
        is_short = is_short_scoring

        debug_info["is_long_candidate_scoring"] = is_long
        debug_info["is_short_candidate_scoring"] = is_short

    # ================= EXECUTE LONG =================
    if is_long:
        # Entry: current close
        entry = close

        # TP: PBEMA cloud bottom (pb_ema_bot)
        tp = pb_bot

        # SL: Below recent swing low or below baseline
        swing_n = 20
        start = max(0, abs_index - swing_n)
        swing_low = float(_low_arr[start:abs_index].min())

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
        swing_high = float(_high_arr[start:abs_index].max())

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
