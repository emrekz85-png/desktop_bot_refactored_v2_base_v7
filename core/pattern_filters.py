# core/pattern_filters.py
# Pattern-Based Trading Filters (Patterns 3-7 from Real Trade Analysis)
#
# This module contains filter functions for patterns identified from real trade analysis.
# Each pattern can be used as an entry filter or signal enhancement.
#
# Patterns:
# - Pattern 3: Liquidity Grab / Fakeout Detection
# - Pattern 4: SSL Baseline Slope Filter (Anti-Ranging)
# - Pattern 5: HTF Bounce Detection
# - Pattern 6: Momentum Loss After Trend
# - Pattern 7: SSL Dynamic Support

from typing import Optional, Tuple, Dict, Literal
import pandas as pd
import numpy as np


# ==========================================
# PATTERN 3: LIQUIDITY GRAB / FAKEOUT DETECTION
# ==========================================
# Real Trade Evidence:
# - NO 1: "hizlica zıplayıp yukarıdaki SSL HYBRID bandına carpmis. Entry yerimizdi"
# - NO 6: "Fakeout yükseliş sırasında entry"
# - NO 16: "hızlı bir liquidity grab yapıp zıplıyor"

def detect_liquidity_grab(
        df: pd.DataFrame,
        index: int,
        lookback: int = 5,
        spike_threshold: float = 0.015,  # 1.5% spike
        reversal_threshold: float = 0.005,  # 0.5% reversal
        return_debug: bool = False,
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Detect liquidity grab (sharp spike + reversal).

    Pattern:
    - Price makes sharp move in one direction (>1.5% within 2-5 candles)
    - Then reverses sharply in opposite direction
    - Creates a long wick candle (stop hunt)

    Args:
        df: DataFrame with OHLCV data
        index: Current candle index
        lookback: Candles to check for spike (default: 5)
        spike_threshold: Minimum spike size (default: 1.5%)
        reversal_threshold: Minimum reversal size (default: 0.5%)
        return_debug: Return debug information

    Returns:
        (signal_type, debug_info)
        - signal_type: "LONG_GRAB", "SHORT_GRAB", or None
        - debug_info: Diagnostic info (if return_debug=True)

    Example:
        >>> signal, debug = detect_liquidity_grab(df, -1, return_debug=True)
        >>> if signal == "LONG_GRAB":
        ...     print("Liquidity grabbed below, expect bounce")
    """

    debug_info = {
        "spike_detected": False,
        "reversal_detected": False,
        "spike_pct": None,
        "reversal_pct": None,
    }

    abs_idx = index if index >= 0 else len(df) + index

    if abs_idx < lookback + 3:
        return (None, debug_info) if return_debug else (None, None)

    try:
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        # === Check for upward spike followed by rejection (SHORT grab) ===
        recent_high = np.max(highs[abs_idx - lookback:abs_idx + 1])
        prev_avg = np.mean(closes[abs_idx - lookback - 3:abs_idx - lookback])

        spike_up = (recent_high - prev_avg) / prev_avg
        debug_info["spike_pct"] = spike_up

        # Did price spike up then fall back?
        curr_close = closes[abs_idx]
        fell_back = (recent_high - curr_close) / recent_high
        debug_info["reversal_pct"] = fell_back

        if spike_up > spike_threshold and fell_back > reversal_threshold:
            debug_info["spike_detected"] = True
            debug_info["reversal_detected"] = True
            return ("SHORT_GRAB", debug_info) if return_debug else ("SHORT_GRAB", None)

        # === Check for downward spike followed by bounce (LONG grab) ===
        recent_low = np.min(lows[abs_idx - lookback:abs_idx + 1])

        spike_down = (prev_avg - recent_low) / prev_avg
        bounced_back = (curr_close - recent_low) / recent_low

        if spike_down > spike_threshold and bounced_back > reversal_threshold:
            debug_info["spike_detected"] = True
            debug_info["reversal_detected"] = True
            debug_info["spike_pct"] = spike_down
            debug_info["reversal_pct"] = bounced_back
            return ("LONG_GRAB", debug_info) if return_debug else ("LONG_GRAB", None)

        return (None, debug_info) if return_debug else (None, None)

    except (IndexError, ValueError, ZeroDivisionError):
        return (None, debug_info) if return_debug else (None, None)


# ==========================================
# PATTERN 4: SSL BASELINE SLOPE FILTER
# ==========================================
# Real Trade Evidence:
# - NO 6: "SSL HYBRID bandının yavaşça asagida doğru yada yanlamasina... momentum yavasladigini işaret eder"
# - NO 3: "SSL HYBRID bandı yukarıya doğru bozulmadan oluşmaya baslamis"

def is_ssl_baseline_ranging(
        df: pd.DataFrame,
        index: int,
        lookback: int = 10,
        flat_threshold: float = 0.0006,  # 0.06% per candle (was 0.15% - too strict)
        return_debug: bool = False,
) -> Tuple[bool, Optional[Dict]]:
    """
    Check if SSL baseline is flat/ranging (avoid trades).

    Ranging = SSL baseline slope < threshold

    Args:
        df: DataFrame with 'baseline' column
        index: Current candle index
        lookback: Candles for slope calculation (default: 10)
        flat_threshold: Threshold for flat detection (default: 0.15%)
        return_debug: Return debug information

    Returns:
        (is_ranging, debug_info)
        - is_ranging: True if ranging (DON'T TRADE), False if trending
        - debug_info: Diagnostic info (if return_debug=True)

    Example:
        >>> is_ranging, debug = is_ssl_baseline_ranging(df, -1, return_debug=True)
        >>> if is_ranging:
        ...     print("SSL is flat - skip trade")
    """

    debug_info = {
        "slope_pct": None,
        "is_flat": False,
    }

    if 'baseline' not in df.columns:
        return (False, debug_info) if return_debug else (False, None)

    abs_idx = index if index >= 0 else len(df) + index

    if abs_idx < lookback + 5:
        return (False, debug_info) if return_debug else (False, None)

    try:
        baseline_values = df['baseline'].values[abs_idx - lookback:abs_idx + 1]

        # Calculate slope (linear regression)
        x = np.arange(len(baseline_values))
        slope, _ = np.polyfit(x, baseline_values, 1)

        # Normalize slope by price (% change per candle)
        avg_baseline = np.mean(baseline_values)
        slope_pct = abs(slope / avg_baseline) if avg_baseline > 0 else 0

        debug_info["slope_pct"] = slope_pct
        is_flat = slope_pct < flat_threshold
        debug_info["is_flat"] = is_flat

        return (is_flat, debug_info) if return_debug else (is_flat, None)

    except (IndexError, ValueError, ZeroDivisionError):
        return (False, debug_info) if return_debug else (False, None)


# ==========================================
# PATTERN 5: HTF BOUNCE DETECTION
# ==========================================
# Real Trade Evidence:
# - NO 3: "Fiyat güçlü sekilde satış yemiş ve muhtemelen HTF bir zonedan ziplama yapmış"
# - NO 4: "Fiyat güclü satış yemiş ve yine muhtemelen tekrardan HTF bir alandan tepki alıp ziplamis"
# - NO 10: "Fiyat güçlü bir satış sonrası muhtemelen HTF önemli bir alandan bounce yiyor"
# - NO 14: "Fiyat güçlü bir satış yemiş muhtemelen HTF key zone dan tepki alıp ziplamis"

def detect_htf_bounce(
        df: pd.DataFrame,
        index: int,
        drop_threshold: float = 0.015,  # 1.5% drop (was 3% - too strict for 15m TF)
        bounce_threshold: float = 0.008,  # 0.8% bounce (was 1.5% - too strict)
        lookback: int = 10,
        return_debug: bool = False,
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Detect HTF support/resistance bounce.

    Pattern:
    - Price drops >3% sharply (within 5-10 candles)
    - Then bounces back >1.5% (reversal from HTF zone)
    - Creates V-shape or bounce candle

    Args:
        df: DataFrame with OHLCV data
        index: Current candle index
        drop_threshold: Minimum drop size (default: 3%)
        bounce_threshold: Minimum bounce size (default: 1.5%)
        lookback: Candles to check for drop (default: 10)
        return_debug: Return debug information

    Returns:
        (signal_type, debug_info)
        - signal_type: "LONG_BOUNCE", "SHORT_BOUNCE", or None
        - debug_info: Diagnostic info (if return_debug=True)

    Example:
        >>> signal, debug = detect_htf_bounce(df, -1, return_debug=True)
        >>> if signal == "LONG_BOUNCE":
        ...     print("HTF support bounce detected")
    """

    debug_info = {
        "drop_pct": None,
        "bounce_pct": None,
    }

    abs_idx = index if index >= 0 else len(df) + index

    if abs_idx < lookback + 5:
        return (None, debug_info) if return_debug else (None, None)

    try:
        closes = df['close'].values
        lows = df['low'].values
        highs = df['high'].values

        # === Check for sharp drop + bounce (LONG) ===
        recent_high = np.max(highs[abs_idx - lookback:abs_idx - 2])
        recent_low = np.min(lows[abs_idx - 5:abs_idx + 1])
        curr_close = closes[abs_idx]

        drop = (recent_high - recent_low) / recent_high
        bounce = (curr_close - recent_low) / recent_low

        debug_info["drop_pct"] = drop
        debug_info["bounce_pct"] = bounce

        if drop > drop_threshold and bounce > bounce_threshold:
            return ("LONG_BOUNCE", debug_info) if return_debug else ("LONG_BOUNCE", None)

        # === Check for sharp rise + rejection (SHORT) ===
        recent_low_short = np.min(lows[abs_idx - lookback:abs_idx - 2])
        recent_high_short = np.max(highs[abs_idx - 5:abs_idx + 1])

        rise = (recent_high_short - recent_low_short) / recent_low_short
        rejection = (recent_high_short - curr_close) / recent_high_short

        if rise > drop_threshold and rejection > bounce_threshold:
            debug_info["drop_pct"] = rise
            debug_info["bounce_pct"] = rejection
            return ("SHORT_BOUNCE", debug_info) if return_debug else ("SHORT_BOUNCE", None)

        return (None, debug_info) if return_debug else (None, None)

    except (IndexError, ValueError, ZeroDivisionError):
        return (None, debug_info) if return_debug else (None, None)


# ==========================================
# PATTERN 6: MOMENTUM LOSS AFTER TREND
# ==========================================
# Real Trade Evidence:
# - NO 2: "ALPHATREND indikatorunun merdiven gibi yükseldiğini görüyoruz ve ilk kirilimda short entry"
# - NO 5: "Fiyat hızla yükselmiş... SSL HYBRID icinde yeni high vermek icin çabalıyor fakat basarisiz"

def detect_momentum_loss_after_trend(
        df: pd.DataFrame,
        index: int,
        lookback: int = 15,
        min_consecutive: int = 3,  # (was 5 - too strict, rarely occurs)
        return_debug: bool = False,
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Detect first momentum break after strong trend.

    Pattern:
    - AlphaTrend was rising/falling like "stairs" (consecutive moves)
    - Then first reversal candle appears (momentum breaks)
    - Entry on counter-trend signal after break

    Args:
        df: DataFrame with AT columns
        index: Current candle index
        lookback: Candles to check for trend (default: 15)
        min_consecutive: Min consecutive AT candles (default: 5)
        return_debug: Return debug information

    Returns:
        (signal_type, debug_info)
        - signal_type: "LONG_BREAK", "SHORT_BREAK", or None
        - debug_info: Diagnostic info (if return_debug=True)

    Example:
        >>> signal, debug = detect_momentum_loss_after_trend(df, -1, return_debug=True)
        >>> if signal == "SHORT_BREAK":
        ...     print("Uptrend momentum broken - short opportunity")
    """

    debug_info = {
        "consecutive_count": 0,
        "trend_broken": False,
    }

    required_cols = ['at_buyers_dominant', 'at_sellers_dominant']
    if not all(col in df.columns for col in required_cols):
        return (None, debug_info) if return_debug else (None, None)

    abs_idx = index if index >= 0 else len(df) + index

    if abs_idx < lookback + 5:
        return (None, debug_info) if return_debug else (None, None)

    try:
        at_buyers = df['at_buyers_dominant'].values
        at_sellers = df['at_sellers_dominant'].values

        # === Check for consecutive buyer dominance (stairs up) ===
        consecutive_buyers = 0
        for i in range(abs_idx - lookback, abs_idx):
            if at_buyers[i]:
                consecutive_buyers += 1
            else:
                consecutive_buyers = 0

        # If had N+ consecutive buyer candles, now broken
        if consecutive_buyers >= min_consecutive and at_sellers[abs_idx]:
            debug_info["consecutive_count"] = consecutive_buyers
            debug_info["trend_broken"] = True
            return ("SHORT_BREAK", debug_info) if return_debug else ("SHORT_BREAK", None)

        # === Check for consecutive seller dominance (stairs down) ===
        consecutive_sellers = 0
        for i in range(abs_idx - lookback, abs_idx):
            if at_sellers[i]:
                consecutive_sellers += 1
            else:
                consecutive_sellers = 0

        # If had N+ consecutive seller candles, now broken
        if consecutive_sellers >= min_consecutive and at_buyers[abs_idx]:
            debug_info["consecutive_count"] = consecutive_sellers
            debug_info["trend_broken"] = True
            return ("LONG_BREAK", debug_info) if return_debug else ("LONG_BREAK", None)

        return (None, debug_info) if return_debug else (None, None)

    except (IndexError, ValueError):
        return (None, debug_info) if return_debug else (None, None)


# ==========================================
# PATTERN 4B: SSL SLOPE DIRECTION FILTER (NEW - From Professional Analysis)
# ==========================================
# Analysis Evidence:
# - 40% of LONG trades are "quick failures" (≤20 bars, 0% recovery)
# - All 8 quick failures show flat or oscillating SSL Baseline
# - SHORT trades have 0% quick failures (better signal quality)
# - SSL slope filter expected to add $60-75/year

def check_ssl_slope_direction(
        df: pd.DataFrame,
        index: int,
        signal_type: str,  # "LONG" or "SHORT"
        lookback: int = 10,
        min_slope_pct: float = 0.003,  # 0.3% minimum slope (from analysis recommendation)
        return_debug: bool = False,
) -> Tuple[bool, Optional[Dict]]:
    """
    Check if SSL baseline slope supports the signal direction.

    Key Insight from Analysis:
    - LONG signals need UPWARD sloping SSL (min 0.3%)
    - SHORT signals need DOWNWARD sloping SSL (min 0.3%)
    - Flat SSL = ranging market = avoid entry

    This filter is specifically designed to eliminate LONG quick failures
    which occur when SSL is flat but price temporarily crosses above baseline.

    Args:
        df: DataFrame with 'baseline' column
        index: Current candle index
        signal_type: "LONG" or "SHORT"
        lookback: Candles for slope calculation (default: 10)
        min_slope_pct: Minimum slope as percentage (default: 0.3%)
        return_debug: Return debug information

    Returns:
        (slope_supports_signal, debug_info)
        - True if slope supports signal direction, False otherwise
        - debug_info: Slope value and direction (if return_debug=True)

    Example:
        >>> ok, debug = check_ssl_slope_direction(df, -1, "LONG", return_debug=True)
        >>> if not ok:
        ...     print(f"SSL slope insufficient: {debug['slope_pct']:.4f}")
    """

    debug_info = {
        "slope_pct": None,
        "slope_direction": None,
        "required_direction": "UP" if signal_type == "LONG" else "DOWN",
        "passes": False,
    }

    if 'baseline' not in df.columns:
        return (False, debug_info) if return_debug else (False, None)

    abs_idx = index if index >= 0 else len(df) + index

    if abs_idx < lookback + 5:
        return (False, debug_info) if return_debug else (False, None)

    try:
        # Get baseline values over lookback period
        start_idx = abs_idx - lookback
        baseline_start = float(df['baseline'].iloc[start_idx])
        baseline_end = float(df['baseline'].iloc[abs_idx])

        # Calculate slope as percentage change
        slope_pct = (baseline_end - baseline_start) / baseline_start

        debug_info["slope_pct"] = slope_pct
        debug_info["slope_direction"] = "UP" if slope_pct > 0 else ("DOWN" if slope_pct < 0 else "FLAT")

        # Check if slope supports signal direction
        if signal_type == "LONG":
            # LONG requires positive slope >= min_slope_pct
            passes = slope_pct >= min_slope_pct
        else:  # SHORT
            # SHORT requires negative slope <= -min_slope_pct
            passes = slope_pct <= -min_slope_pct

        debug_info["passes"] = passes

        return (passes, debug_info) if return_debug else (passes, None)

    except (IndexError, ValueError, ZeroDivisionError):
        return (False, debug_info) if return_debug else (False, None)


# ==========================================
# PATTERN 4C: SSL BASELINE STABILITY CHECK (NEW - From Professional Analysis)
# ==========================================
# Analysis Evidence:
# - Quick failures occur in choppy markets where SSL oscillates
# - Stable baseline = clear trend = better trades
# - Expected to add $25-35/year

def check_ssl_stability(
        df: pd.DataFrame,
        index: int,
        lookback: int = 10,
        max_volatility: float = 0.005,  # Max 0.5% standard deviation
        return_debug: bool = False,
) -> Tuple[bool, Optional[Dict]]:
    """
    Check if SSL baseline is stable (not oscillating erratically).

    Key Insight from Analysis:
    - Erratic SSL baseline indicates choppy market
    - Stable SSL indicates clear trend
    - Filter out signals when SSL is too volatile

    Args:
        df: DataFrame with 'baseline' column
        index: Current candle index
        lookback: Candles to check for stability (default: 10)
        max_volatility: Maximum normalized std deviation (default: 0.5%)
        return_debug: Return debug information

    Returns:
        (is_stable, debug_info)
        - True if SSL is stable, False if erratic
        - debug_info: Volatility metrics (if return_debug=True)

    Example:
        >>> stable, debug = check_ssl_stability(df, -1, return_debug=True)
        >>> if not stable:
        ...     print(f"SSL too volatile: {debug['volatility_pct']:.4f}")
    """

    debug_info = {
        "volatility_pct": None,
        "is_stable": False,
    }

    if 'baseline' not in df.columns:
        return (False, debug_info) if return_debug else (False, None)

    abs_idx = index if index >= 0 else len(df) + index

    if abs_idx < lookback + 5:
        return (True, debug_info) if return_debug else (True, None)  # Default to stable if not enough data

    try:
        # Get baseline values over lookback period
        ssl_recent = df['baseline'].iloc[abs_idx - lookback:abs_idx + 1]

        # Calculate normalized volatility (std/mean)
        ssl_mean = ssl_recent.mean()
        ssl_std = ssl_recent.std()

        volatility_pct = ssl_std / ssl_mean if ssl_mean > 0 else 0

        debug_info["volatility_pct"] = volatility_pct
        is_stable = volatility_pct <= max_volatility
        debug_info["is_stable"] = is_stable

        return (is_stable, debug_info) if return_debug else (is_stable, None)

    except (IndexError, ValueError, ZeroDivisionError):
        return (True, debug_info) if return_debug else (True, None)


# ==========================================
# PATTERN 4D: QUICK FAILURE PREDICTOR (NEW - From Professional Analysis)
# ==========================================
# Analysis Evidence:
# - 57% of losses are "quick failures" (≤20 bars)
# - All 8 quick failures are LONG trades
# - Combined filter to catch potential quick failures

def predict_quick_failure(
        df: pd.DataFrame,
        index: int,
        signal_type: str,
        lookback: int = 10,
        return_debug: bool = False,
) -> Tuple[bool, Optional[Dict]]:
    """
    Predict if a signal is likely to be a "quick failure".

    A quick failure is a trade that hits SL within 20 bars.
    This combines multiple signals that correlate with quick failures:
    1. SSL slope doesn't support direction
    2. SSL baseline is erratic/unstable
    3. Recent SSL direction change (weak flip)

    Args:
        df: DataFrame with indicators
        index: Current candle index
        signal_type: "LONG" or "SHORT"
        lookback: Candles for analysis (default: 10)
        return_debug: Return debug information

    Returns:
        (is_likely_quick_failure, debug_info)
        - True if likely quick failure (DON'T TRADE)
        - debug_info: Risk factors (if return_debug=True)

    Example:
        >>> risky, debug = predict_quick_failure(df, -1, "LONG", return_debug=True)
        >>> if risky:
        ...     print(f"Quick failure risk: {debug['risk_factors']}")
    """

    debug_info = {
        "slope_ok": False,
        "stability_ok": False,
        "direction_established": False,
        "risk_factors": [],
        "risk_score": 0,
    }

    abs_idx = index if index >= 0 else len(df) + index

    if abs_idx < lookback + 10:
        return (False, debug_info) if return_debug else (False, None)

    try:
        risk_score = 0
        risk_factors = []

        # Check 1: SSL Slope Direction
        slope_ok, slope_debug = check_ssl_slope_direction(
            df, index, signal_type, lookback=lookback, min_slope_pct=0.002, return_debug=True
        )
        debug_info["slope_ok"] = slope_ok
        if not slope_ok:
            risk_score += 2  # High weight - this is the main predictor
            risk_factors.append(f"SSL slope wrong direction ({slope_debug.get('slope_pct', 0):.4f})")

        # Check 2: SSL Stability
        stable, stability_debug = check_ssl_stability(
            df, index, lookback=lookback, max_volatility=0.004, return_debug=True
        )
        debug_info["stability_ok"] = stable
        if not stable:
            risk_score += 1
            risk_factors.append(f"SSL unstable ({stability_debug.get('volatility_pct', 0):.4f})")

        # Check 3: Recent Direction Change (weak flip)
        # If SSL direction changed in last 3 candles, the flip may not be sustained
        if 'baseline' in df.columns:
            baseline_now = float(df['baseline'].iloc[abs_idx])
            baseline_3_ago = float(df['baseline'].iloc[abs_idx - 3])
            baseline_10_ago = float(df['baseline'].iloc[abs_idx - 10])

            recent_trend = baseline_now - baseline_3_ago
            prior_trend = baseline_3_ago - baseline_10_ago

            # If trends have opposite signs, there was a recent reversal
            if (recent_trend > 0 and prior_trend < 0) or (recent_trend < 0 and prior_trend > 0):
                debug_info["direction_established"] = False
                risk_score += 1
                risk_factors.append("Recent SSL direction change")
            else:
                debug_info["direction_established"] = True

        debug_info["risk_factors"] = risk_factors
        debug_info["risk_score"] = risk_score

        # Risk score >= 2 means likely quick failure
        is_risky = risk_score >= 2

        return (is_risky, debug_info) if return_debug else (is_risky, None)

    except (IndexError, ValueError, ZeroDivisionError):
        return (False, debug_info) if return_debug else (False, None)


# ==========================================
# PATTERN 7: SSL DYNAMIC SUPPORT
# ==========================================
# Real Trade Evidence:
# - NO 3: "SSL HYBRID bandı yukarıya doğru... SSL HYBRID dinamik bir destek gibi calisip fiyatı ileriye taşıyor"
# - NO 4: "SSL HYBRID dinamik bir support gibi davranarak fiyatı itmeye basliyor"
# - NO 10: "olusan SSL HYBRID bandını destek olarak kullanıp yükselmeye başlıyor"

def is_ssl_acting_as_dynamic_support(
        df: pd.DataFrame,
        index: int,
        lookback: int = 5,
        min_touches: int = 3,
        min_bounce_rate: float = 0.8,
        min_slope: float = 0.001,
        return_debug: bool = False,
) -> Tuple[bool, Optional[Dict]]:
    """
    Check if SSL baseline is actively supporting price.

    Pattern:
    - Price has touched SSL multiple times in recent candles
    - Each touch resulted in bounce (didn't break through)
    - SSL slope is positive (trending up)

    Args:
        df: DataFrame with baseline column
        index: Current candle index
        lookback: Candles to check for touches (default: 5)
        min_touches: Minimum touches required (default: 3)
        min_bounce_rate: Min bounce success rate (default: 80%)
        min_slope: Minimum SSL slope (default: 0.1%)
        return_debug: Return debug information

    Returns:
        (is_active_support, debug_info)
        - is_active_support: True if SSL is active support
        - debug_info: Diagnostic info (if return_debug=True)

    Example:
        >>> is_support, debug = is_ssl_acting_as_dynamic_support(df, -1, return_debug=True)
        >>> if is_support:
        ...     print("SSL is actively supporting price - good LONG setup")
    """

    debug_info = {
        "touches": 0,
        "bounces": 0,
        "bounce_rate": 0.0,
        "ssl_slope": None,
        "is_active": False,
    }

    if 'baseline' not in df.columns:
        return (False, debug_info) if return_debug else (False, None)

    abs_idx = index if index >= 0 else len(df) + index

    if abs_idx < lookback + 5:
        return (False, debug_info) if return_debug else (False, None)

    try:
        closes = df['close'].values
        lows = df['low'].values
        highs = df['high'].values
        baselines = df['baseline'].values

        # Count touches in lookback period
        touches = 0
        bounces = 0

        for i in range(abs_idx - lookback, abs_idx + 1):
            baseline = baselines[i]
            low = lows[i]
            close = closes[i]

            # Did price touch SSL? (within 0.3%)
            if low <= baseline * 1.003:
                touches += 1
                # Did it bounce? (close above baseline)
                if close > baseline:
                    bounces += 1

        debug_info["touches"] = touches
        debug_info["bounces"] = bounces
        bounce_rate = bounces / touches if touches > 0 else 0
        debug_info["bounce_rate"] = bounce_rate

        # SSL is active support if enough touches and high bounce rate
        if touches < min_touches or bounce_rate < min_bounce_rate:
            return (False, debug_info) if return_debug else (False, None)

        # Check SSL slope (should be rising for support)
        ssl_slope = (baselines[abs_idx] - baselines[abs_idx - lookback]) / baselines[abs_idx - lookback]
        debug_info["ssl_slope"] = ssl_slope

        is_active = ssl_slope > min_slope
        debug_info["is_active"] = is_active

        return (is_active, debug_info) if return_debug else (is_active, None)

    except (IndexError, ValueError, ZeroDivisionError):
        return (False, debug_info) if return_debug else (False, None)
