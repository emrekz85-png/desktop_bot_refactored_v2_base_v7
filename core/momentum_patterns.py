# core/momentum_patterns.py
"""
Momentum Pattern Detection - Based on Real Trade Visual Analysis

This module detects momentum exhaustion patterns from user's real trades:
- NO1, NO2, NO5: Stairstepping → Sharp Break → Fakeout → SSL Sideways → Entry

Visual measurements taken from actual trade screenshots (2026-01-05).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


# ============================================================================
# THRESHOLDS (Adjusted Based on Real Data - 2026-01-05)
# ============================================================================
# Original thresholds from visual analysis were too strict (0 patterns found).
# Adjusted to capture real patterns with quality-based filtering.

# Stairstepping Phase - ADJUSTED FOR 15m TF (everything faster!)
STAIRSTEPPING_LOOKBACK = 10          # bars (shorter for 15m)
STAIRSTEPPING_MIN_CONSISTENCY = 0.50  # 50% of bars blue > red (more permissive)
STAIRSTEPPING_MIN_STEPPING = 0.40     # 40% of steps upward (more permissive)
STAIRSTEPPING_MIN_SLOPE = 0.0003      # 0.3% per 10 bars (lower for 15m)

# Quality thresholds for stairstepping
STAIRSTEPPING_EXCELLENT_CONSISTENCY = 0.85  # NO1, NO2, NO5 level
STAIRSTEPPING_GOOD_CONSISTENCY = 0.70       # Good but not perfect
STAIRSTEPPING_MODERATE_CONSISTENCY = 0.60   # Acceptable minimum

# Sharp Break Phase - ADJUSTED (was 0.015, now 0.003)
SHARP_BREAK_MIN_GAP = 0.003           # 0.3% gap (any visible cross)
SHARP_BREAK_MAX_BARS = 5              # Must happen in 1-5 bars
SHARP_BREAK_LOOKBACK = 10             # Check last 10 bars

# Quality thresholds for sharp break
SHARP_BREAK_EXCELLENT_GAP = 0.015     # 1.5%+ gap = excellent (NO1, NO2 level)
SHARP_BREAK_GOOD_GAP = 0.008          # 0.8%+ gap = good
SHARP_BREAK_MODERATE_GAP = 0.003      # 0.3%+ gap = moderate minimum

# Fakeout Rally Phase - ADJUSTED FOR 15m TF
FAKEOUT_MIN_BOUNCE = 0.002            # 0.2% minimum bounce (15m: very small moves)
FAKEOUT_MAX_DURATION = 10             # Up to 10 bars (15m: 2.5 hours)
FAKEOUT_CHECK_AFTER_BREAK = 10        # Check 10 bars after break

# SSL Sideways Phase - UNCHANGED (387/500 bars passed, working well)
SSL_SIDEWAYS_LOOKBACK = 8             # bars
SSL_SIDEWAYS_MAX_SLOPE = 0.0008       # 0.08% per bar (nearly flat)


# ============================================================================
# PHASE 1: STAIRSTEPPING DETECTION
# ============================================================================

def detect_at_stairstepping(
    df: pd.DataFrame,
    index: int,
    lookback: int = STAIRSTEPPING_LOOKBACK,
    signal_type: str = "SHORT"
) -> Tuple[bool, Dict]:
    """
    Detect AlphaTrend "merdiven şekli" (stairstepping pattern).

    Visual Pattern (SHORT):
    - Blue line consistently above red line
    - Blue line stepping upward (each bar higher than previous)
    - Clear upward trend with minimal retracements

    Returns quality level based on consistency:
    - EXCELLENT: 85%+ (NO1, NO2, NO5 level)
    - GOOD: 70%+
    - MODERATE: 60%+
    - POOR: <60%

    Args:
        df: DataFrame with AlphaTrend columns
        index: Current bar index
        lookback: Bars to check (default 12)
        signal_type: "SHORT" or "LONG"

    Returns:
        (is_stairstepping, details_dict)
    """
    details = {
        'consistency': 0.0,
        'stepping_ratio': 0.0,
        'slope': 0.0,
        'bars_checked': 0,
        'quality': 'NONE'
    }

    # Required columns
    if 'alphatrend' not in df.columns or 'alphatrend_2' not in df.columns:
        return False, details

    start_idx = max(0, index - lookback)
    if start_idx >= index:
        return False, details

    at_buyers = df['alphatrend'].values[start_idx:index]
    at_sellers = df['alphatrend_2'].values[start_idx:index]

    if len(at_buyers) < 5:  # Need minimum data
        return False, details

    details['bars_checked'] = len(at_buyers)

    if signal_type == "SHORT":
        # Phase 1A: Blue consistently above red
        buyers_above = at_buyers > at_sellers
        consistency = np.mean(buyers_above)
        details['consistency'] = consistency

        if consistency < STAIRSTEPPING_MIN_CONSISTENCY:
            return False, details

        # Phase 1B: Blue stepping upward
        steps = np.diff(at_buyers)
        steps_up = steps >= 0  # Allow flat (consolidation)
        stepping_ratio = np.mean(steps_up)
        details['stepping_ratio'] = stepping_ratio

        if stepping_ratio < STAIRSTEPPING_MIN_STEPPING:
            return False, details

        # Phase 1C: Calculate slope (trend strength)
        x = np.arange(len(at_buyers))
        slope, _ = np.polyfit(x, at_buyers, 1)
        avg_price = np.mean(at_buyers)
        slope_pct = slope / avg_price if avg_price > 0 else 0
        details['slope'] = slope_pct

        if slope_pct < STAIRSTEPPING_MIN_SLOPE:
            return False, details

        # Determine quality level based on consistency
        if consistency >= STAIRSTEPPING_EXCELLENT_CONSISTENCY:
            details['quality'] = 'EXCELLENT'
        elif consistency >= STAIRSTEPPING_GOOD_CONSISTENCY:
            details['quality'] = 'GOOD'
        else:
            details['quality'] = 'MODERATE'

        return True, details

    else:  # LONG
        # Sellers (red) above buyers (blue), stepping downward
        sellers_above = at_sellers > at_buyers
        consistency = np.mean(sellers_above)
        details['consistency'] = consistency

        if consistency < STAIRSTEPPING_MIN_CONSISTENCY:
            return False, details

        steps = np.diff(at_sellers)
        steps_down = steps <= 0
        stepping_ratio = np.mean(steps_down)
        details['stepping_ratio'] = stepping_ratio

        if stepping_ratio < STAIRSTEPPING_MIN_STEPPING:
            return False, details

        x = np.arange(len(at_sellers))
        slope, _ = np.polyfit(x, at_sellers, 1)
        avg_price = np.mean(at_sellers)
        slope_pct = abs(slope / avg_price) if avg_price > 0 else 0
        details['slope'] = slope_pct

        if slope_pct < STAIRSTEPPING_MIN_SLOPE:
            return False, details

        # Determine quality level
        if consistency >= STAIRSTEPPING_EXCELLENT_CONSISTENCY:
            details['quality'] = 'EXCELLENT'
        elif consistency >= STAIRSTEPPING_GOOD_CONSISTENCY:
            details['quality'] = 'GOOD'
        else:
            details['quality'] = 'MODERATE'

        return True, details


# ============================================================================
# PHASE 2: SHARP SELLOFF DETECTION (Price-Based, Not AlphaTrend)
# ============================================================================
# REVISION (2026-01-05): AlphaTrend is too slow (lagging indicator).
# Sharp selloff should be detected from PRICE action directly.

def detect_sharp_selloff(
    df: pd.DataFrame,
    index: int,
    lookback: int = 8,  # 15m TF: Look back 8 bars (2 hours)
    signal_type: str = "SHORT"
) -> Tuple[bool, Dict]:
    """
    Detect sharp price selloff (momentum break).

    NEW APPROACH: Direct price action, not AlphaTrend crossover.
    AlphaTrend is too slow - by the time it crosses, selloff is over.

    Visual Pattern (SHORT):
    - Strong downward price movement in 1-3 bars
    - At least 2-3% drop from recent high
    - Large red candles with momentum

    Args:
        df: DataFrame
        index: Current bar
        lookback: Bars to check for recent high
        signal_type: "SHORT" or "LONG"

    Returns:
        (sharp_selloff_detected, details_dict)
    """
    details = {
        'selloff_found': False,
        'selloff_index': None,
        'drop_pct': 0.0,
        'bars_since_selloff': None,
        'candle_strength': 0.0,
        'quality': 'NONE'
    }

    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    closes = df['close'].values

    start_idx = max(0, index - lookback)

    if signal_type == "SHORT":
        # Find recent high (before selloff)
        recent_high_idx = start_idx + np.argmax(highs[start_idx:index])
        recent_high = highs[recent_high_idx]

        # Current or recent low
        current_low = lows[index]

        # Calculate drop from recent high
        drop_pct = (recent_high - current_low) / recent_high

        details['drop_pct'] = drop_pct

        # ADJUSTED: Real data shows max 1.54% drop in 500 bars (15m TF)
        # For 15m TF: Even 1% is rare. Be more permissive.
        # Minimum drop: 0.7% (MODERATE), 1.0% (GOOD), 1.3% (EXCELLENT)
        if drop_pct < 0.007:  # 0.7% minimum
            return False, details

        # Check if drop was SHARP (happened in 1-8 bars for 15m TF)
        bars_to_drop = index - recent_high_idx

        if bars_to_drop > 8:  # 15m: Allow up to 8 bars (2 hours)
            return False, details

        # Calculate candle strength (body size relative to range)
        # Large red candles = strong selloff
        candle_bodies = []
        for i in range(recent_high_idx, index + 1):
            body = abs(closes[i] - opens[i])
            candle_range = highs[i] - lows[i]
            if candle_range > 0:
                candle_bodies.append(body / candle_range)

        avg_body_strength = np.mean(candle_bodies) if candle_bodies else 0
        details['candle_strength'] = avg_body_strength

        # Sharp selloff confirmed
        details['selloff_found'] = True
        details['selloff_index'] = recent_high_idx
        details['bars_since_selloff'] = bars_to_drop

        # Quality based on drop size (ADJUSTED for 15m TF)
        if drop_pct >= 0.013:  # 1.3%+ drop = EXCELLENT (rare!)
            details['quality'] = 'EXCELLENT'
        elif drop_pct >= 0.010:  # 1.0%+ drop = GOOD
            details['quality'] = 'GOOD'
        else:  # 0.7%+ drop = MODERATE
            details['quality'] = 'MODERATE'

        return True, details

    else:  # LONG - sharp rally
        # Find recent low
        recent_low_idx = start_idx + np.argmin(lows[start_idx:index])
        recent_low = lows[recent_low_idx]

        # Current or recent high
        current_high = highs[index]

        # Calculate rally from recent low
        rally_pct = (current_high - recent_low) / recent_low

        details['drop_pct'] = rally_pct  # Using same key

        if rally_pct < 0.007:  # Same as SHORT: 0.7% minimum
            return False, details

        # Check if rally was SHARP
        bars_to_rally = index - recent_low_idx

        if bars_to_rally > 8:  # Same as SHORT (15m TF)
            return False, details

        # Candle strength (green candles)
        candle_bodies = []
        for i in range(recent_low_idx, index + 1):
            body = abs(closes[i] - opens[i])
            candle_range = highs[i] - lows[i]
            if candle_range > 0:
                candle_bodies.append(body / candle_range)

        avg_body_strength = np.mean(candle_bodies) if candle_bodies else 0
        details['candle_strength'] = avg_body_strength

        details['selloff_found'] = True
        details['selloff_index'] = recent_low_idx
        details['bars_since_selloff'] = bars_to_rally

        # Quality (same as SHORT)
        if rally_pct >= 0.013:
            details['quality'] = 'EXCELLENT'
        elif rally_pct >= 0.010:
            details['quality'] = 'GOOD'
        else:
            details['quality'] = 'MODERATE'

        return True, details


# ============================================================================
# PHASE 3: FAKEOUT RALLY DETECTION
# ============================================================================

def detect_fakeout_rally(
    df: pd.DataFrame,
    index: int,
    break_index: Optional[int] = None,
    signal_type: str = "SHORT"
) -> Tuple[bool, Dict]:
    """
    Detect fakeout rally = price touched SSL baseline after sharp break.

    REVISED (2026-01-05): Instead of "bounce percentage", check SSL baseline touch.
    This aligns with SSL Flow strategy - baseline touch = entry signal.

    Visual Pattern (SHORT):
    - After selloff, price bounces back UP
    - Price touches SSL baseline (resistance)
    - This is the entry opportunity

    Args:
        df: DataFrame
        index: Current bar
        break_index: Index where sharp break occurred (if known)
        signal_type: "SHORT" or "LONG"

    Returns:
        (fakeout_detected, details_dict)
    """
    details = {
        'fakeout_found': False,
        'baseline_touched': False,
        'touch_distance_pct': 0.0,
        'duration': 0
    }

    if 'baseline' not in df.columns:
        return False, details

    # If break_index not provided, search for it
    if break_index is None:
        is_selloff, selloff_details = detect_sharp_selloff(df, index, signal_type=signal_type)
        if not is_selloff:
            return False, details
        break_index = selloff_details['selloff_index']

    if break_index is None or break_index >= index:
        return False, details

    bars_since_break = index - break_index

    # Fakeout must happen within max duration
    if bars_since_break > FAKEOUT_MAX_DURATION:
        return False, details

    details['duration'] = bars_since_break

    highs = df['high'].values
    lows = df['low'].values
    baseline = df['baseline'].values

    # SSL touch tolerance (0.3% - same as SSL Flow strategy)
    ssl_touch_tolerance = 0.003

    if signal_type == "SHORT":
        # After selloff, did price bounce up and touch baseline?
        current_high = highs[index]
        current_baseline = baseline[index]

        # Check if high touched or exceeded baseline
        touch_distance = (current_baseline - current_high) / current_baseline
        details['touch_distance_pct'] = touch_distance

        # Touched if within tolerance OR exceeded
        baseline_touched = current_high >= current_baseline * (1 - ssl_touch_tolerance)
        details['baseline_touched'] = baseline_touched

        if not baseline_touched:
            return False, details

        details['fakeout_found'] = True
        return True, details

    else:  # LONG
        # After rally down, did price drop and touch baseline?
        current_low = lows[index]
        current_baseline = baseline[index]

        # Check if low touched or went below baseline
        touch_distance = (current_low - current_baseline) / current_baseline
        details['touch_distance_pct'] = touch_distance

        # Touched if within tolerance OR below
        baseline_touched = current_low <= current_baseline * (1 + ssl_touch_tolerance)
        details['baseline_touched'] = baseline_touched

        if not baseline_touched:
            return False, details

        details['fakeout_found'] = True
        return True, details


# ============================================================================
# PHASE 4: SSL SLOPE DIRECTION (Revised 2026-01-05)
# ============================================================================
# OLD: SSL must be "sideways" (nearly flat) - too permissive (85% pass rate)
# NEW: SSL must be turning in signal direction
#      - SHORT: SSL slope negative (turning down)
#      - LONG: SSL slope positive (turning up)

# Minimum slope to confirm direction (not just flat noise)
SSL_MIN_SLOPE_FOR_DIRECTION = 0.0001  # 0.01% per bar minimum

def detect_ssl_sideways(
    df: pd.DataFrame,
    index: int,
    lookback: int = SSL_SIDEWAYS_LOOKBACK,
    signal_type: str = "SHORT"
) -> Tuple[bool, Dict]:
    """
    Detect SSL HYBRID slope direction matching signal.

    REVISED (2026-01-05): Instead of "sideways", check slope direction.
    - SHORT: SSL should be turning DOWN (negative slope)
    - LONG: SSL should be turning UP (positive slope)

    This indicates momentum exhaustion - SSL is no longer supporting
    the previous trend direction.

    Args:
        df: DataFrame with 'baseline' column
        index: Current bar
        lookback: Bars to check
        signal_type: "SHORT" or "LONG"

    Returns:
        (slope_matches_signal, details_dict)
    """
    details = {
        'is_sideways': False,  # Keep for backward compat
        'slope_direction_ok': False,
        'slope_pct': 0.0,
        'slope_raw': 0.0,
        'direction': 'FLAT'
    }

    if 'baseline' not in df.columns:
        return False, details

    start_idx = max(0, index - lookback)
    if start_idx >= index:
        return False, details

    baseline = df['baseline'].values[start_idx:index + 1]

    if len(baseline) < 3:
        return False, details

    # Calculate linear regression slope
    x = np.arange(len(baseline))
    slope, _ = np.polyfit(x, baseline, 1)

    # Normalize by price
    avg_baseline = np.mean(baseline)
    slope_pct = slope / avg_baseline if avg_baseline > 0 else 0

    details['slope_pct'] = slope_pct
    details['slope_raw'] = slope

    # Determine direction
    if slope_pct > SSL_MIN_SLOPE_FOR_DIRECTION:
        details['direction'] = 'UP'
    elif slope_pct < -SSL_MIN_SLOPE_FOR_DIRECTION:
        details['direction'] = 'DOWN'
    else:
        details['direction'] = 'FLAT'

    # Check if slope matches signal direction - STRICT: must be turning, not flat
    if signal_type == "SHORT":
        # For SHORT: SSL MUST be turning DOWN (negative slope)
        slope_ok = slope_pct < -SSL_MIN_SLOPE_FOR_DIRECTION
    else:  # LONG
        # For LONG: SSL MUST be turning UP (positive slope)
        slope_ok = slope_pct > SSL_MIN_SLOPE_FOR_DIRECTION

    details['slope_direction_ok'] = slope_ok
    details['is_sideways'] = slope_ok  # Backward compat

    return slope_ok, details


# ============================================================================
# FULL PATTERN DETECTOR
# ============================================================================

def detect_momentum_exhaustion_pattern(
    df: pd.DataFrame,
    index: int,
    signal_type: str = "SHORT",
    require_all_phases: bool = False
) -> Dict:
    """
    Detect complete momentum exhaustion pattern.

    Full Pattern (from user's real trades NO1, NO2, NO5):
    1. Stairstepping: AT merdiven şeklinde yükseliş (12 bars)
    2. Sharp Break: Ani kırılım (1-2 bars)
    3. Fakeout Rally: Yukarı fakeout (2-6 bars)
    4. SSL Sideways: SSL yana doğru (concurrent)

    Args:
        df: DataFrame
        index: Current bar
        signal_type: "SHORT" or "LONG"
        require_all_phases: If True, all 4 phases must pass

    Returns:
        dict: Pattern detection results
    """
    result = {
        'pattern_detected': False,
        'signal_type': signal_type,
        'phases': {
            'stairstepping': False,
            'sharp_break': False,
            'fakeout': False,
            'ssl_sideways': False
        },
        'details': {},
        'confidence': 0.0,
        'quality': 'NONE'
    }

    # PHASE 1: Stairstepping (10-15 bars ago)
    stairstepping_check_idx = index - 3  # Check before current bar
    is_stairstepping, stair_details = detect_at_stairstepping(
        df, stairstepping_check_idx,
        lookback=STAIRSTEPPING_LOOKBACK,
        signal_type=signal_type
    )
    result['phases']['stairstepping'] = is_stairstepping
    result['details']['stairstepping'] = stair_details

    # PHASE 2: Sharp Selloff (1-8 bars ago) - PRICE-BASED, NOT ALPHATREND
    is_selloff, selloff_details = detect_sharp_selloff(
        df, index,
        lookback=8,  # 15m TF: 2 hours lookback
        signal_type=signal_type
    )
    result['phases']['sharp_break'] = is_selloff  # Keep key name for compatibility
    result['details']['sharp_break'] = selloff_details

    # PHASE 3: Fakeout Rally (current bar)
    is_fakeout, fakeout_details = detect_fakeout_rally(
        df, index,
        break_index=selloff_details.get('selloff_index'),  # Changed from break_index
        signal_type=signal_type
    )
    result['phases']['fakeout'] = is_fakeout
    result['details']['fakeout'] = fakeout_details

    # PHASE 4: SSL Slope Direction (current) - Revised 2026-01-05
    is_sideways, ssl_details = detect_ssl_sideways(
        df, index,
        lookback=SSL_SIDEWAYS_LOOKBACK,
        signal_type=signal_type
    )
    result['phases']['ssl_sideways'] = is_sideways
    result['details']['ssl_sideways'] = ssl_details

    # Calculate confidence
    phases_passed = sum([
        is_stairstepping,
        is_selloff,  # Changed from is_break
        is_fakeout,
        is_sideways
    ])

    # Weighted confidence (sharp selloff is most critical)
    confidence = (
        is_stairstepping * 0.20 +  # Nice to have
        is_selloff * 0.40 +         # CRITICAL (changed from is_break)
        is_fakeout * 0.20 +          # Confirms pattern
        is_sideways * 0.20           # Timing signal
    )

    result['confidence'] = confidence

    # Pattern detected if:
    # - All 4 phases (if require_all_phases=True)
    # - OR sharp selloff + at least 1 other (ADJUSTED - was 2, too strict)
    if require_all_phases:
        result['pattern_detected'] = phases_passed == 4
    else:
        result['pattern_detected'] = is_selloff and phases_passed >= 2  # LOWERED from 3 to 2

    # Quality assessment
    if phases_passed == 4:
        result['quality'] = 'EXCELLENT'
    elif phases_passed == 3 and is_selloff:
        result['quality'] = 'GOOD'
    elif phases_passed == 2 and is_selloff:
        result['quality'] = 'MODERATE'
    else:
        result['quality'] = 'POOR'

    return result
