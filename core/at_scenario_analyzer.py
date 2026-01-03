"""
AlphaTrend Scenario Analyzer

Purpose: Analyze every potential trade with multiple AT configurations
to find the optimal AT settings through data-driven analysis.

This module implements the "What-If" analysis approach:
1. Find ALL potential signals (AT-independent core signals)
2. For each signal, test multiple AT configurations
3. Record AT state and configuration outcomes
4. Simulate trades and record PnL for each scenario
5. Enable Optuna optimization over AT parameters

Created: 2026-01-03
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# AT SCENARIO DEFINITIONS
# ============================================================================

@dataclass
class ATScenario:
    """Definition of an AlphaTrend configuration scenario."""
    name: str
    at_active: bool
    at_mode: str  # "off", "binary", "score", "veto"
    skip_at_flat_filter: bool
    use_ssl_flip_grace: bool
    ssl_flip_grace_bars: int
    at_score_threshold: Optional[float]  # Only for score mode
    regime_filter: str = "off"  # "off", "aligned", "veto"

    def to_config(self) -> Dict:
        """Convert scenario to config dict."""
        return {
            "at_active": self.at_active,
            "at_mode": self.at_mode,
            "skip_at_flat_filter": self.skip_at_flat_filter,
            "use_ssl_flip_grace": self.use_ssl_flip_grace,
            "ssl_flip_grace_bars": self.ssl_flip_grace_bars,
            "regime_filter": self.regime_filter,
        }


# Predefined scenarios to test
AT_SCENARIOS = [
    # Scenario 1: AT completely OFF
    ATScenario(
        name="at_off",
        at_active=False,
        at_mode="off",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=None,
    ),

    # Scenario 2: AT Binary - Strict (current default)
    ATScenario(
        name="at_binary_strict",
        at_active=True,
        at_mode="binary",
        skip_at_flat_filter=False,  # ENFORCE flat check
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=None,
    ),

    # Scenario 3: AT Binary - Lax (skip flat check)
    ATScenario(
        name="at_binary_lax",
        at_active=True,
        at_mode="binary",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=None,
    ),

    # Scenario 4-6: AT Binary with grace period
    ATScenario(
        name="at_binary_grace_2",
        at_active=True,
        at_mode="binary",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=True,
        ssl_flip_grace_bars=2,
        at_score_threshold=None,
    ),
    ATScenario(
        name="at_binary_grace_3",
        at_active=True,
        at_mode="binary",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=True,
        ssl_flip_grace_bars=3,
        at_score_threshold=None,
    ),
    ATScenario(
        name="at_binary_grace_5",
        at_active=True,
        at_mode="binary",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=True,
        ssl_flip_grace_bars=5,
        at_score_threshold=None,
    ),

    # Scenario 7-9: AT Score mode with different thresholds
    ATScenario(
        name="at_score_1.0",
        at_active=True,
        at_mode="score",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=1.0,
    ),
    ATScenario(
        name="at_score_1.5",
        at_active=True,
        at_mode="score",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=1.5,
    ),
    ATScenario(
        name="at_score_2.0",
        at_active=True,
        at_mode="score",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=2.0,
    ),

    # =========================================================================
    # VETO MODE SCENARIOS (NEW - Only block when AT actively opposes)
    # =========================================================================
    # Veto Logic:
    # - LONG + AT sellers_dominant = BLOCK (opposing)
    # - SHORT + AT buyers_dominant = BLOCK (opposing)
    # - All other cases (aligned, flat, neutral) = ALLOW
    #
    # This is less restrictive than binary mode and should allow more trades
    # while still blocking the most dangerous signals (AT actively opposing)

    # Scenario 10: AT Veto Only (basic)
    ATScenario(
        name="at_veto_only",
        at_active=True,
        at_mode="veto",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=None,
    ),

    # Scenario 11: AT Veto + Grace Period (allow recent SSL flips)
    ATScenario(
        name="at_veto_grace_3",
        at_active=True,
        at_mode="veto",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=True,
        ssl_flip_grace_bars=3,
        at_score_threshold=None,
    ),

    # Scenario 12: AT Veto + Grace Period (5 bars)
    ATScenario(
        name="at_veto_grace_5",
        at_active=True,
        at_mode="veto",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=True,
        ssl_flip_grace_bars=5,
        at_score_threshold=None,
    ),

    # =========================================================================
    # REGIME FILTER SCENARIOS
    # =========================================================================
    # Based on ultra_minimal_trades analysis:
    # - ALL WINS: SHORT + bearish_regime
    # - ALL LOSSES: LONG + bearish/neutral regime
    #
    # Regime filter modes:
    # - "aligned": Only allow signals that MATCH regime direction
    #   - LONG only in bullish_regime
    #   - SHORT only in bearish_regime
    #   - neutral_regime: blocks both (conservative)
    #
    # - "veto": Only BLOCK signals that OPPOSE regime direction
    #   - LONG blocked in bearish_regime
    #   - SHORT blocked in bullish_regime
    #   - neutral_regime: allows both (lenient)

    # Scenario 13: Regime Aligned Only (strictest regime filter)
    ATScenario(
        name="regime_aligned_only",
        at_active=False,  # AT off, only regime filter
        at_mode="off",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=None,
        regime_filter="aligned",
    ),

    # Scenario 14: Regime Veto Only (lenient regime filter)
    ATScenario(
        name="regime_veto_only",
        at_active=False,  # AT off, only regime filter
        at_mode="off",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=None,
        regime_filter="veto",
    ),

    # Scenario 15: AT Binary + Regime Aligned (double filter)
    ATScenario(
        name="at_binary_regime_aligned",
        at_active=True,
        at_mode="binary",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=None,
        regime_filter="aligned",
    ),

    # Scenario 16: AT Binary + Regime Veto (AT alignment + regime veto)
    ATScenario(
        name="at_binary_regime_veto",
        at_active=True,
        at_mode="binary",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=None,
        regime_filter="veto",
    ),

    # Scenario 17: AT Veto + Regime Veto (double veto - lenient)
    ATScenario(
        name="at_veto_regime_veto",
        at_active=True,
        at_mode="veto",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=None,
        regime_filter="veto",
    ),

    # Scenario 18: Regime Veto + Neutral Allowed Long (special case)
    # Based on user's note: "bazen düşük TF'lerde rejime ters dikkatli girerim"
    # This allows LONGs in neutral regime but blocks in bearish
    ATScenario(
        name="regime_veto_neutral_long",
        at_active=False,
        at_mode="off",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=None,
        regime_filter="veto_neutral_long",  # Special: veto + allow LONG in neutral
    ),

    # =========================================================================
    # NEUTRAL AVOIDANCE SCENARIOS (Based on analysis: neutral has 19.7% win rate!)
    # =========================================================================

    # Scenario 19: Skip Neutral Regime Only (allow opposing regimes)
    # Key insight: Neutral regime is the real problem, not opposing regimes
    ATScenario(
        name="skip_neutral_only",
        at_active=False,
        at_mode="off",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=None,
        regime_filter="skip_neutral",  # NEW: Skip only neutral regime
    ),

    # Scenario 20: AT Binary + Skip Neutral
    ATScenario(
        name="at_binary_skip_neutral",
        at_active=True,
        at_mode="binary",
        skip_at_flat_filter=True,
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=None,
        regime_filter="skip_neutral",
    ),

    # Scenario 21: AT Strict + Skip Neutral
    ATScenario(
        name="at_strict_skip_neutral",
        at_active=True,
        at_mode="binary",
        skip_at_flat_filter=False,  # Strict flat check
        use_ssl_flip_grace=False,
        ssl_flip_grace_bars=0,
        at_score_threshold=None,
        regime_filter="skip_neutral",
    ),
]


# ============================================================================
# AT STATE EXTRACTOR
# ============================================================================

@dataclass
class ATState:
    """Complete state of AlphaTrend at a given point."""
    at_buyers_dominant: bool
    at_sellers_dominant: bool
    at_is_flat: bool
    at_value: float
    at_upper: float
    at_lower: float
    line_separation: float  # (upper - lower) / close
    bars_since_cross: int
    bars_since_flip: int  # Bars since AT flipped direction
    trend_direction: str  # "up", "down", "flat"
    ssl_aligned: bool  # Does AT agree with SSL direction?
    at_regime: str  # "bullish_regime", "bearish_regime", "neutral_regime"

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame."""
        return {
            "at_buyers_dominant": self.at_buyers_dominant,
            "at_sellers_dominant": self.at_sellers_dominant,
            "at_is_flat": self.at_is_flat,
            "at_value": self.at_value,
            "at_upper": self.at_upper,
            "at_lower": self.at_lower,
            "at_line_separation": self.line_separation,
            "at_bars_since_cross": self.bars_since_cross,
            "at_bars_since_flip": self.bars_since_flip,
            "at_trend_direction": self.trend_direction,
            "at_ssl_aligned": self.ssl_aligned,
            "at_regime": self.at_regime,
        }


def extract_at_state(
    df: pd.DataFrame,
    index: int,
    signal_direction: str,  # "LONG" or "SHORT"
) -> ATState:
    """
    Extract complete AlphaTrend state at a given index.

    Args:
        df: DataFrame with indicator columns
        index: Index to analyze
        signal_direction: The SSL signal direction

    Returns:
        ATState with all AT metrics
    """
    if len(df) < 10:
        return ATState(
            at_buyers_dominant=False,
            at_sellers_dominant=False,
            at_is_flat=True,
            at_value=0,
            at_upper=0,
            at_lower=0,
            line_separation=0,
            bars_since_cross=999,
            bars_since_flip=999,
            trend_direction="flat",
            ssl_aligned=False,
            at_regime="neutral_regime",
        )

    abs_index = index if index >= 0 else (len(df) + index)
    curr = df.iloc[abs_index]
    close = float(curr["close"])

    # Basic AT state
    at_buyers = bool(curr.get("at_buyers_dominant", False))
    at_sellers = bool(curr.get("at_sellers_dominant", False))
    at_flat = bool(curr.get("at_is_flat", False))

    # AT values
    at_value = float(curr.get("alpha_trend", 0))
    at_upper = float(curr.get("at_upper", at_value))
    at_lower = float(curr.get("at_lower", at_value))

    # Line separation (normalized)
    if close > 0 and at_upper > 0 and at_lower > 0:
        line_separation = abs(at_upper - at_lower) / close
    else:
        line_separation = 0.0

    # Bars since AT crossed price
    bars_since_cross = 0
    for lookback in range(1, min(50, abs_index + 1)):
        prev_idx = abs_index - lookback
        if prev_idx < 0:
            break
        prev_at = float(df.iloc[prev_idx].get("alpha_trend", 0))
        prev_close = float(df.iloc[prev_idx]["close"])
        curr_at = float(df.iloc[prev_idx + 1].get("alpha_trend", 0))
        curr_close = float(df.iloc[prev_idx + 1]["close"])

        # Check if crossed
        if (prev_at > prev_close and curr_at < curr_close) or \
           (prev_at < prev_close and curr_at > curr_close):
            bars_since_cross = lookback
            break
    else:
        bars_since_cross = 50

    # Bars since AT flipped (buyers/sellers dominance changed)
    bars_since_flip = 0
    curr_buyers = at_buyers
    for lookback in range(1, min(50, abs_index + 1)):
        prev_idx = abs_index - lookback
        if prev_idx < 0:
            break
        prev_buyers = bool(df.iloc[prev_idx].get("at_buyers_dominant", False))
        if prev_buyers != curr_buyers:
            bars_since_flip = lookback
            break
    else:
        bars_since_flip = 50

    # Trend direction
    if at_buyers:
        trend_direction = "up"
    elif at_sellers:
        trend_direction = "down"
    else:
        trend_direction = "flat"

    # SSL alignment
    if signal_direction == "LONG":
        ssl_aligned = at_buyers
    elif signal_direction == "SHORT":
        ssl_aligned = at_sellers
    else:
        ssl_aligned = False

    # Calculate regime over lookback period
    # (at_regime column may not be pre-calculated, so we calculate it here)
    regime_lookback = 20
    if "at_regime" in df.columns and not pd.isna(curr.get("at_regime")):
        at_regime = str(curr.get("at_regime"))
    else:
        # Calculate regime based on AT dominance over lookback
        if abs_index >= regime_lookback - 1:
            start_idx = abs_index - regime_lookback + 1
            end_idx = abs_index + 1
            buyers_bars = df["at_buyers_dominant"].iloc[start_idx:end_idx].sum()
            sellers_bars = df["at_sellers_dominant"].iloc[start_idx:end_idx].sum()

            regime_threshold = 0.6  # 60% dominance required
            buyers_ratio = buyers_bars / regime_lookback
            sellers_ratio = sellers_bars / regime_lookback

            if buyers_ratio >= regime_threshold:
                at_regime = "bullish_regime"
            elif sellers_ratio >= regime_threshold:
                at_regime = "bearish_regime"
            else:
                at_regime = "neutral_regime"
        else:
            at_regime = "neutral_regime"

    return ATState(
        at_buyers_dominant=at_buyers,
        at_sellers_dominant=at_sellers,
        at_is_flat=at_flat,
        at_value=at_value,
        at_upper=at_upper,
        at_lower=at_lower,
        line_separation=line_separation,
        bars_since_cross=bars_since_cross,
        bars_since_flip=bars_since_flip,
        trend_direction=trend_direction,
        ssl_aligned=ssl_aligned,
        at_regime=at_regime,
    )


# ============================================================================
# CORE SIGNAL DETECTOR (AT-INDEPENDENT)
# ============================================================================

def check_core_signal(
    df: pd.DataFrame,
    index: int,
    min_rr: float = 1.2,  # Lower threshold for more signals
    min_pbema_distance: float = 0.002,  # Relaxed
    require_baseline_touch: bool = False,  # NEW: Require baseline touch
    baseline_touch_lookback: int = 5,  # Bars to look back for touch
) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], str]:
    """
    Check for CORE signal WITHOUT AlphaTrend filtering.

    This detects the base SSL + PBEMA signals that would then
    be filtered by various AT configurations.

    Core conditions (MINIMAL):
    1. Price vs SSL baseline (direction)
    2. PBEMA target exists (path)
    3. Basic RR check
    4. [Optional] Baseline touch in recent candles

    NO AlphaTrend check here!

    Returns:
        (signal_type, entry, tp, sl, reason) or (None, None, None, None, reason)
    """
    if df is None or df.empty or len(df) < 60:
        return None, None, None, None, "No Data"

    # Required columns
    required = ["open", "high", "low", "close", "baseline", "pb_ema_top", "pb_ema_bot"]
    for col in required:
        if col not in df.columns:
            return None, None, None, None, f"Missing {col}"

    try:
        curr = df.iloc[index]
    except (IndexError, KeyError):
        return None, None, None, None, "Index Error"

    abs_index = index if index >= 0 else (len(df) + index)
    if abs_index < 60:
        return None, None, None, None, "Not Enough Warmup"

    # Extract values
    close = float(curr["close"])
    baseline = float(curr["baseline"])
    pb_top = float(curr["pb_ema_top"])
    pb_bot = float(curr["pb_ema_bot"])

    if any(pd.isna([close, baseline, pb_top, pb_bot])):
        return None, None, None, None, "NaN Values"

    # Direction from SSL
    price_above_baseline = close > baseline
    price_below_baseline = close < baseline

    if not (price_above_baseline or price_below_baseline):
        return None, None, None, None, "No Direction"

    # === BASELINE TOUCH CHECK (Optional but recommended) ===
    # This is the key filter that improves signal quality significantly.
    # A baseline touch means price actually interacted with SSL support/resistance.
    baseline_touched_long = False
    baseline_touched_short = False

    if require_baseline_touch:
        lookback_start = max(0, abs_index - baseline_touch_lookback)
        ssl_touch_tolerance = 0.003  # 0.3% tolerance

        if price_above_baseline:
            # For LONG: check if low touched baseline in lookback period
            lookback_lows = df["low"].iloc[lookback_start:abs_index + 1].values
            lookback_baselines = df["baseline"].iloc[lookback_start:abs_index + 1].values
            baseline_touched_long = np.any(
                lookback_lows <= lookback_baselines * (1 + ssl_touch_tolerance)
            )
            if not baseline_touched_long:
                return None, None, None, None, "No Baseline Touch (LONG)"

        if price_below_baseline:
            # For SHORT: check if high touched baseline in lookback period
            lookback_highs = df["high"].iloc[lookback_start:abs_index + 1].values
            lookback_baselines = df["baseline"].iloc[lookback_start:abs_index + 1].values
            baseline_touched_short = np.any(
                lookback_highs >= lookback_baselines * (1 - ssl_touch_tolerance)
            )
            if not baseline_touched_short:
                return None, None, None, None, "No Baseline Touch (SHORT)"

    # PBEMA path check (relaxed)
    pbema_mid = (pb_top + pb_bot) / 2

    # === LONG SIGNAL ===
    if price_above_baseline:
        # Target above price?
        if pb_bot <= close:
            return None, None, None, None, "No PBEMA Path (LONG)"

        # Distance check (relaxed)
        long_pbema_distance = (pb_bot - close) / close
        if long_pbema_distance < min_pbema_distance:
            return None, None, None, None, f"PBEMA Too Close ({long_pbema_distance:.4f})"

        # Entry, TP, SL
        entry = close
        tp = pb_bot
        swing_low = float(df["low"].iloc[max(0, abs_index - 20):abs_index].min())
        sl = min(swing_low * 0.998, baseline * 0.998)

        if tp <= entry or sl >= entry:
            return None, None, None, None, "Invalid Levels (LONG)"

        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "LONG", entry, tp, sl, f"CORE(R:{rr:.2f})"

    # === SHORT SIGNAL ===
    if price_below_baseline:
        # Target below price?
        if pb_top >= close:
            return None, None, None, None, "No PBEMA Path (SHORT)"

        # Distance check (relaxed)
        short_pbema_distance = (close - pb_top) / close
        if short_pbema_distance < min_pbema_distance:
            return None, None, None, None, f"PBEMA Too Close ({short_pbema_distance:.4f})"

        # Entry, TP, SL
        entry = close
        tp = pb_top
        swing_high = float(df["high"].iloc[max(0, abs_index - 20):abs_index].max())
        sl = max(swing_high * 1.002, baseline * 1.002)

        if tp >= entry or sl <= entry:
            return None, None, None, None, "Invalid Levels (SHORT)"

        risk = sl - entry
        reward = entry - tp
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "SHORT", entry, tp, sl, f"CORE(R:{rr:.2f})"

    return None, None, None, None, "No Signal"


# ============================================================================
# AT FILTER EVALUATOR
# ============================================================================

def check_at_allows_signal(
    df: pd.DataFrame,
    index: int,
    signal_type: str,
    scenario: ATScenario,
) -> Tuple[bool, str]:
    """
    Check if a given AT scenario would allow the signal.

    Args:
        df: DataFrame with indicators
        index: Index of the signal
        signal_type: "LONG" or "SHORT"
        scenario: ATScenario to test

    Returns:
        (allowed: bool, reason: str)
    """
    abs_index = index if index >= 0 else (len(df) + index)
    curr = df.iloc[abs_index]

    # Extract AT state
    at_buyers = bool(curr.get("at_buyers_dominant", False))
    at_sellers = bool(curr.get("at_sellers_dominant", False))
    at_flat = bool(curr.get("at_is_flat", False))

    # Calculate regime over lookback period
    regime_lookback = 20
    if "at_regime" in df.columns and not pd.isna(curr.get("at_regime")):
        at_regime = str(curr.get("at_regime"))
    else:
        # Calculate regime based on AT dominance over lookback
        if abs_index >= regime_lookback - 1:
            start_idx = abs_index - regime_lookback + 1
            end_idx = abs_index + 1
            buyers_bars = df["at_buyers_dominant"].iloc[start_idx:end_idx].sum()
            sellers_bars = df["at_sellers_dominant"].iloc[start_idx:end_idx].sum()

            regime_threshold = 0.6  # 60% dominance required
            buyers_ratio = buyers_bars / regime_lookback
            sellers_ratio = sellers_bars / regime_lookback

            if buyers_ratio >= regime_threshold:
                at_regime = "bullish_regime"
            elif sellers_ratio >= regime_threshold:
                at_regime = "bearish_regime"
            else:
                at_regime = "neutral_regime"
        else:
            at_regime = "neutral_regime"

    # =========================================================================
    # REGIME FILTER (checked FIRST, before AT filter)
    # =========================================================================
    regime_filter = getattr(scenario, "regime_filter", "off")

    if regime_filter == "aligned":
        # Strictest: Only allow signals that MATCH regime direction
        # - LONG only in bullish_regime
        # - SHORT only in bearish_regime
        # - neutral_regime: blocks both (conservative)
        if signal_type == "LONG":
            if at_regime != "bullish_regime":
                return False, f"Regime Not Bullish ({at_regime})"
        elif signal_type == "SHORT":
            if at_regime != "bearish_regime":
                return False, f"Regime Not Bearish ({at_regime})"

    elif regime_filter == "veto":
        # Lenient: Only BLOCK signals that OPPOSE regime direction
        # - LONG blocked in bearish_regime
        # - SHORT blocked in bullish_regime
        # - neutral_regime: allows both
        if signal_type == "LONG" and at_regime == "bearish_regime":
            return False, "Regime Veto: Bearish blocks LONG"
        elif signal_type == "SHORT" and at_regime == "bullish_regime":
            return False, "Regime Veto: Bullish blocks SHORT"

    elif regime_filter == "veto_neutral_long":
        # Special: veto + allow LONG in neutral
        # - LONG blocked only in bearish_regime (neutral allowed)
        # - SHORT blocked in bullish_regime
        if signal_type == "LONG" and at_regime == "bearish_regime":
            return False, "Regime Veto: Bearish blocks LONG"
        elif signal_type == "SHORT" and at_regime == "bullish_regime":
            return False, "Regime Veto: Bullish blocks SHORT"

    elif regime_filter == "skip_neutral":
        # NEW: Skip only neutral regime (based on analysis: 19.7% win rate!)
        # - Allow both LONG and SHORT in bullish/bearish
        # - Block everything in neutral_regime
        if at_regime == "neutral_regime":
            return False, "Regime Skip: Neutral is dangerous"

    # =========================================================================
    # AT FILTER (checked AFTER regime filter passes)
    # =========================================================================
    if not scenario.at_active or scenario.at_mode == "off":
        return True, f"AT Off (Regime: {at_regime})"

    # Check SSL flip grace (if enabled)
    ssl_flip_grace = False
    if scenario.use_ssl_flip_grace and abs_index >= 1:
        grace_bars = scenario.ssl_flip_grace_bars
        close_arr = df["close"].values
        baseline_arr = df["baseline"].values

        for lookback in range(1, min(grace_bars + 1, abs_index + 1)):
            prev_idx = abs_index - lookback
            prev_close = float(close_arr[prev_idx])
            prev_baseline = float(baseline_arr[prev_idx])
            curr_close = float(close_arr[abs_index])
            curr_baseline = float(baseline_arr[abs_index])

            # Bullish flip
            if signal_type == "LONG":
                if prev_close < prev_baseline and curr_close > curr_baseline:
                    if not at_sellers:  # Not opposing
                        ssl_flip_grace = True
                        break

            # Bearish flip
            if signal_type == "SHORT":
                if prev_close > prev_baseline and curr_close < curr_baseline:
                    if not at_buyers:  # Not opposing
                        ssl_flip_grace = True
                        break

    # Binary mode
    if scenario.at_mode == "binary":
        # Check flat filter
        if not scenario.skip_at_flat_filter and at_flat:
            return False, "AT Flat"

        # Check alignment
        if signal_type == "LONG":
            if at_buyers or ssl_flip_grace:
                return True, "AT Buyers" if at_buyers else "SSL Flip Grace"
            return False, "AT No Buyers"

        if signal_type == "SHORT":
            if at_sellers or ssl_flip_grace:
                return True, "AT Sellers" if at_sellers else "SSL Flip Grace"
            return False, "AT No Sellers"

    # Score mode (simplified - treat like lax binary for now)
    if scenario.at_mode == "score":
        if signal_type == "LONG":
            # In score mode, we use a threshold on alignment confidence
            # For simplicity, just check if not actively opposing
            if at_sellers:
                return False, "AT Sellers (Opposing)"
            return True, "AT Score OK"

        if signal_type == "SHORT":
            if at_buyers:
                return False, "AT Buyers (Opposing)"
            return True, "AT Score OK"

    # =========================================================================
    # VETO MODE - Only block when AT actively opposes the signal
    # =========================================================================
    # This is LESS restrictive than binary mode:
    # - Binary: Requires AT alignment (buyers for LONG, sellers for SHORT)
    # - Veto: Only blocks when AT is ACTIVELY opposing
    #
    # Expected behavior:
    # - LONG + AT buyers_dominant = ALLOW (aligned)
    # - LONG + AT sellers_dominant = BLOCK (opposing - VETO)
    # - LONG + AT flat/neutral = ALLOW (no opposition)
    # - SHORT + AT sellers_dominant = ALLOW (aligned)
    # - SHORT + AT buyers_dominant = BLOCK (opposing - VETO)
    # - SHORT + AT flat/neutral = ALLOW (no opposition)
    #
    # With SSL flip grace, we also allow signals shortly after SSL flips
    # even if AT hasn't caught up yet.
    if scenario.at_mode == "veto":
        if signal_type == "LONG":
            # Block only if AT actively sellers dominant
            if at_sellers and not ssl_flip_grace:
                return False, "AT VETO: Sellers Dominant"
            # Otherwise allow (includes aligned, flat, neutral, grace period)
            if at_buyers:
                return True, "AT Aligned (Buyers)"
            elif ssl_flip_grace:
                return True, "SSL Flip Grace"
            else:
                return True, "AT Neutral (No Veto)"

        if signal_type == "SHORT":
            # Block only if AT actively buyers dominant
            if at_buyers and not ssl_flip_grace:
                return False, "AT VETO: Buyers Dominant"
            # Otherwise allow (includes aligned, flat, neutral, grace period)
            if at_sellers:
                return True, "AT Aligned (Sellers)"
            elif ssl_flip_grace:
                return True, "SSL Flip Grace"
            else:
                return True, "AT Neutral (No Veto)"

    return True, "Unknown Mode"


# ============================================================================
# TRADE SIMULATOR
# ============================================================================

def simulate_trade(
    df: pd.DataFrame,
    signal_idx: int,
    signal_type: str,
    entry: float,
    tp: float,
    sl: float,
    position_size: float = 35.0,  # $35 per trade
) -> Dict:
    """
    Simulate a trade from signal to exit.

    Returns dict with:
    - exit_idx, exit_price, exit_reason
    - pnl, r_multiple
    - bars_held
    """
    abs_idx = signal_idx if signal_idx >= 0 else (len(df) + signal_idx)

    # Start from next candle after signal
    for i in range(abs_idx + 1, len(df)):
        candle = df.iloc[i]
        high = float(candle["high"])
        low = float(candle["low"])

        if signal_type == "LONG":
            # Check SL first (conservative)
            if low <= sl:
                loss = (sl - entry) / entry * position_size
                return {
                    "exit_idx": i,
                    "exit_price": sl,
                    "exit_reason": "SL",
                    "pnl": loss,
                    "r_multiple": -1.0,
                    "bars_held": i - abs_idx,
                }
            # Then check TP
            if high >= tp:
                profit = (tp - entry) / entry * position_size
                risk = entry - sl
                reward = tp - entry
                r_mult = reward / risk if risk > 0 else 0
                return {
                    "exit_idx": i,
                    "exit_price": tp,
                    "exit_reason": "TP",
                    "pnl": profit,
                    "r_multiple": r_mult,
                    "bars_held": i - abs_idx,
                }

        else:  # SHORT
            # Check SL first
            if high >= sl:
                loss = (entry - sl) / entry * position_size
                return {
                    "exit_idx": i,
                    "exit_price": sl,
                    "exit_reason": "SL",
                    "pnl": loss,
                    "r_multiple": -1.0,
                    "bars_held": i - abs_idx,
                }
            # Then check TP
            if low <= tp:
                profit = (entry - tp) / entry * position_size
                risk = sl - entry
                reward = entry - tp
                r_mult = reward / risk if risk > 0 else 0
                return {
                    "exit_idx": i,
                    "exit_price": tp,
                    "exit_reason": "TP",
                    "pnl": profit,
                    "r_multiple": r_mult,
                    "bars_held": i - abs_idx,
                }

    # No exit found - close at last price
    last_price = float(df.iloc[-1]["close"])
    if signal_type == "LONG":
        pnl = (last_price - entry) / entry * position_size
    else:
        pnl = (entry - last_price) / entry * position_size

    return {
        "exit_idx": len(df) - 1,
        "exit_price": last_price,
        "exit_reason": "EOD",
        "pnl": pnl,
        "r_multiple": pnl / position_size if position_size > 0 else 0,
        "bars_held": len(df) - 1 - abs_idx,
    }


# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class ATScenarioAnalyzer:
    """
    Main analyzer class that runs all AT scenarios on a dataset.

    Usage:
        analyzer = ATScenarioAnalyzer()
        results_df = analyzer.run_analysis(df, symbol="BTCUSDT", timeframe="15m")
        summary = analyzer.get_summary()
    """

    def __init__(
        self,
        scenarios: List[ATScenario] = None,
        min_rr: float = 1.2,
        min_pbema_distance: float = 0.002,
        min_bars_between_signals: int = 5,
        require_baseline_touch: bool = False,
        baseline_touch_lookback: int = 5,
    ):
        self.scenarios = scenarios or AT_SCENARIOS
        self.min_rr = min_rr
        self.min_pbema_distance = min_pbema_distance
        self.min_bars_between = min_bars_between_signals
        self.require_baseline_touch = require_baseline_touch
        self.baseline_touch_lookback = baseline_touch_lookback

        self.results_df = None
        self.summary = None

    def run_analysis(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "15m",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run the full analysis on a dataset.

        Args:
            df: DataFrame with OHLCV and indicator columns
            symbol: Symbol name for reporting
            timeframe: Timeframe for reporting
            verbose: Print progress

        Returns:
            DataFrame with one row per potential signal,
            columns for each scenario's results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"AT SCENARIO ANALYSIS: {symbol} {timeframe}")
            print(f"{'='*60}")
            print(f"Data: {len(df)} candles")
            print(f"Scenarios: {len(self.scenarios)}")
            print(f"Min RR: {self.min_rr}, Min PBEMA Distance: {self.min_pbema_distance}")

        results = []
        last_signal_idx = -self.min_bars_between

        # Scan for all potential signals
        for i in range(60, len(df) - 10):  # Leave room for trade to play out
            # Skip if too close to last signal
            if i - last_signal_idx < self.min_bars_between:
                continue

            # Check for core signal (AT-independent)
            signal_type, entry, tp, sl, reason = check_core_signal(
                df, i,
                min_rr=self.min_rr,
                min_pbema_distance=self.min_pbema_distance,
                require_baseline_touch=self.require_baseline_touch,
                baseline_touch_lookback=self.baseline_touch_lookback,
            )

            if signal_type is None:
                continue

            # Found a core signal!
            last_signal_idx = i

            # Get timestamp
            timestamp = df.index[i] if hasattr(df.index, '__iter__') else i

            # Extract AT state
            at_state = extract_at_state(df, i, signal_type)

            # Simulate trade (we'll use this for all scenarios that allow the signal)
            trade_result = simulate_trade(df, i, signal_type, entry, tp, sl)

            # Build result row
            row = {
                "timestamp": timestamp,
                "index": i,
                "signal_type": signal_type,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "core_reason": reason,
                "trade_pnl": trade_result["pnl"],
                "trade_r": trade_result["r_multiple"],
                "trade_exit_reason": trade_result["exit_reason"],
                "trade_bars_held": trade_result["bars_held"],
            }

            # Add AT state
            row.update(at_state.to_dict())

            # Test each scenario
            for scenario in self.scenarios:
                allowed, filter_reason = check_at_allows_signal(
                    df, i, signal_type, scenario
                )

                # Record if scenario would allow this signal
                row[f"{scenario.name}_allowed"] = allowed
                row[f"{scenario.name}_reason"] = filter_reason

                # Record PnL (0 if not allowed, actual PnL if allowed)
                row[f"{scenario.name}_pnl"] = trade_result["pnl"] if allowed else 0

            results.append(row)

        if verbose:
            print(f"\nFound {len(results)} potential signals")

        # Create DataFrame
        self.results_df = pd.DataFrame(results)

        # Calculate summary
        self._calculate_summary(verbose)

        return self.results_df

    def _calculate_summary(self, verbose: bool = True):
        """Calculate summary statistics for each scenario."""
        if self.results_df is None or self.results_df.empty:
            self.summary = {}
            return

        summary = {}

        for scenario in self.scenarios:
            allowed_col = f"{scenario.name}_allowed"
            pnl_col = f"{scenario.name}_pnl"

            allowed_signals = self.results_df[self.results_df[allowed_col] == True]

            n_signals = len(self.results_df)
            n_allowed = len(allowed_signals)
            n_blocked = n_signals - n_allowed

            if n_allowed > 0:
                total_pnl = allowed_signals[pnl_col].sum()
                avg_pnl = allowed_signals[pnl_col].mean()

                # Count wins/losses based on actual trade results
                wins = len(allowed_signals[allowed_signals["trade_exit_reason"] == "TP"])
                losses = len(allowed_signals[allowed_signals["trade_exit_reason"] == "SL"])
                other = n_allowed - wins - losses

                win_rate = wins / n_allowed if n_allowed > 0 else 0

                # Calculate E[R]
                avg_r = allowed_signals["trade_r"].mean()

                # Calculate max drawdown
                cumulative_pnl = allowed_signals[pnl_col].cumsum()
                peak = cumulative_pnl.cummax()
                drawdown = peak - cumulative_pnl
                max_dd = drawdown.max()
            else:
                total_pnl = 0
                avg_pnl = 0
                wins = 0
                losses = 0
                other = 0
                win_rate = 0
                avg_r = 0
                max_dd = 0

            summary[scenario.name] = {
                "n_signals": n_signals,
                "n_allowed": n_allowed,
                "n_blocked": n_blocked,
                "allow_rate": n_allowed / n_signals if n_signals > 0 else 0,
                "wins": wins,
                "losses": losses,
                "other": other,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "avg_r": avg_r,
                "max_dd": max_dd,
            }

        self.summary = summary

        if verbose:
            self._print_summary()

    def _print_summary(self):
        """Print formatted summary."""
        if not self.summary:
            print("No summary available")
            return

        print(f"\n{'='*80}")
        print("AT SCENARIO COMPARISON")
        print(f"{'='*80}")
        print(f"{'Scenario':<22} {'Allowed':>8} {'Blocked':>8} {'PnL':>12} {'Win%':>8} {'E[R]':>8} {'MaxDD':>10}")
        print("-" * 80)

        # Sort by total PnL
        sorted_scenarios = sorted(
            self.summary.items(),
            key=lambda x: x[1]["total_pnl"],
            reverse=True
        )

        best_scenario = sorted_scenarios[0][0] if sorted_scenarios else None

        for name, stats in sorted_scenarios:
            marker = " <- BEST" if name == best_scenario else ""
            print(f"{name:<22} {stats['n_allowed']:>8} {stats['n_blocked']:>8} "
                  f"${stats['total_pnl']:>10.2f} {stats['win_rate']*100:>7.1f}% "
                  f"{stats['avg_r']:>7.2f}R ${stats['max_dd']:>9.2f}{marker}")

        print(f"\n{'='*80}")
        print(f"BEST SCENARIO: {best_scenario}")
        if best_scenario:
            best_stats = self.summary[best_scenario]
            print(f"  Trades: {best_stats['n_allowed']}")
            print(f"  Win Rate: {best_stats['win_rate']*100:.1f}%")
            print(f"  Total PnL: ${best_stats['total_pnl']:.2f}")
            print(f"  E[R]: {best_stats['avg_r']:.3f}")
        print(f"{'='*80}")

    def get_summary(self) -> Dict:
        """Return summary dict."""
        return self.summary

    def get_results_df(self) -> pd.DataFrame:
        """Return full results DataFrame."""
        return self.results_df

    def export_results(self, filepath: str):
        """Export results to JSON file."""
        if self.results_df is None:
            print("No results to export")
            return

        import json

        export_data = {
            "summary": self.summary,
            "results": self.results_df.to_dict(orient="records"),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"Results exported to {filepath}")

    def get_at_state_patterns(self) -> pd.DataFrame:
        """
        Analyze AT state patterns and their correlation with trade outcomes.

        Returns DataFrame with pattern analysis.
        """
        if self.results_df is None or self.results_df.empty:
            return pd.DataFrame()

        # Group by AT state and analyze outcomes
        patterns = []

        # Pattern 1: AT aligned vs not aligned
        aligned = self.results_df[self.results_df["at_ssl_aligned"] == True]
        not_aligned = self.results_df[self.results_df["at_ssl_aligned"] == False]

        patterns.append({
            "pattern": "AT SSL Aligned",
            "count": len(aligned),
            "win_rate": len(aligned[aligned["trade_exit_reason"] == "TP"]) / len(aligned) if len(aligned) > 0 else 0,
            "avg_pnl": aligned["trade_pnl"].mean() if len(aligned) > 0 else 0,
        })
        patterns.append({
            "pattern": "AT SSL Not Aligned",
            "count": len(not_aligned),
            "win_rate": len(not_aligned[not_aligned["trade_exit_reason"] == "TP"]) / len(not_aligned) if len(not_aligned) > 0 else 0,
            "avg_pnl": not_aligned["trade_pnl"].mean() if len(not_aligned) > 0 else 0,
        })

        # Pattern 2: AT flat vs not flat
        flat = self.results_df[self.results_df["at_is_flat"] == True]
        not_flat = self.results_df[self.results_df["at_is_flat"] == False]

        patterns.append({
            "pattern": "AT Flat",
            "count": len(flat),
            "win_rate": len(flat[flat["trade_exit_reason"] == "TP"]) / len(flat) if len(flat) > 0 else 0,
            "avg_pnl": flat["trade_pnl"].mean() if len(flat) > 0 else 0,
        })
        patterns.append({
            "pattern": "AT Not Flat",
            "count": len(not_flat),
            "win_rate": len(not_flat[not_flat["trade_exit_reason"] == "TP"]) / len(not_flat) if len(not_flat) > 0 else 0,
            "avg_pnl": not_flat["trade_pnl"].mean() if len(not_flat) > 0 else 0,
        })

        # Pattern 3: Line separation high vs low
        median_sep = self.results_df["at_line_separation"].median()
        high_sep = self.results_df[self.results_df["at_line_separation"] > median_sep]
        low_sep = self.results_df[self.results_df["at_line_separation"] <= median_sep]

        patterns.append({
            "pattern": "AT High Separation",
            "count": len(high_sep),
            "win_rate": len(high_sep[high_sep["trade_exit_reason"] == "TP"]) / len(high_sep) if len(high_sep) > 0 else 0,
            "avg_pnl": high_sep["trade_pnl"].mean() if len(high_sep) > 0 else 0,
        })
        patterns.append({
            "pattern": "AT Low Separation",
            "count": len(low_sep),
            "win_rate": len(low_sep[low_sep["trade_exit_reason"] == "TP"]) / len(low_sep) if len(low_sep) > 0 else 0,
            "avg_pnl": low_sep["trade_pnl"].mean() if len(low_sep) > 0 else 0,
        })

        # Pattern 4: Bars since flip
        recent_flip = self.results_df[self.results_df["at_bars_since_flip"] <= 5]
        old_flip = self.results_df[self.results_df["at_bars_since_flip"] > 5]

        patterns.append({
            "pattern": "AT Recent Flip (<=5 bars)",
            "count": len(recent_flip),
            "win_rate": len(recent_flip[recent_flip["trade_exit_reason"] == "TP"]) / len(recent_flip) if len(recent_flip) > 0 else 0,
            "avg_pnl": recent_flip["trade_pnl"].mean() if len(recent_flip) > 0 else 0,
        })
        patterns.append({
            "pattern": "AT Established (>5 bars)",
            "count": len(old_flip),
            "win_rate": len(old_flip[old_flip["trade_exit_reason"] == "TP"]) / len(old_flip) if len(old_flip) > 0 else 0,
            "avg_pnl": old_flip["trade_pnl"].mean() if len(old_flip) > 0 else 0,
        })

        # =====================================================================
        # REGIME PATTERNS (NEW)
        # =====================================================================

        # Pattern 5: Regime types
        if "at_regime" in self.results_df.columns:
            for regime in ["bullish_regime", "bearish_regime", "neutral_regime"]:
                regime_df = self.results_df[self.results_df["at_regime"] == regime]
                if len(regime_df) > 0:
                    patterns.append({
                        "pattern": f"Regime: {regime.replace('_regime', '').title()}",
                        "count": len(regime_df),
                        "win_rate": len(regime_df[regime_df["trade_exit_reason"] == "TP"]) / len(regime_df),
                        "avg_pnl": regime_df["trade_pnl"].mean(),
                    })

            # Pattern 6: Signal-Regime Alignment
            # LONG in bullish or SHORT in bearish = aligned
            long_signals = self.results_df[self.results_df["signal_type"] == "LONG"]
            short_signals = self.results_df[self.results_df["signal_type"] == "SHORT"]

            long_in_bullish = long_signals[long_signals["at_regime"] == "bullish_regime"]
            long_in_bearish = long_signals[long_signals["at_regime"] == "bearish_regime"]
            long_in_neutral = long_signals[long_signals["at_regime"] == "neutral_regime"]

            short_in_bearish = short_signals[short_signals["at_regime"] == "bearish_regime"]
            short_in_bullish = short_signals[short_signals["at_regime"] == "bullish_regime"]
            short_in_neutral = short_signals[short_signals["at_regime"] == "neutral_regime"]

            if len(long_in_bullish) > 0:
                patterns.append({
                    "pattern": "LONG in Bullish (Aligned)",
                    "count": len(long_in_bullish),
                    "win_rate": len(long_in_bullish[long_in_bullish["trade_exit_reason"] == "TP"]) / len(long_in_bullish),
                    "avg_pnl": long_in_bullish["trade_pnl"].mean(),
                })

            if len(long_in_bearish) > 0:
                patterns.append({
                    "pattern": "LONG in Bearish (OPPOSING)",
                    "count": len(long_in_bearish),
                    "win_rate": len(long_in_bearish[long_in_bearish["trade_exit_reason"] == "TP"]) / len(long_in_bearish),
                    "avg_pnl": long_in_bearish["trade_pnl"].mean(),
                })

            if len(long_in_neutral) > 0:
                patterns.append({
                    "pattern": "LONG in Neutral",
                    "count": len(long_in_neutral),
                    "win_rate": len(long_in_neutral[long_in_neutral["trade_exit_reason"] == "TP"]) / len(long_in_neutral),
                    "avg_pnl": long_in_neutral["trade_pnl"].mean(),
                })

            if len(short_in_bearish) > 0:
                patterns.append({
                    "pattern": "SHORT in Bearish (Aligned)",
                    "count": len(short_in_bearish),
                    "win_rate": len(short_in_bearish[short_in_bearish["trade_exit_reason"] == "TP"]) / len(short_in_bearish),
                    "avg_pnl": short_in_bearish["trade_pnl"].mean(),
                })

            if len(short_in_bullish) > 0:
                patterns.append({
                    "pattern": "SHORT in Bullish (OPPOSING)",
                    "count": len(short_in_bullish),
                    "win_rate": len(short_in_bullish[short_in_bullish["trade_exit_reason"] == "TP"]) / len(short_in_bullish),
                    "avg_pnl": short_in_bullish["trade_pnl"].mean(),
                })

            if len(short_in_neutral) > 0:
                patterns.append({
                    "pattern": "SHORT in Neutral",
                    "count": len(short_in_neutral),
                    "win_rate": len(short_in_neutral[short_in_neutral["trade_exit_reason"] == "TP"]) / len(short_in_neutral),
                    "avg_pnl": short_in_neutral["trade_pnl"].mean(),
                })

        return pd.DataFrame(patterns)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ATScenario",
    "AT_SCENARIOS",
    "ATState",
    "extract_at_state",
    "check_core_signal",
    "check_at_allows_signal",
    "simulate_trade",
    "ATScenarioAnalyzer",
]
