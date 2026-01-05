#!/usr/bin/env python3
"""
Minimal SSL + PBEMA Backtest Script

Purpose: Test if the CORE strategy has edge when using only SSL + PBEMA
         (matching user's actual live trading approach)

Live Trading Rules (from user):
- SSL baseline (HMA60) for direction
- PBEMA cloud (EMA200) for TP target
- AlphaTrend helps on 4h+ (optional)
- Visual interpretation of FVG/key points (can't automate)

This script tests:
1. MINIMAL: SSL + PBEMA only (price vs baseline + PBEMA target exists)
2. WITH_AT: SSL + PBEMA + AlphaTrend (adds flow confirmation)
3. FULL: All current filters (baseline backtest for comparison)

Filters in current code vs what we test:
- ADX filter                  [FULL only]
- Regime gating               [FULL only]
- SSL never lost filter       [FULL only]
- Baseline touch detection    [FULL only]
- Body position check         [FULL only]
- AlphaTrend flat filter      [WITH_AT, FULL]
- PBEMA distance check        [WITH_AT, FULL]
- PBEMA-SSL overlap check     [WITH_AT, FULL]
- Wick rejection              [FULL only]
- RSI filter                  [WITH_AT, FULL]

Usage:
    python runners/run_minimal_backtest.py
    python runners/run_minimal_backtest.py --symbols BTCUSDT ETHUSDT
    python runners/run_minimal_backtest.py --timeframes 15m 1h 4h
    python runners/run_minimal_backtest.py --mode minimal  # only minimal test
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    TRADING_CONFIG, SYMBOLS, TIMEFRAMES,
    calculate_indicators, BinanceClient, get_client,
    MarketStructure, get_structure_score, TrendType,
    FVGDetector, get_fvg_score, FVGType,
)


def check_minimal_signal(
    df: pd.DataFrame,
    index: int = -2,
    min_rr: float = 1.5,  # Lower RR for more signals
) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], str]:
    """
    MINIMAL signal check - SSL + PBEMA only.

    This matches the user's live trading approach:
    1. Price above SSL baseline (HMA60) = LONG direction
    2. Price below SSL baseline (HMA60) = SHORT direction
    3. PBEMA exists as target (no distance requirement)

    No other filters. Pure trend direction + target.
    """
    if df is None or df.empty or len(df) < 60:
        return None, None, None, None, "No Data"

    required_cols = ["open", "high", "low", "close", "baseline", "pb_ema_top", "pb_ema_bot"]
    for col in required_cols:
        if col not in df.columns:
            return None, None, None, None, f"Missing {col}"

    try:
        curr = df.iloc[index]
    except (IndexError, KeyError):
        return None, None, None, None, "Index Error"

    abs_index = index if index >= 0 else (len(df) + index)

    # Extract values
    close = float(curr["close"])
    baseline = float(curr["baseline"])
    pb_top = float(curr["pb_ema_top"])
    pb_bot = float(curr["pb_ema_bot"])

    if any(pd.isna([close, baseline, pb_top, pb_bot])):
        return None, None, None, None, "NaN Values"

    # Direction based on SSL baseline only
    is_long = close > baseline
    is_short = close < baseline

    # LONG: PBEMA should be above price (target exists)
    # SHORT: PBEMA should be below price (target exists)
    pbema_mid = (pb_top + pb_bot) / 2

    if is_long and pb_bot > close:  # Target above
        entry = close
        tp = pb_bot

        # Simple SL: below baseline or recent low
        swing_low = float(df["low"].iloc[max(0, abs_index-20):abs_index].min())
        sl = min(swing_low * 0.998, baseline * 0.998)

        if tp <= entry or sl >= entry:
            return None, None, None, None, "Invalid Levels (LONG)"

        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "LONG", entry, tp, sl, f"MINIMAL(R:{rr:.2f})"

    elif is_short and pb_top < close:  # Target below
        entry = close
        tp = pb_top

        # Simple SL: above baseline or recent high
        swing_high = float(df["high"].iloc[max(0, abs_index-20):abs_index].max())
        sl = max(swing_high * 1.002, baseline * 1.002)

        if tp >= entry or sl <= entry:
            return None, None, None, None, "Invalid Levels (SHORT)"

        risk = sl - entry
        reward = entry - tp
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "SHORT", entry, tp, sl, f"MINIMAL(R:{rr:.2f})"

    return None, None, None, None, "No Signal"


def check_with_structure_signal(
    df: pd.DataFrame,
    index: int = -2,
    min_rr: float = 1.5,
    ms: MarketStructure = None,
) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], str]:
    """
    WITH_MS signal check - SSL + PBEMA + Market Structure.

    Adds Market Structure validation:
    1. Price vs SSL baseline (direction)
    2. PBEMA as target
    3. Market Structure trend alignment
    4. Trade at key swing levels (HL for LONG, LH for SHORT)
    """
    if df is None or df.empty or len(df) < 60:
        return None, None, None, None, "No Data"

    required_cols = ["open", "high", "low", "close", "baseline", "pb_ema_top", "pb_ema_bot"]
    for col in required_cols:
        if col not in df.columns:
            return None, None, None, None, f"Missing {col}"

    try:
        curr = df.iloc[index]
    except (IndexError, KeyError):
        return None, None, None, None, "Index Error"

    abs_index = index if index >= 0 else (len(df) + index)

    # Extract values
    close = float(curr["close"])
    baseline = float(curr["baseline"])
    pb_top = float(curr["pb_ema_top"])
    pb_bot = float(curr["pb_ema_bot"])

    if any(pd.isna([close, baseline, pb_top, pb_bot])):
        return None, None, None, None, "NaN Values"

    # Initialize MarketStructure if not provided
    if ms is None:
        ms = MarketStructure(swing_length=5)

    # Get structure score
    is_long_candidate = close > baseline and pb_bot > close
    is_short_candidate = close < baseline and pb_top < close

    if not is_long_candidate and not is_short_candidate:
        return None, None, None, None, "No Direction"

    signal_type = "LONG" if is_long_candidate else "SHORT"
    score, result = get_structure_score(df, signal_type, abs_index)

    # Require minimum structure score of 1.0 (trend-aligned)
    if score < 1.0:
        return None, None, None, None, f"MS Score Low ({score:.1f}, {result.trend.value})"

    if is_long_candidate:
        entry = close
        tp = pb_bot
        swing_low = float(df["low"].iloc[max(0, abs_index-20):abs_index].min())
        sl = min(swing_low * 0.998, baseline * 0.998)

        # Use structure swing low if available and better
        if result.last_swing_low and result.last_swing_low.price < entry:
            sl = min(sl, result.last_swing_low.price * 0.998)

        if tp <= entry or sl >= entry:
            return None, None, None, None, "Invalid Levels (LONG)"

        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "LONG", entry, tp, sl, f"WITH_MS(R:{rr:.2f},S:{score:.1f})"

    else:  # SHORT
        entry = close
        tp = pb_top
        swing_high = float(df["high"].iloc[max(0, abs_index-20):abs_index].max())
        sl = max(swing_high * 1.002, baseline * 1.002)

        # Use structure swing high if available and better
        if result.last_swing_high and result.last_swing_high.price > entry:
            sl = max(sl, result.last_swing_high.price * 1.002)

        if tp >= entry or sl <= entry:
            return None, None, None, None, "Invalid Levels (SHORT)"

        risk = sl - entry
        reward = entry - tp
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "SHORT", entry, tp, sl, f"WITH_MS(R:{rr:.2f},S:{score:.1f})"


def check_with_fvg_signal(
    df: pd.DataFrame,
    index: int = -2,
    min_rr: float = 1.5,
    fvg_detector: FVGDetector = None,
    require_mitigation: bool = True,  # MUST be a fresh mitigation (price returned to FVG)
    min_fvg_score: float = 1.5,       # Only used if require_mitigation=False
) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], str]:
    """
    WITH_FVG signal check - SSL + PBEMA + FVG MITIGATION.

    KEY INSIGHT: User uses FVG for ENTRY TIMING - trade when price RETURNS to FVG.
    Just being inside FVG is not enough - we need FRESH mitigation.

    Mitigation = price was OUTSIDE FVG, then RETURNED to it (optimal entry).

    Validation:
    1. Price vs SSL baseline (direction)
    2. PBEMA as target
    3. Price JUST RETURNED to FVG (mitigation) - entry timing
    4. FVG type matches direction (bullish for LONG, bearish for SHORT)
    5. FVG-based SL levels
    """
    if df is None or df.empty or len(df) < 60:
        return None, None, None, None, "No Data"

    required_cols = ["open", "high", "low", "close", "baseline", "pb_ema_top", "pb_ema_bot"]
    for col in required_cols:
        if col not in df.columns:
            return None, None, None, None, f"Missing {col}"

    try:
        curr = df.iloc[index]
    except (IndexError, KeyError):
        return None, None, None, None, "Index Error"

    abs_index = index if index >= 0 else (len(df) + index)

    # Extract values
    close = float(curr["close"])
    baseline = float(curr["baseline"])
    pb_top = float(curr["pb_ema_top"])
    pb_bot = float(curr["pb_ema_bot"])

    if any(pd.isna([close, baseline, pb_top, pb_bot])):
        return None, None, None, None, "NaN Values"

    # Initialize FVG detector
    if fvg_detector is None:
        fvg_detector = FVGDetector(min_gap_percent=0.10)  # 0.10% minimum gap

    # Check direction
    is_long_candidate = close > baseline and pb_bot > close
    is_short_candidate = close < baseline and pb_top < close

    if not is_long_candidate and not is_short_candidate:
        return None, None, None, None, "No Direction"

    signal_type = "LONG" if is_long_candidate else "SHORT"

    # Get FVG analysis with mitigation detection
    fvg_result = fvg_detector.analyze(df, abs_index, signal_type)
    fvg_levels = fvg_detector.get_fvg_levels(df, abs_index, signal_type)

    # MITIGATION MODE: Require fresh return to FVG
    if require_mitigation:
        if not fvg_result.is_mitigation:
            return None, None, None, None, "No Mitigation"

        # Check FVG type matches signal direction
        if fvg_result.mitigation_fvg is None:
            return None, None, None, None, "No Mitigation FVG"

        if signal_type == "LONG" and fvg_result.mitigation_fvg.fvg_type != FVGType.BULLISH:
            return None, None, None, None, "Wrong FVG Type (need Bullish)"
        if signal_type == "SHORT" and fvg_result.mitigation_fvg.fvg_type != FVGType.BEARISH:
            return None, None, None, None, "Wrong FVG Type (need Bearish)"

        # Use mitigated FVG for SL
        active_fvg = fvg_result.mitigation_fvg
    else:
        # Relaxed mode: use score threshold
        if fvg_result.score < min_fvg_score:
            return None, None, None, None, f"FVG Score Low ({fvg_result.score:.1f})"
        active_fvg = fvg_result.fvg

    if is_long_candidate:
        entry = close
        tp = pb_bot

        # Use FVG boundary as SL (tighter than swing-based)
        if active_fvg:
            # SL just below FVG low (invalidation point)
            sl = active_fvg.low * 0.998
        elif fvg_levels["sl"]:
            sl = fvg_levels["sl"]
        else:
            swing_low = float(df["low"].iloc[max(0, abs_index-20):abs_index].min())
            sl = min(swing_low * 0.998, baseline * 0.998)

        if tp <= entry or sl >= entry:
            return None, None, None, None, "Invalid Levels (LONG)"

        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "LONG", entry, tp, sl, f"FVG_MIT(R:{rr:.2f})"

    else:  # SHORT
        entry = close
        tp = pb_top

        # Use FVG boundary as SL (tighter than swing-based)
        if active_fvg:
            # SL just above FVG high (invalidation point)
            sl = active_fvg.high * 1.002
        elif fvg_levels["sl"]:
            sl = fvg_levels["sl"]
        else:
            swing_high = float(df["high"].iloc[max(0, abs_index-20):abs_index].max())
            sl = max(swing_high * 1.002, baseline * 1.002)

        if tp >= entry or sl <= entry:
            return None, None, None, None, "Invalid Levels (SHORT)"

        risk = sl - entry
        reward = entry - tp
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "SHORT", entry, tp, sl, f"FVG_MIT(R:{rr:.2f})"


def check_with_ms_fvg_signal(
    df: pd.DataFrame,
    index: int = -2,
    min_rr: float = 1.5,
    ms: MarketStructure = None,
    fvg_detector: FVGDetector = None,
    min_ms_score: float = 1.0,    # Must be trend-aligned
    require_mitigation: bool = True,  # MUST be fresh mitigation
) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], str]:
    """
    WITH_MS_FVG signal check - SSL + PBEMA + Market Structure + FVG MITIGATION.

    Complete confluence check:
    1. Price vs SSL baseline (direction)
    2. PBEMA as target
    3. Market Structure trend alignment (REQUIRED)
    4. FVG MITIGATION - price just returned to FVG (entry timing)

    This combines the best of both approaches:
    - MS ensures we're trading with the trend
    - FVG mitigation ensures optimal entry point (not just being in FVG)
    """
    if df is None or df.empty or len(df) < 60:
        return None, None, None, None, "No Data"

    required_cols = ["open", "high", "low", "close", "baseline", "pb_ema_top", "pb_ema_bot"]
    for col in required_cols:
        if col not in df.columns:
            return None, None, None, None, f"Missing {col}"

    try:
        curr = df.iloc[index]
    except (IndexError, KeyError):
        return None, None, None, None, "Index Error"

    abs_index = index if index >= 0 else (len(df) + index)

    # Extract values
    close = float(curr["close"])
    baseline = float(curr["baseline"])
    pb_top = float(curr["pb_ema_top"])
    pb_bot = float(curr["pb_ema_bot"])

    if any(pd.isna([close, baseline, pb_top, pb_bot])):
        return None, None, None, None, "NaN Values"

    # Initialize analyzers
    if ms is None:
        ms = MarketStructure(swing_length=5)
    if fvg_detector is None:
        fvg_detector = FVGDetector(min_gap_percent=0.15)  # Larger gaps only

    # Check direction
    is_long_candidate = close > baseline and pb_bot > close
    is_short_candidate = close < baseline and pb_top < close

    if not is_long_candidate and not is_short_candidate:
        return None, None, None, None, "No Direction"

    signal_type = "LONG" if is_long_candidate else "SHORT"

    # Get Market Structure score - MUST be trend-aligned
    ms_score, ms_result = get_structure_score(df, signal_type, abs_index)

    if ms_score < min_ms_score:
        return None, None, None, None, f"MS Not Aligned ({ms_score:.1f}, {ms_result.trend.value})"

    # Get FVG analysis with mitigation detection
    fvg_result = fvg_detector.analyze(df, abs_index, signal_type)

    # REQUIRE MITIGATION: Price must have just returned to FVG
    if require_mitigation:
        if not fvg_result.is_mitigation:
            return None, None, None, None, "No FVG Mitigation"

        # Check FVG type matches signal direction
        if fvg_result.mitigation_fvg is None:
            return None, None, None, None, "No Mitigation FVG"

        if signal_type == "LONG" and fvg_result.mitigation_fvg.fvg_type != FVGType.BULLISH:
            return None, None, None, None, "Wrong FVG Type (need Bullish)"
        if signal_type == "SHORT" and fvg_result.mitigation_fvg.fvg_type != FVGType.BEARISH:
            return None, None, None, None, "Wrong FVG Type (need Bearish)"

        active_fvg = fvg_result.mitigation_fvg
    else:
        active_fvg = fvg_result.fvg

    # Get FVG levels for SL
    fvg_levels = fvg_detector.get_fvg_levels(df, abs_index, signal_type)

    if is_long_candidate:
        entry = close
        tp = pb_bot

        # Best SL: Use FVG boundary (tightest), MS swing, or standard
        if active_fvg:
            # SL just below FVG low (invalidation point)
            sl = active_fvg.low * 0.998
        elif ms_result.last_swing_low and ms_result.last_swing_low.price < entry:
            sl = ms_result.last_swing_low.price * 0.998
        else:
            swing_low = float(df["low"].iloc[max(0, abs_index-20):abs_index].min())
            sl = min(swing_low * 0.998, baseline * 0.998)

        if tp <= entry or sl >= entry:
            return None, None, None, None, "Invalid Levels (LONG)"

        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "LONG", entry, tp, sl, f"MS+FVG_MIT(R:{rr:.2f},MS:{ms_score:.1f})"

    else:  # SHORT
        entry = close
        tp = pb_top

        # Best SL: Use FVG boundary (tightest), MS swing, or standard
        if active_fvg:
            # SL just above FVG high (invalidation point)
            sl = active_fvg.high * 1.002
        elif ms_result.last_swing_high and ms_result.last_swing_high.price > entry:
            sl = ms_result.last_swing_high.price * 1.002
        else:
            swing_high = float(df["high"].iloc[max(0, abs_index-20):abs_index].max())
            sl = max(swing_high * 1.002, baseline * 1.002)

        if tp >= entry or sl <= entry:
            return None, None, None, None, "Invalid Levels (SHORT)"

        risk = sl - entry
        reward = entry - tp
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "SHORT", entry, tp, sl, f"MS+FVG_MIT(R:{rr:.2f},MS:{ms_score:.1f})"


def check_with_fvg_bonus_signal(
    df: pd.DataFrame,
    index: int = -2,
    min_rr: float = 1.5,
    ms: MarketStructure = None,
    fvg_detector: FVGDetector = None,
    min_ms_score: float = 0.5,    # Lower threshold - more permissive
) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], str, float]:
    """
    FVG BONUS signal check - SSL + PBEMA + MS (required) + FVG (bonus).

    This is the HYBRID approach:
    - Base signal: SSL + PBEMA + Market Structure (REQUIRED)
    - FVG mitigation: BONUS (not required, but improves quality)

    Returns: (signal_type, entry, tp, sl, reason, quality_score)
    Quality score: 1.0 = base signal, 2.0 = with FVG mitigation
    """
    if df is None or df.empty or len(df) < 60:
        return None, None, None, None, "No Data", 0.0

    required_cols = ["open", "high", "low", "close", "baseline", "pb_ema_top", "pb_ema_bot"]
    for col in required_cols:
        if col not in df.columns:
            return None, None, None, None, f"Missing {col}", 0.0

    try:
        curr = df.iloc[index]
    except (IndexError, KeyError):
        return None, None, None, None, "Index Error", 0.0

    abs_index = index if index >= 0 else (len(df) + index)

    # Extract values
    close = float(curr["close"])
    baseline = float(curr["baseline"])
    pb_top = float(curr["pb_ema_top"])
    pb_bot = float(curr["pb_ema_bot"])

    if any(pd.isna([close, baseline, pb_top, pb_bot])):
        return None, None, None, None, "NaN Values", 0.0

    # Initialize analyzers
    if ms is None:
        ms = MarketStructure(swing_length=5)
    if fvg_detector is None:
        fvg_detector = FVGDetector(min_gap_percent=0.08)  # Slightly lower threshold

    # Check direction
    is_long_candidate = close > baseline and pb_bot > close
    is_short_candidate = close < baseline and pb_top < close

    if not is_long_candidate and not is_short_candidate:
        return None, None, None, None, "No Direction", 0.0

    signal_type = "LONG" if is_long_candidate else "SHORT"

    # REQUIRED: Market Structure must be trend-aligned
    ms_score, ms_result = get_structure_score(df, signal_type, abs_index)

    if ms_score < min_ms_score:
        return None, None, None, None, f"MS Low ({ms_score:.1f})", 0.0

    # BONUS: Check for FVG mitigation (not required)
    fvg_result = fvg_detector.analyze(df, abs_index, signal_type)

    has_fvg_bonus = False
    active_fvg = None

    if fvg_result.is_mitigation and fvg_result.mitigation_fvg:
        # Check FVG type matches signal
        if signal_type == "LONG" and fvg_result.mitigation_fvg.fvg_type == FVGType.BULLISH:
            has_fvg_bonus = True
            active_fvg = fvg_result.mitigation_fvg
        elif signal_type == "SHORT" and fvg_result.mitigation_fvg.fvg_type == FVGType.BEARISH:
            has_fvg_bonus = True
            active_fvg = fvg_result.mitigation_fvg

    # Calculate quality score
    quality_score = 1.0 + ms_score * 0.25  # Base: 1.0-1.5 from MS
    if has_fvg_bonus:
        quality_score += 0.5  # FVG bonus: +0.5

    if is_long_candidate:
        entry = close
        tp = pb_bot

        # SL: Use FVG boundary if available (tighter), else MS swing, else standard
        if active_fvg:
            sl = active_fvg.low * 0.998
        elif ms_result.last_swing_low and ms_result.last_swing_low.price < entry:
            sl = ms_result.last_swing_low.price * 0.998
        else:
            swing_low = float(df["low"].iloc[max(0, abs_index-20):abs_index].min())
            sl = min(swing_low * 0.998, baseline * 0.998)

        if tp <= entry or sl >= entry:
            return None, None, None, None, "Invalid Levels (LONG)", 0.0

        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})", 0.0

        fvg_tag = "+FVG" if has_fvg_bonus else ""
        return "LONG", entry, tp, sl, f"MS{fvg_tag}(R:{rr:.2f},Q:{quality_score:.1f})", quality_score

    else:  # SHORT
        entry = close
        tp = pb_top

        # SL: Use FVG boundary if available (tighter), else MS swing, else standard
        if active_fvg:
            sl = active_fvg.high * 1.002
        elif ms_result.last_swing_high and ms_result.last_swing_high.price > entry:
            sl = ms_result.last_swing_high.price * 1.002
        else:
            swing_high = float(df["high"].iloc[max(0, abs_index-20):abs_index].max())
            sl = max(swing_high * 1.002, baseline * 1.002)

        if tp >= entry or sl <= entry:
            return None, None, None, None, "Invalid Levels (SHORT)", 0.0

        risk = sl - entry
        reward = entry - tp
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})", 0.0

        fvg_tag = "+FVG" if has_fvg_bonus else ""
        return "SHORT", entry, tp, sl, f"MS{fvg_tag}(R:{rr:.2f},Q:{quality_score:.1f})", quality_score


def check_with_alphatrend_signal(
    df: pd.DataFrame,
    index: int = -2,
    min_rr: float = 1.5,
    rsi_limit: float = 75.0,  # Looser RSI
) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], str]:
    """
    WITH_AT signal check - SSL + PBEMA + AlphaTrend.

    Adds AlphaTrend confirmation to minimal:
    1. Price vs SSL baseline (direction)
    2. AlphaTrend buyers/sellers dominant (confirmation)
    3. AlphaTrend not flat (flow exists)
    4. PBEMA as target
    5. RSI filter (loose)
    """
    if df is None or df.empty or len(df) < 60:
        return None, None, None, None, "No Data"

    required_cols = [
        "open", "high", "low", "close", "baseline",
        "pb_ema_top", "pb_ema_bot", "rsi",
        "at_buyers_dominant", "at_sellers_dominant", "at_is_flat"
    ]
    for col in required_cols:
        if col not in df.columns:
            return None, None, None, None, f"Missing {col}"

    try:
        curr = df.iloc[index]
    except (IndexError, KeyError):
        return None, None, None, None, "Index Error"

    abs_index = index if index >= 0 else (len(df) + index)

    # Extract values
    close = float(curr["close"])
    baseline = float(curr["baseline"])
    pb_top = float(curr["pb_ema_top"])
    pb_bot = float(curr["pb_ema_bot"])
    rsi_val = float(curr["rsi"])
    at_buyers = bool(curr["at_buyers_dominant"])
    at_sellers = bool(curr["at_sellers_dominant"])
    at_flat = bool(curr["at_is_flat"])

    if any(pd.isna([close, baseline, pb_top, pb_bot, rsi_val])):
        return None, None, None, None, "NaN Values"

    # Skip if AlphaTrend is flat
    if at_flat:
        return None, None, None, None, "AT Flat"

    pbema_mid = (pb_top + pb_bot) / 2

    # LONG conditions
    is_long = (
        close > baseline and      # Price above baseline
        at_buyers and             # AlphaTrend confirms buyers
        pb_bot > close and        # Target exists (PBEMA above)
        rsi_val <= rsi_limit      # Not overbought
    )

    # SHORT conditions
    is_short = (
        close < baseline and      # Price below baseline
        at_sellers and            # AlphaTrend confirms sellers
        pb_top < close and        # Target exists (PBEMA below)
        rsi_val >= (100 - rsi_limit)  # Not oversold
    )

    if is_long:
        entry = close
        tp = pb_bot
        swing_low = float(df["low"].iloc[max(0, abs_index-20):abs_index].min())
        sl = min(swing_low * 0.998, baseline * 0.998)

        if tp <= entry or sl >= entry:
            return None, None, None, None, "Invalid Levels (LONG)"

        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "LONG", entry, tp, sl, f"WITH_AT(R:{rr:.2f})"

    elif is_short:
        entry = close
        tp = pb_top
        swing_high = float(df["high"].iloc[max(0, abs_index-20):abs_index].max())
        sl = max(swing_high * 1.002, baseline * 1.002)

        if tp >= entry or sl <= entry:
            return None, None, None, None, "Invalid Levels (SHORT)"

        risk = sl - entry
        reward = entry - tp
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return None, None, None, None, f"RR Too Low ({rr:.2f})"

        return "SHORT", entry, tp, sl, f"WITH_AT(R:{rr:.2f})"

    return None, None, None, None, "No Signal"


def simulate_trade(
    df: pd.DataFrame,
    signal_idx: int,
    signal_type: str,
    entry: float,
    tp: float,
    sl: float,
    position_size: float = 35.0,  # Fixed $35 position
) -> Dict:
    """
    Simulate a single trade from signal to exit.

    Returns trade result dict with PnL, exit reason, etc.
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


def run_backtest(
    df: pd.DataFrame,
    signal_func,
    mode_name: str,
    min_rr: float = 1.5,
    min_bars_between: int = 5,  # Minimum bars between trades
) -> Dict:
    """
    Run backtest with given signal function.

    Returns dict with trades, stats, etc.
    """
    trades = []
    last_exit_idx = -min_bars_between

    for i in range(60, len(df) - 10):  # Leave room for trade to play out
        # Skip if too close to last trade
        if i - last_exit_idx < min_bars_between:
            continue

        # Check for signal
        signal_type, entry, tp, sl, reason = signal_func(df, index=i, min_rr=min_rr)

        if signal_type is None:
            continue

        # Simulate trade
        result = simulate_trade(df, i, signal_type, entry, tp, sl)

        trade = {
            "signal_idx": i,
            "signal_time": df.index[i] if hasattr(df.index, '__iter__') else i,
            "type": signal_type,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "reason": reason,
            **result
        }
        trades.append(trade)

        last_exit_idx = result["exit_idx"]

    # Calculate stats
    if not trades:
        return {
            "mode": mode_name,
            "trades": [],
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_r": 0,
            "max_drawdown": 0,
        }

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = len(trades) - wins
    total_pnl = sum(t["pnl"] for t in trades)
    avg_r = sum(t["r_multiple"] for t in trades) / len(trades) if trades else 0

    # Calculate drawdown
    equity = [0]
    for t in trades:
        equity.append(equity[-1] + t["pnl"])
    peak = 0
    max_dd = 0
    for e in equity:
        peak = max(peak, e)
        dd = peak - e
        max_dd = max(max_dd, dd)

    return {
        "mode": mode_name,
        "trades": trades,
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) * 100 if trades else 0,
        "total_pnl": total_pnl,
        "avg_r": avg_r,
        "max_drawdown": max_dd,
    }


def fetch_data(symbol: str, timeframe: str, candles: int = 30000) -> pd.DataFrame:
    """Fetch historical data from Binance using existing client."""
    client = get_client()

    print(f"  Fetching {candles} candles for {symbol} {timeframe}...")

    try:
        # BinanceClient.get_klines returns DataFrame with timestamp, open, high, low, close, volume
        # We need to fetch in chunks of 1000 (Binance limit)
        all_dfs = []
        remaining = candles
        end_time = None

        while remaining > 0:
            chunk_size = min(remaining, 1000)

            if end_time:
                # Fetch older data - need raw API call
                import requests
                url = f"{client.BASE_URL}/klines"
                params = {
                    'symbol': symbol,
                    'interval': timeframe,
                    'limit': chunk_size,
                    'endTime': end_time
                }
                res = requests.get(url, params=params, timeout=10)
                if res.status_code != 200:
                    break
                data = res.json()
                if not data:
                    break

                df_chunk = pd.DataFrame(data).iloc[:, :6]
                df_chunk.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms', utc=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')
            else:
                df_chunk = client.get_klines(symbol=symbol, interval=timeframe, limit=chunk_size)

            if df_chunk.empty:
                break

            all_dfs.insert(0, df_chunk)
            remaining -= len(df_chunk)

            # Get oldest timestamp for next iteration
            if 'timestamp' in df_chunk.columns:
                end_time = int(df_chunk['timestamp'].iloc[0].timestamp() * 1000) - 1
            else:
                break

            if len(df_chunk) < chunk_size:
                break  # No more data available

        if not all_dfs:
            return pd.DataFrame()

        # Combine all chunks
        df = pd.concat(all_dfs, ignore_index=True)
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        df.set_index('timestamp', inplace=True)

        # Calculate indicators
        df = calculate_indicators(df, timeframe=timeframe)

        print(f"  Got {len(df)} candles, {df.index[0]} to {df.index[-1]}")

        return df

    except Exception as e:
        import traceback
        print(f"  Error fetching data: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Minimal SSL+PBEMA Backtest")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "LINKUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h", "4h"])
    parser.add_argument("--candles", type=int, default=30000)
    parser.add_argument("--mode", choices=["minimal", "with_ms", "with_fvg", "with_ms_fvg", "fvg_bonus", "with_at", "full", "all"], default="all")
    parser.add_argument("--min-rr", type=float, default=1.5)

    args = parser.parse_args()

    print("=" * 70)
    print("MINIMAL SSL + PBEMA BACKTEST")
    print("=" * 70)
    print(f"Symbols: {args.symbols}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Candles: {args.candles}")
    print(f"Mode: {args.mode}")
    print(f"Min RR: {args.min_rr}")
    print("=" * 70)
    print()

    # Import full signal function for comparison
    from strategies.ssl_flow import check_ssl_flow_signal

    # Results aggregation
    all_results = []

    for symbol in args.symbols:
        for tf in args.timeframes:
            print(f"\n{'='*60}")
            print(f"Testing {symbol} {tf}")
            print("=" * 60)

            # Fetch data
            df = fetch_data(symbol, tf, args.candles)
            if df.empty:
                print(f"  SKIPPED - no data")
                continue

            results = {}

            # Test MINIMAL
            if args.mode in ["minimal", "all"]:
                print("\n  [MINIMAL] SSL + PBEMA only...")
                results["MINIMAL"] = run_backtest(
                    df, check_minimal_signal, "MINIMAL", min_rr=args.min_rr
                )

            # Test WITH_MS (Market Structure)
            if args.mode in ["with_ms", "all"]:
                print("  [WITH_MS] SSL + PBEMA + Market Structure...")
                ms = MarketStructure(swing_length=5)
                def ms_signal(df, index=-2, min_rr=1.5):
                    return check_with_structure_signal(df, index, min_rr, ms)
                results["WITH_MS"] = run_backtest(
                    df, ms_signal, "WITH_MS", min_rr=args.min_rr
                )

            # Test WITH_FVG (Fair Value Gap) - MITIGATION MODE
            if args.mode in ["with_fvg", "all"]:
                print("  [WITH_FVG] SSL + PBEMA + FVG Mitigation...")
                fvg = FVGDetector(min_gap_percent=0.10)  # 0.10% minimum gap
                def fvg_signal(df, index=-2, min_rr=1.5):
                    return check_with_fvg_signal(df, index, min_rr, fvg, require_mitigation=True)
                results["WITH_FVG"] = run_backtest(
                    df, fvg_signal, "WITH_FVG", min_rr=args.min_rr
                )

            # Test WITH_MS_FVG (Market Structure + FVG Mitigation) - STRICT
            if args.mode in ["with_ms_fvg", "all"]:
                print("  [MS_FVG] SSL + PBEMA + MS + FVG Mitigation (strict)...")
                ms = MarketStructure(swing_length=5)
                fvg = FVGDetector(min_gap_percent=0.10)
                def ms_fvg_signal(df, index=-2, min_rr=1.5):
                    return check_with_ms_fvg_signal(df, index, min_rr, ms, fvg, min_ms_score=1.0, require_mitigation=True)
                results["MS_FVG"] = run_backtest(
                    df, ms_fvg_signal, "MS_FVG", min_rr=args.min_rr
                )

            # Test FVG_BONUS (Market Structure required + FVG as bonus)
            if args.mode in ["fvg_bonus", "all"]:
                print("  [FVG_BONUS] SSL + PBEMA + MS + FVG Bonus...")
                ms = MarketStructure(swing_length=5)
                fvg = FVGDetector(min_gap_percent=0.08)
                def fvg_bonus_signal(df, index=-2, min_rr=1.5):
                    result = check_with_fvg_bonus_signal(df, index, min_rr, ms, fvg, min_ms_score=1.0)
                    return result[:5]  # Discard quality_score for backtest compatibility
                results["FVG_BONUS"] = run_backtest(
                    df, fvg_bonus_signal, "FVG_BONUS", min_rr=args.min_rr
                )

            # Test WITH_AT
            if args.mode in ["with_at", "all"]:
                print("  [WITH_AT] SSL + PBEMA + AlphaTrend...")
                results["WITH_AT"] = run_backtest(
                    df, check_with_alphatrend_signal, "WITH_AT", min_rr=args.min_rr
                )

            # Test FULL (current strategy)
            if args.mode in ["full", "all"]:
                print("  [FULL] All current filters...")

                def full_signal(df, index=-2, min_rr=1.5):
                    result = check_ssl_flow_signal(
                        df, index=index, min_rr=min_rr,
                        timeframe=tf,
                        # Current defaults - all filters ON
                        skip_wick_rejection=True,  # Only this is off by default
                        skip_body_position=False,
                        skip_adx_filter=False,
                        skip_at_flat_filter=False,
                        skip_overlap_check=False,
                    )
                    return result[:5]  # signal_type, entry, tp, sl, reason

                results["FULL"] = run_backtest(
                    df, full_signal, "FULL", min_rr=args.min_rr
                )

            # Print comparison
            print(f"\n  {'Mode':<12} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'PnL':>10} {'Avg R':>8} {'Max DD':>10}")
            print("  " + "-" * 66)

            for mode, r in results.items():
                print(f"  {mode:<12} {r['total_trades']:>8} {r['wins']:>6} "
                      f"{r['win_rate']:>7.1f}% ${r['total_pnl']:>9.2f} "
                      f"{r['avg_r']:>7.2f}R ${r['max_drawdown']:>9.2f}")

                all_results.append({
                    "symbol": symbol,
                    "timeframe": tf,
                    "mode": mode,
                    **{k: v for k, v in r.items() if k != "trades"}
                })

    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY BY MODE")
    print("=" * 70)

    if all_results:
        df_results = pd.DataFrame(all_results)

        for mode in df_results["mode"].unique():
            mode_df = df_results[df_results["mode"] == mode]
            total_trades = mode_df["total_trades"].sum()
            total_wins = mode_df["wins"].sum()
            total_pnl = mode_df["total_pnl"].sum()
            avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

            print(f"\n{mode}:")
            print(f"  Total Trades: {total_trades}")
            print(f"  Total Wins: {total_wins} ({avg_wr:.1f}%)")
            print(f"  Total PnL: ${total_pnl:.2f}")
            print(f"  Max Drawdown: ${mode_df['max_drawdown'].max():.2f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("If MINIMAL shows edge but FULL doesn't, filters are blocking profitable trades.")
    print("If MINIMAL doesn't show edge, the core strategy needs revision.")
    print("=" * 70)


if __name__ == "__main__":
    main()
