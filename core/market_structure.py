"""
Market Structure Analysis Module

Detects swing highs/lows, trend direction, and structure breaks (BOS/CHoCH).
Used to validate trade direction alignment with market structure.

Key Concepts:
- Swing High: Local peak with N lower highs on each side
- Swing Low: Local trough with N higher lows on each side
- BOS (Break of Structure): Price breaks previous swing in trend direction
- CHoCH (Change of Character): First break against current trend (reversal signal)

Market Structure States:
- BULLISH: Higher Highs (HH) + Higher Lows (HL)
- BEARISH: Lower Highs (LH) + Lower Lows (LL)
- RANGING: No clear HH/HL or LH/LL pattern

Usage:
    from core.market_structure import MarketStructure, detect_structure

    ms = MarketStructure(swing_length=5)
    result = ms.analyze(df)

    # Check if LONG aligns with structure
    if result.trend == "BULLISH" and result.last_swing_type == "HL":
        # Good LONG setup - buying at higher low in uptrend
        pass
"""

from typing import List, Tuple, Optional, Dict, NamedTuple
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


class SwingType(Enum):
    """Type of swing point."""
    HIGH = "HIGH"
    LOW = "LOW"


class TrendType(Enum):
    """Market trend direction."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"


class StructureBreak(Enum):
    """Type of structure break."""
    BOS_BULLISH = "BOS_BULLISH"    # Break of structure - bullish continuation
    BOS_BEARISH = "BOS_BEARISH"    # Break of structure - bearish continuation
    CHOCH_BULLISH = "CHOCH_BULLISH"  # Change of character - bearish to bullish
    CHOCH_BEARISH = "CHOCH_BEARISH"  # Change of character - bullish to bearish
    NONE = "NONE"


@dataclass
class SwingPoint:
    """Represents a swing high or low."""
    index: int
    price: float
    swing_type: SwingType
    timestamp: Optional[pd.Timestamp] = None

    # Classification after trend analysis
    label: Optional[str] = None  # "HH", "HL", "LH", "LL"


@dataclass
class StructureResult:
    """Result of market structure analysis."""
    trend: TrendType
    trend_strength: float  # 0.0 to 1.0

    # Recent swing points
    swings: List[SwingPoint] = field(default_factory=list)
    last_swing_high: Optional[SwingPoint] = None
    last_swing_low: Optional[SwingPoint] = None

    # Structure breaks
    recent_break: StructureBreak = StructureBreak.NONE
    break_index: Optional[int] = None
    break_price: Optional[float] = None

    # For trade alignment
    long_aligned: bool = False   # True if LONG aligns with structure
    short_aligned: bool = False  # True if SHORT aligns with structure

    # Confidence and details
    hh_count: int = 0  # Higher High count
    hl_count: int = 0  # Higher Low count
    lh_count: int = 0  # Lower High count
    ll_count: int = 0  # Lower Low count

    def __str__(self) -> str:
        return (f"Structure({self.trend.value}, strength={self.trend_strength:.2f}, "
                f"HH={self.hh_count}, HL={self.hl_count}, LH={self.lh_count}, LL={self.ll_count}, "
                f"break={self.recent_break.value})")


class MarketStructure:
    """
    Market Structure analyzer for trend and swing point detection.

    Args:
        swing_length: Number of bars on each side to confirm swing (default: 5)
        min_swings: Minimum swings needed for trend determination (default: 4)
        lookback: Maximum bars to look back for analysis (default: 100)
    """

    def __init__(
        self,
        swing_length: int = 5,
        min_swings: int = 4,
        lookback: int = 100,
    ):
        self.swing_length = swing_length
        self.min_swings = min_swings
        self.lookback = lookback

    def find_swing_points(
        self,
        df: pd.DataFrame,
        end_index: Optional[int] = None,
    ) -> List[SwingPoint]:
        """
        Find swing highs and lows in price data.

        A swing high is confirmed when there are N lower highs on each side.
        A swing low is confirmed when there are N higher lows on each side.

        Args:
            df: DataFrame with 'high' and 'low' columns
            end_index: Last index to analyze (default: len(df))

        Returns:
            List of SwingPoint objects sorted by index
        """
        if df is None or len(df) < self.swing_length * 2 + 1:
            return []

        end_idx = end_index if end_index is not None else len(df)
        start_idx = max(0, end_idx - self.lookback)

        highs = df["high"].values
        lows = df["low"].values

        swings = []
        n = self.swing_length

        # Find swing highs and lows
        for i in range(start_idx + n, end_idx - n):
            # Check for swing high
            is_swing_high = True
            current_high = highs[i]

            for j in range(1, n + 1):
                if highs[i - j] >= current_high or highs[i + j] >= current_high:
                    is_swing_high = False
                    break

            if is_swing_high:
                timestamp = df.index[i] if hasattr(df.index, '__getitem__') else None
                swings.append(SwingPoint(
                    index=i,
                    price=current_high,
                    swing_type=SwingType.HIGH,
                    timestamp=timestamp,
                ))

            # Check for swing low
            is_swing_low = True
            current_low = lows[i]

            for j in range(1, n + 1):
                if lows[i - j] <= current_low or lows[i + j] <= current_low:
                    is_swing_low = False
                    break

            if is_swing_low:
                timestamp = df.index[i] if hasattr(df.index, '__getitem__') else None
                swings.append(SwingPoint(
                    index=i,
                    price=current_low,
                    swing_type=SwingType.LOW,
                    timestamp=timestamp,
                ))

        # Sort by index
        swings.sort(key=lambda x: x.index)

        return swings

    def label_swings(self, swings: List[SwingPoint]) -> List[SwingPoint]:
        """
        Label swing points as HH, HL, LH, LL based on comparison with previous swings.

        HH (Higher High): Swing high > previous swing high
        HL (Higher Low): Swing low > previous swing low
        LH (Lower High): Swing high < previous swing high
        LL (Lower Low): Swing low < previous swing low
        """
        if len(swings) < 2:
            return swings

        prev_high: Optional[SwingPoint] = None
        prev_low: Optional[SwingPoint] = None

        for swing in swings:
            if swing.swing_type == SwingType.HIGH:
                if prev_high is not None:
                    if swing.price > prev_high.price:
                        swing.label = "HH"
                    elif swing.price < prev_high.price:
                        swing.label = "LH"
                    else:
                        swing.label = "EH"  # Equal high
                prev_high = swing
            else:  # LOW
                if prev_low is not None:
                    if swing.price > prev_low.price:
                        swing.label = "HL"
                    elif swing.price < prev_low.price:
                        swing.label = "LL"
                    else:
                        swing.label = "EL"  # Equal low
                prev_low = swing

        return swings

    def determine_trend(
        self,
        swings: List[SwingPoint],
    ) -> Tuple[TrendType, float, Dict[str, int]]:
        """
        Determine trend based on labeled swing points.

        BULLISH: Predominantly HH and HL
        BEARISH: Predominantly LH and LL
        RANGING: Mixed or unclear

        Returns:
            (TrendType, strength, label_counts)
        """
        if len(swings) < self.min_swings:
            return TrendType.RANGING, 0.0, {}

        # Count labels
        counts = {"HH": 0, "HL": 0, "LH": 0, "LL": 0, "EH": 0, "EL": 0}
        for swing in swings:
            if swing.label and swing.label in counts:
                counts[swing.label] += 1

        bullish_score = counts["HH"] + counts["HL"]
        bearish_score = counts["LH"] + counts["LL"]
        total = bullish_score + bearish_score

        if total == 0:
            return TrendType.RANGING, 0.0, counts

        # Determine trend
        if bullish_score > bearish_score * 1.5:
            strength = bullish_score / total
            return TrendType.BULLISH, strength, counts
        elif bearish_score > bullish_score * 1.5:
            strength = bearish_score / total
            return TrendType.BEARISH, strength, counts
        else:
            # Mixed - ranging
            strength = abs(bullish_score - bearish_score) / total
            return TrendType.RANGING, strength, counts

    def detect_structure_break(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        current_trend: TrendType,
        end_index: Optional[int] = None,
    ) -> Tuple[StructureBreak, Optional[int], Optional[float]]:
        """
        Detect if price has broken market structure.

        BOS (Break of Structure):
        - In uptrend: Price breaks above last swing high (bullish continuation)
        - In downtrend: Price breaks below last swing low (bearish continuation)

        CHoCH (Change of Character):
        - In uptrend: Price breaks below last swing low (potential reversal)
        - In downtrend: Price breaks above last swing high (potential reversal)

        Returns:
            (StructureBreak type, break_index, break_price)
        """
        if len(swings) < 2:
            return StructureBreak.NONE, None, None

        end_idx = end_index if end_index is not None else len(df)

        # Get last swing high and low
        last_sh = None
        last_sl = None

        for swing in reversed(swings):
            if swing.swing_type == SwingType.HIGH and last_sh is None:
                last_sh = swing
            elif swing.swing_type == SwingType.LOW and last_sl is None:
                last_sl = swing
            if last_sh and last_sl:
                break

        if not last_sh or not last_sl:
            return StructureBreak.NONE, None, None

        # Check recent price action for breaks
        recent_start = max(last_sh.index, last_sl.index) + 1
        if recent_start >= end_idx:
            return StructureBreak.NONE, None, None

        recent_highs = df["high"].iloc[recent_start:end_idx].values
        recent_lows = df["low"].iloc[recent_start:end_idx].values

        # Check for break above swing high
        broke_high = False
        break_high_idx = None
        for i, h in enumerate(recent_highs):
            if h > last_sh.price:
                broke_high = True
                break_high_idx = recent_start + i
                break

        # Check for break below swing low
        broke_low = False
        break_low_idx = None
        for i, l in enumerate(recent_lows):
            if l < last_sl.price:
                broke_low = True
                break_low_idx = recent_start + i
                break

        # Determine break type based on current trend
        if current_trend == TrendType.BULLISH:
            if broke_high:
                return StructureBreak.BOS_BULLISH, break_high_idx, last_sh.price
            elif broke_low:
                return StructureBreak.CHOCH_BEARISH, break_low_idx, last_sl.price
        elif current_trend == TrendType.BEARISH:
            if broke_low:
                return StructureBreak.BOS_BEARISH, break_low_idx, last_sl.price
            elif broke_high:
                return StructureBreak.CHOCH_BULLISH, break_high_idx, last_sh.price
        else:  # RANGING
            if broke_high:
                return StructureBreak.BOS_BULLISH, break_high_idx, last_sh.price
            elif broke_low:
                return StructureBreak.BOS_BEARISH, break_low_idx, last_sl.price

        return StructureBreak.NONE, None, None

    def analyze(
        self,
        df: pd.DataFrame,
        index: Optional[int] = None,
    ) -> StructureResult:
        """
        Full market structure analysis.

        Args:
            df: DataFrame with OHLC data
            index: Index to analyze up to (default: last bar)

        Returns:
            StructureResult with trend, swings, breaks, and trade alignment
        """
        if df is None or len(df) < self.swing_length * 2 + self.min_swings:
            return StructureResult(trend=TrendType.RANGING, trend_strength=0.0)

        end_idx = index if index is not None else len(df)

        # Find and label swings
        swings = self.find_swing_points(df, end_idx)
        swings = self.label_swings(swings)

        # Determine trend
        trend, strength, counts = self.determine_trend(swings)

        # Detect structure breaks
        break_type, break_idx, break_price = self.detect_structure_break(
            df, swings, trend, end_idx
        )

        # Get last swing high/low
        last_sh = None
        last_sl = None
        for swing in reversed(swings):
            if swing.swing_type == SwingType.HIGH and last_sh is None:
                last_sh = swing
            elif swing.swing_type == SwingType.LOW and last_sl is None:
                last_sl = swing
            if last_sh and last_sl:
                break

        # Determine trade alignment
        # LONG aligned: Bullish trend OR CHoCH bullish OR at higher low
        long_aligned = (
            trend == TrendType.BULLISH or
            break_type == StructureBreak.CHOCH_BULLISH or
            (last_sl and last_sl.label == "HL")
        )

        # SHORT aligned: Bearish trend OR CHoCH bearish OR at lower high
        short_aligned = (
            trend == TrendType.BEARISH or
            break_type == StructureBreak.CHOCH_BEARISH or
            (last_sh and last_sh.label == "LH")
        )

        return StructureResult(
            trend=trend,
            trend_strength=strength,
            swings=swings,
            last_swing_high=last_sh,
            last_swing_low=last_sl,
            recent_break=break_type,
            break_index=break_idx,
            break_price=break_price,
            long_aligned=long_aligned,
            short_aligned=short_aligned,
            hh_count=counts.get("HH", 0),
            hl_count=counts.get("HL", 0),
            lh_count=counts.get("LH", 0),
            ll_count=counts.get("LL", 0),
        )

    def get_nearest_swing(
        self,
        df: pd.DataFrame,
        index: int,
        swing_type: SwingType,
        direction: str = "below",  # "below" or "above"
    ) -> Optional[SwingPoint]:
        """
        Find nearest swing point in specified direction.

        Useful for finding support (nearest swing low below) or
        resistance (nearest swing high above).

        Args:
            df: DataFrame with OHLC data
            index: Current index
            swing_type: HIGH or LOW
            direction: "below" (lower price) or "above" (higher price)

        Returns:
            Nearest SwingPoint or None
        """
        swings = self.find_swing_points(df, index)
        current_price = float(df["close"].iloc[index])

        candidates = []
        for swing in swings:
            if swing.swing_type != swing_type:
                continue
            if direction == "below" and swing.price < current_price:
                candidates.append(swing)
            elif direction == "above" and swing.price > current_price:
                candidates.append(swing)

        if not candidates:
            return None

        # Return nearest by price
        if direction == "below":
            return max(candidates, key=lambda x: x.price)
        else:
            return min(candidates, key=lambda x: x.price)


# Convenience functions

def detect_structure(
    df: pd.DataFrame,
    index: Optional[int] = None,
    swing_length: int = 5,
) -> StructureResult:
    """
    Quick market structure detection.

    Args:
        df: DataFrame with OHLC data
        index: Index to analyze (default: last bar)
        swing_length: Swing detection sensitivity

    Returns:
        StructureResult with trend and alignment info
    """
    ms = MarketStructure(swing_length=swing_length)
    return ms.analyze(df, index)


def is_trade_aligned(
    df: pd.DataFrame,
    signal_type: str,
    index: Optional[int] = None,
    swing_length: int = 5,
) -> Tuple[bool, str]:
    """
    Check if trade direction aligns with market structure.

    Args:
        df: DataFrame with OHLC data
        signal_type: "LONG" or "SHORT"
        index: Index to check (default: last bar)
        swing_length: Swing detection sensitivity

    Returns:
        (is_aligned, reason_string)
    """
    result = detect_structure(df, index, swing_length)

    if signal_type == "LONG":
        if result.long_aligned:
            reasons = []
            if result.trend == TrendType.BULLISH:
                reasons.append(f"Bullish trend ({result.trend_strength:.0%})")
            if result.recent_break == StructureBreak.CHOCH_BULLISH:
                reasons.append("CHoCH bullish")
            if result.last_swing_low and result.last_swing_low.label == "HL":
                reasons.append("At Higher Low")
            return True, " + ".join(reasons) if reasons else "Aligned"
        else:
            return False, f"Counter-trend ({result.trend.value})"

    elif signal_type == "SHORT":
        if result.short_aligned:
            reasons = []
            if result.trend == TrendType.BEARISH:
                reasons.append(f"Bearish trend ({result.trend_strength:.0%})")
            if result.recent_break == StructureBreak.CHOCH_BEARISH:
                reasons.append("CHoCH bearish")
            if result.last_swing_high and result.last_swing_high.label == "LH":
                reasons.append("At Lower High")
            return True, " + ".join(reasons) if reasons else "Aligned"
        else:
            return False, f"Counter-trend ({result.trend.value})"

    return False, "Invalid signal type"


def get_structure_score(
    df: pd.DataFrame,
    signal_type: str,
    index: Optional[int] = None,
    swing_length: int = 5,
) -> Tuple[float, StructureResult]:
    """
    Get a score (0-2) for how well trade aligns with market structure.

    Score breakdown:
    - 0.0: Counter-trend trade
    - 0.5: Ranging market
    - 1.0: Trend-aligned trade
    - 1.5: Trend-aligned + at key level (HL/LH)
    - 2.0: Trend-aligned + structure break confirmation

    Args:
        df: DataFrame with OHLC data
        signal_type: "LONG" or "SHORT"
        index: Index to check
        swing_length: Swing detection sensitivity

    Returns:
        (score, StructureResult)
    """
    result = detect_structure(df, index, swing_length)
    score = 0.0

    if signal_type == "LONG":
        # Base score from trend
        if result.trend == TrendType.BULLISH:
            score = 1.0
        elif result.trend == TrendType.RANGING:
            score = 0.5
        else:  # BEARISH - counter-trend
            score = 0.0

        # Bonus for key level
        if result.last_swing_low and result.last_swing_low.label == "HL":
            score += 0.5

        # Bonus for structure break
        if result.recent_break in [StructureBreak.BOS_BULLISH, StructureBreak.CHOCH_BULLISH]:
            score += 0.5

    elif signal_type == "SHORT":
        # Base score from trend
        if result.trend == TrendType.BEARISH:
            score = 1.0
        elif result.trend == TrendType.RANGING:
            score = 0.5
        else:  # BULLISH - counter-trend
            score = 0.0

        # Bonus for key level
        if result.last_swing_high and result.last_swing_high.label == "LH":
            score += 0.5

        # Bonus for structure break
        if result.recent_break in [StructureBreak.BOS_BEARISH, StructureBreak.CHOCH_BEARISH]:
            score += 0.5

    return min(score, 2.0), result


# DataFrame integration helper
def add_structure_columns(
    df: pd.DataFrame,
    swing_length: int = 5,
) -> pd.DataFrame:
    """
    Add market structure columns to DataFrame.

    Adds columns:
    - ms_trend: "BULLISH", "BEARISH", "RANGING"
    - ms_strength: 0.0-1.0
    - ms_long_aligned: True/False
    - ms_short_aligned: True/False
    - ms_swing_high: Last swing high price
    - ms_swing_low: Last swing low price

    Args:
        df: DataFrame with OHLC data
        swing_length: Swing detection sensitivity

    Returns:
        DataFrame with structure columns added
    """
    df = df.copy()
    ms = MarketStructure(swing_length=swing_length)

    # Initialize columns
    df["ms_trend"] = "RANGING"
    df["ms_strength"] = 0.0
    df["ms_long_aligned"] = False
    df["ms_short_aligned"] = False
    df["ms_swing_high"] = np.nan
    df["ms_swing_low"] = np.nan

    # Calculate for each bar (expensive but comprehensive)
    min_idx = swing_length * 2 + ms.min_swings

    for i in range(min_idx, len(df)):
        result = ms.analyze(df, i)

        df.iloc[i, df.columns.get_loc("ms_trend")] = result.trend.value
        df.iloc[i, df.columns.get_loc("ms_strength")] = result.trend_strength
        df.iloc[i, df.columns.get_loc("ms_long_aligned")] = result.long_aligned
        df.iloc[i, df.columns.get_loc("ms_short_aligned")] = result.short_aligned

        if result.last_swing_high:
            df.iloc[i, df.columns.get_loc("ms_swing_high")] = result.last_swing_high.price
        if result.last_swing_low:
            df.iloc[i, df.columns.get_loc("ms_swing_low")] = result.last_swing_low.price

    return df
