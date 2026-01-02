"""
Fair Value Gap (FVG) Detection Module

Detects and tracks Fair Value Gaps (imbalances) in price action.
FVGs represent areas where price moved too fast, leaving liquidity voids.

Key Concepts:
- Bullish FVG: Gap UP - Candle1.high < Candle3.low (price jumped up)
- Bearish FVG: Gap DOWN - Candle1.low > Candle3.high (price dropped)
- Mitigation: When price returns to fill the gap (retest)
- Unfilled FVG: Gap not yet tested - acts as support/resistance

Trading Application:
- Entry: Enter when price returns to FVG (mitigation)
- SL: Place beyond FVG boundary
- TP: Next unfilled FVG in target direction

Visual:
    Bullish FVG:                Bearish FVG:

        ┌───┐ Candle 3              ┌───┐ Candle 1
        │   │                       │   │
        └───┘                       └───┘
          ↑ GAP (FVG)                 ↑
        ┌───┐ Candle 2              ┌───┐ Candle 2
        │   │  Big move             │   │  Big move
        └───┘                       └───┘
          ↑ GAP (FVG)                 ↑ GAP (FVG)
        ┌───┐ Candle 1              ┌───┐ Candle 3
        │   │                       │   │
        └───┘                       └───┘

Usage:
    from core.fvg_detector import FVGDetector, detect_fvgs

    detector = FVGDetector(min_gap_percent=0.1)
    fvgs = detector.find_fvgs(df)

    # Check if price is in an FVG
    current_fvg = detector.get_fvg_at_price(df, fvgs, current_price)
"""

from typing import List, Tuple, Optional, Dict
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import numpy as np


class FVGType(Enum):
    """Type of Fair Value Gap."""
    BULLISH = "BULLISH"  # Gap up - price jumped
    BEARISH = "BEARISH"  # Gap down - price dropped


class FVGStatus(Enum):
    """Status of FVG fill/mitigation."""
    UNFILLED = "UNFILLED"      # Gap not touched yet
    PARTIALLY_FILLED = "PARTIAL"  # Gap partially filled
    FILLED = "FILLED"          # Gap completely filled
    MITIGATED = "MITIGATED"    # Price touched FVG (entry signal)


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap."""
    index: int              # Index where FVG was created (middle candle)
    fvg_type: FVGType
    high: float             # Upper boundary of gap
    low: float              # Lower boundary of gap
    size: float             # Gap size in price
    size_percent: float     # Gap size as percentage
    timestamp: Optional[pd.Timestamp] = None

    # Status tracking
    status: FVGStatus = FVGStatus.UNFILLED
    fill_percent: float = 0.0
    mitigation_index: Optional[int] = None
    mitigation_price: Optional[float] = None

    @property
    def mid(self) -> float:
        """Middle price of FVG."""
        return (self.high + self.low) / 2

    @property
    def is_valid(self) -> bool:
        """Check if FVG is still valid (unfilled or partial)."""
        return self.status in [FVGStatus.UNFILLED, FVGStatus.PARTIALLY_FILLED]

    def __str__(self) -> str:
        return (f"FVG({self.fvg_type.value}, {self.low:.2f}-{self.high:.2f}, "
                f"{self.size_percent:.2f}%, {self.status.value})")


@dataclass
class FVGResult:
    """Result of FVG analysis for a specific price point."""
    in_fvg: bool = False
    fvg: Optional[FairValueGap] = None
    distance_to_nearest: float = float('inf')
    nearest_bullish: Optional[FairValueGap] = None
    nearest_bearish: Optional[FairValueGap] = None

    # Mitigation detection (price returned to FVG)
    is_mitigation: bool = False  # True if price just returned to FVG
    mitigation_fvg: Optional[FairValueGap] = None

    # Scoring for trade quality
    score: float = 0.0  # 0-2 based on FVG confluence

    def __str__(self) -> str:
        if self.is_mitigation:
            return f"FVGResult(MITIGATION: {self.mitigation_fvg}, score={self.score:.1f})"
        if self.in_fvg:
            return f"FVGResult(IN FVG: {self.fvg}, score={self.score:.1f})"
        return f"FVGResult(nearest={self.distance_to_nearest:.4f}, score={self.score:.1f})"


class FVGDetector:
    """
    Fair Value Gap detector and tracker.

    Args:
        min_gap_percent: Minimum gap size as percentage (default: 0.1%)
        max_fvgs: Maximum FVGs to track (default: 50)
        lookback: Maximum bars to look back (default: 200)
    """

    def __init__(
        self,
        min_gap_percent: float = 0.1,
        max_fvgs: int = 50,
        lookback: int = 200,
    ):
        self.min_gap_percent = min_gap_percent
        self.max_fvgs = max_fvgs
        self.lookback = lookback

    def find_fvgs(
        self,
        df: pd.DataFrame,
        end_index: Optional[int] = None,
    ) -> List[FairValueGap]:
        """
        Find all Fair Value Gaps in price data.

        Args:
            df: DataFrame with 'high' and 'low' columns
            end_index: Last index to analyze (default: len(df))

        Returns:
            List of FairValueGap objects sorted by index (newest last)
        """
        if df is None or len(df) < 3:
            return []

        end_idx = end_index if end_index is not None else len(df)
        start_idx = max(0, end_idx - self.lookback)

        highs = df["high"].values
        lows = df["low"].values

        fvgs = []

        # Need at least 3 candles: i-1, i, i+1
        for i in range(start_idx + 1, end_idx - 1):
            c1_high = highs[i - 1]  # Candle 1 (before)
            c1_low = lows[i - 1]
            c3_high = highs[i + 1]  # Candle 3 (after)
            c3_low = lows[i + 1]

            # Check for Bullish FVG: Candle1.high < Candle3.low
            if c1_high < c3_low:
                gap_low = c1_high
                gap_high = c3_low
                gap_size = gap_high - gap_low
                gap_percent = (gap_size / gap_low) * 100 if gap_low > 0 else 0

                if gap_percent >= self.min_gap_percent:
                    timestamp = df.index[i] if hasattr(df.index, '__getitem__') else None
                    fvgs.append(FairValueGap(
                        index=i,
                        fvg_type=FVGType.BULLISH,
                        high=gap_high,
                        low=gap_low,
                        size=gap_size,
                        size_percent=gap_percent,
                        timestamp=timestamp,
                    ))

            # Check for Bearish FVG: Candle1.low > Candle3.high
            if c1_low > c3_high:
                gap_high = c1_low
                gap_low = c3_high
                gap_size = gap_high - gap_low
                gap_percent = (gap_size / gap_low) * 100 if gap_low > 0 else 0

                if gap_percent >= self.min_gap_percent:
                    timestamp = df.index[i] if hasattr(df.index, '__getitem__') else None
                    fvgs.append(FairValueGap(
                        index=i,
                        fvg_type=FVGType.BEARISH,
                        high=gap_high,
                        low=gap_low,
                        size=gap_size,
                        size_percent=gap_percent,
                        timestamp=timestamp,
                    ))

        # Sort by index and limit count
        fvgs.sort(key=lambda x: x.index)

        if len(fvgs) > self.max_fvgs:
            fvgs = fvgs[-self.max_fvgs:]

        return fvgs

    def update_fvg_status(
        self,
        fvgs: List[FairValueGap],
        df: pd.DataFrame,
        end_index: Optional[int] = None,
    ) -> List[FairValueGap]:
        """
        Update FVG status based on price action after creation.

        Checks if each FVG has been mitigated (price returned to gap).

        Args:
            fvgs: List of FVGs to update
            df: DataFrame with price data
            end_index: Last index to check

        Returns:
            Updated list of FVGs
        """
        if not fvgs or df is None:
            return fvgs

        end_idx = end_index if end_index is not None else len(df)
        highs = df["high"].values
        lows = df["low"].values

        for fvg in fvgs:
            if fvg.status == FVGStatus.FILLED:
                continue  # Already fully filled

            # Check price action after FVG creation
            check_start = fvg.index + 2  # Start after FVG formation
            if check_start >= end_idx:
                continue

            for i in range(check_start, end_idx):
                candle_high = highs[i]
                candle_low = lows[i]

                if fvg.fvg_type == FVGType.BULLISH:
                    # Bullish FVG filled when price drops into it
                    if candle_low <= fvg.high:
                        # Price touched FVG
                        if fvg.status == FVGStatus.UNFILLED:
                            fvg.status = FVGStatus.MITIGATED
                            fvg.mitigation_index = i
                            fvg.mitigation_price = min(candle_low, fvg.high)

                        # Calculate fill percentage
                        if candle_low <= fvg.low:
                            fvg.status = FVGStatus.FILLED
                            fvg.fill_percent = 100.0
                            break
                        else:
                            fill = (fvg.high - candle_low) / fvg.size * 100
                            fvg.fill_percent = max(fvg.fill_percent, fill)
                            if fvg.fill_percent > 0 and fvg.status == FVGStatus.MITIGATED:
                                fvg.status = FVGStatus.PARTIALLY_FILLED

                else:  # BEARISH
                    # Bearish FVG filled when price rises into it
                    if candle_high >= fvg.low:
                        # Price touched FVG
                        if fvg.status == FVGStatus.UNFILLED:
                            fvg.status = FVGStatus.MITIGATED
                            fvg.mitigation_index = i
                            fvg.mitigation_price = max(candle_high, fvg.low)

                        # Calculate fill percentage
                        if candle_high >= fvg.high:
                            fvg.status = FVGStatus.FILLED
                            fvg.fill_percent = 100.0
                            break
                        else:
                            fill = (candle_high - fvg.low) / fvg.size * 100
                            fvg.fill_percent = max(fvg.fill_percent, fill)
                            if fvg.fill_percent > 0 and fvg.status == FVGStatus.MITIGATED:
                                fvg.status = FVGStatus.PARTIALLY_FILLED

        return fvgs

    def get_fvg_at_price(
        self,
        price: float,
        fvgs: List[FairValueGap],
        fvg_type: Optional[FVGType] = None,
    ) -> Optional[FairValueGap]:
        """
        Find FVG that contains the given price.

        Args:
            price: Price to check
            fvgs: List of FVGs
            fvg_type: Filter by type (optional)

        Returns:
            FVG containing price, or None
        """
        for fvg in reversed(fvgs):  # Check newest first
            if fvg_type and fvg.fvg_type != fvg_type:
                continue
            if fvg.low <= price <= fvg.high:
                return fvg
        return None

    def get_nearest_fvg(
        self,
        price: float,
        fvgs: List[FairValueGap],
        direction: str = "any",  # "above", "below", "any"
        fvg_type: Optional[FVGType] = None,
        valid_only: bool = True,
    ) -> Optional[FairValueGap]:
        """
        Find nearest FVG to given price.

        Args:
            price: Current price
            fvgs: List of FVGs
            direction: "above" (higher), "below" (lower), or "any"
            fvg_type: Filter by type
            valid_only: Only consider unfilled/partial FVGs

        Returns:
            Nearest FVG or None
        """
        candidates = []

        for fvg in fvgs:
            if fvg_type and fvg.fvg_type != fvg_type:
                continue
            if valid_only and not fvg.is_valid:
                continue

            fvg_mid = fvg.mid

            if direction == "above" and fvg_mid <= price:
                continue
            if direction == "below" and fvg_mid >= price:
                continue

            distance = abs(fvg_mid - price)
            candidates.append((distance, fvg))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def detect_mitigation(
        self,
        df: pd.DataFrame,
        fvgs: List[FairValueGap],
        index: int,
        lookback: int = 5,
    ) -> Tuple[bool, Optional[FairValueGap]]:
        """
        Detect if current bar is a FRESH mitigation (price returned to FVG).

        Mitigation = Price was OUTSIDE FVG, then RETURNED to it.
        This is the optimal entry point - not just being in FVG.

        Args:
            df: DataFrame with OHLC
            fvgs: List of FVGs
            index: Current index
            lookback: How many bars to check for "was outside"

        Returns:
            (is_mitigation, fvg_being_mitigated)
        """
        if not fvgs or index < lookback + 3:
            return False, None

        current_price = float(df["close"].iloc[index])
        current_low = float(df["low"].iloc[index])
        current_high = float(df["high"].iloc[index])

        for fvg in reversed(fvgs):  # Check newest first
            # Skip if FVG is too old (more than 50 bars ago)
            if index - fvg.index > 50:
                continue

            # Skip if already fully filled
            if fvg.status == FVGStatus.FILLED:
                continue

            # Check if current bar touches FVG
            if fvg.fvg_type == FVGType.BULLISH:
                # Bullish FVG: price should drop INTO it from above
                current_touches = current_low <= fvg.high and current_low >= fvg.low

                if not current_touches:
                    continue

                # Check if previous bars were ABOVE the FVG (price was outside)
                was_outside = True
                for i in range(1, min(lookback + 1, index - fvg.index)):
                    prev_low = float(df["low"].iloc[index - i])
                    if prev_low <= fvg.high:  # Was inside or below FVG
                        was_outside = False
                        break

                if was_outside:
                    return True, fvg

            else:  # BEARISH
                # Bearish FVG: price should rise INTO it from below
                current_touches = current_high >= fvg.low and current_high <= fvg.high

                if not current_touches:
                    continue

                # Check if previous bars were BELOW the FVG (price was outside)
                was_outside = True
                for i in range(1, min(lookback + 1, index - fvg.index)):
                    prev_high = float(df["high"].iloc[index - i])
                    if prev_high >= fvg.low:  # Was inside or above FVG
                        was_outside = False
                        break

                if was_outside:
                    return True, fvg

        return False, None

    def analyze(
        self,
        df: pd.DataFrame,
        index: Optional[int] = None,
        signal_type: Optional[str] = None,  # "LONG" or "SHORT"
    ) -> FVGResult:
        """
        Full FVG analysis for a price point.

        Args:
            df: DataFrame with OHLC data
            index: Index to analyze (default: last bar)
            signal_type: Trade direction for scoring

        Returns:
            FVGResult with FVG info and score
        """
        if df is None or len(df) < 10:
            return FVGResult()

        end_idx = index if index is not None else len(df) - 1
        current_price = float(df["close"].iloc[end_idx])

        # Find and update FVGs
        fvgs = self.find_fvgs(df, end_idx)
        fvgs = self.update_fvg_status(fvgs, df, end_idx)

        # Check if price is in an FVG
        current_fvg = self.get_fvg_at_price(current_price, fvgs)

        # CRITICAL: Check for MITIGATION (fresh return to FVG)
        is_mitigation, mitigation_fvg = self.detect_mitigation(df, fvgs, end_idx)

        # Find nearest FVGs
        nearest_bullish = self.get_nearest_fvg(
            current_price, fvgs, direction="any",
            fvg_type=FVGType.BULLISH, valid_only=True
        )
        nearest_bearish = self.get_nearest_fvg(
            current_price, fvgs, direction="any",
            fvg_type=FVGType.BEARISH, valid_only=True
        )

        # Calculate distance to nearest
        distance = float('inf')
        if nearest_bullish:
            distance = min(distance, abs(nearest_bullish.mid - current_price) / current_price)
        if nearest_bearish:
            distance = min(distance, abs(nearest_bearish.mid - current_price) / current_price)

        # Calculate score based on FVG confluence
        score = 0.0

        if signal_type == "LONG":
            # MITIGATION of bullish FVG = BEST entry for LONG
            if is_mitigation and mitigation_fvg and mitigation_fvg.fvg_type == FVGType.BULLISH:
                score = 2.0  # Fresh mitigation - excellent!
            elif current_fvg and current_fvg.fvg_type == FVGType.BULLISH:
                score = 1.0  # In bullish FVG but not fresh mitigation
            elif nearest_bullish and abs(nearest_bullish.mid - current_price) / current_price < 0.005:
                score = 0.5  # Very close to bullish FVG

            # Bonus if unfilled bearish FVG above (potential target)
            unfilled_bearish_above = self.get_nearest_fvg(
                current_price, fvgs, direction="above",
                fvg_type=FVGType.BEARISH, valid_only=True
            )
            if unfilled_bearish_above:
                score += 0.5

        elif signal_type == "SHORT":
            # MITIGATION of bearish FVG = BEST entry for SHORT
            if is_mitigation and mitigation_fvg and mitigation_fvg.fvg_type == FVGType.BEARISH:
                score = 2.0  # Fresh mitigation - excellent!
            elif current_fvg and current_fvg.fvg_type == FVGType.BEARISH:
                score = 1.0  # In bearish FVG but not fresh mitigation
            elif nearest_bearish and abs(nearest_bearish.mid - current_price) / current_price < 0.005:
                score = 0.5  # Very close to bearish FVG

            # Bonus if unfilled bullish FVG below (potential target)
            unfilled_bullish_below = self.get_nearest_fvg(
                current_price, fvgs, direction="below",
                fvg_type=FVGType.BULLISH, valid_only=True
            )
            if unfilled_bullish_below:
                score += 0.5

        return FVGResult(
            in_fvg=current_fvg is not None,
            fvg=current_fvg,
            distance_to_nearest=distance,
            nearest_bullish=nearest_bullish,
            nearest_bearish=nearest_bearish,
            is_mitigation=is_mitigation,
            mitigation_fvg=mitigation_fvg,
            score=min(score, 2.0),
        )

    def get_fvg_levels(
        self,
        df: pd.DataFrame,
        index: Optional[int] = None,
        signal_type: str = "LONG",
    ) -> Dict[str, Optional[float]]:
        """
        Get FVG-based SL and TP levels.

        For LONG:
        - SL: Below nearest bullish FVG low (support)
        - TP: Nearest unfilled bearish FVG (resistance)

        For SHORT:
        - SL: Above nearest bearish FVG high (resistance)
        - TP: Nearest unfilled bullish FVG (support)

        Args:
            df: DataFrame with OHLC
            index: Current index
            signal_type: "LONG" or "SHORT"

        Returns:
            Dict with 'sl' and 'tp' levels (may be None)
        """
        end_idx = index if index is not None else len(df) - 1
        current_price = float(df["close"].iloc[end_idx])

        fvgs = self.find_fvgs(df, end_idx)
        fvgs = self.update_fvg_status(fvgs, df, end_idx)

        result = {"sl": None, "tp": None, "sl_fvg": None, "tp_fvg": None}

        if signal_type == "LONG":
            # SL: Below nearest bullish FVG
            bullish_below = self.get_nearest_fvg(
                current_price, fvgs, direction="below",
                fvg_type=FVGType.BULLISH, valid_only=True
            )
            if bullish_below:
                result["sl"] = bullish_below.low * 0.998  # Slightly below
                result["sl_fvg"] = bullish_below

            # TP: Nearest unfilled bearish FVG above
            bearish_above = self.get_nearest_fvg(
                current_price, fvgs, direction="above",
                fvg_type=FVGType.BEARISH, valid_only=True
            )
            if bearish_above:
                result["tp"] = bearish_above.low  # Lower bound of bearish FVG
                result["tp_fvg"] = bearish_above

        else:  # SHORT
            # SL: Above nearest bearish FVG
            bearish_above = self.get_nearest_fvg(
                current_price, fvgs, direction="above",
                fvg_type=FVGType.BEARISH, valid_only=True
            )
            if bearish_above:
                result["sl"] = bearish_above.high * 1.002  # Slightly above
                result["sl_fvg"] = bearish_above

            # TP: Nearest unfilled bullish FVG below
            bullish_below = self.get_nearest_fvg(
                current_price, fvgs, direction="below",
                fvg_type=FVGType.BULLISH, valid_only=True
            )
            if bullish_below:
                result["tp"] = bullish_below.high  # Upper bound of bullish FVG
                result["tp_fvg"] = bullish_below

        return result


# Convenience functions

def detect_fvgs(
    df: pd.DataFrame,
    index: Optional[int] = None,
    min_gap_percent: float = 0.1,
) -> List[FairValueGap]:
    """
    Quick FVG detection.

    Args:
        df: DataFrame with OHLC data
        index: End index (default: last bar)
        min_gap_percent: Minimum gap size

    Returns:
        List of FVGs
    """
    detector = FVGDetector(min_gap_percent=min_gap_percent)
    fvgs = detector.find_fvgs(df, index)
    return detector.update_fvg_status(fvgs, df, index)


def get_fvg_score(
    df: pd.DataFrame,
    signal_type: str,
    index: Optional[int] = None,
    min_gap_percent: float = 0.1,
) -> Tuple[float, FVGResult]:
    """
    Get FVG-based score for trade quality.

    Args:
        df: DataFrame with OHLC
        signal_type: "LONG" or "SHORT"
        index: Index to check
        min_gap_percent: Minimum FVG size

    Returns:
        (score, FVGResult)
    """
    detector = FVGDetector(min_gap_percent=min_gap_percent)
    result = detector.analyze(df, index, signal_type)
    return result.score, result


def is_in_fvg(
    df: pd.DataFrame,
    index: Optional[int] = None,
    fvg_type: Optional[FVGType] = None,
) -> Tuple[bool, Optional[FairValueGap]]:
    """
    Check if current price is inside an FVG.

    Args:
        df: DataFrame with OHLC
        index: Index to check
        fvg_type: Filter by type

    Returns:
        (is_in_fvg, fvg_if_found)
    """
    detector = FVGDetector()
    result = detector.analyze(df, index)

    if result.in_fvg:
        if fvg_type is None or result.fvg.fvg_type == fvg_type:
            return True, result.fvg
    return False, None


# DataFrame integration helper
def add_fvg_columns(
    df: pd.DataFrame,
    min_gap_percent: float = 0.1,
) -> pd.DataFrame:
    """
    Add FVG-related columns to DataFrame.

    Adds columns:
    - fvg_bullish: True if bullish FVG detected at this bar
    - fvg_bearish: True if bearish FVG detected at this bar
    - in_bullish_fvg: True if price is inside a bullish FVG
    - in_bearish_fvg: True if price is inside a bearish FVG
    - nearest_fvg_distance: Distance to nearest FVG (as ratio)

    Args:
        df: DataFrame with OHLC data
        min_gap_percent: Minimum FVG size

    Returns:
        DataFrame with FVG columns added
    """
    df = df.copy()
    detector = FVGDetector(min_gap_percent=min_gap_percent)

    # Initialize columns
    df["fvg_bullish"] = False
    df["fvg_bearish"] = False
    df["in_bullish_fvg"] = False
    df["in_bearish_fvg"] = False
    df["nearest_fvg_distance"] = np.nan

    # Find all FVGs first
    all_fvgs = detector.find_fvgs(df)

    # Mark FVG creation bars
    for fvg in all_fvgs:
        if fvg.fvg_type == FVGType.BULLISH:
            df.iloc[fvg.index, df.columns.get_loc("fvg_bullish")] = True
        else:
            df.iloc[fvg.index, df.columns.get_loc("fvg_bearish")] = True

    # Check each bar for FVG containment
    for i in range(len(df)):
        if i < 10:
            continue

        result = detector.analyze(df, i)

        if result.in_fvg:
            if result.fvg.fvg_type == FVGType.BULLISH:
                df.iloc[i, df.columns.get_loc("in_bullish_fvg")] = True
            else:
                df.iloc[i, df.columns.get_loc("in_bearish_fvg")] = True

        if result.distance_to_nearest != float('inf'):
            df.iloc[i, df.columns.get_loc("nearest_fvg_distance")] = result.distance_to_nearest

    return df
