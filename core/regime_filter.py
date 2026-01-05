"""
Enhanced Regime Filter Module - Priority 2 Implementation

This module implements ex-ante (before-the-fact) regime detection to avoid
trading in unfavorable market conditions.

Key Features:
1. BTC Leader Regime - If BTC is ranging, skip all trades
2. Multi-Factor Detection - ADX + ATR + Volatility percentile
3. Regime Confidence Score - Continuous 0-1 score, not just binary
4. Ex-Ante Guarantee - Never uses future data in calculations

Based on Hedge Fund Due Diligence Report findings:
- Strategy only works in TRENDING regimes
- H1 2025 lost money (ranging), H2 2025 made money (trending)
- 79% win rate only in favorable conditions

Usage:
    from core.regime_filter import RegimeFilter, RegimeType

    regime_filter = RegimeFilter()
    regime = regime_filter.detect_regime(df, index=-2)

    if regime.should_trade:
        # Proceed with signal generation
    else:
        # Skip this bar
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np

from .logging_config import get_logger

_logger = get_logger(__name__)


class RegimeType(Enum):
    """Market regime classification."""
    TRENDING_STRONG = "TRENDING_STRONG"   # ADX > 30, clear direction
    TRENDING = "TRENDING"                  # ADX 20-30, moderate trend
    TRANSITIONAL = "TRANSITIONAL"          # ADX 15-20, uncertain
    RANGING = "RANGING"                    # ADX < 15, sideways
    VOLATILE = "VOLATILE"                  # High ATR percentile, choppy


@dataclass
class RegimeResult:
    """Result of regime detection."""
    regime: RegimeType
    should_trade: bool
    confidence: float  # 0.0 to 1.0
    adx_current: float
    adx_avg: float
    atr_percentile: float
    details: str
    btc_regime: Optional[RegimeType] = None  # BTC's regime (for alts)
    btc_aligned: bool = True  # Is this symbol aligned with BTC?


class RegimeFilter:
    """
    Enhanced regime filter for SSL Flow strategy.

    This filter implements multiple layers of regime detection:
    1. Symbol-level regime (ADX-based)
    2. Volatility regime (ATR percentile)
    3. BTC leader regime (market-wide filter)
    """

    def __init__(
        self,
        adx_trending_threshold: float = 25.0,
        adx_ranging_threshold: float = 18.0,
        adx_strong_threshold: float = 30.0,
        atr_volatile_percentile: float = 0.85,
        regime_lookback: int = 50,
        atr_lookback: int = 100,
        require_btc_trend: bool = True,
        min_confidence: float = 0.5,
    ):
        """
        Initialize regime filter.

        Args:
            adx_trending_threshold: ADX above this = trending (default: 25)
            adx_ranging_threshold: ADX below this = ranging (default: 18)
            adx_strong_threshold: ADX above this = strong trend (default: 30)
            atr_volatile_percentile: ATR percentile above this = volatile (default: 0.85)
            regime_lookback: Bars for ADX average calculation (default: 50)
            atr_lookback: Bars for ATR percentile calculation (default: 100)
            require_btc_trend: If True, require BTC to be trending for alts (default: True)
            min_confidence: Minimum confidence to allow trading (default: 0.5)
        """
        self.adx_trending_threshold = adx_trending_threshold
        self.adx_ranging_threshold = adx_ranging_threshold
        self.adx_strong_threshold = adx_strong_threshold
        self.atr_volatile_percentile = atr_volatile_percentile
        self.regime_lookback = regime_lookback
        self.atr_lookback = atr_lookback
        self.require_btc_trend = require_btc_trend
        self.min_confidence = min_confidence

        # Cache for BTC regime (updated once per bar)
        self._btc_regime_cache: Dict[str, RegimeResult] = {}

    def detect_regime(
        self,
        df: pd.DataFrame,
        index: int = -2,
        symbol: str = None,
        btc_df: Optional[pd.DataFrame] = None,
    ) -> RegimeResult:
        """
        Detect market regime for given data.

        Args:
            df: DataFrame with OHLCV data and indicators (must have 'adx' column)
            index: Bar index to check (default: -2, previous closed bar)
            symbol: Symbol name (for BTC leader check)
            btc_df: BTC DataFrame for leader regime check (optional)

        Returns:
            RegimeResult with regime type, tradability, and confidence
        """
        if df is None or len(df) < self.regime_lookback:
            return RegimeResult(
                regime=RegimeType.RANGING,
                should_trade=False,
                confidence=0.0,
                adx_current=0.0,
                adx_avg=0.0,
                atr_percentile=0.0,
                details="Insufficient data for regime detection",
            )

        # Convert negative index to positive
        abs_index = index if index >= 0 else len(df) + index

        # ============== ADX-BASED REGIME ==============
        # Get current ADX
        if "adx" not in df.columns:
            _logger.warning("ADX column missing, cannot detect regime")
            return RegimeResult(
                regime=RegimeType.RANGING,
                should_trade=False,
                confidence=0.0,
                adx_current=0.0,
                adx_avg=0.0,
                atr_percentile=0.0,
                details="ADX column missing",
            )

        adx_current = float(df["adx"].iloc[abs_index])
        if pd.isna(adx_current):
            adx_current = 0.0

        # Calculate ADX average over lookback (EX-ANTE: exclude current bar)
        regime_start = max(0, abs_index - self.regime_lookback)
        adx_window = df["adx"].iloc[regime_start:abs_index]  # Exclude current
        adx_avg = float(adx_window.mean()) if len(adx_window) > 0 else adx_current

        # ============== ATR VOLATILITY PERCENTILE ==============
        atr_percentile = self._calculate_atr_percentile(df, abs_index)

        # ============== REGIME CLASSIFICATION ==============
        regime, confidence = self._classify_regime(adx_current, adx_avg, atr_percentile)

        # ============== BTC LEADER CHECK ==============
        btc_regime = None
        btc_aligned = True

        if self.require_btc_trend and symbol and symbol.upper() != "BTCUSDT":
            if btc_df is not None:
                btc_result = self.detect_regime(btc_df, index, symbol="BTCUSDT")
                btc_regime = btc_result.regime
                btc_aligned = btc_result.regime in [
                    RegimeType.TRENDING_STRONG,
                    RegimeType.TRENDING
                ]
                if not btc_aligned:
                    confidence *= 0.5  # Reduce confidence if BTC not trending

        # ============== FINAL DECISION ==============
        should_trade = (
            regime in [RegimeType.TRENDING_STRONG, RegimeType.TRENDING] and
            confidence >= self.min_confidence and
            (not self.require_btc_trend or btc_aligned or symbol.upper() == "BTCUSDT")
        )

        details = self._build_details(regime, adx_current, adx_avg, atr_percentile, btc_aligned)

        return RegimeResult(
            regime=regime,
            should_trade=should_trade,
            confidence=confidence,
            adx_current=adx_current,
            adx_avg=adx_avg,
            atr_percentile=atr_percentile,
            details=details,
            btc_regime=btc_regime,
            btc_aligned=btc_aligned,
        )

    def _calculate_atr_percentile(self, df: pd.DataFrame, abs_index: int) -> float:
        """Calculate ATR percentile (where current ATR ranks vs history)."""
        if "atr" not in df.columns:
            return 0.5  # Default to middle if no ATR

        atr_start = max(0, abs_index - self.atr_lookback)
        atr_window = df["atr"].iloc[atr_start:abs_index]  # Ex-ante: exclude current

        if len(atr_window) < 20:
            return 0.5

        current_atr = float(df["atr"].iloc[abs_index])
        if pd.isna(current_atr):
            return 0.5

        # Calculate percentile rank
        atr_min = atr_window.min()
        atr_max = atr_window.max()

        if atr_max <= atr_min:
            return 0.5

        percentile = (current_atr - atr_min) / (atr_max - atr_min)
        return float(np.clip(percentile, 0.0, 1.0))

    def _classify_regime(
        self,
        adx_current: float,
        adx_avg: float,
        atr_percentile: float
    ) -> Tuple[RegimeType, float]:
        """
        Classify regime and calculate confidence.

        Returns:
            (RegimeType, confidence)
        """
        # Check for volatile regime first (overrides ADX)
        if atr_percentile > self.atr_volatile_percentile:
            # High volatility - check if it's trending volatile or choppy volatile
            if adx_current > self.adx_trending_threshold:
                # Trending + Volatile = Strong trend, trade with caution
                return RegimeType.TRENDING_STRONG, 0.7
            else:
                # Not trending + Volatile = Choppy, avoid
                return RegimeType.VOLATILE, 0.3

        # ADX-based classification
        if adx_avg >= self.adx_strong_threshold:
            # Strong trend - highest confidence
            confidence = min(1.0, 0.7 + (adx_avg - self.adx_strong_threshold) / 30)
            return RegimeType.TRENDING_STRONG, confidence

        elif adx_avg >= self.adx_trending_threshold:
            # Moderate trend - good confidence
            ratio = (adx_avg - self.adx_trending_threshold) / (self.adx_strong_threshold - self.adx_trending_threshold)
            confidence = 0.5 + ratio * 0.2
            return RegimeType.TRENDING, confidence

        elif adx_avg >= self.adx_ranging_threshold:
            # Transitional - low confidence
            ratio = (adx_avg - self.adx_ranging_threshold) / (self.adx_trending_threshold - self.adx_ranging_threshold)
            confidence = 0.3 + ratio * 0.2
            return RegimeType.TRANSITIONAL, confidence

        else:
            # Ranging - avoid trading
            confidence = max(0.1, adx_avg / self.adx_ranging_threshold * 0.3)
            return RegimeType.RANGING, confidence

    def _build_details(
        self,
        regime: RegimeType,
        adx_current: float,
        adx_avg: float,
        atr_percentile: float,
        btc_aligned: bool
    ) -> str:
        """Build human-readable details string."""
        parts = [
            f"Regime={regime.value}",
            f"ADX={adx_current:.1f}",
            f"ADX_avg={adx_avg:.1f}",
            f"ATR_pct={atr_percentile:.2f}",
        ]
        if not btc_aligned:
            parts.append("BTC_NOT_ALIGNED")
        return " | ".join(parts)

    def get_regime_multiplier(self, regime_result: RegimeResult) -> float:
        """
        Get position size multiplier based on regime.

        Use this to reduce position size in uncertain regimes.

        Returns:
            Multiplier (0.0 to 1.0)
        """
        multipliers = {
            RegimeType.TRENDING_STRONG: 1.0,
            RegimeType.TRENDING: 0.8,
            RegimeType.TRANSITIONAL: 0.5,
            RegimeType.VOLATILE: 0.3,
            RegimeType.RANGING: 0.0,
        }
        base_multiplier = multipliers.get(regime_result.regime, 0.0)

        # Adjust by confidence
        return base_multiplier * regime_result.confidence


# ============== CONVENIENCE FUNCTIONS ==============

def check_regime_for_trade(
    df: pd.DataFrame,
    index: int = -2,
    symbol: str = None,
    btc_df: Optional[pd.DataFrame] = None,
    adx_threshold: float = 20.0,
    require_btc_trend: bool = True,
) -> Tuple[bool, str, float]:
    """
    Quick check if regime allows trading.

    Args:
        df: DataFrame with indicators
        index: Bar index to check
        symbol: Symbol name
        btc_df: BTC data for leader check
        adx_threshold: Minimum ADX average for trading
        require_btc_trend: Require BTC to be trending for alts

    Returns:
        (should_trade, reason, confidence)
    """
    regime_filter = RegimeFilter(
        adx_trending_threshold=adx_threshold,
        require_btc_trend=require_btc_trend,
    )

    result = regime_filter.detect_regime(df, index, symbol, btc_df)

    if result.should_trade:
        return True, f"REGIME_OK: {result.details}", result.confidence
    else:
        return False, f"REGIME_SKIP: {result.details}", result.confidence


def get_btc_regime(
    btc_df: pd.DataFrame,
    index: int = -2,
    cache_key: str = None,
) -> RegimeResult:
    """
    Get BTC's current regime (for use as market leader signal).

    Args:
        btc_df: BTC DataFrame with indicators
        index: Bar index
        cache_key: Optional cache key (e.g., timestamp)

    Returns:
        RegimeResult for BTC
    """
    regime_filter = RegimeFilter(require_btc_trend=False)
    return regime_filter.detect_regime(btc_df, index, symbol="BTCUSDT")


# ============== REGIME STATISTICS ==============

def analyze_regime_distribution(
    df: pd.DataFrame,
    lookback_bars: int = None,
) -> Dict[str, any]:
    """
    Analyze regime distribution over a period.

    Useful for understanding what % of time market is in each regime.

    Args:
        df: DataFrame with indicators
        lookback_bars: Number of bars to analyze (default: all)

    Returns:
        Dictionary with regime statistics
    """
    if df is None or df.empty:
        return {"error": "No data"}

    if lookback_bars:
        df = df.tail(lookback_bars)

    regime_filter = RegimeFilter()
    regimes = []

    for i in range(50, len(df)):  # Start after warmup period
        result = regime_filter.detect_regime(df, i)
        regimes.append(result.regime.value)

    if not regimes:
        return {"error": "Not enough data"}

    # Count distribution
    from collections import Counter
    counts = Counter(regimes)
    total = len(regimes)

    distribution = {
        regime: {
            "count": count,
            "percentage": count / total * 100,
        }
        for regime, count in counts.items()
    }

    # Calculate tradable percentage
    tradable_regimes = ["TRENDING_STRONG", "TRENDING"]
    tradable_count = sum(counts.get(r, 0) for r in tradable_regimes)
    tradable_pct = tradable_count / total * 100 if total > 0 else 0

    return {
        "total_bars": total,
        "distribution": distribution,
        "tradable_percentage": tradable_pct,
        "dominant_regime": counts.most_common(1)[0][0] if counts else "UNKNOWN",
    }
