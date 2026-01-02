"""
Technical indicators module.

Provides indicator calculation functions for trading strategies:
- RSI, ADX
- PBEMA Cloud (EMA 150, 200)
- Keltner Channels
- AlphaTrend
- SSL Baseline (HMA)
"""

import pandas as pd
import numpy as np
from typing import Optional

from .logging_config import get_logger
from .config import ALPHATREND_CONFIG, get_tf_threshold, get_at_validation_thresholds

# Module logger
_logger = get_logger(__name__)

# Lazy import for pandas_ta (heavy library ~5-10s import time)
_ta = None


def _get_ta():
    """Lazy load pandas_ta when first needed."""
    global _ta
    if _ta is None:
        import pandas_ta as ta_module
        _ta = ta_module
    return _ta


def calculate_alphatrend(
    df: pd.DataFrame,
    coeff: float = 1.0,
    ap: int = 14,
    flat_lookback: int = 5,
    flat_threshold: float = None,
    timeframe: str = None
) -> pd.DataFrame:
    """
    Calculate AlphaTrend indicator with DUAL lines (buyers vs sellers).

    AlphaTrend is a trend-following indicator that combines ATR and MFI/RSI.
    It provides dynamic support/resistance levels and buyer/seller dominance signals.

    The indicator calculates (exactly matching TradingView Pine Script):
    - alphatrend: Single line that switches between tracking support (upT) and
      resistance (downT) based on momentum.
    - alphatrend_2: alphatrend shifted by 2 bars (for crossover detection)
    - at_buyers: Same as alphatrend (BLUE line in TradingView)
    - at_sellers: Same as alphatrend_2 (RED line in TradingView)

    TradingView Pine Script reference:
        k1 = plot(AlphaTrend, color=#0022FC)      // BLUE = current
        k2 = plot(AlphaTrend[2], color=#FC0400)   // RED = 2 bars ago

    Trade signals:
    - LONG: at_buyers > at_sellers (blue above red = line rising)
    - SHORT: at_sellers > at_buyers (red above blue = line falling)
    - NO TRADE: at_is_flat = True (no significant movement)

    Args:
        df: DataFrame with OHLCV data
        coeff: ATR multiplier (default: 1.0)
        ap: ATR/MFI period (default: 14)
        flat_lookback: Number of candles to check for flat detection (default: 5)
        flat_threshold: Minimum change ratio to consider non-flat.
            If None, uses TF-adaptive threshold (recommended).
            For backward compatibility, explicit value overrides TF-adaptive.
        timeframe: Timeframe string (e.g., "15m", "1h") for TF-adaptive thresholds.
            If None, uses default 15m threshold (0.002).

    Returns:
        DataFrame with the following columns added:
        - at_buyers: Buyer strength line (blue)
        - at_sellers: Seller strength line (red)
        - at_buyers_dominant: True if buyers > sellers
        - at_sellers_dominant: True if sellers > buyers
        - at_is_flat: True if both lines are flat (no flow)
        - alphatrend: Backward compatible single line (dominant line)
        - alphatrend_2: alphatrend shifted by 2 (backward compat)
    """
    # TF-adaptive flat_threshold
    if flat_threshold is None:
        if timeframe:
            flat_threshold = get_tf_threshold("flat_threshold", timeframe)
        else:
            flat_threshold = 0.002  # Default 15m baseline
    try:
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)

        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype(float)

        ta = _get_ta()
        n = len(df)

        # Calculate True Range and ATR
        # FIX: AlphaTrend Pine Script uses SMA for ATR, not RMA!
        # Pine code: ATR = ta.sma(ta.tr, AP)
        df['_at_tr'] = ta.true_range(df['high'], df['low'], df['close'])
        df['_at_atr'] = ta.sma(df['_at_tr'], length=ap)  # SMA to match Pine Script

        # Use MFI if volume available, otherwise use RSI
        # REVERTED: MFI produces more signals and user confirmed manual trading uses MFI
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['_at_momentum'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=ap)
        else:
            df['_at_momentum'] = ta.rsi(df['close'], length=ap)

        # Calculate upper and lower trend levels
        df['_at_upT'] = df['low'] - df['_at_atr'] * coeff
        df['_at_downT'] = df['high'] + df['_at_atr'] * coeff

        # Pre-extract arrays for performance
        momentum = df['_at_momentum'].fillna(50).values
        upT = df['_at_upT'].ffill().bfill().values
        downT = df['_at_downT'].ffill().bfill().values
        close_vals = df['close'].values

        # ================================================================
        # SINGLE AlphaTrend line (matches TradingView Pine Script exactly)
        # TradingView code:
        #   AlphaTrend := (RSI >= 50) ? max(upT, nz(AlphaTrend[1])) : min(downT, nz(AlphaTrend[1]))
        # ================================================================
        alphatrend = np.zeros(n)
        alphatrend[0] = close_vals[0]

        for i in range(1, n):
            if momentum[i] >= 50:
                # Bullish momentum: track support, ratchet UP
                alphatrend[i] = max(alphatrend[i - 1], upT[i])
            else:
                # Bearish momentum: track resistance, ratchet DOWN
                alphatrend[i] = min(alphatrend[i - 1], downT[i])

        df['alphatrend'] = alphatrend
        df['alphatrend_2'] = df['alphatrend'].shift(2)

        # ================================================================
        # at_buyers / at_sellers - TradingView UYUMLU
        # TradingView'da:
        #   k1 = plot(AlphaTrend, color=#0022FC)      // MAVİ = AlphaTrend (mevcut)
        #   k2 = plot(AlphaTrend[2], color=#FC0400)   // KIRMIZI = AlphaTrend[2] (2 bar önceki)
        #
        # Şimdi log'larda gördüğün at_buyers/at_sellers değerleri
        # TradingView'daki mavi/kırmızı çizgilerle BİREBİR eşleşecek!
        # ================================================================
        df['at_buyers'] = df['alphatrend']      # Mavi çizgi (AlphaTrend)
        df['at_sellers'] = df['alphatrend_2']   # Kırmızı çizgi (AlphaTrend[2])

        # ================================================================
        # DOMINANCE based on LINE POSITION (TradingView standard)
        # - at_buyers > at_sellers = Mavi üstte = Çizgi YÜKSELİYOR = BUYERS dominant
        # - at_sellers > at_buyers = Kırmızı üstte = Çizgi DÜŞÜYOR = SELLERS dominant
        #
        # Pine Script equality handling:
        #   color1 = AlphaTrend > AlphaTrend[2] ? #00E60F :
        #            AlphaTrend < AlphaTrend[2] ? #80000B :
        #            AlphaTrend[1] > AlphaTrend[3] ? #00E60F : #80000B
        # When AlphaTrend == AlphaTrend[2], check AlphaTrend[1] vs AlphaTrend[3]
        # ================================================================
        df['at_buyers_dominant'] = df['at_buyers'] > df['at_sellers']
        df['at_sellers_dominant'] = df['at_sellers'] > df['at_buyers']

        # Handle equality case (Pine Script: AlphaTrend[1] > AlphaTrend[3])
        # When current and 2-bar-ago are equal, check 1-bar-ago vs 3-bar-ago
        # FIX: Use np.isclose for floating point comparison to avoid precision issues
        equal_mask_arr = np.isclose(df['at_buyers'].values, df['at_sellers'].values, rtol=1e-9, atol=1e-12)
        equal_mask = pd.Series(equal_mask_arr, index=df.index)

        if equal_mask.any():
            _logger.debug("AlphaTrend equality detected on %d bars", equal_mask.sum())

            # DÜZELTME: shift(3) için minimum 4 bar gerekli
            # İlk 3 bar için equality check'i atla (veri yetersiz)
            min_required_bars = 4
            valid_idx = pd.Series(False, index=df.index)
            if len(df) >= min_required_bars:
                valid_idx.iloc[min_required_bars - 1:] = True

            equal_and_valid = equal_mask & valid_idx

            if equal_and_valid.any():
                prev_comparison = df['alphatrend'].shift(1) > df['alphatrend'].shift(3)
                df.loc[equal_and_valid, 'at_buyers_dominant'] = prev_comparison[equal_and_valid]
                df.loc[equal_and_valid, 'at_sellers_dominant'] = ~prev_comparison[equal_and_valid]

            # İlk 3 bar için equality durumunda default değer ata (neutral = False, False)
            early_equal = equal_mask & ~valid_idx
            if early_equal.any():
                df.loc[early_equal, 'at_buyers_dominant'] = False
                df.loc[early_equal, 'at_sellers_dominant'] = False

        # Flat/no-flow detection
        # If alphatrend line hasn't moved significantly, market is flat
        df['_at_change'] = df['alphatrend'].pct_change(flat_lookback).abs()

        df['at_is_flat'] = df['_at_change'] < flat_threshold
        # Fill NaN values (first few rows) as False
        df['at_is_flat'] = df['at_is_flat'].fillna(False)

        # Cleanup temporary columns
        cleanup_cols = ['_at_tr', '_at_atr', '_at_momentum', '_at_upT', '_at_downT',
                        '_at_change']
        df.drop(columns=[c for c in cleanup_cols if c in df.columns], inplace=True, errors='ignore')

        return df

    except Exception as e:
        # Fallback: set basic columns to prevent errors
        _logger.error("AlphaTrend calculation error: %s", e)
        df['alphatrend'] = df['close']
        df['alphatrend_2'] = df['close'].shift(2)
        # at_buyers = blue line (alphatrend), at_sellers = red line (alphatrend_2)
        df['at_buyers'] = df['alphatrend']
        df['at_sellers'] = df['alphatrend_2']
        df['at_buyers_dominant'] = df['at_buyers'] > df['at_sellers']
        df['at_sellers_dominant'] = df['at_sellers'] > df['at_buyers']
        # Handle equality case in fallback too
        # FIX: Use np.isclose for floating point comparison to avoid precision issues
        equal_mask_arr = np.isclose(df['at_buyers'].values, df['at_sellers'].values, rtol=1e-9, atol=1e-12)
        equal_mask = pd.Series(equal_mask_arr, index=df.index)
        if equal_mask.any():
            min_required_bars = 4
            valid_idx = pd.Series(False, index=df.index)
            if len(df) >= min_required_bars:
                valid_idx.iloc[min_required_bars - 1:] = True

            equal_and_valid = equal_mask & valid_idx
            if equal_and_valid.any():
                prev_comparison = df['alphatrend'].shift(1) > df['alphatrend'].shift(3)
                df.loc[equal_and_valid, 'at_buyers_dominant'] = prev_comparison[equal_and_valid]
                df.loc[equal_and_valid, 'at_sellers_dominant'] = ~prev_comparison[equal_and_valid]

            early_equal = equal_mask & ~valid_idx
            if early_equal.any():
                df.loc[early_equal, 'at_buyers_dominant'] = False
                df.loc[early_equal, 'at_sellers_dominant'] = False
        df['at_is_flat'] = False
        return df


def add_at_validation_columns(
    df: pd.DataFrame,
    timeframe: str = "15m"
) -> pd.DataFrame:
    """
    Add AlphaTrend validation columns for fake signal filtering (SIMPLIFIED v2).

    SIMPLIFIED APPROACH (based on TradingView chart analysis):
    AlphaTrend moves in STEPS, not continuously. It holds flat then jumps.

    The correct interpretation:
    - "FLAT" = blue and red lines OVERLAPPING (no clear winner)
    - "VALID" = clear separation between lines (one is dominant)

    Previous 3-layer validation was WRONG because:
    - Cross timing check was too restrictive (AT steps infrequently)
    - Momentum check required "rising" but AT naturally holds flat then steps

    NEW SIMPLIFIED VALIDATION:
    1. Dominance: at_buyers > at_sellers (blue above red)
    2. Not flat: Lines have meaningful separation (not overlapping)

    That's it! The dominance already exists from calculate_alphatrend().
    We just add a separation-based "flat" check.

    Args:
        df: DataFrame with AlphaTrend columns (at_buyers, at_sellers, alphatrend)
        timeframe: Timeframe string for TF-adaptive thresholds

    Returns:
        DataFrame with validation columns added:
        - at_line_separation: Percentage gap between lines
        - at_not_flat: True if lines have meaningful separation
        - at_valid_long: Buyers dominant AND not flat
        - at_valid_short: Sellers dominant AND not flat
        - at_cross_up/down: Cross detection (kept for analysis)
        - at_bars_since_*_cross: Bars since cross (kept for analysis)
    """
    if df is None or df.empty:
        return df

    # Get TF-specific thresholds
    thresholds = get_at_validation_thresholds(timeframe)
    min_line_separation = thresholds.get("min_line_separation", 0.001)  # 0.1% default

    n = len(df)

    # Ensure required columns exist
    if "at_buyers" not in df.columns or "at_sellers" not in df.columns:
        _logger.warning("AT validation: Missing at_buyers/at_sellers columns")
        df["at_valid_long"] = False
        df["at_valid_short"] = False
        return df

    at_buyers = df["at_buyers"].values
    at_sellers = df["at_sellers"].values
    close_prices = df["close"].values

    # ================================================================
    # CROSS DETECTION (kept for analysis/debugging, NOT used in validation)
    # ================================================================
    at_cross_up = np.zeros(n, dtype=bool)
    at_cross_down = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(at_buyers[i]) or np.isnan(at_sellers[i]):
            continue
        if np.isnan(at_buyers[i - 1]) or np.isnan(at_sellers[i - 1]):
            continue

        # Bullish cross: was below/equal, now above
        if at_buyers[i - 1] <= at_sellers[i - 1] and at_buyers[i] > at_sellers[i]:
            at_cross_up[i] = True

        # Bearish cross: was above/equal, now below
        if at_buyers[i - 1] >= at_sellers[i - 1] and at_buyers[i] < at_sellers[i]:
            at_cross_down[i] = True

    df["at_cross_up"] = at_cross_up
    df["at_cross_down"] = at_cross_down

    # Bars since cross (kept for analysis)
    bars_since_bullish = np.full(n, 999, dtype=int)
    bars_since_bearish = np.full(n, 999, dtype=int)

    last_bullish_cross = -999
    last_bearish_cross = -999

    for i in range(n):
        if at_cross_up[i]:
            last_bullish_cross = i
        if at_cross_down[i]:
            last_bearish_cross = i

        if last_bullish_cross >= 0:
            bars_since_bullish[i] = i - last_bullish_cross
        if last_bearish_cross >= 0:
            bars_since_bearish[i] = i - last_bearish_cross

    df["at_bars_since_bullish_cross"] = bars_since_bullish
    df["at_bars_since_bearish_cross"] = bars_since_bearish

    # ================================================================
    # LINE SEPARATION (key metric for "flat" detection)
    # Gap between blue and red lines as percentage of price
    # ================================================================
    line_separation = np.zeros(n)
    for i in range(n):
        if close_prices[i] > 0 and not np.isnan(at_buyers[i]) and not np.isnan(at_sellers[i]):
            gap = abs(at_buyers[i] - at_sellers[i])
            line_separation[i] = gap / close_prices[i]

    df["at_line_separation"] = line_separation

    # ================================================================
    # SIMPLIFIED VALIDATION (v2)
    # "FLAT" = lines overlapping (separation < threshold)
    # "VALID" = dominance + separation
    # ================================================================

    # Not flat = lines have meaningful separation
    at_not_flat = line_separation >= min_line_separation
    df["at_not_flat"] = at_not_flat

    # Use existing dominance columns from calculate_alphatrend()
    # at_buyers_dominant = at_buyers > at_sellers (blue above red)
    # at_sellers_dominant = at_sellers > at_buyers (red above blue)

    at_buyers_dominant = df["at_buyers_dominant"].values if "at_buyers_dominant" in df.columns else (at_buyers > at_sellers)
    at_sellers_dominant = df["at_sellers_dominant"].values if "at_sellers_dominant" in df.columns else (at_sellers > at_buyers)

    # SIMPLE VALIDATION:
    # LONG valid = blue dominant AND not flat
    # SHORT valid = red dominant AND not flat
    at_valid_long = at_buyers_dominant & at_not_flat
    at_valid_short = at_sellers_dominant & at_not_flat

    df["at_valid_long"] = at_valid_long
    df["at_valid_short"] = at_valid_short

    # CRITICAL FIX: Prevent NaN propagation from early bars
    # NaN & True = NaN in pandas, which breaks signal checks
    df["at_valid_long"] = df["at_valid_long"].fillna(False)
    df["at_valid_short"] = df["at_valid_short"].fillna(False)

    # Legacy columns (set to match new logic for backward compatibility)
    df["at_momentum_bullish"] = at_valid_long  # Simplified: same as valid_long
    df["at_momentum_bearish"] = at_valid_short  # Simplified: same as valid_short

    _logger.debug(
        "AT Validation SIMPLIFIED [%s]: valid_long=%d (%.1f%%), valid_short=%d (%.1f%%), not_flat=%d (%.1f%%), total=%d",
        timeframe,
        at_valid_long.sum(),
        at_valid_long.sum() / n * 100 if n > 0 else 0,
        at_valid_short.sum(),
        at_valid_short.sum() / n * 100 if n > 0 else 0,
        at_not_flat.sum(),
        at_not_flat.sum() / n * 100 if n > 0 else 0,
        n
    )

    return df


def calculate_at_regime(df: pd.DataFrame, index: int, lookback: int = 20) -> str:
    """
    Calculate AT regime over lookback period ENDING at the given index.

    THREE-TIER AT ARCHITECTURE - TIER 1
    ------------------------------------
    Instead of checking AT per-bar (which causes lag issues),
    determine the OVERALL REGIME over recent history.

    This provides a more stable signal that:
    - Doesn't flip-flop on every bar
    - Captures the dominant market direction
    - Allows SSL to handle timing while AT handles regime

    CRITICAL: Uses bars [index-lookback+1 : index+1] to avoid look-ahead bias.

    Args:
        df: DataFrame with at_buyers_dominant and at_sellers_dominant columns
        index: The bar index to calculate regime FOR (looks back from here)
        lookback: Number of bars to analyze (default: 20)

    Returns:
        str: One of "bullish_regime", "bearish_regime", "neutral_regime"

    Example:
        >>> regime = calculate_at_regime(df, index=-2, lookback=20)
        >>> if regime == "bearish_regime":
        ...     block_longs = True  # Don't fight the regime
    """
    if df is None or df.empty:
        return "neutral_regime"

    if "at_buyers_dominant" not in df.columns:
        return "neutral_regime"

    # Convert negative index to positive
    if index < 0:
        index = len(df) + index

    # Check if we have enough data
    if index < lookback - 1:
        return "neutral_regime"

    # Get AT dominance for lookback bars ENDING at index (no look-ahead!)
    start_idx = index - lookback + 1
    end_idx = index + 1  # +1 because iloc is exclusive on end

    recent_buyers = df["at_buyers_dominant"].iloc[start_idx:end_idx]
    recent_sellers = df["at_sellers_dominant"].iloc[start_idx:end_idx]

    # Count bars where each side is dominant
    buyers_bars = recent_buyers.sum()
    sellers_bars = recent_sellers.sum()

    # Calculate dominance ratio
    regime_threshold = 0.6  # 60% dominance required

    buyers_ratio = buyers_bars / lookback
    sellers_ratio = sellers_bars / lookback

    if buyers_ratio >= regime_threshold:
        return "bullish_regime"
    elif sellers_ratio >= regime_threshold:
        return "bearish_regime"
    else:
        return "neutral_regime"


def calculate_at_score(df: pd.DataFrame, index: int, direction: str) -> float:
    """
    Calculate AT contribution as a SCORE (not binary filter).

    THREE-TIER AT ARCHITECTURE - TIER 3
    ------------------------------------
    Instead of AT blocking signals (binary), AT contributes to signal quality score.
    This allows:
    - Good signals to proceed even if AT is neutral
    - AT alignment to BOOST signal confidence
    - AT misalignment to REDUCE (not block) signal confidence

    Args:
        df: DataFrame with AT columns
        index: Bar index to check
        direction: "long" or "short"

    Returns:
        float: Score contribution
            - +2.0: Strong alignment (AT confirms direction)
            - +0.5: Neutral (AT flat or inconclusive)
            - -1.0: Misalignment (AT opposing direction)

    Example:
        >>> at_score = calculate_at_score(df, -2, "long")
        >>> total_score = base_score + at_score
        >>> if total_score >= threshold:
        ...     take_trade = True
    """
    if df is None or df.empty:
        return 0.0

    try:
        at_buyers = df["at_buyers_dominant"].iloc[index]
        at_sellers = df["at_sellers_dominant"].iloc[index]
        at_flat = df.get("at_is_flat", pd.Series([False] * len(df))).iloc[index]
    except (IndexError, KeyError):
        return 0.0

    if direction == "long":
        if at_buyers:
            return 2.0   # Strong alignment - AT confirms bullish
        elif at_flat:
            return 0.5   # Neutral - no strong opinion
        elif at_sellers:
            return -1.0  # Misalignment - AT says bearish
        else:
            return 0.0   # Unknown state

    elif direction == "short":
        if at_sellers:
            return 2.0   # Strong alignment - AT confirms bearish
        elif at_flat:
            return 0.5   # Neutral - no strong opinion
        elif at_buyers:
            return -1.0  # Misalignment - AT says bullish
        else:
            return 0.0   # Unknown state

    return 0.0


def calculate_indicators(df: pd.DataFrame, timeframe: str = None) -> pd.DataFrame:
    """
    Calculate all technical indicators for trading strategies.

    Indicators calculated:
    - RSI(14): Relative Strength Index
    - ADX(14): Average Directional Index
    - PBEMA Cloud (EMA 200): pb_ema_top, pb_ema_bot (for Keltner Bounce)
    - PBEMA Cloud (EMA 150): pb_ema_top_150, pb_ema_bot_150 (for PBEMA Reaction)
    - Slope: Rate of change for EMA levels
    - SSL Baseline: HMA(TF-adaptive) - v1.7.1
    - Keltner Channels: baseline +/- EMA(TR) * 0.2
    - AlphaTrend: Optional trend filter

    PERFORMANCE NOTE: This function modifies the DataFrame in-place.
    If you need to preserve the original DataFrame, make a copy before calling.

    Args:
        df: DataFrame with OHLCV data (timestamp, open, high, low, close, volume)
        timeframe: Optional timeframe string (e.g., "5m", "15m", "1h") for TF-adaptive lookbacks

    Returns:
        DataFrame with indicator columns added
    """
    # Convert base columns to float
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    ta = _get_ta()

    # RSI and ADX
    df["rsi"] = ta.rsi(df["close"], length=14)
    adx_res = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx"] = adx_res["ADX_14"] if adx_res is not None and "ADX_14" in adx_res.columns else 0.0

    # EMA15 - Fast momentum indicator for Momentum TP Extension (v1.5)
    df["ema15"] = ta.ema(df["close"], length=15)

    # PBEMA Cloud (EMA 200) - for SSL Flow / Keltner Bounce strategy
    df["pb_ema_top"] = ta.ema(df["high"], length=200)
    df["pb_ema_bot"] = ta.ema(df["close"], length=200)

    # PBEMA Cloud (EMA 150) - for PBEMA Reaction strategy
    df["pb_ema_top_150"] = ta.ema(df["high"], length=150)
    df["pb_ema_bot_150"] = ta.ema(df["close"], length=150)

    # Slope (rate of change over 5 periods) - EMA 200
    # FIX: Division by zero guard using np.where
    df["slope_top"] = np.where(
        df["pb_ema_top"] != 0,
        (df["pb_ema_top"].diff(5) / df["pb_ema_top"]) * 1000,
        0.0
    )
    df["slope_bot"] = np.where(
        df["pb_ema_bot"] != 0,
        (df["pb_ema_bot"].diff(5) / df["pb_ema_bot"]) * 1000,
        0.0
    )

    # Slope - EMA 150
    df["slope_top_150"] = np.where(
        df["pb_ema_top_150"] != 0,
        (df["pb_ema_top_150"].diff(5) / df["pb_ema_top_150"]) * 1000,
        0.0
    )
    df["slope_bot_150"] = np.where(
        df["pb_ema_bot_150"] != 0,
        (df["pb_ema_bot_150"].diff(5) / df["pb_ema_bot_150"]) * 1000,
        0.0
    )

    # === TF-ADAPTIVE SSL BASELINE (v1.7.1) ===
    # Shorter lookback for higher TFs (faster response to larger moves)
    # Longer lookback for lower TFs (more noise filtering)
    TF_HMA_LOOKBACK = {
        "5m": 75,   # More smoothing for noisy 5m
        "15m": 60,  # Standard (current baseline)
        "30m": 55,  # Slightly shorter
        "1h": 45,   # Faster response for 1h
        "4h": 40,   # Even faster for 4h
    }
    hma_length = TF_HMA_LOOKBACK.get(timeframe, 60)  # Default to 60 if unknown

    # SSL Baseline (HMA with TF-adaptive length) and Keltner Channels
    df["baseline"] = ta.hma(df["close"], length=hma_length)
    tr = ta.true_range(df["high"], df["low"], df["close"])
    range_ma = ta.ema(tr, length=hma_length)
    df["keltner_upper"] = df["baseline"] + range_ma * 0.2
    df["keltner_lower"] = df["baseline"] - range_ma * 0.2

    # ATR for volatility-based regime detection
    df["atr"] = ta.sma(tr, length=14)

    # AlphaTrend (for optional trend filtering)
    # Use TF-adaptive flat_threshold when timeframe is provided
    # Otherwise fall back to ALPHATREND_CONFIG defaults
    df = calculate_alphatrend(
        df,
        coeff=ALPHATREND_CONFIG.get("coeff", 1.0),
        ap=ALPHATREND_CONFIG.get("ap", 14),
        flat_lookback=ALPHATREND_CONFIG.get("flat_lookback", 5),
        flat_threshold=None,  # Let calculate_alphatrend use TF-adaptive
        timeframe=timeframe   # Pass timeframe for TF-adaptive threshold
    )

    # Add AlphaTrend 3-layer validation columns for fake signal filtering
    # This adds: at_valid_long, at_valid_short based on:
    # 1. Recent AT cross (within X candles)
    # 2. Momentum (winning line moving in signal direction)
    # 3. Line separation (minimum gap between blue/red)
    if timeframe:
        df = add_at_validation_columns(df, timeframe=timeframe)
    else:
        df = add_at_validation_columns(df, timeframe="15m")  # Default to 15m

    return df


def get_indicator_value(df: pd.DataFrame, indicator: str, index: int = -1) -> Optional[float]:
    """
    Get indicator value at a specific index.

    Args:
        df: DataFrame with indicators
        indicator: Column name
        index: Row index (negative for from end)

    Returns:
        Indicator value or None if not available
    """
    if df is None or df.empty:
        return None

    if indicator not in df.columns:
        return None

    try:
        value = df[indicator].iloc[index]
        if pd.isna(value):
            return None
        return float(value)
    except (IndexError, KeyError, ValueError, TypeError):
        return None


def get_candle_data(df: pd.DataFrame, index: int = -1) -> Optional[dict]:
    """
    Get OHLC data for a specific candle.

    Args:
        df: DataFrame with OHLCV data
        index: Row index (negative for from end)

    Returns:
        Dict with 'open', 'high', 'low', 'close', 'timestamp' or None
    """
    if df is None or df.empty:
        return None

    try:
        row = df.iloc[index]
        return {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'timestamp': row['timestamp'] if 'timestamp' in row.index else None,
        }
    except (IndexError, KeyError, ValueError, TypeError):
        return None


def calculate_rr_ratio(
    entry: float,
    tp: float,
    sl: float,
    trade_type: str
) -> float:
    """
    Calculate Risk/Reward ratio.

    Args:
        entry: Entry price
        tp: Take profit price
        sl: Stop loss price
        trade_type: "LONG" or "SHORT"

    Returns:
        R/R ratio (reward / risk)
    """
    if trade_type == "LONG":
        reward = tp - entry
        risk = entry - sl
    else:
        reward = entry - tp
        risk = sl - entry

    if risk <= 0:
        return 0.0

    return reward / risk


def check_wick_rejection(
    candle: dict,
    min_wick_ratio: float = 0.15
) -> dict:
    """
    Check if a candle has quality wick rejection.

    A quality rejection shows the price was pushed back from an extreme level.

    Args:
        candle: Dict with 'open', 'high', 'low', 'close'
        min_wick_ratio: Minimum wick ratio for quality rejection (0-1)

    Returns:
        Dict with 'upper_wick_ratio', 'lower_wick_ratio',
        'long_rejection', 'short_rejection'
    """
    open_ = candle['open']
    high = candle['high']
    low = candle['low']
    close = candle['close']

    candle_range = high - low

    if candle_range > 0:
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range
    else:
        upper_wick_ratio = 0.0
        lower_wick_ratio = 0.0

    return {
        'upper_wick_ratio': upper_wick_ratio,
        'lower_wick_ratio': lower_wick_ratio,
        'long_rejection': lower_wick_ratio >= min_wick_ratio,
        'short_rejection': upper_wick_ratio >= min_wick_ratio,
    }


# ==========================================
# MARKET REGIME DETECTION
# ==========================================

def detect_regime(
    df: pd.DataFrame,
    adx_trending_threshold: float = 25.0,
    adx_ranging_threshold: float = 20.0,
    atr_volatile_percentile: float = 0.80,
    atr_lookback: int = 100,
    index: int = -1
) -> dict:
    """
    Detect market regime based on ADX and ATR.

    3 Regimes:
    - TRENDING: ADX > 25, clear directional movement
    - VOLATILE: ATR percentile > 80%, choppy/explosive moves
    - RANGING: ADX < 20, low volatility consolidation

    Args:
        df: DataFrame with OHLCV data (must have 'adx' column or will calculate)
        adx_trending_threshold: ADX above this = trending (default: 25)
        adx_ranging_threshold: ADX below this = ranging (default: 20)
        atr_volatile_percentile: ATR percentile above this = volatile (default: 0.80)
        atr_lookback: Lookback period for ATR percentile (default: 100)
        index: Row index to check (default: -1 = last row)

    Returns:
        Dict with:
        - regime: "TRENDING", "VOLATILE", or "RANGING"
        - adx: Current ADX value
        - atr_percentile: Current ATR percentile (0-1)
        - details: Human-readable description
    """
    if df is None or len(df) < atr_lookback:
        return {
            "regime": "UNKNOWN",
            "adx": None,
            "atr_percentile": None,
            "details": "Insufficient data"
        }

    ta = _get_ta()

    # Get or calculate ADX
    if "adx" in df.columns:
        adx_value = df["adx"].iloc[index]
    else:
        adx_res = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_res is not None and "ADX_14" in adx_res.columns:
            adx_value = adx_res["ADX_14"].iloc[index]
        else:
            adx_value = 0.0

    # Calculate ATR and its percentile
    tr = ta.true_range(df["high"], df["low"], df["close"])
    atr = ta.sma(tr, length=14)

    # ATR percentile over lookback period
    atr_pct_series = atr.rolling(atr_lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )
    atr_percentile = atr_pct_series.iloc[index] if not pd.isna(atr_pct_series.iloc[index]) else 0.5

    # Classify regime
    if adx_value > adx_trending_threshold:
        regime = "TRENDING"
        details = f"Strong trend (ADX={adx_value:.1f})"
    elif atr_percentile > atr_volatile_percentile:
        regime = "VOLATILE"
        details = f"High volatility (ATR pct={atr_percentile:.2f})"
    elif adx_value < adx_ranging_threshold:
        regime = "RANGING"
        details = f"Consolidation (ADX={adx_value:.1f})"
    else:
        regime = "TRANSITIONAL"
        details = f"Between regimes (ADX={adx_value:.1f}, ATR pct={atr_percentile:.2f})"

    return {
        "regime": regime,
        "adx": float(adx_value) if not pd.isna(adx_value) else None,
        "atr_percentile": float(atr_percentile) if not pd.isna(atr_percentile) else None,
        "details": details
    }


def add_regime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime classification column to DataFrame.

    Adds 'regime' column with values: TRENDING, VOLATILE, RANGING, TRANSITIONAL

    Args:
        df: DataFrame with OHLCV and indicator data

    Returns:
        DataFrame with 'regime' column added
    """
    if df is None or df.empty:
        return df

    ta = _get_ta()

    # Ensure ADX exists
    if "adx" not in df.columns:
        adx_res = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx_res["ADX_14"] if adx_res is not None and "ADX_14" in adx_res.columns else 0.0

    # Calculate ATR
    tr = ta.true_range(df["high"], df["low"], df["close"])
    atr = ta.sma(tr, length=14)

    # ATR percentile (rolling rank)
    atr_percentile = atr.rolling(100, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )

    # Vectorized regime classification
    conditions = [
        df["adx"] > 25,  # TRENDING
        atr_percentile > 0.80,  # VOLATILE
        df["adx"] < 20,  # RANGING
    ]
    choices = ["TRENDING", "VOLATILE", "RANGING"]

    df["regime"] = np.select(conditions, choices, default="TRANSITIONAL")

    return df


# ==========================================
# VOLATILITY REGIME DETECTION (Sinclair's 3-Tier System)
# ==========================================
# Expert Panel Recommendation: Use volatility-adaptive trading rules
# - LOW_VOL: Conservative mode (small positions, strict filters)
# - NORMAL_VOL: Standard mode (baseline parameters)
# - HIGH_VOL: Aggressive mode (larger positions, allow AT lag)


def classify_volatility_regime(
    df: pd.DataFrame,
    index: int = -1,
    lookback: int = 50,
    low_vol_threshold: float = 40.0,
    high_vol_threshold: float = 75.0,
) -> dict:
    """
    Classify volatility regime using ATR percentile (Sinclair's 3-tier system).

    This addresses the AlphaTrend "lag" problem identified in expert panel analysis:
    - In LOW_VOL: SSL whipsaws, AT lag is actually helpful (quality filter)
    - In HIGH_VOL: Strong trends, AT lag hurts (allow grace period)

    The ATR percentile approach is more stable than raw ATR because it:
    - Normalizes across different price levels
    - Adapts to symbol-specific volatility profiles
    - Provides consistent classification across timeframes

    Args:
        df: DataFrame with ATR column (or will calculate)
        index: Bar index to classify (default: -1 = last bar)
        lookback: Bars to calculate percentile over (default: 50)
        low_vol_threshold: ATR percentile below this = LOW_VOL (default: 40)
        high_vol_threshold: ATR percentile above this = HIGH_VOL (default: 75)

    Returns:
        Dict with:
        - regime: "LOW_VOL", "NORMAL_VOL", or "HIGH_VOL"
        - atr_percentile: Current ATR percentile (0-100)
        - position_multiplier: Suggested position size multiplier
        - allow_at_grace: Whether to allow AT lag grace period
        - details: Human-readable description

    Example:
        >>> result = classify_volatility_regime(df, index=-2)
        >>> if result["regime"] == "HIGH_VOL":
        ...     allow_ssl_flip_grace = True  # Let AT lag, strong trend
    """
    if df is None or len(df) < lookback:
        return {
            "regime": "NORMAL_VOL",
            "atr_percentile": 50.0,
            "position_multiplier": 1.0,
            "allow_at_grace": False,
            "details": "Insufficient data, defaulting to NORMAL_VOL"
        }

    ta = _get_ta()

    # Convert negative index to positive
    if index < 0:
        index = len(df) + index

    # Get or calculate ATR
    if "atr" in df.columns:
        atr_series = df["atr"]
    else:
        tr = ta.true_range(df["high"], df["low"], df["close"])
        atr_series = ta.sma(tr, length=14)

    # Calculate ATR percentile over lookback period
    start_idx = max(0, index - lookback + 1)
    end_idx = index + 1
    atr_window = atr_series.iloc[start_idx:end_idx].dropna()

    if len(atr_window) < 10:
        return {
            "regime": "NORMAL_VOL",
            "atr_percentile": 50.0,
            "position_multiplier": 1.0,
            "allow_at_grace": False,
            "details": "Not enough ATR data"
        }

    current_atr = atr_window.iloc[-1]
    atr_min = atr_window.min()
    atr_max = atr_window.max()

    if atr_max - atr_min > 0:
        atr_percentile = ((current_atr - atr_min) / (atr_max - atr_min)) * 100
    else:
        atr_percentile = 50.0

    # Classify regime
    if atr_percentile < low_vol_threshold:
        regime = "LOW_VOL"
        position_multiplier = 0.5   # Conservative: half position
        allow_at_grace = False      # Strict AT confirmation required
        details = f"Low volatility (ATR pct={atr_percentile:.1f}%) - conservative mode"
    elif atr_percentile > high_vol_threshold:
        regime = "HIGH_VOL"
        position_multiplier = 1.5   # Aggressive: larger position
        allow_at_grace = True       # Allow AT lag, strong trends
        details = f"High volatility (ATR pct={atr_percentile:.1f}%) - aggressive mode"
    else:
        regime = "NORMAL_VOL"
        position_multiplier = 1.0   # Standard position
        allow_at_grace = False      # Standard AT rules
        details = f"Normal volatility (ATR pct={atr_percentile:.1f}%) - standard mode"

    return {
        "regime": regime,
        "atr_percentile": float(atr_percentile),
        "position_multiplier": position_multiplier,
        "allow_at_grace": allow_at_grace,
        "details": details
    }


def get_regime_multiplier(
    df: pd.DataFrame,
    adx_trending_threshold: float = 20.0,  # v43: Lowered to avoid TRANSITIONAL zone
    adx_ranging_threshold: float = 20.0,   # v43: Same as trending (no TRANSITIONAL)
    atr_extreme_ratio: float = 2.0,
    atr_high_ratio: float = 1.5,
    atr_lookback: int = 100,
    index: int = -1
) -> float:
    """
    Get risk multiplier based on market regime (ADX + ATR).

    This implements a dual-filter approach:
    - ADX determines trend strength (TRENDING vs RANGING)
    - ATR ratio detects volatility spikes

    Decision Matrix:
    | Regime       | ADX   | ATR Ratio | Multiplier |
    |--------------|-------|-----------|------------|
    | TRENDING     | >25   | Normal    | 1.0        |
    | TRENDING     | >25   | >2.0      | 0.5        |
    | RANGING      | <20   | Any       | 0.0        |
    | TRANSITIONAL | 20-25 | Normal    | 0.5        |
    | TRANSITIONAL | 20-25 | >1.5      | 0.25       |

    Args:
        df: DataFrame with 'adx' and 'atr' columns
        adx_trending_threshold: ADX above this = trending (default: 25)
        adx_ranging_threshold: ADX below this = ranging (default: 20)
        atr_extreme_ratio: ATR ratio above this = extreme volatility (default: 2.0)
        atr_high_ratio: ATR ratio above this = high volatility (default: 1.5)
        atr_lookback: Lookback period for ATR average (default: 100)
        index: Row index to check (default: -1 = last row)

    Returns:
        Risk multiplier (0.0 to 1.0)
    """
    if df is None or len(df) < atr_lookback:
        return 1.0  # Default to full risk if insufficient data

    # Get ADX value
    if "adx" not in df.columns:
        return 1.0  # No ADX = assume full risk

    adx = df["adx"].iloc[index]
    if pd.isna(adx):
        return 1.0

    # Calculate ATR ratio (current vs average)
    if "atr" not in df.columns:
        atr_ratio = 1.0
    else:
        atr_current = df["atr"].iloc[index]
        atr_avg = df["atr"].rolling(atr_lookback, min_periods=20).mean().iloc[index]

        if pd.isna(atr_current) or pd.isna(atr_avg) or atr_avg <= 0:
            atr_ratio = 1.0
        else:
            atr_ratio = atr_current / atr_avg

    # Decision matrix
    if adx > adx_trending_threshold:
        # TRENDING - full risk (unless extreme volatility)
        if atr_ratio > atr_extreme_ratio:
            return 0.5  # Extreme vol in trend = reduce
        return 1.0

    elif adx < adx_ranging_threshold:
        # RANGING - skip trades (our weakness based on backtest!)
        return 0.0

    else:
        # TRANSITIONAL (between thresholds) - half risk
        if atr_ratio > atr_high_ratio:
            return 0.25  # High vol + weak trend = minimal
        return 0.5
