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
    flat_threshold: float = 0.001
) -> pd.DataFrame:
    """
    Calculate AlphaTrend indicator with DUAL lines (buyers vs sellers).

    AlphaTrend is a trend-following indicator that combines ATR and MFI/RSI.
    It provides dynamic support/resistance levels and buyer/seller dominance signals.

    The indicator calculates two lines:
    - at_buyers (blue line): Tracks buyer strength / support levels
    - at_sellers (red line): Tracks seller strength / resistance levels

    Trade signals:
    - LONG: at_buyers > at_sellers (buyers dominant)
    - SHORT: at_sellers > at_buyers (sellers dominant)
    - NO TRADE: at_is_flat = True (no flow, sideways market)

    Args:
        df: DataFrame with OHLCV data
        coeff: ATR multiplier (default: 1.0)
        ap: ATR/MFI period (default: 14)
        flat_lookback: Number of candles to check for flat detection (default: 5)
        flat_threshold: Minimum change ratio to consider non-flat (default: 0.001 = 0.1%)

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
    try:
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)

        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype(float)

        ta = _get_ta()
        n = len(df)

        # Calculate True Range and ATR
        df['_at_tr'] = ta.true_range(df['high'], df['low'], df['close'])
        df['_at_atr'] = ta.sma(df['_at_tr'], length=ap)

        # Use MFI if volume available, otherwise use RSI
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

        # Initialize dual lines
        at_buyers = np.zeros(n)
        at_sellers = np.zeros(n)

        # Initial values
        at_buyers[0] = close_vals[0]
        at_sellers[0] = close_vals[0]

        for i in range(1, n):
            # Buyer line (support tracking)
            # When momentum >= 50, buyers are strong - track higher support
            if momentum[i] >= 50:
                at_buyers[i] = max(at_buyers[i - 1], upT[i])
            else:
                at_buyers[i] = upT[i]

            # Seller line (resistance tracking)
            # When momentum < 50, sellers are strong - track lower resistance
            if momentum[i] < 50:
                at_sellers[i] = min(at_sellers[i - 1], downT[i])
            else:
                at_sellers[i] = downT[i]

        df['at_buyers'] = at_buyers
        df['at_sellers'] = at_sellers

        # Dominance signals
        df['at_buyers_dominant'] = df['at_buyers'] > df['at_sellers']
        df['at_sellers_dominant'] = df['at_sellers'] > df['at_buyers']

        # Flat/no-flow detection
        # If both lines have minimal change over lookback period, market is flat
        df['_at_buyers_change'] = df['at_buyers'].pct_change(flat_lookback).abs()
        df['_at_sellers_change'] = df['at_sellers'].pct_change(flat_lookback).abs()

        df['at_is_flat'] = (
            (df['_at_buyers_change'] < flat_threshold) &
            (df['_at_sellers_change'] < flat_threshold)
        )
        # Fill NaN values (first few rows) as False
        df['at_is_flat'] = df['at_is_flat'].fillna(False)

        # Backward compatibility: single alphatrend line = dominant line
        df['alphatrend'] = np.where(
            df['at_buyers_dominant'],
            df['at_buyers'],
            df['at_sellers']
        )
        df['alphatrend_2'] = df['alphatrend'].shift(2)

        # Cleanup temporary columns
        cleanup_cols = ['_at_tr', '_at_atr', '_at_momentum', '_at_upT', '_at_downT',
                        '_at_buyers_change', '_at_sellers_change']
        df.drop(columns=[c for c in cleanup_cols if c in df.columns], inplace=True, errors='ignore')

        return df

    except Exception as e:
        # Fallback: set basic columns to prevent errors
        print(f"AlphaTrend hesaplama hatası: {e}")
        df['at_buyers'] = df['close']
        df['at_sellers'] = df['close']
        df['at_buyers_dominant'] = True
        df['at_sellers_dominant'] = False
        df['at_is_flat'] = False
        df['alphatrend'] = df['close']
        df['alphatrend_2'] = df['close'].shift(2)
        return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for trading strategies.

    Indicators calculated:
    - RSI(14): Relative Strength Index
    - ADX(14): Average Directional Index
    - PBEMA Cloud (EMA 200): pb_ema_top, pb_ema_bot (for Keltner Bounce)
    - PBEMA Cloud (EMA 150): pb_ema_top_150, pb_ema_bot_150 (for PBEMA Reaction)
    - Slope: Rate of change for EMA levels
    - SSL Baseline: HMA(60)
    - Keltner Channels: baseline +/- EMA(TR) * 0.2
    - AlphaTrend: Optional trend filter

    PERFORMANCE NOTE: This function modifies the DataFrame in-place.
    If you need to preserve the original DataFrame, make a copy before calling.

    Args:
        df: DataFrame with OHLCV data (timestamp, open, high, low, close, volume)

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

    # PBEMA Cloud (EMA 200) - for Keltner Bounce strategy
    df["pb_ema_top"] = ta.ema(df["high"], length=200)
    df["pb_ema_bot"] = ta.ema(df["close"], length=200)

    # PBEMA Cloud (EMA 150) - for PBEMA Reaction strategy
    df["pb_ema_top_150"] = ta.ema(df["high"], length=150)
    df["pb_ema_bot_150"] = ta.ema(df["close"], length=150)

    # Slope (rate of change over 5 periods) - EMA 200
    df["slope_top"] = (df["pb_ema_top"].diff(5) / df["pb_ema_top"]) * 1000
    df["slope_bot"] = (df["pb_ema_bot"].diff(5) / df["pb_ema_bot"]) * 1000

    # Slope - EMA 150
    df["slope_top_150"] = (df["pb_ema_top_150"].diff(5) / df["pb_ema_top_150"]) * 1000
    df["slope_bot_150"] = (df["pb_ema_bot_150"].diff(5) / df["pb_ema_bot_150"]) * 1000

    # SSL Baseline (HMA 60) and Keltner Channels
    df["baseline"] = ta.hma(df["close"], length=60)
    tr = ta.true_range(df["high"], df["low"], df["close"])
    range_ma = ta.ema(tr, length=60)
    df["keltner_upper"] = df["baseline"] + range_ma * 0.2
    df["keltner_lower"] = df["baseline"] - range_ma * 0.2

    # AlphaTrend (for optional trend filtering)
    df = calculate_alphatrend(df, coeff=1, ap=14)

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
    except Exception:
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
    except Exception:
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
