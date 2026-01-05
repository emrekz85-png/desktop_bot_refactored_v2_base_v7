"""
Test Helpers - Common utilities for test scripts.

This module eliminates duplicated sys.path manipulation and common imports
across 20+ test files in the project.
"""

import sys
import os

# Add project root to path (eliminates repeated sys.path.insert in every test file)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def get_binance_client():
    """Get configured BinanceClient for testing."""
    from core.binance_client import BinanceClient
    return BinanceClient()


def get_test_data(symbol: str = "BTCUSDT", timeframe: str = "15m", candles: int = 2000):
    """
    Fetch test data with indicators calculated.

    Args:
        symbol: Trading pair (default: BTCUSDT)
        timeframe: Timeframe (default: 15m)
        candles: Number of candles to fetch (default: 2000)

    Returns:
        DataFrame with OHLCV data and calculated indicators
    """
    from core.binance_client import BinanceClient
    from core.indicators import calculate_indicators

    client = BinanceClient()
    df = client.get_historical_klines(symbol, timeframe, candles)
    df = calculate_indicators(df)
    return df


def get_default_config():
    """Get default strategy configuration for testing."""
    from core.config import DEFAULT_STRATEGY_CONFIG
    return DEFAULT_STRATEGY_CONFIG.copy()
