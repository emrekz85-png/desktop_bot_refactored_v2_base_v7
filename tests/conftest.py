"""
Shared pytest fixtures for trading bot tests.

This module provides common fixtures for testing indicators, signals,
trade management, and risk calculations.
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add parent directory to path to import main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: Do NOT set HEADLESS_MODE here. The main module's GUI classes (LiveBotWorker, etc.)
# inherit from QThread which requires PyQt5 to be imported. Setting HEADLESS_MODE
# prevents PyQt5 import but the class definitions still try to use QThread.
# Instead, ensure PyQt5 and PyQtWebEngine are installed: pip install PyQt5 PyQtWebEngine


@pytest.fixture(scope="session")
def trading_module():
    """Import the main trading module once per session."""
    import desktop_bot_refactored_v2_base_v7 as bot
    return bot


@pytest.fixture
def trading_config(trading_module):
    """Get the trading configuration."""
    return trading_module.TRADING_CONFIG.copy()


@pytest.fixture
def default_strategy_config(trading_module):
    """Get the default strategy configuration."""
    return trading_module.DEFAULT_STRATEGY_CONFIG.copy()


# ============================================
# OHLCV Data Fixtures
# ============================================

@pytest.fixture
def sample_ohlcv_df():
    """Generate a basic OHLCV DataFrame with 500 candles for testing."""
    np.random.seed(42)  # Reproducible randomness
    n = 500

    # Generate realistic price data with some trend
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.01, n)
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV data
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "open": prices,
        "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
        "close": prices * (1 + np.random.normal(0, 0.002, n)),
        "volume": np.random.uniform(1000, 10000, n),
    }

    df = pd.DataFrame(data)
    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


@pytest.fixture
def minimal_ohlcv_df():
    """Generate minimal OHLCV DataFrame (50 candles) for quick tests."""
    np.random.seed(123)
    n = 50

    base_price = 100.0
    prices = base_price + np.cumsum(np.random.normal(0, 0.5, n))

    data = {
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "open": prices,
        "high": prices + np.abs(np.random.normal(0, 0.3, n)),
        "low": prices - np.abs(np.random.normal(0, 0.3, n)),
        "close": prices + np.random.normal(0, 0.1, n),
        "volume": np.random.uniform(1000, 5000, n),
    }

    df = pd.DataFrame(data)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


@pytest.fixture
def bullish_setup_df():
    """Generate OHLCV data that should trigger a LONG signal.

    Creates a scenario where:
    - Price is near Keltner lower band
    - PBEMA cloud is above current price (target)
    - RSI is oversold
    - ADX shows sufficient trend strength
    """
    np.random.seed(42)
    n = 300  # Need enough candles for EMA200 warmup

    # Start high, drop down, then set up for bounce
    base_price = 100.0

    # Create downtrend followed by consolidation near bottom
    prices = np.zeros(n)
    prices[:200] = base_price - np.linspace(0, 15, 200)  # Downtrend
    prices[200:] = 85 + np.random.normal(0, 0.3, 100)    # Consolidation

    data = {
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "open": prices,
        "high": prices + np.abs(np.random.normal(0.2, 0.3, n)),
        "low": prices - np.abs(np.random.normal(0.2, 0.3, n)),
        "close": prices + np.random.normal(0, 0.15, n),
        "volume": np.random.uniform(1000, 5000, n),
    }

    df = pd.DataFrame(data)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


@pytest.fixture
def bearish_setup_df():
    """Generate OHLCV data that should trigger a SHORT signal.

    Creates a scenario where:
    - Price is near Keltner upper band
    - PBEMA cloud is below current price (target)
    - RSI is overbought
    - ADX shows sufficient trend strength
    """
    np.random.seed(42)
    n = 300

    # Start low, go up, then set up for rejection
    base_price = 100.0

    prices = np.zeros(n)
    prices[:200] = base_price + np.linspace(0, 15, 200)  # Uptrend
    prices[200:] = 115 + np.random.normal(0, 0.3, 100)   # Consolidation at top

    data = {
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "open": prices,
        "high": prices + np.abs(np.random.normal(0.2, 0.3, n)),
        "low": prices - np.abs(np.random.normal(0.2, 0.3, n)),
        "close": prices + np.random.normal(0, 0.15, n),
        "volume": np.random.uniform(1000, 5000, n),
    }

    df = pd.DataFrame(data)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


@pytest.fixture
def df_with_indicators(sample_ohlcv_df, trading_module):
    """Sample OHLCV data with all indicators already calculated."""
    df = sample_ohlcv_df.copy()
    return trading_module.TradingEngine.calculate_indicators(df)


# ============================================
# Trade Manager Fixtures
# ============================================

@pytest.fixture
def trade_manager(trading_module):
    """Create a non-persisting TradeManager for testing."""
    return trading_module.TradeManager(persist=False, verbose=False)


@pytest.fixture
def sim_trade_manager(trading_module):
    """Create a SimTradeManager for testing."""
    initial_balance = trading_module.TRADING_CONFIG["initial_balance"]
    return trading_module.SimTradeManager(initial_balance=initial_balance)


@pytest.fixture
def sample_long_trade():
    """Sample LONG trade data for testing."""
    return {
        "symbol": "BTCUSDT",
        "timeframe": "5m",
        "type": "LONG",
        "entry": 100.0,
        "tp": 103.0,
        "sl": 98.0,
        "setup": "TEST_SETUP",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "open_time_utc": datetime.utcnow(),
    }


@pytest.fixture
def sample_short_trade():
    """Sample SHORT trade data for testing."""
    return {
        "symbol": "BTCUSDT",
        "timeframe": "5m",
        "type": "SHORT",
        "entry": 100.0,
        "tp": 97.0,
        "sl": 102.0,
        "setup": "TEST_SETUP",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "open_time_utc": datetime.utcnow(),
    }


# ============================================
# Mock Fixtures
# ============================================

@pytest.fixture
def mock_best_config_cache(trading_module):
    """Mock the BEST_CONFIG_CACHE with test configurations."""
    original_cache = trading_module.BEST_CONFIG_CACHE.copy()

    # Set up test config
    trading_module.BEST_CONFIG_CACHE.clear()
    trading_module.BEST_CONFIG_CACHE["BTCUSDT"] = {
        "5m": {
            "rr": 2.0,
            "rsi": 35,
            "at_active": False,
            "use_trailing": False,
            "use_dynamic_pbema_tp": True,
            "strategy_mode": "keltner_bounce",
            "_confidence": "high",
        },
        "15m": {
            "rr": 1.5,
            "rsi": 40,
            "at_active": True,
            "use_trailing": False,
            "use_dynamic_pbema_tp": True,
            "strategy_mode": "keltner_bounce",
            "_confidence": "high",
        },
    }

    yield trading_module.BEST_CONFIG_CACHE

    # Restore original cache
    trading_module.BEST_CONFIG_CACHE.clear()
    trading_module.BEST_CONFIG_CACHE.update(original_cache)


@pytest.fixture
def mock_api_response():
    """Mock API response for testing HTTP calls."""
    def _make_response(data, status_code=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = data
        return mock_resp
    return _make_response


# ============================================
# Helper Fixtures
# ============================================

@pytest.fixture
def candle_updater():
    """Helper to create candle update dictionaries."""
    def _make_candle(high, low, close, time_offset_minutes=5):
        return {
            "high": high,
            "low": low,
            "close": close,
            "time_offset": time_offset_minutes,
        }
    return _make_candle


@pytest.fixture
def parity_test_setup(trading_module):
    """Set up both TradeManager and SimTradeManager for parity testing."""
    live_tm = trading_module.TradeManager(persist=False, verbose=False)
    sim_tm = trading_module.SimTradeManager(
        initial_balance=trading_module.TRADING_CONFIG["initial_balance"]
    )

    return {
        "live": live_tm,
        "sim": sim_tm,
        "initial_balance": trading_module.TRADING_CONFIG["initial_balance"],
    }


# ============================================
# Cleanup Fixtures
# ============================================

@pytest.fixture(autouse=True)
def reset_network_cooldown(trading_module):
    """Reset network cooldown before each test."""
    trading_module.TradingEngine._network_cooldown_until = 0
    yield
    trading_module.TradingEngine._network_cooldown_until = 0
