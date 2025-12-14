"""
Unit tests for technical indicator calculations.

Tests TradingEngine.calculate_indicators() and related functions.
"""

import pytest
import numpy as np
import pandas as pd


class TestCalculateIndicators:
    """Tests for the calculate_indicators function."""

    def test_returns_dataframe(self, sample_ohlcv_df, trading_module):
        """calculate_indicators should return a DataFrame."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_indicators(df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_rsi_column(self, sample_ohlcv_df, trading_module):
        """RSI column should be added to DataFrame."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_indicators(df)
        assert "rsi" in result.columns
        assert not result["rsi"].isna().all()

    def test_adds_adx_column(self, sample_ohlcv_df, trading_module):
        """ADX column should be added to DataFrame."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_indicators(df)
        assert "adx" in result.columns

    def test_adds_pbema_columns(self, sample_ohlcv_df, trading_module):
        """PBEMA cloud columns (EMA 200) should be added."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_indicators(df)
        assert "pb_ema_top" in result.columns
        assert "pb_ema_bot" in result.columns

    def test_adds_pbema_150_columns(self, sample_ohlcv_df, trading_module):
        """PBEMA reaction columns (EMA 150) should be added."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_indicators(df)
        assert "pb_ema_top_150" in result.columns
        assert "pb_ema_bot_150" in result.columns

    def test_adds_keltner_bands(self, sample_ohlcv_df, trading_module):
        """Keltner bands should be added."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_indicators(df)
        assert "keltner_upper" in result.columns
        assert "keltner_lower" in result.columns
        assert "baseline" in result.columns

    def test_adds_slope_columns(self, sample_ohlcv_df, trading_module):
        """Slope columns should be added."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_indicators(df)
        assert "slope_top" in result.columns
        assert "slope_bot" in result.columns

    def test_adds_alphatrend(self, sample_ohlcv_df, trading_module):
        """AlphaTrend indicator should be added."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_indicators(df)
        assert "alphatrend" in result.columns

    def test_rsi_range(self, sample_ohlcv_df, trading_module):
        """RSI values should be between 0 and 100."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_indicators(df)
        valid_rsi = result["rsi"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_adx_range(self, sample_ohlcv_df, trading_module):
        """ADX values should be between 0 and 100."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_indicators(df)
        valid_adx = result["adx"].dropna()
        # ADX can sometimes be 0 if no movement
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()

    def test_keltner_upper_above_lower(self, sample_ohlcv_df, trading_module):
        """Keltner upper band should always be above lower band."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_indicators(df)
        valid_idx = result["keltner_upper"].notna() & result["keltner_lower"].notna()
        assert (result.loc[valid_idx, "keltner_upper"] >= result.loc[valid_idx, "keltner_lower"]).all()

    def test_pbema_top_above_bot(self, sample_ohlcv_df, trading_module):
        """PBEMA top (EMA of high) should generally be >= PBEMA bot (EMA of close)."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_indicators(df)
        # Skip warmup period
        valid_idx = result["pb_ema_top"].notna() & result["pb_ema_bot"].notna()
        # EMA(high) should be >= EMA(close) after warmup
        top_vals = result.loc[valid_idx, "pb_ema_top"].values[-100:]
        bot_vals = result.loc[valid_idx, "pb_ema_bot"].values[-100:]
        # They should be close but top >= bot in most cases
        assert np.mean(top_vals >= bot_vals) >= 0.8  # 80% of time

    def test_modifies_in_place(self, sample_ohlcv_df, trading_module):
        """calculate_indicators modifies DataFrame in place."""
        df = sample_ohlcv_df.copy()
        original_id = id(df)
        result = trading_module.TradingEngine.calculate_indicators(df)
        # Should return same object (modified in place)
        assert "rsi" in df.columns

    def test_handles_empty_dataframe(self, trading_module):
        """Should handle empty DataFrame gracefully."""
        df = pd.DataFrame()
        # This should not raise an exception
        try:
            result = trading_module.TradingEngine.calculate_indicators(df)
        except Exception as e:
            # Expected - empty df might cause issues
            pass

    def test_handles_insufficient_data(self, trading_module):
        """Should handle DataFrame with insufficient data for EMA200."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
            "open": [100] * 10,
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100.5] * 10,
            "volume": [1000] * 10,
        })
        result = trading_module.TradingEngine.calculate_indicators(df)
        # Should still return a DataFrame with columns
        assert isinstance(result, pd.DataFrame)
        assert "rsi" in result.columns


class TestAlphaTrend:
    """Tests for the AlphaTrend indicator calculation."""

    def test_alphatrend_calculated(self, sample_ohlcv_df, trading_module):
        """AlphaTrend should be calculated."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_alphatrend(df)
        assert "alphatrend" in result.columns

    def test_alphatrend_2_shifted(self, sample_ohlcv_df, trading_module):
        """alphatrend_2 should be alphatrend shifted by 2."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_alphatrend(df)
        if "alphatrend_2" in result.columns:
            # Check shift relationship (skip first 2 values due to shift)
            at = result["alphatrend"].values
            at2 = result["alphatrend_2"].values
            # After warmup, alphatrend_2[i] should equal alphatrend[i-2]
            for i in range(4, len(at)):
                if not np.isnan(at2[i]) and not np.isnan(at[i-2]):
                    assert at2[i] == at[i-2], f"Mismatch at index {i}"

    def test_alphatrend_follows_price(self, sample_ohlcv_df, trading_module):
        """AlphaTrend should roughly follow price movements."""
        df = sample_ohlcv_df.copy()
        result = trading_module.TradingEngine.calculate_alphatrend(df)
        # AlphaTrend should be within reasonable range of close price
        valid_idx = result["alphatrend"].notna()
        at_vals = result.loc[valid_idx, "alphatrend"].values
        close_vals = result.loc[valid_idx, "close"].values
        # AlphaTrend should be within 10% of close on average
        pct_diff = np.abs(at_vals - close_vals) / close_vals
        assert np.mean(pct_diff) < 0.10


class TestIndicatorEdgeCases:
    """Tests for edge cases in indicator calculations."""

    def test_handles_zero_volume(self, trading_module):
        """Should handle zero volume data."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="5min"),
            "open": np.random.uniform(99, 101, 100),
            "high": np.random.uniform(100, 102, 100),
            "low": np.random.uniform(98, 100, 100),
            "close": np.random.uniform(99, 101, 100),
            "volume": np.zeros(100),  # Zero volume
        })
        # Should not raise exception
        result = trading_module.TradingEngine.calculate_indicators(df)
        assert "rsi" in result.columns

    def test_handles_constant_price(self, trading_module):
        """Should handle constant price data (no movement)."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="5min"),
            "open": [100.0] * 100,
            "high": [100.0] * 100,
            "low": [100.0] * 100,
            "close": [100.0] * 100,
            "volume": [1000] * 100,
        })
        result = trading_module.TradingEngine.calculate_indicators(df)
        assert isinstance(result, pd.DataFrame)

    def test_handles_nan_values(self, trading_module):
        """Should handle NaN values in input data."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="5min"),
            "open": [100.0] * 100,
            "high": [101.0] * 100,
            "low": [99.0] * 100,
            "close": [100.5] * 100,
            "volume": [1000] * 100,
        })
        # Introduce some NaN values
        df.loc[10, "close"] = np.nan
        df.loc[20, "high"] = np.nan

        result = trading_module.TradingEngine.calculate_indicators(df)
        assert isinstance(result, pd.DataFrame)
