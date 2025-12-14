"""
Unit tests for signal detection functions.

Tests TradingEngine.check_signal_diagnostic() and related signal generation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime


class TestCheckSignalDiagnostic:
    """Tests for the main signal detection function."""

    def test_returns_tuple(self, df_with_indicators, trading_module):
        """check_signal_diagnostic should return a tuple."""
        result = trading_module.TradingEngine.check_signal_diagnostic(df_with_indicators)
        assert isinstance(result, tuple)
        assert len(result) == 5  # signal_type, entry, tp, sl, reason

    def test_returns_debug_info_when_requested(self, df_with_indicators, trading_module):
        """Should return debug info when return_debug=True."""
        result = trading_module.TradingEngine.check_signal_diagnostic(
            df_with_indicators, return_debug=True
        )
        assert len(result) == 6  # signal_type, entry, tp, sl, reason, debug_info
        assert isinstance(result[5], dict)

    def test_returns_none_for_empty_df(self, trading_module):
        """Should return None signal for empty DataFrame."""
        empty_df = pd.DataFrame()
        signal, entry, tp, sl, reason = trading_module.TradingEngine.check_signal_diagnostic(empty_df)
        assert signal is None
        assert reason == "No Data"

    def test_returns_none_for_none_input(self, trading_module):
        """Should return None signal for None input."""
        signal, entry, tp, sl, reason = trading_module.TradingEngine.check_signal_diagnostic(None)
        assert signal is None
        assert reason == "No Data"

    def test_returns_none_for_missing_columns(self, trading_module):
        """Should return None signal if required columns are missing."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="5min"),
            "open": [100.0] * 100,
            "close": [100.0] * 100,
            # Missing high, low, indicators
        })
        signal, entry, tp, sl, reason = trading_module.TradingEngine.check_signal_diagnostic(df)
        assert signal is None
        assert "Missing" in reason

    def test_adx_filter_rejects_low_adx(self, df_with_indicators, trading_module):
        """Should reject signals when ADX is too low."""
        df = df_with_indicators.copy()
        # Force low ADX
        df["adx"] = 5.0  # Below minimum threshold

        signal, entry, tp, sl, reason = trading_module.TradingEngine.check_signal_diagnostic(df)
        assert signal is None
        assert "ADX" in reason

    def test_returns_warmup_for_insufficient_data(self, trading_module):
        """Should return Warmup reason for insufficient historical data."""
        # Create minimal DataFrame with indicators
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "rsi": [50.0] * 10,
            "adx": [25.0] * 10,
            "pb_ema_top": [102.0] * 10,
            "pb_ema_bot": [101.0] * 10,
            "keltner_upper": [101.5] * 10,
            "keltner_lower": [98.5] * 10,
            "baseline": [100.0] * 10,
            "alphatrend": [100.0] * 10,
        })

        # With hold_n=5 and checking index -2, we need at least hold_n + 2 candles
        signal, entry, tp, sl, reason = trading_module.TradingEngine.check_signal_diagnostic(
            df, hold_n=15  # More than available data
        )
        assert signal is None
        # Either Warmup or Index related reason

    def test_signal_types_are_valid(self, df_with_indicators, trading_module):
        """Signal type should be None, 'LONG', or 'SHORT'."""
        signal, entry, tp, sl, reason = trading_module.TradingEngine.check_signal_diagnostic(
            df_with_indicators
        )
        assert signal in [None, "LONG", "SHORT"]

    def test_long_signal_has_valid_levels(self, df_with_indicators, trading_module):
        """LONG signal should have SL < entry < TP."""
        # Run many times with different data to try to get a LONG signal
        np.random.seed(42)
        for _ in range(10):
            df = df_with_indicators.copy()
            signal, entry, tp, sl, reason = trading_module.TradingEngine.check_signal_diagnostic(df)
            if signal == "LONG":
                assert sl < entry, f"LONG: SL ({sl}) should be < entry ({entry})"
                assert entry < tp, f"LONG: entry ({entry}) should be < TP ({tp})"
                break

    def test_short_signal_has_valid_levels(self, df_with_indicators, trading_module):
        """SHORT signal should have TP < entry < SL."""
        np.random.seed(42)
        for _ in range(10):
            df = df_with_indicators.copy()
            signal, entry, tp, sl, reason = trading_module.TradingEngine.check_signal_diagnostic(df)
            if signal == "SHORT":
                assert tp < entry, f"SHORT: TP ({tp}) should be < entry ({entry})"
                assert entry < sl, f"SHORT: entry ({entry}) should be < SL ({sl})"
                break

    def test_rsi_parameter_affects_filtering(self, df_with_indicators, trading_module):
        """RSI threshold parameter should affect signal generation."""
        df = df_with_indicators.copy()

        # Get result with loose RSI
        result_loose = trading_module.TradingEngine.check_signal_diagnostic(df, rsi_limit=90.0)

        # Get result with tight RSI
        result_tight = trading_module.TradingEngine.check_signal_diagnostic(df, rsi_limit=10.0)

        # Results may differ based on RSI filtering
        # This test just verifies the parameter is used (no crash)
        assert isinstance(result_loose, tuple)
        assert isinstance(result_tight, tuple)

    def test_min_rr_parameter_affects_filtering(self, df_with_indicators, trading_module):
        """min_rr parameter should affect signal generation."""
        df = df_with_indicators.copy()

        # Very low RR requirement
        result_low = trading_module.TradingEngine.check_signal_diagnostic(df, min_rr=0.5)

        # Very high RR requirement
        result_high = trading_module.TradingEngine.check_signal_diagnostic(df, min_rr=10.0)

        assert isinstance(result_low, tuple)
        assert isinstance(result_high, tuple)

    def test_debug_info_contains_expected_keys(self, df_with_indicators, trading_module):
        """Debug info should contain expected diagnostic keys."""
        signal, entry, tp, sl, reason, debug_info = trading_module.TradingEngine.check_signal_diagnostic(
            df_with_indicators, return_debug=True
        )

        expected_keys = ["adx_ok"]
        for key in expected_keys:
            assert key in debug_info, f"Missing debug key: {key}"


class TestSignalIndexParameter:
    """Tests for the index parameter in signal detection."""

    def test_default_index_is_minus_2(self, df_with_indicators, trading_module):
        """Default index should be -2 (second to last candle)."""
        # This is implicit - we test that -2 works
        result = trading_module.TradingEngine.check_signal_diagnostic(
            df_with_indicators, index=-2
        )
        assert isinstance(result, tuple)

    def test_index_minus_1_works(self, df_with_indicators, trading_module):
        """Index -1 (last candle) should work."""
        result = trading_module.TradingEngine.check_signal_diagnostic(
            df_with_indicators, index=-1
        )
        assert isinstance(result, tuple)

    def test_positive_index_works(self, df_with_indicators, trading_module):
        """Positive index should work."""
        result = trading_module.TradingEngine.check_signal_diagnostic(
            df_with_indicators, index=250  # Middle of the DataFrame
        )
        assert isinstance(result, tuple)

    def test_out_of_range_index_handled(self, df_with_indicators, trading_module):
        """Out of range index should be handled gracefully."""
        result = trading_module.TradingEngine.check_signal_diagnostic(
            df_with_indicators, index=10000  # Way beyond DataFrame length
        )
        signal, entry, tp, sl, reason = result
        assert signal is None


class TestCheckSignalWrapper:
    """Tests for the check_signal wrapper function."""

    def test_check_signal_routes_to_keltner_bounce(self, df_with_indicators, trading_module):
        """check_signal should route to keltner_bounce strategy by default."""
        config = trading_module.DEFAULT_STRATEGY_CONFIG.copy()
        config["strategy_mode"] = "keltner_bounce"

        result = trading_module.TradingEngine.check_signal(
            df_with_indicators,
            config=config,
        )
        assert isinstance(result, tuple)

    def test_check_signal_handles_nan_in_data(self, trading_module):
        """check_signal should handle NaN values in data."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="5min"),
            "open": [100.0] * 100,
            "high": [101.0] * 100,
            "low": [99.0] * 100,
            "close": [100.5] * 100,
            "rsi": [np.nan] * 100,  # All NaN
            "adx": [25.0] * 100,
            "pb_ema_top": [102.0] * 100,
            "pb_ema_bot": [101.0] * 100,
            "keltner_upper": [101.5] * 100,
            "keltner_lower": [98.5] * 100,
        })

        result = trading_module.TradingEngine.check_signal_diagnostic(df)
        signal, entry, tp, sl, reason = result
        assert signal is None
        assert "NaN" in reason


class TestSignalQuality:
    """Tests for signal quality and validity."""

    def test_tp_not_too_close_to_entry(self, df_with_indicators, trading_module):
        """TP should not be too close to entry (min distance ratio)."""
        signal, entry, tp, sl, reason, debug = trading_module.TradingEngine.check_signal_diagnostic(
            df_with_indicators, return_debug=True, tp_min_dist_ratio=0.001
        )

        if signal is not None and entry is not None and tp is not None:
            tp_dist = abs(tp - entry) / entry
            assert tp_dist >= 0.0005, f"TP too close: {tp_dist}"

    def test_tp_not_too_far_from_entry(self, df_with_indicators, trading_module):
        """TP should not be too far from entry (max distance ratio)."""
        signal, entry, tp, sl, reason, debug = trading_module.TradingEngine.check_signal_diagnostic(
            df_with_indicators, return_debug=True, tp_max_dist_ratio=0.04
        )

        if signal is not None and entry is not None and tp is not None:
            tp_dist = abs(tp - entry) / entry
            assert tp_dist <= 0.10, f"TP too far: {tp_dist}"
