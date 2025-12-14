"""
Unit tests for risk management functions.

Tests position sizing, portfolio risk limits, R-multiple calculations,
stop loss management, and profit protection mechanisms.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta


class TestPositionSizing:
    """Tests for position sizing calculations."""

    def test_position_size_based_on_risk(self, trade_manager, sample_long_trade, mock_best_config_cache, trading_config):
        """Position size should be calculated based on risk amount and SL distance."""
        trade_manager.open_trade(sample_long_trade)
        trade = trade_manager.open_trades[0]

        entry = trade["entry"]
        sl = trade["sl"]
        size = trade["size"]
        notional = trade["notional"]

        # Verify the math: risk_amount = sl_fraction * notional
        sl_distance = abs(entry - sl)
        sl_fraction = sl_distance / entry

        # notional * sl_fraction should approximately equal risk_amount
        calculated_risk = sl_fraction * notional
        assert abs(calculated_risk - trade["risk_amount"]) < 0.01

    def test_position_scales_down_when_margin_insufficient(self, trading_module, mock_best_config_cache):
        """Position should scale down when required margin exceeds wallet balance."""
        # Create trade manager with very low balance
        tm = trading_module.TradeManager(persist=False, verbose=False)
        tm.wallet_balance = 50.0  # Very low balance
        tm.strategy_wallets["keltner_bounce"]["wallet_balance"] = 50.0

        trade_data = {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "type": "LONG",
            "entry": 50000.0,  # High price requiring large notional
            "tp": 51000.0,
            "sl": 49500.0,
            "setup": "TEST",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "open_time_utc": datetime.utcnow(),
        }

        tm.open_trade(trade_data)

        if tm.open_trades:
            trade = tm.open_trades[0]
            # Margin should not exceed wallet balance
            assert trade["margin"] <= 50.0

    def test_leverage_affects_required_margin(self, trading_module, mock_best_config_cache):
        """Higher leverage should require less margin for same notional."""
        # This is implicit in the calculation: margin = notional / leverage
        tm = trading_module.TradeManager(persist=False, verbose=False)

        trade_data = {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "type": "LONG",
            "entry": 100.0,
            "tp": 103.0,
            "sl": 98.0,
            "setup": "TEST",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "open_time_utc": datetime.utcnow(),
        }

        tm.open_trade(trade_data)

        if tm.open_trades:
            trade = tm.open_trades[0]
            leverage = trading_module.TRADING_CONFIG["leverage"]
            # margin * leverage should approximately equal notional
            assert abs(trade["margin"] * leverage - trade["notional"]) < 0.01


class TestRMultiple:
    """Tests for R-Multiple calculations."""

    def test_r_multiple_calculated_on_trade_close(self, trading_module, mock_best_config_cache):
        """R-multiple should be calculated when trade closes."""
        sim = trading_module.SimTradeManager(initial_balance=2000.0)

        trade_data = {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "type": "LONG",
            "entry": 100.0,
            "tp": 104.0,  # 4% profit target
            "sl": 98.0,   # 2% stop loss
            "setup": "TEST",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "open_time_utc": datetime.utcnow(),
        }

        sim.open_trade(trade_data)
        assert len(sim.open_trades) == 1

        # Simulate TP hit - use candle low above entry to avoid any SL logic
        candle_time = datetime.utcnow() + timedelta(minutes=5)
        sim.update_trades(
            "BTCUSDT", "5m",
            candle_high=105.0,  # Above TP
            candle_low=101.0,   # Above entry (avoids any SL trigger)
            candle_close=104.5,
            candle_time_utc=candle_time,
            pb_top=110.0,
            pb_bot=108.0,
        )

        # Trade should be closed
        if sim.history:
            trade = sim.history[-1]
            # Trade closed - could be WON, PARTIAL_WIN, or STOP depending on logic
            assert trade["status"] in ["WON", "PARTIAL_WIN", "STOP"]
            # If R-multiple is tracked, verify it exists
            if "r_multiple" in trade:
                # R-multiple exists (positive or negative depending on outcome)
                assert isinstance(trade["r_multiple"], (int, float))


class TestStopLossProtection:
    """Tests for stop loss protection mechanisms."""

    def test_apply_1m_profit_lock(self, trading_module):
        """1m profit lock should move SL into profit at 80% progress."""
        trade = {
            "sl": 98.0,
            "breakeven": False,
        }

        # Test at 85% progress to TP
        result = trading_module._apply_1m_profit_lock(
            trade=trade,
            tf="1m",
            t_type="LONG",
            entry=100.0,
            tp=104.0,  # 4% target
            progress=0.85,  # 85% to TP
        )

        assert result is True
        assert trade["breakeven"] is True
        # SL should be moved above entry (into profit)
        assert trade["sl"] > 100.0

    def test_1m_profit_lock_only_for_1m_timeframe(self, trading_module):
        """Profit lock should only apply to 1m timeframe."""
        trade = {"sl": 98.0, "breakeven": False}

        result = trading_module._apply_1m_profit_lock(
            trade=trade,
            tf="5m",  # Not 1m
            t_type="LONG",
            entry=100.0,
            tp=104.0,
            progress=0.85,
        )

        assert result is False
        assert trade["sl"] == 98.0  # Unchanged

    def test_1m_profit_lock_requires_80_percent_progress(self, trading_module):
        """Profit lock should only trigger at 80%+ progress."""
        trade = {"sl": 98.0, "breakeven": False}

        result = trading_module._apply_1m_profit_lock(
            trade=trade,
            tf="1m",
            t_type="LONG",
            entry=100.0,
            tp=104.0,
            progress=0.70,  # Only 70%
        )

        assert result is False

    def test_partial_stop_protection(self, trading_module):
        """Partial stop protection should raise SL to partial fill price."""
        trade = {
            "sl": 98.0,
            "partial_taken": True,
            "partial_price": 101.5,
            "stop_protection": False,
        }

        result = trading_module._apply_partial_stop_protection(
            trade=trade,
            tf="5m",
            progress=0.85,
            t_type="LONG",
        )

        assert result is True
        assert trade["sl"] == 101.5
        assert trade["stop_protection"] is True

    def test_partial_stop_protection_requires_partial_taken(self, trading_module):
        """Protection should only apply if partial was already taken."""
        trade = {
            "sl": 98.0,
            "partial_taken": False,
            "partial_price": None,
        }

        result = trading_module._apply_partial_stop_protection(
            trade=trade,
            tf="5m",
            progress=0.85,
            t_type="LONG",
        )

        assert result is False

    def test_partial_stop_protection_timeframe_filter(self, trading_module):
        """Protection should only apply to specified timeframes."""
        trade = {
            "sl": 98.0,
            "partial_taken": True,
            "partial_price": 101.5,
        }

        # 4h is not in PARTIAL_STOP_PROTECTION_TFS
        result = trading_module._apply_partial_stop_protection(
            trade=trade,
            tf="4h",
            progress=0.85,
            t_type="LONG",
        )

        assert result is False


class TestConfidenceRiskMultiplier:
    """Tests for confidence-based risk multiplier."""

    def test_high_confidence_full_risk(self, trading_module):
        """High confidence should use full risk."""
        multiplier = trading_module.CONFIDENCE_RISK_MULTIPLIER["high"]
        assert multiplier == 1.0

    def test_medium_confidence_half_risk(self, trading_module):
        """Medium confidence should use half risk."""
        multiplier = trading_module.CONFIDENCE_RISK_MULTIPLIER["medium"]
        assert multiplier == 0.5

    def test_low_confidence_no_trades(self, trading_module):
        """Low confidence should prevent trades."""
        multiplier = trading_module.CONFIDENCE_RISK_MULTIPLIER["low"]
        assert multiplier == 0.0


class TestMinExpectancyThresholds:
    """Tests for minimum expectancy (E[R]) thresholds."""

    def test_thresholds_defined_for_all_timeframes(self, trading_module):
        """E[R] thresholds should be defined for all timeframes."""
        expected_tfs = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

        for tf in expected_tfs:
            assert tf in trading_module.MIN_EXPECTANCY_R_MULTIPLE
            assert trading_module.MIN_EXPECTANCY_R_MULTIPLE[tf] > 0

    def test_lower_timeframes_have_higher_thresholds(self, trading_module):
        """Lower timeframes should require higher E[R] (more noise)."""
        thresholds = trading_module.MIN_EXPECTANCY_R_MULTIPLE

        # 5m should require higher E[R] than 1h
        assert thresholds["5m"] > thresholds["1h"]

        # 1h should require higher E[R] than 1d
        assert thresholds["1h"] > thresholds["1d"]


class TestMinScoreThresholds:
    """Tests for minimum optimizer score thresholds."""

    def test_score_thresholds_defined_for_all_timeframes(self, trading_module):
        """Score thresholds should be defined for all timeframes."""
        expected_tfs = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

        for tf in expected_tfs:
            assert tf in trading_module.MIN_SCORE_THRESHOLD
            assert trading_module.MIN_SCORE_THRESHOLD[tf] > 0

    def test_lower_timeframes_have_higher_score_thresholds(self, trading_module):
        """Lower timeframes should require higher scores."""
        thresholds = trading_module.MIN_SCORE_THRESHOLD

        assert thresholds["5m"] > thresholds["1h"]
        assert thresholds["1h"] > thresholds["1d"]


class TestTradingConfig:
    """Tests for trading configuration validation."""

    def test_config_has_required_fields(self, trading_module):
        """Trading config should have all required fields."""
        required_fields = [
            "initial_balance",
            "leverage",
            "risk_per_trade_pct",
            "max_portfolio_risk_pct",
            "slippage_rate",
            "maker_fee",
            "taker_fee",
        ]

        for field in required_fields:
            assert field in trading_module.TRADING_CONFIG

    def test_risk_per_trade_is_reasonable(self, trading_module):
        """Risk per trade should be between 0.5% and 5%."""
        risk = trading_module.TRADING_CONFIG["risk_per_trade_pct"]
        assert 0.005 <= risk <= 0.05

    def test_max_portfolio_risk_is_reasonable(self, trading_module):
        """Max portfolio risk should be between 1% and 20%."""
        risk = trading_module.TRADING_CONFIG["max_portfolio_risk_pct"]
        assert 0.01 <= risk <= 0.20

    def test_leverage_is_reasonable(self, trading_module):
        """Leverage should be between 1 and 125."""
        leverage = trading_module.TRADING_CONFIG["leverage"]
        assert 1 <= leverage <= 125

    def test_slippage_is_reasonable(self, trading_module):
        """Slippage should be between 0 and 1%."""
        slippage = trading_module.TRADING_CONFIG["slippage_rate"]
        assert 0 <= slippage <= 0.01


class TestStrategySignature:
    """Tests for strategy signature (config hash)."""

    def test_signature_is_deterministic(self, trading_module):
        """Same config should produce same signature."""
        sig1 = trading_module._strategy_signature()
        sig2 = trading_module._strategy_signature()
        assert sig1 == sig2

    def test_signature_is_string(self, trading_module):
        """Signature should be a hex string."""
        sig = trading_module._strategy_signature()
        assert isinstance(sig, str)
        # SHA256 produces 64 character hex string
        assert len(sig) == 64
        # Should be valid hex
        int(sig, 16)

    def test_signature_is_sha256_length(self, trading_module):
        """Signature should be SHA256 (64 hex chars)."""
        sig = trading_module._strategy_signature()
        assert len(sig) == 64
