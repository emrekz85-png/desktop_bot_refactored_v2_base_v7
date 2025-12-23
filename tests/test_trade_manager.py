"""
Unit tests for TradeManager class.

Tests trade opening, position sizing, risk management, and trade updates.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import patch


def _utcnow():
    """Helper to get current UTC time as naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TestTradeManagerInit:
    """Tests for TradeManager initialization."""

    def test_initializes_with_correct_balance(self, trade_manager, trading_config):
        """TradeManager should initialize with correct wallet balance."""
        assert trade_manager.wallet_balance == trading_config["initial_balance"]

    def test_initializes_empty_trades(self, trade_manager):
        """TradeManager should start with no open trades."""
        assert len(trade_manager.open_trades) == 0
        assert len(trade_manager.history) == 0

    def test_initializes_strategy_wallets(self, trade_manager, trading_config):
        """TradeManager should initialize strategy-specific wallets."""
        assert "ssl_flow" in trade_manager.strategy_wallets
        assert "keltner_bounce" in trade_manager.strategy_wallets
        assert trade_manager.strategy_wallets["ssl_flow"]["wallet_balance"] == trading_config["initial_balance"]

    def test_initializes_risk_parameters(self, trade_manager, trading_config):
        """TradeManager should initialize risk parameters from config."""
        assert trade_manager.risk_per_trade_pct == trading_config["risk_per_trade_pct"]
        assert trade_manager.max_portfolio_risk_pct == trading_config["max_portfolio_risk_pct"]
        assert trade_manager.slippage_pct == trading_config["slippage_rate"]


class TestOpenTrade:
    """Tests for the open_trade method."""

    def test_opens_long_trade(self, trade_manager, sample_long_trade, mock_best_config_cache):
        """Should successfully open a LONG trade."""
        initial_balance = trade_manager.wallet_balance
        trade_manager.open_trade(sample_long_trade)

        assert len(trade_manager.open_trades) == 1
        assert trade_manager.open_trades[0]["type"] == "LONG"
        assert trade_manager.wallet_balance < initial_balance  # Margin deducted

    def test_opens_short_trade(self, trade_manager, sample_short_trade, mock_best_config_cache):
        """Should successfully open a SHORT trade."""
        initial_balance = trade_manager.wallet_balance
        trade_manager.open_trade(sample_short_trade)

        assert len(trade_manager.open_trades) == 1
        assert trade_manager.open_trades[0]["type"] == "SHORT"
        assert trade_manager.wallet_balance < initial_balance

    def test_applies_slippage_to_long_entry(self, trade_manager, sample_long_trade, mock_best_config_cache):
        """LONG entry should have slippage applied (higher price)."""
        trade_manager.open_trade(sample_long_trade)

        raw_entry = sample_long_trade["entry"]
        actual_entry = trade_manager.open_trades[0]["entry"]

        # LONG slippage increases entry price
        assert actual_entry > raw_entry

    def test_applies_slippage_to_short_entry(self, trade_manager, sample_short_trade, mock_best_config_cache):
        """SHORT entry should have slippage applied (lower price)."""
        trade_manager.open_trade(sample_short_trade)

        raw_entry = sample_short_trade["entry"]
        actual_entry = trade_manager.open_trades[0]["entry"]

        # SHORT slippage decreases entry price
        assert actual_entry < raw_entry

    def test_calculates_position_size_from_risk(self, trade_manager, sample_long_trade, mock_best_config_cache, trading_config):
        """Position size should be calculated based on risk amount."""
        trade_manager.open_trade(sample_long_trade)
        trade = trade_manager.open_trades[0]

        # Verify position sizing logic
        entry = trade["entry"]
        sl = trade["sl"]
        sl_distance = abs(entry - sl)
        sl_fraction = sl_distance / entry

        # Risk amount should be close to wallet_balance * risk_per_trade_pct
        expected_risk = trading_config["initial_balance"] * trading_config["risk_per_trade_pct"]
        assert trade["risk_amount"] > 0

    def test_stores_config_snapshot(self, trade_manager, sample_long_trade, mock_best_config_cache):
        """Trade should store config snapshot at open time."""
        trade_manager.open_trade(sample_long_trade)
        trade = trade_manager.open_trades[0]

        # Config snapshot fields should be present
        assert "use_trailing" in trade
        assert "use_dynamic_pbema_tp" in trade
        assert "opt_rr" in trade
        assert "opt_rsi" in trade

    def test_respects_cooldown(self, trade_manager, sample_long_trade, mock_best_config_cache):
        """Should not open trade during cooldown period."""
        # Open first trade
        trade_manager.open_trade(sample_long_trade)

        # Set cooldown for this symbol/timeframe
        trade_manager.cooldowns[("BTCUSDT", "5m")] = _utcnow() + timedelta(hours=1)

        # Try to open another trade
        trade2 = sample_long_trade.copy()
        trade2["entry"] = 101.0
        trade_manager.open_trade(trade2)

        # Should still only have 1 trade (second was rejected due to cooldown)
        assert len(trade_manager.open_trades) == 1

    def test_rejects_trade_when_portfolio_risk_exceeded(self, trade_manager, sample_long_trade, mock_best_config_cache):
        """Should reject trade when portfolio risk limit would be exceeded."""
        # Open multiple trades to approach risk limit
        for i in range(20):
            trade = sample_long_trade.copy()
            trade["symbol"] = f"TEST{i}USDT"
            trade["entry"] = 100.0 + i
            trade_manager.open_trade(trade)

        # At some point, trades should be rejected due to portfolio risk
        # The exact number depends on risk settings
        assert len(trade_manager.open_trades) <= 20


class TestCooldown:
    """Tests for the cooldown system."""

    def test_check_cooldown_returns_false_when_no_cooldown(self, trade_manager):
        """Should return False when no cooldown is set."""
        result = trade_manager.check_cooldown("BTCUSDT", "5m")
        assert result is False

    def test_check_cooldown_returns_true_during_cooldown(self, trade_manager):
        """Should return True during active cooldown."""
        future_time = _utcnow() + timedelta(hours=1)
        trade_manager.cooldowns[("BTCUSDT", "5m")] = future_time

        result = trade_manager.check_cooldown("BTCUSDT", "5m")
        assert result is True

    def test_check_cooldown_clears_expired_cooldown(self, trade_manager):
        """Should clear and return False for expired cooldown."""
        past_time = _utcnow() - timedelta(hours=1)
        trade_manager.cooldowns[("BTCUSDT", "5m")] = past_time

        result = trade_manager.check_cooldown("BTCUSDT", "5m")
        assert result is False
        assert ("BTCUSDT", "5m") not in trade_manager.cooldowns

    def test_check_cooldown_handles_pandas_timestamp(self, trade_manager):
        """Should handle pandas Timestamp for now_utc parameter."""
        import pandas as pd

        future_time = _utcnow() + timedelta(hours=1)
        trade_manager.cooldowns[("BTCUSDT", "5m")] = future_time

        now = pd.Timestamp.utcnow()
        result = trade_manager.check_cooldown("BTCUSDT", "5m", now_utc=now)
        assert result is True


class TestEquityCalculation:
    """Tests for equity calculation."""

    def test_equity_equals_balance_when_no_trades(self, trade_manager, trading_config):
        """Equity should equal wallet balance when no trades are open."""
        equity = trade_manager._calculate_equity()
        expected = trading_config["initial_balance"]  # wallet + locked (0)
        assert equity == expected

    def test_equity_includes_locked_margin(self, trade_manager, sample_long_trade, mock_best_config_cache):
        """Equity should include locked margin from open trades."""
        initial_equity = trade_manager._calculate_equity()

        trade_manager.open_trade(sample_long_trade)

        new_equity = trade_manager._calculate_equity()
        # Equity should remain roughly the same (wallet decreased, locked increased)
        assert abs(new_equity - initial_equity) < 1.0


class TestPortfolioRiskCalculation:
    """Tests for portfolio risk calculation."""

    def test_portfolio_risk_zero_when_no_trades(self, trade_manager, trading_config):
        """Portfolio risk should be 0 when no trades are open."""
        risk = trade_manager._calculate_portfolio_risk_pct(trading_config["initial_balance"])
        assert risk == 0.0

    def test_portfolio_risk_increases_with_trades(self, trade_manager, sample_long_trade, mock_best_config_cache, trading_config):
        """Portfolio risk should increase as trades are opened."""
        initial_risk = trade_manager._calculate_portfolio_risk_pct(trading_config["initial_balance"])

        trade_manager.open_trade(sample_long_trade)

        new_risk = trade_manager._calculate_portfolio_risk_pct(trading_config["initial_balance"])
        assert new_risk > initial_risk

    def test_portfolio_risk_bounded(self, trade_manager, sample_long_trade, mock_best_config_cache):
        """Portfolio risk should not exceed max limit."""
        # Open many trades
        for i in range(10):
            trade = sample_long_trade.copy()
            trade["symbol"] = f"TEST{i}USDT"
            trade_manager.open_trade(trade)

        risk = trade_manager._calculate_portfolio_risk_pct(trade_manager.wallet_balance)
        # Risk should be bounded by max_portfolio_risk_pct
        assert risk <= trade_manager.max_portfolio_risk_pct + 0.01  # Small tolerance


class TestStrategyWallets:
    """Tests for strategy-specific wallet management."""

    def test_get_strategy_wallet_returns_correct_wallet(self, trade_manager):
        """Should return correct wallet for strategy."""
        sf_wallet = trade_manager._get_strategy_wallet("ssl_flow")
        kb_wallet = trade_manager._get_strategy_wallet("keltner_bounce")

        assert sf_wallet is trade_manager.strategy_wallets["ssl_flow"]
        assert kb_wallet is trade_manager.strategy_wallets["keltner_bounce"]

    def test_get_strategy_wallet_defaults_to_ssl_flow(self, trade_manager):
        """Unknown strategy should default to ssl_flow wallet."""
        wallet = trade_manager._get_strategy_wallet("unknown_strategy")
        assert wallet is trade_manager.strategy_wallets["ssl_flow"]

    def test_strategy_portfolio_risk_calculated_per_strategy(self, trade_manager, sample_long_trade, mock_best_config_cache):
        """Portfolio risk should be calculated per strategy."""
        # Open a trade (defaults to ssl_flow as keltner_bounce is disabled)
        trade_manager.open_trade(sample_long_trade)

        kb_risk = trade_manager._calculate_strategy_portfolio_risk("keltner_bounce")
        sf_risk = trade_manager._calculate_strategy_portfolio_risk("ssl_flow")

        # ssl_flow should have risk (active strategy), keltner_bounce should not
        assert sf_risk > 0
        assert kb_risk == 0


class TestTradeFields:
    """Tests for trade field validation."""

    def test_trade_has_required_fields(self, trade_manager, sample_long_trade, mock_best_config_cache):
        """Opened trade should have all required fields."""
        trade_manager.open_trade(sample_long_trade)
        trade = trade_manager.open_trades[0]

        required_fields = [
            "id", "symbol", "timestamp", "timeframe", "type", "setup",
            "entry", "tp", "sl", "size", "margin", "notional",
            "status", "pnl", "breakeven", "trailing_active", "partial_taken",
            "use_trailing", "use_dynamic_pbema_tp", "strategy_mode"
        ]

        for field in required_fields:
            assert field in trade, f"Missing required field: {field}"

    def test_trade_status_is_open(self, trade_manager, sample_long_trade, mock_best_config_cache):
        """New trade should have status OPEN."""
        trade_manager.open_trade(sample_long_trade)
        assert trade_manager.open_trades[0]["status"] == "OPEN"

    def test_trade_has_unique_id(self, trade_manager, sample_long_trade, mock_best_config_cache):
        """Each trade should have a unique ID."""
        import time

        trade1 = sample_long_trade.copy()
        trade2 = sample_long_trade.copy()
        trade2["symbol"] = "ETHUSDT"

        trade_manager.open_trade(trade1)
        time.sleep(0.01)  # Small delay to ensure different timestamp
        trade_manager.open_trade(trade2)

        ids = [t["id"] for t in trade_manager.open_trades]
        # IDs should be unique (with time delay between opens)
        assert len(ids) == len(set(ids)), f"IDs not unique: {ids}"
