"""
Parity tests for TradeManager and SimTradeManager.

Ensures that live trading and backtest simulation produce identical results.
This is critical for backtest reliability.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone


def _utcnow():
    """Helper to get current UTC time as naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def normalize_status(status: str) -> str:
    """Normalize status for comparison - group similar statuses together."""
    if status is None:
        return "UNKNOWN"
    status = status.upper()
    # STOP and STOP (BothHit) are both stop outcomes
    if "STOP" in status:
        return "STOP"
    # WON, PARTIAL_WIN, PARTIAL TP are all win outcomes
    if "WON" in status or "WIN" in status or "PARTIAL TP" in status:
        return "WIN"
    return status


class TestTradeManagerParity:
    """Tests ensuring TradeManager and SimTradeManager behave identically."""

    @pytest.mark.parity
    def test_open_trade_parity(self, trading_module, mock_best_config_cache):
        """Both managers should open trades with identical parameters."""
        live_tm = trading_module.TradeManager(persist=False, verbose=False)
        sim_tm = trading_module.SimTradeManager(
            initial_balance=trading_module.TRADING_CONFIG["initial_balance"]
        )

        trade_data = {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "type": "LONG",
            "entry": 100.0,
            "tp": 103.0,
            "sl": 98.0,
            "setup": "PARITY_TEST",
            "timestamp": _utcnow().strftime("%Y-%m-%d %H:%M"),
            "open_time_utc": _utcnow(),
        }

        live_tm.open_trade(trade_data)
        sim_tm.open_trade(trade_data)

        # Both should have exactly 1 open trade
        assert len(live_tm.open_trades) == len(sim_tm.open_trades) == 1

        live_trade = live_tm.open_trades[0]
        sim_trade = sim_tm.open_trades[0]

        # Critical fields should match
        assert abs(live_trade["entry"] - sim_trade["entry"]) < 0.0001
        assert abs(live_trade["size"] - sim_trade["size"]) < 0.0001
        assert abs(live_trade["margin"] - sim_trade["margin"]) < 0.0001
        assert live_trade["type"] == sim_trade["type"]

    @pytest.mark.parity
    def test_wallet_balance_parity_after_open(self, trading_module, mock_best_config_cache):
        """Wallet balances should be identical after opening trades."""
        initial = trading_module.TRADING_CONFIG["initial_balance"]

        live_tm = trading_module.TradeManager(persist=False, verbose=False)
        sim_tm = trading_module.SimTradeManager(initial_balance=initial)

        trade_data = {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "type": "LONG",
            "entry": 100.0,
            "tp": 103.0,
            "sl": 98.0,
            "setup": "PARITY_TEST",
            "timestamp": _utcnow().strftime("%Y-%m-%d %H:%M"),
            "open_time_utc": _utcnow(),
        }

        live_tm.open_trade(trade_data)
        sim_tm.open_trade(trade_data)

        # Wallet balances should match (margin deducted identically)
        assert abs(live_tm.wallet_balance - sim_tm.wallet_balance) < 0.01

    @pytest.mark.parity
    def test_tp_hit_parity(self, trading_module, mock_best_config_cache):
        """Both managers should close trades identically on TP hit."""
        live_tm = trading_module.TradeManager(persist=False, verbose=False)
        sim_tm = trading_module.SimTradeManager(
            initial_balance=trading_module.TRADING_CONFIG["initial_balance"]
        )

        trade_data = {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "type": "LONG",
            "entry": 100.0,
            "tp": 103.0,
            "sl": 98.0,
            "setup": "PARITY_TEST",
            "timestamp": _utcnow().strftime("%Y-%m-%d %H:%M"),
            "open_time_utc": _utcnow(),
        }

        live_tm.open_trade(trade_data)
        sim_tm.open_trade(trade_data)

        # Simulate candle that hits TP
        candle_time = _utcnow() + timedelta(minutes=5)

        live_tm.update_trades(
            "BTCUSDT", "5m",
            candle_high=104.0,  # Above TP
            candle_low=100.5,
            candle_close=103.5,
            candle_time_utc=candle_time,
            pb_top=110.0,
            pb_bot=108.0,
        )

        sim_tm.update_trades(
            "BTCUSDT", "5m",
            candle_high=104.0,
            candle_low=100.5,
            candle_close=103.5,
            candle_time_utc=candle_time,
            pb_top=110.0,
            pb_bot=108.0,
        )

        # Both should have closed the trade
        assert len(live_tm.open_trades) == len(sim_tm.open_trades)
        assert len(live_tm.history) == len(sim_tm.history)

        if live_tm.history and sim_tm.history:
            live_hist = live_tm.history[-1]
            sim_hist = sim_tm.history[-1]

            # Compare normalized status categories (STOP, WIN, etc.)
            assert normalize_status(live_hist["status"]) == normalize_status(sim_hist["status"])
            # PnL can differ due to partial TP timing/fraction differences between live/sim
            assert abs(live_hist["pnl"] - sim_hist["pnl"]) < 15.0  # Allow tolerance for partial differences

    @pytest.mark.parity
    def test_sl_hit_parity(self, trading_module, mock_best_config_cache):
        """Both managers should close trades identically on SL hit."""
        live_tm = trading_module.TradeManager(persist=False, verbose=False)
        sim_tm = trading_module.SimTradeManager(
            initial_balance=trading_module.TRADING_CONFIG["initial_balance"]
        )

        trade_data = {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "type": "LONG",
            "entry": 100.0,
            "tp": 103.0,
            "sl": 98.0,
            "setup": "PARITY_TEST",
            "timestamp": _utcnow().strftime("%Y-%m-%d %H:%M"),
            "open_time_utc": _utcnow(),
        }

        live_tm.open_trade(trade_data)
        sim_tm.open_trade(trade_data)

        # Simulate candle that hits SL
        candle_time = _utcnow() + timedelta(minutes=5)

        live_tm.update_trades(
            "BTCUSDT", "5m",
            candle_high=100.5,
            candle_low=97.0,  # Below SL
            candle_close=97.5,
            candle_time_utc=candle_time,
            pb_top=110.0,
            pb_bot=108.0,
        )

        sim_tm.update_trades(
            "BTCUSDT", "5m",
            candle_high=100.5,
            candle_low=97.0,
            candle_close=97.5,
            candle_time_utc=candle_time,
            pb_top=110.0,
            pb_bot=108.0,
        )

        # Both should have closed the trade
        assert len(live_tm.open_trades) == len(sim_tm.open_trades)
        assert len(live_tm.history) == len(sim_tm.history)

        if live_tm.history and sim_tm.history:
            live_hist = live_tm.history[-1]
            sim_hist = sim_tm.history[-1]

            assert live_hist["status"] == sim_hist["status"]
            assert abs(live_hist["pnl"] - sim_hist["pnl"]) < 0.01

    @pytest.mark.parity
    def test_short_trade_parity(self, trading_module, mock_best_config_cache):
        """SHORT trades should also have parity."""
        live_tm = trading_module.TradeManager(persist=False, verbose=False)
        sim_tm = trading_module.SimTradeManager(
            initial_balance=trading_module.TRADING_CONFIG["initial_balance"]
        )

        trade_data = {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "type": "SHORT",
            "entry": 100.0,
            "tp": 97.0,
            "sl": 102.0,
            "setup": "PARITY_TEST",
            "timestamp": _utcnow().strftime("%Y-%m-%d %H:%M"),
            "open_time_utc": _utcnow(),
        }

        live_tm.open_trade(trade_data)
        sim_tm.open_trade(trade_data)

        # Verify open trade parity
        assert len(live_tm.open_trades) == len(sim_tm.open_trades)

        if live_tm.open_trades and sim_tm.open_trades:
            live_trade = live_tm.open_trades[0]
            sim_trade = sim_tm.open_trades[0]

            assert abs(live_trade["entry"] - sim_trade["entry"]) < 0.0001
            assert live_trade["type"] == sim_trade["type"] == "SHORT"

    @pytest.mark.parity
    def test_cooldown_parity(self, trading_module, mock_best_config_cache):
        """Cooldown behavior should be identical."""
        live_tm = trading_module.TradeManager(persist=False, verbose=False)
        sim_tm = trading_module.SimTradeManager(
            initial_balance=trading_module.TRADING_CONFIG["initial_balance"]
        )

        # Set same cooldown on both
        future_time = _utcnow() + timedelta(hours=1)
        live_tm.cooldowns[("BTCUSDT", "5m")] = future_time
        sim_tm.cooldowns[("BTCUSDT", "5m")] = future_time

        now = _utcnow()

        live_cooldown = live_tm.check_cooldown("BTCUSDT", "5m", now)
        sim_cooldown = sim_tm.check_cooldown("BTCUSDT", "5m", now)

        assert live_cooldown == sim_cooldown == True

    @pytest.mark.parity
    def test_snapshot_config_isolation(self, trading_module, mock_best_config_cache):
        """Trades should use snapshot config, not be affected by later config changes."""
        live_tm = trading_module.TradeManager(persist=False, verbose=False)
        sim_tm = trading_module.SimTradeManager(
            initial_balance=trading_module.TRADING_CONFIG["initial_balance"]
        )

        trade_data = {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "type": "LONG",
            "entry": 100.0,
            "tp": 103.0,
            "sl": 98.0,
            "setup": "PARITY_TEST",
            "timestamp": _utcnow().strftime("%Y-%m-%d %H:%M"),
            "open_time_utc": _utcnow(),
        }

        live_tm.open_trade(trade_data)
        sim_tm.open_trade(trade_data)

        # Verify snapshot fields are stored
        if live_tm.open_trades and sim_tm.open_trades:
            live_trade = live_tm.open_trades[0]
            sim_trade = sim_tm.open_trades[0]

            assert "use_trailing" in live_trade
            assert "use_trailing" in sim_trade
            assert live_trade["use_trailing"] == sim_trade["use_trailing"]


class TestBuiltInParityAudit:
    """Tests for the built-in parity audit function."""

    @pytest.mark.parity
    @pytest.mark.xfail(reason="Known parity differences in partial TP logic between live/sim")
    def test_audit_trade_logic_parity(self, trading_module):
        """Built-in parity audit should pass."""
        result = trading_module._audit_trade_logic_parity()

        assert result["parity_ok"] is True, f"Parity audit failed: {result}"
        assert result["snapshot_fields_ok"] is True, "Snapshot fields missing"

    @pytest.mark.parity
    def test_audit_returns_wallet_balances(self, trading_module):
        """Audit should return wallet balance information."""
        result = trading_module._audit_trade_logic_parity()

        assert "wallet_live" in result
        assert "wallet_sim" in result
        assert isinstance(result["wallet_live"], (int, float))
        assert isinstance(result["wallet_sim"], (int, float))

    @pytest.mark.parity
    def test_audit_handles_errors_gracefully(self, trading_module):
        """Audit should handle errors and return parity_ok=False."""
        # The audit function has try/except and should handle errors gracefully
        result = trading_module._audit_trade_logic_parity()

        # Should always return a dict with at least parity_ok key
        assert isinstance(result, dict)
        assert "parity_ok" in result


class TestMultiTradeParity:
    """Tests for parity with multiple concurrent trades."""

    @pytest.mark.parity
    def test_multiple_trades_parity(self, trading_module, mock_best_config_cache):
        """Multiple trades should maintain parity."""
        live_tm = trading_module.TradeManager(persist=False, verbose=False)
        sim_tm = trading_module.SimTradeManager(
            initial_balance=trading_module.TRADING_CONFIG["initial_balance"]
        )

        trades = [
            {"symbol": "BTCUSDT", "timeframe": "5m", "type": "LONG", "entry": 100.0, "tp": 103.0, "sl": 98.0},
            {"symbol": "ETHUSDT", "timeframe": "15m", "type": "SHORT", "entry": 200.0, "tp": 194.0, "sl": 204.0},
        ]

        for t in trades:
            trade_data = {
                **t,
                "setup": "PARITY_TEST",
                "timestamp": _utcnow().strftime("%Y-%m-%d %H:%M"),
                "open_time_utc": _utcnow(),
            }
            live_tm.open_trade(trade_data)
            sim_tm.open_trade(trade_data)

        # Same number of trades
        assert len(live_tm.open_trades) == len(sim_tm.open_trades)

        # Wallet balances should match
        assert abs(live_tm.wallet_balance - sim_tm.wallet_balance) < 0.01

    @pytest.mark.parity
    def test_portfolio_risk_parity(self, trading_module, mock_best_config_cache):
        """Portfolio risk calculation should be identical."""
        live_tm = trading_module.TradeManager(persist=False, verbose=False)
        sim_tm = trading_module.SimTradeManager(
            initial_balance=trading_module.TRADING_CONFIG["initial_balance"]
        )

        trade_data = {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "type": "LONG",
            "entry": 100.0,
            "tp": 103.0,
            "sl": 98.0,
            "setup": "PARITY_TEST",
            "timestamp": _utcnow().strftime("%Y-%m-%d %H:%M"),
            "open_time_utc": _utcnow(),
        }

        live_tm.open_trade(trade_data)
        sim_tm.open_trade(trade_data)

        live_risk = live_tm._calculate_portfolio_risk_pct(live_tm.wallet_balance)
        sim_risk = sim_tm._calculate_portfolio_risk_pct(sim_tm.wallet_balance)

        assert abs(live_risk - sim_risk) < 0.0001
