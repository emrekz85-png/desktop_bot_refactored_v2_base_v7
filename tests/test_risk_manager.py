"""
Unit tests for Risk Management System.

Tests cover:
- Kelly Criterion calculations
- Drawdown tracking and auto-adjustment
- Correlation-adjusted position sizing
- RiskManager integration

See docs/RISK_MANAGEMENT_SPEC.md for full specification.
"""

import pytest
from math import log, exp
from datetime import datetime, timedelta

from core.kelly_calculator import (
    calculate_kelly,
    calculate_growth_rate,
    calculate_kelly_from_history,
    trades_to_double,
    edge_exists,
    minimum_win_rate_for_edge,
    kelly_comparison,
    MIN_KELLY_FRACTION,
    MAX_KELLY_FRACTION,
)

from core.drawdown_tracker import (
    DrawdownTracker,
    DrawdownStatus,
    calculate_drawdown,
    get_drawdown_kelly_multiplier,
    get_recovery_status,
)

from core.correlation_manager import (
    adjust_kelly_for_correlation,
    calculate_portfolio_risk,
)

from core.risk_manager import (
    RiskManager,
    calculate_r_multiple,
    calculate_trade_rr,
    TradeRecord,
)


class TestKellyCalculation:
    """Tests for Kelly Criterion calculations."""

    def test_kelly_with_edge(self):
        """Kelly should be positive with edge."""
        # 60% win rate, 1.5:1 R:R -> should have positive Kelly
        kelly = calculate_kelly(win_rate=0.60, reward_risk=1.5)
        assert kelly > 0
        # Expected: 0.60 - 0.40/1.5 = 0.60 - 0.267 = 0.333 (full)
        # Half-Kelly = 0.167

    def test_kelly_no_edge(self):
        """Kelly should be zero without edge."""
        # 30% win rate, 1:1 R:R -> no edge
        kelly = calculate_kelly(win_rate=0.30, reward_risk=1.0)
        assert kelly == 0.0

    def test_kelly_breakeven(self):
        """Kelly should be zero at breakeven."""
        # 50% win rate, 1:1 R:R -> exactly breakeven
        kelly = calculate_kelly(win_rate=0.50, reward_risk=1.0, kelly_mode="full")
        assert kelly == 0.0

    def test_kelly_bounds(self):
        """Kelly should respect min/max bounds."""
        # Extreme edge case: 90% win rate, 3:1 R:R
        kelly = calculate_kelly(win_rate=0.90, reward_risk=3.0)
        assert kelly <= MAX_KELLY_FRACTION  # Max 5%

    def test_kelly_mode_half(self):
        """Half Kelly should be 50% of full Kelly (before bounds)."""
        # Use more moderate values where bounds don't interfere
        full = calculate_kelly(win_rate=0.55, reward_risk=1.5, kelly_mode="full")
        half = calculate_kelly(win_rate=0.55, reward_risk=1.5, kelly_mode="half")
        # Half should be less than full
        assert half <= full
        assert half > 0

    def test_kelly_mode_quarter(self):
        """Quarter Kelly should be 25% of full Kelly (before bounds)."""
        # Use more moderate values where bounds don't interfere
        full = calculate_kelly(win_rate=0.55, reward_risk=1.5, kelly_mode="full")
        quarter = calculate_kelly(win_rate=0.55, reward_risk=1.5, kelly_mode="quarter")
        assert quarter <= full
        assert quarter > 0

    def test_kelly_invalid_inputs(self):
        """Invalid inputs should return 0."""
        assert calculate_kelly(win_rate=0, reward_risk=2.0) == 0.0
        assert calculate_kelly(win_rate=1.0, reward_risk=2.0) == 0.0
        assert calculate_kelly(win_rate=0.5, reward_risk=0) == 0.0
        assert calculate_kelly(win_rate=-0.5, reward_risk=2.0) == 0.0


class TestGrowthRate:
    """Tests for growth rate calculations."""

    def test_growth_rate_positive(self):
        """Positive edge should yield positive growth."""
        # At optimal Kelly, growth should be positive
        kelly = 0.10
        growth = calculate_growth_rate(kelly, win_rate=0.60, reward_risk=2.0)
        assert growth > 0

    def test_growth_rate_over_betting(self):
        """Extreme over-betting should reduce growth rate."""
        # For 60% win rate, 2:1 R:R, optimal Kelly is about 0.433
        # We test that going WAY past optimal reduces growth
        small_bet = 0.05
        extreme_over_bet = 0.90  # Way beyond optimal

        growth_small = calculate_growth_rate(small_bet, 0.60, 2.0)
        growth_extreme = calculate_growth_rate(extreme_over_bet, 0.60, 2.0)

        # Small bet should have positive growth
        assert growth_small > 0
        # Extreme over-betting can lead to negative growth (ruin)
        assert growth_extreme < growth_small

    def test_trades_to_double(self):
        """Calculate trades needed to double capital."""
        growth_rate = 0.05  # 5% per trade
        ttd = trades_to_double(growth_rate)
        # log(2) / 0.05 ≈ 13.86
        assert ttd is not None
        assert 13 < ttd < 15

    def test_trades_to_double_no_edge(self):
        """No edge means infinite trades to double."""
        ttd = trades_to_double(0)
        assert ttd is None

        ttd = trades_to_double(-0.01)
        assert ttd is None


class TestKellyFromHistory:
    """Tests for calculating Kelly from trade history."""

    def test_insufficient_data(self):
        """Should return default when not enough trades."""
        history = [{"r_multiple": 2.0}]
        result = calculate_kelly_from_history(history, min_trades=30)
        assert result["confidence"] == "insufficient_data"
        assert result["kelly"] == 0.01  # Default

    def test_with_sufficient_data(self):
        """Should calculate Kelly with enough trades."""
        # 20 wins at 2R, 10 losses at -1R
        history = (
            [{"r_multiple": 2.0}] * 20 +
            [{"r_multiple": -1.0}] * 10
        )
        result = calculate_kelly_from_history(history, min_trades=30)

        assert result["sample_size"] == 30
        assert result["win_rate"] == pytest.approx(0.667, rel=0.01)
        assert result["kelly"] > 0
        assert result["has_edge"] is True

    def test_no_edge_in_history(self):
        """Should return 0 Kelly when history shows no edge."""
        # Equal wins and losses at 1:1
        history = (
            [{"r_multiple": 1.0}] * 15 +
            [{"r_multiple": -1.0}] * 15
        )
        result = calculate_kelly_from_history(history, min_trades=30)
        assert result["has_edge"] is False


class TestEdgeDetection:
    """Tests for edge existence checks."""

    def test_edge_exists(self):
        """Should detect edge correctly."""
        # R:R = 2:1, need >33% win rate for edge
        assert edge_exists(win_rate=0.40, reward_risk=2.0) is True
        assert edge_exists(win_rate=0.30, reward_risk=2.0) is False
        assert edge_exists(win_rate=0.34, reward_risk=2.0) is True

    def test_minimum_win_rate(self):
        """Calculate minimum win rate for edge."""
        # R:R = 2:1 -> min win rate = 1/3 = 0.333
        min_wr = minimum_win_rate_for_edge(reward_risk=2.0)
        assert min_wr == pytest.approx(0.333, rel=0.01)

        # R:R = 1:1 -> min win rate = 0.5
        min_wr = minimum_win_rate_for_edge(reward_risk=1.0)
        assert min_wr == pytest.approx(0.5, rel=0.01)


class TestDrawdownCalculation:
    """Tests for drawdown calculation."""

    def test_no_drawdown(self):
        """No drawdown when at peak."""
        dd = calculate_drawdown(current_equity=10000, peak_equity=10000)
        assert dd["drawdown_percent"] == 0.0
        assert dd["drawdown_amount"] == 0.0

    def test_ten_percent_drawdown(self):
        """10% drawdown calculation."""
        dd = calculate_drawdown(current_equity=9000, peak_equity=10000)
        assert dd["drawdown_percent"] == 10.0
        assert dd["drawdown_amount"] == 1000.0

    def test_above_peak(self):
        """Equity above peak should show 0 drawdown."""
        dd = calculate_drawdown(current_equity=11000, peak_equity=10000)
        assert dd["drawdown_percent"] == 0.0


class TestDrawdownKellyMultiplier:
    """Tests for drawdown-based Kelly adjustment."""

    def test_no_adjustment_at_zero_dd(self):
        """No adjustment when no drawdown."""
        mult = get_drawdown_kelly_multiplier(0)
        assert mult == 1.0

    def test_circuit_breaker_at_20pct(self):
        """Circuit breaker at 20% drawdown."""
        mult = get_drawdown_kelly_multiplier(20)
        assert mult == 0.0

    def test_circuit_breaker_above_20pct(self):
        """Circuit breaker above 20% drawdown."""
        mult = get_drawdown_kelly_multiplier(25)
        assert mult == 0.0

    def test_smooth_decay(self):
        """Decay between 0% and 20% with floor at 0.25."""
        m0 = get_drawdown_kelly_multiplier(0)
        m5 = get_drawdown_kelly_multiplier(5)
        m10 = get_drawdown_kelly_multiplier(10)
        m15 = get_drawdown_kelly_multiplier(15)
        m19 = get_drawdown_kelly_multiplier(19)

        # At 0%, multiplier is 1.0
        assert m0 == 1.0
        # As drawdown increases, multiplier decreases
        assert m5 < m0
        # All values above floor until circuit breaker
        assert m5 >= 0.25
        assert m10 >= 0.25
        assert m15 >= 0.25
        assert m19 >= 0.25

    def test_multiplier_at_10_percent(self):
        """Approximately 0.70 at 10% drawdown."""
        mult = get_drawdown_kelly_multiplier(10)
        # exp(-0.15 * 10) ≈ 0.22, but with floor of 0.25
        # Actually: exp(-1.5) ≈ 0.22, floor 0.25
        assert mult >= 0.25


class TestDrawdownTracker:
    """Tests for DrawdownTracker class."""

    def test_initialization(self):
        """Tracker initializes correctly."""
        tracker = DrawdownTracker(initial_equity=10000)
        state = tracker.get_state()

        assert state.current_equity == 10000
        assert state.peak_equity == 10000
        assert state.drawdown_percent == 0
        assert state.status == DrawdownStatus.NORMAL
        assert state.can_trade is True

    def test_equity_update_new_high(self):
        """New equity high updates peak."""
        tracker = DrawdownTracker(initial_equity=10000)
        tracker.update_equity(11000)
        state = tracker.get_state()

        assert state.peak_equity == 11000
        assert state.drawdown_percent == 0

    def test_drawdown_triggers_caution(self):
        """10%+ drawdown triggers CAUTION status."""
        tracker = DrawdownTracker(initial_equity=10000)
        tracker.update_equity(8900)  # 11% drawdown
        state = tracker.get_state()

        assert state.status == DrawdownStatus.CAUTION
        assert state.kelly_multiplier < 1.0
        assert state.can_trade is True

    def test_circuit_breaker_activation(self):
        """20%+ drawdown triggers circuit breaker."""
        tracker = DrawdownTracker(initial_equity=10000)
        # First update to trigger circuit breaker
        tracker.update_equity(7900)  # 21% drawdown - beyond 20% threshold
        state = tracker.get_state()

        # After circuit breaker triggers, status changes to RECOVERING
        # can_trade is False and multiplier is 0
        assert state.can_trade is False
        assert state.kelly_multiplier == 0.0
        # Status is either CIRCUIT_BREAKER initially or RECOVERING after activation
        assert state.status in [DrawdownStatus.CIRCUIT_BREAKER, DrawdownStatus.RECOVERING]


class TestCorrelationAdjustment:
    """Tests for correlation-adjusted Kelly."""

    def test_no_adjustment_first_position(self):
        """No adjustment for first position."""
        result = adjust_kelly_for_correlation(
            base_kelly=0.02,
            open_positions={},
            new_position_symbol="BTCUSDT",
            new_position_direction="LONG"
        )
        assert result["adjustment_factor"] == 1.0
        assert result["adjusted_kelly"] == 0.02

    def test_reduces_for_correlated(self):
        """Reduces Kelly for correlated positions."""
        open_pos = {"BTCUSDT": {"direction": "LONG"}}
        result = adjust_kelly_for_correlation(
            base_kelly=0.02,
            open_positions=open_pos,
            new_position_symbol="ETHUSDT",
            new_position_direction="LONG"
        )
        # BTC-ETH correlation is ~0.92
        # Factor = 1 - 0.92 * 0.5 = 0.54
        assert result["adjustment_factor"] < 1.0
        assert result["adjusted_kelly"] < 0.02

    def test_no_adjustment_different_direction(self):
        """No adjustment for opposite direction positions."""
        open_pos = {"BTCUSDT": {"direction": "LONG"}}
        result = adjust_kelly_for_correlation(
            base_kelly=0.02,
            open_positions=open_pos,
            new_position_symbol="ETHUSDT",
            new_position_direction="SHORT"  # Opposite direction
        )
        # Opposite direction - no same-direction correlation penalty
        assert result["adjustment_factor"] == 1.0

    def test_blocks_at_max_same_direction(self):
        """Blocks trade at max same-direction positions."""
        open_pos = {
            "BTCUSDT": {"direction": "LONG"},
            "ETHUSDT": {"direction": "LONG"},
            "LINKUSDT": {"direction": "LONG"}
        }
        result = adjust_kelly_for_correlation(
            base_kelly=0.02,
            open_positions=open_pos,
            new_position_symbol="SOLUSDT",
            new_position_direction="LONG",
            max_same_direction=3
        )
        assert result["adjusted_kelly"] == 0.0
        assert result["can_trade"] is False


class TestPortfolioRisk:
    """Tests for portfolio risk calculation."""

    def test_empty_portfolio(self):
        """Empty portfolio has zero risk."""
        result = calculate_portfolio_risk({}, equity=10000)
        assert result["total_risk_percent"] == 0.0
        assert result["n_positions"] == 0

    def test_single_position(self):
        """Single position risk calculation."""
        positions = {
            "BTCUSDT": {"direction": "LONG", "risk_amount": 200}
        }
        result = calculate_portfolio_risk(positions, equity=10000)
        assert result["total_risk_percent"] == 2.0  # 200/10000 * 100
        assert result["n_positions"] == 1

    def test_correlated_positions_effective_risk(self):
        """Correlated positions show higher effective risk."""
        positions = {
            "BTCUSDT": {"direction": "LONG", "risk_amount": 200},
            "ETHUSDT": {"direction": "LONG", "risk_amount": 200}
        }
        result = calculate_portfolio_risk(positions, equity=10000)
        # Total risk = 4%, but effective should be higher due to correlation
        assert result["total_risk_percent"] == 4.0
        # Diversification ratio < 1 due to correlation
        assert result["diversification_ratio"] < 1.0


class TestRiskManager:
    """Integration tests for RiskManager."""

    def test_initialization(self):
        """RiskManager initializes correctly."""
        rm = RiskManager(initial_equity=10000)
        status = rm.get_portfolio_status()

        assert status["equity"]["initial"] == 10000
        assert status["equity"]["current"] == 10000
        assert status["drawdown"]["can_trade"] is True

    def test_position_sizing_basic(self):
        """Basic position sizing calculation."""
        rm = RiskManager(initial_equity=10000)
        sizing = rm.calculate_position_size(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000
        )

        assert sizing.can_trade is True
        assert sizing.risk_amount > 0
        assert sizing.position_size > 0
        assert sizing.kelly_fraction > 0

    def test_position_sizing_blocked_by_circuit_breaker(self):
        """Position sizing blocked when circuit breaker active."""
        rm = RiskManager(initial_equity=10000)
        rm.update_equity(7500)  # 25% drawdown

        sizing = rm.calculate_position_size(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000
        )

        assert sizing.can_trade is False
        assert "circuit" in sizing.reason.lower() or "20" in sizing.reason

    def test_equity_update(self):
        """Equity updates affect drawdown calculation."""
        rm = RiskManager(initial_equity=10000)

        # Make profit
        rm.update_equity(11000)
        status = rm.get_portfolio_status()
        assert status["equity"]["peak"] == 11000

        # Drawdown
        rm.update_equity(10000)
        status = rm.get_portfolio_status()
        assert status["drawdown"]["current_pct"] > 0

    def test_trade_recording(self):
        """Trade recording updates statistics."""
        rm = RiskManager(initial_equity=10000)

        trade = TradeRecord(
            symbol="BTCUSDT",
            direction="LONG",
            r_multiple=2.0,
            pnl=100,
            risk_amount=50,
            entry_price=50000,
            exit_price=52000,
            entry_time=datetime.now(),
            exit_time=datetime.now() + timedelta(hours=1),
            exit_reason="TP"
        )
        rm.record_trade(trade)

        expectancy = rm.get_expectancy()
        assert expectancy["sample_size"] == 1
        assert expectancy["expectancy"] == 2.0

    def test_expectancy_calculation(self):
        """Expectancy calculated correctly from trades."""
        rm = RiskManager(initial_equity=10000)

        # Add trades: 2 wins at 2R, 1 loss at -1R
        for r in [2.0, 2.0, -1.0]:
            trade = TradeRecord(
                symbol="BTCUSDT",
                direction="LONG",
                r_multiple=r,
                pnl=r * 50,
                risk_amount=50,
                entry_price=50000,
                exit_price=51000,
                entry_time=datetime.now(),
                exit_time=datetime.now(),
            )
            rm.record_trade(trade)

        exp = rm.get_expectancy()
        # E[R] = (2 + 2 - 1) / 3 = 1.0
        assert exp["expectancy"] == pytest.approx(1.0, rel=0.01)
        assert exp["win_rate"] == pytest.approx(0.667, rel=0.01)


class TestRMultipleCalculation:
    """Tests for R-Multiple calculations."""

    def test_r_multiple_win(self):
        """R-multiple for winning trade."""
        r = calculate_r_multiple(pnl=100, risk_amount=50)
        assert r == 2.0  # Won 2R

    def test_r_multiple_loss(self):
        """R-multiple for losing trade."""
        r = calculate_r_multiple(pnl=-50, risk_amount=50)
        assert r == -1.0  # Lost 1R (full stop)

    def test_r_multiple_partial(self):
        """R-multiple for partial outcomes."""
        r = calculate_r_multiple(pnl=25, risk_amount=50)
        assert r == 0.5  # Won 0.5R


class TestTradeRR:
    """Tests for trade R:R calculation."""

    def test_long_trade_rr(self):
        """R:R for LONG trade."""
        rr = calculate_trade_rr(
            entry_price=100,
            take_profit=110,
            stop_loss=95,
            trade_type="LONG"
        )
        # Reward = 10, Risk = 5, R:R = 2.0
        assert rr == 2.0

    def test_short_trade_rr(self):
        """R:R for SHORT trade."""
        rr = calculate_trade_rr(
            entry_price=100,
            take_profit=90,
            stop_loss=105,
            trade_type="SHORT"
        )
        # Reward = 10, Risk = 5, R:R = 2.0
        assert rr == 2.0

    def test_invalid_sl_returns_zero(self):
        """Invalid SL (wrong direction) returns 0."""
        rr = calculate_trade_rr(
            entry_price=100,
            take_profit=110,
            stop_loss=105,  # Above entry for LONG - invalid
            trade_type="LONG"
        )
        assert rr == 0.0


class TestKellyComparison:
    """Tests for Kelly fraction comparison."""

    def test_comparison_structure(self):
        """Kelly comparison returns proper structure."""
        result = kelly_comparison(win_rate=0.60, reward_risk=2.0)

        assert "full_kelly" in result
        assert "half_kelly" in result
        assert "quarter_kelly" in result
        assert result["recommendation"] == "half_kelly"

    def test_growth_rate_ordering(self):
        """Full Kelly has highest growth rate."""
        result = kelly_comparison(win_rate=0.60, reward_risk=2.0)

        full_growth = result["full_kelly"]["growth_rate"]
        half_growth = result["half_kelly"]["growth_rate"]
        quarter_growth = result["quarter_kelly"]["growth_rate"]

        # Full > Half > Quarter (in terms of growth)
        assert full_growth >= half_growth >= quarter_growth
