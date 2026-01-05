"""
Risk Manager Module

Central risk management coordinator that integrates:
- Kelly Criterion position sizing
- Correlation-adjusted portfolio risk
- Drawdown-based auto-adjustment
- R-Multiple tracking

This is the main interface for the risk management system.

References:
    - docs/RISK_MANAGEMENT_SPEC.md
    - core/kelly_calculator.py
    - core/drawdown_tracker.py
    - core/correlation_manager.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from statistics import mean

from .kelly_calculator import (
    calculate_kelly,
    calculate_kelly_from_history,
    calculate_growth_rate,
    trades_to_double,
    MIN_KELLY_FRACTION,
    MAX_KELLY_FRACTION,
    MIN_TRADES_FOR_KELLY
)
from .drawdown_tracker import (
    DrawdownTracker,
    DrawdownStatus,
    calculate_drawdown,
    get_drawdown_kelly_multiplier
)
from .correlation_manager import (
    CorrelationManager,
    adjust_kelly_for_correlation,
    calculate_portfolio_risk
)
from .logging_config import get_logger

_logger = get_logger(__name__)


# Default Configuration
RISK_CONFIG = {
    # Kelly Parameters
    "kelly_mode": "half",                    # "full", "half", "quarter", "dynamic"
    "min_trades_for_kelly": 30,              # Minimum trades before using Kelly
    "default_kelly_fraction": 0.0175,        # 1.75% default (matches existing config)

    # Position Limits
    "max_single_position_risk": 0.05,        # 5% max per trade
    "max_portfolio_risk": 0.15,              # 15% total risk
    "max_positions_same_direction": 3,       # Max 3 LONG or 3 SHORT
    "max_total_positions": 5,                # Max 5 total positions

    # Drawdown Parameters
    "max_drawdown": 20.0,                    # 20% circuit breaker
    "drawdown_recovery_required": 5.0,       # 5% recovery to resume
    "drawdown_kelly_decay": 0.15,            # Exponential decay constant

    # Correlation Parameters
    "default_correlation": 0.70,             # Default if unknown
    "correlation_adjustment_enabled": True,
    "max_correlation_reduction": 0.50,       # Max 50% reduction for correlated

    # R-Multiple Thresholds
    "min_expectancy_to_trade": 0.05,         # E[R] > 0.05 required
    "edge_degradation_threshold": 0.50,      # 50% of all-time E[R]
    "rolling_window": 20,                    # Trades for rolling E[R]
}


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    can_trade: bool
    risk_amount: float
    position_size: float
    notional_value: float
    margin_required: float
    kelly_fraction: float
    kelly_components: Dict = field(default_factory=dict)
    reason: str = ""


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    symbol: str
    direction: str
    r_multiple: float
    pnl: float
    risk_amount: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str = ""


class RiskManager:
    """
    Central risk management coordinator.

    Integrates Kelly Criterion position sizing, correlation-adjusted portfolio risk,
    drawdown-based auto-adjustment, and R-Multiple tracking.

    Usage:
        rm = RiskManager(initial_equity=10000)

        # Before opening a trade
        sizing = rm.calculate_position_size(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000
        )

        if sizing.can_trade:
            # Open trade with sizing.position_size
            ...

        # After closing a trade
        rm.record_trade(trade_record)
        rm.update_equity(new_balance)
    """

    def __init__(
        self,
        initial_equity: float,
        config: Dict = None,
        leverage: float = 10.0
    ):
        """
        Initialize risk manager.

        Args:
            initial_equity: Starting account equity
            config: Risk configuration (uses RISK_CONFIG defaults if not provided)
            leverage: Trading leverage (default 10x)
        """
        self.config = {**RISK_CONFIG, **(config or {})}
        self.leverage = leverage

        # Equity tracking
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.peak_equity = initial_equity

        # Drawdown tracker
        self.drawdown_tracker = DrawdownTracker(
            initial_equity=initial_equity,
            max_drawdown_percent=self.config["max_drawdown"],
            decay_k=self.config["drawdown_kelly_decay"],
            recovery_required=self.config["drawdown_recovery_required"]
        )

        # Correlation manager
        self.correlation_manager = CorrelationManager(
            max_positions_same_direction=self.config["max_positions_same_direction"],
            max_total_positions=self.config["max_total_positions"]
        )

        # Trade history
        self.trade_history: List[TradeRecord] = []

        # Open positions tracking
        self.open_positions: Dict[str, Dict] = {}

        # Kelly state
        self._cached_kelly: Optional[float] = None
        self._kelly_last_updated: Optional[datetime] = None

        _logger.info(
            f"RiskManager initialized: equity=${initial_equity}, "
            f"kelly_mode={self.config['kelly_mode']}, "
            f"max_dd={self.config['max_drawdown']}%"
        )

    def calculate_position_size(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> PositionSizeResult:
        """
        Master method for position sizing.

        Steps:
        1. Check circuit breaker
        2. Calculate base Kelly from history (or use default)
        3. Adjust for drawdown
        4. Adjust for correlation
        5. Apply constraints
        6. Calculate final position size

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            direction: "LONG" or "SHORT"
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            PositionSizeResult with all sizing details
        """
        kelly_components = {}

        # Step 1: Check circuit breaker
        dd_state = self.drawdown_tracker.get_state()
        if not dd_state.can_trade:
            return PositionSizeResult(
                can_trade=False,
                risk_amount=0.0,
                position_size=0.0,
                notional_value=0.0,
                margin_required=0.0,
                kelly_fraction=0.0,
                kelly_components={"circuit_breaker": True},
                reason=dd_state.message
            )

        # Step 2: Calculate base Kelly
        base_kelly = self._calculate_base_kelly()
        kelly_components["base_kelly"] = base_kelly

        # Step 3: Adjust for drawdown
        dd_multiplier = dd_state.kelly_multiplier
        kelly_after_dd = base_kelly * dd_multiplier
        kelly_components["drawdown_multiplier"] = dd_multiplier
        kelly_components["kelly_after_drawdown"] = kelly_after_dd

        # Step 4: Adjust for correlation
        corr_result = adjust_kelly_for_correlation(
            base_kelly=kelly_after_dd,
            open_positions=self.open_positions,
            new_position_symbol=symbol,
            new_position_direction=direction,
            max_same_direction=self.config["max_positions_same_direction"]
        )

        if not corr_result["can_trade"]:
            return PositionSizeResult(
                can_trade=False,
                risk_amount=0.0,
                position_size=0.0,
                notional_value=0.0,
                margin_required=0.0,
                kelly_fraction=0.0,
                kelly_components=kelly_components,
                reason=corr_result["reason"]
            )

        final_kelly = corr_result["adjusted_kelly"]
        kelly_components["correlation_adjustment"] = corr_result["adjustment_factor"]
        kelly_components["final_kelly"] = final_kelly

        # Step 5: Apply constraints
        final_kelly = max(MIN_KELLY_FRACTION, min(MAX_KELLY_FRACTION, final_kelly))

        # Also check portfolio risk limit
        portfolio_risk = calculate_portfolio_risk(
            self.open_positions,
            self.current_equity
        )
        remaining_risk_budget = (
            self.config["max_portfolio_risk"] * 100 -
            portfolio_risk["total_risk_percent"]
        )

        if remaining_risk_budget <= 0:
            return PositionSizeResult(
                can_trade=False,
                risk_amount=0.0,
                position_size=0.0,
                notional_value=0.0,
                margin_required=0.0,
                kelly_fraction=0.0,
                kelly_components=kelly_components,
                reason=f"Portfolio risk limit reached ({portfolio_risk['total_risk_percent']:.1f}%)"
            )

        # Cap Kelly to remaining budget
        max_kelly_from_budget = remaining_risk_budget / 100
        final_kelly = min(final_kelly, max_kelly_from_budget)
        kelly_components["final_kelly_after_constraints"] = final_kelly

        # Step 6: Calculate position size
        sizing = self._calculate_position_from_kelly(
            kelly_fraction=final_kelly,
            entry_price=entry_price,
            stop_loss=stop_loss,
            direction=direction
        )

        return PositionSizeResult(
            can_trade=True,
            risk_amount=sizing["risk_amount"],
            position_size=sizing["position_size"],
            notional_value=sizing["notional_value"],
            margin_required=sizing["margin_required"],
            kelly_fraction=final_kelly,
            kelly_components=kelly_components,
            reason="OK"
        )

    def _calculate_base_kelly(self) -> float:
        """
        Calculate base Kelly fraction from trade history.

        Returns default if insufficient data.
        """
        if len(self.trade_history) < self.config["min_trades_for_kelly"]:
            return self.config["default_kelly_fraction"]

        # Calculate Kelly from history
        history_dicts = [
            {"r_multiple": t.r_multiple}
            for t in self.trade_history
        ]

        kelly_result = calculate_kelly_from_history(
            trade_history=history_dicts,
            min_trades=self.config["min_trades_for_kelly"],
            kelly_mode=self.config["kelly_mode"]
        )

        return kelly_result["kelly"]

    def _calculate_position_from_kelly(
        self,
        kelly_fraction: float,
        entry_price: float,
        stop_loss: float,
        direction: str
    ) -> Dict:
        """
        Calculate actual position size from Kelly fraction.

        Args:
            kelly_fraction: Fraction of equity to risk
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: LONG or SHORT

        Returns:
            Dict with risk_amount, position_size, notional_value, margin_required
        """
        # Risk amount = Kelly × Equity
        risk_amount = kelly_fraction * self.current_equity

        # Stop loss distance
        if direction.upper() == "LONG":
            sl_distance = entry_price - stop_loss
        else:
            sl_distance = stop_loss - entry_price

        sl_distance = abs(sl_distance)

        if sl_distance <= 0:
            return {
                "risk_amount": 0.0,
                "position_size": 0.0,
                "notional_value": 0.0,
                "margin_required": 0.0
            }

        # Position size = Risk / SL distance (in base units)
        position_size = risk_amount / sl_distance

        # Notional value
        notional_value = position_size * entry_price

        # Margin required
        margin_required = notional_value / self.leverage

        return {
            "risk_amount": round(risk_amount, 2),
            "position_size": round(position_size, 8),
            "notional_value": round(notional_value, 2),
            "margin_required": round(margin_required, 2)
        }

    def update_equity(self, new_equity: float) -> None:
        """
        Update current equity and trigger drawdown recalculation.

        Args:
            new_equity: Current account equity
        """
        old_equity = self.current_equity
        self.current_equity = new_equity

        # Update peak
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity

        # Update drawdown tracker
        self.drawdown_tracker.update_equity(new_equity)

        # Log significant changes
        change_pct = ((new_equity - old_equity) / old_equity * 100) if old_equity > 0 else 0
        if abs(change_pct) >= 1.0:
            _logger.info(
                f"Equity updated: ${old_equity:.2f} → ${new_equity:.2f} "
                f"({change_pct:+.1f}%)"
            )

    def record_trade(self, trade: TradeRecord) -> None:
        """
        Record a completed trade for statistics.

        Args:
            trade: TradeRecord with trade details
        """
        self.trade_history.append(trade)

        # Update open positions
        if trade.symbol in self.open_positions:
            del self.open_positions[trade.symbol]

        # Invalidate Kelly cache
        self._cached_kelly = None

        _logger.debug(
            f"Trade recorded: {trade.symbol} {trade.direction} "
            f"R={trade.r_multiple:.2f} PnL=${trade.pnl:.2f}"
        )

    def register_open_position(
        self,
        symbol: str,
        direction: str,
        risk_amount: float,
        entry_price: float
    ) -> None:
        """
        Register an open position for tracking.

        Args:
            symbol: Trading symbol
            direction: LONG or SHORT
            risk_amount: Amount at risk
            entry_price: Entry price
        """
        self.open_positions[symbol] = {
            "direction": direction.upper(),
            "risk_amount": risk_amount,
            "entry_price": entry_price,
            "entry_time": datetime.now()
        }

        self.correlation_manager.register_position(
            symbol=symbol,
            direction=direction,
            size=risk_amount
        )

    def close_position(self, symbol: str) -> None:
        """Remove a closed position from tracking."""
        if symbol in self.open_positions:
            del self.open_positions[symbol]

        self.correlation_manager.close_position(symbol)

    def get_expectancy(self) -> Dict:
        """
        Calculate E[R] - Expected R-Multiple per trade.

        Returns:
            Dict with expectancy, win_rate, sample_size, etc.
        """
        if not self.trade_history:
            return {
                "expectancy": 0.0,
                "win_rate": 0.0,
                "avg_win_r": 0.0,
                "avg_loss_r": 0.0,
                "sample_size": 0,
                "profitable": False
            }

        r_multiples = [t.r_multiple for t in self.trade_history]

        expectancy = mean(r_multiples)

        wins = [r for r in r_multiples if r > 0]
        losses = [r for r in r_multiples if r < 0]

        win_rate = len(wins) / len(r_multiples)
        avg_win = mean(wins) if wins else 0
        avg_loss = abs(mean(losses)) if losses else 0

        return {
            "expectancy": round(expectancy, 4),
            "win_rate": round(win_rate, 4),
            "avg_win_r": round(avg_win, 4),
            "avg_loss_r": round(avg_loss, 4),
            "sample_size": len(self.trade_history),
            "profitable": expectancy > 0
        }

    def get_rolling_expectancy(self, window: int = None) -> Dict:
        """
        Calculate rolling E[R] to detect edge degradation.

        Args:
            window: Number of recent trades to consider

        Returns:
            Dict with recent_expectancy, all_time_expectancy, status, etc.
        """
        window = window or self.config["rolling_window"]

        if len(self.trade_history) < window:
            return {
                "has_edge": None,
                "message": f"Insufficient data ({len(self.trade_history)}/{window} trades)"
            }

        # Recent trades
        recent_r = [t.r_multiple for t in self.trade_history[-window:]]
        recent_exp = mean(recent_r)

        # All-time
        all_r = [t.r_multiple for t in self.trade_history]
        all_time_exp = mean(all_r)

        # Edge ratio
        edge_ratio = recent_exp / all_time_exp if all_time_exp != 0 else 0

        # Determine status
        if recent_exp <= 0:
            status = "NO_EDGE"
            action = "stop_trading"
        elif edge_ratio < self.config["edge_degradation_threshold"]:
            status = "DEGRADED"
            action = "reduce_size"
        elif edge_ratio < 0.8:
            status = "WEAKENING"
            action = "monitor"
        else:
            status = "HEALTHY"
            action = "continue"

        return {
            "recent_expectancy": round(recent_exp, 4),
            "all_time_expectancy": round(all_time_exp, 4),
            "edge_ratio": round(edge_ratio, 4),
            "status": status,
            "recommended_action": action,
            "window": window
        }

    def get_portfolio_status(self) -> Dict:
        """
        Get comprehensive portfolio risk status.

        Returns:
            Dict with drawdown, portfolio risk, Kelly state, etc.
        """
        dd_state = self.drawdown_tracker.get_state()
        portfolio_risk = calculate_portfolio_risk(
            self.open_positions,
            self.current_equity
        )
        expectancy = self.get_expectancy()

        return {
            "equity": {
                "initial": self.initial_equity,
                "current": self.current_equity,
                "peak": self.peak_equity,
                "pnl": round(self.current_equity - self.initial_equity, 2),
                "pnl_pct": round(
                    (self.current_equity - self.initial_equity)
                    / self.initial_equity * 100, 2
                )
            },
            "drawdown": {
                "current_pct": round(dd_state.drawdown_percent, 2),
                "amount": round(dd_state.drawdown_amount, 2),
                "status": dd_state.status.value,
                "kelly_multiplier": round(dd_state.kelly_multiplier, 2),
                "can_trade": dd_state.can_trade
            },
            "portfolio_risk": portfolio_risk,
            "expectancy": expectancy,
            "kelly": {
                "mode": self.config["kelly_mode"],
                "base_fraction": round(self._calculate_base_kelly(), 4),
                "trade_count": len(self.trade_history)
            },
            "positions": {
                "count": len(self.open_positions),
                "symbols": list(self.open_positions.keys())
            }
        }

    def can_trade(self) -> Tuple[bool, str]:
        """
        Quick check if trading is allowed.

        Returns:
            (can_trade: bool, reason: str)
        """
        dd_state = self.drawdown_tracker.get_state()
        if not dd_state.can_trade:
            return False, dd_state.message

        expectancy = self.get_expectancy()
        if (
            expectancy["sample_size"] >= self.config["min_trades_for_kelly"]
            and expectancy["expectancy"] < self.config["min_expectancy_to_trade"]
        ):
            return False, f"Edge too low: E[R]={expectancy['expectancy']:.3f}"

        return True, "OK"

    def reset(self, new_equity: float = None) -> None:
        """
        Reset risk manager state.

        Args:
            new_equity: New initial equity (uses current if not provided)
        """
        equity = new_equity or self.current_equity

        self.initial_equity = equity
        self.current_equity = equity
        self.peak_equity = equity

        self.drawdown_tracker = DrawdownTracker(
            initial_equity=equity,
            max_drawdown_percent=self.config["max_drawdown"]
        )

        self.trade_history.clear()
        self.open_positions.clear()
        self.correlation_manager.clear_all_positions()

        _logger.info(f"RiskManager reset with equity=${equity}")

    def get_statistics(self) -> Dict:
        """
        Get comprehensive trading statistics.

        Returns:
            Dict with performance metrics
        """
        if not self.trade_history:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "expectancy": 0.0,
                "max_drawdown": 0.0
            }

        trades = self.trade_history
        pnl_list = [t.pnl for t in trades]
        r_multiples = [t.r_multiple for t in trades]

        wins = [t for t in trades if t.r_multiple > 0]
        losses = [t for t in trades if t.r_multiple < 0]

        # Calculate drawdown statistics
        dd_stats = self.drawdown_tracker.get_statistics()

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(len(wins) / len(trades), 4) if trades else 0,
            "total_pnl": round(sum(pnl_list), 2),
            "avg_pnl": round(mean(pnl_list), 2) if pnl_list else 0,
            "avg_r_multiple": round(mean(r_multiples), 4) if r_multiples else 0,
            "expectancy": round(mean(r_multiples), 4) if r_multiples else 0,
            "best_trade_r": round(max(r_multiples), 4) if r_multiples else 0,
            "worst_trade_r": round(min(r_multiples), 4) if r_multiples else 0,
            "max_drawdown_seen": round(dd_stats["max_drawdown_seen"], 2),
            "circuit_breaker_count": dd_stats["circuit_breaker_count"],
            "current_kelly": round(self._calculate_base_kelly(), 4)
        }


# ============== CONVENIENCE FUNCTIONS ==============

def calculate_r_multiple(
    pnl: float,
    risk_amount: float
) -> float:
    """
    Calculate R-Multiple for a completed trade.

    R = PnL / Risk

    Args:
        pnl: Profit/Loss amount
        risk_amount: Initial risk amount

    Returns:
        R-multiple (positive for wins, negative for losses)

    Example:
        >>> calculate_r_multiple(100, 50)
        2.0  # Won 2R
        >>> calculate_r_multiple(-50, 50)
        -1.0  # Lost 1R (full stop)
    """
    if risk_amount <= 0:
        return 0.0

    return pnl / risk_amount


def calculate_trade_rr(
    entry_price: float,
    take_profit: float,
    stop_loss: float,
    trade_type: str
) -> float:
    """
    Calculate R:R ratio for a specific trade setup.

    Args:
        entry_price: Entry price
        take_profit: Take profit price
        stop_loss: Stop loss price
        trade_type: "LONG" or "SHORT"

    Returns:
        Reward-to-Risk ratio
    """
    if trade_type.upper() == "LONG":
        reward_distance = take_profit - entry_price
        risk_distance = entry_price - stop_loss
    else:  # SHORT
        reward_distance = entry_price - take_profit
        risk_distance = stop_loss - entry_price

    if risk_distance <= 0:
        return 0.0

    return reward_distance / risk_distance
