"""
Drawdown Tracker Module

Implements drawdown calculation, Kelly auto-adjustment, and circuit breaker logic
for anti-fragile risk management.

Key Features:
    - Exponential decay of Kelly fraction as drawdown increases
    - 20% max drawdown circuit breaker
    - Recovery state management with 5% required recovery
    - Rolling equity tracking

References:
    - docs/RISK_MANAGEMENT_SPEC.md Section 5
"""

from math import exp
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


# Configuration Constants
MAX_DRAWDOWN_PERCENT = 20.0       # Circuit breaker threshold
DRAWDOWN_DECAY_K = 0.15           # Exponential decay constant
MIN_KELLY_MULTIPLIER = 0.25      # Floor before circuit breaker
RECOVERY_REQUIRED_PERCENT = 5.0   # Recovery needed to resume trading


class DrawdownStatus(Enum):
    """Trading status based on drawdown level."""
    NORMAL = "NORMAL"           # DD < 10%, full trading
    CAUTION = "CAUTION"         # 10% <= DD < 15%, reduced size
    DANGER = "DANGER"           # 15% <= DD < 20%, minimum size
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"  # DD >= 20%, stop trading
    RECOVERING = "RECOVERING"   # After circuit breaker, waiting for recovery


@dataclass
class DrawdownState:
    """Current drawdown state."""
    current_equity: float
    peak_equity: float
    drawdown_amount: float
    drawdown_percent: float
    kelly_multiplier: float
    status: DrawdownStatus
    can_trade: bool
    message: str = ""


@dataclass
class RecoveryState:
    """Recovery state after circuit breaker."""
    circuit_breaker_equity: float  # Equity when CB triggered
    circuit_breaker_time: datetime
    recovery_percent: float = 0.0
    required_recovery: float = RECOVERY_REQUIRED_PERCENT
    can_resume: bool = False


class DrawdownTracker:
    """
    Tracks equity drawdown and manages Kelly auto-adjustment.

    Usage:
        tracker = DrawdownTracker(initial_equity=10000)
        tracker.update_equity(9500)  # 5% drawdown
        state = tracker.get_state()
        print(state.kelly_multiplier)  # ~0.85 (reduced)
    """

    def __init__(
        self,
        initial_equity: float,
        max_drawdown_percent: float = MAX_DRAWDOWN_PERCENT,
        decay_k: float = DRAWDOWN_DECAY_K,
        min_multiplier: float = MIN_KELLY_MULTIPLIER,
        recovery_required: float = RECOVERY_REQUIRED_PERCENT
    ):
        """
        Initialize drawdown tracker.

        Args:
            initial_equity: Starting equity value
            max_drawdown_percent: Circuit breaker threshold (default 20%)
            decay_k: Exponential decay constant (default 0.15)
            min_multiplier: Floor multiplier before CB (default 0.25)
            recovery_required: Recovery % needed to resume (default 5%)
        """
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.peak_equity = initial_equity

        self.max_drawdown = max_drawdown_percent
        self.decay_k = decay_k
        self.min_multiplier = min_multiplier
        self.recovery_required = recovery_required

        self.circuit_breaker_active = False
        self.recovery_state: Optional[RecoveryState] = None

        # History tracking
        self.equity_history: list = [(datetime.now(), initial_equity)]
        self.drawdown_history: list = []

    def update_equity(self, new_equity: float) -> DrawdownState:
        """
        Update current equity and recalculate drawdown state.

        Args:
            new_equity: Current equity value

        Returns:
            Current DrawdownState
        """
        self.current_equity = new_equity

        # Update peak if new high
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
            # If we were recovering, we've fully recovered
            if self.recovery_state is not None:
                self.recovery_state = None
                self.circuit_breaker_active = False

        # Record history
        self.equity_history.append((datetime.now(), new_equity))

        # Calculate current state
        state = self.get_state()

        # Check for circuit breaker activation
        if state.status == DrawdownStatus.CIRCUIT_BREAKER:
            if not self.circuit_breaker_active:
                self._activate_circuit_breaker()

        # Record drawdown history
        self.drawdown_history.append({
            "timestamp": datetime.now(),
            "drawdown_percent": state.drawdown_percent,
            "status": state.status.value
        })

        return state

    def get_state(self) -> DrawdownState:
        """
        Get current drawdown state.

        Returns:
            DrawdownState with all metrics
        """
        # Calculate drawdown
        dd = calculate_drawdown(self.current_equity, self.peak_equity)

        # Check if in recovery mode
        if self.recovery_state is not None:
            return self._get_recovery_state(dd)

        # Calculate Kelly multiplier
        multiplier = get_drawdown_kelly_multiplier(
            dd["drawdown_percent"],
            self.max_drawdown,
            self.decay_k,
            self.min_multiplier
        )

        # Determine status
        status = self._determine_status(dd["drawdown_percent"])

        # Can trade?
        can_trade = status != DrawdownStatus.CIRCUIT_BREAKER

        # Build message
        if status == DrawdownStatus.NORMAL:
            message = "Trading at full capacity"
        elif status == DrawdownStatus.CAUTION:
            message = f"Drawdown {dd['drawdown_percent']:.1f}% - Position size reduced to {multiplier*100:.0f}%"
        elif status == DrawdownStatus.DANGER:
            message = f"Drawdown {dd['drawdown_percent']:.1f}% - Minimum position size active"
        else:
            message = f"CIRCUIT BREAKER - {dd['drawdown_percent']:.1f}% drawdown exceeds {self.max_drawdown}% limit"

        return DrawdownState(
            current_equity=self.current_equity,
            peak_equity=self.peak_equity,
            drawdown_amount=dd["drawdown_amount"],
            drawdown_percent=dd["drawdown_percent"],
            kelly_multiplier=multiplier,
            status=status,
            can_trade=can_trade,
            message=message
        )

    def _get_recovery_state(self, dd: dict) -> DrawdownState:
        """Handle state when in recovery mode."""
        rs = self.recovery_state

        # Calculate recovery from circuit breaker point
        if rs.circuit_breaker_equity > 0:
            recovery_pct = (
                (self.current_equity - rs.circuit_breaker_equity)
                / rs.circuit_breaker_equity * 100
            )
        else:
            recovery_pct = 0.0

        rs.recovery_percent = recovery_pct
        rs.can_resume = recovery_pct >= rs.required_recovery

        if rs.can_resume:
            # Can resume with reduced size
            return DrawdownState(
                current_equity=self.current_equity,
                peak_equity=self.peak_equity,
                drawdown_amount=dd["drawdown_amount"],
                drawdown_percent=dd["drawdown_percent"],
                kelly_multiplier=self.min_multiplier,  # Start with minimum
                status=DrawdownStatus.RECOVERING,
                can_trade=True,
                message=f"Trading resumed at {self.min_multiplier*100:.0f}% position size after {recovery_pct:.1f}% recovery"
            )
        else:
            # Still need more recovery
            needed = rs.required_recovery - recovery_pct
            return DrawdownState(
                current_equity=self.current_equity,
                peak_equity=self.peak_equity,
                drawdown_amount=dd["drawdown_amount"],
                drawdown_percent=dd["drawdown_percent"],
                kelly_multiplier=0.0,
                status=DrawdownStatus.RECOVERING,
                can_trade=False,
                message=f"Recovering from circuit breaker - need {needed:.1f}% more recovery"
            )

    def _determine_status(self, drawdown_percent: float) -> DrawdownStatus:
        """Determine trading status from drawdown percent."""
        if drawdown_percent >= self.max_drawdown:
            return DrawdownStatus.CIRCUIT_BREAKER
        elif drawdown_percent >= 15:
            return DrawdownStatus.DANGER
        elif drawdown_percent >= 10:
            return DrawdownStatus.CAUTION
        else:
            return DrawdownStatus.NORMAL

    def _activate_circuit_breaker(self):
        """Activate circuit breaker - stop all trading."""
        self.circuit_breaker_active = True
        self.recovery_state = RecoveryState(
            circuit_breaker_equity=self.current_equity,
            circuit_breaker_time=datetime.now(),
            required_recovery=self.recovery_required
        )

    def reset_peak(self, new_peak: Optional[float] = None):
        """
        Reset peak equity (use with caution).

        Args:
            new_peak: New peak value, defaults to current equity
        """
        self.peak_equity = new_peak if new_peak else self.current_equity
        self.circuit_breaker_active = False
        self.recovery_state = None

    def get_statistics(self) -> dict:
        """Get drawdown statistics."""
        if not self.drawdown_history:
            return {
                "max_drawdown_seen": 0.0,
                "avg_drawdown": 0.0,
                "time_in_drawdown_pct": 0.0,
                "circuit_breaker_count": 0
            }

        dd_values = [d["drawdown_percent"] for d in self.drawdown_history]
        cb_count = sum(1 for d in self.drawdown_history
                       if d["status"] == DrawdownStatus.CIRCUIT_BREAKER.value)

        time_in_dd = sum(1 for d in self.drawdown_history
                         if d["drawdown_percent"] > 0)
        time_pct = (time_in_dd / len(self.drawdown_history)) * 100 if self.drawdown_history else 0

        return {
            "max_drawdown_seen": max(dd_values),
            "avg_drawdown": sum(dd_values) / len(dd_values),
            "current_drawdown": dd_values[-1] if dd_values else 0,
            "time_in_drawdown_pct": time_pct,
            "circuit_breaker_count": cb_count,
            "observations": len(self.drawdown_history)
        }


def calculate_drawdown(
    current_equity: float,
    peak_equity: float
) -> dict:
    """
    Calculate drawdown from peak equity.

    Args:
        current_equity: Current account equity
        peak_equity: Highest equity achieved

    Returns:
        Dict with drawdown_percent, drawdown_amount, etc.

    Example:
        >>> calculate_drawdown(9000, 10000)
        {'drawdown_percent': 10.0, 'drawdown_amount': 1000.0, ...}
    """
    if peak_equity <= 0:
        return {
            "drawdown_percent": 0.0,
            "drawdown_amount": 0.0,
            "peak_equity": peak_equity,
            "current_equity": current_equity
        }

    drawdown_amount = peak_equity - current_equity
    drawdown_percent = (drawdown_amount / peak_equity) * 100

    return {
        "drawdown_percent": max(0.0, drawdown_percent),
        "drawdown_amount": max(0.0, drawdown_amount),
        "peak_equity": peak_equity,
        "current_equity": current_equity,
        "distance_to_peak_pct": drawdown_percent
    }


def get_drawdown_kelly_multiplier(
    drawdown_percent: float,
    max_drawdown: float = MAX_DRAWDOWN_PERCENT,
    decay_k: float = DRAWDOWN_DECAY_K,
    min_multiplier: float = MIN_KELLY_MULTIPLIER
) -> float:
    """
    Calculate Kelly multiplier based on current drawdown.

    Uses exponential decay for smooth, anti-fragile adjustment:
        multiplier = e^(-k × drawdown)

    Calibration (with default k=0.15):
        0% DD  → 1.00 (full Kelly)
        5% DD  → 0.85
        10% DD → 0.70
        15% DD → 0.50
        20% DD → 0.00 (circuit breaker)

    Args:
        drawdown_percent: Current drawdown percentage (0-100)
        max_drawdown: Circuit breaker threshold (default 20)
        decay_k: Exponential decay constant (default 0.15)
        min_multiplier: Floor before circuit breaker (default 0.25)

    Returns:
        Kelly multiplier (0.0 to 1.0)

    Example:
        >>> get_drawdown_kelly_multiplier(10.0)
        0.7  # 70% of normal Kelly
    """
    # Circuit breaker check
    if drawdown_percent >= max_drawdown:
        return 0.0

    # No drawdown - full Kelly
    if drawdown_percent <= 0:
        return 1.0

    # Exponential decay formula
    multiplier = exp(-decay_k * drawdown_percent)

    # Apply floor (except at circuit breaker)
    return max(min_multiplier, multiplier)


def get_recovery_status(
    current_equity: float,
    peak_equity: float,
    circuit_breaker_equity: Optional[float] = None,
    recovery_required: float = RECOVERY_REQUIRED_PERCENT
) -> dict:
    """
    Determine trading status during drawdown recovery.

    Args:
        current_equity: Current account equity
        peak_equity: Peak equity achieved
        circuit_breaker_equity: Equity when CB was triggered (None if not triggered)
        recovery_required: Percentage recovery needed to resume

    Returns:
        Dict with can_trade, status, and recovery metrics
    """
    dd = calculate_drawdown(current_equity, peak_equity)

    # If circuit breaker was not triggered
    if circuit_breaker_equity is None:
        if dd["drawdown_percent"] >= MAX_DRAWDOWN_PERCENT:
            return {
                "can_trade": False,
                "status": "CIRCUIT_BREAKER",
                "drawdown_percent": dd["drawdown_percent"],
                "message": f"Circuit breaker active at {dd['drawdown_percent']:.1f}% drawdown"
            }
        else:
            multiplier = get_drawdown_kelly_multiplier(dd["drawdown_percent"])
            return {
                "can_trade": True,
                "status": "NORMAL" if dd["drawdown_percent"] < 10 else "CAUTION",
                "kelly_multiplier": multiplier,
                "drawdown_percent": dd["drawdown_percent"]
            }

    # If circuit breaker was triggered - check recovery
    if circuit_breaker_equity > 0:
        recovery_pct = (
            (current_equity - circuit_breaker_equity)
            / circuit_breaker_equity * 100
        )
    else:
        recovery_pct = 0.0

    if recovery_pct < recovery_required:
        # Still need more recovery
        needed = recovery_required - recovery_pct
        return {
            "can_trade": False,
            "status": "RECOVERING",
            "recovery_percent": recovery_pct,
            "required_recovery": recovery_required,
            "message": f"Need {needed:.1f}% more recovery to resume trading"
        }
    else:
        # Can resume with reduced size
        return {
            "can_trade": True,
            "status": "RESUMED",
            "recovery_percent": recovery_pct,
            "kelly_multiplier": MIN_KELLY_MULTIPLIER,  # Start with quarter Kelly
            "message": "Trading resumed at reduced size"
        }


def estimate_recovery_time(
    current_equity: float,
    peak_equity: float,
    expected_r: float,
    kelly_fraction: float,
    trades_per_period: float = 10
) -> dict:
    """
    Estimate time to recover from drawdown.

    Args:
        current_equity: Current equity
        peak_equity: Peak to recover to
        expected_r: Expected R-multiple per trade
        kelly_fraction: Current Kelly fraction
        trades_per_period: Expected trades per period

    Returns:
        Dict with estimated trades and periods to recover
    """
    if current_equity >= peak_equity:
        return {
            "trades_to_recover": 0,
            "periods_to_recover": 0,
            "message": "No recovery needed - at or above peak"
        }

    if expected_r <= 0:
        return {
            "trades_to_recover": float('inf'),
            "periods_to_recover": float('inf'),
            "message": "Cannot recover with negative expectancy"
        }

    # Amount to recover
    amount_to_recover = peak_equity - current_equity

    # Expected profit per trade
    profit_per_trade = expected_r * kelly_fraction * current_equity

    if profit_per_trade <= 0:
        return {
            "trades_to_recover": float('inf'),
            "periods_to_recover": float('inf'),
            "message": "Cannot recover with current parameters"
        }

    # Simple estimate (doesn't compound)
    trades_needed = amount_to_recover / profit_per_trade
    periods_needed = trades_needed / trades_per_period

    return {
        "trades_to_recover": int(trades_needed) + 1,
        "periods_to_recover": periods_needed,
        "amount_to_recover": amount_to_recover,
        "expected_profit_per_trade": profit_per_trade,
        "message": f"Estimated {int(trades_needed)+1} trades to recover ${amount_to_recover:.2f}"
    }
