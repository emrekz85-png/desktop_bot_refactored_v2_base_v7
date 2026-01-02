"""
Correlation Management Module - Priority 4 Implementation

This module addresses the critical finding from the hedge fund due diligence:
- BTC/ETH/LINK correlation: 0.85-0.95 typically
- 3 positions = only 1.07 effective positions (essentially 1 concentrated bet)

Key Features:
1. Position size reduction when correlated signals align
2. Max positions per direction limit (default: 2)
3. Effective position calculation
4. Correlation-adjusted risk management

Mathematical Foundation:
    Effective Positions = N / (1 + (N-1) × avg_correlation)

    Where:
    - N = number of open positions
    - avg_correlation = average pairwise correlation

    Example: 3 positions with 0.90 correlation
    = 3 / (1 + 2 × 0.90)
    = 3 / 2.80
    = 1.07 effective positions

Usage:
    from core.correlation_manager import CorrelationManager

    cm = CorrelationManager()

    # Check before opening new position
    can_open, adj_size, reason = cm.check_new_position(
        symbol="ETHUSDT",
        direction="LONG",
        base_position_size=35.0,
        open_positions={"BTCUSDT": {"direction": "LONG", "size": 35.0}}
    )

    if can_open:
        # Use adj_size instead of base_position_size
        open_trade(symbol, direction, adj_size)
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum
import numpy as np

from .logging_config import get_logger

_logger = get_logger(__name__)


class PositionDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class CorrelationCheckResult:
    """Result of correlation check for new position."""
    can_open: bool
    adjusted_size: float
    original_size: float
    size_multiplier: float
    reason: str
    effective_positions: float
    same_direction_count: int
    correlation_with_open: float


# Pre-defined correlation matrix for major crypto pairs
# Based on historical data (typically 0.85-0.95 for major pairs)
DEFAULT_CORRELATION_MATRIX = {
    ("BTCUSDT", "ETHUSDT"): 0.92,
    ("BTCUSDT", "LINKUSDT"): 0.85,
    ("BTCUSDT", "SOLUSDT"): 0.88,
    ("BTCUSDT", "BNBUSDT"): 0.82,
    ("BTCUSDT", "XRPUSDT"): 0.78,
    ("BTCUSDT", "DOGEUSDT"): 0.75,
    ("BTCUSDT", "LTCUSDT"): 0.80,
    ("ETHUSDT", "LINKUSDT"): 0.88,
    ("ETHUSDT", "SOLUSDT"): 0.90,
    ("ETHUSDT", "BNBUSDT"): 0.80,
    ("ETHUSDT", "XRPUSDT"): 0.75,
    ("ETHUSDT", "DOGEUSDT"): 0.72,
    ("ETHUSDT", "LTCUSDT"): 0.78,
    ("LINKUSDT", "SOLUSDT"): 0.82,
    ("LINKUSDT", "BNBUSDT"): 0.75,
    ("SOLUSDT", "BNBUSDT"): 0.78,
}


class CorrelationManager:
    """
    Manages position correlation and risk concentration.

    Prevents over-concentration in highly correlated assets by:
    1. Limiting max positions in same direction
    2. Reducing position size based on correlation
    3. Tracking effective portfolio diversification
    """

    def __init__(
        self,
        max_positions_same_direction: int = 2,
        max_total_positions: int = 4,
        high_correlation_threshold: float = 0.80,
        position_reduction_factor: float = 0.50,
        correlation_matrix: Dict[Tuple[str, str], float] = None,
        enable_size_reduction: bool = True,
        enable_direction_limit: bool = True,
    ):
        """
        Initialize correlation manager.

        Args:
            max_positions_same_direction: Max positions allowed in LONG or SHORT (default: 2)
            max_total_positions: Max total open positions (default: 4)
            high_correlation_threshold: Correlation above this triggers size reduction (default: 0.80)
            position_reduction_factor: Multiply position size by this when correlated (default: 0.50)
            correlation_matrix: Custom correlation matrix (default: built-in crypto correlations)
            enable_size_reduction: Enable position size reduction for correlated assets
            enable_direction_limit: Enable max positions per direction limit
        """
        self.max_positions_same_direction = max_positions_same_direction
        self.max_total_positions = max_total_positions
        self.high_correlation_threshold = high_correlation_threshold
        self.position_reduction_factor = position_reduction_factor
        self.correlation_matrix = correlation_matrix or DEFAULT_CORRELATION_MATRIX
        self.enable_size_reduction = enable_size_reduction
        self.enable_direction_limit = enable_direction_limit

        # Track open positions
        self._open_positions: Dict[str, Dict] = {}

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Get correlation between two symbols.

        Args:
            symbol1: First symbol
            symbol2: Second symbol

        Returns:
            Correlation coefficient (0.0 to 1.0)
        """
        if symbol1 == symbol2:
            return 1.0

        # Normalize symbols
        s1, s2 = symbol1.upper(), symbol2.upper()

        # Check both orderings
        if (s1, s2) in self.correlation_matrix:
            return self.correlation_matrix[(s1, s2)]
        if (s2, s1) in self.correlation_matrix:
            return self.correlation_matrix[(s2, s1)]

        # Default correlation for unknown pairs
        # Conservative assumption: moderate correlation
        return 0.70

    def calculate_effective_positions(
        self,
        positions: Dict[str, Dict],
    ) -> float:
        """
        Calculate effective number of positions accounting for correlation.

        Formula: N / (1 + (N-1) × avg_correlation)

        Args:
            positions: Dict of {symbol: {"direction": str, "size": float}}

        Returns:
            Effective position count (always <= actual count)
        """
        if not positions:
            return 0.0

        n = len(positions)
        if n == 1:
            return 1.0

        # Calculate average pairwise correlation
        symbols = list(positions.keys())
        correlations = []

        for i, s1 in enumerate(symbols):
            for s2 in symbols[i+1:]:
                # Only count correlation if same direction
                if positions[s1].get("direction") == positions[s2].get("direction"):
                    corr = self.get_correlation(s1, s2)
                    correlations.append(corr)
                else:
                    # Opposite directions reduce effective correlation
                    # (they hedge each other somewhat)
                    corr = self.get_correlation(s1, s2)
                    correlations.append(-corr * 0.5)  # Partial hedge effect

        if not correlations:
            return float(n)

        avg_correlation = np.mean(correlations)

        # Effective positions formula
        effective = n / (1 + (n - 1) * max(avg_correlation, 0))

        return max(1.0, effective)

    def check_new_position(
        self,
        symbol: str,
        direction: str,
        base_position_size: float,
        open_positions: Dict[str, Dict] = None,
    ) -> CorrelationCheckResult:
        """
        Check if a new position can be opened and calculate adjusted size.

        Args:
            symbol: Symbol to open position for
            direction: "LONG" or "SHORT"
            base_position_size: Original position size (before adjustment)
            open_positions: Current open positions {symbol: {"direction": str, "size": float}}
                           If None, uses internal tracking

        Returns:
            CorrelationCheckResult with decision and adjusted size
        """
        positions = open_positions if open_positions is not None else self._open_positions

        # Count positions by direction
        same_direction_count = sum(
            1 for p in positions.values()
            if p.get("direction", "").upper() == direction.upper()
        )

        total_positions = len(positions)

        # Check 1: Max positions per direction
        if self.enable_direction_limit:
            if same_direction_count >= self.max_positions_same_direction:
                return CorrelationCheckResult(
                    can_open=False,
                    adjusted_size=0.0,
                    original_size=base_position_size,
                    size_multiplier=0.0,
                    reason=f"Max {direction} positions reached ({same_direction_count}/{self.max_positions_same_direction})",
                    effective_positions=self.calculate_effective_positions(positions),
                    same_direction_count=same_direction_count,
                    correlation_with_open=0.0,
                )

        # Check 2: Max total positions
        if total_positions >= self.max_total_positions:
            return CorrelationCheckResult(
                can_open=False,
                adjusted_size=0.0,
                original_size=base_position_size,
                size_multiplier=0.0,
                reason=f"Max total positions reached ({total_positions}/{self.max_total_positions})",
                effective_positions=self.calculate_effective_positions(positions),
                same_direction_count=same_direction_count,
                correlation_with_open=0.0,
            )

        # Check 3: Calculate correlation with existing positions
        max_correlation = 0.0
        correlated_symbols = []

        for existing_symbol, pos_info in positions.items():
            if pos_info.get("direction", "").upper() == direction.upper():
                corr = self.get_correlation(symbol, existing_symbol)
                if corr > max_correlation:
                    max_correlation = corr
                if corr >= self.high_correlation_threshold:
                    correlated_symbols.append((existing_symbol, corr))

        # Calculate size adjustment
        size_multiplier = 1.0

        if self.enable_size_reduction and max_correlation >= self.high_correlation_threshold:
            # Reduce position size based on number of correlated positions
            num_correlated = len(correlated_symbols)

            # Progressive reduction: more correlated positions = smaller size
            # 1 correlated: 50% size
            # 2+ correlated: 33% size
            if num_correlated >= 2:
                size_multiplier = self.position_reduction_factor * 0.67  # ~33%
            else:
                size_multiplier = self.position_reduction_factor  # 50%

        adjusted_size = base_position_size * size_multiplier

        # Calculate effective positions including new one
        test_positions = dict(positions)
        test_positions[symbol] = {"direction": direction, "size": adjusted_size}
        effective_positions = self.calculate_effective_positions(test_positions)

        # Build reason
        if size_multiplier < 1.0:
            corr_str = ", ".join([f"{s}({c:.2f})" for s, c in correlated_symbols])
            reason = f"Size reduced {size_multiplier:.0%} due to correlation with: {corr_str}"
        else:
            reason = "OK - No significant correlation"

        return CorrelationCheckResult(
            can_open=True,
            adjusted_size=adjusted_size,
            original_size=base_position_size,
            size_multiplier=size_multiplier,
            reason=reason,
            effective_positions=effective_positions,
            same_direction_count=same_direction_count + 1,  # Including new position
            correlation_with_open=max_correlation,
        )

    def register_position(
        self,
        symbol: str,
        direction: str,
        size: float,
    ) -> None:
        """Register an opened position for tracking."""
        self._open_positions[symbol] = {
            "direction": direction.upper(),
            "size": size,
        }
        _logger.debug(f"Registered position: {symbol} {direction} size={size}")

    def close_position(self, symbol: str) -> None:
        """Remove a closed position from tracking."""
        if symbol in self._open_positions:
            del self._open_positions[symbol]
            _logger.debug(f"Closed position: {symbol}")

    def get_open_positions(self) -> Dict[str, Dict]:
        """Get current open positions."""
        return dict(self._open_positions)

    def get_portfolio_summary(self) -> Dict:
        """
        Get summary of current portfolio state.

        Returns:
            Dict with portfolio metrics
        """
        positions = self._open_positions

        if not positions:
            return {
                "total_positions": 0,
                "long_positions": 0,
                "short_positions": 0,
                "effective_positions": 0.0,
                "concentration_ratio": 0.0,
                "symbols": [],
            }

        long_count = sum(1 for p in positions.values() if p["direction"] == "LONG")
        short_count = sum(1 for p in positions.values() if p["direction"] == "SHORT")
        effective = self.calculate_effective_positions(positions)

        # Concentration ratio: how concentrated vs diversified
        # 1.0 = perfectly diversified, lower = more concentrated
        concentration_ratio = effective / len(positions) if positions else 1.0

        return {
            "total_positions": len(positions),
            "long_positions": long_count,
            "short_positions": short_count,
            "effective_positions": round(effective, 2),
            "concentration_ratio": round(concentration_ratio, 2),
            "symbols": list(positions.keys()),
            "positions": positions,
        }

    def suggest_hedge(
        self,
        symbol: str,
        direction: str,
    ) -> Optional[Dict]:
        """
        Suggest a hedge position for the given trade.

        This is a simple inverse correlation hedge suggestion.

        Args:
            symbol: Symbol being traded
            direction: Direction of trade

        Returns:
            Dict with hedge suggestion or None
        """
        # Find lowest correlated symbol
        all_symbols = set()
        for pair in self.correlation_matrix.keys():
            all_symbols.add(pair[0])
            all_symbols.add(pair[1])

        # Remove the symbol itself
        candidates = [s for s in all_symbols if s != symbol]

        if not candidates:
            return None

        # Find lowest correlation
        min_corr = 1.0
        hedge_symbol = None

        for candidate in candidates:
            corr = self.get_correlation(symbol, candidate)
            if corr < min_corr:
                min_corr = corr
                hedge_symbol = candidate

        if hedge_symbol and min_corr < 0.85:
            hedge_direction = "SHORT" if direction.upper() == "LONG" else "LONG"
            return {
                "symbol": hedge_symbol,
                "direction": hedge_direction,
                "correlation": min_corr,
                "reason": f"Lowest correlation ({min_corr:.2f}) hedge for {symbol} {direction}",
            }

        return None

    def clear_all_positions(self) -> None:
        """Clear all tracked positions."""
        self._open_positions.clear()


# ============== CONVENIENCE FUNCTIONS ==============

def check_correlation_risk(
    new_symbol: str,
    new_direction: str,
    open_positions: Dict[str, Dict],
    max_same_direction: int = 2,
) -> Tuple[bool, str, float]:
    """
    Quick check if new position violates correlation rules.

    Args:
        new_symbol: Symbol to open
        new_direction: LONG or SHORT
        open_positions: Current open positions
        max_same_direction: Max allowed in same direction

    Returns:
        (can_open, reason, size_multiplier)
    """
    cm = CorrelationManager(max_positions_same_direction=max_same_direction)
    result = cm.check_new_position(
        symbol=new_symbol,
        direction=new_direction,
        base_position_size=1.0,  # Normalized
        open_positions=open_positions,
    )
    return result.can_open, result.reason, result.size_multiplier


def calculate_portfolio_effective_positions(positions: Dict[str, Dict]) -> float:
    """
    Calculate effective positions for a portfolio.

    Args:
        positions: Dict of {symbol: {"direction": str}}

    Returns:
        Effective position count
    """
    cm = CorrelationManager()
    return cm.calculate_effective_positions(positions)


# ============== CORRELATION MATRIX UTILITIES ==============

def update_correlation_matrix(
    symbol1: str,
    symbol2: str,
    correlation: float,
) -> None:
    """
    Update the default correlation matrix with new value.

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        correlation: New correlation value (0.0 to 1.0)
    """
    s1, s2 = sorted([symbol1.upper(), symbol2.upper()])
    DEFAULT_CORRELATION_MATRIX[(s1, s2)] = correlation


def get_all_correlations() -> Dict[Tuple[str, str], float]:
    """Get the current correlation matrix."""
    return dict(DEFAULT_CORRELATION_MATRIX)


# ============== KELLY ADJUSTMENT INTEGRATION ==============

def adjust_kelly_for_correlation(
    base_kelly: float,
    open_positions: Dict[str, Dict],
    new_position_symbol: str,
    new_position_direction: str,
    max_same_direction: int = 3,
    correlation_matrix: Dict[Tuple[str, str], float] = None
) -> Dict:
    """
    Adjust Kelly fraction based on existing portfolio correlation.

    This integrates with the Kelly-based position sizing system from
    core/kelly_calculator.py to provide correlation-aware risk management.

    Args:
        base_kelly: Base Kelly fraction (e.g., 0.02 for 2%)
        open_positions: Dict of {symbol: {"direction": str, ...}}
        new_position_symbol: Symbol for new position
        new_position_direction: "LONG" or "SHORT"
        max_same_direction: Maximum positions allowed in same direction
        correlation_matrix: Optional custom correlation matrix

    Returns:
        Dict with adjusted_kelly, adjustment_factor, reason, etc.

    Example:
        >>> positions = {"BTCUSDT": {"direction": "LONG"}}
        >>> result = adjust_kelly_for_correlation(0.02, positions, "ETHUSDT", "LONG")
        >>> result['adjusted_kelly']
        0.014  # Reduced due to BTC-ETH correlation (0.92)
    """
    cm = CorrelationManager(
        max_positions_same_direction=max_same_direction,
        correlation_matrix=correlation_matrix
    )

    # No existing positions - no adjustment needed
    if not open_positions:
        return {
            "adjusted_kelly": base_kelly,
            "adjustment_factor": 1.0,
            "reason": "no_existing_positions",
            "can_trade": True,
            "n_same_direction": 0,
            "avg_correlation": 0.0
        }

    # Count same-direction positions
    same_direction = [
        symbol for symbol, pos in open_positions.items()
        if pos.get("direction", "").upper() == new_position_direction.upper()
    ]

    # Rule: Max positions in same direction
    if len(same_direction) >= max_same_direction:
        return {
            "adjusted_kelly": 0.0,
            "adjustment_factor": 0.0,
            "reason": f"max_same_direction_reached ({len(same_direction)}/{max_same_direction})",
            "can_trade": False,
            "n_same_direction": len(same_direction),
            "avg_correlation": 0.0
        }

    # Calculate average correlation with existing same-direction positions
    if not same_direction:
        return {
            "adjusted_kelly": base_kelly,
            "adjustment_factor": 1.0,
            "reason": "no_same_direction_positions",
            "can_trade": True,
            "n_same_direction": 0,
            "avg_correlation": 0.0
        }

    correlations = [
        cm.get_correlation(new_position_symbol, symbol)
        for symbol in same_direction
    ]
    avg_corr = np.mean(correlations)

    # Adjustment formula from RISK_MANAGEMENT_SPEC.md:
    # High correlation (0.9) → 0.55 factor
    # Medium correlation (0.5) → 0.85 factor
    # Low correlation (0.2) → 0.95 factor
    # Formula: adjustment = 1 - (avg_corr * 0.5)
    adjustment_factor = 1 - (avg_corr * 0.5)

    adjusted_kelly = base_kelly * adjustment_factor

    return {
        "adjusted_kelly": adjusted_kelly,
        "adjustment_factor": round(adjustment_factor, 4),
        "avg_correlation": round(avg_corr, 4),
        "n_same_direction": len(same_direction),
        "correlated_symbols": same_direction,
        "reason": "correlation_adjusted",
        "can_trade": True
    }


def calculate_portfolio_risk(
    open_positions: Dict[str, Dict],
    equity: float,
    correlation_matrix: Dict[Tuple[str, str], float] = None
) -> Dict:
    """
    Calculate total portfolio risk considering correlations.

    This provides portfolio-level risk metrics for the RiskManager.

    Args:
        open_positions: Dict of {symbol: {"direction": str, "risk_amount": float}}
        equity: Current account equity
        correlation_matrix: Optional custom correlation matrix

    Returns:
        Dict with total_risk_percent, effective_risk_percent, etc.

    Example:
        >>> positions = {
        ...     "BTCUSDT": {"direction": "LONG", "risk_amount": 100},
        ...     "ETHUSDT": {"direction": "LONG", "risk_amount": 100}
        ... }
        >>> result = calculate_portfolio_risk(positions, 10000)
        >>> result['effective_risk_percent']
        1.5  # Less than 2% due to diversification
    """
    if not open_positions or equity <= 0:
        return {
            "total_risk_percent": 0.0,
            "effective_risk_percent": 0.0,
            "n_positions": 0,
            "n_effective": 0.0,
            "diversification_ratio": 1.0
        }

    cm = CorrelationManager(correlation_matrix=correlation_matrix)

    # Sum of individual risks
    total_risk = sum(
        pos.get("risk_amount", 0) for pos in open_positions.values()
    )
    total_risk_percent = (total_risk / equity) * 100

    # Effective positions
    n_effective = cm.calculate_effective_positions(open_positions)
    n_actual = len(open_positions)

    # Diversification benefit
    diversification_ratio = n_effective / n_actual if n_actual > 0 else 1.0

    # Effective risk (accounting for diversification)
    # Using sqrt for portfolio variance reduction effect
    effective_risk = total_risk * np.sqrt(diversification_ratio)
    effective_risk_percent = (effective_risk / equity) * 100

    return {
        "total_risk_percent": round(total_risk_percent, 2),
        "effective_risk_percent": round(effective_risk_percent, 2),
        "total_risk_amount": round(total_risk, 2),
        "effective_risk_amount": round(effective_risk, 2),
        "n_positions": n_actual,
        "n_effective": round(n_effective, 2),
        "diversification_ratio": round(diversification_ratio, 4),
        "diversification_benefit": round((1 - diversification_ratio) * 100, 1)  # % benefit
    }
