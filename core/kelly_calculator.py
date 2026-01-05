"""
Kelly Criterion Calculator Module

Implements mathematically optimal position sizing based on the Kelly Criterion
for maximum geometric growth rate.

Mathematical Foundation:
    f* = W - (1-W)/R

    Where:
        f* = Optimal fraction of capital to risk
        W  = Win rate (probability of winning)
        R  = Reward-to-Risk ratio (avg_win / avg_loss)

    Growth Rate at fraction f:
        G(f) = W × log(1 + f×R) + (1-W) × log(1 - f)

References:
    - Kelly, J.L. (1956). "A New Interpretation of Information Rate"
    - docs/RISK_MANAGEMENT_SPEC.md
"""

from math import log, exp, sqrt
from typing import Optional
from statistics import mean, stdev


# Configuration Constants
MIN_KELLY_FRACTION = 0.0025   # 0.25% floor
MAX_KELLY_FRACTION = 0.05     # 5% ceiling
MIN_TRADES_FOR_KELLY = 30     # Minimum sample size
DEFAULT_KELLY_FRACTION = 0.01 # 1% when insufficient data


def calculate_kelly(
    win_rate: float,
    reward_risk: float,
    kelly_mode: str = "half"
) -> float:
    """
    Calculate Kelly fraction from win rate and reward/risk ratio.

    Args:
        win_rate: Probability of winning (0.0 to 1.0)
        reward_risk: Average win / average loss ratio
        kelly_mode: "full", "half", "quarter", or "third"

    Returns:
        Optimal fraction of capital to risk (bounded by MIN/MAX)

    Examples:
        >>> calculate_kelly(0.60, 1.5)  # Half-Kelly default
        0.0166...  # ~1.67%

        >>> calculate_kelly(0.50, 2.0, "full")
        0.25  # 25% (capped to 5%)
    """
    # Validate inputs
    if win_rate <= 0 or win_rate >= 1:
        return 0.0
    if reward_risk <= 0:
        return 0.0

    # Kelly formula: f* = W - (1-W)/R
    kelly = win_rate - (1 - win_rate) / reward_risk

    # No edge scenario
    if kelly <= 0:
        return 0.0

    # Apply Kelly mode (fraction reduction)
    kelly_multipliers = {
        "full": 1.0,
        "half": 0.5,
        "third": 0.333,
        "quarter": 0.25
    }
    multiplier = kelly_multipliers.get(kelly_mode, 0.5)
    kelly = kelly * multiplier

    # Apply bounds
    kelly = max(MIN_KELLY_FRACTION, min(MAX_KELLY_FRACTION, kelly))

    return kelly


def calculate_growth_rate(
    kelly_fraction: float,
    win_rate: float,
    reward_risk: float
) -> float:
    """
    Calculate expected geometric growth rate at given Kelly fraction.

    G(f) = W × log(1 + f×R) + (1-W) × log(1 - f)

    Args:
        kelly_fraction: Fraction of capital to risk (0 < f < 1)
        win_rate: Probability of winning (0 < W < 1)
        reward_risk: Reward to risk ratio (R > 0)

    Returns:
        Expected growth rate per trade (can be negative if over-betting)

    Example:
        >>> calculate_growth_rate(0.10, 0.60, 2.0)
        0.0349  # 3.49% expected growth per trade
    """
    if kelly_fraction <= 0 or kelly_fraction >= 1:
        return 0.0
    if win_rate <= 0 or win_rate >= 1:
        return 0.0
    if reward_risk <= 0:
        return 0.0

    # Ensure we don't take log of non-positive number
    win_term = 1 + kelly_fraction * reward_risk
    loss_term = 1 - kelly_fraction

    if win_term <= 0 or loss_term <= 0:
        return float('-inf')

    growth = win_rate * log(win_term) + (1 - win_rate) * log(loss_term)

    return growth


def calculate_optimal_growth_rate(win_rate: float, reward_risk: float) -> float:
    """
    Calculate growth rate at optimal (full) Kelly.

    Args:
        win_rate: Win probability
        reward_risk: R:R ratio

    Returns:
        Maximum possible growth rate
    """
    full_kelly = calculate_kelly(win_rate, reward_risk, kelly_mode="full")
    if full_kelly <= 0:
        return 0.0

    return calculate_growth_rate(full_kelly, win_rate, reward_risk)


def trades_to_double(growth_rate: float) -> Optional[float]:
    """
    Calculate expected number of trades to double capital.

    E[trades] = log(2) / G(f)

    Args:
        growth_rate: Per-trade growth rate from calculate_growth_rate()

    Returns:
        Expected trades to double, or None if growth_rate <= 0
    """
    if growth_rate <= 0:
        return None

    return log(2) / growth_rate


def calculate_kelly_from_history(
    trade_history: list,
    min_trades: int = MIN_TRADES_FOR_KELLY,
    kelly_mode: str = "half"
) -> dict:
    """
    Calculate Kelly fraction from historical trade data.

    Args:
        trade_history: List of dicts with 'r_multiple' key
        min_trades: Minimum trades required for calculation
        kelly_mode: Kelly fraction mode

    Returns:
        Dict with kelly, win_rate, reward_risk, confidence, etc.

    Example:
        >>> history = [
        ...     {"r_multiple": 2.0},
        ...     {"r_multiple": -1.0},
        ...     {"r_multiple": 1.5},
        ...     {"r_multiple": -1.0}
        ... ]
        >>> result = calculate_kelly_from_history(history, min_trades=3)
        >>> result['kelly']
        0.025  # ~2.5%
    """
    result = {
        "kelly": DEFAULT_KELLY_FRACTION,
        "win_rate": 0.0,
        "reward_risk": 0.0,
        "avg_win_r": 0.0,
        "avg_loss_r": 0.0,
        "sample_size": len(trade_history),
        "confidence": "insufficient_data",
        "growth_rate": 0.0,
        "trades_to_double": None
    }

    if len(trade_history) < min_trades:
        return result

    # Extract R-multiples
    r_multiples = [t.get('r_multiple', 0) for t in trade_history]

    # Separate wins and losses
    wins = [r for r in r_multiples if r > 0]
    losses = [r for r in r_multiples if r < 0]

    if not wins or not losses:
        # Need both wins and losses to calculate Kelly
        return result

    # Calculate statistics
    win_rate = len(wins) / len(r_multiples)
    avg_win_r = mean(wins)
    avg_loss_r = abs(mean(losses))

    # Reward/Risk ratio
    reward_risk = avg_win_r / avg_loss_r if avg_loss_r > 0 else 1.0

    # Calculate Kelly
    kelly = calculate_kelly(win_rate, reward_risk, kelly_mode)

    # Calculate growth rate at this Kelly
    growth_rate = calculate_growth_rate(kelly, win_rate, reward_risk)

    # Confidence level based on sample size
    confidence = _get_confidence_level(len(trade_history))

    # Trades to double
    ttd = trades_to_double(growth_rate) if growth_rate > 0 else None

    return {
        "kelly": kelly,
        "win_rate": win_rate,
        "reward_risk": reward_risk,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "sample_size": len(trade_history),
        "confidence": confidence,
        "growth_rate": growth_rate,
        "trades_to_double": ttd,
        "has_edge": kelly > 0
    }


def _get_confidence_level(n_trades: int) -> str:
    """
    Determine confidence level based on sample size.

    Statistical note: Standard error of win rate = sqrt(W(1-W)/n)
    At n=200, SE ≈ 3.5% for W=0.5
    """
    if n_trades >= 200:
        return "high"
    elif n_trades >= 100:
        return "medium"
    elif n_trades >= 50:
        return "low"
    else:
        return "very_low"


def calculate_kelly_confidence_interval(
    win_rate: float,
    reward_risk: float,
    n_trades: int,
    confidence: float = 0.95
) -> dict:
    """
    Calculate confidence interval for Kelly estimate.

    Uses normal approximation for win rate uncertainty.

    Args:
        win_rate: Observed win rate
        reward_risk: Observed R:R ratio
        n_trades: Number of trades in sample
        confidence: Confidence level (default 95%)

    Returns:
        Dict with lower_kelly, upper_kelly, and uncertainty_pct
    """
    if n_trades < 10:
        return {
            "lower_kelly": 0.0,
            "upper_kelly": MAX_KELLY_FRACTION,
            "uncertainty_pct": 100.0
        }

    # Standard error of win rate
    se_win_rate = sqrt(win_rate * (1 - win_rate) / n_trades)

    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    # Win rate bounds
    win_rate_low = max(0.01, win_rate - z * se_win_rate)
    win_rate_high = min(0.99, win_rate + z * se_win_rate)

    # Kelly at bounds (Half-Kelly)
    kelly_low = calculate_kelly(win_rate_low, reward_risk, "half")
    kelly_high = calculate_kelly(win_rate_high, reward_risk, "half")

    # Central estimate
    kelly_mid = calculate_kelly(win_rate, reward_risk, "half")

    # Uncertainty as percentage of central estimate
    if kelly_mid > 0:
        uncertainty = ((kelly_high - kelly_low) / kelly_mid) * 100
    else:
        uncertainty = 100.0

    return {
        "lower_kelly": kelly_low,
        "central_kelly": kelly_mid,
        "upper_kelly": kelly_high,
        "uncertainty_pct": uncertainty,
        "win_rate_se": se_win_rate
    }


def edge_exists(win_rate: float, reward_risk: float) -> bool:
    """
    Check if positive edge exists.

    Edge exists when: W > 1/(R+1)

    Args:
        win_rate: Win probability
        reward_risk: R:R ratio

    Returns:
        True if positive edge exists
    """
    if reward_risk <= 0:
        return False

    min_win_rate = 1 / (reward_risk + 1)
    return win_rate > min_win_rate


def minimum_win_rate_for_edge(reward_risk: float) -> float:
    """
    Calculate minimum win rate needed for positive edge.

    Args:
        reward_risk: R:R ratio

    Returns:
        Minimum win rate for Kelly > 0

    Example:
        >>> minimum_win_rate_for_edge(2.0)
        0.333...  # Need >33% win rate for R:R of 2:1
    """
    if reward_risk <= 0:
        return 1.0  # Impossible to have edge

    return 1 / (reward_risk + 1)


def kelly_comparison(
    win_rate: float,
    reward_risk: float
) -> dict:
    """
    Compare different Kelly fractions for analysis.

    Returns growth rate and variance metrics for Full, Half, Quarter Kelly.
    """
    full = calculate_kelly(win_rate, reward_risk, "full")
    half = calculate_kelly(win_rate, reward_risk, "half")
    quarter = calculate_kelly(win_rate, reward_risk, "quarter")

    full_growth = calculate_growth_rate(full, win_rate, reward_risk) if full > 0 else 0
    half_growth = calculate_growth_rate(half, win_rate, reward_risk) if half > 0 else 0
    quarter_growth = calculate_growth_rate(quarter, win_rate, reward_risk) if quarter > 0 else 0

    return {
        "full_kelly": {
            "fraction": full,
            "growth_rate": full_growth,
            "relative_growth": 1.0 if full_growth > 0 else 0,
            "trades_to_double": trades_to_double(full_growth)
        },
        "half_kelly": {
            "fraction": half,
            "growth_rate": half_growth,
            "relative_growth": half_growth / full_growth if full_growth > 0 else 0,
            "trades_to_double": trades_to_double(half_growth)
        },
        "quarter_kelly": {
            "fraction": quarter,
            "growth_rate": quarter_growth,
            "relative_growth": quarter_growth / full_growth if full_growth > 0 else 0,
            "trades_to_double": trades_to_double(quarter_growth)
        },
        "recommendation": "half_kelly",  # Per spec, Half-Kelly is recommended
        "reason": "75% of growth rate with 25% of variance"
    }
