"""
Risk Calculator Module - Trade Frequency Optimization Tools

This module provides mathematical tools for analyzing the relationship
between trade frequency and risk-adjusted returns.

Key Functions:
- Kelly Criterion calculation
- Effective positions (correlation-adjusted)
- Drawdown recovery estimation
- Monte Carlo drawdown simulation
- R-Multiple tracking and analysis

Usage:
    from tools.risk_calculator import (
        kelly_fraction,
        effective_positions,
        recovery_trades,
        monte_carlo_drawdown,
        RMultipleTracker
    )

    # Calculate optimal position size
    kelly = kelly_fraction(win_rate=0.79, avg_win=10, avg_loss=25)

    # Track trades
    tracker = RMultipleTracker()
    tracker.add_trade("BTCUSDT", 95000, 96500, 94500, "LONG", "TP")
    print(tracker.get_summary())
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime


# ============== KELLY CRITERION ==============

def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate optimal Kelly fraction for position sizing.

    The Kelly Criterion maximizes the expected log of wealth,
    providing the optimal fraction of capital to risk per trade.

    Formula: f* = (bp - q) / b
    Where:
        b = odds (avg_win / avg_loss)
        p = probability of winning
        q = probability of losing (1 - p)

    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average winning trade amount (positive number)
        avg_loss: Average losing trade amount (positive number)

    Returns:
        Optimal fraction of capital to risk (0-1)
        Returns 0 if no edge exists

    Example:
        >>> kelly_fraction(0.79, 10, 25)
        0.265  # 26.5% of capital (use 1/4 Kelly = 6.6%)
    """
    if avg_loss <= 0:
        return 0.0

    b = avg_win / avg_loss  # Odds ratio
    q = 1 - win_rate

    kelly = (b * win_rate - q) / b
    return max(0.0, kelly)


def fractional_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25
) -> float:
    """
    Calculate fractional Kelly (more conservative).

    Full Kelly is mathematically optimal but has high variance.
    Fractional Kelly (typically 1/4 to 1/2) reduces variance
    at the cost of some expected growth.

    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount
        fraction: Kelly fraction to use (default 0.25 = 1/4 Kelly)

    Returns:
        Recommended position size as fraction of capital
    """
    full_kelly = kelly_fraction(win_rate, avg_win, avg_loss)
    return full_kelly * fraction


# ============== CORRELATION ANALYSIS ==============

def effective_positions(
    n_positions: int,
    avg_correlation: float
) -> float:
    """
    Calculate effective number of independent positions.

    High correlation between positions reduces diversification benefit.
    This function calculates how many truly independent bets you have.

    Formula: N_eff = N / (1 + (N-1) * rho)

    Args:
        n_positions: Number of actual positions
        avg_correlation: Average pairwise correlation (0-1)

    Returns:
        Effective number of independent positions

    Example:
        >>> effective_positions(3, 0.90)
        1.07  # 3 positions at 0.90 correlation = 1.07 independent bets
    """
    if n_positions <= 0:
        return 0.0
    if n_positions == 1:
        return 1.0

    return n_positions / (1 + (n_positions - 1) * max(0, avg_correlation))


def diversification_loss(
    n_positions: int,
    avg_correlation: float
) -> float:
    """
    Calculate percentage of diversification lost to correlation.

    Args:
        n_positions: Number of actual positions
        avg_correlation: Average pairwise correlation

    Returns:
        Percentage of diversification lost (0-100)

    Example:
        >>> diversification_loss(3, 0.90)
        64.3  # 64.3% of diversification benefit is lost
    """
    if n_positions <= 1:
        return 0.0

    eff = effective_positions(n_positions, avg_correlation)
    return (1 - eff / n_positions) * 100


def position_size_adjustment(
    base_size: float,
    correlation_with_open: float,
    n_same_direction: int
) -> Tuple[float, str]:
    """
    Adjust position size based on correlation with open positions.

    Args:
        base_size: Base position size in USD
        correlation_with_open: Average correlation with same-direction positions
        n_same_direction: Number of existing positions in same direction

    Returns:
        Tuple of (adjusted_size, reason_string)

    Example:
        >>> position_size_adjustment(35.0, 0.90, 1)
        (17.5, "Reduced 50% due to 0.90 correlation with 1 position")
    """
    if n_same_direction == 0:
        return base_size, "Full size - first position"

    # Progressive reduction based on correlation and count
    # At 0.90 correlation: reduce by 50%
    # At 0.50 correlation: reduce by 25%
    # Each additional same-direction position: extra 10% reduction

    base_reduction = correlation_with_open * 0.55
    count_reduction = 0.10 * (n_same_direction - 1)
    total_reduction = min(0.67, base_reduction + count_reduction)  # Cap at 67%

    adjusted = base_size * (1 - total_reduction)
    reason = f"Reduced {total_reduction:.0%} due to {correlation_with_open:.2f} correlation with {n_same_direction} position(s)"

    return adjusted, reason


# ============== DRAWDOWN ANALYSIS ==============

def recovery_trades(
    drawdown_pct: float,
    expected_r: float,
    risk_pct: float = 0.0175
) -> int:
    """
    Estimate trades needed to recover from drawdown.

    Formula: n = ln(1 + DD) / ln(1 + r)

    Args:
        drawdown_pct: Drawdown as decimal (e.g., 0.10 for 10%)
        expected_r: Expected R-multiple per trade
        risk_pct: Risk per trade as decimal (default 1.75%)

    Returns:
        Estimated number of trades to recover

    Example:
        >>> recovery_trades(0.10, 0.17, 0.0175)
        32  # 32 trades needed to recover from 10% drawdown
    """
    if expected_r <= 0 or risk_pct <= 0:
        return float('inf')

    r_per_trade = expected_r * risk_pct
    if r_per_trade <= 0:
        return float('inf')

    trades = np.log(1 + drawdown_pct) / np.log(1 + r_per_trade)
    return int(np.ceil(trades))


def recovery_time_months(
    drawdown_pct: float,
    expected_r: float,
    trades_per_month: float,
    risk_pct: float = 0.0175
) -> float:
    """
    Estimate months needed to recover from drawdown.

    Args:
        drawdown_pct: Drawdown as decimal
        expected_r: Expected R-multiple per trade
        trades_per_month: Average trades per month
        risk_pct: Risk per trade as decimal

    Returns:
        Estimated months to recover

    Example:
        >>> recovery_time_months(0.10, 0.17, 1.1, 0.0175)
        29.1  # About 29 months to recover at current frequency
    """
    trades = recovery_trades(drawdown_pct, expected_r, risk_pct)
    if trades == float('inf'):
        return float('inf')

    return trades / trades_per_month


def monte_carlo_drawdown(
    n_trades: int,
    win_rate: float,
    avg_win_r: float = 0.28,
    avg_loss_r: float = 0.71,
    risk_pct: float = 0.0175,
    n_simulations: int = 10000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Monte Carlo simulation of maximum drawdown.

    Simulates many paths of trade outcomes to estimate
    the distribution of maximum drawdown.

    Args:
        n_trades: Number of trades per simulation
        win_rate: Probability of winning (0-1)
        avg_win_r: Average winning R-multiple
        avg_loss_r: Average losing R-multiple (positive)
        risk_pct: Risk per trade as decimal
        n_simulations: Number of Monte Carlo paths
        seed: Random seed for reproducibility

    Returns:
        Dict with mean, median, p95, p99 max drawdown percentages

    Example:
        >>> monte_carlo_drawdown(24, 0.79)
        {'mean': 0.028, 'median': 0.022, 'p95': 0.065, 'p99': 0.092}
    """
    np.random.seed(seed)
    max_dds = []

    for _ in range(n_simulations):
        equity = 1.0
        peak = 1.0
        max_dd = 0.0

        for _ in range(n_trades):
            if np.random.random() < win_rate:
                equity *= (1 + avg_win_r * risk_pct)
            else:
                equity *= (1 - avg_loss_r * risk_pct)

            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        max_dds.append(max_dd)

    return {
        'mean': np.mean(max_dds),
        'median': np.median(max_dds),
        'p95': np.percentile(max_dds, 95),
        'p99': np.percentile(max_dds, 99),
        'std': np.std(max_dds)
    }


# ============== FREQUENCY OPTIMIZATION ==============

def sharpe_ratio_by_frequency(
    base_expected_r: float,
    base_sigma_r: float,
    frequency_multiplier: float,
    ir_decay_rate: float = 0.05
) -> Dict[str, float]:
    """
    Calculate Sharpe ratio at different trade frequencies.

    As frequency increases (filters relaxed), information ratio decays,
    but sqrt(N) effect can still improve Sharpe.

    Args:
        base_expected_r: Expected R at current frequency
        base_sigma_r: Std dev of R at current frequency
        frequency_multiplier: Multiple of current frequency
        ir_decay_rate: IR decay per 2x frequency increase

    Returns:
        Dict with new_ir, sharpe_ratio, optimal analysis
    """
    # Information Ratio at base
    base_ir = base_expected_r / base_sigma_r

    # IR decays as filters are relaxed
    log_mult = np.log2(max(1, frequency_multiplier))
    new_ir = base_ir * (1 - ir_decay_rate * log_mult)

    # Sharpe = sqrt(N) * IR
    sharpe = np.sqrt(frequency_multiplier) * new_ir

    # Marginal benefit
    base_sharpe = np.sqrt(1) * base_ir
    improvement = (sharpe - base_sharpe) / base_sharpe * 100

    return {
        'frequency_multiplier': frequency_multiplier,
        'base_ir': base_ir,
        'new_ir': new_ir,
        'ir_decay': 1 - new_ir / base_ir,
        'sharpe_ratio': sharpe,
        'sharpe_improvement_pct': improvement
    }


def optimal_frequency_search(
    base_expected_r: float,
    base_sigma_r: float,
    ir_decay_rate: float = 0.05,
    max_multiplier: float = 50
) -> Dict[str, float]:
    """
    Find optimal trade frequency multiplier.

    Searches for the frequency that maximizes Sharpe ratio,
    balancing sqrt(N) benefit against IR decay.

    Args:
        base_expected_r: Expected R at current frequency
        base_sigma_r: Std dev of R at current frequency
        ir_decay_rate: IR decay per 2x frequency
        max_multiplier: Maximum frequency multiple to consider

    Returns:
        Dict with optimal multiplier and expected Sharpe
    """
    best_sharpe = 0
    best_mult = 1

    for mult in np.linspace(1, max_multiplier, 1000):
        result = sharpe_ratio_by_frequency(
            base_expected_r, base_sigma_r, mult, ir_decay_rate
        )
        if result['sharpe_ratio'] > best_sharpe:
            best_sharpe = result['sharpe_ratio']
            best_mult = mult

    return {
        'optimal_multiplier': best_mult,
        'optimal_sharpe': best_sharpe,
        'base_sharpe': base_expected_r / base_sigma_r,
        'sharpe_improvement': best_sharpe / (base_expected_r / base_sigma_r) - 1
    }


# ============== R-MULTIPLE TRACKING ==============

@dataclass
class Trade:
    """Single trade record."""
    trade_id: int
    symbol: str
    entry: float
    exit: float
    sl: float
    direction: str  # "LONG" or "SHORT"
    exit_type: str  # "TP", "SL", "PARTIAL", "MANUAL"
    r_multiple: float
    pnl_usd: float
    timestamp: datetime = field(default_factory=datetime.now)


class RMultipleTracker:
    """
    Track trades in R-multiple terms for objective analysis.

    R-Multiple = (Exit - Entry) / (Entry - SL) for LONG
    R-Multiple = (Entry - Exit) / (SL - Entry) for SHORT

    1R = max planned loss
    +2R = profit of 2x the planned loss
    """

    def __init__(self, risk_per_trade_usd: float = 35.0):
        """
        Initialize tracker.

        Args:
            risk_per_trade_usd: Dollar amount risked per trade (1R)
        """
        self.risk_per_trade = risk_per_trade_usd
        self.trades: List[Trade] = []
        self._next_id = 1

    def add_trade(
        self,
        symbol: str,
        entry: float,
        exit: float,
        sl: float,
        direction: str,
        exit_type: str = "TP",
        timestamp: datetime = None
    ) -> Trade:
        """
        Add a completed trade.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            entry: Entry price
            exit: Exit price
            sl: Stop loss price
            direction: "LONG" or "SHORT"
            exit_type: "TP", "SL", "PARTIAL", "MANUAL"
            timestamp: Trade time (default: now)

        Returns:
            Trade object
        """
        if direction == "LONG":
            risk = entry - sl
            reward = exit - entry
        else:
            risk = sl - entry
            reward = entry - exit

        if risk <= 0:
            r_multiple = 0.0
        else:
            r_multiple = reward / risk

        pnl_usd = r_multiple * self.risk_per_trade

        trade = Trade(
            trade_id=self._next_id,
            symbol=symbol,
            entry=entry,
            exit=exit,
            sl=sl,
            direction=direction,
            exit_type=exit_type,
            r_multiple=r_multiple,
            pnl_usd=pnl_usd,
            timestamp=timestamp or datetime.now()
        )

        self.trades.append(trade)
        self._next_id += 1
        return trade

    def get_summary(self) -> Dict:
        """
        Get comprehensive trading statistics.

        Returns:
            Dict with expectancy, win rate, Sharpe, etc.
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'expectancy_r': 0,
                'win_rate': 0,
                'profit_factor': 0
            }

        r_multiples = [t.r_multiple for t in self.trades]
        wins = [r for r in r_multiples if r > 0]
        losses = [r for r in r_multiples if r <= 0]

        total = len(r_multiples)
        win_count = len(wins)
        loss_count = len(losses)

        win_rate = win_count / total if total > 0 else 0
        avg_win_r = np.mean(wins) if wins else 0
        avg_loss_r = abs(np.mean(losses)) if losses else 0

        expectancy = np.mean(r_multiples)
        total_r = np.sum(r_multiples)
        total_pnl = total_r * self.risk_per_trade

        gross_profit = sum(r for r in r_multiples if r > 0) * self.risk_per_trade
        gross_loss = abs(sum(r for r in r_multiples if r < 0) * self.risk_per_trade)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe approximation
        if len(r_multiples) > 1:
            sharpe = np.mean(r_multiples) / np.std(r_multiples) * np.sqrt(len(r_multiples))
        else:
            sharpe = 0

        # Kelly calculation
        kelly = kelly_fraction(win_rate, avg_win_r, avg_loss_r) if avg_loss_r > 0 else 0

        return {
            'total_trades': total,
            'wins': win_count,
            'losses': loss_count,
            'win_rate': win_rate,
            'avg_win_r': avg_win_r,
            'avg_loss_r': avg_loss_r,
            'expectancy_r': expectancy,
            'total_r': total_r,
            'total_pnl_usd': total_pnl,
            'profit_factor': profit_factor,
            'sharpe_approx': sharpe,
            'kelly_full': kelly,
            'kelly_quarter': kelly * 0.25,
            'max_r': max(r_multiples) if r_multiples else 0,
            'min_r': min(r_multiples) if r_multiples else 0
        }

    def get_rolling_expectancy(self, window: int = 20) -> List[float]:
        """
        Calculate rolling E[R] over last N trades.

        Args:
            window: Number of trades to include

        Returns:
            List of rolling E[R] values
        """
        if len(self.trades) < window:
            return []

        r_multiples = [t.r_multiple for t in self.trades]
        rolling = []

        for i in range(window, len(r_multiples) + 1):
            window_r = r_multiples[i-window:i]
            rolling.append(np.mean(window_r))

        return rolling

    def get_drawdown_curve(self) -> Tuple[List[float], List[float]]:
        """
        Calculate equity curve and drawdown over time.

        Returns:
            Tuple of (equity_curve, drawdown_curve)
        """
        equity = [1.0]  # Start at 1.0 for normalized view

        for trade in self.trades:
            pct_change = trade.r_multiple * (self.risk_per_trade / 2000)  # Assume $2000 account
            equity.append(equity[-1] * (1 + pct_change))

        # Calculate drawdown
        peak = 1.0
        drawdowns = [0.0]

        for e in equity[1:]:
            peak = max(peak, e)
            dd = (peak - e) / peak
            drawdowns.append(dd)

        return equity, drawdowns

    def print_trade_log(self):
        """Print formatted trade log."""
        print("\n" + "=" * 80)
        print("TRADE LOG (R-MULTIPLE)")
        print("=" * 80)
        print(f"{'#':<4} {'Symbol':<10} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'R-Mult':>8} {'Cum R':>8}")
        print("-" * 80)

        cum_r = 0.0
        for t in self.trades:
            cum_r += t.r_multiple
            print(f"{t.trade_id:<4} {t.symbol:<10} {t.direction:<6} "
                  f"{t.entry:>10.2f} {t.exit:>10.2f} "
                  f"{t.r_multiple:>+8.2f}R {cum_r:>+8.2f}R")

        print("-" * 80)
        summary = self.get_summary()
        print(f"Total Trades: {summary['total_trades']} | "
              f"Win Rate: {summary['win_rate']:.1%} | "
              f"E[R]: {summary['expectancy_r']:+.2f}R | "
              f"Total: {summary['total_r']:+.2f}R (${summary['total_pnl_usd']:+.2f})")
        print("=" * 80 + "\n")


# ============== SCENARIO ANALYSIS ==============

def compare_frequency_scenarios(
    current_trades: int = 13,
    current_win_rate: float = 0.79,
    current_avg_win_r: float = 0.28,
    current_avg_loss_r: float = 0.71,
    risk_pct: float = 0.0175
) -> None:
    """
    Compare current vs high frequency scenarios.

    Prints detailed comparison of risk metrics.
    """
    print("\n" + "=" * 80)
    print("FREQUENCY SCENARIO COMPARISON")
    print("=" * 80)

    scenarios = [
        ("Current", current_trades, current_win_rate, current_avg_win_r, current_avg_loss_r),
        ("2x Freq", current_trades * 2, 0.70, 0.35, 0.80),
        ("5x Freq", current_trades * 5, 0.60, 0.50, 0.90),
        ("10x Freq", current_trades * 10, 0.50, 0.75, 1.00),
    ]

    print(f"\n{'Scenario':<12} {'Trades':>8} {'WinRate':>8} {'E[R]':>8} {'Kelly':>8} {'Recovery':>10}")
    print("-" * 80)

    for name, trades, wr, avg_win, avg_loss in scenarios:
        exp_r = wr * avg_win - (1 - wr) * avg_loss
        kelly = kelly_fraction(wr, avg_win, avg_loss)
        rec = recovery_trades(0.10, exp_r, risk_pct)

        print(f"{name:<12} {trades:>8} {wr:>8.1%} {exp_r:>+8.2f}R {kelly:>8.1%} {rec:>10} trades")

    print("\n" + "=" * 80)


# ============== MAIN (DEMO) ==============

if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("RISK CALCULATOR - DEMO")
    print("=" * 60)

    # Kelly Criterion
    print("\n[1] Kelly Criterion Analysis")
    print("-" * 40)
    full_kelly = kelly_fraction(0.79, 10, 25)
    quarter_kelly = fractional_kelly(0.79, 10, 25, 0.25)
    print(f"Win Rate: 79%, Avg Win: $10, Avg Loss: $25")
    print(f"Full Kelly: {full_kelly:.1%}")
    print(f"1/4 Kelly (Recommended): {quarter_kelly:.1%}")

    # Effective Positions
    print("\n[2] Correlation Analysis")
    print("-" * 40)
    eff = effective_positions(3, 0.883)
    loss = diversification_loss(3, 0.883)
    print(f"3 positions at 0.883 avg correlation:")
    print(f"Effective positions: {eff:.2f}")
    print(f"Diversification lost: {loss:.1f}%")

    # Drawdown Recovery
    print("\n[3] Drawdown Recovery")
    print("-" * 40)
    trades_13 = recovery_trades(0.10, 0.17, 0.0175)
    trades_130 = recovery_trades(0.10, 0.25, 0.00875)
    print(f"10% DD recovery at 13 trades/yr: {trades_13} trades ({trades_13/13*12:.0f} months)")
    print(f"10% DD recovery at 130 trades/yr: {trades_130} trades ({trades_130/130*12:.0f} months)")

    # Monte Carlo
    print("\n[4] Monte Carlo Max Drawdown")
    print("-" * 40)
    mc = monte_carlo_drawdown(24, 0.79)
    print(f"24 trades at 79% win rate:")
    print(f"Mean Max DD: {mc['mean']:.1%}")
    print(f"95th percentile: {mc['p95']:.1%}")

    # R-Multiple Tracker Demo
    print("\n[5] R-Multiple Tracker Demo")
    print("-" * 40)
    tracker = RMultipleTracker(risk_per_trade_usd=35.0)

    # Add sample trades (from actual backtest pattern)
    sample_trades = [
        ("BTCUSDT", 95000, 96200, 94200, "LONG", "TP"),
        ("ETHUSDT", 3200, 3150, 3280, "LONG", "SL"),
        ("BTCUSDT", 97000, 98500, 96000, "LONG", "TP"),
        ("LINKUSDT", 18.5, 19.8, 17.8, "LONG", "TP"),
        ("ETHUSDT", 3400, 3520, 3320, "LONG", "TP"),
    ]

    for s, e, x, sl, d, t in sample_trades:
        tracker.add_trade(s, e, x, sl, d, t)

    tracker.print_trade_log()

    # Frequency comparison
    compare_frequency_scenarios()
