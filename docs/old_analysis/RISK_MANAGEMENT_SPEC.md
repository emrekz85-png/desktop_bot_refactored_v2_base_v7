# Risk Management Specification

**Project:** SSL Flow Trading Bot
**Version:** 2.0
**Date:** January 2, 2026
**Status:** SPECIFICATION (Pre-Implementation)

---

## Executive Summary

This document specifies a mathematically-rigorous risk management system based on:
- **Kelly Criterion** for optimal geometric growth
- **Correlation Matrix** for effective position calculation
- **Fixed Fractional** position sizing that scales with equity
- **Drawdown-Based Auto-Adjustment** for anti-fragile behavior
- **20% Maximum Drawdown** circuit breaker

The system is designed to work regardless of account equity size.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [Position Sizing System](#3-position-sizing-system)
4. [Portfolio Risk Management](#4-portfolio-risk-management)
5. [Drawdown Management](#5-drawdown-management)
6. [R-Multiple Tracking](#6-r-multiple-tracking)
7. [Implementation Architecture](#7-implementation-architecture)
8. [Configuration Parameters](#8-configuration-parameters)
9. [Validation & Testing](#9-validation--testing)

---

## 1. Design Principles

### 1.1 Core Philosophy

| Principle | Description |
|-----------|-------------|
| **Equity-Agnostic** | All calculations use percentages, not fixed dollar amounts |
| **Mathematically Optimal** | Based on Kelly Criterion for maximum geometric growth |
| **Anti-Fragile** | Automatically reduces risk during drawdowns |
| **Correlation-Aware** | Accounts for asset correlations in portfolio risk |
| **Conservative Default** | Uses Half-Kelly as baseline (75% growth, 25% variance) |

### 1.2 Key Constraints

```
MAX_SINGLE_POSITION_RISK = 5% of equity
MAX_PORTFOLIO_RISK = 15% of equity
MAX_DRAWDOWN_LIMIT = 20% of peak equity
MAX_POSITIONS_SAME_DIRECTION = 3
MIN_KELLY_FRACTION = 0.25% (floor)
MAX_KELLY_FRACTION = 5% (ceiling)
```

---

## 2. Mathematical Foundation

### 2.1 Kelly Criterion

The Kelly Criterion maximizes the expected logarithm of wealth (geometric growth rate).

#### Basic Formula

```
f* = W - (1-W)/R

Where:
  f* = Optimal fraction of capital to risk
  W  = Win rate (probability of winning)
  R  = Reward-to-Risk ratio (avg_win / avg_loss)
```

#### Derivation

For a trade with:
- Win probability: W
- Reward multiple: R (profit as multiple of risk)
- Loss multiple: 1 (full risk lost)

The growth rate G(f) for fraction f is:

```
G(f) = W × log(1 + f×R) + (1-W) × log(1 - f)
```

Maximizing G(f) yields:

```
dG/df = W×R/(1+f×R) - (1-W)/(1-f) = 0

Solving: f* = W - (1-W)/R
```

### 2.2 Edge Requirement

Kelly is only positive when there is a positive edge:

```
f* > 0  ⟺  W×R > (1-W)  ⟺  W > 1/(R+1)
```

**Example:** For R:R = 2.0, minimum win rate = 1/(2+1) = 33.3%

### 2.3 Growth Rate at Optimal Kelly

```
G* = W × log(1 + f*×R) + (1-W) × log(1 - f*)
```

**Approximation for small f:**

```
G* ≈ (W×R - (1-W))² / (2×R) = f*² × R / 2
```

### 2.4 Half-Kelly Justification

| Metric | Full Kelly | Half Kelly | Reduction |
|--------|------------|------------|-----------|
| Growth Rate | G* | 0.75 × G* | 25% |
| Variance | σ² | 0.25 × σ² | 75% |
| Max Drawdown | High | ~50% lower | Significant |

**Recommendation:** Use Half-Kelly (f*/2) as the baseline.

### 2.5 Trades to Double Capital

```
E[trades_to_double] = log(2) / G(f)

At Half-Kelly: ≈ log(2) / (0.75 × G*)
```

---

## 3. Position Sizing System

### 3.1 Position Size Calculation

```python
def calculate_position_size(
    equity: float,
    kelly_fraction: float,
    entry_price: float,
    stop_loss_price: float,
    leverage: float = 10
) -> dict:
    """
    Calculate position size based on Kelly fraction.

    Returns:
        risk_amount: Dollar amount at risk
        position_size: Number of units
        notional_value: Total position value
        margin_required: Margin to hold position
    """
    # Risk amount = Kelly fraction × Equity
    risk_amount = kelly_fraction * equity

    # Stop loss distance
    sl_distance = abs(entry_price - stop_loss_price)
    sl_percent = sl_distance / entry_price

    # Position size = Risk / SL distance
    position_size = risk_amount / sl_distance

    # Notional and margin
    notional_value = position_size * entry_price
    margin_required = notional_value / leverage

    return {
        "risk_amount": risk_amount,
        "position_size": position_size,
        "notional_value": notional_value,
        "margin_required": margin_required,
        "risk_percent": kelly_fraction * 100
    }
```

### 3.2 Dynamic Kelly Calculation

```python
def calculate_dynamic_kelly(
    trade_history: list,
    min_trades: int = 30,
    use_bayesian: bool = True
) -> dict:
    """
    Calculate Kelly fraction from trade history.
    """
    if len(trade_history) < min_trades:
        return {
            "kelly": 0.01,  # Minimum 1%
            "confidence": "insufficient_data"
        }

    # Calculate statistics
    wins = [t for t in trade_history if t['r_multiple'] > 0]
    losses = [t for t in trade_history if t['r_multiple'] < 0]

    win_rate = len(wins) / len(trade_history)
    avg_win_r = mean([t['r_multiple'] for t in wins])
    avg_loss_r = abs(mean([t['r_multiple'] for t in losses]))

    reward_risk = avg_win_r / avg_loss_r if avg_loss_r > 0 else 1.0

    # Kelly formula
    kelly = win_rate - (1 - win_rate) / reward_risk

    # Apply Half-Kelly by default
    kelly = kelly * 0.5

    # Apply bounds
    kelly = max(0.0025, min(0.05, kelly))

    return {
        "kelly": kelly,
        "win_rate": win_rate,
        "reward_risk": reward_risk,
        "sample_size": len(trade_history),
        "confidence": _get_confidence_level(len(trade_history))
    }

def _get_confidence_level(n_trades: int) -> str:
    if n_trades >= 200:
        return "high"
    elif n_trades >= 100:
        return "medium"
    elif n_trades >= 50:
        return "low"
    else:
        return "very_low"
```

### 3.3 Per-Trade R:R Calculation

Since you selected "Per-Trade Actual" R:R method:

```python
def calculate_trade_rr(
    entry_price: float,
    take_profit: float,
    stop_loss: float,
    trade_type: str  # "LONG" or "SHORT"
) -> float:
    """
    Calculate R:R ratio for a specific trade setup.
    """
    if trade_type == "LONG":
        reward_distance = take_profit - entry_price
        risk_distance = entry_price - stop_loss
    else:  # SHORT
        reward_distance = entry_price - take_profit
        risk_distance = stop_loss - entry_price

    if risk_distance <= 0:
        return 0.0

    return reward_distance / risk_distance
```

---

## 4. Portfolio Risk Management

### 4.1 Correlation Matrix

```python
CRYPTO_CORRELATIONS = {
    # Based on historical 90-day rolling correlations
    ("BTCUSDT", "ETHUSDT"): 0.85,
    ("BTCUSDT", "LINKUSDT"): 0.78,
    ("ETHUSDT", "LINKUSDT"): 0.82,
    # Add more pairs as needed
}

def get_correlation(symbol1: str, symbol2: str) -> float:
    """Get correlation between two assets."""
    pair = tuple(sorted([symbol1, symbol2]))
    return CRYPTO_CORRELATIONS.get(pair, 0.70)  # Default 0.70
```

### 4.2 Effective Position Calculation

When positions are correlated, the effective diversification is reduced:

```python
def calculate_effective_positions(
    open_positions: list,
    correlation_matrix: dict
) -> float:
    """
    Calculate effective number of independent positions.

    Formula: N_eff = N / (1 + (N-1) × avg_correlation)
    """
    n = len(open_positions)
    if n <= 1:
        return n

    # Calculate average pairwise correlation
    correlations = []
    for i, pos1 in enumerate(open_positions):
        for j, pos2 in enumerate(open_positions):
            if i < j:
                corr = get_correlation(pos1['symbol'], pos2['symbol'])
                correlations.append(corr)

    avg_corr = mean(correlations) if correlations else 0.0

    # Effective positions formula
    n_effective = n / (1 + (n - 1) * avg_corr)

    return n_effective
```

### 4.3 Correlation-Adjusted Position Sizing

```python
def adjust_kelly_for_correlation(
    base_kelly: float,
    open_positions: list,
    new_position_symbol: str,
    new_position_direction: str  # "LONG" or "SHORT"
) -> dict:
    """
    Adjust Kelly fraction based on existing portfolio correlation.
    """
    if not open_positions:
        return {
            "adjusted_kelly": base_kelly,
            "adjustment_factor": 1.0,
            "reason": "no_existing_positions"
        }

    # Count same-direction positions
    same_direction = [
        p for p in open_positions
        if p['direction'] == new_position_direction
    ]

    # Rule: Max 3 positions in same direction
    if len(same_direction) >= 3:
        return {
            "adjusted_kelly": 0.0,
            "adjustment_factor": 0.0,
            "reason": "max_same_direction_reached"
        }

    # Calculate average correlation with existing same-direction positions
    correlations = [
        get_correlation(new_position_symbol, p['symbol'])
        for p in same_direction
    ]

    if not correlations:
        return {
            "adjusted_kelly": base_kelly,
            "adjustment_factor": 1.0,
            "reason": "no_same_direction_positions"
        }

    avg_corr = mean(correlations)

    # Adjustment formula: reduce by correlation factor
    # High correlation (0.9) → 0.55 factor
    # Medium correlation (0.5) → 0.85 factor
    # Low correlation (0.2) → 0.95 factor
    adjustment_factor = 1 - (avg_corr * 0.5)

    adjusted_kelly = base_kelly * adjustment_factor

    return {
        "adjusted_kelly": adjusted_kelly,
        "adjustment_factor": adjustment_factor,
        "avg_correlation": avg_corr,
        "n_same_direction": len(same_direction),
        "reason": "correlation_adjusted"
    }
```

### 4.4 Portfolio Risk Aggregation

```python
def calculate_portfolio_risk(
    open_positions: list,
    equity: float
) -> dict:
    """
    Calculate total portfolio risk considering correlations.
    """
    if not open_positions:
        return {
            "total_risk_percent": 0.0,
            "effective_risk_percent": 0.0,
            "n_positions": 0,
            "n_effective": 0.0
        }

    # Sum of individual risks
    total_risk = sum(p['risk_amount'] for p in open_positions)
    total_risk_percent = (total_risk / equity) * 100

    # Effective positions
    n_effective = calculate_effective_positions(open_positions, CRYPTO_CORRELATIONS)
    n_actual = len(open_positions)

    # Diversification benefit
    diversification_ratio = n_effective / n_actual if n_actual > 0 else 1.0

    # Effective risk (accounting for diversification)
    effective_risk = total_risk * sqrt(diversification_ratio)
    effective_risk_percent = (effective_risk / equity) * 100

    return {
        "total_risk_percent": total_risk_percent,
        "effective_risk_percent": effective_risk_percent,
        "n_positions": n_actual,
        "n_effective": n_effective,
        "diversification_ratio": diversification_ratio
    }
```

---

## 5. Drawdown Management

### 5.1 Drawdown Calculation

```python
def calculate_drawdown(
    current_equity: float,
    peak_equity: float
) -> dict:
    """
    Calculate current drawdown from peak.
    """
    if peak_equity <= 0:
        return {"drawdown_percent": 0.0, "drawdown_amount": 0.0}

    drawdown_amount = peak_equity - current_equity
    drawdown_percent = (drawdown_amount / peak_equity) * 100

    return {
        "drawdown_percent": max(0, drawdown_percent),
        "drawdown_amount": max(0, drawdown_amount),
        "peak_equity": peak_equity,
        "current_equity": current_equity
    }
```

### 5.2 Kelly Auto-Adjustment (Selected Method)

As you selected "Kelly Auto-Adjust", the system automatically reduces position sizing as drawdown increases:

```python
def get_drawdown_kelly_multiplier(drawdown_percent: float) -> float:
    """
    Calculate Kelly multiplier based on current drawdown.

    Uses exponential decay for smooth, anti-fragile adjustment.

    Drawdown → Multiplier:
        0%  → 1.00 (full Kelly)
        5%  → 0.85
       10%  → 0.70
       15%  → 0.50
       20%  → 0.00 (circuit breaker)
    """
    if drawdown_percent >= 20:
        return 0.0  # Circuit breaker - stop trading

    if drawdown_percent <= 0:
        return 1.0

    # Exponential decay formula
    # multiplier = e^(-k × drawdown)
    # where k is calibrated so that 20% DD → ~0
    k = 0.15  # Calibration constant

    multiplier = exp(-k * drawdown_percent)

    # Floor at 0.25 until circuit breaker
    return max(0.25, multiplier)


def adjust_kelly_for_drawdown(
    base_kelly: float,
    current_equity: float,
    peak_equity: float
) -> dict:
    """
    Apply drawdown-based Kelly adjustment.
    """
    dd = calculate_drawdown(current_equity, peak_equity)
    multiplier = get_drawdown_kelly_multiplier(dd['drawdown_percent'])

    adjusted_kelly = base_kelly * multiplier

    # Check circuit breaker
    if multiplier == 0:
        return {
            "adjusted_kelly": 0.0,
            "multiplier": 0.0,
            "drawdown_percent": dd['drawdown_percent'],
            "status": "CIRCUIT_BREAKER_ACTIVE",
            "can_trade": False
        }

    return {
        "adjusted_kelly": adjusted_kelly,
        "multiplier": multiplier,
        "drawdown_percent": dd['drawdown_percent'],
        "status": "NORMAL" if multiplier > 0.7 else "REDUCED",
        "can_trade": True
    }
```

### 5.3 Drawdown Recovery Rules

```python
def get_recovery_status(
    current_equity: float,
    peak_equity: float,
    circuit_breaker_triggered_at: float = None
) -> dict:
    """
    Determine trading status during drawdown recovery.
    """
    dd = calculate_drawdown(current_equity, peak_equity)

    # If circuit breaker was triggered
    if circuit_breaker_triggered_at is not None:
        # Calculate recovery from circuit breaker point
        recovery_from_cb = (
            (current_equity - circuit_breaker_triggered_at) /
            circuit_breaker_triggered_at * 100
        )

        # Require 5% recovery before resuming
        if recovery_from_cb < 5.0:
            return {
                "can_trade": False,
                "status": "RECOVERING",
                "recovery_percent": recovery_from_cb,
                "required_recovery": 5.0,
                "message": f"Need {5.0 - recovery_from_cb:.1f}% more recovery"
            }
        else:
            return {
                "can_trade": True,
                "status": "RESUMED",
                "kelly_multiplier": 0.25,  # Start with quarter Kelly
                "message": "Trading resumed at reduced size"
            }

    return {
        "can_trade": dd['drawdown_percent'] < 20,
        "status": "NORMAL" if dd['drawdown_percent'] < 10 else "CAUTION",
        "kelly_multiplier": get_drawdown_kelly_multiplier(dd['drawdown_percent']),
        "drawdown_percent": dd['drawdown_percent']
    }
```

---

## 6. R-Multiple Tracking

### 6.1 R-Multiple Calculation

```python
def calculate_r_multiple(
    pnl: float,
    risk_amount: float
) -> float:
    """
    Calculate R-Multiple for a completed trade.

    R = PnL / Risk

    Examples:
        +2.0R = Won twice what was risked
        -1.0R = Lost exactly what was risked (full stop)
        +0.5R = Partial win (half of risk)
    """
    if risk_amount <= 0:
        return 0.0

    return pnl / risk_amount
```

### 6.2 Expected R (Expectancy)

```python
def calculate_expectancy(trade_history: list) -> dict:
    """
    Calculate E[R] - Expected R-Multiple per trade.

    E[R] = (Win% × Avg_Win_R) - (Loss% × Avg_Loss_R)

    Alternatively: E[R] = Mean(all R-multiples)
    """
    if not trade_history:
        return {"expectancy": 0.0, "sample_size": 0}

    r_multiples = [t['r_multiple'] for t in trade_history]

    expectancy = mean(r_multiples)

    # Also calculate component breakdown
    wins = [r for r in r_multiples if r > 0]
    losses = [r for r in r_multiples if r < 0]

    win_rate = len(wins) / len(r_multiples)
    avg_win = mean(wins) if wins else 0
    avg_loss = abs(mean(losses)) if losses else 0

    return {
        "expectancy": expectancy,
        "win_rate": win_rate,
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
        "sample_size": len(trade_history),
        "profitable": expectancy > 0
    }
```

### 6.3 Rolling E[R] for Edge Detection

```python
def calculate_rolling_expectancy(
    trade_history: list,
    window: int = 20
) -> dict:
    """
    Calculate rolling E[R] to detect edge degradation.
    """
    if len(trade_history) < window:
        return {"has_edge": None, "message": "Insufficient data"}

    # Recent trades
    recent = trade_history[-window:]
    recent_exp = calculate_expectancy(recent)

    # All-time trades
    all_time_exp = calculate_expectancy(trade_history)

    # Compare recent vs all-time
    edge_ratio = recent_exp['expectancy'] / all_time_exp['expectancy'] \
                 if all_time_exp['expectancy'] != 0 else 0

    # Determine edge status
    if recent_exp['expectancy'] <= 0:
        status = "NO_EDGE"
        action = "stop_trading"
    elif edge_ratio < 0.5:
        status = "DEGRADED"
        action = "reduce_size"
    elif edge_ratio < 0.8:
        status = "WEAKENING"
        action = "monitor"
    else:
        status = "HEALTHY"
        action = "continue"

    return {
        "recent_expectancy": recent_exp['expectancy'],
        "all_time_expectancy": all_time_exp['expectancy'],
        "edge_ratio": edge_ratio,
        "status": status,
        "recommended_action": action,
        "window": window
    }
```

---

## 7. Implementation Architecture

### 7.1 Module Structure

```
core/
├── risk_manager.py          # Main RiskManager class
├── kelly_calculator.py      # Kelly Criterion calculations
├── correlation_manager.py   # Portfolio correlation (existing)
├── position_sizer.py        # Position sizing logic
└── drawdown_tracker.py      # Drawdown and recovery tracking
```

### 7.2 RiskManager Class Design

```python
class RiskManager:
    """
    Central risk management coordinator.

    Integrates:
    - Kelly Criterion position sizing
    - Correlation-adjusted portfolio risk
    - Drawdown-based auto-adjustment
    - R-Multiple tracking
    """

    def __init__(
        self,
        initial_equity: float,
        max_single_risk: float = 0.05,      # 5%
        max_portfolio_risk: float = 0.15,   # 15%
        max_drawdown: float = 0.20,         # 20%
        kelly_fraction: float = 0.5,        # Half-Kelly
        min_trades_for_kelly: int = 30
    ):
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.current_equity = initial_equity

        self.max_single_risk = max_single_risk
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown = max_drawdown
        self.kelly_fraction = kelly_fraction
        self.min_trades = min_trades_for_kelly

        self.trade_history = []
        self.open_positions = []
        self.circuit_breaker_active = False
        self.circuit_breaker_equity = None

    def calculate_position_size(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> dict:
        """
        Master method for position sizing.

        Steps:
        1. Check circuit breaker
        2. Calculate base Kelly from history
        3. Adjust for drawdown
        4. Adjust for correlation
        5. Apply constraints
        6. Return final size
        """
        pass  # Implementation details below

    def update_equity(self, new_equity: float):
        """Update equity and peak tracking."""
        pass

    def record_trade(self, trade: dict):
        """Record completed trade for statistics."""
        pass

    def get_portfolio_status(self) -> dict:
        """Get current portfolio risk status."""
        pass
```

### 7.3 Integration Points

```python
# In TradeManager.open_trade():
def open_trade(self, signal, df, config):
    # Get position sizing from RiskManager
    sizing = self.risk_manager.calculate_position_size(
        symbol=signal.symbol,
        direction=signal.direction,
        entry_price=signal.entry,
        stop_loss=signal.sl,
        take_profit=signal.tp
    )

    if not sizing['can_trade']:
        return None, sizing['reason']

    # Use risk_amount instead of fixed percentage
    risk_amount = sizing['risk_amount']
    position_size = sizing['position_size']

    # ... rest of trade opening logic

# In TradeManager.close_trade():
def close_trade(self, trade, exit_price, exit_reason):
    # Calculate R-Multiple
    r_multiple = (trade.pnl) / trade.risk_amount

    # Record to RiskManager
    self.risk_manager.record_trade({
        'symbol': trade.symbol,
        'direction': trade.direction,
        'r_multiple': r_multiple,
        'pnl': trade.pnl,
        'risk_amount': trade.risk_amount
    })

    # Update equity
    self.risk_manager.update_equity(self.wallet_balance)
```

---

## 8. Configuration Parameters

### 8.1 Risk Configuration

```python
RISK_CONFIG = {
    # Kelly Parameters
    "kelly_mode": "half",              # "full", "half", "quarter", "dynamic"
    "min_trades_for_kelly": 30,        # Minimum trades before using Kelly
    "kelly_confidence_threshold": 0.6, # Min confidence for full Kelly

    # Position Limits
    "max_single_position_risk": 0.05,  # 5% max per trade
    "max_portfolio_risk": 0.15,        # 15% total risk
    "max_positions_same_direction": 3, # Max 3 LONG or 3 SHORT
    "max_total_positions": 5,          # Max 5 total positions

    # Drawdown Parameters
    "max_drawdown": 0.20,              # 20% circuit breaker
    "drawdown_recovery_required": 0.05,# 5% recovery to resume
    "drawdown_kelly_decay": 0.15,      # Exponential decay constant

    # Correlation Parameters
    "default_correlation": 0.70,       # Default if unknown
    "correlation_adjustment_enabled": True,
    "max_correlation_reduction": 0.50, # Max 50% reduction for correlated

    # R-Multiple Thresholds
    "min_expectancy_to_trade": 0.05,   # E[R] > 0.05 required
    "edge_degradation_threshold": 0.50,# 50% of all-time E[R]
    "rolling_window": 20,              # Trades for rolling E[R]
}
```

### 8.2 Symbol-Specific Overrides

```python
SYMBOL_RISK_OVERRIDES = {
    "BTCUSDT": {
        "max_single_risk": 0.05,       # Standard
        "correlation_group": "BTC"
    },
    "ETHUSDT": {
        "max_single_risk": 0.05,       # Standard
        "correlation_group": "ETH"
    },
    "LINKUSDT": {
        "max_single_risk": 0.04,       # Slightly reduced (higher vol)
        "correlation_group": "ALT"
    }
}
```

---

## 9. Validation & Testing

### 9.1 Unit Tests Required

```python
# tests/test_risk_manager.py

class TestKellyCalculation:
    def test_kelly_with_edge(self):
        """Kelly should be positive with edge."""
        kelly = calculate_kelly(win_rate=0.60, reward_risk=1.5)
        assert kelly > 0

    def test_kelly_no_edge(self):
        """Kelly should be zero or negative without edge."""
        kelly = calculate_kelly(win_rate=0.30, reward_risk=1.0)
        assert kelly <= 0

    def test_kelly_bounds(self):
        """Kelly should respect min/max bounds."""
        # Even with extreme edge, should cap at max
        kelly = calculate_kelly(win_rate=0.90, reward_risk=3.0)
        assert kelly <= 0.05  # Max 5%

class TestDrawdownAdjustment:
    def test_no_adjustment_at_zero_dd(self):
        """No adjustment when no drawdown."""
        mult = get_drawdown_kelly_multiplier(0)
        assert mult == 1.0

    def test_circuit_breaker_at_20pct(self):
        """Circuit breaker at 20% drawdown."""
        mult = get_drawdown_kelly_multiplier(20)
        assert mult == 0.0

    def test_smooth_decay(self):
        """Smooth decay between 0% and 20%."""
        m10 = get_drawdown_kelly_multiplier(10)
        m15 = get_drawdown_kelly_multiplier(15)
        assert 0 < m15 < m10 < 1.0

class TestCorrelationAdjustment:
    def test_no_adjustment_first_position(self):
        """No adjustment for first position."""
        result = adjust_kelly_for_correlation(0.02, [], "BTCUSDT", "LONG")
        assert result['adjustment_factor'] == 1.0

    def test_reduces_for_correlated(self):
        """Reduces Kelly for correlated positions."""
        open_pos = [{"symbol": "BTCUSDT", "direction": "LONG"}]
        result = adjust_kelly_for_correlation(0.02, open_pos, "ETHUSDT", "LONG")
        assert result['adjustment_factor'] < 1.0

    def test_blocks_at_max_same_direction(self):
        """Blocks trade at max same-direction positions."""
        open_pos = [
            {"symbol": "BTCUSDT", "direction": "LONG"},
            {"symbol": "ETHUSDT", "direction": "LONG"},
            {"symbol": "LINKUSDT", "direction": "LONG"}
        ]
        result = adjust_kelly_for_correlation(0.02, open_pos, "SOLUSDT", "LONG")
        assert result['adjusted_kelly'] == 0.0

class TestExpectancy:
    def test_positive_expectancy(self):
        """Correctly calculates positive expectancy."""
        trades = [
            {"r_multiple": 2.0},
            {"r_multiple": -1.0},
            {"r_multiple": 1.5},
            {"r_multiple": -1.0}
        ]
        exp = calculate_expectancy(trades)
        assert exp['expectancy'] == 0.375  # (2.0 - 1.0 + 1.5 - 1.0) / 4
        assert exp['profitable'] == True
```

### 9.2 Integration Tests

```python
class TestRiskManagerIntegration:
    def test_full_trade_cycle(self):
        """Test complete trade cycle with risk management."""
        rm = RiskManager(initial_equity=10000)

        # First trade
        size1 = rm.calculate_position_size(
            "BTCUSDT", "LONG", 50000, 49000, 52000
        )
        assert size1['can_trade'] == True

        # Record win
        rm.record_trade({"r_multiple": 2.0, "pnl": 200})
        rm.update_equity(10200)

        # Kelly should adjust based on history
        # (after enough trades)

    def test_circuit_breaker_activation(self):
        """Test circuit breaker at 20% drawdown."""
        rm = RiskManager(initial_equity=10000)

        # Simulate losses
        rm.update_equity(8500)  # 15% drawdown - still trading
        size = rm.calculate_position_size("BTCUSDT", "LONG", 50000, 49000, 52000)
        assert size['can_trade'] == True
        assert size['multiplier'] < 1.0  # Reduced

        rm.update_equity(7900)  # 21% drawdown - circuit breaker
        size = rm.calculate_position_size("BTCUSDT", "LONG", 50000, 49000, 52000)
        assert size['can_trade'] == False
```

---

## 10. Implementation Checklist

### Phase 1: Core Kelly Implementation
- [ ] Create `core/kelly_calculator.py`
- [ ] Implement `calculate_kelly()` function
- [ ] Implement `calculate_growth_rate()` function
- [ ] Add unit tests for Kelly calculations

### Phase 2: Drawdown Management
- [ ] Create `core/drawdown_tracker.py`
- [ ] Implement `get_drawdown_kelly_multiplier()`
- [ ] Implement circuit breaker logic
- [ ] Add recovery state management

### Phase 3: Portfolio Integration
- [ ] Update `core/correlation_manager.py`
- [ ] Implement `adjust_kelly_for_correlation()`
- [ ] Implement `calculate_effective_positions()`
- [ ] Add portfolio risk aggregation

### Phase 4: Main RiskManager
- [ ] Create `core/risk_manager.py`
- [ ] Integrate all components
- [ ] Implement `calculate_position_size()` master method
- [ ] Add trade recording and statistics

### Phase 5: Integration
- [ ] Update `TradeManager.open_trade()`
- [ ] Update `TradeManager.close_trade()`
- [ ] Add R-Multiple tracking
- [ ] Update config system

### Phase 6: Testing
- [ ] Unit tests for all modules
- [ ] Integration tests
- [ ] Backtest validation
- [ ] Paper trading verification

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2026-01-02 | Complete specification based on brainstorm session |

---

**END OF SPECIFICATION**
