# Dynamic Risk Scaling Proposal

**Date:** 2026-01-01
**Status:** Proposed
**Problem:** Strategy generates signals but loses money in current market conditions

---

## Executive Summary

The diagnostic analysis revealed a critical insight:
- **BTCUSDT 15m**: 57/75 configs produce 5+ trades, but **100% have negative PnL**
- This is NOT a filter cascade problem - signals ARE being generated
- This IS an **edge quality problem** in current market conditions

The strategy works in trending markets (H2 2025: +$157) but loses in ranging markets.

---

## Root Cause Analysis

### Diagnostic Results (Last 60 Days)

| Stream | Configs 5+ Trades | Negative PnL | Passed |
|--------|-------------------|--------------|--------|
| BTCUSDT 15m | 57 | **57 (100%)** | 0 |
| BTCUSDT 1h | 6 | 0 | 6 |
| ETHUSDT 15m | 24 | 6 | 15 |
| LINKUSDT 15m | 18 | 18 | 0 |

### Key Insight

The optimizer isn't rejecting configs because of:
- ❌ Too few trades (many configs have 5+ trades)
- ❌ Low E[R] (only 4% rejected for this)

**The optimizer is correctly rejecting configs because they LOSE MONEY.**

---

## Proposed Solution: Dynamic Risk Scaling

Instead of trying to force more trades in losing conditions, we should:

### 1. Rolling E[R] Based Position Sizing

```python
# Calculate rolling expected R-multiple from last N trades
rolling_er = calculate_rolling_er(last_20_trades)

# Scale position size based on edge quality
if rolling_er >= 0.15:
    risk_multiplier = 1.0   # Full size
elif rolling_er >= 0.08:
    risk_multiplier = 0.75  # 75% size
elif rolling_er >= 0.03:
    risk_multiplier = 0.50  # Half size
elif rolling_er >= 0.0:
    risk_multiplier = 0.25  # Quarter size
else:
    risk_multiplier = 0.0   # NO TRADE (negative edge)
```

### 2. Adaptive Regime Detection

Rather than binary TRENDING/RANGING, use a continuous regime score:

```python
# Calculate regime quality score
regime_score = calculate_regime_score(adx, bb_width, atr_percentile)

# 0.0 = strong ranging (no trade)
# 0.5 = mixed regime (half size)
# 1.0 = strong trending (full size)
```

### 3. Time-Decay Confidence

Reduce confidence in optimizer config as time passes:

```python
# Days since optimization
days_since_opt = (current_date - optimization_date).days

# Decay factor (halves every 14 days)
confidence_decay = 0.5 ** (days_since_opt / 14)

# Apply to position sizing
final_size = base_size * confidence_decay
```

---

## Implementation Plan

### Phase 1: Rolling E[R] Tracking (CRITICAL)

**File:** `core/trade_manager.py`

```python
class RollingPerformanceTracker:
    """Track rolling performance metrics for adaptive sizing."""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.trade_r_multiples = []  # Circular buffer

    def add_trade(self, r_multiple: float):
        self.trade_r_multiples.append(r_multiple)
        if len(self.trade_r_multiples) > self.window_size:
            self.trade_r_multiples.pop(0)

    def get_rolling_er(self) -> float:
        if not self.trade_r_multiples:
            return 0.0
        return sum(self.trade_r_multiples) / len(self.trade_r_multiples)

    def get_risk_multiplier(self) -> float:
        er = self.get_rolling_er()
        if er >= 0.15:
            return 1.0
        elif er >= 0.08:
            return 0.75
        elif er >= 0.03:
            return 0.50
        elif er >= 0.0:
            return 0.25
        else:
            return 0.0  # Pause trading
```

### Phase 2: Regime Score Integration

**File:** `strategies/ssl_flow.py`

```python
def calculate_regime_score(df: pd.DataFrame, index: int) -> float:
    """Calculate continuous regime quality score (0.0 to 1.0)."""

    # ADX component (0-1)
    adx = df["adx"].iloc[index]
    adx_score = min(1.0, max(0.0, (adx - 15) / 15))  # 15=0, 30=1

    # BB Width component (0-1)
    bb_width = df["bb_width"].iloc[index]
    bb_width_pct = df["bb_width"].iloc[max(0,index-100):index].rank(pct=True).iloc[-1]
    bb_score = bb_width_pct  # Higher width = stronger trend

    # ATR Percentile component (0-1)
    atr_pct = df["atr_percentile"].iloc[index]
    atr_score = min(1.0, max(0.0, (atr_pct - 30) / 40))  # 30=0, 70=1

    # Weighted average
    regime_score = 0.4 * adx_score + 0.3 * bb_score + 0.3 * atr_score

    return regime_score
```

### Phase 3: Config Confidence Decay

**File:** `core/optimizer.py` or `runners/rolling_wf.py`

```python
def calculate_config_confidence(optimization_date: datetime, current_date: datetime) -> float:
    """Calculate time-decay confidence for optimizer config."""
    days_elapsed = (current_date - optimization_date).days
    half_life_days = 14  # Confidence halves every 2 weeks
    return 0.5 ** (days_elapsed / half_life_days)
```

---

## Expected Outcomes

### Conservative Estimate

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Trades/Year | 13 | 20-30 | +54-131% |
| Win Rate | 31% | 35-40% | +4-9% |
| PnL | -$39.90 | -$20 to +$20 | Reduced loss |
| Max DD | $98 | $70-80 | -18-28% |

### Why This Works

1. **Don't fight losing conditions** - Reduce size when edge is weak
2. **Preserve capital** - Smaller losses in bad periods
3. **Maximize winners** - Full size when edge is strong
4. **Adaptive to market** - Automatically adjusts without manual intervention

---

## Risks and Mitigations

### Risk: Over-reduction of position sizes

**Mitigation:** Minimum floor of 0.25x normal size to maintain skin in the game

### Risk: Lag in E[R] calculation

**Mitigation:** Use 20-trade window (approximately 2-4 weeks of data)

### Risk: Complexity increase

**Mitigation:** All logic contained in single class with clear interfaces

---

## Decision Required

**Options:**

1. **Implement Dynamic Risk Scaling** - 2-3 hours development + testing
2. **Continue with current approach** - Accept -$40/year performance
3. **Pause live trading** - Wait for better market conditions

**Recommendation:** Option 1 (Dynamic Risk Scaling)

---

## Next Steps

1. Create `RollingPerformanceTracker` class in `core/trade_manager.py`
2. Integrate with position sizing in `TradingEngine`
3. Add regime score calculation to `ssl_flow.py`
4. Run full-year backtest with new system
5. Compare results with baseline

---

## References

- Diagnostic script: `diagnose_optimizer_rejections.py`
- Volatility test: `run_volatility_adaptive_test.py`
- CLAUDE.md: Known experiments and results
