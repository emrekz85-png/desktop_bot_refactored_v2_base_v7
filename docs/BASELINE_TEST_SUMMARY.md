# Baseline Test - Core Only Configuration
## Expert Panel Phase 1 Implementation

**Date:** January 3, 2026
**Status:** IN PROGRESS
**Goal:** Establish statistical foundation with 100+ trades

---

## What We Did

### 1. Created Core-Only Configuration

**File:** `core/config_core_only.py`

**Changes:**
- **Disabled 7 optional filters** (vs current configuration)
- **Loosened 5 key thresholds** for more signals
- **Kept only 4 essential checks:**
  1. Price direction (above/below SSL)
  2. AlphaTrend momentum confirmation
  3. PBEMA distance > 0.2% (loosened from 0.4%)
  4. Risk/reward > 1.0 (loosened from 2.0)

### 2. Filter Comparison

| Filter | Current (v2.0.0) | Core Only | Change |
|--------|------------------|-----------|--------|
| **skip_body_position** | False (active) | True (disabled) | DISABLED |
| **skip_adx_filter** | False (active) | True (disabled) | DISABLED |
| **skip_overlap_check** | False (active) | True (disabled) | DISABLED |
| **skip_at_flat_filter** | False (active) | True (disabled) | DISABLED |
| **use_ssl_never_lost_filter** | True (active) | False (disabled) | DISABLED |
| **use_market_structure** | True (active) | False (disabled) | DISABLED |
| **use_fvg_bonus** | True (active) | False (disabled) | DISABLED |

### 3. Threshold Comparison

| Parameter | Current | Core Only | Change |
|-----------|---------|-----------|--------|
| **rr** (min risk/reward) | 2.0 | 1.0 | LOOSENED |
| **rsi** (limit) | 70 | 75 | LOOSENED |
| **min_pbema_distance** | 0.4% | 0.2% | LOOSENED (50%) |
| **ssl_touch_tolerance** | 0.3% | 0.5% | LOOSENED (67%) |
| **lookback_candles** | 5 | 7 | EXTENDED (40%) |

---

## Expected Results

### Trade Frequency

| Metric | Current (v2.0.0) | Core Only Target | Multiplier |
|--------|------------------|------------------|------------|
| **Trades per year** | 13 | 80-150 | 7-10x |
| **Active filters** | 7 | 0 | -100% |
| **Statistical validity** | ❌ Meaningless | ✅ Valid | - |

### Performance Expectations

**Current Results:**
- Trades: 13 (statistically insignificant)
- PnL: -$39.90
- Win Rate: 31%
- 95% CI on Win Rate: ±27% (useless!)

**Core Only Expected:**
- Trades: 80-150 (statistically significant)
- PnL: Unknown (possibly negative)
- Win Rate: 35-45% expected
- 95% CI on Win Rate: ±10% (useful!)

**KEY POINT:**
> We EXPECT the PnL to possibly get worse initially. That's OK!
> The goal is statistical foundation, NOT optimization.
> You can't optimize 13 random trades. You CAN optimize 100+ valid trades.

---

## Expert Panel Rationale

### Andreas Clenow (Trade Frequency Expert):
> "13 trades is not a backtest, it's a coin flip session. Get to 100+ trades first,
> then we can talk about optimization. You're building a house on sand right now."

**Problem Identified:**
```
Current Optimization Cycle:
1. Start: 51 trades (baseline)
2. Add filter A → 35 trades ("looks better!")
3. Add filter B → 23 trades ("even better PnL!")
4. Add filter C → 13 trades ("best PnL yet!")

Reality Check:
├─ You filtered out 87% of your data
├─ Result is CURVE-FITTED to 13 lucky trades
└─ Statistical confidence interval: ±27% (useless!)
```

**Solution:**
```
Reverse the Process:
1. Strip to core → 100-150 trades (statistical foundation)
2. Journal signals → identify YOUR patterns
3. Add filters ONE BY ONE → test impact on 100+ trades
4. Keep only filters that IMPROVE Sharpe ratio
```

### Perry Kaufman (Optimization Expert):
> "Optimizing on 13 trades is like flipping a coin 13 times and declaring you've
> discovered a pattern. It's statistically bankrupt."

**Minimum Requirements for Valid Optimization:**
- **Basic significance:** 30 trades minimum
- **Reliable optimization:** 100 trades minimum
- **Production confidence:** 200+ trades ideal
- **Current gap:** Need 7.7x more trades (13 → 100)

---

## Implementation

### Files Created

1. **`core/config_core_only.py`**
   - Core-only configuration
   - Helper functions for comparison
   - Active filter counter

2. **`run_baseline_core_only.py`**
   - Standalone test script
   - Comparison reporting
   - Statistical validity checks

### Usage

```bash
# Full year test (recommended)
python run_baseline_core_only.py

# Quick 3-month test
python run_baseline_core_only.py --quick

# BTC only (fastest validation)
python run_baseline_core_only.py --btc-only --quick

# Custom date range
python run_baseline_core_only.py --start-date 2025-01-01 --end-date 2025-06-30
```

---

## Success Criteria

### Phase 1 COMPLETE if:
- ✅ 100+ trades in 6-month backtest
- ✅ 95% CI on win rate < ±10%
- ✅ All 3 recommended symbols tested
- ✅ Results statistically valid for optimization

### Phase 1 INCOMPLETE if:
- ❌ < 50 trades
- ❌ 95% CI on win rate > ±15%
- ❌ Data quality issues
- ❌ Execution errors

---

## Next Steps After Phase 1

If Phase 1 succeeds (100+ trades):

### Week 2-3: Cognitive Process Mapping
1. **Signal Journaling (2 weeks)**
   - Document EVERY bot signal
   - Record your take/skip decision
   - Note reasons (explicit and gut feeling)
   - Target: 30-50 documented decisions

2. **Pattern Extraction**
   - Analyze journal for recurring themes
   - Identify 3-5 key decision patterns
   - Categorize by importance
   - Examples: HTF alignment, volume confirmation, failed levels

### Week 4: Regime Detection
1. **Implement Regime Detector**
   - RANGING / TRANSITIONAL / TRENDING classification
   - Multi-indicator approach (ADX, ATR percentile, BB width)
   - Validate on historical ranging/trending periods

2. **Regime-Based Filtering**
   - RANGING: 100% position size (optimal)
   - TRANSITIONAL: 50% position size (cautious)
   - TRENDING: 25% or skip (suboptimal)

### Week 5-6: Pattern Implementation
1. **Code Pattern #1** (most important from journal)
2. **Test on 100+ trade sample**
3. **Measure impact:** Win rate, Sharpe ratio, trade frequency
4. **Decision:** Keep if improves Sharpe, discard otherwise
5. **Repeat for Pattern #2, #3**

### Week 7-8: Robust Optimization
1. **Multi-Window Walk-Forward**
   - 5 rolling windows (not just 1)
   - Stability > peak performance
   - Consistency requirement (4/5 windows positive)

2. **Select STABLE config**
   - Not highest return
   - Lowest variance across windows
   - Predictable behavior

### Week 9-10: Validation
1. **Out-of-sample test** (unseen 2024 data)
2. **Stress testing** (various market conditions)
3. **Testnet deployment** ($1000, 20+ trades)

### Week 11+: Live Deployment
1. **Conservative start** (0.5% risk)
2. **Gradual scaling** (0.5% → 1.0% → 1.75%)
3. **Continuous monitoring**
4. **Weekly performance reviews**

---

## Risk Warnings

### Short-Term PnL May Decline
- ✅ EXPECTED: PnL may get worse initially
- ✅ ACCEPTABLE: This is not the goal yet
- ✅ CRITICAL: Need statistical foundation first
- ❌ UNACCEPTABLE: Optimizing 13 random trades

### This Is A Process, Not A Quick Fix
- **Timeline:** 10-12 weeks to profitability
- **Phase 1:** Statistical foundation (Week 1)
- **Phase 2:** Pattern discovery (Week 2-6)
- **Phase 3:** Optimization (Week 7-8)
- **Phase 4:** Validation & deployment (Week 9-12)

### You're Already A Successful Trader
> "You don't need to invent a new strategy. You already have one that works
> (your manual trading proves it). You just need to make it EXPLICIT, test it
> PROPERLY, optimize it ROBUSTLY, and deploy it CAREFULLY."
> - Dr. Andrew Lo

---

## Current Test Status

### Quick BTC Test (3 months)
**Status:** RUNNING
**Purpose:** Validate configuration works
**Expected time:** 3-5 minutes
**Expected trades:** 20-40 (quick period)

### Full Year Test (Next)
**Status:** PENDING
**Purpose:** Establish full statistical baseline
**Expected time:** 15-20 minutes
**Expected trades:** 80-150

---

## Monitoring

Check test progress:
```bash
# View latest results
ls -ltr baseline_core_only_*.json

# View detailed log
tail -f /tmp/claude/*/tasks/*.output
```

Results will be saved to:
- `baseline_core_only_YYYY-MM-DD_YYYY-MM-DD.json`
- Console summary automatically displayed

---

## References

- **Expert Panel Report:** `docs/EXPERT_SPECIFICATION_PANEL_RECOMMENDATIONS.md`
- **Core Config:** `core/config_core_only.py`
- **Test Script:** `run_baseline_core_only.py`
- **Requirements:** `docs/REQUIREMENTS.md`

---

**Document Version:** 1.0
**Last Updated:** January 3, 2026
**Status:** Test in progress
