# Changelog
**Project:** Cryptocurrency Futures Trading Bot
**Last Updated:** January 2, 2026

All notable changes to this project are documented in this file.

---

## Table of Contents

1. [Version History](#version-history)
2. [Failed Experiments](#failed-experiments)
3. [Determinism Fix](#determinism-fix)
4. [Lessons Learned](#lessons-learned)

---

## Version History

### v2.4.0 (January 3, 2026)

#### Expert Panel Recommendations Implementation [NEW]

**Major Feature:** Comprehensive implementation of expert panel recommendations to address overfiltering (13 trades/year issue).

**Source:** Expert Panel Analysis (`docs/EXPERT_PANEL_ANALYSIS_TRADING_BOT.md`)

**Implemented Features:**

| Feature | Expert | Description |
|---------|--------|-------------|
| Volatility Regime | Sinclair | 3-tier system (LOW/NORMAL/HIGH_VOL) |
| Filter Hierarchy | Clenow | Tier 1/2/3 filter system |
| Min Sample Size | Kaufman | 50 trades minimum (was 15) |
| Position Sizing | Sinclair | Regime-adaptive sizing |

**Key Changes:**

1. **Volatility Regime Detection** (`core/indicators.py`)
   - `classify_volatility_regime()` - ATR percentile based classification
   - LOW_VOL: 0.5x position, strict AT confirmation
   - HIGH_VOL: 1.5x position, allow AT lag grace period

2. **Filter Hierarchy** (`strategies/ssl_flow.py`)
   - `filter_tier_level` parameter (1=core, 2=quality, 3=risk)
   - Tier 2 recommended based on backtest results

3. **Position Sizing** (`core/trade_manager.py`)
   - `vol_position_multiplier` from signal debug info
   - Adapts risk based on volatility regime

4. **Optimizer Update** (`core/optuna_optimizer.py`)
   - Minimum trades: 15 → 50 (statistical significance)
   - Trade count sweet spot: 50-100 trades

**Full Year Backtest Results (2025):**

| Metric | Value |
|--------|-------|
| Total Trades | 149 |
| Win Rate | 31.5% |
| E[R] | 0.283 |
| Est. PnL ($1000) | **$737** |

**Recommended Configuration:**
```python
"filter_tier_level": 2
"use_volatility_regime": True
"vol_norm_max_atr": 20.0
symbols = ["BTCUSDT", "ETHUSDT"]
timeframes = ["15m", "1h"]
```

**Documentation:** `docs/EXPERT_PANEL_IMPLEMENTATION.md`

---

### v2.0.0 (January 2, 2026)

#### Kelly Criterion Risk Management System [NEW]

**Major Feature:** Complete mathematically-rigorous risk management system implementation

**Source:** Brainstorm session with user-selected preferences:
- Risk Goal: **Growth Optimization** (Kelly Criterion)
- Scaling: **Fixed Fractional** (scales with equity)
- Correlation: **Correlation Matrix** (effective position calculation)
- Max Drawdown: **20% Moderate** (circuit breaker)
- Losing Streaks: **Kelly Auto-Adjust** (anti-fragile behavior)
- R:R Method: **Per-Trade Actual**

**New Modules Created:**

| File | Lines | Purpose |
|------|-------|---------|
| `core/kelly_calculator.py` | ~260 | Kelly Criterion calculations |
| `core/drawdown_tracker.py` | ~340 | Drawdown tracking & auto-adjustment |
| `core/risk_manager.py` | ~460 | Central risk management coordinator |
| `tests/test_risk_manager.py` | ~350 | 49 comprehensive unit tests |
| `docs/RISK_MANAGEMENT_SPEC.md` | ~1030 | Complete specification document |

**Mathematical Foundation:**

```
Kelly Criterion: f* = W - (1-W)/R

Where:
  f* = Optimal fraction of capital to risk
  W  = Win rate (probability of winning)
  R  = Reward-to-Risk ratio (avg_win / avg_loss)

Growth Rate: G(f) = W × log(1 + f×R) + (1-W) × log(1 - f)
```

**Key Features:**

| Feature | Implementation |
|---------|---------------|
| Kelly Criterion | `calculate_kelly()` with Half-Kelly default |
| Growth Rate | `calculate_growth_rate()` for geometric growth |
| Drawdown Auto-Adjust | Exponential decay: 0%→1.0, 10%→0.70, 20%→0.0 |
| Circuit Breaker | 20% max drawdown stops all trading |
| Recovery Mode | 5% recovery required to resume at 25% size |
| Correlation Adjustment | `adjust_kelly_for_correlation()` reduces size for correlated positions |
| Portfolio Risk | `calculate_portfolio_risk()` with effective positions |
| R-Multiple Tracking | `calculate_r_multiple()` and expectancy calculation |
| Edge Detection | `edge_exists()`, `minimum_win_rate_for_edge()` |

**Drawdown Kelly Multiplier (Calibration):**

| Drawdown | Multiplier | Status |
|----------|------------|--------|
| 0% | 1.00 | NORMAL |
| 5% | 0.85 | NORMAL |
| 10% | 0.70 | CAUTION |
| 15% | 0.50 | DANGER |
| 20% | 0.00 | CIRCUIT_BREAKER |

**Usage Example:**

```python
from core import RiskManager, TradeRecord

# Initialize
rm = RiskManager(initial_equity=10000)

# Calculate position size
sizing = rm.calculate_position_size(
    symbol="BTCUSDT",
    direction="LONG",
    entry_price=50000,
    stop_loss=49000,
    take_profit=52000
)

if sizing.can_trade:
    # Use sizing.risk_amount and sizing.position_size
    print(f"Risk: ${sizing.risk_amount}, Size: {sizing.position_size}")
    print(f"Kelly: {sizing.kelly_fraction:.2%}")

# Record completed trade
rm.record_trade(TradeRecord(
    symbol="BTCUSDT",
    direction="LONG",
    r_multiple=2.0,
    pnl=100,
    risk_amount=50,
    entry_price=50000,
    exit_price=52000,
    entry_time=datetime.now(),
    exit_time=datetime.now()
))

# Update equity
rm.update_equity(10100)

# Get portfolio status
status = rm.get_portfolio_status()
```

**Test Results:** 49/49 tests passing

**Files Changed:**
- `core/kelly_calculator.py` (NEW)
- `core/drawdown_tracker.py` (NEW)
- `core/risk_manager.py` (NEW)
- `core/correlation_manager.py` (UPDATED - added Kelly integration functions)
- `core/__init__.py` (UPDATED - new exports)
- `tests/test_risk_manager.py` (NEW)
- `docs/RISK_MANAGEMENT_SPEC.md` (NEW)

---

### v1.13.0 (December 31, 2025)

#### Lookback Days Increase [SUCCESS]

**Change:** Increased rolling WF lookback from 30 to 60 days

**Results:**
| Metric | lookback=30 | lookback=60 | Delta |
|--------|-------------|-------------|-------|
| PnL | -$270.99 | -$161.99 | +$109 |
| Trades | 50 | 51 | +1 |
| Win Rate | 46% | 41% | -5% |
| Drawdown | $373 | $208 | -$165 |

**Why it works:** Optimizer has more data to find robust configurations

**Files Changed:**
- `runners/rolling_wf_optimized.py`
- `runners/rolling_wf.py`

---

#### Correlation Management [SUCCESS]

**Source:** Hedge Fund Due Diligence Report (Priority 4)

**Problem:** BTC/ETH/LINK correlation 0.85-0.95, 3 positions = only 1.07 effective positions

**Solution:** Correlation-based risk management module

**Features:**
- Max 2 positions in same direction (LONG or SHORT)
- Correlated assets get 50% reduced position size
- Effective position calculation: `N / (1 + (N-1) × avg_correlation)`

**Results:**
```
2 correlated LONG (BTC+ETH) = 1.04 effective positions (48% diversification loss)
3 correlated LONG (BTC+ETH+LINK) = 1.08 effective positions (64% diversification loss)
Mixed directions (LONG+SHORT) = 2.0 effective positions (hedging benefit)
```

**Files Changed:**
- `core/correlation_manager.py` (NEW)

---

### v1.12.0 (December 30, 2025)

#### Filter Simplification Tests

**Test Results (Full Year - 35,000 candles):**

| Filter Removed | Trade Delta | PnL Delta | Win Rate Delta |
|----------------|-------------|-----------|----------------|
| Body Position | +10 | -$17.14 | -0.1% |
| Wick Rejection | +101 | **+$30.08** | +1.0% |
| Both | +115 | +$13.46 | +0.9% |

**Findings:**

1. **Body Position Filter (skip_body_position)**
   - Hedge fund report said 99.9% pass rate - predicted "useless"
   - **WRONG:** Removing causes -$17 loss
   - **Decision:** `skip_body_position=False` (KEEP)

2. **Wick Rejection Filter (skip_wick_rejection)**
   - 68.8% pass rate with "test removal" recommendation
   - **CORRECT:** Filter was HURTING performance
   - Removing gives +$30 improvement + 101 extra trades
   - **Decision:** `skip_wick_rejection=True` (REMOVE)

**Why Wick Rejection was harmful:**
- Designed as bounce signal, but "real bounce" definition too restrictive
- 10% minimum wick ratio too high for real markets
- 20% of profitable trades were blocked by this filter

**Files Changed:**
- `strategies/ssl_flow.py`: Added `skip_wick_rejection` parameter

---

### v1.11.0 (December 30, 2025)

#### Enhanced Regime Filter [DISABLED]

**Source:** Hedge Fund Due Diligence Report (Priority 2)

**Problem:** Strategy only works in TRENDING regimes (H1 loss, H2 profit)

**Attempted Solution:** BTC leader regime filter + multi-factor detection

**Test Results:**
- Full year backtest: -$141 (filter OFF) vs -$200 (filter ON)
- Regime filter **REDUCED** performance (-$59.32 loss)
- 29 trades filtered, but some were WINNING trades
- 1h timeframe profitable trades were blocked

**Decision:** `use_btc_regime_filter=False` permanently

**Why it failed:**
- BTC and altcoins don't always move in sync
- Existing ADX-based regime gating is sufficient
- Additional filtering reduces already scarce trade opportunities

**Files Changed:**
- `core/regime_filter.py` (NEW, but disabled)
- `strategies/ssl_flow.py`

---

### v1.10.1 (December 28, 2025)

#### Look-Ahead Bias Fix [CRITICAL]

**Problem:** ADX regime calculation included current bar

**Before:**
```python
adx_window = df["adx"].iloc[start:index+1]  # WRONG - includes current bar
```

**After:**
```python
adx_window = df["adx"].iloc[start:index]  # CORRECT - excludes current bar
```

**Impact:** Backtest results $157 → $145 (more realistic)

**Why important:** Old result was unachievable in live trading, new result is realistic

**Files Changed:**
- `strategies/ssl_flow.py` line 576

---

### v1.10.0 (December 28, 2025)

#### ROC Momentum Filter [DISABLED]

**Objective:** Block counter-trend entries using Rate of Change

**Problem:** ROC filter conflicts with optimizer

**Issue Flow:**
1. ROC filter active during optimization blocks different signals
2. Optimizer selects different configurations
3. Different configs = completely different trades = worse results

**Test Results:**
| Threshold | PnL | Trades | Win% |
|-----------|-----|--------|------|
| Disabled | $145.39 | 24 | 79% |
| 2.5% | $145.39 | 24 | 79% |
| 1.5% | $145.39 | 24 | 79% |
| 0.5% | $101.22 | 21 | 86% |
| 1.0% | $59.18 | 23 | 70% |

**Solution:**
- `use_roc_filter=False` added to optimizer config grid
- ROC filter only active in production, disabled during backtest/optimization

**Files Changed:**
- `strategies/ssl_flow.py`
- `strategies/router.py`
- `core/optimizer.py`

---

### v43.x (December 27, 2025)

#### Optimizer Determinism Fix [CRITICAL]

**Problem:** Same parameters produced different results ($191 variance)

**Root Cause:** `as_completed()` returned futures in random order

**Solution:**
1. Random seed: `random.seed(42)`, `np.random.seed(42)`
2. Results sorted by config hash (deterministic ordering)
3. Float comparisons use epsilon tolerance
4. Tie-breaking uses config hash

**Validation:** Same test run twice produces identical output:
- Full Year (01-12): $109.15, 58 trades, 81% win rate
- H2 (06-12): $157.10, 25 trades

**Files Changed:**
- `core/optimizer.py`
- `run_rolling_wf_test.py`

---

#### Batch Mode Optimization

**Change:** Parallel processing for multi-symbol tests

**Results:**
- Single run_id for all symbols (folder creation bug fixed)
- ~4x speedup (8 symbols: 32 min → 8 min)

**Usage:** `run_rolling_wf_test.py --batch`

---

#### Weighted Scoring System [EXPERIMENTAL]

**Concept:** Scoring system alternative to AND logic

**Implementation:** 10-point scale with configurable threshold

**Result:** Ineffective - 4 core filters still mandatory:
- `price_above_baseline`
- `at_buyers_dominant`
- `at_is_flat` (NOT)
- `rsi_ok`

Scoring only applies to secondary filters:
- `baseline_touch`
- `wick_rejection`
- `pbema_distance`
- `body_position`
- `overlap`

---

### v42.x (December 2025)

#### PBEMA-SSL Overlap Fix

**Change:** Block trades when PBEMA cloud and SSL baseline overlap

**Implementation:**
```python
OVERLAP_THRESHOLD = 0.005  # 0.5%
baseline_pbema_distance = abs(baseline - pbema_mid) / pbema_mid
is_overlapping = baseline_pbema_distance < OVERLAP_THRESHOLD
```

**Rules:**
- LONG: PBEMA target must be above baseline
- SHORT: PBEMA target must be below baseline
- Overlap detected → signal rejected ("SSL-PBEMA Overlap (No Flow)")

---

#### Minimum SL Distance

**Change:** Prevent noise-driven stop-outs with minimum SL distance

**Thresholds:**
- BTC/ETH: 1.0% minimum SL distance
- Altcoins: 1.5% minimum SL distance

**Modes:**
- `"off"`: Use baseline SL as-is
- `"reject"`: Reject trade if SL too tight
- `"widen"`: Widen SL and adjust TP proportionally

---

#### Exit Profile System

**CLIP Profile:**
- Early partial at 45%: Close 50% position
- Higher hit-rate focus

**RUNNER Profile:**
- Late partial at 70%: Close 33% position
- Let winners run focus

---

#### Circuit Breaker

**Stream-Level:**
- Max loss: -$200
- Max drawdown: $100
- Consecutive full stops: 2

**Global-Level:**
- Daily loss: -$400
- Weekly loss: -$800
- Max drawdown: 20%
- Rolling E[R] edge detection

---

## Failed Experiments

### Summary Table (v1.7.3 - December 28, 2025)

**IMPORTANT:** These changes were tested and made NO DIFFERENCE. Do not retry!

| Experiment | Change | Result | Why Failed |
|------------|--------|--------|------------|
| skip_wick_rejection=True | Skip wick rejection filter | $157.10 (unchanged) | Not bottleneck (68.8% pass rate) |
| trailing_after_partial=True | Trailing SL after partial TP | $157.10 (unchanged) | Optimizer selects same configs |
| min_pbema_distance=0.002 | Reduce PBEMA distance 0.004→0.002 | $157.10 (unchanged) | Low quality signals rejected by optimizer |
| regime_adx_threshold=25 | Increase regime threshold 20→25 | $10.95 (BAD) | Too restrictive, trade count too low |
| risk_per_trade=2.0% | Increase from 1.75% | $78.75 (BAD) | Optimizer selects different config |
| risk_per_trade=2.5% | Increase from 1.75% | $86.75 (BAD) | Optimizer selects different config |
| avoid_hours=[6,14,21,22] | Skip losing hours | $127.71 (BAD) | Blocked trades were actually profitable |
| use_trend_filter=True | SMA + HH/LL trend filter | $87.06 (BAD) | Too restrictive, 25→9 trades |
| use_sweep_detection=True | SL swing proximity check | $0.00 (DISASTER) | Blocked ALL trades |
| use_smart_reentry=True | Fast re-entry after SL | $145.39 (unchanged) | Never triggered: 2h window + price gap |
| use_roc_filter=True | ROC momentum filter | $59.18 (BAD) | Breaks optimizer |
| use_btc_regime_filter=True | BTC leader regime filter | -$200 vs -$141 (BAD) | Blocks profitable 1h trades |

---

### Senior Quant Optimizer Experiments (v1.13.0 - December 31, 2025)

**Context:** After hedge fund recommendations, only 5 trades/year were generated. Senior quant analysis identified "Optimizer-Filter Death Spiral" and recommended changes.

| Experiment | Change | Result | Reason |
|------------|--------|--------|--------|
| hard_min_trades=3 | Reduce from 5 | -$172 (44 trades, 31.8% WR) | Noisy configs accepted, OOS fails |
| hard_min_trades=4 | Reduce from 5 | -$172 (44 trades, 31.8% WR) | Same as 3 - no difference |
| use_ssl_never_lost_filter=False | Disable filter | -$172 (44 trades) | Low quality signals, optimizer breaks |
| use_time_invalidation=False | Disable filter | -$172 (44 trades) | Low quality signals, optimizer breaks |
| **lookback_days=60** | Increase from 30 | **-$162 (51 trades, 41% WR)** | **SUCCESS!** +$109 better than baseline |

**Critical Findings:**

1. **lookback_days=60 is the ONLY successful change**
2. **Disabling filters is BAD** - filters provide quality
3. **Reducing hard_min_trades is BAD** - 5 minimum is necessary
4. **Quant analysis was PARTIALLY WRONG:**
   - "Relax filters" recommendation: WRONG
   - "Increase lookback" recommendation: CORRECT
   - "Reduce hard_min_trades" recommendation: WRONG

---

### Filter Statistics (BTCUSDT-15m, 10000 bars)

```
8_pbema_distance         | Pass:  18.8% | BOTTLENECK (but relaxing doesn't help)
3_at_not_flat            | Pass:  39.8% | RESTRICTIVE
10_pbema_above_baseline  | Pass:  42.1% | RESTRICTIVE
9_wick_rejection         | Pass:  68.8% | LOOSE (changing ineffective)
6_baseline_touch         | Pass:  69.3% | LOOSE
7_body_position          | Pass:  99.9% | LOOSE
```

---

## Determinism Fix

### Problem

Same parameters produced different results ($191 variance between runs)

### Root Cause

`concurrent.futures.as_completed()` returns futures in arbitrary order based on completion time

### Solution

1. **Random Seed Control:**
```python
import random
import numpy as np
random.seed(42)
np.random.seed(42)
```

2. **Deterministic Result Ordering:**
```python
# Sort by config hash, not completion order
results.sort(key=lambda x: hash(str(x.config)))
```

3. **Float Comparison Tolerance:**
```python
def float_equal(a, b, epsilon=1e-9):
    return abs(a - b) < epsilon
```

4. **Tie-Breaking:**
```python
# When metrics equal, use config hash
if float_equal(a.pnl, b.pnl):
    return hash(str(a.config)) - hash(str(b.config))
```

### Validation

Same test run twice with identical output:
- Full Year (01-12): $109.15, 58 trades, 81% WR
- H2 (06-12): $157.10, 25 trades

### Period Test Results (BTC+ETH+LINK)

| Period | PnL | Trades | Win% |
|--------|-----|--------|------|
| H1 (01-06) | -$5.05 | 24 | 87.5% |
| H2 (06-12) | +$157.10 | 25 | 84.0% |
| Full Year | +$109.15 | 58 | 81.0% |

Note: H1+H2 ≠ Full Year due to 30-day lookback gap in June

### Symbol Results (Full Year 2025)

| Symbol | PnL | Trades | Win% |
|--------|-----|--------|------|
| BTCUSDT | +$43.97 | 19 | 78.9% |
| ETHUSDT | +$16.91 | 24 | 79.2% |
| LINKUSDT | +$48.27 | 15 | 86.7% |

---

## Lessons Learned

### Key Insights

1. **Real bottleneck is PBEMA distance (18.8% pass rate)** - but relaxing it doesn't help because low-quality signals get rejected by optimizer

2. **Optimizer filters are MORE restrictive than signal filters** - more signals ≠ better results

3. **Current strategy is already optimal** - parameter relaxation only adds noise

4. **Position sizing increase breaks optimizer** - larger positions = different config selection = worse results

5. **Hour filter has overfitting risk** - historical "bad hours" were profitable out-of-sample

6. **Additional filters block PROFITABLE trades too** - trend/sweep filters prevent losses but also cut gains

7. **SL is already calculated from swing** - sweep detection conflicts with SL calculation logic

8. **Re-entry threshold (0.3%) is too tight** - price doesn't return exactly after liquidity grab

9. **Data freshness affects PnL** - each test fetches new candles, $157 baseline becomes $145 on different day

10. **Lookback increase is the ONLY improvement** - more data = better optimization, not more signals

### Recommended Portfolio

Based on determinism fix validation (January 2025 - December 2025):

**Use ONLY:** BTC, ETH, LINK

**Avoid:** All other symbols either lost money or produced no trades

| Symbol | Status | Notes |
|--------|--------|-------|
| BTCUSDT | **RECOMMENDED** | Best performance |
| ETHUSDT | **RECOMMENDED** | Consistent results |
| LINKUSDT | **RECOMMENDED** | High win rate |
| DOGEUSDT | No Trades | No valid config found |
| SUIUSDT | No Trades | No valid config found |
| LTCUSDT | Losing | Inconsistent |
| BNBUSDT | Losing | Poor performance |
| XRPUSDT | Losing | Too many trades, negative expectancy |
| SOLUSDT | Losing | 0% win rate in some periods |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.1 | 2026-01-02 | Added v2.0.0 Kelly Criterion Risk Management System |
| 1.0 | 2026-01-02 | Initial changelog extracted from CLAUDE.md |

---

**END OF CHANGELOG**
