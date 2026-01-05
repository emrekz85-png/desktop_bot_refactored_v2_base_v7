# SSL Flow Filter Optimization Guide

## Problem Statement

SSL Flow strategy uses 7 AND filters, each with ~60% pass rate:
- `0.6^7 = 2.8%` total pass rate
- Results in only **9 trades/year** (too few trades)

## Solution: Filter Discovery System

The Filter Discovery system tests all 128 filter combinations (2^7) to find the optimal balance between:
- **Trade Frequency**: More signals = more opportunities
- **Edge Quality**: E[R] (expected R-multiple) > 0.05

---

## Quick Start

### 1. Analysis Mode (Fastest - 2-3 minutes)
Quick diagnostic to identify bottleneck filters:

```bash
python run_filter_discovery.py --analyze --symbol BTCUSDT --timeframe 15m
```

**Output:**
```
Filter Pass Rates:
  adx_filter            - Pass:  450/500 (90.0%) - Lost: 10.0%
  regime_gating         - Pass:  350/500 (70.0%) - Lost: 30.0%  <-- BOTTLENECK
  baseline_touch        - Pass:  225/500 (45.0%) - Lost: 55.0%  <-- MAJOR BOTTLENECK
  pbema_distance        - Pass:  400/500 (80.0%) - Lost: 20.0%
  body_position         - Pass:  300/500 (60.0%) - Lost: 40.0%
  ssl_pbema_overlap     - Pass:  425/500 (85.0%) - Lost: 15.0%
  wick_rejection        - Pass:  275/500 (55.0%) - Lost: 45.0%

Bottleneck filters (most restrictive):
  1. baseline_touch     - 45.0% pass rate (blocks 55.0%)
  2. wick_rejection     - 55.0% pass rate (blocks 45.0%)
  3. body_position      - 60.0% pass rate (blocks 40.0%)
```

**Use Case:** Quickly identify which filters are blocking the most signals.

---

### 2. Pareto Mode (Medium - 1-2 hours)
Find Pareto-optimal filter combinations (best trade-offs):

```bash
python run_filter_discovery.py --pareto --symbol BTCUSDT --timeframe 15m
```

**Output:**
```
Pareto-optimal frontier:
Rank   Trades   E[R]     Win%     Filters
--------------------------------------------------------------------------------
1      45       0.08     52.3%    ADX+REGIME+PBEMA_DIST+OVERLAP
2      32       0.12     58.7%    ADX+REGIME+BASELINE_TOUCH+PBEMA_DIST+OVERLAP
3      24       0.15     62.5%    ADX+REGIME+BASELINE_TOUCH+PBEMA_DIST+BODY+OVERLAP
4      18       0.18     67.2%    ADX+REGIME+BASELINE_TOUCH+PBEMA_DIST+BODY+OVERLAP+WICK  <-- Current baseline
```

**Use Case:** Visualize the trade-off between frequency and quality. Pick the config that matches your risk tolerance.

---

### 3. Sensitivity Mode (Slow - 2-3 hours)
Test parameter robustness for top combinations:

```bash
python run_filter_discovery.py --sensitivity --symbol BTCUSDT --timeframe 15m --top-n 3
```

**Output:**
```
Testing combo #1: ADX+REGIME+PBEMA_DIST+OVERLAP
Baseline: Trades=45, E[R]=0.080

  adx_min:
    10.0000 -> Trades= 52, E[R]=0.065, Win%=48.1%
    15.0000 -> Trades= 45, E[R]=0.080, Win%=52.3%  <-- Baseline
    20.0000 -> Trades= 38, E[R]=0.092, Win%=55.3%
    25.0000 -> Trades= 31, E[R]=0.105, Win%=58.1%

  min_pbema_distance:
    0.0020 -> Trades= 58, E[R]=0.068, Win%=50.0%
    0.0030 -> Trades= 48, E[R]=0.076, Win%=52.1%
    0.0040 -> Trades= 45, E[R]=0.080, Win%=52.3%  <-- Baseline
    0.0050 -> Trades= 39, E[R]=0.088, Win%=54.5%
```

**Use Case:** Verify parameters are robust. If small changes cause large performance swings, the config is overfit.

---

### 4. Comprehensive Report Mode (Slowest - 1-2 hours)
Generates full analysis with actionable recommendations:

```bash
python run_filter_discovery.py --report --symbol BTCUSDT --timeframe 15m
```

**Output:**
```
================================================================================
FILTER OPTIMIZATION REPORT
================================================================================
Symbol: BTCUSDT
Timeframe: 15m
Generated: 2025-12-26 14:30:00
================================================================================

CURRENT BASELINE (all filters ON):
  Trades: 18/year
  E[R]: 0.18
  Win Rate: 67.2%
  PnL: $450.00

BOTTLENECK ANALYSIS:
  Most restrictive filters:
  1. baseline_touch (45.0% pass) - Costs 55.0% of signals
  2. wick_rejection (55.0% pass) - Costs 45.0% of signals
  3. body_position (60.0% pass) - Costs 40.0% of signals

RECOMMENDED CONFIG:
  KEEP: adx_filter, regime_gating, pbema_distance, ssl_pbema_overlap
  REMOVE: baseline_touch, body_position, wick_rejection

  Expected:
    Trades: 45/year (+150%)
    E[R]: 0.08

PARETO-OPTIMAL ALTERNATIVES:
  Config 1: High Quality (E[R]=0.18, Trades=18)
  Config 2: Balanced (E[R]=0.12, Trades=32)  <-- RECOMMENDED
  Config 3: High Frequency (E[R]=0.08, Trades=45)

WALK-FORWARD VALIDATION:
  Train E[R]: 0.12
  WF E[R]: 0.10
  Ratio: 0.83 (PASS - not overfit)

================================================================================
```

**Use Case:** Get final recommendations for production deployment.

---

## CLI Reference

### Modes

| Mode | Command | Speed | Purpose |
|------|---------|-------|---------|
| **Analysis** | `--analyze` | Fast (2-3 min) | Identify bottleneck filters |
| **Pareto** | `--pareto` | Medium (1-2 hrs) | Find optimal filter combinations |
| **Sensitivity** | `--sensitivity` | Slow (2-3 hrs) | Test parameter robustness |
| **Report** | `--report` | Slow (1-2 hrs) | Comprehensive analysis + recommendations |
| **Pilot** | `--pilot` | Slow (1-2 hrs) | Full discovery on BTCUSDT-15m |
| **Full** | `--full` | Very slow (days) | Discovery on all symbols/timeframes |
| **Validate** | `--validate` | Fast (2-3 min) | Test combo on holdout data |

### Common Arguments

```bash
--symbol BTCUSDT          # Trading symbol (default: BTCUSDT)
--timeframe 15m           # Timeframe (default: 15m)
--candles 30000           # Number of candles (default: 30000)
--top-n 3                 # Top N combos for sensitivity (default: 3)
--no-parallel             # Disable parallel processing (debugging)
--workers 8               # Max parallel workers (default: auto)
```

---

## Understanding Results

### Filter Pass Rates
- **High pass rate (>80%)**: Filter is not very restrictive
- **Medium pass rate (50-80%)**: Filter is moderately selective
- **Low pass rate (<50%)**: **BOTTLENECK** - filter blocks most signals

### Pareto-Optimal Frontier
A combination is Pareto-optimal if NO other combination has:
- **BOTH** higher E[R] **AND** more trades

Example:
```
Config A: E[R]=0.10, Trades=30  <-- Pareto-optimal
Config B: E[R]=0.08, Trades=45  <-- Pareto-optimal (more trades, lower E[R])
Config C: E[R]=0.09, Trades=25  <-- NOT Pareto-optimal (dominated by A)
```

### Walk-Forward Validation
- **Ratio >= 0.50**: Config is robust (not overfit)
- **Ratio < 0.50**: Config may be overfit to training data

Example:
```
Train E[R]: 0.15
WF E[R]: 0.12
Ratio: 0.80  <-- PASS (12/15 = 0.80 >= 0.50)
```

---

## Success Criteria

Before deploying a filter configuration, ensure:

1. **E[R] >= 0.05** - Positive edge
2. **Trades >= 20/year** - At least 2x baseline
3. **WF/Train ratio >= 0.50** - Not overfit
4. **Win rate >= 40%** - Reasonable win rate

---

## Workflow

### Step 1: Quick Diagnostic (5 min)
```bash
python run_filter_discovery.py --analyze --symbol BTCUSDT --timeframe 15m
```
- Identify bottleneck filters
- Understand which filters are too strict

### Step 2: Find Optimal Combinations (1-2 hrs)
```bash
python run_filter_discovery.py --pareto --symbol BTCUSDT --timeframe 15m
```
- Find Pareto-optimal configurations
- Choose based on your risk preference:
  - **Conservative**: Higher E[R], fewer trades
  - **Aggressive**: Lower E[R], more trades
  - **Balanced**: Middle ground

### Step 3: Validate Robustness (2-3 hrs)
```bash
python run_filter_discovery.py --sensitivity --symbol BTCUSDT --timeframe 15m --top-n 3
```
- Test if parameters are stable
- Ensure config is not overfit

### Step 4: Final Validation (5 min)
```bash
python run_filter_discovery.py --validate 0 --results data/filter_discovery_runs/pareto_BTCUSDT_15m_20251226_143000/results.json
```
- Test best config on holdout data (never seen during search)
- Verify E[R] holds up on fresh data

### Step 5: Deploy
Update `strategies/ssl_flow.py` or `core/config.py` with chosen filter configuration.

---

## Output Files

All results are saved to `data/filter_discovery_runs/{run_id}/`:

```
data/filter_discovery_runs/report_BTCUSDT_15m_20251226_143000/
├── results.json                   # All 128 combinations with full metrics
├── top_10.txt                     # Human-readable top 10 report
├── filter_pass_rates.json         # Individual filter pass rates
├── pareto_optimal.json            # Pareto-optimal combinations
├── comprehensive_report.txt       # Final recommendations
└── holdout_validation_combo_0.txt # Holdout validation (if run)
```

---

## Examples

### Find Best Config for Day Trading (More Trades)
```bash
# Priority: High frequency (30+ trades/year)
python run_filter_discovery.py --report --symbol BTCUSDT --timeframe 15m

# Look for configs with:
# - Trades >= 30
# - E[R] >= 0.08
# - WF ratio >= 0.50
```

### Find Best Config for Swing Trading (Higher Quality)
```bash
# Priority: High E[R] (0.15+)
python run_filter_discovery.py --report --symbol BTCUSDT --timeframe 1h

# Look for configs with:
# - E[R] >= 0.15
# - Trades >= 15
# - WF ratio >= 0.60
```

### Test Specific Filter Hypothesis
Example: "Does removing wick_rejection improve trade count without hurting E[R]?"

```bash
# 1. Run analysis to see current wick_rejection pass rate
python run_filter_discovery.py --analyze --symbol BTCUSDT --timeframe 15m

# 2. Run full discovery and compare configs with/without wick_rejection
python run_filter_discovery.py --pilot

# 3. Check results.json - filter by wick_rejection=True vs False
```

---

## Advanced: Manual Filter Testing

You can also test filter combinations programmatically:

```python
from core.filter_discovery import FilterDiscoveryEngine, FilterCombination
from core import TradingEngine, calculate_indicators
import pandas as pd

# Fetch data
df = TradingEngine.get_historical_data_pagination("BTCUSDT", "15m", total_candles=30000)
df = calculate_indicators(df, timeframe="15m")

# Create engine
engine = FilterDiscoveryEngine(
    symbol="BTCUSDT",
    timeframe="15m",
    data=df,
    baseline_trades=9
)

# Test specific combination
combo = FilterCombination(
    adx_filter=True,
    regime_gating=True,
    baseline_touch=False,  # Disable this filter
    pbema_distance=True,
    body_position=False,   # Disable this filter
    ssl_pbema_overlap=True,
    wick_rejection=False,  # Disable this filter
)

result = engine.evaluate_combination(combo)
print(f"Trades: {result.train_trades}")
print(f"E[R]: {result.train_expected_r:.3f}")
print(f"Win%: {result.train_win_rate:.1%}")
```

---

## Troubleshooting

### "No baseline signals found!"
- Data may not have enough history (need warmup period)
- Try increasing `--candles` to 50000

### "No valid combinations found!"
- Criteria too strict (min_trades or min_expected_r)
- Lower thresholds in `find_pareto_optimal_combinations()`

### Discovery is too slow
- Use `--no-parallel` to debug (single-threaded)
- Reduce `--candles` to 10000 for faster testing
- Start with `--analyze` mode for quick diagnostics

### WF ratio is always < 0.50 (overfit)
- Data may be too small (increase `--candles`)
- Try different timeframes (higher timeframes are more stable)
- Consider using simpler filters (fewer filters = less overfitting)

---

## References

- **Filter Discovery Engine**: `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/core/filter_discovery.py`
- **CLI Runner**: `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/run_filter_discovery.py`
- **SSL Flow Strategy**: `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/strategies/ssl_flow.py`

---

## Support

For issues or questions, check:
1. `CLAUDE.md` - Project documentation
2. `run_filter_discovery.py --help` - CLI help
3. Error logs in `error_log.txt`
