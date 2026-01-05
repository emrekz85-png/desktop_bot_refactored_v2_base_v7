# Grid Search Optimizer Documentation

## Overview

The Grid Search Optimizer (`core/grid_optimizer.py`) implements a hierarchical combinatorial grid search for SSL Flow trading strategy parameters with statistical validation and robustness testing.

This optimizer is designed to prevent overfitting through:
- **Multiple comparison correction** (Bonferroni)
- **Statistical significance testing** (t-tests)
- **Bootstrap confidence intervals** (Sharpe ratio)
- **Parameter sensitivity analysis** (perturbation testing)

---

## Quick Start

### Installation

```bash
# Install scipy (required for statistical tests)
pip install scipy>=1.9.0
```

### Basic Usage

```bash
# Quick mode: Coarse grid only (~5-10 min)
python run_grid_optimizer.py --symbol BTCUSDT --quick

# Full mode: Coarse + fine grid (~15-30 min)
python run_grid_optimizer.py --symbol BTCUSDT --full

# Robust mode: Full + robustness testing (~30-45 min)
python run_grid_optimizer.py --symbol BTCUSDT --robust
```

### Custom Options

```bash
# Specific timeframe and date range
python run_grid_optimizer.py \
    --symbol BTCUSDT \
    --timeframe 1h \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --full

# Control parallel workers
python run_grid_optimizer.py --symbol ETHUSDT --workers 4 --quick

# Quiet mode (suppress progress output)
python run_grid_optimizer.py --symbol SOLUSDT --quiet --full
```

---

## Methodology

### Three-Phase Approach

#### Phase 1: Coarse Grid Search

**Purpose:** Fast screening of parameter space

**Parameters tested:**
- `rsi_threshold`: [60, 65, 70, 75]
- `adx_threshold`: [20, 25, 30]
- `rr_ratio`: [1.5, 2.0, 2.5, 3.0]
- `regime_adx_avg`: [15, 20, 25]
- `at_active`: [True] (mandatory for SSL Flow)

**Total combinations:** 4 × 3 × 4 × 3 = 144 configurations

**Output:** Top performers ranked by robust score

---

#### Phase 2: Fine Grid Search

**Purpose:** Refinement around top performers

**Method:**
- Takes top 5 configurations from coarse search
- Generates ±10% variations around each parameter
- Tests 3 values per parameter (low, base, high)

**Example:**
```
If coarse best = RSI 70:
  Fine grid tests: RSI [63, 70, 77]
```

**Total combinations:** ~5 × 3^4 = ~400 configurations

**Output:** Refined top performers with tighter parameter bounds

---

#### Phase 3: Robustness Testing

**Purpose:** Validate parameter stability

**Method:**
- Perturbs each parameter by ±10%
- Tests 8 neighbors per config (4 params × 2 directions)
- Calculates stability percentage (profitable neighbors / total)

**Good config criteria:**
- Stability ≥ 70% (most neighbors profitable)
- Parameters have wide "profitable zone"
- Not overfitted to exact values

---

### Scoring Function

The robust score combines multiple metrics to prevent overfitting:

```python
base_score = (
    sharpe_ratio * 0.30 +           # Risk-adjusted returns
    profit_factor * 0.20 +          # Gross profit / gross loss
    (1 - max_drawdown) * 0.20 +     # Drawdown control
    stability * 0.20 +              # Return consistency
    trade_frequency * 0.10          # Sample size adequacy
)

penalties = (
    trade_penalty +                 # Trades < 30
    complexity_penalty              # Active filters penalty
)

final_score = base_score - penalties
```

**Key features:**
- **Trade penalty:** Progressive penalty for < 30 trades
- **Complexity penalty:** 2% per non-default parameter (prevents overfitting)
- **Stability bonus:** Consistent returns weighted higher than lucky runs

---

### Statistical Validation

#### 1. Bonferroni Correction

**Problem:** Testing 144+ configurations inflates false positive risk

**Solution:** Adjust significance level
```
alpha = 0.05 / n_tests
```

**Example:**
- 144 tests → alpha = 0.05 / 144 = 0.000347
- Stricter threshold prevents false discoveries

#### 2. t-Statistic

**Test:** Is E[R] significantly different from zero?

**Formula:**
```
t = mean(R) / (std(R) / sqrt(n))
```

**Interpretation:**
- t > 2.0: Strong evidence of edge
- p < 0.05/n: Statistically significant

#### 3. Bootstrap Confidence Intervals

**Method:** 1000 bootstrap samples

**Output:** 95% confidence interval for Sharpe ratio
```
Sharpe: 1.42 [95% CI: 0.98, 1.85]
```

**Interpretation:**
- Narrow CI: Consistent performance
- Wide CI: High variance, unstable

---

## Output Files

Results are saved to `data/grid_search_runs/{run_id}/`:

### 1. `results.json`

Complete results in JSON format:
```json
{
  "run_id": "grid_BTCUSDT_15m_20241226_143022",
  "symbol": "BTCUSDT",
  "timeframe": "15m",
  "coarse_results": [...],
  "fine_results": [...],
  "robustness_results": {...}
}
```

### 2. `top_10.txt`

Human-readable top 10 report:
```
==================================================================
TOP 10 PARAMETER COMBINATIONS
Symbol: BTCUSDT | Timeframe: 15m
==================================================================

#1 | Score: 2.4567

  Parameters:
    RSI Threshold:      70
    ADX Threshold:      25
    RR Ratio:           2.0
    Regime ADX Avg:     20

  Performance:
    Total PnL:          $450.23
    Trade Count:        45
    Win Rate:           62.2%
    Profit Factor:      2.15
    Sharpe Ratio:       1.42
    Max Drawdown:       12.5%
    E[R]:               0.145

  Statistical Validation:
    Significant:        YES (p=0.000012)
    t-statistic:        4.23
    Sharpe 95% CI:      [0.98, 1.85]

  Robustness:
    Stable Neighbors:   75.0%
```

### 3. `significance_report.txt`

Statistical significance analysis:
```
==================================================================
STATISTICAL SIGNIFICANCE REPORT
Symbol: BTCUSDT | Timeframe: 15m
==================================================================

Total Configurations Tested: 544
Statistically Significant:   12 (2.2%)

Bonferroni Correction Applied: Yes
Significance Level (alpha):     0.05 / 544 = 0.000092

==================================================================
SIGNIFICANT CONFIGURATIONS
==================================================================

#1 | p-value: 0.000012 | t-stat: 4.23
------------------------------------------------------------------
  RSI=70, ADX=25, RR=2.0, RegimeADX=20
  PnL: $450.23 | E[R]: 0.145 | Trades: 45
  Sharpe: 1.42 [95% CI: 0.98, 1.85]
```

---

## Interpreting Results

### Good Configuration Indicators

✅ **Statistical significance:**
- p-value < alpha (Bonferroni corrected)
- t-statistic > 2.0
- Narrow Sharpe CI

✅ **Robustness:**
- Stability ≥ 70%
- Profitable across parameter variations
- Not overfitted to exact values

✅ **Performance:**
- E[R] > MIN_EXPECTANCY_R_MULTIPLE
- Trade count ≥ MIN_TRADES_THRESHOLD
- Low drawdown (< 20%)

### Red Flags

⚠️ **Potential overfitting:**
- p-value > alpha (not significant)
- Stability < 50%
- Very high Sharpe with wide CI

⚠️ **Insufficient data:**
- Trade count < 20
- Wide confidence intervals
- High parameter sensitivity

⚠️ **Lucky runs:**
- High score, low robustness
- Significant but unstable
- Narrow profitable zone

---

## Parameter Recommendations

Based on grid search results, choose parameters that:

1. **Pass statistical significance** (p < alpha)
2. **Show robustness** (stability ≥ 70%)
3. **Have adequate samples** (trades ≥ 20)
4. **Balance complexity** (fewer active filters = less overfitting)

### Example Decision Process

```
Config A: Score=2.8, p=0.00001, Stability=85%, Trades=52
Config B: Score=3.1, p=0.00450, Stability=45%, Trades=18

Choose Config A:
  ✓ Statistically significant
  ✓ High robustness
  ✓ Adequate sample size
  ✗ Slightly lower score (acceptable)

Reject Config B:
  ✓ Higher score (but...)
  ✗ Borderline significance
  ✗ Low robustness (overfitted)
  ✗ Small sample size
```

---

## Advanced Usage

### Python API

```python
from core.grid_optimizer import GridSearchOptimizer
from core import TradingEngine

# Fetch data
df = TradingEngine.get_historical_data_pagination(
    'BTCUSDT', '15m',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Calculate indicators
df = TradingEngine.calculate_indicators(df, timeframe='15m')

# Initialize optimizer
optimizer = GridSearchOptimizer(
    symbol='BTCUSDT',
    timeframe='15m',
    data=df,
    verbose=True,
    max_workers=8
)

# Run hierarchical search
results = optimizer.run_full_search(quick=False, robust=True)

# Access results
best_config = results['fine_results'][0]
print(f"Best score: {best_config.robust_score:.4f}")
print(f"Parameters: {best_config.config.to_dict()}")
```

### Custom Grid

Modify `COARSE_GRID` in `core/grid_optimizer.py`:

```python
COARSE_GRID = {
    'rsi_threshold': [60, 65, 70, 75, 80],  # Added 80
    'adx_threshold': [15, 20, 25, 30, 35],   # More values
    'rr_ratio': [1.5, 2.0, 2.5, 3.0],
    'regime_adx_avg': [15, 20, 25],
    'at_active': [True],
}
```

---

## Performance Considerations

### Execution Time

**Coarse grid (144 configs):**
- 1 year data: ~5-10 minutes (8 workers)
- 2 year data: ~10-20 minutes

**Fine grid (~400 configs):**
- 1 year data: ~15-30 minutes
- 2 year data: ~30-60 minutes

**Robustness (top 3, 24 neighbors):**
- 1 year data: ~5-10 minutes
- 2 year data: ~10-20 minutes

**Total (full robust mode):**
- 1 year data: ~25-50 minutes
- 2 year data: ~50-100 minutes

### Optimization Tips

1. **Use --quick first:** Screen parameters quickly
2. **Increase workers:** More CPU cores = faster
3. **Reduce date range:** 6 months may be sufficient
4. **Run overnight:** Robust mode for final validation

---

## Troubleshooting

### Common Issues

#### "No significant configurations found"

**Cause:** All configs rejected by Bonferroni correction

**Solutions:**
- Increase data range (more trades)
- Lower MIN_TRADES_THRESHOLD
- Check if strategy has edge on this symbol/timeframe

#### "All configs have low robustness"

**Cause:** Parameters are overfitted to exact values

**Solutions:**
- Use coarser grid (fewer values)
- Reduce complexity penalty
- Accept lower robustness for low-frequency strategies

#### "Long execution time"

**Cause:** Large grid × long data range

**Solutions:**
- Use --quick mode first
- Reduce date range
- Increase --workers
- Use faster machine

---

## References

- **Bonferroni Correction:** https://en.wikipedia.org/wiki/Bonferroni_correction
- **Bootstrap Methods:** Efron & Tibshirani (1993)
- **Walk-Forward Analysis:** Pardo (2008)

---

## Support

For issues or questions:
1. Check existing runs in `data/grid_search_runs/`
2. Review `significance_report.txt` for statistical details
3. Adjust MIN_TRADES_THRESHOLD if needed
4. Contact: See CLAUDE.md for project info
