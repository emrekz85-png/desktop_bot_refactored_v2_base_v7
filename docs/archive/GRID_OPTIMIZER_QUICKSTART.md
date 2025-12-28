# Grid Optimizer Quick Start Guide

## Installation

```bash
# Install scipy (required)
pip install scipy>=1.9.0
```

## Basic Usage

### 1. Quick Mode (Fastest - Recommended First)
```bash
python run_grid_optimizer.py --symbol BTCUSDT --quick
```
**Time:** ~5-10 minutes
**Output:** Top coarse grid results

### 2. Full Mode (Refinement)
```bash
python run_grid_optimizer.py --symbol BTCUSDT --full
```
**Time:** ~15-30 minutes
**Output:** Refined parameters with fine grid

### 3. Robust Mode (Final Validation)
```bash
python run_grid_optimizer.py --symbol BTCUSDT --robust
```
**Time:** ~30-45 minutes
**Output:** Full validation with robustness testing

## Custom Options

```bash
# Different symbol/timeframe
python run_grid_optimizer.py --symbol ETHUSDT --timeframe 1h --full

# Custom date range
python run_grid_optimizer.py --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31 --full

# Control workers (parallel execution)
python run_grid_optimizer.py --symbol BTCUSDT --workers 8 --quick

# Quiet mode
python run_grid_optimizer.py --symbol BTCUSDT --quiet --full
```

## Understanding Results

### Output Location
```
data/grid_search_runs/{run_id}/
├── results.json              # Complete results (machine-readable)
├── top_10.txt                # Top 10 configs (human-readable)
└── significance_report.txt   # Statistical validation
```

### What to Look For

✅ **Good Configuration:**
- Score > 2.0
- Significant: YES (p < alpha)
- Trades ≥ 20
- Stability ≥ 70% (if robustness tested)
- Sharpe ≥ 1.0
- Max Drawdown < 20%

⚠️ **Red Flags:**
- p-value > alpha (not significant)
- Stability < 50% (overfitted)
- Trades < 20 (insufficient data)
- Very high Sharpe with wide CI (lucky run)

## Typical Workflow

```bash
# Step 1: Quick screening
python run_grid_optimizer.py --symbol BTCUSDT --quick

# Step 2: Review top_10.txt
cat data/grid_search_runs/grid_BTCUSDT_*/top_10.txt

# Step 3: Full refinement
python run_grid_optimizer.py --symbol BTCUSDT --full

# Step 4: Robustness validation
python run_grid_optimizer.py --symbol BTCUSDT --robust

# Step 5: Review significance report
cat data/grid_search_runs/grid_BTCUSDT_*/significance_report.txt

# Step 6: Test with walk-forward
python run_rolling_wf_test.py --quick-btc
```

## Python API Example

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

# Run optimizer
optimizer = GridSearchOptimizer(
    symbol='BTCUSDT',
    timeframe='15m',
    data=df,
    verbose=True,
    max_workers=8
)

results = optimizer.run_full_search(quick=False, robust=True)

# Get best config
best = results['fine_results'][0]
print(f"Best config: {best.config.to_dict()}")
```

## Common Issues

### "No significant configurations found"
**Solution:** Increase date range or lower MIN_TRADES_THRESHOLD in code

### "Execution too slow"
**Solution:** Use `--quick` mode first, or increase `--workers`

### "All configs have low robustness"
**Solution:** Normal for low-frequency strategies, focus on significance

## Next Steps

After finding good parameters:

1. Copy config to `best_configs.json`
2. Run walk-forward test: `python run_rolling_wf_test.py --quick-btc`
3. Monitor live performance vs backtest
4. Re-optimize periodically (monthly/quarterly)

## Help & Documentation

- **Full docs:** `/docs/GRID_OPTIMIZER.md`
- **Implementation summary:** `/GRID_OPTIMIZER_SUMMARY.md`
- **Example script:** `/examples/grid_optimizer_example.py`
- **CLI help:** `python run_grid_optimizer.py --help`

## Quick Reference

| Command | Time | Output |
|---------|------|--------|
| `--quick` | 5-10 min | Coarse grid (144 configs) |
| `--full` | 15-30 min | Coarse + fine (~550 configs) |
| `--robust` | 30-45 min | Full + robustness (~570 configs) |

**Recommendation:** Start with `--quick`, then `--full` for best performers.
