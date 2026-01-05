# Production Runner Scripts

This directory contains production-ready utility scripts for strategy analysis and visualization.

## Scripts

### `run_strategy_autopsy.py`
Comprehensive strategy performance analysis and debugging tool.

**Usage:**
```bash
python runners/run_strategy_autopsy.py
```

**Features:**
- Trade-by-trade analysis
- Win/loss pattern identification
- Risk/reward distribution
- Temporal performance analysis

### `run_strategy_sanity_tests.py`
Strategy validation and sanity check suite.

**Usage:**
```bash
python runners/run_strategy_sanity_tests.py
```

**Features:**
- Configuration validation
- Signal generation checks
- Risk management verification
- Live/sim parity tests

### `run_trade_visualizer.py`
Interactive trade visualization and charting tool.

**Usage:**
```bash
python runners/run_trade_visualizer.py
```

**Features:**
- Trade entry/exit visualization
- Indicator overlay
- Multi-timeframe analysis
- Performance metrics overlay

## Note

For main backtesting and optimization, use the root-level scripts:
- `run_backtest.py` - Quick backtest runner
- `run_rolling_wf_test.py` - Rolling walk-forward optimization

For experimental and diagnostic scripts, see `scripts/` directory.
