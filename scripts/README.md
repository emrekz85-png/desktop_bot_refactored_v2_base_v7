# Scripts Directory

This directory contains various utility, diagnostic, analysis, and experimental scripts.

## Structure

### `diagnostic/`
Diagnostic scripts for troubleshooting and debugging:
- `diagnose_htf_signals.py` - Higher timeframe signal analysis
- `diagnose_optimizer_issue.py` - Optimizer behavior diagnostics
- `diagnose_optimizer_rejections.py` - Analyze why configs are rejected
- `diagnose_signal_generation.py` - Signal generation debugging

### `analysis/`
Analysis and validation scripts:
- `analyze_htf_filters.py` - HTF filter effectiveness analysis
- `validate_optimizer_fix.py` - Optimizer determinism validation
- `verify_m3_performance.py` - Performance verification for M3 config
- `visualize_trades.py` - Trade visualization and charting

### `experimental/`
Experimental and archived feature tests (may be obsolete):
- `run_combined_filter_test.py` - Combined filter testing
- `run_extended_sample_test.py` - Statistical significance testing
- `run_filter_discovery.py` - Filter combination discovery
- `run_filter_simplification_test.py` - Filter simplification experiments
- `run_grid_optimizer.py` - Grid-based optimizer
- `run_improvement_comparison.py` - A/B testing improvements
- `run_optimizer_diagnostic.py` - Optimizer diagnostic tests
- `run_partial_tp_comparison.py` - Partial TP profile comparison
- `run_regime_filter_backtest.py` - Regime filter backtesting (disabled feature)
- `run_ssl_never_lost_test.py` - SSL Never Lost filter testing
- `run_volatility_adaptive_test.py` - Volatility adaptive position sizing

**Note:** Many experimental scripts test features that have been disabled or proven ineffective. See CLAUDE.md for details on failed experiments.

## Usage

Most scripts can be run directly:

```bash
# Diagnostic
python scripts/diagnostic/diagnose_signal_generation.py

# Analysis
python scripts/analysis/validate_optimizer_fix.py

# Experimental
python scripts/experimental/run_filter_discovery.py --pilot
```

Some scripts may require command-line arguments. Use `--help` for details:

```bash
python scripts/experimental/run_filter_discovery.py --help
```
