# Manual Test Scripts

This directory contains manual test scripts that were previously in the project root. These are standalone test scripts that complement the main pytest test suite in `tests/`.

## Test Scripts

### Correlation & Risk Management
- `test_correlation_manager.py` - Correlation-based position sizing tests
- `test_correlation_impact.py` - Impact analysis of correlation filtering

### Optimizer Testing
- `test_optimizer_minimal.py` - Minimal optimizer functionality test
- `test_scoring_system.py` - Weighted scoring system tests

### Filter Testing
- `test_filter_optimization.py` - Filter optimization experiments
- `test_htf_filter.py` - Higher timeframe filter tests (disabled feature)
- `test_regime_filter.py` - Regime filter tests (disabled feature)
- `test_roc_filter.py` - ROC filter tests (disabled feature)

### Feature Testing
- `test_smart_reentry.py` - Smart reentry logic tests (disabled feature)
- `test_relaxed_rejection.py` - Relaxed filter rejection tests
- `test_signal_detection.py` - Signal detection accuracy tests

### Multi-Symbol Testing
- `test_all_symbols.py` - Test suite across all configured symbols

### Utilities
- `test_helpers.py` - Helper functions for manual tests

## Usage

Run individual test scripts directly:

```bash
python tests/manual/test_correlation_manager.py
python tests/manual/test_optimizer_minimal.py
```

## Note

These are **manual** test scripts, separate from the pytest test suite in `tests/`. Many test features that have been disabled or proven ineffective (see CLAUDE.md for details on failed experiments).

For the main pytest test suite, use:

```bash
pytest                    # Run all pytest tests
pytest tests/test_*.py    # Run specific test modules
```
