# runners/__init__.py
# Runner functions for backtesting and walk-forward testing
# Moved from main file for modularity (v40.5)

from .rolling_wf import (
    run_rolling_walkforward,
    compare_rolling_modes,
    compare_rolling_modes_fast,
    run_quick_rolling_test,
)

from .portfolio import run_portfolio_backtest

__all__ = [
    'run_rolling_walkforward',
    'compare_rolling_modes',
    'compare_rolling_modes_fast',
    'run_quick_rolling_test',
    'run_portfolio_backtest',
]
