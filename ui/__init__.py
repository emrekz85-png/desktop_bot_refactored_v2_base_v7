# ==========================================
# UI Package - PyQt5 GUI Components
# ==========================================
# This package contains the GUI components for the trading bot:
# - MainWindow: Main application window with all tabs
# - Workers: QThread workers for background operations
#   - LiveBotWorker: Live trading loop
#   - OptimizerWorker: Parameter optimization
#   - BacktestWorker: Portfolio backtesting
#   - AutoBacktestWorker: Scheduled auto-backtesting
# ==========================================

from .workers import (
    LiveBotWorker,
    OptimizerWorker,
    BacktestWorker,
    AutoBacktestWorker,
)

from .main_window import MainWindow

__all__ = [
    'MainWindow',
    'LiveBotWorker',
    'OptimizerWorker',
    'BacktestWorker',
    'AutoBacktestWorker',
]
