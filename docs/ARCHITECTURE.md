# Architecture Document
**Project:** Cryptocurrency Futures Trading Bot
**Version:** 1.0
**Date:** January 2, 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [File Structure](#2-file-structure)
3. [Module Dependencies](#3-module-dependencies)
4. [Configuration System](#4-configuration-system)
5. [Build and Setup](#5-build-and-setup)
6. [Running the Application](#6-running-the-application)
7. [Development Guidelines](#7-development-guidelines)
8. [Quick Reference](#8-quick-reference)

---

## 1. System Overview

### 1.1 Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| GUI Framework | PyQt5 + PyQtWebEngine |
| Data Processing | pandas, numpy, pandas_ta |
| Exchange API | python-binance |
| Notifications | python-telegram-bot |
| Testing | pytest |
| Async | asyncio, ThreadPoolExecutor |

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   PyQt5 GUI     │  │   CLI/Headless  │  │  Google Colab   │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼──────────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Trading Engine                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Signal Router  │──│  Trade Manager  │──│ Circuit Breaker │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────────────┘  │
└───────────┼─────────────────────┼────────────────────────────────┘
            │                     │
            ▼                     ▼
┌─────────────────────┐  ┌─────────────────────────────────────────┐
│     Strategies      │  │              Core Modules                │
│  ┌───────────────┐  │  │  ┌───────────┐  ┌───────────────────┐  │
│  │   SSL Flow    │  │  │  │ Indicators│  │   Config Loader   │  │
│  └───────────────┘  │  │  └───────────┘  └───────────────────┘  │
│  ┌───────────────┐  │  │  ┌───────────┐  ┌───────────────────┐  │
│  │Keltner Bounce │  │  │  │ Optimizer │  │     Utilities     │  │
│  │  (disabled)   │  │  │  └───────────┘  └───────────────────┘  │
│  └───────────────┘  │  │                                         │
└─────────────────────┘  └─────────────────────────────────────────┘
            │                              │
            ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      External Services                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Binance API   │  │    Telegram     │  │   Data Storage  │  │
│  │  REST + WebSocket│  │  Notifications  │  │  CSV + JSON     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Modular package structure | Maintainability, testability |
| Strategy isolation | Easy addition/removal of strategies |
| Walk-forward optimization | Prevent overfitting |
| R-Multiple risk management | Consistent position sizing |
| Circuit breaker pattern | Capital protection |

---

## 2. File Structure

```
desktop_bot_refactored_v2_base_v7/
│
├── desktop_bot_refactored_v2_base_v7.py  # Main application (~8800 lines)
│
├── core/                                  # Core package
│   ├── __init__.py                        # Package exports
│   ├── config.py                          # Constants, trading config
│   ├── config_loader.py                   # Load/save optimized configs
│   ├── indicators.py                      # Technical indicator calculations
│   ├── trade_manager.py                   # BaseTradeManager & SimTradeManager
│   ├── trading_engine.py                  # TradingEngine class
│   ├── binance_client.py                  # Binance API client with retry
│   ├── telegram.py                        # Telegram notifications
│   ├── utils.py                           # Helper functions
│   ├── logging_config.py                  # Centralized logging
│   ├── optimizer.py                       # Walk-forward optimizer
│   ├── correlation_manager.py             # Correlation-based risk management
│   ├── regime_filter.py                   # Market regime detection
│   ├── safe_eval.py                       # Safe expression evaluation
│   └── safe_pickle.py                     # Safe pickle loading
│
├── strategies/                            # Strategy implementations
│   ├── __init__.py                        # Strategy registry
│   ├── base.py                            # SignalResult, STRATEGY_MODES
│   ├── router.py                          # Signal routing logic
│   ├── ssl_flow.py                        # SSL Flow strategy [ACTIVE]
│   └── keltner_bounce.py                  # Keltner Bounce [DISABLED]
│
├── ui/                                    # PyQt5 GUI components
│   ├── __init__.py                        # Package exports
│   ├── main_window.py                     # MainWindow class
│   └── workers.py                         # QThread workers
│
├── runners/                               # CLI runners
│   ├── rolling_wf.py                      # Rolling walk-forward runner
│   ├── rolling_wf_optimized.py            # Optimized version
│   └── run_strategy_autopsy.py            # Strategy analysis
│
├── tests/                                 # Pytest test suite
│   ├── __init__.py
│   ├── conftest.py                        # Pytest fixtures
│   ├── fixtures/                          # Test data fixtures
│   ├── test_config.py                     # Configuration tests
│   ├── test_indicators.py                 # Indicator tests
│   ├── test_parity.py                     # Live/sim parity tests
│   ├── test_risk.py                       # Risk management tests
│   ├── test_signals.py                    # Signal detection tests
│   └── test_trade_manager.py              # Trade manager tests
│
├── docs/                                  # Documentation
│   ├── REQUIREMENTS.md                    # Requirements specification
│   ├── ARCHITECTURE.md                    # This file
│   └── CHANGELOG.md                       # Version history
│
├── data/                                  # Data directory (gitignored)
│
├── scripts/                               # Utility scripts
├── tools/                                 # Development tools
├── examples/                              # Usage examples
│
├── run_backtest.py                        # Quick backtest runner
├── run_rolling_wf_test.py                 # Rolling WF test
├── requirements.txt                       # Python dependencies
├── pytest.ini                             # Pytest configuration
├── CLAUDE.md                              # AI assistant guide
│
├── start_bot.sh / start_bot.bat           # Start scripts
├── start_backtest.sh / start_backtest.bat # Backtest scripts
└── .gitignore
```

### 2.1 Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `core/config.py` | All constants, trading config, symbol lists |
| `core/indicators.py` | Technical indicator calculations (HMA, EMA, RSI, etc.) |
| `core/trade_manager.py` | Trade lifecycle (open, update, close) |
| `core/optimizer.py` | Walk-forward parameter optimization |
| `strategies/ssl_flow.py` | SSL Flow signal generation logic |
| `strategies/router.py` | Route signals to appropriate strategy |
| `ui/main_window.py` | PyQt5 GUI implementation |

---

## 3. Module Dependencies

### 3.1 Dependency Graph

```
                    ┌─────────────┐
                    │   config    │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ indicators │  │   utils    │  │  telegram  │
    └──────┬─────┘  └──────┬─────┘  └────────────┘
           │               │
           ▼               ▼
    ┌────────────────────────────┐
    │      trade_manager         │
    └──────────────┬─────────────┘
                   │
           ┌───────┼───────┐
           ▼       ▼       ▼
    ┌──────────┐ ┌─────┐ ┌──────────┐
    │ ssl_flow │ │router│ │ optimizer│
    └──────────┘ └─────┘ └──────────┘
                   │
                   ▼
           ┌───────────────┐
           │trading_engine │
           └───────────────┘
```

### 3.2 Import Dependencies

| Module | Dependencies |
|--------|--------------|
| `core/config.py` | None (leaf node) |
| `core/indicators.py` | config, pandas_ta |
| `core/trade_manager.py` | config, utils, telegram |
| `strategies/ssl_flow.py` | strategies/base |
| `strategies/router.py` | ssl_flow, keltner_bounce |
| `core/optimizer.py` | config, indicators, trade_manager |
| `ui/main_window.py` | core/*, strategies/* |

---

## 4. Configuration System

### 4.1 Configuration Hierarchy

All configuration lives in `core/config.py` (Single Source of Truth):

```python
# Symbols
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", ...]

# Timeframes
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "12h", "1d"]

# Trading Configuration
TRADING_CONFIG = {
    "initial_balance": 2000.0,
    "leverage": 10,
    "risk_per_trade_pct": 0.0175,  # 1.75%
    "max_portfolio_risk_pct": 0.05,  # 5%
    ...
}
```

### 4.2 Strategy Configuration

`DEFAULT_STRATEGY_CONFIG` defines strategy parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rr` | Risk/Reward ratio | 1.0 |
| `rsi` | RSI threshold | 70 |
| `at_active` | AlphaTrend filter | True |
| `use_trailing` | Trailing stop | False |
| `strategy_mode` | Active strategy | "ssl_flow" |
| `exit_profile` | Exit profile | "clip" |
| `sl_validation_mode` | SL validation | "off" |

### 4.3 Environment Variables

Recommended for sensitive data:

```bash
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_secret"
```

### 4.4 Config Files

| File | Purpose |
|------|---------|
| `config.json` | Runtime configuration |
| `best_configs.json` | Optimized configs per symbol/timeframe |

---

## 5. Build and Setup

### 5.1 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### 5.2 Installation

```bash
# Clone repository
git clone <repository_url>
cd desktop_bot_refactored_v2_base_v7

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 5.3 Dependencies

Key packages in `requirements.txt`:

```
pandas>=1.3.0
numpy>=1.20.0
pandas_ta>=0.3.0
python-binance>=1.0.0
PyQt5>=5.15.0
PyQtWebEngine>=5.15.0
python-telegram-bot>=13.0
pytest>=6.0.0
```

### 5.4 Verification

```bash
# Verify syntax
python -m py_compile desktop_bot_refactored_v2_base_v7.py

# Run tests
pytest

# Quick test
python run_backtest.py --quick
```

---

## 6. Running the Application

### 6.1 GUI Mode (Desktop)

```bash
python desktop_bot_refactored_v2_base_v7.py
```

Requires PyQt5 and PyQtWebEngine. Opens graphical interface for:
- Symbol/timeframe selection
- Live trading control
- Backtest execution
- Results visualization

### 6.2 Headless/CLI Mode

```bash
python desktop_bot_refactored_v2_base_v7.py --headless
```

For servers without display. Supports:
- Automated backtesting
- Telegram notifications
- Background operation

### 6.3 CLI Options

```bash
python desktop_bot_refactored_v2_base_v7.py --headless \
    --symbols BTCUSDT ETHUSDT \
    --timeframes 5m 15m 1h \
    --candles 30000 \
    --no-optimize
```

| Option | Description |
|--------|-------------|
| `--headless` | Run without GUI |
| `--symbols` | Symbols to trade |
| `--timeframes` | Timeframes to monitor |
| `--candles` | Number of historical candles |
| `--no-optimize` | Skip optimization |

### 6.4 Google Colab

```python
from desktop_bot_refactored_v2_base_v7 import run_cli_backtest, colab_quick_test

# Full backtest
results = run_cli_backtest(
    symbols=['BTCUSDT', 'ETHUSDT'],
    timeframes=['5m', '15m', '1h'],
    candles=30000,
    optimize=True
)

# Quick single-stream test
results = colab_quick_test(
    symbol='BTCUSDT',
    timeframe='15m',
    candles=10000
)
```

### 6.5 Backtest Runners

```bash
# Quick backtest (3 symbols, 2000 candles)
python run_backtest.py

# Rolling walk-forward test
python run_rolling_wf_test.py

# Full year test
python run_rolling_wf_test.py --full-year

# Quick test (3 months)
python run_rolling_wf_test.py --quick
```

### 6.6 Test Procedures

```bash
# All tests
pytest

# Specific test file
pytest tests/test_signals.py

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Skip slow tests
pytest -m "not slow"

# Coverage report
pytest --cov=core --cov=strategies --cov-report=html
```

---

## 7. Development Guidelines

### 7.1 Code Style

| Aspect | Standard |
|--------|----------|
| Language | Python 3.8+ |
| Comments | Turkish (codebase origin) |
| Type hints | Used for function signatures |
| Imports | stdlib, third-party, local |
| Config | Single Source: `core/config.py` |

### 7.2 Adding New Strategies

1. Create new module in `strategies/`:

```python
# strategies/new_strategy.py
from strategies.base import SignalResult

def check_new_strategy_signal(df, config, index=-2):
    """
    Generate signal for new strategy.

    Returns:
        SignalResult: signal_type, entry, tp, sl, reason
    """
    # Implementation
    return SignalResult(signal_type, entry, tp, sl, reason)
```

2. Export from `strategies/__init__.py`:

```python
from strategies.new_strategy import check_new_strategy_signal
```

3. Add to `STRATEGY_REGISTRY`:

```python
STRATEGY_REGISTRY = {
    "ssl_flow": check_ssl_flow_signal,
    "new_strategy": check_new_strategy_signal,
}
```

4. Update router in `strategies/router.py`:

```python
def check_signal(df, config, index=-2, return_debug=False):
    mode = config.get("strategy_mode", "ssl_flow")

    if mode == "new_strategy":
        return check_new_strategy_signal(df, config, index)
    # ...
```

5. Add strategy wallet in `TradeManager.__init__()`:

```python
self.strategy_wallets = {
    "ssl_flow": {...},
    "new_strategy": {...},
}
```

### 7.3 Modifying Indicators

All indicators in `core/indicators.py`:

```python
def calculate_new_indicator(df, period=14):
    """
    Calculate new indicator.

    Args:
        df: DataFrame with OHLCV data
        period: Calculation period

    Returns:
        Series: Indicator values
    """
    # Implementation
    return result
```

Update `calculate_indicators()` to include new indicator:

```python
def calculate_indicators(df, config=None):
    # Existing indicators...

    df['new_indicator'] = calculate_new_indicator(df, period)

    return df
```

### 7.4 Testing Guidelines

- Write unit tests for all new functions
- Test edge cases (empty data, boundary values)
- Use fixtures from `tests/fixtures/`
- Mark slow tests with `@pytest.mark.slow`

```python
# tests/test_new_feature.py
import pytest
from core.indicators import calculate_new_indicator

def test_new_indicator_basic():
    df = create_test_dataframe()
    result = calculate_new_indicator(df, period=14)
    assert len(result) == len(df)
    assert result.iloc[-1] > 0

@pytest.mark.slow
def test_new_indicator_large_data():
    # Long-running test
    pass
```

---

## 8. Quick Reference

### 8.1 Key Files

| Purpose | File |
|---------|------|
| Main application | `desktop_bot_refactored_v2_base_v7.py` |
| Core config | `core/config.py` |
| Indicators | `core/indicators.py` |
| Trade logic | `core/trade_manager.py` |
| SSL Flow strategy | `strategies/ssl_flow.py` |
| Optimized configs | `best_configs.json` |
| Trade results | `backtest_trades.csv` |

### 8.2 Import Patterns

```python
# Core imports
from core import (
    SYMBOLS, TIMEFRAMES, TRADING_CONFIG,
    TradingEngine, SimTradeManager,
    calculate_indicators,
    send_telegram,
)

# Strategy imports
from strategies import check_signal, STRATEGY_REGISTRY
```

### 8.3 Signal Debug

```python
from strategies import check_signal

# Get detailed debug info
signal_type, entry, tp, sl, reason, debug_info = check_signal(
    df, config, index=-2, return_debug=True
)
print(debug_info)
```

### 8.4 Common Commands

```bash
# Run backtest
python run_backtest.py

# Run walk-forward test
python run_rolling_wf_test.py

# Run tests
pytest

# Start GUI
python desktop_bot_refactored_v2_base_v7.py

# Start headless
python desktop_bot_refactored_v2_base_v7.py --headless
```

### 8.5 Output Files

| File | Content |
|------|---------|
| `backtest_trades.csv` | Individual trade records |
| `backtest_summary.csv` | Stream-level summaries |
| `best_configs.json` | Optimized configurations |
| `data/*.csv` | Historical data cache |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-02 | Initial architecture document |

---

**END OF ARCHITECTURE DOCUMENT**
