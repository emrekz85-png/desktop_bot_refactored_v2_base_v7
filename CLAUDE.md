# CLAUDE.md - AI Assistant Guide for Trading Bot Codebase

## Project Overview

This is a sophisticated **cryptocurrency futures trading bot** that implements technical analysis-based strategies for Binance Futures. The bot supports both GUI (PyQt5) and headless (CLI/Colab) modes, with comprehensive backtesting and optimization capabilities.

**Current Version:** v40.x - Modular Architecture with Walk-Forward Optimization

### Key Capabilities
- Live trading with real-time WebSocket data streaming
- Portfolio backtesting across multiple symbols and timeframes
- Walk-forward parameter optimization
- R-Multiple based position sizing and risk management
- Telegram notifications (with environment variable support)
- Google Colab / headless server support
- Modular architecture for maintainability

---

## Repository Structure

```
desktop_bot_refactored_v2_base_v7/
├── desktop_bot_refactored_v2_base_v7.py  # Main application (~8800 lines)
├── run_backtest.py                        # Quick backtest runner script
├── run_rolling_wf_test.py                 # Rolling walk-forward test
├── run_baseline_comparison.py             # Baseline comparison utility
├── run_strategy_autopsy.py                # Strategy performance analysis
├── run_optimizer_diagnostic.py            # Optimizer diagnostics
├── run_strategy_sanity_tests.py           # Strategy sanity checks
├── fast_start.py                          # Fast startup script
├── precompile.py                          # Bytecode precompilation
├── requirements.txt                       # Python dependencies
├── pytest.ini                             # Pytest configuration
│
├── core/                                  # Core package (modular components)
│   ├── __init__.py                        # Package exports
│   ├── config.py                          # Constants, trading config, blacklists
│   ├── config_loader.py                   # Load/save optimized configs
│   ├── indicators.py                      # Technical indicator calculations
│   ├── trade_manager.py                   # BaseTradeManager & SimTradeManager
│   ├── trading_engine.py                  # TradingEngine class
│   ├── binance_client.py                  # Binance API client with retry
│   ├── telegram.py                        # Telegram notifications
│   ├── utils.py                           # Helper functions
│   └── logging_config.py                  # Centralized logging
│
├── strategies/                            # Strategy implementations
│   ├── __init__.py                        # Strategy registry
│   ├── base.py                            # SignalResult, STRATEGY_MODES
│   ├── router.py                          # Signal routing logic
│   ├── ssl_flow.py                        # SSL Flow strategy [ACTIVE]
│   └── keltner_bounce.py                  # Keltner Bounce strategy [DISABLED]
│
├── ui/                                    # PyQt5 GUI components
│   ├── __init__.py                        # Package exports
│   ├── main_window.py                     # MainWindow class
│   └── workers.py                         # QThread workers
│
├── tests/                                 # Pytest test suite
│   ├── __init__.py
│   ├── conftest.py                        # Pytest fixtures
│   ├── fixtures/                          # Test data fixtures
│   ├── test_config.py                     # Configuration tests
│   ├── test_indicators.py                 # Indicator calculation tests
│   ├── test_parity.py                     # Live/sim parity tests
│   ├── test_risk.py                       # Risk management tests
│   ├── test_signals.py                    # Signal detection tests
│   └── test_trade_manager.py              # Trade manager tests
│
├── docs/
│   └── pbema_annotation.md                # PBEMA cloud documentation
│
├── data/                                  # Data directory (gitignored contents)
│
├── start_bot.sh / start_bot.bat           # Start bot scripts
├── start_backtest.sh / start_backtest.bat # Start backtest scripts
├── start_fast.bat                         # Fast startup (Windows)
│
├── .gitignore
└── CLAUDE.md                              # This file
```

### Generated Files (gitignored)
- `trades.csv` - Live trade history
- `backtest_trades.csv` - Backtest trade results
- `backtest_summary.csv` - Backtest summary statistics
- `best_configs.json` - Optimized strategy configurations
- `config.json` - User configuration (Telegram, etc.)
- `*_prices.csv` - Historical price data per symbol/timeframe
- `error_log.txt` - Error log file

---

## Architecture Overview

### Modular Core Package (`core/`)

The codebase uses a modular architecture with the `core` package as the single source of truth:

| Module | Purpose |
|--------|---------|
| `config.py` | Constants, SYMBOLS, TIMEFRAMES, TRADING_CONFIG, blacklist management |
| `config_loader.py` | Load/save optimized strategy configurations |
| `indicators.py` | Technical indicator calculations (RSI, ADX, PBEMA, Keltner, AlphaTrend) |
| `trade_manager.py` | BaseTradeManager (shared logic), SimTradeManager (backtesting) |
| `trading_engine.py` | TradingEngine class - data fetching, signal detection |
| `binance_client.py` | Binance API client with retry logic and rate limiting |
| `telegram.py` | Thread-safe Telegram notifications |
| `utils.py` | Helper functions (time conversion, funding calculation, R-multiple) |
| `logging_config.py` | Centralized logging with file rotation |

### Strategy Package (`strategies/`)

| Module | Purpose |
|--------|---------|
| `base.py` | SignalResult dataclass, STRATEGY_MODES enum |
| `router.py` | Routes signal checks to appropriate strategy |
| `ssl_flow.py` | SSL Flow trend following strategy [ACTIVE] |
| `keltner_bounce.py` | Mean reversion strategy using Keltner bands [DISABLED] |

### UI Package (`ui/`)

| Module | Purpose |
|--------|---------|
| `main_window.py` | MainWindow - PyQt5 application with all tabs |
| `workers.py` | QThread workers: LiveBotWorker, OptimizerWorker, BacktestWorker, AutoBacktestWorker |

### Main Application Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `TradingEngine` | `core/trading_engine.py` | Core trading logic, data fetching, indicators, signals |
| `BaseTradeManager` | `core/trade_manager.py` | Shared trade management logic |
| `SimTradeManager` | `core/trade_manager.py` | Simulation trade manager for backtesting |
| `TradeManager` | `desktop_bot_...py` | Live trade execution (extends BaseTradeManager) |
| `MainWindow` | `ui/main_window.py` | PyQt5 GUI application |
| `BinanceClient` | `core/binance_client.py` | Binance API wrapper |
| `TelegramNotifier` | `core/telegram.py` | Thread-safe notifications |

---

## Trading Strategies

### 1. SSL Flow [ACTIVE - Default]
**Mode:** `strategy_mode: "ssl_flow"`

Trend following strategy using SSL HYBRID baseline with AlphaTrend confirmation:
- Entry: Price retests SSL baseline (HMA60) as support/resistance
- AlphaTrend confirms buyer/seller dominance (filters fake signals)
- Target: PBEMA cloud (EMA200)
- Stop Loss: Beyond swing high/low

**Signal Function:** `strategies.ssl_flow.check_ssl_flow_signal()`

### 2. Keltner Bounce [DISABLED]
**Mode:** `strategy_mode: "keltner_bounce"`

Mean reversion strategy using Keltner bands with PBEMA cloud as target:
- Entry: Price touches Keltner band and rejects
- Target: PBEMA cloud (EMA200)
- Stop Loss: Beyond Keltner band

**Signal Function:** `strategies.keltner_bounce.check_keltner_bounce_signal()`

**Note:** This strategy is currently disabled. All symbols use SSL Flow by default.

### Technical Indicators Used
- **RSI(14)** - Relative Strength Index
- **ADX(14)** - Average Directional Index
- **PBEMA Cloud** - EMA200(high) and EMA200(close) as TP target
- **SSL Baseline** - HMA60(close) - main trend indicator
- **Keltner Bands** - baseline ± EMA60(TrueRange) * 0.2
- **AlphaTrend** - Dual-line trend filter with flow detection (buyers vs sellers)

---

## Configuration System

### Configuration Hierarchy (Single Source of Truth)

All configuration lives in `core/config.py`:

```python
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", ...]  # Traded symbols
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "12h", "1d"]
TRADING_CONFIG = {
    "initial_balance": 2000.0,
    "leverage": 10,
    "risk_per_trade_pct": 0.0175,  # 1.75% per trade
    "max_portfolio_risk_pct": 0.05,  # 5% total
    ...
}
```

### Symbol-Specific Parameters
`SYMBOL_PARAMS` dict in `core/config.py`:
```python
SYMBOL_PARAMS = {
    "BTCUSDT": {
        "5m": {"rr": 2.0, "rsi": 70, "at_active": True, "strategy_mode": "ssl_flow"},
        "15m": {...},
        ...
    },
    ...
}
```
**Note:** `at_active: True` is MANDATORY for all configs. AlphaTrend is essential for SSL Flow strategy.

### Strategy Configuration
`DEFAULT_STRATEGY_CONFIG` defines strategy parameters:
- `rr` - Risk/Reward ratio
- `rsi` - RSI threshold
- `at_active` - AlphaTrend filter active (essential for SSL Flow)
- `use_trailing` - Trailing stop enabled
- `strategy_mode` - "ssl_flow" (default) or "keltner_bounce" (disabled)

### Environment Variables (Recommended for Telegram)
```bash
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

---

## Risk Management

### R-Multiple System
Position sizing based on R-multiple (risk units):
- `E[R]` = Expected R-multiple per trade (average R across all trades)
- Minimum E[R] thresholds per timeframe in `MIN_EXPECTANCY_R_MULTIPLE`
- Confidence-based risk multipliers in `CONFIDENCE_RISK_MULTIPLIER`

### Portfolio Risk Limits
- Per-trade risk: 1.75% (configurable)
- Max portfolio risk: 5% (configurable)
- Strategy-isolated wallets (each strategy has separate balance for isolation)

### Gating System
Multi-layer gating prevents weak-edge configs:
1. Minimum E[R] threshold (account-size independent)
2. Minimum optimizer score threshold (varies by timeframe)
3. Confidence-based risk multiplier
4. Walk-forward out-of-sample validation

### Circuit Breaker
Configured in `CIRCUIT_BREAKER_CONFIG` - disables streams after consecutive losses.

---

## Running the Application

### GUI Mode (Desktop)
```bash
python desktop_bot_refactored_v2_base_v7.py
```
Requires PyQt5 and PyQtWebEngine.

### Headless/CLI Mode
```bash
python desktop_bot_refactored_v2_base_v7.py --headless
```

### CLI Backtest with Options
```bash
python desktop_bot_refactored_v2_base_v7.py --headless \
    --symbols BTCUSDT ETHUSDT \
    --timeframes 5m 15m 1h \
    --candles 30000 \
    --no-optimize  # Skip optimization, use cached configs
```

### Quick Backtest Script
```bash
python run_backtest.py
```

### Fast Startup (Precompiled)
```bash
python fast_start.py
# or on Windows:
start_fast.bat
```

### Running Tests
```bash
pytest                           # Run all tests
pytest tests/test_signals.py     # Run specific test file
pytest -v                        # Verbose output
pytest -x                        # Stop on first failure
```

### Google Colab
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
results = colab_quick_test(symbol='BTCUSDT', timeframe='15m', candles=10000)
```

---

## Development Guidelines

### Code Style
- **Language:** Python 3.8+
- **Comments:** Turkish (codebase originated in Turkey)
- **Type hints:** Used for function signatures
- **Imports:** Standard library, then third-party, then local
- **Single Source of Truth:** All constants and configs from `core/config.py`

### Performance Considerations
1. **Lazy imports** - Heavy libraries (pandas_ta, plotly) imported lazily for fast startup
2. **Hot loops use NumPy arrays** - Pre-extract arrays before loops
3. **Parallel data fetching** - ThreadPoolExecutor with max 5 workers
4. **In-place DataFrame operations** - `calculate_indicators()` modifies in-place
5. **Caching** - Best configs cached in `BEST_CONFIG_CACHE` and `best_configs.json`

### Adding New Symbols
1. Add to `SYMBOLS` list in `core/config.py`
2. Add entry in `SYMBOL_PARAMS` with per-timeframe settings
3. Run optimization to find best parameters

### Adding New Strategies
1. Create new module in `strategies/` (see `ssl_flow.py` as template)
2. Export from `strategies/__init__.py`
3. Add to `STRATEGY_REGISTRY` in `strategies/__init__.py`
4. Update `check_signal()` router in `strategies/router.py`
5. Add strategy wallet in `TradeManager.__init__()`

### Modifying Indicators
All indicators calculated in `core/indicators.py`:
- Add new indicator calculation function
- Ensure column name is consistent
- Update signal functions to use new indicator

---

## Testing

### Syntax Validation
```bash
python3 -m py_compile desktop_bot_refactored_v2_base_v7.py
```

### Run Test Suite
```bash
pytest                           # All tests
pytest tests/test_signals.py     # Signal tests
pytest tests/test_parity.py      # Live/sim parity tests
pytest tests/test_risk.py        # Risk management tests
pytest tests/test_indicators.py  # Indicator tests
```

### Backtest Validation
Run backtests and verify:
- Trade counts are reasonable
- Win rates are within expected range
- PnL is positive for enabled configs
- No errors in console output

### Signal Debugging
Use diagnostic functions:
```python
# Check specific candle conditions
TradingEngine.debug_base_short(df, -2)

# Plot a backtest trade
debug_plot_backtest_trade('BTCUSDT', '15m', trade_id=123)
```

---

## Common Tasks for AI Assistants

### 1. Fixing Bugs
- Read the relevant function first
- Check if similar logic exists in both `TradeManager` (live) and `SimTradeManager` (backtest)
- Ensure parity between live and simulation logic
- Run `pytest tests/test_parity.py` to verify parity

### 2. Optimizing Performance
- Look for `df.iloc[i]` in loops - replace with pre-extracted NumPy arrays
- Check for unnecessary `.copy()` calls
- Consider parallel processing for independent operations
- Use lazy imports for heavy libraries

### 3. Modifying Strategy Parameters
- Update `DEFAULT_STRATEGY_CONFIG` in `core/config.py` for global changes
- Update `SYMBOL_PARAMS` in `core/config.py` for symbol-specific changes
- Re-run optimization after changes

### 4. Adding Features
- Prefer editing existing code over creating new files
- Follow existing patterns (see similar functionality)
- Add configuration options to `core/config.py`
- Update CLI arguments if needed
- Add tests in `tests/` directory

### 5. Debugging Trade Issues
- Check `trades.csv` for trade history
- Use `check_signal(..., return_debug=True)` for signal diagnostics
- Check cooldown logic in `TradeManager.check_cooldown()`
- Verify config is not disabled in `POST_PORTFOLIO_BLACKLIST`

---

## API Integration

### Binance Futures API
- Client: `core/binance_client.py` (BinanceClient class)
- Base URL: `https://fapi.binance.com/fapi/v1/`
- Endpoints used: `/klines`, `/ticker/price`
- WebSocket: `wss://fstream.binance.com/stream`
- Rate limiting: Max 5 parallel requests, exponential backoff on 429/5xx

### Telegram API
- Notifier: `core/telegram.py` (TelegramNotifier class)
- Thread-safe with connection pooling
- Configure via environment variables or `config.json`
- Token and chat_id required

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "No Data" signals | Check internet connection, API rate limits |
| GUI not starting | Ensure PyQt5 installed, or use `--headless` |
| Optimization too slow | Use `quick_mode=True` or reduce candle count |
| Configs all disabled | Lower thresholds in `MIN_EXPECTANCY_R_MULTIPLE` |
| WebSocket disconnects | Auto-reconnect built-in, check logs |
| Import errors | Ensure `core/` package is in path |
| Slow startup | Use `fast_start.py` or run `precompile.py` first |

### Error Logging
- Errors logged to `error_log.txt`
- Console output includes timestamps and context
- Use `verbose=True` in CLI functions for detailed output
- Centralized logging via `core/logging_config.py`

---

## Version History Notes

- **v40.x**: Modular architecture, core package refactoring, lazy imports
- **v39.0**: R-Multiple based optimizer gating, walk-forward validation
- **v37.0**: Dynamic optimizer controls, removed hardcoded disabled states
- **v30.4**: Initial PROFIT ENGINE release

---

## Quick Reference

### Entry Points
- GUI: `MainWindow` class in `ui/main_window.py`
- CLI: `run_cli_backtest()`, `colab_quick_test()`
- Backtest: `run_portfolio_backtest()`

### Key Files
- Main code: `desktop_bot_refactored_v2_base_v7.py`
- Core config: `core/config.py`
- Indicators: `core/indicators.py`
- Trade logic: `core/trade_manager.py`
- Strategies: `strategies/`
- Configs: `best_configs.json`, `config.json`
- Results: `backtest_trades.csv`, `backtest_summary.csv`

### Import Pattern
```python
from core import (
    SYMBOLS, TIMEFRAMES, TRADING_CONFIG,
    TradingEngine, SimTradeManager,
    calculate_indicators,
    send_telegram,
)
from strategies import check_signal, STRATEGY_REGISTRY
```
