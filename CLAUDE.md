# CLAUDE.md - AI Assistant Guide for Trading Bot Codebase

## Project Overview

This is a sophisticated **cryptocurrency futures trading bot** that implements technical analysis-based strategies for Binance Futures. The bot supports both GUI (PyQt5) and headless (CLI/Colab) modes, with comprehensive backtesting and optimization capabilities.

**Current Version:** v39.0 - R-Multiple Based with Walk-Forward Optimization

### Key Capabilities
- Live trading with real-time WebSocket data streaming
- Portfolio backtesting across multiple symbols and timeframes
- Walk-forward parameter optimization
- R-Multiple based position sizing and risk management
- Telegram notifications
- Google Colab / headless server support

---

## Repository Structure

```
desktop_bot_refactored_v2_base_v7/
├── desktop_bot_refactored_v2_base_v7.py  # Main application (~7600 lines)
├── run_backtest.py                        # Quick backtest runner script
├── docs/
│   └── pbema_annotation.md               # PBEMA cloud documentation
├── .gitignore
└── CLAUDE.md                             # This file
```

### Generated Files (gitignored)
- `trades.csv` - Live trade history
- `backtest_trades.csv` - Backtest trade results
- `backtest_summary.csv` - Backtest summary statistics
- `best_configs.json` - Optimized strategy configurations
- `config.json` - User configuration (Telegram, etc.)
- `*_prices.csv` - Historical price data per symbol/timeframe

---

## Architecture Overview

### Core Classes

| Class | Purpose | Lines |
|-------|---------|-------|
| `TradeManager` | Live trade execution, position management, risk control | 1599-2465 |
| `SimTradeManager` | Simulation trade manager for backtesting | 5713-6234 |
| `TradingEngine` | Core trading logic, data fetching, indicators, signals | 2496-3707 |
| `MainWindow` | PyQt5 GUI application | 4594-5700 |
| `LiveBotWorker` | QThread for live trading loop | 3865-4205 |
| `OptimizerWorker` | QThread for parameter optimization | 4206-4451 |
| `BacktestWorker` | QThread for backtesting | 4540-4593 |
| `BinanceWebSocketKlineStream` | Real-time WebSocket data streamer | 3710-3864 |
| `PotentialTradeRecorder` | Logs potential trade setups | 2468-2495 |

### Key Functions

| Function | Purpose |
|----------|---------|
| `run_portfolio_backtest()` | Multi-symbol/timeframe portfolio backtesting |
| `run_cli_backtest()` | CLI/Colab backtest interface |
| `_optimize_backtest_configs()` | Walk-forward parameter optimization |
| `_score_config_for_stream()` | Score a config for a symbol/timeframe pair |
| `TradingEngine.check_signal()` | Signal detection wrapper (routes to strategy) |
| `TradingEngine.calculate_indicators()` | Calculate all technical indicators |
| `load_optimized_config()` | Load best config for symbol/timeframe |
| `save_best_configs()` | Save optimized configurations |

---

## Trading Strategies

### 1. Keltner Bounce (Default)
**Mode:** `strategy_mode: "keltner_bounce"`

Mean reversion strategy using Keltner bands with PBEMA cloud as target:
- Entry: Price touches Keltner band and rejects
- Target: PBEMA cloud (EMA200)
- Stop Loss: Beyond Keltner band

**Signal Function:** `TradingEngine.check_signal_diagnostic()`

### 2. PBEMA Reaction
**Mode:** `strategy_mode: "pbema_reaction"`

Strategy based on price reaction to PBEMA cloud (EMA150):
- Entry: Price approaches or touches PBEMA cloud
- Uses frontrunning margin for entry
- Different indicator periods (150 vs 200 EMA)

**Signal Function:** `TradingEngine.check_pbema_reaction_signal()`

### Technical Indicators Used
- **RSI(14)** - Relative Strength Index
- **ADX(14)** - Average Directional Index
- **PBEMA Cloud** - EMA200(high) and EMA200(close) for Base, EMA150 for Reaction
- **SSL Baseline** - HMA60(close)
- **Keltner Bands** - baseline ± EMA60(TrueRange) * 0.2
- **AlphaTrend** - Optional trend filter

---

## Configuration System

### Global Constants (top of file)
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
`SYMBOL_PARAMS` dict contains per-symbol, per-timeframe settings:
```python
SYMBOL_PARAMS = {
    "BTCUSDT": {
        "5m": {"rr": 2.4, "rsi": 35, "slope": 0.2, "at_active": False, ...},
        "15m": {...},
        ...
    },
    ...
}
```

### Strategy Configuration
`DEFAULT_STRATEGY_CONFIG` defines strategy parameters:
- `rr` - Risk/Reward ratio
- `rsi` - RSI threshold
- `slope` - Slope threshold (currently disabled for mean reversion)
- `at_active` - AlphaTrend filter active
- `use_trailing` - Trailing stop enabled
- `strategy_mode` - "keltner_bounce" or "pbema_reaction"

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
- Strategy-isolated wallets (keltner_bounce and pbema_reaction have separate balances)

### Gating System
Multi-layer gating prevents weak-edge configs:
1. Minimum E[R] threshold (account-size independent)
2. Minimum optimizer score threshold (varies by timeframe)
3. Confidence-based risk multiplier
4. Walk-forward out-of-sample validation

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

### Performance Considerations
1. **Hot loops use NumPy arrays** - Pre-extract arrays before loops to avoid `df.iloc[i]` overhead
2. **Parallel data fetching** - ThreadPoolExecutor with max 5 workers (API rate limits)
3. **In-place DataFrame operations** - `calculate_indicators()` modifies in-place for speed
4. **Caching** - Best configs cached in `BEST_CONFIG_CACHE` and `best_configs.json`

### Adding New Symbols
1. Add to `SYMBOLS` list at top of file
2. Add entry in `SYMBOL_PARAMS` with per-timeframe settings
3. Run optimization to find best parameters

### Adding New Strategies
1. Create new signal detection function (see `check_pbema_reaction_signal` as template)
2. Add strategy mode to `check_signal()` router
3. Add strategy wallet in `TradeManager.__init__()`
4. Update `_generate_candidate_configs()` to include new strategy

### Modifying Indicators
All indicators calculated in `TradingEngine.calculate_indicators()`:
- Add new indicator calculation
- Ensure column name is consistent
- Update signal functions to use new indicator

---

## Testing

### Syntax Validation
```bash
python3 -m py_compile desktop_bot_refactored_v2_base_v7.py
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

### 2. Optimizing Performance
- Look for `df.iloc[i]` in loops - replace with pre-extracted NumPy arrays
- Check for unnecessary `.copy()` calls
- Consider parallel processing for independent operations

### 3. Modifying Strategy Parameters
- Update `DEFAULT_STRATEGY_CONFIG` for global changes
- Update `SYMBOL_PARAMS` for symbol-specific changes
- Re-run optimization after changes

### 4. Adding Features
- Prefer editing existing code over creating new files
- Follow existing patterns (see similar functionality)
- Add configuration options to appropriate config dicts
- Update CLI arguments if needed

### 5. Debugging Trade Issues
- Check `trades.csv` for trade history
- Use `check_signal(..., return_debug=True)` for signal diagnostics
- Check cooldown logic in `TradeManager.check_cooldown()`
- Verify config is not disabled in `POST_PORTFOLIO_BLACKLIST`

---

## API Integration

### Binance Futures API
- Base URL: `https://fapi.binance.com/fapi/v1/`
- Endpoints used: `/klines`, `/ticker/price`
- WebSocket: `wss://fstream.binance.com/stream`
- Rate limiting: Max 5 parallel requests, exponential backoff on 429/5xx

### Telegram API
- Used for trade notifications
- Configure via GUI or `config.json`
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

### Error Logging
- Errors logged to `error_log.txt`
- Console output includes timestamps and context
- Use `verbose=True` in CLI functions for detailed output

---

## Version History Notes

- **v39.0**: R-Multiple based optimizer gating, walk-forward validation
- **v37.0**: Dynamic optimizer controls, removed hardcoded disabled states
- **v30.4**: Initial PROFIT ENGINE release

---

## Quick Reference

### Entry Points
- GUI: `MainWindow` class
- CLI: `run_cli_backtest()`, `colab_quick_test()`
- Backtest: `run_portfolio_backtest()`

### Key Files
- Main code: `desktop_bot_refactored_v2_base_v7.py`
- Configs: `best_configs.json`, `config.json`
- Results: `backtest_trades.csv`, `backtest_summary.csv`

### Important Line Numbers
- Constants: 79-420
- TradeManager: 1599-2465
- TradingEngine: 2496-3707
- Signal Detection: 2755-3478
- Backtest: 6236-6926
- CLI Interface: 7387-7608
