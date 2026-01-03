# Project Index: Desktop Bot Trading System

**Generated:** 2026-01-03
**Version:** v2.0.0 (indicator-parity-fix)
**Type:** Cryptocurrency Futures Trading Bot for Binance

---

## Quick Navigation

| Section | Description |
|---------|-------------|
| [Project Structure](#project-structure) | Directory layout |
| [Entry Points](#entry-points) | How to run the bot |
| [Core Modules](#core-modules-core) | Configuration, indicators, trading |
| [Strategies](#strategies-strategies) | SSL Flow strategy |
| [Runners](#runners-runners) | Backtesting and optimization tools |
| [Filter Discovery](#filter-discovery-system) | Bottom-up filter optimization |
| [Tests](#tests-tests) | Test suite |
| [Quick Commands](#quick-commands) | Common operations |

---

## Project Structure

```
desktop_bot_refactored_v2_base_v7/
├── core/                    # Core modules (config, indicators, trade mgmt)
│   ├── config.py            # All constants and configuration
│   ├── indicators.py        # Technical indicator calculations
│   ├── trade_manager.py     # Trade lifecycle management
│   ├── trading_engine.py    # Main trading loop
│   ├── optimizer.py         # Walk-forward optimization
│   ├── at_scenario_analyzer.py  # AlphaTrend scenario analysis
│   └── ...
├── strategies/              # Trading strategy implementations
│   ├── ssl_flow.py          # Main strategy (SSL + AT + PBEMA)
│   └── router.py            # Signal routing
├── runners/                 # Backtest and walk-forward runners
│   ├── run_filter_combo_test.py    # Filter combination discovery
│   ├── run_rolling_wf_combo.py     # WF validation with filter combo
│   ├── run_at_scenario_analysis.py # AT scenario testing
│   └── rolling_wf.py        # Core walk-forward logic
├── ui/                      # PyQt5 GUI components
├── tests/                   # Pytest test suite
├── docs/                    # Documentation
├── data/                    # Data directory (gitignored)
│   └── filter_combo_logs/   # Filter test results
└── *.py                     # Entry points and utility scripts
```

---

## Entry Points

| File | Purpose | Command |
|------|---------|---------|
| `desktop_bot_refactored_v2_base_v7.py` | Main application | `python desktop_bot_refactored_v2_base_v7.py [--headless]` |
| `run_rolling_wf_test.py` | Rolling walk-forward test | `python run_rolling_wf_test.py [--full-year]` |
| `run_backtest.py` | Quick backtest runner | `python run_backtest.py` |

---

## Core Modules (`core/`)

| Module | Key Exports | Purpose |
|--------|-------------|---------|
| `config.py` | `SYMBOLS`, `TIMEFRAMES`, `TRADING_CONFIG`, `DEFAULT_STRATEGY_CONFIG` | All constants and configuration |
| `indicators.py` | `calculate_indicators`, `calculate_alphatrend` | Technical indicator calculations |
| `trade_manager.py` | `BaseTradeManager`, `SimTradeManager` | Trade lifecycle management |
| `trading_engine.py` | `TradingEngine` | Main trading loop |
| `optimizer.py` | `_optimize_backtest_configs` | Walk-forward optimization |
| `at_scenario_analyzer.py` | `check_core_signal`, `ATScenarioAnalyzer` | Minimal signal detection |
| `binance_client.py` | `BinanceClient`, `get_client` | Binance API with retry |
| `regime_filter.py` | `RegimeFilter`, `RegimeType` | Market regime detection |

---

## Strategies (`strategies/`)

| Strategy | Status | Description |
|----------|--------|-------------|
| `ssl_flow.py` | **ACTIVE** | SSL HYBRID baseline + AlphaTrend + PBEMA cloud |
| `keltner_bounce.py` | DISABLED | Mean reversion using Keltner bands |
| `router.py` | - | Signal routing via `check_signal()` |

**Active Strategy: SSL Flow**
- Entry: Price above/below SSL baseline (HMA60) + AlphaTrend confirmation
- TP: PBEMA cloud (EMA200)
- SL: Swing low/high or baseline

---

## Runners (`runners/`)

### Standard Runners

| Module | Purpose |
|--------|---------|
| `rolling_wf.py` | Core walk-forward implementation |
| `rolling_wf_optimized.py` | Parallel/cached version (4x faster) |
| `portfolio.py` | Multi-symbol portfolio backtest |
| `run_direct_backtest.py` | Direct backtest with real `check_signal` |

### Filter Discovery Runners

| Module | Purpose | Command |
|--------|---------|---------|
| `run_filter_combo_test.py` | Test filter combinations | See [Filter Discovery](#filter-discovery-system) |
| `run_rolling_wf_combo.py` | WF validation with filter config | See [Filter Discovery](#filter-discovery-system) |
| `run_at_scenario_analysis.py` | AT scenario comparison | `python runners/run_at_scenario_analysis.py` |

---

## Filter Discovery System

### Philosophy

```
MİNİMAL BAŞLA → BOL TRADE → ANALİZ ET → AKILLI FİLTRELE
```

Instead of top-down (strict filters → no trades), use bottom-up:
1. Start with minimal signals (`check_core_signal`) → 1684 baseline signals
2. Add filters one by one and measure impact
3. Find optimal combination
4. Validate with Rolling WF

### Commands

```bash
# Incremental filter test (one filter at a time)
python runners/run_filter_combo_test.py --incremental

# Full scan (all 511 combinations)
python runners/run_filter_combo_test.py --full-scan

# Specific combo test with logging
python runners/run_filter_combo_test.py \
    --specific "regime,at_flat_filter,adx_filter"

# OOS validation for specific combo
python runners/run_filter_combo_test.py \
    --specific "regime,at_flat_filter,adx_filter" --oos

# Rolling WF validation with filter combo
python runners/run_rolling_wf_combo.py \
    --symbol BTCUSDT --timeframe 15m \
    --filters "regime,at_flat_filter,adx_filter" \
    --full-year
```

### Available Filters

| Filter | Description | Effect |
|--------|-------------|--------|
| `regime` | Skip neutral regime | **MANDATORY BASE** |
| `at_flat_filter` | Skip when AT is flat | Best single filter (+$14.62) |
| `adx_filter` | ADX > 15 required | Good trend filter (+$13.12) |
| `at_binary` | AT alignment required | Mixed results |
| `ssl_touch` | SSL touch in last 5 bars | Neutral |
| `rsi_filter` | RSI limits | Minimal impact |
| `pbema_distance` | PBEMA distance check | Minimal impact |
| `overlap_check` | SSL-PBEMA gap check | Slightly positive |
| `body_position` | Candle body position | Negative |
| `wick_rejection` | Wick ratio filter | Negative |

### Current Best Config (BTCUSDT 15m)

```python
filters = ["regime", "at_flat_filter", "adx_filter"]
# Result: 238 trades, +$15.91 PnL, $5.62 MaxDD
# WF Validation: 208 trades, +$11.39 PnL (PASSED)
```

### Log Files

Results are logged to `data/filter_combo_logs/`:
- `combo_tests_{SYMBOL}_{TF}.jsonl` - Append-only log of all tests
- `{SYMBOL}_{TF}_{DAYS}d_{TIMESTAMP}.json` - Full scan results
- `{SYMBOL}_{TF}_{DAYS}d_{TIMESTAMP}.txt` - Human-readable report

---

## Tests (`tests/`)

| Test File | Coverage |
|-----------|----------|
| `test_config.py` | Configuration validation |
| `test_indicators.py` | Indicator calculations |
| `test_signals.py` | Signal detection |
| `test_trade_manager.py` | Trade lifecycle |
| `test_parity.py` | Live/sim parity |
| `test_risk.py` | Risk management |

**Run:** `pytest` or `pytest -v`

---

## Quick Commands

```bash
# GUI mode
python desktop_bot_refactored_v2_base_v7.py

# Headless/CLI mode
python desktop_bot_refactored_v2_base_v7.py --headless

# Rolling walk-forward test (standard)
python run_rolling_wf_test.py --full-year

# Filter combo discovery workflow
python runners/run_filter_combo_test.py --incremental  # Step 1: Find filters
python runners/run_rolling_wf_combo.py --filters "regime,at_flat_filter" --full-year  # Step 2: Validate

# Quick backtest
python run_backtest.py

# Run tests
pytest -v
```

---

## Key Configuration

**Symbols:** BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, LINKUSDT, XRPUSDT, DOGEUSDT, LTCUSDT
**Timeframes:** 5m, 15m, 30m, 1h, 4h, 12h, 1d
**Risk:** 1.75% per trade, 5% max portfolio

**Recommended Portfolio:** BTC + ETH + LINK

---

## Critical Documentation

| Document | Description |
|----------|-------------|
| `CLAUDE.md` | Master reference for AI assistants |
| `docs/REQUIREMENTS.md` | IEEE 830 requirements spec |
| `docs/ARCHITECTURE.md` | Technical architecture |
| `docs/FILTER_DISCOVERY_METHODOLOGY.md` | Filter discovery approach |
| `VERSION.md` | Version history |

---

## Import Patterns

```python
# Core imports
from core import (
    SYMBOLS, TIMEFRAMES, TRADING_CONFIG,
    TradingEngine, SimTradeManager,
    calculate_indicators, send_telegram,
)

# Strategy imports
from strategies import check_signal, STRATEGY_REGISTRY

# Filter combo imports
from runners.run_filter_combo_test import (
    run_specific_combo,
    run_oos_validation,
    log_combo_result,
)

# Minimal signal (for filter discovery)
from core.at_scenario_analyzer import check_core_signal
```

---

## Workflow: Symbol Validation

```
1. Filter Combo Discovery
   └── python runners/run_filter_combo_test.py --symbol X --timeframe Y --incremental
   └── Find best filter combination
   └── Results logged to data/filter_combo_logs/

2. Rolling WF Validation
   └── python runners/run_rolling_wf_combo.py --symbol X --timeframe Y --filters "..." --full-year
   └── If PnL > 0: Symbol validated
   └── If PnL < 0: Config may be overfit, try different filters

3. Production
   └── Use validated config in live trading
```

---

## Dependencies

- pandas >= 1.5.0
- pandas_ta >= 0.3.14b
- numpy >= 1.21.0
- requests >= 2.28.0
- matplotlib >= 3.5.0
- tqdm >= 4.64.0
- PyQt5 >= 5.15.0 (optional, for GUI)

---

**Index Size:** ~5KB | **Full Codebase:** ~60K tokens | **Savings:** 92%
