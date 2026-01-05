# Project Index: SSL Flow Trading Bot

**Generated:** 2026-01-03
**Version:** v2.0.0 (Filter Combo Discovery + Trade Visualizer)
**Type:** Cryptocurrency Futures Trading Bot

---

## Quick Start

```bash
# Full Pipeline Analysis (Recommended)
python runners/run_full_pipeline.py --symbol BTCUSDT --timeframe 15m

# Pipeline with trade logging
python runners/run_full_pipeline.py --symbol BTCUSDT --timeframe 15m --save-trades

# Trade Visualizer
python runners/run_trade_visualizer.py --symbol BTCUSDT --timeframe 15m \
    --entry-time "2025-03-02 21:00" --signal-type SHORT

# GUI Mode
python desktop_bot_refactored_v2_base_v7.py

# Headless Mode
python desktop_bot_refactored_v2_base_v7.py --headless
```

---

## Project Structure

```
desktop_bot_refactored_v2_base_v7/
├── desktop_bot_refactored_v2_base_v7.py  # Main entry point (GUI/headless)
├── CLAUDE.md                              # AI assistant instructions
├── FOCUS.md                               # Active work context
├── PROJECT_INDEX.md                       # This file
│
├── core/                    # Core business logic (34 files)
│   ├── __init__.py          # Public API exports
│   ├── config.py            # Trading configuration & constants
│   ├── indicators.py        # Technical indicators (HMA, ATR, RSI, AlphaTrend)
│   ├── trade_manager.py     # Trade execution & position management
│   ├── trading_engine.py    # Main trading loop & stream management
│   ├── trade_visualizer.py  # Chart generation for trades (NEW)
│   ├── at_scenario_analyzer.py  # Core signal detection (NEW)
│   ├── binance_client.py    # Binance API wrapper
│   ├── optuna_optimizer.py  # Bayesian optimization
│   ├── regime_filter.py     # Market regime detection
│   ├── risk_manager.py      # Position sizing & risk controls
│   └── ...                  # Other utilities
│
├── strategies/              # Trading strategies (5 files)
│   ├── ssl_flow.py          # SSL Flow strategy (MAIN)
│   ├── keltner_bounce.py    # Keltner Channel strategy
│   └── router.py            # Strategy router
│
├── runners/                 # Execution scripts (12+ files)
│   ├── run_full_pipeline.py       # Full analysis pipeline (MAIN)
│   ├── run_at_scenario_analysis.py # AT scenario testing
│   ├── run_filter_combo_test.py   # Filter combination discovery
│   ├── run_rolling_wf_combo.py    # WF with filters
│   ├── run_realistic_backtest.py  # Cost-aware backtest
│   ├── run_trade_visualizer.py    # Trade chart generation
│   ├── rolling_wf.py              # Walk-forward logic
│   └── portfolio.py               # Portfolio management
│
├── ui/                      # PyQt6 GUI (3 files)
│   ├── main_window.py       # Main application window
│   └── workers.py           # Background workers
│
├── tests/                   # Test suite (15+ files)
│   ├── test_indicators.py   # Indicator tests
│   ├── test_signals.py      # Signal tests
│   └── manual/              # Manual test scripts
│
├── data/                    # Data storage
│   ├── pipeline_reports/    # Pipeline analysis results
│   │   └── BTCUSDT/         # Symbol-specific reports
│   ├── optuna_runs/         # Optimization results
│   └── rolling_wf_runs/     # Backtest results
│
├── trade_charts/            # Trade visualizations
│   ├── btc_15m_all/         # Batch visualizations
│   └── *.png                # Individual charts
│
└── docs/                    # Documentation (97 files)
    ├── REQUIREMENTS.md      # IEEE 830 specification
    ├── ARCHITECTURE.md      # Technical architecture
    └── CHANGELOG.md         # Version history
```

---

## File Statistics

| Category | Count |
|----------|-------|
| Python files | 108 |
| Markdown docs | 97 |
| Core modules | 34 |
| Strategy files | 5 |
| Runner scripts | 12 |
| Test files | 15+ |

---

## Core Modules

### Entry Points

| File | Purpose | Command |
|------|---------|---------|
| `desktop_bot_refactored_v2_base_v7.py` | Main GUI/headless app | `python desktop_bot_refactored_v2_base_v7.py` |
| `runners/run_full_pipeline.py` | Full analysis pipeline | `python runners/run_full_pipeline.py` |
| `runners/run_trade_visualizer.py` | Trade visualization | `python runners/run_trade_visualizer.py` |

### Core Logic

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `core/config.py` | Configuration | `TRADING_CONFIG`, `SYMBOLS`, `TIMEFRAMES` |
| `core/indicators.py` | Technical indicators | `calculate_indicators()`, `calculate_alphatrend()` |
| `core/trade_manager.py` | Trade execution | `SimTradeManager`, `Trade` |
| `core/trade_visualizer.py` | Chart generation | `TradeVisualizer`, `visualize_trade()` |
| `core/at_scenario_analyzer.py` | Signal detection | `check_core_signal()`, `ATScenarioAnalyzer` |

### Strategies

| Strategy | File | Description |
|----------|------|-------------|
| **SSL Flow** | `strategies/ssl_flow.py` | Main trend-following strategy |
| Keltner Bounce | `strategies/keltner_bounce.py` | Mean-reversion strategy |

---

## Current Status (2026-01-03)

**Strategy Edge < Trading Costs = Net Loss**

| Metric | Value |
|--------|-------|
| Best Config | REGIME + at_flat_filter |
| Trades (1 year) | 242 |
| Win Rate | 31% |
| Ideal PnL | +$14.62 |
| Cost-Aware PnL | **-$5.70** |
| Edge | 0.17% |
| Costs | 0.24% |
| Recommendation | **DO NOT TRADE** |

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | AI assistant instructions |
| `FOCUS.md` | Active work context (read first!) |
| `docs/REQUIREMENTS.md` | Formal specification (IEEE 830) |
| `docs/ARCHITECTURE.md` | Technical architecture |
| `docs/CHANGELOG.md` | Version history |

---

## Import Pattern

```python
from core import (
    SYMBOLS, TIMEFRAMES, TRADING_CONFIG,
    TradingEngine, SimTradeManager,
    calculate_indicators, get_client,
)
from strategies import check_signal, STRATEGY_REGISTRY
from core.at_scenario_analyzer import check_core_signal
from core.trade_visualizer import TradeVisualizer
```

---

## Next Steps

1. **Portfolio System** - $2000 balance ile gerçekçi backtest (yeni yazılacak)
2. **Entry Timing** - Limit order ile slippage azalt
3. **TP/SL Optimization** - Risk/reward ratio iyileştir
4. **Trade Pattern Analysis** - Kaybeden trade'lerde ortak patern bul

---

*Index auto-generated. Last update: 2026-01-03*
