# SSL Flow Trading Bot

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](VERSION.md)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Private-red.svg)]()
[![Status](https://img.shields.io/badge/status-Active-success.svg)]()

> Advanced cryptocurrency futures trading bot implementing the SSL Flow trend-following strategy for Binance Futures.

---

## 🎯 Overview

**SSL Flow Trading Bot** is a sophisticated algorithmic trading system that combines technical analysis with robust risk management. The bot implements the SSL Flow strategy—a trend-following approach based on SSL Baseline, AlphaTrend, and PBEMA Cloud indicators.

### Core Concept

> **"There is a path from SSL HYBRID to PBEMA cloud!"**

The strategy identifies high-probability trend continuation trades by:
1. **SSL Baseline (HMA60)** - Identifies trend direction and support/resistance
2. **AlphaTrend** - Confirms buyer/seller dominance to filter false signals
3. **PBEMA Cloud (EMA200)** - Provides take-profit targets

---

## ✨ Key Features

### 🎯 **Advanced Strategy Implementation**
- **SSL Flow Strategy** with multi-indicator confluence
- **Walk-forward optimization** to prevent overfitting
- **Regime-aware trading** (trending/ranging/transitional)
- **Multiple timeframe support** (1m, 5m, 15m, 1h, 4h)

### 🛡️ **Robust Risk Management**
- **R-multiple based position sizing** (1.75% per trade)
- **Dynamic circuit breaker** (20% max drawdown)
- **Kelly Criterion adjustment** based on drawdown
- **Correlation-aware position limits**
- **Portfolio-level risk caps** (5% max exposure)

### 📊 **Sophisticated Analytics**
- **Real-time performance tracking**
- **Drawdown monitoring and recovery**
- **Win rate and R-multiple analysis**
- **Regime distribution analysis**
- **Trade visualization and charts**

### 🔧 **Production-Ready Architecture**
- **Modular design** with clean separation of concerns
- **Comprehensive testing** (unit, integration, backtesting)
- **Secure credential management** (environment variables)
- **Thread-safe operations** for concurrent trading
- **Efficient caching** for performance optimization

---

## 📊 Performance

### Recommended Portfolio (2025 Full-Year Backtest)

| Symbol | PnL | Win Rate | Trades | Max DD | Status |
|--------|-----|----------|--------|--------|--------|
| BTCUSDT | Best | ~80% | High | Low | ✅ **RECOMMENDED** |
| ETHUSDT | Good | ~78% | Medium | Low | ✅ **RECOMMENDED** |
| LINKUSDT | Excellent | ~82% | Medium | Low | ✅ **RECOMMENDED** |

**Combined Portfolio (BTC+ETH+LINK):**
- **PnL:** ~$145 (H2 2025)
- **Win Rate:** ~79%
- **Max Drawdown:** ~$44
- **Total Trades:** Moderate frequency

> ⚠️ **Important:** Avoid all other symbols—they either lost money or produced no trades.

### Current Version Performance (v2.0.0)

**Improvements from Baseline:**
- PnL: **+$122** improvement (-$162 → -$40)
- Max Drawdown: **-$110** reduction ($208 → $98)
- Better TradingView indicator parity

**Trade-offs:**
- Lower trade frequency (51 → 13 trades)
- Win rate decreased (41% → 31%)
- Still optimization-dependent

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Binance Futures Account** (Testnet or Live)
- **Git** for version control

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd desktop_bot_refactored_v2_base_v7

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional but recommended)
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

### Configuration

1. **API Keys** (for live trading):
   - Get Binance Futures API keys from [Binance](https://www.binance.com/en/futures/BTCUSDT)
   - Store securely (never commit to git)

2. **Telegram Notifications** (optional):
   - Create bot via [@BotFather](https://t.me/botfather)
   - Get chat ID from [@userinfobot](https://t.me/userinfobot)
   - Configure in `core/config.py` or environment variables

### Running the Bot

#### Backtest Mode
```bash
# Simple backtest
python run_backtest.py

# Walk-forward test (recommended)
python run_rolling_wf_test.py

# Full year test
python run_rolling_wf_test.py --full-year
```

#### GUI Mode
```bash
# Start with GUI
python desktop_bot_refactored_v2_base_v7.py

# Start headless (no GUI)
python desktop_bot_refactored_v2_base_v7.py --headless
```

#### Live Trading (Testnet)
```bash
# Use testnet for testing
python desktop_bot_refactored_v2_base_v7.py --testnet
```

---

## 📁 Project Structure

```
desktop_bot_refactored_v2_base_v7/
│
├── README.md                           # This file
├── VERSION.md                          # Version history
├── CHANGELOG.md                        # Detailed changelog
├── CLAUDE.md                           # Developer guide for AI
│
├── core/                               # Core package
│   ├── __init__.py                     # Package exports
│   ├── config.py                       # Configuration constants
│   ├── indicators.py                   # Technical indicators
│   ├── trade_manager.py                # Trade execution logic
│   ├── risk_manager.py                 # Risk management
│   ├── binance_client.py               # Binance API wrapper
│   ├── optimizer.py                    # Strategy optimization
│   └── ...                             # Other core modules
│
├── strategies/                         # Trading strategies
│   ├── ssl_flow.py                     # SSL Flow strategy (main)
│   ├── router.py                       # Strategy router
│   └── base.py                         # Strategy base class
│
├── runners/                            # Execution scripts
│   ├── portfolio.py                    # Portfolio management
│   ├── rolling_wf.py                   # Walk-forward testing
│   └── ...                             # Other runners
│
├── ui/                                 # User interface
│   ├── main_window.py                  # PyQt5 GUI
│   └── workers.py                      # Background workers
│
├── tests/                              # Test suite
│   ├── test_signals.py                 # Strategy tests
│   ├── test_risk_manager.py            # Risk management tests
│   └── manual/                         # Manual testing scripts
│
├── docs/                               # Documentation
│   ├── README.md                       # Documentation index
│   ├── REQUIREMENTS.md                 # Formal requirements (IEEE 830)
│   ├── ARCHITECTURE.md                 # Technical architecture
│   ├── HEDGE_FUND_DUE_DILIGENCE_REPORT.md  # Expert analysis
│   ├── reports/                        # Analysis reports
│   ├── guides/                         # Quick reference guides
│   └── research/                       # Research documents
│
├── scripts/                            # Utility scripts
│   └── experimental/                   # Experimental features
│
├── examples/                           # Usage examples
│   └── ...                             # Example scripts
│
└── data/                               # Data directory (gitignored)
    ├── logs/                           # Log files
    ├── cache/                          # Performance cache
    └── configs/                        # Saved configurations
```

---

## 📚 Documentation

### Essential Reading

| Document | Description | Audience |
|----------|-------------|----------|
| **[CLAUDE.md](CLAUDE.md)** | Developer quick reference and project overview | Developers, AI assistants |
| **[VERSION.md](VERSION.md)** | Version history and test results | All users |
| **[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)** | Formal requirements specification (IEEE 830) | Technical users |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Technical architecture and design | Developers |
| **[docs/HEDGE_FUND_DUE_DILIGENCE_REPORT.md](docs/HEDGE_FUND_DUE_DILIGENCE_REPORT.md)** | Independent expert analysis | Traders, investors |

### Quick Guides

- **[Optimization Guide](docs/guides/)** - Strategy optimization best practices
- **[Walk-forward Testing](docs/guides/)** - Preventing overfitting
- **[Risk Management](docs/RISK_MANAGEMENT_SPEC.md)** - Risk management specification

### Research & Analysis

- **[Research Documents](docs/research/)** - Experimental findings
- **[Analysis Reports](docs/reports/)** - Code analysis and improvements
- **[Trade Analysis](docs/)** - Deep dive into trade patterns

---

## 🔧 Configuration

### Strategy Parameters

Key parameters in `core/config.py`:

```python
# SSL Flow Strategy
DEFAULT_STRATEGY_CONFIG = {
    "ssl_period": 60,              # SSL Baseline period
    "atr_period": 14,              # ATR period
    "atr_multiplier": 2.5,         # AlphaTrend ATR multiplier
    "pbema_period": 200,           # PBEMA cloud period
    "rsi_period": 14,              # RSI period
    "adx_period": 14,              # ADX period

    # Filter settings
    "skip_wick_rejection": True,   # v2.0.0 change
    "flat_threshold": 0.002,       # v2.0.0 change (was 0.001)
    "min_pbema_distance": 0.004,   # Minimum SSL-PBEMA distance

    # Trade management
    "partial_trigger_1": 0.40,     # First partial profit at 40% TP
    "partial_fraction_1": 0.33,    # Take 33% profit
    "partial_trigger_2": 0.70,     # Second partial at 70% TP
    "partial_fraction_2": 0.50,    # Take 50% of remaining
}
```

### Risk Management

```python
TRADING_CONFIG = {
    "risk_per_trade": 0.0175,      # 1.75% per trade
    "max_portfolio_risk": 0.05,    # 5% total portfolio risk
    "circuit_breaker_dd": 0.20,    # 20% drawdown stops trading
    "recovery_required": 0.05,     # 5% recovery to resume
}
```

### Symbols & Timeframes

```python
# Recommended symbols
SYMBOLS = ["BTCUSDT", "ETHUSDT", "LINKUSDT"]

# Supported timeframes
TIMEFRAMES = ["15m", "1h"]  # Recommended: 15m, 1h
```

---

## 🧪 Testing

### Run Test Suite

```bash
# All tests
pytest

# Specific test file
pytest tests/test_signals.py

# With coverage
pytest --cov=core --cov=strategies
```

### Manual Testing

```bash
# Test signal detection
python tests/manual/test_signal_detection.py

# Test risk management
python tests/manual/test_risk_manager.py

# Test regime filter
python tests/manual/test_regime_filter.py
```

### Backtesting

```bash
# Quick backtest (30 days)
python run_backtest.py

# Walk-forward test (prevents overfitting)
python run_rolling_wf_test.py --lookback 60 --forward 7

# Full year test (recommended before live trading)
python run_rolling_wf_test.py --full-year
```

---

## 🔐 Security

### API Key Management

**Never commit API keys to git!**

Use environment variables:
```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

Or use a `.env` file (add to `.gitignore`):
```bash
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
```

### Credentials Storage

- Telegram credentials: `data/telegram_config.json` (gitignored)
- Config files: `data/configs/` (gitignored)
- Logs: `data/logs/` (gitignored)

### Security Best Practices

✅ **DO:**
- Use Testnet for testing
- Enable IP whitelist on Binance
- Use withdrawal restrictions
- Monitor logs regularly
- Keep dependencies updated

❌ **DON'T:**
- Commit credentials
- Share API keys
- Disable security features
- Run with admin privileges
- Trust unverified code changes

---

## 🚨 Risk Disclaimer

**⚠️ IMPORTANT: READ BEFORE USING**

This software is provided for **educational and research purposes only**.

- **Trading cryptocurrencies involves substantial risk of loss**
- **Past performance does not guarantee future results**
- **The bot can lose money, even with proper configuration**
- **You are solely responsible for your trading decisions**
- **No warranty or guarantee of profitability is provided**
- **Start with small amounts and use Testnet first**

**By using this software, you acknowledge and accept all risks involved in automated trading.**

---

## 🤝 Contributing

This is a private project, but improvements are welcome:

1. **Test thoroughly** - All changes must pass existing tests
2. **Document changes** - Update relevant documentation
3. **Follow patterns** - Match existing code style
4. **Update CHANGELOG** - Record all changes in `docs/CHANGELOG.md`
5. **Version tracking** - Use version system in `core/version.py`

### Failed Experiments

Before implementing changes, check `docs/CHANGELOG.md` → "Failed Experiments" section to avoid repeating unsuccessful experiments.

---

## 📋 Known Issues

See [CLAUDE.md](CLAUDE.md) for current issues and TODOs:

- [ ] AlphaTrend equality case handling (TradingView compatibility)
- [ ] ATR percentile calculation optimization
- [ ] PR-2: Carry-forward config system testing
- [x] Optimizer determinism fix (v43.x)
- [x] Look-ahead bias fix (v1.10.1)
- [x] Lookback days increase 30→60 (v1.13.0)

---

## 🗺️ Roadmap

### Short-term (v2.x)
- [ ] Improve trade frequency while maintaining edge
- [ ] Enhanced regime filtering
- [ ] Multi-symbol correlation management
- [ ] Advanced position sizing algorithms

### Medium-term (v3.x)
- [ ] Machine learning integration for parameter optimization
- [ ] Advanced risk analytics dashboard
- [ ] Cloud deployment support
- [ ] Multi-exchange support

### Long-term (v4.x+)
- [ ] Portfolio optimization across strategies
- [ ] Advanced market microstructure analysis
- [ ] Options and derivatives support
- [ ] Institutional-grade reporting

---

## 📞 Support

### Documentation
- Check `docs/` folder for detailed documentation
- Read `CLAUDE.md` for developer quick reference
- Review `docs/ARCHITECTURE.md` for technical details

### Issues
- Review known issues in [Known Issues](#-known-issues) section
- Check `docs/CHANGELOG.md` for failed experiments
- Consult `docs/HEDGE_FUND_DUE_DILIGENCE_REPORT.md` for expert analysis

### Testing
- Use Testnet before live trading
- Start with small position sizes
- Monitor logs in `data/logs/`

---

## 📝 License

**Private/Proprietary** - All rights reserved.

This code is not licensed for public use, modification, or distribution without explicit permission.

---

## 🙏 Acknowledgments

- **SSL Baseline** indicator concept
- **AlphaTrend** indicator by Kivanc Ozbilgic
- **TradingView** community for technical analysis insights
- **Binance** for robust futures API
- **Python** ecosystem for excellent libraries

---

## 📊 Version Information

**Current Version:** v2.0.0 - Indicator Parity Fix
**Release Date:** 2026-01-01
**Status:** Active Development

See [VERSION.md](VERSION.md) for detailed version history and test results.

---

## 🔗 Quick Links

- [Installation](#installation)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Performance](#-performance)
- [Risk Disclaimer](#-risk-disclaimer)

---

**Built with ❤️ for algorithmic trading**

*Remember: The best strategy is the one you understand and can stick with through different market conditions.*
