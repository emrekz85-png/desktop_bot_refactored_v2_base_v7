# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

---

## ⚡ ACTIVE FOCUS

**ÖNCE [FOCUS.md](FOCUS.md) DOSYASINI OKU!**

Aktif çalışma: **Integrated Test Pipeline**

### Tek Komut: `run.py`

```bash
# Quick test (sabit config)
python run.py test BTCUSDT 15m

# Full pipeline (discovery + WF + portfolio)
python run.py test BTCUSDT 15m --full

# Visualize trades
python run.py viz BTCUSDT 15m

# Show all results
python run.py report
```

### Pipeline Akisi (--full mode)
1. Fetch data → 2. Baseline → 3. Filter discovery → 4. Rolling WF → 5. Portfolio → 6. Verdict

**Detaylar icin FOCUS.md'ye bak.**

---

## Project Overview

**Cryptocurrency Futures Trading Bot** implementing SSL Flow strategy for Binance Futures.

**Current Version:** v2.0.0 - Indicator Parity Fix

> 📖 **For general information, see [README.md](README.md)**
> This document is specifically for AI assistants and developers working on the codebase.

### Key Documents

| Document | Description |
|----------|-------------|
| `README.md` | **User-facing documentation** - Installation, usage, quick start, performance |
| `docs/REQUIREMENTS.md` | **Formal requirements specification** - IEEE 830 format, all functional and non-functional requirements |
| `docs/ARCHITECTURE.md` | **Technical architecture** - File structure, module dependencies, development guidelines |
| `docs/CHANGELOG.md` | **Version history** - All changes, failed experiments, lessons learned |
| `docs/HEDGE_FUND_DUE_DILIGENCE_REPORT.md` | Independent expert analysis of SSL Flow strategy |
| `VERSION.md` | **Version tracking** - Current version info and test results |

---

## Strategy Summary (SSL Flow)

**Core Concept:** "There is a path from SSL HYBRID to PBEMA cloud!"

SSL Flow is a trend-following strategy:
1. **SSL Baseline (HMA60)** - Primary trend indicator (support/resistance)
2. **AlphaTrend** - Confirms buyer/seller dominance (filters false signals)
3. **PBEMA Cloud (EMA200)** - Take-profit target

**Signal Conditions (LONG):**
- Price above SSL baseline
- AlphaTrend buyers dominant
- Baseline touched in last 5 candles
- PBEMA above price and baseline (path exists)
- No PBEMA-SSL overlap (>0.5% distance)
- RSI below limit, ADX above minimum

**For full specification:** See `docs/REQUIREMENTS.md` Section 3.1

---

## Current Status (2026-01-03)

**⚠️ UYARI: Strateji şu anki haliyle maliyetler sonrası kârlı DEĞİL.**

Son pipeline sonuçları (BTCUSDT 15m, 1 yıl):
- Best Config: REGIME + at_flat_filter
- 242 trades, 31% win rate
- Ideal PnL: +$14.62
- Cost-Aware PnL: **-$5.70**
- Edge: 0.17% < Cost: 0.24%

**Öneri:** Trade etme, stratejiyi iyileştir.

**Çalışma Alanları:**
- Portfolio sistemi yazılacak
- Entry timing iyileştirmesi
- TP/SL optimizasyonu

---

## Known Issues / TODOs

- [ ] AlphaTrend equality case handling (TradingView Pine Script compatibility)
- [ ] ATR percentile calculation optimization
- [ ] PR-2: Carry-forward config system testing
- [x] Optimizer determinism fix (v43.x)
- [x] Look-ahead bias fix (v1.10.1)
- [x] Lookback days increase 30→60 (v1.13.0)

---

## Quick Reference

### Key Files

| Purpose | File |
|---------|------|
| Main application | `desktop_bot_refactored_v2_base_v7.py` |
| **Full Pipeline** | `runners/run_full_pipeline.py` |
| **Trade Visualizer** | `core/trade_visualizer.py` |
| Core config | `core/config.py` |
| Indicators | `core/indicators.py` |
| Core signal detection | `core/at_scenario_analyzer.py` |
| Trade logic | `core/trade_manager.py` |
| SSL Flow strategy | `strategies/ssl_flow.py` |

### Common Commands

```bash
# Full Pipeline (Önerilen - tüm analiz adımları)
python runners/run_full_pipeline.py --symbol BTCUSDT --timeframe 15m

# Pipeline with trade logging
python runners/run_full_pipeline.py --symbol BTCUSDT --timeframe 15m --save-trades

# Trade Visualizer
python runners/run_trade_visualizer.py --symbol BTCUSDT --timeframe 15m \
    --entry-time "2025-03-02 21:00" --signal-type SHORT

# Run tests
pytest

# Start GUI
python desktop_bot_refactored_v2_base_v7.py

# Start headless
python desktop_bot_refactored_v2_base_v7.py --headless
```

### Import Pattern

```python
from core import (
    SYMBOLS, TIMEFRAMES, TRADING_CONFIG,
    TradingEngine, SimTradeManager,
    calculate_indicators,
)
from strategies import check_signal, STRATEGY_REGISTRY
```

### Signal Debug

```python
from strategies import check_signal
signal_type, entry, tp, sl, reason, debug_info = check_signal(
    df, config, index=-2, return_debug=True
)
print(debug_info)
```

---

## Important Notes

### Failed Experiments

Many parameter changes have been tested and **FAILED**. Before making changes, check `docs/CHANGELOG.md` Section "Failed Experiments" to avoid repeating unsuccessful experiments.

**Key learnings:**
- Relaxing filters does NOT improve performance
- Position sizing increase breaks optimizer
- Hour-based filtering causes overfitting
- Current strategy is already near-optimal

### Determinism

The optimizer uses fixed random seeds for reproducibility:
```python
random.seed(42)
np.random.seed(42)
```

Same inputs MUST produce same outputs. If not, check determinism controls.

### Risk Management

- Per-trade risk: 1.75% of account
- Max portfolio risk: 5% of account
- Circuit breaker: -$200 per stream, 20% global drawdown

---

## For More Information

- **Requirements details:** `docs/REQUIREMENTS.md`
- **Architecture details:** `docs/ARCHITECTURE.md`
- **Version history:** `docs/CHANGELOG.md`
- **Strategy analysis:** `docs/HEDGE_FUND_DUE_DILIGENCE_REPORT.md`
