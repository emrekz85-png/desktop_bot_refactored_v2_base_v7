# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

---

## ⚡ ACTIVE FOCUS

**ÖNCE [FOCUS.md](FOCUS.md) DOSYASINI OKU!**

**Son Güncelleme:** 2026-01-04
**Aktif çalışma:** Quick Failure Predictor - **IMPLEMENTED & PASSING**

### SON SONUÇLAR (BTCUSDT 15m, 1 Year)

| Metric | Eski | **YENİ** | Değişim |
|--------|------|----------|---------|
| **Trades** | 26 | **15** | -42% |
| **Win Rate** | 46.2% | **60.0%** | **+13.8%** |
| **PnL** | +$72.99 | **+$91.12** | **+24.8%** |
| **Max DD** | $35.70 | **$24.15** | **-32.3%** |
| **Profit Factor** | 1.45 | **2.30** | **+58.6%** |

### Tek Komut: `run.py`

```bash
# Quick test (yeni config: regime + at_flat_filter + min_sl_filter + quick_failure_predictor)
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

**Current Version:** v2.2.0 - Pattern Integration & Momentum Exit

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

## Available Strategies

### 1. SSL Flow (Primary Strategy)

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

### 2. PBEMA Retest (NEW - Pattern 2)

**Core Concept:** "PBEMADAN güçlü fiyat sekmesi yaşanabilir!"

PBEMA Retest trades the PBEMA cloud as support/resistance after breakout:
1. **Breakout Detection** - Price crosses above/below PBEMA cloud
2. **Retest Confirmation** - Price returns to touch PBEMA boundary
3. **Bounce/Rejection** - Wick rejection confirms level is holding
4. **Entry** - Enter on confirmed bounce with TP at SSL baseline

**Signal Conditions (LONG):**
- Price broke above PBEMA in last 20 candles (>0.5% beyond cloud)
- Currently retesting PBEMA top as support (within 0.3% tolerance)
- Lower wick rejection (>15% of candle range)
- Optional: AlphaTrend buyers dominant
- Optional: Multiple retests (2+) for stronger confirmation

**Evidence:** Real trade analysis shows 33% of profitable setups use this pattern

**Implementation:**
- Strategy: `strategies/pbema_retest.py`
- Config: `core/config.py::PBEMA_RETEST_CONFIG`
- Test: `python test_pbema_retest.py`

**Usage:**
```python
from strategies import check_pbema_retest_signal

signal_type, entry, tp, sl, reason = check_pbema_retest_signal(
    df,
    index=-2,
    require_at_confirmation=False,  # Optional AT filter
    require_multiple_retests=False,  # Optional retest count filter
)
```

---

## Current Status (2026-01-04)

### Son Test Sonuçları (BTCUSDT 15m, 1 yıl)

| Sistem | Trade | WR | PnL | Max DD | PF |
|--------|-------|-----|-----|--------|-----|
| **SSL Flow + Quick Failure Predictor** | **15** | **60.0%** | **$91.12** ✅ | $24.15 | **2.30** |
| SSL Flow (eski config) | 26 | 46.2% | $72.99 | $35.70 | 1.45 |
| PBEMA Retest | 107 | 45.8% | -$4.72 | - | - |

**Durum:** Quick Failure Predictor ile **%24.8 PnL artışı** ve **%32.3 risk azalması** sağlandı!

### Kullanılabilir Stratejiler

1. **SSL Flow (Ana)** - `regime + at_flat_filter + min_sl_filter + quick_failure_predictor` ← **YENİ DEFAULT**
2. **PBEMA Retest (Yeni)** - Çalışıyor, 107 trade/yıl (kayıplı)
3. **Momentum Exit** - Opsiyonel, WR artırır ama PnL düşürür

### Önerilen Kullanım

```bash
# Quick test (yeni config ile)
python run.py test BTCUSDT 15m

# Full pipeline
python run.py test BTCUSDT 15m --full
```

---

## Known Issues / TODOs

- [ ] AlphaTrend equality case handling (TradingView Pine Script compatibility)
- [ ] ATR percentile calculation optimization
- [ ] PR-2: Carry-forward config system testing
- [ ] Out-of-sample (2024) validation with new config
- [x] **Quick Failure Predictor implementation** (2026-01-04) ← NEW
- [x] PBEMA Retest çalışır hale getirildi (v2.2.0)
- [x] Momentum Exit entegre edildi (v2.2.0)
- [x] Pattern filters düzeltildi (v2.2.0)
- [x] Optimizer determinism fix (v43.x)
- [x] Look-ahead bias fix (v1.10.1)
- [x] Lookback days increase 30→60 (v1.13.0)

---

## Quick Reference

### Key Files

| Purpose | File |
|---------|------|
| Main application | `desktop_bot_refactored_v2_base_v7.py` |
| **Comprehensive Test** | `run_comprehensive_test.py` |
| **Full Pipeline** | `runners/run_full_pipeline.py` |
| **Trade Visualizer** | `core/trade_visualizer.py` |
| Core config | `core/config.py` |
| Indicators | `core/indicators.py` |
| Core signal detection | `core/at_scenario_analyzer.py` |
| Trade logic | `core/trade_manager.py` |
| SSL Flow strategy | `strategies/ssl_flow.py` |
| **PBEMA Retest** | `strategies/pbema_retest.py` |
| **Momentum Exit** | `core/momentum_exit.py` |
| **Pattern Filters** | `core/pattern_filters.py` |

### Common Commands

```bash
# Comprehensive Test (Önerilen - tüm stratejileri karşılaştır)
python run_comprehensive_test.py BTCUSDT 15m --days 365

# Full Pipeline
python runners/run_full_pipeline.py --symbol BTCUSDT --timeframe 15m

# Trade Visualizer
python runners/run_trade_visualizer.py --symbol BTCUSDT --timeframe 15m \
    --entry-time "2025-03-02 21:00" --signal-type SHORT

# Run tests
pytest

# Start GUI
python desktop_bot_refactored_v2_base_v7.py
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

### PBEMA Retest Usage

```python
from strategies import check_pbema_retest_signal

signal_type, entry, tp, sl, reason = check_pbema_retest_signal(df, index=-2)
if signal_type:
    print(f"{signal_type} at {entry}, TP: {tp}, SL: {sl}")
```

### Momentum Exit Usage

```python
from runners.run_filter_combo_test import simulate_trade

# Normal trade (TP/SL exit)
trade = simulate_trade(df, idx, signal_type, entry, tp, sl)

# With momentum exit (exits early when momentum slows)
trade = simulate_trade(df, idx, signal_type, entry, tp, sl, use_momentum_exit=True)
print(f"Exit type: {trade['exit_type']}")  # TP, SL, MOMENTUM, or EOD
```

---

## Important Notes

### Key Learnings (v2.2.0)

1. **SSL Flow (Current Default) is optimal** - Pattern filters don't improve PnL
2. **PBEMA Retest works independently** - Don't apply regime filter (has own trend detection)
3. **Momentum Exit increases WR but decreases PnL** - Use for risk-averse trading
4. **Pattern filter thresholds matter** - Too strict = 0 signals, too loose = bad quality

### Failed Experiments

Many parameter changes have been tested and **FAILED**. Before making changes, check `CHANGELOG.md` to avoid repeating unsuccessful experiments.

**Key learnings:**
- Adding pattern filters to SSL Flow does NOT improve PnL
- Combined portfolio (SSL + PBEMA) reduces PnL due to overlap
- Momentum Exit trades WR for PnL (not always desirable)

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
