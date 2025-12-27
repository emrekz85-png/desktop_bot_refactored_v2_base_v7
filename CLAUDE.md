# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Strateji Mantigi Ozeti (Strategy Logic Summary)

### SSL Flow Stratejisi (Aktif Strateji)

**Temel Konsept:** "SSL HYBRID'den PBEMA bulutuna bir yol vardir!"

SSL Flow, trend takip eden bir stratejidir. Ana mantik:
1. **SSL HYBRID (HMA60)** trend yonunu belirler (destek/direnc)
2. **AlphaTrend** alici/satici hakimiyetini onaylar (sahte sinyalleri filtreler)
3. **PBEMA Cloud (EMA200)** kar hedefi olarak kullanilir

#### Giris Kosullari

**LONG Sinyali:**
- Fiyat SSL baseline (HMA60) ustunde
- AlphaTrend BUYERS dominant (mavi cizgi ustte, cizgi yukseliyor)
- Son 5 mumda baseline'a temas (retest) olmus
- Mum govdesi baseline ustunde
- PBEMA fiyatin ustunde (kar hedefine yer var)
- Alt fitil reddi (bounce confirmation)
- PBEMA baseline'in USTUNDE (hedef ulasilabilir - "yol var")
- RSI <= rsi_limit (asiri alimda degil)

**SHORT Sinyali:**
- Fiyat SSL baseline (HMA60) altinda
- AlphaTrend SELLERS dominant (kirmizi cizgi ustte, cizgi dusuyor)
- Son 5 mumda baseline'a temas (retest) olmus
- Mum govdesi baseline altinda
- PBEMA fiyatin altinda (kar hedefine yer var)
- Ust fitil reddi (rejection confirmation)
- PBEMA baseline'in ALTINDA (hedef ulasilabilir - "yol var")
- RSI >= 100-rsi_limit (asiri satimda degil)

#### PBEMA-SSL Overlap Kontrolu (Son Eklenen)

**Kritik Kural:** "PBEMA ve SSL Hybrid bantlari IC ICE oldugunda islem ALINMAZ"

```python
OVERLAP_THRESHOLD = 0.005  # %0.5 esik degeri
baseline_pbema_distance = abs(baseline - pbema_mid) / pbema_mid
is_overlapping = baseline_pbema_distance < OVERLAP_THRESHOLD
```

- LONG icin: PBEMA hedefi baseline'in ustunde olmali
- SHORT icin: PBEMA hedefi baseline'in altinda olmali
- Overlap varsa sinyal uretilmez ("SSL-PBEMA Overlap (No Flow)")

#### Gostergeler

| Gosterge | Hesaplama | Kullanim |
|----------|-----------|----------|
| SSL Baseline | HMA(close, 60) | Trend yonu, destek/direnc |
| PBEMA Cloud | EMA(high, 200) & EMA(close, 200) | TP hedefi |
| AlphaTrend | ATR + MFI/RSI tabanli | Flow onaylama |
| RSI(14) | Standart RSI | Asiri alim/satim filtresi |
| ADX(14) | Average Directional Index | Trend gucu filtresi (min 15) |
| Keltner Channels | baseline +/- EMA(TR)*0.2 | Volatilite bantlari |

#### TP/SL Hesaplama

**Take Profit:**
- LONG: `pb_ema_bot` (PBEMA cloud alt siniri)
- SHORT: `pb_ema_top` (PBEMA cloud ust siniri)

**Stop Loss:**
- LONG: `min(swing_low * 0.998, baseline * 0.998)` - Son 20 mumun en dusugu veya baseline
- SHORT: `max(swing_high * 1.002, baseline * 1.002)` - Son 20 mumun en yuksegi veya baseline

**Minimum SL Mesafesi (Opsiyonel):**
- BTC/ETH: %1.0 minimum SL mesafesi
- Altcoinler: %1.5 minimum SL mesafesi
- `sl_validation_mode`: "off" (baseline), "reject", veya "widen"

---

## Dosya Yapisi ve Sorumluluklar (File Structure)

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

### Modul Bagimliliklari

| Modul | Bagimliliklar | Sorumluluk |
|-------|---------------|------------|
| `core/config.py` | - | Tum sabitler ve konfigurasyonlar |
| `core/indicators.py` | config, pandas_ta | Teknik gosterge hesaplamalari |
| `core/trade_manager.py` | config, utils | Trade yonetimi (acma/kapama/guncelleme) |
| `strategies/ssl_flow.py` | base | SSL Flow sinyal uretimi |
| `strategies/router.py` | ssl_flow, keltner_bounce | Strateji yonlendirme |
| `ui/main_window.py` | core/*, strategies/* | PyQt5 GUI |

---

## Aktif Gelistirme Alanlari (Active Development)

### Son Yapilan Degisiklikler

1. **Optimizer Determinism Fix (v43.x) [CRITICAL]**
   - **Problem:** Ayni parametreler farkli sonuclar uretiyordu ($191 varyans)
   - **Cozum:**
     - Random seed eklendi: `random.seed(42)`, `np.random.seed(42)`
     - Sonuclar config hash'e gore sirala (deterministic ordering)
     - Float karsilastirmalarinda epsilon toleransi
     - Esitlik durumunda config hash ile tie-breaking
   - **Konum:** `core/optimizer.py` ve `run_rolling_wf_test.py`

2. **Batch Mode Optimization (v43.x)**
   - Coklu sembol testleri icin paralel islem
   - Tek run_id ile tum semboller (folder creation bug fix)
   - ~4x hizlanma (8 sembol: 32 dk -> 8 dk)
   - Kullanim: `run_rolling_wf_test.py --batch`

3. **Weighted Scoring System (v43.x) [EXPERIMENTAL]**
   - AND logic alternatifi olarak puanlama sistemi
   - 10-puanlik olcek, konfigurasyonlu esik degeri
   - **Sonuc:** Test sonrasi etkisiz - 4 core filter mandatory kaldigi icin
   - Core filters (relaxable degil): `price_above_baseline`, `at_buyers_dominant`, `not at_is_flat`, `rsi_ok`
   - Scoring sadece secondary filters icin: `baseline_touch`, `wick_rejection`, `pbema_distance`, `body_position`, `overlap`

4. **PBEMA-SSL Overlap Fix (v42.x)**
   - SSL baseline ve PBEMA bulutunun ic ice oldugu durumlarda islem engelleme
   - %0.5 esik degeri ile overlap kontrolu
   - LONG icin PBEMA > baseline, SHORT icin PBEMA < baseline kontrolu

5. **Minimum SL Distance (v42.x)**
   - Piyasa gurutusu nedeniyle erken stop-out'u onleme
   - BTC/ETH: %1.0, Altcoinler: %1.5 minimum mesafe
   - `sl_validation_mode`: "off" (baseline), "reject", "widen"
   - RR koruma: SL genisletildiginde TP de orantili olarak genisletilir

6. **Exit Profile System (v42.x)**
   - CLIP profili: Erken partial (%45), buyuk partial (%50), daha yuksek hit-rate
   - RUNNER profili: Gec partial (%70), kucuk partial (%33), kazananlari kostur

7. **Circuit Breaker (v40.2-v42.x)**
   - Stream-level: Max kayip, drawdown, ardisik full stop kontrolu
   - Global-level: Gunluk/haftalik kayip limitleri
   - Rolling E[R] kontrolu ile edge kaybi tespiti

### Performans Optimizasyon Firsatlari

1. **Lazy Imports** - `pandas_ta` gibi agir kutuphaneler gerektiginde yuklenir
2. **NumPy Array Pre-extraction** - Hot loop'larda `df.iloc[i]` yerine array kullanimi
3. **Parallel Data Fetching** - ThreadPoolExecutor ile max 5 worker
4. **Config Caching** - `BEST_CONFIG_CACHE` ile disk I/O azaltma
5. **Master Cache** - Rolling WF testlerinde veri tekrar kullanimi

### Onerilen Semboller (Recommended Symbols)

**Test Donemi:** 2025-01-01 - 2025-12-01 (determinism fix sonrasi guvenilir sonuclar)

| Sembol | Durum | Notlar |
|--------|-------|--------|
| BTCUSDT | **ONERILEN** | En iyi performans |
| ETHUSDT | **ONERILEN** | Tutarli sonuclar |
| LINKUSDT | **ONERILEN** | Yuksek win rate |
| DOGEUSDT | Islem Yok | Gecerli config bulunamadi |
| SUIUSDT | Islem Yok | Gecerli config bulunamadi |
| FARTCOINUSDT | Islem Yok | Gecerli config bulunamadi |
| LTCUSDT | Kaybettiren | Tutarsiz |
| BNBUSDT | Kaybettiren | Dusuk performans |
| HYPEUSDT | Kaybettiren | Yuksek drawdown |
| XRPUSDT | Kaybettiren | Cok islem, negatif beklenti |
| SOLUSDT | Kaybettiren | Bazi donemlerde %0 win rate |

**Onerilen Portfoy:** BTC + ETH + LINK
- PnL (Haziran-Aralik): $157.10
- Toplam Islem: 25
- Win Rate: %81
- Max Drawdown: ~$140

### Bilinen Sorunlar / TODO'lar

- [ ] AlphaTrend equality case handling (TradingView Pine Script uyumu)
- [ ] ATR percentile hesaplama optimizasyonu
- [ ] Regime detection multiplier devre disi (v43 - test edilecek)
- [ ] PR-2: Carry-forward config sistemi test edilecek
- [x] Optimizer determinism fix tamamlandi (v43.x)

---

## Filter Combination Discovery System

### Overview

The Filter Discovery System finds the optimal combination of AND filters for the SSL Flow strategy to balance signal quality with trade frequency.

**Problem:** Currently 6+ filters must ALL pass → only 9 trades/year (over-filtering)

**Solution:** Test 2^6 = 64 filter combinations to find the sweet spot

### Quick Start

```bash
# Phase 1: Pilot on BTCUSDT-15m (recommended first run)
python run_filter_discovery.py --pilot

# Phase 2: Validate top result on holdout data
python run_filter_discovery.py --validate 0 --results data/filter_discovery_runs/pilot_YYYYMMDD_HHMMSS/results.json

# Phase 3: Full test on all symbols/timeframes (if pilot looks good)
python run_filter_discovery.py --full
```

### Toggleable Filters (6 total)

1. **adx_filter**: ADX >= adx_min (trend strength)
2. **regime_gating**: ADX_avg >= threshold over N bars (window-level regime)
3. **baseline_touch**: Price touched baseline in lookback (entry timing)
4. **pbema_distance**: Min distance to PBEMA target (room for profit)
5. **body_position**: Candle body above/below baseline (confirmation)
6. **ssl_pbema_overlap**: Check for SSL-PBEMA overlap (flow existence)

**CORE filters (NEVER toggle - always ON):**
- AlphaTrend dominance (buyers/sellers)
- Price position vs baseline (LONG/SHORT direction)
- RR validation (min_rr threshold)

### Key Files

- **`core/filter_discovery.py`** - Main discovery engine (450 lines)
- **`run_filter_discovery.py`** - CLI runner (550 lines)
- **`docs/filter_discovery.md`** - Comprehensive documentation

### Expected Output

- `data/filter_discovery_runs/{run_id}/results.json` - Full results for all 64 combinations
- `data/filter_discovery_runs/{run_id}/top_10.txt` - Human-readable top 10 report
- `data/filter_discovery_runs/{run_id}/filter_pass_rates.json` - Diagnostic statistics

### Data Split

- **60% Train** - Search period (find best combinations)
- **20% Walk-Forward** - Validation period (detect overfitting)
- **20% Holdout** - Final test (never seen during search)

### Scoring Function

```python
score = (
    expected_r * 0.40 +      # 40% weight on E[R]
    freq_bonus * 0.30 +       # 30% weight on trade frequency
    sharpe * 0.10 +           # 10% weight on consistency
    wr_factor * 0.20          # 20% weight on win rate
) * net_pnl
```

**Success criteria:**
- E[R] >= 0.08 (same or better than baseline)
- Trades >= 15-20/year (significant increase from 9)
- WF E[R] / Train E[R] >= 0.50 (not overfitted)
- Holdout E[R] >= 0.05 (validates on unseen data)

For detailed documentation, see `docs/filter_discovery.md`.

---

## Test Prosedürleri (Test Procedures)

### Pytest Komutlari

```bash
# Tum testleri calistir
pytest

# Belirli test dosyasi
pytest tests/test_signals.py

# Verbose cikti
pytest -v

# Ilk hatada dur
pytest -x

# Yavas testleri atla
pytest -m "not slow"

# Sadece parity testleri (live/sim)
pytest -m parity

# Sadece integration testleri
pytest -m integration

# Coverage raporu
pytest --cov=core --cov=strategies --cov-report=html
```

### Backtest Calistirma

```bash
# Hizli backtest (3 sembol, 2000 mum)
python run_backtest.py

# CLI backtest with options
python desktop_bot_refactored_v2_base_v7.py --headless \
    --symbols BTCUSDT ETHUSDT \
    --timeframes 5m 15m 1h \
    --candles 30000 \
    --no-optimize
```

### Rolling Walk-Forward Test

```bash
# Varsayilan test (son 6 ay, weekly mode)
python run_rolling_wf_test.py

# 2025 tam yil testi
python run_rolling_wf_test.py --full-year

# Hizli test (3 ay, az sembol)
python run_rolling_wf_test.py --quick

# Tarih araligii belirtme
python run_rolling_wf_test.py --start-date 2025-06-01 --end-date 2025-12-01

# Optimized mode (master cache + parallel) - default
python run_rolling_wf_test.py --fast

# Sequential mode (debug icin)
python run_rolling_wf_test.py --no-fast
```

**Rolling WF Modlari:**
- `fixed`: Sabit config, re-optimization yok
- `weekly`: Haftalik re-optimization (30 gun lookback, 7 gun forward)
- `monthly`: Aylik re-optimization
- `triday`: 3 gunluk re-optimization

### Strateji Dogrulama (stratcheck)

```bash
# Strateji saglik kontrolu
python run_strategy_sanity_tests.py

# Optimizer diagnostikleri
python run_optimizer_diagnostic.py

# Strateji performans analizi
python run_strategy_autopsy.py
```

### Backtest Dogrulama Kontrolleri

Backtest sonrasinda kontrol edilecekler:
1. Trade sayilari makul mu (cok az veya cok fazla degil)
2. Win rate beklenen aralikta mi (%45-65)
3. PnL pozitif mi
4. Console'da hata yok
5. Circuit breaker dogru calisiyor mu

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

### Strategy Configuration

`DEFAULT_STRATEGY_CONFIG` defines strategy parameters:
- `rr` - Risk/Reward ratio
- `rsi` - RSI threshold
- `at_active` - AlphaTrend filter active (MANDATORY for SSL Flow)
- `use_trailing` - Trailing stop enabled
- `strategy_mode` - "ssl_flow" (default) or "keltner_bounce" (disabled)
- `exit_profile` - "clip" or "runner"
- `sl_validation_mode` - "off", "reject", or "widen"

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

### Circuit Breaker
2-Level kill switch:
- **Stream-level:** Max loss (-$200), drawdown ($100), consecutive full stops (2)
- **Global-level:** Daily (-$400), weekly (-$800), max drawdown (20%)

---

## Build and Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Activate virtual environment (if using venv)
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Verify syntax
python3 -m py_compile desktop_bot_refactored_v2_base_v7.py
```

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

## Quick Reference

### Key Files
- Main code: `desktop_bot_refactored_v2_base_v7.py`
- Core config: `core/config.py`
- Indicators: `core/indicators.py`
- Trade logic: `core/trade_manager.py`
- SSL Flow strategy: `strategies/ssl_flow.py`
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

### Signal Debug
```python
# Check specific candle conditions with debug info
from strategies import check_signal
signal_type, entry, tp, sl, reason, debug_info = check_signal(
    df, config, index=-2, return_debug=True
)
print(debug_info)
```
