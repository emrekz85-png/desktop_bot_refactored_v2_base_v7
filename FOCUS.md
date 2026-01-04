# FOCUS: Trading Bot Test System

**Son Guncelleme:** 2026-01-04
**Aktif Calisma:** Quick Failure Predictor Implementation

---

## CLAUDE ICIN OZET

Bu proje bir **crypto futures trading bot**. Ana strateji **SSL Flow** (trend-following).

### run.py Nedir?

`run.py` tek giris noktasi. Iki modu var:

1. **Quick Test** (default): Sabit config ile hizli test
2. **Full Pipeline** (`--full`): Filter discovery + Rolling WF + Portfolio

### Pipeline Akisi (--full mode)

```
1. FETCH DATA         → Binance'den veri cek
2. BASELINE           → Sadece regime filter ile test (referans)
3. FILTER DISCOVERY   → Incremental filter kombinasyonlarini dene
4. ROLLING WF         → 7-gun pencerelerle walk-forward validation
5. PORTFOLIO          → Gercekci position sizing ile backtest
6. VERDICT            → PASS/MARGINAL/FAIL karari
```

### Kullanilan Moduller

```python
# run.py icinde import edilen moduller:
from core import get_client, calculate_indicators, set_backtest_mode
from core.at_scenario_analyzer import check_core_signal  # Sinyal uretimi
from core.simple_portfolio import PortfolioConfig, run_portfolio_backtest
from runners.run_filter_combo_test import apply_filters  # Filter logic
```

### Kritik Fonksiyonlar

| Fonksiyon | Gorev |
|-----------|-------|
| `fetch_data()` | Binance'den OHLCV veri ceker |
| `run_backtest()` | Basit backtest, dollar PnL hesaplar |
| `run_filter_discovery()` | Incremental filter kombinasyonlarini test eder |
| `run_rolling_wf()` | 7-gun pencerelerle WF validation |
| `run_portfolio_backtest()` | Gercekci position sizing ile son test |
| `run_full_pipeline()` | Tum adimlari sirayla calistirir |

### Onemli Bug Fix (2026-01-03)

`run_backtest()` fonksiyonunda PnL hesabi duzeltildi:
```python
# ESKI (HATALI): Tum kayiplar icin sabit -1 kullaniyordu
trades.append({"pnl": -1, "win": False})

# YENI (DOGRU): Gercek dollar PnL hesapliyor
exit_price = sl * (1 - slippage)
pnl = (exit_price - actual_entry) / actual_entry * position_size
pnl -= position_size * fee * 2  # Slippage + fee dahil
```

---

## TEK KOMUT: `run.py`

```bash
# Quick test (sabit config: regime + at_flat_filter + min_sl_filter)
python run.py test BTCUSDT 15m

# Full pipeline (discovery + WF + portfolio)
python run.py test BTCUSDT 15m --full

# Quick 90-day test
python run.py test BTCUSDT 15m --quick

# Visualize trades from latest test
python run.py viz BTCUSDT 15m

# Show all test results
python run.py report
```

**Output:** `data/results/{SYMBOL}_{TF}_{TIMESTAMP}/`

---

## SON SONUCLAR (BTCUSDT 15m, 1 Year)

### YENI CONFIG (2026-01-04) - Quick Failure Predictor

| Metric | Eski | **YENI** | Degisim |
|--------|------|----------|---------|
| **Filters** | regime + at_flat + min_sl | **+ quick_failure_predictor** | +1 filter |
| **Trades** | 26 | **15** | -42% |
| **Win Rate** | 46.2% | **60.0%** | **+13.8%** |
| **PnL** | +$72.99 | **+$91.12** | **+24.8%** |
| **Max DD** | $35.70 | **$24.15** | **-32.3%** |
| **Profit Factor** | 1.45 | **2.30** | **+58.6%** |

**VERDICT: PASS** - Quick Failure Predictor basarili!

### Key Insight: Quick Failure Predictor

Analiz raporunda LONG trade'lerin %40'inin "quick failure" (≤20 bar icinde SL hit) oldugu tespit edildi.
Bu trade'lerin ortak ozellikleri:
1. SSL Baseline slope signal yonunde degil
2. SSL Baseline erratic/unstable
3. Son 3 mumda SSL yon degisikligi

`quick_failure_predictor` bu 3 faktoru birlikte degerlendirip LONG quick failure'lari engelliyor.

### Min SL Filter (Onceki Iyilestirme)

SL distance < 1.5% olan trade'ler noise ile SL hit oluyor.

| SL Distance | Win Rate | Status |
|-------------|----------|--------|
| < 1.0% | 23.7% | REJECT |
| 1.0 - 1.5% | 34.2% | REJECT |
| > 1.5% | 55.0%+ | ACCEPT |

**min_sl_filter = 1.5%** ile kayiplar kontrol altina alindi.

---

## MEVCUT KONFIGURASYON

```python
# run.py icindeki sabit ayarlar (2026-01-04 guncellendi)
DEFAULT_FILTERS = ["regime", "at_flat_filter", "min_sl_filter", "quick_failure_predictor"]

# Portfolio settings (core/simple_portfolio.py)
initial_balance = 1000
risk_per_trade = 1%   # $10 per trade
leverage = 10x
max_position = 10%
slippage = 0.05%
fee = 0.07%
```

---

## DOSYA YAPISI

```
run.py                              # TEK GIRIS NOKTASI
├── test   → Full pipeline          # Signal + Filter + Portfolio
├── viz    → Trade charts           # Visualize results
├── report → Summary                # All test results
└── list   → Result dirs            # List outputs

core/
├── at_scenario_analyzer.py         # check_core_signal() - sinyal uretimi
├── simple_portfolio.py             # Portfolio backtest logic
├── indicators.py                   # calculate_indicators()
└── config.py                       # Trading config

runners/
├── run_filter_combo_test.py        # apply_filters() fonksiyonu
├── rolling_wf.py                   # Rolling WF validation
└── ...                             # Diger legacy runner'lar

data/results/                       # CONSOLIDATED OUTPUT
├── BTCUSDT_15m_20260103_*/
│   ├── result.json                 # Main summary
│   ├── signals.json                # All signals
│   ├── trades.json                 # All trades
│   ├── summary.txt                 # Human-readable
│   └── charts/                     # Visualizations
```

---

## AKTIF FILTRELER

| Filter | Aciklama | Default |
|--------|----------|---------|
| `regime` | Sadece trending market (ADX-based) | ON |
| `at_flat_filter` | AlphaTrend flat degilse trade | ON |
| `min_sl_filter` | SL >= 1.5% olmali | ON |
| `quick_failure_predictor` | **YENİ:** LONG quick failure riski tespit et | ON |

### Yeni Filtreler (2026-01-04)

| Filter | Aciklama | Etki |
|--------|----------|------|
| `ssl_slope_direction` | SSL slope yonu signal yonunde olmali | WR +11%, PnL -$1.27 |
| `ssl_stability` | SSL baseline stable olmali | WR ayni, PnL -$14 |
| `quick_failure_predictor` | **ONERILEN** - 3 faktoru birlikte kullanir | **WR +13.8%, PnL +$18** |

Inactive (gelecek arastirma icin):
- `adx_filter`, `ssl_touch`, `rsi_filter`, `pbema_distance`

---

## SONRAKI ADIMLAR

1. [x] min_sl_filter implementation
2. [x] Portfolio sizing integration
3. [x] Simplified run.py
4. [x] PnL calculation bug fix
5. [x] **Quick Failure Predictor implementation** (2026-01-04)
6. [ ] ETH/SOL test with same config
7. [ ] Multi-symbol portfolio optimization
8. [ ] Out-of-sample (2024) validation with new config

---

## NOTLAR

- Her test `data/results/` klasorune kaydedilir
- `python run.py report` ile tum sonuclari gor
- Trade chart'lari `viz` komutu ile otomatik olusturulur
- Eski `data/pipeline_reports/`, `data/filter_combo_logs/` legacy

---

**Bu dosyayi her session basinda oku.**
