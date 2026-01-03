# FOCUS: Trading Bot Test System

**Son Guncelleme:** 2026-01-03
**Aktif Calisma:** Simplified Test Runner + Min SL Filter

---

## TEK KOMUT: `run.py`

```bash
# Full 1-year test (recommended)
python run.py test BTCUSDT 15m

# Quick 90-day test
python run.py test BTCUSDT 15m --quick

# Test all recommended symbols
python run.py test --all

# Visualize trades from latest test
python run.py viz BTCUSDT 15m

# Show all test results
python run.py report
```

**Output:** `data/results/{SYMBOL}_{TF}_{TIMESTAMP}/`

---

## SON SONUCLAR (BTCUSDT 15m, 1 Year)

| Metric | Deger |
|--------|-------|
| **Filters** | regime, at_flat_filter, min_sl_filter |
| **Signals** | 1683 raw → 40 final (97.6% filtered) |
| **Trades** | 26 |
| **Win Rate** | 46.15% |
| **PnL** | +$72.99 (+7.3%) |
| **Max DD** | $35.70 (3.57%) |
| **Profit Factor** | 1.45 |
| **Verdict** | PASS |

### Key Insight: Min SL Filter

SL distance < 1.5% olan trade'ler noise ile SL hit oluyor.

| SL Distance | Win Rate | Status |
|-------------|----------|--------|
| < 1.0% | 23.7% | REJECT |
| 1.0 - 1.5% | 34.2% | REJECT |
| > 1.5% | 55.0%+ | ACCEPT |

**min_sl_filter = 1.5%** ile kayiplar kontrol altina alindi.

---

## MEVCUT KONFIGÜRASYON

```python
# run.py icindeki ayarlar
BEST_FILTERS = ["regime", "at_flat_filter", "min_sl_filter"]

# Portfolio settings
initial_balance = 1000
risk_per_trade = 1%   # $10 per trade
leverage = 10x
max_position = 10%
slippage = 0.05%
fee = 0.07%
```

---

## DOSYA YAPISI (Simplified)

```
run.py                              # TEK GIRIS NOKTASI
├── test   → Full pipeline          # Signal + Filter + Portfolio
├── viz    → Trade charts           # Visualize results
├── report → Summary                # All test results
└── list   → Result dirs            # List outputs

data/results/                       # CONSOLIDATED OUTPUT
├── BTCUSDT_15m_20260103_*/
│   ├── result.json                 # Main summary
│   ├── signals.json                # All signals
│   ├── trades.json                 # All trades
│   ├── summary.txt                 # Human-readable
│   └── charts/                     # Visualizations
```

### Legacy (Deprecated)

Eski runner'lar hala calisiyor ama `run.py` tercih edilmeli:
```
runners/
├── run_full_pipeline.py     # Deprecated → use run.py test
├── run_filter_combo_test.py # Deprecated
├── run_portfolio_test.py    # Deprecated
└── ...
```

---

## WORKFLOW

```
1. python run.py test BTCUSDT 15m        # Full test
2. python run.py viz BTCUSDT 15m         # Check trades visually
3. python run.py report                   # Compare results
```

Yeni sembol/timeframe icin:
```
1. python run.py test ETHUSDT 15m --quick  # Quick validation
2. If PASS → python run.py test ETHUSDT 15m  # Full test
3. Update docs/SYMBOL_SETTINGS.md with results
```

---

## AKTIF FILTRELER

| Filter | Aciklama | Default |
|--------|----------|---------|
| `regime` | Sadece trending market | ON |
| `at_flat_filter` | AT flat degilse trade | ON |
| `min_sl_filter` | SL >= 1.5% olmali | ON |

Inactive (gelecek arastirma icin):
- `adx_filter`, `ssl_touch`, `rsi_filter`, `pbema_distance`

---

## SONRAKI ADIMLAR

1. [x] min_sl_filter implementation
2. [x] Portfolio sizing integration
3. [x] Simplified run.py
4. [ ] ETH/SOL test with same config
5. [ ] Multi-symbol portfolio optimization

---

## NOTLAR

- Her test `data/results/` klasorune kaydedilir
- `python run.py report` ile tum sonuclari gor
- Trade chart'lari `viz` komutu ile otomatik olusturulur
- Eski `data/pipeline_reports/`, `data/filter_combo_logs/` legacy

---

**Bu dosyayi her session basinda oku.**
