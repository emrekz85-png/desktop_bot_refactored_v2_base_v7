# Symbol & Timeframe Settings Registry

Bu dosya her sembol ve timeframe icin optimal ayarlari icerir.

---

## Portfolio Logic Aciklamasi

### Position Sizing Formulu

```
risk_amount = balance * risk_per_trade_pct    # $1000 * 1% = $10
position_notional = risk_amount / sl_distance_pct
position_size = position_notional / entry_price
```

**Ornek - +$46 Trade (2025-03-02 22:15 SHORT):**
```
Balance: $1031
Risk: $10.31 (1%)
SL Distance: 1.60%
Position Notional: $10.31 / 0.016 = $644.37
Actual Position: $645.96 (with slippage)

Trade Result:
- Entry: $94,252.85
- Exit: $87,316.35 (TP hit at 7.41% distance)
- PnL: +$46.67 (4.53% portfolio return)
- R:R Ratio: 4.6:1
```

### Neden Bu Trade Bu Kadar Iyi?

1. **Genis TP Mesafesi (7.41%):** Path uzun, TP'ye kadar cok alan var
2. **Yeterli SL Mesafesi (1.60%):** Noise'dan korunuyor (min 1.5% filter)
3. **Mukemmel R:R (4.6:1):** Risk $10, kazanc $46
4. **Leveraged Exposure:** 10x leverage ile $644 pozisyon $1031 balance'da guvenli

### Position Size Caps

```python
# core/simple_portfolio.py settings
max_position_pct = 0.10      # Max 10% of balance per position (unleveraged)
max_notional = balance * 0.10 * leverage  # $1000 * 0.10 * 10 = $1000 max
min_sl_distance = 0.002      # Min 0.2% SL (prevents huge positions)
```

---

## BTCUSDT Settings

### 15m Timeframe (Primary)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Filters** | regime, at_flat_filter, min_sl_filter | En iyi combo |
| **Min SL Distance** | 1.5% | <1.5% noise ile SL hit |
| **Risk Per Trade** | 1% | $10 risk for $1000 balance |
| **Leverage** | 10x | Conservative |
| **Max Position** | 10% * leverage = 100% | $1000 max notional |

**Backtest Results (1 Year):**
```
Trades: 26
Win Rate: 46.15%
Total PnL: +$72.99 (+7.3%)
Max DD: $35.70 (3.57%)
Profit Factor: 1.45
Avg Win: +$19.60
Avg Loss: -$11.59
```

**PnL Distribution:**
| Range | Count | Notes |
|-------|-------|-------|
| +$40 to +$50 | 1 | Best trade: +$46.67 (SHORT 2025-03-02) |
| +$30 to +$40 | 1 | +$32.74 (LONG 2025-11-04) |
| +$20 to +$30 | 2 | Good R:R trades |
| +$10 to +$20 | 8 | Standard wins |
| $0 to -$12 | 14 | Controlled losses |

**Key Insight:** Buyuk kazanclar genis TP mesafeli tradelerden geliyor. Min SL filter kayiplari kontrol altinda tutuyor (~-$11-12 max).

### 1h Timeframe

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Filters** | TBD | Test edilmedi |
| **Status** | UNTESTED | 15m ile ayni config test edilmeli |

### 4h Timeframe

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Filters** | TBD | Test edilmedi |
| **Status** | UNTESTED | Daha az trade beklenebilir |

---

## ETHUSDT Settings

### 15m Timeframe

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Filters** | TBD | Test edilmedi |
| **Min SL Distance** | TBD | BTCUSDT ile ayni (1.5%) baslangic |
| **Status** | UNTESTED | |

---

## LINKUSDT Settings

### 15m Timeframe

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Filters** | TBD | Test edilmedi |
| **Status** | UNTESTED | Daha volatil, SL ayari farkli olabilir |

---

## Filter Definitions

### 1. regime (Regime Filter)
Sadece trending marketlerde trade acar. ADX-based trend detection.

### 2. at_flat_filter (AlphaTrend Flat Filter)
AlphaTrend indicator flat degilken trade acar.

### 3. min_sl_filter (Minimum SL Distance Filter)
SL mesafesi minimum threshold'dan buyuk olmali.
```python
if sl_distance_pct < 1.5%:
    reject_signal()
```

**Neden Onemli:** Tight SL (<1%) market noise ile kolayca tetiklenir ve %77 loss rate yaratir.

---

## Trade Categories by PnL

### Category A: Big Winners (+$30 to +$50)
- Genis TP mesafesi (>5%)
- Yeterli SL mesafesi (>1.5%)
- R:R > 3:1
- Trend yonunde giris

### Category B: Standard Winners (+$10 to +$30)
- Normal TP mesafesi (3-5%)
- R:R 2:1 - 3:1
- Cogu trade bu kategoride

### Category C: Controlled Losses (-$10 to -$12)
- SL hit but controlled
- Her trade icin risk amount kadar kayip (~$10-11)
- Min SL filter sayesinde cok kucuk SL yok

---

## Risk Management

### Per-Symbol Limits
```python
# Her sembol icin max risk
single_trade_risk = 1%        # $10 for $1000
max_concurrent_risk = 3%      # Max 3 trade ayni anda
```

### Drawdown Limits
```python
daily_dd_limit = 5%           # Gunluk max kayip
weekly_dd_limit = 10%         # Haftalik max kayip
total_dd_limit = 25%          # Toplam max kayip (STOP TRADING)
```

---

## Command Reference (Simplified)

**TEK KOMUT:** `run.py` kullan!

```bash
# Full 1-year test (filter + portfolio backtest)
python run.py test BTCUSDT 15m

# Quick 90-day test
python run.py test BTCUSDT 15m --quick

# Test all recommended symbols
python run.py test --all

# Visualize trades from latest test
python run.py viz BTCUSDT 15m

# Show all test results
python run.py report

# List result directories
python run.py list
```

### Output Location

All results go to: `data/results/{SYMBOL}_{TF}_{TIMESTAMP}/`

```
data/results/BTCUSDT_15m_20260103_200051/
├── result.json     # Main result summary
├── signals.json    # All signals found
├── trades.json     # All trades executed
├── summary.txt     # Human-readable summary
└── charts/         # Trade visualizations (if run viz)
```

### Legacy Commands (Deprecated)

Eski komutlar hala calisiyor ama `run.py` tercih edilmeli:
```bash
# Deprecated - use run.py test instead
python runners/run_portfolio_test.py --symbol BTCUSDT --timeframe 15m
python runners/run_filter_combo_test.py --symbol BTCUSDT --timeframe 15m
python runners/run_full_pipeline.py --symbol BTCUSDT --timeframe 15m
```

---

## Changelog

| Date | Symbol | TF | Change |
|------|--------|-----|--------|
| 2026-01-03 | BTCUSDT | 15m | min_sl_filter eklendi (1.5%) |
| 2026-01-03 | BTCUSDT | 15m | Baseline filters: regime, at_flat_filter |
| 2026-01-03 | ALL | ALL | Portfolio logic documented |

---

## Notes

1. **+$46 Trade Analizi:** Bu trade'in bu kadar iyi olmasi SL/TP mesafeleri ile alakali. 1.6% SL ve 7.4% TP ile R:R 4.6:1. Portfolio sizing $10 risk ile $46 kazanc sagladi.

2. **Fixed $35 vs Portfolio:** Fixed size ile ayni tradeler $14.89 kazandirirken, portfolio sizing ile $72.99 kazandirdi. Portfolio sizing 5x daha etkili.

3. **Loss Control:** Tum kayiplar -$10 ile -$12 arasinda. Min SL filter ve 1% risk rule sayesinde.
