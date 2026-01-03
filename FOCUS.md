# FOCUS: Filter Combo Discovery System

**Son Güncelleme:** 2026-01-03
**Aktif Çalışma:** Filter-based signal optimization

---

## MEVCUT DURUM

### Yaklaşım: Bottom-Up Filter Discovery
```
MİNİMAL BAŞLA → BOL TRADE → ANALİZ ET → AKILLI FİLTRELE
```

Eski sistem (top-down) yerine yeni yaklaşım:
1. `check_core_signal` ile baseline al (1500+ sinyal)
2. Regime filter ekle
3. Incremental olarak filtre test et
4. Rolling WF ile validate et
5. Cost-aware test ile gerçekçiliği kontrol et

---

## SON TEST SONUÇLARI (BTCUSDT)

| TF | Best Config | Trades | İdeal PnL | Gerçekçi PnL | Durum |
|----|-------------|--------|-----------|--------------|-------|
| 5m | REGIME + at_binary | 1289 | +$18.90 | -$89.36 | Edge < Cost |
| 15m | REGIME + at_flat + adx | 238 | +$15.91 | -$4.07 | Edge < Cost |
| 1h | BASELINE (no filter) | 499 | +$38.40 | -$3.54 | Edge < Cost |
| 4h | - | - | - | - | Yetersiz veri |

### Problem
- Strateji edge: %0.04 - %0.21
- Trading maliyeti: %0.24 (slippage + commission)
- **Edge < Maliyet = Net Zarar**

### Hedef
- Edge'i %0.30+ yapmak VEYA
- Maliyeti %0.15 altına düşürmek

---

## KULLANILACAK ARAÇLAR

### 0. FULL PIPELINE (Önerilen)
```bash
# Tek symbol/timeframe
python runners/run_full_pipeline.py --symbol BTCUSDT --timeframe 15m

# BTC tüm timeframe'ler
python runners/run_full_pipeline.py --btc-only

# Birden fazla symbol
python runners/run_full_pipeline.py --symbols BTCUSDT,ETHUSDT --timeframes 15m,1h

# Tüm kombinasyonlar
python runners/run_full_pipeline.py --all
```
Tüm adımları otomatik çalıştırır ve kapsamlı rapor üretir.

### 1. AT Scenario Analysis
```bash
python runners/run_at_scenario_analysis.py --symbol BTCUSDT --timeframe 15m --days 365
```
Core sinyalleri analiz eder, AT modlarını karşılaştırır.

### 2. Filter Combo Test
```bash
# Incremental (tek tek filtre ekle)
python runners/run_filter_combo_test.py --symbol BTCUSDT --timeframe 15m --incremental

# Full scan (tüm kombinasyonlar)
python runners/run_filter_combo_test.py --symbol BTCUSDT --timeframe 15m --full-scan

# Specific combo
python runners/run_filter_combo_test.py --specific "regime,at_flat_filter,adx_filter"
```

### 3. Rolling WF Validation
```bash
python runners/run_rolling_wf_combo.py --symbol BTCUSDT --timeframe 15m \
    --filters "regime,at_flat_filter,adx_filter" --full-year
```

### 4. Cost-Aware Test (Gerçekçi)
```bash
# Karşılaştırma (ideal vs gerçekçi)
python runners/run_realistic_backtest.py --filters "regime,at_flat_filter" --compare

# Sadece cost-aware
python runners/run_realistic_backtest.py --filters "regime,at_flat_filter" --cost-aware
```

---

## MEVCUT FİLTRELER

| Filtre | Açıklama | Etki |
|--------|----------|------|
| `regime` | Neutral regime'i atla | **TEMEL** |
| `at_binary` | AT alignment gerekli | 5m'de iyi |
| `at_flat_filter` | AT flat ise atla | 15m'de iyi |
| `adx_filter` | ADX > 15 gerekli | Trend filtresi |
| `ssl_touch` | Son 5 bar'da SSL touch | Nötr |
| `rsi_filter` | RSI limitleri | Minimal etki |
| `pbema_distance` | PBEMA mesafesi | Minimal etki |
| `overlap_check` | SSL-PBEMA gap | Hafif pozitif |
| `body_position` | Mum gövde pozisyonu | Değişken |
| `wick_rejection` | Wick oranı | Değişken |

---

## ÇALIŞMA AKIŞI

```
1. Symbol/TF seç
2. AT Scenario Analysis çalıştır → baseline al
3. Filter Combo Test (incremental) → en iyi filtreleri bul
4. Rolling WF Validation → OOS test
5. Cost-Aware Test → gerçekçi sonuç
6. Eğer kârlı değilse → yeni filtre/parametre dene
```

---

## DOSYA YAPISI

```
runners/
├── run_at_scenario_analysis.py   # AT scenario tester
├── run_filter_combo_test.py      # Filter combo discovery
├── run_rolling_wf_combo.py       # WF validation with filters
├── run_realistic_backtest.py     # Cost-aware backtest
└── rolling_wf.py                 # Core WF logic

data/filter_combo_logs/           # Test sonuçları
├── combo_tests_BTCUSDT_15m.jsonl # Append-only log
└── BTCUSDT_15m_365d_*.json       # Full scan results

docs/
├── COST_ANALYSIS_REPORT.md       # Maliyet analizi
└── PROJECT_INDEX.md              # Proje indeksi
```

---

## SONRAKİ ADIMLAR (ÖNERİLER)

1. **TP/SL Optimizasyonu** - Risk/reward oranını iyileştir
2. **Entry Timing** - Limit order ile slippage azalt
3. **Yeni Filtreler** - Volume, volatility bazlı filtreler
4. **Multi-TF Confirmation** - 1h sinyal + 15m entry
5. **Farklı Semboller** - ETH, LINK için aynı analizi yap

---

## UYARILAR

- ❌ Eski `run_rolling_wf_test.py` KULLANMA (eski sistem)
- ❌ `check_signal` yerine `check_core_signal` kullan
- ❌ İdeal sonuçlara güvenme, her zaman cost-aware test yap
- ✅ Her değişikliği incremental test et
- ✅ Sonuçları `data/filter_combo_logs/` klasörüne logla

---

**Bu dosyayı her session başında oku.**
