# Filter Discovery Methodology

## Problem Statement

**Gerçek hayatta gördüğüm edge'i otomasyonda kaybediyorum.**

| Durum | Trade Sayısı | Sonuç |
|-------|--------------|-------|
| Manuel trading | Bol trade | Pozitif edge |
| Sıkı filtreli bot | 5 trade/yıl | İstatistiksel anlamsız |

Eski yaklaşım (top-down):
- Tüm filtreleri aktif et → çok az trade → analiz imkansız → körlemesine ayar

---

## Yeni Yaklaşım: Bottom-Up Filter Discovery

### Felsefe

```
MİNİMAL BAŞLA → BOL TRADE → ANALİZ ET → AKILLI FİLTRELE
```

### Metodoloji

```
1. Stratejinin ÖZÜNÜ tanımla (SSL + PBEMA)
   ↓
2. Minimal filtrelerle BASELINE oluştur (1500+ trade)
   ↓
3. Her filtreyi TEK TEK ekle ve etkisini ölç
   ↓
4. Kombinasyonları test et
   ↓
5. Raporları incele, ÖĞREN
   ↓
6. Optimal konfigürasyonu belirle
```

---

## Araçlar

### 1. AT Scenario Analyzer
**Dosya:** `core/at_scenario_analyzer.py`
**Runner:** `runners/run_at_scenario_analysis.py`

**Amaç:** Minimal sinyalleri bul, farklı AT senaryolarını karşılaştır.

```bash
python runners/run_at_scenario_analysis.py --symbol BTCUSDT --timeframe 15m --days 365
```

**Core Signal Tanımı (`check_core_signal`):**
- Price vs SSL baseline (yön)
- PBEMA target exists (hedef)
- min_rr = 1.2 (düşük eşik)
- min_pbema_distance = 0.002 (gevşek)
- NO AlphaTrend check

**Çıktı:** 1500-1700 sinyal (yeterli istatistik için)

---

### 2. Filter Combination Tester
**Dosya:** `runners/run_filter_combo_test.py`

**Amaç:** Baseline üzerine filtreleri tek tek veya kombinasyon olarak test et.

```bash
# Tek tek filtre testi
python runners/run_filter_combo_test.py --incremental

# Tüm kombinasyonlar
python runners/run_filter_combo_test.py --full-scan
```

**Mevcut Filtreler:**
| Filtre | Açıklama |
|--------|----------|
| `regime_filter` | Neutral regime'de trade alma |
| `at_binary` | AT alignment zorunlu (buyers/sellers) |
| `at_flat_filter` | AT flat olduğunda trade alma |
| `adx_filter` | ADX > 15 zorunlu |
| `ssl_touch` | Son 5 bar'da SSL touch zorunlu |

---

## Mevcut Bulgular (2026-01-03)

### Baseline
| Metrik | Değer |
|--------|-------|
| Symbol | BTCUSDT 15m |
| Period | 365 gün |
| Core Signals | 1684 |
| PnL (filtresiz) | -$12.83 |

### Filtre Etkileri

| Filtre | Trades | WR | PnL | MaxDD | Değerlendirme |
|--------|--------|-----|------|-------|---------------|
| Baseline (yok) | 1684 | 31.2% | -$12.83 | $29.69 | ❌ Referans |
| + regime_filter | 1510 | 32.3% | +$1.80 | $25.87 | ⚠️ İyileşme |
| + at_flat_filter | 242 | 31.0% | **+$14.62** | **$6.21** | ✅ EN İYİ |
| + adx_filter | 1337 | 32.7% | +$13.17 | $24.65 | ⚠️ İyi ama DD yüksek |
| + at_binary | 599 | 33.4% | -$0.35 | $24.96 | ❌ Negatif |
| + ssl_touch | 1365 | 31.3% | -$18.93 | $29.87 | ❌ Zararlı |

### Sonuç

**Optimal Konfigürasyon:**
```python
config = {
    "regime_filter": "skip_neutral",  # Neutral regime skip
    "skip_at_flat_filter": False,     # AT flat filter AKTIF
}
```

**Beklenti:** ~242 trade/yıl, +$14.62 PnL, $6.21 max DD

---

## Nasıl Kullanılır

### Yeni Filtre Denemek

1. `run_filter_combo_test.py`'a yeni filtre ekle:
```python
# apply_filters() fonksiyonuna ekle
if use_new_filter:
    # filtre mantığı
    if condition:
        return False, "New Filter: Reason"
```

2. Test et:
```bash
python runners/run_filter_combo_test.py --incremental
```

3. Sonuçları karşılaştır.

### Farklı Sembol/TF Test Etmek

```bash
python runners/run_filter_combo_test.py --symbol ETHUSDT --timeframe 1h --days 365
```

### Rapor Oluşturmak

```bash
# Full scan ile tüm kombinasyonları test et
python runners/run_filter_combo_test.py --full-scan > filter_report_$(date +%Y%m%d).txt
```

---

## Önemli Prensipler

1. **Minimal baseline = öğrenme fırsatı**
   - Az trade ile analiz yapılamaz
   - 1000+ trade ile pattern'lar görünür

2. **Her filtre ayrı test edilmeli**
   - Birden fazla filtre aynı anda ekleme
   - Hangi filtre ne kadar etki ediyor, ölç

3. **Negatif etki = filtre zararlı**
   - PnL düşüyorsa filtre kaldır
   - "Mantıklı görünen" her filtre işe yaramaz

4. **Trade sayısı vs kalite dengesi**
   - Çok az trade = istatistiksel güvenilmez
   - Çok fazla trade = düşük kalite
   - Sweet spot: Pozitif PnL + kabul edilebilir trade sayısı

5. **Gerçek edge'i koru**
   - Manuel trading'de çalışan şeyi bozma
   - Filtrelerin amacı: kötü trade'leri ele, iyi trade'leri koru

---

## Sonraki Adımlar

- [ ] Multi-symbol test (ETH, LINK)
- [ ] Multi-timeframe test (5m, 1h, 4h)
- [ ] Yeni filtre fikirleri:
  - Volume filter
  - Volatility filter
  - Time-of-day filter
- [ ] Out-of-sample validation
- [ ] Live paper trading test

---

## Versiyon Geçmişi

| Tarih | Değişiklik |
|-------|------------|
| 2026-01-03 | İlk versiyon, BTCUSDT 15m test tamamlandı |
