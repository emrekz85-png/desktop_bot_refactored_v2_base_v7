# Version History

Bu dosya tüm bot güncellemelerini ve test sonuçlarını takip eder.

---

## Current Version: v2.0.0 - Indicator Parity Fix

**Tarih:** 2026-01-01
**Kod Adı:** `indicator-parity-fix`

### Değişiklikler

| Dosya | Değişiklik | Açıklama |
|-------|------------|----------|
| `core/indicators.py` | ATR: SMA → RMA | TradingView uyumu için Wilder's smoothing |
| `core/indicators.py` | Momentum: MFI (korundu) | Volume varsa MFI, yoksa RSI |
| `core/config.py` | `skip_wick_rejection: True` | Gereksiz filtre kaldırıldı (+$30 test) |
| `core/config.py` | `flat_threshold: 0.002` | 0.001'den 0.002'ye (daha az kısıtlayıcı) |

### Test Sonuçları (Full Year BTC+ETH+LINK)

| Metrik | Baseline | v2.0.0 | Değişim |
|--------|----------|--------|---------|
| PnL | -$161.99 | -$39.90 | **+$122.09** |
| Trades | 51 | 13 | -38 |
| Win Rate | 41% | 31% | -10% |
| Max DD | $208 | $98 | **-$110** |

### Notlar
- PnL hala negatif ama baseline'dan önemli iyileşme
- Trade sayısı düşük - optimizer çok az config buluyor
- TRENDING rejimlerde kayıp (-$87), RANGING/TRANSITIONAL'da kazanç (+$47)

---

## v1.0.0 - Original Baseline

**Tarih:** 2025-12-31 (tahmini)
**Kod Adı:** `original-baseline`

### Konfigürasyon
- ATR: SMA
- Momentum: MFI (volume varsa)
- skip_wick_rejection: False
- flat_threshold: 0.001
- lookback_days: 60

### Test Sonuçları
- PnL: -$161.99
- Trades: 51
- Win Rate: 41%
- Max DD: $208

---

## Version Format

```
vMAJOR.MINOR.PATCH - Description

MAJOR: Büyük değişiklikler (strateji mantığı, indicator hesaplaması)
MINOR: Orta değişiklikler (filter ayarları, threshold'lar)
PATCH: Küçük düzeltmeler (bug fix, typo)
```

## Test Çalıştırırken

Her test çalıştırıldığında, çıktının başında şu bilgiler gösterilir:
- Version numarası
- Aktif değişiklikler
- Test parametreleri

Örnek:
```
======================================================================
📊 TEST - Version: v2.0.0 (indicator-parity-fix)
   Değişiklikler: ATR=RMA, skip_wick=True, flat=0.002
   Semboller: BTC+ETH+LINK | TF: 15m, 1h | Lookback: 60d
======================================================================
```
