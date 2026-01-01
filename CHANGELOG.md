# Changelog

Tüm önemli değişiklikler bu dosyada belgelenir.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versiyonlama: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [v2.0.0] - 2026-01-01 - Indicator Parity Fix

### Özet
TradingView ile indicator uyumu sağlandı. PnL $122 iyileşti ama hala negatif.

### Değişiklikler (Changes)

#### Eklenen (Added)
- `core/version.py` - Versiyon takip modülü
- `VERSION.md` - Versiyon geçmişi dokümantasyonu
- `analysis/CORRECTED_FORENSIC_ANALYSIS.md` - Düzeltilmiş analiz raporu
- Test scriptlerine versiyon banner'ı eklendi

#### Değiştirilen (Changed)
- **ATR Hesaplama**: SMA → RMA (Wilder's smoothing)
  - Dosya: `core/indicators.py:95`
  - Sebep: TradingView ATR fonksiyonu RMA kullanıyor
  - Etki: AlphaTrend seviyeleri daha smooth

- **Wick Rejection Filter**: False → True (devre dışı)
  - Dosya: `core/config.py:594`
  - Sebep: P3 testinde +$30 iyileşme kanıtlandı
  - Etki: Daha fazla sinyal, daha az restrictive

- **Flat Threshold**: 0.001 → 0.002
  - Dosya: `core/config.py:514`
  - Sebep: %60 rejection rate çok yüksekti
  - Etki: AlphaTrend "flat" tespiti daha toleranslı

#### Denenen ve Geri Alınan (Tried & Reverted)
- **RSI yerine MFI**: RSI denendi, sinyal sayısı %70 düştü → MFI'ya geri dönüldü
  - Dosya: `core/indicators.py:97-102`
  - Sebep: RSI ile çok az sinyal üretildi (16 → 13 trade/yıl)

- **Trade Management Değişiklikleri**: Partial TP 40%→65%, BE tranche 1→2
  - Test sonucu: PnL $76→$8 düştü
  - Geri alındı, orijinal değerler korundu

### Test Sonuçları

| Metrik | v1.0.0 (Baseline) | v2.0.0 | Değişim |
|--------|-------------------|--------|---------|
| PnL | -$161.99 | -$39.90 | **+$122.09** ✅ |
| Trades | 51 | 13 | -38 ⚠️ |
| Win Rate | 41% | 31% | -10% ⚠️ |
| Max Drawdown | $208 | $98 | **-$110** ✅ |

### Avantajlar (Pros)
- ✅ PnL $122 iyileşti
- ✅ Drawdown yarıya indi ($208 → $98)
- ✅ TradingView ile ATR uyumu sağlandı

### Dezavantajlar (Cons)
- ⚠️ Trade sayısı çok düştü (51 → 13)
- ⚠️ Win rate düştü (41% → 31%)
- ⚠️ Hala negatif PnL (-$40)
- ⚠️ TRENDING rejimlerde kayıp (-$87)

### Bilinen Sorunlar (Known Issues)
- Optimizer çok az config buluyor (hard_min_trades=5 ile)
- TRENDING dönemlerde strateji başarısız
- Trade sayısı yıllık hedefin (50+) altında

---

## [v1.0.0] - 2025-12-31 - Original Baseline

### Özet
Orijinal baseline konfigürasyonu. Tüm karşılaştırmalar için referans noktası.

### Konfigürasyon
```python
ATR_METHOD = "SMA"
MOMENTUM_SOURCE = "MFI"  # if volume else RSI
skip_wick_rejection = False
flat_threshold = 0.001
lookback_days = 60
hard_min_trades = 5
```

### Test Sonuçları (Full Year BTC+ETH+LINK)
- PnL: -$161.99
- Trades: 51
- Win Rate: 41%
- Max Drawdown: $208

---

## Başarısız Deneyler Arşivi

Bu bölüm test edilen ve BAŞARISIZ olan değişiklikleri içerir. **Tekrar denemeyin!**

### Trade Management Değişiklikleri (v2.0.0-beta)
| Değişiklik | Sonuç | Neden Başarısız |
|------------|-------|-----------------|
| Partial TP 40%→65% | -$68 | Kazançlar erken kilitlenemiyor |
| BE after tranche 2 | -$68 | Pozisyon koruma gecikiyor |
| BE buffer 0.5→1.0 ATR | -$68 | Kombine etki |

### RSI Denemesi (v2.0.0-alpha)
| Değişiklik | Sonuç | Neden Başarısız |
|------------|-------|-----------------|
| MFI→RSI | -$31 vs MFI | RSI daha az sinyal üretiyor |

### Önceki Başarısız Deneyler (CLAUDE.md'den)
| Deney | Sonuç | Neden |
|-------|-------|-------|
| skip_wick_rejection=True | +$30 ✅ | BAŞARILI - v2.0.0'da uygulandı |
| regime_adx_threshold=25 | -$146 | Çok kısıtlayıcı |
| risk_per_trade=2.0% | -$78 | Optimizer farklı config seçiyor |
| use_trend_filter=True | -$58 | Karlı trade'leri de engelliyor |
| use_btc_regime_filter=True | -$59 | BTC-altcoin korelasyonu zayıf |
| hard_min_trades=3 | -$108 | Noisy config'ler kabul ediliyor |

---

## Versiyon Formatı

```
[vMAJOR.MINOR.PATCH] - YYYY-MM-DD - Kısa Başlık

MAJOR: Strateji mantığı veya indicator değişikliği
MINOR: Filter/threshold ayarları
PATCH: Bug fix, küçük düzeltme
```

## Değişiklik Kategorileri

- **Added**: Yeni özellik
- **Changed**: Mevcut işlevsellik değişikliği
- **Deprecated**: Yakında kaldırılacak özellik
- **Removed**: Kaldırılan özellik
- **Fixed**: Bug düzeltmesi
- **Security**: Güvenlik düzeltmesi
- **Tried & Reverted**: Denenen ve geri alınan değişiklik
