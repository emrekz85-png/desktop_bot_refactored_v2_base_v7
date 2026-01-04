# Changelog

Tüm önemli değişiklikler bu dosyada belgelenir.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versiyonlama: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [v2.2.0] - 2026-01-04 - Pattern Integration & Momentum Exit

### Özet
7 trading pattern'i (gerçek trade analizinden) sisteme entegre edildi. PBEMA Retest stratejisi düzeltildi ve çalışır hale getirildi. Momentum Exit özelliği trade loop'a eklendi.

### Değişiklikler (Changes)

#### Eklenen (Added)

**PBEMA Retest Stratejisi Düzeltmesi:**
- `strategies/pbema_retest.py` - Breakout detection logic düzeltildi
  - Eski: `prev_close < pb_bot AND close > pb_top` (tek mumda tüm bulutu geçme - imkansız)
  - Yeni: `prev_close <= pb_top AND close > pb_top` (bulut kenarını geçme - gerçekçi)
  - `min_rr`: 1.5 → 1.0 (daha fazla sinyal)
  - `breakout_lookback`: 20 → 30 (daha geniş arama)
  - `min_breakout_strength`: 0.5% → 0.2% (daha gerçekçi)
  - Sonuç: 0 sinyal → **450 sinyal**, %52.7 WR, +$12.60 PnL

**Momentum Exit Entegrasyonu:**
- `runners/run_filter_combo_test.py::simulate_trade()` güncellendi
  - `use_momentum_exit` parametresi eklendi
  - Trade loop içinde momentum exhaustion kontrolü
  - Sadece kârdayken momentum exit kontrol edilir
  - Exit types: TP, SL, MOMENTUM, EOD

- `run_comprehensive_test.py` güncellendi
  - Momentum Exit test fonksiyonları eklendi
  - Exit type istatistikleri gösterimi
  - SSL Flow ve PBEMA Retest için momentum exit karşılaştırması

**Pattern Filter Düzeltmeleri:**
- `runners/run_filter_combo_test.py::apply_filters()` düzeltildi
  - Pattern 3 (Liquidity Grab): Artık grab detection ZORUNLU
  - Pattern 7 (SSL Dynamic): SHORT için de destek eklendi

- `core/pattern_filters.py` threshold ayarları:
  - Pattern 4 (SSL Slope): `flat_threshold` 0.0015 → 0.0006
  - Pattern 5 (HTF Bounce): `drop_threshold` 3% → 1.5%
  - Pattern 6 (Momentum Loss): `min_consecutive` 5 → 3

#### Değiştirilen (Changed)
- `run_comprehensive_test.py` - 6 adımlı test pipeline'ı
- PBEMA Retest için regime filter kaldırıldı (kendi trend detection'ı var)

### Test Sonuçları (1 Yıl, BTCUSDT 15m)

| Sistem | Trade | WR | PnL | Not |
|--------|-------|-----|-----|-----|
| SSL Flow (Current Default) | 34 | 50.0% | **$24.39** | 🥇 En iyi PnL |
| SSL Flow + SSL Slope | 25 | 52.0% | $22.13 | |
| SSL Flow + SSL Dynamic | 8 | 75.0% | $19.75 | En yüksek WR |
| PBEMA Retest | 450 | 52.7% | $12.60 | Çok trade |
| SSL + Momentum Exit | 40 | 70.0% | $4.86 | Yüksek WR, düşük PnL |
| PBEMA + Momentum Exit | 450 | 60.7% | $0.70 | |

### Ana Bulgular

1. **SSL Flow (Current Default) hala en iyi** - $24.39 PnL ile birinci
2. **PBEMA Retest artık çalışıyor** - 450 trade, %52.7 WR, +$12.60
3. **Pattern filtreleri iyileştirme sağlamıyor** - P3-P7 PnL'i düşürüyor
4. **Momentum Exit trade-off'u:**
   - Win rate artıyor (%50 → %70)
   - PnL düşüyor ($24 → $5)
   - Erken çıkış = daha az kâr

### Kullanım

```bash
# Comprehensive test (tüm stratejiler)
python run_comprehensive_test.py BTCUSDT 15m --days 365

# PBEMA Retest kullanımı
from strategies import check_pbema_retest_signal
signal_type, entry, tp, sl, reason = check_pbema_retest_signal(df, index=-2)

# Momentum Exit ile trade simulation
from runners.run_filter_combo_test import simulate_trade
trade = simulate_trade(df, idx, signal_type, entry, tp, sl, use_momentum_exit=True)
```

---

## [v2.1.0] - 2026-01-02 - Kelly Criterion Risk Management

### Özet
Matematiksel olarak optimize edilmiş risk yönetim sistemi eklendi.

### Değişiklikler (Changes)

#### Eklenen (Added)
- `core/kelly_calculator.py` - Kelly Criterion hesaplamaları
  - `calculate_kelly()` - Optimal pozisyon boyutu
  - `calculate_growth_rate()` - Geometrik büyüme oranı
  - `trades_to_double()` - Sermayeyi ikiye katlamak için gereken trade sayısı

- `core/drawdown_tracker.py` - Drawdown takibi ve oto-ayarlama
  - `DrawdownTracker` sınıfı - Equity ve peak takibi
  - `get_drawdown_kelly_multiplier()` - Üstel azalma ile Kelly çarpanı
  - Circuit breaker: %20 max drawdown

- `core/risk_manager.py` - Merkezi risk yönetimi koordinatörü
  - `RiskManager` sınıfı - Tüm bileşenleri entegre eder
  - `calculate_position_size()` - Master pozisyon boyutlandırma metodu
  - R-Multiple takibi ve beklenti hesaplaması

- `tests/test_risk_manager.py` - 49 kapsamlı unit test
- `docs/RISK_MANAGEMENT_SPEC.md` - Tam spesifikasyon dokümanı (~1030 satır)

#### Değiştirilen (Changed)
- `core/correlation_manager.py` - Kelly entegrasyon fonksiyonları eklendi
  - `adjust_kelly_for_correlation()` - Korelasyon bazlı Kelly ayarlama
  - `calculate_portfolio_risk()` - Portföy risk hesaplama
- `core/__init__.py` - Yeni modül exportları

### Temel Özellikler

| Özellik | Açıklama |
|---------|----------|
| Kelly Criterion | f* = W - (1-W)/R, Half-Kelly varsayılan |
| Drawdown Auto-Adjust | 0%→1.0, 10%→0.70, 20%→0.0 üstel azalma |
| Circuit Breaker | %20 max drawdown tüm işlemleri durdurur |
| Recovery Mode | %5 recovery gerekli, %25 boyutta devam |
| Korelasyon Ayarlama | Korelasyonlu pozisyonlar için boyut azaltma |

### Test Sonuçları
- 49/49 unit test başarılı

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
