# Experiment Log - Trading Bot Optimizasyonu

Bu dosya, denenen degisiklikleri, sonuclari ve ogrenimleri kaydeder.
Amac: Ayni hatalarin tekrar edilmesini onlemek.

**Baseline Metrikleri (BTC-Only Referans):**
- Versiyon: v1.7.0 (current)
- H1 2025: -$62.79
- H2 2025: +$81.15
- Yillik: +$18.36
- Bakiye: $2,000 (sabit)

**Onceki Baseline (v1.6.2):**
- H1 2025: -$100.44
- H2 2025: +$81.15
- Yillik: -$19.29

---

## BASARISIZ DENEMELER - TEKRAR DENEME!

Bu denemeler test edildi ve BASARISIZ oldu. Benzer fikirleri tekrar denemeyin.

---

### 1. ADX Max Filtresi (4 kez denendi, 4 kez basarisiz)

**Deneme Tarihi:** 2025 Q4

**Degisiklik:**
- `adx_max=40` filtresi eklendi
- ADX > 40 olan trade'leri engellemek icin

**Sonuc:** -$316 kayip (baseline'dan cok kotu)

**Neden Basarisiz:**
- Optimizer'i etkiliyor - guclu trendleri eliyor
- Trend-following stratejide yuksek ADX IYIDIR
- ADX yuksekken trend gucleniyor, tam o zaman trade almak lazim

**Ders:**
> ASLA trend-following stratejide ADX max siniri koyma.
> ADX ne kadar yuksek = trend o kadar guclu = daha iyi sinyal.

---

### 2. Tighter Swing SL (20 -> 10 bar)

**Deneme Tarihi:** v1.1

**Degisiklik:**
- `swing_lookback` 20'den 10'a dusuruldu
- Daha siki stop-loss icin

**Sonuc:** REVERTED (negatif etki)

**Neden Basarisiz:**
- Cok erken stop-out'lar
- Normal volatilite hareketlerinde zarar kesme
- SL'nin yeterince "nefes almasi" gerekiyor

**Ders:**
> Swing lookback'i 20'nin altina dusurme.
> Piyasanin normal volatilitesine yer ver.

---

### 3. Relaxed Good Config Criteria

**Deneme Tarihi:** v1.6.3

**Degisiklik:**
- Minimum E[R] esiklerini dusurduk
- Daha fazla config'in aktif olmasini sagladik

**Sonuc:** -$106 kayip

**Neden Basarisiz:**
- Zayif edge'li config'ler aktif oldu
- Dusuk kaliteli trade'ler kar'i yedi

**Ders:**
> Minimum E[R] esiklerini gevsetme.
> Kalite > Miktar

---

### 4. Carry-Forward Relaxation

**Deneme Tarihi:** 2025 Q4

**Degisiklik:**
- Onceki window'un config'ini sonrakine tasimak
- Re-optimization yerine carry-forward

**Sonuc:** BASARISIZ (detaylar eksik)

**Neden Basarisiz:**
- Market koşullari degisiyor
- Gecmis config'ler gelecekte calismayabilir

**Ders:**
> Her window icin fresh optimization yap.
> Carry-forward guvenilir degil.

---

### 5. Baseline Fallback Mekanizmasi

**Deneme Tarihi:** 2025 Q4

**Degisiklik:**
- Optimizer bulamazsa baseline config kullan

**Sonuc:** -$708 kayip

**Neden Basarisiz:**
- Baseline her zaman calismiyor
- Bazen "no trade" en iyi karar
- Zorla trade almak zarar

**Ders:**
> "No good config = no trade" prensibi dogru.
> Fallback ile zorla trade alma.

---

### 6. AlphaTrend Momentum Confirmation (v1.7.1-atmomentum)

**Deneme Tarihi:** 2025-12-26

**Degisiklik:**
- AlphaTrend cizgisinin aktif olarak hareket etmesini kontrol et
- 5 bar lookback ile AT momentum hesapla
- LONG: AT momentum >= +0.1%
- SHORT: AT momentum <= -0.1%

**Sonuc:** 0 trade (tum sinyaller filtrelendi!)

| Metrik | v1.7.0 | v1.7.1 |
|--------|--------|--------|
| H1 2025 | -$62.79 | $0 |
| H2 2025 | +$81.15 | $0 |

**Neden Basarisiz:**
- %0.1 esik deger cok yuksek
- AlphaTrend cizgisi kisa surede bu kadar hareket etmiyor
- Optimizer 0 config buluyor (tum sinyaller reddediliyor)

**Ders:**
> AlphaTrend zaten at_is_flat kontrolu yapiyor.
> Ekstra momentum filtresi gereksiz ve zarar verici.
> Signal-level filtreler optimizer'i da etkiler!

---

### 7. Dynamic Partial TP by Regime (v1.7.1-dyntp)

**Deneme Tarihi:** 2025-12-26

**Degisiklik:**
- Guclu trend (ADX > 30) durumunda partial TP'yi geciktir
- Tranche trigger: 0.40 → 0.55, 0.70 → 0.85
- Tranche fraction: %40 azaltildi (0.33 → 0.20)
- Amac: Guclu trendlerde kazancin daha fazla calismasini sagla

**Sonuc:** Hafif negatif etki (-$0.50)

| Metrik | v1.7.0 | v1.7.1 |
|--------|--------|--------|
| H1 2025 | -$62.79 | -$62.79 |
| H2 2025 | +$81.15 | +$80.65 |
| Yillik | +$18.36 | +$17.86 |

**Neden Basarisiz:**
- Guclu trendlerde partial geciktirmek fayda saglamadi
- Bazi trade'lerde partial alinmadan fiyat geri dondu
- Mevcut partial sistemi zaten iyi calisiyor

**Ders:**
> Partial TP sistemi (3-tranche) iyi optimize edilmis.
> Regime-bazli dinamik ayarlama gereksiz karmasiklik.
> "If it ain't broke, don't fix it"

---

### 8. TF-Adaptive SSL Lookback (v1.7.1)

**Deneme Tarihi:** 2025-12-26

**Degisiklik:**
- SSL baseline HMA periyodunu timeframe'e gore ayarla
- 5m: HMA(75) - daha fazla smoothing
- 15m: HMA(60) - standart (degismedi)
- 1h: HMA(45) - daha hizli tepki

**Sonuc:** NOTR (hic fark yok)

| Metrik | v1.7.0 | v1.7.1 |
|--------|--------|--------|
| H1 2025 | -$62.79 | -$62.79 |
| H2 2025 | +$81.15 | +$81.15 |
| Yillik | +$18.36 | +$18.36 |

**Neden Etkisiz:**
- Optimizer ayni config'leri buluyor
- HMA periyodu degisikliginin sonuca etkisi yok
- Sinyal kalitesi ayni kaliyor

**Ders:**
> SSL baseline HMA periyodu kritik bir parametre degil.
> Teorik olarak mantikli ama pratik fayda yok.
> Tutulabilir (zarar yok) veya kaldirilabilir (gereksiz karmasiklik).

**Karar:** TUTULDU (zarar vermediginden)

---

## BASARILI DEGISIKLIKLER

Bu degisiklikler baseline'da tutulmaktadir.

---

### 0. Window-Level Regime Gating (v1.7.0) - EN SON

**Deneme Tarihi:** 2025-12-26

**Degisiklik:**
- `ssl_flow.py` icine ADX ortalama kontrolu eklendi
- 50 bar lookback ile ADX ortalamasini hesapla
- ADX_avg < 20 ise trade ATLA (RANGING regime)
- ADX_avg >= 20 ise normal trade (TRENDING regime)

**Kod Degisikligi:**
```python
# ================= REGIME GATING (v1.7.0) =================
REGIME_ADX_THRESHOLD = 20.0  # ADX average below this = RANGING
REGIME_LOOKBACK = 50  # ~7 days for 1h TF

regime_start = max(0, abs_index - REGIME_LOOKBACK)
adx_window = df["adx"].iloc[regime_start:abs_index + 1]
adx_avg = float(adx_window.mean())

if adx_avg < REGIME_ADX_THRESHOLD:
    return "RANGING Regime - skip trade"
```

**Sonuc:**
| Metrik | v1.6.2 | v1.7.0 | Fark |
|--------|--------|--------|------|
| H1 2025 | -$100.44 | -$62.79 | **+$37.65** |
| H2 2025 | +$81.15 | +$81.15 | $0 |
| Yillik | -$19.29 | +$18.36 | **+$37.65** |

**Neden Calisti:**
- RANGING piyasalarda SSL Flow stratejisi kotu calisiyor
- ADX_avg < 20 = trendless, sideways market
- Bu piyasalarda trade almamak en iyi karar
- H2'yi etkilemedi cunku H2 zaten trending market

**Ders:**
> Trend-following strateji icin SADECE trend olan piyasalarda trade al.
> ADX ortalamasi < 20 ise sabret, trade alma.

---

### 1. Momentum TP Extension (v1.5)

**Degisiklik:**
- EMA15 kullanarak momentum kontrolu
- Guclu momentum varsa TP'yi erken kapatma

**Sonuc:** Baseline'da kalici

**Neden Calisti:**
- Momentum devam ederken kar kacirilmiyor
- Trend hareketlerinden daha fazla kar

---

### 2. AlphaTrend Filter (Core Feature)

**Degisiklik:**
- `at_active: True` zorunlu
- AlphaTrend buyers/sellers tespiti

**Sonuc:** Zorunlu ozellik

**Neden Calisti:**
- Yanlis yon trade'lerini filtreler
- Trend yonunu dogrular

---

### 3. 30m Timeframe Kaldirildi (v1.2+)

**Degisiklik:**
- 30m timeframe test listesinden cikarildi
- Sadece 5m, 15m, 1h kullanildi

**Sonuc:** Test suresi azaldi, sonuclar benzer

**Neden Calisti:**
- 30m genelde 15m ve 1h arasinda sikiyor
- Ekstra deger katmiyor

---

## MEVCUT BASELINE

```python
BASELINE_CONFIG = {
    "rr": 2.0,
    "rsi": 70,
    "at_active": True,
    "use_trailing": False,
    "strategy_mode": "ssl_flow",
}
```

**Metrikler (BTC-Only, v1.7.0):**
| Metrik | H1 2025 | H2 2025 | Yillik |
|--------|---------|---------|--------|
| PnL    | -$62.79 | +$81.15 | +$18.36 |
| Win%   | 0%      | 71.4%   | ~55%    |
| Trades | 2       | 7       | 9       |

---

## TEST PROSEDURU

Yeni bir degisiklik test ederken:

1. **Hizli Test (BTC-only):**
   ```bash
   python run_rolling_wf_test.py --quick-btc
   ```
   - Sadece BTCUSDT
   - 3-5 dakika
   - Ilk sonuclari gor

2. **Tam Test (3 Symbol):**
   ```bash
   python run_rolling_wf_test.py
   ```
   - BTC, ETH, SOL
   - 15-20 dakika
   - Gercek performans

3. **Sonuc Karsilastirmasi:**
   - Baseline: +$204
   - Yeni sonuc >= baseline ise KABUL
   - Yeni sonuc < baseline ise REVERT

4. **Dokumantasyon:**
   - Bu dosyaya ekle (basarili/basarisiz)
   - Ogrenimleri yaz

---

## ACIK SORULAR / ARASTIRILACAKLAR

- [ ] H1 2025 neden negatif? Piyasa kosullari mi?
- [ ] Daha iyi TP mekanizmasi? Partial TP?
- [ ] Timeframe bazli ayri config'ler?

---

## VERSIYON GECMISI

| Versiyon | Tarih | Degisiklik | Sonuc |
|----------|-------|------------|-------|
| v1.0 | - | Ilk versiyon | Baseline |
| v1.1 | - | Tighter swing SL | REVERTED |
| v1.2 | - | 30m kaldirildi | KABUL |
| v1.5 | - | Momentum TP | KABUL |
| v1.6.2 | - | Onceki baseline | AKTIF |
| v1.6.3 | - | Relaxed criteria | REVERTED |
| v1.7-adxmax | - | ADX max=40 | REVERTED (-$316) |
| **v1.7.0** | **2025-12-26** | **Regime Gating (ADX_avg<20=skip)** | **AKTIF (+$37.65)** |
| v1.7.1-atmomentum | 2025-12-26 | AT Momentum filter | REVERTED (0 trades) |
| v1.7.1-dyntp | 2025-12-26 | Dynamic Partial TP | REVERTED (-$0.50) |
| **v1.7.1** | **2025-12-26** | **TF-Adaptive SSL** | **NOTR (tutuldu)** |
| **v1.7.2** | **2025-12-26** | **Grid Search Optimizer + Configurable Regime** | **AKTIF (+$47.24)** |

---

### 9. Grid Search Optimizer (v1.7.2)

**Deneme Tarihi:** 2025-12-26

**Degisiklik:**
- Yeni `core/grid_optimizer.py` modulu eklendi
- `run_grid_optimizer.py` CLI araci eklendi
- `regime_adx_threshold` parametresi konfigure edilebilir hale getirildi
- `regime_lookback` parametresi konfigure edilebilir hale getirildi
- Win rate hesaplama hatasi duzeltildi (partial trades)

**Sonuc:** +$47.24 (v1.7.1: $15.79'dan %199 artis)

| Metrik | v1.7.1 | v1.7.2 |
|--------|--------|--------|
| Yillik PnL | $15.79 | $47.24 |
| Trades | 9 | 10 |
| Win Rate | 55.6% | 60.0% |
| Max DD | 3.1% | 3.1% |

**Yeni Ozellikler:**
```bash
# Grid Search Optimizer kullanimi
python run_grid_optimizer.py --symbol BTCUSDT --quick   # Hizli test
python run_grid_optimizer.py --symbol BTCUSDT --full    # Tam test
python run_grid_optimizer.py --symbol BTCUSDT --robust  # Robustness dahil
```

**Ders:**
> Configurable regime gating parametreleri optimizer'a daha fazla esneklik sagladi.
> Grid Search + Walk-Forward entegrasyonu gelecekte denenebilir.

---

*Son guncelleme: 2025-12-26*
