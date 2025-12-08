import sys
import os
import time
import json
import io
import contextlib
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import dateutil.parser
import itertools
from datetime import datetime, timedelta
import traceback
import tempfile
import shutil
from typing import Tuple, Optional
import matplotlib

# Matplotlib çizimlerini arka planda üretmek için GUI gerektirmeyen backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PyQt5 Modülleri (QSpinBox EKLENDİ)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QTabWidget, QTextEdit, QLabel,
                             QTableWidget, QTableWidgetItem, QPushButton, QHeaderView,
                             QGroupBox, QDoubleSpinBox, QComboBox, QMessageBox, QCheckBox,
                             QLineEdit, QSpinBox, QFrame)  # <--- EKLENDİ
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QFont
import plotly.graph_objects as go
import plotly.utils

# ==========================================
# ⚙️ GENEL AYARLAR VE SABİTLER (MERKEZİ YÖNETİM)
# ==========================================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["1m", "5m", "15m", "1h"]
candles = 50000
REFRESH_RATE = 3
CSV_FILE = "trades.csv"
CONFIG_FILE = "config.json"
# Backtestler için maks. mum sayısı sınırları
BACKTEST_CANDLE_LIMITS = {"1m": 4000, "5m": 4000, "15m": 4000, "1h": 4000}
# Günlük raporlar için özel mum sayısı sınırı
DAILY_REPORT_CANDLE_LIMITS = {"1m": 15000, "5m": 15000, "15m": 15000, "1h": 15000}
BEST_CONFIGS_FILE = "best_configs.json"
BEST_CONFIG_CACHE = {}
BACKTEST_META_FILE = "backtest_meta.json"

# --- 💰 EKONOMİK MODEL (Tüm Modüller Burayı Kullanacak) ---
#  uyarınca tek bir konfigürasyon yapısı:
TRADING_CONFIG = {
    "initial_balance": 2000.0,
    "leverage": 10,
    "usable_balance_pct": 0.20,  # Bakiyenin %20'si
    "slippage_rate": 0.0005,     # %0.05 Kayma Payı
    "funding_rate_8h": 0.0001,   # %0.01 Fonlama (8 saatlik)
    "maker_fee": 0.0002,         # %0.02 Limit Emir Komisyonu
    "taker_fee": 0.0005,         # %0.05 Piyasa Emir Komisyonu
    "total_fee": 0.0007          # %0.07 (Giriş + Çıkış Tahmini) - Güvenlik marjı
}

# ==========================================
# 🚀 v30.5 - FİNAL PROFIT MAX CONFIG (VERİ ODAKLI)
# ==========================================
SYMBOL_PARAMS = {
    "BTCUSDT": {
        # 1m: Daha esnek scalp (daha fazla sinyal için RR ve slope düşürüldü)
        "1m": {"rr": 1.3, "rsi": 70, "slope": 0.4, "at_active": False, "use_trailing": False},

        # 5m: AlphaTrend opsiyonel, slope yumuşatıldı
        "5m": {"rr": 2.0, "rsi": 70, "slope": 0.4, "at_active": False, "use_trailing": False},

        # 4h: AlphaTrend KAPALI (Saf Trend) (16.53 R)
        "4h": {"rr": 2.0, "rsi": 30, "slope": 0.3, "at_active": False, "use_trailing": False},

        # Ara dönemler (1h fena değil, eklendi)
        "15m": {"rr": 1.8, "rsi": 55, "slope": 0.6, "at_active": False, "use_trailing": False},
        "1h": {"rr": 1.8, "rsi": 45, "slope": 0.8, "at_active": True, "use_trailing": False}
    },
    "ETHUSDT": {
        # 1m: AlphaTrend KAPALI daha iyi (19.25 R)
        "1m": {"rr": 2.0, "rsi": 30, "slope": 0.9, "at_active": False, "use_trailing": False},

        # 5m: AlphaTrend AÇIK (23.10 R)
        "5m": {"rr": 2.5, "rsi": 60, "slope": 0.9, "at_active": True, "use_trailing": True},

        # 15m: 18.24 R (Güzel sürpriz, aktif edilebilir)
        "15m": {"rr": 3.0, "rsi": 40, "slope": 0.9, "at_active": True, "use_trailing": False},

        "1h": {"rr": 3.0, "rsi": 30, "slope": 0.3, "at_active": True, "use_trailing": False},
        "4h": {"rr": 3.0, "rsi": 30, "slope": 0.3, "at_active": True, "use_trailing": False}
    },
    "SOLUSDT": {
        # 1m: Zayıf (7.14 R). Pasif kalabilir.
        "1m": {"rr": 3.0, "rsi": 30, "slope": 0.5, "at_active": True, "use_trailing": False},

        # 5m: EFSANE (69.53 R). Kesinlikle bu ayar.
        "5m": {"rr": 3.0, "rsi": 40, "slope": 0.9, "at_active": True, "use_trailing": True},

        # 1h: 16.54 R. İkinci motor.
        "1h": {"rr": 3.0, "rsi": 40, "slope": 0.9, "at_active": True, "use_trailing": False},

        "15m": {"rr": 3.0, "rsi": 30, "slope": 0.3, "at_active": True, "use_trailing": False},
        "4h": {"rr": 3.0, "rsi": 30, "slope": 0.3, "at_active": True, "use_trailing": False}
    }
}

# --- HTML ŞABLONU ---
CHART_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body { margin: 0; background-color: #1e1e1e; overflow: hidden; font-family: sans-serif; }
        .modebar { display: none !important; }
        #loading {
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            color: #00ccff; font-size: 20px; font-weight: bold;
        }
    </style>
</head>
<body>
    <div id="loading">Grafik Bekleniyor...</div>
    <div id="chart" style="width:100%;height:100vh;"></div>
    <script>
        var plotDiv = document.getElementById('chart');
        var loadingDiv = document.getElementById('loading');
        var isInitialized = false;

        window.updateChartData = function(jsonStr) {
            try {
                if (loadingDiv) loadingDiv.style.display = 'none';
                var dataObj = JSON.parse(jsonStr);
                var layout = {
                    dragmode: 'pan',
                    uirevision: dataObj.symbol, 
                    xaxis: { type: 'date', rangeslider: {visible: false}, gridcolor: '#333' },
                    yaxis: { gridcolor: '#333', fixedrange: false, autorange: true },
                    paper_bgcolor: '#121212',
                    plot_bgcolor: '#121212',
                    font: { color: '#ddd' },
                    margin: { l: 50, r: 60, t: 30, b: 30 },
                    showlegend: false,
                    shapes: dataObj.shapes || []
                };
                if (!isInitialized) {
                    Plotly.newPlot(plotDiv, dataObj.traces, layout, {responsive: true, scrollZoom: true});
                    isInitialized = true;
                } else {
                    Plotly.react(plotDiv, dataObj.traces, layout);
                }
            } catch (e) { console.error(e); }
        }
    </script>
</body>
</html>
"""


DEFAULT_STRATEGY_CONFIG = {
    "rr": 2.0,
    "rsi": 65,
    "slope": 0.4,
    "at_active": False,
    "use_trailing": False,
    "use_dynamic_pbema_tp": True,
    "hold_n": 4,
    "min_hold_frac": 0.65,
    "pb_touch_tolerance": 0.0018,
    "body_tolerance": 0.0020,
    "cloud_keltner_gap_min": 0.0025,
    "tp_min_dist_ratio": 0.0010,
    "tp_max_dist_ratio": 0.035,
    "adx_min": 10.0,
}


def _generate_candidate_configs():
    """Create a compact grid of configs to search for higher trade density."""

    rr_vals = np.arange(1.2, 2.6, 0.3)
    rsi_vals = np.arange(35, 76, 10)
    slope_vals = np.arange(0.2, 0.9, 0.2)
    at_vals = [False, True]

    candidates = []
    for rr, rsi, slope, at_active in itertools.product(rr_vals, rsi_vals, slope_vals, at_vals):
        candidates.append(
            {
                "rr": round(float(rr), 2),
                "rsi": int(rsi),
                "slope": round(float(slope), 2),
                "at_active": bool(at_active),
                "use_trailing": False,
                "use_dynamic_pbema_tp": False,
            }
        )

    # Birkaç agresif trailing seçeneği ekle
    trailing_extras = []
    for base in candidates[:: max(1, len(candidates) // 20)]:  # toplamı şişirmeden örnekle
        cfg = dict(base)
        cfg["use_trailing"] = True
        trailing_extras.append(cfg)

    return candidates + trailing_extras


def _score_config_for_stream(df: pd.DataFrame, sym: str, tf: str, config: dict) -> Tuple[float, int]:
    """Simulate a single timeframe with the given config and return (net_pnl, trades).

    Not: Öndeki uyumsuzluk, tarayıcıdaki skorlamanın mum kapanışından
    aynı mumda giriş yapıp cooldown/açık trade kontrollerini atlaması yüzünden
    backtestten daha fazla trade ve PnL raporlamasından kaynaklanıyordu. Burada
    backtest ile birebir aynı kuralları (cooldown, açık trade engeli ve bir
    sonraki mum açılışından giriş) uygularız ki skor ve backtest sonuçları
    tutarlı kalsın.
    """

    tm = SimTradeManager(initial_balance=TRADING_CONFIG["initial_balance"])
    warmup = 250
    end = len(df) - 2
    if end <= warmup:
        return 0.0, 0

    for i in range(warmup, end):
        row = df.iloc[i]
        event_time = row["timestamp"] + _tf_to_timedelta(tf)
        tm.update_trades(
            sym,
            tf,
            candle_high=float(row["high"]),
            candle_low=float(row["low"]),
            candle_close=float(row["close"]),
            candle_time_utc=event_time,
            pb_top=float(row.get("pb_ema_top", row["close"])),
            pb_bot=float(row.get("pb_ema_bot", row["close"])),
        )

        s_type, s_entry, s_tp, s_sl, s_reason = TradingEngine.check_signal_diagnostic(
            df,
            index=i,
            min_rr=config["rr"],
            rsi_limit=config["rsi"],
            slope_thresh=config["slope"],
            use_alphatrend=config.get("at_active", False),
            hold_n=config.get("hold_n", DEFAULT_STRATEGY_CONFIG["hold_n"]),
            min_hold_frac=config.get("min_hold_frac", DEFAULT_STRATEGY_CONFIG["min_hold_frac"]),
            pb_touch_tolerance=config.get("pb_touch_tolerance", DEFAULT_STRATEGY_CONFIG["pb_touch_tolerance"]),
            body_tolerance=config.get("body_tolerance", DEFAULT_STRATEGY_CONFIG["body_tolerance"]),
            cloud_keltner_gap_min=config.get("cloud_keltner_gap_min", DEFAULT_STRATEGY_CONFIG["cloud_keltner_gap_min"]),
            tp_min_dist_ratio=config.get("tp_min_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_min_dist_ratio"]),
            tp_max_dist_ratio=config.get("tp_max_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_max_dist_ratio"]),
            adx_min=config.get("adx_min", DEFAULT_STRATEGY_CONFIG["adx_min"]),
        )

        if not (s_type and "ACCEPTED" in s_reason):
            continue

        has_open = any(
            t.get("symbol") == sym and t.get("timeframe") == tf for t in tm.open_trades
        )
        if has_open or tm.check_cooldown(sym, tf, event_time):
            continue

        next_row = df.iloc[i + 1]
        entry_open = float(next_row["open"])
        open_ts = next_row["timestamp"]
        ts_str = (open_ts + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")

        tm.open_trade(
            {
                "symbol": sym,
                "timeframe": tf,
                "type": s_type,
                "setup": s_reason,
                "entry": entry_open,
                "tp": s_tp,
                "sl": s_sl,
                "timestamp": ts_str,
                "open_time_utc": open_ts,
                "use_trailing": config.get("use_trailing", False),
                "use_dynamic_pbema_tp": config.get("use_dynamic_pbema_tp", False),
            }
        )

    unique_trades = len({t.get("id") for t in tm.history}) if tm.history else 0
    return tm.total_pnl, unique_trades


def _optimize_backtest_configs(streams: dict, requested_pairs: list, progress_callback=None):
    """Brute-force search to find the best config (by net pnl) per symbol/timeframe."""

    def log(msg: str):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    candidates = _generate_candidate_configs()
    total_jobs = len([1 for pair in requested_pairs if pair in streams]) * len(candidates)
    if total_jobs == 0:
        return {}

    best_by_pair = {}
    completed = 0
    next_progress = 5

    log(f"[OPT] {len(candidates)} farklı ayar taranacak (her zaman dilimi için).")

    for sym, tf in requested_pairs:
        if (sym, tf) not in streams:
            continue

        df = streams[(sym, tf)]
        best_cfg = None
        best_pnl = -float("inf")
        best_trades = 0

        for cfg in candidates:
            net_pnl, trades = _score_config_for_stream(df, sym, tf, cfg)
            completed += 1
            progress = (completed / total_jobs) * 100
            if progress >= next_progress:
                log(f"[OPT] %{progress:.1f} tamamlandı...")
                next_progress += 5

            if trades == 0:
                continue

            if net_pnl > best_pnl:
                best_pnl = net_pnl
                best_cfg = cfg
                best_trades = trades

        if best_cfg:
            best_by_pair[(sym, tf)] = {**best_cfg, "_net_pnl": best_pnl, "_trades": best_trades}
            log(
                f"[OPT][{sym}-{tf}] En iyi ayar: RR={best_cfg['rr']}, RSI={best_cfg['rsi']}, "
                f"Slope={best_cfg['slope']}, AT={'Açık' if best_cfg['at_active'] else 'Kapalı'} | "
                f"Net PnL={best_pnl:.2f}, Trades={best_trades}"
            )
        else:
            log(f"[OPT][{sym}-{tf}] Uygun ayar bulunamadı (yetersiz trade)")

    log("[OPT] Tarama tamamlandı. Bulunan ayarlar backtest'e uygulanacak.")
    return best_by_pair


def load_optimized_config(symbol, timeframe):
    """Return optimized config for given symbol/timeframe with safe defaults.

    Öncelik sırası:
    1. GUI'den veya CLI'den yapılan backtest sonucunda kaydedilen en iyi ayarlar
    2. SYMBOL_PARAMS içinde tanımlı manuel ayarlar
    3. Güvenli varsayılanlar
    """

    def _load_best_configs():
        global BEST_CONFIG_CACHE
        if BEST_CONFIG_CACHE:
            return BEST_CONFIG_CACHE
        if os.path.exists(BEST_CONFIGS_FILE):
            try:
                with open(BEST_CONFIGS_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Beklenen format: {"SYMBOL": {"tf": {...}}}
                if isinstance(raw, dict):
                    BEST_CONFIG_CACHE = raw
            except Exception:
                BEST_CONFIG_CACHE = {}
        return BEST_CONFIG_CACHE

    defaults = {
        "rr": 3.0,
        "rsi": 60,
        "slope": 0.5,
        "at_active": False,
        "use_trailing": False,
        "use_dynamic_pbema_tp": True,
    }

    best_cfgs = _load_best_configs()
    symbol_cfg = SYMBOL_PARAMS.get(symbol, {})
    tf_cfg = symbol_cfg.get(timeframe, {}) if isinstance(symbol_cfg, dict) else {}

    if isinstance(best_cfgs, dict):
        sym_dict = best_cfgs.get(symbol, {}) if isinstance(best_cfgs.get(symbol), dict) else {}
        if isinstance(sym_dict, dict) and timeframe in sym_dict:
            tf_cfg = {**tf_cfg, **sym_dict.get(timeframe, {})}

    merged = {**defaults, **tf_cfg}

    # Backward compatibility: ensure missing keys fall back to defaults
    for k, v in defaults.items():
        merged.setdefault(k, v)

    return merged


def save_best_configs(best_configs: dict):
    """Persist best backtest configs to disk and cache for live bot usage."""

    global BEST_CONFIG_CACHE
    cleaned = {}
    for (key, cfg) in best_configs.items():
        # key can be tuple (sym, tf) or nested dict
        if isinstance(key, tuple) and len(key) == 2:
            sym, tf = key
            cleaned.setdefault(sym, {})
            cleaned[sym][tf] = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
        elif isinstance(cfg, dict):
            # already nested
            cleaned[key] = cfg

    BEST_CONFIG_CACHE = cleaned
    try:
        with open(BEST_CONFIGS_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
    except Exception:
        pass




# ==========================================
# ==========================================
# 🛠️ TRADE MANAGER (THREAD-SAFE & LOGGING)
# ==========================================
import threading  # Lock mekanizması için gerekli


class TradeManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.open_trades = []
        self.history = []
        self.balances = {tf: TRADING_CONFIG["initial_balance"] for tf in TIMEFRAMES}
        self.cooldowns = {}

        # --- ANA KASA DEĞİŞKENLERİ ---
        self.wallet_balance = TRADING_CONFIG["initial_balance"]
        self.locked_margin = 0.0
        self.total_pnl = 0.0

        # --- MERKEZİ AYARLARDAN OKUMA ---
        self.slippage_pct = TRADING_CONFIG["slippage_rate"]
        # -----------------------------

        self.load_trades()
        print("✅ TRADE MANAGER BAŞLATILDI: Veriler Yüklendi 📂")

    def check_cooldown(self, symbol, timeframe, now_utc=None):
        """
        İşlem sonrası cooldown kontrolü.
        - now_utc pandas.Timestamp da olabilir, datetime da olabilir.
        - Hepsini offset-naive (tzinfo=None) datetime'a çevirip karşılaştırıyoruz.
        """
        k = (symbol, timeframe)

        if now_utc is None:
            now_utc = datetime.utcnow()

        # now_utc'yi normalize et
        if isinstance(now_utc, pd.Timestamp):
            now_utc = now_utc.to_pydatetime()
        if hasattr(now_utc, "tzinfo") and now_utc.tzinfo is not None:
            now_utc = now_utc.replace(tzinfo=None)

        if k not in self.cooldowns:
            return False

        expiry = self.cooldowns[k]

        # expiry'yi de normalize et (her ihtimale karşı)
        if isinstance(expiry, pd.Timestamp):
            expiry = expiry.to_pydatetime()
        if hasattr(expiry, "tzinfo") and expiry.tzinfo is not None:
            expiry = expiry.replace(tzinfo=None)

        # dict içinde normalize edilmiş halini sakla
        self.cooldowns[k] = expiry

        if now_utc < expiry:
            # hâlâ cooldown içindeyiz
            return True

        # cooldown süresi dolduysa kaydı temizle
        del self.cooldowns[k]
        return False

    def open_trade(self, signal_data):
        with self.lock:
            tf = signal_data["timeframe"]
            sym = signal_data["symbol"]

            if (sym, tf) in self.cooldowns:
                if datetime.now() < self.cooldowns[(sym, tf)]:
                    return
                else:
                    del self.cooldowns[(sym, tf)]

            setup_type = signal_data.get("setup", "Unknown")

            if self.wallet_balance < 10:
                print(f"⚠️ Yetersiz Bakiye (${self.wallet_balance:.2f}). İşlem açılamadı.")
                return

            # SLIPPAGE MODELLEMESİ
            raw_entry = float(signal_data["entry"])
            trade_type = signal_data["type"]

            if trade_type == "LONG":
                real_entry = raw_entry * (1 + self.slippage_pct)
            else:
                real_entry = raw_entry * (1 - self.slippage_pct)

            # GLOBAL AYARLARDAN MARJİN HESABI
            margin_to_use = self.wallet_balance * TRADING_CONFIG["usable_balance_pct"]
            trade_size = margin_to_use * TRADING_CONFIG["leverage"]

            new_trade = {
                "id": int(time.time() * 1000), "symbol": sym, "timestamp": signal_data["timestamp"],
                "open_time_utc": (signal_data.get("open_time_utc") or datetime.utcnow()).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "timeframe": tf, "type": trade_type, "setup": setup_type,
                "entry": real_entry,
                "tp": float(signal_data["tp"]), "sl": float(signal_data["sl"]),
                "size": trade_size, "margin": margin_to_use,
                "status": "OPEN", "pnl": 0.0,
                "breakeven": False, "trailing_active": False, "partial_taken": False,
                "has_cash": True, "close_time": "", "close_price": ""
            }

            self.wallet_balance -= margin_to_use
            self.locked_margin += margin_to_use

            self.open_trades.append(new_trade)
            self.save_trades()

    def update_trades(self, symbol, tf,
                      candle_high, candle_low, candle_close,
                      candle_time_utc=None,
                      pb_top=None, pb_bot=None):
        """
        Trade update modeli (daha gerçekçi):
        - Mum içi (high/low) ile TP/SL tetiklerini yakalar.
        - TP, mümkünse dinamik olarak PBEMA cloud seviyesine göre değerlendirilir.
        - Aynı mumda hem TP hem SL görülürse konservatif olarak STOP seçer.
        - Partial TP (%50) + breakeven / trailing SL desteklenir.
        - Çıkışta slippage + komisyon + basit funding maliyeti düşer.
        """
        with self.lock:
            if candle_time_utc is None:
                candle_time_utc = datetime.utcnow()

            closed_indices = []
            trades_updated = False
            just_closed_trades = []

            for i, trade in enumerate(self.open_trades):
                if trade.get("symbol") != symbol:
                    continue
                if trade.get("timeframe") != tf:
                    continue

                entry = float(trade["entry"])
                tp = float(trade["tp"])
                sl = float(trade["sl"])
                size = float(trade["size"])
                t_type = trade["type"]
                initial_margin = float(trade.get("margin", size / TRADING_CONFIG["leverage"]))

                config = load_optimized_config(symbol, tf)
                use_trailing = config.get("use_trailing", False)
                use_partial = not use_trailing
                use_dynamic_tp = config.get("use_dynamic_pbema_tp", True)

                # --- Fiyatlar ---
                if t_type == "LONG":
                    close_price = candle_close
                    fav_price = candle_high  # long için en iyi fiyat wick-high
                    pnl_percent_close = (close_price - entry) / entry
                    pnl_percent_fav = (fav_price - entry) / entry
                    in_profit = fav_price > entry
                else:
                    close_price = candle_close
                    fav_price = candle_low  # short için en iyi fiyat wick-low
                    pnl_percent_close = (entry - close_price) / entry
                    pnl_percent_fav = (entry - fav_price) / entry
                    in_profit = fav_price < entry

                # Dinamik PBEMA TP: varsa her mumda bulutun güncel seviyesini hedefle
                dyn_tp = tp
                if use_dynamic_tp:
                    try:
                        if pb_top is not None and pb_bot is not None:
                            dyn_tp = float(pb_bot) if t_type == "LONG" else float(pb_top)
                    except Exception:
                        dyn_tp = tp
                    self.open_trades[i]["tp"] = dyn_tp

                # Ekranda gösterilecek anlık PnL (kapanışa göre)
                self.open_trades[i]["pnl"] = pnl_percent_close * size

                # Hedefe ilerleme oranı (en iyi fiyata göre)
                total_dist = abs(dyn_tp - entry)
                if total_dist <= 0:
                    continue
                current_dist = abs(fav_price - entry)
                progress = current_dist / total_dist if total_dist > 0 else 0.0

                # ---------- PARTIAL TP + BREAKEVEN ----------
                if in_profit and use_partial:
                    if (not trade.get("partial_taken")) and progress >= 0.50:
                        partial_size = size / 2.0
                        partial_pnl = pnl_percent_fav * partial_size
                        commission = partial_size * TRADING_CONFIG["total_fee"]
                        net_partial_pnl = partial_pnl - commission
                        margin_release = initial_margin / 2.0

                        self.wallet_balance += margin_release + net_partial_pnl
                        self.locked_margin -= margin_release
                        self.total_pnl += net_partial_pnl

                        partial_record = trade.copy()
                        partial_record["size"] = partial_size
                        partial_record["pnl"] = net_partial_pnl
                        partial_record["status"] = "PARTIAL TP (50%)"
                        partial_record["close_time"] = (candle_time_utc + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
                        partial_record["close_price"] = float(fav_price)
                        partial_record["pb_ema_top"] = pb_top
                        partial_record["pb_ema_bot"] = pb_bot
                        self.history.append(partial_record)

                        # Açık trade'i güncelle: yarı pozisyon kaldı, margin yarıya indi
                        self.open_trades[i]["size"] = partial_size
                        self.open_trades[i]["margin"] = margin_release
                        self.open_trades[i]["partial_taken"] = True
                        # Breakeven'e çek
                        self.open_trades[i]["sl"] = entry
                        self.open_trades[i]["breakeven"] = True
                        trades_updated = True

                    elif (not trade.get("breakeven")) and progress >= 0.40:
                        self.open_trades[i]["sl"] = entry
                        self.open_trades[i]["breakeven"] = True
                        trades_updated = True

                # ---------- TRAILING SL ----------
                if in_profit and use_trailing:
                    if (not trade.get("breakeven")) and progress >= 0.40:
                        self.open_trades[i]["sl"] = entry
                        self.open_trades[i]["breakeven"] = True
                        trades_updated = True

                    if progress >= 0.50:
                        trail_buffer = total_dist * 0.40
                        current_sl = float(self.open_trades[i]["sl"])
                        if t_type == "LONG":
                            new_sl = close_price - trail_buffer
                            if new_sl > current_sl:
                                self.open_trades[i]["sl"] = new_sl
                                self.open_trades[i]["trailing_active"] = True
                                trades_updated = True
                        else:
                            new_sl = close_price + trail_buffer
                            if new_sl < current_sl:
                                self.open_trades[i]["sl"] = new_sl
                                self.open_trades[i]["trailing_active"] = True
                                trades_updated = True

                # ---------- SL / TP KONTROLÜ ----------
                sl = float(self.open_trades[i]["sl"])

                if t_type == "LONG":
                    hit_tp = candle_high >= dyn_tp
                    hit_sl = candle_low <= sl
                else:
                    hit_tp = candle_low <= dyn_tp
                    hit_sl = candle_high >= sl

                if not (hit_tp or hit_sl):
                    continue

                if hit_tp and hit_sl:
                    reason = "STOP (BothHit)"
                    exit_level = sl
                elif hit_tp:
                    reason = "WIN (TP)"
                    exit_level = dyn_tp
                else:
                    reason = "STOP"
                    exit_level = sl

                # ---------- POZİSYONU KAPAT ----------
                current_size = float(self.open_trades[i]["size"])
                margin_release = float(self.open_trades[i].get("margin", initial_margin))

                if t_type == "LONG":
                    exit_fill = float(exit_level) * (1 - self.slippage_pct)
                    pnl_percent = (exit_fill - entry) / entry
                else:
                    exit_fill = float(exit_level) * (1 + self.slippage_pct)
                    pnl_percent = (entry - exit_fill) / entry

                gross_pnl = pnl_percent * current_size
                commission = current_size * TRADING_CONFIG["total_fee"]

                funding_cost = 0.0
                try:
                    open_time_str = trade.get("open_time_utc", "")
                    if open_time_str:
                        open_dt = datetime.strptime(open_time_str, "%Y-%m-%dT%H:%M:%SZ")
                        hours = max(0.0, (candle_time_utc - open_dt).total_seconds() / 3600.0)
                        funding_cost = abs(current_size) * TRADING_CONFIG["funding_rate_8h"] * (hours / 8.0)
                except Exception:
                    funding_cost = 0.0

                final_net_pnl = gross_pnl - commission - funding_cost

                self.wallet_balance += margin_release + final_net_pnl
                self.locked_margin -= margin_release
                self.total_pnl += final_net_pnl

                # Cooldown sadece gerçek STOP durumunda
                if "STOP" in reason:
                    wait_minutes = 10 if tf == "1m" else (30 if tf == "5m" else 60)
                    self.cooldowns[(symbol, tf)] = datetime.utcnow() + timedelta(minutes=wait_minutes)

                # BE statüsünü ayır
                if trade.get("breakeven") and abs(final_net_pnl) < 1e-6 and "STOP" in reason:
                    reason = "BE"

                trade["status"] = reason
                trade["pnl"] = final_net_pnl
                trade["close_time"] = (candle_time_utc + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
                trade["close_price"] = float(exit_fill)
                trade["pb_ema_top"] = pb_top
                trade["pb_ema_bot"] = pb_bot

                self.history.append(trade)
                just_closed_trades.append(trade)
                closed_indices.append(i)
                trades_updated = True

            for idx in sorted(closed_indices, reverse=True):
                del self.open_trades[idx]

            if trades_updated:
                self.save_trades()

            return just_closed_trades

    def save_trades(self):
        with self.lock:
            try:
                cols = ["id", "symbol", "timestamp", "timeframe", "type", "setup", "entry", "tp", "sl", "size",
                        "margin",
                        "status", "pnl", "breakeven", "trailing_active", "partial_taken", "has_cash", "close_time",
                        "close_price"]

                if not self.open_trades and not self.history:
                    df_all = pd.DataFrame(columns=cols)
                else:
                    df_all = pd.concat([pd.DataFrame(self.open_trades), pd.DataFrame(self.history)], ignore_index=True)

                for c in cols:
                    if c not in df_all.columns: df_all[c] = ""

                # ATOMIC WRITE
                fd, tmp_path = tempfile.mkstemp(prefix="trades_temp_", suffix=".csv", dir=".")
                os.close(fd)
                df_all[cols].to_csv(tmp_path, index=False)
                shutil.move(tmp_path, CSV_FILE)

            except Exception as e:
                print(f"KAYIT HATASI: {e}")
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                with open("error_log.txt", "a") as f:
                    f.write(f"\n[{datetime.now()}] SAVE_TRADES HATA: {str(e)}\n")
                    f.write(traceback.format_exc())

    def load_trades(self):
        with self.lock:
            self.wallet_balance = TRADING_CONFIG["initial_balance"]
            self.locked_margin = 0.0
            self.total_pnl = 0.0

            if os.path.exists(CSV_FILE):
                try:
                    if os.path.getsize(CSV_FILE) == 0: return
                    df = pd.read_csv(CSV_FILE)
                    if "symbol" in df.columns:
                        self.open_trades = df[df["status"].astype(str).str.contains("OPEN")].to_dict('records')
                        self.history = df[~df["status"].astype(str).str.contains("OPEN")].to_dict('records')

                        for trade in self.history:
                            self.total_pnl += float(trade['pnl'])

                        open_pnl = 0.0
                        for trade in self.open_trades:
                            m = float(trade.get('margin', float(trade['size']) / TRADING_CONFIG["leverage"]))
                            self.locked_margin += m
                            open_pnl += float(trade.get('pnl', 0.0))

                        # Kullanılabilir bakiye = başlangıç + kapalı işlemlerden net PnL - kilitli marj
                        self.wallet_balance = TRADING_CONFIG["initial_balance"] + self.total_pnl - self.locked_margin
                        total_equity = self.wallet_balance + self.locked_margin + open_pnl
                        print(
                            "📂 Veriler Yüklendi. "
                            f"Toplam Varlık (Equity): ${total_equity:.2f} | "
                            f"Kullanılabilir Bakiye: ${self.wallet_balance:.2f} | "
                            f"Kilitli Marj: ${self.locked_margin:.2f}")

                except Exception as e:
                    print(f"YÜKLEME HATASI: {e}")
                    with open("error_log.txt", "a") as f:
                        f.write(f"\n[{datetime.now()}] LOAD_TRADES HATA: {str(e)}\n")
                    self.open_trades = []
                    self.history = []

    def reset_logs(self):
        with self.lock:
            self.open_trades = []
            self.history = []
            self.wallet_balance = TRADING_CONFIG["initial_balance"]
            self.locked_margin = 0.0
            self.total_pnl = 0.0
            self.cooldowns = {}
            self.save_trades()

    def reset_balances(self):
        with self.lock:
            self.reset_logs()


trade_manager = TradeManager()


# --- TRADING ENGINE (ROBUST API & RETRY MECHANISM) ---
class TradingEngine:
    @staticmethod
    def send_telegram(token, chat_id, message):
        if not token or not chat_id: return

        def sender():
            try:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                data = {"chat_id": chat_id, "text": message}
                requests.post(url, data=data, timeout=10)
            except Exception as e:
                print(f"TELEGRAM HATA: {e}")

        import threading
        threading.Thread(target=sender).start()

    # --- YENİ: AKILLI İSTEK FONKSİYONU (RETRY LOGIC) ---
    @staticmethod
    def http_get_with_retry(url, params, max_retries=3, timeout=10):
        """Hata durumunda bekleyip tekrar deneyen güvenli istek fonksiyonu"""
        delay = 1
        for attempt in range(max_retries):
            try:
                res = requests.get(url, params=params, timeout=timeout)

                # Eğer Binance "Çok Hızlısın" (429) veya "Sunucu Hatası" (5xx) derse:
                if res.status_code == 429 or res.status_code >= 500:
                    # Hatayı logla ama çökme
                    print(f"API HATA {res.status_code} (Deneme {attempt + 1}/{max_retries}). Bekleniyor...")
                    time.sleep(delay)
                    delay *= 2  # Bekleme süresini katla (1sn -> 2sn -> 4sn)
                    continue

                # Diğer hatalarda (404 vb) direkt döndür
                return res
            except requests.exceptions.RequestException as e:
                print(f"BAĞLANTI HATASI (Deneme {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)
                delay *= 2

        return None  # Tüm denemeler başarısız olduysa

    @staticmethod
    def get_data(symbol, interval, limit=500):
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}

            # Eski requests.get yerine yeni akıllı fonksiyonu kullanıyoruz
            res = TradingEngine.http_get_with_retry(url, params)

            if res is None: return pd.DataFrame()  # Başarısız oldu

            data = res.json()
            if not data or not isinstance(data, list): return pd.DataFrame()

            df = pd.DataFrame(data).iloc[:, :6]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            print(f"VERİ ÇEKME HATASI ({symbol}): {e}")
            with open("error_log.txt", "a") as f:
                f.write(f"\n[{datetime.now()}] GET_DATA HATA ({symbol}): {str(e)}\n")
            return pd.DataFrame()

    @staticmethod
    def fetch_worker(args):
        symbol, tf = args
        try:
            df = TradingEngine.get_data(symbol, tf, limit=500)
            return (symbol, tf, df)
        except Exception as e:
            return (symbol, tf, pd.DataFrame())

    @staticmethod
    def get_all_candles_parallel(symbol_list, timeframe_list):
        import concurrent.futures
        tasks = list(itertools.product(symbol_list, timeframe_list))
        results = {}

        # --- DÜZELTME: Thread sayısını 20'den 5'e düşürdük (Rate Limit Koruması) ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_task = {executor.submit(TradingEngine.fetch_worker, t): t for t in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    sym, tf, df = future.result()
                    results[(sym, tf)] = df
                except Exception as e:
                    print(f"Paralel Veri Hatası: {e}")
        return results

    @staticmethod
    def get_latest_prices(symbols):
        """Lightweight ticker fetcher to refresh UI prices without heavy kline calls."""

        prices = {}
        url = "https://fapi.binance.com/fapi/v1/ticker/price"

        for sym in symbols:
            try:
                res = TradingEngine.http_get_with_retry(url, {"symbol": sym}, max_retries=2)
                if res is None:
                    continue
                data = res.json()
                if isinstance(data, dict) and "price" in data:
                    prices[sym] = float(data["price"])
            except Exception as e:
                print(f"[PRICE] {sym} fiyatı alınamadı: {e}")

        return prices

    @staticmethod
    def get_historical_data_pagination(symbol, interval, total_candles=5000):
        all_data = []
        end_time = int(time.time() * 1000)
        limit_per_req = 1000
        loops = int(np.ceil(total_candles / limit_per_req))

        for _ in range(loops):
            try:
                url = "https://fapi.binance.com/fapi/v1/klines"
                params = {'symbol': symbol, 'interval': interval, 'limit': limit_per_req, 'endTime': end_time}

                # Burada da akıllı retry kullanıyoruz
                res = TradingEngine.http_get_with_retry(url, params)
                if res is None: break

                data = res.json()
                if not data or not isinstance(data, list): break

                all_data = data + all_data  # Eskiden yeniye doğru birleştir
                end_time = data[0][0] - 1
                time.sleep(0.1)  # Kısa bir mola (Rate limit nezaketi)
            except:
                break

        if not all_data: return pd.DataFrame()

        df = pd.DataFrame(all_data).iloc[:, :6]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.tail(total_candles).reset_index(drop=True)

    @staticmethod
    def calculate_alphatrend(df, coeff=1, ap=14):
        try:
            import warnings;
            warnings.simplefilter(action='ignore', category=FutureWarning)
            if 'volume' in df.columns: df['volume'] = df['volume'].astype(float)
            df['tr'] = ta.true_range(df['high'], df['low'], df['close'])
            df['atr_at'] = ta.sma(df['tr'], length=ap)
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=ap)
                condition = df['mfi'] >= 50
            else:
                df['rsi_at'] = ta.rsi(df['close'], length=ap)
                condition = df['rsi_at'] >= 50
            df['upT'] = df['low'] - df['atr_at'] * coeff
            df['downT'] = df['high'] + df['atr_at'] * coeff
            alpha_trend = np.zeros(len(df))
            upT_vals = df['upT'].values
            downT_vals = df['downT'].values
            cond_vals = condition.fillna(False).values.astype(bool)
            close_vals = df['close'].values
            alpha_trend[0] = close_vals[0]
            for i in range(1, len(df)):
                prev_at = alpha_trend[i - 1]
                if cond_vals[i]:
                    alpha_trend[i] = prev_at if upT_vals[i] < prev_at else upT_vals[i]
                else:
                    alpha_trend[i] = prev_at if downT_vals[i] > prev_at else downT_vals[i]
            df['alphatrend'] = alpha_trend
            df['alphatrend_2'] = df['alphatrend'].shift(2)
            return df
        except:
            df['alphatrend'] = df['close'];
            return df

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Base Setup için kullanılan tüm indikatörleri hesaplar.

        - RSI(14)
        - ADX(14)
        - PBEMA cloud: EMA200(high) ve EMA200(close)
        - SSL baseline: HMA60(close)
        - Keltner bandı: baseline ± EMA60(TrueRange) * 0.2
        - AlphaTrend: opsiyonel filtre için hazırlanır
        """
        df = df.copy()

        # Temel kolonları float'a çevir
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # RSI ve ADX
        df["rsi"] = ta.rsi(df["close"], length=14)
        adx_res = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx_res["ADX_14"] if adx_res is not None and "ADX_14" in adx_res.columns else 0.0

        # PBEMA cloud
        df["pb_ema_top"] = ta.ema(df["high"], length=200)
        df["pb_ema_bot"] = ta.ema(df["close"], length=200)

        # Slope (şimdilik sadece bilgi amaçlı)
        df["slope_top"] = (df["pb_ema_top"].diff(5) / df["pb_ema_top"]) * 1000
        df["slope_bot"] = (df["pb_ema_bot"].diff(5) / df["pb_ema_bot"]) * 1000

        # SSL baseline (HMA60) ve Keltner bantları
        df["baseline"] = ta.hma(df["close"], length=60)
        tr = ta.true_range(df["high"], df["low"], df["close"])
        range_ma = ta.ema(tr, length=60)
        df["keltner_upper"] = df["baseline"] + range_ma * 0.2
        df["keltner_lower"] = df["baseline"] - range_ma * 0.2

        # AlphaTrend (isteğe göre filtrede kullanılacak)
        df = TradingEngine.calculate_alphatrend(df, coeff=1, ap=14)

        return df

    @staticmethod
    def check_signal_diagnostic(
            df: pd.DataFrame,
            index: int = -2,
            min_rr: float = 2.0,
            rsi_limit: float = 60.0,
            slope_thresh: float = 0.5,
            use_alphatrend: bool = True,
            hold_n: int = 5,
            min_hold_frac: float = 0.8,
            pb_touch_tolerance: float = 0.0012,
            body_tolerance: float = 0.0015,
            cloud_keltner_gap_min: float = 0.003,
            tp_min_dist_ratio: float = 0.0015,
            tp_max_dist_ratio: float = 0.03,
            adx_min: float = 12.0,
    ) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], str]:
        """
        Base Setup için LONG / SHORT sinyali üretir.

        Filtreler:
        - ADX düşükse alma
        - Keltner holding + retest
        - PBEMA cloud hizalaması
        - Keltner bandı ile PBEMA TP hedefi arasında minimum mesafe
        - TP çok yakın / çok uzak değil
        - RR >= min_rr   (RR = reward / risk)
        - ***Trend filtresi (hafif):***
              * Güçlü uptrend + fiyat PBEMA üstünde => SHORT yasak
              * Güçlü downtrend + fiyat PBEMA altında => LONG yasak
        """

        if df is None or df.empty:
            return None, None, None, None, "No Data"

        required_cols = [
            "open", "high", "low", "close",
            "rsi", "adx",
            "pb_ema_top", "pb_ema_bot",
            "keltner_upper", "keltner_lower",
        ]
        for col in required_cols:
            if col not in df.columns:
                return None, None, None, None, f"Missing {col}"

        try:
            curr = df.iloc[index]
        except Exception:
            return None, None, None, None, "Index Error"

        for c in required_cols:
            v = curr.get(c)
            if pd.isna(v):
                return None, None, None, None, f"NaN in {c}"

        abs_index = index if index >= 0 else (len(df) + index)
        if abs_index < 0 or abs_index >= len(df):
            return None, None, None, None, "Index Out of Range"

        # --- Parametreler ---
        hold_n = int(max(1, hold_n or 1))
        min_hold_frac = float(min_hold_frac if min_hold_frac is not None else 0.8)
        touch_tol = float(pb_touch_tolerance if pb_touch_tolerance is not None else 0.0012)
        body_tol = float(body_tolerance if body_tolerance is not None else 0.0015)
        cloud_keltner_gap_min = float(cloud_keltner_gap_min if cloud_keltner_gap_min is not None else 0.003)
        tp_min_dist_ratio = float(tp_min_dist_ratio if tp_min_dist_ratio is not None else 0.0015)
        tp_max_dist_ratio = float(tp_max_dist_ratio if tp_max_dist_ratio is not None else 0.03)
        adx_min = float(adx_min if adx_min is not None else 12.0)

        # ADX filtresi
        if float(curr["adx"]) < adx_min:
            return None, None, None, None, "ADX Low"

        if abs_index < hold_n + 1:
            return None, None, None, None, "Warmup"

        slc = slice(abs_index - hold_n, abs_index)
        closes_slice = df["close"].iloc[slc]
        upper_slice = df["keltner_upper"].iloc[slc]
        lower_slice = df["keltner_lower"].iloc[slc]

        close = float(curr["close"])
        open_ = float(curr["open"])
        high = float(curr["high"])
        low = float(curr["low"])
        upper_band = float(curr["keltner_upper"])
        lower_band = float(curr["keltner_lower"])
        pb_top = float(curr["pb_ema_top"])
        pb_bot = float(curr["pb_ema_bot"])

        # --- Hafif trend filtresi (yalnızca aşırı ters işlemleri keser) ---
        slope_top = float(curr.get("slope_top", 0.0) or 0.0)
        slope_bot = float(curr.get("slope_bot", 0.0) or 0.0)
        slope_thresh = slope_thresh or 0.0

        # güçlü yukarı trend ve fiyat PBEMA bulutunun ÜSTÜNDE ise => short yasak
        trend_up_strong = (
                slope_top > slope_thresh and
                pb_top >= pb_bot and
                close > pb_top
        )

        # güçlü aşağı trend ve fiyat PBEMA bulutunun ALTINDA ise => long yasak
        trend_down_strong = (
                slope_bot < -slope_thresh and
                pb_bot <= pb_top and
                close < pb_bot
        )

        long_direction_ok = not trend_down_strong
        short_direction_ok = not trend_up_strong

        # ================= LONG =================
        holding_long = (closes_slice > lower_slice).mean() >= min_hold_frac

        retest_long = (
                (low <= lower_band * (1 + touch_tol))
                and (close > lower_band)
                and (min(open_, close) > lower_band * (1 - body_tol))
        )

        keltner_pb_gap_long = (pb_bot - lower_band) / lower_band if lower_band != 0 else 0.0

        within_cloud_long = pb_bot <= close <= pb_top * (1 + touch_tol)
        pb_target_long = (
                long_direction_ok and
                ((close <= pb_bot * (1 + touch_tol)) or within_cloud_long) and
                (keltner_pb_gap_long >= cloud_keltner_gap_min)
        )

        is_long = holding_long and retest_long and pb_target_long

        # ================= SHORT =================
        holding_short = (closes_slice < upper_slice).mean() >= min_hold_frac

        retest_short = (
                (high >= upper_band * (1 - touch_tol))
                and (close < upper_band)
                and (max(open_, close) < upper_band * (1 + body_tol))
        )

        keltner_pb_gap_short = (upper_band - pb_top) / upper_band if upper_band != 0 else 0.0

        within_cloud_short = pb_bot * (1 - touch_tol) <= close <= pb_top
        pb_target_short = (
                short_direction_ok and
                ((close >= pb_top * (1 - touch_tol)) or within_cloud_short) and
                (keltner_pb_gap_short >= cloud_keltner_gap_min)
        )

        is_short = holding_short and retest_short and pb_target_short

        # --- RSI (LONG için üst sınır) ---
        long_rsi_limit = rsi_limit + 10.0
        rsi_val = float(curr["rsi"])
        if is_long and rsi_val > long_rsi_limit:
            is_long = False

        # --- AlphaTrend (opsiyonel) ---
        if use_alphatrend and "alphatrend" in df.columns:
            at_val = float(curr["alphatrend"])
            if is_long and close < at_val:
                is_long = False
            if is_short and close > at_val:
                is_short = False

        # ---------- LONG ----------
        if is_long:
            swing_n = 20
            start = max(0, abs_index - swing_n)
            swing_low = float(df["low"].iloc[start:abs_index].min())
            if swing_low <= 0:
                return None, None, None, None, "Invalid Swing Low"

            sl_candidate = swing_low * 0.997
            band_sl = lower_band * 0.998
            sl = min(sl_candidate, band_sl)

            entry = close
            tp = pb_bot

            if tp <= entry:
                return None, None, None, None, "TP Below Entry"
            if sl >= entry:
                sl = min(swing_low * 0.995, entry * 0.997)

            risk = entry - sl
            reward = tp - entry
            if risk <= 0 or reward <= 0:
                return None, None, None, None, "Invalid RR"

            rr = reward / risk
            tp_dist_ratio = reward / entry

            if tp_dist_ratio < tp_min_dist_ratio:
                return None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})"
            if tp_dist_ratio > tp_max_dist_ratio:
                return None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})"
            if rr < min_rr:
                return None, None, None, None, f"RR Too Low ({rr:.2f})"

            reason = f"ACCEPTED(Base,R:{rr:.2f})"
            return "LONG", entry, tp, sl, reason

        # ---------- SHORT ----------
        if is_short:
            swing_n = 20
            start = max(0, abs_index - swing_n)
            swing_high = float(df["high"].iloc[start:abs_index].max())
            if swing_high <= 0:
                return None, None, None, None, "Invalid Swing High"

            sl_candidate = swing_high * 1.003
            band_sl = upper_band * 1.002
            sl = max(sl_candidate, band_sl)

            entry = close
            tp = pb_top

            if tp >= entry:
                return None, None, None, None, "TP Above Entry"
            if sl <= entry:
                sl = max(swing_high * 1.005, entry * 1.003)

            risk = sl - entry
            reward = entry - tp
            if risk <= 0 or reward <= 0:
                return None, None, None, None, "Invalid RR"

            rr = reward / risk
            tp_dist_ratio = reward / entry

            if tp_dist_ratio < tp_min_dist_ratio:
                return None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})"
            if tp_dist_ratio > tp_max_dist_ratio:
                return None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})"
            if rr < min_rr:
                return None, None, None, None, f"RR Too Low ({rr:.2f})"

            reason = f"ACCEPTED(Base,R:{rr:.2f})"
            return "SHORT", entry, tp, sl, reason

        return None, None, None, None, "No Signal"

    def debug_plot_backtest_trade(symbol: str,
                                  timeframe: str,
                                  trade_id: int,
                                  trades_csv: str = "bt_trades_base_setup.csv",
                                  window: int = 40):
        """
        Backtest sonrası belirli bir trade'i dahili grafikle görmek için yardımcı fonksiyon.
        - Önce run_portfolio_backtest çalıştırılmış olmalı.
        - <symbol>_<timeframe>_prices.csv ve trades_csv dosyaları mevcut olmalı.
        """

        # 1) Fiyat datası
        prices_path = f"{symbol}_{timeframe}_prices.csv"
        if not os.path.exists(prices_path):
            raise FileNotFoundError(f"Fiyat datası bulunamadı: {prices_path}")

        df_prices = pd.read_csv(prices_path)
        if "timestamp" in df_prices.columns:
            df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"], utc=True)

        # 2) Trade datası
        if not os.path.exists(trades_csv):
            raise FileNotFoundError(f"Trades CSV bulunamadı: {trades_csv}")

        df_trades = pd.read_csv(trades_csv)

        # 3) Plot
        plot_trade(df_prices, df_trades, trade_id=trade_id, window=window)
        plt.show()

    @staticmethod
    def debug_base_short(df, index):
        """
        Belirli bir mum için Base SHORT koşullarını tek tek döndürür.
        index: df.iloc[index] mantığında (örn. -1 son mum)
        """
        import pandas as pd

        curr = df.iloc[index]
        abs_index = index if index >= 0 else (len(df) + index)

        hold_n = 4
        min_hold_frac = 0.50
        touch_tol = 0.0012
        slope_thresh = 0.5

        # ADX ve slope
        adx_ok = curr["adx"] >= 15
        slope_ok_short = curr["slope_top"] <= slope_thresh

        if abs_index >= hold_n + 1:
            hold_slice = slice(abs_index - hold_n, abs_index)
            holding_short = (
                    (df["close"].iloc[hold_slice] < df["keltner_upper"].iloc[hold_slice])
                    .mean() >= min_hold_frac
            )
        else:
            holding_short = False

        retest_short = (
                               curr["high"] >= curr["keltner_upper"] * (1 - touch_tol)
                       ) and (
                               curr["close"] < curr["keltner_upper"]
                       )

        pb_target_short = curr["pb_ema_top"] < curr["close"]

        # RSI limiti (Base short)
        rsi_limit = 60  # 5m config'ine göre
        rsi_thresh = (100 - rsi_limit) - 10
        rsi_ok = curr["rsi"] >= rsi_thresh

        return {
            "time": str(getattr(curr, "name", "")),
            "adx": float(curr["adx"]),
            "rsi": float(curr["rsi"]),
            "adx_ok": adx_ok,
            "slope_ok_short": slope_ok_short,
            "holding_short": bool(holding_short),
            "retest_short": bool(retest_short),
            "pb_target_short": bool(pb_target_short),
            "rsi_ok": rsi_ok,
        }

    @staticmethod
    def create_chart_data_json(df, interval, symbol="BTCUSDT", signal=None, active_trades=[], show_rr=True):
        try:
            plot_df = df.tail(300).copy()
            if len(plot_df) > 1:
                if interval.endswith('m'):
                    interval_mins = int(interval[:-1])
                elif interval.endswith('h'):
                    interval_mins = int(interval[:-1]) * 60
                else:
                    interval_mins = 240
            else:
                interval_mins = 15

            # --- DÜZELTME: TIMESTAMP PARSING (Rapordaki Talep) ---
            # Pandas zaten datetime nesnelerini iyi yönetir, string dönüşümünde .astype(str) veya .strftime en güvenlisidir.
            istanbul_time_series = plot_df['timestamp'] + timedelta(hours=3)
            timestamps_str = istanbul_time_series.dt.strftime('%Y-%m-%d %H:%M').tolist()
            # -----------------------------------------------------

            traces = []
            traces.append({
                'type': 'candlestick', 'x': timestamps_str,
                'open': plot_df['open'].tolist(), 'high': plot_df['high'].tolist(),
                'low': plot_df['low'].tolist(), 'close': plot_df['close'].tolist(),
                'name': symbol, 'increasing': {'line': {'color': '#26a69a'}},
                'decreasing': {'line': {'color': '#ef5350'}}
            })

            def add_line(name, data, color, width=1, dash=None):
                line_data = {'type': 'scatter', 'mode': 'lines', 'x': timestamps_str, 'y': data.fillna(0).tolist(),
                             'line': {'color': color, 'width': width}, 'name': name, 'hoverinfo': 'skip'}
                if dash: line_data['line']['dash'] = dash
                traces.append(line_data)

            # PBEMA bulutu: açık mavi bant (Matplotlib backtest görünümüyle eşleştirildi)
            pb_color = '#42a5f5'
            traces.append({
                'type': 'scatter', 'mode': 'lines', 'x': timestamps_str,
                'y': plot_df['pb_ema_bot'].fillna(0).tolist(),
                'line': {'color': pb_color, 'width': 1},
                'name': 'PB Bot', 'hoverinfo': 'skip',
                'fill': None
            })
            traces.append({
                'type': 'scatter', 'mode': 'lines', 'x': timestamps_str,
                'y': plot_df['pb_ema_top'].fillna(0).tolist(),
                'line': {'color': pb_color, 'width': 1},
                'name': 'PB Top', 'hoverinfo': 'skip',
                'fill': 'tonexty', 'fillcolor': 'rgba(66, 165, 245, 0.18)'
            })

            # Keltner bantları ve baseline (rapor renkleriyle hizalı)
            add_line('Keltner Up', plot_df['keltner_upper'], 'rgba(255, 50, 50, 0.8)', 1, 'dot')
            add_line('Keltner Low', plot_df['keltner_lower'], 'rgba(50, 255, 50, 0.8)', 1, 'dot')
            add_line('Baseline', plot_df['baseline'], 'rgba(255, 215, 0, 0.95)', 1)
            if 'alphatrend' in plot_df.columns: add_line('AlphaTrend', plot_df['alphatrend'], '#00ccff', 2)

            shapes = []
            if show_rr:
                time_diff = plot_df['timestamp'].iloc[-1] - plot_df['timestamp'].iloc[-2]
                past_trades = [t for t in trade_manager.history if
                               t['timeframe'] == interval and t['symbol'] == symbol][-5:]
                all_trades_to_show = active_trades + past_trades
                all_trades_to_show.sort(key=lambda x: x['id'])

                trades_with_visibility = []
                for i, trade in enumerate(all_trades_to_show):
                    draw_box = True
                    if i < len(all_trades_to_show) - 1:
                        next_trade = all_trades_to_show[i + 1]
                        time_diff_ms = next_trade['id'] - trade['id']
                        time_diff_mins = time_diff_ms / 1000 / 60
                        if time_diff_mins < (interval_mins * 15): draw_box = False
                    trades_with_visibility.append((trade, draw_box))

                for trade, draw_box in trades_with_visibility:
                    t_type = trade['type'];
                    entry = float(trade['entry']);
                    tp = float(trade['tp']);
                    sl = float(trade['sl'])

                    # Timestamp güvenli parse
                    start_ts_str = trade.get('timestamp', trade.get('time', ''))
                    try:
                        # dateutil.parser kullanmak daha esnektir
                        start_dt = dateutil.parser.parse(start_ts_str)
                        future_dt = start_dt + (time_diff * 20)
                        future_ts_str = future_dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        continue

                    is_active = trade in active_trades;
                    opacity = "0.3" if is_active else "0.1"
                    fill_prof = f"rgba(0, 255, 0, {opacity})";
                    fill_loss = f"rgba(255, 0, 0, {opacity})"

                    if draw_box:
                        y0_prof, y1_prof = (entry, tp) if t_type == "LONG" else (tp, entry)
                        y0_loss, y1_loss = (sl, entry) if t_type == "LONG" else (entry, sl)

                        shapes.append(
                            {'type': 'rect', 'x0': start_ts_str, 'x1': future_ts_str, 'y0': y0_prof, 'y1': y1_prof,
                             'fillcolor': fill_prof, 'line': {'width': 0}})
                        shapes.append(
                            {'type': 'rect', 'x0': start_ts_str, 'x1': future_ts_str, 'y0': y0_loss, 'y1': y1_loss,
                             'fillcolor': fill_loss, 'line': {'width': 0}})
                        shapes.append(
                            {'type': 'line', 'x0': start_ts_str, 'x1': future_ts_str, 'y0': entry, 'y1': entry,
                             'line': {'color': 'gray', 'width': 1, 'dash': 'solid'}})
                        shapes.append({'type': 'line', 'x0': start_ts_str, 'x1': future_ts_str, 'y0': sl, 'y1': sl,
                                       'line': {'color': 'red', 'width': 1, 'dash': 'dash'}})

                    m_color = '#00ff00' if t_type == "LONG" else '#ff0000';
                    m_opacity = 1.0 if draw_box else 0.5
                    traces.append({'type': 'scatter', 'x': [start_ts_str], 'y': [entry], 'mode': 'markers',
                                   'marker': {'size': 12, 'color': m_color, 'symbol': 'star', 'opacity': m_opacity},
                                   'name': f'{t_type}', 'showlegend': False})

            return json.dumps({'traces': traces, 'shapes': shapes, 'symbol': symbol},
                              cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            return "{}"


# --- WORKERS ---
class LiveBotWorker(QThread):
    update_ui_signal = pyqtSignal(str, str, str, str)
    trade_signal = pyqtSignal(dict)
    price_signal = pyqtSignal(str, float)

    def __init__(self, current_params, tg_token, tg_chat_id, show_rr):
        super().__init__()
        self.is_running = True
        self.tg_token = tg_token;
        self.tg_chat_id = tg_chat_id;
        self.show_rr = show_rr
        self.last_signals = {sym: {tf: None for tf in TIMEFRAMES} for sym in SYMBOLS}

    def update_settings(self, symbol, tf, rr, rsi, slope):
        if symbol in SYMBOL_PARAMS:
            current = SYMBOL_PARAMS[symbol].get(tf, {})
            at = current.get("at_active", False)
            updated = current.copy()
            updated.update({
                "rr": rr,
                "rsi": rsi,
                "slope": slope,
                "at_active": at,
                "use_trailing": current.get("use_trailing", False),
            })
            SYMBOL_PARAMS[symbol][tf] = updated

    def update_show_rr(self, s):
        self.show_rr = s

    def update_telegram_creds(self, t, c):
        self.tg_token = t;
        self.tg_chat_id = c

    def run(self):
        # Telegram mesajlarını asenkron yapmak için bu import gerekli
        import threading

        next_price_time = 0
        next_candle_time = 0
        while self.is_running:
            now = time.time()

            if now >= next_price_time:
                try:
                    latest_prices = TradingEngine.get_latest_prices(SYMBOLS)
                    for sym, price in latest_prices.items():
                        self.price_signal.emit(sym, price)
                except Exception as e:
                    print(f"[LIVE] Fiyat güncelleme hatası: {e}")
                next_price_time = now + 1.0

            if now >= next_candle_time:
                try:
                    # 1. TÜM VERİLERİ AYNI ANDA ÇEK (HIZ DEVRİMİ BURADA 🚀)
                    # Eskiden 7 saniye sürüyordu, şimdi 0.5 saniye sürecek.
                    bulk_data = TradingEngine.get_all_candles_parallel(SYMBOLS, TIMEFRAMES)

                    # 2. Gelen verileri işle (Bu kısım işlemci hızında akar, milisaniyeler sürer)
                    for (sym, tf), df in bulk_data.items():
                        if df.empty: continue

                        try:

                            if len(df) < 3:
                                continue
                            # Binance kline: son satır çoğunlukla oluşan (henüz kapanmamış) mumdur.
                            closed = df.iloc[-2]
                            forming = df.iloc[-1]
                            curr_price = float(closed['close'])
                            closed_ts_utc = closed['timestamp']
                            forming_ts_utc = forming['timestamp']
                            istanbul_time = closed_ts_utc + timedelta(hours=3)
                            ts_str = istanbul_time.strftime("%Y-%m-%d %H:%M")
                            # Backtest ile uyumlu fill: sinyal mumu kapandıktan sonraki mumun OPEN fiyatı
                            next_open_price = float(forming['open'])
                            next_open_ts_str = (forming_ts_utc + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
                            # Fiyatı Arayüze Gönder (Sadece 1m mumlarında veya her döngüde bir kere)
                            if tf == "1m":
                                self.price_signal.emit(sym, curr_price)

                            # --- Trade Manager Güncellemesi ---

                            closed_trades = trade_manager.update_trades(
                                sym, tf,
                                candle_high=float(closed['high']),
                                candle_low=float(closed['low']),
                                candle_close=float(closed['close']),
                                candle_time_utc=closed_ts_utc,
                                pb_top=float(closed.get('pb_ema_top', closed['close'])),
                                pb_bot=float(closed.get('pb_ema_bot', closed['close']))
                            )
                            if closed_trades:
                                for ct in closed_trades:
                                    if ct['timeframe'] == tf:
                                        reason = ct['status']
                                        pnl = float(ct['pnl'])
                                        pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
                                        icon = "✅" if "WIN" in reason else "🛑"
                                        close_log = f"🏁 {ct['symbol']} KAPANDI ({tf}): {reason} | {pnl_str} | Setup: {ct['setup']}"
                                        self.update_ui_signal.emit(sym, tf, "{}", f"⚠️ {close_log}")

                                        # Telegram (Asenkron - Beklemeden Gönder)
                                        tg_msg = (f"{icon} KAPANDI: {ct['symbol']}\nTF: {tf}\nSetup: {ct['setup']}\n"
                                                  f"Sonuç: {reason}\nNet PnL: {pnl_str}")
                                        TradingEngine.send_telegram(self.tg_token, self.tg_chat_id, tg_msg)

                            # --- İndikatör ve Sinyal Hesabı ---

                            df_ind = TradingEngine.calculate_indicators(df.copy())
                            df_closed = df_ind.iloc[:-1].copy()  # oluşan mumu çıkar
                            config = load_optimized_config(sym, tf)
                            rr, rsi, slope = config['rr'], config['rsi'], config['slope']
                            use_at = config['at_active']
                            at_status_log = "AT:ON" if use_at else "AT:OFF"

                            s_type, s_entry, s_tp, s_sl, s_reason = TradingEngine.check_signal_diagnostic(
                                df_closed,
                                index=-1,
                                min_rr=rr,
                                rsi_limit=rsi,
                                slope_thresh=slope,
                                use_alphatrend=use_at,
                                hold_n=config.get("hold_n"),
                                min_hold_frac=config.get("min_hold_frac"),
                                pb_touch_tolerance=config.get("pb_touch_tolerance"),
                                body_tolerance=config.get("body_tolerance"),
                                cloud_keltner_gap_min=config.get("cloud_keltner_gap_min"),
                                tp_min_dist_ratio=config.get("tp_min_dist_ratio"),
                                tp_max_dist_ratio=config.get("tp_max_dist_ratio"),
                                adx_min=config.get("adx_min"),
                            )

                            setup_tag = "Unknown"
                            if "ACCEPTED" in s_reason:
                                start = s_reason.find("(") + 1
                                end = s_reason.find(")")
                                if start > 0 and end > 0: setup_tag = s_reason[start:end]

                            active_trades = [t for t in trade_manager.open_trades if
                                             t['timeframe'] == tf and t['symbol'] == sym]

                            # PnL Gösterimi
                            live_pnl_str = ""
                            if active_trades:
                                t = active_trades[0]
                                current_pnl = float(t.get('pnl', 0))
                                sign = "+" if current_pnl >= 0 else "-"
                                partial_info = " (Part.Taken)" if t.get("partial_taken") else ""
                                live_pnl_str = f" | PnL: {sign}${abs(current_pnl):.2f}{partial_info}"

                            # Sinyal Yönetimi
                            if s_type and "ACCEPTED" in s_reason:
                                has_open = False
                                for t in trade_manager.open_trades:
                                    if t['symbol'] == sym and t['timeframe'] == tf: has_open = True; break

                                if has_open:
                                    log_msg = f"{tf} | {curr_price} | ⚠️ Pozisyon Var{live_pnl_str}"
                                    json_data = TradingEngine.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                     active_trades if self.show_rr else [])
                                    self.update_ui_signal.emit(sym, tf, json_data, log_msg)

                                elif self.last_signals[sym][tf] != closed_ts_utc:
                                    if trade_manager.check_cooldown(sym, tf, forming_ts_utc):
                                        log_msg = f"{tf} | {curr_price} | ❄️ SOĞUMA SÜRECİNDE"
                                        json_data = TradingEngine.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                         active_trades if self.show_rr else [])
                                        self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                    else:
                                        # YENİ İŞLEM AÇ
                                        trade_data = {
                                            "symbol": sym, "timestamp": next_open_ts_str, "open_time_utc": forming_ts_utc,
                                            "timeframe": tf, "type": s_type,
                                            "entry": next_open_price, "tp": s_tp, "sl": s_sl, "setup": setup_tag
                                        }
                                        trade_manager.open_trade(trade_data)
                                        self.trade_signal.emit(trade_data)

                                        # Telegram (Asenkron)
                                        msg = (f"🚀 SİNYAL: {s_type}\nSembol: {sym}\nTF: {tf}\nSetup: {setup_tag}\n"
                                               f"Fiyat: {next_open_price:.4f}\nTP: {s_tp:.4f}")
                                        TradingEngine.send_telegram(self.tg_token, self.tg_chat_id, msg)

                                        self.last_signals[sym][tf] = closed_ts_utc
                                        log_msg = f"{tf} | {curr_price} | 🔥 {s_type} ({setup_tag})"
                                        json_data = TradingEngine.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                         active_trades if self.show_rr else [])
                                        self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                else:
                                    log_msg = f"{tf} | {curr_price} | ⏳ İşlemde...{live_pnl_str}"
                                    json_data = TradingEngine.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                     active_trades if self.show_rr else [])
                                    self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                            else:
                                # Sinyal Yoksa Logla
                                if s_reason and "REJECT" in s_reason:
                                    log_msg = f"{tf} | {curr_price} | ⚠️ {s_reason}{live_pnl_str}"
                                else:
                                    log_msg = f"{tf} | {curr_price} | {at_status_log}{live_pnl_str}"

                                json_data = TradingEngine.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                 active_trades if self.show_rr else [])
                                self.update_ui_signal.emit(sym, tf, json_data, log_msg)

                        except Exception as e:
                            print(f"Loop Processing Error ({sym}-{tf}): {e}")
                            with open("error_log.txt", "a") as f:
                                f.write(f"\n[{datetime.now()}] LOOP HATA: {str(e)}\n")
                                f.write(traceback.format_exc())

                except Exception as e:
                    print(f"Main Loop Error: {e}")
                    time.sleep(1)  # Hata olursa 1 sn bekle, işlemciyi yakma

                next_candle_time = now + REFRESH_RATE

            time.sleep(0.1)


# --- OPTIMIZER WORKER (v35.0 - MATHEMATICALLY CORRECT R-CALC) ---
class OptimizerWorker(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, symbol, candle_limit, rr_range, rsi_range, slope_range, use_alphatrend, monte_carlo_mode=False):
        super().__init__()
        self.symbol = symbol
        self.candle_limit = candle_limit
        self.rr_range = rr_range
        self.rsi_range = rsi_range
        self.slope_range = slope_range
        self.monte_carlo_mode = monte_carlo_mode

        # --- MERKEZİ AYARLARDAN OKUMA ---
        self.slippage_rate = TRADING_CONFIG["slippage_rate"]
        self.funding_rate_8h = TRADING_CONFIG["funding_rate_8h"]
        self.total_fee = TRADING_CONFIG["total_fee"]  # Giriş + Çıkış toplam komisyon oranı
        self.leverage = TRADING_CONFIG["leverage"]

    def run(self):
        try:
            mode_text = "🎲 MONTE CARLO (RASTGELE VERİ)" if self.monte_carlo_mode else "🧠 NORMAL (GERÇEK VERİ)"
            self.result_signal.emit(f"🚀 {self.symbol} İÇİN TARAMA BAŞLADI: {mode_text}\n")

            # Trend verisi hazırlığı (Sadece normal modda)
            df_trend = pd.DataFrame()
            if not self.monte_carlo_mode:
                try:
                    df_trend = TradingEngine.get_historical_data_pagination(self.symbol, "1h", total_candles=2500)
                    if not df_trend.empty:
                        df_trend['ema_trend'] = ta.ema(df_trend['close'], length=200)
                        df_trend = df_trend[['timestamp', 'ema_trend']].dropna()
                except:
                    pass

            data_cache = {}
            for tf in TIMEFRAMES:
                self.result_signal.emit(f"⬇️ {tf} verisi hazırlanıyor...\n")
                df = TradingEngine.get_historical_data_pagination(self.symbol, tf, total_candles=self.candle_limit)

                if not df.empty:
                    # --- MONTE CARLO KARIŞTIRMA ---
                    if self.monte_carlo_mode:
                        df = df.sample(frac=1).reset_index(drop=True)
                        df = df.fillna(method='ffill').fillna(method='bfill')
                    # ------------------------------

                    df = TradingEngine.calculate_indicators(df)

                    if not df_trend.empty and not self.monte_carlo_mode:
                        df = df.sort_values('timestamp')
                        df_trend = df_trend.sort_values('timestamp')
                        df = pd.merge_asof(df, df_trend, on='timestamp', direction='backward')
                    else:
                        df['ema_trend'] = np.nan
                    data_cache[tf] = df

            # Kombinasyonlar
            rr_vals = np.arange(self.rr_range[0], self.rr_range[1] + 0.01, self.rr_range[2])
            rsi_vals = np.arange(self.rsi_range[0], self.rsi_range[1] + 1, self.rsi_range[2])
            slope_vals = np.arange(self.slope_range[0], self.slope_range[1] + 0.01, self.slope_range[2])
            at_vals = [True, False]

            combinations = list(itertools.product(rr_vals, rsi_vals, slope_vals, at_vals))
            total_combs = len(combinations)
            results_by_tf = {tf: [] for tf in TIMEFRAMES}
            TRAILING_ALLOWED_TFS = ["5m"]

            start_time = time.time()

            for idx, (rr, rsi, slope, at_active) in enumerate(combinations):
                if idx % 10 == 0:
                    progress = (idx / total_combs) * 100
                    self.result_signal.emit(f"⏳ %{progress:.1f} | Hesapla...\n")

                for tf, df in data_cache.items():
                    if df is None or df.empty: continue
                    is_trailing_active = (tf in TRAILING_ALLOWED_TFS)
                    net_r = 0
                    wins = 0
                    losses = 0

                    start_idx = 200
                    limit_idx = len(df) - 1
                    cooldown = 0

                    for i in range(start_idx, limit_idx):
                        s_type, _, s_tp_raw, s_sl_raw, s_reason = TradingEngine.check_signal_diagnostic(
                            df,
                            index=i,
                            min_rr=rr,
                            rsi_limit=rsi,
                            slope_thresh=slope,
                            use_alphatrend=at_active,
                            hold_n=DEFAULT_STRATEGY_CONFIG["hold_n"],
                            min_hold_frac=DEFAULT_STRATEGY_CONFIG["min_hold_frac"],
                            pb_touch_tolerance=DEFAULT_STRATEGY_CONFIG["pb_touch_tolerance"],
                            body_tolerance=DEFAULT_STRATEGY_CONFIG["body_tolerance"],
                            cloud_keltner_gap_min=DEFAULT_STRATEGY_CONFIG["cloud_keltner_gap_min"],
                            tp_min_dist_ratio=DEFAULT_STRATEGY_CONFIG["tp_min_dist_ratio"],
                            tp_max_dist_ratio=DEFAULT_STRATEGY_CONFIG["tp_max_dist_ratio"],
                            adx_min=DEFAULT_STRATEGY_CONFIG["adx_min"],
                        )

                        if s_type and "ACCEPTED" in s_reason:
                            if i + 1 >= len(df): break

                            next_candle = df.iloc[i + 1]
                            real_entry_price = next_candle['open']
                            entry_time = next_candle['timestamp']

                            # Slippage Uygula
                            if s_type == "LONG":
                                real_entry_price *= (1 + self.slippage_rate)
                            else:
                                real_entry_price *= (1 - self.slippage_rate)

                            if i < cooldown: continue

                            outcome = "Open"
                            sim_sl = s_sl_raw
                            sim_tp = s_tp_raw
                            has_breakeven = False
                            partial_taken = False
                            curr_size_ratio = 1.0
                            limit_j = min(i + 100, len(df))
                            exit_time = entry_time

                            for j in range(i + 1, limit_j):
                                row = df.iloc[j]
                                curr_high = row['high'];
                                curr_low = row['low'];
                                curr_close = row['close']
                                exit_time = row['timestamp']

                                if s_type == "LONG":
                                    if curr_low <= sim_sl:
                                        outcome = "WIN (Trailing)" if sim_sl > real_entry_price else (
                                            "BE" if sim_sl == real_entry_price else "LOSS")
                                        cooldown = j + (10 if tf == "1m" else 6)
                                        break
                                    if curr_high >= sim_tp: outcome = "WIN (TP)"; cooldown = j + 1; break
                                else:
                                    if curr_high >= sim_sl:
                                        outcome = "WIN (Trailing)" if sim_sl < real_entry_price else (
                                            "BE" if sim_sl == real_entry_price else "LOSS")
                                        cooldown = j + (10 if tf == "1m" else 6)
                                        break
                                    if curr_low <= sim_tp: outcome = "WIN (TP)"; cooldown = j + 1; break

                                # Trailing/Partial Logic
                                total_dist = abs(sim_tp - real_entry_price)
                                if total_dist > 0:
                                    prog = abs(curr_close - real_entry_price) / total_dist
                                    if is_trailing_active:
                                        if not has_breakeven and prog >= 0.40: sim_sl = real_entry_price; has_breakeven = True
                                    else:
                                        if not partial_taken and prog >= 0.50:
                                            net_r += (0.5 * (rr * 0.5));
                                            curr_size_ratio = 0.5;
                                            partial_taken = True;
                                            sim_sl = real_entry_price;
                                            has_breakeven = True
                                        elif not has_breakeven and prog >= 0.40:
                                            sim_sl = real_entry_price; has_breakeven = True

                            # --- MATEMATİKSEL KESİN KOMİSYON HESABI ---
                            # Risk Yüzdesi: Stop olduğumda paramın yüzde kaçı gidiyor?
                            # Örn: Giriş 100, Stop 99 ise risk %1 (0.01).
                            risk_pct = abs(real_entry_price - s_sl_raw) / real_entry_price

                            # Eğer risk %0 ise (imkansız ama önlem) 0.01 al
                            if risk_pct == 0: risk_pct = 0.01

                            # Fee Maliyeti (R cinsinden) = (Fee Oranı) / (Risk Oranı)
                            # Örn: Fee %0.07, Risk %1 ise -> Fee Maliyeti 0.07 R
                            fee_cost_r = self.total_fee / risk_pct

                            # Funding Maliyeti
                            duration_hours = 0
                            if not self.monte_carlo_mode:
                                try:
                                    duration_hours = (exit_time - entry_time).total_seconds() / 3600
                                except:
                                    duration_hours = 0

                            # Funding (R cinsinden) = (Funding Rate * Kaldıraç * Periyot) / Risk Oranı
                            funding_cost_r = ((duration_hours / 8) * self.funding_rate_8h * self.leverage) / risk_pct

                            if "WIN" in outcome:
                                reward_dist = abs(sim_tp - real_entry_price) if "TP" in outcome else abs(
                                    sim_sl - real_entry_price)
                                risk_dist = abs(real_entry_price - s_sl_raw)

                                if risk_dist > 0:
                                    raw_r = (reward_dist / risk_dist) * curr_size_ratio
                                    # Kazançtan masrafları düş
                                    net_r += (raw_r - fee_cost_r - funding_cost_r)
                                wins += 1

                            elif outcome == "LOSS":
                                loss_r = 1.0 if not partial_taken else 0
                                # Kayıpta: 1R kayıp + Fee + Funding
                                net_r += (-loss_r - fee_cost_r - funding_cost_r)
                                losses += 1

                            elif outcome == "BE":
                                # BE olsa bile Fee ödenir!
                                net_r -= (fee_cost_r + funding_cost_r)

                    results_by_tf[tf].append(
                        {"RR": rr, "RSI": rsi, "Slope": slope, "AT": at_active, "Wins": wins, "Losses": losses,
                         "Net_R": net_r})

            elapsed = time.time() - start_time
            self.result_signal.emit(f"\n✅ İŞLEM BİTTİ ({elapsed:.1f}sn)\n{'=' * 40}\n")

            for tf in TIMEFRAMES:
                data_list = results_by_tf.get(tf, [])
                if not data_list:
                    self.result_signal.emit(f"[{tf}] ⚠️ VERİ YOK\n")
                    continue

                res = sorted(data_list, key=lambda x: x['Net_R'], reverse=True)[:1]

                if res:
                    r = res[0]
                    tot = r['Wins'] + r['Losses']
                    wr = (r['Wins'] / tot * 100) if tot > 0 else 0.0
                    prefix = "🎲 MC SONUCU" if self.monte_carlo_mode else "🧠 REALITY SONUCU"

                    if tot == 0:
                        self.result_signal.emit(f"[{tf}] {prefix}: 🛡️ GÜVENLİ: HİÇ İŞLEM AÇILMADI (0 Trade)\n\n")
                    else:
                        self.result_signal.emit(f"[{tf}] {prefix}: NET R: {r['Net_R']:.2f} | WR: %{wr:.1f}\n")
                        self.result_signal.emit(
                            f"   AYAR: RR={r['RR']:.1f}, RSI={r['RSI']}, AT={'AÇIK' if r['AT'] else 'KAPALI'}\n\n")

        except Exception as e:
            self.result_signal.emit(f"❌ BEKLENMEYEN HATA: {str(e)}\n")
            print(e)


# --- 🌙 AUTO BACKTEST WORKER (v30.7 - TAM EŞİTLİK & DETAYLI RAPOR) ---
class AutoBacktestWorker(QThread):
    def __init__(self):
        super().__init__()
        self.is_running = True
        self.force_run = False

    def run(self):
        print("🌙 Gece Bekçisi Devrede... (03:00 Bekleniyor)")
        while self.is_running:
            now = datetime.now()
            # Tetikleyici: 03:00 veya Manuel Buton
            if (now.hour == 3 and now.minute == 0 and now.second < 10) or self.force_run:
                print("🚀 Otomatik Tarama Başlatılıyor... (Bu işlem zaman alabilir)")
                self.run_full_analysis()
                self.force_run = False
                time.sleep(65)
            time.sleep(5)

    def run_full_analysis(self):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        report_lines = [
            f"--- GÜNLÜK BACKTEST RAPORU ({timestamp}) ---",
            f"Gerçekçi Mod: %{TRADING_CONFIG['slippage_rate'] * 100} Slippage, %{TRADING_CONFIG['total_fee'] * 100} Fee",
            "BTC/ETH/SOL | TF: 1m,5m,15m,1h | Mum: 15000",
            "",
        ]

        try:
            max_daily_candles = max(DAILY_REPORT_CANDLE_LIMITS.values())
            print("[AUTO] Günlük backtest başlıyor (15k mum)")
            result = run_portfolio_backtest(
                symbols=SYMBOLS,
                timeframes=[tf for tf in TIMEFRAMES if tf in DAILY_REPORT_CANDLE_LIMITS],
                candles=max_daily_candles,
                out_trades_csv="daily_report_trades.csv",
                out_summary_csv="daily_report_summary.csv",
                limit_map=DAILY_REPORT_CANDLE_LIMITS,
                draw_trades=False,
            ) or {}

            summary_rows = result.get("summary", []) if isinstance(result, dict) else []
            best_configs = result.get("best_configs", {}) if isinstance(result, dict) else {}

            if summary_rows:
                report_lines.append("Özet Tablosu:")
                for row in summary_rows:
                    report_lines.append(
                        f"- {row['symbol']}-{row['timeframe']}: Trades={row['trades']}, WR={row['win_rate_pct']:.1f}%, NetPnL={row['net_pnl']:.2f}"
                    )
                report_lines.append("")
            else:
                report_lines.append("⚠️ Veri bulunamadı veya backtest başarısız.")

            if best_configs:
                save_best_configs(best_configs)
                report_lines.append("En İyi Ayarlar (Net PnL'e göre):")
                for (sym, tf), cfg in sorted(best_configs.items()):
                    report_lines.append(
                        f"- {sym}-{tf}: RR={cfg['rr']}, RSI={cfg['rsi']}, Slope={cfg['slope']}, AT={'Açık' if cfg.get('at_active') else 'Kapalı'}, Trailing={cfg.get('use_trailing', False)} | NetPnL={cfg.get('_net_pnl', 0):.2f}, Trades={cfg.get('_trades', 0)}"
                    )
                report_lines.append("")

        except Exception as e:
            err_msg = f"Rapor hatası: {e}"
            print(err_msg)
            report_lines.append(err_msg)

        # --- DOSYA KAYDETME ---
        try:
            report_dir = os.path.join(os.getcwd(), "raporlar")
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)

            file_name = f"Rapor_{datetime.now().strftime('%Y-%m-%d_%H%M')}.txt"
            file_path = os.path.join(report_dir, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            print(f"✅ Rapor Kaydedildi: {file_path}")

            with open(CONFIG_FILE, 'r') as f:
                c = json.load(f)
                TradingEngine.send_telegram(c.get("telegram_token"), c.get("telegram_chat_id"),
                                            f"🌙 Rapor Hazır: {file_name}")
        except Exception as e:
            print(f"Rapor hatası: {e}")


class BacktestWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, symbols, timeframes, candles):
        super().__init__()
        self.symbols = symbols
        self.timeframes = timeframes
        self.candles = candles

    def run(self):
        result = {}

        try:
            result = run_portfolio_backtest(
                symbols=self.symbols,
                timeframes=self.timeframes,
                candles=self.candles,
                progress_callback=self.log_signal.emit,
                draw_trades=True,
                max_draw_trades=30,
            ) or {}
        except Exception as e:
            self.log_signal.emit(f"\n[BACKTEST][GUI] Hata: {e}\n{traceback.format_exc()}\n")
        finally:
            self.finished_signal.emit(result if isinstance(result, dict) else {})


# --- GUI (ANA PENCERE) ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SSL PB Sniper - v30.4 (PROFIT ENGINE)")
        self.setGeometry(100, 100, 1600, 1000)
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background-color: #2d2d2d; color: #cccccc; padding: 10px; font-weight: bold; }
            QTabBar::tab:selected { background-color: #007acc; color: #ffffff; }
            QTextEdit { background-color: #0a0a0a; color: #00ff00; border: 1px solid #333; font-family: Consolas; }
            QTableWidget { background-color: #1e1e1e; color: white; gridline-color: #333; }
            QHeaderView::section { background-color: #333; color: white; padding: 5px; font-weight: bold; }
            QLabel { color: white; font-weight: bold; }
            QPushButton { background-color: #007acc; color: white; padding: 8px; border: none; border-radius: 4px; }
            QPushButton:hover { background-color: #005f9e; }
            QGroupBox { border: 1px solid #444; margin-top: 10px; }
            QGroupBox::title { color: white; padding: 0 3px; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background-color: #333; color: white; padding: 5px; border: 1px solid #555; }
            QCheckBox { color: white; }
        """)

        self.tg_token = "";
        self.tg_chat_id = ""
        self.load_config()
        self.views_ready = {}
        self.show_rr_tools = True
        self.data_cache = {sym: {tf: (None, None) for tf in TIMEFRAMES} for sym in SYMBOLS}
        self.current_symbol = SYMBOLS[0]
        self.backtest_worker = None
        self.backtest_meta = None

        central = QWidget();
        self.setCentralWidget(central);
        main_layout = QVBoxLayout(central)

        # --- CANLI FİYAT TABELASI (TICKER) ---
        ticker_widget = QWidget()
        ticker_widget.setStyleSheet("background-color: #1a1a1a; border-bottom: 2px solid #00ccff;")
        ticker_layout = QHBoxLayout(ticker_widget)
        ticker_layout.setContentsMargins(10, 5, 10, 5)

        self.price_labels = {}
        for sym in SYMBOLS:
            lbl_name = QLabel(sym.replace("USDT", ""))
            lbl_name.setStyleSheet("color: #888; font-weight: bold; font-size: 14px;")

            lbl_price = QLabel("---")
            lbl_price.setStyleSheet("color: #00ccff; font-weight: bold; font-size: 16px;")
            self.price_labels[sym] = lbl_price

            ticker_layout.addWidget(lbl_name)
            ticker_layout.addWidget(lbl_price)
            ticker_layout.addSpacing(20)

        ticker_layout.addStretch()
        main_layout.addWidget(ticker_widget)
        # ---------------------------------------------

        self.main_tabs = QTabWidget();
        main_layout.addWidget(self.main_tabs)

        # 1. SEKME: Dashboard
        live_widget = QWidget();
        live_layout = QVBoxLayout(live_widget)
        top_panel = QHBoxLayout()

        coin_group = QGroupBox("Takip");
        coin_layout = QHBoxLayout()
        self.combo_symbol = QComboBox();
        self.combo_symbol.addItems(SYMBOLS)
        self.combo_symbol.currentTextChanged.connect(self.on_symbol_changed)
        self.combo_symbol.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px; color: #00ccff;")
        coin_layout.addWidget(self.combo_symbol);
        coin_group.setLayout(coin_layout);
        top_panel.addWidget(coin_group, stretch=1)

        tg_group = QGroupBox("Telegram");
        tg_layout = QHBoxLayout()
        self.txt_token = QLineEdit(self.tg_token);
        tg_layout.addWidget(QLabel("T:"));
        tg_layout.addWidget(self.txt_token)
        self.txt_chatid = QLineEdit(self.tg_chat_id);
        tg_layout.addWidget(QLabel("ID:"));
        tg_layout.addWidget(self.txt_chatid)
        btn_save_tg = QPushButton("Kaydet");
        btn_save_tg.clicked.connect(self.save_config);
        tg_layout.addWidget(btn_save_tg)
        tg_group.setLayout(tg_layout);
        top_panel.addWidget(tg_group, stretch=2)

        settings_group = QGroupBox("Manuel Ayar");
        sets_layout = QHBoxLayout()
        self.combo_tf = QComboBox();
        self.combo_tf.addItems(TIMEFRAMES);
        self.combo_tf.currentTextChanged.connect(self.load_tf_settings)
        sets_layout.addWidget(QLabel("TF:"));
        sets_layout.addWidget(self.combo_tf)
        self.spin_rr = QDoubleSpinBox();
        self.spin_rr.setRange(0.1, 10.0);
        self.spin_rr.setSingleStep(0.1)
        sets_layout.addWidget(QLabel("RR:"));
        sets_layout.addWidget(self.spin_rr)
        self.spin_rsi = QSpinBox();
        self.spin_rsi.setRange(10, 90);
        self.spin_rsi.setSingleStep(5)
        sets_layout.addWidget(QLabel("RSI:"));
        sets_layout.addWidget(self.spin_rsi)
        self.spin_slope = QDoubleSpinBox();
        self.spin_slope.setRange(0.1, 5.0);
        self.spin_slope.setSingleStep(0.1)
        sets_layout.addWidget(QLabel("Slope:"));
        sets_layout.addWidget(self.spin_slope)
        btn_apply = QPushButton("Güncelle");
        btn_apply.clicked.connect(self.apply_settings);
        sets_layout.addWidget(btn_apply)
        self.chk_rr = QCheckBox("RR");
        self.chk_rr.setChecked(True);
        self.chk_rr.stateChanged.connect(self.toggle_rr)
        sets_layout.addWidget(self.chk_rr);
        settings_group.setLayout(sets_layout);
        top_panel.addWidget(settings_group, stretch=4)
        live_layout.addLayout(top_panel)

        chart_container = QWidget();
        grid_charts = QGridLayout(chart_container);
        grid_charts.setContentsMargins(0, 0, 0, 0);
        grid_charts.setSpacing(5)
        self.web_views = {}
        positions = {"1m": (0, 0), "5m": (0, 1), "15m": (0, 2), "1h": (1, 0), "4h": (1, 1)}
        for tf in TIMEFRAMES:
            box = QGroupBox(f"{tf} Grafiği");
            box.setStyleSheet("QGroupBox { border: 1px solid #333; font-weight: bold; color: #00ccff; }")
            box_layout = QVBoxLayout(box);
            box_layout.setContentsMargins(0, 15, 0, 0)
            view = QWebEngineView();
            view.setHtml(CHART_TEMPLATE)
            view.loadFinished.connect(lambda ok, t=tf: self.on_load_finished(ok, t))
            box_layout.addWidget(view);
            self.web_views[tf] = view
            r, c = positions[tf];
            grid_charts.addWidget(box, r, c)
        live_layout.addWidget(chart_container, stretch=6)
        self.logs = QTextEdit();
        self.logs.setReadOnly(True);
        self.logs.setMaximumHeight(100)
        live_layout.addWidget(self.logs, stretch=1);
        self.main_tabs.addTab(live_widget, "📡 Dashboard")

        # 2. SEKME: Açık İşlemler
        open_trades_widget = QWidget();
        ot_layout = QVBoxLayout(open_trades_widget)
        ot_group = QGroupBox("Açık İşlemler");
        ot_in = QVBoxLayout()
        self.open_trades_table = QTableWidget();

        # Sütun Sayısı 11 (Size Eklendi)
        self.open_trades_table.setColumnCount(11)
        self.open_trades_table.setHorizontalHeaderLabels(
            ["Zaman", "Coin", "TF", "Yön", "Setup", "Giriş", "TP", "SL", "Büyüklük", "PnL", "Durum"])
        self.open_trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        ot_in.addWidget(self.open_trades_table);
        ot_group.setLayout(ot_in);
        ot_layout.addWidget(ot_group)
        self.main_tabs.addTab(open_trades_widget, "⚡ İşlemler")

        # 3. SEKME: Geçmiş & Varlıklarım
        history_widget = QWidget();
        hist_layout = QVBoxLayout(history_widget)

        # --- YENİ VARLIK PANELİ (ASSET WIDGET) ---
        asset_group = QGroupBox("Varlıklarım & Performans");
        asset_group.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 14px; border: 1px solid #444; margin-top: 10px; } QGroupBox::title { color: #00ccff; }")
        asset_layout = QGridLayout()

        # 4 Kutu Tasarımı
        def create_stat_box(title, val_id):
            box = QFrame();
            box.setStyleSheet("background-color: #222; border-radius: 5px; padding: 10px;")
            l = QVBoxLayout(box)
            lbl_t = QLabel(title);
            lbl_t.setStyleSheet("color: #aaa; font-size: 12px;")
            lbl_v = QLabel("$0.00");
            lbl_v.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
            l.addWidget(lbl_t);
            l.addWidget(lbl_v)
            return box, lbl_v

        # Kutuları Oluştur ve Referansları Kaydet
        box1, self.lbl_equity_val = create_stat_box("Toplam Varlık (Equity)", "lbl_equity")
        box2, self.lbl_avail_val = create_stat_box("Kullanılabilir Bakiye", "lbl_avail")
        box3, self.lbl_total_pnl_val = create_stat_box("Toplam Kâr/Zarar (Total PnL)", "lbl_total_pnl")
        box4, self.lbl_daily_pnl_val = create_stat_box("Bugünün Kârı (Daily PnL)", "lbl_daily_pnl")

        asset_layout.addWidget(box1, 0, 0)
        asset_layout.addWidget(box2, 0, 1)
        asset_layout.addWidget(box3, 1, 0)
        asset_layout.addWidget(box4, 1, 1)

        asset_group.setLayout(asset_layout)
        hist_layout.addWidget(asset_group)

        # Portföy tablosu (canlı işlemlerle senkron)
        portfolio_group = QGroupBox("Portföy Durumu")
        port_layout = QVBoxLayout()
        self.portfolio_table = QTableWidget()
        self.portfolio_table.setColumnCount(9)
        self.portfolio_table.setHorizontalHeaderLabels([
            "Sembol", "TF", "Yön", "Giriş", "TP", "SL", "Kilitli Marj", "Poz. Büyüklüğü", "Anlık PnL"
        ])
        self.portfolio_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        port_layout.addWidget(self.portfolio_table)
        portfolio_group.setLayout(port_layout)
        hist_layout.addWidget(portfolio_group)

        # Tabloyu oluştur
        self.pnl_table = self.create_pnl_table()

        # Tabloyu istatistik grubunun hemen altına ekle
        hist_layout.addWidget(self.pnl_table)
        # -----------------------------------------

        # Geçmiş Tablosu (Aynı Kalıyor)
        hist_group = QGroupBox("Geçmiş İşlemler");
        hist_in = QVBoxLayout()
        self.history_table = QTableWidget();
        self.history_table.setColumnCount(11)
        self.history_table.setHorizontalHeaderLabels(
            ["Açılış", "Kapanış", "Coin", "TF", "Yön", "Setup", "Giriş", "Çıkış", "Sonuç", "PnL", "Kasa"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        hist_in.addWidget(self.history_table)

        # Butonlar
        btn_layout = QHBoxLayout()
        btn_reset_bal = QPushButton("Bakiyeyi Sıfırla ($2000)");
        btn_reset_bal.clicked.connect(self.reset_balances)
        btn_reset_logs = QPushButton("Tüm Geçmişi Temizle");
        btn_reset_logs.clicked.connect(self.reset_logs)
        btn_layout.addWidget(btn_reset_bal);
        btn_layout.addWidget(btn_reset_logs)
        hist_in.addLayout(btn_layout)

        hist_group.setLayout(hist_in);
        hist_layout.addWidget(hist_group)
        self.main_tabs.addTab(history_widget, "📜 Geçmiş & Varlık")

        # 4. SEKME: Backtest
        backtest_widget = QWidget();
        backtest_layout = QVBoxLayout(backtest_widget)
        bt_cfg = QHBoxLayout()
        bt_cfg.addWidget(QLabel("Semboller:"));
        bt_cfg.addWidget(QLabel(", ".join(SYMBOLS)))
        bt_cfg.addWidget(QLabel("TF:"));
        bt_cfg.addWidget(QLabel(", ".join(TIMEFRAMES)))
        bt_cfg.addWidget(QLabel("Mum Sayısı:"));
        self.backtest_candles = QSpinBox();
        self.backtest_candles.setRange(500, 6000);
        self.backtest_candles.setValue(3000);
        bt_cfg.addWidget(self.backtest_candles)
        self.btn_run_backtest = QPushButton("🧪 Backtest Çalıştır");
        self.btn_run_backtest.clicked.connect(self.start_backtest);
        bt_cfg.addWidget(self.btn_run_backtest)
        backtest_layout.addLayout(bt_cfg)

        self.backtest_logs = QTextEdit();
        self.backtest_logs.setReadOnly(True);
        backtest_layout.addWidget(self.backtest_logs)
        self.main_tabs.addTab(backtest_widget, "🧪 Backtest")

        # 5. SEKME: Optimizasyon (Temizlendi & Otomatikleştirildi)
        opt_widget = QWidget();
        opt_layout = QVBoxLayout(opt_widget)
        grid_group = QGroupBox("Parametre Aralıkları");
        grid_layout = QHBoxLayout()
        # --- YENİ: OTOMATİK RAPOR TEST BUTONU ---
        btn_test_report = QPushButton("🌙 GÜNLÜK RAPORU ŞİMDİ OLUŞTUR (TEST)");
        btn_test_report.setStyleSheet("background-color: #444; color: #aaa; margin-top: 10px;")
        btn_test_report.clicked.connect(self.force_daily_report)
        opt_layout.addWidget(btn_test_report)
        # ----------------------------------------

        # RR Ayarları
        grid_layout.addWidget(QLabel("RR:"));
        self.opt_rr_start = QDoubleSpinBox();
        self.opt_rr_start.setValue(1.5);
        self.opt_rr_end = QDoubleSpinBox();
        self.opt_rr_end.setValue(3.0);
        self.opt_rr_step = QDoubleSpinBox();
        self.opt_rr_step.setValue(0.5)
        grid_layout.addWidget(self.opt_rr_start);
        grid_layout.addWidget(self.opt_rr_end);
        grid_layout.addWidget(self.opt_rr_step)

        # RSI Ayarları
        grid_layout.addWidget(QLabel("RSI:"));
        self.opt_rsi_start = QSpinBox();
        self.opt_rsi_start.setValue(30);
        self.opt_rsi_end = QSpinBox();
        self.opt_rsi_end.setValue(60);
        self.opt_rsi_step = QSpinBox();
        self.opt_rsi_step.setValue(10)
        grid_layout.addWidget(self.opt_rsi_start);
        grid_layout.addWidget(self.opt_rsi_end);
        grid_layout.addWidget(self.opt_rsi_step)

        # Slope Ayarları
        grid_layout.addWidget(QLabel("Slope:"));
        self.opt_slope_start = QDoubleSpinBox();
        self.opt_slope_start.setValue(0.3);
        self.opt_slope_end = QDoubleSpinBox();
        self.opt_slope_end.setValue(0.9);
        self.opt_slope_step = QDoubleSpinBox();
        self.opt_slope_step.setValue(0.2)
        grid_layout.addWidget(self.opt_slope_start);
        grid_layout.addWidget(self.opt_slope_end);
        grid_layout.addWidget(self.opt_slope_step)

        grid_group.setLayout(grid_layout);
        opt_layout.addWidget(grid_group)

        # Alt Panel (Checkbox kaldırıldı)
        candles_layout = QHBoxLayout()
        candles_layout.addWidget(QLabel("Coin:"));
        self.combo_opt_symbol = QComboBox();
        self.combo_opt_symbol.addItems(SYMBOLS);
        candles_layout.addWidget(self.combo_opt_symbol)
        candles_layout.addWidget(QLabel("Mum Sayısı:"));
        self.opt_candles = QSpinBox();
        self.opt_candles.setRange(1000, 20000);
        self.opt_candles.setValue(3500);
        candles_layout.addWidget(self.opt_candles)
        self.chk_monte_carlo = QCheckBox("🎲 Monte Carlo Testi (Random Walk)")
        self.chk_monte_carlo.setStyleSheet("color: #ff9900; font-weight: bold;")
        candles_layout.addWidget(self.chk_monte_carlo)
        # ----------------------------

        self.btn_run_opt = QPushButton("🚀 TAM TARAMA BAŞLAT");
        self.btn_run_opt.clicked.connect(self.run_optimization);
        candles_layout.addWidget(self.btn_run_opt)
        opt_layout.addLayout(candles_layout)

        self.opt_logs = QTextEdit();
        self.opt_logs.setReadOnly(True);
        opt_layout.addWidget(self.opt_logs)
        self.main_tabs.addTab(opt_widget, "🔧 Optimizasyon")

        # Açılışta canlı takip sekmesini öne çıkar
        self.main_tabs.setCurrentWidget(live_widget)

        # BAŞLATMA
        self.current_params = {}
        self.live_worker = LiveBotWorker(self.current_params, self.tg_token, self.tg_chat_id, self.show_rr_tools)
        self.live_worker.update_ui_signal.connect(self.update_ui)
        self.live_worker.price_signal.connect(self.on_price_update)
        self.live_worker.start()
        self.logs.append(">>> Sistem Başlatıldı. v30.4 (PROFIT ENGINE)")
        self.load_tf_settings("1m")
        self.table_timer = QTimer();
        self.table_timer.timeout.connect(self.refresh_trade_table_from_manager);
        self.table_timer.start(1000)
        # --- OTO BACKTEST BAŞLAT ---
        self.auto_backtest = AutoBacktestWorker()
        self.auto_backtest.start()
        # ---------------------------

        self.logs.append(">>> Sistem Başlatıldı. v30.6 (Auto Report)")

        # Backtest geçmişini göster
        self.load_backtest_meta()
        self.show_saved_backtest_summary()

    def on_load_finished(self, ok, tf):
        if ok: self.views_ready[tf] = True

    def on_symbol_changed(self, text):
        self.current_symbol = text;
        self.logs.clear();
        self.logs.append(f">>> {text} Seçildi")
        self.load_tf_settings(self.combo_tf.currentText())
        if text in self.data_cache:
            for tf in TIMEFRAMES:
                cached_data = self.data_cache[text][tf]
                if cached_data[0] is not None: self.render_chart_and_log(tf, cached_data[0], cached_data[1])

    def update_ui(self, symbol, tf, json_data, log_msg):
        self.data_cache[symbol][tf] = (json_data, log_msg)
        if symbol == self.current_symbol: self.render_chart_and_log(tf, json_data, log_msg)

    def render_chart_and_log(self, tf, json_data, log_msg):
        if self.views_ready.get(tf, False) and json_data and json_data != "{}":
            safe_json = json_data.replace("'", "\\'").replace("\\", "\\\\")
            js = f"if(window.updateChartData) window.updateChartData('{safe_json}');"
            self.web_views[tf].page().runJavaScript(js)
        if log_msg:
            if "🔥" in log_msg:
                fmt = f"<span style='color:#00ff00; font-weight:bold'>{log_msg}</span>"
            elif "⚠️" in log_msg:
                fmt = f"<span style='color:orange'>{log_msg}</span>"
            elif "Trend" in log_msg:
                fmt = f"<span style='color:#888888'>{log_msg}</span>"
            else:
                fmt = f"<span style='color:white'>{log_msg}</span>"
            self.logs.append(fmt);
            self.logs.verticalScrollBar().setValue(self.logs.verticalScrollBar().maximum())

    # --- FİYAT GÜNCELLEME (Ticker) ---
    def on_price_update(self, symbol, price):
        if symbol in self.price_labels:
            self.price_labels[symbol].setText(f"${price:.2f}")
            self.price_labels[symbol].setStyleSheet("color: #00ff00; font-weight: bold; font-size: 16px;")

    def refresh_trade_table_from_manager(self):
        try:
            open_trades = list(trade_manager.open_trades)
            # --- 1. GÜNCEL KASA VERİLERİNİ ÇEK ---
            wallet_bal = trade_manager.wallet_balance  # Kullanılabilir
            locked = trade_manager.locked_margin  # İşlemdeki
            open_pnl = sum(float(t.get("pnl", 0)) for t in open_trades)
            total_equity = wallet_bal + locked + open_pnl  # Toplam Varlık + açık pozisyon PnL'i
            total_pnl = trade_manager.total_pnl  # Toplam Net Kâr (Komisyon düşülmüş)

            # Günlük PnL Hesapla (Bugün kapanan işlemler)
            today_str = datetime.now().strftime("%Y-%m-%d")
            daily_pnl = 0.0
            for t in trade_manager.history:
                if t.get("close_time", "").startswith(today_str):
                    daily_pnl += float(t["pnl"])

            # --- 2. ETİKETLERİ GÜNCELLE ---
            if hasattr(self, 'lbl_equity_val'):
                self.lbl_equity_val.setText(f"${total_equity:,.2f}")
                self.lbl_avail_val.setText(f"${wallet_bal:,.2f}")

                self.lbl_total_pnl_val.setText(f"${total_pnl:,.2f}")
                if total_pnl > 0:
                    self.lbl_total_pnl_val.setStyleSheet("color: #00ff00; font-size: 16px; font-weight: bold;")
                elif total_pnl < 0:
                    self.lbl_total_pnl_val.setStyleSheet("color: #ff5555; font-size: 16px; font-weight: bold;")
                else:
                    self.lbl_total_pnl_val.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")

                self.lbl_daily_pnl_val.setText(f"${daily_pnl:,.2f}")
                if daily_pnl > 0:
                    self.lbl_daily_pnl_val.setStyleSheet("color: #00ff00; font-size: 16px; font-weight: bold;")
                elif daily_pnl < 0:
                    self.lbl_daily_pnl_val.setStyleSheet("color: #ff5555; font-size: 16px; font-weight: bold;")
                else:
                    self.lbl_daily_pnl_val.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")

            # --- 3. AÇIK İŞLEMLER TABLOSU ---
            open_trades.sort(key=lambda x: x['id'], reverse=True)
            self.open_trades_table.setRowCount(len(open_trades))
            cols_open = ["timestamp", "symbol", "timeframe", "type", "setup", "entry", "tp", "sl", "size", "pnl",
                         "status"]

            for row_idx, trade in enumerate(open_trades):
                for col_idx, col_key in enumerate(cols_open):
                    val = trade.get(col_key, "")
                    item = QTableWidgetItem(str(val))

                    if col_key == "size":
                        is_partial = trade.get("partial_taken", False)
                        if is_partial:
                            item.setText(f"📉 ${float(val):,.0f} (Yarım)")
                            item.setForeground(QColor("orange"))
                        else:
                            item.setText(f"${float(val):,.0f} (Tam)")
                            item.setForeground(QColor("yellow"))
                        item.setFont(QFont("Arial", 10, QFont.Bold))
                    elif col_key == "pnl":
                        pnl = float(val)
                        item.setText(f"${pnl:.2f}")
                        if pnl > 0:
                            item.setBackground(QColor(0, 100, 0))
                        else:
                            item.setBackground(QColor(100, 0, 0))
                        item.setForeground(QColor("white"))
                        item.setFont(QFont("Arial", 10, QFont.Bold))
                    elif isinstance(val, float):
                        item.setText(f"{val:.4f}")
                    if col_key == "type":
                        item.setForeground(QColor("#00ff00") if val == "LONG" else QColor("#ff0000"))
                        item.setFont(QFont("Arial", 10, QFont.Bold))
                    self.open_trades_table.setItem(row_idx, col_idx, item)

            # Portföy tablosunu güncelle
            self.update_portfolio_table(open_trades)

            # --- 4. GEÇMİŞ İŞLEMLER TABLOSU (BE GÜNCELLEMESİ EKLENDİ) ---
            hist_trades = list(trade_manager.history)
            hist_trades.sort(key=lambda x: x['close_time'], reverse=True)
            self.history_table.setRowCount(len(hist_trades))
            cols_hist = ["timestamp", "close_time", "symbol", "timeframe", "type", "setup", "entry", "close_price",
                         "status", "pnl", "has_cash"]

            for row_idx, trade in enumerate(hist_trades):
                for col_idx, col_key in enumerate(cols_hist):
                    val = trade.get(col_key, "")
                    item = QTableWidgetItem(str(val))

                    # -- DURUM (STATUS) RENKLENDİRME --
                    if col_key == "status":
                        pnl_val = float(trade.get("pnl", 0))

                        # STOP olmuş ama PnL >= -0.5 ise "BE (Başabaş)" olarak göster
                        if "STOP" in str(val):
                            if pnl_val >= -0.5:
                                item.setText("BE (Başabaş)")
                                item.setBackground(QColor(50, 50, 0))  # Sarımsı Arka Plan
                                item.setForeground(QColor("yellow"))
                            else:
                                item.setBackground(QColor(50, 0, 0))  # Kırmızı Arka Plan
                        elif "WIN" in str(val):
                            item.setBackground(QColor(0, 50, 0))  # Yeşil Arka Plan

                    # -- PNL RENKLENDİRME --
                    elif col_key == "pnl":
                        pnl = float(val)
                        item.setText(f"${pnl:.2f}")
                        if pnl > 0:
                            item.setForeground(QColor("#00ff00"))
                        else:
                            item.setForeground(QColor("#ff5555"))

                    elif col_key in ["entry", "close_price"]:
                        item.setText(f"{float(val):.4f}")

                    self.history_table.setItem(row_idx, col_idx, item)

            # --- 5. İSTATİSTİK TABLOSUNU GÜNCELLE ---
            self.update_pnl_table_data()


        except Exception as e:
            print(f"KRİTİK HATA: {e}")
            # Hatayı dosyaya yaz (Log tut)
            with open("error_log.txt", "a") as f:
                f.write(f"\n[{datetime.now()}] HATA: {str(e)}\n")
                f.write(traceback.format_exc())  # Hatanın hangi satırda olduğunu yazar

    def update_portfolio_table(self, open_trades):
        if not hasattr(self, "portfolio_table"):
            return

        self.portfolio_table.setRowCount(len(open_trades))
        cols = ["symbol", "timeframe", "type", "entry", "tp", "sl", "margin", "size", "pnl"]

        for row_idx, trade in enumerate(open_trades):
            for col_idx, key in enumerate(cols):
                val = trade.get(key, "")

                if key in {"entry", "tp", "sl"}:
                    display = f"{float(val):.4f}" if val != "" else "-"
                elif key == "margin":
                    display = f"${float(val):,.2f}"
                elif key == "size":
                    display = f"${float(val):,.2f}"
                elif key == "pnl":
                    pnl_val = float(val)
                    display = f"${pnl_val:,.2f}"
                else:
                    display = str(val)

                item = QTableWidgetItem(display)

                if key == "type":
                    item.setForeground(QColor("#00ff00") if val == "LONG" else QColor("#ff5555"))
                    item.setFont(QFont("Arial", 10, QFont.Bold))
                elif key == "pnl":
                    pnl_val = float(val)
                    if pnl_val > 0:
                        item.setForeground(QColor("#00ff00"))
                    elif pnl_val < 0:
                        item.setForeground(QColor("#ff5555"))
                elif key == "margin":
                    item.setForeground(QColor("#00ccff"))

                self.portfolio_table.setItem(row_idx, col_idx, item)

    def save_config(self):
        self.tg_token = self.txt_token.text().strip();
        self.tg_chat_id = self.txt_chatid.text().strip()
        with open(CONFIG_FILE, 'w') as f: json.dump(
            {"telegram_token": self.tg_token, "telegram_chat_id": self.tg_chat_id}, f)
        self.live_worker.update_telegram_creds(self.tg_token, self.tg_chat_id)
        QMessageBox.information(self, "Başarılı", "Kaydedildi.")

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    c = json.load(f); self.tg_token = c.get("telegram_token", ""); self.tg_chat_id = c.get(
                        "telegram_chat_id", "")
            except Exception as e:
                print(f"KRİTİK HATA: {e}")
                # Hatayı dosyaya yaz (Log tut)
                with open("error_log.txt", "a") as f:
                    f.write(f"\n[{datetime.now()}] HATA: {str(e)}\n")
                    f.write(traceback.format_exc())  # Hatanın hangi satırda olduğunu yazar

    def load_tf_settings(self, tf):
        current_sym = self.combo_symbol.currentText();
        settings = SYMBOL_PARAMS.get(current_sym, SYMBOL_PARAMS["BTCUSDT"]);
        p = settings.get(tf, settings.get("1m"))
        if p: self.spin_rr.setValue(p['rr']); self.spin_rsi.setValue(p['rsi']); self.spin_slope.setValue(p['slope'])

    def apply_settings(self):
        current_sym = self.combo_symbol.currentText();
        tf = self.combo_tf.currentText()
        rr = self.spin_rr.value();
        rsi = self.spin_rsi.value();
        slope = self.spin_slope.value()
        self.live_worker.update_settings(current_sym, tf, rr, rsi, slope)
        self.logs.append(f">>> {current_sym} - {tf} Ayarı Güncellendi.")

    def toggle_rr(self):
        self.live_worker.update_show_rr(self.chk_rr.isChecked())

    def reset_logs(self):
        if QMessageBox.question(self, 'Onay', "Sil?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes: trade_manager.reset_logs(); self.refresh_trade_table_from_manager()

    def reset_balances(self):
        if QMessageBox.question(self, 'Onay', "Sıfırla?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes: trade_manager.reset_balances(); self.refresh_trade_table_from_manager()

    def start_backtest(self):
        if getattr(self, "backtest_worker", None) and self.backtest_worker.isRunning():
            QMessageBox.information(self, "Devam Ediyor", "Backtest zaten çalışıyor...")
            return

        # Önceki sonucu koru ve göster
        self.load_backtest_meta()
        previous_lines = self.format_backtest_summary_lines()

        self.backtest_logs.clear()
        if previous_lines:
            self.backtest_logs.append("🗂️ Önceki sonuç:")
            for line in previous_lines:
                self.backtest_logs.append(line)
            self.backtest_logs.append("-" * 40)
        else:
            self.backtest_logs.append("ℹ️ Önceki backtest kaydı bulunamadı.")

        self.backtest_logs.append("🧪 Backtest başlatıldı. Lütfen bekleyin...")
        candles = self.backtest_candles.value()

        self.backtest_worker = BacktestWorker(SYMBOLS, TIMEFRAMES, candles)
        self.backtest_worker.log_signal.connect(self.append_backtest_log)
        self.backtest_worker.finished_signal.connect(self.on_backtest_finished)
        self.btn_run_backtest.setEnabled(False)
        self.backtest_worker.start()

    def append_backtest_log(self, text):
        self.backtest_logs.append(text)

    def on_backtest_finished(self, result: dict):
        self.btn_run_backtest.setEnabled(True)
        best_configs = result.get("best_configs", {}) if isinstance(result, dict) else {}
        summary_rows = result.get("summary", []) if isinstance(result, dict) else []

        if summary_rows:
            finished_at = datetime.utcnow().isoformat() + "Z"
            meta = {
                "finished_at": finished_at,
                "summary": summary_rows,
                "summary_csv": result.get("summary_csv"),
            }
            self.save_backtest_meta(meta)
            self.backtest_logs.append("📊 Özet tablo kaydedildi:")
            for line in self.format_backtest_summary_lines(meta):
                self.backtest_logs.append(line)
        else:
            self.backtest_logs.append("⚠️ Backtest sonucu bulunamadı.")

        if best_configs:
            save_best_configs(best_configs)
            self.backtest_logs.append("✅ En iyi ayarlar canlı trade'e aktarıldı.")

    def save_backtest_meta(self, meta: dict):
        try:
            with open(BACKTEST_META_FILE, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            self.backtest_meta = meta
        except Exception as e:
            self.backtest_logs.append(f"⚠️ Meta kayıt hatası: {e}")

    def load_backtest_meta(self):
        try:
            if os.path.exists(BACKTEST_META_FILE):
                with open(BACKTEST_META_FILE, "r", encoding="utf-8") as f:
                    self.backtest_meta = json.load(f)
            else:
                self.backtest_meta = None
        except Exception as e:
            self.backtest_meta = None
            if hasattr(self, "backtest_logs"):
                self.backtest_logs.append(f"⚠️ Meta okuma hatası: {e}")

    def format_backtest_summary_lines(self, meta: Optional[dict] = None):
        meta = meta or self.backtest_meta
        if not meta or not meta.get("summary"):
            return []

        finished_at = meta.get("finished_at")
        try:
            finished_dt = dateutil.parser.isoparse(finished_at)
            finished_str = finished_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            finished_str = finished_at or "-"

        lines = [f"📅 Tamamlanma: {finished_str}", "📈 Özet Tablo:"]
        for row in meta.get("summary", []):
            lines.append(
                f"- {row.get('symbol', '?')}-{row.get('timeframe', '?')}: "
                f"Trades={row.get('trades', 0)}, WR={float(row.get('win_rate_pct', 0)):.1f}%, "
                f"NetPnL={float(row.get('net_pnl', 0)):.2f}"
            )
        return lines

    def show_saved_backtest_summary(self):
        if not hasattr(self, "backtest_logs"):
            return

        self.backtest_logs.clear()
        lines = self.format_backtest_summary_lines()
        if lines:
            self.backtest_logs.append("🗂️ Son Backtest Özeti:")
            for line in lines:
                self.backtest_logs.append(line)
        else:
            self.backtest_logs.append(
                "ℹ️ Backtest geçmişi henüz yok. İlk backtest tamamlandığında özet burada görünecek."
            )

    # --- OPTIMIZATION STARTUP (FIXED) ---
        # --- GÜNCELLENMİŞ RUN OPTIMIZATION ---
    def run_optimization(self):
        candles = self.opt_candles.value()
        rr_range = (self.opt_rr_start.value(), self.opt_rr_end.value(), self.opt_rr_step.value())
        rsi_range = (self.opt_rsi_start.value(), self.opt_rsi_end.value(), self.opt_rsi_step.value())
        slope_range = (self.opt_slope_start.value(), self.opt_slope_end.value(), self.opt_slope_step.value())

        # Checkbox'tan değeri al (Birazdan aşağıda ekleyeceğiz)
        is_monte_carlo = self.chk_monte_carlo.isChecked()
        use_at = False

        selected_sym = self.combo_opt_symbol.currentText()
        self.opt_logs.clear()
        self.btn_run_opt.setEnabled(False)

        # Worker'a monte_carlo parametresini gönder
        self.opt_worker = OptimizerWorker(selected_sym, candles, rr_range, rsi_range, slope_range, use_at,
                                              is_monte_carlo)
        self.opt_worker.result_signal.connect(self.on_opt_update)
        self.opt_worker.start()

    def on_opt_update(self, msg):
        self.opt_logs.append(msg);
        if "Tamamlandı" in msg: self.btn_run_opt.setEnabled(True)

    def force_daily_report(self):
        self.opt_logs.append("🌙 Manuel Rapor İsteği Gönderildi. Arka planda çalışıyor, lütfen bekleyin...")
        self.auto_backtest.force_run = True

    def create_pnl_table(self):
        # Tabloyu oluşturur
        table = QTableWidget()
        table.setColumnCount(3)  # 3 Sütun: Zaman, Win Rate, PnL
        table.setHorizontalHeaderLabels(["Zaman Dilimi", "Başarı %", "PnL ($)"])

        # Tablo başlıklarının genişliğini ayarla
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        # Örnek boş satırlar ekleyelim (Daha sonra gerçek veriyle dolacak)
        timeframes = ["1m", "5m", "15m", "1h", "4h"]
        table.setRowCount(len(timeframes))

        for i, tf in enumerate(timeframes):
            table.setItem(i, 0, QTableWidgetItem(tf))  # Zaman
            table.setItem(i, 1, QTableWidgetItem("%0.0"))  # Başarı
            table.setItem(i, 2, QTableWidgetItem("$0.00"))  # PnL

        return table

    def update_pnl_table_data(self):
        # 1. Verileri Analiz Et (Global trade_manager'dan çekiyoruz)
        stats = {}

        # trade_manager.history -> Kapanan işlemler listesi
        for trade in trade_manager.history:
            tf = trade.get('timeframe', 'Bilinmiyor')
            pnl = float(trade.get('pnl', 0))

            if tf not in stats:
                stats[tf] = {'wins': 0, 'count': 0, 'total_pnl': 0.0}

            stats[tf]['count'] += 1
            stats[tf]['total_pnl'] += pnl

            # Kâr eden işlem sayısı
            if pnl > 0:
                stats[tf]['wins'] += 1

        # 2. Tabloyu Temizle ve Yeniden Yaz
        self.pnl_table.setRowCount(0)  # Eski satırları sil

        # Zaman dilimlerini belli bir sıraya göre dizmek istersen (isteğe bağlı)
        sirali_tf = ["1m", "5m", "15m", "1h", "4h"]
        mevcut_keys = list(stats.keys())
        # Sadece istatistiği olanları listele, sıralı listede varsa öncelik ver
        final_list = [t for t in sirali_tf if t in mevcut_keys] + [t for t in mevcut_keys if t not in sirali_tf]

        row = 0
        for tf in final_list:
            data = stats[tf]
            self.pnl_table.insertRow(row)

            # Hesaplamalar
            count = data['count']
            wins = data['wins']
            pnl = data['total_pnl']
            win_rate = (wins / count * 100) if count > 0 else 0

            # Hücreleri Hazırla
            # Sütun 0: Zaman Dilimi
            self.pnl_table.setItem(row, 0, QTableWidgetItem(str(tf)))

            # Sütun 1: Başarı Oranı
            self.pnl_table.setItem(row, 1, QTableWidgetItem(f"%{win_rate:.1f} ({wins}/{count})"))

            # Sütun 2: PnL (Renkli)
            pnl_item = QTableWidgetItem(f"${pnl:.2f}")
            if pnl >= 0:
                pnl_item.setForeground(QColor("#00ff00"))  # Parlak Yeşil
            else:
                pnl_item.setForeground(QColor("#ff5555"))  # Kırmızı

            self.pnl_table.setItem(row, 2, pnl_item)

            row += 1


# ==========================================
# 🧪 CLI BACKTEST (Portföy Senkron) - v2
# ==========================================
def _tf_to_timedelta(tf: str) -> timedelta:
    if tf.endswith("m"):
        return timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h"):
        return timedelta(hours=int(tf[:-1]))
    raise ValueError(f"Unsupported timeframe: {tf}")

class SimTradeManager:
    """Dosya IO olmadan (CSV yazmadan) aynı ekonomik modelle backtest yapmak için."""
    def __init__(self, initial_balance=None):
        self.open_trades = []
        self.history = []
        self.cooldowns = {}
        self.wallet_balance = float(initial_balance if initial_balance is not None else TRADING_CONFIG["initial_balance"])
        self.locked_margin = 0.0
        self.total_pnl = 0.0
        self.slippage_pct = TRADING_CONFIG["slippage_rate"]
        self._id = 1

    def check_cooldown(self, symbol, tf, now_utc) -> bool:
        """
        Backtest cooldown kontrolü.
        now_utc ve cooldown zamanı pandas.Timestamp veya datetime olabilir;
        hepsini offset-naive datetime'a çevirip karşılaştırıyoruz.
        """
        k = (symbol, tf)

        if k not in self.cooldowns:
            return False

        expiry = self.cooldowns[k]

        # Yardımcı: ne gelirse gelsin offset-naive datetime'a çevir
        def _to_naive(dt):
            import pandas as pd
            if isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()
            if getattr(dt, "tzinfo", None) is not None:
                dt = dt.replace(tzinfo=None)
            return dt

        now_naive = _to_naive(now_utc)
        exp_naive = _to_naive(expiry)

        if now_naive < exp_naive:
            # hâlâ cooldown süresi içindeyiz
            return True

        # süresi doldu, kaydı sil
        del self.cooldowns[k]
        return False

    def _next_id(self):
        tid = self._id
        self._id += 1
        return tid

    def open_trade(self, trade_data):
        tf = trade_data["timeframe"]
        sym = trade_data["symbol"]

        if (sym, tf) in self.cooldowns:
            if datetime.utcnow() < self.cooldowns[(sym, tf)]:
                return
            del self.cooldowns[(sym, tf)]

        setup_type = trade_data.get("setup", "Unknown")

        if self.wallet_balance < 10:
            return

        raw_entry = float(trade_data["entry"])
        trade_type = trade_data["type"]

        if trade_type == "LONG":
            real_entry = raw_entry * (1 + self.slippage_pct)
        else:
            real_entry = raw_entry * (1 - self.slippage_pct)

        margin_to_use = self.wallet_balance * TRADING_CONFIG["usable_balance_pct"]
        trade_size = margin_to_use * TRADING_CONFIG["leverage"]

        open_time_val = trade_data.get("open_time_utc") or datetime.utcnow()
        if isinstance(open_time_val, pd.Timestamp):
            open_time_val = open_time_val.to_pydatetime()
        open_time_str = open_time_val.strftime("%Y-%m-%dT%H:%M:%SZ")

        new_trade = {
            "id": self._next_id(),
            "symbol": sym,
            "timestamp": trade_data.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
            "open_time_utc": open_time_str,
            "timeframe": tf,
            "type": trade_type,
            "setup": setup_type,
            "entry": real_entry,
            "tp": float(trade_data["tp"]),
            "sl": float(trade_data["sl"]),
            "size": trade_size,
            "margin": margin_to_use,
            "status": "OPEN",
            "pnl": 0.0,
            "breakeven": False,
            "trailing_active": False,
            "partial_taken": False,
            "has_cash": True,
            "close_time": "",
            "close_price": "",
        }

        self.wallet_balance -= margin_to_use
        self.locked_margin += margin_to_use
        self.open_trades.append(new_trade)

    def update_trades(
        self,
        symbol,
        tf,
        candle_high,
        candle_low,
        candle_close,
        candle_time_utc=None,
        pb_top=None,
        pb_bot=None,
    ):
        """
        Trade update modeli (Sim backtest):
        - Mum içi (high/low) ile TP/SL tetiklerini yakalar.
        - TP, mümkünse dinamik olarak PBEMA cloud seviyesine göre değerlendirilir.
        - Aynı mumda hem TP hem SL görülürse konservatif olarak STOP seçer.
        - Partial TP (%50) + breakeven / trailing SL desteklenir.
        - Çıkışta slippage + komisyon + basit funding maliyeti düşer.
        """
        if candle_time_utc is None:
            candle_time_utc = datetime.utcnow()

        closed_indices = []
        just_closed_trades = []

        for i, trade in enumerate(self.open_trades):
            if trade.get("symbol") != symbol:
                continue
            if trade.get("timeframe") != tf:
                continue

            entry = float(trade["entry"])
            tp = float(trade["tp"])
            sl = float(trade["sl"])
            size = float(trade["size"])
            t_type = trade["type"]
            initial_margin = float(trade.get("margin", size / TRADING_CONFIG["leverage"]))

            config = load_optimized_config(symbol, tf)
            use_trailing = config.get("use_trailing", False)
            use_partial = not use_trailing
            use_dynamic_tp = config.get("use_dynamic_pbema_tp", True)

            if t_type == "LONG":
                close_price = candle_close
                fav_price = candle_high
                pnl_percent_close = (close_price - entry) / entry
                pnl_percent_fav = (fav_price - entry) / entry
                in_profit = fav_price > entry
            else:
                close_price = candle_close
                fav_price = candle_low
                pnl_percent_close = (entry - close_price) / entry
                pnl_percent_fav = (entry - fav_price) / entry
                in_profit = fav_price < entry

            dyn_tp = tp
            if use_dynamic_tp:
                try:
                    if pb_top is not None and pb_bot is not None:
                        dyn_tp = float(pb_bot) if t_type == "LONG" else float(pb_top)
                except Exception:
                    dyn_tp = tp
                self.open_trades[i]["tp"] = dyn_tp

            self.open_trades[i]["pnl"] = pnl_percent_close * size

            total_dist = abs(dyn_tp - entry)
            if total_dist <= 0:
                continue
            current_dist = abs(fav_price - entry)
            progress = current_dist / total_dist if total_dist > 0 else 0.0

            if in_profit and use_partial:
                if (not trade.get("partial_taken")) and progress >= 0.50:
                    partial_size = size / 2.0

                    if t_type == "LONG":
                        partial_fill = float(fav_price) * (1 - self.slippage_pct)
                        partial_pnl_percent = (partial_fill - entry) / entry
                    else:
                        partial_fill = float(fav_price) * (1 + self.slippage_pct)
                        partial_pnl_percent = (entry - partial_fill) / entry

                    partial_pnl = partial_pnl_percent * partial_size
                    commission = partial_size * TRADING_CONFIG["total_fee"]
                    net_partial_pnl = partial_pnl - commission
                    margin_release = initial_margin / 2.0

                    self.wallet_balance += margin_release + net_partial_pnl
                    self.locked_margin -= margin_release
                    self.total_pnl += net_partial_pnl

                    partial_record = trade.copy()
                    partial_record["size"] = partial_size
                    partial_record["pnl"] = net_partial_pnl
                    partial_record["status"] = "PARTIAL TP (50%)"
                    partial_record["close_time"] = (
                        candle_time_utc + timedelta(hours=3)
                    ).strftime("%Y-%m-%d %H:%M")
                    partial_record["close_price"] = float(partial_fill)
                    partial_record["pb_ema_top"] = pb_top
                    partial_record["pb_ema_bot"] = pb_bot
                    self.history.append(partial_record)

                    self.open_trades[i]["size"] = partial_size
                    self.open_trades[i]["margin"] = margin_release
                    self.open_trades[i]["partial_taken"] = True
                    self.open_trades[i]["sl"] = entry
                    self.open_trades[i]["breakeven"] = True

                elif (not trade.get("breakeven")) and progress >= 0.40:
                    self.open_trades[i]["sl"] = entry
                    self.open_trades[i]["breakeven"] = True

            if in_profit and use_trailing:
                if (not trade.get("breakeven")) and progress >= 0.40:
                    self.open_trades[i]["sl"] = entry
                    self.open_trades[i]["breakeven"] = True

                if progress >= 0.50:
                    trail_buffer = total_dist * 0.40
                    current_sl = float(self.open_trades[i]["sl"])
                    if t_type == "LONG":
                        new_sl = close_price - trail_buffer
                        if new_sl > current_sl:
                            self.open_trades[i]["sl"] = new_sl
                            self.open_trades[i]["trailing_active"] = True
                    else:
                        new_sl = close_price + trail_buffer
                        if new_sl < current_sl:
                            self.open_trades[i]["sl"] = new_sl
                            self.open_trades[i]["trailing_active"] = True

            sl = float(self.open_trades[i]["sl"])

            if t_type == "LONG":
                hit_tp = candle_high >= dyn_tp
                hit_sl = candle_low <= sl
            else:
                hit_tp = candle_low <= dyn_tp
                hit_sl = candle_high >= sl

            if not (hit_tp or hit_sl):
                continue

            if hit_tp and hit_sl:
                reason = "STOP (BothHit)"
                exit_level = sl
            elif hit_tp:
                reason = "WIN (TP)"
                exit_level = dyn_tp
            else:
                reason = "STOP"
                exit_level = sl

            current_size = float(self.open_trades[i]["size"])
            margin_release = float(self.open_trades[i].get("margin", initial_margin))

            if t_type == "LONG":
                exit_fill = float(exit_level) * (1 - self.slippage_pct)
                pnl_percent = (exit_fill - entry) / entry
            else:
                exit_fill = float(exit_level) * (1 + self.slippage_pct)
                pnl_percent = (entry - exit_fill) / entry

            gross_pnl = pnl_percent * current_size
            commission = current_size * TRADING_CONFIG["total_fee"]

            funding_cost = 0.0
            try:
                open_time_str = trade.get("open_time_utc", "")
                if open_time_str:
                    open_dt = datetime.strptime(open_time_str, "%Y-%m-%dT%H:%M:%SZ")
                    hours = max(0.0, (candle_time_utc - open_dt).total_seconds() / 3600.0)
                    funding_cost = abs(current_size) * TRADING_CONFIG["funding_rate_8h"] * (hours / 8.0)
            except Exception:
                funding_cost = 0.0

            final_net_pnl = gross_pnl - commission - funding_cost

            self.wallet_balance += margin_release + final_net_pnl
            self.locked_margin -= margin_release
            self.total_pnl += final_net_pnl

            if "STOP" in reason:
                wait_minutes = 10 if tf == "1m" else (30 if tf == "5m" else 60)
                self.cooldowns[(symbol, tf)] = datetime.utcnow() + timedelta(minutes=wait_minutes)

            if trade.get("breakeven") and abs(final_net_pnl) < 1e-6 and "STOP" in reason:
                reason = "BE"

            trade["status"] = reason
            trade["pnl"] = final_net_pnl
            trade["close_time"] = (candle_time_utc + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
            trade["close_price"] = float(exit_fill)
            trade["pb_ema_top"] = pb_top
            trade["pb_ema_bot"] = pb_bot

            self.history.append(trade)
            just_closed_trades.append(trade)
            closed_indices.append(i)

        for idx in sorted(closed_indices, reverse=True):
            del self.open_trades[idx]

        return just_closed_trades



def run_portfolio_backtest(
    symbols,
    timeframes,
    candles: int = 3000,
    out_trades_csv: str = "backtest_trades.csv",
    out_summary_csv: str = "backtest_summary.csv",
    limit_map: Optional[dict] = None,
    progress_callback=None,
    draw_trades: bool = True,
    max_draw_trades: Optional[int] = None,
):
    def log(msg: str):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    accepted_signals_raw = {}
    opened_signals = {}
    # ---- HER ÇALIŞTIRMA ÖNCESİ CSV TEMİZLE ----
    if os.path.exists(out_trades_csv):
        os.remove(out_trades_csv)
    if os.path.exists(out_summary_csv):
        os.remove(out_summary_csv)
    """
    Çoklu sembol / timeframe için portföy backtest'i.
    - Veriyi TradingEngine.get_historical_data_pagination ile çeker
    - İndikatörleri hesaplar
    - SimTradeManager ile tüm sinyal / trade akışını simüle eder
    - Trade geçmişini ve özetini CSV'ye yazar
    - Her sembol/timeframe için fiyat datasını da <symbol>_<tf>_prices.csv olarak kaydeder (plot için)
    """
    import heapq

    streams = {}
    limit_map = limit_map or {}
    requested_pairs = list(itertools.product(symbols, timeframes))
    tf_limit_log = set()
    for sym in symbols:
        for tf in timeframes:
            active_limit_map = limit_map if limit_map else BACKTEST_CANDLE_LIMITS
            tf_candle_limit = active_limit_map.get(tf, candles)
            if tf_candle_limit:
                tf_candle_limit = min(candles, tf_candle_limit)
            else:
                tf_candle_limit = candles

            if tf not in tf_limit_log and tf_candle_limit != candles:
                log(f"[BACKTEST] {tf} mum geçmişi {tf_candle_limit} ile sınırlandı.")
                tf_limit_log.add(tf)

            df = TradingEngine.get_historical_data_pagination(sym, tf, total_candles=tf_candle_limit)
            if df is None or df.empty or len(df) < 400:
                log(f"[BACKTEST] {sym}-{tf} datası bulunamadı veya yetersiz (len={0 if df is None else len(df)})")
                continue

            df = TradingEngine.calculate_indicators(df)

            # Plot / debug için fiyat datasını CSV olarak kaydediyoruz
            # Örn: BTCUSDT_5m_prices.csv
            df.to_csv(f"{sym}_{tf}_prices.csv", index=False)

            streams[(sym, tf)] = df.reset_index(drop=True)

    if not streams:
        log("Backtest için veri yok (internet / Binance erişimi?)")
        return

    # --- 1) Her sembol/zaman dilimi için en iyi ayarı tara ---
    best_configs = _optimize_backtest_configs(streams, requested_pairs, progress_callback=progress_callback)

    # Çoklu stream için zaman bazlı event kuyruğu
    heap = []
    ptr = {}
    total_events = 0
    for (sym, tf), df in streams.items():
        warmup = 250
        end = len(df) - 2
        if end <= warmup:
            continue
        ptr[(sym, tf)] = warmup
        total_events += max(0, end - warmup)
        heapq.heappush(
            heap,
            (df.loc[warmup, "timestamp"] + _tf_to_timedelta(tf), sym, tf),
        )

    tm = SimTradeManager(initial_balance=TRADING_CONFIG["initial_balance"])
    logged_cfg_pairs = set()
    processed_events = 0
    next_progress = 10

    # Ana backtest döngüsü
    while heap:
        event_time, sym, tf = heapq.heappop(heap)
        df = streams[(sym, tf)]
        i = ptr[(sym, tf)]
        if i >= len(df) - 1:
            continue

        row = df.iloc[i]

        # Açık pozisyonları güncelle (PBEMA ile birlikte)
        tm.update_trades(
            sym,
            tf,
            candle_high=float(row["high"]),
            candle_low=float(row["low"]),
            candle_close=float(row["close"]),
            candle_time_utc=row["timestamp"] + _tf_to_timedelta(tf),
            pb_top=float(row.get("pb_ema_top", row["close"])),
            pb_bot=float(row.get("pb_ema_bot", row["close"])),
        )

        # Bu sembol/timeframe için optimize edilmiş config
        config = best_configs.get((sym, tf)) or load_optimized_config(sym, tf)
        if (sym, tf) not in logged_cfg_pairs:
            cfg_info = dict(config)
            extra = ""
            if (sym, tf) in best_configs:
                meta = best_configs[(sym, tf)]
                extra = f" | OPT NetPnL={meta['_net_pnl']:.2f} Trades={meta['_trades']}"
            log(f"[BACKTEST][CFG] {sym}-{tf} -> {cfg_info}{extra}")
            logged_cfg_pairs.add((sym, tf))
        rr, rsi, slope = config["rr"], config["rsi"], config["slope"]
        use_at = config.get("at_active", False)

        # Sinyal kontrolü (Base setup mantığı burada)
        s_type, s_entry, s_tp, s_sl, s_reason = TradingEngine.check_signal_diagnostic(
            df,
            index=i,
            min_rr=rr,
            rsi_limit=rsi,
            slope_thresh=slope,
            use_alphatrend=use_at,
            hold_n=config.get("hold_n"),
            min_hold_frac=config.get("min_hold_frac"),
            pb_touch_tolerance=config.get("pb_touch_tolerance"),
            body_tolerance=config.get("body_tolerance"),
            cloud_keltner_gap_min=config.get("cloud_keltner_gap_min"),
            tp_min_dist_ratio=config.get("tp_min_dist_ratio"),
            tp_max_dist_ratio=config.get("tp_max_dist_ratio"),
            adx_min=config.get("adx_min"),
        )

        if s_type and "ACCEPTED" in str(s_reason):
            accepted_signals_raw[(sym, tf)] = accepted_signals_raw.get((sym, tf), 0) + 1
            # Aynı sembol/timeframe için açık trade var mı?
            has_open = any(
                t["symbol"] == sym and t["timeframe"] == tf
                for t in tm.open_trades
            )

            # Cooldown kontrolü
            if (not has_open) and (not tm.check_cooldown(sym, tf, event_time)):
                next_row = df.iloc[i + 1]
                entry_open = float(next_row["open"])
                open_ts = next_row["timestamp"]
                ts_str = (open_ts + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")

                # Setup tag (ör: Base), reason string içinden çekiliyor
                setup_tag = "Unknown"
                s_reason_str = str(s_reason)
                if "ACCEPTED" in s_reason_str and "(" in s_reason_str and ")" in s_reason_str:
                    setup_tag = s_reason_str[s_reason_str.find("(") + 1 : s_reason_str.find(")")]

                tm.open_trade(
                    {
                        "symbol": sym,
                        "timeframe": tf,
                        "type": s_type,
                        "entry": entry_open,
                        "tp": float(s_tp),
                        "sl": float(s_sl),
                        "setup": setup_tag,
                        "timestamp": ts_str,
                        "open_time_utc": open_ts,
                    }
                )
                opened_signals[(sym, tf)] = opened_signals.get((sym, tf), 0) + 1

        # Sonraki bara ilerle
        i2 = i + 1
        ptr[(sym, tf)] = i2
        if i2 < len(df) - 1:
            heapq.heappush(
                heap,
                (df.loc[i2, "timestamp"] + _tf_to_timedelta(tf), sym, tf),
            )

        # Progress log
        processed_events += 1
        if total_events > 0:
            progress = (processed_events / total_events) * 100
            if progress >= next_progress:
                log(f"[BACKTEST] %{progress:.1f} tamamlandı...")
                next_progress += 10
    log(f"[DEBUG] Toplam kapatılmış trade sayısı: {len(tm.history)}")
    if tm.history:
        log(f"[DEBUG] İlk trade örneği: {tm.history[0]}")
    if accepted_signals_raw:
        log("[DEBUG] Kabul edilen (ham) sinyal sayıları:")
        for (sym, tf), cnt in sorted(accepted_signals_raw.items()):
            log(f"  - {sym}-{tf}: {cnt}")
    if opened_signals:
        log("[DEBUG] Açılışa dönüşen sinyal sayıları (backtest tablo ile hizalı):")
        for (sym, tf), cnt in sorted(opened_signals.items()):
            log(f"  - {sym}-{tf}: {cnt}")

    # Tüm history'den DataFrame oluştur ve CSV / özet yaz
    trades_df = pd.DataFrame(tm.history)
    if not trades_df.empty:
        trades_df.to_csv(out_trades_csv, index=False)

    summary_rows = []
    if not trades_df.empty:
        # Aynı id'ye ait tüm bacakları (partial + final) toplayıp
        # trade başına net sonucu hesapla
        grouped_by_trade = {}

        for (sym, tf, tid), g in trades_df.groupby(["symbol", "timeframe", "id"]):
            net = g["pnl"].astype(float).sum()
            key = (sym, tf)
            grouped_by_trade.setdefault(key, []).append(net)

        for (sym, tf), pnl_list in grouped_by_trade.items():
            total = len(pnl_list)
            wins = sum(1 for x in pnl_list if x > 0)
            pnl = sum(pnl_list)
            wr = (wins / total * 100.0) if total else 0.0

            summary_rows.append(
                {
                    "symbol": sym,
                    "timeframe": tf,
                    "trades": total,
                    "win_rate_pct": wr,
                    "net_pnl": pnl,
                }
            )

    # Eksik kalan sembol/zaman dilimlerini 0 trade ile rapora ekle
    existing_pairs = {(row["symbol"], row["timeframe"]) for row in summary_rows}
    for sym, tf in requested_pairs:
        if (sym, tf) in existing_pairs:
            continue
        summary_rows.append(
            {
                "symbol": sym,
                "timeframe": tf,
                "trades": 0,
                "win_rate_pct": 0.0,
                "net_pnl": 0.0,
            }
        )

    summary_df = (
        pd.DataFrame(summary_rows).sort_values(["symbol", "timeframe"])
        if summary_rows
        else pd.DataFrame()
    )
    if not summary_df.empty:
        summary_df.to_csv(out_summary_csv, index=False)

    log("Backtest bitti.")
    if not summary_df.empty:
        log(summary_df.to_string(index=False))

    if best_configs:
        log("\n[OPT] En iyi ayar özeti (Net PnL'e göre):")
        for (sym, tf), cfg in sorted(best_configs.items()):
            log(
                f"  - {sym}-{tf}: RR={cfg['rr']}, RSI={cfg['rsi']}, Slope={cfg['slope']}, "
                f"AT={'Açık' if cfg['at_active'] else 'Kapalı'}, Trailing={cfg.get('use_trailing', False)} | "
                f"NetPnL={cfg['_net_pnl']:.2f}, Trades={cfg['_trades']}"
            )

    log(f"Final Wallet (sim): ${tm.wallet_balance:.2f} | Total PnL: ${tm.total_pnl:.2f}")

    total_trades = trades_df["id"].nunique() if not trades_df.empty and "id" in trades_df.columns else 0
    if total_trades < 5:
        log("[BACKTEST] Çok az trade bulundu. Daha fazla sonuç için RR/RSI/Slope limitlerini biraz gevşetmeyi düşünebilirsin.")

    # Sonuçları GUI/LIVE ile paylaşmak için kaydet
    save_best_configs(best_configs)
    result = {
        "summary": summary_rows,
        "best_configs": best_configs,
        "trades_csv": out_trades_csv,
        "summary_csv": out_summary_csv,
    }

    if draw_trades:
        try:
            replay_backtest_trades(trades_csv=out_trades_csv, max_trades=max_draw_trades)
        except Exception as e:
            log(f"[BACKTEST] Trade çiziminde hata: {e}")

    return result

def plot_trade(
    df_prices: pd.DataFrame,
    df_trades: pd.DataFrame,
    trade_id: int,
    window: int = 40,
    save_dir: Optional[str] = "replay_charts",
    show: bool = False,
):
    """
    Tek bir trade'i (id) fiyat grafiği üzerinde gösterir.

    Görsel öğeler:
    - Fiyat (close)
    - Keltner upper / lower + baseline
    - PBEMA cloud (top / bottom + fill)
    - Entry / TP / SL çizgileri
    - RR kutuları (risk / reward alanları, TradingView RR Tool benzeri)

    Kaydetme/izleme:
    - save_dir: Grafiklerin diske kaydedileceği klasör (None verilirse kaydedilmez)
    - show    : True ise ek olarak plt.show() ile görüntü açılır (varsayılan False)
    """

    trades_for_id = df_trades[df_trades["id"] == trade_id].copy()
    if trades_for_id.empty:
        print(f"[PLOT] Trade not found: {trade_id}")
        return

    def _safe_dt_col(df: pd.DataFrame, col: str):
        if col in df.columns:
            return pd.to_datetime(df[col], utc=True, errors="coerce")
        return pd.Series([pd.NaT] * len(df))

    trades_for_id["_close_dt"] = _safe_dt_col(trades_for_id, "close_time")
    trades_for_id["_open_dt"] = _safe_dt_col(trades_for_id, "open_time_utc")
    trades_for_id["_ts_dt"] = _safe_dt_col(trades_for_id, "timestamp")
    trades_for_id["_sort_dt"] = trades_for_id["_close_dt"].fillna(trades_for_id["_open_dt"]).fillna(
        trades_for_id["_ts_dt"]
    )
    trades_for_id["_row_order"] = np.arange(len(trades_for_id))

    non_partial = trades_for_id[
        ~trades_for_id["status"].astype(str).str.contains("PARTIAL", case=False, na=False)
    ]
    if not non_partial.empty:
        tr = non_partial.sort_values(["_sort_dt", "_row_order"]).iloc[-1]
    else:
        tr = trades_for_id.sort_values(["_sort_dt", "_row_order"]).iloc[-1]

    # 1) Trade zamanı
    ts_trade_utc = None
    if "open_time_utc" in tr and not pd.isna(tr["open_time_utc"]):
        ts_trade_utc = pd.to_datetime(tr["open_time_utc"], utc=True, errors="coerce")
    elif "timestamp" in tr and not pd.isna(tr["timestamp"]):
        ts_trade_utc = pd.to_datetime(tr["timestamp"], utc=True, errors="coerce")

    if ts_trade_utc is None or pd.isna(ts_trade_utc):
        print(f"[PLOT] Trade timestamp not found for id={trade_id}")
        return

    if "timestamp" not in df_prices.columns:
        print("[PLOT] df_prices içinde 'timestamp' kolonu yok.")
        return

    df_prices = df_prices.copy()
    df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"], utc=True, errors="coerce")

    # 2) En yakın mumu bul
    diffs = (df_prices["timestamp"] - ts_trade_utc).abs()
    center_idx = diffs.idxmin()
    center_ts = df_prices["timestamp"].loc[center_idx]

    # 3) Pencere
    start_idx = max(0, center_idx - window)
    end_idx = min(len(df_prices) - 1, center_idx + window)
    w = df_prices.iloc[start_idx: end_idx + 1].copy()

    # 4) Trade metrikleri
    entry = float(tr["entry"])
    tp = float(tr["tp"])
    sl = float(tr["sl"])
    ttype = str(tr.get("type", "UNKNOWN")).upper()
    status = str(tr.get("status", "UNKNOWN"))
    symbol = str(tr.get("symbol", ""))
    timeframe = str(tr.get("timeframe", ""))

    # 5) Figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Close fiyatı
    ax.plot(w["timestamp"], w["close"], linewidth=1.0, label="Close")

    # Keltner + baseline
    if "keltner_upper" in w.columns and "keltner_lower" in w.columns:
        ax.plot(
            w["timestamp"], w["keltner_upper"],
            linewidth=0.9, color="#ff5252", linestyle=":", label="Keltner Upper"
        )
        ax.plot(
            w["timestamp"], w["keltner_lower"],
            linewidth=0.9, color="#4caf50", linestyle=":", label="Keltner Lower"
        )
        if "baseline" in w.columns:
            ax.plot(
                w["timestamp"], w["baseline"],
                linewidth=0.9, color="#ffd700", label="Baseline"
            )

    # PBEMA cloud
    if "pb_ema_top" in w.columns and "pb_ema_bot" in w.columns:
        pb_color = "#42a5f5"
        ax.plot(w["timestamp"], w["pb_ema_top"], linewidth=1.0, color=pb_color, label="PBEMA Top")
        ax.plot(w["timestamp"], w["pb_ema_bot"], linewidth=1.0, color=pb_color, label="PBEMA Bottom")
        ax.fill_between(
            w["timestamp"],
            w["pb_ema_bot"],
            w["pb_ema_top"],
            alpha=0.18,
            color=pb_color,
            label="PBEMA Cloud",
        )

    # Entry / TP / SL çizgileri
    ax.axhline(entry, linestyle="--", linewidth=1.2, label=f"ENTRY {entry:.1f}")
    ax.axhline(tp, linestyle="--", linewidth=1.2, label=f"TP {tp:.1f}")
    ax.axhline(sl, linestyle="--", linewidth=1.2, label=f"SL {sl:.1f}")

    # Entry dikey çizgi ve marker
    ax.axvline(center_ts, linestyle=":", linewidth=1.0)
    marker_color = "green" if ttype == "LONG" else "red"
    ax.scatter(center_ts, entry, s=80, marker="*", color=marker_color, zorder=5)

    # 6) RR kutuları (Plotly'deki shapes mantığına benzer, ~20 mum genişlik)
    if len(w) >= 2:
        candle_delta = w["timestamp"].iloc[1] - w["timestamp"].iloc[0]
    else:
        # fallback: 5 dakika
        candle_delta = pd.Timedelta(minutes=5)

    box_x0 = center_ts
    box_x1 = center_ts + candle_delta * 20  # canlı grafikteki future_ts_str ~ time_diff * 20

    def draw_rr_box(y0, y1, color, alpha=0.25):
        ys = [y0, y0, y1, y1]
        xs = [box_x0, box_x1, box_x1, box_x0]
        ax.fill(xs, ys, alpha=alpha, color=color, linewidth=0)

    if ttype == "LONG":
        # Reward kutusu: entry -> TP
        y_prof0, y_prof1 = sorted((entry, tp))
        # Risk kutusu: SL -> entry
        y_loss0, y_loss1 = sorted((sl, entry))
        draw_rr_box(y_prof0, y_prof1, "green", alpha=0.25)
        draw_rr_box(y_loss0, y_loss1, "red", alpha=0.25)
    elif ttype == "SHORT":
        # Reward kutusu: TP -> entry
        y_prof0, y_prof1 = sorted((tp, entry))
        # Risk kutusu: entry -> SL
        y_loss0, y_loss1 = sorted((entry, sl))
        draw_rr_box(y_prof0, y_prof1, "green", alpha=0.25)
        draw_rr_box(y_loss0, y_loss1, "red", alpha=0.25)

    # Başlık, grid, legend
    ax.set_title(f"{symbol} {timeframe} | Trade ID {trade_id} — {ttype} ({status})")
    ax.set_xlabel("Zaman")
    ax.set_ylabel("Fiyat")
    ax.grid(True)
    plt.xticks(rotation=45)
    ax.legend(loc="best")
    plt.tight_layout()

    saved_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"trade_{trade_id}_{symbol}_{timeframe}.png"
        saved_path = os.path.join(save_dir, filename)
        fig.savefig(saved_path)
        print(f"[PLOT] Kaydedildi: {saved_path}")

    if show:
        plt.show()

    plt.close(fig)




def replay_backtest_trades(
    trades_csv: str = "backtest_trades.csv",
    max_trades: Optional[int] = None,
    window: int = 60,
    save_dir: Optional[str] = "replay_charts",
    show: bool = False,
):
    """
    Backtest sonrası üretilen trade'leri grafik üzerinde sırayla gösterir.

    - trades_csv: run_portfolio_backtest'in yazdığı trade CSV
    - max_trades: en fazla kaç trade çizilecek
    - window    : her trade etrafında kaç mumluk pencere gösterilecek
    - save_dir  : grafiklerin kaydedileceği klasör (None verilirse kaydetmez)
    - show      : True ise matplotlib penceresi açılır (default False, GUI block riskine karşı)
    """

    if not os.path.exists(trades_csv):
        print(f"[REPLAY] Trades CSV bulunamadı: {trades_csv}")
        return

    df_trades = pd.read_csv(trades_csv)

    if "id" not in df_trades.columns:
        print("[REPLAY] 'id' kolonu yok, trade'ler beklenen formatta değil.")
        return

    # Status kolonu varsa, en azından boş olmayanları bırak ki tüm kapananlar çizilsin
    if "status" in df_trades.columns:
        df_trades = df_trades[df_trades["status"].notna()]
        # Yaşanacak olası format değişikliklerinde kazanım/kayıp etiketleri yine de gösterilsin
        df_trades = df_trades[df_trades["status"].astype(str).str.len() > 0]
        if df_trades.empty:
            print("[REPLAY] 'status' kolonu boş, çizilecek trade yok.")
            return

    # Zaman sırasına göre sırala
    if "open_time_utc" in df_trades.columns:
        df_trades["open_time_utc"] = pd.to_datetime(df_trades["open_time_utc"], errors="coerce")
        df_trades = df_trades.sort_values("open_time_utc")
    elif "timestamp" in df_trades.columns:
        df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"], errors="coerce")
        df_trades = df_trades.sort_values("timestamp")

    # En fazla max_trades kadarını al (None verilirse hepsini çiz)
    if isinstance(max_trades, int) and max_trades > 0:
        df_trades = df_trades.head(max_trades)

    print(f"[REPLAY] Toplam {len(df_trades)} trade çizilecek.")

    for _, tr in df_trades.iterrows():
        trade_id = int(tr["id"])
        symbol = str(tr.get("symbol", "UNKNOWN"))
        timeframe = str(tr.get("timeframe", "UNKNOWN"))

        prices_path = f"{symbol}_{timeframe}_prices.csv"
        if not os.path.exists(prices_path):
            print(f"[REPLAY] Fiyat datası bulunamadı: {prices_path} (trade id={trade_id})")
            continue

        df_prices = pd.read_csv(prices_path)

        print(f"[REPLAY] Çiziliyor: id={trade_id}, {symbol} {timeframe}")
        plot_trade(
            df_prices,
            df_trades,
            trade_id=trade_id,
            window=window,
            save_dir=save_dir,
            show=show,
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())









