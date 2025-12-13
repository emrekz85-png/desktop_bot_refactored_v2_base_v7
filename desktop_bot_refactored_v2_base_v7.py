import sys
import os
import time
import json
import io
import socket
import ssl
import base64
import contextlib
import threading
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import dateutil.parser
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta
import traceback
import tempfile
import shutil
from typing import Tuple, Optional
import matplotlib
import hashlib

# ==========================================
# 🚀 GOOGLE COLAB / HEADLESS MODE SUPPORT
# ==========================================
# Detect if running in Google Colab or headless environment
IS_COLAB = 'google.colab' in sys.modules or os.environ.get('COLAB_RELEASE_TAG') is not None
IS_HEADLESS = os.environ.get('HEADLESS_MODE', '').lower() in ('1', 'true', 'yes') or IS_COLAB
IS_NOTEBOOK = 'ipykernel' in sys.modules

# Try to import tqdm for progress bars (Colab/Jupyter friendly)
try:
    if IS_NOTEBOOK:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

# Print Colab status
if IS_COLAB:
    print("🚀 Google Colab ortamı tespit edildi - Headless mod aktif")
elif IS_HEADLESS:
    print("🖥️ Headless mod aktif - GUI devre dışı")

# Matplotlib çizimlerini arka planda üretmek için GUI gerektirmeyen backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PyQt5 Modülleri - sadece GUI modunda yükle
if not IS_HEADLESS:
    try:
        from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                     QHBoxLayout, QGridLayout, QTabWidget, QTextEdit, QLabel,
                                     QTableWidget, QTableWidgetItem, QPushButton, QHeaderView,
                                     QGroupBox, QDoubleSpinBox, QComboBox, QMessageBox, QCheckBox,
                                     QLineEdit, QSpinBox, QFrame)
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
        from PyQt5.QtGui import QColor, QFont
        HAS_GUI = True
    except ImportError:
        print("⚠️ PyQt5 bulunamadı - GUI devre dışı, sadece CLI modu kullanılabilir")
        HAS_GUI = False
        IS_HEADLESS = True
else:
    HAS_GUI = False

import plotly.graph_objects as go
import plotly.utils

# ==========================================
# ⚙️ GENEL AYARLAR VE SABİTLER (MERKEZİ YÖNETİM)
# ==========================================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "LINKUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT", "DOGEUSDT", "SUIUSDT", "FARTCOINUSDT"]

# ==========================================
# 🛡️ HTF ONLY MODE (Opsiyonel)
# ==========================================
# True = Sadece 1h+ timeframe kullan (daha az trade, daha az gürültü)
# False = Tüm timeframe'leri kullan (5m, 15m, 1h, 4h, 12h, 1d)
# NOT: Problem timeframe değil, optimizasyon periyodu. Canlı trading'de
# düzenli re-optimizasyon (her 1-2 haftada) yapılması önerilir.
HTF_ONLY_MODE = False

# 1m removed - too noisy, inconsistent results across all symbols
# 5m, 15m, 30m - Eğer HTF_ONLY_MODE aktifse devre dışı kalır
_ALL_LOWER_TIMEFRAMES = ["5m", "15m", "30m", "1h"]
LOWER_TIMEFRAMES = ["1h"] if HTF_ONLY_MODE else _ALL_LOWER_TIMEFRAMES
HTF_TIMEFRAMES = ["4h", "12h", "1d"]
TIMEFRAMES = LOWER_TIMEFRAMES + HTF_TIMEFRAMES
candles = 50000
REFRESH_RATE = 3

# Grafik güncelleme - False = daha hızlı başlatma, daha az CPU
ENABLE_CHARTS = False
CSV_FILE = "trades.csv"
CONFIG_FILE = "config.json"
# Backtestler için maks. mum sayısı sınırları (yüksek limit - kullanıcının isteğine göre)
BACKTEST_CANDLE_LIMITS = {
    "1m": 100000,
    "5m": 100000,
    "15m": 100000,
    "30m": 100000,
    "1h": 100000,
    "4h": 50000,
    "12h": 30000,
    "1d": 20000,
}
# Günlük raporlar için özel mum sayısı sınırı
DAILY_REPORT_CANDLE_LIMITS = {
    "1m": 15000,
    "5m": 15000,
    "15m": 15000,
    "30m": 15000,
    "1h": 15000,
    "4h": 8000,
    "12h": 5000,
    "1d": 4000,
}
BEST_CONFIGS_FILE = "best_configs.json"
BEST_CONFIG_CACHE = {}
BEST_CONFIG_WARNING_FLAGS = {
    "missing_signature": False,
    "signature_mismatch": False,
}
BACKTEST_META_FILE = "backtest_meta.json"
POT_LOG_FILE = "potential_trades.json"  # Persistent storage for potential trade logs
# Çökme veya kapanma durumlarında otomatik yeniden başlatma gecikmesi (saniye)
AUTO_RESTART_DELAY_SECONDS = 5

# --- 💰 EKONOMİK MODEL (Tüm Modüller Burayı Kullanacak) ---
#  uyarınca tek bir konfigürasyon yapısı:
TRADING_CONFIG = {
    "initial_balance": 2000.0,
    "leverage": 10,
    "usable_balance_pct": 0.20,  # Bakiyenin %20'si
    # INCREASED from 0.015 to 0.0175 for additional profit potential
    "risk_per_trade_pct": 0.0175,  # Her işlemde %1.75 risk (was 1.5%)
    "max_portfolio_risk_pct": 0.05,  # Toplam portföy riski %5 (was 4.5%)
    "slippage_rate": 0.0005,     # %0.05 Kayma Payı
    "funding_rate_8h": 0.0001,   # %0.01 Fonlama (8 saatlik)
    "maker_fee": 0.0002,         # %0.02 Limit Emir Komisyonu
    "taker_fee": 0.0005,         # %0.05 Piyasa Emir Komisyonu
    "total_fee": 0.0007          # %0.07 (Giriş + Çıkış Tahmini) - Güvenlik marjı
}

# ==========================================
# 🎯 v39.0 - R-MULTIPLE BASED OPTIMIZER GATING
# ==========================================
# Multi-layer gating to prevent weak-edge configs from trading:
# 1. Minimum E[R] threshold (account-size independent)
# 2. Minimum score threshold (varies by timeframe)
# 3. Confidence-based risk multiplier
# 4. Walk-forward out-of-sample validation
# ==========================================

# R-Multiple = PnL / Risk Amount
# E[R] = Expected R-multiple per trade (average R across all trades)
# - E[R] > 0: Positive expectancy (profitable system)
# - E[R] = 1.0: Average win equals average risk
# - E[R] = -1.0: Average loss equals average risk (typical SL hit)
#
# Bu metrik hesap büyüklüğünden, kaldıraçtan ve komisyondan bağımsızdır.
# Modelin gerçek "edge"ini ölçer.

# Minimum E[R] (expected R-multiple) threshold by timeframe
# Configs with lower E[R] are considered "barely positive" and rejected
MIN_EXPECTANCY_R_MULTIPLE = {
    "1m": 0.10,   # Very noisy - need strong edge
    "5m": 0.06,   # Noisy - need decent edge per trade
    "15m": 0.05,  # Moderate noise
    "30m": 0.04,
    "1h": 0.04,   # Cleaner signals
    "4h": 0.03,   # Low noise
    "1d": 0.02,   # Very clean
}

# DEPRECATED: Eski $/trade eşikleri (geriye uyumluluk için korunuyor)
# Bu değerler artık kullanılmıyor, E[R] kullanılıyor
MIN_EXPECTANCY_PER_TRADE = {
    "1m": 6.0,   # Very noisy - need strong edge
    "5m": 4.0,   # Noisy - need decent edge per trade
    "15m": 3.0,  # Moderate noise
    "30m": 2.5,
    "1h": 2.0,   # Cleaner signals
    "4h": 1.5,   # Low noise
    "1d": 1.0,   # Very clean
}

# Minimum optimizer score threshold by timeframe
# Higher timeframes have lower bars because fewer trades naturally
MIN_SCORE_THRESHOLD = {
    "1m": 80.0,
    "5m": 40.0,   # High bar for noisy timeframe
    "15m": 15.0,  # Medium bar
    "30m": 10.0,
    "1h": 8.0,    # Lower bar
    "4h": 5.0,    # Lower bar
    "1d": 3.0,
}

# Confidence-based risk multiplier
# Reduces position size for medium-confidence streams
CONFIDENCE_RISK_MULTIPLIER = {
    "high": 1.0,    # Full risk
    "medium": 0.5,  # Half risk - protects against optimizer overfitting
    "low": 0.0,     # No trades (effectively disabled)
}

# ==========================================
# 🚀 v37.0 - DYNAMIC OPTIMIZER CONTROLS DISABLED STATE
# ==========================================
# All hardcoded "disabled: True" removed - optimizer decides at runtime
# based on whether positive PnL config exists for each stream.
# This allows streams to be re-enabled when market conditions change.
# ==========================================
SYMBOL_PARAMS = {
    "BTCUSDT": {
        "5m": {"rr": 2.4, "rsi": 35, "slope": 0.2, "at_active": False, "use_trailing": False},
        "15m": {"rr": 1.2, "rsi": 45, "slope": 0.2, "at_active": True, "use_trailing": False},
        "30m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": False, "use_trailing": False},
        "1h": {"rr": 2.1, "rsi": 45, "slope": 0.2, "at_active": False, "use_trailing": False},
        "4h": {"rr": 2.1, "rsi": 35, "slope": 0.2, "at_active": False, "use_trailing": False},
        "12h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False},
        "1d": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False}
    },
    "ETHUSDT": {
        "5m": {"rr": 1.5, "rsi": 35, "slope": 0.2, "at_active": True, "use_trailing": False},
        "15m": {"rr": 1.2, "rsi": 45, "slope": 0.2, "at_active": True, "use_trailing": False},
        "30m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": False, "use_trailing": False},
        "1h": {"rr": 1.2, "rsi": 45, "slope": 0.2, "at_active": False, "use_trailing": False},
        "4h": {"rr": 2.0, "rsi": 30, "slope": 0.3, "at_active": True, "use_trailing": False},
        "12h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": True, "use_trailing": False},
        "1d": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": True, "use_trailing": False}
    },
    "SOLUSDT": {
        "5m": {"rr": 1.5, "rsi": 45, "slope": 0.2, "at_active": True, "use_trailing": True},
        "15m": {"rr": 1.5, "rsi": 35, "slope": 0.2, "at_active": True, "use_trailing": False},
        "30m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": False, "use_trailing": False},
        "1h": {"rr": 1.2, "rsi": 35, "slope": 0.2, "at_active": False, "use_trailing": False},
        "4h": {"rr": 2.0, "rsi": 30, "slope": 0.3, "at_active": True, "use_trailing": False},
        "12h": {"rr": 2.0, "rsi": 35, "slope": 0.4, "at_active": True, "use_trailing": False},
        "1d": {"rr": 2.0, "rsi": 35, "slope": 0.4, "at_active": True, "use_trailing": False}
    },
    "HYPEUSDT": {
        "5m": {"rr": 1.2, "rsi": 35, "slope": 0.2, "at_active": True, "use_trailing": True},
        "15m": {"rr": 1.5, "rsi": 55, "slope": 0.2, "at_active": True, "use_trailing": False},
        "30m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": False, "use_trailing": False},
        "1h": {"rr": 1.2, "rsi": 35, "slope": 0.2, "at_active": False, "use_trailing": False},
        "4h": {"rr": 1.2, "rsi": 35, "slope": 0.2, "at_active": False, "use_trailing": False},
        "12h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": True, "use_trailing": False},
        "1d": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": True, "use_trailing": False}
    },
    "LINKUSDT": {
        "5m": {"rr": 1.2, "rsi": 35, "slope": 0.2, "at_active": True, "use_trailing": True},
        "15m": {"rr": 1.2, "rsi": 45, "slope": 0.2, "at_active": True, "use_trailing": False},
        "30m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": False, "use_trailing": False},
        "1h": {"rr": 1.2, "rsi": 35, "slope": 0.2, "at_active": False, "use_trailing": False},
        "4h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False},
        "12h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False},
        "1d": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False}
    },
    "BNBUSDT": {
        "5m": {"rr": 1.5, "rsi": 35, "slope": 0.2, "at_active": True, "use_trailing": True},
        "15m": {"rr": 2.4, "rsi": 35, "slope": 0.2, "at_active": False, "use_trailing": False},
        "30m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": False, "use_trailing": False},
        "1h": {"rr": 1.2, "rsi": 55, "slope": 0.2, "at_active": False, "use_trailing": False},
        "4h": {"rr": 1.8, "rsi": 35, "slope": 0.2, "at_active": False, "use_trailing": False},
        "12h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False},
        "1d": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False}
    },
    "XRPUSDT": {
        "5m": {"rr": 2.4, "rsi": 35, "slope": 0.4, "at_active": False, "use_trailing": False},
        "15m": {"rr": 2.4, "rsi": 45, "slope": 0.2, "at_active": False, "use_trailing": False},
        "30m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": False, "use_trailing": False},
        "1h": {"rr": 1.2, "rsi": 35, "slope": 0.2, "at_active": True, "use_trailing": False},
        "4h": {"rr": 1.2, "rsi": 45, "slope": 0.2, "at_active": True, "use_trailing": False},
        "12h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False},
        "1d": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False}
    },
    "LTCUSDT": {
        "5m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": True, "use_trailing": True},
        "15m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": True, "use_trailing": False},
        "30m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": False, "use_trailing": False},
        "1h": {"rr": 1.8, "rsi": 35, "slope": 0.2, "at_active": False, "use_trailing": False},
        "4h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False},
        "12h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False},
        "1d": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False}
    },
    "DOGEUSDT": {
        "5m": {"rr": 1.5, "rsi": 35, "slope": 0.2, "at_active": True, "use_trailing": True},
        "15m": {"rr": 1.5, "rsi": 35, "slope": 0.2, "at_active": True, "use_trailing": False},
        "30m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": False, "use_trailing": False},
        "1h": {"rr": 1.2, "rsi": 35, "slope": 0.2, "at_active": False, "use_trailing": False},
        "4h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False},
        "12h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False},
        "1d": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False}
    },
    "SUIUSDT": {
        "5m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": True, "use_trailing": True},
        "15m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": True, "use_trailing": False},
        "30m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": False, "use_trailing": False},
        "1h": {"rr": 1.2, "rsi": 35, "slope": 0.2, "at_active": False, "use_trailing": False},
        "4h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False},
        "12h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False},
        "1d": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False}
    },
    "FARTCOINUSDT": {
        # New memecoin - let optimizer find best params
        "5m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": True, "use_trailing": True},
        "15m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": True, "use_trailing": False},
        "30m": {"rr": 1.5, "rsi": 40, "slope": 0.2, "at_active": False, "use_trailing": False},
        "1h": {"rr": 1.5, "rsi": 35, "slope": 0.2, "at_active": False, "use_trailing": False},
        "4h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False},
        "12h": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False},
        "1d": {"rr": 2.0, "rsi": 35, "slope": 0.3, "at_active": False, "use_trailing": False}
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
    # LOOSENED PARAMETERS for more trade opportunities:
    "min_hold_frac": 0.50,           # Was 0.65 - now 50% holding is enough
    "pb_touch_tolerance": 0.0025,    # Was 0.0018 - more tolerance for Keltner touch
    "body_tolerance": 0.0025,        # Was 0.0020 - more tolerance for candle body
    "cloud_keltner_gap_min": 0.0015, # Was 0.0025 - smaller gap required
    "tp_min_dist_ratio": 0.0008,     # Was 0.0010 - allow closer TPs
    "tp_max_dist_ratio": 0.040,      # Was 0.035 - allow further TPs
    "adx_min": 8.0,                  # Was 10.0 - less strict ADX requirement
    # Strategy mode: "keltner_bounce" (default) or "pbema_reaction"
    "strategy_mode": "keltner_bounce",
    # PBEMA reaction strategy parameters (used when strategy_mode="pbema_reaction")
    "pbema_approach_tolerance": 0.003,  # 0.3% tolerance for PBEMA approach/touch
    "pbema_frontrun_margin": 0.002,     # 0.2% margin for frontrunning
}

PARTIAL_STOP_PROTECTION_TFS = {"5m", "15m", "30m", "1h"}


def _strategy_signature() -> str:
    """Create a deterministic fingerprint of the current strategy inputs.

    The hash combines the live trading configuration, the default strategy
    parameters and the symbol-specific overrides so that backtest results are
    only consumed when they were produced with the exact same settings. This
    helps keep live trading aligned with the simulated runs.
    """

    payload = {
        "trading": TRADING_CONFIG,
        "strategy": DEFAULT_STRATEGY_CONFIG,
        "symbol_params": SYMBOL_PARAMS,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _apply_1m_profit_lock(
    trade: dict, tf: str, t_type: str, entry: float, tp: float, progress: float
) -> bool:
    """Shift SL into profit on 1m trades once price is very close to TP.

    When the price reaches 80% of the distance to TP, move SL to a level that
    locks 40% of the entry-to-TP distance as profit. Applies to both live and
    backtest flows.
    """

    if tf != "1m" or progress < 0.80:
        return False

    total_dist = abs(tp - entry)
    if total_dist <= 0:
        return False

    current_sl = float(trade.get("sl", entry))
    lock_distance = total_dist * 0.40

    if t_type == "LONG":
        target_sl = entry + lock_distance
        if target_sl > current_sl:
            trade["sl"] = target_sl
            trade["breakeven"] = True
            return True
    else:
        target_sl = entry - lock_distance
        if target_sl < current_sl:
            trade["sl"] = target_sl
            trade["breakeven"] = True
            return True

    return False


def _apply_partial_stop_protection(trade: dict, tf: str, progress: float, t_type: str) -> bool:
    """Raise SL to partial fill price after deeper TP progress on higher timeframes.

    %80 progress'te SL'yi partial fill seviyesine çeker (kar koruma).
    """

    if tf not in PARTIAL_STOP_PROTECTION_TFS:
        return False

    if not trade.get("partial_taken") or progress < 0.80:
        return False

    p_price = trade.get("partial_price")
    if p_price is None:
        return False

    p_price = float(p_price)
    current_sl = float(trade.get("sl", p_price))

    if t_type == "LONG" and p_price > current_sl:
        trade["sl"] = p_price
        trade["stop_protection"] = True
        return True
    if t_type == "SHORT" and p_price < current_sl:
        trade["sl"] = p_price
        trade["stop_protection"] = True
        return True

    return False


def _append_trade_event(trade: dict, event_type: str, event_time, price: Optional[float] = None):
    """Append a serializable lifecycle event to the trade for plotting/logging parity."""

    try:
        events = trade.get("events", [])
        if isinstance(events, str):
            try:
                events = json.loads(events)
            except Exception:
                events = []
        if not isinstance(events, list):
            events = []

        # event_time'ı datetime'a çevir (numpy.datetime64, pd.Timestamp veya datetime olabilir)
        et = event_time or datetime.utcnow()
        if isinstance(et, np.datetime64):
            et = pd.Timestamp(et).to_pydatetime()
        elif isinstance(et, pd.Timestamp):
            et = et.to_pydatetime()

        events.append(
            {
                "type": event_type,
                "time": et.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "price": float(price) if price is not None else None,
            }
        )
        trade["events"] = events
    except Exception:
        trade["events"] = trade.get("events", [])


def _audit_trade_logic_parity() -> dict:
    """Run a lightweight deterministic simulation on both managers to ensure parity.

    Includes config change simulation to verify that trades use snapshot values
    and are not affected by mid-trade config changes.
    """

    global BEST_CONFIG_CACHE

    try:
        symbol = "TESTCOIN"
        tf = "5m"
        seed_ts = datetime.utcnow()
        trade_data = {
            "symbol": symbol,
            "timeframe": tf,
            "type": "LONG",
            "entry": 100.0,
            "tp": 103.0,
            "sl": 99.0,
            "setup": "PARITY",
            "timestamp": seed_ts.strftime("%Y-%m-%d %H:%M"),
            "open_time_utc": seed_ts,
        }

        # Backup current config cache
        original_cache = BEST_CONFIG_CACHE.copy() if BEST_CONFIG_CACHE else {}

        live_tm = TradeManager(persist=False, verbose=False)
        sim_tm = SimTradeManager(initial_balance=TRADING_CONFIG["initial_balance"])

        live_tm.open_trade(trade_data)
        sim_tm.open_trade(trade_data)

        # Verify snapshot fields were written to trade
        snapshot_fields_ok = True
        for tm, name in [(live_tm, "live"), (sim_tm, "sim")]:
            if tm.open_trades:
                t = tm.open_trades[0]
                if "use_trailing" not in t or "use_dynamic_pbema_tp" not in t:
                    snapshot_fields_ok = False

        # CRITICAL: Simulate config change AFTER trade is open
        # This tests that update_trades uses snapshot values, not fresh config
        BEST_CONFIG_CACHE[symbol] = {
            tf: {
                "use_trailing": True,  # Changed from default False
                "use_dynamic_pbema_tp": False,  # Changed from default True
                "rr": 99.0,  # Obviously different value
                "rsi": 99,
            }
        }

        candle_time = seed_ts
        candles = [
            {"high": 101.6, "low": 99.2, "close": 101.0},
            {"high": 103.4, "low": 100.5, "close": 103.0},
        ]

        for c in candles:
            candle_time += timedelta(minutes=5)
            live_tm.update_trades(
                symbol,
                tf,
                candle_high=c["high"],
                candle_low=c["low"],
                candle_close=c["close"],
                candle_time_utc=candle_time,
                pb_top=104.0,
                pb_bot=102.5,
            )
            sim_tm.update_trades(
                symbol,
                tf,
                candle_high=c["high"],
                candle_low=c["low"],
                candle_close=c["close"],
                candle_time_utc=candle_time,
                pb_top=104.0,
                pb_bot=102.5,
            )

        # Restore original config cache
        BEST_CONFIG_CACHE.clear()
        BEST_CONFIG_CACHE.update(original_cache)

        live_hist = live_tm.history
        sim_hist = sim_tm.history

        def _normalize(hist):
            result = []
            for t in hist:
                result.append(
                    {
                        "status": t.get("status"),
                        "pnl": round(float(t.get("pnl", 0.0)), 6),
                        "close_price": round(float(t.get("close_price", 0.0)), 6)
                        if t.get("close_price") not in (None, "")
                        else None,
                    }
                )
            return result

        parity_ok = (
            len(live_hist) == len(sim_hist)
            and _normalize(live_hist) == _normalize(sim_hist)
            and abs(live_tm.wallet_balance - sim_tm.wallet_balance) < 1e-6
        )

        return {
            "parity_ok": parity_ok,
            "snapshot_fields_ok": snapshot_fields_ok,
            "live_trades": _normalize(live_hist),
            "sim_trades": _normalize(sim_hist),
            "wallet_live": live_tm.wallet_balance,
            "wallet_sim": sim_tm.wallet_balance,
        }
    except Exception as exc:
        # Restore original config cache on error
        try:
            BEST_CONFIG_CACHE.clear()
            BEST_CONFIG_CACHE.update(original_cache)
        except Exception:
            pass
        return {"parity_ok": False, "error": str(exc)}


def _generate_candidate_configs():
    """Create a compact grid of configs to search for higher trade density.

    NOT: Slope parametresi artık taranmıyor çünkü:
    - PBEMA (200 EMA) çok yavaş hareket eder
    - Mean reversion stratejisinde slope filter mantıksız
    - Slope taramak sadece zaman kaybı

    İki strateji modu test edilir:
    - keltner_bounce: Keltner band bounce (mevcut)
    - pbema_reaction: PBEMA tepki stratejisi (yeni)
    """

    rr_vals = np.arange(1.2, 2.6, 0.3)
    rsi_vals = np.arange(35, 76, 10)
    at_vals = [False, True]
    # Include both dynamic TP options to ensure optimizer matches what live will use
    dyn_tp_vals = [True, False]
    # Test both strategy modes
    strategy_modes = ["keltner_bounce", "pbema_reaction"]

    candidates = []

    # Keltner bounce strategy configs
    for rr, rsi, at_active, dyn_tp in itertools.product(
        rr_vals, rsi_vals, at_vals, dyn_tp_vals
    ):
        candidates.append(
            {
                "rr": round(float(rr), 2),
                "rsi": int(rsi),
                "slope": 0.5,  # Sabit değer - artık kullanılmıyor
                "at_active": bool(at_active),
                "use_trailing": False,
                "use_dynamic_pbema_tp": bool(dyn_tp),
                "strategy_mode": "keltner_bounce",
            }
        )

    # PBEMA reaction strategy configs
    # PBEMA stratejisi için farklı parametre kombinasyonları
    pbema_approach_vals = [0.002, 0.003, 0.004]  # PBEMA yaklaşım toleransı
    pbema_frontrun_vals = [0.0015, 0.002, 0.003]  # Frontrun marjı

    for rr, rsi, approach_tol, frontrun_margin in itertools.product(
        rr_vals, rsi_vals, pbema_approach_vals, pbema_frontrun_vals
    ):
        candidates.append(
            {
                "rr": round(float(rr), 2),
                "rsi": int(rsi),
                "slope": 0.5,
                "at_active": False,  # PBEMA stratejisinde AT kullanılmıyor
                "use_trailing": False,
                "use_dynamic_pbema_tp": False,  # PBEMA stratejisinde dinamik TP yok
                "strategy_mode": "pbema_reaction",
                "pbema_approach_tolerance": approach_tol,
                "pbema_frontrun_margin": frontrun_margin,
            }
        )

    # Birkaç agresif trailing seçeneği ekle (sadece keltner bounce için)
    trailing_extras = []
    keltner_configs = [c for c in candidates if c.get("strategy_mode") == "keltner_bounce"]
    for base in keltner_configs[:: max(1, len(keltner_configs) // 20)]:  # toplamı şişirmeden örnekle
        cfg = dict(base)
        cfg["use_trailing"] = True
        trailing_extras.append(cfg)

    return candidates + trailing_extras


def _generate_quick_candidate_configs():
    """Create a minimal config grid for quick testing (~24 configs instead of ~120).

    Used when quick_mode=True for faster backtest iterations.
    Covers key combinations without exhaustive search.
    Both strategy modes (keltner_bounce and pbema_reaction) are tested.
    """
    # Sadece en önemli RR ve RSI değerlerini kullan
    rr_vals = [1.2, 1.8, 2.4]  # 3 değer (vs 5)
    rsi_vals = [35, 55]        # 2 değer (vs 5)
    at_vals = [False, True]    # 2 değer
    dyn_tp_vals = [True]       # Sadece 1 değer (dinamik TP genelde daha iyi)

    candidates = []

    # Keltner bounce strategy configs
    for rr, rsi, at_active, dyn_tp in itertools.product(
        rr_vals, rsi_vals, at_vals, dyn_tp_vals
    ):
        candidates.append(
            {
                "rr": round(float(rr), 2),
                "rsi": int(rsi),
                "slope": 0.5,
                "at_active": bool(at_active),
                "use_trailing": False,
                "use_dynamic_pbema_tp": bool(dyn_tp),
                "strategy_mode": "keltner_bounce",
            }
        )

    # PBEMA reaction strategy configs (minimal grid)
    pbema_approach_vals = [0.003]  # Sadece 1 değer
    pbema_frontrun_vals = [0.002]  # Sadece 1 değer

    for rr, rsi, approach_tol, frontrun_margin in itertools.product(
        rr_vals, rsi_vals, pbema_approach_vals, pbema_frontrun_vals
    ):
        candidates.append(
            {
                "rr": round(float(rr), 2),
                "rsi": int(rsi),
                "slope": 0.5,
                "at_active": False,
                "use_trailing": False,
                "use_dynamic_pbema_tp": False,
                "strategy_mode": "pbema_reaction",
                "pbema_approach_tolerance": approach_tol,
                "pbema_frontrun_margin": frontrun_margin,
            }
        )

    # 1 trailing config ekle (sadece keltner bounce için)
    trailing_cfg = dict(candidates[0])
    trailing_cfg["use_trailing"] = True
    candidates.append(trailing_cfg)

    return candidates  # ~19 config (vs ~120)


def _get_min_trades_for_timeframe(tf: str, num_candles: int = 20000) -> int:
    """Return minimum trade count for statistical significance based on timeframe AND data size.

    The key insight: min_trades must scale with available data.
    - 20000 candles on 15m = ~208 days → can expect ~40 trades
    - 3000 candles on 15m = ~31 days → can only expect ~6 trades
    - 3000 candles on 5m = ~10 days → can only expect ~3 trades

    Formula: min_trades = base_rate * (num_candles / base_candles)
    Where base_rate is calibrated for 20000 candles.
    """
    # Base rates for 20000 candles (these are the "ideal" minimums)
    base_candles = 20000
    tf_base_rates = {
        "1m": 150,   # Very noisy
        "5m": 80,    # Noisy
        "15m": 35,   # Moderate
        "30m": 25,   # Less noise
        "1h": 20,    # Even less noise
        "4h": 15,    # Low noise
        "1d": 10,    # Very low noise
    }

    base_rate = tf_base_rates.get(tf, 30)

    # Scale min_trades proportionally to data size
    # But enforce absolute minimum of 5 trades (below this is pure noise)
    scaled_min = max(5, int(base_rate * (num_candles / base_candles)))

    # Cap at base_rate (don't require MORE trades for small data)
    return min(scaled_min, base_rate)


# ==========================================
# 🔬 WALK-FORWARD / OUT-OF-SAMPLE TESTING
# ==========================================
# Overfitting'i önlemek için:
# 1. Veriyi train (in-sample) ve test (out-of-sample) olarak böl
# 2. Train verisinde optimize et
# 3. Test verisinde sadece test et (tekrar optimize etme)
# 4. Test E[R] / Train E[R] oranı "overfit ratio"
# 5. Overfit ratio < 0.5 ise config reddedilir
# ==========================================

WALK_FORWARD_CONFIG = {
    "train_ratio": 0.70,        # %70 train, %30 test
    "min_test_trades": 3,       # Test için minimum trade sayısı
    "min_overfit_ratio": 0.50,  # Test E[R] / Train E[R] minimum oranı
    "enabled": True,            # Walk-forward testi etkinleştir
}


def _split_data_walk_forward(df: pd.DataFrame, train_ratio: float = 0.70) -> tuple:
    """Split data into train (in-sample) and test (out-of-sample) periods.

    Args:
        df: DataFrame with candle data (sorted by timestamp ascending)
        train_ratio: Ratio of data to use for training (default 0.70 = 70%)

    Returns:
        (train_df, test_df, oos_start_time): Tuple of DataFrames and OOS start timestamp
        oos_start_time is None if walk-forward is skipped
    """
    n = len(df)
    train_end = int(n * train_ratio)

    # Ensure we have enough data for both periods
    if train_end < 300 or (n - train_end) < 100:
        # Not enough data, return full df for both (skip walk-forward)
        return df, None, None

    train_df = df.iloc[:train_end].reset_index(drop=True)
    test_df = df.iloc[train_end:].reset_index(drop=True)

    # Get the OOS start timestamp for filtering backtest results
    oos_start_time = test_df['timestamp'].iloc[0] if 'timestamp' in test_df.columns else None

    return train_df, test_df, oos_start_time


def _validate_config_oos(df_test: pd.DataFrame, sym: str, tf: str, config: dict) -> dict:
    """Validate a config on out-of-sample test data.

    Args:
        df_test: Test DataFrame (out-of-sample data)
        sym: Symbol
        tf: Timeframe
        config: The optimized config to test

    Returns:
        dict with OOS metrics: {
            'oos_pnl': float,
            'oos_trades': int,
            'oos_expected_r': float,
            'oos_win_rate': float
        }
    """
    if df_test is None or len(df_test) < 100:
        return None

    try:
        net_pnl, trades, trade_pnls, trade_r_multiples = _score_config_for_stream(
            df_test, sym, tf, config
        )
    except Exception:
        return None

    if trades == 0:
        return {
            'oos_pnl': 0.0,
            'oos_trades': 0,
            'oos_expected_r': 0.0,
            'oos_win_rate': 0.0
        }

    expected_r = sum(trade_r_multiples) / len(trade_r_multiples) if trade_r_multiples else 0.0
    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = wins / trades if trades > 0 else 0.0

    return {
        'oos_pnl': net_pnl,
        'oos_trades': trades,
        'oos_expected_r': expected_r,
        'oos_win_rate': win_rate
    }


def _check_overfit(train_expected_r: float, oos_result: dict, tf: str) -> tuple:
    """Check if a config is overfitted by comparing train and OOS performance.

    Args:
        train_expected_r: E[R] from training data (in-sample)
        oos_result: Dict with OOS metrics from _validate_config_oos
        tf: Timeframe

    Returns:
        (is_overfit: bool, overfit_ratio: float, reason: str)
    """
    if oos_result is None:
        return False, 1.0, "no_oos_data"

    min_test_trades = WALK_FORWARD_CONFIG.get("min_test_trades", 3)
    min_overfit_ratio = WALK_FORWARD_CONFIG.get("min_overfit_ratio", 0.50)

    oos_trades = oos_result.get('oos_trades', 0)
    oos_expected_r = oos_result.get('oos_expected_r', 0.0)

    # Not enough OOS trades for statistical significance
    if oos_trades < min_test_trades:
        return False, 1.0, f"insufficient_oos_trades ({oos_trades}<{min_test_trades})"

    # OOS is negative = clear overfit
    if oos_expected_r < 0:
        return True, 0.0, "negative_oos_expected_r"

    # Calculate overfit ratio (how much of IS performance carries to OOS)
    if train_expected_r > 0:
        overfit_ratio = oos_expected_r / train_expected_r
    else:
        # Train was also bad, can't calculate ratio
        return False, 1.0, "train_not_positive"

    # Check if OOS degradation is too severe
    if overfit_ratio < min_overfit_ratio:
        return True, overfit_ratio, f"oos_degradation ({overfit_ratio:.2f}<{min_overfit_ratio})"

    return False, overfit_ratio, "ok"


def _compute_optimizer_score(net_pnl: float, trades: int, trade_pnls: list,
                              min_trades: int = 40, hard_min_trades: int = 5,
                              reject_negative_pnl: bool = True, tf: str = "15m",
                              trade_r_multiples: list = None) -> float:
    """Compute a robust optimizer score that penalizes overfitting.

    Uses a modified Sortino-like approach with aggressive anti-overfit measures:
    - HARD REJECT configs with negative net PnL (no edge = no config)
    - HARD REJECT configs with fewer than hard_min_trades
    - HARD REJECT configs with E[R] below MIN_EXPECTANCY_R_MULTIPLE threshold
    - Penalizes low trade counts (statistical insignificance)
    - Penalizes high variance in returns (inconsistent edge)
    - Penalizes extreme drawdowns
    - Rewards positive expectancy with consistency

    Args:
        net_pnl: Total net profit/loss
        trades: Number of trades
        trade_pnls: List of individual trade PnLs
        min_trades: Trades needed for full confidence (default 40)
        hard_min_trades: Absolute minimum trades, below this = reject (default 5)
        reject_negative_pnl: If True, reject configs with net_pnl <= 0 (default True)
        tf: Timeframe for looking up expectancy threshold
        trade_r_multiples: List of R-multiples per trade (PnL/Risk for each trade)

    Returns:
        Composite score (higher is better, -inf for rejected configs)
    """
    # HARD REJECT: Negative PnL = no edge, don't use this config
    if reject_negative_pnl and net_pnl <= 0:
        return -float("inf")

    # HARD REJECT: Too few trades = statistically meaningless
    if trades < hard_min_trades:
        return -float("inf")

    if trades == 0:
        return -float("inf")

    # HARD REJECT: E[R] too low = barely positive, not a real edge
    # R-Multiple based threshold (account-size independent)
    if trade_r_multiples and len(trade_r_multiples) > 0:
        expected_r = sum(trade_r_multiples) / len(trade_r_multiples)
        min_expected_r = MIN_EXPECTANCY_R_MULTIPLE.get(tf, 0.08)
        if expected_r < min_expected_r:
            return -float("inf")
    else:
        # Fallback to old $/trade method if R-multiples not provided
        expectancy = net_pnl / trades
        min_expectancy = MIN_EXPECTANCY_PER_TRADE.get(tf, 2.0)
        if expectancy < min_expectancy:
            return -float("inf")

    # Trade count confidence factor - MUCH more aggressive penalty for low counts
    # Uses logarithmic scaling: need ~min_trades to reach 90% confidence
    # 10 trades = ~50%, 20 trades = ~70%, 40 trades = ~90%, 60+ trades = ~100%
    if trades >= min_trades:
        trade_confidence = 1.0
    else:
        # Logarithmic penalty: confidence = 0.3 + 0.7 * log(trades) / log(min_trades)
        import math
        log_ratio = math.log(max(trades, 1)) / math.log(max(min_trades, 2))
        trade_confidence = max(0.2, min(0.95, 0.2 + 0.75 * log_ratio))

    # Average PnL per trade (expectancy)
    avg_pnl = net_pnl / trades

    # Downside deviation (Sortino-style) - only penalize negative variance
    if len(trade_pnls) >= 2:
        negative_pnls = [p for p in trade_pnls if p < 0]
        if negative_pnls:
            downside_std = (sum(p**2 for p in negative_pnls) / len(negative_pnls)) ** 0.5
        else:
            downside_std = 0.0

        # Consistency bonus: lower variance = more reliable edge
        if downside_std > 0 and avg_pnl != 0:
            consistency_ratio = abs(avg_pnl) / (downside_std + 1e-6)
            consistency_factor = min(1.5, 0.5 + consistency_ratio * 0.25)
        else:
            consistency_factor = 1.0 if avg_pnl > 0 else 0.5

        # NEW: Max drawdown penalty
        # Calculate running max drawdown from trade PnLs
        cumulative = 0
        peak = 0
        max_dd = 0
        for pnl in trade_pnls:
            cumulative += pnl
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        # Penalize if max drawdown exceeds 50% of total profit (or is large absolute)
        if net_pnl > 0 and max_dd > net_pnl * 0.5:
            dd_penalty = 0.7  # 30% penalty for excessive drawdown
        elif max_dd > 100:  # $100 absolute drawdown threshold
            dd_penalty = 0.85
        else:
            dd_penalty = 1.0
    else:
        consistency_factor = 0.5  # Single trade = very unreliable
        dd_penalty = 0.5  # Heavy penalty for single trade

    # Win rate bonus (slight preference for higher win rates at equal PnL)
    if trade_pnls:
        win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)
        # Penalize very low win rates even with good RR
        if win_rate < 0.25:
            win_rate_factor = 0.7  # Too low WR = high variance
        elif win_rate < 0.35:
            win_rate_factor = 0.85
        else:
            win_rate_factor = 0.9 + win_rate * 0.2  # 0.9 to 1.1
    else:
        win_rate_factor = 1.0

    # Note: Negative PnL configs are already hard-rejected above if reject_negative_pnl=True
    # This branch only executes for positive PnL configs

    # Final score = net_pnl * confidence * consistency * win_rate * dd_penalty
    score = net_pnl * trade_confidence * consistency_factor * win_rate_factor * dd_penalty

    return score


def _score_config_for_stream(df: pd.DataFrame, sym: str, tf: str, config: dict) -> Tuple[float, int, list]:
    """Simulate a single timeframe with the given config and return (net_pnl, trades, trade_pnls).

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

    # PERFORMANCE: Extract NumPy arrays once before the loop (10-50x faster than df.iloc[i])
    # Ensure timestamp is datetime64 (may become object after multiprocessing serialization)
    timestamps = pd.to_datetime(df["timestamp"]).values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    pb_tops = df.get("pb_ema_top", df["close"]).values if "pb_ema_top" in df.columns else closes
    pb_bots = df.get("pb_ema_bot", df["close"]).values if "pb_ema_bot" in df.columns else closes

    for i in range(warmup, end):
        event_time = pd.Timestamp(timestamps[i]) + _tf_to_timedelta(tf)
        tm.update_trades(
            sym,
            tf,
            candle_high=float(highs[i]),
            candle_low=float(lows[i]),
            candle_close=float(closes[i]),
            candle_time_utc=event_time,
            pb_top=float(pb_tops[i]),
            pb_bot=float(pb_bots[i]),
        )

        # Use wrapper function to support both keltner_bounce and pbema_reaction strategies
        s_type, s_entry, s_tp, s_sl, s_reason = TradingEngine.check_signal(
            df,
            config=config,
            index=i,
            return_debug=False,
        )

        if not (s_type and "ACCEPTED" in s_reason):
            continue

        has_open = any(
            t.get("symbol") == sym and t.get("timeframe") == tf for t in tm.open_trades
        )
        if has_open or tm.check_cooldown(sym, tf, event_time):
            continue

        # Access next candle for entry price (more realistic simulation)
        entry_open = float(opens[i + 1])
        open_ts = timestamps[i + 1]
        ts_str = (pd.Timestamp(open_ts) + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")

        # RR re-validation with actual entry price (next candle open)
        # Signal calculated RR based on close, but actual entry is at next open
        # This prevents edge degradation from slippage
        min_rr = config["rr"]
        if s_type == "LONG":
            actual_risk = entry_open - s_sl
            actual_reward = s_tp - entry_open
        else:  # SHORT
            actual_risk = s_sl - entry_open
            actual_reward = entry_open - s_tp

        if actual_risk <= 0 or actual_reward <= 0:
            continue  # Invalid RR after slippage

        actual_rr = actual_reward / actual_risk
        if actual_rr < min_rr * 0.9:  # 10% tolerance for slippage
            continue  # RR degraded too much after slippage

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
    trade_pnls = [t.get("pnl", 0.0) for t in tm.history] if tm.history else []
    # R-Multiple listesi (E[R] hesabı için)
    trade_r_multiples = tm.trade_r_multiples if hasattr(tm, 'trade_r_multiples') else []
    return tm.total_pnl, unique_trades, trade_pnls, trade_r_multiples


def _optimize_backtest_configs(
    streams: dict,
    requested_pairs: list,
    progress_callback=None,
    log_to_stdout: bool = True,
    use_walk_forward: bool = None,  # None = WALK_FORWARD_CONFIG["enabled"] kullan
    quick_mode: bool = False,  # True = azaltılmış config grid (daha hızlı)
):
    """Brute-force search to find the best config (by net pnl) per symbol/timeframe.

    Walk-forward validation enabled by default:
    - Splits data 70% train, 30% test
    - Optimizes on train data
    - Validates best config on test data
    - Rejects overfitted configs (OOS E[R] < 50% of train E[R])

    Quick mode (quick_mode=True):
    - Uses reduced config grid (24 instead of 120 configs)
    - ~5x faster optimization
    """

    def log(msg: str):
        if log_to_stdout:
            print(msg)
        if progress_callback:
            progress_callback(msg)

    # Walk-forward ayarı
    walk_forward_enabled = use_walk_forward if use_walk_forward is not None else WALK_FORWARD_CONFIG.get("enabled", True)

    # Quick mode: Azaltılmış config grid kullan
    if quick_mode:
        candidates = _generate_quick_candidate_configs()
        mode_str = "HIZLI"
    else:
        candidates = _generate_candidate_configs()
        mode_str = "TAM"

    total_jobs = len([1 for pair in requested_pairs if pair in streams]) * len(candidates)
    if total_jobs == 0:
        return {}

    best_by_pair = {}
    completed = 0
    next_progress = 5

    max_workers = max(1, (os.cpu_count() or 1) - 1)
    wf_status = "Açık" if walk_forward_enabled else "Kapalı"
    quick_icon = "⚡" if quick_mode else ""
    log(
        f"[OPT]{quick_icon} {len(candidates)} farklı ayar taranacak ({mode_str}). "
        f"Paralel: {max_workers}, Walk-Forward: {wf_status}"
    )

    for sym, tf in requested_pairs:
        if (sym, tf) not in streams:
            continue

        # Skip disabled symbol/timeframe combinations
        sym_cfg = SYMBOL_PARAMS.get(sym, {})
        tf_cfg = sym_cfg.get(tf, {}) if isinstance(sym_cfg, dict) else {}
        if tf_cfg.get("disabled", False):
            log(f"[OPT][{sym}-{tf}] Atlandı (disabled)")
            continue

        df_full = streams[(sym, tf)]
        num_candles_full = len(df_full)

        # Walk-forward: veriyi train/test olarak böl
        oos_start_time = None  # OOS başlangıç zamanı (backtest filtreleme için)
        if walk_forward_enabled:
            train_ratio = WALK_FORWARD_CONFIG.get("train_ratio", 0.70)
            df_train, df_test, oos_start_time = _split_data_walk_forward(df_full, train_ratio)
            if df_test is not None:
                log(f"[OPT][{sym}-{tf}] Walk-Forward: Train={len(df_train)} mum, Test={len(df_test)} mum")
            else:
                log(f"[OPT][{sym}-{tf}] Walk-Forward: Yetersiz veri, tam veri kullanılacak")
        else:
            df_train = df_full
            df_test = None

        df = df_train  # Optimizasyon için train verisini kullan
        num_candles = len(df)
        best_cfg = None
        best_score = -float("inf")
        best_pnl = -float("inf")
        best_trades = 0
        best_expected_r = 0.0  # E[R] değeri

        def handle_result(cfg, net_pnl, trades, trade_pnls, trade_r_multiples=None):
            nonlocal completed, next_progress, best_cfg, best_score, best_pnl, best_trades, best_expected_r

            completed += 1
            progress = (completed / total_jobs) * 100
            if progress >= next_progress:
                log(f"[OPT] %{progress:.1f} tamamlandı...")
                next_progress += 5

            if trades == 0:
                return

            # Use timeframe AND data-size aware min_trades for statistical significance
            tf_min_trades = _get_min_trades_for_timeframe(tf, num_candles)
            # Hard minimum: at least 3 trades (absolute floor)
            hard_min = max(3, tf_min_trades // 3)

            # Use the new composite score with anti-overfit measures
            # reject_negative_pnl=True means configs with net_pnl <= 0 are rejected
            # trade_r_multiples enables R-multiple based E[R] threshold
            score = _compute_optimizer_score(
                net_pnl, trades, trade_pnls,
                min_trades=tf_min_trades,
                hard_min_trades=hard_min,
                reject_negative_pnl=True,
                tf=tf,
                trade_r_multiples=trade_r_multiples
            )

            if score > best_score:
                best_score = score
                best_pnl = net_pnl
                best_cfg = cfg
                best_trades = trades
                # E[R] hesapla
                if trade_r_multiples and len(trade_r_multiples) > 0:
                    best_expected_r = sum(trade_r_multiples) / len(trade_r_multiples)
                else:
                    best_expected_r = 0.0

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_score_config_for_stream, df, sym, tf, cfg): cfg
                    for cfg in candidates
                }

                for future in as_completed(futures):
                    cfg = futures[future]
                    try:
                        net_pnl, trades, trade_pnls, trade_r_multiples = future.result()
                    except BrokenProcessPool:
                        # Havuza ait iş parçacığı çöktüyse seri moda düş.
                        raise
                    except Exception as exc:
                        log(f"[OPT][{sym}-{tf}] Skorlama hatası (cfg={cfg}): {exc}")
                        continue

                    handle_result(cfg, net_pnl, trades, trade_pnls, trade_r_multiples)

        except BrokenProcessPool as exc:
            log(
                f"[OPT][{sym}-{tf}] Paralel işleme havuzu durdu (neden: {exc}). "
                "Seri moda düşülüyor."
            )

            for cfg in candidates:
                try:
                    net_pnl, trades, trade_pnls, trade_r_multiples = _score_config_for_stream(df, sym, tf, cfg)
                except Exception as exc:
                    log(f"[OPT][{sym}-{tf}] Seri skorlama hatası (cfg={cfg}): {exc}")
                    continue

                handle_result(cfg, net_pnl, trades, trade_pnls, trade_r_multiples)

        # Get min_trades for this timeframe for logging (using actual num_candles)
        tf_min_trades = _get_min_trades_for_timeframe(tf, num_candles)
        hard_min = max(3, tf_min_trades // 3)
        min_score = MIN_SCORE_THRESHOLD.get(tf, 10.0)
        # Walk-Forward kullanılıyorsa, score eşiğini train_ratio ile ayarla
        # (Training seti daha küçük olduğu için daha az trade ve daha düşük score olur)
        if walk_forward_enabled:
            train_ratio = WALK_FORWARD_CONFIG.get("train_ratio", 0.70)
            min_score = min_score * train_ratio
        min_expected_r = MIN_EXPECTANCY_R_MULTIPLE.get(tf, 0.08)

        # Additional gate: score must meet minimum threshold for this timeframe
        if best_cfg and best_score < min_score:
            log(
                f"[OPT][{sym}-{tf}] Score ({best_score:.2f}) < min threshold ({min_score:.2f}) "
                f"→ Weak edge, DEVRE DIŞI"
            )
            best_cfg = None  # Reject weak-edge config

        # Walk-Forward OOS Validation
        oos_result = None
        overfit_ratio = 1.0
        if best_cfg and walk_forward_enabled and df_test is not None:
            log(f"[OPT][{sym}-{tf}] Walk-Forward: Out-of-Sample test yapılıyor...")
            oos_result = _validate_config_oos(df_test, sym, tf, best_cfg)

            if oos_result:
                is_overfit, overfit_ratio, overfit_reason = _check_overfit(
                    best_expected_r, oos_result, tf
                )

                oos_expected_r = oos_result.get('oos_expected_r', 0)
                oos_trades = oos_result.get('oos_trades', 0)
                oos_pnl = oos_result.get('oos_pnl', 0)

                if is_overfit:
                    log(
                        f"[OPT][{sym}-{tf}] ❌ OVERFIT TESPİT EDİLDİ! "
                        f"Train E[R]={best_expected_r:.3f}, OOS E[R]={oos_expected_r:.3f}, "
                        f"Ratio={overfit_ratio:.2f} | Sebep: {overfit_reason} → DEVRE DIŞI"
                    )
                    best_cfg = None  # Reject overfitted config
                else:
                    log(
                        f"[OPT][{sym}-{tf}] ✓ OOS Doğrulandı: "
                        f"Train E[R]={best_expected_r:.3f}, OOS E[R]={oos_expected_r:.3f}, "
                        f"OOS PnL=${oos_pnl:.2f}, OOS Trades={oos_trades}, "
                        f"Ratio={overfit_ratio:.2f}"
                    )

        if best_cfg:
            # Confidence level for risk multiplier (stored as string key)
            if best_trades >= tf_min_trades:
                confidence_level = "high"
                confidence_display = "✓ Yüksek"
            elif best_trades >= tf_min_trades * 0.6:
                confidence_level = "medium"
                confidence_display = "~ Orta"
            else:
                confidence_level = "low"
                confidence_display = "⚠ Düşük"

            # Low confidence = no trades (risk multiplier = 0)
            if confidence_level == "low":
                log(
                    f"[OPT][{sym}-{tf}] Düşük güven ({best_trades}/{tf_min_trades} trade) "
                    f"→ Risk çarpanı=0, DEVRE DIŞI"
                )
                best_by_pair[(sym, tf)] = {"disabled": True, "_reason": "low_confidence"}
            else:
                # Explicit disabled=False ensures stream is enabled even if previously disabled
                config_result = {
                    **best_cfg,
                    "disabled": False,  # Explicitly enable - overwrites any previous disabled state
                    "_net_pnl": best_pnl,
                    "_trades": best_trades,
                    "_score": best_score,
                    "_confidence": confidence_level,  # For risk multiplier
                    "_expectancy": best_pnl / best_trades if best_trades > 0 else 0,
                    "_expected_r": best_expected_r,  # E[R] - R-multiple bazlı expectancy (train)
                }

                # Walk-Forward OOS metrikleri ekle (varsa)
                if oos_result:
                    config_result["_oos_expected_r"] = oos_result.get('oos_expected_r', 0)
                    config_result["_oos_pnl"] = oos_result.get('oos_pnl', 0)
                    config_result["_oos_trades"] = oos_result.get('oos_trades', 0)
                    config_result["_overfit_ratio"] = overfit_ratio
                    config_result["_walk_forward_validated"] = True
                    # OOS başlangıç zamanını kaydet - backtest sonuçlarını filtrelemek için
                    if oos_start_time is not None:
                        config_result["_oos_start_time"] = oos_start_time
                else:
                    config_result["_walk_forward_validated"] = False

                best_by_pair[(sym, tf)] = config_result

                dyn_tp_str = "Açık" if best_cfg.get('use_dynamic_pbema_tp') else "Kapalı"
                risk_mult = CONFIDENCE_RISK_MULTIPLIER.get(confidence_level, 1.0)

                # OOS bilgisi log'a ekle (varsa)
                oos_info = ""
                if oos_result and overfit_ratio > 0:
                    oos_info = f", OOS Ratio={overfit_ratio:.2f}"

                log(
                    f"[OPT][{sym}-{tf}] En iyi ayar: RR={best_cfg['rr']}, RSI={best_cfg['rsi']}, "
                    f"Slope={best_cfg['slope']}, AT={'Açık' if best_cfg['at_active'] else 'Kapalı'}, "
                    f"DynTP={dyn_tp_str} | Net PnL=${best_pnl:.2f}, E[R]={best_expected_r:.3f}{oos_info}, "
                    f"Trades={best_trades}/{tf_min_trades}, Score={best_score:.2f}, "
                    f"Güven={confidence_display}, Risk={risk_mult:.0%}"
                )
        else:
            log(
                f"[OPT][{sym}-{tf}] Geçerli config bulunamadı - trade<{hard_min} veya PnL<=0 veya "
                f"E[R]<{min_expected_r:.2f} (min {tf_min_trades} trade) → DEVRE DIŞI"
            )
            # Mark this stream as disabled so backtest skips it
            best_by_pair[(sym, tf)] = {"disabled": True, "_reason": "no_positive_config"}

    log("[OPT] Tarama tamamlandı. Bulunan ayarlar backtest'e uygulanacak.")
    return best_by_pair


def _is_best_config_signature_valid(best_cfgs: dict) -> bool:
    """Ensure cached best configs belong to the current strategy signature."""

    global BEST_CONFIG_WARNING_FLAGS

    if not isinstance(best_cfgs, dict):
        return False

    meta = best_cfgs.get("_meta", {}) if isinstance(best_cfgs.get("_meta"), dict) else {}
    stored_sig = meta.get("strategy_signature")
    if not stored_sig:
        # Eski kayıtlar imzasız olabilir; uyumsuzluk riskini azaltmak için uyarı ver.
        if not BEST_CONFIG_WARNING_FLAGS.get("missing_signature", False):
            print("[CFG] Uyarı: Kaydedilmiş backtest imzası bulunamadı. En iyi ayarlar göz ardı edilecek.")
            BEST_CONFIG_WARNING_FLAGS["missing_signature"] = True
        return False

    current_sig = _strategy_signature()
    if stored_sig != current_sig:
        if not BEST_CONFIG_WARNING_FLAGS.get("signature_mismatch", False):
            print(
                "[CFG] Uyarı: Backtest ayar imzası mevcut stratejiyle eşleşmiyor. "
                "Canlı trade için varsayılan ayarlar kullanılacak; lütfen backtesti yeniden çalıştırın."
            )
            BEST_CONFIG_WARNING_FLAGS["signature_mismatch"] = True
        return False

    return True


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
    signature_ok = _is_best_config_signature_valid(best_cfgs)
    symbol_cfg = SYMBOL_PARAMS.get(symbol, {})
    tf_cfg = symbol_cfg.get(timeframe, {}) if isinstance(symbol_cfg, dict) else {}

    if isinstance(best_cfgs, dict) and signature_ok:
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

    global BEST_CONFIG_CACHE, BEST_CONFIG_WARNING_FLAGS
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

    cleaned["_meta"] = {
        "strategy_signature": _strategy_signature(),
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }

    BEST_CONFIG_CACHE = cleaned
    BEST_CONFIG_WARNING_FLAGS = {
        "missing_signature": False,
        "signature_mismatch": False,
    }
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
    def __init__(self, persist: bool = True, verbose: bool = True):
        self.persist = persist
        self.verbose = verbose

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
        self.risk_per_trade_pct = TRADING_CONFIG.get("risk_per_trade_pct", 0.01)
        self.max_portfolio_risk_pct = TRADING_CONFIG.get("max_portfolio_risk_pct", 0.03)
        # -----------------------------

        if self.persist:
            self.load_trades()
            # Startup'ta stale trade'leri temizle
            self.force_close_stale_trades()
        if self.verbose:
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

    def _calculate_portfolio_risk_pct(self, wallet_balance: float) -> float:
        if wallet_balance <= 0:
            return 0.0

        total_open_risk = 0.0
        for trade in self.open_trades:
            entry_price = float(trade.get("entry", 0.0))
            sl_price = float(trade.get("sl", entry_price))
            size = abs(float(trade.get("size", 0.0)))
            if entry_price <= 0 or size <= 0:
                continue
            sl_fraction = abs(entry_price - sl_price) / entry_price
            open_risk_amount = sl_fraction * size * entry_price
            total_open_risk += open_risk_amount

        return total_open_risk / wallet_balance

    def open_trade(self, signal_data):
        with self.lock:
            tf = signal_data["timeframe"]
            sym = signal_data["symbol"]

            cooldown_ref_time = signal_data.get("open_time_utc") or datetime.utcnow()
            if self.check_cooldown(sym, tf, cooldown_ref_time):
                return

            setup_type = signal_data.get("setup", "Unknown")

            # Trade açılırken config'i snapshot olarak al ve trade'e göm
            # Bu sayede trade yaşam döngüsü boyunca aynı kurallar kullanılır
            config_snapshot = load_optimized_config(sym, tf)
            use_trailing = config_snapshot.get("use_trailing", False)
            use_dynamic_pbema_tp = config_snapshot.get("use_dynamic_pbema_tp", True)
            opt_rr = config_snapshot.get("rr", 3.0)
            opt_rsi = config_snapshot.get("rsi", 60)

            # Confidence-based risk multiplier: reduce position size for medium confidence
            confidence_level = config_snapshot.get("_confidence", "high")
            risk_multiplier = CONFIDENCE_RISK_MULTIPLIER.get(confidence_level, 1.0)
            if risk_multiplier <= 0:
                print(f"⚠️ [{sym}-{tf}] Düşük güven seviyesi, işlem açılmadı.")
                return

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

            sl_price = float(signal_data["sl"])

            # Apply risk multiplier to effective risk per trade
            effective_risk_pct = self.risk_per_trade_pct * risk_multiplier
            current_portfolio_risk_pct = self._calculate_portfolio_risk_pct(self.wallet_balance)
            if current_portfolio_risk_pct + effective_risk_pct > self.max_portfolio_risk_pct:
                print(
                    f"⚠️ Portföy risk limiti aşılıyor: mevcut %{current_portfolio_risk_pct * 100:.2f}, "
                    f"yeni işlem riski %{effective_risk_pct * 100:.2f}, limit %{self.max_portfolio_risk_pct * 100:.2f}"
                )
                return

            wallet_balance = self.wallet_balance
            if wallet_balance <= 0:
                print("⚠️ Cüzdan bakiyesi 0 veya negatif, işlem açılamadı.")
                return

            risk_amount = wallet_balance * effective_risk_pct
            sl_distance = abs(real_entry - sl_price)
            if sl_distance <= 0:
                print("⚠️ Geçersiz SL mesafesi, işlem atlandı.")
                return

            sl_fraction = sl_distance / real_entry
            if sl_fraction <= 0:
                print("⚠️ Geçersiz SL oranı, işlem atlandı.")
                return

            position_notional = risk_amount / sl_fraction
            position_size = position_notional / real_entry

            leverage = TRADING_CONFIG["leverage"]
            required_margin = position_notional / leverage

            if required_margin > wallet_balance:
                max_notional = wallet_balance * leverage
                if max_notional <= 0:
                    print("⚠️ Yetersiz bakiye nedeniyle işlem açılamadı.")
                    return
                scale_factor = max_notional / position_notional
                position_notional = max_notional
                position_size = position_notional / real_entry
                required_margin = position_notional / leverage
                print(
                    f"⚠️ Gerekli marjin bakiyeyi aşıyor, pozisyon {scale_factor:.2f} oranında düşürüldü."
                )

            # open_time_utc'yi datetime string'e çevir (numpy.datetime64, pd.Timestamp veya datetime olabilir)
            _otv = signal_data.get("open_time_utc") or datetime.utcnow()
            if isinstance(_otv, np.datetime64):
                _otv = pd.Timestamp(_otv).to_pydatetime()
            elif isinstance(_otv, pd.Timestamp):
                _otv = _otv.to_pydatetime()

            new_trade = {
                "id": int(time.time() * 1000), "symbol": sym, "timestamp": signal_data["timestamp"],
                "open_time_utc": _otv.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "timeframe": tf, "type": trade_type, "setup": setup_type,
                "entry": real_entry,
                "tp": float(signal_data["tp"]), "sl": sl_price,
                "size": position_size, "margin": required_margin,
                "notional": position_notional, "events": [],
                "status": "OPEN", "pnl": 0.0,
                "breakeven": False, "trailing_active": False, "partial_taken": False, "partial_price": None,
                "has_cash": True, "close_time": "", "close_price": "",
                # Trade açılırken snapshot edilen config ayarları (yaşam döngüsü boyunca sabit kalır)
                "use_trailing": use_trailing,
                "use_dynamic_pbema_tp": use_dynamic_pbema_tp,
                "opt_rr": opt_rr,
                "opt_rsi": opt_rsi,
            }

            self.wallet_balance -= required_margin
            self.locked_margin += required_margin

            self.open_trades.append(new_trade)
            new_portfolio_risk_pct = self._calculate_portfolio_risk_pct(self.wallet_balance)

            print(
                f"📈 İşlem Açıldı | Entry: {real_entry:.4f}, SL: {sl_price:.4f}, "
                f"Size: {position_size:.6f}, Notional: ${position_notional:.2f}, "
                f"Margin: ${required_margin:.2f}, Risk%: {effective_risk_pct * 100:.2f}% "
                f"({confidence_level}), Risk$: ${risk_amount:.2f}, Portföy: {new_portfolio_risk_pct * 100:.2f}%"
            )

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

                # Config'i trade dict'inden oku (açılışta snapshot edildi)
                # Eski trade'ler için fallback olarak load_optimized_config kullan ve trade'e yaz
                if "use_trailing" in trade:
                    use_trailing = trade.get("use_trailing", False)
                    use_dynamic_tp = trade.get("use_dynamic_pbema_tp", True)
                else:
                    # Backward compatibility: eski trade'ler için config'den oku ve trade'e yaz
                    # Bu sayede sadece bir kere config'ten okunur, sonraki mumlarda trade'den okunur
                    config = load_optimized_config(symbol, tf)
                    use_trailing = config.get("use_trailing", False)
                    use_dynamic_tp = config.get("use_dynamic_pbema_tp", True)
                    # Trade'e yaz - sonraki mumlarda trade'den okunacak
                    self.open_trades[i]["use_trailing"] = use_trailing
                    self.open_trades[i]["use_dynamic_pbema_tp"] = use_dynamic_tp
                    self.open_trades[i]["opt_rr"] = config.get("rr", 3.0)
                    self.open_trades[i]["opt_rsi"] = config.get("rsi", 60)
                use_partial = not use_trailing

                # --- Fiyatlar ---
                # Partial TP için conservative fill hesaplaması
                # Ama progress için gerçek candle extreme kullanılmalı
                if t_type == "LONG":
                    close_price = candle_close
                    extreme_price = candle_high  # Progress için gerçek high
                    # Partial fill için conservative: 70% close + 30% extreme
                    partial_fill_price = close_price * 0.70 + extreme_price * 0.30
                    pnl_percent_close = (close_price - entry) / entry
                    in_profit = extreme_price > entry
                else:
                    close_price = candle_close
                    extreme_price = candle_low  # Progress için gerçek low
                    # Partial fill için conservative: 70% close + 30% extreme
                    partial_fill_price = close_price * 0.70 + extreme_price * 0.30
                    pnl_percent_close = (entry - close_price) / entry
                    in_profit = extreme_price < entry

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
                if t_type == "LONG":
                    live_pnl = (close_price - entry) * size
                else:
                    live_pnl = (entry - close_price) * size
                self.open_trades[i]["pnl"] = live_pnl

                # Hedefe ilerleme oranı (GERÇEK extreme'e göre, conservative değil)
                total_dist = abs(dyn_tp - entry)
                if total_dist <= 0:
                    continue
                current_dist = abs(extreme_price - entry)
                progress = current_dist / total_dist if total_dist > 0 else 0.0

                # ---------- PARTIAL TP + BREAKEVEN ----------
                if in_profit and use_partial:
                    if (not self.open_trades[i].get("partial_taken")) and progress >= 0.50:
                        partial_size = size / 2.0

                        # Partial fill için conservative fiyat kullan
                        if t_type == "LONG":
                            partial_fill = float(partial_fill_price) * (1 - self.slippage_pct)
                            partial_pnl_percent = (partial_fill - entry) / entry
                        else:
                            partial_fill = float(partial_fill_price) * (1 + self.slippage_pct)
                            partial_pnl_percent = (entry - partial_fill) / entry

                        partial_pnl = partial_pnl_percent * (entry * partial_size)
                        partial_notional = abs(partial_size) * abs(partial_fill)
                        commission = partial_notional * TRADING_CONFIG["total_fee"]
                        net_partial_pnl = partial_pnl - commission
                        margin_release = initial_margin / 2.0

                        self.wallet_balance += margin_release + net_partial_pnl
                        self.locked_margin -= margin_release
                        self.total_pnl += net_partial_pnl

                        partial_record = trade.copy()
                        partial_record["size"] = partial_size
                        partial_record["notional"] = partial_notional
                        partial_record["pnl"] = net_partial_pnl
                        partial_record["status"] = "PARTIAL TP (50%)"
                        partial_record["close_time"] = (pd.Timestamp(candle_time_utc) + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
                        partial_record["close_price"] = float(partial_fill)
                        partial_record["pb_ema_top"] = pb_top
                        partial_record["pb_ema_bot"] = pb_bot
                        partial_record["events"] = json.dumps(self.open_trades[i].get("events", []))
                        self.history.append(partial_record)

                        # Açık trade'i güncelle: yarı pozisyon kaldı, margin yarıya indi
                        self.open_trades[i]["size"] = partial_size
                        self.open_trades[i]["notional"] = partial_notional
                        self.open_trades[i]["margin"] = margin_release
                        self.open_trades[i]["partial_price"] = float(partial_fill)
                        self.open_trades[i]["partial_taken"] = True
                        _append_trade_event(self.open_trades[i], "PARTIAL", candle_time_utc, partial_fill)
                        # Breakeven'e çek - küçük buffer ile (spread/slippage koruması)
                        be_buffer = 0.0003  # %0.03 buffer
                        if t_type == "LONG":
                            be_sl = entry * (1 + be_buffer)
                        else:
                            be_sl = entry * (1 - be_buffer)
                        self.open_trades[i]["sl"] = be_sl
                        self.open_trades[i]["breakeven"] = True
                        _append_trade_event(self.open_trades[i], "BE_SET", candle_time_utc, be_sl)
                        trades_updated = True

                    elif (not trade.get("breakeven")) and progress >= 0.40:
                        # Breakeven'e çek - küçük buffer ile
                        be_buffer = 0.0003  # %0.03 buffer
                        if t_type == "LONG":
                            be_sl = entry * (1 + be_buffer)
                        else:
                            be_sl = entry * (1 - be_buffer)
                        self.open_trades[i]["sl"] = be_sl
                        self.open_trades[i]["breakeven"] = True
                        _append_trade_event(self.open_trades[i], "BE_SET", candle_time_utc, be_sl)
                        trades_updated = True

                # 1m için fiyat TP'ye çok yaklaşınca SL'i kâra çek
                if _apply_1m_profit_lock(self.open_trades[i], tf, t_type, entry, dyn_tp, progress):
                    _append_trade_event(self.open_trades[i], "PROFIT_LOCK", candle_time_utc, self.open_trades[i].get("sl"))
                    trades_updated = True

                # ---------- TRAILING SL ----------
                if in_profit and use_trailing:
                    if (not trade.get("breakeven")) and progress >= 0.40:
                        # Breakeven'e çek - küçük buffer ile
                        be_buffer = 0.0003  # %0.03 buffer
                        if t_type == "LONG":
                            be_sl = entry * (1 + be_buffer)
                        else:
                            be_sl = entry * (1 - be_buffer)
                        self.open_trades[i]["sl"] = be_sl
                        self.open_trades[i]["breakeven"] = True
                        _append_trade_event(self.open_trades[i], "BE_SET", candle_time_utc, be_sl)
                        trades_updated = True

                    if progress >= 0.50:
                        trail_buffer = total_dist * 0.40
                        current_sl = float(self.open_trades[i]["sl"])
                        if t_type == "LONG":
                            new_sl = close_price - trail_buffer
                            if new_sl > current_sl:
                                self.open_trades[i]["sl"] = new_sl
                                self.open_trades[i]["trailing_active"] = True
                                _append_trade_event(self.open_trades[i], "TRAIL_SL", candle_time_utc, new_sl)
                                trades_updated = True
                        else:
                            new_sl = close_price + trail_buffer
                            if new_sl < current_sl:
                                self.open_trades[i]["sl"] = new_sl
                                self.open_trades[i]["trailing_active"] = True
                                _append_trade_event(self.open_trades[i], "TRAIL_SL", candle_time_utc, new_sl)
                                trades_updated = True

                # ---------- SL / TP KONTROLÜ ----------
                if _apply_partial_stop_protection(self.open_trades[i], tf, progress, t_type):
                    _append_trade_event(self.open_trades[i], "STOP_PROTECTION", candle_time_utc, self.open_trades[i].get("sl"))
                    trades_updated = True

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
                    gross_pnl = (exit_fill - entry) * current_size
                else:
                    exit_fill = float(exit_level) * (1 + self.slippage_pct)
                    gross_pnl = (entry - exit_fill) * current_size

                commission_notional = abs(current_size) * abs(exit_fill)
                commission = commission_notional * TRADING_CONFIG["total_fee"]

                funding_cost = 0.0
                try:
                    open_time_str = trade.get("open_time_utc", "")
                    if open_time_str:
                        open_dt = datetime.strptime(open_time_str, "%Y-%m-%dT%H:%M:%SZ")
                        hours = max(0.0, (candle_time_utc - open_dt).total_seconds() / 3600.0)
                        notional_entry = abs(current_size) * entry
                        funding_cost = notional_entry * TRADING_CONFIG["funding_rate_8h"] * (hours / 8.0)
                except Exception:
                    funding_cost = 0.0

                final_net_pnl = gross_pnl - commission - funding_cost

                self.wallet_balance += margin_release + final_net_pnl
                self.locked_margin -= margin_release
                self.total_pnl += final_net_pnl

                # Cooldown sadece gerçek STOP durumunda
                if "STOP" in reason:
                    wait_minutes = 10 if tf == "1m" else (30 if tf == "5m" else 60)

                    cooldown_base = pd.Timestamp(candle_time_utc)
                    self.cooldowns[(symbol, tf)] = cooldown_base + pd.Timedelta(minutes=wait_minutes)

                # BE statüsünü ayır
                if trade.get("breakeven") and abs(final_net_pnl) < 1e-6 and "STOP" in reason:
                    reason = "BE"

                trade["status"] = reason
                trade["pnl"] = final_net_pnl
                trade["close_time"] = (pd.Timestamp(candle_time_utc) + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
                trade["close_price"] = float(exit_fill)
                trade["pb_ema_top"] = pb_top
                trade["pb_ema_bot"] = pb_bot

                serialized_trade = trade.copy()
                serialized_trade["events"] = json.dumps(trade.get("events", []))

                self.history.append(serialized_trade)
                just_closed_trades.append(serialized_trade)
                closed_indices.append(i)
                trades_updated = True

            for idx in sorted(closed_indices, reverse=True):
                del self.open_trades[idx]

            if trades_updated:
                self.save_trades()

            return just_closed_trades

    def update_live_pnl_with_price(self, symbol: str, latest_price: float):
        """Update open trade PnL values using the most recent price tick.

        This is a lightweight calculation that keeps the UI-sensitive PnL column
        in sync with faster price updates without triggering trade lifecycle
        changes or disk writes. It intentionally avoids modifying SL/TP logic.
        """
        with self.lock:
            for i, trade in enumerate(self.open_trades):
                if trade.get("symbol") != symbol:
                    continue

                entry = float(trade.get("entry", 0))
                size = float(trade.get("size", 0))
                if entry <= 0 or size == 0:
                    continue

                if trade.get("type") == "LONG":
                    gross_pnl = (latest_price - entry) * size
                else:
                    gross_pnl = (entry - latest_price) * size

                self.open_trades[i]["pnl"] = gross_pnl

    def save_trades(self):
        if not self.persist:
            return
        with self.lock:
            try:
                cols = [
                    "id", "symbol", "timestamp", "timeframe", "type", "setup", "entry", "tp", "sl", "size",
                    "margin", "notional",
                    "status", "pnl", "breakeven", "trailing_active", "partial_taken", "stop_protection", "has_cash",
                    "close_time", "close_price", "events"
                ]

                if not self.open_trades and not self.history:
                    df_all = pd.DataFrame(columns=cols)
                else:
                    df_all = pd.concat([pd.DataFrame(self.open_trades), pd.DataFrame(self.history)], ignore_index=True)

                for c in cols:
                    if c not in df_all.columns:
                        df_all[c] = ""

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
        if not self.persist:
            return
        with self.lock:
            self.wallet_balance = TRADING_CONFIG["initial_balance"]
            self.locked_margin = 0.0
            self.total_pnl = 0.0

            if os.path.exists(CSV_FILE):
                try:
                    if os.path.getsize(CSV_FILE) == 0:
                        return
                    df = pd.read_csv(CSV_FILE)
                    if "symbol" in df.columns:
                        self.open_trades = df[df["status"].astype(str).str.contains("OPEN")].to_dict('records')
                        self.history = df[~df["status"].astype(str).str.contains("OPEN")].to_dict('records')

                        for trade in self.history:
                            self.total_pnl += float(trade['pnl'])

                        open_pnl = 0.0
                        for trade in self.open_trades:
                            m = float(trade.get('margin', float(trade['size']) / TRADING_CONFIG["leverage"]))
                            if not trade.get('notional'):
                                try:
                                    trade['notional'] = float(trade.get('entry', 0)) * float(trade.get('size', 0))
                                except Exception:
                                    trade['notional'] = 0.0
                            events_val = trade.get("events")
                            if isinstance(events_val, str):
                                try:
                                    trade["events"] = json.loads(events_val)
                                except Exception:
                                    trade["events"] = []
                            self.locked_margin += m
                            open_pnl += float(trade.get('pnl', 0.0))

                        # Kullanılabilir bakiye = başlangıç + kapalı işlemlerden net PnL - kilitli marj
                        self.wallet_balance = TRADING_CONFIG["initial_balance"] + self.total_pnl - self.locked_margin
                        total_equity = self.wallet_balance + self.locked_margin + open_pnl
                        if self.verbose:
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

    def force_close_stale_trades(self):
        """Force-close trades that should have been closed but weren't due to code changes.

        Fetches current market price and checks TP/SL conditions immediately.
        Called on startup to clean up any stale trades.
        """
        if not self.open_trades:
            return

        print("[CLEANUP] Açık trade'ler kontrol ediliyor...")

        with self.lock:
            closed_indices = []

            for i, trade in enumerate(self.open_trades):
                symbol = trade.get("symbol")
                tf = trade.get("timeframe")
                t_type = trade.get("type")
                entry = float(trade.get("entry", 0))
                tp = float(trade.get("tp", 0))
                sl = float(trade.get("sl", 0))

                if not all([symbol, t_type, entry, tp, sl]):
                    continue

                # Breakeven trade kontrolü: SL entry'ye çok yakınsa (breakeven durumu)
                # bu trade'i hemen kapatma, normal update döngüsünde işlensin
                sl_entry_diff_pct = abs(sl - entry) / entry
                is_breakeven_trade = sl_entry_diff_pct < 0.001  # %0.1'den küçükse breakeven
                if is_breakeven_trade:
                    print(f"[CLEANUP] {symbol}-{tf} breakeven trade, normal döngüde işlenecek (SL: {sl:.4f}, Entry: {entry:.4f})")
                    continue

                # Fetch current price
                try:
                    url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
                    resp = requests.get(url, timeout=5)
                    current_price = float(resp.json()["price"])
                except Exception as e:
                    print(f"[CLEANUP] {symbol} fiyat alınamadı: {e}")
                    continue

                # Check TP/SL conditions
                should_close = False
                reason = ""

                if t_type == "LONG":
                    if current_price >= tp:
                        should_close = True
                        reason = "WIN (TP) [Stale]"
                    elif current_price <= sl:
                        should_close = True
                        reason = "STOP [Stale]"
                else:  # SHORT
                    if current_price <= tp:
                        should_close = True
                        reason = "WIN (TP) [Stale]"
                    elif current_price >= sl:
                        should_close = True
                        reason = "STOP [Stale]"

                if should_close:
                    print(f"[CLEANUP] {symbol}-{tf} {t_type} kapatılıyor: {reason} (Fiyat: {current_price:.4f}, TP: {tp:.4f}, SL: {sl:.4f})")

                    # Calculate PnL
                    size = float(trade.get("size", 0))
                    if t_type == "LONG":
                        exit_fill = current_price * (1 - self.slippage_pct)
                        gross_pnl = (exit_fill - entry) * size
                    else:
                        exit_fill = current_price * (1 + self.slippage_pct)
                        gross_pnl = (entry - exit_fill) * size

                    commission = abs(size * exit_fill) * TRADING_CONFIG["total_fee"]
                    net_pnl = gross_pnl - commission

                    # Release margin
                    margin = float(trade.get("margin", size / TRADING_CONFIG["leverage"]))
                    self.wallet_balance += margin + net_pnl
                    self.locked_margin -= margin
                    self.total_pnl += net_pnl

                    # Update trade record
                    trade["status"] = reason
                    trade["pnl"] = net_pnl
                    trade["close_price"] = exit_fill
                    trade["close_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

                    self.history.append(trade)
                    closed_indices.append(i)

            # Remove closed trades
            for idx in sorted(closed_indices, reverse=True):
                self.open_trades.pop(idx)

            if closed_indices:
                self.save_trades()
                print(f"[CLEANUP] {len(closed_indices)} stale trade kapatıldı.")
            else:
                print("[CLEANUP] Kapatılması gereken stale trade bulunamadı.")

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


class PotentialTradeRecorder:
    def __init__(self, max_entries: int = 500):
        self.max_entries = max_entries
        self.lock = threading.Lock()
        self.entries = []

    def add(self, entry: dict):
        if not isinstance(entry, dict):
            return
        with self.lock:
            self.entries.append(entry)
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]

    def get_all(self):
        with self.lock:
            return list(self.entries)

    def clear(self):
        with self.lock:
            self.entries.clear()


potential_trades = PotentialTradeRecorder()
trade_manager = TradeManager()


# --- TRADING ENGINE (ROBUST API & RETRY MECHANISM) ---
class TradingEngine:
    # Ağ çökmelerinde tekrar tekrar DNS denemelerini önlemek için kısa süreli kilit
    _network_cooldown_until = 0

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

        # DNS çözememe gibi hatalar tekrar denense de çözülmeyecekse boşuna istek atma
        now = time.time()
        if now < TradingEngine._network_cooldown_until:
            cooldown_left = int(TradingEngine._network_cooldown_until - now)
            print(f"BAĞLANTI HATASI: Ağ erişimi yok. {cooldown_left}s sonra yeniden denenecek.")
            return None

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
                TradingEngine._network_cooldown_until = 0
                return res
            except requests.exceptions.RequestException as e:
                is_dns_error = isinstance(e, requests.exceptions.ConnectionError) and "NameResolutionError" in str(e)
                print(f"BAĞLANTI HATASI (Deneme {attempt + 1}/{max_retries}): {e}")

                # DNS çözememe (getaddrinfo) durumunda 5 dakikalığına istekleri durdur
                if is_dns_error:
                    TradingEngine._network_cooldown_until = time.time() + 300
                    break

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

        PERFORMANCE NOTE: This function modifies the DataFrame in-place by adding indicator columns.
        If you need to preserve the original DataFrame, make a copy before calling this function.
        """
        # PERFORMANCE: Removed df.copy() - function now modifies in-place (20-30% faster)
        # Callers that need the original should copy before calling

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
            return_debug: bool = False,
    ) -> Tuple:
        """
        Base Setup için LONG / SHORT sinyali üretir.

        Filtreler:
        - ADX düşükse alma
        - Keltner holding + retest
        - PBEMA cloud hizalaması
        - Keltner bandı ile PBEMA TP hedefi arasında minimum mesafe
        - TP çok yakın / çok uzak değil
        - RR >= min_rr   (RR = reward / risk)

        Not: Bu kurgu trend-takip eden değil, PBEMA bulutunu mıknatıs gibi
        kullanan mean reversion yaklaşımıdır; Keltner dokunuşları hem üstten
        SHORT hem alttan LONG için tetikleyici olabilir.
        """

        debug_info = {
            "adx_ok": None,
            "trend_up_strong": None,
            "trend_down_strong": None,
            "holding_long": None,
            "retest_long": None,
            "pb_target_long": None,
            "long_rsi_ok": None,
            "holding_short": None,
            "retest_short": None,
            "pb_target_short": None,
            "short_rsi_ok": None,
            "tp_dist_ratio": None,
            "rr_value": None,
            "long_rr_ok": None,
            "short_rr_ok": None,
        }

        def _ret(s_type, entry, tp, sl, reason):
            if return_debug:
                return s_type, entry, tp, sl, reason, debug_info
            return s_type, entry, tp, sl, reason

        if df is None or df.empty:
            return _ret(None, None, None, None, "No Data")

        required_cols = [
            "open", "high", "low", "close",
            "rsi", "adx",
            "pb_ema_top", "pb_ema_bot",
            "keltner_upper", "keltner_lower",
        ]
        for col in required_cols:
            if col not in df.columns:
                return _ret(None, None, None, None, f"Missing {col}")

        try:
            curr = df.iloc[index]
        except Exception:
            return _ret(None, None, None, None, "Index Error")

        for c in required_cols:
            v = curr.get(c)
            if pd.isna(v):
                return _ret(None, None, None, None, f"NaN in {c}")

        abs_index = index if index >= 0 else (len(df) + index)
        if abs_index < 0 or abs_index >= len(df):
            return _ret(None, None, None, None, "Index Out of Range")

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
        debug_info["adx_ok"] = float(curr["adx"]) >= adx_min
        if not debug_info["adx_ok"]:
            return _ret(None, None, None, None, "ADX Low")

        if abs_index < hold_n + 1:
            return _ret(None, None, None, None, "Warmup")

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

        # --- Mean reversion: Slope filter DEVRE DIŞI ---
        # PBEMA (200 EMA) çok yavaş hareket eder:
        # - Slope değişmesini beklemek = trade kaçırmak
        # - Mean reversion'da fiyat ortalamaya ÇEKİLİR
        # - Slope değişene kadar hareketin çoğu bitmiş olur
        # Sonuç: PBEMA için slope filter YANLIŞ yaklaşım
        slope_top = float(curr.get("slope_top", 0.0) or 0.0)
        slope_bot = float(curr.get("slope_bot", 0.0) or 0.0)

        # Sadece debug için tut, FİLTRELEME YAPMA
        debug_info["slope_top"] = slope_top
        debug_info["slope_bot"] = slope_bot

        # Mean reversion = yön kısıtı YOK
        long_direction_ok = True
        short_direction_ok = True
        debug_info["long_direction_ok"] = long_direction_ok
        debug_info["short_direction_ok"] = short_direction_ok

        # ================= WICK REJECTION QUALITY =================
        candle_range = high - low
        if candle_range > 0:
            upper_wick = high - max(open_, close)
            lower_wick = min(open_, close) - low
            upper_wick_ratio = upper_wick / candle_range
            lower_wick_ratio = lower_wick / candle_range
        else:
            upper_wick_ratio = 0.0
            lower_wick_ratio = 0.0

        # Minimum wick ratio for quality rejection (0.15 = 15% of candle is wick)
        # Lowered from 0.3 to 0.15 for softer filtering
        min_wick_ratio = 0.15
        long_rejection_quality = lower_wick_ratio >= min_wick_ratio
        short_rejection_quality = upper_wick_ratio >= min_wick_ratio

        debug_info.update({
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "long_rejection_quality": long_rejection_quality,
            "short_rejection_quality": short_rejection_quality,
        })

        # ================= PRICE-PBEMA POSITION CHECK =================
        # LONG: Fiyat PBEMA'nın ALTINDA olmalı (PBEMA üstte/kırmızı)
        # SHORT: Fiyat PBEMA'nın ÜSTÜNDE olmalı (PBEMA altta/yeşil)
        price_below_pbema = close < pb_bot
        price_above_pbema = close > pb_top

        debug_info.update({
            "price_below_pbema": price_below_pbema,
            "price_above_pbema": price_above_pbema,
        })

        # ================= KELTNER PENETRATION (TRAP) DETECTION =================
        # Son N mumda Keltner'ı aşıp geri dönen mum var mı? (daha güçlü sinyal)
        penetration_lookback = min(3, len(df) - abs_index - 1) if abs_index < len(df) - 1 else 0

        # Long: Son mumlarda alt Keltner'ın altına inip geri dönen var mı?
        long_penetration = False
        if penetration_lookback > 0:
            for i in range(1, penetration_lookback + 1):
                if abs_index - i >= 0:
                    past_low = float(df["low"].iloc[abs_index - i])
                    past_lower_band = float(df["keltner_lower"].iloc[abs_index - i])
                    if past_low < past_lower_band:
                        long_penetration = True
                        break

        # Short: Son mumlarda üst Keltner'ın üstüne çıkıp geri dönen var mı?
        short_penetration = False
        if penetration_lookback > 0:
            for i in range(1, penetration_lookback + 1):
                if abs_index - i >= 0:
                    past_high = float(df["high"].iloc[abs_index - i])
                    past_upper_band = float(df["keltner_upper"].iloc[abs_index - i])
                    if past_high > past_upper_band:
                        short_penetration = True
                        break

        debug_info.update({
            "long_penetration": long_penetration,
            "short_penetration": short_penetration,
        })

        # ================= LONG =================
        holding_long = (closes_slice > lower_slice).mean() >= min_hold_frac

        retest_long = (
                (low <= lower_band * (1 + touch_tol))
                and (close > lower_band)
                and (min(open_, close) > lower_band * (1 - body_tol))
        )

        keltner_pb_gap_long = (pb_bot - lower_band) / lower_band if lower_band != 0 else 0.0

        pb_target_long = (
                long_direction_ok and
                (keltner_pb_gap_long >= cloud_keltner_gap_min)
        )

        # Quality indicators (for analysis, NOT mandatory filters)
        # These are kept for debugging and future optimization
        long_quality_ok = long_rejection_quality or long_penetration

        # SOFT VERSION: Only core filters are mandatory
        # price_below_pbema and quality_ok are tracked but NOT required
        is_long = holding_long and retest_long and pb_target_long
        debug_info.update({
            "holding_long": holding_long,
            "retest_long": retest_long,
            "pb_target_long": pb_target_long,
            "long_quality_ok": long_quality_ok,
            "price_below_pbema": price_below_pbema,
        })

        # ================= SHORT =================
        holding_short = (closes_slice < upper_slice).mean() >= min_hold_frac

        retest_short = (
                (high >= upper_band * (1 - touch_tol))
                and (close < upper_band)
                and (max(open_, close) < upper_band * (1 + body_tol))
        )

        keltner_pb_gap_short = (upper_band - pb_top) / upper_band if upper_band != 0 else 0.0

        pb_target_short = (
                short_direction_ok and
                (keltner_pb_gap_short >= cloud_keltner_gap_min)
        )

        # Quality indicators (for analysis, NOT mandatory filters)
        # These are kept for debugging and future optimization
        short_quality_ok = short_rejection_quality or short_penetration

        # SOFT VERSION: Only core filters are mandatory
        # price_above_pbema and quality_ok are tracked but NOT required
        is_short = holding_short and retest_short and pb_target_short
        debug_info.update({
            "holding_short": holding_short,
            "retest_short": retest_short,
            "pb_target_short": pb_target_short,
            "short_quality_ok": short_quality_ok,
            "price_above_pbema": price_above_pbema,
        })

        # --- RSI filters (symmetric for LONG and SHORT) ---
        rsi_val = float(curr["rsi"])
        debug_info["rsi_value"] = rsi_val

        # LONG: RSI should not be too high (overbought territory)
        # Using rsi_limit + 10 as upper bound
        long_rsi_limit = rsi_limit + 10.0
        long_rsi_ok = rsi_val <= long_rsi_limit
        debug_info["long_rsi_ok"] = long_rsi_ok
        debug_info["long_rsi_limit"] = long_rsi_limit
        if is_long and not long_rsi_ok:
            is_long = False

        # SHORT: RSI should not be too low (oversold territory)
        # Mirror of long: 100 - (rsi_limit + 10) as lower bound
        # If rsi_limit=45, long_upper=55, short_lower=100-55=45
        short_rsi_limit = 100.0 - long_rsi_limit
        short_rsi_ok = rsi_val >= short_rsi_limit
        debug_info["short_rsi_ok"] = short_rsi_ok
        debug_info["short_rsi_limit"] = short_rsi_limit
        if is_short and not short_rsi_ok:
            is_short = False

        # --- AlphaTrend (opsiyonel) ---
        if use_alphatrend and "alphatrend" in df.columns:
            at_val = float(curr["alphatrend"])
            if is_long and close < at_val:
                is_long = False
            if is_short and close > at_val:
                is_short = False

        # ---------- LONG ----------
        debug_info["long_candidate"] = is_long
        if is_long:
            swing_n = 20
            start = max(0, abs_index - swing_n)
            swing_low = float(df["low"].iloc[start:abs_index].min())
            if swing_low <= 0:
                return _ret(None, None, None, None, "Invalid Swing Low")

            sl_candidate = swing_low * 0.997
            band_sl = lower_band * 0.998
            sl = min(sl_candidate, band_sl)

            entry = close
            tp = pb_bot

            if tp <= entry:
                return _ret(None, None, None, None, "TP Below Entry")
            if sl >= entry:
                sl = min(swing_low * 0.995, entry * 0.997)

            risk = entry - sl
            reward = tp - entry
            if risk <= 0 or reward <= 0:
                return _ret(None, None, None, None, "Invalid RR")

            rr = reward / risk
            tp_dist_ratio = reward / entry

            debug_info.update({
                "tp_dist_ratio": tp_dist_ratio,
                "rr_value": rr,
                "long_rr_ok": rr >= min_rr,
            })

            if tp_dist_ratio < tp_min_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
            if tp_dist_ratio > tp_max_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
            if rr < min_rr:
                return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

            reason = f"ACCEPTED(Base,R:{rr:.2f})"
            return _ret("LONG", entry, tp, sl, reason)

        # ---------- SHORT ----------
        debug_info["short_candidate"] = is_short
        if is_short:
            swing_n = 20
            start = max(0, abs_index - swing_n)
            swing_high = float(df["high"].iloc[start:abs_index].max())
            if swing_high <= 0:
                return _ret(None, None, None, None, "Invalid Swing High")

            sl_candidate = swing_high * 1.003
            band_sl = upper_band * 1.002
            sl = max(sl_candidate, band_sl)

            entry = close
            tp = pb_top

            if tp >= entry:
                return _ret(None, None, None, None, "TP Above Entry")
            if sl <= entry:
                sl = max(swing_high * 1.005, entry * 1.003)

            risk = sl - entry
            reward = entry - tp
            if risk <= 0 or reward <= 0:
                return _ret(None, None, None, None, "Invalid RR")

            rr = reward / risk
            tp_dist_ratio = reward / entry

            debug_info.update({
                "tp_dist_ratio": tp_dist_ratio,
                "rr_value": rr,
                "short_rr_ok": rr >= min_rr,
            })

            if tp_dist_ratio < tp_min_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
            if tp_dist_ratio > tp_max_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
            if rr < min_rr:
                return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

            reason = f"ACCEPTED(Base,R:{rr:.2f})"
            return _ret("SHORT", entry, tp, sl, reason)

        return _ret(None, None, None, None, "No Signal")

    @staticmethod
    def check_pbema_reaction_signal(
            df: pd.DataFrame,
            index: int = -2,
            min_rr: float = 2.0,
            rsi_limit: float = 60.0,
            slope_thresh: float = 0.5,
            use_alphatrend: bool = False,
            pbema_approach_tolerance: float = 0.003,
            pbema_frontrun_margin: float = 0.002,
            tp_min_dist_ratio: float = 0.0015,
            tp_max_dist_ratio: float = 0.04,
            adx_min: float = 8.0,
            return_debug: bool = False,
    ) -> Tuple:
        """
        PBEMA Reaction Strategy - Fiyat PBEMA bulutuna yaklaştığında/değdiğinde işlem aç.

        Konsept:
        - PBEMA bulutu güçlü bir destek/direnç bölgesi olarak çalışıyor
        - Fiyat PBEMA'ya yaklaştığında oradan tepki alması bekleniyor
        - SHORT: Fiyat aşağıdan PBEMA'ya yaklaşıyor → satış baskısı bekleniyor
        - LONG: Fiyat yukarıdan PBEMA'ya yaklaşıyor → alış baskısı bekleniyor

        Parametreler:
        - pbema_approach_tolerance: PBEMA'ya ne kadar yaklaşınca sinyal üret (örn. 0.003 = %0.3)
        - pbema_frontrun_margin: Frontrun marjı (SL için PBEMA + bu marj)
        """

        debug_info = {
            "adx_ok": None,
            "price_near_pbema_top": None,
            "price_near_pbema_bot": None,
            "approaching_from_below": None,
            "approaching_from_above": None,
            "short_rsi_ok": None,
            "long_rsi_ok": None,
            "tp_dist_ratio": None,
            "rr_value": None,
        }

        def _ret(s_type, entry, tp, sl, reason):
            if return_debug:
                return s_type, entry, tp, sl, reason, debug_info
            return s_type, entry, tp, sl, reason

        if df is None or df.empty:
            return _ret(None, None, None, None, "No Data")

        required_cols = [
            "open", "high", "low", "close",
            "rsi", "adx",
            "pb_ema_top", "pb_ema_bot",
            "keltner_upper", "keltner_lower",
        ]
        for col in required_cols:
            if col not in df.columns:
                return _ret(None, None, None, None, f"Missing {col}")

        try:
            curr = df.iloc[index]
        except Exception:
            return _ret(None, None, None, None, "Index Error")

        abs_index = index if index >= 0 else (len(df) + index)
        if abs_index < 30:  # Need enough history for swing detection
            return _ret(None, None, None, None, "Not Enough Data")

        # Extract current values
        open_ = float(curr["open"])
        high = float(curr["high"])
        low = float(curr["low"])
        close = float(curr["close"])
        pb_top = float(curr["pb_ema_top"])
        pb_bot = float(curr["pb_ema_bot"])
        lower_band = float(curr["keltner_lower"])
        upper_band = float(curr["keltner_upper"])
        adx_val = float(curr["adx"])
        rsi_val = float(curr["rsi"])

        # Check for NaN values
        if any(pd.isna([open_, high, low, close, pb_top, pb_bot, lower_band, upper_band, adx_val, rsi_val])):
            return _ret(None, None, None, None, "NaN Values")

        # ADX filter - need some volatility
        adx_ok = adx_val >= adx_min
        debug_info["adx_ok"] = adx_ok
        if not adx_ok:
            return _ret(None, None, None, None, f"ADX Too Low ({adx_val:.1f})")

        # Calculate distances to PBEMA cloud
        dist_to_pb_top = abs(high - pb_top) / pb_top if pb_top > 0 else 1.0
        dist_to_pb_bot = abs(low - pb_bot) / pb_bot if pb_bot > 0 else 1.0

        # Check if price is approaching PBEMA from below (for SHORT)
        # Conditions: Price was below PBEMA and now touching/near it
        price_below_pbema = close < pb_bot
        price_near_pbema_top = (high >= pb_top * (1 - pbema_approach_tolerance)) and (high <= pb_top * (1 + pbema_frontrun_margin))
        approaching_from_below = (
            not price_below_pbema and  # Currently at or above PBEMA
            (dist_to_pb_top <= pbema_approach_tolerance or high >= pb_top)  # Near or touched pb_top
        )

        # Check if price is approaching PBEMA from above (for LONG)
        # Conditions: Price was above PBEMA and now touching/near it
        price_above_pbema = close > pb_top
        price_near_pbema_bot = (low <= pb_bot * (1 + pbema_approach_tolerance)) and (low >= pb_bot * (1 - pbema_frontrun_margin))
        approaching_from_above = (
            not price_above_pbema and  # Currently at or below PBEMA
            (dist_to_pb_bot <= pbema_approach_tolerance or low <= pb_bot)  # Near or touched pb_bot
        )

        debug_info.update({
            "price_near_pbema_top": price_near_pbema_top,
            "price_near_pbema_bot": price_near_pbema_bot,
            "approaching_from_below": approaching_from_below,
            "approaching_from_above": approaching_from_above,
        })

        # ================= SHORT (PBEMA Resistance) =================
        # Price approached PBEMA from below - expecting rejection
        is_short = price_near_pbema_top and close < pb_top

        # Rejection candle check: wick into PBEMA but closed below
        if is_short:
            rejection_wick_short = (high >= pb_top * (1 - pbema_approach_tolerance)) and (close < pb_top)
            candle_body_below = max(open_, close) < pb_top
            is_short = rejection_wick_short and candle_body_below

        # RSI filter for SHORT - not too oversold
        short_rsi_limit = 100.0 - (rsi_limit + 10.0)
        short_rsi_ok = rsi_val >= short_rsi_limit
        debug_info["short_rsi_ok"] = short_rsi_ok
        if is_short and not short_rsi_ok:
            is_short = False

        # Slope filter - PBEMA should be sloping down or flat for SHORT
        if is_short and "slope_top" in curr.index:
            slope_val = float(curr["slope_top"])
            if slope_val > slope_thresh:  # Strong uptrend - skip SHORT
                is_short = False

        if is_short:
            # Find swing low for TP
            swing_n = 20
            start = max(0, abs_index - swing_n)
            swing_low = float(df["low"].iloc[start:abs_index].min())

            # TP options: swing low or Keltner lower band (whichever is closer)
            tp_swing = swing_low * 0.998
            tp_keltner = lower_band * 0.998
            tp = max(tp_swing, tp_keltner)  # Choose closer TP

            # Entry at close
            entry = close

            # SL above PBEMA with frontrun margin
            sl = pb_top * (1 + pbema_frontrun_margin + 0.002)

            if tp >= entry:
                return _ret(None, None, None, None, "TP Above Entry (SHORT)")
            if sl <= entry:
                sl = pb_top * (1 + pbema_frontrun_margin + 0.005)

            risk = sl - entry
            reward = entry - tp
            if risk <= 0 or reward <= 0:
                return _ret(None, None, None, None, "Invalid RR (SHORT)")

            rr = reward / risk
            tp_dist_ratio = reward / entry

            debug_info.update({
                "tp_dist_ratio": tp_dist_ratio,
                "rr_value": rr,
            })

            if tp_dist_ratio < tp_min_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
            if tp_dist_ratio > tp_max_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
            if rr < min_rr:
                return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

            reason = f"ACCEPTED(PBEMA_Reaction,R:{rr:.2f})"
            return _ret("SHORT", entry, tp, sl, reason)

        # ================= LONG (PBEMA Support) =================
        # Price approached PBEMA from above - expecting bounce
        is_long = price_near_pbema_bot and close > pb_bot

        # Rejection candle check: wick into PBEMA but closed above
        if is_long:
            rejection_wick_long = (low <= pb_bot * (1 + pbema_approach_tolerance)) and (close > pb_bot)
            candle_body_above = min(open_, close) > pb_bot
            is_long = rejection_wick_long and candle_body_above

        # RSI filter for LONG - not too overbought
        long_rsi_limit = rsi_limit + 10.0
        long_rsi_ok = rsi_val <= long_rsi_limit
        debug_info["long_rsi_ok"] = long_rsi_ok
        if is_long and not long_rsi_ok:
            is_long = False

        # Slope filter - PBEMA should be sloping up or flat for LONG
        if is_long and "slope_bot" in curr.index:
            slope_val = float(curr["slope_bot"])
            if slope_val < -slope_thresh:  # Strong downtrend - skip LONG
                is_long = False

        if is_long:
            # Find swing high for TP
            swing_n = 20
            start = max(0, abs_index - swing_n)
            swing_high = float(df["high"].iloc[start:abs_index].max())

            # TP options: swing high or Keltner upper band (whichever is closer)
            tp_swing = swing_high * 1.002
            tp_keltner = upper_band * 1.002
            tp = min(tp_swing, tp_keltner)  # Choose closer TP

            # Entry at close
            entry = close

            # SL below PBEMA with frontrun margin
            sl = pb_bot * (1 - pbema_frontrun_margin - 0.002)

            if tp <= entry:
                return _ret(None, None, None, None, "TP Below Entry (LONG)")
            if sl >= entry:
                sl = pb_bot * (1 - pbema_frontrun_margin - 0.005)

            risk = entry - sl
            reward = tp - entry
            if risk <= 0 or reward <= 0:
                return _ret(None, None, None, None, "Invalid RR (LONG)")

            rr = reward / risk
            tp_dist_ratio = reward / entry

            debug_info.update({
                "tp_dist_ratio": tp_dist_ratio,
                "rr_value": rr,
            })

            if tp_dist_ratio < tp_min_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
            if tp_dist_ratio > tp_max_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
            if rr < min_rr:
                return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

            reason = f"ACCEPTED(PBEMA_Reaction,R:{rr:.2f})"
            return _ret("LONG", entry, tp, sl, reason)

        return _ret(None, None, None, None, "No Signal")

    @staticmethod
    def check_signal(
            df: pd.DataFrame,
            config: dict,
            index: int = -2,
            return_debug: bool = False,
    ) -> Tuple:
        """
        Wrapper fonksiyon - strategy_mode'a göre uygun strateji fonksiyonunu çağırır.

        strategy_mode değerleri:
        - "keltner_bounce" (default): Keltner band bounce stratejisi
        - "pbema_reaction": PBEMA tepki stratejisi

        Args:
            df: OHLCV + indikatör dataframe
            config: Strateji konfigürasyonu (rr, rsi, slope, strategy_mode, vs.)
            index: Sinyal kontrol edilecek mum indeksi
            return_debug: Debug bilgisi döndür

        Returns:
            (s_type, entry, tp, sl, reason) veya debug ile birlikte
        """
        strategy_mode = config.get("strategy_mode", DEFAULT_STRATEGY_CONFIG.get("strategy_mode", "keltner_bounce"))

        if strategy_mode == "pbema_reaction":
            return TradingEngine.check_pbema_reaction_signal(
                df,
                index=index,
                min_rr=config.get("rr", DEFAULT_STRATEGY_CONFIG["rr"]),
                rsi_limit=config.get("rsi", DEFAULT_STRATEGY_CONFIG["rsi"]),
                slope_thresh=config.get("slope", DEFAULT_STRATEGY_CONFIG["slope"]),
                use_alphatrend=config.get("at_active", DEFAULT_STRATEGY_CONFIG["at_active"]),
                pbema_approach_tolerance=config.get("pbema_approach_tolerance", DEFAULT_STRATEGY_CONFIG["pbema_approach_tolerance"]),
                pbema_frontrun_margin=config.get("pbema_frontrun_margin", DEFAULT_STRATEGY_CONFIG["pbema_frontrun_margin"]),
                tp_min_dist_ratio=config.get("tp_min_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_min_dist_ratio"]),
                tp_max_dist_ratio=config.get("tp_max_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_max_dist_ratio"]),
                adx_min=config.get("adx_min", DEFAULT_STRATEGY_CONFIG["adx_min"]),
                return_debug=return_debug,
            )
        else:
            # Default: keltner_bounce strategy
            return TradingEngine.check_signal_diagnostic(
                df,
                index=index,
                min_rr=config.get("rr", DEFAULT_STRATEGY_CONFIG["rr"]),
                rsi_limit=config.get("rsi", DEFAULT_STRATEGY_CONFIG["rsi"]),
                slope_thresh=config.get("slope", DEFAULT_STRATEGY_CONFIG["slope"]),
                use_alphatrend=config.get("at_active", DEFAULT_STRATEGY_CONFIG["at_active"]),
                hold_n=config.get("hold_n", DEFAULT_STRATEGY_CONFIG["hold_n"]),
                min_hold_frac=config.get("min_hold_frac", DEFAULT_STRATEGY_CONFIG["min_hold_frac"]),
                pb_touch_tolerance=config.get("pb_touch_tolerance", DEFAULT_STRATEGY_CONFIG["pb_touch_tolerance"]),
                body_tolerance=config.get("body_tolerance", DEFAULT_STRATEGY_CONFIG["body_tolerance"]),
                cloud_keltner_gap_min=config.get("cloud_keltner_gap_min", DEFAULT_STRATEGY_CONFIG["cloud_keltner_gap_min"]),
                tp_min_dist_ratio=config.get("tp_min_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_min_dist_ratio"]),
                tp_max_dist_ratio=config.get("tp_max_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_max_dist_ratio"]),
                adx_min=config.get("adx_min", DEFAULT_STRATEGY_CONFIG["adx_min"]),
                return_debug=return_debug,
            )

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
            istanbul_time_series = plot_df['timestamp'] + pd.Timedelta(hours=3)
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

                def _trade_sort_key(trade: dict) -> float:
                    tid = trade.get('id')
                    if isinstance(tid, (int, float)):
                        tid_val = float(tid)
                        return tid_val / 1000 if tid_val > 1e12 else tid_val
                    ts_val = trade.get('timestamp') or trade.get('time') or ''
                    try:
                        return dateutil.parser.parse(ts_val).timestamp()
                    except Exception:
                        return 0.0

                # Aynı trade'i tekrar eklememek için ID/timestamp bazlı deduplikasyon
                dedup = {}
                for tr in active_trades + past_trades:
                    key = tr.get('id') or tr.get('timestamp') or tr.get('time') or id(tr)
                    dedup[key] = tr

                all_trades_sorted = sorted(dedup.values(), key=_trade_sort_key)
                all_trades_to_show = all_trades_sorted[-2:]  # Sadece en güncel 2 trade (açıklar dahil)

                trades_with_visibility = []
                for i, trade in enumerate(all_trades_to_show):
                    draw_box = True
                    if i < len(all_trades_to_show) - 1:
                        next_trade = all_trades_to_show[i + 1]
                        time_diff_secs = _trade_sort_key(next_trade) - _trade_sort_key(trade)
                        time_diff_mins = time_diff_secs / 60
                        if time_diff_mins < (interval_mins * 15): draw_box = False
                    trades_with_visibility.append((trade, draw_box))

                for trade, draw_box in trades_with_visibility:
                    t_type = trade['type'];
                    entry = float(trade['entry']);
                    tp = float(trade['tp']);
                    sl = float(trade['sl'])

                    # Timestamp güvenli parse ve mum aralığı dahilinde mi kontrolü
                    start_ts_raw = trade.get('timestamp', trade.get('time', ''))
                    try:
                        start_dt = dateutil.parser.parse(start_ts_raw)
                        start_dt = pd.to_datetime(start_dt, utc=True)
                        if pd.isna(start_dt) or start_dt < candle_start or start_dt > candle_end:
                            continue
                        future_dt = start_dt + (time_diff * 20)
                        start_ts_str = start_dt.strftime('%Y-%m-%d %H:%M')
                        future_ts_str = future_dt.strftime('%Y-%m-%d %H:%M')
                    except Exception:
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


# --- REAL-TIME DATA STREAMER (Binance Futures WebSocket) ---
class BinanceWebSocketKlineStream(threading.Thread):
    def __init__(self, symbols, timeframes, max_candles=1000):
        super().__init__(daemon=True)
        self.symbols = symbols
        self.timeframes = timeframes
        self.max_candles = max_candles
        self._running = False
        self._socket = None
        self._lock = threading.Lock()
        self._host = "fstream.binance.com"
        self._url_path = self._build_stream_path()
        self._data = {(s, tf): TradingEngine.get_data(s, tf, limit=max_candles) for s in symbols for tf in timeframes}

    def _build_stream_path(self):
        streams = [f"{s.lower()}@kline_{tf}" for s in self.symbols for tf in self.timeframes]
        stream_query = "/stream?streams=" + "/".join(streams)
        return stream_query

    def _recv_exact(self, length):
        data = b""
        while len(data) < length:
            chunk = self._socket.recv(length - len(data))
            if not chunk:
                raise ConnectionError("WebSocket bağlantısı kapandı")
            data += chunk
        return data

    def _read_frame(self):
        header = self._recv_exact(2)
        b1, b2 = header[0], header[1]
        opcode = b1 & 0x0F
        masked = b2 & 0x80
        length = b2 & 0x7F

        if length == 126:
            length = int.from_bytes(self._recv_exact(2), "big")
        elif length == 127:
            length = int.from_bytes(self._recv_exact(8), "big")

        mask = self._recv_exact(4) if masked else None
        payload = self._recv_exact(length) if length else b""

        if mask:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

        return opcode, payload

    def _send_pong(self, payload=b""):
        if not self._socket:
            return
        frame_head = bytes([0x8A, len(payload)])
        self._socket.sendall(frame_head + payload)

    def _close_socket(self):
        try:
            if self._socket:
                self._socket.close()
        except Exception:
            pass
        self._socket = None

    def stop(self):
        self._running = False
        self._close_socket()

    def _perform_handshake(self):
        port = 443
        raw_sock = socket.create_connection((self._host, port), timeout=10)
        context = ssl.create_default_context()
        self._socket = context.wrap_socket(raw_sock, server_hostname=self._host)

        key = base64.b64encode(os.urandom(16)).decode()
        request_lines = [
            f"GET {self._url_path} HTTP/1.1",
            f"Host: {self._host}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
            "Origin: https://fstream.binance.com",
            "\r\n",
        ]
        self._socket.sendall("\r\n".join(request_lines).encode())

        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self._socket.recv(1024)
            if not chunk:
                break
            response += chunk

        if b"101" not in response:
            raise ConnectionError(f"WebSocket el sıkışma hatası: {response[:200]}")

    def _update_dataframe(self, symbol, interval, kline):
        ts = pd.to_datetime(kline.get("t"), unit="ms", utc=True)
        new_row = {
            "timestamp": ts,
            "open": float(kline.get("o", 0)),
            "high": float(kline.get("h", 0)),
            "low": float(kline.get("l", 0)),
            "close": float(kline.get("c", 0)),
            "volume": float(kline.get("v", 0)),
        }

        key = (symbol, interval)
        df = self._data.get(key, pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']))

        if not df.empty and df.iloc[-1]['timestamp'] == ts:
            df.iloc[-1] = list(new_row.values())
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        if len(df) > self.max_candles:
            df = df.iloc[-self.max_candles:].reset_index(drop=True)

        self._data[key] = df

    def _handle_message(self, payload):
        try:
            obj = json.loads(payload)
            kline = obj.get("data", {}).get("k", {})
            symbol = kline.get("s")
            interval = kline.get("i")
            if not symbol or not interval:
                return
            with self._lock:
                self._update_dataframe(symbol, interval, kline)
        except Exception:
            pass

    def get_latest_bulk(self):
        with self._lock:
            return {(s, tf): df.copy() for (s, tf), df in self._data.items() if not df.empty}

    def run(self):
        self._running = True
        while self._running:
            try:
                self._perform_handshake()
                while self._running:
                    opcode, payload = self._read_frame()
                    if opcode == 0x1:  # text
                        self._handle_message(payload.decode())
                    elif opcode == 0x8:  # close
                        break
                    elif opcode == 0x9:  # ping
                        self._send_pong(payload)
            except Exception as e:
                print(f"[WS] Yeniden bağlanılıyor: {e}")
                time.sleep(2)
            finally:
                self._close_socket()

# --- WORKERS ---
class LiveBotWorker(QThread):
    update_ui_signal = pyqtSignal(str, str, str, str)
    trade_signal = pyqtSignal(dict)
    price_signal = pyqtSignal(str, float)
    potential_signal = pyqtSignal(dict)

    def __init__(self, current_params, tg_token, tg_chat_id, show_rr):
        super().__init__()
        self.is_running = True
        self.tg_token = tg_token;
        self.tg_chat_id = tg_chat_id;
        self.show_rr = show_rr
        self.last_signals = {sym: {tf: None for tf in TIMEFRAMES} for sym in SYMBOLS}
        self.last_potential = {sym: {tf: None for tf in TIMEFRAMES} for sym in SYMBOLS}
        self.ws_stream = BinanceWebSocketKlineStream(SYMBOLS, TIMEFRAMES, max_candles=1200)

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

    def stop(self):
        self.is_running = False
        self.ws_stream.stop()

    def run(self):
        # Telegram mesajlarını asenkron yapmak için bu import gerekli
        import threading

        self.ws_stream.start()
        next_price_time = 0
        next_candle_time = 0
        try:
            while self.is_running:
                now = time.time()

                stream_snapshot = None

                if now >= next_price_time:
                    try:
                        stream_snapshot = self.ws_stream.get_latest_bulk()
                        for sym in SYMBOLS:
                            df_price = stream_snapshot.get((sym, "1m"))

                            if (df_price is None or df_price.empty) and TIMEFRAMES:
                                df_price = stream_snapshot.get((sym, TIMEFRAMES[0]))

                            if df_price is not None and not df_price.empty:
                                price = float(df_price.iloc[-1]['close'])
                                self.price_signal.emit(sym, price)
                                trade_manager.update_live_pnl_with_price(sym, price)
                        if not stream_snapshot:
                            latest_prices = TradingEngine.get_latest_prices(SYMBOLS)
                            for sym, price in latest_prices.items():
                                self.price_signal.emit(sym, price)
                                trade_manager.update_live_pnl_with_price(sym, price)
                    except Exception as e:
                        print(f"[LIVE] Fiyat güncelleme hatası: {e}")
                    next_price_time = now + 0.5

                if now >= next_candle_time:
                    try:
                        # 1. TÜM VERİLERİ WEBSOCKET'TEN AKTİF OLARAK TOPLA
                        if stream_snapshot is None:
                            stream_snapshot = self.ws_stream.get_latest_bulk()
                        bulk_data = stream_snapshot

                        # WebSocket verisi henüz hazır değilse eski REST çekimine düş
                        if not bulk_data:
                            bulk_data = TradingEngine.get_all_candles_parallel(SYMBOLS, TIMEFRAMES)

                        # 2. Gelen verileri işle (Bu kısım işlemci hızında akar, milisaniyeler sürer)
                        for (sym, tf), df in bulk_data.items():
                            if df.empty: continue

                            try:
                                # Skip disabled symbol/timeframe combinations EARLY
                                sym_cfg = SYMBOL_PARAMS.get(sym, {})
                                tf_cfg = sym_cfg.get(tf, {}) if isinstance(sym_cfg, dict) else {}
                                if tf_cfg.get("disabled", False):
                                    continue

                                if len(df) < 3:
                                    continue
                                # Binance kline: son satır çoğunlukla oluşan (henüz kapanmamış) mumdur.
                                closed = df.iloc[-2]
                                forming = df.iloc[-1]
                                curr_price = float(closed['close'])
                                closed_ts_utc = closed['timestamp']
                                forming_ts_utc = forming['timestamp']
                                istanbul_time = pd.Timestamp(closed_ts_utc) + pd.Timedelta(hours=3)
                                ts_str = istanbul_time.strftime("%Y-%m-%d %H:%M")
                                # Backtest ile uyumlu fill: sinyal mumu kapandıktan sonraki mumun OPEN fiyatı
                                next_open_price = float(forming['open'])
                                next_open_ts_str = (pd.Timestamp(forming_ts_utc) + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
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

                                # Skip if insufficient data for indicators (need 200+ candles for EMA200)
                                if len(df) < 250:
                                    continue

                                df_ind = TradingEngine.calculate_indicators(df.copy())
                                df_closed = df_ind.iloc[:-1].copy()  # oluşan mumu çıkar
                                config = load_optimized_config(sym, tf)

                                # Skip disabled symbol/timeframe combinations
                                if config.get("disabled", False):
                                    continue

                                rr, rsi, slope = config['rr'], config['rsi'], config['slope']
                                use_at = config['at_active']
                                at_status_log = "AT:ON" if use_at else "AT:OFF"
                                strategy_mode = config.get("strategy_mode", "keltner_bounce")
                                strategy_log = f"Mode:{strategy_mode[:7]}"

                                # Use wrapper function to support both strategies
                                s_type, s_entry, s_tp, s_sl, s_reason, s_debug = TradingEngine.check_signal(
                                    df_closed,
                                    config=config,
                                    index=-1,
                                    return_debug=True,
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

                                decision = None
                                reject_reason = ""
                                # Sinyal Yönetimi
                                if s_type and "ACCEPTED" in s_reason:
                                    has_open = False
                                    for t in trade_manager.open_trades:
                                        if t['symbol'] == sym and t['timeframe'] == tf: has_open = True; break

                                    if has_open:
                                        decision = "Rejected"
                                        reject_reason = "Open Position"
                                        log_msg = f"{tf} | {curr_price} | ⚠️ Pozisyon Var{live_pnl_str}"
                                        json_data = TradingEngine.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                         active_trades if self.show_rr else [])
                                        self.update_ui_signal.emit(sym, tf, json_data, log_msg)

                                    elif self.last_signals[sym][tf] != closed_ts_utc:
                                        if trade_manager.check_cooldown(sym, tf, forming_ts_utc):
                                            decision = "Rejected"
                                            reject_reason = "Cooldown"
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
                                            decision = "Accepted"
                                            log_msg = f"{tf} | {curr_price} | 🔥 {s_type} ({setup_tag})"
                                            json_data = TradingEngine.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                             active_trades if self.show_rr else [])
                                            self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                    else:
                                        decision = "Rejected"
                                        reject_reason = "Duplicate Signal"
                                        log_msg = f"{tf} | {curr_price} | ⏳ İşlemde...{live_pnl_str}"
                                        json_data = TradingEngine.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                         active_trades if self.show_rr else [])
                                        self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                else:
                                    # Sinyal Yoksa Logla
                                    decision = f"Rejected: {s_reason}" if s_reason else "Rejected"
                                    if s_reason and "REJECT" in s_reason:
                                        log_msg = f"{tf} | {curr_price} | ⚠️ {s_reason}{live_pnl_str}"
                                    else:
                                        log_msg = f"{tf} | {curr_price} | {at_status_log}{live_pnl_str}"

                                    json_data = TradingEngine.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                     active_trades if self.show_rr else [])
                                    self.update_ui_signal.emit(sym, tf, json_data, log_msg)

                                if s_type and "ACCEPTED" in s_reason and self.last_potential[sym][tf] != closed_ts_utc:
                                    direction = s_type or ("LONG" if s_debug.get("holding_long") else ("SHORT" if s_debug.get("holding_short") else ""))

                                    checks = dict(s_debug or {})
                                    if "rsi_value" not in checks:
                                        try:
                                            checks["rsi_value"] = float(df_closed["rsi"].iloc[-1])
                                        except Exception:
                                            pass

                                    final_decision = decision or "Rejected"
                                    if final_decision == "Accepted":
                                        decision_text = "Accepted"
                                        reason_text = ""
                                    else:
                                        reason_text = reject_reason or (s_reason if s_reason != "No Signal" else "Unknown")
                                        decision_text = f"Rejected: {reason_text}"

                                    diag_entry = {
                                        "timestamp": ts_str,
                                        "symbol": sym,
                                        "timeframe": tf,
                                        "type": direction,
                                        "reason": reason_text,
                                        "decision": decision_text,
                                        "price": curr_price,
                                        "next_open": next_open_price,
                                        "setup": setup_tag,
                                        "checks": checks,
                                    }
                                    potential_trades.add(diag_entry)
                                    self.potential_signal.emit(diag_entry)
                                    self.last_potential[sym][tf] = closed_ts_utc

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
        finally:
            self.ws_stream.stop()


# --- OPTIMIZER WORKER (v35.0 - MATHEMATICALLY CORRECT R-CALC) ---
class OptimizerWorker(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, symbol, candle_limit, rr_range, rsi_range, slope_range, use_alphatrend,
                 monte_carlo_mode=False, timeframes=None):
        super().__init__()
        self.symbol = symbol
        self.candle_limit = candle_limit
        self.rr_range = rr_range
        self.rsi_range = rsi_range
        self.slope_range = slope_range
        self.monte_carlo_mode = monte_carlo_mode
        self.timeframes = timeframes or list(TIMEFRAMES)

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
            for tf in self.timeframes:
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
            results_by_tf = {tf: [] for tf in self.timeframes}
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

                    # Build config for wrapper function
                    loop_config = {
                        **DEFAULT_STRATEGY_CONFIG,
                        "rr": rr,
                        "rsi": rsi,
                        "slope": slope,
                        "at_active": at_active,
                    }

                    for i in range(start_idx, limit_idx):
                        s_type, _, s_tp_raw, s_sl_raw, s_reason = TradingEngine.check_signal(
                            df,
                            config=loop_config,
                            index=i,
                            return_debug=False,
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
                            risk_dist = abs(real_entry_price - s_sl_raw)
                            partial_realized_r = 0.0
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
                                            if risk_dist > 0:
                                                partial_realized_r = (abs(curr_close - real_entry_price) / risk_dist) * 0.5
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

                                if risk_dist > 0:
                                    raw_r = partial_realized_r + (reward_dist / risk_dist) * curr_size_ratio
                                    # Kazançtan masrafları düş
                                    net_r += (raw_r - fee_cost_r - funding_cost_r)
                                wins += 1

                            elif outcome == "LOSS":
                                loss_r = 1.0 if not partial_taken else 0
                                # Kayıpta: 1R kayıp + Fee + Funding
                                net_r += (partial_realized_r - loss_r - fee_cost_r - funding_cost_r)
                                losses += 1

                            elif outcome == "BE":
                                # BE olsa bile Fee ödenir!
                                net_r += (partial_realized_r - fee_cost_r - funding_cost_r)

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

    def __init__(self, symbols, timeframes, candles, skip_optimization=False, quick_mode=False):
        super().__init__()
        self.symbols = symbols
        self.timeframes = timeframes
        self.candles = candles
        self.skip_optimization = skip_optimization
        self.quick_mode = quick_mode
        self._last_log_time = 0.0
        self._pending_log = None

    def _throttled_log(self, text: str):
        """Reduce log spam so UI can stay responsive during heavy backtests."""
        now = time.time()
        if (now - self._last_log_time) >= 0.4:
            self.log_signal.emit(text)
            self._last_log_time = now
            self._pending_log = None
        else:
            # Keep the latest message to emit on the next window
            self._pending_log = text

    def _flush_pending_log(self):
        if self._pending_log:
            self.log_signal.emit(self._pending_log)
            self._pending_log = None
            self._last_log_time = time.time()

    def run(self):
        result = {}

        try:
            result = run_portfolio_backtest(
                symbols=self.symbols,
                timeframes=self.timeframes,
                candles=self.candles,
                progress_callback=self._throttled_log,
                draw_trades=True,
                max_draw_trades=30,
                skip_optimization=self.skip_optimization,
                quick_mode=self.quick_mode,
            ) or {}
            # Ensure the last status arrives even if throttled
            self._flush_pending_log()
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
        self.potential_entries = self._load_potential_entries()

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

        self.web_views = {}
        chart_tabs = QTabWidget()

        def build_chart_grid(timeframes):
            widget = QWidget()
            grid = QGridLayout(widget)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(5)

            for idx, tf in enumerate(timeframes):
                box = QGroupBox(f"{tf} Grafiği")
                box.setStyleSheet("QGroupBox { border: 1px solid #333; font-weight: bold; color: #00ccff; }")
                box_layout = QVBoxLayout(box)
                box_layout.setContentsMargins(0, 15, 0, 0)

                if ENABLE_CHARTS:
                    view = QWebEngineView()
                    view.setHtml(CHART_TEMPLATE)
                    view.loadFinished.connect(lambda ok, t=tf: self.on_load_finished(ok, t))
                    box_layout.addWidget(view)
                    self.web_views[tf] = view
                else:
                    # Grafik kapalı - basit placeholder
                    placeholder = QLabel(f"📊 {tf} - Grafik devre dışı (ENABLE_CHARTS=False)")
                    placeholder.setStyleSheet("color: #666; padding: 20px; font-size: 14px;")
                    placeholder.setAlignment(Qt.AlignCenter)
                    box_layout.addWidget(placeholder)

                grid.addWidget(box, idx // 2, idx % 2)

            return widget

        chart_tabs.addTab(build_chart_grid(LOWER_TIMEFRAMES), "LTF")
        chart_tabs.addTab(build_chart_grid(HTF_TIMEFRAMES), "HTF")
        live_layout.addWidget(chart_tabs, stretch=6)
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
        self.open_trades_table.setColumnCount(12)
        self.open_trades_table.setHorizontalHeaderLabels(
            ["Zaman", "Coin", "TF", "Yön", "Setup", "Giriş", "TP", "SL", "Büyüklük ($)", "PnL", "Durum", "Bilgi"])
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

        # 4. SEKME: Potansiyel İşlemler (detaylı red/şart takibi)
        potential_widget = QWidget();
        pot_layout = QVBoxLayout(potential_widget)

        # Toolbar for potential trades tab
        pot_toolbar = QHBoxLayout()
        pot_toolbar.addStretch()
        self.btn_clear_pot = QPushButton("🗑️ Logları Temizle")
        self.btn_clear_pot.setStyleSheet("background-color: #8b0000; color: white; padding: 5px 15px;")
        self.btn_clear_pot.clicked.connect(self.clear_potential_entries)
        pot_toolbar.addWidget(self.btn_clear_pot)
        pot_layout.addLayout(pot_toolbar)

        pot_group = QGroupBox("Potansiyel İşlemler (Kalıcı - bot yeniden başlasa da korunur)");
        pot_inner = QVBoxLayout()

        self.potential_table = QTableWidget();
        self.potential_table.setColumnCount(14)
        self.potential_table.setHorizontalHeaderLabels([
            "Zaman", "Coin", "TF", "Yön", "Karar", "Sebep", "ADX", "Hold", "Retest",
            "PB/Cloud", "Trend", "RSI", "RR", "TP%",
        ])
        self.potential_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        pot_inner.addWidget(self.potential_table)
        pot_group.setLayout(pot_inner)
        pot_layout.addWidget(pot_group)
        self.main_tabs.addTab(potential_widget, "🔍 Potansiyel")

        # 5. SEKME: Backtest
        backtest_widget = QWidget();
        backtest_layout = QVBoxLayout(backtest_widget)
        bt_cfg = QHBoxLayout()
        bt_cfg.addWidget(QLabel("Semboller:"));
        bt_cfg.addWidget(QLabel(", ".join(SYMBOLS)))
        bt_cfg.addWidget(QLabel("TF Seçimi:"));
        self.backtest_tf_checks = {}
        bt_tf_layout = QHBoxLayout()
        for tf in TIMEFRAMES:
            cb = QCheckBox(tf)
            cb.setChecked(True)
            bt_tf_layout.addWidget(cb)
            self.backtest_tf_checks[tf] = cb
        bt_cfg.addLayout(bt_tf_layout)
        bt_cfg.addWidget(QLabel("Mum Sayısı:"));
        self.backtest_candles = QSpinBox();
        self.backtest_candles.setRange(500, 100000);  # Maksimum limit kaldırıldı
        self.backtest_candles.setValue(3000);
        bt_cfg.addWidget(self.backtest_candles)

        # Hız ayarları
        speed_layout = QHBoxLayout()
        self.chk_skip_optimization = QCheckBox("⚡ Optimizer Atla (Kayıtlı Config)")
        self.chk_skip_optimization.setToolTip("Optimizer çalıştırmadan, önceden kaydedilmiş config'leri kullanır. ÇOK HIZLI!")
        speed_layout.addWidget(self.chk_skip_optimization)
        self.chk_quick_mode = QCheckBox("🚀 Hızlı Mod (13 config)")
        self.chk_quick_mode.setToolTip("Azaltılmış config grid kullanır (120 yerine 13). ~5x daha hızlı.")
        speed_layout.addWidget(self.chk_quick_mode)
        bt_cfg.addLayout(speed_layout)

        self.btn_run_backtest = QPushButton("🧪 Backtest Çalıştır");
        self.btn_run_backtest.clicked.connect(self.start_backtest);
        bt_cfg.addWidget(self.btn_run_backtest)
        backtest_layout.addLayout(bt_cfg)

        self.backtest_logs = QTextEdit();
        self.backtest_logs.setReadOnly(True);
        backtest_layout.addWidget(self.backtest_logs)
        self.main_tabs.addTab(backtest_widget, "🧪 Backtest")

        # 6. SEKME: Optimizasyon (Temizlendi & Otomatikleştirildi)
        opt_widget = QWidget();
        opt_layout = QVBoxLayout(opt_widget)
        grid_group = QGroupBox("Parametre Aralıkları");
        grid_layout = QHBoxLayout()
        tf_group = QGroupBox("Zaman Dilimleri")
        tf_layout = QHBoxLayout()
        self.opt_tf_checks = {}
        for tf in TIMEFRAMES:
            cb = QCheckBox(tf)
            cb.setChecked(True)
            tf_layout.addWidget(cb)
            self.opt_tf_checks[tf] = cb
        tf_group.setLayout(tf_layout)
        opt_layout.addWidget(tf_group)
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
        self.live_worker.potential_signal.connect(self.append_potential_trade)
        self.live_worker.start()
        self.logs.append(">>> Sistem Başlatıldı. v30.4 (PROFIT ENGINE)")
        self.load_tf_settings("1m")
        self.table_timer = QTimer();
        self.table_timer.timeout.connect(self.refresh_trade_table_from_manager);
        self.table_timer.start(1000)
        # --- OTO BACKTEST BAŞLAT ---
        self.auto_backtest = None  # Gece otomatik backtest geçici olarak devre dışı
        # ---------------------------

        self.logs.append(">>> Sistem Başlatıldı. v30.6 (Auto Report - Otomatik Backtest Kapalı)")

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
        # Grafik güncelleme sadece ENABLE_CHARTS=True ise
        if ENABLE_CHARTS and self.views_ready.get(tf, False) and json_data and json_data != "{}":
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

    def _fmt_bool(self, value):
        return "✓" if bool(value) else "×"

    def _load_potential_entries(self) -> list:
        """Load potential trade entries from persistent storage."""
        try:
            if os.path.exists(POT_LOG_FILE):
                with open(POT_LOG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data[-400:]  # Keep last 400
        except Exception as e:
            print(f"[POT] Failed to load potential entries: {e}")
        return []

    def _save_potential_entries(self):
        """Save potential trade entries to persistent storage."""
        try:
            with open(POT_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.potential_entries[-400:], f, indent=2, default=str)
        except Exception as e:
            print(f"[POT] Failed to save potential entries: {e}")

    def clear_potential_entries(self):
        """Clear all potential trade entries (called by reset button)."""
        self.potential_entries = []
        self._save_potential_entries()
        self.refresh_potential_table()
        self.append_log("[POT] Potansiyel işlem logları temizlendi.")

    def append_potential_trade(self, entry: dict):
        if not isinstance(entry, dict):
            return
        self.potential_entries.append(entry)
        if len(self.potential_entries) > 400:
            self.potential_entries = self.potential_entries[-400:]
        self._save_potential_entries()  # Persist to disk
        self.refresh_potential_table()

    def refresh_potential_table(self):
        data = list(reversed(self.potential_entries))
        self.potential_table.setRowCount(len(data))

        for row_idx, entry in enumerate(data):
            checks = entry.get("checks", {}) or {}
            direction = entry.get("type", "") or "-"

            if direction == "LONG":
                trend_ok = not bool(checks.get("trend_down_strong"))
                hold_ok = bool(checks.get("holding_long"))
                retest_ok = bool(checks.get("retest_long"))
                pb_ok = bool(checks.get("pb_target_long"))
            else:
                trend_ok = not bool(checks.get("trend_up_strong"))
                hold_ok = bool(checks.get("holding_short"))
                retest_ok = bool(checks.get("retest_short"))
                pb_ok = bool(checks.get("pb_target_short"))

            rr_val = checks.get("rr_value")
            tp_ratio = checks.get("tp_dist_ratio")
            rsi_val = checks.get("rsi_value")

            values = [
                entry.get("timestamp", "-"),
                entry.get("symbol", "-"),
                entry.get("timeframe", "-"),
                direction,
                entry.get("decision", "-"),
                entry.get("reason", "-"),
                self._fmt_bool(checks.get("adx_ok")),
                self._fmt_bool(hold_ok),
                self._fmt_bool(retest_ok),
                self._fmt_bool(pb_ok),
                self._fmt_bool(trend_ok),
                f"{float(rsi_val):.2f}" if isinstance(rsi_val, (int, float)) else "-",
                f"{rr_val:.2f}" if isinstance(rr_val, (int, float)) else "-",
                f"{tp_ratio*100:.2f}%" if isinstance(tp_ratio, (int, float)) else "-",
            ]

            for col_idx, val in enumerate(values):
                item = QTableWidgetItem(str(val))
                if col_idx in {6, 7, 8, 9, 10}:
                    if val == "✓":
                        item.setForeground(QColor("#00ff00"))
                    elif val == "×":
                        item.setForeground(QColor("#ff5555"))
                if col_idx == 4 and str(entry.get("decision", "")).startswith("Accepted"):
                    item.setForeground(QColor("#00c853"))
                self.potential_table.setItem(row_idx, col_idx, item)

            # Accepted satırlarını görsel olarak ayır
            if str(entry.get("decision", "")).startswith("Accepted"):
                for col in range(len(values)):
                    existing_item = self.potential_table.item(row_idx, col)
                    if existing_item:
                        existing_item.setBackground(QColor(0, 60, 30))
                        existing_item.setForeground(QColor("#e8ffe8"))

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
                         "status", "info"]

            for row_idx, trade in enumerate(open_trades):
                for col_idx, col_key in enumerate(cols_open):
                    if col_key == "info":
                        val = self.describe_trade_state(trade)
                    else:
                        val = trade.get(col_key, "")
                    item = QTableWidgetItem(str(val))

                    if col_key == "size":
                        entry_price = float(trade.get("entry", 0))
                        size = float(trade.get("size", 0))
                        notional = float(trade.get("notional", entry_price * size))
                        is_partial = trade.get("partial_taken", False)
                        if is_partial:
                            item.setText(f"📉 ${notional:,.0f} (Yarım)")
                            item.setForeground(QColor("orange"))
                        else:
                            item.setText(f"${notional:,.0f} (Tam)")
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
                    if col_key == "status":
                        item.setForeground(QColor("#00ccff"))
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

    def describe_trade_state(self, trade: dict) -> str:
        parts = []

        if trade.get("partial_taken"):
            partial_price = trade.get("partial_price")
            price_note = f" @{float(partial_price):.4f}" if partial_price else ""
            parts.append(f"Partial alındı{price_note}")
        else:
            parts.append("Tam pozisyon")

        if trade.get("breakeven"):
            parts.append("SL BE/ileri çekildi")
        if trade.get("trailing_active"):
            parts.append("Trailing aktif")
        if trade.get("stop_protection"):
            parts.append("Koruma SL")

        return " | ".join(parts)

    def get_selected_timeframes(self, checkbox_map: dict) -> list:
        selected = [tf for tf, cb in (checkbox_map or {}).items() if cb.isChecked()]
        return selected if selected else list(TIMEFRAMES)

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
                    entry_price = float(trade.get("entry", 0))
                    size = float(trade.get("size", 0))
                    notional = float(trade.get("notional", entry_price * size))
                    display = f"${notional:,.2f}"
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
        selected_tfs = self.get_selected_timeframes(getattr(self, "backtest_tf_checks", {}))

        # Hız ayarlarını oku
        skip_opt = self.chk_skip_optimization.isChecked()
        quick = self.chk_quick_mode.isChecked()
        if skip_opt:
            self.backtest_logs.append("⚡ Optimizer atlanıyor (kayıtlı config kullanılacak)")
        elif quick:
            self.backtest_logs.append("🚀 Hızlı mod aktif (13 config)")

        self.backtest_worker = BacktestWorker(SYMBOLS, selected_tfs, candles, skip_opt, quick)
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
                "strategy_signature": result.get("strategy_signature") or _strategy_signature(),
            }
            self.save_backtest_meta(meta)
            self.backtest_logs.append("📊 Özet tablo kaydedildi:")
            for line in self.format_backtest_summary_lines(meta):
                self.backtest_logs.append(line)
            parity = result.get("parity_report", {})
            if parity:
                icon = "✅" if parity.get("parity_ok") else "⚠️"
                self.backtest_logs.append(
                    f"{icon} Parity kontrolü: sim ve canlı mantık {'eşleşiyor' if parity.get('parity_ok') else 'uyuşmuyor'}"
                )
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

        selected_tfs = self.get_selected_timeframes(getattr(self, "opt_tf_checks", {}))

        selected_sym = self.combo_opt_symbol.currentText()
        self.opt_logs.clear()
        self.btn_run_opt.setEnabled(False)

        # Worker'a monte_carlo parametresini gönder
        self.opt_worker = OptimizerWorker(selected_sym, candles, rr_range, rsi_range, slope_range, use_at,
                                          is_monte_carlo, selected_tfs)
        self.opt_worker.result_signal.connect(self.on_opt_update)
        self.opt_worker.start()

    def on_opt_update(self, msg):
        self.opt_logs.append(msg);
        if "Tamamlandı" in msg: self.btn_run_opt.setEnabled(True)

    def force_daily_report(self):
        self.opt_logs.append("🌙 Otomatik rapor ve gece backtest özelliği şu an kapalı.")

    def create_pnl_table(self):
        # Tabloyu oluşturur
        table = QTableWidget()
        table.setColumnCount(3)  # 3 Sütun: Zaman, Win Rate, PnL
        table.setHorizontalHeaderLabels(["Zaman Dilimi", "Başarı %", "PnL ($)"])

        # Tablo başlıklarının genişliğini ayarla
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        # Örnek boş satırlar ekleyelim (Daha sonra gerçek veriyle dolacak)
        table.setRowCount(len(TIMEFRAMES))

        for i, tf in enumerate(TIMEFRAMES):
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
        sirali_tf = list(TIMEFRAMES)
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
def _tf_to_timedelta(tf: str) -> pd.Timedelta:
    """Convert timeframe string to pandas Timedelta (compatible with pd.Timestamp and pickle-safe for multiprocessing)."""
    if tf.endswith("m"):
        return pd.Timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h"):
        return pd.Timedelta(hours=int(tf[:-1]))
    if tf.endswith("d"):
        return pd.Timedelta(days=int(tf[:-1]))
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
        self.risk_per_trade_pct = TRADING_CONFIG.get("risk_per_trade_pct", 0.01)
        self.max_portfolio_risk_pct = TRADING_CONFIG.get("max_portfolio_risk_pct", 0.03)
        self._id = 1
        # R-Multiple tracking: PnL / Risk Amount per trade
        # E[R] = sum(r_multiples) / len(r_multiples)
        self.trade_r_multiples = []

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

    def _calculate_portfolio_risk_pct(self, wallet_balance: float) -> float:
        if wallet_balance <= 0:
            return 0.0

        total_open_risk = 0.0
        for trade in self.open_trades:
            entry_price = float(trade.get("entry", 0.0))
            sl_price = float(trade.get("sl", entry_price))
            size = abs(float(trade.get("size", 0.0)))
            if entry_price <= 0 or size <= 0:
                continue
            sl_fraction = abs(entry_price - sl_price) / entry_price
            open_risk_amount = sl_fraction * size * entry_price
            total_open_risk += open_risk_amount

        return total_open_risk / wallet_balance

    def _next_id(self):
        tid = self._id
        self._id += 1
        return tid

    def open_trade(self, trade_data):
        """Returns True if trade was opened successfully, False otherwise."""
        tf = trade_data["timeframe"]
        sym = trade_data["symbol"]

        cooldown_ref_time = trade_data.get("open_time_utc") or datetime.utcnow()
        if self.check_cooldown(sym, tf, cooldown_ref_time):
            return False

        # Aynı sembol ve timeframe için halihazırda açık bir pozisyon varsa
        # yeni trade'i reddet (üst katmanda kontrol kaçsa bile güvenlik katmanı).
        if any(
            t.get("symbol") == sym and t.get("timeframe") == tf
            for t in self.open_trades
        ):
            return False

        setup_type = trade_data.get("setup", "Unknown")

        # Trade açılırken config'i snapshot olarak al ve trade'e göm
        # Bu sayede trade yaşam döngüsü boyunca aynı kurallar kullanılır
        config_snapshot = load_optimized_config(sym, tf)
        use_trailing = config_snapshot.get("use_trailing", False)
        use_dynamic_pbema_tp = config_snapshot.get("use_dynamic_pbema_tp", True)
        opt_rr = config_snapshot.get("rr", 3.0)
        opt_rsi = config_snapshot.get("rsi", 60)

        # Confidence-based risk multiplier: reduce position size for medium confidence
        confidence_level = config_snapshot.get("_confidence", "high")
        risk_multiplier = CONFIDENCE_RISK_MULTIPLIER.get(confidence_level, 1.0)
        if risk_multiplier <= 0:
            # Low confidence = no trades
            return False

        if self.wallet_balance < 10:
            return False

        raw_entry = float(trade_data["entry"])
        trade_type = trade_data["type"]

        if trade_type == "LONG":
            real_entry = raw_entry * (1 + self.slippage_pct)
        else:
            real_entry = raw_entry * (1 - self.slippage_pct)

        sl_price = float(trade_data["sl"])

        # Apply risk multiplier to effective risk per trade
        effective_risk_pct = self.risk_per_trade_pct * risk_multiplier
        current_portfolio_risk_pct = self._calculate_portfolio_risk_pct(self.wallet_balance)
        if current_portfolio_risk_pct + effective_risk_pct > self.max_portfolio_risk_pct:
            return False

        wallet_balance = self.wallet_balance
        if wallet_balance <= 0:
            return False

        risk_amount = wallet_balance * effective_risk_pct
        sl_distance = abs(real_entry - sl_price)
        if sl_distance <= 0:
            return False

        sl_fraction = sl_distance / real_entry
        if sl_fraction <= 0:
            return False

        position_notional = risk_amount / sl_fraction
        position_size = position_notional / real_entry

        leverage = TRADING_CONFIG["leverage"]
        required_margin = position_notional / leverage

        if required_margin > wallet_balance:
            max_notional = wallet_balance * leverage
            if max_notional <= 0:
                return False
            position_notional = max_notional
            position_size = position_notional / real_entry
            required_margin = position_notional / leverage

        open_time_val = trade_data.get("open_time_utc") or datetime.utcnow()
        # numpy.datetime64, pd.Timestamp veya datetime olabilir - hepsini datetime'a çevir
        if isinstance(open_time_val, np.datetime64):
            open_time_val = pd.Timestamp(open_time_val).to_pydatetime()
        elif isinstance(open_time_val, pd.Timestamp):
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
            "sl": sl_price,
            "size": position_size,
            "margin": required_margin,
            "notional": position_notional,
            "status": "OPEN",
            "pnl": 0.0,
            "breakeven": False,
            "trailing_active": False,
            "partial_taken": False,
            "partial_price": None,
            "has_cash": True,
            "close_time": "",
            "close_price": "",
            "events": [],
            # Trade açılırken snapshot edilen config ayarları (yaşam döngüsü boyunca sabit kalır)
            "use_trailing": use_trailing,
            "use_dynamic_pbema_tp": use_dynamic_pbema_tp,
            "opt_rr": opt_rr,
            "opt_rsi": opt_rsi,
            # R-multiple hesaplaması için risk tutarı (trade açılırken kaydedilir)
            "risk_amount": risk_amount,
        }

        self.wallet_balance -= required_margin
        self.locked_margin += required_margin
        self.open_trades.append(new_trade)
        new_portfolio_risk_pct = self._calculate_portfolio_risk_pct(self.wallet_balance)

        print(
            f"[SIM] İşlem Açıldı | Entry: {real_entry:.4f}, SL: {sl_price:.4f}, "
            f"Size: {position_size:.6f}, Notional: ${position_notional:.2f}, "
            f"Margin: ${required_margin:.2f}, Risk%: {effective_risk_pct * 100:.2f}% "
            f"({confidence_level}), Risk$: ${risk_amount:.2f}, Portföy: {new_portfolio_risk_pct * 100:.2f}%"
        )
        return True

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

            # Config'i trade dict'inden oku (açılışta snapshot edildi)
            # Eski trade'ler için fallback olarak load_optimized_config kullan ve trade'e yaz
            if "use_trailing" in trade:
                use_trailing = trade.get("use_trailing", False)
                use_dynamic_tp = trade.get("use_dynamic_pbema_tp", True)
            else:
                # Backward compatibility: eski trade'ler için config'den oku ve trade'e yaz
                # Bu sayede sadece bir kere config'ten okunur, sonraki mumlarda trade'den okunur
                config = load_optimized_config(symbol, tf)
                use_trailing = config.get("use_trailing", False)
                use_dynamic_tp = config.get("use_dynamic_pbema_tp", True)
                # Trade'e yaz - sonraki mumlarda trade'den okunacak
                self.open_trades[i]["use_trailing"] = use_trailing
                self.open_trades[i]["use_dynamic_pbema_tp"] = use_dynamic_tp
                self.open_trades[i]["opt_rr"] = config.get("rr", 3.0)
                self.open_trades[i]["opt_rsi"] = config.get("rsi", 60)
            use_partial = not use_trailing

            # --- Fiyatlar ---
            # Partial TP için conservative fill hesaplaması
            # Ama progress için gerçek candle extreme kullanılmalı
            if t_type == "LONG":
                close_price = candle_close
                extreme_price = candle_high  # Progress için gerçek high
                # Partial fill için conservative: 70% close + 30% extreme
                partial_fill_price = close_price * 0.70 + extreme_price * 0.30
                pnl_percent_close = (close_price - entry) / entry
                in_profit = extreme_price > entry
            else:
                close_price = candle_close
                extreme_price = candle_low  # Progress için gerçek low
                # Partial fill için conservative: 70% close + 30% extreme
                partial_fill_price = close_price * 0.70 + extreme_price * 0.30
                pnl_percent_close = (entry - close_price) / entry
                in_profit = extreme_price < entry

            dyn_tp = tp
            if use_dynamic_tp:
                try:
                    if pb_top is not None and pb_bot is not None:
                        dyn_tp = float(pb_bot) if t_type == "LONG" else float(pb_top)
                except Exception:
                    dyn_tp = tp
                self.open_trades[i]["tp"] = dyn_tp

            if t_type == "LONG":
                live_pnl = (close_price - entry) * size
            else:
                live_pnl = (entry - close_price) * size
            self.open_trades[i]["pnl"] = live_pnl

            # Hedefe ilerleme oranı (GERÇEK extreme'e göre, conservative değil)
            total_dist = abs(dyn_tp - entry)
            if total_dist <= 0:
                continue
            current_dist = abs(extreme_price - entry)
            progress = current_dist / total_dist if total_dist > 0 else 0.0

            if in_profit and use_partial:
                if (not trade.get("partial_taken")) and progress >= 0.50:
                    partial_size = size / 2.0

                    # Partial fill için conservative fiyat kullan
                    if t_type == "LONG":
                        partial_fill = float(partial_fill_price) * (1 - self.slippage_pct)
                        partial_pnl_percent = (partial_fill - entry) / entry
                    else:
                        partial_fill = float(partial_fill_price) * (1 + self.slippage_pct)
                        partial_pnl_percent = (entry - partial_fill) / entry

                    partial_pnl = partial_pnl_percent * (entry * partial_size)
                    partial_notional = abs(partial_size) * abs(partial_fill)
                    commission = partial_notional * TRADING_CONFIG["total_fee"]
                    net_partial_pnl = partial_pnl - commission
                    margin_release = initial_margin / 2.0

                    # Partial TP için R-Multiple hesaplama (yarı risk tutarına göre)
                    trade_risk_amount = float(trade.get("risk_amount", 0))
                    partial_risk = trade_risk_amount / 2.0  # Partial = yarı pozisyon, yarı risk
                    if partial_risk > 0:
                        partial_r_multiple = net_partial_pnl / partial_risk
                        self.trade_r_multiples.append(partial_r_multiple)
                    else:
                        partial_r_multiple = 0.0

                    self.wallet_balance += margin_release + net_partial_pnl
                    self.locked_margin -= margin_release
                    self.total_pnl += net_partial_pnl

                    partial_record = trade.copy()
                    partial_record["size"] = partial_size
                    partial_record["pnl"] = net_partial_pnl
                    partial_record["r_multiple"] = partial_r_multiple
                    partial_record["status"] = "PARTIAL TP (50%)"
                    partial_record["close_time"] = (
                        pd.Timestamp(candle_time_utc) + pd.Timedelta(hours=3)
                    ).strftime("%Y-%m-%d %H:%M")
                    partial_record["close_price"] = float(partial_fill)
                    partial_record["pb_ema_top"] = pb_top
                    partial_record["pb_ema_bot"] = pb_bot
                    partial_record["events"] = json.dumps(trade.get("events", []))
                    self.history.append(partial_record)

                    self.open_trades[i]["size"] = partial_size
                    self.open_trades[i]["notional"] = partial_notional
                    self.open_trades[i]["margin"] = margin_release
                    self.open_trades[i]["partial_taken"] = True
                    self.open_trades[i]["partial_price"] = float(partial_fill)
                    # Breakeven'e çek - küçük buffer ile (spread/slippage koruması)
                    be_buffer = 0.0003  # %0.03 buffer
                    if t_type == "LONG":
                        be_sl = entry * (1 + be_buffer)
                    else:
                        be_sl = entry * (1 - be_buffer)
                    self.open_trades[i]["sl"] = be_sl
                    self.open_trades[i]["breakeven"] = True
                    # Kalan pozisyon için risk tutarını güncelle (yarıya düştü)
                    self.open_trades[i]["risk_amount"] = partial_risk
                    _append_trade_event(self.open_trades[i], "PARTIAL", candle_time_utc, partial_fill)
                    _append_trade_event(self.open_trades[i], "BE_SET", candle_time_utc, be_sl)

                elif (not trade.get("breakeven")) and progress >= 0.40:
                    # Breakeven'e çek - küçük buffer ile
                    be_buffer = 0.0003  # %0.03 buffer
                    if t_type == "LONG":
                        be_sl = entry * (1 + be_buffer)
                    else:
                        be_sl = entry * (1 - be_buffer)
                    self.open_trades[i]["sl"] = be_sl
                    self.open_trades[i]["breakeven"] = True
                    _append_trade_event(self.open_trades[i], "BE_SET", candle_time_utc, be_sl)

            if _apply_1m_profit_lock(self.open_trades[i], tf, t_type, entry, dyn_tp, progress):
                _append_trade_event(self.open_trades[i], "PROFIT_LOCK", candle_time_utc, self.open_trades[i].get("sl"))

            if in_profit and use_trailing:
                if (not trade.get("breakeven")) and progress >= 0.40:
                    # Breakeven'e çek - küçük buffer ile
                    be_buffer = 0.0003  # %0.03 buffer
                    if t_type == "LONG":
                        be_sl = entry * (1 + be_buffer)
                    else:
                        be_sl = entry * (1 - be_buffer)
                    self.open_trades[i]["sl"] = be_sl
                    self.open_trades[i]["breakeven"] = True
                    _append_trade_event(self.open_trades[i], "BE_SET", candle_time_utc, be_sl)

                if progress >= 0.50:
                    trail_buffer = total_dist * 0.40
                    current_sl = float(self.open_trades[i]["sl"])
                    if t_type == "LONG":
                        new_sl = close_price - trail_buffer
                        if new_sl > current_sl:
                            self.open_trades[i]["sl"] = new_sl
                            self.open_trades[i]["trailing_active"] = True
                            _append_trade_event(self.open_trades[i], "TRAIL_SL", candle_time_utc, new_sl)
                    else:
                        new_sl = close_price + trail_buffer
                        if new_sl < current_sl:
                            self.open_trades[i]["sl"] = new_sl
                            self.open_trades[i]["trailing_active"] = True
                            _append_trade_event(self.open_trades[i], "TRAIL_SL", candle_time_utc, new_sl)

            if _apply_partial_stop_protection(self.open_trades[i], tf, progress, t_type):
                _append_trade_event(self.open_trades[i], "STOP_PROTECTION", candle_time_utc, self.open_trades[i].get("sl"))

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
                gross_pnl = (exit_fill - entry) * current_size
            else:
                exit_fill = float(exit_level) * (1 + self.slippage_pct)
                gross_pnl = (entry - exit_fill) * current_size

            commission_notional = abs(current_size) * abs(exit_fill)
            commission = commission_notional * TRADING_CONFIG["total_fee"]

            funding_cost = 0.0
            try:
                open_time_str = trade.get("open_time_utc", "")
                if open_time_str:
                    open_dt = datetime.strptime(open_time_str, "%Y-%m-%dT%H:%M:%SZ")
                    hours = max(0.0, (candle_time_utc - open_dt).total_seconds() / 3600.0)
                    notional_entry = abs(current_size) * entry
                    funding_cost = notional_entry * TRADING_CONFIG["funding_rate_8h"] * (hours / 8.0)
            except Exception:
                funding_cost = 0.0

            final_net_pnl = gross_pnl - commission - funding_cost

            # R-Multiple hesaplaması: PnL / Risk Amount
            # R = 1.0 demek: Riske ettiğiniz kadar kazandınız
            # R = -1.0 demek: Riske ettiğiniz kadar kaybettiniz (tipik SL)
            trade_risk_amount = float(trade.get("risk_amount", 0))
            if trade_risk_amount > 0:
                r_multiple = final_net_pnl / trade_risk_amount
                self.trade_r_multiples.append(r_multiple)
                trade["r_multiple"] = r_multiple
            else:
                trade["r_multiple"] = 0.0

            self.wallet_balance += margin_release + final_net_pnl
            self.locked_margin -= margin_release
            self.total_pnl += final_net_pnl

            if "STOP" in reason:
                wait_minutes = 10 if tf == "1m" else (30 if tf == "5m" else 60)
                cooldown_base = pd.Timestamp(candle_time_utc)
                self.cooldowns[(symbol, tf)] = cooldown_base + pd.Timedelta(minutes=wait_minutes)

            if trade.get("breakeven") and abs(final_net_pnl) < 1e-6 and "STOP" in reason:
                reason = "BE"

            trade["status"] = reason
            trade["pnl"] = final_net_pnl
            trade["close_time"] = (pd.Timestamp(candle_time_utc) + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
            trade["close_price"] = float(exit_fill)
            trade["pb_ema_top"] = pb_top
            trade["pb_ema_bot"] = pb_bot

            trade["events"] = json.dumps(trade.get("events", []))

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
    skip_optimization: bool = False,  # True = cached config kullan, optimizer atla (HIZLI)
    quick_mode: bool = False,  # True = azaltılmış config grid (daha hızlı optimizer)
):
    allowed_log_categories = {"progress", "potential", "summary"}
    strategy_sig = _strategy_signature()

    def log(msg: str, category: str = None):
        if category not in allowed_log_categories:
            return
        if progress_callback:
            progress_callback(msg)
        print(msg)

    parity_report = _audit_trade_logic_parity()
    if parity_report:
        status_icon = "✅" if parity_report.get("parity_ok") else "⚠️"
        diff_text = "" if parity_report.get("parity_ok") else " (sim/real ayrımı tespit edildi)"
        log(
            f"{status_icon} TradeManager/SimTradeManager parity kontrolü tamamlandı{diff_text}.",
            category="summary",
        )

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

    limit_map = limit_map or {}
    requested_pairs = list(itertools.product(symbols, timeframes))

    def build_streams(target_candles: int, write_prices: bool = False):
        result = {}
        active_limit_map = limit_map if limit_map else BACKTEST_CANDLE_LIMITS

        # Paralel veri çekme için iş listesi oluştur
        jobs = []
        for sym in symbols:
            for tf in timeframes:
                tf_candle_limit = active_limit_map.get(tf, target_candles)
                if tf_candle_limit:
                    tf_candle_limit = min(target_candles, tf_candle_limit)
                else:
                    tf_candle_limit = target_candles
                jobs.append((sym, tf, tf_candle_limit))

        total_jobs = len(jobs)
        log(f"📥 {total_jobs} stream için veri indiriliyor...", category="summary")

        def fetch_one(job):
            sym, tf, candle_limit = job
            try:
                df = TradingEngine.get_historical_data_pagination(sym, tf, total_candles=candle_limit)
                if df is None or df.empty or len(df) < 400:
                    return None
                df = TradingEngine.calculate_indicators(df)
                if write_prices:
                    df.to_csv(f"{sym}_{tf}_prices.csv", index=False)
                return (sym, tf, df.reset_index(drop=True))
            except Exception as e:
                return None

        # Paralel veri çekme (max 5 thread - API rate limit'e dikkat)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        completed = 0
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_one, job): job for job in jobs}
            for future in as_completed(futures):
                completed += 1
                res = future.result()
                if res:
                    sym, tf, df = res
                    result[(sym, tf)] = df
                # Her 10 stream'de bir progress göster
                if completed % 10 == 0 or completed == total_jobs:
                    log(f"   📊 {completed}/{total_jobs} stream yüklendi", category="progress")

        log(f"   ✓ {len(result)} stream hazır", category="summary")
        return result

    streams = build_streams(candles, write_prices=True)
    if not streams:
        log("Backtest için veri yok (internet / Binance erişimi?)", category="summary")
        return

    # --- 1) Her sembol/zaman dilimi için en iyi ayarı tara ---
    if skip_optimization:
        # HIZLI MOD: Kayıtlı config'leri kullan, optimizer'ı atla
        log("⚡ [HIZLI MOD] Optimizer atlanıyor, kayıtlı config'ler kullanılıyor...", category="summary")
        best_configs = {}
        if os.path.exists(BEST_CONFIGS_FILE):
            try:
                with open(BEST_CONFIGS_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # JSON formatından tuple key formatına dönüştür
                for sym, tf_dict in raw.items():
                    if isinstance(tf_dict, dict):
                        for tf, cfg in tf_dict.items():
                            if isinstance(cfg, dict):
                                best_configs[(sym, tf)] = cfg
                log(f"   ✓ {len(best_configs)} config yüklendi: {BEST_CONFIGS_FILE}", category="summary")
            except Exception as e:
                log(f"   ⚠️ Config yükleme hatası: {e}", category="summary")
        if not best_configs:
            log("   ⚠️ Kayıtlı config bulunamadı, optimizer çalıştırılıyor...", category="summary")
            skip_optimization = False  # Fallback to optimizer

    if not skip_optimization:
        opt_streams = build_streams(target_candles=1500, write_prices=False)
        best_configs = _optimize_backtest_configs(
            opt_streams,
            requested_pairs,
            progress_callback=progress_callback,
            log_to_stdout=False,
            quick_mode=quick_mode,  # Hızlı mod için azaltılmış config grid
        )

    # --- CONFIG SOURCE LOGGING (for debugging optimizer vs backtest divergence) ---
    enabled_streams = []
    disabled_streams = []
    for (sym, tf) in requested_pairs:
        cfg = best_configs.get((sym, tf), {})
        if cfg.get("disabled", False):
            reason = cfg.get("_reason", "unknown")
            disabled_streams.append(f"{sym}-{tf} ({reason})")
        else:
            conf = cfg.get("_confidence", "high")
            score = cfg.get("_score", 0)
            exp = cfg.get("_expectancy", 0)
            risk_mult = CONFIDENCE_RISK_MULTIPLIER.get(conf, 1.0)
            enabled_streams.append(
                f"{sym}-{tf}: RR={cfg.get('rr', '-')}, RSI={cfg.get('rsi', '-')}, "
                f"Score={score:.1f}, Exp=${exp:.2f}, Risk={risk_mult:.0%}"
            )

    log(f"[CFG] Aktif stream sayısı: {len(enabled_streams)}, Devre dışı: {len(disabled_streams)}", category="summary")

    # PERFORMANCE: Pre-extract NumPy arrays for all streams to avoid df.iloc[i] overhead
    streams_arrays = {}
    for (sym, tf), df in streams.items():
        streams_arrays[(sym, tf)] = {
            "timestamps": pd.to_datetime(df["timestamp"]).values,
            "highs": df["high"].values,
            "lows": df["low"].values,
            "closes": df["close"].values,
            "opens": df["open"].values,
            "pb_tops": df.get("pb_ema_top", df["close"]).values if "pb_ema_top" in df.columns else df["close"].values,
            "pb_bots": df.get("pb_ema_bot", df["close"]).values if "pb_ema_bot" in df.columns else df["close"].values,
        }

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
            (pd.Timestamp(df.loc[warmup, "timestamp"]) + _tf_to_timedelta(tf), sym, tf),
        )

    tm = SimTradeManager(initial_balance=TRADING_CONFIG["initial_balance"])
    logged_cfg_pairs = set()
    processed_events = 0
    next_progress = 10

    # Ana backtest döngüsü
    while heap:
        event_time, sym, tf = heapq.heappop(heap)
        df = streams[(sym, tf)]
        arrays = streams_arrays[(sym, tf)]
        i = ptr[(sym, tf)]
        if i >= len(df) - 1:
            continue

        # Açık pozisyonları güncelle (PBEMA ile birlikte)
        tm.update_trades(
            sym,
            tf,
            candle_high=float(arrays["highs"][i]),
            candle_low=float(arrays["lows"][i]),
            candle_close=float(arrays["closes"][i]),
            candle_time_utc=pd.Timestamp(arrays["timestamps"][i]) + _tf_to_timedelta(tf),
            pb_top=float(arrays["pb_tops"][i]),
            pb_bot=float(arrays["pb_bots"][i]),
        )

        # Bu sembol/timeframe için optimize edilmiş config
        config = best_configs.get((sym, tf)) or load_optimized_config(sym, tf)

        # Skip disabled symbol/timeframe combinations in backtest
        # OPT sets disabled=True for streams with no positive PnL config,
        # or disabled=False for streams with valid config. SYMBOL_PARAMS no longer has hardcoded disabled.
        sym_params = SYMBOL_PARAMS.get(sym, {})
        tf_params = sym_params.get(tf, {}) if isinstance(sym_params, dict) else {}
        if tf_params.get("disabled", False) or config.get("disabled", False):
            continue

        if (sym, tf) not in logged_cfg_pairs:
            logged_cfg_pairs.add((sym, tf))
        rr, rsi, slope = config["rr"], config["rsi"], config["slope"]
        use_at = config.get("at_active", False)

        # Sinyal kontrolü (wrapper ile - keltner_bounce veya pbema_reaction destekler)
        s_type, s_entry, s_tp, s_sl, s_reason = TradingEngine.check_signal(
            df,
            config=config,
            index=i,
            return_debug=False,
        )

        if s_type and "ACCEPTED" in str(s_reason):
            accepted_signals_raw[(sym, tf)] = accepted_signals_raw.get((sym, tf), 0) + 1
            # Aynı sembol/timeframe için açık trade var mı?
            has_open = any(
                t["symbol"] == sym and t["timeframe"] == tf
                for t in tm.open_trades
            )

            cooldown_active = tm.check_cooldown(sym, tf, event_time)
            signal_ts_str = (pd.Timestamp(event_time) + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")

            if has_open:
                log(
                    f"[POT][{sym}-{tf}] {s_type} {signal_ts_str} reddedildi: açık pozisyon var.",
                    category="potential",
                )
            elif cooldown_active:
                log(
                    f"[POT][{sym}-{tf}] {s_type} {signal_ts_str} reddedildi: cooldown aktif.",
                    category="potential",
                )
            else:
                entry_open = float(arrays["opens"][i + 1])
                open_ts = arrays["timestamps"][i + 1]
                ts_str = (pd.Timestamp(open_ts) + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")

                # Setup tag (ör: Base), reason string içinden çekiliyor
                setup_tag = "Unknown"
                s_reason_str = str(s_reason)
                if "ACCEPTED" in s_reason_str and "(" in s_reason_str and ")" in s_reason_str:
                    setup_tag = s_reason_str[s_reason_str.find("(") + 1 : s_reason_str.find(")")]

                # R/R from reason string
                rr_str = ""
                if "R:" in s_reason_str:
                    rr_start = s_reason_str.find("R:") + 2
                    rr_end = s_reason_str.find(")", rr_start) if ")" in s_reason_str[rr_start:] else len(s_reason_str)
                    rr_str = s_reason_str[rr_start:rr_start + (rr_end - rr_start)].split(",")[0].split(")")[0]

                # Try to open trade - only log if successful
                trade_opened = tm.open_trade(
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

                if trade_opened:
                    log(
                        f"[POT][{sym}-{tf}] {s_type} {signal_ts_str} kabul edildi (setup={setup_tag},R:{rr_str}).",
                        category="potential",
                    )
                    opened_signals[(sym, tf)] = opened_signals.get((sym, tf), 0) + 1
                else:
                    log(
                        f"[POT][{sym}-{tf}] {s_type} {signal_ts_str} reddedildi: portföy risk limiti.",
                        category="potential",
                    )

        # Sonraki bara ilerle
        i2 = i + 1
        ptr[(sym, tf)] = i2
        if i2 < len(df) - 1:
            heapq.heappush(
                heap,
                (pd.Timestamp(df.loc[i2, "timestamp"]) + _tf_to_timedelta(tf), sym, tf),
            )

        # Progress log
        processed_events += 1
        if total_events > 0:
            progress = (processed_events / total_events) * 100
            if progress >= next_progress:
                log(f"[BACKTEST] %{progress:.1f} tamamlandı...", category="progress")
                next_progress += 10

    # ==========================================
    # BACKTEST SONUNDA AÇIK POZİSYONLARI KAPAT
    # ==========================================
    # Açık kalan pozisyonları son bilinen fiyattan zorla kapat
    if tm.open_trades:
        log(f"[BACKTEST] {len(tm.open_trades)} açık pozisyon kapatılıyor...", category="summary")
        for trade in list(tm.open_trades):
            sym = trade["symbol"]
            tf = trade["timeframe"]
            entry = float(trade["entry"])
            t_type = trade["type"]
            current_size = float(trade.get("size", 0))
            initial_margin = float(trade.get("margin", 0))

            # Son kapanış fiyatını bul
            if (sym, tf) in streams:
                df = streams[(sym, tf)]
                if not df.empty:
                    last_close = float(df.iloc[-1]["close"])
                else:
                    last_close = entry  # Veri yoksa entry fiyatından kapat
            else:
                last_close = entry

            # PnL hesapla
            if t_type == "LONG":
                exit_fill = last_close * (1 - tm.slippage_pct)
                gross_pnl = (exit_fill - entry) * current_size
            else:
                exit_fill = last_close * (1 + tm.slippage_pct)
                gross_pnl = (entry - exit_fill) * current_size

            # Komisyon
            exit_notional = abs(current_size) * abs(exit_fill)
            commission = exit_notional * TRADING_CONFIG["total_fee"]
            net_pnl = gross_pnl - commission

            # Margin geri ver ve PnL ekle
            tm.wallet_balance += initial_margin + net_pnl
            tm.locked_margin -= initial_margin
            tm.total_pnl += net_pnl

            # Trade'i history'ye ekle
            trade["status"] = "FORCE_CLOSE"
            trade["pnl"] = net_pnl
            trade["close_price"] = exit_fill
            trade["close_time"] = "BACKTEST_END"
            tm.history.append(trade)

        # Açık trade listesini temizle
        tm.open_trades.clear()
        log(f"[BACKTEST] Açık pozisyonlar kapatıldı. Yeni bakiye: ${tm.wallet_balance:.2f}", category="summary")

    total_closed_legs = len(tm.history)
    unique_trades = len({t.get("id") for t in tm.history}) if tm.history else 0
    partial_legs = sum(1 for t in tm.history if "PARTIAL" in str(t.get("status", "")))
    full_exits = total_closed_legs - partial_legs

    # Tüm history'den DataFrame oluştur ve CSV / özet yaz
    trades_df = pd.DataFrame(tm.history)
    if not trades_df.empty:
        trades_df.to_csv(out_trades_csv, index=False)

    summary_rows = []
    strategy_summary_rows = []  # Strateji bazlı özet için
    if not trades_df.empty:
        # Aynı id'ye ait tüm bacakları (partial + final) toplayıp
        # trade başına net sonucu hesapla
        # Walk-Forward aktifse, sadece OOS dönemindeki trade'leri say
        grouped_by_trade = {}
        grouped_by_trade_all = {}  # Tüm trade'ler (karşılaştırma için)
        grouped_by_strategy = {}  # Strateji bazlı gruplandırma

        for (sym, tf, tid), g in trades_df.groupby(["symbol", "timeframe", "id"]):
            net = g["pnl"].astype(float).sum()
            key = (sym, tf)

            # Setup/strategy bilgisini al (örn: "Base,R:2.5" -> "Base" veya "PBEMA_Reaction,R:2.5" -> "PBEMA_Reaction")
            setup_raw = g["setup"].iloc[0] if "setup" in g.columns else "Unknown"
            strategy = setup_raw.split(",")[0] if "," in str(setup_raw) else str(setup_raw)

            # Tüm trade'leri kaydet (karşılaştırma için)
            grouped_by_trade_all.setdefault(key, []).append(net)

            # Strateji bazlı gruplandırma
            strategy_key = (sym, tf, strategy)
            grouped_by_strategy.setdefault(strategy_key, []).append(net)

            # OOS filtreleme: Eğer bu stream için OOS start time varsa, sadece OOS trade'leri say
            opt_cfg = best_configs.get((sym, tf), {})
            oos_start = opt_cfg.get("_oos_start_time")

            if oos_start is not None:
                # Trade'in açılış zamanını al
                trade_time = g["timestamp"].iloc[0] if "timestamp" in g.columns else None
                if trade_time is not None:
                    # Timestamp tiplerini eşitle (string vs Timestamp uyumsuzluğunu çöz)
                    try:
                        trade_time_ts = pd.to_datetime(trade_time)
                        oos_start_ts = pd.to_datetime(oos_start)
                        # Trade OOS döneminde mi?
                        if trade_time_ts >= oos_start_ts:
                            grouped_by_trade.setdefault(key, []).append(net)
                        # OOS öncesi trade'ler atlanır (curve-fitted dönem)
                    except Exception:
                        # Dönüştürme başarısız olursa tüm trade'leri say
                        grouped_by_trade.setdefault(key, []).append(net)
                else:
                    # Timestamp yoksa tüm trade'leri say (fallback)
                    grouped_by_trade.setdefault(key, []).append(net)
            else:
                # OOS start time yoksa tüm trade'leri say
                grouped_by_trade.setdefault(key, []).append(net)

        for (sym, tf), pnl_list in grouped_by_trade.items():
            total = len(pnl_list)
            wins = sum(1 for x in pnl_list if x > 0)
            pnl = sum(pnl_list)
            wr = (wins / total * 100.0) if total else 0.0

            # Karşılaştırma için tüm trade sayısını da kaydet
            all_trades = len(grouped_by_trade_all.get((sym, tf), []))
            oos_filtered = all_trades > total  # OOS filtreleme yapıldı mı?

            summary_rows.append(
                {
                    "symbol": sym,
                    "timeframe": tf,
                    "trades": total,
                    "win_rate_pct": wr,
                    "net_pnl": pnl,
                    "_all_trades": all_trades if oos_filtered else None,  # Filtreleme yapıldıysa göster
                }
            )

        # Strateji bazlı özet oluştur
        for (sym, tf, strategy), pnl_list in grouped_by_strategy.items():
            total = len(pnl_list)
            wins = sum(1 for x in pnl_list if x > 0)
            pnl = sum(pnl_list)
            wr = (wins / total * 100.0) if total else 0.0
            strategy_summary_rows.append(
                {
                    "strategy": strategy,
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

    # Strateji bazlı özet DataFrame oluştur ve kaydet
    strategy_summary_df = (
        pd.DataFrame(strategy_summary_rows).sort_values(["strategy", "symbol", "timeframe"])
        if strategy_summary_rows
        else pd.DataFrame()
    )
    strategy_summary_csv = out_summary_csv.replace(".csv", "_by_strategy.csv")
    if not strategy_summary_df.empty:
        strategy_summary_df.to_csv(strategy_summary_csv, index=False)

    log("Backtest bitti.", category="summary")

    # OOS filtreleme yapıldıysa bilgi ver
    oos_filtered_streams = [r for r in summary_rows if r.get("_all_trades") is not None]
    if oos_filtered_streams:
        log(
            f"\n📊 [OOS FİLTRE] {len(oos_filtered_streams)} stream için sadece Out-of-Sample (test dönemi) trade'leri sayıldı.",
            category="summary"
        )
        log("   (Training dönemindeki trade'ler curve-fitted olduğu için dışlandı)", category="summary")

    if not summary_df.empty:
        log(summary_df.to_string(index=False), category="summary")

    # Strateji bazlı özet tablosu yazdır
    if not strategy_summary_df.empty:
        log("\n" + "=" * 60, category="summary")
        log("📊 STRATEJİ BAZLI ÖZET (Keltner Bounce vs PBEMA Reaction)", category="summary")
        log("=" * 60, category="summary")
        log(strategy_summary_df.to_string(index=False), category="summary")

        # Strateji toplamları
        log("\n📈 STRATEJİ TOPLAMLARI:", category="summary")
        for strategy in strategy_summary_df["strategy"].unique():
            strat_data = strategy_summary_df[strategy_summary_df["strategy"] == strategy]
            total_trades = strat_data["trades"].sum()
            total_pnl = strat_data["net_pnl"].sum()
            avg_wr = strat_data["win_rate_pct"].mean()
            streams_count = len(strat_data)
            pnl_sign = "+" if total_pnl >= 0 else ""
            log(
                f"  [{strategy}] Streams: {streams_count}, Trades: {total_trades}, "
                f"Net PnL: {pnl_sign}${total_pnl:.2f}, Avg WR: {avg_wr:.1f}%",
                category="summary"
            )
        log("=" * 60 + "\n", category="summary")

    if best_configs:
        # Filter out disabled configs for summary
        active_configs = {k: v for k, v in best_configs.items() if not v.get("disabled", False)}
        disabled_count = len(best_configs) - len(active_configs)

        if active_configs:
            log("\n[OPT] En iyi ayar özeti (Net PnL'e göre):", category="summary")
            for (sym, tf), cfg in sorted(active_configs.items()):
                # skip_optimization modunda _net_pnl/_trades olmayabilir
                net_pnl = cfg.get('_net_pnl', 0)
                trades = cfg.get('_trades', 0)
                strategy_mode = cfg.get('strategy_mode', 'keltner_bounce')
                strategy_tag = "KB" if strategy_mode == "keltner_bounce" else "PR"  # KB=Keltner Bounce, PR=PBEMA Reaction
                log(
                    f"  - {sym}-{tf} [{strategy_tag}]: RR={cfg.get('rr', '-')}, RSI={cfg.get('rsi', '-')}, "
                    f"AT={'Açık' if cfg.get('at_active') else 'Kapalı'}, Trailing={cfg.get('use_trailing', False)} | "
                    f"NetPnL={net_pnl:.2f}, Trades={trades}",
                    category="summary",
                )
        if disabled_count > 0:
            log(f"\n[OPT] {disabled_count} stream devre dışı bırakıldı (pozitif PnL'li config bulunamadı)", category="summary")

    # --- OPTIMIZER VS BACKTEST DIVERGENCE DETECTION ---
    # Compare OOS predictions with actual OOS backtest results
    # (OOS = Out-of-Sample, training dönemindeki trade'ler hariç)
    divergent_streams = []
    for row in summary_rows:
        sym, tf = row["symbol"], row["timeframe"]
        actual_pnl = row["net_pnl"]
        actual_trades = row["trades"]

        opt_cfg = best_configs.get((sym, tf), {})
        if opt_cfg.get("disabled", False):
            continue  # Skip disabled streams

        # Walk-Forward varsa OOS değerlerini kullan, yoksa training değerlerini
        if opt_cfg.get("_walk_forward_validated"):
            opt_pnl = opt_cfg.get("_oos_pnl", 0)
            opt_trades = opt_cfg.get("_oos_trades", 0)
        else:
            opt_pnl = opt_cfg.get("_net_pnl", 0)
            opt_trades = opt_cfg.get("_trades", 0)

        # Detect significant divergence: optimizer predicted positive but actual negative
        if opt_pnl > 0 and actual_pnl < -10:  # Allow some tolerance
            divergent_streams.append({
                "stream": f"{sym}-{tf}",
                "opt_pnl": opt_pnl,
                "actual_pnl": actual_pnl,
                "opt_trades": opt_trades,
                "actual_trades": actual_trades,
                "divergence": actual_pnl - opt_pnl,
            })

    if divergent_streams:
        log(f"\n⚠️ [DIVERGENCE] {len(divergent_streams)} stream OOS beklenti vs gerçek uyuşmazlığı:", category="summary")
        for d in sorted(divergent_streams, key=lambda x: x["divergence"]):
            log(
                f"  - {d['stream']}: OOS=${d['opt_pnl']:.0f}({d['opt_trades']}tr) → "
                f"ACTUAL=${d['actual_pnl']:.0f}({d['actual_trades']}tr) [Δ${d['divergence']:.0f}]",
                category="summary"
            )
        log("  (Nedenler: portfolio constraint'ler, pozisyon çakışmaları, risk limitleri)", category="summary")

    log(
        f"Final Wallet (sim): ${tm.wallet_balance:.2f} | Total PnL: ${tm.total_pnl:.2f}",
        category="summary",
    )

    total_trades = trades_df["id"].nunique() if not trades_df.empty and "id" in trades_df.columns else 0
    if total_trades < 5:
        log(
            "[BACKTEST] Çok az trade bulundu. Daha fazla sonuç için RR/RSI/Slope limitlerini biraz gevşetmeyi düşünebilirsin.",
            category="summary",
        )

    # Sonuçları GUI/LIVE ile paylaşmak için kaydet
    save_best_configs(best_configs)
    result = {
        "summary": summary_rows,
        "best_configs": best_configs,
        "trades_csv": out_trades_csv,
        "summary_csv": out_summary_csv,
        "strategy_signature": strategy_sig,
        "parity_report": parity_report,
    }

    if draw_trades:
        try:
            replay_backtest_trades(trades_csv=out_trades_csv, max_trades=max_draw_trades)
        except Exception as e:
            log(f"[BACKTEST] Trade çiziminde hata: {e}", category="summary")

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

    def _status_key(status_text: str) -> str:
        s = (status_text or "").upper()
        if "PARTIAL" in s:
            return "PARTIAL"
        if "BOTH" in s:
            return "STOP_BOTH"
        if "WIN" in s or "TP" in s:
            return "WIN"
        if s == "BE" or "BREAKEVEN" in s:
            return "BE"
        if "STOP" in s:
            return "STOP"
        return "OTHER"

    status_palette = {
        "WIN": {"color": "#00c853", "marker": "^", "label": "Take Profit"},
        "STOP": {"color": "#ef5350", "marker": "v", "label": "Stop"},
        "STOP_BOTH": {"color": "#ff9800", "marker": "v", "label": "SL/TP (Same Candle)"},
        "BE": {"color": "#90a4ae", "marker": "X", "label": "Breakeven"},
        "PARTIAL": {"color": "#00bcd4", "marker": "D", "label": "Partial"},
        "EV_BE_SET": {"color": "#8d6e63", "marker": "|", "label": "SL BE/Taşındı"},
        "EV_TRAIL_SL": {"color": "#ab47bc", "marker": "_", "label": "Trailing SL"},
        "EV_PROFIT_LOCK": {"color": "#26c6da", "marker": "s", "label": "%80 Profit Lock"},
        "EV_STOP_PROTECTION": {"color": "#ffb74d", "marker": "1", "label": "Koruma SL"},
        "OTHER": {"color": "#ab47bc", "marker": "o", "label": "Exit"},
    }

    rr_colors = {
        "WIN": {"profit": "#1de9b6", "loss": "#ef5350"},
        "STOP": {"profit": "#81c784", "loss": "#ef5350"},
        "STOP_BOTH": {"profit": "#ffb74d", "loss": "#ff7043"},
        "BE": {"profit": "#90a4ae", "loss": "#ef5350"},
        "PARTIAL": {"profit": "#00bcd4", "loss": "#ef5350"},
        "OTHER": {"profit": "#64b5f6", "loss": "#ef5350"},
    }

    def _parse_event_time(row) -> Optional[pd.Timestamp]:
        close_time_val = row.get("close_time")
        if close_time_val:
            try:
                ts = pd.to_datetime(close_time_val)
                if ts.tzinfo is None:
                    ts = (ts - pd.Timedelta(hours=3)).tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                return ts
            except Exception:
                pass

        for fallback_field in ("open_time_utc", "timestamp"):
            if fallback_field in row:
                try:
                    ts = pd.to_datetime(row.get(fallback_field), utc=True)
                    return ts
                except Exception:
                    continue
        return None

    def _parse_event_log(raw_val):
        if isinstance(raw_val, str):
            try:
                return json.loads(raw_val)
            except Exception:
                return []
        if isinstance(raw_val, list):
            return raw_val
        return []

    event_points = []
    for _, ev in trades_for_id.iterrows():
        status_val = str(ev.get("status", "")).strip()
        if not status_val or status_val.upper() == "OPEN":
            continue

        event_ts = _parse_event_time(ev)
        price_val = ev.get("close_price", np.nan)
        status_key = _status_key(status_val)

        if (pd.isna(price_val) or price_val == ""):
            if status_key in {"WIN", "PARTIAL"}:
                price_val = ev.get("tp", np.nan)
            elif status_key in {"STOP", "STOP_BOTH", "BE"}:
                price_val = ev.get("sl", np.nan)
            else:
                price_val = ev.get("entry", np.nan)

        event_points.append(
            {
                "time": event_ts,
                "price": float(price_val) if not pd.isna(price_val) else None,
                "status": status_val,
                "key": status_key,
            }
        )

        for raw_ev in _parse_event_log(ev.get("events")):
            try:
                ev_type = str(raw_ev.get("type", "")).upper()
                ev_key = f"EV_{ev_type}" if ev_type else "OTHER"
                ev_time = pd.to_datetime(raw_ev.get("time"), utc=True, errors="coerce")
                if pd.isna(ev_time):
                    ev_time = None
                event_points.append(
                    {
                        "time": ev_time,
                        "price": raw_ev.get("price"),
                        "status": ev_type,
                        "key": ev_key if ev_key in status_palette else "OTHER",
                    }
                )
            except Exception:
                continue

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
    status_key_final = _status_key(status)
    status_labels = [p["status"] for p in event_points] or [status]
    status_text = " / ".join(dict.fromkeys(status_labels))
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

    palette_colors = rr_colors.get(status_key_final, rr_colors["OTHER"])
    if ttype == "LONG":
        # Reward kutusu: entry -> TP
        y_prof0, y_prof1 = sorted((entry, tp))
        # Risk kutusu: SL -> entry
        y_loss0, y_loss1 = sorted((sl, entry))
        draw_rr_box(y_prof0, y_prof1, palette_colors["profit"], alpha=0.25)
        draw_rr_box(y_loss0, y_loss1, palette_colors["loss"], alpha=0.25)
    elif ttype == "SHORT":
        # Reward kutusu: TP -> entry
        y_prof0, y_prof1 = sorted((tp, entry))
        # Risk kutusu: entry -> SL
        y_loss0, y_loss1 = sorted((entry, sl))
        draw_rr_box(y_prof0, y_prof1, palette_colors["profit"], alpha=0.25)
        draw_rr_box(y_loss0, y_loss1, palette_colors["loss"], alpha=0.25)

    # Çıkış / partial / breakeven noktalarını vurgula
    legend_labels = set()
    for ev in event_points:
        if ev["time"] is None:
            continue

        style = status_palette.get(ev["key"], status_palette["OTHER"])
        marker_label = style["label"] if style["label"] not in legend_labels else None
        legend_labels.add(style["label"])

        ax.axvline(ev["time"], linestyle="--", linewidth=0.8, color=style["color"], alpha=0.6)
        ax.scatter(
            ev["time"],
            ev["price"] if ev["price"] is not None else entry,
            s=85,
            marker=style["marker"],
            color=style["color"],
            zorder=6,
            label=marker_label,
        )

        label_text = ev["status"]
        if ev.get("price") is not None:
            try:
                label_text = f"{ev['status']}\n@ {ev['price']:.2f}"
            except Exception:
                label_text = ev["status"]

        ax.annotate(
            label_text,
            xy=(ev["time"], ev["price"] if ev["price"] is not None else entry),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=style["color"],
            bbox=dict(boxstyle="round,pad=0.25", fc="#0f0f0f", ec=style["color"], alpha=0.45),
        )

    # Başlık, grid, legend
    ax.set_title(f"{symbol} {timeframe} | Trade ID {trade_id} — {ttype} ({status_text})")
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


def _launch_application_once() -> int:
    """Uygulamayı tek seferlik başlatır ve çıkış kodunu döner."""

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    exit_code = app.exec_()

    # Event loop kapandıktan sonra uygulamayı tamamen temizle
    try:
        app.quit()
    except Exception:
        # Qt bazı platformlarda tekrar tekrar quit çağrısına izin vermeyebilir
        pass

    return exit_code


def run_with_auto_restart(restart_delay: int = AUTO_RESTART_DELAY_SECONDS) -> None:
    """Çökme veya kapanma sonrası uygulamayı otomatik yeniden başlatır."""

    restart_counter = 0
    while True:
        restart_counter += 1
        try:
            exit_code = _launch_application_once()
            print(
                f"[RESTART] Uygulama döngüsü {restart_counter} sona erdi (exit={exit_code})."
            )
        except Exception as exc:  # En dış seviye güvenlik ağı
            print(f"[RESTART] Uygulama hatası: {exc}\n{traceback.format_exc()}")

        print(
            f"[RESTART] {restart_delay} saniye sonra yeniden başlatılıyor..."
        )
        time.sleep(restart_delay)


# ==========================================
# 🚀 GOOGLE COLAB / CLI BACKTEST FUNCTIONS
# ==========================================
# These functions allow running backtests without GUI (for Colab, servers, etc.)

def run_cli_backtest(
    symbols: list = None,
    timeframes: list = None,
    candles: int = 50000,
    optimize: bool = True,
    walk_forward: bool = True,
    save_results: bool = True,
    verbose: bool = True,
) -> dict:
    """Run a complete backtest from CLI/Colab without GUI.

    Args:
        symbols: List of symbols to test (default: SYMBOLS)
        timeframes: List of timeframes to test (default: TIMEFRAMES)
        candles: Number of candles per timeframe (default: 50000)
        optimize: Whether to run optimization first (default: True)
        walk_forward: Enable walk-forward OOS validation (default: True)
        save_results: Save results to CSV and JSON (default: True)
        verbose: Print progress messages (default: True)

    Returns:
        dict with:
        - 'summary': DataFrame with backtest summary
        - 'trades': DataFrame with all trades
        - 'configs': dict of best configs per stream
        - 'metrics': dict of aggregate metrics

    Example usage in Colab:
        >>> from desktop_bot_refactored_v2_base_v7 import run_cli_backtest
        >>> results = run_cli_backtest(
        ...     symbols=['BTCUSDT', 'ETHUSDT'],
        ...     timeframes=['5m', '15m', '1h'],
        ...     candles=30000,
        ...     optimize=True
        ... )
        >>> print(results['summary'])
    """
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES

    def log(msg):
        if verbose:
            print(msg)

    log("=" * 60)
    log("🚀 CLI/Colab Backtest Başlatılıyor")
    log(f"   Symbols: {symbols}")
    log(f"   Timeframes: {timeframes}")
    log(f"   Candles: {candles}")
    log(f"   Optimize: {optimize}")
    log(f"   Walk-Forward: {walk_forward}")
    log("=" * 60)

    # Step 1: Fetch data for all symbol/timeframe pairs
    log("\n📊 Veri indiriliyor...")
    streams = {}
    pairs = [(s, tf) for s in symbols for tf in timeframes]

    # Use tqdm if available for progress bar
    iterator = tqdm(pairs, desc="Veri indirme") if HAS_TQDM else pairs

    for sym, tf in iterator:
        try:
            limit = BACKTEST_CANDLE_LIMITS.get(tf, candles)
            limit = min(limit, candles)

            # Fetch data using the existing TradingEngine method
            df = TradingEngine.fetch_historical_data_paginated(sym, tf, limit)
            if df is not None and len(df) > 300:
                # Calculate indicators
                df = TradingEngine.calculate_indicators(df)
                streams[(sym, tf)] = df
                if not HAS_TQDM:
                    log(f"   ✓ {sym}-{tf}: {len(df)} mum")
            else:
                if not HAS_TQDM:
                    log(f"   ✗ {sym}-{tf}: Yetersiz veri")
        except Exception as e:
            log(f"   ✗ {sym}-{tf}: Hata - {e}")

    if not streams:
        log("❌ Hiç veri indirilemedi!")
        return {"summary": None, "trades": None, "configs": {}, "metrics": {}}

    log(f"\n✓ {len(streams)} stream hazır")

    # Step 2: Run optimization if enabled
    best_configs = {}
    if optimize:
        log("\n⚙️ Optimizasyon başlatılıyor...")

        def opt_progress(msg):
            if verbose and not HAS_TQDM:
                print(f"   {msg}")

        best_configs = _optimize_backtest_configs(
            streams=streams,
            requested_pairs=list(streams.keys()),
            progress_callback=opt_progress,
            log_to_stdout=verbose,
            use_walk_forward=walk_forward,
        )

        # Save configs
        if save_results and best_configs:
            save_best_configs(best_configs)
            log("   ✓ En iyi ayarlar kaydedildi: best_configs.json")

    # Step 3: Run final backtest with optimized configs
    log("\n🔬 Final backtest çalıştırılıyor...")

    result = run_portfolio_backtest(
        symbols=symbols,
        timeframes=timeframes,
        candles=candles,
        out_trades_csv="backtest_trades.csv" if save_results else None,
        out_summary_csv="backtest_summary.csv" if save_results else None,
        progress_callback=lambda msg: log(f"   {msg}") if verbose else None,
        draw_trades=False,  # Don't draw charts in CLI mode
    )

    # Extract results
    summary_rows = result.get("summary_rows", [])
    all_trades = result.get("all_trades", [])

    # Convert to DataFrames
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    # Calculate aggregate metrics
    metrics = {}
    if not trades_df.empty:
        pnl_col = 'pnl' if 'pnl' in trades_df.columns else None
        if pnl_col:
            metrics['total_pnl'] = trades_df[pnl_col].sum()
            metrics['total_trades'] = len(trades_df)
            metrics['winning_trades'] = len(trades_df[trades_df[pnl_col] > 0])
            metrics['losing_trades'] = len(trades_df[trades_df[pnl_col] < 0])
            metrics['win_rate'] = metrics['winning_trades'] / max(1, metrics['total_trades'])
            metrics['avg_pnl'] = metrics['total_pnl'] / max(1, metrics['total_trades'])

            # R-Multiple metrics
            if 'r_multiple' in trades_df.columns:
                metrics['total_r'] = trades_df['r_multiple'].sum()
                metrics['avg_r'] = trades_df['r_multiple'].mean()
                metrics['expected_r'] = metrics['avg_r']

    # Print summary
    log("\n" + "=" * 60)
    log("📈 BACKTEST SONUÇLARI")
    log("=" * 60)
    if metrics:
        log(f"   Toplam PnL: ${metrics.get('total_pnl', 0):.2f}")
        log(f"   Toplam Trade: {metrics.get('total_trades', 0)}")
        log(f"   Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
        log(f"   E[R]: {metrics.get('expected_r', 0):.3f}")
    log("=" * 60)

    return {
        "summary": summary_df,
        "trades": trades_df,
        "configs": best_configs,
        "metrics": metrics,
    }


def colab_quick_test(
    symbol: str = "BTCUSDT",
    timeframe: str = "15m",
    candles: int = 10000,
) -> dict:
    """Quick single-stream test for Colab experimentation.

    Args:
        symbol: Symbol to test (default: BTCUSDT)
        timeframe: Timeframe to test (default: 15m)
        candles: Number of candles (default: 10000)

    Returns:
        dict with test results
    """
    return run_cli_backtest(
        symbols=[symbol],
        timeframes=[timeframe],
        candles=candles,
        optimize=True,
        walk_forward=True,
        verbose=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trading Bot v39.0 - R-Multiple Based with Walk-Forward")
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (CLI backtest)')
    parser.add_argument('--symbols', nargs='+', help='Symbols to test (e.g., BTCUSDT ETHUSDT)')
    parser.add_argument('--timeframes', nargs='+', help='Timeframes to test (e.g., 5m 15m 1h)')
    parser.add_argument('--candles', type=int, default=50000, help='Number of candles (default: 50000)')
    parser.add_argument('--no-optimize', action='store_true', help='Skip optimization')
    parser.add_argument('--no-walk-forward', action='store_true', help='Disable walk-forward validation')

    args = parser.parse_args()

    # Determine run mode
    if args.headless or IS_HEADLESS or not HAS_GUI:
        # CLI/Colab mode
        print("\n🖥️ CLI/Colab Mod - GUI olmadan çalışıyor\n")
        results = run_cli_backtest(
            symbols=args.symbols,
            timeframes=args.timeframes,
            candles=args.candles,
            optimize=not args.no_optimize,
            walk_forward=not args.no_walk_forward,
        )
        print("\n✅ Backtest tamamlandı!")
        if results['metrics']:
            print(f"   Total PnL: ${results['metrics'].get('total_pnl', 0):.2f}")
            print(f"   E[R]: {results['metrics'].get('expected_r', 0):.3f}")
    else:
        # GUI mode
        run_with_auto_restart()






