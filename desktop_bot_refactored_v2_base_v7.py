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

# ==========================================
# 🆕 MODULAR CORE PACKAGE INTEGRATION (v40.0)
# ==========================================
# The core package provides refactored modules for better maintainability:
# - core.config: Centralized configuration
# - core.utils: Helper functions (time conversion, funding calculation)
# - core.trade_manager: BaseTradeManager with shared logic
# - core.telegram: Improved Telegram notifications
#
# Use environment variables for Telegram (more secure than config.json):
# - TELEGRAM_BOT_TOKEN
# - TELEGRAM_CHAT_ID
#
# Key improvements:
# - Eliminated code duplication between TradeManager and SimTradeManager
# - Fixed funding calculation to use actual hours (not bar count)
# - Fixed UTC/time naming (close_time_utc vs close_time_local)
# - Thread pool for Telegram (prevents thread accumulation)
# ==========================================
# ==========================================
# 🔗 CORE PACKAGE - ZORUNLU MODÜLLER (v40.1)
# ==========================================
# Core paketi bu repoda bulunur ve zorunludur.
# Fallback kodlar kaldırıldı - tek kaynak core paketi.
from core import (
    # Utils
    utcnow,  # Replacement for deprecated utcnow()
    normalize_datetime,
    tf_to_timedelta,
    calculate_funding_cost,
    format_time_utc,
    format_time_local,
    append_trade_event,
    apply_1m_profit_lock,
    apply_partial_stop_protection,
    calculate_r_multiple,
    calculate_expected_r,
    # Trade managers
    SimTradeManager,
    BaseTradeManager,
    # Telegram
    send_telegram as _core_send_telegram,
    get_notifier,
    TelegramNotifier,
    save_telegram_config,
    load_telegram_config,
    # Binance client
    BinanceClient,
    get_client,
    # Indicators
    calculate_indicators as core_calculate_indicators,
    calculate_alphatrend as core_calculate_alphatrend,
    get_indicator_value,
    get_candle_data,
    # Config - Blacklist functions imported from core (single source of truth)
    is_stream_blacklisted as core_is_stream_blacklisted,
    load_dynamic_blacklist as core_load_dynamic_blacklist,
    update_dynamic_blacklist as core_update_dynamic_blacklist,
    save_dynamic_blacklist as core_save_dynamic_blacklist,
    DYNAMIC_BLACKLIST_CONFIG as CORE_DYNAMIC_BLACKLIST_CONFIG,
    DYNAMIC_BLACKLIST_CACHE as CORE_DYNAMIC_BLACKLIST_CACHE,
    POST_PORTFOLIO_BLACKLIST as CORE_POST_PORTFOLIO_BLACKLIST,
    DYNAMIC_BLACKLIST_FILE as CORE_DYNAMIC_BLACKLIST_FILE,
    # Strategy configs (single source of truth for signature generation)
    DEFAULT_STRATEGY_CONFIG, SYMBOL_PARAMS, TRADING_CONFIG,
    # Trading Engine (core trading logic, signals, data fetching)
    TradingEngine,
    # File paths - SINGLE SOURCE OF TRUTH (v40.2)
    # All file paths come from core.config to avoid path mismatches
    DATA_DIR, CSV_FILE, CONFIG_FILE, BEST_CONFIGS_FILE,
    BACKTEST_META_FILE, POT_LOG_FILE,
    BACKTEST_CANDLE_LIMITS, DAILY_REPORT_CANDLE_LIMITS,
    CANDLES_PER_DAY, MINUTES_PER_CANDLE,
    days_to_candles, days_to_candles_map,
    # Symbols and Timeframes - SINGLE SOURCE OF TRUTH (v40.3)
    SYMBOLS, TIMEFRAMES, LOWER_TIMEFRAMES, HTF_TIMEFRAMES, HTF_ONLY_MODE,
    # Thresholds - SINGLE SOURCE OF TRUTH (v40.3)
    MIN_EXPECTANCY_R_MULTIPLE, MIN_SCORE_THRESHOLD, CONFIDENCE_RISK_MULTIPLIER,
    # Walk-forward and circuit breaker configs - SINGLE SOURCE OF TRUTH (v40.3)
    WALK_FORWARD_CONFIG, MIN_OOS_TRADES_BY_TF,
    CIRCUIT_BREAKER_CONFIG, ROLLING_ER_CONFIG,
    # Config loader functions - SINGLE SOURCE OF TRUTH (v40.4)
    load_optimized_config, save_best_configs, invalidate_config_cache,
    _strategy_signature,
    # Optimizer functions - SINGLE SOURCE OF TRUTH (v40.5)
    _optimize_backtest_configs, _generate_candidate_configs, _generate_quick_candidate_configs,
    _get_min_trades_for_timeframe, _split_data_walk_forward, _validate_config_oos,
    _check_overfit, _compute_optimizer_score, _score_config_for_stream,
    MIN_EXPECTANCY_PER_TRADE, STRATEGY_BLACKLIST,
)

# ==========================================
# 🚀 FAST STARTUP - Lazy imports for heavy libraries
# ==========================================
# Performance note: pandas_ta and plotly are imported lazily
# to reduce startup time from ~30-40s to ~10-15s

import pandas as pd
import numpy as np
import requests
import dateutil.parser
import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta
import traceback
import tempfile
import shutil
from typing import Tuple, Optional, Dict
import matplotlib
import hashlib

# Lazy import wrapper for pandas_ta (heavy library ~5-10s import time)
_ta = None
def get_ta():
    """Lazy load pandas_ta when first needed."""
    global _ta
    if _ta is None:
        import pandas_ta as ta_module
        _ta = ta_module
    return _ta

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
# Lazy import for matplotlib.pyplot - only import when needed
_plt = None
def get_plt():
    """Lazy load matplotlib.pyplot when first needed."""
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt_module
        _plt = plt_module
    return _plt

# PyQt5 Modülleri - sadece GUI modunda yükle
# NOT: QWebEngineView lazy import edilir (çok ağır - 15-25 saniye)
if not IS_HEADLESS:
    try:
        from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                     QHBoxLayout, QGridLayout, QTabWidget, QTextEdit, QLabel,
                                     QTableWidget, QTableWidgetItem, QPushButton, QHeaderView,
                                     QGroupBox, QDoubleSpinBox, QComboBox, QMessageBox, QCheckBox,
                                     QLineEdit, QSpinBox, QFrame, QRadioButton, QDateEdit)
        # QWebEngineView lazy import - sadece grafik gösterildiğinde yüklenir
        _QWebEngineView = None
        def get_QWebEngineView():
            """Lazy load QWebEngineView when first needed (saves 15-25s startup time)."""
            global _QWebEngineView
            if _QWebEngineView is None:
                from PyQt5.QtWebEngineWidgets import QWebEngineView as WebEngine
                _QWebEngineView = WebEngine
            return _QWebEngineView
        from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
        from PyQt5.QtGui import QColor, QFont
        HAS_GUI = True
    except ImportError:
        # Only print warning once in main process (avoid spam in parallel workers)
        import multiprocessing
        if multiprocessing.current_process().name == 'MainProcess':
            print("⚠️ PyQt5 bulunamadı - GUI devre dışı, sadece CLI modu kullanılabilir")
        HAS_GUI = False
        IS_HEADLESS = True
        # Placeholder classes when PyQt5 import fails
        class QThread:
            def __init__(self, *args, **kwargs): pass
            def start(self): pass
            def wait(self): pass
            def isRunning(self): return False
            def quit(self): pass
            def terminate(self): pass

        class pyqtSignal:
            def __init__(self, *args, **kwargs): pass
            def emit(self, *args, **kwargs): pass
            def connect(self, *args, **kwargs): pass

        class QTimer:
            def __init__(self, *args, **kwargs): pass
            def start(self, *args): pass
            def stop(self): pass

        class Qt:
            AlignCenter = 0
            AlignLeft = 0
            AlignRight = 0

        class QMainWindow:
            """Placeholder QMainWindow for when PyQt5 is not available."""
            def __init__(self, *args, **kwargs): pass
            def setWindowTitle(self, *args): pass
            def setGeometry(self, *args): pass
            def setStyleSheet(self, *args): pass
            def setCentralWidget(self, *args): pass
            def show(self): pass
            def close(self): pass
else:
    HAS_GUI = False
    # Placeholder classes for headless mode - allows class definitions to work
    class QThread:
        """Placeholder QThread for headless mode."""
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def wait(self): pass
        def isRunning(self): return False
        def quit(self): pass
        def terminate(self): pass

    class pyqtSignal:
        """Placeholder pyqtSignal for headless mode."""
        def __init__(self, *args, **kwargs): pass
        def emit(self, *args, **kwargs): pass
        def connect(self, *args, **kwargs): pass

    # Other placeholder classes
    class QTimer:
        def __init__(self, *args, **kwargs): pass
        def start(self, *args): pass
        def stop(self): pass

    class Qt:
        AlignCenter = 0
        AlignLeft = 0
        AlignRight = 0

    class QMainWindow:
        """Placeholder QMainWindow for headless mode."""
        def __init__(self, *args, **kwargs): pass
        def setWindowTitle(self, *args): pass
        def setGeometry(self, *args): pass
        def setStyleSheet(self, *args): pass
        def setCentralWidget(self, *args): pass
        def show(self): pass
        def close(self): pass

# Lazy import for plotly (heavy library, only needed for charts)
_plotly_go = None
_plotly_utils = None
def get_plotly():
    """Lazy load plotly when first needed."""
    global _plotly_go, _plotly_utils
    if _plotly_go is None:
        import plotly.graph_objects as go_module
        import plotly.utils as utils_module
        _plotly_go = go_module
        _plotly_utils = utils_module
    return _plotly_go, _plotly_utils

# ==========================================
# ⚙️ GENEL AYARLAR VE SABİTLER (MERKEZİ YÖNETİM - v40.3)
# ==========================================
# SYMBOLS, TIMEFRAMES, HTF_ONLY_MODE artık core.config'den import ediliyor
# Bu değişiklik tutarsızlıkları önler ve tek kaynak prensibini uygular.
# Değişiklik yapmak için: core/config.py dosyasını düzenleyin.
candles = 50000
REFRESH_RATE = 3

# Grafik güncelleme - False = daha hızlı başlatma, daha az CPU
# Grafikler: fast_start.py ile başlatıldığında Qt sırası sorunu oluşuyor
# Doğrudan "python desktop_bot_refactored_v2_base_v7.py" ile çalıştırırsan True yapabilirsin
ENABLE_CHARTS = False

# ==========================================
# 📁 DATA DIRECTORY SETUP
# ==========================================
# v40.2: All file paths now come from core.config (single source of truth)
# DATA_DIR, CSV_FILE, CONFIG_FILE, BEST_CONFIGS_FILE, BACKTEST_CANDLE_LIMITS,
# DAILY_REPORT_CANDLE_LIMITS, CANDLES_PER_DAY, days_to_candles are imported from core
# This prevents path mismatches between main file and core modules
os.makedirs(DATA_DIR, exist_ok=True)  # DATA_DIR imported from core

# v40.2: File paths (BEST_CONFIGS_FILE, BACKTEST_META_FILE, POT_LOG_FILE, DYNAMIC_BLACKLIST_FILE)
# are now imported from core.config - no local definitions to avoid mismatches

# Config cache - kept local for backward compatibility with existing code
BEST_CONFIG_CACHE = {}
BEST_CONFIG_WARNING_FLAGS = {
    "missing_signature": False,
    "signature_mismatch": False,
    "json_error": False,  # Bozuk JSON dosyası hatası için flag
    "load_error": False,  # Genel yükleme hatası için flag
}

# Çökme veya kapanma durumlarında otomatik yeniden başlatma gecikmesi (saniye)
AUTO_RESTART_DELAY_SECONDS = 5

# Note: TRADING_CONFIG is now imported from core.config for single source of truth
# (ensures signature consistency for backtest config validation)

# ==========================================
# 🎯 v40.3 - R-MULTIPLE BASED OPTIMIZER GATING
# ==========================================
# Tüm threshold'lar artık core.config'den import ediliyor:
# - MIN_EXPECTANCY_R_MULTIPLE
# - MIN_SCORE_THRESHOLD
# - CONFIDENCE_RISK_MULTIPLIER
# - WALK_FORWARD_CONFIG
# - MIN_OOS_TRADES_BY_TF
# - CIRCUIT_BREAKER_CONFIG
# - ROLLING_ER_CONFIG
#
# Değişiklik yapmak için: core/config.py dosyasını düzenleyin.
# ==========================================

# NOTE: MIN_EXPECTANCY_PER_TRADE is now imported from core.optimizer (v40.5)

# POST_PORTFOLIO_BLACKLIST artık core.config'den import ediliyor
# Lokal alias kullanılıyor (backward compatibility için)
POST_PORTFOLIO_BLACKLIST = CORE_POST_PORTFOLIO_BLACKLIST

# ==========================================
# 🔄 DYNAMIC BLACKLIST SYSTEM (v40.2 - REFACTORED)
# ==========================================
# All blacklist logic is now in core.config module (single source of truth).
# These are aliases for backward compatibility.
# ==========================================
DYNAMIC_BLACKLIST_CONFIG = CORE_DYNAMIC_BLACKLIST_CONFIG
DYNAMIC_BLACKLIST_CACHE = CORE_DYNAMIC_BLACKLIST_CACHE
DYNAMIC_BLACKLIST_FILE = CORE_DYNAMIC_BLACKLIST_FILE

# Use core implementations - eliminates shadowing risk
load_dynamic_blacklist = core_load_dynamic_blacklist
save_dynamic_blacklist = core_save_dynamic_blacklist
update_dynamic_blacklist = core_update_dynamic_blacklist
is_stream_blacklisted = core_is_stream_blacklisted


# NOTE: STRATEGY_BLACKLIST is now imported from core.optimizer (v40.5)

# Note: SYMBOL_PARAMS and DEFAULT_STRATEGY_CONFIG are now imported from core.config
# for single source of truth (signature consistency across save/load)

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

PARTIAL_STOP_PROTECTION_TFS = {"5m", "15m", "30m", "1h"}

# NOTE: _strategy_signature is now imported from core.config_loader

# ==========================================
# 🔗 CORE WRAPPER FONKSİYONLAR (Geriye Uyumluluk)
# ==========================================
# Bu wrapper'lar TradeManager'daki mevcut çağrılar için geriye uyumluluk sağlar.
# Tüm mantık core paketinde tek bir yerde tanımlanmıştır.

def _apply_1m_profit_lock(trade: dict, tf: str, t_type: str, entry: float, tp: float, progress: float) -> bool:
    """Wrapper: core.utils.apply_1m_profit_lock kullanır."""
    return apply_1m_profit_lock(trade, tf, t_type, entry, tp, progress)


def _apply_partial_stop_protection(trade: dict, tf: str, progress: float, t_type: str) -> bool:
    """Wrapper: core.utils.apply_partial_stop_protection kullanır."""
    return apply_partial_stop_protection(trade, tf, progress, t_type)


def _append_trade_event(trade: dict, event_type: str, event_time, price: Optional[float] = None):
    """Wrapper: core.utils.append_trade_event kullanır."""
    append_trade_event(trade, event_type, event_time, price)


def _audit_trade_logic_parity() -> dict:
    """Run a lightweight deterministic simulation on both managers to ensure parity.

    Includes config change simulation to verify that trades use snapshot values
    and are not affected by mid-trade config changes.
    """

    global BEST_CONFIG_CACHE

    try:
        symbol = "TESTCOIN"
        tf = "5m"
        seed_ts = utcnow()
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
        except (RuntimeError, KeyError):
            pass
        return {"parity_ok": False, "error": str(exc)}



# ==========================================
# 🔧 OPTIMIZER FUNCTIONS (v40.5 - MODULAR)
# ==========================================
# All optimizer functions are now in core/optimizer.py:
# - _generate_candidate_configs, _generate_quick_candidate_configs
# - _get_min_trades_for_timeframe, _split_data_walk_forward
# - _validate_config_oos, _check_overfit, _compute_optimizer_score
# - _score_config_for_stream, _optimize_backtest_configs
# Imported above from core module (SINGLE SOURCE OF TRUTH)
# ==========================================

# NOTE: _is_best_config_signature_valid, load_optimized_config, save_best_configs
# are now imported from core.config_loader (SINGLE SOURCE OF TRUTH v40.4)


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

        # --- STRATEJİ BAZLI CÜZDANLAR ---
        # Her strateji kendi wallet balance'ına sahip, böylece birbirlerini etkilemezler
        initial_bal = TRADING_CONFIG["initial_balance"]
        self.strategy_wallets = {
            "ssl_flow": {"wallet_balance": initial_bal, "locked_margin": 0.0, "total_pnl": 0.0},
            "keltner_bounce": {"wallet_balance": initial_bal, "locked_margin": 0.0, "total_pnl": 0.0},
        }
        # -----------------------------

        # --- CIRCUIT BREAKER TRACKING (v40.2, weekly added v40.4) ---
        # Stream-level: {(sym, tf): {"cumulative_pnl": 0, "peak_pnl": 0, "trades": 0, "r_multiples": []}}
        self._stream_pnl_tracker: Dict[Tuple[str, str], Dict] = {}
        # Killed streams: {(sym, tf): {"reason": str, "at_pnl": float, "at_trade": int}}
        self._circuit_breaker_killed: Dict[Tuple[str, str], Dict] = {}
        # Global tracking
        self._global_cumulative_pnl = 0.0
        self._global_peak_pnl = 0.0
        # Weekly tracking (v40.4)
        self._global_weekly_pnl = 0.0
        self._current_week_start: Optional[datetime] = None
        # Thread safety for circuit breaker (v44.x - Race Condition Fix)
        # Note: self.lock (RLock) already protects most operations, but this dedicated lock
        # provides explicit protection for circuit breaker check-and-update atomicity
        self._circuit_breaker_lock = threading.Lock()
        # -----------------------------

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

    def _get_strategy_wallet(self, strategy_mode: str) -> dict:
        """Strateji için cüzdan bilgilerini döndür"""
        if strategy_mode not in self.strategy_wallets:
            strategy_mode = "ssl_flow"  # Default (keltner_bounce is DISABLED)
        return self.strategy_wallets[strategy_mode]

    def _calculate_strategy_portfolio_risk(self, strategy_mode: str) -> float:
        """Belirli bir strateji için portföy risk yüzdesini hesapla"""
        wallet = self._get_strategy_wallet(strategy_mode)
        equity = wallet["wallet_balance"] + wallet["locked_margin"]
        if equity <= 0:
            return 0.0

        total_open_risk = 0.0
        for trade in self.open_trades:
            # Sadece bu stratejiye ait trade'leri say
            trade_strategy = trade.get("strategy_mode", "ssl_flow")
            if trade_strategy != strategy_mode:
                continue

            entry_price = float(trade.get("entry", 0.0))
            sl_price = float(trade.get("sl", entry_price))
            size = abs(float(trade.get("size", 0.0)))
            if entry_price <= 0 or size <= 0:
                continue
            sl_fraction = abs(entry_price - sl_price) / entry_price
            open_risk_amount = sl_fraction * size * entry_price
            total_open_risk += open_risk_amount

        return total_open_risk / equity

    def check_cooldown(self, symbol, timeframe, now_utc=None):
        """
        İşlem sonrası cooldown kontrolü.
        - now_utc pandas.Timestamp da olabilir, datetime da olabilir.
        - Hepsini offset-naive (tzinfo=None) datetime'a çevirip karşılaştırıyoruz.
        """
        k = (symbol, timeframe)

        if now_utc is None:
            now_utc = utcnow()

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

    def _calculate_equity(self, current_prices: dict = None) -> float:
        """Calculate total equity = wallet_balance + locked_margin + unrealized_pnl.

        This is the proper base for risk calculations (quant trader recommendation).
        Using wallet_balance alone causes risk % to inflate as margin is locked.
        """
        equity = self.wallet_balance + self.locked_margin

        # Add unrealized PnL if current prices available
        if current_prices and self.open_trades:
            for trade in self.open_trades:
                sym = trade.get("symbol")
                if sym not in current_prices:
                    continue
                current_price = current_prices[sym]
                entry = float(trade.get("entry", 0))
                size = float(trade.get("size", 0))
                trade_type = trade.get("type")
                if trade_type == "LONG":
                    unrealized = (current_price - entry) * size
                else:
                    unrealized = (entry - current_price) * size
                equity += unrealized

        return equity

    def _calculate_portfolio_risk_pct(self, wallet_balance: float) -> float:
        """Calculate portfolio risk as percentage of EQUITY (not wallet_balance).

        Fix: Using equity instead of wallet_balance prevents risk % from
        inflating as margin is locked (quant trader recommendation).
        """
        # Use equity for proper risk calculation
        equity = self._calculate_equity()
        if equity <= 0:
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

        return total_open_risk / equity

    # ==========================================
    # CIRCUIT BREAKER METHODS (v40.2, weekly added v40.4)
    # ==========================================

    def _get_week_start(self, dt: datetime) -> datetime:
        """Get the Monday 00:00 UTC of the week containing dt."""
        days_since_monday = dt.weekday()
        week_start = dt - timedelta(days=days_since_monday)
        return week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    def _check_and_reset_week(self, trade_time: datetime = None):
        """Check if we've entered a new week and reset weekly PnL if so."""
        if trade_time is None:
            trade_time = utcnow()

        # Normalize to naive datetime
        if hasattr(trade_time, 'tzinfo') and trade_time.tzinfo is not None:
            trade_time = trade_time.replace(tzinfo=None)

        current_week_start = self._get_week_start(trade_time)

        if self._current_week_start is None:
            self._current_week_start = current_week_start
        elif current_week_start > self._current_week_start:
            self._global_weekly_pnl = 0.0
            self._current_week_start = current_week_start

    def _update_circuit_breaker_tracking(self, sym: str, tf: str, pnl: float, r_multiple: float = None):
        """Update circuit breaker tracking after a trade closes."""
        # THREAD SAFETY: Protect circuit breaker state updates
        with self._circuit_breaker_lock:
            key = (sym, tf)

            # Initialize tracker if needed
            if key not in self._stream_pnl_tracker:
                self._stream_pnl_tracker[key] = {
                    "cumulative_pnl": 0.0,
                    "peak_pnl": 0.0,
                    "trades": 0,
                    "r_multiples": [],
                }

            tracker = self._stream_pnl_tracker[key]
            tracker["cumulative_pnl"] += pnl
            tracker["trades"] += 1
            tracker["peak_pnl"] = max(tracker["peak_pnl"], tracker["cumulative_pnl"])

            if r_multiple is not None:
                tracker["r_multiples"].append(r_multiple)

            # Update global tracking
            self._global_cumulative_pnl += pnl
            self._global_peak_pnl = max(self._global_peak_pnl, self._global_cumulative_pnl)

            # Update weekly tracking (v40.4)
            self._check_and_reset_week()
            self._global_weekly_pnl += pnl

    def check_stream_circuit_breaker(self, sym: str, tf: str) -> Tuple[bool, Optional[str]]:
        """Check if stream circuit breaker should trigger.

        Returns:
            (should_kill, reason) - reason is None if not triggered
        """
        if not CIRCUIT_BREAKER_CONFIG.get("enabled", True):
            return False, None

        key = (sym, tf)

        # Already killed?
        if key in self._circuit_breaker_killed:
            return True, self._circuit_breaker_killed[key].get("reason", "already_killed")

        # No tracking yet?
        if key not in self._stream_pnl_tracker:
            return False, None

        tracker = self._stream_pnl_tracker[key]

        # Minimum trades before circuit breaker activates
        min_trades = CIRCUIT_BREAKER_CONFIG.get("stream_min_trades_before_kill", 5)
        if tracker["trades"] < min_trades:
            return False, None

        # Check 1: Absolute loss limit
        max_loss = CIRCUIT_BREAKER_CONFIG.get("stream_max_loss", -200.0)
        if tracker["cumulative_pnl"] < max_loss:
            reason = f"max_loss_exceeded (PnL=${tracker['cumulative_pnl']:.2f} < ${max_loss})"
            self._kill_stream(key, reason, tracker)
            return True, reason

        # Check 2: Drawdown from peak (DOLLAR-BASED)
        max_dd_dollars = CIRCUIT_BREAKER_CONFIG.get("stream_max_drawdown_dollars", 100.0)
        if tracker["peak_pnl"] > 0:
            drawdown_dollars = tracker["peak_pnl"] - tracker["cumulative_pnl"]
            if drawdown_dollars > max_dd_dollars:
                reason = f"drawdown_exceeded (${drawdown_dollars:.2f} drop from peak ${tracker['peak_pnl']:.2f})"
                self._kill_stream(key, reason, tracker)
                return True, reason

        # Check 3: Rolling E[R] check
        if ROLLING_ER_CONFIG.get("enabled", True):
            r_multiples = tracker.get("r_multiples", [])
            min_trades_er = ROLLING_ER_CONFIG.get("min_trades_before_check", 10)

            if len(r_multiples) >= min_trades_er:
                window = ROLLING_ER_CONFIG.get("window_by_tf", {}).get(tf, 15)
                recent_r = r_multiples[-window:] if len(r_multiples) >= window else r_multiples

                if ROLLING_ER_CONFIG.get("use_confidence_band", True) and len(recent_r) >= 5:
                    import statistics
                    mean_r = statistics.mean(recent_r)
                    stdev_r = statistics.stdev(recent_r) if len(recent_r) > 1 else 0
                    factor = ROLLING_ER_CONFIG.get("confidence_band_factor", 0.5)
                    lower_bound = mean_r - (stdev_r * factor)

                    kill_thresh = ROLLING_ER_CONFIG.get("kill_threshold", -0.05)
                    if lower_bound < kill_thresh:
                        reason = f"rolling_er_negative (E[R]={mean_r:.3f}, lower_bound={lower_bound:.3f})"
                        self._kill_stream(key, reason, tracker)
                        return True, reason
                else:
                    # Simple threshold check
                    rolling_er = sum(recent_r) / len(recent_r)
                    kill_thresh = ROLLING_ER_CONFIG.get("kill_threshold", -0.05)
                    if rolling_er < kill_thresh:
                        reason = f"rolling_er_below_threshold (E[R]={rolling_er:.3f} < {kill_thresh})"
                        self._kill_stream(key, reason, tracker)
                        return True, reason

        return False, None

    def check_global_circuit_breaker(self) -> Tuple[bool, Optional[str]]:
        """Check if global circuit breaker should trigger.

        Returns:
            (should_kill, reason) - reason is None if not triggered
        """
        if not CIRCUIT_BREAKER_CONFIG.get("enabled", True):
            return False, None

        # Check daily loss limit (session-based for live)
        daily_max = CIRCUIT_BREAKER_CONFIG.get("global_daily_max_loss", -400.0)
        if self._global_cumulative_pnl < daily_max:
            return True, f"daily_loss_exceeded (${self._global_cumulative_pnl:.2f} < ${daily_max})"

        # Check weekly loss limit (v40.4)
        weekly_max = CIRCUIT_BREAKER_CONFIG.get("global_weekly_max_loss", -800.0)
        if self._global_weekly_pnl < weekly_max:
            return True, f"weekly_loss_exceeded (${self._global_weekly_pnl:.2f} < ${weekly_max})"

        # Check global drawdown (equity-based)
        initial_balance = TRADING_CONFIG.get("initial_balance", 2000.0)
        peak_equity = initial_balance + self._global_peak_pnl
        current_equity = initial_balance + self._global_cumulative_pnl
        max_dd_pct = CIRCUIT_BREAKER_CONFIG.get("global_max_drawdown_pct", 0.20)

        if peak_equity > initial_balance:  # Only check if we've had profits
            dd_pct = (peak_equity - current_equity) / peak_equity
            if dd_pct > max_dd_pct:
                return True, f"global_drawdown_exceeded ({dd_pct:.1%} > {max_dd_pct:.1%})"

        return False, None

    def _kill_stream(self, key: Tuple[str, str], reason: str, tracker: Dict):
        """Mark a stream as killed by circuit breaker."""
        self._circuit_breaker_killed[key] = {
            "reason": reason,
            "at_pnl": tracker.get("cumulative_pnl", 0),
            "at_trade": tracker.get("trades", 0),
        }
        if self.verbose:
            print(f"🛑 CIRCUIT BREAKER: {key[0]}-{key[1]} killed - {reason}")

    def is_stream_killed(self, sym: str, tf: str) -> bool:
        """Check if a stream has been killed by circuit breaker."""
        return (sym, tf) in self._circuit_breaker_killed

    def get_circuit_breaker_report(self) -> Dict:
        """Get circuit breaker status report."""
        return {
            "killed_streams": dict(self._circuit_breaker_killed),
            "stream_trackers": dict(self._stream_pnl_tracker),
            "global_pnl": self._global_cumulative_pnl,
            "global_peak": self._global_peak_pnl,
            "global_weekly_pnl": self._global_weekly_pnl,
            "current_week_start": self._current_week_start.isoformat() if self._current_week_start else None,
        }

    def open_trade(self, signal_data):
        with self.lock:
            tf = signal_data["timeframe"]
            sym = signal_data["symbol"]

            # ==========================================
            # ATOMIC CIRCUIT BREAKER CHECK (v44.x)
            # ==========================================
            # CRITICAL: Dedicated lock protects against race condition where two threads
            # could both pass is_stream_killed() before either marks the stream as active.
            # This ensures check-and-update is atomic in multi-threaded live trading.
            with self._circuit_breaker_lock:
                # Circuit breaker check - defense in depth
                # This guarantees no trade opens even if signal-side check was bypassed
                if self.is_stream_killed(sym, tf):
                    print(f"🛑 [{sym}-{tf}] Circuit breaker aktif - trade açılmadı")
                    return

                # Early validation check (inside circuit breaker lock for atomicity)
                cooldown_ref_time = signal_data.get("open_time_utc") or utcnow()
                if self.check_cooldown(sym, tf, cooldown_ref_time):
                    return

                # Mark stream as active by proceeding with trade opening
                # (The actual trade opening continues outside the circuit breaker lock)

            setup_type = signal_data.get("setup", "Unknown")

            # KRITIK: Config snapshot'ı önce signal_data'dan al (sinyal üretiminde kullanılan config)
            # Bu sayede sinyal üretimi ve trade yönetimi aynı config ile yapılır
            # Fallback: signal_data'da yoksa diskten yükle (eski trade'ler için)
            config_snapshot = signal_data.get("config_snapshot") or load_optimized_config(sym, tf)
            use_trailing = config_snapshot.get("use_trailing", False)
            use_partial = config_snapshot.get("use_partial", True)  # Partial TP default True
            use_dynamic_pbema_tp = config_snapshot.get("use_dynamic_pbema_tp", True)
            opt_rr = config_snapshot.get("rr", 3.0)
            opt_rsi = config_snapshot.get("rsi", 60)

            # Strateji modunu al - pozisyon büyüklüğü strateji cüzdanından hesaplanacak
            strategy_mode = config_snapshot.get("strategy_mode", "ssl_flow")
            strategy_wallet = self._get_strategy_wallet(strategy_mode)

            # Confidence-based risk multiplier: reduce position size for medium confidence
            # Backward compat: eski JSON'larda _confidence olabilir
            confidence_level = config_snapshot.get("confidence") or config_snapshot.get("_confidence", "high")
            risk_multiplier = CONFIDENCE_RISK_MULTIPLIER.get(confidence_level, 1.0)
            if risk_multiplier <= 0:
                print(f"⚠️ [{sym}-{tf}] Düşük güven seviyesi, işlem açılmadı.")
                return

            # Strateji cüzdanı kontrolü
            strategy_balance = strategy_wallet["wallet_balance"]
            if strategy_balance < 10:
                print(f"⚠️ [{strategy_mode}] Yetersiz Bakiye (${strategy_balance:.2f}). İşlem açılamadı.")
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
            # Strateji bazlı portföy riski kullan
            effective_risk_pct = self.risk_per_trade_pct * risk_multiplier
            current_portfolio_risk_pct = self._calculate_strategy_portfolio_risk(strategy_mode)
            if current_portfolio_risk_pct + effective_risk_pct > self.max_portfolio_risk_pct:
                print(
                    f"⚠️ [{strategy_mode}] Portföy risk limiti aşılıyor: mevcut %{current_portfolio_risk_pct * 100:.2f}, "
                    f"yeni işlem riski %{effective_risk_pct * 100:.2f}, limit %{self.max_portfolio_risk_pct * 100:.2f}"
                )
                return

            wallet_balance = strategy_balance  # Strateji cüzdanını kullan
            if wallet_balance <= 0:
                print(f"⚠️ [{strategy_mode}] Cüzdan bakiyesi 0 veya negatif, işlem açılamadı.")
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
                # CRITICAL FIX: Recalculate risk_amount after scale-down
                # Otherwise R-multiple and optimizer scoring will be wrong
                risk_amount = sl_fraction * position_notional
                print(
                    f"⚠️ Gerekli marjin bakiyeyi aşıyor, pozisyon {scale_factor:.2f} oranında düşürüldü."
                )

            # === TP/SL DIRECTION VALIDATION ===
            # Critical: Ensure TP and SL are on correct sides of entry
            # Without this check, trades can get stuck if TP is unreachable
            # This mirrors the validation in SimTradeManager.open_trade()
            tp_price = float(signal_data["tp"])
            if trade_type == "LONG":
                # LONG: TP must be above entry, SL must be below entry
                if tp_price <= real_entry:
                    print(f"⚠️ [{sym}-{tf}] LONG trade rejected: TP ({tp_price:.6f}) at or below entry ({real_entry:.6f})")
                    return
                if sl_price >= real_entry:
                    print(f"⚠️ [{sym}-{tf}] LONG trade rejected: SL ({sl_price:.6f}) at or above entry ({real_entry:.6f})")
                    return
            elif trade_type == "SHORT":
                # SHORT: TP must be below entry, SL must be above entry
                if tp_price >= real_entry:
                    print(f"⚠️ [{sym}-{tf}] SHORT trade rejected: TP ({tp_price:.6f}) at or above entry ({real_entry:.6f})")
                    return
                if sl_price <= real_entry:
                    print(f"⚠️ [{sym}-{tf}] SHORT trade rejected: SL ({sl_price:.6f}) at or below entry ({real_entry:.6f})")
                    return

            # open_time_utc'yi datetime string'e çevir (numpy.datetime64, pd.Timestamp veya datetime olabilir)
            _otv = signal_data.get("open_time_utc") or utcnow()
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
                "initial_tp": float(signal_data["tp"]),  # Progress hesabı için sabit referans (dinamik TP değişse bile)
                "size": position_size, "margin": required_margin,
                "notional": position_notional, "risk_amount": risk_amount,  # For R-multiple calculation
                "events": [],
                "status": "OPEN", "pnl": 0.0,
                "breakeven": False, "trailing_active": False, "partial_taken": False, "partial_price": None,
                "trailing_be_to_partial": False,
                "has_cash": True, "close_time": "", "close_price": "",
                # Trade açılırken snapshot edilen config ayarları (yaşam döngüsü boyunca sabit kalır)
                "use_trailing": use_trailing,
                "use_partial": use_partial,  # Partial TP config'den
                "use_dynamic_pbema_tp": use_dynamic_pbema_tp,
                "opt_rr": opt_rr,
                "opt_rsi": opt_rsi,
                # Strateji modu - PnL hesaplamasında doğru cüzdanı güncellemek için
                "strategy_mode": strategy_mode,
            }

            # Strateji cüzdanından margin düş
            strategy_wallet["wallet_balance"] -= required_margin
            strategy_wallet["locked_margin"] += required_margin
            # Geriye uyumluluk için global değişkenleri de güncelle
            self.wallet_balance -= required_margin
            self.locked_margin += required_margin

            self.open_trades.append(new_trade)
            new_portfolio_risk_pct = self._calculate_strategy_portfolio_risk(strategy_mode)

            strategy_short = "SF" if strategy_mode == "ssl_flow" else "KB"
            print(
                f"📈 [{strategy_short}] İşlem Açıldı | Entry: {real_entry:.4f}, SL: {sl_price:.4f}, "
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
                candle_time_utc = utcnow()

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
                # Fallback: margin yoksa doğru hesapla (size * entry / leverage, size / leverage DEĞİL)
                initial_margin = float(trade.get("margin", abs(size) * entry / TRADING_CONFIG["leverage"]))

                # Config'i trade dict'inden oku (açılışta snapshot edildi)
                # Eski trade'ler için fallback olarak load_optimized_config kullan ve trade'e yaz
                if "use_trailing" in trade:
                    use_trailing = trade.get("use_trailing", False)
                    use_partial = trade.get("use_partial", True)  # Trade'den oku
                    use_dynamic_tp = trade.get("use_dynamic_pbema_tp", True)
                else:
                    # Backward compatibility: eski trade'ler için config'den oku ve trade'e yaz
                    # Bu sayede sadece bir kere config'ten okunur, sonraki mumlarda trade'den okunur
                    config = load_optimized_config(symbol, tf)
                    use_trailing = config.get("use_trailing", False)
                    use_partial = config.get("use_partial", True)  # Config'den oku
                    use_dynamic_tp = config.get("use_dynamic_pbema_tp", True)
                    # Trade'e yaz - sonraki mumlarda trade'den okunacak
                    self.open_trades[i]["use_trailing"] = use_trailing
                    self.open_trades[i]["use_partial"] = use_partial  # Partial config'i de kaydet
                    self.open_trades[i]["use_dynamic_pbema_tp"] = use_dynamic_tp
                    self.open_trades[i]["opt_rr"] = config.get("rr", 3.0)
                    self.open_trades[i]["opt_rsi"] = config.get("rsi", 60)

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
                    except (ValueError, TypeError):
                        dyn_tp = tp
                    self.open_trades[i]["tp"] = dyn_tp

                # Ekranda gösterilecek anlık PnL (kapanışa göre)
                if t_type == "LONG":
                    live_pnl = (close_price - entry) * size
                else:
                    live_pnl = (entry - close_price) * size
                self.open_trades[i]["pnl"] = live_pnl

                # Hedefe ilerleme oranı (GERÇEK extreme'e göre, conservative değil)
                # KRITIK: Progress için initial_tp kullan, dinamik TP değil!
                # Dinamik TP değiştikçe progress zıplamasın, partial/breakeven erken tetiklenmesin
                initial_tp = float(trade.get("initial_tp", tp))  # Backward compat: yoksa tp kullan
                total_dist = abs(initial_tp - entry)
                if total_dist <= 0:
                    continue
                current_dist = abs(extreme_price - entry)
                progress = current_dist / total_dist if total_dist > 0 else 0.0

                # ---------- PARTIAL TP + BREAKEVEN ----------
                if in_profit and use_partial:
                    if (not self.open_trades[i].get("partial_taken")) and progress >= 0.40:
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

                        # Strateji cüzdanını güncelle
                        trade_strategy = trade.get("strategy_mode", "ssl_flow")
                        strat_wallet = self._get_strategy_wallet(trade_strategy)
                        strat_wallet["wallet_balance"] += margin_release + net_partial_pnl
                        strat_wallet["locked_margin"] -= margin_release
                        strat_wallet["total_pnl"] += net_partial_pnl
                        # Geriye uyumluluk için global değişkenleri de güncelle
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
                        # Breakeven'e çek - buffer ile (spread/slippage koruması)
                        be_buffer = 0.002  # %0.2 buffer
                        if t_type == "LONG":
                            be_sl = entry * (1 + be_buffer)
                        else:
                            be_sl = entry * (1 - be_buffer)
                        self.open_trades[i]["sl"] = be_sl
                        self.open_trades[i]["breakeven"] = True
                        _append_trade_event(self.open_trades[i], "BE_SET", candle_time_utc, be_sl)
                        trades_updated = True

                    elif (not trade.get("breakeven")) and progress >= 0.40:
                        # Breakeven'e çek - buffer ile
                        be_buffer = 0.002  # %0.2 buffer
                        if t_type == "LONG":
                            be_sl = entry * (1 + be_buffer)
                        else:
                            be_sl = entry * (1 - be_buffer)
                        self.open_trades[i]["sl"] = be_sl
                        self.open_trades[i]["breakeven"] = True
                        _append_trade_event(self.open_trades[i], "BE_SET", candle_time_utc, be_sl)
                        trades_updated = True

                # ---------- TRAILING BE TO PARTIAL ----------
                # %90 progress'e ulaşınca BE'yi partial fiyatına taşı (daha fazla kar koruma)
                if (self.open_trades[i].get("partial_taken") and
                    self.open_trades[i].get("breakeven") and
                    progress >= 0.90 and
                    not self.open_trades[i].get("trailing_be_to_partial")):

                    partial_price = self.open_trades[i].get("partial_price")
                    if partial_price is not None:
                        current_sl = float(self.open_trades[i]["sl"])
                        be_buffer = 0.002  # %0.2 buffer

                        if t_type == "LONG":
                            new_be = partial_price * (1 + be_buffer)
                            if new_be > current_sl:
                                self.open_trades[i]["sl"] = new_be
                                self.open_trades[i]["trailing_be_to_partial"] = True
                                _append_trade_event(self.open_trades[i], "BE_TO_PARTIAL", candle_time_utc, new_be)
                                trades_updated = True
                        else:
                            new_be = partial_price * (1 - be_buffer)
                            if new_be < current_sl:
                                self.open_trades[i]["sl"] = new_be
                                self.open_trades[i]["trailing_be_to_partial"] = True
                                _append_trade_event(self.open_trades[i], "BE_TO_PARTIAL", candle_time_utc, new_be)
                                trades_updated = True

                # 1m için fiyat TP'ye çok yaklaşınca SL'i kâra çek
                if _apply_1m_profit_lock(self.open_trades[i], tf, t_type, entry, dyn_tp, progress):
                    _append_trade_event(self.open_trades[i], "PROFIT_LOCK", candle_time_utc, self.open_trades[i].get("sl"))
                    trades_updated = True

                # ---------- TRAILING SL ----------
                if in_profit and use_trailing:
                    if (not trade.get("breakeven")) and progress >= 0.40:
                        # Breakeven'e çek - buffer ile
                        be_buffer = 0.002  # %0.2 buffer
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

                # Funding cost - merkezi fonksiyon kullan (v40.4)
                funding_cost = calculate_funding_cost(
                    open_time=trade.get("open_time_utc", ""),
                    close_time=candle_time_utc,
                    notional_value=abs(current_size) * entry
                )

                final_net_pnl = gross_pnl - commission - funding_cost

                # Strateji cüzdanını güncelle
                trade_strategy = trade.get("strategy_mode", "ssl_flow")
                strat_wallet = self._get_strategy_wallet(trade_strategy)
                strat_wallet["wallet_balance"] += margin_release + final_net_pnl
                strat_wallet["locked_margin"] -= margin_release
                strat_wallet["total_pnl"] += final_net_pnl
                # Geriye uyumluluk için global değişkenleri de güncelle
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

                # Update circuit breaker tracking
                self._update_circuit_breaker_tracking(
                    symbol, tf,
                    final_net_pnl,
                    serialized_trade.get("r_multiple")  # May be None in live
                )

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

    def check_realtime_sl(self, symbol: str, latest_price: float) -> list:
        """Real-time SL check - mum kapanmasını beklemeden SL'e ulaşan pozisyonları kapat.

        Bu fonksiyon kritik bir güvenlik mekanizmasıdır:
        - Fiyat her güncellendiğinde çağrılır (her ~0.5 saniye)
        - SL'e ulaşan pozisyonları anında kapatır
        - Mum kapanmasını beklemekten kaynaklanan aşırı kayıpları önler

        Returns:
            Kapatılan trade'lerin listesi (Telegram bildirimi için)
        """
        with self.lock:
            closed_indices = []
            just_closed_trades = []

            for i, trade in enumerate(self.open_trades):
                if trade.get("symbol") != symbol:
                    continue

                t_type = trade.get("type")
                entry = float(trade.get("entry", 0))
                sl = float(trade.get("sl", 0))
                tp = float(trade.get("tp", 0))

                if entry <= 0 or sl <= 0:
                    continue

                # SL kontrolü
                hit_sl = False
                if t_type == "LONG":
                    hit_sl = latest_price <= sl
                else:  # SHORT
                    hit_sl = latest_price >= sl

                if not hit_sl:
                    continue

                # --- POZİSYONU ACİL KAPAT ---
                tf = trade.get("timeframe", "?")
                size = float(trade.get("size", 0))
                initial_margin = float(trade.get("margin", abs(size) * entry / TRADING_CONFIG["leverage"]))
                margin_release = float(trade.get("margin", initial_margin))

                # Slippage uygula
                if t_type == "LONG":
                    exit_fill = latest_price * (1 - self.slippage_pct)
                    gross_pnl = (exit_fill - entry) * size
                else:
                    exit_fill = latest_price * (1 + self.slippage_pct)
                    gross_pnl = (entry - exit_fill) * size

                # Komisyon hesapla
                commission_notional = abs(size) * abs(exit_fill)
                commission = commission_notional * TRADING_CONFIG["total_fee"]

                # Funding cost - merkezi fonksiyon kullan (v40.4)
                funding_cost = calculate_funding_cost(
                    open_time=trade.get("open_time_utc", ""),
                    close_time=utcnow(),
                    notional_value=abs(size) * entry
                )

                final_net_pnl = gross_pnl - commission - funding_cost

                # Strateji cüzdanını güncelle
                trade_strategy = trade.get("strategy_mode", "ssl_flow")
                strat_wallet = self._get_strategy_wallet(trade_strategy)
                strat_wallet["wallet_balance"] += margin_release + final_net_pnl
                strat_wallet["locked_margin"] -= margin_release
                strat_wallet["total_pnl"] += final_net_pnl
                # Geriye uyumluluk için global değişkenleri de güncelle
                self.wallet_balance += margin_release + final_net_pnl
                self.locked_margin -= margin_release
                self.total_pnl += final_net_pnl

                # Cooldown ayarla
                wait_minutes = 10 if tf == "1m" else (30 if tf == "5m" else 60)
                self.cooldowns[(symbol, tf)] = utcnow() + pd.Timedelta(minutes=wait_minutes)

                # BE statüsünü ayır
                reason = "STOP [RT]"  # RT = Real-Time kapatma
                if trade.get("breakeven") and abs(final_net_pnl) < 1e-6:
                    reason = "BE [RT]"

                trade["status"] = reason
                trade["pnl"] = final_net_pnl
                trade["close_time"] = (utcnow() + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
                trade["close_price"] = float(exit_fill)

                serialized_trade = trade.copy()
                serialized_trade["events"] = json.dumps(trade.get("events", []))

                self.history.append(serialized_trade)
                just_closed_trades.append(serialized_trade)
                closed_indices.append(i)

                # Update circuit breaker tracking
                self._update_circuit_breaker_tracking(
                    symbol, tf,
                    final_net_pnl,
                    serialized_trade.get("r_multiple")  # May be None in live
                )

                print(f"🚨 [RT-SL] {symbol}-{tf} {t_type} ACİL KAPATILDI | Fiyat: {latest_price:.4f}, SL: {sl:.4f}, PnL: ${final_net_pnl:+.2f}")

            # Kapatılan trade'leri listeden çıkar
            for idx in sorted(closed_indices, reverse=True):
                del self.open_trades[idx]

            if closed_indices:
                self.save_trades()

            return just_closed_trades

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
                fd, tmp_path = tempfile.mkstemp(prefix="trades_temp_", suffix=".csv", dir=DATA_DIR)
                os.close(fd)
                df_all[cols].to_csv(tmp_path, index=False)
                shutil.move(tmp_path, CSV_FILE)

            except Exception as e:
                print(f"KAYIT HATASI: {e}")
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                with open(os.path.join(DATA_DIR, "error_log.txt"), "a") as f:
                    f.write(f"\n[{datetime.now()}] SAVE_TRADES HATA: {str(e)}\n")
                    f.write(traceback.format_exc())

    def load_trades(self):
        if not self.persist:
            return
        with self.lock:
            self.wallet_balance = TRADING_CONFIG["initial_balance"]
            self.locked_margin = 0.0
            self.total_pnl = 0.0

            # Strateji cüzdanlarını sıfırla
            initial_bal = TRADING_CONFIG["initial_balance"]
            self.strategy_wallets = {
                "keltner_bounce": {"wallet_balance": initial_bal, "locked_margin": 0.0, "total_pnl": 0.0},
                "ssl_flow": {"wallet_balance": initial_bal, "locked_margin": 0.0, "total_pnl": 0.0},
            }

            if os.path.exists(CSV_FILE):
                try:
                    if os.path.getsize(CSV_FILE) == 0:
                        return
                    df = pd.read_csv(CSV_FILE)
                    if "symbol" in df.columns:
                        self.open_trades = df[df["status"].astype(str).str.contains("OPEN")].to_dict('records')
                        self.history = df[~df["status"].astype(str).str.contains("OPEN")].to_dict('records')

                        for trade in self.history:
                            trade_pnl = float(trade['pnl'])
                            self.total_pnl += trade_pnl
                            # Strateji bazlı PnL hesapla
                            trade_strategy = trade.get("strategy_mode", "ssl_flow")
                            strat_wallet = self._get_strategy_wallet(trade_strategy)
                            strat_wallet["total_pnl"] += trade_pnl

                        open_pnl = 0.0
                        for trade in self.open_trades:
                            m = float(trade.get('margin', float(trade['size']) / TRADING_CONFIG["leverage"]))
                            if not trade.get('notional'):
                                try:
                                    trade['notional'] = float(trade.get('entry', 0)) * float(trade.get('size', 0))
                                except (ValueError, TypeError):
                                    trade['notional'] = 0.0
                            events_val = trade.get("events")
                            if isinstance(events_val, str):
                                try:
                                    trade["events"] = json.loads(events_val)
                                except (json.JSONDecodeError, TypeError):
                                    trade["events"] = []
                            self.locked_margin += m
                            open_pnl += float(trade.get('pnl', 0.0))
                            # Strateji bazlı locked margin
                            trade_strategy = trade.get("strategy_mode", "ssl_flow")
                            strat_wallet = self._get_strategy_wallet(trade_strategy)
                            strat_wallet["locked_margin"] += m

                        # Kullanılabilir bakiye = başlangıç + kapalı işlemlerden net PnL - kilitli marj
                        self.wallet_balance = TRADING_CONFIG["initial_balance"] + self.total_pnl - self.locked_margin
                        total_equity = self.wallet_balance + self.locked_margin + open_pnl

                        # Strateji bazlı bakiyeleri hesapla
                        for strategy_mode, strat_wallet in self.strategy_wallets.items():
                            strat_wallet["wallet_balance"] = (
                                TRADING_CONFIG["initial_balance"] +
                                strat_wallet["total_pnl"] -
                                strat_wallet["locked_margin"]
                            )

                        if self.verbose:
                            print(
                                "📂 Veriler Yüklendi. "
                                f"Toplam Varlık (Equity): ${total_equity:.2f} | "
                                f"Kullanılabilir Bakiye: ${self.wallet_balance:.2f} | "
                                f"Kilitli Marj: ${self.locked_margin:.2f}")
                            sf_wallet = self.strategy_wallets["ssl_flow"]
                            kb_wallet = self.strategy_wallets["keltner_bounce"]
                            print(
                                f"   [SF] Bakiye: ${sf_wallet['wallet_balance']:.2f} | PnL: ${sf_wallet['total_pnl']:+.2f}")
                            print(
                                f"   [KB] Bakiye: ${kb_wallet['wallet_balance']:.2f} | PnL: ${kb_wallet['total_pnl']:+.2f}")

                except Exception as e:
                    print(f"YÜKLEME HATASI: {e}")
                    with open(os.path.join(DATA_DIR, "error_log.txt"), "a") as f:
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

                    # Release margin - strateji cüzdanını güncelle
                    margin = float(trade.get("margin", size / TRADING_CONFIG["leverage"]))
                    trade_strategy = trade.get("strategy_mode", "ssl_flow")
                    strat_wallet = self._get_strategy_wallet(trade_strategy)
                    strat_wallet["wallet_balance"] += margin + net_pnl
                    strat_wallet["locked_margin"] -= margin
                    strat_wallet["total_pnl"] += net_pnl
                    # Geriye uyumluluk için global değişkenleri de güncelle
                    self.wallet_balance += margin + net_pnl
                    self.locked_margin -= margin
                    self.total_pnl += net_pnl

                    # Update trade record
                    trade["status"] = reason
                    trade["pnl"] = net_pnl
                    trade["close_price"] = exit_fill
                    trade["close_time"] = utcnow().strftime("%Y-%m-%d %H:%M")

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

# ==========================================
# 🔧 LAZY TRADE MANAGER INITIALIZATION (v40.2)
# ==========================================
# trade_manager is now lazily initialized to avoid side effects at module import time.
# This allows better testing and multiple import scenarios.
# ==========================================
_trade_manager = None


def get_trade_manager() -> TradeManager:
    """Get or create the global TradeManager instance (lazy initialization)."""
    global _trade_manager
    if _trade_manager is None:
        _trade_manager = TradeManager()
    return _trade_manager


# For backward compatibility: create a module-level property-like accessor
# Note: Direct access via `trade_manager` is deprecated, use `get_trade_manager()` instead
# We keep this for existing code that references trade_manager directly
class _TradeManagerProxy:
    """Proxy object that lazily creates TradeManager on first access."""
    def __getattr__(self, name):
        return getattr(get_trade_manager(), name)

    def __setattr__(self, name, value):
        setattr(get_trade_manager(), name, value)


trade_manager = _TradeManagerProxy()


# ==========================================
# 🔧 TRADING ENGINE - Imported from core.trading_engine
# ==========================================
# TradingEngine class is now in core/trading_engine.py
# It provides: data fetching, signal detection, indicator calculations
# Import: from core import TradingEngine
#
# GUI-specific functions (create_chart_data_json, debug_plot_backtest_trade)
# remain here as standalone functions due to plotting/GUI dependencies.
# ==========================================


def debug_plot_backtest_trade(symbol: str,
                              timeframe: str,
                              trade_id: int,
                              trades_csv: str = None,
                              window: int = 40):
    """
    Debug helper to visualize a specific backtest trade.
    - run_portfolio_backtest must have been run first.
    - <symbol>_<timeframe>_prices.csv and trades_csv must exist.
    """
    # Set default trades_csv path
    if trades_csv is None:
        trades_csv = os.path.join(DATA_DIR, "backtest_trades.csv")

    # 1) Price data
    prices_path = os.path.join(DATA_DIR, f"{symbol}_{timeframe}_prices.csv")
    if not os.path.exists(prices_path):
        raise FileNotFoundError(f"Price data not found: {prices_path}")

    df_prices = pd.read_csv(prices_path)
    if "timestamp" in df_prices.columns:
        df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"], utc=True)

    # 2) Trade data
    if not os.path.exists(trades_csv):
        raise FileNotFoundError(f"Trades CSV not found: {trades_csv}")

    df_trades = pd.read_csv(trades_csv)

    # 3) Plot
    plot_trade(df_prices, df_trades, trade_id=trade_id, window=window)
    get_plt().show()


def create_chart_data_json(df, interval, symbol="BTCUSDT", signal=None, active_trades=[], show_rr=True):
    """Create JSON data for chart visualization (GUI/web).

    This function creates Plotly-compatible chart data including:
    - Candlestick chart
    - PBEMA cloud
    - Keltner bands
    - Trade markers and TP/SL zones
    """
    try:
        plot_df = df.tail(300).copy()
        # Define candle time range for filtering trades
        candle_start = plot_df['timestamp'].iloc[0] if len(plot_df) > 0 else pd.Timestamp.min.tz_localize('UTC')
        candle_end = plot_df['timestamp'].iloc[-1] if len(plot_df) > 0 else pd.Timestamp.max.tz_localize('UTC')
        if len(plot_df) > 1:
            if interval.endswith('m'):
                interval_mins = int(interval[:-1])
            elif interval.endswith('h'):
                interval_mins = int(interval[:-1]) * 60
            else:
                interval_mins = 240
        else:
            interval_mins = 15

        # Timestamp parsing
        istanbul_time_series = plot_df['timestamp'] + pd.Timedelta(hours=3)
        timestamps_str = istanbul_time_series.dt.strftime('%Y-%m-%d %H:%M').tolist()

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

        # PBEMA cloud
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

        # Keltner bands and baseline
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
                except (ValueError, TypeError, AttributeError):
                    return 0.0

            # Deduplication
            dedup = {}
            for tr in active_trades + past_trades:
                key = tr.get('id') or tr.get('timestamp') or tr.get('time') or id(tr)
                dedup[key] = tr

            all_trades_sorted = sorted(dedup.values(), key=_trade_sort_key)
            all_trades_to_show = all_trades_sorted[-2:]

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

                start_ts_raw = trade.get('timestamp', trade.get('time', ''))
                try:
                    start_dt = dateutil.parser.parse(start_ts_raw)
                    start_dt = pd.to_datetime(start_dt, utc=True)
                    if pd.isna(start_dt) or start_dt < candle_start or start_dt > candle_end:
                        continue
                    future_dt = start_dt + (time_diff * 20)
                    start_ts_str = start_dt.strftime('%Y-%m-%d %H:%M')
                    future_ts_str = future_dt.strftime('%Y-%m-%d %H:%M')
                except (ValueError, TypeError, AttributeError):
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

        _, plotly_utils = get_plotly()
        return json.dumps({'traces': traces, 'shapes': shapes, 'symbol': symbol},
                          cls=plotly_utils.PlotlyJSONEncoder)
    except Exception as e:
        return "{}"


# NOTE: TradingEngine has been moved to core/trading_engine.py
# See: from core import TradingEngine


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
        # ⚡ FAST STARTUP: Paralel veri yükleme (77 sıralı çağrı yerine 5 paralel thread)
        print("[STARTUP] Veriler paralel olarak yükleniyor...")
        self._data = self._load_initial_data_parallel()

    def _build_stream_path(self):
        streams = [f"{s.lower()}@kline_{tf}" for s in self.symbols for tf in self.timeframes]
        stream_query = "/stream?streams=" + "/".join(streams)
        return stream_query

    def _load_initial_data_parallel(self):
        """Başlangıç verilerini paralel olarak yükle (5x daha hızlı)."""
        import concurrent.futures
        import time as _time

        start_time = _time.time()
        tasks = [(s, tf) for s in self.symbols for tf in self.timeframes]
        results = {}
        total = len(tasks)
        completed = 0

        def fetch_one(args):
            symbol, tf = args
            try:
                return (symbol, tf, TradingEngine.get_data(symbol, tf, limit=self.max_candles))
            except (ValueError, KeyError, ConnectionError, OSError):
                return (symbol, tf, pd.DataFrame())

        # 10 paralel thread kullan (API rate limit'e dikkat ederek)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_one, t): t for t in tasks}
            for future in concurrent.futures.as_completed(futures):
                try:
                    sym, tf, df = future.result()
                    results[(sym, tf)] = df
                    completed += 1
                    # Her 10 tamamlananda bir progress göster
                    if completed % 10 == 0 or completed == total:
                        print(f"[STARTUP] Veri yükleme: {completed}/{total} ({100*completed//total}%)")
                except Exception as e:
                    print(f"[STARTUP] Veri hatası: {e}")

        elapsed = _time.time() - start_time
        print(f"[STARTUP] Tüm veriler yüklendi! ({elapsed:.1f} saniye)")
        return results

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
        except (OSError, socket.error):
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
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
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

# ==========================================
# 🖥️ UI COMPONENTS (v40.3)
# ==========================================

# --- WORKERS ---
class LiveBotWorker(QThread):
    update_ui_signal = pyqtSignal(str, str, str, str)
    trade_signal = pyqtSignal(dict)
    price_signal = pyqtSignal(str, float)
    potential_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)  # Bot aktiflik durumu için sinyal

    def __init__(self, current_params, tg_token, tg_chat_id, show_rr):
        super().__init__()
        self.is_running = True
        self.tg_token = tg_token;
        self.tg_chat_id = tg_chat_id;
        self.show_rr = show_rr
        self.last_signals = {sym: {tf: None for tf in TIMEFRAMES} for sym in SYMBOLS}
        self.last_potential = {sym: {tf: None for tf in TIMEFRAMES} for sym in SYMBOLS}
        self.ws_stream = BinanceWebSocketKlineStream(SYMBOLS, TIMEFRAMES, max_candles=1200)
        # Startup protection: İlk döngüde eski sinyalleri işleme, sadece timestamp'leri kaydet
        self._startup_warmup_done = False

    def update_settings(self, symbol, tf, rr, rsi, slope):
        if symbol in SYMBOL_PARAMS:
            current = SYMBOL_PARAMS[symbol].get(tf, {})
            at = current.get("at_active", True)
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
        next_status_time = 0  # Status güncellemesi için timer
        try:
            while self.is_running:
                now = time.time()

                stream_snapshot = None

                # Periyodik status güncellemesi (her 2 saniyede bir)
                if now >= next_status_time:
                    open_count = len(trade_manager.open_trades)
                    if open_count > 0:
                        self.status_signal.emit(f"AÇIK POZİSYON: {open_count}")
                    else:
                        self.status_signal.emit("TRADE ARIYOR...")
                    next_status_time = now + 2.0

                if now >= next_price_time:
                    try:
                        # WebSocket stream'i candle verileri için al (later use)
                        stream_snapshot = self.ws_stream.get_latest_bulk()

                        # KRITIK FIX (v40.4): Canlı fiyat için REST API kullan
                        # WebSocket kline close değeri güvenilir bir anlık fiyat kaynağı DEĞİL:
                        # - Farklı timeframe'ler farklı zamanlarda güncellenir
                        # - 1m timeframe TIMEFRAMES listesinde yok
                        # - Forming candle close değeri tick bazlı değil, kline update bazlı
                        # REST API /ticker/price endpoint'i gerçek anlık fiyatı döndürür
                        latest_prices = TradingEngine.get_latest_prices(SYMBOLS)
                        for sym, price in latest_prices.items():
                            self.price_signal.emit(sym, price)
                            trade_manager.update_live_pnl_with_price(sym, price)
                            # Real-time SL kontrolü - mum kapanmasını beklemeden SL'e ulaşan pozisyonları kapat
                            rt_closed = trade_manager.check_realtime_sl(sym, price)
                            for ct in rt_closed:
                                tf = ct.get('timeframe', '?')
                                pnl = float(ct.get('pnl', 0))
                                pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
                                close_log = f"🚨 {ct['symbol']} ACİL KAPATILDI ({tf}): {ct['status']} | {pnl_str}"
                                self.update_ui_signal.emit(sym, tf, "{}", f"⚠️ {close_log}")
                                # Telegram bildirimi
                                tg_msg = f"🚨 ACİL KAPATMA: {ct['symbol']}\nTF: {tf}\nSonuç: {ct['status']}\nNet PnL: {pnl_str}\n⚠️ Real-time SL tetiklendi"
                                TradingEngine.send_telegram(self.tg_token, self.tg_chat_id, tg_msg)
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

                                # Skip if insufficient data for indicators (need 200+ candles for EMA200)
                                if len(df) < 250:
                                    continue

                                # --- İndikatör Hesabı (ÖNCE yapılmalı - PBEMA değerleri için gerekli) ---
                                df_ind = TradingEngine.calculate_indicators(df.copy())

                                # Binance kline: son satır çoğunlukla oluşan (henüz kapanmamış) mumdur.
                                closed = df_ind.iloc[-2]
                                forming = df_ind.iloc[-1]
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

                                # Config'i erken yükle - strategy_mode için gerekli
                                config = load_optimized_config(sym, tf)
                                strategy_mode = config.get("strategy_mode", "ssl_flow")

                                # --- Trade Manager Güncellemesi ---
                                # SSL Flow ve Keltner Bounce icin EMA200 kullan
                                if strategy_mode == "ssl_flow_DISABLED":  # EMA150 artik kullanilmiyor
                                    pb_top_col = 'pb_ema_top_150'
                                    pb_bot_col = 'pb_ema_bot_150'
                                else:
                                    pb_top_col = 'pb_ema_top'
                                    pb_bot_col = 'pb_ema_bot'

                                closed_trades = trade_manager.update_trades(
                                    sym, tf,
                                    candle_high=float(closed['high']),
                                    candle_low=float(closed['low']),
                                    candle_close=float(closed['close']),
                                    candle_time_utc=closed_ts_utc,
                                    pb_top=float(closed.get(pb_top_col, closed['close'])),
                                    pb_bot=float(closed.get(pb_bot_col, closed['close']))
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

                                # --- Sinyal Hesabı ---
                                df_closed = df_ind.iloc[:-1].copy()  # oluşan mumu çıkar

                                # Skip disabled symbol/timeframe combinations
                                if config.get("disabled", False):
                                    continue

                                # 🔄 Dynamic blacklist kontrolü - backtest'te negatif PnL veren streamler atlanır
                                if is_stream_blacklisted(sym, tf):
                                    continue

                                rr, rsi, slope = config['rr'], config['rsi'], config['slope']
                                use_at = config['at_active']
                                at_status_log = "AT:ON" if use_at else "AT:OFF"
                                # strategy_mode already loaded earlier for PBEMA selection
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
                                        json_data = create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                         active_trades if self.show_rr else [])
                                        self.update_ui_signal.emit(sym, tf, json_data, log_msg)

                                    elif self.last_signals[sym][tf] != closed_ts_utc:
                                        # STARTUP PROTECTION: İlk döngüde eski sinyalleri işleme
                                        # last_signals None ise bu ilk döngü demek - sadece timestamp kaydet
                                        if self.last_signals[sym][tf] is None:
                                            self.last_signals[sym][tf] = closed_ts_utc
                                            self.last_potential[sym][tf] = closed_ts_utc  # Prevent potential_trades logging
                                            decision = "Rejected"
                                            reject_reason = "Startup Sync"
                                            log_msg = f"{tf} | {curr_price} | 🔄 Başlangıç senkronizasyonu"
                                            json_data = create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                             active_trades if self.show_rr else [])
                                            self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                        elif trade_manager.check_cooldown(sym, tf, forming_ts_utc):
                                            decision = "Rejected"
                                            reject_reason = "Cooldown"
                                            log_msg = f"{tf} | {curr_price} | ❄️ SOĞUMA SÜRECİNDE"
                                            json_data = create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                             active_trades if self.show_rr else [])
                                            self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                        else:
                                            # YENİ İŞLEM AÇ
                                            # KRITIK: Sinyal üretirken kullanılan config'i trade'e göm
                                            # Expert Panel (Sinclair): vol_position_multiplier for regime-adaptive sizing
                                            vol_pos_mult = s_debug.get("vol_position_multiplier", 1.0) if s_debug else 1.0
                                            trade_data = {
                                                "symbol": sym, "timestamp": next_open_ts_str, "open_time_utc": forming_ts_utc,
                                                "timeframe": tf, "type": s_type,
                                                "entry": next_open_price, "tp": s_tp, "sl": s_sl, "setup": setup_tag,
                                                "config_snapshot": config,  # Sinyal üretiminde kullanılan config
                                                "vol_position_multiplier": vol_pos_mult,  # Expert Panel: Regime-adaptive sizing
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
                                            json_data = create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                             active_trades if self.show_rr else [])
                                            self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                    else:
                                        decision = "Rejected"
                                        reject_reason = "Duplicate Signal"
                                        log_msg = f"{tf} | {curr_price} | ⏳ İşlemde...{live_pnl_str}"
                                        json_data = create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                         active_trades if self.show_rr else [])
                                        self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                else:
                                    # Sinyal Yoksa Logla
                                    decision = f"Rejected: {s_reason}" if s_reason else "Rejected"
                                    if s_reason and "REJECT" in s_reason:
                                        log_msg = f"{tf} | {curr_price} | ⚠️ {s_reason}{live_pnl_str}"
                                    else:
                                        log_msg = f"{tf} | {curr_price} | {at_status_log}{live_pnl_str}"

                                    json_data = create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                     active_trades if self.show_rr else [])
                                    self.update_ui_signal.emit(sym, tf, json_data, log_msg)

                                # Startup Sync rejections should not be logged to potential_trades
                                # (they are just old signals from before bot startup, not real opportunities)
                                if s_type and "ACCEPTED" in s_reason and self.last_potential[sym][tf] != closed_ts_utc and reject_reason != "Startup Sync":
                                    direction = s_type or ("LONG" if s_debug.get("holding_long") else ("SHORT" if s_debug.get("holding_short") else ""))

                                    checks = dict(s_debug or {})
                                    if "rsi_value" not in checks:
                                        try:
                                            checks["rsi_value"] = float(df_closed["rsi"].iloc[-1])
                                        except (IndexError, KeyError, ValueError, TypeError):
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
                                with open(os.path.join(DATA_DIR, "error_log.txt"), "a") as f:
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

    def __init__(self, symbol, candle_limit_or_days, rr_range, rsi_range, slope_range, use_alphatrend,
                 monte_carlo_mode=False, timeframes=None, use_days=False):
        super().__init__()
        self.symbol = symbol
        self.use_days = use_days
        self.timeframes = timeframes or list(TIMEFRAMES)
        if use_days:
            # Gün modunda: her TF için ayrı mum sayısı
            self.days = candle_limit_or_days
            self.candle_limit = None
            self.candle_limit_map = days_to_candles_map(candle_limit_or_days, self.timeframes)
        else:
            # Eski mum modu (geriye uyumluluk)
            self.days = None
            self.candle_limit = candle_limit_or_days
            self.candle_limit_map = None
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
                        df_trend['ema_trend'] = get_ta().ema(df_trend['close'], length=200)
                        df_trend = df_trend[['timestamp', 'ema_trend']].dropna()
                except (ValueError, KeyError, ConnectionError, OSError):
                    pass

            data_cache = {}
            for tf in self.timeframes:
                # Gün modunda her TF için farklı mum sayısı
                if self.use_days and self.candle_limit_map:
                    tf_candle_limit = self.candle_limit_map.get(tf, 720)  # varsayılan 30 gün * 24
                    self.result_signal.emit(f"⬇️ {tf} verisi hazırlanıyor ({self.days} gün = {tf_candle_limit} mum)...\n")
                else:
                    tf_candle_limit = self.candle_limit
                    self.result_signal.emit(f"⬇️ {tf} verisi hazırlanıyor...\n")
                df = TradingEngine.get_historical_data_pagination(self.symbol, tf, total_candles=tf_candle_limit)

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
            # AlphaTrend is now MANDATORY for SSL_Flow - only True option
            at_vals = [True]

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
                                except (TypeError, AttributeError):
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
                out_trades_csv=os.path.join(DATA_DIR, "daily_report_trades.csv"),
                out_summary_csv=os.path.join(DATA_DIR, "daily_report_summary.csv"),
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

    def __init__(self, symbols, timeframes, candles_or_days, skip_optimization=False, quick_mode=False, use_days=False, start_date=None, end_date=None):
        super().__init__()
        self.symbols = symbols
        self.timeframes = timeframes
        self.use_days = use_days
        self.start_date = start_date  # Sabit tarih modu için
        self.end_date = end_date
        if start_date is not None:
            # Sabit tarih aralığı modu - tutarlı sonuçlar için
            self.days = None
            self.candles = None
            self.limit_map = None
        elif use_days:
            # Gün modunda: her TF için ayrı mum sayısı hesapla
            self.days = candles_or_days
            self.candles = None  # limit_map kullanılacak
            self.limit_map = days_to_candles_map(candles_or_days, timeframes)
        else:
            # Eski mum modu (geriye uyumluluk)
            self.days = None
            self.candles = candles_or_days
            self.limit_map = None
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
            # Sabit tarih aralığı modu (tutarlı sonuçlar için)
            if self.start_date is not None:
                result = run_portfolio_backtest(
                    symbols=self.symbols,
                    timeframes=self.timeframes,
                    candles=50000,  # Tarih modunda kullanılmaz, fallback
                    progress_callback=self._throttled_log,
                    draw_trades=True,
                    max_draw_trades=30,
                    skip_optimization=self.skip_optimization,
                    quick_mode=self.quick_mode,
                    start_date=self.start_date,
                    end_date=self.end_date,
                ) or {}
            # Gün modunda limit_map kullan
            elif self.use_days and self.limit_map:
                result = run_portfolio_backtest(
                    symbols=self.symbols,
                    timeframes=self.timeframes,
                    candles=max(self.limit_map.values()),  # En yüksek değeri fallback olarak kullan
                    limit_map=self.limit_map,
                    progress_callback=self._throttled_log,
                    draw_trades=True,
                    max_draw_trades=30,
                    skip_optimization=self.skip_optimization,
                    quick_mode=self.quick_mode,
                ) or {}
            else:
                # Eski candles modu
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

        # Bot aktiflik göstergesi
        self.status_label = QLabel("TRADE ARIYOR...")
        self.status_label.setStyleSheet("""
            color: #00ff00;
            font-weight: bold;
            font-size: 14px;
            padding: 5px 15px;
            background-color: #004400;
            border-radius: 10px;
            border: 1px solid #00ff00;
        """)
        ticker_layout.addWidget(self.status_label)

        # Config yaşı göstergesi
        self.config_age_label = QLabel("")
        self.config_age_label.setStyleSheet("""
            color: #888;
            font-weight: bold;
            font-size: 12px;
            padding: 5px 10px;
            background-color: #1a1a1a;
            border-radius: 8px;
        """)
        ticker_layout.addWidget(self.config_age_label)
        self._update_config_age_label()  # Başlangıçta config yaşını kontrol et

        # Blacklist göstergesi
        self.blacklist_label = QLabel("")
        self.blacklist_label.setStyleSheet("""
            color: #888;
            font-weight: bold;
            font-size: 12px;
            padding: 5px 10px;
            background-color: #1a1a1a;
            border-radius: 8px;
        """)
        ticker_layout.addWidget(self.blacklist_label)
        self._update_blacklist_label()  # Başlangıçta blacklist'i kontrol et

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
        self.txt_token.setEchoMode(QLineEdit.Password)  # 🔒 Güvenlik: API token'ı yıldızlarla göster
        self.txt_token.setPlaceholderText("Bot Token")
        tg_layout.addWidget(QLabel("T:"));
        tg_layout.addWidget(self.txt_token)
        self.txt_chatid = QLineEdit(self.tg_chat_id);
        self.txt_chatid.setEchoMode(QLineEdit.Password)  # 🔒 Güvenlik: Chat ID'yi yıldızlarla göster
        self.txt_chatid.setPlaceholderText("Chat ID")
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
        # Grafik güncelleme toggle
        self.chk_charts = QCheckBox("📊 Grafik");
        self.chk_charts.setChecked(True);
        self.chk_charts.setToolTip("Grafik güncellemelerini açar/kapatır. Kapatmak CPU kullanımını azaltır.")
        sets_layout.addWidget(self.chk_charts);
        settings_group.setLayout(sets_layout);
        top_panel.addWidget(settings_group, stretch=4)
        live_layout.addLayout(top_panel)

        # Grafik güncelleme throttling için son güncelleme zamanları
        self.last_chart_update = {tf: 0 for tf in TIMEFRAMES}
        self.chart_update_interval = 3.0  # Saniye cinsinden minimum güncelleme aralığı

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
                    QWebEngineView = get_QWebEngineView()
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

        # Tarih/Gün seçimi - İKİ SEÇENEKLİ
        date_group = QGroupBox("Veri Aralığı")
        date_layout = QVBoxLayout()

        # Seçenek 1: Gün sayısı (mevcut davranış)
        days_row = QHBoxLayout()
        self.radio_days = QRadioButton("Son X Gün:")
        self.radio_days.setChecked(True)
        self.radio_days.setToolTip("Şu andan geriye doğru belirtilen gün sayısı kadar veri çeker")
        days_row.addWidget(self.radio_days)
        self.backtest_days = QSpinBox()
        self.backtest_days.setRange(7, 365)
        self.backtest_days.setValue(30)
        self.backtest_days.setToolTip("Her timeframe için bu kadar günlük veri test edilecek")
        days_row.addWidget(self.backtest_days)
        days_row.addStretch()
        date_layout.addLayout(days_row)

        # Seçenek 2: Sabit tarih aralığı (tutarlı sonuçlar için)
        from PyQt5.QtCore import QDate

        fixed_row = QHBoxLayout()
        self.radio_fixed_dates = QRadioButton("Sabit Tarih:")
        self.radio_fixed_dates.setToolTip("Tutarlı/karşılaştırılabilir sonuçlar için sabit tarih aralığı")
        fixed_row.addWidget(self.radio_fixed_dates)

        fixed_row.addWidget(QLabel("Başlangıç:"))
        self.backtest_start_date = QDateEdit()
        self.backtest_start_date.setCalendarPopup(True)
        self.backtest_start_date.setDate(QDate.currentDate().addDays(-30))
        self.backtest_start_date.setDisplayFormat("yyyy-MM-dd")
        self.backtest_start_date.setEnabled(False)
        fixed_row.addWidget(self.backtest_start_date)

        fixed_row.addWidget(QLabel("Bitiş:"))
        self.backtest_end_date = QDateEdit()
        self.backtest_end_date.setCalendarPopup(True)
        self.backtest_end_date.setDate(QDate.currentDate())
        self.backtest_end_date.setDisplayFormat("yyyy-MM-dd")
        self.backtest_end_date.setEnabled(False)
        fixed_row.addWidget(self.backtest_end_date)
        fixed_row.addStretch()
        date_layout.addLayout(fixed_row)

        # Radio button toggle - tarih alanlarını aktif/pasif yap
        def toggle_date_inputs():
            use_fixed = self.radio_fixed_dates.isChecked()
            self.backtest_days.setEnabled(not use_fixed)
            self.backtest_start_date.setEnabled(use_fixed)
            self.backtest_end_date.setEnabled(use_fixed)

        self.radio_days.toggled.connect(toggle_date_inputs)
        self.radio_fixed_dates.toggled.connect(toggle_date_inputs)

        date_group.setLayout(date_layout)
        bt_cfg.addWidget(date_group)

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
        candles_layout.addWidget(QLabel("Gün Sayısı:"));
        self.opt_days = QSpinBox();
        self.opt_days.setRange(7, 180);  # 7 gün - 6 ay arası
        self.opt_days.setValue(30);  # Varsayılan 30 gün
        self.opt_days.setToolTip("Her timeframe için bu kadar günlük veri optimize edilecek");
        candles_layout.addWidget(self.opt_days)
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

        # ========== ⚙️ CONFIG TAB ==========
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setSpacing(10)

        # Üst bilgi satırı
        config_header = QHBoxLayout()

        # Config durumu grubu
        status_group = QGroupBox("📋 Config Durumu")
        status_group.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        status_layout = QVBoxLayout(status_group)

        self.config_status_text = QTextEdit()
        self.config_status_text.setReadOnly(True)
        self.config_status_text.setMaximumHeight(150)
        self.config_status_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
                font-family: Consolas, monospace;
                font-size: 12px;
                border: 1px solid #333;
            }
        """)
        status_layout.addWidget(self.config_status_text)

        self.btn_refresh_config = QPushButton("🔄 Yenile")
        self.btn_refresh_config.clicked.connect(self._refresh_config_tab)
        status_layout.addWidget(self.btn_refresh_config)

        config_header.addWidget(status_group)

        # İstatistik grubu
        stats_group = QGroupBox("📊 İstatistikler")
        stats_group.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        stats_layout = QVBoxLayout(stats_group)

        self.config_stats_text = QTextEdit()
        self.config_stats_text.setReadOnly(True)
        self.config_stats_text.setMaximumHeight(150)
        self.config_stats_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #aaaaff;
                font-family: Consolas, monospace;
                font-size: 12px;
                border: 1px solid #333;
            }
        """)
        stats_layout.addWidget(self.config_stats_text)

        self.btn_delete_config = QPushButton("🗑️ Config Dosyasını Sil")
        self.btn_delete_config.setStyleSheet("QPushButton { background-color: #662222; } QPushButton:hover { background-color: #883333; }")
        self.btn_delete_config.clicked.connect(self._delete_config_file)
        stats_layout.addWidget(self.btn_delete_config)

        config_header.addWidget(stats_group)
        config_layout.addLayout(config_header)

        # Aktif streamler tablosu
        active_group = QGroupBox("✅ Aktif Streamler (Backtest'ten geçenler)")
        active_group.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        active_layout = QVBoxLayout(active_group)

        self.active_streams_table = QTableWidget()
        self.active_streams_table.setColumnCount(7)
        self.active_streams_table.setHorizontalHeaderLabels(["Symbol", "TF", "RR", "RSI", "AT", "Trailing", "Strateji"])
        self.active_streams_table.horizontalHeader().setStretchLastSection(True)
        self.active_streams_table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                color: white;
                gridline-color: #333;
            }
            QHeaderView::section {
                background-color: #114411;
                color: white;
                font-weight: bold;
                padding: 5px;
            }
        """)
        active_layout.addWidget(self.active_streams_table)
        config_layout.addWidget(active_group)

        # Devre dışı streamler tablosu
        disabled_group = QGroupBox("🚫 Devre Dışı Streamler (Blacklist + Disabled)")
        disabled_group.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        disabled_layout = QVBoxLayout(disabled_group)

        self.disabled_streams_table = QTableWidget()
        self.disabled_streams_table.setColumnCount(4)
        self.disabled_streams_table.setHorizontalHeaderLabels(["Symbol", "TF", "Sebep", "PnL"])
        self.disabled_streams_table.horizontalHeader().setStretchLastSection(True)
        self.disabled_streams_table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                color: white;
                gridline-color: #333;
            }
            QHeaderView::section {
                background-color: #441111;
                color: white;
                font-weight: bold;
                padding: 5px;
            }
        """)
        disabled_layout.addWidget(self.disabled_streams_table)
        config_layout.addWidget(disabled_group)

        self.main_tabs.addTab(config_widget, "⚙️ Config")

        # Açılışta canlı takip sekmesini öne çıkar
        self.main_tabs.setCurrentWidget(live_widget)

        # BAŞLATMA
        self.current_params = {}
        self.live_worker = LiveBotWorker(self.current_params, self.tg_token, self.tg_chat_id, self.show_rr_tools)
        self.live_worker.update_ui_signal.connect(self.update_ui)
        self.live_worker.price_signal.connect(self.on_price_update)
        self.live_worker.potential_signal.connect(self.append_potential_trade)
        self.live_worker.status_signal.connect(self.update_bot_status)
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

        # Config sekmesini başlangıçta yükle
        self._refresh_config_tab()
        self._update_config_age_label()
        self._update_blacklist_label()

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

    def update_bot_status(self, status_text):
        """Bot aktiflik durumunu güncelle"""
        self.status_label.setText(status_text)
        if "TRADE ARIYOR" in status_text:
            self.status_label.setStyleSheet("""
                color: #00ff00;
                font-weight: bold;
                font-size: 14px;
                padding: 5px 15px;
                background-color: #004400;
                border-radius: 10px;
                border: 1px solid #00ff00;
            """)
        elif "AÇIK POZİSYON" in status_text:
            self.status_label.setStyleSheet("""
                color: #ffcc00;
                font-weight: bold;
                font-size: 14px;
                padding: 5px 15px;
                background-color: #443300;
                border-radius: 10px;
                border: 1px solid #ffcc00;
            """)

    def render_chart_and_log(self, tf, json_data, log_msg):
        # Grafik güncelleme sadece ENABLE_CHARTS=True ve UI toggle açıksa
        import time as _time
        charts_enabled = getattr(self, 'chk_charts', None) and self.chk_charts.isChecked()
        if ENABLE_CHARTS and charts_enabled and self.views_ready.get(tf, False) and json_data and json_data != "{}":
            # Throttling: Aynı timeframe için minimum güncelleme aralığı
            now = _time.time()
            last_update = self.last_chart_update.get(tf, 0)
            if now - last_update >= self.chart_update_interval:
                safe_json = json_data.replace("'", "\\'").replace("\\", "\\\\")
                js = f"if(window.updateChartData) window.updateChartData('{safe_json}');"
                self.web_views[tf].page().runJavaScript(js)
                self.last_chart_update[tf] = now
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
            with open(os.path.join(DATA_DIR, "error_log.txt"), "a") as f:
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
                with open(os.path.join(DATA_DIR, "error_log.txt"), "a") as f:
                    f.write(f"\n[{datetime.now()}] HATA: {str(e)}\n")
                    f.write(traceback.format_exc())  # Hatanın hangi satırda olduğunu yazar

    def _update_config_age_label(self):
        """Backtest config yaşını kontrol et ve UI'da göster.

        Config 7 günden eskiyse sarı uyarı, 14 günden eskiyse kırmızı uyarı göster.
        """
        try:
            if not os.path.exists(BEST_CONFIGS_FILE):
                self.config_age_label.setText("⚠️ Config yok")
                self.config_age_label.setStyleSheet("""
                    color: #ff6600;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 5px 10px;
                    background-color: #332200;
                    border-radius: 8px;
                    border: 1px solid #ff6600;
                """)
                return

            try:
                with open(BEST_CONFIGS_FILE, 'r', encoding='utf-8') as f:
                    best_cfgs = json.load(f)
            except json.JSONDecodeError:
                self.config_age_label.setText("⚠️ Config bozuk")
                self.config_age_label.setStyleSheet("""
                    color: #ff6600;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 5px 10px;
                    background-color: #332200;
                    border-radius: 8px;
                    border: 1px solid #ff6600;
                """)
                self.config_age_label.setToolTip("best_configs.json dosyası bozuk. Backtest çalıştırın.")
                return

            if not isinstance(best_cfgs, dict):
                return

            meta = best_cfgs.get("_meta", {})
            saved_at_str = meta.get("saved_at", "")

            if not saved_at_str:
                self.config_age_label.setText("⚠️ Config tarihi yok")
                self.config_age_label.setStyleSheet("""
                    color: #ff6600;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 5px 10px;
                    background-color: #332200;
                    border-radius: 8px;
                    border: 1px solid #ff6600;
                """)
                return

            # ISO format parse (2024-01-15T12:30:45Z)
            saved_at = datetime.fromisoformat(saved_at_str.replace("Z", "+00:00"))
            # Her iki datetime'ı da naive UTC olarak karşılaştır
            saved_at_naive = saved_at.replace(tzinfo=None)
            now_naive = utcnow()
            age = now_naive - saved_at_naive
            age_days = age.days
            age_hours = age.seconds // 3600

            # Yaş metni
            if age_days == 0:
                age_text = f"{age_hours}s önce"
            elif age_days == 1:
                age_text = "1 gün önce"
            else:
                age_text = f"{age_days} gün önce"

            # Renk ve stil
            if age_days >= 14:
                # Kırmızı - acil backtest gerekli
                self.config_age_label.setText(f"🔴 Config: {age_text}")
                self.config_age_label.setStyleSheet("""
                    color: #ff4444;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 5px 10px;
                    background-color: #441111;
                    border-radius: 8px;
                    border: 1px solid #ff4444;
                """)
                self.config_age_label.setToolTip("Config 14+ gün eski! Backtest çalıştırmanız önerilir.")
            elif age_days >= 7:
                # Sarı - uyarı
                self.config_age_label.setText(f"🟡 Config: {age_text}")
                self.config_age_label.setStyleSheet("""
                    color: #ffaa00;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 5px 10px;
                    background-color: #332200;
                    border-radius: 8px;
                    border: 1px solid #ffaa00;
                """)
                self.config_age_label.setToolTip("Config 7+ gün eski. Backtest çalıştırmayı düşünün.")
            else:
                # Yeşil - güncel
                self.config_age_label.setText(f"🟢 Config: {age_text}")
                self.config_age_label.setStyleSheet("""
                    color: #44ff44;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 5px 10px;
                    background-color: #114411;
                    border-radius: 8px;
                    border: 1px solid #44ff44;
                """)
                self.config_age_label.setToolTip("Config güncel.")

        except Exception as e:
            self.config_age_label.setText("⚠️ Config okunamadı")
            self.config_age_label.setStyleSheet("""
                color: #888;
                font-weight: bold;
                font-size: 12px;
                padding: 5px 10px;
                background-color: #1a1a1a;
                border-radius: 8px;
            """)
            print(f"[UI] Config yaşı okunamadı: {e}")

    def _update_blacklist_label(self):
        """Blacklist durumunu kontrol et ve UI'da göster."""
        try:
            # Dinamik blacklist'i yükle
            dynamic_bl = load_dynamic_blacklist()
            static_bl = POST_PORTFOLIO_BLACKLIST

            # Disabled streams from best_configs.json
            disabled_streams = []
            if os.path.exists(BEST_CONFIGS_FILE):
                try:
                    with open(BEST_CONFIGS_FILE, 'r', encoding='utf-8') as f:
                        best_cfgs = json.load(f)
                    if isinstance(best_cfgs, dict):
                        for sym in SYMBOLS:
                            sym_cfg = best_cfgs.get(sym, {})
                            if isinstance(sym_cfg, dict):
                                for tf in TIMEFRAMES:
                                    tf_cfg = sym_cfg.get(tf, {})
                                    if isinstance(tf_cfg, dict) and tf_cfg.get("disabled", False):
                                        disabled_streams.append((sym, tf))
                except json.JSONDecodeError:
                    pass  # JSON bozuksa disabled_streams boş kalır

            # Toplam sayıları hesapla
            total_streams = len(SYMBOLS) * len(TIMEFRAMES)
            blacklisted_count = len(dynamic_bl) + len([k for k in static_bl.keys() if static_bl.get(k)])
            disabled_count = len(disabled_streams)
            active_count = total_streams - blacklisted_count - disabled_count

            # Tüm devre dışı streamlerin listesini hazırla (tooltip için)
            all_inactive = []
            for key in dynamic_bl.keys():
                info = dynamic_bl[key]
                pnl = info.get('pnl', 0)
                all_inactive.append(f"{key[0]}-{key[1]} (PnL: ${pnl:.0f})")
            for key, val in static_bl.items():
                if val and key not in dynamic_bl:
                    all_inactive.append(f"{key[0]}-{key[1]} (manuel)")
            for key in disabled_streams:
                if key not in dynamic_bl and key not in static_bl:
                    all_inactive.append(f"{key[0]}-{key[1]} (disabled)")

            if blacklisted_count + disabled_count == 0:
                # Tümü aktif
                self.blacklist_label.setText(f"✅ {active_count}/{total_streams} aktif")
                self.blacklist_label.setStyleSheet("""
                    color: #44ff44;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 5px 10px;
                    background-color: #114411;
                    border-radius: 8px;
                    border: 1px solid #44ff44;
                """)
                self.blacklist_label.setToolTip("Tüm coin/timeframe kombinasyonları aktif.")
            elif active_count > total_streams * 0.5:
                # Çoğunluk aktif
                self.blacklist_label.setText(f"📊 {active_count}/{total_streams} aktif")
                self.blacklist_label.setStyleSheet("""
                    color: #aaaaff;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 5px 10px;
                    background-color: #222244;
                    border-radius: 8px;
                    border: 1px solid #6666aa;
                """)
                tooltip = "Devre dışı streamler:\n" + "\n".join(all_inactive[:20])
                if len(all_inactive) > 20:
                    tooltip += f"\n... ve {len(all_inactive) - 20} tane daha"
                self.blacklist_label.setToolTip(tooltip)
            else:
                # Az sayıda aktif
                self.blacklist_label.setText(f"⚠️ {active_count}/{total_streams} aktif")
                self.blacklist_label.setStyleSheet("""
                    color: #ffaa00;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 5px 10px;
                    background-color: #332200;
                    border-radius: 8px;
                    border: 1px solid #ffaa00;
                """)
                tooltip = "Devre dışı streamler:\n" + "\n".join(all_inactive[:20])
                if len(all_inactive) > 20:
                    tooltip += f"\n... ve {len(all_inactive) - 20} tane daha"
                self.blacklist_label.setToolTip(tooltip)

        except Exception as e:
            self.blacklist_label.setText("⚠️ Blacklist?")
            self.blacklist_label.setStyleSheet("""
                color: #888;
                font-weight: bold;
                font-size: 12px;
                padding: 5px 10px;
                background-color: #1a1a1a;
                border-radius: 8px;
            """)
            print(f"[UI] Blacklist okunamadı: {e}")

    def _refresh_config_tab(self):
        """Config sekmesindeki tüm bilgileri yenile."""
        try:
            # --- Config Durumu ---
            status_lines = []

            if not os.path.exists(BEST_CONFIGS_FILE):
                status_lines.append("❌ Config dosyası bulunamadı")
                status_lines.append(f"   Dosya: {BEST_CONFIGS_FILE}")
                status_lines.append("")
                status_lines.append("💡 Backtest çalıştırarak config oluşturun.")
                self.config_status_text.setPlainText("\n".join(status_lines))
                self.config_stats_text.setPlainText("Config yok - istatistik hesaplanamadı")
                self.active_streams_table.setRowCount(0)
                self.disabled_streams_table.setRowCount(0)
                return

            try:
                with open(BEST_CONFIGS_FILE, 'r', encoding='utf-8') as f:
                    best_cfgs = json.load(f)
            except json.JSONDecodeError as e:
                status_lines.append("❌ Config dosyası bozuk (JSON hatası)")
                status_lines.append(f"   Hata: {e}")
                status_lines.append("")
                status_lines.append("💡 'Config Dosyasını Sil' butonuna basıp yeni backtest çalıştırın.")
                self.config_status_text.setPlainText("\n".join(status_lines))
                self.config_stats_text.setPlainText("Config bozuk - istatistik hesaplanamadı")
                self.active_streams_table.setRowCount(0)
                self.disabled_streams_table.setRowCount(0)
                return

            if not isinstance(best_cfgs, dict):
                status_lines.append("❌ Config formatı geçersiz")
                self.config_status_text.setPlainText("\n".join(status_lines))
                return

            # Meta bilgileri
            meta = best_cfgs.get("_meta", {})
            saved_at_str = meta.get("saved_at", "Bilinmiyor")
            signature = meta.get("strategy_signature", "Yok")

            # Yaş hesapla
            age_text = "Bilinmiyor"
            if saved_at_str and saved_at_str != "Bilinmiyor":
                try:
                    saved_at = datetime.fromisoformat(saved_at_str.replace("Z", "+00:00"))
                    saved_at_naive = saved_at.replace(tzinfo=None)
                    now_naive = utcnow()
                    age = now_naive - saved_at_naive
                    age_days = age.days
                    age_hours = age.seconds // 3600
                    if age_days == 0:
                        age_text = f"{age_hours} saat önce"
                    elif age_days == 1:
                        age_text = "1 gün önce"
                    else:
                        age_text = f"{age_days} gün önce"
                except (ValueError, AttributeError, TypeError):
                    # Invalid date format or missing attributes - keep default "Bilinmiyor"
                    pass

            # İmza kontrolü
            current_sig = _strategy_signature()
            sig_match = signature == current_sig

            status_lines.append(f"✅ Config dosyası mevcut")
            status_lines.append(f"   Dosya: {BEST_CONFIGS_FILE}")
            status_lines.append(f"   Kayıt: {saved_at_str}")
            status_lines.append(f"   Yaş: {age_text}")
            status_lines.append(f"   İmza: {signature[:12]}...")
            if sig_match:
                status_lines.append(f"   ✅ İmza eşleşiyor (güncel)")
            else:
                status_lines.append(f"   ⚠️ İmza eşleşmiyor! Backtest çalıştırın.")
                status_lines.append(f"      Beklenen: {current_sig[:12]}...")

            self.config_status_text.setPlainText("\n".join(status_lines))

            # --- İstatistikler ---
            dynamic_bl = load_dynamic_blacklist()
            static_bl = POST_PORTFOLIO_BLACKLIST

            active_streams = []
            disabled_streams = []

            for sym in SYMBOLS:
                sym_cfg = best_cfgs.get(sym, {})
                if not isinstance(sym_cfg, dict):
                    continue
                for tf in TIMEFRAMES:
                    tf_cfg = sym_cfg.get(tf, {})
                    if not isinstance(tf_cfg, dict):
                        continue

                    key = (sym, tf)
                    is_disabled = tf_cfg.get("disabled", False)
                    is_dynamic_bl = key in dynamic_bl
                    is_static_bl = static_bl.get(key, False)

                    if is_disabled or is_dynamic_bl or is_static_bl:
                        reason = []
                        pnl = 0
                        if is_disabled:
                            reason.append("disabled")
                        if is_dynamic_bl:
                            reason.append("dynamic_bl")
                            pnl = dynamic_bl[key].get('pnl', 0)
                        if is_static_bl:
                            reason.append("static_bl")
                        disabled_streams.append({
                            "sym": sym, "tf": tf,
                            "reason": ", ".join(reason),
                            "pnl": pnl
                        })
                    else:
                        active_streams.append({
                            "sym": sym, "tf": tf,
                            "rr": tf_cfg.get("rr", "-"),
                            "rsi": tf_cfg.get("rsi", "-"),
                            "at": "Açık" if tf_cfg.get("at_active", True) else "Kapalı",
                            "trailing": "Açık" if tf_cfg.get("use_trailing", False) else "Kapalı",
                            "strategy": tf_cfg.get("strategy_mode", "ssl_flow")[:10]
                        })

            total_streams = len(SYMBOLS) * len(TIMEFRAMES)
            stats_lines = [
                f"📊 Toplam stream sayısı: {total_streams}",
                f"✅ Aktif streamler: {len(active_streams)}",
                f"🚫 Devre dışı: {len(disabled_streams)}",
                f"   - Dinamik blacklist: {len(dynamic_bl)}",
                f"   - Statik blacklist: {len([k for k,v in static_bl.items() if v])}",
                f"   - Config disabled: {len([d for d in disabled_streams if 'disabled' in d['reason']])}",
            ]
            self.config_stats_text.setPlainText("\n".join(stats_lines))

            # --- Aktif Streamler Tablosu ---
            self.active_streams_table.setRowCount(len(active_streams))
            for i, stream in enumerate(active_streams):
                self.active_streams_table.setItem(i, 0, QTableWidgetItem(stream["sym"]))
                self.active_streams_table.setItem(i, 1, QTableWidgetItem(stream["tf"]))
                self.active_streams_table.setItem(i, 2, QTableWidgetItem(str(stream["rr"])))
                self.active_streams_table.setItem(i, 3, QTableWidgetItem(str(stream["rsi"])))
                self.active_streams_table.setItem(i, 4, QTableWidgetItem(stream["at"]))
                self.active_streams_table.setItem(i, 5, QTableWidgetItem(stream["trailing"]))
                self.active_streams_table.setItem(i, 6, QTableWidgetItem(stream["strategy"]))

            # --- Devre Dışı Streamler Tablosu ---
            self.disabled_streams_table.setRowCount(len(disabled_streams))
            for i, stream in enumerate(disabled_streams):
                self.disabled_streams_table.setItem(i, 0, QTableWidgetItem(stream["sym"]))
                self.disabled_streams_table.setItem(i, 1, QTableWidgetItem(stream["tf"]))
                self.disabled_streams_table.setItem(i, 2, QTableWidgetItem(stream["reason"]))
                pnl_text = f"${stream['pnl']:.0f}" if stream['pnl'] != 0 else "-"
                self.disabled_streams_table.setItem(i, 3, QTableWidgetItem(pnl_text))

            # Ayrıca üstteki label'ları da güncelle
            self._update_config_age_label()
            self._update_blacklist_label()

        except Exception as e:
            self.config_status_text.setPlainText(f"❌ Hata: {e}")
            print(f"[UI] Config sekmesi yenileme hatası: {e}")

    def _delete_config_file(self):
        """Config dosyasını sil ve UI'ı güncelle."""
        try:
            if os.path.exists(BEST_CONFIGS_FILE):
                os.remove(BEST_CONFIGS_FILE)
                print(f"[CFG] ✓ Config dosyası silindi: {BEST_CONFIGS_FILE}")

                # Cache'i temizle
                global BEST_CONFIG_CACHE, BEST_CONFIG_WARNING_FLAGS
                BEST_CONFIG_CACHE.clear()
                BEST_CONFIG_WARNING_FLAGS["missing_signature"] = False
                BEST_CONFIG_WARNING_FLAGS["signature_mismatch"] = False
                BEST_CONFIG_WARNING_FLAGS["json_error"] = False
                BEST_CONFIG_WARNING_FLAGS["load_error"] = False

                # UI'ı güncelle
                self._refresh_config_tab()
                self._update_config_age_label()
                self._update_blacklist_label()
            else:
                print("[CFG] Config dosyası zaten yok.")
        except Exception as e:
            print(f"[CFG] ⚠️ Config silme hatası: {e}")

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

        selected_tfs = self.get_selected_timeframes(getattr(self, "backtest_tf_checks", {}))

        # Sabit tarih aralığı mı, gün sayısı mı?
        use_fixed_dates = self.radio_fixed_dates.isChecked()

        if use_fixed_dates:
            # Sabit tarih aralığı modu - tutarlı sonuçlar için
            start_date = self.backtest_start_date.date().toString("yyyy-MM-dd")
            end_date = self.backtest_end_date.date().toString("yyyy-MM-dd")
            self.backtest_logs.append(f"🧪 Backtest başlatıldı (📅 {start_date} → {end_date})")
            self.backtest_logs.append("✅ Sabit tarih modu: Tutarlı/karşılaştırılabilir sonuçlar")

            # Hız ayarlarını oku
            skip_opt = self.chk_skip_optimization.isChecked()
            quick = self.chk_quick_mode.isChecked()
            if skip_opt:
                self.backtest_logs.append("⚡ Optimizer atlanıyor (kayıtlı config kullanılacak)")
            elif quick:
                self.backtest_logs.append("🚀 Hızlı mod aktif (13 config)")

            self.backtest_worker = BacktestWorker(
                SYMBOLS, selected_tfs, 0, skip_opt, quick,
                use_days=False, start_date=start_date, end_date=end_date
            )
        else:
            # Gün sayısı modu - ARTIK SABİT TARİH ARAILIĞINA DÖNÜŞTÜRÜLECek
            # Bu şekilde her çalıştırmada tutarlı sonuçlar alınır
            days = self.backtest_days.value()

            # Bugünden X gün öncesini hesapla
            from datetime import date, timedelta
            end_date = date.today().strftime("%Y-%m-%d")
            start_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")

            self.backtest_logs.append(f"🧪 Backtest başlatıldı ({days} gün: {start_date} → {end_date})")
            self.backtest_logs.append("✅ Tüm timeframe'ler aynı tarih aralığını kullanacak")

            # Hız ayarlarını oku
            skip_opt = self.chk_skip_optimization.isChecked()
            quick = self.chk_quick_mode.isChecked()
            if skip_opt:
                self.backtest_logs.append("⚡ Optimizer atlanıyor (kayıtlı config kullanılacak)")
            elif quick:
                self.backtest_logs.append("🚀 Hızlı mod aktif (13 config)")

            # Artık tarih aralığı modunu kullan (tutarlı sonuçlar için)
            self.backtest_worker = BacktestWorker(
                SYMBOLS, selected_tfs, 0, skip_opt, quick,
                use_days=False, start_date=start_date, end_date=end_date
            )

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
            finished_at = utcnow().isoformat() + "Z"
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
            # Config yaşı ve blacklist göstergelerini güncelle
            self._update_config_age_label()
            self._update_blacklist_label()
            # Config sekmesini de güncelle
            self._refresh_config_tab()

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
        except (ValueError, TypeError, AttributeError):
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
        days = self.opt_days.value()
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

        # Worker'a days parametresini gönder (TF başına mum sayısı dönüşümü worker içinde yapılacak)
        self.opt_worker = OptimizerWorker(selected_sym, days, rr_range, rsi_range, slope_range, use_at,
                                          is_monte_carlo, selected_tfs, use_days=True)
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
# 📊 PORTFOLIO BACKTEST RUNNER (v40.5 - MODULAR)
# ==========================================
# run_portfolio_backtest is now in runners/portfolio.py
# Imported below from runners package (SINGLE SOURCE OF TRUTH)
# ==========================================

from runners import run_portfolio_backtest


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
    # Lazy import matplotlib.pyplot
    plt = get_plt()

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
            except (ValueError, TypeError, AttributeError):
                pass

        for fallback_field in ("open_time_utc", "timestamp"):
            if fallback_field in row:
                try:
                    ts = pd.to_datetime(row.get(fallback_field), utc=True)
                    return ts
                except (ValueError, TypeError, AttributeError):
                    continue
        return None

    def _parse_event_log(raw_val):
        if isinstance(raw_val, str):
            try:
                return json.loads(raw_val)
            except (json.JSONDecodeError, TypeError):
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
            except (KeyError, ValueError, TypeError):
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
            except (ValueError, TypeError):
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
    trades_csv: str = None,  # Default: DATA_DIR/backtest_trades.csv
    max_trades: Optional[int] = None,
    window: int = 60,
    save_dir: Optional[str] = None,  # Default: DATA_DIR/replay_charts
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
    # Set default paths to DATA_DIR
    if trades_csv is None:
        trades_csv = os.path.join(DATA_DIR, "backtest_trades.csv")
    if save_dir is None:
        save_dir = os.path.join(DATA_DIR, "replay_charts")

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

        prices_path = os.path.join(DATA_DIR, f"{symbol}_{timeframe}_prices.csv")
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
    except (RuntimeError, AttributeError):
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
    start_date: str = None,
    end_date: str = None,
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
        start_date: Start date in "YYYY-MM-DD" format (for consistent/reproducible backtests)
        end_date: End date in "YYYY-MM-DD" format (default: today)

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
        >>> # For reproducible results with fixed date range:
        >>> results = run_cli_backtest(
        ...     symbols=['BTCUSDT'],
        ...     timeframes=['15m'],
        ...     start_date='2024-11-15',
        ...     end_date='2024-12-15'
        ... )
        >>> print(results['summary'])
    """
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES

    # Tarih aralığı modu mu kontrol et
    use_date_range = start_date is not None

    def log(msg):
        if verbose:
            print(msg)

    log("=" * 60)
    log("🚀 CLI/Colab Backtest Başlatılıyor")
    log(f"   Symbols: {symbols}")
    log(f"   Timeframes: {timeframes}")
    if use_date_range:
        log(f"   Date Range: {start_date} → {end_date or 'bugün'}")
    else:
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
            if use_date_range:
                # Tarih aralığı modu
                df = TradingEngine.get_historical_data_pagination(
                    sym, tf, start_date=start_date, end_date=end_date
                )
            else:
                # Candle sayısı modu
                limit = BACKTEST_CANDLE_LIMITS.get(tf, candles)
                limit = min(limit, candles)
                df = TradingEngine.get_historical_data_pagination(sym, tf, total_candles=limit)

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
        out_trades_csv=os.path.join(DATA_DIR, "backtest_trades.csv") if save_results else None,
        out_summary_csv=os.path.join(DATA_DIR, "backtest_summary.csv") if save_results else None,
        progress_callback=lambda msg: log(f"   {msg}") if verbose else None,
        draw_trades=False,  # Don't draw charts in CLI mode
        start_date=start_date,
        end_date=end_date,
    )

    # Extract results
    summary_rows = result.get("summary_rows", [])
    all_trades = result.get("all_trades", [])

    # Convert to DataFrames
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    # Calculate aggregate metrics - group by unique trade ID to avoid counting partial legs
    metrics = {}
    if not trades_df.empty:
        pnl_col = 'pnl' if 'pnl' in trades_df.columns else None
        if pnl_col:
            # Group by trade ID to get net PnL per trade (handles partial exits)
            if 'id' in trades_df.columns:
                trade_pnls = trades_df.groupby('id')[pnl_col].sum()
                metrics['total_pnl'] = trade_pnls.sum()
                metrics['total_trades'] = len(trade_pnls)
                metrics['winning_trades'] = len(trade_pnls[trade_pnls > 0])
                metrics['losing_trades'] = len(trade_pnls[trade_pnls < 0])
            else:
                metrics['total_pnl'] = trades_df[pnl_col].sum()
                metrics['total_trades'] = len(trades_df)
                metrics['winning_trades'] = len(trades_df[trades_df[pnl_col] > 0])
                metrics['losing_trades'] = len(trades_df[trades_df[pnl_col] < 0])

            metrics['win_rate'] = metrics['winning_trades'] / max(1, metrics['total_trades'])
            metrics['avg_pnl'] = metrics['total_pnl'] / max(1, metrics['total_trades'])

            # R-Multiple metrics
            if 'r_multiple' in trades_df.columns:
                if 'id' in trades_df.columns:
                    trade_r = trades_df.groupby('id')['r_multiple'].sum()
                    metrics['total_r'] = trade_r.sum()
                    metrics['avg_r'] = trade_r.mean()
                else:
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


# ==========================================
# 🧪 BASELINE TEST - Optimizer'sız Karşılaştırma (v40.5)
# ==========================================
# Bu fonksiyonlar optimizer'ın gerçekten değer katıp katmadığını test eder.
# Sabit config ile backtest yapılır, sonuçlar optimized ile karşılaştırılır.
# ==========================================

# Baseline config - "makul" sabit değerler
BASELINE_CONFIG = {
    "rr": 2.0,
    "rsi": 70,
    "slope": 0.5,
    "at_active": True,  # AlphaTrend ZORUNLU - SSL Flow için gerekli
    "use_trailing": False,
    "use_partial": True,  # Partial TP aktif
    "use_dynamic_pbema_tp": True,
    "strategy_mode": "ssl_flow",
    "disabled": False,
    "confidence": "high",
    # === Partial TP Parameters ===
    "partial_trigger": 0.65,         # Progress seviyesi (0.65 = %65'te partial TP al)
    "partial_fraction": 0.33,        # Kapatılacak pozisyon oranı (%33)
    "partial_rr_adjustment": True,   # RR'a göre partial trigger ayarlama
    "partial_rr_high_threshold": 1.8,  # Yüksek RR eşiği
    "partial_rr_high_trigger": 0.75,   # Yüksek RR için partial trigger (%75)
    "partial_rr_low_threshold": 1.2,   # Düşük RR eşiği
    "partial_rr_low_trigger": 0.55,    # Düşük RR için partial trigger (%55)
    # === Dynamic TP Parameters ===
    "dynamic_tp_only_after_partial": True,  # Dynamic TP sadece partial sonrası aktif
    "dynamic_tp_min_distance": 0.004,       # Minimum mesafe (entry'den %0.4)
}

# Alternatif baseline configs (karşılaştırma için)
# NOT: AlphaTrend tüm config'lerde ZORUNLU açık
# NOT: Yeni partial TP parametreleri BASELINE_CONFIG'dan miras alınacak
BASELINE_CONFIGS_ALT = {
    "conservative": {
        "rr": 1.2, "rsi": 45, "at_active": True,
        "use_trailing": False, "use_partial": True, "use_dynamic_pbema_tp": True,
        "strategy_mode": "ssl_flow", "disabled": False, "confidence": "high",
        "partial_trigger": 0.55, "partial_fraction": 0.33, "partial_rr_adjustment": True,
        "dynamic_tp_only_after_partial": True, "dynamic_tp_min_distance": 0.004,
    },
    "aggressive": {
        "rr": 2.1, "rsi": 35, "at_active": True,
        "use_trailing": False, "use_partial": True, "use_dynamic_pbema_tp": True,
        "strategy_mode": "ssl_flow", "disabled": False, "confidence": "high",
        "partial_trigger": 0.75, "partial_fraction": 0.33, "partial_rr_adjustment": True,
        "dynamic_tp_only_after_partial": True, "dynamic_tp_min_distance": 0.004,
    },
    "standard": {
        "rr": 2.0, "rsi": 70, "at_active": True,
        "use_trailing": False, "use_partial": True, "use_dynamic_pbema_tp": True,
        "strategy_mode": "ssl_flow", "disabled": False, "confidence": "high",
        "partial_trigger": 0.65, "partial_fraction": 0.33, "partial_rr_adjustment": True,
        "dynamic_tp_only_after_partial": True, "dynamic_tp_min_distance": 0.004,
    },
}


def run_baseline_test(
    symbols: list = None,
    timeframes: list = None,
    start_date: str = None,
    end_date: str = None,
    candles: int = 50000,
    baseline_config: dict = None,
    verbose: bool = True,
) -> dict:
    """Run backtest with fixed config (no optimization).

    Bu fonksiyon optimizer'ı tamamen atlar ve tüm streamler için
    aynı sabit config kullanır. Optimizer'ın değer katıp katmadığını
    test etmek için kullanılır.

    Args:
        symbols: Test edilecek semboller (default: SYMBOLS)
        timeframes: Test edilecek zaman dilimleri (default: TIMEFRAMES)
        start_date: Başlangıç tarihi "YYYY-MM-DD" (zorunlu)
        end_date: Bitiş tarihi "YYYY-MM-DD" (default: bugün)
        candles: Mum sayısı (start_date verilmezse kullanılır)
        baseline_config: Sabit config (default: BASELINE_CONFIG)
        verbose: Detaylı çıktı

    Returns:
        dict with:
        - 'summary': DataFrame with backtest summary
        - 'trades': DataFrame with all trades
        - 'metrics': dict of aggregate metrics
        - 'config_used': Kullanılan sabit config
    """
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES
    baseline_config = baseline_config or BASELINE_CONFIG

    def log(msg):
        if verbose:
            print(msg)

    log("=" * 60)
    log("🧪 BASELINE TEST - Optimizer KAPALI")
    log("=" * 60)
    log(f"   Config: RR={baseline_config['rr']}, RSI={baseline_config['rsi']}, "
        f"AT={'Açık' if baseline_config['at_active'] else 'Kapalı'}")
    log(f"   Symbols: {len(symbols)} adet")
    log(f"   Timeframes: {timeframes}")
    if start_date:
        log(f"   Date Range: {start_date} → {end_date or 'bugün'}")
    else:
        log(f"   Candles: {candles}")
    log("=" * 60)

    # Step 1: Fetch data
    log("\n📊 Veri indiriliyor...")
    streams = {}
    pairs = [(s, tf) for s in symbols for tf in timeframes]

    for sym, tf in pairs:
        try:
            if start_date:
                df = TradingEngine.get_historical_data_pagination(
                    sym, tf, start_date=start_date, end_date=end_date
                )
            else:
                limit = BACKTEST_CANDLE_LIMITS.get(tf, candles)
                limit = min(limit, candles)
                df = TradingEngine.get_historical_data_pagination(sym, tf, total_candles=limit)

            if df is not None and len(df) > 300:
                df = TradingEngine.calculate_indicators(df)
                streams[(sym, tf)] = df
                log(f"   ✓ {sym}-{tf}: {len(df)} mum")
            else:
                log(f"   ✗ {sym}-{tf}: Yetersiz veri")
        except Exception as e:
            log(f"   ✗ {sym}-{tf}: Hata - {e}")

    if not streams:
        log("❌ Hiç veri indirilemedi!")
        return {"summary": None, "trades": None, "metrics": {}, "config_used": baseline_config}

    log(f"\n✓ {len(streams)} stream hazır")

    # Step 2: Create fixed configs for all streams (NO OPTIMIZATION)
    log("\n⚙️ Sabit config tüm streamlere uygulanıyor (optimizer YOK)...")
    best_configs = {}
    for (sym, tf) in streams.keys():
        # Her stream için aynı sabit config
        best_configs[(sym, tf)] = {
            **baseline_config,
            "_net_pnl": 0,  # Bilinmiyor
            "_trades": 0,
            "_score": 0,
            "_expected_r": 0,
        }

    # Temporarily save configs to file for run_portfolio_backtest to use
    temp_configs_file = os.path.join(DATA_DIR, "baseline_configs_temp.json")
    try:
        # Convert to JSON format
        json_configs = {}
        for (sym, tf), cfg in best_configs.items():
            if sym not in json_configs:
                json_configs[sym] = {}
            json_configs[sym][tf] = cfg

        with open(temp_configs_file, "w", encoding="utf-8") as f:
            json.dump(json_configs, f, indent=2)

        # Backup original best_configs.json
        original_configs_backup = None
        if os.path.exists(BEST_CONFIGS_FILE):
            with open(BEST_CONFIGS_FILE, "r", encoding="utf-8") as f:
                original_configs_backup = f.read()

        # Replace with baseline configs
        with open(BEST_CONFIGS_FILE, "w", encoding="utf-8") as f:
            json.dump(json_configs, f, indent=2)

        # Step 3: Run backtest with skip_optimization=True
        log("\n🔬 Baseline backtest çalıştırılıyor...")

        result = run_portfolio_backtest(
            symbols=symbols,
            timeframes=timeframes,
            candles=candles,
            out_trades_csv=os.path.join(DATA_DIR, "baseline_trades.csv"),
            out_summary_csv=os.path.join(DATA_DIR, "baseline_summary.csv"),
            progress_callback=lambda msg: log(f"   {msg}") if verbose else None,
            draw_trades=False,
            skip_optimization=True,  # CRITICAL: Optimizer'ı atla
            start_date=start_date,
            end_date=end_date,
        )

    finally:
        # Restore original best_configs.json
        if original_configs_backup:
            with open(BEST_CONFIGS_FILE, "w", encoding="utf-8") as f:
                f.write(original_configs_backup)
        elif os.path.exists(BEST_CONFIGS_FILE):
            os.remove(BEST_CONFIGS_FILE)

        # Clean up temp file
        if os.path.exists(temp_configs_file):
            os.remove(temp_configs_file)

    # Extract results
    summary_rows = result.get("summary_rows", [])
    all_trades = result.get("all_trades", [])

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    # Filter out FORCE_CLOSE trades (backtest artifacts, not real results)
    force_close_count = 0
    force_close_pnl = 0.0
    if not trades_df.empty and 'status' in trades_df.columns:
        force_close_mask = trades_df['status'].str.contains('FORCE', case=False, na=False)
        force_close_count = force_close_mask.sum()
        if force_close_count > 0 and 'pnl' in trades_df.columns:
            force_close_pnl = trades_df.loc[force_close_mask, 'pnl'].sum()
            log(f"\n⚠️ {force_close_count} FORCE_CLOSE trade çıkarıldı (yapay PnL=${force_close_pnl:.2f})")
        trades_df = trades_df[~force_close_mask].copy()

    # Calculate metrics - group by unique trade ID to avoid counting partial legs multiple times
    metrics = {}
    if not trades_df.empty:
        pnl_col = 'pnl' if 'pnl' in trades_df.columns else None
        if pnl_col:
            # Group by trade ID to get net PnL per trade (handles partial exits)
            if 'id' in trades_df.columns:
                trade_pnls = trades_df.groupby('id')[pnl_col].sum()
                metrics['total_pnl'] = trade_pnls.sum()
                metrics['total_trades'] = len(trade_pnls)
                metrics['winning_trades'] = len(trade_pnls[trade_pnls > 0])
                metrics['losing_trades'] = len(trade_pnls[trade_pnls < 0])
            else:
                metrics['total_pnl'] = trades_df[pnl_col].sum()
                metrics['total_trades'] = len(trades_df)
                metrics['winning_trades'] = len(trades_df[trades_df[pnl_col] > 0])
                metrics['losing_trades'] = len(trades_df[trades_df[pnl_col] < 0])

            metrics['win_rate'] = metrics['winning_trades'] / max(1, metrics['total_trades'])
            metrics['avg_pnl'] = metrics['total_pnl'] / max(1, metrics['total_trades'])

            if 'r_multiple' in trades_df.columns:
                # R-multiple should also be summed per trade
                if 'id' in trades_df.columns:
                    trade_r = trades_df.groupby('id')['r_multiple'].sum()
                    metrics['total_r'] = trade_r.sum()
                    metrics['avg_r'] = trade_r.mean()
                else:
                    metrics['total_r'] = trades_df['r_multiple'].sum()
                    metrics['avg_r'] = trades_df['r_multiple'].mean()
                metrics['expected_r'] = metrics['avg_r']

    # Print summary
    log("\n" + "=" * 60)
    log("📈 BASELINE TEST SONUÇLARI")
    log("=" * 60)
    log(f"   Config: RR={baseline_config['rr']}, RSI={baseline_config['rsi']}, "
        f"AT={'Açık' if baseline_config['at_active'] else 'Kapalı'}")
    if metrics:
        log(f"   Toplam PnL: ${metrics.get('total_pnl', 0):.2f}")
        log(f"   Toplam Trade: {metrics.get('total_trades', 0)}")
        log(f"   Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
        log(f"   E[R]: {metrics.get('expected_r', 0):.3f}")
    log("=" * 60)

    # Anomaly detection - flag individual trades with unusually high PnL
    # NOTE: This is WARNING ONLY - anomalies are NOT excluded from results
    # Purpose: Help identify potential bugs in PnL calculation
    anomalies = []
    if not trades_df.empty and 'id' in trades_df.columns and 'pnl' in trades_df.columns:
        # Expected max PnL per SINGLE trade:
        # Risk per trade: 1.75% of $2000 = $35
        # Max RR in config grid: 2.4
        # With some buffer for price movement: $35 * 2.4 * 1.5 = $126
        # Anything above $200 per single trade is suspicious
        max_expected_pnl_per_trade = 200

        trade_pnls = trades_df.groupby('id').agg({
            'pnl': 'sum',
            'symbol': 'first',
            'timeframe': 'first',
            'entry': 'first',
            'close_price': 'last',
            'size': 'first',
        }).reset_index()

        for _, row in trade_pnls.iterrows():
            # Flag only if SINGLE trade PnL is abnormally high
            if abs(row['pnl']) > max_expected_pnl_per_trade:
                anomalies.append({
                    'id': row['id'],
                    'symbol': row['symbol'],
                    'timeframe': row['timeframe'],
                    'pnl': row['pnl'],
                    'entry': row.get('entry'),
                    'exit': row.get('close_price'),
                    'size': row.get('size'),
                })

        if anomalies:
            log("\n⚠️ ANOMALI UYARISI (sonuçlar ETKİLENMİYOR, sadece bilgi):")
            log(f"   {len(anomalies)} trade beklenenden yüksek PnL gösteriyor (>${max_expected_pnl_per_trade}/trade)")
            for a in anomalies:
                implied_rr = abs(a['pnl']) / 35 if a['pnl'] else 0  # $35 = expected risk
                log(f"   Trade #{a['id']} {a['symbol']}-{a['timeframe']}: "
                    f"PnL=${a['pnl']:.2f} (implied RR={implied_rr:.1f}x), "
                    f"Entry={a['entry']}, Exit={a['exit']}, Size={a['size']:.6f}")
            log("   💡 Bu trade'leri doğrulamak için CSV'yi inceleyin")

    return {
        "summary": summary_df,
        "trades": trades_df,
        "metrics": metrics,
        "config_used": baseline_config,
        "anomalies": anomalies,
    }


def compare_baseline_vs_optimized(
    symbols: list = None,
    timeframes: list = None,
    start_date: str = None,
    end_date: str = None,
    candles: int = 50000,
    verbose: bool = True,
) -> dict:
    """Run both baseline and optimized backtests and compare results.

    Bu fonksiyon aynı veri üzerinde:
    1. Sabit config ile baseline test
    2. Optimizer ile optimized test
    çalıştırır ve sonuçları karşılaştırır.

    Args:
        symbols: Test edilecek semboller
        timeframes: Test edilecek zaman dilimleri
        start_date: Başlangıç tarihi "YYYY-MM-DD"
        end_date: Bitiş tarihi "YYYY-MM-DD"
        candles: Mum sayısı
        verbose: Detaylı çıktı

    Returns:
        dict with comparison results
    """
    def log(msg):
        if verbose:
            print(msg)

    log("\n" + "=" * 70)
    log("🔬 BASELINE vs OPTIMIZED KARŞILAŞTIRMA TESTİ")
    log("=" * 70)

    # Test 1: Baseline (no optimization)
    log("\n" + "─" * 70)
    log("TEST 1: BASELINE (Optimizer KAPALI)")
    log("─" * 70)

    baseline_result = run_baseline_test(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        candles=candles,
        verbose=verbose,
    )

    # Test 2: Optimized
    log("\n" + "─" * 70)
    log("TEST 2: OPTIMIZED (Optimizer AÇIK)")
    log("─" * 70)

    optimized_result = run_cli_backtest(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        candles=candles,
        optimize=True,
        walk_forward=True,
        verbose=verbose,
    )

    # Compare results
    baseline_metrics = baseline_result.get('metrics', {})
    optimized_metrics = optimized_result.get('metrics', {})

    baseline_pnl = baseline_metrics.get('total_pnl', 0)
    optimized_pnl = optimized_metrics.get('total_pnl', 0)

    baseline_trades = baseline_metrics.get('total_trades', 0)
    optimized_trades = optimized_metrics.get('total_trades', 0)

    baseline_wr = baseline_metrics.get('win_rate', 0) * 100
    optimized_wr = optimized_metrics.get('win_rate', 0) * 100

    baseline_er = baseline_metrics.get('expected_r', 0)
    optimized_er = optimized_metrics.get('expected_r', 0)

    # Calculate differences
    pnl_diff = optimized_pnl - baseline_pnl
    pnl_diff_pct = (pnl_diff / abs(baseline_pnl) * 100) if baseline_pnl != 0 else 0

    # Print comparison
    log("\n" + "=" * 70)
    log("📊 KARŞILAŞTIRMA SONUÇLARI")
    log("=" * 70)
    log(f"{'Metrik':<20} {'Baseline':>15} {'Optimized':>15} {'Fark':>15}")
    log("─" * 70)
    log(f"{'Toplam PnL':<20} ${baseline_pnl:>14.2f} ${optimized_pnl:>14.2f} ${pnl_diff:>+14.2f}")
    log(f"{'Trade Sayısı':<20} {baseline_trades:>15} {optimized_trades:>15} {optimized_trades - baseline_trades:>+15}")
    log(f"{'Win Rate':<20} {baseline_wr:>14.1f}% {optimized_wr:>14.1f}% {optimized_wr - baseline_wr:>+14.1f}%")
    log(f"{'E[R]':<20} {baseline_er:>15.3f} {optimized_er:>15.3f} {optimized_er - baseline_er:>+15.3f}")
    log("=" * 70)

    # Verdict
    log("\n🎯 KARAR:")
    if pnl_diff > 0:
        log(f"   Optimizer ${pnl_diff:.2f} ({pnl_diff_pct:+.1f}%) daha fazla kazandı")
        if pnl_diff_pct > 20:
            log("   ✓ Optimizer DEĞER KATIYOR - devam edilebilir")
        else:
            log("   ~ Optimizer marjinal fayda sağlıyor - basitleştirme düşünülebilir")
    elif pnl_diff < 0:
        log(f"   Baseline ${-pnl_diff:.2f} ({-pnl_diff_pct:.1f}%) daha fazla kazandı")
        log("   ❌ Optimizer DEĞER KATMIYOR - sabit config daha iyi!")
    else:
        log("   Sonuçlar eşit - optimizer gereksiz karmaşıklık ekliyor olabilir")

    log("=" * 70)

    return {
        "baseline": baseline_result,
        "optimized": optimized_result,
        "comparison": {
            "baseline_pnl": baseline_pnl,
            "optimized_pnl": optimized_pnl,
            "pnl_difference": pnl_diff,
            "pnl_difference_pct": pnl_diff_pct,
            "baseline_trades": baseline_trades,
            "optimized_trades": optimized_trades,
            "baseline_win_rate": baseline_wr,
            "optimized_win_rate": optimized_wr,
            "baseline_er": baseline_er,
            "optimized_er": optimized_er,
            "optimizer_adds_value": pnl_diff > 0,
        }
    }


# ==========================================
# 🔬 ROLLING WALK-FORWARD FRAMEWORK (v41.0)
# ==========================================
# Stitched OOS backtest for reliable, additive results.
# Solves: "30 days ≠ 15 days + 15 days" problem
#
# Key rules:
# 1. Train [t-lookback, t) and Trade [t, t+forward) NEVER overlap
# 2. Persist isolation: best_configs/blacklist writes disabled
# 3. Position snapshot: Open trades managed by entry snapshot, not global config
# 4. Realized PnL only: Trade PnL goes to window where it CLOSES
# ==========================================



# ==========================================
# 🔬 ROLLING WALK-FORWARD RUNNERS (v40.5 - MODULAR)
# ==========================================
# All rolling WF functions are now in runners/rolling_wf.py:
# - run_rolling_walkforward, compare_rolling_modes, compare_rolling_modes_fast
# - run_quick_rolling_test
# Imported below from runners package (SINGLE SOURCE OF TRUTH)
# ==========================================

from runners import (
    run_rolling_walkforward,
    compare_rolling_modes,
    compare_rolling_modes_fast,
    run_quick_rolling_test,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trading Bot v39.0 - R-Multiple Based with Walk-Forward")
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (CLI backtest)')
    parser.add_argument('--symbols', nargs='+', help='Symbols to test (e.g., BTCUSDT ETHUSDT)')
    parser.add_argument('--timeframes', nargs='+', help='Timeframes to test (e.g., 5m 15m 1h)')
    parser.add_argument('--candles', type=int, default=50000, help='Number of candles (default: 50000)')
    parser.add_argument('--start-date', type=str, help='Start date YYYY-MM-DD (for reproducible backtests)')
    parser.add_argument('--end-date', type=str, help='End date YYYY-MM-DD (default: today)')
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
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print("\n✅ Backtest tamamlandı!")
        if results['metrics']:
            print(f"   Total PnL: ${results['metrics'].get('total_pnl', 0):.2f}")
            print(f"   E[R]: {results['metrics'].get('expected_r', 0):.3f}")
    else:
        # GUI mode
        run_with_auto_restart()






