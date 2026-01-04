# ==========================================
# MainWindow - PyQt5 Main Application Window
# ==========================================
# This module contains the main GUI window for the trading bot.
# Includes: Dashboard, Open Trades, History, Potential Trades,
# Backtest, Optimization, and Config tabs.
# ==========================================

import os
import sys
import json
import time
import traceback
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd
import dateutil.parser

# Core imports
from core import (
    TradingEngine,
    SYMBOL_PARAMS,
    TRADING_CONFIG,
    utcnow,  # Replacement for deprecated utcnow()
)

# Import workers
from .workers import (
    LiveBotWorker,
    OptimizerWorker,
    BacktestWorker,
    AutoBacktestWorker,
)

# Lazy import of main module
_main_module = None

def _get_main_module():
    """Lazy import of main module to avoid circular imports."""
    global _main_module
    if _main_module is None:
        import desktop_bot_refactored_v2_base_v7 as main
        _main_module = main
    return _main_module


# PyQt5 imports - handle headless mode
IS_HEADLESS = os.environ.get('HEADLESS_MODE', '').lower() in ('1', 'true', 'yes')
IS_COLAB = 'google.colab' in sys.modules or os.environ.get('COLAB_RELEASE_TAG') is not None

if not IS_HEADLESS and not IS_COLAB:
    try:
        from PyQt5.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout,
            QHBoxLayout, QGridLayout, QTabWidget, QTextEdit, QLabel,
            QTableWidget, QTableWidgetItem, QPushButton, QHeaderView,
            QGroupBox, QDoubleSpinBox, QComboBox, QMessageBox, QCheckBox,
            QLineEdit, QSpinBox, QFrame, QRadioButton, QDateEdit
        )
        from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QDate
        from PyQt5.QtGui import QColor, QFont
        HAS_GUI = True
    except ImportError:
        HAS_GUI = False
else:
    HAS_GUI = False


class MainWindow(QMainWindow):
    """Main application window with all trading bot UI components."""

    def __init__(self):
        super().__init__()
        main = _get_main_module()

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

        self.tg_token = ""
        self.tg_chat_id = ""
        self.load_config()
        self.views_ready = {}
        self.show_rr_tools = True
        self.data_cache = {sym: {tf: (None, None) for tf in main.TIMEFRAMES} for sym in main.SYMBOLS}
        self.current_symbol = main.SYMBOLS[0]
        self.backtest_worker = None
        self.backtest_meta = None
        self.potential_entries = self._load_potential_entries()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- TICKER WIDGET ---
        ticker_widget = QWidget()
        ticker_widget.setStyleSheet("background-color: #1a1a1a; border-bottom: 2px solid #00ccff;")
        ticker_layout = QHBoxLayout(ticker_widget)
        ticker_layout.setContentsMargins(10, 5, 10, 5)

        self.price_labels = {}
        for sym in main.SYMBOLS:
            lbl_name = QLabel(sym.replace("USDT", ""))
            lbl_name.setStyleSheet("color: #888; font-weight: bold; font-size: 14px;")
            lbl_price = QLabel("---")
            lbl_price.setStyleSheet("color: #00ccff; font-weight: bold; font-size: 16px;")
            self.price_labels[sym] = lbl_price
            ticker_layout.addWidget(lbl_name)
            ticker_layout.addWidget(lbl_price)
            ticker_layout.addSpacing(20)

        ticker_layout.addStretch()

        self.status_label = QLabel("TRADE ARIYOR...")
        self.status_label.setStyleSheet("""
            color: #00ff00; font-weight: bold; font-size: 14px;
            padding: 5px 15px; background-color: #004400;
            border-radius: 10px; border: 1px solid #00ff00;
        """)
        ticker_layout.addWidget(self.status_label)

        self.config_age_label = QLabel("")
        self.config_age_label.setStyleSheet("""
            color: #888; font-weight: bold; font-size: 12px;
            padding: 5px 10px; background-color: #1a1a1a; border-radius: 8px;
        """)
        ticker_layout.addWidget(self.config_age_label)
        self._update_config_age_label()

        self.blacklist_label = QLabel("")
        self.blacklist_label.setStyleSheet("""
            color: #888; font-weight: bold; font-size: 12px;
            padding: 5px 10px; background-color: #1a1a1a; border-radius: 8px;
        """)
        ticker_layout.addWidget(self.blacklist_label)
        self._update_blacklist_label()

        main_layout.addWidget(ticker_widget)

        self.main_tabs = QTabWidget()
        main_layout.addWidget(self.main_tabs)

        # 1. DASHBOARD TAB
        self._setup_dashboard_tab()

        # 2. OPEN TRADES TAB
        self._setup_open_trades_tab()

        # 3. HISTORY TAB
        self._setup_history_tab()

        # 4. POTENTIAL TRADES TAB
        self._setup_potential_tab()

        # 5. BACKTEST TAB
        self._setup_backtest_tab()

        # 6. OPTIMIZATION TAB
        self._setup_optimization_tab()

        # 7. CONFIG TAB
        self._setup_config_tab()

        # Set initial tab
        self.main_tabs.setCurrentIndex(0)

        # START WORKERS
        self.current_params = {}
        self.live_worker = LiveBotWorker(self.current_params, self.tg_token, self.tg_chat_id, self.show_rr_tools)
        self.live_worker.update_ui_signal.connect(self.update_ui)
        self.live_worker.price_signal.connect(self.on_price_update)
        self.live_worker.potential_signal.connect(self.append_potential_trade)
        self.live_worker.status_signal.connect(self.update_bot_status)
        self.live_worker.start()

        self.logs.append(">>> Sistem Başlatıldı. v30.4 (PROFIT ENGINE)")
        self.load_tf_settings("1m")

        self.table_timer = QTimer()
        self.table_timer.timeout.connect(self.refresh_trade_table_from_manager)
        self.table_timer.start(1000)

        self.auto_backtest = None
        self.logs.append(">>> Sistem Başlatıldı. v30.6 (Auto Report - Otomatik Backtest Kapalı)")

        self.load_backtest_meta()
        self.show_saved_backtest_summary()
        self._refresh_config_tab()
        self._update_config_age_label()
        self._update_blacklist_label()

    def _setup_dashboard_tab(self):
        """Setup the dashboard tab with charts and controls."""
        main = _get_main_module()

        live_widget = QWidget()
        live_layout = QVBoxLayout(live_widget)
        top_panel = QHBoxLayout()

        # Symbol selector
        coin_group = QGroupBox("Takip")
        coin_layout = QHBoxLayout()
        self.combo_symbol = QComboBox()
        self.combo_symbol.addItems(main.SYMBOLS)
        self.combo_symbol.currentTextChanged.connect(self.on_symbol_changed)
        self.combo_symbol.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px; color: #00ccff;")
        coin_layout.addWidget(self.combo_symbol)
        coin_group.setLayout(coin_layout)
        top_panel.addWidget(coin_group, stretch=1)

        # Telegram settings
        tg_group = QGroupBox("Telegram")
        tg_layout = QHBoxLayout()
        self.txt_token = QLineEdit(self.tg_token)
        self.txt_token.setEchoMode(QLineEdit.Password)
        self.txt_token.setPlaceholderText("Bot Token")
        tg_layout.addWidget(QLabel("T:"))
        tg_layout.addWidget(self.txt_token)
        self.txt_chatid = QLineEdit(self.tg_chat_id)
        self.txt_chatid.setEchoMode(QLineEdit.Password)
        self.txt_chatid.setPlaceholderText("Chat ID")
        tg_layout.addWidget(QLabel("ID:"))
        tg_layout.addWidget(self.txt_chatid)
        btn_save_tg = QPushButton("Kaydet")
        btn_save_tg.clicked.connect(self.save_config)
        tg_layout.addWidget(btn_save_tg)
        tg_group.setLayout(tg_layout)
        top_panel.addWidget(tg_group, stretch=2)

        # Manual settings
        settings_group = QGroupBox("Manuel Ayar")
        sets_layout = QHBoxLayout()
        self.combo_tf = QComboBox()
        self.combo_tf.addItems(main.TIMEFRAMES)
        self.combo_tf.currentTextChanged.connect(self.load_tf_settings)
        sets_layout.addWidget(QLabel("TF:"))
        sets_layout.addWidget(self.combo_tf)
        self.spin_rr = QDoubleSpinBox()
        self.spin_rr.setRange(0.1, 10.0)
        self.spin_rr.setSingleStep(0.1)
        sets_layout.addWidget(QLabel("RR:"))
        sets_layout.addWidget(self.spin_rr)
        self.spin_rsi = QSpinBox()
        self.spin_rsi.setRange(10, 90)
        self.spin_rsi.setSingleStep(5)
        sets_layout.addWidget(QLabel("RSI:"))
        sets_layout.addWidget(self.spin_rsi)
        self.spin_slope = QDoubleSpinBox()
        self.spin_slope.setRange(0.1, 5.0)
        self.spin_slope.setSingleStep(0.1)
        sets_layout.addWidget(QLabel("Slope:"))
        sets_layout.addWidget(self.spin_slope)
        btn_apply = QPushButton("Güncelle")
        btn_apply.clicked.connect(self.apply_settings)
        sets_layout.addWidget(btn_apply)
        self.chk_rr = QCheckBox("RR")
        self.chk_rr.setChecked(True)
        self.chk_rr.stateChanged.connect(self.toggle_rr)
        sets_layout.addWidget(self.chk_rr)
        self.chk_charts = QCheckBox("Grafik")
        self.chk_charts.setChecked(True)
        self.chk_charts.setToolTip("Grafik güncellemelerini açar/kapatır")
        sets_layout.addWidget(self.chk_charts)
        settings_group.setLayout(sets_layout)
        top_panel.addWidget(settings_group, stretch=4)
        live_layout.addLayout(top_panel)

        # Chart update throttling
        self.last_chart_update = {tf: 0 for tf in main.TIMEFRAMES}
        self.chart_update_interval = 3.0

        # Charts
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

                if main.ENABLE_CHARTS:
                    QWebEngineView = main.get_QWebEngineView()
                    view = QWebEngineView()
                    view.setHtml(main.CHART_TEMPLATE)
                    view.loadFinished.connect(lambda ok, t=tf: self.on_load_finished(ok, t))
                    box_layout.addWidget(view)
                    self.web_views[tf] = view
                else:
                    placeholder = QLabel(f"📊 {tf} - Grafik devre dışı (ENABLE_CHARTS=False)")
                    placeholder.setStyleSheet("color: #666; padding: 20px; font-size: 14px;")
                    placeholder.setAlignment(Qt.AlignCenter)
                    box_layout.addWidget(placeholder)

                grid.addWidget(box, idx // 2, idx % 2)
            return widget

        chart_tabs.addTab(build_chart_grid(main.LOWER_TIMEFRAMES), "LTF")
        chart_tabs.addTab(build_chart_grid(main.HTF_TIMEFRAMES), "HTF")
        live_layout.addWidget(chart_tabs, stretch=6)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumHeight(100)
        live_layout.addWidget(self.logs, stretch=1)
        self.main_tabs.addTab(live_widget, "📡 Dashboard")

    def _setup_open_trades_tab(self):
        """Setup open trades tab."""
        open_trades_widget = QWidget()
        ot_layout = QVBoxLayout(open_trades_widget)
        ot_group = QGroupBox("Açık İşlemler")
        ot_in = QVBoxLayout()
        self.open_trades_table = QTableWidget()
        self.open_trades_table.setColumnCount(12)
        self.open_trades_table.setHorizontalHeaderLabels(
            ["Zaman", "Coin", "TF", "Yön", "Setup", "Giriş", "TP", "SL", "Büyüklük ($)", "PnL", "Durum", "Bilgi"])
        self.open_trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        ot_in.addWidget(self.open_trades_table)
        ot_group.setLayout(ot_in)
        ot_layout.addWidget(ot_group)
        self.main_tabs.addTab(open_trades_widget, "⚡ İşlemler")

    def _setup_history_tab(self):
        """Setup history and assets tab."""
        main = _get_main_module()

        history_widget = QWidget()
        hist_layout = QVBoxLayout(history_widget)

        # Asset panel
        asset_group = QGroupBox("Varlıklarım & Performans")
        asset_group.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 14px; border: 1px solid #444; margin-top: 10px; } "
            "QGroupBox::title { color: #00ccff; }")
        asset_layout = QGridLayout()

        def create_stat_box(title, val_id):
            box = QFrame()
            box.setStyleSheet("background-color: #222; border-radius: 5px; padding: 10px;")
            l = QVBoxLayout(box)
            lbl_t = QLabel(title)
            lbl_t.setStyleSheet("color: #aaa; font-size: 12px;")
            lbl_v = QLabel("$0.00")
            lbl_v.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
            l.addWidget(lbl_t)
            l.addWidget(lbl_v)
            return box, lbl_v

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

        # Portfolio table
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

        # PnL table
        self.pnl_table = self.create_pnl_table()
        hist_layout.addWidget(self.pnl_table)

        # History table
        hist_group = QGroupBox("Geçmiş İşlemler")
        hist_in = QVBoxLayout()
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(11)
        self.history_table.setHorizontalHeaderLabels(
            ["Açılış", "Kapanış", "Coin", "TF", "Yön", "Setup", "Giriş", "Çıkış", "Sonuç", "PnL", "Kasa"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        hist_in.addWidget(self.history_table)

        btn_layout = QHBoxLayout()
        btn_reset_bal = QPushButton("Bakiyeyi Sıfırla ($2000)")
        btn_reset_bal.clicked.connect(self.reset_balances)
        btn_reset_logs = QPushButton("Tüm Geçmişi Temizle")
        btn_reset_logs.clicked.connect(self.reset_logs)
        btn_layout.addWidget(btn_reset_bal)
        btn_layout.addWidget(btn_reset_logs)
        hist_in.addLayout(btn_layout)

        hist_group.setLayout(hist_in)
        hist_layout.addWidget(hist_group)
        self.main_tabs.addTab(history_widget, "📜 Geçmiş & Varlık")

    def _setup_potential_tab(self):
        """Setup potential trades tab."""
        potential_widget = QWidget()
        pot_layout = QVBoxLayout(potential_widget)

        pot_toolbar = QHBoxLayout()
        pot_toolbar.addStretch()
        self.btn_clear_pot = QPushButton("🗑️ Logları Temizle")
        self.btn_clear_pot.setStyleSheet("background-color: #8b0000; color: white; padding: 5px 15px;")
        self.btn_clear_pot.clicked.connect(self.clear_potential_entries)
        pot_toolbar.addWidget(self.btn_clear_pot)
        pot_layout.addLayout(pot_toolbar)

        pot_group = QGroupBox("Potansiyel İşlemler (Kalıcı)")
        pot_inner = QVBoxLayout()
        self.potential_table = QTableWidget()
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

    def _setup_backtest_tab(self):
        """Setup backtest tab."""
        main = _get_main_module()

        backtest_widget = QWidget()
        backtest_layout = QVBoxLayout(backtest_widget)
        bt_cfg = QHBoxLayout()
        bt_cfg.addWidget(QLabel("Semboller:"))
        bt_cfg.addWidget(QLabel(", ".join(main.SYMBOLS)))
        bt_cfg.addWidget(QLabel("TF Seçimi:"))

        self.backtest_tf_checks = {}
        bt_tf_layout = QHBoxLayout()
        for tf in main.TIMEFRAMES:
            cb = QCheckBox(tf)
            cb.setChecked(True)
            bt_tf_layout.addWidget(cb)
            self.backtest_tf_checks[tf] = cb
        bt_cfg.addLayout(bt_tf_layout)

        # Date selection
        date_group = QGroupBox("Veri Aralığı")
        date_layout = QVBoxLayout()

        days_row = QHBoxLayout()
        self.radio_days = QRadioButton("Son X Gün:")
        self.radio_days.setChecked(True)
        days_row.addWidget(self.radio_days)
        self.backtest_days = QSpinBox()
        self.backtest_days.setRange(7, 365)
        self.backtest_days.setValue(30)
        days_row.addWidget(self.backtest_days)
        days_row.addStretch()
        date_layout.addLayout(days_row)

        fixed_row = QHBoxLayout()
        self.radio_fixed_dates = QRadioButton("Sabit Tarih:")
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

        def toggle_date_inputs():
            use_fixed = self.radio_fixed_dates.isChecked()
            self.backtest_days.setEnabled(not use_fixed)
            self.backtest_start_date.setEnabled(use_fixed)
            self.backtest_end_date.setEnabled(use_fixed)

        self.radio_days.toggled.connect(toggle_date_inputs)
        self.radio_fixed_dates.toggled.connect(toggle_date_inputs)
        date_group.setLayout(date_layout)
        bt_cfg.addWidget(date_group)

        speed_layout = QHBoxLayout()
        self.chk_skip_optimization = QCheckBox("⚡ Optimizer Atla")
        speed_layout.addWidget(self.chk_skip_optimization)
        self.chk_quick_mode = QCheckBox("🚀 Hızlı Mod")
        speed_layout.addWidget(self.chk_quick_mode)
        bt_cfg.addLayout(speed_layout)

        self.btn_run_backtest = QPushButton("🧪 Backtest Çalıştır")
        self.btn_run_backtest.clicked.connect(self.start_backtest)
        bt_cfg.addWidget(self.btn_run_backtest)
        backtest_layout.addLayout(bt_cfg)

        self.backtest_logs = QTextEdit()
        self.backtest_logs.setReadOnly(True)
        backtest_layout.addWidget(self.backtest_logs)
        self.main_tabs.addTab(backtest_widget, "🧪 Backtest")

    def _setup_optimization_tab(self):
        """Setup optimization tab."""
        main = _get_main_module()

        opt_widget = QWidget()
        opt_layout = QVBoxLayout(opt_widget)

        # Timeframe selection
        tf_group = QGroupBox("Zaman Dilimleri")
        tf_layout = QHBoxLayout()
        self.opt_tf_checks = {}
        for tf in main.TIMEFRAMES:
            cb = QCheckBox(tf)
            cb.setChecked(True)
            tf_layout.addWidget(cb)
            self.opt_tf_checks[tf] = cb
        tf_group.setLayout(tf_layout)
        opt_layout.addWidget(tf_group)

        btn_test_report = QPushButton("🌙 GÜNLÜK RAPORU ŞİMDİ OLUŞTUR (TEST)")
        btn_test_report.setStyleSheet("background-color: #444; color: #aaa; margin-top: 10px;")
        btn_test_report.clicked.connect(self.force_daily_report)
        opt_layout.addWidget(btn_test_report)

        # Parameter ranges
        grid_group = QGroupBox("Parametre Aralıkları")
        grid_layout = QHBoxLayout()

        grid_layout.addWidget(QLabel("RR:"))
        self.opt_rr_start = QDoubleSpinBox()
        self.opt_rr_start.setValue(1.5)
        self.opt_rr_end = QDoubleSpinBox()
        self.opt_rr_end.setValue(3.0)
        self.opt_rr_step = QDoubleSpinBox()
        self.opt_rr_step.setValue(0.5)
        grid_layout.addWidget(self.opt_rr_start)
        grid_layout.addWidget(self.opt_rr_end)
        grid_layout.addWidget(self.opt_rr_step)

        grid_layout.addWidget(QLabel("RSI:"))
        self.opt_rsi_start = QSpinBox()
        self.opt_rsi_start.setValue(30)
        self.opt_rsi_end = QSpinBox()
        self.opt_rsi_end.setValue(60)
        self.opt_rsi_step = QSpinBox()
        self.opt_rsi_step.setValue(10)
        grid_layout.addWidget(self.opt_rsi_start)
        grid_layout.addWidget(self.opt_rsi_end)
        grid_layout.addWidget(self.opt_rsi_step)

        grid_layout.addWidget(QLabel("Slope:"))
        self.opt_slope_start = QDoubleSpinBox()
        self.opt_slope_start.setValue(0.3)
        self.opt_slope_end = QDoubleSpinBox()
        self.opt_slope_end.setValue(0.9)
        self.opt_slope_step = QDoubleSpinBox()
        self.opt_slope_step.setValue(0.2)
        grid_layout.addWidget(self.opt_slope_start)
        grid_layout.addWidget(self.opt_slope_end)
        grid_layout.addWidget(self.opt_slope_step)

        grid_group.setLayout(grid_layout)
        opt_layout.addWidget(grid_group)

        # Bottom panel
        candles_layout = QHBoxLayout()
        candles_layout.addWidget(QLabel("Coin:"))
        self.combo_opt_symbol = QComboBox()
        self.combo_opt_symbol.addItems(main.SYMBOLS)
        candles_layout.addWidget(self.combo_opt_symbol)
        candles_layout.addWidget(QLabel("Gün Sayısı:"))
        self.opt_days = QSpinBox()
        self.opt_days.setRange(7, 180)
        self.opt_days.setValue(30)
        candles_layout.addWidget(self.opt_days)
        self.chk_monte_carlo = QCheckBox("🎲 Monte Carlo Testi")
        self.chk_monte_carlo.setStyleSheet("color: #ff9900; font-weight: bold;")
        candles_layout.addWidget(self.chk_monte_carlo)

        self.btn_run_opt = QPushButton("🚀 TAM TARAMA BAŞLAT")
        self.btn_run_opt.clicked.connect(self.run_optimization)
        candles_layout.addWidget(self.btn_run_opt)
        opt_layout.addLayout(candles_layout)

        self.opt_logs = QTextEdit()
        self.opt_logs.setReadOnly(True)
        opt_layout.addWidget(self.opt_logs)
        self.main_tabs.addTab(opt_widget, "🔧 Optimizasyon")

    def _setup_config_tab(self):
        """Setup config tab."""
        main = _get_main_module()

        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setSpacing(10)

        config_header = QHBoxLayout()

        # Config status
        status_group = QGroupBox("📋 Config Durumu")
        status_group.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        status_layout = QVBoxLayout(status_group)
        self.config_status_text = QTextEdit()
        self.config_status_text.setReadOnly(True)
        self.config_status_text.setMaximumHeight(150)
        self.config_status_text.setStyleSheet("""
            QTextEdit { background-color: #1a1a1a; color: #00ff00;
            font-family: Consolas, monospace; font-size: 12px; border: 1px solid #333; }
        """)
        status_layout.addWidget(self.config_status_text)
        self.btn_refresh_config = QPushButton("🔄 Yenile")
        self.btn_refresh_config.clicked.connect(self._refresh_config_tab)
        status_layout.addWidget(self.btn_refresh_config)
        config_header.addWidget(status_group)

        # Stats
        stats_group = QGroupBox("📊 İstatistikler")
        stats_group.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        stats_layout = QVBoxLayout(stats_group)
        self.config_stats_text = QTextEdit()
        self.config_stats_text.setReadOnly(True)
        self.config_stats_text.setMaximumHeight(150)
        self.config_stats_text.setStyleSheet("""
            QTextEdit { background-color: #1a1a1a; color: #aaaaff;
            font-family: Consolas, monospace; font-size: 12px; border: 1px solid #333; }
        """)
        stats_layout.addWidget(self.config_stats_text)
        self.btn_delete_config = QPushButton("🗑️ Config Dosyasını Sil")
        self.btn_delete_config.setStyleSheet("QPushButton { background-color: #662222; } QPushButton:hover { background-color: #883333; }")
        self.btn_delete_config.clicked.connect(self._delete_config_file)
        stats_layout.addWidget(self.btn_delete_config)
        config_header.addWidget(stats_group)
        config_layout.addLayout(config_header)

        # Active streams table
        active_group = QGroupBox("✅ Aktif Streamler")
        active_group.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        active_layout = QVBoxLayout(active_group)
        self.active_streams_table = QTableWidget()
        self.active_streams_table.setColumnCount(7)
        self.active_streams_table.setHorizontalHeaderLabels(["Symbol", "TF", "RR", "RSI", "AT", "Trailing", "Strateji"])
        self.active_streams_table.horizontalHeader().setStretchLastSection(True)
        self.active_streams_table.setStyleSheet("""
            QTableWidget { background-color: #1a1a1a; color: white; gridline-color: #333; }
            QHeaderView::section { background-color: #114411; color: white; font-weight: bold; padding: 5px; }
        """)
        active_layout.addWidget(self.active_streams_table)
        config_layout.addWidget(active_group)

        # Disabled streams table
        disabled_group = QGroupBox("🚫 Devre Dışı Streamler")
        disabled_group.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        disabled_layout = QVBoxLayout(disabled_group)
        self.disabled_streams_table = QTableWidget()
        self.disabled_streams_table.setColumnCount(4)
        self.disabled_streams_table.setHorizontalHeaderLabels(["Symbol", "TF", "Sebep", "PnL"])
        self.disabled_streams_table.horizontalHeader().setStretchLastSection(True)
        self.disabled_streams_table.setStyleSheet("""
            QTableWidget { background-color: #1a1a1a; color: white; gridline-color: #333; }
            QHeaderView::section { background-color: #441111; color: white; font-weight: bold; padding: 5px; }
        """)
        disabled_layout.addWidget(self.disabled_streams_table)
        config_layout.addWidget(disabled_group)

        self.main_tabs.addTab(config_widget, "⚙️ Config")

    # --- EVENT HANDLERS ---

    def on_load_finished(self, ok, tf):
        if ok:
            self.views_ready[tf] = True

    def on_symbol_changed(self, text):
        main = _get_main_module()
        self.current_symbol = text
        self.logs.clear()
        self.logs.append(f">>> {text} Seçildi")
        self.load_tf_settings(self.combo_tf.currentText())
        if text in self.data_cache:
            for tf in main.TIMEFRAMES:
                cached_data = self.data_cache[text][tf]
                if cached_data[0] is not None:
                    self.render_chart_and_log(tf, cached_data[0], cached_data[1])

    def update_ui(self, symbol, tf, json_data, log_msg):
        self.data_cache[symbol][tf] = (json_data, log_msg)
        if symbol == self.current_symbol:
            self.render_chart_and_log(tf, json_data, log_msg)

    def update_bot_status(self, status_text):
        self.status_label.setText(status_text)
        if "TRADE ARIYOR" in status_text:
            self.status_label.setStyleSheet("""
                color: #00ff00; font-weight: bold; font-size: 14px;
                padding: 5px 15px; background-color: #004400;
                border-radius: 10px; border: 1px solid #00ff00;
            """)
        elif "AÇIK POZİSYON" in status_text:
            self.status_label.setStyleSheet("""
                color: #ffcc00; font-weight: bold; font-size: 14px;
                padding: 5px 15px; background-color: #443300;
                border-radius: 10px; border: 1px solid #ffcc00;
            """)

    def render_chart_and_log(self, tf, json_data, log_msg):
        main = _get_main_module()
        charts_enabled = getattr(self, 'chk_charts', None) and self.chk_charts.isChecked()
        if main.ENABLE_CHARTS and charts_enabled and self.views_ready.get(tf, False) and json_data and json_data != "{}":
            now = time.time()
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
            self.logs.append(fmt)
            self.logs.verticalScrollBar().setValue(self.logs.verticalScrollBar().maximum())

    def on_price_update(self, symbol, price):
        if symbol in self.price_labels:
            self.price_labels[symbol].setText(f"${price:.2f}")
            self.price_labels[symbol].setStyleSheet("color: #00ff00; font-weight: bold; font-size: 16px;")

    # --- CONFIG METHODS ---

    def save_config(self):
        main = _get_main_module()
        self.tg_token = self.txt_token.text().strip()
        self.tg_chat_id = self.txt_chatid.text().strip()
        with open(main.CONFIG_FILE, 'w') as f:
            json.dump({"telegram_token": self.tg_token, "telegram_chat_id": self.tg_chat_id}, f)
        self.live_worker.update_telegram_creds(self.tg_token, self.tg_chat_id)
        QMessageBox.information(self, "Başarılı", "Kaydedildi.")

    def load_config(self):
        main = _get_main_module()
        if os.path.exists(main.CONFIG_FILE):
            try:
                with open(main.CONFIG_FILE, 'r') as f:
                    c = json.load(f)
                    self.tg_token = c.get("telegram_token", "")
                    self.tg_chat_id = c.get("telegram_chat_id", "")
            except Exception as e:
                print(f"Config load error: {e}")

    def load_tf_settings(self, tf):
        current_sym = self.combo_symbol.currentText()
        settings = SYMBOL_PARAMS.get(current_sym, SYMBOL_PARAMS.get("BTCUSDT", {}))
        p = settings.get(tf, settings.get("1m", {}))
        if p:
            self.spin_rr.setValue(p.get('rr', 2.0))
            self.spin_rsi.setValue(p.get('rsi', 40))
            self.spin_slope.setValue(p.get('slope', 0.5))

    def apply_settings(self):
        current_sym = self.combo_symbol.currentText()
        tf = self.combo_tf.currentText()
        rr = self.spin_rr.value()
        rsi = self.spin_rsi.value()
        slope = self.spin_slope.value()
        self.live_worker.update_settings(current_sym, tf, rr, rsi, slope)
        self.logs.append(f">>> {current_sym} - {tf} Ayarı Güncellendi.")

    def toggle_rr(self):
        self.live_worker.update_show_rr(self.chk_rr.isChecked())

    def reset_logs(self):
        main = _get_main_module()
        if QMessageBox.question(self, 'Onay', "Sil?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            main.trade_manager.reset_logs()
            self.refresh_trade_table_from_manager()

    def reset_balances(self):
        main = _get_main_module()
        if QMessageBox.question(self, 'Onay', "Sıfırla?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            main.trade_manager.reset_balances()
            self.refresh_trade_table_from_manager()

    # --- POTENTIAL TRADES ---

    def _fmt_bool(self, value):
        return "✓" if bool(value) else "×"

    def _load_potential_entries(self) -> list:
        main = _get_main_module()
        try:
            if os.path.exists(main.POT_LOG_FILE):
                with open(main.POT_LOG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data[-400:]
        except Exception as e:
            print(f"[POT] Failed to load: {e}")
        return []

    def _save_potential_entries(self):
        main = _get_main_module()
        try:
            with open(main.POT_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.potential_entries[-400:], f, indent=2, default=str)
        except Exception as e:
            print(f"[POT] Failed to save: {e}")

    def clear_potential_entries(self):
        self.potential_entries = []
        self._save_potential_entries()
        self.refresh_potential_table()
        self.logs.append("[POT] Potansiyel işlem logları temizlendi.")

    def append_potential_trade(self, entry: dict):
        if not isinstance(entry, dict):
            return
        self.potential_entries.append(entry)
        if len(self.potential_entries) > 400:
            self.potential_entries = self.potential_entries[-400:]
        self._save_potential_entries()
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

            if str(entry.get("decision", "")).startswith("Accepted"):
                for col in range(len(values)):
                    existing_item = self.potential_table.item(row_idx, col)
                    if existing_item:
                        existing_item.setBackground(QColor(0, 60, 30))
                        existing_item.setForeground(QColor("#e8ffe8"))

    # --- TRADE TABLE ---

    def refresh_trade_table_from_manager(self):
        main = _get_main_module()
        try:
            open_trades = list(main.trade_manager.open_trades)
            wallet_bal = main.trade_manager.wallet_balance
            locked = main.trade_manager.locked_margin
            open_pnl = sum(float(t.get("pnl", 0)) for t in open_trades)
            total_equity = wallet_bal + locked + open_pnl
            total_pnl = main.trade_manager.total_pnl

            today_str = datetime.now().strftime("%Y-%m-%d")
            daily_pnl = 0.0
            for t in main.trade_manager.history:
                if t.get("close_time", "").startswith(today_str):
                    daily_pnl += float(t["pnl"])

            if hasattr(self, 'lbl_equity_val'):
                self.lbl_equity_val.setText(f"${total_equity:,.2f}")
                self.lbl_avail_val.setText(f"${wallet_bal:,.2f}")
                self.lbl_total_pnl_val.setText(f"${total_pnl:,.2f}")
                color = "#00ff00" if total_pnl > 0 else ("#ff5555" if total_pnl < 0 else "white")
                self.lbl_total_pnl_val.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
                self.lbl_daily_pnl_val.setText(f"${daily_pnl:,.2f}")
                color = "#00ff00" if daily_pnl > 0 else ("#ff5555" if daily_pnl < 0 else "white")
                self.lbl_daily_pnl_val.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")

            open_trades.sort(key=lambda x: x['id'], reverse=True)
            self.open_trades_table.setRowCount(len(open_trades))
            cols_open = ["timestamp", "symbol", "timeframe", "type", "setup", "entry", "tp", "sl", "size", "pnl", "status", "info"]

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
                        item.setBackground(QColor(0, 100, 0) if pnl > 0 else QColor(100, 0, 0))
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

            self.update_portfolio_table(open_trades)

            hist_trades = list(main.trade_manager.history)
            hist_trades.sort(key=lambda x: x['close_time'], reverse=True)
            self.history_table.setRowCount(len(hist_trades))
            cols_hist = ["timestamp", "close_time", "symbol", "timeframe", "type", "setup", "entry", "close_price", "status", "pnl", "has_cash"]

            for row_idx, trade in enumerate(hist_trades):
                for col_idx, col_key in enumerate(cols_hist):
                    val = trade.get(col_key, "")
                    item = QTableWidgetItem(str(val))

                    if col_key == "status":
                        pnl_val = float(trade.get("pnl", 0))
                        if "STOP" in str(val):
                            if pnl_val >= -0.5:
                                item.setText("BE (Başabaş)")
                                item.setBackground(QColor(50, 50, 0))
                                item.setForeground(QColor("yellow"))
                            else:
                                item.setBackground(QColor(50, 0, 0))
                        elif "WIN" in str(val):
                            item.setBackground(QColor(0, 50, 0))
                    elif col_key == "pnl":
                        pnl = float(val)
                        item.setText(f"${pnl:.2f}")
                        item.setForeground(QColor("#00ff00") if pnl > 0 else QColor("#ff5555"))
                    elif col_key in ["entry", "close_price"]:
                        item.setText(f"{float(val):.4f}")

                    self.history_table.setItem(row_idx, col_idx, item)

            self.update_pnl_table_data()

        except Exception as e:
            print(f"KRİTİK HATA: {e}")
            main = _get_main_module()
            with open(os.path.join(main.DATA_DIR, "error_log.txt"), "a") as f:
                f.write(f"\n[{datetime.now()}] HATA: {str(e)}\n")
                f.write(traceback.format_exc())

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
        main = _get_main_module()
        selected = [tf for tf, cb in (checkbox_map or {}).items() if cb.isChecked()]
        return selected if selected else list(main.TIMEFRAMES)

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
                    item.setForeground(QColor("#00ff00") if pnl_val > 0 else QColor("#ff5555") if pnl_val < 0 else QColor("white"))
                elif key == "margin":
                    item.setForeground(QColor("#00ccff"))
                self.portfolio_table.setItem(row_idx, col_idx, item)

    def create_pnl_table(self):
        main = _get_main_module()
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Zaman Dilimi", "Başarı %", "PnL ($)"])
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        table.setRowCount(len(main.TIMEFRAMES))
        for i, tf in enumerate(main.TIMEFRAMES):
            table.setItem(i, 0, QTableWidgetItem(tf))
            table.setItem(i, 1, QTableWidgetItem("%0.0"))
            table.setItem(i, 2, QTableWidgetItem("$0.00"))
        return table

    def update_pnl_table_data(self):
        main = _get_main_module()
        stats = {}
        for trade in main.trade_manager.history:
            tf = trade.get('timeframe', 'Bilinmiyor')
            pnl = float(trade.get('pnl', 0))
            if tf not in stats:
                stats[tf] = {'wins': 0, 'count': 0, 'total_pnl': 0.0}
            stats[tf]['count'] += 1
            stats[tf]['total_pnl'] += pnl
            if pnl > 0:
                stats[tf]['wins'] += 1

        self.pnl_table.setRowCount(0)
        sirali_tf = list(main.TIMEFRAMES)
        mevcut_keys = list(stats.keys())
        final_list = [t for t in sirali_tf if t in mevcut_keys] + [t for t in mevcut_keys if t not in sirali_tf]

        row = 0
        for tf in final_list:
            data = stats[tf]
            self.pnl_table.insertRow(row)
            count = data['count']
            wins = data['wins']
            pnl = data['total_pnl']
            win_rate = (wins / count * 100) if count > 0 else 0

            self.pnl_table.setItem(row, 0, QTableWidgetItem(str(tf)))
            self.pnl_table.setItem(row, 1, QTableWidgetItem(f"%{win_rate:.1f} ({wins}/{count})"))
            pnl_item = QTableWidgetItem(f"${pnl:.2f}")
            pnl_item.setForeground(QColor("#00ff00") if pnl >= 0 else QColor("#ff5555"))
            self.pnl_table.setItem(row, 2, pnl_item)
            row += 1

    # --- BACKTEST ---

    def start_backtest(self):
        main = _get_main_module()
        if getattr(self, "backtest_worker", None) and self.backtest_worker.isRunning():
            QMessageBox.information(self, "Devam Ediyor", "Backtest zaten çalışıyor...")
            return

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
        use_fixed_dates = self.radio_fixed_dates.isChecked()

        if use_fixed_dates:
            start_date = self.backtest_start_date.date().toString("yyyy-MM-dd")
            end_date = self.backtest_end_date.date().toString("yyyy-MM-dd")
            self.backtest_logs.append(f"🧪 Backtest başlatıldı (📅 {start_date} → {end_date})")
        else:
            days = self.backtest_days.value()
            end_date = date.today().strftime("%Y-%m-%d")
            start_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
            self.backtest_logs.append(f"🧪 Backtest başlatıldı ({days} gün: {start_date} → {end_date})")

        skip_opt = self.chk_skip_optimization.isChecked()
        quick = self.chk_quick_mode.isChecked()
        if skip_opt:
            self.backtest_logs.append("⚡ Optimizer atlanıyor")
        elif quick:
            self.backtest_logs.append("🚀 Hızlı mod aktif")

        self.backtest_worker = BacktestWorker(
            main.SYMBOLS, selected_tfs, 0, skip_opt, quick,
            use_days=False, start_date=start_date, end_date=end_date
        )
        self.backtest_worker.log_signal.connect(self.append_backtest_log)
        self.backtest_worker.finished_signal.connect(self.on_backtest_finished)
        self.btn_run_backtest.setEnabled(False)
        self.backtest_worker.start()

    def append_backtest_log(self, text):
        self.backtest_logs.append(text)

    def on_backtest_finished(self, result: dict):
        main = _get_main_module()
        self.btn_run_backtest.setEnabled(True)
        best_configs = result.get("best_configs", {}) if isinstance(result, dict) else {}
        summary_rows = result.get("summary", []) if isinstance(result, dict) else []

        if summary_rows:
            finished_at = utcnow().isoformat() + "Z"
            meta = {
                "finished_at": finished_at,
                "summary": summary_rows,
                "summary_csv": result.get("summary_csv"),
                "strategy_signature": result.get("strategy_signature") or main._strategy_signature(),
            }
            self.save_backtest_meta(meta)
            self.backtest_logs.append("📊 Özet tablo kaydedildi:")
            for line in self.format_backtest_summary_lines(meta):
                self.backtest_logs.append(line)
        else:
            self.backtest_logs.append("⚠️ Backtest sonucu bulunamadı.")

        if best_configs:
            main.save_best_configs(best_configs)
            self.backtest_logs.append("✅ En iyi ayarlar canlı trade'e aktarıldı.")
            self._update_config_age_label()
            self._update_blacklist_label()
            self._refresh_config_tab()

    def save_backtest_meta(self, meta: dict):
        main = _get_main_module()
        try:
            with open(main.BACKTEST_META_FILE, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            self.backtest_meta = meta
        except Exception as e:
            self.backtest_logs.append(f"⚠️ Meta kayıt hatası: {e}")

    def load_backtest_meta(self):
        main = _get_main_module()
        try:
            if os.path.exists(main.BACKTEST_META_FILE):
                with open(main.BACKTEST_META_FILE, "r", encoding="utf-8") as f:
                    self.backtest_meta = json.load(f)
            else:
                self.backtest_meta = None
        except Exception as e:
            self.backtest_meta = None

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
            self.backtest_logs.append("ℹ️ Backtest geçmişi henüz yok.")

    # --- OPTIMIZATION ---

    def run_optimization(self):
        main = _get_main_module()
        days = self.opt_days.value()
        rr_range = (self.opt_rr_start.value(), self.opt_rr_end.value(), self.opt_rr_step.value())
        rsi_range = (self.opt_rsi_start.value(), self.opt_rsi_end.value(), self.opt_rsi_step.value())
        slope_range = (self.opt_slope_start.value(), self.opt_slope_end.value(), self.opt_slope_step.value())
        is_monte_carlo = self.chk_monte_carlo.isChecked()
        use_at = False
        selected_tfs = self.get_selected_timeframes(getattr(self, "opt_tf_checks", {}))
        selected_sym = self.combo_opt_symbol.currentText()
        self.opt_logs.clear()
        self.btn_run_opt.setEnabled(False)
        self.opt_worker = OptimizerWorker(selected_sym, days, rr_range, rsi_range, slope_range, use_at,
                                          is_monte_carlo, selected_tfs, use_days=True)
        self.opt_worker.result_signal.connect(self.on_opt_update)
        self.opt_worker.start()

    def on_opt_update(self, msg):
        self.opt_logs.append(msg)
        if "Tamamlandı" in msg:
            self.btn_run_opt.setEnabled(True)

    def force_daily_report(self):
        self.opt_logs.append("🌙 Otomatik rapor ve gece backtest özelliği şu an kapalı.")

    # --- CONFIG TAB METHODS ---

    def _update_config_age_label(self):
        main = _get_main_module()
        try:
            if not os.path.exists(main.BEST_CONFIGS_FILE):
                self.config_age_label.setText("⚠️ Config yok")
                self.config_age_label.setStyleSheet("""
                    color: #ff6600; font-weight: bold; font-size: 12px;
                    padding: 5px 10px; background-color: #332200;
                    border-radius: 8px; border: 1px solid #ff6600;
                """)
                return

            with open(main.BEST_CONFIGS_FILE, 'r', encoding='utf-8') as f:
                best_cfgs = json.load(f)

            if not isinstance(best_cfgs, dict):
                return

            meta = best_cfgs.get("_meta", {})
            saved_at_str = meta.get("saved_at", "")

            if not saved_at_str:
                self.config_age_label.setText("⚠️ Config tarihi yok")
                return

            saved_at = datetime.fromisoformat(saved_at_str.replace("Z", "+00:00"))
            saved_at_naive = saved_at.replace(tzinfo=None)
            now_naive = utcnow()
            age = now_naive - saved_at_naive
            age_days = age.days
            age_hours = age.seconds // 3600

            if age_days == 0:
                age_text = f"{age_hours}s önce"
            elif age_days == 1:
                age_text = "1 gün önce"
            else:
                age_text = f"{age_days} gün önce"

            if age_days >= 14:
                self.config_age_label.setText(f"🔴 Config: {age_text}")
                self.config_age_label.setStyleSheet("""
                    color: #ff4444; font-weight: bold; font-size: 12px;
                    padding: 5px 10px; background-color: #441111;
                    border-radius: 8px; border: 1px solid #ff4444;
                """)
            elif age_days >= 7:
                self.config_age_label.setText(f"🟡 Config: {age_text}")
                self.config_age_label.setStyleSheet("""
                    color: #ffaa00; font-weight: bold; font-size: 12px;
                    padding: 5px 10px; background-color: #332200;
                    border-radius: 8px; border: 1px solid #ffaa00;
                """)
            else:
                self.config_age_label.setText(f"🟢 Config: {age_text}")
                self.config_age_label.setStyleSheet("""
                    color: #44ff44; font-weight: bold; font-size: 12px;
                    padding: 5px 10px; background-color: #114411;
                    border-radius: 8px; border: 1px solid #44ff44;
                """)

        except Exception as e:
            self.config_age_label.setText("⚠️ Config okunamadı")
            print(f"[UI] Config age error: {e}")

    def _update_blacklist_label(self):
        main = _get_main_module()
        try:
            dynamic_bl = main.load_dynamic_blacklist()
            static_bl = main.POST_PORTFOLIO_BLACKLIST

            disabled_streams = []
            if os.path.exists(main.BEST_CONFIGS_FILE):
                try:
                    with open(main.BEST_CONFIGS_FILE, 'r', encoding='utf-8') as f:
                        best_cfgs = json.load(f)
                    if isinstance(best_cfgs, dict):
                        for sym in main.SYMBOLS:
                            sym_cfg = best_cfgs.get(sym, {})
                            if isinstance(sym_cfg, dict):
                                for tf in main.TIMEFRAMES:
                                    tf_cfg = sym_cfg.get(tf, {})
                                    if isinstance(tf_cfg, dict) and tf_cfg.get("disabled", False):
                                        disabled_streams.append((sym, tf))
                except json.JSONDecodeError:
                    pass

            total_streams = len(main.SYMBOLS) * len(main.TIMEFRAMES)
            blacklisted_count = len(dynamic_bl) + len([k for k in static_bl.keys() if static_bl.get(k)])
            disabled_count = len(disabled_streams)
            active_count = total_streams - blacklisted_count - disabled_count

            if blacklisted_count + disabled_count == 0:
                self.blacklist_label.setText(f"✅ {active_count}/{total_streams} aktif")
                self.blacklist_label.setStyleSheet("""
                    color: #44ff44; font-weight: bold; font-size: 12px;
                    padding: 5px 10px; background-color: #114411;
                    border-radius: 8px; border: 1px solid #44ff44;
                """)
            elif active_count > total_streams * 0.5:
                self.blacklist_label.setText(f"📊 {active_count}/{total_streams} aktif")
                self.blacklist_label.setStyleSheet("""
                    color: #aaaaff; font-weight: bold; font-size: 12px;
                    padding: 5px 10px; background-color: #222244;
                    border-radius: 8px; border: 1px solid #6666aa;
                """)
            else:
                self.blacklist_label.setText(f"⚠️ {active_count}/{total_streams} aktif")
                self.blacklist_label.setStyleSheet("""
                    color: #ffaa00; font-weight: bold; font-size: 12px;
                    padding: 5px 10px; background-color: #332200;
                    border-radius: 8px; border: 1px solid #ffaa00;
                """)

        except Exception as e:
            self.blacklist_label.setText("⚠️ Blacklist?")
            print(f"[UI] Blacklist error: {e}")

    def _refresh_config_tab(self):
        main = _get_main_module()
        try:
            status_lines = []

            if not os.path.exists(main.BEST_CONFIGS_FILE):
                status_lines.append("❌ Config dosyası bulunamadı")
                status_lines.append(f"   Dosya: {main.BEST_CONFIGS_FILE}")
                status_lines.append("")
                status_lines.append("💡 Backtest çalıştırarak config oluşturun.")
                self.config_status_text.setPlainText("\n".join(status_lines))
                self.config_stats_text.setPlainText("Config yok - istatistik hesaplanamadı")
                self.active_streams_table.setRowCount(0)
                self.disabled_streams_table.setRowCount(0)
                return

            try:
                with open(main.BEST_CONFIGS_FILE, 'r', encoding='utf-8') as f:
                    best_cfgs = json.load(f)
            except json.JSONDecodeError as e:
                status_lines.append("❌ Config dosyası bozuk (JSON hatası)")
                status_lines.append(f"   Hata: {e}")
                self.config_status_text.setPlainText("\n".join(status_lines))
                return

            if not isinstance(best_cfgs, dict):
                status_lines.append("❌ Config formatı geçersiz")
                self.config_status_text.setPlainText("\n".join(status_lines))
                return

            meta = best_cfgs.get("_meta", {})
            saved_at_str = meta.get("saved_at", "Bilinmiyor")
            signature = meta.get("strategy_signature", "Yok")

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
                except:
                    pass

            current_sig = main._strategy_signature()
            sig_match = signature == current_sig

            status_lines.append(f"✅ Config dosyası mevcut")
            status_lines.append(f"   Dosya: {main.BEST_CONFIGS_FILE}")
            status_lines.append(f"   Kayıt: {saved_at_str}")
            status_lines.append(f"   Yaş: {age_text}")
            status_lines.append(f"   İmza: {signature[:12]}...")
            if sig_match:
                status_lines.append(f"   ✅ İmza eşleşiyor (güncel)")
            else:
                status_lines.append(f"   ⚠️ İmza eşleşmiyor! Backtest çalıştırın.")
                status_lines.append(f"      Beklenen: {current_sig[:12]}...")

            self.config_status_text.setPlainText("\n".join(status_lines))

            dynamic_bl = main.load_dynamic_blacklist()
            static_bl = main.POST_PORTFOLIO_BLACKLIST

            active_streams = []
            disabled_streams = []

            for sym in main.SYMBOLS:
                sym_cfg = best_cfgs.get(sym, {})
                if not isinstance(sym_cfg, dict):
                    continue
                for tf in main.TIMEFRAMES:
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

            total_streams = len(main.SYMBOLS) * len(main.TIMEFRAMES)
            stats_lines = [
                f"📊 Toplam stream sayısı: {total_streams}",
                f"✅ Aktif streamler: {len(active_streams)}",
                f"🚫 Devre dışı: {len(disabled_streams)}",
                f"   - Dinamik blacklist: {len(dynamic_bl)}",
                f"   - Statik blacklist: {len([k for k,v in static_bl.items() if v])}",
                f"   - Config disabled: {len([d for d in disabled_streams if 'disabled' in d['reason']])}",
            ]
            self.config_stats_text.setPlainText("\n".join(stats_lines))

            self.active_streams_table.setRowCount(len(active_streams))
            for i, stream in enumerate(active_streams):
                self.active_streams_table.setItem(i, 0, QTableWidgetItem(stream["sym"]))
                self.active_streams_table.setItem(i, 1, QTableWidgetItem(stream["tf"]))
                self.active_streams_table.setItem(i, 2, QTableWidgetItem(str(stream["rr"])))
                self.active_streams_table.setItem(i, 3, QTableWidgetItem(str(stream["rsi"])))
                self.active_streams_table.setItem(i, 4, QTableWidgetItem(stream["at"]))
                self.active_streams_table.setItem(i, 5, QTableWidgetItem(stream["trailing"]))
                self.active_streams_table.setItem(i, 6, QTableWidgetItem(stream["strategy"]))

            self.disabled_streams_table.setRowCount(len(disabled_streams))
            for i, stream in enumerate(disabled_streams):
                self.disabled_streams_table.setItem(i, 0, QTableWidgetItem(stream["sym"]))
                self.disabled_streams_table.setItem(i, 1, QTableWidgetItem(stream["tf"]))
                self.disabled_streams_table.setItem(i, 2, QTableWidgetItem(stream["reason"]))
                pnl_text = f"${stream['pnl']:.0f}" if stream['pnl'] != 0 else "-"
                self.disabled_streams_table.setItem(i, 3, QTableWidgetItem(pnl_text))

            self._update_config_age_label()
            self._update_blacklist_label()

        except Exception as e:
            self.config_status_text.setPlainText(f"❌ Hata: {e}")
            print(f"[UI] Config tab error: {e}")

    def _delete_config_file(self):
        main = _get_main_module()
        try:
            if os.path.exists(main.BEST_CONFIGS_FILE):
                os.remove(main.BEST_CONFIGS_FILE)
                print(f"[CFG] ✓ Config dosyası silindi: {main.BEST_CONFIGS_FILE}")
                main.BEST_CONFIG_CACHE.clear()
                main.BEST_CONFIG_WARNING_FLAGS["missing_signature"] = False
                main.BEST_CONFIG_WARNING_FLAGS["signature_mismatch"] = False
                main.BEST_CONFIG_WARNING_FLAGS["json_error"] = False
                main.BEST_CONFIG_WARNING_FLAGS["load_error"] = False
                self._refresh_config_tab()
                self._update_config_age_label()
                self._update_blacklist_label()
            else:
                print("[CFG] Config dosyası zaten yok.")
        except Exception as e:
            print(f"[CFG] ⚠️ Config silme hatası: {e}")

    def append_log(self, text):
        """Helper to append text to the main logs."""
        if hasattr(self, 'logs'):
            self.logs.append(text)
