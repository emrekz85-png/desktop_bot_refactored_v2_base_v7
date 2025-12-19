# ==========================================
# UI Workers - QThread-based Background Workers
# ==========================================
# This module contains QThread workers for background operations:
# - LiveBotWorker: Real-time live trading loop
# - OptimizerWorker: Parameter grid optimization
# - BacktestWorker: Portfolio backtesting
# - AutoBacktestWorker: Scheduled automatic backtesting
# ==========================================

import os
import sys
import time
import json
import traceback
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import itertools

# Core imports
from core import (
    TradingEngine,
    DEFAULT_STRATEGY_CONFIG,
    SYMBOL_PARAMS,
    TRADING_CONFIG,
)

# Import from main module for backward compatibility
# These will be imported lazily to avoid circular imports
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
        from PyQt5.QtCore import QThread, pyqtSignal
        HAS_GUI = True
    except ImportError:
        HAS_GUI = False
        # Placeholder classes
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
else:
    HAS_GUI = False
    # Placeholder classes for headless mode
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


class LiveBotWorker(QThread):
    """Live trading worker that monitors WebSocket data and generates signals."""

    update_ui_signal = pyqtSignal(str, str, str, str)
    trade_signal = pyqtSignal(dict)
    price_signal = pyqtSignal(str, float)
    potential_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)

    def __init__(self, current_params, tg_token, tg_chat_id, show_rr):
        super().__init__()
        main = _get_main_module()
        self.is_running = True
        self.tg_token = tg_token
        self.tg_chat_id = tg_chat_id
        self.show_rr = show_rr
        self.last_signals = {sym: {tf: None for tf in main.TIMEFRAMES} for sym in main.SYMBOLS}
        self.last_potential = {sym: {tf: None for tf in main.TIMEFRAMES} for sym in main.SYMBOLS}
        self.ws_stream = main.BinanceWebSocketKlineStream(main.SYMBOLS, main.TIMEFRAMES, max_candles=1200)
        self._startup_warmup_done = False

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
        self.tg_token = t
        self.tg_chat_id = c

    def stop(self):
        self.is_running = False
        self.ws_stream.stop()

    def run(self):
        import threading
        main = _get_main_module()

        self.ws_stream.start()
        next_price_time = 0
        next_candle_time = 0
        next_status_time = 0

        try:
            while self.is_running:
                now = time.time()
                stream_snapshot = None

                # Status update every 2 seconds
                if now >= next_status_time:
                    open_count = len(main.trade_manager.open_trades)
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
                        latest_prices = TradingEngine.get_latest_prices(main.SYMBOLS)
                        for sym, price in latest_prices.items():
                            self.price_signal.emit(sym, price)
                            main.trade_manager.update_live_pnl_with_price(sym, price)
                            rt_closed = main.trade_manager.check_realtime_sl(sym, price)
                            for ct in rt_closed:
                                tf = ct.get('timeframe', '?')
                                pnl = float(ct.get('pnl', 0))
                                pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
                                close_log = f"🚨 {ct['symbol']} ACİL KAPATILDI ({tf}): {ct['status']} | {pnl_str}"
                                self.update_ui_signal.emit(sym, tf, "{}", f"⚠️ {close_log}")
                                tg_msg = f"🚨 ACİL KAPATMA: {ct['symbol']}\nTF: {tf}\nSonuç: {ct['status']}\nNet PnL: {pnl_str}\n⚠️ Real-time SL tetiklendi"
                                TradingEngine.send_telegram(self.tg_token, self.tg_chat_id, tg_msg)
                    except Exception as e:
                        print(f"[LIVE] Fiyat güncelleme hatası: {e}")
                    next_price_time = now + 0.5

                if now >= next_candle_time:
                    try:
                        if stream_snapshot is None:
                            stream_snapshot = self.ws_stream.get_latest_bulk()
                        bulk_data = stream_snapshot

                        if not bulk_data:
                            bulk_data = TradingEngine.get_all_candles_parallel(main.SYMBOLS, main.TIMEFRAMES)

                        for (sym, tf), df in bulk_data.items():
                            if df.empty:
                                continue

                            try:
                                sym_cfg = SYMBOL_PARAMS.get(sym, {})
                                tf_cfg = sym_cfg.get(tf, {}) if isinstance(sym_cfg, dict) else {}
                                if tf_cfg.get("disabled", False):
                                    continue

                                if len(df) < 3:
                                    continue

                                if len(df) < 250:
                                    continue

                                df_ind = TradingEngine.calculate_indicators(df.copy())
                                closed = df_ind.iloc[-2]
                                forming = df_ind.iloc[-1]
                                curr_price = float(closed['close'])
                                closed_ts_utc = closed['timestamp']
                                forming_ts_utc = forming['timestamp']
                                istanbul_time = pd.Timestamp(closed_ts_utc) + pd.Timedelta(hours=3)
                                ts_str = istanbul_time.strftime("%Y-%m-%d %H:%M")
                                next_open_price = float(forming['open'])
                                next_open_ts_str = (pd.Timestamp(forming_ts_utc) + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")

                                if tf == "1m":
                                    self.price_signal.emit(sym, curr_price)

                                config = main.load_optimized_config(sym, tf)
                                strategy_mode = config.get("strategy_mode", "ssl_flow")

                                # Her iki strateji icin de EMA200 kullan
                                pb_top_col = 'pb_ema_top'
                                pb_bot_col = 'pb_ema_bot'

                                closed_trades = main.trade_manager.update_trades(
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
                                            tg_msg = (f"{icon} KAPANDI: {ct['symbol']}\nTF: {tf}\nSetup: {ct['setup']}\n"
                                                      f"Sonuç: {reason}\nNet PnL: {pnl_str}")
                                            TradingEngine.send_telegram(self.tg_token, self.tg_chat_id, tg_msg)

                                            # Check if circuit breaker should now kill this stream
                                            cb_triggered, cb_reason = main.trade_manager.check_stream_circuit_breaker(sym, tf)
                                            if cb_triggered and "already_killed" not in (cb_reason or ""):
                                                cb_msg = f"🛑 CIRCUIT BREAKER TRIGGERED\n{sym}-{tf}\nReason: {cb_reason}"
                                                TradingEngine.send_telegram(self.tg_token, self.tg_chat_id, cb_msg)
                                                self.update_ui_signal.emit(sym, tf, "{}", f"🛑 CIRCUIT BREAKER: {cb_reason}")

                                df_closed = df_ind.iloc[:-1].copy()

                                if config.get("disabled", False):
                                    continue

                                if main.is_stream_blacklisted(sym, tf):
                                    continue

                                rr, rsi, slope = config['rr'], config['rsi'], config['slope']
                                use_at = config['at_active']
                                at_status_log = "AT:ON" if use_at else "AT:OFF"
                                strategy_log = f"Mode:{strategy_mode[:7]}"

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
                                    if start > 0 and end > 0:
                                        setup_tag = s_reason[start:end]

                                active_trades = [t for t in main.trade_manager.open_trades if
                                                 t['timeframe'] == tf and t['symbol'] == sym]

                                live_pnl_str = ""
                                if active_trades:
                                    t = active_trades[0]
                                    current_pnl = float(t.get('pnl', 0))
                                    sign = "+" if current_pnl >= 0 else "-"
                                    partial_info = " (Part.Taken)" if t.get("partial_taken") else ""
                                    live_pnl_str = f" | PnL: {sign}${abs(current_pnl):.2f}{partial_info}"

                                decision = None
                                reject_reason = ""

                                if s_type and "ACCEPTED" in s_reason:
                                    has_open = False
                                    for t in main.trade_manager.open_trades:
                                        if t['symbol'] == sym and t['timeframe'] == tf:
                                            has_open = True
                                            break

                                    if has_open:
                                        decision = "Rejected"
                                        reject_reason = "Open Position"
                                        log_msg = f"{tf} | {curr_price} | ⚠️ Pozisyon Var{live_pnl_str}"
                                        json_data = main.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                active_trades if self.show_rr else [])
                                        self.update_ui_signal.emit(sym, tf, json_data, log_msg)

                                    elif self.last_signals[sym][tf] != closed_ts_utc:
                                        if self.last_signals[sym][tf] is None:
                                            self.last_signals[sym][tf] = closed_ts_utc
                                            self.last_potential[sym][tf] = closed_ts_utc
                                            decision = "Rejected"
                                            reject_reason = "Startup Sync"
                                            log_msg = f"{tf} | {curr_price} | 🔄 Başlangıç senkronizasyonu"
                                            json_data = main.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                    active_trades if self.show_rr else [])
                                            self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                        elif main.trade_manager.check_cooldown(sym, tf, forming_ts_utc):
                                            decision = "Rejected"
                                            reject_reason = "Cooldown"
                                            log_msg = f"{tf} | {curr_price} | ❄️ SOĞUMA SÜRECİNDE"
                                            json_data = main.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                    active_trades if self.show_rr else [])
                                            self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                        elif main.trade_manager.is_stream_killed(sym, tf):
                                            # Circuit breaker killed this stream
                                            decision = "Rejected"
                                            reject_reason = "Circuit Breaker"
                                            log_msg = f"{tf} | {curr_price} | 🛑 CIRCUIT BREAKER - Stream devre dışı"
                                            json_data = main.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                    active_trades if self.show_rr else [])
                                            self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                        elif main.trade_manager.check_global_circuit_breaker()[0]:
                                            # Global circuit breaker triggered
                                            decision = "Rejected"
                                            reject_reason = "Global Circuit Breaker"
                                            _, cb_reason = main.trade_manager.check_global_circuit_breaker()
                                            log_msg = f"{tf} | {curr_price} | 🛑 GLOBAL CB: {cb_reason}"
                                            json_data = main.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                    active_trades if self.show_rr else [])
                                            self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                        else:
                                            trade_data = {
                                                "symbol": sym, "timestamp": next_open_ts_str, "open_time_utc": forming_ts_utc,
                                                "timeframe": tf, "type": s_type,
                                                "entry": next_open_price, "tp": s_tp, "sl": s_sl, "setup": setup_tag,
                                                "config_snapshot": config,
                                            }
                                            main.trade_manager.open_trade(trade_data)
                                            self.trade_signal.emit(trade_data)

                                            msg = (f"🚀 SİNYAL: {s_type}\nSembol: {sym}\nTF: {tf}\nSetup: {setup_tag}\n"
                                                   f"Fiyat: {next_open_price:.4f}\nTP: {s_tp:.4f}")
                                            TradingEngine.send_telegram(self.tg_token, self.tg_chat_id, msg)

                                            self.last_signals[sym][tf] = closed_ts_utc
                                            decision = "Accepted"
                                            log_msg = f"{tf} | {curr_price} | 🔥 {s_type} ({setup_tag})"
                                            json_data = main.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                    active_trades if self.show_rr else [])
                                            self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                    else:
                                        decision = "Rejected"
                                        reject_reason = "Duplicate Signal"
                                        log_msg = f"{tf} | {curr_price} | ⏳ İşlemde...{live_pnl_str}"
                                        json_data = main.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                                active_trades if self.show_rr else [])
                                        self.update_ui_signal.emit(sym, tf, json_data, log_msg)
                                else:
                                    decision = f"Rejected: {s_reason}" if s_reason else "Rejected"
                                    if s_reason and "REJECT" in s_reason:
                                        log_msg = f"{tf} | {curr_price} | ⚠️ {s_reason}{live_pnl_str}"
                                    else:
                                        log_msg = f"{tf} | {curr_price} | {at_status_log}{live_pnl_str}"

                                    json_data = main.create_chart_data_json(df_closed, tf, sym, s_type,
                                                                            active_trades if self.show_rr else [])
                                    self.update_ui_signal.emit(sym, tf, json_data, log_msg)

                                if s_type and "ACCEPTED" in s_reason and self.last_potential[sym][tf] != closed_ts_utc and reject_reason != "Startup Sync":
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
                                    main.potential_trades.add(diag_entry)
                                    self.potential_signal.emit(diag_entry)
                                    self.last_potential[sym][tf] = closed_ts_utc

                            except Exception as e:
                                print(f"Loop Processing Error ({sym}-{tf}): {e}")
                                with open(os.path.join(main.DATA_DIR, "error_log.txt"), "a") as f:
                                    f.write(f"\n[{datetime.now()}] LOOP HATA: {str(e)}\n")
                                    f.write(traceback.format_exc())

                    except Exception as e:
                        print(f"Main Loop Error: {e}")
                        time.sleep(1)

                    next_candle_time = now + main.REFRESH_RATE

                time.sleep(0.1)
        finally:
            self.ws_stream.stop()


class OptimizerWorker(QThread):
    """Worker for parameter optimization across timeframes."""

    result_signal = pyqtSignal(str)

    def __init__(self, symbol, candle_limit_or_days, rr_range, rsi_range, slope_range, use_alphatrend,
                 monte_carlo_mode=False, timeframes=None, use_days=False):
        super().__init__()
        main = _get_main_module()
        self.symbol = symbol
        self.use_days = use_days
        self.timeframes = timeframes or list(main.TIMEFRAMES)
        if use_days:
            self.days = candle_limit_or_days
            self.candle_limit = None
            self.candle_limit_map = main.days_to_candles_map(candle_limit_or_days, self.timeframes)
        else:
            self.days = None
            self.candle_limit = candle_limit_or_days
            self.candle_limit_map = None
        self.rr_range = rr_range
        self.rsi_range = rsi_range
        self.slope_range = slope_range
        self.monte_carlo_mode = monte_carlo_mode

        self.slippage_rate = TRADING_CONFIG["slippage_rate"]
        self.funding_rate_8h = TRADING_CONFIG["funding_rate_8h"]
        self.total_fee = TRADING_CONFIG["total_fee"]
        self.leverage = TRADING_CONFIG["leverage"]

    def run(self):
        main = _get_main_module()
        try:
            # Get pandas_ta
            ta = main.get_ta()

            df_trend = pd.DataFrame()
            if not self.monte_carlo_mode:
                try:
                    df_trend = TradingEngine.get_data(self.symbol, "1d", limit=500)
                    if not df_trend.empty:
                        df_trend['ema_trend'] = ta.ema(df_trend['close'], length=200)
                        df_trend = df_trend[['timestamp', 'ema_trend']].dropna()
                except Exception:
                    pass

            data_cache = {}
            for tf in self.timeframes:
                if self.use_days and self.candle_limit_map:
                    tf_candle_limit = self.candle_limit_map.get(tf, 720)
                    self.result_signal.emit(f"⬇️ {tf} verisi hazırlanıyor ({self.days} gün = {tf_candle_limit} mum)...\n")
                else:
                    tf_candle_limit = self.candle_limit
                    self.result_signal.emit(f"⬇️ {tf} verisi hazırlanıyor...\n")
                df = TradingEngine.get_historical_data_pagination(self.symbol, tf, total_candles=tf_candle_limit)

                if not df.empty:
                    if self.monte_carlo_mode:
                        df = df.sample(frac=1).reset_index(drop=True)
                        df = df.fillna(method='ffill').fillna(method='bfill')

                    df = TradingEngine.calculate_indicators(df)

                    if not df_trend.empty and not self.monte_carlo_mode:
                        df = df.sort_values('timestamp')
                        df_trend = df_trend.sort_values('timestamp')
                        df = pd.merge_asof(df, df_trend, on='timestamp', direction='backward')
                    else:
                        df['ema_trend'] = np.nan
                    data_cache[tf] = df

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
                    if df is None or df.empty:
                        continue
                    is_trailing_active = (tf in TRAILING_ALLOWED_TFS)
                    net_r = 0
                    wins = 0
                    losses = 0

                    start_idx = 200
                    limit_idx = len(df) - 1
                    cooldown = 0

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
                            if i + 1 >= len(df):
                                break

                            next_candle = df.iloc[i + 1]
                            real_entry_price = next_candle['open']
                            entry_time = next_candle['timestamp']

                            if s_type == "LONG":
                                real_entry_price *= (1 + self.slippage_rate)
                            else:
                                real_entry_price *= (1 - self.slippage_rate)

                            if i < cooldown:
                                continue

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
                                curr_high = row['high']
                                curr_low = row['low']
                                curr_close = row['close']
                                exit_time = row['timestamp']

                                if s_type == "LONG":
                                    if curr_low <= sim_sl:
                                        outcome = "WIN (Trailing)" if sim_sl > real_entry_price else (
                                            "BE" if sim_sl == real_entry_price else "LOSS")
                                        cooldown = j + (10 if tf == "1m" else 6)
                                        break
                                    if curr_high >= sim_tp:
                                        outcome = "WIN (TP)"
                                        cooldown = j + 1
                                        break
                                else:
                                    if curr_high >= sim_sl:
                                        outcome = "WIN (Trailing)" if sim_sl < real_entry_price else (
                                            "BE" if sim_sl == real_entry_price else "LOSS")
                                        cooldown = j + (10 if tf == "1m" else 6)
                                        break
                                    if curr_low <= sim_tp:
                                        outcome = "WIN (TP)"
                                        cooldown = j + 1
                                        break

                                total_dist = abs(sim_tp - real_entry_price)
                                if total_dist > 0:
                                    prog = abs(curr_close - real_entry_price) / total_dist
                                    if is_trailing_active:
                                        if not has_breakeven and prog >= 0.40:
                                            sim_sl = real_entry_price
                                            has_breakeven = True
                                    else:
                                        if not partial_taken and prog >= 0.50:
                                            if risk_dist > 0:
                                                partial_realized_r = (abs(curr_close - real_entry_price) / risk_dist) * 0.5
                                            curr_size_ratio = 0.5
                                            partial_taken = True
                                            sim_sl = real_entry_price
                                            has_breakeven = True
                                        elif not has_breakeven and prog >= 0.40:
                                            sim_sl = real_entry_price
                                            has_breakeven = True

                            risk_pct = abs(real_entry_price - s_sl_raw) / real_entry_price
                            if risk_pct == 0:
                                risk_pct = 0.01

                            fee_cost_r = self.total_fee / risk_pct

                            duration_hours = 0
                            if not self.monte_carlo_mode:
                                try:
                                    duration_hours = (exit_time - entry_time).total_seconds() / 3600
                                except Exception:
                                    duration_hours = 0

                            funding_cost_r = ((duration_hours / 8) * self.funding_rate_8h * self.leverage) / risk_pct

                            if "WIN" in outcome:
                                reward_dist = abs(sim_tp - real_entry_price) if "TP" in outcome else abs(
                                    sim_sl - real_entry_price)

                                if risk_dist > 0:
                                    raw_r = partial_realized_r + (reward_dist / risk_dist) * curr_size_ratio
                                    net_r += (raw_r - fee_cost_r - funding_cost_r)
                                wins += 1

                            elif outcome == "LOSS":
                                loss_r = 1.0 if not partial_taken else 0
                                net_r += (partial_realized_r - loss_r - fee_cost_r - funding_cost_r)
                                losses += 1

                            elif outcome == "BE":
                                net_r += (partial_realized_r - fee_cost_r - funding_cost_r)

                    results_by_tf[tf].append(
                        {"RR": rr, "RSI": rsi, "Slope": slope, "AT": at_active, "Wins": wins, "Losses": losses,
                         "Net_R": net_r})

            elapsed = time.time() - start_time
            self.result_signal.emit(f"\n✅ İŞLEM BİTTİ ({elapsed:.1f}sn)\n{'=' * 40}\n")

            for tf in main.TIMEFRAMES:
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


class AutoBacktestWorker(QThread):
    """Worker for scheduled automatic backtesting at 03:00."""

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.force_run = False

    def run(self):
        main = _get_main_module()
        print("🌙 Gece Bekçisi Devrede... (03:00 Bekleniyor)")
        while self.is_running:
            now = datetime.now()
            if (now.hour == 3 and now.minute == 0 and now.second < 10) or self.force_run:
                print("🚀 Otomatik Tarama Başlatılıyor... (Bu işlem zaman alabilir)")
                self.run_full_analysis()
                self.force_run = False
                time.sleep(65)
            time.sleep(5)

    def run_full_analysis(self):
        main = _get_main_module()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        report_lines = [
            f"--- GÜNLÜK BACKTEST RAPORU ({timestamp}) ---",
            f"Gerçekçi Mod: %{TRADING_CONFIG['slippage_rate'] * 100} Slippage, %{TRADING_CONFIG['total_fee'] * 100} Fee",
            "BTC/ETH/SOL | TF: 1m,5m,15m,1h | Mum: 15000",
            "",
        ]

        try:
            max_daily_candles = max(main.DAILY_REPORT_CANDLE_LIMITS.values())
            print("[AUTO] Günlük backtest başlıyor (15k mum)")
            result = main.run_portfolio_backtest(
                symbols=main.SYMBOLS,
                timeframes=[tf for tf in main.TIMEFRAMES if tf in main.DAILY_REPORT_CANDLE_LIMITS],
                candles=max_daily_candles,
                out_trades_csv=os.path.join(main.DATA_DIR, "daily_report_trades.csv"),
                out_summary_csv=os.path.join(main.DATA_DIR, "daily_report_summary.csv"),
                limit_map=main.DAILY_REPORT_CANDLE_LIMITS,
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
                main.save_best_configs(best_configs)
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

        try:
            report_dir = os.path.join(os.getcwd(), "raporlar")
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)

            file_name = f"Rapor_{datetime.now().strftime('%Y-%m-%d_%H%M')}.txt"
            file_path = os.path.join(report_dir, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            print(f"✅ Rapor Kaydedildi: {file_path}")

            with open(main.CONFIG_FILE, 'r') as f:
                c = json.load(f)
                TradingEngine.send_telegram(c.get("telegram_token"), c.get("telegram_chat_id"),
                                            f"🌙 Rapor Hazır: {file_name}")
        except Exception as e:
            print(f"Rapor hatası: {e}")


class BacktestWorker(QThread):
    """Worker for portfolio backtesting."""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, symbols, timeframes, candles_or_days, skip_optimization=False, quick_mode=False, use_days=False, start_date=None, end_date=None):
        super().__init__()
        main = _get_main_module()
        self.symbols = symbols
        self.timeframes = timeframes
        self.use_days = use_days
        self.start_date = start_date
        self.end_date = end_date
        if start_date is not None:
            self.days = None
            self.candles = None
            self.limit_map = None
        elif use_days:
            self.days = candles_or_days
            self.candles = None
            self.limit_map = main.days_to_candles_map(candles_or_days, timeframes)
        else:
            self.days = None
            self.candles = candles_or_days
            self.limit_map = None
        self.skip_optimization = skip_optimization
        self.quick_mode = quick_mode
        self._last_log_time = 0.0
        self._pending_log = None

    def _throttled_log(self, text: str):
        now = time.time()
        if (now - self._last_log_time) >= 0.4:
            self.log_signal.emit(text)
            self._last_log_time = now
            self._pending_log = None
        else:
            self._pending_log = text

    def _flush_pending_log(self):
        if self._pending_log:
            self.log_signal.emit(self._pending_log)
            self._pending_log = None
            self._last_log_time = time.time()

    def run(self):
        main = _get_main_module()
        result = {}

        try:
            if self.start_date is not None:
                result = main.run_portfolio_backtest(
                    symbols=self.symbols,
                    timeframes=self.timeframes,
                    candles=50000,
                    progress_callback=self._throttled_log,
                    draw_trades=True,
                    max_draw_trades=30,
                    skip_optimization=self.skip_optimization,
                    quick_mode=self.quick_mode,
                    start_date=self.start_date,
                    end_date=self.end_date,
                ) or {}
            elif self.use_days and self.limit_map:
                result = main.run_portfolio_backtest(
                    symbols=self.symbols,
                    timeframes=self.timeframes,
                    candles=max(self.limit_map.values()),
                    limit_map=self.limit_map,
                    progress_callback=self._throttled_log,
                    draw_trades=True,
                    max_draw_trades=30,
                    skip_optimization=self.skip_optimization,
                    quick_mode=self.quick_mode,
                ) or {}
            else:
                result = main.run_portfolio_backtest(
                    symbols=self.symbols,
                    timeframes=self.timeframes,
                    candles=self.candles,
                    progress_callback=self._throttled_log,
                    draw_trades=True,
                    max_draw_trades=30,
                    skip_optimization=self.skip_optimization,
                    quick_mode=self.quick_mode,
                ) or {}
            self._flush_pending_log()
        except Exception as e:
            self.log_signal.emit(f"\n[BACKTEST][GUI] Hata: {e}\n{traceback.format_exc()}\n")
        finally:
            self.finished_signal.emit(result if isinstance(result, dict) else {})
