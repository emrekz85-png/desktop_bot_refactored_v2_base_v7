# runners/portfolio.py
# Portfolio backtest runner function
# Moved from main file for modularity (v40.5)

import os
import csv
import heapq
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from core import (
    DATA_DIR, TRADING_CONFIG,
    TradingEngine, SimTradeManager, tf_to_timedelta,
    _optimize_backtest_configs, _strategy_signature,
)
from strategies import check_signal


def run_portfolio_backtest(
    symbols,
    timeframes,
    candles: int = 3000,
    out_trades_csv: str = None,  # Default: DATA_DIR/backtest_trades.csv
    out_summary_csv: str = None,  # Default: DATA_DIR/backtest_summary.csv
    limit_map: Optional[dict] = None,
    progress_callback=None,
    draw_trades: bool = True,
    max_draw_trades: Optional[int] = None,
    skip_optimization: bool = False,  # True = cached config kullan, optimizer atla (HIZLI)
    quick_mode: bool = False,  # True = azaltılmış config grid (daha hızlı optimizer)
    start_date: str = None,  # Başlangıç tarihi "YYYY-MM-DD" formatında (tutarlı backtest için)
    end_date: str = None,  # Bitiş tarihi "YYYY-MM-DD" formatında (default: bugün)
):
    # Set default output paths to DATA_DIR
    if out_trades_csv is None:
        out_trades_csv = os.path.join(DATA_DIR, "backtest_trades.csv")
    if out_summary_csv is None:
        out_summary_csv = os.path.join(DATA_DIR, "backtest_summary.csv")

    allowed_log_categories = {"progress", "potential", "summary"}
    strategy_sig = _strategy_signature()

    def log(msg: str, category: str = None):
        if category not in allowed_log_categories:
            return
        if progress_callback:
            progress_callback(msg)
        print(msg)

    # Lazy import to avoid circular dependency
    from desktop_bot_refactored_v2_base_v7 import _audit_trade_logic_parity
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

        # Tarih aralığı modu mu kontrol et
        use_date_range = start_date is not None

        # Paralel veri çekme için iş listesi oluştur
        jobs = []
        for sym in symbols:
            for tf in timeframes:
                if use_date_range:
                    # Tarih aralığı modunda candle limit kullanılmaz
                    jobs.append((sym, tf, None))
                else:
                    tf_candle_limit = active_limit_map.get(tf, target_candles)
                    if tf_candle_limit:
                        tf_candle_limit = min(target_candles, tf_candle_limit)
                    else:
                        tf_candle_limit = target_candles
                    jobs.append((sym, tf, tf_candle_limit))

        total_jobs = len(jobs)
        if use_date_range:
            date_info = f"{start_date} → {end_date or 'bugün'}"
            log(f"📥 {total_jobs} stream için veri indiriliyor ({date_info})...", category="summary")
        else:
            log(f"📥 {total_jobs} stream için veri indiriliyor...", category="summary")

        def fetch_one(job):
            sym, tf, candle_limit = job
            try:
                if use_date_range:
                    # Tarih aralığı modu
                    df = TradingEngine.get_historical_data_pagination(
                        sym, tf, start_date=start_date, end_date=end_date
                    )
                else:
                    # Candle sayısı modu
                    df = TradingEngine.get_historical_data_pagination(sym, tf, total_candles=candle_limit)

                if df is None or df.empty or len(df) < 400:
                    return None
                df = TradingEngine.calculate_indicators(df)
                if write_prices:
                    df.to_csv(os.path.join(DATA_DIR, f"{sym}_{tf}_prices.csv"), index=False)
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
            # Backward compat: eski JSON'larda _confidence olabilir
            conf = cfg.get("confidence") or cfg.get("_confidence", "high")
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
        # SSL Flow ve Keltner Bounce icin EMA200 kullan
        stream_config = best_configs.get((sym, tf)) or load_optimized_config(sym, tf)
        strategy_mode = stream_config.get("strategy_mode", "ssl_flow")
        if strategy_mode == "ssl_flow_DISABLED":  # EMA150 artik kullanilmiyor
            pb_top_col = "pb_ema_top_150"
            pb_bot_col = "pb_ema_bot_150"
        else:
            pb_top_col = "pb_ema_top"
            pb_bot_col = "pb_ema_bot"

        streams_arrays[(sym, tf)] = {
            "timestamps": pd.to_datetime(df["timestamp"]).values,
            "highs": df["high"].values,
            "lows": df["low"].values,
            "closes": df["close"].values,
            "opens": df["open"].values,
            "pb_tops": df.get(pb_top_col, df["close"]).values if pb_top_col in df.columns else df["close"].values,
            "pb_bots": df.get(pb_bot_col, df["close"]).values if pb_bot_col in df.columns else df["close"].values,
            # v1.5: EMA15 for Momentum TP Extension check
            "ema15s": df["ema15"].values if "ema15" in df.columns else df["close"].values,
        }

    # Çoklu stream için zaman bazlı event kuyruğu
    heap = []
    ptr = {}
    total_events = 0
    # DETERMINISTIC: Sorted iteration ensures consistent order across runs
    for (sym, tf), df in sorted(streams.items()):
        warmup = 250
        end = len(df) - 2
        if end <= warmup:
            continue
        ptr[(sym, tf)] = warmup
        total_events += max(0, end - warmup)
        # Heap tuple: (timestamp, symbol, timeframe) - symbol/tf provide stable secondary sort
        heapq.heappush(
            heap,
            (pd.Timestamp(df.loc[warmup, "timestamp"]) + tf_to_timedelta(tf), sym, tf),
        )

    # ==========================================
    # IKI AYRI PORTFOY: SSL_Flow ve Keltner_Bounce stratejileri birbirini etkilemez
    # ==========================================
    # Her strateji kendi wallet balance'ına sahip, böylece:
    # - Bir stratejinin kaybı diğerinin pozisyon büyüklüğünü etkilemez
    # - Her strateji bağımsız değerlendirilebilir
    initial_balance = TRADING_CONFIG["initial_balance"]
    tm_ssl = SimTradeManager(initial_balance=initial_balance)     # SSL Flow icin
    tm_kb = SimTradeManager(initial_balance=initial_balance)      # Keltner Bounce icin

    # Stream -> strategy mapping (hangi stream hangi TM'yi kullanacak)
    stream_strategy_map = {}
    for (sym, tf) in requested_pairs:
        cfg = best_configs.get((sym, tf)) or load_optimized_config(sym, tf)
        strategy_mode = cfg.get("strategy_mode", "ssl_flow")
        stream_strategy_map[(sym, tf)] = strategy_mode

    def get_tm_for_stream(sym, tf):
        """Stream'in stratejisine göre doğru TradeManager'ı döndür"""
        strategy = stream_strategy_map.get((sym, tf), "keltner_bounce")
        return tm_ssl if strategy == "ssl_flow" else tm_kb

    # ==========================================
    # CIRCUIT BREAKER & ROLLING E[R] TRACKING
    # ==========================================
    # Import config from core.config
    from core.config import CIRCUIT_BREAKER_CONFIG, ROLLING_ER_CONFIG

    # Stream-level trackers
    stream_pnl_tracker = {}       # {(sym, tf): {"cumulative_pnl": 0, "peak_pnl": 0, "trades": 0}}
    stream_r_tracker = {}         # {(sym, tf): {"r_multiples": [], "ema_er": None}}
    circuit_breaker_killed = {}   # {(sym, tf): {"reason": str, "at_pnl": float, "at_trade": int}}

    # Global trackers (across all streams)
    # NOT: Backtest'te session bazında kümülatif PnL takip eder
    # Weekly tracking simüle edilmiş zamanla çalışır (v40.4)
    session_cumulative_pnl = 0.0  # Session bazında kümülatif PnL
    session_peak_pnl = 0.0        # Session peak PnL
    session_weekly_pnl = 0.0      # Haftalık PnL (v40.4)
    current_week_start = None     # Mevcut haftanın başlangıcı (v40.4)
    global_circuit_breaker_triggered = False

    def get_week_start(dt) -> datetime:
        """Get the Monday 00:00 UTC of the week containing dt."""
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        days_since_monday = dt.weekday()
        week_start = dt - timedelta(days=days_since_monday)
        return week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    def check_and_reset_week(trade_time) -> None:
        """Check if we've entered a new week and reset weekly PnL if so."""
        nonlocal session_weekly_pnl, current_week_start

        week_start = get_week_start(trade_time)

        if current_week_start is None:
            current_week_start = week_start
        elif week_start > current_week_start:
            session_weekly_pnl = 0.0
            current_week_start = week_start

    def check_stream_circuit_breaker(sym: str, tf: str, new_pnl: float, new_r: float = None) -> tuple:
        """Check if stream circuit breaker should trigger.

        Returns: (should_kill, reason)
        """
        if not CIRCUIT_BREAKER_CONFIG.get("enabled", True):
            return False, None

        key = (sym, tf)

        # Initialize tracker if needed
        if key not in stream_pnl_tracker:
            stream_pnl_tracker[key] = {"cumulative_pnl": 0.0, "peak_pnl": 0.0, "trades": 0}

        tracker = stream_pnl_tracker[key]
        tracker["cumulative_pnl"] += new_pnl
        tracker["trades"] += 1
        tracker["peak_pnl"] = max(tracker["peak_pnl"], tracker["cumulative_pnl"])

        # Track R-multiples for rolling E[R] check
        if new_r is not None and ROLLING_ER_CONFIG.get("enabled", True):
            if key not in stream_r_tracker:
                stream_r_tracker[key] = {"r_multiples": [], "ema_er": None}

            r_tracker = stream_r_tracker[key]
            r_tracker["r_multiples"].append(new_r)

            # Update EMA of E[R]
            alpha = ROLLING_ER_CONFIG.get("ema_alpha", 0.3)
            if r_tracker["ema_er"] is None:
                r_tracker["ema_er"] = new_r
            else:
                r_tracker["ema_er"] = alpha * new_r + (1 - alpha) * r_tracker["ema_er"]

        min_trades = CIRCUIT_BREAKER_CONFIG.get("stream_min_trades_before_kill", 5)
        if tracker["trades"] < min_trades:
            return False, None

        # Check 1: Absolute loss limit
        max_loss = CIRCUIT_BREAKER_CONFIG.get("stream_max_loss", -200.0)
        if tracker["cumulative_pnl"] < max_loss:
            return True, f"max_loss_exceeded (PnL=${tracker['cumulative_pnl']:.2f} < ${max_loss})"

        # Check 2: Drawdown from peak (DOLLAR-BASED for stream level)
        # Note: Percentage-based drawdown at stream level is problematic because:
        # - Initial balance is shared across all streams
        # - Small profit followed by loss gives absurd percentages
        # Solution: Use dollar-based drawdown for streams (simple and robust)
        max_dd_dollars = CIRCUIT_BREAKER_CONFIG.get("stream_max_drawdown_dollars", 100.0)
        if tracker["peak_pnl"] > 0:
            drawdown_dollars = tracker["peak_pnl"] - tracker["cumulative_pnl"]
            if drawdown_dollars > max_dd_dollars:
                return True, f"drawdown_exceeded (${drawdown_dollars:.2f} drop from peak ${tracker['peak_pnl']:.2f})"

        # Check 3: Rolling E[R] check
        if new_r is not None and ROLLING_ER_CONFIG.get("enabled", True):
            r_tracker = stream_r_tracker[key]
            window = ROLLING_ER_CONFIG.get("window_by_tf", {}).get(tf, 15)
            min_trades_er = ROLLING_ER_CONFIG.get("min_trades_before_check", 10)

            if len(r_tracker["r_multiples"]) >= min_trades_er:
                recent_r = r_tracker["r_multiples"][-window:] if len(r_tracker["r_multiples"]) >= window else r_tracker["r_multiples"]

                if ROLLING_ER_CONFIG.get("use_confidence_band", True) and len(recent_r) >= 5:
                    import statistics
                    mean_r = statistics.mean(recent_r)
                    stdev_r = statistics.stdev(recent_r) if len(recent_r) > 1 else 0
                    factor = ROLLING_ER_CONFIG.get("confidence_band_factor", 0.5)
                    lower_bound = mean_r - (stdev_r * factor)

                    kill_thresh = ROLLING_ER_CONFIG.get("kill_threshold", -0.05)
                    if lower_bound < kill_thresh:
                        return True, f"rolling_er_negative (E[R]={mean_r:.3f}, lower_bound={lower_bound:.3f})"
                else:
                    # Simple threshold check
                    rolling_er = sum(recent_r) / len(recent_r)
                    kill_thresh = ROLLING_ER_CONFIG.get("kill_threshold", -0.05)
                    if rolling_er < kill_thresh:
                        return True, f"rolling_er_below_threshold (E[R]={rolling_er:.3f} < {kill_thresh})"

        return False, None

    # Aktif portföy sayısını hesapla (dinamik - hardcoded "2" yerine)
    # stream_strategy_map'ten unique stratejileri bul
    active_strategies = set(stream_strategy_map.values())
    active_portfolio_count = len(active_strategies) if active_strategies else 1

    def check_global_circuit_breaker(new_pnl: float, trade_time=None) -> tuple:
        """Check if global circuit breaker should trigger.

        Args:
            new_pnl: PnL from the trade
            trade_time: Time of trade close (for weekly tracking)

        Returns: (should_kill, reason)
        """
        nonlocal session_cumulative_pnl, session_peak_pnl, session_weekly_pnl

        if not CIRCUIT_BREAKER_CONFIG.get("enabled", True):
            return False, None

        session_cumulative_pnl += new_pnl
        session_peak_pnl = max(session_peak_pnl, session_cumulative_pnl)

        # Update weekly tracking (v40.4)
        if trade_time is not None:
            check_and_reset_week(trade_time)
        session_weekly_pnl += new_pnl

        # Check session loss limit (backtest'te günlük reset yok, session bazlı)
        session_max_loss = CIRCUIT_BREAKER_CONFIG.get("global_daily_max_loss", -400.0)
        if session_cumulative_pnl < session_max_loss:
            return True, f"session_loss_exceeded (${session_cumulative_pnl:.2f} < ${session_max_loss})"

        # Check weekly loss limit (v40.4)
        weekly_max_loss = CIRCUIT_BREAKER_CONFIG.get("global_weekly_max_loss", -800.0)
        if session_weekly_pnl < weekly_max_loss:
            return True, f"weekly_loss_exceeded (${session_weekly_pnl:.2f} < ${weekly_max_loss})"

        # Check global drawdown (EQUITY-BASED, not PnL-based)
        # Bug fix: PnL-based drawdown gives absurd results (100%+) when PnL is small
        # Dinamik portföy sayısı: Sadece aktif stratejilerin bakiyelerini say
        total_initial = initial_balance * active_portfolio_count
        peak_equity = total_initial + session_peak_pnl
        current_equity = total_initial + session_cumulative_pnl
        max_dd_pct = CIRCUIT_BREAKER_CONFIG.get("global_max_drawdown_pct", 0.20)
        if peak_equity > total_initial:  # Only check if we've had profits
            dd_pct = (peak_equity - current_equity) / peak_equity
            if dd_pct > max_dd_pct:
                return True, f"global_drawdown_exceeded ({dd_pct:.1%} > {max_dd_pct:.1%})"

        return False, None

    logged_cfg_pairs = set()
    processed_events = 0
    next_progress = 10

    # Ana backtest döngüsü
    while heap:
        # Global circuit breaker kontrolü
        if global_circuit_breaker_triggered:
            log(f"🛑 [CIRCUIT BREAKER] Global circuit breaker tetiklendi - backtest durduruluyor", category="summary")
            break

        event_time, sym, tf = heapq.heappop(heap)
        df = streams[(sym, tf)]
        arrays = streams_arrays[(sym, tf)]
        i = ptr[(sym, tf)]
        if i >= len(df) - 1:
            continue

        # Açık pozisyonları güncelle (her iki TM için de güncelle - açık trade varsa)
        # Her TM sadece kendi açık trade'lerini günceller
        tm_for_stream = get_tm_for_stream(sym, tf)
        closed_trades = tm_for_stream.update_trades(
            sym,
            tf,
            candle_high=float(arrays["highs"][i]),
            candle_low=float(arrays["lows"][i]),
            candle_close=float(arrays["closes"][i]),
            candle_time_utc=pd.Timestamp(arrays["timestamps"][i]) + tf_to_timedelta(tf),
            pb_top=float(arrays["pb_tops"][i]),
            pb_bot=float(arrays["pb_bots"][i]),
            # v1.5: Pass EMA15 for Momentum TP Extension check
            candle_data={"ema15": float(arrays["ema15s"][i])},
        )

        # ==========================================
        # CIRCUIT BREAKER KONTROLÜ (trade kapandığında)
        # ==========================================
        for closed_trade in closed_trades:
            trade_pnl = closed_trade.get("pnl", 0)
            trade_risk = closed_trade.get("risk_amount", 1)  # Avoid division by zero
            trade_r = trade_pnl / trade_risk if trade_risk > 0 else 0

            # Stream circuit breaker kontrolü
            should_kill, kill_reason = check_stream_circuit_breaker(sym, tf, trade_pnl, trade_r)
            if should_kill and (sym, tf) not in circuit_breaker_killed:
                tracker = stream_pnl_tracker.get((sym, tf), {})
                circuit_breaker_killed[(sym, tf)] = {
                    "reason": kill_reason,
                    "at_pnl": tracker.get("cumulative_pnl", trade_pnl),
                    "at_trade": tracker.get("trades", 1),
                }
                log(
                    f"🛑 [CIRCUIT BREAKER] {sym}-{tf} DURDURULDU: {kill_reason}",
                    category="summary"
                )

            # Global circuit breaker kontrolü
            should_kill_global, global_reason = check_global_circuit_breaker(trade_pnl, event_time)
            if should_kill_global and not global_circuit_breaker_triggered:
                global_circuit_breaker_triggered = True
                log(
                    f"🛑 [CIRCUIT BREAKER] GLOBAL STOP: {global_reason}",
                    category="summary"
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

        # Skip if stream circuit breaker triggered
        if (sym, tf) in circuit_breaker_killed:
            continue

        if (sym, tf) not in logged_cfg_pairs:
            logged_cfg_pairs.add((sym, tf))
        rr, rsi, slope = config["rr"], config["rsi"], config["slope"]
        use_at = config.get("at_active", True)

        # Sinyal kontrolu (wrapper ile - ssl_flow veya keltner_bounce destekler)
        s_type, s_entry, s_tp, s_sl, s_reason = TradingEngine.check_signal(
            df,
            config=config,
            index=i,
            return_debug=False,
        )

        if s_type and "ACCEPTED" in str(s_reason):
            accepted_signals_raw[(sym, tf)] = accepted_signals_raw.get((sym, tf), 0) + 1
            # Aynı sembol/timeframe için açık trade var mı? (doğru TM'de kontrol et)
            has_open = any(
                t["symbol"] == sym and t["timeframe"] == tf
                for t in tm_for_stream.open_trades
            )

            cooldown_active = tm_for_stream.check_cooldown(sym, tf, event_time)
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

                # Try to open trade - only log if successful (doğru TM'de aç)
                # KRITIK: Sinyal üretirken kullanılan config'i trade'e göm
                # Bu sayede trade yönetimi aynı config ile yapılır (diskten tekrar yükleme yok)
                trade_opened = tm_for_stream.open_trade(
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
                        "config_snapshot": config,  # Sinyal üretiminde kullanılan config
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
                (pd.Timestamp(df.loc[i2, "timestamp"]) + tf_to_timedelta(tf), sym, tf),
            )

        # Progress log
        processed_events += 1
        if total_events > 0:
            progress = (processed_events / total_events) * 100
            if progress >= next_progress:
                log(f"[BACKTEST] %{progress:.1f} tamamlandı...", category="progress")
                next_progress += 10

    # ==========================================
    # BACKTEST SONUNDA AÇIK POZİSYONLARI KAPAT (Her iki TM için)
    # ==========================================
    def force_close_open_trades(tm, tm_name):
        """Bir TM'deki tüm açık pozisyonları zorla kapat"""
        if not tm.open_trades:
            return
        log(f"[BACKTEST] {tm_name}: {len(tm.open_trades)} açık pozisyon kapatılıyor...", category="summary")
        for trade in list(tm.open_trades):
            sym = trade["symbol"]
            tf = trade["timeframe"]
            entry = float(trade["entry"])
            t_type = trade["type"]
            current_size = float(trade.get("size", 0))
            initial_margin = float(trade.get("margin", 0))

            # Son kapanış fiyatını bul
            if (sym, tf) in streams:
                df_stream = streams[(sym, tf)]
                if not df_stream.empty:
                    last_close = float(df_stream.iloc[-1]["close"])
                else:
                    last_close = entry
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

        tm.open_trades.clear()
        log(f"[BACKTEST] {tm_name} açık pozisyonlar kapatıldı. Bakiye: ${tm.wallet_balance:.2f}", category="summary")

    force_close_open_trades(tm_ssl, "SSL_Flow")
    force_close_open_trades(tm_kb, "Keltner_Bounce")

    # Her iki TM'nin history'sini birlestir
    combined_history = tm_ssl.history + tm_kb.history

    total_closed_legs = len(combined_history)
    unique_trades = len({t.get("id") for t in combined_history}) if combined_history else 0
    partial_legs = sum(1 for t in combined_history if "PARTIAL" in str(t.get("status", "")))
    full_exits = total_closed_legs - partial_legs

    # Tüm history'den DataFrame oluştur ve CSV / özet yaz
    trades_df = pd.DataFrame(combined_history)
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
                    except (ValueError, TypeError, pd.errors.ParserError):
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
                strategy_tag = "SF" if strategy_mode == "ssl_flow" else "KB"  # SF=SSL Flow, KB=Keltner Bounce
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

    # ==========================================
    # CIRCUIT BREAKER RAPORU
    # ==========================================
    if circuit_breaker_killed:
        log(f"\n🛑 [CIRCUIT BREAKER] {len(circuit_breaker_killed)} stream otomatik durduruldu:", category="summary")
        total_saved = 0.0
        for (sym, tf), info in sorted(circuit_breaker_killed.items()):
            reason = info.get("reason", "unknown")
            at_pnl = info.get("at_pnl", 0)
            at_trade = info.get("at_trade", 0)

            # Bu stream'in backtest sonucundaki toplam PnL'i bul
            stream_final_pnl = next(
                (r["net_pnl"] for r in summary_rows if r["symbol"] == sym and r["timeframe"] == tf),
                at_pnl
            )
            # Eğer durdurulmasaydı ne kadar daha kaybedilirdi (tahmin)
            potential_loss_avoided = at_pnl - stream_final_pnl if stream_final_pnl < at_pnl else 0
            total_saved += abs(potential_loss_avoided) if potential_loss_avoided < 0 else 0

            log(
                f"  - {sym}-{tf}: Durduğu PnL=${at_pnl:.2f} (trade #{at_trade}) | Sebep: {reason}",
                category="summary"
            )

        # Özet istatistikler
        stream_pnls_at_kill = [info.get("at_pnl", 0) for info in circuit_breaker_killed.values()]
        avg_pnl_at_kill = sum(stream_pnls_at_kill) / len(stream_pnls_at_kill) if stream_pnls_at_kill else 0
        log(
            f"  📊 Circuit Breaker Özet: {len(circuit_breaker_killed)} stream durduruldu, "
            f"ortalama duruş PnL=${avg_pnl_at_kill:.2f}",
            category="summary"
        )

    if global_circuit_breaker_triggered:
        log(
            f"\n🛑 [GLOBAL CIRCUIT BREAKER] Bot tamamen durduruldu! "
            f"Session PnL=${session_cumulative_pnl:.2f}",
            category="summary"
        )

    # ==========================================
    # EXIT PROFILE & SL WIDENING STATS (v42.x)
    # ==========================================
    exit_stats = tm_ssl.get_exit_profile_stats()
    if exit_stats["total_trades"] > 0:
        profile_counts = exit_stats["profile_counts"]
        profile_pnl = exit_stats["profile_pnl"]
        log(f"\n📊 [EXIT PROFILE STATS] v42.x:", category="summary")
        for profile in ["clip", "runner"]:
            count = profile_counts.get(profile, 0)
            pnl = profile_pnl.get(profile, 0.0)
            if count > 0:
                log(f"  - {profile.upper()}: {count} trades, PnL=${pnl:+.2f}", category="summary")

        # SL widening stats
        sl_total = exit_stats["sl_widened_total"]
        if sl_total > 0:
            sl_wins = exit_stats["sl_widened_wins"]
            sl_losses = exit_stats["sl_widened_losses"]
            sl_wr = (sl_wins / sl_total * 100) if sl_total > 0 else 0
            log(
                f"  - SL Widened: {sl_total} trades ({sl_total/exit_stats['total_trades']*100:.1f}%), "
                f"WR={sl_wr:.1f}%",
                category="summary"
            )

    # Her strateji için ayrı portföy sonuçları
    combined_wallet = tm_ssl.wallet_balance + tm_kb.wallet_balance
    combined_pnl = tm_ssl.total_pnl + tm_kb.total_pnl
    log(
        f"\n📊 PORTFÖY SONUÇLARI (Ayrı Cüzdanlar):",
        category="summary",
    )
    log(
        f"  [SSL_Flow] Wallet: ${tm_ssl.wallet_balance:.2f} | PnL: ${tm_ssl.total_pnl:+.2f}",
        category="summary",
    )
    log(
        f"  [Keltner_Bounce] Wallet: ${tm_kb.wallet_balance:.2f} | PnL: ${tm_kb.total_pnl:+.2f}",
        category="summary",
    )
    log(
        f"  [TOPLAM] Combined Wallet: ${combined_wallet:.2f} | Combined PnL: ${combined_pnl:+.2f}",
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

    # 🔄 Dynamic blacklist güncellemesi - negatif PnL'li streamler blacklist'e alınır
    dynamic_bl = update_dynamic_blacklist(summary_rows)
    result = {
        "summary": summary_rows,
        "summary_rows": summary_rows,  # Backward compatibility alias
        "all_trades": combined_history,  # Raw trade data for metrics calculation
        "best_configs": best_configs,
        "trades_csv": out_trades_csv,
        "summary_csv": out_summary_csv,
        "strategy_signature": strategy_sig,
        "parity_report": parity_report,
        "dynamic_blacklist": dynamic_bl,
    }

    if draw_trades:
        try:
            replay_backtest_trades(trades_csv=out_trades_csv, max_trades=max_draw_trades)
        except Exception as e:
            log(f"[BACKTEST] Trade çiziminde hata: {e}", category="summary")

    return result

