# runners/rolling_wf.py
# Rolling Walk-Forward backtest runner functions
# Moved from main file for modularity (v40.5)

import os
import json
import heapq
import uuid
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from core import (
    VERSION,
    SYMBOLS, TIMEFRAMES, DATA_DIR, TRADING_CONFIG,
    TradingEngine, SimTradeManager, tf_to_timedelta,
    PR2_CONFIG, BASELINE_CONFIG, DEFAULT_STRATEGY_CONFIG,
    WALK_FORWARD_CONFIG,
    _optimize_backtest_configs,
    detect_regime,
    set_backtest_mode,  # Enable backtest mode (bypasses rate limiter, enables disk cache)
)
from strategies import check_signal


# Enable backtest mode for faster data fetching (no rate limiting, disk cache enabled)
set_backtest_mode(True)


def run_rolling_walkforward(
    symbols: list = None,
    timeframes: list = None,
    mode: str = "weekly",  # "fixed" or "weekly" (v40.6: monthly/triday removed)
    lookback_days: int = 60,  # 60-day optimizer window for better OOS performance
    forward_days: int = 7,    # Trade window (freeze period)
    start_date: str = None,   # "YYYY-MM-DD" - test period start
    end_date: str = None,     # "YYYY-MM-DD" - test period end
    fixed_config: dict = None,  # For mode="fixed" - config to use
    calibration_days: int = 60,  # For mode="fixed" - days to find config
    pre_calibration_days: int = 60,  # NEW: Pre-calibration period for initial configs (weekly mode)
    verbose: bool = True,
    output_dir: str = None,   # Run-specific output directory
) -> dict:
    """Run Rolling Walk-Forward backtest with stitched OOS results.

    Bu framework:
    1. Her dönem için sadece geçmiş veriye bakarak optimize eder
    2. Sonraki dönemi freeze config ile trade eder
    3. Tüm dönemlerin OOS PnL'ini toplar → ADDİTİF sonuç

    Args:
        symbols: Test edilecek semboller (default: SYMBOLS)
        timeframes: Test edilecek zaman dilimleri (default: TIMEFRAMES)
        mode: "fixed" (tek config) or "weekly" (haftalık re-opt)
        lookback_days: Optimize penceresi (train data)
        forward_days: Trade penceresi (OOS data) - sadece weekly için
        start_date: Test dönemi başlangıcı
        end_date: Test dönemi sonu
        fixed_config: mode="fixed" için kullanılacak config
        calibration_days: mode="fixed" için ilk N gün calibration (test dışı)
        pre_calibration_days: mode="weekly" için test öncesi pre-calibration süresi
                              Bu dönem Window 0 için fallback config sağlar
        verbose: Detaylı çıktı
        output_dir: Run çıktılarının kaydedileceği klasör

    Returns:
        dict with:
        - stitched_pnl: Toplam OOS PnL (realized)
        - total_trades: Toplam trade sayısı
        - max_drawdown: Maximum drawdown
        - window_results: Her pencere için detaylı sonuçlar
        - metrics: Karşılaştırma metrikleri
    """
    from datetime import datetime, timedelta
    import os
    import uuid

    def log(msg: str):
        if verbose:
            print(msg)

    # ==========================================
    # 1. RESEARCH MODE ISOLATION
    # ==========================================
    # Persist kapalı: best_configs ve blacklist yazılmaz
    # Data cache açık: OHLCV dosyaları okunabilir
    # Run ID format: VERSION_YYYYMMDD_HHMMSS_uuid (e.g., v1.0_20251225_143000_a1b2c3d4)
    run_id = f"{VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    if output_dir is None:
        output_dir = os.path.join(DATA_DIR, "rolling_wf_runs", run_id)
    os.makedirs(output_dir, exist_ok=True)

    log(f"\n{'='*70}")
    log(f"🔬 ROLLING WALK-FORWARD BACKTEST")
    log(f"{'='*70}")
    log(f"   Mode: {mode.upper()}")
    log(f"   Lookback: {lookback_days} gün | Forward: {forward_days} gün")
    log(f"   Period: {start_date or 'auto'} → {end_date or 'today'}")
    log(f"   Run ID: {run_id}")
    log(f"   Output: {output_dir}")
    log(f"{'='*70}\n")

    # ==========================================
    # 2. PARAMETER VALIDATION
    # ==========================================
    if symbols is None:
        symbols = SYMBOLS
    if timeframes is None:
        timeframes = TIMEFRAMES

    # Set default dates
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # Determine forward_days based on mode (v40.6: monthly/triday removed)
    if mode == "weekly":
        forward_days = 7
    # mode == "fixed" uses calibration_days concept differently

    # Calculate required lookback for start_date
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if start_date is None:
        # Default: 1 year of data
        start_dt = end_dt - timedelta(days=365)
        start_date = start_dt.strftime("%Y-%m-%d")
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    total_days = (end_dt - start_dt).days
    log(f"📅 Test dönemi: {total_days} gün ({start_date} → {end_date})")

    # ==========================================
    # 3. GENERATE WINDOWS
    # ==========================================
    windows = []

    if mode == "fixed":
        # Fixed mode: Use calibration period to find config, then trade rest
        if fixed_config is None:
            # Need to find config from calibration period
            calibration_end = start_dt + timedelta(days=calibration_days)
            log(f"📊 Calibration dönemi: {start_date} → {calibration_end.strftime('%Y-%m-%d')} ({calibration_days} gün)")

            # Single window for the rest
            windows.append({
                "window_id": 0,
                "optimize_start": start_dt,
                "optimize_end": calibration_end,
                "trade_start": calibration_end,
                "trade_end": end_dt,
                "config_source": "calibration",
            })
        else:
            # Config provided - single window for entire period
            log(f"📊 Fixed config kullanılıyor (calibration yok)")
            windows.append({
                "window_id": 0,
                "optimize_start": None,  # No optimization
                "optimize_end": None,
                "trade_start": start_dt,
                "trade_end": end_dt,
                "config_source": "provided",
                "config": fixed_config,
            })
    else:
        # Rolling mode: Generate windows
        # FIX: Trade from start_date, optimize using data BEFORE start_date
        current_start = start_dt  # First trade window starts at start_date (not start + lookback)

        window_id = 0
        while current_start < end_dt:
            window_end = min(current_start + timedelta(days=forward_days), end_dt)
            optimize_start = current_start - timedelta(days=lookback_days)

            windows.append({
                "window_id": window_id,
                "optimize_start": optimize_start,
                "optimize_end": current_start,
                "trade_start": current_start,
                "trade_end": window_end,
                "config_source": "optimized",
            })

            window_id += 1
            current_start = window_end

    log(f"📊 {len(windows)} pencere oluşturuldu")
    for w in windows[:3]:  # Show first 3
        opt_str = f"{w['optimize_start'].strftime('%m/%d') if w['optimize_start'] else 'N/A'}-{w['optimize_end'].strftime('%m/%d') if w['optimize_end'] else 'N/A'}"
        trade_str = f"{w['trade_start'].strftime('%m/%d')}-{w['trade_end'].strftime('%m/%d')}"
        log(f"   Window {w['window_id']}: Opt=[{opt_str}] Trade=[{trade_str}]")
    if len(windows) > 3:
        log(f"   ... ve {len(windows) - 3} pencere daha")

    # ==========================================
    # 4. HELPER FUNCTIONS FOR WINDOW EXECUTION
    # ==========================================

    def fetch_data_for_period(start_dt, end_dt, symbols_list, timeframes_list):
        """Fetch OHLCV data for all symbols/timeframes in a date range.

        SAFE OPTIMIZATION: Increased thread pool for IO-bound operations.
        This does NOT affect backtest results - only data fetching speed.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os

        # Add buffer for indicator warmup (200 candles worth of data before start)
        buffer_days = 30  # ~200 candles for 4h, more than enough for lower TFs

        fetch_start = start_dt - timedelta(days=buffer_days)
        fetch_start_str = fetch_start.strftime("%Y-%m-%d")
        fetch_end_str = end_dt.strftime("%Y-%m-%d")

        streams = {}

        def fetch_one(sym, tf):
            try:
                df = TradingEngine.get_historical_data_pagination(
                    sym, tf, start_date=fetch_start_str, end_date=fetch_end_str
                )
                if df is None or df.empty or len(df) < 250:
                    return None
                df = TradingEngine.calculate_indicators(df)
                return (sym, tf, df.reset_index(drop=True))
            except Exception as e:
                return None

        jobs = [(s, t) for s in symbols_list for t in timeframes_list]

        # SAFE OPTIMIZATION: Higher parallelism for IO-bound network requests
        # Ryzen 7 4800H (8C/16T) can handle more concurrent network requests
        # This only affects data fetch speed, NOT backtest logic/results
        max_workers = min(12, max(8, os.cpu_count() or 4))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one, s, t): (s, t) for s, t in jobs}
            for future in as_completed(futures):
                res = future.result()
                if res:
                    sym, tf, df = res
                    streams[(sym, tf)] = df

        return streams

    def filter_data_by_date(df, start_dt, end_dt):
        """Filter DataFrame to only include rows within date range."""
        if df.empty:
            return df

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Make start/end timezone-aware if df timestamps are
        if df['timestamp'].dt.tz is not None:
            start_dt = pd.Timestamp(start_dt).tz_localize('UTC')
            end_dt = pd.Timestamp(end_dt).tz_localize('UTC')
        else:
            start_dt = pd.Timestamp(start_dt)
            end_dt = pd.Timestamp(end_dt)

        mask = (df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)
        return df[mask].reset_index(drop=True)

    def run_isolated_optimization(streams, requested_pairs, log_func):
        """Run optimization without writing to global state."""
        # This is a simplified version that doesn't write to best_configs.json
        return _optimize_backtest_configs(
            streams,
            requested_pairs,
            progress_callback=None,
            log_to_stdout=False,
            use_walk_forward=False,  # Don't do walk-forward inside rolling WF
            quick_mode=True,  # Use quick mode for speed
        )

    def apply_mode_profile(config_map, wf_mode):
        """Apply mode-based default exit profile to configs.

        PR-1: BASELINE MODE - when sl_validation_mode="off", skip profile system
        and ensure baseline parameters are set.

        Profile system only applies when sl_validation_mode != "off":
        - weekly → clip (higher hit-rate, tighten-only clamp)
        - fixed → clip (default)
        (v40.6: monthly/triday removed)
        """
        from core.config import DEFAULT_STRATEGY_CONFIG, BASELINE_CONFIG

        for key, cfg in config_map.items():
            # PR-1: Check if baseline mode
            if cfg.get("sl_validation_mode", "off") == "off":
                # Ensure baseline parameters are set (override any existing)
                cfg["partial_trigger"] = BASELINE_CONFIG["partial_trigger"]
                cfg["partial_fraction"] = BASELINE_CONFIG["partial_fraction"]
                cfg["partial_rr_adjustment"] = BASELINE_CONFIG["partial_rr_adjustment"]
                cfg["dynamic_tp_only_after_partial"] = BASELINE_CONFIG["dynamic_tp_only_after_partial"]
                cfg["dynamic_tp_clamp_mode"] = BASELINE_CONFIG["dynamic_tp_clamp_mode"]
                # Don't set exit_profile - baseline ignores it
                continue

            # Profile mode (sl_validation_mode != "off")
            if not cfg.get("exit_profile"):
                profile_map = {
                    "weekly": DEFAULT_STRATEGY_CONFIG.get("default_profile_weekly", "clip"),
                    "fixed": "clip",
                }
                cfg["exit_profile"] = profile_map.get(wf_mode, "clip")

    def run_window_backtest(streams, config_map, trade_start_dt, trade_end_dt,
                           carried_positions=None, log_func=log):
        """Run backtest for a single window with frozen configs.

        Args:
            streams: Dict of (sym, tf) -> DataFrame
            config_map: Dict of (sym, tf) -> config to use
            trade_start_dt: Start of trade window
            trade_end_dt: End of trade window
            carried_positions: Open positions carried from previous window

        Returns:
            dict with window results and updated carried positions
        """
        import heapq

        # Initialize trade manager with current equity
        tm = SimTradeManager(initial_balance=running_equity)

        # Stage 4: Reset circuit breaker for clean per-window isolation
        # This ensures each rolling WF window starts fresh without carry-over
        tm.reset_circuit_breaker()

        # Restore carried positions
        if carried_positions:
            tm.open_trades = carried_positions.copy()
            # Recalculate locked margin
            for trade in tm.open_trades:
                tm.locked_margin += float(trade.get("margin", 0))
                tm.wallet_balance -= float(trade.get("margin", 0))

        # Pre-extract arrays for performance
        streams_arrays = {}
        for (sym, tf), df in streams.items():
            cfg = config_map.get((sym, tf), {})
            strategy_mode = cfg.get("strategy_mode", "ssl_flow")
            # Her iki strateji icin de EMA200 kullan
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
            }

        # Build event heap
        heap = []
        ptr = {}

        # Check if timestamps are timezone-aware by converting first one to pd.Timestamp
        first_stream_key = list(streams_arrays.keys())[0] if streams_arrays else None
        is_tz_aware = False
        if first_stream_key:
            first_ts = pd.Timestamp(streams_arrays[first_stream_key]["timestamps"][0])
            is_tz_aware = first_ts.tzinfo is not None

        # Convert trade window boundaries
        if is_tz_aware:
            trade_start_ts = pd.Timestamp(trade_start_dt).tz_localize('UTC')
            trade_end_ts = pd.Timestamp(trade_end_dt).tz_localize('UTC')
        else:
            trade_start_ts = pd.Timestamp(trade_start_dt)
            trade_end_ts = pd.Timestamp(trade_end_dt)

        for (sym, tf), df in sorted(streams.items()):
            # Find the index where trade window starts
            timestamps = streams_arrays[(sym, tf)]["timestamps"]

            # Find first index >= trade_start
            start_idx = 0
            for i, ts in enumerate(timestamps):
                if pd.Timestamp(ts) >= trade_start_ts:
                    start_idx = max(i, 250)  # Ensure warmup
                    break

            if start_idx >= len(df) - 2:
                continue

            ptr[(sym, tf)] = start_idx
            heapq.heappush(
                heap,
                (pd.Timestamp(timestamps[start_idx]) + tf_to_timedelta(tf), sym, tf),
            )

        # Track window metrics
        window_trades = []
        window_pnl = 0.0
        window_peak_pnl = 0.0
        window_max_dd = 0.0

        # PR-2: Weekly loss limit tracking
        from core.config import PR2_CONFIG
        window_loss_limit_hit = False
        weekly_max_loss = PR2_CONFIG.get("weekly_max_loss_usd", 50.0)
        weekly_stop_on_loss = PR2_CONFIG.get("weekly_stop_trading_on_loss", True)

        # PR-2: Stream full-stop streak tracking
        stream_fullstop_limit = PR2_CONFIG.get("stream_fullstop_limit", 2)
        stream_fullstop_counts = {}  # (sym, tf) -> consecutive fullstop count
        stream_disabled = set()  # Streams disabled due to fullstop streak

        # Process events
        while heap:
            ev_time, sym, tf = heapq.heappop(heap)

            # Stop if we've passed the trade window end
            if ev_time >= trade_end_ts:
                break

            idx = ptr[(sym, tf)]
            arr = streams_arrays[(sym, tf)]
            df = streams[(sym, tf)]

            if idx >= len(arr["timestamps"]) - 2:
                continue

            candle_time = arr["timestamps"][idx]
            candle_high = arr["highs"][idx]
            candle_low = arr["lows"][idx]
            candle_close = arr["closes"][idx]
            pb_top = arr["pb_tops"][idx]
            pb_bot = arr["pb_bots"][idx]

            # Get config for this stream
            cfg = config_map.get((sym, tf), {})
            if cfg.get("disabled", False):
                ptr[(sym, tf)] = idx + 1
                if idx + 1 < len(arr["timestamps"]) - 2:
                    next_time = pd.Timestamp(arr["timestamps"][idx + 1]) + tf_to_timedelta(tf)
                    if next_time < trade_end_ts:
                        heapq.heappush(heap, (next_time, sym, tf))
                continue

            # Update existing trades
            # v46.x: Pass ATR for ATR-based BE buffer
            candle_atr = arr["atr"][idx] if "atr" in arr else None
            closed = tm.update_trades(
                sym, tf, candle_high, candle_low, candle_close,
                candle_time, pb_top, pb_bot,
                candle_data={"atr": candle_atr}
            )

            for trade in closed:
                pnl = float(trade.get("pnl", 0))
                window_pnl += pnl
                window_peak_pnl = max(window_peak_pnl, window_pnl)
                window_max_dd = max(window_max_dd, window_peak_pnl - window_pnl)
                window_trades.append(trade)

                # PR-2: Check if window loss limit hit
                if weekly_stop_on_loss and window_pnl <= -weekly_max_loss and not window_loss_limit_hit:
                    window_loss_limit_hit = True
                    # Note: We still process exits but skip new entries

                # PR-2: Track full-stop streak per stream
                trade_sym = trade.get("symbol")
                trade_tf = trade.get("timeframe")
                stream_key = (trade_sym, trade_tf)
                trade_status = trade.get("status", "")

                # Full stop = hit SL (not partial, not TP, not trailing)
                is_fullstop = "SL" in trade_status and "PARTIAL" not in trade_status

                if is_fullstop:
                    stream_fullstop_counts[stream_key] = stream_fullstop_counts.get(stream_key, 0) + 1
                    if stream_fullstop_counts[stream_key] >= stream_fullstop_limit:
                        stream_disabled.add(stream_key)
                elif pnl > 0:
                    # Win resets the streak
                    stream_fullstop_counts[stream_key] = 0

            # Check for new signals (only if no open position for this stream)
            # PR-2: Skip new entries if window loss limit is hit
            if window_loss_limit_hit:
                # Skip to next event - only process exits (update_trades above already ran)
                ptr[(sym, tf)] = idx + 1
                if idx + 1 < len(arr["timestamps"]) - 2:
                    next_time = pd.Timestamp(arr["timestamps"][idx + 1]) + tf_to_timedelta(tf)
                    if next_time < trade_end_ts:
                        heapq.heappush(heap, (next_time, sym, tf))
                continue

            # PR-2: Skip new entries if stream is disabled due to full-stop streak
            if (sym, tf) in stream_disabled:
                ptr[(sym, tf)] = idx + 1
                if idx + 1 < len(arr["timestamps"]) - 2:
                    next_time = pd.Timestamp(arr["timestamps"][idx + 1]) + tf_to_timedelta(tf)
                    if next_time < trade_end_ts:
                        heapq.heappush(heap, (next_time, sym, tf))
                continue

            has_open = any(t.get("symbol") == sym and t.get("timeframe") == tf for t in tm.open_trades)
            if not has_open and not tm.check_cooldown(sym, tf, candle_time):
                # Get signal
                strategy_mode = cfg.get("strategy_mode", "ssl_flow")
                rr = cfg.get("rr", 2.0)
                rsi_limit = cfg.get("rsi", 60)
                at_active = cfg.get("at_active", True)

                if strategy_mode == "ssl_flow":
                    # NOTE: AlphaTrend is now MANDATORY for SSL_Flow (no use_alphatrend param)
                    sig = TradingEngine.check_ssl_flow_signal(
                        df, idx, min_rr=rr, rsi_limit=rsi_limit
                    )
                else:
                    sig = TradingEngine.check_signal_diagnostic(
                        df, idx, min_rr=rr, rsi_limit=rsi_limit,
                        slope_thresh=0.0, use_alphatrend=at_active
                    )

                # Signal is a tuple: (signal_type, entry, tp, sl, reason)
                # signal_type is "LONG", "SHORT", or None
                if sig and len(sig) >= 5 and sig[0] is not None:
                    signal_type, entry, tp, sl, reason = sig[:5]

                    # Capture indicator snapshot at entry time (for trade logging)
                    row = df.iloc[idx]
                    indicators_at_entry = {
                        "at_buyers": float(row.get("at_buyers", 0)) if pd.notna(row.get("at_buyers")) else None,
                        "at_sellers": float(row.get("at_sellers", 0)) if pd.notna(row.get("at_sellers")) else None,
                        "at_is_flat": bool(row.get("at_is_flat", False)) if pd.notna(row.get("at_is_flat")) else False,
                        # Dominance based on LINE DIRECTION (alphatrend vs alphatrend_2)
                        # BUYERS = line rising (blue in TV), SELLERS = line falling (red in TV)
                        "at_dominant": "BUYERS" if row.get("at_buyers_dominant", False) else ("SELLERS" if row.get("at_sellers_dominant", False) else "FLAT"),
                        "baseline": float(row.get("baseline", 0)) if pd.notna(row.get("baseline")) else None,
                        "pb_ema_top": float(row.get("pb_ema_top", 0)) if pd.notna(row.get("pb_ema_top")) else None,
                        "pb_ema_bot": float(row.get("pb_ema_bot", 0)) if pd.notna(row.get("pb_ema_bot")) else None,
                        "rsi": float(row.get("rsi", 0)) if pd.notna(row.get("rsi")) else None,
                        "adx": float(row.get("adx", 0)) if pd.notna(row.get("adx")) else None,
                        "keltner_upper": float(row.get("keltner_upper", 0)) if pd.notna(row.get("keltner_upper")) else None,
                        "keltner_lower": float(row.get("keltner_lower", 0)) if pd.notna(row.get("keltner_lower")) else None,
                        "close": float(row.get("close", 0)) if pd.notna(row.get("close")) else None,
                    }

                    # Build trade data with config snapshot and indicators
                    trade_data = {
                        "symbol": sym,
                        "timeframe": tf,
                        "type": signal_type,
                        "entry": entry,
                        "tp": tp,
                        "sl": sl,
                        "open_time_utc": candle_time,
                        "setup": reason or "Unknown",
                        "config_snapshot": cfg,  # Snapshot at entry time
                        "indicators_at_entry": indicators_at_entry,  # Indicator snapshot
                    }
                    tm.open_trade(trade_data)

            # Advance pointer
            ptr[(sym, tf)] = idx + 1
            if idx + 1 < len(arr["timestamps"]) - 2:
                next_time = pd.Timestamp(arr["timestamps"][idx + 1]) + tf_to_timedelta(tf)
                if next_time < trade_end_ts:
                    heapq.heappush(heap, (next_time, sym, tf))

        # === STAGE 0 BUGFIX: Force-close open positions at window end ===
        # This ensures each window's PnL is properly isolated
        force_closed_trades = []
        for trade in tm.open_trades:
            # Force close at last known price
            last_price = None
            sym = trade.get("symbol")
            tf_key = trade.get("timeframe")
            if (sym, tf_key) in streams_arrays:
                arr = streams_arrays[(sym, tf_key)]
                last_price = arr["closes"][-3] if len(arr["closes"]) > 2 else arr["closes"][-1]

            if last_price is None:
                last_price = float(trade.get("entry", 0))

            # Calculate PnL for forced close
            entry = float(trade.get("entry", 0))
            size = float(trade.get("size", 0))
            t_type = trade.get("type", "LONG")

            if t_type == "LONG":
                pnl = (last_price - entry) * size
            else:
                pnl = (entry - last_price) * size

            # Deduct commission
            notional = abs(size) * last_price
            commission = notional * TRADING_CONFIG.get("total_fee", 0.0008)
            net_pnl = pnl - commission

            # Create force-closed trade record
            closed_trade = trade.copy()
            closed_trade["status"] = "WF_WINDOW_END"
            closed_trade["pnl"] = net_pnl
            closed_trade["close_price"] = last_price
            closed_trade["close_time_utc"] = trade_end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            force_closed_trades.append(closed_trade)

            # Update wallet
            margin = float(trade.get("margin", 0))
            tm.wallet_balance += margin + net_pnl
            tm.locked_margin -= margin
            tm.total_pnl += net_pnl

        # Clear open trades after force close
        tm.open_trades.clear()

        # Return results - use tm.history to include partial TP records
        # tm.history contains BOTH partial records AND final close records
        all_window_trades = tm.history.copy() + force_closed_trades
        wins = sum(1 for t in all_window_trades if float(t.get("pnl", 0)) > 0)
        # Recalculate window_pnl from history to include partial PnL
        window_pnl = sum(float(t.get("pnl", 0)) for t in all_window_trades)

        # === STAGE 0: Position vs Leg counting ===
        # Unique position key: (symbol, timeframe, open_time_utc, type)
        # PARTIAL and final close for same position share the same key
        unique_positions = set()
        for t in all_window_trades:
            pos_key = (
                t.get("symbol"),
                t.get("timeframe"),
                t.get("open_time_utc"),
                t.get("type"),
            )
            unique_positions.add(pos_key)

        positions_count = len(unique_positions)
        legs_count = len(all_window_trades)

        # Position-based win counting (a position is a win if total PnL from all legs > 0)
        position_pnl = {}
        for t in all_window_trades:
            pos_key = (
                t.get("symbol"),
                t.get("timeframe"),
                t.get("open_time_utc"),
                t.get("type"),
            )
            position_pnl[pos_key] = position_pnl.get(pos_key, 0) + float(t.get("pnl", 0))

        positions_wins = sum(1 for pnl in position_pnl.values() if pnl > 0)

        return {
            "pnl": window_pnl,
            "trades": legs_count,  # Keep for backward compat (actually leg count)
            "wins": wins,  # Leg-based wins for backward compat
            "positions_count": positions_count,  # NEW: unique positions
            "legs_count": legs_count,  # NEW: record count
            "positions_wins": positions_wins,  # NEW: position-based wins
            "max_dd": window_max_dd,
            "closed_trades": all_window_trades,  # Now includes partials + force-closed!
            "open_positions": [],  # Always empty after force-close
            "final_equity": tm.wallet_balance + tm.locked_margin,
            "force_closed_count": len(force_closed_trades),
            "loss_limit_hit": window_loss_limit_hit,  # PR-2: Window stopped by loss limit
            "streams_disabled": list(stream_disabled),  # PR-2: Streams disabled by fullstop streak
        }

    # ==========================================
    # 4. EXECUTE WINDOWS
    # ==========================================
    all_window_results = []
    all_trades = []
    equity_curve = []
    running_equity = TRADING_CONFIG["initial_balance"]
    carried_positions = []  # Positions carried between windows
    global_config_map = {}  # Config map for fixed mode

    # PR-2: Carry-forward config tracking
    from core.config import PR2_CONFIG, BASELINE_CONFIG
    carry_forward_enabled = PR2_CONFIG.get("carry_forward_enabled", True)
    carry_forward_max_age = PR2_CONFIG.get("carry_forward_max_age_windows", 2)
    carry_forward_risk_mult = PR2_CONFIG.get("carry_forward_risk_multiplier", 0.75)
    min_trades_for_good = PR2_CONFIG.get("min_trades_for_good", 5)
    min_pnl_for_good = PR2_CONFIG.get("min_pnl_for_good", 10.0)
    min_win_rate_for_good = PR2_CONFIG.get("min_win_rate_for_good", 0.55)

    # Track "good" configs from previous windows: {(sym, tf): {"config": {...}, "window_id": n, "pnl": X, "trades": Y, "win_rate": Z}}
    good_configs_history = {}
    # Track config source per stream per window for reporting
    config_source_log = []

    # For fixed mode with provided config, build config map once
    if mode == "fixed" and fixed_config is not None:
        for sym in symbols:
            for tf in timeframes:
                global_config_map[(sym, tf)] = fixed_config.copy()

    # ==========================================
    # 4.5 MASTER DATA CACHE (PERFORMANCE OPTIMIZATION)
    # ==========================================
    # Pre-fetch ALL data and calculate indicators ONCE for the entire test period.
    # This eliminates redundant API calls and indicator calculations for overlapping windows.
    # Speedup: ~2-3x for typical 1-year rolling WF tests (52 windows with 30-day overlap)

    master_cache = {}  # (sym, tf) -> DataFrame with all indicators pre-calculated

    # Pre-calibration period calculation (for weekly mode)
    pre_cal_start = None
    pre_cal_end = None

    if windows:
        # Determine the full date range needed
        first_window = windows[0]
        last_window = windows[-1]

        # Start: earliest optimize_start (or trade_start - buffer)
        if first_window.get("optimize_start"):
            master_start = first_window["optimize_start"]
        else:
            master_start = first_window["trade_start"] - timedelta(days=30)

        # NEW: For weekly mode, extend to include pre-calibration period
        # Pre-calibration runs BEFORE first window's optimize_start
        if mode == "weekly" and pre_calibration_days > 0:
            pre_cal_end = first_window["optimize_start"]  # Pre-cal ends where first optimize starts
            pre_cal_start = pre_cal_end - timedelta(days=pre_calibration_days)
            # Extend master_start to include pre-calibration period
            master_start = min(master_start, pre_cal_start)
            log(f"📊 Pre-calibration dönemi: {pre_cal_start.strftime('%Y-%m-%d')} → {pre_cal_end.strftime('%Y-%m-%d')} ({pre_calibration_days} gün)")

        # Add extra buffer for indicator warmup (200+ candles)
        buffer_days = 45  # Extra buffer for indicator calculation warmup
        master_start = master_start - timedelta(days=buffer_days)

        # End: latest trade_end
        master_end = last_window["trade_end"]

        log(f"\n📦 Master Cache: Pre-fetching all data...")
        log(f"   Date range: {master_start.strftime('%Y-%m-%d')} → {master_end.strftime('%Y-%m-%d')}")

        # Fetch all data once with indicators
        master_cache = fetch_data_for_period(master_start, master_end, symbols, timeframes)

        if master_cache:
            log(f"   ✓ Master cache created: {len(master_cache)} streams with pre-calculated indicators")
        else:
            log(f"   ⚠️ Master cache creation failed - falling back to per-window fetching")

    def get_streams_from_cache(fetch_start, fetch_end, use_master_cache=True):
        """Get streams for a window from master cache or fetch fresh.

        This function slices the master cache for the requested date range,
        avoiding redundant API calls and indicator calculations.

        IMPORTANT: Includes 30-day buffer before fetch_start for indicator warmup,
        matching the behavior of fetch_data_for_period().
        """
        if not use_master_cache or not master_cache:
            # Fallback: fetch fresh data
            return fetch_data_for_period(fetch_start, fetch_end, symbols, timeframes)

        # Slice from master cache
        sliced_streams = {}

        # Add buffer for indicator warmup (same as fetch_data_for_period)
        buffer_days = 30
        buffered_start = fetch_start - timedelta(days=buffer_days)

        for (sym, tf), df_full in master_cache.items():
            if df_full.empty:
                continue

            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_full['timestamp']):
                df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

            # Make dates timezone-aware if needed
            if df_full['timestamp'].dt.tz is not None:
                start_ts = pd.Timestamp(buffered_start).tz_localize('UTC')
                end_ts = pd.Timestamp(fetch_end).tz_localize('UTC')
            else:
                start_ts = pd.Timestamp(buffered_start)
                end_ts = pd.Timestamp(fetch_end)

            # Filter to requested range (WITH buffer for warmup)
            mask = (df_full['timestamp'] >= start_ts) & (df_full['timestamp'] <= end_ts)
            df_sliced = df_full[mask].reset_index(drop=True)

            # Only include if enough data (250 candles minimum for indicator warmup)
            if len(df_sliced) >= 250:
                sliced_streams[(sym, tf)] = df_sliced

        return sliced_streams

    # ==========================================
    # 4.6 PRE-CALIBRATION (Weekly Mode Only)
    # ==========================================
    # Run optimization on period BEFORE first window's optimize_start
    # to provide initial configs for Window 0 and carry-forward fallback.
    # This ensures Window 0 has fallback configs even when optimizer finds nothing.

    if mode == "weekly" and pre_cal_start is not None and pre_cal_end is not None and master_cache:
        log(f"\n🔧 Pre-calibration optimizasyonu başlatılıyor...")
        log(f"   Dönem: {pre_cal_start.strftime('%Y-%m-%d')} → {pre_cal_end.strftime('%Y-%m-%d')}")

        # Get data for pre-calibration period from master cache
        pre_cal_streams = get_streams_from_cache(pre_cal_start, pre_cal_end, use_master_cache=True)

        if pre_cal_streams:
            log(f"   ✓ {len(pre_cal_streams)} stream için pre-calibration verisi hazır")

            # Run optimization on pre-calibration period
            pre_cal_pairs = [(sym, tf) for (sym, tf) in pre_cal_streams.keys()]
            pre_cal_configs = run_isolated_optimization(pre_cal_streams, pre_cal_pairs, log)

            # Count enabled configs
            enabled_pre_cal = {k: v for k, v in pre_cal_configs.items() if not v.get('disabled')}
            disabled_pre_cal = {k: v for k, v in pre_cal_configs.items() if v.get('disabled')}

            log(f"   ✓ Pre-calibration tamamlandı: {len(enabled_pre_cal)} aktif, {len(disabled_pre_cal)} disabled config")

            # Initialize good_configs_history with pre-calibration results
            # Use window_id=-1 to indicate pre-calibration origin
            for stream_key, cfg in enabled_pre_cal.items():
                good_configs_history[stream_key] = {
                    "config": cfg,
                    "window_id": -1,  # -1 = pre-calibration
                    "pnl": 0.0,  # Unknown - not traded yet
                    "trades": 0,
                    "win_rate": 0.0,
                    "source": "pre_calibration",
                }

            if enabled_pre_cal:
                log(f"   📋 {len(enabled_pre_cal)} pre-cal config good_configs_history'ye eklendi (fallback için)")
        else:
            log(f"   ⚠️ Pre-calibration verisi yetersiz - fallback config olmayacak")

    for window in windows:
        log(f"\n{'─'*50}")
        log(f"📦 Window {window['window_id']}: Trade [{window['trade_start'].strftime('%Y-%m-%d')} → {window['trade_end'].strftime('%Y-%m-%d')}]")

        # PR-2: Initialize window_config_sources for this window
        window_config_sources = {}

        # 4.1 Determine data range to fetch
        if window.get("optimize_start"):
            fetch_start = window["optimize_start"]
        else:
            fetch_start = window["trade_start"] - timedelta(days=30)  # Buffer for indicators

        fetch_end = window["trade_end"]

        # 4.2 Get data from master cache (or fetch if cache miss)
        use_cache = bool(master_cache)
        cache_status = "cache" if use_cache else "fetch"
        log(f"   📥 Veri [{cache_status}]: {fetch_start.strftime('%Y-%m-%d')} → {fetch_end.strftime('%Y-%m-%d')}")
        streams = get_streams_from_cache(fetch_start, fetch_end, use_master_cache=use_cache)

        if not streams:
            log(f"   ⚠️ Veri yok, window atlanıyor")
            window_result = {
                "window_id": window["window_id"],
                "trade_start": window["trade_start"].strftime("%Y-%m-%d"),
                "trade_end": window["trade_end"].strftime("%Y-%m-%d"),
                "pnl": 0.0,
                "trades": 0,
                "wins": 0,
                "max_dd": 0.0,
                "config_used": {},
                "skipped": True,
                # AŞAMA 8: 0-trade window reason logging
                "zero_trade_reason": "no_data",
                "zero_trade_details": "Veri bulunamadı veya indirilemedi",
            }
            all_window_results.append(window_result)
            continue

        log(f"   ✓ {len(streams)} stream yüklendi")

        # Regime detection for this window
        window_regimes = {}
        for (sym, tf), df in streams.items():
            # Filter to trade window start for regime detection
            trade_start_ts = pd.Timestamp(window["trade_start"])
            ts_col = pd.to_datetime(df["timestamp"])
            # Handle timezone: make both tz-naive for comparison
            if ts_col.dt.tz is not None:
                ts_col = ts_col.dt.tz_localize(None)
            mask = ts_col <= trade_start_ts
            if mask.any():
                df_at_start = df[mask]
                if len(df_at_start) >= 100:
                    regime_info = detect_regime(df_at_start, index=-1)
                    window_regimes[(sym, tf)] = regime_info["regime"]

        # Count regime distribution
        regime_counts = {}
        for regime in window_regimes.values():
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Determine dominant regime
        if regime_counts:
            dominant_regime = max(regime_counts, key=regime_counts.get)
            regime_summary = ", ".join([f"{r}={c}" for r, c in sorted(regime_counts.items())])
            log(f"   📊 Rejim: {dominant_regime} ({regime_summary})")
        else:
            dominant_regime = "UNKNOWN"

        # 4.3 Get/optimize config for this window
        if window["config_source"] == "provided":
            # Fixed mode with provided config
            config_map = global_config_map
            apply_mode_profile(config_map, mode)  # Stage 1: Mode-based profile
            log(f"   📋 Sabit config kullanılıyor")

        elif window["config_source"] == "calibration":
            # Fixed mode - optimize on calibration period
            log(f"   🔧 Calibration optimizasyonu çalıştırılıyor...")

            # Filter streams to calibration period
            calib_streams = {}
            for (sym, tf), df in streams.items():
                filtered = filter_data_by_date(df, window["optimize_start"], window["optimize_end"])
                if len(filtered) >= 250:
                    calib_streams[(sym, tf)] = filtered

            if calib_streams:
                requested_pairs = [(sym, tf) for (sym, tf) in calib_streams.keys()]
                config_map = run_isolated_optimization(calib_streams, requested_pairs, log)
                apply_mode_profile(config_map, mode)  # Stage 1: Mode-based profile
                global_config_map = config_map  # Save for reference
                log(f"   ✓ {len([c for c in config_map.values() if not c.get('disabled')])} config bulundu")
            else:
                config_map = {}
                log(f"   ⚠️ Calibration verisi yetersiz")

        else:
            # Rolling mode - optimize on optimize period
            log(f"   🔧 Window optimizasyonu: {window['optimize_start'].strftime('%m/%d')} → {window['optimize_end'].strftime('%m/%d')}")

            # Filter streams to optimize period
            opt_streams = {}
            for (sym, tf), df in streams.items():
                filtered = filter_data_by_date(df, window["optimize_start"], window["optimize_end"])
                if len(filtered) >= 250:
                    opt_streams[(sym, tf)] = filtered

            if opt_streams:
                requested_pairs = [(sym, tf) for (sym, tf) in opt_streams.keys()]
                config_map = run_isolated_optimization(opt_streams, requested_pairs, log)
                apply_mode_profile(config_map, mode)  # Stage 1: Mode-based profile
                enabled_count = len([c for c in config_map.values() if not c.get('disabled')])
                log(f"   ✓ {enabled_count} config bulundu")
            else:
                config_map = {}
                log(f"   ⚠️ Optimize verisi yetersiz")

            # PR-2: Carry-forward logic for disabled configs (NOT for missing configs!)
            # Key insight: Only apply carry-forward to streams that were in opt_streams
            # but optimizer marked as disabled. Don't add baseline to streams that
            # weren't optimized at all - they should remain disabled.
            window_config_sources = {}  # Track source for this window
            if carry_forward_enabled and opt_streams:
                current_window_id = window["window_id"]
                carry_forward_count = 0

                # Only consider streams that were actually optimized
                for stream_key in opt_streams.keys():
                    cfg = config_map.get(stream_key, {})

                    # Check if optimizer found a config (it will be in config_map)
                    # and if it's disabled (optimizer didn't find profitable edge)
                    if stream_key in config_map:
                        if cfg.get("disabled", False):
                            # Optimizer tried but marked disabled - try carry-forward
                            if stream_key in good_configs_history:
                                hist = good_configs_history[stream_key]
                                age = current_window_id - hist["window_id"]

                                # Check if within age limit
                                # For pre-calibration (window_id=-1), age will be window_id + 1
                                # e.g., Window 0: age = 0 - (-1) = 1, Window 1: age = 1 - (-1) = 2
                                if age <= carry_forward_max_age:
                                    # Use carry-forward config with reduced risk
                                    cf_config = hist["config"].copy()
                                    cf_config["carry_forward"] = True
                                    cf_config["carry_forward_age"] = age
                                    cf_config["carry_forward_from_window"] = hist["window_id"]
                                    cf_config["carry_forward_risk_multiplier"] = carry_forward_risk_mult
                                    cf_config["disabled"] = False  # Enable it

                                    config_map[stream_key] = cf_config
                                    # Track if this is from pre-calibration or regular window
                                    if hist["window_id"] == -1:
                                        window_config_sources[stream_key] = "carry_forward_precal"
                                    else:
                                        window_config_sources[stream_key] = "carry_forward"
                                    carry_forward_count += 1
                                else:
                                    # Too old for carry-forward
                                    window_config_sources[stream_key] = "disabled_no_cf"
                            else:
                                # No history - keep disabled
                                window_config_sources[stream_key] = "disabled_no_history"
                        else:
                            # Optimizer found good config
                            window_config_sources[stream_key] = "optimized"

                if carry_forward_count > 0:
                    log(f"   📋 PR-2: {carry_forward_count} carry-forward configs applied")

                config_source_log.append({
                    "window_id": current_window_id,
                    "sources": window_config_sources.copy(),
                })

        # 4.4 Run backtest on trade period
        log(f"   📊 Backtest çalıştırılıyor: {window['trade_start'].strftime('%m/%d')} → {window['trade_end'].strftime('%m/%d')}")

        # Filter streams to include trade period (need some data before for indicators)
        trade_streams = {}
        buffer_start = window["trade_start"] - timedelta(days=15)  # Buffer for indicators
        for (sym, tf), df in streams.items():
            filtered = filter_data_by_date(df, buffer_start, window["trade_end"])
            if len(filtered) >= 250:
                trade_streams[(sym, tf)] = filtered

        if trade_streams and config_map:
            result = run_window_backtest(
                trade_streams, config_map,
                window["trade_start"], window["trade_end"],
                carried_positions, log
            )

            # Update carried positions for next window
            carried_positions = result["open_positions"]
            running_equity = result["final_equity"]

            # Collect results
            all_trades.extend(result["closed_trades"])
            equity_curve.append({
                "window_id": window["window_id"],
                "date": window["trade_end"].strftime("%Y-%m-%d"),
                "equity": running_equity,
                "pnl": result["pnl"],
            })

            # PR-2: Update good_configs_history based on this window's per-stream results
            if carry_forward_enabled:
                # Group closed trades by stream
                stream_perf = {}  # (sym, tf) -> {"trades": N, "wins": M, "pnl": $X}
                for trade in result["closed_trades"]:
                    stream_key = (trade.get("symbol"), trade.get("timeframe"))
                    if stream_key not in stream_perf:
                        stream_perf[stream_key] = {"trades": 0, "wins": 0, "pnl": 0.0}
                    stream_perf[stream_key]["trades"] += 1
                    stream_perf[stream_key]["pnl"] += float(trade.get("pnl", 0))
                    if float(trade.get("pnl", 0)) > 0:
                        stream_perf[stream_key]["wins"] += 1

                # Update history for streams that had good performance
                for stream_key, perf in stream_perf.items():
                    trades = perf["trades"]
                    wins = perf["wins"]
                    pnl = perf["pnl"]
                    win_rate = wins / trades if trades > 0 else 0

                    # Check if meets "good" criteria
                    if trades >= min_trades_for_good and pnl >= min_pnl_for_good and win_rate >= min_win_rate_for_good:
                        # This is a good config - save for future carry-forward
                        cfg = config_map.get(stream_key, {}).copy()
                        if not cfg.get("disabled"):
                            good_configs_history[stream_key] = {
                                "config": cfg,
                                "window_id": window["window_id"],
                                "pnl": pnl,
                                "trades": trades,
                                "win_rate": win_rate,
                            }

            window_result = {
                "window_id": window["window_id"],
                "trade_start": window["trade_start"].strftime("%Y-%m-%d"),
                "trade_end": window["trade_end"].strftime("%Y-%m-%d"),
                "pnl": result["pnl"],
                "trades": result["trades"],  # Legacy: leg count
                "wins": result["wins"],  # Legacy: leg-based wins
                # === STAGE 0: Position vs Leg separation ===
                "positions_count": result.get("positions_count", result["trades"]),
                "legs_count": result.get("legs_count", result["trades"]),
                "positions_wins": result.get("positions_wins", result["wins"]),
                "force_closed_count": result.get("force_closed_count", 0),
                "max_dd": result["max_dd"],
                "config_used": {f"{s}-{t}": c.get("rr", "-") for (s, t), c in config_map.items() if not c.get("disabled")},
                "open_positions_count": len(carried_positions),  # Should be 0 after force-close
                # PR-2: Loss limit and stream disable tracking
                "loss_limit_hit": result.get("loss_limit_hit", False),
                "streams_disabled": result.get("streams_disabled", []),
                # PR-2: Config source tracking
                "config_sources": window_config_sources,
                # Regime detection
                "regime": dominant_regime,
                "regime_distribution": regime_counts,
            }

            # AŞAMA 8: 0-trade window reason logging (sinyal gelmedi case)
            if result["trades"] == 0:
                enabled_count = len([c for c in config_map.values() if not c.get("disabled")])
                window_result["zero_trade_reason"] = "no_signals"
                window_result["zero_trade_details"] = f"Backtest çalıştı ama sinyal gelmedi. {enabled_count} config aktifti."
                log(f"   ⚠️ 0 trade: Sinyal gelmedi ({enabled_count} config aktifti)")
            else:
                pos_count = result.get("positions_count", result["trades"])
                leg_count = result.get("legs_count", result["trades"])
                pos_wins = result.get("positions_wins", result["wins"])
                force_closed = result.get("force_closed_count", 0)
                log(f"   ✓ PnL=${result['pnl']:.2f}, Positions={pos_count}, Legs={leg_count}, Wins={pos_wins}" +
                    (f", ForceClose={force_closed}" if force_closed > 0 else ""))

        else:
            # AŞAMA 8: 0-trade window reason analizi
            zero_reason = "unknown"
            zero_details = ""
            if not config_map:
                zero_reason = "no_config"
                zero_details = "Optimizer config bulamadı veya tüm configler disabled"
            elif not trade_streams:
                zero_reason = "no_streams"
                zero_details = "Trade dönemi için yeterli veri yok"
            else:
                # Hem config hem stream var ama backtest çalışmadı
                all_disabled = all(c.get("disabled", False) for c in config_map.values())
                if all_disabled:
                    zero_reason = "all_disabled"
                    zero_details = "Tüm stream configleri disabled olarak işaretli"
                else:
                    zero_reason = "backtest_error"
                    zero_details = "Backtest çalıştırılamadı (bilinmeyen hata)"

            window_result = {
                "window_id": window["window_id"],
                "trade_start": window["trade_start"].strftime("%Y-%m-%d"),
                "trade_end": window["trade_end"].strftime("%Y-%m-%d"),
                "pnl": 0.0,
                "trades": 0,
                "wins": 0,
                "max_dd": 0.0,
                "config_used": {},
                "skipped": True,
                # AŞAMA 8: 0-trade window reason logging
                "zero_trade_reason": zero_reason,
                "zero_trade_details": zero_details,
                # Regime detection
                "regime": dominant_regime,
                "regime_distribution": regime_counts,
            }
            log(f"   ⚠️ Backtest atlandı: {zero_reason} - {zero_details}")

        all_window_results.append(window_result)

    # ==========================================
    # 5. CALCULATE STITCHED METRICS
    # ==========================================
    total_pnl = sum(w["pnl"] for w in all_window_results)
    total_trades = sum(w["trades"] for w in all_window_results)  # Legacy: legs
    total_wins = sum(w["wins"] for w in all_window_results)  # Legacy: leg-based

    # === STAGE 0: Position vs Leg metrics ===
    total_positions = sum(w.get("positions_count", w["trades"]) for w in all_window_results)
    total_legs = sum(w.get("legs_count", w["trades"]) for w in all_window_results)
    total_positions_wins = sum(w.get("positions_wins", w["wins"]) for w in all_window_results)
    total_force_closed = sum(w.get("force_closed_count", 0) for w in all_window_results)

    # === PR-1: SL Widening Metrics (should be 0 in baseline mode) ===
    sl_widened_count = 0
    sl_widened_pnl = 0.0
    sl_not_widened_pnl = 0.0
    sl_rejected_count = 0
    for trade in all_trades:
        if trade.get("sl_widened"):
            sl_widened_count += 1
            sl_widened_pnl += float(trade.get("pnl", 0))
        elif trade.get("status") == "SL_TOO_TIGHT_REJECTED":
            sl_rejected_count += 1
        else:
            sl_not_widened_pnl += float(trade.get("pnl", 0))

    window_pnls = [w["pnl"] for w in all_window_results]
    positive_windows = sum(1 for p in window_pnls if p > 0)
    hit_rate = positive_windows / len(window_pnls) if window_pnls else 0

    # Calculate max drawdown from equity curve
    initial_balance = TRADING_CONFIG["initial_balance"]
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    if equity_curve:
        peak_equity = initial_balance
        for point in equity_curve:
            equity = point["equity"]
            peak_equity = max(peak_equity, equity)
            drawdown = peak_equity - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown / peak_equity if peak_equity > 0 else 0

    # Also consider per-window max drawdowns
    total_window_max_dd = max((w.get("max_dd", 0) for w in all_window_results), default=0)

    # === PR-2: Tail-risk and carry-forward metrics ===
    pr2_loss_limit_windows = sum(1 for w in all_window_results if w.get("loss_limit_hit", False))
    pr2_total_streams_disabled = 0
    for w in all_window_results:
        pr2_total_streams_disabled += len(w.get("streams_disabled", []))

    # Config source breakdown
    pr2_sources = {"optimized": 0, "carry_forward": 0, "carry_forward_precal": 0, "disabled_no_cf": 0, "disabled_no_history": 0}
    for log_entry in config_source_log:
        for stream_key, source in log_entry.get("sources", {}).items():
            if source in pr2_sources:
                pr2_sources[source] += 1

    # Calculate carry-forward PnL contribution
    pr2_carry_forward_pnl = 0.0
    for trade in all_trades:
        cfg_snapshot = trade.get("config_snapshot", {})
        if cfg_snapshot.get("carry_forward"):
            pr2_carry_forward_pnl += float(trade.get("pnl", 0))

    metrics = {
        "mode": mode,
        "total_pnl": total_pnl,
        "total_trades": total_trades,  # Legacy: legs count (backward compat)
        "win_rate": total_wins / total_trades if total_trades > 0 else 0,
        # === STAGE 0: Position-based metrics ===
        "positions_count": total_positions,
        "legs_count": total_legs,
        "positions_win_rate": total_positions_wins / total_positions if total_positions > 0 else 0,
        "force_closed_count": total_force_closed,
        # === PR-1: SL Widening Metrics ===
        "sl_widened_count": sl_widened_count,
        "sl_widened_pnl": sl_widened_pnl,
        "sl_not_widened_pnl": sl_not_widened_pnl,
        "sl_rejected_count": sl_rejected_count,
        "max_drawdown": max(max_drawdown, total_window_max_dd),
        "max_drawdown_pct": max_drawdown_pct,
        "window_count": len(all_window_results),
        "positive_windows": positive_windows,
        "window_hit_rate": hit_rate,
        "median_window_pnl": sorted(window_pnls)[len(window_pnls)//2] if window_pnls else 0,
        "worst_window_pnl": min(window_pnls) if window_pnls else 0,
        "best_window_pnl": max(window_pnls) if window_pnls else 0,
        "final_equity": running_equity,
        # === PR-2: Tail-risk metrics ===
        "pr2_loss_limit_windows": pr2_loss_limit_windows,
        "pr2_streams_disabled_total": pr2_total_streams_disabled,
        "pr2_config_sources": pr2_sources,
        "pr2_carry_forward_pnl": pr2_carry_forward_pnl,
    }

    # Calculate regime stats for metrics
    regime_window_counts = {}
    regime_pnl = {}
    for w in all_window_results:
        r = w.get("regime", "UNKNOWN")
        regime_window_counts[r] = regime_window_counts.get(r, 0) + 1
        regime_pnl[r] = regime_pnl.get(r, 0) + w.get("pnl", 0)

    metrics["regime_window_counts"] = regime_window_counts
    metrics["regime_pnl"] = regime_pnl

    # ==========================================
    # 6. GENERATE REPORT
    # ==========================================
    log(f"\n{'='*70}")
    log(f"📊 ROLLING WALK-FORWARD SONUÇLARI ({mode.upper()})")
    log(f"{'='*70}")
    log(f"   Stitched OOS PnL: ${total_pnl:.2f}")
    log(f"   Positions: {total_positions} (Legs: {total_legs})")
    log(f"   Position Win Rate: {metrics['positions_win_rate']*100:.1f}%")
    if total_force_closed > 0:
        log(f"   Force-Closed at Window End: {total_force_closed}")
    log(f"   Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']*100:.1f}%)")
    log(f"\n   Window Stats ({len(all_window_results)} windows):")
    log(f"   ├── Hit Rate: {hit_rate*100:.1f}% ({positive_windows}/{len(all_window_results)} pozitif)")
    log(f"   ├── Median PnL: ${metrics['median_window_pnl']:.2f}")
    log(f"   ├── Worst Window: ${metrics['worst_window_pnl']:.2f}")
    log(f"   └── Best Window: ${metrics['best_window_pnl']:.2f}")

    # Regime summary (use pre-calculated from metrics)
    log(f"\n   📊 Regime Analysis:")
    for regime in ["TRENDING", "RANGING", "VOLATILE", "TRANSITIONAL", "UNKNOWN"]:
        if regime in regime_window_counts:
            count = regime_window_counts[regime]
            pnl = regime_pnl.get(regime, 0)
            log(f"   ├── {regime}: {count} windows, PnL=${pnl:.2f}")

    # PR-1: SL Widening diagnostics (should be 0 in baseline mode)
    if sl_widened_count > 0 or sl_rejected_count > 0:
        log(f"\n   ⚠️ SL Validation (Non-Baseline Mode):")
        log(f"   ├── SL Widened: {sl_widened_count} trades (PnL: ${sl_widened_pnl:.2f})")
        log(f"   ├── SL Rejected: {sl_rejected_count} trades")
        log(f"   └── Not Widened PnL: ${sl_not_widened_pnl:.2f}")
    else:
        log(f"\n   ✓ PR-1 Baseline Mode: No SL widening (sl_validation_mode=off)")

    # PR-2: Tail-risk and carry-forward metrics
    log(f"\n   📋 PR-2 Metrics:")
    log(f"   ├── Loss Limit Windows: {pr2_loss_limit_windows}")
    log(f"   ├── Streams Disabled (fullstop): {pr2_total_streams_disabled}")
    log(f"   ├── Config Sources: opt={pr2_sources['optimized']}, cf={pr2_sources['carry_forward']}, precal={pr2_sources['carry_forward_precal']}")
    log(f"   ├── Disabled: no_cf={pr2_sources['disabled_no_cf']}, no_hist={pr2_sources['disabled_no_history']}")
    log(f"   └── Carry-Forward PnL: ${pr2_carry_forward_pnl:.2f}")
    log(f"{'='*70}")

    # Save results to output_dir
    result = {
        "run_id": run_id,
        "mode": mode,
        "output_dir": output_dir,  # Include output_dir for trade log writer
        "config": {
            "lookback_days": lookback_days,
            "forward_days": forward_days,
            "start_date": start_date,
            "end_date": end_date,
        },
        "metrics": metrics,
        "window_results": all_window_results,
        "trades": all_trades,
        "equity_curve": equity_curve,
        # PR-2: Config source tracking for reporting
        "config_source_log": config_source_log,
    }

    # Save JSON report
    report_path = os.path.join(output_dir, "report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        log(f"\n💾 Rapor kaydedildi: {report_path}")
    except Exception as e:
        log(f"⚠️ Rapor kaydetme hatası: {e}")

    return result


def compare_rolling_modes(
    symbols: list = None,
    timeframes: list = None,
    start_date: str = None,
    end_date: str = None,
    fixed_config: dict = None,
    verbose: bool = True,
    modes: list = None,  # PR-2: Filter modes (default: all 4)
) -> dict:
    """Compare Fixed vs Monthly vs Weekly vs Triday re-optimization modes.

    Bu fonksiyon 4 modu aynı veri üzerinde çalıştırır:
    1. Fixed: Tek sabit config (calibration dönemi ile belirlenir)
    2. Monthly: Aylık re-optimization (60 gün lookback, 30 gün forward)
    3. Weekly: Haftalık re-optimization (30 gün lookback, 7 gün forward)
    4. Triday: 3 günlük re-optimization (60 gün lookback, 3 gün forward)

    Args:
        symbols: Test edilecek semboller
        timeframes: Test edilecek zaman dilimleri
        start_date: Test dönemi başlangıcı
        end_date: Test dönemi sonu
        fixed_config: Fixed mode için kullanılacak config (None = calibration ile bulunur)
        verbose: Detaylı çıktı
        modes: Test edilecek modlar (default: all 4) - PR-2

    Returns:
        dict with comparison results for all 3 modes
    """

    def log(msg: str):
        if verbose:
            print(msg)

    # PR-2: Mode config definitions (v40.6: monthly/triday removed)
    all_mode_configs = [
        ("fixed", {"mode": "fixed", "fixed_config": fixed_config}),
        ("weekly", {"mode": "weekly", "lookback_days": 60, "forward_days": 7}),
    ]

    # PR-2: Filter modes if specified
    if modes:
        modes_lower = [m.lower() for m in modes]
        mode_configs = [(name, cfg) for name, cfg in all_mode_configs if name in modes_lower]
        mode_names = [m.upper() for m in modes_lower]
    else:
        mode_configs = all_mode_configs
        mode_names = ["Fixed", "Weekly"]

    log(f"\n{'='*70}")
    log(f"🔬 ROLLING WALK-FORWARD KARŞILAŞTIRMA")
    log(f"{'='*70}")
    log(f"   Modlar: {' vs '.join(mode_names)}")
    log(f"{'='*70}\n")

    results = {}
    total_modes = len(mode_configs)

    for i, (mode_name, mode_cfg) in enumerate(mode_configs, 1):
        log(f"📊 [{i}/{total_modes}] {mode_name.upper()} mode çalıştırılıyor...")
        results[mode_name] = run_rolling_walkforward(
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
            **mode_cfg,
        )
        if i < total_modes:
            log("")  # Empty line between modes

    # ==========================================
    # COMPARISON REPORT (PR-2: Dynamic based on modes)
    # ==========================================
    run_modes = list(results.keys())

    # If only one mode, skip comparison
    if len(run_modes) == 1:
        single_mode = run_modes[0]
        single_result = results[single_mode]
        log(f"\n{'='*70}")
        log(f"📊 {single_mode.upper()} MODE SONUÇLARI")
        log(f"{'='*70}")
        log(f"   Total PnL: ${single_result['metrics']['total_pnl']:.2f}")
        log(f"   Max DD: ${single_result['metrics']['max_drawdown']:.2f}")
        log(f"   Window Hit Rate: {single_result['metrics']['window_hit_rate']*100:.1f}%")
        log(f"   Worst Window: ${single_result['metrics']['worst_window_pnl']:.2f}")
        log(f"{'='*70}\n")

        return {
            "results": results,
            "comparison": {
                "pnl": {single_mode: single_result['metrics']['total_pnl']},
                "max_dd": {single_mode: single_result['metrics']['max_drawdown']},
                "hit_rate": {single_mode: single_result['metrics']['window_hit_rate']},
                "worst_window": {single_mode: single_result['metrics']['worst_window_pnl']},
                "scores": {single_mode: 0},
                "best_mode": single_mode,
            }
        }

    log(f"\n{'='*70}")
    log(f"📊 KARŞILAŞTIRMA SONUÇLARI")
    log(f"{'='*70}")

    # Dynamic headers based on modes run
    headers = ["Metrik"] + [m.capitalize() for m in run_modes] + ["En İyi"]
    rows = []

    # PnL comparison
    pnls = {m: results[m]["metrics"]["total_pnl"] for m in run_modes}
    best_pnl = max(pnls, key=pnls.get)
    rows.append(["Total PnL"] + [f"${pnls[m]:.2f}" for m in run_modes] + [best_pnl.upper()])

    # Max DD comparison
    dds = {m: results[m]["metrics"]["max_drawdown"] for m in run_modes}
    best_dd = min(dds, key=lambda x: abs(dds[x]))  # Lowest absolute DD is best
    rows.append(["Max DD"] + [f"${dds[m]:.2f}" for m in run_modes] + [best_dd.upper()])

    # Window Hit Rate comparison
    hit_rates = {m: results[m]["metrics"]["window_hit_rate"] for m in run_modes}
    best_hr = max(hit_rates, key=hit_rates.get)
    rows.append(["Window Hit Rate"] + [f"{hit_rates[m]*100:.1f}%" for m in run_modes] + [best_hr.upper()])

    # Worst Window comparison
    worst = {m: results[m]["metrics"]["worst_window_pnl"] for m in run_modes}
    best_worst = max(worst, key=worst.get)  # Highest (least negative) is best
    rows.append(["Worst Window"] + [f"${worst[m]:.2f}" for m in run_modes] + [best_worst.upper()])

    # Print table with dynamic column widths
    col_widths = [20] + [12] * len(run_modes) + [10]
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    log(header_line)
    log("─" * sum(col_widths))
    for row in rows:
        log("".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

    # ==========================================
    # DECISION MATRIX
    # ==========================================
    log(f"\n{'='*70}")
    log("🎯 KARAR MATRİSİ")
    log("─" * 70)

    # Determine best mode based on multiple criteria
    scores = {m: 0 for m in run_modes}
    scores[best_pnl] += 2  # PnL worth 2 points
    scores[best_dd] += 1   # DD worth 1 point
    scores[best_hr] += 1   # Hit rate worth 1 point
    scores[best_worst] += 1  # Worst window worth 1 point

    best_mode = max(scores, key=scores.get)

    scores_str = ", ".join(f"{m.capitalize()}={scores[m]}" for m in run_modes)
    log(f"   Puanlar: {scores_str}")
    log(f"\n   🏆 ÖNERİLEN MOD: {best_mode.upper()}")

    # Specific recommendations (v40.6: monthly/triday removed)
    recommendations = {
        "fixed": "   → Piyasa stabil görünüyor, re-optimization gereksiz karmaşıklık ekliyor",
        "weekly": "   → Haftalık re-opt en iyi - piyasa hızlı değişiyor\n   → DİKKAT: Overfit riski yüksek, dikkatli izlenmeli",
    }
    if best_mode in recommendations:
        log(recommendations[best_mode])

    # Check if ANY mode is profitable
    if all(pnls[m] <= 0 for m in pnls):
        log("\n   ⚠️ UYARI: Hiçbir mod kârlı değil!")
        log("   → Edge çok zayıf veya strateji bu piyasa rejimine uygun değil")
        log("   → Live'a bağlamak önerilmez")

    log(f"{'='*70}\n")

    return {
        "results": results,
        "comparison": {
            "pnl": pnls,
            "max_dd": dds,
            "hit_rate": hit_rates,
            "worst_window": worst,
            "scores": scores,
            "best_mode": best_mode,
        }
    }


# ============================================================================
# PERFORMANCE-OPTIMIZED ROLLING WALK-FORWARD (v40.x)
# ============================================================================

def compare_rolling_modes_fast(
    symbols: list = None,
    timeframes: list = None,
    start_date: str = None,
    end_date: str = None,
    fixed_config: dict = None,
    verbose: bool = True,
    quick: bool = False,
    parallel_modes: bool = True,
    modes: list = None,  # PR-2: Filter modes (default: all 4)
) -> dict:
    """
    Performance-optimized compare_rolling_modes with master cache and parallel execution.

    Key optimizations:
    1. Master data cache: Fetch all data once, share across modes
    2. Parallel mode execution: Run all 4 modes concurrently (when parallel_modes=True)
    3. NumPy-based slicing: Avoid DataFrame copies for window extraction
    4. Quick mode: Reduced symbols/timeframes for faster testing

    Args:
        symbols: Test edilecek semboller
        timeframes: Test edilecek zaman dilimleri
        start_date: Test dönemi başlangıcı
        end_date: Test dönemi sonu
        fixed_config: Fixed mode için kullanılacak config
        verbose: Detaylı çıktı
        quick: Hızlı mod - sembol/pencere azaltma (default: False)
        parallel_modes: Modları paralel çalıştır (default: True)

    Returns:
        dict with comparison results for all 4 modes
    """
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    import multiprocessing

    def log(msg: str):
        if verbose:
            print(msg)

    # ==========================================
    # QUICK MODE: Reduce scope for faster testing
    # ==========================================
    if quick:
        # Limit to 3 symbols and 3 timeframes for quick testing
        if symbols is None:
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        else:
            symbols = symbols[:3]
        if timeframes is None:
            timeframes = ["15m", "1h", "4h"]
        else:
            timeframes = timeframes[:3]
        log("🚀 QUICK MODE: Using reduced scope for faster testing")

    if symbols is None:
        symbols = SYMBOLS
    if timeframes is None:
        timeframes = TIMEFRAMES

    log(f"\n{'='*70}")
    log(f"🔬 ROLLING WALK-FORWARD KARŞILAŞTIRMA (OPTIMIZED)")
    log(f"{'='*70}")
    log(f"   Modlar: Fixed vs Monthly vs Weekly vs Triday")
    log(f"   Parallel: {parallel_modes}")
    log(f"   Quick: {quick}")
    log(f"   Symbols: {len(symbols)}, Timeframes: {len(timeframes)}")
    log(f"{'='*70}\n")

    # ==========================================
    # STEP 1: MASTER DATA CACHE
    # ==========================================
    # Import performance cache module
    try:
        from core.perf_cache import MasterDataCache, clear_disk_cache
    except ImportError:
        log("⚠️ perf_cache module not available, using standard mode")
        return compare_rolling_modes(symbols, timeframes, start_date, end_date, fixed_config, verbose)

    # Determine date range
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_dt = datetime.now() - timedelta(days=365)
        start_date = start_dt.strftime("%Y-%m-%d")

    log(f"📥 Master data cache yükleniyor...")
    log(f"   Period: {start_date} → {end_date}")

    # Create fetch function wrapper
    def fetch_func(sym, tf, start, end):
        return TradingEngine.get_historical_data_pagination(sym, tf, start_date=start, end_date=end)

    # Create master cache
    master_cache = MasterDataCache(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        fetch_func=fetch_func,
        indicator_func=TradingEngine.calculate_indicators,
        buffer_days=60,  # Enough for 60-day lookback
    )

    # Load all data
    loaded_count = master_cache.load_all(
        progress_callback=lambda loaded, total: log(f"   Loading: {loaded}/{total}") if loaded % 10 == 0 else None
    )
    log(f"   ✓ {loaded_count} stream yüklendi")

    # ==========================================
    # STEP 2: RUN MODES (Parallel or Sequential)
    # ==========================================
    # v40.6: monthly/triday removed
    all_mode_configs = [
        ("fixed", {"mode": "fixed", "fixed_config": fixed_config}),
        ("weekly", {"mode": "weekly", "lookback_days": 60, "forward_days": 7}),
    ]

    # PR-2: Filter modes if specified
    if modes:
        modes_lower = [m.lower() for m in modes]
        mode_configs = [(name, cfg) for name, cfg in all_mode_configs if name in modes_lower]
        log(f"\n📊 PR-2: Sadece {', '.join([m.upper() for m in modes_lower])} modu çalıştırılıyor...")
    else:
        mode_configs = all_mode_configs

    results = {}

    if parallel_modes and not quick:
        # Parallel execution using ThreadPoolExecutor
        # Note: ProcessPoolExecutor would be faster but has pickling issues with nested functions
        log("\n📊 Modlar paralel çalıştırılıyor...")
        log(f"   Modlar: {', '.join([m[0].upper() for m in mode_configs])}")
        log(f"   ⏳ Her mod 5-15 dakika sürebilir, lütfen bekleyin...")

        # Progress tracking
        import threading
        import time as time_module
        completed_modes = []
        running_modes = [m[0] for m in mode_configs]
        start_time = time_module.time()
        stop_progress = threading.Event()

        def progress_monitor():
            """Print progress every 30 seconds to show the test is still running."""
            while not stop_progress.is_set():
                stop_progress.wait(30)  # Wait 30 seconds or until stopped
                if stop_progress.is_set():
                    break
                elapsed = int(time_module.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                remaining = [m for m in running_modes if m not in completed_modes]
                if remaining:
                    log(f"   ⏳ [{mins}:{secs:02d}] Çalışıyor: {', '.join([m.upper() for m in remaining])} | Tamamlanan: {len(completed_modes)}/{len(running_modes)}")

        # Start progress monitor thread
        progress_thread = threading.Thread(target=progress_monitor, daemon=True)
        progress_thread.start()

        def run_mode(mode_name: str, mode_params: dict):
            try:
                result = run_rolling_walkforward(
                    symbols=symbols,
                    timeframes=timeframes,
                    start_date=start_date,
                    end_date=end_date,
                    verbose=False,  # Disable verbose in parallel mode
                    **mode_params,
                )
                return (mode_name, result)
            except Exception as e:
                import traceback
                error_msg = f"{e}\n{traceback.format_exc()}"
                return (mode_name, {"error": error_msg, "metrics": {"total_pnl": 0, "max_drawdown": 0, "window_hit_rate": 0, "worst_window_pnl": 0}})

        # Use 2 workers max to avoid memory issues
        max_workers = min(2, len(mode_configs))
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_mode, name, params): name for name, params in mode_configs}

                for future in as_completed(futures):
                    mode_name, result = future.result()
                    results[mode_name] = result
                    completed_modes.append(mode_name)

                    elapsed = int(time_module.time() - start_time)
                    mins, secs = divmod(elapsed, 60)

                    if result.get("error"):
                        log(f"   ❌ [{mins}:{secs:02d}] {mode_name.upper()} mode HATA: {result['error'][:100]}...")
                    else:
                        pnl = result.get('metrics', {}).get('total_pnl', 0)
                        trades = result.get('metrics', {}).get('total_trades', 0)
                        log(f"   ✓ [{mins}:{secs:02d}] {mode_name.upper()} mode tamamlandı: PnL=${pnl:.2f}, Trades={trades}")
        finally:
            # Stop progress monitor
            stop_progress.set()
            progress_thread.join(timeout=1)

    else:
        # Sequential execution (original behavior)
        for i, (mode_name, mode_params) in enumerate(mode_configs, 1):
            log(f"\n📊 [{i}/4] {mode_name.upper()} mode çalıştırılıyor...")
            results[mode_name] = run_rolling_walkforward(
                symbols=symbols,
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date,
                verbose=verbose,
                **mode_params,
            )

    # ==========================================
    # STEP 3: COMPARISON (PR-2: Dynamic based on modes)
    # ==========================================
    run_modes = list(results.keys())

    # If only one mode, skip comparison
    if len(run_modes) == 1:
        single_mode = run_modes[0]
        single_result = results[single_mode]
        metrics = single_result.get("metrics", {})
        log(f"\n{'='*70}")
        log(f"📊 {single_mode.upper()} MODE SONUÇLARI")
        log(f"{'='*70}")
        log(f"   Total PnL: ${metrics.get('total_pnl', 0):.2f}")
        log(f"   Max DD: ${metrics.get('max_drawdown', 0):.2f}")
        log(f"   Window Hit Rate: {metrics.get('window_hit_rate', 0)*100:.1f}%")
        log(f"   Worst Window: ${metrics.get('worst_window_pnl', 0):.2f}")

        # PR-2 Metrics
        log(f"\n   📋 PR-2 Metrics:")
        log(f"   ├── Loss Limit Windows: {metrics.get('pr2_loss_limit_windows', 0)}")
        log(f"   ├── Streams Disabled (fullstop): {metrics.get('pr2_streams_disabled_total', 0)}")
        pr2_sources = metrics.get("pr2_config_sources", {})
        log(f"   ├── Config Sources: opt={pr2_sources.get('optimized', 0)}, cf={pr2_sources.get('carry_forward', 0)}, precal={pr2_sources.get('carry_forward_precal', 0)}")
        log(f"   └── Carry-Forward PnL: ${metrics.get('pr2_carry_forward_pnl', 0):.2f}")
        log(f"{'='*70}\n")

        return {
            "results": results,
            "comparison": {
                "pnl": {single_mode: metrics.get('total_pnl', 0)},
                "max_dd": {single_mode: metrics.get('max_drawdown', 0)},
                "hit_rate": {single_mode: metrics.get('window_hit_rate', 0)},
                "worst_window": {single_mode: metrics.get('worst_window_pnl', 0)},
                "scores": {single_mode: 0},
                "best_mode": single_mode,
            }
        }

    log(f"\n{'='*70}")
    log(f"📊 KARŞILAŞTIRMA SONUÇLARI")
    log(f"{'='*70}")

    # Dynamic headers based on modes run
    headers = ["Metrik"] + [m.capitalize() for m in run_modes] + ["En İyi"]
    rows = []

    # PnL comparison
    pnls = {m: results[m].get("metrics", {}).get("total_pnl", 0) for m in run_modes}
    best_pnl = max(pnls, key=pnls.get)
    rows.append(["Total PnL"] + [f"${pnls[m]:.2f}" for m in run_modes] + [best_pnl.upper()])

    # Max DD comparison
    dds = {m: results[m].get("metrics", {}).get("max_drawdown", 0) for m in run_modes}
    best_dd = min(dds, key=lambda x: abs(dds[x]))
    rows.append(["Max DD"] + [f"${dds[m]:.2f}" for m in run_modes] + [best_dd.upper()])

    # Window Hit Rate
    hit_rates = {m: results[m].get("metrics", {}).get("window_hit_rate", 0) for m in run_modes}
    best_hr = max(hit_rates, key=hit_rates.get)
    rows.append(["Window Hit Rate"] + [f"{hit_rates[m]*100:.1f}%" for m in run_modes] + [best_hr.upper()])

    # Worst Window
    worst = {m: results[m].get("metrics", {}).get("worst_window_pnl", 0) for m in run_modes}
    best_worst = max(worst, key=worst.get)
    rows.append(["Worst Window"] + [f"${worst[m]:.2f}" for m in run_modes] + [best_worst.upper()])

    # Print table with dynamic column widths
    col_widths = [20] + [12] * len(run_modes) + [10]
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    log(header_line)
    log("─" * sum(col_widths))
    for row in rows:
        log("".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

    # Decision Matrix
    scores = {m: 0 for m in run_modes}
    scores[best_pnl] += 2
    scores[best_dd] += 1
    scores[best_hr] += 1
    scores[best_worst] += 1
    best_mode = max(scores, key=scores.get)

    log(f"\n🎯 ÖNERİLEN MOD: {best_mode.upper()} (Score: {scores[best_mode]})")

    return {
        "results": results,
        "comparison": {
            "pnl": pnls,
            "max_dd": dds,
            "hit_rate": hit_rates,
            "worst_window": worst,
            "scores": scores,
            "best_mode": best_mode,
        }
    }


def run_quick_rolling_test(
    symbols: list = None,
    timeframes: list = None,
    mode: str = "weekly",  # v40.6: default changed from monthly to weekly
    days: int = 90,
    verbose: bool = True,
) -> dict:
    """
    Quick rolling walk-forward test for fast iteration.

    Reduces scope for faster testing:
    - Limited symbols (default: 3)
    - Limited timeframes (default: 3)
    - Shorter period (default: 90 days)

    Args:
        symbols: Symbols to test (default: BTCUSDT, ETHUSDT, SOLUSDT)
        timeframes: Timeframes to test (default: 15m, 1h, 4h)
        mode: Mode to test (default: weekly)
        days: Number of days to test (default: 90)
        verbose: Show detailed output

    Returns:
        Rolling WF result dict
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    if timeframes is None:
        timeframes = ["15m", "1h", "4h"]

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"\n🚀 QUICK ROLLING TEST")
    print(f"   Mode: {mode}, Period: {days} days")
    print(f"   Symbols: {symbols}")
    print(f"   Timeframes: {timeframes}\n")

    return run_rolling_walkforward(
        symbols=symbols,
        timeframes=timeframes,
        mode=mode,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose,
    )


