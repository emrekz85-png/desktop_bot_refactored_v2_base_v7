#!/usr/bin/env python3
"""
FILTER COMBINATION TESTER

AYNI check_core_signal fonksiyonunu kullanır ki AT Scenario ile tutarlı olsun.
Baseline: 1659 sinyal (AT Scenario'dan)
Regime filter ile: ~1481 sinyal

Kullanım:
    python runners/run_filter_combo_test.py
    python runners/run_filter_combo_test.py --incremental
    python runners/run_filter_combo_test.py --full-scan
"""

import sys
import os
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from itertools import combinations
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import calculate_indicators, get_client
# ORIJINAL fonksiyonu import et!
from core.at_scenario_analyzer import check_core_signal


def apply_filters(
    df: pd.DataFrame,
    index: int,
    signal_type: str,
    # Filter flags - EXISTING
    use_regime_filter: bool = True,
    use_at_binary: bool = False,
    use_at_flat_filter: bool = False,
    use_adx_filter: bool = False,
    use_ssl_touch: bool = False,
    # Filter flags - NEW (from Optuna)
    use_rsi_filter: bool = False,
    use_pbema_distance: bool = False,
    use_overlap_check: bool = False,
    use_body_position: bool = False,
    use_wick_rejection: bool = False,
    # Filter flags - SL DISTANCE (from deep analysis)
    use_min_sl_filter: bool = False,
    # Filter flags - PATTERN FILTERS (Real Trade Analysis 2026-01-04)
    use_momentum_exit: bool = False,
    use_pbema_retest: bool = False,
    use_liquidity_grab: bool = False,
    use_ssl_slope_filter: bool = False,
    use_htf_bounce: bool = False,
    use_momentum_loss: bool = False,
    use_ssl_dynamic_support: bool = False,
    # Signal values (needed for SL filter)
    entry_price: float = None,
    sl_price: float = None,
    # Parameters
    adx_min: float = 15.0,
    regime_lookback: int = 20,
    regime_threshold: float = 0.6,
    ssl_touch_lookback: int = 5,
    rsi_upper: float = 70.0,
    rsi_lower: float = 30.0,
    pbema_min_dist: float = 0.004,
    overlap_min_gap: float = 0.005,
    wick_ratio_max: float = 0.6,
    min_sl_pct: float = 1.5,  # Minimum SL distance in percent
) -> Tuple[bool, str]:
    """
    Apply optional filters to a core signal.
    Returns (pass, reason) - True if signal passes all filters.
    """
    abs_index = index if index >= 0 else (len(df) + index)
    curr = df.iloc[abs_index]

    close = float(curr["close"])
    baseline = float(curr["baseline"])
    pb_top = float(curr.get("pb_ema_top", np.nan))
    pb_bot = float(curr.get("pb_ema_bot", np.nan))

    # ========== FILTER 1: REGIME FILTER ==========
    if use_regime_filter:
        at_regime = "neutral"

        if "at_buyers_dominant" in df.columns and "at_sellers_dominant" in df.columns:
            if abs_index >= regime_lookback:
                start_idx = abs_index - regime_lookback
                buyers_bars = df["at_buyers_dominant"].iloc[start_idx:abs_index].sum()
                sellers_bars = df["at_sellers_dominant"].iloc[start_idx:abs_index].sum()

                buyers_ratio = buyers_bars / regime_lookback
                sellers_ratio = sellers_bars / regime_lookback

                if buyers_ratio >= regime_threshold:
                    at_regime = "bullish"
                elif sellers_ratio >= regime_threshold:
                    at_regime = "bearish"

        if at_regime == "neutral":
            return False, "Regime: Neutral"

    # ========== FILTER 2: AT BINARY (alignment) ==========
    if use_at_binary:
        at_buyers = bool(curr.get("at_buyers_dominant", False))
        at_sellers = bool(curr.get("at_sellers_dominant", False))

        if signal_type == "LONG" and not at_buyers:
            return False, "AT: No Buyers"
        if signal_type == "SHORT" and not at_sellers:
            return False, "AT: No Sellers"

    # ========== FILTER 3: AT FLAT ==========
    if use_at_flat_filter:
        at_flat = bool(curr.get("at_is_flat", False))
        if at_flat:
            return False, "AT: Flat"

    # ========== FILTER 4: ADX FILTER ==========
    if use_adx_filter:
        adx_val = float(curr.get("adx", 0))
        if adx_val < adx_min:
            return False, f"ADX Low ({adx_val:.1f})"

    # ========== FILTER 5: SSL TOUCH ==========
    if use_ssl_touch:
        touch_found = False
        lookback_start = max(0, abs_index - ssl_touch_lookback)

        for i in range(lookback_start, abs_index + 1):
            row = df.iloc[i]
            bl = float(row["baseline"])
            lo, hi = float(row["low"]), float(row["high"])

            if signal_type == "LONG":
                if lo <= bl * 1.003:
                    touch_found = True
                    break
            else:
                if hi >= bl * 0.997:
                    touch_found = True
                    break

        if not touch_found:
            return False, "SSL: No Touch"

    # ========== FILTER 6: RSI FILTER (from Optuna) ==========
    if use_rsi_filter:
        rsi_val = float(curr.get("rsi", 50))
        if signal_type == "LONG" and rsi_val > rsi_upper:
            return False, f"RSI High ({rsi_val:.1f})"
        if signal_type == "SHORT" and rsi_val < rsi_lower:
            return False, f"RSI Low ({rsi_val:.1f})"

    # ========== FILTER 7: PBEMA DISTANCE (from Optuna) ==========
    if use_pbema_distance:
        if not pd.isna(pb_bot) and not pd.isna(pb_top):
            if signal_type == "LONG":
                dist = (pb_bot - close) / close
            else:
                dist = (close - pb_top) / close

            if dist < pbema_min_dist:
                return False, f"PBEMA Close ({dist:.4f})"

    # ========== FILTER 8: OVERLAP CHECK (from Optuna) ==========
    if use_overlap_check:
        if not pd.isna(pb_bot) and not pd.isna(pb_top):
            if signal_type == "LONG":
                gap = (pb_bot - baseline) / baseline
            else:
                gap = (baseline - pb_top) / baseline

            if gap < overlap_min_gap:
                return False, f"SSL-PBEMA Overlap ({gap:.4f})"

    # ========== FILTER 9: BODY POSITION (from Optuna) ==========
    if use_body_position:
        open_price = float(curr["open"])
        high = float(curr["high"])
        low = float(curr["low"])

        body_top = max(open_price, close)
        body_bot = min(open_price, close)
        candle_range = high - low

        if candle_range > 0:
            if signal_type == "LONG":
                # Body should be in upper half for bullish
                body_position = (body_bot - low) / candle_range
                if body_position < 0.3:  # Body too low
                    return False, "Body Position: Low"
            else:
                # Body should be in lower half for bearish
                body_position = (high - body_top) / candle_range
                if body_position < 0.3:  # Body too high
                    return False, "Body Position: High"

    # ========== FILTER 10: WICK REJECTION (from Optuna) ==========
    if use_wick_rejection:
        open_price = float(curr["open"])
        high = float(curr["high"])
        low = float(curr["low"])

        body_size = abs(close - open_price)
        candle_range = high - low

        if candle_range > 0:
            if signal_type == "LONG":
                upper_wick = high - max(open_price, close)
                wick_ratio = upper_wick / candle_range
                if wick_ratio > wick_ratio_max:
                    return False, f"Upper Wick ({wick_ratio:.2f})"
            else:
                lower_wick = min(open_price, close) - low
                wick_ratio = lower_wick / candle_range
                if wick_ratio > wick_ratio_max:
                    return False, f"Lower Wick ({wick_ratio:.2f})"

    # ========== FILTER 11: MINIMUM SL DISTANCE (from deep analysis) ==========
    if use_min_sl_filter:
        if entry_price is not None and sl_price is not None and entry_price > 0:
            if signal_type == "LONG":
                sl_distance_pct = (entry_price - sl_price) / entry_price * 100
            else:
                sl_distance_pct = (sl_price - entry_price) / entry_price * 100

            if sl_distance_pct < min_sl_pct:
                return False, f"SL Too Tight ({sl_distance_pct:.2f}% < {min_sl_pct}%)"

    # ========== PATTERN FILTERS (Real Trade Analysis 2026-01-04) ==========

    # Pattern 3: Liquidity Grab (Entry Enhancement)
    # FIXED: Now REQUIRES a liquidity grab to be detected (was passing when no grab)
    if use_liquidity_grab:
        from core import detect_liquidity_grab
        grab_type, _ = detect_liquidity_grab(df, index)

        # Require liquidity grab to exist AND match signal direction
        if grab_type is None:
            return False, "Liquidity Grab: No Grab Detected"
        if signal_type == "LONG" and grab_type != "LONG_GRAB":
            return False, "Liquidity Grab: Wrong Direction"
        if signal_type == "SHORT" and grab_type != "SHORT_GRAB":
            return False, "Liquidity Grab: Wrong Direction"

    # Pattern 4: SSL Baseline Slope Filter (Anti-Ranging)
    if use_ssl_slope_filter:
        from core import is_ssl_baseline_ranging
        is_ranging, _ = is_ssl_baseline_ranging(df, index)

        if is_ranging:
            return False, "SSL: Ranging Market"

    # Pattern 5: HTF Bounce Detection (Entry Confirmation)
    if use_htf_bounce:
        from core import detect_htf_bounce
        bounce_type, _ = detect_htf_bounce(df, index)

        # Require HTF bounce to match signal direction
        if bounce_type is None:
            return False, "HTF: No Bounce"

        if signal_type == "LONG" and bounce_type != "LONG_BOUNCE":
            return False, "HTF: Wrong Bounce Direction"
        if signal_type == "SHORT" and bounce_type != "SHORT_BOUNCE":
            return False, "HTF: Wrong Bounce Direction"

    # Pattern 6: Momentum Loss After Trend (Counter-Trend Entry)
    if use_momentum_loss:
        from core import detect_momentum_loss_after_trend
        break_type, _ = detect_momentum_loss_after_trend(df, index)

        # Require momentum break to match signal direction
        if break_type is None:
            return False, "Momentum: No Break"

        if signal_type == "LONG" and break_type != "LONG_BREAK":
            return False, "Momentum: Wrong Break Direction"
        if signal_type == "SHORT" and break_type != "SHORT_BREAK":
            return False, "Momentum: Wrong Break Direction"

    # Pattern 7: SSL Dynamic Support/Resistance (Entry Confirmation)
    # FIXED: Now handles both LONG (support) and SHORT (resistance)
    if use_ssl_dynamic_support:
        from core import is_ssl_acting_as_dynamic_support
        is_active, _ = is_ssl_acting_as_dynamic_support(df, index)

        # For LONG: SSL must be acting as support (price bouncing off SSL from above)
        if signal_type == "LONG" and not is_active:
            return False, "SSL: Not Active Support"

        # For SHORT: SSL must NOT be active support (indicates resistance instead)
        # If SSL is strong support, SHORT signals are risky
        if signal_type == "SHORT" and is_active:
            return False, "SSL: Active Support Blocks SHORT"

    # Note: Pattern 1 (momentum_exit) is an EXIT filter, not entry filter
    # Note: Pattern 2 (pbema_retest) is a separate strategy, not a filter

    return True, "OK"


def simulate_trade(df, signal_idx, signal_type, entry, tp, sl, position_size=35.0,
                   use_momentum_exit=False, min_profit_for_momentum=0.005):
    """
    Simulate a single trade with optional momentum-based exit.

    Args:
        df: DataFrame with OHLCV + indicators
        signal_idx: Entry signal index
        signal_type: "LONG" or "SHORT"
        entry: Entry price
        tp: Take profit price
        sl: Stop loss price
        position_size: Position size in dollars
        use_momentum_exit: If True, exit when momentum exhausts (Pattern 1)
        min_profit_for_momentum: Minimum profit % before checking momentum (0.5%)

    Returns:
        dict: Trade result with pnl, win, exit_idx, exit_type
    """
    abs_idx = signal_idx if signal_idx >= 0 else (len(df) + signal_idx)

    # Import momentum exit function if needed
    if use_momentum_exit:
        from core.momentum_exit import should_exit_on_momentum

    for i in range(abs_idx + 1, len(df)):
        candle = df.iloc[i]
        high, low, close = float(candle["high"]), float(candle["low"]), float(candle["close"])

        if signal_type == "LONG":
            # Check SL first
            if low <= sl:
                return {"pnl": (sl - entry) / entry * position_size, "win": False,
                        "exit_idx": i, "exit_type": "SL"}
            # Check TP
            if high >= tp:
                return {"pnl": (tp - entry) / entry * position_size, "win": True,
                        "exit_idx": i, "exit_type": "TP"}

            # Check momentum exit (only if in profit)
            if use_momentum_exit:
                current_profit_pct = (close - entry) / entry
                if current_profit_pct >= min_profit_for_momentum:
                    if should_exit_on_momentum(df, i, signal_type):
                        pnl = (close - entry) / entry * position_size
                        return {"pnl": pnl, "win": pnl > 0,
                                "exit_idx": i, "exit_type": "MOMENTUM"}
        else:  # SHORT
            # Check SL first
            if high >= sl:
                return {"pnl": (entry - sl) / entry * position_size, "win": False,
                        "exit_idx": i, "exit_type": "SL"}
            # Check TP
            if low <= tp:
                return {"pnl": (entry - tp) / entry * position_size, "win": True,
                        "exit_idx": i, "exit_type": "TP"}

            # Check momentum exit (only if in profit)
            if use_momentum_exit:
                current_profit_pct = (entry - close) / entry
                if current_profit_pct >= min_profit_for_momentum:
                    if should_exit_on_momentum(df, i, signal_type):
                        pnl = (entry - close) / entry * position_size
                        return {"pnl": pnl, "win": pnl > 0,
                                "exit_idx": i, "exit_type": "MOMENTUM"}

    # EOD - End of Data
    last = float(df.iloc[-1]["close"])
    if signal_type == "LONG":
        pnl = (last - entry) / entry * position_size
    else:
        pnl = (entry - last) / entry * position_size
    return {"pnl": pnl, "win": pnl > 0, "exit_idx": len(df) - 1, "exit_type": "EOD"}


def run_combo_test(df, filter_flags, min_bars_between=5):  # 5 = AT Scenario default
    """
    Run backtest using ORIGINAL check_core_signal + optional filters.
    Uses SIGNAL index for spacing (like AT Scenario), not exit index.
    """
    trades = []
    signals_found = 0
    signals_filtered = 0
    last_signal_idx = -min_bars_between  # Use SIGNAL idx like AT Scenario!

    for i in range(60, len(df) - 10):  # Start at 60 like AT Scenario
        if i - last_signal_idx < min_bars_between:
            continue

        # Step 1: Get core signal (SAME as AT Scenario!)
        signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)

        if signal_type is None:
            continue

        signals_found += 1
        last_signal_idx = i  # Update on SIGNAL, not exit!

        # Step 2: Apply optional filters (pass entry/sl for min_sl_filter)
        passed, filter_reason = apply_filters(
            df, i, signal_type,
            entry_price=entry,
            sl_price=sl,
            **filter_flags
        )

        if not passed:
            signals_filtered += 1
            continue

        # Step 3: Simulate trade
        trade = simulate_trade(df, i, signal_type, entry, tp, sl)
        trades.append(trade)

    if not trades:
        return {
            "trades": 0, "wins": 0, "wr": 0, "pnl": 0, "dd": 0,
            "signals_found": signals_found, "signals_filtered": signals_filtered
        }

    wins = sum(1 for t in trades if t["win"])
    pnl = sum(t["pnl"] for t in trades)

    # Drawdown
    equity = [0]
    for t in trades:
        equity.append(equity[-1] + t["pnl"])
    peak, dd = 0, 0
    for e in equity:
        peak = max(peak, e)
        dd = max(dd, peak - e)

    return {
        "trades": len(trades),
        "wins": wins,
        "wr": wins / len(trades) * 100,
        "pnl": pnl,
        "dd": dd,
        "signals_found": signals_found,
        "signals_filtered": signals_filtered
    }


def fetch_data(symbol, timeframe, days=365):
    """Fetch data with indicators."""
    client = get_client()

    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
    minutes = tf_minutes.get(timeframe, 15)
    candles = (days * 24 * 60) // minutes + 500

    print(f"Fetching {symbol} {timeframe} ({days} days)...")

    all_dfs = []
    remaining = candles
    end_time = None

    while remaining > 0:
        chunk = min(remaining, 1000)

        if end_time:
            import requests
            url = f"{client.BASE_URL}/klines"
            params = {'symbol': symbol, 'interval': timeframe, 'limit': chunk, 'endTime': end_time}
            res = requests.get(url, params=params, timeout=30)
            if res.status_code != 200 or not res.json():
                break
            df_c = pd.DataFrame(res.json()).iloc[:, :6]
            df_c.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_c['timestamp'] = pd.to_datetime(df_c['timestamp'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
        else:
            df_c = client.get_klines(symbol=symbol, interval=timeframe, limit=chunk)

        if df_c.empty:
            break

        all_dfs.insert(0, df_c)
        remaining -= len(df_c)

        if 'timestamp' in df_c.columns:
            end_time = int(df_c['timestamp'].iloc[0].timestamp() * 1000) - 1
        else:
            break

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    df = calculate_indicators(df, timeframe=timeframe)

    print(f"Got {len(df)} candles")
    return df


def log_combo_result(symbol: str, timeframe: str, days: int, combo_name: str,
                     filter_flags: dict, result: dict, test_type: str = "specific"):
    """
    Log a specific combination test result.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        days: Days of data
        combo_name: Name of the combination (e.g., "REGIME + at_flat + adx")
        filter_flags: The filter flags dict used
        result: The test result dict
        test_type: Type of test (specific, oos, validation)
    """
    import json
    from datetime import datetime as dt

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "filter_combo_logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")

    # Create log entry
    log_entry = {
        "timestamp": timestamp,
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "test_type": test_type,
        "combo_name": combo_name,
        "filter_flags": filter_flags,
        "result": {
            "trades": result.get("trades", 0),
            "wins": result.get("wins", 0),
            "wr": round(result.get("wr", 0), 2),
            "pnl": round(result.get("pnl", 0), 2),
            "dd": round(result.get("dd", 0), 2),
            "signals_found": result.get("signals_found", 0),
            "signals_filtered": result.get("signals_filtered", 0),
        }
    }

    # Append to daily log file
    daily_log = os.path.join(log_dir, f"combo_tests_{symbol}_{timeframe}.jsonl")
    with open(daily_log, 'a') as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"📝 Logged: {combo_name} → {daily_log}")
    return log_entry


def run_specific_combo(symbol: str, timeframe: str, days: int,
                       filter_list: list, log_result: bool = True) -> dict:
    """
    Run a specific filter combination and optionally log it.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        days: Days of data
        filter_list: List of filter names (e.g., ["regime", "at_flat_filter", "adx_filter"])
        log_result: Whether to log the result

    Returns:
        Result dict with trades, wins, wr, pnl, dd, etc.
    """
    # Fetch data
    df = fetch_data(symbol, timeframe, days)
    if df.empty:
        return {"error": "No data"}

    # Build filter flags
    filter_flags = {
        "use_regime_filter": "regime" in filter_list,
        "use_at_binary": "at_binary" in filter_list,
        "use_at_flat_filter": "at_flat_filter" in filter_list,
        "use_adx_filter": "adx_filter" in filter_list,
        "use_ssl_touch": "ssl_touch" in filter_list,
        "use_rsi_filter": "rsi_filter" in filter_list,
        "use_pbema_distance": "pbema_distance" in filter_list,
        "use_overlap_check": "overlap_check" in filter_list,
        "use_body_position": "body_position" in filter_list,
        "use_wick_rejection": "wick_rejection" in filter_list,
        "use_min_sl_filter": "min_sl_filter" in filter_list,
    }

    # Run test
    result = run_combo_test(df, filter_flags)

    # Generate combo name
    active_filters = [f for f in filter_list if f != "regime"]
    if active_filters:
        combo_name = "REGIME + " + " + ".join(active_filters)
    else:
        combo_name = "REGIME only" if "regime" in filter_list else "BASELINE"

    # Log if requested
    if log_result:
        log_combo_result(symbol, timeframe, days, combo_name, filter_flags, result, "specific")

    return {
        "combo_name": combo_name,
        "filter_flags": filter_flags,
        **result
    }


def run_oos_validation(symbol: str, timeframe: str, days: int,
                       filter_list: list, oos_ratio: float = 0.25) -> dict:
    """
    Run out-of-sample validation for a filter combination.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        days: Total days of data
        filter_list: List of filter names
        oos_ratio: Ratio of data to use for out-of-sample (default 25%)

    Returns:
        Dict with in_sample and out_of_sample results
    """
    # Fetch full data
    df = fetch_data(symbol, timeframe, days)
    if df.empty:
        return {"error": "No data"}

    # Split data
    split_idx = int(len(df) * (1 - oos_ratio))
    df_in = df.iloc[:split_idx].copy()
    df_out = df.iloc[split_idx:].copy()

    # Build filter flags
    filter_flags = {
        "use_regime_filter": "regime" in filter_list,
        "use_at_binary": "at_binary" in filter_list,
        "use_at_flat_filter": "at_flat_filter" in filter_list,
        "use_adx_filter": "adx_filter" in filter_list,
        "use_ssl_touch": "ssl_touch" in filter_list,
        "use_rsi_filter": "rsi_filter" in filter_list,
        "use_pbema_distance": "pbema_distance" in filter_list,
        "use_overlap_check": "overlap_check" in filter_list,
        "use_body_position": "body_position" in filter_list,
        "use_wick_rejection": "wick_rejection" in filter_list,
        "use_min_sl_filter": "min_sl_filter" in filter_list,
    }

    # Run tests
    result_in = run_combo_test(df_in, filter_flags)
    result_out = run_combo_test(df_out, filter_flags)

    # Generate combo name
    active_filters = [f for f in filter_list if f != "regime"]
    combo_name = "REGIME + " + " + ".join(active_filters) if active_filters else "REGIME only"

    # Log both results
    in_months = int(days * (1 - oos_ratio) / 30)
    out_months = int(days * oos_ratio / 30)

    log_combo_result(symbol, timeframe, int(days * (1 - oos_ratio)),
                     f"{combo_name} [IN-SAMPLE {in_months}m]", filter_flags, result_in, "oos_in_sample")
    log_combo_result(symbol, timeframe, int(days * oos_ratio),
                     f"{combo_name} [OUT-OF-SAMPLE {out_months}m]", filter_flags, result_out, "oos_out_sample")

    return {
        "combo_name": combo_name,
        "filter_flags": filter_flags,
        "in_sample": {
            "period": f"{df_in.index[0].date()} to {df_in.index[-1].date()}",
            **result_in
        },
        "out_of_sample": {
            "period": f"{df_out.index[0].date()} to {df_out.index[-1].date()}",
            **result_out
        },
        "oos_valid": result_out.get("pnl", 0) > 0
    }


def main():
    parser = argparse.ArgumentParser(description="Filter Combination Tester (AT Scenario Compatible)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--full-scan", action="store_true", help="Test ALL combinations")
    parser.add_argument("--incremental", action="store_true", help="Add filters one by one")
    parser.add_argument("--specific", type=str, help="Test specific combo, e.g., 'regime,at_flat_filter,adx_filter'")
    parser.add_argument("--oos", action="store_true", help="Run out-of-sample validation with --specific")

    args = parser.parse_args()

    # Handle specific combo test
    if args.specific:
        filter_list = [f.strip() for f in args.specific.split(",")]
        print(f"Testing specific combo: {filter_list}")

        if args.oos:
            result = run_oos_validation(args.symbol, args.timeframe, args.days, filter_list)
            print(f"\n{'='*60}")
            print(f"OOS VALIDATION: {result['combo_name']}")
            print(f"{'='*60}")
            print(f"In-Sample:  {result['in_sample']['trades']} trades, ${result['in_sample']['pnl']:.2f} PnL")
            print(f"Out-Sample: {result['out_of_sample']['trades']} trades, ${result['out_of_sample']['pnl']:.2f} PnL")
            print(f"OOS Valid:  {'✅ YES' if result['oos_valid'] else '❌ NO'}")
        else:
            result = run_specific_combo(args.symbol, args.timeframe, args.days, filter_list)
            print(f"\n{'='*60}")
            print(f"RESULT: {result['combo_name']}")
            print(f"{'='*60}")
            print(f"Trades: {result['trades']} | WR: {result['wr']:.1f}% | PnL: ${result['pnl']:.2f} | DD: ${result['dd']:.2f}")
        return

    print("=" * 80)
    print("FILTER COMBINATION TESTER (AT Scenario Compatible)")
    print("=" * 80)
    print(f"Symbol: {args.symbol} | TF: {args.timeframe} | Days: {args.days}")
    print("Uses ORIGINAL check_core_signal from at_scenario_analyzer.py")
    print("=" * 80)

    # Fetch data
    df = fetch_data(args.symbol, args.timeframe, args.days)
    if df.empty:
        print("ERROR: No data!")
        return

    results = []

    # All filter keys for easy management
    ALL_FILTERS = [
        "at_binary", "at_flat_filter", "adx_filter", "ssl_touch",  # EXISTING
        "rsi_filter", "pbema_distance", "overlap_check", "body_position", "wick_rejection",  # OPTUNA
        "min_sl_filter"  # SL DISTANCE (from deep analysis - 57% WR vs 31%)
    ]

    def make_flags(active_filters):
        """Create flags dict with only specified filters active."""
        return {
            "use_regime_filter": "regime" in active_filters,
            "use_at_binary": "at_binary" in active_filters,
            "use_at_flat_filter": "at_flat_filter" in active_filters,
            "use_adx_filter": "adx_filter" in active_filters,
            "use_ssl_touch": "ssl_touch" in active_filters,
            "use_rsi_filter": "rsi_filter" in active_filters,
            "use_pbema_distance": "pbema_distance" in active_filters,
            "use_overlap_check": "overlap_check" in active_filters,
            "use_body_position": "body_position" in active_filters,
            "use_wick_rejection": "wick_rejection" in active_filters,
            "use_min_sl_filter": "min_sl_filter" in active_filters,
        }

    # ========== BASELINE: No filters ==========
    print("\n[BASELINE] No filters (pure check_core_signal)...")
    baseline = run_combo_test(df, make_flags([]))
    results.append({"combo": "BASELINE (no filters)", **baseline, "filters": []})
    print(f"   Core signals found: {baseline['signals_found']}")

    # ========== BASELINE + REGIME ==========
    print("[REGIME] regime_filter only...")
    regime_only = run_combo_test(df, make_flags(["regime"]))
    results.append({"combo": "REGIME only", **regime_only, "filters": ["regime"]})

    if args.incremental:
        print("\n--- INCREMENTAL: Adding one filter at a time to REGIME ---")
        print(f"    Testing {len(ALL_FILTERS)} filters...")
        for f in ALL_FILTERS:
            print(f"[REGIME + {f}]...")
            r = run_combo_test(df, make_flags(["regime", f]))
            results.append({"combo": f"REGIME + {f}", **r, "filters": ["regime", f]})

    if args.full_scan:
        total_combos = sum(1 for size in range(1, len(ALL_FILTERS) + 1) for _ in combinations(ALL_FILTERS, size))
        print(f"\n--- FULL SCAN: {total_combos} combinations ---")
        count = 0
        for size in range(1, len(ALL_FILTERS) + 1):
            for combo in combinations(ALL_FILTERS, size):
                count += 1
                combo_name = "REGIME + " + " + ".join(combo)
                if len(combo_name) > 60:
                    combo_name = f"REGIME + {len(combo)} filters"
                print(f"[{count}/{total_combos}] {combo_name}...")
                r = run_combo_test(df, make_flags(["regime"] + list(combo)))
                results.append({"combo": combo_name, **r, "filters": ["regime"] + list(combo)})

    # ========== RESULTS TABLE ==========
    print("\n" + "=" * 100)
    print("RESULTS (sorted by PnL)")
    print("=" * 100)
    print(f"{'Combination':<45} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'PnL':>12} {'MaxDD':>10}")
    print("-" * 100)

    results_sorted = sorted(results, key=lambda x: x["pnl"], reverse=True)

    for r in results_sorted:
        wr_str = f"{r['wr']:.1f}%" if r['trades'] > 0 else "N/A"
        print(f"{r['combo']:<45} {r['trades']:>8} {r['wins']:>6} {wr_str:>8} ${r['pnl']:>10.2f} ${r['dd']:>9.2f}")

    # ========== TOP 5 ==========
    print("\n" + "=" * 100)
    print("TOP 5 COMBINATIONS")
    print("=" * 100)

    for i, r in enumerate(results_sorted[:5], 1):
        print(f"\n#{i}: {r['combo']}")
        print(f"   Trades: {r['trades']} | WR: {r['wr']:.1f}% | PnL: ${r['pnl']:.2f} | DD: ${r['dd']:.2f}")

    print("\n" + "=" * 100)
    print(f"Baseline core signals: {baseline['signals_found']}")
    print("=" * 100)

    # ========== SAVE RESULTS ==========
    import json
    from datetime import datetime as dt

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "filter_combo_logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.symbol}_{args.timeframe}_{args.days}d_{timestamp}"

    # Save JSON
    json_path = os.path.join(log_dir, f"{base_name}.json")
    log_data = {
        "timestamp": timestamp,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "days": args.days,
        "baseline_signals": baseline['signals_found'],
        "mode": "full_scan" if args.full_scan else ("incremental" if args.incremental else "basic"),
        "results": [
            {
                "combo": r["combo"],
                "filters": r["filters"],
                "trades": r["trades"],
                "wins": r["wins"],
                "wr": round(r["wr"], 2),
                "pnl": round(r["pnl"], 2),
                "dd": round(r["dd"], 2),
            }
            for r in results_sorted
        ]
    }
    with open(json_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    # Save readable text
    txt_path = os.path.join(log_dir, f"{base_name}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"FILTER COMBO TEST RESULTS\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Symbol: {args.symbol}\n")
        f.write(f"Timeframe: {args.timeframe}\n")
        f.write(f"Days: {args.days}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Baseline Signals: {baseline['signals_found']}\n")
        f.write(f"=" * 60 + "\n\n")

        f.write(f"{'Rank':<5} {'Combination':<40} {'Trades':>8} {'WR%':>8} {'PnL':>12} {'MaxDD':>10}\n")
        f.write("-" * 85 + "\n")

        for i, r in enumerate(results_sorted, 1):
            wr_str = f"{r['wr']:.1f}%"
            f.write(f"{i:<5} {r['combo']:<40} {r['trades']:>8} {wr_str:>8} ${r['pnl']:>10.2f} ${r['dd']:>9.2f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("TOP 3 RECOMMENDATIONS:\n")
        for i, r in enumerate(results_sorted[:3], 1):
            f.write(f"  #{i}: {r['combo']} → PnL: ${r['pnl']:.2f}, DD: ${r['dd']:.2f}\n")

    print(f"\n📁 Results saved:")
    print(f"   JSON: {json_path}")
    print(f"   TXT:  {txt_path}")


if __name__ == "__main__":
    main()
