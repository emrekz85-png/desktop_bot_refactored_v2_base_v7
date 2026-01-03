#!/usr/bin/env python3
"""
ROLLING WALK-FORWARD with FILTER COMBO CONFIG

Filter Combo Test'ten çıkan en iyi konfigürasyonu Rolling WF ile doğrular.
AYNI sinyal mantığını kullanır: check_core_signal + apply_filters

Kullanım:
    # Tek filtre combo testi
    python runners/run_rolling_wf_combo.py --symbol BTCUSDT --timeframe 15m \
        --filters "regime,at_flat_filter,adx_filter"

    # Full year test
    python runners/run_rolling_wf_combo.py --symbol BTCUSDT --timeframe 15m \
        --filters "regime,at_flat_filter,adx_filter" --full-year
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import calculate_indicators, get_client, set_backtest_mode
from core.at_scenario_analyzer import check_core_signal
from runners.run_filter_combo_test import apply_filters, simulate_trade, log_combo_result

# Enable backtest mode
set_backtest_mode(True)


def fetch_data(symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
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


def run_window_test(df: pd.DataFrame, filter_flags: dict, min_bars_between: int = 5) -> dict:
    """
    Run backtest on a window using check_core_signal + apply_filters.
    """
    trades = []
    last_signal_idx = -min_bars_between

    for i in range(60, len(df) - 10):
        if i - last_signal_idx < min_bars_between:
            continue

        # Core signal (same as AT Scenario)
        signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)

        if signal_type is None:
            continue

        last_signal_idx = i

        # Apply filters
        passed, filter_reason = apply_filters(df, i, signal_type, **filter_flags)

        if not passed:
            continue

        # Simulate trade
        trade = simulate_trade(df, i, signal_type, entry, tp, sl)
        trade["signal_time"] = str(df.index[i])
        trade["signal_type"] = signal_type
        trades.append(trade)

    if not trades:
        return {"trades": 0, "wins": 0, "wr": 0, "pnl": 0, "dd": 0, "trade_list": []}

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
        "trade_list": trades
    }


def run_rolling_wf_combo(
    symbol: str,
    timeframe: str,
    filter_list: list,
    lookback_days: int = 60,
    forward_days: int = 7,
    start_date: str = None,
    end_date: str = None,
    verbose: bool = True,
) -> dict:
    """
    Run Rolling Walk-Forward backtest with filter combo config.

    Her window için:
    1. Geçmiş lookback_days verisinde test yap (sadece metric için)
    2. Sonraki forward_days'de AYNI config ile trade et
    3. OOS sonuçları topla

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        filter_list: List of filter names (e.g., ["regime", "at_flat_filter", "adx_filter"])
        lookback_days: Days to look back (not used for optimization, just context)
        forward_days: Days to trade forward (OOS window)
        start_date: Start date for test period
        end_date: End date for test period
        verbose: Print progress

    Returns:
        Dict with stitched results
    """

    def log(msg: str):
        if verbose:
            print(msg)

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
    }

    # Generate combo name
    active_filters = [f for f in filter_list if f != "regime"]
    combo_name = "REGIME + " + " + ".join(active_filters) if active_filters else "REGIME only"

    log(f"\n{'='*70}")
    log(f"ROLLING WALK-FORWARD with FILTER COMBO")
    log(f"{'='*70}")
    log(f"Symbol: {symbol} | TF: {timeframe}")
    log(f"Config: {combo_name}")
    log(f"Lookback: {lookback_days}d | Forward: {forward_days}d")
    log(f"Period: {start_date or 'auto'} → {end_date or 'today'}")
    log(f"{'='*70}\n")

    # Set default dates
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
        start_date = start_dt.strftime("%Y-%m-%d")

    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days

    # Fetch all data (with extra for lookback)
    fetch_days = total_days + lookback_days + 30
    df = fetch_data(symbol, timeframe, fetch_days)
    if df.empty:
        return {"error": "No data"}

    # Generate windows
    windows = []
    current_start = start_dt
    window_id = 0

    while current_start < end_dt:
        window_end = min(current_start + timedelta(days=forward_days), end_dt)
        windows.append({
            "window_id": window_id,
            "start": current_start,
            "end": window_end,
        })
        current_start = window_end
        window_id += 1

    log(f"Generated {len(windows)} windows")

    # Run each window
    all_trades = []
    window_results = []

    for w in windows:
        # Filter data for this window
        mask = (df.index >= pd.Timestamp(w["start"], tz="UTC")) & \
               (df.index < pd.Timestamp(w["end"], tz="UTC"))
        df_window = df[mask].copy()

        if len(df_window) < 10:
            log(f"[Window {w['window_id']}] Skipping - insufficient data ({len(df_window)} candles)")
            continue

        # Run test
        result = run_window_test(df_window, filter_flags)

        log(f"[Window {w['window_id']}] {w['start'].date()} → {w['end'].date()}: "
            f"{result['trades']} trades, ${result['pnl']:.2f} PnL")

        window_results.append({
            "window_id": w["window_id"],
            "start": str(w["start"].date()),
            "end": str(w["end"].date()),
            **{k: v for k, v in result.items() if k != "trade_list"}
        })

        all_trades.extend(result["trade_list"])

    # Aggregate results
    if not all_trades:
        return {
            "combo_name": combo_name,
            "filter_flags": filter_flags,
            "total_trades": 0,
            "total_pnl": 0,
            "total_wins": 0,
            "win_rate": 0,
            "max_drawdown": 0,
            "window_results": window_results,
        }

    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t["win"])
    total_pnl = sum(t["pnl"] for t in all_trades)

    # Calculate max drawdown
    equity = [0]
    for t in all_trades:
        equity.append(equity[-1] + t["pnl"])
    peak, max_dd = 0, 0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    result = {
        "combo_name": combo_name,
        "filter_flags": filter_flags,
        "symbol": symbol,
        "timeframe": timeframe,
        "period": f"{start_date} to {end_date}",
        "total_days": total_days,
        "total_trades": total_trades,
        "total_wins": total_wins,
        "win_rate": total_wins / total_trades * 100 if total_trades else 0,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "windows": len(windows),
        "window_results": window_results,
        "trades": all_trades,
    }

    # Log result
    log_combo_result(symbol, timeframe, total_days, f"{combo_name} [WF-VALIDATION]",
                     filter_flags, {
                         "trades": total_trades,
                         "wins": total_wins,
                         "wr": result["win_rate"],
                         "pnl": total_pnl,
                         "dd": max_dd,
                     }, "wf_validation")

    # Print summary
    log(f"\n{'='*70}")
    log(f"ROLLING WF RESULTS: {combo_name}")
    log(f"{'='*70}")
    log(f"Period: {start_date} → {end_date} ({total_days} days)")
    log(f"Windows: {len(windows)}")
    log(f"Total Trades: {total_trades}")
    log(f"Wins: {total_wins} ({result['win_rate']:.1f}%)")
    log(f"Total PnL: ${total_pnl:.2f}")
    log(f"Max Drawdown: ${max_dd:.2f}")
    log(f"{'='*70}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Rolling Walk-Forward with Filter Combo")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--filters", type=str, required=True,
                        help="Comma-separated filter list, e.g., 'regime,at_flat_filter,adx_filter'")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback days")
    parser.add_argument("--forward", type=int, default=7, help="Forward days (OOS window)")
    parser.add_argument("--start-date", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--full-year", action="store_true", help="Run full year test")
    parser.add_argument("--save-trades", action="store_true", help="Save trade list to JSON")

    args = parser.parse_args()

    # Parse filters
    filter_list = [f.strip() for f in args.filters.split(",")]

    # Set dates for full year
    if args.full_year:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
        args.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    result = run_rolling_wf_combo(
        symbol=args.symbol,
        timeframe=args.timeframe,
        filter_list=filter_list,
        lookback_days=args.lookback,
        forward_days=args.forward,
        start_date=args.start_date,
        end_date=args.end_date,
        verbose=True,
    )

    # Save trades if requested
    if args.save_trades and result.get("trades"):
        output_file = f"wf_combo_trades_{args.symbol}_{args.timeframe}.json"
        with open(output_file, "w") as f:
            json.dump(result["trades"], f, indent=2, default=str)
        print(f"\nTrades saved to: {output_file}")

    # Final assessment
    print(f"\n{'='*70}")
    if result.get("total_pnl", 0) > 0:
        print(f"✅ WALK-FORWARD VALIDATION PASSED")
        print(f"   {args.symbol} {args.timeframe} with {result['combo_name']}")
        print(f"   ${result['total_pnl']:.2f} PnL over {result['total_days']} days")
    else:
        print(f"❌ WALK-FORWARD VALIDATION FAILED")
        print(f"   {args.symbol} {args.timeframe} with {result['combo_name']}")
        print(f"   ${result.get('total_pnl', 0):.2f} PnL - config may be overfit")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
