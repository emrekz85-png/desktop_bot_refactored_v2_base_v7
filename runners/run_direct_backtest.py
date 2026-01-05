#!/usr/bin/env python3
"""
DIRECT BACKTEST - Gerçek check_signal fonksiyonuyla test

Bu script AYNI signal fonksiyonunu kullanır ki live bot kullanacak.
Rolling WF optimizer'dan FARKLI - optimizer'ın kendi check_signal_fast'ı var.

Kullanım:
    python runners/run_direct_backtest.py
    python runners/run_direct_backtest.py --symbol BTCUSDT --timeframe 15m
    python runners/run_direct_backtest.py --days 365
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import calculate_indicators, get_client
from core.config import DEFAULT_STRATEGY_CONFIG
from strategies import check_signal  # GERÇEK signal fonksiyonu!


def fetch_data(symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical data."""
    client = get_client()

    # Calculate candle count
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    minutes = tf_minutes.get(timeframe, 15)
    candles_needed = (days * 24 * 60) // minutes + 500  # Extra for indicators

    print(f"Fetching {symbol} {timeframe} for {days} days (~{candles_needed} candles)...")

    all_dfs = []
    remaining = candles_needed
    end_time = None

    while remaining > 0:
        chunk_size = min(remaining, 1000)

        if end_time:
            import requests
            url = f"{client.BASE_URL}/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': chunk_size,
                'endTime': end_time
            }
            res = requests.get(url, params=params, timeout=30)
            if res.status_code != 200:
                break
            data = res.json()
            if not data:
                break

            df_chunk = pd.DataFrame(data).iloc[:, :6]
            df_chunk.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')
        else:
            df_chunk = client.get_klines(symbol=symbol, interval=timeframe, limit=chunk_size)

        if df_chunk.empty:
            break

        all_dfs.insert(0, df_chunk)
        remaining -= len(df_chunk)

        if 'timestamp' in df_chunk.columns:
            end_time = int(df_chunk['timestamp'].iloc[0].timestamp() * 1000) - 1
        else:
            break

        if len(df_chunk) < chunk_size:
            break

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df.set_index('timestamp', inplace=True)

    # Calculate indicators
    df = calculate_indicators(df, timeframe=timeframe)

    print(f"Got {len(df)} candles: {df.index[0]} to {df.index[-1]}")
    return df


def simulate_trade(
    df: pd.DataFrame,
    signal_idx: int,
    signal_type: str,
    entry: float,
    tp: float,
    sl: float,
    position_size: float = 35.0,
) -> Dict:
    """Simulate a single trade."""
    abs_idx = signal_idx if signal_idx >= 0 else (len(df) + signal_idx)

    for i in range(abs_idx + 1, len(df)):
        candle = df.iloc[i]
        high = float(candle["high"])
        low = float(candle["low"])

        if signal_type == "LONG":
            if low <= sl:
                loss = (sl - entry) / entry * position_size
                return {"exit_idx": i, "exit_price": sl, "exit_reason": "SL", "pnl": loss}
            if high >= tp:
                profit = (tp - entry) / entry * position_size
                return {"exit_idx": i, "exit_price": tp, "exit_reason": "TP", "pnl": profit}
        else:
            if high >= sl:
                loss = (entry - sl) / entry * position_size
                return {"exit_idx": i, "exit_price": sl, "exit_reason": "SL", "pnl": loss}
            if low <= tp:
                profit = (entry - tp) / entry * position_size
                return {"exit_idx": i, "exit_price": tp, "exit_reason": "TP", "pnl": profit}

    # No exit - close at last price
    last_price = float(df.iloc[-1]["close"])
    if signal_type == "LONG":
        pnl = (last_price - entry) / entry * position_size
    else:
        pnl = (entry - last_price) / entry * position_size

    return {"exit_idx": len(df) - 1, "exit_price": last_price, "exit_reason": "EOD", "pnl": pnl}


def run_backtest(
    df: pd.DataFrame,
    config: dict,
    timeframe: str,
    min_bars_between: int = 5,
) -> Dict:
    """
    Run backtest using REAL check_signal function.

    This is the SAME function that the live bot uses!
    """
    trades = []
    rejection_reasons = {}
    last_exit_idx = -min_bars_between

    print(f"\nRunning backtest with config:")
    print(f"  regime_filter: {config.get('regime_filter', 'NOT SET')}")
    print(f"  at_mode: {config.get('at_mode', 'NOT SET')}")
    print(f"  rr: {config.get('rr', 'NOT SET')}")
    print(f"  rsi: {config.get('rsi', 'NOT SET')}")
    print()

    for i in range(200, len(df) - 10):
        if i - last_exit_idx < min_bars_between:
            continue

        # Use REAL check_signal function!
        result = check_signal(df, config, index=i, return_debug=False, timeframe=timeframe)
        signal_type, entry, tp, sl, reason = result

        if signal_type is None:
            # Track rejection reasons
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue

        # Simulate trade
        trade_result = simulate_trade(df, i, signal_type, entry, tp, sl)

        trade = {
            "signal_idx": i,
            "signal_time": str(df.index[i]),
            "type": signal_type,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "reason": reason,
            **trade_result
        }
        trades.append(trade)
        last_exit_idx = trade_result["exit_idx"]

    # Calculate stats
    if not trades:
        return {
            "trades": [],
            "total_trades": 0,
            "wins": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "max_drawdown": 0,
            "rejection_reasons": rejection_reasons,
        }

    wins = sum(1 for t in trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in trades)

    # Drawdown
    equity = [0]
    for t in trades:
        equity.append(equity[-1] + t["pnl"])
    peak = 0
    max_dd = 0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    return {
        "trades": trades,
        "total_trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) * 100 if trades else 0,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "rejection_reasons": rejection_reasons,
    }


def main():
    parser = argparse.ArgumentParser(description="Direct Backtest with REAL check_signal")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--compare-regime", action="store_true", help="Compare with/without regime filter")

    args = parser.parse_args()

    print("=" * 70)
    print("DIRECT BACKTEST - GERÇEK check_signal FONKSİYONU")
    print("=" * 70)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Days: {args.days}")
    print("=" * 70)

    # Fetch data
    df = fetch_data(args.symbol, args.timeframe, args.days)
    if df.empty:
        print("ERROR: No data fetched!")
        return

    # Get current config
    config = DEFAULT_STRATEGY_CONFIG.copy()

    if args.compare_regime:
        # Test both configurations
        print("\n" + "=" * 70)
        print("COMPARISON: regime_filter=off vs skip_neutral")
        print("=" * 70)

        # Test 1: regime_filter OFF
        config_off = config.copy()
        config_off["regime_filter"] = "off"
        print("\n[1] regime_filter = OFF")
        result_off = run_backtest(df, config_off, args.timeframe)

        # Test 2: regime_filter skip_neutral
        config_skip = config.copy()
        config_skip["regime_filter"] = "skip_neutral"
        print("\n[2] regime_filter = skip_neutral")
        result_skip = run_backtest(df, config_skip, args.timeframe)

        # Comparison table
        print("\n" + "=" * 70)
        print("SONUÇLAR")
        print("=" * 70)
        print(f"{'Config':<25} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'PnL':>12} {'MaxDD':>10}")
        print("-" * 70)
        print(f"{'regime_filter=off':<25} {result_off['total_trades']:>8} {result_off['wins']:>6} "
              f"{result_off['win_rate']:>7.1f}% ${result_off['total_pnl']:>10.2f} ${result_off['max_drawdown']:>9.2f}")
        print(f"{'regime_filter=skip_neutral':<25} {result_skip['total_trades']:>8} {result_skip['wins']:>6} "
              f"{result_skip['win_rate']:>7.1f}% ${result_skip['total_pnl']:>10.2f} ${result_skip['max_drawdown']:>9.2f}")

        # Difference
        pnl_diff = result_skip['total_pnl'] - result_off['total_pnl']
        trades_diff = result_skip['total_trades'] - result_off['total_trades']
        print("-" * 70)
        print(f"{'FARK (skip - off)':<25} {trades_diff:>8} {'-':>6} "
              f"{'-':>8} ${pnl_diff:>10.2f}")

    else:
        # Single test with current config
        print(f"\nCurrent config regime_filter: {config.get('regime_filter', 'NOT SET')}")
        result = run_backtest(df, config, args.timeframe)

        print("\n" + "=" * 70)
        print("SONUÇLAR")
        print("=" * 70)
        print(f"Total Trades: {result['total_trades']}")
        print(f"Wins: {result['wins']} ({result['win_rate']:.1f}%)")
        print(f"Total PnL: ${result['total_pnl']:.2f}")
        print(f"Max Drawdown: ${result['max_drawdown']:.2f}")

        print("\n📊 Top Rejection Reasons:")
        sorted_reasons = sorted(result['rejection_reasons'].items(), key=lambda x: -x[1])[:10]
        for reason, count in sorted_reasons:
            print(f"  {reason}: {count}")

        # Save trades
        if result['trades']:
            output_file = f"direct_backtest_{args.symbol}_{args.timeframe}.json"
            with open(output_file, 'w') as f:
                json.dump(result['trades'], f, indent=2, default=str)
            print(f"\nTrades saved to: {output_file}")

    print("\n" + "=" * 70)
    print("ÖNEMLİ: Bu test GERÇEK check_signal fonksiyonunu kullanır.")
    print("Live bot AYNI sonuçları üretecektir (aynı data ile).")
    print("=" * 70)


if __name__ == "__main__":
    main()
