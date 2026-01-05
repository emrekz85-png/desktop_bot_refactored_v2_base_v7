#!/usr/bin/env python3
"""Test PBEMA Retest v2 Strategy with Strength Score System."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import get_client, calculate_indicators, set_backtest_mode
from strategies.pbema_retest_v2 import check_pbema_retest_signal_v2 as check_pbema_retest_signal
import pandas as pd
import requests
from datetime import datetime, timedelta


def fetch_data(symbol: str, timeframe: str, days: int = 365):
    """Fetch and prepare data with indicators (from run.py)."""
    set_backtest_mode(True)
    client = get_client()

    candles_map = {"5m": 288, "15m": 96, "1h": 24, "4h": 6}
    candles = min(days * candles_map.get(timeframe, 96), 35000)

    all_dfs = []
    remaining = candles
    end_time = None

    while remaining > 0:
        chunk = min(remaining, 1000)

        if end_time:
            url = f"{client.BASE_URL}/klines"
            params = {"symbol": symbol, "interval": timeframe, "limit": chunk, "endTime": end_time}
            res = requests.get(url, params=params, timeout=30)
            if res.status_code != 200 or not res.json():
                break
            df_c = pd.DataFrame(res.json()).iloc[:, :6]
            df_c.columns = ["timestamp", "open", "high", "low", "close", "volume"]
            df_c["timestamp"] = pd.to_datetime(df_c["timestamp"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df_c[col] = pd.to_numeric(df_c[col], errors="coerce")
        else:
            df_c = client.get_klines(symbol=symbol, interval=timeframe, limit=chunk)

        if df_c.empty:
            break

        all_dfs.insert(0, df_c)
        remaining -= len(df_c)

        if "timestamp" in df_c.columns:
            end_time = int(df_c["timestamp"].iloc[0].timestamp() * 1000) - 1
        else:
            break

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df.set_index("timestamp", inplace=True)
    df = calculate_indicators(df, timeframe=timeframe)

    return df


def main():
    print("=" * 60)
    print("PBEMA RETEST v2 TEST - With Strength Score System")
    print("=" * 60)

    # Fetch data
    print("Fetching BTCUSDT 15m data (1 year)...")
    df = fetch_data("BTCUSDT", "15m", days=365)
    print(f"Data ready: {len(df)} candles")
    print()

    # Run backtest
    print("Running PBEMA Retest v2 backtest...")
    print("=" * 60)

    trades = []

    # Default params - pozitif sonuç veriyor
    for i in range(100, len(df) - 10):
        result = check_pbema_retest_signal(
            df, index=i, return_debug=True,
        )
        signal_type, entry, tp, sl, reason, debug = result

        if signal_type:
            trade = {
                "time": df.index[i],
                "type": signal_type,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "reason": reason,
                "strength_score": debug.get("avg_wick_ratio", 0),
                "strong_rejections": debug.get("prior_rejections", 0),
            }

            # Simulate trade
            for j in range(i + 1, min(i + 96, len(df))):  # Max 24h hold
                candle = df.iloc[j]
                if signal_type == "LONG":
                    if candle["low"] <= sl:
                        trade["exit"] = sl
                        trade["result"] = "LOSS"
                        trade["pnl"] = (sl - entry) / entry
                        break
                    elif candle["high"] >= tp:
                        trade["exit"] = tp
                        trade["result"] = "WIN"
                        trade["pnl"] = (tp - entry) / entry
                        break
                else:  # SHORT
                    if candle["high"] >= sl:
                        trade["exit"] = sl
                        trade["result"] = "LOSS"
                        trade["pnl"] = (entry - sl) / entry
                        break
                    elif candle["low"] <= tp:
                        trade["exit"] = tp
                        trade["result"] = "WIN"
                        trade["pnl"] = (entry - tp) / entry
                        break
            else:
                trade["exit"] = df.iloc[min(i + 96, len(df) - 1)]["close"]
                trade["result"] = "EOD"
                if signal_type == "LONG":
                    trade["pnl"] = (trade["exit"] - entry) / entry
                else:
                    trade["pnl"] = (entry - trade["exit"]) / entry

            trades.append(trade)

    # Results
    print(f"Total Trades: {len(trades)}")

    if trades:
        wins = sum(1 for t in trades if t["result"] == "WIN")
        losses = sum(1 for t in trades if t["result"] == "LOSS")
        eod = sum(1 for t in trades if t["result"] == "EOD")

        total_pnl = sum(t["pnl"] for t in trades)
        pnl_dollar = total_pnl * 1000  # $1000 account

        win_rate = wins / len(trades) * 100 if trades else 0

        avg_score = sum(t["strength_score"] for t in trades) / len(trades)
        avg_rejections = sum(t["strong_rejections"] for t in trades) / len(trades)

        print()
        print("RESULTS:")
        print("=" * 60)
        print(f"Trades: {len(trades)} (WIN: {wins}, LOSS: {losses}, EOD: {eod})")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total PnL: {total_pnl*100:.2f}% (${pnl_dollar:.2f})")
        print()
        print(f"Avg Strength Score: {avg_score:.2f}")
        print(f"Avg Strong Rejections: {avg_rejections:.1f}")
        print()
        print("Sample trades:")
        for t in trades[:5]:
            time_str = str(t["time"])[:16]
            print(f"  {time_str} {t['type']} {t['result']} "
                  f"Score:{t['strength_score']:.2f} Rej:{t['strong_rejections']}")

        # LONG vs SHORT breakdown
        long_trades = [t for t in trades if t["type"] == "LONG"]
        short_trades = [t for t in trades if t["type"] == "SHORT"]

        long_wins = sum(1 for t in long_trades if t["result"] == "WIN")
        short_wins = sum(1 for t in short_trades if t["result"] == "WIN")

        long_pnl = sum(t["pnl"] for t in long_trades) * 1000
        short_pnl = sum(t["pnl"] for t in short_trades) * 1000

        long_wr = (long_wins / len(long_trades) * 100) if long_trades else 0
        short_wr = (short_wins / len(short_trades) * 100) if short_trades else 0

        print()
        print("=" * 60)
        print("LONG vs SHORT Breakdown:")
        print("=" * 60)
        print(f"LONG:  {len(long_trades)} trades, {long_wr:.1f}% WR, ${long_pnl:.2f} PnL")
        print(f"SHORT: {len(short_trades)} trades, {short_wr:.1f}% WR, ${short_pnl:.2f} PnL")

        # Monthly breakdown
        from collections import defaultdict
        monthly = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
        for t in trades:
            month = str(t["time"])[:7]  # YYYY-MM
            monthly[month]["trades"] += 1
            if t["result"] == "WIN":
                monthly[month]["wins"] += 1
            monthly[month]["pnl"] += t["pnl"] * 1000

        print()
        print("Monthly Performance:")
        print("-" * 40)
        for month in sorted(monthly.keys()):
            m = monthly[month]
            wr = (m["wins"] / m["trades"] * 100) if m["trades"] else 0
            print(f"{month}: {m['trades']:3d} trades, {wr:5.1f}% WR, ${m['pnl']:7.2f}")

        # Compare with old strategy stats
        print()
        print("=" * 60)
        print("COMPARISON (Old vs New):")
        print("=" * 60)
        print("Old PBEMA Strategy: 107 trades, 46% WR, -$4.72 PnL")
        print(f"New PBEMA v2:       {len(trades)} trades, {win_rate:.0f}% WR, ${pnl_dollar:.2f} PnL")
        print()
        print(f"PnL improvement: +${pnl_dollar + 4.72:.2f}")
    else:
        print("No trades found!")
        print()
        print("This is expected if the filters are too strict.")
        print("Try relaxing: min_strong_rejections=2 or min_strength_score=0.50")


if __name__ == "__main__":
    main()
