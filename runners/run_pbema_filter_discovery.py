#!/usr/bin/env python3
"""
PBEMA Filter Discovery System

Base: pbema_retest_v2.py → +$48.95 (2165 trades, 30% WR)

Tek tek filtre ekleyerek test eder.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import get_client, calculate_indicators, set_backtest_mode
from strategies.pbema_retest_v2 import check_pbema_retest_signal_v2
import pandas as pd
import requests
from datetime import datetime
from collections import defaultdict


def fetch_data(symbol: str, timeframe: str, days: int = 365):
    """Fetch and prepare data."""
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


def simulate_trade(df, i, signal_type, entry, tp, sl, max_hold=96):
    """Simulate a single trade."""
    for j in range(i + 1, min(i + max_hold, len(df))):
        candle = df.iloc[j]
        if signal_type == "LONG":
            if candle["low"] <= sl:
                return "LOSS", (sl - entry) / entry
            elif candle["high"] >= tp:
                return "WIN", (tp - entry) / entry
        else:  # SHORT
            if candle["high"] >= sl:
                return "LOSS", (entry - sl) / entry
            elif candle["low"] <= tp:
                return "WIN", (entry - tp) / entry

    # EOD exit
    exit_price = df.iloc[min(i + max_hold, len(df) - 1)]["close"]
    if signal_type == "LONG":
        return "EOD", (exit_price - entry) / entry
    else:
        return "EOD", (entry - exit_price) / entry


def run_backtest(df, filter_config):
    """Run backtest with given filter config."""
    trades = []

    # Extract strategy params (only ones the strategy accepts)
    strategy_params = {}
    for key in ["min_wick_ratio", "approach_threshold", "require_trend_alignment",
                "touch_tolerance", "tp_percentage", "sl_buffer"]:
        if key in filter_config:
            strategy_params[key] = filter_config[key]

    for i in range(100, len(df) - 10):
        result = check_pbema_retest_signal_v2(
            df, index=i, return_debug=False,
            **strategy_params
        )
        signal_type, entry, tp, sl, reason = result

        if not signal_type:
            continue

        # Apply direction filter
        if filter_config.get("long_only", False) and signal_type == "SHORT":
            continue
        if filter_config.get("short_only", False) and signal_type == "LONG":
            continue

        # Apply regime filter (ADX)
        if filter_config.get("regime_filter", False):
            if "adx" in df.columns:
                adx = df.iloc[i]["adx"]
                if pd.isna(adx) or adx < filter_config.get("regime_adx_min", 20):
                    continue

        # Apply AT confirmation filter
        if filter_config.get("at_confirmation", False):
            if signal_type == "LONG":
                if "at_buyers_dominant" in df.columns:
                    if not df.iloc[i]["at_buyers_dominant"]:
                        continue
            elif signal_type == "SHORT":
                if "at_sellers_dominant" in df.columns:
                    if not df.iloc[i]["at_sellers_dominant"]:
                        continue

        result_type, pnl = simulate_trade(df, i, signal_type, entry, tp, sl)
        trades.append({
            "time": df.index[i],
            "type": signal_type,
            "result": result_type,
            "pnl": pnl,
        })

    return trades


def analyze_trades(trades, name=""):
    """Analyze trade results."""
    if not trades:
        return {
            "name": name,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "wr": 0,
            "pnl": 0,
            "long_pnl": 0,
            "short_pnl": 0,
        }

    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = sum(1 for t in trades if t["result"] == "LOSS")
    eod = sum(1 for t in trades if t["result"] == "EOD")
    total_pnl = sum(t["pnl"] for t in trades) * 1000

    long_trades = [t for t in trades if t["type"] == "LONG"]
    short_trades = [t for t in trades if t["type"] == "SHORT"]
    long_pnl = sum(t["pnl"] for t in long_trades) * 1000
    short_pnl = sum(t["pnl"] for t in short_trades) * 1000

    wr = wins / len(trades) * 100 if trades else 0

    return {
        "name": name,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "eod": eod,
        "wr": wr,
        "pnl": total_pnl,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
    }


def main():
    print("=" * 70)
    print("PBEMA FILTER DISCOVERY")
    print("=" * 70)

    # Fetch data
    print("\nFetching BTCUSDT 15m data (1 year)...")
    df = fetch_data("BTCUSDT", "15m", days=365)
    print(f"Data ready: {len(df)} candles")

    # Define filter configurations to test
    filter_configs = [
        # BASE: No extra filters
        {"name": "BASE (default params)", "config": {}},

        # Filter 1: Regime filter (ADX)
        {"name": "+regime_filter (ADX>20)", "config": {"regime_filter": True, "regime_adx_min": 20}},
        {"name": "+regime_filter (ADX>25)", "config": {"regime_filter": True, "regime_adx_min": 25}},
        {"name": "+regime_filter (ADX>30)", "config": {"regime_filter": True, "regime_adx_min": 30}},

        # Filter 2: AT confirmation
        {"name": "+at_confirmation", "config": {"at_confirmation": True}},

        # Filter 3: Direction only
        {"name": "+long_only", "config": {"long_only": True}},
        {"name": "+short_only", "config": {"short_only": True}},

        # Filter 4: Trend alignment
        {"name": "+trend_alignment", "config": {"require_trend_alignment": True}},

        # Filter 5: Higher wick ratio
        {"name": "+min_wick_0.20", "config": {"min_wick_ratio": 0.20}},
        {"name": "+min_wick_0.25", "config": {"min_wick_ratio": 0.25}},

        # Filter 6: Higher approach threshold
        {"name": "+approach_0.75", "config": {"approach_threshold": 0.75}},
        {"name": "+approach_0.80", "config": {"approach_threshold": 0.80}},

        # Combinations
        {"name": "+long_only +regime(ADX>25)", "config": {"long_only": True, "regime_filter": True, "regime_adx_min": 25}},
        {"name": "+long_only +at_confirmation", "config": {"long_only": True, "at_confirmation": True}},
        {"name": "+long_only +min_wick_0.20", "config": {"long_only": True, "min_wick_ratio": 0.20}},
        {"name": "+regime(ADX>25) +at_confirmation", "config": {"regime_filter": True, "regime_adx_min": 25, "at_confirmation": True}},
    ]

    # Run all tests
    print("\n" + "=" * 70)
    print("RUNNING FILTER TESTS")
    print("=" * 70)

    results = []
    for fc in filter_configs:
        print(f"\nTesting: {fc['name']}...")
        trades = run_backtest(df, fc["config"])
        result = analyze_trades(trades, fc["name"])
        results.append(result)
        print(f"  → {result['trades']} trades, {result['wr']:.1f}% WR, ${result['pnl']:.2f} PnL")

    # Sort by PnL
    results.sort(key=lambda x: x["pnl"], reverse=True)

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS (sorted by PnL)")
    print("=" * 70)
    print(f"{'Config':<40} {'Trades':>8} {'WR':>8} {'PnL':>12} {'L_PnL':>10} {'S_PnL':>10}")
    print("-" * 90)

    for r in results:
        print(f"{r['name']:<40} {r['trades']:>8} {r['wr']:>7.1f}% ${r['pnl']:>10.2f} ${r['long_pnl']:>9.2f} ${r['short_pnl']:>9.2f}")

    # Best config
    best = results[0]
    print("\n" + "=" * 70)
    print(f"BEST CONFIG: {best['name']}")
    print(f"  Trades: {best['trades']}")
    print(f"  Win Rate: {best['wr']:.1f}%")
    print(f"  PnL: ${best['pnl']:.2f}")
    print(f"  LONG PnL: ${best['long_pnl']:.2f}")
    print(f"  SHORT PnL: ${best['short_pnl']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
