#!/usr/bin/env python3
"""Diagnose why optimizer is rejecting configs in rolling walk-forward.

This script analyzes the specific rejection reasons to identify:
1. Are signals being generated but not meeting hard_min_trades?
2. Are configs profitable in IS but rejected for E[R]?
3. What's the exact bottleneck preventing config selection?
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, "/Users/emreoksuz/desktop_bot_refactored_v2_base_v7")

from core import SYMBOLS, TIMEFRAMES, TRADING_CONFIG
from core.config import DEFAULT_STRATEGY_CONFIG, MIN_EXPECTANCY_R_MULTIPLE
from core.binance_client import get_client
from core.indicators import calculate_indicators
from core.trade_manager import SimTradeManager
from core.trading_engine import TradingEngine, set_backtest_mode
from core.optimizer import _generate_candidate_configs, _compute_optimizer_score
from strategies import check_signal

# Enable backtest mode
set_backtest_mode(True)


def fetch_data(symbol: str, tf: str, days: int) -> pd.DataFrame:
    """Fetch historical data."""
    client = get_client()

    # Calculate candle limit based on timeframe
    tf_minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "12h": 720, "1d": 1440
    }
    minutes_per_candle = tf_minutes.get(tf, 15)
    total_minutes = days * 24 * 60
    candle_limit = total_minutes // minutes_per_candle + 100

    # Use paginated version for large requests
    df = client.get_klines_paginated(symbol, tf, total_candles=candle_limit)

    # Ensure 'time' column exists (some functions expect it)
    if "time" not in df.columns and "timestamp" in df.columns:
        df["time"] = df["timestamp"]

    return df


def simulate_trades(df: pd.DataFrame, config: dict, tf: str) -> Tuple[float, int, List[float]]:
    """Simulate trades for a config and return PnL, trades, and R-multiples."""
    balance = TRADING_CONFIG["initial_balance"]
    risk_pct = TRADING_CONFIG["risk_per_trade_pct"]

    trade_pnls = []
    trade_r_multiples = []

    for i in range(200, len(df) - 2):
        signal_type, entry, tp, sl, reason, _ = check_signal(
            df, config, index=i, return_debug=True
        )

        if signal_type == "NEUTRAL":
            continue

        # Skip if entry/tp/sl are None
        if entry is None or tp is None or sl is None:
            continue

        # Calculate risk and position size
        if signal_type == "LONG":
            risk_per_unit = entry - sl
        else:
            risk_per_unit = sl - entry

        if risk_per_unit <= 0:
            continue

        risk_dollars = balance * risk_pct
        position_size = risk_dollars / risk_per_unit

        # Simulate trade outcome at next candle
        next_candle = df.iloc[i + 1]

        if signal_type == "LONG":
            if next_candle["low"] <= sl:
                pnl = -risk_dollars
                r_multiple = -1.0
            elif next_candle["high"] >= tp:
                pnl = position_size * (tp - entry)
                r_multiple = pnl / risk_dollars
            else:
                # Exit at close
                pnl = position_size * (next_candle["close"] - entry)
                r_multiple = pnl / risk_dollars
        else:  # SHORT
            if next_candle["high"] >= sl:
                pnl = -risk_dollars
                r_multiple = -1.0
            elif next_candle["low"] <= tp:
                pnl = position_size * (entry - tp)
                r_multiple = pnl / risk_dollars
            else:
                # Exit at close
                pnl = position_size * (entry - next_candle["close"])
                r_multiple = pnl / risk_dollars

        trade_pnls.append(pnl)
        trade_r_multiples.append(r_multiple)
        balance += pnl

    return sum(trade_pnls), len(trade_pnls), trade_r_multiples


def diagnose_window(symbol: str, tf: str, days: int = 60):
    """Diagnose optimizer rejections for a specific window."""
    print(f"\n{'='*70}")
    print(f"OPTIMIZER REJECTION DIAGNOSIS: {symbol} {tf} ({days} days)")
    print(f"{'='*70}")

    # Fetch data
    print("\nFetching data...")
    df = fetch_data(symbol, tf, days)
    df = calculate_indicators(df)
    print(f"  Loaded {len(df)} candles")

    # Generate configs
    configs = _generate_candidate_configs()
    print(f"\nTesting {len(configs)} configs...")

    # Track rejection reasons
    rejection_counts = {
        "too_few_trades": 0,
        "negative_pnl": 0,
        "low_expected_r": 0,
        "passed": 0
    }

    passed_configs = []
    all_trade_counts = []

    min_expected_r = MIN_EXPECTANCY_R_MULTIPLE.get(tf, 0.08)
    hard_min_trades = 5

    for i, config in enumerate(configs):
        # Merge with defaults
        full_config = {**DEFAULT_STRATEGY_CONFIG, **config}

        # Simulate trades
        net_pnl, trades, r_multiples = simulate_trades(df, full_config, tf)
        all_trade_counts.append(trades)

        # Check rejection reasons
        if trades < hard_min_trades:
            rejection_counts["too_few_trades"] += 1
            continue

        if net_pnl <= 0:
            rejection_counts["negative_pnl"] += 1
            continue

        expected_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0
        if expected_r < min_expected_r:
            rejection_counts["low_expected_r"] += 1
            continue

        rejection_counts["passed"] += 1
        passed_configs.append({
            "config": full_config,
            "pnl": net_pnl,
            "trades": trades,
            "expected_r": expected_r
        })

    # Print results
    print(f"\nRESULTS:")
    print(f"  Total configs tested: {len(configs)}")
    print(f"  ❌ Too few trades (<{hard_min_trades}): {rejection_counts['too_few_trades']} ({100*rejection_counts['too_few_trades']/len(configs):.1f}%)")
    print(f"  ❌ Negative PnL: {rejection_counts['negative_pnl']} ({100*rejection_counts['negative_pnl']/len(configs):.1f}%)")
    print(f"  ❌ Low E[R] (<{min_expected_r}): {rejection_counts['low_expected_r']} ({100*rejection_counts['low_expected_r']/len(configs):.1f}%)")
    print(f"  ✅ PASSED: {rejection_counts['passed']} ({100*rejection_counts['passed']/len(configs):.1f}%)")

    # Trade count distribution
    print(f"\nTRADE COUNT DISTRIBUTION:")
    if all_trade_counts:
        print(f"  Min: {min(all_trade_counts)}, Max: {max(all_trade_counts)}, Median: {np.median(all_trade_counts):.0f}")
        print(f"  0 trades: {all_trade_counts.count(0)} configs")
        print(f"  1-4 trades: {sum(1 for t in all_trade_counts if 0 < t < 5)} configs")
        print(f"  5+ trades: {sum(1 for t in all_trade_counts if t >= 5)} configs")

    if passed_configs:
        print(f"\nTOP 5 PASSED CONFIGS:")
        passed_configs.sort(key=lambda x: x["pnl"], reverse=True)
        for i, pc in enumerate(passed_configs[:5]):
            print(f"  {i+1}. PnL=${pc['pnl']:.2f}, Trades={pc['trades']}, E[R]={pc['expected_r']:.3f}")
            print(f"     RR={pc['config']['rr']}, RSI={pc['config']['rsi']}")

    return rejection_counts, passed_configs


def main():
    print("="*70)
    print("OPTIMIZER REJECTION DIAGNOSIS")
    print("="*70)

    # Test multiple symbol/timeframe combinations
    test_cases = [
        ("BTCUSDT", "15m", 60),
        ("BTCUSDT", "1h", 60),
        ("ETHUSDT", "15m", 60),
        ("LINKUSDT", "15m", 60),
    ]

    summary = {}

    for symbol, tf, days in test_cases:
        rejection_counts, passed = diagnose_window(symbol, tf, days)
        summary[f"{symbol}_{tf}"] = rejection_counts

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\nRejection breakdown by stream:")
    print(f"{'Stream':<20} {'TooFew':<10} {'NegPnL':<10} {'LowE[R]':<10} {'PASSED':<10}")
    print("-"*60)

    for key, counts in summary.items():
        print(f"{key:<20} {counts['too_few_trades']:<10} {counts['negative_pnl']:<10} {counts['low_expected_r']:<10} {counts['passed']:<10}")

    # Calculate overall bottleneck
    total_configs = sum(sum(c.values()) for c in summary.values())
    total_too_few = sum(c["too_few_trades"] for c in summary.values())
    total_neg_pnl = sum(c["negative_pnl"] for c in summary.values())
    total_low_er = sum(c["low_expected_r"] for c in summary.values())
    total_passed = sum(c["passed"] for c in summary.values())

    print("\n" + "-"*60)
    print(f"{'TOTAL':<20} {total_too_few:<10} {total_neg_pnl:<10} {total_low_er:<10} {total_passed:<10}")

    print("\n🔍 PRIMARY BOTTLENECK:")
    bottlenecks = [
        ("Too Few Trades", total_too_few),
        ("Negative PnL", total_neg_pnl),
        ("Low E[R]", total_low_er)
    ]
    bottlenecks.sort(key=lambda x: x[1], reverse=True)

    for name, count in bottlenecks:
        pct = 100 * count / (total_configs / 4)  # Divided by 4 because 4 streams
        print(f"  {name}: {count} configs rejected ({pct:.1f}%)")

    if bottlenecks[0][0] == "Too Few Trades":
        print("\n💡 RECOMMENDATION: The filter cascade is too strict.")
        print("   Signal frequency is the bottleneck, not edge quality.")
        print("   Consider relaxing: min_pbema_distance, regime_adx_threshold")
    elif bottlenecks[0][0] == "Negative PnL":
        print("\n💡 RECOMMENDATION: Signals are generated but unprofitable.")
        print("   The strategy edge is weak in current market conditions.")
        print("   Consider: Different RR values, or market regime filtering")
    else:
        print("\n💡 RECOMMENDATION: Signals exist and are profitable,")
        print("   but E[R] is below threshold. Lower the E[R] threshold slightly.")


if __name__ == "__main__":
    main()
