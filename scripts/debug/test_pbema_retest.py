#!/usr/bin/env python3
"""
Test Script for PBEMA Retest Strategy

This script tests the PBEMA retest detection on BTCUSDT 15m data.
It will scan recent data for PBEMA breakout + retest setups.

Usage:
    python test_pbema_retest.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import get_client, calculate_indicators, set_backtest_mode
from strategies import check_pbema_retest_signal


def fetch_test_data(symbol="BTCUSDT", timeframe="15m", limit=500):
    """Fetch recent data for testing."""
    set_backtest_mode(True)
    client = get_client()

    print(f"Fetching {limit} candles of {symbol} {timeframe}...")
    df = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)

    print(f"Calculating indicators...")
    df = calculate_indicators(df, timeframe=timeframe)

    print(f"Data ready: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df


def test_pbema_retest_detection():
    """Test PBEMA retest signal detection."""
    print("="*70)
    print("PBEMA RETEST STRATEGY TEST")
    print("="*70)
    print()

    # Fetch data
    df = fetch_test_data()

    # Test different parameter configurations
    configs = [
        {
            "name": "Baseline (minimal filters)",
            "params": {
                "require_at_confirmation": False,
                "require_multiple_retests": False,
                "min_rr": 1.5,
            }
        },
        {
            "name": "Conservative (AT + multiple retests)",
            "params": {
                "require_at_confirmation": True,
                "require_multiple_retests": True,
                "min_retests": 2,
                "min_rr": 2.0,
            }
        },
        {
            "name": "Aggressive (low RR, no filters)",
            "params": {
                "require_at_confirmation": False,
                "require_multiple_retests": False,
                "min_rr": 1.2,
            }
        },
    ]

    for config in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {config['name']}")
        print(f"{'='*70}\n")

        signals_found = 0

        # Scan last 100 candles for signals
        for i in range(len(df) - 100, len(df) - 10):
            signal_type, entry, tp, sl, reason, debug = check_pbema_retest_signal(
                df,
                index=i,
                return_debug=True,
                **config["params"]
            )

            if signal_type is not None:
                signals_found += 1
                candle_time = df.index[i]

                rr = (tp - entry) / (entry - sl) if signal_type == "LONG" else (entry - tp) / (sl - entry)

                print(f"[{candle_time}] {signal_type} Signal #{signals_found}")
                print(f"  Entry: {entry:.2f} | TP: {tp:.2f} | SL: {sl:.2f} | R:R = {rr:.2f}")
                print(f"  Reason: {reason}")
                print(f"  Breakout: {debug['breakout_direction']} at candle {debug['breakout_candle_idx']}")
                print(f"  Breakout distance: {debug['breakout_distance']*100:.2f}%")
                print(f"  Retest count: {debug['retest_count']}")
                print(f"  Wick ratio: {debug['wick_ratio']:.2f}")
                print()

        print(f"Total signals found: {signals_found}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


def test_single_signal():
    """Test signal detection on a single recent candle with full debug output."""
    print("="*70)
    print("SINGLE CANDLE DEBUG TEST")
    print("="*70)
    print()

    df = fetch_test_data()

    # Test on second-to-last candle (index=-2)
    signal_type, entry, tp, sl, reason, debug = check_pbema_retest_signal(
        df,
        index=-2,
        require_at_confirmation=False,
        return_debug=True
    )

    print(f"Signal Type: {signal_type}")
    print(f"Entry: {entry}")
    print(f"TP: {tp}")
    print(f"SL: {sl}")
    print(f"Reason: {reason}")
    print()
    print("Debug Info:")
    for key, value in debug.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PBEMA Retest Strategy")
    parser.add_argument("--single", action="store_true", help="Test single candle with debug output")
    parser.add_argument("--scan", action="store_true", help="Scan for signals (default)")

    args = parser.parse_args()

    if args.single:
        test_single_signal()
    else:
        test_pbema_retest_detection()
