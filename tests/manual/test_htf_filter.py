#!/usr/bin/env python3
"""
HTF Trend Filter Test Script

Tests the 4H Higher Timeframe trend filter implementation:
1. Loads BTCUSDT 15m data (signal generation)
2. Loads BTCUSDT 4h data (trend filter)
3. Compares signals with and without HTF filter
4. Shows which signals would be blocked by the filter
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from core import set_backtest_mode
from core.binance_client import BinanceClient
from core.indicators import calculate_indicators
from strategies.ssl_flow import check_ssl_flow_signal, detect_htf_trend

set_backtest_mode(True)


def load_data(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load and prepare data with indicators."""
    client = BinanceClient()

    # Calculate candle count
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end_dt - start_dt).days + 60  # Extra for warmup

    tf_minutes = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
    minutes_per_candle = tf_minutes.get(timeframe, 15)
    candles = int((days * 24 * 60) / minutes_per_candle)

    print(f"Loading {symbol} {timeframe}: {candles} candles...")

    df = client.get_klines_paginated(
        symbol=symbol,
        interval=timeframe,
        total_candles=min(candles, 10000),
    )

    if df is None or len(df) < 100:
        print(f"Error: Could not load enough data for {symbol} {timeframe}")
        return None

    # Set timestamp as index if needed
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)

    # Calculate indicators
    df = calculate_indicators(df, timeframe)

    # Filter to date range
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df.index >= start_ts) & (df.index < end_ts)]

    print(f"  → {len(df)} candles in date range")
    return df


def find_signals_without_htf(df_15m: pd.DataFrame, min_rr: float = 2.0) -> List[Dict]:
    """Find all signals without HTF filter."""
    signals = []

    for i in range(200, len(df_15m) - 2):
        result = check_ssl_flow_signal(
            df_15m, i,
            min_rr=min_rr,
            rsi_limit=70,
            return_debug=True,
            use_htf_filter=False,  # No HTF filter
        )

        if result and len(result) >= 6:
            signal_type, entry, tp, sl, reason, debug_info = result[:6]

            if signal_type is not None:
                signals.append({
                    "index": i,
                    "time": df_15m.index[i],
                    "type": signal_type,
                    "entry": entry,
                    "tp": tp,
                    "sl": sl,
                    "reason": reason,
                    "price": df_15m["close"].iloc[i],
                })

    return signals


def test_htf_filter_on_signals(
    signals: List[Dict],
    df_15m: pd.DataFrame,
    df_4h: pd.DataFrame,
    htf_method: str = "baseline",
    htf_lookback: int = 3,
) -> Tuple[List[Dict], List[Dict]]:
    """Test which signals pass/fail the HTF filter."""
    passed = []
    blocked = []

    for sig in signals:
        i = sig["index"]
        signal_time = sig["time"]

        # Find corresponding 4h bar
        htf_mask = df_4h.index <= signal_time
        if not htf_mask.any():
            # No HTF data, allow signal
            passed.append(sig)
            continue

        htf_idx = htf_mask.sum() - 1

        # Detect HTF trend
        htf_df_slice = df_4h.iloc[:htf_idx + 1]
        htf_trend, htf_confidence = detect_htf_trend(htf_df_slice, method=htf_method, lookback=htf_lookback)

        sig["htf_trend"] = htf_trend
        sig["htf_confidence"] = htf_confidence

        # Check if blocked
        if htf_trend == "UP" and sig["type"] == "SHORT":
            sig["block_reason"] = f"HTF UP blocks SHORT (conf={htf_confidence:.2f})"
            blocked.append(sig)
        elif htf_trend == "DOWN" and sig["type"] == "LONG":
            sig["block_reason"] = f"HTF DOWN blocks LONG (conf={htf_confidence:.2f})"
            blocked.append(sig)
        else:
            passed.append(sig)

    return passed, blocked


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HTF Trend Filter Test")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to test")
    parser.add_argument("--start-date", default="2025-06-01", help="Start date")
    parser.add_argument("--end-date", default="2025-12-01", help="End date")
    parser.add_argument("--method", default="baseline", choices=["baseline", "ema"],
                       help="HTF trend detection method")
    parser.add_argument("--lookback", type=int, default=3,
                       help="HTF trend lookback candles")

    args = parser.parse_args()

    print("=" * 70)
    print("HTF TREND FILTER TEST")
    print("=" * 70)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Method: {args.method}")
    print()

    # Load data
    print("-" * 70)
    print("Loading data...")
    print("-" * 70)

    df_15m = load_data(args.symbol, "15m", args.start_date, args.end_date)
    df_4h = load_data(args.symbol, "4h", args.start_date, args.end_date)

    if df_15m is None or df_4h is None:
        print("Error loading data!")
        return

    # Find signals without HTF filter
    print("\n" + "-" * 70)
    print("Finding signals (without HTF filter)...")
    print("-" * 70)

    signals = find_signals_without_htf(df_15m)
    print(f"Found {len(signals)} signals")

    if not signals:
        print("No signals found!")
        return

    # Test HTF filter
    print("\n" + "-" * 70)
    print("Testing HTF filter...")
    print("-" * 70)

    passed, blocked = test_htf_filter_on_signals(signals, df_15m, df_4h, args.method, args.lookback)

    print(f"\nResults:")
    print(f"  Passed:  {len(passed)} signals ({len(passed)/len(signals)*100:.1f}%)")
    print(f"  Blocked: {len(blocked)} signals ({len(blocked)/len(signals)*100:.1f}%)")

    # Show blocked signals
    if blocked:
        print("\n" + "-" * 70)
        print("BLOCKED SIGNALS (Counter-trend trades prevented)")
        print("-" * 70)

        for sig in blocked[:20]:  # Show first 20
            print(f"\n  {sig['time']}")
            print(f"    Type: {sig['type']} @ ${sig['price']:.2f}")
            print(f"    HTF Trend: {sig['htf_trend']} (conf={sig['htf_confidence']:.2f})")
            print(f"    Block: {sig['block_reason']}")

        if len(blocked) > 20:
            print(f"\n  ... and {len(blocked) - 20} more blocked signals")

    # HTF trend distribution
    print("\n" + "-" * 70)
    print("HTF TREND DISTRIBUTION")
    print("-" * 70)

    all_signals = passed + blocked
    htf_up = sum(1 for s in all_signals if s.get("htf_trend") == "UP")
    htf_down = sum(1 for s in all_signals if s.get("htf_trend") == "DOWN")
    htf_neutral = sum(1 for s in all_signals if s.get("htf_trend") == "NEUTRAL")

    print(f"  HTF UP:      {htf_up} ({htf_up/len(all_signals)*100:.1f}%)")
    print(f"  HTF DOWN:    {htf_down} ({htf_down/len(all_signals)*100:.1f}%)")
    print(f"  HTF NEUTRAL: {htf_neutral} ({htf_neutral/len(all_signals)*100:.1f}%)")

    # Signal type distribution
    print("\n" + "-" * 70)
    print("SIGNAL TYPE DISTRIBUTION")
    print("-" * 70)

    longs = sum(1 for s in signals if s["type"] == "LONG")
    shorts = sum(1 for s in signals if s["type"] == "SHORT")

    longs_passed = sum(1 for s in passed if s["type"] == "LONG")
    shorts_passed = sum(1 for s in passed if s["type"] == "SHORT")

    longs_blocked = sum(1 for s in blocked if s["type"] == "LONG")
    shorts_blocked = sum(1 for s in blocked if s["type"] == "SHORT")

    print(f"  LONG signals:  {longs} total → {longs_passed} passed, {longs_blocked} blocked")
    print(f"  SHORT signals: {shorts} total → {shorts_passed} passed, {shorts_blocked} blocked")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if len(blocked) == 0:
        print("\n⚠️  HTF filter blocked NO signals!")
        print("    This could mean:")
        print("    1. All signals already align with HTF trend")
        print("    2. HTF is mostly NEUTRAL")
        print("    3. Filter needs parameter tuning")
    else:
        block_rate = len(blocked) / len(signals) * 100
        print(f"\n✅ HTF filter blocked {len(blocked)} counter-trend signals ({block_rate:.1f}%)")
        print(f"   These are trades that go AGAINST the 4H trend direction")

        if block_rate > 50:
            print("\n⚠️  WARNING: Filter is very aggressive (>50% blocked)")
            print("   Consider loosening the HTF trend detection parameters")
        elif block_rate < 10:
            print("\n⚠️  WARNING: Filter is very permissive (<10% blocked)")
            print("   Most signals already align with HTF trend")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
