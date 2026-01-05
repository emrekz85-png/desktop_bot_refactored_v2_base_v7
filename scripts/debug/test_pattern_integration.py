#!/usr/bin/env python3
"""
Quick Test: Pattern Integration

Tests all 7 patterns integrated into the filter system.

Usage:
    python test_pattern_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import get_client, calculate_indicators, set_backtest_mode
from runners.run_filter_combo_test import apply_filters
from core.at_scenario_analyzer import check_core_signal


def test_pattern_filters():
    """Test all pattern filters on recent data."""
    print("="*70)
    print("PATTERN FILTER INTEGRATION TEST")
    print("="*70)
    print()

    # Setup
    set_backtest_mode(True)
    client = get_client()

    # Fetch data
    symbol = "BTCUSDT"
    timeframe = "15m"
    limit = 500

    print(f"Fetching {symbol} {timeframe} data ({limit} candles)...")
    df = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)

    if df.empty:
        print("❌ Failed to fetch data")
        return

    # Set timestamp as index
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)

    print(f"✅ Fetched {len(df)} candles")
    print()

    print("Calculating indicators...")
    df = calculate_indicators(df, timeframe=timeframe)
    print("✅ Indicators ready")
    print()

    # Test each pattern filter individually
    pattern_tests = [
        ("No Filters", {}),
        ("Pattern 3: Liquidity Grab", {"use_liquidity_grab": True}),
        ("Pattern 4: SSL Slope Filter", {"use_ssl_slope_filter": True}),
        ("Pattern 5: HTF Bounce", {"use_htf_bounce": True}),
        ("Pattern 6: Momentum Loss", {"use_momentum_loss": True}),
        ("Pattern 7: SSL Dynamic Support", {"use_ssl_dynamic_support": True}),
        ("All Patterns Combined", {
            "use_liquidity_grab": True,
            "use_ssl_slope_filter": True,
            "use_htf_bounce": True,
            "use_momentum_loss": True,
            "use_ssl_dynamic_support": True,
        }),
    ]

    results = []

    for test_name, filter_flags in pattern_tests:
        print(f"\n{'='*70}")
        print(f"Testing: {test_name}")
        print(f"{'='*70}\n")

        signals_checked = 0
        signals_generated = 0
        signals_passed = 0
        failures = {}

        # Scan candles for signals
        for i in range(100, len(df) - 10):
            signals_checked += 1

            # Get core signal
            signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)

            if signal_type is None:
                continue

            signals_generated += 1

            # Apply filters
            passed, filter_reason = apply_filters(
                df=df,
                index=i,
                signal_type=signal_type,
                entry_price=entry,
                sl_price=sl,
                use_regime_filter=False,  # Disable regime for pattern testing
                **filter_flags
            )

            if passed:
                signals_passed += 1
            else:
                # Track failure reasons
                failures[filter_reason] = failures.get(filter_reason, 0) + 1

        # Calculate stats
        pass_rate = (signals_passed / signals_generated * 100) if signals_generated > 0 else 0

        print(f"Signals Checked:   {signals_checked}")
        print(f"Core Signals:      {signals_generated}")
        print(f"Passed Filters:    {signals_passed}")
        print(f"Pass Rate:         {pass_rate:.1f}%")

        if failures:
            print(f"\nTop Failure Reasons:")
            sorted_failures = sorted(failures.items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_failures[:5]:
                print(f"  - {reason}: {count}")

        results.append({
            "test": test_name,
            "checked": signals_checked,
            "generated": signals_generated,
            "passed": signals_passed,
            "pass_rate": pass_rate,
        })

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Test':<35} {'Signals':<10} {'Passed':<10} {'Pass Rate':<10}")
    print("-"*70)

    for r in results:
        print(f"{r['test']:<35} {r['generated']:<10} {r['passed']:<10} {r['pass_rate']:<10.1f}%")

    print("="*70)
    print()

    # Recommendations
    print("RECOMMENDATIONS:")
    print("-"*70)

    baseline_passed = results[0]['passed']

    for r in results[1:]:
        if r['test'] == "All Patterns Combined":
            continue

        delta = r['passed'] - baseline_passed
        if delta < 0:
            impact = f"❌ Filters out {abs(delta)} signals"
        elif delta > 0:
            impact = f"✅ Adds {delta} signals"
        else:
            impact = "⚪ No impact"

        print(f"{r['test']:<35} {impact}")

    print("="*70)


if __name__ == "__main__":
    test_pattern_filters()
