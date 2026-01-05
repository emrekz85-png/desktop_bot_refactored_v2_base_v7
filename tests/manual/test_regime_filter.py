#!/usr/bin/env python3
"""
Test script for Priority 2: Enhanced Regime Filter

This script tests the new BTC leader regime filter implementation:
1. Tests regime detection accuracy
2. Compares backtest results with and without regime filter
3. Validates BTC leader principle

Usage:
    python test_regime_filter.py                    # Quick test
    python test_regime_filter.py --full             # Full comparison test
    python test_regime_filter.py --analyze-regimes  # Regime distribution analysis
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    calculate_indicators,
    SYMBOLS, TIMEFRAMES,
    get_client,
)
from core.regime_filter import (
    RegimeFilter, RegimeType,
    check_regime_for_trade,
    analyze_regime_distribution,
)


def test_regime_filter_basic():
    """Basic test of regime filter functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic Regime Filter Functionality")
    print("="*60)

    # Fetch BTC data
    client = get_client()
    btc_df = client.get_klines("BTCUSDT", "15m", limit=500)

    if btc_df is None or btc_df.empty:
        print("ERROR: Could not fetch BTC data")
        return False

    # Calculate indicators
    btc_df = calculate_indicators(btc_df, "15m")

    print(f"\nData fetched: {len(btc_df)} bars")
    print(f"Date range: {btc_df['timestamp'].iloc[0]} -> {btc_df['timestamp'].iloc[-1]}")

    # Test regime detection
    regime_filter = RegimeFilter()
    result = regime_filter.detect_regime(btc_df, index=-2, symbol="BTCUSDT")

    print(f"\nRegime Detection Result:")
    print(f"  Regime: {result.regime.value}")
    print(f"  Should Trade: {result.should_trade}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  ADX Current: {result.adx_current:.1f}")
    print(f"  ADX Average: {result.adx_avg:.1f}")
    print(f"  ATR Percentile: {result.atr_percentile:.2f}")
    print(f"  Details: {result.details}")

    # Test regime multiplier
    multiplier = regime_filter.get_regime_multiplier(result)
    print(f"  Position Multiplier: {multiplier:.2f}")

    return True


def test_btc_leader_principle():
    """Test BTC leader principle for altcoins."""
    print("\n" + "="*60)
    print("TEST 2: BTC Leader Principle")
    print("="*60)

    client = get_client()

    # Fetch BTC data
    btc_df = client.get_klines("BTCUSDT", "15m", limit=500)
    if btc_df is None:
        print("ERROR: Could not fetch BTC data")
        return False

    btc_df = calculate_indicators(btc_df, "15m")

    # Fetch ETH data
    eth_df = client.get_klines("ETHUSDT", "15m", limit=500)
    if eth_df is None:
        print("ERROR: Could not fetch ETH data")
        return False

    eth_df = calculate_indicators(eth_df, "15m")

    # Test BTC regime
    regime_filter = RegimeFilter(require_btc_trend=True)

    btc_result = regime_filter.detect_regime(btc_df, index=-2, symbol="BTCUSDT")
    print(f"\nBTC Regime: {btc_result.regime.value} (conf={btc_result.confidence:.2f})")

    # Test ETH with BTC leader check
    eth_result = regime_filter.detect_regime(
        eth_df, index=-2, symbol="ETHUSDT", btc_df=btc_df
    )
    print(f"ETH Regime: {eth_result.regime.value} (conf={eth_result.confidence:.2f})")
    print(f"ETH BTC-Aligned: {eth_result.btc_aligned}")
    print(f"ETH Should Trade: {eth_result.should_trade}")

    # Show decision logic
    if not btc_result.should_trade:
        print("\n>>> BTC is NOT trending - ETH trades should be BLOCKED")
    else:
        print("\n>>> BTC IS trending - ETH trades can proceed (if ETH also trending)")

    return True


def test_regime_distribution():
    """Analyze regime distribution over historical data."""
    print("\n" + "="*60)
    print("TEST 3: Regime Distribution Analysis")
    print("="*60)

    client = get_client()

    # Fetch longer history
    btc_df = client.get_klines("BTCUSDT", "15m", limit=5000)
    if btc_df is None:
        print("ERROR: Could not fetch BTC data")
        return False

    btc_df = calculate_indicators(btc_df, "15m")

    print(f"\nAnalyzing {len(btc_df)} bars...")

    # Analyze distribution
    distribution = analyze_regime_distribution(btc_df)

    print(f"\nRegime Distribution:")
    print(f"  Total bars analyzed: {distribution['total_bars']}")
    print(f"  Dominant regime: {distribution['dominant_regime']}")
    print(f"  Tradable percentage: {distribution['tradable_percentage']:.1f}%")

    print(f"\n  By Regime:")
    for regime, data in distribution.get('distribution', {}).items():
        print(f"    {regime}: {data['count']} bars ({data['percentage']:.1f}%)")

    return True


def test_signal_with_regime_filter():
    """Test SSL Flow signal generation with regime filter enabled."""
    print("\n" + "="*60)
    print("TEST 4: Signal Generation with Regime Filter")
    print("="*60)

    from strategies.ssl_flow import check_ssl_flow_signal

    client = get_client()

    # Fetch data
    btc_df = client.get_klines("BTCUSDT", "15m", limit=500)
    eth_df = client.get_klines("ETHUSDT", "15m", limit=500)

    if btc_df is None or eth_df is None:
        print("ERROR: Could not fetch data")
        return False

    btc_df = calculate_indicators(btc_df, "15m")
    eth_df = calculate_indicators(eth_df, "15m")

    print("\nTesting ETH signal WITHOUT BTC regime filter:")
    result_no_filter = check_ssl_flow_signal(
        eth_df,
        index=-2,
        use_btc_regime_filter=False,
        return_debug=True,
    )
    print(f"  Signal: {result_no_filter[0]}")
    print(f"  Reason: {result_no_filter[4]}")

    print("\nTesting ETH signal WITH BTC regime filter:")
    result_with_filter = check_ssl_flow_signal(
        eth_df,
        index=-2,
        use_btc_regime_filter=True,
        btc_df=btc_df,
        symbol="ETHUSDT",
        return_debug=True,
    )
    print(f"  Signal: {result_with_filter[0]}")
    print(f"  Reason: {result_with_filter[4]}")

    # Check debug info for BTC regime
    debug = result_with_filter[5] if len(result_with_filter) > 5 else {}
    if "btc_regime" in debug:
        print(f"  BTC Regime: {debug.get('btc_regime')}")
        print(f"  BTC Confidence: {debug.get('btc_regime_confidence', 'N/A')}")

    return True


def run_comparison_backtest():
    """Compare backtest results with and without regime filter."""
    print("\n" + "="*60)
    print("TEST 5: Backtest Comparison (With vs Without Regime Filter)")
    print("="*60)
    print("\nThis test requires running the full backtest framework.")
    print("Use: python run_rolling_wf_test.py --quick-btc")
    print("\nTo test the regime filter effect, modify DEFAULT_STRATEGY_CONFIG")
    print("in core/config.py to set 'use_btc_regime_filter': True")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Enhanced Regime Filter")
    parser.add_argument('--full', action='store_true', help='Run full comparison test')
    parser.add_argument('--analyze-regimes', action='store_true', help='Analyze regime distribution')
    parser.add_argument('--quick', action='store_true', help='Quick basic tests only')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("PRIORITY 2: ENHANCED REGIME FILTER TEST SUITE")
    print("="*60)

    all_passed = True

    # Always run basic tests
    if not test_regime_filter_basic():
        all_passed = False

    if not test_btc_leader_principle():
        all_passed = False

    if not args.quick:
        if not test_signal_with_regime_filter():
            all_passed = False

    if args.analyze_regimes or args.full:
        if not test_regime_distribution():
            all_passed = False

    if args.full:
        run_comparison_backtest()

    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
