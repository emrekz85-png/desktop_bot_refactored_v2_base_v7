#!/usr/bin/env python3
"""
Test script for ROC (Rate of Change) Momentum Filter
Verifies the filter correctly blocks counter-trend entries
"""

import pandas as pd
import numpy as np
from strategies.ssl_flow import calculate_roc_filter

def test_roc_filter_basic():
    """Test basic ROC filter functionality"""
    print("\n" + "="*60)
    print("TEST 1: Basic ROC Filter Functionality")
    print("="*60)

    # Create test data with uptrend (rising prices)
    close_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    df = pd.DataFrame({'close': close_prices})

    # Test at index -1 (last candle)
    long_ok, short_ok, roc_value = calculate_roc_filter(df, index=-1, roc_period=10, roc_threshold=2.5)

    print(f"\nUptrend Test:")
    print(f"  Price change: {close_prices[0]} -> {close_prices[-1]} (+{close_prices[-1] - close_prices[0]})")
    print(f"  ROC Value: {roc_value:.2f}%")
    print(f"  LONG allowed: {long_ok}")
    print(f"  SHORT allowed: {short_ok}")
    print(f"  Expected: LONG=True, SHORT=False (strong uptrend blocks SHORT)")

    assert long_ok == True, "LONG should be allowed in uptrend"
    assert short_ok == False, "SHORT should be BLOCKED in strong uptrend"
    assert roc_value > 2.5, f"ROC should be > 2.5%, got {roc_value:.2f}%"
    print("  ✓ PASSED")


def test_roc_filter_downtrend():
    """Test ROC filter in downtrend"""
    print("\n" + "="*60)
    print("TEST 2: ROC Filter in Downtrend")
    print("="*60)

    # Create test data with downtrend (falling prices)
    close_prices = [110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
    df = pd.DataFrame({'close': close_prices})

    long_ok, short_ok, roc_value = calculate_roc_filter(df, index=-1, roc_period=10, roc_threshold=2.5)

    print(f"\nDowntrend Test:")
    print(f"  Price change: {close_prices[0]} -> {close_prices[-1]} ({close_prices[-1] - close_prices[0]})")
    print(f"  ROC Value: {roc_value:.2f}%")
    print(f"  LONG allowed: {long_ok}")
    print(f"  SHORT allowed: {short_ok}")
    print(f"  Expected: LONG=False, SHORT=True (strong downtrend blocks LONG)")

    assert long_ok == False, "LONG should be BLOCKED in strong downtrend"
    assert short_ok == True, "SHORT should be allowed in downtrend"
    assert roc_value < -2.5, f"ROC should be < -2.5%, got {roc_value:.2f}%"
    print("  ✓ PASSED")


def test_roc_filter_sideways():
    """Test ROC filter in sideways/consolidation"""
    print("\n" + "="*60)
    print("TEST 3: ROC Filter in Sideways Market")
    print("="*60)

    # Create test data with sideways movement
    close_prices = [100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100.5]
    df = pd.DataFrame({'close': close_prices})

    long_ok, short_ok, roc_value = calculate_roc_filter(df, index=-1, roc_period=10, roc_threshold=2.5)

    print(f"\nSideways Test:")
    print(f"  Price change: {close_prices[0]} -> {close_prices[-1]} ({close_prices[-1] - close_prices[0]:+.2f})")
    print(f"  ROC Value: {roc_value:.2f}%")
    print(f"  LONG allowed: {long_ok}")
    print(f"  SHORT allowed: {short_ok}")
    print(f"  Expected: LONG=True, SHORT=True (weak momentum allows both)")

    assert long_ok == True, "LONG should be allowed in sideways"
    assert short_ok == True, "SHORT should be allowed in sideways"
    assert -2.5 <= roc_value <= 2.5, f"ROC should be within [-2.5, +2.5]%, got {roc_value:.2f}%"
    print("  ✓ PASSED")


def test_roc_filter_edge_cases():
    """Test edge cases: insufficient data, NaN values"""
    print("\n" + "="*60)
    print("TEST 4: Edge Cases (Insufficient Data, NaN)")
    print("="*60)

    # Test 1: Insufficient data
    df_small = pd.DataFrame({'close': [100, 101, 102, 103, 104]})
    long_ok, short_ok, roc_value = calculate_roc_filter(df_small, index=-1, roc_period=10, roc_threshold=2.5)

    print(f"\nInsufficient Data Test (5 bars, need 10):")
    print(f"  LONG allowed: {long_ok}")
    print(f"  SHORT allowed: {short_ok}")
    print(f"  ROC Value: {roc_value:.2f}%")
    print(f"  Expected: Both allowed (conservative default)")

    assert long_ok == True, "Should allow LONG when insufficient data"
    assert short_ok == True, "Should allow SHORT when insufficient data"
    assert roc_value == 0.0, "ROC should be 0.0 when insufficient data"
    print("  ✓ PASSED")

    # Test 2: NaN at current index
    df_nan_current = pd.DataFrame({'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, np.nan]})
    long_ok, short_ok, roc_value = calculate_roc_filter(df_nan_current, index=-1, roc_period=10, roc_threshold=2.5)

    print(f"\nNaN at Current Index Test:")
    print(f"  LONG allowed: {long_ok}")
    print(f"  SHORT allowed: {short_ok}")
    print(f"  ROC Value: {roc_value:.2f}%")
    print(f"  Expected: Both allowed (NaN at current candle)")

    assert long_ok == True, "Should allow LONG when current close is NaN"
    assert short_ok == True, "Should allow SHORT when current close is NaN"
    assert roc_value == 0.0, "ROC should be 0.0 when current close is NaN"
    print("  ✓ PASSED")

    # Test 3: NaN at lookback index
    df_nan_lookback = pd.DataFrame({'close': [np.nan, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]})
    long_ok, short_ok, roc_value = calculate_roc_filter(df_nan_lookback, index=-1, roc_period=10, roc_threshold=2.5)

    print(f"\nNaN at Lookback Index Test:")
    print(f"  LONG allowed: {long_ok}")
    print(f"  SHORT allowed: {short_ok}")
    print(f"  ROC Value: {roc_value:.2f}%")
    print(f"  Expected: Both allowed (NaN at lookback candle)")

    assert long_ok == True, "Should allow LONG when lookback close is NaN"
    assert short_ok == True, "Should allow SHORT when lookback close is NaN"
    assert roc_value == 0.0, "ROC should be 0.0 when lookback close is NaN"
    print("  ✓ PASSED")


def test_roc_filter_threshold_tuning():
    """Test different ROC thresholds"""
    print("\n" + "="*60)
    print("TEST 5: ROC Threshold Tuning")
    print("="*60)

    # Create moderate uptrend: 100 -> 105 = 5% over 10 bars
    close_prices = [100, 100.5, 101, 101.5, 102, 102.5, 103, 103.5, 104, 104.5, 105]
    df = pd.DataFrame({'close': close_prices})

    thresholds = [1.0, 2.5, 5.0, 7.5]

    print(f"\nModerate Uptrend Test (5% gain over 10 bars):")
    for threshold in thresholds:
        long_ok, short_ok, roc_value = calculate_roc_filter(df, index=-1, roc_period=10, roc_threshold=threshold)
        print(f"  Threshold {threshold:4.1f}%: ROC={roc_value:+5.2f}%  LONG={long_ok}  SHORT={short_ok}")

    # At 2.5% threshold, should block SHORT
    long_ok, short_ok, roc_value = calculate_roc_filter(df, index=-1, roc_period=10, roc_threshold=2.5)
    assert short_ok == False, "SHORT should be blocked at 2.5% threshold with 5% uptrend"
    assert long_ok == True, "LONG should be allowed at 2.5% threshold with 5% uptrend"

    # At 7.5% threshold, should allow both
    long_ok, short_ok, roc_value = calculate_roc_filter(df, index=-1, roc_period=10, roc_threshold=7.5)
    assert short_ok == True, "SHORT should be allowed at 7.5% threshold with 5% uptrend"
    assert long_ok == True, "LONG should be allowed at 7.5% threshold with 5% uptrend"

    print("  ✓ PASSED")


def test_realistic_scenario():
    """Test with realistic crypto price data"""
    print("\n" + "="*60)
    print("TEST 6: Realistic Crypto Scenario")
    print("="*60)

    # Simulate BTC dump: 50000 -> 47000 in 10 bars (-6%)
    prices_dump = [50000, 49500, 49000, 48500, 48000, 47800, 47500, 47300, 47100, 47000, 47000]
    df_dump = pd.DataFrame({'close': prices_dump})

    long_ok, short_ok, roc_value = calculate_roc_filter(df_dump, index=-1, roc_period=10, roc_threshold=2.5)

    print(f"\nBTC Dump Scenario (-6% over 10 bars):")
    print(f"  Price: ${prices_dump[0]:,} -> ${prices_dump[-1]:,}")
    print(f"  ROC: {roc_value:.2f}%")
    print(f"  LONG allowed: {long_ok} (Expected: False - don't catch falling knife)")
    print(f"  SHORT allowed: {short_ok} (Expected: True)")

    assert long_ok == False, "Should block LONG in strong dump"
    assert short_ok == True, "Should allow SHORT in dump"
    print("  ✓ PASSED - Filter correctly blocks LONG in falling knife scenario")

    # Simulate BTC pump: 47000 -> 50000 in 10 bars (+6.4%)
    prices_pump = [47000, 47500, 48000, 48500, 49000, 49200, 49500, 49700, 49900, 50000, 50000]
    df_pump = pd.DataFrame({'close': prices_pump})

    long_ok, short_ok, roc_value = calculate_roc_filter(df_pump, index=-1, roc_period=10, roc_threshold=2.5)

    print(f"\nBTC Pump Scenario (+6.4% over 10 bars):")
    print(f"  Price: ${prices_pump[0]:,} -> ${prices_pump[-1]:,}")
    print(f"  ROC: {roc_value:.2f}%")
    print(f"  LONG allowed: {long_ok} (Expected: True)")
    print(f"  SHORT allowed: {short_ok} (Expected: False - don't short into strength)")

    assert long_ok == True, "Should allow LONG in pump"
    assert short_ok == False, "Should block SHORT in strong pump"
    print("  ✓ PASSED - Filter correctly blocks SHORT in strong uptrend scenario")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ROC MOMENTUM FILTER TEST SUITE")
    print("Testing calculate_roc_filter() function")
    print("="*60)

    try:
        test_roc_filter_basic()
        test_roc_filter_downtrend()
        test_roc_filter_sideways()
        test_roc_filter_edge_cases()
        test_roc_filter_threshold_tuning()
        test_realistic_scenario()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nROC Filter Summary:")
        print("  • Blocks SHORT when ROC > +2.5% (strong uptrend)")
        print("  • Blocks LONG when ROC < -2.5% (strong downtrend)")
        print("  • Allows both when |ROC| < 2.5% (weak momentum)")
        print("  • Handles edge cases gracefully (insufficient data, NaN)")
        print("  • Default: roc_period=10, roc_threshold=2.5%")
        print("  • Enabled by default in ssl_flow.py (v1.10.0)")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
