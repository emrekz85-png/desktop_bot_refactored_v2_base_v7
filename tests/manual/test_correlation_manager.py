#!/usr/bin/env python3
"""
Test Correlation Management - Priority 4 Validation

This script tests the correlation management module to ensure:
1. Position limits are enforced (max 2 per direction)
2. Position size reduction works for correlated assets
3. Effective position calculation is accurate

Usage:
    python test_correlation_manager.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.correlation_manager import (
    CorrelationManager,
    check_correlation_risk,
    calculate_portfolio_effective_positions,
    DEFAULT_CORRELATION_MATRIX,
)


def test_basic_functionality():
    """Test basic correlation manager functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)

    cm = CorrelationManager()

    # Test correlation lookup
    btc_eth_corr = cm.get_correlation("BTCUSDT", "ETHUSDT")
    print(f"\nBTC-ETH correlation: {btc_eth_corr:.2f}")
    assert btc_eth_corr > 0.8, "BTC-ETH should be highly correlated"

    btc_link_corr = cm.get_correlation("BTCUSDT", "LINKUSDT")
    print(f"BTC-LINK correlation: {btc_link_corr:.2f}")
    assert btc_link_corr > 0.7, "BTC-LINK should be correlated"

    # Test self-correlation
    self_corr = cm.get_correlation("BTCUSDT", "BTCUSDT")
    print(f"BTC-BTC correlation: {self_corr:.2f}")
    assert self_corr == 1.0, "Self-correlation should be 1.0"

    print("\n[PASS] Basic functionality test passed")
    return True


def test_position_direction_limit():
    """Test max positions per direction limit."""
    print("\n" + "="*60)
    print("TEST 2: Position Direction Limit (Max 2 per direction)")
    print("="*60)

    cm = CorrelationManager(max_positions_same_direction=2)

    # First LONG position - should be allowed
    result1 = cm.check_new_position(
        symbol="BTCUSDT",
        direction="LONG",
        base_position_size=35.0,
        open_positions={},
    )
    print(f"\n1st LONG (BTCUSDT): can_open={result1.can_open}, reason='{result1.reason}'")
    assert result1.can_open, "First LONG should be allowed"

    # Second LONG position - should be allowed
    open_pos = {"BTCUSDT": {"direction": "LONG", "size": 35.0}}
    result2 = cm.check_new_position(
        symbol="ETHUSDT",
        direction="LONG",
        base_position_size=35.0,
        open_positions=open_pos,
    )
    print(f"2nd LONG (ETHUSDT): can_open={result2.can_open}, reason='{result2.reason}'")
    assert result2.can_open, "Second LONG should be allowed"

    # Third LONG position - should be BLOCKED
    open_pos = {
        "BTCUSDT": {"direction": "LONG", "size": 35.0},
        "ETHUSDT": {"direction": "LONG", "size": 35.0},
    }
    result3 = cm.check_new_position(
        symbol="LINKUSDT",
        direction="LONG",
        base_position_size=35.0,
        open_positions=open_pos,
    )
    print(f"3rd LONG (LINKUSDT): can_open={result3.can_open}, reason='{result3.reason}'")
    assert not result3.can_open, "Third LONG should be blocked"

    # SHORT position should still be allowed
    result4 = cm.check_new_position(
        symbol="LINKUSDT",
        direction="SHORT",
        base_position_size=35.0,
        open_positions=open_pos,
    )
    print(f"1st SHORT (LINKUSDT): can_open={result4.can_open}, reason='{result4.reason}'")
    assert result4.can_open, "First SHORT should be allowed even with 2 LONGs"

    print("\n[PASS] Position direction limit test passed")
    return True


def test_position_size_reduction():
    """Test position size reduction for correlated assets."""
    print("\n" + "="*60)
    print("TEST 3: Position Size Reduction for Correlated Assets")
    print("="*60)

    cm = CorrelationManager(
        high_correlation_threshold=0.80,
        position_reduction_factor=0.50,
    )

    # First position - full size
    result1 = cm.check_new_position(
        symbol="BTCUSDT",
        direction="LONG",
        base_position_size=35.0,
        open_positions={},
    )
    print(f"\n1st position (BTCUSDT LONG):")
    print(f"   Original size: ${result1.original_size:.2f}")
    print(f"   Adjusted size: ${result1.adjusted_size:.2f}")
    print(f"   Multiplier: {result1.size_multiplier:.0%}")
    assert result1.size_multiplier == 1.0, "First position should be full size"

    # Second position (correlated) - reduced size
    open_pos = {"BTCUSDT": {"direction": "LONG", "size": 35.0}}
    result2 = cm.check_new_position(
        symbol="ETHUSDT",
        direction="LONG",
        base_position_size=35.0,
        open_positions=open_pos,
    )
    print(f"\n2nd position (ETHUSDT LONG, correlated with BTC):")
    print(f"   Original size: ${result2.original_size:.2f}")
    print(f"   Adjusted size: ${result2.adjusted_size:.2f}")
    print(f"   Multiplier: {result2.size_multiplier:.0%}")
    print(f"   Correlation with open: {result2.correlation_with_open:.2f}")
    assert result2.size_multiplier < 1.0, "Second correlated position should be reduced"

    # Opposite direction - full size (hedging effect)
    result3 = cm.check_new_position(
        symbol="ETHUSDT",
        direction="SHORT",
        base_position_size=35.0,
        open_positions=open_pos,
    )
    print(f"\n3rd position (ETHUSDT SHORT, opposite direction):")
    print(f"   Original size: ${result3.original_size:.2f}")
    print(f"   Adjusted size: ${result3.adjusted_size:.2f}")
    print(f"   Multiplier: {result3.size_multiplier:.0%}")
    assert result3.size_multiplier == 1.0, "Opposite direction should not be reduced"

    print("\n[PASS] Position size reduction test passed")
    return True


def test_effective_positions():
    """Test effective position calculation."""
    print("\n" + "="*60)
    print("TEST 4: Effective Position Calculation")
    print("="*60)

    cm = CorrelationManager()

    # Single position = 1.0 effective
    positions1 = {"BTCUSDT": {"direction": "LONG", "size": 35.0}}
    eff1 = cm.calculate_effective_positions(positions1)
    print(f"\n1 position (BTC LONG):")
    print(f"   Actual: 1, Effective: {eff1:.2f}")
    assert eff1 == 1.0, "Single position should have 1.0 effective"

    # Two correlated LONGs - less than 2 effective
    positions2 = {
        "BTCUSDT": {"direction": "LONG", "size": 35.0},
        "ETHUSDT": {"direction": "LONG", "size": 35.0},
    }
    eff2 = cm.calculate_effective_positions(positions2)
    print(f"\n2 positions (BTC+ETH LONG, correlated):")
    print(f"   Actual: 2, Effective: {eff2:.2f}")
    print(f"   Diversification loss: {(2-eff2)/2*100:.0f}%")
    assert eff2 < 2.0, "Two correlated positions should have < 2 effective"

    # Three correlated LONGs - even less effective
    positions3 = {
        "BTCUSDT": {"direction": "LONG", "size": 35.0},
        "ETHUSDT": {"direction": "LONG", "size": 35.0},
        "LINKUSDT": {"direction": "LONG", "size": 35.0},
    }
    eff3 = cm.calculate_effective_positions(positions3)
    print(f"\n3 positions (BTC+ETH+LINK LONG, all correlated):")
    print(f"   Actual: 3, Effective: {eff3:.2f}")
    print(f"   Diversification loss: {(3-eff3)/3*100:.0f}%")
    assert eff3 < eff2 + 1, "Three correlated positions should not be fully additive"

    # Mixed directions - better diversification
    positions4 = {
        "BTCUSDT": {"direction": "LONG", "size": 35.0},
        "ETHUSDT": {"direction": "SHORT", "size": 35.0},
    }
    eff4 = cm.calculate_effective_positions(positions4)
    print(f"\n2 positions (BTC LONG, ETH SHORT - hedged):")
    print(f"   Actual: 2, Effective: {eff4:.2f}")
    assert eff4 > eff2, "Opposite directions should be more diversified"

    print("\n[PASS] Effective position calculation test passed")
    return True


def test_portfolio_summary():
    """Test portfolio summary functionality."""
    print("\n" + "="*60)
    print("TEST 5: Portfolio Summary")
    print("="*60)

    cm = CorrelationManager()

    # Register some positions
    cm.register_position("BTCUSDT", "LONG", 35.0)
    cm.register_position("ETHUSDT", "LONG", 17.5)
    cm.register_position("LINKUSDT", "SHORT", 35.0)

    summary = cm.get_portfolio_summary()

    print(f"\nPortfolio Summary:")
    print(f"   Total Positions: {summary['total_positions']}")
    print(f"   Long Positions: {summary['long_positions']}")
    print(f"   Short Positions: {summary['short_positions']}")
    print(f"   Effective Positions: {summary['effective_positions']}")
    print(f"   Concentration Ratio: {summary['concentration_ratio']:.2f}")
    print(f"   Symbols: {summary['symbols']}")

    assert summary['total_positions'] == 3
    assert summary['long_positions'] == 2
    assert summary['short_positions'] == 1
    assert summary['effective_positions'] < 3.0

    # Close a position
    cm.close_position("BTCUSDT")
    summary2 = cm.get_portfolio_summary()
    assert summary2['total_positions'] == 2
    print(f"\nAfter closing BTCUSDT: {summary2['total_positions']} positions")

    print("\n[PASS] Portfolio summary test passed")
    return True


def test_hedge_suggestion():
    """Test hedge suggestion functionality."""
    print("\n" + "="*60)
    print("TEST 6: Hedge Suggestion")
    print("="*60)

    cm = CorrelationManager()

    # Get hedge suggestion for BTC LONG
    hedge = cm.suggest_hedge("BTCUSDT", "LONG")

    if hedge:
        print(f"\nHedge suggestion for BTCUSDT LONG:")
        print(f"   Symbol: {hedge['symbol']}")
        print(f"   Direction: {hedge['direction']}")
        print(f"   Correlation: {hedge['correlation']:.2f}")
        print(f"   Reason: {hedge['reason']}")
        assert hedge['direction'] == "SHORT", "Hedge should be opposite direction"
    else:
        print("\nNo hedge suggestion found (all pairs highly correlated)")

    print("\n[PASS] Hedge suggestion test passed")
    return True


def test_real_world_scenario():
    """Test real-world trading scenario."""
    print("\n" + "="*60)
    print("TEST 7: Real-World Trading Scenario")
    print("="*60)

    cm = CorrelationManager(
        max_positions_same_direction=2,
        high_correlation_threshold=0.80,
        position_reduction_factor=0.50,
    )

    base_size = 35.0  # $35 risk per trade

    print("\nSimulating a trading session...")
    print("-"*50)

    # Signal 1: BTC LONG
    result1 = cm.check_new_position("BTCUSDT", "LONG", base_size, cm.get_open_positions())
    print(f"\n1. BTC LONG signal:")
    print(f"   Decision: {'OPEN' if result1.can_open else 'SKIP'}")
    print(f"   Size: ${result1.adjusted_size:.2f} (from ${base_size:.2f})")
    if result1.can_open:
        cm.register_position("BTCUSDT", "LONG", result1.adjusted_size)

    # Signal 2: ETH LONG (correlated with BTC)
    result2 = cm.check_new_position("ETHUSDT", "LONG", base_size, cm.get_open_positions())
    print(f"\n2. ETH LONG signal:")
    print(f"   Decision: {'OPEN' if result2.can_open else 'SKIP'}")
    print(f"   Size: ${result2.adjusted_size:.2f} (from ${base_size:.2f})")
    print(f"   Reason: {result2.reason}")
    if result2.can_open:
        cm.register_position("ETHUSDT", "LONG", result2.adjusted_size)

    # Signal 3: LINK LONG (would be 3rd LONG - blocked)
    result3 = cm.check_new_position("LINKUSDT", "LONG", base_size, cm.get_open_positions())
    print(f"\n3. LINK LONG signal:")
    print(f"   Decision: {'OPEN' if result3.can_open else 'SKIP'}")
    print(f"   Reason: {result3.reason}")
    if result3.can_open:
        cm.register_position("LINKUSDT", "LONG", result3.adjusted_size)

    # Signal 4: SOL SHORT (opposite direction - allowed)
    result4 = cm.check_new_position("SOLUSDT", "SHORT", base_size, cm.get_open_positions())
    print(f"\n4. SOL SHORT signal:")
    print(f"   Decision: {'OPEN' if result4.can_open else 'SKIP'}")
    print(f"   Size: ${result4.adjusted_size:.2f}")
    if result4.can_open:
        cm.register_position("SOLUSDT", "SHORT", result4.adjusted_size)

    # Final portfolio summary
    summary = cm.get_portfolio_summary()
    print("\n" + "-"*50)
    print("Final Portfolio:")
    print(f"   Positions: {summary['total_positions']} ({summary['long_positions']}L, {summary['short_positions']}S)")
    print(f"   Effective Positions: {summary['effective_positions']}")
    print(f"   Concentration Ratio: {summary['concentration_ratio']:.2f}")

    # Calculate total risk
    total_risk = sum(p['size'] for p in summary['positions'].values())
    print(f"   Total Risk: ${total_risk:.2f}")

    print("\n[PASS] Real-world scenario test passed")
    return True


def main():
    print("\n" + "="*60)
    print("PRIORITY 4: CORRELATION MANAGEMENT TEST SUITE")
    print("="*60)

    all_passed = True

    tests = [
        test_basic_functionality,
        test_position_direction_limit,
        test_position_size_reduction,
        test_effective_positions,
        test_portfolio_summary,
        test_hedge_suggestion,
        test_real_world_scenario,
    ]

    for test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except AssertionError as e:
            print(f"\n[FAIL] {test_func.__name__}: {e}")
            all_passed = False
        except Exception as e:
            print(f"\n[ERROR] {test_func.__name__}: {e}")
            all_passed = False

    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*60)

    # Show correlation matrix
    print("\nDefault Correlation Matrix:")
    print("-"*40)
    for (s1, s2), corr in sorted(DEFAULT_CORRELATION_MATRIX.items()):
        print(f"   {s1} - {s2}: {corr:.2f}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
