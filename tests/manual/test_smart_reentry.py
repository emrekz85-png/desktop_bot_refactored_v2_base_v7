#!/usr/bin/env python3
"""
Test script for Smart Re-Entry system (v1.9.0)

This script tests the smart re-entry functionality to ensure:
1. SL trades are properly stored
2. Re-entry conditions are correctly evaluated
3. Cooldown bypass works when conditions are met
4. Statistics are properly tracked
"""

from datetime import datetime, timedelta
from core.trade_manager import SimTradeManager
from core.utils import utcnow

def test_smart_reentry():
    """Test smart re-entry system end-to-end."""

    print("=" * 80)
    print("SMART RE-ENTRY SYSTEM TEST")
    print("=" * 80)

    # Create trade manager
    tm = SimTradeManager(initial_balance=2000.0)

    # Test 1: Store SL trade
    print("\n[Test 1] Storing SL trade details...")

    # Simulate a trade that will be stopped out
    trade_data = {
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "type": "LONG",
        "entry": 100000.0,
        "tp": 101000.0,
        "sl": 99500.0,
        "setup": "SSL_Flow",
        "open_time_utc": utcnow(),
        "timestamp": utcnow().strftime("%Y-%m-%d %H:%M"),
        "indicators_at_entry": {
            "at_buyers_dominant": True,
            "at_sellers_dominant": False,
        },
    }

    # Open the trade
    opened = tm.open_trade(trade_data)
    print(f"  Trade opened: {opened}")

    if not opened:
        print("  ERROR: Failed to open trade")
        return False

    # Simulate SL being hit (price drops to 99400)
    print("\n[Test 2] Simulating SL hit...")
    closed_trades = tm.update_trades(
        symbol="BTCUSDT",
        tf="15m",
        candle_high=99800.0,
        candle_low=99400.0,
        candle_close=99600.0,
        candle_time_utc=utcnow(),
    )

    if closed_trades:
        print(f"  Trade closed: {closed_trades[0].get('status')}")
        print(f"  PnL: ${closed_trades[0].get('pnl', 0):.2f}")
    else:
        print("  ERROR: Trade was not closed by SL")
        return False

    # Check that SL trade was stored
    key = ("BTCUSDT", "15m")
    if key in tm.last_sl_trades:
        print(f"  SL trade stored successfully")
        print(f"    Entry: {tm.last_sl_trades[key]['entry']}")
        print(f"    Side: {tm.last_sl_trades[key]['side']}")
    else:
        print("  ERROR: SL trade was not stored")
        return False

    # Test 3: Check re-entry conditions
    print("\n[Test 3] Testing re-entry conditions...")

    # Scenario A: Price recovers to near entry (within 0.3%)
    current_time = utcnow() + timedelta(hours=1)
    current_price = 100100.0  # Within 0.3% of entry (100000)

    can_reenter = tm.check_quick_reentry(
        symbol="BTCUSDT",
        timeframe="15m",
        current_price=current_price,
        current_time=current_time,
        at_dominant="BUYERS",
    )

    print(f"  Scenario A (price near entry, AlphaTrend confirms):")
    print(f"    Current price: {current_price} (entry was {100000.0})")
    print(f"    Time since SL: 1 hour")
    print(f"    AlphaTrend: BUYERS (matches original LONG)")
    print(f"    Can re-enter: {can_reenter}")

    if not can_reenter:
        print("  ERROR: Re-entry should be allowed")
        return False

    # Test 4: Re-entry conditions NOT met (different scenarios)
    print("\n[Test 4] Testing rejection scenarios...")

    # Scenario B: Price too far from entry
    # NOTE: v46.x uses 1.0% threshold (wider than original 0.3%)
    # Entry was 100050 (after slippage), so >1% would be >1000.5 distance
    can_reenter_b = tm.check_quick_reentry(
        symbol="BTCUSDT",
        timeframe="15m",
        current_price=101500.0,  # ~1.45% away from entry (> 1.0% threshold)
        current_time=current_time,
        at_dominant="BUYERS",
    )
    print(f"  Scenario B (price too far): {can_reenter_b}")
    if can_reenter_b:
        print("  ERROR: Re-entry should be rejected (price too far)")
        return False

    # Scenario C: AlphaTrend changed direction
    can_reenter_c = tm.check_quick_reentry(
        symbol="BTCUSDT",
        timeframe="15m",
        current_price=100100.0,
        current_time=current_time,
        at_dominant="SELLERS",  # Changed direction!
    )
    print(f"  Scenario C (AlphaTrend changed): {can_reenter_c}")
    if can_reenter_c:
        print("  ERROR: Re-entry should be rejected (AlphaTrend changed)")
        return False

    # Need to re-add the SL trade since Scenario C deleted it
    tm.last_sl_trades[key] = {
        'symbol': "BTCUSDT",
        'timeframe': "15m",
        'side': 'LONG',
        'entry': 100000.0,
        'sl': 99500.0,
        'tp': 101000.0,
        'exit_time': utcnow(),
        'reentry_used': False,
    }

    # Scenario D: Too much time passed (>4 hours)
    # NOTE: v46.x uses 4h window (wider than original 2h)
    old_time = utcnow() + timedelta(hours=5)
    can_reenter_d = tm.check_quick_reentry(
        symbol="BTCUSDT",
        timeframe="15m",
        current_price=100100.0,
        current_time=old_time,
        at_dominant="BUYERS",
    )
    print(f"  Scenario D (>4 hours passed): {can_reenter_d}")
    if can_reenter_d:
        print("  ERROR: Re-entry should be rejected (expired)")
        return False

    # Test 5: Mark re-entry as used
    print("\n[Test 5] Testing re-entry usage tracking...")

    # Re-add the SL trade for final test
    tm.last_sl_trades[key] = {
        'symbol': "BTCUSDT",
        'timeframe': "15m",
        'side': 'LONG',
        'entry': 100000.0,
        'sl': 99500.0,
        'tp': 101000.0,
        'exit_time': utcnow(),
        'reentry_used': False,
    }

    # Mark as used
    tm.mark_reentry_used("BTCUSDT", "15m")

    # Try to re-enter again (should fail)
    can_reenter_again = tm.check_quick_reentry(
        symbol="BTCUSDT",
        timeframe="15m",
        current_price=100100.0,
        current_time=utcnow() + timedelta(minutes=30),
        at_dominant="BUYERS",
    )
    print(f"  Second re-entry attempt (should fail): {can_reenter_again}")
    if can_reenter_again:
        print("  ERROR: Second re-entry should be rejected (already used)")
        return False

    # Test 6: Statistics
    print("\n[Test 6] Testing statistics tracking...")
    stats = tm.get_reentry_stats()
    print(f"  Re-entry attempts: {stats['reentry_attempts']}")
    print(f"  Pending re-entries: {stats['pending_reentry_count']}")

    if stats['reentry_attempts'] != 1:
        print("  ERROR: Should have 1 re-entry attempt")
        return False

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_smart_reentry()
    exit(0 if success else 1)
