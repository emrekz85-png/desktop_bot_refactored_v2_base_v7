#!/usr/bin/env python3
"""
Test script to demonstrate SSL Flow scoring system.

This script shows how to use the new scoring mode vs traditional AND logic.
"""

import pandas as pd
import numpy as np
from strategies import check_signal
from core import calculate_indicators

# Sample config for testing
CONFIG_AND_LOGIC = {
    "rr": 2.0,
    "rsi": 70,
    "at_active": True,
    "strategy_mode": "ssl_flow",
    "use_scoring": False,  # Traditional AND logic
}

CONFIG_SCORING = {
    "rr": 2.0,
    "rsi": 70,
    "at_active": True,
    "strategy_mode": "ssl_flow",
    "use_scoring": True,   # New scoring system
    "score_threshold": 6.0,  # 6/10 threshold
}

CONFIG_SCORING_RELAXED = {
    "rr": 2.0,
    "rsi": 70,
    "at_active": True,
    "strategy_mode": "ssl_flow",
    "use_scoring": True,   # New scoring system
    "score_threshold": 5.0,  # More relaxed 5/10 threshold
}


def test_scoring_system(df: pd.DataFrame, symbol: str, timeframe: str):
    """Test scoring system on sample data."""
    print(f"\n{'='*80}")
    print(f"Testing SSL Flow Scoring System: {symbol} {timeframe}")
    print(f"{'='*80}\n")

    # Calculate indicators
    df_with_indicators = calculate_indicators(df.copy())

    if df_with_indicators is None or df_with_indicators.empty:
        print("ERROR: Failed to calculate indicators")
        return

    print(f"Data points: {len(df_with_indicators)}")
    print(f"Indicators calculated successfully\n")

    # Test with AND logic
    print("1. TRADITIONAL AND LOGIC (use_scoring=False)")
    print("-" * 50)
    result_and = check_signal(df_with_indicators, CONFIG_AND_LOGIC, return_debug=True)
    s_type_and, entry_and, tp_and, sl_and, reason_and, debug_and = result_and

    print(f"Signal: {s_type_and or 'None'}")
    print(f"Reason: {reason_and}")
    if s_type_and:
        print(f"Entry: {entry_and:.2f}, TP: {tp_and:.2f}, SL: {sl_and:.2f}")
        rr = abs(tp_and - entry_and) / abs(entry_and - sl_and) if sl_and != entry_and else 0
        print(f"RR: {rr:.2f}")
    print()

    # Test with scoring (6.0 threshold)
    print("2. SCORING SYSTEM (use_scoring=True, threshold=6.0)")
    print("-" * 50)
    result_scoring = check_signal(df_with_indicators, CONFIG_SCORING, return_debug=True)
    s_type_score, entry_score, tp_score, sl_score, reason_score, debug_score = result_scoring

    print(f"Signal: {s_type_score or 'None'}")
    print(f"Reason: {reason_score}")
    if s_type_score:
        print(f"Entry: {entry_score:.2f}, TP: {tp_score:.2f}, SL: {sl_score:.2f}")
        rr = abs(tp_score - entry_score) / abs(entry_score - sl_score) if sl_score != entry_score else 0
        print(f"RR: {rr:.2f}")

    # Show score breakdown if available
    if debug_score.get("use_scoring"):
        print(f"\nLONG Score: {debug_score.get('long_score', 0):.2f} / 10.0")
        if debug_score.get('long_score_breakdown'):
            breakdown = debug_score['long_score_breakdown']
            print("  Breakdown:")
            for key, val in breakdown.items():
                print(f"    {key:20s}: {val:.2f}")

        print(f"\nSHORT Score: {debug_score.get('short_score', 0):.2f} / 10.0")
        if debug_score.get('short_score_breakdown'):
            breakdown = debug_score['short_score_breakdown']
            print("  Breakdown:")
            for key, val in breakdown.items():
                print(f"    {key:20s}: {val:.2f}")
    print()

    # Test with relaxed scoring (5.0 threshold)
    print("3. SCORING SYSTEM - RELAXED (use_scoring=True, threshold=5.0)")
    print("-" * 50)
    result_relaxed = check_signal(df_with_indicators, CONFIG_SCORING_RELAXED, return_debug=True)
    s_type_relax, entry_relax, tp_relax, sl_relax, reason_relax, debug_relax = result_relaxed

    print(f"Signal: {s_type_relax or 'None'}")
    print(f"Reason: {reason_relax}")
    if s_type_relax:
        print(f"Entry: {entry_relax:.2f}, TP: {tp_relax:.2f}, SL: {sl_relax:.2f}")
        rr = abs(tp_relax - entry_relax) / abs(entry_relax - sl_relax) if sl_relax != entry_relax else 0
        print(f"RR: {rr:.2f}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"AND Logic:              {s_type_and or 'No signal'}")
    print(f"Scoring (6.0):          {s_type_score or 'No signal'}")
    print(f"Scoring Relaxed (5.0):  {s_type_relax or 'No signal'}")
    print()

    if debug_score.get("use_scoring"):
        long_score = debug_score.get('long_score', 0)
        short_score = debug_score.get('short_score', 0)
        print(f"LONG Score:  {long_score:.2f} / 10.0  {'[PASS @ 6.0]' if long_score >= 6.0 else '[FAIL @ 6.0]'}  {'[PASS @ 5.0]' if long_score >= 5.0 else '[FAIL @ 5.0]'}")
        print(f"SHORT Score: {short_score:.2f} / 10.0  {'[PASS @ 6.0]' if short_score >= 6.0 else '[FAIL @ 6.0]'}  {'[PASS @ 5.0]' if short_score >= 5.0 else '[FAIL @ 5.0]'}")
    print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SSL FLOW SCORING SYSTEM - TEST SCRIPT")
    print("="*80)
    print("\nThis script demonstrates the new scoring system vs traditional AND logic.")
    print("Scoring allows more nuanced signal detection vs binary all-or-nothing filters.")
    print("\nUsage: python test_scoring_system.py")
    print("\nNote: You need real market data to test. This is a template script.")
    print("      Integrate with your data fetching system to run real tests.")
    print("\n" + "="*80 + "\n")

    # Example: Create dummy data (replace with real data fetch)
    # from core import TradingEngine
    # engine = TradingEngine()
    # df = engine.fetch_ohlcv("BTCUSDT", "15m", limit=1000)
    # if df is not None:
    #     test_scoring_system(df, "BTCUSDT", "15m")

    print("To run a real test:")
    print("1. Fetch data: df = TradingEngine().fetch_ohlcv('BTCUSDT', '15m', limit=1000)")
    print("2. Run test: test_scoring_system(df, 'BTCUSDT', '15m')")
    print()
