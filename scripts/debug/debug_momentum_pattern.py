#!/usr/bin/env python3
"""
Debug Momentum Pattern Detection

Check what's happening with each phase.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core import get_client, calculate_indicators, set_backtest_mode
from core.momentum_patterns import (
    detect_at_stairstepping,
    detect_sharp_selloff,
    detect_fakeout_rally,
    detect_ssl_sideways,
    detect_momentum_exhaustion_pattern
)
import pandas as pd
import numpy as np


def fetch_sample_data():
    """Fetch small sample for debugging."""
    set_backtest_mode(True)
    client = get_client()

    df = client.get_klines(symbol='BTCUSDT', interval='15m', limit=500)
    df = calculate_indicators(df, timeframe='15m')

    return df


def debug_phases(df, index):
    """Debug each phase individually."""
    print(f"\n{'='*70}")
    print(f"DEBUG: Bar #{index} - {df.index[index]}")
    print(f"{'='*70}")

    # Check columns
    print(f"\n📋 Available AlphaTrend columns:")
    at_cols = [c for c in df.columns if 'alpha' in c.lower() or 'at_' in c.lower()]
    print(f"   {at_cols}")

    if 'alphatrend' in df.columns:
        print(f"\n   ✓ alphatrend: {df.iloc[index]['alphatrend']:.2f}")
    else:
        print(f"\n   ❌ 'alphatrend' column NOT FOUND")

    if 'alphatrend_2' in df.columns:
        print(f"   ✓ alphatrend_2: {df.iloc[index]['alphatrend_2']:.2f}")
    else:
        print(f"   ❌ 'alphatrend_2' column NOT FOUND")

    # Check baseline
    if 'baseline' in df.columns:
        print(f"   ✓ baseline: {df.iloc[index]['baseline']:.2f}")
    else:
        print(f"   ❌ 'baseline' column NOT FOUND")

    # Test each phase
    print(f"\n{'─'*70}")
    print(f"PHASE TESTING:")
    print(f"{'─'*70}")

    # Phase 1: Stairstepping
    print(f"\n1️⃣  STAIRSTEPPING (SHORT):")
    is_stair, details = detect_at_stairstepping(df, index, lookback=12, signal_type="SHORT")
    print(f"   Result: {'✅ PASS' if is_stair else '❌ FAIL'}")
    print(f"   Details: {details}")

    # Phase 2: Sharp Selloff (NEW - PRICE-BASED)
    print(f"\n2️⃣  SHARP SELLOFF (SHORT) - NEW PRICE-BASED:")
    is_selloff, details = detect_sharp_selloff(df, index, lookback=5, signal_type="SHORT")
    print(f"   Result: {'✅ PASS' if is_selloff else '❌ FAIL'}")
    print(f"   Details: {details}")

    # Phase 3: Fakeout
    print(f"\n3️⃣  FAKEOUT (SHORT):")
    is_fakeout, details = detect_fakeout_rally(df, index, signal_type="SHORT")
    print(f"   Result: {'✅ PASS' if is_fakeout else '❌ FAIL'}")
    print(f"   Details: {details}")

    # Phase 4: SSL Sideways
    print(f"\n4️⃣  SSL SIDEWAYS:")
    is_sideways, details = detect_ssl_sideways(df, index, lookback=8)
    print(f"   Result: {'✅ PASS' if is_sideways else '❌ FAIL'}")
    print(f"   Details: {details}")

    # Full pattern
    print(f"\n{'─'*70}")
    print(f"FULL PATTERN:")
    print(f"{'─'*70}")
    pattern = detect_momentum_exhaustion_pattern(df, index, signal_type="SHORT")
    print(f"\n   Pattern Detected: {'✅ YES' if pattern['pattern_detected'] else '❌ NO'}")
    print(f"   Quality: {pattern['quality']}")
    print(f"   Confidence: {pattern['confidence']:.2%}")
    print(f"   Phases: {pattern['phases']}")


def scan_for_any_phase(df):
    """Scan data to find ANY phase passing."""
    print(f"\n{'='*70}")
    print(f"SCANNING FOR ANY PHASE MATCHES")
    print(f"{'='*70}")

    phase_counts = {
        'stairstepping': 0,
        'sharp_break': 0,
        'fakeout': 0,
        'ssl_sideways': 0
    }

    for i in range(100, len(df)):
        # Check each phase
        is_stair, _ = detect_at_stairstepping(df, i, signal_type="SHORT")
        if is_stair:
            phase_counts['stairstepping'] += 1

        is_selloff, _ = detect_sharp_selloff(df, i, signal_type="SHORT")
        if is_selloff:
            phase_counts['sharp_break'] += 1

        is_fakeout, _ = detect_fakeout_rally(df, i, signal_type="SHORT")
        if is_fakeout:
            phase_counts['fakeout'] += 1

        is_sideways, _ = detect_ssl_sideways(df, i)
        if is_sideways:
            phase_counts['ssl_sideways'] += 1

    print(f"\n📊 Phase Occurrence in {len(df)} bars:")
    print(f"   Stairstepping: {phase_counts['stairstepping']}")
    print(f"   Sharp Break: {phase_counts['sharp_break']}")
    print(f"   Fakeout: {phase_counts['fakeout']}")
    print(f"   SSL Sideways: {phase_counts['ssl_sideways']}")

    return phase_counts


def main():
    print("="*70)
    print("MOMENTUM PATTERN - DEBUG MODE")
    print("="*70)

    # Fetch data
    print("\n📥 Fetching sample data...")
    df = fetch_sample_data()
    print(f"   ✓ Fetched {len(df)} bars")

    # Debug specific bars
    test_indices = [
        len(df) - 2,   # Recent bar
        len(df) // 2,  # Middle
        200,           # Early
    ]

    for idx in test_indices:
        debug_phases(df, idx)

    # Scan for any matches
    scan_for_any_phase(df)

    print("\n" + "="*70)
    print("✅ DEBUG COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
