#!/usr/bin/env python3
"""
Test Momentum Exhaustion Pattern Detection

Tests pattern detector on real BTCUSDT data to validate thresholds
derived from visual analysis (NO1, NO2, NO5).
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from core import get_client, calculate_indicators, set_backtest_mode
from core.momentum_patterns import detect_momentum_exhaustion_pattern


def fetch_data(symbol: str, timeframe: str, days: int = 365):
    """Fetch and prepare data."""
    import requests

    set_backtest_mode(True)
    client = get_client()

    candles_map = {"5m": 288, "15m": 96, "1h": 24, "2h": 12, "4h": 6}
    candles = min(days * candles_map.get(timeframe, 96), 35000)

    all_dfs = []
    remaining = candles
    end_time = None

    print(f"   Fetching {candles} candles...")

    while remaining > 0:
        chunk = min(remaining, 1000)

        if end_time:
            url = f"{client.BASE_URL}/klines"
            params = {'symbol': symbol, 'interval': timeframe, 'limit': chunk, 'endTime': end_time}
            res = requests.get(url, params=params, timeout=30)
            if res.status_code != 200 or not res.json():
                break
            df_c = pd.DataFrame(res.json()).iloc[:, :6]
            df_c.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_c['timestamp'] = pd.to_datetime(df_c['timestamp'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
        else:
            df_c = client.get_klines(symbol=symbol, interval=timeframe, limit=chunk)

        if df_c.empty:
            break

        all_dfs.insert(0, df_c)
        remaining -= len(df_c)

        if 'timestamp' in df_c.columns:
            end_time = int(df_c['timestamp'].iloc[0].timestamp() * 1000) - 1
        else:
            break

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    df = calculate_indicators(df, timeframe=timeframe)

    print(f"   ✓ Fetched {len(df)} candles")
    return df


def test_pattern_detection(df: pd.DataFrame, timeframe: str):
    """
    Test momentum pattern detection on data.

    Returns:
        dict: Detection results
    """
    print(f"\n{'='*70}")
    print(f"Testing Pattern Detection: {timeframe}")
    print(f"{'='*70}")

    results = {
        'timeframe': timeframe,
        'total_bars': len(df),
        'patterns_found': [],
        'stats': {
            'total_detected': 0,
            'excellent_quality': 0,
            'good_quality': 0,
            'moderate_quality': 0,
            'avg_confidence': 0.0
        }
    }

    # Check each bar (skip first 100 for indicator warmup)
    for i in range(100, len(df)):
        # Test SHORT patterns
        pattern_short = detect_momentum_exhaustion_pattern(df, i, signal_type="SHORT", require_all_phases=False)

        if pattern_short['pattern_detected']:
            pattern_short['index'] = i
            pattern_short['timestamp'] = df.index[i]
            pattern_short['direction'] = 'SHORT'
            results['patterns_found'].append(pattern_short)

        # Test LONG patterns
        pattern_long = detect_momentum_exhaustion_pattern(df, i, signal_type="LONG", require_all_phases=False)

        if pattern_long['pattern_detected']:
            pattern_long['index'] = i
            pattern_long['timestamp'] = df.index[i]
            pattern_long['direction'] = 'LONG'
            results['patterns_found'].append(pattern_long)

    # Calculate statistics
    total = len(results['patterns_found'])
    results['stats']['total_detected'] = total

    if total > 0:
        results['stats']['excellent_quality'] = sum(1 for p in results['patterns_found'] if p['quality'] == 'EXCELLENT')
        results['stats']['good_quality'] = sum(1 for p in results['patterns_found'] if p['quality'] == 'GOOD')
        results['stats']['moderate_quality'] = sum(1 for p in results['patterns_found'] if p['quality'] == 'MODERATE')
        results['stats']['avg_confidence'] = np.mean([p['confidence'] for p in results['patterns_found']])

    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"   Total bars analyzed: {results['total_bars']:,}")
    print(f"   Patterns detected: {total}")
    print(f"   - EXCELLENT quality: {results['stats']['excellent_quality']}")
    print(f"   - GOOD quality: {results['stats']['good_quality']}")
    print(f"   - MODERATE quality: {results['stats']['moderate_quality']}")
    print(f"   Average confidence: {results['stats']['avg_confidence']:.2%}")

    # Show top 5 patterns
    if total > 0:
        print(f"\n🎯 TOP 5 PATTERNS (Highest Confidence):")
        sorted_patterns = sorted(results['patterns_found'], key=lambda x: x['confidence'], reverse=True)[:5]

        for idx, p in enumerate(sorted_patterns, 1):
            phases_str = ', '.join([k for k, v in p['phases'].items() if v])
            print(f"\n   {idx}. {p['timestamp'].strftime('%Y-%m-%d %H:%M')} ({p['direction']})")
            print(f"      Quality: {p['quality']}, Confidence: {p['confidence']:.2%}")
            print(f"      Phases: {phases_str}")

            # Show details
            if p['phases']['sharp_break']:
                gap = p['details']['sharp_break'].get('gap_pct', 0)
                print(f"      Sharp break gap: {gap:.2%}")

            if p['phases']['stairstepping']:
                consistency = p['details']['stairstepping'].get('consistency', 0)
                print(f"      Stairstepping consistency: {consistency:.1%}")

    return results


def compare_timeframes(results_list):
    """Compare pattern detection across timeframes."""
    print(f"\n{'='*70}")
    print(f"TIMEFRAME COMPARISON")
    print(f"{'='*70}")

    print(f"\n{'Timeframe':<12} {'Patterns':<12} {'Excellent':<12} {'Good':<12} {'Avg Conf':<12}")
    print(f"{'-'*70}")

    for res in results_list:
        tf = res['timeframe']
        total = res['stats']['total_detected']
        excellent = res['stats']['excellent_quality']
        good = res['stats']['good_quality']
        avg_conf = res['stats']['avg_confidence']

        print(f"{tf:<12} {total:<12} {excellent:<12} {good:<12} {avg_conf:<12.1%}")

    # Annual rate calculation
    print(f"\n📈 ANNUAL PROJECTION:")
    for res in results_list:
        tf = res['timeframe']
        total = res['stats']['total_detected']
        bars = res['total_bars']

        # Approximate bars per year
        bars_per_year = {'5m': 105120, '15m': 35040, '1h': 8760, '2h': 4380, '4h': 2190}

        if tf in bars_per_year and bars > 0:
            annual_rate = (total / bars) * bars_per_year[tf]
            print(f"   {tf}: ~{annual_rate:.0f} patterns per year")


def main():
    """Main test routine."""
    print("="*70)
    print("MOMENTUM EXHAUSTION PATTERN - VALIDATION TEST")
    print("="*70)
    print("\nTesting pattern detector on real BTCUSDT data...")
    print("Thresholds derived from visual analysis of NO1, NO2, NO5 trades.\n")

    # Test on multiple timeframes (matching real trades)
    timeframes = ['15m', '1h', '2h']  # NO5=1h, NO1/NO2=2h

    all_results = []

    for tf in timeframes:
        print(f"\n📥 Fetching {tf} data...")
        try:
            df = fetch_data('BTCUSDT', tf, days=365)
            results = test_pattern_detection(df, tf)
            all_results.append(results)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    # Compare results
    if all_results:
        compare_timeframes(all_results)

    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)

    # Save results
    output_file = Path("data/results/momentum_pattern_test_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    import json

    # Convert timestamps to strings for JSON serialization
    for res in all_results:
        for p in res['patterns_found']:
            p['timestamp'] = p['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
