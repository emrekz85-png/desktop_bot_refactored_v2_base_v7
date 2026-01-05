#!/usr/bin/env python3
"""
Filter Pass Rate Analysis Per Timeframe

Diagnoses WHY the SSL Flow bot produces ZERO signals on 1h/4h/1d
timeframes while working on 15m.

Key hypothesis: Percentage-based thresholds don't scale across timeframes.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.indicators import calculate_indicators
from core.config import ALPHATREND_CONFIG, DEFAULT_STRATEGY_CONFIG
from core.binance_client import BinanceClient


def fetch_klines(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch historical klines from Binance using the project's BinanceClient."""
    client = BinanceClient()
    df = client.get_klines(symbol, timeframe, limit)
    return df


def analyze_filter_pass_rates(df: pd.DataFrame, timeframe: str) -> Dict[str, float]:
    """
    Analyze pass rates for each filter in SSL Flow strategy.

    Returns dict of {filter_name: pass_rate_percent}
    """
    # Get config values
    ssl_touch_tolerance = DEFAULT_STRATEGY_CONFIG.get('ssl_touch_tolerance', 0.003)
    ssl_body_tolerance = DEFAULT_STRATEGY_CONFIG.get('ssl_body_tolerance', 0.003)
    min_pbema_distance = DEFAULT_STRATEGY_CONFIG.get('min_pbema_distance', 0.004)
    lookback_candles = DEFAULT_STRATEGY_CONFIG.get('lookback_candles', 5)
    adx_min = DEFAULT_STRATEGY_CONFIG.get('adx_min', 15.0)
    regime_adx_threshold = DEFAULT_STRATEGY_CONFIG.get('regime_adx_threshold', 20.0)
    regime_lookback = 50
    flat_threshold = ALPHATREND_CONFIG.get('flat_threshold', 0.002)
    flat_lookback = ALPHATREND_CONFIG.get('flat_lookback', 5)
    OVERLAP_THRESHOLD = 0.005
    min_wick_ratio = 0.10

    # Calculate indicators
    df = calculate_indicators(df.copy(), timeframe=timeframe)

    n = len(df)
    start_idx = max(200, regime_lookback)  # Need warmup for indicators

    # Initialize counters
    results = {
        '01_adx_ok': 0,
        '02_regime_ok': 0,
        '03_at_buyers_dominant': 0,
        '04_at_sellers_dominant': 0,
        '05_at_NOT_flat': 0,
        '06_price_above_baseline': 0,
        '07_price_below_baseline': 0,
        '08_baseline_touch_long': 0,
        '09_baseline_touch_short': 0,
        '10_body_above_baseline': 0,
        '11_body_below_baseline': 0,
        '12_pbema_distance_long': 0,
        '13_pbema_distance_short': 0,
        '14_pbema_above_baseline': 0,
        '15_pbema_below_baseline': 0,
        '16_no_overlap': 0,
        '17_wick_rejection_long': 0,
        '18_wick_rejection_short': 0,
        'COMBO_long_all_pass': 0,
        'COMBO_short_all_pass': 0,
    }

    total_bars = 0

    # Pre-extract arrays
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    open_arr = df['open'].values
    baseline_arr = df['baseline'].values
    pb_top_arr = df['pb_ema_top'].values
    pb_bot_arr = df['pb_ema_bot'].values
    adx_arr = df['adx'].values
    at_buyers_dom = df['at_buyers_dominant'].values
    at_sellers_dom = df['at_sellers_dominant'].values
    at_is_flat = df['at_is_flat'].values

    for i in range(start_idx, n):
        total_bars += 1

        close = close_arr[i]
        high = high_arr[i]
        low = low_arr[i]
        open_ = open_arr[i]
        baseline = baseline_arr[i]
        pb_top = pb_top_arr[i]
        pb_bot = pb_bot_arr[i]
        adx_val = adx_arr[i]

        # Skip if NaN
        if pd.isna(baseline) or pd.isna(adx_val):
            total_bars -= 1
            continue

        # 1. ADX Filter
        adx_ok = adx_val >= adx_min
        if adx_ok:
            results['01_adx_ok'] += 1

        # 2. Regime Gating (ADX average over lookback)
        regime_start = max(0, i - regime_lookback)
        adx_window = adx_arr[regime_start:i]  # Exclude current bar (look-ahead fix)
        adx_avg = float(np.nanmean(adx_window)) if len(adx_window) > 0 else adx_val
        regime_ok = adx_avg >= regime_adx_threshold
        if regime_ok:
            results['02_regime_ok'] += 1

        # 3-4. AlphaTrend Dominance
        if at_buyers_dom[i]:
            results['03_at_buyers_dominant'] += 1
        if at_sellers_dom[i]:
            results['04_at_sellers_dominant'] += 1

        # 5. AlphaTrend NOT Flat (CRITICAL - this is our main hypothesis)
        if not at_is_flat[i]:
            results['05_at_NOT_flat'] += 1

        # 6-7. Price Position vs Baseline
        price_above = close > baseline
        price_below = close < baseline
        if price_above:
            results['06_price_above_baseline'] += 1
        if price_below:
            results['07_price_below_baseline'] += 1

        # 8-9. Baseline Touch Detection
        lookback_start = max(0, i - lookback_candles)

        # LONG touch: low touched baseline from above
        lookback_lows = low_arr[lookback_start:i+1]
        lookback_baselines = baseline_arr[lookback_start:i+1]
        baseline_touch_long = np.any(lookback_lows <= lookback_baselines * (1 + ssl_touch_tolerance))

        # SHORT touch: high touched baseline from below
        lookback_highs = high_arr[lookback_start:i+1]
        baseline_touch_short = np.any(lookback_highs >= lookback_baselines * (1 - ssl_touch_tolerance))

        if baseline_touch_long:
            results['08_baseline_touch_long'] += 1
        if baseline_touch_short:
            results['09_baseline_touch_short'] += 1

        # 10-11. Body Position
        body_min = min(open_, close)
        body_max = max(open_, close)
        body_above = body_min > baseline * (1 - ssl_body_tolerance)
        body_below = body_max < baseline * (1 + ssl_body_tolerance)

        if body_above:
            results['10_body_above_baseline'] += 1
        if body_below:
            results['11_body_below_baseline'] += 1

        # 12-13. PBEMA Distance
        long_pbema_dist = (pb_bot - close) / close if close > 0 else 0
        short_pbema_dist = (close - pb_top) / close if close > 0 else 0

        if long_pbema_dist >= min_pbema_distance:
            results['12_pbema_distance_long'] += 1
        if short_pbema_dist >= min_pbema_distance:
            results['13_pbema_distance_short'] += 1

        # 14-15. PBEMA vs Baseline Position
        pbema_mid = (pb_top + pb_bot) / 2
        if pbema_mid > baseline:
            results['14_pbema_above_baseline'] += 1
        if pbema_mid < baseline:
            results['15_pbema_below_baseline'] += 1

        # 16. Overlap Check
        baseline_pbema_dist = abs(baseline - pbema_mid) / pbema_mid if pbema_mid > 0 else 0
        no_overlap = baseline_pbema_dist >= OVERLAP_THRESHOLD
        if no_overlap:
            results['16_no_overlap'] += 1

        # 17-18. Wick Rejection
        candle_range = high - low
        if candle_range > 0:
            lower_wick = body_min - low
            upper_wick = high - body_max
            lower_wick_ratio = lower_wick / candle_range
            upper_wick_ratio = upper_wick / candle_range

            if lower_wick_ratio >= min_wick_ratio:
                results['17_wick_rejection_long'] += 1
            if upper_wick_ratio >= min_wick_ratio:
                results['18_wick_rejection_short'] += 1

        # COMBO: Check if ALL LONG conditions pass
        long_all_pass = (
            adx_ok and
            regime_ok and
            at_buyers_dom[i] and
            not at_is_flat[i] and
            price_above and
            baseline_touch_long and
            body_above and
            long_pbema_dist >= min_pbema_distance and
            pbema_mid > baseline and
            no_overlap and
            lower_wick_ratio >= min_wick_ratio if candle_range > 0 else False
        )

        # COMBO: Check if ALL SHORT conditions pass
        short_all_pass = (
            adx_ok and
            regime_ok and
            at_sellers_dom[i] and
            not at_is_flat[i] and
            price_below and
            baseline_touch_short and
            body_below and
            short_pbema_dist >= min_pbema_distance and
            pbema_mid < baseline and
            no_overlap and
            upper_wick_ratio >= min_wick_ratio if candle_range > 0 else False
        )

        if long_all_pass:
            results['COMBO_long_all_pass'] += 1
        if short_all_pass:
            results['COMBO_short_all_pass'] += 1

    # Convert to percentages
    if total_bars > 0:
        for key in results:
            results[key] = (results[key] / total_bars) * 100

    results['_total_bars'] = total_bars

    return results


def analyze_at_flat_threshold_scaling(df: pd.DataFrame, timeframe: str) -> Dict:
    """
    Specifically analyze the at_is_flat threshold behavior.

    On HTF, price moves slower in percentage terms, so the same 0.002 threshold
    that works on 15m may be too restrictive on 4h.
    """
    df = calculate_indicators(df.copy(), timeframe=timeframe)

    flat_lookback = ALPHATREND_CONFIG.get('flat_lookback', 5)

    # Calculate the alphatrend percentage change
    df['at_pct_change'] = df['alphatrend'].pct_change(flat_lookback).abs()

    # Get statistics
    at_changes = df['at_pct_change'].dropna()

    if len(at_changes) == 0:
        return {}

    stats = {
        'mean_pct_change': float(at_changes.mean()) * 100,
        'median_pct_change': float(at_changes.median()) * 100,
        'p25_pct_change': float(at_changes.quantile(0.25)) * 100,
        'p75_pct_change': float(at_changes.quantile(0.75)) * 100,
        'p90_pct_change': float(at_changes.quantile(0.90)) * 100,
        'current_threshold': ALPHATREND_CONFIG.get('flat_threshold', 0.002) * 100,
    }

    # Calculate what % of bars would pass at various thresholds
    thresholds = [0.001, 0.002, 0.003, 0.004, 0.005, 0.010, 0.015, 0.020]
    for thresh in thresholds:
        pass_rate = (at_changes >= thresh).sum() / len(at_changes) * 100
        stats[f'pass_rate_at_{thresh:.3f}'] = pass_rate

    return stats


def main():
    """Run filter analysis across timeframes."""
    print("=" * 80)
    print("SSL FLOW FILTER PASS RATE ANALYSIS BY TIMEFRAME")
    print("=" * 80)
    print()

    symbol = "BTCUSDT"
    timeframes = ["15m", "1h", "4h", "1d"]

    all_results = {}
    at_flat_analysis = {}

    for tf in timeframes:
        print(f"\nFetching {symbol} {tf} data...")
        try:
            df = fetch_klines(symbol, tf, limit=1000)
            print(f"  Got {len(df)} candles")

            # Filter pass rates
            results = analyze_filter_pass_rates(df, tf)
            all_results[tf] = results

            # AT flat threshold analysis
            at_stats = analyze_at_flat_threshold_scaling(df, tf)
            at_flat_analysis[tf] = at_stats

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Print summary table
    print("\n" + "=" * 120)
    print("FILTER PASS RATE COMPARISON TABLE")
    print("=" * 120)

    # Get all filter names
    if all_results:
        filter_names = [k for k in list(all_results.values())[0].keys() if not k.startswith('_')]

        # Header
        header = f"{'Filter':<35}"
        for tf in timeframes:
            header += f" | {tf:>8}"
        header += " | Bottleneck?"
        print(header)
        print("-" * 120)

        # Rows
        for filter_name in filter_names:
            row = f"{filter_name:<35}"
            rates = []
            for tf in timeframes:
                if tf in all_results:
                    rate = all_results[tf].get(filter_name, 0)
                    rates.append(rate)
                    row += f" | {rate:>7.1f}%"
                else:
                    row += f" | {'N/A':>8}"

            # Detect bottleneck (15m pass rate > 2x HTF pass rate)
            bottleneck = ""
            if len(rates) >= 2:
                if rates[0] > 0 and rates[-1] < rates[0] * 0.3:  # 4h/1d < 30% of 15m
                    bottleneck = "YES - HTF FAILS"
                elif rates[-1] < 10:
                    bottleneck = "CRITICAL (<10%)"

            row += f" | {bottleneck}"
            print(row)

    # Print AT flat threshold analysis
    print("\n" + "=" * 80)
    print("ALPHATREND FLAT THRESHOLD ANALYSIS")
    print("(Key hypothesis: 0.2% threshold too tight for HTF)")
    print("=" * 80)

    for tf in timeframes:
        if tf in at_flat_analysis:
            stats = at_flat_analysis[tf]
            print(f"\n{tf} Timeframe:")
            print(f"  AlphaTrend pct_change stats over {ALPHATREND_CONFIG.get('flat_lookback', 5)} bars:")
            print(f"    Mean:   {stats.get('mean_pct_change', 0):.4f}%")
            print(f"    Median: {stats.get('median_pct_change', 0):.4f}%")
            print(f"    P25:    {stats.get('p25_pct_change', 0):.4f}%")
            print(f"    P75:    {stats.get('p75_pct_change', 0):.4f}%")
            print(f"    P90:    {stats.get('p90_pct_change', 0):.4f}%")
            print(f"  Current threshold: {stats.get('current_threshold', 0):.4f}%")
            print(f"  Pass rates at various thresholds:")
            for thresh in [0.001, 0.002, 0.003, 0.005, 0.010, 0.015, 0.020]:
                key = f'pass_rate_at_{thresh:.3f}'
                if key in stats:
                    marker = " <-- CURRENT" if abs(thresh - ALPHATREND_CONFIG.get('flat_threshold', 0.002)) < 0.0001 else ""
                    print(f"    {thresh*100:.2f}%: {stats[key]:.1f}% pass{marker}")

    # Print config values for reference
    print("\n" + "=" * 80)
    print("CURRENT THRESHOLD VALUES (from config)")
    print("=" * 80)
    print(f"  flat_threshold:       {ALPHATREND_CONFIG.get('flat_threshold', 0.002)*100:.2f}% (at_is_flat)")
    print(f"  ssl_touch_tolerance:  {DEFAULT_STRATEGY_CONFIG.get('ssl_touch_tolerance', 0.003)*100:.2f}% (baseline touch)")
    print(f"  ssl_body_tolerance:   {DEFAULT_STRATEGY_CONFIG.get('ssl_body_tolerance', 0.003)*100:.2f}% (body position)")
    print(f"  min_pbema_distance:   {DEFAULT_STRATEGY_CONFIG.get('min_pbema_distance', 0.004)*100:.2f}% (TP room)")
    print(f"  OVERLAP_THRESHOLD:    0.50% (SSL-PBEMA overlap)")
    print(f"  min_wick_ratio:       10.00% (wick rejection)")
    print(f"  adx_min:              {DEFAULT_STRATEGY_CONFIG.get('adx_min', 15.0)} (trend strength)")
    print(f"  regime_adx_threshold: {DEFAULT_STRATEGY_CONFIG.get('regime_adx_threshold', 20.0)} (regime gating)")

    # Recommendations
    print("\n" + "=" * 80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)

    if all_results:
        # Find biggest bottlenecks
        bottlenecks = []
        for filter_name in filter_names:
            if filter_name.startswith('COMBO'):
                continue
            rates = [all_results[tf].get(filter_name, 0) for tf in timeframes if tf in all_results]
            if len(rates) >= 2 and rates[0] > 0:
                decline_ratio = rates[-1] / rates[0] if rates[0] > 0 else 0
                if decline_ratio < 0.5:  # 50%+ decline from 15m to 4h/1d
                    bottlenecks.append((filter_name, rates[0], rates[-1], decline_ratio))

        if bottlenecks:
            bottlenecks.sort(key=lambda x: x[3])  # Sort by worst decline
            print("\nTOP BOTTLENECK FILTERS (15m->4h/1d decline):")
            for i, (name, rate_15m, rate_htf, ratio) in enumerate(bottlenecks[:5], 1):
                print(f"  {i}. {name}")
                print(f"     15m: {rate_15m:.1f}% -> 4h/1d: {rate_htf:.1f}% ({ratio*100:.0f}% of 15m)")

        # Check COMBO rates
        print("\n\nFINAL SIGNAL RATES (all filters combined):")
        for tf in timeframes:
            if tf in all_results:
                long_rate = all_results[tf].get('COMBO_long_all_pass', 0)
                short_rate = all_results[tf].get('COMBO_short_all_pass', 0)
                print(f"  {tf}: LONG={long_rate:.2f}%, SHORT={short_rate:.2f}%")


if __name__ == "__main__":
    main()
