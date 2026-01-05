#!/usr/bin/env python3
# run_filter_discovery.py
# Filter Combination Discovery Runner
#
# Usage:
#   python run_filter_discovery.py --pilot                  # BTCUSDT-15m only (recommended first run)
#   python run_filter_discovery.py --full                   # All symbols/timeframes
#   python run_filter_discovery.py --validate combo_id      # Validate specific combo on holdout
#
# Modes:
#   --pilot: Fast pilot test on BTCUSDT-15m (128 combos × 3min ≈ 6-8 hours)
#   --full: Full test on all symbols and timeframes (slower)
#   --validate: Validate a specific filter combination on holdout data

import os
import sys
import json
import time
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    SYMBOLS, TIMEFRAMES, DATA_DIR, VERSION,
    TradingEngine, calculate_indicators,
)
from core.filter_discovery import FilterDiscoveryEngine, FilterCombination


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def fetch_and_prepare_data(symbol: str, timeframe: str, candles: int = 30000) -> pd.DataFrame:
    """Fetch data and calculate indicators for a symbol/timeframe.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "15m")
        candles: Number of candles to fetch

    Returns:
        DataFrame with OHLCV + indicators
    """
    print(f"[DATA] Fetching {candles} candles for {symbol}-{timeframe}...")

    # Use static method for pagination (handles large fetches)
    df = TradingEngine.get_historical_data_pagination(symbol, timeframe, total_candles=candles)

    if df is None or df.empty:
        raise ValueError(f"Failed to fetch data for {symbol}-{timeframe}")

    print(f"[DATA] Fetched {len(df)} candles, calculating indicators...")
    df = calculate_indicators(df, timeframe=timeframe)

    print(f"[DATA] Ready: {len(df)} candles with indicators")
    return df


def save_results(
    results: List,
    symbol: str,
    timeframe: str,
    run_id: str,
    output_dir: str
):
    """Save discovery results to files.

    Saves:
    - results.json: All combinations with full metrics
    - top_10.txt: Human-readable top 10 report
    - filter_pass_rates.json: Pass rate per filter (diagnostic)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save full results JSON
    results_file = os.path.join(output_dir, "results.json")
    results_data = {
        'run_id': run_id,
        'symbol': symbol,
        'timeframe': timeframe,
        'version': VERSION,
        'timestamp': datetime.now().isoformat(),
        'total_combinations': len(results),
        'results': [r.to_dict() for r in results],
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[SAVE] Full results saved to: {results_file}")

    # 2. Save top 10 human-readable report
    top_10_file = os.path.join(output_dir, "top_10.txt")
    with open(top_10_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"FILTER DISCOVERY TOP 10 RESULTS\n")
        f.write(f"Symbol: {symbol}, Timeframe: {timeframe}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        for i, result in enumerate(results[:10], 1):
            f.write(f"{'='*80}\n")
            f.write(f"#{i} - {result.combination.to_string()}\n")
            f.write(f"{'='*80}\n")
            f.write(f"TRAINING (60% of data):\n")
            f.write(f"  PnL: ${result.train_pnl:.2f}\n")
            f.write(f"  Trades: {result.train_trades}\n")
            f.write(f"  E[R]: {result.train_expected_r:.3f}\n")
            f.write(f"  Win Rate: {result.train_win_rate:.1%}\n")
            f.write(f"  Score: {result.train_score:.2f}\n")
            f.write(f"\n")
            f.write(f"WALK-FORWARD (20% of data):\n")
            f.write(f"  PnL: ${result.wf_pnl:.2f}\n")
            f.write(f"  Trades: {result.wf_trades}\n")
            f.write(f"  E[R]: {result.wf_expected_r:.3f}\n")
            f.write(f"  Win Rate: {result.wf_win_rate:.1%}\n")
            f.write(f"\n")
            f.write(f"OVERFIT CHECK:\n")
            f.write(f"  Overfit Ratio: {result.overfit_ratio:.2f} (WF E[R] / Train E[R])\n")
            f.write(f"  Is Overfit: {'YES ❌' if result.is_overfit else 'NO ✓'}\n")
            f.write(f"\n")

            # Show which filters are enabled
            combo = result.combination
            f.write(f"ENABLED FILTERS:\n")
            f.write(f"  ADX Filter: {'✓' if combo.adx_filter else '✗'}\n")
            f.write(f"  Regime Gating: {'✓' if combo.regime_gating else '✗'}\n")
            f.write(f"  Baseline Touch: {'✓' if combo.baseline_touch else '✗'}\n")
            f.write(f"  PBEMA Distance: {'✓' if combo.pbema_distance else '✗'}\n")
            f.write(f"  Body Position: {'✓' if combo.body_position else '✗'}\n")
            f.write(f"  SSL-PBEMA Overlap: {'✓' if combo.ssl_pbema_overlap else '✗'}\n")
            f.write(f"  Wick Rejection: {'✓' if combo.wick_rejection else '✗'}\n")
            f.write(f"\n\n")

    print(f"[SAVE] Top 10 report saved to: {top_10_file}")

    # 3. Calculate and save filter pass rates (diagnostic)
    filter_stats = {
        'adx_filter': {'enabled': 0, 'total_score': 0.0},
        'regime_gating': {'enabled': 0, 'total_score': 0.0},
        'baseline_touch': {'enabled': 0, 'total_score': 0.0},
        'pbema_distance': {'enabled': 0, 'total_score': 0.0},
        'body_position': {'enabled': 0, 'total_score': 0.0},
        'ssl_pbema_overlap': {'enabled': 0, 'total_score': 0.0},
        'wick_rejection': {'enabled': 0, 'total_score': 0.0},
    }

    for result in results:
        combo = result.combination
        score = result.train_score if result.train_score > 0 else 0

        if combo.adx_filter:
            filter_stats['adx_filter']['enabled'] += 1
            filter_stats['adx_filter']['total_score'] += score
        if combo.regime_gating:
            filter_stats['regime_gating']['enabled'] += 1
            filter_stats['regime_gating']['total_score'] += score
        if combo.baseline_touch:
            filter_stats['baseline_touch']['enabled'] += 1
            filter_stats['baseline_touch']['total_score'] += score
        if combo.pbema_distance:
            filter_stats['pbema_distance']['enabled'] += 1
            filter_stats['pbema_distance']['total_score'] += score
        if combo.body_position:
            filter_stats['body_position']['enabled'] += 1
            filter_stats['body_position']['total_score'] += score
        if combo.ssl_pbema_overlap:
            filter_stats['ssl_pbema_overlap']['enabled'] += 1
            filter_stats['ssl_pbema_overlap']['total_score'] += score
        if combo.wick_rejection:
            filter_stats['wick_rejection']['enabled'] += 1
            filter_stats['wick_rejection']['total_score'] += score

    # Calculate pass rates and average scores
    total_combos = len(results)
    for filter_name, stats in filter_stats.items():
        stats['pass_rate'] = stats['enabled'] / total_combos if total_combos > 0 else 0
        stats['avg_score'] = (
            stats['total_score'] / stats['enabled']
            if stats['enabled'] > 0 else 0
        )

    filter_stats_file = os.path.join(output_dir, "filter_pass_rates.json")
    with open(filter_stats_file, 'w') as f:
        json.dump(filter_stats, f, indent=2)

    print(f"[SAVE] Filter pass rates saved to: {filter_stats_file}")


def print_summary(results: List, symbol: str, timeframe: str):
    """Print summary of discovery results to console."""
    print(f"\n{'='*80}")
    print(f"FILTER DISCOVERY SUMMARY: {symbol}-{timeframe}")
    print(f"{'='*80}")
    print(f"Total combinations tested: {len(results)}")
    print(f"Non-overfitted: {sum(1 for r in results if not r.is_overfit)}")
    print()

    print("TOP 5 FILTER COMBINATIONS:")
    print(f"{'='*80}")

    for i, result in enumerate(results[:5], 1):
        combo_str = result.combination.to_string()
        overfit_str = "❌ OVERFIT" if result.is_overfit else "✓ Valid"

        print(f"\n#{i} - {combo_str}")
        print(f"  Train: ${result.train_pnl:.2f}, {result.train_trades} trades, E[R]={result.train_expected_r:.3f}, Score={result.train_score:.2f}")
        print(f"  WF:    ${result.wf_pnl:.2f}, {result.wf_trades} trades, E[R]={result.wf_expected_r:.3f}, Ratio={result.overfit_ratio:.2f}")
        print(f"  {overfit_str}")

    print(f"\n{'='*80}\n")


# ==========================================
# MAIN DISCOVERY FUNCTIONS
# ==========================================

def run_pilot_discovery(
    candles: int = 30000,
    parallel: bool = True,
    max_workers: int = None
) -> str:
    """Run pilot discovery on BTCUSDT-15m only.

    Fast test to find promising filter combinations before running full test.

    Args:
        candles: Number of candles to fetch
        parallel: Use parallel processing
        max_workers: Max parallel workers

    Returns:
        Output directory path
    """
    symbol = "BTCUSDT"
    timeframe = "15m"

    print(f"\n{'='*80}")
    print(f"FILTER DISCOVERY - PILOT MODE")
    print(f"{'='*80}")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles: {candles}")
    print(f"Parallel: {parallel}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Fetch and prepare data
    df = fetch_and_prepare_data(symbol, timeframe, candles)

    # Create discovery engine
    engine = FilterDiscoveryEngine(
        symbol=symbol,
        timeframe=timeframe,
        data=df,
        baseline_trades=9,  # Current baseline is 9 trades/year
    )

    # Run discovery
    results = engine.run_discovery(parallel=parallel, max_workers=max_workers)

    elapsed = time.time() - start_time

    # Print summary
    print_summary(results, symbol, timeframe)

    # Save results
    run_id = f"pilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(DATA_DIR, "filter_discovery_runs", run_id)
    save_results(results, symbol, timeframe, run_id, output_dir)

    print(f"\n[COMPLETE] Pilot discovery finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[COMPLETE] Results saved to: {output_dir}")

    return output_dir


def run_full_discovery(
    symbols: List[str] = None,
    timeframes: List[str] = None,
    candles: int = 30000,
    parallel: bool = True,
    max_workers: int = None
) -> Dict[str, str]:
    """Run full discovery on multiple symbols and timeframes.

    Args:
        symbols: List of symbols to test (default: all SYMBOLS)
        timeframes: List of timeframes to test (default: all TIMEFRAMES)
        candles: Number of candles to fetch
        parallel: Use parallel processing
        max_workers: Max parallel workers

    Returns:
        Dict mapping (symbol, timeframe) to output directory path
    """
    if symbols is None:
        symbols = SYMBOLS
    if timeframes is None:
        timeframes = TIMEFRAMES

    print(f"\n{'='*80}")
    print(f"FILTER DISCOVERY - FULL MODE")
    print(f"{'='*80}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Candles: {candles}")
    print(f"Parallel: {parallel}")
    print(f"{'='*80}\n")

    start_time = time.time()
    output_dirs = {}

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n{'='*80}")
            print(f"Processing: {symbol}-{timeframe}")
            print(f"{'='*80}\n")

            try:
                # Fetch and prepare data
                df = fetch_and_prepare_data(symbol, timeframe, candles)

                # Create discovery engine
                engine = FilterDiscoveryEngine(
                    symbol=symbol,
                    timeframe=timeframe,
                    data=df,
                    baseline_trades=9,
                )

                # Run discovery
                results = engine.run_discovery(parallel=parallel, max_workers=max_workers)

                # Print summary
                print_summary(results, symbol, timeframe)

                # Save results
                run_id = f"full_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                output_dir = os.path.join(DATA_DIR, "filter_discovery_runs", run_id)
                save_results(results, symbol, timeframe, run_id, output_dir)

                output_dirs[(symbol, timeframe)] = output_dir

            except Exception as exc:
                print(f"\n[ERROR] Failed to process {symbol}-{timeframe}: {exc}")
                import traceback
                traceback.print_exc()
                continue

    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"FULL DISCOVERY COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Processed: {len(output_dirs)} symbol/timeframe pairs")
    print(f"{'='*80}\n")

    return output_dirs


def validate_combination_on_holdout(
    combo_id: str,
    results_file: str,
    candles: int = 30000
):
    """Validate a specific filter combination on holdout data.

    Args:
        combo_id: Index of combination in results (e.g., "0" for top result)
        results_file: Path to results.json file from previous run
        candles: Number of candles to fetch (should match original run)
    """
    print(f"\n{'='*80}")
    print(f"HOLDOUT VALIDATION")
    print(f"{'='*80}")
    print(f"Results file: {results_file}")
    print(f"Combo ID: {combo_id}")
    print(f"{'='*80}\n")

    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)

    symbol = data['symbol']
    timeframe = data['timeframe']
    results = data['results']

    # Get the combination to validate
    try:
        combo_idx = int(combo_id)
        if combo_idx < 0 or combo_idx >= len(results):
            raise ValueError(f"Invalid combo ID: {combo_id} (must be 0-{len(results)-1})")
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return

    result_data = results[combo_idx]
    combo_dict = result_data['combination']

    # Reconstruct FilterCombination
    combo = FilterCombination(**combo_dict)

    print(f"[VALIDATE] Testing combination #{combo_idx}: {combo.to_string()}")
    print(f"[VALIDATE] Train E[R]: {result_data['train_expected_r']:.3f}")
    print(f"[VALIDATE] WF E[R]: {result_data['wf_expected_r']:.3f}")
    print()

    # Fetch fresh data
    df = fetch_and_prepare_data(symbol, timeframe, candles)

    # Create discovery engine
    engine = FilterDiscoveryEngine(
        symbol=symbol,
        timeframe=timeframe,
        data=df,
        baseline_trades=9,
    )

    # Validate on holdout
    print(f"[VALIDATE] Running holdout validation...")
    holdout_result = engine.validate_on_holdout(combo)

    # Print results
    print(f"\n{'='*80}")
    print(f"HOLDOUT VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Combination: {combo.to_string()}")
    print()
    print(f"HOLDOUT (20% of data - never seen during search):")
    print(f"  PnL: ${holdout_result['holdout_pnl']:.2f}")
    print(f"  Trades: {holdout_result['holdout_trades']}")
    print(f"  E[R]: {holdout_result['holdout_expected_r']:.3f}")
    print(f"  Win Rate: {holdout_result['holdout_win_rate']:.1%}")
    print()

    # Compare to train/WF
    print(f"COMPARISON:")
    print(f"  Train E[R]:   {result_data['train_expected_r']:.3f}")
    print(f"  WF E[R]:      {result_data['wf_expected_r']:.3f}")
    print(f"  Holdout E[R]: {holdout_result['holdout_expected_r']:.3f}")
    print()

    if result_data['train_expected_r'] > 0:
        holdout_ratio = holdout_result['holdout_expected_r'] / result_data['train_expected_r']
        print(f"  Holdout/Train Ratio: {holdout_ratio:.2f}")

        if holdout_ratio >= 0.50:
            print(f"  ✓ Holdout validates the combination (ratio >= 0.50)")
        else:
            print(f"  ❌ Holdout suggests overfitting (ratio < 0.50)")

    print(f"{'='*80}\n")

    # Save holdout validation result
    output_dir = os.path.dirname(results_file)
    holdout_file = os.path.join(output_dir, f"holdout_validation_combo_{combo_idx}.txt")
    with open(holdout_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HOLDOUT VALIDATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Combination #{combo_idx}: {combo.to_string()}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(f"HOLDOUT METRICS:\n")
        f.write(f"  PnL: ${holdout_result['holdout_pnl']:.2f}\n")
        f.write(f"  Trades: {holdout_result['holdout_trades']}\n")
        f.write(f"  E[R]: {holdout_result['holdout_expected_r']:.3f}\n")
        f.write(f"  Win Rate: {holdout_result['holdout_win_rate']:.1%}\n")
        f.write("\n")
        f.write(f"COMPARISON:\n")
        f.write(f"  Train E[R]:   {result_data['train_expected_r']:.3f}\n")
        f.write(f"  WF E[R]:      {result_data['wf_expected_r']:.3f}\n")
        f.write(f"  Holdout E[R]: {holdout_result['holdout_expected_r']:.3f}\n")
        if result_data['train_expected_r'] > 0:
            f.write(f"  Holdout/Train Ratio: {holdout_ratio:.2f}\n")

    print(f"[SAVE] Holdout validation saved to: {holdout_file}")


# ==========================================
# CLI ENTRY POINT
# ==========================================

def run_analysis_mode(
    symbol: str = "BTCUSDT",
    timeframe: str = "15m",
    candles: int = 30000
):
    """Run filter pass rate analysis only (fast diagnostic).

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        candles: Number of candles to fetch

    Returns:
        Filter pass rate dict
    """
    print(f"\n{'='*80}")
    print(f"FILTER ANALYSIS MODE")
    print(f"{'='*80}")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles: {candles}")
    print(f"{'='*80}\n")

    # Fetch and prepare data
    df = fetch_and_prepare_data(symbol, timeframe, candles)

    # Create discovery engine
    engine = FilterDiscoveryEngine(
        symbol=symbol,
        timeframe=timeframe,
        data=df,
        baseline_trades=9,
    )

    # Run filter analysis
    filter_stats = engine.analyze_individual_filter_pass_rates()

    # Save results
    run_id = f"analysis_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(DATA_DIR, "filter_discovery_runs", run_id)
    os.makedirs(output_dir, exist_ok=True)

    analysis_file = os.path.join(output_dir, "filter_pass_rates.json")
    with open(analysis_file, 'w') as f:
        json.dump(filter_stats, f, indent=2)

    print(f"\n[SAVE] Filter analysis saved to: {analysis_file}")

    return filter_stats


def run_pareto_mode(
    symbol: str = "BTCUSDT",
    timeframe: str = "15m",
    candles: int = 30000,
    parallel: bool = True,
    max_workers: int = None
):
    """Run Pareto-optimal filter combination discovery.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        candles: Number of candles to fetch
        parallel: Use parallel processing
        max_workers: Max parallel workers

    Returns:
        List of Pareto-optimal results
    """
    print(f"\n{'='*80}")
    print(f"PARETO OPTIMAL MODE")
    print(f"{'='*80}")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles: {candles}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Fetch and prepare data
    df = fetch_and_prepare_data(symbol, timeframe, candles)

    # Create discovery engine
    engine = FilterDiscoveryEngine(
        symbol=symbol,
        timeframe=timeframe,
        data=df,
        baseline_trades=9,
    )

    # Run discovery
    results = engine.run_discovery(parallel=parallel, max_workers=max_workers)

    # Find Pareto-optimal combinations
    pareto_optimal = engine.find_pareto_optimal_combinations(results, min_trades=5, min_expected_r=0.0)

    elapsed = time.time() - start_time

    # Save results
    run_id = f"pareto_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(DATA_DIR, "filter_discovery_runs", run_id)
    save_results(results, symbol, timeframe, run_id, output_dir)

    # Save Pareto-optimal list
    pareto_file = os.path.join(output_dir, "pareto_optimal.json")
    with open(pareto_file, 'w') as f:
        json.dump([r.to_dict() for r in pareto_optimal], f, indent=2)

    print(f"\n[SAVE] Pareto-optimal combinations saved to: {pareto_file}")
    print(f"[COMPLETE] Pareto mode finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return pareto_optimal


def run_sensitivity_mode(
    symbol: str = "BTCUSDT",
    timeframe: str = "15m",
    candles: int = 30000,
    top_n: int = 3
):
    """Run parameter sensitivity analysis.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        candles: Number of candles to fetch
        top_n: Number of top combinations to test

    Returns:
        Sensitivity results dict
    """
    print(f"\n{'='*80}")
    print(f"PARAMETER SENSITIVITY MODE")
    print(f"{'='*80}")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles: {candles}")
    print(f"Top N: {top_n}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Fetch and prepare data
    df = fetch_and_prepare_data(symbol, timeframe, candles)

    # Create discovery engine
    engine = FilterDiscoveryEngine(
        symbol=symbol,
        timeframe=timeframe,
        data=df,
        baseline_trades=9,
    )

    # Run sensitivity analysis
    sensitivity_results = engine.generate_parameter_sensitivity_grid(top_n_combinations=top_n)

    elapsed = time.time() - start_time

    # Save results
    run_id = f"sensitivity_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(DATA_DIR, "filter_discovery_runs", run_id)
    os.makedirs(output_dir, exist_ok=True)

    sensitivity_file = os.path.join(output_dir, "sensitivity_results.json")

    # Convert to JSON-serializable format
    serializable_results = {}
    for combo_idx, combo_data in sensitivity_results.items():
        serializable_results[str(combo_idx)] = {
            'combination_str': combo_data['combination'].to_string(),
            'baseline_result': combo_data['baseline_result'].to_dict(),
            'parameter_tests': combo_data['parameter_tests'],
        }

    with open(sensitivity_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n[SAVE] Sensitivity results saved to: {sensitivity_file}")
    print(f"[COMPLETE] Sensitivity mode finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return sensitivity_results


def run_comprehensive_report(
    symbol: str = "BTCUSDT",
    timeframe: str = "15m",
    candles: int = 30000,
    parallel: bool = True,
    max_workers: int = None
):
    """Run comprehensive filter optimization report.

    Combines:
    - Filter pass rate analysis
    - Pareto-optimal discovery
    - Full discovery results
    - Comprehensive report generation

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        candles: Number of candles to fetch
        parallel: Use parallel processing
        max_workers: Max parallel workers

    Returns:
        Output directory path
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE REPORT MODE")
    print(f"{'='*80}")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles: {candles}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Fetch and prepare data
    df = fetch_and_prepare_data(symbol, timeframe, candles)

    # Create discovery engine
    engine = FilterDiscoveryEngine(
        symbol=symbol,
        timeframe=timeframe,
        data=df,
        baseline_trades=9,
    )

    # Step 1: Filter pass rate analysis
    print(f"\n{'='*80}")
    print(f"STEP 1: Filter Pass Rate Analysis")
    print(f"{'='*80}\n")
    filter_stats = engine.analyze_individual_filter_pass_rates()

    # Step 2: Run full discovery
    print(f"\n{'='*80}")
    print(f"STEP 2: Full Filter Combination Discovery")
    print(f"{'='*80}\n")
    results = engine.run_discovery(parallel=parallel, max_workers=max_workers)

    # Step 3: Find Pareto-optimal combinations
    print(f"\n{'='*80}")
    print(f"STEP 3: Pareto-Optimal Analysis")
    print(f"{'='*80}\n")
    pareto_optimal = engine.find_pareto_optimal_combinations(results, min_trades=5, min_expected_r=0.0)

    # Step 4: Generate comprehensive report
    print(f"\n{'='*80}")
    print(f"STEP 4: Generate Comprehensive Report")
    print(f"{'='*80}\n")

    run_id = f"report_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(DATA_DIR, "filter_discovery_runs", run_id)
    os.makedirs(output_dir, exist_ok=True)

    report_file = os.path.join(output_dir, "comprehensive_report.txt")
    report = engine.generate_comprehensive_report(
        results=results,
        filter_pass_rates=filter_stats,
        pareto_optimal=pareto_optimal,
        output_file=report_file
    )

    # Print report to console
    print("\n" + report)

    # Save all results
    save_results(results, symbol, timeframe, run_id, output_dir)

    # Save Pareto-optimal list
    pareto_file = os.path.join(output_dir, "pareto_optimal.json")
    with open(pareto_file, 'w') as f:
        json.dump([r.to_dict() for r in pareto_optimal], f, indent=2)

    elapsed = time.time() - start_time

    print(f"\n[COMPLETE] Comprehensive report finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[COMPLETE] All results saved to: {output_dir}")

    return output_dir


def main():
    """CLI entry point for filter discovery."""
    parser = argparse.ArgumentParser(
        description="Filter Combination Discovery for SSL Flow Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analysis mode - Quick filter pass rate analysis (fast)
  python run_filter_discovery.py --analyze --symbol BTCUSDT --timeframe 15m

  # Pareto mode - Find Pareto-optimal combinations
  python run_filter_discovery.py --pareto --symbol BTCUSDT --timeframe 15m

  # Sensitivity mode - Test parameter sensitivity
  python run_filter_discovery.py --sensitivity --symbol BTCUSDT --timeframe 15m --top-n 3

  # Report mode - Comprehensive report (combines all analyses)
  python run_filter_discovery.py --report --symbol BTCUSDT --timeframe 15m

  # Pilot mode (fast test on BTCUSDT-15m)
  python run_filter_discovery.py --pilot

  # Full mode (all symbols/timeframes)
  python run_filter_discovery.py --full

  # Validate specific combination on holdout data
  python run_filter_discovery.py --validate 0 --results data/filter_discovery_runs/pilot_20250126_123456/results.json
        """
    )

    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run filter pass rate analysis (fast, diagnostic)'
    )
    parser.add_argument(
        '--pareto',
        action='store_true',
        help='Find Pareto-optimal filter combinations'
    )
    parser.add_argument(
        '--sensitivity',
        action='store_true',
        help='Run parameter sensitivity analysis'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate comprehensive report (combines all analyses)'
    )
    parser.add_argument(
        '--pilot',
        action='store_true',
        help='Run pilot discovery on BTCUSDT-15m only (fast, recommended first run)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full discovery on all symbols and timeframes (slow)'
    )
    parser.add_argument(
        '--validate',
        type=str,
        metavar='COMBO_ID',
        help='Validate specific combination on holdout data (requires --results)'
    )
    parser.add_argument(
        '--results',
        type=str,
        metavar='PATH',
        help='Path to results.json file (for --validate mode)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading symbol (default: BTCUSDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='15m',
        help='Timeframe (default: 15m)'
    )
    parser.add_argument(
        '--candles',
        type=int,
        default=30000,
        help='Number of candles to fetch (default: 30000)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=3,
        help='Number of top combinations to test in sensitivity mode (default: 3)'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing (slower, useful for debugging)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Max parallel workers (default: auto)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.analyze, args.pareto, args.sensitivity, args.report, args.pilot, args.full, args.validate]):
        parser.error("Must specify one of: --analyze, --pareto, --sensitivity, --report, --pilot, --full, or --validate")

    if args.validate and not args.results:
        parser.error("--validate requires --results")

    parallel = not args.no_parallel

    # Run appropriate mode
    if args.analyze:
        run_analysis_mode(
            symbol=args.symbol,
            timeframe=args.timeframe,
            candles=args.candles
        )

    elif args.pareto:
        run_pareto_mode(
            symbol=args.symbol,
            timeframe=args.timeframe,
            candles=args.candles,
            parallel=parallel,
            max_workers=args.workers
        )

    elif args.sensitivity:
        run_sensitivity_mode(
            symbol=args.symbol,
            timeframe=args.timeframe,
            candles=args.candles,
            top_n=args.top_n
        )

    elif args.report:
        run_comprehensive_report(
            symbol=args.symbol,
            timeframe=args.timeframe,
            candles=args.candles,
            parallel=parallel,
            max_workers=args.workers
        )

    elif args.pilot:
        run_pilot_discovery(
            candles=args.candles,
            parallel=parallel,
            max_workers=args.workers
        )

    elif args.full:
        run_full_discovery(
            candles=args.candles,
            parallel=parallel,
            max_workers=args.workers
        )

    elif args.validate:
        validate_combination_on_holdout(
            combo_id=args.validate,
            results_file=args.results,
            candles=args.candles
        )


if __name__ == "__main__":
    main()
