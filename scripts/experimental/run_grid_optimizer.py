#!/usr/bin/env python3
"""
Grid Search Optimizer Runner

Hierarchical grid search for SSL Flow strategy parameters with
statistical validation and robustness testing.

Usage:
    python run_grid_optimizer.py --symbol BTCUSDT --quick            # Coarse grid only
    python run_grid_optimizer.py --symbol BTCUSDT --full             # Full hierarchical
    python run_grid_optimizer.py --symbol BTCUSDT --robust           # With robustness
    python run_grid_optimizer.py --symbol BTCUSDT --timeframe 1h     # Specific timeframe
    python run_grid_optimizer.py --symbol BTCUSDT --start 2024-01-01 # Custom date range

Modes:
    --quick:    Coarse grid only (fast screening) - ~5-10 min
    --full:     Coarse + fine grid (hierarchical) - ~15-30 min
    --robust:   Coarse + fine + robustness test - ~30-45 min

The optimizer will:
    1. Phase 1: Test coarse parameter grid (4x3x4x3 = 144 combinations)
    2. Phase 2: Refine around top 5 performers (fine grid)
    3. Phase 3: Test parameter stability (perturbation analysis)
    4. Apply Bonferroni correction for multiple comparisons
    5. Calculate bootstrap confidence intervals for Sharpe
    6. Save results to data/grid_search_runs/{run_id}/
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import TradingEngine
from core.grid_optimizer import GridSearchOptimizer


def main():
    parser = argparse.ArgumentParser(
        description='Grid Search Optimizer for SSL Flow Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading symbol (default: BTCUSDT)')

    # Optional arguments
    parser.add_argument('--timeframe', type=str, default='15m',
                       help='Timeframe (default: 15m)')
    parser.add_argument('--start', type=str, default=None,
                       help='Start date YYYY-MM-DD (default: 1 year ago)')
    parser.add_argument('--end', type=str, default=None,
                       help='End date YYYY-MM-DD (default: today)')

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--quick', action='store_true',
                           help='Quick mode: coarse grid only (fastest)')
    mode_group.add_argument('--full', action='store_true',
                           help='Full mode: coarse + fine grid (default)')
    mode_group.add_argument('--robust', action='store_true',
                           help='Robust mode: coarse + fine + robustness test')

    # Other options
    parser.add_argument('--workers', type=int, default=None,
                       help='Max parallel workers (default: auto)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    # Determine dates
    if args.end is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end

    if args.start is None:
        # Default: 1 year ago
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=365)
        start_date = start_dt.strftime('%Y-%m-%d')
    else:
        start_date = args.start

    # Determine mode
    if args.quick:
        mode = 'quick'
    elif args.robust:
        mode = 'robust'
    else:
        mode = 'full'

    print(f"\n{'='*70}")
    print(f"GRID SEARCH OPTIMIZER")
    print(f"{'='*70}")
    print(f"Symbol:      {args.symbol}")
    print(f"Timeframe:   {args.timeframe}")
    print(f"Date Range:  {start_date} -> {end_date}")
    print(f"Mode:        {mode.upper()}")
    print(f"{'='*70}\n")

    # Fetch data
    print(f"[1/3] Fetching historical data...")
    df = TradingEngine.get_historical_data_pagination(
        args.symbol,
        args.timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    if df is None or df.empty:
        print(f"❌ Error: No data available for {args.symbol}-{args.timeframe}")
        return 1

    print(f"      Loaded {len(df)} candles")

    # Calculate indicators
    print(f"\n[2/3] Calculating indicators...")
    df = TradingEngine.calculate_indicators(df, timeframe=args.timeframe)
    print(f"      Indicators calculated")

    # Initialize optimizer
    print(f"\n[3/3] Running grid search...")
    optimizer = GridSearchOptimizer(
        symbol=args.symbol,
        timeframe=args.timeframe,
        data=df,
        verbose=not args.quiet,
        max_workers=args.workers,
    )

    # Run search
    if mode == 'quick':
        results = optimizer.run_full_search(quick=True, robust=False)
    elif mode == 'robust':
        results = optimizer.run_full_search(quick=False, robust=True)
    else:  # full
        results = optimizer.run_full_search(quick=False, robust=False)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    coarse_count = len(results.get('coarse_results', []))
    fine_count = len(results.get('fine_results', []))

    print(f"Coarse grid:    {coarse_count} configurations tested")
    if fine_count > 0:
        print(f"Fine grid:      {fine_count} configurations tested")

    # Best result
    best_results = results.get('fine_results') or results.get('coarse_results', [])
    if best_results:
        best = best_results[0]
        print(f"\n{'='*70}")
        print(f"BEST CONFIGURATION")
        print(f"{'='*70}")
        print(f"Score:          {best.robust_score:.4f}")
        print(f"Sharpe Ratio:   {best.sharpe_ratio:.2f}")
        print(f"Total PnL:      ${best.total_pnl:.2f}")
        print(f"Trade Count:    {best.trade_count}")
        print(f"Win Rate:       {best.win_rate*100:.1f}%")
        print(f"E[R]:           {best.expected_r:.3f}")
        print(f"\nParameters:")
        print(f"  RSI Threshold:      {best.config.rsi_threshold}")
        print(f"  ADX Threshold:      {best.config.adx_threshold}")
        print(f"  RR Ratio:           {best.config.rr_ratio}")
        print(f"  Regime ADX Avg:     {best.config.regime_adx_avg}")

        if best.is_significant:
            print(f"\nStatistical Validation:")
            print(f"  Significant:        YES (p={best.p_value:.6f})")
            print(f"  t-statistic:        {best.t_statistic:.2f}")
            if best.sharpe_ci_lower != 0:
                print(f"  Sharpe 95% CI:      [{best.sharpe_ci_lower:.2f}, {best.sharpe_ci_upper:.2f}]")

        if best.robustness_score > 0:
            print(f"\nRobustness:")
            print(f"  Stable Neighbors:   {best.stable_neighbors_pct:.1f}%")

    print(f"\n{'='*70}")
    print(f"Results saved to: {results.get('output_dir')}")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
