#!/usr/bin/env python3
"""
Grid Search Optimizer Example

Quick example demonstrating the grid optimizer API usage.
This script runs a quick grid search on BTCUSDT-15m data.

Usage:
    python examples/grid_optimizer_example.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grid_optimizer import GridSearchOptimizer
from core import TradingEngine


def main():
    print("="*70)
    print("GRID OPTIMIZER API EXAMPLE")
    print("="*70)
    print()

    # Configuration
    symbol = 'BTCUSDT'
    timeframe = '15m'
    start_date = '2024-06-01'
    end_date = '2024-12-01'

    print(f"Configuration:")
    print(f"  Symbol:      {symbol}")
    print(f"  Timeframe:   {timeframe}")
    print(f"  Date Range:  {start_date} -> {end_date}")
    print()

    # Step 1: Fetch data
    print("[1/4] Fetching historical data...")
    df = TradingEngine.get_historical_data_pagination(
        symbol,
        timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    if df is None or df.empty:
        print("Error: No data available")
        return 1

    print(f"      Loaded {len(df)} candles")

    # Step 2: Calculate indicators
    print()
    print("[2/4] Calculating indicators...")
    df = TradingEngine.calculate_indicators(df, timeframe=timeframe)
    print(f"      Indicators calculated")

    # Step 3: Initialize optimizer
    print()
    print("[3/4] Initializing grid search optimizer...")
    optimizer = GridSearchOptimizer(
        symbol=symbol,
        timeframe=timeframe,
        data=df,
        verbose=True,
        max_workers=4,  # Use 4 workers for demo
    )
    print(f"      Optimizer initialized (Run ID: {optimizer.run_id})")

    # Step 4: Run quick search
    print()
    print("[4/4] Running quick grid search (coarse grid only)...")
    print()

    results = optimizer.run_full_search(
        quick=True,   # Coarse grid only (fastest)
        robust=False  # Skip robustness testing
    )

    # Print summary
    print()
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    coarse_results = results.get('coarse_results', [])
    print(f"\nTested {len(coarse_results)} configurations")

    if coarse_results:
        print(f"\nTop 5 Configurations:")
        print("-" * 70)

        for i, result in enumerate(coarse_results[:5], 1):
            cfg = result.config
            print(f"\n#{i} | Score: {result.robust_score:.4f}")
            print(f"    RSI={cfg.rsi_threshold}, ADX={cfg.adx_threshold}, "
                  f"RR={cfg.rr_ratio}, RegimeADX={cfg.regime_adx_avg}")
            print(f"    PnL: ${result.total_pnl:.2f} | Trades: {result.trade_count} | "
                  f"Win Rate: {result.win_rate*100:.1f}% | E[R]: {result.expected_r:.3f}")
            print(f"    Sharpe: {result.sharpe_ratio:.2f} | "
                  f"Max DD: {result.max_drawdown_pct*100:.1f}%")

        # Show best config in detail
        best = coarse_results[0]
        print()
        print("="*70)
        print("BEST CONFIGURATION DETAILS")
        print("="*70)
        print(f"\nParameters:")
        print(f"  RSI Threshold:      {best.config.rsi_threshold}")
        print(f"  ADX Threshold:      {best.config.adx_threshold}")
        print(f"  RR Ratio:           {best.config.rr_ratio}")
        print(f"  Regime ADX Avg:     {best.config.regime_adx_avg}")
        print(f"  AlphaTrend Active:  {best.config.at_active}")

        print(f"\nPerformance Metrics:")
        print(f"  Robust Score:       {best.robust_score:.4f}")
        print(f"  Total PnL:          ${best.total_pnl:.2f}")
        print(f"  Trade Count:        {best.trade_count}")
        print(f"  Win Rate:           {best.win_rate*100:.1f}%")
        print(f"  Avg Win:            ${best.avg_win:.2f}")
        print(f"  Avg Loss:           ${best.avg_loss:.2f}")
        print(f"  Profit Factor:      {best.profit_factor:.2f}")
        print(f"  Sharpe Ratio:       {best.sharpe_ratio:.2f}")
        print(f"  Max Drawdown:       {best.max_drawdown_pct*100:.1f}%")
        print(f"  E[R]:               {best.expected_r:.3f}")

        # Save config to dict
        best_config_dict = best.config.to_dict()
        print(f"\nBacktest Config (copy to best_configs.json):")
        print(f"  {best_config_dict}")

    else:
        print("\nNo valid configurations found.")
        print("Try:")
        print("  - Increasing date range")
        print("  - Lowering MIN_TRADES_THRESHOLD")
        print("  - Checking if strategy has edge on this symbol")

    # Show output location
    print()
    print("="*70)
    print(f"Detailed results saved to:")
    print(f"  {results['output_dir']}")
    print("="*70)
    print()

    # Next steps
    print("Next Steps:")
    print("  1. Review top_10.txt for detailed parameter analysis")
    print("  2. Run full hierarchical search for refinement:")
    print(f"     python run_grid_optimizer.py --symbol {symbol} --timeframe {timeframe} --full")
    print("  3. Run robustness test on best config:")
    print(f"     python run_grid_optimizer.py --symbol {symbol} --timeframe {timeframe} --robust")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
