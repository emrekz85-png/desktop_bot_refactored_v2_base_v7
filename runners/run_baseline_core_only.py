#!/usr/bin/env python3
"""
BASELINE TEST - CORE ONLY Configuration
Expert Panel Phase 1 Recommendation

Purpose:
    Establish statistical foundation with 100+ trades by stripping all optional filters.
    This is the CRITICAL FIRST STEP before any optimization.

What this does:
    - Uses CORE ONLY config (only 4 essential filters vs current 15+)
    - Runs full-year backtest on recommended symbols (BTC, ETH, LINK)
    - Expected: 80-150 trades (vs current 13)
    - Goal: Statistical significance, NOT optimization

Expert Panel Quote:
    "13 trades is not a backtest, it's a coin flip session. Get to 100+ trades first,
     then we can talk about optimization." - Andreas Clenow

Usage:
    python run_baseline_core_only.py              # Full year test (recommended)
    python run_baseline_core_only.py --quick      # Quick test (3 months)
    python run_baseline_core_only.py --btc-only   # BTC only (fastest)
"""

import sys
import os
from datetime import datetime, timedelta
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from core.version import print_version_banner
from core.config_core_only import (
    CORE_ONLY_STRATEGY_CONFIG,
    print_config_comparison,
    get_active_filter_count
)
from runners.rolling_wf_optimized import run_rolling_walkforward_optimized


def print_baseline_banner():
    """Print baseline test banner."""
    print("\n" + "="*80)
    print("║" + " "*78 + "║")
    print("║" + " "*20 + "BASELINE TEST - CORE ONLY CONFIG" + " "*26 + "║")
    print("║" + " "*15 + "Expert Panel Phase 1: Statistical Foundation" + " "*20 + "║")
    print("║" + " "*78 + "║")
    print("="*80)
    print("")
    print("  GOAL: Achieve 100+ trades for statistical significance")
    print("  METHOD: Strip all optional filters, keep only essentials")
    print("  EXPECTED: 7-10x trade increase (13 → 100+)")
    print("")
    print("="*80)


def run_baseline_test(
    start_date: str,
    end_date: str,
    symbols: list = None,
    timeframes: list = None,
    verbose: bool = True
):
    """
    Run baseline test with CORE ONLY configuration.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        symbols: List of symbols to test (default: BTC, ETH, LINK)
        timeframes: List of timeframes (default: 15m, 1h)
        verbose: Print detailed output
    """
    if symbols is None:
        # Recommended symbols from expert panel analysis
        symbols = ["BTCUSDT", "ETHUSDT", "LINKUSDT"]

    if timeframes is None:
        # Recommended timeframes
        timeframes = ["15m", "1h"]

    # Print configuration comparison
    if verbose:
        print_baseline_banner()
        print_config_comparison()
        print("\n" + "="*80)
        print("TEST PARAMETERS")
        print("="*80)
        print(f"  Period: {start_date} → {end_date}")
        print(f"  Symbols: {', '.join(symbols)}")
        print(f"  Timeframes: {', '.join(timeframes)}")
        print(f"  Active Filters: {get_active_filter_count(CORE_ONLY_STRATEGY_CONFIG)}")
        print("="*80)
        print("\nStarting baseline test...\n")

    # Run optimized rolling walk-forward with core-only config
    # Mode: "fixed" (no optimization, just use core config)
    results = run_rolling_walkforward_optimized(
        mode="fixed",
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        timeframes=timeframes,
        fixed_config=CORE_ONLY_STRATEGY_CONFIG,  # Use core-only config
        lookback_days=60,
        forward_days=7,
        use_master_cache=True,
        parallel_optimization=True,
        verbose=verbose
    )

    return results


def print_results_summary(results: dict):
    """Print summary of baseline test results."""
    print("\n" + "="*80)
    print("BASELINE TEST RESULTS - CORE ONLY")
    print("="*80)

    # Extract overall metrics
    overall = results.get("overall", {})
    total_pnl = overall.get("pnl", 0)
    total_trades = overall.get("trades", 0)
    win_rate = overall.get("win_rate", 0)
    wins = int(total_trades * win_rate) if total_trades > 0 else 0
    losses = total_trades - wins

    print(f"\n  Total Trades: {total_trades}")
    print(f"  Win Rate: {win_rate*100:.1f}% ({wins}W / {losses}L)")
    print(f"  Total PnL: ${total_pnl:+.2f}")

    if total_trades > 0:
        avg_pnl = total_pnl / total_trades
        print(f"  Avg PnL/Trade: ${avg_pnl:+.2f}")

    # Statistical validity check
    print("\n" + "-"*80)
    print("STATISTICAL VALIDITY CHECK")
    print("-"*80)

    if total_trades >= 100:
        print(f"  ✅ EXCELLENT: {total_trades} trades (100+ target met)")
        print("     → Results are statistically significant")
        print("     → Ready for optimization in Phase 2")
    elif total_trades >= 50:
        print(f"  ✅ GOOD: {total_trades} trades (50+ acceptable)")
        print("     → Results have moderate statistical significance")
        print("     → Can proceed to Phase 2 with caution")
    elif total_trades >= 30:
        print(f"  ⚠️  MARGINAL: {total_trades} trades (30+ minimum)")
        print("     → Results have weak statistical significance")
        print("     → Consider loosening filters further")
    else:
        print(f"  ❌ INSUFFICIENT: {total_trades} trades (<30)")
        print("     → Results are statistically unreliable")
        print("     → MUST loosen filters more or extend test period")

    # Comparison with current system
    print("\n" + "-"*80)
    print("COMPARISON WITH CURRENT SYSTEM")
    print("-"*80)
    print("  Metric              Current (v2.0.0)    Core Only      Change")
    print("  " + "-"*72)

    current_trades = 13
    current_pnl = -39.90
    current_wr = 0.31

    trade_change = ((total_trades / current_trades) - 1) * 100 if current_trades > 0 else 0
    pnl_change = total_pnl - current_pnl
    wr_change = (win_rate - current_wr) * 100

    print(f"  Trades              {current_trades:>6}           {total_trades:>6}        {trade_change:+.0f}%")
    print(f"  Win Rate            {current_wr*100:>6.1f}%         {win_rate*100:>6.1f}%       {wr_change:+.1f}pp")
    print(f"  PnL                 ${current_pnl:>6.2f}       ${total_pnl:>7.2f}     ${pnl_change:+.2f}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)

    if total_trades >= 50:
        print("\n  Phase 1 COMPLETE ✅")
        print("\n  Proceed to Phase 2:")
        print("  1. Start signal journaling (2 weeks)")
        print("  2. Extract your cognitive patterns")
        print("  3. Implement regime detector")
        print("  4. Add discovered patterns incrementally")
        print("\n  See: docs/EXPERT_SPECIFICATION_PANEL_RECOMMENDATIONS.md")
    else:
        print("\n  Phase 1 INCOMPLETE ⚠️")
        print("\n  Action Required:")
        print("  1. Review core-only config (too strict?)")
        print("  2. Consider even looser thresholds")
        print("  3. Extend test period (try full year)")
        print("  4. Check data availability")

    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Test - Core Only Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_baseline_core_only.py                # Full year (recommended)
  python run_baseline_core_only.py --quick        # Quick 3-month test
  python run_baseline_core_only.py --btc-only     # BTC only (fastest)

Expert Panel Phase 1 Goal:
  Achieve 100+ trades for statistical significance before optimization.
        """
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test (3 months instead of full year)'
    )
    parser.add_argument(
        '--btc-only',
        action='store_true',
        help='Test only BTCUSDT (fastest, for validation)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD), default: 2025-01-01'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD), default: 2025-12-31'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)'
    )
    parser.add_argument(
        '--timeframes',
        type=str,
        help='Comma-separated timeframes (e.g., 15m,1h)'
    )

    args = parser.parse_args()

    # Print version banner
    print_version_banner()

    # Determine date range
    if args.quick:
        # 3-month quick test
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
    elif args.start_date and args.end_date:
        # Custom date range
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        # Full year (default)
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 12, 31)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Determine symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    elif args.btc_only:
        symbols = ["BTCUSDT"]
    else:
        symbols = ["BTCUSDT", "ETHUSDT", "LINKUSDT"]

    # Determine timeframes
    if args.timeframes:
        timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    else:
        timeframes = ["15m", "1h"]

    # Run baseline test
    try:
        results = run_baseline_test(
            start_date=start_date_str,
            end_date=end_date_str,
            symbols=symbols,
            timeframes=timeframes,
            verbose=True
        )

        # Print results summary
        print_results_summary(results)

        # Save results to file
        output_file = f"baseline_core_only_{start_date_str}_{end_date_str}.json"
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
