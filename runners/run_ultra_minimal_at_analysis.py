#!/usr/bin/env python3
"""
Ultra-Minimal Test with AlphaTrend Scoring Analysis

Purpose:
    Scientific approach to determine optimal AlphaTrend integration:
    1. Run test with AT in SCORE mode (doesn't block signals)
    2. Save AT scores with each trade
    3. Analyze correlation between AT scores and trade outcomes
    4. Determine data-driven AT integration strategy

Expected Outcome:
    - 50-100 trades (vs 4 in core-only)
    - Each trade has AT metadata for analysis
    - Post-test analysis reveals optimal AT usage

Usage:
    python run_ultra_minimal_at_analysis.py              # Full year
    python run_ultra_minimal_at_analysis.py --quick      # Quick test
    python run_ultra_minimal_at_analysis.py --btc-only   # BTC only
"""

import sys
import os
from datetime import datetime, timedelta
import argparse
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.version import print_version_banner
from core.config_ultra_minimal import (
    ULTRA_MINIMAL_STRATEGY_CONFIG,
    print_config_comparison,
    get_analysis_metadata_fields,
    AT_SCORE_INTERPRETATION,
)
from runners.rolling_wf_optimized import run_rolling_walkforward_optimized


def print_test_banner():
    """Print test banner."""
    print("\n" + "="*80)
    print("║" + " "*78 + "║")
    print("║" + " "*15 + "ULTRA-MINIMAL TEST - AlphaTrend Analysis" + " "*23 + "║")
    print("║" + " "*10 + "Scientific Approach: Data Collection → Analysis → Decision" + " "*11 + "║")
    print("║" + " "*78 + "║")
    print("="*80)
    print("")
    print("  METHOD: AlphaTrend in SCORE mode (doesn't block)")
    print("  GOAL: 50-100 trades with AT metadata for analysis")
    print("  ANALYSIS: Correlation between AT scores and trade outcomes")
    print("")
    print("="*80)


def run_ultra_minimal_test(
    start_date: str,
    end_date: str,
    symbols: list = None,
    timeframes: list = None,
    verbose: bool = True
):
    """
    Run ultra-minimal test with AT scoring enabled.
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "LINKUSDT"]

    if timeframes is None:
        timeframes = ["15m", "1h"]

    if verbose:
        print_test_banner()
        print_config_comparison()
        print("\n" + "="*80)
        print("TEST PARAMETERS")
        print("="*80)
        print(f"  Period: {start_date} → {end_date}")
        print(f"  Symbols: {', '.join(symbols)}")
        print(f"  Timeframes: {', '.join(timeframes)}")
        print(f"  AT Mode: SCORE (data collection)")
        print(f"  Blocks signals: NO (AT won't reject)")
        print("="*80)
        print("\nStarting ultra-minimal test...\n")

    # Run with ultra-minimal config
    results = run_rolling_walkforward_optimized(
        mode="fixed",
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        timeframes=timeframes,
        fixed_config=ULTRA_MINIMAL_STRATEGY_CONFIG,
        lookback_days=60,
        forward_days=7,
        use_master_cache=True,
        parallel_optimization=True,
        verbose=verbose
    )

    return results


def extract_trade_metadata(results: dict):
    """
    Extract trade-level metadata including AT scores.

    Returns:
        list: Trade records with metadata
    """
    trades = []

    # Extract from top-level trades list
    all_trades = results.get("trades", [])

    for trade in all_trades:
        # Extract indicators_at_entry if available
        indicators = trade.get("indicators_at_entry", {})

        # Extract basic trade info
        trade_record = {
            "symbol": trade.get("symbol"),
            "timeframe": trade.get("timeframe"),
            "type": trade.get("type"),
            "entry": trade.get("entry"),
            "exit": trade.get("close_price"),  # Changed from "exit" to "close_price"
            "pnl": trade.get("pnl"),
            "r_multiple": trade.get("r_multiple"),
            "status": trade.get("status"),
            "duration_hours": trade.get("duration_hours"),

            # AT metadata from indicators_at_entry
            "at_score": indicators.get("at_score", None),
            "at_regime": indicators.get("at_regime", "unknown"),
            "at_buyers_dominant": indicators.get("at_buyers_dominant", False),
            "at_sellers_dominant": indicators.get("at_sellers_dominant", False),
            "at_is_flat": indicators.get("at_is_flat", False),

            # Outcome
            "is_win": trade.get("pnl", 0) > 0,
        }

        trades.append(trade_record)

    return trades


def save_trades_for_analysis(trades: list, filename: str):
    """Save trades with metadata to JSON for analysis."""
    with open(filename, 'w') as f:
        json.dump(trades, f, indent=2, default=str)

    print(f"\n✅ Trades saved for analysis: {filename}")
    print(f"   Total trades: {len(trades)}")
    print(f"   Use: python scripts/analyze_at_correlation.py {filename}")


def print_initial_summary(results: dict):
    """Print initial summary before deep analysis."""
    metrics = results.get("metrics", {})

    total_trades = metrics.get("total_trades", 0)
    win_rate = metrics.get("win_rate", 0)
    total_pnl = metrics.get("total_pnl", 0)

    print("\n" + "="*80)
    print("ULTRA-MINIMAL TEST RESULTS - Initial Summary")
    print("="*80)

    print(f"\n  Total Trades: {total_trades}")
    print(f"  Win Rate: {win_rate*100:.1f}%")
    print(f"  Total PnL: ${total_pnl:+.2f}")

    # Success check
    print("\n" + "-"*80)
    print("PHASE 1 SUCCESS CHECK")
    print("-"*80)

    if total_trades >= 100:
        print(f"  ✅ EXCELLENT: {total_trades} trades!")
        print("     → Achieved 100+ target")
        print("     → Ready for deep AT analysis")
        print("     → Statistical significance achieved")
    elif total_trades >= 50:
        print(f"  ✅ GOOD: {total_trades} trades")
        print("     → Achieved 50+ minimum")
        print("     → Can proceed with AT analysis")
        print("     → Moderate statistical power")
    elif total_trades >= 30:
        print(f"  ⚠️  MARGINAL: {total_trades} trades")
        print("     → Below target but analyzable")
        print("     → Consider loosening further")
        print("     → Weak statistical power")
    else:
        print(f"  ❌ INSUFFICIENT: {total_trades} trades")
        print("     → Still below minimum")
        print("     → Need to investigate further")
        print("     → Check logs for rejection reasons")

    # Comparison
    print("\n" + "-"*80)
    print("COMPARISON WITH PREVIOUS TESTS")
    print("-"*80)
    print("  Test                Trades     Change from Baseline")
    print("  " + "-"*70)
    print(f"  Current (v2.0.0)    13         (baseline)")
    print(f"  Core-Only           4          -69% ❌")
    print(f"  Ultra-Minimal       {total_trades:<10} {((total_trades/13-1)*100):+.0f}%{'✅' if total_trades > 13 else '❌'}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-Minimal Test with AT Scoring Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--quick', action='store_true', help='Quick 3-month test')
    parser.add_argument('--btc-only', action='store_true', help='BTC only (fastest)')
    parser.add_argument('--start-date', type=str, help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, help='End date YYYY-MM-DD')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--timeframes', type=str, help='Comma-separated timeframes')

    args = parser.parse_args()

    print_version_banner()

    # Determine date range
    if args.quick:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
    elif args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        # Full year
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

    try:
        # Run test
        results = run_ultra_minimal_test(
            start_date=start_date_str,
            end_date=end_date_str,
            symbols=symbols,
            timeframes=timeframes,
            verbose=True
        )

        # Print initial summary
        print_initial_summary(results)

        # Extract trade metadata
        trades = extract_trade_metadata(results)

        # Save results
        results_file = f"ultra_minimal_results_{start_date_str}_{end_date_str}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nFull results saved: {results_file}")

        # Save trades for AT analysis
        trades_file = f"ultra_minimal_trades_{start_date_str}_{end_date_str}.json"
        save_trades_for_analysis(trades, trades_file)

        # Next steps
        print("\n" + "="*80)
        print("NEXT STEPS - AT CORRELATION ANALYSIS")
        print("="*80)
        print(f"""
1. Run deep analysis:
   python scripts/analyze_at_correlation.py {trades_file}

2. Analysis will answer:
   ✓ Do high AT scores predict winning trades?
   ✓ What's the optimal AT threshold (if any)?
   ✓ Should we use Binary, Score, or Off mode?
   ✓ Are there regimes where AT helps vs hurts?
   ✓ Does AT flat indicate good reversal setups?

3. Generate recommendations:
   - Optimal AT integration strategy
   - Data-driven thresholds
   - Regime-specific usage

4. Implement findings:
   - Update config based on analysis
   - Re-test with optimal AT settings
   - Proceed to Phase 2 (signal journaling)
        """)
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
