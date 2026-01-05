#!/usr/bin/env python3
"""
SSL Never Lost Filter A/B Test

Tests the impact of enabling the SSL Never Lost filter:
- Filter OFF (baseline): Takes all signals regardless of trend strength
- Filter ON (test): Skips counter-trend trades when SSL baseline was never broken

This filter addresses the annotation: "SSL HYBRID not even lost for a moment,
should not been a short trade here" - preventing counter-trend entries against
strong momentum.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import config as config_module
from runners.rolling_wf_optimized import run_rolling_walkforward_optimized


def run_test(
    use_ssl_never_lost_filter: bool,
    ssl_never_lost_lookback: int = 20,
    symbols: list = None,
    start_date: str = "2025-06-01",
    end_date: str = "2025-12-01",
) -> Dict[str, Any]:
    """Run a single test configuration."""

    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "LINKUSDT"]

    # Set the config
    original_config = config_module.DEFAULT_STRATEGY_CONFIG.copy()
    config_module.DEFAULT_STRATEGY_CONFIG["use_ssl_never_lost_filter"] = use_ssl_never_lost_filter
    config_module.DEFAULT_STRATEGY_CONFIG["ssl_never_lost_lookback"] = ssl_never_lost_lookback

    label = "ON" if use_ssl_never_lost_filter else "OFF"
    print(f"\n{'='*60}")
    print(f"Running test: SSL Never Lost Filter = {label}")
    print(f"Lookback: {ssl_never_lost_lookback} candles")
    print(f"Symbols: {symbols}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}\n")

    # Silence circuit breaker logs during test
    import logging
    logging.getLogger("core.trade_manager").setLevel(logging.WARNING)

    try:
        results = run_rolling_walkforward_optimized(
            symbols=symbols,
            timeframes=["15m", "30m"],
            start_date=start_date,
            end_date=end_date,
            lookback_days=60,
            forward_days=7,
            mode="weekly",
            use_master_cache=True,
            parallel_optimization=True,
            verbose=True,  # Enable verbose to see progress
        )

        # Debug: Print raw metrics
        if results and "metrics" in results:
            m = results["metrics"]
            print(f"\n[DEBUG] Raw metrics: PnL=${m.get('total_pnl', 0):.2f}, "
                  f"Trades={m.get('total_trades', 0)}, "
                  f"WinRate={m.get('win_rate', 0)*100:.1f}%")

        # Restore original config
        config_module.DEFAULT_STRATEGY_CONFIG.update(original_config)

        return {
            "filter_enabled": use_ssl_never_lost_filter,
            "lookback": ssl_never_lost_lookback,
            "results": results,
        }

    except Exception as e:
        # Restore original config
        config_module.DEFAULT_STRATEGY_CONFIG.update(original_config)
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_metrics(result: Dict) -> Tuple[float, int, float, float]:
    """Extract key metrics from test result."""
    if result is None or result.get("results") is None:
        return 0.0, 0, 0.0, 0.0

    results = result["results"]

    # The result is a dict with "metrics" key from rolling_wf_optimized
    if isinstance(results, dict) and "metrics" in results:
        metrics = results["metrics"]
        total_pnl = metrics.get("total_pnl", 0)
        total_trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0) * 100  # Convert to percentage
        max_dd = abs(metrics.get("max_drawdown", 0))
        return total_pnl, total_trades, win_rate, max_dd

    # Fallback: Results is a list of stream results
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    max_dd = 0.0

    if isinstance(results, list):
        for stream_result in results:
            if isinstance(stream_result, dict):
                total_pnl += stream_result.get("total_pnl", 0)
                total_trades += stream_result.get("total_trades", 0)
                total_wins += stream_result.get("wins", 0)
                max_dd = max(max_dd, abs(stream_result.get("max_drawdown", 0)))

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

    return total_pnl, total_trades, win_rate, max_dd


def print_comparison(baseline: Dict, test: Dict):
    """Print comparison of baseline vs test results."""

    baseline_pnl, baseline_trades, baseline_wr, baseline_dd = extract_metrics(baseline)
    test_pnl, test_trades, test_wr, test_dd = extract_metrics(test)

    pnl_delta = test_pnl - baseline_pnl
    trades_delta = test_trades - baseline_trades
    wr_delta = test_wr - baseline_wr
    dd_delta = test_dd - baseline_dd

    print("\n" + "="*70)
    print("SSL NEVER LOST FILTER A/B TEST RESULTS")
    print("="*70)
    print(f"\n{'Metric':<25} {'Baseline (OFF)':<18} {'Test (ON)':<18} {'Delta':<15}")
    print("-"*70)

    # PnL
    pnl_color = "+" if pnl_delta > 0 else ""
    print(f"{'Total PnL':<25} ${baseline_pnl:<17.2f} ${test_pnl:<17.2f} {pnl_color}${pnl_delta:.2f}")

    # Trades
    trades_color = "+" if trades_delta > 0 else ""
    print(f"{'Total Trades':<25} {baseline_trades:<18} {test_trades:<18} {trades_color}{trades_delta}")

    # Win Rate
    wr_color = "+" if wr_delta > 0 else ""
    print(f"{'Win Rate':<25} {baseline_wr:<17.1f}% {test_wr:<17.1f}% {wr_color}{wr_delta:.1f}%")

    # Max Drawdown
    dd_color = "+" if dd_delta < 0 else ""  # Lower is better
    print(f"{'Max Drawdown':<25} ${baseline_dd:<17.2f} ${test_dd:<17.2f} {dd_color}${dd_delta:.2f}")

    print("-"*70)

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if pnl_delta > 0:
        print(f"\n✅ SSL Never Lost Filter IMPROVED PnL by ${pnl_delta:.2f}")
        if trades_delta < 0:
            filtered_trades = abs(trades_delta)
            print(f"   → Filtered {filtered_trades} counter-trend trades")
            if pnl_delta > 0 and filtered_trades > 0:
                avg_saved = pnl_delta / filtered_trades
                print(f"   → Average saved per filtered trade: ${avg_saved:.2f}")
    elif pnl_delta < 0:
        print(f"\n❌ SSL Never Lost Filter REDUCED PnL by ${abs(pnl_delta):.2f}")
        print("   → Filter may be too aggressive or conflicting with other logic")
    else:
        print("\n⚪ SSL Never Lost Filter had NO IMPACT on PnL")

    if wr_delta > 0:
        print(f"\n✅ Win rate improved by {wr_delta:.1f}%")
    elif wr_delta < 0:
        print(f"\n⚠️  Win rate decreased by {abs(wr_delta):.1f}%")

    print("\n" + "="*70)


def main():
    """Run the A/B test."""
    import argparse

    parser = argparse.ArgumentParser(description="SSL Never Lost Filter A/B Test")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "LINKUSDT"],
                       help="Symbols to test")
    parser.add_argument("--start-date", default="2025-06-01", help="Start date")
    parser.add_argument("--end-date", default="2025-12-01", help="End date")
    parser.add_argument("--lookback", type=int, default=20,
                       help="SSL Never Lost lookback period (candles)")
    parser.add_argument("--full-year", action="store_true",
                       help="Run full year test (2025-01-01 to 2025-12-01)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with fewer symbols (BTCUSDT only)")

    args = parser.parse_args()

    if args.full_year:
        args.start_date = "2025-01-01"
        args.end_date = "2025-12-01"

    if args.quick:
        args.symbols = ["BTCUSDT"]

    print("\n" + "="*70)
    print("SSL NEVER LOST FILTER A/B TEST")
    print("="*70)
    print(f"\nThis test compares:")
    print(f"  BASELINE: use_ssl_never_lost_filter = False (current)")
    print(f"  TEST:     use_ssl_never_lost_filter = True")
    print(f"\nThe filter prevents counter-trend trades when SSL baseline")
    print(f"was never broken in the lookback period ({args.lookback} candles).")
    print(f"\nThis addresses your annotation:")
    print(f'  "SSL HYBRID not even lost for a moment, should not been a short trade here"')

    start_time = datetime.now()

    # Run baseline (filter OFF)
    print("\n" + "-"*70)
    print("PHASE 1: Running BASELINE (filter OFF)")
    print("-"*70)
    baseline = run_test(
        use_ssl_never_lost_filter=False,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Run test (filter ON)
    print("\n" + "-"*70)
    print("PHASE 2: Running TEST (filter ON)")
    print("-"*70)
    test = run_test(
        use_ssl_never_lost_filter=True,
        ssl_never_lost_lookback=args.lookback,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Print comparison
    print_comparison(baseline, test)

    elapsed = datetime.now() - start_time
    print(f"\nTotal test time: {elapsed}")

    return baseline, test


if __name__ == "__main__":
    main()
