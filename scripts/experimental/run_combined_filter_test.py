#!/usr/bin/env python3
"""
Combined Filter A/B Test: SSL Never Lost + Confirmation Candle

Tests the combined impact of:
1. P1: SSL Never Lost Filter - prevents counter-trend trades
2. P2: Confirmation Candle - waits for momentum confirmation before entry

This addresses user annotations:
- "SSL HYBRID not even lost for a moment, should not been a short trade here"
- "Entry too early"
- "Early entry, need to find a way to re-enter"
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import config as config_module
from runners.rolling_wf_optimized import run_rolling_walkforward_optimized
import logging

# Suppress noisy logs
logging.getLogger("core.trade_manager").setLevel(logging.WARNING)


def run_test(test_name: str, config_overrides: dict, symbols: list, start_date: str, end_date: str):
    """Run a single test configuration."""

    # Save original config
    original_config = config_module.DEFAULT_STRATEGY_CONFIG.copy()

    # Apply overrides
    for key, value in config_overrides.items():
        config_module.DEFAULT_STRATEGY_CONFIG[key] = value

    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"Config: {config_overrides}")
    print(f"{'='*60}\n")

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
            verbose=True,
        )

        # Extract metrics
        if results and "metrics" in results:
            m = results["metrics"]
            return {
                "name": test_name,
                "pnl": m.get("total_pnl", 0),
                "trades": m.get("total_trades", 0),
                "win_rate": m.get("win_rate", 0) * 100,
                "max_dd": abs(m.get("max_drawdown", 0)),
                "positions": m.get("positions_count", 0),
            }
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original config
        config_module.DEFAULT_STRATEGY_CONFIG.update(original_config)

    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Combined Filter A/B Test")
    parser.add_argument("--quick", action="store_true", help="Quick test (BTCUSDT only)")
    parser.add_argument("--full-year", action="store_true", help="Full year test")
    args = parser.parse_args()

    symbols = ["BTCUSDT"] if args.quick else ["BTCUSDT", "ETHUSDT", "LINKUSDT"]
    start_date = "2025-01-01" if args.full_year else "2025-06-01"
    end_date = "2025-12-01"

    print("\n" + "="*70)
    print("COMBINED FILTER A/B TEST")
    print("="*70)
    print("\nComparing:")
    print("  BASELINE: Both filters OFF (original behavior)")
    print("  TEST:     SSL Never Lost ON + Confirmation Candle ON")
    print(f"\nSymbols: {symbols}")
    print(f"Period: {start_date} to {end_date}")

    start_time = datetime.now()

    # Test 1: Baseline (both OFF)
    baseline = run_test(
        "BASELINE (Both OFF)",
        {
            "use_ssl_never_lost_filter": False,
            "use_confirmation_candle": False,
        },
        symbols, start_date, end_date
    )

    # Test 2: Combined (both ON)
    combined = run_test(
        "COMBINED (Both ON)",
        {
            "use_ssl_never_lost_filter": True,
            "ssl_never_lost_lookback": 20,
            "use_confirmation_candle": True,
            "confirmation_candle_mode": "close",
        },
        symbols, start_date, end_date
    )

    # Print comparison
    print("\n" + "="*70)
    print("COMBINED FILTER A/B TEST RESULTS")
    print("="*70)

    if baseline and combined:
        print(f"\n{'Metric':<20} {'Baseline':<15} {'Combined':<15} {'Delta':<15}")
        print("-"*65)

        pnl_delta = combined["pnl"] - baseline["pnl"]
        trades_delta = combined["trades"] - baseline["trades"]
        wr_delta = combined["win_rate"] - baseline["win_rate"]
        dd_delta = combined["max_dd"] - baseline["max_dd"]

        pnl_sign = "+" if pnl_delta > 0 else ""
        print(f"{'PnL':<20} ${baseline['pnl']:<14.2f} ${combined['pnl']:<14.2f} {pnl_sign}${pnl_delta:.2f}")

        trades_sign = "+" if trades_delta > 0 else ""
        print(f"{'Trades':<20} {baseline['trades']:<15} {combined['trades']:<15} {trades_sign}{trades_delta}")

        wr_sign = "+" if wr_delta > 0 else ""
        print(f"{'Win Rate':<20} {baseline['win_rate']:<14.1f}% {combined['win_rate']:<14.1f}% {wr_sign}{wr_delta:.1f}%")

        dd_sign = "+" if dd_delta < 0 else ""
        print(f"{'Max Drawdown':<20} ${baseline['max_dd']:<14.2f} ${combined['max_dd']:<14.2f} {dd_sign}${dd_delta:.2f}")

        print("-"*65)

        # Verdict
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)

        improvements = 0
        if pnl_delta > 0:
            print(f"\n✅ PnL improved by ${pnl_delta:.2f}")
            improvements += 1
        elif pnl_delta < 0:
            print(f"\n❌ PnL decreased by ${abs(pnl_delta):.2f}")

        if wr_delta > 0:
            print(f"✅ Win rate improved by {wr_delta:.1f}%")
            improvements += 1
        elif wr_delta < 0:
            print(f"❌ Win rate decreased by {abs(wr_delta):.1f}%")

        if dd_delta < 0:
            print(f"✅ Drawdown reduced by ${abs(dd_delta):.2f}")
            improvements += 1
        elif dd_delta > 0:
            print(f"❌ Drawdown increased by ${dd_delta:.2f}")

        if trades_delta != 0:
            if trades_delta < 0:
                print(f"\n📊 Filtered {abs(trades_delta)} trades (more selective)")
            else:
                print(f"\n📊 Added {trades_delta} trades (found more opportunities)")

        print(f"\n{'='*70}")
        if improvements >= 2:
            print("🎉 COMBINED FILTERS SHOW POSITIVE IMPACT")
        elif improvements == 1:
            print("⚠️  MIXED RESULTS - NEEDS FURTHER ANALYSIS")
        else:
            print("❌ COMBINED FILTERS DID NOT IMPROVE PERFORMANCE")
        print("="*70)

    elapsed = datetime.now() - start_time
    print(f"\nTotal test time: {elapsed}")


if __name__ == "__main__":
    main()
