#!/usr/bin/env python3
"""
Improvement Comparison Test - Simple Version

Runs BASELINE test with all v46.x improvements disabled.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Temporarily patch config BEFORE importing other modules
import core.config as config_module

# Store original values
ORIGINAL_VALUES = {
    "use_ssl_never_lost_filter": config_module.DEFAULT_STRATEGY_CONFIG.get("use_ssl_never_lost_filter", True),
    "be_atr_multiplier": config_module.DEFAULT_STRATEGY_CONFIG.get("be_atr_multiplier", 0.5),
    "use_time_invalidation": config_module.DEFAULT_STRATEGY_CONFIG.get("use_time_invalidation", True),
    "use_smart_reentry": config_module.DEFAULT_STRATEGY_CONFIG.get("use_smart_reentry", True),
}

def set_baseline_config():
    """Disable all v46.x improvements"""
    config_module.DEFAULT_STRATEGY_CONFIG["use_ssl_never_lost_filter"] = False
    config_module.DEFAULT_STRATEGY_CONFIG["be_atr_multiplier"] = 0.0  # Disable ATR buffer
    config_module.DEFAULT_STRATEGY_CONFIG["use_time_invalidation"] = False
    config_module.DEFAULT_STRATEGY_CONFIG["use_smart_reentry"] = False
    print("Config set to BASELINE (all improvements OFF)")

def restore_config():
    """Restore original config values"""
    for key, value in ORIGINAL_VALUES.items():
        config_module.DEFAULT_STRATEGY_CONFIG[key] = value
    print("Config restored to CURRENT (all improvements ON)")

def run_test(label: str):
    """Run a single rolling WF test"""
    from runners.rolling_wf_optimized import run_rolling_walkforward_optimized

    print(f"\n{'='*60}")
    print(f"Running {label} test...")
    print(f"{'='*60}")

    result = run_rolling_walkforward_optimized(
        symbols=["BTCUSDT", "ETHUSDT", "LINKUSDT"],
        timeframes=["15m", "30m", "1h"],
        start_date="2025-01-01",
        end_date="2025-12-01",
        mode="weekly",
        lookback_days=60,
        forward_days=7,
        run_id=f"{label.lower()}_comparison",
    )

    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Run BASELINE test (improvements OFF)")
    parser.add_argument("--current", action="store_true", help="Run CURRENT test (improvements ON)")
    args = parser.parse_args()

    if args.baseline:
        set_baseline_config()
        result = run_test("BASELINE")
        print(f"\nBASELINE Results:")
        print(f"  PnL: ${result.get('total_pnl', 0):.2f}")
        print(f"  Trades: {result.get('total_positions', 0)}")
    elif args.current:
        restore_config()
        result = run_test("CURRENT")
        print(f"\nCURRENT Results:")
        print(f"  PnL: ${result.get('total_pnl', 0):.2f}")
        print(f"  Trades: {result.get('total_positions', 0)}")
    else:
        print("Usage: python run_improvement_comparison.py --baseline | --current")
        print("\nRun --baseline first, then --current to compare")
