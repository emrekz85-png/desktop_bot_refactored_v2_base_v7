#!/usr/bin/env python3
"""
Partial TP Parameter Comparison Test

Tests different partial_trigger and partial_fraction combinations
to find optimal values for WEEKLY rolling walk-forward.

Usage: ./venv/bin/python run_partial_tp_comparison.py

SECURITY NOTES:
- All test configurations are hardcoded (no user input)
- Subprocess uses list form (no shell injection risk)
- Input validation ensures values are within safe ranges
"""

import sys
import os
import subprocess
import re

# Test configurations to compare
TEST_CONFIGS = [
    {"name": "A", "trigger": 0.50, "fraction": 0.40, "desc": "Orta trigger, orta fraction"},
    {"name": "B", "trigger": 0.55, "fraction": 0.35, "desc": "Orta-geç trigger, küçük fraction"},
    {"name": "C", "trigger": 0.65, "fraction": 0.33, "desc": "Geç trigger, küçük fraction (eski)"},
]

# SECURITY: Use relative path from script directory instead of hardcoded absolute path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "core", "config.py")


def _validate_config_value(value: float, name: str, min_val: float = 0.0, max_val: float = 1.0) -> bool:
    """
    Validate that a config value is a float within expected range.

    SECURITY: Prevents injection of invalid values into config files.
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value)}")
    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return True


def update_baseline_config(trigger: float, fraction: float):
    """Update BASELINE_CONFIG in config.py with new partial TP values.

    SECURITY: Validates input values before writing to config file.
    """
    # SECURITY: Validate inputs before modifying config
    _validate_config_value(trigger, "trigger", 0.1, 0.99)
    _validate_config_value(fraction, "fraction", 0.1, 0.99)

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update partial_trigger (using precise float formatting)
    content = re.sub(
        r'("partial_trigger":\s*)[\d.]+',
        f'\\g<1>{trigger:.2f}',
        content
    )

    # Update partial_fraction (using precise float formatting)
    content = re.sub(
        r'("partial_fraction":\s*)[\d.]+',
        f'\\g<1>{fraction:.2f}',
        content
    )

    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"   Updated BASELINE_CONFIG: trigger={trigger}, fraction={fraction}")


def run_weekly_test():
    """Run rolling WF test with WEEKLY mode only and return PnL.

    SECURITY: Uses list-form subprocess call (no shell injection).
    Paths are derived from script location, not hardcoded.
    """
    # SECURITY: Build Python code with escaped path
    python_code = f"""
import sys
sys.path.insert(0, {repr(SCRIPT_DIR)})
from desktop_bot_refactored_v2_base_v7 import run_rolling_wf_comparison_optimized

result = run_rolling_wf_comparison_optimized(
    parallel=True,
    quick=True,
    modes=["WEEKLY"],
)

if result and "WEEKLY" in result:
    w = result["WEEKLY"]
    print(f"RESULT_PNL={{w.get('total_pnl', 0):.2f}}")
    print(f"RESULT_TRADES={{w.get('total_trades', 0)}}")
    print(f"RESULT_MAX_DD={{w.get('max_dd', 0):.2f}}")
else:
    print("RESULT_PNL=0.00")
    print("RESULT_TRADES=0")
    print("RESULT_MAX_DD=0.00")
"""

    # SECURITY: Use list form (not shell=True) to prevent shell injection
    cmd = [sys.executable, "-c", python_code]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
            cwd=SCRIPT_DIR  # Use script directory instead of hardcoded path
        )

        output = result.stdout + result.stderr

        # Parse results
        pnl_match = re.search(r'RESULT_PNL=([-\d.]+)', output)
        trades_match = re.search(r'RESULT_TRADES=(\d+)', output)
        dd_match = re.search(r'RESULT_MAX_DD=([-\d.]+)', output)

        pnl = float(pnl_match.group(1)) if pnl_match else 0.0
        trades = int(trades_match.group(1)) if trades_match else 0
        max_dd = float(dd_match.group(1)) if dd_match else 0.0

        return {"pnl": pnl, "trades": trades, "max_dd": max_dd}

    except subprocess.TimeoutExpired:
        print("   TIMEOUT!")
        return {"pnl": 0, "trades": 0, "max_dd": 0, "error": "timeout"}
    except Exception as e:
        print(f"   ERROR: {e}")
        return {"pnl": 0, "trades": 0, "max_dd": 0, "error": str(e)}


def main():
    print("\n" + "=" * 70)
    print("PARTIAL TP PARAMETER COMPARISON TEST")
    print("=" * 70)
    print("\nTest Configurations:")
    for cfg in TEST_CONFIGS:
        print(f"  {cfg['name']}: trigger={cfg['trigger']}, fraction={cfg['fraction']} ({cfg['desc']})")
    print("\n" + "=" * 70)

    # Store original values to restore later
    with open(CONFIG_FILE, 'r') as f:
        original_content = f.read()

    results = {}

    try:
        for test_cfg in TEST_CONFIGS:
            print(f"\n{'=' * 70}")
            print(f"TEST {test_cfg['name']}: trigger={test_cfg['trigger']}, fraction={test_cfg['fraction']}")
            print(f"{'=' * 70}")

            # Update config
            update_baseline_config(test_cfg['trigger'], test_cfg['fraction'])

            # Run test
            print("   Running WEEKLY rolling WF test (quick mode)...")
            result = run_weekly_test()

            results[test_cfg['name']] = {
                **test_cfg,
                **result
            }

            print(f"   Result: PnL=${result.get('pnl', 0):.2f}, Trades={result.get('trades', 0)}")

    finally:
        # Restore original config
        with open(CONFIG_FILE, 'w') as f:
            f.write(original_content)
        print("\n   Original config restored.")

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Test':<6} {'Trigger':<10} {'Fraction':<10} {'PnL':<15} {'Trades':<10} {'Max DD':<12}")
    print("-" * 70)

    best_test = None
    best_pnl = float('-inf')

    for name in ["A", "B", "C"]:
        r = results.get(name, {})
        if "error" in r:
            print(f"{name:<6} {r.get('trigger', 0):<10.2f} {r.get('fraction', 0):<10.2f} ERROR: {r.get('error', '')}")
        else:
            pnl = r.get("pnl", 0)
            print(f"{name:<6} {r.get('trigger', 0):<10.2f} {r.get('fraction', 0):<10.2f} ${pnl:<14.2f} {r.get('trades', 0):<10} ${r.get('max_dd', 0):<11.2f}")

            if pnl > best_pnl:
                best_pnl = pnl
                best_test = name

    print("-" * 70)

    if best_test:
        best = results[best_test]
        print(f"\nEN IYI: Test {best_test}")
        print(f"  partial_trigger = {best['trigger']}")
        print(f"  partial_fraction = {best['fraction']}")
        print(f"  PnL = ${best['pnl']:.2f}")

        print(f"\nBu degerleri BASELINE_CONFIG'e uygulamak icin:")
        print(f'  BASELINE_CONFIG["partial_trigger"] = {best["trigger"]}')
        print(f'  BASELINE_CONFIG["partial_fraction"] = {best["fraction"]}')

    print("\n" + "=" * 70)
    print("TEST TAMAMLANDI")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    main()
