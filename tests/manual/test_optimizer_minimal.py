#!/usr/bin/env python3
"""
Minimal test to verify optimizer returns configs.

This is a stripped-down version of the rolling WF test to isolate the optimizer.
"""

import sys
from datetime import datetime

from test_helpers import get_binance_client
from core.optimizer import _optimize_backtest_configs
from core.indicators import calculate_indicators

print("="*80)
print("MINIMAL OPTIMIZER TEST")
print("="*80)

# Fetch data for one stream
print("\n[1] Fetching data for BTCUSDT 15m...")
client = get_binance_client()
df = client.get_klines("BTCUSDT", "15m", limit=5000)
print(f"✓ Fetched {len(df)} candles")

# Calculate indicators
print("\n[2] Calculating indicators...")
calculate_indicators(df)
print("✓ Indicators calculated")

# Create streams dict
streams = {
    ("BTCUSDT", "15m"): df,
}

requested_pairs = [("BTCUSDT", "15m")]

# Run optimizer
print("\n[3] Running optimizer...")
print("    This should find at least 1 config if everything works correctly.")
print("    Quick mode uses 6-7 configs, so there should be trades.\n")

best_configs = _optimize_backtest_configs(
    streams=streams,
    requested_pairs=requested_pairs,
    progress_callback=None,
    log_to_stdout=True,
    use_walk_forward=False,  # Disable WF for simplicity
    quick_mode=True,  # Use quick mode (fewer configs)
)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

if not best_configs:
    print("✗ FAILED: Optimizer returned empty dict!")
    print("  This means _optimize_backtest_configs returned no configs.")
    sys.exit(1)

for (sym, tf), config in best_configs.items():
    if config.get("disabled"):
        reason = config.get("_reason", "unknown")
        print(f"\n[{sym}-{tf}] DISABLED")
        print(f"  Reason: {reason}")
    else:
        print(f"\n[{sym}-{tf}] ENABLED")
        print(f"  RR: {config.get('rr')}")
        print(f"  RSI: {config.get('rsi')}")
        print(f"  Net PnL: ${config.get('_net_pnl', 0):.2f}")
        print(f"  Trades: {config.get('_trades', 0)}")
        print(f"  Score: {config.get('_score', 0):.2f}")
        print(f"  E[R]: {config.get('_expected_r', 0):.3f}")
        print(f"  Confidence: {config.get('confidence', 'unknown')}")
        print(f"  Progressive Partial: {config.get('use_progressive_partial', False)}")
        print(f"  Partial Tranches: {config.get('partial_tranches', [])}")

if all(c.get("disabled") for c in best_configs.values()):
    print("\n⚠️  All configs disabled - this is the issue!")
    print("   Possible causes:")
    print("   1. Not enough trades generated (< minimum threshold)")
    print("   2. E[R] too low (below MIN_EXPECTANCY_R_MULTIPLE)")
    print("   3. Score too low (below MIN_SCORE_THRESHOLD)")
    print("   4. PnL negative")
else:
    print("\n✓ SUCCESS: At least one config enabled!")

print("="*80)
