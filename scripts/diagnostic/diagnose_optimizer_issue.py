#!/usr/bin/env python3
"""
Diagnostic script to investigate why optimizer returns 0 configs in rolling WF tests.

This script tests:
1. Config generation (are configs being created?)
2. Config serialization (can configs be pickled for multiprocessing?)
3. Trade simulation (does SimTradeManager work with new params?)
4. Scoring function (does _score_config_for_stream return valid results?)
"""

import sys
import os
import json
import pickle  # Only used for diagnostic testing (trusted data only)
from pprint import pprint

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# SECURITY NOTE: This script uses pickle for diagnostic testing only.
# Pickle is safe here because:
# 1. This is a diagnostic script, not production code
# 2. We only pickle/unpickle data we create ourselves (trusted)
# 3. No user input or external data is unpickled
# For production code, prefer JSON or other safe serialization formats.

from core.optimizer import (
    _generate_quick_candidate_configs,
    _generate_candidate_configs,
    _score_config_for_stream,
)
from core.config import TRADING_CONFIG, DEFAULT_STRATEGY_CONFIG, BASELINE_CONFIG
from core.trade_manager import SimTradeManager
from core.trading_engine import TradingEngine

print("="*80)
print("OPTIMIZER DIAGNOSTIC SCRIPT")
print("="*80)

# ============================================================================
# TEST 1: Config Generation
# ============================================================================
print("\n[TEST 1] Generating candidate configs...")
try:
    quick_configs = _generate_quick_candidate_configs()
    full_configs = _generate_candidate_configs()

    print(f"✓ Quick configs generated: {len(quick_configs)} configs")
    print(f"✓ Full configs generated: {len(full_configs)} configs")

    # Show first config
    print("\nFirst quick config:")
    pprint(quick_configs[0])

    # Check for new parameters
    sample_config = quick_configs[0]
    has_momentum = "momentum_tp_extension" in sample_config
    has_progressive = "use_progressive_partial" in sample_config
    has_tranches = "partial_tranches" in sample_config

    print(f"\n✓ Has momentum_tp_extension: {has_momentum}")
    print(f"✓ Has use_progressive_partial: {has_progressive}")
    print(f"✓ Has partial_tranches: {has_tranches}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 2: Config Serialization (Multiprocessing Compatibility)
# ============================================================================
print("\n[TEST 2] Testing config serialization for multiprocessing...")
try:
    test_config = quick_configs[0]

    # SECURITY: Test pickle serialization (DIAGNOSTIC ONLY - trusted data)
    # This tests if configs can be serialized for ProcessPoolExecutor
    # WARNING: Never pickle untrusted data - prefer JSON for production
    pickled = pickle.dumps(test_config)
    unpickled = pickle.loads(pickled)

    print(f"✓ Config can be pickled (size: {len(pickled)} bytes)")

    # Check if partial_tranches survived
    if "partial_tranches" in test_config:
        original_tranches = test_config["partial_tranches"]
        unpickled_tranches = unpickled["partial_tranches"]

        print(f"✓ Original tranches: {original_tranches}")
        print(f"✓ Unpickled tranches: {unpickled_tranches}")

        if original_tranches == unpickled_tranches:
            print("✓ Tranches survived serialization")
        else:
            print("✗ WARNING: Tranches changed during serialization!")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠️  Config serialization failed - multiprocessing will fail!")
    sys.exit(1)

# ============================================================================
# TEST 3: SimTradeManager with New Params
# ============================================================================
print("\n[TEST 3] Testing SimTradeManager with new parameters...")
try:
    tm = SimTradeManager(initial_balance=TRADING_CONFIG["initial_balance"])

    # Create a test trade with new params
    from datetime import datetime
    test_trade_data = {
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "type": "LONG",
        "setup": "TEST_SIGNAL",
        "entry": 100000.0,
        "tp": 101000.0,
        "sl": 99500.0,
        "timestamp": "2025-01-01 00:00",
        "open_time_utc": datetime(2025, 1, 1, 0, 0, 0),  # Use datetime object instead of string
        "config_snapshot": {
            **DEFAULT_STRATEGY_CONFIG,
            "momentum_tp_extension": True,
            "momentum_extension_threshold": 0.80,
            "momentum_extension_multiplier": 1.5,
            "use_progressive_partial": True,
            "partial_tranches": [
                {"trigger": 0.40, "fraction": 0.33},
                {"trigger": 0.70, "fraction": 0.50},
            ],
            "progressive_be_after_tranche": 1,
        }
    }

    success = tm.open_trade(test_trade_data)

    if success:
        print(f"✓ Trade opened successfully")
        print(f"  Open trades: {len(tm.open_trades)}")

        # Check if new params were stored
        opened_trade = tm.open_trades[0]
        print(f"  momentum_tp_extension: {opened_trade.get('momentum_tp_extension')}")
        print(f"  use_progressive_partial: {opened_trade.get('use_progressive_partial')}")
        print(f"  partial_tranches: {opened_trade.get('partial_tranches')}")

        # Check type of partial_tranches
        tranches = opened_trade.get('partial_tranches')
        if tranches is not None:
            print(f"  tranches type: {type(tranches)}")
            print(f"  tranches value: {tranches}")
    else:
        print(f"✗ Trade failed to open")
        print(f"  Wallet balance: {tm.wallet_balance}")
        print(f"  Circuit breaker killed: {tm.is_stream_killed('BTCUSDT', '15m')}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 4: Fetch Real Data and Test Scoring
# ============================================================================
print("\n[TEST 4] Testing _score_config_for_stream with real data...")
try:
    # Fetch small sample of real data
    print("  Fetching BTC 15m data (1000 candles)...")

    from core.binance_client import BinanceClient
    from core.indicators import calculate_indicators

    client = BinanceClient()
    df = client.get_klines("BTCUSDT", "15m", limit=1000)

    print(f"✓ Fetched {len(df)} candles")

    # Calculate indicators
    print("  Calculating indicators...")
    calculate_indicators(df)  # Only takes df as argument
    print(f"✓ Indicators calculated")

    # Test scoring with first quick config
    print("  Running _score_config_for_stream...")
    test_config = quick_configs[0]

    # IMPORTANT: Add new params to test config
    test_config["momentum_tp_extension"] = True
    test_config["momentum_extension_threshold"] = 0.80
    test_config["momentum_extension_multiplier"] = 1.5
    test_config["use_progressive_partial"] = True
    test_config["partial_tranches"] = [
        {"trigger": 0.40, "fraction": 0.33},
        {"trigger": 0.70, "fraction": 0.50},
    ]
    test_config["progressive_be_after_tranche"] = 1

    print(f"\n  Test config:")
    print(f"    rr: {test_config['rr']}")
    print(f"    rsi: {test_config['rsi']}")
    print(f"    at_active: {test_config['at_active']}")
    print(f"    momentum_tp_extension: {test_config.get('momentum_tp_extension')}")
    print(f"    use_progressive_partial: {test_config.get('use_progressive_partial')}")
    print(f"    partial_tranches: {test_config.get('partial_tranches')}")

    net_pnl, trades, trade_pnls, trade_r_multiples = _score_config_for_stream(
        df, "BTCUSDT", "15m", test_config
    )

    print(f"\n✓ Scoring completed:")
    print(f"  Net PnL: ${net_pnl:.2f}")
    print(f"  Trades: {trades}")
    print(f"  Trade PnLs: {trade_pnls[:5] if trade_pnls else []}")
    print(f"  R-Multiples: {trade_r_multiples[:5] if trade_r_multiples else []}")

    if trades == 0:
        print("\n⚠️  WARNING: 0 trades generated!")
        print("  This could mean:")
        print("  1. Signal detection is too strict")
        print("  2. Data is insufficient (need warmup period)")
        print("  3. Config parameters are invalid")
        print("  4. Exception is being silently caught")
    else:
        print(f"\n✓ Generated {trades} trades - scoring is working!")

        # Calculate expected R
        if trade_r_multiples:
            expected_r = sum(trade_r_multiples) / len(trade_r_multiples)
            print(f"  Expected R: {expected_r:.3f}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 5: Check Config Generator Defaults
# ============================================================================
print("\n[TEST 5] Checking if config generators include new params...")
try:
    # Check quick config generator
    print("\nChecking _generate_quick_candidate_configs source...")

    # Read the source to see if new params are included
    import inspect
    source = inspect.getsource(_generate_quick_candidate_configs)

    has_momentum_in_source = "momentum_tp_extension" in source
    has_progressive_in_source = "use_progressive_partial" in source
    has_tranches_in_source = "partial_tranches" in source

    print(f"  momentum_tp_extension in source: {has_momentum_in_source}")
    print(f"  use_progressive_partial in source: {has_progressive_in_source}")
    print(f"  partial_tranches in source: {has_tranches_in_source}")

    if not (has_momentum_in_source or has_progressive_in_source or has_tranches_in_source):
        print("\n⚠️  WARNING: New parameters NOT found in config generator!")
        print("  Config generator may need to be updated to include:")
        print("    - momentum_tp_extension")
        print("    - use_progressive_partial")
        print("    - partial_tranches")
    else:
        print("\n✓ Config generator includes new parameters")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)
print("\nIf all tests passed:")
print("  1. Config generation: OK")
print("  2. Serialization: OK")
print("  3. SimTradeManager: OK")
print("  4. Scoring function: OK")
print("  5. Config defaults: OK")
print("\nNext steps:")
print("  - Run a minimal rolling WF test with verbose logging")
print("  - Check for exceptions in optimizer parallel execution")
print("  - Verify that configs are reaching _score_config_for_stream")
print("\nIf Test 4 returned 0 trades:")
print("  - Check signal detection logic")
print("  - Verify data has required indicators")
print("  - Check if circuit breaker is rejecting trades")
print("="*80)
