#!/usr/bin/env python3
"""
Diagnose Signal Generation Issues

This script bypasses the optimizer entirely and just counts:
1. How many signals are generated with current filter settings
2. Which filters are blocking the most signals
"""

import sys
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, "/Users/emreoksuz/desktop_bot_refactored_v2_base_v7")

from core.config import DEFAULT_STRATEGY_CONFIG, TRADING_CONFIG
from core.indicators import calculate_indicators
from core.binance_client import BinanceClient
from strategies.ssl_flow import check_ssl_flow_signal as ssl_flow_signal

def fetch_data(symbol: str, timeframe: str, days: int = 60):
    """Fetch historical data from Binance."""
    client = BinanceClient()
    candles = days * 24 * 4 if timeframe == "15m" else days * 24  # Rough estimate
    df = client.get_klines(symbol, timeframe, limit=min(candles, 1500))
    return df

def count_signals(df: pd.DataFrame, config: dict):
    """
    Count signals and filter rejection reasons.
    """
    if df is None or len(df) < 300:
        print(f"   Insufficient data: {len(df) if df is not None else 0} bars")
        return {}

    # Calculate indicators
    df = calculate_indicators(df, timeframe="15m")

    rejection_counts = {}
    signal_counts = {"LONG": 0, "SHORT": 0, "TOTAL": 0}

    warmup = 250
    for i in range(warmup, len(df) - 1):
        # Call with individual parameters matching the function signature
        result = ssl_flow_signal(
            df=df,
            index=i,
            min_rr=config.get("rr", 1.5),
            rsi_limit=config.get("rsi", 65),
            min_pbema_distance=config.get("min_pbema_distance", 0.002),
            lookback_candles=config.get("lookback_candles", 10),
            skip_body_position=config.get("skip_body_position", True),
            skip_adx_filter=config.get("skip_adx_filter", True),
            use_ssl_never_lost_filter=config.get("use_ssl_never_lost_filter", False),
            skip_overlap_check=config.get("skip_overlap_check", True),
            skip_at_flat_filter=config.get("skip_at_flat_filter", True),
            regime_adx_threshold=config.get("regime_adx_threshold", 15.0),
            return_debug=True
        )

        if len(result) == 6:  # With debug
            signal_type, entry, tp, sl, reason, debug = result
        else:
            signal_type, entry, tp, sl, reason = result
            debug = {}

        if signal_type:
            signal_counts[signal_type] += 1
            signal_counts["TOTAL"] += 1
        else:
            # Count rejection reasons
            if reason:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    return {
        "signals": signal_counts,
        "rejections": rejection_counts,
        "bars_tested": len(df) - warmup - 1
    }

def main():
    print("=" * 70)
    print("🔍 SIGNAL GENERATION DIAGNOSTIC")
    print("=" * 70)

    # Print current config
    from core.config import ALPHATREND_CONFIG
    print("\n📋 Current Filter Settings:")
    print(f"   skip_adx_filter: {DEFAULT_STRATEGY_CONFIG.get('skip_adx_filter', False)}")
    print(f"   skip_body_position: {DEFAULT_STRATEGY_CONFIG.get('skip_body_position', False)}")
    print(f"   skip_overlap_check: {DEFAULT_STRATEGY_CONFIG.get('skip_overlap_check', False)}")
    print(f"   skip_at_flat_filter: {DEFAULT_STRATEGY_CONFIG.get('skip_at_flat_filter', False)}")
    print(f"   use_ssl_never_lost_filter: {DEFAULT_STRATEGY_CONFIG.get('use_ssl_never_lost_filter', True)}")
    print(f"   min_pbema_distance: {DEFAULT_STRATEGY_CONFIG.get('min_pbema_distance', 0.004)}")
    print(f"   lookback_candles: {DEFAULT_STRATEGY_CONFIG.get('lookback_candles', 5)}")
    print(f"   regime_adx_threshold: {DEFAULT_STRATEGY_CONFIG.get('regime_adx_threshold', 20.0)}")
    print(f"   flat_threshold (AlphaTrend): {ALPHATREND_CONFIG.get('flat_threshold', 0.001)}")

    # Test symbols
    symbols = ["BTCUSDT", "ETHUSDT", "LINKUSDT"]

    for symbol in symbols:
        print(f"\n{'─' * 60}")
        print(f"📊 {symbol} (15m)")
        print(f"{'─' * 60}")

        try:
            df = fetch_data(symbol, "15m", days=60)
            if df is None or len(df) == 0:
                print("   ❌ Failed to fetch data")
                continue

            print(f"   Fetched {len(df)} bars")

            # Test with current config
            config = dict(DEFAULT_STRATEGY_CONFIG)
            config["rr"] = 1.5
            config["rsi"] = 65
            config["at_active"] = True

            result = count_signals(df, config)

            if not result:
                continue

            # Print results
            print(f"\n   ✅ SIGNALS GENERATED:")
            print(f"      LONG: {result['signals']['LONG']}")
            print(f"      SHORT: {result['signals']['SHORT']}")
            print(f"      TOTAL: {result['signals']['TOTAL']}")
            print(f"      Bars tested: {result['bars_tested']}")

            if result['signals']['TOTAL'] > 0:
                rate = result['signals']['TOTAL'] / result['bars_tested'] * 100
                print(f"      Signal Rate: {rate:.2f}%")

            # Print top rejections
            if result['rejections']:
                print(f"\n   ❌ TOP REJECTION REASONS:")
                sorted_rejections = sorted(
                    result['rejections'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                for reason, count in sorted_rejections:
                    pct = count / result['bars_tested'] * 100
                    print(f"      {count:5d} ({pct:5.1f}%) - {reason}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
