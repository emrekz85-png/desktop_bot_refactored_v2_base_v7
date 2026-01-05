#!/usr/bin/env python3
"""
HTF Signal Diagnostic Tool

Bu script 1h ve 4h timeframe'lerde sinyal uretimini analiz eder.
Hangi filter'in sinyalleri blokladigini tespit eder.

Kullanim:
    python diagnose_htf_signals.py
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import Counter

# Add project root to path
sys.path.insert(0, '.')

from core import (
    SYMBOLS, calculate_indicators, TRADING_CONFIG,
    get_tf_threshold, get_tf_thresholds,
    DEFAULT_STRATEGY_CONFIG
)
from core.binance_client import get_client
from strategies.ssl_flow import check_ssl_flow_signal

# Get client instance
client = get_client()


def analyze_filter_pass_rates(df: pd.DataFrame, timeframe: str, config: dict) -> Dict:
    """
    Her filter icin pass rate hesapla.
    """
    results = {
        "total_candles": 0,
        "filters": {},
        "block_reasons": Counter(),
        "signal_count": {"LONG": 0, "SHORT": 0, "NONE": 0}
    }

    # TF-adaptive thresholds
    tf_thresholds = get_tf_thresholds(timeframe)
    print(f"\n  TF-Adaptive Thresholds for {timeframe}:")
    for k, v in tf_thresholds.items():
        print(f"    {k}: {v}")

    # Analyze each candle
    start_idx = 100  # Skip warmup
    end_idx = len(df) - 1

    filter_passes = {
        "price_above_baseline": 0,
        "price_below_baseline": 0,
        "at_buyers_dominant": 0,
        "at_sellers_dominant": 0,
        "at_is_flat": 0,  # Count how many are flagged as flat
        "baseline_touch_long": 0,
        "baseline_touch_short": 0,
        "pbema_distance_long_ok": 0,
        "pbema_distance_short_ok": 0,
        "regime_trending": 0,
        "wick_rejection_long": 0,
        "wick_rejection_short": 0,
    }

    total = 0

    for i in range(start_idx, end_idx):
        total += 1

        # Get signal with debug info
        try:
            signal_type, entry, tp, sl, reason, debug = check_ssl_flow_signal(
                df,
                index=i,
                min_rr=config.get("rr", 2.0),
                rsi_limit=config.get("rsi", 70.0),
                skip_wick_rejection=config.get("skip_wick_rejection", True),
                skip_body_position=config.get("skip_body_position", False),
                skip_adx_filter=config.get("skip_adx_filter", False),
                skip_at_flat_filter=config.get("skip_at_flat_filter", False),
                timeframe=timeframe,
                return_debug=True
            )
        except Exception as e:
            continue

        # Count filter passes
        if debug.get("price_above_baseline"):
            filter_passes["price_above_baseline"] += 1
        if debug.get("price_below_baseline"):
            filter_passes["price_below_baseline"] += 1
        if debug.get("at_buyers_dominant"):
            filter_passes["at_buyers_dominant"] += 1
        if debug.get("at_sellers_dominant"):
            filter_passes["at_sellers_dominant"] += 1
        if debug.get("at_is_flat"):
            filter_passes["at_is_flat"] += 1  # This is BAD - means flat detected
        if debug.get("baseline_touch_long"):
            filter_passes["baseline_touch_long"] += 1
        if debug.get("baseline_touch_short"):
            filter_passes["baseline_touch_short"] += 1
        if debug.get("long_pbema_distance", 0) >= tf_thresholds.get("min_pbema_distance", 0.003):
            filter_passes["pbema_distance_long_ok"] += 1
        if debug.get("short_pbema_distance", 0) >= tf_thresholds.get("min_pbema_distance", 0.003):
            filter_passes["pbema_distance_short_ok"] += 1
        if debug.get("regime") == "TRENDING":
            filter_passes["regime_trending"] += 1
        if debug.get("long_rejection"):
            filter_passes["wick_rejection_long"] += 1
        if debug.get("short_rejection"):
            filter_passes["wick_rejection_short"] += 1

        # Count signals and block reasons
        if signal_type == "LONG":
            results["signal_count"]["LONG"] += 1
        elif signal_type == "SHORT":
            results["signal_count"]["SHORT"] += 1
        else:
            results["signal_count"]["NONE"] += 1
            results["block_reasons"][reason] += 1

    results["total_candles"] = total
    results["filters"] = filter_passes

    # Calculate pass rates
    results["pass_rates"] = {}
    for filter_name, count in filter_passes.items():
        rate = (count / total * 100) if total > 0 else 0
        results["pass_rates"][filter_name] = rate

    return results


def print_analysis(symbol: str, timeframe: str, results: Dict):
    """
    Analiz sonuclarini yazdir.
    """
    print(f"\n{'='*60}")
    print(f"  {symbol} - {timeframe}")
    print(f"{'='*60}")
    print(f"  Total Candles Analyzed: {results['total_candles']}")
    print(f"  Signals Found: LONG={results['signal_count']['LONG']}, SHORT={results['signal_count']['SHORT']}")
    print(f"  No Signal: {results['signal_count']['NONE']}")

    print(f"\n  Filter Pass Rates:")
    print(f"  {'-'*50}")

    # Sort by pass rate
    sorted_rates = sorted(results["pass_rates"].items(), key=lambda x: x[1])

    for filter_name, rate in sorted_rates:
        bar = "#" * int(rate / 5)  # 20 chars = 100%
        emoji = "LOW" if rate < 30 else "MED" if rate < 70 else "OK"
        print(f"    {filter_name:30} | {rate:5.1f}% | {bar:20} | {emoji}")

    print(f"\n  Top Block Reasons:")
    print(f"  {'-'*50}")
    for reason, count in results["block_reasons"].most_common(10):
        pct = count / results["total_candles"] * 100
        print(f"    {reason:40} | {count:5} ({pct:5.1f}%)")


def main():
    print("="*60)
    print(" HTF SIGNAL DIAGNOSTIC TOOL")
    print(" Analyzing why bot produces ZERO signals on 1h/4h")
    print("="*60)

    # Config
    config = DEFAULT_STRATEGY_CONFIG.copy()
    config["skip_wick_rejection"] = True  # v2.0.0 change

    # Test multiple timeframes
    timeframes = ["15m", "1h", "4h"]
    symbols = ["BTCUSDT"]

    all_results = {}

    for symbol in symbols:
        for tf in timeframes:
            print(f"\n>>> Fetching {symbol} {tf} data...")

            try:
                # Fetch recent data (max 1000 per request)
                df = client.get_klines(symbol, tf, limit=1000)

                print(f"  Fetched {len(df) if df is not None else 0} candles")

                if df is None or len(df) < 150:
                    print(f"  ERROR: Not enough data for {symbol} {tf} (got {len(df) if df is not None else 0})")
                    continue

                print(f"  Calculating indicators...")
                df = calculate_indicators(df, timeframe=tf)

                print(f"  Analyzing filters...")
                results = analyze_filter_pass_rates(df, tf, config)

                all_results[(symbol, tf)] = results
                print_analysis(symbol, tf, results)

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary comparison
    print("\n" + "="*60)
    print(" SUMMARY: Pass Rate Comparison Across Timeframes")
    print("="*60)

    key_filters = [
        "at_is_flat",
        "regime_trending",
        "baseline_touch_long",
        "at_buyers_dominant",
        "pbema_distance_long_ok"
    ]

    print(f"\n{'Filter':<30} | {'15m':>8} | {'1h':>8} | {'4h':>8}")
    print("-" * 60)

    for f in key_filters:
        row = f"{f:<30} |"
        for tf in timeframes:
            key = ("BTCUSDT", tf)
            if key in all_results:
                rate = all_results[key]["pass_rates"].get(f, 0)
                row += f" {rate:7.1f}% |"
            else:
                row += "     N/A |"
        print(row)

    # Signal count comparison
    print(f"\n{'Signals':<30} | {'15m':>8} | {'1h':>8} | {'4h':>8}")
    print("-" * 60)

    for sig_type in ["LONG", "SHORT", "NONE"]:
        row = f"{sig_type:<30} |"
        for tf in timeframes:
            key = ("BTCUSDT", tf)
            if key in all_results:
                count = all_results[key]["signal_count"].get(sig_type, 0)
                row += f" {count:8} |"
            else:
                row += "     N/A |"
        print(row)

    print("\n" + "="*60)
    print(" ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
