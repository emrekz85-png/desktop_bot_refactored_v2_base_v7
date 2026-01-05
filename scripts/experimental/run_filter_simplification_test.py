#!/usr/bin/env python3
"""
Filter Simplification Backtest - Priority 3 Implementation

This script tests the impact of removing low-value filters from the SSL Flow strategy
based on the hedge fund due diligence report findings.

Filters to test:
- body_position: 99.9% pass rate -> effectively useless
- wick_rejection: 68.8% pass rate -> test removal impact

Usage:
    python run_filter_simplification_test.py              # Standard test
    python run_filter_simplification_test.py --quick      # Quick test (fewer candles)
    python run_filter_simplification_test.py --full-year  # Full year test
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    calculate_indicators, get_client, DATA_DIR,
    SimTradeManager, TRADING_CONFIG,
)
from strategies.ssl_flow import check_ssl_flow_signal


def run_single_stream_backtest(
    df,
    symbol: str,
    timeframe: str,
    skip_body_position: bool = False,
    skip_wick_rejection: bool = False,
    config: dict = None,
) -> Dict:
    """Run backtest on a single stream with filter configuration."""

    if config is None:
        config = {
            "min_rr": 2.0,
            "rsi_limit": 70.0,
            "regime_adx_threshold": 20.0,
        }

    trades = []
    signals_checked = 0
    filter_blocks = defaultdict(int)

    # Simulate through the data
    for i in range(100, len(df) - 1):  # Start after warmup, leave room for trade execution
        signals_checked += 1

        # Check for signal with filter configuration
        result = check_ssl_flow_signal(
            df,
            index=i,
            min_rr=config.get("min_rr", 2.0),
            rsi_limit=config.get("rsi_limit", 70.0),
            regime_adx_threshold=config.get("regime_adx_threshold", 20.0),
            skip_body_position=skip_body_position,
            skip_wick_rejection=skip_wick_rejection,
            return_debug=True,
        )

        signal_type = result[0]
        entry = result[1]
        tp = result[2]
        sl = result[3]
        reason = result[4]

        # Track filter blocks
        if signal_type is None and reason:
            filter_blocks[reason] += 1

        if signal_type and entry and tp and sl:
            # Simulate trade execution
            trade_result = simulate_trade(df, i, signal_type, entry, tp, sl, symbol, timeframe)
            if trade_result:
                trades.append(trade_result)

    # Calculate metrics
    total_pnl = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    total_trades = len(trades)
    win_rate = wins / total_trades if total_trades > 0 else 0

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "total_trades": total_trades,
        "wins": wins,
        "losses": total_trades - wins,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "signals_checked": signals_checked,
        "filter_blocks": dict(filter_blocks),
        "trades": trades,
        "config": {
            "skip_body_position": skip_body_position,
            "skip_wick_rejection": skip_wick_rejection,
        }
    }


def simulate_trade(df, entry_idx: int, signal_type: str, entry: float, tp: float, sl: float,
                   symbol: str, timeframe: str) -> Dict:
    """Simulate trade execution and calculate PnL."""

    # Look forward to see if TP or SL was hit
    for j in range(entry_idx + 1, min(entry_idx + 100, len(df))):
        candle = df.iloc[j]
        high = float(candle["high"])
        low = float(candle["low"])

        if signal_type == "LONG":
            # Check SL first (conservative)
            if low <= sl:
                pnl = calculate_pnl(entry, sl, signal_type)
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "type": signal_type,
                    "entry": entry,
                    "exit": sl,
                    "tp": tp,
                    "sl": sl,
                    "pnl": pnl,
                    "result": "SL",
                    "bars_held": j - entry_idx,
                }
            # Check TP
            if high >= tp:
                pnl = calculate_pnl(entry, tp, signal_type)
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "type": signal_type,
                    "entry": entry,
                    "exit": tp,
                    "tp": tp,
                    "sl": sl,
                    "pnl": pnl,
                    "result": "TP",
                    "bars_held": j - entry_idx,
                }
        else:  # SHORT
            # Check SL first
            if high >= sl:
                pnl = calculate_pnl(entry, sl, signal_type)
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "type": signal_type,
                    "entry": entry,
                    "exit": sl,
                    "tp": tp,
                    "sl": sl,
                    "pnl": pnl,
                    "result": "SL",
                    "bars_held": j - entry_idx,
                }
            # Check TP
            if low <= tp:
                pnl = calculate_pnl(entry, tp, signal_type)
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "type": signal_type,
                    "entry": entry,
                    "exit": tp,
                    "tp": tp,
                    "sl": sl,
                    "pnl": pnl,
                    "result": "TP",
                    "bars_held": j - entry_idx,
                }

    # Trade didn't complete within lookforward window
    return None


def calculate_pnl(entry: float, exit: float, signal_type: str) -> float:
    """Calculate PnL for a trade."""
    position_size = 2000 * 0.0175  # $35 risk

    if signal_type == "LONG":
        pct_change = (exit - entry) / entry
    else:
        pct_change = (entry - exit) / entry

    # Apply leverage
    leverage = 10
    pnl = position_size * pct_change * leverage

    return pnl


def run_filter_comparison(
    symbols: List[str] = None,
    timeframes: List[str] = None,
    candles: int = 5000,
) -> Dict:
    """Run backtest comparison for different filter configurations."""

    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "LINKUSDT"]
    if timeframes is None:
        timeframes = ["15m", "1h"]

    print("\n" + "="*70)
    print("FILTER SIMPLIFICATION BACKTEST - PRIORITY 3")
    print("Testing marginal contribution of body_position and wick_rejection filters")
    print("="*70)
    print(f"\nSymbols: {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Candles per stream: {candles}")

    client = get_client()

    # Filter configurations to test
    configurations = [
        {"name": "Baseline (All Filters)", "skip_body": False, "skip_wick": False},
        {"name": "Skip Body Position", "skip_body": True, "skip_wick": False},
        {"name": "Skip Wick Rejection", "skip_body": False, "skip_wick": True},
        {"name": "Skip Both Filters", "skip_body": True, "skip_wick": True},
    ]

    all_results = {cfg["name"]: [] for cfg in configurations}

    # Run backtests
    for symbol in symbols:
        for tf in timeframes:
            print(f"\n{'='*50}")
            print(f"Testing {symbol}-{tf}...")
            print(f"{'='*50}")

            # Fetch data
            if candles > 1000:
                df = client.get_klines_paginated(symbol, tf, total_candles=candles)
            else:
                df = client.get_klines(symbol, tf, limit=candles)

            if df is None or df.empty:
                print(f"   Could not fetch data for {symbol}-{tf}")
                continue

            df = calculate_indicators(df, tf)

            # Test each configuration
            for cfg in configurations:
                result = run_single_stream_backtest(
                    df, symbol, tf,
                    skip_body_position=cfg["skip_body"],
                    skip_wick_rejection=cfg["skip_wick"],
                )
                all_results[cfg["name"]].append(result)

                print(f"   {cfg['name']}: {result['total_trades']} trades, "
                      f"${result['total_pnl']:.2f} PnL, "
                      f"{result['win_rate']*100:.0f}% WR")

    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)

    aggregates = {}
    for cfg_name, results in all_results.items():
        total_trades = sum(r["total_trades"] for r in results)
        total_pnl = sum(r["total_pnl"] for r in results)
        total_wins = sum(r["wins"] for r in results)
        win_rate = total_wins / total_trades if total_trades > 0 else 0

        aggregates[cfg_name] = {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "total_wins": total_wins,
        }

    print(f"\n{'Configuration':<25} {'Trades':>10} {'PnL':>12} {'Win Rate':>10}")
    print("-"*60)
    for cfg_name, agg in aggregates.items():
        print(f"{cfg_name:<25} {agg['total_trades']:>10} ${agg['total_pnl']:>10.2f} "
              f"{agg['win_rate']*100:>9.1f}%")

    # Calculate marginal contributions
    print("\n" + "="*70)
    print("MARGINAL FILTER CONTRIBUTION ANALYSIS")
    print("="*70)

    baseline = aggregates["Baseline (All Filters)"]

    print(f"\n{'Filter Removed':<25} {'Trades Delta':>12} {'PnL Delta':>12} {'WR Delta':>10}")
    print("-"*60)

    # Body position contribution
    skip_body = aggregates["Skip Body Position"]
    body_trade_delta = skip_body["total_trades"] - baseline["total_trades"]
    body_pnl_delta = skip_body["total_pnl"] - baseline["total_pnl"]
    body_wr_delta = (skip_body["win_rate"] - baseline["win_rate"]) * 100

    print(f"{'Body Position':<25} {body_trade_delta:>+12} ${body_pnl_delta:>+11.2f} "
          f"{body_wr_delta:>+9.1f}%")

    # Wick rejection contribution
    skip_wick = aggregates["Skip Wick Rejection"]
    wick_trade_delta = skip_wick["total_trades"] - baseline["total_trades"]
    wick_pnl_delta = skip_wick["total_pnl"] - baseline["total_pnl"]
    wick_wr_delta = (skip_wick["win_rate"] - baseline["win_rate"]) * 100

    print(f"{'Wick Rejection':<25} {wick_trade_delta:>+12} ${wick_pnl_delta:>+11.2f} "
          f"{wick_wr_delta:>+9.1f}%")

    # Both filters contribution
    skip_both = aggregates["Skip Both Filters"]
    both_trade_delta = skip_both["total_trades"] - baseline["total_trades"]
    both_pnl_delta = skip_both["total_pnl"] - baseline["total_pnl"]
    both_wr_delta = (skip_both["win_rate"] - baseline["win_rate"]) * 100

    print(f"{'Both Filters':<25} {both_trade_delta:>+12} ${both_pnl_delta:>+11.2f} "
          f"{both_wr_delta:>+9.1f}%")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    # Determine best configuration
    best_config = max(aggregates.items(), key=lambda x: x[1]["total_pnl"])

    if best_config[0] == "Baseline (All Filters)":
        print("""
KEEP ALL FILTERS

The current filter configuration produces the best results.
Removing filters does not improve performance.
        """)
    elif best_config[0] == "Skip Body Position":
        print("""
REMOVE BODY_POSITION FILTER

Recommendation: Set skip_body_position=True in DEFAULT_STRATEGY_CONFIG

Reason:
- Filter has 99.9% pass rate (effectively useless)
- Removing it improves PnL
- No meaningful impact on win rate
        """)
    elif best_config[0] == "Skip Wick Rejection":
        print("""
REMOVE WICK_REJECTION FILTER

Recommendation: Set skip_wick_rejection=True in DEFAULT_STRATEGY_CONFIG

Reason:
- Removing this filter improves PnL
- More trades generated, better risk/reward
        """)
    else:
        print("""
REMOVE BOTH FILTERS

Recommendation: Set skip_body_position=True AND skip_wick_rejection=True

Reason:
- Combined removal produces best results
- Both filters are adding complexity without value
        """)

    print(f"\nBest Configuration: {best_config[0]}")
    print(f"   Trades: {best_config[1]['total_trades']}")
    print(f"   PnL: ${best_config[1]['total_pnl']:.2f}")
    print(f"   Win Rate: {best_config[1]['win_rate']*100:.1f}%")

    # Save results
    output_dir = os.path.join(DATA_DIR, "filter_simplification")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "symbols": symbols,
            "timeframes": timeframes,
            "candles": candles,
        },
        "aggregates": aggregates,
        "marginal_contributions": {
            "body_position": {
                "trade_delta": body_trade_delta,
                "pnl_delta": body_pnl_delta,
                "wr_delta": body_wr_delta,
            },
            "wick_rejection": {
                "trade_delta": wick_trade_delta,
                "pnl_delta": wick_pnl_delta,
                "wr_delta": wick_wr_delta,
            },
            "both": {
                "trade_delta": both_trade_delta,
                "pnl_delta": both_pnl_delta,
                "wr_delta": both_wr_delta,
            },
        },
        "recommendation": best_config[0],
    }

    results_path = os.path.join(output_dir, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Filter Simplification Backtest")
    parser.add_argument('--quick', action='store_true', help='Quick test (fewer candles)')
    parser.add_argument('--full-year', action='store_true', help='Full year test')
    parser.add_argument('--symbols', nargs='+', help='Symbols to test')
    parser.add_argument('--timeframes', nargs='+', help='Timeframes to test')

    args = parser.parse_args()

    candles = 5000  # Default
    if args.quick:
        candles = 2000
    elif args.full_year:
        candles = 35000  # ~1 year of 15m candles

    symbols = args.symbols or ["BTCUSDT", "ETHUSDT", "LINKUSDT"]
    timeframes = args.timeframes or ["15m", "1h"]

    results = run_filter_comparison(
        symbols=symbols,
        timeframes=timeframes,
        candles=candles,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
