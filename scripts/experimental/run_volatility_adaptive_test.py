#!/usr/bin/env python3
"""
Volatility-Adaptive Filter Test

Tests the hypothesis that relaxing filters during high volatility periods
can increase trade frequency without significantly hurting edge quality.

This script:
1. Loads historical data for recommended symbols
2. Calculates ATR percentile and BB width for each candle
3. Simulates trades with volatility-adaptive filter thresholds
4. Compares results against baseline (fixed thresholds)

Usage:
    python run_volatility_adaptive_test.py [--symbols BTC ETH LINK] [--days 180]
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    TradingEngine, SimTradeManager, TRADING_CONFIG,
    calculate_indicators, set_backtest_mode,
)
from strategies.ssl_flow import check_ssl_flow_signal


# Enable backtest mode
set_backtest_mode(True)


def calculate_atr_percentile(atr_series, lookback=100, index=-1):
    """
    Calculate ATR percentile over lookback period.
    Returns value 0-1 where 1 = highest volatility in lookback.
    """
    if index < lookback:
        return 0.5  # Default to middle

    atr_window = atr_series.iloc[index-lookback:index]
    atr_current = atr_series.iloc[index]

    atr_min = atr_window.min()
    atr_max = atr_window.max()

    if atr_max == atr_min:
        return 0.5

    return (atr_current - atr_min) / (atr_max - atr_min)


def calculate_bbwidth(close_series, period=20, std_dev=2.0, index=-1):
    """
    Calculate Bollinger Band Width = (Upper - Lower) / Middle
    Returns percentage width.
    """
    if index < period:
        return 0.03  # Default to moderate

    window = close_series.iloc[index-period:index]
    middle = window.mean()
    std = window.std()

    if middle == 0 or np.isnan(std):
        return 0.03

    upper = middle + std_dev * std
    lower = middle - std_dev * std

    return (upper - lower) / middle


def get_dynamic_thresholds(atr_pct, bb_width):
    """
    Get volatility-adaptive thresholds based on current volatility regime.

    Returns dict with adjusted thresholds.
    """
    # Base thresholds (15m baseline from config)
    base = {
        "min_pbema_distance": 0.004,
        "flat_threshold": 0.002,
        "adx_min": 15.0,
        "regime_adx_threshold": 20.0,
        "skip_regime_filter": False,
    }

    # Determine volatility regime
    if atr_pct > 0.70 or bb_width > 0.06:
        # HIGH VOLATILITY: Relax filters significantly
        return {
            "min_pbema_distance": 0.002,  # 50% looser
            "flat_threshold": 0.004,      # 100% looser
            "adx_min": 10.0,              # 33% looser
            "regime_adx_threshold": 15.0, # 25% looser
            "skip_regime_filter": True,   # Bypass regime entirely
            "volatility_regime": "HIGH",
        }
    elif atr_pct > 0.50 or bb_width > 0.04:
        # ELEVATED VOLATILITY: Moderate relaxation
        return {
            "min_pbema_distance": 0.003,  # 25% looser
            "flat_threshold": 0.003,      # 50% looser
            "adx_min": 12.0,              # 20% looser
            "regime_adx_threshold": 18.0, # 10% looser
            "skip_regime_filter": False,
            "volatility_regime": "ELEVATED",
        }
    elif atr_pct < 0.20 and bb_width < 0.02:
        # LOW VOLATILITY: Tighten filters
        return {
            "min_pbema_distance": 0.005,  # 25% tighter
            "flat_threshold": 0.0015,     # 25% tighter
            "adx_min": 20.0,              # 33% tighter
            "regime_adx_threshold": 25.0, # 25% tighter
            "skip_regime_filter": False,
            "volatility_regime": "LOW",
        }
    else:
        # NORMAL: Use baseline
        base["volatility_regime"] = "NORMAL"
        return base


def run_signal_scan(df, config, use_dynamic=False, timeframe="15m"):
    """
    Scan DataFrame for signals with optional dynamic thresholds.

    Returns list of signal events with metadata.
    """
    signals = []
    warmup = 250

    # Pre-calculate volatility metrics if using dynamic mode
    if use_dynamic:
        atr_series = df["atr"] if "atr" in df.columns else pd.Series([0] * len(df))
        close_series = df["close"]

    for i in range(warmup, len(df) - 2):
        # Get volatility metrics for this candle
        if use_dynamic:
            atr_pct = calculate_atr_percentile(atr_series, 100, i)
            bb_width = calculate_bbwidth(close_series, 20, 2.0, i)
            dynamic_cfg = get_dynamic_thresholds(atr_pct, bb_width)

            # Merge dynamic thresholds with base config
            test_config = {**config, **dynamic_cfg}
        else:
            test_config = config
            atr_pct = 0.5
            bb_width = 0.03
            dynamic_cfg = {"volatility_regime": "BASELINE"}

        # Check for signal
        result = check_ssl_flow_signal(
            df,
            index=i,
            min_rr=test_config.get("rr", 2.0),
            rsi_limit=test_config.get("rsi", 70),
            min_pbema_distance=test_config.get("min_pbema_distance", 0.004),
            adx_min=test_config.get("adx_min", 15.0),
            regime_adx_threshold=test_config.get("regime_adx_threshold", 20.0),
            skip_wick_rejection=True,  # Already proven beneficial
            return_debug=True,
            timeframe=timeframe,
        )

        signal_type, entry, tp, sl, reason, debug = result

        if signal_type and "ACCEPTED" in reason:
            signals.append({
                "index": i,
                "timestamp": df["timestamp"].iloc[i] if "timestamp" in df.columns else i,
                "signal": signal_type,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "rr": debug.get("rr_value", 0),
                "atr_pct": atr_pct,
                "bb_width": bb_width,
                "volatility_regime": dynamic_cfg.get("volatility_regime", "BASELINE"),
                "reason": reason,
            })

    return signals


def simulate_trades(df, signals, initial_balance=2000.0):
    """
    Simulate trades from signals and calculate performance metrics.
    """
    if not signals:
        return {
            "pnl": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_rr": 0.0,
            "max_dd": 0.0,
        }

    # Simple simulation: Check if TP or SL hit first
    results = []

    for sig in signals:
        idx = sig["index"]
        signal_type = sig["signal"]
        entry = sig["entry"]
        tp = sig["tp"]
        sl = sig["sl"]

        # Look forward for exit
        outcome = None
        exit_price = None

        for j in range(idx + 1, min(idx + 100, len(df))):
            high = df["high"].iloc[j]
            low = df["low"].iloc[j]

            if signal_type == "LONG":
                if low <= sl:
                    outcome = "SL"
                    exit_price = sl
                    break
                elif high >= tp:
                    outcome = "TP"
                    exit_price = tp
                    break
            else:  # SHORT
                if high >= sl:
                    outcome = "SL"
                    exit_price = sl
                    break
                elif low <= tp:
                    outcome = "TP"
                    exit_price = tp
                    break

        if outcome is None:
            # Timeout - exit at last close
            outcome = "TIMEOUT"
            exit_price = df["close"].iloc[min(idx + 100, len(df) - 1)]

        # Calculate PnL
        risk_per_trade = initial_balance * 0.0175  # 1.75% risk

        if signal_type == "LONG":
            r_multiple = (exit_price - entry) / (entry - sl) if entry > sl else 0
        else:
            r_multiple = (entry - exit_price) / (sl - entry) if sl > entry else 0

        pnl = risk_per_trade * r_multiple

        results.append({
            **sig,
            "outcome": outcome,
            "exit_price": exit_price,
            "r_multiple": r_multiple,
            "pnl": pnl,
        })

    # Calculate metrics
    total_pnl = sum(r["pnl"] for r in results)
    wins = sum(1 for r in results if r["pnl"] > 0)
    losses = sum(1 for r in results if r["pnl"] <= 0)

    # Calculate max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for r in results:
        cumulative += r["pnl"]
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    return {
        "pnl": total_pnl,
        "trades": len(results),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(results) if results else 0,
        "avg_rr": sum(r["r_multiple"] for r in results) / len(results) if results else 0,
        "max_dd": max_dd,
        "results": results,
    }


def run_comparison_test(symbol, timeframe, days, verbose=True):
    """
    Run comparison between baseline and volatility-adaptive approaches.
    """
    print(f"\n{'='*60}")
    print(f"Testing {symbol} on {timeframe} ({days} days)")
    print(f"{'='*60}")

    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 30)  # Extra for warmup

    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

    df = TradingEngine.get_historical_data_pagination(
        symbol, timeframe,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )

    if df is None or df.empty:
        print(f"  ERROR: No data for {symbol}")
        return None

    # Calculate indicators
    df = calculate_indicators(df, timeframe)
    print(f"  Loaded {len(df)} candles")

    # Base config
    base_config = {
        "rr": 2.0,
        "rsi": 70,
        "at_active": True,
        "strategy_mode": "ssl_flow",
        "min_pbema_distance": 0.004,
        "adx_min": 15.0,
        "regime_adx_threshold": 20.0,
    }

    # Run baseline scan
    print("\n  Running BASELINE scan (fixed thresholds)...")
    baseline_signals = run_signal_scan(df, base_config, use_dynamic=False, timeframe=timeframe)
    baseline_results = simulate_trades(df, baseline_signals)

    print(f"    Signals: {len(baseline_signals)}")
    print(f"    Trades: {baseline_results['trades']}")
    print(f"    PnL: ${baseline_results['pnl']:.2f}")
    print(f"    Win Rate: {baseline_results['win_rate']*100:.1f}%")
    print(f"    Avg R: {baseline_results['avg_rr']:.2f}")
    print(f"    Max DD: ${baseline_results['max_dd']:.2f}")

    # Run volatility-adaptive scan
    print("\n  Running VOLATILITY-ADAPTIVE scan (dynamic thresholds)...")
    adaptive_signals = run_signal_scan(df, base_config, use_dynamic=True, timeframe=timeframe)
    adaptive_results = simulate_trades(df, adaptive_signals)

    print(f"    Signals: {len(adaptive_signals)}")
    print(f"    Trades: {adaptive_results['trades']}")
    print(f"    PnL: ${adaptive_results['pnl']:.2f}")
    print(f"    Win Rate: {adaptive_results['win_rate']*100:.1f}%")
    print(f"    Avg R: {adaptive_results['avg_rr']:.2f}")
    print(f"    Max DD: ${adaptive_results['max_dd']:.2f}")

    # Analyze volatility regime distribution
    if adaptive_signals:
        regime_counts = {}
        for sig in adaptive_signals:
            regime = sig.get("volatility_regime", "UNKNOWN")
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        print("\n  Volatility Regime Distribution:")
        for regime, count in sorted(regime_counts.items()):
            print(f"    {regime}: {count} signals ({count/len(adaptive_signals)*100:.1f}%)")

    # Calculate improvement
    print("\n  COMPARISON:")
    trade_increase = adaptive_results['trades'] - baseline_results['trades']
    trade_increase_pct = (trade_increase / baseline_results['trades'] * 100) if baseline_results['trades'] > 0 else 0
    pnl_delta = adaptive_results['pnl'] - baseline_results['pnl']
    wr_delta = (adaptive_results['win_rate'] - baseline_results['win_rate']) * 100

    print(f"    Trade Increase: {trade_increase:+d} ({trade_increase_pct:+.1f}%)")
    print(f"    PnL Delta: ${pnl_delta:+.2f}")
    print(f"    Win Rate Delta: {wr_delta:+.1f}%")

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "baseline": baseline_results,
        "adaptive": adaptive_results,
        "trade_increase": trade_increase,
        "trade_increase_pct": trade_increase_pct,
        "pnl_delta": pnl_delta,
        "wr_delta": wr_delta,
    }


def main():
    parser = argparse.ArgumentParser(description="Volatility-Adaptive Filter Test")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "LINKUSDT"],
                       help="Symbols to test")
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h"],
                       help="Timeframes to test")
    parser.add_argument("--days", type=int, default=180,
                       help="Number of days to test")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("VOLATILITY-ADAPTIVE FILTER TEST")
    print("="*70)
    print(f"Symbols: {args.symbols}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Period: {args.days} days")
    print("="*70)

    all_results = []

    for symbol in args.symbols:
        for timeframe in args.timeframes:
            result = run_comparison_test(symbol, timeframe, args.days)
            if result:
                all_results.append(result)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if not all_results:
        print("No results to summarize.")
        return

    print(f"\n{'Symbol':<12} {'TF':<6} {'Base Trades':<12} {'Adpt Trades':<12} {'Increase':<10} {'PnL Delta':<12}")
    print("-"*70)

    total_base_trades = 0
    total_adpt_trades = 0
    total_pnl_delta = 0

    for r in all_results:
        base_trades = r["baseline"]["trades"]
        adpt_trades = r["adaptive"]["trades"]
        increase = r["trade_increase_pct"]
        pnl_delta = r["pnl_delta"]

        total_base_trades += base_trades
        total_adpt_trades += adpt_trades
        total_pnl_delta += pnl_delta

        print(f"{r['symbol']:<12} {r['timeframe']:<6} {base_trades:<12} {adpt_trades:<12} {increase:+.1f}%{'':<5} ${pnl_delta:+.2f}")

    print("-"*70)
    overall_increase = ((total_adpt_trades - total_base_trades) / total_base_trades * 100) if total_base_trades > 0 else 0
    print(f"{'TOTAL':<12} {'':<6} {total_base_trades:<12} {total_adpt_trades:<12} {overall_increase:+.1f}%{'':<5} ${total_pnl_delta:+.2f}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if overall_increase > 0 and total_pnl_delta >= 0:
        print(f"SUCCESS: Volatility-adaptive approach increased trades by {overall_increase:.1f}%")
        print(f"         with ${total_pnl_delta:.2f} PnL improvement")
        print("\n  Recommendation: Consider implementing volatility-adaptive thresholds")
    elif overall_increase > 0 and total_pnl_delta < 0:
        print(f"TRADEOFF: Volatility-adaptive approach increased trades by {overall_increase:.1f}%")
        print(f"          but reduced PnL by ${abs(total_pnl_delta):.2f}")
        print("\n  Recommendation: Fine-tune thresholds or use smaller position sizes")
    else:
        print(f"NO IMPROVEMENT: Volatility-adaptive approach did not increase trades")
        print("\n  Recommendation: Review filter logic or try different volatility metrics")


if __name__ == "__main__":
    main()
