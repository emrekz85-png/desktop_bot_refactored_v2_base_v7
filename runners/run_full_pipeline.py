#!/usr/bin/env python3
"""
FULL ANALYSIS PIPELINE

Tek komutla tüm analiz sürecini çalıştırır:
1. AT Scenario Analysis → Baseline ve AT modları
2. Filter Combo Test → En iyi filtre kombinasyonu
3. Rolling WF Validation → OOS doğrulama
4. Cost-Aware Test → Gerçekçi maliyet analizi

Kullanım:
    # Tek symbol/timeframe
    python runners/run_full_pipeline.py --symbol BTCUSDT --timeframe 15m

    # Birden fazla symbol
    python runners/run_full_pipeline.py --symbols BTCUSDT,ETHUSDT --timeframes 15m,1h

    # Tüm kombinasyonlar
    python runners/run_full_pipeline.py --all
"""

import sys
import os
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import get_client, calculate_indicators, set_backtest_mode
from core.at_scenario_analyzer import check_core_signal, ATScenarioAnalyzer
from runners.run_filter_combo_test import apply_filters as original_apply_filters

# Enable backtest mode
set_backtest_mode(True)

# Default symbols and timeframes
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
DEFAULT_TIMEFRAMES = ["5m", "15m", "1h", "4h"]

# Filter definitions
ALL_FILTERS = [
    "regime", "at_binary", "at_flat_filter", "adx_filter",
    "ssl_touch", "rsi_filter", "pbema_distance", "overlap_check",
    "body_position", "wick_rejection", "min_sl_filter"
]


def apply_filters(df, idx, signal_type, entry_price=None, sl_price=None, **filter_flags) -> Tuple[bool, str]:
    """Wrapper for original apply_filters from run_filter_combo_test."""
    return original_apply_filters(
        df=df,
        index=idx,
        signal_type=signal_type,
        entry_price=entry_price,
        sl_price=sl_price,
        use_regime_filter=filter_flags.get("use_regime_filter", False),
        use_at_binary=filter_flags.get("use_at_binary", False),
        use_at_flat_filter=filter_flags.get("use_at_flat_filter", False),
        use_adx_filter=filter_flags.get("use_adx_filter", False),
        use_ssl_touch=filter_flags.get("use_ssl_touch", False),
        use_rsi_filter=filter_flags.get("use_rsi_filter", False),
        use_pbema_distance=filter_flags.get("use_pbema_distance", False),
        use_overlap_check=filter_flags.get("use_overlap_check", False),
        use_body_position=filter_flags.get("use_body_position", False),
        use_wick_rejection=filter_flags.get("use_wick_rejection", False),
        use_min_sl_filter=filter_flags.get("use_min_sl_filter", False),
    )


def fetch_data(symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
    """Fetch and prepare data with indicators."""
    client = get_client()

    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
    minutes = tf_minutes.get(timeframe, 15)
    candles = (days * 24 * 60) // minutes + 500

    all_dfs = []
    remaining = candles
    end_time = None

    while remaining > 0:
        chunk = min(remaining, 1000)

        if end_time:
            import requests
            url = f"{client.BASE_URL}/klines"
            params = {'symbol': symbol, 'interval': timeframe, 'limit': chunk, 'endTime': end_time}
            res = requests.get(url, params=params, timeout=30)
            if res.status_code != 200 or not res.json():
                break
            df_c = pd.DataFrame(res.json()).iloc[:, :6]
            df_c.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_c['timestamp'] = pd.to_datetime(df_c['timestamp'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
        else:
            df_c = client.get_klines(symbol=symbol, interval=timeframe, limit=chunk)

        if df_c.empty:
            break

        all_dfs.insert(0, df_c)
        remaining -= len(df_c)

        if 'timestamp' in df_c.columns:
            end_time = int(df_c['timestamp'].iloc[0].timestamp() * 1000) - 1
        else:
            break

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    df = calculate_indicators(df, timeframe=timeframe)

    return df


def simulate_trade(df, idx, signal_type, entry, tp, sl, position_size=35.0,
                   slippage=0.0, fee=0.0) -> dict:
    """Simulate a single trade with optional costs."""
    # Apply slippage to entry
    if signal_type == "LONG":
        actual_entry = entry * (1 + slippage)
    else:
        actual_entry = entry * (1 - slippage)

    for i in range(idx + 1, len(df)):
        candle = df.iloc[i]
        high, low = float(candle["high"]), float(candle["low"])

        if signal_type == "LONG":
            if low <= sl:
                exit_price = sl * (1 - slippage)
                pnl = (exit_price - actual_entry) / actual_entry * position_size
                pnl -= position_size * fee * 2
                return {"pnl": pnl, "win": False, "exit_reason": "SL", "bars_held": i - idx}
            if high >= tp:
                exit_price = tp * (1 - slippage)
                pnl = (exit_price - actual_entry) / actual_entry * position_size
                pnl -= position_size * fee * 2
                return {"pnl": pnl, "win": pnl > 0, "exit_reason": "TP", "bars_held": i - idx}
        else:
            if high >= sl:
                exit_price = sl * (1 + slippage)
                pnl = (actual_entry - exit_price) / actual_entry * position_size
                pnl -= position_size * fee * 2
                return {"pnl": pnl, "win": False, "exit_reason": "SL", "bars_held": i - idx}
            if low <= tp:
                exit_price = tp * (1 + slippage)
                pnl = (actual_entry - exit_price) / actual_entry * position_size
                pnl -= position_size * fee * 2
                return {"pnl": pnl, "win": pnl > 0, "exit_reason": "TP", "bars_held": i - idx}

    # EOD
    last = float(df.iloc[-1]["close"])
    if signal_type == "LONG":
        pnl = (last - actual_entry) / actual_entry * position_size
    else:
        pnl = (actual_entry - last) / actual_entry * position_size
    pnl -= position_size * fee * 2
    return {"pnl": pnl, "win": pnl > 0, "exit_reason": "EOD", "bars_held": len(df) - idx}


def run_backtest(df, filter_flags, slippage=0.0, fee=0.0, min_bars=5) -> dict:
    """Run backtest with given filter config."""
    trades = []
    last_idx = -min_bars

    for i in range(60, len(df) - 10):
        if i - last_idx < min_bars:
            continue

        signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)
        if signal_type is None:
            continue

        last_idx = i

        passed, _ = apply_filters(df, i, signal_type, entry_price=entry, sl_price=sl, **filter_flags)
        if not passed:
            continue

        trade = simulate_trade(df, i, signal_type, entry, tp, sl,
                               slippage=slippage, fee=fee)
        trades.append(trade)

    if not trades:
        return {"trades": 0, "pnl": 0, "wr": 0, "dd": 0}

    wins = sum(1 for t in trades if t["win"])
    pnl = sum(t["pnl"] for t in trades)

    # Drawdown
    equity = [0]
    for t in trades:
        equity.append(equity[-1] + t["pnl"])
    peak, dd = 0, 0
    for e in equity:
        peak = max(peak, e)
        dd = max(dd, peak - e)

    return {
        "trades": len(trades),
        "wins": wins,
        "wr": wins / len(trades) * 100,
        "pnl": pnl,
        "dd": dd,
    }


def run_rolling_wf(df, filter_flags, forward_days=7, slippage=0.0, fee=0.0) -> dict:
    """Run rolling walk-forward test."""
    from datetime import timedelta

    start_date = df.index[0]
    end_date = df.index[-1]

    all_trades = []
    window_results = []
    current = start_date
    window_id = 0

    while current < end_date:
        window_end = current + timedelta(days=forward_days)
        mask = (df.index >= current) & (df.index < window_end)
        df_window = df[mask]

        if len(df_window) >= 10:
            result = run_backtest(df_window, filter_flags, slippage, fee)
            window_results.append({
                "window_id": window_id,
                "trades": result["trades"],
                "pnl": result["pnl"],
            })

        current = window_end
        window_id += 1

    total_trades = sum(w["trades"] for w in window_results)
    total_pnl = sum(w["pnl"] for w in window_results)
    positive_windows = sum(1 for w in window_results if w["pnl"] > 0)

    return {
        "windows": len(window_results),
        "trades": total_trades,
        "pnl": total_pnl,
        "positive_windows": positive_windows,
        "window_hit_rate": positive_windows / len(window_results) * 100 if window_results else 0,
    }


def run_full_pipeline(
    symbol: str,
    timeframe: str,
    days: int = 365,
    verbose: bool = True,
) -> dict:
    """
    Run the complete analysis pipeline for a symbol/timeframe.

    Steps:
    1. Fetch data
    2. AT Scenario Analysis (baseline)
    3. Filter Combo Test (find best filters)
    4. Rolling WF Validation
    5. Cost-Aware Test

    Returns comprehensive report dict.
    """
    report = {
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "timestamp": datetime.now().isoformat(),
        "steps": {},
    }

    def log(msg):
        if verbose:
            print(msg)

    log(f"\n{'='*70}")
    log(f"FULL PIPELINE: {symbol} {timeframe}")
    log(f"{'='*70}")

    # Step 1: Fetch Data
    log(f"\n[1/5] Fetching data...")
    df = fetch_data(symbol, timeframe, days)
    if df.empty:
        report["error"] = "No data"
        return report

    report["data"] = {
        "candles": len(df),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
    }
    log(f"      Got {len(df)} candles")

    # Step 2: AT Scenario Analysis (Baseline)
    log(f"\n[2/5] AT Scenario Analysis (Baseline)...")

    # Count core signals
    core_signals = 0
    for i in range(60, len(df) - 10):
        signal_type, _, _, _, _ = check_core_signal(df, index=i)
        if signal_type:
            core_signals += 1

    # Run baseline (no filters)
    baseline = run_backtest(df, {})

    report["steps"]["baseline"] = {
        "core_signals": core_signals,
        "trades": baseline["trades"],
        "pnl": baseline["pnl"],
        "wr": baseline["wr"],
        "dd": baseline["dd"],
    }
    log(f"      Core signals: {core_signals}")
    log(f"      Baseline: {baseline['trades']} trades, ${baseline['pnl']:.2f} PnL")

    # Step 3: Filter Combo Test
    log(f"\n[3/5] Filter Combo Test (Incremental)...")

    filter_results = []

    # Test regime first
    regime_flags = {"use_regime_filter": True}
    regime_result = run_backtest(df, regime_flags)
    filter_results.append({
        "name": "REGIME only",
        "filters": ["regime"],
        **regime_result
    })

    # Test each filter added to regime
    for filter_name in ALL_FILTERS[1:]:  # Skip regime
        test_flags = {"use_regime_filter": True, f"use_{filter_name}": True}
        result = run_backtest(df, test_flags)
        filter_results.append({
            "name": f"REGIME + {filter_name}",
            "filters": ["regime", filter_name],
            **result
        })

    # Sort by PnL
    filter_results.sort(key=lambda x: x["pnl"], reverse=True)

    # Find best combo
    best_combo = filter_results[0]

    report["steps"]["filter_combo"] = {
        "tested": len(filter_results),
        "results": filter_results[:5],  # Top 5
        "best": best_combo,
    }

    log(f"      Tested {len(filter_results)} combinations")
    log(f"      Best: {best_combo['name']}")
    log(f"            {best_combo['trades']} trades, ${best_combo['pnl']:.2f} PnL, {best_combo['wr']:.1f}% WR")

    # Step 4: Rolling WF Validation
    log(f"\n[4/5] Rolling Walk-Forward Validation...")

    # Build filter flags for best combo
    best_flags = {}
    for f in best_combo["filters"]:
        if f == "regime":
            best_flags["use_regime_filter"] = True
        else:
            best_flags[f"use_{f}"] = True

    wf_result = run_rolling_wf(df, best_flags)

    report["steps"]["rolling_wf"] = {
        "config": best_combo["name"],
        "windows": wf_result["windows"],
        "trades": wf_result["trades"],
        "pnl": wf_result["pnl"],
        "positive_windows": wf_result["positive_windows"],
        "window_hit_rate": wf_result["window_hit_rate"],
        "passed": wf_result["pnl"] > 0,
    }

    log(f"      {wf_result['windows']} windows, {wf_result['trades']} trades")
    log(f"      PnL: ${wf_result['pnl']:.2f}")
    log(f"      Window hit rate: {wf_result['window_hit_rate']:.1f}%")

    # Step 5: Cost-Aware Test
    log(f"\n[5/5] Cost-Aware Test...")

    # Ideal (no costs)
    ideal = run_backtest(df, best_flags, slippage=0, fee=0)

    # With costs
    cost_aware = run_backtest(df, best_flags, slippage=0.0005, fee=0.0007)

    cost_impact = ideal["pnl"] - cost_aware["pnl"]
    edge_per_trade = ideal["pnl"] / ideal["trades"] if ideal["trades"] else 0
    edge_pct = edge_per_trade / 35 * 100

    report["steps"]["cost_aware"] = {
        "ideal_pnl": ideal["pnl"],
        "ideal_trades": ideal["trades"],
        "cost_aware_pnl": cost_aware["pnl"],
        "cost_impact": cost_impact,
        "edge_per_trade": edge_per_trade,
        "edge_pct": edge_pct,
        "profitable_after_costs": cost_aware["pnl"] > 0,
    }

    log(f"      Ideal: ${ideal['pnl']:.2f}")
    log(f"      With costs: ${cost_aware['pnl']:.2f}")
    log(f"      Cost impact: ${cost_impact:.2f}")
    log(f"      Edge: {edge_pct:.3f}% per trade")

    # Summary
    report["summary"] = {
        "baseline_signals": core_signals,
        "best_config": best_combo["name"],
        "final_trades": cost_aware["trades"] if cost_aware["trades"] else ideal["trades"],
        "ideal_pnl": ideal["pnl"],
        "realistic_pnl": cost_aware["pnl"],
        "edge_pct": edge_pct,
        "wf_passed": wf_result["pnl"] > 0,
        "profitable_after_costs": cost_aware["pnl"] > 0,
        "recommendation": "TRADE" if cost_aware["pnl"] > 0 else "DO NOT TRADE",
    }

    # Print summary
    log(f"\n{'='*70}")
    log(f"PIPELINE SUMMARY: {symbol} {timeframe}")
    log(f"{'='*70}")
    log(f"")
    log(f"Signal Flow:")
    log(f"  Core Signals:    {core_signals:>6}")
    log(f"  After Baseline:  {baseline['trades']:>6} ({baseline['trades']/core_signals*100:.1f}%)")
    log(f"  After Best Cfg:  {best_combo['trades']:>6} ({best_combo['trades']/core_signals*100:.1f}%)")
    log(f"  After WF:        {wf_result['trades']:>6}")
    log(f"")
    log(f"Best Config: {best_combo['name']}")
    log(f"")
    log(f"Results:")
    log(f"  Ideal PnL:      ${ideal['pnl']:>8.2f}")
    log(f"  Realistic PnL:  ${cost_aware['pnl']:>8.2f}")
    log(f"  Edge:           {edge_pct:>8.3f}%")
    log(f"")
    if cost_aware["pnl"] > 0:
        log(f"✅ RECOMMENDATION: TRADE (profitable after costs)")
    else:
        log(f"❌ RECOMMENDATION: DO NOT TRADE (edge < costs)")
    log(f"{'='*70}")

    return report


def run_multi_pipeline(
    symbols: List[str],
    timeframes: List[str],
    days: int = 365,
    output_dir: str = "data/pipeline_reports",
) -> dict:
    """Run pipeline for multiple symbols and timeframes."""

    os.makedirs(output_dir, exist_ok=True)

    all_reports = {}
    summary_rows = []

    total = len(symbols) * len(timeframes)
    current = 0

    print(f"\n{'='*70}")
    print(f"MULTI-PIPELINE: {len(symbols)} symbols × {len(timeframes)} timeframes = {total} runs")
    print(f"{'='*70}")

    for symbol in symbols:
        all_reports[symbol] = {}
        for tf in timeframes:
            current += 1
            print(f"\n[{current}/{total}] {symbol} {tf}")

            try:
                report = run_full_pipeline(symbol, tf, days, verbose=True)
                all_reports[symbol][tf] = report

                if "summary" in report:
                    summary_rows.append({
                        "symbol": symbol,
                        "timeframe": tf,
                        "core_signals": report["summary"]["baseline_signals"],
                        "best_config": report["summary"]["best_config"],
                        "trades": report["summary"]["final_trades"],
                        "ideal_pnl": report["summary"]["ideal_pnl"],
                        "realistic_pnl": report["summary"]["realistic_pnl"],
                        "edge_pct": report["summary"]["edge_pct"],
                        "wf_passed": report["summary"]["wf_passed"],
                        "profitable": report["summary"]["profitable_after_costs"],
                        "recommendation": report["summary"]["recommendation"],
                    })
            except Exception as e:
                print(f"      ERROR: {e}")
                all_reports[symbol][tf] = {"error": str(e)}

    # Save detailed reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{output_dir}/full_pipeline_{timestamp}.json"
    with open(report_file, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*100}")
    print("MULTI-PIPELINE SUMMARY")
    print(f"{'='*100}")
    print(f"{'Symbol':<10} {'TF':<5} {'Signals':>8} {'Trades':>7} {'Ideal':>10} {'Real':>10} {'Edge%':>8} {'WF':>5} {'Rec':<12}")
    print("-"*100)

    for row in summary_rows:
        wf_status = "✓" if row["wf_passed"] else "✗"
        print(f"{row['symbol']:<10} {row['timeframe']:<5} {row['core_signals']:>8} {row['trades']:>7} "
              f"${row['ideal_pnl']:>8.2f} ${row['realistic_pnl']:>8.2f} {row['edge_pct']:>7.3f}% "
              f"{wf_status:>5} {row['recommendation']:<12}")

    print("-"*100)

    # Count recommendations
    trade_count = sum(1 for r in summary_rows if r["recommendation"] == "TRADE")
    print(f"\nRecommendations: {trade_count}/{len(summary_rows)} TRADE")
    print(f"\nDetailed report saved: {report_file}")

    return {
        "reports": all_reports,
        "summary": summary_rows,
        "report_file": report_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Full Analysis Pipeline")
    parser.add_argument("--symbol", type=str, help="Single symbol")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--timeframe", type=str, help="Single timeframe")
    parser.add_argument("--timeframes", type=str, help="Comma-separated timeframes")
    parser.add_argument("--days", type=int, default=365, help="Days of data")
    parser.add_argument("--all", action="store_true", help="Run all default combinations")
    parser.add_argument("--btc-only", action="store_true", help="Run BTC only with all timeframes")

    args = parser.parse_args()

    if args.all:
        symbols = DEFAULT_SYMBOLS
        timeframes = DEFAULT_TIMEFRAMES
    elif args.btc_only:
        symbols = ["BTCUSDT"]
        timeframes = DEFAULT_TIMEFRAMES
    elif args.symbol:
        symbols = [args.symbol]
        # Check for both --timeframe and --timeframes
        if args.timeframes:
            timeframes = [t.strip() for t in args.timeframes.split(",")]
        elif args.timeframe:
            timeframes = [args.timeframe]
        else:
            timeframes = ["15m"]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
        if args.timeframes:
            timeframes = [t.strip() for t in args.timeframes.split(",")]
        elif args.timeframe:
            timeframes = [args.timeframe]
        else:
            timeframes = ["15m"]
    else:
        # Default: BTCUSDT 15m only
        symbols = ["BTCUSDT"]
        timeframes = ["15m"]

    if len(symbols) == 1 and len(timeframes) == 1:
        symbol = symbols[0]
        tf = timeframes[0]
        report = run_full_pipeline(symbol, tf, args.days)

        # Save single report to symbol-specific folder
        symbol_dir = f"data/pipeline_reports/{symbol}"
        os.makedirs(symbol_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{symbol_dir}/{symbol}_{tf}_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved: {report_file}")
    elif len(symbols) == 1:
        # Single symbol, multiple timeframes - save to symbol folder
        symbol = symbols[0]
        symbol_dir = f"data/pipeline_reports/{symbol}"
        os.makedirs(symbol_dir, exist_ok=True)

        result = run_multi_pipeline(symbols, timeframes, args.days, output_dir=symbol_dir)

        # Rename report file to include symbol and timeframes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tfs_str = "_".join(timeframes)
        new_report_file = f"{symbol_dir}/{symbol}_{tfs_str}_{timestamp}.json"
        if os.path.exists(result["report_file"]):
            os.rename(result["report_file"], new_report_file)
            print(f"Report renamed to: {new_report_file}")
    else:
        # Multiple symbols - save with symbol names in filename
        symbols_str = "_".join(symbols)
        tfs_str = "_".join(timeframes)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result = run_multi_pipeline(symbols, timeframes, args.days)

        # Rename to include symbols and timeframes
        new_report_file = f"data/pipeline_reports/multi_{symbols_str}_{tfs_str}_{timestamp}.json"
        if os.path.exists(result["report_file"]):
            os.rename(result["report_file"], new_report_file)
            print(f"Report renamed to: {new_report_file}")


if __name__ == "__main__":
    main()
