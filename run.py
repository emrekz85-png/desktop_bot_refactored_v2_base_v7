#!/usr/bin/env python3
"""
Integrated Test Runner

Tek komutla tum testleri calistir ve sonuclari tek yerde topla.

Usage:
    python run.py test BTCUSDT 15m              # Quick test (fixed config)
    python run.py test BTCUSDT 15m --full       # Full pipeline (discovery + WF + portfolio)
    python run.py test BTCUSDT 15m --quick      # 90-day quick test
    python run.py viz BTCUSDT 15m               # Visualize latest trades
    python run.py report                        # Summary of all tests
"""

import sys
import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Constants
RESULTS_DIR = Path("data/results")

# All available filters for discovery
ALL_FILTERS = [
    "at_flat_filter", "adx_filter", "at_binary",
    "ssl_touch", "rsi_filter", "pbema_distance",
    "overlap_check", "body_position", "wick_rejection",
    "min_sl_filter"
]

# Default config (for quick tests)
DEFAULT_FILTERS = ["regime", "at_flat_filter", "min_sl_filter"]


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_result_dir(symbol: str, timeframe: str, mode: str = "") -> Path:
    """Get or create result directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{mode}" if mode else ""
    result_dir = RESULTS_DIR / f"{symbol}_{timeframe}_{timestamp}{suffix}"
    ensure_dir(result_dir)
    return result_dir


def get_latest_result(symbol: str, timeframe: str) -> Path:
    """Get latest result directory."""
    pattern = f"{symbol}_{timeframe}_*"
    matches = sorted(RESULTS_DIR.glob(pattern), reverse=True)
    return matches[0] if matches else None


def fetch_data(symbol: str, timeframe: str, days: int = 365):
    """Fetch and prepare data with indicators."""
    from core import get_client, calculate_indicators, set_backtest_mode
    import pandas as pd
    import requests

    set_backtest_mode(True)
    client = get_client()

    candles_map = {"5m": 288, "15m": 96, "1h": 24, "4h": 6}
    candles = min(days * candles_map.get(timeframe, 96), 35000)

    all_dfs = []
    remaining = candles
    end_time = None

    while remaining > 0:
        chunk = min(remaining, 1000)

        if end_time:
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

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    df = calculate_indicators(df, timeframe=timeframe)

    return df


def make_filter_flags(filter_list: List[str]) -> Dict:
    """Convert filter list to flags dict."""
    return {
        "use_regime_filter": "regime" in filter_list,
        "use_at_flat_filter": "at_flat_filter" in filter_list,
        "use_min_sl_filter": "min_sl_filter" in filter_list,
        "use_at_binary": "at_binary" in filter_list,
        "use_adx_filter": "adx_filter" in filter_list,
        "use_ssl_touch": "ssl_touch" in filter_list,
        "use_rsi_filter": "rsi_filter" in filter_list,
        "use_pbema_distance": "pbema_distance" in filter_list,
        "use_overlap_check": "overlap_check" in filter_list,
        "use_body_position": "body_position" in filter_list,
        "use_wick_rejection": "wick_rejection" in filter_list,
    }


def run_backtest(df, filter_flags: Dict, min_bars: int = 5, position_size: float = 100.0) -> Dict:
    """
    Run simple backtest with given filters.

    Uses fixed position size and calculates dollar PnL properly.
    Includes slippage and fees for realistic comparison.
    """
    from core.at_scenario_analyzer import check_core_signal
    from runners.run_filter_combo_test import apply_filters

    trades = []
    last_idx = -min_bars

    slippage = 0.0005  # 0.05%
    fee = 0.0007  # 0.07%

    for i in range(60, len(df) - 10):
        if i - last_idx < min_bars:
            continue

        signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)
        if signal_type is None:
            continue

        last_idx = i

        passed, _ = apply_filters(
            df=df, index=i, signal_type=signal_type,
            entry_price=entry, sl_price=sl, **filter_flags
        )

        if not passed:
            continue

        # Apply entry slippage
        if signal_type == "LONG":
            actual_entry = entry * (1 + slippage)
        else:
            actual_entry = entry * (1 - slippage)

        # Simulate trade
        for j in range(i + 1, min(i + 200, len(df))):
            candle = df.iloc[j]
            high, low = float(candle["high"]), float(candle["low"])

            if signal_type == "LONG":
                if low <= sl:
                    exit_price = sl * (1 - slippage)
                    pnl = (exit_price - actual_entry) / actual_entry * position_size
                    pnl -= position_size * fee * 2
                    trades.append({"pnl": pnl, "win": False})
                    break
                if high >= tp:
                    exit_price = tp * (1 - slippage)
                    pnl = (exit_price - actual_entry) / actual_entry * position_size
                    pnl -= position_size * fee * 2
                    trades.append({"pnl": pnl, "win": True})
                    break
            else:
                if high >= sl:
                    exit_price = sl * (1 + slippage)
                    pnl = (actual_entry - exit_price) / actual_entry * position_size
                    pnl -= position_size * fee * 2
                    trades.append({"pnl": pnl, "win": False})
                    break
                if low <= tp:
                    exit_price = tp * (1 + slippage)
                    pnl = (actual_entry - exit_price) / actual_entry * position_size
                    pnl -= position_size * fee * 2
                    trades.append({"pnl": pnl, "win": True})
                    break

    if not trades:
        return {"trades": 0, "wins": 0, "wr": 0, "pnl": 0}

    wins = sum(1 for t in trades if t["win"])
    pnl = sum(t["pnl"] for t in trades)

    return {
        "trades": len(trades),
        "wins": wins,
        "wr": wins / len(trades) * 100,
        "pnl": pnl,
    }


def run_rolling_wf(df, filter_flags: Dict, forward_days: int = 7) -> Dict:
    """Run rolling walk-forward validation."""
    import pandas as pd

    start_date = df.index[0]
    end_date = df.index[-1]

    all_results = []
    current = start_date

    while current < end_date:
        window_end = current + timedelta(days=forward_days)
        mask = (df.index >= current) & (df.index < window_end)
        df_window = df[mask]

        if len(df_window) >= 20:
            result = run_backtest(df_window, filter_flags)
            all_results.append(result)

        current = window_end

    if not all_results:
        return {"windows": 0, "trades": 0, "pnl": 0, "positive_windows": 0}

    total_trades = sum(r["trades"] for r in all_results)
    total_pnl = sum(r["pnl"] for r in all_results)
    positive = sum(1 for r in all_results if r["pnl"] > 0)

    return {
        "windows": len(all_results),
        "trades": total_trades,
        "pnl": total_pnl,
        "positive_windows": positive,
        "window_wr": positive / len(all_results) * 100 if all_results else 0,
    }


def run_portfolio_backtest(df, filter_list: List[str]) -> Dict:
    """Run portfolio backtest with realistic sizing."""
    from core.at_scenario_analyzer import check_core_signal
    from core.simple_portfolio import PortfolioConfig, run_portfolio_backtest as portfolio_bt
    from runners.run_filter_combo_test import apply_filters

    filter_flags = make_filter_flags(filter_list)

    # Collect signals
    signals = []
    last_idx = -5

    for i in range(60, len(df) - 10):
        if i - last_idx < 5:
            continue

        signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)
        if signal_type is None:
            continue

        last_idx = i

        passed, _ = apply_filters(
            df=df, index=i, signal_type=signal_type,
            entry_price=entry, sl_price=sl, **filter_flags
        )

        if not passed:
            continue

        signals.append({
            "idx": i,
            "signal_type": signal_type,
            "entry": entry,
            "tp": tp,
            "sl": sl,
        })

    # Portfolio config
    config = PortfolioConfig(
        initial_balance=1000.0,
        risk_per_trade_pct=0.01,
        leverage=10,
        max_position_pct=0.10,
        slippage_pct=0.0005,
        fee_pct=0.0007,
        total_dd_limit=0.25,
    )

    return portfolio_bt(df, signals, config)


def run_full_pipeline(symbol: str, timeframe: str, days: int = 365) -> Dict:
    """
    Run full integrated pipeline:
    1. Fetch data
    2. Baseline (regime only)
    3. Filter discovery (incremental)
    4. Rolling WF validation
    5. Portfolio backtest
    6. Final recommendation
    """
    from core import set_backtest_mode
    set_backtest_mode(True)

    result_dir = get_result_dir(symbol, timeframe, "full")

    print(f"\n{'='*70}")
    print(f"FULL PIPELINE: {symbol} {timeframe} ({days} days)")
    print(f"{'='*70}")
    print(f"Output: {result_dir}")
    print(f"{'='*70}\n")

    # ===== STEP 1: Fetch Data =====
    print("[1/6] Fetching data...")
    df = fetch_data(symbol, timeframe, days)
    print(f"      {len(df)} candles: {df.index[0].date()} to {df.index[-1].date()}")

    # ===== STEP 2: Baseline (regime only) =====
    print("\n[2/6] Baseline (regime only)...")
    baseline_flags = make_filter_flags(["regime"])
    baseline = run_backtest(df, baseline_flags)
    print(f"      {baseline['trades']} trades | {baseline['wr']:.1f}% WR | {baseline['pnl']:.2f} PnL")

    # ===== STEP 3: Filter Discovery =====
    print("\n[3/6] Filter discovery (incremental)...")
    discovery_results = []

    # Test regime + each filter
    for filter_name in ALL_FILTERS:
        test_filters = ["regime", filter_name]
        test_flags = make_filter_flags(test_filters)
        result = run_backtest(df, test_flags)
        result["filters"] = test_filters
        result["name"] = f"regime + {filter_name}"
        discovery_results.append(result)
        print(f"      {result['name']:<35} → {result['trades']:>4} trades, {result['pnl']:>8.2f} PnL")

    # Test top combinations (regime + filter1 + filter2)
    print("\n      Testing 2-filter combos...")
    top_singles = sorted(discovery_results, key=lambda x: x["pnl"], reverse=True)[:3]

    for i, r1 in enumerate(top_singles):
        for r2 in top_singles[i+1:]:
            f1 = r1["filters"][1]
            f2 = r2["filters"][1]
            test_filters = ["regime", f1, f2]
            test_flags = make_filter_flags(test_filters)
            result = run_backtest(df, test_flags)
            result["filters"] = test_filters
            result["name"] = f"regime + {f1} + {f2}"
            discovery_results.append(result)
            print(f"      {result['name']:<35} → {result['trades']:>4} trades, {result['pnl']:>8.2f} PnL")

    # Find best combo
    valid_results = [r for r in discovery_results if r["trades"] >= 10]
    if not valid_results:
        valid_results = discovery_results

    best_combo = max(valid_results, key=lambda x: x["pnl"])
    print(f"\n      BEST: {best_combo['name']} ({best_combo['trades']} trades, {best_combo['pnl']:.2f} PnL)")

    # ===== STEP 4: Rolling WF Validation =====
    print("\n[4/6] Rolling Walk-Forward validation...")
    best_flags = make_filter_flags(best_combo["filters"])
    wf_result = run_rolling_wf(df, best_flags)
    print(f"      {wf_result['windows']} windows | {wf_result['trades']} trades")
    print(f"      PnL: {wf_result['pnl']:.2f} | Positive windows: {wf_result['positive_windows']}/{wf_result['windows']} ({wf_result['window_wr']:.1f}%)")

    wf_passed = wf_result["pnl"] > 0 and wf_result["window_wr"] >= 50

    # ===== STEP 5: Portfolio Backtest =====
    print("\n[5/6] Portfolio backtest ($1000, 1% risk)...")
    portfolio = run_portfolio_backtest(df, best_combo["filters"])
    print(f"      Trades: {portfolio['trades']} | WR: {portfolio['win_rate']:.1f}%")
    print(f"      PnL: ${portfolio['total_pnl']:.2f} | DD: ${portfolio['max_drawdown']:.2f}")
    print(f"      PF: {portfolio['profit_factor']:.2f}")

    # ===== STEP 6: Final Results =====
    print("\n[6/6] Saving results...")

    # Determine verdict
    if portfolio['total_pnl'] > 0 and wf_passed and not portfolio['stopped']:
        verdict = "PASS"
        recommendation = "TRADE"
    elif portfolio['total_pnl'] > 0 and not portfolio['stopped']:
        verdict = "MARGINAL"
        recommendation = "PAPER TRADE"
    else:
        verdict = "FAIL"
        recommendation = "DO NOT TRADE"

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "timestamp": datetime.now().isoformat(),
        "mode": "full_pipeline",
        "baseline": baseline,
        "discovery": {
            "tested": len(discovery_results),
            "best_combo": best_combo,
            "top_5": sorted(discovery_results, key=lambda x: x["pnl"], reverse=True)[:5],
        },
        "rolling_wf": wf_result,
        "portfolio": {
            "filters": best_combo["filters"],
            "initial": 1000.0,
            "final": portfolio['final_balance'],
            "pnl": portfolio['total_pnl'],
            "pnl_pct": portfolio['total_pnl_pct'],
            "trades": portfolio['trades'],
            "wins": portfolio['wins'],
            "losses": portfolio['losses'],
            "win_rate": portfolio['win_rate'],
            "max_dd": portfolio['max_drawdown'],
            "max_dd_pct": portfolio['max_drawdown_pct'],
            "pf": portfolio['profit_factor'],
            "stopped": portfolio['stopped'],
        },
        "verdict": verdict,
        "recommendation": recommendation,
    }

    # Save files
    with open(result_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    with open(result_dir / "trades.json", "w") as f:
        json.dump(portfolio.get('trades_list', []), f, indent=2, default=str)

    # Print summary
    summary = f"""
{'='*70}
FULL PIPELINE RESULT: {symbol} {timeframe}
{'='*70}

Period: {df.index[0].date()} to {df.index[-1].date()} ({days} days)

STEP 1 - BASELINE (regime only):
  Trades: {baseline['trades']} | WR: {baseline['wr']:.1f}% | PnL: {baseline['pnl']:.2f}

STEP 2 - FILTER DISCOVERY:
  Tested: {len(discovery_results)} combinations
  Best: {best_combo['name']}
        {best_combo['trades']} trades | {best_combo['wr']:.1f}% WR | {best_combo['pnl']:.2f} PnL

STEP 3 - ROLLING WALK-FORWARD:
  Windows: {wf_result['windows']} | Positive: {wf_result['positive_windows']} ({wf_result['window_wr']:.1f}%)
  Total PnL: {wf_result['pnl']:.2f}
  Status: {'PASSED' if wf_passed else 'FAILED'}

STEP 4 - PORTFOLIO BACKTEST ($1000, 1% risk, 10x leverage):
  Best Config: {', '.join(best_combo['filters'])}
  Final Balance: ${portfolio['final_balance']:.2f}
  PnL: ${portfolio['total_pnl']:.2f} ({portfolio['total_pnl_pct']:.2f}%)
  Trades: {portfolio['trades']} (W:{portfolio['wins']} L:{portfolio['losses']})
  Win Rate: {portfolio['win_rate']:.1f}%
  Max DD: ${portfolio['max_drawdown']:.2f} ({portfolio['max_drawdown_pct']:.2f}%)
  Profit Factor: {portfolio['profit_factor']:.2f}

{'='*70}
VERDICT: {verdict}
RECOMMENDATION: {recommendation}
{'='*70}

Results saved to: {result_dir}
"""
    print(summary)

    with open(result_dir / "summary.txt", "w") as f:
        f.write(summary)

    return result


def run_quick_test(symbol: str, timeframe: str, days: int = 365) -> Dict:
    """Run quick test with default config (no discovery)."""
    from core import set_backtest_mode
    set_backtest_mode(True)

    result_dir = get_result_dir(symbol, timeframe, "quick")

    print(f"\n{'='*70}")
    print(f"QUICK TEST: {symbol} {timeframe} ({days} days)")
    print(f"{'='*70}")
    print(f"Config: {DEFAULT_FILTERS} (fixed)")
    print(f"{'='*70}\n")

    # Fetch data
    print("[1/3] Fetching data...")
    df = fetch_data(symbol, timeframe, days)
    print(f"      {len(df)} candles")

    # Portfolio backtest
    print("\n[2/3] Portfolio backtest...")
    portfolio = run_portfolio_backtest(df, DEFAULT_FILTERS)
    print(f"      Trades: {portfolio['trades']} | WR: {portfolio['win_rate']:.1f}%")
    print(f"      PnL: ${portfolio['total_pnl']:.2f}")

    # Save
    print("\n[3/3] Saving...")
    verdict = "PASS" if portfolio['total_pnl'] > 0 and not portfolio['stopped'] else "FAIL"

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "mode": "quick",
        "filters": DEFAULT_FILTERS,
        "portfolio": {
            "pnl": portfolio['total_pnl'],
            "trades": portfolio['trades'],
            "win_rate": portfolio['win_rate'],
            "pf": portfolio['profit_factor'],
            "max_dd_pct": portfolio['max_drawdown_pct'],
        },
        "verdict": verdict,
    }

    with open(result_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"{symbol} {timeframe} | {verdict}")
    print(f"PnL: ${portfolio['total_pnl']:.2f} | Trades: {portfolio['trades']} | WR: {portfolio['win_rate']:.1f}%")
    print(f"{'='*70}")

    return result


def run_viz(symbol: str, timeframe: str):
    """Visualize trades from latest test."""
    from core.trade_visualizer import TradeVisualizer
    from core import get_client, calculate_indicators, set_backtest_mode
    import pandas as pd

    set_backtest_mode(True)

    latest = get_latest_result(symbol, timeframe)
    if not latest:
        print(f"No results for {symbol} {timeframe}")
        return

    trades_file = latest / "trades.json"
    if not trades_file.exists():
        print(f"No trades in {latest}")
        return

    with open(trades_file) as f:
        trades = json.load(f)

    if not trades:
        print("No trades to visualize")
        return

    print(f"\nVisualizing {len(trades)} trades from {latest}...")

    client = get_client()
    df = client.get_klines(symbol=symbol, interval=timeframe, limit=1000)
    df = calculate_indicators(df, timeframe=timeframe)

    viz_dir = latest / "charts"
    ensure_dir(viz_dir)
    visualizer = TradeVisualizer(output_dir=str(viz_dir))

    count = 0
    for trade in trades:
        try:
            viz_trade = {
                "signal_type": trade.get("signal_type", "LONG"),
                "entry_time": trade.get("entry_time", ""),
                "entry": trade.get("entry_price", 0),
                "tp": trade.get("tp_price", 0),
                "sl": trade.get("sl_price", 0),
                "exit_time": trade.get("exit_time", ""),
                "exit_price": trade.get("exit_price", 0),
                "exit_reason": trade.get("exit_reason", ""),
                "pnl": trade.get("pnl", 0),
                "status": "WON" if trade.get("win") else "LOST",
            }
            if visualizer.create_trade_chart(df, viz_trade, symbol, timeframe):
                count += 1
        except:
            pass

    print(f"Created {count} charts in {viz_dir}")


def run_report():
    """Show all test results."""
    print(f"\n{'='*70}")
    print("ALL TEST RESULTS")
    print(f"{'='*70}\n")

    if not RESULTS_DIR.exists():
        print("No results yet. Run: python run.py test BTCUSDT 15m --full")
        return

    results = []
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        rf = d / "result.json"
        if not rf.exists():
            continue
        with open(rf) as f:
            r = json.load(f)
        p = r.get("portfolio", {})
        results.append({
            "dir": d.name,
            "symbol": r.get("symbol", "?"),
            "tf": r.get("timeframe", "?"),
            "mode": r.get("mode", "?"),
            "trades": p.get("trades", 0),
            "pnl": p.get("pnl", 0),
            "wr": p.get("win_rate", 0),
            "verdict": r.get("verdict", "?"),
        })

    print(f"{'Symbol':<10} {'TF':<6} {'Mode':<8} {'Trades':<8} {'PnL':<12} {'WR%':<8} {'Verdict'}")
    print("-" * 75)

    for r in results[:15]:
        pnl = f"${r['pnl']:.2f}" if r['pnl'] else "$0"
        if r['pnl'] and r['pnl'] > 0:
            pnl = f"+{pnl}"
        print(f"{r['symbol']:<10} {r['tf']:<6} {r['mode']:<8} {r['trades']:<8} {pnl:<12} {r['wr']:.1f}%{'':<4} {r['verdict']}")


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py test BTCUSDT 15m --full    # Full pipeline (recommended)
  python run.py test BTCUSDT 15m           # Quick test (fixed config)
  python run.py test BTCUSDT 15m --quick   # 90-day quick test
  python run.py viz BTCUSDT 15m            # Visualize trades
  python run.py report                     # Show all results
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # test
    test_p = subparsers.add_parser("test", help="Run test")
    test_p.add_argument("symbol", help="Symbol (e.g., BTCUSDT)")
    test_p.add_argument("timeframe", help="Timeframe (e.g., 15m)")
    test_p.add_argument("--days", type=int, default=365)
    test_p.add_argument("--full", action="store_true", help="Full pipeline (discovery + WF + portfolio)")
    test_p.add_argument("--quick", action="store_true", help="90-day quick test")

    # viz
    viz_p = subparsers.add_parser("viz", help="Visualize trades")
    viz_p.add_argument("symbol")
    viz_p.add_argument("timeframe")

    # report
    subparsers.add_parser("report", help="Show all results")

    args = parser.parse_args()

    if args.command == "test":
        days = 90 if args.quick else args.days
        if args.full:
            run_full_pipeline(args.symbol, args.timeframe, days)
        else:
            run_quick_test(args.symbol, args.timeframe, days)
    elif args.command == "viz":
        run_viz(args.symbol, args.timeframe)
    elif args.command == "report":
        run_report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
