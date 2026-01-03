#!/usr/bin/env python3
"""
Simplified Test Runner

Tek komutla tum testleri calistir ve sonuclari tek yerde topla.

Usage:
    python run.py test BTCUSDT 15m          # Full pipeline + portfolio test
    python run.py test BTCUSDT 15m --quick  # Quick 90-day test
    python run.py test --all                # Test all recommended symbols
    python run.py viz BTCUSDT 15m           # Visualize latest trades
    python run.py report                    # Summary of all tests
    python run.py list                      # List all test results
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Constants
RESULTS_DIR = Path("data/results")

# Recommended symbols (from SYMBOL_SETTINGS.md)
RECOMMENDED_SYMBOLS = ["BTCUSDT", "ETHUSDT"]  # Only tested/validated ones
RECOMMENDED_TF = "15m"

# Best filter config (validated)
BEST_FILTERS = ["regime", "at_flat_filter", "min_sl_filter"]


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_result_dir(symbol: str, timeframe: str) -> Path:
    """Get or create result directory for symbol/timeframe."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_DIR / f"{symbol}_{timeframe}_{timestamp}"
    ensure_dir(result_dir)
    return result_dir


def get_latest_result(symbol: str, timeframe: str) -> Path:
    """Get latest result directory for symbol/timeframe."""
    pattern = f"{symbol}_{timeframe}_*"
    matches = sorted(RESULTS_DIR.glob(pattern), reverse=True)
    return matches[0] if matches else None


def run_test(symbol: str, timeframe: str, days: int = 365, quick: bool = False) -> dict:
    """
    Run full test pipeline for a symbol/timeframe.

    Steps:
    1. Fetch data
    2. Find signals (check_core_signal)
    3. Apply filters (BEST_FILTERS)
    4. Portfolio backtest (realistic sizing)
    5. Save consolidated results
    """
    from core import get_client, calculate_indicators, set_backtest_mode
    from core.at_scenario_analyzer import check_core_signal
    from core.simple_portfolio import PortfolioConfig, run_portfolio_backtest
    from runners.run_filter_combo_test import apply_filters
    import pandas as pd

    set_backtest_mode(True)

    if quick:
        days = 90

    result_dir = get_result_dir(symbol, timeframe)

    print(f"\n{'='*70}")
    print(f"TEST: {symbol} {timeframe} ({days} days)")
    print(f"{'='*70}")
    print(f"Filters: {BEST_FILTERS}")
    print(f"Output: {result_dir}")
    print(f"{'='*70}\n")

    # ===== STEP 1: Fetch Data =====
    print("[1/4] Fetching data...")
    client = get_client()

    candles_map = {"5m": 288, "15m": 96, "1h": 24, "4h": 6}
    candles = min(days * candles_map.get(timeframe, 96), 35000)

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

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    df = calculate_indicators(df, timeframe=timeframe)

    print(f"    {len(df)} candles: {df.index[0].date()} to {df.index[-1].date()}")

    # ===== STEP 2: Find & Filter Signals =====
    print("\n[2/4] Finding signals...")

    filter_flags = {f"use_{f.replace('regime', 'regime_filter')}": True for f in BEST_FILTERS}
    # Fix the key names
    filter_flags = {
        "use_regime_filter": "regime" in BEST_FILTERS,
        "use_at_flat_filter": "at_flat_filter" in BEST_FILTERS,
        "use_min_sl_filter": "min_sl_filter" in BEST_FILTERS,
        "use_at_binary": False,
        "use_adx_filter": False,
        "use_ssl_touch": False,
        "use_rsi_filter": False,
        "use_pbema_distance": False,
        "use_overlap_check": False,
        "use_body_position": False,
        "use_wick_rejection": False,
    }

    signals = []
    signals_raw = 0
    signals_filtered = 0
    last_idx = -5

    for i in range(60, len(df) - 10):
        if i - last_idx < 5:
            continue

        signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)
        if signal_type is None:
            continue

        signals_raw += 1
        last_idx = i

        passed, filter_reason = apply_filters(
            df=df, index=i, signal_type=signal_type,
            entry_price=entry, sl_price=sl, **filter_flags
        )

        if not passed:
            signals_filtered += 1
            continue

        signals.append({
            "idx": i,
            "time": str(df.index[i]),
            "signal_type": signal_type,
            "entry": entry,
            "tp": tp,
            "sl": sl,
        })

    filter_rate = signals_filtered / signals_raw * 100 if signals_raw > 0 else 0
    print(f"    Raw: {signals_raw} → Filtered: {signals_filtered} ({filter_rate:.1f}%) → Final: {len(signals)}")

    # ===== STEP 3: Portfolio Backtest =====
    print("\n[3/4] Portfolio backtest ($1000, 1% risk)...")

    config = PortfolioConfig(
        initial_balance=1000.0,
        risk_per_trade_pct=0.01,
        leverage=10,
        max_position_pct=0.10,
        slippage_pct=0.0005,
        fee_pct=0.0007,
        total_dd_limit=0.25,
    )

    portfolio = run_portfolio_backtest(df, signals, config)

    print(f"    Trades: {portfolio['trades']} | WR: {portfolio['win_rate']:.1f}%")
    print(f"    PnL: ${portfolio['total_pnl']:.2f} | DD: ${portfolio['max_drawdown']:.2f}")
    print(f"    PF: {portfolio['profit_factor']:.2f}")

    # ===== STEP 4: Save Results =====
    print("\n[4/4] Saving results...")

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "timestamp": datetime.now().isoformat(),
        "filters": BEST_FILTERS,
        "signals": {
            "raw": signals_raw,
            "filtered": signals_filtered,
            "final": len(signals),
        },
        "portfolio": {
            "initial": config.initial_balance,
            "final": portfolio['final_balance'],
            "pnl": portfolio['total_pnl'],
            "pnl_pct": portfolio['total_pnl_pct'],
            "trades": portfolio['trades'],
            "wins": portfolio['wins'],
            "losses": portfolio['losses'],
            "win_rate": portfolio['win_rate'],
            "avg_win": portfolio['avg_win'],
            "avg_loss": portfolio['avg_loss'],
            "max_dd": portfolio['max_drawdown'],
            "max_dd_pct": portfolio['max_drawdown_pct'],
            "pf": portfolio['profit_factor'],
            "stopped": portfolio['stopped'],
        },
    }

    # Save files
    with open(result_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    with open(result_dir / "signals.json", "w") as f:
        json.dump(signals, f, indent=2, default=str)

    with open(result_dir / "trades.json", "w") as f:
        json.dump(portfolio.get('trades_list', []), f, indent=2, default=str)

    # Summary
    verdict = "PASS" if portfolio['total_pnl'] > 0 and not portfolio['stopped'] else "FAIL"
    summary = f"""
{'='*70}
{symbol} {timeframe} | {days} days | {verdict}
{'='*70}
Period: {df.index[0].date()} to {df.index[-1].date()}
Filters: {', '.join(BEST_FILTERS)}

Signals: {signals_raw} raw → {len(signals)} final ({filter_rate:.1f}% filtered)

Portfolio ($1000 start, 1% risk, 10x leverage):
  Final: ${portfolio['final_balance']:.2f}
  PnL: ${portfolio['total_pnl']:.2f} ({portfolio['total_pnl_pct']:.2f}%)
  Trades: {portfolio['trades']} (W:{portfolio['wins']} L:{portfolio['losses']})
  WR: {portfolio['win_rate']:.1f}%
  Avg Win: ${portfolio['avg_win']:.2f} | Avg Loss: ${portfolio['avg_loss']:.2f}
  Max DD: ${portfolio['max_drawdown']:.2f} ({portfolio['max_drawdown_pct']:.2f}%)
  PF: {portfolio['profit_factor']:.2f}

Results: {result_dir}
{'='*70}
"""
    print(summary)

    with open(result_dir / "summary.txt", "w") as f:
        f.write(summary)

    return result


def run_test_all(days: int = 365, quick: bool = False):
    """Test all recommended symbols."""
    print(f"\n{'='*70}")
    print(f"TESTING ALL RECOMMENDED SYMBOLS")
    print(f"{'='*70}")
    print(f"Symbols: {RECOMMENDED_SYMBOLS}")
    print(f"Timeframe: {RECOMMENDED_TF}")
    print(f"{'='*70}\n")

    results = []
    for symbol in RECOMMENDED_SYMBOLS:
        try:
            result = run_test(symbol, RECOMMENDED_TF, days, quick)
            results.append(result)
        except Exception as e:
            print(f"ERROR testing {symbol}: {e}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Symbol':<12} {'Trades':<8} {'PnL':<12} {'WR%':<8} {'PF':<6}")
    print("-" * 50)

    for r in results:
        p = r['portfolio']
        pnl_str = f"${p['pnl']:.2f}"
        if p['pnl'] > 0:
            pnl_str = f"+{pnl_str}"
        print(f"{r['symbol']:<12} {p['trades']:<8} {pnl_str:<12} {p['win_rate']:.1f}%{'':<4} {p['pf']:.2f}")

    print(f"{'='*70}")


def run_viz(symbol: str, timeframe: str):
    """Visualize trades from latest test."""
    from core.trade_visualizer import TradeVisualizer
    from core import get_client, calculate_indicators, set_backtest_mode
    import pandas as pd

    set_backtest_mode(True)

    latest = get_latest_result(symbol, timeframe)
    if not latest:
        print(f"No results for {symbol} {timeframe}. Run: python run.py test {symbol} {timeframe}")
        return

    trades_file = latest / "trades.json"
    if not trades_file.exists():
        print(f"No trades file in {latest}")
        return

    with open(trades_file) as f:
        trades = json.load(f)

    if not trades:
        print("No trades to visualize")
        return

    print(f"\n{'='*70}")
    print(f"VISUALIZE: {symbol} {timeframe}")
    print(f"{'='*70}")
    print(f"Source: {latest}")
    print(f"Trades: {len(trades)}")

    # Fetch fresh data
    print("\nFetching data...")
    client = get_client()
    df = client.get_klines(symbol=symbol, interval=timeframe, limit=1000)
    df = calculate_indicators(df, timeframe=timeframe)

    # Output directory
    viz_dir = latest / "charts"
    ensure_dir(viz_dir)

    # Use TradeVisualizer
    visualizer = TradeVisualizer(output_dir=str(viz_dir))

    # Generate charts for each trade
    wins = 0
    losses = 0

    for i, trade in enumerate(trades):
        try:
            # Parse entry time
            entry_time_str = trade.get("entry_time", "")
            if not entry_time_str:
                continue

            # Find entry in dataframe
            entry_dt = pd.to_datetime(entry_time_str)

            # Create trade dict for visualizer
            viz_trade = {
                "signal_type": trade.get("signal_type", "LONG"),
                "entry_time": entry_time_str,
                "entry": trade.get("entry_price", 0),
                "tp": trade.get("tp_price", 0),
                "sl": trade.get("sl_price", 0),
                "exit_time": trade.get("exit_time", ""),
                "exit_price": trade.get("exit_price", 0),
                "exit_reason": trade.get("exit_reason", ""),
                "pnl": trade.get("pnl", 0),
                "status": "WON" if trade.get("win") else "LOST",
            }

            # Generate chart
            chart_file = visualizer.create_trade_chart(
                df=df,
                trade=viz_trade,
                symbol=symbol,
                timeframe=timeframe,
            )

            if chart_file:
                if trade.get("win"):
                    wins += 1
                else:
                    losses += 1

        except Exception as e:
            print(f"  Error on trade {i}: {e}")

    print(f"\nCreated {wins + losses} charts in {viz_dir}")
    print(f"  WIN: {wins} | LOSS: {losses}")


def run_report():
    """Show summary of all test results."""
    print(f"\n{'='*70}")
    print("ALL TEST RESULTS")
    print(f"{'='*70}\n")

    if not RESULTS_DIR.exists():
        print("No results. Run: python run.py test BTCUSDT 15m")
        return

    results = []
    for result_dir in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if not result_dir.is_dir():
            continue

        result_file = result_dir / "result.json"
        if not result_file.exists():
            continue

        with open(result_file) as f:
            r = json.load(f)

        p = r.get("portfolio", {})
        results.append({
            "dir": result_dir.name,
            "symbol": r.get("symbol", "?"),
            "tf": r.get("timeframe", "?"),
            "trades": p.get("trades", 0),
            "pnl": p.get("pnl", 0),
            "wr": p.get("win_rate", 0),
            "pf": p.get("pf", 0),
            "dd": p.get("max_dd_pct", 0),
        })

    if not results:
        print("No results found")
        return

    print(f"{'Symbol':<10} {'TF':<6} {'Trades':<8} {'PnL':<12} {'WR%':<8} {'PF':<6} {'DD%':<8}")
    print("-" * 70)

    for r in results[:20]:
        pnl_str = f"${r['pnl']:.2f}"
        if r['pnl'] > 0:
            pnl_str = f"+{pnl_str}"
        print(f"{r['symbol']:<10} {r['tf']:<6} {r['trades']:<8} {pnl_str:<12} "
              f"{r['wr']:.1f}%{'':<4} {r['pf']:.2f}{'':<3} {r['dd']:.1f}%")

    if len(results) > 20:
        print(f"\n... and {len(results) - 20} more")


def run_list():
    """List all result directories."""
    print(f"\n{'='*70}")
    print("RESULT DIRECTORIES")
    print(f"{'='*70}\n")

    if not RESULTS_DIR.exists():
        print("No results")
        return

    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if d.is_dir():
            files = list(d.iterdir())
            print(f"{d.name}/  ({len(files)} files)")


def main():
    parser = argparse.ArgumentParser(
        description="Simplified Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py test BTCUSDT 15m          # Full 1-year test
  python run.py test BTCUSDT 15m --quick  # Quick 90-day test
  python run.py test --all                # Test all recommended symbols
  python run.py viz BTCUSDT 15m           # Visualize trades
  python run.py report                    # Summary of all results
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # test
    test_p = subparsers.add_parser("test", help="Run test")
    test_p.add_argument("symbol", nargs="?", help="Symbol (e.g., BTCUSDT)")
    test_p.add_argument("timeframe", nargs="?", help="Timeframe (e.g., 15m)")
    test_p.add_argument("--days", type=int, default=365)
    test_p.add_argument("--quick", action="store_true", help="90-day test")
    test_p.add_argument("--all", action="store_true", help="Test all recommended")

    # viz
    viz_p = subparsers.add_parser("viz", help="Visualize trades")
    viz_p.add_argument("symbol", help="Symbol")
    viz_p.add_argument("timeframe", help="Timeframe")

    # report
    subparsers.add_parser("report", help="Show summary")

    # list
    subparsers.add_parser("list", help="List results")

    args = parser.parse_args()

    if args.command == "test":
        if args.all:
            run_test_all(args.days, args.quick)
        elif args.symbol and args.timeframe:
            run_test(args.symbol, args.timeframe, args.days, args.quick)
        else:
            print("Usage: python run.py test BTCUSDT 15m")
            print("       python run.py test --all")
    elif args.command == "viz":
        run_viz(args.symbol, args.timeframe)
    elif args.command == "report":
        run_report()
    elif args.command == "list":
        run_list()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
