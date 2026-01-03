#!/usr/bin/env python3
"""
Portfolio Backtest Runner

Tests the simple portfolio system with pipeline signals.
Compares $1000 portfolio results vs fixed $35 position size.
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import get_client, calculate_indicators, set_backtest_mode
from core.at_scenario_analyzer import check_core_signal
from core.simple_portfolio import SimplePortfolio, PortfolioConfig, run_portfolio_backtest
from runners.run_filter_combo_test import apply_filters as combo_apply_filters

# Enable backtest mode
set_backtest_mode(True)

# Filter flags (best config from pipeline + min_sl_filter from deep analysis)
BEST_FILTERS = ["regime", "at_flat_filter", "min_sl_filter"]


def fetch_data(symbol: str, timeframe: str, days: int = 365):
    """Fetch and prepare data (copied from run_full_pipeline.py)."""
    import pandas as pd
    import requests

    print(f"Fetching {symbol} {timeframe} data ({days} days)...")
    client = get_client()

    candles = {
        "5m": days * 288,
        "15m": days * 96,
        "1h": days * 24,
        "4h": days * 6,
    }.get(timeframe, days * 96)

    candles = min(candles, 35000)

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

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    df = calculate_indicators(df, timeframe=timeframe)

    return df


def collect_signals(df, filter_list: list) -> list:
    """Collect all signals that pass filters."""
    signals = []
    last_idx = -5

    # Convert filter list to flags dict for combo_apply_filters
    filter_flags = {
        "use_regime_filter": "regime" in filter_list,
        "use_at_binary": "at_binary" in filter_list,
        "use_at_flat_filter": "at_flat_filter" in filter_list,
        "use_adx_filter": "adx_filter" in filter_list,
        "use_ssl_touch": "ssl_touch" in filter_list,
        "use_rsi_filter": "rsi_filter" in filter_list,
        "use_pbema_distance": "pbema_distance" in filter_list,
        "use_overlap_check": "overlap_check" in filter_list,
        "use_body_position": "body_position" in filter_list,
        "use_wick_rejection": "wick_rejection" in filter_list,
        "use_min_sl_filter": "min_sl_filter" in filter_list,
    }

    for i in range(60, len(df) - 10):
        if i - last_idx < 5:
            continue

        signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)
        if signal_type is None:
            continue

        last_idx = i

        # Use combo_apply_filters from run_filter_combo_test
        passed, filter_reason = combo_apply_filters(
            df=df,
            index=i,
            signal_type=signal_type,
            entry_price=entry,
            sl_price=sl,
            **filter_flags
        )
        if not passed:
            continue

        signals.append({
            "idx": i,
            "signal_type": signal_type,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "reason": reason,
        })

    return signals


def run_fixed_size_backtest(df, signals: list, position_size: float = 35.0) -> dict:
    """Run simple fixed-size backtest for comparison."""
    trades = []
    total_pnl = 0

    slippage = 0.0005
    fee = 0.0007

    for signal in signals:
        idx = signal["idx"]
        signal_type = signal["signal_type"]
        entry = signal["entry"]
        tp = signal["tp"]
        sl = signal["sl"]

        is_long = signal_type == "LONG"

        # Apply entry slippage
        if is_long:
            actual_entry = entry * (1 + slippage)
        else:
            actual_entry = entry * (1 - slippage)

        # Simulate trade
        exit_price = None
        exit_reason = None

        for j in range(idx + 1, len(df)):
            candle = df.iloc[j]
            high, low = float(candle["high"]), float(candle["low"])

            if is_long:
                if low <= sl:
                    exit_price = sl * (1 - slippage)
                    exit_reason = "SL"
                    break
                if high >= tp:
                    exit_price = tp * (1 - slippage)
                    exit_reason = "TP"
                    break
            else:
                if high >= sl:
                    exit_price = sl * (1 + slippage)
                    exit_reason = "SL"
                    break
                if low <= tp:
                    exit_price = tp * (1 + slippage)
                    exit_reason = "TP"
                    break

        if exit_price is None:
            exit_price = float(df.iloc[-1]["close"])
            exit_reason = "EOD"

        # Calculate PnL
        if is_long:
            pnl = (exit_price - actual_entry) / actual_entry * position_size
        else:
            pnl = (actual_entry - exit_price) / actual_entry * position_size

        # Subtract fees
        pnl -= position_size * fee * 2

        trades.append({
            "pnl": pnl,
            "win": pnl > 0,
            "exit_reason": exit_reason,
        })
        total_pnl += pnl

    wins = sum(1 for t in trades if t["win"])

    return {
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) * 100 if trades else 0,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(trades) if trades else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Portfolio Backtest Test")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--risk", type=float, default=0.01, help="Risk per trade (0.01 = 1%%)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"PORTFOLIO BACKTEST: {args.symbol} {args.timeframe}")
    print(f"{'='*70}")

    # Fetch data
    df = fetch_data(args.symbol, args.timeframe, args.days)
    if df.empty:
        print("ERROR: No data")
        return

    print(f"Data: {len(df)} candles")

    # Collect signals
    print(f"\nCollecting signals with filters: {BEST_FILTERS}")
    signals = collect_signals(df, BEST_FILTERS)
    print(f"Total signals: {len(signals)}")

    if not signals:
        print("No signals found!")
        return

    # Run fixed-size backtest
    print(f"\n{'='*70}")
    print("FIXED SIZE BACKTEST ($35 per trade)")
    print(f"{'='*70}")

    fixed_result = run_fixed_size_backtest(df, signals, position_size=35.0)
    print(f"Trades:   {fixed_result['trades']}")
    print(f"Wins:     {fixed_result['wins']} ({fixed_result['win_rate']:.1f}%)")
    print(f"PnL:      ${fixed_result['total_pnl']:.2f}")
    print(f"Avg PnL:  ${fixed_result['avg_pnl']:.4f}")

    # Run portfolio backtest
    print(f"\n{'='*70}")
    print(f"PORTFOLIO BACKTEST (${args.balance}, {args.risk*100:.1f}% risk)")
    print(f"{'='*70}")

    config = PortfolioConfig(
        initial_balance=args.balance,
        risk_per_trade_pct=args.risk,
        leverage=10,
        max_position_pct=0.10,
        slippage_pct=0.0005,
        fee_pct=0.0007,
        total_dd_limit=0.25,
    )

    portfolio_result = run_portfolio_backtest(df, signals, config)

    print(f"Initial:  ${config.initial_balance:.2f}")
    print(f"Final:    ${portfolio_result['final_balance']:.2f}")
    print(f"PnL:      ${portfolio_result['total_pnl']:.2f} ({portfolio_result['total_pnl_pct']:.2f}%)")
    print(f"Trades:   {portfolio_result['trades']}")
    print(f"Wins:     {portfolio_result['wins']} ({portfolio_result['win_rate']:.1f}%)")
    print(f"Avg Win:  ${portfolio_result['avg_win']:.2f}")
    print(f"Avg Loss: ${portfolio_result['avg_loss']:.2f}")
    print(f"Max DD:   ${portfolio_result['max_drawdown']:.2f} ({portfolio_result['max_drawdown_pct']:.2f}%)")
    print(f"PF:       {portfolio_result['profit_factor']:.2f}")

    if portfolio_result['stopped']:
        print(f"STOPPED:  {portfolio_result['stop_reason']}")

    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    print(f"\n{'Metric':<20} {'Fixed $35':<15} {'Portfolio':<15}")
    print("-" * 50)
    print(f"{'Trades':<20} {fixed_result['trades']:<15} {portfolio_result['trades']:<15}")
    print(f"{'Win Rate':<20} {fixed_result['win_rate']:.1f}%{'':<10} {portfolio_result['win_rate']:.1f}%")
    print(f"{'Total PnL':<20} ${fixed_result['total_pnl']:<13.2f} ${portfolio_result['total_pnl']:.2f}")

    # Risk-adjusted comparison
    if portfolio_result['max_drawdown'] > 0:
        portfolio_return_dd = portfolio_result['total_pnl'] / portfolio_result['max_drawdown']
        print(f"{'Return/DD':<20} {'N/A':<15} {portfolio_return_dd:.2f}")

    # Save results
    if args.save:
        output_dir = f"data/portfolio_tests/{args.symbol}"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result = {
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "days": args.days,
            "timestamp": timestamp,
            "signals_count": len(signals),
            "fixed_size": fixed_result,
            "portfolio": portfolio_result,
            "config": {
                "initial_balance": config.initial_balance,
                "risk_per_trade_pct": config.risk_per_trade_pct,
                "leverage": config.leverage,
            }
        }

        report_file = f"{output_dir}/portfolio_test_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved: {report_file}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
