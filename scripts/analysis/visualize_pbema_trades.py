#!/usr/bin/env python3
"""
Visualize PBEMA v2 trades for verification.
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import get_client, calculate_indicators, set_backtest_mode
from core.trade_visualizer import TradeVisualizer
from pathlib import Path
import pandas as pd


def main():
    set_backtest_mode(True)

    # Load trades from latest PBEMA results
    results_dir = Path("data/results")
    pbema_dirs = sorted([d for d in results_dir.iterdir() if "pbema_v2" in d.name], reverse=True)

    if not pbema_dirs:
        print("No PBEMA results found!")
        return

    latest = pbema_dirs[0]
    trades_file = latest / "trades.json"

    if not trades_file.exists():
        print(f"No trades.json in {latest}")
        return

    with open(trades_file) as f:
        trades = json.load(f)

    print(f"Loaded {len(trades)} trades from {latest}")

    # Visualize ALL trades
    print(f"\nVisualizing ALL {len(trades)} trades...")

    # Create output directory
    output_dir = latest / "charts"
    output_dir.mkdir(exist_ok=True)

    # Initialize visualizer
    visualizer = TradeVisualizer(output_dir=str(output_dir))

    # Visualize trades
    count = 0
    for trade in trades:
        try:
            entry_time = trade.get("entry_time", "")
            if not entry_time:
                continue

            # Prepare trade dict for visualizer
            viz_trade = {
                "symbol": "BTCUSDT-15m",
                "timeframe": "15m",
                "type": trade.get("signal_type", "LONG"),
                "open_time_utc": entry_time,
                "entry": trade.get("entry_price", 0),
                "tp": trade.get("tp_price", 0),
                "sl": trade.get("sl_price", 0),
                "close_time_utc": trade.get("exit_time", ""),
                "close_price": trade.get("exit_price", 0),
                "exit_reason": trade.get("exit_type", ""),
                "pnl": trade.get("pnl", 0),
                "r_multiple": trade.get("pnl", 0) / 10.0,  # $10 risk per trade
                "status": "WON" if trade.get("win") else "LOST",
            }

            chart_path = visualizer.visualize_trade(viz_trade)
            if chart_path:
                count += 1
                status = "WIN" if trade.get("win") else "LOSS"
                if count % 20 == 0 or count <= 5:
                    print(f"  [{count}/{len(trades)}] {entry_time[:16]} {status} ${trade.get('pnl', 0):.2f}")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nCreated {count} charts in {output_dir}")

    # Also print summary stats
    wins = [t for t in trades if t.get("win")]
    losses = [t for t in trades if not t.get("win")]

    print("\n" + "="*60)
    print("PBEMA v2 TRADE SUMMARY")
    print("="*60)
    print(f"Total trades: {len(trades)}")
    print(f"Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
    print(f"Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")

    total_pnl = sum(t.get("pnl", 0) for t in trades)
    print(f"Total PnL: ${total_pnl:.2f}")

    # Exit type breakdown
    exit_types = {}
    for t in trades:
        et = t.get("exit_type", "UNKNOWN")
        exit_types[et] = exit_types.get(et, 0) + 1

    print(f"\nExit Types:")
    for et, et_count in sorted(exit_types.items()):
        print(f"  {et}: {et_count}")


if __name__ == "__main__":
    main()
