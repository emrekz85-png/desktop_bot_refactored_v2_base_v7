#!/usr/bin/env python3
"""
Trade Visualizer Demo Script

Quick demonstration of the Trade Visualization System.
Generates sample charts from the most recent Rolling WF run.

Usage:
    python demo_trade_visualizer.py

Author: Claude (Anthropic)
Date: December 28, 2024
"""

import os
import sys
import glob

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.trade_visualizer import TradeVisualizer
from run_trade_visualizer import parse_trades_from_detailed_txt


def find_latest_wf_run():
    """Find the most recent Rolling WF run directory."""
    pattern = "data/rolling_wf_runs/v*"
    runs = glob.glob(pattern)

    if not runs:
        print("No Rolling WF runs found in data/rolling_wf_runs/")
        return None

    # Sort by modification time
    runs.sort(key=os.path.getmtime, reverse=True)

    for run_dir in runs:
        trades_file = os.path.join(run_dir, "trades_detailed.txt")
        if os.path.exists(trades_file):
            return trades_file

    return None


def demo_single_trade(viz, trades):
    """Demo: Visualize first trade."""
    print("\n" + "="*80)
    print("DEMO 1: Single Trade Visualization")
    print("="*80)

    if not trades:
        print("No trades available")
        return

    trade = trades[0]
    print(f"\nVisualizing first trade:")
    print(f"  Symbol: {trade['symbol']}")
    print(f"  Type: {trade['type']}")
    print(f"  Entry: {trade['open_time_utc']} @ ${trade['entry']:,.2f}")
    print(f"  PnL: ${trade['pnl']:+.2f} | R: {trade['r_multiple']:+.2f}x")
    print(f"  Result: {trade['status']}")

    chart_path = viz.visualize_trade(trade)

    if chart_path:
        print(f"\n✅ Chart generated: {chart_path}")
        print(f"   File size: {os.path.getsize(chart_path) / 1024:.1f} KB")
    else:
        print("\n❌ Failed to generate chart")


def demo_win_vs_loss(viz, trades):
    """Demo: Compare a winning and losing trade."""
    print("\n" + "="*80)
    print("DEMO 2: Win vs Loss Comparison")
    print("="*80)

    # Find first win and first loss
    winning_trade = None
    losing_trade = None

    for trade in trades:
        if trade['status'] == 'WON' and winning_trade is None:
            winning_trade = trade
        elif trade['status'] == 'LOST' and losing_trade is None:
            losing_trade = trade

        if winning_trade and losing_trade:
            break

    if winning_trade:
        print(f"\n📈 WINNING TRADE:")
        print(f"  {winning_trade['symbol']} {winning_trade['type']}")
        print(f"  PnL: ${winning_trade['pnl']:+.2f} | R: {winning_trade['r_multiple']:+.2f}x")

        win_chart = viz.visualize_trade(winning_trade)
        if win_chart:
            print(f"  Chart: {win_chart}")

    if losing_trade:
        print(f"\n📉 LOSING TRADE:")
        print(f"  {losing_trade['symbol']} {losing_trade['type']}")
        print(f"  PnL: ${losing_trade['pnl']:+.2f} | R: {losing_trade['r_multiple']:+.2f}x")

        loss_chart = viz.visualize_trade(losing_trade)
        if loss_chart:
            print(f"  Chart: {loss_chart}")


def demo_batch_processing(viz, trades):
    """Demo: Batch process first 5 trades."""
    print("\n" + "="*80)
    print("DEMO 3: Batch Processing (First 5 Trades)")
    print("="*80)

    sample_trades = trades[:5]

    print(f"\nProcessing {len(sample_trades)} trades...")

    chart_paths = viz.visualize_all_trades(sample_trades)

    print(f"\n✅ Generated {len(chart_paths)} charts:")
    for path in chart_paths:
        filename = os.path.basename(path)
        size_kb = os.path.getsize(path) / 1024
        print(f"  - {filename} ({size_kb:.1f} KB)")


def demo_summary_stats(trades):
    """Demo: Print trade statistics."""
    print("\n" + "="*80)
    print("DEMO 4: Trade Statistics Summary")
    print("="*80)

    total = len(trades)
    wins = sum(1 for t in trades if t['status'] == 'WON')
    losses = total - wins

    total_pnl = sum(t['pnl'] for t in trades)
    avg_win = sum(t['pnl'] for t in trades if t['status'] == 'WON') / wins if wins > 0 else 0
    avg_loss = sum(t['pnl'] for t in trades if t['status'] == 'LOST') / losses if losses > 0 else 0

    print(f"\nTotal Trades: {total}")
    print(f"Wins: {wins} ({wins/total*100:.1f}%)")
    print(f"Losses: {losses} ({losses/total*100:.1f}%)")
    print(f"\nTotal PnL: ${total_pnl:+.2f}")
    print(f"Avg Win: ${avg_win:+.2f}")
    print(f"Avg Loss: ${avg_loss:+.2f}")
    print(f"Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "N/A")

    # Symbol breakdown
    symbols = {}
    for trade in trades:
        symbol = trade['symbol']
        if symbol not in symbols:
            symbols[symbol] = {'wins': 0, 'losses': 0, 'pnl': 0}

        if trade['status'] == 'WON':
            symbols[symbol]['wins'] += 1
        else:
            symbols[symbol]['losses'] += 1

        symbols[symbol]['pnl'] += trade['pnl']

    print(f"\nSymbol Breakdown:")
    for symbol, stats in sorted(symbols.items()):
        total_sym = stats['wins'] + stats['losses']
        wr = stats['wins'] / total_sym * 100 if total_sym > 0 else 0
        print(f"  {symbol:<15} {stats['wins']:>2}W/{stats['losses']:>2}L ({wr:>5.1f}%)  PnL: ${stats['pnl']:>+8.2f}")


def main():
    """Main demo entry point."""
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "TRADE VISUALIZER DEMO" + " "*37 + "║")
    print("║" + " "*15 + "Professional TradingView-Style Charts" + " "*26 + "║")
    print("╚" + "="*78 + "╝")

    # Find latest run
    print("\nSearching for latest Rolling WF run...")
    trades_file = find_latest_wf_run()

    if not trades_file:
        print("❌ No trades_detailed.txt file found")
        print("\nPlease run a Rolling WF test first:")
        print("  python run_rolling_wf_test.py")
        sys.exit(1)

    print(f"✅ Found: {trades_file}")

    # Parse trades
    print("\nParsing trades...")
    trades = parse_trades_from_detailed_txt(trades_file)

    if not trades:
        print("❌ No trades parsed from file")
        sys.exit(1)

    print(f"✅ Parsed {len(trades)} trades")

    # Initialize visualizer
    output_dir = "demo_charts"
    viz = TradeVisualizer(output_dir=output_dir, dpi=300)

    print(f"\nOutput directory: {output_dir}/")

    # Run demos
    demo_summary_stats(trades)
    demo_single_trade(viz, trades)
    demo_win_vs_loss(viz, trades)
    demo_batch_processing(viz, trades)

    # Final summary
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)

    chart_count = len(glob.glob(f"{output_dir}/*.png"))
    print(f"\nGenerated {chart_count} charts in {output_dir}/")
    print("\nNext steps:")
    print("  1. Open charts in your image viewer")
    print("  2. Analyze trade quality and indicator behavior")
    print("  3. Use insights to refine strategy parameters")
    print("\nFor full usage, see:")
    print("  python run_trade_visualizer.py --help")
    print("  TRADE_VISUALIZER_README.md")
    print()


if __name__ == '__main__':
    main()
