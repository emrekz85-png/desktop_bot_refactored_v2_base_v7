#!/usr/bin/env python3
"""
Regenerate PBEMA Retest Charts with Correct Format

This script regenerates charts for PBEMA Retest trades using TradeVisualizer
after the trade data has been converted to the correct format.

Usage:
    python tools/regenerate_pbema_charts.py data/results/PBEMA_RETEST_trades/trades.json
    python tools/regenerate_pbema_charts.py data/results/PBEMA_RETEST_trades/trades.json --sample 10
"""

import json
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.trade_visualizer import TradeVisualizer


def main():
    parser = argparse.ArgumentParser(description='Regenerate PBEMA Retest Charts')
    parser.add_argument('trades_file', help='Path to trades.json file')
    parser.add_argument('--sample', type=int, help='Only process first N trades (for testing)')
    parser.add_argument('--output-dir', help='Override output directory')
    args = parser.parse_args()

    trades_path = Path(args.trades_file)

    if not trades_path.exists():
        print(f"Error: Trades file not found: {trades_path}")
        sys.exit(1)

    print("=" * 80)
    print("PBEMA RETEST CHART REGENERATION")
    print("=" * 80)
    print(f"Trades file: {trades_path}")
    print()

    # Load trades
    with open(trades_path, 'r') as f:
        trades = json.load(f)

    print(f"Loaded {len(trades)} trades")

    # Sample if requested
    if args.sample:
        trades = trades[:args.sample]
        print(f"Processing sample of {len(trades)} trades")

    print()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(trades_path.parent / "charts_regenerated")

    print(f"Output directory: {output_dir}")
    print()

    # Create visualizer
    visualizer = TradeVisualizer(output_dir=output_dir, dpi=150)

    # Validate first trade format
    if trades:
        first_trade = trades[0]
        required_fields = ['symbol', 'timeframe', 'type', 'entry', 'tp', 'sl',
                          'open_time_utc', 'close_time_utc', 'close_price', 'status']
        missing_fields = [f for f in required_fields if f not in first_trade]

        if missing_fields:
            print("ERROR: Trade data missing required fields!")
            print(f"Missing: {missing_fields}")
            print()
            print("Please run the converter first:")
            print(f"  python tools/convert_pbema_trades.py {trades_path} BTCUSDT 15m")
            sys.exit(1)

        print("✓ Trade format validation passed")
        print()

    # Generate charts
    print("Generating charts...")
    print("-" * 80)

    successful = 0
    failed = 0

    for i, trade in enumerate(trades, 1):
        try:
            symbol_tf = trade.get('symbol', 'UNKNOWN')
            trade_type = trade.get('type', 'UNKNOWN')
            pnl = trade.get('pnl', 0)

            print(f"[{i}/{len(trades)}] {symbol_tf} {trade_type} PnL: ${pnl:+.2f}...", end=' ')

            chart_path = visualizer.visualize_trade(
                trade,
                candles_before=250,
                candles_after=35
            )

            if chart_path:
                print("✓")
                successful += 1
            else:
                print("✗ (no chart generated)")
                failed += 1

        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1

    print("-" * 80)
    print()
    print(f"Results: {successful} successful, {failed} failed")
    print()

    if successful > 0:
        print(f"Charts saved to: {output_dir}")
        print()
        print("Compare with original charts:")
        print(f"  Original: {trades_path.parent}/charts/")
        print(f"  New:      {output_dir}/")


if __name__ == "__main__":
    main()
