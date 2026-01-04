#!/usr/bin/env python3
"""
Trade Visualization Runner

Visualize individual trades from Rolling Walk-Forward backtest results.
Creates TradingView-style charts to diagnose trade quality.

Usage:
    python run_trade_visualizer.py --file data/rolling_wf_runs/.../trades_detailed.txt
    python run_trade_visualizer.py --file trades_detailed.txt --trade 5
    python run_trade_visualizer.py --file trades_detailed.txt --all
    python run_trade_visualizer.py --file trades_detailed.txt --browse

Author: Claude (Anthropic)
Date: December 28, 2024
"""

import os
import sys
import re
import argparse
from typing import Dict, List
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.trade_visualizer import TradeVisualizer
from core.logging_config import get_logger

_logger = get_logger(__name__)


def parse_trades_from_detailed_txt(filepath: str) -> List[Dict]:
    """
    Parse trades from trades_detailed.txt file.

    This parses the formatted text output from rolling WF tests.

    Args:
        filepath: Path to trades_detailed.txt

    Returns:
        List of trade dictionaries
    """
    trades = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by trade blocks (each starts with "Trade #")
        trade_blocks = re.split(r'┌─ Trade #\d+', content)

        for block in trade_blocks[1:]:  # Skip first empty block
            try:
                trade = _parse_single_trade_block(block)
                if trade:
                    trades.append(trade)
            except Exception as e:
                _logger.warning(f"Failed to parse trade block: {e}")
                continue

        _logger.info(f"Parsed {len(trades)} trades from {filepath}")
        return trades

    except Exception as e:
        _logger.error(f"Failed to read file {filepath}: {e}")
        return []


def _parse_single_trade_block(block: str) -> Dict:
    """
    Parse a single trade block from trades_detailed.txt.

    Args:
        block: Text block for one trade

    Returns:
        Trade dictionary
    """
    trade = {}

    # Extract status (WIN/LOSS)
    if '✅ WIN' in block or 'WIN' in block:
        trade['status'] = 'WON'
    elif '❌ LOSS' in block or 'LOSS' in block:
        trade['status'] = 'LOST'
    else:
        trade['status'] = 'UNKNOWN'

    # Extract fields using regex
    patterns = {
        'symbol': r'Symbol:\s+([A-Z]+)-(\w+)',
        'type': r'Type:\s+(LONG|SHORT)',
        'entry_time': r'Entry Time:\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)',
        'entry': r'Entry:\s+\$?([\d,]+\.?\d*)',
        'tp': r'TP Target:\s+\$?([\d,]+\.?\d*)',
        'sl': r'SL Target:\s+\$?([\d,]+\.?\d*)',
        'exit_time': r'Exit Time:\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)',
        'exit_price': r'Exit Price:\s+\$?([\d,]+\.?\d*)',
        'pnl': r'PnL:\s+\$?([+-]?[\d,]+\.?\d*)',
        'r_multiple': r'R-Multiple:\s+([+-]?[\d.]+)',
        'at_buyers': r'AT Buyers:\s+([\d,]+\.?\d*)',
        'at_sellers': r'AT Sellers:\s+([\d,]+\.?\d*)',
        'at_dominant': r'AT Dominant:\s+(BUYERS|SELLERS)',
        'baseline': r'Baseline:\s+([\d,]+\.?\d*)',
        'pbema_top': r'PBEMA Top:\s+([\d,]+\.?\d*)',
        'rsi': r'RSI:\s+([\d.]+)',
        'adx': r'ADX:\s+([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, block)
        if match:
            value = match.group(1)

            # Handle symbol/timeframe
            if key == 'symbol':
                trade['symbol'] = f"{match.group(1)}-{match.group(2)}"
                trade['timeframe'] = match.group(2)
                continue

            # Remove commas from numbers
            if key not in ['type', 'entry_time', 'exit_time', 'at_dominant', 'status']:
                value = value.replace(',', '')

            # Type conversion
            if key in ['entry', 'tp', 'sl', 'exit_price', 'pnl', 'r_multiple',
                      'at_buyers', 'at_sellers', 'baseline', 'pbema_top', 'rsi', 'adx']:
                try:
                    trade[key] = float(value)
                except ValueError:
                    trade[key] = 0.0
            else:
                trade[key] = value

    # Rename fields to match expected format
    if 'entry_time' in trade:
        trade['open_time_utc'] = trade.pop('entry_time')
    if 'exit_time' in trade:
        trade['close_time_utc'] = trade.pop('exit_time')
    if 'exit_price' not in trade and 'close_price' not in trade:
        trade['close_price'] = trade.get('tp', 0.0) if trade.get('status') == 'WON' else trade.get('sl', 0.0)
    elif 'exit_price' in trade:
        trade['close_price'] = trade.pop('exit_price')

    # Store indicators at entry
    trade['indicators_at_entry'] = {
        'at_buyers': trade.get('at_buyers', 0.0),
        'at_sellers': trade.get('at_sellers', 0.0),
        'at_dominant': trade.get('at_dominant', 'UNKNOWN'),
        'baseline': trade.get('baseline', 0.0),
        'pb_ema_top': trade.get('pbema_top', 0.0),
        'rsi': trade.get('rsi', 0.0),
        'adx': trade.get('adx', 0.0),
    }

    # Validation
    required_fields = ['symbol', 'timeframe', 'type', 'open_time_utc',
                      'close_time_utc', 'entry', 'tp', 'sl', 'pnl', 'status']

    if not all(field in trade for field in required_fields):
        missing = [f for f in required_fields if f not in trade]
        _logger.warning(f"Trade missing fields: {missing}")
        return None

    return trade


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Trade Visualization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all trades
  python run_trade_visualizer.py --file data/rolling_wf_runs/.../trades_detailed.txt --all

  # Visualize specific trade
  python run_trade_visualizer.py --file trades_detailed.txt --trade 5

  # Interactive browser mode
  python run_trade_visualizer.py --file trades_detailed.txt --browse

  # Custom output directory
  python run_trade_visualizer.py --file trades_detailed.txt --all --output my_charts/
        """
    )

    parser.add_argument(
        '--file', '-f',
        type=str,
        required=True,
        help='Path to trades_detailed.txt file'
    )

    parser.add_argument(
        '--trade', '-t',
        type=int,
        help='Specific trade number to visualize'
    )

    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Visualize all trades'
    )

    parser.add_argument(
        '--browse', '-b',
        action='store_true',
        help='Interactive browser mode'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='trade_charts',
        help='Output directory for charts (default: trade_charts/)'
    )

    parser.add_argument(
        '--candles-before',
        type=int,
        default=75,
        help='Number of candles before entry (default: 75)'
    )

    parser.add_argument(
        '--candles-after',
        type=int,
        default=35,
        help='Number of candles after exit (default: 35)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Chart resolution DPI (default: 300)'
    )

    args = parser.parse_args()

    # Validate file exists
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    # Parse trades
    print(f"Loading trades from {args.file}...")
    trades = parse_trades_from_detailed_txt(args.file)

    if not trades:
        print("No trades found in file")
        sys.exit(1)

    print(f"Found {len(trades)} trades")

    # Initialize visualizer
    viz = TradeVisualizer(output_dir=args.output, dpi=args.dpi)

    # Execute based on mode
    if args.browse:
        # Interactive mode
        viz.interactive_browser(trades)

    elif args.all:
        # Visualize all trades
        print(f"\nVisualizing all {len(trades)} trades...")
        chart_paths = viz.visualize_all_trades(trades)
        print(f"\nDone! Generated {len(chart_paths)} charts in {args.output}/")

    elif args.trade is not None:
        # Visualize specific trade
        idx = args.trade - 1

        if idx < 0 or idx >= len(trades):
            print(f"Error: Trade number must be between 1 and {len(trades)}")
            sys.exit(1)

        trade = trades[idx]
        print(f"\nVisualizing trade #{args.trade}:")
        print(f"  {trade['symbol']} {trade['type']} @ {trade['open_time_utc']}")
        print(f"  PnL: ${trade['pnl']:+.2f} | R: {trade['r_multiple']:+.2f}x | {trade['status']}")

        chart_path = viz.visualize_trade(
            trade,
            candles_before=args.candles_before,
            candles_after=args.candles_after
        )

        if chart_path:
            print(f"\nChart saved: {chart_path}")
        else:
            print("Failed to generate chart")
            sys.exit(1)

    else:
        # No mode specified
        parser.print_help()
        print("\nError: Please specify --all, --trade N, or --browse")
        sys.exit(1)


if __name__ == '__main__':
    main()
