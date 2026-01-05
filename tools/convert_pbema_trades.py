#!/usr/bin/env python3
"""
Convert PBEMA Trade Data to TradeVisualizer Format

This script converts PBEMA Retest trade data from the simplified format
to the format expected by TradeVisualizer for proper chart generation.

Usage:
    python tools/convert_pbema_trades.py data/results/PBEMA_RETEST_trades/trades.json BTCUSDT 15m
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def convert_trade_format(trade, symbol, timeframe):
    """
    Convert simplified trade format to TradeVisualizer format.

    Args:
        trade: Trade dict in simplified format
        symbol: Trading symbol (e.g., 'BTCUSDT')
        timeframe: Timeframe (e.g., '15m')

    Returns:
        Trade dict in TradeVisualizer format
    """
    # Extract exit price from TP/SL based on exit type
    if trade['exit_type'] == 'TP':
        exit_price = trade['tp_price']
    elif trade['exit_type'] == 'SL':
        exit_price = trade['sl_price']
    else:
        # For other exit types, estimate based on P&L
        entry = trade['entry_price']
        position_size = trade.get('position_size', 35.0)
        pnl_pct = trade['pnl'] / position_size

        if trade['signal_type'] == 'LONG':
            exit_price = entry * (1 + pnl_pct)
        else:
            exit_price = entry * (1 - pnl_pct)

    # Calculate R-multiple
    entry = trade['entry_price']
    tp = trade['tp_price']
    sl = trade['sl_price']

    if trade['signal_type'] == 'LONG':
        risk = entry - sl
        actual_gain = exit_price - entry
    else:
        risk = sl - entry
        actual_gain = entry - exit_price

    r_multiple = actual_gain / risk if risk > 0 else 0

    # Determine status
    status = 'WON' if trade['win'] else 'LOST'

    # Build TradeVisualizer-compatible trade dict
    converted = {
        'symbol': f"{symbol}-{timeframe}",
        'timeframe': timeframe,
        'type': trade['signal_type'],
        'entry': trade['entry_price'],
        'tp': trade['tp_price'],
        'sl': trade['sl_price'],
        'open_time_utc': trade['entry_time'],
        'close_time_utc': trade['exit_time'],
        'close_price': exit_price,
        'pnl': trade['pnl'],
        'status': status,
        'r_multiple': r_multiple,
        # Optional fields for enhanced visualization
        'exit_type': trade['exit_type'],
        'position_size': trade.get('position_size', 35.0),
        'exit_idx': trade.get('exit_idx'),
    }

    return converted


def main():
    if len(sys.argv) < 4:
        print("Usage: python tools/convert_pbema_trades.py <input_json> <symbol> <timeframe>")
        print("Example: python tools/convert_pbema_trades.py data/results/PBEMA_RETEST_trades/trades.json BTCUSDT 15m")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    symbol = sys.argv[2]
    timeframe = sys.argv[3]

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print("=" * 80)
    print("PBEMA TRADE FORMAT CONVERTER")
    print("=" * 80)
    print(f"Input:     {input_path}")
    print(f"Symbol:    {symbol}")
    print(f"Timeframe: {timeframe}")
    print()

    # Load trades
    with open(input_path, 'r') as f:
        trades = json.load(f)

    print(f"Loaded {len(trades)} trades")
    print()

    # Convert trades
    converted_trades = []
    for i, trade in enumerate(trades):
        try:
            converted = convert_trade_format(trade, symbol, timeframe)
            converted_trades.append(converted)
        except Exception as e:
            print(f"Warning: Failed to convert trade {i}: {e}")
            continue

    print(f"Successfully converted {len(converted_trades)}/{len(trades)} trades")
    print()

    # Save converted trades
    output_path = input_path.parent / "trades_converted.json"
    with open(output_path, 'w') as f:
        json.dump(converted_trades, f, indent=2, default=str)

    print(f"Converted trades saved to: {output_path}")
    print()

    # Show sample conversion
    if converted_trades:
        print("Sample conversion:")
        print("-" * 80)
        print("ORIGINAL FORMAT:")
        print(json.dumps(trades[0], indent=2))
        print()
        print("CONVERTED FORMAT:")
        print(json.dumps(converted_trades[0], indent=2, default=str))
        print("-" * 80)

    print()
    print("Next steps:")
    print("  1. Backup original: mv trades.json trades_original.json")
    print("  2. Use converted: mv trades_converted.json trades.json")
    print("  3. Regenerate charts with TradeVisualizer")
    print()


if __name__ == "__main__":
    main()
