#!/usr/bin/env python3
"""
Generate a concise summary table of all SSL Flow trades
"""

import json
from typing import List, Dict, Any
from datetime import datetime


def load_trades(file_path: str) -> List[Dict[str, Any]]:
    """Load trades from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_r_multiple(trade: Dict[str, Any]) -> float:
    """Calculate R-multiple for a trade."""
    entry = trade['entry_price']
    exit_price = trade['exit_price']
    sl = trade['sl_price']
    signal_type = trade['signal_type']

    if signal_type == 'LONG':
        risk = entry - sl
        profit = exit_price - entry
    else:
        risk = sl - entry
        profit = entry - exit_price

    return profit / risk if risk != 0 else 0.0


def generate_summary_table():
    """Generate a summary table of all trades."""
    trades_file = '/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/data/results/BTCUSDT_15m_20260104_115258_full_v3/trades.json'
    trades = load_trades(trades_file)

    print("="*120)
    print("SSL FLOW TRADE SUMMARY TABLE")
    print("="*120)
    print()

    # Header
    print(f"{'#':<3} {'Date':<12} {'Dir':<5} {'Entry':<10} {'Exit':<10} {'SL':<10} {'TP':<10} {'Bars':<5} {'PnL':<8} {'R':<7} {'Result':<6}")
    print("-"*120)

    # Trades
    running_balance = 1000.0
    for i, trade in enumerate(trades, 1):
        entry_date = datetime.fromisoformat(trade['entry_time'].replace('+00:00', '')).strftime('%Y-%m-%d')
        direction = trade['signal_type']
        entry = trade['entry_price']
        exit_price = trade['exit_price']
        sl = trade['sl_price']
        tp = trade['tp_price']
        bars = trade['bars_held']
        pnl = trade['pnl']
        r_mult = calculate_r_multiple(trade)
        result = trade['exit_reason']

        # Color coding
        result_symbol = "✓" if trade['win'] else "✗"

        print(f"{i:<3} {entry_date:<12} {direction:<5} {entry:<10.2f} {exit_price:<10.2f} {sl:<10.2f} {tp:<10.2f} {bars:<5} ${pnl:>6.2f} {r_mult:>6.3f} {result_symbol} {result}")

        running_balance += pnl

    print("-"*120)
    print()

    # Summary statistics
    print("SUMMARY STATISTICS:")
    print("-"*120)

    total_trades = len(trades)
    winners = [t for t in trades if t['win']]
    losers = [t for t in trades if not t['win']]
    longs = [t for t in trades if t['signal_type'] == 'LONG']
    shorts = [t for t in trades if t['signal_type'] == 'SHORT']

    total_pnl = sum(t['pnl'] for t in trades)
    total_r = sum(calculate_r_multiple(t) for t in trades)

    print(f"Total Trades:        {total_trades}")
    print(f"  LONG:              {len(longs)} ({len(longs)/total_trades*100:.1f}%)")
    print(f"  SHORT:             {len(shorts)} ({len(shorts)/total_trades*100:.1f}%)")
    print()

    print(f"Winners:             {len(winners)} ({len(winners)/total_trades*100:.1f}%)")
    print(f"  LONG winners:      {len([t for t in longs if t['win']])} / {len(longs)} ({len([t for t in longs if t['win']])/len(longs)*100:.1f}%)")
    print(f"  SHORT winners:     {len([t for t in shorts if t['win']])} / {len(shorts)} ({len([t for t in shorts if t['win']])/len(shorts)*100:.1f}%)")
    print()

    print(f"Losers:              {len(losers)} ({len(losers)/total_trades*100:.1f}%)")
    print(f"  LONG losers:       {len([t for t in longs if not t['win']])} / {len(longs)} ({len([t for t in longs if not t['win']])/len(longs)*100:.1f}%)")
    print(f"  SHORT losers:      {len([t for t in shorts if not t['win']])} / {len(shorts)} ({len([t for t in shorts if not t['win']])/len(shorts)*100:.1f}%)")
    print()

    print(f"Total PnL:           ${total_pnl:.2f}")
    print(f"  LONG PnL:          ${sum(t['pnl'] for t in longs):.2f}")
    print(f"  SHORT PnL:         ${sum(t['pnl'] for t in shorts):.2f}")
    print()

    print(f"Total R:             {total_r:.3f}R")
    print(f"Avg R per trade:     {total_r/total_trades:.3f}R")
    print()

    print(f"Starting Balance:    ${trades[0]['balance_before']:.2f}")
    print(f"Ending Balance:      ${trades[-1]['balance_after']:.2f}")
    print(f"Return:              {(trades[-1]['balance_after']/trades[0]['balance_before'] - 1)*100:.2f}%")
    print()

    # Quick failure analysis
    quick_failures = [t for t in losers if t['bars_held'] <= 20]
    quick_long_failures = [t for t in quick_failures if t['signal_type'] == 'LONG']
    quick_short_failures = [t for t in quick_failures if t['signal_type'] == 'SHORT']

    print(f"Quick SL Failures (≤20 bars):")
    print(f"  Total:             {len(quick_failures)} / {len(losers)} losses ({len(quick_failures)/len(losers)*100:.1f}%)")
    print(f"  LONG:              {len(quick_long_failures)} / {len(longs)} trades ({len(quick_long_failures)/len(longs)*100:.1f}%)")
    print(f"  SHORT:             {len(quick_short_failures)} / {len(shorts)} trades ({len(quick_short_failures)/len(shorts)*100:.1f}%)")
    print()

    print("="*120)

    # Save to CSV for Excel analysis
    csv_file = '/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/ssl_flow_trades_summary.csv'
    with open(csv_file, 'w') as f:
        # Header
        f.write("Trade#,Date,Direction,Entry,Exit,SL,TP,Bars,PnL,R-Multiple,Win,Exit_Reason\n")

        # Data
        for i, trade in enumerate(trades, 1):
            entry_date = datetime.fromisoformat(trade['entry_time'].replace('+00:00', '')).strftime('%Y-%m-%d')
            direction = trade['signal_type']
            entry = trade['entry_price']
            exit_price = trade['exit_price']
            sl = trade['sl_price']
            tp = trade['tp_price']
            bars = trade['bars_held']
            pnl = trade['pnl']
            r_mult = calculate_r_multiple(trade)
            win = 1 if trade['win'] else 0
            exit_reason = trade['exit_reason']

            f.write(f"{i},{entry_date},{direction},{entry:.2f},{exit_price:.2f},{sl:.2f},{tp:.2f},{bars},{pnl:.2f},{r_mult:.3f},{win},{exit_reason}\n")

    print(f"\nCSV file saved to: {csv_file}")
    print("Import this into Excel/Google Sheets for further analysis!")


if __name__ == '__main__':
    generate_summary_table()
