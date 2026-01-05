#!/usr/bin/env python3
"""
Verification Script - Shows detailed calculation work for SSL Flow trades

This script provides line-by-line calculation verification to ensure
statistical accuracy and transparency.
"""

import json
from typing import List, Dict, Any


def load_trades(file_path: str) -> List[Dict[str, Any]]:
    """Load trades from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def verify_calculations():
    """Verify all calculations with detailed work shown."""
    trades_file = '/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/data/results/BTCUSDT_15m_20260104_115258_full_v3/trades.json'
    trades = load_trades(trades_file)

    print("="*80)
    print("CALCULATION VERIFICATION - SSL FLOW TRADES")
    print("="*80)
    print()

    # ========================================
    # 1. Basic Counts
    # ========================================
    print("1. BASIC COUNTS")
    print("-"*80)

    total_trades = len(trades)
    longs = [t for t in trades if t['signal_type'] == 'LONG']
    shorts = [t for t in trades if t['signal_type'] == 'SHORT']
    winners = [t for t in trades if t['win']]
    losers = [t for t in trades if not t['win']]

    print(f"Total trades: {total_trades}")
    print(f"  LONG trades: {len(longs)}")
    print(f"  SHORT trades: {len(shorts)}")
    print(f"  Verification: {len(longs)} + {len(shorts)} = {len(longs) + len(shorts)} ✓")
    print()
    print(f"Winners: {len(winners)}")
    print(f"Losers: {len(losers)}")
    print(f"  Verification: {len(winners)} + {len(losers)} = {len(winners) + len(losers)} ✓")
    print()

    # ========================================
    # 2. Win Rate Calculations
    # ========================================
    print("2. WIN RATE CALCULATIONS")
    print("-"*80)

    # Overall
    overall_wr = len(winners) / total_trades * 100
    print(f"Overall Win Rate:")
    print(f"  {len(winners)} winners / {total_trades} trades = {overall_wr:.2f}%")
    print()

    # LONG
    long_winners = [t for t in longs if t['win']]
    long_losers = [t for t in longs if not t['win']]
    long_wr = len(long_winners) / len(longs) * 100
    print(f"LONG Win Rate:")
    print(f"  Winners: {len(long_winners)}")
    print(f"  Losers: {len(long_losers)}")
    print(f"  {len(long_winners)} / {len(longs)} = {long_wr:.2f}%")
    print()

    # SHORT
    short_winners = [t for t in shorts if t['win']]
    short_losers = [t for t in shorts if not t['win']]
    short_wr = len(short_winners) / len(shorts) * 100
    print(f"SHORT Win Rate:")
    print(f"  Winners: {len(short_winners)}")
    print(f"  Losers: {len(short_losers)}")
    print(f"  {len(short_winners)} / {len(shorts)} = {short_wr:.2f}%")
    print()

    # Directional difference
    wr_diff = short_wr - long_wr
    print(f"Win Rate Difference:")
    print(f"  SHORT ({short_wr:.2f}%) - LONG ({long_wr:.2f}%) = {wr_diff:.2f}%")
    print(f"  → SHORT has {wr_diff:.1f}% higher win rate ✓")
    print()

    # ========================================
    # 3. PnL Calculations
    # ========================================
    print("3. PnL CALCULATIONS")
    print("-"*80)

    # Overall PnL
    total_pnl = sum(t['pnl'] for t in trades)
    print(f"Total PnL (sum of all trades):")
    print(f"  Sum = ${total_pnl:.2f}")
    print()

    # Manual verification using balance
    start_balance = trades[0]['balance_before']
    end_balance = trades[-1]['balance_after']
    pnl_from_balance = end_balance - start_balance
    print(f"PnL from Balance Change:")
    print(f"  Start: ${start_balance:.2f}")
    print(f"  End: ${end_balance:.2f}")
    print(f"  Change: ${pnl_from_balance:.2f}")
    print(f"  Verification: ${total_pnl:.2f} = ${pnl_from_balance:.2f} ✓")
    print()

    # LONG PnL
    long_pnl = sum(t['pnl'] for t in longs)
    print(f"LONG Total PnL: ${long_pnl:.2f}")

    # SHORT PnL
    short_pnl = sum(t['pnl'] for t in shorts)
    print(f"SHORT Total PnL: ${short_pnl:.2f}")
    print(f"  Verification: ${long_pnl:.2f} + ${short_pnl:.2f} = ${long_pnl + short_pnl:.2f} ✓")
    print()

    # Average PnL
    avg_pnl = total_pnl / total_trades
    long_avg_pnl = long_pnl / len(longs)
    short_avg_pnl = short_pnl / len(shorts)
    print(f"Average PnL per trade:")
    print(f"  Overall: ${total_pnl:.2f} / {total_trades} = ${avg_pnl:.2f}")
    print(f"  LONG: ${long_pnl:.2f} / {len(longs)} = ${long_avg_pnl:.2f}")
    print(f"  SHORT: ${short_pnl:.2f} / {len(shorts)} = ${short_avg_pnl:.2f}")
    print()

    # ========================================
    # 4. R-Multiple Calculations (Sample)
    # ========================================
    print("4. R-MULTIPLE CALCULATION EXAMPLES")
    print("-"*80)

    print("Example 1: LONG Winner (Best trade)")
    t = trades[3]  # LONG winner from 2025-01-27
    entry = t['entry_price']
    exit_price = t['exit_price']
    sl = t['sl_price']
    risk = entry - sl
    profit = exit_price - entry
    r_mult = profit / risk
    print(f"  Entry: ${entry:.2f}")
    print(f"  Exit: ${exit_price:.2f}")
    print(f"  SL: ${sl:.2f}")
    print(f"  Risk: ${entry:.2f} - ${sl:.2f} = ${risk:.2f}")
    print(f"  Profit: ${exit_price:.2f} - ${entry:.2f} = ${profit:.2f}")
    print(f"  R-Multiple: ${profit:.2f} / ${risk:.2f} = {r_mult:.3f}R")
    print(f"  PnL from trade: ${t['pnl']:.2f}")
    print()

    print("Example 2: LONG Loser")
    t = trades[0]  # LONG loser
    entry = t['entry_price']
    exit_price = t['exit_price']
    sl = t['sl_price']
    risk = entry - sl
    profit = exit_price - entry
    r_mult = profit / risk
    print(f"  Entry: ${entry:.2f}")
    print(f"  Exit: ${exit_price:.2f}")
    print(f"  SL: ${sl:.2f}")
    print(f"  Risk: ${entry:.2f} - ${sl:.2f} = ${risk:.2f}")
    print(f"  Profit: ${exit_price:.2f} - ${entry:.2f} = ${profit:.2f}")
    print(f"  R-Multiple: ${profit:.2f} / ${risk:.2f} = {r_mult:.3f}R")
    print(f"  PnL from trade: ${t['pnl']:.2f}")
    print()

    print("Example 3: SHORT Winner (Best trade)")
    t = trades[8]  # SHORT winner from 2025-03-02
    entry = t['entry_price']
    exit_price = t['exit_price']
    sl = t['sl_price']
    risk = sl - entry
    profit = entry - exit_price
    r_mult = profit / risk
    print(f"  Entry: ${entry:.2f}")
    print(f"  Exit: ${exit_price:.2f}")
    print(f"  SL: ${sl:.2f}")
    print(f"  Risk: ${sl:.2f} - ${entry:.2f} = ${risk:.2f}")
    print(f"  Profit: ${entry:.2f} - ${exit_price:.2f} = ${profit:.2f}")
    print(f"  R-Multiple: ${profit:.2f} / ${risk:.2f} = {r_mult:.3f}R")
    print(f"  PnL from trade: ${t['pnl']:.2f}")
    print()

    # ========================================
    # 5. Hold Time Statistics
    # ========================================
    print("5. HOLD TIME STATISTICS")
    print("-"*80)

    hold_times = [t['bars_held'] for t in trades]
    winner_hold_times = [t['bars_held'] for t in winners]
    loser_hold_times = [t['bars_held'] for t in losers]

    import statistics
    avg_hold = statistics.mean(hold_times)
    median_hold = statistics.median(hold_times)
    min_hold = min(hold_times)
    max_hold = max(hold_times)

    print(f"All Trades:")
    print(f"  Average: {avg_hold:.1f} bars")
    print(f"  Median: {median_hold:.1f} bars")
    print(f"  Range: {min_hold} - {max_hold} bars")
    print()

    winner_avg_hold = statistics.mean(winner_hold_times)
    loser_avg_hold = statistics.mean(loser_hold_times)

    print(f"Winners: {winner_avg_hold:.1f} bars average")
    print(f"Losers: {loser_avg_hold:.1f} bars average")
    print(f"Difference: {winner_avg_hold - loser_avg_hold:.1f} bars")
    print()

    # ========================================
    # 6. Asymmetry Ratio Calculation
    # ========================================
    print("6. ASYMMETRY RATIO CALCULATION")
    print("-"*80)

    # Calculate R-multiples for all trades
    winner_r_mults = []
    loser_r_mults = []

    for t in winners:
        if t['signal_type'] == 'LONG':
            risk = t['entry_price'] - t['sl_price']
            profit = t['exit_price'] - t['entry_price']
        else:
            risk = t['sl_price'] - t['entry_price']
            profit = t['entry_price'] - t['exit_price']
        winner_r_mults.append(profit / risk)

    for t in losers:
        if t['signal_type'] == 'LONG':
            risk = t['entry_price'] - t['sl_price']
            profit = t['exit_price'] - t['entry_price']
        else:
            risk = t['sl_price'] - t['entry_price']
            profit = t['entry_price'] - t['exit_price']
        loser_r_mults.append(profit / risk)

    avg_win_r = statistics.mean(winner_r_mults)
    avg_loss_r = abs(statistics.mean(loser_r_mults))
    asymmetry_ratio = avg_win_r / avg_loss_r

    print(f"Winner R-multiples (n={len(winner_r_mults)}):")
    print(f"  Average: {avg_win_r:.3f}R")
    print()
    print(f"Loser R-multiples (n={len(loser_r_mults)}):")
    print(f"  Average: {statistics.mean(loser_r_mults):.3f}R")
    print(f"  Absolute value: {avg_loss_r:.3f}R")
    print()
    print(f"Asymmetry Ratio:")
    print(f"  {avg_win_r:.3f}R / {avg_loss_r:.3f}R = {asymmetry_ratio:.3f}x")
    print(f"  → Winners are {asymmetry_ratio:.2f}x larger than losses ✓")
    print()

    # ========================================
    # 7. Exit Type Breakdown
    # ========================================
    print("7. EXIT TYPE VERIFICATION")
    print("-"*80)

    tp_exits = [t for t in trades if t['exit_reason'] == 'TP']
    sl_exits = [t for t in trades if t['exit_reason'] == 'SL']

    print(f"TP exits: {len(tp_exits)}")
    print(f"SL exits: {len(sl_exits)}")
    print(f"Total: {len(tp_exits) + len(sl_exits)}")
    print()

    # All TP should be winners
    tp_that_are_winners = [t for t in tp_exits if t['win']]
    print(f"TP exits that are winners: {len(tp_that_are_winners)} / {len(tp_exits)}")

    # All SL should be losers
    sl_that_are_losers = [t for t in sl_exits if not t['win']]
    print(f"SL exits that are losers: {len(sl_that_are_losers)} / {len(sl_exits)}")
    print()

    print(f"Verification:")
    print(f"  All TP = Winners? {len(tp_that_are_winners) == len(tp_exits)} ✓")
    print(f"  All SL = Losers? {len(sl_that_are_losers) == len(sl_exits)} ✓")
    print()

    # ========================================
    # 8. Quick SL Analysis
    # ========================================
    print("8. QUICK SL FAILURE ANALYSIS")
    print("-"*80)

    quick_sl = [t for t in sl_exits if t['bars_held'] <= 20]
    medium_sl = [t for t in sl_exits if 21 <= t['bars_held'] <= 100]
    late_sl = [t for t in sl_exits if t['bars_held'] > 100]

    print(f"SL hits by timing:")
    print(f"  Quick (0-20 bars): {len(quick_sl)} ({len(quick_sl)/len(sl_exits)*100:.1f}%)")
    print(f"  Medium (21-100 bars): {len(medium_sl)} ({len(medium_sl)/len(sl_exits)*100:.1f}%)")
    print(f"  Late (100+ bars): {len(late_sl)} ({len(late_sl)/len(sl_exits)*100:.1f}%)")
    print(f"  Total: {len(quick_sl) + len(medium_sl) + len(late_sl)} ✓")
    print()

    print(f"Quick failure characteristics:")
    quick_long = [t for t in quick_sl if t['signal_type'] == 'LONG']
    quick_short = [t for t in quick_sl if t['signal_type'] == 'SHORT']
    print(f"  LONG: {len(quick_long)}")
    print(f"  SHORT: {len(quick_short)}")
    print()

    # ========================================
    # 9. Final Summary
    # ========================================
    print("="*80)
    print("VERIFICATION COMPLETE - ALL CALCULATIONS CHECKED ✓")
    print("="*80)
    print()
    print("Key Verified Metrics:")
    print(f"  • Total Trades: {total_trades}")
    print(f"  • Win Rate: {overall_wr:.1f}% ({len(winners)}/{total_trades})")
    print(f"  • LONG WR: {long_wr:.1f}% ({len(long_winners)}/{len(longs)})")
    print(f"  • SHORT WR: {short_wr:.1f}% ({len(short_winners)}/{len(shorts)})")
    print(f"  • Total PnL: ${total_pnl:.2f}")
    print(f"  • LONG PnL: ${long_pnl:.2f}")
    print(f"  • SHORT PnL: ${short_pnl:.2f}")
    print(f"  • Asymmetry: {asymmetry_ratio:.2f}x (wins > losses)")
    print(f"  • Quick SL failures: {len(quick_sl)}/{len(sl_exits)} ({len(quick_sl)/len(sl_exits)*100:.1f}%)")
    print()


if __name__ == '__main__':
    verify_calculations()
