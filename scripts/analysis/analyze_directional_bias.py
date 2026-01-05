#!/usr/bin/env python3
"""
Deep Dive: LONG vs SHORT Performance Analysis

Investigates why LONG has 40% WR vs SHORT's 66.7% WR
"""

import json
from typing import List, Dict, Any
import statistics


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


def analyze_directional_bias():
    """Deep analysis of LONG vs SHORT performance."""
    trades_file = '/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/data/results/BTCUSDT_15m_20260104_115258_full_v3/trades.json'
    trades = load_trades(trades_file)

    longs = [t for t in trades if t['signal_type'] == 'LONG']
    shorts = [t for t in trades if t['signal_type'] == 'SHORT']

    print("="*80)
    print("DIRECTIONAL BIAS DEEP DIVE: LONG vs SHORT")
    print("="*80)
    print()

    # ========================================
    # 1. Basic Statistics Comparison
    # ========================================
    print("1. BASIC STATISTICS")
    print("-"*80)
    print(f"{'Metric':<30} {'LONG':<25} {'SHORT':<25}")
    print("-"*80)
    print(f"{'Total Trades':<30} {len(longs):<25} {len(shorts):<25}")

    long_winners = [t for t in longs if t['win']]
    long_losers = [t for t in longs if not t['win']]
    short_winners = [t for t in shorts if t['win']]
    short_losers = [t for t in shorts if not t['win']]

    print(f"{'Winners':<30} {len(long_winners):<25} {len(short_winners):<25}")
    print(f"{'Losers':<30} {len(long_losers):<25} {len(short_losers):<25}")
    print(f"{'Win Rate':<30} {len(long_winners)/len(longs)*100:.2f}%{'':<19} {len(short_winners)/len(shorts)*100:.2f}%")
    print()

    # ========================================
    # 2. PnL Distribution
    # ========================================
    print("2. PnL DISTRIBUTION")
    print("-"*80)

    long_pnls = [t['pnl'] for t in longs]
    short_pnls = [t['pnl'] for t in shorts]

    print(f"{'Metric':<30} {'LONG':<25} {'SHORT':<25}")
    print("-"*80)
    print(f"{'Total PnL':<30} ${sum(long_pnls):.2f}{'':<19} ${sum(short_pnls):.2f}")
    print(f"{'Avg PnL':<30} ${statistics.mean(long_pnls):.2f}{'':<19} ${statistics.mean(short_pnls):.2f}")
    print(f"{'Median PnL':<30} ${statistics.median(long_pnls):.2f}{'':<19} ${statistics.median(short_pnls):.2f}")
    print(f"{'Min PnL':<30} ${min(long_pnls):.2f}{'':<19} ${min(short_pnls):.2f}")
    print(f"{'Max PnL':<30} ${max(long_pnls):.2f}{'':<19} ${max(short_pnls):.2f}")
    print()

    # Winners vs Losers
    long_winner_pnls = [t['pnl'] for t in long_winners]
    long_loser_pnls = [t['pnl'] for t in long_losers]
    short_winner_pnls = [t['pnl'] for t in short_winners]
    short_loser_pnls = [t['pnl'] for t in short_losers]

    print("WINNER PnL:")
    if long_winner_pnls:
        print(f"  LONG avg: ${statistics.mean(long_winner_pnls):.2f}")
    if short_winner_pnls:
        print(f"  SHORT avg: ${statistics.mean(short_winner_pnls):.2f}")
    print()

    print("LOSER PnL:")
    if long_loser_pnls:
        print(f"  LONG avg: ${statistics.mean(long_loser_pnls):.2f}")
    if short_loser_pnls:
        print(f"  SHORT avg: ${statistics.mean(short_loser_pnls):.2f}")
    print()

    # ========================================
    # 3. Hold Time Comparison
    # ========================================
    print("3. HOLD TIME COMPARISON")
    print("-"*80)

    long_hold_times = [t['bars_held'] for t in longs]
    short_hold_times = [t['bars_held'] for t in shorts]

    print(f"{'Metric':<30} {'LONG':<25} {'SHORT':<25}")
    print("-"*80)
    print(f"{'Avg Hold Time':<30} {statistics.mean(long_hold_times):.1f} bars{'':<15} {statistics.mean(short_hold_times):.1f} bars")
    print(f"{'Median Hold Time':<30} {statistics.median(long_hold_times):.1f} bars{'':<15} {statistics.median(short_hold_times):.1f} bars")
    print(f"{'Min Hold Time':<30} {min(long_hold_times)} bars{'':<19} {min(short_hold_times)} bars")
    print(f"{'Max Hold Time':<30} {max(long_hold_times)} bars{'':<18} {max(short_hold_times)} bars")
    print()

    # Winners vs Losers hold time
    long_winner_holds = [t['bars_held'] for t in long_winners]
    long_loser_holds = [t['bars_held'] for t in long_losers]
    short_winner_holds = [t['bars_held'] for t in short_winners]
    short_loser_holds = [t['bars_held'] for t in short_losers]

    print("HOLD TIME BY OUTCOME:")
    if long_winner_holds:
        print(f"  LONG winners: {statistics.mean(long_winner_holds):.1f} bars avg")
    if long_loser_holds:
        print(f"  LONG losers: {statistics.mean(long_loser_holds):.1f} bars avg")
    if short_winner_holds:
        print(f"  SHORT winners: {statistics.mean(short_winner_holds):.1f} bars avg")
    if short_loser_holds:
        print(f"  SHORT losers: {statistics.mean(short_loser_holds):.1f} bars avg")
    print()

    # ========================================
    # 4. R-Multiple Comparison
    # ========================================
    print("4. R-MULTIPLE COMPARISON")
    print("-"*80)

    long_r_mults = [calculate_r_multiple(t) for t in longs]
    short_r_mults = [calculate_r_multiple(t) for t in shorts]

    print(f"{'Metric':<30} {'LONG':<25} {'SHORT':<25}")
    print("-"*80)
    print(f"{'Avg R-Multiple':<30} {statistics.mean(long_r_mults):.3f}R{'':<17} {statistics.mean(short_r_mults):.3f}R")
    print(f"{'Median R-Multiple':<30} {statistics.median(long_r_mults):.3f}R{'':<17} {statistics.median(short_r_mults):.3f}R")
    print(f"{'Min R-Multiple':<30} {min(long_r_mults):.3f}R{'':<17} {min(short_r_mults):.3f}R")
    print(f"{'Max R-Multiple':<30} {max(long_r_mults):.3f}R{'':<17} {max(short_r_mults):.3f}R")
    print()

    # Winners vs Losers R-multiple
    long_winner_r = [calculate_r_multiple(t) for t in long_winners]
    long_loser_r = [calculate_r_multiple(t) for t in long_losers]
    short_winner_r = [calculate_r_multiple(t) for t in short_winners]
    short_loser_r = [calculate_r_multiple(t) for t in short_losers]

    print("R-MULTIPLE BY OUTCOME:")
    if long_winner_r:
        print(f"  LONG winners: {statistics.mean(long_winner_r):.3f}R avg (median: {statistics.median(long_winner_r):.3f}R)")
    if long_loser_r:
        print(f"  LONG losers: {statistics.mean(long_loser_r):.3f}R avg (median: {statistics.median(long_loser_r):.3f}R)")
    if short_winner_r:
        print(f"  SHORT winners: {statistics.mean(short_winner_r):.3f}R avg (median: {statistics.median(short_winner_r):.3f}R)")
    if short_loser_r:
        print(f"  SHORT losers: {statistics.mean(short_loser_r):.3f}R avg (median: {statistics.median(short_loser_r):.3f}R)")
    print()

    # ========================================
    # 5. Quick SL Failure Analysis
    # ========================================
    print("5. QUICK SL FAILURE ANALYSIS (0-20 bars)")
    print("-"*80)

    long_quick_sl = [t for t in long_losers if t['bars_held'] <= 20 and t['exit_reason'] == 'SL']
    short_quick_sl = [t for t in short_losers if t['bars_held'] <= 20 and t['exit_reason'] == 'SL']

    print(f"Quick SL failures (0-20 bars):")
    print(f"  LONG: {len(long_quick_sl)} / {len(long_losers)} losers ({len(long_quick_sl)/len(long_losers)*100:.1f}%)")
    print(f"  SHORT: {len(short_quick_sl)} / {len(short_losers)} losers ({len(short_quick_sl)/len(short_losers)*100 if short_losers else 0:.1f}%)")
    print()

    print(f"INSIGHT: {len(long_quick_sl)} out of {len(long_losers)} LONG losses are quick failures")
    print(f"         This represents {len(long_quick_sl)/len(longs)*100:.1f}% of ALL LONG trades")
    print()

    # ========================================
    # 6. Exit Reason Breakdown
    # ========================================
    print("6. EXIT REASON BREAKDOWN")
    print("-"*80)

    long_tp = [t for t in longs if t['exit_reason'] == 'TP']
    long_sl = [t for t in longs if t['exit_reason'] == 'SL']
    short_tp = [t for t in shorts if t['exit_reason'] == 'TP']
    short_sl = [t for t in shorts if t['exit_reason'] == 'SL']

    print(f"{'Exit Type':<20} {'LONG':<30} {'SHORT':<30}")
    print("-"*80)
    print(f"{'TP (Take Profit)':<20} {len(long_tp)} ({len(long_tp)/len(longs)*100:.1f}%){'':<17} {len(short_tp)} ({len(short_tp)/len(shorts)*100:.1f}%)")
    print(f"{'SL (Stop Loss)':<20} {len(long_sl)} ({len(long_sl)/len(longs)*100:.1f}%){'':<17} {len(short_sl)} ({len(short_sl)/len(shorts)*100:.1f}%)")
    print()

    # ========================================
    # 7. Best and Worst Trades
    # ========================================
    print("7. BEST AND WORST TRADES")
    print("-"*80)

    print("BEST LONG TRADE:")
    best_long = max(longs, key=lambda t: calculate_r_multiple(t))
    r_mult = calculate_r_multiple(best_long)
    print(f"  Date: {best_long['entry_time']}")
    print(f"  R-Multiple: {r_mult:.3f}R")
    print(f"  PnL: ${best_long['pnl']:.2f}")
    print(f"  Hold time: {best_long['bars_held']} bars")
    print(f"  Exit: {best_long['exit_reason']}")
    print()

    print("WORST LONG TRADE:")
    worst_long = min(longs, key=lambda t: calculate_r_multiple(t))
    r_mult = calculate_r_multiple(worst_long)
    print(f"  Date: {worst_long['entry_time']}")
    print(f"  R-Multiple: {r_mult:.3f}R")
    print(f"  PnL: ${worst_long['pnl']:.2f}")
    print(f"  Hold time: {worst_long['bars_held']} bars")
    print(f"  Exit: {worst_long['exit_reason']}")
    print()

    print("BEST SHORT TRADE:")
    best_short = max(shorts, key=lambda t: calculate_r_multiple(t))
    r_mult = calculate_r_multiple(best_short)
    print(f"  Date: {best_short['entry_time']}")
    print(f"  R-Multiple: {r_mult:.3f}R")
    print(f"  PnL: ${best_short['pnl']:.2f}")
    print(f"  Hold time: {best_short['bars_held']} bars")
    print(f"  Exit: {best_short['exit_reason']}")
    print()

    print("WORST SHORT TRADE:")
    worst_short = min(shorts, key=lambda t: calculate_r_multiple(t))
    r_mult = calculate_r_multiple(worst_short)
    print(f"  Date: {worst_short['entry_time']}")
    print(f"  R-Multiple: {r_mult:.3f}R")
    print(f"  PnL: ${worst_short['pnl']:.2f}")
    print(f"  Hold time: {worst_short['bars_held']} bars")
    print(f"  Exit: {worst_short['exit_reason']}")
    print()

    # ========================================
    # 8. Temporal Distribution
    # ========================================
    print("8. TEMPORAL DISTRIBUTION")
    print("-"*80)

    # Extract months from entry times
    import datetime
    long_months = [datetime.datetime.fromisoformat(t['entry_time'].replace('+00:00', '')).month for t in longs]
    short_months = [datetime.datetime.fromisoformat(t['entry_time'].replace('+00:00', '')).month for t in shorts]

    print("Trades by month (sample):")
    print(f"  LONG: {len(longs)} trades across {len(set(long_months))} different months")
    print(f"  SHORT: {len(shorts)} trades across {len(set(short_months))} different months")
    print()

    # ========================================
    # 9. KEY INSIGHTS
    # ========================================
    print("="*80)
    print("KEY INSIGHTS: WHY SHORT OUTPERFORMS LONG")
    print("="*80)
    print()

    print("1. SAMPLE SIZE DIFFERENCE:")
    print(f"   - LONG: {len(longs)} trades")
    print(f"   - SHORT: {len(shorts)} trades")
    print(f"   - SHORT has {len(longs)/len(shorts):.1f}x fewer trades (smaller sample)")
    print()

    print("2. QUICK FAILURE RATE:")
    long_quick_rate = len(long_quick_sl) / len(longs) * 100
    short_quick_rate = len(short_quick_sl) / len(shorts) * 100 if shorts else 0
    print(f"   - LONG quick SL failures: {long_quick_rate:.1f}% of all LONG trades")
    print(f"   - SHORT quick SL failures: {short_quick_rate:.1f}% of all SHORT trades")
    print(f"   - LONG has {len(long_quick_sl)} quick failures vs SHORT's {len(short_quick_sl)}")
    print()

    print("3. WINNER QUALITY:")
    if long_winner_r and short_winner_r:
        print(f"   - LONG winners: {statistics.mean(long_winner_r):.3f}R average")
        print(f"   - SHORT winners: {statistics.mean(short_winner_r):.3f}R average")
        print(f"   - SHORT winners are {statistics.mean(short_winner_r)/statistics.mean(long_winner_r):.2f}x larger")
    print()

    print("4. LOSER QUALITY:")
    if long_loser_r and short_loser_r:
        print(f"   - LONG losers: {statistics.mean(long_loser_r):.3f}R average")
        print(f"   - SHORT losers: {statistics.mean(short_loser_r):.3f}R average")
        print(f"   - Losses are similar in magnitude")
    print()

    print("5. OVERALL ASYMMETRY:")
    if long_winner_r and long_loser_r:
        long_asymmetry = statistics.mean(long_winner_r) / abs(statistics.mean(long_loser_r))
        print(f"   - LONG: {long_asymmetry:.2f}x asymmetry (wins vs losses)")
    if short_winner_r and short_loser_r:
        short_asymmetry = statistics.mean(short_winner_r) / abs(statistics.mean(short_loser_r))
        print(f"   - SHORT: {short_asymmetry:.2f}x asymmetry (wins vs losses)")
    print()

    print("CONCLUSION:")
    print("-"*80)
    print("The LONG vs SHORT performance gap is primarily driven by:")
    print()
    print(f"  1. HIGH LONG FAILURE RATE: {len(long_quick_sl)} out of {len(longs)} LONG trades")
    print(f"     fail quickly (within 20 bars), representing {long_quick_rate:.1f}% of all")
    print(f"     LONG trades. This drags down the overall LONG win rate.")
    print()
    print(f"  2. SMALL SHORT SAMPLE: Only {len(shorts)} SHORT trades vs {len(longs)} LONG trades.")
    print(f"     The 66.7% SHORT win rate is based on a much smaller sample and")
    print(f"     may not be statistically robust.")
    print()
    print(f"  3. BETTER SHORT ASYMMETRY: SHORT winners average {statistics.mean(short_winner_r) if short_winner_r else 0:.3f}R")
    print(f"     vs LONG winners at {statistics.mean(long_winner_r) if long_winner_r else 0:.3f}R, making SHORT trades more")
    print(f"     profitable when they work.")
    print()
    print("="*80)


if __name__ == '__main__':
    analyze_directional_bias()
