#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis of SSL Flow Trades

Analyzes trades.json to extract:
- Trade classification (winners/losers, long/short)
- Pattern analysis (common characteristics)
- Temporal analysis (hold times, exit timing)
- R-multiple analysis
- Directional performance analysis
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import statistics


def load_trades(file_path: str) -> List[Dict[str, Any]]:
    """Load trades from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_r_multiple(trade: Dict[str, Any]) -> float:
    """
    Calculate R-multiple for a trade.
    R-multiple = Profit / Initial Risk

    For LONG:
        Risk = entry - SL
        Profit = exit - entry
    For SHORT:
        Risk = SL - entry
        Profit = entry - exit
    """
    entry = trade['entry_price']
    exit_price = trade['exit_price']
    sl = trade['sl_price']
    signal_type = trade['signal_type']

    if signal_type == 'LONG':
        risk = entry - sl
        profit = exit_price - entry
    else:  # SHORT
        risk = sl - entry
        profit = entry - exit_price

    # Avoid division by zero
    if risk == 0:
        return 0.0

    return profit / risk


def classify_trades(trades: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Classify trades into categories."""
    classification = {
        'all': trades,
        'winners': [],
        'losers': [],
        'longs': [],
        'shorts': [],
        'long_winners': [],
        'long_losers': [],
        'short_winners': [],
        'short_losers': [],
        'tp_exits': [],
        'sl_exits': [],
    }

    for trade in trades:
        # Winner/Loser
        if trade['win']:
            classification['winners'].append(trade)
        else:
            classification['losers'].append(trade)

        # Long/Short
        if trade['signal_type'] == 'LONG':
            classification['longs'].append(trade)
            if trade['win']:
                classification['long_winners'].append(trade)
            else:
                classification['long_losers'].append(trade)
        else:
            classification['shorts'].append(trade)
            if trade['win']:
                classification['short_winners'].append(trade)
            else:
                classification['short_losers'].append(trade)

        # Exit type
        if trade['exit_reason'] == 'TP':
            classification['tp_exits'].append(trade)
        elif trade['exit_reason'] == 'SL':
            classification['sl_exits'].append(trade)

    return classification


def calculate_metrics(trades: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Calculate comprehensive metrics for a group of trades."""
    if not trades:
        return {
            'label': label,
            'count': 0,
            'win_rate': 0.0,
            'avg_pnl': 0.0,
            'total_pnl': 0.0,
            'avg_pnl_pct': 0.0,
            'avg_hold_time': 0.0,
            'avg_r_multiple': 0.0,
        }

    winners = [t for t in trades if t['win']]
    losers = [t for t in trades if not t['win']]

    pnls = [t['pnl'] for t in trades]
    pnl_pcts = [t['pnl_pct'] for t in trades]
    hold_times = [t['bars_held'] for t in trades]
    r_multiples = [calculate_r_multiple(t) for t in trades]

    metrics = {
        'label': label,
        'count': len(trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(trades) * 100,
        'avg_pnl': statistics.mean(pnls),
        'median_pnl': statistics.median(pnls),
        'total_pnl': sum(pnls),
        'avg_pnl_pct': statistics.mean(pnl_pcts),
        'median_pnl_pct': statistics.median(pnl_pcts),
        'avg_hold_time': statistics.mean(hold_times),
        'median_hold_time': statistics.median(hold_times),
        'min_hold_time': min(hold_times),
        'max_hold_time': max(hold_times),
        'avg_r_multiple': statistics.mean(r_multiples),
        'median_r_multiple': statistics.median(r_multiples),
    }

    # Separate winner/loser metrics
    if winners:
        winner_pnls = [t['pnl'] for t in winners]
        winner_pnl_pcts = [t['pnl_pct'] for t in winners]
        winner_hold_times = [t['bars_held'] for t in winners]
        winner_r_multiples = [calculate_r_multiple(t) for t in winners]

        metrics['winner_avg_pnl'] = statistics.mean(winner_pnls)
        metrics['winner_avg_pnl_pct'] = statistics.mean(winner_pnl_pcts)
        metrics['winner_avg_hold_time'] = statistics.mean(winner_hold_times)
        metrics['winner_avg_r_multiple'] = statistics.mean(winner_r_multiples)
        metrics['winner_median_r_multiple'] = statistics.median(winner_r_multiples)

    if losers:
        loser_pnls = [t['pnl'] for t in losers]
        loser_pnl_pcts = [t['pnl_pct'] for t in losers]
        loser_hold_times = [t['bars_held'] for t in losers]
        loser_r_multiples = [calculate_r_multiple(t) for t in losers]

        metrics['loser_avg_pnl'] = statistics.mean(loser_pnls)
        metrics['loser_avg_pnl_pct'] = statistics.mean(loser_pnl_pcts)
        metrics['loser_avg_hold_time'] = statistics.mean(loser_hold_times)
        metrics['loser_avg_r_multiple'] = statistics.mean(loser_r_multiples)
        metrics['loser_median_r_multiple'] = statistics.median(loser_r_multiples)

    return metrics


def analyze_hold_time_distribution(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze hold time distribution with histogram buckets."""
    hold_times = [t['bars_held'] for t in trades]

    # Define buckets (in bars)
    buckets = {
        '0-10': 0,
        '11-20': 0,
        '21-50': 0,
        '51-100': 0,
        '101-200': 0,
        '201-500': 0,
        '500+': 0,
    }

    for ht in hold_times:
        if ht <= 10:
            buckets['0-10'] += 1
        elif ht <= 20:
            buckets['11-20'] += 1
        elif ht <= 50:
            buckets['21-50'] += 1
        elif ht <= 100:
            buckets['51-100'] += 1
        elif ht <= 200:
            buckets['101-200'] += 1
        elif ht <= 500:
            buckets['201-500'] += 1
        else:
            buckets['500+'] += 1

    return {
        'buckets': buckets,
        'avg': statistics.mean(hold_times),
        'median': statistics.median(hold_times),
        'min': min(hold_times),
        'max': max(hold_times),
    }


def analyze_exit_timing(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze when exits occur (early vs late)."""
    sl_trades = [t for t in trades if t['exit_reason'] == 'SL']
    tp_trades = [t for t in trades if t['exit_reason'] == 'TP']

    sl_hold_times = [t['bars_held'] for t in sl_trades]
    tp_hold_times = [t['bars_held'] for t in tp_trades]

    # Categorize SL hits by timing
    sl_timing = {
        'quick_sl (0-20 bars)': len([ht for ht in sl_hold_times if ht <= 20]),
        'medium_sl (21-100 bars)': len([ht for ht in sl_hold_times if 21 <= ht <= 100]),
        'late_sl (100+ bars)': len([ht for ht in sl_hold_times if ht > 100]),
    }

    return {
        'sl_count': len(sl_trades),
        'tp_count': len(tp_trades),
        'sl_avg_hold_time': statistics.mean(sl_hold_times) if sl_hold_times else 0,
        'tp_avg_hold_time': statistics.mean(tp_hold_times) if tp_hold_times else 0,
        'sl_timing': sl_timing,
    }


def analyze_r_multiples(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detailed R-multiple analysis."""
    r_multiples = [calculate_r_multiple(t) for t in trades]

    winners = [t for t in trades if t['win']]
    losers = [t for t in trades if not t['win']]

    winner_r_multiples = [calculate_r_multiple(t) for t in winners]
    loser_r_multiples = [calculate_r_multiple(t) for t in losers]

    # Distribution buckets
    buckets = {
        'huge_loss (<-2R)': len([r for r in r_multiples if r < -2]),
        'loss (-2R to -0.5R)': len([r for r in r_multiples if -2 <= r < -0.5]),
        'small_loss (-0.5R to 0R)': len([r for r in r_multiples if -0.5 <= r < 0]),
        'small_win (0R to 0.5R)': len([r for r in r_multiples if 0 <= r < 0.5]),
        'win (0.5R to 2R)': len([r for r in r_multiples if 0.5 <= r < 2]),
        'huge_win (2R+)': len([r for r in r_multiples if r >= 2]),
    }

    return {
        'all_trades': {
            'avg_r': statistics.mean(r_multiples),
            'median_r': statistics.median(r_multiples),
            'min_r': min(r_multiples),
            'max_r': max(r_multiples),
        },
        'winners': {
            'avg_r': statistics.mean(winner_r_multiples) if winner_r_multiples else 0,
            'median_r': statistics.median(winner_r_multiples) if winner_r_multiples else 0,
            'min_r': min(winner_r_multiples) if winner_r_multiples else 0,
            'max_r': max(winner_r_multiples) if winner_r_multiples else 0,
        },
        'losers': {
            'avg_r': statistics.mean(loser_r_multiples) if loser_r_multiples else 0,
            'median_r': statistics.median(loser_r_multiples) if loser_r_multiples else 0,
            'min_r': min(loser_r_multiples) if loser_r_multiples else 0,
            'max_r': max(loser_r_multiples) if loser_r_multiples else 0,
        },
        'distribution': buckets,
    }


def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)


def print_section_header(title: str):
    """Print a section header."""
    print_separator()
    print(f"  {title}")
    print_separator()
    print()


def print_metrics_table(metrics: Dict[str, Any]):
    """Print metrics in a formatted table."""
    print(f"{'Metric':<30} {'Value':<20}")
    print('-' * 50)
    print(f"{'Total Trades':<30} {metrics['count']:<20}")
    print(f"{'Winners':<30} {metrics['winners']:<20}")
    print(f"{'Losers':<30} {metrics['losers']:<20}")
    print(f"{'Win Rate':<30} {metrics['win_rate']:.2f}%")
    print(f"{'Total PnL':<30} ${metrics['total_pnl']:.2f}")
    print(f"{'Avg PnL':<30} ${metrics['avg_pnl']:.2f}")
    print(f"{'Median PnL':<30} ${metrics['median_pnl']:.2f}")
    print(f"{'Avg PnL %':<30} {metrics['avg_pnl_pct']:.2f}%")
    print(f"{'Median PnL %':<30} {metrics['median_pnl_pct']:.2f}%")
    print(f"{'Avg Hold Time (bars)':<30} {metrics['avg_hold_time']:.1f}")
    print(f"{'Median Hold Time (bars)':<30} {metrics['median_hold_time']:.1f}")
    print(f"{'Min Hold Time (bars)':<30} {metrics['min_hold_time']}")
    print(f"{'Max Hold Time (bars)':<30} {metrics['max_hold_time']}")
    print(f"{'Avg R-Multiple':<30} {metrics['avg_r_multiple']:.3f}R")
    print(f"{'Median R-Multiple':<30} {metrics['median_r_multiple']:.3f}R")

    if 'winner_avg_pnl' in metrics:
        print()
        print("WINNER STATISTICS:")
        print(f"{'  Avg PnL':<30} ${metrics['winner_avg_pnl']:.2f}")
        print(f"{'  Avg PnL %':<30} {metrics['winner_avg_pnl_pct']:.2f}%")
        print(f"{'  Avg Hold Time':<30} {metrics['winner_avg_hold_time']:.1f} bars")
        print(f"{'  Avg R-Multiple':<30} {metrics['winner_avg_r_multiple']:.3f}R")
        print(f"{'  Median R-Multiple':<30} {metrics['winner_median_r_multiple']:.3f}R")

    if 'loser_avg_pnl' in metrics:
        print()
        print("LOSER STATISTICS:")
        print(f"{'  Avg PnL':<30} ${metrics['loser_avg_pnl']:.2f}")
        print(f"{'  Avg PnL %':<30} {metrics['loser_avg_pnl_pct']:.2f}%")
        print(f"{'  Avg Hold Time':<30} {metrics['loser_avg_hold_time']:.1f} bars")
        print(f"{'  Avg R-Multiple':<30} {metrics['loser_avg_r_multiple']:.3f}R")
        print(f"{'  Median R-Multiple':<30} {metrics['loser_median_r_multiple']:.3f}R")

    print()


def main():
    """Main analysis function."""
    # Load trades
    trades_file = '/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/data/results/BTCUSDT_15m_20260104_115258_full_v3/trades.json'
    trades = load_trades(trades_file)

    print_section_header("SSL FLOW TRADE ANALYSIS")
    print(f"Analyzing {len(trades)} trades from: {trades_file}")
    print()

    # Classify trades
    classified = classify_trades(trades)

    # ========================================
    # 1. OVERALL METRICS
    # ========================================
    print_section_header("1. OVERALL METRICS")
    all_metrics = calculate_metrics(trades, 'All Trades')
    print_metrics_table(all_metrics)

    # ========================================
    # 2. DIRECTIONAL ANALYSIS
    # ========================================
    print_section_header("2. DIRECTIONAL ANALYSIS (LONG vs SHORT)")

    print("\nLONG TRADES:")
    print_separator('-', 50)
    long_metrics = calculate_metrics(classified['longs'], 'LONG')
    print_metrics_table(long_metrics)

    print("\nSHORT TRADES:")
    print_separator('-', 50)
    short_metrics = calculate_metrics(classified['shorts'], 'SHORT')
    print_metrics_table(short_metrics)

    print("\nDIRECTIONAL COMPARISON:")
    print_separator('-', 50)
    print(f"{'Metric':<35} {'LONG':<20} {'SHORT':<20}")
    print('-' * 75)
    print(f"{'Trade Count':<35} {long_metrics['count']:<20} {short_metrics['count']:<20}")
    print(f"{'Win Rate':<35} {long_metrics['win_rate']:.2f}%{'':<13} {short_metrics['win_rate']:.2f}%")
    print(f"{'Total PnL':<35} ${long_metrics['total_pnl']:.2f}{'':<13} ${short_metrics['total_pnl']:.2f}")
    print(f"{'Avg PnL':<35} ${long_metrics['avg_pnl']:.2f}{'':<13} ${short_metrics['avg_pnl']:.2f}")
    print(f"{'Avg R-Multiple':<35} {long_metrics['avg_r_multiple']:.3f}R{'':<13} {short_metrics['avg_r_multiple']:.3f}R")
    print(f"{'Avg Hold Time':<35} {long_metrics['avg_hold_time']:.1f} bars{'':<10} {short_metrics['avg_hold_time']:.1f} bars")
    print()

    # ========================================
    # 3. WINNER/LOSER PATTERN ANALYSIS
    # ========================================
    print_section_header("3. WINNER/LOSER PATTERN ANALYSIS")

    print("\nWINNERS:")
    print_separator('-', 50)
    winner_metrics = calculate_metrics(classified['winners'], 'Winners')
    print_metrics_table(winner_metrics)

    print("\nLOSERS:")
    print_separator('-', 50)
    loser_metrics = calculate_metrics(classified['losers'], 'Losers')
    print_metrics_table(loser_metrics)

    # Exit type analysis
    print("\nEXIT TYPE BREAKDOWN:")
    print_separator('-', 50)
    print(f"{'Exit Type':<20} {'Count':<10} {'% of Total':<15}")
    print('-' * 45)
    tp_count = len(classified['tp_exits'])
    sl_count = len(classified['sl_exits'])
    print(f"{'TP (Take Profit)':<20} {tp_count:<10} {tp_count/len(trades)*100:.1f}%")
    print(f"{'SL (Stop Loss)':<20} {sl_count:<10} {sl_count/len(trades)*100:.1f}%")
    print()

    # ========================================
    # 4. TEMPORAL ANALYSIS
    # ========================================
    print_section_header("4. TEMPORAL ANALYSIS")

    print("\nHOLD TIME DISTRIBUTION (All Trades):")
    print_separator('-', 50)
    hold_dist = analyze_hold_time_distribution(trades)
    print(f"{'Bucket':<20} {'Count':<10} {'% of Total':<15}")
    print('-' * 45)
    for bucket, count in hold_dist['buckets'].items():
        pct = count / len(trades) * 100
        print(f"{bucket:<20} {count:<10} {pct:.1f}%")
    print()
    print(f"Average hold time: {hold_dist['avg']:.1f} bars")
    print(f"Median hold time: {hold_dist['median']:.1f} bars")
    print(f"Range: {hold_dist['min']} - {hold_dist['max']} bars")
    print()

    print("\nEXIT TIMING ANALYSIS:")
    print_separator('-', 50)
    exit_timing = analyze_exit_timing(trades)
    print(f"Total TP exits: {exit_timing['tp_count']}")
    print(f"Total SL exits: {exit_timing['sl_count']}")
    print(f"Avg hold time (TP): {exit_timing['tp_avg_hold_time']:.1f} bars")
    print(f"Avg hold time (SL): {exit_timing['sl_avg_hold_time']:.1f} bars")
    print()
    print("SL HIT TIMING:")
    for timing, count in exit_timing['sl_timing'].items():
        pct = count / exit_timing['sl_count'] * 100 if exit_timing['sl_count'] > 0 else 0
        print(f"  {timing:<25} {count:<5} ({pct:.1f}%)")
    print()

    # ========================================
    # 5. R-MULTIPLE ANALYSIS
    # ========================================
    print_section_header("5. R-MULTIPLE ANALYSIS")

    r_analysis = analyze_r_multiples(trades)

    print("\nR-MULTIPLE STATISTICS:")
    print_separator('-', 50)
    print(f"{'Category':<20} {'Avg R':<15} {'Median R':<15} {'Min R':<15} {'Max R':<15}")
    print('-' * 80)
    print(f"{'All Trades':<20} {r_analysis['all_trades']['avg_r']:.3f}R{'':<7} {r_analysis['all_trades']['median_r']:.3f}R{'':<7} {r_analysis['all_trades']['min_r']:.3f}R{'':<7} {r_analysis['all_trades']['max_r']:.3f}R")
    print(f"{'Winners':<20} {r_analysis['winners']['avg_r']:.3f}R{'':<7} {r_analysis['winners']['median_r']:.3f}R{'':<7} {r_analysis['winners']['min_r']:.3f}R{'':<7} {r_analysis['winners']['max_r']:.3f}R")
    print(f"{'Losers':<20} {r_analysis['losers']['avg_r']:.3f}R{'':<7} {r_analysis['losers']['median_r']:.3f}R{'':<7} {r_analysis['losers']['min_r']:.3f}R{'':<7} {r_analysis['losers']['max_r']:.3f}R")
    print()

    print("\nR-MULTIPLE DISTRIBUTION:")
    print_separator('-', 50)
    print(f"{'Bucket':<25} {'Count':<10} {'% of Total':<15}")
    print('-' * 50)
    for bucket, count in r_analysis['distribution'].items():
        pct = count / len(trades) * 100
        print(f"{bucket:<25} {count:<10} {pct:.1f}%")
    print()

    # Calculate asymmetry ratio
    avg_win_r = r_analysis['winners']['avg_r']
    avg_loss_r = abs(r_analysis['losers']['avg_r'])
    asymmetry_ratio = avg_win_r / avg_loss_r if avg_loss_r > 0 else 0

    print(f"\nASYMMETRY ANALYSIS:")
    print_separator('-', 50)
    print(f"Average winning R-multiple: {avg_win_r:.3f}R")
    print(f"Average losing R-multiple: {avg_loss_r:.3f}R")
    print(f"Win/Loss Asymmetry Ratio: {asymmetry_ratio:.3f}x")
    if asymmetry_ratio > 1:
        print(f"  → Wins are {asymmetry_ratio:.2f}x larger than losses (POSITIVE asymmetry)")
    else:
        print(f"  → Losses are {1/asymmetry_ratio:.2f}x larger than wins (NEGATIVE asymmetry)")
    print()

    # ========================================
    # 6. DETAILED TRADE EXAMPLES
    # ========================================
    print_section_header("6. DETAILED TRADE EXAMPLES")

    print("\nBEST WINNING TRADES (by R-multiple):")
    print_separator('-', 50)
    winners_with_r = [(t, calculate_r_multiple(t)) for t in classified['winners']]
    winners_with_r.sort(key=lambda x: x[1], reverse=True)

    for i, (trade, r_mult) in enumerate(winners_with_r[:5], 1):
        print(f"\n{i}. {trade['signal_type']} - {trade['entry_time']}")
        print(f"   Entry: ${trade['entry_price']:.2f}, Exit: ${trade['exit_price']:.2f}")
        print(f"   PnL: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
        print(f"   R-Multiple: {r_mult:.3f}R")
        print(f"   Hold time: {trade['bars_held']} bars")
        print(f"   Exit: {trade['exit_reason']}")

    print("\n\nWORST LOSING TRADES (by R-multiple):")
    print_separator('-', 50)
    losers_with_r = [(t, calculate_r_multiple(t)) for t in classified['losers']]
    losers_with_r.sort(key=lambda x: x[1])

    for i, (trade, r_mult) in enumerate(losers_with_r[:5], 1):
        print(f"\n{i}. {trade['signal_type']} - {trade['entry_time']}")
        print(f"   Entry: ${trade['entry_price']:.2f}, Exit: ${trade['exit_price']:.2f}")
        print(f"   PnL: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
        print(f"   R-Multiple: {r_mult:.3f}R")
        print(f"   Hold time: {trade['bars_held']} bars")
        print(f"   Exit: {trade['exit_reason']}")

    # ========================================
    # 7. KEY INSIGHTS
    # ========================================
    print_section_header("7. KEY INSIGHTS & OBSERVATIONS")

    print("DIRECTIONAL BIAS:")
    if long_metrics['win_rate'] > short_metrics['win_rate']:
        diff = long_metrics['win_rate'] - short_metrics['win_rate']
        print(f"  ✓ LONG trades have {diff:.1f}% higher win rate than SHORT")
    else:
        diff = short_metrics['win_rate'] - long_metrics['win_rate']
        print(f"  ✓ SHORT trades have {diff:.1f}% higher win rate than LONG")

    if long_metrics['total_pnl'] > short_metrics['total_pnl']:
        print(f"  ✓ LONG trades generated ${long_metrics['total_pnl']:.2f} vs SHORT ${short_metrics['total_pnl']:.2f}")
    else:
        print(f"  ✓ SHORT trades generated ${short_metrics['total_pnl']:.2f} vs LONG ${long_metrics['total_pnl']:.2f}")

    print("\nHOLD TIME PATTERNS:")
    if winner_metrics['avg_hold_time'] > loser_metrics['avg_hold_time']:
        diff = winner_metrics['avg_hold_time'] - loser_metrics['avg_hold_time']
        print(f"  ✓ Winners hold {diff:.1f} bars LONGER than losers on average")
    else:
        diff = loser_metrics['avg_hold_time'] - winner_metrics['avg_hold_time']
        print(f"  ✓ Losers hold {diff:.1f} bars LONGER than winners on average")

    quick_sl_pct = exit_timing['sl_timing']['quick_sl (0-20 bars)'] / exit_timing['sl_count'] * 100
    print(f"  ✓ {quick_sl_pct:.1f}% of SL hits occur in first 20 bars (quick failures)")

    print("\nR-MULTIPLE FINDINGS:")
    if asymmetry_ratio > 1:
        print(f"  ✓ POSITIVE asymmetry: Winners average {asymmetry_ratio:.2f}x larger than losers")
    else:
        print(f"  ✗ NEGATIVE asymmetry: Losers average {1/asymmetry_ratio:.2f}x larger than winners")

    print(f"  ✓ Best win: {r_analysis['winners']['max_r']:.3f}R")
    print(f"  ✓ Worst loss: {r_analysis['losers']['min_r']:.3f}R")

    print("\nOVERALL PERFORMANCE:")
    print(f"  ✓ Win Rate: {all_metrics['win_rate']:.1f}%")
    print(f"  ✓ Total PnL: ${all_metrics['total_pnl']:.2f}")
    print(f"  ✓ Avg R per trade: {r_analysis['all_trades']['avg_r']:.3f}R")
    print(f"  ✓ Profitability: {'PROFITABLE' if all_metrics['total_pnl'] > 0 else 'UNPROFITABLE'}")

    print()
    print_separator('=', 80)
    print("Analysis complete!")
    print_separator('=', 80)


if __name__ == '__main__':
    main()
