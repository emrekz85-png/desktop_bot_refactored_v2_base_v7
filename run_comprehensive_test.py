#!/usr/bin/env python3
"""
Comprehensive Strategy & Pattern Test

Tests all strategies and filter combinations on a single symbol/timeframe.

Usage:
    python run_comprehensive_test.py BTCUSDT 15m
    python run_comprehensive_test.py BTCUSDT 15m --days 180
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run import fetch_data, make_filter_flags, run_backtest
from core.at_scenario_analyzer import check_core_signal
from strategies.pbema_retest import check_pbema_retest_signal
from runners.run_filter_combo_test import apply_filters, simulate_trade


def test_ssl_flow_strategy(df, filter_configs):
    """Test SSL Flow strategy with different filter combinations."""
    results = []

    for name, filters in filter_configs:
        flags = make_filter_flags(filters)
        result = run_backtest(df, flags)
        result['name'] = name
        result['filters'] = filters
        result['strategy'] = 'SSL Flow'
        results.append(result)

    return results


def test_ssl_with_momentum_exit(df, filter_configs):
    """Test SSL Flow strategy with momentum-based exit."""
    from core.at_scenario_analyzer import check_core_signal
    from runners.run_filter_combo_test import apply_filters, simulate_trade

    results = []

    for name, filters, use_momentum in filter_configs:
        signals = []
        trades = []
        exit_types = {"TP": 0, "SL": 0, "MOMENTUM": 0, "EOD": 0}

        # Collect signals
        for i in range(200, len(df) - 10):
            signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)
            if signal_type is None:
                continue

            # Apply filters
            flags = make_filter_flags(filters)
            passed, _ = apply_filters(
                df=df, index=i, signal_type=signal_type,
                entry_price=entry, sl_price=sl,
                **flags
            )
            if not passed:
                continue

            signals.append({
                'idx': i, 'type': signal_type,
                'entry': entry, 'tp': tp, 'sl': sl
            })

        # Simulate trades (no overlapping)
        last_exit = -1
        for sig in signals:
            if sig['idx'] <= last_exit:
                continue

            trade = simulate_trade(
                df, sig['idx'], sig['type'],
                sig['entry'], sig['tp'], sig['sl'],
                position_size=35.0,
                use_momentum_exit=use_momentum
            )
            if trade:
                trades.append(trade)
                last_exit = trade.get('exit_idx', sig['idx'] + 20)
                exit_types[trade.get('exit_type', 'EOD')] += 1

        if trades:
            wins = sum(1 for t in trades if t['win'])
            pnl = sum(t['pnl'] for t in trades)
            wr = 100 * wins / len(trades)
        else:
            wins, pnl, wr = 0, 0, 0

        results.append({
            'name': name,
            'trades': len(trades),
            'wr': wr,
            'pnl': pnl,
            'wins': wins,
            'losses': len(trades) - wins,
            'exit_types': exit_types,
            'momentum_enabled': use_momentum,
        })

    return results


def test_pbema_retest_strategy(df, use_filters=True, use_momentum_exit=False):
    """Test PBEMA Retest strategy with optional momentum exit."""
    signals = []
    trades = []
    exit_types = {"TP": 0, "SL": 0, "MOMENTUM": 0, "EOD": 0}

    for i in range(200, len(df) - 10):
        # Generate PBEMA Retest signal
        result = check_pbema_retest_signal(df, index=i)
        signal_type, entry, tp, sl, reason = result[:5]

        if signal_type is None:
            continue

        # Optionally apply filters
        # NOTE: PBEMA Retest has its own trend detection (breakout), so NO regime filter
        if use_filters:
            passed, filter_reason = apply_filters(
                df=df,
                index=i,
                signal_type=signal_type,
                entry_price=entry,
                sl_price=sl,
                use_regime_filter=False,  # PBEMA has its own trend detection
                use_min_sl_filter=True,
            )
            if not passed:
                continue

        signals.append({
            'idx': i,
            'type': signal_type,
            'entry': entry,
            'tp': tp,
            'sl': sl,
            'reason': reason,
        })

    # Simulate trades (no overlapping - same as SSL Flow)
    last_exit = -1
    for sig in signals:
        if sig['idx'] <= last_exit:
            continue

        trade = simulate_trade(
            df, sig['idx'], sig['type'],
            sig['entry'], sig['tp'], sig['sl'],
            position_size=35.0,
            use_momentum_exit=use_momentum_exit
        )
        if trade:
            trades.append(trade)
            last_exit = trade.get('exit_idx', sig['idx'] + 20)
            exit_types[trade.get('exit_type', 'EOD')] += 1

    # Calculate stats
    if not trades:
        name_suffix = ''
        if use_filters:
            name_suffix += ' + Filters'
        if use_momentum_exit:
            name_suffix += ' + MomExit'
        return {
            'name': 'PBEMA Retest' + name_suffix,
            'strategy': 'PBEMA Retest',
            'signals': len(signals),
            'trades': 0,
            'wr': 0,
            'pnl': 0,
            'wins': 0,
            'losses': 0,
            'exit_types': exit_types,
        }

    wins = sum(1 for t in trades if t['win'])
    losses = len(trades) - wins
    pnl = sum(t['pnl'] for t in trades)
    wr = 100 * wins / len(trades) if trades else 0

    name_suffix = ''
    if use_filters:
        name_suffix += ' + Filters'
    if use_momentum_exit:
        name_suffix += ' + MomExit'

    return {
        'name': 'PBEMA Retest' + name_suffix,
        'strategy': 'PBEMA Retest',
        'signals': len(signals),
        'trades': len(trades),
        'wr': wr,
        'pnl': pnl,
        'wins': wins,
        'losses': losses,
        'exit_types': exit_types,
    }


def test_combined_strategies(df):
    """Test combining both strategies."""
    all_signals = []

    # Collect SSL Flow signals
    for i in range(200, len(df) - 10):
        signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)
        if signal_type:
            passed, _ = apply_filters(
                df=df, index=i, signal_type=signal_type,
                entry_price=entry, sl_price=sl,
                use_regime_filter=True,
                use_at_flat_filter=True,
                use_min_sl_filter=True,
            )
            if passed:
                all_signals.append({
                    'idx': i, 'type': signal_type, 'entry': entry,
                    'tp': tp, 'sl': sl, 'strategy': 'SSL Flow'
                })

    # Collect PBEMA Retest signals
    # NOTE: PBEMA Retest has its own trend detection and RR calc, no external filters
    for i in range(200, len(df) - 10):
        result = check_pbema_retest_signal(df, index=i)
        signal_type, entry, tp, sl, reason = result[:5]
        if signal_type:
            # Check if not duplicate (same candle as SSL Flow signal)
            if not any(s['idx'] == i for s in all_signals):
                all_signals.append({
                    'idx': i, 'type': signal_type, 'entry': entry,
                    'tp': tp, 'sl': sl, 'strategy': 'PBEMA Retest'
                })

    # Sort by index
    all_signals.sort(key=lambda x: x['idx'])

    # Simulate trades (no overlapping)
    trades = []
    last_exit = -1

    for sig in all_signals:
        if sig['idx'] <= last_exit:
            continue

        trade = simulate_trade(
            df, sig['idx'], sig['type'],
            sig['entry'], sig['tp'], sig['sl'],
            position_size=35.0
        )
        if trade:
            trade['strategy'] = sig['strategy']
            trades.append(trade)
            last_exit = trade.get('exit_idx', sig['idx'] + 10)

    if not trades:
        return {'name': 'Combined (SSL + PBEMA)', 'trades': 0, 'wr': 0, 'pnl': 0}

    wins = sum(1 for t in trades if t['win'])
    pnl = sum(t['pnl'] for t in trades)
    ssl_trades = sum(1 for t in trades if t.get('strategy') == 'SSL Flow')
    pbema_trades = sum(1 for t in trades if t.get('strategy') == 'PBEMA Retest')

    return {
        'name': 'Combined (SSL + PBEMA)',
        'strategy': 'Combined',
        'trades': len(trades),
        'wr': 100 * wins / len(trades),
        'pnl': pnl,
        'ssl_trades': ssl_trades,
        'pbema_trades': pbema_trades,
    }


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Strategy Test')
    parser.add_argument('symbol', help='Symbol (e.g., BTCUSDT)')
    parser.add_argument('timeframe', help='Timeframe (e.g., 15m)')
    parser.add_argument('--days', type=int, default=365, help='Days of data')
    args = parser.parse_args()

    print("=" * 80)
    print(f"COMPREHENSIVE STRATEGY TEST: {args.symbol} {args.timeframe}")
    print(f"Period: {args.days} days")
    print("=" * 80)
    print()

    # Fetch data
    print("[1/5] Fetching data...")
    df = fetch_data(args.symbol, args.timeframe, days=args.days)
    print(f"      {len(df)} candles loaded")
    print()

    # ========== SSL FLOW TESTS ==========
    print("[2/5] Testing SSL Flow Strategy...")
    print("-" * 80)

    ssl_configs = [
        ("SSL: Baseline (regime)", ["regime"]),
        ("SSL: Default Config", ["regime", "at_flat_filter", "min_sl_filter"]),
        ("SSL: + SSL Slope", ["regime", "at_flat_filter", "min_sl_filter", "ssl_slope_filter"]),
        ("SSL: + HTF Bounce", ["regime", "at_flat_filter", "min_sl_filter", "htf_bounce"]),
        ("SSL: + SSL Dynamic", ["regime", "at_flat_filter", "min_sl_filter", "ssl_dynamic_support"]),
        ("SSL: + Liquidity Grab", ["regime", "at_flat_filter", "min_sl_filter", "liquidity_grab"]),
        ("SSL: + Momentum Loss", ["regime", "at_flat_filter", "min_sl_filter", "momentum_loss"]),
    ]

    ssl_results = test_ssl_flow_strategy(df, ssl_configs)

    print(f"{'Config':<40} {'Trades':<8} {'WR':<8} {'PnL':<12}")
    print("-" * 80)
    for r in ssl_results:
        print(f"{r['name']:<40} {r['trades']:<8} {r['wr']:<8.1f} ${r['pnl']:<12.2f}")
    print()

    # ========== MOMENTUM EXIT TESTS ==========
    print("[3/6] Testing Momentum Exit (Pattern 1)...")
    print("-" * 80)

    momentum_configs = [
        ("SSL Default (No MomExit)", ["regime", "at_flat_filter", "min_sl_filter"], False),
        ("SSL Default + MomExit", ["regime", "at_flat_filter", "min_sl_filter"], True),
    ]

    momentum_results = test_ssl_with_momentum_exit(df, momentum_configs)

    print(f"{'Config':<35} {'Trades':<8} {'WR':<8} {'PnL':<10} {'Exit Types':<30}")
    print("-" * 100)
    for r in momentum_results:
        exit_str = f"TP:{r['exit_types']['TP']} SL:{r['exit_types']['SL']} MOM:{r['exit_types']['MOMENTUM']}"
        print(f"{r['name']:<35} {r['trades']:<8} {r['wr']:<8.1f} ${r['pnl']:<10.2f} {exit_str:<30}")
    print()

    # ========== PBEMA RETEST TESTS ==========
    print("[4/6] Testing PBEMA Retest Strategy...")
    print("-" * 80)

    # PBEMA Retest works best WITHOUT external filters
    # It has its own trend detection (breakout) and RR calculation
    pbema_no_filter = test_pbema_retest_strategy(df, use_filters=False, use_momentum_exit=False)
    pbema_with_momentum = test_pbema_retest_strategy(df, use_filters=False, use_momentum_exit=True)

    pbema_results = [pbema_no_filter, pbema_with_momentum]

    print(f"{'Config':<35} {'Signals':<10} {'Trades':<8} {'WR':<8} {'PnL':<10} {'Exit Types':<30}")
    print("-" * 100)
    for r in pbema_results:
        exit_str = f"TP:{r['exit_types']['TP']} SL:{r['exit_types']['SL']} MOM:{r['exit_types']['MOMENTUM']}"
        print(f"{r['name']:<35} {r['signals']:<10} {r['trades']:<8} {r['wr']:<8.1f} ${r['pnl']:<10.2f} {exit_str:<30}")
    print()

    # ========== COMBINED TEST ==========
    print("[5/6] Testing Combined Strategies...")
    print("-" * 80)

    combined = test_combined_strategies(df)

    print(f"Combined Results:")
    print(f"  Total Trades: {combined['trades']}")
    print(f"  Win Rate: {combined['wr']:.1f}%")
    print(f"  PnL: ${combined['pnl']:.2f}")
    if 'ssl_trades' in combined:
        print(f"  - SSL Flow trades: {combined['ssl_trades']}")
        print(f"  - PBEMA Retest trades: {combined['pbema_trades']}")
    print()

    # ========== FINAL SUMMARY ==========
    print("[6/6] Final Summary")
    print("=" * 80)

    # Find best SSL config
    best_ssl = max(ssl_results, key=lambda x: x['pnl'])

    print(f"\n{'STRATEGY COMPARISON':^80}")
    print("=" * 80)
    print(f"{'Strategy':<45} {'Trades':<10} {'WR':<10} {'PnL':<15}")
    print("-" * 80)

    # Best SSL
    print(f"{'SSL Flow (Best: ' + best_ssl['name'].split(': ')[1] + ')':<45} {best_ssl['trades']:<10} {best_ssl['wr']:<10.1f} ${best_ssl['pnl']:<15.2f}")

    # PBEMA with filters
    print(f"{'PBEMA Retest':<45} {pbema_no_filter['trades']:<10} {pbema_no_filter['wr']:<10.1f} ${pbema_no_filter['pnl']:<15.2f}")

    # Combined
    print(f"{'Combined (SSL + PBEMA)':<45} {combined['trades']:<10} {combined['wr']:<10.1f} ${combined['pnl']:<15.2f}")

    print("=" * 80)

    # Recommendation
    print("\n RECOMMENDATION:")
    print("-" * 80)

    all_results = [
        ('SSL Flow', best_ssl['pnl'], best_ssl['trades']),
        ('PBEMA Retest', pbema_no_filter['pnl'], pbema_no_filter['trades']),
        ('Combined', combined['pnl'], combined['trades']),
    ]

    best_overall = max(all_results, key=lambda x: x[1])

    if best_overall[1] > 0:
        print(f"  Best Strategy: {best_overall[0]}")
        print(f"  PnL: ${best_overall[1]:.2f} with {best_overall[2]} trades")
    else:
        print("  WARNING: No profitable strategy found!")
        print("  Consider: Different timeframe, symbol, or parameter tuning")

    print("=" * 80)

    # ========== SAVE RESULTS ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/results") / f"comprehensive_{args.symbol}_{args.timeframe}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare results for JSON
    results_data = {
        "timestamp": timestamp,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "days": args.days,
        "candles": len(df),
        "ssl_flow_results": ssl_results,
        "pbema_retest_results": pbema_results,
        "combined_result": combined,
        "best_strategy": {
            "name": best_overall[0],
            "pnl": best_overall[1],
            "trades": best_overall[2],
        },
        "recommendation": best_ssl['name'] if best_ssl['pnl'] > 0 else "No profitable config",
    }

    # Save JSON
    json_path = output_dir / "result.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    # Save summary text
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"COMPREHENSIVE TEST: {args.symbol} {args.timeframe}\n")
        f.write(f"Period: {args.days} days ({len(df)} candles)\n")
        f.write(f"Date: {timestamp}\n")
        f.write("=" * 60 + "\n\n")

        f.write("SSL FLOW RESULTS:\n")
        f.write("-" * 60 + "\n")
        for r in ssl_results:
            f.write(f"{r['name']:<40} {r['trades']:>4} trades  {r['wr']:>5.1f}% WR  ${r['pnl']:>8.2f}\n")
        f.write("\n")

        f.write("PBEMA RETEST RESULTS:\n")
        f.write("-" * 60 + "\n")
        for r in pbema_results:
            f.write(f"{r['name']:<40} {r['trades']:>4} trades  {r['wr']:>5.1f}% WR  ${r['pnl']:>8.2f}\n")
        f.write("\n")

        f.write("COMBINED RESULT:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Trades: {combined['trades']}, WR: {combined['wr']:.1f}%, PnL: ${combined['pnl']:.2f}\n")
        f.write("\n")

        f.write("BEST STRATEGY:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{best_overall[0]}: ${best_overall[1]:.2f} with {best_overall[2]} trades\n")

    print(f"\n Results saved to: {output_dir}")
    print(f"   - {json_path.name}")
    print(f"   - {summary_path.name}")


if __name__ == "__main__":
    main()
