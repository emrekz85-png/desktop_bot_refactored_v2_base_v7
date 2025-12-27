#!/usr/bin/env python3
"""Test all symbols with rolling walk-forward - BATCH MODE"""

import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runners.rolling_wf_optimized import compare_rolling_modes_optimized
from core import BASELINE_CONFIG

if __name__ == '__main__':
    # All symbols to test (excluding already tested BTC, ETH, SOL)
    symbols_to_test = ['HYPEUSDT', 'LINKUSDT', 'BNBUSDT', 'XRPUSDT',
                       'LTCUSDT', 'DOGEUSDT', 'SUIUSDT', 'FARTCOINUSDT']

    print("="*70)
    print("TUM SEMBOLLER ROLLING WALK-FORWARD TEST (BATCH MODE)")
    print(f"   [Symbols: {len(symbols_to_test)} adet - SINGLE RUN]")
    print("   [Timeframes: 5m, 15m, 1h]")
    print("   [Period: 2025-01-01 to 2025-12-26]")
    print("="*70)
    print()

    # Test ALL symbols at once - SINGLE data fetch!
    result = compare_rolling_modes_optimized(
        symbols=symbols_to_test,  # ALL SYMBOLS IN ONE CALL
        timeframes=['5m', '15m', '1h'],
        start_date='2025-01-01',
        end_date='2025-12-26',
        fixed_config=BASELINE_CONFIG,
        verbose=True,
        modes=['weekly'],
    )

    # Extract per-symbol stats
    mode_results = result.get('results', {})
    weekly = mode_results.get('weekly', {})
    trades = weekly.get('trades', [])
    metrics = weekly.get('metrics', {})

    # Group by symbol
    by_symbol = defaultdict(lambda: {'trades': [], 'pnl': 0.0, 'wins': 0})
    for trade in trades:
        sym = trade.get('symbol')
        by_symbol[sym]['trades'].append(trade)
        pnl = float(trade.get('pnl', 0))
        by_symbol[sym]['pnl'] += pnl
        if pnl > 0:
            by_symbol[sym]['wins'] += 1

    # Print results table
    print()
    print("="*70)
    print("TUM SEMBOLLER SONUC TABLOSU")
    print("="*70)
    print()
    print(f"{'Sembol':<15} {'PnL':>12} {'Trades':>8} {'Wins':>6} {'Win%':>8} {'Durum':>8}")
    print("-"*70)

    # Add previously tested symbols (from earlier tests)
    all_results = [
        {'symbol': 'BTCUSDT', 'pnl': 47.24, 'trades': 21, 'wins': 15, 'win_rate': 71.4},
        {'symbol': 'ETHUSDT', 'pnl': 1.91, 'trades': 30, 'wins': 24, 'win_rate': 80.0},
        {'symbol': 'SOLUSDT', 'pnl': -153.84, 'trades': 4, 'wins': 0, 'win_rate': 0.0},
    ]

    # Add new results
    for sym in symbols_to_test:
        data = by_symbol.get(sym, {'trades': [], 'pnl': 0.0, 'wins': 0})
        trade_count = len(data['trades'])
        wins = data['wins']
        pnl = data['pnl']
        win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
        all_results.append({
            'symbol': sym,
            'pnl': pnl,
            'trades': trade_count,
            'wins': wins,
            'win_rate': win_rate,
        })

    # Sort by PnL
    all_results.sort(key=lambda x: x['pnl'], reverse=True)

    for r in all_results:
        status = "OK" if r['pnl'] > 0 else ("~" if r['pnl'] > -10 else "X")
        print(f"{r['symbol']:<15} ${r['pnl']:>+10.2f} {r['trades']:>8} {r['wins']:>6} {r['win_rate']:>7.1f}% {status:>8}")

    # Summary
    print("-"*70)
    profitable = [r for r in all_results if r['pnl'] > 0]
    losing = [r for r in all_results if r['pnl'] <= 0]
    total_pnl = sum(r['pnl'] for r in all_results)
    total_trades = sum(r['trades'] for r in all_results)

    print(f"\nOZET:")
    print(f"   Toplam Sembol: {len(all_results)}")
    print(f"   Karli Semboller: {len(profitable)}")
    print(f"   Zararli Semboller: {len(losing)}")
    print(f"   Toplam Trade: {total_trades}")
    print(f"   Toplam PnL: ${total_pnl:+.2f}")

    # Overall metrics
    print(f"\n   Weekly Mode Metrics:")
    print(f"   Total PnL: ${metrics.get('total_pnl', 0):+.2f}")
    print(f"   Max Drawdown: ${metrics.get('max_drawdown', 0):.2f}")
    print(f"   Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
