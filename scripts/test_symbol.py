#!/usr/bin/env python3
"""
Quick Symbol/Timeframe Tester

Usage:
    python scripts/test_symbol.py SOLUSDT 1h
    python scripts/test_symbol.py XRPUSDT 15m
    python scripts/test_symbol.py DOGEUSDT 15m,1h
"""
import sys
import argparse
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, '.')

from core.binance_client import BinanceClient
from core import calculate_indicators
from strategies import check_signal
from core.config import DEFAULT_STRATEGY_CONFIG


def test_symbol(symbol: str, timeframe: str, days: int = 365):
    """Test a symbol/timeframe combination."""
    client = BinanceClient()

    # Calculate candles needed
    candles_per_day = {'1m': 1440, '5m': 288, '15m': 96, '1h': 24, '4h': 6}
    total_candles = candles_per_day.get(timeframe, 24) * days

    print(f"\nFetching {total_candles} candles ({days} days)...")
    df = client.get_klines_paginated(symbol, timeframe, total_candles=total_candles)
    print(f"Candles fetched: {len(df)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Calculate indicators
    df = calculate_indicators(df, timeframe)

    # Scan for signals
    signals = []
    for i in range(200, len(df) - 1):
        sig, entry, tp, sl, reason, debug = check_signal(
            df, DEFAULT_STRATEGY_CONFIG, index=i, return_debug=True
        )
        if sig in ['LONG', 'SHORT']:
            signals.append({
                'date': df.index[i],
                'type': sig,
                'entry': entry,
                'tp': tp,
                'sl': sl
            })

    print(f"Signals found: {len(signals)}")

    if not signals:
        return {'signals': 0, 'trades': 0, 'win_rate': 0, 'e_r': 0}

    # Simple backtest
    wins = 0
    losses = 0
    total_r = 0.0

    for s in signals:
        idx = df.index.get_loc(s['date'])
        if idx + 50 >= len(df):
            continue

        entry = s['entry']
        tp = s['tp']
        sl = s['sl']

        # Check next 50 candles for result
        for j in range(1, 50):
            if idx + j >= len(df):
                break
            candle = df.iloc[idx + j]

            if s['type'] == 'LONG':
                if candle['low'] <= sl:
                    losses += 1
                    total_r -= 1
                    break
                elif candle['high'] >= tp:
                    wins += 1
                    r = (tp - entry) / (entry - sl)
                    total_r += r
                    break
            else:  # SHORT
                if candle['high'] >= sl:
                    losses += 1
                    total_r -= 1
                    break
                elif candle['low'] <= tp:
                    wins += 1
                    r = (entry - tp) / (sl - entry)
                    total_r += r
                    break

    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0
    e_r = (total_r / total) if total > 0 else 0

    return {
        'signals': len(signals),
        'trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_r': total_r,
        'e_r': e_r
    }


def print_result(symbol: str, timeframe: str, result: dict):
    """Print formatted result."""
    print(f"\n{'='*50}")
    print(f"RESULT: {symbol} on {timeframe}")
    print(f"{'='*50}")
    print(f"  Signals:   {result['signals']}")
    print(f"  Trades:    {result['trades']}")

    if result['trades'] > 0:
        print(f"  Wins:      {result['wins']}")
        print(f"  Losses:    {result['losses']}")
        print(f"  Win Rate:  {result['win_rate']:.1f}%")
        print(f"  Total R:   {result['total_r']:.2f}")
        print(f"  E[R]:      {result['e_r']:.3f}")

        # Recommendation (Kaufman: minimum 50 trades for statistical significance)
        print()
        if result['trades'] >= 50 and result['e_r'] > 0.1:
            print(f"  ✅ RECOMMEND adding {symbol}-{timeframe} to portfolio")
        elif result['trades'] >= 30 and result['e_r'] > 0:
            print(f"  ⚠️  MARGINAL - consider with caution")
        elif result['e_r'] < 0:
            print(f"  ❌ AVOID - negative expectancy")
        else:
            print(f"  ⚪ INSUFFICIENT DATA - need more trades")
    else:
        print(f"  ⚪ NO COMPLETED TRADES")

    # Profit estimation (for $1000 account, 1.75% risk)
    if result['trades'] > 0 and result['e_r'] > 0:
        est_profit = result['e_r'] * 17.50 * result['trades']
        print(f"\n  Est. Profit ($1000, 1.75% risk): ${est_profit:.0f}")


def main():
    parser = argparse.ArgumentParser(description='Test a symbol/timeframe combination')
    parser.add_argument('symbol', help='Symbol to test (e.g., SOLUSDT)')
    parser.add_argument('timeframes', help='Timeframe(s) to test (e.g., 1h or 15m,1h)')
    parser.add_argument('--days', type=int, default=365, help='Days of history (default: 365)')

    args = parser.parse_args()

    symbol = args.symbol.upper()
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]

    print(f"\n{'#'*50}")
    print(f"# Testing {symbol}")
    print(f"# Timeframes: {', '.join(timeframes)}")
    print(f"# Period: {args.days} days")
    print(f"{'#'*50}")

    for tf in timeframes:
        result = test_symbol(symbol, tf, args.days)
        print_result(symbol, tf, result)


if __name__ == '__main__':
    main()
