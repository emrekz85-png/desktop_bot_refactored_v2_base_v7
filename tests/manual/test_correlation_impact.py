#!/usr/bin/env python3
"""
Test Correlation Management Impact on Backtest Results

Compares backtest performance WITH and WITHOUT correlation management:
- Max 2 positions per direction (LONG or SHORT)
- Position size reduction for correlated assets (50%)

Usage:
    python test_correlation_impact.py [--candles 10000]
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    TRADING_CONFIG,
    SimTradeManager,
    calculate_indicators,
    set_backtest_mode,
)
from core.binance_client import BinanceClient
from strategies import check_signal


def run_backtest(symbols, timeframes, candles, use_correlation_management=True):
    """Run a simple backtest with or without correlation management."""
    set_backtest_mode(True)

    # Create trade manager
    tm = SimTradeManager(initial_balance=TRADING_CONFIG["initial_balance"])

    # Enable/disable correlation management
    tm._use_correlation_management = use_correlation_management

    # Fetch data (create new client for each to avoid state issues)
    all_data = {}

    print(f"\nFetching data for {len(symbols)} symbols, {len(timeframes)} timeframes...")
    for symbol in symbols:
        for tf in timeframes:
            try:
                client = BinanceClient()  # New client for each fetch
                df = client.get_klines(symbol, tf, limit=candles)
                if df is not None and len(df) > 200:
                    df = calculate_indicators(df, tf)
                    all_data[(symbol, tf)] = df
                    print(f"  {symbol}-{tf}: {len(df)} candles")
                else:
                    print(f"  {symbol}-{tf}: No data or too few candles")
            except Exception as e:
                print(f"  Error fetching {symbol}-{tf}: {e}")

    print(f"  Loaded {len(all_data)} streams")

    # Default config for signals
    default_config = {
        "rr": 2.0,
        "rsi": 70,
        "at_active": True,
        "use_trailing": False,
        "use_partial": True,
        "strategy_mode": "ssl_flow",
    }

    # Run backtest
    blocked_by_correlation = 0
    signals_generated = 0

    for (symbol, tf), df in all_data.items():
        for i in range(200, len(df)):
            # Update existing trades
            row = df.iloc[i]
            candle_time = row.name if hasattr(row.name, 'strftime') else datetime.now()

            closed = tm.update_trades(
                symbol=symbol,
                tf=tf,
                candle_high=float(row["high"]),
                candle_low=float(row["low"]),
                candle_close=float(row["close"]),
                candle_time_utc=candle_time,
                pb_top=float(row.get("pb_ema_top", row["high"])),
                pb_bot=float(row.get("pb_ema_bot", row["low"])),
            )

            # Check for new signals (only if no open trade on this stream)
            has_open_trade = any(
                t.get("symbol") == symbol and t.get("timeframe") == tf
                for t in tm.open_trades
            )

            if not has_open_trade:
                signal_type, entry, tp, sl, reason = check_signal(df, default_config, index=i)

                if signal_type in ("LONG", "SHORT"):
                    signals_generated += 1

                    trade_data = {
                        "symbol": symbol,
                        "timeframe": tf,
                        "type": signal_type,
                        "entry": entry,
                        "tp": tp,
                        "sl": sl,
                        "setup": reason,
                        "open_time_utc": candle_time,
                        "config_snapshot": default_config,
                    }

                    opened = tm.open_trade(trade_data)
                    if not opened and use_correlation_management:
                        # Check if blocked by correlation
                        blocked_by_correlation += 1

    # Get results
    history = tm.history
    stats = tm.get_correlation_stats()

    return {
        "total_trades": len(history),
        "total_pnl": sum(t.get("pnl", 0) for t in history),
        "wins": sum(1 for t in history if t.get("pnl", 0) > 0),
        "losses": sum(1 for t in history if t.get("pnl", 0) < 0),
        "signals_generated": signals_generated,
        "blocked_by_correlation": blocked_by_correlation,
        "correlation_reduced_trades": stats.get("correlation_reduced_trades", 0),
        "correlation_reduced_pnl": stats.get("correlation_reduced_pnl", 0),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Test Correlation Management Impact")
    parser.add_argument("--candles", type=int, default=15000, help="Number of candles")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "LINKUSDT"],
                       help="Symbols to test")
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h"],
                       help="Timeframes to test")
    args = parser.parse_args()

    print("=" * 70)
    print("CORRELATION MANAGEMENT IMPACT TEST")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Symbols: {args.symbols}")
    print(f"  Timeframes: {args.timeframes}")
    print(f"  Candles: {args.candles}")

    # Run WITH correlation management
    print("\n" + "-" * 70)
    print("TEST 1: WITH Correlation Management (Max 2 per direction, 50% size reduction)")
    print("-" * 70)

    results_with = run_backtest(
        args.symbols, args.timeframes, args.candles,
        use_correlation_management=True
    )

    print(f"\nResults WITH correlation management:")
    print(f"  Signals Generated: {results_with['signals_generated']}")
    print(f"  Trades Opened: {results_with['total_trades']}")
    print(f"  Total PnL: ${results_with['total_pnl']:.2f}")
    print(f"  Wins: {results_with['wins']}, Losses: {results_with['losses']}")
    win_rate = results_with['wins'] / max(results_with['total_trades'], 1) * 100
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Correlation-reduced trades: {results_with['correlation_reduced_trades']}")
    print(f"  PnL from reduced trades: ${results_with['correlation_reduced_pnl']:.2f}")

    # Run WITHOUT correlation management
    print("\n" + "-" * 70)
    print("TEST 2: WITHOUT Correlation Management (No limits)")
    print("-" * 70)

    results_without = run_backtest(
        args.symbols, args.timeframes, args.candles,
        use_correlation_management=False
    )

    print(f"\nResults WITHOUT correlation management:")
    print(f"  Signals Generated: {results_without['signals_generated']}")
    print(f"  Trades Opened: {results_without['total_trades']}")
    print(f"  Total PnL: ${results_without['total_pnl']:.2f}")
    print(f"  Wins: {results_without['wins']}, Losses: {results_without['losses']}")
    win_rate = results_without['wins'] / max(results_without['total_trades'], 1) * 100
    print(f"  Win Rate: {win_rate:.1f}%")

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    trade_diff = results_with['total_trades'] - results_without['total_trades']
    pnl_diff = results_with['total_pnl'] - results_without['total_pnl']

    print(f"\nTrade Count Difference: {trade_diff:+d}")
    print(f"PnL Difference: ${pnl_diff:+.2f}")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    if trade_diff < 0:
        print(f"\n  Correlation management BLOCKED {abs(trade_diff)} trades")
        print(f"  (Max 2 positions per direction limit)")

    if results_with['correlation_reduced_trades'] > 0:
        print(f"\n  {results_with['correlation_reduced_trades']} trades had REDUCED position size (50%)")
        print(f"  PnL from these reduced-size trades: ${results_with['correlation_reduced_pnl']:.2f}")

    if pnl_diff > 0:
        print(f"\n  [POSITIVE IMPACT] Correlation management IMPROVED PnL by ${pnl_diff:.2f}")
    elif pnl_diff < 0:
        print(f"\n  [NEGATIVE IMPACT] Correlation management REDUCED PnL by ${abs(pnl_diff):.2f}")
        print(f"  Note: This may be acceptable if it reduced risk concentration")
    else:
        print(f"\n  [NEUTRAL] No significant PnL difference")

    # Risk analysis - check trade distribution by symbol/direction
    print("\n" + "-" * 70)
    print("TRADE DISTRIBUTION BY SYMBOL & DIRECTION")
    print("-" * 70)

    def analyze_trades(history, label):
        by_symbol = {}
        by_direction = {"LONG": 0, "SHORT": 0}
        for t in history:
            sym = t.get("symbol", "?")
            direction = t.get("type", "?")
            by_symbol[sym] = by_symbol.get(sym, 0) + 1
            if direction in by_direction:
                by_direction[direction] += 1
        print(f"\n  {label}:")
        print(f"    By Symbol: {by_symbol}")
        print(f"    By Direction: {by_direction}")

    analyze_trades(results_with['history'], "WITH correlation management")
    analyze_trades(results_without['history'], "WITHOUT correlation management")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if results_with['total_trades'] == results_without['total_trades']:
        print("\n  No trades were blocked by correlation management.")
        print("  This means signals were naturally spread out over time,")
        print("  or different streams didn't generate overlapping signals.")
    else:
        blocked = abs(trade_diff)
        print(f"\n  Correlation management blocked {blocked} potentially risky trades.")
        if pnl_diff >= 0:
            print(f"  This IMPROVED overall performance by ${pnl_diff:.2f}")
        else:
            avg_blocked_pnl = abs(pnl_diff) / max(blocked, 1)
            print(f"  The blocked trades would have averaged ${avg_blocked_pnl:.2f} each")
            print(f"  Risk was reduced, but some profitable trades were also blocked.")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
