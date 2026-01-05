#!/usr/bin/env python3
"""
Regime Filter Backtest - Priority 2 Validation

This script runs a backtest comparison with and without the BTC regime filter
to validate the hedge fund recommendation implementation.

Expected Results:
- Fewer trades with regime filter (filters out ranging markets)
- Higher win rate (only trades in favorable regimes)
- Better risk-adjusted returns (avoids H1-style losses)

Usage:
    python run_regime_filter_backtest.py              # 6-month test
    python run_regime_filter_backtest.py --full-year  # Full year test
    python run_regime_filter_backtest.py --quick      # Quick 3-month test
"""

import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    calculate_indicators, get_client, DATA_DIR,
    SimTradeManager, TRADING_CONFIG,
)
from core.regime_filter import RegimeFilter, RegimeType, analyze_regime_distribution
from strategies.ssl_flow import check_ssl_flow_signal


def run_single_stream_backtest(
    df,
    symbol: str,
    timeframe: str,
    btc_df=None,
    use_regime_filter: bool = False,
    config: dict = None,
) -> Dict:
    """Run backtest on a single stream with optional regime filter."""

    if config is None:
        config = {
            "min_rr": 2.0,
            "rsi_limit": 70.0,
            "regime_adx_threshold": 20.0,
        }

    trades = []
    signals_checked = 0
    signals_blocked_by_regime = 0

    # Simulate through the data
    for i in range(100, len(df) - 1):  # Start after warmup, leave room for trade execution
        signals_checked += 1

        # Check for signal
        result = check_ssl_flow_signal(
            df,
            index=i,
            min_rr=config.get("min_rr", 2.0),
            rsi_limit=config.get("rsi_limit", 70.0),
            regime_adx_threshold=config.get("regime_adx_threshold", 20.0),
            use_btc_regime_filter=use_regime_filter,
            btc_df=btc_df,
            symbol=symbol,
            regime_min_confidence=0.5,
            return_debug=True,
        )

        signal_type = result[0]
        entry = result[1]
        tp = result[2]
        sl = result[3]
        reason = result[4]

        # Track regime blocks
        if "BTC Not Trending" in reason or "Regime Filter" in reason:
            signals_blocked_by_regime += 1

        if signal_type and entry and tp and sl:
            # Simulate trade execution
            trade_result = simulate_trade(df, i, signal_type, entry, tp, sl, symbol, timeframe)
            if trade_result:
                trades.append(trade_result)

    # Calculate metrics
    total_pnl = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    total_trades = len(trades)
    win_rate = wins / total_trades if total_trades > 0 else 0

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "total_trades": total_trades,
        "wins": wins,
        "losses": total_trades - wins,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "signals_checked": signals_checked,
        "signals_blocked_by_regime": signals_blocked_by_regime,
        "trades": trades,
    }


def simulate_trade(df, entry_idx: int, signal_type: str, entry: float, tp: float, sl: float,
                   symbol: str, timeframe: str) -> Dict:
    """Simulate trade execution and calculate PnL."""

    # Look forward to see if TP or SL was hit
    for j in range(entry_idx + 1, min(entry_idx + 100, len(df))):
        candle = df.iloc[j]
        high = float(candle["high"])
        low = float(candle["low"])

        if signal_type == "LONG":
            # Check SL first (conservative)
            if low <= sl:
                pnl = calculate_pnl(entry, sl, signal_type)
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "type": signal_type,
                    "entry": entry,
                    "exit": sl,
                    "tp": tp,
                    "sl": sl,
                    "pnl": pnl,
                    "result": "SL",
                    "bars_held": j - entry_idx,
                }
            # Check TP
            if high >= tp:
                pnl = calculate_pnl(entry, tp, signal_type)
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "type": signal_type,
                    "entry": entry,
                    "exit": tp,
                    "tp": tp,
                    "sl": sl,
                    "pnl": pnl,
                    "result": "TP",
                    "bars_held": j - entry_idx,
                }
        else:  # SHORT
            # Check SL first
            if high >= sl:
                pnl = calculate_pnl(entry, sl, signal_type)
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "type": signal_type,
                    "entry": entry,
                    "exit": sl,
                    "tp": tp,
                    "sl": sl,
                    "pnl": pnl,
                    "result": "SL",
                    "bars_held": j - entry_idx,
                }
            # Check TP
            if low <= tp:
                pnl = calculate_pnl(entry, tp, signal_type)
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "type": signal_type,
                    "entry": entry,
                    "exit": tp,
                    "tp": tp,
                    "sl": sl,
                    "pnl": pnl,
                    "result": "TP",
                    "bars_held": j - entry_idx,
                }

    # Trade didn't complete within lookforward window
    return None


def calculate_pnl(entry: float, exit: float, signal_type: str) -> float:
    """Calculate PnL for a trade."""
    position_size = 2000 * 0.0175  # $35 risk

    if signal_type == "LONG":
        pct_change = (exit - entry) / entry
    else:
        pct_change = (entry - exit) / entry

    # Apply leverage
    leverage = 10
    pnl = position_size * pct_change * leverage

    return pnl


def run_comparison_backtest(
    symbols: List[str] = None,
    timeframes: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    candles: int = 5000,
) -> Dict:
    """Run backtest comparison with and without regime filter."""

    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "LINKUSDT"]
    if timeframes is None:
        timeframes = ["15m", "1h"]

    print("\n" + "="*70)
    print("REGIME FILTER BACKTEST COMPARISON")
    print("Priority 2: BTC Leader Regime Filter Validation")
    print("="*70)
    print(f"\nSymbols: {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Candles per stream: {candles}")

    client = get_client()

    # First, fetch BTC data (leader)
    print("\n📥 Fetching BTC data (market leader)...")
    btc_data = {}
    for tf in timeframes:
        # Use paginated fetch for > 1000 candles
        if candles > 1000:
            btc_df = client.get_klines_paginated("BTCUSDT", tf, total_candles=candles)
        else:
            btc_df = client.get_klines("BTCUSDT", tf, limit=candles)
        if btc_df is not None and not btc_df.empty:
            btc_df = calculate_indicators(btc_df, tf)
            btc_data[tf] = btc_df
            print(f"   ✓ BTCUSDT-{tf}: {len(btc_df)} bars")

    # Results containers
    results_without_filter = []
    results_with_filter = []

    # Run backtests
    for symbol in symbols:
        for tf in timeframes:
            print(f"\n📊 Testing {symbol}-{tf}...")

            # Fetch data (use paginated for > 1000 candles)
            if candles > 1000:
                df = client.get_klines_paginated(symbol, tf, total_candles=candles)
            else:
                df = client.get_klines(symbol, tf, limit=candles)
            if df is None or df.empty:
                print(f"   ⚠️ Could not fetch data for {symbol}-{tf}")
                continue

            df = calculate_indicators(df, tf)

            # Get BTC data for this timeframe
            btc_df = btc_data.get(tf)

            # Run WITHOUT regime filter
            print(f"   Running without regime filter...")
            result_no_filter = run_single_stream_backtest(
                df, symbol, tf,
                btc_df=None,
                use_regime_filter=False,
            )
            results_without_filter.append(result_no_filter)

            # Run WITH regime filter
            print(f"   Running with regime filter...")
            result_with_filter = run_single_stream_backtest(
                df, symbol, tf,
                btc_df=btc_df,
                use_regime_filter=True,
            )
            results_with_filter.append(result_with_filter)

            # Print comparison for this stream
            print(f"   Without filter: {result_no_filter['total_trades']} trades, "
                  f"${result_no_filter['total_pnl']:.2f} PnL, "
                  f"{result_no_filter['win_rate']*100:.0f}% WR")
            print(f"   With filter:    {result_with_filter['total_trades']} trades, "
                  f"${result_with_filter['total_pnl']:.2f} PnL, "
                  f"{result_with_filter['win_rate']*100:.0f}% WR")
            print(f"   Regime blocks:  {result_with_filter['signals_blocked_by_regime']}")

    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)

    # Without filter
    total_trades_no = sum(r["total_trades"] for r in results_without_filter)
    total_pnl_no = sum(r["total_pnl"] for r in results_without_filter)
    total_wins_no = sum(r["wins"] for r in results_without_filter)
    win_rate_no = total_wins_no / total_trades_no if total_trades_no > 0 else 0

    # With filter
    total_trades_with = sum(r["total_trades"] for r in results_with_filter)
    total_pnl_with = sum(r["total_pnl"] for r in results_with_filter)
    total_wins_with = sum(r["wins"] for r in results_with_filter)
    win_rate_with = total_wins_with / total_trades_with if total_trades_with > 0 else 0
    total_blocked = sum(r["signals_blocked_by_regime"] for r in results_with_filter)

    print(f"\n{'Metric':<25} {'Without Filter':>18} {'With Filter':>18} {'Change':>12}")
    print("-"*70)
    print(f"{'Total Trades':<25} {total_trades_no:>18} {total_trades_with:>18} {total_trades_with - total_trades_no:>+12}")
    print(f"{'Total PnL':<25} ${total_pnl_no:>17.2f} ${total_pnl_with:>17.2f} ${total_pnl_with - total_pnl_no:>+11.2f}")
    print(f"{'Win Rate':<25} {win_rate_no*100:>17.1f}% {win_rate_with*100:>17.1f}% {(win_rate_with - win_rate_no)*100:>+11.1f}%")
    print(f"{'Signals Blocked':<25} {'-':>18} {total_blocked:>18}")

    # Calculate regime impact
    trades_filtered = total_trades_no - total_trades_with
    pnl_diff = total_pnl_with - total_pnl_no

    print("\n" + "="*70)
    print("REGIME FILTER IMPACT ANALYSIS")
    print("="*70)

    if trades_filtered > 0:
        print(f"\n📊 Trades filtered out: {trades_filtered}")
        print(f"💰 PnL difference: ${pnl_diff:+.2f}")

        if pnl_diff > 0:
            print(f"\n✅ POSITIVE IMPACT: Regime filter improved results by ${pnl_diff:.2f}")
            print("   The filtered trades would have been losers on average.")
        elif pnl_diff < 0:
            print(f"\n⚠️ NEGATIVE IMPACT: Regime filter reduced results by ${abs(pnl_diff):.2f}")
            print("   Some filtered trades would have been winners.")
        else:
            print(f"\n➖ NEUTRAL IMPACT: No significant change in PnL")
    else:
        print("\n📊 No trades were filtered by the regime filter.")
        print("   This could mean:")
        print("   - Market was mostly trending during the test period")
        print("   - Existing regime gating already blocks ranging markets")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    if pnl_diff > 0 and win_rate_with >= win_rate_no:
        print("""
✅ ENABLE REGIME FILTER FOR PRODUCTION

The BTC leader regime filter improves results:
- Higher PnL
- Same or better win rate
- Fewer trades in unfavorable conditions

To enable, set in your config:
  use_btc_regime_filter=True
        """)
    elif trades_filtered == 0:
        print("""
➖ REGIME FILTER HAD NO EFFECT

The existing ADX-based regime gating is sufficient for the current
test period. Consider keeping the filter disabled to avoid complexity.
        """)
    else:
        print("""
⚠️ REGIME FILTER REDUCED PERFORMANCE

The filtered trades included some winners. Possible reasons:
- Test period was mostly trending
- BTC and alts had different regime characteristics

Consider:
1. Adjusting regime thresholds (adx_trending_threshold)
2. Testing on a longer period with more regime changes
3. Keeping the filter disabled for now
        """)

    # Save results
    output_dir = os.path.join(DATA_DIR, "regime_filter_backtest")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "symbols": symbols,
            "timeframes": timeframes,
            "candles": candles,
        },
        "without_filter": {
            "total_trades": total_trades_no,
            "total_pnl": total_pnl_no,
            "win_rate": win_rate_no,
            "details": [
                {k: v for k, v in r.items() if k != "trades"}
                for r in results_without_filter
            ],
        },
        "with_filter": {
            "total_trades": total_trades_with,
            "total_pnl": total_pnl_with,
            "win_rate": win_rate_with,
            "signals_blocked": total_blocked,
            "details": [
                {k: v for k, v in r.items() if k != "trades"}
                for r in results_with_filter
            ],
        },
        "impact": {
            "trades_filtered": trades_filtered,
            "pnl_difference": pnl_diff,
            "win_rate_change": win_rate_with - win_rate_no,
        },
    }

    results_path = os.path.join(output_dir, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {results_path}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Regime Filter Backtest Comparison")
    parser.add_argument('--quick', action='store_true', help='Quick test (fewer candles)')
    parser.add_argument('--full-year', action='store_true', help='Full year test')
    parser.add_argument('--symbols', nargs='+', help='Symbols to test')
    parser.add_argument('--timeframes', nargs='+', help='Timeframes to test')

    args = parser.parse_args()

    candles = 2000  # Default
    if args.quick:
        candles = 1000
    elif args.full_year:
        candles = 35000  # ~1 year of 15m candles

    symbols = args.symbols or ["BTCUSDT", "ETHUSDT", "LINKUSDT"]
    timeframes = args.timeframes or ["15m", "1h"]

    results = run_comparison_backtest(
        symbols=symbols,
        timeframes=timeframes,
        candles=candles,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
