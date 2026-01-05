#!/usr/bin/env python3
"""
Backtest Comparison: SSL Flow (Baseline) vs PBEMA Retest vs Combined

This script compares:
1. SSL Flow only (baseline)
2. PBEMA Retest only (Pattern 2)
3. Combined (both strategies)

Metrics calculated:
- Trade count
- Win rate
- Total PnL (ideal and cost-aware)
- Average R-multiple
- Max drawdown
- Sharpe ratio (if enough trades)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core import get_client, calculate_indicators, set_backtest_mode, TRADING_CONFIG
from strategies import check_ssl_flow_signal, check_pbema_retest_signal


class SimpleBacktester:
    """Simple backtester for strategy comparison."""

    def __init__(self, initial_balance=2000.0):
        self.initial_balance = initial_balance
        self.slippage = TRADING_CONFIG.get("slippage_rate", 0.0005)
        self.fee = TRADING_CONFIG.get("total_fee", 0.0007)
        self.trades = []

    def simulate_trade(self, entry, tp, sl, signal_type, entry_time, df_slice):
        """Simulate a single trade's outcome."""
        # Calculate position size (fixed $100 for comparison)
        position_size = 100.0

        # Risk and reward
        if signal_type == "LONG":
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp

        rr = reward / risk if risk > 0 else 0

        # Simulate outcome by checking future candles
        hit_tp = False
        hit_sl = False
        exit_price = None
        exit_time = None
        candles_held = 0

        for i in range(len(df_slice)):
            candle = df_slice.iloc[i]
            candles_held = i + 1

            if signal_type == "LONG":
                # Check SL first (more conservative)
                if candle['low'] <= sl:
                    hit_sl = True
                    exit_price = sl
                    exit_time = candle.name
                    break
                # Then check TP
                if candle['high'] >= tp:
                    hit_tp = True
                    exit_price = tp
                    exit_time = candle.name
                    break
            else:  # SHORT
                # Check SL first
                if candle['high'] >= sl:
                    hit_sl = True
                    exit_price = sl
                    exit_time = candle.name
                    break
                # Then check TP
                if candle['low'] <= tp:
                    hit_tp = True
                    exit_price = tp
                    exit_time = candle.name
                    break

        # If neither hit, exit at last candle (timeout)
        if not hit_tp and not hit_sl and len(df_slice) > 0:
            exit_price = df_slice.iloc[-1]['close']
            exit_time = df_slice.iloc[-1].name
            candles_held = len(df_slice)

        # Calculate PnL
        if exit_price is None:
            return None

        if signal_type == "LONG":
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry

        # Ideal PnL (no costs)
        ideal_pnl = position_size * pnl_pct

        # Cost-aware PnL (slippage + fees)
        total_cost_pct = self.slippage + self.fee
        cost_aware_pnl = ideal_pnl - (position_size * total_cost_pct)

        # R-multiple
        actual_gain = exit_price - entry if signal_type == "LONG" else entry - exit_price
        r_multiple = actual_gain / risk if risk > 0 else 0

        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'signal_type': signal_type,
            'entry': entry,
            'tp': tp,
            'sl': sl,
            'exit_price': exit_price,
            'hit_tp': hit_tp,
            'hit_sl': hit_sl,
            'candles_held': candles_held,
            'ideal_pnl': ideal_pnl,
            'cost_aware_pnl': cost_aware_pnl,
            'r_multiple': r_multiple,
            'rr': rr,
        }

        self.trades.append(trade)
        return trade

    def get_metrics(self):
        """Calculate performance metrics."""
        if not self.trades:
            return {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'ideal_pnl': 0,
                'cost_aware_pnl': 0,
                'avg_r': 0,
                'max_dd': 0,
            }

        wins = sum(1 for t in self.trades if t['cost_aware_pnl'] > 0)
        losses = sum(1 for t in self.trades if t['cost_aware_pnl'] <= 0)

        ideal_pnl = sum(t['ideal_pnl'] for t in self.trades)
        cost_aware_pnl = sum(t['cost_aware_pnl'] for t in self.trades)

        avg_r = np.mean([t['r_multiple'] for t in self.trades])

        # Max drawdown
        cumulative_pnl = np.cumsum([t['cost_aware_pnl'] for t in self.trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

        return {
            'trades': len(self.trades),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(self.trades) * 100 if self.trades else 0,
            'ideal_pnl': ideal_pnl,
            'cost_aware_pnl': cost_aware_pnl,
            'avg_r': avg_r,
            'max_dd': max_dd,
        }


def run_backtest(df, strategy_name, strategy_func, min_bars=5):
    """Run backtest for a single strategy."""
    print(f"\n{'='*70}")
    print(f"BACKTESTING: {strategy_name}")
    print(f"{'='*70}\n")

    backtester = SimpleBacktester()

    signals_checked = 0
    signals_found = 0

    last_trade_idx = -100  # Last trade index (for spacing)

    for i in range(100, len(df) - 20):  # Leave room at end for trade exits
        signals_checked += 1

        # Require spacing between trades
        if i - last_trade_idx < min_bars:
            continue

        # Get signal
        if strategy_name == "SSL Flow":
            # Use minimal config for SSL Flow (baseline)
            result = strategy_func(df, index=i, return_debug=False)
        else:
            # PBEMA Retest
            result = strategy_func(df, index=i, return_debug=False)

        signal_type, entry, tp, sl, reason = result[:5]

        if signal_type is None:
            continue

        signals_found += 1
        entry_time = df.index[i]

        # Simulate trade on future candles
        future_slice = df.iloc[i+1:i+100]  # Max 100 candles to hit TP/SL

        trade = backtester.simulate_trade(
            entry, tp, sl, signal_type, entry_time, future_slice
        )

        if trade:
            last_trade_idx = i

            # Print trade info
            outcome = "✅ WIN" if trade['cost_aware_pnl'] > 0 else "❌ LOSS"
            print(f"[{entry_time}] {signal_type} {outcome}")
            print(f"  Entry: {entry:.2f} | TP: {tp:.2f} | SL: {sl:.2f}")
            print(f"  Exit: {trade['exit_price']:.2f} | R: {trade['r_multiple']:.2f}")
            print(f"  PnL: ${trade['cost_aware_pnl']:.2f} | Held: {trade['candles_held']} candles")
            print()

    metrics = backtester.get_metrics()

    print(f"\n{'='*70}")
    print(f"RESULTS: {strategy_name}")
    print(f"{'='*70}")
    print(f"Signals Checked: {signals_checked:,}")
    print(f"Signals Found:   {signals_found:,}")
    print(f"Trades Executed: {metrics['trades']}")
    print(f"Wins:            {metrics['wins']}")
    print(f"Losses:          {metrics['losses']}")
    print(f"Win Rate:        {metrics['win_rate']:.1f}%")
    print(f"Ideal PnL:       ${metrics['ideal_pnl']:.2f}")
    print(f"Cost-Aware PnL:  ${metrics['cost_aware_pnl']:.2f}")
    print(f"Avg R-multiple:  {metrics['avg_r']:.3f}")
    print(f"Max Drawdown:    ${metrics['max_dd']:.2f}")
    print(f"{'='*70}\n")

    return backtester, metrics


def run_combined_backtest(df, min_bars=5):
    """Run backtest using both strategies (try SSL first, then PBEMA)."""
    print(f"\n{'='*70}")
    print(f"BACKTESTING: Combined (SSL Flow + PBEMA Retest)")
    print(f"{'='*70}\n")

    backtester = SimpleBacktester()

    signals_checked = 0
    ssl_signals = 0
    pbema_signals = 0

    last_trade_idx = -100

    for i in range(100, len(df) - 20):
        signals_checked += 1

        if i - last_trade_idx < min_bars:
            continue

        # Try SSL Flow first
        ssl_result = check_ssl_flow_signal(df, index=i, return_debug=False)
        signal_type, entry, tp, sl, reason = ssl_result[:5]
        strategy_used = None

        if signal_type is not None:
            ssl_signals += 1
            strategy_used = "SSL"
        else:
            # Try PBEMA Retest
            pbema_result = check_pbema_retest_signal(df, index=i, return_debug=False)
            signal_type, entry, tp, sl, reason = pbema_result[:5]

            if signal_type is not None:
                pbema_signals += 1
                strategy_used = "PBEMA"

        if signal_type is None:
            continue

        entry_time = df.index[i]
        future_slice = df.iloc[i+1:i+100]

        trade = backtester.simulate_trade(
            entry, tp, sl, signal_type, entry_time, future_slice
        )

        if trade:
            last_trade_idx = i

            outcome = "✅ WIN" if trade['cost_aware_pnl'] > 0 else "❌ LOSS"
            print(f"[{entry_time}] {signal_type} {outcome} [{strategy_used}]")
            print(f"  Entry: {entry:.2f} | TP: {tp:.2f} | SL: {sl:.2f}")
            print(f"  Exit: {trade['exit_price']:.2f} | R: {trade['r_multiple']:.2f}")
            print(f"  PnL: ${trade['cost_aware_pnl']:.2f}")
            print()

    metrics = backtester.get_metrics()

    print(f"\n{'='*70}")
    print(f"RESULTS: Combined Strategy")
    print(f"{'='*70}")
    print(f"Signals Checked: {signals_checked:,}")
    print(f"SSL Signals:     {ssl_signals}")
    print(f"PBEMA Signals:   {pbema_signals}")
    print(f"Total Trades:    {metrics['trades']}")
    print(f"Wins:            {metrics['wins']}")
    print(f"Losses:          {metrics['losses']}")
    print(f"Win Rate:        {metrics['win_rate']:.1f}%")
    print(f"Ideal PnL:       ${metrics['ideal_pnl']:.2f}")
    print(f"Cost-Aware PnL:  ${metrics['cost_aware_pnl']:.2f}")
    print(f"Avg R-multiple:  {metrics['avg_r']:.3f}")
    print(f"Max Drawdown:    ${metrics['max_dd']:.2f}")
    print(f"{'='*70}\n")

    return backtester, metrics


def main():
    """Main backtest comparison."""
    print("="*70)
    print("PATTERN 2 BACKTEST COMPARISON")
    print("="*70)
    print()

    # Setup
    set_backtest_mode(True)
    client = get_client()

    # Fetch data
    symbol = "BTCUSDT"
    timeframe = "15m"
    total_candles = 5000  # ~50 days of 15m data

    print(f"Fetching {symbol} {timeframe} data ({total_candles} candles)...")
    print("Using paginated fetch (Binance API limit is 1000 per request)...")
    df = client.get_klines_paginated(symbol=symbol, interval=timeframe, total_candles=total_candles)

    if df is None or df.empty:
        print("❌ Failed to fetch data. DataFrame is empty.")
        print("Please check your API connection and try again.")
        return

    # Set timestamp as index
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)

    print(f"✅ Fetched {len(df)} candles")
    if len(df) > 0:
        print(f"   Period: {df.index[0]} to {df.index[-1]}")
    print()

    print("Calculating indicators...")
    df = calculate_indicators(df, timeframe=timeframe)
    print(f"✅ Indicators ready")
    print()

    # Run backtests
    ssl_bt, ssl_metrics = run_backtest(df, "SSL Flow", check_ssl_flow_signal)
    pbema_bt, pbema_metrics = run_backtest(df, "PBEMA Retest", check_pbema_retest_signal)
    combined_bt, combined_metrics = run_combined_backtest(df)

    # Comparison table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print()
    print(f"{'Metric':<25} {'SSL Flow':<15} {'PBEMA Retest':<15} {'Combined':<15}")
    print("-"*70)
    print(f"{'Trades':<25} {ssl_metrics['trades']:<15} {pbema_metrics['trades']:<15} {combined_metrics['trades']:<15}")
    print(f"{'Win Rate':<25} {ssl_metrics['win_rate']:<15.1f} {pbema_metrics['win_rate']:<15.1f} {combined_metrics['win_rate']:<15.1f}")
    print(f"{'Ideal PnL':<25} ${ssl_metrics['ideal_pnl']:<14.2f} ${pbema_metrics['ideal_pnl']:<14.2f} ${combined_metrics['ideal_pnl']:<14.2f}")
    print(f"{'Cost-Aware PnL':<25} ${ssl_metrics['cost_aware_pnl']:<14.2f} ${pbema_metrics['cost_aware_pnl']:<14.2f} ${combined_metrics['cost_aware_pnl']:<14.2f}")
    print(f"{'Avg R-multiple':<25} {ssl_metrics['avg_r']:<15.3f} {pbema_metrics['avg_r']:<15.3f} {combined_metrics['avg_r']:<15.3f}")
    print(f"{'Max Drawdown':<25} ${ssl_metrics['max_dd']:<14.2f} ${pbema_metrics['max_dd']:<14.2f} ${combined_metrics['max_dd']:<14.2f}")
    print("="*70)
    print()

    # Verdict
    print("VERDICT:")
    print("-"*70)

    best_pnl = max(ssl_metrics['cost_aware_pnl'], pbema_metrics['cost_aware_pnl'], combined_metrics['cost_aware_pnl'])

    if combined_metrics['cost_aware_pnl'] == best_pnl:
        print("🏆 WINNER: Combined Strategy")
        print(f"   PnL improvement over SSL Flow: ${combined_metrics['cost_aware_pnl'] - ssl_metrics['cost_aware_pnl']:.2f}")
        print(f"   Trade count increase: +{combined_metrics['trades'] - ssl_metrics['trades']} trades")
    elif pbema_metrics['cost_aware_pnl'] == best_pnl:
        print("🏆 WINNER: PBEMA Retest Only")
        print(f"   PnL improvement over SSL Flow: ${pbema_metrics['cost_aware_pnl'] - ssl_metrics['cost_aware_pnl']:.2f}")
    else:
        print("🏆 WINNER: SSL Flow (Baseline)")
        print("   Pattern 2 did not improve performance on this dataset")

    print("="*70)


if __name__ == "__main__":
    main()
