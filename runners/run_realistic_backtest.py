#!/usr/bin/env python3
"""
REALISTIC BACKTEST with SimTradeManager

Filter Combo sonuçlarını gerçek bot altyapısıyla test eder:
- SimTradeManager ile balance tracking
- Portfolio risk limitleri
- Commission ve slippage
- Circuit breaker
- Cooldowns

Kullanım:
    python runners/run_realistic_backtest.py --symbol BTCUSDT --timeframe 15m \
        --filters "regime,at_flat_filter,adx_filter"
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    calculate_indicators, get_client, set_backtest_mode,
    SimTradeManager, TRADING_CONFIG,
)
from core.at_scenario_analyzer import check_core_signal
from runners.run_filter_combo_test import apply_filters, log_combo_result

# Enable backtest mode
set_backtest_mode(True)


def fetch_data(symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
    """Fetch data with indicators."""
    client = get_client()

    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
    minutes = tf_minutes.get(timeframe, 15)
    candles = (days * 24 * 60) // minutes + 500

    print(f"Fetching {symbol} {timeframe} ({days} days)...")

    all_dfs = []
    remaining = candles
    end_time = None

    while remaining > 0:
        chunk = min(remaining, 1000)

        if end_time:
            import requests
            url = f"{client.BASE_URL}/klines"
            params = {'symbol': symbol, 'interval': timeframe, 'limit': chunk, 'endTime': end_time}
            res = requests.get(url, params=params, timeout=30)
            if res.status_code != 200 or not res.json():
                break
            df_c = pd.DataFrame(res.json()).iloc[:, :6]
            df_c.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_c['timestamp'] = pd.to_datetime(df_c['timestamp'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
        else:
            df_c = client.get_klines(symbol=symbol, interval=timeframe, limit=chunk)

        if df_c.empty:
            break

        all_dfs.insert(0, df_c)
        remaining -= len(df_c)

        if 'timestamp' in df_c.columns:
            end_time = int(df_c['timestamp'].iloc[0].timestamp() * 1000) - 1
        else:
            break

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    df = calculate_indicators(df, timeframe=timeframe)

    print(f"Got {len(df)} candles")
    return df


def run_realistic_backtest(
    symbol: str,
    timeframe: str,
    filter_list: list,
    days: int = 365,
    initial_balance: float = None,
    verbose: bool = True,
    fixed_position: float = None,  # If set, use fixed position size instead of risk-based
) -> dict:
    """
    Run realistic backtest using SimTradeManager.

    Uses the SAME trade management system as the live bot:
    - Balance tracking with margin
    - Risk-based position sizing
    - Commission and slippage
    - Cooldowns between trades

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        filter_list: List of filter names
        days: Days of data
        initial_balance: Starting balance (default from TRADING_CONFIG)
        verbose: Print progress

    Returns:
        Dict with realistic backtest results
    """

    def log(msg: str):
        if verbose:
            print(msg)

    # Build filter flags
    filter_flags = {
        "use_regime_filter": "regime" in filter_list,
        "use_at_binary": "at_binary" in filter_list,
        "use_at_flat_filter": "at_flat_filter" in filter_list,
        "use_adx_filter": "adx_filter" in filter_list,
        "use_ssl_touch": "ssl_touch" in filter_list,
        "use_rsi_filter": "rsi_filter" in filter_list,
        "use_pbema_distance": "pbema_distance" in filter_list,
        "use_overlap_check": "overlap_check" in filter_list,
        "use_body_position": "body_position" in filter_list,
        "use_wick_rejection": "wick_rejection" in filter_list,
    }

    # Generate combo name
    active_filters = [f for f in filter_list if f != "regime"]
    combo_name = "REGIME + " + " + ".join(active_filters) if active_filters else "REGIME only"

    log(f"\n{'='*70}")
    log(f"REALISTIC BACKTEST with SimTradeManager")
    log(f"{'='*70}")
    log(f"Symbol: {symbol} | TF: {timeframe}")
    log(f"Config: {combo_name}")
    log(f"Initial Balance: ${initial_balance or TRADING_CONFIG['initial_balance']:.2f}")
    log(f"Risk per Trade: {TRADING_CONFIG.get('risk_per_trade_pct', 0.0175)*100:.2f}%")
    log(f"Slippage: {TRADING_CONFIG.get('slippage_rate', 0.001)*100:.2f}%")
    log(f"{'='*70}\n")

    # Fetch data
    df = fetch_data(symbol, timeframe, days)
    if df.empty:
        return {"error": "No data"}

    # Initialize SimTradeManager
    tm = SimTradeManager(initial_balance=initial_balance)

    # Track signals and trades
    signals_found = 0
    signals_filtered = 0
    trades_opened = 0
    trades_rejected = 0
    rejection_reasons = {}

    last_signal_idx = -5

    for i in range(60, len(df) - 10):
        curr = df.iloc[i]
        curr_time = df.index[i]
        high = float(curr["high"])
        low = float(curr["low"])
        close = float(curr["close"])

        # Get PBEMA values for dynamic TP
        pb_top = float(curr.get("pb_ema_top", np.nan)) if "pb_ema_top" in curr else None
        pb_bot = float(curr.get("pb_ema_bot", np.nan)) if "pb_ema_bot" in curr else None

        # ==========================================
        # STEP 1: Update any open trades with current candle
        # ==========================================
        # update_trades handles everything: TP/SL detection, closing, history append
        closed_trades = tm.update_trades(
            symbol=symbol,
            tf=timeframe,
            candle_high=high,
            candle_low=low,
            candle_close=close,
            candle_time_utc=curr_time,
            pb_top=pb_top,
            pb_bot=pb_bot,
        )

        # ==========================================
        # STEP 2: Look for new signals (only if no open trade for this symbol/tf)
        # ==========================================
        has_open = any(t.get("symbol") == symbol and t.get("timeframe") == timeframe
                       for t in tm.open_trades)
        if has_open:
            continue

        if i - last_signal_idx < 5:
            continue

        # Get core signal
        signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)

        if signal_type is None:
            continue

        signals_found += 1
        last_signal_idx = i

        # Apply filters (pass entry/sl for min_sl_filter)
        passed, filter_reason = apply_filters(df, i, signal_type, entry_price=entry, sl_price=sl, **filter_flags)

        if not passed:
            signals_filtered += 1
            rejection_reasons[filter_reason] = rejection_reasons.get(filter_reason, 0) + 1
            continue

        # ==========================================
        # STEP 3: Open trade through SimTradeManager
        # ==========================================
        # Use NEXT candle open for realistic entry
        if i + 1 >= len(df):
            continue

        next_candle = df.iloc[i + 1]
        realistic_entry = float(next_candle["open"])

        # Adjust TP/SL proportionally
        original_sl_dist = abs(entry - sl) / entry
        original_tp_dist = abs(tp - entry) / entry

        if signal_type == "LONG":
            realistic_sl = realistic_entry * (1 - original_sl_dist)
            realistic_tp = realistic_entry * (1 + original_tp_dist)
        else:
            realistic_sl = realistic_entry * (1 + original_sl_dist)
            realistic_tp = realistic_entry * (1 - original_tp_dist)

        trade_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "type": signal_type,
            "entry": realistic_entry,
            "tp": realistic_tp,
            "sl": realistic_sl,
            "open_time_utc": df.index[i + 1],
            "signal_reason": reason,
        }

        success = tm.open_trade(trade_data)

        if success:
            trades_opened += 1
        else:
            trades_rejected += 1
            rejection_reasons["TradeManager Rejected"] = rejection_reasons.get("TradeManager Rejected", 0) + 1

    # ==========================================
    # Close any remaining open trades at last price
    # ==========================================
    last_candle = df.iloc[-1]
    last_time = df.index[-1]
    # Final update will close trades if they hit TP/SL
    tm.update_trades(
        symbol=symbol,
        tf=timeframe,
        candle_high=float(last_candle["high"]),
        candle_low=float(last_candle["low"]),
        candle_close=float(last_candle["close"]),
        candle_time_utc=last_time,
    )

    # ==========================================
    # Compile results
    # ==========================================
    final_balance = tm.wallet_balance
    total_pnl = final_balance - (initial_balance or TRADING_CONFIG["initial_balance"])

    # Calculate stats from history
    history = tm.history
    total_trades = len(history)
    wins = sum(1 for t in history if t.get("pnl", 0) > 0)
    win_rate = wins / total_trades * 100 if total_trades else 0

    # Drawdown from history
    equity = [initial_balance or TRADING_CONFIG["initial_balance"]]
    for t in history:
        equity.append(equity[-1] + t.get("pnl", 0))
    peak, max_dd = equity[0], 0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    result = {
        "combo_name": combo_name,
        "filter_flags": filter_flags,
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "initial_balance": initial_balance or TRADING_CONFIG["initial_balance"],
        "final_balance": final_balance,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl / (initial_balance or TRADING_CONFIG["initial_balance"]) * 100,
        "total_trades": total_trades,
        "wins": wins,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd / (initial_balance or TRADING_CONFIG["initial_balance"]) * 100,
        "signals_found": signals_found,
        "signals_filtered": signals_filtered,
        "trades_rejected_by_tm": trades_rejected,
        "rejection_reasons": rejection_reasons,
        "history": history,
    }

    # Log result
    log_combo_result(symbol, timeframe, days, f"{combo_name} [REALISTIC-SIM]",
                     filter_flags, {
                         "trades": total_trades,
                         "wins": wins,
                         "wr": win_rate,
                         "pnl": total_pnl,
                         "dd": max_dd,
                     }, "realistic_sim")

    # Print summary
    log(f"\n{'='*70}")
    log(f"REALISTIC BACKTEST RESULTS: {combo_name}")
    log(f"{'='*70}")
    log(f"Initial Balance: ${result['initial_balance']:.2f}")
    log(f"Final Balance:   ${result['final_balance']:.2f}")
    log(f"Total PnL:       ${result['total_pnl']:.2f} ({result['total_pnl_pct']:.1f}%)")
    log(f"{'='*70}")
    log(f"Signals Found:    {signals_found}")
    log(f"Signals Filtered: {signals_filtered}")
    log(f"Trades Opened:    {total_trades}")
    log(f"TM Rejections:    {trades_rejected}")
    log(f"{'='*70}")
    log(f"Wins: {wins} ({win_rate:.1f}%)")
    log(f"Max Drawdown: ${max_dd:.2f} ({result['max_drawdown_pct']:.1f}%)")
    log(f"{'='*70}")

    if rejection_reasons:
        log("\nRejection Reasons:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            log(f"  {reason}: {count}")

    return result


def simulate_trade_with_costs(
    df, signal_idx, signal_type, entry, tp, sl,
    position_size=35.0,
    slippage_rate=0.0005,  # 0.05% slippage
    fee_rate=0.0007,       # 0.07% total fee (maker+taker)
):
    """
    Simulate trade with realistic costs (slippage + commission).
    Same logic as ideal simulate_trade but with costs.
    """
    abs_idx = signal_idx if signal_idx >= 0 else (len(df) + signal_idx)

    # Apply slippage to entry (worse price)
    if signal_type == "LONG":
        realistic_entry = entry * (1 + slippage_rate)
    else:
        realistic_entry = entry * (1 - slippage_rate)

    for i in range(abs_idx + 1, len(df)):
        candle = df.iloc[i]
        high, low = float(candle["high"]), float(candle["low"])

        if signal_type == "LONG":
            if low <= sl:
                # SL hit - apply slippage (exit at worse price)
                exit_price = sl * (1 - slippage_rate)
                gross_pnl = (exit_price - realistic_entry) / realistic_entry * position_size
                fee_cost = position_size * fee_rate * 2  # Entry + exit
                net_pnl = gross_pnl - fee_cost
                return {"pnl": net_pnl, "win": False, "exit_idx": i, "gross_pnl": gross_pnl, "fees": fee_cost}
            if high >= tp:
                exit_price = tp * (1 - slippage_rate)
                gross_pnl = (exit_price - realistic_entry) / realistic_entry * position_size
                fee_cost = position_size * fee_rate * 2
                net_pnl = gross_pnl - fee_cost
                return {"pnl": net_pnl, "win": net_pnl > 0, "exit_idx": i, "gross_pnl": gross_pnl, "fees": fee_cost}
        else:  # SHORT
            if high >= sl:
                exit_price = sl * (1 + slippage_rate)
                gross_pnl = (realistic_entry - exit_price) / realistic_entry * position_size
                fee_cost = position_size * fee_rate * 2
                net_pnl = gross_pnl - fee_cost
                return {"pnl": net_pnl, "win": False, "exit_idx": i, "gross_pnl": gross_pnl, "fees": fee_cost}
            if low <= tp:
                exit_price = tp * (1 + slippage_rate)
                gross_pnl = (realistic_entry - exit_price) / realistic_entry * position_size
                fee_cost = position_size * fee_rate * 2
                net_pnl = gross_pnl - fee_cost
                return {"pnl": net_pnl, "win": net_pnl > 0, "exit_idx": i, "gross_pnl": gross_pnl, "fees": fee_cost}

    # EOD - close at last price with costs
    last = float(df.iloc[-1]["close"])
    if signal_type == "LONG":
        exit_price = last * (1 - slippage_rate)
        gross_pnl = (exit_price - realistic_entry) / realistic_entry * position_size
    else:
        exit_price = last * (1 + slippage_rate)
        gross_pnl = (realistic_entry - exit_price) / realistic_entry * position_size

    fee_cost = position_size * fee_rate * 2
    net_pnl = gross_pnl - fee_cost
    return {"pnl": net_pnl, "win": net_pnl > 0, "exit_idx": len(df) - 1, "gross_pnl": gross_pnl, "fees": fee_cost}


def run_cost_aware_backtest(
    symbol: str,
    timeframe: str,
    filter_list: list,
    days: int = 365,
    position_size: float = 35.0,
    slippage_rate: float = 0.0005,
    fee_rate: float = 0.0007,
    verbose: bool = True,
) -> dict:
    """
    Run backtest with fixed position size + realistic costs.

    This is the apple-to-apple comparison with ideal test:
    - Same fixed $35 position
    - Same simulate_trade logic
    - BUT with slippage and commission

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        filter_list: List of filter names
        days: Days of data
        position_size: Fixed position size (default $35)
        slippage_rate: Slippage per trade (default 0.05%)
        fee_rate: Total fee per trade (default 0.07%)
        verbose: Print progress
    """
    def log(msg: str):
        if verbose:
            print(msg)

    # Build filter flags
    filter_flags = {
        "use_regime_filter": "regime" in filter_list,
        "use_at_binary": "at_binary" in filter_list,
        "use_at_flat_filter": "at_flat_filter" in filter_list,
        "use_adx_filter": "adx_filter" in filter_list,
        "use_ssl_touch": "ssl_touch" in filter_list,
        "use_rsi_filter": "rsi_filter" in filter_list,
        "use_pbema_distance": "pbema_distance" in filter_list,
        "use_overlap_check": "overlap_check" in filter_list,
        "use_body_position": "body_position" in filter_list,
        "use_wick_rejection": "wick_rejection" in filter_list,
    }

    active_filters = [f for f in filter_list if f != "regime"]
    combo_name = "REGIME + " + " + ".join(active_filters) if active_filters else "REGIME only"

    log(f"\n{'='*70}")
    log(f"COST-AWARE BACKTEST (Fixed Position + Costs)")
    log(f"{'='*70}")
    log(f"Symbol: {symbol} | TF: {timeframe}")
    log(f"Config: {combo_name}")
    log(f"Position Size: ${position_size:.2f} (fixed)")
    log(f"Slippage: {slippage_rate*100:.3f}%")
    log(f"Fee Rate: {fee_rate*100:.3f}%")
    log(f"{'='*70}\n")

    # Fetch data
    df = fetch_data(symbol, timeframe, days)
    if df.empty:
        return {"error": "No data"}

    trades = []
    signals_found = 0
    signals_filtered = 0
    last_signal_idx = -5

    for i in range(60, len(df) - 10):
        if i - last_signal_idx < 5:
            continue

        signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)

        if signal_type is None:
            continue

        signals_found += 1
        last_signal_idx = i

        passed, filter_reason = apply_filters(df, i, signal_type, entry_price=entry, sl_price=sl, **filter_flags)

        if not passed:
            signals_filtered += 1
            continue

        # Simulate trade WITH costs
        trade = simulate_trade_with_costs(
            df, i, signal_type, entry, tp, sl,
            position_size=position_size,
            slippage_rate=slippage_rate,
            fee_rate=fee_rate,
        )
        trade["signal_time"] = str(df.index[i])
        trade["signal_type"] = signal_type
        trades.append(trade)

    if not trades:
        return {
            "combo_name": combo_name,
            "trades": 0, "wins": 0, "wr": 0, "pnl": 0, "dd": 0,
            "gross_pnl": 0, "total_fees": 0,
        }

    wins = sum(1 for t in trades if t["win"])
    pnl = sum(t["pnl"] for t in trades)
    gross_pnl = sum(t.get("gross_pnl", 0) for t in trades)
    total_fees = sum(t.get("fees", 0) for t in trades)

    # Drawdown
    equity = [0]
    for t in trades:
        equity.append(equity[-1] + t["pnl"])
    peak, dd = 0, 0
    for e in equity:
        peak = max(peak, e)
        dd = max(dd, peak - e)

    result = {
        "combo_name": combo_name,
        "filter_flags": filter_flags,
        "trades": len(trades),
        "wins": wins,
        "wr": wins / len(trades) * 100,
        "pnl": pnl,
        "dd": dd,
        "gross_pnl": gross_pnl,
        "total_fees": total_fees,
        "signals_found": signals_found,
        "signals_filtered": signals_filtered,
        "position_size": position_size,
        "slippage_rate": slippage_rate,
        "fee_rate": fee_rate,
    }

    log(f"\n{'='*70}")
    log(f"COST-AWARE RESULTS: {combo_name}")
    log(f"{'='*70}")
    log(f"Trades: {len(trades)}")
    log(f"Wins: {wins} ({result['wr']:.1f}%)")
    log(f"Gross PnL: ${gross_pnl:.2f}")
    log(f"Total Fees: ${total_fees:.2f}")
    log(f"Net PnL: ${pnl:.2f}")
    log(f"Max DD: ${dd:.2f}")
    log(f"{'='*70}")

    return result


def compare_ideal_vs_realistic(
    symbol: str,
    timeframe: str,
    filter_list: list,
    days: int = 365,
) -> dict:
    """
    Compare ideal backtest (no costs) vs cost-aware (same position, with costs).
    This is the proper apple-to-apple comparison.
    """
    from runners.run_filter_combo_test import run_specific_combo

    print(f"\n{'='*70}")
    print("COMPARISON: IDEAL vs COST-AWARE")
    print(f"{'='*70}\n")

    # Run ideal test
    print("Running IDEAL backtest (no costs)...")
    ideal = run_specific_combo(symbol, timeframe, days, filter_list, log_result=False)

    # Run cost-aware test (same position, with costs)
    print("\nRunning COST-AWARE backtest (fixed $35 + costs)...")
    cost_aware = run_cost_aware_backtest(symbol, timeframe, filter_list, days, verbose=False)

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS (Apple-to-Apple)")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Ideal':>15} {'With Costs':>15} {'Impact':>15}")
    print("-" * 70)

    ideal_pnl = ideal.get("pnl", 0)
    cost_pnl = cost_aware.get("pnl", 0)
    pnl_diff = cost_pnl - ideal_pnl

    ideal_trades = ideal.get("trades", 0)
    cost_trades = cost_aware.get("trades", 0)

    ideal_wr = ideal.get("wr", 0)
    cost_wr = cost_aware.get("wr", 0)

    ideal_dd = ideal.get("dd", 0)
    cost_dd = cost_aware.get("dd", 0)

    print(f"{'Trades':<25} {ideal_trades:>15} {cost_trades:>15} {cost_trades - ideal_trades:>+15}")
    print(f"{'Win Rate':<25} {ideal_wr:>14.1f}% {cost_wr:>14.1f}% {cost_wr - ideal_wr:>+14.1f}%")
    print(f"{'Gross PnL':<25} ${ideal_pnl:>13.2f} ${cost_aware.get('gross_pnl', 0):>13.2f}")
    print(f"{'Total Fees':<25} {'$0.00':>15} ${cost_aware.get('total_fees', 0):>13.2f}")
    print(f"{'Net PnL':<25} ${ideal_pnl:>13.2f} ${cost_pnl:>13.2f} ${pnl_diff:>+13.2f}")
    print(f"{'Max Drawdown':<25} ${ideal_dd:>13.2f} ${cost_dd:>13.2f} ${cost_dd - ideal_dd:>+13.2f}")

    pnl_reduction = (1 - cost_pnl / ideal_pnl) * 100 if ideal_pnl != 0 else 0
    print("-" * 70)
    print(f"{'Cost Impact':<25} {'':<15} {'':<15} {pnl_reduction:>+13.1f}%")

    print(f"\n{'='*70}")
    if cost_pnl > 0:
        print("✅ COST-AWARE TEST PASSED - Strategy is profitable with real costs")
    else:
        print("❌ COST-AWARE TEST FAILED - Strategy loses money with real costs")
    print(f"{'='*70}")

    return {
        "ideal": ideal,
        "cost_aware": cost_aware,
        "pnl_reduction_pct": pnl_reduction,
    }


def main():
    parser = argparse.ArgumentParser(description="Realistic Backtest with SimTradeManager")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--filters", type=str, required=True,
                        help="Comma-separated filter list, e.g., 'regime,at_flat_filter,adx_filter'")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--balance", type=float, default=None, help="Initial balance (default from config)")
    parser.add_argument("--compare", action="store_true", help="Compare ideal vs cost-aware (apple-to-apple)")
    parser.add_argument("--cost-aware", action="store_true", help="Run cost-aware test (fixed position + costs)")
    parser.add_argument("--sim-trade-manager", action="store_true", help="Run with SimTradeManager (variable position)")
    parser.add_argument("--save-history", action="store_true", help="Save trade history to JSON")

    args = parser.parse_args()

    # Parse filters
    filter_list = [f.strip() for f in args.filters.split(",")]

    if args.compare:
        result = compare_ideal_vs_realistic(args.symbol, args.timeframe, filter_list, args.days)
    elif args.cost_aware:
        result = run_cost_aware_backtest(
            symbol=args.symbol,
            timeframe=args.timeframe,
            filter_list=filter_list,
            days=args.days,
            verbose=True,
        )
    elif args.sim_trade_manager:
        result = run_realistic_backtest(
            symbol=args.symbol,
            timeframe=args.timeframe,
            filter_list=filter_list,
            days=args.days,
            initial_balance=args.balance,
            verbose=True,
        )
    else:
        # Default: run comparison
        result = compare_ideal_vs_realistic(args.symbol, args.timeframe, filter_list, args.days)

    # Save history if requested
    if args.save_history and result.get("history"):
        output_file = f"realistic_trades_{args.symbol}_{args.timeframe}.json"
        with open(output_file, "w") as f:
            json.dump(result["history"], f, indent=2, default=str)
        print(f"\nTrade history saved to: {output_file}")


if __name__ == "__main__":
    main()
