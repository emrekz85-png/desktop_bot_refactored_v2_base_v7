#!/usr/bin/env python3
"""
Strategy Sanity Tests

İki kritik test:
1. Test A - Window Equality Test: Rolling WF window PnL ile normal backtest PnL karşılaştırması
2. Test B - Both-Hit Bias Measurement: Aynı candle'da hem TP hem SL vuran trade sayısı

Bu testler, framework hatasını (window slicing, state reset, fee/slippage) strateji
edge yokluğundan ayırmak için kritik.

Kullanım:
    python run_strategy_sanity_tests.py             # Her iki testi çalıştır
    python run_strategy_sanity_tests.py --test-a    # Sadece Window Equality
    python run_strategy_sanity_tests.py --test-b    # Sadece Both-Hit
    python run_strategy_sanity_tests.py --quick     # Hızlı test (az sembol, kısa period)
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from desktop_bot_refactored_v2_base_v7 import (
    run_rolling_walkforward,
    run_portfolio_backtest,
    TradingEngine,
    SimTradeManager,
    SYMBOLS,
    TIMEFRAMES,
    TRADING_CONFIG,
    BASELINE_CONFIG,
    DATA_DIR,
    tf_to_timedelta,
)


def run_test_a_window_equality(
    symbol: str = "BTCUSDT",
    timeframe: str = "15m",
    window_start: str = "2025-10-01",
    window_end: str = "2025-10-31",
    config: Dict = None,
    verbose: bool = True
) -> Dict:
    """
    Test A: Window Equality Test

    Bu test, rolling walk-forward framework'ün bir penceredeki PnL hesaplamasının,
    normal backtest ile aynı olup olmadığını kontrol eder.

    Args:
        symbol: Test edilecek sembol
        timeframe: Test edilecek timeframe
        window_start: Pencere başlangıcı (YYYY-MM-DD)
        window_end: Pencere sonu (YYYY-MM-DD)
        config: Kullanılacak config (None ise BASELINE_CONFIG kullanılır)
        verbose: Detaylı çıktı

    Returns:
        dict with test results
    """
    import heapq

    def log(msg: str):
        if verbose:
            print(msg)

    log("\n" + "="*70)
    log("🧪 TEST A: WINDOW EQUALITY TEST")
    log("="*70)
    log(f"   Symbol: {symbol}")
    log(f"   Timeframe: {timeframe}")
    log(f"   Window: {window_start} → {window_end}")
    log("="*70 + "\n")

    if config is None:
        config = BASELINE_CONFIG.copy()

    # Calculate buffer for indicator warmup
    buffer_days = 30  # For indicator warmup
    fetch_start = datetime.strptime(window_start, "%Y-%m-%d") - timedelta(days=buffer_days)
    fetch_end = datetime.strptime(window_end, "%Y-%m-%d")

    # Fetch data once
    log("📥 Veri çekiliyor...")
    df = TradingEngine.get_historical_data_pagination(
        symbol, timeframe,
        start_date=fetch_start.strftime("%Y-%m-%d"),
        end_date=fetch_end.strftime("%Y-%m-%d")
    )

    if df is None or df.empty or len(df) < 250:
        log("❌ Yeterli veri yok")
        return {"error": "insufficient_data", "passed": False}

    df = TradingEngine.calculate_indicators(df)
    df = df.reset_index(drop=True)
    log(f"   ✓ {len(df)} mum yüklendi")

    # Determine PBEMA columns based on strategy mode
    strategy_mode = config.get("strategy_mode", "keltner_bounce")
    if strategy_mode == "pbema_reaction":
        pb_top_col = "pb_ema_top_150"
        pb_bot_col = "pb_ema_bot_150"
    else:
        pb_top_col = "pb_ema_top"
        pb_bot_col = "pb_ema_bot"

    # Pre-extract arrays
    timestamps = pd.to_datetime(df["timestamp"]).values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    pb_tops = df.get(pb_top_col, df["close"]).values if pb_top_col in df.columns else df["close"].values
    pb_bots = df.get(pb_bot_col, df["close"]).values if pb_bot_col in df.columns else df["close"].values

    # Convert window boundaries
    window_start_ts = pd.Timestamp(window_start)
    window_end_ts = pd.Timestamp(window_end)

    # Find indices within window
    first_ts = pd.Timestamp(timestamps[0])
    if first_ts.tzinfo is not None:
        window_start_ts = window_start_ts.tz_localize('UTC')
        window_end_ts = window_end_ts.tz_localize('UTC')

    # Find start index
    start_idx = 250  # Warmup
    for i, ts in enumerate(timestamps):
        if pd.Timestamp(ts) >= window_start_ts:
            start_idx = max(i, 250)
            break

    log(f"   Start index: {start_idx}, Total candles: {len(df)}")

    # ==========================================
    # METHOD 1: Manual Window Backtest (like run_window_backtest)
    # ==========================================
    log("\n📊 Method 1: Manual Window Backtest (Rolling WF style)...")

    tm1 = SimTradeManager(initial_balance=TRADING_CONFIG["initial_balance"])
    method1_trades = []
    method1_pnl = 0.0

    idx = start_idx
    while idx < len(df) - 2:
        candle_ts = pd.Timestamp(timestamps[idx])
        if candle_ts >= window_end_ts:
            break

        candle_high = float(highs[idx])
        candle_low = float(lows[idx])
        candle_close = float(closes[idx])
        pb_top = float(pb_tops[idx])
        pb_bot = float(pb_bots[idx])
        candle_time = timestamps[idx]

        # Update existing trades
        closed = tm1.update_trades(
            symbol, timeframe, candle_high, candle_low, candle_close,
            candle_time, pb_top, pb_bot
        )

        for trade in closed:
            pnl = float(trade.get("pnl", 0))
            method1_pnl += pnl
            method1_trades.append(trade)

        # Check for new signals
        has_open = any(t.get("symbol") == symbol and t.get("timeframe") == timeframe
                      for t in tm1.open_trades)
        if not has_open and not tm1.check_cooldown(symbol, timeframe, candle_time):
            rr = config.get("rr", 2.0)
            rsi_limit = config.get("rsi", 60)
            at_active = config.get("at_active", False)

            if strategy_mode == "pbema_reaction":
                sig = TradingEngine.check_pbema_reaction_signal(
                    df, idx, min_rr=rr, rsi_limit=rsi_limit,
                    use_alphatrend=at_active
                )
            else:
                sig = TradingEngine.check_signal_diagnostic(
                    df, idx, min_rr=rr, rsi_limit=rsi_limit,
                    slope_thresh=0.0, use_alphatrend=at_active
                )

            if sig and len(sig) >= 5 and sig[0] is not None:
                signal_type, entry, tp, sl, reason = sig[:5]
                trade_data = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "type": signal_type,
                    "entry": entry,
                    "tp": tp,
                    "sl": sl,
                    "open_time_utc": candle_time,
                    "setup": reason or "Unknown",
                    "config_snapshot": config,
                }
                tm1.open_trade(trade_data)

        idx += 1

    log(f"   ✓ Method 1: {len(method1_trades)} trades, PnL = ${method1_pnl:.2f}")

    # ==========================================
    # METHOD 2: run_portfolio_backtest with same date range
    # ==========================================
    log("\n📊 Method 2: run_portfolio_backtest (Standard backtest)...")

    # Save trades to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_trades_csv = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_summary_csv = f.name

    try:
        # Run portfolio backtest for same period
        # IMPORTANT: Force use the same config by temporarily overwriting best_configs.json
        # This ensures we compare apples to apples
        from core import BEST_CONFIGS_FILE
        import shutil

        # Backup existing best_configs.json
        backup_file = None
        if os.path.exists(BEST_CONFIGS_FILE):
            backup_file = BEST_CONFIGS_FILE + ".backup"
            shutil.copy(BEST_CONFIGS_FILE, backup_file)

        # Write our test config
        test_config = {
            symbol: {
                timeframe: {
                    **config,
                    "disabled": False,  # Force enable
                    "_source": "sanity_test",
                }
            }
        }
        with open(BEST_CONFIGS_FILE, 'w') as f:
            json.dump(test_config, f)

        try:
            run_portfolio_backtest(
                symbols=[symbol],
                timeframes=[timeframe],
                start_date=window_start,
                end_date=window_end,
                out_trades_csv=temp_trades_csv,
                out_summary_csv=temp_summary_csv,
                skip_optimization=True,  # Use our forced config
                draw_trades=False,
            )
        finally:
            # Restore original best_configs.json
            if backup_file and os.path.exists(backup_file):
                shutil.move(backup_file, BEST_CONFIGS_FILE)
            elif backup_file is None and os.path.exists(BEST_CONFIGS_FILE):
                os.remove(BEST_CONFIGS_FILE)

        # Read trades
        if os.path.exists(temp_trades_csv):
            trades_df = pd.read_csv(temp_trades_csv)
            method2_trades = trades_df.to_dict('records')
            method2_pnl = trades_df['pnl'].sum() if 'pnl' in trades_df.columns else 0.0
        else:
            method2_trades = []
            method2_pnl = 0.0

        log(f"   ✓ Method 2: {len(method2_trades)} trades, PnL = ${method2_pnl:.2f}")

    finally:
        # Cleanup temp files
        if os.path.exists(temp_trades_csv):
            os.remove(temp_trades_csv)
        if os.path.exists(temp_summary_csv):
            os.remove(temp_summary_csv)

    # ==========================================
    # COMPARE RESULTS
    # ==========================================
    log("\n" + "-"*50)
    log("📊 COMPARISON RESULTS:")
    log("-"*50)

    trade_count_diff = len(method1_trades) - len(method2_trades)
    pnl_diff = method1_pnl - method2_pnl
    pnl_diff_pct = abs(pnl_diff / max(abs(method1_pnl), abs(method2_pnl), 1)) * 100

    log(f"   Method 1 (Window BT): {len(method1_trades)} trades, ${method1_pnl:.2f}")
    log(f"   Method 2 (Portfolio): {len(method2_trades)} trades, ${method2_pnl:.2f}")
    log(f"   Trade count diff: {trade_count_diff}")
    log(f"   PnL diff: ${pnl_diff:.2f} ({pnl_diff_pct:.1f}%)")

    # Determine pass/fail
    # Allow small differences due to float precision
    TOLERANCE_PCT = 1.0  # 1% tolerance
    TOLERANCE_ABS = 5.0  # $5 absolute tolerance

    passed = (
        abs(trade_count_diff) <= 2 and
        (pnl_diff_pct < TOLERANCE_PCT or abs(pnl_diff) < TOLERANCE_ABS)
    )

    if passed:
        log("\n✅ TEST A PASSED: Window equality confirmed")
    else:
        log("\n❌ TEST A FAILED: Significant divergence detected")
        if abs(trade_count_diff) > 2:
            log(f"   ⚠️ Trade count difference too large: {trade_count_diff}")
        if pnl_diff_pct >= TOLERANCE_PCT and abs(pnl_diff) >= TOLERANCE_ABS:
            log(f"   ⚠️ PnL difference too large: ${pnl_diff:.2f} ({pnl_diff_pct:.1f}%)")
        log("\n   Possible causes:")
        log("   - Window slicing logic mismatch")
        log("   - State reset between windows")
        log("   - Fee/slippage calculation difference")
        log("   - Position carry-over handling")

    return {
        "passed": passed,
        "method1_trades": len(method1_trades),
        "method1_pnl": method1_pnl,
        "method2_trades": len(method2_trades),
        "method2_pnl": method2_pnl,
        "trade_count_diff": trade_count_diff,
        "pnl_diff": pnl_diff,
        "pnl_diff_pct": pnl_diff_pct,
    }


def run_test_b_both_hit_measurement(
    symbols: List[str] = None,
    timeframes: List[str] = None,
    start_date: str = "2025-06-01",
    end_date: str = "2025-12-01",
    verbose: bool = True
) -> Dict:
    """
    Test B: Both-Hit Bias Measurement

    Aynı candle'da hem TP hem SL vuran trade'lerin oranını ölçer.
    Bu oran yüksekse, intrabar çözüm (1m magnifier) veya best/worst case analizi gerekebilir.

    Args:
        symbols: Test edilecek semboller
        timeframes: Test edilecek timeframe'ler
        start_date: Test başlangıç tarihi
        end_date: Test bitiş tarihi
        verbose: Detaylı çıktı

    Returns:
        dict with both-hit statistics
    """
    def log(msg: str):
        if verbose:
            print(msg)

    log("\n" + "="*70)
    log("🧪 TEST B: BOTH-HIT BIAS MEASUREMENT")
    log("="*70)
    log(f"   Symbols: {len(symbols or SYMBOLS)} sembol")
    log(f"   Timeframes: {timeframes or TIMEFRAMES}")
    log(f"   Period: {start_date} → {end_date}")
    log("="*70 + "\n")

    if symbols is None:
        symbols = SYMBOLS[:5]  # Limit for speed
    if timeframes is None:
        timeframes = ["15m", "1h", "4h"]

    # Run backtest
    log("📥 Backtest çalıştırılıyor...")

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_trades_csv = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_summary_csv = f.name

    try:
        run_portfolio_backtest(
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            out_trades_csv=temp_trades_csv,
            out_summary_csv=temp_summary_csv,
            skip_optimization=True,
            draw_trades=False,
        )

        # Read trades
        if os.path.exists(temp_trades_csv):
            trades_df = pd.read_csv(temp_trades_csv)
        else:
            log("❌ Trade CSV bulunamadı")
            return {"error": "no_trades_csv", "passed": False}

    finally:
        # Cleanup temp files
        if os.path.exists(temp_trades_csv):
            os.remove(temp_trades_csv)
        if os.path.exists(temp_summary_csv):
            os.remove(temp_summary_csv)

    if trades_df.empty:
        log("❌ Trade bulunamadı")
        return {"error": "no_trades", "passed": False}

    log(f"   ✓ {len(trades_df)} trade analiz ediliyor...")

    # Analyze both-hit trades
    # Status column contains "STOP (BothHit)", "STOP", "WIN (TP)", etc.
    status_col = 'status' if 'status' in trades_df.columns else None

    if status_col is None:
        log("❌ 'status' column bulunamadı")
        return {"error": "no_status_column", "passed": False}

    total_trades = len(trades_df)
    both_hit_trades = trades_df[trades_df[status_col].str.contains("BothHit", case=False, na=False)]
    both_hit_count = len(both_hit_trades)
    both_hit_pct = (both_hit_count / total_trades) * 100 if total_trades > 0 else 0

    log("\n" + "-"*50)
    log("📊 BOTH-HIT ANALYSIS:")
    log("-"*50)
    log(f"   Total trades: {total_trades}")
    log(f"   Both-hit trades: {both_hit_count}")
    log(f"   Both-hit percentage: {both_hit_pct:.2f}%")

    # Break down by timeframe
    log("\n   By Timeframe:")
    tf_stats = {}
    for tf in timeframes:
        tf_trades = trades_df[trades_df['timeframe'] == tf] if 'timeframe' in trades_df.columns else pd.DataFrame()
        tf_total = len(tf_trades)
        if tf_total > 0:
            tf_both_hit = len(tf_trades[tf_trades[status_col].str.contains("BothHit", case=False, na=False)])
            tf_pct = (tf_both_hit / tf_total) * 100
            tf_stats[tf] = {
                "total": tf_total,
                "both_hit": tf_both_hit,
                "percentage": tf_pct
            }
            log(f"   - {tf}: {tf_both_hit}/{tf_total} ({tf_pct:.1f}%)")

    # Calculate impact on PnL
    if 'pnl' in trades_df.columns:
        both_hit_pnl = both_hit_trades['pnl'].sum()
        total_pnl = trades_df['pnl'].sum()

        log(f"\n   Both-hit PnL impact: ${both_hit_pnl:.2f}")
        log(f"   Total PnL: ${total_pnl:.2f}")

        # What if both-hit were wins instead? (best case)
        if 'risk_amount' in trades_df.columns:
            # Estimate what TP would have been
            avg_rr = trades_df['r_multiple'].mean() if 'r_multiple' in trades_df.columns else 2.0
            best_case_pnl = total_pnl - both_hit_pnl + (both_hit_count * avg_rr * trades_df['risk_amount'].mean())
            worst_case_pnl = total_pnl  # Already worst case (all BothHit → SL)

            log(f"\n   📈 Best-case scenario (BothHit → TP): ${best_case_pnl:.2f}")
            log(f"   📉 Worst-case scenario (current, BothHit → SL): ${worst_case_pnl:.2f}")
    else:
        both_hit_pnl = 0.0
        total_pnl = 0.0

    # Determine pass/fail
    # If both-hit rate is > 10%, it's concerning for low TFs
    HIGH_CONCERN_PCT = 15.0
    MEDIUM_CONCERN_PCT = 10.0

    any_high_concern = any(
        stats["percentage"] > HIGH_CONCERN_PCT
        for tf, stats in tf_stats.items()
        if tf in ["5m", "15m"]  # Low TFs are more affected
    )

    if both_hit_pct < MEDIUM_CONCERN_PCT and not any_high_concern:
        passed = True
        log("\n✅ TEST B PASSED: Both-hit rate is acceptable")
    else:
        passed = False
        log("\n⚠️ TEST B WARNING: Elevated both-hit rate detected")
        log("\n   Recommendations:")
        log("   1. Consider intrabar resolution (1m magnifier) for low TFs")
        log("   2. Run best-case vs worst-case analysis")
        log("   3. Consider random assignment (50/50) instead of worst-case")
        log("   4. Use HTF-only mode if low TF both-hit rate is too high")

    return {
        "passed": passed,
        "total_trades": total_trades,
        "both_hit_count": both_hit_count,
        "both_hit_percentage": both_hit_pct,
        "both_hit_pnl": both_hit_pnl,
        "total_pnl": total_pnl,
        "by_timeframe": tf_stats,
        "concern_level": "high" if any_high_concern else ("medium" if both_hit_pct >= MEDIUM_CONCERN_PCT else "low"),
    }


def run_all_sanity_tests(quick: bool = False, verbose: bool = True) -> Dict:
    """Run both sanity tests and return combined results."""

    results = {
        "test_a": None,
        "test_b": None,
        "overall_passed": False,
    }

    # Test A
    if quick:
        results["test_a"] = run_test_a_window_equality(
            symbol="BTCUSDT",
            timeframe="1h",
            window_start="2025-11-01",
            window_end="2025-11-15",
            verbose=verbose
        )
    else:
        results["test_a"] = run_test_a_window_equality(
            symbol="BTCUSDT",
            timeframe="15m",
            window_start="2025-10-01",
            window_end="2025-10-31",
            verbose=verbose
        )

    # Test B
    if quick:
        results["test_b"] = run_test_b_both_hit_measurement(
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframes=["1h", "4h"],
            start_date="2025-10-01",
            end_date="2025-11-01",
            verbose=verbose
        )
    else:
        results["test_b"] = run_test_b_both_hit_measurement(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "AVAXUSDT"],
            timeframes=["15m", "1h", "4h"],
            start_date="2025-06-01",
            end_date="2025-12-01",
            verbose=verbose
        )

    # Overall assessment
    test_a_passed = results["test_a"].get("passed", False) if results["test_a"] else False
    test_b_passed = results["test_b"].get("passed", False) if results["test_b"] else False

    results["overall_passed"] = test_a_passed and test_b_passed

    print("\n" + "="*70)
    print("📋 SANITY TEST SUMMARY")
    print("="*70)
    print(f"   Test A (Window Equality): {'✅ PASSED' if test_a_passed else '❌ FAILED'}")
    print(f"   Test B (Both-Hit Bias):   {'✅ PASSED' if test_b_passed else '⚠️ WARNING'}")
    print("="*70)

    if results["overall_passed"]:
        print("\n✅ All sanity tests passed!")
        print("   → Framework logic appears correct")
        print("   → Both-hit bias is within acceptable limits")
        print("   → Can proceed with strategy evaluation")
    else:
        print("\n⚠️ Some sanity tests failed or raised warnings!")
        if not test_a_passed:
            print("   → Framework may have window slicing/state issues")
            print("   → DO NOT trust rolling WF results until fixed")
        if not test_b_passed:
            print("   → Consider intrabar resolution for accurate results")
            print("   → Run best/worst case bounds analysis")

    return results


def main():
    parser = argparse.ArgumentParser(description="Strategy Sanity Tests")
    parser.add_argument('--test-a', action='store_true', help='Run only Test A (Window Equality)')
    parser.add_argument('--test-b', action='store_true', help='Run only Test B (Both-Hit Bias)')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer symbols, shorter period)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol for Test A')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe for Test A')
    parser.add_argument('--start', type=str, default='2025-10-01', help='Start date')
    parser.add_argument('--end', type=str, default='2025-10-31', help='End date')

    args = parser.parse_args()

    if args.test_a and not args.test_b:
        # Only Test A
        result = run_test_a_window_equality(
            symbol=args.symbol,
            timeframe=args.timeframe,
            window_start=args.start,
            window_end=args.end,
        )
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")

    elif args.test_b and not args.test_a:
        # Only Test B
        result = run_test_b_both_hit_measurement(
            start_date=args.start,
            end_date=args.end,
        )
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")

    else:
        # Run both tests
        results = run_all_sanity_tests(quick=args.quick)

        # Save results
        output_file = os.path.join(DATA_DIR, "sanity_test_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
