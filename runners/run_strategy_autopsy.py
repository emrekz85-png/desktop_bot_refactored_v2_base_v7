#!/usr/bin/env python3
"""
Strategy Autopsy - Edge Analysis

Bu script strateji edge'inin nerede öldüğünü analiz eder:
1. Timeframe bazlı E[R] karşılaştırması
2. Setup bazlı kayıp analizi
3. R dağılımı (avg win R, avg loss R, winrate)
4. Baseline vs Optimizer karşılaştırması
5. Optimizer decision analysis (why streams were disabled)

Kullanım:
    python run_strategy_autopsy.py                # Full autopsy
    python run_strategy_autopsy.py --quick        # Quick smoke test
    python run_strategy_autopsy.py --baseline-only  # Sadece baseline
    python run_strategy_autopsy.py --fair         # Disable circuit breaker for fair comparison
"""

import sys
import os
import argparse
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from desktop_bot_refactored_v2_base_v7 import (
    run_portfolio_backtest,
    SYMBOLS,
    TIMEFRAMES,
    BASELINE_CONFIG,
    DATA_DIR,
)
from core.config import (
    CIRCUIT_BREAKER_CONFIG,
    ROLLING_ER_CONFIG,
    BEST_CONFIGS_FILE,
    SYMBOLS as CONFIG_SYMBOLS,
    TIMEFRAMES as CONFIG_TIMEFRAMES,
)


def run_autopsy(
    symbols: list = None,
    timeframes: list = None,
    start_date: str = "2025-06-01",
    end_date: str = "2025-12-01",
    run_optimizer: bool = True,
    fair_mode: bool = False,
    verbose: bool = True
) -> dict:
    """
    Run strategy autopsy with detailed breakdown.

    Args:
        symbols: List of symbols to test
        timeframes: List of timeframes to test
        start_date: Backtest start date
        end_date: Backtest end date
        run_optimizer: Whether to run optimizer comparison
        fair_mode: Disable circuit breaker for fair comparison
        verbose: Print detailed output
    """
    import tempfile

    def log(msg: str):
        if verbose:
            print(msg)

    if symbols is None:
        symbols = CONFIG_SYMBOLS  # Use all bot symbols
    if timeframes is None:
        timeframes = CONFIG_TIMEFRAMES  # Use all bot timeframes

    # Fair mode: disable circuit breakers temporarily
    original_cb_enabled = CIRCUIT_BREAKER_CONFIG.get("enabled", True)
    original_er_enabled = ROLLING_ER_CONFIG.get("enabled", True)

    if fair_mode:
        CIRCUIT_BREAKER_CONFIG["enabled"] = False
        ROLLING_ER_CONFIG["enabled"] = False
        log("⚠️ FAIR MODE: Circuit breaker and Rolling E[R] disabled for comparison")

    log("\n" + "="*70)
    log("🔬 STRATEGY AUTOPSY - Edge Analysis")
    log("="*70)
    log(f"   Symbols: {symbols}")
    log(f"   Timeframes: {timeframes}")
    log(f"   Period: {start_date} → {end_date}")
    log(f"   Fair Mode: {'ON' if fair_mode else 'OFF'}")
    log("="*70 + "\n")

    results = {
        "baseline": None,
        "optimizer": None,
        "comparison": None,
        "optimizer_decisions": None,
    }

    # ==========================================
    # PHASE 1: BASELINE RUN (skip_optimization=True)
    # ==========================================
    log("📊 PHASE 1: BASELINE RUN (sabit config, optimizer kapalı)")
    log("-"*50)

    with tempfile.NamedTemporaryFile(mode='w', suffix='_baseline.csv', delete=False) as f:
        baseline_trades_csv = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='_baseline_summary.csv', delete=False) as f:
        baseline_summary_csv = f.name

    try:
        run_portfolio_backtest(
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            out_trades_csv=baseline_trades_csv,
            out_summary_csv=baseline_summary_csv,
            skip_optimization=True,  # Use cached/baseline config
            draw_trades=False,
        )

        if os.path.exists(baseline_trades_csv):
            baseline_trades = pd.read_csv(baseline_trades_csv)
            results["baseline"] = analyze_trades(baseline_trades, "BASELINE", log)
        else:
            log("   ⚠️ Baseline trades CSV bulunamadı")
            results["baseline"] = {"error": "no_trades"}

    finally:
        for f in [baseline_trades_csv, baseline_summary_csv]:
            if os.path.exists(f):
                os.remove(f)

    # ==========================================
    # PHASE 2: OPTIMIZER RUN (if requested)
    # ==========================================
    if run_optimizer:
        log("\n" + "="*70)
        log("📊 PHASE 2: OPTIMIZER RUN (dinamik config)")
        log("-"*50)

        with tempfile.NamedTemporaryFile(mode='w', suffix='_opt.csv', delete=False) as f:
            opt_trades_csv = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='_opt_summary.csv', delete=False) as f:
            opt_summary_csv = f.name

        try:
            run_portfolio_backtest(
                symbols=symbols,
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date,
                out_trades_csv=opt_trades_csv,
                out_summary_csv=opt_summary_csv,
                skip_optimization=False,  # Run optimizer
                quick_mode=True,  # Use quick mode for speed
                draw_trades=False,
            )

            if os.path.exists(opt_trades_csv):
                opt_trades = pd.read_csv(opt_trades_csv)
                results["optimizer"] = analyze_trades(opt_trades, "OPTIMIZER", log)
            else:
                log("   ⚠️ Optimizer trades CSV bulunamadı")
                results["optimizer"] = {"error": "no_trades"}

        finally:
            for f in [opt_trades_csv, opt_summary_csv]:
                if os.path.exists(f):
                    os.remove(f)

        # ==========================================
        # PHASE 3: COMPARISON
        # ==========================================
        log("\n" + "="*70)
        log("📊 PHASE 3: BASELINE vs OPTIMIZER COMPARISON")
        log("="*70)

        results["comparison"] = compare_results(
            results["baseline"],
            results["optimizer"],
            log
        )

        # ==========================================
        # PHASE 4: OPTIMIZER DECISIONS ANALYSIS
        # ==========================================
        log("\n" + "="*70)
        log("📊 PHASE 4: OPTIMIZER DECISIONS ANALYSIS")
        log("="*70)

        results["optimizer_decisions"] = analyze_optimizer_decisions(
            results["baseline"],
            results["optimizer"],
            log
        )

    # Restore circuit breaker settings
    if fair_mode:
        CIRCUIT_BREAKER_CONFIG["enabled"] = original_cb_enabled
        ROLLING_ER_CONFIG["enabled"] = original_er_enabled

    return results


def analyze_optimizer_decisions(baseline: dict, optimizer: dict, log) -> dict:
    """
    Analyze what the optimizer decided for each stream.

    Identifies:
    1. Hidden gems: Streams baseline found profitable but optimizer disabled
    2. Smart cuts: Streams baseline found unprofitable and optimizer disabled
    3. Bad cuts: Streams optimizer disabled that would have been profitable
    """

    if not baseline or not optimizer:
        log("   ⚠️ Yetersiz veri")
        return {}

    decisions = {
        "hidden_gems": [],
        "smart_cuts": [],
        "bad_cuts": [],
        "disabled_streams": [],
    }

    # Load best_configs.json to see optimizer decisions
    if os.path.exists(BEST_CONFIGS_FILE):
        try:
            with open(BEST_CONFIGS_FILE, 'r') as f:
                best_configs = json.load(f)
        except:
            best_configs = {}
    else:
        best_configs = {}
        log("   ⚠️ best_configs.json bulunamadı - optimizer decisions not available")
        return decisions

    # Get baseline by stream
    baseline_by_stream = baseline.get("by_stream", {})

    # Check each stream
    for sym, sym_cfg in best_configs.items():
        if sym.startswith("_"):  # Skip _meta
            continue

        if not isinstance(sym_cfg, dict):
            continue

        for tf, tf_cfg in sym_cfg.items():
            if not isinstance(tf_cfg, dict):
                continue

            stream_key = f"{sym}-{tf}"
            is_disabled = tf_cfg.get("disabled", False)
            reason = tf_cfg.get("_reason", "unknown")

            if is_disabled:
                decisions["disabled_streams"].append({
                    "stream": stream_key,
                    "reason": reason,
                })

                # Check baseline performance for this stream
                baseline_pnl = baseline_by_stream.get(stream_key, {}).get("pnl", 0)

                if baseline_pnl > 20:  # Significant profit threshold
                    decisions["hidden_gems"].append({
                        "stream": stream_key,
                        "baseline_pnl": baseline_pnl,
                        "reason": reason,
                    })
                    log(f"   💎 HIDDEN GEM: {stream_key} (Baseline=${baseline_pnl:.2f}) disabled due to: {reason}")
                elif baseline_pnl < -10:  # Significant loss threshold
                    decisions["smart_cuts"].append({
                        "stream": stream_key,
                        "baseline_pnl": baseline_pnl,
                        "reason": reason,
                    })
                    log(f"   ✅ SMART CUT: {stream_key} (Baseline=${baseline_pnl:.2f}) correctly disabled")

    # Summary
    log(f"\n   Disabled Streams: {len(decisions['disabled_streams'])}")
    log(f"   Hidden Gems (missed profit): {len(decisions['hidden_gems'])}")
    log(f"   Smart Cuts (avoided loss): {len(decisions['smart_cuts'])}")

    if decisions["hidden_gems"]:
        total_missed = sum(g["baseline_pnl"] for g in decisions["hidden_gems"])
        log(f"\n   ⚠️ Total missed profit from hidden gems: ${total_missed:.2f}")

    return decisions


def analyze_trades(trades_df: pd.DataFrame, label: str, log) -> dict:
    """Analyze trades and return statistics."""

    if trades_df.empty:
        log(f"   ⚠️ {label}: Hiç trade yok")
        return {"error": "no_trades", "total_trades": 0}

    log(f"\n   📈 {label} Analizi ({len(trades_df)} trade)")
    log("-"*50)

    result = {
        "total_trades": len(trades_df),
        "total_pnl": 0.0,
        "by_timeframe": {},
        "by_setup": {},
        "r_distribution": {},
    }

    # Basic stats
    if 'pnl' in trades_df.columns:
        result["total_pnl"] = trades_df['pnl'].sum()
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        result["win_count"] = len(wins)
        result["loss_count"] = len(losses)
        result["win_rate"] = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0

        log(f"   Total PnL: ${result['total_pnl']:.2f}")
        log(f"   Win Rate: {result['win_rate']:.1f}% ({result['win_count']}/{result['total_trades']})")

    # ==========================================
    # 1. TIMEFRAME BREAKDOWN
    # ==========================================
    log(f"\n   📊 Timeframe Breakdown:")
    if 'timeframe' in trades_df.columns:
        for tf in trades_df['timeframe'].unique():
            tf_trades = trades_df[trades_df['timeframe'] == tf]
            tf_pnl = tf_trades['pnl'].sum() if 'pnl' in tf_trades.columns else 0
            tf_wins = len(tf_trades[tf_trades['pnl'] > 0]) if 'pnl' in tf_trades.columns else 0
            tf_wr = tf_wins / len(tf_trades) * 100 if len(tf_trades) > 0 else 0

            # R-multiple stats
            if 'r_multiple' in tf_trades.columns:
                avg_r = tf_trades['r_multiple'].mean()
                win_r = tf_trades[tf_trades['pnl'] > 0]['r_multiple'].mean() if tf_wins > 0 else 0
                loss_r = tf_trades[tf_trades['pnl'] <= 0]['r_multiple'].mean() if len(tf_trades) - tf_wins > 0 else 0
                expected_r = (tf_wr/100 * win_r) + ((1-tf_wr/100) * loss_r) if win_r and loss_r else avg_r
            else:
                avg_r = win_r = loss_r = expected_r = 0

            result["by_timeframe"][tf] = {
                "trades": len(tf_trades),
                "pnl": tf_pnl,
                "win_rate": tf_wr,
                "avg_r": avg_r,
                "win_r": win_r,
                "loss_r": loss_r,
                "expected_r": expected_r,
            }

            status = "✅" if tf_pnl > 0 else "❌"
            log(f"   {status} {tf}: {len(tf_trades)} trades, ${tf_pnl:.2f}, WR={tf_wr:.0f}%, E[R]={expected_r:.3f}")

    # ==========================================
    # 2. SETUP/REASON BREAKDOWN
    # ==========================================
    log(f"\n   📊 Setup Breakdown:")
    setup_col = 'setup' if 'setup' in trades_df.columns else ('reason' if 'reason' in trades_df.columns else None)

    if setup_col:
        for setup in trades_df[setup_col].unique():
            setup_trades = trades_df[trades_df[setup_col] == setup]
            setup_pnl = setup_trades['pnl'].sum() if 'pnl' in setup_trades.columns else 0
            setup_wins = len(setup_trades[setup_trades['pnl'] > 0]) if 'pnl' in setup_trades.columns else 0
            setup_wr = setup_wins / len(setup_trades) * 100 if len(setup_trades) > 0 else 0

            result["by_setup"][str(setup)] = {
                "trades": len(setup_trades),
                "pnl": setup_pnl,
                "win_rate": setup_wr,
            }

            status = "✅" if setup_pnl > 0 else "❌"
            log(f"   {status} {setup}: {len(setup_trades)} trades, ${setup_pnl:.2f}, WR={setup_wr:.0f}%")

    # ==========================================
    # 3. R-MULTIPLE DISTRIBUTION
    # ==========================================
    log(f"\n   📊 R-Multiple Distribution:")
    if 'r_multiple' in trades_df.columns:
        r_vals = trades_df['r_multiple']
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]

        avg_win_r = wins['r_multiple'].mean() if len(wins) > 0 else 0
        avg_loss_r = losses['r_multiple'].mean() if len(losses) > 0 else 0
        overall_avg_r = r_vals.mean()
        wr = len(wins) / len(trades_df) * 100

        # Expected R = WR * Avg_Win_R + (1-WR) * Avg_Loss_R
        expected_r = (wr/100 * avg_win_r) + ((1-wr/100) * avg_loss_r)

        result["r_distribution"] = {
            "avg_win_r": avg_win_r,
            "avg_loss_r": avg_loss_r,
            "overall_avg_r": overall_avg_r,
            "expected_r": expected_r,
            "win_rate": wr,
        }

        log(f"   Avg Win R: {avg_win_r:.3f}")
        log(f"   Avg Loss R: {avg_loss_r:.3f}")
        log(f"   Overall Avg R: {overall_avg_r:.3f}")
        log(f"   E[R] = {wr:.1f}% × {avg_win_r:.2f} + {100-wr:.1f}% × {avg_loss_r:.2f} = {expected_r:.3f}")

        if expected_r < 0:
            log(f"   ⚠️ Negative expectancy! Edge yok.")
        elif expected_r < 0.1:
            log(f"   ⚠️ Marginal expectancy. Edge çok zayıf.")
        else:
            log(f"   ✅ Positive expectancy.")

    # ==========================================
    # 4. STREAM-LEVEL BREAKDOWN (for optimizer analysis)
    # ==========================================
    result["by_stream"] = {}
    if 'symbol' in trades_df.columns and 'timeframe' in trades_df.columns:
        stream_stats = trades_df.groupby(['symbol', 'timeframe']).agg({
            'pnl': 'sum',
            'symbol': 'count'  # trade count
        }).rename(columns={'symbol': 'trades'})

        for (sym, tf), row in stream_stats.iterrows():
            stream_key = f"{sym}-{tf}"
            result["by_stream"][stream_key] = {
                "pnl": row['pnl'],
                "trades": int(row['trades']),
            }

    # ==========================================
    # 5. WORST PERFORMERS
    # ==========================================
    log(f"\n   📊 En Kötü Performans Gösteren Stream'ler:")
    if 'symbol' in trades_df.columns and 'timeframe' in trades_df.columns:
        stream_stats = trades_df.groupby(['symbol', 'timeframe']).agg({
            'pnl': 'sum',
            'symbol': 'count'  # trade count
        }).rename(columns={'symbol': 'trades'})

        worst = stream_stats.nsmallest(5, 'pnl')
        for (sym, tf), row in worst.iterrows():
            log(f"   ❌ {sym}-{tf}: ${row['pnl']:.2f} ({int(row['trades'])} trades)")

        # Also show best performers
        log(f"\n   📊 En İyi Performans Gösteren Stream'ler:")
        best = stream_stats.nlargest(5, 'pnl')
        for (sym, tf), row in best.iterrows():
            log(f"   ✅ {sym}-{tf}: ${row['pnl']:.2f} ({int(row['trades'])} trades)")

    return result


def compare_results(baseline: dict, optimizer: dict, log) -> dict:
    """Compare baseline vs optimizer results."""

    if not baseline or not optimizer:
        log("   ⚠️ Karşılaştırma için yeterli veri yok")
        return {}

    if baseline.get("error") or optimizer.get("error"):
        log("   ⚠️ Bir veya her iki koşuda hata var")
        return {}

    comparison = {
        "baseline_trades": baseline.get("total_trades", 0),
        "optimizer_trades": optimizer.get("total_trades", 0),
        "baseline_pnl": baseline.get("total_pnl", 0),
        "optimizer_pnl": optimizer.get("total_pnl", 0),
    }

    log(f"\n   BASELINE:  {comparison['baseline_trades']} trades, ${comparison['baseline_pnl']:.2f}")
    log(f"   OPTIMIZER: {comparison['optimizer_trades']} trades, ${comparison['optimizer_pnl']:.2f}")

    trade_diff = comparison['optimizer_trades'] - comparison['baseline_trades']
    pnl_diff = comparison['optimizer_pnl'] - comparison['baseline_pnl']

    comparison["trade_diff"] = trade_diff
    comparison["pnl_diff"] = pnl_diff

    log(f"\n   Trade farkı: {trade_diff:+d}")
    log(f"   PnL farkı: ${pnl_diff:+.2f}")

    # Analyze what optimizer did
    if trade_diff < 0:
        log(f"\n   📉 Optimizer {abs(trade_diff)} daha az trade aldı")
        if pnl_diff > 0:
            log(f"   ✅ Ama PnL daha iyi! Optimizer kötü trade'leri engelledi.")
        else:
            log(f"   ❌ Ve PnL daha kötü. Optimizer iyi trade'leri de engelledi.")
    else:
        log(f"\n   📈 Optimizer {trade_diff} daha fazla trade aldı")
        if pnl_diff > 0:
            log(f"   ✅ Ve PnL daha iyi! Optimizer edge'i buldu.")
        else:
            log(f"   ❌ Ama PnL daha kötü. Optimizer kötü trade'ler ekledi.")

    # Compare by timeframe
    log(f"\n   Timeframe Karşılaştırması:")
    baseline_tf = baseline.get("by_timeframe", {})
    optimizer_tf = optimizer.get("by_timeframe", {})

    all_tfs = set(baseline_tf.keys()) | set(optimizer_tf.keys())
    for tf in sorted(all_tfs):
        b_pnl = baseline_tf.get(tf, {}).get("pnl", 0)
        o_pnl = optimizer_tf.get(tf, {}).get("pnl", 0)
        diff = o_pnl - b_pnl
        status = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        log(f"   {tf}: Baseline=${b_pnl:.2f}, Optimizer=${o_pnl:.2f} ({status}${abs(diff):.2f})")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Strategy Autopsy")
    parser.add_argument('--quick', action='store_true', help='Quick smoke test')
    parser.add_argument('--baseline-only', action='store_true', help='Only run baseline')
    parser.add_argument('--fair', action='store_true', help='Disable circuit breaker for fair comparison')
    parser.add_argument('--start', type=str, default='2025-06-01', help='Start date')
    parser.add_argument('--end', type=str, default='2025-12-01', help='End date')

    args = parser.parse_args()

    if args.quick:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        timeframes = ["15m", "1h", "4h"]
        start_date = "2025-10-01"
        end_date = "2025-11-01"
    else:
        # Use all bot symbols and timeframes by default
        symbols = CONFIG_SYMBOLS  # 11 symbols
        timeframes = CONFIG_TIMEFRAMES  # 7 timeframes (5m, 15m, 30m, 1h, 4h, 12h, 1d)
        start_date = args.start
        end_date = args.end

    results = run_autopsy(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        run_optimizer=not args.baseline_only,
        fair_mode=args.fair,
    )

    # Save results
    output_file = os.path.join(DATA_DIR, "autopsy_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📁 Results saved to: {output_file}")

    # Final summary
    print("\n" + "="*70)
    print("🔬 AUTOPSY SUMMARY")
    print("="*70)

    if results.get("baseline"):
        baseline = results["baseline"]
        print(f"\n   BASELINE:")
        print(f"   - Total Trades: {baseline.get('total_trades', 0)}")
        print(f"   - Total PnL: ${baseline.get('total_pnl', 0):.2f}")
        print(f"   - Win Rate: {baseline.get('win_rate', 0):.1f}%")

        r_dist = baseline.get("r_distribution", {})
        if r_dist:
            print(f"   - E[R]: {r_dist.get('expected_r', 0):.3f}")

    if results.get("optimizer"):
        optimizer = results["optimizer"]
        print(f"\n   OPTIMIZER:")
        print(f"   - Total Trades: {optimizer.get('total_trades', 0)}")
        print(f"   - Total PnL: ${optimizer.get('total_pnl', 0):.2f}")
        print(f"   - Win Rate: {optimizer.get('win_rate', 0):.1f}%")

        r_dist = optimizer.get("r_distribution", {})
        if r_dist:
            print(f"   - E[R]: {r_dist.get('expected_r', 0):.3f}")

    if results.get("comparison"):
        comp = results["comparison"]
        print(f"\n   VERDICT:")
        if comp.get("pnl_diff", 0) > 0:
            print(f"   ✅ Optimizer ${comp['pnl_diff']:.2f} daha iyi")
        else:
            print(f"   ❌ Optimizer ${abs(comp.get('pnl_diff', 0)):.2f} daha kötü")


if __name__ == "__main__":
    main()
