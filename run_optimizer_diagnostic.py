#!/usr/bin/env python3
"""
Optimizer Diagnostic Script

Diagnoses why the optimizer disabled a specific stream (e.g., ETHUSDT-15m).
Shows all decision points and thresholds to understand optimizer gating.

Usage:
    python run_optimizer_diagnostic.py                     # Default: ETHUSDT-15m
    python run_optimizer_diagnostic.py --symbol BTCUSDT --tf 15m
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import argparse


def run_diagnostic(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    days: int = 120,
    verbose: bool = True
) -> dict:
    """
    Run optimizer diagnostic for a specific stream.

    Shows all decision points:
    1. Train E[R] vs threshold
    2. OOS E[R] vs threshold
    3. Overfit ratio vs threshold
    4. Score vs threshold
    5. Trade count vs confidence threshold
    """
    # Import after path setup
    from desktop_bot_refactored_v2_base_v7 import (
        TradingEngine,
        _score_config_for_stream,
        _compute_optimizer_score,
        _validate_config_oos,
        _check_overfit,
        _generate_candidate_configs,
        TRADING_CONFIG,
        WALK_FORWARD_CONFIG,
    )
    from core.config import (
        MIN_EXPECTANCY_R_MULTIPLE,
        MIN_SCORE_THRESHOLD,
        MIN_OOS_TRADES_BY_TF,
        CANDLES_PER_DAY,
    )

    log_lines = []

    def log(msg: str):
        """Log to both console and result."""
        print(msg)
        log_lines.append(msg)

    log("=" * 80)
    log(f"OPTIMIZER DIAGNOSTIC: {symbol}-{timeframe}")
    log("=" * 80)

    # Calculate candle count
    candles_per_day = CANDLES_PER_DAY.get(timeframe, 96)
    total_candles = days * candles_per_day

    log(f"\nConfiguration:")
    log(f"  Days: {days}")
    log(f"  Total candles: {total_candles}")

    # Get thresholds
    min_er = MIN_EXPECTANCY_R_MULTIPLE.get(timeframe, 0.05)
    min_score_base = MIN_SCORE_THRESHOLD.get(timeframe, 15.0)
    min_oos_trades = MIN_OOS_TRADES_BY_TF.get(timeframe, 12)
    train_ratio = WALK_FORWARD_CONFIG.get("train_ratio", 0.70)
    min_overfit_ratio = WALK_FORWARD_CONFIG.get("min_overfit_ratio", 0.70)

    # Walk-forward adjusts score threshold
    min_score = min_score_base * train_ratio

    log(f"\nThresholds for {timeframe}:")
    log(f"  MIN_EXPECTANCY_R_MULTIPLE: {min_er}")
    log(f"  MIN_SCORE_THRESHOLD (base): {min_score_base}")
    log(f"  MIN_SCORE_THRESHOLD (WF adj): {min_score:.2f} (base * {train_ratio})")
    log(f"  MIN_OOS_TRADES: {min_oos_trades}")
    log(f"  MIN_OVERFIT_RATIO: {min_overfit_ratio}")

    # Fetch data
    log(f"\n" + "-" * 40)
    log(f"Fetching {total_candles} candles for {symbol}-{timeframe}...")

    engine = TradingEngine(TRADING_CONFIG)
    df_full = engine.get_klines(symbol, timeframe, limit=total_candles)

    if df_full is None or len(df_full) < 500:
        log(f"ERROR: Insufficient data ({len(df_full) if df_full is not None else 0} candles)")
        return {"error": "insufficient_data"}

    log(f"Fetched {len(df_full)} candles")
    log(f"Date range: {df_full.iloc[0]['open_time']} to {df_full.iloc[-1]['open_time']}")

    # Split into train/test
    split_idx = int(len(df_full) * train_ratio)
    df_train = df_full.iloc[:split_idx].copy()
    df_test = df_full.iloc[split_idx:].copy()

    log(f"\nWalk-Forward Split:")
    log(f"  Train: {len(df_train)} candles ({df_train.iloc[0]['open_time']} to {df_train.iloc[-1]['open_time']})")
    log(f"  Test:  {len(df_test)} candles ({df_test.iloc[0]['open_time']} to {df_test.iloc[-1]['open_time']})")

    # Generate candidate configs
    log(f"\n" + "-" * 40)
    log(f"Generating candidate configs...")
    candidates = _generate_candidate_configs(quick_mode=False)
    log(f"Generated {len(candidates)} candidate configs")

    # Score each config on TRAIN data
    log(f"\n" + "-" * 40)
    log(f"Scoring configs on TRAIN data...")

    best_cfg = None
    best_score = -float('inf')
    best_pnl = 0
    best_trades = 0
    best_expected_r = 0

    # Track why configs fail
    fail_reasons = {
        "negative_pnl": 0,
        "low_trades": 0,
        "low_er": 0,
        "low_score": 0,
    }

    positive_configs = []

    for i, cfg in enumerate(candidates):
        try:
            net_pnl, trades, trade_pnls, trade_r_multiples = _score_config_for_stream(
                df_train, symbol, timeframe, cfg
            )
        except Exception as e:
            continue

        if trades == 0:
            fail_reasons["low_trades"] += 1
            continue

        if net_pnl <= 0:
            fail_reasons["negative_pnl"] += 1
            continue

        expected_r = sum(trade_r_multiples) / len(trade_r_multiples) if trade_r_multiples else 0

        if expected_r < min_er:
            fail_reasons["low_er"] += 1
            continue

        score = _compute_optimizer_score(
            net_pnl=net_pnl,
            trades=trades,
            trade_pnls=trade_pnls,
            tf=timeframe,
            trade_r_multiples=trade_r_multiples
        )

        if score < min_score:
            fail_reasons["low_score"] += 1
            # Still track it for comparison

        # Track all positive configs
        positive_configs.append({
            "cfg": cfg,
            "pnl": net_pnl,
            "trades": trades,
            "expected_r": expected_r,
            "score": score,
        })

        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_pnl = net_pnl
            best_trades = trades
            best_expected_r = expected_r

    log(f"\nConfig Scoring Results (Train):")
    log(f"  Total candidates: {len(candidates)}")
    log(f"  Rejected (negative PnL): {fail_reasons['negative_pnl']}")
    log(f"  Rejected (0 trades): {fail_reasons['low_trades']}")
    log(f"  Rejected (E[R] < {min_er}): {fail_reasons['low_er']}")
    log(f"  Below score threshold: {fail_reasons['low_score']}")
    log(f"  Positive configs found: {len(positive_configs)}")

    if positive_configs:
        # Sort by score and show top 5
        positive_configs.sort(key=lambda x: x['score'], reverse=True)
        log(f"\n  Top 5 configs by score:")
        for i, pc in enumerate(positive_configs[:5]):
            c = pc['cfg']
            log(f"    {i+1}. RR={c['rr']}, RSI={c['rsi']}, AT={c['at_active']} | "
                f"PnL=${pc['pnl']:.2f}, E[R]={pc['expected_r']:.3f}, "
                f"Trades={pc['trades']}, Score={pc['score']:.2f}")

    if not best_cfg:
        log(f"\n❌ NO POSITIVE CONFIG FOUND!")
        log(f"   All configs either had negative PnL, 0 trades, or E[R] below threshold")
        return {
            "result": "disabled",
            "reason": "no_positive_config",
            "log": log_lines
        }

    log(f"\n" + "-" * 40)
    log(f"BEST TRAIN CONFIG:")
    log(f"  RR={best_cfg['rr']}, RSI={best_cfg['rsi']}, AT={best_cfg['at_active']}")
    log(f"  Train PnL: ${best_pnl:.2f}")
    log(f"  Train Trades: {best_trades}")
    log(f"  Train E[R]: {best_expected_r:.4f}")
    log(f"  Train Score: {best_score:.2f}")

    # Check score threshold
    log(f"\nScore Check:")
    if best_score >= min_score:
        log(f"  ✅ PASS: Score {best_score:.2f} >= threshold {min_score:.2f}")
    else:
        log(f"  ❌ FAIL: Score {best_score:.2f} < threshold {min_score:.2f}")
        log(f"     → STREAM WOULD BE DISABLED (weak edge)")
        return {
            "result": "disabled",
            "reason": "low_score",
            "best_score": best_score,
            "min_score": min_score,
            "log": log_lines
        }

    # Walk-Forward OOS Validation
    log(f"\n" + "-" * 40)
    log(f"WALK-FORWARD OOS VALIDATION")

    oos_result = _validate_config_oos(df_test, symbol, timeframe, best_cfg)

    if oos_result is None:
        log(f"  ⚠️ OOS validation failed (not enough data)")
    else:
        oos_pnl = oos_result.get('oos_pnl', 0)
        oos_trades = oos_result.get('oos_trades', 0)
        oos_expected_r = oos_result.get('oos_expected_r', 0)
        oos_win_rate = oos_result.get('oos_win_rate', 0)

        log(f"  OOS PnL: ${oos_pnl:.2f}")
        log(f"  OOS Trades: {oos_trades}")
        log(f"  OOS E[R]: {oos_expected_r:.4f}")
        log(f"  OOS Win Rate: {oos_win_rate:.1%}")

        # Check for overfit
        is_overfit, overfit_ratio, overfit_reason = _check_overfit(
            best_expected_r, oos_result, timeframe
        )

        log(f"\n  Overfit Check:")
        log(f"    Train E[R]: {best_expected_r:.4f}")
        log(f"    OOS E[R]: {oos_expected_r:.4f}")
        log(f"    Overfit Ratio: {overfit_ratio:.2f} (OOS/Train)")
        log(f"    Min Required Ratio: {min_overfit_ratio}")
        log(f"    Min OOS Trades Required: {min_oos_trades}")
        log(f"    Actual OOS Trades: {oos_trades}")

        if oos_trades < min_oos_trades:
            log(f"\n  ⚠️ INSUFFICIENT OOS TRADES: {oos_trades} < {min_oos_trades}")
            log(f"     → Overfit check SKIPPED (not enough statistical significance)")
        elif is_overfit:
            log(f"\n  ❌ OVERFIT DETECTED: {overfit_reason}")
            log(f"     → STREAM WOULD BE DISABLED")
            return {
                "result": "disabled",
                "reason": f"overfit:{overfit_reason}",
                "overfit_ratio": overfit_ratio,
                "train_er": best_expected_r,
                "oos_er": oos_expected_r,
                "log": log_lines
            }
        else:
            log(f"\n  ✅ OOS VALIDATED: {overfit_reason}")

    # Confidence Level Check
    log(f"\n" + "-" * 40)
    log(f"CONFIDENCE LEVEL CHECK")

    # Get min trades for this timeframe (from optimizer logic)
    # The optimizer uses a base of 40 trades, scaled by min_er
    tf_base_trades = {
        "1m": 80, "5m": 60, "15m": 40, "30m": 35, "1h": 30, "4h": 20, "12h": 15, "1d": 10
    }
    tf_min_trades = tf_base_trades.get(timeframe, 40)

    log(f"  Train Trades: {best_trades}")
    log(f"  Min for High Confidence: {tf_min_trades}")
    log(f"  Min for Medium Confidence: {int(tf_min_trades * 0.6)}")

    if best_trades >= tf_min_trades:
        confidence = "high"
        log(f"  ✅ HIGH CONFIDENCE: {best_trades} >= {tf_min_trades}")
    elif best_trades >= tf_min_trades * 0.6:
        confidence = "medium"
        log(f"  ⚠️ MEDIUM CONFIDENCE: {best_trades} >= {int(tf_min_trades * 0.6)}")
    else:
        confidence = "low"
        log(f"  ❌ LOW CONFIDENCE: {best_trades} < {int(tf_min_trades * 0.6)}")
        log(f"     → STREAM WOULD BE DISABLED (risk multiplier = 0)")
        return {
            "result": "disabled",
            "reason": "low_confidence",
            "trades": best_trades,
            "min_trades": int(tf_min_trades * 0.6),
            "log": log_lines
        }

    # Final Result
    log(f"\n" + "=" * 80)
    log(f"FINAL RESULT: ✅ STREAM SHOULD BE ENABLED")
    log(f"=" * 80)
    log(f"  Best Config: RR={best_cfg['rr']}, RSI={best_cfg['rsi']}, AT={best_cfg['at_active']}")
    log(f"  Train: PnL=${best_pnl:.2f}, E[R]={best_expected_r:.4f}, Trades={best_trades}")
    if oos_result:
        log(f"  OOS: PnL=${oos_pnl:.2f}, E[R]={oos_expected_r:.4f}, Trades={oos_trades}")
        log(f"  Overfit Ratio: {overfit_ratio:.2f}")
    log(f"  Confidence: {confidence.upper()}")
    log(f"  Score: {best_score:.2f} (min={min_score:.2f})")

    return {
        "result": "enabled",
        "config": best_cfg,
        "train_pnl": best_pnl,
        "train_er": best_expected_r,
        "train_trades": best_trades,
        "train_score": best_score,
        "oos_pnl": oos_pnl if oos_result else None,
        "oos_er": oos_expected_r if oos_result else None,
        "oos_trades": oos_trades if oos_result else None,
        "overfit_ratio": overfit_ratio if oos_result else None,
        "confidence": confidence,
        "log": log_lines
    }


def main():
    parser = argparse.ArgumentParser(description="Optimizer Diagnostic Script")
    parser.add_argument('--symbol', '-s', type=str, default="ETHUSDT", help='Symbol to diagnose')
    parser.add_argument('--tf', '-t', type=str, default="15m", help='Timeframe to diagnose')
    parser.add_argument('--days', '-d', type=int, default=120, help='Days of data to use')

    args = parser.parse_args()

    result = run_diagnostic(
        symbol=args.symbol,
        timeframe=args.tf,
        days=args.days,
        verbose=True
    )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    if result.get("error"):
        print(f"Error: {result['error']}")
    elif result.get("result") == "disabled":
        print(f"Stream would be DISABLED")
        print(f"Reason: {result.get('reason', 'unknown')}")
    else:
        print(f"Stream would be ENABLED")
        print(f"Confidence: {result.get('confidence', 'unknown').upper()}")


if __name__ == "__main__":
    main()
