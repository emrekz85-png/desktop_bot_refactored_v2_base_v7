#!/usr/bin/env python3
"""
Optuna-Based Optimizer for SSL Flow Strategy

Bayesian optimization to find the best combination of ALL parameters:
- 25+ parameters optimized simultaneously
- 100-200 trials instead of 60,000+ grid search combinations
- Walk-forward validation built-in
- CPU optimized for MacBook Air M-series

Usage:
    # Quick pilot test (50 trials, 1 symbol)
    python runners/run_optuna_optimizer.py --pilot

    # Standard optimization (150 trials)
    python runners/run_optuna_optimizer.py --symbols BTCUSDT ETHUSDT LINKUSDT

    # Full year data
    python runners/run_optuna_optimizer.py --full-year

    # Custom trials
    python runners/run_optuna_optimizer.py --trials 200 --symbols BTCUSDT

Author: Claude Code
Created: 2025-01-02
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    TRADING_CONFIG, SYMBOLS, TIMEFRAMES,
    calculate_indicators, BinanceClient, get_client,
)

# Import Optuna optimizer
from core.optuna_optimizer import (
    OPTUNA_AVAILABLE,
    SSLFlowOptimizer,
    optimize_multiple_streams,
    ParameterSpace,
)


# ==========================================
# CONSTANTS
# ==========================================

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "LINKUSDT"]
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h"]
DEFAULT_CANDLES = 20000  # ~208 days for 15m

# Pilot mode settings
PILOT_TRIALS = 50
PILOT_SYMBOLS = ["BTCUSDT"]
PILOT_TIMEFRAMES = ["15m"]
PILOT_CANDLES = 10000


# ==========================================
# DATA LOADING
# ==========================================

def load_data(
    symbols: List[str],
    timeframes: List[str],
    candles: int,
    verbose: bool = True,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Load and prepare data for optimization.

    Returns dict mapping (symbol, tf) to DataFrame with indicators.
    """
    client = get_client()
    streams = {}

    total = len(symbols) * len(timeframes)
    current = 0

    for sym in symbols:
        for tf in timeframes:
            current += 1
            if verbose:
                print(f"[DATA] Loading {sym}-{tf}... ({current}/{total})")

            try:
                # Fetch candles using BinanceClient
                # Use paginated method for >1000 candles
                if candles > 1000:
                    df = client.get_klines_paginated(sym, tf, total_candles=candles)
                else:
                    df = client.get_klines(sym, tf, limit=candles)

                if df is None or len(df) == 0:
                    if verbose:
                        print(f"[DATA] {sym}-{tf}: No data returned")
                    continue

                # Calculate indicators
                df = calculate_indicators(df, tf)

                if len(df) < 300:
                    if verbose:
                        print(f"[DATA] {sym}-{tf}: Insufficient data ({len(df)} candles)")
                    continue

                streams[(sym, tf)] = df
                if verbose:
                    print(f"[DATA] {sym}-{tf}: {len(df)} candles loaded")

            except Exception as e:
                print(f"[DATA] Error loading {sym}-{tf}: {e}")
                continue

    return streams


# ==========================================
# MAIN OPTIMIZATION
# ==========================================

def run_optimization(
    symbols: List[str],
    timeframes: List[str],
    candles: int,
    n_trials: int,
    train_ratio: float = 0.70,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict:
    """Run Optuna optimization for all symbol-timeframe combinations.

    Returns dict with:
    - 'configs': Best config per stream
    - 'summary': Summary statistics
    - 'importances': Parameter importances
    """
    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna not installed. Run: pip install optuna")
        return {}

    print("\n" + "=" * 60)
    print("OPTUNA SSL FLOW OPTIMIZER")
    print("=" * 60)
    print(f"\nSymbols: {symbols}")
    print(f"Timeframes: {timeframes}")
    print(f"Candles: {candles}")
    print(f"Trials per stream: {n_trials}")
    print(f"Train/Test split: {train_ratio:.0%}/{1-train_ratio:.0%}")

    # Load data
    print("\n[STEP 1] Loading data...")
    streams = load_data(symbols, timeframes, candles, verbose)

    if not streams:
        print("ERROR: No data loaded")
        return {}

    print(f"\nLoaded {len(streams)} streams")

    # Run optimization
    print("\n[STEP 2] Running Optuna optimization...")
    results = optimize_multiple_streams(
        streams=streams,
        n_trials=n_trials,
        train_ratio=train_ratio,
        verbose=verbose,
    )

    # Compile summary
    print("\n[STEP 3] Compiling results...")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "symbols": symbols,
        "timeframes": timeframes,
        "candles": candles,
        "n_trials": n_trials,
        "train_ratio": train_ratio,
        "total_streams": len(streams),
        "enabled_streams": 0,
        "total_pnl": 0.0,
        "total_trades": 0,
    }

    all_importances = {}

    for (sym, tf), config in results.items():
        if config.get("disabled"):
            continue

        summary["enabled_streams"] += 1
        summary["total_pnl"] += config.get("_net_pnl", 0)
        summary["total_trades"] += config.get("_trades", 0)

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nEnabled Streams: {summary['enabled_streams']}/{summary['total_streams']}")
    print(f"Total PnL (Train): ${summary['total_pnl']:.2f}")
    print(f"Total Trades: {summary['total_trades']}")

    print("\n--- Best Configs per Stream ---")
    for (sym, tf), config in results.items():
        if config.get("disabled"):
            reason = config.get("_reason", "unknown")
            print(f"\n{sym}-{tf}: DISABLED ({reason})")
            continue

        print(f"\n{sym}-{tf}:")
        print(f"  RR: {config.get('rr')}, RSI: {config.get('rsi')}")
        print(f"  PnL: ${config.get('_net_pnl', 0):.2f}, Trades: {config.get('_trades', 0)}")
        print(f"  E[R]: {config.get('_expected_r', 0):.3f}, Win Rate: {config.get('_win_rate', 0):.1%}")

        # Key filter states
        filters = []
        if config.get("use_market_structure"):
            filters.append(f"MS(score≥{config.get('min_ms_score', 0)})")
        if config.get("use_fvg_bonus"):
            filters.append("FVG+")
        if config.get("skip_wick_rejection"):
            filters.append("NoWick")
        if config.get("skip_body_position"):
            filters.append("NoBody")
        if config.get("use_trailing"):
            filters.append("Trail")
        if config.get("momentum_tp_extension"):
            filters.append("MomTP")

        print(f"  Filters: {', '.join(filters) if filters else 'Standard'}")

        # OOS validation
        if config.get("_walk_forward_validated"):
            oos_pnl = config.get("_oos_pnl", 0)
            oos_ratio = config.get("_overfit_ratio", 0)
            print(f"  OOS: ${oos_pnl:.2f} (ratio: {oos_ratio:.2f}) ✓")
        else:
            print(f"  OOS: Not validated")

    # Save results
    if save_results:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "optuna_runs"
        )
        os.makedirs(output_dir, exist_ok=True)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"optuna_{run_id}.json")

        # Prepare serializable output
        output = {
            "summary": summary,
            "configs": {},
        }

        for (sym, tf), config in results.items():
            key = f"{sym}-{tf}"
            # Filter out non-serializable items
            clean_config = {}
            for k, v in config.items():
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    clean_config[k] = v
            output["configs"][key] = clean_config

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    return {
        "configs": results,
        "summary": summary,
    }


# ==========================================
# CLI ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Optuna-based optimizer for SSL Flow strategy"
    )

    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run quick pilot test (50 trials, 1 symbol)"
    )

    parser.add_argument(
        "--full-year",
        action="store_true",
        help="Use full year data (35000 candles)"
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to optimize (default: BTCUSDT ETHUSDT LINKUSDT)"
    )

    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
        help="Timeframes to optimize (default: 15m 1h 4h)"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=150,
        help="Number of Optuna trials per stream (default: 150)"
    )

    parser.add_argument(
        "--candles",
        type=int,
        default=None,
        help="Number of candles to use (default: 20000)"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train/test split ratio (default: 0.70)"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Pilot mode overrides
    if args.pilot:
        symbols = PILOT_SYMBOLS
        timeframes = PILOT_TIMEFRAMES
        candles = PILOT_CANDLES
        n_trials = PILOT_TRIALS
        print("\n[MODE] Pilot test mode")
    else:
        symbols = args.symbols or DEFAULT_SYMBOLS
        timeframes = args.timeframes or DEFAULT_TIMEFRAMES
        candles = args.candles or (35000 if args.full_year else DEFAULT_CANDLES)
        n_trials = args.trials

    # Run optimization
    run_optimization(
        symbols=symbols,
        timeframes=timeframes,
        candles=candles,
        n_trials=n_trials,
        train_ratio=args.train_ratio,
        save_results=not args.no_save,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
