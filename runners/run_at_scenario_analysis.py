#!/usr/bin/env python3
"""
AT Scenario Analysis Runner

Purpose: Run the AlphaTrend scenario analysis on BTC 15m data
to find the optimal AT configuration through data-driven analysis.

This script:
1. Fetches 1 year of BTC 15m data
2. Runs ATScenarioAnalyzer with multiple AT configurations
3. Outputs comparison of all scenarios
4. Identifies the optimal AT settings
5. Analyzes AT state patterns for insights

Usage:
    python runners/run_at_scenario_analysis.py
    python runners/run_at_scenario_analysis.py --symbol BTCUSDT --timeframe 15m
    python runners/run_at_scenario_analysis.py --days 180  # Last 6 months
    python runners/run_at_scenario_analysis.py --export results.json

Created: 2026-01-03
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import calculate_indicators, get_client
from core.at_scenario_analyzer import ATScenarioAnalyzer, AT_SCENARIOS


def fetch_data(
    symbol: str,
    timeframe: str,
    days: int = 365,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical data from Binance.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Candlestick interval (e.g., "15m")
        days: Number of days of history
        verbose: Print progress

    Returns:
        DataFrame with OHLCV and indicator columns
    """
    import requests

    if verbose:
        print(f"\nFetching {days} days of {symbol} {timeframe} data...")

    client = get_client()

    # Calculate number of candles
    candles_per_day = {
        "15m": 96,
        "1h": 24,
        "4h": 6,
        "1d": 1,
    }
    n_candles = days * candles_per_day.get(timeframe, 96)

    if verbose:
        print(f"  Target: {n_candles} candles")

    # Fetch in chunks
    all_dfs = []
    remaining = n_candles
    end_time = None

    while remaining > 0:
        chunk_size = min(remaining, 1000)

        try:
            if end_time:
                url = f"{client.BASE_URL}/klines"
                params = {
                    'symbol': symbol,
                    'interval': timeframe,
                    'limit': chunk_size,
                    'endTime': end_time
                }
                res = requests.get(url, params=params, timeout=15)
                if res.status_code != 200:
                    print(f"  Error: HTTP {res.status_code}")
                    break
                data = res.json()
                if not data:
                    break

                df_chunk = pd.DataFrame(data).iloc[:, :6]
                df_chunk.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms', utc=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')
            else:
                df_chunk = client.get_klines(symbol=symbol, interval=timeframe, limit=chunk_size)

            if df_chunk.empty:
                break

            all_dfs.insert(0, df_chunk)
            remaining -= len(df_chunk)

            # Get oldest timestamp for next iteration
            if 'timestamp' in df_chunk.columns:
                end_time = int(df_chunk['timestamp'].iloc[0].timestamp() * 1000) - 1
            else:
                break

            if len(df_chunk) < chunk_size:
                break

        except Exception as e:
            print(f"  Error fetching chunk: {e}")
            break

    if not all_dfs:
        return pd.DataFrame()

    # Combine all chunks
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df.set_index('timestamp', inplace=True)

    if verbose:
        print(f"  Got {len(df)} candles: {df.index[0]} to {df.index[-1]}")

    # Calculate indicators
    if verbose:
        print("  Calculating indicators...")
    df = calculate_indicators(df, timeframe=timeframe)

    # Verify required columns
    required_cols = [
        "baseline", "pb_ema_top", "pb_ema_bot",
        "at_buyers_dominant", "at_sellers_dominant", "at_is_flat",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns: {missing}")

    if verbose:
        print(f"  Ready: {len(df)} candles with indicators")

    return df


def print_pattern_analysis(analyzer: ATScenarioAnalyzer):
    """Print AT state pattern analysis."""
    patterns_df = analyzer.get_at_state_patterns()

    if patterns_df.empty:
        print("\nNo pattern analysis available")
        return

    print(f"\n{'='*60}")
    print("AT STATE PATTERN ANALYSIS")
    print(f"{'='*60}")
    print(f"{'Pattern':<30} {'Count':>8} {'Win%':>10} {'Avg PnL':>12}")
    print("-" * 60)

    for _, row in patterns_df.iterrows():
        print(f"{row['pattern']:<30} {row['count']:>8} "
              f"{row['win_rate']*100:>9.1f}% ${row['avg_pnl']:>11.2f}")

    print(f"{'='*60}")


def print_recommendations(analyzer: ATScenarioAnalyzer):
    """Print recommendations based on analysis."""
    summary = analyzer.get_summary()
    if not summary:
        return

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")

    # Find best scenario
    best_scenario = max(summary.items(), key=lambda x: x[1]["total_pnl"])
    best_name = best_scenario[0]
    best_stats = best_scenario[1]

    # Find scenario with best win rate (min 20 trades)
    valid_for_winrate = [(n, s) for n, s in summary.items() if s["n_allowed"] >= 20]
    if valid_for_winrate:
        best_winrate = max(valid_for_winrate, key=lambda x: x[1]["win_rate"])
    else:
        best_winrate = best_scenario

    print(f"\n1. BEST OVERALL (PnL): {best_name}")
    print(f"   - Trades: {best_stats['n_allowed']}")
    print(f"   - PnL: ${best_stats['total_pnl']:.2f}")
    print(f"   - Win Rate: {best_stats['win_rate']*100:.1f}%")
    print(f"   - E[R]: {best_stats['avg_r']:.3f}")

    if best_winrate[0] != best_name:
        print(f"\n2. BEST WIN RATE: {best_winrate[0]}")
        print(f"   - Trades: {best_winrate[1]['n_allowed']}")
        print(f"   - Win Rate: {best_winrate[1]['win_rate']*100:.1f}%")
        print(f"   - PnL: ${best_winrate[1]['total_pnl']:.2f}")

    # Compare AT off vs best AT
    if "at_off" in summary and best_name != "at_off":
        at_off_stats = summary["at_off"]
        print(f"\n3. AT IMPACT (vs AT OFF):")
        print(f"   - AT Off Trades: {at_off_stats['n_allowed']}, AT On: {best_stats['n_allowed']}")
        print(f"   - AT Off PnL: ${at_off_stats['total_pnl']:.2f}, AT On: ${best_stats['total_pnl']:.2f}")
        pnl_improvement = best_stats['total_pnl'] - at_off_stats['total_pnl']
        print(f"   - AT Value Add: ${pnl_improvement:.2f}")

    # Pattern-based recommendations
    patterns_df = analyzer.get_at_state_patterns()
    if not patterns_df.empty:
        print(f"\n4. PATTERN INSIGHTS:")

        # AT alignment impact
        aligned = patterns_df[patterns_df['pattern'] == 'AT SSL Aligned'].iloc[0]
        not_aligned = patterns_df[patterns_df['pattern'] == 'AT SSL Not Aligned'].iloc[0]

        if aligned['count'] > 5 and not_aligned['count'] > 5:
            if aligned['win_rate'] > not_aligned['win_rate'] + 0.1:
                print(f"   - AT ALIGNMENT MATTERS: Aligned win rate {aligned['win_rate']*100:.1f}% vs Not Aligned {not_aligned['win_rate']*100:.1f}%")
                print(f"     -> USE AT as filter (binary or score mode)")
            else:
                print(f"   - AT ALIGNMENT WEAK: Aligned {aligned['win_rate']*100:.1f}% vs Not {not_aligned['win_rate']*100:.1f}%")
                print(f"     -> Consider AT with grace period or score mode")

        # AT flat impact
        flat = patterns_df[patterns_df['pattern'] == 'AT Flat'].iloc[0]
        not_flat = patterns_df[patterns_df['pattern'] == 'AT Not Flat'].iloc[0]

        if flat['count'] > 5 and not_flat['count'] > 5:
            if flat['win_rate'] < not_flat['win_rate'] - 0.1:
                print(f"   - AT FLAT FILTER HELPS: Flat win rate {flat['win_rate']*100:.1f}% vs Not Flat {not_flat['win_rate']*100:.1f}%")
                print(f"     -> Keep AT flat filter ON (skip_at_flat_filter=False)")
            else:
                print(f"   - AT FLAT FILTER WEAK: Flat {flat['win_rate']*100:.1f}% vs Not {not_flat['win_rate']*100:.1f}%")
                print(f"     -> Can skip flat filter for more trades")

    print(f"\n{'='*60}")
    print("SUGGESTED CONFIG:")
    print(f"{'='*60}")

    # Map scenario name to config
    config_suggestions = {
        "at_off": 'at_mode = "off"',
        "at_binary_strict": 'at_mode = "binary", skip_at_flat_filter = False',
        "at_binary_lax": 'at_mode = "binary", skip_at_flat_filter = True',
        "at_binary_grace_2": 'at_mode = "binary", use_ssl_flip_grace = True, ssl_flip_grace_bars = 2',
        "at_binary_grace_3": 'at_mode = "binary", use_ssl_flip_grace = True, ssl_flip_grace_bars = 3',
        "at_binary_grace_5": 'at_mode = "binary", use_ssl_flip_grace = True, ssl_flip_grace_bars = 5',
        "at_score_1.0": 'at_mode = "score", at_score_threshold = 1.0',
        "at_score_1.5": 'at_mode = "score", at_score_threshold = 1.5',
        "at_score_2.0": 'at_mode = "score", at_score_threshold = 2.0',
    }

    config = config_suggestions.get(best_name, f"# See {best_name}")
    print(f"\n{config}\n")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="AT Scenario Analysis")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--timeframe", default="15m", help="Candlestick interval")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--min-rr", type=float, default=1.2, help="Minimum R:R for signals")
    parser.add_argument("--min-pbema-dist", type=float, default=0.002, help="Minimum PBEMA distance")
    parser.add_argument("--baseline-touch", action="store_true", help="Require baseline touch")
    parser.add_argument("--baseline-lookback", type=int, default=5, help="Baseline touch lookback bars")
    parser.add_argument("--export", type=str, default=None, help="Export results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    print("=" * 60)
    print("AT SCENARIO ANALYSIS")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Days: {args.days}")
    print(f"Min RR: {args.min_rr}")
    print(f"Min PBEMA Distance: {args.min_pbema_dist}")
    print(f"Baseline Touch Required: {args.baseline_touch}")
    if args.baseline_touch:
        print(f"Baseline Touch Lookback: {args.baseline_lookback} bars")
    print("=" * 60)

    # Fetch data
    df = fetch_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        verbose=not args.quiet,
    )

    if df.empty:
        print("ERROR: No data fetched")
        return

    # Run analysis
    analyzer = ATScenarioAnalyzer(
        min_rr=args.min_rr,
        min_pbema_distance=args.min_pbema_dist,
        require_baseline_touch=args.baseline_touch,
        baseline_touch_lookback=args.baseline_lookback,
    )

    results_df = analyzer.run_analysis(
        df,
        symbol=args.symbol,
        timeframe=args.timeframe,
        verbose=not args.quiet,
    )

    # Print pattern analysis
    if not args.quiet:
        print_pattern_analysis(analyzer)

    # Print recommendations
    print_recommendations(analyzer)

    # Export if requested
    if args.export:
        analyzer.export_results(args.export)

    # Print summary for easy reference
    print("\n" + "=" * 60)
    print("QUICK REFERENCE")
    print("=" * 60)
    summary = analyzer.get_summary()
    if summary:
        best = max(summary.items(), key=lambda x: x[1]["total_pnl"])
        print(f"Best Scenario: {best[0]}")
        print(f"PnL: ${best[1]['total_pnl']:.2f}")
        print(f"Trades: {best[1]['n_allowed']}")
        print(f"Win Rate: {best[1]['win_rate']*100:.1f}%")
        print(f"E[R]: {best[1]['avg_r']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
