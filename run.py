#!/usr/bin/env python3
"""
Integrated Test Runner

Tek komutla tum testleri calistir ve sonuclari tek yerde topla.

Usage:
    python run.py test BTCUSDT 15m              # Quick test (fixed config)
    python run.py test BTCUSDT 15m --full       # Full pipeline (discovery + WF + portfolio)
    python run.py test BTCUSDT 15m --quick      # 90-day quick test
    python run.py viz BTCUSDT 15m               # Visualize latest trades
    python run.py report                        # Summary of all tests
"""

import sys
import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Constants
RESULTS_DIR = Path("data/results")

# All available filters for discovery
ALL_FILTERS = [
    # === Original Filters ===
    "at_flat_filter", "adx_filter", "at_binary",
    "ssl_touch", "rsi_filter", "pbema_distance",
    "overlap_check", "body_position", "wick_rejection",
    "min_sl_filter",

    # === Pattern Filters (Real Trade Analysis 2026-01-04) ===
    "momentum_exit",        # Pattern 1: Exit on momentum exhaustion
    "pbema_retest",         # Pattern 2: PBEMA retest strategy
    "liquidity_grab",       # Pattern 3: Liquidity grab detection
    "ssl_slope_filter",     # Pattern 4: SSL baseline slope filter
    "htf_bounce",           # Pattern 5: HTF bounce detection
    "momentum_loss",        # Pattern 6: Momentum loss after trend
    "ssl_dynamic_support",  # Pattern 7: SSL dynamic support

    # === NEW: Professional Analysis Filters (2026-01-04) ===
    "ssl_slope_direction",      # Priority 1A: Directional slope filter (+$60-75/yr)
    "ssl_stability",            # Priority 1C: SSL stability check (+$25-35/yr)
    "quick_failure_predictor",  # Combined predictor (targets LONG quick failures)
]

# Default config (for quick tests)
# Updated 2026-01-04: Added quick_failure_predictor based on Professional Analysis
# - PnL increased from $72.99 to $91.12 (+24.8%)
# - WR increased from 46.2% to 60.0% (+13.8%)
# - Max DD decreased from $35.70 to $24.15 (-32.3%)
DEFAULT_FILTERS = ["regime", "at_flat_filter", "min_sl_filter", "quick_failure_predictor"]


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_result_dir(symbol: str, timeframe: str, mode: str = "") -> Path:
    """Get or create result directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{mode}" if mode else ""
    result_dir = RESULTS_DIR / f"{symbol}_{timeframe}_{timestamp}{suffix}"
    ensure_dir(result_dir)
    return result_dir


def get_latest_result(symbol: str, timeframe: str) -> Path:
    """Get latest result directory."""
    pattern = f"{symbol}_{timeframe}_*"
    matches = sorted(RESULTS_DIR.glob(pattern), reverse=True)
    return matches[0] if matches else None


def fetch_data(symbol: str, timeframe: str, days: int = 365):
    """Fetch and prepare data with indicators."""
    from core import get_client, calculate_indicators, set_backtest_mode
    import pandas as pd
    import requests

    set_backtest_mode(True)
    client = get_client()

    candles_map = {"5m": 288, "15m": 96, "1h": 24, "4h": 6}
    candles = min(days * candles_map.get(timeframe, 96), 35000)

    all_dfs = []
    remaining = candles
    end_time = None

    while remaining > 0:
        chunk = min(remaining, 1000)

        if end_time:
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

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    df = calculate_indicators(df, timeframe=timeframe)

    return df


def fetch_data_for_year(symbol: str, timeframe: str, year: int):
    """
    Fetch data for a specific year (OOS validation).

    Expert recommendation (Clenow): Test strategy on previous year data
    to detect overfitting.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        timeframe: Candle interval (e.g., 15m)
        year: Year to fetch (e.g., 2024)

    Returns:
        DataFrame with OHLCV and indicators for the specified year
    """
    from core import get_client, calculate_indicators, set_backtest_mode
    import pandas as pd
    import requests

    set_backtest_mode(True)
    client = get_client()

    # Calculate start and end timestamps for the year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23, 59, 59)
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    candles_map = {"5m": 288, "15m": 96, "1h": 24, "4h": 6}
    candles_per_day = candles_map.get(timeframe, 96)
    max_candles = 365 * candles_per_day

    all_dfs = []
    current_end = end_ts
    fetched_count = 0

    print(f"      Fetching {year} data ({max_candles} candles expected)...")

    while current_end > start_ts and fetched_count < max_candles:
        url = f"{client.BASE_URL}/klines"
        # Don't use startTime - just endTime and iterate backwards
        params = {
            'symbol': symbol,
            'interval': timeframe,
            'limit': 1000,
            'endTime': current_end
        }

        try:
            res = requests.get(url, params=params, timeout=30)
            if res.status_code != 200 or not res.json():
                break

            data = res.json()
            if not data:
                break

            df_c = pd.DataFrame(data).iloc[:, :6]
            df_c.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_c['timestamp'] = pd.to_datetime(df_c['timestamp'], unit='ms', utc=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_c[col] = pd.to_numeric(df_c[col], errors='coerce')

            all_dfs.insert(0, df_c)
            fetched_count += len(df_c)

            # Move to earlier data
            earliest_ts = int(df_c['timestamp'].iloc[0].timestamp() * 1000)
            if earliest_ts <= start_ts:
                break
            current_end = earliest_ts - 1

            # Progress indicator
            if len(all_dfs) % 10 == 0:
                print(f"      ...fetched {fetched_count} candles")

        except Exception as e:
            print(f"      Error fetching data: {e}")
            break

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

    # Filter to exact year
    df = df[(df['timestamp'] >= pd.Timestamp(start_date, tz='UTC')) &
            (df['timestamp'] <= pd.Timestamp(end_date, tz='UTC'))]

    print(f"      Filtered to {len(df)} candles for {year}")

    df.set_index('timestamp', inplace=True)
    df = calculate_indicators(df, timeframe=timeframe)

    return df


def run_oos_validation(symbol: str, timeframe: str, filter_list: List[str], year: int = 2024) -> Dict:
    """
    Run Out-of-Sample validation on historical data.

    Expert recommendation (Clenow): Test config found on recent data
    against older data to detect overfitting.

    Args:
        symbol: Trading pair
        timeframe: Candle interval
        filter_list: Filters to test (found on in-sample data)
        year: Year to test on (default 2024)

    Returns:
        Dict with OOS validation results
    """
    df_oos = fetch_data_for_year(symbol, timeframe, year)

    if df_oos.empty or len(df_oos) < 100:
        return {
            "year": year,
            "error": f"Insufficient data for {year}",
            "trades": 0,
            "pnl": 0,
            "passed": False
        }

    # Run backtest on OOS data
    filter_flags = make_filter_flags(filter_list)
    result = run_backtest(df_oos, filter_flags)

    # Also run trade-based WF on OOS
    wf_result = run_trade_based_wf(df_oos, filter_flags, trades_per_window=5)

    passed = result["pnl"] > 0 and result["trades"] >= 5

    return {
        "year": year,
        "candles": len(df_oos),
        "period": f"{df_oos.index[0].date()} to {df_oos.index[-1].date()}",
        "trades": result["trades"],
        "wins": result["wins"],
        "wr": result["wr"],
        "pnl": result["pnl"],
        "wf_windows": wf_result.get("windows", 0),
        "wf_positive": wf_result.get("positive_windows", 0),
        "wf_wr": wf_result.get("window_wr", 0),
        "passed": passed
    }


def make_filter_flags(filter_list: List[str]) -> Dict:
    """Convert filter list to flags dict."""
    return {
        # === Original Filters ===
        "use_regime_filter": "regime" in filter_list,
        "use_at_flat_filter": "at_flat_filter" in filter_list,
        "use_min_sl_filter": "min_sl_filter" in filter_list,
        "use_at_binary": "at_binary" in filter_list,
        "use_adx_filter": "adx_filter" in filter_list,
        "use_ssl_touch": "ssl_touch" in filter_list,
        "use_rsi_filter": "rsi_filter" in filter_list,
        "use_pbema_distance": "pbema_distance" in filter_list,
        "use_overlap_check": "overlap_check" in filter_list,
        "use_body_position": "body_position" in filter_list,
        "use_wick_rejection": "wick_rejection" in filter_list,

        # === Pattern Filters (Real Trade Analysis 2026-01-04) ===
        "use_momentum_exit": "momentum_exit" in filter_list,
        "use_pbema_retest": "pbema_retest" in filter_list,
        "use_liquidity_grab": "liquidity_grab" in filter_list,
        "use_ssl_slope_filter": "ssl_slope_filter" in filter_list,
        "use_htf_bounce": "htf_bounce" in filter_list,
        "use_momentum_loss": "momentum_loss" in filter_list,
        "use_ssl_dynamic_support": "ssl_dynamic_support" in filter_list,

        # === NEW: Professional Analysis Filters (2026-01-04) ===
        "use_ssl_slope_direction": "ssl_slope_direction" in filter_list,
        "use_ssl_stability": "ssl_stability" in filter_list,
        "use_quick_failure_predictor": "quick_failure_predictor" in filter_list,
    }


def run_backtest(df, filter_flags: Dict, min_bars: int = 5, position_size: float = 100.0) -> Dict:
    """
    Run simple backtest with given filters.

    Uses fixed position size and calculates dollar PnL properly.
    Includes slippage and fees for realistic comparison.
    """
    from core.at_scenario_analyzer import check_core_signal
    from runners.run_filter_combo_test import apply_filters

    trades = []
    last_idx = -min_bars

    slippage = 0.0005  # 0.05%
    fee = 0.0007  # 0.07%

    for i in range(60, len(df) - 10):
        if i - last_idx < min_bars:
            continue

        signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)
        if signal_type is None:
            continue

        last_idx = i

        passed, _ = apply_filters(
            df=df, index=i, signal_type=signal_type,
            entry_price=entry, sl_price=sl, **filter_flags
        )

        if not passed:
            continue

        # Apply entry slippage
        if signal_type == "LONG":
            actual_entry = entry * (1 + slippage)
        else:
            actual_entry = entry * (1 - slippage)

        # Simulate trade
        for j in range(i + 1, min(i + 200, len(df))):
            candle = df.iloc[j]
            high, low = float(candle["high"]), float(candle["low"])

            if signal_type == "LONG":
                if low <= sl:
                    exit_price = sl * (1 - slippage)
                    pnl = (exit_price - actual_entry) / actual_entry * position_size
                    pnl -= position_size * fee * 2
                    trades.append({"pnl": pnl, "win": False})
                    break
                if high >= tp:
                    exit_price = tp * (1 - slippage)
                    pnl = (exit_price - actual_entry) / actual_entry * position_size
                    pnl -= position_size * fee * 2
                    trades.append({"pnl": pnl, "win": True})
                    break
            else:
                if high >= sl:
                    exit_price = sl * (1 + slippage)
                    pnl = (actual_entry - exit_price) / actual_entry * position_size
                    pnl -= position_size * fee * 2
                    trades.append({"pnl": pnl, "win": False})
                    break
                if low <= tp:
                    exit_price = tp * (1 + slippage)
                    pnl = (actual_entry - exit_price) / actual_entry * position_size
                    pnl -= position_size * fee * 2
                    trades.append({"pnl": pnl, "win": True})
                    break

    if not trades:
        return {"trades": 0, "wins": 0, "wr": 0, "pnl": 0}

    wins = sum(1 for t in trades if t["win"])
    pnl = sum(t["pnl"] for t in trades)

    return {
        "trades": len(trades),
        "wins": wins,
        "wr": wins / len(trades) * 100,
        "pnl": pnl,
    }


def run_rolling_wf(df, filter_flags: Dict, forward_days: int = 7) -> Dict:
    """Run rolling walk-forward validation (TIME-BASED - legacy)."""
    start_date = df.index[0]
    end_date = df.index[-1]

    all_results = []
    current = start_date

    while current < end_date:
        window_end = current + timedelta(days=forward_days)
        mask = (df.index >= current) & (df.index < window_end)
        df_window = df[mask]

        if len(df_window) >= 20:
            result = run_backtest(df_window, filter_flags)
            all_results.append(result)

        current = window_end

    if not all_results:
        return {"windows": 0, "trades": 0, "pnl": 0, "positive_windows": 0}

    total_trades = sum(r["trades"] for r in all_results)
    total_pnl = sum(r["pnl"] for r in all_results)
    positive = sum(1 for r in all_results if r["pnl"] > 0)

    return {
        "windows": len(all_results),
        "trades": total_trades,
        "pnl": total_pnl,
        "positive_windows": positive,
        "window_wr": positive / len(all_results) * 100 if all_results else 0,
    }


def run_trade_based_wf(df, filter_flags: Dict, trades_per_window: int = 5) -> Dict:
    """
    Run TRADE-BASED walk-forward validation.

    Expert recommendation: Use trade count instead of time windows.
    This gives meaningful statistics for low-frequency strategies.

    Args:
        df: DataFrame with OHLCV and indicators
        filter_flags: Filter configuration
        trades_per_window: Minimum trades per window (default 5)

    Returns:
        Dict with window statistics
    """
    from core.at_scenario_analyzer import check_core_signal
    from runners.run_filter_combo_test import apply_filters

    # First, collect ALL trades with their PnL
    all_trades = []
    last_idx = -5

    slippage = 0.0005
    fee = 0.0007
    position_size = 100.0

    for i in range(60, len(df) - 10):
        if i - last_idx < 5:
            continue

        signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)
        if signal_type is None:
            continue

        last_idx = i

        passed, _ = apply_filters(
            df=df, index=i, signal_type=signal_type,
            entry_price=entry, sl_price=sl, **filter_flags
        )

        if not passed:
            continue

        # Simulate trade
        trade_pnl = None
        if signal_type == "LONG":
            actual_entry = entry * (1 + slippage)
        else:
            actual_entry = entry * (1 - slippage)

        for j in range(i + 1, min(i + 200, len(df))):
            candle = df.iloc[j]
            high, low = float(candle["high"]), float(candle["low"])

            if signal_type == "LONG":
                if low <= sl:
                    exit_price = sl * (1 - slippage)
                    trade_pnl = (exit_price - actual_entry) / actual_entry * position_size
                    trade_pnl -= position_size * fee * 2
                    break
                if high >= tp:
                    exit_price = tp * (1 - slippage)
                    trade_pnl = (exit_price - actual_entry) / actual_entry * position_size
                    trade_pnl -= position_size * fee * 2
                    break
            else:  # SHORT
                if high >= sl:
                    exit_price = sl * (1 + slippage)
                    trade_pnl = (actual_entry - exit_price) / actual_entry * position_size
                    trade_pnl -= position_size * fee * 2
                    break
                if low <= tp:
                    exit_price = tp * (1 + slippage)
                    trade_pnl = (actual_entry - exit_price) / actual_entry * position_size
                    trade_pnl -= position_size * fee * 2
                    break

        if trade_pnl is not None:
            all_trades.append({"idx": i, "pnl": trade_pnl})

    # Not enough trades for WF
    if len(all_trades) < trades_per_window * 2:
        return {
            "windows": 0,
            "trades": len(all_trades),
            "pnl": sum(t["pnl"] for t in all_trades),
            "positive_windows": 0,
            "window_wr": 0,
            "method": "trade_based",
            "error": f"Not enough trades ({len(all_trades)}) for {trades_per_window}-trade windows"
        }

    # Split into trade-based windows
    windows = []
    for i in range(0, len(all_trades), trades_per_window):
        window_trades = all_trades[i:i + trades_per_window]
        if len(window_trades) >= trades_per_window:  # Full window only
            window_pnl = sum(t["pnl"] for t in window_trades)
            windows.append({
                "trades": len(window_trades),
                "pnl": window_pnl,
                "positive": window_pnl > 0
            })

    if not windows:
        return {
            "windows": 0,
            "trades": len(all_trades),
            "pnl": sum(t["pnl"] for t in all_trades),
            "positive_windows": 0,
            "window_wr": 0,
            "method": "trade_based"
        }

    positive_windows = sum(1 for w in windows if w["positive"])

    return {
        "windows": len(windows),
        "trades": len(all_trades),
        "pnl": sum(t["pnl"] for t in all_trades),
        "positive_windows": positive_windows,
        "window_wr": positive_windows / len(windows) * 100,
        "method": "trade_based",
        "trades_per_window": trades_per_window,
    }


def run_purged_cv(df, filter_list: List[str], n_splits: int = 5, purge_bars: int = 100) -> Dict:
    """
    Run PURGED Cross-Validation for filter selection.

    Expert recommendation (Lopez de Prado): Prevent overfitting by testing
    each filter combo across multiple non-overlapping folds with purge gaps.

    Args:
        df: DataFrame with OHLCV and indicators
        filter_list: List of filters to test
        n_splits: Number of CV folds (default 5)
        purge_bars: Gap between train/test to prevent data leakage

    Returns:
        Dict with CV results and robustness score
    """
    filter_flags = make_filter_flags(filter_list)
    fold_size = len(df) // n_splits
    fold_results = []

    for fold in range(n_splits):
        # Define test fold boundaries
        test_start = fold * fold_size
        test_end = test_start + fold_size

        # Apply purge gap (prevent data leakage)
        purge_start = max(0, test_start - purge_bars)
        purge_end = min(len(df), test_end + purge_bars)

        # Get test data only (we don't train, just validate)
        test_df = df.iloc[test_start:test_end].copy()

        if len(test_df) < 100:  # Minimum data for valid test
            fold_results.append({"pnl": 0, "trades": 0, "valid": False})
            continue

        # Run backtest on this fold
        result = run_backtest(test_df, filter_flags)
        fold_results.append({
            "pnl": result["pnl"],
            "trades": result["trades"],
            "wr": result["wr"],
            "valid": result["trades"] >= 3,  # Need at least 3 trades
            "positive": result["pnl"] > 0
        })

    # Calculate robustness metrics
    valid_folds = [f for f in fold_results if f["valid"]]
    if not valid_folds:
        return {
            "n_splits": n_splits,
            "valid_folds": 0,
            "positive_folds": 0,
            "fold_wr": 0,
            "robust": False,
            "total_pnl": 0,
            "avg_pnl": 0,
            "fold_details": fold_results
        }

    positive_folds = sum(1 for f in valid_folds if f["positive"])
    total_pnl = sum(f["pnl"] for f in valid_folds)

    return {
        "n_splits": n_splits,
        "valid_folds": len(valid_folds),
        "positive_folds": positive_folds,
        "fold_wr": positive_folds / len(valid_folds) * 100,
        "robust": positive_folds >= len(valid_folds) - 1,  # 4/5 or better
        "all_positive": positive_folds == len(valid_folds),
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(valid_folds),
        "fold_details": fold_results
    }


def calculate_r_multiples(trades: List[Dict], risk_per_trade: float = 10.0) -> Dict:
    """
    Calculate R-multiple statistics for trades.

    Expert recommendation (Van Tharp): Use risk units instead of dollars.
    R-multiple = PnL / Risk amount

    Args:
        trades: List of trade dicts with 'pnl' key
        risk_per_trade: Risk amount per trade in dollars

    Returns:
        Dict with R-multiple statistics
    """
    import numpy as np

    if not trades:
        return {
            "avg_r": 0,
            "avg_win_r": 0,
            "avg_loss_r": 0,
            "expectancy_r": 0,
            "max_win_r": 0,
            "max_loss_r": 0,
            "total_r": 0,
        }

    r_multiples = [t["pnl"] / risk_per_trade for t in trades]
    wins = [r for r in r_multiples if r > 0]
    losses = [r for r in r_multiples if r < 0]

    return {
        "avg_r": float(np.mean(r_multiples)),
        "avg_win_r": float(np.mean(wins)) if wins else 0,
        "avg_loss_r": float(np.mean(losses)) if losses else 0,
        "expectancy_r": float(np.mean(r_multiples)),
        "max_win_r": float(max(r_multiples)) if r_multiples else 0,
        "max_loss_r": float(min(r_multiples)) if r_multiples else 0,
        "total_r": float(sum(r_multiples)),
        "r_list": r_multiples,
    }


def calculate_kelly(win_rate: float, avg_win: float, avg_loss: float) -> Dict:
    """
    Calculate Kelly Criterion for optimal position sizing.

    Expert recommendation (Ralph Vince): Use Kelly for position sizing,
    but apply half-Kelly for safety.

    Args:
        win_rate: Win rate as decimal (0.46 for 46%)
        avg_win: Average winning trade in dollars
        avg_loss: Average losing trade in dollars (positive number)

    Returns:
        Dict with Kelly percentages
    """
    if avg_loss <= 0 or avg_win <= 0:
        return {"kelly_pct": 0, "half_kelly": 0, "quarter_kelly": 0}

    payoff_ratio = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / payoff_ratio

    return {
        "kelly_pct": max(0, kelly * 100),  # Full Kelly %
        "half_kelly": max(0, kelly * 50),  # Half Kelly (recommended)
        "quarter_kelly": max(0, kelly * 25),  # Quarter Kelly (conservative)
        "payoff_ratio": payoff_ratio,
        "edge": kelly,  # Positive = edge exists
    }


def run_portfolio_backtest(df, filter_list: List[str]) -> Dict:
    """Run portfolio backtest with realistic sizing."""
    from core.at_scenario_analyzer import check_core_signal
    from core.simple_portfolio import PortfolioConfig, run_portfolio_backtest as portfolio_bt
    from runners.run_filter_combo_test import apply_filters

    filter_flags = make_filter_flags(filter_list)

    # Collect signals
    signals = []
    last_idx = -5

    for i in range(60, len(df) - 10):
        if i - last_idx < 5:
            continue

        signal_type, entry, tp, sl, reason = check_core_signal(df, index=i)
        if signal_type is None:
            continue

        last_idx = i

        passed, _ = apply_filters(
            df=df, index=i, signal_type=signal_type,
            entry_price=entry, sl_price=sl, **filter_flags
        )

        if not passed:
            continue

        signals.append({
            "idx": i,
            "signal_type": signal_type,
            "entry": entry,
            "tp": tp,
            "sl": sl,
        })

    # Portfolio config
    config = PortfolioConfig(
        initial_balance=1000.0,
        risk_per_trade_pct=0.01,
        leverage=10,
        max_position_pct=0.10,
        slippage_pct=0.0005,
        fee_pct=0.0007,
        total_dd_limit=0.25,
    )

    return portfolio_bt(df, signals, config)


def run_pbema_retest_backtest(df, use_momentum_exit: bool = False) -> Dict:
    """
    Run PBEMA Retest strategy backtest.

    PBEMA Retest is a separate strategy (not a filter) that trades
    PBEMA cloud as support/resistance after breakout.

    Args:
        df: DataFrame with indicators
        use_momentum_exit: If True, exit when momentum exhausts

    Returns:
        dict: Backtest results
    """
    from strategies.pbema_retest import check_pbema_retest_signal
    from runners.run_filter_combo_test import simulate_trade

    signals = []
    trades = []
    exit_types = {"TP": 0, "SL": 0, "MOMENTUM": 0, "EOD": 0}

    # Collect PBEMA Retest signals
    for i in range(200, len(df) - 10):
        result = check_pbema_retest_signal(df, index=i)
        signal_type, entry, tp, sl, reason = result[:5]

        if signal_type:
            signals.append({
                'idx': i,
                'type': signal_type,
                'entry': entry,
                'tp': tp,
                'sl': sl,
            })

    # Simulate trades (no overlapping)
    last_exit = -1
    for sig in signals:
        if sig['idx'] <= last_exit:
            continue

        trade = simulate_trade(
            df, sig['idx'], sig['type'],
            sig['entry'], sig['tp'], sig['sl'],
            position_size=35.0,
            use_momentum_exit=use_momentum_exit
        )
        if trade:
            trades.append(trade)
            last_exit = trade.get('exit_idx', sig['idx'] + 20)
            exit_types[trade.get('exit_type', 'EOD')] += 1

    # Calculate stats
    if not trades:
        return {
            'signals': len(signals),
            'trades': 0,
            'wins': 0,
            'wr': 0,
            'pnl': 0,
            'exit_types': exit_types,
            'momentum_exit': use_momentum_exit,
        }

    wins = sum(1 for t in trades if t['win'])
    pnl = sum(t['pnl'] for t in trades)

    return {
        'signals': len(signals),
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'wr': 100 * wins / len(trades),
        'pnl': pnl,
        'exit_types': exit_types,
        'momentum_exit': use_momentum_exit,
    }


def run_full_pipeline(symbol: str, timeframe: str, days: int = 365) -> Dict:
    """
    Run ENHANCED full integrated pipeline with expert recommendations:
    1. Fetch data
    2. Baseline (regime only)
    3. PBEMA Retest strategy test (separate strategy)
    4. Filter discovery (incremental) + Purged CV validation
    5. Trade-based WF validation (not time-based)
    6. Portfolio backtest with R-multiple tracking
    7. Kelly sizing calculation
    8. OOS validation
    9. Final recommendation with strategy comparison
    """
    from core import set_backtest_mode
    set_backtest_mode(True)

    result_dir = get_result_dir(symbol, timeframe, "full_v3")

    print(f"\n{'='*70}")
    print(f"ENHANCED PIPELINE v3: {symbol} {timeframe} ({days} days)")
    print(f"{'='*70}")
    print(f"Output: {result_dir}")
    print(f"{'='*70}\n")

    # ===== STEP 1: Fetch Data =====
    print("[1/10] Fetching data...")
    df = fetch_data(symbol, timeframe, days)
    print(f"      {len(df)} candles: {df.index[0].date()} to {df.index[-1].date()}")

    # ===== STEP 2: Baseline & Default Config =====
    print("\n[2/10] SSL Flow Baseline Tests...")

    # Baseline (regime only)
    baseline_flags = make_filter_flags(["regime"])
    baseline = run_backtest(df, baseline_flags)
    print(f"      Baseline (regime):     {baseline['trades']:>4} trades | {baseline['wr']:.1f}% WR | ${baseline['pnl']:.2f} PnL")

    # Default Config (regime + at_flat_filter + min_sl_filter)
    default_flags = make_filter_flags(DEFAULT_FILTERS)
    default_result = run_backtest(df, default_flags)
    print(f"      Default Config:        {default_result['trades']:>4} trades | {default_result['wr']:.1f}% WR | ${default_result['pnl']:.2f} PnL")

    # ===== STEP 3: PBEMA Retest Strategy =====
    print("\n[3/10] PBEMA Retest Strategy (separate from SSL Flow)...")
    pbema_result = run_pbema_retest_backtest(df, use_momentum_exit=False)
    pbema_mom_result = run_pbema_retest_backtest(df, use_momentum_exit=True)

    print(f"      PBEMA Retest:          {pbema_result['trades']:>4} trades | {pbema_result['wr']:.1f}% WR | ${pbema_result['pnl']:.2f} PnL")
    print(f"      PBEMA + Momentum Exit: {pbema_mom_result['trades']:>4} trades | {pbema_mom_result['wr']:.1f}% WR | ${pbema_mom_result['pnl']:.2f} PnL")

    # ===== STEP 4: Filter Discovery =====
    print("\n[4/10] Filter discovery (incremental)...")
    discovery_results = []

    # Include default config in discovery
    default_result["filters"] = DEFAULT_FILTERS
    default_result["name"] = "Default Config"
    discovery_results.append(default_result)
    print(f"      {'Default Config':<35} → {default_result['trades']:>4} trades, {default_result['pnl']:>8.2f} PnL (baseline)")

    # Test regime + each filter
    for filter_name in ALL_FILTERS:
        test_filters = ["regime", filter_name]
        test_flags = make_filter_flags(test_filters)
        result = run_backtest(df, test_flags)
        result["filters"] = test_filters
        result["name"] = f"regime + {filter_name}"
        discovery_results.append(result)
        print(f"      {result['name']:<35} → {result['trades']:>4} trades, {result['pnl']:>8.2f} PnL")

    # Test top combinations (regime + filter1 + filter2)
    print("\n      Testing 2-filter combos...")
    top_singles = sorted(discovery_results, key=lambda x: x["pnl"], reverse=True)[:3]

    for i, r1 in enumerate(top_singles):
        for r2 in top_singles[i+1:]:
            f1 = r1["filters"][1]
            f2 = r2["filters"][1]
            test_filters = ["regime", f1, f2]
            test_flags = make_filter_flags(test_filters)
            result = run_backtest(df, test_flags)
            result["filters"] = test_filters
            result["name"] = f"regime + {f1} + {f2}"
            discovery_results.append(result)
            print(f"      {result['name']:<35} → {result['trades']:>4} trades, {result['pnl']:>8.2f} PnL")

    # Find best combo
    valid_results = [r for r in discovery_results if r["trades"] >= 10]
    if not valid_results:
        valid_results = discovery_results

    best_combo = max(valid_results, key=lambda x: x["pnl"])
    print(f"\n      BEST: {best_combo['name']} ({best_combo['trades']} trades, {best_combo['pnl']:.2f} PnL)")

    # ===== STEP 5: Purged CV Validation =====
    print("\n[5/10] Purged Cross-Validation (overfitting check)...")
    cv_result = run_purged_cv(df, best_combo["filters"], n_splits=5, purge_bars=100)
    print(f"      Folds: {cv_result['valid_folds']}/{cv_result['n_splits']} valid")
    print(f"      Positive: {cv_result['positive_folds']}/{cv_result['valid_folds']} ({cv_result['fold_wr']:.1f}%)")
    print(f"      Robust: {'YES' if cv_result['robust'] else 'NO (possible overfitting!)'}")

    # ===== STEP 6: Trade-Based WF Validation =====
    print("\n[6/10] Trade-Based Walk-Forward validation...")
    best_flags = make_filter_flags(best_combo["filters"])
    wf_result = run_trade_based_wf(df, best_flags, trades_per_window=5)

    if "error" in wf_result:
        print(f"      WARNING: {wf_result['error']}")
        wf_passed = False
    else:
        print(f"      {wf_result['windows']} windows (5 trades each) | {wf_result['trades']} total trades")
        print(f"      PnL: {wf_result['pnl']:.2f} | Positive: {wf_result['positive_windows']}/{wf_result['windows']} ({wf_result['window_wr']:.1f}%)")
        wf_passed = wf_result["pnl"] > 0 and wf_result["window_wr"] >= 50

    # ===== STEP 7: Portfolio Backtest =====
    print("\n[7/10] Portfolio backtest ($1000, 1% risk)...")
    portfolio = run_portfolio_backtest(df, best_combo["filters"])
    print(f"      Trades: {portfolio['trades']} | WR: {portfolio['win_rate']:.1f}%")
    print(f"      PnL: ${portfolio['total_pnl']:.2f} | DD: ${portfolio['max_drawdown']:.2f}")
    print(f"      PF: {portfolio['profit_factor']:.2f}")

    # ===== STEP 8: R-Multiple & Kelly Analysis =====
    print("\n[8/10] R-Multiple & Kelly Analysis...")

    # Get individual trade PnLs for R-multiple
    trades_list = portfolio.get('trades_list', [])
    trade_pnls = [{"pnl": t.get("pnl", 0)} for t in trades_list] if trades_list else []

    r_multiples = calculate_r_multiples(trade_pnls, risk_per_trade=10.0)
    print(f"      Avg Win: {r_multiples['avg_win_r']:.2f}R | Avg Loss: {r_multiples['avg_loss_r']:.2f}R")
    print(f"      Expectancy: {r_multiples['expectancy_r']:.2f}R per trade")

    # Calculate Kelly
    wins = [t for t in trade_pnls if t["pnl"] > 0]
    losses = [t for t in trade_pnls if t["pnl"] < 0]
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses)) if losses else 0
    win_rate = portfolio['win_rate'] / 100

    kelly = calculate_kelly(win_rate, avg_win, avg_loss)
    print(f"      Kelly: {kelly['kelly_pct']:.1f}% | Half-Kelly: {kelly['half_kelly']:.1f}% | Edge: {kelly['edge']:.3f}")

    # ===== STEP 9: OOS Validation (2024) =====
    print("\n[9/10] Out-of-Sample Validation (2024)...")
    oos_result = run_oos_validation(symbol, timeframe, best_combo["filters"], year=2024)

    if "error" in oos_result:
        print(f"      WARNING: {oos_result['error']}")
        oos_passed = False
    else:
        print(f"      Trades: {oos_result['trades']} | WR: {oos_result['wr']:.1f}%")
        print(f"      PnL: {oos_result['pnl']:.2f}")
        print(f"      WF Windows: {oos_result['wf_windows']} | Positive: {oos_result['wf_positive']} ({oos_result['wf_wr']:.1f}%)")
        oos_passed = oos_result['passed']

    # ===== STEP 10: Final Results =====
    print("\n[10/10] Saving results...")

    # Enhanced verdict logic with OOS
    cv_passed = cv_result['robust']

    # New verdict: requires OOS to pass for full PASS
    if portfolio['total_pnl'] > 0 and wf_passed and cv_passed and oos_passed and not portfolio['stopped']:
        verdict = "PASS"
        recommendation = "READY FOR PAPER TRADE"
    elif portfolio['total_pnl'] > 0 and wf_passed and oos_passed and not portfolio['stopped']:
        verdict = "GOOD"
        recommendation = "PAPER TRADE WITH CAUTION"
    elif portfolio['total_pnl'] > 0 and (wf_passed or cv_passed) and not portfolio['stopped']:
        verdict = "MARGINAL"
        recommendation = "NEEDS MORE VALIDATION"
    elif portfolio['total_pnl'] > 0 and not portfolio['stopped']:
        verdict = "WEAK"
        recommendation = "LIKELY OVERFITTING - DO NOT TRADE"
    else:
        verdict = "FAIL"
        recommendation = "DO NOT TRADE"

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "timestamp": datetime.now().isoformat(),
        "mode": "full_pipeline_v3",
        "baseline": baseline,
        "default_config": {
            "filters": DEFAULT_FILTERS,
            "trades": default_result['trades'],
            "wr": default_result['wr'],
            "pnl": default_result['pnl'],
        },
        "pbema_retest": {
            "standard": {
                "signals": pbema_result['signals'],
                "trades": pbema_result['trades'],
                "wr": pbema_result['wr'],
                "pnl": pbema_result['pnl'],
            },
            "with_momentum_exit": {
                "trades": pbema_mom_result['trades'],
                "wr": pbema_mom_result['wr'],
                "pnl": pbema_mom_result['pnl'],
                "exit_types": pbema_mom_result['exit_types'],
            },
        },
        "discovery": {
            "tested": len(discovery_results),
            "best_combo": best_combo,
            "top_5": sorted(discovery_results, key=lambda x: x["pnl"], reverse=True)[:5],
        },
        "purged_cv": {
            "n_splits": cv_result['n_splits'],
            "valid_folds": cv_result['valid_folds'],
            "positive_folds": cv_result['positive_folds'],
            "fold_wr": cv_result['fold_wr'],
            "robust": cv_result['robust'],
            "passed": cv_passed,
        },
        "trade_based_wf": {
            "windows": wf_result.get('windows', 0),
            "trades": wf_result.get('trades', 0),
            "pnl": wf_result.get('pnl', 0),
            "positive_windows": wf_result.get('positive_windows', 0),
            "window_wr": wf_result.get('window_wr', 0),
            "passed": wf_passed,
        },
        "oos_2024": {
            "year": oos_result.get('year', 2024),
            "trades": oos_result.get('trades', 0),
            "wins": oos_result.get('wins', 0),
            "wr": oos_result.get('wr', 0),
            "pnl": oos_result.get('pnl', 0),
            "wf_windows": oos_result.get('wf_windows', 0),
            "wf_positive": oos_result.get('wf_positive', 0),
            "passed": oos_passed,
        },
        "portfolio": {
            "filters": best_combo["filters"],
            "initial": 1000.0,
            "final": portfolio['final_balance'],
            "pnl": portfolio['total_pnl'],
            "pnl_pct": portfolio['total_pnl_pct'],
            "trades": portfolio['trades'],
            "wins": portfolio['wins'],
            "losses": portfolio['losses'],
            "win_rate": portfolio['win_rate'],
            "max_dd": portfolio['max_drawdown'],
            "max_dd_pct": portfolio['max_drawdown_pct'],
            "pf": portfolio['profit_factor'],
            "stopped": portfolio['stopped'],
        },
        "r_multiples": {
            "avg_r": r_multiples['avg_r'],
            "avg_win_r": r_multiples['avg_win_r'],
            "avg_loss_r": r_multiples['avg_loss_r'],
            "expectancy_r": r_multiples['expectancy_r'],
            "max_win_r": r_multiples['max_win_r'],
            "max_loss_r": r_multiples['max_loss_r'],
        },
        "kelly": {
            "kelly_pct": kelly['kelly_pct'],
            "half_kelly": kelly['half_kelly'],
            "quarter_kelly": kelly['quarter_kelly'],
            "edge": kelly['edge'],
        },
        "verdict": verdict,
        "recommendation": recommendation,
    }

    # Save files
    with open(result_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    with open(result_dir / "trades.json", "w") as f:
        json.dump(portfolio.get('trades_list', []), f, indent=2, default=str)

    # Print summary
    cv_status = "PASSED" if cv_passed else "FAILED"
    wf_status = "PASSED" if wf_passed else "FAILED"
    oos_status = "PASSED" if oos_passed else "FAILED"

    # OOS summary info
    oos_trades = oos_result.get('trades', 0)
    oos_pnl = oos_result.get('pnl', 0)
    oos_wr = oos_result.get('wr', 0)
    oos_year = oos_result.get('year', 2024)

    summary = f"""
{'='*70}
ENHANCED PIPELINE v3 RESULT: {symbol} {timeframe}
{'='*70}

Period: {df.index[0].date()} to {df.index[-1].date()} ({days} days)

STEP 1 - SSL FLOW BASELINES:
  Baseline (regime):  {baseline['trades']} trades | {baseline['wr']:.1f}% WR | ${baseline['pnl']:.2f}
  Default Config:     {default_result['trades']} trades | {default_result['wr']:.1f}% WR | ${default_result['pnl']:.2f}

STEP 2 - PBEMA RETEST STRATEGY (separate strategy):
  Standard:      {pbema_result['trades']} trades | {pbema_result['wr']:.1f}% WR | ${pbema_result['pnl']:.2f}
  + Momentum:    {pbema_mom_result['trades']} trades | {pbema_mom_result['wr']:.1f}% WR | ${pbema_mom_result['pnl']:.2f}

STEP 3 - FILTER DISCOVERY:
  Tested: {len(discovery_results)} combinations
  Best: {best_combo['name']}
        {best_combo['trades']} trades | {best_combo['wr']:.1f}% WR | {best_combo['pnl']:.2f} PnL

STEP 4 - PURGED CV (Overfitting Check):
  Folds: {cv_result['valid_folds']}/{cv_result['n_splits']} | Positive: {cv_result['positive_folds']}
  Robust: {cv_result['robust']} | Status: {cv_status}

STEP 5 - TRADE-BASED WALK-FORWARD:
  Windows: {wf_result.get('windows', 0)} (5 trades each)
  Positive: {wf_result.get('positive_windows', 0)} ({wf_result.get('window_wr', 0):.1f}%)
  Status: {wf_status}

STEP 6 - PORTFOLIO BACKTEST ($1000, 1% risk, 10x leverage):
  Best Config: {', '.join(best_combo['filters'])}
  Final Balance: ${portfolio['final_balance']:.2f}
  PnL: ${portfolio['total_pnl']:.2f} ({portfolio['total_pnl_pct']:.2f}%)
  Trades: {portfolio['trades']} (W:{portfolio['wins']} L:{portfolio['losses']})
  Win Rate: {portfolio['win_rate']:.1f}%
  Max DD: ${portfolio['max_drawdown']:.2f} ({portfolio['max_drawdown_pct']:.2f}%)
  Profit Factor: {portfolio['profit_factor']:.2f}

STEP 7 - R-MULTIPLE ANALYSIS:
  Avg Win: {r_multiples['avg_win_r']:.2f}R | Avg Loss: {r_multiples['avg_loss_r']:.2f}R
  Expectancy: {r_multiples['expectancy_r']:.2f}R per trade
  Max Win: {r_multiples['max_win_r']:.2f}R | Max Loss: {r_multiples['max_loss_r']:.2f}R

STEP 8 - KELLY CRITERION:
  Full Kelly: {kelly['kelly_pct']:.1f}%
  Half Kelly (recommended): {kelly['half_kelly']:.1f}%
  Edge: {kelly['edge']:.3f}

STEP 9 - OOS VALIDATION ({oos_year}):
  Trades: {oos_trades} | WR: {oos_wr:.1f}% | PnL: {oos_pnl:.2f}
  Status: {oos_status}

{'='*70}
VERDICT: {verdict}
RECOMMENDATION: {recommendation}
{'='*70}

Validation Summary:
  [{'PASS' if cv_passed else 'FAIL'}] Purged CV: {cv_status}
  [{'PASS' if wf_passed else 'FAIL'}] Trade-Based WF: {wf_status}
  [{'PASS' if oos_passed else 'FAIL'}] OOS {oos_year}: {oos_status}
  [{'PASS' if portfolio['total_pnl'] > 0 else 'FAIL'}] Profitable: {'YES' if portfolio['total_pnl'] > 0 else 'NO'}

Results saved to: {result_dir}
"""
    print(summary)

    with open(result_dir / "summary.txt", "w") as f:
        f.write(summary)

    return result


def run_quick_test(symbol: str, timeframe: str, days: int = 365) -> Dict:
    """Run quick test with default config (no discovery)."""
    from core import set_backtest_mode
    set_backtest_mode(True)

    result_dir = get_result_dir(symbol, timeframe, "quick")

    print(f"\n{'='*70}")
    print(f"QUICK TEST: {symbol} {timeframe} ({days} days)")
    print(f"{'='*70}")
    print(f"Config: {DEFAULT_FILTERS} (fixed)")
    print(f"{'='*70}\n")

    # Fetch data
    print("[1/3] Fetching data...")
    df = fetch_data(symbol, timeframe, days)
    print(f"      {len(df)} candles")

    # Portfolio backtest
    print("\n[2/3] Portfolio backtest...")
    portfolio = run_portfolio_backtest(df, DEFAULT_FILTERS)
    print(f"      Trades: {portfolio['trades']} | WR: {portfolio['win_rate']:.1f}%")
    print(f"      PnL: ${portfolio['total_pnl']:.2f}")

    # Save
    print("\n[3/3] Saving...")
    verdict = "PASS" if portfolio['total_pnl'] > 0 and not portfolio['stopped'] else "FAIL"

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "mode": "quick",
        "filters": DEFAULT_FILTERS,
        "portfolio": {
            "pnl": portfolio['total_pnl'],
            "trades": portfolio['trades'],
            "win_rate": portfolio['win_rate'],
            "pf": portfolio['profit_factor'],
            "max_dd_pct": portfolio['max_drawdown_pct'],
        },
        "verdict": verdict,
    }

    with open(result_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"{symbol} {timeframe} | {verdict}")
    print(f"PnL: ${portfolio['total_pnl']:.2f} | Trades: {portfolio['trades']} | WR: {portfolio['win_rate']:.1f}%")
    print(f"{'='*70}")

    return result


def run_viz(symbol: str, timeframe: str):
    """Visualize trades from latest test."""
    from core.trade_visualizer import TradeVisualizer
    from core import get_client, calculate_indicators, set_backtest_mode
    import pandas as pd

    set_backtest_mode(True)

    latest = get_latest_result(symbol, timeframe)
    if not latest:
        print(f"No results for {symbol} {timeframe}")
        return

    trades_file = latest / "trades.json"
    if not trades_file.exists():
        print(f"No trades in {latest}")
        return

    with open(trades_file) as f:
        trades = json.load(f)

    if not trades:
        print("No trades to visualize")
        return

    print(f"\nVisualizing {len(trades)} trades from {latest}...")

    client = get_client()
    df = client.get_klines(symbol=symbol, interval=timeframe, limit=1000)
    df = calculate_indicators(df, timeframe=timeframe)

    viz_dir = latest / "charts"
    ensure_dir(viz_dir)
    visualizer = TradeVisualizer(output_dir=str(viz_dir))

    count = 0
    for trade in trades:
        try:
            # Calculate R-multiple
            entry_p = trade.get("entry_price", 0)
            sl_p = trade.get("sl_price", 0)
            risk = abs(entry_p - sl_p) if entry_p and sl_p else 1
            r_mult = trade.get("pnl", 0) / (risk * trade.get("position_size", 1)) if risk else 0

            viz_trade = {
                "symbol": f"{symbol}-{timeframe}",
                "timeframe": timeframe,
                "type": trade.get("signal_type", "LONG"),
                "open_time_utc": trade.get("entry_time", ""),
                "entry": trade.get("entry_price", 0),
                "tp": trade.get("tp_price", 0),
                "sl": trade.get("sl_price", 0),
                "close_time_utc": trade.get("exit_time", ""),
                "close_price": trade.get("exit_price", 0),
                "exit_reason": trade.get("exit_reason", ""),
                "pnl": trade.get("pnl", 0),
                "r_multiple": r_mult,
                "status": "WON" if trade.get("win") else "LOST",
            }
            chart_path = visualizer.visualize_trade(viz_trade)
            if chart_path:
                count += 1
                print(f"  [{count}] {trade.get('entry_time', '')[:10]} {trade.get('signal_type')} ${trade.get('pnl', 0):.2f}")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nCreated {count} charts in {viz_dir}")


def run_report():
    """Show all test results."""
    print(f"\n{'='*70}")
    print("ALL TEST RESULTS")
    print(f"{'='*70}\n")

    if not RESULTS_DIR.exists():
        print("No results yet. Run: python run.py test BTCUSDT 15m --full")
        return

    results = []
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        rf = d / "result.json"
        if not rf.exists():
            continue
        with open(rf) as f:
            r = json.load(f)
        p = r.get("portfolio", {})
        results.append({
            "dir": d.name,
            "symbol": r.get("symbol", "?"),
            "tf": r.get("timeframe", "?"),
            "mode": r.get("mode", "?"),
            "trades": p.get("trades", 0),
            "pnl": p.get("pnl", 0),
            "wr": p.get("win_rate", 0),
            "verdict": r.get("verdict", "?"),
        })

    print(f"{'Symbol':<10} {'TF':<6} {'Mode':<8} {'Trades':<8} {'PnL':<12} {'WR%':<8} {'Verdict'}")
    print("-" * 75)

    for r in results[:15]:
        pnl = f"${r['pnl']:.2f}" if r['pnl'] else "$0"
        if r['pnl'] and r['pnl'] > 0:
            pnl = f"+{pnl}"
        print(f"{r['symbol']:<10} {r['tf']:<6} {r['mode']:<8} {r['trades']:<8} {pnl:<12} {r['wr']:.1f}%{'':<4} {r['verdict']}")


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py test BTCUSDT 15m --full    # Full pipeline (recommended)
  python run.py test BTCUSDT 15m           # Quick test (fixed config)
  python run.py test BTCUSDT 15m --quick   # 90-day quick test
  python run.py viz BTCUSDT 15m            # Visualize trades
  python run.py report                     # Show all results
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # test
    test_p = subparsers.add_parser("test", help="Run test")
    test_p.add_argument("symbol", help="Symbol (e.g., BTCUSDT)")
    test_p.add_argument("timeframe", help="Timeframe (e.g., 15m)")
    test_p.add_argument("--days", type=int, default=365)
    test_p.add_argument("--full", action="store_true", help="Full pipeline (discovery + WF + portfolio)")
    test_p.add_argument("--quick", action="store_true", help="90-day quick test")

    # viz
    viz_p = subparsers.add_parser("viz", help="Visualize trades")
    viz_p.add_argument("symbol")
    viz_p.add_argument("timeframe")

    # report
    subparsers.add_parser("report", help="Show all results")

    args = parser.parse_args()

    if args.command == "test":
        days = 90 if args.quick else args.days
        if args.full:
            run_full_pipeline(args.symbol, args.timeframe, days)
        else:
            run_quick_test(args.symbol, args.timeframe, days)
    elif args.command == "viz":
        run_viz(args.symbol, args.timeframe)
    elif args.command == "report":
        run_report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
