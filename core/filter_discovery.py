# core/filter_discovery.py
# Filter Combination Discovery System for SSL Flow Strategy
#
# Purpose: Discover the optimal AND filter combination for SSL Flow strategy
# Problem: Currently 7+ filters → only 9 trades/year (over-filtering)
# Solution: Test 2^7 = 128 filter combinations to find sweet spot
#
# Approach:
# - Phase 1: Pilot on BTCUSDT-15m (128 combos × ~3min = 6-8 hours)
# - Phase 2: Validate top combinations on full universe
# - Walk-forward validation: 60% search, 20% WF, 20% holdout
#
# CORE filters (NEVER toggle, always ON):
# - AlphaTrend dominance check
# - Price position vs baseline (determines LONG/SHORT)
# - RR validation (can tune threshold, not disable)

import os
import sys
import json
import time
import itertools
import threading
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from core.config import DATA_DIR, TRADING_CONFIG
from core.trade_manager import SimTradeManager
from core.utils import tf_to_timedelta
from core.trading_engine import TradingEngine


# ==========================================
# FILTER COMBINATION DATA STRUCTURES
# ==========================================

@dataclass
class FilterCombination:
    """Represents a single combination of toggleable filters.

    TOGGLEABLE filters (can be enabled/disabled) - 7 filters = 128 combinations:
    - adx_filter: ADX >= adx_min
    - regime_gating: ADX_avg >= threshold over N bars
    - baseline_touch: Price touched baseline in lookback
    - pbema_distance: Min distance to PBEMA target
    - body_position: Candle body above/below baseline
    - ssl_pbema_overlap: Check for SSL-PBEMA overlap
    - wick_rejection: Wick rejection quality (10% min wick)

    CORE filters (NEVER toggle - always ON):
    - AlphaTrend dominance (buyers/sellers)
    - Price position vs baseline (LONG/SHORT direction)
    - RR validation (min_rr threshold)
    """
    adx_filter: bool = True
    regime_gating: bool = True
    baseline_touch: bool = True
    pbema_distance: bool = True
    body_position: bool = True
    ssl_pbema_overlap: bool = True
    wick_rejection: bool = True  # 7th filter: wick rejection (10% min wick)

    def to_config_overrides(self) -> dict:
        """Convert to config dict that can be passed to check_ssl_flow_signal.

        When a filter is disabled (False), we pass parameters that effectively
        disable the filter check:
        - adx_filter=False → adx_min=-999 (always passes)
        - regime_gating=False → regime_adx_threshold=-999 (always passes)
        - baseline_touch=False → lookback_candles=999 (always finds touch)
        - pbema_distance=False → min_pbema_distance=-999 (always passes)
        - body_position=False → ssl_body_tolerance=999 (always passes)
        - ssl_pbema_overlap=False → skip overlap check (need custom flag)
        """
        overrides = {}

        # ADX filter: set adx_min very low to disable
        if not self.adx_filter:
            overrides['adx_min'] = -999.0

        # Regime gating: set threshold very low to disable
        if not self.regime_gating:
            overrides['regime_adx_threshold'] = -999.0

        # Baseline touch: set lookback very high so it always finds a touch
        if not self.baseline_touch:
            overrides['lookback_candles'] = 9999

        # PBEMA distance: set min distance very low to disable
        if not self.pbema_distance:
            overrides['min_pbema_distance'] = -999.0

        # Body position: set tolerance very high to always pass
        if not self.body_position:
            overrides['ssl_body_tolerance'] = 999.0

        # SSL-PBEMA overlap: requires custom flag (add to overrides)
        if not self.ssl_pbema_overlap:
            overrides['skip_overlap_check'] = True

        # Wick rejection: requires custom flag (add to overrides)
        if not self.wick_rejection:
            overrides['skip_wick_rejection'] = True

        return overrides

    def to_string(self) -> str:
        """Human-readable string representation."""
        enabled = []
        if self.adx_filter:
            enabled.append("ADX")
        if self.regime_gating:
            enabled.append("REGIME")
        if self.baseline_touch:
            enabled.append("TOUCH")
        if self.pbema_distance:
            enabled.append("PBEMA_DIST")
        if self.body_position:
            enabled.append("BODY")
        if self.ssl_pbema_overlap:
            enabled.append("OVERLAP")
        if self.wick_rejection:
            enabled.append("WICK")

        if not enabled:
            return "NO_FILTERS"
        return "+".join(enabled)

    def __hash__(self):
        return hash((
            self.adx_filter,
            self.regime_gating,
            self.baseline_touch,
            self.pbema_distance,
            self.body_position,
            self.ssl_pbema_overlap,
            self.wick_rejection,
        ))


@dataclass
class FilterDiscoveryResult:
    """Results from testing a single filter combination."""
    combination: FilterCombination

    # Training metrics (in-sample search period)
    train_pnl: float
    train_trades: int
    train_expected_r: float
    train_win_rate: float
    train_score: float

    # Walk-forward validation metrics (out-of-sample)
    wf_pnl: float
    wf_trades: int
    wf_expected_r: float
    wf_win_rate: float

    # Holdout validation metrics (final test)
    holdout_pnl: Optional[float] = None
    holdout_trades: Optional[int] = None
    holdout_expected_r: Optional[float] = None
    holdout_win_rate: Optional[float] = None

    # Overfit detection
    overfit_ratio: float = 0.0  # WF E[R] / Train E[R]
    is_overfit: bool = False

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        result = {
            'combination': asdict(self.combination),
            'combination_str': self.combination.to_string(),
            'train_pnl': self.train_pnl,
            'train_trades': self.train_trades,
            'train_expected_r': self.train_expected_r,
            'train_win_rate': self.train_win_rate,
            'train_score': self.train_score,
            'wf_pnl': self.wf_pnl,
            'wf_trades': self.wf_trades,
            'wf_expected_r': self.wf_expected_r,
            'wf_win_rate': self.wf_win_rate,
            'overfit_ratio': self.overfit_ratio,
            'is_overfit': self.is_overfit,
        }

        if self.holdout_pnl is not None:
            result.update({
                'holdout_pnl': self.holdout_pnl,
                'holdout_trades': self.holdout_trades,
                'holdout_expected_r': self.holdout_expected_r,
                'holdout_win_rate': self.holdout_win_rate,
            })

        return result


# ==========================================
# SCORING FUNCTION
# ==========================================

def filter_discovery_score(
    net_pnl: float,
    trades: int,
    trade_pnls: list,
    trade_r_multiples: list,
    baseline_trades: int = 9
) -> float:
    """Scoring function for filter combinations.

    Goals:
    - Prefer positive edge (E[R] > 0.05)
    - Prefer more trades than baseline (9 trades/year is too low)
    - Reward consistency (good Sharpe-like ratio)
    - Penalize very low win rates

    Components:
    - 40% E[R] (expected R-multiple)
    - 30% frequency bonus (want more trades)
    - 10% Sharpe-like ratio
    - 20% win rate factor

    Args:
        net_pnl: Total PnL
        trades: Number of trades
        trade_pnls: List of individual trade PnLs
        trade_r_multiples: List of R-multiples per trade
        baseline_trades: Baseline trade count (default 9)

    Returns:
        Composite score (higher is better)
    """
    # Hard reject: no trades or negative PnL
    if trades == 0 or net_pnl <= 0:
        return -float("inf")

    # Calculate expected R-multiple
    if trade_r_multiples and len(trade_r_multiples) > 0:
        expected_r = sum(trade_r_multiples) / len(trade_r_multiples)
    else:
        expected_r = 0.0

    # Hard reject: E[R] too low (barely positive, not a real edge)
    if expected_r < 0.05:
        return -float("inf")

    # Trade frequency bonus (want more trades than baseline)
    # freq_ratio = 1.0 means same as baseline
    # freq_ratio > 1.0 means more trades (good)
    # freq_ratio < 1.0 means fewer trades (bad)
    freq_ratio = trades / baseline_trades if baseline_trades > 0 else 1.0

    if freq_ratio >= 1.0:
        # Bonus for more trades, capped at 2x
        freq_bonus = min(2.0, 1.0 + (freq_ratio - 1.0) * 0.3)
    else:
        # Penalty for fewer trades
        freq_bonus = 0.5

    # Win rate factor
    win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls) if trade_pnls else 0.0

    if win_rate < 0.35:
        wr_factor = 0.8  # Penalty for very low win rate
    elif win_rate < 0.50:
        wr_factor = 1.0  # Neutral
    else:
        wr_factor = 1.1  # Slight bonus for high win rate

    # Sharpe-like ratio (mean / std of PnLs)
    if len(trade_pnls) >= 5:
        mean_pnl = sum(trade_pnls) / len(trade_pnls)
        variance = sum((p - mean_pnl) ** 2 for p in trade_pnls) / len(trade_pnls)
        std_pnl = variance ** 0.5
        sharpe = mean_pnl / (std_pnl + 1e-6) if std_pnl > 0 else 0.5
        sharpe = min(sharpe, 2.0)  # Cap at 2.0
    else:
        sharpe = 0.5  # Not enough data for Sharpe

    # Combined score: 40% E[R], 30% frequency, 10% Sharpe, 20% win rate
    # Multiply by net_pnl to scale by absolute performance
    score = (
        expected_r * 0.40 +
        freq_bonus * 0.30 +
        sharpe * 0.10 +
        wr_factor * 0.20
    ) * net_pnl

    return score


# ==========================================
# PROGRESS TRACKER
# ==========================================

class ProgressTracker:
    """Real-time progress tracker for filter discovery."""

    def __init__(self, total: int, description: str = "Progress"):
        """Initialize progress tracker.

        Args:
            total: Total number of items to process
            description: Description of the task
        """
        self.total = total
        self.description = description
        self.completed = 0
        self.start_time = time.time()
        self.last_update_time = 0
        self.update_interval = 0.5  # Update display every 0.5 seconds minimum
        self.lock = threading.Lock()  # Thread-safe counter updates

    def update(self, increment: int = 1, current_item: str = None):
        """Update progress counter.

        Args:
            increment: Number of items completed
            current_item: Description of current item being processed
        """
        with self.lock:  # Thread-safe increment
            self.completed += increment
            current_time = time.time()

            # Only update display if enough time has passed (avoid spam)
            if current_time - self.last_update_time < self.update_interval:
                return

            self.last_update_time = current_time

        self._display(current_item)  # Display outside lock to avoid blocking

    def _display(self, current_item: str = None):
        """Display progress bar."""
        elapsed = time.time() - self.start_time
        progress_pct = (self.completed / self.total) * 100 if self.total > 0 else 0

        # Calculate ETA
        if self.completed > 0:
            avg_time_per_item = elapsed / self.completed
            remaining_items = self.total - self.completed
            eta_seconds = avg_time_per_item * remaining_items
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."

        # Build progress bar
        bar_width = 40
        filled = int(bar_width * self.completed / self.total) if self.total > 0 else 0

        # Handle edge case: if filled equals bar_width, don't show ">"
        if filled >= bar_width:
            bar = "=" * bar_width
        else:
            bar = "=" * filled + ">" + " " * (bar_width - filled - 1)

        # Format elapsed time
        elapsed_str = self._format_time(elapsed)

        # Build status line - clear previous line first
        status = f"\r{' ' * 150}\r"  # Clear with spaces
        status += f"[{bar}] {self.completed}/{self.total} ({progress_pct:.1f}%)"
        status += f" | Elapsed: {elapsed_str} | ETA: {eta_str}"

        if current_item:
            # Truncate current_item if too long
            max_item_len = 30
            if len(current_item) > max_item_len:
                current_item = current_item[:max_item_len-3] + "..."
            status += f" | Current: {current_item}"

        # Write to stdout and flush
        sys.stdout.write(status)
        sys.stdout.flush()

    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string (e.g., "2h 34m", "45s")
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def finish(self):
        """Complete progress tracking and print final newline."""
        self._display()
        sys.stdout.write("\n")
        sys.stdout.flush()


# ==========================================
# FILTER DISCOVERY ENGINE
# ==========================================

class FilterDiscoveryEngine:
    """Engine for discovering optimal filter combinations."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        base_config: dict = None,
        baseline_trades: int = 9,
    ):
        """Initialize filter discovery engine.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (e.g., "15m")
            data: Full DataFrame with OHLCV + indicators
            base_config: Base strategy config (rr, rsi, etc.)
            baseline_trades: Baseline trade count for scoring
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = data
        self.baseline_trades = baseline_trades

        # Default base config if not provided
        if base_config is None:
            base_config = {
                "rr": 2.0,
                "rsi": 70,
                "at_active": True,
                "use_trailing": False,
                "use_partial": True,
                "use_dynamic_pbema_tp": False,
                "strategy_mode": "ssl_flow",
            }
        self.base_config = base_config

        # Split data into train (60%), walk-forward (20%), holdout (20%)
        self._split_data()

    def _split_data(self):
        """Split data into train, walk-forward, and holdout periods."""
        n = len(self.data)

        # 60% train, 20% WF, 20% holdout
        train_end = int(n * 0.60)
        wf_end = int(n * 0.80)

        self.train_data = self.data.iloc[:train_end].reset_index(drop=True)
        self.wf_data = self.data.iloc[train_end:wf_end].reset_index(drop=True)
        self.holdout_data = self.data.iloc[wf_end:].reset_index(drop=True)

        print(f"[DISCOVERY][{self.symbol}-{self.timeframe}] Data split:")
        print(f"  Train: {len(self.train_data)} candles ({train_end/n*100:.1f}%)")
        print(f"  WF: {len(self.wf_data)} candles ({(wf_end-train_end)/n*100:.1f}%)")
        print(f"  Holdout: {len(self.holdout_data)} candles ({(n-wf_end)/n*100:.1f}%)")

    def generate_combinations(self) -> List[FilterCombination]:
        """Generate all 2^7 = 128 filter combinations."""
        # All possible True/False values for 7 filters
        filter_names = [
            'adx_filter',
            'regime_gating',
            'baseline_touch',
            'pbema_distance',
            'body_position',
            'ssl_pbema_overlap',
            'wick_rejection',
        ]

        combinations = []

        # Generate all 128 combinations using itertools.product
        for values in itertools.product([True, False], repeat=7):
            combo = FilterCombination(**dict(zip(filter_names, values)))
            combinations.append(combo)

        return combinations

    def evaluate_combination(
        self,
        combo: FilterCombination,
        on_holdout: bool = False
    ) -> FilterDiscoveryResult:
        """Evaluate a single filter combination using walk-forward validation.

        Args:
            combo: Filter combination to test
            on_holdout: If True, use holdout data; else use train/WF

        Returns:
            FilterDiscoveryResult with train and WF metrics
        """
        # Build config with filter overrides
        config = {**self.base_config}
        overrides = combo.to_config_overrides()
        config.update(overrides)

        if on_holdout:
            # Holdout validation: test on holdout data only
            holdout_pnl, holdout_trades, holdout_pnls, holdout_r_multiples = self._score_config(
                self.holdout_data, config
            )

            holdout_expected_r = (
                sum(holdout_r_multiples) / len(holdout_r_multiples)
                if holdout_r_multiples else 0.0
            )
            holdout_win_rate = (
                sum(1 for p in holdout_pnls if p > 0) / len(holdout_pnls)
                if holdout_pnls else 0.0
            )

            # Return minimal result with only holdout metrics
            return FilterDiscoveryResult(
                combination=combo,
                train_pnl=0.0,
                train_trades=0,
                train_expected_r=0.0,
                train_win_rate=0.0,
                train_score=0.0,
                wf_pnl=0.0,
                wf_trades=0,
                wf_expected_r=0.0,
                wf_win_rate=0.0,
                holdout_pnl=holdout_pnl,
                holdout_trades=holdout_trades,
                holdout_expected_r=holdout_expected_r,
                holdout_win_rate=holdout_win_rate,
            )

        # Train period evaluation
        train_pnl, train_trades, train_pnls, train_r_multiples = self._score_config(
            self.train_data, config
        )

        train_expected_r = (
            sum(train_r_multiples) / len(train_r_multiples)
            if train_r_multiples else 0.0
        )
        train_win_rate = (
            sum(1 for p in train_pnls if p > 0) / len(train_pnls)
            if train_pnls else 0.0
        )

        # Compute train score
        train_score = filter_discovery_score(
            train_pnl, train_trades, train_pnls, train_r_multiples, self.baseline_trades
        )

        # Walk-forward period evaluation
        wf_pnl, wf_trades, wf_pnls, wf_r_multiples = self._score_config(
            self.wf_data, config
        )

        wf_expected_r = (
            sum(wf_r_multiples) / len(wf_r_multiples)
            if wf_r_multiples else 0.0
        )
        wf_win_rate = (
            sum(1 for p in wf_pnls if p > 0) / len(wf_pnls)
            if wf_pnls else 0.0
        )

        # Overfit detection (require min E[R] to avoid division by tiny number)
        if train_expected_r > 0.01 and wf_trades >= 3:
            overfit_ratio = wf_expected_r / train_expected_r
            # Mark as overfit if WF E[R] < 50% of train E[R]
            is_overfit = overfit_ratio < 0.50 or wf_expected_r < 0
        else:
            overfit_ratio = 0.0
            is_overfit = True  # Not enough WF trades or train not positive

        return FilterDiscoveryResult(
            combination=combo,
            train_pnl=train_pnl,
            train_trades=train_trades,
            train_expected_r=train_expected_r,
            train_win_rate=train_win_rate,
            train_score=train_score,
            wf_pnl=wf_pnl,
            wf_trades=wf_trades,
            wf_expected_r=wf_expected_r,
            wf_win_rate=wf_win_rate,
            overfit_ratio=overfit_ratio,
            is_overfit=is_overfit,
        )

    def _score_config(
        self,
        df: pd.DataFrame,
        config: dict
    ) -> Tuple[float, int, List[float], List[float]]:
        """Run backtest simulation with given config on given data.

        Returns:
            (net_pnl, trades, trade_pnls, trade_r_multiples)
        """
        tm = SimTradeManager(initial_balance=TRADING_CONFIG["initial_balance"])
        warmup = 250
        end = len(df) - 2

        if end <= warmup:
            return 0.0, 0, [], []

        # Extract NumPy arrays for performance
        timestamps = pd.to_datetime(df["timestamp"]).values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        opens = df["open"].values
        pb_tops = df["pb_ema_top"].values if "pb_ema_top" in df.columns else closes
        pb_bots = df["pb_ema_bot"].values if "pb_ema_bot" in df.columns else closes

        # AlphaTrend arrays for momentum TP extension
        at_buyers_arr = (
            df["at_buyers_dominant"].values
            if "at_buyers_dominant" in df.columns else None
        )
        at_sellers_arr = (
            df["at_sellers_dominant"].values
            if "at_sellers_dominant" in df.columns else None
        )

        # OPTIMIZATION 1: Pre-compute event times outside loop
        tf_delta = tf_to_timedelta(self.timeframe)
        event_times = pd.to_datetime(timestamps) + tf_delta

        # OPTIMIZATION 2: Pre-allocate candle_data template for reuse
        _candle_data_template = {"at_buyers_dominant": False, "at_sellers_dominant": False}

        for i in range(warmup, end):
            event_time = event_times[i]

            # Build candle_data dict - reuse template if AlphaTrend available
            candle_data = None
            if at_buyers_arr is not None and at_sellers_arr is not None:
                _candle_data_template["at_buyers_dominant"] = bool(at_buyers_arr[i])
                _candle_data_template["at_sellers_dominant"] = bool(at_sellers_arr[i])
                candle_data = _candle_data_template

            # Update trades
            tm.update_trades(
                self.symbol,
                self.timeframe,
                candle_high=float(highs[i]),
                candle_low=float(lows[i]),
                candle_close=float(closes[i]),
                candle_time_utc=event_time,
                pb_top=float(pb_tops[i]),
                pb_bot=float(pb_bots[i]),
                candle_data=candle_data,
            )

            # Check for signal
            s_type, s_entry, s_tp, s_sl, s_reason = TradingEngine.check_signal(
                df, config=config, index=i, return_debug=False
            )

            if not (s_type and "ACCEPTED" in s_reason):
                continue

            # OPTIMIZATION 5: Short-circuit empty trade list check
            has_open = bool(tm.open_trades) and any(
                t.get("symbol") == self.symbol and t.get("timeframe") == self.timeframe
                for t in tm.open_trades
            )
            if has_open or tm.check_cooldown(self.symbol, self.timeframe, event_time):
                continue

            # Entry at next candle open
            entry_open = float(opens[i + 1])
            open_ts = timestamps[i + 1]
            ts_str = (pd.Timestamp(open_ts) + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")

            # RR re-validation with actual entry price
            min_rr = config.get("rr", 2.0)
            if s_type == "LONG":
                actual_risk = entry_open - s_sl
                actual_reward = s_tp - entry_open
            else:  # SHORT
                actual_risk = s_sl - entry_open
                actual_reward = entry_open - s_tp

            if actual_risk <= 0 or actual_reward <= 0:
                continue

            actual_rr = actual_reward / actual_risk
            if actual_rr < min_rr * 0.9:  # 10% tolerance
                continue

            # Open trade
            tm.open_trade({
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "type": s_type,
                "setup": s_reason,
                "entry": entry_open,
                "tp": s_tp,
                "sl": s_sl,
                "timestamp": ts_str,
                "open_time_utc": open_ts,
                "use_trailing": config.get("use_trailing", False),
                "use_dynamic_pbema_tp": config.get("use_dynamic_pbema_tp", False),
            })

        # Extract results
        trade_pnls = [t.get("pnl", 0.0) for t in tm.history] if tm.history else []
        trade_r_multiples = (
            tm.trade_r_multiples if hasattr(tm, 'trade_r_multiples') else []
        )
        unique_trades = len({t.get("id") for t in tm.history}) if tm.history else 0

        return tm.total_pnl, unique_trades, trade_pnls, trade_r_multiples

    def run_discovery(
        self,
        parallel: bool = True,
        max_workers: int = None,
    ) -> List[FilterDiscoveryResult]:
        """Run discovery on all filter combinations.

        Args:
            parallel: Use parallel processing (default: True)
            max_workers: Max parallel workers (default: auto)

        Returns:
            List of FilterDiscoveryResult, sorted by train_score descending
        """
        combinations = self.generate_combinations()

        print(f"\n[DISCOVERY][{self.symbol}-{self.timeframe}] Testing {len(combinations)} filter combinations...")
        print(f"Parallel: {parallel}, Workers: {max_workers or 'auto'}")
        print(f"Starting discovery at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)

        results = []

        # Initialize progress tracker
        progress = ProgressTracker(
            total=len(combinations),
            description=f"Discovery {self.symbol}-{self.timeframe}"
        )

        if parallel:
            # Parallel execution
            if max_workers is None:
                max_workers = max(1, os.cpu_count() - 1)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.evaluate_combination, combo): combo
                    for combo in combinations
                }

                for future in as_completed(futures):
                    combo = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        # Update progress with current combo name
                        progress.update(increment=1, current_item=combo.to_string())
                    except Exception as exc:
                        # Print error on new line to not mess up progress bar
                        sys.stdout.write("\n")
                        print(f"[DISCOVERY] Error evaluating {combo.to_string()}: {exc}")
                        progress.update(increment=1, current_item=combo.to_string())
                        continue
        else:
            # Serial execution (for debugging)
            for combo in combinations:
                try:
                    result = self.evaluate_combination(combo)
                    results.append(result)
                    progress.update(increment=1, current_item=combo.to_string())
                except Exception as exc:
                    # Print error on new line to not mess up progress bar
                    sys.stdout.write("\n")
                    print(f"[DISCOVERY] Error evaluating {combo.to_string()}: {exc}")
                    progress.update(increment=1, current_item=combo.to_string())
                    continue

        # Finish progress tracking
        progress.finish()

        # Sort by train_score descending
        results.sort(key=lambda r: r.train_score, reverse=True)

        print("-" * 80)
        print(f"[DISCOVERY][{self.symbol}-{self.timeframe}] Discovery complete!")
        print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total combinations tested: {len(results)}")
        print(f"Non-overfitted: {sum(1 for r in results if not r.is_overfit)}")

        return results

    def validate_on_holdout(self, combo: FilterCombination) -> dict:
        """Final validation of a filter combination on holdout data.

        Args:
            combo: Filter combination to validate

        Returns:
            Dict with holdout metrics
        """
        result = self.evaluate_combination(combo, on_holdout=True)

        return {
            'holdout_pnl': result.holdout_pnl,
            'holdout_trades': result.holdout_trades,
            'holdout_expected_r': result.holdout_expected_r,
            'holdout_win_rate': result.holdout_win_rate,
        }

    def analyze_individual_filter_pass_rates(self) -> dict:
        """Analyze how often each individual filter passes.

        This helps identify bottleneck filters that are too restrictive.

        Returns:
            Dict mapping filter name to pass statistics:
            {
                'filter_name': {
                    'pass_count': int,
                    'total_count': int,
                    'pass_rate': float,
                    'signals_lost': float  # % of signals blocked by this filter
                }
            }
        """
        from strategies.ssl_flow import check_ssl_flow_signal

        print(f"\n[FILTER ANALYSIS] Analyzing individual filter pass rates...")
        print(f"[FILTER ANALYSIS] Testing on train data: {len(self.train_data)} candles")

        # Build base config
        config = {**self.base_config}

        # Get baseline signal count (no filters - except CORE filters)
        baseline_combo = FilterCombination(
            adx_filter=False,
            regime_gating=False,
            baseline_touch=False,
            pbema_distance=False,
            body_position=False,
            ssl_pbema_overlap=False,
            wick_rejection=False,
        )
        baseline_overrides = baseline_combo.to_config_overrides()
        baseline_config = {**config, **baseline_overrides}

        # Count baseline signals
        warmup = 250
        end = len(self.train_data) - 2
        baseline_signals = 0

        for i in range(warmup, end):
            s_type, _, _, _, s_reason = TradingEngine.check_signal(
                self.train_data, config=baseline_config, index=i, return_debug=False
            )
            if s_type:
                baseline_signals += 1

        print(f"[FILTER ANALYSIS] Baseline signals (no filters): {baseline_signals}")

        if baseline_signals == 0:
            print(f"[FILTER ANALYSIS] Warning: No baseline signals found!")
            return {}

        # Test each filter individually
        filter_names = [
            'adx_filter',
            'regime_gating',
            'baseline_touch',
            'pbema_distance',
            'body_position',
            'ssl_pbema_overlap',
            'wick_rejection',
        ]

        filter_stats = {}

        for filter_name in filter_names:
            # Create combo with only this filter enabled
            combo_dict = {fn: False for fn in filter_names}
            combo_dict[filter_name] = True
            combo = FilterCombination(**combo_dict)

            # Test signals with this filter
            test_overrides = combo.to_config_overrides()
            test_config = {**config, **test_overrides}

            signals_with_filter = 0

            for i in range(warmup, end):
                s_type, _, _, _, s_reason = TradingEngine.check_signal(
                    self.train_data, config=test_config, index=i, return_debug=False
                )
                if s_type:
                    signals_with_filter += 1

            pass_rate = signals_with_filter / baseline_signals if baseline_signals > 0 else 0
            signals_lost = 1.0 - pass_rate

            filter_stats[filter_name] = {
                'pass_count': signals_with_filter,
                'total_count': baseline_signals,
                'pass_rate': pass_rate,
                'signals_lost': signals_lost,
            }

            print(f"  {filter_name:20s} - Pass: {signals_with_filter:4d}/{baseline_signals:4d} ({pass_rate:6.1%}) - Lost: {signals_lost:6.1%}")

        # Sort by pass_rate (ascending) to find bottlenecks
        sorted_filters = sorted(filter_stats.items(), key=lambda x: x[1]['pass_rate'])

        print(f"\n[FILTER ANALYSIS] Bottleneck filters (most restrictive):")
        for i, (filter_name, stats) in enumerate(sorted_filters[:3], 1):
            print(f"  {i}. {filter_name:20s} - {stats['pass_rate']:6.1%} pass rate (blocks {stats['signals_lost']:6.1%})")

        return filter_stats

    def find_pareto_optimal_combinations(
        self,
        results: List[FilterDiscoveryResult],
        min_trades: int = 5,
        min_expected_r: float = 0.0,
    ) -> List[FilterDiscoveryResult]:
        """Find Pareto-optimal filter combinations.

        A combination is Pareto-optimal if no other combination has:
        - BOTH higher E[R] AND more trades

        This helps visualize the trade-off between trade frequency and edge quality.

        Args:
            results: List of FilterDiscoveryResult from discovery
            min_trades: Minimum trades to consider (filter out low-frequency configs)
            min_expected_r: Minimum E[R] to consider (filter out negative edge)

        Returns:
            List of Pareto-optimal FilterDiscoveryResult, sorted by trades descending
        """
        print(f"\n[PARETO] Finding Pareto-optimal combinations...")
        print(f"[PARETO] Filters: min_trades={min_trades}, min_expected_r={min_expected_r}")

        # Filter by minimum criteria
        valid_results = [
            r for r in results
            if r.train_trades >= min_trades and r.train_expected_r >= min_expected_r and not r.is_overfit
        ]

        print(f"[PARETO] Valid combinations: {len(valid_results)}/{len(results)}")

        if not valid_results:
            print(f"[PARETO] No valid combinations found!")
            return []

        # Find Pareto frontier
        pareto_optimal = []

        for candidate in valid_results:
            is_dominated = False

            # Check if any other result dominates this candidate
            for other in valid_results:
                if other == candidate:
                    continue

                # Other dominates candidate if it has:
                # - Higher or equal E[R] AND more trades
                # - OR higher E[R] AND equal or more trades
                if (other.train_expected_r >= candidate.train_expected_r and
                    other.train_trades > candidate.train_trades) or \
                   (other.train_expected_r > candidate.train_expected_r and
                    other.train_trades >= candidate.train_trades):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_optimal.append(candidate)

        # Sort by trades descending (high frequency to low frequency)
        pareto_optimal.sort(key=lambda r: r.train_trades, reverse=True)

        print(f"[PARETO] Found {len(pareto_optimal)} Pareto-optimal combinations")

        # Print summary
        print(f"\n[PARETO] Pareto-optimal frontier:")
        print(f"{'Rank':<6} {'Trades':<8} {'E[R]':<8} {'Win%':<8} {'Filters'}")
        print("-" * 80)

        for i, result in enumerate(pareto_optimal, 1):
            combo_str = result.combination.to_string()
            if len(combo_str) > 40:
                combo_str = combo_str[:37] + "..."

            print(f"{i:<6} {result.train_trades:<8} {result.train_expected_r:<8.3f} "
                  f"{result.train_win_rate:<8.1%} {combo_str}")

        return pareto_optimal

    def generate_parameter_sensitivity_grid(
        self,
        param_grids: dict = None,
        top_n_combinations: int = 5,
    ) -> dict:
        """Test parameter sensitivity for top filter combinations.

        This tests how sensitive the top combinations are to parameter changes.
        Helps identify robust parameter settings.

        Args:
            param_grids: Dict of parameter grids to test, e.g.:
                {
                    'adx_min': [10, 15, 20, 25],
                    'min_pbema_distance': [0.002, 0.003, 0.004, 0.005],
                    'lookback_candles': [3, 5, 8, 10],
                }
            top_n_combinations: Number of top combinations to test

        Returns:
            Dict with sensitivity results
        """
        if param_grids is None:
            param_grids = {
                'adx_min': [10, 15, 20, 25],
                'min_pbema_distance': [0.002, 0.003, 0.004, 0.005],
                'lookback_candles': [3, 5, 8, 10],
                'regime_adx_threshold': [15, 20, 25, 30],
            }

        print(f"\n[SENSITIVITY] Running parameter sensitivity analysis...")
        print(f"[SENSITIVITY] Testing top {top_n_combinations} combinations")
        print(f"[SENSITIVITY] Parameter grids:")
        for param, values in param_grids.items():
            print(f"  {param}: {values}")

        # Get top combinations from a quick discovery run
        print(f"\n[SENSITIVITY] Running baseline discovery to get top combinations...")
        baseline_results = self.run_discovery(parallel=True, max_workers=None)
        top_combinations = baseline_results[:top_n_combinations]

        sensitivity_results = {}

        for combo_idx, base_result in enumerate(top_combinations):
            combo = base_result.combination
            combo_str = combo.to_string()

            print(f"\n[SENSITIVITY] Testing combo #{combo_idx+1}: {combo_str}")
            print(f"[SENSITIVITY] Baseline: Trades={base_result.train_trades}, E[R]={base_result.train_expected_r:.3f}")

            combo_sensitivity = {
                'combination': combo,
                'baseline_result': base_result,
                'parameter_tests': {},
            }

            # Test each parameter
            for param_name, param_values in param_grids.items():
                param_results = []

                for param_value in param_values:
                    # Build config with this parameter value
                    config = {**self.base_config}
                    overrides = combo.to_config_overrides()
                    config.update(overrides)
                    config[param_name] = param_value

                    # Run backtest with this config
                    train_pnl, train_trades, train_pnls, train_r_multiples = self._score_config(
                        self.train_data, config
                    )

                    train_expected_r = (
                        sum(train_r_multiples) / len(train_r_multiples)
                        if train_r_multiples else 0.0
                    )
                    train_win_rate = (
                        sum(1 for p in train_pnls if p > 0) / len(train_pnls)
                        if train_pnls else 0.0
                    )

                    param_results.append({
                        'value': param_value,
                        'trades': train_trades,
                        'expected_r': train_expected_r,
                        'win_rate': train_win_rate,
                        'pnl': train_pnl,
                    })

                combo_sensitivity['parameter_tests'][param_name] = param_results

                # Print results for this parameter
                print(f"  {param_name}:")
                for pr in param_results:
                    print(f"    {pr['value']:8.4f} -> Trades={pr['trades']:3d}, E[R]={pr['expected_r']:.3f}, Win%={pr['win_rate']:.1%}")

            sensitivity_results[combo_idx] = combo_sensitivity

        return sensitivity_results

    def generate_comprehensive_report(
        self,
        results: List[FilterDiscoveryResult],
        filter_pass_rates: dict = None,
        pareto_optimal: List[FilterDiscoveryResult] = None,
        output_file: str = None,
    ) -> str:
        """Generate comprehensive filter optimization report.

        Args:
            results: All discovery results
            filter_pass_rates: Individual filter pass rate analysis
            pareto_optimal: Pareto-optimal combinations
            output_file: Path to save report (optional)

        Returns:
            Report as string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("FILTER OPTIMIZATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Symbol: {self.symbol}")
        lines.append(f"Timeframe: {self.timeframe}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")

        # CURRENT BASELINE
        baseline_result = results[0] if results else None
        if baseline_result:
            lines.append("CURRENT BASELINE (all filters ON):")
            lines.append(f"  Trades: {baseline_result.train_trades}/year")
            lines.append(f"  E[R]: {baseline_result.train_expected_r:.2f}")
            lines.append(f"  Win Rate: {baseline_result.train_win_rate:.1%}")
            lines.append(f"  PnL: ${baseline_result.train_pnl:.2f}")
            lines.append("")

        # BOTTLENECK ANALYSIS
        if filter_pass_rates:
            lines.append("BOTTLENECK ANALYSIS:")
            lines.append("  Most restrictive filters:")

            sorted_filters = sorted(
                filter_pass_rates.items(),
                key=lambda x: x[1]['pass_rate']
            )

            for i, (filter_name, stats) in enumerate(sorted_filters[:3], 1):
                lines.append(f"  {i}. {filter_name} ({stats['pass_rate']:.1%} pass) - Costs {stats['signals_lost']:.1%} of signals")

            lines.append("")

        # RECOMMENDED CONFIG
        if pareto_optimal and len(pareto_optimal) >= 3:
            # Pick the "balanced" config (middle of Pareto frontier)
            balanced_idx = len(pareto_optimal) // 2
            recommended = pareto_optimal[balanced_idx]

            lines.append("RECOMMENDED CONFIG:")

            # Show which filters to keep/remove
            combo = recommended.combination
            enabled_filters = []
            disabled_filters = []

            if combo.adx_filter:
                enabled_filters.append("adx_filter")
            else:
                disabled_filters.append("adx_filter")

            if combo.regime_gating:
                enabled_filters.append("regime_gating")
            else:
                disabled_filters.append("regime_gating")

            if combo.baseline_touch:
                enabled_filters.append("baseline_touch")
            else:
                disabled_filters.append("baseline_touch")

            if combo.pbema_distance:
                enabled_filters.append("pbema_distance")
            else:
                disabled_filters.append("pbema_distance")

            if combo.body_position:
                enabled_filters.append("body_position")
            else:
                disabled_filters.append("body_position")

            if combo.ssl_pbema_overlap:
                enabled_filters.append("ssl_pbema_overlap")
            else:
                disabled_filters.append("ssl_pbema_overlap")

            if combo.wick_rejection:
                enabled_filters.append("wick_rejection")
            else:
                disabled_filters.append("wick_rejection")

            lines.append(f"  KEEP: {', '.join(enabled_filters) if enabled_filters else 'NONE'}")
            lines.append(f"  REMOVE: {', '.join(disabled_filters) if disabled_filters else 'NONE'}")
            lines.append("")
            lines.append("  Expected:")
            lines.append(f"    Trades: {recommended.train_trades}/year (+{(recommended.train_trades/baseline_result.train_trades-1)*100:.0f}%)")
            lines.append(f"    E[R]: {recommended.train_expected_r:.2f}")
            lines.append("")

        # PARETO-OPTIMAL ALTERNATIVES
        if pareto_optimal and len(pareto_optimal) >= 3:
            lines.append("PARETO-OPTIMAL ALTERNATIVES:")

            # High quality (lowest trades, highest E[R])
            high_quality = pareto_optimal[-1]
            lines.append(f"  Config 1: High Quality (E[R]={high_quality.train_expected_r:.2f}, Trades={high_quality.train_trades})")

            # Balanced (middle)
            balanced = pareto_optimal[len(pareto_optimal) // 2]
            lines.append(f"  Config 2: Balanced (E[R]={balanced.train_expected_r:.2f}, Trades={balanced.train_trades})  <-- RECOMMENDED")

            # High frequency (most trades, lowest E[R])
            high_freq = pareto_optimal[0]
            lines.append(f"  Config 3: High Frequency (E[R]={high_freq.train_expected_r:.2f}, Trades={high_freq.train_trades})")
            lines.append("")

        # WALK-FORWARD VALIDATION
        if baseline_result:
            lines.append("WALK-FORWARD VALIDATION:")
            lines.append(f"  Train E[R]: {baseline_result.train_expected_r:.2f}")
            lines.append(f"  WF E[R]: {baseline_result.wf_expected_r:.2f}")
            lines.append(f"  Ratio: {baseline_result.overfit_ratio:.2f} ({'PASS' if not baseline_result.is_overfit else 'FAIL'} - {'not overfit' if not baseline_result.is_overfit else 'overfit'})")
            lines.append("")

        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"[REPORT] Saved to: {output_file}")

        return report


# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    'FilterCombination',
    'FilterDiscoveryResult',
    'FilterDiscoveryEngine',
    'ProgressTracker',
    'filter_discovery_score',
]
