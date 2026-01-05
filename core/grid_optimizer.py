# core/grid_optimizer.py
# Grid Search Optimizer for SSL Flow Strategy Parameters
#
# Hierarchical combinatorial grid search with statistical validation
# and robustness testing to prevent overfitting.
#
# Key Features:
# - Phase 1: Coarse grid (fast screening)
# - Phase 2: Fine grid around top performers
# - Phase 3: Robustness/sensitivity testing
# - Bonferroni correction for multiple comparisons
# - Bootstrap confidence intervals for Sharpe ratio
# - Parameter sensitivity analysis

import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from core.config import (
    TRADING_CONFIG, BASELINE_CONFIG, DEFAULT_STRATEGY_CONFIG,
    MIN_EXPECTANCY_R_MULTIPLE, MIN_SCORE_THRESHOLD,
    WALK_FORWARD_CONFIG, DATA_DIR,
)
from core.optimizer import _score_config_for_stream
from core.trade_manager import SimTradeManager
from core.utils import tf_to_timedelta


# ==========================================
# CONFIGURATION
# ==========================================

# Minimum trades threshold (quant analyst recommendation: lower for yearly tests)
MIN_TRADES_THRESHOLD = 20  # Was 30, lowered since yearly test only has 9 trades

# Penalty weights for scoring
SCORE_WEIGHTS = {
    'sharpe': 0.30,           # Sharpe ratio
    'profit_factor': 0.20,    # Profit factor (gross_profit / gross_loss)
    'drawdown': 0.20,         # (1 - max_dd_pct) - reward low drawdown
    'stability': 0.20,        # Stability bonus (return/volatility consistency)
    'trade_frequency': 0.10,  # Trade count normalized (min(count/50, 1.0))
}

# Penalties
TRADE_PENALTY_THRESHOLD = 30  # Trades below this get penalty
COMPLEXITY_PENALTY_PER_FILTER = 0.02  # 2% penalty per active filter


# ==========================================
# DATA CLASSES
# ==========================================

@dataclass
class GridConfig:
    """SSL Flow parameter configuration for grid search."""
    rsi_threshold: int
    adx_threshold: float
    rr_ratio: float
    regime_adx_avg: float
    at_active: bool = True  # Mandatory for SSL Flow

    # Additional SSL Flow params (kept at defaults)
    ssl_touch_tolerance: float = 0.003
    ssl_body_tolerance: float = 0.003
    min_pbema_distance: float = 0.004
    lookback_candles: int = 5

    # Exit management (baseline)
    use_partial: bool = True
    partial_trigger: float = 0.40
    partial_fraction: float = 0.50
    use_dynamic_pbema_tp: bool = True

    # Strategy mode
    strategy_mode: str = "ssl_flow"

    def to_dict(self) -> dict:
        """Convert to dictionary for backtest config."""
        base = asdict(self)
        # Add BASELINE_CONFIG defaults
        base.update({
            'sl_validation_mode': 'off',
            'partial_rr_adjustment': False,
            'dynamic_tp_only_after_partial': False,
            'dynamic_tp_clamp_mode': 'none',
            'use_trailing': False,
            'momentum_tp_extension': True,
            'momentum_extension_threshold': 0.80,
            'momentum_extension_multiplier': 1.5,
            'use_progressive_partial': True,
            'partial_tranches': [
                {'trigger': 0.40, 'fraction': 0.33},
                {'trigger': 0.70, 'fraction': 0.50},
            ],
            'progressive_be_after_tranche': 1,
        })
        # Rename rr_ratio to rr for compatibility
        base['rr'] = base.pop('rr_ratio')
        # Rename fields
        base['rsi'] = base.pop('rsi_threshold')
        base['adx_min'] = base.pop('adx_threshold')
        return base

    def __hash__(self):
        """Hash for deduplication."""
        return hash((
            self.rsi_threshold, self.adx_threshold, self.rr_ratio,
            self.regime_adx_avg, self.at_active
        ))


@dataclass
class GridSearchResult:
    """Results from testing a single parameter combination."""
    config: GridConfig

    # Performance metrics
    total_pnl: float
    trade_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float

    # R-multiple metrics
    expected_r: float  # E[R]
    r_multiples: List[float]

    # Composite score
    robust_score: float

    # Statistical validation
    t_statistic: float = 0.0
    p_value: float = 1.0
    sharpe_ci_lower: float = 0.0
    sharpe_ci_upper: float = 0.0
    is_significant: bool = False

    # Robustness
    robustness_score: float = 0.0
    stable_neighbors_pct: float = 0.0

    # Metadata
    phase: str = 'coarse'  # 'coarse', 'fine', 'robustness'
    timestamp: str = ''

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert config to dict
        result['config'] = self.config.to_dict()
        return result


# ==========================================
# PARAMETER GRIDS
# ==========================================

COARSE_GRID = {
    'rsi_threshold': [60, 65, 70, 75],
    'adx_threshold': [20, 25, 30],
    'rr_ratio': [1.5, 2.0, 2.5, 3.0],
    'regime_adx_avg': [15, 20, 25],
    'at_active': [True],  # Mandatory
}

# Fine grid: ±10% around coarse values
FINE_GRID_MARGIN = 0.10


# ==========================================
# GRID SEARCH OPTIMIZER
# ==========================================

class GridSearchOptimizer:
    """
    Hierarchical grid search optimizer for SSL Flow strategy.

    Three-phase approach:
    1. Coarse grid: Fast screening of parameter space
    2. Fine grid: Refinement around top performers
    3. Robustness: Sensitivity testing and validation

    Statistical validation:
    - Bonferroni correction for multiple comparisons
    - Bootstrap confidence intervals
    - t-statistics for significance testing

    Robustness testing:
    - Parameter perturbation (±10%)
    - Neighbor stability analysis
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        verbose: bool = True,
        max_workers: int = None,
    ):
        """
        Initialize grid search optimizer.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '15m', '1h')
            data: Historical OHLCV DataFrame with indicators
            verbose: Print progress messages
            max_workers: Max parallel workers (None = auto)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = data
        self.verbose = verbose
        self.max_workers = max_workers or max(1, (os.cpu_count() or 1) - 1)

        # Results storage
        self.coarse_results: List[GridSearchResult] = []
        self.fine_results: List[GridSearchResult] = []
        self.robustness_results: Dict[str, Any] = {}

        # Run metadata
        self.run_id = f"grid_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = os.path.join(DATA_DIR, 'grid_search_runs', self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)

    def log(self, msg: str):
        """Log message if verbose."""
        if self.verbose:
            print(msg)

    def generate_coarse_configs(self) -> List[GridConfig]:
        """Generate coarse grid configurations."""
        configs = []

        for rsi, adx, rr, regime_adx in itertools.product(
            COARSE_GRID['rsi_threshold'],
            COARSE_GRID['adx_threshold'],
            COARSE_GRID['rr_ratio'],
            COARSE_GRID['regime_adx_avg'],
        ):
            config = GridConfig(
                rsi_threshold=rsi,
                adx_threshold=adx,
                rr_ratio=rr,
                regime_adx_avg=regime_adx,
                at_active=True,
            )
            configs.append(config)

        # Deduplicate
        configs = list(set(configs))
        return configs

    def generate_fine_configs(self, top_coarse: List[GridSearchResult], top_k: int = 5) -> List[GridConfig]:
        """
        Generate fine grid around top performers.

        Args:
            top_coarse: Top results from coarse search
            top_k: Number of top configs to refine

        Returns:
            List of fine grid configs
        """
        configs = []

        for result in top_coarse[:top_k]:
            base = result.config

            # Generate variations (±10%)
            rsi_vals = [
                int(base.rsi_threshold * (1 - FINE_GRID_MARGIN)),
                base.rsi_threshold,
                int(base.rsi_threshold * (1 + FINE_GRID_MARGIN)),
            ]
            adx_vals = [
                base.adx_threshold * (1 - FINE_GRID_MARGIN),
                base.adx_threshold,
                base.adx_threshold * (1 + FINE_GRID_MARGIN),
            ]
            rr_vals = [
                round(base.rr_ratio * (1 - FINE_GRID_MARGIN), 2),
                base.rr_ratio,
                round(base.rr_ratio * (1 + FINE_GRID_MARGIN), 2),
            ]
            regime_vals = [
                base.regime_adx_avg * (1 - FINE_GRID_MARGIN),
                base.regime_adx_avg,
                base.regime_adx_avg * (1 + FINE_GRID_MARGIN),
            ]

            # Clamp values to reasonable ranges
            rsi_vals = [max(30, min(80, v)) for v in rsi_vals]
            adx_vals = [max(10, min(40, v)) for v in adx_vals]
            rr_vals = [max(1.0, min(5.0, v)) for v in rr_vals]
            regime_vals = [max(10, min(35, v)) for v in regime_vals]

            for rsi, adx, rr, regime_adx in itertools.product(
                rsi_vals, adx_vals, rr_vals, regime_vals
            ):
                config = GridConfig(
                    rsi_threshold=int(rsi),
                    adx_threshold=round(adx, 1),
                    rr_ratio=round(rr, 2),
                    regime_adx_avg=round(regime_adx, 1),
                    at_active=True,
                )
                configs.append(config)

        # Deduplicate
        configs = list(set(configs))
        return configs

    def evaluate_config(self, config: GridConfig, phase: str = 'coarse') -> Optional[GridSearchResult]:
        """
        Evaluate a single configuration.

        Args:
            config: Parameter configuration to test
            phase: Search phase ('coarse', 'fine', 'robustness')

        Returns:
            GridSearchResult or None if evaluation fails
        """
        try:
            # Convert to backtest format
            backtest_config = config.to_dict()

            # Run backtest simulation
            net_pnl, trade_count, trade_pnls, r_multiples = _score_config_for_stream(
                self.data, self.symbol, self.timeframe, backtest_config
            )

            # Not enough trades
            if trade_count < MIN_TRADES_THRESHOLD:
                return None

            # Calculate metrics
            # Note: trade_pnls may have multiple entries per trade due to partial closes
            # Use len(trade_pnls) for consistent win_rate calculation
            wins = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p < 0]
            total_closes = len(trade_pnls)

            win_rate = len(wins) / total_closes if total_closes > 0 else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0

            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            # Sharpe ratio (annualized)
            if len(trade_pnls) >= 2:
                returns_std = np.std(trade_pnls)
                sharpe = (np.mean(trade_pnls) / returns_std) * np.sqrt(252) if returns_std > 0 else 0
            else:
                sharpe = 0

            # Max drawdown
            cumulative = np.cumsum(trade_pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
            max_dd_pct = max_dd / np.max(running_max) if np.max(running_max) > 0 else 0

            # E[R]
            expected_r = np.mean(r_multiples) if r_multiples else 0

            # Calculate robust score
            robust_score = self.calculate_robust_score(
                sharpe=sharpe,
                profit_factor=profit_factor,
                max_dd_pct=max_dd_pct,
                trade_count=trade_count,
                config=config,
            )

            return GridSearchResult(
                config=config,
                total_pnl=net_pnl,
                trade_count=trade_count,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe,
                max_drawdown_pct=max_dd_pct,
                expected_r=expected_r,
                r_multiples=r_multiples,
                robust_score=robust_score,
                phase=phase,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            self.log(f"[GRID] Error evaluating config: {e}")
            return None

    def calculate_robust_score(
        self,
        sharpe: float,
        profit_factor: float,
        max_dd_pct: float,
        trade_count: int,
        config: GridConfig,
    ) -> float:
        """
        Calculate robust composite score with penalties.

        Formula:
            base_score = sharpe * 0.30 +
                         profit_factor * 0.20 +
                         (1 - max_dd) * 0.20 +
                         stability * 0.20 +
                         trade_freq * 0.10

            penalties = trade_penalty + complexity_penalty

            final_score = base_score - penalties

        Args:
            sharpe: Sharpe ratio
            profit_factor: Gross profit / gross loss
            max_dd_pct: Max drawdown percentage
            trade_count: Number of trades
            config: Parameter configuration

        Returns:
            Composite score (higher = better)
        """
        # Base metrics with weights
        sharpe_component = sharpe * SCORE_WEIGHTS['sharpe']
        pf_component = min(profit_factor, 5.0) * SCORE_WEIGHTS['profit_factor']  # Cap at 5
        dd_component = (1 - min(max_dd_pct, 1.0)) * SCORE_WEIGHTS['drawdown']

        # Stability: consistent returns (use profit_factor as proxy)
        stability = min(profit_factor / 2.0, 1.0)
        stability_component = stability * SCORE_WEIGHTS['stability']

        # Trade frequency: normalized (50 trades = 100%)
        trade_freq = min(trade_count / 50.0, 1.0)
        trade_freq_component = trade_freq * SCORE_WEIGHTS['trade_frequency']

        base_score = (
            sharpe_component +
            pf_component +
            dd_component +
            stability_component +
            trade_freq_component
        )

        # Penalties
        trade_penalty = 0
        if trade_count < TRADE_PENALTY_THRESHOLD:
            # Progressive penalty: 0-50% based on how far below threshold
            penalty_pct = 1 - (trade_count / TRADE_PENALTY_THRESHOLD)
            trade_penalty = base_score * penalty_pct * 0.5  # Max 50% penalty

        # Complexity penalty: more filters = more overfitting risk
        active_filters = sum([
            config.rsi_threshold != 70,  # Non-default RSI
            config.adx_threshold != 25,  # Non-default ADX
            config.rr_ratio != 2.0,      # Non-default RR
            config.regime_adx_avg != 20, # Non-default regime
        ])
        complexity_penalty = active_filters * COMPLEXITY_PENALTY_PER_FILTER * base_score

        final_score = base_score - trade_penalty - complexity_penalty
        return final_score

    def run_coarse_search(self) -> List[GridSearchResult]:
        """
        Phase 1: Coarse grid search.

        Returns:
            Sorted list of results (best first)
        """
        self.log(f"\n{'='*70}")
        self.log(f"PHASE 1: COARSE GRID SEARCH")
        self.log(f"{'='*70}")

        configs = self.generate_coarse_configs()
        self.log(f"Generated {len(configs)} coarse grid configurations")

        results = []

        # Parallel evaluation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.evaluate_config, cfg, 'coarse'): cfg for cfg in configs}

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

                completed += 1
                if completed % 10 == 0:
                    self.log(f"Progress: {completed}/{len(configs)} ({completed/len(configs)*100:.1f}%)")

        # Sort by robust_score
        results.sort(key=lambda r: r.robust_score, reverse=True)

        self.log(f"\nCoarse search complete: {len(results)} valid configurations")
        if results:
            best = results[0]
            self.log(f"Best score: {best.robust_score:.4f} | Sharpe: {best.sharpe_ratio:.2f} | "
                    f"PnL: ${best.total_pnl:.2f} | Trades: {best.trade_count}")

        self.coarse_results = results
        return results

    def run_fine_search(self, top_k: int = 5) -> List[GridSearchResult]:
        """
        Phase 2: Fine grid search around top performers.

        Args:
            top_k: Number of top coarse configs to refine

        Returns:
            Sorted list of fine search results
        """
        if not self.coarse_results:
            raise ValueError("Must run coarse search first")

        self.log(f"\n{'='*70}")
        self.log(f"PHASE 2: FINE GRID SEARCH (Top {top_k})")
        self.log(f"{'='*70}")

        configs = self.generate_fine_configs(self.coarse_results, top_k=top_k)
        self.log(f"Generated {len(configs)} fine grid configurations")

        results = []

        # Parallel evaluation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.evaluate_config, cfg, 'fine'): cfg for cfg in configs}

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

                completed += 1
                if completed % 10 == 0:
                    self.log(f"Progress: {completed}/{len(configs)} ({completed/len(configs)*100:.1f}%)")

        # Sort by robust_score
        results.sort(key=lambda r: r.robust_score, reverse=True)

        self.log(f"\nFine search complete: {len(results)} valid configurations")
        if results:
            best = results[0]
            self.log(f"Best score: {best.robust_score:.4f} | Sharpe: {best.sharpe_ratio:.2f} | "
                    f"PnL: ${best.total_pnl:.2f} | Trades: {best.trade_count}")

        self.fine_results = results
        return results

    def run_statistical_validation(self, results: List[GridSearchResult]) -> List[GridSearchResult]:
        """
        Apply statistical validation to results.

        - Bonferroni correction for multiple comparisons
        - Bootstrap confidence intervals for Sharpe
        - t-statistics for significance

        Args:
            results: List of results to validate

        Returns:
            Results with updated statistical fields
        """
        self.log(f"\n{'='*70}")
        self.log(f"STATISTICAL VALIDATION")
        self.log(f"{'='*70}")

        n_tests = len(results)
        bonferroni_alpha = 0.05 / n_tests if n_tests > 0 else 0.05

        self.log(f"Bonferroni correction: alpha = {bonferroni_alpha:.6f} ({n_tests} tests)")

        for result in results:
            r_multiples = result.r_multiples

            if len(r_multiples) < 2:
                continue

            # t-statistic: test if E[R] > 0
            t_stat, p_value = stats.ttest_1samp(r_multiples, 0)
            result.t_statistic = t_stat
            result.p_value = p_value
            result.is_significant = p_value < bonferroni_alpha

            # Bootstrap confidence interval for Sharpe
            if len(r_multiples) >= 10:
                sharpe_boots = []
                for _ in range(1000):
                    sample = np.random.choice(r_multiples, size=len(r_multiples), replace=True)
                    if np.std(sample) > 0:
                        sharpe_boot = np.mean(sample) / np.std(sample) * np.sqrt(252)
                        sharpe_boots.append(sharpe_boot)

                if sharpe_boots:
                    result.sharpe_ci_lower = np.percentile(sharpe_boots, 2.5)
                    result.sharpe_ci_upper = np.percentile(sharpe_boots, 97.5)

        significant_count = sum(1 for r in results if r.is_significant)
        self.log(f"Significant results: {significant_count}/{len(results)} "
                f"({significant_count/len(results)*100:.1f}%)")

        return results

    def run_robustness_test(self, config: GridConfig, perturbation: float = 0.10) -> dict:
        """
        Test parameter robustness via perturbation.

        Perturbs each parameter by ±perturbation and checks if
        neighbors are still profitable (stable edge).

        Args:
            config: Base configuration
            perturbation: Perturbation fraction (default 10%)

        Returns:
            Dict with robustness metrics
        """
        neighbors = []

        # Perturb each parameter
        for param in ['rsi_threshold', 'adx_threshold', 'rr_ratio', 'regime_adx_avg']:
            base_val = getattr(config, param)

            for direction in [-1, 1]:
                perturbed_val = base_val * (1 + direction * perturbation)

                # Create perturbed config
                perturbed = GridConfig(
                    rsi_threshold=int(perturbed_val) if param == 'rsi_threshold' else config.rsi_threshold,
                    adx_threshold=round(perturbed_val, 1) if param == 'adx_threshold' else config.adx_threshold,
                    rr_ratio=round(perturbed_val, 2) if param == 'rr_ratio' else config.rr_ratio,
                    regime_adx_avg=round(perturbed_val, 1) if param == 'regime_adx_avg' else config.regime_adx_avg,
                    at_active=True,
                )

                result = self.evaluate_config(perturbed, phase='robustness')
                if result:
                    neighbors.append({
                        'param': param,
                        'direction': 'increase' if direction > 0 else 'decrease',
                        'pnl': result.total_pnl,
                        'trades': result.trade_count,
                        'sharpe': result.sharpe_ratio,
                        'profitable': result.total_pnl > 0,
                    })

        # Calculate stability
        profitable_neighbors = sum(1 for n in neighbors if n['profitable'])
        total_neighbors = len(neighbors)
        stability_pct = profitable_neighbors / total_neighbors if total_neighbors > 0 else 0

        return {
            'neighbors': neighbors,
            'total_neighbors': total_neighbors,
            'profitable_neighbors': profitable_neighbors,
            'stability_pct': stability_pct,
        }

    def run_robustness_analysis(self, top_k: int = 3) -> dict:
        """
        Phase 3: Robustness analysis on top configs.

        Args:
            top_k: Number of top configs to test

        Returns:
            Dict with robustness results
        """
        if not self.fine_results:
            raise ValueError("Must run fine search first")

        self.log(f"\n{'='*70}")
        self.log(f"PHASE 3: ROBUSTNESS ANALYSIS (Top {top_k})")
        self.log(f"{'='*70}")

        robustness_results = {}

        for i, result in enumerate(self.fine_results[:top_k], 1):
            self.log(f"\nTesting config #{i}...")
            self.log(f"  Base: Sharpe={result.sharpe_ratio:.2f}, PnL=${result.total_pnl:.2f}")

            rob = self.run_robustness_test(result.config)

            self.log(f"  Robustness: {rob['profitable_neighbors']}/{rob['total_neighbors']} "
                    f"neighbors profitable ({rob['stability_pct']*100:.1f}%)")

            # Update result
            result.robustness_score = rob['stability_pct']
            result.stable_neighbors_pct = rob['stability_pct'] * 100

            robustness_results[f'config_{i}'] = {
                'config': result.config.to_dict(),
                'base_metrics': {
                    'sharpe': result.sharpe_ratio,
                    'pnl': result.total_pnl,
                    'trades': result.trade_count,
                },
                'robustness': rob,
            }

        self.robustness_results = robustness_results
        return robustness_results

    def save_results(self):
        """Save all results to JSON files."""
        self.log(f"\n{'='*70}")
        self.log(f"SAVING RESULTS")
        self.log(f"{'='*70}")

        # All results
        all_results = {
            'run_id': self.run_id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': datetime.now().isoformat(),
            'coarse_results': [r.to_dict() for r in self.coarse_results],
            'fine_results': [r.to_dict() for r in self.fine_results],
            'robustness_results': self.robustness_results,
        }

        results_path = os.path.join(self.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        self.log(f"Results saved: {results_path}")

        # Top 10 report
        top_10 = self.fine_results[:10] if self.fine_results else self.coarse_results[:10]
        self.write_top_10_report(top_10)

        # Significance report
        self.write_significance_report(top_10)

    def write_top_10_report(self, results: List[GridSearchResult]):
        """Write human-readable top 10 report."""
        report_path = os.path.join(self.output_dir, 'top_10.txt')

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"TOP 10 PARAMETER COMBINATIONS\n")
            f.write(f"Symbol: {self.symbol} | Timeframe: {self.timeframe}\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write("="*70 + "\n\n")

            for i, result in enumerate(results[:10], 1):
                cfg = result.config
                f.write(f"#{i} | Score: {result.robust_score:.4f}\n")
                f.write("-" * 70 + "\n")
                f.write(f"  Parameters:\n")
                f.write(f"    RSI Threshold:      {cfg.rsi_threshold}\n")
                f.write(f"    ADX Threshold:      {cfg.adx_threshold}\n")
                f.write(f"    RR Ratio:           {cfg.rr_ratio}\n")
                f.write(f"    Regime ADX Avg:     {cfg.regime_adx_avg}\n")
                f.write(f"    AlphaTrend Active:  {cfg.at_active}\n")
                f.write(f"\n")
                f.write(f"  Performance:\n")
                f.write(f"    Total PnL:          ${result.total_pnl:.2f}\n")
                f.write(f"    Trade Count:        {result.trade_count}\n")
                f.write(f"    Win Rate:           {result.win_rate*100:.1f}%\n")
                f.write(f"    Profit Factor:      {result.profit_factor:.2f}\n")
                f.write(f"    Sharpe Ratio:       {result.sharpe_ratio:.2f}\n")
                f.write(f"    Max Drawdown:       {result.max_drawdown_pct*100:.1f}%\n")
                f.write(f"    E[R]:               {result.expected_r:.3f}\n")
                f.write(f"\n")

                if result.is_significant:
                    f.write(f"  Statistical Validation:\n")
                    f.write(f"    Significant:        YES (p={result.p_value:.6f})\n")
                    f.write(f"    t-statistic:        {result.t_statistic:.2f}\n")
                    if result.sharpe_ci_lower != 0:
                        f.write(f"    Sharpe 95% CI:      [{result.sharpe_ci_lower:.2f}, {result.sharpe_ci_upper:.2f}]\n")
                    f.write(f"\n")

                if result.robustness_score > 0:
                    f.write(f"  Robustness:\n")
                    f.write(f"    Stable Neighbors:   {result.stable_neighbors_pct:.1f}%\n")
                    f.write(f"\n")

                f.write("\n")

        self.log(f"Top 10 report saved: {report_path}")

    def write_significance_report(self, results: List[GridSearchResult]):
        """Write statistical significance report."""
        report_path = os.path.join(self.output_dir, 'significance_report.txt')

        significant = [r for r in results if r.is_significant]

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"STATISTICAL SIGNIFICANCE REPORT\n")
            f.write(f"Symbol: {self.symbol} | Timeframe: {self.timeframe}\n")
            f.write("="*70 + "\n\n")

            f.write(f"Total Configurations Tested: {len(self.coarse_results) + len(self.fine_results)}\n")
            f.write(f"Statistically Significant:   {len(significant)} ({len(significant)/max(len(results), 1)*100:.1f}%)\n")
            f.write(f"\n")
            f.write(f"Bonferroni Correction Applied: Yes\n")
            f.write(f"Significance Level (alpha):     0.05 / {len(results)} = {0.05/max(len(results), 1):.6f}\n")
            f.write(f"\n")

            if significant:
                f.write("="*70 + "\n")
                f.write("SIGNIFICANT CONFIGURATIONS\n")
                f.write("="*70 + "\n\n")

                for i, result in enumerate(significant, 1):
                    cfg = result.config
                    f.write(f"#{i} | p-value: {result.p_value:.6f} | t-stat: {result.t_statistic:.2f}\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"  RSI={cfg.rsi_threshold}, ADX={cfg.adx_threshold}, "
                           f"RR={cfg.rr_ratio}, RegimeADX={cfg.regime_adx_avg}\n")
                    f.write(f"  PnL: ${result.total_pnl:.2f} | E[R]: {result.expected_r:.3f} | "
                           f"Trades: {result.trade_count}\n")
                    if result.sharpe_ci_lower != 0:
                        f.write(f"  Sharpe: {result.sharpe_ratio:.2f} "
                               f"[95% CI: {result.sharpe_ci_lower:.2f}, {result.sharpe_ci_upper:.2f}]\n")
                    f.write(f"\n")
            else:
                f.write("No configurations reached statistical significance.\n")
                f.write("This suggests potential overfitting or insufficient data.\n")

        self.log(f"Significance report saved: {report_path}")

    def run_full_search(self, quick: bool = False, robust: bool = False) -> dict:
        """
        Run complete hierarchical grid search.

        Args:
            quick: If True, skip fine search (coarse only)
            robust: If True, include robustness analysis

        Returns:
            Dict with all results
        """
        start_time = time.time()

        self.log(f"\n{'='*70}")
        self.log(f"GRID SEARCH OPTIMIZER")
        self.log(f"{'='*70}")
        self.log(f"Symbol: {self.symbol}")
        self.log(f"Timeframe: {self.timeframe}")
        self.log(f"Data: {len(self.data)} candles")
        self.log(f"Mode: {'QUICK' if quick else 'FULL'}")
        self.log(f"Robustness: {'YES' if robust else 'NO'}")
        self.log(f"{'='*70}")

        # Phase 1: Coarse search
        coarse_results = self.run_coarse_search()

        if not quick and coarse_results:
            # Phase 2: Fine search
            fine_results = self.run_fine_search(top_k=5)

            # Statistical validation
            if fine_results:
                fine_results = self.run_statistical_validation(fine_results)
                self.fine_results = fine_results

            # Phase 3: Robustness (optional)
            if robust and fine_results:
                self.run_robustness_analysis(top_k=3)

        # Save results
        self.save_results()

        elapsed = time.time() - start_time
        self.log(f"\n{'='*70}")
        self.log(f"GRID SEARCH COMPLETE")
        self.log(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        self.log(f"Results saved: {self.output_dir}")
        self.log(f"{'='*70}")

        return {
            'run_id': self.run_id,
            'output_dir': self.output_dir,
            'coarse_results': self.coarse_results,
            'fine_results': self.fine_results,
            'robustness_results': self.robustness_results,
            'elapsed_seconds': elapsed,
        }


# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    'GridSearchOptimizer',
    'GridConfig',
    'GridSearchResult',
    'COARSE_GRID',
    'MIN_TRADES_THRESHOLD',
]
