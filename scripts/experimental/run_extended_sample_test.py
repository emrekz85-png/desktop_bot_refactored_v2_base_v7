#!/usr/bin/env python3
"""
Extended Sample Size Test - Priority 1 Recommendation

This script implements the Priority 1 recommendation from the hedge fund analysis:
"Extend Sample Size - Run backtest on 2+ years of data, target 100+ trades"

The goal is to achieve statistical significance:
- Minimum 100 trades for meaningful confidence intervals
- Multi-year data to capture different market regimes
- Calculate Wilson score confidence intervals
- Determine if edge is statistically significant

Usage:
    python run_extended_sample_test.py                    # Default: 2 years
    python run_extended_sample_test.py --years 3          # 3 years of data
    python run_extended_sample_test.py --target-trades 150  # Target 150+ trades
    python run_extended_sample_test.py --all-symbols      # Include all symbols

Statistical Significance Thresholds:
    - 95% CI lower bound > 50% win rate = Statistically significant edge
    - 100+ trades = Reliable sample size
    - E[R] > 0 with 95% confidence = Deployable strategy
"""

import sys
import os
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def wilson_score_interval(wins: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for binomial proportion.

    This is more accurate than normal approximation for small samples.

    Args:
        wins: Number of successes
        total: Total trials
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) as proportions
    """
    if total == 0:
        return (0.0, 1.0)

    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    p_hat = wins / total

    denominator = 1 + z**2 / total
    centre_adjusted_probability = p_hat + z**2 / (2 * total)
    adjusted_std_dev = math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)

    lower = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper = (centre_adjusted_probability + z * adjusted_std_dev) / denominator

    return (max(0, lower), min(1, upper))


def calculate_required_sample_size(
    estimated_win_rate: float = 0.65,
    margin_of_error: float = 0.10,
    confidence: float = 0.95
) -> int:
    """
    Calculate required sample size for desired margin of error.

    Args:
        estimated_win_rate: Expected win rate (default 0.65 based on observations)
        margin_of_error: Desired margin of error (default 0.10 = 10%)
        confidence: Confidence level (default 0.95)

    Returns:
        Required sample size (number of trades)
    """
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    p = estimated_win_rate
    n = (z**2 * p * (1 - p)) / margin_of_error**2

    return int(math.ceil(n))


def analyze_statistical_significance(trades: List[Dict]) -> Dict:
    """
    Comprehensive statistical significance analysis.

    Args:
        trades: List of trade dictionaries with 'pnl' field

    Returns:
        Dictionary with statistical metrics
    """
    if not trades:
        return {"error": "No trades to analyze"}

    total = len(trades)
    wins = sum(1 for t in trades if float(t.get("pnl", 0)) > 0)
    losses = total - wins

    win_rate = wins / total if total > 0 else 0

    # Wilson score intervals at multiple confidence levels
    ci_90 = wilson_score_interval(wins, total, 0.90)
    ci_95 = wilson_score_interval(wins, total, 0.95)
    ci_99 = wilson_score_interval(wins, total, 0.99)

    # Calculate PnL statistics
    pnls = [float(t.get("pnl", 0)) for t in trades]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / total if total > 0 else 0

    winning_pnls = [p for p in pnls if p > 0]
    losing_pnls = [p for p in pnls if p <= 0]

    avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
    avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0

    # Calculate R-multiples
    r_multiples = [t.get("r_multiple", 0) for t in trades if t.get("r_multiple") is not None]
    avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0

    # Profit factor
    gross_profit = sum(winning_pnls)
    gross_loss = abs(sum(losing_pnls))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Expected value calculation
    expected_value = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Standard deviation of PnL
    if total > 1:
        variance = sum((p - avg_pnl)**2 for p in pnls) / (total - 1)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0

    # Sharpe-like ratio (assuming 0 risk-free rate)
    sharpe = (avg_pnl / std_dev) * math.sqrt(252) if std_dev > 0 else 0  # Annualized

    # Is edge statistically significant?
    edge_significant_90 = ci_90[0] > 0.50
    edge_significant_95 = ci_95[0] > 0.50
    edge_significant_99 = ci_99[0] > 0.50

    # Required trades for tighter CI
    required_for_5pct_margin = calculate_required_sample_size(win_rate, 0.05, 0.95)
    required_for_10pct_margin = calculate_required_sample_size(win_rate, 0.10, 0.95)

    return {
        "sample_size": {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "is_sufficient": total >= 100,
            "required_for_5pct_margin": required_for_5pct_margin,
            "required_for_10pct_margin": required_for_10pct_margin,
        },
        "confidence_intervals": {
            "ci_90": {"lower": ci_90[0], "upper": ci_90[1]},
            "ci_95": {"lower": ci_95[0], "upper": ci_95[1]},
            "ci_99": {"lower": ci_99[0], "upper": ci_99[1]},
        },
        "pnl_statistics": {
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expected_value": expected_value,
            "std_dev": std_dev,
            "sharpe_ratio": sharpe,
        },
        "r_multiple": {
            "avg_r": avg_r,
            "total_r": sum(r_multiples) if r_multiples else 0,
        },
        "statistical_significance": {
            "edge_significant_90": edge_significant_90,
            "edge_significant_95": edge_significant_95,
            "edge_significant_99": edge_significant_99,
            "verdict": "SIGNIFICANT" if edge_significant_95 else "NOT SIGNIFICANT",
        },
    }


def analyze_by_regime(trades: List[Dict], window_results: List[Dict]) -> Dict:
    """
    Analyze performance by market regime.

    Args:
        trades: All trades
        window_results: Window results with regime info

    Returns:
        Regime breakdown analysis
    """
    regime_trades = defaultdict(list)

    # Map trades to regimes based on window
    for window in window_results:
        regime = window.get("regime", "UNKNOWN")
        w_start = window.get("trade_start", "")
        w_end = window.get("trade_end", "")

        for trade in trades:
            trade_time = str(trade.get("open_time_utc", ""))[:10]
            if w_start <= trade_time < w_end:
                regime_trades[regime].append(trade)

    regime_analysis = {}
    for regime, rtrades in regime_trades.items():
        if rtrades:
            stats = analyze_statistical_significance(rtrades)
            regime_analysis[regime] = {
                "trades": len(rtrades),
                "win_rate": stats["sample_size"]["win_rate"],
                "pnl": stats["pnl_statistics"]["total_pnl"],
                "avg_r": stats["r_multiple"]["avg_r"],
                "ci_95": stats["confidence_intervals"]["ci_95"],
            }

    return regime_analysis


def analyze_by_symbol(trades: List[Dict]) -> Dict:
    """Analyze performance by symbol."""
    symbol_trades = defaultdict(list)

    for trade in trades:
        symbol = trade.get("symbol", "UNKNOWN")
        symbol_trades[symbol].append(trade)

    symbol_analysis = {}
    for symbol, strades in symbol_trades.items():
        if strades:
            stats = analyze_statistical_significance(strades)
            symbol_analysis[symbol] = {
                "trades": len(strades),
                "win_rate": stats["sample_size"]["win_rate"],
                "pnl": stats["pnl_statistics"]["total_pnl"],
                "avg_r": stats["r_multiple"]["avg_r"],
                "ci_95": stats["confidence_intervals"]["ci_95"],
                "significant": stats["statistical_significance"]["edge_significant_95"],
            }

    return symbol_analysis


def print_statistical_report(analysis: Dict, symbol_analysis: Dict, regime_analysis: Dict):
    """Print comprehensive statistical report."""

    print("\n" + "="*80)
    print("                    STATISTICAL SIGNIFICANCE REPORT")
    print("                    Priority 1 Recommendation Analysis")
    print("="*80)

    # Sample Size Section
    ss = analysis["sample_size"]
    print(f"\n📊 SAMPLE SIZE ANALYSIS")
    print("-"*40)
    print(f"   Total Trades:     {ss['total_trades']}")
    print(f"   Wins:             {ss['wins']} ({ss['win_rate']*100:.1f}%)")
    print(f"   Losses:           {ss['losses']} ({(1-ss['win_rate'])*100:.1f}%)")
    print(f"   Sufficient:       {'✅ YES' if ss['is_sufficient'] else '❌ NO'} (target: 100+)")

    if not ss['is_sufficient']:
        print(f"\n   ⚠️  Need {100 - ss['total_trades']} more trades for minimum sample size")

    print(f"\n   For 5% margin of error: Need {ss['required_for_5pct_margin']} trades")
    print(f"   For 10% margin of error: Need {ss['required_for_10pct_margin']} trades")

    # Confidence Intervals Section
    ci = analysis["confidence_intervals"]
    print(f"\n📈 WIN RATE CONFIDENCE INTERVALS")
    print("-"*40)
    print(f"   Observed Win Rate: {ss['win_rate']*100:.1f}%")
    print(f"   90% CI: [{ci['ci_90']['lower']*100:.1f}%, {ci['ci_90']['upper']*100:.1f}%]")
    print(f"   95% CI: [{ci['ci_95']['lower']*100:.1f}%, {ci['ci_95']['upper']*100:.1f}%]")
    print(f"   99% CI: [{ci['ci_99']['lower']*100:.1f}%, {ci['ci_99']['upper']*100:.1f}%]")

    # Statistical Significance
    sig = analysis["statistical_significance"]
    print(f"\n🎯 STATISTICAL SIGNIFICANCE")
    print("-"*40)
    print(f"   Edge significant at 90%: {'✅ YES' if sig['edge_significant_90'] else '❌ NO'}")
    print(f"   Edge significant at 95%: {'✅ YES' if sig['edge_significant_95'] else '❌ NO'}")
    print(f"   Edge significant at 99%: {'✅ YES' if sig['edge_significant_99'] else '❌ NO'}")
    print(f"\n   VERDICT: {sig['verdict']}")

    if sig['edge_significant_95']:
        print("   → 95% CI lower bound > 50% = Edge exists with high confidence")
    else:
        print("   → Cannot rule out that true win rate is <= 50%")

    # PnL Statistics
    pnl = analysis["pnl_statistics"]
    print(f"\n💰 PnL STATISTICS")
    print("-"*40)
    print(f"   Total PnL:          ${pnl['total_pnl']:+,.2f}")
    print(f"   Avg PnL/Trade:      ${pnl['avg_pnl_per_trade']:+,.2f}")
    print(f"   Avg Win:            ${pnl['avg_win']:+,.2f}")
    print(f"   Avg Loss:           ${pnl['avg_loss']:+,.2f}")
    print(f"   Profit Factor:      {pnl['profit_factor']:.2f}")
    print(f"   Expected Value:     ${pnl['expected_value']:+,.2f}")
    print(f"   Std Dev:            ${pnl['std_dev']:.2f}")
    print(f"   Sharpe Ratio:       {pnl['sharpe_ratio']:.2f}")

    # R-Multiple
    r = analysis["r_multiple"]
    print(f"\n📐 R-MULTIPLE ANALYSIS")
    print("-"*40)
    print(f"   Average R:          {r['avg_r']:+.2f}")
    print(f"   Total R:            {r['total_r']:+.2f}")

    # Symbol Breakdown
    print(f"\n🪙 PERFORMANCE BY SYMBOL")
    print("-"*40)
    print(f"   {'Symbol':<12} {'Trades':>7} {'Win%':>7} {'PnL':>10} {'Avg R':>7} {'Sig?':>6}")
    print(f"   {'-'*12} {'-'*7} {'-'*7} {'-'*10} {'-'*7} {'-'*6}")

    for symbol in sorted(symbol_analysis.keys()):
        data = symbol_analysis[symbol]
        sig_mark = "✅" if data['significant'] else "❌"
        print(f"   {symbol:<12} {data['trades']:>7} {data['win_rate']*100:>6.1f}% ${data['pnl']:>9.2f} {data['avg_r']:>+6.2f} {sig_mark:>6}")

    # Regime Breakdown
    if regime_analysis:
        print(f"\n📊 PERFORMANCE BY REGIME")
        print("-"*40)
        print(f"   {'Regime':<15} {'Trades':>7} {'Win%':>7} {'PnL':>10} {'95% CI':>20}")
        print(f"   {'-'*15} {'-'*7} {'-'*7} {'-'*10} {'-'*20}")

        for regime in ["TRENDING", "RANGING", "VOLATILE", "TRANSITIONAL", "UNKNOWN"]:
            if regime in regime_analysis:
                data = regime_analysis[regime]
                ci_str = f"[{data['ci_95']['lower']*100:.0f}%-{data['ci_95']['upper']*100:.0f}%]"
                print(f"   {regime:<15} {data['trades']:>7} {data['win_rate']*100:>6.1f}% ${data['pnl']:>9.2f} {ci_str:>20}")

    # Final Recommendation
    print(f"\n" + "="*80)
    print("                         RECOMMENDATION")
    print("="*80)

    if ss['total_trades'] >= 100 and sig['edge_significant_95']:
        print("""
   ✅ SAMPLE SIZE SUFFICIENT & EDGE STATISTICALLY SIGNIFICANT

   The strategy shows a statistically significant edge at 95% confidence.
   Proceed to Priority 2 validation (paper trading) with confidence.

   Next Steps:
   1. Run 90-day paper trading to validate live execution
   2. Monitor for regime dependency (check TRENDING vs RANGING performance)
   3. Implement position sizing based on Kelly criterion
        """)
    elif ss['total_trades'] >= 100:
        print("""
   ⚠️  SAMPLE SIZE SUFFICIENT BUT EDGE NOT STATISTICALLY SIGNIFICANT

   While we have enough trades, the 95% confidence interval includes
   the possibility of no edge (win rate <= 50%).

   Recommendations:
   1. Continue testing to narrow confidence interval
   2. Investigate regime-specific performance
   3. Consider strategy modifications before live deployment
        """)
    else:
        needed = 100 - ss['total_trades']
        print(f"""
   ❌ INSUFFICIENT SAMPLE SIZE

   Need {needed} more trades to reach minimum threshold of 100.

   Options:
   1. Extend test period further
   2. Add more symbols (if edge is universal)
   3. Reduce filter strictness to increase trade frequency

   Current projection: At ~4 trades/month, need {needed/4:.0f} more months
        """)

    print("="*80 + "\n")


def run_extended_sample_test(
    years: int = 2,
    target_trades: int = 100,
    symbols: List[str] = None,
    timeframes: List[str] = None,
    all_symbols: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Run extended sample size test.

    Args:
        years: Number of years of data to test
        target_trades: Minimum target number of trades
        symbols: Symbols to test (default: BTC, ETH, LINK)
        timeframes: Timeframes to test (default: 5m, 15m, 1h)
        all_symbols: Use all configured symbols
        verbose: Print detailed output

    Returns:
        Complete test results with statistical analysis
    """
    from runners.rolling_wf import run_rolling_walkforward
    from core import SYMBOLS, TIMEFRAMES, BASELINE_CONFIG, DATA_DIR

    # Default to recommended symbols
    if symbols is None:
        if all_symbols:
            symbols = SYMBOLS
        else:
            symbols = ["BTCUSDT", "ETHUSDT", "LINKUSDT"]

    if timeframes is None:
        timeframes = ["5m", "15m", "1h"]

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    # Adjust start date if data might not be available
    # Binance Futures launched in 2019, so cap at 2020
    min_date = datetime(2020, 1, 1)
    if start_date < min_date:
        start_date = min_date
        actual_years = (end_date - start_date).days / 365
        print(f"⚠️  Adjusted start date to {start_date.strftime('%Y-%m-%d')} (data availability)")
        print(f"   Actual test period: {actual_years:.1f} years")

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print("\n" + "="*80)
    print("             EXTENDED SAMPLE SIZE TEST")
    print("             Priority 1 Recommendation Implementation")
    print("="*80)
    print(f"\n   Target:     {target_trades}+ trades")
    print(f"   Period:     {start_str} → {end_str} ({years} years)")
    print(f"   Symbols:    {', '.join(symbols)}")
    print(f"   Timeframes: {', '.join(timeframes)}")
    print("="*80 + "\n")

    # Run the backtest
    print("🔄 Running extended backtest...")
    print("   This may take 10-30 minutes depending on data size.\n")

    result = run_rolling_walkforward(
        symbols=symbols,
        timeframes=timeframes,
        mode="weekly",
        lookback_days=30,
        forward_days=7,
        start_date=start_str,
        end_date=end_str,
        verbose=verbose,
    )

    # Extract trades and analyze
    trades = result.get("trades", [])
    window_results = result.get("window_results", [])
    metrics = result.get("metrics", {})

    print(f"\n✅ Backtest complete. Total trades: {len(trades)}")

    # Run statistical analysis
    print("\n🔬 Running statistical significance analysis...")

    analysis = analyze_statistical_significance(trades)
    symbol_analysis = analyze_by_symbol(trades)
    regime_analysis = analyze_by_regime(trades, window_results)

    # Print report
    print_statistical_report(analysis, symbol_analysis, regime_analysis)

    # Save results
    output_dir = result.get("output_dir", os.path.join(DATA_DIR, "extended_sample_test"))
    os.makedirs(output_dir, exist_ok=True)

    full_results = {
        "test_config": {
            "years": years,
            "target_trades": target_trades,
            "symbols": symbols,
            "timeframes": timeframes,
            "start_date": start_str,
            "end_date": end_str,
        },
        "backtest_metrics": metrics,
        "statistical_analysis": analysis,
        "symbol_analysis": symbol_analysis,
        "regime_analysis": regime_analysis,
        "recommendation": analysis["statistical_significance"]["verdict"],
    }

    results_path = os.path.join(output_dir, "statistical_analysis.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"💾 Results saved to: {results_path}")

    return full_results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extended Sample Size Test - Priority 1 Recommendation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_extended_sample_test.py                    # Default 2 years
    python run_extended_sample_test.py --years 3          # 3 years
    python run_extended_sample_test.py --all-symbols      # All symbols
    python run_extended_sample_test.py --target-trades 150
        """
    )

    parser.add_argument('--years', type=int, default=2,
                        help='Number of years of historical data (default: 2)')
    parser.add_argument('--target-trades', type=int, default=100,
                        help='Target minimum number of trades (default: 100)')
    parser.add_argument('--symbols', nargs='+',
                        help='Specific symbols to test (default: BTC, ETH, LINK)')
    parser.add_argument('--timeframes', nargs='+',
                        help='Specific timeframes to test (default: 5m, 15m, 1h)')
    parser.add_argument('--all-symbols', action='store_true',
                        help='Use all configured symbols')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    result = run_extended_sample_test(
        years=args.years,
        target_trades=args.target_trades,
        symbols=args.symbols,
        timeframes=args.timeframes,
        all_symbols=args.all_symbols,
        verbose=not args.quiet,
    )

    # Exit with appropriate code
    if result.get("recommendation") == "SIGNIFICANT":
        print("\n✅ Edge is statistically significant. Ready for Phase 2 validation.")
        sys.exit(0)
    else:
        print("\n⚠️  Edge not confirmed. See report for recommendations.")
        sys.exit(1)


if __name__ == "__main__":
    main()
