#!/usr/bin/env python3
"""
AlphaTrend Correlation Analysis
Deep statistical analysis to determine optimal AT integration

This script analyzes the relationship between AlphaTrend scores/states
and trade outcomes to answer:
1. Do high AT scores predict winning trades?
2. What's the optimal AT threshold?
3. Binary vs Score vs Off mode - which is best?
4. Regime-specific effectiveness
5. False signal patterns

Usage:
    python scripts/analyze_at_correlation.py ultra_minimal_trades_2025-01-01_2025-12-31.json
"""

import sys
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple
import statistics


class ATCorrelationAnalyzer:
    """Deep analysis of AlphaTrend correlation with trade outcomes."""

    def __init__(self, trades: List[Dict]):
        """
        Initialize analyzer with trade data.

        Args:
            trades: List of trade dictionaries with AT metadata
        """
        self.trades = trades
        self.total_trades = len(trades)
        self.wins = [t for t in trades if t.get("is_win", False)]
        self.losses = [t for t in trades if not t.get("is_win", False)]
        self.overall_win_rate = len(self.wins) / self.total_trades if self.total_trades > 0 else 0

    def print_header(self):
        """Print analysis header."""
        print("\n" + "="*80)
        print("AlphaTrend Correlation Analysis")
        print("="*80)
        print(f"\nTotal Trades: {self.total_trades}")
        print(f"Wins: {len(self.wins)} ({len(self.wins)/self.total_trades*100:.1f}%)")
        print(f"Losses: {len(self.losses)} ({len(self.losses)/self.total_trades*100:.1f}%)")
        print(f"Overall Win Rate: {self.overall_win_rate*100:.1f}%")
        print("="*80)

    def analyze_by_at_dominant(self):
        """Analyze win rate by AT buyers/sellers dominant state."""
        print("\n" + "-"*80)
        print("ANALYSIS 1: AT Dominant State vs Win Rate")
        print("-"*80)

        categories = {
            "Buyers Dominant": [t for t in self.trades if t.get("at_buyers_dominant") and t.get("type") == "LONG"],
            "Sellers Dominant": [t for t in self.trades if t.get("at_sellers_dominant") and t.get("type") == "SHORT"],
            "Not Aligned (LONG)": [t for t in self.trades if not t.get("at_buyers_dominant") and t.get("type") == "LONG"],
            "Not Aligned (SHORT)": [t for t in self.trades if not t.get("at_sellers_dominant") and t.get("type") == "SHORT"],
        }

        print(f"\n{'Category':<25} {'Trades':<10} {'Win Rate':<12} {'vs Overall':<15}")
        print("-"*70)

        for category, trades_subset in categories.items():
            if len(trades_subset) == 0:
                continue

            wins = sum(1 for t in trades_subset if t.get("is_win"))
            wr = wins / len(trades_subset) if len(trades_subset) > 0 else 0
            diff = (wr - self.overall_win_rate) * 100

            diff_str = f"{diff:+.1f}pp"
            status = "✅" if diff > 5 else "⚠️" if diff > 0 else "❌"

            print(f"{category:<25} {len(trades_subset):<10} {wr*100:<12.1f}% {diff_str:<10} {status}")

        print("\n💡 Key Insights:")
        print("  - If 'Buyers Dominant' has higher WR → AT alignment helps (use binary mode)")
        print("  - If 'Not Aligned' has similar/better WR → AT doesn't help (disable or score mode)")
        print("  - Difference >10pp → Strong signal, worth filtering")
        print("  - Difference <5pp → Weak signal, don't filter")

    def analyze_by_at_flat(self):
        """Analyze win rate when AT is flat vs not flat."""
        print("\n" + "-"*80)
        print("ANALYSIS 2: AT Flat State vs Win Rate")
        print("-"*80)

        flat_trades = [t for t in self.trades if t.get("at_is_flat")]
        not_flat_trades = [t for t in self.trades if not t.get("at_is_flat")]

        def calc_wr(trades_list):
            if len(trades_list) == 0:
                return 0
            return sum(1 for t in trades_list if t.get("is_win")) / len(trades_list)

        flat_wr = calc_wr(flat_trades)
        not_flat_wr = calc_wr(not_flat_trades)

        print(f"\n{'State':<20} {'Trades':<10} {'Win Rate':<12} {'vs Overall':<15}")
        print("-"*65)

        for label, trades_subset, wr in [("AT Flat", flat_trades, flat_wr), ("AT Not Flat", not_flat_trades, not_flat_wr)]:
            diff = (wr - self.overall_win_rate) * 100
            diff_str = f"{diff:+.1f}pp"
            status = "✅" if diff > 5 else "⚠️" if diff > 0 else "❌"

            print(f"{label:<20} {len(trades_subset):<10} {wr*100:<12.1f}% {diff_str:<10} {status}")

        print("\n💡 Key Insights:")
        if flat_wr > not_flat_wr + 0.05:
            print("  ✅ AT Flat = BETTER win rate!")
            print("     → Flat markets favor SSL Flow reversals")
            print("     → Consider REVERSING the flat filter (take when flat, skip when trending)")
        elif abs(flat_wr - not_flat_wr) < 0.05:
            print("  ⚠️  AT Flat state doesn't matter")
            print("     → Don't filter based on flat/not-flat")
        else:
            print("  ❌ AT Flat = WORSE win rate")
            print("     → Current flat filter is correct (skip when flat)")

    def analyze_by_at_regime(self):
        """Analyze win rate by AT regime."""
        print("\n" + "-"*80)
        print("ANALYSIS 3: AT Regime vs Win Rate")
        print("-"*80)

        regime_categories = defaultdict(list)
        for trade in self.trades:
            regime = trade.get("at_regime", "unknown")
            regime_categories[regime].append(trade)

        print(f"\n{'Regime':<20} {'Trades':<10} {'Win Rate':<12} {'vs Overall':<15}")
        print("-"*65)

        for regime, trades_subset in sorted(regime_categories.items()):
            if len(trades_subset) == 0:
                continue

            wins = sum(1 for t in trades_subset if t.get("is_win"))
            wr = wins / len(trades_subset)
            diff = (wr - self.overall_win_rate) * 100
            diff_str = f"{diff:+.1f}pp"
            status = "✅" if diff > 5 else "⚠️" if diff > 0 else "❌"

            print(f"{regime:<20} {len(trades_subset):<10} {wr*100:<12.1f}% {diff_str:<10} {status}")

        print("\n💡 Key Insights:")
        print("  - Bullish/Bearish regime: Strong directional bias")
        print("  - Neutral regime: No clear direction (ranging)")
        print("  - If neutral > bullish/bearish WR → Strategy prefers ranging!")

    def analyze_score_thresholds(self):
        """Analyze win rate by AT score thresholds."""
        print("\n" + "-"*80)
        print("ANALYSIS 4: AT Score Threshold Optimization")
        print("-"*80)

        # Filter trades with AT scores
        scored_trades = [t for t in self.trades if t.get("at_score") is not None]

        if len(scored_trades) == 0:
            print("\n⚠️  No AT scores found in trades")
            print("   AT scores may not have been saved properly")
            return

        # Test different thresholds
        thresholds = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

        print(f"\n{'Threshold':<12} {'Trades (≥)':<15} {'Win Rate':<12} {'vs Overall':<15}")
        print("-"*60)

        for threshold in thresholds:
            above_threshold = [t for t in scored_trades if t.get("at_score", 0) >= threshold]

            if len(above_threshold) == 0:
                continue

            wins = sum(1 for t in above_threshold if t.get("is_win"))
            wr = wins / len(above_threshold)
            diff = (wr - self.overall_win_rate) * 100
            diff_str = f"{diff:+.1f}pp"
            status = "✅" if diff > 10 else "⚠️" if diff > 5 else "❌"

            print(f"≥ {threshold:<10.1f} {len(above_threshold):<15} {wr*100:<12.1f}% {diff_str:<10} {status}")

        print("\n💡 Key Insights:")
        print("  - Find threshold where WR increases >10pp")
        print("  - Balance: Higher threshold = fewer trades but higher WR")
        print("  - Optimal: Threshold where WR gain > trade loss")

    def analyze_binary_vs_score_simulation(self):
        """Simulate what would happen with binary mode vs score mode vs off."""
        print("\n" + "-"*80)
        print("ANALYSIS 5: Binary vs Score vs Off Mode Simulation")
        print("-"*80)

        # Simulate Binary Mode (AT must be aligned)
        binary_longs = [t for t in self.trades if t.get("type") == "LONG" and t.get("at_buyers_dominant")]
        binary_shorts = [t for t in self.trades if t.get("type") == "SHORT" and t.get("at_sellers_dominant")]
        binary_trades = binary_longs + binary_shorts

        # Score Mode (all trades, but could weight by score)
        score_trades = self.trades

        # Off Mode (all trades)
        off_trades = self.trades

        modes = [
            ("Off (AT disabled)", off_trades),
            ("Score (AT weights)", score_trades),
            ("Binary (AT blocks)", binary_trades),
        ]

        print(f"\n{'Mode':<25} {'Trades':<10} {'Win Rate':<12} {'Total PnL*':<12}")
        print("-"*65)

        for mode_name, trades_subset in modes:
            if len(trades_subset) == 0:
                continue

            wins = sum(1 for t in trades_subset if t.get("is_win"))
            wr = wins / len(trades_subset) if len(trades_subset) > 0 else 0

            # Estimate PnL (simplified)
            total_pnl = sum(t.get("pnl", 0) for t in trades_subset)

            status = "✅" if len(trades_subset) >= 30 else "⚠️" if len(trades_subset) >= 15 else "❌"

            print(f"{mode_name:<25} {len(trades_subset):<10} {wr*100:<12.1f}% ${total_pnl:<11.2f} {status}")

        print("\n* Estimated based on actual trade PnLs")
        print("\n💡 Key Insights:")
        print("  ✅ Binary Mode: Fewer trades, higher quality (if WR increases)")
        print("  ⚠️  Score Mode: All trades, weighted by AT score")
        print("  ❌ Off Mode: Most trades, no AT filtering")
        print("\nOptimal Choice:")
        print("  - If Binary WR > Off WR + 15pp → Use Binary")
        print("  - If Binary WR ≈ Off WR → Use Off (more trades)")
        print("  - If Binary trades <30 → Can't use Binary (insufficient sample)")

    def generate_recommendations(self):
        """Generate data-driven recommendations for AT integration."""
        print("\n" + "="*80)
        print("RECOMMENDATIONS - Data-Driven AT Integration Strategy")
        print("="*80)

        # Simulate binary mode effectiveness
        binary_longs = [t for t in self.trades if t.get("type") == "LONG" and t.get("at_buyers_dominant")]
        binary_shorts = [t for t in self.trades if t.get("type") == "SHORT" and t.get("at_sellers_dominant")]
        binary_trades = binary_longs + binary_shorts

        binary_wr = sum(1 for t in binary_trades if t.get("is_win")) / len(binary_trades) if len(binary_trades) > 0 else 0
        wr_improvement = (binary_wr - self.overall_win_rate) * 100

        print(f"\n📊 Current Results:")
        print(f"   Total Trades (Off mode): {self.total_trades}")
        print(f"   Win Rate (Off mode): {self.overall_win_rate*100:.1f}%")
        print(f"   Binary Mode Would Have: {len(binary_trades)} trades ({len(binary_trades)/self.total_trades*100:.0f}% of total)")
        print(f"   Binary Mode Win Rate: {binary_wr*100:.1f}%")
        print(f"   Win Rate Improvement: {wr_improvement:+.1f}pp")

        print("\n" + "-"*80)
        print("RECOMMENDED CONFIGURATION")
        print("-"*80)

        # Decision logic
        if len(binary_trades) < 30:
            print("\n❌ Binary Mode: INSUFFICIENT DATA")
            print(f"   Only {len(binary_trades)} trades would pass AT binary filter")
            print("   Need minimum 30 trades for reliable statistics")
            print("\n✅ RECOMMENDATION: Use OFF mode or SCORE mode")
            print("   Config: at_active = False  OR  at_mode = 'score'")

        elif wr_improvement > 15:
            print("\n✅ Binary Mode: STRONG IMPROVEMENT")
            print(f"   Win rate improvement: {wr_improvement:+.1f}pp")
            print(f"   Trade reduction acceptable: {len(binary_trades)/self.total_trades*100:.0f}% kept")
            print("\n✅ RECOMMENDATION: Use BINARY mode")
            print("   Config: at_active = True, at_mode = 'binary'")

        elif wr_improvement > 5:
            print("\n⚠️  Binary Mode: MODERATE IMPROVEMENT")
            print(f"   Win rate improvement: {wr_improvement:+.1f}pp (moderate)")
            print(f"   Trade reduction: {(1 - len(binary_trades)/self.total_trades)*100:.0f}%")
            print("\n⚠️  RECOMMENDATION: Use SCORE mode (balanced)")
            print("   Config: at_active = True, at_mode = 'score'")
            print("   Reasoning: AT adds value but not enough to justify 50%+ trade reduction")

        else:
            print("\n❌ Binary Mode: NO IMPROVEMENT")
            print(f"   Win rate improvement: {wr_improvement:+.1f}pp (negligible)")
            print("   AT filter doesn't add value")
            print("\n✅ RECOMMENDATION: DISABLE AlphaTrend")
            print("   Config: at_active = False")
            print("   Reasoning: AT doesn't improve results, removing it increases sample size")

        # Additional insights
        print("\n" + "-"*80)
        print("ADDITIONAL INSIGHTS")
        print("-"*80)

        # Check flat state
        flat_trades = [t for t in self.trades if t.get("at_is_flat")]
        not_flat_trades = [t for t in self.trades if not t.get("at_is_flat")]

        if len(flat_trades) > 0 and len(not_flat_trades) > 0:
            flat_wr = sum(1 for t in flat_trades if t.get("is_win")) / len(flat_trades)
            not_flat_wr = sum(1 for t in not_flat_trades if t.get("is_win")) / len(not_flat_trades)

            if flat_wr > not_flat_wr + 0.10:
                print("\n💡 AT Flat = Higher Win Rate!")
                print("   SSL Flow works BETTER in ranging markets (AT flat)")
                print("   Consider: REVERSE flat filter or use regime detector")

        print("\n" + "="*80)

    def run_full_analysis(self):
        """Run all analyses."""
        self.print_header()
        self.analyze_by_at_dominant()
        self.analyze_by_at_flat()
        self.analyze_by_at_regime()
        self.analyze_score_thresholds()
        self.analyze_binary_vs_score_simulation()
        self.generate_recommendations()


def main():
    parser = argparse.ArgumentParser(description="Analyze AT correlation with trade outcomes")
    parser.add_argument("trades_file", help="JSON file with trade data")

    args = parser.parse_args()

    # Load trades
    try:
        with open(args.trades_file, 'r') as f:
            trades = json.load(f)

        if len(trades) == 0:
            print("❌ ERROR: No trades found in file")
            sys.exit(1)

        # Run analysis
        analyzer = ATCorrelationAnalyzer(trades)
        analyzer.run_full_analysis()

        print("\n✅ Analysis complete!")
        print("\nNext steps:")
        print("  1. Review recommendations above")
        print("  2. Update config based on findings")
        print("  3. Re-run test with optimal AT settings")
        print("  4. Proceed to Phase 2 (signal journaling)\n")

    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {args.trades_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ ERROR: Invalid JSON in file: {args.trades_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
