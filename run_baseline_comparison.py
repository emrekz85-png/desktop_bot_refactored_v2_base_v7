#!/usr/bin/env python3
"""
Baseline vs Optimized Karşılaştırma Testi

Bu script optimizer'ın değer katıp katmadığını test eder.
"""

from desktop_bot_refactored_v2_base_v7 import compare_baseline_vs_optimized

if __name__ == "__main__":
    # 30 günlük test (11.01-12.01)
    result = compare_baseline_vs_optimized(
        start_date='2024-11-01',
        end_date='2024-12-01',
    )

    # Sonuç özeti
    comparison = result.get('comparison', {})
    print("\n" + "=" * 50)
    print("ÖZET:")
    print(f"  Optimizer değer katıyor mu? {comparison.get('optimizer_adds_value', 'N/A')}")
    print(f"  PnL Farkı: ${comparison.get('pnl_difference', 0):.2f}")
    print("=" * 50)
