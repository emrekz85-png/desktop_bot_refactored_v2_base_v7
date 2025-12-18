#!/usr/bin/env python3
"""
Rolling Walk-Forward Test Script

Bu script, Rolling Walk-Forward framework'ünü test eder:
1. Fixed vs Monthly vs Weekly karşılaştırması yapar
2. 2025 yılı için stitched OOS sonuçlarını hesaplar
3. En iyi modu önerir

Kullanım:
    python run_rolling_wf_test.py                    # Varsayılan test (son 6 ay)
    python run_rolling_wf_test.py --full-year       # 2025 tam yıl testi
    python run_rolling_wf_test.py --quick           # Hızlı test (3 ay, az sembol)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from desktop_bot_refactored_v2_base_v7 import (
    run_rolling_walkforward,
    compare_rolling_modes,
    BASELINE_CONFIG,
    SYMBOLS,
    TIMEFRAMES,
)


def run_quick_test():
    """Hızlı test - 3 ay, az sembol"""
    print("\n" + "="*70)
    print("🧪 HIZLI TEST (3 ay, 3 sembol)")
    print("="*70 + "\n")

    result = run_rolling_walkforward(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        timeframes=["15m", "1h"],
        mode="monthly",
        lookback_days=60,
        forward_days=30,
        start_date="2025-09-01",
        end_date="2025-12-01",
        verbose=True,
    )

    return result


def run_comparison_test(start_date: str = None, end_date: str = None):
    """Fixed vs Monthly vs Weekly karşılaştırma testi"""
    print("\n" + "="*70)
    print("🔬 ROLLING WALK-FORWARD KARŞILAŞTIRMA TESTİ")
    print("="*70 + "\n")

    # Use BASELINE_CONFIG for fixed mode
    result = compare_rolling_modes(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "LINKUSDT"],
        timeframes=["15m", "1h", "4h"],
        start_date=start_date or "2025-06-01",
        end_date=end_date or "2025-12-18",
        fixed_config=BASELINE_CONFIG,
        verbose=True,
    )

    return result


def run_full_year_test():
    """2025 tam yıl testi"""
    print("\n" + "="*70)
    print("📊 2025 TAM YIL TESTİ")
    print("="*70 + "\n")

    result = compare_rolling_modes(
        symbols=SYMBOLS,  # Tüm semboller
        timeframes=TIMEFRAMES,  # Tüm timeframe'ler
        start_date="2025-01-01",
        end_date="2025-12-18",
        fixed_config=BASELINE_CONFIG,
        verbose=True,
    )

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rolling Walk-Forward Test Script")
    parser.add_argument('--quick', action='store_true', help='Hızlı test (3 ay, az sembol)')
    parser.add_argument('--full-year', action='store_true', help='2025 tam yıl testi')
    parser.add_argument('--start-date', type=str, help='Başlangıç tarihi YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, help='Bitiş tarihi YYYY-MM-DD')

    args = parser.parse_args()

    if args.quick:
        result = run_quick_test()
    elif args.full_year:
        result = run_full_year_test()
    else:
        result = run_comparison_test(args.start_date, args.end_date)

    print("\n" + "="*70)
    print("✅ TEST TAMAMLANDI")
    print("="*70)

    # Print summary if comparison was run
    if "comparison" in result:
        comp = result["comparison"]
        print(f"\n🏆 EN İYİ MOD: {comp['best_mode'].upper()}")
        print(f"   PnL: Fixed=${comp['pnl']['fixed']:.2f}, Monthly=${comp['pnl']['monthly']:.2f}, Weekly=${comp['pnl']['weekly']:.2f}")


if __name__ == "__main__":
    main()
