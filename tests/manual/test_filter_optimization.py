#!/usr/bin/env python3
# test_filter_optimization.py
# Quick test to verify Filter Optimization system works

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.filter_discovery import (
    FilterDiscoveryEngine,
    FilterCombination,
    FilterDiscoveryResult,
)
from core import TradingEngine, calculate_indicators
import pandas as pd
import numpy as np


def test_imports():
    """Test that all required imports work."""
    print("[TEST] Testing imports...")

    # Test FilterDiscoveryEngine has new methods
    assert hasattr(FilterDiscoveryEngine, 'analyze_individual_filter_pass_rates')
    assert hasattr(FilterDiscoveryEngine, 'find_pareto_optimal_combinations')
    assert hasattr(FilterDiscoveryEngine, 'generate_parameter_sensitivity_grid')
    assert hasattr(FilterDiscoveryEngine, 'generate_comprehensive_report')

    print("[TEST] All imports successful!")
    return True


def test_filter_combination():
    """Test FilterCombination dataclass."""
    print("\n[TEST] Testing FilterCombination...")

    # Create a combo with some filters disabled
    combo = FilterCombination(
        adx_filter=True,
        regime_gating=True,
        baseline_touch=False,  # Disabled
        pbema_distance=True,
        body_position=False,   # Disabled
        ssl_pbema_overlap=True,
        wick_rejection=False,  # Disabled
    )

    # Test to_string
    combo_str = combo.to_string()
    assert "ADX" in combo_str
    assert "REGIME" in combo_str
    assert "TOUCH" not in combo_str  # Should be missing (disabled)
    assert "PBEMA_DIST" in combo_str
    assert "BODY" not in combo_str  # Should be missing (disabled)

    # Test to_config_overrides
    overrides = combo.to_config_overrides()
    assert 'lookback_candles' in overrides  # baseline_touch disabled
    assert overrides['lookback_candles'] == 9999
    assert 'ssl_body_tolerance' in overrides  # body_position disabled
    assert 'skip_wick_rejection' in overrides  # wick_rejection disabled

    print(f"[TEST] FilterCombination: {combo_str}")
    print(f"[TEST] Config overrides: {overrides}")
    print("[TEST] FilterCombination tests passed!")
    return True


def test_mock_discovery():
    """Test FilterDiscoveryEngine with mock data."""
    print("\n[TEST] Testing FilterDiscoveryEngine with mock data...")

    # Create minimal mock DataFrame
    n = 500
    dates = pd.date_range(start='2024-01-01', periods=n, freq='15min')

    # Generate random OHLCV data
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.random.rand(n) * 200
    low = close - np.random.rand(n) * 200
    open_ = close + np.random.randn(n) * 50
    volume = np.random.rand(n) * 1000

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'rsi': 50 + np.random.randn(n) * 10,
        'adx': 20 + np.random.rand(n) * 10,
        'baseline': close + np.random.randn(n) * 100,
        'pb_ema_top': close + 500 + np.random.rand(n) * 100,
        'pb_ema_bot': close + 400 + np.random.rand(n) * 100,
        'at_buyers_dominant': np.random.rand(n) > 0.5,
        'at_sellers_dominant': np.random.rand(n) > 0.5,
        'at_is_flat': np.random.rand(n) > 0.8,
        'alphatrend': close + np.random.randn(n) * 50,
        'alphatrend_2': close + np.random.randn(n) * 50,
    })

    # Create engine
    engine = FilterDiscoveryEngine(
        symbol="MOCK",
        timeframe="15m",
        data=df,
        baseline_trades=10,
    )

    print(f"[TEST] Created FilterDiscoveryEngine with {len(df)} candles")
    print(f"[TEST] Train: {len(engine.train_data)}, WF: {len(engine.wf_data)}, Holdout: {len(engine.holdout_data)}")

    # Test combination generation
    combos = engine.generate_combinations()
    assert len(combos) == 128  # 2^7 combinations
    print(f"[TEST] Generated {len(combos)} filter combinations")

    # Test a single combination evaluation (fast)
    test_combo = FilterCombination(
        adx_filter=True,
        regime_gating=True,
        baseline_touch=True,
        pbema_distance=True,
        body_position=True,
        ssl_pbema_overlap=True,
        wick_rejection=True,
    )

    result = engine.evaluate_combination(test_combo)
    assert isinstance(result, FilterDiscoveryResult)
    print(f"[TEST] Evaluated test combo: Trades={result.train_trades}, E[R]={result.train_expected_r:.3f}")

    print("[TEST] FilterDiscoveryEngine tests passed!")
    return True


def test_pareto_analysis():
    """Test Pareto-optimal analysis with mock results."""
    print("\n[TEST] Testing Pareto-optimal analysis...")

    # Create mock results
    mock_results = []

    # Create some mock combinations with different trade-offs
    configs = [
        (50, 0.06, False),  # High frequency, low E[R]
        (40, 0.08, False),  # Medium frequency, medium E[R]
        (30, 0.10, False),  # Lower frequency, higher E[R]
        (35, 0.07, False),  # Dominated by (40, 0.08) - NOT Pareto-optimal
        (20, 0.12, False),  # Low frequency, high E[R]
        (25, 0.09, True),   # Overfit - should be excluded
    ]

    for trades, expected_r, is_overfit in configs:
        combo = FilterCombination(
            adx_filter=True,
            regime_gating=True,
            baseline_touch=True,
            pbema_distance=True,
            body_position=True,
            ssl_pbema_overlap=True,
            wick_rejection=True,
        )

        result = FilterDiscoveryResult(
            combination=combo,
            train_pnl=trades * 10,
            train_trades=trades,
            train_expected_r=expected_r,
            train_win_rate=0.55,
            train_score=trades * expected_r,
            wf_pnl=trades * 8,
            wf_trades=trades - 5,
            wf_expected_r=expected_r * 0.8,
            wf_win_rate=0.50,
            overfit_ratio=0.8 if not is_overfit else 0.3,
            is_overfit=is_overfit,
        )
        mock_results.append(result)

    # Create minimal engine for testing
    import pandas as pd
    df = pd.DataFrame({'close': [50000] * 100})
    engine = FilterDiscoveryEngine(
        symbol="MOCK",
        timeframe="15m",
        data=df,
        baseline_trades=10,
    )

    # Find Pareto-optimal
    pareto = engine.find_pareto_optimal_combinations(mock_results, min_trades=5, min_expected_r=0.0)

    # Should exclude:
    # - (35, 0.07) - dominated by (40, 0.08)
    # - (25, 0.09) - overfit
    # Expected Pareto-optimal:
    # - (50, 0.06), (40, 0.08), (30, 0.10), (20, 0.12)

    assert len(pareto) == 4, f"Expected 4 Pareto-optimal configs, got {len(pareto)}"

    # Check they're sorted by trades descending
    trades_list = [r.train_trades for r in pareto]
    assert trades_list == sorted(trades_list, reverse=True), "Should be sorted by trades descending"

    print(f"[TEST] Found {len(pareto)} Pareto-optimal combinations (expected 4)")
    for i, r in enumerate(pareto, 1):
        print(f"  {i}. Trades={r.train_trades}, E[R]={r.train_expected_r:.2f}")

    print("[TEST] Pareto-optimal analysis tests passed!")
    return True


def test_comprehensive_report():
    """Test comprehensive report generation."""
    print("\n[TEST] Testing comprehensive report generation...")

    # Create mock results (minimal for testing)
    combo = FilterCombination(
        adx_filter=True,
        regime_gating=True,
        baseline_touch=True,
        pbema_distance=True,
        body_position=True,
        ssl_pbema_overlap=True,
        wick_rejection=True,
    )

    result = FilterDiscoveryResult(
        combination=combo,
        train_pnl=500.0,
        train_trades=30,
        train_expected_r=0.12,
        train_win_rate=0.58,
        train_score=100.0,
        wf_pnl=400.0,
        wf_trades=25,
        wf_expected_r=0.10,
        wf_win_rate=0.55,
        overfit_ratio=0.83,
        is_overfit=False,
    )

    # Create minimal engine
    df = pd.DataFrame({'close': [50000] * 100})
    engine = FilterDiscoveryEngine(
        symbol="BTCUSDT",
        timeframe="15m",
        data=df,
        baseline_trades=10,
    )

    # Generate report
    report = engine.generate_comprehensive_report(
        results=[result],
        filter_pass_rates=None,
        pareto_optimal=None,
        output_file=None,
    )

    # Check report contains expected sections
    assert "FILTER OPTIMIZATION REPORT" in report
    assert "CURRENT BASELINE" in report
    assert "BTCUSDT" in report
    assert "15m" in report
    assert "30" in report  # Trades
    assert "0.12" in report  # E[R]

    print(f"[TEST] Generated comprehensive report ({len(report)} chars)")
    print("[TEST] Report sections found:")
    for section in ["FILTER OPTIMIZATION REPORT", "CURRENT BASELINE", "WALK-FORWARD VALIDATION"]:
        if section in report:
            print(f"  - {section}")

    print("[TEST] Comprehensive report tests passed!")
    return True


def main():
    """Run all tests."""
    print("="*80)
    print("FILTER OPTIMIZATION SYSTEM - TEST SUITE")
    print("="*80)

    tests = [
        ("Imports", test_imports),
        ("FilterCombination", test_filter_combination),
        ("FilterDiscoveryEngine", test_mock_discovery),
        ("Pareto Analysis", test_pareto_analysis),
        ("Comprehensive Report", test_comprehensive_report),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"[FAIL] {name} test failed!")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {name} test failed with exception: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"TEST RESULTS: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    print("="*80)

    if failed == 0:
        print("\nALL TESTS PASSED!")
        return 0
    else:
        print(f"\n{failed} TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
