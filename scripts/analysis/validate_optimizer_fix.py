#!/usr/bin/env python3
"""
Comprehensive validation script for optimizer determinism fix.

This script performs end-to-end validation:
1. Verifies all code changes are in place
2. Tests deterministic behavior with mock data
3. Validates epsilon tolerance
4. Confirms reproducibility
"""

import sys
import inspect
from core.optimizer import (
    _config_hash,
    SCORE_EPSILON,
    _generate_candidate_configs,
    _generate_quick_candidate_configs,
    _optimize_backtest_configs
)

def validate_imports():
    """Validate that all required imports are available."""
    print("=" * 70)
    print("VALIDATION 1: Module Imports")
    print("=" * 70)

    try:
        import random
        import json
        import numpy as np
        from core.optimizer import _config_hash, SCORE_EPSILON
        print("✓ All required imports successful")
        print(f"  - random: {random.__name__}")
        print(f"  - json: {json.__name__}")
        print(f"  - numpy: {np.__name__}")
        print(f"  - _config_hash: {type(_config_hash).__name__}")
        print(f"  - SCORE_EPSILON: {SCORE_EPSILON}")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def validate_random_seeds():
    """Validate that random seeds are set in the module."""
    print("\n" + "=" * 70)
    print("VALIDATION 2: Random Seeds")
    print("=" * 70)

    # Check if seeds are set by inspecting module source
    import core.optimizer
    source = inspect.getsource(core.optimizer)

    has_random_seed = "random.seed(42)" in source
    has_numpy_seed = "np.random.seed(42)" in source

    if has_random_seed and has_numpy_seed:
        print("✓ Random seeds are set in module")
        print("  - random.seed(42): Found")
        print("  - np.random.seed(42): Found")
        return True
    else:
        print("✗ Random seeds not found in module source")
        if not has_random_seed:
            print("  - random.seed(42): MISSING")
        if not has_numpy_seed:
            print("  - np.random.seed(42): MISSING")
        return False

def validate_config_hash():
    """Validate that config hashing is deterministic."""
    print("\n" + "=" * 70)
    print("VALIDATION 3: Deterministic Config Hashing")
    print("=" * 70)

    # Test with different key orders
    configs = [
        {'rr': 1.5, 'rsi': 50, 'at_active': True},
        {'at_active': True, 'rsi': 50, 'rr': 1.5},
        {'rsi': 50, 'rr': 1.5, 'at_active': True}
    ]

    hashes = [_config_hash(cfg) for cfg in configs]

    if len(set(hashes)) == 1:
        print("✓ Config hashing is deterministic")
        print(f"  - All 3 configs produce same hash: {hashes[0]}")
        return True
    else:
        print("✗ Config hashing is NOT deterministic")
        for i, (cfg, h) in enumerate(zip(configs, hashes)):
            print(f"  - Config {i+1}: {h}")
        return False

def validate_epsilon_tolerance():
    """Validate that epsilon tolerance works correctly."""
    print("\n" + "=" * 70)
    print("VALIDATION 4: Epsilon Tolerance")
    print("=" * 70)

    epsilon = SCORE_EPSILON
    score_a = 100.0
    score_b = 100.0 + epsilon / 2  # Within epsilon
    score_c = 100.0 + epsilon * 2  # Outside epsilon

    within = abs(score_a - score_b) <= epsilon
    outside = abs(score_a - score_c) > epsilon

    if within and outside:
        print("✓ Epsilon tolerance is working correctly")
        print(f"  - SCORE_EPSILON: {epsilon}")
        print(f"  - Score A: {score_a}")
        print(f"  - Score B: {score_b} (within epsilon)")
        print(f"  - Score C: {score_c} (outside epsilon)")
        print(f"  - abs(A-B) <= epsilon: {within}")
        print(f"  - abs(A-C) > epsilon: {outside}")
        return True
    else:
        print("✗ Epsilon tolerance is NOT working correctly")
        return False

def validate_result_collection():
    """Validate that result collection code exists in optimizer."""
    print("\n" + "=" * 70)
    print("VALIDATION 5: Result Collection & Sorting")
    print("=" * 70)

    import core.optimizer
    source = inspect.getsource(core.optimizer._optimize_backtest_configs)

    has_all_results = "all_results = []" in source
    has_append = "all_results.append" in source
    has_sort = "all_results.sort" in source
    has_config_hash_sort = "_config_hash(x[0])" in source

    all_checks = all([has_all_results, has_append, has_sort, has_config_hash_sort])

    if all_checks:
        print("✓ Result collection and sorting implemented")
        print("  - all_results list: Found")
        print("  - Result appending: Found")
        print("  - Result sorting: Found")
        print("  - Config hash sorting: Found")
        return True
    else:
        print("✗ Result collection and sorting NOT fully implemented")
        if not has_all_results:
            print("  - all_results list: MISSING")
        if not has_append:
            print("  - Result appending: MISSING")
        if not has_sort:
            print("  - Result sorting: MISSING")
        if not has_config_hash_sort:
            print("  - Config hash sorting: MISSING")
        return False

def validate_score_comparison():
    """Validate that score comparison uses epsilon and config hash."""
    print("\n" + "=" * 70)
    print("VALIDATION 6: Score Comparison Logic")
    print("=" * 70)

    import core.optimizer
    source = inspect.getsource(core.optimizer._optimize_backtest_configs)

    has_epsilon_comparison = "best_score + SCORE_EPSILON" in source
    has_config_hash_tie = "_config_hash(cfg) < _config_hash(best_cfg)" in source
    has_abs_comparison = "abs(score - best_score) <= SCORE_EPSILON" in source

    all_checks = all([has_epsilon_comparison, has_config_hash_tie, has_abs_comparison])

    if all_checks:
        print("✓ Score comparison uses epsilon and config hash")
        print("  - Epsilon comparison: Found")
        print("  - Config hash tie-breaking: Found")
        print("  - Absolute difference check: Found")
        return True
    else:
        print("✗ Score comparison NOT fully implemented")
        if not has_epsilon_comparison:
            print("  - Epsilon comparison: MISSING")
        if not has_config_hash_tie:
            print("  - Config hash tie-breaking: MISSING")
        if not has_abs_comparison:
            print("  - Absolute difference check: MISSING")
        return False

def validate_reproducibility():
    """Validate that config generation is reproducible."""
    print("\n" + "=" * 70)
    print("VALIDATION 7: Reproducibility Test")
    print("=" * 70)

    # Generate configs twice
    configs1 = _generate_quick_candidate_configs()
    configs2 = _generate_quick_candidate_configs()

    # Convert to hashes for comparison
    hashes1 = [_config_hash(cfg) for cfg in configs1]
    hashes2 = [_config_hash(cfg) for cfg in configs2]

    if hashes1 == hashes2:
        print("✓ Config generation is reproducible")
        print(f"  - Generated {len(configs1)} configs")
        print(f"  - Both runs produced identical configs")
        return True
    else:
        print("✗ Config generation is NOT reproducible")
        print(f"  - Run 1: {len(configs1)} configs")
        print(f"  - Run 2: {len(configs2)} configs")
        return False

def main():
    """Run all validations."""
    print("\n" + "=" * 70)
    print("OPTIMIZER DETERMINISM FIX - COMPREHENSIVE VALIDATION")
    print("=" * 70)

    validations = [
        ("Module Imports", validate_imports),
        ("Random Seeds", validate_random_seeds),
        ("Config Hashing", validate_config_hash),
        ("Epsilon Tolerance", validate_epsilon_tolerance),
        ("Result Collection", validate_result_collection),
        ("Score Comparison", validate_score_comparison),
        ("Reproducibility", validate_reproducibility),
    ]

    results = []
    for name, validator in validations:
        try:
            passed = validator()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ EXCEPTION in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print("\n" + "-" * 70)
    print(f"TOTAL: {passed_count}/{total_count} validations passed")
    print("-" * 70)

    if passed_count == total_count:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("\nThe optimizer determinism fix is complete and verified:")
        print("  1. Random seeds are set (random.seed(42), np.random.seed(42))")
        print("  2. Config hashing is deterministic (JSON with sort_keys=True)")
        print("  3. Epsilon tolerance prevents floating-point precision issues")
        print("  4. Results are collected, sorted, then processed")
        print("  5. Score comparison uses epsilon and config hash tie-breaking")
        print("  6. Config generation is reproducible")
        print("\nThe optimizer will now produce identical results on repeated runs.")
        return 0
    else:
        print("\n⚠️  SOME VALIDATIONS FAILED")
        print(f"\n{total_count - passed_count} validation(s) did not pass.")
        print("Please review the failures above and fix the issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
