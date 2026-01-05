# Optimizer Determinism Fix - Summary

## Problem
The optimizer was producing non-deterministic results on repeated runs due to:
1. Parallel execution with `as_completed()` returning results in random order
2. Floating-point comparison without epsilon tolerance
3. No tie-breaking mechanism for configs with equal scores
4. No random seeds set for reproducibility

This caused different configs to be selected on different runs, making it impossible to reproduce optimization results.

## Solution
Applied 4 critical fixes to `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/core/optimizer.py`:

### Fix 1: Set Random Seeds (Lines 14-15, 33-35)
```python
import random
import json  # Added for deterministic hashing

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
```

**Why:** Ensures any random operations are reproducible across runs.

### Fix 2: Add Deterministic Config Hash Function (Lines 42-43, 64-70)
```python
# Epsilon for floating point comparison (deterministic tie-breaking)
SCORE_EPSILON = 1e-10

def _config_hash(cfg: dict) -> str:
    """Deterministic hash for config tie-breaking.

    Used to ensure consistent config selection when scores are equal.
    Returns a deterministic JSON string representation of the config.
    """
    return json.dumps(cfg, sort_keys=True)
```

**Why:** Creates a unique, deterministic hash for each config that can be used for:
- Sorting configs in a consistent order
- Tie-breaking when scores are equal

### Fix 3: Use Epsilon Tolerance in Score Comparison (Lines 769-775)
**Before:**
```python
if score > best_score:
    best_score = score
    best_pnl = net_pnl
    best_cfg = cfg
    # ...
```

**After:**
```python
# Deterministic score comparison with epsilon tolerance
# If scores are equal (within epsilon), use config hash for tie-breaking
if score > best_score + SCORE_EPSILON or (
    abs(score - best_score) <= SCORE_EPSILON and
    best_cfg is not None and
    _config_hash(cfg) < _config_hash(best_cfg)
):
    best_score = score
    best_pnl = net_pnl
    best_cfg = cfg
    # ...
```

**Why:**
- Prevents floating-point precision issues from causing non-determinism
- When scores are equal (within epsilon), uses config hash for tie-breaking
- Ensures the same config is always selected for tied scores

### Fix 4: Collect and Sort Results Before Processing (Lines 797-819)
**Before:**
```python
for future in as_completed(futures):
    cfg = futures[future]
    try:
        net_pnl, trades, trade_pnls, trade_r_multiples = future.result()
    except Exception as exc:
        log(f"[OPT][{sym}-{tf}] Skorlama hatası (cfg={cfg}): {exc}")
        continue

    handle_result(cfg, net_pnl, trades, trade_pnls, trade_r_multiples)
```

**After:**
```python
# Collect all results first (non-deterministic order from as_completed)
all_results = []
for future in as_completed(futures):
    cfg = futures[future]
    try:
        result = future.result()
        all_results.append((cfg, result))
    except BrokenProcessPool:
        raise
    except Exception as exc:
        log(f"[OPT][{sym}-{tf}] Skorlama hatası (cfg={cfg}): {exc}")
        all_results.append((cfg, None))

# Sort by config hash for deterministic processing order
all_results.sort(key=lambda x: _config_hash(x[0]))

# Process in deterministic order
for cfg, result in all_results:
    if result is None:
        continue
    net_pnl, trades, trade_pnls, trade_r_multiples = result
    handle_result(cfg, net_pnl, trades, trade_pnls, trade_r_multiples)
```

**Why:**
- `as_completed()` returns futures in the order they complete (random)
- Collecting all results first, then sorting ensures deterministic processing
- Even if parallel execution is non-deterministic, final selection is deterministic

## Testing
Created comprehensive test suite in `test_determinism.py` that verifies:

1. **Config Hash Determinism**: Same config with different key order produces identical hash
2. **Config Hash Ordering**: Different configs have different hashes and can be sorted
3. **Epsilon Comparison**: Scores within epsilon are considered equal, outside are different
4. **Deterministic Selection**: Repeated runs produce identical results

All tests pass successfully.

## Impact
- Optimizer will now produce **identical results** on repeated runs with same data
- Configs with nearly-equal scores will be selected deterministically (by config hash)
- No performance impact (sorting adds negligible overhead)
- Reproducible optimization results for debugging and validation

## Verification
Run the test suite to verify the fix:
```bash
python3 test_determinism.py
```

Or test optimizer directly:
```bash
# Run optimizer twice - should produce identical best_configs.json
python3 run_backtest.py --optimize
cp best_configs.json best_configs_run1.json

python3 run_backtest.py --optimize
cp best_configs.json best_configs_run2.json

# Compare - should be identical
diff best_configs_run1.json best_configs_run2.json
```

## Files Modified
1. `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/core/optimizer.py` - Applied all 4 fixes

## Files Created
1. `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/test_determinism.py` - Test suite
2. `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/DETERMINISM_FIX_SUMMARY.md` - This document
