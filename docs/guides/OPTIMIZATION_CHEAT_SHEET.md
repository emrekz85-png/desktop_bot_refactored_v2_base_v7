# Walk-Forward Optimization: Cheat Sheet

**Print this. Keep it visible while implementing.**

---

## The Three Wins (Phase 1)

```
CURRENT STATE:
├─ Indicator calculation: Happens 100x (one per config) ❌
├─ Config evaluation: Sequential (1 at a time) ❌
└─ Search space: Full grid (100-120 configs) ❌
Result: 30 minutes

WIN 1: Indicator Caching
├─ Calculate once, reuse 100x
└─ Speedup: 20-30%

WIN 2: Parallel Execution
├─ Evaluate 6-7 configs simultaneously
└─ Speedup: 2-2.5x

WIN 3: Smart Grid
├─ Test 50-70 promising configs instead of 100-120
└─ Speedup: 20-30%

AFTER PHASE 1:
├─ Indicators: 1x calculation ✅
├─ Evaluation: 6-7 parallel workers ✅
└─ Search space: Adaptive focus ✅
Result: 10-15 minutes (2.5-3x gain)
```

---

## Quick Implementation Reference

### Win 1: Indicator Caching (2-3 hours)

**File:** `core/indicators.py`

**Add this function:**
```python
def pre_calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators once per window."""
    df = df.copy()
    df['hma60'] = ta.hma(df['close'], length=60)
    df['rsi'] = ta.rsi(df['close'], length=14)
    # ... add ALL indicator calculations
    return df
```

**File:** `core/optimizer.py` (main optimizer function)

**Change from:**
```python
for config in candidates:
    df_with_indicators = calculate_indicators(df_raw, config)  # WRONG!
    score = _score_config_for_stream(df_with_indicators, config)
```

**Change to:**
```python
df_with_indicators = pre_calculate_indicators(df_raw)  # ONCE

for config in candidates:
    score = _score_config_for_stream(df_with_indicators, config)
```

---

### Win 2: Parallel Execution (4-6 hours)

**File:** `core/optimizer.py` (top of file)

**Add import:**
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
```

**Add function:**
```python
def _evaluate_config_parallel(df, symbol, timeframe, config):
    """Single config evaluation for parallel execution."""
    return _score_config_for_stream(df, symbol, timeframe, config)
```

**Change from:**
```python
best_score = -float('inf')
for config in candidates:
    score = _score_config_for_stream(df, symbol, timeframe, config)
    if score > best_score:
        best_score = score
        best_config = config
```

**Change to:**
```python
best_config = None
best_score = -float('inf')

with ProcessPoolExecutor(max_workers=6) as executor:
    futures = {
        executor.submit(
            _evaluate_config_parallel,
            df, symbol, timeframe, cfg
        ): cfg
        for cfg in candidates
    }

    for future in as_completed(futures):
        cfg = futures[future]
        score = future.result()
        if score > best_score:
            best_score = score
            best_config = cfg
```

**Test:**
```python
# Run twice, verify IDENTICAL results
result1 = optimize_configs(window_data)
result2 = optimize_configs(window_data)
assert result1['best_config'] == result2['best_config']
```

---

### Win 3: Smart Grid (2-4 hours)

**File:** `core/optimizer.py`

**Add function:**
```python
def _generate_adaptive_candidates(config_history, full_grid):
    """Adaptive grid with warm-start."""

    candidates = []

    # Keep best recent configs as-is
    if config_history:
        candidates.extend(config_history[-3:])

    # Fine-tune around best
    if config_history:
        best = config_history[0]
        for rr_d in [-0.1, 0, 0.1]:
            for rsi_d in [-5, 0, 5]:
                candidates.append({
                    'rr': round(best['rr'] + rr_d, 2),
                    'rsi': int(best['rsi'] + rsi_d),
                    # ... copy other fields from best
                })

    # Add diverse configs
    candidates.extend(full_grid[::3])  # Every 3rd from full grid

    return candidates[:50]  # Limit to 50
```

**Change optimizer call:**
```python
# From:
candidates = _generate_candidate_configs()

# To:
full_grid = _generate_candidate_configs()
candidates = _generate_adaptive_candidates(config_history, full_grid)
```

---

## Testing Checklist

After each win:

```
Win 1 (Indicator Caching):
☐ Indicators calculated correctly (same values as before)
☐ Cached dataframe passed to all configs
☐ Runtime reduced 20-30%

Win 2 (Parallelization):
☐ ProcessPoolExecutor using 6 workers
☐ Determinism test: Run twice, same result
☐ No crashes or errors in parallel execution
☐ Runtime reduced 2-2.5x

Win 3 (Smart Grid):
☐ Warm-start includes best recent configs
☐ Fine-tuning around best works
☐ Grid size reduced 50%
☐ Walk-Forward Efficiency > 50% (no overfitting)

Full Phase 1 Test:
☐ Run full year test (30 windows × 3 symbols)
☐ Compare to baseline (within ±2% acceptable)
☐ Check memory usage < 3GB
☐ Verify determinism 2x run
☐ Document actual speedup achieved
```

---

## Phase 2: Bayesian Optimization (Weeks 2-3)

**Install:**
```bash
pip install optuna
```

**File:** `core/optimizer.py` (add at top)

```python
import optuna
from optuna.samplers import TPESampler
```

**Add function:**
```python
def _optimize_bayesian(df, symbol, timeframe, config_history):
    """Bayesian optimization using Optuna."""

    def objective(trial):
        # Define search space
        config = {
            'rr': trial.suggest_float('rr', 1.2, 2.5, step=0.1),
            'rsi': trial.suggest_int('rsi', 35, 75, step=5),
            'use_dynamic_pbema_tp': trial.suggest_categorical(
                'dyn_tp', [True, False]
            ),
            # ... add other params
        }

        # Evaluate
        score = _score_config_for_stream(
            df, symbol, timeframe, config
        )
        return score

    # Seeded sampler for reproducibility
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        sampler=sampler,
        direction='maximize'
    )

    # Optimize (40 trials instead of 100+)
    study.optimize(
        objective,
        n_trials=40,
        show_progress_bar=False,
        gc_after_trial=True
    )

    return study.best_params
```

**Replace grid search:**
```python
# From:
best_config = _optimize_grid_search(df, symbol, timeframe, candidates)

# To:
best_config = _optimize_bayesian(df, symbol, timeframe, config_history)
```

**Verify:**
```python
# Run twice with same seed
config1 = _optimize_bayesian(df, 'BTCUSDT', '15m', [])
config2 = _optimize_bayesian(df, 'BTCUSDT', '15m', [])
assert config1 == config2, "Determinism broken!"
```

---

## Performance Timeline

```
START:        30 minutes
After Win 1:  26 minutes (13% faster)
After Win 2:  14 minutes (46% faster)
After Win 3:  10 minutes (67% faster)
─────────────────────────────────────────
Phase 1:      2.5-3x speedup ✅

After Bayesian: 4-6 minutes (80-87% faster)
─────────────────────────────────────────
Phase 1+2:    5-7x speedup ✅✅
```

---

## Critical Checkpoints

### Checkpoint 1: After Indicator Caching

**Test:**
```python
df1 = pre_calculate_indicators(data)
df2 = pre_calculate_indicators(data)
assert df1.equals(df2), "Indicators not deterministic!"
```

**Success:** Identical indicators each time

### Checkpoint 2: After Parallelization

**Test:**
```python
# Must run twice and get IDENTICAL best config
config1 = optimize_parallel(data, candidates)
config2 = optimize_parallel(data, candidates)
assert config1 == config2, "Parallelization broke determinism!"
```

**Success:** Same best config both times

### Checkpoint 3: Full Phase 1 Validation

**Test:**
```python
# Compare to baseline
baseline_pnl = original_backtest()  # Original method
optimized_pnl = optimized_backtest()  # New method
assert abs(baseline_pnl - optimized_pnl) < baseline_pnl * 0.02
# Within 2% is acceptable
```

**Success:** Results match baseline (within 2%)

### Checkpoint 4: After Bayesian

**Test:**
```python
# Bayesian should find same or better configs
grid_configs = [(score1, cfg1), (score2, cfg2), ...]
bayesian_configs = [(score1, cfg1), (score2, cfg2), ...]

grid_best = max(grid_configs)[0]
bayesian_best = max(bayesian_configs)[0]
assert bayesian_best >= grid_best - 5
# Bayesian should match or beat grid
```

**Success:** Bayesian configs equal or better quality

---

## Troubleshooting

### Problem: "Determinism test failed"

**Cause:** Random elements not seeded
**Fix:**
```python
# At module top (core/optimizer.py)
import random
import numpy as np

random.seed(42)
np.random.seed(42)
```

---

### Problem: "ProcessPoolExecutor hangs"

**Cause:** Data not serializable (e.g., generators, database connections)
**Fix:**
- Ensure df_with_indicators is a pandas DataFrame (serializable)
- Config dict must only contain basic types (int, float, bool, str)
- No lambda functions in config

---

### Problem: "Bayesian gives different result each time"

**Cause:** Seed not passed to sampler
**Fix:**
```python
# WRONG:
study = optuna.create_study(direction='maximize')

# RIGHT:
sampler = TPESampler(seed=42)
study = optuna.create_study(sampler=sampler, direction='maximize')
```

---

### Problem: "Walk-Forward Efficiency dropped to 30%"

**Cause:** Likely look-ahead bias or overfitting
**Fix:**
- Verify ADX window: `iloc[start:index]` not `iloc[start:index+1]`
- Check if config_history includes too many old configs
- Run quick 1-symbol test to isolate problem
- Compare with Phase 1 baseline

---

## Validation Commands

```bash
# Test indicator caching
python -c "from core.indicators import pre_calculate_indicators; \
import pandas as pd; \
df = pd.read_csv('test_data.csv'); \
df1 = pre_calculate_indicators(df); \
df2 = pre_calculate_indicators(df); \
assert df1.equals(df2)"

# Test parallelization
python -c "from concurrent.futures import ProcessPoolExecutor; \
print('ProcessPoolExecutor available: OK')"

# Test Optuna installation
pip list | grep optuna
python -c "import optuna; print(f'Optuna version: {optuna.__version__}')"
```

---

## Key Numbers to Remember

```
Current system:      30 minutes per full year test

Phase 1 targets:
  Win 1: 20-30% speedup (6 min saved)
  Win 2: 2-2.5x speedup (20-22 min saved)
  Win 3: 20-30% speedup (4 min saved)
  Combined: 2.5-3x (20 min saved) → 10 min total

Phase 2 targets:
  Evaluation reduction: 100 → 40 configs
  Speedup: 2-3x on optimizer phase
  Combined: 5-7x total (25 min saved) → 5 min total

Your system:
  8 CPU cores → use 6-7 for workers
  32GB RAM → no memory constraints
  52 windows/year → 26 hours optimization/year (manageable)
```

---

## Risk Summary

```
Win 1 (Caching):      ZERO RISK
                      Same calculations, just reused
                      Easy to test (just compare indicators)

Win 2 (Parallelization): VERY LOW RISK
                      ProcessPoolExecutor is standard library
                      Determinism easy to verify
                      Can fall back to sequential if issues

Win 3 (Smart Grid):   LOW RISK
                      Warm-start with history is proven
                      Can still test full grid as fallback
                      WFE metric catches overfitting

Phase 2 (Bayesian):   LOW RISK
                      Optuna is production-ready
                      Many companies use it
                      Easy to A/B test vs grid
```

---

## Success Criteria

After Phase 1, you succeed if:

```
✅ 30 min → 10-15 min (2.5-3x faster)
✅ Run twice, same result (determinism OK)
✅ Backtest results match baseline (within 2%)
✅ WFE > 50% (no overfitting introduced)
✅ Memory < 3GB peak (no issues)
```

After Phase 2, you succeed if:

```
✅ 10-15 min → 4-6 min (additional 2-3x)
✅ Bayesian configs = or > quality than Phase 1
✅ All checkpoints pass
✅ Walk-Forward Efficiency maintained
```

---

## One-Pager: What to Do Right Now

**Today (30 min):**
1. Read WALKFORWARD_QUICK_GUIDE.md
2. Review your core/optimizer.py
3. Create test_comparison.py to verify results

**This Week:**
1. Implement Win 1 (Indicator Caching)
2. Implement Win 2 (Parallelization)
3. Implement Win 3 (Smart Grid)
4. Run full year test, verify 2.5-3x speedup

**Next Week:**
1. Evaluate Phase 2 (Bayesian)
2. Decide: implement it or stop here?
3. If yes: implement Bayesian, validate

**Goal:** 30 min → 4-6 min by January 10, 2026

---

**Print this. Tape it to your monitor. Reference while coding.**

Good luck! You've got this.

---

**Version:** 1.0
**Date:** December 30, 2025
**Status:** Ready to Use
