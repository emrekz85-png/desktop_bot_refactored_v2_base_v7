# Walk-Forward Optimization: Quick Implementation Guide

**Status:** Ready to implement
**Baseline System:** 30-minute rolling walk-forward test (52 windows, 3 symbols, 15m timeframe)
**Target:** 5-7x speedup (30 min → 4-6 min) within 2-3 weeks

---

## The Three Easiest Wins (This Week)

### Win 1: Indicator Caching (20-30% speedup, 2-3 hours)

**Why:** Indicators (SSL, AlphaTrend, RSI, ADX, PBEMA) are recalculated redundantly for each config, but they don't depend on trading parameters.

**Current Flow:**
```
For each of 100-120 configs:
  1. Calculate indicators (redundant!)
  2. Run backtest
  3. Score result
```

**After Fix:**
```
1. Calculate indicators once (outside loop)
2. For each of 100-120 configs:
   2a. Run backtest (use cached indicators)
   2b. Score result
```

**Implementation:** ~30 lines in `core/indicators.py`

```python
def pre_calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators once per window."""
    df = df.copy()
    df['hma60'] = ta.hma(df['close'], length=60)
    df['rsi'] = ta.rsi(df['close'], length=14)
    # ... etc for all indicators
    return df
```

**Where to Add:**
```python
# In core/optimizer.py, main optimizer function:
df_with_indicators = pre_calculate_indicators(df_raw)  # ONCE

for config in candidate_configs:
    score = _score_config_for_stream(
        df_with_indicators,  # Use pre-calculated
        symbol, timeframe, config
    )
```

---

### Win 2: Parallel Config Evaluation (2-2.5x speedup, 4-6 hours)

**Why:** Evaluating 100-120 configs sequentially is slow. With 8 CPU cores, you can evaluate 6-7 configs simultaneously.

**Current:** ~34 seconds per window (sequential)
**After:** ~14-18 seconds per window (parallel)

**Implementation:** Replace sequential loop with ProcessPoolExecutor

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def optimize_parallel(df, symbol, timeframe, candidates, max_workers=6):
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_score_config_for_stream, df, symbol, timeframe, cfg): cfg
            for cfg in candidates
        }

        for future in as_completed(futures):
            cfg = futures[future]
            score = future.result()
            results[cfg_hash(cfg)] = (score, cfg)

    # Find best
    best_score, best_cfg = max((s, c) for s, c in results.values())
    return best_cfg
```

**Key Points:**
- Use 6-7 workers (leave 1-2 cores for system)
- ProcessPoolExecutor bypasses Python's GIL (true parallelism)
- Data is serialized/deserialized (overhead included in 2.5x estimate)
- Determinism maintained (seeding already in place)

---

### Win 3: Smart Grid Reduction (20-30% speedup, 2-4 hours)

**Why:** Don't test all 100-120 configs blindly. Seed with recent best configs and focus on promising regions.

**Current:** Test 100-120 configs per window
**After:** Test 50-70 configs per window (skip obviously bad zones)

**Implementation:**

```python
def generate_adaptive_grid(config_history, full_grid, max_candidates=50):
    """Adaptive grid: warm-start from history, focus on promising regions."""

    candidates = []

    # 1. Include best recent configs (don't modify them)
    if config_history:
        candidates.extend(config_history[-3:])

    # 2. Refine around best
    if config_history:
        best = config_history[0]
        for rr_delta in [-0.1, 0, 0.1]:
            for rsi_delta in [-5, 0, 5]:
                candidates.append({
                    'rr': round(best['rr'] + rr_delta, 2),
                    'rsi': int(best['rsi'] + rsi_delta),
                    # ... other params same
                })

    # 3. Add coarse grid around best region
    avg_rr = np.mean([c['rr'] for c in config_history[-5:]]) if config_history else 1.8
    for rr in np.linspace(max(1.2, avg_rr - 0.5), min(2.5, avg_rr + 0.5), 4):
        for rsi in np.linspace(45, 65, 4):
            candidates.append({...})

    return candidates[:max_candidates]
```

**Trade-offs:**
- Pro: 20-30% faster, still high quality results
- Con: Small risk of missing global optimum if market regime shifts dramatically
- Mitigation: Use with weekly rebalance (you already do this)

---

## Combined Effect (Phase 1: All Three Wins)

| Step | Time | Cumulative |
|------|------|-----------|
| Baseline (current) | 30 min | 30 min |
| + Indicator caching (Win 1) | -6 min | **24 min** |
| + Parallel execution (Win 2) | -10 min | **14 min** |
| + Grid reduction (Win 3) | -4 min | **10 min** |
| **Phase 1 Total** | | **2.5-3x speedup** |

---

## Next Level: Bayesian Optimization (Week 2-3, 3-4x additional)

Once you've completed Phase 1, the biggest remaining bottleneck is grid search itself.

**Problem:** Testing 50-70 configs still means blind evaluation
**Solution:** Bayesian optimization intelligently selects which configs to test

```python
import optuna
from optuna.samplers import TPESampler

def optimize_bayesian(df, symbol, timeframe, n_trials=40):
    """Bayesian optimization: test only 30-40 promising configs instead of 100+."""

    def objective(trial):
        config = {
            'rr': trial.suggest_float('rr', 1.2, 2.5, step=0.1),
            'rsi': trial.suggest_int('rsi', 35, 75, step=5),
            # ... other params
        }
        return _score_config_for_stream(df, symbol, timeframe, config)

    sampler = TPESampler(seed=42)  # Reproducibility
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
```

**Why It Works:**
- Evaluates ~40 configs instead of 100+
- Each evaluation informs the next (learns from results)
- Often finds same or better config
- Research shows: 67 iterations vs 810 for grid search, same quality

**After Phase 2:**
- 10 min (Phase 1) → 3-5 min (Phase 2 adds Bayesian)
- **Total: 5-7x speedup**

---

## Implementation Timeline

```
Week 1: Phase 1 (Quick Wins)
├─ Day 1-2: Indicator caching (2-3 hrs)
├─ Day 3-4: Parallel executor (4-6 hrs)
└─ Day 5: Grid reduction + testing (2-4 hrs)
   Target: 30 min → 10-15 min

Week 2-3: Phase 2 (Bayesian Optimization)
├─ Day 6-7: Implement Bayesian (3-4 hrs)
├─ Day 8: Test & validate (2-3 hrs)
└─ Day 9-10: Full comparison test (4-6 hrs)
   Target: 10 min → 4-6 min
```

---

## What NOT to Try

- ❌ **Anchored walk-forward** - Less suitable for 15m intraday trading
- ❌ **Reduce window size** - Risks missing regime changes
- ❌ **GPU acceleration** - Pandas operations don't benefit much
- ❌ **Distributed computing** - Not worth complexity for 30min baseline
- ❌ **Monte Carlo only** - Use as validation, not optimization

---

## Verification Checklist

Before and after each phase:

- [ ] Run same test 2x, verify identical results (seed control)
- [ ] Compare config selections (should be same or better)
- [ ] Verify PnL within expected range (no look-ahead bias)
- [ ] Check walk-forward efficiency ratio (>50% rule)
- [ ] Monitor memory usage (should stay <2GB)

---

## Files to Modify

| File | Change | Impact |
|------|--------|--------|
| `core/indicators.py` | Extract `pre_calculate_indicators()` | Win 1 |
| `core/optimizer.py` | Add parallel executor | Win 2 |
| `core/optimizer.py` | Add adaptive grid generator | Win 3 |
| `requirements.txt` | Add `optuna` | Phase 2 |
| `core/optimizer.py` | Add Bayesian function | Phase 2 |

---

## Expected Outcomes

**Phase 1 (Conservative):** 2.5-3x speedup, Zero risk
- 30 min → 10-15 min
- Same accuracy, no overfitting risk

**Phase 2 (Recommended):** 5-7x speedup total, Low risk
- 30 min → 4-6 min
- Same or better configs, proven methodology (Bayesian)

**Phase 3 (Optional):** 5-10x total if including advanced validation
- 30 min → 3-5 min (optimization loop)
- + 10 min/month CPCV validation (robustness)

---

## Success Metrics

- ✅ Config selection identical (or better quality)
- ✅ Full year test complete in <10 minutes
- ✅ PnL results identical to baseline
- ✅ Walk-forward efficiency >50%
- ✅ Deterministic (same results on repeated runs)
- ✅ Zero overfitting introduced

---

## Reference: Full Analysis

See `WALKFORWARD_OPTIMIZATION_RESEARCH.md` for:
- Detailed technical explanations
- Academic research background
- Alternative methodologies (CPCV, Monte Carlo, etc.)
- Industry best practices
- Common pitfalls to avoid
- Code examples for all approaches

---

**Ready to start? Begin with Win 1 (Indicator Caching)** - it's the fastest ROI with zero risk.

Questions? The detailed research doc covers everything.
