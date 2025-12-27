# Filter Combination Discovery System

## Overview

The Filter Discovery System finds the optimal combination of AND filters for the SSL Flow trading strategy.

**Problem:** Currently 6+ filters must ALL pass → only 9 trades/year (over-filtering)

**Solution:** Test 2^6 = 64 filter combinations to find the sweet spot between signal quality and trade frequency.

## Architecture

### Core Components

1. **FilterCombination** (`core/filter_discovery.py`)
   - Represents one combination of 6 toggleable filters
   - Converts to config overrides for signal checking
   - 64 total combinations (all True/False permutations)

2. **FilterDiscoveryEngine** (`core/filter_discovery.py`)
   - Main discovery engine
   - Data splitting: 60% train, 20% WF, 20% holdout
   - Parallel execution support
   - Overfit detection using WF validation

3. **CLI Runner** (`run_filter_discovery.py`)
   - Three modes: --pilot, --full, --validate
   - Result saving and reporting
   - Holdout validation

### Toggleable Filters (6 total)

These filters can be independently enabled/disabled:

1. **adx_filter**: ADX >= adx_min (trend strength)
2. **regime_gating**: ADX_avg >= threshold over N bars (window-level regime)
3. **baseline_touch**: Price touched baseline in lookback (entry timing)
4. **pbema_distance**: Min distance to PBEMA target (room for profit)
5. **body_position**: Candle body above/below baseline (confirmation)
6. **ssl_pbema_overlap**: Check for SSL-PBEMA overlap (flow existence)

### CORE Filters (NEVER toggle - always ON)

These are essential to the strategy and cannot be disabled:

- **AlphaTrend dominance**: Buyers/sellers flow confirmation
- **Price position vs baseline**: Determines LONG/SHORT direction
- **RR validation**: Minimum risk/reward ratio check

## Usage

### Phase 1: Pilot Discovery

Fast test on BTCUSDT-15m only to find promising combinations:

```bash
python run_filter_discovery.py --pilot
```

**Expected runtime:** 3-4 hours (64 combinations × ~3 min each)

**Output:**
- `data/filter_discovery_runs/pilot_YYYYMMDD_HHMMSS/`
  - `results.json` - Full results for all 64 combinations
  - `top_10.txt` - Human-readable top 10 report
  - `filter_pass_rates.json` - Diagnostic filter statistics

### Phase 2: Full Discovery

Test on all symbols and timeframes (after pilot identifies promising combinations):

```bash
python run_filter_discovery.py --full
```

**Expected runtime:** Much longer (64 combos × N symbols × M timeframes)

**Recommendation:** Only run this after pilot confirms the approach works.

### Phase 3: Holdout Validation

Validate the top combination on never-before-seen holdout data:

```bash
python run_filter_discovery.py --validate 0 --results data/filter_discovery_runs/pilot_YYYYMMDD_HHMMSS/results.json
```

**What this does:**
- Tests combination #0 (top result) on holdout data (20% of data)
- Compares holdout E[R] to train/WF E[R]
- Validates that the combination generalizes (not overfitted)

## Scoring Function

The discovery uses a custom scoring function designed to find combinations that:

1. **Have positive edge:** E[R] > 0.05 (hard requirement)
2. **Increase trade frequency:** More than baseline 9 trades/year
3. **Show consistency:** Good Sharpe-like ratio
4. **Avoid very low win rates:** < 35% is penalized

**Formula:**
```
score = (
    expected_r * 0.40 +      # 40% weight on E[R]
    freq_bonus * 0.30 +       # 30% weight on trade frequency
    sharpe * 0.10 +           # 10% weight on consistency
    wr_factor * 0.20          # 20% weight on win rate
) * net_pnl
```

**Frequency bonus:**
- freq_ratio = trades / baseline_trades
- If ratio >= 1.0: bonus = min(2.0, 1.0 + (ratio - 1.0) * 0.3)
- If ratio < 1.0: bonus = 0.5 (penalty)

## Overfit Detection

The system uses walk-forward validation to detect overfitting:

1. **Data split:** 60% train (search), 20% WF (validation), 20% holdout (final test)
2. **Overfit criterion:** WF E[R] < 50% of train E[R]
3. **Holdout test:** Never seen during combination search

**Why this matters:**
- Combinations that work great on train data but fail on WF are overfit
- Only non-overfitted combinations are valid for deployment
- Holdout provides final confirmation before using in production

## Implementation Details

### Filter Override Mechanism

When a filter is disabled, the system passes parameters that make it always pass:

```python
overrides = {}

if not adx_filter:
    overrides['adx_min'] = -999.0  # Always passes

if not regime_gating:
    overrides['regime_adx_threshold'] = -999.0  # Always passes

if not baseline_touch:
    overrides['lookback_candles'] = 9999  # Always finds a touch

if not pbema_distance:
    overrides['min_pbema_distance'] = -999.0  # Always passes

if not body_position:
    overrides['ssl_body_tolerance'] = 999.0  # Always passes

if not ssl_pbema_overlap:
    overrides['skip_overlap_check'] = True  # Skip the check
```

### Integration with Existing Code

The system integrates seamlessly with existing infrastructure:

1. **Uses `_score_config_for_stream`** from `core/optimizer.py` for backtesting
2. **Uses `TradingEngine.check_signal`** for signal detection
3. **Uses `SimTradeManager`** for trade simulation
4. **Follows existing config override pattern** in `strategies/router.py`

### Performance Optimizations

- **Parallel execution:** ThreadPoolExecutor for independent combinations
- **NumPy arrays:** Pre-extracted for hot loop performance
- **Same optimizations as optimizer:** Uses proven fast backtest logic

## Output Format

### results.json Structure

```json
{
  "run_id": "pilot_20250126_123456",
  "symbol": "BTCUSDT",
  "timeframe": "15m",
  "version": "v1.7.2",
  "timestamp": "2025-01-26T12:34:56",
  "total_combinations": 64,
  "results": [
    {
      "combination": {
        "adx_filter": true,
        "regime_gating": false,
        "baseline_touch": true,
        "pbema_distance": true,
        "body_position": true,
        "ssl_pbema_overlap": true
      },
      "combination_str": "ADX+TOUCH+PBEMA_DIST+BODY+OVERLAP",
      "train_pnl": 250.50,
      "train_trades": 25,
      "train_expected_r": 0.18,
      "train_win_rate": 0.44,
      "train_score": 45.23,
      "wf_pnl": 120.30,
      "wf_trades": 12,
      "wf_expected_r": 0.15,
      "wf_win_rate": 0.42,
      "overfit_ratio": 0.83,
      "is_overfit": false
    },
    ...
  ]
}
```

### top_10.txt Structure

Human-readable report showing:
- Filter combination name
- Training metrics (PnL, trades, E[R], win rate, score)
- Walk-forward metrics (PnL, trades, E[R], win rate)
- Overfit check (ratio, is_overfit)
- Enabled filters (checkmarks)

### filter_pass_rates.json

Diagnostic statistics showing which filters appear most often in top combinations:

```json
{
  "adx_filter": {
    "enabled": 45,
    "pass_rate": 0.70,
    "avg_score": 32.5
  },
  "regime_gating": {
    "enabled": 20,
    "pass_rate": 0.31,
    "avg_score": 28.3
  },
  ...
}
```

**Interpretation:**
- High pass_rate = filter appears in many good combinations
- Low pass_rate = filter often disabled in best combinations
- avg_score = average score when filter is enabled

## Expected Results

### Baseline (Current Strategy)

- All 6 filters ON
- ~9 trades/year
- High precision, low recall (too conservative)

### Hypotheses

1. **Regime gating might be redundant** with ADX filter
2. **Baseline touch might be too strict** (5-candle lookback)
3. **Body position might filter out valid wicks**

### Success Criteria

A successful combination should:
- E[R] >= 0.08 (same or better than baseline)
- Trades >= 15-20/year (significant increase from 9)
- WF E[R] / Train E[R] >= 0.50 (not overfitted)
- Holdout E[R] >= 0.05 (validates on unseen data)

## Troubleshooting

### Common Issues

**Issue:** All combinations have negative scores

**Solution:** Check that data has enough history (recommend 30000 candles)

---

**Issue:** Very slow execution

**Solution:** Use --no-parallel for debugging, or reduce --candles

---

**Issue:** Holdout validation shows much worse E[R]

**Solution:** This indicates overfitting. Try different data period or relax scoring thresholds

---

**Issue:** Import errors

**Solution:** Ensure running from project root and all dependencies installed

## Next Steps After Discovery

1. **Analyze top 10 combinations** - Look for patterns in which filters are ON/OFF
2. **Validate winner on holdout** - Ensure it generalizes
3. **Test on multiple symbols** - Check if pattern holds across assets
4. **Update strategy config** - Apply winning combination to production
5. **Monitor live performance** - Verify improvement in trade frequency

## Code Files

- **`core/filter_discovery.py`** - Main engine (450 lines)
- **`run_filter_discovery.py`** - CLI runner (550 lines)
- **`strategies/ssl_flow.py`** - Modified to accept skip_overlap_check
- **`strategies/router.py`** - Modified to pass filter overrides

## Example Workflow

```bash
# Step 1: Run pilot on BTCUSDT-15m
python run_filter_discovery.py --pilot

# Step 2: Check results
cat data/filter_discovery_runs/pilot_20250126_123456/top_10.txt

# Step 3: Validate top combination on holdout
python run_filter_discovery.py --validate 0 --results data/filter_discovery_runs/pilot_20250126_123456/results.json

# Step 4: If winner looks good, test on full universe
python run_filter_discovery.py --full

# Step 5: Apply winning combination to strategy config
# (Manual step - update DEFAULT_STRATEGY_CONFIG with new filter settings)
```

## References

- **Quant Analyst Recommendations:** See original specification for scoring function rationale
- **SSL Flow Strategy:** `strategies/ssl_flow.py` for filter implementation details
- **Optimizer Infrastructure:** `core/optimizer.py` for walk-forward validation approach
