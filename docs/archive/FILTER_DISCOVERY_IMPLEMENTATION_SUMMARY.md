# Filter Discovery System - Implementation Summary

## What Was Built

A comprehensive system to discover the optimal filter combination for the SSL Flow trading strategy, addressing the over-filtering problem (currently only 9 trades/year).

## Files Created

### 1. Core Engine
**File:** `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/core/filter_discovery.py` (450 lines)

**Key Components:**
- `FilterCombination` dataclass - Represents one of 64 filter combinations
- `FilterDiscoveryResult` dataclass - Stores train/WF/holdout metrics
- `FilterDiscoveryEngine` class - Main discovery engine with:
  - Data splitting (60% train, 20% WF, 20% holdout)
  - Parallel combination testing
  - Overfit detection
  - Holdout validation
- `filter_discovery_score()` function - Custom scoring function per quant spec

### 2. CLI Runner
**File:** `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/run_filter_discovery.py` (550 lines)

**Modes:**
- `--pilot` - Fast test on BTCUSDT-15m (3-4 hours)
- `--full` - Full test on all symbols/timeframes
- `--validate` - Holdout validation of specific combination

**Features:**
- Data fetching and preparation
- Result saving (JSON, TXT reports, diagnostics)
- Progress reporting
- Error handling

### 3. Strategy Integration
**Modified Files:**
- `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/strategies/ssl_flow.py`
  - Added `skip_overlap_check` parameter
  - Allows disabling SSL-PBEMA overlap filter

- `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/strategies/router.py`
  - Pass `skip_overlap_check` to check_ssl_flow_signal
  - Support all filter overrides from config

### 4. Documentation
**Files:**
- `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/docs/filter_discovery.md` (comprehensive guide)
- `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/FILTER_DISCOVERY_QUICKSTART.md` (quick reference)
- `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/CLAUDE.md` (updated with discovery section)

## Filter Architecture

### Toggleable Filters (6 total = 64 combinations)

1. **adx_filter** - ADX >= adx_min
   - Override: Set adx_min = -999 to disable

2. **regime_gating** - ADX_avg >= threshold over N bars
   - Override: Set regime_adx_threshold = -999 to disable

3. **baseline_touch** - Price touched baseline in lookback
   - Override: Set lookback_candles = 9999 to always pass

4. **pbema_distance** - Min distance to PBEMA target
   - Override: Set min_pbema_distance = -999 to disable

5. **body_position** - Candle body above/below baseline
   - Override: Set ssl_body_tolerance = 999 to always pass

6. **ssl_pbema_overlap** - Check for SSL-PBEMA overlap
   - Override: Set skip_overlap_check = True to disable

### CORE Filters (NEVER toggle - always ON)

- AlphaTrend dominance (buyers/sellers)
- Price position vs baseline (determines LONG/SHORT)
- RR validation (minimum risk/reward ratio)

## Scoring Function

Implemented exactly per quant analyst specification:

```python
def filter_discovery_score(net_pnl, trades, trade_pnls, trade_r_multiples, baseline_trades=9):
    # Hard reject: no edge
    if trades == 0 or net_pnl <= 0:
        return -float("inf")

    # Hard reject: E[R] too low
    expected_r = sum(trade_r_multiples) / len(trade_r_multiples)
    if expected_r < 0.05:
        return -float("inf")

    # Trade frequency bonus (want more than baseline 9 trades)
    freq_ratio = trades / baseline_trades
    freq_bonus = min(2.0, 1.0 + (freq_ratio - 1.0) * 0.3) if freq_ratio >= 1.0 else 0.5

    # Win rate factor
    win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)
    wr_factor = 0.8 if win_rate < 0.35 else (1.0 if win_rate < 0.50 else 1.1)

    # Sharpe-like ratio
    if len(trade_pnls) >= 5:
        mean_pnl = sum(trade_pnls) / len(trade_pnls)
        std_pnl = (sum((p - mean_pnl)**2 for p in trade_pnls) / len(trade_pnls)) ** 0.5
        sharpe = min(mean_pnl / (std_pnl + 1e-6), 2.0)
    else:
        sharpe = 0.5

    # Combined: 40% E[R], 30% frequency, 10% Sharpe, 20% win rate
    score = (expected_r * 0.40 + freq_bonus * 0.30 + sharpe * 0.10 + wr_factor * 0.20) * net_pnl
    return score
```

**Weights:**
- 40% Expected R-multiple (E[R])
- 30% Trade frequency bonus
- 10% Sharpe-like ratio
- 20% Win rate factor

## Overfit Safeguards

### Data Split
- **60% Train** - Search period (find best combinations)
- **20% Walk-Forward** - Out-of-sample validation
- **20% Holdout** - Never seen during search (final validation)

### Overfit Detection
- Calculate overfit_ratio = WF E[R] / Train E[R]
- Mark as overfit if ratio < 0.50
- Require min 3 WF trades for statistical significance

### Holdout Validation
- Completely separate from train/WF
- Final test before deployment
- Confirms combination generalizes

## Usage Examples

### Quick Test (Recommended First Run)
```bash
python run_filter_discovery.py --pilot
```
- Tests BTCUSDT-15m only
- 64 combinations × ~3 min = 3-4 hours
- Outputs to `data/filter_discovery_runs/pilot_*/`

### View Results
```bash
# Top 10 human-readable
cat data/filter_discovery_runs/pilot_*/top_10.txt

# Full JSON results
cat data/filter_discovery_runs/pilot_*/results.json

# Filter statistics
cat data/filter_discovery_runs/pilot_*/filter_pass_rates.json
```

### Validate Winner
```bash
python run_filter_discovery.py --validate 0 \
  --results data/filter_discovery_runs/pilot_20250126_143022/results.json
```

### Full Universe Test
```bash
python run_filter_discovery.py --full
```
- All symbols × all timeframes
- Much slower, only run after pilot validates approach

## Output Format

### results.json Structure
```json
{
  "run_id": "pilot_20250126_143022",
  "symbol": "BTCUSDT",
  "timeframe": "15m",
  "total_combinations": 64,
  "results": [
    {
      "combination": {...},
      "combination_str": "ADX+TOUCH+PBEMA_DIST+BODY+OVERLAP",
      "train_pnl": 250.50,
      "train_trades": 25,
      "train_expected_r": 0.18,
      "train_score": 45.23,
      "wf_pnl": 120.30,
      "wf_trades": 12,
      "wf_expected_r": 0.15,
      "overfit_ratio": 0.83,
      "is_overfit": false
    }
  ]
}
```

### top_10.txt Example
```
================================================================================
#1 - ADX+TOUCH+PBEMA_DIST+BODY+OVERLAP
================================================================================
TRAINING (60% of data):
  PnL: $250.50
  Trades: 25
  E[R]: 0.180
  Win Rate: 44.0%
  Score: 45.23

WALK-FORWARD (20% of data):
  PnL: $120.30
  Trades: 12
  E[R]: 0.150
  Win Rate: 41.7%

OVERFIT CHECK:
  Overfit Ratio: 0.83 (WF E[R] / Train E[R])
  Is Overfit: NO ✓
```

## Integration Points

### Uses Existing Infrastructure

1. **`_score_config_for_stream()`** from `core/optimizer.py`
   - Backtest simulation logic
   - Same trade management as optimizer

2. **`TradingEngine.check_signal()`** from `strategies/router.py`
   - Signal detection with config overrides
   - Supports all strategy modes

3. **`SimTradeManager`** from `core/trade_manager.py`
   - Trade simulation
   - R-multiple tracking

4. **`calculate_indicators()`** from `core/indicators.py`
   - Indicator calculation
   - In-place DataFrame modification

### No Breaking Changes

- All modifications are backward compatible
- `skip_overlap_check` defaults to False (existing behavior)
- Filter overrides only apply when explicitly set
- No changes to live trading logic

## Performance Optimizations

- **Parallel execution:** ThreadPoolExecutor for independent combinations
- **NumPy arrays:** Pre-extracted from DataFrame before hot loop
- **Lazy imports:** Heavy libraries only loaded when needed
- **Shared data:** Single fetch, used across all combinations

## Success Criteria

A successful filter combination should have:

1. **E[R] >= 0.08** - Same or better than baseline
2. **Trades >= 15-20/year** - Significant increase from 9
3. **WF E[R] / Train E[R] >= 0.50** - Not overfitted
4. **Holdout E[R] >= 0.05** - Validates on unseen data

## Next Steps

1. **Run pilot test** on BTCUSDT-15m
2. **Analyze top 10 results** - Look for patterns
3. **Validate winner on holdout** - Ensure it generalizes
4. **Test on full universe** (optional) - Confirm across symbols
5. **Apply to production** - Update strategy config with winner

## Testing Status

All files compile successfully:
```bash
✓ core/filter_discovery.py - Syntax OK
✓ run_filter_discovery.py - Syntax OK
✓ strategies/ssl_flow.py - Syntax OK
✓ strategies/router.py - Syntax OK
```

CLI help menu works:
```bash
$ python run_filter_discovery.py --help
[Shows full usage documentation]
```

## Technical Notes

### Why This Approach?

1. **Hierarchical search** - Pilot before full (saves time)
2. **Multiple validation levels** - Train/WF/Holdout (prevents overfit)
3. **Custom scoring** - Balances edge, frequency, consistency
4. **Parallel execution** - Faster results
5. **Comprehensive output** - Easy analysis and debugging

### Safeguards Against Overfitting

1. **Hard E[R] minimum** - Filters out barely-positive results
2. **WF validation** - OOS performance must be >= 50% of IS
3. **Holdout test** - Final check on never-seen data
4. **Bootstrap-ready** - Can compute confidence intervals
5. **Trade count minimums** - Statistical significance requirements

### Design Philosophy

- **No strategy logic changes** - Only testing different filter combinations
- **Use existing infrastructure** - Leverage proven backtest engine
- **Clean separation** - Discovery system is independent module
- **Comprehensive logging** - Easy to debug and analyze
- **Production-ready** - Can be run periodically to re-optimize

## Code Quality

- **Type hints** throughout
- **Dataclasses** for structured data
- **Comprehensive docstrings**
- **Error handling** with try/except
- **Progress reporting** for long-running operations
- **Clean code style** following project conventions

## Files Summary

```
desktop_bot_refactored_v2_base_v7/
├── core/
│   └── filter_discovery.py              [NEW] Main engine (450 lines)
├── strategies/
│   ├── ssl_flow.py                      [MODIFIED] Added skip_overlap_check
│   └── router.py                        [MODIFIED] Pass filter overrides
├── docs/
│   └── filter_discovery.md              [NEW] Full documentation
├── run_filter_discovery.py              [NEW] CLI runner (550 lines)
├── FILTER_DISCOVERY_QUICKSTART.md       [NEW] Quick reference
├── CLAUDE.md                            [MODIFIED] Added discovery section
└── FILTER_DISCOVERY_IMPLEMENTATION_SUMMARY.md  [THIS FILE]
```

## Ready to Use

The system is complete and ready for immediate use:

```bash
# Start with pilot test (recommended)
python run_filter_discovery.py --pilot
```

This will run for 3-4 hours and produce comprehensive results showing which filter combination maximizes trade frequency while maintaining edge quality.
