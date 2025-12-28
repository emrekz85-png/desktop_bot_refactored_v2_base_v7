# Filter Discovery Quick Start Guide

## TL;DR

Find the optimal filter combination to increase trade frequency from 9/year while maintaining edge.

```bash
# Run pilot test (3-4 hours)
python run_filter_discovery.py --pilot

# Check top results
cat data/filter_discovery_runs/pilot_*/top_10.txt

# Validate winner on holdout
python run_filter_discovery.py --validate 0 --results data/filter_discovery_runs/pilot_*/results.json
```

## Problem

SSL Flow strategy has 6+ AND filters → only 9 trades/year (over-filtering)

## Solution

Test 2^6 = 64 filter combinations:
- 60% data for search
- 20% for walk-forward validation
- 20% for holdout test

## Toggleable Filters

1. ADX filter (trend strength)
2. Regime gating (window-level ADX average)
3. Baseline touch (recent retest check)
4. PBEMA distance (room for profit)
5. Body position (candle confirmation)
6. SSL-PBEMA overlap (flow existence)

**CORE filters (always ON):**
- AlphaTrend dominance
- Price vs baseline position
- RR validation

## Commands

```bash
# Pilot (BTCUSDT-15m only, ~3-4 hours)
python run_filter_discovery.py --pilot

# Full (all symbols/TFs, much slower)
python run_filter_discovery.py --full

# Validate combination #N on holdout
python run_filter_discovery.py --validate N --results path/to/results.json

# Advanced options
python run_filter_discovery.py --pilot --candles 50000  # More data
python run_filter_discovery.py --pilot --no-parallel    # Serial execution (debug)
python run_filter_discovery.py --pilot --workers 4      # Limit workers
```

## Output Files

```
data/filter_discovery_runs/{run_id}/
├── results.json              # Full results (all 64 combinations)
├── top_10.txt                # Human-readable top 10
└── filter_pass_rates.json    # Filter statistics
```

## Success Criteria

- E[R] >= 0.08 (maintain edge)
- Trades >= 15-20/year (increase from 9)
- WF E[R] / Train E[R] >= 0.50 (not overfitted)
- Holdout E[R] >= 0.05 (validates)

## Example Workflow

```bash
# 1. Run pilot
python run_filter_discovery.py --pilot

# 2. View results
cat data/filter_discovery_runs/pilot_20250126_143022/top_10.txt

# Output shows:
# #1 - ADX+TOUCH+PBEMA_DIST+BODY+OVERLAP
#   Train: $250.50, 25 trades, E[R]=0.18, Score=45.23
#   WF:    $120.30, 12 trades, E[R]=0.15, Ratio=0.83
#   ✓ Valid (not overfit)

# 3. Validate winner on holdout
python run_filter_discovery.py --validate 0 \
  --results data/filter_discovery_runs/pilot_20250126_143022/results.json

# Output shows:
# Holdout E[R]: 0.14 (good - validates the combination)

# 4. Apply to config (manual step)
# Update DEFAULT_STRATEGY_CONFIG with winning filter settings
```

## Interpreting Results

### Good Signs
- Top combos have similar E[R] on train and WF (low overfit)
- Trade count significantly higher than baseline 9
- Holdout E[R] close to WF E[R]

### Bad Signs
- Train E[R] >> WF E[R] (overfitting)
- Very few trades (< 15) even with filters removed
- Negative holdout E[R] (doesn't generalize)

## Filter Pass Rates

The `filter_pass_rates.json` shows which filters appear in top combinations:

```json
{
  "regime_gating": {
    "enabled": 20,
    "pass_rate": 0.31,      // Low = often disabled in best combos
    "avg_score": 28.3
  },
  "baseline_touch": {
    "enabled": 52,
    "pass_rate": 0.81,      // High = important filter
    "avg_score": 35.7
  }
}
```

**Interpretation:**
- Low pass_rate → Filter might be too strict
- High pass_rate → Filter is important
- avg_score shows performance when enabled

## Next Steps

After finding winner:

1. Check if pattern holds across symbols (run --full)
2. Test on longer history (--candles 50000)
3. Update strategy config with winning combination
4. Monitor live performance

## Common Issues

**Q: All combinations have negative scores**
A: Data might be too short. Try --candles 50000

**Q: Very slow execution**
A: Use --no-parallel for debugging, or reduce --candles

**Q: Holdout much worse than WF**
A: Overfit detected. Try different data period or scoring

## Technical Details

See `docs/filter_discovery.md` for:
- Implementation details
- Scoring function rationale
- Integration with existing code
- Troubleshooting guide

## Files Modified

- `strategies/ssl_flow.py` - Added skip_overlap_check parameter
- `strategies/router.py` - Pass filter overrides to signal check
- `core/filter_discovery.py` - Main discovery engine
- `run_filter_discovery.py` - CLI runner

## Quant Analyst Notes

This implementation follows the exact specifications:
- 6 toggleable filters (64 combinations)
- 3 core filters (always ON)
- Hierarchical search (pilot → full)
- Custom scoring function (E[R], frequency, Sharpe, WR)
- Overfit safeguards (60/20/20 split, ratio check)
- Bootstrap-ready for CI (holdout test)
