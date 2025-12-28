# Progress Bar Usage Guide

## Quick Start

The progress bar is automatically enabled for all filter discovery runs. No configuration needed!

### Run Pilot Discovery (Recommended First)
```bash
python run_filter_discovery.py --pilot
```

Expected output:
```
[DISCOVERY][BTCUSDT-15m] Testing 128 filter combinations...
Parallel: True, Workers: 7
Starting discovery at 2025-12-26 14:30:00
--------------------------------------------------------------------------------
[=====>                                  ] 15/128 (11.7%) | Elapsed: 34m | ETA: 4h 23m | Current: ADX+REGIME+TOUCH
```

### Run Full Discovery (All Symbols/Timeframes)
```bash
python run_filter_discovery.py --full
```

## What You'll See

### Progress Bar Components

The progress bar shows 6 key pieces of information:

1. **Visual Bar**: `[=====>                                  ]`
   - Filled portion shows completed work
   - `>` symbol shows the progress head
   - Empty space shows remaining work

2. **Counter**: `45/128`
   - Current combination / Total combinations

3. **Percentage**: `35.2%`
   - Progress as a percentage

4. **Elapsed Time**: `2h 34m`
   - How long the test has been running

5. **ETA**: `4h 23m`
   - Estimated time remaining
   - Based on average speed so far

6. **Current Item**: `ADX+REGIME+TOUCH`
   - Which filter combination is being tested
   - Shows which filters are enabled

### Example Progression

**Start (0-10%):**
```
[>                                       ] 5/128 (3.9%) | Elapsed: 12m | ETA: 4h 45m | Current: NO_FILTERS
```

**Early (10-30%):**
```
[======>                                 ] 25/128 (19.5%) | Elapsed: 58m | ETA: 4h 2m | Current: PBEMA_DIST+BODY
```

**Midpoint (40-60%):**
```
[====================>                   ] 64/128 (50.0%) | Elapsed: 3h 12m | ETA: 3h 12m | Current: ALL_FILTERS
```

**Late (70-90%):**
```
[===============================>        ] 100/128 (78.1%) | Elapsed: 5h 5m | ETA: 1h 25m | Current: REGIME+WICK
```

**Complete (100%):**
```
[========================================] 128/128 (100.0%) | Elapsed: 6h 30m | ETA: 0s
--------------------------------------------------------------------------------
[DISCOVERY][BTCUSDT-15m] Discovery complete!
Completed at 2025-12-26 21:00:00
Total combinations tested: 128
Non-overfitted: 47
```

## Understanding the Display

### Filter Combination Names

The progress bar shows which filters are enabled in the current test:

- `NO_FILTERS`: All filters disabled (baseline)
- `ADX`: Only ADX filter enabled
- `ADX+REGIME`: ADX and regime gating enabled
- `ALL_FILTERS`: All 7 filters enabled
- `ADX+REGIME+TOUCH+PBEMA_DIST+BODY+OVERLAP+WICK`: Full combination name

Full filter list (7 toggleable filters):
1. `ADX` - ADX >= threshold
2. `REGIME` - Regime gating (ADX average)
3. `TOUCH` - Baseline touch in lookback
4. `PBEMA_DIST` - Minimum distance to PBEMA
5. `BODY` - Candle body position
6. `OVERLAP` - SSL-PBEMA overlap check
7. `WICK` - Wick rejection quality

### Time Formatting

Time is displayed in the most appropriate unit:
- **Seconds**: `45s` (under 1 minute)
- **Minutes**: `15m 30s` (under 1 hour)
- **Hours**: `2h 34m` (1 hour or more)

### ETA Accuracy

The ETA becomes more accurate as more combinations are tested:
- **First 10%**: ETA may fluctuate significantly
- **After 25%**: ETA stabilizes and becomes reliable
- **After 50%**: ETA is highly accurate

## Parallel vs Serial Mode

### Parallel Mode (Default)
```bash
python run_filter_discovery.py --pilot  # Uses all CPU cores - 1
```

Progress updates as workers complete (unordered):
```
[=====>                                  ] 15/128 (11.7%) | Current: REGIME+WICK
[======>                                 ] 20/128 (15.6%) | Current: ADX+BODY
[=======>                                ] 25/128 (19.5%) | Current: NO_FILTERS
```

### Serial Mode (Debugging)
```bash
python run_filter_discovery.py --pilot --no-parallel
```

Progress updates sequentially (ordered):
```
[=====>                                  ] 15/128 (11.7%) | Current: COMBO_15
[======>                                 ] 16/128 (12.5%) | Current: COMBO_16
[======>                                 ] 17/128 (13.3%) | Current: COMBO_17
```

## Error Handling

If an error occurs during testing, you'll see:
```
[=====>                                  ] 15/128 (11.7%) | Current: ADX+REGIME

[DISCOVERY] Error evaluating ADX+REGIME: Division by zero
[=====>                                  ] 16/128 (12.5%) | Current: TOUCH+BODY
```

Errors are printed on a new line to preserve the progress bar, then progress continues.

## Tips for Long Runs

### Monitor Progress Remotely

If running on a remote server, use `screen` or `tmux`:
```bash
# Start screen session
screen -S discovery

# Run discovery
python run_filter_discovery.py --pilot

# Detach: Ctrl+A, then D

# Reattach later to check progress
screen -r discovery
```

### Check Progress at Specific Times

The progress bar updates every 0.5 seconds minimum, so you can:
- Check occasionally without missing updates
- ETA gives you a good sense of when to come back
- Completion timestamp helps plan your schedule

### Estimate Total Time

Based on pilot runs, estimate full run time:
- **Pilot (BTCUSDT-15m)**: ~6-8 hours for 128 combinations
- **Full run**: Multiply by number of symbol/timeframe pairs
- Example: 10 symbols × 7 timeframes = 70 pairs × 6 hours = 420 hours (17.5 days)

## Testing the Progress Bar

### Quick Test (6 seconds)
```bash
python test_progress_bar.py
```

Runs a fast simulation to verify:
- Progress bar renders correctly
- ETA calculation works
- Current item displays properly
- Completion handling works

### Visual Demo (12 seconds)
```bash
python demo_progress_visual.py
```

Shows snapshots at different completion stages:
- 1% (1/128)
- 12.5% (16/128)
- 25% (32/128)
- 50% (64/128)
- 75% (96/128)
- 100% (128/128)

## Troubleshooting

### Progress Bar Not Updating

**Issue**: Progress bar seems frozen
**Solution**:
- Wait at least 1 second (updates throttled to 0.5s)
- Check if process is actually running (CPU usage)
- In parallel mode, some combinations take longer than others

### Progress Bar Overwriting Text

**Issue**: Progress bar messes up previous output
**Solution**:
- This is normal behavior (in-place updates)
- Errors are printed on new line to avoid this
- Final output is always clean

### ETA Seems Wrong

**Issue**: ETA fluctuates or seems inaccurate
**Solution**:
- ETA is based on average speed so far
- Early estimates (< 10%) are less reliable
- Some combinations take longer than others
- ETA stabilizes after 25% completion

### Can't See Full Combination Name

**Issue**: Current combination is truncated
**Solution**:
- Long names are automatically truncated to 30 chars
- Full combination name is in results JSON file
- This prevents progress bar from wrapping lines

## Output Files

While progress bar shows real-time status, final results are saved to:

### results.json
Full results for all 128 combinations with metrics:
```json
{
  "run_id": "pilot_20251226_143000",
  "symbol": "BTCUSDT",
  "timeframe": "15m",
  "total_combinations": 128,
  "results": [...]
}
```

### top_10.txt
Human-readable report of top 10 combinations:
```
#1 - ADX+REGIME+TOUCH+PBEMA_DIST+BODY+OVERLAP+WICK
TRAINING (60% of data):
  PnL: $1234.56
  Trades: 45
  E[R]: 0.345
  Win Rate: 55.6%
  Score: 456.78
...
```

### filter_pass_rates.json
Diagnostic info on filter effectiveness:
```json
{
  "adx_filter": {
    "enabled": 64,
    "pass_rate": 0.50,
    "avg_score": 234.56
  },
  ...
}
```

## Summary

The progress bar provides real-time visibility during long discovery runs:
- ✓ Visual progress indicator
- ✓ Accurate ETA
- ✓ Current test visibility
- ✓ Works with parallel/serial modes
- ✓ Clean, professional output
- ✓ No configuration needed

Just run `python run_filter_discovery.py --pilot` and watch the progress!
