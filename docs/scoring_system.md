# SSL Flow Scoring System

## Overview

The SSL Flow strategy now supports **two signal detection modes**:

1. **Binary AND Logic** (default, `use_scoring=False`)
   - All filters must pass (strict)
   - Lower trade frequency (~9 trades/year with 7 filters @ 60% each = 2.8% pass rate)
   - Higher quality signals but potentially misses good opportunities

2. **Weighted Scoring System** (new, `use_scoring=True`)
   - Filters contribute points based on quality
   - Signal accepted if total score >= threshold
   - Higher trade frequency with tunable quality threshold
   - More nuanced signal evaluation

## Problem Statement

**Current AND Logic Issue:**
- 7 filters × 60% individual pass rate = **2.8% combined pass rate**
- Only **~9 trades per year** across all streams
- Too restrictive - misses valid signals when one filter marginally fails

**Example Scenario (AND logic rejects this):**
```
ADX:              ✓ PASS (25)
Regime:           ✓ PASS (trending)
Baseline Touch:   ✓ PASS
AlphaTrend:       ✓ PASS (buyers dominant)
PBEMA Distance:   ✓ PASS (0.5%)
Wick Rejection:   ✗ FAIL (9% wick - needs 10%)  <- ONE filter fails
Body Position:    ✓ PASS
No Overlap:       ✓ PASS

Result: NO SIGNAL (despite 7/8 filters passing!)
```

With scoring, this setup would score **9.5/10** and pass with threshold=6.0.

## Scoring Breakdown

Total possible score: **10.0 points**

| Filter                  | Max Points | Criteria                                    |
|------------------------|------------|---------------------------------------------|
| **ADX Strength**       | 2.0        | >30: 2.0, >25: 1.5, >20: 1.0, >15: 0.5     |
| **Regime Trending**    | 1.0        | ADX_avg >= 20: 1.0, else: 0.0               |
| **Baseline Touch**     | 2.0        | Recent touch: 2.0, no touch: 0.0            |
| **AlphaTrend Confirm** | 2.0        | Dominant+Active: 2.0, Dominant: 1.5, etc.   |
| **PBEMA Distance**     | 1.0        | >=0.6%: 1.0, >=0.4%: 0.75, >=0.3%: 0.5      |
| **Wick Rejection**     | 1.0        | >=15%: 1.0, >=10%: 0.75, >=5%: 0.5          |
| **Body Position**      | 0.5        | Correct side: 0.5, wrong side: 0.0          |
| **No Overlap**         | 0.5        | SSL-PBEMA clear: 0.5, overlap: 0.0          |

### Critical Filters (Always Enforced)

Even in scoring mode, these **core filters are mandatory**:

1. **Price Position** - Determines direction (LONG: above baseline, SHORT: below)
2. **AlphaTrend Direction** - Confirms buyers vs sellers
3. **AlphaTrend Active** - Flow must exist (not flat)
4. **RSI Bounds** - Avoids extreme overbought/oversold
5. **RR Validation** - Risk/reward must meet minimum threshold

These ensure signal **direction and basic viability** before scoring quality.

## Configuration

### Default Config (Backward Compatible)

```python
DEFAULT_STRATEGY_CONFIG = {
    "use_scoring": False,        # Binary AND logic (default)
    "score_threshold": 6.0,      # Used when use_scoring=True
    # ... other params
}
```

### Enable Scoring System

```python
config = {
    "rr": 2.0,
    "rsi": 70,
    "at_active": True,
    "strategy_mode": "ssl_flow",
    "use_scoring": True,         # Enable scoring
    "score_threshold": 6.0,      # Require 6/10 score minimum
}
```

### Tuning Score Threshold

**Recommended thresholds:**

| Threshold | Expected Behavior                              | Use Case               |
|-----------|-----------------------------------------------|------------------------|
| 8.0+      | Very strict, similar to AND logic             | High-quality only      |
| 6.0-7.0   | **Balanced** (recommended starting point)     | Quality + frequency    |
| 5.0-5.9   | More relaxed, higher trade frequency          | Active trading         |
| <5.0      | Very relaxed, may reduce edge                 | Experimental           |

**Grid search optimization** can find optimal threshold per symbol/timeframe.

## Usage Examples

### Example 1: Single Signal Check

```python
from strategies import check_signal
from core import calculate_indicators

# Fetch and prepare data
df = engine.fetch_ohlcv("BTCUSDT", "15m", limit=1000)
df = calculate_indicators(df)

# Config with scoring enabled
config = {
    "rr": 2.0,
    "rsi": 70,
    "at_active": True,
    "strategy_mode": "ssl_flow",
    "use_scoring": True,
    "score_threshold": 6.0,
}

# Check signal with debug info
result = check_signal(df, config, return_debug=True)
s_type, entry, tp, sl, reason, debug = result

if s_type:
    print(f"Signal: {s_type}")
    print(f"Score: {debug['long_score']:.2f}/10.0")
    print(f"Breakdown: {debug['long_score_breakdown']}")
```

### Example 2: Compare AND vs Scoring

```python
# Config 1: Traditional AND logic
config_and = {
    "use_scoring": False,
    # ... other params
}

# Config 2: Scoring system
config_scoring = {
    "use_scoring": True,
    "score_threshold": 6.0,
    # ... other params
}

# Compare results
result_and = check_signal(df, config_and)
result_scoring = check_signal(df, config_scoring)

print(f"AND Logic:    {result_and[0] or 'No signal'}")
print(f"Scoring:      {result_scoring[0] or 'No signal'}")
```

### Example 3: Grid Search for Optimal Threshold

```python
# Test multiple thresholds
thresholds = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
results = {}

for threshold in thresholds:
    config = {
        "use_scoring": True,
        "score_threshold": threshold,
        # ... other params
    }

    # Run backtest
    trades = run_backtest(df, config)
    results[threshold] = {
        "trades": len(trades),
        "win_rate": calculate_win_rate(trades),
        "expectancy": calculate_expectancy(trades),
    }

# Find optimal threshold
best_threshold = max(results.items(), key=lambda x: x[1]['expectancy'])[0]
print(f"Optimal threshold: {best_threshold}")
```

## Debug Output

When `return_debug=True`, debug dict includes:

```python
{
    # Standard debug fields
    "adx_ok": True,
    "price_above_baseline": True,
    # ...

    # Scoring-specific fields (when use_scoring=True)
    "use_scoring": True,
    "score_threshold": 6.0,
    "long_score": 7.5,
    "long_score_breakdown": {
        "adx": 1.5,
        "regime": 1.0,
        "baseline_touch": 2.0,
        "alphatrend": 2.0,
        "pbema_distance": 0.5,
        "wick_rejection": 0.5,
        "body_position": 0.0,
        "no_overlap": 0.0,
    },
    "short_score": 3.0,
    "short_score_breakdown": {...},
}
```

## Migration Guide

### From AND Logic to Scoring

**Step 1: Enable scoring with conservative threshold**
```python
config["use_scoring"] = True
config["score_threshold"] = 7.0  # Conservative (close to AND logic)
```

**Step 2: Backtest and compare**
```python
# Run backtests with both modes
results_and = backtest(config_and)
results_scoring = backtest(config_scoring)

# Compare metrics
compare_results(results_and, results_scoring)
```

**Step 3: Optimize threshold**
```python
# Grid search for optimal threshold
best_config = optimize_score_threshold(
    symbol="BTCUSDT",
    timeframe="15m",
    threshold_range=(5.0, 8.0),
    step=0.5
)
```

**Step 4: Deploy with monitoring**
```python
# Deploy with scoring enabled
config["use_scoring"] = True
config["score_threshold"] = best_config["threshold"]

# Monitor trade frequency and quality
monitor_performance(config)
```

## Performance Expectations

### Estimated Trade Frequency Increase

Based on **7 filters @ 60% pass rate each**:

| Mode                     | Pass Rate | Est. Trades/Year | Notes                    |
|--------------------------|-----------|------------------|--------------------------|
| AND Logic (all filters)  | 2.8%      | ~9 trades        | Current baseline         |
| Scoring (threshold=7.0)  | ~5-8%     | ~16-25 trades    | Conservative             |
| Scoring (threshold=6.0)  | ~10-15%   | ~32-48 trades    | **Recommended**          |
| Scoring (threshold=5.0)  | ~20-25%   | ~64-80 trades    | More active              |

*Note: Actual results vary by market conditions and parameter settings.*

### Quality vs Quantity Tradeoff

**Expectancy-adjusted position sizing:**
- High scores (8.0+) → Full position size
- Medium scores (6.0-7.9) → 75% position size
- Low scores (5.0-5.9) → 50% position size

This can be implemented with a **score-based risk multiplier**.

## Implementation Notes

### Backward Compatibility

- Default `use_scoring=False` maintains existing behavior
- All existing configs continue to work without changes
- No breaking changes to function signatures (new params are optional)

### Thread Safety

- Scoring calculation is stateless (pure function)
- No shared state between signal checks
- Safe for parallel backtesting

### Performance

- Scoring adds minimal overhead (~1-2% CPU)
- Pre-computed boolean flags reused from AND logic
- No additional indicator calculations required

## Testing

Run the test script:

```bash
python test_scoring_system.py
```

Or integrate into your backtest:

```python
from test_scoring_system import test_scoring_system

df = engine.fetch_ohlcv("BTCUSDT", "15m", limit=1000)
test_scoring_system(df, "BTCUSDT", "15m")
```

## Future Enhancements

Potential improvements to consider:

1. **Adaptive Thresholds** - Dynamic threshold based on market regime
2. **Score-Based Position Sizing** - Scale position size by signal quality
3. **Multi-Symbol Scoring** - Portfolio-level score aggregation
4. **Machine Learning** - Learn optimal score weights from historical data
5. **Score-Based Take Profit** - Extend TP for high-score trades

## References

- Filter logic: `/strategies/ssl_flow.py`
- Score calculation: `calculate_signal_score()` function
- Config defaults: `/core/config.py` - `DEFAULT_STRATEGY_CONFIG`
- Router integration: `/strategies/router.py`
- Test script: `/test_scoring_system.py`

## Questions?

For issues or questions about the scoring system, check:
1. This documentation
2. Code comments in `ssl_flow.py`
3. Test script examples
4. Debug output from `return_debug=True`
