# ROC (Rate of Change) Momentum Filter Implementation

**Version:** v1.10.0
**Date:** December 28, 2025
**Status:** ENABLED by default

## Overview

The ROC Momentum Filter prevents counter-trend entries in the SSL Flow trading strategy by measuring price velocity and blocking trades that go against strong momentum.

## Problem Statement

Analysis of losing trades showed that **8 out of 14 losing trades (57%)** were TREND ERRORS - counter-trend entries where:
- LONG signals were taken during strong downtrends (buying falling knives)
- SHORT signals were taken during strong uptrends (shorting into strength)

ADX-based solutions failed because ADX measures trend **strength**, not **direction**. ROC is different because it measures price **velocity** with direction.

## Implementation Details

### Location

File: `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/strategies/ssl_flow.py`

### New Function: `calculate_roc_filter()`

**Location:** Line 321-381 (after `check_sl_sweep_risk()`)

```python
def calculate_roc_filter(df: pd.DataFrame, index: int = -2,
                         roc_period: int = 10,
                         roc_threshold: float = 2.5) -> Tuple[bool, bool, float]:
    """
    Rate of Change momentum filter to prevent counter-trend entries.

    ROC = (close - close[n]) / close[n] * 100

    Returns:
        (long_allowed: bool, short_allowed: bool, roc_value: float)
    """
```

**Logic:**
- Calculates price change percentage over `roc_period` bars
- Blocks SHORT if ROC > +threshold (strong uptrend - don't short into strength)
- Blocks LONG if ROC < -threshold (strong downtrend - don't buy falling knife)

**Edge Case Handling:**
- Insufficient data (< roc_period bars): Allow all trades (conservative default)
- NaN values: Allow all trades (conservative default)
- Division by zero: Allow all trades
- Extreme ROC (>100%): Allow all trades (likely bad data)

### New Parameters in `check_ssl_flow_signal()`

**Location:** Lines 402-404

```python
use_roc_filter: bool = True,  # v1.10.0: ROC Momentum Filter - ENABLED by default
roc_period: int = 10,          # v1.10.0: ROC lookback period (default 10 bars)
roc_threshold: float = 2.5,    # v1.10.0: ROC threshold percentage (default 2.5%)
```

**Parameter Meanings:**
- `use_roc_filter`: Enable/disable the filter (default: True)
- `roc_period`: Number of bars to look back for ROC calculation (default: 10)
- `roc_threshold`: Percentage threshold to reject counter-trend trades (default: 2.5%)

### Filter Integration

**Location:** Lines 734-761 (after Trend Strength Filter, before LONG/SHORT signal determination)

```python
# ================= ROC MOMENTUM FILTER (v1.10.0) =================
roc_long_ok = True
roc_short_ok = True
roc_value = 0.0

if use_roc_filter:
    roc_long_ok, roc_short_ok, roc_value = calculate_roc_filter(
        df, index, roc_period, roc_threshold
    )
    debug_info["roc_value"] = roc_value
    debug_info["roc_long_ok"] = roc_long_ok
    debug_info["roc_short_ok"] = roc_short_ok
    debug_info["roc_period"] = roc_period
    debug_info["roc_threshold"] = roc_threshold

    # Early rejection with explicit reason
    if price_above_baseline and at_buyers_dominant and not roc_long_ok:
        return _ret(None, None, None, None, f"ROC Filter: Strong downtrend (ROC={roc_value:.2f}%)")

    if price_below_baseline and at_sellers_dominant and not roc_short_ok:
        return _ret(None, None, None, None, f"ROC Filter: Strong uptrend (ROC={roc_value:.2f}%)")
```

### Signal Condition Updates

**LONG Signal (Line 772):**
```python
is_long = (
    price_above_baseline and
    at_buyers_dominant and
    baseline_touch_long and
    body_above_baseline and
    long_pbema_distance >= min_pbema_distance and
    (skip_wick_rejection or long_rejection) and
    pbema_above_baseline and
    long_trend_ok and
    roc_long_ok  # NEW: v1.10.0 ROC momentum filter
)
```

**SHORT Signal (Line 805):**
```python
is_short = (
    price_below_baseline and
    at_sellers_dominant and
    baseline_touch_short and
    body_below_baseline and
    short_pbema_distance >= min_pbema_distance and
    (skip_wick_rejection or short_rejection) and
    pbema_below_baseline and
    short_trend_ok and
    roc_short_ok  # NEW: v1.10.0 ROC momentum filter
)
```

## Behavior Examples

### Example 1: Strong Uptrend (ROC = +6.4%)

```
Price: $47,000 -> $50,000 over 10 bars
ROC = +6.4%

Result:
- LONG allowed: True (uptrend supports LONG)
- SHORT allowed: False (blocked - don't short into strength)
- Rejection reason: "ROC Filter: Strong uptrend (ROC=+6.38%)"
```

### Example 2: Strong Downtrend (ROC = -6.0%)

```
Price: $50,000 -> $47,000 over 10 bars
ROC = -6.0%

Result:
- LONG allowed: False (blocked - don't catch falling knife)
- SHORT allowed: True (downtrend supports SHORT)
- Rejection reason: "ROC Filter: Strong downtrend (ROC=-6.00%)"
```

### Example 3: Sideways Market (ROC = +0.5%)

```
Price: $100 -> $100.50 over 10 bars
ROC = +0.5%

Result:
- LONG allowed: True (weak momentum, within threshold)
- SHORT allowed: True (weak momentum, within threshold)
- No rejection
```

## Debug Information

When `return_debug=True`, the following keys are added to `debug_info`:

```python
{
    "roc_value": float,        # Calculated ROC percentage
    "roc_long_ok": bool,       # Whether LONG is allowed by ROC filter
    "roc_short_ok": bool,      # Whether SHORT is allowed by ROC filter
    "roc_period": int,         # Period used for calculation
    "roc_threshold": float,    # Threshold used for rejection
}
```

## Testing

A comprehensive test suite is provided in `/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/test_roc_filter.py`

**Test Coverage:**
1. Basic uptrend detection (blocks SHORT)
2. Basic downtrend detection (blocks LONG)
3. Sideways market (allows both)
4. Edge cases (insufficient data, NaN values)
5. Threshold tuning (1.0%, 2.5%, 5.0%, 7.5%)
6. Realistic crypto scenarios (BTC pump/dump)

**Run Tests:**
```bash
python3 test_roc_filter.py
```

**Expected Output:**
```
ALL TESTS PASSED ✓

ROC Filter Summary:
  • Blocks SHORT when ROC > +2.5% (strong uptrend)
  • Blocks LONG when ROC < -2.5% (strong downtrend)
  • Allows both when |ROC| < 2.5% (weak momentum)
  • Handles edge cases gracefully (insufficient data, NaN)
  • Default: roc_period=10, roc_threshold=2.5%
  • Enabled by default in ssl_flow.py (v1.10.0)
```

## Performance Characteristics

**Computation:** O(1) - single array access, one division, one subtraction
**Memory:** O(1) - no additional memory allocation
**Optimization:** Uses numpy arrays for fast vectorized operations

## Expected Impact

Based on historical analysis:
- **Target:** Should block 6-7 of the 8 trend error trades
- **Trade Reduction:** Minimal (only blocks counter-trend entries)
- **Win Rate:** Expected to improve by filtering out high-risk counter-trend setups
- **Drawdown:** Expected to reduce by avoiding catching falling knives / shorting into pumps

## Configuration Recommendations

### Default Settings (Conservative)
```python
use_roc_filter=True
roc_period=10
roc_threshold=2.5
```

### Aggressive Settings (More Filtering)
```python
use_roc_filter=True
roc_period=10
roc_threshold=1.5  # Lower threshold = more filtering
```

### Relaxed Settings (Less Filtering)
```python
use_roc_filter=True
roc_period=10
roc_threshold=5.0  # Higher threshold = less filtering
```

### Longer Timeframe
```python
use_roc_filter=True
roc_period=20      # Longer period = smoother, less sensitive
roc_threshold=2.5
```

## Integration with Existing Filters

The ROC filter works alongside existing filters:

| Filter | Purpose | ROC Relationship |
|--------|---------|------------------|
| ADX | Trend strength | Complementary (ADX = strength, ROC = direction) |
| Regime Gating | Window-level market state | Complementary (Regime = overall, ROC = immediate) |
| Trend Alignment | Broader trend check | Similar but simpler (ROC is velocity-based) |
| AlphaTrend | Flow confirmation | Complementary (AT = buyers/sellers, ROC = momentum) |

**Filter Execution Order:**
1. ADX Filter
2. Regime Gating
3. Hour Filter
4. Baseline Position
5. AlphaTrend
6. PBEMA Distance
7. Wick Rejection
8. Trend Alignment (if enabled)
9. **ROC Filter** (NEW - v1.10.0)
10. LONG/SHORT Determination

## Disabling the Filter

To disable the ROC filter (e.g., for A/B testing):

```python
signal = check_ssl_flow_signal(
    df,
    use_roc_filter=False,  # Disable ROC filter
    # ... other parameters
)
```

## Version History

- **v1.10.0** (Dec 28, 2025): Initial implementation
  - Added `calculate_roc_filter()` function
  - Added `use_roc_filter`, `roc_period`, `roc_threshold` parameters
  - Integrated into LONG/SHORT signal logic
  - Enabled by default
  - Full test coverage

## Future Enhancements (Potential)

1. **Dynamic Threshold**: Adjust ROC threshold based on volatility (ATR)
2. **Multi-Period ROC**: Check ROC across multiple timeframes
3. **ROC Divergence**: Detect when price and momentum diverge
4. **Adaptive Period**: Automatically adjust `roc_period` based on market conditions

## Notes

- The filter is **conservative** by default - when in doubt, it allows trades
- Early rejection provides clear visibility in logs/debug output
- Filter is part of the AND logic chain, so it can be selectively disabled without breaking other logic
- ROC calculation uses simple percentage change (not log returns) for interpretability
