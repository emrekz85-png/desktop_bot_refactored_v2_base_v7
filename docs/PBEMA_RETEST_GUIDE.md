# PBEMA Retest Strategy - Implementation Guide

## Overview

The PBEMA Retest Strategy is a new trading pattern identified from real trade analysis. It trades the PBEMA cloud as a dynamic support/resistance level after price breaks through it.

**Key Insight:** After price breaks above/below PBEMA, it often returns to "retest" the level. A successful retest (bounce/rejection) provides a high-probability entry opportunity.

**Evidence:** Analysis of 18 real trades showed this pattern accounts for **~33% of profitable setups** that the original SSL Flow strategy misses.

---

## Real Trade Examples

From the Strategy_Real_Trades analysis:

### NO 7 - 15m BTC (RIGHT CHART)
**Pattern:** PBEMA as Support after Breakout
- Price broke above PBEMA cloud
- Retested PBEMA as support
- **Entry:** On bounce from PBEMA
- **TP:** Momentum exhaustion
- **Annotation:** "PBEMADAN güçlü fiyat sekmesi yaşanabileceği icin long entry"

### NO 11 - 15m BTC
**Pattern:** PBEMA Breakout + Retest
- Price broke above PBEMA
- Retested PBEMA from above
- **Entry:** On successful retest
- **Annotation:** "Fiyat PBEMA bandını kazanıyor ve retest ediyor, bu retest sırasında entry aldım"

### NO 12-13 - 15m BTC
**Pattern:** Multiple PBEMA Retests
- Price established above PBEMA
- **Multiple retests** confirmed strength
- **Entry:** On each retest
- **Annotation:** "PBEMA retestinde entry aldım"

### NO 18 - 5m BTC
**Pattern:** Proven PBEMA Level
- Price established above PBEMA
- Multiple successful retests (3+)
- **Entry:** On next retest with confirmed momentum
- **Annotation:** "Fiyat PBEMA üzerinde yer edinmiş, bir çok kez retest edip entry aldım"

---

## How It Works

### Step 1: Breakout Detection
The strategy searches recent history (default: 20 candles) for a PBEMA breakout:

**Bullish Breakout:**
- Previous candle: `close < PBEMA_bot`
- Current candle: `close > PBEMA_top`
- Distance: `(close - PBEMA_top) / PBEMA_top >= 0.5%`

**Bearish Breakout:**
- Previous candle: `close > PBEMA_top`
- Current candle: `close < PBEMA_bot`
- Distance: `(PBEMA_bot - close) / PBEMA_bot >= 0.5%`

### Step 2: Retest Confirmation
Check if price is currently touching the PBEMA level:

**LONG Setup (bullish breakout retest):**
- Price is retesting PBEMA from above (support test)
- `low <= PBEMA_top * (1 + 0.3%)`  # Within tolerance
- `close > PBEMA_mid`  # Still above cloud

**SHORT Setup (bearish breakout retest):**
- Price is retesting PBEMA from below (resistance test)
- `high >= PBEMA_bot * (1 - 0.3%)`  # Within tolerance
- `close < PBEMA_mid`  # Still below cloud

### Step 3: Bounce/Rejection via Wick
Confirm the level is holding via wick rejection:

**LONG:** Lower wick >= 15% of candle range
**SHORT:** Upper wick >= 15% of candle range

### Step 4: Entry Execution

**LONG Entry:**
- Entry: Current close
- TP: SSL Baseline (or +1.5% if baseline below entry)
- SL: Below PBEMA_bot - 0.3%

**SHORT Entry:**
- Entry: Current close
- TP: SSL Baseline (or -1.5% if baseline above entry)
- SL: Above PBEMA_top + 0.3%

---

## Usage

### Basic Usage (Minimal Filters)

```python
from strategies import check_pbema_retest_signal

# Simple setup - no extra filters
signal_type, entry, tp, sl, reason = check_pbema_retest_signal(
    df,
    index=-2,
    min_rr=1.5,  # Minimum 1.5:1 risk/reward
)

if signal_type == "LONG":
    print(f"LONG at {entry}, TP {tp}, SL {sl}")
elif signal_type == "SHORT":
    print(f"SHORT at {entry}, TP {tp}, SL {sl}")
```

### Conservative Setup (Multiple Filters)

```python
# Require AlphaTrend confirmation + multiple retests
signal_type, entry, tp, sl, reason = check_pbema_retest_signal(
    df,
    index=-2,
    min_rr=2.0,  # Higher RR requirement
    require_at_confirmation=True,  # AlphaTrend must confirm
    require_multiple_retests=True,  # Need 2+ retests
    min_retests=2,
)
```

### Debug Mode

```python
# Get full debug information
signal_type, entry, tp, sl, reason, debug = check_pbema_retest_signal(
    df,
    index=-2,
    return_debug=True,
)

if signal_type:
    print(f"Signal: {signal_type}")
    print(f"Breakout direction: {debug['breakout_direction']}")
    print(f"Breakout distance: {debug['breakout_distance']*100:.2f}%")
    print(f"Retest count: {debug['retest_count']}")
    print(f"Wick ratio: {debug['wick_ratio']:.2f}")
```

---

## Configuration

All parameters can be customized via `core/config.py::PBEMA_RETEST_CONFIG`:

```python
PBEMA_RETEST_CONFIG = {
    # Core parameters
    "min_rr": 1.5,                    # Minimum R:R
    "breakout_lookback": 20,          # Candles to search for breakout
    "min_breakout_strength": 0.005,   # Min 0.5% beyond PBEMA
    "retest_tolerance": 0.003,        # 0.3% touch tolerance
    "min_wick_ratio": 0.15,           # 15% wick rejection

    # TP/SL
    "tp_target": "baseline",          # "baseline" or "percentage"
    "tp_percentage": 0.015,           # 1.5% if using percentage
    "sl_buffer": 0.003,               # 0.3% buffer beyond PBEMA

    # Optional filters
    "require_at_confirmation": False,
    "require_multiple_retests": False,
    "min_retests": 2,
}
```

---

## Testing

### Quick Test
Test on recent BTCUSDT 15m data:

```bash
python test_pbema_retest.py
```

### Single Candle Debug
Test with full debug output:

```bash
python test_pbema_retest.py --single
```

### Expected Output
```
[2025-01-04 10:15:00] LONG Signal #1
  Entry: 95234.50 | TP: 95756.20 | SL: 94890.30 | R:R = 1.52
  Reason: PBEMA_RETEST_LONG(R:1.52)
  Breakout: BULLISH at candle 485
  Breakout distance: 0.58%
  Retest count: 1
  Wick ratio: 0.18
```

---

## Integration with run.py

The PBEMA Retest strategy can be used:

### 1. Standalone Strategy
Test PBEMA retest signals independently:

```python
from strategies import check_pbema_retest_signal

for i in range(60, len(df) - 10):
    signal_type, entry, tp, sl, reason = check_pbema_retest_signal(df, index=i)
    if signal_type:
        # Trade logic here
        pass
```

### 2. Combined with SSL Flow
Run both strategies in parallel for maximum coverage:

```python
from strategies import check_ssl_flow_signal, check_pbema_retest_signal

# Try SSL Flow first
ssl_signal = check_ssl_flow_signal(df, index=i)

# If no SSL signal, try PBEMA Retest
if ssl_signal[0] is None:
    pbema_signal = check_pbema_retest_signal(df, index=i)
    if pbema_signal[0]:
        # Use PBEMA signal
        pass
```

### 3. Filter Discovery Mode
Add "pbema_retest" as a filter option in run.py:

```python
# In run.py ALL_FILTERS
ALL_FILTERS = [
    "at_flat_filter", "adx_filter", "at_binary",
    "ssl_touch", "rsi_filter", "pbema_distance",
    "overlap_check", "body_position", "wick_rejection",
    "min_sl_filter",
    "pbema_retest",  # NEW: PBEMA retest as alternative strategy
]
```

---

## Performance Expectations

Based on real trade analysis:

### Coverage
- **SSL Flow:** Captures ~67% of setups (trend-following entries)
- **PBEMA Retest:** Captures ~33% of setups (retest entries)
- **Combined:** ~100% coverage of profitable patterns

### Signal Characteristics
- **Frequency:** Lower than SSL Flow (requires breakout first)
- **Quality:** Higher win rate expected (established level)
- **Risk/Reward:** Typically 1.5-2.5 R:R
- **Best Timeframes:** 15m, 1h (enough retests, not too noisy)

### Recommended Configuration
Start with **baseline config** (minimal filters):
- `require_at_confirmation = False`
- `require_multiple_retests = False`
- `min_rr = 1.5`

Then optimize based on backtest results.

---

## Next Steps

1. **Run Test Script:** `python test_pbema_retest.py` to see live examples
2. **Backtest Strategy:** Add to run.py and run full pipeline
3. **Compare Performance:** SSL Flow vs PBEMA Retest vs Combined
4. **Optimize Parameters:** Use filter discovery to find best config
5. **Implement Remaining Patterns:** Patterns 1, 3-7 from analysis

---

## Notes

### Differences from SSL Flow
| Aspect | SSL Flow | PBEMA Retest |
|--------|----------|--------------|
| **Entry Trigger** | SSL baseline touch | PBEMA retest after breakout |
| **TP Target** | PBEMA cloud | SSL baseline |
| **Trade Direction** | With trend (SSL direction) | Reversal from PBEMA |
| **Setup Frequency** | Higher | Lower |
| **Setup Quality** | Trend confirmation | Level confirmation |

### When to Use Which
- **SSL Flow:** During clear trends (price following SSL)
- **PBEMA Retest:** After momentum shifts (price broke PBEMA, now retesting)
- **Combined:** Maximum coverage (catches both patterns)

### Common Mistakes to Avoid
1. **Don't trade both signals simultaneously** - Pick one per candle
2. **Respect the breakout requirement** - No breakout = no retest trade
3. **Wait for wick confirmation** - Don't enter on touch alone
4. **Check timeframe alignment** - Works best on 15m+, noisy on 5m

---

## Support

For questions or issues:
1. Check `strategies/pbema_retest.py` for implementation details
2. Review real trade examples in `Strategy_Real_Trades/`
3. Run test script with `--single` flag for debug output
4. See ultra-deep analysis report for pattern details
