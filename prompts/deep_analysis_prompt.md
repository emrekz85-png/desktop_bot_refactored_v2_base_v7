# Deep Forensic Analysis Prompt: Manual-to-Automated Translation Gap

## CRITICAL CONTEXT - READ THIS FIRST

**THE ACTUAL SITUATION:**
1. The user has NOT done any live trading yet
2. Even rolling walk-forward (WF) backtests are NOT producing positive results
3. The strategy IS PROFITABLE when the user trades it MANUALLY on TradingView charts
4. The problem is: Manual trading works, but the automated bot does NOT replicate it correctly
5. The goal is to make rolling WF tests produce positive results FIRST before going live

**THE REAL PROBLEM TO SOLVE:**
- Why does the SSL Flow strategy work when traded manually but fail when automated?
- Is the signal logic correctly translating what the user sees on charts?
- Is there a mismatch between TradingView indicators and Python calculations?
- Is the rolling WF testing methodology itself flawed?
- Are there bugs in how the optimizer evaluates and selects configs?

---

## Mission Brief

You are a team of elite quantitative analysts, software architects, and TradingView-to-Python translation specialists conducting a forensic investigation of a cryptocurrency futures trading bot that SHOULD be profitable (based on manual trading success) but is systematically failing when automated.

**The Core Mystery:**
- Manual trader sees SSL HYBRID + AlphaTrend setup on TradingView -> Takes trade -> Profitable
- Bot runs same logic in Python -> Takes different trades OR same trades with different outcomes -> Unprofitable

**Performance Gap:**
- Manual trading expectation: ~$150-200 profit over 6-month period
- Current Rolling WF results: -$150 to -$270 (negative)
- This is a ~$300-400 gap that needs explanation

---

## Investigation Framework

### Phase 1: TradingView-to-Python Indicator Parity

**CRITICAL HYPOTHESIS:** The Python indicator calculations may not match TradingView's exactly.

**Files to Examine:**
- `core/indicators.py` - All indicator calculations
- Compare against TradingView Pine Script implementations

**Key Indicators to Verify:**

#### 1. SSL HYBRID Baseline (HMA60)
```python
# Current implementation (indicators.py line 319):
df["baseline"] = ta.hma(df["close"], length=hma_length)
# hma_length varies by timeframe: 5m=75, 15m=60, 30m=55, 1h=45, 4h=40
```

**Questions:**
- Is the HMA calculation in `pandas_ta` identical to TradingView's `hma()` function?
- Does TradingView SSL HYBRID actually use HMA, or some variant (WMA-based)?
- Are the timeframe-adaptive lookback periods (75/60/55/45/40) matching TradingView settings?

**Verification Method:**
1. Take a specific timestamp (e.g., BTCUSDT 2025-06-15 14:00 UTC on 15m chart)
2. Get the HMA60 value from TradingView
3. Get the HMA60 value from Python code
4. Compare - they should be IDENTICAL

#### 2. AlphaTrend Indicator
```python
# Current implementation (indicators.py lines 35-231):
# Uses: MFI (or RSI fallback), ATR, momentum threshold of 50
alphatrend[i] = max(alphatrend[i - 1], upT[i]) if momentum[i] >= 50 else min(alphatrend[i - 1], downT[i])
```

**TradingView Pine Script Reference:**
```pine
AlphaTrend := (RSI >= 50) ? max(upT, nz(AlphaTrend[1])) : min(downT, nz(AlphaTrend[1]))
```

**Questions:**
- Does TradingView use MFI or RSI? Code uses MFI if volume available, RSI otherwise
- Is the ATR calculation period (14) matching TradingView?
- Is the coefficient (1.0) matching TradingView's multiplier?
- How does TradingView handle the `nz()` function for NaN values?

**Dominance Logic:**
```python
# at_buyers_dominant = alphatrend > alphatrend[2] (line rising)
# at_sellers_dominant = alphatrend[2] > alphatrend (line falling)
df['at_buyers_dominant'] = df['at_buyers'] > df['at_sellers']
df['at_sellers_dominant'] = df['at_sellers'] > df['at_buyers']
```

**Critical Question:** When the user sees "buyers dominant" on TradingView, does the Python code also detect "buyers dominant" at the SAME candle?

#### 3. PBEMA Cloud (EMA200)
```python
# Current implementation (indicators.py lines 273-275):
df["pb_ema_top"] = ta.ema(df["high"], length=200)
df["pb_ema_bot"] = ta.ema(df["close"], length=200)
```

**Questions:**
- Is TradingView's PBEMA using the same formula (EMA of high vs EMA of close)?
- Are there any smoothing differences?
- Is length=200 correct for all timeframes?

---

### Phase 2: Signal Generation Logic Audit

**File:** `strategies/ssl_flow.py` (836 lines)

**Manual Trading Logic (What User Sees):**
1. Price above SSL baseline (HMA60) -> Bullish flow
2. AlphaTrend buyers dominant (blue line above red) -> Confirmed
3. Price recently touched/retested baseline -> Entry opportunity
4. PBEMA cloud above price -> Room for TP
5. Take the LONG trade

**Automated Logic (What Code Does):**
```python
# ssl_flow.py lines 582-591
is_long = (
    price_above_baseline and
    at_buyers_dominant and
    baseline_touch_long and
    (skip_body_position or body_above_baseline) and
    long_pbema_distance >= min_pbema_distance and
    (skip_wick_rejection or long_rejection) and
    pbema_above_baseline and
    baseline_ever_lost_bullish
)
```

**KEY QUESTIONS:**

1. **Baseline Touch Detection (lines 441-456):**
```python
# Check if low touched baseline in last 5 candles
lookback_lows = _low_arr[lookback_start:abs_index + 1]
lookback_baselines_long = _baseline_arr[lookback_start:abs_index + 1]
baseline_touch_long = np.any(lookback_lows <= lookback_baselines_long * (1 + ssl_touch_tolerance))
```
- Does "touch" mean the SAME thing to a manual trader?
- Tolerance is 0.3% - is this too tight or too loose?
- Manual trader might see "close enough" visually but code says "no touch"

2. **PBEMA Distance Requirement (line 587):**
```python
long_pbema_distance >= min_pbema_distance  # default 0.004 = 0.4%
```
- This requires 0.4% minimum distance to PBEMA
- Filter analysis shows 18.8% pass rate - VERY restrictive
- Does manual trader have a mental "minimum distance" rule? Probably not.

3. **AlphaTrend Flat Detection (lines 509-512):**
```python
if at_is_flat and not skip_at_flat_filter:
    return _ret(None, None, None, None, "AlphaTrend Flat (No Flow)")
```
- "Flat" = less than 0.1% change over 5 candles
- Manual trader might not notice such small flatness
- This could be blocking trades the manual trader would take

4. **Regime Gating (lines 379-398):**
```python
# Window-level regime detection using ADX average
adx_window = df["adx"].iloc[regime_start:abs_index]  # Note: excludes current bar (look-ahead fix)
adx_avg = float(adx_window.mean())
if regime == "RANGING":  # adx_avg < 20
    return _ret(None, None, None, None, f"RANGING Regime (ADX_avg={adx_avg:.1f})")
```
- This blocks ALL trades when market is ranging
- Manual trader doesn't have this filter!
- Could be blocking profitable counter-trend bounces

**TRADE COMPARISON EXERCISE:**

For each of the following scenarios, trace through the code:

1. **A Trade Manual Trader Would Take:**
   - Timestamp: [pick a specific winning manual trade]
   - What signals does TradingView show?
   - What does the Python code see at the same timestamp?
   - WHERE does the code reject the signal?

2. **A Trade Bot Takes But Shouldn't:**
   - Find a losing trade in backtest results
   - What made the code think this was a good entry?
   - Would a manual trader have taken this trade?

---

### Phase 3: Entry Timing Investigation

**Critical Bug Pattern:** Signal on candle N, but entry executed incorrectly.

**Current Implementation (optimizer.py lines 617-641):**
```python
# Signal generated at index i (candle close)
s_type, s_entry, s_tp, s_sl, s_reason = TradingEngine.check_signal(df, config=config, index=i, ...)

# Entry uses NEXT candle open
entry_open = float(opens[i + 1])

# RR re-validation with actual entry price
actual_rr = actual_reward / actual_risk
if actual_rr < min_rr * 0.9:  # 10% tolerance
    continue
```

**Questions:**
1. When does a manual trader enter? On close of signal candle or open of next?
2. Is the 10% RR tolerance causing RR drift that invalidates good trades?
3. Is there a mismatch between signal's `s_entry` (close) and actual `entry_open` (next open)?

**Signal Index Investigation:**
```python
# ssl_flow.py line 172
def check_ssl_flow_signal(df, index=-2, ...)  # Default: second-to-last candle
```

**Why index=-2?**
- Current candle (index=-1) is not closed yet
- But index=-2 means we're looking at a candle that closed 1 period ago
- Is this causing 1-bar delay in signals vs what manual trader sees?

---

### Phase 4: Rolling Walk-Forward Methodology

**Current Implementation (rolling_wf.py):**
- 60-day lookback window for optimization
- 7-day forward window for trading
- Weekly re-optimization

**Potential Issues:**

1. **Optimizer-Filter Death Spiral (from CLAUDE.md):**
   - When filters are relaxed, optimizer selects DIFFERENT configs
   - These different configs may be worse quality
   - Result: Relaxing filters makes performance WORSE, not better

2. **Insufficient Training Data:**
   - 60 days = ~4000 candles on 15m timeframe
   - With 13+ AND filters, only 0.0076% of candles produce signals
   - That's ~0.3 signals per 60-day window!
   - Optimizer can't find statistically significant patterns

3. **Walk-Forward Overfitting:**
   - Optimizer finds config that works in-sample
   - Config fails out-of-sample
   - This is detected by `_check_overfit()` but configs still degrade

**Config Selection Investigation:**
```python
# optimizer.py lines 408-532
def _compute_optimizer_score(...):
    # E[R] is primary score driver
    # Trade count is a PENALTY for low counts
    # Consistency factor, drawdown penalty, win rate factor
```

**Questions:**
1. Is the scoring function selecting for noise rather than edge?
2. Is minimum 5 trades (`hard_min_trades`) too low for significance?
3. Is the 50% overfit ratio threshold (`min_overfit_ratio`) too permissive?

---

### Phase 5: Trade Management Discrepancies

**File:** `core/trade_manager.py`

**TP/SL Calculation (ssl_flow.py lines 730-774):**
```python
# LONG TP: PBEMA cloud bottom
tp = pb_bot  # EMA200 of close

# LONG SL: Below swing low OR below baseline
swing_low = float(_low_arr[start:abs_index].min())
sl_swing = swing_low * 0.998
sl_baseline = baseline * 0.998
sl = min(sl_swing, sl_baseline)
```

**Questions:**
1. Does manual trader use the SAME TP target (PBEMA bot)?
2. Does manual trader place SL at swing low or baseline?
3. Is the 0.2% buffer (0.998) appropriate for crypto volatility?

**Partial TP Mechanics (from config):**
```python
# Two-tranche partial TP:
# 1. At 40% progress, take 33% of position
# 2. At 70% progress, take 50% of remaining
```

**Questions:**
1. Does manual trader use partial TP or let full position ride?
2. Could partial TP be locking in profits too early?
3. Could partial TP be preventing bigger winners?

---

### Phase 6: The "Why Filters Make It Worse" Paradox

This is the MOST IMPORTANT mystery to solve.

**Documented Failed Experiments (from CLAUDE.md):**

| Filter Change | Expected Result | Actual Result |
|--------------|-----------------|---------------|
| skip_wick_rejection=True | More signals | No change |
| min_pbema_distance=0.002 | More signals | No change |
| use_scoring=True | Flexible filtering | Same as AND logic |
| skip_body_position=True | More signals | -$17 worse |
| hard_min_trades=3 | More configs validated | -$108 worse |

**Why Does This Happen?**

**Hypothesis 1: Optimizer Overfitting**
- More signals = more noise
- Optimizer finds pattern in noise (overfits)
- Pattern doesn't exist in OOS data

**Hypothesis 2: Config Grid Interaction**
- Relaxed filter changes which configs are "valid"
- Optimizer selects different config from grid
- New config has different (worse) characteristics

**Hypothesis 3: Signal Quality vs Quantity Tradeoff**
- Tight filters = few but high-quality signals
- Relaxed filters = many but low-quality signals
- Optimizer can't distinguish quality from quantity

**Investigation Steps:**
1. Run same backtest with IDENTICAL config but different filter settings
2. Compare trade-by-trade: which trades are added/removed?
3. Are the added trades winners or losers?

---

### Phase 7: Specific Code Audit Points

**Location 1: ADX Window Look-Ahead Fix (ssl_flow.py line 388)**
```python
# FIX: Look-ahead bias - exclude current bar
adx_window = df["adx"].iloc[regime_start:abs_index]  # Correct: excludes current bar
```
- This was recently fixed (documented in CLAUDE.md v1.10.1)
- Verify the fix is actually in place
- Check if similar issues exist elsewhere

**Location 2: AlphaTrend Equality Handling (indicators.py lines 156-183)**
```python
# When AlphaTrend == AlphaTrend[2], check previous comparison
equal_mask = np.isclose(df['at_buyers'].values, df['at_sellers'].values, rtol=1e-9, atol=1e-12)
if equal_mask.any():
    prev_comparison = df['alphatrend'].shift(1) > df['alphatrend'].shift(3)
```
- Does TradingView handle equality the same way?
- Could this cause dominance flip at wrong moment?

**Location 3: Baseline Touch Window (ssl_flow.py line 184)**
```python
lookback_candles: int = 5  # Check last 5 candles for baseline touch
```
- Is 5 candles the right lookback?
- Manual trader might look back further or shorter

**Location 4: PBEMA-SSL Overlap Threshold (ssl_flow.py line 531)**
```python
OVERLAP_THRESHOLD = 0.005  # 0.5%
baseline_pbema_distance = abs(baseline - pbema_mid) / pbema_mid
is_overlapping = baseline_pbema_distance < OVERLAP_THRESHOLD
```
- Is 0.5% the right threshold?
- Manual trader probably doesn't measure this precisely

---

## Required Output Format

### Section 1: Executive Summary
- **Root Cause #1:** [Most likely cause with evidence]
- **Root Cause #2:** [Second most likely cause]
- **Root Cause #3:** [Third possibility]
- Confidence level (High/Medium/Low) for each

### Section 2: TradingView Parity Report
For each indicator:
```
INDICATOR: [Name]
TradingView Value at [timestamp]: [value]
Python Value at [timestamp]: [value]
MATCH: Yes/No
IMPACT: [How mismatch affects signals]
```

### Section 3: Signal Discrepancy Analysis
For 5 specific trades:
```
TRADE #[N]: [timestamp]
Manual Trader Action: [Would take / Would skip]
Bot Action: [Took / Skipped]
DISCREPANCY: [Why they differ]
Filter That Blocked/Allowed: [which filter]
```

### Section 4: Detailed Findings
```
FINDING #[N]: [Title]
Severity: Critical/High/Medium/Low
Location: [file:line_number]
Description: [What the issue is]
Evidence: [Code snippet or test result]
Impact: [How this affects manual-vs-bot gap]
Recommended Fix: [Specific code change]
```

### Section 5: Optimizer Analysis
- Is the optimizer finding ANY good configs?
- What configs is it selecting vs rejecting?
- Is the scoring function appropriate?

### Section 6: Prioritized Fix Order
1. [Fix with highest impact]
2. [Second priority fix]
3. [Third priority fix]
...

---

## Meta-Instructions for Analysis

1. **Compare Against Manual Trading:**
   - The manual trader is PROFITABLE
   - Every discrepancy should be framed as: "What does manual trader see vs what does code see?"

2. **Test Indicator Parity First:**
   - If indicators don't match TradingView, nothing else matters
   - This is the MOST IMPORTANT first step

3. **Trace Specific Trades:**
   - Don't analyze abstractly
   - Pick real timestamps and trace through the code
   - Use debug mode: `check_ssl_flow_signal(df, index, return_debug=True)`

4. **Question Every Filter:**
   - Each AND filter reduces signals
   - Manual trader probably doesn't apply all these filters mentally
   - Which filters are causing the gap?

5. **Don't Recommend Already-Failed Experiments:**
   - See "Failed Experiments" table in CLAUDE.md
   - These have been tested and MADE THINGS WORSE
   - Focus on NEW hypotheses

---

## Files to Analyze

| File | Focus |
|------|-------|
| `core/indicators.py` | TradingView parity |
| `strategies/ssl_flow.py` | Signal logic translation |
| `core/optimizer.py` | Config selection |
| `runners/rolling_wf.py` | Testing methodology |
| `core/trade_manager.py` | Trade execution |
| `core/config.py` | Default settings |

---

## The Ultimate Question

**If a manual trader looks at a chart and sees a valid SSL Flow setup, will the Python code generate a signal at the SAME moment with the SAME entry/TP/SL levels?**

If the answer is NO, find out WHY and fix it.

If the answer is YES, then the problem is in:
- Optimizer config selection
- Walk-forward methodology
- Trade management (TP/SL/Partials)

This investigation should answer this question definitively with evidence.
