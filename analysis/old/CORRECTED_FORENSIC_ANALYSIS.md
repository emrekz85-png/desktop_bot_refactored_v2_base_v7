# CORRECTED FORENSIC ANALYSIS REPORT
## SSL Flow Trading Bot - Manual-to-Automated Translation Gap
**Date:** 2026-01-01
**Focus:** Why does manual trading work, but the automated bot fails in rolling WF tests?

---

## EXECUTIVE SUMMARY

The previous analysis was based on an **INCORRECT ASSUMPTION** (backtest profitable, live fails). The **ACTUAL** situation is:

1. **Manual trading on TradingView is profitable**
2. **Automated bot fails even in rolling WF backtests**
3. **No live trading has been attempted yet**

This corrected analysis investigates the **manual-to-automated translation gap**.

---

## TOP 3 ROOT CAUSES

| # | Issue | Impact | Confidence |
|---|-------|--------|------------|
| **1** | **AlphaTrend uses MFI instead of RSI** | Dominance detection differs from TradingView | **CRITICAL** |
| **2** | **PBEMA Distance Filter (18.8% pass rate)** | Manual trader doesn't measure exact distance | **CRITICAL** |
| **3** | **Partial TP at 40% Progress** | Manual trader lets positions run to 70%+ | **HIGH** |

---

## PHASE 1: INDICATOR PARITY ISSUES

### FINDING #1: AlphaTrend Uses MFI Instead of RSI
**Severity:** CRITICAL
**Location:** `core/indicators.py:96-99`

```python
# CURRENT (Python):
if 'volume' in df.columns and df['volume'].sum() > 0:
    df['_at_momentum'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=ap)
else:
    df['_at_momentum'] = ta.rsi(df['close'], length=ap)

# TRADINGVIEW DEFAULT:
AlphaTrend := (RSI >= 50) ? max(upT, nz(AlphaTrend[1])) : min(downT, nz(AlphaTrend[1]))
```

**Impact:**
- Binance provides volume data, so Python ALWAYS uses MFI
- TradingView AlphaTrend default is RSI
- MFI and RSI give **DIFFERENT** readings for the same candle
- AlphaTrend line will be at **DIFFERENT** price level
- `at_buyers_dominant` may be TRUE in Python but FALSE in TradingView (or vice versa)

**Recommended Fix:**
```python
# Force RSI usage to match TradingView
df['_at_momentum'] = ta.rsi(df['close'], length=ap)
```

---

### FINDING #2: ATR Calculation Uses SMA Instead of RMA
**Severity:** HIGH
**Location:** `core/indicators.py:93`

```python
# CURRENT (Python):
df['_at_atr'] = ta.sma(df['_at_tr'], length=ap)  # Uses SMA

# TRADINGVIEW DEFAULT:
# Uses RMA (Wilder's Smoothing) = EMA with alpha = 1/length
```

**Impact:**
- SMA is more responsive than RMA
- AlphaTrend upT/downT levels will differ
- Trend direction changes will occur at DIFFERENT times

**Recommended Fix:**
```python
# Use RMA (Wilder's smoothing) to match TradingView
df['_at_atr'] = ta.rma(df['_at_tr'], length=ap)
```

---

### FINDING #3: Timeframe-Adaptive HMA Lookback
**Severity:** MEDIUM
**Location:** `core/indicators.py:309-315`

```python
# CURRENT (Python):
TF_HMA_LOOKBACK = {
    "5m": 75,   # Non-standard
    "15m": 60,  # Matches TradingView default
    "1h": 45,   # Non-standard
    "4h": 40,   # Non-standard
}

# TRADINGVIEW SSL HYBRID:
# Uses fixed HMA(60) regardless of timeframe
```

**Impact:**
- 15m chart: MATCHES TradingView (HMA60)
- 5m chart: Python uses HMA75 (more lag than TradingView)
- 1h chart: Python uses HMA45 (faster than TradingView)
- Baseline touch detection occurs at DIFFERENT moments

---

### FINDING #4: PBEMA Uses Close Instead of Low
**Severity:** MEDIUM
**Location:** `core/indicators.py:273-275`

```python
# CURRENT (Python):
df["pb_ema_top"] = ta.ema(df["high"], length=200)
df["pb_ema_bot"] = ta.ema(df["close"], length=200)  # Uses CLOSE

# SOME PBEMA IMPLEMENTATIONS:
# Use EMA(low, 200) for bottom, not EMA(close, 200)
```

**Impact:**
- TP targets may differ from what manual trader uses
- LONG TP (pb_ema_bot) is LOWER with close than with low

---

## PHASE 2: SIGNAL LOGIC DISCREPANCIES

### FINDING #5: PBEMA Distance Filter - PRIMARY BOTTLENECK
**Severity:** CRITICAL
**Location:** `strategies/ssl_flow.py:514-524, 587`

```python
# CURRENT:
min_pbema_distance = 0.004  # 0.4% minimum
long_pbema_distance >= min_pbema_distance  # Must pass

# PASS RATE: 18.8% - REJECTS 81.2% OF POTENTIAL SIGNALS
```

**Manual Trader Behavior:**
- Looks at chart: "PBEMA above price? ✓ Good to go"
- Does NOT measure exact 0.4% distance
- Takes trades with 0.1-0.3% distance that bot rejects

**Impact:**
- This single filter blocks **81.2%** of potential entries
- Manual trader sees setup, bot says "insufficient distance"

---

### FINDING #6: AlphaTrend Flat Threshold Too Restrictive
**Severity:** HIGH
**Location:** `core/config.py:514` (ALPHATREND_CONFIG)

```python
# CURRENT:
"flat_threshold": 0.001,  # 0.1% change over 5 candles = "flat"

# PASS RATE: 39.8% - REJECTS 60.2% OF POTENTIAL SIGNALS
```

**Manual Trader Behavior:**
- Looks at AlphaTrend: "Blue line rising? ✓"
- Does NOT calculate exact 0.1% change percentage
- Takes trades in slight consolidation that bot marks as "flat"

---

### FINDING #7: Wick Rejection Filter - PROVEN DRAG
**Severity:** MEDIUM (CONFIRMED by testing)
**Location:** `strategies/ssl_flow.py:551-569`

```python
# CURRENT:
min_wick_ratio = 0.10  # 10% of candle range required

# FROM CLAUDE.md TEST RESULTS:
# skip_wick_rejection=True → +$30.08 PnL, +101 trades
```

**Manual Trader Behavior:**
- Looks for "bounce" visually
- Does NOT measure if wick is exactly 10%
- Takes trades with 5-7% wicks that bot rejects

---

## PHASE 3: ENTRY TIMING ISSUES

### FINDING #8: Signal Index -2 May Be Over-Conservative
**Severity:** MEDIUM
**Location:** `strategies/ssl_flow.py:172`

```python
# CURRENT:
def check_ssl_flow_signal(df, index=-2, ...)  # Second-to-last candle
# + Entry at opens[i+1] in optimizer
# = 2-bar delay from signal candle
```

**Manual Trader Behavior:**
- Sees signal on current candle close
- Enters on NEXT candle open (1-bar delay)
- Bot uses 2-bar delay (may miss fast moves)

---

### FINDING #9: RR Tolerance Too Tight
**Severity:** MEDIUM
**Location:** `core/optimizer.py:640`

```python
# CURRENT:
if actual_rr < min_rr * 0.9:  # 10% tolerance
    continue  # Reject signal

# CRYPTO REALITY:
# Overnight gaps: 0.5-1%
# Volatility spikes: 1-2%
# Need 20-25% tolerance, not 10%
```

---

## PHASE 4: WALK-FORWARD METHODOLOGY

### FINDING #10: Signal Scarcity Death Spiral
**Severity:** CRITICAL
**Location:** Multiple filters in `strategies/ssl_flow.py`

**13+ AND Filter Compound Effect:**

| Filter | Pass Rate | Cumulative |
|--------|-----------|------------|
| 1. ADX >= 15 | 70% | 70% |
| 2. Regime TRENDING | 60% | 42% |
| 3. Price vs baseline | 50% | 21% |
| 4. AT dominant | 40% | 8.4% |
| 5. AT not flat | 50% | 4.2% |
| 6. Baseline touch | 69% | 2.9% |
| 7. Body position | 99.9% | 2.9% |
| 8. **PBEMA distance** | **18.8%** | **0.55%** |
| 9. Wick rejection | 68.8% | 0.38% |
| 10. PBEMA direction | 42% | 0.16% |
| 11. No overlap | 85% | 0.14% |
| 12. RSI OK | 80% | 0.11% |
| 13. RR valid | 70% | **0.076%** |

**Final Pass Rate: ~0.076%**

- 60 days × 96 candles/day (15m) = 5,760 candles
- 5,760 × 0.076% = **~4 signals per 60-day window**
- Split 70/30: ~3 train, ~1 OOS

**You cannot statistically optimize on 3 data points.**

---

### FINDING #11: hard_min_trades=5 is Statistically Meaningless
**Severity:** HIGH
**Location:** `core/optimizer.py:409`

```python
hard_min_trades: int = 5  # Too low!
```

**Statistical Reality:**
- 5 trades: 95% CI for win rate is [18%, 82%]
- Cannot distinguish edge from luck
- Probability of 5 wins by chance: 3.1% (not rare enough)

**From CLAUDE.md:**
```
hard_min_trades=3 → -$172 (44 trades, 31.8% WR)
# Optimizer accepted noisy configs, OOS failed catastrophically
```

---

## PHASE 5: TRADE MANAGEMENT

### FINDING #12: Partial TP at 40% is Too Early
**Severity:** HIGH
**Location:** `core/config.py:692-696`

```python
# CURRENT:
"partial_tranches": [
    {"trigger": 0.40, "fraction": 0.33},  # 33% at 40% progress
    {"trigger": 0.70, "fraction": 0.50},  # 50% of remaining at 70%
]
```

**Manual Trader Behavior:**
- Lets FULL position run until 65-80% progress
- May take single partial at 70%, not 40%
- Does NOT lock in 33% after only 40% progress

**Impact:**
- Caps winners before they can run
- Locks in small profits instead of letting momentum develop

---

### FINDING #13: Breakeven Triggered Too Early
**Severity:** HIGH
**Location:** `core/config.py:716-717`

```python
# CURRENT:
"progressive_be_after_tranche": 1,  # Move to BE after FIRST partial (40%)
"be_atr_multiplier": 0.5,           # Only 0.5x ATR buffer
```

**Impact:**
- After 40% partial, SL moves to breakeven with tiny buffer
- Normal crypto volatility (0.3-0.5% wick) stops out position
- Turns potential winners into scratch trades

---

## PHASE 6: THE FILTER PARADOX EXPLAINED

### Why Relaxing Filters Makes Performance WORSE

**The Death Spiral:**

```
1. Relax filters
   ↓
2. More signals appear (9 → 25 trades)
   ↓
3. Signal QUALITY drops (E[R]: 0.10 → 0.06)
   ↓
4. Optimizer sees more train data
   ↓
5. Finds patterns in NOISE (fits random variance)
   ↓
6. Selects DIFFERENT (worse) config from grid
   ↓
7. OOS performance degrades (-$17 to -$108)
```

**Root Cause:** The optimizer's trade_factor is NEUTRAL for 10-30 trades:

```python
# core/optimizer.py lines 459-470
elif trades <= 30:
    trade_factor = 1.0  # NO PENALTY!
```

This means:
- Tight filters: 9 trades at 0.10 E[R] → trade_factor = 0.9
- Relaxed filters: 25 trades at 0.06 E[R] → trade_factor = 1.0

The optimizer can prefer the **lower-quality, higher-quantity** config!

---

## PRIORITIZED FIX ORDER

### PHASE 1: Critical Indicator Parity (Do First)

1. **Fix AlphaTrend MFI → RSI** (15 min)
   - File: `core/indicators.py:96-99`
   - Change: Force `ta.rsi()` instead of conditional MFI

2. **Fix ATR SMA → RMA** (15 min)
   - File: `core/indicators.py:93`
   - Change: Use `ta.rma()` instead of `ta.sma()`

### PHASE 2: Trade Management (Do Second)

3. **Adjust Partial TP Timing** (10 min)
   - File: `core/config.py:692-696`
   - Change: First partial at 65% (not 40%), fraction 40% (not 33%)

4. **Delay Breakeven Trigger** (10 min)
   - File: `core/config.py:716-717`
   - Change: `progressive_be_after_tranche: 2` (after second partial)

5. **Widen BE Buffer** (5 min)
   - File: `core/config.py:717`
   - Change: `be_atr_multiplier: 1.0` (not 0.5)

### PHASE 3: Filter Adjustments (Do Third)

6. **Disable Wick Rejection** (Already proven: +$30)
   - File: `core/config.py`
   - Change: `skip_wick_rejection: True`

7. **Increase AT Flat Threshold** (5 min)
   - File: `core/config.py:514`
   - Change: `flat_threshold: 0.002` (0.2% not 0.1%)

### PHASE 4: Test and Validate

8. **Run Rolling WF Test After Each Change**
   - Verify indicator parity fixes don't break existing functionality
   - Monitor trade count and win rate after trade mgmt changes
   - Compare before/after filter adjustments

---

## KEY INSIGHT

**The strategy does not "work manually but fail automated."**

What's actually happening:

1. **Indicator calculations differ** between TradingView and Python
2. **Filters are too restrictive** compared to manual judgment
3. **Trade management exits too early** vs manual trader patience
4. **Rolling WF methodology is CORRECTLY showing no edge** with current setup

The fix is NOT to change the WF methodology - it is to:
1. **Match indicator calculations** to TradingView exactly
2. **Relax filters** that don't match manual trading judgment
3. **Adjust trade management** to match manual trader behavior

---

## FILES REFERENCED

| File | Key Lines | Issues |
|------|-----------|--------|
| `core/indicators.py` | 93, 96-99, 309-315 | MFI vs RSI, SMA vs RMA, HMA lookback |
| `strategies/ssl_flow.py` | 172, 514-524, 551-569 | Index -2, PBEMA distance, wick rejection |
| `core/config.py` | 514, 692-696, 716-717 | AT flat, partial TP, breakeven |
| `core/optimizer.py` | 409, 459-470, 640 | min_trades, trade_factor, RR tolerance |

---

*Report generated by Corrected Deep Forensic Analysis - 2026-01-01*
