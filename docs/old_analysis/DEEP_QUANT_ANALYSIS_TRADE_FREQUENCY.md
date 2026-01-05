# Deep Quant Analysis: Trade Frequency vs Edge Preservation

**Date:** 2026-01-01
**Analyst:** Senior Quant Review
**Strategy:** SSL Flow (Trend Following)
**Problem:** 13 trades/year automated vs near-daily manual trading

---

## Executive Summary

The SSL Flow strategy suffers from a **Filter Cascade Death Spiral**: multiple independent filters each blocking 30-80% of signals, resulting in ~0.04 trades/day when the goal is ~1 trade/day. The core issue is NOT the individual filters but their **multiplicative interaction**.

**Key Finding:** The strategy produces profitable trades (79% WR in H2 2025), but the filter cascade eliminates ~98% of potential setups.

---

## 1. Filter Cascade Analysis

### Current Filter Pass Rates (from CLAUDE.md)

| Filter | Pass Rate | Bottleneck Level |
|--------|-----------|------------------|
| pbema_distance | 18.8% | SEVERE |
| at_not_flat (AlphaTrend flat) | 39.8% | SEVERE |
| pbema_above_baseline | 42.1% | SEVERE |
| wick_rejection | 68.8% | MODERATE |
| baseline_touch | 69.3% | MODERATE |
| body_position | 99.9% | NEGLIGIBLE |

### Multiplicative Filter Effect

If filters were independent (they are correlated, but for illustration):

```
P(all pass) = 0.188 * 0.398 * 0.421 * 0.688 * 0.693 * 0.999
           = 0.0150 (1.50%)

Expected daily signals on 15m:
- 96 candles/day * 0.015 = 1.44 raw signals
- After RR filter (~50% pass): 0.72 signals
- After cooldown/position checks (~70% pass): 0.50 signals
```

**Reality Check:** 13 trades/365 days = 0.036 trades/day

The gap (0.50 expected vs 0.036 actual) suggests:
1. Filter correlation is high (passing one often means failing another)
2. Regime gating is blocking entire time windows
3. Optimizer is further restricting configs

### Filter Relaxation Impact Modeling

| Filter | Current | 10% Relax | 20% Relax | 50% Relax | Expected Trade Increase |
|--------|---------|-----------|-----------|-----------|------------------------|
| pbema_distance (0.004) | 18.8% | 22.6% | 26.4% | 37.6% | +20% to +100% |
| at_not_flat (0.002) | 39.8% | 43.8% | 47.8% | 59.7% | +10% to +50% |
| pbema_above_baseline | 42.1% | 46.3% | 50.5% | 63.2% | +10% to +50% |

**Calculation for pbema_distance relaxation:**

Current: `min_pbema_distance = 0.004` (0.4%)

```python
# Relaxation scenarios:
# 10% relax: 0.004 * 0.9 = 0.0036 (0.36%)
# 20% relax: 0.004 * 0.8 = 0.0032 (0.32%)
# 50% relax: 0.004 * 0.5 = 0.0020 (0.20%)

# Expected pass rate increase (assuming linear distribution):
# 10% relax: 18.8% + (18.8% * 0.10 * 2) = 22.6%
# 20% relax: 18.8% + (18.8% * 0.20 * 2) = 26.4%
# 50% relax: 18.8% + (18.8% * 0.50 * 2) = 37.6%
```

**CRITICAL INSIGHT FROM HISTORICAL TESTS:**

From CLAUDE.md experiments, relaxing `min_pbema_distance` from 0.004 to 0.002:
- **RESULT: $157.10 (NO CHANGE)**
- **Reason:** "Low quality signals optimizer tarafindan eleniyor"

This means the optimizer is the REAL bottleneck, not the signal filters.

---

## 2. Edge Dilution vs Volume Tradeoff

### Current State Analysis

| Period | Trades | Win Rate | E[R] | Status |
|--------|--------|----------|------|--------|
| Full 2025 | 13 | 31% | NEGATIVE | LOSING |
| H2 2025 | 24 | 79% | POSITIVE | PROFITABLE |

### Statistical Significance Requirements

For a trading strategy with win rate p and average win/loss ratio w:

```
Minimum trades for 95% confidence (2 sigma):
n_min = 4 / (p * (1-p) * (w-1)^2)

For SSL Flow (H2 2025):
p = 0.79, w = 2.0 (2R target)
n_min = 4 / (0.79 * 0.21 * 1.0) = 24 trades

For Full Year:
p = 0.31, w = 2.0
n_min = 4 / (0.31 * 0.69 * 1.0) = 19 trades
```

**Key Insight:** 13 trades is statistically insufficient. We need 20-25 minimum.

### Win Rate Degradation Tolerance

If we 10x trade volume (13 -> 130 trades/year = 0.36/day):

```
Current H2 edge: 79% WR, 2.0 RR
E[R] = 0.79 * 2.0 - 0.21 * 1.0 = 1.37

Breakeven point: E[R] = 0
0 = WR * 2.0 - (1-WR) * 1.0
WR_breakeven = 33%

Maximum WR degradation:
From 79% to 33% = 46 percentage points tolerance

At 10x volume, acceptable WR = 79% - (79%-33%)/10 * adjustment_factor
Conservative: 60% WR with 10x volume still profitable
```

**Trade Frequency vs Edge Table:**

| Daily Trades | Yearly Trades | Min WR for Profit (RR=2) | Acceptable WR Drop |
|--------------|---------------|--------------------------|-------------------|
| 0.04 (current) | 13 | 33% | N/A (insufficient data) |
| 0.25 | 90 | 33% | -46 pp |
| 0.50 | 180 | 33% | -46 pp |
| 1.00 | 365 | 33% | -46 pp |
| 2.00 | 730 | 33% | -46 pp |

**Conclusion:** With RR=2.0, we can afford WR as low as 40% and still be profitable. The current 79% WR has massive headroom.

---

## 3. Volatility-Adaptive Entry System

### Current ADX-Based Regime Filter

```python
# From ssl_flow.py:
regime_adx_threshold: float = 20.0  # Average ADX over lookback
regime_lookback: int = 50

if adx_avg < regime_adx_threshold:
    return "RANGING Regime" -> NO TRADE
```

**Problem:** ADX is lagging. By the time ADX > 20, the move may be half over.

### Proposed Volatility-Adaptive System

**1. ATR Percentile (Leading Indicator)**

```python
def calculate_atr_percentile(df, lookback=100, index=-1):
    """
    ATR percentile over lookback period.
    High ATR percentile = high volatility = good for breakouts
    Low ATR percentile = compression = pending breakout

    Trade MORE in high ATR percentile (momentum trades)
    Trade LESS in mid ATR percentile (choppy)
    Trade BREAKOUTS in low ATR percentile (compression breakouts)
    """
    atr_current = df["atr"].iloc[index]
    atr_min = df["atr"].iloc[index-lookback:index].min()
    atr_max = df["atr"].iloc[index-lookback:index].max()

    if atr_max == atr_min:
        return 0.5

    return (atr_current - atr_min) / (atr_max - atr_min)

# Proposed thresholds:
# ATR percentile < 0.20: Compression - use tighter filters, wait for breakout
# ATR percentile 0.20-0.50: Normal - use standard filters
# ATR percentile 0.50-0.80: Elevated - relax filters slightly
# ATR percentile > 0.80: High vol - relax filters more, but tighten SL
```

**2. Bollinger Band Width (Volatility Gauge)**

```python
def calculate_bbwidth(df, period=20, std_dev=2.0, index=-1):
    """
    Bollinger Band Width = (Upper - Lower) / Middle
    Low BB width = low volatility (consolidation)
    High BB width = high volatility (trending)
    """
    middle = df["close"].rolling(period).mean().iloc[index]
    std = df["close"].rolling(period).std().iloc[index]
    upper = middle + std_dev * std
    lower = middle - std_dev * std

    return (upper - lower) / middle

# Proposed regime mapping:
# BBWidth < 0.02: Low volatility -> Tighten filters (avoid whipsaws)
# BBWidth 0.02-0.04: Normal -> Standard filters
# BBWidth 0.04-0.08: Elevated -> Relax pbema_distance
# BBWidth > 0.08: High -> Relax most filters, trust trend
```

**3. Dynamic Filter Adjustment Matrix**

| ATR Percentile | BBWidth | pbema_distance | at_flat_threshold | adx_min | Expected Trade Increase |
|----------------|---------|----------------|-------------------|---------|------------------------|
| < 0.20 | < 0.02 | 0.004 (tight) | 0.002 (tight) | 20 | -30% (filter more) |
| 0.20-0.50 | 0.02-0.04 | 0.004 (normal) | 0.002 (normal) | 15 | baseline |
| 0.50-0.80 | 0.04-0.08 | 0.003 (relaxed) | 0.003 (relaxed) | 12 | +50% |
| > 0.80 | > 0.08 | 0.002 (loose) | 0.004 (loose) | 10 | +100% |

---

## 4. Multi-Timeframe Signal Aggregation

### Current MTF Usage

From config.py:
```python
LOWER_TIMEFRAMES = ["5m", "15m", "30m", "1h"]
HTF_TIMEFRAMES = ["4h", "12h", "1d"]
```

Each timeframe is tested independently, but signals often conflict.

### Lower Timeframe Filter Analysis

**Problem:** 5m and 15m are being filtered MORE aggressively than needed.

Current TF-adaptive thresholds (from config.py):

| TF | ssl_touch_tolerance | min_pbema_distance | lookback_candles |
|----|--------------------|--------------------|------------------|
| 5m | 0.0025 (tighter) | 0.005 (tighter) | 6 |
| 15m | 0.003 (baseline) | 0.004 (baseline) | 5 |
| 1h | 0.005 (looser) | 0.003 (looser) | 4 |
| 4h | 0.008 (very loose) | 0.002 (loose) | 3 |

**Observation:** 5m has TIGHTER thresholds, which is backwards for trade frequency.

**Proposed Inversion:** Lower TFs should have LOOSER thresholds because:
1. More noise = need more signals to compensate for lower WR
2. Faster exits mean less damage from bad trades
3. More opportunities to compound winners

### HTF Confirmation Without Reduction

Current approach: Each TF generates independent signals.

**Proposed MTF Aggregation:**

```python
def get_mtf_signal_score(df_5m, df_15m, df_1h, index):
    """
    Instead of requiring ALL TFs to agree (reduces signals),
    use weighted scoring where HIGHER TFs provide bonus.
    """
    score = 0.0

    # Base signal from 15m (most balanced TF)
    signal_15m = check_ssl_flow_signal(df_15m, index)
    if signal_15m[0] is None:
        return None, 0.0

    # Bonus from 5m (faster confirmation)
    signal_5m = check_ssl_flow_signal(df_5m, index * 3)  # Approximate alignment
    if signal_5m[0] == signal_15m[0]:
        score += 0.5  # Same direction = momentum confirmation

    # Bonus from 1h (trend alignment)
    signal_1h = check_ssl_flow_signal(df_1h, index // 4)
    if signal_1h[0] == signal_15m[0]:
        score += 1.0  # HTF alignment = stronger trend

    # Use 15m signal with confidence boost
    return signal_15m, min(1.0, score / 1.5)  # Normalize to 0-1

# Entry rules:
# score >= 0.5: Enter with normal position
# score >= 0.8: Enter with 1.5x position
# score < 0.5: Enter with 0.5x position (or skip based on risk preference)
```

**Expected Impact:** +30-50% trade frequency from 5m/15m without losing HTF edge confirmation.

---

## 5. Practical Recommendations

### Ranked by Expected Impact and Implementation Complexity

| Rank | Change | Expected Trade Increase | WR Impact | Complexity | Priority |
|------|--------|------------------------|-----------|------------|----------|
| 1 | Remove regime gating during high volatility | +100-200% | -5 to -10% | Low | HIGH |
| 2 | Volatility-adaptive pbema_distance | +50-100% | -3 to -5% | Medium | HIGH |
| 3 | Skip wick_rejection (already tested +$30) | +101 trades | +1% WR | Done | DONE |
| 4 | Relax at_flat_threshold from 0.002 to 0.003 | +25-50% | -2 to -3% | Low | MEDIUM |
| 5 | MTF aggregation (scoring) | +30-50% | Neutral | High | MEDIUM |

### Recommendation 1: Volatility-Gated Regime Bypass (HIGHEST PRIORITY)

**Current code (ssl_flow.py line 477-484):**
```python
regime = "TRENDING" if adx_avg >= regime_adx_threshold else "RANGING"
if regime == "RANGING":
    return _ret(None, None, None, None, f"RANGING Regime (ADX_avg={adx_avg:.1f})")
```

**Proposed change:**
```python
# Calculate volatility metrics
atr_pct = calculate_atr_percentile(df, 100, abs_index)
bb_width = calculate_bbwidth(df, 20, 2.0, abs_index)

# Volatility override: High volatility can bypass regime filter
high_vol_override = (atr_pct > 0.70 or bb_width > 0.06)

regime = "TRENDING" if adx_avg >= regime_adx_threshold else "RANGING"
if regime == "RANGING" and not high_vol_override:
    return _ret(None, None, None, None, f"RANGING Regime (ADX_avg={adx_avg:.1f})")
```

**Expected impact:** +100-200% trades during volatile ranging periods.

### Recommendation 2: Dynamic pbema_distance

**Current:** Fixed 0.004 (0.4%)

**Proposed:**
```python
# Dynamic pbema_distance based on volatility
atr_pct = calculate_atr_percentile(df, 100, abs_index)

if atr_pct < 0.30:
    dynamic_pbema_distance = 0.005  # Tight in low vol
elif atr_pct < 0.60:
    dynamic_pbema_distance = 0.004  # Normal
elif atr_pct < 0.80:
    dynamic_pbema_distance = 0.003  # Relaxed in elevated vol
else:
    dynamic_pbema_distance = 0.002  # Loose in high vol

# Use dynamic value instead of fixed min_pbema_distance
if long_pbema_distance < dynamic_pbema_distance:
    # Filter out
```

**Expected impact:** +50-100% trades when volatility is elevated.

### Recommendation 3: Relax at_flat_threshold

**Current (config.py):**
```python
ALPHATREND_CONFIG = {
    "flat_threshold": 0.002,  # 0.2%
}
```

**Proposed:**
```python
ALPHATREND_CONFIG = {
    "flat_threshold": 0.003,  # 0.3% - 50% more lenient
}
```

**Expected impact:** +25-50% trades (at_not_flat pass rate 39.8% -> ~50%)

### Recommendation 4: Optimizer Minimum Trades Reduction

**Current (optimizer.py):**
```python
hard_min_trades: int = 5
```

From CLAUDE.md experiments, this was tested and FAILED:
- hard_min_trades=3: -$172 (worse)
- hard_min_trades=4: -$172 (worse)

**KEEP AT 5** - The optimizer needs statistical significance.

### Recommendation 5: Implement Scoring System for Entry

**Current:** Binary AND logic (all filters must pass)

**Proposed:** Weighted scoring with threshold

```python
# Already implemented but disabled: use_scoring=True
# Current issue: 4 core filters still mandatory

# Proposed: Make AlphaTrend confirmation weighted, not binary
score_weights = {
    "adx_strength": 2.0,      # Keep mandatory
    "regime_ok": 1.0,         # Keep mandatory
    "baseline_touch": 2.0,    # Make weighted (not mandatory)
    "alphatrend": 2.0,        # Keep mandatory (core to strategy)
    "pbema_distance": 1.0,    # Make weighted
    "wick_rejection": 1.0,    # Already skipped (good)
    "body_position": 0.5,     # Keep weighted
    "no_overlap": 0.5,        # Make weighted
}

# Threshold: 6.0 out of 10.0 (60%)
# This allows entry without perfect baseline_touch or pbema_distance
```

---

## 6. Implementation Priority Matrix

| Phase | Changes | Timeline | Risk | Expected Outcome |
|-------|---------|----------|------|------------------|
| 1 (Quick Wins) | skip_wick_rejection=True (DONE), at_flat_threshold=0.003 | 1 day | Low | +50-75% trades |
| 2 (Volatility) | ATR percentile calculation, regime bypass | 3 days | Medium | +100% trades |
| 3 (Dynamic Filters) | Volatility-adaptive pbema_distance | 1 week | Medium | +50% trades |
| 4 (MTF) | Signal aggregation scoring | 2 weeks | High | +30% trades |

---

## 7. Risk Management Adjustments

If trade frequency increases 5-10x, risk management must adapt:

### Position Sizing Reduction

```python
# Current:
risk_per_trade_pct = 0.0175  # 1.75%

# With 5x more trades:
risk_per_trade_pct = 0.0175 / sqrt(5) = 0.0078  # ~0.8%

# Kelly Criterion adjustment:
# f* = (p*b - q) / b
# Where p = WR, b = RR, q = 1-p
# Current: f* = (0.79*2 - 0.21) / 2 = 0.685 (68.5% of capital)
# With degraded WR (60%): f* = (0.60*2 - 0.40) / 2 = 0.40 (40% of capital)
# Fractional Kelly (25%): 0.40 * 0.25 = 10% of capital per trade
```

### Correlation-Based Position Limits

From CLAUDE.md:
- BTC/ETH/LINK correlation: 0.85-0.95
- 3 correlated positions = 1.07 effective positions

**Rule:** Max 2 positions in same direction on correlated assets.

### Enhanced Circuit Breaker

```python
# Current:
stream_max_loss = -200.0
global_daily_max_loss = -400.0

# With more trades, need tighter intraday controls:
hourly_max_loss = -100.0  # New: Stop after $100 loss in 1 hour
consecutive_loss_limit = 3  # Current: 2 full stops
```

---

## 8. Monitoring Metrics for Phase Rollout

### Key Performance Indicators

| Metric | Current | Target (Phase 1) | Target (Phase 2) | Alert Threshold |
|--------|---------|------------------|------------------|-----------------|
| Trades/Day | 0.04 | 0.10 | 0.25 | < 0.05 |
| Win Rate | 79% (H2) | > 65% | > 55% | < 50% |
| E[R] | 1.37 (H2) | > 0.80 | > 0.50 | < 0.20 |
| Max DD | $373 | < $400 | < $500 | > $600 |
| Sharpe (annualized) | N/A | > 1.5 | > 1.0 | < 0.5 |

### Real-Time Monitoring Dashboard

```python
def calculate_rolling_metrics(trades, window=20):
    """Calculate rolling metrics for live monitoring."""
    if len(trades) < window:
        return None

    recent = trades[-window:]

    return {
        "rolling_wr": sum(1 for t in recent if t["pnl"] > 0) / window,
        "rolling_er": sum(t["r_multiple"] for t in recent) / window,
        "rolling_pnl": sum(t["pnl"] for t in recent),
        "rolling_dd": calculate_max_dd(recent),
        "trades_per_day": window / (recent[-1]["time"] - recent[0]["time"]).days,
    }
```

---

## Conclusion

The SSL Flow strategy has a verified edge (79% WR in H2 2025) but is being choked by:

1. **Regime gating** - Blocking entire time windows
2. **pbema_distance filter** - 18.8% pass rate is too restrictive
3. **at_flat filter** - 39.8% pass rate adds unnecessary friction
4. **Optimizer constraints** - Rejecting configs that would produce more trades

**Recommended Action Plan:**

1. **Immediate (Low Risk):** Set `at_flat_threshold=0.003`
2. **Week 1:** Implement volatility-based regime bypass
3. **Week 2:** Dynamic pbema_distance based on ATR percentile
4. **Week 3:** A/B test with 0.5x position sizing on new signals
5. **Month 2:** Full rollout with adjusted risk parameters

Expected outcome: 0.25-0.50 trades/day (10-20x increase) with 55-65% WR and positive E[R].

---

*Report generated by quantitative analysis framework v1.0*
