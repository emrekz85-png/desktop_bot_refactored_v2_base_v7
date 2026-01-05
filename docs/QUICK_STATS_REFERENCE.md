# SSL Flow - Quick Stats Reference Card

**Data:** BTCUSDT 15m, 1 Year (2025-01-08 to 2025-12-01)

---

## Bottom Line

| Metric | Value |
|--------|-------|
| **Total PnL** | $72.99 |
| **Return** | +7.30% |
| **Win Rate** | 46.2% (12/26) |
| **Avg R** | 0.351R |
| **Profitability** | PROFITABLE |

---

## Overall Performance

```
Trades:         26
  Winners:      12 (46.2%)
  Losers:       14 (53.8%)

PnL:            $72.99
  Avg per trade: $2.81
  Median:        -$10.89

R-Multiple:     0.351R avg
  Winners:      1.956R avg
  Losers:       -1.024R avg
  Asymmetry:    1.91x (wins > losses)

Hold Time:      97 bars avg
  Winners:      116.6 bars
  Losers:       80.4 bars
```

---

## LONG vs SHORT

### LONG (n=20)
```
Win Rate:       40.0% (8/20)
PnL:            $8.28
Avg R:          0.118R
Hold Time:      104 bars avg
```

**Problem:** 8/20 trades (40%) are quick SL failures (≤20 bars)

### SHORT (n=6)
```
Win Rate:       66.7% (4/6)
PnL:            $64.70
Avg R:          1.128R
Hold Time:      73 bars avg
```

**Advantage:** 0/6 trades are quick failures

### Comparison

| Metric | LONG | SHORT | Winner |
|--------|------|-------|--------|
| Win Rate | 40.0% | 66.7% | SHORT +26.7% |
| Total PnL | $8.28 | $64.70 | SHORT 7.8x |
| Avg R | 0.118R | 1.128R | SHORT 9.6x |

---

## Winner Characteristics

```
Count:          12 (46.2%)
Avg PnL:        $19.60
Avg R:          1.956R (nearly 2x risk)
Median R:       1.503R
Hold Time:      116.6 bars avg (36 bars longer than losers)
Exit Type:      100% TP
Best Win:       4.611R ($46.67, SHORT)
```

---

## Loser Characteristics

```
Count:          14 (53.8%)
Avg PnL:        -$11.59
Avg R:          -1.024R (tight SL adherence)
Median R:       -1.026R
Hold Time:      80.4 bars avg
Exit Type:      100% SL
Worst Loss:     -1.032R (-$12.16, LONG)

Quick Failures: 8/14 (57.1% hit SL within 20 bars)
  - ALL 8 are LONG trades
  - 0 are SHORT trades
```

---

## R-Multiple Distribution

```
Huge Loss (<-2R):       0    (0.0%)
Loss (-2R to -0.5R):   14   (53.8%)  ← All cluster at -1.0R
Small Loss (-0.5 to 0): 0    (0.0%)
Small Win (0 to 0.5):   0    (0.0%)
Win (0.5R to 2R):       8   (30.8%)
Huge Win (2R+):         4   (15.4%)
```

**Binary outcomes:** Trades either hit SL (~-1R) or TP (~1.5-2R)

---

## Hold Time Distribution

```
0-10 bars:      1   (3.8%)
11-20 bars:     9  (34.6%)  ← Most common
21-50 bars:     3  (11.5%)
51-100 bars:    6  (23.1%)
101-200 bars:   3  (11.5%)
201-500 bars:   3  (11.5%)
500+ bars:      1   (3.8%)
```

**46% of trades** resolve within 20 bars

---

## Exit Timing

### TP Exits (n=12)
```
Avg hold time:  116.6 bars
All are winners (100%)
```

### SL Exits (n=14)
```
Avg hold time:  80.4 bars
All are losers (100%)

Timing breakdown:
  Quick (0-20 bars):   8 (57.1%)  ← Problem area
  Medium (21-100):     3 (21.4%)
  Late (100+):         3 (21.4%)
```

---

## Top 5 Wins (by R-multiple)

| Rank | Date | Dir | R-Multiple | PnL | Bars |
|------|------|-----|------------|-----|------|
| 1 | 2025-03-02 | SHORT | 4.611R | $46.67 | 81 |
| 2 | 2025-11-04 | LONG | 3.132R | $32.74 | 490 |
| 3 | 2025-01-27 | LONG | 2.600R | $24.95 | 126 |
| 4 | 2025-02-03 | LONG | 2.028R | $20.11 | 42 |
| 5 | 2025-02-28 | LONG | 1.619R | $15.93 | 13 |

---

## Critical Findings

1. **Quick LONG Failures:** 8/20 LONG trades (40%) fail within 20 bars
2. **SHORT Outperformance:** 66.7% WR vs 40.0% for LONG
3. **Positive Asymmetry:** 1.91x (enables profitability at 46% WR)
4. **Tight Risk Control:** All losses at -1.0R (max -1.032R)
5. **Binary Outcomes:** No trades between -0.5R and 0.5R

---

## Recommendations

### Fix LONG Quick Failures
- 40% of LONG trades fail quickly
- Add stricter entry filters for LONG
- Consider: momentum, volume, or trend strength filters

### Increase SHORT Frequency
- Only 6 SHORT trades in 1 year
- May be missing profitable setups
- Consider relaxing SHORT entry criteria

### Maintain Risk Control
- Current SL placement is excellent
- Do NOT widen stop losses
- Focus on improving entry quality

---

## Files Generated

All analysis files located at:
`/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/`

1. **analyze_ssl_trades.py** - Comprehensive analysis
2. **verify_calculations.py** - Calculation verification
3. **analyze_directional_bias.py** - LONG vs SHORT deep dive
4. **generate_trade_summary_table.py** - Table + CSV export
5. **SSL_FLOW_TRADE_ANALYSIS_SUMMARY.md** - Full report
6. **ssl_flow_trades_summary.csv** - Data for Excel/Sheets

---

## Verified Calculations

All calculations verified:
- Total trades: 26 = 20 LONG + 6 SHORT ✓
- Winners + Losers: 12 + 14 = 26 ✓
- PnL from balance: $72.99 ✓
- PnL from sum: $72.99 ✓
- LONG + SHORT PnL: $8.28 + $64.70 = $72.99 ✓
- All TP = Winners: 12/12 ✓
- All SL = Losers: 14/14 ✓

**No calculation errors detected.**
