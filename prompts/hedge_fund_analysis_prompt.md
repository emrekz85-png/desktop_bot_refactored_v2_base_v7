# Comprehensive Expert Analysis Prompt for SSL Flow Strategy

## Instructions

Copy the entire prompt below (within the code fence) and paste it into your preferred LLM (Claude, GPT-4, etc.) for a comprehensive expert analysis of your SSL Flow trading strategy.

---

## The Prompt

```
You are a senior quantitative portfolio manager at a multi-strategy hedge fund with 15+ years of experience in systematic trading, including 8+ years specifically in cryptocurrency derivatives markets. You have designed, validated, and deployed algorithmic trading systems managing $500M+ in AUM. Your expertise spans statistical validation, market microstructure, risk management, and the unique characteristics of 24/7 crypto futures markets.

You have been engaged as an independent consultant to conduct a rigorous due diligence review of a cryptocurrency futures trading strategy called "SSL Flow" before potential capital allocation. The strategy owner seeks your candid, unbiased assessment—not validation of their existing approach.

---

## STRATEGY SPECIFICATION

### Core Philosophy
The SSL Flow strategy is a trend-following system based on the concept: "There is a path from SSL HYBRID to PBEMA cloud." It attempts to capture moves from a dynamic support/resistance level (SSL Baseline) to a mean-reversion target (PBEMA Cloud).

### Indicator Stack

| Indicator | Calculation | Purpose |
|-----------|-------------|---------|
| SSL Baseline | HMA(close, 60) | Trend direction, dynamic S/R |
| AlphaTrend | ATR(14) + MFI/RSI weighted | Buyer/seller dominance filter |
| PBEMA Cloud | EMA(high, 200) & EMA(close, 200) | Take profit target zone |
| RSI(14) | Standard RSI | Overbought/oversold filter |
| ADX(14) | Average Directional Index | Trend strength gate (min 15) |
| Keltner Channels | Baseline +/- EMA(TR) * 0.2 | Volatility bands |

### Entry Logic (LONG)
All conditions must be TRUE simultaneously:
1. Price > SSL Baseline (HMA60)
2. AlphaTrend indicates BUYERS dominant (blue line above, line rising)
3. Baseline retest occurred within last 5 candles
4. Candle body positioned above baseline
5. PBEMA Cloud above current price (profit target exists)
6. Lower wick rejection present (bounce confirmation)
7. PBEMA Cloud above baseline (target is reachable - "path exists")
8. RSI <= rsi_limit (not overbought)
9. ADX >= 15 (sufficient trend strength)
10. No SSL-PBEMA overlap (distance > 0.5%)

SHORT entry mirrors this logic inversely.

### Exit Logic
- **Take Profit (LONG):** PBEMA Cloud lower boundary
- **Take Profit (SHORT):** PBEMA Cloud upper boundary
- **Stop Loss (LONG):** min(swing_low_20 * 0.998, baseline * 0.998)
- **Stop Loss (SHORT):** max(swing_high_20 * 1.002, baseline * 1.002)

### Position Sizing
- Risk per trade: 1.75% of account
- Leverage: 10x
- R-Multiple based sizing with confidence multipliers

---

## BACKTEST RESULTS

### Primary Test Period (June 1 - December 1, 2025)
**Rolling Walk-Forward Optimization: 30-day train, 7-day forward, weekly re-optimization**

| Metric | Value |
|--------|-------|
| Initial Balance | $2,000 |
| Final PnL | +$145.39 |
| Total Trades | 24 |
| Win Rate | 79.2% |
| Max Drawdown | ~$44 (2.2%) |
| Avg Trade Duration | Not specified |
| Profit Factor | Not specified |
| Sharpe Ratio | Not specified |

### Symbol-Level Performance (Full Year 2025)

| Symbol | PnL | Trades | Win Rate |
|--------|-----|--------|----------|
| BTCUSDT | +$43.97 | 19 | 78.9% |
| ETHUSDT | +$16.91 | 24 | 79.2% |
| LINKUSDT | +$48.27 | 15 | 86.7% |
| DOGEUSDT | $0 | 0 | N/A (no valid configs) |
| SOLUSDT | Loss | - | Some periods 0% win rate |
| XRPUSDT | Loss | - | High trade count, negative expectancy |
| 5 others | Loss or $0 | - | - |

### Regime Analysis
| Period | PnL | Trades | Win Rate |
|--------|-----|--------|----------|
| H1 (Jan-June 2025) | -$5.05 | 24 | 87.5% |
| H2 (June-Dec 2025) | +$157.10 | 25 | 84.0% |
| Full Year | +$109.15 | 58 | 81.0% |

Note: H1+H2 ≠ Full Year due to 30-day lookback gap in June.

---

## DOCUMENTED FAILED EXPERIMENTS

The strategy owner has tested and rejected the following modifications. Each produced worse results than baseline:

| Experiment | Change | Result | Failure Reason |
|------------|--------|--------|----------------|
| skip_wick_rejection | Remove wick filter | $145.39 (no change) | Not a bottleneck (68.8% pass rate) |
| trailing_after_partial | Trailing SL after partial TP | $145.39 (no change) | Optimizer selects same configs |
| min_pbema_distance=0.002 | Reduce from 0.004 | $145.39 (no change) | Low quality signals rejected by optimizer |
| regime_adx_threshold=25 | Increase from 20 | $10.95 | Too restrictive, minimal trades |
| risk_per_trade=2.0% | Increase from 1.75% | $78.75 | Optimizer selects different (worse) configs |
| risk_per_trade=2.5% | Increase from 1.75% | $86.75 | Same issue as above |
| avoid_hours=[6,14,21,22] | Time-based filter | $127.71 | Blocked profitable trades (-$29.39 impact) |
| use_trend_filter | SMA + HH/LL filter | $87.06 | Too restrictive (25→9 trades) |
| use_sweep_detection | SL near swing check | $0.00 | Blocked ALL trades |
| use_smart_reentry | Re-enter after SL | $145.39 (no change) | Never triggered (constraints too tight) |
| use_roc_filter | ROC momentum filter | $59.18 | Breaks optimizer behavior |

### Filter Pass Rate Analysis (BTCUSDT-15m, 10,000 bars)

| Filter | Pass Rate | Classification |
|--------|-----------|----------------|
| pbema_distance | 18.8% | BOTTLENECK |
| at_not_flat | 39.8% | RESTRICTIVE |
| pbema_above_baseline | 42.1% | RESTRICTIVE |
| wick_rejection | 68.8% | LOOSE |
| baseline_touch | 69.3% | LOOSE |
| body_position | 99.9% | LOOSE |

---

## INFRASTRUCTURE DETAILS

### Walk-Forward Optimization
- Training window: 30 days
- Forward test window: 7 days
- Re-optimization frequency: Weekly
- Optimized parameters: RR ratio, RSI threshold
- Determinism: Fixed via random.seed(42), np.random.seed(42)
- Config selection: Sorted by hash for tie-breaking

### Risk Management
- **Stream-level circuit breaker:** Max loss (-$200), drawdown ($100), consecutive full stops (2)
- **Global circuit breaker:** Daily (-$400), weekly (-$800), max drawdown (20%)
- **Partial TP profiles:** "Clip" (early partial 45%, larger size 50%) vs "Runner" (late partial 70%, smaller size 33%)

### Known Bug Fixes Applied
1. **Look-ahead bias (CRITICAL):** ADX regime calculation included current bar; fixed to exclude
2. **Optimizer determinism:** Results varied by $191 due to concurrent execution ordering; fixed with seeding
3. **PBEMA-SSL overlap:** Added 0.5% threshold check to prevent trades when indicators converge

---

## YOUR ANALYSIS MANDATE

Provide a comprehensive, structured report addressing the following areas. For each section, clearly state your confidence level (High/Medium/Low) and the basis for your assessment.

### SECTION 1: Strategy Logic Integrity

1.1. **Mathematical Coherence**
- Analyze the logical flow from entry conditions through exit. Are there circular dependencies or contradictions?
- The strategy requires price above baseline AND baseline retest in last 5 candles. Is this logically consistent or does it create a narrow, unstable entry window?
- Evaluate whether HMA(60) as a baseline creates appropriate responsiveness for 15m/1h timeframes.

1.2. **Indicator Redundancy Analysis**
- AlphaTrend uses ATR + MFI/RSI. RSI is also used as a separate filter. ADX measures trend strength. Are these measuring the same underlying phenomenon?
- Quantify the expected correlation between these indicators.
- Which indicators, if any, could be removed without degrading signal quality?

1.3. **Target Logic Assessment**
- PBEMA Cloud (EMA200) as TP: Is this methodologically sound for a trend-following strategy?
- The "path exists" check (PBEMA above baseline for LONG) - does this create look-ahead-like behavior or is it valid at decision time?
- Stop loss at swing low OR baseline: What is the expected distribution of stop distances? Does this create inconsistent R-multiples?

1.4. **Entry Condition Probability**
- With 10 conditions required simultaneously, calculate the approximate probability of entry under random market conditions.
- 18.8% pass rate on pbema_distance × 39.8% on at_not_flat × ... results in what effective pass rate?
- Is the low trade frequency (24 trades in 6 months = 1 trade per week) a feature or a bug?

### SECTION 2: Statistical Validity Assessment

2.1. **Sample Size Analysis**
- 24 trades in 6 months: Calculate the confidence interval around the 79% win rate.
- What is the probability that the true win rate is below 50%?
- How many trades would be required to achieve 95% confidence that win rate > 55%?

2.2. **Walk-Forward Reliability**
- 30-day train / 7-day forward: Is this ratio appropriate for the observed market regime characteristics?
- Weekly re-optimization with only 24 total trades means approximately 1 trade per optimization window. Is this sufficient to validate parameter changes?
- What is the risk of the optimizer fitting to noise?

2.3. **Regime Sensitivity**
- H1: -$5.05, H2: +$157.10 represents a stark regime shift. What does this imply about strategy robustness?
- Is the strategy viable only in specific market conditions? If so, can these be identified ex-ante?
- 11 symbols tested, only 3 profitable: What does this 27% success rate across symbols suggest about edge universality?

2.4. **Overfitting Diagnostics**
- 79% win rate with trend-following logic is unusually high. Professional trend-following typically sees 35-45% win rates with higher reward-to-risk. Is this a red flag?
- The optimizer selecting "different configs" when risk is increased suggests parameter sensitivity. How should this be interpreted?
- Failed experiments uniformly produced same or worse results—is this evidence of a local maximum or evidence of overfitting to specific conditions?

### SECTION 3: Risk Management Evaluation

3.1. **Position Sizing Mathematics**
- 1.75% risk × 10x leverage = 17.5% notional position per trade. Is this appropriate for crypto volatility?
- With 3 simultaneous positions possible across BTC/ETH/LINK, max exposure = 52.5% notional. Evaluate this concentration.
- R-Multiple system with confidence multipliers: How does this interact with the fixed 1.75% base risk?

3.2. **Circuit Breaker Thresholds**
- Stream-level: -$200 max loss on $2,000 account = 10% stop. Is this too wide given the 2.2% max drawdown observed?
- 2 consecutive full stops trigger circuit breaker: What is the probability of this occurring by chance vs. genuine edge degradation?
- How were these thresholds calibrated? Are they based on statistical analysis or arbitrary?

3.3. **Tail Risk Considerations**
- Crypto flash crashes can exceed 20% in minutes. How does this strategy handle gap risk?
- No mention of maximum position hold time. What is the exposure to overnight/weekend risk?
- Leverage + futures + crypto: What is the realistic liquidation risk scenario?

3.4. **Expected Drawdown**
- $44 max drawdown on $2,000 = 2.2% seems low for a 6-month period. Is this realistic or an artifact of low trade frequency?
- What is the expected max drawdown based on the return profile?
- How does the observed Sharpe ratio compare to a bootstrap of the trade returns?

### SECTION 4: Return Analysis

4.1. **Performance Decomposition**
- $145.39 / $2,000 / 6 months = 7.25% semi-annual = ~14.5% annualized.
- How does this compare to: (a) buy-and-hold BTC, (b) buy-and-hold ETH, (c) simple trend-following benchmarks?
- Is the complexity justified by the returns?

4.2. **Trade-Level Analysis**
- 24 trades × 79% win rate = ~19 winners, ~5 losers
- $145.39 total / 24 trades = $6.06 average profit per trade
- With 1.75% risk = $35 per R, average profit = 0.17R per trade. Is this expectancy sustainable?

4.3. **Why Returns Are Limited**
- Trade frequency: 1/week limits compounding
- Asset selection: Only 3/11 symbols profitable limits diversification
- Regime dependency: H1 negative suggests ~50% of time the strategy loses
- Identify the primary constraint: Is it opportunity (not enough trades) or edge (not enough profit per trade)?

4.4. **Scaling Considerations**
- At what account size would slippage become material?
- For BTC/ETH/LINK futures on Binance, estimate market impact for $100k, $1M, $10M positions.
- Is this strategy capacity-constrained?

### SECTION 5: Missing Elements Assessment

5.1. **Professional Trading System Standards**
What elements are present in institutional-grade trading systems that appear absent here?
Consider:
- Execution algorithms (TWAP, VWAP, etc.)
- Real-time Greeks/risk monitoring
- Correlation management across positions
- Factor exposure analysis
- Regime detection models
- Drawdown-based position scaling
- Order book analysis
- Funding rate considerations

5.2. **Market Microstructure**
- No mention of bid-ask spread analysis. What is the expected slippage cost per trade?
- Binance Futures funding rates can be significant during trends. Is this accounted for?
- Are there execution timing considerations (e.g., avoiding funding settlement times)?

5.3. **Correlation Analysis**
- BTC, ETH, LINK are highly correlated. What is the marginal diversification benefit?
- Has the strategy been tested for correlation regime changes?
- Is there implicit beta exposure that could be hedged?

5.4. **Benchmark Comparison**
- Where is the performance attribution vs. simple benchmarks?
- What is the alpha after adjusting for market exposure?
- How does risk-adjusted return compare to passive crypto exposure?

### SECTION 6: Over-Engineering Analysis

6.1. **Complexity Inventory**
Count the degrees of freedom in the system:
- Entry conditions: 10
- Indicator parameters: HMA(60), EMA(200), ATR(14), RSI(14), ADX(14)
- Optimized parameters: RR ratio, RSI threshold
- Risk parameters: 1.75%, leverage, circuit breakers
- Total optimization space complexity

6.2. **Occam's Razor Test**
- Could a simpler system (e.g., HMA crossover + ATR stop) achieve similar results?
- Which layers of complexity have demonstrated measurable improvement?
- Failed experiments suggest modifications don't improve results—does this indicate the core system is already overfit?

6.3. **Filter Analysis**
- 10 simultaneous AND conditions: What is the marginal contribution of each?
- body_position at 99.9% pass rate is effectively not filtering anything—remove it?
- Is the filter cascade order-dependent? Could efficiency be improved?

### SECTION 7: Live Trading Readiness

7.1. **Backtest-to-Live Gap**
Estimate the expected performance degradation from backtest to live:
- Slippage: ___% per trade
- Funding costs: ___% annualized
- Execution delays: Impact on signal freshness
- Data quality differences: Binance API vs. historical data

7.2. **Recommended Validation Protocol**
Propose a specific pre-live validation plan:
- Paper trading duration: ___ days/trades
- Success criteria for go-live
- Phased capital deployment schedule
- Kill switch criteria

7.3. **Monitoring Framework**
What should be tracked in live trading?
- Lead indicators of edge decay
- Benchmark comparison metrics
- System health indicators
- Regime change signals

### SECTION 8: Recommendations

8.1. **Immediate Actions (Do Before Live Trading)**
Provide specific, actionable recommendations ranked by priority:
1. [Priority 1]: ...
2. [Priority 2]: ...
3. ...

8.2. **Strategy Improvements**
What modifications would you test (despite failed experiments list)?
- Why do you believe these might work when others failed?
- Proposed testing methodology for each

8.3. **Risk Adjustments**
Specific recommendations for:
- Position sizing
- Circuit breaker calibration
- Correlation management
- Drawdown protocols

8.4. **Infrastructure Improvements**
What technical or process improvements would you mandate?

8.5. **Go/No-Go Assessment**
Based on all evidence, provide your recommendation:
- [ ] Ready for live trading with current configuration
- [ ] Ready for live trading with specific modifications
- [ ] Requires additional validation before live trading
- [ ] Fundamental issues require strategy redesign
- [ ] Not viable for live trading

Justify your recommendation with the top 3 supporting factors and top 3 risk factors.

---

## OUTPUT FORMAT

Please structure your response as a formal due diligence report with:

1. **Executive Summary** (1 page maximum)
   - Overall assessment
   - Key strengths
   - Critical concerns
   - Go/No-Go recommendation with confidence level

2. **Detailed Section Analyses** (per sections 1-8 above)
   - Each section clearly numbered
   - Specific findings with evidence
   - Confidence level for each major conclusion
   - Quantitative support where possible

3. **Appendix: Mathematical Derivations**
   - Confidence interval calculations
   - Expected value calculations
   - Any statistical tests performed

4. **Risk Disclosure**
   - Limitations of this analysis
   - Assumptions made
   - Areas requiring additional information

---

## ANALYSIS PARAMETERS

- Assume you have NOT seen the actual code, only this specification
- Base your analysis on the information provided; clearly state when you are making assumptions
- Do not assume the strategy works—approach with professional skepticism
- Prioritize identification of critical flaws over minor improvements
- Consider both Type I (false positive: deploying a bad strategy) and Type II (false negative: rejecting a good strategy) errors
- Apply institutional-grade rigor, not retail trader standards

Take whatever time you need. Depth and accuracy are more valuable than speed.
```

---

## Usage Guidelines

1. **Copy the entire prompt within the code block** to your target LLM
2. **Do not modify the data tables**—they are designed to be comprehensive
3. **Allow extended response length**—this prompt requests depth (expect 3000-5000 words)
4. **For Claude:** The prompt is optimized for Claude's analytical strengths
5. **For GPT-4:** Consider adding "Think step by step" before each section
6. **Follow-up prompts** can drill into specific sections: "Expand on Section 2.3 with specific statistical tests"

---

## Key Prompt Engineering Techniques Used

| Technique | Purpose |
|-----------|---------|
| Specific role with credentials | Anchors response quality to institutional standards |
| Adversarial framing | Prevents sycophantic/validating analysis |
| Structured data tables | Enables systematic quantitative analysis |
| Confidence level requirements | Forces uncertainty acknowledgment |
| Failed experiments list | Prevents redundant recommendations |
| Output format specification | Ensures actionable, organized report |
| Type I/II error framing | Balances false positive/negative risks |

---

## Expected Analysis Outcomes

A high-quality response should include:

1. **Quantitative critique of 24-trade sample size** (likely: insufficient for high confidence)
2. **Identification of regime dependency** as major concern (H1 vs H2 divergence)
3. **Recognition of high win rate as anomalous** for trend-following
4. **Specific calculations** on confidence intervals, expected drawdown
5. **Actionable recommendations** that differ from failed experiments
6. **Likely recommendation:** "Requires additional validation before live trading"

---

## Version

- Created: 2025-12-30
- Strategy Version: v1.7.2
- Codebase: desktop_bot_refactored_v2_base_v7
