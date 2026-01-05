# Software Requirements Specification
**Project:** Cryptocurrency Futures Trading Bot - SSL Flow Strategy
**Version:** 1.0
**Date:** January 2, 2026
**Status:** Approved

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-02 | Development Team | Initial requirements specification |

## Approval Signatures

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Product Owner | | | |
| Lead Developer | | | |
| Risk Manager | | | |
| QA Lead | | | |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Stakeholder Goals](#2-stakeholder-goals)
3. [Functional Requirements](#3-functional-requirements)
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [Data Requirements](#5-data-requirements)
6. [Interface Requirements](#6-interface-requirements)
7. [Acceptance Criteria](#7-acceptance-criteria)
8. [Specification by Example](#8-specification-by-example)
9. [Glossary](#9-glossary)
10. [Requirements Traceability](#10-requirements-traceability)

---

## 1. Introduction

### 1.1 Purpose

This document specifies the software requirements for a cryptocurrency futures trading bot implementing the SSL Flow trading strategy. The system SHALL automate trade execution based on technical analysis signals while managing risk through position sizing, stop-loss orders, and circuit breaker mechanisms.

### 1.2 Scope

**In Scope:**
- Automated signal generation using SSL Flow strategy
- Real-time trade execution on Binance Futures
- Risk management and circuit breaker protection
- Portfolio backtesting and walk-forward optimization
- Telegram notifications for trade events
- GUI (PyQt5) and headless (CLI/Colab) operation modes

**Out of Scope:**
- Manual trading interface with trader discretion override
- Support for exchanges other than Binance Futures
- Tax reporting and compliance automation
- Mobile application

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| **SSL Baseline** | Hull Moving Average (HMA) with period 60, primary trend indicator |
| **PBEMA Cloud** | Pair of EMAs (EMA200 of high and close) creating profit target zone |
| **AlphaTrend** | Proprietary indicator combining ATR and MFI/RSI for trend confirmation |
| **R-Multiple** | Ratio of trade profit/loss to initial risk (PnL / risk_amount) |
| **E[R]** | Expected R-multiple, average of all trade R-multiples |
| **Circuit Breaker** | Automated trading halt triggered by loss/drawdown thresholds |
| **Walk-Forward** | Time-series optimization validating on out-of-sample forward periods |
| **Overlap** | PBEMA cloud and SSL baseline within 0.5% distance (no clear flow) |

### 1.4 References

- SSL Flow Strategy Design Document (internal)
- Binance Futures API Documentation v1.0
- IEEE 830-1998: Recommended Practice for Software Requirements Specifications
- Hedge Fund Due Diligence Report (internal analysis)

---

## 2. Stakeholder Goals

### 2.1 Primary Stakeholder: Retail Trader

**Profile:**
- Individual trader with $2,000-$50,000 capital
- Seeks passive income through automated trading
- Limited time for active market monitoring
- Risk-averse personality preferring capital preservation

**Goals:**

| ID | Goal | Success Metric |
|----|------|----------------|
| G-RT-001 | Generate consistent monthly income | Positive E[R] >= 0.08 per trade |
| G-RT-002 | Minimize manual intervention | < 1 hour/week system monitoring |
| G-RT-003 | Protect capital from catastrophic loss | Max drawdown <= 20% |
| G-RT-004 | Understand system performance | Real-time Telegram notifications |

**Pain Points Addressed:**
- Manual trading requires constant monitoring -> Automated 24/7 signal detection
- Emotional decisions reduce profitability -> Systematic rule-based execution
- Difficulty identifying optimal entries -> Multi-indicator confluence signals
- Risk of catastrophic losses -> Circuit breaker protection

### 2.2 Secondary Stakeholder: Algorithm Developer

**Profile:**
- Developer/quant responsible for strategy improvement
- Conducts backtesting and parameter optimization
- Requires reproducible results for scientific validation

**Goals:**

| ID | Goal | Success Metric |
|----|------|----------------|
| G-AD-001 | Iterate strategy improvements rapidly | Backtest < 10 min for 60-day period |
| G-AD-002 | Validate strategy edge statistically | 95% confidence on E[R] estimation |
| G-AD-003 | Document failed experiments | All changes tracked with impact |

### 2.3 Watchdog Stakeholder: Risk Manager

**Profile:**
- Risk oversight role monitoring compliance with limits
- Ensures capital preservation during extreme markets

**Goals:**

| ID | Goal | Success Metric |
|----|------|----------------|
| G-RM-001 | Prevent catastrophic capital loss | Max drawdown < 20% enforced |
| G-RM-002 | Enforce position size limits | Risk per trade <= 1.75% |
| G-RM-003 | Audit trade execution | 100% trades recorded with metadata |

---

## 3. Functional Requirements

### 3.1 Signal Generation

#### R-SIGNAL-001: SSL Flow LONG Signal Generation

**Priority:** MUST
**Source:** SSL Flow Strategy Design
**Rationale:** Multi-indicator confluence required to achieve 70%+ win rate

**Specification:**
The system SHALL generate a LONG signal when ALL of the following conditions are satisfied simultaneously at candle close:

| Sub-ID | Condition | Measurement | Rationale |
|--------|-----------|-------------|-----------|
| R-SIGNAL-001.1 | Price above SSL baseline | `close > ssl_baseline` (HMA60) | Confirms uptrend |
| R-SIGNAL-001.2 | AlphaTrend buyers dominant | `at_buyers_dominant == TRUE AND at_is_flat == FALSE` | Confirms buyer control |
| R-SIGNAL-001.3 | Baseline touched in last 5 candles | `min(low[-5:]) <= ssl_baseline[-5:]` | Confirms retest |
| R-SIGNAL-001.4 | PBEMA above price | `pbema_bot > close` | Profit target reachable |
| R-SIGNAL-001.5 | PBEMA above baseline | `pbema_mid > ssl_baseline` | Flow exists |
| R-SIGNAL-001.6 | No PBEMA-SSL overlap | `abs(baseline - pbema_mid) / pbema_mid >= 0.005` | Clear path to target |
| R-SIGNAL-001.7 | RSI below limit | `rsi <= rsi_limit` (typically 70) | Not overbought |
| R-SIGNAL-001.8 | ADX shows trend strength | `adx >= adx_min` (typically 15) | Trend confirmed |

**Dependencies:** R-IND-001, R-IND-002, R-IND-003, R-IND-004, R-DATA-001

**Test Method:** Unit test with synthetic candle data, backtest validation

---

#### R-SIGNAL-002: SSL Flow SHORT Signal Generation

**Priority:** MUST
**Source:** SSL Flow Strategy Design

**Specification:**
The system SHALL generate a SHORT signal when ALL conditions are satisfied (inverse of LONG):

| Sub-ID | Condition | Measurement |
|--------|-----------|-------------|
| R-SIGNAL-002.1 | Price below SSL baseline | `close < ssl_baseline` |
| R-SIGNAL-002.2 | AlphaTrend sellers dominant | `at_sellers_dominant == TRUE AND at_is_flat == FALSE` |
| R-SIGNAL-002.3 | Baseline touched in last 5 candles | `max(high[-5:]) >= ssl_baseline[-5:]` |
| R-SIGNAL-002.4 | PBEMA below price | `pbema_top < close` |
| R-SIGNAL-002.5 | PBEMA below baseline | `pbema_mid < ssl_baseline` |
| R-SIGNAL-002.6 | No PBEMA-SSL overlap | `abs(baseline - pbema_mid) / pbema_mid >= 0.005` |
| R-SIGNAL-002.7 | RSI above inverse limit | `rsi >= (100 - rsi_limit)` |
| R-SIGNAL-002.8 | ADX shows trend strength | `adx >= adx_min` |

**Dependencies:** R-IND-001, R-IND-002, R-IND-003, R-IND-004, R-DATA-001

---

#### R-SIGNAL-003: PBEMA-SSL Overlap Detection

**Priority:** MUST
**Source:** Strategy Enhancement v42.x

**Specification:**
The system SHALL detect overlap between PBEMA cloud and SSL baseline:

```
overlap_distance = abs(ssl_baseline - pbema_mid) / pbema_mid
is_overlapping = overlap_distance < 0.005  # 0.5% threshold
```

When overlap is detected:
- Signal SHALL be rejected
- Rejection reason SHALL be "SSL-PBEMA Overlap (No Flow)"

**Rationale:** Overlapping indicators suggest no clear trend direction

---

#### R-SIGNAL-004: Signal Rejection Logging

**Priority:** SHOULD

**Specification:**
The system SHALL log rejection reasons when signal conditions are not met:

| Rejection Reason | Condition |
|------------------|-----------|
| "Price below baseline" | LONG: close <= ssl_baseline |
| "AlphaTrend not buyers" | LONG: at_buyers_dominant == FALSE |
| "AlphaTrend flat" | at_is_flat == TRUE |
| "No baseline touch" | No touch in lookback period |
| "RSI above limit" | LONG: rsi > rsi_limit |
| "ADX too low" | adx < adx_min |
| "SSL-PBEMA Overlap" | overlap_distance < 0.005 |

---

### 3.2 Indicator Calculation

#### R-IND-001: SSL Baseline (HMA60)

**Priority:** MUST

**Specification:**
The system SHALL calculate SSL Baseline using Hull Moving Average with period 60:

```
WMA1 = WMA(close, period/2)
WMA2 = WMA(close, period)
HMA = WMA(2 * WMA1 - WMA2, sqrt(period))
ssl_baseline = HMA(close, 60)
```

**Implementation:** `core/indicators.py`

---

#### R-IND-002: PBEMA Cloud (EMA200)

**Priority:** MUST

**Specification:**
The system SHALL calculate PBEMA Cloud as two EMA200 lines:

```
pbema_top = EMA(high, 200)
pbema_bot = EMA(close, 200)
pbema_mid = (pbema_top + pbema_bot) / 2
```

**Usage:**
- LONG TP target: pbema_bot
- SHORT TP target: pbema_top

---

#### R-IND-003: AlphaTrend Indicator

**Priority:** MUST

**Specification:**
The system SHALL calculate AlphaTrend for buyer/seller dominance:

```
atr = ATR(14)
mfi = MFI(14)
upT = low - atr * multiplier
downT = high + atr * multiplier

if mfi >= 50:
    alphatrend = max(upT, prev_alphatrend)
else:
    alphatrend = min(downT, prev_alphatrend)
```

**Outputs:**
- `at_buyers_dominant`: AlphaTrend rising AND price above AlphaTrend
- `at_sellers_dominant`: AlphaTrend falling AND price below AlphaTrend
- `at_is_flat`: AlphaTrend unchanged for N periods

---

#### R-IND-004: RSI Indicator

**Priority:** MUST

**Specification:**
The system SHALL calculate RSI with period 14:

```
rsi = RSI(close, 14)
```

---

#### R-IND-005: ADX Indicator

**Priority:** MUST

**Specification:**
The system SHALL calculate ADX with period 14:

```
adx = ADX(14)
```

---

#### R-IND-006: Keltner Channels

**Priority:** SHOULD

**Specification:**
The system SHALL calculate Keltner Channels around SSL baseline:

```
tr = TrueRange(high, low, close)
keltner_upper = ssl_baseline + EMA(tr, 20) * 0.2
keltner_lower = ssl_baseline - EMA(tr, 20) * 0.2
```

---

### 3.3 Trade Execution

#### R-EXEC-001: Market Order Placement

**Priority:** MUST

**Specification:**
When a valid signal is generated, the system SHALL:
1. Calculate position size based on risk parameters (R-RISK-003)
2. Submit market order to Binance Futures API within 200ms of signal
3. Place TP bracket order at calculated target
4. Place SL bracket order at calculated stop level

**Pre-conditions:**
- Circuit breaker status == ACTIVE
- Portfolio risk < max_portfolio_risk_pct
- Account balance sufficient for position
- Binance API connection healthy

**Success Guarantee:**
- Order filled within 500ms (95th percentile)
- TP and SL orders active within 200ms of fill
- Trade metadata recorded to database

---

#### R-EXEC-002: Take-Profit Calculation

**Priority:** MUST

**Specification:**
The system SHALL calculate take-profit price:

**For LONG trades:**
```
take_profit = pbema_bot  # PBEMA cloud lower bound
```

**For SHORT trades:**
```
take_profit = pbema_top  # PBEMA cloud upper bound
```

---

#### R-EXEC-003: Stop-Loss Calculation

**Priority:** MUST

**Specification:**
The system SHALL calculate stop-loss price:

**For LONG trades:**
```
swing_low = min(low[-20:])  # Lowest low in 20 candles
stop_loss = min(swing_low * 0.998, ssl_baseline * 0.998)
```

**For SHORT trades:**
```
swing_high = max(high[-20:])  # Highest high in 20 candles
stop_loss = max(swing_high * 1.002, ssl_baseline * 1.002)
```

**Rationale:** 0.2% buffer provides clearance beyond exact swing level

---

#### R-EXEC-004: Minimum Stop-Loss Distance

**Priority:** SHOULD

**Specification:**
The system SHALL optionally enforce minimum SL distance:

| Symbol Type | Minimum Distance |
|-------------|------------------|
| BTC, ETH | 1.0% from entry |
| Altcoins | 1.5% from entry |

**Validation Modes:**
- `"off"`: No minimum distance
- `"reject"`: Reject trade if SL distance < minimum
- `"widen"`: Widen SL to minimum and adjust TP proportionally

**Configuration:** `sl_validation_mode` (default: "off")

---

#### R-EXEC-005: Risk-Reward Validation

**Priority:** MUST

**Specification:**
The system SHALL validate risk-reward ratio before trade:

```
rr_ratio = abs(take_profit - entry) / abs(entry - stop_loss)
```

Trade SHALL be rejected if `rr_ratio < min_rr` (configurable, default 0.5)

---

#### R-EXEC-006: Partial Take-Profit (Exit Profiles)

**Priority:** SHOULD

**Specification:**
The system SHALL support exit profile configurations:

**CLIP Profile:**
- Partial TP at 45% of target: Close 50% position
- Final TP at 100% of target: Close remaining 50%

**RUNNER Profile:**
- Partial TP at 70% of target: Close 33% position
- Final TP at 100% of target: Close remaining 67%

**Configuration:** `exit_profile` ("clip" or "runner")

---

### 3.4 Risk Management

#### R-RISK-001: Risk Per Trade Limit

**Priority:** MUST

**Specification:**
The system SHALL limit risk per trade to configurable percentage:

```
max_risk = account_balance * risk_per_trade_pct
risk = position_size * abs(entry - stop_loss)
REQUIRE: risk <= max_risk
```

**Default:** 1.75% of account balance

---

#### R-RISK-002: Maximum Portfolio Risk

**Priority:** MUST

**Specification:**
The system SHALL limit total portfolio risk:

```
total_risk = sum(position.risk for position in open_positions)
max_portfolio_risk = account_balance * max_portfolio_risk_pct
REQUIRE: total_risk <= max_portfolio_risk
```

**Default:** 5% of account balance

If exceeded, new positions SHALL be rejected with reason "Portfolio risk limit exceeded"

---

#### R-RISK-003: R-Multiple Position Sizing

**Priority:** MUST

**Specification:**
The system SHALL calculate position size using R-multiple methodology:

```
risk_amount = account_balance * risk_per_trade_pct
price_risk = abs(entry - stop_loss)
position_size = risk_amount / price_risk
```

**Example:**
- Account: $2,000
- Risk: 1.75% = $35
- Entry: $45,000, SL: $44,500
- Risk per BTC: $500
- Position: $35 / $500 = 0.07 BTC

---

#### R-RISK-004: Strategy Wallet Isolation

**Priority:** SHOULD

**Specification:**
The system SHALL maintain isolated balance tracking per strategy:

```
strategy_wallet = {
    "ssl_flow": {"balance": 2000.0, "pnl": 0.0},
    "keltner_bounce": {"balance": 0.0, "pnl": 0.0}
}
```

Each strategy's losses SHALL NOT affect other strategy wallets.

---

#### R-RISK-005: Correlation-Based Position Sizing

**Priority:** SHOULD
**Source:** Hedge Fund Due Diligence Report

**Specification:**
The system SHALL adjust position size based on asset correlation:

**Rules:**
1. Maximum 2 positions in same direction (LONG or SHORT)
2. Correlated assets (correlation > 0.7) reduce position size by 50%
3. Opposite direction positions provide hedging benefit

**Calculation:**
```
effective_positions = N / (1 + (N-1) * avg_correlation)
```

**Example:**
- 2 LONG positions (BTC+ETH, corr=0.85): 1.04 effective positions
- Position 2 size: 50% of normal ($35 -> $17.50)

---

### 3.5 Circuit Breaker

#### R-CB-001: Stream-Level Max Loss

**Priority:** MUST

**Specification:**
The system SHALL halt trading for a stream when cumulative loss exceeds threshold:

```
if stream_cumulative_pnl <= -200:  # Default: -$200
    stream.status = KILLED
```

**Effects:**
- No new signals evaluated for stream
- Open positions allowed to complete
- Telegram alert sent

**Recovery:** Manual reset required

---

#### R-CB-002: Stream-Level Max Drawdown

**Priority:** MUST

**Specification:**
The system SHALL halt trading when stream drawdown exceeds threshold:

```
peak_balance = max(stream_balance_history)
drawdown = peak_balance - current_balance
if drawdown >= 100:  # Default: $100
    stream.status = KILLED
```

---

#### R-CB-003: Stream-Level Consecutive Stops

**Priority:** SHOULD

**Specification:**
The system SHALL halt trading after consecutive full stop-losses:

```
if consecutive_full_stops >= 2:
    stream.status = KILLED
```

---

#### R-CB-004: Global Max Drawdown

**Priority:** MUST

**Specification:**
The system SHALL halt ALL trading when account drawdown exceeds threshold:

```
peak = all_time_high_balance
drawdown_pct = (peak - current) / peak
if drawdown_pct >= 0.20:  # 20%
    kill_all_streams()
    close_all_positions()
```

**Effects:**
- All streams KILLED
- All positions closed at market
- System requires manual restart
- Telegram alert sent

---

#### R-CB-005: Global Daily Loss Limit

**Priority:** SHOULD

**Specification:**
The system SHALL halt trading when daily loss exceeds threshold:

```
if daily_pnl <= -400:  # Default: -$400
    kill_all_streams()
```

---

#### R-CB-006: Global Weekly Loss Limit

**Priority:** SHOULD

**Specification:**
The system SHALL halt trading when weekly loss exceeds threshold:

```
if weekly_pnl <= -800:  # Default: -$800
    kill_all_streams()
```

---

#### R-CB-007: Rolling E[R] Edge Detection

**Priority:** SHOULD

**Specification:**
The system SHALL monitor rolling E[R] for edge degradation:

```
rolling_er = mean(last_20_trades.r_multiple)
if rolling_er < min_expectancy_threshold:
    send_warning("Edge degradation detected")
```

---

### 3.6 Optimizer

#### R-OPT-001: Walk-Forward Optimization

**Priority:** MUST

**Specification:**
The system SHALL perform walk-forward optimization:

```
for each window:
    train_data = data[window_start - lookback : window_start]
    test_data = data[window_start : window_start + forward_days]

    best_config = optimize(train_data, candidate_configs)
    results = backtest(test_data, best_config)
```

**Default Parameters:**
- Lookback: 60 days
- Forward: 7 days
- Mode: Weekly re-optimization

---

#### R-OPT-002: Deterministic Results

**Priority:** MUST
**Source:** v43.x Determinism Fix

**Specification:**
The system SHALL produce identical results for identical inputs:

**Requirements:**
1. Random seed set: `random.seed(42)`, `np.random.seed(42)`
2. Results sorted by config hash (deterministic ordering)
3. Float comparisons use epsilon tolerance
4. Tie-breaking uses config hash

**Validation:** Same test run twice SHALL produce identical output

---

#### R-OPT-003: Minimum Trade Threshold

**Priority:** MUST

**Specification:**
The system SHALL reject configurations with insufficient trades:

```
if in_sample_trades < hard_min_trades:  # Default: 5
    reject_config()
```

**Rationale:** Prevents overfitting to sparse data

---

#### R-OPT-004: Parallel Optimization

**Priority:** SHOULD

**Specification:**
The system SHALL support parallel candidate evaluation:

- ThreadPoolExecutor with max 4 workers
- Results collected and sorted deterministically
- ~4x speedup for 8-symbol tests

---

#### R-OPT-005: Master Cache

**Priority:** SHOULD

**Specification:**
The system SHALL cache historical data across optimization windows:

```
master_cache = {
    "BTCUSDT-15m": DataFrame,
    "ETHUSDT-15m": DataFrame,
    ...
}
```

Reduces data fetching time by ~80%

---

---

## 4. Non-Functional Requirements

### 4.1 Performance

#### R-PERF-001: Signal Generation Latency

**Priority:** MUST

**Specification:**
The system SHALL generate signal decision within latency targets:

| Percentile | Max Latency |
|------------|-------------|
| 95th | 100ms |
| 99th | 250ms |
| Mean | 50ms |

**Measurement:** Time from candle close to signal decision

---

#### R-PERF-002: Multi-Symbol Concurrent Monitoring

**Priority:** MUST

**Specification:**
The system SHALL monitor 60 concurrent streams (20 symbols x 3 timeframes):

| Metric | Target |
|--------|--------|
| CPU usage | < 80% sustained |
| Memory usage | < 2GB |
| Missed candles | 0 |

**Degradation:** If targets exceeded, prioritize higher timeframes (1h > 15m > 5m)

---

#### R-PERF-003: Backtest Execution Time

**Priority:** SHOULD

**Specification:**
The system SHALL complete 60-day backtest within 10 minutes:

- Single symbol-timeframe stream
- ~5,760 candles for 15m timeframe
- Full indicator calculation and signal generation

---

#### R-PERF-004: Order Execution Latency

**Priority:** MUST

**Specification:**
The system SHALL execute orders within latency targets:

| Metric | Target |
|--------|--------|
| Order submission | < 200ms from signal |
| Order fill (95th) | < 500ms |
| Bracket orders | < 200ms after fill |

---

### 4.2 Reliability

#### R-REL-001: System Uptime

**Priority:** MUST

**Specification:**
The system SHALL maintain 99.5% uptime during market hours (24/7):

- 3.65 hours downtime/month maximum
- Planned maintenance: 2 hours/month max
- Unplanned outages: 1.65 hours/month max

---

#### R-REL-002: WebSocket Reconnection

**Priority:** MUST

**Specification:**
The system SHALL automatically recover from WebSocket disconnections:

1. Detect disconnection within 10 seconds
2. Wait 5 seconds (cooldown)
3. Reconnect with exponential backoff (max 3 attempts)
4. Resume streaming
5. If failed, send Telegram alert

---

#### R-REL-003: Order Retry Policy

**Priority:** MUST

**Specification:**
The system SHALL retry failed orders with exponential backoff:

| Attempt | Wait Time |
|---------|-----------|
| 1 | 0s (immediate) |
| 2 | 1s |
| 3 | 2s |
| 4 | 4s |

**Non-Retriable Errors:**
- Invalid parameters (HTTP 400)
- Insufficient balance (HTTP 400)
- Invalid API key (HTTP 401) - halt trading

---

#### R-REL-004: Data Integrity

**Priority:** MUST

**Specification:**
The system SHALL validate all trade records:

- All trades logged with complete metadata
- No trades lost during crashes
- Transaction log recoverable

---

### 4.3 Security

#### R-SEC-001: API Key Encryption

**Priority:** MUST

**Specification:**
The system SHALL encrypt API keys at rest:

- Encryption: AES-256
- Key derivation: PBKDF2 with 100,000 iterations
- User prompted for passphrase on startup

---

#### R-SEC-002: TLS Communication

**Priority:** MUST

**Specification:**
The system SHALL use TLS 1.2+ for all API communication:

- HTTPS for REST API
- WSS for WebSocket streams
- Certificate validation enabled

---

#### R-SEC-003: Safe Expression Evaluation

**Priority:** MUST

**Specification:**
The system SHALL use AST-based safe evaluation instead of `eval()`:

**Allowed:**
- Comparisons: `>`, `<`, `>=`, `<=`, `==`, `!=`
- Boolean: `and`, `or`, `not`
- Arithmetic: `+`, `-`, `*`, `/`, `%`, `**`
- Variable access from context

**Blocked:**
- Function calls
- Imports
- Code execution (`eval`, `exec`)
- Attribute access

---

#### R-SEC-004: Safe Pickle Loading

**Priority:** SHOULD

**Specification:**
The system SHALL validate pickle files before loading:

- Whitelist allowed classes
- Reject unknown class instantiation
- Log warnings for suspicious content

---

---

## 5. Data Requirements

### 5.1 Real-Time Data

#### R-DATA-001: Real-Time OHLCV

**Priority:** MUST

**Specification:**
The system SHALL receive real-time OHLCV data via WebSocket:

| Field | Description |
|-------|-------------|
| open | Opening price |
| high | Highest price |
| low | Lowest price |
| close | Closing price |
| volume | Trade volume |
| timestamp | Candle close time |

**Freshness:** Data received within 1 second of candle close

---

#### R-DATA-002: Data Completeness

**Priority:** MUST

**Specification:**
The system SHALL ensure data completeness:

- Zero missed candles during normal operation
- Backfill gaps on reconnection
- Validate: high >= low, high >= close, low <= close

---

### 5.2 Historical Data

#### R-DATA-003: Historical Data Range

**Priority:** MUST

**Specification:**
The system SHALL fetch historical data for backtesting:

- Minimum: 365 days for annual backtests
- Typical: 60-90 days for walk-forward optimization

---

#### R-DATA-004: Data Validation

**Priority:** MUST

**Specification:**
The system SHALL validate historical data:

- No missing candles in range
- Timestamps sequential and evenly spaced
- Price values within reasonable bounds

---

---

## 6. Interface Requirements

### 6.1 Binance API

#### R-API-001: REST API Integration

**Priority:** MUST

**Specification:**
The system SHALL integrate with Binance Futures REST API for:

- Account balance queries
- Historical data fetching
- Order placement (market, limit, stop-loss, take-profit)
- Order status queries
- Position queries

---

#### R-API-002: Rate Limit Compliance

**Priority:** MUST

**Specification:**
The system SHALL respect Binance rate limits:

- 1200 requests/minute maximum
- Implement request throttling
- Handle HTTP 429 with backoff

---

#### R-API-003: WebSocket Streams

**Priority:** MUST

**Specification:**
The system SHALL connect to WebSocket streams:

- Kline streams for configured symbols/timeframes
- User data stream for execution updates
- Heartbeat/ping-pong for keep-alive

---

### 6.2 Telegram Notifications

#### R-TG-001: Trade Open Notification

**Priority:** SHOULD

**Specification:**
The system SHALL send notification when trade opens:

```
[LONG/SHORT] BTCUSDT 15m
Entry: $45,000
TP: $45,500 (1.11% gain)
SL: $44,500 (1.11% risk)
RR: 1.00
Position: 0.07 BTC ($3,150)
Risk: $35 (1.75%)
Reason: SSL Flow LONG - All conditions met
```

---

#### R-TG-002: Trade Close Notification

**Priority:** SHOULD

**Specification:**
The system SHALL send notification when trade closes:

```
[LONG/SHORT] BTCUSDT 15m CLOSED
Entry: $45,000 -> Exit: $45,400
PnL: +$28 (+0.89%)
R-Multiple: +0.80R
Duration: 2h 15m
Reason: Take Profit
```

---

#### R-TG-003: Circuit Breaker Alert

**Priority:** MUST

**Specification:**
The system SHALL send alert when circuit breaker activates:

```
CIRCUIT BREAKER ACTIVATED
Stream: BTCUSDT-15m
Trigger: Max loss exceeded (-$205)
Action: Trading halted
Recovery: Manual reset required
```

---

#### R-TG-004: Error Alert

**Priority:** SHOULD

**Specification:**
The system SHALL send alert on critical errors:

- WebSocket reconnection failed
- Order execution failed
- API authentication error

---

---

## 7. Acceptance Criteria

### 7.1 Strategy Performance Acceptance

The system SHALL be considered production-ready when meeting ALL criteria on 6-month backtest:

| Metric | Minimum | Target |
|--------|---------|--------|
| Win Rate | >= 65% | >= 70% |
| Expected R (E[R]) | >= 0.05 | >= 0.08 |
| Max Drawdown | <= 25% | <= 20% |
| Trade Frequency | >= 12/year | >= 20/year |
| Sharpe Ratio | >= 0.8 | >= 1.2 |

**Test Symbols:** BTCUSDT, ETHUSDT, LINKUSDT
**Test Period:** 2025-01-01 to 2025-06-30

---

### 7.2 System Performance Acceptance

| Metric | Requirement |
|--------|-------------|
| Signal latency (95th) | < 100ms |
| Order execution (95th) | < 500ms |
| Concurrent streams | 60 |
| CPU usage | < 80% |
| Memory usage | < 2GB |
| Uptime | >= 99.5% |

---

---

## 8. Specification by Example

### 8.1 Valid LONG Signal Generation

```gherkin
Feature: SSL Flow LONG Signal Generation

  Background:
    Given symbol is "BTCUSDT"
    And timeframe is "15m"

  Scenario: Valid LONG signal with all conditions satisfied
    Given the current candle has:
      | Attribute | Value |
      | close | 45,000 |
      | ssl_baseline | 44,600 |
      | pbema_top | 46,000 |
      | pbema_bot | 45,400 |
      | pbema_mid | 45,700 |
      | at_buyers_dominant | TRUE |
      | at_is_flat | FALSE |
      | rsi | 68 |
      | adx | 22 |
    And baseline was touched at index -2
    When signal generation is triggered
    Then signal_type SHALL be "LONG"
    And entry_price SHALL be 45,000
    And take_profit SHALL be 45,400
    And stop_loss SHALL be min(swing_low * 0.998, 44,600 * 0.998)
```

---

### 8.2 LONG Signal Rejected - Overlap

```gherkin
Scenario: LONG signal rejected due to PBEMA-SSL overlap
  Given ssl_baseline is 45,000
  And pbema_mid is 45,200
  When overlap distance is calculated
  Then overlap_distance SHALL be 0.0044 (0.44%)
  And is_overlapping SHALL be TRUE
  And signal SHALL be "NO_SIGNAL"
  And rejection_reason SHALL be "SSL-PBEMA Overlap (No Flow)"
```

---

### 8.3 Position Size Calculation

```gherkin
Scenario: Position size respects risk limit
  Given account balance is $2,000
  And risk_per_trade_pct is 0.0175
  And entry price is 45,000
  And stop_loss is 44,500
  When position size is calculated
  Then max_risk SHALL be $35
  And price_risk SHALL be $500
  And position_size SHALL be 0.07 BTC
```

---

### 8.4 Circuit Breaker Activation

```gherkin
Scenario: Stream killed after max loss threshold
  Given stream "BTCUSDT-15m" has PnL of -$180
  And max_loss is -$200
  When trade closes with -$25 loss
  Then total_pnl SHALL be -$205
  And stream.status SHALL be KILLED
  And Telegram alert SHALL be sent
  And no new signals SHALL be evaluated
```

---

### 8.5 Portfolio Risk Limit

```gherkin
Scenario: Trade rejected when portfolio risk exceeded
  Given account balance is $2,000
  And max_portfolio_risk_pct is 0.05
  And current total_risk is $75
  And new trade risk is $30
  When portfolio check is performed
  Then new_total_risk SHALL be $105
  And max_allowed SHALL be $100
  And trade SHALL be REJECTED
  And reason SHALL be "Portfolio risk limit exceeded"
```

---

---

## 9. Glossary

| Term | Definition |
|------|------------|
| **SSL Baseline** | Hull Moving Average (HMA) with period 60, primary trend indicator serving as dynamic support/resistance |
| **PBEMA Cloud** | Pair of EMAs (EMA200 of high and close) creating a "cloud" zone used as take-profit target |
| **AlphaTrend** | Proprietary trend indicator combining ATR and MFI/RSI to determine buyer/seller dominance |
| **R-Multiple** | Performance metric: ratio of trade PnL to initial risk. +2.0R means profit was 2x initial risk |
| **E[R]** | Expected R-multiple, average of all trade R-multiples, represents strategy edge |
| **Circuit Breaker** | Automated kill switch halting trading when loss/drawdown thresholds exceeded |
| **Walk-Forward** | Time-series optimization: optimize on historical window, validate on forward (OOS) window |
| **Overlap** | Condition where PBEMA cloud and SSL baseline are within 0.5% distance |
| **Baseline Touch** | Price touching SSL baseline within lookback period (5 candles), confirms retest |
| **Flow** | Clear path from SSL baseline to PBEMA target with no overlap |
| **Swing Low/High** | Lowest low / highest high in lookback period (20 candles) for SL calculation |
| **OOS** | Out-of-sample, data not used during optimization for validation |
| **Lookback** | Historical period used for optimization (default 60 days) |
| **Forward Period** | Future period for walk-forward validation (default 7 days) |

---

## 10. Requirements Traceability

### 10.1 Requirement to Code Mapping

| Requirement | Implementation | Test |
|-------------|----------------|------|
| R-SIGNAL-001 | `strategies/ssl_flow.py:check_ssl_flow_signal_long()` | `tests/test_signals.py:test_long_signal()` |
| R-SIGNAL-002 | `strategies/ssl_flow.py:check_ssl_flow_signal_short()` | `tests/test_signals.py:test_short_signal()` |
| R-SIGNAL-003 | `strategies/ssl_flow.py:check_overlap()` | `tests/test_signals.py:test_overlap_detection()` |
| R-IND-001 | `core/indicators.py:calculate_hma()` | `tests/test_indicators.py:test_hma()` |
| R-IND-002 | `core/indicators.py:calculate_pbema()` | `tests/test_indicators.py:test_pbema()` |
| R-IND-003 | `core/indicators.py:calculate_alphatrend()` | `tests/test_indicators.py:test_alphatrend()` |
| R-EXEC-001 | `core/trade_manager.py:open_trade()` | `tests/test_trade_manager.py:test_market_order()` |
| R-EXEC-002 | `strategies/ssl_flow.py` (TP calculation) | `tests/test_signals.py:test_tp_calculation()` |
| R-EXEC-003 | `strategies/ssl_flow.py` (SL calculation) | `tests/test_signals.py:test_sl_calculation()` |
| R-RISK-001 | `core/trade_manager.py:calculate_position_size()` | `tests/test_risk.py:test_risk_limit()` |
| R-RISK-002 | `core/trade_manager.py:check_portfolio_risk()` | `tests/test_risk.py:test_portfolio_risk()` |
| R-CB-001 | `core/trade_manager.py:check_circuit_breaker()` | `tests/test_circuit_breaker.py` |
| R-CB-004 | `core/trade_manager.py:check_global_circuit_breaker()` | `tests/test_circuit_breaker.py` |
| R-OPT-001 | `core/optimizer.py:walk_forward_optimize()` | `tests/test_optimizer.py` |
| R-OPT-002 | `core/optimizer.py` (determinism) | `tests/test_optimizer.py:test_determinism()` |
| R-SEC-003 | `core/safe_eval.py:SafeEvaluator` | `core/safe_eval.py:__main__` |

### 10.2 Requirements Coverage Summary

| Category | Total | Implemented | Tested | Coverage |
|----------|-------|-------------|--------|----------|
| Signal Generation | 4 | 4 | 4 | 100% |
| Indicators | 6 | 6 | 6 | 100% |
| Trade Execution | 6 | 6 | 5 | 83% |
| Risk Management | 5 | 5 | 4 | 80% |
| Circuit Breaker | 7 | 7 | 4 | 57% |
| Optimizer | 5 | 5 | 3 | 60% |
| Performance | 4 | 3 | 2 | 50% |
| Reliability | 4 | 4 | 2 | 50% |
| Security | 4 | 3 | 2 | 50% |
| Data | 4 | 4 | 3 | 75% |
| Interface | 7 | 6 | 4 | 57% |
| **TOTAL** | **56** | **53** | **39** | **70%** |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-02 | Initial requirements specification from CLAUDE.md analysis |

---

**END OF REQUIREMENTS SPECIFICATION**
