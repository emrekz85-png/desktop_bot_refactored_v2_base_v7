#!/usr/bin/env python3
"""
Demo script showing ROC filter integration with SSL Flow strategy
Demonstrates how the filter blocks counter-trend entries
"""

import pandas as pd
import numpy as np

# Mock minimal data for demonstration
def create_mock_data_with_trend(trend='up', bars=100):
    """Create mock OHLCV data with indicators for testing"""

    if trend == 'up':
        # Uptrend: price rises from 100 to 110 over last 15 bars (strong pump)
        base_prices = np.ones(bars) * 100
        base_prices[-15:] = np.linspace(100, 110, 15)  # 10% gain in last 15 bars
    elif trend == 'down':
        # Downtrend: price falls from 110 to 100 over last 15 bars (strong dump)
        base_prices = np.ones(bars) * 110
        base_prices[-15:] = np.linspace(110, 100, 15)  # -9% loss in last 15 bars
    else:
        # Sideways: price oscillates around 100
        base_prices = 100 + np.sin(np.linspace(0, 4*np.pi, bars)) * 1

    # Add some noise
    noise = np.random.normal(0, 0.5, bars)
    close = base_prices + noise

    # Create OHLC
    high = close * 1.005
    low = close * 0.995
    open_ = close - (high - low) * 0.3

    # Convert to Series for rolling operations
    close_series = pd.Series(close)

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.uniform(1000, 5000, bars),

        # SSL Flow indicators (simplified)
        'baseline': close_series.rolling(60, min_periods=1).mean().values,  # HMA approximation
        'pb_ema_top': close_series.rolling(200, min_periods=1).max().values,
        'pb_ema_bot': close_series.rolling(200, min_periods=1).min().values,

        # AlphaTrend indicators (mock - buyers/sellers based on trend)
        'alphatrend': close,
        'alphatrend_2': close * 0.99,
        'at_buyers_dominant': trend == 'up',
        'at_sellers_dominant': trend == 'down',
        'at_is_flat': trend == 'sideways',

        # Other required indicators
        'rsi': 50.0,  # Neutral RSI
        'adx': 25.0,  # Trending ADX
    })

    return df


def demo_roc_filter():
    """Demonstrate ROC filter in different market conditions"""
    from strategies.ssl_flow import calculate_roc_filter

    print("\n" + "="*70)
    print("ROC FILTER DEMO - SSL FLOW STRATEGY")
    print("="*70)

    # Scenario 1: Strong Uptrend
    print("\n" + "-"*70)
    print("SCENARIO 1: Strong Uptrend")
    print("-"*70)

    df_up = create_mock_data_with_trend('up', 100)
    long_ok, short_ok, roc = calculate_roc_filter(df_up, index=-1, roc_period=10, roc_threshold=2.5)

    print(f"\nMarket: UPTREND")
    print(f"Price Movement: {df_up['close'].iloc[0]:.2f} -> {df_up['close'].iloc[-1]:.2f}")
    print(f"ROC (10-bar): {roc:+.2f}%")
    print(f"\nFilter Decision:")
    print(f"  LONG Entry:  {'✓ ALLOWED' if long_ok else '✗ BLOCKED'}")
    print(f"  SHORT Entry: {'✓ ALLOWED' if short_ok else '✗ BLOCKED (Strong uptrend - dont short into strength)'}")

    # Scenario 2: Strong Downtrend
    print("\n" + "-"*70)
    print("SCENARIO 2: Strong Downtrend")
    print("-"*70)

    df_down = create_mock_data_with_trend('down', 100)
    long_ok, short_ok, roc = calculate_roc_filter(df_down, index=-1, roc_period=10, roc_threshold=2.5)

    print(f"\nMarket: DOWNTREND")
    print(f"Price Movement: {df_down['close'].iloc[0]:.2f} -> {df_down['close'].iloc[-1]:.2f}")
    print(f"ROC (10-bar): {roc:+.2f}%")
    print(f"\nFilter Decision:")
    print(f"  LONG Entry:  {'✓ ALLOWED' if long_ok else '✗ BLOCKED (Strong downtrend - dont buy falling knife)'}")
    print(f"  SHORT Entry: {'✓ ALLOWED' if short_ok else '✗ BLOCKED'}")

    # Scenario 3: Sideways
    print("\n" + "-"*70)
    print("SCENARIO 3: Sideways/Consolidation")
    print("-"*70)

    df_sideways = create_mock_data_with_trend('sideways', 100)
    long_ok, short_ok, roc = calculate_roc_filter(df_sideways, index=-1, roc_period=10, roc_threshold=2.5)

    print(f"\nMarket: SIDEWAYS")
    print(f"Price Movement: {df_sideways['close'].iloc[0]:.2f} -> {df_sideways['close'].iloc[-1]:.2f}")
    print(f"ROC (10-bar): {roc:+.2f}%")
    print(f"\nFilter Decision:")
    print(f"  LONG Entry:  {'✓ ALLOWED (Weak momentum)' if long_ok else '✗ BLOCKED'}")
    print(f"  SHORT Entry: {'✓ ALLOWED (Weak momentum)' if short_ok else '✗ BLOCKED'}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
The ROC (Rate of Change) Momentum Filter prevents counter-trend entries by:

1. BLOCKING SHORT entries when ROC > +2.5% (strong uptrend)
   → Don't short into strength

2. BLOCKING LONG entries when ROC < -2.5% (strong downtrend)
   → Don't buy falling knives

3. ALLOWING both entries when |ROC| < 2.5% (weak momentum)
   → No strong directional bias

This filter is ENABLED by default in ssl_flow.py (v1.10.0)

Expected to reduce TREND ERRORS which accounted for 8/14 losing trades (57%)
""")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducible demo
    np.random.seed(42)

    try:
        demo_roc_filter()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
