#!/usr/bin/env python3
"""
Test signal detection to see why no trades are being generated.
"""

from test_helpers import get_binance_client, get_default_config
from core.indicators import calculate_indicators
from core.trading_engine import TradingEngine

print("="*80)
print("SIGNAL DETECTION TEST")
print("="*80)

# Fetch data
print("\n[1] Fetching BTCUSDT 15m data (5000 candles)...")
client = get_binance_client()
df = client.get_klines("BTCUSDT", "15m", limit=5000)
print(f"✓ Fetched {len(df)} candles")

# Calculate indicators
print("\n[2] Calculating indicators...")
calculate_indicators(df)
print("✓ Indicators calculated")

# Test config - use loose parameters for maximum signals
test_config = {
    "rr": 1.2,  # Low RR for easier entry
    "rsi": 70,  # Liberal RSI threshold
    "at_active": True,
    "use_trailing": False,
    "use_partial": True,
    "use_dynamic_pbema_tp": True,
    "strategy_mode": "ssl_flow",
    "sl_validation_mode": "off",
    "partial_trigger": 0.40,
    "partial_fraction": 0.50,
    "partial_rr_adjustment": False,
    "dynamic_tp_only_after_partial": False,
    "dynamic_tp_clamp_mode": "none",
}

print("\n[3] Scanning for signals...")
print(f"    Config: RR={test_config['rr']}, RSI={test_config['rsi']}, AT={test_config['at_active']}")

warmup = 250
signals_found = 0
long_signals = 0
short_signals = 0
rejected_signals = 0

for i in range(warmup, len(df) - 2):
    s_type, s_entry, s_tp, s_sl, s_reason = TradingEngine.check_signal(
        df,
        config=test_config,
        index=i,
        return_debug=False,
    )

    if s_type:
        signals_found += 1

        if "ACCEPTED" in s_reason:
            if s_type == "LONG":
                long_signals += 1
            else:
                short_signals += 1

            # Show first 5 accepted signals
            if (long_signals + short_signals) <= 5:
                timestamp = df.iloc[i]["timestamp"]
                print(f"  ✓ Signal #{long_signals + short_signals}: {s_type} at {timestamp}")
                print(f"     Entry: {s_entry:.2f}, TP: {s_tp:.2f}, SL: {s_sl:.2f}")
                print(f"     Reason: {s_reason}")
        else:
            rejected_signals += 1
            # Show first 3 rejections
            if rejected_signals <= 3:
                timestamp = df.iloc[i]["timestamp"]
                print(f"  ✗ Rejected at {timestamp}: {s_reason}")

print("\n" + "="*80)
print("SIGNAL SUMMARY")
print("="*80)
print(f"Total signals detected: {signals_found}")
print(f"  LONG accepted: {long_signals}")
print(f"  SHORT accepted: {short_signals}")
print(f"  Rejected: {rejected_signals}")
print()

if long_signals + short_signals == 0:
    print("⚠️  NO SIGNALS ACCEPTED!")
    print("\nPossible reasons:")
    print("  1. SSL baseline not touching price (lookback too short?)")
    print("  2. AlphaTrend flat filter rejecting all signals")
    print("  3. PBEMA distance too small")
    print("  4. RR not met after entry validation")
    print("\nNext step: Run with return_debug=True to see rejection reasons")
else:
    print(f"✓ Found {long_signals + short_signals} accepted signals")
    print(f"  Signal rate: {(long_signals + short_signals) / (len(df) - warmup) * 100:.2f}% of candles")

print("="*80)
