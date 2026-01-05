#!/usr/bin/env python3
"""Check price drops in data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core import get_client, calculate_indicators, set_backtest_mode
import pandas as pd
import numpy as np

set_backtest_mode(True)
client = get_client()

df = client.get_klines(symbol='BTCUSDT', interval='15m', limit=500)
df = calculate_indicators(df, timeframe='15m')

print("="*70)
print("PRICE DROP ANALYSIS (Last 500 bars)")
print("="*70)

highs = df['high'].values
lows = df['low'].values

# Find all sharp drops
drops = []

for i in range(5, len(df)):
    # Find recent high (last 5 bars)
    recent_high_idx = i - 5 + np.argmax(highs[i-5:i])
    recent_high = highs[recent_high_idx]

    # Current low
    current_low = lows[i]

    # Drop %
    drop_pct = (recent_high - current_low) / recent_high

    # Bars to drop
    bars = i - recent_high_idx

    if drop_pct >= 0.01:  # 1%+ drop
        drops.append((i, drop_pct, bars, recent_high, current_low))

print(f"\n📊 Drops >= 1% in 500 bars: {len(drops)}")

if drops:
    print(f"\n📉 TOP 20 DROPS (by %):")
    sorted_drops = sorted(drops, key=lambda x: x[1], reverse=True)[:20]

    for idx, (i, drop, bars, high, low) in enumerate(sorted_drops, 1):
        print(f"   {idx}. Bar #{i}: {drop:.2%} drop in {bars} bars (${high:.0f} → ${low:.0f})")

    # Stats
    all_drop_pcts = [d[1] for d in drops]
    all_bars = [d[2] for d in drops]

    print(f"\n📐 DROP STATISTICS:")
    print(f"   Min drop: {min(all_drop_pcts):.2%}")
    print(f"   Max drop: {max(all_drop_pcts):.2%}")
    print(f"   Mean drop: {np.mean(all_drop_pcts):.2%}")
    print(f"   Median drop: {np.median(all_drop_pcts):.2%}")

    print(f"\n   Avg bars to drop: {np.mean(all_bars):.1f}")

    print(f"\n   Drops >= 2%: {sum(1 for d in all_drop_pcts if d >= 0.02)}")
    print(f"   Drops >= 3%: {sum(1 for d in all_drop_pcts if d >= 0.03)}")
    print(f"   Drops >= 5%: {sum(1 for d in all_drop_pcts if d >= 0.05)}")

    print(f"\n   Drops in 1-3 bars: {sum(1 for b in all_bars if b <= 3)}")
    print(f"   Drops in 4-5 bars: {sum(1 for b in all_bars if 4 <= b <= 5)}")
