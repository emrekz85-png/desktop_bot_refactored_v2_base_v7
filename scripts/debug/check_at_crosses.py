#!/usr/bin/env python3
"""Check AlphaTrend crossovers in real data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core import get_client, calculate_indicators, set_backtest_mode
import pandas as pd

set_backtest_mode(True)
client = get_client()

df = client.get_klines(symbol='BTCUSDT', interval='15m', limit=2000)
df = calculate_indicators(df, timeframe='15m')

print("="*70)
print("ALPHATREND CROSSOVER ANALYSIS")
print("="*70)

at_buyers = df['alphatrend'].values
at_sellers = df['alphatrend_2'].values

# Find all crossovers
crosses_down = []
crosses_up = []

for i in range(1, len(df)):
    prev_buyers_above = at_buyers[i-1] > at_sellers[i-1]
    now_buyers_below = at_buyers[i] < at_sellers[i]

    if prev_buyers_above and now_buyers_below:
        gap = (at_sellers[i] - at_buyers[i]) / at_buyers[i]
        crosses_down.append((i, gap))

    prev_sellers_above = at_sellers[i-1] > at_buyers[i-1]
    now_sellers_below = at_sellers[i] < at_buyers[i]

    if prev_sellers_above and now_sellers_below:
        gap = (at_buyers[i] - at_sellers[i]) / at_sellers[i]
        crosses_up.append((i, gap))

print(f"\n📊 Crossovers in {len(df)} bars:")
print(f"   Cross DOWN (bearish): {len(crosses_down)}")
print(f"   Cross UP (bullish): {len(crosses_up)}")

if crosses_down:
    print(f"\n📉 BEARISH CROSSES (TOP 10 BY GAP):")
    sorted_down = sorted(crosses_down, key=lambda x: x[1], reverse=True)[:10]
    for idx, (i, gap) in enumerate(sorted_down, 1):
        print(f"   {idx}. Bar #{i}: Gap = {gap:.4%}")

if crosses_up:
    print(f"\n📈 BULLISH CROSSES (TOP 10 BY GAP):")
    sorted_up = sorted(crosses_up, key=lambda x: x[1], reverse=True)[:10]
    for idx, (i, gap) in enumerate(sorted_up, 1):
        print(f"   {idx}. Bar #{i}: Gap = {gap:.4%}")

# Check gap distribution
all_gaps = [g for _, g in crosses_down] + [g for _, g in crosses_up]
if all_gaps:
    import numpy as np
    print(f"\n📐 GAP STATISTICS:")
    print(f"   Min: {min(all_gaps):.4%}")
    print(f"   Max: {max(all_gaps):.4%}")
    print(f"   Mean: {np.mean(all_gaps):.4%}")
    print(f"   Median: {np.median(all_gaps):.4%}")

    print(f"\n   Gaps > 0.3%: {sum(1 for g in all_gaps if g > 0.003)}")
    print(f"   Gaps > 0.8%: {sum(1 for g in all_gaps if g > 0.008)}")
    print(f"   Gaps > 1.5%: {sum(1 for g in all_gaps if g > 0.015)}")
