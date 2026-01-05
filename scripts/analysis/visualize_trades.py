#!/usr/bin/env python3
"""
Visualize Trade Results from Rolling Walk-Forward Test
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

# Trade data extracted from the log
trades = [
    # Window 0/2 - LINKUSDT
    {"date": "2025-03-05", "symbol": "LINKUSDT-30m", "type": "SHORT", "pnl": 3.98, "r": 0.34, "win": True, "at_flat": True},
    {"date": "2025-03-05", "symbol": "LINKUSDT-30m", "type": "SHORT", "pnl": -0.45, "r": -0.02, "win": False, "at_flat": True},

    # Window 13 - HYPEUSDT
    {"date": "2025-06-03", "symbol": "HYPEUSDT-15m", "type": "SHORT", "pnl": -40.86, "r": -1.17, "win": False, "at_flat": True},

    # Window 17 - HYPEUSDT (17 trades)
    {"date": "2025-06-29", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 1.32, "r": 0.23, "win": True, "at_flat": True},
    {"date": "2025-06-29", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -0.90, "r": -0.08, "win": False, "at_flat": True},
    {"date": "2025-06-30", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 1.18, "r": 0.21, "win": True, "at_flat": True},
    {"date": "2025-06-30", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 0.34, "r": 0.03, "win": True, "at_flat": True},
    {"date": "2025-06-30", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 2.28, "r": 0.40, "win": True, "at_flat": True},
    {"date": "2025-06-30", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -0.28, "r": -0.02, "win": False, "at_flat": True},
    {"date": "2025-07-03", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -21.93, "r": -1.27, "win": False, "at_flat": False},
    {"date": "2025-07-03", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 2.50, "r": 0.45, "win": True, "at_flat": False},
    {"date": "2025-07-03", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -0.55, "r": -0.05, "win": False, "at_flat": False},
    {"date": "2025-07-03", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 4.55, "r": 0.81, "win": True, "at_flat": True},
    {"date": "2025-07-03", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -0.47, "r": -0.04, "win": False, "at_flat": True},
    {"date": "2025-07-03", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 2.40, "r": 0.43, "win": True, "at_flat": True},
    {"date": "2025-07-03", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -0.42, "r": -0.04, "win": False, "at_flat": True},
    {"date": "2025-07-05", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -0.03, "r": -0.01, "win": False, "at_flat": True},
    {"date": "2025-07-05", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -1.46, "r": -0.13, "win": False, "at_flat": True},
    {"date": "2025-07-05", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 0.45, "r": 0.08, "win": True, "at_flat": True},
    {"date": "2025-07-05", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -1.55, "r": -0.14, "win": False, "at_flat": True},

    # Window 18 - HYPEUSDT
    {"date": "2025-07-09", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -48.49, "r": -1.42, "win": False, "at_flat": True},
    {"date": "2025-07-10", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -39.21, "r": -1.18, "win": False, "at_flat": False},

    # Window 20 - HYPEUSDT
    {"date": "2025-07-26", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 5.98, "r": 0.56, "win": True, "at_flat": True},
    {"date": "2025-07-26", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -1.90, "r": -0.09, "win": False, "at_flat": True},

    # Window 27 - HYPEUSDT
    {"date": "2025-09-07", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -40.74, "r": -1.25, "win": False, "at_flat": True},
    {"date": "2025-09-11", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 11.46, "r": 1.09, "win": True, "at_flat": True},
    {"date": "2025-09-11", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 15.73, "r": 1.47, "win": True, "at_flat": True},
    {"date": "2025-09-11", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 0.79, "r": 0.07, "win": True, "at_flat": True},
    {"date": "2025-09-12", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 10.27, "r": 0.96, "win": True, "at_flat": False},
    {"date": "2025-09-12", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": 1.11, "r": 0.05, "win": True, "at_flat": False},

    # Window 28 - HYPEUSDT
    {"date": "2025-09-18", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -39.47, "r": -1.21, "win": False, "at_flat": True},

    # Window 30 - HYPEUSDT
    {"date": "2025-09-29", "symbol": "HYPEUSDT-30m", "type": "SHORT", "pnl": -55.64, "r": -1.74, "win": False, "at_flat": True},
]

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle('SSL Flow Strategy - Full Year 2025 Trade Analysis\n(v1.8.2 P5.2 Filter Relaxations)', fontsize=14, fontweight='bold')

# 1. Equity Curve (top left)
ax1 = fig.add_subplot(2, 2, 1)
cumulative_pnl = np.cumsum([t["pnl"] for t in trades])
dates = [datetime.strptime(t["date"], "%Y-%m-%d") for t in trades]
colors = ['green' if t["win"] else 'red' for t in trades]

ax1.plot(range(len(cumulative_pnl)), cumulative_pnl, 'b-', linewidth=2, label='Cumulative PnL')
ax1.fill_between(range(len(cumulative_pnl)), cumulative_pnl, alpha=0.3,
                  color='green' if cumulative_pnl[-1] > 0 else 'red')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.scatter(range(len(cumulative_pnl)), cumulative_pnl, c=colors, s=50, zorder=5)
ax1.set_xlabel('Trade Number')
ax1.set_ylabel('Cumulative PnL ($)')
ax1.set_title(f'Equity Curve: ${cumulative_pnl[-1]:.2f}')
ax1.grid(True, alpha=0.3)

# 2. Individual Trade PnL (top right)
ax2 = fig.add_subplot(2, 2, 2)
pnls = [t["pnl"] for t in trades]
bar_colors = ['green' if p > 0 else 'red' for p in pnls]
ax2.bar(range(len(pnls)), pnls, color=bar_colors, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax2.set_xlabel('Trade Number')
ax2.set_ylabel('PnL ($)')
ax2.set_title('Individual Trade PnL')
ax2.grid(True, alpha=0.3, axis='y')

# Annotate worst trades
worst_indices = sorted(range(len(pnls)), key=lambda i: pnls[i])[:3]
for idx in worst_indices:
    ax2.annotate(f'${pnls[idx]:.0f}', xy=(idx, pnls[idx]),
                 xytext=(idx, pnls[idx]-10), fontsize=8, color='red',
                 ha='center')

# 3. R-Multiple Distribution (bottom left)
ax3 = fig.add_subplot(2, 2, 3)
r_multiples = [t["r"] for t in trades]
r_colors = ['green' if r > 0 else 'red' for r in r_multiples]
ax3.bar(range(len(r_multiples)), r_multiples, color=r_colors, alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax3.axhline(y=np.mean(r_multiples), color='blue', linestyle='--', alpha=0.7,
            label=f'Avg R: {np.mean(r_multiples):.2f}')
ax3.set_xlabel('Trade Number')
ax3.set_ylabel('R-Multiple')
ax3.set_title(f'R-Multiple Distribution (E[R]: {np.mean(r_multiples):.2f})')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Summary Stats (bottom right)
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

# Calculate stats
total_trades = len(trades)
wins = sum(1 for t in trades if t["win"])
losses = total_trades - wins
win_rate = (wins / total_trades) * 100
total_pnl = sum(t["pnl"] for t in trades)
avg_win = np.mean([t["pnl"] for t in trades if t["win"]])
avg_loss = np.mean([t["pnl"] for t in trades if not t["win"]])
profit_factor = abs(sum(t["pnl"] for t in trades if t["win"]) / sum(t["pnl"] for t in trades if not t["win"]))
at_flat_trades = sum(1 for t in trades if t["at_flat"])
best_trade = max(trades, key=lambda t: t["pnl"])
worst_trade = min(trades, key=lambda t: t["pnl"])

stats_text = f"""
╔══════════════════════════════════════════════════╗
║              TRADE STATISTICS                    ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  Total Trades:     {total_trades:>5}                         ║
║  Wins:             {wins:>5} ({win_rate:.1f}%)                    ║
║  Losses:           {losses:>5} ({100-win_rate:.1f}%)                    ║
║                                                  ║
║  Total PnL:        ${total_pnl:>7.2f}                     ║
║  Average Win:      ${avg_win:>7.2f}                     ║
║  Average Loss:     ${avg_loss:>7.2f}                     ║
║  Profit Factor:    {profit_factor:>7.2f}                      ║
║                                                  ║
║  Best Trade:       ${best_trade['pnl']:>7.2f} (R:{best_trade['r']:+.2f})         ║
║  Worst Trade:      ${worst_trade['pnl']:>7.2f} (R:{worst_trade['r']:+.2f})         ║
║                                                  ║
║  E[R]:             {np.mean(r_multiples):>7.2f}                      ║
║  Sharpe (R):       {np.mean(r_multiples)/np.std(r_multiples):.2f}                         ║
║                                                  ║
╠══════════════════════════════════════════════════╣
║  ⚠️  AT Flat Trades: {at_flat_trades}/{total_trades} ({at_flat_trades/total_trades*100:.0f}%)                 ║
║  (P5.2 skip enabled - these would be blocked)    ║
╚══════════════════════════════════════════════════╝
"""

ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='center', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/trade_charts/full_year_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('/Users/emreoksuz/desktop_bot_refactored_v2_base_v7/trade_charts/full_year_analysis.pdf', bbox_inches='tight')
print("Charts saved to trade_charts/full_year_analysis.png")

# Show the chart
plt.show()
