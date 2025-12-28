# Trade Visualizer - Quick Start Guide

Professional trade chart visualization for Rolling Walk-Forward backtests.

## 60-Second Quick Start

```bash
# 1. Activate virtual environment
cd /Users/emreoksuz/desktop_bot_refactored_v2_base_v7
source venv/bin/activate

# 2. Run demo (generates sample charts)
python demo_trade_visualizer.py

# 3. View charts
open demo_charts/

# 4. Visualize your own trades
python run_trade_visualizer.py \
    --file data/rolling_wf_runs/YOUR_RUN/trades_detailed.txt \
    --all
```

## Common Use Cases

### Analyze a Failed Trade
```bash
# Interactive browser mode
python run_trade_visualizer.py \
    --file data/rolling_wf_runs/.../trades_detailed.txt \
    --browse

# Then select the trade number when prompted
```

### Generate Report Charts
```bash
# High-resolution charts for all trades
python run_trade_visualizer.py \
    --file trades_detailed.txt \
    --all \
    --dpi 600 \
    --output report_charts/
```

### Compare Winning vs Losing Trades
```bash
# Visualize first 10 trades
python run_trade_visualizer.py \
    --file trades_detailed.txt \
    --browse

# Manually compare chart patterns
```

## What You Get

Each chart includes:
- **Candlesticks** with professional dark theme
- **SSL Baseline** (HMA 60) - trend line
- **PBEMA Cloud** (EMA 200) - profit target zone
- **RSI(14)** - momentum indicator
- **ADX(14)** - trend strength
- **Volume** - colored by direction
- **Entry/Exit markers** - clear visual points
- **TP/SL levels** - horizontal lines
- **Trade info panel** - key metrics

## Files Created

| File | Purpose |
|------|---------|
| `core/trade_visualizer.py` | Visualization engine (650 lines) |
| `run_trade_visualizer.py` | CLI runner (350 lines) |
| `demo_trade_visualizer.py` | Demo script (250 lines) |
| `TRADE_VISUALIZER_README.md` | Full documentation |
| `TRADE_VISUALIZER_SUMMARY.md` | Implementation details |

## Need Help?

- **Full docs:** `TRADE_VISUALIZER_README.md`
- **Implementation:** `TRADE_VISUALIZER_SUMMARY.md`
- **Command help:** `python run_trade_visualizer.py --help`

## Example Output

**Chart:** `BTCUSDT_15m_20250813T2130_SHORT_LOSS.png`
- **Size:** 381 KB (300 DPI)
- **Dimensions:** 20 x 12 inches
- **Format:** PNG with transparency
- **Quality:** Print-ready

**Statistics from demo run:**
- Total Trades: 24
- Win Rate: 91.7%
- Total PnL: $145.37
- Charts Generated: 24/24 (100%)

---

**Ready to use!** No additional setup required.
