#!/usr/bin/env python3
"""
Quick backtest runner for Keltner-PBEMA strategy analysis
"""
import sys
import os

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from desktop_bot_refactored_v2_base_v7 import run_portfolio_backtest

def main():
    # Test symbols and timeframes
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    timeframes = ["5m", "15m", "1h"]

    print("=" * 60)
    print("KELTNER-PBEMA BACKTEST")
    print("=" * 60)
    print(f"Symbols: {symbols}")
    print(f"Timeframes: {timeframes}")
    print("=" * 60)

    def progress_cb(msg):
        print(msg)

    run_portfolio_backtest(
        symbols=symbols,
        timeframes=timeframes,
        candles=2000,  # Son 2000 mum
        out_trades_csv="backtest_trades.csv",
        out_summary_csv="backtest_summary.csv",
        progress_callback=progress_cb,
        draw_trades=False,
        max_draw_trades=None
    )

    # Print summary
    if os.path.exists("backtest_summary.csv"):
        print("\n" + "=" * 60)
        print("SUMMARY FROM backtest_summary.csv:")
        print("=" * 60)
        with open("backtest_summary.csv", "r") as f:
            print(f.read())

if __name__ == "__main__":
    main()
