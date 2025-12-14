#!/bin/bash
# ==========================================
# Trading Bot Backtest Script (Linux/Mac)
# ==========================================
# Bu script backtest'i hizlica calistirir.
# Sonuclar data/ klasorune kaydedilir.
# ==========================================

echo ""
echo "=========================================="
echo "   Trading Bot - Backtest Mode"
echo "=========================================="
echo ""

# Script'in bulundugu klasore git
cd "$(dirname "$0")"

# Python kontrolu
if ! command -v python3 &> /dev/null; then
    echo "[HATA] Python3 bulunamadi!"
    exit 1
fi

# Data klasoru olustur
mkdir -p data

echo "Backtest baslatiliyor..."
echo "(Bu islem 5-15 dakika surebilir)"
echo ""

# Headless modda backtest calistir
python3 desktop_bot_refactored_v2_base_v7.py --headless

echo ""
echo "=========================================="
echo "Backtest tamamlandi!"
echo "Sonuclar data/ klasorunde:"
echo "  - data/backtest_trades.csv"
echo "  - data/backtest_summary.csv"
echo "  - data/best_configs.json"
echo "=========================================="
