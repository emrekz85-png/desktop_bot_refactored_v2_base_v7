#!/bin/bash
# ==========================================
# Trading Bot Starter Script (Linux/Mac)
# ==========================================
# Bu script trading bot'u hizlica baslatir.
# Kullanim: ./start_bot.sh
# ==========================================

echo ""
echo "=========================================="
echo "   Trading Bot v39.0 - R-Multiple Based"
echo "=========================================="
echo ""

# Script'in bulundugu klasore git
cd "$(dirname "$0")"

# Python kontrolu
if ! command -v python3 &> /dev/null; then
    echo "[HATA] Python3 bulunamadi!"
    echo "Lutfen Python3 yukleyin."
    exit 1
fi

# Data klasoru olustur
mkdir -p data

echo "Bot baslatiliyor..."
echo "(Kapatmak icin Ctrl+C basin)"
echo ""

# GUI modunda baslat
python3 desktop_bot_refactored_v2_base_v7.py
