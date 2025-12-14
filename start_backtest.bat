@echo off
REM ==========================================
REM Trading Bot Backtest Script (Windows)
REM ==========================================
REM Bu script backtest'i hizlica calistirir.
REM Sonuclar data/ klasorune kaydedilir.
REM ==========================================

title Trading Bot - Backtest
cd /d "%~dp0"

echo.
echo ==========================================
echo    Trading Bot - Backtest Mode
echo ==========================================
echo.

REM Python kontrolu
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [HATA] Python bulunamadi!
    pause
    exit /b 1
)

REM Data klasoru olustur
if not exist "data" mkdir data

echo Backtest baslatiliyor...
echo (Bu islem 5-15 dakika surebilir)
echo.

REM Headless modda backtest calistir
python desktop_bot_refactored_v2_base_v7.py --headless

echo.
echo ==========================================
echo Backtest tamamlandi!
echo Sonuclar data/ klasorunde:
echo   - data/backtest_trades.csv
echo   - data/backtest_summary.csv
echo   - data/best_configs.json
echo ==========================================
pause
