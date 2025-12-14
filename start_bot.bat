@echo off
REM ==========================================
REM Trading Bot Starter Script (Windows)
REM ==========================================
REM Bu script trading bot'u hizlica baslatir.
REM Cift tiklayarak veya terminalden calistirabilirsiniz.
REM ==========================================

title Trading Bot v39.0
cd /d "%~dp0"

echo.
echo ==========================================
echo    Trading Bot v39.0 - R-Multiple Based
echo ==========================================
echo.

REM Hizli baslangic - sadece ilk kurulumda kontrol yap
if exist "data\.setup_complete" goto :start_bot

REM Ilk kurulum - Python kontrolu
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [HATA] Python bulunamadi!
    echo Lutfen Python'u yukleyin: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Ilk kurulum - kutuphaneleri kontrol et
echo Ilk kurulum: Kutuphaneler kontrol ediliyor...
python -c "import pandas, numpy, requests, PyQt5" 2>nul
if %errorlevel% neq 0 (
    echo [UYARI] Bazi kutuphaneler eksik. Yukluyor...
    pip install pandas numpy requests pandas_ta plotly PyQt5 PyQtWebEngine matplotlib tqdm
)

REM Data klasoru olustur ve kurulum tamamlandi isareti birak
if not exist "data" mkdir data
echo. > "data\.setup_complete"
echo Kurulum tamamlandi! Sonraki baslangiclar daha hizli olacak.

:start_bot
echo.
echo Bot baslatiliyor...

REM GUI modunda baslat
python desktop_bot_refactored_v2_base_v7.py

REM Eger hata varsa pencereyi acik tut
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Bot beklenmedik sekilde kapandi.
    pause
)
