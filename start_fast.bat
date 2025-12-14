@echo off
REM ==========================================
REM Trading Bot - FAST LAUNCHER (Windows)
REM ==========================================
REM Splash screen ile hizli baslatma
REM Yukleme sirasinda gorsel geri bildirim verir
REM ==========================================

title Trading Bot v39.0 - Fast Start
cd /d "%~dp0"

REM Data klasoru olustur
if not exist "data" mkdir data

REM Hizli baslat - splash screen ile
python fast_start.py

REM Hata varsa bekle
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Bot beklenmedik sekilde kapandi.
    pause
)
