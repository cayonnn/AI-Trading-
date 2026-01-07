@echo off
echo ============================================================
echo    AI TRADING SYSTEM - PRODUCTION SERVER
echo ============================================================
echo.
echo Starting production server...
echo.
cd /d "%~dp0"
python trading_server.py --threads 4
pause
