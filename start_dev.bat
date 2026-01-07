@echo off
echo ============================================================
echo    AI TRADING SYSTEM - DEVELOPMENT SERVER
echo ============================================================
echo.
echo Starting development server with hot reload...
echo.
cd /d "%~dp0"
python trading_server.py --dev
pause
