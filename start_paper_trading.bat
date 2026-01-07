@echo off
REM ================================================================
REM Gold Trading System - Start Paper Trading
REM ================================================================
REM Production-Grade Quick Start Script
REM Version: 1.0.0
REM ================================================================

echo.
echo ========================================================================
echo   GOLD TRADING SYSTEM - PAPER TRADING MODE
echo ========================================================================
echo.
echo   MODE: Paper Trading (NO REAL MONEY AT RISK)
echo   INTERVAL: Every 5 minutes
echo   DATABASE: trading_data.db
echo   LOGS: logs\production_mt5_*.log
echo.
echo ========================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.9+
    pause
    exit /b 1
)

REM Check if MT5 Python module is installed
python -c "import MetaTrader5" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] MetaTrader5 module not found!
    echo Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Create logs directory if not exists
if not exist logs mkdir logs

REM Display configuration
echo [INFO] Starting paper trading system...
echo [INFO] Press Ctrl+C to stop
echo.

REM Start paper trading
python production_system_mt5.py --paper-trading --interval 300

REM If script exits
echo.
echo ========================================================================
echo   PAPER TRADING STOPPED
echo ========================================================================
echo.
echo Check logs\production_mt5_*.log for details
echo.
pause
