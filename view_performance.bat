@echo off
REM ================================================================
REM Gold Trading System - View Performance Report
REM ================================================================
REM Quick Performance Analysis
REM Version: 1.0.0
REM ================================================================

echo.
echo ========================================================================
echo   GOLD TRADING SYSTEM - PERFORMANCE REPORT
echo ========================================================================
echo.

REM Check if database exists
if not exist trading_data.db (
    echo [ERROR] No trading data found!
    echo Please run paper trading first to generate data
    pause
    exit /b 1
)

echo [INFO] Generating performance report...
echo.

REM Generate and display report
python performance_analytics.py

echo.
echo ========================================================================
echo   REPORT COMPLETE
echo ========================================================================
echo.
echo Check performance_report_*.txt for saved report
echo.
pause
