@echo off
REM ================================================================
REM Gold Trading System - View Performance Dashboard
REM ================================================================
REM Generate and open HTML dashboard
REM Version: 1.0.0
REM ================================================================

echo.
echo ========================================================================
echo   GOLD TRADING SYSTEM - PERFORMANCE DASHBOARD
echo ========================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

echo [INFO] Generating dashboard...
echo.

REM Generate dashboard
python performance_dashboard.py

REM Open in browser if generation successful
if not errorlevel 1 (
    echo.
    echo [INFO] Opening dashboard in browser...
    start dashboard\dashboard.html
)

echo.
echo ========================================================================
echo   DASHBOARD READY
echo ========================================================================
echo.
echo Dashboard will auto-refresh every 60 seconds
echo.
pause
