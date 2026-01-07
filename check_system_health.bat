@echo off
REM ================================================================
REM Gold Trading System - System Health Check
REM ================================================================
REM Automated Health Monitoring
REM Version: 1.0.0
REM ================================================================

echo.
echo ========================================================================
echo   GOLD TRADING SYSTEM - SYSTEM HEALTH CHECK
echo ========================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

echo [INFO] Running comprehensive health check...
echo.

REM Run health monitor
python system_health_monitor.py

echo.
echo ========================================================================
echo   HEALTH CHECK COMPLETE
echo ========================================================================
echo.
echo Check health_report_*.txt for detailed report
echo.
pause
