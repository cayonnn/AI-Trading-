@echo off
REM ================================================================
REM Gold Trading System - Run All Tests
REM ================================================================
REM Automated Testing Script
REM Version: 1.0.0
REM ================================================================

echo.
echo ========================================================================
echo   GOLD TRADING SYSTEM - TEST SUITE
echo ========================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

echo [INFO] Starting test suite...
echo.

REM Test 1: MT5 Connection
echo ========================================================================
echo TEST 1: MT5 Connection Test
echo ========================================================================
echo.
python test_mt5_connection.py
if errorlevel 1 (
    echo.
    echo [FAIL] MT5 Connection test failed!
    echo Please check MT5 is running and configured correctly
    pause
    exit /b 1
)
echo.
echo [PASS] MT5 Connection test PASSED
echo.

REM Test 2: Full Integration
echo ========================================================================
echo TEST 2: Full Integration Test
echo ========================================================================
echo.
python test_mt5_integration.py
if errorlevel 1 (
    echo.
    echo [FAIL] Integration test failed!
    pause
    exit /b 1
)
echo.
echo [PASS] Integration test PASSED
echo.

REM Summary
echo ========================================================================
echo   ALL TESTS PASSED!
echo ========================================================================
echo.
echo System is ready for paper trading!
echo.
echo Next step: Run start_paper_trading.bat
echo.
pause
