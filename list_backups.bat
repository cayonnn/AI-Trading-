@echo off
REM ================================================================
REM Gold Trading System - List Available Backups
REM ================================================================
REM Show all available backup files
REM Version: 1.0.0
REM ================================================================

echo.
echo ========================================================================
echo   GOLD TRADING SYSTEM - AVAILABLE BACKUPS
echo ========================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

REM List backups
python automated_backup.py --list

echo.
pause
