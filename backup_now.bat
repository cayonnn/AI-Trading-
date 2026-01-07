@echo off
REM ================================================================
REM Gold Trading System - Create Backup Now
REM ================================================================
REM Immediate backup of database and configuration
REM Version: 1.0.0
REM ================================================================

echo.
echo ========================================================================
echo   GOLD TRADING SYSTEM - CREATE BACKUP
echo ========================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

echo [INFO] Creating full system backup...
echo.

REM Create backup
python automated_backup.py --backup

if not errorlevel 1 (
    echo.
    echo ========================================================================
    echo   BACKUP COMPLETE!
    echo ========================================================================
    echo.
    echo Backup files saved to: backups\
    echo.
) else (
    echo.
    echo [ERROR] Backup failed! Check logs for details.
    echo.
)

pause
