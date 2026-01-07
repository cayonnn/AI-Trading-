@echo off
REM ================================================================
REM Gold Trading System - Schedule Daily Backup
REM ================================================================
REM Setup Windows Task Scheduler for daily backups
REM Version: 1.0.0
REM ================================================================

echo.
echo ========================================================================
echo   GOLD TRADING SYSTEM - SCHEDULE DAILY BACKUP
echo ========================================================================
echo.

echo This script will create a Windows Task Scheduler task to run
echo backups automatically every day at 2:00 AM.
echo.

pause

echo.
echo [INFO] Creating scheduled task...
echo.

REM Get current directory
set SCRIPT_DIR=%~dp0
set BACKUP_SCRIPT=%SCRIPT_DIR%backup_now.bat

REM Create scheduled task
schtasks /create /tn "GoldTradingSystem_DailyBackup" /tr "\"%BACKUP_SCRIPT%\"" /sc daily /st 02:00 /f

if not errorlevel 1 (
    echo.
    echo ========================================================================
    echo   SCHEDULED TASK CREATED SUCCESSFULLY!
    echo ========================================================================
    echo.
    echo Task Name: GoldTradingSystem_DailyBackup
    echo Schedule: Daily at 2:00 AM
    echo Script: %BACKUP_SCRIPT%
    echo.
    echo To view/manage the task:
    echo   - Open Task Scheduler (taskschd.msc)
    echo   - Look for "GoldTradingSystem_DailyBackup"
    echo.
    echo To disable the task:
    echo   schtasks /change /tn "GoldTradingSystem_DailyBackup" /disable
    echo.
    echo To delete the task:
    echo   schtasks /delete /tn "GoldTradingSystem_DailyBackup" /f
    echo.
) else (
    echo.
    echo [ERROR] Failed to create scheduled task!
    echo Please run this script as Administrator.
    echo.
)

pause
