@echo off
title Autonomous AI Trading System
color 0A
echo.
echo ============================================================
echo    AUTONOMOUS AI TRADING SYSTEM
echo    Version 1.0 - Full Module Suite
echo ============================================================
echo.
echo    Modules Loaded:
echo    - Error Analyzer
echo    - Self Corrector  
echo    - Meta Learner (MAML)
echo    - Curiosity Module
echo    - Strategy Evolution
echo    - Position Optimizer (Kelly)
echo    - Auto Trainer
echo    - Shadow Trader
echo    - Knowledge Base
echo    - Evolution Engine
echo.
echo ============================================================
echo.

cd /d "%~dp0"

echo Select Mode:
echo   [1] Paper Trading (Recommended for testing)
echo   [2] Live Trading (Real money - BE CAREFUL!)
echo   [3] Run Single Check (No loop)
echo   [4] System Status
echo   [5] Exit
echo.

set /p choice=Enter choice (1-5): 

if "%choice%"=="1" goto paper
if "%choice%"=="2" goto live
if "%choice%"=="3" goto single
if "%choice%"=="4" goto status
if "%choice%"=="5" goto end

:paper
echo.
echo Starting Paper Trading Mode...
echo Press Ctrl+C to stop
echo.
python -c "from ai_agent.autonomous_mt5 import create_autonomous_mt5; trader = create_autonomous_mt5(paper_trading=True); trader.run(interval_seconds=120)"
goto end

:live
echo.
echo WARNING: Live trading will use REAL MONEY!
set /p confirm=Type 'YES' to confirm: 
if not "%confirm%"=="YES" goto end
echo.
echo Starting Live Trading...
python -c "from ai_agent.autonomous_mt5 import create_autonomous_mt5; trader = create_autonomous_mt5(paper_trading=False); trader.run(interval_seconds=120)"
goto end

:single
echo.
echo Running single market check...
python -c "from ai_agent.autonomous_mt5 import create_autonomous_mt5; trader = create_autonomous_mt5(paper_trading=True); result = trader.check_and_execute(); print(f'Decision: {result}')"
goto end

:status
echo.
echo Checking system status...
python -c "from ai_agent.autonomous_ai import create_autonomous_ai; ai = create_autonomous_ai(); print('System OK')"
goto end

:end
echo.
pause
