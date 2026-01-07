@echo off
title AI Quick Start - Paper Trading
cd /d "%~dp0"
echo Starting Paper Trading (Safe Mode)...
python -c "from ai_agent.autonomous_mt5 import create_autonomous_mt5; t = create_autonomous_mt5(paper_trading=True); t.run(interval_seconds=60, max_iterations=10)"
pause
