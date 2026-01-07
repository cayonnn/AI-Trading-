@echo off
title AI System Check
cd /d "%~dp0"
echo.
echo Testing all AI modules...
echo.

python -c "print('Testing modules...'); from ai_agent.autonomous_ai import create_autonomous_ai; ai = create_autonomous_ai(); status = ai.get_status(); print('All modules OK'); print(f'Active Version: {status[\"modules\"][\"evolution\"].get(\"active_version\", \"default\")}')"

echo.
pause
