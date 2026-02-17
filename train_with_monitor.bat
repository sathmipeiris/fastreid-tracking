@echo off
REM FastReID Training with Progress Monitor
REM Run this to see beautiful real-time training progress

echo.
echo Activating FastReID Virtual Environment...
call fastreid_env\Scripts\activate.bat

echo.
echo Running Training Monitor...
python monitor_training.py

pause
