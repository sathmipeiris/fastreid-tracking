@echo off
REM Windows PyTorch Multiprocessing Fix for FastReID Training
REM This batch file disables problematic features that cause shared file mapping errors

setlocal enabledelayedexpansion

REM Set critical environment variables
set TORCH_HOME=%CD%\.torch_cache
set KMP_DUPLICATE_LIB_OK=True
set CUDA_LAUNCH_BLOCKING=1
set PYTHONPATH=%CD%\fast-reid;%PYTHONPATH%
set FASTREID_DATASETS=%CD%\fast-reid\datasets

echo.
echo ============================================================
echo Windows PyTorch Fix - Training Launcher
echo ============================================================
echo TORCH_HOME: %TORCH_HOME%
echo PYTHONPATH: %PYTHONPATH%
echo FASTREID_DATASETS: %FASTREID_DATASETS%
echo ============================================================
echo.

REM Run training with the fix script
python train_windows_fixed.py %*

pause
