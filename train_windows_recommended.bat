@echo off
REM ============================================================================
REM WINDOWS PYTORCH FIX - RECOMMENDED TRAINING LAUNCHER
REM ============================================================================
REM This script runs FastReID training with fixes for Windows multiprocessing
REM errors (RuntimeError: Couldn't open shared file mapping)
REM ============================================================================

setlocal enabledelayedexpansion

REM Set environment variables to prevent multiprocessing errors
set TORCH_HOME=%CD%\.torch_cache
set KMP_DUPLICATE_LIB_OK=True
set CUDA_LAUNCH_BLOCKING=1
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set NUMEXPR_MAX_THREADS=1
set PYTHONPATH=%CD%\fast-reid;%PYTHONPATH%
set FASTREID_DATASETS=%CD%\fast-reid\datasets

REM Create cache directory if needed
if not exist ".torch_cache" (
    mkdir .torch_cache
)

echo.
echo ============================================================================
echo FastReID Windows Training - Multiprocessing Fix Applied
echo ============================================================================
echo.

REM Use the patched version (data_utils.py has been modified)
REM Run with explicit num_workers=0 to prevent any worker processes
python fast-reid/tools/train_net.py ^
    --config-file custom_configs/plateau_solutions/solution_5_smaller_batch_higher_lr.yml ^
    --num-gpus 1 ^
    OUTPUT_DIR logs/market1501/plateau_solver ^
    DATALOADER.NUM_WORKERS 0

echo.
echo Training completed or interrupted.
echo.
pause
