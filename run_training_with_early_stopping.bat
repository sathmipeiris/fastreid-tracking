@echo off
REM Training script with early stopping validation
REM Monitors mAP every epoch and stops if no improvement for N epochs

cd /d "%~dp0"

echo.
echo ============================================================
echo    ReID Training with Early Stopping and Overfitting Detection
echo ============================================================
echo.

REM Activate virtual environment
call fastreid_env\Scripts\activate.bat

REM Set environment variables
set PYTHONPATH=%cd%\fast-reid;%PYTHONPATH%
set FASTREID_DATASETS=fast-reid/datasets

REM Run training with early stopping
echo Starting training with validation every epoch...
echo Early stopping will trigger if mAP doesn't improve for 10 epochs
echo.

python train_with_early_stopping.py ^
    --config-file custom_configs/bagtricks_R50-ibn.yml ^
    OUTPUT_DIR logs/market1501/bagtricks_R50-ibn ^
    SOLVER.EARLY_STOP_PATIENCE 10

echo.
echo ============================================================
echo    Training completed
echo ============================================================
echo.
echo Analyzing validation history...
python analyze_training.py logs/market1501/bagtricks_R50-ibn/validation_history.json

echo.
echo To visualize plots, run:
echo   python analyze_training.py logs/market1501/bagtricks_R50-ibn/validation_history.json --plot
echo.

pause
