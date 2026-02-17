@echo off
set "VENV_NAME=fastreid_env"

echo Checking for python...
python --version
if %errorlevel% neq 0 (
    echo Python is not found in PATH. Please install Python.
    pause
    exit /b 1
)

if not exist "%VENV_NAME%" (
    echo Creating virtual environment %VENV_NAME%...
    python -m venv %VENV_NAME%
)

echo Activating environment...
call %VENV_NAME%\Scripts\activate

echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Installing Dependencies...
pip install opencv-python faiss-cpu yacs termcolor tabulate cloudpickle tqdm wheel scikit-learn tensorboard

echo Setting PYTHONPATH...
set PYTHONPATH=%CD%\fast-reid;%PYTHONPATH%

echo Starting Training...
python fast-reid/tools/train_net.py --config-file custom_configs/bagtricks_R50-ibn.yml OUTPUT_DIR logs/market_r50_ibn

pause
