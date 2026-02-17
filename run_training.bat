@echo off
set "ENV_NAME=fastreid"

echo Checking Conda Environment...
call conda env list | findstr %ENV_NAME%
if %errorlevel% neq 0 (
    echo Creating conda environment %ENV_NAME%...
    call conda create -n %ENV_NAME% python=3.8 -y
)

echo Activating environment...
call conda activate %ENV_NAME%

echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Installing Dependencies...
pip install opencv-python faiss-cpu yacs termcolor tabulate cloudpickle tqdm wheel scikit-learn tensorboard

echo Setting PYTHONPATH...
set PYTHONPATH=%CD%\fast-reid;%PYTHONPATH%

echo Starting Training...
python fast-reid/tools/train_net.py --config-file custom_configs/bagtricks_R50-ibn.yml OUTPUT_DIR logs/market_r50_ibn

pause
