@echo off
set "VENV_NAME=fastreid_env"

echo ==========================================
echo      FastReID Training Setup & Run
echo ==========================================

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.7 or 3.8.
    pause
    exit /b 1
)

:: 2. Create Virtual Environment
if not exist "%VENV_NAME%" (
    echo [INFO] Creating virtual environment '%VENV_NAME%'...
    python -m venv %VENV_NAME%
) else (
    echo [INFO] Virtual environment '%VENV_NAME%' already exists.
)

:: 3. Activate Environment
echo [INFO] Activating environment...
call %VENV_NAME%\Scripts\activate

:: 4. Install Core Dependencies
echo [INFO] Checking for PyTorch...
python -c "import torch; print(torch.__version__)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing PyTorch (CUDA 11.8)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [INFO] PyTorch already installed. Skipping.
)

echo [INFO] Installing FastReID dependencies...
pip install -r fast-reid/requirements.txt
pip install opencv-python faiss-cpu tensorboard

:: 5. Install FastReID in Editable Mode
echo [INFO] Installing FastReID package...
python -c "import fastreid" >nul 2>&1
if %errorlevel% neq 0 (
    pip install -e fast-reid
) else (
    echo [INFO] FastReID already installed. Skipping.
)

:: 6. Launch TensorBoard (Background)
echo [INFO] Starting TensorBoard for visualization...
echo [INFO] Access TensorBoard at http://localhost:6006
start http://localhost:6006
start cmd /c "%VENV_NAME%\Scripts\tensorboard.exe --logdir logs/market_r50_ibn --port 6006"

:: 7. Start Training
echo [INFO] Starting Training...
echo [INFO] Logs will be saved to: logs/market_r50_ibn
python fast-reid/tools/train_net.py --config-file custom_configs/bagtricks_R50-ibn.yml OUTPUT_DIR logs/market_r50_ibn

echo [INFO] Training Completed.
