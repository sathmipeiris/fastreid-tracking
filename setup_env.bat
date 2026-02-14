@echo off
set "ENV_NAME=fastreid"

echo Creating conda environment %ENV_NAME%...
call conda create -n %ENV_NAME% python=3.8 -y
if %errorlevel% neq 0 exit /b %errorlevel%

echo Activating environment...
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo Failed to activate conda environment. Make sure conda is in your PATH.
    exit /b %errorlevel%
)

echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 exit /b %errorlevel%

if not exist "fast-reid" (
    echo Cloning FastReID...
    git clone https://github.com/JDAI-CV/fast-reid.git
)

echo Installing FastReID dependencies...
cd fast-reid
pip install -r requirements.txt
pip install faiss-cpu
pip install opencv-python
echo Installing FastReID...
python setup.py develop

echo Setup complete!
pause
