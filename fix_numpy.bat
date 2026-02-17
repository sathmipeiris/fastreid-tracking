@echo off
REM Fix NumPy 2.x compatibility issue with fastreid_env
REM This downgrade NumPy to 1.x which is compatible with existing compiled modules

echo.
echo ============================================================================
echo NumPy Compatibility Fix
echo ============================================================================
echo.
echo Some modules were compiled with NumPy 1.x and cannot run with NumPy 2.x
echo Downgrading NumPy to version 1.x...
echo.

pip install "numpy<2" --upgrade

echo.
echo ============================================================================
echo NumPy downgrade complete!
echo ============================================================================
echo.
echo You can now run training again:
echo   .\train_windows_recommended.bat
echo.
pause
