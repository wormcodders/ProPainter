@echo off
echo ===================================================
echo FIXING PYTORCH FOR NVIDIA GPU (CUDA)
echo ===================================================
echo.
echo Uninstalling CPU versions...
call .\venv\Scripts\pip uninstall -y torch torchvision torchaudio

echo.
echo Installing CUDA 11.8 versions (Optimized for RTX)...
call .\venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

echo.
echo Checking GPU Availability:
call .\venv\Scripts\python -c "import torch; print('\n[RESULT] GPU is available!' if torch.cuda.is_available() else '\n[RESULT] FAILED: GPU is still not seen by PyTorch. Check your NVIDIA drivers.')"

echo.
pause
