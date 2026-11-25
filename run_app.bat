@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting WatermarkRM + SeedVR2 Super-Resolution Web UI
echo ========================================================
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
call .\venv\Scripts\python.exe app.py
pause
