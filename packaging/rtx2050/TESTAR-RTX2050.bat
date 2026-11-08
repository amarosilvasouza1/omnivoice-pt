@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Diagnostico da RTX 2050 Laptop
echo ============================================================
echo.
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo Driver NVIDIA nao encontrado.
  echo Instale o driver mais recente em https://www.nvidia.com/Download/index.aspx
  pause
  exit /b 1
)

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo.
echo O nome esperado e NVIDIA GeForce RTX 2050.
echo Esta edicao foi compilada exclusivamente para CUDA sm_86.
echo Se a placa estiver correta, execute GERAR-RTX2050.bat.
pause
