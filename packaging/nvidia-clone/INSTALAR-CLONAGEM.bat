@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Preparar clonagem de voz - executar somente uma vez
echo ============================================================
echo.
where py >nul 2>&1
if errorlevel 1 (
  echo Python nao encontrado.
  echo Instale Python 3.10 ou 3.11 de https://www.python.org/downloads/
  echo Marque a opcao "Add Python to PATH" durante a instalacao.
  pause
  exit /b 1
)

if not exist "clonagem-env\Scripts\python.exe" (
  py -3.10 -m venv clonagem-env 2>nul
  if errorlevel 1 py -3.11 -m venv clonagem-env
)
if not exist "clonagem-env\Scripts\python.exe" (
  echo Nao foi possivel criar o ambiente Python.
  pause
  exit /b 1
)

echo Instalando PyTorch CPU e dependencias. Isso pode demorar...
clonagem-env\Scripts\python.exe -m pip install --upgrade pip
clonagem-env\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
clonagem-env\Scripts\python.exe -m pip install "transformers>=5.3.0" accelerate pydub numpy soundfile librosa huggingface_hub
if errorlevel 1 (
  echo A instalacao falhou. Confira sua internet e o espaco em disco.
  pause
  exit /b 1
)

echo.
echo Pronto. Agora execute CLONAR-MINHA-VOZ.bat.
pause
