@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   OmniVoice-PT - RTX 4060 (CUDA sm_89)
echo ============================================================
echo.
echo   1 - Q8   (recomendado: rapido e leve)
echo   2 - FP16 (qualidade maxima; usa mais VRAM)
set /p Q=Modelo [1]:
if "%Q%"=="2" (
  set "MODELO=modelos\omnivoice-pt-f16.gguf"
) else (
  set "MODELO=modelos\omnivoice-pt-q8_0.gguf"
)

echo.
echo Vozes clonadas disponiveis:
dir /b "vozes\*.ovprompt" 2>nul
set /p VOZ=Nome da voz sem .ovprompt (Enter = automatica):
set "PROMPT="
if not "%VOZ%"=="" (
  if not exist "vozes\%VOZ%.ovprompt" (
    echo Perfil nao encontrado.
    pause
    exit /b 1
  )
  set "PROMPT=-p vozes\%VOZ%.ovprompt"
)

set /p TEXTO=Texto:
if "%TEXTO%"=="" exit /b 1
set "SAIDA=saida_%RANDOM%.wav"
omnivoice-pt.exe -m "%MODELO%" -d "modelos\higgs-decoder-f16.gguf" -k "modelos\tokenizer.bin" %PROMPT% -t "%TEXTO%" -o "%SAIDA%" --gpu --steps 32
if exist "%SAIDA%" (
  echo Pronto: %SAIDA%
  start "" "%SAIDA%"
) else (
  echo Falha na geracao. Rode TESTAR-RTX4060.bat e atualize o driver NVIDIA.
)
pause
