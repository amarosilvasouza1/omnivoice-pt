@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   OmniVoice-PT - AMD Radeon RX 7600 (Vulkan)
echo ============================================================
echo.
echo Modelo:
echo   1 - Q8    (recomendado, mais rapido e leve)
echo   2 - FP16  (qualidade maxima; usa mais VRAM)
set /p OP=Escolha [1]:
if "%OP%"=="2" (
  set "MODEL=modelos\omnivoice-pt-f16.gguf"
) else (
  set "MODEL=modelos\omnivoice-pt-q8_0.gguf"
)

echo.
echo Vozes disponiveis em vozes\:
dir /b "vozes\*.ovprompt" 2>nul
echo.
set /p VOZ=Nome da voz sem .ovprompt (Enter = automatica):
set "PROMPT="
if not "%VOZ%"=="" (
  if not exist "vozes\%VOZ%.ovprompt" (
    echo ERRO: voz "vozes\%VOZ%.ovprompt" nao encontrada.
    pause
    exit /b 1
  )
  set "PROMPT=-p vozes\%VOZ%.ovprompt"
)

set /p TEXTO=Texto:
if "%TEXTO%"=="" exit /b 1
set "SAIDA=saida_%RANDOM%.wav"
omnivoice-pt.exe -m "%MODEL%" -d "modelos\higgs-decoder-f16.gguf" -k "modelos\tokenizer.bin" %PROMPT% -t "%TEXTO%" -o "%SAIDA%" --gpu --steps 32
if exist "%SAIDA%" (
  echo Pronto: %SAIDA%
  start "" "%SAIDA%"
) else (
  echo Falha. Atualize o driver AMD Adrenalin com suporte Vulkan e rode TESTAR-RX7600.bat.
)
pause
