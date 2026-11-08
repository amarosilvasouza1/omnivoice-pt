@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   OmniVoice-PT - AMD RX 580 (Vulkan)
echo ============================================================
echo.
echo Modelo:
echo   1 - Q8    (recomendado, menor uso de VRAM)
echo   2 - FP16  (maior qualidade e uso de VRAM)
set /p OP=Escolha [1]: 
if "%OP%"=="2" (
  set "MODEL=modelos\omnivoice-pt-f16.gguf"
) else (
  set "MODEL=modelos\omnivoice-pt-q8_0.gguf"
)

echo.
echo Perfis disponíveis em vozes\
dir /b "vozes\*.ovprompt" 2>nul
echo.
set /p VOZ=Nome do perfil sem extensao (Enter = voz automatica): 
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

omnivoice-pt.exe -m "%MODEL%" -d "modelos\higgs-decoder-f16.gguf" -k "modelos\tokenizer.bin" %PROMPT% -t "%TEXTO%" -o "saida.wav" --gpu --steps 32
if errorlevel 1 (
  echo.
  echo Falha. Atualize o driver AMD Adrenalin com suporte Vulkan.
) else (
  echo.
  echo Pronto: %CD%\saida.wav
)
pause
