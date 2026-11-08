@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist "clonagem-env\Scripts\python.exe" (
  echo Execute INSTALAR-CLONAGEM.bat primeiro.
  pause
  exit /b 1
)

echo Use um WAV limpo com 3 a 10 segundos, sem musica e sem outra voz.
set /p AUDIO=Caminho completo do arquivo WAV:
set "AUDIO=%AUDIO:"=%"
if not exist "%AUDIO%" (
  echo Arquivo nao encontrado.
  pause
  exit /b 1
)

set /p NOME=Nome curto da voz, sem espacos:
if "%NOME%"=="" exit /b 1
set /p TEXTO=Digite exatamente o que foi falado no WAV:
if "%TEXTO%"=="" (
  echo A transcricao exata e obrigatoria nesta versao enxuta.
  pause
  exit /b 1
)

if not exist vozes mkdir vozes
clonagem-env\Scripts\python.exe tools\make_prompt.py --ref_audio "%AUDIO%" --ref_text "%TEXTO%" --out "vozes\%NOME%.ovprompt"
if errorlevel 1 (
  echo Falha ao criar o perfil.
) else (
  echo.
  echo Perfil criado: vozes\%NOME%.ovprompt
  echo Agora execute GERAR-RTX4060.bat e escolha %NOME%.
)
pause
