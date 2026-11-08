@echo off
chcp 65001 >nul
title OmniVoice-PT - Servico (LINK FIXO)
cd /d "%~dp0"

if not exist ngrok-dominio.txt (
  echo Voce ainda nao configurou o link fixo.
  echo Rode primeiro:  CONFIGURAR-NGROK.bat
  echo.
  pause
  exit /b
)
set /p DOM=<ngrok-dominio.txt
REM Corrige configuracoes antigas em que foi colado https:// junto do dominio.
set "DOM=%DOM:https://=%"
set "DOM=%DOM:http://=%"
for /f "tokens=1 delims=/" %%D in ("%DOM%") do set "DOM=%%D"
if "%DOM%"=="" (
  echo Dominio do ngrok invalido. Rode CONFIGURAR-NGROK.bat novamente.
  pause
  exit /b
)

echo ============================================================
echo    Iniciando OmniVoice-PT com LINK FIXO
echo    https://%DOM%
echo ============================================================
echo.

REM --- 1. servidor da API (so sobe se ainda nao estiver no ar) ---
powershell -NoProfile -Command "try{if((Invoke-WebRequest http://127.0.0.1:5180/docs -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200){exit 0}}catch{}; exit 1"
if not errorlevel 1 (
  echo Servidor ja estava rodando. Reaproveitando.
  goto tunel
)
echo Subindo o servidor e um worker ^(uns 15 segundos^)...
start "OmniVoice - Servidor (nao feche)" .venv\Scripts\python.exe tools\api_server.py
:esperar_servidor
timeout /t 3 >nul
powershell -NoProfile -Command "try{if((Invoke-WebRequest http://127.0.0.1:5180/docs -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200){exit 0}}catch{}; exit 1"
if errorlevel 1 goto esperar_servidor
echo Servidor pronto!

:tunel
echo.
echo Ligando o link fixo ^(ngrok^)...
taskkill /f /im ngrok.exe >nul 2>&1
taskkill /f /im cloudflared.exe >nul 2>&1
start "OmniVoice - Link Fixo (nao feche)" tools\bin\ngrok.exe http 5180 --url=https://%DOM%

REM a URL e fixa, entao ja salvamos direto
echo https://%DOM%>LINK-ATUAL.txt

REM espera 4s o ngrok conectar
timeout /t 4 >nul

echo.
echo ============================================================
echo    TUDO PRONTO!  Link FIXO (nunca muda):
echo ============================================================
echo.
echo    Documentacao:          https://%DOM%/docs
echo    Link para um amigo:    https://%DOM%/app?key=CHAVE_DO_AMIGO
echo    Seu painel de admin:   https://%DOM%/admin
echo.
echo    Crie chaves de amigos com  CRIAR-LINK-AMIGO.bat
echo    O token do admin esta na janela "OmniVoice - Servidor".
echo.
echo    NAO FECHE as janelas "Servidor" e "Link Fixo".
echo    Para desligar tudo:  PARAR-SERVICO.bat
echo ============================================================
echo.
pause
