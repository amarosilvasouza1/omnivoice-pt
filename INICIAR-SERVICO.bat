@echo off
chcp 65001 >nul
title OmniVoice-PT - Servico
cd /d "%~dp0"

echo ============================================================
echo    Iniciando o servico OmniVoice-PT
echo    (servidor de voz + link publico para os amigos)
echo ============================================================
echo.

REM --- 1. servidor da API (so sobe se ainda nao estiver no ar) ---
powershell -NoProfile -Command "try{if((Invoke-WebRequest http://127.0.0.1:5180/docs -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200){exit 0}}catch{}; exit 1"
if not errorlevel 1 (
  echo Servidor ja estava rodando. Reaproveitando.
  goto tunnel
)

echo Subindo o servidor e um worker ^(uns 15 segundos^)...
start "OmniVoice - Servidor (nao feche)" .venv\Scripts\python.exe tools\api_server.py

:esperar_servidor
timeout /t 3 >nul
powershell -NoProfile -Command "try{if((Invoke-WebRequest http://127.0.0.1:5180/docs -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200){exit 0}}catch{}; exit 1"
if errorlevel 1 goto esperar_servidor
echo Servidor pronto!

:tunnel
echo.
echo Criando o link publico ^(Cloudflare^)...

REM mata tuneis antigos para nao duplicar, e comeca um log novo
taskkill /f /im cloudflared.exe >nul 2>&1
if exist cloudflared.log del cloudflared.log >nul 2>&1

start "OmniVoice - Tunel (nao feche)" cmd /c "tools\bin\cloudflared.exe tunnel --url http://localhost:5180 > cloudflared.log 2>&1"

REM espera a URL publica aparecer no log (ate ~40s)
set URL=
set /a TENTATIVAS=0
:achar_url
timeout /t 2 >nul
set /a TENTATIVAS+=1
for /f "usebackq tokens=*" %%u in (`powershell -NoProfile -Command "if(Test-Path 'cloudflared.log'){ $m = Select-String -Path 'cloudflared.log' -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1; if($m){ $m.Matches[0].Value } }"`) do set URL=%%u
if not "%URL%"=="" goto pronto
if %TENTATIVAS% geq 20 goto sem_url
goto achar_url

:pronto
echo.
echo ============================================================
echo    TUDO PRONTO!  Servico no ar.
echo ============================================================
echo.
echo    Documentacao ^(mande para os amigos^):
echo       %URL%/docs
echo.
echo    Link direto para um amigo ^(troque a CHAVE^):
echo       %URL%/app?key=CHAVE_DO_AMIGO
echo.
echo    Seu painel de admin ^(ver chaves e atividade^):
echo       %URL%/admin
echo.
echo    O token do admin aparece na janela "OmniVoice - Servidor".
echo.
echo    IMPORTANTE: nao feche as janelas "Servidor" e "Tunel"
echo    enquanto quiser manter o servico funcionando.
echo    Para desligar tudo, rode  PARAR-SERVICO.bat
echo ============================================================
echo.
echo O link tambem esta salvo no arquivo  LINK-ATUAL.txt
echo %URL%> LINK-ATUAL.txt
pause
exit /b

:sem_url
echo.
echo Nao consegui ler o link publico automaticamente.
echo Abra a janela "OmniVoice - Tunel" e procure a linha
echo   https://alguma-coisa.trycloudflare.com
echo (o servidor local ja esta funcionando em http://127.0.0.1:5180)
echo.
pause
