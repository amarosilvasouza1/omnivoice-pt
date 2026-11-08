@echo off
chcp 65001 >nul
title Configurar link fixo (ngrok) - so 1 vez
cd /d "%~dp0"

echo ============================================================
echo    Configuracao do LINK FIXO (ngrok)
echo    So precisa fazer isto UMA vez.
echo ============================================================
echo.
echo  Passo 1) Crie uma conta gratis (2 min):
echo             https://dashboard.ngrok.com/signup
echo.
echo  Passo 2) Copie seu Authtoken:
echo             https://dashboard.ngrok.com/get-started/your-authtoken
echo.
echo  Passo 3) Pegue seu dominio fixo gratis:
echo             https://dashboard.ngrok.com/domains
echo             (algo como  seunome.ngrok-free.app)
echo.
echo ------------------------------------------------------------
echo.
set /p TOKEN=Cole aqui seu Authtoken:
if "%TOKEN%"=="" ( echo. & echo Token vazio. Saindo. & pause & exit /b )
tools\bin\ngrok.exe config add-authtoken %TOKEN%
if errorlevel 1 ( echo. & echo Falhou ao salvar o token. Confira e tente de novo. & pause & exit /b )

echo.
set /p DOM=Cole aqui seu dominio fixo (ex: seunome.ngrok-free.app):
if "%DOM%"=="" ( echo. & echo Dominio vazio. Saindo. & pause & exit /b )
REM Aceita tanto o dominio puro quanto uma URL colada por engano.
set "DOM=%DOM:https://=%"
set "DOM=%DOM:http://=%"
for /f "tokens=1 delims=/" %%D in ("%DOM%") do set "DOM=%%D"
if "%DOM%"=="" ( echo. & echo Dominio invalido. Saindo. & pause & exit /b )
echo %DOM%>ngrok-dominio.txt

echo.
echo ============================================================
echo   Configurado com sucesso!
echo   Seu link fixo sera:  https://%DOM%
echo.
echo   Agora use  INICIAR-SERVICO-FIXO.bat  para subir o servico.
echo   Esse link NUNCA MAIS muda.
echo ============================================================
echo.
pause
