@echo off
chcp 65001 >nul
title Criar link para um amigo
cd /d D:\omnivoice-pt

echo ============================================================
echo    Criar um link exclusivo para um amigo
echo ============================================================
echo.
set /p NOME=Digite o nome do amigo:
if "%NOME%"=="" set NOME=amigo

.venv\Scripts\python.exe tools\criar_amigo.py "%NOME%"

echo.
echo (O amigo pode trocar de modelo e enviar a propria voz pela
echo  pagina ou pela API. A voz dele fica privada, so ele usa.)
echo.
pause
