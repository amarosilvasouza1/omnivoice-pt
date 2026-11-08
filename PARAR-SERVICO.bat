@echo off
chcp 65001 >nul
title OmniVoice-PT - Parar
echo Desligando o servico OmniVoice-PT...

REM 1. tuneis (Cloudflare e ngrok)
taskkill /f /im cloudflared.exe >nul 2>&1
taskkill /f /im ngrok.exe >nul 2>&1

REM 2. o worker unico (processo do modelo)
taskkill /f /im omnivoice-pt.exe >nul 2>&1

REM 3. o servidor (o python que escuta na porta 5180)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :5180 ^| findstr LISTENING') do taskkill /f /pid %%p >nul 2>&1

echo.
echo Servico parado. Pode fechar esta janela.
timeout /t 3 >nul
