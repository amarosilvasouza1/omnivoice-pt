@echo off
chcp 65001 >nul
cd /d "%~dp0"
omnivoice-pt.exe -m "modelos\omnivoice-pt-q8_0.gguf" -d "modelos\higgs-decoder-f16.gguf" -k "modelos\tokenizer.bin" -t "Teste da versão para placa de vídeo AMD." -o "teste-rx580.wav" --gpu --steps 4
if errorlevel 1 (
  echo ERRO: confira o driver AMD e o suporte Vulkan.
) else (
  echo OK: teste-rx580.wav foi criado.
)
pause
