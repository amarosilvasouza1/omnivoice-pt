@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Diagnostico da AMD Radeon RX 7600 (Vulkan)
echo ============================================================
echo.
omnivoice-pt.exe -m "modelos\omnivoice-pt-q8_0.gguf" -d "modelos\higgs-decoder-f16.gguf" -k "modelos\tokenizer.bin" -t "Teste da versao para placa de video AMD RX 7600." -o "teste-rx7600.wav" --gpu --steps 4
if exist "teste-rx7600.wav" (
  echo.
  echo OK: teste-rx7600.wav foi criado. A GPU Vulkan esta funcionando.
  start "" "teste-rx7600.wav"
) else (
  echo.
  echo ERRO: nenhum dispositivo Vulkan usavel.
  echo Atualize o driver AMD Adrenalin pelo site oficial e reinicie o PC.
)
pause
