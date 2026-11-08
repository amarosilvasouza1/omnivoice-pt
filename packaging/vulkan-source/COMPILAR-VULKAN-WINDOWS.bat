@echo off
setlocal
cd /d "%~dp0"

where cmake >nul 2>nul
if errorlevel 1 (
  echo ERRO: CMake nao foi encontrado no PATH.
  echo Instale o CMake 3.20 ou mais recente e abra um terminal novo.
  pause
  exit /b 1
)

if "%VULKAN_SDK%"=="" (
  echo ERRO: a variavel VULKAN_SDK nao foi encontrada.
  echo Instale o Vulkan SDK e abra um terminal novo.
  pause
  exit /b 1
)

echo Configurando OmniVoice-PT com Vulkan...
cmake -S . -B build-vulkan ^
  -DGGML_VULKAN=ON ^
  -DGGML_CUDA=OFF ^
  -DOVPT_BUILD_TESTS=OFF ^
  -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 goto :erro

echo Compilando...
cmake --build build-vulkan --config Release --target omnivoice-pt --parallel
if errorlevel 1 goto :erro

echo.
echo Compilacao concluida.
echo Procure o executavel em build-vulkan\bin ou build-vulkan\bin\Release.
pause
exit /b 0

:erro
echo.
echo A compilacao falhou. Leia a mensagem acima para identificar a dependencia ausente.
pause
exit /b 1

