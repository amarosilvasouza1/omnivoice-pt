@echo off
rem Build NVIDIA "universal": um unico pacote que roda da GTX 950 a RTX 50.
rem
rem Cada arquitetura vira codigo nativo (-real), entao nenhuma placa depende
rem de compilacao JIT na primeira execucao. O driver carrega na VRAM so o
rem codigo da arquitetura da placa; as outras ficam paradas no arquivo.
rem
rem   52  Maxwell    GTX 950 a 980 Ti (a GTX 750/750 Ti e sm_50: nao coberta,
rem                  usa a edicao Vulkan; cubin 5.2 nao roda em 5.0)
rem   61  Pascal     GTX 1050 a 1080 Ti
rem   75  Turing     GTX 1650/1660, RTX 2060 a 2080 Ti
rem   86  Ampere     RTX 3050 a 3090, RTX 2050 (o cubin 8.6 tambem roda em 8.9)
rem   89  Ada        RTX 4060 a 4090
rem   120a Blackwell RTX 5050 a 5090
rem   89-virtual     PTX para placas futuras (compilado pelo driver na hora)
rem
rem Fora: Kepler (sem suporte no CUDA 12), GTX 750/750 Ti (sm_50), Titan V
rem (Volta, sm_70) e placas de data center sem cubin proprio.
rem
rem GGML_NATIVE=OFF: a parte de CPU usa AVX2 generico, sem depender do
rem processador da maquina que compila.
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
set "CMAKE=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "BUILD=cmake-build-cuda-universal"

if not exist "%BUILD%\build.ninja" (
  "%CMAKE%" -S . -B "%BUILD%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    "-DCMAKE_CUDA_ARCHITECTURES=52-real;61-real;75-real;86-real;89-real;120a-real;89-virtual" ^
    -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler ^
    -DCMAKE_MAKE_PROGRAM=%~dp0.venv\Scripts\ninja.exe ^
    -DGGML_CUDA=ON ^
    -DGGML_CUDA_GRAPHS=ON ^
    -DGGML_CUDA_NCCL=OFF ^
    -DGGML_CUDA_COMPRESSION_MODE=size ^
    -DGGML_NATIVE=OFF ^
    -DOVPT_BUILD_TESTS=OFF
  if errorlevel 1 exit /b 1
)

"%CMAKE%" --build "%BUILD%" --target %*
