@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d D:\omnivoice-pt
set "CMAKE=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "BUILD=cmake-build-cuda-rtx2050-sm86"

if not exist "%BUILD%\build.ninja" (
  "%CMAKE%" -S . -B "%BUILD%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_ARCHITECTURES=86 ^
    -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler ^
    -DCMAKE_MAKE_PROGRAM=D:\omnivoice-pt\.venv\Scripts\ninja.exe ^
    -DGGML_CUDA=ON ^
    -DGGML_CUDA_GRAPHS=ON ^
    -DGGML_CUDA_NCCL=OFF ^
    -DGGML_CUDA_COMPRESSION_MODE=speed ^
    -DGGML_NATIVE=OFF ^
    -DOVPT_BUILD_TESTS=OFF
  if errorlevel 1 exit /b 1
)

"%CMAKE%" --build "%BUILD%" --target %*
