@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d D:\omnivoice-pt
cmake -S . -B cmake-build-cuda >nul 2>&1
cmake --build cmake-build-cuda --target %*
