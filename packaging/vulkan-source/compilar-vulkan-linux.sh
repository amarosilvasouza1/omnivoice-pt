#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

if ! command -v cmake >/dev/null 2>&1; then
  echo "ERRO: CMake 3.20 ou mais recente nao foi encontrado." >&2
  exit 1
fi

if ! command -v glslc >/dev/null 2>&1 && ! command -v glslangValidator >/dev/null 2>&1; then
  echo "ERRO: instale glslc ou glslangValidator, alem dos headers/loader Vulkan." >&2
  exit 1
fi

cmake -S . -B build-vulkan \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_CUDA=OFF \
  -DOVPT_BUILD_TESTS=OFF

cmake --build build-vulkan --config Release --target omnivoice-pt --parallel

echo "Compilacao concluida: build-vulkan/bin/omnivoice-pt"

