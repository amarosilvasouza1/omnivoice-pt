#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"

cmake --preset linux-q4-sandybridge
cmake --build --preset linux-q4-sandybridge

echo
echo "Binario criado em: $ROOT/cmake-build-linux-q4/bin/omnivoice-pt"
