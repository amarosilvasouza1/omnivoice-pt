#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BIN="$ROOT/omnivoice-pt"
MODEL="$ROOT/modelos/omnivoice-pt-q4_0.gguf"
DECODER="$ROOT/modelos/higgs-decoder-f16.gguf"
TOKENIZER="$ROOT/modelos/tokenizer.bin"

if [ ! -x "$BIN" ]; then
    echo "Erro: executavel $BIN nao encontrado ou sem permissao." >&2
    echo "Tente: chmod +x '$BIN' '$0'" >&2
    exit 1
fi

if [ -r /proc/cpuinfo ] && ! grep -qiw avx /proc/cpuinfo; then
    echo "Erro: esta versao requer AVX (Intel Core de 2a geracao ou mais novo)." >&2
    exit 1
fi

# Evita usar Hyper-Threading como se fossem nucleos reais. Em Sandy Bridge,
# quatro threads fisicas normalmente vencem oito threads logicas neste grafo.
if [ -n "${OV_THREADS:-}" ]; then
    THREADS=$OV_THREADS
elif [ -r /proc/cpuinfo ]; then
    THREADS=$(awk -F: '/^cpu cores[[:space:]]*:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' /proc/cpuinfo)
    [ -n "$THREADS" ] || THREADS=2
else
    THREADS=2
fi

STEPS=${OV_STEPS:-8}

if [ "$#" -eq 0 ]; then
    printf "Texto: "
    IFS= read -r TEXT
else
    TEXT=$1
    shift
fi

if [ -z "$TEXT" ]; then
    echo "Nenhum texto informado." >&2
    exit 2
fi

if [ "$#" -gt 0 ]; then
    OUTPUT=$1
else
    OUTPUT="$ROOT/saida-$(date +%Y%m%d-%H%M%S).wav"
fi

set -- "$BIN" -m "$MODEL" -d "$DECODER" -k "$TOKENIZER" \
    -t "$TEXT" -o "$OUTPUT" --threads "$THREADS" --steps "$STEPS"

if [ -n "${OV_PROMPT:-}" ]; then
    set -- "$@" -p "$OV_PROMPT"
fi

echo "Q4 Linux: $THREADS threads, $STEPS passos"
"$@"
echo "Audio salvo em: $OUTPUT"
