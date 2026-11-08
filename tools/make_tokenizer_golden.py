#!/usr/bin/env python3
"""Gera vetores dourados de tokenização para validar o BPE do runtime C.

Salva build/tokenizer_golden.json: lista de {texto, ids} produzidos pelo
tokenizer HF original (o runtime C deve reproduzir exatamente).
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

SNAPSHOT = (
    r"D:\omnivoice-pt\hf-cache\hub\models--k2-fsa--OmniVoice"
    r"\snapshots\c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
)

CASES = [
    "Olá, mundo!",
    "O rato roeu a roupa do rei de Roma.",
    "São Paulo, 10 de julho de 2026 — R$ 1.234,56.",
    "A otimização de memória é fundamental para GPUs pequenas.",
    "coração, ação, pão, maçã, você, português",
    "<|lang_start|>pt<|lang_end|><|instruct_start|>None<|instruct_end|>",
    "<|denoise|><|text_start|>Teste de síntese.<|text_end|>",
    "email@teste.com.br, https://exemplo.pt/página?q=1",
    "Ele disse: “não vou”… e foi embora (às 15h30).",
    "[laughter] Que engraçado! [sigh]",
    "Olá! Esta é a primeira síntese de fala feita inteiramente em C.",
    "é ó à â ê ô õ ç É Ó",
    "<|text_start|>Olá! Esta é a primeira síntese.<|text_end|>",
]


def main():
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    out = []
    for text in CASES:
        ids = tok(text, add_special_tokens=False).input_ids
        out.append({"text": text, "ids": ids})
        print(f"{len(ids):4d} tokens | {text[:60]}")

    path = Path(r"D:\omnivoice-pt\build\tokenizer_golden.json")
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"salvo: {path}")

    # versão binária para o teste em C:
    #   magic "OVGT" | n_cases u32
    #   n_cases x { text_len u32 | text utf8 | n_ids u32 | ids u32[] }
    import struct

    bin_path = Path(r"D:\omnivoice-pt\build\tokenizer_golden.bin")
    with open(bin_path, "wb") as f:
        f.write(b"OVGT")
        f.write(struct.pack("<I", len(out)))
        for case in out:
            tb = case["text"].encode("utf-8")
            f.write(struct.pack("<I", len(tb)))
            f.write(tb)
            f.write(struct.pack("<I", len(case["ids"])))
            f.write(struct.pack(f"<{len(case['ids'])}I", *case["ids"]))
    print(f"salvo: {bin_path}")


if __name__ == "__main__":
    main()
