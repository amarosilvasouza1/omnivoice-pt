#!/usr/bin/env python3
import sys

from transformers import AutoTokenizer

SNAPSHOT = (
    r"D:\omnivoice-pt\hf-cache\hub\models--k2-fsa--OmniVoice"
    r"\snapshots\c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
)

t = AutoTokenizer.from_pretrained(SNAPSHOT)
ids = [int(x) for x in sys.argv[1:]]
for i in ids:
    tok = t.convert_ids_to_tokens([i])[0]
    print(i, repr(tok), "->", repr(t.decode([i])))

print(t("Olá! Esta é a primeira síntese.", add_special_tokens=False).input_ids)
