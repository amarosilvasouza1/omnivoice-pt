#!/usr/bin/env python3
"""Inspeciona o HiggsAudioV2TokenizerModel: o que o decode realmente usa."""

import sys

from transformers import HiggsAudioV2TokenizerModel

SNAPSHOT = (
    r"D:\omnivoice-pt\hf-cache\hub\models--k2-fsa--OmniVoice"
    r"\snapshots\c5fdb5ccb189668d56333f77ba2629f4cd7535f4\audio_tokenizer"
)

m = HiggsAudioV2TokenizerModel.from_pretrained(SNAPSHOT, device_map="cpu")

total = 0
groups: dict[str, int] = {}
for name, p in m.named_parameters():
    top = ".".join(name.split(".")[:2])
    groups[top] = groups.get(top, 0) + p.numel()
    total += p.numel()

print("--- parametros por grupo ---")
for k, v in sorted(groups.items(), key=lambda x: -x[1]):
    print(f"{k:40s} {v / 1e6:8.2f}M")
print(f"TOTAL: {total / 1e6:.1f}M\n")

print("--- submodulos de primeiro nivel ---")
for name, mod in m.named_children():
    n = sum(p.numel() for p in mod.parameters())
    print(f"{name:30s} {type(mod).__name__:40s} {n / 1e6:8.2f}M")

print("\n--- assinatura do decode ---")
import inspect

print(inspect.signature(m.decode))
src = inspect.getsource(m.decode)
print(src[:3000])
