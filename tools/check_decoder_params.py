#!/usr/bin/env python3
"""Confere nomes de parâmetros vs buffers do caminho de decode."""

from transformers import HiggsAudioV2TokenizerModel

SNAPSHOT = (
    r"D:\omnivoice-pt\hf-cache\hub\models--k2-fsa--OmniVoice"
    r"\snapshots\c5fdb5ccb189668d56333f77ba2629f4cd7535f4\audio_tokenizer"
)

m = HiggsAudioV2TokenizerModel.from_pretrained(SNAPSHOT, device_map="cpu")

print("--- quantizer: parameters ---")
for n, p in list(m.quantizer.named_parameters())[:6]:
    print(f"  {n}: {tuple(p.shape)}")
print("--- quantizer: buffers ---")
for n, b in list(m.quantizer.named_buffers())[:8]:
    print(f"  {n}: {tuple(b.shape)}")

print("--- acoustic_decoder: primeiros parametros ---")
for n, p in list(m.acoustic_decoder.named_parameters())[:10]:
    print(f"  {n}: {tuple(p.shape)}")
print("--- acoustic_decoder: buffers ---")
for n, b in list(m.acoustic_decoder.named_buffers())[:5]:
    print(f"  {n}: {tuple(b.shape)}")
