#!/usr/bin/env python3
"""Roda o pipeline original (PyTorch) na GPU e mede tempo + VRAM de pico.

Usa o MESMO texto e o mesmo modo (auto voice, sem ref_audio) do teste do
runtime C, para comparação lado a lado justa.
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "omnivoice-src"))

from omnivoice.models.omnivoice import OmniVoice  # noqa: E402

SNAPSHOT = str(
    Path(__file__).resolve().parent.parent / "hf-cache" / "hub"
    / "models--k2-fsa--OmniVoice" / "snapshots" / "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
)

TEXT = (
    "Olá! Esta é a primeira síntese de fala feita inteiramente em C, "
    "com menos de um gigabyte de memória de vídeo."
)
OUT = str(Path(__file__).resolve().parent.parent / "build" / "teste_pytorch_gpu.wav")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

t0 = time.time()
model = OmniVoice.from_pretrained(SNAPSHOT, device_map=device, dtype=torch.float16)
t_load = time.time() - t0
vram_after_load = (
    torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0
)
vram_reserved_load = (
    torch.cuda.max_memory_reserved() / 1e6 if device == "cuda" else 0
)

t1 = time.time()
audios = model.generate(text=TEXT, language="pt")
t_gen = time.time() - t1

vram_peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0
vram_peak_reserved = (
    torch.cuda.max_memory_reserved() / 1e6 if device == "cuda" else 0
)

sf.write(OUT, audios[0], model.sampling_rate)
dur = len(audios[0]) / model.sampling_rate

print(f"device            : {device}")
print(f"tempo de carga    : {t_load:.2f}s")
print(f"VRAM apos carga   : {vram_after_load:.0f} MB (alocado) / {vram_reserved_load:.0f} MB (reservado)")
print(f"tempo de geracao  : {t_gen:.2f}s")
print(f"duracao do audio  : {dur:.2f}s")
print(f"RTF               : {t_gen / dur:.3f}")
print(f"VRAM pico total   : {vram_peak:.0f} MB (alocado) / {vram_peak_reserved:.0f} MB (reservado)")
print(f"salvo em          : {OUT}")
