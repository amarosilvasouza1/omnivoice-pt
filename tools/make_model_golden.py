#!/usr/bin/env python3
"""Golden do forward do OmniVoice para validar o grafo GGML em C.

Monta uma entrada de inferência real (prefixo de estilo + texto PT +
10 posições de áudio mascaradas), roda o forward em float32 na CPU e
salva entradas + logits em binários planos:

  build/model_golden.bin:
    magic "OVMG" | n_codebooks u32 | seq_len u32 | n_audio u32 | vocab_audio u32
    input_ids  i32[n_codebooks * seq_len]   (row-major: codebook, seq)
    audio_mask u8[seq_len]
    logits     f32[n_codebooks * n_audio * vocab_audio]
               (apenas as últimas n_audio posições; ordem: cb, pos, vocab)
"""

import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "omnivoice-src"))

from omnivoice.models.omnivoice import OmniVoice  # noqa: E402

SNAPSHOT = str(
    Path(__file__).resolve().parent.parent / "hf-cache" / "hub"
    / "models--k2-fsa--OmniVoice" / "snapshots" / "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
)

TEXT = "Bom dia, tudo bem?"
N_AUDIO = 10


def main():
    model = OmniVoice.from_pretrained(SNAPSHOT, device_map="cpu", dtype=torch.float32)
    model.eval()

    inputs = model._prepare_inference_inputs(
        text=TEXT,
        num_target_tokens=N_AUDIO,
        ref_text=None,
        ref_audio_tokens=None,
        lang="pt",
        instruct=None,
        denoise=True,  # sem ref_audio o <|denoise|> não entra; igual ao runtime
    )
    input_ids = inputs["input_ids"]  # (1, C, S)
    audio_mask = inputs["audio_mask"]  # (1, S)

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=torch.ones(
                1, 1, input_ids.size(2), input_ids.size(2), dtype=torch.bool
            ),
        ).logits  # (1, C, S, V)

    C = input_ids.size(1)
    S = input_ids.size(2)
    V = logits.size(-1)
    tail = logits[0, :, S - N_AUDIO : S, :].contiguous().float().numpy()

    out = Path(__file__).resolve().parent.parent / "build" / "model_golden.bin"
    with open(out, "wb") as f:
        f.write(b"OVMG")
        f.write(struct.pack("<IIII", C, S, N_AUDIO, V))
        f.write(input_ids[0].numpy().astype(np.int32).tobytes())
        f.write(audio_mask[0].numpy().astype(np.uint8).tobytes())
        f.write(tail.astype(np.float32).tobytes())

    print(f"seq_len={S} codebooks={C} n_audio={N_AUDIO} vocab={V}")
    print(f"input_ids[0,:12] = {input_ids[0, 0, :12].tolist()}")
    print(f"logits tail: mean={tail.mean():.4f} std={tail.std():.4f}")
    print(f"argmax cb0 pos0 = {tail[0, 0].argmax()}")
    print(f"salvo: {out}")


if __name__ == "__main__":
    main()
