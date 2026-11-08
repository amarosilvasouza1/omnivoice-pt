#!/usr/bin/env python3
"""Exporta APENAS o caminho de decode do HiggsAudioV2Tokenizer para GGUF.

decode(audio_codes) = acoustic_decoder(fc2(quantizer.decode(audio_codes)))

Também grava um vetor de teste dourado (códigos aleatórios fixos + áudio
resultante) para validar o port C do decoder amostra a amostra.

Uso:
    python tools/export_decoder.py --out D:/omnivoice-pt/build
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = "k2-fsa/OmniVoice"
# Mesma revisão fixada em export.py: decoder e modelo vêm do mesmo commit.
REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
SNAPSHOT = (
    Path(__file__).resolve().parent.parent / "hf-cache" / "hub"
    / "models--k2-fsa--OmniVoice" / "snapshots" / REVISION / "audio_tokenizer"
)


def default_model_dir() -> str:
    """Usa o cache local do projeto; se não existir (outra máquina), baixa
    só o audio_tokenizer da revisão fixada."""
    if SNAPSHOT.is_dir():
        return str(SNAPSHOT)
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        REPO, revision=REVISION, allow_patterns=["audio_tokenizer/*"]
    )
    return str(Path(root) / "audio_tokenizer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="pasta do audio_tokenizer (padrão: revisão fixada)")
    ap.add_argument("--out", default="build")
    args = ap.parse_args()
    if args.model is None:
        args.model = default_model_dir()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import gguf
    from transformers import HiggsAudioV2TokenizerModel

    m = HiggsAudioV2TokenizerModel.from_pretrained(args.model, device_map="cpu")
    m.eval()

    cfg = m.config

    # --- estrutura completa do caminho de decode (para o port C) ---
    struct_path = out_dir / "decoder_structure.txt"
    with open(struct_path, "w", encoding="utf-8") as f:
        f.write("=== config ===\n")
        f.write(json.dumps(cfg.to_dict(), indent=2, default=str))
        f.write("\n\n=== quantizer ===\n")
        f.write(str(m.quantizer))
        f.write("\n\n=== fc2 ===\n")
        f.write(str(m.fc2))
        f.write("\n\n=== acoustic_decoder ===\n")
        f.write(str(m.acoustic_decoder))
    print(f"estrutura salva em {struct_path}")

    # --- pesos do decode para GGUF (F16, convs pequenas em F32) ---
    out_path = out_dir / "higgs-decoder-f16.gguf"
    w = gguf.GGUFWriter(str(out_path), arch="higgs-decoder")
    w.add_uint32("higgs.sampling_rate", cfg.sample_rate)
    w.add_uint32("higgs.hop_length", cfg.hop_length)
    w.add_uint32("higgs.frame_rate", int(cfg.sample_rate // cfg.hop_length))

    Q = gguf.GGMLQuantizationType
    n_tensors = 0
    total_bytes = 0

    def add_tensor(full: str, data):
        nonlocal n_tensors, total_bytes
        data = data.detach().numpy().astype(np.float32)
        # pesos de conv em F16 (o conv_transpose_1d do ggml exige F16);
        # biases, alphas do snake e RVQ ficam em F32
        if full.endswith(".weight") and "conv" in full:
            w.add_tensor(full, data.astype(np.float16), raw_dtype=Q.F16)
            total_bytes += data.size * 2
        else:
            w.add_tensor(full, data, raw_dtype=Q.F32)
            total_bytes += data.size * 4
        n_tensors += 1

    def add_module(prefix: str, module: torch.nn.Module):
        for name, p in module.named_parameters():
            add_tensor(f"{prefix}.{name}", p)
        # o embed dos codebooks RVQ é buffer, não parâmetro
        for name, b in module.named_buffers():
            if name.endswith(".codebook.embed"):
                add_tensor(f"{prefix}.{name}", b)

    add_module("quantizer", m.quantizer)
    add_module("fc2", m.fc2)
    add_module("decoder", m.acoustic_decoder)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"GGUF do decoder: {out_path} "
        f"({n_tensors} tensores, {total_bytes / 1e6:.1f} MB)"
    )

    # --- vetor de teste dourado ---
    torch.manual_seed(42)
    n_cb = 8  # OmniVoice usa 8 codebooks
    n_frames = 50  # 2s de audio
    codes = torch.randint(0, 1024, (1, n_cb, n_frames), dtype=torch.long)
    with torch.inference_mode():
        audio = m.decode(codes, return_dict=False)  # (1, 1, T)
    audio_np = audio.squeeze().numpy().astype(np.float32)

    np.savez(
        out_dir / "decoder_golden.npz",
        codes=codes.numpy().astype(np.int32),
        audio=audio_np,
    )
    # versão binária plana para o teste em C:
    #   magic "OVDG" | n_codebooks u32 | n_frames u32 | n_samples u32
    #   codes i32[n_cb * n_frames] | audio f32[n_samples]
    import struct

    with open(out_dir / "decoder_golden.bin", "wb") as f:
        f.write(b"OVDG")
        f.write(struct.pack("<III", n_cb, n_frames, audio_np.shape[0]))
        f.write(codes.numpy().astype(np.int32).tobytes())
        f.write(audio_np.tobytes())
    print(
        f"golden test: codes {tuple(codes.shape)} -> audio {audio_np.shape} "
        f"(rms {np.sqrt((audio_np**2).mean()):.4f})"
    )


if __name__ == "__main__":
    main()
