#!/usr/bin/env python3
"""Gera um voice prompt binário (.ovprompt) a partir de um WAV de referência.

Roda UMA vez por voz. Faz em Python (CPU) todo o trabalho pesado que o
runtime C nunca mais precisa repetir:
  - pré-processamento do áudio (remoção de silêncio, trim, RMS)
  - transcrição automática via Whisper (se --ref_text não for dado)
  - encode do áudio em tokens RVQ via HiggsAudioV2Tokenizer (encoder + HuBERT)

Formato .ovprompt (little-endian):
  magic "OVPR" | version u32 | n_codebooks u32 | n_frames u32 | ref_rms f32
  ref_text_len u32 | ref_text (utf-8)
  tokens u16[n_codebooks * n_frames]   (ordem: codebook-major)

Uso:
    python tools/make_prompt.py --ref_audio voz.wav --out voz.ovprompt \
        [--ref_text "transcrição exata do áudio"]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# Reutiliza o pré-processamento do próprio OmniVoice (código idêntico ⇒
# tokens idênticos aos do pipeline original).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "omnivoice-src"))

from omnivoice.utils.audio import load_audio, remove_silence, trim_long_audio  # noqa: E402
from omnivoice.utils.text import add_punctuation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_audio", required=True)
    ap.add_argument("--ref_text", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument(
        "--asr_model",
        default="openai/whisper-large-v3-turbo",
        help="Usado apenas se --ref_text não for fornecido.",
    )
    ap.add_argument(
        "--max_ref_seconds",
        type=float,
        default=None,
        help="Encurta a referência para no máximo N segundos (corta num "
        "silêncio próximo, se houver). Referência menor = clonagem mais rápida "
        "e mais leve na VRAM. Ideal 4-8s.",
    )
    args = ap.parse_args()

    from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel
    from huggingface_hub import snapshot_download
    import os

    if os.path.isdir(args.model):
        model_dir = Path(args.model)
    else:
        model_dir = Path(
            snapshot_download(args.model, allow_patterns=["audio_tokenizer/*"])
        )
    tok_dir = model_dir / "audio_tokenizer"

    audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
        str(tok_dir), device_map="cpu"
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(tok_dir))
    sampling_rate = feature_extractor.sampling_rate

    # --- mesmo pré-processamento de create_voice_clone_prompt ---
    ref_wav = load_audio(args.ref_audio, sampling_rate)

    ref_rms = float(np.sqrt(np.mean(ref_wav**2)))
    if 0 < ref_rms < 0.1:
        ref_wav = ref_wav * 0.1 / ref_rms

    if args.ref_text is None:
        ref_wav = trim_long_audio(ref_wav, sampling_rate, trim_threshold=20.0)
    ref_wav = remove_silence(
        ref_wav, sampling_rate, mid_sil=200, lead_sil=100, trail_sil=200
    )
    if ref_wav.shape[-1] == 0:
        raise ValueError("Áudio de referência vazio após remoção de silêncio.")

    # Encurta a referência para acelerar/aliviar a clonagem. Corta no maior
    # silêncio dentro de [0.6*N, N] segundos para não terminar no meio de uma
    # palavra; se não houver silêncio, corta em N.
    trimmed = False
    if args.max_ref_seconds is not None:
        target = int(args.max_ref_seconds * sampling_rate)
        if ref_wav.shape[-1] > target:
            lo = int(0.6 * target)
            window = ref_wav[0, lo:target]
            # ponto de menor energia local = provável silêncio/pausa
            frame = max(1, sampling_rate // 100)  # 10 ms
            energies = [
                (i, float(np.mean(window[i : i + frame] ** 2)))
                for i in range(0, max(1, len(window) - frame), frame)
            ]
            if energies:
                cut_rel = min(energies, key=lambda x: x[1])[0]
                cut = lo + cut_rel
            else:
                cut = target
            ref_wav = ref_wav[:, :cut]
            trimmed = True
            print(f"referência encurtada para {cut / sampling_rate:.1f}s")

    dur = ref_wav.shape[-1] / sampling_rate
    if dur > 20.0:
        print(f"aviso: referência com {dur:.1f}s (>20s); ideal é 3-10s")

    # A transcrição PRECISA corresponder ao áudio final. Se o áudio foi
    # cortado, um --ref_text fornecido (transcrição do áudio ORIGINAL, mais
    # longo) não bate mais — o modelo então "vaza" o texto de referência
    # sobrando na saída (visto numa referência longa). Nesse caso, re-transcreve o
    # trecho cortado com o Whisper para garantir que texto e áudio casem.
    ref_text = args.ref_text
    if trimmed and ref_text is not None:
        print("áudio foi cortado — re-transcrevendo para casar com o texto...")
        ref_text = None
    if ref_text is None:
        from transformers import pipeline as hf_pipeline

        print(f"transcrevendo com {args.asr_model} (CPU, uma única vez)...")
        asr = hf_pipeline(
            "automatic-speech-recognition", model=args.asr_model, device_map="cpu"
        )
        ref_text = asr({"array": np.squeeze(ref_wav), "sampling_rate": sampling_rate})[
            "text"
        ].strip()
        print(f"transcrição: {ref_text}")

    ref_text = add_punctuation(ref_text)

    chunk_size = audio_tokenizer.config.hop_length
    clip = int(ref_wav.shape[-1] % chunk_size)
    if clip > 0:
        ref_wav = ref_wav[:, :-clip]

    with torch.inference_mode():
        wav_t = torch.from_numpy(ref_wav).unsqueeze(0)
        tokens = audio_tokenizer.encode(wav_t).audio_codes.squeeze(0)  # (C, T)

    tokens_np = tokens.cpu().numpy().astype(np.uint16)
    n_codebooks, n_frames = tokens_np.shape
    print(f"tokens de referência: {n_codebooks} codebooks x {n_frames} frames "
          f"({n_frames / audio_tokenizer.config.frame_rate:.1f}s)")

    text_bytes = ref_text.encode("utf-8")
    with open(args.out, "wb") as f:
        f.write(b"OVPR")
        f.write(struct.pack("<IIIf", 1, n_codebooks, n_frames, ref_rms))
        f.write(struct.pack("<I", len(text_bytes)))
        f.write(text_bytes)
        f.write(tokens_np.tobytes())

    size = Path(args.out).stat().st_size
    print(f"salvo: {args.out} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
