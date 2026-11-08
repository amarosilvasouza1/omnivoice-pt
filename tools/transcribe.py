#!/usr/bin/env python3
"""Transcreve um WAV com Whisper para validar inteligibilidade."""

import sys

from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device_map="cpu",
)
result = asr(sys.argv[1], generate_kwargs={"language": "portuguese"})
print("TRANSCRICAO:", result["text"])
