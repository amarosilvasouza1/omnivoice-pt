#!/usr/bin/env python3
"""Extrai WAVs do voices.zip e converte cada um em .ovprompt.

Roda UMA vez (ou quando adicionar novas vozes). Gera os arquivos .ovprompt
em data/voices/ — o api_server.py carrega-os automaticamente no startup.

Uso:
    .venv\\Scripts\\python.exe tools\\setup_voices.py

Lê os nomes e as transcrições de data/voices/vozes.json (local, fora do git).

Opções:
    --max-ref-seconds N   Encurta referências longas (padrão: 8s)
    --force               Re-gera .ovprompt mesmo que já exista
    --only NOME           Gera apenas a voz especificada
"""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# O console do Windows usa cp1252 por padrão, que não representa "→" nem
# outros caracteres usados nos prints abaixo — sem isso, o script crasha na
# primeira mensagem com UnicodeEncodeError assim que rodado fora de um shell
# que já force UTF-8 (ex.: direto no cmd.exe/PowerShell, ou por outra
# ferramenta que não define PYTHONIOENCODING).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
VOICES_ZIP = ROOT / "voices.zip"
VOICES_DIR = ROOT / "data" / "voices"
MAKE_PROMPT = ROOT / "tools" / "make_prompt.py"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

# Cadastro das vozes: nome -> idioma e transcrição exata de cada WAV.
# Fica num arquivo local (data/voices/ não vai para o git): as transcrições
# são falas de pessoas reais e muitas vezes dizem quem elas são. Formato:
#   {"minha_voz": {"language": "pt", "ref_text": "o que é dito no WAV"}}
VOICES_JSON = VOICES_DIR / "vozes.json"
VOICE_REGISTRY: dict[str, dict] = {}


def load_registry() -> dict:
    if not VOICES_JSON.exists():
        print(f"ERRO: {VOICES_JSON} não encontrado.")
        print('Crie o arquivo no formato {"nome": {"language": "pt", '
              '"ref_text": "transcrição exata do WAV"}}.')
        sys.exit(1)
    with open(VOICES_JSON, encoding="utf-8") as f:
        return json.load(f)


def extract_wavs(force: bool = False):
    """Extrai WAVs do voices.zip → data/voices/."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    if not VOICES_ZIP.exists():
        print(f"ERRO: {VOICES_ZIP} não encontrado.")
        return False

    with zipfile.ZipFile(VOICES_ZIP, "r") as zf:
        for entry in zf.infolist():
            if not entry.filename.endswith(".wav"):
                continue
            name = Path(entry.filename).name
            dest = VOICES_DIR / name
            if dest.exists() and not force:
                print(f"  já existe: {name}")
                continue
            with zf.open(entry) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"  extraído: {name} ({dest.stat().st_size:,} bytes)")

    return True


def convert_to_ovprompt(
    name: str,
    info: dict,
    max_ref_seconds: float | None = 8.0,
    force: bool = False,
):
    """Converte um WAV em .ovprompt via make_prompt.py."""
    wav_path = VOICES_DIR / f"{name}.wav"
    out_path = VOICES_DIR / f"{name}.ovprompt"

    if out_path.exists() and not force:
        print(f"  já existe: {out_path.name} ({out_path.stat().st_size:,} bytes)")
        return True

    if not wav_path.exists():
        print(f"  ERRO: WAV não encontrado: {wav_path}")
        return False

    args = [
        str(PYTHON),
        str(MAKE_PROMPT),
        "--ref_audio", str(wav_path),
        "--ref_text", info["ref_text"],
        "--out", str(out_path),
    ]
    if max_ref_seconds is not None:
        args += ["--max_ref_seconds", str(max_ref_seconds)]

    print(f"  convertendo {name}.wav -> {name}.ovprompt ...")
    # PYTHONIOENCODING força o make_prompt.py (subprocesso) a escrever seus
    # prints em UTF-8 mesmo quando herda um console cp1252 — sem isso, ele
    # imprime bytes cp1252 para acentos e o decode abaixo (encoding="utf-8")
    # quebra com UnicodeDecodeError.
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", env=child_env
    )

    if proc.returncode != 0:
        print(f"  ERRO ao converter {name}:")
        print((proc.stderr or "")[-1000:])
        return False

    # Mostra a saída do make_prompt (tamanho, duração, etc.)
    for line in (proc.stdout or "").strip().split("\n"):
        if line.strip():
            print(f"    {line.strip()}")
    for line in (proc.stderr or "").strip().split("\n"):
        if line.strip() and ("salvo" in line or "tokens" in line or "encurtada" in line):
            print(f"    {line.strip()}")

    if out_path.exists():
        print(f"  OK: {out_path.name} ({out_path.stat().st_size:,} bytes)")
        return True
    else:
        print(f"  ERRO: arquivo de saída não foi criado.")
        return False


def save_registry():
    """Salva o registro de vozes em data/voices/registry.json."""
    registry_path = VOICES_DIR / "registry.json"
    registry = {}
    for name, info in VOICE_REGISTRY.items():
        ovprompt = VOICES_DIR / f"{name}.ovprompt"
        registry[name] = {
            "name": name,
            "language": info["language"],
            "ref_text": info["ref_text"],
            "wav_file": f"{name}.wav",
            "prompt_file": f"{name}.ovprompt",
            "prompt_exists": ovprompt.exists(),
            "prompt_size_kb": round(ovprompt.stat().st_size / 1024, 1)
            if ovprompt.exists()
            else None,
        }
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"\nregistro salvo: {registry_path}")


def main():
    ap = argparse.ArgumentParser(description="Extrai e converte vozes para .ovprompt")
    ap.add_argument(
        "--max-ref-seconds",
        type=float,
        default=8.0,
        help="Encurta referências longas para N segundos (padrão: 8)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-gera .ovprompt mesmo que já exista",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="Gera apenas a voz especificada",
    )
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Pula extração do ZIP (WAVs já existem)",
    )
    args = ap.parse_args()
    VOICE_REGISTRY.update(load_registry())

    print("=" * 60)
    print("OmniVoice-PT — Setup de Vozes Clonadas")
    print("=" * 60)

    # 1. Extrair WAVs
    if not args.skip_extract:
        print("\n[1/3] Extraindo WAVs do voices.zip...")
        if not extract_wavs(force=args.force):
            sys.exit(1)
    else:
        print("\n[1/3] Extração pulada (--skip-extract)")

    # 2. Converter WAVs → .ovprompt
    voices = VOICE_REGISTRY
    if args.only:
        if args.only not in voices:
            print(f"ERRO: voz '{args.only}' não encontrada no registro.")
            print(f"Vozes disponíveis: {', '.join(voices.keys())}")
            sys.exit(1)
        voices = {args.only: voices[args.only]}

    print(f"\n[2/3] Convertendo {len(voices)} voz(es) para .ovprompt...")
    ok = 0
    fail = 0
    for name, info in voices.items():
        if convert_to_ovprompt(name, info, args.max_ref_seconds, args.force):
            ok += 1
        else:
            fail += 1

    # 3. Salvar registro
    print("\n[3/3] Salvando registro...")
    save_registry()

    # Resumo
    print("\n" + "=" * 60)
    print(f"Concluído: {ok} OK, {fail} falha(s)")
    if fail > 0:
        print("Rode novamente com --force para tentar re-gerar os que falharam.")
    print("\nVozes .ovprompt prontas em: data/voices/")
    print("O api_server.py carrega automaticamente ao iniciar.")
    print("=" * 60)


if __name__ == "__main__":
    main()
