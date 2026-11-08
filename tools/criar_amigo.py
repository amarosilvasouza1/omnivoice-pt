#!/usr/bin/env python3
"""Cria uma conexão (chave) para um amigo e imprime o link dele.

Insere direto no banco (build/api.db), então funciona com o servidor no ar
ou não. O amigo criado já vem com:
  - todas as vozes liberadas ('*') + as privadas que ele mesmo enviar
  - os 3 modelos disponíveis (q8_0, f16, q4_0)
  - limites generosos (500 caracteres, 12 pedidos/min)

Uso: python tools/criar_amigo.py "Nome do Amigo"
"""

import secrets
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\omnivoice-pt")
DB = ROOT / "build" / "api.db"

name = (sys.argv[1] if len(sys.argv) > 1 else "amigo").strip() or "amigo"
key = "ov_" + secrets.token_urlsafe(18)

if not DB.exists():
    print("ERRO: banco build/api.db não existe. Rode o serviço uma vez primeiro.")
    sys.exit(1)

conn = sqlite3.connect(str(DB))
try:
    conn.execute(
        "INSERT INTO connections "
        "(name, api_key, enabled, max_chars, max_steps, allowed_quant, "
        " allowed_voices, rate_limit_per_min, notes, created_at) "
        "VALUES (?, ?, 1, 500, 32, 'q8_0,f16,q4_0', '*', 12, '', ?)",
        (name, key, time.time()),
    )
    conn.commit()
finally:
    conn.close()

url = ""
link_file = ROOT / "LINK-ATUAL.txt"
if link_file.exists():
    url = link_file.read_text(encoding="utf-8").strip()

print()
print("=" * 60)
print(f"  Link criado para: {name}")
print("=" * 60)
print()
if url:
    print("  >> MANDE ESTE LINK PARA O SEU AMIGO (pagina pronta):")
    print(f"     {url}/app?key={key}")
    print()
    print("  >> Para ele usar na IA/codigo dele (API):")
    print(f"     Endereco base : {url}")
    print(f"     Chave (key)   : {key}")
    print("     Header ngrok  : ngrok-skip-browser-warning: 1")
    print(f"     Documentacao  : {url}/docs")
else:
    print(f"  Chave do amigo: {key}")
    print()
    print("  (LINK-ATUAL.txt nao encontrado — rode INICIAR-SERVICO.bat")
    print("   para criar o link publico; depois o link fica:")
    print(f"     SEU_LINK/app?key={key} )")
print()
print("  Ele pode: escolher o modelo (q8/f16/q4), enviar uma voz")
print("  para clonar (fica privada, so dele) e gerar audio.")
print("=" * 60)
