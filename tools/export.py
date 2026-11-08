#!/usr/bin/env python3
"""Exporta o OmniVoice (k2-fsa) para um GGUF enxuto, otimizado para português.

O que este script faz (roda UMA vez, em CPU):
  1. Baixa k2-fsa/OmniVoice do HuggingFace (cache em D:).
  2. Monta o conjunto de tokens de texto que o português precisa
     (filtro de charset latino + byte-tokens + tokens especiais).
     O tokenizer NÃO é alterado — a tokenização continua idêntica ao
     original; uma tabela de remapeamento id_original -> linha_podada
     traduz para a matriz de embeddings reduzida. Tokens fora do conjunto
     caem para os byte-tokens (sempre mantidos), então qualquer texto
     continua representável.
  3. Converte os pesos para GGUF:
       - backbone Qwen3-0.6B em Q8_0 (norms em F32)
       - embeddings de texto podados em Q8_0
       - audio_embeddings / audio_heads em F16
  4. Gera tokenizer.bin (vocab + merges em formato binário compacto
     para o runtime C) e remap.bin (int32[vocab] id->linha ou -1).

Uso:
    python tools/export.py --out build
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Charset do português (generoso de propósito)
# ---------------------------------------------------------------------------

_PT_EXTRA = set(
    "áàâãäéèêëíìîïóòôõöúùûüçñýÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇÑ"
    "ªº°–—‘’“”…€«»·§†‡™®© "
)


def _is_pt_char(ch: str) -> bool:
    o = ord(ch)
    if 0x20 <= o <= 0x7E:  # ASCII imprimível (letras, dígitos, pontuação)
        return True
    if ch in ("\n", "\t", "\r"):
        return True
    return ch in _PT_EXTRA


def _token_is_pt(text: str) -> bool:
    return all(_is_pt_char(c) for c in text)


# ---------------------------------------------------------------------------
# Decodificação byte-level BPE (estilo GPT-2): token string -> bytes
# ---------------------------------------------------------------------------


def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip([chr(c) for c in cs], bs))


_UNICODE_TO_BYTE = _bytes_to_unicode()


def token_to_bytes(token: str) -> bytes:
    """Converte a forma unicode do token BPE para os bytes reais."""
    return bytes(_UNICODE_TO_BYTE[c] for c in token)


# ---------------------------------------------------------------------------
# Keep-set: quais linhas da matriz de embeddings sobrevivem
# ---------------------------------------------------------------------------


def _bytes_plausible_pt(raw: bytes) -> bool:
    """Aceita bytes possíveis em texto PT, incluindo tokens BPE que terminam
    no meio de um caractere UTF-8 multi-byte (ex.: " \\xC3" de " é")."""
    for b in raw:
        if 0x20 <= b <= 0x7E or b in (0x09, 0x0A, 0x0D):
            continue  # ASCII imprimível + whitespace
        if b in (0xC2, 0xC3):
            continue  # lead de U+0080–U+00FF (acentos latinos)
        if b == 0xE2:
            continue  # lead de U+2000–U+2FFF (–—“”…€)
        if 0x80 <= b <= 0xBF:
            continue  # byte de continuação
        return False
    return True


def build_keep_set(tokenizer_json: dict, vocab_size: int) -> list[int]:
    vocab: dict[str, int] = tokenizer_json["model"]["vocab"]
    keep = set()

    # 1. Tokens especiais/adicionados (<|lang_start|>, <|denoise|>, etc.)
    for tok in tokenizer_json.get("added_tokens", []):
        if tok["id"] < vocab_size:
            keep.add(tok["id"])

    n_special = len(keep)
    n_bytes = 0
    n_pt = 0

    for tok_str, tok_id in vocab.items():
        if tok_id >= vocab_size:
            continue  # ids além do embedding do modelo
        try:
            raw = token_to_bytes(tok_str)
        except KeyError:
            keep.add(tok_id)  # não é byte-level puro: mantém por segurança
            continue
        if len(raw) == 1:
            keep.add(tok_id)  # byte-token: fallback universal
            n_bytes += 1
            continue
        try:
            text = raw.decode("utf-8")
            ok = _token_is_pt(text)
        except UnicodeDecodeError:
            # UTF-8 parcial: o BPE divide caracteres acentuados em
            # tokens que isoladamente não decodificam
            ok = _bytes_plausible_pt(raw)
        if ok:
            keep.add(tok_id)
            n_pt += 1

    print(
        f"keep-set: {len(keep)} de {vocab_size} tokens "
        f"({n_special} especiais, {n_bytes} bytes, {n_pt} PT)"
    )
    return sorted(keep)


# ---------------------------------------------------------------------------
# tokenizer.bin: formato compacto para o runtime C
#
#   magic "OVTK" u32 | version u32 (=2) | n_vocab u32 | n_merges u32
#   n_vocab x { u16 len | bytes }            (string do token, forma unicode BPE)
#   n_merges x { u32 left_id | u32 right_id | u32 result_id }
#   n_special u32 | n_special x u32          (ids dos added_tokens, atômicos)
# ---------------------------------------------------------------------------


def write_tokenizer_bin(tokenizer_json: dict, path: Path):
    vocab: dict[str, int] = tokenizer_json["model"]["vocab"]
    merges = tokenizer_json["model"]["merges"]

    id_to_str = {}
    for s, i in vocab.items():
        id_to_str[i] = s
    special_ids = []
    for tok in tokenizer_json.get("added_tokens", []):
        id_to_str[tok["id"]] = tok["content"]
        special_ids.append(tok["id"])
    n_vocab = max(id_to_str) + 1

    merge_triples = []
    for m in merges:
        if isinstance(m, str):
            left, right = m.split(" ", 1)
        else:
            left, right = m
        merged = left + right
        if left in vocab and right in vocab and merged in vocab:
            merge_triples.append((vocab[left], vocab[right], vocab[merged]))

    with open(path, "wb") as f:
        f.write(b"OVTK")
        f.write(struct.pack("<III", 2, n_vocab, len(merge_triples)))
        for i in range(n_vocab):
            s = id_to_str.get(i, "")
            b = s.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
        for l, r, res in merge_triples:
            f.write(struct.pack("<III", l, r, res))
        f.write(struct.pack("<I", len(special_ids)))
        for sid in special_ids:
            f.write(struct.pack("<I", sid))

    print(
        f"tokenizer.bin: {n_vocab} tokens, {len(merge_triples)} merges, "
        f"{len(special_ids)} especiais -> {path}"
    )


# ---------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------


def export_gguf(model_dir: Path, out_dir: Path, keep_ids: list[int], quant: str):
    import gguf
    from safetensors import safe_open

    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    llm = cfg["llm_config"]

    st_path = model_dir / "model.safetensors"
    tensors = {}
    with safe_open(str(st_path), framework="numpy") as st:
        for name in st.keys():
            tensors[name] = st.get_tensor(name)

    print(f"{len(tensors)} tensores carregados de model.safetensors")
    for name in sorted(tensors):
        print(f"  {name}: {tensors[name].shape} {tensors[name].dtype}")

    n_layer = llm["num_hidden_layers"]
    out_path = out_dir / f"omnivoice-pt-{quant.lower()}.gguf"

    w = gguf.GGUFWriter(str(out_path), arch="omnivoice")
    w.add_uint32("omnivoice.n_layer", n_layer)
    w.add_uint32("omnivoice.n_embd", llm["hidden_size"])
    w.add_uint32("omnivoice.n_head", llm["num_attention_heads"])
    w.add_uint32("omnivoice.n_head_kv", llm["num_key_value_heads"])
    w.add_uint32("omnivoice.head_dim", llm["head_dim"])
    w.add_uint32("omnivoice.n_ff", llm["intermediate_size"])
    w.add_float32("omnivoice.rms_norm_eps", llm["rms_norm_eps"])
    w.add_float32("omnivoice.rope_theta", llm.get("rope_theta", 1000000.0))
    w.add_uint32("omnivoice.vocab_size_full", llm["vocab_size"])
    w.add_uint32("omnivoice.vocab_size_kept", len(keep_ids))
    w.add_uint32("omnivoice.audio_vocab_size", cfg["audio_vocab_size"])
    w.add_uint32("omnivoice.audio_mask_id", cfg["audio_mask_id"])
    w.add_uint32("omnivoice.num_audio_codebook", cfg["num_audio_codebook"])

    Q = gguf.GGMLQuantizationType

    def add(name_out: str, data: np.ndarray, qtype):
        data = np.ascontiguousarray(data.astype(np.float32))
        if qtype == Q.F32:
            w.add_tensor(name_out, data, raw_dtype=Q.F32)
        elif qtype == Q.F16:
            w.add_tensor(name_out, data.astype(np.float16), raw_dtype=Q.F16)
        else:
            qdata = gguf.quants.quantize(data, qtype)
            w.add_tensor(name_out, qdata, raw_dtype=qtype)

    if quant.upper() == "Q8_0":
        wq = Q.Q8_0
    elif quant.upper() == "Q4_0":
        wq = Q.Q4_0
    else:
        wq = Q.F16

    # --- tabela de remapeamento (i32, -1 = token podado) ---
    remap = np.full(llm["vocab_size"], -1, dtype=np.int32)
    for new_id, old_id in enumerate(keep_ids):
        remap[old_id] = new_id
    w.add_tensor("token_remap", remap, raw_dtype=Q.I32)

    # --- embeddings de texto podados ---
    embd = tensors["llm.embed_tokens.weight"]
    add("token_embd.weight", embd[keep_ids, :], wq)

    # --- audio ---
    add("audio_embd.weight", tensors["audio_embeddings.weight"], Q.F16)
    add("audio_heads.weight", tensors["audio_heads.weight"], Q.F16)

    # --- camadas ---
    for i in range(n_layer):
        p = f"llm.layers.{i}."
        o = f"blk.{i}."
        add(o + "attn_norm.weight", tensors[p + "input_layernorm.weight"], Q.F32)
        add(o + "attn_q.weight", tensors[p + "self_attn.q_proj.weight"], wq)
        add(o + "attn_k.weight", tensors[p + "self_attn.k_proj.weight"], wq)
        add(o + "attn_v.weight", tensors[p + "self_attn.v_proj.weight"], wq)
        add(o + "attn_output.weight", tensors[p + "self_attn.o_proj.weight"], wq)
        add(o + "attn_q_norm.weight", tensors[p + "self_attn.q_norm.weight"], Q.F32)
        add(o + "attn_k_norm.weight", tensors[p + "self_attn.k_norm.weight"], Q.F32)
        add(o + "ffn_norm.weight", tensors[p + "post_attention_layernorm.weight"], Q.F32)
        add(o + "ffn_gate.weight", tensors[p + "mlp.gate_proj.weight"], wq)
        add(o + "ffn_up.weight", tensors[p + "mlp.up_proj.weight"], wq)
        add(o + "ffn_down.weight", tensors[p + "mlp.down_proj.weight"], wq)

    add("output_norm.weight", tensors["llm.norm.weight"], Q.F32)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    size_mb = out_path.stat().st_size / 1e6
    print(f"GGUF escrito: {out_path} ({size_mb:.0f} MB)")


# ---------------------------------------------------------------------------


# Revisão do k2-fsa/OmniVoice da qual os GGUFs em build/ foram exportados.
# Fixada para que outra máquina baixe exatamente os mesmos pesos: sem isso,
# uma atualização do modelo no HF mudaria o resultado do export em silêncio.
MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument("--revision", default=MODEL_REVISION,
                    help="commit do repo no HF (ignorado se --model for pasta)")
    ap.add_argument("--out", default="build")
    ap.add_argument("--quant", default="Q8_0", choices=["Q8_0", "F16", "Q4_0"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if os.path.isdir(args.model):
        model_dir = Path(args.model)
    else:
        from huggingface_hub import snapshot_download

        model_dir = Path(
            snapshot_download(
                args.model,
                revision=args.revision,
                allow_patterns=["*.json", "*.safetensors", "*.jinja"],
            )
        )
    print(f"modelo em: {model_dir}")

    with open(model_dir / "tokenizer.json", encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    vocab_size = cfg["llm_config"]["vocab_size"]

    keep_ids = build_keep_set(tokenizer_json, vocab_size)
    write_tokenizer_bin(tokenizer_json, out_dir / "tokenizer.bin")
    export_gguf(model_dir, out_dir, keep_ids, args.quant)


if __name__ == "__main__":
    main()
