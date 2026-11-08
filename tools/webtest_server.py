#!/usr/bin/env python3
"""Servidor local para testar lado a lado: runtime C (GGML) vs pipeline
PyTorch original. Digite um texto na página e ouça os dois na hora, com
VRAM e tempo medidos de verdade.

Uso:
    D:\\omnivoice-pt\\.venv\\Scripts\\python.exe tools\\webtest_server.py
Depois abra http://127.0.0.1:5175
"""

import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "omnivoice-src"))

ROOT = Path(r"D:\omnivoice-pt")
BUILD = ROOT / "build"
AUDIO_DIR = BUILD / "webtest_audio"
AUDIO_DIR.mkdir(exist_ok=True)

C_EXE = ROOT / "cmake-build-cuda" / "bin" / "omnivoice-pt.exe"
GGUF_BY_QUANT = {
    "q8_0": BUILD / "omnivoice-pt-q8_0.gguf",
    "f16": BUILD / "omnivoice-pt-f16.gguf",
    "q4_0": BUILD / "omnivoice-pt-q4_0.gguf",
}
DECODER_GGUF = BUILD / "higgs-decoder-f16.gguf"
TOKENIZER_BIN = BUILD / "tokenizer.bin"

# Vozes clonáveis (módulos .ovprompt). "botan" usa a referência curta (~6s),
# bem mais rápida que a completa de 14s com fidelidade equivalente.
VOICE_PROMPTS = {
    "botan": BUILD / "botan_6s.ovprompt",
    "botan_full": BUILD / "botan.ovprompt",
}

# Carrega vozes clonadas adicionais de data/voices/
_VOICES_DIR = ROOT / "data" / "voices"
if _VOICES_DIR.exists():
    _loaded = 0
    for _ovp in sorted(_VOICES_DIR.glob("*.ovprompt")):
        if _ovp.stem not in VOICE_PROMPTS:
            VOICE_PROMPTS[_ovp.stem] = _ovp
            _loaded += 1
    if _loaded:
        print(f"vozes clonadas de data/voices/: {_loaded} "
              f"({', '.join(n for n in VOICE_PROMPTS)})")

SNAPSHOT = (
    r"D:\omnivoice-pt\hf-cache\hub\models--k2-fsa--OmniVoice"
    r"\snapshots\c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
)

app = Flask(__name__)

# ---------------------------------------------------------------------
# Medição de VRAM: cada engine reporta a SUA — nada de nvidia-smi durante
# a geração (o polling serializa a GPU e falseava o tempo dos dois lados).
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Pipeline PyTorch — carregado UMA vez, fica residente na GPU
# ---------------------------------------------------------------------

print("carregando pipeline PyTorch original (fica residente na GPU)...")
import torch  # noqa: E402
from omnivoice.models.omnivoice import OmniVoice  # noqa: E402

_py_model = OmniVoice.from_pretrained(SNAPSHOT, device_map="cuda", dtype=torch.float16)
_py_resident_mb = round(torch.cuda.memory_allocated() / 1e6)
print(f"pipeline PyTorch pronto. VRAM alocada: {_py_resident_mb} MB")

# Clonagem de voz = MÓDULO. O "módulo" é o arquivo .ovprompt (poucos KB) que
# contém só o essencial: tokens da referência + transcrição + volume. É lido
# sob demanda; sem clonagem, nada disso entra na VRAM. Ambos os pipelines usam
# EXATAMENTE a mesma referência (carregada do mesmo .ovprompt) para comparação
# justa. Usamos a referência curta (~6s) por ser bem mais rápida.
import struct as _struct  # noqa: E402

from omnivoice.models.omnivoice import VoiceClonePrompt  # noqa: E402


def load_ovprompt_as_pytorch(path: Path) -> VoiceClonePrompt:
    with open(path, "rb") as f:
        assert f.read(4) == b"OVPR", "formato .ovprompt inválido"
        _ver, n_cb, n_fr, ref_rms = _struct.unpack("<IIIf", f.read(16))
        (text_len,) = _struct.unpack("<I", f.read(4))
        ref_text = f.read(text_len).decode("utf-8")
        toks = np.frombuffer(f.read(n_cb * n_fr * 2), dtype=np.uint16)
    tokens = torch.from_numpy(toks.astype(np.int64).copy()).view(n_cb, n_fr)
    return VoiceClonePrompt(
        ref_audio_tokens=tokens.to(_py_model.device),
        ref_text=ref_text,
        ref_rms=float(ref_rms),
    )


_py_clone = {}
for _name, _path in VOICE_PROMPTS.items():
    try:
        if _path.exists():
            _py_clone[_name] = load_ovprompt_as_pytorch(_path)
            print(f"módulo de clonagem '{_name}' pronto ({_path.name}).")
    except Exception as e:  # noqa: BLE001
        print(f"aviso: clonagem PyTorch '{_name}' indisponivel: {e}")


def run_pytorch(text: str, voice: str = "auto") -> dict:
    out_path = AUDIO_DIR / f"py_{uuid.uuid4().hex[:8]}.wav"
    clone = _py_clone.get(voice)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    if clone is not None:
        audios = _py_model.generate(text=text, language="pt", voice_clone_prompt=clone)
    else:
        audios = _py_model.generate(text=text, language="pt")
    torch.cuda.synchronize()
    gen_s = time.time() - t0
    vram_peak = round(torch.cuda.max_memory_allocated() / 1e6)

    import soundfile as sf

    sf.write(str(out_path), audios[0], _py_model.sampling_rate)
    dur_s = len(audios[0]) / _py_model.sampling_rate
    return {
        "audio_url": f"/audio/{out_path.name}",
        "gen_s": round(gen_s, 2),
        "dur_s": round(dur_s, 2),
        "rtf": round(gen_s / dur_s, 3) if dur_s > 0 else None,
        "vram_mb": vram_peak,
    }


# ---------------------------------------------------------------------
# Runtime C — processo novo a cada chamada (uso real de CLI)
# ---------------------------------------------------------------------

_STDERR_RE = re.compile(
    r"\(([\d.]+)s de audio\).*?"
    r"geracao ([\d.]+)s, decode ([\d.]+)s, RTF ([\d.]+)",
    re.DOTALL,
)
_WEIGHTS_RE = re.compile(r"pesos em (\d+) MB")
_GRAPH_RE = re.compile(r"alocado, (\d+) MB")


def run_c_runtime(
    text: str, quant: str = "q8_0", steps: int = 32, voice: str = "auto", device: str = "gpu", threads: int = 0
) -> dict:
    out_path = AUDIO_DIR / f"c_{uuid.uuid4().hex[:8]}.wav"
    gguf = GGUF_BY_QUANT.get(quant, GGUF_BY_QUANT["q8_0"])
    steps = max(4, min(64, int(steps)))
    args = [
        str(C_EXE),
        "-m", str(gguf),
        "-d", str(DECODER_GGUF),
        "-k", str(TOKENIZER_BIN),
        "-t", text,
        "-o", str(out_path),
        "--steps", str(steps),
    ]
    if device == "gpu":
        args.append("--gpu")
    if threads > 0:
        args += ["--threads", str(threads)]
    prompt = VOICE_PROMPTS.get(voice)
    if prompt and prompt.exists():
        args += ["-p", str(prompt)]
    t0 = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    wall_s = time.time() - t0

    if proc.returncode != 0:
        return {"error": proc.stderr[-2000:] or "falha desconhecida no runtime C"}

    m = _STDERR_RE.search(proc.stderr)
    w = _WEIGHTS_RE.search(proc.stderr)
    gen_s = dec_s = rtf = dur_s = None
    if m:
        dur_s = float(m.group(1))
        gen_s = float(m.group(2))
        dec_s = float(m.group(3))
        rtf = float(m.group(4))
    weights_mb = int(w.group(1)) if w else 0
    # VRAM do modelo = pesos + maior grafo de ativações (o exe reporta ambos)
    graph_mb = max((int(x) for x in _GRAPH_RE.findall(proc.stderr)), default=0)
    vram_mb = weights_mb + graph_mb

    return {
        "audio_url": f"/audio/{out_path.name}",
        "wall_s": round(wall_s, 2),
        "gen_s": gen_s,
        "decode_s": dec_s,
        "dur_s": dur_s,
        "rtf": rtf,
        "vram_mb": vram_mb,
        "weights_mb": weights_mb,
    }


# ---------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------


@app.route("/audio/<path:name>")
def serve_audio(name):
    return send_from_directory(str(AUDIO_DIR), name)


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    engine = data.get("engine")
    if not text:
        return jsonify({"error": "texto vazio"}), 400
    if len(text) > 500:
        return jsonify({"error": "texto longo demais para este teste (max 500 caracteres)"}), 400

    quant = data.get("quant", "q8_0")
    steps = data.get("steps", 32)
    voice = data.get("voice", "auto")
    device = data.get("device", "gpu")
    threads = int(data.get("threads") or 0)
    try:
        if engine == "c":
            result = run_c_runtime(text, quant=quant, steps=steps, voice=voice, device=device, threads=threads)
        elif engine == "py":
            result = run_pytorch(text, voice=voice)
        else:
            return jsonify({"error": "engine invalido"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


@app.route("/")
def index():
    return INDEX_HTML


INDEX_HTML = r"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OmniVoice-PT — teste ao vivo</title>
<style>
  :root {
    --bg: #eef1ec; --bg-raised: #e3e8e1; --ink: #1c231e; --ink-dim: #576155;
    --ink-faint: #8a9186; --line: rgba(28,35,30,.12);
    --ours: #1f8f6e; --ours-strong: #0f7a5c; --ours-tint: rgba(31,143,110,.14);
    --base: #b5591f; --base-strong: #9c4a15; --base-tint: rgba(181,89,31,.14);
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#111510; --bg-raised:#171d16; --ink:#dde5da; --ink-dim:#99a494;
      --ink-faint:#616b5d; --line: rgba(221,229,218,.12);
      --ours:#4fd8ac; --ours-strong:#6ee6bf; --ours-tint: rgba(79,216,172,.14);
      --base:#e8935c; --base-strong:#f0aa7c; --base-tint: rgba(232,147,92,.14); }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font-family: -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; }
  .mono { font-family: ui-monospace,"Cascadia Code","SFMono-Regular",Consolas,monospace; }
  .page { max-width: 860px; margin: 0 auto; padding: 48px 24px 80px; }
  .eyebrow { font-family: ui-monospace,monospace; font-size:12px; letter-spacing:.12em;
    text-transform:uppercase; color:var(--ours); display:flex; align-items:center; gap:10px;
    margin-bottom:16px; }
  .eyebrow::before { content:""; width:7px; height:7px; border-radius:50%; background:var(--ours);
    box-shadow:0 0 0 3px var(--ours-tint); }
  h1 { font-size:30px; line-height:1.2; margin:0 0 12px; font-weight:650; text-wrap:balance; }
  .dek { font-size:15.5px; color:var(--ink-dim); max-width:64ch; margin:0 0 32px; line-height:1.6; }
  textarea { width:100%; min-height:96px; padding:14px; border-radius:10px; border:1px solid var(--line);
    background:var(--bg-raised); color:var(--ink); font-size:15px; line-height:1.5; resize:vertical;
    font-family: inherit; }
  textarea:focus { outline:2px solid var(--ours); outline-offset:1px; }
  .controls { display:flex; gap:18px; margin-top:14px; flex-wrap:wrap; align-items:flex-end; }
  .ctrl { display:flex; flex-direction:column; gap:5px; }
  .ctrl label { font-family: ui-monospace,monospace; font-size:10.5px; letter-spacing:.08em;
    text-transform:uppercase; color:var(--ink-faint); }
  .ctrl select { font-family: inherit; font-size:14px; padding:8px 10px; border-radius:8px;
    border:1px solid var(--line); background:var(--bg-raised); color:var(--ink); cursor:pointer; }
  .ctrl select:focus { outline:2px solid var(--ours); outline-offset:1px; }
  .ctrl .hint { font-size:11px; color:var(--ink-faint); }
  .actions { display:flex; gap:10px; margin-top:14px; flex-wrap:wrap; }
  button { font-family: ui-monospace,monospace; font-size:13px; font-weight:600; letter-spacing:.02em;
    padding:11px 18px; border-radius:8px; border:1px solid var(--line); cursor:pointer;
    background:var(--bg-raised); color:var(--ink); transition: transform .1s; }
  button:hover { transform: translateY(-1px); }
  button:active { transform: translateY(0); }
  button:disabled { opacity:.5; cursor:not-allowed; transform:none; }
  button.primary { background:var(--ours); color:#04130d; border-color:var(--ours); }
  button.secondary { background:var(--base); color:#1a0d04; border-color:var(--base); }
  hr.rule { border:none; border-top:1px solid var(--line); margin:36px 0; }
  .channels { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:640px) { .channels { grid-template-columns:1fr; } }
  .channel { background:var(--bg-raised); border:1px solid var(--line); border-radius:10px;
    padding:18px; min-height:150px; display:flex; flex-direction:column; }
  .channel-head { display:flex; align-items:baseline; justify-content:space-between; margin-bottom:4px; }
  .channel-name { font-weight:650; font-size:15px; }
  .channel-chip { font-family: ui-monospace,monospace; font-size:10.5px; letter-spacing:.06em;
    text-transform:uppercase; padding:2px 7px; border-radius:4px; font-weight:600; }
  .channel.ours .channel-chip { background:var(--ours-tint); color:var(--ours-strong); }
  .channel.base .channel-chip { background:var(--base-tint); color:var(--base-strong); }
  .channel-sub { font-size:12px; color:var(--ink-faint); margin-bottom:14px; }
  .placeholder { color:var(--ink-faint); font-size:13px; flex:1; display:flex; align-items:center;
    justify-content:center; text-align:center; padding:20px 0; }
  .spinner { width:16px; height:16px; border-radius:50%; border:2px solid var(--line);
    border-top-color:var(--ours); animation:spin .7s linear infinite; display:inline-block;
    margin-right:8px; vertical-align:-3px; }
  .channel.base .spinner { border-top-color:var(--base); }
  @keyframes spin { to { transform:rotate(360deg); } }
  audio { width:100%; height:34px; margin-top:auto; }
  table.spec { width:100%; border-collapse:collapse; font-size:13.5px; margin-top:14px; }
  table.spec th { text-align:left; font-family: ui-monospace,monospace; font-size:10.5px;
    letter-spacing:.08em; text-transform:uppercase; color:var(--ink-faint); font-weight:600;
    padding:0 12px 8px 0; border-bottom:1px solid var(--line); }
  table.spec th.num, table.spec td.num { text-align:right; }
  table.spec td { padding:10px 12px 10px 0; border-bottom:1px solid var(--line); }
  table.spec td.num { font-family: ui-monospace,monospace; font-variant-numeric:tabular-nums; font-weight:600; }
  table.spec tr:last-child td { border-bottom:none; }
  .err { color:var(--base-strong); font-size:13px; }
  footer { margin-top:40px; font-size:12px; color:var(--ink-faint); line-height:1.7; }
  footer code { font-family: ui-monospace,monospace; background:var(--bg-raised); padding:1px 5px; border-radius:3px; }
</style>
</head>
<body>
<div class="page">
  <div class="eyebrow">Teste ao vivo · Síntese de voz em PT</div>
  <h1>OmniVoice-PT: digite um texto e compare</h1>
  <p class="dek">Roda os dois pipelines na sua GPU de verdade — runtime C/GGML de um lado,
    pipeline PyTorch original do outro. Cada engine reporta a própria VRAM e o próprio tempo, sem interferência.</p>

  <textarea id="text" maxlength="500">Este é um teste ao vivo do sintetizador de voz em português, rodando direto na placa de vídeo.</textarea>

  <div class="controls">
    <div class="ctrl">
      <label for="voice">Voz (módulo de clonagem)</label>
      <select id="voice">
        <option value="auto" selected>Automática — sem clonagem</option>
""" + "".join(
    f'        <option value="{n}">Clonar: {n.replace("_"," ").title()}</option>\n'
    for n in VOICE_PROMPTS
    if VOICE_PROMPTS[n] is not None
) + r"""      </select>
    </div>
    <div class="ctrl">
      <label for="quant">Qualidade dos pesos (runtime C)</label>
      <select id="quant">
        <option value="f16">F16 — máxima qualidade (~1,1 GB)</option>
        <option value="q8_0" selected>Q8 — rápido e leve (~0,6 GB)</option>
        <option value="q4_0">Q4_0 — ultra leve para CPU (~0,35 GB)</option>
      </select>
    </div>
    <div class="ctrl">
      <label for="device">Dispositivo (runtime C)</label>
      <select id="device">
        <option value="gpu" selected>GPU (CUDA)</option>
        <option value="cpu">CPU (GGML CPU)</option>
      </select>
    </div>
    <div class="ctrl" id="ctrl-threads" style="display:none;">
      <label for="threads">Threads do CPU</label>
      <select id="threads">
        <option value="4">4 threads</option>
        <option value="8">8 threads</option>
        <option value="12">12 threads</option>
        <option value="16" selected>16 threads (Ryzen 9)</option>
        <option value="24">24 threads</option>
        <option value="32">32 threads</option>
      </select>
    </div>
    <div class="ctrl">
      <label for="steps">Passos de difusão (runtime C)</label>
      <select id="steps">
        <option value="16">16 — mais rápido</option>
        <option value="24">24 — equilibrado</option>
        <option value="32" selected>32 — padrão</option>
        <option value="48">48 — mais nítido</option>
        <option value="64">64 — qualidade máxima</option>
      </select>
    </div>
    <div class="ctrl"><span class="hint">mais passos e F16 = mais qualidade,<br>porém um pouco mais lento</span></div>
  </div>

  <div class="actions">
    <button class="primary" id="btn-both">▶ Gerar com os dois</button>
    <button id="btn-c">Só runtime C</button>
    <button id="btn-py">Só PyTorch original</button>
  </div>

  <hr class="rule">

  <div class="channels">
    <div class="channel ours" id="ch-c">
      <div class="channel-head"><span class="channel-name">Runtime C · GGML</span><span class="channel-chip">GPU · Q8_0</span></div>
      <div class="channel-sub">processo novo a cada chamada</div>
      <div class="placeholder" id="ph-c">aguardando geração...</div>
    </div>
    <div class="channel base" id="ch-py">
      <div class="channel-head"><span class="channel-name">Original · PyTorch</span><span class="channel-chip">GPU · FP16</span></div>
      <div class="channel-sub">modelo residente na GPU</div>
      <div class="placeholder" id="ph-py">aguardando geração...</div>
    </div>
  </div>

  <table class="spec" id="spec-table" style="display:none">
    <thead><tr><th>Métrica</th><th class="num">Runtime C</th><th class="num">PyTorch orig.</th></tr></thead>
    <tbody id="spec-body"></tbody>
  </table>

  <footer>
    Servidor local em <code>127.0.0.1:5175</code>. VRAM: pico alocado por cada engine
    (PyTorch via <code>torch.cuda</code>, runtime C = pesos + ativações que ele mesmo reporta).
    Tempo = só a geração+decode, sem contar carga. Nota: as duas engines dividem a mesma GPU
    com o PyTorch residente, então o tempo do runtime C aqui é um pouco pior que rodando sozinho
    na linha de comando (onde marca RTF ~0,24).
  </footer>
</div>

<script>
const state = { c: null, py: null };

function fmt(v, suf) { return (v === null || v === undefined) ? "—" : v + (suf || ""); }

function renderChannel(id, phId, engine, data) {
  const ch = document.getElementById(id);
  const ph = document.getElementById(phId);
  if (data.error) {
    ph.className = "placeholder";
    ph.innerHTML = '<span class="err">Erro: ' + data.error.slice(0, 300) + '</span>';
    return;
  }
  ph.remove();
  let audio = ch.querySelector("audio");
  if (!audio) {
    audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    ch.appendChild(audio);
  }
  audio.src = data.audio_url + "?t=" + Date.now();
  state[engine] = data;
  updateTable();
}

function setLoading(phId) {
  const ph = document.getElementById(phId);
  ph.className = "placeholder";
  ph.innerHTML = '<span class="spinner"></span>gerando...';
}

function updateTable() {
  const c = state.c, py = state.py;
  if (!c && !py) return;
  document.getElementById("spec-table").style.display = "table";
  const rows = [
    ["VRAM do modelo na GPU", c && (c.vram_mb + " MB"), py && (py.vram_mb + " MB")],
    ["Tempo de geração", c && (c.gen_s + " s"), py && (py.gen_s + " s")],
    ["RTF", c && c.rtf, py && py.rtf],
    ["Duração do áudio", c && (c.dur_s + " s"), py && (py.dur_s + " s")],
  ];
  document.getElementById("spec-body").innerHTML = rows.map(r =>
    `<tr><td>${r[0]}</td><td class="num">${fmt(r[1])}</td><td class="num">${fmt(r[2])}</td></tr>`
  ).join("");
}

async function runEngine(engine, chId, phId) {
  setLoading(phId);
  const text = document.getElementById("text").value.trim();
  const quant = document.getElementById("quant").value;
  const steps = parseInt(document.getElementById("steps").value, 10);
  const voice = document.getElementById("voice").value;
  const device = document.getElementById("device").value;
  const threads = document.getElementById("threads").value;
  // reflete a config escolhida no cabeçalho do canal C
  if (engine === "c") {
    const chip = document.querySelector("#ch-c .channel-chip");
    if (chip) {
      const qname = quant === "f16" ? "F16" : (quant === "q8_0" ? "Q8" : "Q4");
      chip.textContent = device.toUpperCase() + " · " + qname + " · " + steps + "p";
    }
  }
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, engine, quant, steps, voice, device, threads }),
    });
    const data = await res.json();
    renderChannel(chId, phId, engine, data);
  } catch (e) {
    renderChannel(chId, phId, engine, { error: String(e) });
  }
}

document.getElementById("btn-c").onclick = () => runEngine("c", "ch-c", "ph-c");
document.getElementById("btn-py").onclick = () => runEngine("py", "ch-py", "ph-py");
document.getElementById("btn-both").onclick = async () => {
  await runEngine("c", "ch-c", "ph-c");
  await runEngine("py", "ch-py", "ph-py");
};
document.getElementById("device").onchange = function() {
  const isCpu = this.value === "cpu";
  document.getElementById("ctrl-threads").style.display = isCpu ? "block" : "none";
};
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("abra http://127.0.0.1:5175")
    app.run(host="127.0.0.1", port=5175, threaded=True)
