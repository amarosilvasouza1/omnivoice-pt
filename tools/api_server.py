#!/usr/bin/env python3
"""API pública do OmniVoice-PT + painel de administração.

Serviço para compartilhar o runtime C (leve, GPU) com outras pessoas na
rede: cada "conexão" tem sua própria chave de API, limites configuráveis
(tamanho de texto, vozes permitidas, passos máximos, taxa por minuto) e
pode ser ativada/desativada a qualquer momento. Todo pedido fica
registrado (texto, tempo, VRAM, RTF, áudio) e aparece ao vivo no painel
/admin.

Uso:
    .venv\\Scripts\\python.exe tools\\api_server.py
Depois:
    - Painel de admin: http://SEU_IP:5180/admin  (senha impressa no console
      na primeira execução, salva em build/admin_token.txt)
    - Amigos usam:     http://SEU_IP:5180/app?key=CHAVE_DA_CONEXAO
    - Ou direto por código: POST http://SEU_IP:5180/api/v1/generate

Não carrega o pipeline PyTorch (fica de fora de propósito: o objetivo
deste serviço é expor a versão otimizada e leve, não a original pesada).
"""

import json
import logging
import os
import queue
import re
import secrets
import socket
import sqlite3
import subprocess
import threading
import time
import uuid
from collections import defaultdict, deque
from html import escape as html_escape
from pathlib import Path

from flask import Flask, g, jsonify, request, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
AUDIO_DIR = BUILD / "api_audio"
AUDIO_DIR.mkdir(exist_ok=True)
DB_PATH = BUILD / "api.db"
ADMIN_TOKEN_PATH = BUILD / "admin_token.txt"
VOICES_DIR = ROOT / "data" / "voices"

GGUF_BY_QUANT = {
    "q8_0": BUILD / "omnivoice-pt-q8_0.gguf",
    "f16": BUILD / "omnivoice-pt-f16.gguf",
    "q4_0": BUILD / "omnivoice-pt-q4_0.gguf",
}
DECODER_GGUF = BUILD / "higgs-decoder-f16.gguf"
TOKENIZER_BIN = BUILD / "tokenizer.bin"

# ---------------------------------------------------------------------
# Configuração do servidor (editável pelo painel /admin, persistida em
# build/server_config.json). Controla: qual GPU/build usar, quantos
# workers rodam em paralelo e se o modelo fica sempre carregado na VRAM.
# ---------------------------------------------------------------------
CONFIG_PATH = BUILD / "server_config.json"

# Builds do runtime C. "universal" (build_cuda_universal.bat) traz código
# nativo da GTX 950 à RTX 50; "default" (build_cuda.bat) só o da placa para a
# qual foi compilado. Na mesma placa os dois geram o mesmo áudio.
BUILD_DIRS = {
    "default": ROOT / "cmake-build-cuda",
    "universal": ROOT / "cmake-build-cuda-universal",
}

DEFAULT_CONFIG = {
    "gpu_target": "auto",     # "auto" | "default" | "universal"
    "workers": 1,             # gerações simultâneas (workers do modelo ativo)
    "keep_loaded": True,      # manter o modelo sempre na VRAM
    "idle_minutes": 10,       # se keep_loaded=False: descarrega após N min ocioso
    "default_quant": "q8_0",  # modelo pré-aquecido no arranque
}


def _detect_gpu_name() -> str:
    """Nome da GPU via nvidia-smi (vazio se não houver/CUDA indisponível)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=6,
        )
        lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
        return lines[0] if lines else ""
    except Exception:  # noqa: BLE001
        return ""


GPU_NAME = _detect_gpu_name()


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            print(f"aviso: server_config.json inválido ({e}); usando padrões")
    # sanidade dos valores
    try:
        cfg["workers"] = max(1, min(4, int(cfg.get("workers", 1))))
    except (TypeError, ValueError):
        cfg["workers"] = 1
    try:
        cfg["idle_minutes"] = max(0, int(cfg.get("idle_minutes", 10)))
    except (TypeError, ValueError):
        cfg["idle_minutes"] = 10
    cfg["keep_loaded"] = bool(cfg.get("keep_loaded", True))
    if cfg.get("gpu_target") not in ("auto", *BUILD_DIRS):
        cfg["gpu_target"] = "auto"
    if cfg.get("default_quant") not in GGUF_BY_QUANT:
        cfg["default_quant"] = "q8_0"
    return cfg


CONFIG = load_config()


def save_config():
    CONFIG_PATH.write_text(
        json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def resolved_gpu_target() -> str:
    """Alvo efetivo: 'auto' prefere o universal, que roda em qualquer placa
    suportada; o default só roda na placa para a qual foi compilado."""
    t = CONFIG.get("gpu_target", "auto")
    if t == "auto":
        universal_exe = BUILD_DIRS["universal"] / "bin" / "omnivoice-pt.exe"
        t = "universal" if universal_exe.exists() else "default"
    return t if t in BUILD_DIRS else "default"


def c_exe() -> Path:
    """Caminho do omnivoice-pt.exe do build compatível com a GPU atual."""
    return BUILD_DIRS[resolved_gpu_target()] / "bin" / "omnivoice-pt.exe"


def workers_for(quant: str) -> int:
    """Quantos workers subir para uma qualidade. Env var (se definida) tem
    prioridade sobre o config; senão usa o número escolhido no painel."""
    env = {
        "q8_0": os.environ.get("OVTTS_WORKERS_Q8"),
        "f16": os.environ.get("OVTTS_WORKERS_F16"),
        "q4_0": os.environ.get("OVTTS_WORKERS_Q4"),
    }.get(quant)
    if env:
        try:
            return max(1, min(4, int(env)))
        except ValueError:
            pass
    return CONFIG["workers"]

# Vozes clonáveis: começa com as fixas (botan) e depois carrega todas de
# data/voices/*.ovprompt automaticamente.
VOICE_PROMPTS = {
    "auto": None,
    "botan": BUILD / "botan_6s.ovprompt",
    "botan_full": BUILD / "botan.ovprompt",
}

# Metadados das vozes clonadas (ref_text, idioma, etc.) — carregado do
# registry.json gerado pelo setup_voices.py.
VOICE_REGISTRY: dict[str, dict] = {}


def _load_clone_voices():
    """Carrega todos os .ovprompt de data/voices/ no VOICE_PROMPTS."""
    if not VOICES_DIR.exists():
        return
    # Carrega registry.json se existir (metadados das vozes)
    registry_path = VOICES_DIR / "registry.json"
    if registry_path.exists():
        try:
            with open(registry_path, encoding="utf-8") as f:
                VOICE_REGISTRY.update(json.load(f))
        except Exception as e:  # noqa: BLE001
            print(f"aviso: não consegui ler {registry_path}: {e}")
    # Carrega cada .ovprompt encontrado
    loaded = 0
    for ovp in sorted(VOICES_DIR.glob("*.ovprompt")):
        name = ovp.stem
        if name not in VOICE_PROMPTS:  # não sobrescreve botan etc.
            VOICE_PROMPTS[name] = ovp
            loaded += 1
    if loaded:
        print(f"vozes clonadas carregadas de data/voices/: {loaded} "
              f"({', '.join(n for n in VOICE_PROMPTS if VOICE_PROMPTS[n] is not None)})")


_load_clone_voices()

app = Flask(__name__)
# Atrás do cloudflared/ngrok, a conexão real é HTTPS até a borda do túnel e
# HTTP só no laço local; sem isso, request.is_secure sempre daria False e o
# cookie de admin marcado "secure" nunca seria aceito de volta pelo navegador.
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_for=1, x_host=1)

# Teto de qualquer requisição. O maior caso legítimo é o WAV de clonagem
# (uma referência de 1 min em 48 kHz estéreo tem ~11 MB). Sem isso, uma chave
# de amigo conseguia encher o disco com um único upload.
MAX_UPLOAD_MB = 20
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.errorhandler(413)
def _too_large(_e):
    return jsonify({"error": f"arquivo grande demais (máx. {MAX_UPLOAD_MB} MB)"}), 413


class _RedactApiKeys(logging.Filter):
    """Tira as chaves (?key=...) das linhas de acesso do werkzeug. O link do
    amigo é /app?key=CHAVE, e sem isso toda chave ficava gravada nos logs."""

    _rx = re.compile(r"([?&]key=)[^&\s\"]+")

    def filter(self, record):
        msg = record.getMessage()
        red = self._rx.sub(r"\1***", msg)
        if red != msg:
            record.msg, record.args = red, ()
        return True


logging.getLogger("werkzeug").addFilter(_RedactApiKeys())

# ---------------------------------------------------------------------
# Banco de dados (SQLite): conexões (chaves de API) + histórico de pedidos
# ---------------------------------------------------------------------


def db():
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            max_chars INTEGER NOT NULL DEFAULT 300,
            max_steps INTEGER NOT NULL DEFAULT 32,
            allowed_quant TEXT NOT NULL DEFAULT 'q8_0,f16,q4_0',
            allowed_voices TEXT NOT NULL DEFAULT '*',
            rate_limit_per_min INTEGER NOT NULL DEFAULT 6,
            notes TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            connection_id INTEGER REFERENCES connections(id) ON DELETE CASCADE,
            ts REAL NOT NULL,
            text TEXT NOT NULL,
            voice TEXT,
            quant TEXT,
            steps INTEGER,
            gen_s REAL,
            decode_s REAL,
            rtf REAL,
            dur_s REAL,
            vram_mb INTEGER,
            audio_file TEXT,
            error TEXT,
            client_ip TEXT
        );
        """
    )
    conn.execute("UPDATE connections SET allowed_quant = allowed_quant || ',q4_0' WHERE allowed_quant NOT LIKE '%q4_0%'")
    conn.commit()

    # Conexão padrão "amigos" na primeira execução, para já ter algo a
    # compartilhar sem precisar abrir o admin primeiro.
    cur = conn.execute("SELECT COUNT(*) FROM connections")
    if cur.fetchone()[0] == 0:
        key = "ov_" + secrets.token_urlsafe(18)
        conn.execute(
            "INSERT INTO connections "
            "(name, api_key, enabled, max_chars, max_steps, allowed_quant, "
            " allowed_voices, rate_limit_per_min, notes, created_at) "
            "VALUES (?, ?, 1, 300, 32, 'q8_0,f16,q4_0', '*', 6, "
            " 'conexão padrão, criada automaticamente', ?)",
            ("amigos", key, time.time()),
        )
        conn.commit()
        print(f"conexão padrão 'amigos' criada. chave: {key}")

    # Migração: conexões antigas ficaram travadas em 'auto,botan' (criadas
    # antes das vozes clonadas existirem). Libera todas com o coringa '*'
    # para que os usuários já cadastrados vejam as vozes novas. Restrições
    # específicas (uma lista sem '*') são preservadas.
    migrated = conn.execute(
        "UPDATE connections SET allowed_voices = '*' "
        "WHERE allowed_voices = 'auto,botan'"
    ).rowcount
    conn.commit()
    if migrated:
        print(f"migração: {migrated} conexão(ões) antiga(s) liberada(s) "
              f"para todas as vozes ('*').")

    conn.close()


def get_admin_token() -> str:
    if ADMIN_TOKEN_PATH.exists():
        return ADMIN_TOKEN_PATH.read_text(encoding="utf-8").strip()
    token = secrets.token_urlsafe(24)
    ADMIN_TOKEN_PATH.write_text(token, encoding="utf-8")
    return token


ADMIN_TOKEN = get_admin_token()

# ---------------------------------------------------------------------
# Limite de taxa em memória: {api_key: deque[timestamps]}
# ---------------------------------------------------------------------

_rate_state: dict[str, deque] = defaultdict(deque)


def check_rate_limit(bucket: str, limit: int, window_s: float = 60) -> bool:
    now = time.time()
    dq = _rate_state[bucket]
    while dq and now - dq[0] > window_s:
        dq.popleft()
    if len(dq) >= limit:
        return False
    dq.append(now)
    return True


# Limites que valem para toda conexão, além dos configuráveis no painel.
# Trocar de modelo derruba e reergue os workers de TODO mundo; clonar roda o
# Whisper num lock global. Os dois precisavam de freio próprio.
MODEL_SWITCHES_PER_MIN = 2
UPLOADS_PER_HOUR = 5
MAX_PRIVATE_VOICES = 10
MAX_REF_TEXT_CHARS = 500
MAX_REF_SECONDS = 30.0


# ---------------------------------------------------------------------
# Geração via POOL DE WORKERS. Cada worker é um processo omnivoice-pt --serve
# com o modelo residente na VRAM, processando um job por vez. N workers =
# N gerações simultâneas; quando todos estão ocupados, os pedidos ficam numa
# fila (bloqueiam em free.get()) até um worker liberar. Isto substitui o
# antigo "um processo novo por pedido".
# ---------------------------------------------------------------------

# Quantos workers por qualidade vem de workers_for() (painel /admin ou env).
# O servidor mantém somente UMA qualidade carregada por vez
# (ensure_quant_loaded); pedidos extras entram na fila.


class Worker:
    """Um processo omnivoice-pt --serve com o modelo já carregado."""

    def __init__(self, wid: int, quant: str):
        self.wid = wid
        self.quant = quant
        self.jobs_done = 0
        self.vram_mb = 0
        self.busy = False
        self.alive = False
        self._start()

    def _start(self):
        gguf = GGUF_BY_QUANT.get(self.quant, GGUF_BY_QUANT["q8_0"])
        args = [
            str(c_exe()),
            "-m", str(gguf),
            "-d", str(DECODER_GGUF),
            "-k", str(TOKENIZER_BIN),
        ]
        device = os.environ.get("OVTTS_DEVICE_Q4", "cpu").lower()
        if self.quant != "q4_0" or device == "gpu":
            args.append("--gpu")
        args.append("--serve")
        # stderr num arquivo por worker (para depurar sem encher o pipe)
        self._logf = open(BUILD / f"worker_{self.quant}_{self.wid}.log", "wb")
        self.proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._logf, bufsize=0,
        )
        # espera a linha READY\t<vram_mb>
        line = self.proc.stdout.readline().decode("utf-8", "replace").strip()
        if line.startswith("READY"):
            parts = line.split("\t")
            self.vram_mb = int(parts[1]) if len(parts) > 1 else 0
            self.alive = True
        else:
            self.alive = False

    def generate(self, text, steps, seed, prompt_path, out_path) -> dict:
        if not self.alive or self.proc.poll() is not None:
            return {"error": "worker morto"}
        # o protocolo é uma linha por job: TAB separa os campos, texto por
        # último — troca TAB/quebras por espaço (o modelo já filtra \n mesmo)
        clean = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
        job = f"{out_path}\t{steps}\t{seed}\t{prompt_path or '-'}\t{clean}\n"
        try:
            self.proc.stdin.write(job.encode("utf-8"))
            self.proc.stdin.flush()
            resp = self.proc.stdout.readline().decode("utf-8", "replace").strip()
        except (BrokenPipeError, OSError):
            self.alive = False
            return {"error": "worker caiu durante a geração"}
        if not resp:
            self.alive = False
            return {"error": "worker fechou a conexão"}
        parts = resp.split("\t")
        if parts[0] == "OK" and len(parts) >= 6:
            self.jobs_done += 1
            return {
                "gen_s": float(parts[1]),
                "decode_s": float(parts[2]),
                "dur_s": float(parts[3]),
                "rtf": float(parts[4]),
                "vram_mb": int(parts[5]),
            }
        return {"error": parts[1] if len(parts) > 1 else "erro desconhecido"}

    def stop(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
        except OSError:
            pass


class WorkerPool:
    """N workers de uma dada qualidade; distribui jobs e enfileira o excesso."""

    def __init__(self, quant: str, n: int):
        self.quant = quant
        self.workers = []
        self._free = queue.Queue()
        self._waiting = 0  # pedidos bloqueados esperando um worker livre
        self._wlock = threading.Lock()
        for i in range(n):
            w = Worker(i, quant)
            self.workers.append(w)
            if w.alive:
                self._free.put(w)
        self.n_alive = sum(1 for w in self.workers if w.alive)
        print(f"pool '{quant}': {self.n_alive}/{n} workers vivos "
              f"({self.workers[0].vram_mb if self.workers else 0} MB cada)")

    def submit(self, text, steps, seed, prompt_path, out_path) -> dict:
        # free.get() BLOQUEIA quando todos ocupados => é a fila automática
        with self._wlock:
            self._waiting += 1
        w = self._free.get()
        with self._wlock:
            self._waiting -= 1
        try:
            w.busy = True
            result = w.generate(text, steps, seed, prompt_path, out_path)
            # se o worker morreu, tenta reerguer para não degradar o pool
            if not w.alive:
                try:
                    w._start()
                except Exception:  # noqa: BLE001
                    pass
            return result
        finally:
            w.busy = False
            self._free.put(w)

    def status(self) -> dict:
        return {
            "quant": self.quant,
            "workers": len(self.workers),
            "alive": sum(1 for w in self.workers if w.alive),
            "busy": sum(1 for w in self.workers if w.busy),
            "queue_waiting": self._waiting,
            "vram_mb_each": self.workers[0].vram_mb if self.workers else 0,
            "jobs_done": sum(w.jobs_done for w in self.workers),
        }

    def shutdown(self):
        """Encerra todos os workers deste pool (libera a VRAM)."""
        for w in self.workers:
            w.stop()
        self.workers = []
        try:
            while True:
                self._free.get_nowait()
        except queue.Empty:
            pass


_pools: dict[str, WorkerPool] = {}
_active_quant: str | None = None
_pools_lock = threading.Lock()

# Última atividade (para o descarregador de ociosidade quando keep_loaded=off).
_last_activity = time.time()
_activity_lock = threading.Lock()


def _touch_activity():
    global _last_activity
    with _activity_lock:
        _last_activity = time.time()


def ensure_quant_loaded(quant: str) -> WorkerPool:
    """Garante que SÓ o modelo `quant` está carregado. Se outro estava ativo,
    descarrega-o antes (libera a VRAM) — um único modelo por vez, como pedido.
    Trocar de modelo é raro (uma vez, quando o usuário escolhe), então o custo
    de subir/descer os workers é aceitável."""
    global _active_quant
    if quant not in GGUF_BY_QUANT:
        quant = "q8_0"
    with _pools_lock:
        # descarrega qualquer outro modelo ativo
        for q in list(_pools.keys()):
            if q != quant:
                print(f"descarregando modelo {q.upper()} (liberando VRAM)...")
                _pools[q].shutdown()
                del _pools[q]
        if quant not in _pools:
            print(f"carregando modelo {quant.upper()} ({workers_for(quant)} worker(s))...")
            _pools[quant] = WorkerPool(quant, workers_for(quant))
        _active_quant = quant
        return _pools[quant]


def unload_all_pools():
    """Derruba todos os workers (libera a VRAM). Próximo pedido recarrega."""
    global _active_quant
    with _pools_lock:
        for q in list(_pools.keys()):
            _pools[q].shutdown()
            del _pools[q]
        _active_quant = None


def reload_active_pool():
    """Reergue o pool ativo — usado para aplicar nº de workers/GPU novos."""
    global _active_quant
    with _pools_lock:
        active = _active_quant
        for q in list(_pools.keys()):
            _pools[q].shutdown()
            del _pools[q]
        _active_quant = None
    if active:
        ensure_quant_loaded(active)


def _idle_monitor():
    """Se keep_loaded=off, descarrega o modelo após idle_minutes sem uso."""
    while True:
        time.sleep(20)
        if CONFIG["keep_loaded"] or CONFIG["idle_minutes"] <= 0:
            continue
        with _pools_lock:
            has_pool = bool(_pools)
            any_busy = any(w.busy for p in _pools.values() for w in p.workers)
        if not has_pool or any_busy:
            continue
        with _activity_lock:
            idle_s = time.time() - _last_activity
        if idle_s >= CONFIG["idle_minutes"] * 60:
            print(f"ocioso há {int(idle_s)}s — descarregando modelo "
                  f"(keep_loaded=off)")
            unload_all_pools()


def active_quant() -> str | None:
    return _active_quant


def run_generation(text: str, quant: str, steps: int, voice: str) -> dict:
    out_path = AUDIO_DIR / f"{uuid.uuid4().hex[:12]}.wav"
    prompt = VOICE_PROMPTS.get(voice)
    prompt_str = str(prompt) if (prompt and prompt.exists()) else "-"
    _touch_activity()
    pool = ensure_quant_loaded(quant)  # troca de modelo se necessário
    seed = int.from_bytes(os.urandom(4), "little")
    result = pool.submit(text, steps, seed, prompt_str, str(out_path))
    _touch_activity()
    if "error" not in result:
        result["audio_file"] = out_path.name
    return result


# ---------------------------------------------------------------------
# API pública (chave por conexão)
# ---------------------------------------------------------------------


def get_connection_by_key(api_key: str):
    row = db().execute(
        "SELECT * FROM connections WHERE api_key = ?", (api_key,)
    ).fetchone()
    return row


def log_request(conn_id, text, voice, quant, steps, result, client_ip):
    db().execute(
        "INSERT INTO requests "
        "(connection_id, ts, text, voice, quant, steps, gen_s, decode_s, "
        " rtf, dur_s, vram_mb, audio_file, error, client_ip) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            conn_id,
            time.time(),
            text,
            voice,
            quant,
            steps,
            result.get("gen_s"),
            result.get("decode_s"),
            result.get("rtf"),
            result.get("dur_s"),
            result.get("vram_mb"),
            result.get("audio_file"),
            result.get("error"),
            client_ip,
        ),
    )
    db().commit()


# Vozes privadas: o nome interno começa com "c<conn_id>__". São visíveis e
# usáveis SÓ pela conexão dona (o amigo que enviou). Vozes globais (sem "__")
# aparecem para todos conforme allowed_voices/'*'.


def is_private_voice(name: str) -> bool:
    return "__" in name and name.startswith("c")


def private_prefix(conn_id: int) -> str:
    return f"c{conn_id}__"


def all_voice_names() -> list[str]:
    """Vozes GLOBAIS (auto + clonadas públicas). Exclui as privadas."""
    return [n for n in VOICE_PROMPTS if not is_private_voice(n)]


def resolve_allowed_voices(conn) -> list[str]:
    """Vozes que esta conexão pode usar: as globais (por allowed_voices/'*')
    MAIS as privadas dela própria (que ela mesma enviou)."""
    raw = [v.strip() for v in conn["allowed_voices"].split(",") if v.strip()]
    globals_ = all_voice_names() if "*" in raw else [v for v in raw if not is_private_voice(v)]
    pref = private_prefix(conn["id"])
    mine = [n for n in VOICE_PROMPTS if n.startswith(pref)]
    return globals_ + mine


@app.route("/api/v1/generate", methods=["POST"])
def api_generate():
    api_key = request.headers.get("X-API-Key") or request.args.get("key")
    if not api_key:
        return jsonify({"error": "chave de API ausente (header X-API-Key ou ?key=)"}), 401

    conn = get_connection_by_key(api_key)
    if not conn:
        return jsonify({"error": "chave de API inválida"}), 403
    if not conn["enabled"]:
        return jsonify({"error": "esta conexão está desativada pelo administrador"}), 403

    if not check_rate_limit(api_key, conn["rate_limit_per_min"]):
        return jsonify({"error": "limite de pedidos por minuto excedido, aguarde"}), 429

    data = request.get_json(force=True, silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "corpo deve ser um objeto JSON"}), 400
    raw_text = data.get("text")
    text = raw_text.strip() if isinstance(raw_text, str) else ""
    if not text:
        return jsonify({"error": "campo 'text' vazio"}), 400
    if len(text) > conn["max_chars"]:
        return jsonify(
            {"error": f"texto excede o limite desta conexão ({conn['max_chars']} caracteres)"}
        ), 400

    allowed_quant = conn["allowed_quant"].split(",")
    # sem quant explícito: usa o modelo já carregado (evita trocas acidentais);
    # se o ativo não for permitido p/ esta conexão, cai no primeiro permitido.
    quant = data.get("quant")
    if not quant:
        quant = active_quant() if active_quant() in allowed_quant else allowed_quant[0]
    if quant not in allowed_quant:
        return jsonify({"error": f"quant '{quant}' não permitido nesta conexão"}), 400
    # pedir outro quant com um modelo já carregado = troca para todo mundo
    if active_quant() is not None and quant != active_quant():
        if not check_rate_limit(f"model:{api_key}", MODEL_SWITCHES_PER_MIN):
            return jsonify(
                {"error": "trocas de modelo demais por minuto, aguarde"}
            ), 429

    allowed_voices = resolve_allowed_voices(conn)
    voice = data.get("voice") or (allowed_voices[0] if allowed_voices else "auto")
    # Clientes de IA frequentemente enviam o label exibido (ex.: "Nescau")
    # em vez do identificador interno ("nescau"). Resolve sem diferenciar
    # maiúsculas/minúsculas, mas preserva o nome canônico usado nos arquivos.
    if isinstance(voice, str):
        voice_by_case = {name.casefold(): name for name in allowed_voices}
        voice = voice_by_case.get(voice.strip().casefold(), voice.strip())
    if voice not in allowed_voices:
        return jsonify({"error": f"voz '{voice}' não permitida nesta conexão"}), 400

    try:
        steps = int(data.get("steps") or conn["max_steps"])
    except (TypeError, ValueError):
        return jsonify({"error": "campo 'steps' deve ser um número inteiro"}), 400
    steps = max(4, min(conn["max_steps"], steps))

    # ProxyFix já resolveu o IP real (último salto do túnel); o cabeçalho cru
    # traz também o que o cliente quiser inventar.
    client_ip = request.remote_addr
    result = run_generation(text, quant, steps, voice)
    log_request(conn["id"], text, voice, quant, steps, result, client_ip)

    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    return jsonify(
        {
            "audio_url": f"/api/v1/audio/{result['audio_file']}",
            "gen_s": result["gen_s"],
            "decode_s": result["decode_s"],
            "dur_s": result["dur_s"],
            "rtf": result["rtf"],
            "vram_mb": result["vram_mb"],
        }
    )


@app.route("/api/v1/voices", methods=["GET"])
def api_voices():
    api_key = request.headers.get("X-API-Key") or request.args.get("key")
    conn = get_connection_by_key(api_key) if api_key else None
    if not conn:
        return jsonify({"error": "chave de API inválida"}), 403
    allowed = resolve_allowed_voices(conn)
    # Monta lista de vozes com metadados do registry quando disponível
    voices_detail = []
    for v in allowed:
        info = VOICE_REGISTRY.get(v, {})
        voices_detail.append({
            "name": v,
            "label": _voice_label(v),
            "ref_text": info.get("ref_text"),
            "language": info.get("language"),
        })
    return jsonify(
        {
            "voices": allowed,
            "voices_detail": voices_detail,
            "quant": conn["allowed_quant"].split(","),
            "max_chars": conn["max_chars"],
            "max_steps": conn["max_steps"],
        }
    )


@app.route("/api/v1/model", methods=["GET"])
def api_get_model():
    """Diz qual modelo está carregado agora."""
    api_key = request.headers.get("X-API-Key") or request.args.get("key")
    conn = get_connection_by_key(api_key) if api_key else None
    if not conn:
        return jsonify({"error": "chave de API inválida"}), 403
    return jsonify(
        {
            "active": active_quant(),
            "available": conn["allowed_quant"].split(","),
        }
    )


@app.route("/api/v1/model", methods=["POST"])
def api_set_model():
    """Carrega o modelo escolhido (q8_0 ou f16). Descarrega o outro para
    liberar VRAM — só um modelo fica ativo por vez."""
    api_key = request.headers.get("X-API-Key") or request.args.get("key")
    conn = get_connection_by_key(api_key) if api_key else None
    if not conn:
        return jsonify({"error": "chave de API inválida"}), 403
    if not conn["enabled"]:
        return jsonify({"error": "conexão desativada"}), 403

    data = request.get_json(force=True, silent=True) or {}
    if not isinstance(data, dict):
        data = {}
    quant = data.get("quant") or data.get("model") or ""
    quant = quant.strip().lower() if isinstance(quant, str) else ""
    if quant in ("fp16", "f16", "16"):
        quant = "f16"
    elif quant in ("q8", "q8_0", "8"):
        quant = "q8_0"
    allowed = conn["allowed_quant"].split(",")
    if quant not in allowed:
        return jsonify(
            {"error": f"modelo '{quant}' não disponível. Opções: {allowed}"}
        ), 400
    if quant != active_quant() and not check_rate_limit(
        f"model:{api_key}", MODEL_SWITCHES_PER_MIN
    ):
        return jsonify({"error": "trocas de modelo demais por minuto, aguarde"}), 429

    # troca o modelo (carrega o pedido, descarrega o outro)
    pool = ensure_quant_loaded(quant)
    return jsonify(
        {
            "ok": True,
            "active": quant,
            "workers": len(pool.workers),
            "vram_mb_each": pool.workers[0].vram_mb if pool.workers else 0,
            "mensagem": f"modelo {quant.upper()} carregado; os outros foram descarregados.",
        }
    )


def _voice_label(name: str) -> str:
    """Label amigável para uma voz (usado no HTML e na API)."""
    labels = {
        "auto": "Automática",
        "botan": "Clonar: Botan (rápido)",
        "botan_full": "Clonar: Botan (fidelidade)",
    }
    if name in labels:
        return labels[name]
    if is_private_voice(name):  # "c12__minhavoz" -> "Minha voz: Minhavoz"
        short = name.split("__", 1)[1]
        return f"Minha voz: {short.replace('_', ' ').title()}"
    # Vozes do registry: capitaliza o nome
    display = name.replace("_", " ").title()
    return f"Clonar: {display}"


@app.route("/api/v1/clone-voices", methods=["GET"])
def api_clone_voices():
    """Lista todas as vozes clonadas disponíveis no sistema."""
    api_key = request.headers.get("X-API-Key") or request.args.get("key")
    conn = get_connection_by_key(api_key) if api_key else None
    if not conn:
        return jsonify({"error": "chave de API inválida"}), 403

    allowed = set(resolve_allowed_voices(conn))
    voices = []
    for name in allowed:
        path = VOICE_PROMPTS.get(name)
        if name == "auto" or path is None:
            continue
        info = VOICE_REGISTRY.get(name, {})
        voices.append({
            "name": name,
            "label": _voice_label(name),
            "language": info.get("language", "pt"),
            "private": is_private_voice(name),
            "prompt_size_kb": round(path.stat().st_size / 1024, 1)
            if path.exists() else None,
        })
    return jsonify({"voices": voices})


_upload_lock = threading.Lock()


@app.route("/api/v1/clone-voices/upload", methods=["POST"])
def api_clone_upload():
    """Envia um WAV e a API clona a voz. multipart/form-data:
      - audio: arquivo WAV (obrigatório)
      - voice_name: apelido da voz (letras/números/_) (obrigatório)
      - ref_text: transcrição do áudio (OPCIONAL — se vazio, transcreve sozinho)
      - max_ref_seconds: (opcional) encurtar referência (padrão 8)

    Autenticação: a chave da conexão (X-API-Key ou ?key=). A voz criada fica
    PRIVADA — só a conexão que enviou consegue ver e usar. (Admin cria voz
    global, sem prefixo.)
    """
    api_key = request.headers.get("X-API-Key") or request.args.get("key")
    conn = get_connection_by_key(api_key) if api_key else None
    is_admin = is_admin_request()
    if not conn and not is_admin:
        return jsonify({"error": "chave de API inválida"}), 403
    if conn and not conn["enabled"]:
        return jsonify({"error": "conexão desativada"}), 403

    raw_name = (request.form.get("voice_name") or "").strip().lower()
    ref_text = (request.form.get("ref_text") or "").strip()
    max_ref_s = request.form.get("max_ref_seconds", "8")

    if not raw_name:
        return jsonify({"error": "campo 'voice_name' obrigatório"}), 400
    if "audio" not in request.files:
        return jsonify({"error": "arquivo 'audio' (WAV) obrigatório"}), 400
    if not re.fullmatch(r"[a-z0-9_]{1,40}", raw_name):
        return jsonify(
            {"error": "voice_name deve ter até 40 letras minúsculas, números e _"}
        ), 400
    if len(ref_text) > MAX_REF_TEXT_CHARS:
        return jsonify(
            {"error": f"ref_text longo demais (máx. {MAX_REF_TEXT_CHARS} caracteres)"}
        ), 400

    # nome interno: privado por conexão (prefixo), ou global se for admin
    voice_name = raw_name if is_admin and not conn else f"{private_prefix(conn['id'])}{raw_name}"

    if conn:
        pref = private_prefix(conn["id"])
        n_mine = sum(1 for n in VOICE_PROMPTS if n.startswith(pref))
        if voice_name not in VOICE_PROMPTS and n_mine >= MAX_PRIVATE_VOICES:
            return jsonify(
                {"error": f"limite de {MAX_PRIVATE_VOICES} vozes por conexão; "
                          "reenvie com o nome de uma voz sua para substituí-la"}
            ), 400
        if not check_rate_limit(f"upload:{conn['id']}", UPLOADS_PER_HOUR, 3600):
            return jsonify(
                {"error": f"limite de {UPLOADS_PER_HOUR} clonagens por hora, aguarde"}
            ), 429

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = VOICES_DIR / f"{voice_name}.wav"
    out_path = VOICES_DIR / f"{voice_name}.ovprompt"
    # o .ovprompt é gerado à parte e só substitui o atual se der certo: uma
    # reclonagem que falha não pode apagar a voz que já funcionava. A subpasta
    # fica fora do glob("*.ovprompt") que carrega as vozes no arranque.
    tmp_dir = VOICES_DIR / "_upload_tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_id = uuid.uuid4().hex
    tmp_wav = tmp_dir / f"{tmp_id}.wav"
    tmp_out = tmp_dir / f"{tmp_id}.ovprompt"

    request.files["audio"].save(str(tmp_wav))

    make_prompt = ROOT / "tools" / "make_prompt.py"
    python_exe = ROOT / ".venv" / "Scripts" / "python.exe"
    args = [
        str(python_exe), str(make_prompt),
        "--ref_audio", str(tmp_wav),
        "--out", str(tmp_out),
        "--asr_model", "openai/whisper-small",  # usado se ref_text vazio
    ]
    if ref_text:
        args += ["--ref_text", ref_text]
    # referência longa deixa a geração lenta para todos e infla a duração da
    # fala (ver PLANO.md); sem valor válido, usa o padrão documentado (8 s)
    try:
        max_s = float(max_ref_s)
    except (TypeError, ValueError):
        max_s = 8.0
    if not (0 < max_s <= MAX_REF_SECONDS):  # também pega NaN
        max_s = MAX_REF_SECONDS
    args += ["--max_ref_seconds", str(max_s)]

    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8",
                 "HF_HOME": str(ROOT / "hf-cache")}
    # serializa uploads: make_prompt usa a GPU/Whisper; evita dois ao mesmo tempo
    with _upload_lock:
        proc = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=child_env,
        )
    if proc.returncode != 0 or not tmp_out.exists():
        tmp_wav.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)
        # traceback completo só no console; para o cliente vai a última linha
        # (a mensagem do erro), sem caminhos internos do servidor
        print(f"[clone {voice_name}] make_prompt falhou:\n{proc.stderr or ''}")
        lines = [ln for ln in (proc.stderr or "").splitlines() if ln.strip()]
        reason = lines[-1][-200:] if lines else "erro desconhecido"
        return jsonify({"error": f"falha ao clonar a voz: {reason}"}), 500
    os.replace(tmp_wav, wav_path)
    os.replace(tmp_out, out_path)

    # Registra em memória (fica disponível na hora, sem reiniciar)
    VOICE_PROMPTS[voice_name] = out_path
    VOICE_REGISTRY[voice_name] = {
        "name": voice_name, "language": "pt", "ref_text": ref_text or "(auto)",
        "prompt_file": out_path.name, "prompt_exists": True,
        "prompt_size_kb": round(out_path.stat().st_size / 1024, 1),
        "owner": conn["id"] if conn else None,
    }
    registry_path = VOICES_DIR / "registry.json"
    try:
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(VOICE_REGISTRY, f, ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001
        pass

    return jsonify({
        "ok": True,
        "voice": voice_name,        # use este nome no campo "voice" ao gerar
        "label": _voice_label(voice_name),
        "mensagem": "voz clonada e pronta para usar (privada, só sua).",
    })


@app.route("/api/v1/audio/<path:name>")
def api_audio(name):
    return send_from_directory(str(AUDIO_DIR), name)


# ---------------------------------------------------------------------
# Página simples para amigos usarem (form + player), sem precisar de código
# ---------------------------------------------------------------------


@app.route("/app")
def friend_app():
    return FRIEND_HTML


@app.route("/docs")
def docs_page():
    # host_url vem do cabeçalho Host / X-Forwarded-Host: escapa antes de
    # colar no HTML
    base_url = html_escape(request.host_url.rstrip("/"))
    return DOCS_HTML.replace("__BASE_URL__", base_url)


# ---------------------------------------------------------------------
# Admin: autenticação simples por token compartilhado (cookie de sessão)
# ---------------------------------------------------------------------

_admin_sessions: set[str] = set()
# Freio contra força bruta no login: poucas tentativas por IP a cada 5 min.
# Importante agora que o painel pode ficar exposto por um túnel público.
_admin_login_attempts: dict[str, deque] = defaultdict(deque)
ADMIN_LOGIN_MAX_ATTEMPTS = 5
ADMIN_LOGIN_WINDOW_S = 300


def is_admin_request() -> bool:
    sid = request.cookies.get("ov_admin_sid")
    return sid in _admin_sessions


@app.route("/admin/login", methods=["POST"])
def admin_login():
    ip = request.remote_addr or "desconhecido"
    now = time.time()
    attempts = _admin_login_attempts[ip]
    while attempts and now - attempts[0] > ADMIN_LOGIN_WINDOW_S:
        attempts.popleft()
    if len(attempts) >= ADMIN_LOGIN_MAX_ATTEMPTS:
        return jsonify({"error": "muitas tentativas — aguarde alguns minutos"}), 429

    data = request.get_json(force=True, silent=True) or {}
    token = data.get("token") if isinstance(data, dict) else None
    # compare_digest: o tempo da comparação não revela quantos caracteres batem
    if not isinstance(token, str) or not secrets.compare_digest(
        token.encode("utf-8"), ADMIN_TOKEN.encode("utf-8")
    ):
        attempts.append(now)
        return jsonify({"error": "token incorreto"}), 403

    attempts.clear()
    sid = secrets.token_urlsafe(24)
    _admin_sessions.add(sid)
    resp = jsonify({"ok": True})
    resp.set_cookie(
        "ov_admin_sid", sid, httponly=True, samesite="Lax", secure=request.is_secure
    )
    return resp


def require_admin():
    if not is_admin_request():
        return jsonify({"error": "não autenticado"}), 401
    return None


@app.route("/admin")
def admin_page():
    return ADMIN_HTML


@app.route("/admin/api/connections", methods=["GET"])
def admin_list_connections():
    if (err := require_admin()) is not None:
        return err
    rows = db().execute(
        "SELECT c.*, "
        " (SELECT COUNT(*) FROM requests r WHERE r.connection_id = c.id) AS n_requests, "
        " (SELECT MAX(ts) FROM requests r WHERE r.connection_id = c.id) AS last_used "
        "FROM connections c ORDER BY c.id"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/admin/api/connections", methods=["POST"])
def admin_create_connection():
    if (err := require_admin()) is not None:
        return err
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "novo amigo").strip()
    key = "ov_" + secrets.token_urlsafe(18)
    db().execute(
        "INSERT INTO connections "
        "(name, api_key, enabled, max_chars, max_steps, allowed_quant, "
        " allowed_voices, rate_limit_per_min, notes, created_at) "
        "VALUES (?, ?, 1, 300, 32, 'q8_0,f16,q4_0', '*', 6, '', ?)",
        (name, key, time.time()),
    )
    db().commit()
    return jsonify({"ok": True, "api_key": key})


@app.route("/admin/api/connections/<int:conn_id>", methods=["PATCH"])
def admin_update_connection(conn_id):
    if (err := require_admin()) is not None:
        return err
    data = request.get_json(force=True, silent=True) or {}
    fields = []
    values = []
    for key in (
        "name", "enabled", "max_chars", "max_steps",
        "allowed_quant", "allowed_voices", "rate_limit_per_min", "notes",
    ):
        if key in data:
            fields.append(f"{key} = ?")
            values.append(data[key])
    if not fields:
        return jsonify({"error": "nada para atualizar"}), 400
    values.append(conn_id)
    db().execute(f"UPDATE connections SET {', '.join(fields)} WHERE id = ?", values)
    db().commit()
    return jsonify({"ok": True})


@app.route("/admin/api/connections/<int:conn_id>", methods=["DELETE"])
def admin_delete_connection(conn_id):
    if (err := require_admin()) is not None:
        return err
    db().execute("DELETE FROM connections WHERE id = ?", (conn_id,))
    db().commit()
    return jsonify({"ok": True})


@app.route("/admin/api/requests", methods=["GET"])
def admin_list_requests():
    if (err := require_admin()) is not None:
        return err
    since = request.args.get("since", type=float, default=0)
    limit = request.args.get("limit", type=int, default=100)
    rows = db().execute(
        "SELECT r.*, c.name AS connection_name FROM requests r "
        "JOIN connections c ON c.id = r.connection_id "
        "WHERE r.ts > ? ORDER BY r.ts DESC LIMIT ?",
        (since, limit),
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/admin/api/stats", methods=["GET"])
def admin_stats():
    if (err := require_admin()) is not None:
        return err
    total = db().execute("SELECT COUNT(*) AS n FROM requests").fetchone()["n"]
    errors = db().execute(
        "SELECT COUNT(*) AS n FROM requests WHERE error IS NOT NULL"
    ).fetchone()["n"]
    active_conns = db().execute(
        "SELECT COUNT(*) AS n FROM connections WHERE enabled = 1"
    ).fetchone()["n"]
    avg_rtf = db().execute(
        "SELECT AVG(rtf) AS v FROM requests WHERE rtf IS NOT NULL "
        "AND ts > ?", (time.time() - 3600,)
    ).fetchone()["v"]
    return jsonify(
        {
            "total_requests": total,
            "errors": errors,
            "active_connections": active_conns,
            "avg_rtf_last_hour": round(avg_rtf, 3) if avg_rtf else None,
        }
    )


@app.route("/admin/api/pools", methods=["GET"])
def admin_pools():
    if (err := require_admin()) is not None:
        return err
    with _pools_lock:
        pools = [p.status() for p in _pools.values()]
    return jsonify({"pools": pools})


@app.route("/admin/api/config", methods=["GET"])
def admin_get_config():
    if (err := require_admin()) is not None:
        return err
    target = resolved_gpu_target()
    exe = c_exe()
    return jsonify({
        "config": CONFIG,
        "gpu_name": GPU_NAME or "desconhecida",
        "resolved_target": target,
        "exe": str(exe),
        "exe_exists": exe.exists(),
        "active_quant": _active_quant,
        "targets": list(BUILD_DIRS.keys()),
    })


@app.route("/admin/api/config", methods=["POST"])
def admin_set_config():
    if (err := require_admin()) is not None:
        return err
    data = request.get_json(force=True, silent=True) or {}
    old = dict(CONFIG)
    if "workers" in data:
        try:
            CONFIG["workers"] = max(1, min(4, int(data["workers"])))
        except (TypeError, ValueError):
            pass
    if "keep_loaded" in data:
        CONFIG["keep_loaded"] = bool(data["keep_loaded"])
    if "idle_minutes" in data:
        try:
            CONFIG["idle_minutes"] = max(0, int(data["idle_minutes"]))
        except (TypeError, ValueError):
            pass
    if data.get("gpu_target") in ("auto", *BUILD_DIRS):
        CONFIG["gpu_target"] = data["gpu_target"]
    if data.get("default_quant") in GGUF_BY_QUANT:
        CONFIG["default_quant"] = data["default_quant"]
    save_config()

    # aplica ao vivo
    with _pools_lock:
        active = _active_quant
    need_reload = (
        CONFIG["workers"] != old["workers"]
        or CONFIG["gpu_target"] != old["gpu_target"]
    )
    if need_reload and active:
        reload_active_pool()
    elif CONFIG["keep_loaded"] and not active:
        # ligou "manter carregado" e nada estava carregado: pré-aquece
        ensure_quant_loaded(CONFIG["default_quant"])
    _touch_activity()

    return jsonify({
        "ok": True,
        "config": CONFIG,
        "resolved_target": resolved_gpu_target(),
        "exe_exists": c_exe().exists(),
    })


# ---------------------------------------------------------------------
# HTML — mesma linguagem visual dos outros paineis deste projeto
# ---------------------------------------------------------------------

_STYLE = r"""
:root {
  --bg: #eef1ec; --bg-raised: #e3e8e1; --ink: #1c231e; --ink-dim: #576155;
  --ink-faint: #8a9186; --line: rgba(28,35,30,.12);
  --ours: #1f8f6e; --ours-strong: #0f7a5c; --ours-tint: rgba(31,143,110,.14);
  --base: #b5591f; --base-strong: #9c4a15; --base-tint: rgba(181,89,31,.14);
  --bad: #b5301f; --bad-tint: rgba(181,48,31,.14);
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#111510; --bg-raised:#171d16; --ink:#dde5da; --ink-dim:#99a494;
    --ink-faint:#616b5d; --line: rgba(221,229,218,.12);
    --ours:#4fd8ac; --ours-strong:#6ee6bf; --ours-tint: rgba(79,216,172,.14);
    --base:#e8935c; --base-strong:#f0aa7c; --base-tint: rgba(232,147,92,.14);
    --bad:#e85c4c; --bad-tint: rgba(232,92,76,.16); }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font-family: -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; }
.mono { font-family: ui-monospace,"Cascadia Code","SFMono-Regular",Consolas,monospace; }
.eyebrow { font-family: ui-monospace,monospace; font-size:12px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--ours); display:flex; align-items:center; gap:10px;
  margin-bottom:16px; }
.eyebrow::before { content:""; width:7px; height:7px; border-radius:50%; background:var(--ours);
  box-shadow:0 0 0 3px var(--ours-tint); }
h1 { font-size:26px; line-height:1.2; margin:0 0 10px; font-weight:650; text-wrap:balance; }
.dek { font-size:14.5px; color:var(--ink-dim); max-width:70ch; margin:0 0 28px; line-height:1.6; }
input, select, textarea { font-family: inherit; font-size:14px; padding:8px 10px; border-radius:8px;
  border:1px solid var(--line); background:var(--bg-raised); color:var(--ink); }
input:focus, select:focus, textarea:focus { outline:2px solid var(--ours); outline-offset:1px; }
button { font-family: ui-monospace,monospace; font-size:12.5px; font-weight:600; letter-spacing:.02em;
  padding:9px 15px; border-radius:8px; border:1px solid var(--line); cursor:pointer;
  background:var(--bg-raised); color:var(--ink); }
button:hover { filter:brightness(1.05); }
button.primary { background:var(--ours); color:#04130d; border-color:var(--ours); }
button.danger { background:var(--bad-tint); color:var(--bad); border-color:var(--bad); }
button:disabled { opacity:.5; cursor:not-allowed; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; font-family: ui-monospace,monospace; font-size:10.5px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--ink-faint); font-weight:600; padding:0 10px 8px 0;
  border-bottom:1px solid var(--line); }
td { padding:9px 10px 9px 0; border-bottom:1px solid var(--line); vertical-align:middle; }
.pill { display:inline-block; padding:2px 8px; border-radius:99px; font-family: ui-monospace,monospace;
  font-size:10.5px; font-weight:600; }
.pill.on { background:var(--ours-tint); color:var(--ours-strong); }
.pill.off { background:var(--bad-tint); color:var(--bad); }
.card { background:var(--bg-raised); border:1px solid var(--line); border-radius:10px; padding:16px; }
.hint { font-size:11.5px; color:var(--ink-faint); }
.key { font-family: ui-monospace,monospace; font-size:11.5px; background:var(--bg-raised);
  padding:2px 6px; border-radius:4px; cursor:pointer; }
"""


LOGIN_HTML = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Admin — OmniVoice-PT</title><style>__STYLE__
.wrap{max-width:380px;margin:15vh auto;padding:0 24px}
</style></head><body><div class="wrap">
<div class="eyebrow">Painel de administração</div>
<h1>Entrar</h1>
<p class="dek">Cole o token salvo em <code class="mono">build/admin_token.txt</code>
(também impresso no console ao iniciar o servidor).</p>
<input id="token" type="password" placeholder="token de admin" style="width:100%;margin-bottom:10px">
<button class="primary" id="go" style="width:100%">Entrar</button>
<p id="err" class="hint" style="color:var(--bad-tint,#b5301f);margin-top:10px"></p>
<script>
document.getElementById('go').onclick = async () => {
  const token = document.getElementById('token').value;
  const res = await fetch('/admin/login', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({token})});
  if (res.ok) location.reload();
  else document.getElementById('err').textContent = 'token incorreto';
};
document.getElementById('token').addEventListener('keydown', e => { if(e.key==='Enter') document.getElementById('go').click(); });
</script></div></body></html>"""


ADMIN_HTML_BODY = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OmniVoice-PT — Admin</title><style>__STYLE__
.page{max-width:1180px;margin:0 auto;padding:40px 24px 80px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px}
.stat{background:var(--bg-raised);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.stat .v{font-family:ui-monospace,monospace;font-size:22px;font-weight:700}
.stat .l{font-size:11px;color:var(--ink-faint);text-transform:uppercase;letter-spacing:.06em;margin-top:2px}
.section{margin-bottom:36px}
.section-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.section-head h2{font-size:15px;margin:0;font-weight:650}
.conn-row td{vertical-align:top}
.conn-edit{display:none;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:8px;padding:12px;
  background:var(--bg);border-radius:8px}
.conn-edit.open{display:grid}
.conn-edit label{display:block;font-size:10.5px;color:var(--ink-faint);text-transform:uppercase;
  letter-spacing:.06em;margin-bottom:4px}
.conn-edit input,.conn-edit select{width:100%}
.req-row.error{background:var(--bad-tint)}
.req-text{max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
audio{height:30px;width:220px}
.actions-row{display:flex;gap:6px}
.new-conn{display:flex;gap:8px;align-items:center}
.copyhint{font-size:10.5px;color:var(--ink-faint)}
.pools{display:flex;flex-direction:column;gap:14px}
.pool{background:var(--bg-raised);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.pool-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.pool-title{font-weight:650;font-size:14px}
.pool-title span{font-family:ui-monospace,monospace;font-size:11px;color:var(--ink-faint);font-weight:400}
.workers{display:flex;gap:8px;flex-wrap:wrap}
.wk{width:52px;height:52px;border-radius:9px;border:1px solid var(--line);display:flex;
  flex-direction:column;align-items:center;justify-content:center;gap:2px;font-family:ui-monospace,monospace}
.wk .wi{font-size:15px;font-weight:700}
.wk .ws{font-size:8.5px;text-transform:uppercase;letter-spacing:.04em}
.wk.free{background:var(--ours-tint);color:var(--ours-strong);border-color:var(--ours)}
.wk.busy{background:var(--base-tint);color:var(--base-strong);border-color:var(--base)}
.wk.dead{background:var(--bad-tint);color:var(--bad);border-color:var(--bad)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}
.wk.busy{animation:pulse 1.1s ease-in-out infinite}
.queue-badge{font-family:ui-monospace,monospace;font-size:12px;padding:3px 9px;border-radius:99px;
  background:var(--bg);border:1px solid var(--line)}
.queue-badge.hot{background:var(--base-tint);color:var(--base-strong);border-color:var(--base)}
.cfg-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;background:var(--bg-raised);
  border:1px solid var(--line);border-radius:10px;padding:16px}
@media(max-width:720px){.cfg-grid{grid-template-columns:1fr}}
.cfg-item{display:flex;flex-direction:column;gap:5px}
.cfg-item label{font-size:10.5px;color:var(--ink-faint);text-transform:uppercase;letter-spacing:.06em}
.cfg-item select,.cfg-item input{width:100%}
.cfg-actions{grid-column:1/-1;flex-direction:row;align-items:center;gap:12px}
.cfg-badge{font-family:ui-monospace,monospace;font-size:11px;padding:2px 8px;border-radius:99px;
  background:var(--ours-tint);color:var(--ours-strong);border:1px solid var(--ours)}
.cfg-badge.warn{background:var(--bad-tint);color:var(--bad);border-color:var(--bad)}
</style></head><body><div class="page">
  <div class="eyebrow">Painel de administração · OmniVoice-PT</div>
  <h1>Conexões e atividade em tempo real</h1>
  <p class="dek">Cada conexão tem sua própria chave — compartilhe o link
  <code class="mono">/app?key=...</code> com um amigo. Ative, desative ou
  ajuste limites de qualquer uma a qualquer momento.</p>

  <div class="stats" id="stats"></div>

  <div class="section">
    <div class="section-head"><h2>Workers (geração paralela)</h2>
      <span class="hint">cada worker tem o modelo na VRAM · atualiza a cada 2s</span></div>
    <div class="pools" id="pools"></div>
  </div>

  <div class="section">
    <div class="section-head"><h2>Configuração do servidor</h2>
      <span class="hint" id="cfg-gpu">—</span></div>
    <div class="cfg-grid">
      <div class="cfg-item">
        <label>GPU alvo (runtime C)</label>
        <select id="cfg-target">
          <option value="auto">Automático (detecta a GPU)</option>
          <option value="universal">Universal (GTX 950 a RTX 50)</option>
          <option value="default">Build desta máquina (build_cuda.bat)</option>
        </select>
      </div>
      <div class="cfg-item">
        <label>Workers (gerações simultâneas)</label>
        <select id="cfg-workers">
          <option value="1">1 — leve, menos VRAM</option>
          <option value="2">2</option>
          <option value="3">3</option>
          <option value="4">4 — máximo paralelo</option>
        </select>
      </div>
      <div class="cfg-item">
        <label>Manter modelo sempre carregado</label>
        <select id="cfg-keep">
          <option value="1">Sim — resposta instantânea (ocupa VRAM)</option>
          <option value="0">Não — libera a VRAM quando ocioso</option>
        </select>
      </div>
      <div class="cfg-item" id="cfg-idle-wrap">
        <label>Descarregar após ocioso (min)</label>
        <input id="cfg-idle" type="number" min="0" step="1" value="10">
      </div>
      <div class="cfg-item">
        <label>Modelo padrão (no arranque)</label>
        <select id="cfg-quant">
          <option value="q8_0">Q8 — rápido e leve</option>
          <option value="f16">F16 — máxima qualidade</option>
          <option value="q4_0">Q4 — ultraleve</option>
        </select>
      </div>
      <div class="cfg-item cfg-actions">
        <button class="primary" id="btn-save-cfg">Salvar e aplicar</button>
        <span class="hint" id="cfg-status"></span>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-head">
      <h2>Conexões</h2>
      <div class="new-conn">
        <input id="new-name" placeholder="nome do amigo" style="width:160px">
        <button class="primary" id="btn-new-conn">+ Nova conexão</button>
      </div>
    </div>
    <table>
      <thead><tr><th>Nome</th><th>Chave</th><th>Status</th><th>Pedidos</th>
        <th>Último uso</th><th>Ações</th></tr></thead>
      <tbody id="conn-body"></tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-head"><h2>Atividade recente</h2>
      <span class="hint">atualiza a cada 3s</span></div>
    <table>
      <thead><tr><th>Quando</th><th>Conexão</th><th>Texto</th><th>Voz</th>
        <th>Áudio</th><th>Geração</th><th>RTF</th><th>VRAM</th></tr></thead>
      <tbody id="req-body"></tbody>
    </table>
  </div>
</div>
<script>
// Tudo que vem do servidor passa por esc() antes de entrar em innerHTML: o
// texto dos pedidos é escrito pelos amigos e, sem isso, um "texto" com
// <img onerror=...> rodaria JavaScript aqui, com a sessão de admin.
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, ch =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[ch]);
}
function fmtAgo(ts) {
  const s = Math.round(Date.now()/1000 - ts);
  if (s < 60) return s + "s atrás";
  if (s < 3600) return Math.round(s/60) + "min atrás";
  return Math.round(s/3600) + "h atrás";
}
function copyKey(k) { navigator.clipboard.writeText(k); }

async function loadStats() {
  const s = await (await fetch('/admin/api/stats')).json();
  document.getElementById('stats').innerHTML = [
    ['Pedidos totais', s.total_requests],
    ['Erros', s.errors],
    ['Conexões ativas', s.active_connections],
    ['RTF médio (1h)', s.avg_rtf_last_hour ?? '—'],
  ].map(([l,v]) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
}

async function loadPools() {
  const data = await (await fetch('/admin/api/pools')).json();
  if (!data.pools || !data.pools.length) {
    document.getElementById('pools').innerHTML =
      '<div class="hint">nenhum pool ativo ainda — o pool sobe no primeiro pedido.</div>';
    return;
  }
  document.getElementById('pools').innerHTML = data.pools.map(p => {
    // reconstrói os slots: N workers, "busy" primeiro, resto "free"
    const slots = [];
    for (let i = 0; i < p.workers; i++) {
      const state = i < (p.workers - p.alive) ? 'dead' : (i < p.busy ? 'busy' : 'free');
      const label = state === 'busy' ? 'gerando' : (state === 'dead' ? 'morto' : 'livre');
      slots.push(`<div class="wk ${state}"><div class="wi">${i+1}</div><div class="ws">${label}</div></div>`);
    }
    const qcls = p.queue_waiting > 0 ? 'queue-badge hot' : 'queue-badge';
    return `<div class="pool">
      <div class="pool-head">
        <div class="pool-title">Pool ${p.quant.toUpperCase()}
          <span>· ${p.alive} workers · ${p.vram_mb_each} MB cada · ${p.jobs_done} gerações</span></div>
        <div class="${qcls}">${p.busy}/${p.workers} ocupados${p.queue_waiting ? ' · '+p.queue_waiting+' na fila' : ''}</div>
      </div>
      <div class="workers">${slots.join('')}</div>
    </div>`;
  }).join('');
}

let _cfgLoaded = false;
function syncIdleVisibility() {
  document.getElementById('cfg-idle-wrap').style.display =
    document.getElementById('cfg-keep').value === '0' ? 'flex' : 'none';
}
async function loadConfig() {
  const d = await (await fetch('/admin/api/config')).json();
  const c = d.config;
  document.getElementById('cfg-target').value = c.gpu_target;
  document.getElementById('cfg-workers').value = c.workers;
  document.getElementById('cfg-keep').value = c.keep_loaded ? '1' : '0';
  document.getElementById('cfg-idle').value = c.idle_minutes;
  document.getElementById('cfg-quant').value = c.default_quant;
  syncIdleVisibility();
  const warn = d.exe_exists ? '' : ' warn';
  const exeMsg = d.exe_exists ? '' : ' · binário NÃO encontrado (compile este build!)';
  document.getElementById('cfg-gpu').innerHTML =
    `GPU: <b>${esc(d.gpu_name)}</b> · build ativo <span class="cfg-badge${warn}">${esc(d.resolved_target)}</span>${exeMsg}`;
  _cfgLoaded = true;
}
document.getElementById('cfg-keep').onchange = syncIdleVisibility;
document.getElementById('btn-save-cfg').onclick = async () => {
  const btn = document.getElementById('btn-save-cfg');
  const status = document.getElementById('cfg-status');
  btn.disabled = true; status.textContent = 'aplicando (pode reerguer os workers)...';
  try {
    const body = {
      gpu_target: document.getElementById('cfg-target').value,
      workers: parseInt(document.getElementById('cfg-workers').value, 10),
      keep_loaded: document.getElementById('cfg-keep').value === '1',
      idle_minutes: parseInt(document.getElementById('cfg-idle').value, 10) || 0,
      default_quant: document.getElementById('cfg-quant').value,
    };
    const r = await (await fetch('/admin/api/config', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    })).json();
    status.textContent = r.ok ? 'salvo ✓' : 'erro ao salvar';
    await loadConfig();
    loadPools();
  } catch (e) {
    status.textContent = 'erro: ' + e;
  } finally {
    btn.disabled = false;
    setTimeout(() => { status.textContent = ''; }, 4000);
  }
};

async function loadConnections() {
  const conns = await (await fetch('/admin/api/connections')).json();
  document.getElementById('conn-body').innerHTML = conns.map(c => `
    <tr class="conn-row">
      <td><b>${esc(c.name)}</b><br><span class="hint">${esc(c.notes)}</span></td>
      <td><span class="key" onclick="copyKey(this.textContent)" title="clicar para copiar">${esc(c.api_key)}</span>
        <div class="copyhint">/app?key=${esc(c.api_key)}</div></td>
      <td><span class="pill ${c.enabled ? 'on':'off'}">${c.enabled ? 'ativa':'desativada'}</span></td>
      <td>${c.n_requests}</td>
      <td>${c.last_used ? fmtAgo(c.last_used) : 'nunca'}</td>
      <td class="actions-row">
        <button onclick="toggleConn(${c.id}, ${c.enabled?0:1})">${c.enabled?'Desativar':'Ativar'}</button>
        <button onclick="toggleEdit(${c.id})">Config</button>
        <button class="danger" onclick="delConn(${c.id})">Remover</button>
      </td>
    </tr>
    <tr><td colspan="6">
      <div class="conn-edit" id="edit-${c.id}">
        <div><label>Máx. caracteres</label><input id="mc-${c.id}" type="number" value="${c.max_chars}"></div>
        <div><label>Máx. passos</label><input id="ms-${c.id}" type="number" value="${c.max_steps}"></div>
        <div><label>Pedidos/min</label><input id="rl-${c.id}" type="number" value="${c.rate_limit_per_min}"></div>
        <div><label>Quant. permitidas</label><input id="aq-${c.id}" value="${esc(c.allowed_quant)}"></div>
        <div><label>Vozes permitidas</label><input id="av-${c.id}" value="${esc(c.allowed_voices)}"></div>
        <div><label>Notas</label><input id="nt-${c.id}" value="${esc(c.notes)}"></div>
        <div style="grid-column:1/-1"><button class="primary" onclick="saveConn(${c.id})">Salvar</button></div>
      </div>
    </td></tr>
  `).join('');
}

function toggleEdit(id) { document.getElementById('edit-'+id).classList.toggle('open'); }

async function toggleConn(id, enabled) {
  await fetch('/admin/api/connections/'+id, {method:'PATCH', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({enabled})});
  loadConnections();
}
async function saveConn(id) {
  const body = {
    max_chars: parseInt(document.getElementById('mc-'+id).value,10),
    max_steps: parseInt(document.getElementById('ms-'+id).value,10),
    rate_limit_per_min: parseInt(document.getElementById('rl-'+id).value,10),
    allowed_quant: document.getElementById('aq-'+id).value,
    allowed_voices: document.getElementById('av-'+id).value,
    notes: document.getElementById('nt-'+id).value,
  };
  await fetch('/admin/api/connections/'+id, {method:'PATCH', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)});
  loadConnections();
}
async function delConn(id) {
  if (!confirm('Remover esta conexão? A chave deixará de funcionar.')) return;
  await fetch('/admin/api/connections/'+id, {method:'DELETE'});
  loadConnections();
}
document.getElementById('btn-new-conn').onclick = async () => {
  const name = document.getElementById('new-name').value.trim() || 'novo amigo';
  const res = await fetch('/admin/api/connections', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name})});
  const data = await res.json();
  document.getElementById('new-name').value = '';
  loadConnections();
  alert('Conexão criada!\\nChave: ' + data.api_key + '\\n\\nLink para o amigo:\\n' +
    location.origin + '/app?key=' + data.api_key);
};

async function loadRequests() {
  const reqs = await (await fetch('/admin/api/requests?limit=60')).json();
  document.getElementById('req-body').innerHTML = reqs.map(r => `
    <tr class="req-row ${r.error ? 'error':''}">
      <td class="hint">${fmtAgo(r.ts)}</td>
      <td>${esc(r.connection_name)}</td>
      <td class="req-text" title="${esc(r.text)}">${r.error ? esc('erro: '+r.error.slice(0,80)) : esc(r.text)}</td>
      <td>${esc(r.voice||'—')}</td>
      <td>${r.audio_file ? `<audio controls preload="none" src="/api/v1/audio/${encodeURIComponent(r.audio_file)}"></audio>` : '—'}</td>
      <td class="mono">${r.gen_s ?? '—'}s</td>
      <td class="mono">${r.rtf ?? '—'}</td>
      <td class="mono">${r.vram_mb ?? '—'} MB</td>
    </tr>
  `).join('');
}

async function tick() { await Promise.all([loadStats(), loadPools(), loadConnections(), loadRequests()]); }
tick();
loadConfig();  // uma vez no arranque; não fica recarregando p/ não atrapalhar edição
setInterval(() => { loadStats(); loadRequests(); }, 3000);
setInterval(loadPools, 2000);  // workers atualizam mais rápido para ver o "ao vivo"
</script></body></html>"""


FRIEND_HTML_BODY = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OmniVoice-PT</title><style>__STYLE__
.page{max-width:640px;margin:0 auto;padding:44px 24px 70px}
textarea{width:100%;min-height:100px;resize:vertical;line-height:1.5}
.row{display:flex;gap:10px;margin-top:12px;flex-wrap:wrap}
.row select{flex:1;min-width:140px}
.result{margin-top:22px;padding:16px;border-radius:10px;background:var(--bg-raised);
  border:1px solid var(--line);display:none}
.result.show{display:block}
.meta{display:flex;gap:16px;margin-top:10px;font-size:12px;color:var(--ink-faint)}
.meta b{color:var(--ink);font-family:ui-monospace,monospace}
audio{width:100%;margin-top:10px}
.err{color:var(--bad,#b5301f);font-size:13px;margin-top:12px}
.badkey{max-width:480px;margin:15vh auto;padding:0 24px;text-align:center}
</style></head><body>
<div id="root"></div>
<script>
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, ch =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[ch]);
}
const params = new URLSearchParams(location.search);
const key = params.get('key') || '';
const root = document.getElementById('root');

async function init() {
  if (!key) {
    root.innerHTML = `<div class="badkey"><div class="eyebrow">OmniVoice-PT</div>
      <h1>Link incompleto</h1><p class="dek">Peça ao administrador o link completo
      com <code class="mono">?key=...</code>.</p></div>`;
    return;
  }
  // a chave vai no cabeçalho, não na URL: URLs acabam em logs
  const info = await (await fetch('/api/v1/voices', {headers: {'X-API-Key': key}})).json();
  if (info.error) {
    root.innerHTML = `<div class="badkey"><div class="eyebrow">OmniVoice-PT</div>
      <h1>Acesso negado</h1><p class="dek">${esc(info.error)}</p></div>`;
    return;
  }
  // Monta labels a partir dos metadados da API (voices_detail)
  const voiceLabels = {};
  if (info.voices_detail) {
    info.voices_detail.forEach(v => { voiceLabels[v.name] = v.label; });
  } else {
    // Fallback se a API não retornar voices_detail
    Object.assign(voiceLabels, {auto:'Automática', botan:'Clonar: Botan (rápido)', botan_full:'Clonar: Botan (fidelidade)'});
  }
  const quantLabels = {q8_0:'Rápido (Q8)', f16:'Máxima qualidade (F16)', q4_0:'Otimizado CPU (Q4)'};
  root.innerHTML = `<div class="page">
    <div class="eyebrow">OmniVoice-PT · síntese de voz em PT</div>
    <h1>Gerar fala</h1>
    <p class="dek">Digite o texto e clique em gerar. Roda direto no host (GPU/CPU).</p>
    <textarea id="text" maxlength="${info.max_chars}" placeholder="Digite até ${info.max_chars} caracteres..."></textarea>
    <div class="row">
      <select id="voice">${info.voices.map(v=>`<option value="${esc(v)}">${esc(voiceLabels[v]||v)}</option>`).join('')}</select>
      <select id="quant">${info.quant.map(q=>`<option value="${esc(q)}">${esc(quantLabels[q]||q)}</option>`).join('')}</select>
      <button class="primary" id="go">▶ Gerar</button>
    </div>
    <div class="result" id="result"></div>
  </div>`;

  document.getElementById('go').onclick = async () => {
    const text = document.getElementById('text').value.trim();
    if (!text) return;
    const btn = document.getElementById('go');
    btn.disabled = true; btn.textContent = 'gerando...';
    const res = await fetch('/api/v1/generate', {
      method: 'POST', headers: {'Content-Type':'application/json', 'X-API-Key': key},
      body: JSON.stringify({
        text,
        voice: document.getElementById('voice').value,
        quant: document.getElementById('quant').value,
      }),
    });
    const data = await res.json();
    const box = document.getElementById('result');
    box.className = 'result show';
    if (data.error) {
      box.innerHTML = `<div class="err">${esc(data.error)}</div>`;
    } else {
      box.innerHTML = `<audio controls autoplay src="${esc(data.audio_url)}"></audio>
        <div class="meta">
          <span>geração <b>${esc(data.gen_s)}s</b></span>
          <span>RTF <b>${esc(data.rtf)}</b></span>
          <span>VRAM <b>${esc(data.vram_mb)} MB</b></span>
        </div>`;
    }
    btn.disabled = false; btn.textContent = '▶ Gerar';
  };
}
init();
</script></body></html>"""


DOCS_HTML_BODY = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Como usar — OmniVoice-PT</title><style>__STYLE__
.page{max-width:760px;margin:0 auto;padding:44px 24px 90px}
.dek{margin-bottom:36px}
nav.toc{display:flex;flex-wrap:wrap;gap:8px 18px;margin-bottom:40px;padding:14px 16px;
  background:var(--bg-raised);border:1px solid var(--line);border-radius:10px}
nav.toc a{font-family:ui-monospace,monospace;font-size:12px;color:var(--ink-dim);
  text-decoration:none}
nav.toc a:hover{color:var(--ours-strong)}
section{margin-bottom:44px;scroll-margin-top:20px}
section h2{font-size:19px;margin:0 0 6px;font-weight:650}
section .lede{font-size:13.5px;color:var(--ink-dim);margin:0 0 18px;line-height:1.6;max-width:64ch}
.step{display:flex;gap:14px;margin-bottom:16px;align-items:flex-start}
.step .n{font-family:ui-monospace,monospace;font-size:12px;font-weight:700;color:var(--ours-strong);
  background:var(--ours-tint);width:24px;height:24px;border-radius:50%;display:flex;
  align-items:center;justify-content:center;flex:none;margin-top:1px}
.step p{margin:0;font-size:14px;line-height:1.55}
.code{position:relative;background:var(--bg-raised);border:1px solid var(--line);border-radius:9px;
  padding:14px 16px;font-family:ui-monospace,monospace;font-size:12.5px;line-height:1.65;
  overflow-x:auto;white-space:pre;margin:10px 0}
.code .c1{color:var(--ink-faint)}
.copy-btn{position:absolute;top:8px;right:8px;font-size:10px;padding:4px 8px;border-radius:6px}
table.ref{width:100%;border-collapse:collapse;font-size:13px;margin:14px 0}
table.ref th{text-align:left;font-family:ui-monospace,monospace;font-size:10.5px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--ink-faint);font-weight:600;padding:0 12px 8px 0;
  border-bottom:1px solid var(--line)}
table.ref td{padding:10px 12px 10px 0;border-bottom:1px solid var(--line);vertical-align:top}
table.ref td.mono{font-family:ui-monospace,monospace;font-size:12.5px;white-space:nowrap}
.badge{display:inline-block;font-family:ui-monospace,monospace;font-size:10.5px;font-weight:700;
  padding:2px 7px;border-radius:5px}
.badge.get{background:var(--ours-tint);color:var(--ours-strong)}
.badge.post{background:var(--base-tint);color:var(--base-strong)}
.route-head{display:flex;align-items:center;gap:10px;margin:22px 0 4px}
.route-head code{font-family:ui-monospace,monospace;font-size:14px;font-weight:600}
.route-head:first-of-type{margin-top:0}
.note{font-size:12.5px;color:var(--ink-faint);background:var(--bg-raised);border:1px solid var(--line);
  border-left:3px solid var(--ours);border-radius:0 8px 8px 0;padding:10px 14px;margin:14px 0;line-height:1.6}
</style></head><body><div class="page">
  <div class="eyebrow">Documentação · OmniVoice-PT</div>
  <h1>Como usar</h1>
  <p class="dek">Um sintetizador de voz em português rodando na GPU deste computador.
  Se você só quer gerar áudio, vá direto na seção 1. Se curte programar, a
  seção 3 tem a referência completa da API.</p>

  <nav class="toc">
    <a href="#simples">1. Uso simples</a>
    <a href="#autenticacao">2. Sua chave</a>
    <a href="#api">3. Referência da API</a>
    <a href="#limites">4. Limites</a>
    <a href="#erros">5. Erros comuns</a>
  </nav>

  <section id="simples">
    <h2>1. Uso simples (sem programar nada)</h2>
    <p class="lede">Quem te mandou o link já te deu uma chave própria embutida nele.</p>
    <div class="step"><div class="n">1</div><p>Abra o link que você recebeu — algo como
      <code class="mono">__BASE_URL__/app?key=SUA_CHAVE</code>.</p></div>
    <div class="step"><div class="n">2</div><p>Digite o texto que quer ouvir, escolha a voz
      (Automática ou uma voz clonada, se disponível) e a qualidade.</p></div>
    <div class="step"><div class="n">3</div><p>Clique em <b>▶ Gerar</b>. O áudio toca sozinho
      quando fica pronto — junto aparece quanto tempo levou e quanta memória de vídeo (VRAM)
      foi usada, só de curiosidade.</p></div>
    <div class="note">Se o link não abrir ou disser "acesso negado", sua chave pode ter sido
      desativada — chame quem te mandou o link.</div>
  </section>

  <section id="autenticacao">
    <h2>2. Sua chave de acesso</h2>
    <p class="lede">Toda chamada à API precisa da sua chave, de um destes dois jeitos:</p>
    <div class="code">// opção A — na URL (mais simples)
?key=SUA_CHAVE

// opção B — em um cabeçalho (recomendado em código)
X-API-Key: SUA_CHAVE</div>
    <p class="lede">A chave é pessoal: ela define seu limite de texto, quais vozes e qualidades
    você pode usar, e quantos pedidos por minuto. Não precisa (nem deve) compartilhar a sua
    chave — peça ao administrador uma nova se quiser dar acesso a outra pessoa.</p>
  </section>

  <section id="api">
    <h2>3. Referência da API</h2>
    <p class="lede">Base da URL: <code class="mono">__BASE_URL__</code></p>
    <p class="lede">Ao usar um domínio gratuito do ngrok por código/IA, envie também
    <code class="mono">ngrok-skip-browser-warning: 1</code>. Esse cabeçalho evita que
    a resposta seja substituída pela página de aviso do ngrok.</p>

    <div class="route-head"><span class="badge post">POST</span><code>/api/v1/generate</code></div>
    <p class="lede">Gera um áudio a partir de um texto. Retorna o link do arquivo pronto.</p>
    <div class="code">curl -X POST "__BASE_URL__/api/v1/generate?key=SUA_CHAVE" \\
  -H "ngrok-skip-browser-warning: 1" \\
  -H "Content-Type: application/json" \\
  --data-raw '{"text": "Olá, mundo!", "voice": "auto", "quant": "q8_0"}'</div>
    <table class="ref">
      <thead><tr><th>Campo (corpo JSON)</th><th>Obrigatório</th><th>Descrição</th></tr></thead>
      <tbody>
        <tr><td class="mono">text</td><td>sim</td><td>Texto a sintetizar (respeita o limite de
          caracteres da sua chave).</td></tr>
        <tr><td class="mono">voice</td><td>não</td><td>Qual voz usar. <code class="mono">auto</code>
          deixa o modelo escolher; outras vozes (se sua chave permitir) clonam um timbre específico.</td></tr>
        <tr><td class="mono">quant</td><td>não</td><td><code class="mono">q8_0</code> (mais rápido)
          ou <code class="mono">f16</code> (mais nítido), se sua chave permitir ambos.</td></tr>
      </tbody>
    </table>
    <p class="lede">Resposta (200 OK):</p>
    <div class="code">{
  "audio_url": "/api/v1/audio/xxxxx.wav",
  "gen_s": 0.9,        <span class="c1">// segundos para gerar</span>
  "decode_s": 0.1,     <span class="c1">// segundos para decodificar o áudio</span>
  "dur_s": 2.9,        <span class="c1">// duração do áudio gerado, em segundos</span>
  "rtf": 0.31,         <span class="c1">// tempo real; abaixo de 1.0 = mais rápido que ouvir</span>
  "vram_mb": 613       <span class="c1">// memória de vídeo usada nesta geração</span>
}</div>
    <p class="lede">Em erro, a resposta é <code class="mono">{"error": "mensagem"}</code> com um
    código HTTP condizente (veja a seção 5).</p>

    <div class="route-head"><span class="badge get">GET</span><code>/api/v1/audio/&lt;arquivo&gt;</code></div>
    <p class="lede">Baixa o áudio gerado. <code class="mono">audio_url</code> na resposta acima já
    vem pronto para usar — é só montar a URL completa:
    <code class="mono">__BASE_URL__ + audio_url</code>.</p>

    <div class="route-head"><span class="badge get">GET</span><code>/api/v1/voices</code></div>
    <p class="lede">Mostra o que sua chave pode usar — vozes, qualidades e limites.</p>
    <div class="code">curl "__BASE_URL__/api/v1/voices?key=SUA_CHAVE" \\
  -H "ngrok-skip-browser-warning: 1"</div>
    <div class="code">{
  "voices": ["auto", "botan"],
  "quant": ["q8_0", "f16", "q4_0"],
  "max_chars": 300,
  "max_steps": 32
}</div>

    <div class="route-head"><span class="badge post">POST</span><code>/api/v1/model</code></div>
    <p class="lede">Escolhe qual modelo carregar: <code class="mono">q8</code> (rápido),
    <code class="mono">f16</code> (qualidade) ou <code class="mono">q4</code> (leve).
    Fica só um carregado por vez — ao trocar, o outro é descarregado (libera memória).
    Use <code class="mono">GET /api/v1/model</code> para ver o ativo.</p>
    <div class="code">curl -X POST "__BASE_URL__/api/v1/model?key=SUA_CHAVE" \\
  -H "ngrok-skip-browser-warning: 1" \\
  -H "Content-Type: application/json" \\
  --data-raw '{"quant": "f16"}'</div>

    <div class="route-head"><span class="badge post">POST</span><code>/api/v1/clone-voices/upload</code></div>
    <p class="lede">Envia um áudio (WAV, 3–10s de fala limpa) e a API clona a voz.
    Ela fica <b>privada — só a sua chave</b> consegue ver e usar. Depois é só passar
    o nome retornado no campo <code class="mono">voice</code> ao gerar. O
    <code class="mono">ref_text</code> é opcional (se não mandar, a API transcreve).</p>
    <div class="code">curl -X POST "__BASE_URL__/api/v1/clone-voices/upload?key=SUA_CHAVE" \\
  -H "ngrok-skip-browser-warning: 1" \\
  -F "voice_name=minhavoz" \\
  -F "audio=@caminho/para/voz.wav"

# resposta: {"ok": true, "voice": "c7__minhavoz", ...}
# depois: gere com  "voice": "c7__minhavoz"</div>

    <p class="lede" style="margin-top:26px">Exemplo em Python:</p>
    <div class="code">import requests

r = requests.post(
    "__BASE_URL__/api/v1/generate",
    params={"key": "SUA_CHAVE"},
    headers={"ngrok-skip-browser-warning": "1"},
    json={"text": "Testando por código.", "voice": "auto"},
)
data = r.json()
print(data["gen_s"], "segundos —", data["audio_url"])</div>
  </section>

  <section id="limites">
    <h2>4. Limites</h2>
    <p class="lede">Cada chave tem os próprios limites, configurados pelo administrador.
    Use <code class="mono">GET /api/v1/voices</code> (seção 3) para ver os seus na hora.
    Em geral:</p>
    <table class="ref">
      <thead><tr><th>Limite</th><th>O que significa</th></tr></thead>
      <tbody>
        <tr><td>Máx. caracteres</td><td>Tamanho máximo do texto por pedido.</td></tr>
        <tr><td>Pedidos por minuto</td><td>Quantas gerações sua chave pode pedir em 60s —
          passou disso, espere um pouco e tente de novo.</td></tr>
        <tr><td>Vozes/qualidades permitidas</td><td>Só as listadas em
          <code class="mono">/api/v1/voices</code> funcionam para sua chave.</td></tr>
      </tbody>
    </table>
  </section>

  <section id="erros">
    <h2>5. Erros comuns</h2>
    <table class="ref">
      <thead><tr><th>Código</th><th>Significa</th></tr></thead>
      <tbody>
        <tr><td class="mono">401</td><td>Faltou a chave (<code class="mono">?key=</code> ou
          cabeçalho <code class="mono">X-API-Key</code>).</td></tr>
        <tr><td class="mono">403</td><td>Chave inválida ou desativada pelo administrador.</td></tr>
        <tr><td class="mono">429</td><td>Passou do limite de pedidos por minuto — aguarde.</td></tr>
        <tr><td class="mono">400</td><td>Pedido inválido: texto vazio, longo demais, ou voz/qualidade
          que sua chave não permite.</td></tr>
        <tr><td class="mono">500</td><td>Falha ao gerar o áudio — raro; tente de novo ou avise
          o administrador.</td></tr>
      </tbody>
    </table>
  </section>
</div>
<script>
document.querySelectorAll('.code').forEach(block => {
  const originalText = block.textContent; // capturado ANTES de inserir o botão
  const btn = document.createElement('button');
  btn.textContent = 'copiar';
  btn.className = 'copy-btn';
  btn.onclick = () => {
    navigator.clipboard.writeText(originalText.trim());
    btn.textContent = 'copiado!';
    setTimeout(() => { btn.textContent = 'copiar'; }, 1200);
  };
  block.appendChild(btn);
});
</script>
</body></html>"""

ADMIN_HTML = ADMIN_HTML_BODY.replace("__STYLE__", _STYLE)
FRIEND_HTML = FRIEND_HTML_BODY.replace("__STYLE__", _STYLE)
DOCS_HTML = DOCS_HTML_BODY.replace("__STYLE__", _STYLE)


_LOGIN_HTML = LOGIN_HTML.replace("__STYLE__", _STYLE)


@app.before_request
def _admin_gate():
    if request.path == "/admin" and not is_admin_request():
        return _LOGIN_HTML
    return None


if __name__ == "__main__":
    init_db()

    def lan_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"
        finally:
            s.close()

    _target = resolved_gpu_target()
    print(f"GPU detectada   : {GPU_NAME or 'desconhecida'}  ->  build '{_target}'")
    print(f"runtime C       : {c_exe()}")
    if not c_exe().exists():
        print("AVISO: o binário deste build NÃO existe — compile-o antes de gerar.")
        print("       rode  build_cuda_universal.bat omnivoice-pt  (ou build_cuda.bat)")

    # Descarregador de ociosidade (só age quando keep_loaded=off).
    threading.Thread(target=_idle_monitor, daemon=True).start()

    # Se "manter modelo sempre carregado" estiver ligado, pré-aquece o modelo
    # padrão: sobe os workers com o modelo já na VRAM, prontos desde o primeiro
    # pedido. Se estiver desligado, o modelo sobe sob demanda e é liberado
    # quando ficar ocioso.
    if CONFIG["keep_loaded"]:
        dq = CONFIG["default_quant"]
        print(f"subindo worker {dq} ({workers_for(dq)} worker(s))...")
        ensure_quant_loaded(dq)
    else:
        print(f"keep_loaded=off — o modelo sobe no 1º pedido e é liberado "
              f"após {CONFIG['idle_minutes']} min ocioso")

    ip = lan_ip()
    print("=" * 60)
    print(f"painel de admin : http://{ip}:5180/admin")
    print(f"token de admin  : {ADMIN_TOKEN}")
    print(f"(também salvo em {ADMIN_TOKEN_PATH})")
    print(f"workers         : {CONFIG['workers']}  ·  "
          f"manter carregado: {'sim' if CONFIG['keep_loaded'] else 'não'}")
    print("=" * 60)
    # use_reloader=False: senão o Flask duplicaria o processo e subiria os
    # workers duas vezes (dobrando a VRAM).
    app.run(host="0.0.0.0", port=5180, threaded=True, use_reloader=False)
