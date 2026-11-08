"""Mede a variação de VRAM total durante uma geração sintética."""

import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def gpu_used_mb() -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(out.splitlines()[0].strip())


parser = argparse.ArgumentParser()
parser.add_argument("quant", choices=["q8_0", "f16"])
parser.add_argument("--runs", type=int, default=1)
parser.add_argument(
    "--prompt", default="-",
    help="voz .ovprompt para medir com clonagem (padrão: voz automática)",
)
args = parser.parse_args()

exe = ROOT / "cmake-build-cuda" / "bin" / "omnivoice-pt.exe"
model = ROOT / "build" / f"omnivoice-pt-{args.quant}.gguf"
decoder = ROOT / "build" / "higgs-decoder-f16.gguf"
tokenizer = ROOT / "build" / "tokenizer.bin"
prompt = args.prompt
output = ROOT / "build" / f"benchmark-vram-{args.quant}.wav"

baseline_samples = [gpu_used_mb() for _ in range(5)]
baseline = round(statistics.median(baseline_samples))

proc = subprocess.Popen(
    [
        str(exe), "-m", str(model), "-d", str(decoder), "-k", str(tokenizer),
        "--gpu", "--serve",
    ],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
    bufsize=1,
)
ready = proc.stdout.readline().strip()
runtime_estimate = int(ready.split("\t")[1]) if ready.startswith("READY\t") else None
time.sleep(1)
loaded = gpu_used_mb()

text = (
    "Ola, este e um teste completo da voz. Hoje o dia esta muito bonito e "
    "vamos conversar com calma para medir o uso da placa de video durante "
    "toda a geracao do audio em portugues."
)
runs = []
all_samples = [loaded]
for run_index in range(args.runs):
    response = {"line": None}

    def read_response():
        response["line"] = proc.stdout.readline().strip()

    run_output = output.with_name(f"{output.stem}-{run_index + 1}.wav")
    started = time.perf_counter()
    proc.stdin.write(f"{run_output}\t32\t12345\t{prompt}\t{text}\n")
    proc.stdin.flush()
    reader = threading.Thread(target=read_response)
    reader.start()
    samples = [gpu_used_mb()]
    while reader.is_alive():
        samples.append(gpu_used_mb())
        time.sleep(0.05)
    reader.join()
    elapsed = time.perf_counter() - started
    samples.append(gpu_used_mb())
    all_samples.extend(samples)
    runs.append({
        "run": run_index + 1,
        "wall_time_s": round(elapsed, 3),
        "peak_total_mb": max(samples),
        "peak_delta_mb": max(samples) - baseline,
        "response": response["line"],
        "output": str(run_output),
    })

proc.terminate()
proc.wait(timeout=10)
time.sleep(2)
after = gpu_used_mb()

print(json.dumps({
    "quant": args.quant,
    "baseline_mb": baseline,
    "loaded_total_mb": loaded,
    "peak_total_mb": max(all_samples),
    "after_mb": after,
    "loaded_delta_mb": loaded - baseline,
    "peak_delta_mb": max(all_samples) - baseline,
    "runtime_estimate_mb": runtime_estimate,
    "samples": len(all_samples),
    "runs": runs,
}, ensure_ascii=False))
