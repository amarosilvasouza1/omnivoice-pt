"""Acrescenta uma voz global às conexões com lista explícita de vozes."""

import argparse
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("voice")
args = parser.parse_args()

conn = sqlite3.connect(ROOT / "build" / "api.db")
rows = conn.execute(
    "SELECT id, allowed_voices FROM connections WHERE enabled = 1"
).fetchall()
updates = []
for conn_id, raw in rows:
    allowed = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if "*" not in allowed and args.voice not in allowed:
        allowed.append(args.voice)
        updates.append((",".join(allowed), conn_id))
conn.executemany(
    "UPDATE connections SET allowed_voices = ? WHERE id = ?", updates
)
conn.commit()
print(f"connections={len(rows)} changed={len(updates)}")
