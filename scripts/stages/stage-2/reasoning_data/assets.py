"""Where this experiment's input assets live, and how each one is read.

Every path below is an existing, hash-recorded asset this experiment AUDITS or
may reuse; none is created here. The owner of each asset's identity is named
beside it, and readers verify the recorded hash before trusting the bytes, so a
moved or edited file is refused rather than silently audited.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

CONFIG_DIR = REPO / "configs/stages/stage-2/reasoning_data"
LOG_DIR = REPO / "logs/stages/stage-2/reasoning_data"
S0 = REPO / "logs/stages/stage-0/calibration_volume"

#: corpus_v2 -- owner: its manifest (`outputs` carries both file hashes).
CORPUS_V2 = REPO / "artifacts/stages/stage-3/corpus_v2"
#: ladder_uniform_probe -- owner: its ladder.json (`outputs`) and
#: configs/stages/stage-3/recipes/recovery.json (`pack.blocks_sha256`).
LADDER = REPO / "artifacts/shared/instruments/ladder_uniform_probe"
#: base_pool_v1 -- owner: its manifest.
BASE_POOL = REPO / "artifacts/stages/stage-0/calibration_volume/base_pool_v1"
#: teacher_cal -- owner: runs/teachercal_20261018_112755/closeout/outcome.json.
TEACHER_CAL = (REPO / "artifacts/stages/stage-0/calibration_volume/teachercal_20261018_112755"
               / "teacher_cal/teacher_cal.jsonl")
TEACHER_CAL_OUTCOME = S0 / "runs/teachercal_20261018_112755/closeout/outcome.json"
CALIBRATION_V2 = REPO / "data/stages/stage-0/calibration_volume/tasks/calibration_v2.jsonl"
#: The tokenizer and chat template the E-series and Stage-0 recoveries rendered
#: with (`configs/stages/stage-3/e1/e1_r0860k_sa_pca.json :: tokenizer_source`).
TOKENIZER_DIR = REPO / "artifacts/stages/stage-1/qwen3_0p6b_init_v0/checkpoint"


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def verified(path: Path, expected: str) -> Path:
    """`path`, after its bytes are shown to have the recorded sha256."""
    got = sha256_file(path)
    if got != expected:
        raise RuntimeError(f"{path}: sha256 {got[:12]} is not the recorded {expected[:12]}")
    return path


def corpus_v2_manifest() -> dict:
    return read_json(CORPUS_V2 / "manifest.json")


def corpus_v2_candidates() -> Iterator[dict]:
    m = corpus_v2_manifest()
    return jsonl(verified(CORPUS_V2 / "candidates.jsonl", m["outputs"]["candidates"]))


def corpus_v2_sessions() -> Iterator[dict]:
    m = corpus_v2_manifest()
    return jsonl(verified(CORPUS_V2 / "sessions.jsonl", m["outputs"]["sessions"]))


def ladder_meta() -> dict:
    return read_json(LADDER / "ladder.json")


def ladder_audit() -> list[dict]:
    meta = ladder_meta()
    return list(jsonl(verified(LADDER / "audit.jsonl", meta["outputs"]["audit"])))


def base_pool_manifest() -> dict:
    return read_json(BASE_POOL / "manifest.json")


def base_pool_items() -> Iterator[dict]:
    return jsonl(BASE_POOL / "items.jsonl")


def teacher_cal_records() -> list[dict]:
    outcome = read_json(TEACHER_CAL_OUTCOME)
    expected = outcome["units"]["teacher_cal"]["teacher_cal.jsonl"]["sha256"]
    return list(jsonl(verified(TEACHER_CAL, expected)))


def calibration_tasks() -> dict[str, dict]:
    return {t["task_id"]: t for t in jsonl(CALIBRATION_V2)}


def split_raw(raw: str) -> tuple[str, str] | None:
    """(reasoning, answer) of a generation whose `<think>` was pre-opened."""
    if "</think>" not in raw:
        return None
    reasoning, _, answer = raw.partition("</think>")
    return reasoning.strip("\n"), answer.strip("\n")
