"""A versioned autoregressive decoding protocol: what the engine samples with, and each request's seed.

A protocol file (``configs/shared/evaluation/decoding/<id>.json``) is read
through `load`, which refuses one that is not pure nucleus sampling (the
maintainer's rule of 2026-10-24: a non-zero temperature, top-p below 1, top-k
and min-p disabled). `request_seed` derives a request's seed from the
protocol's base and the request's key alone -- never from the model -- so every
checkpoint evaluated under one protocol draws with the same seed for the same
item, turn and sample.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "aadistill.decoding_protocol/v1"
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                 "presence_penalty", "frequency_penalty")


def load(path) -> dict:
    path = Path(path)
    doc = json.loads(path.read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, not {SCHEMA}")
    s = doc["sampling"]
    if sorted(s) != sorted(SAMPLING_KEYS):
        raise ValueError(f"{path}: sampling must name exactly {SAMPLING_KEYS}")
    problems = ([] if s["temperature"] > 0 else ["temperature must be above 0"]) \
        + ([] if 0 < s["top_p"] < 1 else ["top_p must be in (0, 1)"]) \
        + ([] if s["top_k"] == -1 else ["top_k must be -1 (disabled): the protocol is pure top-p"]) \
        + ([] if s["min_p"] == 0 else ["min_p must be 0 (disabled): the protocol is pure top-p"]) \
        + ([] if (s["repetition_penalty"], s["presence_penalty"], s["frequency_penalty"]) == (1.0, 0.0, 0.0)
           else ["no penalty may be applied"]) \
        + ([] if int(doc["samples_per_item"]) >= 1 else ["samples_per_item must be at least 1"])
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return {**doc, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def request_seed(protocol: dict, namespace: str, item_id: str, turn: int = 0, sample: int = 0) -> int:
    key = f"{protocol['seed_rule']['base']}|{namespace}|{item_id}|{turn}|{sample}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % 2 ** 63


def sampling(protocol: dict) -> dict:
    """The keyword arguments the engine's sampling parameters receive."""
    return dict(protocol["sampling"])


def identity(protocol: dict) -> dict:
    return {"protocol_id": protocol["protocol_id"], "path": protocol["path"], "sha256": protocol["sha256"],
            "sampling": sampling(protocol), "samples_per_item": protocol["samples_per_item"],
            "seed_rule": protocol["seed_rule"]}
