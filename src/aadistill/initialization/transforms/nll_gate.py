"""An initialization checkpoint is incomplete until its own NLL is measured.

E8's binding rule: every time a new initialization checkpoint is produced, that
checkpoint's NLL is measured **before** any recovery training, and the result is
bound to that checkpoint's hash. Nothing may be inherited, copied, interpolated
or assumed from a previous initialization, however closely related the recipe is.

A rule stated in prose is a rule that gets skipped at 2am on a paid pod, so it is
mechanical here:

* the record names the checkpoint's `model.safetensors` sha256, and the gate
  recomputes that hash from the checkpoint it is about to train — so a record
  copied from a sibling initialization simply does not match;
* every individual measurement *also* carries the checkpoint hash it was taken
  on, so a single stream's result cannot be spliced in from another run while the
  envelope still claims the right checkpoint;
* a record that advertises itself as inherited is rejected outright, rather than
  being silently accepted because its hash happens to match;
* a missing required measurement is a failure, not a shorter report. The point of
  the gate is to know what the initialization changed *before* recovery training
  obscures it, and a report missing the general-language series cannot say that.

The gate deliberately does **not** look at the NLL *values*. A worse or better
initialization NLL must not cancel or promote E8 — the endpoint is autonomous
behaviour after matched recovery. The only value-level check is validity:
finite, positive, and not silently zero.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

# The three general-language / teacher-native series E8 requires per checkpoint.
# `holdout_v1` is the historical 40-document series, preserved for continuity;
# `fineweb_val_e7` is E7's 20x larger disjoint stream; `teacher_native_val` is
# the pack's own held-out validation slice, the teacher-native counterpart. They
# are separate keys because they are separate quantities and must never be
# averaged, substituted or reported as one number.
REQUIRED_MEASUREMENTS = ("holdout_v1", "fineweb_val_e7", "teacher_native_val")


class InitNllGateError(RuntimeError):
    """Raised when a checkpoint may not proceed to recovery training."""


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_fingerprint(ckpt_dir: str | Path) -> dict:
    """What identifies an initialization checkpoint for gating purposes."""
    d = Path(ckpt_dir)
    weights = d / "model.safetensors"
    if not weights.is_file():
        raise InitNllGateError(f"{weights} is missing; not a checkpoint directory")
    config_path = d / "config.json"
    if not config_path.is_file():
        raise InitNllGateError(f"{config_path} is missing; not a checkpoint directory")
    config = json.loads(config_path.read_text())
    return {
        "path": str(d),
        "model_sha256": sha256_path(weights),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
    }


def require_init_nll(
    ckpt_dir: str | Path,
    record_path: str | Path,
    *,
    required: tuple[str, ...] = REQUIRED_MEASUREMENTS,
) -> dict:
    """Return the record, or raise. Call this before any training step.

    Recomputes the checkpoint hash rather than trusting the record's own copy of
    it: the whole guarantee is that the measurement was taken on *these* weights.
    """
    record_path = Path(record_path)
    if not record_path.is_file():
        raise InitNllGateError(
            f"no initialization-NLL artifact at {record_path}; an initialization "
            "checkpoint is not complete until its own NLL is measured, and "
            "training must not start without it")
    try:
        record = json.loads(record_path.read_text())
    except json.JSONDecodeError as exc:
        raise InitNllGateError(f"{record_path} is not readable JSON: {exc}") from exc

    for flag in ("inherited", "copied_from", "source_record", "interpolated"):
        if record.get(flag):
            raise InitNllGateError(
                f"{record_path} declares {flag}={record[flag]!r}; an inherited "
                "initialization NLL is exactly what E8 forbids")

    fingerprint = checkpoint_fingerprint(ckpt_dir)
    claimed = (record.get("checkpoint") or {}).get("model_sha256")
    if claimed != fingerprint["model_sha256"]:
        raise InitNllGateError(
            f"{record_path} was measured on model.safetensors {claimed} but "
            f"{ckpt_dir} hashes to {fingerprint['model_sha256']}; this record "
            "belongs to a different initialization")

    measurements = record.get("measurements") or {}
    missing = [k for k in required if k not in measurements]
    if missing:
        raise InitNllGateError(
            f"{record_path} is missing required measurements {missing}; present: "
            f"{sorted(measurements)}")

    for name in required:
        m = measurements[name]
        if m.get("measured_checkpoint_sha256") != fingerprint["model_sha256"]:
            raise InitNllGateError(
                f"measurement {name!r} in {record_path} carries checkpoint hash "
                f"{m.get('measured_checkpoint_sha256')}, not "
                f"{fingerprint['model_sha256']}; a single measurement cannot be "
                "spliced in from another run")
        nll = m.get("nll")
        if not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll <= 0:
            raise InitNllGateError(
                f"measurement {name!r} in {record_path} reports nll={nll!r}, which "
                "is not a valid negative log-likelihood")

    return record


def gate_summary(record: dict, required: tuple[str, ...] = REQUIRED_MEASUREMENTS) -> dict:
    """One flat row per checkpoint for the step-0 comparison table."""
    m = record.get("measurements") or {}
    out = {
        "checkpoint_sha256": (record.get("checkpoint") or {}).get("model_sha256"),
        "kept_teacher_layers": (record.get("depth_map") or {}).get(
            "kept_teacher_layers"),
        "removed_teacher_layers": (record.get("depth_map") or {}).get(
            "removed_teacher_layers"),
        "depth_map_source": (record.get("depth_map") or {}).get("source"),
        "num_parameters": (record.get("checkpoint") or {}).get("num_parameters"),
    }
    for name in required:
        entry = m.get(name) or {}
        for field in ("nll", "ppl", "kl", "top1", "mean_rank", "positions"):
            if field in entry:
                out[f"{name}.{field}"] = entry[field]
    return out
