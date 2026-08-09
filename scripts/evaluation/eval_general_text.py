#!/usr/bin/env python3
"""General-text diagnostics for one checkpoint on a dense held-out stream.

Reports FineWeb held-out NLL, teacher→student KL, top-1, mean target rank and
confidence. **All diagnostics.** They describe what happened to the student's
distribution over ordinary prose; they may not promote a checkpoint. E7's
promotion decision is the frozen autonomous rollout evaluation and nothing else.

    PYTHONPATH=src python scripts/evaluation/eval_general_text.py \\
        --model artifacts/stage3/e7_fineweb_r1600k_sa/checkpoints/step_001761/model \\
        --stream artifacts/stage3/e7_fineweb_val \\
        --teacher Qwen/Qwen3-4B-Thinking-2507 \\
        --out artifacts/audit/e7_general_text/e7_fineweb_sa.json

The `--stream` count is checked against the stream's manifest, so a truncated or
substituted validation set is an error rather than a quietly different number.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.extra_stream import load_extra_stream  # noqa: E402
from aadistill.evaluation.general_text import general_text_metrics  # noqa: E402
from aadistill.infrastructure.env import code_state, hardware_report  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--teacher", default="")
    ap.add_argument("--teacher-revision", default="")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--micro-blocks", type=int, default=1)
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    ids, content, meta = load_extra_stream(
        args.stream if Path(args.stream).is_absolute() else REPO_ROOT / args.stream)
    if int(ids.shape[0]) != int(meta["n_blocks"]):
        raise SystemExit("stream does not match its manifest")

    student = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    student.config.use_cache = False
    teacher = None
    teacher_identity = None
    if args.teacher:
        from aadistill.models.teacher import load_teacher
        teacher, _, teacher_identity = load_teacher(
            args.teacher, args.teacher_revision or None, dtype=args.dtype,
            device=device)
        teacher.config.use_cache = False

    metrics = general_text_metrics(
        student, ids, teacher=teacher, device=device,
        micro_blocks=args.micro_blocks,
        max_blocks=args.max_blocks or None)

    expected = metrics["blocks"] * (int(meta["block_len"]) - 1)
    if metrics["positions"] != expected:
        raise SystemExit(
            f"scored {metrics['positions']} positions, expected {expected} for "
            "a dense stream; the evaluation set is not what it claims to be")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "model": args.model,
        "stream": args.stream,
        "stream_manifest_sha256": sha256_json(meta),
        "stream_source": meta.get("source"),
        "teacher": teacher_identity,
        "dtype": args.dtype,
        "device": device,
        "metrics": metrics,
        "metric_status": "DIAGNOSTIC ONLY — never promotes a checkpoint "
                         "(decision record 2026-08-09)",
        "code_state": code_state(str(REPO_ROOT)),
        "hardware": hardware_report(),
    }
    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
