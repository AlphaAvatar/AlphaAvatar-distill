#!/usr/bin/env python3
"""Measure one initialization checkpoint's own NLL, and bind it to that checkpoint.

E8's mandatory step between constructing an initialization and training it:

    construct init -> hash and validate -> measure NLL -> record + hash -> train

Three series, kept separate because they are separate quantities:

``holdout_v1``
    The historical 40-document FineWeb series (`eval_ppl.py` protocol,
    `max_seq_len` 1024). Small and seed-noisy, kept only so the series that runs
    back to the Stage 1 gate stays continuous.
``fineweb_val_e7``
    E7's dense 512x1024 disjoint FineWeb stream — 20x the tokens, plus teacher
    KL, top-1, mean rank and confidence. The primary general-language number.
``teacher_native_val``
    The pack's own held-out validation slice over assistant-target positions:
    the teacher-native counterpart, and the closest thing to what the trainer
    reports as `val_ce`.

All three are **diagnostics**. They may not promote a checkpoint or cancel an arm
(decision record 2026-08-09). They exist so that what an initialization changed is
known *before* recovery training obscures it.

    PYTHONPATH=src python scripts/evaluation/measure_init_nll.py \\
        --checkpoint artifacts/stage1/e8_contribution_init_v1/checkpoint \\
        --label e8-contribution-init \\
        --holdout data/warmup/holdout_v1.jsonl \\
        --fineweb-val artifacts/stage3/e7_fineweb_val \\
        --pack artifacts/stage3/ladder_uniform_probe \\
        --teacher Qwen/Qwen3-4B-Thinking-2507 \\
        --out artifacts/stage1/e8_contribution_init_v1/init_nll.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.extra_stream import load_extra_stream  # noqa: E402
from aadistill.data.ladder import ladder_blocks  # noqa: E402
from aadistill.evaluation.general_text import general_text_metrics  # noqa: E402
from aadistill.evaluation.init_nll import masked_teacher_native_metrics  # noqa: E402
from aadistill.infrastructure.env import code_state, hardware_report  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from aadistill.init.nll_gate import (  # noqa: E402
    REQUIRED_MEASUREMENTS, checkpoint_fingerprint,
)

sys.path.insert(0, str(REPO_ROOT / "scripts/evaluation"))


def holdout_nll(model, tokenizer, path: Path, max_seq_len: int, device: str) -> dict:
    """The historical series, through the same function `eval_ppl.py` uses."""
    from eval_ppl import mean_nll

    samples = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    nll, tokens = mean_nll(model, tokenizer, samples, max_seq_len, device)
    return {"nll": round(float(nll), 6), "positions": int(tokens),
            "samples": len(samples), "max_seq_len": max_seq_len,
            "protocol": "eval_ppl.mean_nll, per-document truncation, batch 1"}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--holdout", default="data/warmup/holdout_v1.jsonl")
    ap.add_argument("--holdout-max-seq-len", type=int, default=1024)
    ap.add_argument("--fineweb-val", default="artifacts/stage3/e7_fineweb_val")
    ap.add_argument("--pack", default="artifacts/stage3/ladder_uniform_probe")
    ap.add_argument("--rung", type=int, default=2960000)
    ap.add_argument("--val-blocks", type=int, default=16)
    ap.add_argument("--teacher", default="")
    ap.add_argument("--teacher-revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--micro-blocks", type=int, default=1)
    ap.add_argument("--max-fineweb-blocks", type=int, default=0)
    ap.add_argument("--only", action="append", default=[],
                    help="restrict to named measurements (smoke tests only; the "
                         "gate still requires all of them)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else REPO_ROOT / p

    ckpt = resolve(args.checkpoint)
    fingerprint = checkpoint_fingerprint(ckpt)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    wanted = tuple(args.only) if args.only else REQUIRED_MEASUREMENTS
    print(f"{args.label}: {fingerprint['model_sha256'][:16]}… on {device}, "
          f"measuring {list(wanted)}", flush=True)

    student = AutoModelForCausalLM.from_pretrained(str(ckpt), dtype=dtype).to(device)
    student.config.use_cache = False
    student.eval()
    n_params = sum(p.numel() for p in student.parameters())

    # Not optional, and learned the hard way: transformers 5.x writes rope_theta
    # inside `rope_parameters`, a 4.x reader silently falls back to the class
    # default 10,000 instead of this checkpoint's 5,000,000, and the resulting NLL
    # looks plausible. Measured on this exact checkpoint, that skew moves
    # holdout_v1 from 11.7482 to 11.3953 with nothing raising. An initialization
    # NLL taken with the wrong positional basis is worse than no measurement.
    from aadistill.models.student import assert_rope_matches_config
    rope_base = assert_rope_matches_config(student, student.config, str(ckpt))

    teacher = None
    teacher_identity = None
    if args.teacher:
        from aadistill.models.teacher import load_teacher
        teacher, _, teacher_identity = load_teacher(
            args.teacher, args.teacher_revision or None, dtype=args.dtype,
            device=device)
        teacher.config.use_cache = False

    measurements: dict[str, dict] = {}
    sources: dict[str, dict] = {}

    if "holdout_v1" in wanted:
        started = time.time()
        # The historical series was measured with the *teacher's* tokenizer via
        # each checkpoint's own; an init checkpoint saved by an older transformers
        # may not reload its tokenizer, so fall back to the pinned teacher's,
        # which is the same vocabulary by construction (vocab_size is asserted
        # equal at initialization).
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
            tokenizer_source = "checkpoint"
        except Exception as exc:                     # noqa: BLE001 - recorded
            tokenizer = AutoTokenizer.from_pretrained(
                args.teacher or "Qwen/Qwen3-4B-Thinking-2507",
                revision=args.teacher_revision or None)
            tokenizer_source = f"pinned_teacher (checkpoint load failed: {exc!r})"
        path = resolve(args.holdout)
        m = holdout_nll(student, tokenizer, path, args.holdout_max_seq_len, device)
        m["tokenizer_source"] = tokenizer_source
        m["seconds"] = round(time.time() - started, 1)
        measurements["holdout_v1"] = m
        sources["holdout_v1"] = {"path": str(args.holdout),
                                 "sha256": sha256_file(path)}

    if "fineweb_val_e7" in wanted:
        started = time.time()
        stream_dir = resolve(args.fineweb_val)
        ids, _, meta = load_extra_stream(stream_dir)
        if int(ids.shape[0]) != int(meta["n_blocks"]):
            raise SystemExit("fineweb validation stream does not match its manifest")
        m = general_text_metrics(
            student, ids, teacher=teacher, device=device,
            micro_blocks=args.micro_blocks,
            max_blocks=args.max_fineweb_blocks or None)
        expected = m["blocks"] * (int(meta["block_len"]) - 1)
        if m["positions"] != expected:
            raise SystemExit(
                f"scored {m['positions']} positions, expected {expected} for a "
                "dense stream; the evaluation set is not what it claims to be")
        m["seconds"] = round(time.time() - started, 1)
        measurements["fineweb_val_e7"] = m
        sources["fineweb_val_e7"] = {"path": str(args.fineweb_val),
                                     "manifest_sha256": sha256_json(meta),
                                     "n_blocks": int(meta["n_blocks"]),
                                     "block_len": int(meta["block_len"])}

    if "teacher_native_val" in wanted:
        started = time.time()
        pack = resolve(args.pack)
        _, val, stats = ladder_blocks(pack, args.rung, n_val=args.val_blocks)
        val_ids, val_mask, _ = val
        m = masked_teacher_native_metrics(
            student, val_ids, val_mask, teacher=teacher, device=device,
            micro_blocks=args.micro_blocks)
        m["seconds"] = round(time.time() - started, 1)
        measurements["teacher_native_val"] = m
        sources["teacher_native_val"] = {
            "pack": str(args.pack), "rung": args.rung,
            "val_blocks": args.val_blocks,
            "ladder_json_sha256": sha256_file(pack / "ladder.json"),
            "blocks_sha256": sha256_file(pack / "blocks.npz"),
            "val_token_mix": stats.get("val_token_mix"),
        }

    # Every measurement carries the checkpoint hash it was taken on, so no single
    # series can be spliced in from another run behind a correct envelope.
    for m in measurements.values():
        m["measured_checkpoint_sha256"] = fingerprint["model_sha256"]

    depth_map = {"source": None, "kept_teacher_layers": None,
                 "removed_teacher_layers": None}
    init_manifest = ckpt.parent / "manifest.json"
    if init_manifest.is_file():
        diag = (json.loads(init_manifest.read_text()).get("init_diagnostics")
                or {})
        depth_map = {
            "source": diag.get("depth_map_source"),
            "kept_teacher_layers": diag.get("kept_teacher_layers"),
            "removed_teacher_layers": diag.get("removed_teacher_layers"),
            "init_manifest_sha256": sha256_file(init_manifest),
        }

    record = {
        "artifact": "initialization_nll_record",
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "checkpoint": {**fingerprint, "num_parameters": n_params,
                       "resolved_rope_base": rope_base},
        "depth_map": depth_map,
        "dtype": args.dtype,
        "device": device,
        "environment": {
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
        },
        "teacher": teacher_identity,
        "measurements": measurements,
        "sources": sources,
        "metric_status": "DIAGNOSTIC ONLY — an initialization NLL may neither "
                         "promote nor cancel an arm (decision record 2026-08-09); "
                         "it exists so the step-0 state is known before recovery "
                         "training obscures it",
        "required_measurements": list(REQUIRED_MEASUREMENTS),
        "complete": all(k in measurements for k in REQUIRED_MEASUREMENTS),
        "code_state": code_state(str(REPO_ROOT)),
        "hardware": hardware_report(),
    }
    record["record_sha256"] = sha256_json(record)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")

    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk in ("nll", "ppl", "kl", "top1", "mean_rank",
                                    "positions")}
                      for k, v in measurements.items()}, indent=2))
    print(f"complete={record['complete']} -> {out}")
    if not record["complete"]:
        print("NOT COMPLETE: the gate will refuse to start training on this "
              "checkpoint until every required measurement exists", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
