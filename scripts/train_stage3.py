"""Stage 3 CLI: recovery training over the Stage 2 offline mixture.

Usage:
    uv run python scripts/train_stage3.py --config configs/stage3_<name>.json
    uv run python scripts/train_stage3.py --config ... --resume [step_XXXXXX]

A fresh run refuses to write into an out_dir that already contains
checkpoints (pass --resume, or pick a new out_dir). --resume without an
argument continues from the latest checkpoint; with a tag it continues from
that checkpoint. Resume verifies the config hash, so the same config file
must be used; the training jsonl keeps appending across resumes.

The same config runs on CPU or GPU ("device": "auto") — hardware never
changes the experiment definition (AGENTS.md P8.1/P8.2).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from aadistill.env import code_state, hardware_report, set_determinism
from aadistill.manifest import sha256_file, sha256_json, write_manifest
from aadistill.teacher import DTYPES, load_teacher, tokenizer_hash
from aadistill.train import (
    JsonlLogger,
    Trainer,
    build_blocks,
    validate_train_config,
)


def resolve_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"config requests device {pref!r} but CUDA is unavailable")
    return pref


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="TAG",
        help="resume from the latest (or a named step_XXXXXX) checkpoint",
    )
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    validate_train_config(cfg)
    set_determinism(cfg["seed"])
    device = resolve_device(cfg["device"])
    out_dir = REPO_ROOT / cfg["out_dir"]
    ckpt_root = out_dir / "checkpoints"

    if args.resume:
        tag = args.resume
        if tag == "latest":
            latest = ckpt_root / "latest.txt"
            if not latest.is_file():
                raise FileNotFoundError(f"no {latest} to resume from")
            tag = latest.read_text().strip()
        resume_ckpt = ckpt_root / tag
        if not resume_ckpt.is_dir():
            raise FileNotFoundError(f"checkpoint {resume_ckpt} does not exist")
        model_path = resume_ckpt / "model"
    else:
        if ckpt_root.is_dir() and any(ckpt_root.iterdir()):
            raise RuntimeError(
                f"{ckpt_root} already contains checkpoints; pass --resume or "
                "choose a fresh out_dir"
            )
        resume_ckpt = None
        model_path = REPO_ROOT / cfg["student_path"]

    logger = JsonlLogger(out_dir / "train_log.jsonl")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(REPO_ROOT / cfg["student_path"])
    print(f"device {device}; encoding Stage 2 mixture from {cfg['data_dir']} ...")
    data_dir = REPO_ROOT / cfg["data_dir"]
    train_blocks = build_blocks(
        tokenizer, data_dir, "train", cfg["block_len"], cfg["groups"]
    )
    val_blocks = build_blocks(tokenizer, data_dir, "val", cfg["block_len"], cfg["groups"])
    logger.log(
        "dataset_loaded",
        data_dir=cfg["data_dir"],
        block_len=cfg["block_len"],
        tokenizer_sha256=tokenizer_hash(tokenizer),
        train=train_blocks[3],
        val=val_blocks[3],
    )

    teacher = None
    teacher_identity = None
    if cfg["teacher"] is not None:
        t = cfg["teacher"]
        print(f"loading teacher {t['model_id']} ...")
        teacher, teacher_tok, teacher_identity = load_teacher(
            t["model_id"], t["revision"], dtype=t["dtype"], device=device
        )
        if tokenizer_hash(teacher_tok) != tokenizer_hash(tokenizer):
            raise ValueError("teacher and student tokenizers differ; refusing to train")
        logger.log("teacher_loaded", **teacher_identity)

    print(f"loading student from {model_path} ...")
    student = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=DTYPES[cfg["dtype"]]
    )
    logger.log(
        "student_loaded",
        path=str(model_path),
        dtype=cfg["dtype"],
        num_parameters=sum(p.numel() for p in student.parameters()),
    )

    trainer = Trainer(
        cfg,
        student,
        (train_blocks[0], train_blocks[1]),
        (val_blocks[0], val_blocks[1]),
        teacher=teacher,
        device=device,
        out_dir=out_dir,
        logger=logger,
    )
    if resume_ckpt is not None:
        trainer.restore(resume_ckpt)

    manifest_name = (
        "run_manifest.json"
        if resume_ckpt is None
        else f"run_manifest_resume_step{trainer.step:06d}.json"
    )
    write_manifest(
        out_dir / manifest_name,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "config": cfg,
            "config_sha256": sha256_json(cfg),
            "resumed_from": str(resume_ckpt) if resume_ckpt else None,
            "device": device,
            "data_manifests": {
                p.name: sha256_file(p) for p in sorted(data_dir.glob("*.manifest.json"))
            },
            "tokenizer_sha256": tokenizer_hash(tokenizer),
            "teacher": teacher_identity,
            "student_source": str(model_path),
            "trainable_params": trainer.freeze_report["trainable_params"],
            "total_params": trainer.freeze_report["total_params"],
            "code_state": code_state(str(REPO_ROOT)),
            "hardware": hardware_report(),
        },
    )
    logger.log("config_loaded", config=args.config, config_sha256=sha256_json(cfg))

    summary = trainer.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
