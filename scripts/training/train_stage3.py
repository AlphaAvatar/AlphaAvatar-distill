"""Stage 3 CLI: recovery training over an offline mixture or a packed token ladder.

Usage:
    uv run python scripts/training/train_stage3.py --config configs/stage3_<name>.json
    uv run python scripts/training/train_stage3.py --config ... --resume [step_XXXXXX]

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
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from aadistill.autoinit.recovery import (
    RuntimeEnvironmentFingerprint,
    trainer_source_digest,
)
from aadistill.infrastructure.env import code_state, hardware_report, set_determinism
from aadistill.infrastructure.manifest import sha256_file, sha256_json, write_manifest
from aadistill.models.teacher import DTYPES, load_teacher, tokenizer_hash
from aadistill.data.ladder import ladder_blocks
from aadistill.training.train import (
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
    source = ("packed token ladder" if cfg.get("packing") == "ladder"
              else "Stage 2 mixture")
    print(f"device {device}; loading {source} from {cfg['data_dir']} ...")
    data_dir = REPO_ROOT / cfg["data_dir"]
    # kd_scope "all_no_think" needs the think tokens; resolve them here rather
    # than in the core, which stays model-agnostic (P3). Single-token ids are
    # required so the span scan is unambiguous.
    think_ids = None
    if cfg["loss"]["kd_scope"] == "all_no_think":
        opened = tokenizer.encode("<think>", add_special_tokens=False)
        closed = tokenizer.encode("</think>", add_special_tokens=False)
        if len(opened) != 1 or len(closed) != 1:
            raise ValueError(
                "kd_scope 'all_no_think' needs <think>/</think> to be single "
                f"tokens for this tokenizer; got {opened} and {closed}"
            )
        think_ids = (opened[0], closed[0])

    packing = cfg.get("packing", "concat")
    extra_val_blocks = {}
    extra_val_stats = {}
    if packing == "ladder":
        # The pack is the data. Blocks are read as cut, so the rung trained is
        # byte-identical to the rung the corpus gate validated.
        train_tuple, val_tuple, ladder_stats = ladder_blocks(
            data_dir, cfg["rung"], n_val=cfg.get("val_blocks", 16)
        )
        if int(ladder_stats["block_len"]) != cfg["block_len"]:
            raise ValueError(
                f"pack was cut at block_len {ladder_stats['block_len']} but the "
                f"config says {cfg['block_len']}")
        train_stats, val_stats = ladder_stats, ladder_stats["val_token_mix"]
    else:
        train_blocks = build_blocks(
            tokenizer, data_dir, "train", cfg["block_len"], cfg["groups"],
            packing=packing, seed=cfg["seed"],
        )
        val_blocks = build_blocks(
            tokenizer, data_dir, "val", cfg["block_len"], cfg["groups"],
            packing=packing, seed=cfg["seed"],
        )
        train_tuple = (train_blocks[0], train_blocks[1], train_blocks[4])
        val_tuple = (val_blocks[0], val_blocks[1], val_blocks[4])
        train_stats, val_stats = train_blocks[3], val_blocks[3]
        for name, extra_dir in (cfg.get("extra_val") or {}).items():
            blocks = build_blocks(
                tokenizer, REPO_ROOT / extra_dir, "val", cfg["block_len"], None,
                packing=packing, seed=cfg["seed"],
            )
            extra_val_blocks[name] = (blocks[0], blocks[1], blocks[4])
            extra_val_stats[name] = blocks[3]
    logger.log(
        "dataset_loaded",
        data_dir=cfg["data_dir"],
        block_len=cfg["block_len"],
        packing=packing,
        rung=cfg.get("rung"),
        kd_scope=cfg["loss"]["kd_scope"],
        think_ids=list(think_ids) if think_ids else None,
        tokenizer_sha256=tokenizer_hash(tokenizer),
        train=train_stats,
        val=val_stats,
        extra_val=extra_val_stats,
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

    # The second, KD-only stream (E7). Loaded here rather than inside Trainer so
    # its manifest — the treatment's identity — is read once, hashed into the
    # run manifest, and logged before a single step runs.
    extra_stream_blocks = extra_stream_meta = None
    if cfg.get("extra_stream") is not None:
        from aadistill.data.extra_stream import load_extra_stream, stream_budget

        extra_dir = REPO_ROOT / cfg["extra_stream"]["data_dir"]
        print(f"loading extra KD stream from {extra_dir} ...")
        e_ids, e_content, extra_stream_meta = load_extra_stream(extra_dir)
        extra_stream_blocks = (e_ids, e_content)
        budget = stream_budget(
            int(e_ids.shape[0]), int(e_ids.shape[1]),
            total_steps=cfg["schedule"]["total_steps"],
            blocks_per_step=cfg["extra_stream"]["blocks_per_step"],
            every_n_steps=cfg["extra_stream"]["every_n_steps"])
        logger.log("extra_stream_loaded", data_dir=cfg["extra_stream"]["data_dir"],
                   kind=cfg["extra_stream"]["kind"],
                   lambda_extra=cfg["extra_stream"]["lambda_extra"],
                   manifest_sha256=sha256_json(extra_stream_meta),
                   source=extra_stream_meta.get("source"),
                   budget=budget)
        print(f"  {budget['kd_positions']:,} extra KD positions over "
              f"{budget['active_steps']} steps ({budget['exposures']} exposures)")

    trainer = Trainer(
        cfg,
        student,
        train_tuple,
        val_tuple,
        teacher=teacher,
        device=device,
        out_dir=out_dir,
        logger=logger,
        extra_val_blocks=extra_val_blocks,
        think_ids=think_ids,
        extra_stream_blocks=extra_stream_blocks,
    )
    if resume_ckpt is not None:
        trainer.restore(resume_ckpt)

    manifest_name = (
        "run_manifest.json"
        if resume_ckpt is None
        else f"run_manifest_resume_step{trainer.step:06d}.json"
    )
    image_digest = os.environ.get("AADISTILL_IMAGE_DIGEST") or None
    runtime = RuntimeEnvironmentFingerprint.observe(image_digest=image_digest)
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
                p.name: sha256_file(p)
                for d in [data_dir]
                + [REPO_ROOT / e for e in (cfg.get("extra_val") or {}).values()]
                for p in sorted(Path(d).glob("*.manifest.json"))
            },
            # A ladder run's data identity is the pack itself, not a mixture
            # manifest: these hashes pin the exact blocks that trained (P4).
            "ladder": (
                {**{k: v for k, v in ladder_stats.items()
                    if k != "val_block_indices"},
                 "blocks_sha256": sha256_file(data_dir / "blocks.npz"),
                 "ladder_json_sha256": sha256_file(data_dir / "ladder.json"),
                 "audit_sha256": sha256_file(data_dir / "audit.jsonl")}
                if cfg.get("packing") == "ladder" else None
            ),
            "tokenizer_sha256": tokenizer_hash(tokenizer),
            "teacher": teacher_identity,
            "student_source": str(model_path),
            "trainable_params": trainer.freeze_report["trainable_params"],
            # Recorded separately so a run can always state how much of its
            # capacity was full-rank and how much was low-rank (P4).
            "full_rank_trainable_params":
                trainer.freeze_report["full_rank_trainable_params"],
            "lora_trainable_params": trainer.freeze_report["lora_trainable_params"],
            "lora": (trainer.lora_cfg.to_dict() if trainer.lora_cfg else None),
            "lora_modules": trainer.freeze_report.get("lora_modules"),
            "total_params": trainer.freeze_report["total_params"],
            # The extra stream's full manifest, not a reference to it: this run
            # is only reproducible if the exact FineWeb range, revision, content
            # hashes and packing policy travel with the record (P4).
            "extra_stream": (
                {"config": cfg["extra_stream"],
                 "manifest": extra_stream_meta,
                 "manifest_sha256": sha256_json(extra_stream_meta),
                 "planned_kd_positions": trainer.planned_extra_kd_positions()}
                if extra_stream_meta is not None else None),
            # What actually ran, read off the executing objects. This block is
            # what makes a run's protocol *reconstructable from its own
            # artifacts* instead of restated from the config a verifier already
            # holds — see RecoveryProtocolFingerprint.from_run_artifacts.
            #
            # `image_digest` cannot be observed from inside a container; it is
            # supplied by the launcher through AADISTILL_IMAGE_DIGEST, the same
            # source the preflight's Stage-0 attestation uses. Absent, it stays
            # null and the strict reconstruction fails closed rather than
            # accepting an unpinned runtime.
            "execution": {
                **trainer.execution_record(),
                "trainer_source": trainer_source_digest(REPO_ROOT),
                "runtime": runtime.as_dict(),
                "runtime_digest": runtime.digest,
                "image_digest_source": (
                    "env AADISTILL_IMAGE_DIGEST" if image_digest
                    else "ABSENT — runtime is not pinned"),
            },
            "code_state": code_state(str(REPO_ROOT)),
            "hardware": hardware_report(),
        },
    )
    logger.log("config_loaded", config=args.config, config_sha256=sha256_json(cfg))

    summary = trainer.run()
    print(json.dumps(summary, indent=2))

    # The run's own completion evidence. `run_manifest.json` is written before
    # the first step, so it can only say what was *about* to run; a control that
    # stopped early would still have a perfect manifest. Step and consumed-block
    # accounting are therefore recorded after `run()` returns, and the strict
    # reconstruction requires them.
    write_manifest(
        out_dir / ("run_completion.json" if resume_ckpt is None
                   else f"run_completion_resume_step{trainer.step:06d}.json"),
        {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256_json(cfg),
            "summary": summary,
            "final_step": trainer.step,
            "planned_total_steps": cfg["schedule"]["total_steps"],
            "completed_all_steps": trainer.step >= cfg["schedule"]["total_steps"],
            "consumed_blocks": trainer.consumed_blocks(),
            "execution": trainer.execution_record(),
        },
    )


if __name__ == "__main__":
    main()
