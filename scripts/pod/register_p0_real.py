#!/usr/bin/env python
"""Verify the two existing 0.86M PCA runs against their immutable manifests and
register them as `P0-real-sa` / `P0-real-sb`.

    PYTHONPATH=src python scripts/pod/register_p0_real.py \
        --runs artifacts/audit/p0_real --pack artifacts/stage3/ladder_uniform_probe \
        --out artifacts/audit/p0_real_registration.json

This is an aliasing step, not a training step: nothing is retrained and no weight
is touched. Every field below is read from the run's own `run_manifest.json` or
from the pack it names, and any mismatch is reported rather than reconciled — a
silently substituted checkpoint would make every downstream diagnostic
meaningless.

`kd_scope` deserves a word. The brief asks for `kd_scope=real_tokens`; the
implemented spelling is `"all"`, which `training.train.prediction_mask` resolves
to *every position whose content mask is true*, i.e. every real (non-padding)
token. The equivalence is checked here against the code rather than assumed from
the name.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.env import code_state, library_versions  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

# The E1 recovery recipe trains attention + FFN + norms, NOT embeddings or the
# LM head: 440,467,456 of 596,049,920 parameters. Taken from the tracked config,
# not assumed.
TRAINABLE_PATTERNS = [
    "\\.self_attn\\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\\.",
    "\\.mlp\\.(gate_proj|up_proj|down_proj)\\.",
    "input_layernorm", "post_attention_layernorm", "model\\.norm\\.",
]
EXPECTED = {
    "init_path": "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint",
    "init_sha256": "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54",
    "teacher": "Qwen/Qwen3-4B-Thinking-2507",
    "teacher_revision": "768f209d9ea81521153ed38c47d515654e938aea",
    "tokenizer_sha256": "7781771acc3798ee454c1253c751f930eb1c18c1c3df62e2552cc6f1d394f654",
    "total_params": 596049920,
    "trainable_params": 440467456,
    "rung": 860000,
    "kd_scope": "all",          # == "every real token"; verified against the code
    "ce_weight": 0.25,
    "kd_weight": 1.0,
    "kd_temperature": 1.0,
    "packing": "ladder",
    "block_len": 8192,
    "trainable_patterns": "all",
    "total_steps": 1023,
    "blocks_per_step": 2,
    "micro_blocks": 1,
}
ALIASES = {"e1_r0860k_sa_pca": "P0-real-sa", "e1_r0860k_sb_pca": "P0-real-sb"}


def kd_scope_is_every_real_token() -> tuple[bool, str]:
    """Check the claim against the implementation, not the config string."""
    import torch
    from aadistill.training.train import prediction_mask
    loss_mask = torch.zeros(2, 6, dtype=torch.bool)
    content = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
                           dtype=torch.bool)
    got = prediction_mask(loss_mask, "all", content)
    want = content[:, 1:]
    ok = bool(torch.equal(got, want))
    return ok, ("kd_scope 'all' selects exactly the real (non-padding) prediction "
                f"positions: {int(got.sum())} of {content.numel()} slots")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--configs", type=Path,
                    default=REPO_ROOT / "configs/stage3/e1")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import hashlib
    import numpy as np

    pack_meta = json.loads((args.pack / "ladder.json").read_text())
    rung_entry = next(r for r in pack_meta["rungs"]
                      if r["target_supervised_tokens"] == EXPECTED["rung"])
    scope_ok, scope_note = kd_scope_is_every_real_token()

    # The run manifests pin the rung by COUNTS, not by content. Compute the
    # membership and packing-order hashes here so the registration pins what the
    # manifests do not: which sessions, in which block order, with which tokens.
    n_blocks = rung_entry["n_blocks"]
    audit_rows = []
    with (args.pack / "audit.jsonl").open() as f:
        for i, line in enumerate(f):
            if i >= n_blocks:
                break
            audit_rows.append(json.loads(line))
    order = [[s["session_id"] for s in r["sessions"]] for r in audit_rows]
    membership = hashlib.sha256(
        json.dumps(sorted({s for blk in order for s in blk})).encode()).hexdigest()
    packing_order = hashlib.sha256(json.dumps(order).encode()).hexdigest()
    arrays = np.load(args.pack / "blocks.npz")
    rung_tokens = hashlib.sha256(
        arrays["input_ids"][:n_blocks].tobytes()).hexdigest()
    rung_ce = hashlib.sha256(arrays["ce_mask"][:n_blocks].tobytes()).hexdigest()
    rung_hashes = {
        "n_blocks": n_blocks,
        "n_sessions": len({s for blk in order for s in blk}),
        "membership_sha256": membership,
        "packing_order_sha256": packing_order,
        "rung_input_ids_sha256": rung_tokens,
        "rung_ce_mask_sha256": rung_ce,
    }
    print("rung hashes:", json.dumps(rung_hashes, indent=1))

    report, all_ok = {}, scope_ok
    for run_name, alias in ALIASES.items():
        run_dir = args.runs / run_name
        man = json.loads((run_dir / "run_manifest.json").read_text())
        cfg = man["config"]
        checks, mism = {}, []

        def check(field, got, want):
            ok = got == want
            checks[field] = {"expected": want, "actual": got, "ok": ok}
            if not ok:
                mism.append(field)

        # The manifest records the init as a PATH, not a hash (a P4 gap noted in
        # the output). Check the path, and separately check the tracked config
        # this run's config_sha256 must reproduce.
        check("student_source_path",
              str(man.get("student_source", "")).endswith(EXPECTED["init_path"]),
              True)
        check("teacher.model_id", man["teacher"].get("model_id"), EXPECTED["teacher"])
        check("teacher.revision", man["teacher"].get("revision"),
              EXPECTED["teacher_revision"])
        check("tokenizer_sha256", man.get("tokenizer_sha256"),
              EXPECTED["tokenizer_sha256"])
        check("total_params", man.get("total_params"), EXPECTED["total_params"])
        check("trainable_params", man.get("trainable_params"),
              EXPECTED["trainable_params"])
        tracked = args.configs / f"{run_name}.json"
        if tracked.is_file():
            from aadistill.infrastructure.manifest import sha256_json
            check("config_sha256_matches_tracked_config",
                  sha256_json(json.loads(tracked.read_text())),
                  man.get("config_sha256"))
        check("rung", cfg.get("rung"), EXPECTED["rung"])
        check("packing", cfg.get("packing"), EXPECTED["packing"])
        check("block_len", cfg.get("block_len"), EXPECTED["block_len"])
        check("trainable_patterns", cfg.get("trainable_patterns"),
              TRAINABLE_PATTERNS)
        for k in ("ce_weight", "kd_weight", "kd_temperature"):
            check(f"loss.{k}", cfg["loss"].get(k), EXPECTED[k])
        check("loss.kd_scope", cfg["loss"].get("kd_scope"), EXPECTED["kd_scope"])
        check("schedule.total_steps", cfg["schedule"].get("total_steps"),
              EXPECTED["total_steps"])
        check("batch.blocks_per_step", cfg["batch"].get("blocks_per_step"),
              EXPECTED["blocks_per_step"])
        check("batch.micro_blocks", cfg["batch"].get("micro_blocks"),
              EXPECTED["micro_blocks"])

        # the rung the manifest names must be the rung this pack cuts
        ladder = man.get("ladder", {})
        check("ladder.train_blocks", ladder.get("train_blocks"),
              rung_entry["n_blocks"])
        check("ladder.train_supervised_tokens",
              ladder.get("train_supervised_tokens"),
              rung_entry["actual_supervised_tokens"])

        seed = cfg.get("seed")
        model_cfg = json.loads((run_dir / "step_001023/model/config.json").read_text())
        gen_cfg = json.loads(
            (run_dir / "step_001023/model/generation_config.json").read_text())
        rope = model_cfg.get("rope_parameters") or {
            "rope_theta": model_cfg.get("rope_theta")}

        report[alias] = {
            "run_name": run_name,
            "seed": seed,
            "config_sha256": man.get("config_sha256"),
            "tokenizer_sha256": man.get("tokenizer_sha256"),
            "code_state": man.get("code_state"),
            "hardware": man.get("hardware"),
            "created_utc": man.get("created_utc"),
            "total_params": man.get("total_params"),
            "trainable_params": man.get("trainable_params"),
            "optimizer": cfg.get("optim"),
            "schedule": cfg.get("schedule"),
            "batch": cfg.get("batch"),
            "loss": cfg.get("loss"),
            "ladder": ladder,
            "data_manifests": man.get("data_manifests"),
            "model_config": {
                "hidden_size": model_cfg.get("hidden_size"),
                "num_hidden_layers": model_cfg.get("num_hidden_layers"),
                "intermediate_size": model_cfg.get("intermediate_size"),
                "num_attention_heads": model_cfg.get("num_attention_heads"),
                "num_key_value_heads": model_cfg.get("num_key_value_heads"),
                "head_dim": model_cfg.get("head_dim"),
                "vocab_size": model_cfg.get("vocab_size"),
                "tie_word_embeddings": model_cfg.get("tie_word_embeddings"),
                "max_position_embeddings": model_cfg.get("max_position_embeddings"),
                "rope": rope,
            },
            "generation_config": gen_cfg,
            "manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "checks": checks,
            "mismatches": mism,
            "verdict": "match" if not mism else "MISMATCH",
        }
        all_ok &= not mism

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "verdict": "registered" if all_ok else "NOT REGISTERED",
        "kd_scope_equivalence": {"ok": scope_ok, "note": scope_note,
                                 "config_spelling": "all",
                                 "brief_spelling": "real_tokens"},
        "pack": str(args.pack),
        "pack_rung_entry": rung_entry,
        "pack_outputs_sha256": pack_meta["outputs"],
        "rung_identity": rung_hashes,
        "manifest_gaps": [
            "run_manifest records student_source as a PATH, not a checkpoint hash",
            "run_manifest data_manifests is empty; the pack is named by path only",
            "the transformers version is not recorded (fixed for future runs "
            "2026-08-04, but absent from these two manifests)",
        ],
        "runs": report,
        "libraries": library_versions(),
        "code_state": code_state(REPO_ROOT),
        "note": ("Aliasing only. No weights were read, modified or retrained; "
                 "every field is quoted from the run's immutable manifest."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"verdict: {out['verdict']}")
    print(f"kd_scope 'all' == every real token: {scope_ok} — {scope_note}")
    for alias, r in report.items():
        print(f"\n{alias}  <- {r['run_name']}   {r['verdict']}")
        print(f"  seed {r['seed']}  config_sha {str(r['config_sha256'])[:16]}…  "
              f"manifest_sha {r['manifest_sha256'][:16]}…")
        print(f"  loss {r['loss']}")
        print(f"  ladder {r['ladder'].get('train_blocks')} blocks / "
              f"{r['ladder'].get('train_supervised_tokens')} supervised tokens")
        print(f"  rope {r['model_config']['rope']}  "
              f"ctx {r['model_config']['max_position_embeddings']}")
        for f in r["mismatches"]:
            print(f"    MISMATCH {f}: {r['checks'][f]}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
