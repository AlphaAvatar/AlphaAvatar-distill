#!/usr/bin/env python3
"""Restore H1's fixed initialization from the relay and prove its identity. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.restore_initialization \
        [--dest <dir>] [--record]

The initialization is the maintainer's fixed choice (Stage-0 d4 rank 1; decision
2026-10-21). Its identity is read from the records that own it, never typed:

* file list, sizes and per-file sha256 -- the d4 finalists manifest;
* `arch_signature`, `config_sha256`, `tokenizer_sha256`, `artifact_digest` --
  the leaf's own row in the d4 search journal (`states.jsonl`).

The relay copy is downloaded into `--dest` (default: the dev-box path the Stage-0
session configs name for this candidate), every file's sha256 is checked, and the
checkpoint identity is REBUILT from the bytes that arrived by the same function a
pod uses (`aadistill.runtime.leaf_durability.verify_transferred_leaf`). Relay
metadata and the read-back are reported separately. With `--record`, the result
is written to `logs/stages/stage-2/reasoning_data/assets/d4_top1_materialization.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A

from aadistill.initialization.adapters import register_builtin_adapters
from aadistill.initialization.specs.arch import get_adapter
from aadistill.runtime.leaf_durability import verify_transferred_leaf

RELAY = "AlphaAvatar/aadistill-artifacts"
STATE_ID = "1301deb71685aa1ca5c7b6cf2c491ac4"
DOSE_RUN = A.S0 / "runs/dose_d4_20261020_212310"
JOURNAL = Path("/home/ecs-user/aad-artifacts/stage0_calvol/dose_d4_20261020_212310/store/d4/search/search/states.jsonl")
DEFAULT_DEST = A.REPO / f"artifacts/stages/stage-0/calibration_volume/candidates/d4/{STATE_ID}"
RECORD = A.LOG_DIR / "assets/d4_top1_materialization.json"


def manifest_files() -> list[dict]:
    m = A.read_json(DOSE_RUN / "evidence/d4/finalists/finalists_manifest.json")
    return [f for f in m["files"] if f["state_id"] == STATE_ID]


def journal_identity() -> dict:
    for line in open(JOURNAL):
        if STATE_ID in line:
            row = json.loads(line)
            if row.get("state_id") == STATE_ID and row.get("artifact"):
                return row["artifact"] | {"materialization": row.get("materialization")}
    raise SystemExit(f"{JOURNAL}: no artifact row for {STATE_ID}")


def relay_metadata(token: str) -> dict:
    from huggingface_hub import HfApi
    info = HfApi(token=token).get_paths_info(
        RELAY, [f["path"] for f in manifest_files()], expand=True)
    return {i.path: {"size": i.size, "lfs_sha256": i.lfs.sha256 if i.lfs else None,
                     "blob_id": i.blob_id} for i in info}


def download(dest: Path, token: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for f in manifest_files():
        got = Path(hf_hub_download(RELAY, f["path"], token=token,
                                   local_dir=str(dest / ".relay")))
        target = dest / f["name"]
        shutil.move(str(got), target)
        sha = A.sha256_file(target)
        out.append({"name": f["name"], "bytes": target.stat().st_size, "sha256": sha,
                    "manifest_sha256": f["sha256"], "matched": sha == f["sha256"]
                    and target.stat().st_size == f["bytes"]})
    shutil.rmtree(dest / ".relay", ignore_errors=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args(argv)
    token = os.environ.get("HF_TOKEN") or Path.home().joinpath(".cache/huggingface/token").read_text().strip()
    dest = Path(args.dest)
    ident = journal_identity()
    meta = relay_metadata(token)
    files = download(dest, token)
    register_builtin_adapters()
    verdict = verify_transferred_leaf(dest, ident, adapter=get_adapter("qwen3"))
    ok = all(f["matched"] for f in files) and verdict["matched"] and verdict["weights_digest_matched"]
    doc = {
        "schema": "aadistill.stage2.reasoning_data.initialization_materialization/v1",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decision": "logs/budget/decisions.md 2026-10-21 (maintainer: H1 fixes Stage-0 D4 Top-1)",
        "method_identity": {
            "baseline": "stage1_init.d1_target_aware.scored_v1@v1",
            "selection": "configs/shared/baselines/selection.json (unchanged; the baseline record is frozen and is NOT edited by this record)",
            "execution_fingerprint": A.read_json(DOSE_RUN / "evidence/d4/search/search_record.json")["session"]["baseline"]["numerical_execution_fingerprint"],
            "prediction_rows": "scored_v1",
            "search_evidence": ["logs/stages/stage-0/calibration_volume/runs/dose_d4_20261020_212310/evidence/d4/search/search_record.json",
                                "logs/stages/stage-0/calibration_volume/runs/dose_d4_20261020_212310/evidence/d4/search/stage1_selection.json",
                                str(JOURNAL)],
        },
        "materialization_identity": {
            "state_id": STATE_ID,
            "path": ident.get("materialization", {}) and ident["materialization"],
            "checkpoint_sha256": next(f["sha256"] for f in manifest_files() if f["name"] == "model.safetensors"),
            "artifact_digest": ident["artifact_digest"], "weights_digest": ident["weights_digest"],
            "config_sha256": ident["config_sha256"], "arch_signature": ident["arch_signature"],
            "tokenizer_sha256": ident["tokenizer_sha256"], "num_parameters": ident["num_parameters"],
            "identity_owners": ["logs/stages/stage-0/calibration_volume/runs/dose_d4_20261020_212310/evidence/d4/finalists/finalists_manifest.json",
                                f"{JOURNAL} (states.jsonl artifact row)"],
        },
        "tokenizer_binding": {
            "stored_with_checkpoint": False,
            "artifact_digest_covers_tokenizer": "no (tokenizer_sha256 null in the journal, as for every Stage-0 candidate)",
            "source": "artifacts/stages/stage-1/qwen3_0p6b_init_v0/checkpoint (the tokenizer_source of every E-series and Stage-0 recovery)",
            "tokenizer_json_sha256": A.sha256_file(A.TOKENIZER_DIR / "tokenizer.json"),
            "chat_template_sha256": A.sha256_file(A.TOKENIZER_DIR / "chat_template.jinja"),
        },
        "relay_metadata": meta,
        "read_back": {"dest": str(dest.relative_to(A.REPO)) if dest.is_relative_to(A.REPO) else str(dest),
                      "files": files, "identity": {k: verdict[k] for k in
                                                   ("artifact_digest", "recorded_digest", "matched",
                                                    "weights_digest", "weights_digest_matched", "config_sha256")}},
        "verified": ok,
        "copies": ["relay (HF private) -- metadata above", "dev box (this read-back) -- " + ("VERIFIED" if ok else "FAILED")],
        "authorizes": "nothing",
    }
    if args.record:
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        print(RECORD.relative_to(A.REPO))
    print("VERIFIED" if ok else "FAILED", json.dumps(doc["read_back"]["identity"]))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
