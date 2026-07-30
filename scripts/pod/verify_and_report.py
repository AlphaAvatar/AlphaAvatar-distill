#!/usr/bin/env python3
"""Independently verify that a run's artifacts reached the private HF repo.

``verify --run <name>``
            Checks that every expected artifact exists in the relay and that its
            content matches the sha256 computed on the pod. Large LFS files are
            checked via the hub's recorded LFS sha256 (no multi-GB download);
            small files are downloaded and hashed locally. Exits non-zero unless
            every file matches. ``scripts/pod/orchestrate.sh`` treats this as the
            safety condition for deleting a paid pod.

**Verify-only.** The former ``report`` subcommand hardcoded one experiment's
pre-registered decision rules (the 2026-07-28 packing control), so pointing it
at any later session auto-generated a confident write-up of the wrong
hypothesis. It was removed in the 2026-07-31 cleanup; a session supplies its own
write-up via ``REPORT_CMD`` in ``run_env.sh``. Git history has it at ``866dac2``.

Nothing here is specific to one run: names, paths and step tags come from the
arguments and from ``run_env.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HF_REPO = "AlphaAvatar/aadistill-artifacts"
# Must match run_env.sh's HF_PREFIX_BASE, which is what post_run.sh uploads
# under. It was a hardcoded "stage3" here while a session could set any prefix,
# so a session that changed it saw all four arms fail verification against a
# path nothing had ever written to — a false negative that, by design, blocks
# teardown and leaves a paid pod running.
HF_PREFIX_BASE = os.environ.get("HF_PREFIX_BASE", "stage3")

RUN_FILES = [
    "train_log.jsonl",
    "run_manifest.json",
    "eval_holdout_v1.json",
    "eval_holdout_v1_int8.json",
    "eval_holdout_v1_int8_decoder.json",
    "eval_behavior_v0.json",
    "eval_behavior_v0.generations.jsonl",
    "gen_smoke.json",
    "console.log",
]
MODEL_FILES = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]




def run_dir(run: str) -> Path:
    return REPO / "artifacts/stage3" / run


def hash_file_for(run: str) -> Path:
    matches = sorted((REPO / "artifacts/stage3").glob(f"{run}_artifact_hashes_*.txt"))
    if not matches:
        sys.exit(f"FAIL: no pod hash file for run {run}")
    return matches[-1]


def load_pod_hashes(run: str) -> dict[str, str]:
    """basename -> sha256, from the hash file computed on the pod."""
    path = hash_file_for(run)
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, p = line.partition("  ")
        if not p:
            continue
        out[Path(p.strip()).name] = digest.strip()
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def step_tag_of(run: str) -> str:
    """Final checkpoint tag, read from the run's own manifest/log."""
    ev = read_events(run)
    ends = [e for e in ev if e.get("event") == "run_end"]
    steps = ends[0].get("steps") if ends else None
    if steps is None:
        starts = [e for e in ev if e.get("event") == "run_start"]
        steps = starts[0].get("total_steps") if starts else 0
    return f"step_{int(steps):06d}"


def cmd_verify(run: str) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    pod_hashes = load_pod_hashes(run)
    prefix = f"{HF_PREFIX_BASE}/{run}"
    ckpt_tag = step_tag_of(run)
    hashfile_name = hash_file_for(run).name

    expected = [f"{prefix}/{n}" for n in RUN_FILES]
    expected += [f"{prefix}/{ckpt_tag}/model/{n}" for n in MODEL_FILES]
    expected.append(f"{prefix}/{hashfile_name}")

    tree = {
        item.path: item
        for item in api.list_repo_tree(
            HF_REPO, path_in_repo=prefix, recursive=True,
            expand=True, repo_type="model",
        )
        if getattr(item, "size", None) is not None
    }

    problems: list[str] = []
    checked = 0
    print(f"remote files found under {prefix}/: {len(tree)}")

    for path in expected:
        item = tree.get(path)
        if item is None:
            problems.append(f"MISSING on HF: {path}")
            continue
        name = Path(path).name
        want = pod_hashes.get(name)
        if want is None:
            # The hash file itself is not listed inside itself.
            if name == hashfile_name:
                print(f"  ok (presence+size {item.size}) {path}")
                checked += 1
                continue
            problems.append(f"no pod-side sha256 recorded for {name}")
            continue

        lfs = getattr(item, "lfs", None)
        if isinstance(lfs, dict):
            got = lfs.get("sha256")
        else:
            got = getattr(lfs, "sha256", None) if lfs is not None else None
        how = "lfs-sha256"
        if got is None:
            local = Path(
                hf_hub_download(
                    HF_REPO, path, repo_type="model",
                    cache_dir=str(REPO / "artifacts/stage3/.verify_cache"),
                )
            )
            got = sha256_file(local)
            how = "downloaded"

        if got == want:
            print(f"  ok ({how}, {item.size} B) {path}")
            checked += 1
        else:
            problems.append(f"HASH MISMATCH {path}: hf={got} pod={want}")

    if problems:
        print("\nVERIFICATION FAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    print(f"\nVERIFICATION PASSED: {checked}/{len(expected)} files match pod sha256")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="check a run's upload against pod hashes")
    v.add_argument("--run", required=True)
    args = ap.parse_args()
    return cmd_verify(args.run)


if __name__ == "__main__":
    sys.exit(main())
