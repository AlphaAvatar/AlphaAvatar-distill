#!/usr/bin/env python
"""Stage E6's six evaluation checkpoints and verify every one against its hash.

    /opt/train/bin/python scripts/pod/e6_stage_checkpoints.py \
        --registration logs/e6_registration.json --relay-dest /workspace/ckpt \
        --devbox-src /workspace/ckpt_local --init <stage1 checkpoint> \
        --out artifacts/audit/e6_checkpoint_manifest.json

Two stores feed one evaluation. Four arms live on the relay; two (`e1_r2960k_sb`
and `e1_r5500k_sb`) exist only on the dev box, because the relay hit its private
LFS limit before Experiment 1 finished uploading and deleting objects there
cannot reclaim quota. The launcher scp's those two; this script treats both
routes identically and refuses either on a hash mismatch.

`save_checkpoint` never wrote tokenizer files, so each staged model directory is
dressed from the Stage 1 init — the same files every one of these arms trained
and was previously evaluated with. That is a copy of pinned assets, not a
substitution: the tokenizer, template and generation config are hashed into the
manifest so the record shows exactly which bytes were used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

DRESS = ("config.json", "generation_config.json", "tokenizer.json",
         "tokenizer_config.json", "chat_template.jinja")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_relay(prefix: str, dest: Path, token: str) -> Path:
    from huggingface_hub import hf_hub_download
    dest.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download("AlphaAvatar/aadistill-artifacts",
                          f"{prefix}/model.safetensors",
                          repo_type="model", token=token)
    shutil.copy(src, dest / "model.safetensors")
    return dest


def stage_devbox(src_dir: Path, dest: Path) -> Path:
    if not (src_dir / "model.safetensors").is_file():
        sys.exit(f"dev-box checkpoint missing: {src_dir}/model.safetensors")
    dest.mkdir(parents=True, exist_ok=True)
    if (src_dir / "model.safetensors").resolve() != (dest / "model.safetensors").resolve():
        shutil.copy(src_dir / "model.safetensors", dest / "model.safetensors")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True, type=Path)
    ap.add_argument("--relay-dest", required=True, type=Path)
    ap.add_argument("--devbox-src", required=True, type=Path)
    ap.add_argument("--init", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    reg = json.loads(args.registration.read_text())
    token = Path("/workspace/hf/token").read_text().strip() \
        if Path("/workspace/hf/token").is_file() else None

    dressing = {f: sha256(args.init / f) for f in DRESS}
    manifest, failures = {}, []
    for alias, arm in sorted(reg["arms"].items()):
        if not arm["generate"]:
            continue                                   # re-scored, never staged
        store, ref = arm["source"]
        dest = args.relay_dest / arm["run"] / "model"
        if store == "relay":
            stage_relay(ref, dest, token)
        elif store == "devbox":
            stage_devbox(args.devbox_src / arm["run"], dest)
        else:
            sys.exit(f"{alias}: unknown store {store!r}")

        got = sha256(dest / "model.safetensors")
        ok = got == arm["weights_sha256"]
        if not ok:
            failures.append(f"{alias}: {got} != {arm['weights_sha256']}")
        for f in DRESS:
            shutil.copy(args.init / f, dest / f)
        manifest[alias] = {
            "run": arm["run"], "seed": arm["seed"], "rung": arm["rung"],
            "step": arm["step"], "store": store, "source_ref": ref,
            "path": str(dest), "weights_sha256": got,
            "weights_sha256_matches_registration": ok,
            "dressed_from": str(args.init), "dressing_sha256": dressing,
        }
        print(f"  {alias:14s} {store:6s} {got[:16]}… "
              f"{'OK' if ok else 'MISMATCH'}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"registration_sha256": reg["registration_sha256"],
         "arms": manifest, "failures": failures}, indent=2) + "\n")
    if failures:
        sys.exit("CHECKPOINT HASH MISMATCH:\n  " + "\n  ".join(failures))
    expected = sum(1 for a in reg["arms"].values() if a["generate"])
    if len(manifest) != expected:
        sys.exit(f"staged {len(manifest)} arms, registration declares {expected}")
    print(f"all {len(manifest)} evaluation checkpoints staged and hash-verified")


if __name__ == "__main__":
    main()
