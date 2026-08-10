#!/usr/bin/env python3
"""Build the E8 treatment initialization from the frozen map, then stage it.

Runs on the dev box at $0, between the two paid pods. It exists as a script rather
than a sequence of commands because it is the one step where the causal variable
is actually applied, and every check below is something that would otherwise be
discovered on pod B after a 45-minute setup:

* the map must be the frozen one — 28 strictly increasing teacher layers, hash
  matching the search artifact, and **not** the positional map, because an
  identical map means there is no treatment to train;
* the resulting student config must be byte-identical to the control's, since only
  the depth map may change;
* the parameter count must match the control's exactly;
* the checkpoint must reload and its RoPE base must resolve to the teacher's.

Then it uploads and **verifies from the relay side** — re-downloads and re-hashes,
because an upload that returns 200 is not evidence.

    PYTHONPATH=src python scripts/training/build_and_stage_e8_init.py \\
        --frozen-map <fetched e8_frozen_depth_map.json> [--skip-upload]

Exit codes: 0 built, verified and staged; 9 a check failed or an upload did not
verify. On 9, do not launch pod B.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.manifest import sha256_file  # noqa: E402
from aadistill.init.nll_gate import checkpoint_fingerprint  # noqa: E402
from aadistill.init.sandwich import depth_span_map  # noqa: E402

RELAY = "AlphaAvatar/aadistill-artifacts"
PREFIX = "e8_init_20260810"
SEARCH_DIR = REPO_ROOT / "artifacts/stage1/e8_depth_search"
INIT_DIR = REPO_ROOT / "artifacts/stage1/e8_contribution_init_v1"
CONTROL_INIT = REPO_ROOT / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
CONTROL_SHA = "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"
CONTROL_PARAMS = 596_049_920
CONFIG = "configs/stage1/qwen3_0p6b_from_4b_thinking_contribution.json"
CKPT_FILES = ("config.json", "generation_config.json", "model.safetensors",
              "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 9


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen-map", required=True,
                    help="e8_frozen_depth_map.json fetched from pod A")
    ap.add_argument("--depth-map", default="",
                    help="depth_map.json from pod A (default: beside --frozen-map)")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--out", default="logs/e8_init_stage_manifest.json")
    args = ap.parse_args()

    frozen_path = Path(args.frozen_map)
    frozen = json.loads(frozen_path.read_text())
    dm_src = Path(args.depth_map) if args.depth_map else frozen_path.parent / "depth_map.json"
    if not dm_src.is_file():
        return fail(f"{dm_src} not found; pod A's depth_map.json is required")

    kept = frozen["kept_teacher_layers"]
    removed = frozen["removed_teacher_layers"]
    positional = sorted(set(range(36)) - {s["representative"]
                                         for s in depth_span_map(36, 28)})
    if len(kept) != 28 or kept != sorted(set(kept)) or not all(0 <= k < 36 for k in kept):
        return fail(f"frozen map is not 28 strictly increasing teacher layers: {kept}")
    if sorted(set(kept) | set(removed)) != list(range(36)):
        return fail("kept + removed is not a partition of the teacher's 36 layers")
    if sorted(removed) == positional:
        return fail("the frozen map IS the positional map — there is no treatment "
                    "to train. This is a real result; report it, do not build an "
                    "initialization identical to the control's")
    dm = json.loads(dm_src.read_text())
    if dm["kept_teacher_layers"] != kept:
        return fail("depth_map.json disagrees with the frozen map")
    if sha256_file(dm_src) != frozen["depth_map_sha256"]:
        return fail(f"depth_map.json hashes {sha256_file(dm_src)}, frozen record "
                    f"says {frozen['depth_map_sha256']}")

    print(f"frozen map verified: keeps {kept}")
    print(f"  removes {removed}  (positional removes {positional})")
    print(f"  calibration KL {frozen['primary_kl']:.6f} vs positional "
          f"{frozen['positional_baseline_primary_kl']:.6f}")

    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    for src, name in ((dm_src, "depth_map.json"),
                      (frozen_path, "e8_frozen_depth_map.json")):
        dest = SEARCH_DIR / name
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())

    print("\nbuilding the treatment initialization ...", flush=True)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/training/init_stage1.py"),
         "--config", str(REPO_ROOT / CONFIG)],
        cwd=REPO_ROOT, env={"PYTHONPATH": str(REPO_ROOT / "src"),
                            "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    if proc.returncode != 0:
        return fail(f"init_stage1 exited {proc.returncode}")

    ckpt = INIT_DIR / "checkpoint"
    fp = checkpoint_fingerprint(ckpt)
    control_fp = checkpoint_fingerprint(CONTROL_INIT)
    manifest = json.loads((INIT_DIR / "manifest.json").read_text())
    diag = manifest["init_diagnostics"]

    if control_fp["model_sha256"] != CONTROL_SHA:
        return fail("the control initialization on this box is not the pinned one")
    if fp["config_sha256"] != control_fp["config_sha256"]:
        return fail("treatment config differs from the control's — only the depth "
                    "map may change")
    if fp["model_sha256"] == control_fp["model_sha256"]:
        return fail("treatment weights are identical to the control's")
    if manifest["student"]["num_parameters"] != CONTROL_PARAMS:
        return fail(f"parameter count {manifest['student']['num_parameters']} != "
                    f"{CONTROL_PARAMS}")
    if diag["depth_map_source"] != "explicit_kept_layers":
        return fail(f"init used depth map source {diag['depth_map_source']!r}")
    if diag["kept_teacher_layers"] != kept:
        return fail("the built init does not use the frozen map")
    if abs(manifest["student"]["resolved_rope_base"] - 5_000_000) / 5e6 > 1e-2:
        return fail(f"resolved RoPE base {manifest['student']['resolved_rope_base']}")

    print(f"\ntreatment init built and verified")
    print(f"  model.safetensors sha256 {fp['model_sha256']}")
    print(f"  parameters {manifest['student']['num_parameters']:,}  "
          f"config sha256 {fp['config_sha256'][:16]}… (== control)")
    print(f"  projection energy {diag['projection']['energy_captured_frac']:.10f}")

    staged = []
    if not args.skip_upload:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        uploads = [(ckpt / f, f"{PREFIX}/e8_contribution_init_v1/checkpoint/{f}")
                   for f in CKPT_FILES]
        uploads += [(INIT_DIR / "manifest.json",
                     f"{PREFIX}/e8_contribution_init_v1/manifest.json"),
                    (SEARCH_DIR / "depth_map.json", f"{PREFIX}/depth_map.json"),
                    (SEARCH_DIR / "e8_frozen_depth_map.json",
                     f"{PREFIX}/e8_frozen_depth_map.json")]
        search_report = SEARCH_DIR / "depth_search.json"
        rounds = SEARCH_DIR / "rounds.jsonl"
        for p, name in ((search_report, "depth_search.json"),
                        (rounds, "rounds.jsonl")):
            if p.is_file():
                uploads.append((p, f"{PREFIX}/{name}"))
        for local, remote in uploads:
            if not local.is_file():
                return fail(f"{local} missing; cannot stage a partial init")
            sha = sha256_file(local)
            print(f"uploading {local.name} ({local.stat().st_size / 1e6:.1f} MB)",
                  flush=True)
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                            repo_id=RELAY, repo_type="model")
            back = sha256_file(Path(hf_hub_download(RELAY, remote, repo_type="model")))
            ok = back == sha
            staged.append({"local": str(local.relative_to(REPO_ROOT)),
                           "relay": remote, "sha256": sha,
                           "relay_sha256": back, "verified": ok})
            print(f"  roundtrip {'OK' if ok else 'MISMATCH'}", flush=True)
            if not ok:
                return fail(f"{remote} did not verify from the relay")

    record = {
        "artifact": "e8_treatment_init_stage_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "frozen_map": {k: frozen[k] for k in (
            "kept_teacher_layers", "removed_teacher_layers", "removal_order",
            "primary_kl", "positional_baseline_primary_kl",
            "lower_kl_than_positional", "depth_map_sha256",
            "search_report_sha256", "calibration_content_sha256")},
        "treatment_init": {
            "path": str(ckpt.relative_to(REPO_ROOT)),
            "model_sha256": fp["model_sha256"],
            "config_sha256": fp["config_sha256"],
            "num_parameters": manifest["student"]["num_parameters"],
            "resolved_rope_base": manifest["student"]["resolved_rope_base"],
            "init_manifest_sha256": sha256_file(INIT_DIR / "manifest.json"),
        },
        "control_init": {"model_sha256": control_fp["model_sha256"],
                         "config_sha256": control_fp["config_sha256"]},
        "config_identical_to_control": True,
        "relay_prefix": PREFIX,
        "staged": staged,
        "all_verified": all(s["verified"] for s in staged) if staged else None,
    }
    (REPO_ROOT / args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(f"\n-> {args.out}")
    print(f"\nPOD B LAUNCH ARGUMENT:\n  --treatment-init-sha256 {fp['model_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
