#!/usr/bin/env python3
"""Make every irreplaceable E8 input durable off the dev box, and prove it.

E8 found the Stage 0 activation cache missing — 1.95 GB that Stage 1 cannot build
any initialization without, absent from the dev box and never on the relay. It was
recoverable only because the pipeline is deterministic and its hash was logged.
Regenerating it and leaving it dev-box-only again would be repeating the mistake
with the evidence in hand.

So this stages the whole irreplaceable set, and **verifies from the relay side**:
an upload that returns 200 is not evidence. Every file is re-downloaded and
re-hashed against the local value.

What counts as irreplaceable here is not "expensive" but "unreproducible or
gating":

| artifact | why |
| --- | --- |
| `activation_stats.safetensors` | Stage 1 cannot construct any init without it |
| its manifest | carries the token count and per-sample record |
| `warmup_v1.jsonl` | **Stage 0's input.** Without it the cache is not regenerable at all, and it streams four pinned datasets |
| `holdout_v1.jsonl` | the historical NLL series runs back to the Stage 1 gate through it |
| the E8 calibration set | frozen selector input; its leakage proofs are part of the record |

Already on the relay and re-verified rather than re-uploaded: the pinned control
initialization, the canonical ladder pack, corpus v2, the E7 FineWeb streams.

    PYTHONPATH=src python scripts/data/stage_e8_inputs.py --verify-roundtrip

Exit codes: 0 staged and verified; 9 a hash mismatch, a missing file, or an
upload the relay refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

RELAY = "AlphaAvatar/aadistill-artifacts"
REPO_TYPE = "model"
PREFIX = "e8_inputs_20260810"

# local path -> relay path under PREFIX. Ordered small-first so a quota refusal on
# the 1.95 GB file still leaves the cheap-but-critical inputs durable.
UPLOAD = [
    ("data/warmup/warmup_v1.jsonl", "warmup/warmup_v1.jsonl"),
    ("data/warmup/warmup_v1.manifest.json", "warmup/warmup_v1.manifest.json"),
    ("data/warmup/holdout_v1.jsonl", "warmup/holdout_v1.jsonl"),
    ("data/warmup/holdout_v1.manifest.json", "warmup/holdout_v1.manifest.json"),
    ("artifacts/stage1/e8_calibration_v1/items.jsonl", "calibration_v1/items.jsonl"),
    ("artifacts/stage1/e8_calibration_v1/docs.jsonl", "calibration_v1/docs.jsonl"),
    ("artifacts/stage1/e8_calibration_v1/general_docs.jsonl",
     "calibration_v1/general_docs.jsonl"),
    ("artifacts/stage1/e8_calibration_v1/general_docs.manifest.json",
     "calibration_v1/general_docs.manifest.json"),
    ("artifacts/stage1/e8_calibration_v1/manifest.json", "calibration_v1/manifest.json"),
    ("artifacts/stage1/e8_calibration_v1/leakage.json", "calibration_v1/leakage.json"),
    ("artifacts/stage1/e8_calibration_v1/general_disjointness.json",
     "calibration_v1/general_disjointness.json"),
    ("artifacts/stage0/qwen3_4b_thinking_v1/manifest.json",
     "stage0/qwen3_4b_thinking_v1/manifest.json"),
    ("artifacts/stage0/qwen3_4b_thinking_v1/activation_stats.safetensors",
     "stage0/qwen3_4b_thinking_v1/activation_stats.safetensors"),
]

# Already durable; re-verified against the recorded hashes rather than re-sent.
EXPECTED_PRESENT = {
    "stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors":
        "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54",
    "stage3_recovery_corpus_v2/ladder_uniform/blocks.npz": None,
    "stage3_recovery_corpus_v2/ladder_uniform/ladder.json": None,
    "stage3_recovery_corpus_v2/ladder_uniform/audit.jsonl": None,
    "e7_streams_20260809/e7_fineweb_val/blocks.npz": None,
    "e7_streams_20260809/e7_fineweb_val/manifest.json": None,
}
CANONICAL_STAGE0_SHA = ("aaeb2e4c1ec67e6f6dd21ca40eceb0c193a9da5b010e8d12fcd5d24"
                        "376cc47c1")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relay", default=RELAY)
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--verify-roundtrip", action="store_true")
    ap.add_argument("--out", default="logs/e8_relay_manifest.json")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()

    # The Stage 0 cache is the reason this script exists; refuse to stage a file
    # that is not the canonical one.
    stats = REPO_ROOT / "artifacts/stage0/qwen3_4b_thinking_v1/activation_stats.safetensors"
    if not stats.is_file():
        print(f"MISSING {stats}", file=sys.stderr)
        return 9
    local_stats_sha = sha256_file(stats)
    if local_stats_sha != CANONICAL_STAGE0_SHA:
        print(f"REFUSING: local Stage 0 cache hashes {local_stats_sha}, not the "
              f"canonical {CANONICAL_STAGE0_SHA}", file=sys.stderr)
        return 9

    staged, failures = [], []
    for local_rel, relay_rel in UPLOAD:
        local = REPO_ROOT / local_rel
        if not local.is_file():
            failures.append({"file": local_rel, "error": "missing locally"})
            continue
        sha = sha256_file(local)
        size = local.stat().st_size
        path_in_repo = f"{args.prefix}/{relay_rel}"
        print(f"uploading {local_rel} ({size / 1e6:.1f} MB) -> {path_in_repo}",
              flush=True)
        try:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                            repo_id=args.relay, repo_type=REPO_TYPE)
        except Exception as exc:                              # noqa: BLE001
            failures.append({"file": local_rel, "error": repr(exc)[:400]})
            print(f"  FAILED: {exc!r}"[:400], flush=True)
            continue
        record = {"local": local_rel, "relay": path_in_repo,
                  "bytes": size, "sha256": sha, "roundtrip_verified": False}
        if args.verify_roundtrip:
            try:
                back = hf_hub_download(args.relay, path_in_repo,
                                       repo_type=REPO_TYPE)
                back_sha = sha256_file(Path(back))
                record["relay_sha256"] = back_sha
                record["roundtrip_verified"] = back_sha == sha
                if not record["roundtrip_verified"]:
                    failures.append({"file": local_rel,
                                     "error": f"relay sha {back_sha} != {sha}"})
                print(f"  roundtrip {'OK' if record['roundtrip_verified'] else 'MISMATCH'}",
                      flush=True)
            except Exception as exc:                          # noqa: BLE001
                failures.append({"file": local_rel,
                                 "error": f"roundtrip failed: {exc!r}"[:300]})
        staged.append(record)

    present = set(api.list_repo_files(args.relay, repo_type=REPO_TYPE))
    already = []
    for path, expect in EXPECTED_PRESENT.items():
        ok = path in present
        entry = {"relay": path, "present": ok}
        if not ok:
            failures.append({"file": path, "error": "expected already on the relay"})
        elif expect and args.verify_roundtrip:
            back_sha = sha256_file(Path(hf_hub_download(
                args.relay, path, repo_type=REPO_TYPE)))
            entry["relay_sha256"] = back_sha
            entry["matches_record"] = back_sha == expect
            if not entry["matches_record"]:
                failures.append({"file": path,
                                 "error": f"relay sha {back_sha} != recorded {expect}"})
        already.append(entry)

    manifest = {
        "artifact": "e8_relay_input_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "relay": args.relay, "prefix": args.prefix,
        "purpose": "durability for every irreplaceable E8 input; the Stage 0 cache "
                   "was lost once and must never be dev-box-only again",
        "stage0_cache_sha256": local_stats_sha,
        "stage0_cache_matches_canonical": True,
        "staged": staged,
        "already_present": already,
        "failures": failures,
        "all_verified": not failures and all(
            s["roundtrip_verified"] for s in staged) if args.verify_roundtrip
        else not failures,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nstaged {len(staged)}/{len(UPLOAD)}; "
          f"{len(failures)} failure(s) -> {out.relative_to(REPO_ROOT)}")
    for f in failures:
        print(f"  FAIL {f['file']}: {f['error'][:200]}")
    return 0 if manifest["all_verified"] else 9


if __name__ == "__main__":
    raise SystemExit(main())
