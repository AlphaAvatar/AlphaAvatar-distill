"""Verify the frozen search assets against PREREGISTERED constants. Setup gate.

    PYTHONPATH=src python scripts/autoinit/verify_frozen_assets.py

Runs during pod setup, before any scientific measurement. It answers "are these
the assets Phase A preregistered", which is a different question from "is this
file self-consistent" — and only the first one matters. Reading
`manifest["content_sha256"]` and comparing it to itself proves the second and
looks like the first.

**Three hash conventions appear in these two manifests and they are not
interchangeable.** Getting this wrong produces a mismatch that looks like
corruption and is not; it has already cost this project one false alarm.

    content_sha256   a hash over the loaded ITEMS (tokens/prompts), computed by
                     the builder from content, not from any file
    manifest_sha256  sha256_json over the manifest with `manifest_sha256`
                     removed — self-referential, so it can live inside the file
                     it describes. NOT sha256_file of the raw bytes.
    items_sha256     sha256_file over the raw bytes of items.jsonl

Any mismatch is a setup blocker: record it, do not enter Stage 1, tear down.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.recovery import recovery_scoring_contract  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

#: Frozen at preregistration 9b4229c8..., transcribed here as constants so the
#: check cannot be satisfied by whatever file happens to be on the pod.
FROZEN = {
    "state_eval_v1": {
        "root": "artifacts/stage1/state_eval_v1",
        "content_sha256": "a1197205e43aad0e71c0e1bb436ee7babba3b5d8bb25b9c4d5c464f659db20fc",
        "manifest_sha256": "95204907efa7efc681ef334f223073da6ca35dbf889b029a7672f5817ab72b05",
        "items_sha256": "2a4a1d3bf8ed165b4cfa881f75488bcbbdee5e74d34afddb91fecbf76d1e106d",
    },
    "recovery_search_v1": {
        "root": "artifacts/stage3/recovery_search_v1",
        "content_sha256": "a1b22778b00d95b6aba358c14a5af5b559fd807bb371c92131eacca59479f323",
        "manifest_sha256": "72d8c0535e7752faf704d9075b7835a47610fd3cd26866cf5be7d48eb7b40ad1",
        "items_sha256": None,          # this asset is seven per-set files
    },
}
FROZEN_SCORING_CONTRACT = "recovery_search_scoring@v2"
FROZEN_SCORING_DIGEST = ("f76008d5459c781cdfd0f11e39fc379c74af641f7567a168c10bf"
                         "48e6a3e66fb")


def canonical_manifest_sha256(path: Path) -> str:
    """The producer's convention: sha256_json of the manifest minus its own hash."""
    manifest = json.loads(path.read_text())
    return sha256_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO_ROOT))
    ap.add_argument("--out", default="artifacts/audit/autoinit_preflight/"
                                     "frozen_asset_verification.json")
    args = ap.parse_args()
    repo = Path(args.repo)

    report = {"schema": "aadistill.autoinit.frozen_asset_verification/v1",
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "expected_from": ("preregistered constants in this file, NOT from "
                                "the manifests on disk"),
              "assets": {}, "problems": []}

    for name, frozen in FROZEN.items():
        root = repo / frozen["root"]
        entry = {"root": frozen["root"], "present": root.is_dir(), "checks": {}}
        if not entry["present"]:
            report["problems"].append(f"{name}: {frozen['root']} is absent")
            report["assets"][name] = entry
            continue
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        observed = {
            "content_sha256": manifest.get("content_sha256"),
            "manifest_sha256": canonical_manifest_sha256(manifest_path),
        }
        if frozen["items_sha256"] is not None:
            observed["items_sha256"] = sha256_file(root / "items.jsonl")

        for key, expected in frozen.items():
            if key == "root" or expected is None:
                continue
            got = observed.get(key)
            ok = got == expected
            entry["checks"][key] = {"expected": expected, "observed": got, "match": ok}
            if not ok:
                report["problems"].append(
                    f"{name}.{key}: expected {expected}, observed {got}")
        # The manifest must also agree with its own recorded value, which
        # catches an edit that updated the payload but not the self-hash.
        entry["manifest_self_consistent"] = (
            manifest.get("manifest_sha256") == observed["manifest_sha256"])
        if not entry["manifest_self_consistent"]:
            report["problems"].append(f"{name}: manifest does not match its own "
                                      "manifest_sha256")
        entry["hash_conventions"] = {
            "content_sha256": "builder hash over loaded items",
            "manifest_sha256": "sha256_json over manifest minus manifest_sha256",
            "items_sha256": "sha256_file over raw items.jsonl bytes",
        }
        report["assets"][name] = entry

    contract = recovery_scoring_contract(repo)
    report["scoring_contract"] = {
        "expected": FROZEN_SCORING_CONTRACT,
        "expected_digest": FROZEN_SCORING_DIGEST,
        "observed": contract["contract"], "observed_digest": contract["digest"],
        "match": (contract["contract"] == FROZEN_SCORING_CONTRACT
                  and contract["digest"] == FROZEN_SCORING_DIGEST),
    }
    if not report["scoring_contract"]["match"]:
        report["problems"].append(
            f"scoring contract is {contract['contract']} {contract['digest']}, "
            f"expected {FROZEN_SCORING_CONTRACT} {FROZEN_SCORING_DIGEST}")

    report["passed"] = not report["problems"]
    report["report_sha256"] = sha256_json(report)
    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"],
                      "problems": report["problems"][:5]}, indent=2))
    if not report["passed"]:
        print("BLOCKER: the pod is not running the preregistered assets. Do not "
              "enter Stage 1; record evidence and tear down.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
