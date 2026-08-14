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
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.recovery import recovery_scoring_contract  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

#: Frozen at preregistration 1d70a91a... (9b4229c8 before the 2026-08-13
#: re-emission, which moved identity digests only), transcribed here as constants
#: so the check cannot be satisfied by whatever file happens to be on the pod.
FROZEN = {
    "state_eval_v1": {
        "root": "artifacts/stage1/state_eval_v1",
        "content_sha256": "a1197205e43aad0e71c0e1bb436ee7babba3b5d8bb25b9c4d5c464f659db20fc",
        "manifest_sha256": "95204907efa7efc681ef334f223073da6ca35dbf889b029a7672f5817ab72b05",
        "items_sha256": "2a4a1d3bf8ed165b4cfa881f75488bcbbdee5e74d34afddb91fecbf76d1e106d",
    },
    # v2 supersedes v1, which was INVALID before first use: its `tools` kept the
    # xLAM source serialization and rendered 0/20. `content_sha256` is unchanged
    # by construction — it hashes id:prompt_sha256 pairs, so its equality is the
    # proof that the migration touched only the tools representation.
    "recovery_search_v2": {
        "root": "artifacts/stage3/recovery_search_v2",
        "content_sha256": "a1b22778b00d95b6aba358c14a5af5b559fd807bb371c92131eacca59479f323",
        "manifest_sha256": "58ae5c6dcbe32eb28c343a66830d7224a14537362deeeff2ce8219d0a31679d6",
        "tools_materialization_sha256": "3016f1c421a705cf72ce852a76f26ad6ef6dcaf84021a80de9c130bf224befdf",
        "items_sha256": None,          # this asset is seven per-set files
    },
}
FROZEN_SCORING_CONTRACT = "recovery_search_scoring@v2"
#: Re-pinned 2026-08-14 from `69591aab…` (itself re-pinned from `f76008d5…`),
#: still **@v2**. The recovery_search_v2 migration exposed a real scoring defect
#: — `as_openai_tools` reads xLAM-shaped entries, so under the canonical envelope
#: every declared tool name resolved to None and `tool_name_valid` was 0.0 for
#: every item — and the scorer's input is now normalized through
#: `aadistill.data.tools.normalize_tools`, which reads either representation and
#: still fails closed. The scoring *semantics* are unchanged, and that is
#: measured, not asserted: `validate_recovery_scoring.py` over nine policies x
#: 190 prompts reproduces **every number** of the pre-migration record. The
#: version stays at 2 because the metric did not move.
#:
#: Earlier note, still true: the contract is a digest
#: over whole files, and `src/aadistill/autoinit/recovery.py` gained the strict
#: observed-protocol reconstruction (`observe_recovery_protocol`,
#: `from_run_artifacts`) that the Stage-2 verification needs. No scoring function
#: changed, and that is not asserted but measured: re-running
#: `validate_recovery_scoring.py` over nine policies x 190 frozen prompts
#: reproduces every number of the `f76008d5…` record exactly
#: (`logs/autoinit_recovery_scoring_validation.json`). The version stays at 2
#: because the metric did not move; bumping it would falsely signal that it had.
FROZEN_SCORING_DIGEST = ("799398e716a429ab3cabef4372ea5aa9b40bc1ae34a015fc7e65a"
                         "2afa3dc80f6")


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
        if frozen.get("items_sha256") is not None:
            observed["items_sha256"] = sha256_file(root / "items.jsonl")
        if frozen.get("tools_materialization_sha256") is not None:
            # Recomputed from the items, not read from the manifest: the
            # manifest is what is being checked. This is the hash that
            # distinguishes v2 from the INVALID v1, since `content_sha256`
            # covers prompts only and is identical in both by construction.
            tool_rows = [json.loads(line) for line in
                         (root / "tool.jsonl").read_text().splitlines()
                         if line.strip()]
            observed["tools_materialization_sha256"] = hashlib.sha256(
                "".join(f"{r['id']}:{json.dumps(r['tools'], sort_keys=True)}\n"
                        for r in tool_rows).encode()).hexdigest()

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
