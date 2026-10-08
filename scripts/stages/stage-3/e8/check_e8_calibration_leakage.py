#!/usr/bin/env python3
"""Prove the E8 calibration set cannot see the decision it will be judged by.

`check_stream_disjointness.py` already proves the general-text side, because that
side ships a `docs.jsonl` of raw documents with content hashes. The templated side
needs its own proof, and a stricter one, because of how this project's promotion
evaluation is built: the frozen 150-prompt behaviour battery is **stratified-
sampled from the 0.86M rung's verified-correct sessions**, and the 0.86M rung is a
prefix of the 2.96M rung the E8 arms train on. A depth map calibrated on a
session that the battery later asks about would be a map chosen partly on the test.

Four independent checks, all fail-closed:

1. **source-id separation** — no calibration session shares a `source_id` with
   anything the pack consumes through the excluded rung. Recomputed here from
   `audit.jsonl` rather than trusting the builder's own filter.
2. **candidate separation** — no calibration session's `candidate_sha256` appears
   among the consumed sessions'.
3. **prompt-content separation** — no calibration prompt text hashes to the same
   value as any consumed session's prompt, any `eval_behavior_v0` prompt, or any
   prompt in the frozen capability battery. This is what catches two different
   source items that happen to carry identical text.
4. **token-sequence separation** — no calibration item's token ids are byte-equal
   to another's, which would silently reweight a sub-type.
5. **validation-slice separation** — no calibration session appears in the pack's
   canonical 16-block validation slice. That slice lives in the tail beyond the
   largest rung, which is exactly where the calibration sessions come from, so
   this is the one collision the rung filter structurally cannot catch. The first
   build of the set contained one, and it would have calibrated the depth map on
   the teacher-native held-out CE the step-0 comparison reports.

    python3 scripts/data/check_e8_calibration_leakage.py \\
        --calibration artifacts/stage1/e8_calibration_v1 \\
        --pack artifacts/stage3/ladder_uniform_probe \\
        --sessions artifacts/stage3/corpus_v2/sessions.jsonl \\
        --reserved data/eval_behavior_v0/prompts.jsonl \\
        --reserved 'artifacts/eval/battery_v2/*.jsonl' \\
        --out artifacts/stage1/e8_calibration_v1/leakage.json

Exit codes: 0 clean; 6 leakage, or an input that could not be read.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.extra_stream import content_sha256  # noqa: E402
from aadistill.data.ladder import select_val_blocks  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts/data"))
from check_stream_disjointness import row_text  # noqa: E402


def prompt_text(session: dict) -> str:
    """Everything the model is shown, excluding the assistant's own output.

    Hashing the prompt rather than the whole session is deliberate: the battery
    reuses the *questions*, not the teacher's answers, so a shared question is the
    leak and a shared answer would only be a curiosity.
    """
    parts = []
    for m in session.get("messages", []):
        if m.get("role") == "assistant":
            continue
        parts.append(str(m.get("content", "")))
    return "\n".join(parts)


def consumed(audit_path: Path, n_blocks: int):
    source_ids, candidates, session_ids = set(), set(), set()
    for i, line in enumerate(audit_path.open()):
        if i >= n_blocks:
            break
        for s in json.loads(line)["sessions"]:
            session_ids.add(s["session_id"])
            source_ids.add(str(s["session_id"]).split("#")[0])
            if s.get("candidate_sha256"):
                candidates.add(s["candidate_sha256"])
    return session_ids, source_ids, candidates


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", default="artifacts/stage1/e8_calibration_v1")
    ap.add_argument("--pack", default="artifacts/stage3/ladder_uniform_probe")
    ap.add_argument("--sessions", default="artifacts/stage3/corpus_v2/sessions.jsonl")
    ap.add_argument("--val-blocks", type=int, default=16)
    ap.add_argument("--reserved", action="append", default=[])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else REPO_ROOT / p

    calib = resolve(args.calibration)
    pack = resolve(args.pack)
    manifest = json.loads((calib / "manifest.json").read_text())
    items = [json.loads(l) for l in (calib / "items.jsonl").open() if l.strip()]
    excluded_rung = manifest["leakage_control"]["excluded_through_rung"]

    ladder = json.loads((pack / "ladder.json").read_text())
    rungs = {r["target_supervised_tokens"]: r for r in ladder["rungs"]}
    n_blocks = rungs[excluded_rung]["n_blocks"]
    if n_blocks != manifest["leakage_control"]["excluded_blocks"]:
        raise SystemExit(
            f"the pack says rung {excluded_rung} is {n_blocks} blocks, the "
            f"calibration manifest says "
            f"{manifest['leakage_control']['excluded_blocks']}; different pack")
    con_sessions, con_sources, con_candidates = consumed(pack / "audit.jsonl", n_blocks)

    audit = [json.loads(l) for l in (pack / "audit.jsonl").open() if l.strip()]
    max_rung_blocks = max(int(r["n_blocks"]) for r in ladder["rungs"]
                          if r.get("reachable", False))
    val_idx = select_val_blocks(audit, max_rung_blocks, args.val_blocks)
    val_sessions = {s["session_id"] for i in val_idx for s in audit[i]["sessions"]}
    val_sources = {sid.split("#")[0] for sid in val_sessions}

    # Prompt hashes of everything the calibration set must not have seen.
    reserved_hashes: dict[str, set[str]] = {}
    for pattern in args.reserved:
        matches = sorted(glob.glob(str(resolve(pattern))))
        if not matches:
            raise SystemExit(f"--reserved {pattern!r} matched no file; refusing to "
                             "report separation from something that was not read")
        for m in matches:
            p = Path(m)
            hashes = set()
            for line in p.read_text(errors="replace").splitlines():
                if line.strip():
                    row = json.loads(line)
                    hashes.add(content_sha256(row_text(row)))
            reserved_hashes[f"reserved:{p.name}"] = hashes

    consumed_prompt_hashes = set()
    calib_sessions = {i["session_id"] for i in items if i.get("session_id")}
    for line in resolve(args.sessions).open():
        d = json.loads(line)
        if d["id"] in con_sessions:
            consumed_prompt_hashes.add(content_sha256(prompt_text(d)))
    reserved_hashes[f"consumed_prompts_through_{excluded_rung}"] = consumed_prompt_hashes

    calib_prompt_hashes: dict[str, str] = {}
    for line in resolve(args.sessions).open():
        d = json.loads(line)
        if d["id"] in calib_sessions:
            calib_prompt_hashes[d["id"]] = content_sha256(prompt_text(d))

    findings: list[dict] = []
    shared_sources = sorted({i["source_id"] for i in items
                             if i.get("source_id") in con_sources})
    if shared_sources:
        findings.append({"check": "source_id", "n": len(shared_sources),
                         "examples": shared_sources[:5]})
    shared_sessions = sorted(calib_sessions & con_sessions)
    if shared_sessions:
        findings.append({"check": "session_id", "n": len(shared_sessions),
                         "examples": shared_sessions[:5]})
    shared_val = sorted(calib_sessions & val_sessions)
    if shared_val:
        findings.append({"check": "ladder_validation_slice", "n": len(shared_val),
                         "examples": shared_val[:5]})
    shared_val_src = sorted({i["source_id"] for i in items
                             if i.get("source_id") in val_sources})
    if shared_val_src:
        findings.append({"check": "ladder_validation_source_id",
                         "n": len(shared_val_src), "examples": shared_val_src[:5]})
    shared_candidates = sorted({i["candidate_sha256"] for i in items
                                if i.get("candidate_sha256") in con_candidates})
    if shared_candidates:
        findings.append({"check": "candidate_sha256", "n": len(shared_candidates),
                         "examples": shared_candidates[:5]})
    for label, hashes in reserved_hashes.items():
        hit = sorted(sid for sid, h in calib_prompt_hashes.items() if h in hashes)
        if hit:
            findings.append({"check": f"prompt_content_vs_{label}", "n": len(hit),
                             "examples": hit[:5]})

    seen: dict[str, str] = {}
    duplicates = []
    for i in items:
        h = hashlib.sha256(",".join(map(str, i["ids"])).encode()).hexdigest()
        if h in seen:
            duplicates.append([seen[h], i["item_id"]])
        seen[h] = i["item_id"]
    if duplicates:
        findings.append({"check": "duplicate_token_sequences",
                         "n": len(duplicates), "examples": duplicates[:5]})

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "calibration": {
            "path": args.calibration,
            "manifest_sha256": manifest.get("manifest_sha256"),
            "content_sha256": manifest.get("content_sha256"),
            "items": len(items),
            "templated_sessions": len(calib_sessions),
            "items_sha256": sha256_file(calib / "items.jsonl"),
        },
        "excluded": {
            "rung": excluded_rung, "blocks": n_blocks,
            "sessions": len(con_sessions), "source_ids": len(con_sources),
            "candidate_hashes": len(con_candidates),
            "why": "the frozen 150-prompt behaviour battery is sampled from the "
                   "0.86M rung, a prefix of the excluded range",
        },
        "compared_against": {k: len(v) for k, v in reserved_hashes.items()},
        "ladder_validation_slice": {
            "block_indices": val_idx, "sessions": len(val_sessions),
            "why": "the pack's val slice shares the tail the calibration set is "
                   "drawn from; the rung filter cannot see it",
        },
        "checks": ["source_id", "session_id", "candidate_sha256",
                   "prompt_content", "duplicate_token_sequences",
                   "ladder_validation_slice"],
        "findings": findings,
        "clean": not findings,
    }
    if args.out:
        out = resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"-> {out}")
    print(json.dumps({"clean": report["clean"], "findings": findings,
                      "compared_against": report["compared_against"]}, indent=2))
    return 0 if report["clean"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
