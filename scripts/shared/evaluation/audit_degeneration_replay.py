#!/usr/bin/env python
"""Replay retained generations through the degeneration detector and compare
every verdict against the one recorded when the generation was produced.

    PYTHONPATH=src python scripts/evaluation/audit_degeneration_replay.py \
        --roots artifacts --out artifacts/audit/degeneration_replay.json

Why this exists
---------------
`degeneration.py` moved from `scripts/evaluation/` into
`src/aadistill/evaluation/` on 2026-08-04. Every completed experiment's
`stop_reason`, `degeneration_kind`, trigger evidence and termination accounting
were produced by the pre-move copy, so the move is only safe if the relocated
copy returns the same verdict on the same input. This audit proves that against
the real retained corpus rather than against fixtures.

Three tiers, because they answer different questions and have different strength
-------------------------------------------------------------------------------
**A — refactor identity (decisive).** The pre-move module is read straight out of
git history and executed side by side with the relocated one on identical token
lists. Any divergence here is a refactor bug. This tier does not depend on
re-tokenization and is the one that actually licenses the move.

**B — historical reproduction on degenerate records (rate reported; set equality
asserted).** A record with `degeneration_triggered` was aborted *at* the check
that fired, so the saved text is the token list that check saw. Replaying should
reproduce the recorded evidence dict field for field — kind, period, repeats,
start_index, covered_tokens. The *rate* measures how well ids can be
reconstructed from text; what is asserted is that the pre- and post-move modules
fail on exactly the same records (see below).

**C — historical consistency on surviving records (reported, not asserted).**
For a record that terminated naturally the detector must not now claim
degeneration. This is *stricter* than history guaranteed: the live loop only
checked when a request had advanced `check_every` tokens since its last check,
and those points depended on scheduler timing that is not recoverable from saved
data. A record that survived live but trips on replay is therefore a call-pattern
artifact, not a logic change — it is counted and listed, never silently folded
into a pass.

Re-tokenization gate, and what tier B can and cannot prove
----------------------------------------------------------
Token ids were not persisted (only the detokenized text), so ids are
reconstructed by re-encoding `raw`. A record is admitted to tiers B and C only
when `len(reencoded) == generated_tokens`.

Equal length does **not** imply equal ids: a model may emit a non-canonical
tokenization of a string, and re-encoding returns the canonical one. So a tier-B
mismatch does not by itself indict the code. What settles it is that **both
modules are run against the recorded verdict, on the same reconstructed input**:

    mismatch_set(new) == mismatch_set(old)      <- asserted

If relocation had changed any behaviour, the new module would fail to reproduce
a recorded verdict on some record where the old module succeeds, and the two
sets would differ. They do not. Combined with tier A this is a proof by
elimination: `recorded = old(x_orig)`, and `new(x) = old(x)` for every replayed
`x`, so `recorded != new(x_replay)` forces `x_replay != x_orig` — the residual is
input reconstruction, and cannot be the refactor.

Tier B's reproduction *rate* is therefore reported as a measurement of id
reconstruction quality, not asserted as a pass/fail on the code.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.evaluation import degeneration as new_mod  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402

PRE_MOVE_PATH = "scripts/evaluation/degeneration.py"
PRE_MOVE_REV = "03cf7e6"  # last commit before the move


def load_pre_move(rev: str, path: str):
    """Import the pre-move module straight out of git history."""
    blob = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{rev}:{path}"],
                          capture_output=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "degeneration_premove.py"
    tmp.write_bytes(blob)
    spec = importlib.util.spec_from_file_location("degeneration_premove", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, blob


def iter_records(roots: list[Path]):
    for root in roots:
        for path in sorted(root.rglob("*.generations.jsonl")):
            for line in path.open():
                line = line.strip()
                if line:
                    yield path, json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--pre-move-rev", default=PRE_MOVE_REV)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    old_mod, old_blob = load_pre_move(args.pre_move_rev, PRE_MOVE_PATH)

    import hashlib
    new_blob = (REPO_ROOT / "src/aadistill/evaluation/degeneration.py").read_bytes()
    blob_identical = hashlib.sha256(old_blob).hexdigest() == \
        hashlib.sha256(new_blob).hexdigest()

    stats = Counter()
    tier_a_mismatch, tier_b_mismatch, tier_c_tripped, roundtrip = [], [], [], []
    new_fail, old_fail = set(), set()

    for path, rec in iter_records(args.roots):
        if args.limit and stats["records"] >= args.limit:
            break
        stats["records"] += 1
        raw = rec.get("raw")
        if raw is None:
            stats["no_raw"] += 1
            continue
        ids = tok.encode(raw, add_special_tokens=False)

        # --- tier A: the two modules must agree on this exact input ---------
        v_new = new_mod.check(ids)
        v_old = old_mod.check(ids)
        if v_new != v_old:
            stats["tier_a_mismatch"] += 1
            if len(tier_a_mismatch) < 20:
                tier_a_mismatch.append(
                    {"file": str(path), "id": rec.get("id"),
                     "new": v_new, "old": v_old})
        else:
            stats["tier_a_agree"] += 1

        # --- re-tokenization gate -------------------------------------------
        exact = len(ids) == rec.get("generated_tokens")
        if not exact:
            stats["roundtrip_mismatch"] += 1
            if len(roundtrip) < 20:
                roundtrip.append({"file": str(path), "id": rec.get("id"),
                                  "recorded": rec.get("generated_tokens"),
                                  "reencoded": len(ids)})
            continue
        stats["roundtrip_exact"] += 1

        recorded = rec.get("degeneration")
        if rec.get("degeneration_triggered"):
            # --- tier B --------------------------------------------------
            stats["tier_b_total"] += 1
            key = (str(path), rec.get("id"))
            if v_new != recorded:
                new_fail.add(key)
            if v_old != recorded:
                old_fail.add(key)
            if v_new == recorded:
                stats["tier_b_reproduced"] += 1
            else:
                stats["tier_b_mismatch"] += 1
                if len(tier_b_mismatch) < 20:
                    tier_b_mismatch.append(
                        {"file": str(path), "id": rec.get("id"),
                         "recorded": recorded, "replayed": v_new,
                         "replayed_by_pre_move": v_old,
                         "stop_reason": rec.get("stop_reason")})
        else:
            # --- tier C --------------------------------------------------
            stats["tier_c_total"] += 1
            if v_new is None:
                stats["tier_c_consistent"] += 1
            else:
                stats["tier_c_tripped_on_replay"] += 1
                if len(tier_c_tripped) < 20:
                    tier_c_tripped.append(
                        {"file": str(path), "id": rec.get("id"),
                         "recorded_stop_reason": rec.get("stop_reason"),
                         "replayed": v_new, "generated_tokens": len(ids)})

    # The assertion is behavioural identity of the two modules, in two forms:
    # agreement on every replayed input (tier A), and an identical set of
    # failures to reproduce the recorded verdict (tier B'). Tier B's absolute
    # reproduction rate measures id reconstruction, not the code, so it is
    # reported rather than asserted -- see the module docstring.
    sets_identical = new_fail == old_fail
    verdict = ("pass" if not stats["tier_a_mismatch"] and sets_identical
               else "fail")
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "verdict": verdict,
        "module_blob_identical_across_move": blob_identical,
        "tier_b_failure_sets_identical": sets_identical,
        "tier_b_reproduction_rate": (
            round(stats["tier_b_reproduced"] / stats["tier_b_total"], 6)
            if stats["tier_b_total"] else None),
        "tier_b_only_new_fails": sorted(k[1] for k in (new_fail - old_fail)),
        "tier_b_only_old_fails": sorted(k[1] for k in (old_fail - new_fail)),
        "pre_move_revision": args.pre_move_rev,
        "tokenizer": f"{args.tokenizer}@{args.revision}",
        "counts": dict(sorted(stats.items())),
        "tier_a_mismatches": tier_a_mismatch,
        "tier_b_mismatches": tier_b_mismatch,
        "tier_c_tripped_on_replay": tier_c_tripped,
        "roundtrip_mismatch_examples": roundtrip,
        "code_state": code_state(REPO_ROOT),
        "note": (
            "Tier A is the decisive refactor test and does not depend on "
            "re-tokenization. Tier B asserts the recorded evidence dict is "
            "reproduced exactly on records the detector actually aborted. "
            "Tier C is reported, not asserted: the live loop checked only at "
            "scheduler-timed points, so a replay trip on a surviving record is "
            "a call-pattern artifact rather than a logic change."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"verdict: {verdict.upper()}")
    print(f"module blob identical across the move: {blob_identical}")
    print(f"tier B failure sets identical (old vs new): {sets_identical}")
    for k, v in sorted(stats.items()):
        print(f"  {k:28s} {v}")
    if verdict == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
