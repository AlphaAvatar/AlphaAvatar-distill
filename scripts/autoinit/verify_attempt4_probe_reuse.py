#!/usr/bin/env python3
"""Can a later session cite Attempt 4's one purchased `sb` probe?

    PYTHONPATH=src python scripts/autoinit/verify_attempt4_probe_reuse.py \
        --out logs/autoinit_attempt4_probe_reuse.json

Attempt 4 ran to `ALL_DONE` and paid for exactly one genuinely new observation:
`autoinit.v1.phase_a.rung2.fe9683e6a9c7.sb`, the single missing rung-2 probe.
Its *decision* was wrong — the inherited pooling let a historical `sc` leak into
the rung-2 comparison — but the probe itself is a real, finished measurement of
the right checkpoint on the right seed, and nothing about the pooling defect
touches it.

**The probe survives the decision.** Re-buying it would cost another ~72 minutes
of L40S for evidence that already exists, so this applies the **same** strict
reconstruction the historical and Attempt-5 citations must pass. "It exists" is
not the standard; a probe belongs to a checkpoint only if the bytes still say so.

Per probe:

1. the record is `complete`;
2. its seed is the frozen `SEED_SB` — not merely *a* seed, and not `SEED_SA`;
3. its `student_artifact_digest` **re-derives from the retained checkpoint
   bytes**, proving it belongs to the initialization it would be cited for;
4. its battery is the frozen `recovery_search_v2`, by content and manifest hash;
5. its scoring contract equals the **live** `recovery_scoring_contract()`;
6. its evaluation-protocol hash equals the protocol Attempt 4 attested;
7. it carries the counts `POOLED_COUNTS_V2` requires, so it can be pooled rather
   than merely stored;
8. no conflicting duplicate observation exists for the same `(state_id, seed)`.

The open leg is the same one the other two reuse records carry: a later session's
runtime must be comparable under `generation_runtime_comparability@v2`, which is
a Stage-0 precondition checked there and fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.arch import get_adapter  # noqa: E402
from aadistill.autoinit.artifact import identify_checkpoint  # noqa: E402
from aadistill.autoinit.recovery import (  # noqa: E402
    POOLED_COUNTS_V2, SEED_SB, recovery_scoring_contract,
)
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

ATTEMPT = REPO_ROOT / "logs/autoinit_continuation_b_attempt4"
PROBES = ATTEMPT / "probes"
ATTESTED = ATTEMPT / "attested_evaluation_protocol.json"

#: The one probe Attempt 4 actually PAID for. Every other record in that
#: directory is an imported citation already covered by
#: `verify_historical_probe_reuse.py` or `verify_attempt5_probe_reuse.py`;
#: re-verifying them here would double-count the same evidence under a third
#: name.
FRESH = ("fe9683e6a9c7",)

#: Where the candidate's checkpoint is retained. The digest is re-derived from
#: these bytes, not trusted.
_STORE = REPO_ROOT.parent / "aad-artifacts/autoinit/phase_a"

BATTERY_CONTENT = "a1b22778b00d95b6aba358c14a5af5b559fd807bb371c92131eacca59479f323"
BATTERY_MANIFEST = "58ae5c6dcbe32eb28c343a66830d7224a14537362deeeff2ce8219d0a31679d6"
PROBE_RE = re.compile(r"^autoinit\.v1\.phase_a\.rung(\d)\.([^.]+)\.(s[abc])\.json$")


def probes_dir_digest(root: Path = PROBES) -> str:
    return hashlib.sha256("".join(
        f"{p.name}:{sha256_file(p)}\n" for p in sorted(root.iterdir())
        if PROBE_RE.match(p.name)).encode()).hexdigest()


def checkpoint_dir(state_id_12: str) -> Path | None:
    matches = sorted(p for p in _STORE.glob(f"{state_id_12}*") if p.is_dir())
    return matches[0] if matches else None


def checkpoint_digest(directory: Path) -> tuple[str | None, str | None]:
    from transformers import AutoConfig

    if directory is None or not directory.is_dir():
        return None, f"{directory} is not a directory"
    try:
        adapter = get_adapter("qwen3")
        spec = adapter.spec_from_config(AutoConfig.from_pretrained(str(directory)))
        ident = identify_checkpoint(directory, adapter=adapter, spec=spec,
                                    num_parameters=adapter.param_count(spec))
        return ident.artifact_digest, None
    except Exception as exc:                     # noqa: BLE001 - report, never crash
        return None, f"{type(exc).__name__}: {exc}"


def verify(root: Path = PROBES) -> dict:
    live_contract = recovery_scoring_contract()
    attested = json.loads(ATTESTED.read_text()) if ATTESTED.is_file() else {}
    attested_protocol = attested.get("evaluation_protocol_hash")

    seen_pairs: dict[tuple[str, int], list[str]] = {}
    for path in sorted(root.iterdir()):
        m = PROBE_RE.match(path.name)
        if not m:
            continue
        record = json.loads(path.read_text())
        key = (m.group(2), int(record.get("seed", -1)))
        seen_pairs.setdefault(key, []).append(path.name)

    results = []
    for candidate in FRESH:
        path = root / f"autoinit.v1.phase_a.rung2.{candidate}.sb.json"
        if not path.is_file():
            results.append({"candidate": candidate, "reusable": False,
                            "failed": ["record_present"], "checkpoint_error": None,
                            "note": f"{path.name} is not in the retained evidence"})
            continue
        record = json.loads(path.read_text())
        result = record.get("result") or {}
        battery = result.get("battery") or {}
        contract = result.get("scoring_contract") or {}
        directory = checkpoint_dir(candidate)
        recomputed, ckpt_error = checkpoint_digest(directory)
        recorded = record.get("student_artifact_digest")
        pair = (candidate, int(record.get("seed", -1)))

        checks = {
            "complete": record.get("complete") is True,
            "seed_is_the_frozen_sb": record.get("seed") == SEED_SB,
            "rung_is_2": record.get("rung") == 2,
            "artifact_digest_re_derives_from_bytes":
                recomputed is not None and recomputed == recorded,
            "battery_is_recovery_search_v2":
                battery.get("content_sha256") == BATTERY_CONTENT
                and battery.get("manifest_sha256") == BATTERY_MANIFEST,
            "scoring_contract_matches_live":
                contract.get("digest") == live_contract["digest"],
            "protocol_hash_matches_attested":
                attested_protocol is not None
                and record.get("evaluation_protocol_hash") == attested_protocol,
            "carries_the_pooled_counts":
                all(k in result for k in POOLED_COUNTS_V2.required_counts),
            "no_conflicting_duplicate_observation": len(seen_pairs.get(pair, [])) == 1,
        }
        results.append({
            "probe_id": record.get("probe_id"), "candidate": candidate,
            "seed": record.get("seed"), "rung": record.get("rung"),
            "checkpoint_dir": str(directory) if directory else None,
            "checks": checks,
            "reusable": all(checks.values()),
            "failed": sorted(k for k, v in checks.items() if not v),
            "checkpoint_error": ckpt_error,
            "recomputed_artifact_digest": recomputed,
            "recorded_artifact_digest": recorded,
            "correct": result.get("correct"),
            "n_scorable": result.get("n_scorable"),
            "correct_overall": result.get("correct_overall"),
            "duplicate_records_for_this_state_and_seed": seen_pairs.get(pair, []),
        })

    failures = [r for r in results if not r["reusable"]]
    return {
        "schema": "aadistill.autoinit.attempt4_probe_reuse/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "_contract": (
            "Strict reconstruction of the ONE rung-2 `sb` probe Attempt 4 PAID "
            "for, to the same standard as the historical and Attempt-5 "
            "citations. Attempt 4's DECISION was withdrawn — the inherited "
            "pooling let a historical `sc` leak into the rung-2 comparison — but "
            "the probe is a valid finished measurement and must never be "
            "repurchased. Not an authorization."),
        "why_the_decision_does_not_taint_the_probe": (
            "the defect was in which records the selector pooled, not in how any "
            "single probe was trained, evaluated or scored. This record checks "
            "the probe against the checkpoint bytes, the frozen battery, the live "
            "scoring contract and the attested protocol — none of which the "
            "pooling defect touches."),
        "source": "logs/autoinit_continuation_b_attempt4/probes",
        "probes_dir_digest": probes_dir_digest(root),
        "attested_protocol_hash": attested_protocol,
        "live_scoring_contract_digest": live_contract["digest"],
        "seed_sb": SEED_SB,
        "n_probes": len(results),
        "reuse_verified": not failures and len(results) == len(FRESH),
        "reusable_probes": sorted(f"{r['candidate']}/sb" for r in results
                                  if r["reusable"]),
        "failures": [{"candidate": r["candidate"], "failed": r["failed"],
                      "checkpoint_error": r.get("checkpoint_error")}
                     for r in failures],
        "probes": results,
        "open_precondition": {
            "what": ("a later session's runtime must be comparable to Attempt "
                     "4's under generation_runtime_comparability@v2"),
            "why_not_checkable_now": "that session's runtime does not exist yet",
            "checked_where": "Stage 0 of the eventual session, fail-closed",
            "if_it_fails": ("ALL cited behavioural evidence is lost at once, "
                            "historical, Attempt-5 and Attempt-4 alike"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_attempt4_probe_reuse.json")
    args = ap.parse_args()
    result = verify()
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / args.out
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"probes checked   {result['n_probes']}")
    print(f"reuse_verified   {result['reuse_verified']}")
    print(f"reusable         {', '.join(result['reusable_probes']) or '(none)'}")
    for f in result["failures"]:
        print(f"  FAILED {f['candidate']}: {f['failed']} {f['checkpoint_error'] or ''}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
