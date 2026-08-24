#!/usr/bin/env python3
"""Can Phase B reuse Phase A's probes as evidence? Zero cost; launches nothing.

    PYTHONPATH=src python scripts/autoinit/verify_historical_probe_reuse.py \
        --out logs/autoinit_historical_probe_reuse.json

Phase B's terminal procedure reuses historical sa/sb/sc results "only after
strict reconstruction proves the same materialized recovery protocol and seed".
`price_phase_b.py` previously treated a probe as reusable because its **file
existed**, which is not that check — it is the absence of one.

This runs the check. Per probe:

1. the record is `complete`;
2. its seed is the frozen seed for its rung — not merely *a* seed;
3. its `student_artifact_digest` **re-derives from the retained checkpoint
   bytes**. This is the load-bearing one: it proves the probe belongs to the
   checkpoint Phase B would reuse it for, rather than to something that once had
   the same name;
4. its battery is the frozen `recovery_search_v2`, by content and manifest hash;
5. its scoring contract equals the **live** `recovery_scoring_contract()`, so a
   change to the scoring source since Phase A invalidates reuse rather than
   silently re-interpreting old numbers;
6. its evaluation protocol hash equals the attested Phase-A protocol's.

**One leg cannot be closed at `$0` and is reported, not assumed.** Reuse also
requires Phase B's own runtime to be *comparable* to Phase A's under
`generation_runtime_comparability@v2`, and Phase B's runtime does not exist yet.
That is a Stage-0 precondition of the eventual run. If it fails there, **every**
historical probe is lost at once — which is why the pricing carries a separate
no-reuse scenario rather than folding it into a range.
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
    SEED_SA, SEED_SB, SEED_SC, recovery_scoring_contract,
)
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

ATTEMPT = REPO_ROOT / "logs/autoinit_recovery_continuation_attempt7"
PROBES = ATTEMPT / "probes"
ATTESTED = ATTEMPT / "attested_evaluation_protocol.json"

#: Where each candidate's retained checkpoint actually lives. **All five searched
#: leaves are declared, not only the two the procedure admits** — otherwise the
#: three unadmitted ones would be reported as identity failures when what is
#: actually true is that they are verifiable and simply not in the candidate set.
#: "We did not look" and "it does not check out" must not print the same.
_STORE = "/home/ecs-user/aad-artifacts/autoinit/phase_a"
CHECKPOINTS = {
    "cca699c93f34": f"{_STORE}/cca699c93f34dad7e94a5d13a25b2bc2",
    "85bde4ded2c3": f"{_STORE}/85bde4ded2c31953f802e39cf2252c87",
    "158b96cf651f": f"{_STORE}/158b96cf651fd8ba8f8ceaefefde2067",
    "281a02c3ac18": f"{_STORE}/281a02c3ac18419b70e896296dac0d03",
    "4e429f7ed722": f"{_STORE}/4e429f7ed722b180dd662c779895693f",
    "control-qwen": str(REPO_ROOT / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"),
}

#: The candidate set the 2026-08-25 terminal procedure admits as priors. Reuse
#: eligibility (identity checks out) and admission (the procedure wants it) are
#: different questions and are reported separately.
ADMITTED = ("cca699c93f34", "85bde4ded2c3", "control-qwen")

#: The frozen battery identity Phase A scored on.
BATTERY_CONTENT = "a1b22778b00d95b6aba358c14a5af5b559fd807bb371c92131eacca59479f323"
BATTERY_MANIFEST = "58ae5c6dcbe32eb28c343a66830d7224a14537362deeeff2ce8219d0a31679d6"

SEEDS = {"sa": SEED_SA, "sb": SEED_SB, "sc": SEED_SC}
PROBE_RE = re.compile(r"^autoinit\.v1\.phase_a\.rung(\d)\.([^.]+)\.(s[abc])\.json$")


def probes_dir_digest(root: Path = PROBES) -> str:
    """Identity of the evidence set, so a consumer can detect drift."""
    return hashlib.sha256("".join(
        f"{p.name}:{sha256_file(p)}\n" for p in sorted(root.iterdir())
        if PROBE_RE.match(p.name)).encode()).hexdigest()


def checkpoint_digest(path: str) -> tuple[str | None, str | None]:
    from transformers import AutoConfig

    d = Path(path)
    if not d.is_dir():
        return None, f"{d} is not a directory"
    try:
        adapter = get_adapter("qwen3")
        spec = adapter.spec_from_config(AutoConfig.from_pretrained(str(d)))
        ident = identify_checkpoint(d, adapter=adapter, spec=spec,
                                    num_parameters=adapter.param_count(spec))
        return ident.artifact_digest, None
    except Exception as exc:                     # noqa: BLE001 - report, never crash
        return None, f"{type(exc).__name__}: {exc}"


def verify(root: Path = PROBES) -> dict:
    live_contract = recovery_scoring_contract()
    attested = json.loads(ATTESTED.read_text()) if ATTESTED.is_file() else {}
    attested_protocol = attested.get("evaluation_protocol_hash")

    digests: dict[str, tuple[str | None, str | None]] = {}
    results = []
    for path in sorted(root.iterdir()):
        m = PROBE_RE.match(path.name)
        if not m:
            continue
        _, candidate, seed_name = m.groups()
        rec = json.loads(path.read_text())
        battery = rec.get("result", {}).get("battery", {})
        contract = rec.get("result", {}).get("scoring_contract", {})

        if candidate not in digests:
            digests[candidate] = checkpoint_digest(CHECKPOINTS[candidate]) \
                if candidate in CHECKPOINTS else (None, "no retained checkpoint declared")
        recomputed, ckpt_error = digests[candidate]

        checks = {
            "complete": rec.get("complete") is True,
            "seed_is_the_frozen_one": rec.get("seed") == SEEDS[seed_name],
            "artifact_digest_re_derives_from_bytes":
                recomputed is not None and recomputed == rec.get("student_artifact_digest"),
            "battery_is_recovery_search_v2":
                battery.get("content_sha256") == BATTERY_CONTENT
                and battery.get("manifest_sha256") == BATTERY_MANIFEST,
            "scoring_contract_matches_live":
                contract.get("digest") == live_contract["digest"],
            "protocol_hash_matches_attested":
                attested_protocol is not None
                and rec.get("evaluation_protocol_hash") == attested_protocol,
        }
        results.append({
            "probe_id": rec.get("probe_id"), "candidate": candidate, "seed": seed_name,
            "checks": checks,
            "admitted_by_the_procedure": candidate in ADMITTED,
            "reusable": all(checks.values()),
            "failed": sorted(k for k, v in checks.items() if not v),
            "checkpoint_error": ckpt_error,
            "recomputed_artifact_digest": recomputed,
            "recorded_artifact_digest": rec.get("student_artifact_digest"),
        })

    reusable = sorted({r["candidate"] + "/" + r["seed"] for r in results if r["reusable"]})
    admitted = sorted({r["candidate"] + "/" + r["seed"] for r in results
                       if r["reusable"] and r["admitted_by_the_procedure"]})
    verifiable_but_unadmitted = sorted({
        r["candidate"] + "/" + r["seed"] for r in results
        if r["reusable"] and not r["admitted_by_the_procedure"]})
    failures = [r for r in results if not r["reusable"]]
    return {
        "schema": "aadistill.autoinit.historical_probe_reuse/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "_contract": ("The strict reconstruction check Phase B's reuse is "
                      "conditional on. Consumed by scripts/autoinit/price_phase_b.py, "
                      "which fails closed if this record is missing or its "
                      "probes_dir_digest no longer matches."),
        "probes_dir_digest": probes_dir_digest(root),
        "live_scoring_contract_digest": live_contract["digest"],
        "attested_protocol_hash": attested_protocol,
        "seeds": {k: v for k, v in SEEDS.items()},
        "n_probes": len(results),
        "reuse_verified": not failures and bool(results),
        "reusable_probes": reusable,
        "admitted_reusable_probes": admitted,
        "verifiable_but_not_admitted": verifiable_but_unadmitted,
        "failures": [{"probe_id": r["probe_id"], "failed": r["failed"],
                      "checkpoint_error": r["checkpoint_error"]} for r in failures],
        "probes": results,
        "open_precondition": {
            "what": ("Phase B's own runtime must be comparable to Phase A's under "
                     "generation_runtime_comparability@v2"),
            "why_not_checkable_now": "Phase B's runtime does not exist yet",
            "checked_where": "Stage 0 of the eventual Phase-B session, fail-closed",
            "if_it_fails": ("ALL historical reuse is lost at once, not part of it; "
                            "priced as a separate no-reuse scenario"),
            "phase_a_comparability": attested.get("comparability"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_historical_probe_reuse.json")
    args = ap.parse_args()
    result = verify()
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / args.out
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"probes checked      {result['n_probes']}")
    print(f"reuse_verified      {result['reuse_verified']}")
    print(f"admitted+reusable   {', '.join(result['admitted_reusable_probes'])}")
    print(f"verified, unadmitted {', '.join(result['verifiable_but_not_admitted']) or '-'}")
    for f in result["failures"]:
        print(f"  FAILED {f['probe_id']}: {f['failed']} {f['checkpoint_error'] or ''}")
    print(f"open precondition   {result['open_precondition']['what']}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    sys.exit(0 if result["reuse_verified"] else 1)


if __name__ == "__main__":
    main()
