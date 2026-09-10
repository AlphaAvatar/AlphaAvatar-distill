"""The historical probe-reuse record as it reads under the probes' OWN contract.

Phase A's probes were scored under `recovery_search_scoring@v2`. The scorer has
since relocated twice and the contract legitimately moved to v3, so every one of
the eleven probes now fails `scoring_contract_matches_live` — and only that
check. `logs/autoinit_historical_reuse_position.json` derives the same thing and
records it as conclusion 4: **live reuse under v3 is REFUSED**, deliberately and
without relaxation, because admitting a superseded contract is a maintainer
decision rather than a migration one.

That refusal is correct, and it is also not what the Phase-B driver tests are
about. Those tests ask what the driver does with a record it *may* cite: which
probes it imports, which admitted-but-excluded leaves it still skips, how it
refuses a thin or permissive record. Pointing them at a record the current
contract refuses would test the refusal eleven times over and leave the import
logic uncovered.

So this derives the record the probes satisfy under their own contract —
by dropping exactly one check, from real data, with the drop asserted — and the
live record's refusal is asserted separately and explicitly by
`test_the_live_record_is_currently_refused` in each consumer.

Deriving rather than fixture-ing matters: a hand-written record would let the
import tests keep passing after the real one changed shape, which is the failure
mode that left this record stale through the v2 -> v3 migration in the first
place.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIVE_RECORD = REPO / "logs/autoinit_historical_probe_reuse.json"

#: The one check the historical contract cannot satisfy, by construction.
LIVE_CONTRACT_CHECK = "scoring_contract_matches_live"


class NotOnlyTheContractCheck(AssertionError):
    """A probe failed something other than the live-contract identity.

    Then this derivation is wrong to apply: it would be admitting a probe with a
    real defect, not one with a superseded scorer digest.
    """


def under_historical_contract(live: dict | None = None) -> dict:
    """The live record with `scoring_contract_matches_live` set aside.

    Raises if any probe fails anything else, so this can never quietly widen
    into "ignore whatever is failing today".
    """
    doc = json.loads(json.dumps(live if live is not None
                                else json.loads(LIVE_RECORD.read_text())))
    for p in doc["probes"]:
        failed = [f for f in p["failed"] if f != LIVE_CONTRACT_CHECK]
        if failed:
            raise NotOnlyTheContractCheck(
                f"{p['probe_id']} also fails {failed}; the historical-contract "
                "derivation only sets aside the live-contract identity")
        p["failed"] = []
        p["reusable"] = True
        p["checks"] = {**p["checks"], LIVE_CONTRACT_CHECK: True}

    reusable = [f"{p['candidate']}/{p['seed']}" for p in doc["probes"]]
    admitted = [f"{p['candidate']}/{p['seed']}" for p in doc["probes"]
                if p["admitted_by_the_procedure"]]
    doc.update(
        reuse_verified=True,
        reusable_probes=reusable,
        admitted_reusable_probes=admitted,
        verifiable_but_not_admitted=[r for r in reusable if r not in admitted],
        failures=[],
    )
    return doc


def write_historical_contract_record(tmp_path: Path) -> Path:
    path = tmp_path / "historical_contract_reuse.json"
    path.write_text(json.dumps(under_historical_contract()))
    return path
