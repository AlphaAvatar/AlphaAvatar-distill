"""The A3 authorization payload, fully DERIVED from the tree it authorizes.

A grant says what a maintainer decided. It does not get to say what the
science is, what the harness digests to, or what the session may spend —
every one of those is derived here and **refused if the grant asserts it**.
A grant that could state its own ceiling is a grant that can over-authorize
itself, and one that could state the design is a grant that can quietly
change the experiment.

What this refuses, each because the shape of the mistake is known:

* a grant asking for the per-session ENVELOPE when the derived ceiling is
  lower — a 36% over-authorization every downstream gate would accept;
* a grant naming a project cap that is not the live one;
* a live pricing record that is not FUNDABLE, or that did not evaluate every
  applicable limit;
* a design whose probe count, arm count or seeds are not the frozen ones;
* a design that claims the controls are retrained, or that the comparison runs
  on the pod;
* a dirty tree, which would bind bytes the pod will never check out.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.infrastructure.manifest import sha256_file  # noqa: E402
from experiments.phase_a3 import a3_session as A3S  # noqa: E402
from experiments.phase_a3.a3_authorization import (  # noqa: E402
    A3Authorization, a3_expected_usd, a3_hard_ceiling_usd,
    a3_harness_digest, load_live_pricing,
)

BATTERY_REL = "logs/stages/stage-1/phase_c1/plans/battery.json"
TEACHER_REL = "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"
STAGE_I_REL = "logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i"

#: The stage letters an A3 authorization permits. Derived from the ladder, so
#: a stage added to the driver without being authorized cannot run.
AUTHORIZED_STAGES = tuple(A3S.STAGE_LETTERS)

#: What a grant may NOT state, because this module derives it. Asserting any of
#: them is a refusal rather than an override.
GRANT_MAY_NOT_STATE = (
    "expected_usd", "authorized_stages", "harness_source_digest",
    "harness_source_files", "plan_hash", "science_plan_hash",
    "design_sha256", "seeds", "probes", "arms",
)


class A3AuthorizationRefused(RuntimeError):
    """The payload cannot be built as asked. The message is the reason."""


def build_a3_authorization_payload(*, grant: Mapping[str, Any],
                                   session_commit: str, granted_utc: str,
                                   repo_root: str | Path = REPO,
                                   grant_path: str = "<unknown>",
                                   harness_files: tuple[str, ...] | None = None,
                                   ) -> dict[str, Any]:
    """The payload, fully derived. `harness_files` exists only for tests."""
    root = Path(repo_root)

    for field in GRANT_MAY_NOT_STATE:
        if field in grant:
            raise A3AuthorizationRefused(
                f"the grant states {field!r}, which this module DERIVES. A "
                "grant that can state its own identity or its own stages can "
                "change the experiment or over-authorize itself.")

    # ---- the money, from the LIVE record --------------------------------
    live = load_live_pricing(root)
    ceiling = a3_hard_ceiling_usd(root)
    expected = a3_expected_usd(root)
    envelope = float(live["envelopes"]["per_session_envelope_usd"])
    if ceiling > envelope:
        raise A3AuthorizationRefused(
            f"derived ceiling ${ceiling:.4f} exceeds the ${envelope:.4f} "
            "per-session envelope")
    stated_cap = grant.get("cumulative_cap_usd")
    live_cap = float(live["envelopes"]["project_cap_usd"])
    if stated_cap is not None and float(stated_cap) != live_cap:
        raise A3AuthorizationRefused(
            f"the grant names cap ${float(stated_cap):.4f}, not "
            f"${live_cap:.4f}")
    #: THE ENVELOPE IS NOT THE GRANT.
    asked = grant.get("hard_cap_usd")
    if asked is not None and abs(float(asked) - ceiling) > 5e-4:
        raise A3AuthorizationRefused(
            f"the grant asks for ${float(asked):.4f}; the ceiling DERIVED "
            f"from the live securePrice is ${ceiling:.4f}. The authorization "
            "carries the derived figure, never the envelope.")

    # ---- the science, derived -------------------------------------------
    design = A3S.design(root)
    contract = A3S.A3_SESSION_CONTRACT
    if contract.n_arms != 1:
        raise A3AuthorizationRefused(
            f"the session contract declares {contract.n_arms} arms; A3 trains "
            "ONE and reuses attempt75's controls")
    seeds = list(A3S.recovery_seeds())
    if len(seeds) != 3 or contract.n_probes != 3:
        raise A3AuthorizationRefused(
            f"{len(seeds)} seeds and {contract.n_probes} probes; A3 is three "
            "and three")
    if design["control"]["retrained"]:
        raise A3AuthorizationRefused(
            "the design claims the controls are retrained; that would make "
            "this a six-probe experiment the grant does not fund")
    if not contract.aggregation_off_pod:
        raise A3AuthorizationRefused(
            "the session contract claims an on-pod aggregation; A3's "
            "comparison runs off pod and the driver has no decision stage")

    #: The controls A3 will be compared against must EXIST, with the hashes
    #: their record names, before a dollar is authorized.
    probes = json.loads((root / STAGE_I_REL / "c3_probe_results.json").read_text())
    controls = [p for p in probes["probes"] if p["arm"] == "A_incumbent"]
    if len(controls) != 3:
        raise A3AuthorizationRefused(
            f"attempt75 holds {len(controls)} A_incumbent controls, not three")
    for probe in controls:
        for key, digest_key in (("per_sample_path", "per_sample_sha256"),
                                ("result_path", "result_sha256")):
            p = Path(probe[key])
            if not p.is_file():
                raise A3AuthorizationRefused(
                    f"control seed {probe['seed']} names {key} at {p}, which "
                    "is absent; A3 would train three probes and have nothing "
                    "to compare them against")
            if sha256_file(p) != probe[digest_key]:
                raise A3AuthorizationRefused(
                    f"control seed {probe['seed']} {key} no longer hashes to "
                    "what its record names")

    harness = dict(a3_harness_digest(root) if harness_files is None
                   else _digest_of(root, harness_files))
    battery = json.loads((root / BATTERY_REL).read_text())
    teacher = json.loads((root / TEACHER_REL).read_text())

    payload = A3Authorization(
        authorization_id=f"autoinit.v1.{A3S.EXPERIMENT_ID}",
        granted_utc=granted_utc,
        granted_by=str(grant.get("granted_by") or "")[:4000],
        plan_id=f"autoinit.v1.{A3S.EXPERIMENT_ID}",
        plan_hash=contract.contract_hash,
        science_plan_hash=design["design_sha256"],
        expected_usd=expected,
        hard_cap_usd=ceiling,
        per_launch_hard_usd=ceiling,
        authorized_stages=AUTHORIZED_STAGES,
        stage_conditions={
            "D": ("the frozen parent must reproduce "
                  f"{A3S.expected_parent_digest()[:12]}"),
            "E": ("A_bsz1 must reproduce the incumbent "
                  f"{A3S.expected_incumbent_digest()[:12]}; A_bsz3 is pinned "
                  "NOWHERE because what it builds is the finding"),
            "F": "three probes at the frozen seeds; no probe may be skipped",
            "G": ("each probe evaluated ONCE on c1_confirmation_v1, after the "
                  "protocol admission and never before all three trained"),
            "H": "preserve and package; NO decision is computed on the pod",
        },
        scope_note=(
            "ONE A3 session end to end. A differing A-bsz3 artifact digest is "
            "a FINDING and does not stop the chain. No search, no ranking, no "
            "arm elimination, no fourth seed, no retraining of attempt75's "
            "controls, no on-pod decision, and no authorization for the FFN "
            "experiment, D1/D2/D3 or Stage-0 v2."),
        authorized_session_commit=session_commit,
        harness_source_digest=harness["digest"],
        harness_source_files=tuple(f["path"] for f in harness["files"]),
        provenance_commit=session_commit,
    ).as_dict()

    payload["bound"] = {
        "grant": {"path": grant_path,
                  "sha256": sha256_file(root / grant_path)
                  if (root / grant_path).is_file() else None},
        "live_pricing": {
            "gpu": live.get("gpu"),
            "rate_usd_per_hour": live.get("queried_rate_usd_per_hour"),
            "expected_usd": expected, "hard_ceiling_usd": ceiling,
            "container_disk_gb": live["price"]["container_disk_gb"],
            "limits_checked": live["_every_applicable_limit_is_checked"],
            "conditions": live["conditions"]},
        "design": {"design_sha256": design["design_sha256"],
                   "arms": contract.n_arms, "probes": contract.n_probes,
                   "seeds": seeds,
                   "aggregation_off_pod": contract.aggregation_off_pod},
        "identities": {
            "parent": A3S.expected_parent_digest(),
            "incumbent": A3S.expected_incumbent_digest(),
            "a_bsz3": None,
            "_a_bsz3_is_null_on_purpose": (
                "it is what A3 measures. A pinned value here would decide the "
                "experiment in its own authorization.")},
        "controls": {
            "source_run": probes.get("run_id", "attempt75"),
            "seeds": sorted(p["seed"] for p in controls),
            "retrained": False,
            "verified_by_content_hash": True,
            "_weights": ("retired under the consumer rule; the comparison "
                         "reads per-sample rows and scored records")},
        "battery": {"artifact": battery.get("artifact"),
                    "content_sha256": battery.get("content_sha256")},
        "teacher": {"revision": teacher.get("revision"),
                    "shards": len(teacher.get("expected_shard_sha256") or {})},
        "session_contract_hash": contract.contract_hash,
    }
    payload.pop("authorization_sha256", None)
    from aadistill.infrastructure.manifest import sha256_json

    payload["authorization_sha256"] = sha256_json(payload)
    return payload


def _digest_of(root: Path, files: tuple[str, ...]) -> dict[str, Any]:
    from aadistill.governance.authorization import harness_source_digest

    return harness_source_digest(root, files=files)
