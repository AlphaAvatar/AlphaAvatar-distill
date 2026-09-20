#!/usr/bin/env python3
"""Regenerate the Phase-C2 behavioural grant PROPOSAL. Authorizes nothing.

    python scripts/autoinit/write_c2_behavioural_proposal.py

Every number is DERIVED when this runs — the protocol, the schedule, the six
arms' identities, the storage residency, the materialization bound and the
money. The previous version was hand-written, and when the candidate transport
changed it went stale in five places at once: the ceiling, the storage prose,
the execution sequence, the implementation state and the driver description. A
document nobody can regenerate is a document that silently stops describing the
thing it proposes.

It is a PROPOSAL. It is not a grant, not a readiness record, not an
authorization and not a bundle, and writing it permits nothing.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

from experiments.phase_c2 import behavioural as BH  # noqa: E402
from experiments.phase_c2 import behavioural_governance as BG  # noqa: E402

OUT = ("logs/stages/stage-1/phase_c2_behavioural/plans/"
       "c2_behavioural_grant_proposal.json")

def project_position(all_in: float) -> dict:
    """Where this ceiling sits against the cumulative cap.

    From `derive_budget.py`, which is the canonical deriver: the cap is owned by
    `configs/experiments/phase_c1/authorization.json ::
    accepted_pricing.cumulative_cap_usd` and the spend is computed from the run
    closeouts. Regexing the ledger prose instead — which an earlier draft of
    this script did — would make a narrative file the authority for money, and
    the ledger itself says every figure in it is derived from that config rather
    than restated.
    """
    import subprocess

    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/consolidate/derive_budget.py"),
         "--json"],
        capture_output=True, text=True, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    if out.returncode != 0:
        raise SystemExit(f"derive_budget.py failed: {out.stderr[-800:]}")
    derived = json.loads(out.stdout)
    project = derived["project"]
    spent = float(project["cumulative_spend_usd"])
    cap = float(project["cap_usd"])
    return {
        "source": "scripts/consolidate/derive_budget.py --json :: project",
        #: WHICH BALANCE THIS SESSION IS CHECKED AGAINST, stated rather than
        #: left to be inferred. `derive_budget.py` reports four balances that
        #: bind SEPARATELY and do not transfer into one another. This session
        #: reads the PROJECT envelope only.
        "funding_ownership": {
            "checked_against": "the $370 project accounting envelope",
            "is_a_new_c2_behavioural_authorization": True,
            "consumes_c1_execution_package": False,
            "subject_to_the_c1_per_session_ceiling": False,
            "c1_balances_are_separate_and_non_transferable": {
                "formal_remaining_usd":
                    derived["formal"]["remaining_usd"],
                "engineering_remaining_usd":
                    derived["engineering"]["remaining_usd"],
                "package_remaining_usd":
                    derived["package"]["remaining_usd"],
                "c1_per_session_ceiling_usd":
                    derived["per_session_ceiling_usd"],
            },
            "_why": (
                "the C1 execution package funds C1 formal sessions and its own "
                "GPU engineering validation, at a per-session ceiling derived "
                "for that work. This is a different experiment with a ceiling "
                "derived from its own topology, so it needs its own "
                "authorization and must not be presented as fitting inside a "
                "remainder reserved for something else. Dividing the wrong "
                "balance by the wrong per-session ceiling is how this "
                "repository once came to state that three full-ceiling "
                "sessions were fundable when two were."),
            "_the_envelope_is_not_permission": (
                "fitting the project envelope is an accounting fact. It is not "
                "launch permission and never has been."),
        },
        "cumulative_spend_usd": spent,
        "cumulative_cap_usd": cap,
        "authorized_cap_usd": cap,
        "remaining_before_usd": round(cap - spent, 4),
        "this_ceiling_all_in_usd": all_in,
        "cumulative_if_fully_spent_usd": round(spent + all_in, 4),
        "headroom_after_usd": round(cap - spent - all_in, 4),
        "_no_cap_increase_requested": True,
        "_remaining_is_not_permission": True,
        "_not_permission": (
            "remaining headroom is not authorization. A paid run needs a "
            "maintainer-written grant naming this session."),
    }


def build() -> dict:
    proto = BH.protocol(REPO)
    beh = proto["behavioural_selection"]
    sched = BH.schedule(REPO)
    candidates = BH.candidate_manifest(REPO)
    binding = BH.b_binding(REPO, device="cuda")
    storage = BH.storage_requirement(candidates, sched, REPO,
                                     b_must_materialize=True)
    money = BG.ceiling(REPO)
    closure = BG.current_executable(REPO)

    doc = {
        "schema": "aadistill.autoinit.c2_behavioural_proposal/v3",
        "_contract": (
            "A PROPOSAL for the Phase-C2 behavioural selection. Every figure is "
            "derived by scripts/autoinit/write_c2_behavioural_proposal.py from "
            "the frozen protocol, the committed evidence and the live tree. "
            "AUTHORIZES NOTHING."),
        "authorizes": "nothing",
        "plan_id": BG.PLAN_ID,
        "session_id": BG.SESSION_ID,
        "proposed_utc": datetime.now(timezone.utc).isoformat(),
        "plan_hash": BG.plan_hash(REPO),

        "campaign": {
            "campaign_id": BG.CAMPAIGN_ID,
            "is": (
                "ONE 12-probe scientific experiment. The campaign is the unit "
                "the grant funds and the unit within which a completed probe "
                "may be reused; a run attempt is one launcher invocation and "
                "the provider resource it draws."),
            "_they_were_the_same_string": (
                "the launcher passed the run id as the campaign id, which made "
                "the registered continuation policy unreachable: R1 permits "
                "reuse only inside one campaign, so a replacement resource "
                "became a different campaign and had to refuse every probe the "
                "previous resource had trained and verified off-pod."),
            "continuation": (
                "a replacement resource is a new RESOURCE and a new run attempt "
                "inside the same campaign. It may consume that campaign's "
                "destination-verified probes under the four conditions the "
                "resume preregistration registers — descriptor identity, "
                "re-identified bytes, a provider-confirmed non-billing "
                "predecessor, and cumulative spend inside the campaign "
                "ceiling. That is continuation of one preregistered experiment, "
                "not pooling across experiments."),
            "_the_ceiling_is_cumulative": (
                "all_in_hard_usd below bounds the CAMPAIGN across every "
                "resource and run attempt it takes. With a ceiling sized for "
                "one full session, a continuation after a resource that already "
                "spent real money is REFUSED at the gate rather than permitted "
                "to overspend. Funding a campaign for more than one full "
                "session is a maintainer decision."),
        },

        "authorization_terms": {
            "at_the_quoted_rate": BG.authorization_terms(
                REPO, rate_usd_per_hour=BG.QUOTED_RATE_USD_PER_HOUR),
            "_derive_again_at_issuance": (
                "these are the amounts at the rate this proposal was priced "
                "at. Issuance must re-quote gpuTypes.securePrice and derive "
                "them again at that rate: an authorization carrying a dollar "
                "window derived from a stale quote is a window that does not "
                "bound the bill. If the re-quote materially changes the dollar "
                "authorization, it goes back to the maintainer."),
            "_five_distinct_amounts": (
                "rate, authorized runtime, GPU dollars, separately billed disk "
                "dollars and the all-in total. The launcher derives its "
                "deadline from gpu_hard_usd and hard_runtime_minutes at the "
                "live rate — the shorter of the two — so a card above the "
                "authorized rate is refused and torn down rather than paid "
                "for, and a card below it does not extend the experiment."),
        },

        "protocol_binding": {
            "document": BH.PROTOCOL,
            "protocol_sha256": proto["protocol_sha256"],
            "schedule": sched,
            "seeds": beh["seeds"],
            "batteries": beh["batteries"],
            "decision_rule": beh["frozen_science"]["decision_rule"],
            "terminal_results": beh["terminal_results"],
        },

        "candidate_inputs": {
            "n": len(candidates),
            "source": (
                "attempt 3's frozen Top-5 selection, joined to the "
                "destination-verified products the replay reconstructed"),
            "candidates": [
                {k: c[k] for k in ("state_id", "artifact_digest",
                                   "weights_digest", "arch_signature",
                                   "num_parameters",
                                   "rank_in_frozen_selection")}
                for c in candidates],
        },

        "candidate_transport": {
            "mode": BG.CANDIDATE_TRANSPORT,
            "_why_not_staged": (
                "each arm is a 1.19 GB checkpoint and there are six. The scp "
                "path copies local assets AFTER the pod exists, so they bill, "
                "and the shared runner gives each asset a hardcoded 600-second "
                "timeout; one of these needs ~1650 s against a dev-box uplink "
                "measured at 0.44-0.79 MB/s. Recovery-continuation attempt 2 "
                "died on exactly that, staging exactly this size, and its "
                "write-up concluded the failure was 'arithmetic rather than "
                "luck'. The hub-relay route that repaired attempt 2 needs quota "
                "this account does not have: the LFS batch endpoint was asked "
                "directly at $0 on 2026-09-19 and again on 2026-09-20 and "
                "ACCEPTS one 1.19 GB object while REFUSING 5.95 GB for "
                "'Private repository storage limit reached'."),
            "_what_is_done_instead": (
                "every arm is materialized on the pod from the teacher along a "
                "path pinned at EVERY step to the artifact digest the frozen "
                "record holds — the mechanism the replay proved by reproducing "
                "all five byte-for-byte. It is not a search: no beam, no "
                "expansion, no ranking, no selection. A digest mismatch stops "
                "the session as a scientific finding."),
            "_maintainer_alternative": (
                "freeing or buying Hugging Face private storage would allow the "
                "arms to be published at $0 before launch and pulled at hub "
                "speed, removing the materialization minutes below. That is a "
                "maintainer decision and never an autonomous repair."),
            "materialization": money["materialization"],
        },

        "incumbent_b": {
            "role": binding["role"],
            "construction": binding["construction"],
            "required_identity": binding["required_identity"],
            "availability": binding["availability"],
        },

        "storage": storage,
        "money": money,
        "project_position": project_position(
            money["hard_ceiling"]["all_in_usd"]),

        "durability": {
            "mechanism": (
                "the driver announces each finished unit's identity to its "
                "evidence; the launcher's poll hook pulls the bytes off-pod "
                "during the run and re-identifies them at the destination "
                "against artifact_digest, weights_digest, config_sha256, "
                "single_shard_sha256, arch_signature and num_parameters."),
            "destination": "/home/ecs-user/aad-artifacts/phase_c2_behavioural",
            "_backend_has_room": (
                "12 probes x 1.11 GiB = 13.3 GiB against 43 GB free, checked "
                "by destination_gate before a pod is created. The Hugging Face "
                "route is NOT used: C1 attempt 18 ran a correct preservation "
                "mechanism on six probes and preserved none, because every "
                "upload was refused for quota."),
            "_preservation_is_not_permission": (
                "saving a probe authorizes nothing about reusing it. Reuse is "
                "governed by the resume rules, not by the existence of a file."),
        },

        "resume_policy": {
            "record": ("logs/stages/stage-1/phase_c2_behavioural/plans/"
                       "c2_behavioural_resume_preregistration.json"),
            "summary": (
                "a probe may be reused only within the same scientific "
                "CAMPAIGN — across run attempts and replacement resources of "
                "that campaign, never across campaigns — only when it matches "
                "the descriptor its rung derives from the frozen protocol, and "
                "only when its bytes still re-identify to what was announced. "
                "It is never retrained. Screening commits once: a continuation "
                "confirms the candidate its own campaign advanced and may not "
                "rerun screening for another outcome. Ranking waits for all six "
                "screening results; no verdict is computed from a partial "
                "confirmation field."),
        },

        "explicitly_not_proposed": [
            "any beam search or expansion",
            "re-ranking the frozen Top-5 or regenerating a selection",
            "re-measuring B's completed state evaluation",
            "retraining a completed probe for a different outcome",
            "a fourth seed, a tie-break rung or a forced winner",
            "any C3 or C4 work",
        ],

        "executable_closure": {
            "digest": closure["digest"],
            "n_files": closure["n_files"],
            "entry_points": list(BG.ENTRY_POINTS),
            "declared_inputs": list(BG.declared_inputs(REPO)),
        },

        "execution_sequence": [
            "P  verify the teacher by shard hash, then materialize all six arms "
            "— the five candidates and B — each along its pinned path and each "
            "gated on its exact recorded identity. A mismatch stops the session "
            "as a provenance finding; no probe starts.",
            "S  six fresh screening probes — five candidates and B on the one "
            "frozen screening seed — each announced for durability the moment "
            "it finishes and pulled off-pod while the next one trains.",
            "R  score c2_screening_v1 through its own pinned scorer, rank by "
            "paired delta against B, advance EXACTLY one by the frozen "
            "tie-break. No verdict may leave this stage.",
            "C  six fresh confirmation probes — the advanced candidate and B, "
            "paired across the three frozen seeds. Cannot begin until every "
            "screening probe is trained AND scored.",
            "D  score c1_confirmation_v1, apply the frozen Phase-C rule at C2's "
            "own bootstrap seed, and STOP. GO, NO_GO and INCONCLUSIVE are all "
            "complete results.",
        ],

        "implementation_state": {
            "governance_module":
                "scripts/experiments/phase_c2/behavioural.py — BUILT",
            "launch_governance":
                "scripts/experiments/phase_c2/behavioural_governance.py — BUILT",
            "schedule_control_flow":
                "scripts/experiments/phase_c2/behavioural_schedule.py — BUILT",
            "decision":
                "scripts/experiments/phase_c2/behavioural_decision.py — BUILT. "
                "Composes C1's paired_differences, decision_inputs, "
                "stratified_cluster_bootstrap and decide; supplies a C2 "
                "decision-rule view derived from the frozen protocol rather "
                "than instantiating a C1IsolationPlan whose invariants do not "
                "describe this experiment.",
            "screening_scorer":
                "scripts/autoinit/score_c2_screening.py — BUILT. Reuses C1's "
                "battery-agnostic scoring loop and result builder unchanged, "
                "pinned to the C2 screening battery's own identity record. C1's "
                "entry point keeps its equality pins.",
            "driver":
                "scripts/pod/autoinit_c2_behavioural_driver.py — BUILT, "
                "STANDALONE. Composes train_stage3.py, uncapped_eval.py, the "
                "two scorers, build_evaluation_package, the admission gate and "
                "the Phase-C inference. It does NOT subclass C1Driver, which "
                "would inherit C1's authorization, plan identity, seeds and "
                "audit roots.",
            "launcher":
                "scripts/pod/autoinit_c2_behavioural_launch.py — BUILT. Eight "
                "$0 prechecks including the campaign continuation gate, a "
                "budget built from the ONE canonical phase decomposition, a "
                "deadline derived from the authorization's own rate, GPU "
                "dollars and authorized runtime, per-poll durability, "
                "destination re-identification and a teardown gate that "
                "refuses while evidence is unreadable.",
            "b_binding": "BUILT",
            "storage_derivation": "BUILT",
            "rehearsal":
                "tests/c2_behavioural_preflight/ — BUILT. One production-path "
                "rehearsal drives the real driver P through D and reaches all "
                "three terminal states from separate deterministic fixtures, "
                "replacing only hardware-bound calls.",
            "grant": "NOT REQUESTED — no grant, readiness record, "
                     "authorization, bundle or provider resource exists for "
                     "this session, and none may be created without a "
                     "maintainer decision.",
        },
    }
    doc["proposal_sha256"] = sha256_json(doc)
    return doc


def main() -> int:
    doc = build()
    path = REPO / OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    money = doc["money"]
    pos = doc["project_position"]
    print(f"wrote {OUT}")
    print(f"  expected ${money['expected']['all_in_usd']} · "
          f"hard ${money['hard_ceiling']['all_in_usd']} · "
          f"{doc['storage']['provision_gb']} GB")
    print(f"  cumulative if fully spent ${pos['cumulative_if_fully_spent_usd']} "
          f"of ${pos['cumulative_cap_usd']} "
          f"(headroom ${pos['headroom_after_usd']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
