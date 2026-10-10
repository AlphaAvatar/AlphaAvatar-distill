"""Build and check D1 BEHAVIOURAL one-use authorizations. All substance here.

`issue_d1_behavioural_authorization.py` is the command around `build_payload`;
this module owns what the artifact binds and every reason to refuse issuing it.
The money helpers are IMPORTED from `d1_authorization` -- `live_money`,
`live_secure_price`, `check_the_four_conditions` are rung-agnostic and a second
copy of any of them is a second thing that can disagree about the budget.
What is behavioural-specific lives here: the rung's priced cell (read from the
design's `budget.chain.sessions`, the one pricing owner), the behavioural
executable closure, the preregistration binding, and the payload.

TWO RUNGS, TWO ARTIFACTS, issued separately: a confirmation artifact cannot
exist before screening has named an advancing candidate, and `build_payload`
refuses to pre-name one. The candidate a confirmation issuance binds must be
read from the screening rung's committed selection record, never typed.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

from stages.phase_d1 import behavioural_governance as BG  # noqa: E402
from stages.phase_d1.behavioural_governance import (  # noqa: E402
    D1BehaviouralRefused,
)
from stages.phase_d1.d1_authorization import (  # noqa: E402
    check_the_four_conditions,
    live_money,
    live_secure_price,
)

#: Where this session's own subruns book what they spent. P12.1: "The budget is
#: cumulative across every resource and subrun of the task. A rerun does not
#: reset it."
RUNS_GLOB = "logs/stages/stage-1/phase_d1/runs/d1_behavioural_*/closeout/outcome.json"

DESIGN_REL = "logs/stages/stage-1/phase_d1/plans/d1_design.json"
PREREG_REL = ("logs/stages/stage-1/phase_d1/plans/"
              "d1_behavioural_preregistration.json")

#: A grant may not state what this module derives. The rung IS grant-stated --
#: it is the scope the maintainer funded -- and is required to match the
#: issuance; everything identity- or price-shaped is derived.
GRANT_MAY_NOT_STATE = (
    "authorization_id", "plan_id", "plan_hash", "design_hash",
    "contract_hash", "battery_role", "battery_content_id", "recipe_id",
    "n_probes", "seeds", "expected_usd", "per_launch_hard_usd",
    "authorized_stages", "stage_conditions", "preregistration_sha256",
)

#: The behavioural chain's entry points. The closure is DERIVED from these by
#: an import walk -- never a hand-maintained list -- and it is a DIFFERENT
#: closure from the search's: the two sessions execute different code and an
#: authorization over the wrong set would verify an executable nobody runs.
BEHAVIOURAL_ENTRY_POINTS: tuple[str, ...] = (
    "scripts/stages/stage-1/phase_d1/autoinit_d1_behavioural_launch.py",
    "scripts/stages/stage-1/phase_d1/autoinit_d1_behavioural_driver.py",
    "scripts/stages/stage-1/phase_d1/behavioural.py",
    "scripts/stages/stage-1/phase_d1/behavioural_materialize.py",
    "scripts/stages/stage-1/phase_d1/behavioural_governance.py",
    "scripts/stages/stage-1/phase_d1/score_d1_screening.py",
    "scripts/stages/stage-1/phase_d1/score_d1_confirmation.py",
    "scripts/shared/pod/collect_artifacts.py",
)

#: Files no import edge reaches, whose bytes still decide what runs or what is
#: authorized. The DESIGN is deliberately absent (`plan_hash` is its hash); the
#: PREREGISTRATION and the REPLAY PLAN are present because the pod consumes
#: both and neither is imported.
BEHAVIOURAL_DECLARED_INPUTS: tuple[str, ...] = (
    "scripts/shared/pod/autoinit_preflight_setup.sh",
    "configs/stages/stage-1/phase_c1/authorization.json",
    "configs/stages/stage-1/phase_d1/d1_behavioural_artifacts.json",
    "configs/stages/stage-1/phase_d1/d1_behavioural_artifacts_failed.json",
    "scripts/stages/stage-1/families/d_series/scoring_protocol.py",
    "logs/stages/stage-1/phase_d1/plans/d1_behavioural_preregistration.json",
    "logs/stages/stage-1/phase_d1/plans/d1_behavioural_replay_plan.json",
)

SOURCE_ROOTS: tuple[str, ...] = ("src", "scripts", "scripts/pod",
                                 "scripts/autoinit")

AUTHORIZED_STAGES: tuple[str, ...] = ("preflight", "materialize", "contract",
                                      "probes", "preserve")

STAGE_CONDITIONS: dict[str, str] = {
    "preflight": ("the battery bytes against the manifest, the role "
                  "disjointness, B's construction pins and the recovery "
                  "pack's digest -- refused before any weights load"),
    "materialize": ("every arm this rung trains from, built along its "
                    "digest-pinned path and gated on its exact recorded "
                    "identity; a mismatch is a scientific stop, not a retry"),
    "contract": ("the location-free session contract recomputed on the pod "
                 "and required equal to the hash this artifact binds, before "
                 "any probe trains"),
    "probes": ("each probe trained under the frozen recipe (overrides only "
               "inside C1's allowed set), evaluated uncapped through the "
               "pinned evaluation package, scored by the rung's own pinned "
               "scorer, and announced for durability the moment it finishes"),
    "preserve": ("screening: the mechanical ranking and the one advancing "
                 "candidate (or none). Confirmation: the probes' evidence and "
                 "STOP -- the verdict is computed off-pod at $0. No promotion "
                 "is computed in any session."),
}


class D1BehaviouralIssuanceRefused(D1BehaviouralRefused):
    """A behavioural issuance condition does not hold."""


def behavioural_current_executable(repo_root: str | Path = REPO
                                   ) -> dict[str, Any]:
    """What a behavioural session would execute NOW, derived from the tree."""
    from aadistill.governance.closure import ClosureError, derive

    try:
        return derive(repo_root, "phase_d1_behavioural",
                      BEHAVIOURAL_ENTRY_POINTS, BEHAVIOURAL_DECLARED_INPUTS,
                      roots=SOURCE_ROOTS)
    except ClosureError as exc:
        raise D1BehaviouralIssuanceRefused(
            f"cannot derive the behavioural executable set: {exc}") from exc


def session_cell(rung: str, repo_root: str | Path = REPO) -> dict[str, Any]:
    """The rung's PRICED CELL, read from the design -- the one pricing owner.

    `budget.chain.sessions.screening` / `.confirmation`, written by
    `write_d1_design.py` from the measured production basis. A second pricing
    formula here could disagree with the design the maintainer funded.
    """
    if rung not in ("screening", "confirmation"):
        raise D1BehaviouralIssuanceRefused(f"unknown rung {rung!r}")
    design = json.loads((Path(repo_root) / DESIGN_REL).read_text())
    cell = design["budget"]["chain"]["sessions"][rung]
    return {
        "hard_ceiling_usd": float(cell["hard_ceiling_usd"]),
        "hard_ceiling_minutes": float(cell["hard_ceiling_minutes"]),
        "expected_minutes": float(cell["expected_minutes"]),
        "price_per_hour": float(cell["price_per_hour"]),
        "container_disk_usd": float(cell["container_disk_usd"]),
        "container_disk_gb": int(cell["container_disk_gb"]),
        "gpu_usd": float(cell["gpu_usd"]),
        "n_probes": int(cell["n_probes"]),
        "n_arms": int(cell["n_arms"]),
        "arm_materialization_minutes": float(
            cell["arm_materialization_minutes"]),
        "session_overhead_minutes": float(cell["session_overhead_minutes"]),
        "probe_minutes": float(cell["probe_minutes"]),
        "overrun_factor": float(cell["overrun_factor"]),
        "_owner": f"{DESIGN_REL} :: budget.chain.sessions.{rung}",
    }


def reprice_rung_at(rate_usd_per_hour: float, rung: str,
                    repo_root: str | Path = REPO) -> dict[str, Any]:
    """The SAME accepted minute bound, costed at a different GPU rate.

    No minute assumption changes and no GPU timing is repeated: the accepted
    `hard_ceiling_minutes` and the accepted container-disk dollars are kept,
    and only the GPU term moves with the rate. A lower live rate therefore
    LOWERS the authorized ceiling -- the half of the contract an
    abort-if-higher launcher check cannot provide. Mirrors
    `d1_authorization.reprice_at`, which is bound to the SEARCH's priced cell
    and cannot be pointed at another without becoming two pricing owners.
    """
    cell = session_cell(rung, repo_root)
    minutes = cell["hard_ceiling_minutes"]
    expected_minutes = cell["expected_minutes"]
    disk = cell["container_disk_usd"]
    gpu_hard = round(minutes / 60.0 * rate_usd_per_hour, 4)
    hard = round(gpu_hard + disk, 4)
    expected = round(expected_minutes / 60.0 * rate_usd_per_hour
                     + disk * expected_minutes / minutes, 4)
    return {
        "rung": rung,
        "hard_ceiling_usd": hard,
        "gpu_hard_usd": gpu_hard,
        "disk_hard_usd": disk,
        "expected_usd": expected,
        "hard_ceiling_minutes": minutes,
        "expected_minutes": expected_minutes,
        "price_per_hour": float(rate_usd_per_hour),
        "accepted_at_price_per_hour": cell["price_per_hour"],
        "accepted_hard_ceiling_usd": cell["hard_ceiling_usd"],
        "container_disk_gb": cell["container_disk_gb"],
        "_what_moved": (
            "ONLY the GPU rate. The accepted minute bound and the accepted "
            "container-disk dollars are unchanged; this is the dollar "
            "consequence of an already-accepted bound."),
    }


#: Minutes of all-in spend held back from the attempt's runtime bound so that
#: teardown and final evidence preservation are FUNDED rather than hoped for.
#:
#: Measured basis: attempt 132919's teardown took 2.2 minutes (timeline
#: 190.0 -> 192.2) and its off-pod evidence securing took 2.3 (165.4 -> 167.7).
#: A completing screening session has more to move at the end -- ten probes'
#: scores, per-sample rows and raw generations -- so the reserve is set at an
#: order of magnitude above the measured teardown rather than at it. It is
#: expressed in MINUTES because that is what was measured; the dollars follow
#: from the live rate.
TEARDOWN_RESERVE_MINUTES = 45.0


def narrow_to_remaining(priced: Mapping[str, Any], *, remaining_usd: float,
                        reserve_minutes: float = TEARDOWN_RESERVE_MINUTES
                        ) -> dict[str, Any]:
    """THIS ATTEMPT's bounds: the priced cell, capped by what is left.

    The frozen design's cell is not touched -- `reprice_rung_at` still owns the
    accepted minute bound and the accepted disk dollars. This narrows the bound
    a single attempt may carry, which is a different quantity: after four
    subruns the envelope's remainder is smaller than one full session's price,
    so reusing the design's 1355.71-minute bound would arm a watchdog that
    permits more spend than the maintainer authorized.

    Both terms move with the runtime, GPU and disk alike -- a bound that scaled
    only the GPU term would understate what a long run actually costs, which is
    the same GPU-only-is-not-all-in error one level up.

    The minute figure is FLOORED. It is a limit, and a limit that rounds up is
    a limit that can be exceeded.
    """
    import math

    minutes_priced = float(priced["hard_ceiling_minutes"])
    disk_per_min = float(priced["disk_hard_usd"]) / minutes_priced
    gpu_per_min = float(priced["price_per_hour"]) / 60.0
    all_in_per_min = gpu_per_min + disk_per_min

    reserve_usd = round(reserve_minutes * all_in_per_min, 4)
    spendable = remaining_usd - reserve_usd
    if spendable <= 0:
        raise D1BehaviouralIssuanceRefused(
            f"the remaining ${remaining_usd:.4f} does not even cover the "
            f"${reserve_usd:.4f} teardown and evidence-preservation reserve. "
            "An attempt that cannot fund its own teardown must not start.")

    minutes = min(minutes_priced, math.floor(spendable / all_in_per_min))
    gpu_hard = round(minutes / 60.0 * float(priced["price_per_hour"]), 4)
    disk_hard = round(minutes * disk_per_min, 4)
    return {
        "hard_runtime_minutes": float(minutes),
        "gpu_hard_usd": gpu_hard,
        "disk_hard_usd": disk_hard,
        "all_in_hard_usd": round(gpu_hard + disk_hard, 4),
        "teardown_reserve_usd": reserve_usd,
        "teardown_reserve_minutes": float(reserve_minutes),
        "remaining_usd": round(remaining_usd, 4),
        "all_in_usd_per_minute": round(all_in_per_min, 6),
        "narrowed": minutes < minutes_priced,
        "priced_hard_ceiling_minutes": minutes_priced,
        "_why": ("the attempt's bound is min(the design's accepted minute "
                 "bound, what the remaining envelope funds after holding back "
                 "teardown). Both the GPU and the disk term scale with it."),
    }


def consumed_by_prior_subruns(repo_root: str | Path = REPO, *,
                              rung: str | None = None) -> dict[str, Any]:
    """What this RUNG's earlier subruns have already spent, from closeouts.

    Reads `cost.actual_usd` -- the field `derive_budget` books from, so this
    agrees with the budget book by construction rather than by a second
    convention. A run with no closeout contributes nothing: it has either not
    terminated or not been closed, and in both cases there is no settled
    figure to net off. That is deliberately the conservative direction here --
    an unclosed subrun makes the remainder look LARGER, which is why closing a
    subrun is part of its terminal path and not optional tidying.

    SCOPED BY RUNG, because the two rungs are two one-use grants with two
    envelopes. Screening's four subruns spent $8.2180 of screening's $25.0053;
    netting that against confirmation's separate $15.2025 would refuse a
    confirmation launch for money it never had access to. A closeout that does
    not say which rung it belongs to cannot be attributed, and is refused
    rather than guessed at.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(repo_root).glob(RUNS_GLOB)):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise D1BehaviouralIssuanceRefused(
                f"{path} cannot be read ({type(exc).__name__}), so this "
                "session's consumed total is unknown. A paid authorization "
                "may not be issued against an unknown remainder.") from exc
        #: ALL-IN FIRST, exactly as `derive_budget` reads it. `cost.actual_usd`
        #: is GPU price x elapsed, and the envelope is denominated all-in: the
        #: GPU-only figures understated these four subruns by $0.1125, which is
        #: a remainder that looks larger than it is. `money.all_in_usd` is the
        #: provider-measured charge written by
        #: `reconcile_d1_behavioural_spend.py`.
        money = record.get("money") or {}
        usd = money.get("all_in_usd")
        basis = "money.all_in_usd"
        if usd is None:
            usd = (record.get("cost") or {}).get("actual_usd")
            basis = "cost.actual_usd (GPU ONLY -- not reconciled)"
        if usd is None:
            raise D1BehaviouralIssuanceRefused(
                f"{path} is a closeout with no money.all_in_usd and no "
                "cost.actual_usd. Its spend cannot be netted off the "
                "envelope, so the remainder is unknown.")
        its_rung = record.get("rung")
        if not its_rung:
            raise D1BehaviouralIssuanceRefused(
                f"{path} names no rung, so ${float(usd):.4f} cannot be "
                "attributed to an envelope. The two rungs are two one-use "
                "grants; an unattributable spend makes both remainders "
                "unknown.")
        if rung is not None and its_rung != rung:
            continue
        rows.append({"run_id": record.get("run_id"), "rung": its_rung,
                     "usd": float(usd), "basis": basis,
                     "minutes": (record.get("cost") or {}).get(
                         "elapsed_minutes")})
    return {"n_subruns": len(rows),
            "consumed_usd": round(sum(r["usd"] for r in rows), 4),
            "subruns": rows,
            "rung": rung,
            "all_reconciled": all(r["basis"] == "money.all_in_usd"
                                  for r in rows),
            "_source": RUNS_GLOB}


def preregistration(repo_root: str | Path = REPO) -> dict[str, Any]:
    """The committed preregistration, verified against its own hash AND
    against what this tree derives. A stale preregistration names a session
    nobody would run."""
    import importlib.util

    path = Path(repo_root) / PREREG_REL
    if not path.is_file():
        raise D1BehaviouralIssuanceRefused(
            f"{PREREG_REL} does not exist; write it first with "
            "write_d1_behavioural_preregistration.py --write. C0 requires the "
            "seeds hash-bound there before any candidate result exists, and "
            "an authorization without it binds a session nothing "
            "preregistered.")
    doc = json.loads(path.read_text())
    src = (Path(repo_root) /
           "scripts/stages/stage-1/phase_d1/"
           "write_d1_behavioural_preregistration.py")
    spec = importlib.util.spec_from_file_location("_d1b_prereg_writer", src)
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    stated = doc.get("preregistration_sha256")
    recomputed = writer.preregistration_identity(doc)
    if not stated or stated != recomputed:
        raise D1BehaviouralIssuanceRefused(
            f"{PREREG_REL} states preregistration_sha256={stated} and "
            f"recomputes to {recomputed}; it is not the document it claims to "
            "be. Regenerate it rather than issuing over it.")
    live = writer.preregistration_identity(writer.build(Path(repo_root)))
    if live != recomputed:
        raise D1BehaviouralIssuanceRefused(
            f"this tree derives preregistration {live[:16]} and the committed "
            f"record carries {recomputed[:16]}. An owner record moved since "
            "the preregistration was written; regenerate and re-review before "
            "issuing.")
    return doc


def build_payload(*, grant: Mapping[str, Any], session_commit: str,
                  granted_utc: str, run_id: str, rung: str,
                  advancing_candidate: str | None = None,
                  repo_root: str | Path = REPO,
                  consumed_usd: float | None = None,
                  live_rate: float | None = None) -> dict[str, Any]:
    """The behavioural authorization payload, fully derived from this tree.

    `live_rate` is quoted here by default, immediately before issuance, which
    is what the package contract requires; passing one is for tests.

    `consumed_usd` is what this session's earlier subruns already spent, and is
    DERIVED from their closeouts by default -- an issuance must not have to be
    told that the envelope is already partly spent. Passing it explicitly is
    for tests that are about something other than the remainder; passing `0.0`
    asserts a pristine envelope, so a test that does so is stating an
    assumption rather than quietly inheriting one.
    """
    from stages.phase_d1 import behavioural as B

    root = Path(repo_root)
    for f in GRANT_MAY_NOT_STATE:
        if f in grant:
            raise D1BehaviouralIssuanceRefused(
                f"the grant states {f!r}, which this module DERIVES. A grant "
                "that can state its own identity or its own stages can change "
                "the experiment or over-authorize itself.")
    if rung not in ("screening", "confirmation"):
        raise D1BehaviouralIssuanceRefused(f"unknown rung {rung!r}")
    granted_rung = str(grant.get("rung") or "")
    if granted_rung != rung:
        raise D1BehaviouralIssuanceRefused(
            f"the grant funds the {granted_rung!r} rung and this issuance is "
            f"for {rung!r}; the two rungs have different probe counts, seeds, "
            "batteries and prices, and neither grant covers the other.")
    if not session_commit or len(session_commit) != 40:
        raise D1BehaviouralIssuanceRefused(
            f"session_commit {session_commit!r} is not a full commit id; the "
            "pod checks out this tree and the authorization must bind which "
            "one")
    if not str(run_id or "").strip():
        raise D1BehaviouralIssuanceRefused(
            "no run_id: a one-use artifact without a run identity is "
            "reusable by accident")
    if rung == "screening" and advancing_candidate:
        raise D1BehaviouralIssuanceRefused(
            "a screening issuance may not name an advancing candidate; "
            "screening is what produces one")
    if rung == "confirmation" and not advancing_candidate:
        raise D1BehaviouralIssuanceRefused(
            "a confirmation issuance must name the advancing candidate, read "
            "from the screening rung's committed selection record")

    design = json.loads((root / DESIGN_REL).read_text())
    blockers = list(design["open_blockers"])
    if blockers:
        raise D1BehaviouralIssuanceRefused(
            f"the design still reports open blockers {blockers}; an "
            "authorization may not be issued over one")

    prereg = preregistration(root)

    money = live_money(root)
    if "phase_d1" not in money["funds_formal_sessions_of"]:
        raise D1BehaviouralIssuanceRefused(
            "phase_d1 is not in funds_formal_sessions_of "
            f"{money['funds_formal_sessions_of']}; its formal spend would "
            "reach the project cumulative while the formal book reported it "
            "as never having happened")

    quote = ({"gpu": "NVIDIA L40S", "usd_per_hour": float(live_rate),
              "_quoted": "supplied by the caller (tests only)"}
             if live_rate is not None else live_secure_price())
    priced = reprice_rung_at(quote["usd_per_hour"], rung, root)
    ceiling = priced["hard_ceiling_usd"]
    failed = check_the_four_conditions(ceiling=ceiling, money=money)
    if failed:
        raise D1BehaviouralIssuanceRefused(
            f"at the live rate ${quote['usd_per_hour']:.4f}/h the {rung} "
            "session is not fundable:\n  - " + "\n  - ".join(failed)
            + "\n\nThe science does not change to absorb a price movement: "
              "neither the probe schedule nor the minute bound may be "
              "narrowed. Stop at $0.")

    #: WHAT THIS SESSION'S EARLIER SUBRUNS ALREADY SPENT, netted off before
    #: anything is authorized.
    #:
    #: THE DEFECT THIS CLOSES. `per_launch_hard_usd` used to be the full
    #: `ceiling`, which made `require_within_launch_limit` -- whose own
    #: docstring says "One launch may not spend the cumulative allowance of
    #: several" -- vacuous for this session: the per-launch limit and the
    #: cumulative cap were the same number, so the guard could never fire.
    #: Four screening subruns each booked "within authorization" at $25.0053
    #: while draining the envelope to $16.79, and a fifth would have carried a
    #: $24.63 hard GPU bound against a $16.79 remainder -- authorizing an
    #: ~$8 overspend of the maintainer's envelope that no check would catch.
    #: Two coinciding numbers hid which one was being read.
    spent = (consumed_by_prior_subruns(root, rung=rung)
             if consumed_usd is None else
             {"n_subruns": None, "consumed_usd": round(float(consumed_usd), 4),
              "subruns": [], "rung": rung,
              "_source": "supplied by the caller"})
    #: THE ENVELOPE THE REMAINDER IS TAKEN FROM.
    #:
    #: By default it is the price of one session -- the original one-use shape,
    #: where the grant funds a single launch. A grant may instead state a
    #: CUMULATIVE envelope, which is what the 2026-10-11 maintainer decision
    #: did: it raised D1 screening's cumulative all-in envelope from $25.0053
    #: to $30.0000 and charged all four historical subruns against it. That is
    #: additive rather than a replacement, so a grant written in the older
    #: shape still issues exactly as before.
    envelope = ceiling
    envelope_source = "the rung's priced ceiling at the live rate"
    stated_envelope = grant.get("cumulative_envelope_usd")
    if stated_envelope is not None:
        envelope = round(float(stated_envelope), 4)
        envelope_source = "the grant's stated CUMULATIVE envelope"
        prior = grant.get("previous_envelope_usd")
        added = grant.get("additional_authorized_usd")
        if prior is not None and added is not None and \
                abs((float(prior) + float(added)) - envelope) > 5e-4:
            raise D1BehaviouralIssuanceRefused(
                f"the grant states a ${envelope:.4f} cumulative envelope but "
                f"${float(prior):.4f} previous + ${float(added):.4f} "
                "additional do not sum to it. A budget amendment whose parts "
                "disagree with its total cannot be audited.")
        if envelope > float(money["per_session_envelope_usd"]) + 5e-4:
            raise D1BehaviouralIssuanceRefused(
                f"the stated ${envelope:.4f} cumulative envelope exceeds the "
                f"${money['per_session_envelope_usd']:.4f} per-session limit, "
                "which this amendment was required to preserve.")

    remaining = round(envelope - spent["consumed_usd"], 4)
    if spent["consumed_usd"] > 0 and remaining < priced["expected_usd"]:
        who = ("a caller-supplied prior spend of" if spent["n_subruns"] is None
               else f"this session's {spent['n_subruns']} earlier subrun(s) "
                    "consumed")
        raise D1BehaviouralIssuanceRefused(
            f"{who} "
            f"${spent['consumed_usd']:.4f} of the ${envelope:.4f} "
            f"envelope ({envelope_source}), leaving ${remaining:.4f}. A "
            f"complete {rung} run is "
            f"priced at ${priced['expected_usd']:.4f} expected, so the "
            "remainder cannot fund one -- and a schedule that stops partway "
            "yields no verdict under the frozen design.\n\n"
            "The science does not shrink to fit the remainder: neither the "
            "probe schedule nor the minute bound may be narrowed. A budget "
            "increase is a MAINTAINER decision (P12.1: this authority 'never "
            "covers additional budget'). Stop at $0 and report.")

    asked = grant.get("hard_cap_usd")
    if asked is not None and stated_envelope is not None:
        raise D1BehaviouralIssuanceRefused(
            "the grant states BOTH a per-attempt hard_cap_usd and a "
            "cumulative_envelope_usd. In envelope mode the per-attempt cap is "
            "DERIVED from the envelope minus what is already spent; a second "
            "stated figure is a second thing that can disagree with it.")
    if asked is not None and abs(float(asked) - ceiling) > 5e-4:
        raise D1BehaviouralIssuanceRefused(
            f"the grant asks for ${float(asked):.4f}; the ceiling DERIVED at "
            f"the live ${quote['usd_per_hour']:.4f}/h is ${ceiling:.4f}. The "
            "authorization carries the derived figure.")

    #: THIS ATTEMPT's bounds, narrowed to the remainder. The maintainer's
    #: 2026-10-11 condition is explicit that the previous 1355.71-minute hard
    #: runtime estimate may not be blindly reused when the per-launch available
    #: budget is lower, and that the launcher/watchdog bound must be no greater
    #: than the remaining authorized ALL-IN allowance with a real teardown and
    #: evidence-preservation reserve.
    bounds = narrow_to_remaining(priced, remaining_usd=remaining)
    stated_cap = grant.get("cumulative_cap_usd")
    if stated_cap is not None and \
            abs(float(stated_cap) - money["project_cap_usd"]) > 5e-4:
        raise D1BehaviouralIssuanceRefused(
            f"the grant names cap ${float(stated_cap):.4f}, not "
            f"${money['project_cap_usd']:.4f}")

    #: THE CONTRACT THE DRIVER WILL ASSERT, built and hashed here so the
    #: artifact binds the identities the run will actually carry. Building it
    #: verifies the arms' bytes on this host, the battery against the
    #: manifest, and the schedule against the priced design.
    contract = B.session_contract(rung, root,
                                  advancing_candidate=advancing_candidate)
    contract_hash = sha256_json(contract)
    cell = session_cell(rung, root)
    if contract["n_probes"] != cell["n_probes"]:
        raise D1BehaviouralIssuanceRefused(
            f"the schedule yields {contract['n_probes']} probes and the "
            f"priced {rung} cell funds {cell['n_probes']}; a different count "
            "is a different experiment at the same price")

    closure = behavioural_current_executable(root)

    auth = BG.D1BehaviouralAuthorization(
        authorization_id=f"autoinit.v1.phase_d1_behavioural_{rung}",
        granted_utc=granted_utc,
        granted_by=str(grant.get("granted_by") or "")[:4000],
        plan_id="autoinit.v1.phase_d1_behavioural",
        plan_hash=design["design_hash"],
        expected_usd=priced["expected_usd"],
        #: THE ATTEMPT's cap, which is the narrowed all-in figure rather than
        #: the design cell's. In the original one-use shape with nothing spent
        #: these coincide; after four subruns against a raised cumulative
        #: envelope they do not, and the smaller one is what was authorized.
        hard_cap_usd=bounds["all_in_hard_usd"],
        #: The REMAINDER, not the ceiling -- so a relaunch's watchdog bound is
        #: what is actually left of the envelope rather than what the whole
        #: session was once worth.
        per_launch_hard_usd=remaining,
        authorized_stages=AUTHORIZED_STAGES,
        stage_conditions=STAGE_CONDITIONS,
        scope_note=str(grant.get("covers") or "")[:4000],
        authorized_session_commit=session_commit,
        harness_source_digest=closure["digest"],
        harness_source_files=tuple(f["path"] for f in closure["files"]),
        provenance_commit=session_commit,
        action_policy=BG.D1_BEHAVIOURAL_POLICY,
        rung=rung,
        design_hash=design["design_hash"],
        contract_hash=contract_hash,
        battery_role=contract["battery"]["role"],
        battery_content_id=contract["battery"]["family_content_id"],
        recipe_id=contract["recovery_recipe"]["recipe_id"],
        n_probes=int(contract["n_probes"]),
        seeds=tuple(int(s) for s in contract["seeds"]),
        advancing_candidate=str(advancing_candidate or ""),
        run_id=str(run_id).strip(),
        money={**money, "derived_session": priced, "live_quote": quote,
               "attempt_bounds": bounds,
               "envelope": {"envelope_usd": envelope,
                            "envelope_source": envelope_source,
                            "consumed_usd": spent["consumed_usd"],
                            "remaining_usd": remaining},
               "four_conditions": "all four checked at the LIVE rate; see "
                                  "check_the_four_conditions"},
        rate_usd_per_hour=quote["usd_per_hour"],
        hard_runtime_minutes=bounds["hard_runtime_minutes"],
        gpu_hard_usd=bounds["gpu_hard_usd"],
        disk_hard_usd=bounds["disk_hard_usd"],
        all_in_hard_usd=bounds["all_in_hard_usd"],
        one_use=("ONE grant, ONE issuance, ONE launcher session. A failure is "
                 "this session's result; a retry is a new grant and a new "
                 "authorization."),
    )
    payload = auth.as_dict()
    payload["preregistration_sha256"] = prereg["preregistration_sha256"]
    payload["replay_plan_sha256"] = (
        prereg["arm_materialization"]["plan_sha256"])
    payload["authorizes"] = (
        f"ONE D1 behavioural {rung.upper()} session on the frozen design, "
        f"stages {'/'.join(AUTHORIZED_STAGES)}. It does NOT authorize a beam "
        "search, a re-selection of the candidate field, a re-measurement of "
        "B, a promotion outside the frozen decision rule, the other "
        "behavioural rung, D2, D3, or any repetition of a completed "
        "measurement.")
    payload["harness"] = {
        "n_files": closure["n_files"],
        "entry_points": list(closure["entry_points"]),
        "digest": closure["digest"],
        "_derived": ("live from the tree by aadistill.governance.closure, "
                     "never a hand-maintained list"),
    }
    #: The remainder, auditable in the artifact itself. A reader must be able
    #: to see WHY the per-launch bound is below the ceiling without re-summing
    #: the closeouts.
    payload["envelope"] = {
        "one_use_ceiling_usd": envelope,
        "rung_priced_ceiling_usd": ceiling,
        "consumed_by_prior_subruns_usd": spent["consumed_usd"],
        "remaining_usd": remaining,
        "prior_subruns": spent["subruns"],
        "all_reconciled_to_all_in": spent.get("all_reconciled"),
        "envelope_source": envelope_source,
        "attempt_bounds": bounds,
        "_rule": ("P12.1: the budget is cumulative across every resource and "
                  "subrun of the task; a rerun does not reset it. "
                  "per_launch_hard_usd carries the remainder."),
    }
    payload.pop("authorization_sha256", None)
    payload["authorization_sha256"] = sha256_json(payload)
    return payload


__all__ = ["AUTHORIZED_STAGES", "BEHAVIOURAL_DECLARED_INPUTS",
           "BEHAVIOURAL_ENTRY_POINTS", "DESIGN_REL",
           "D1BehaviouralIssuanceRefused", "GRANT_MAY_NOT_STATE", "PREREG_REL",
           "STAGE_CONDITIONS", "behavioural_current_executable",
           "TEARDOWN_RESERVE_MINUTES", "build_payload",
           "consumed_by_prior_subruns", "narrow_to_remaining",
           "preregistration", "reprice_rung_at", "session_cell"]
