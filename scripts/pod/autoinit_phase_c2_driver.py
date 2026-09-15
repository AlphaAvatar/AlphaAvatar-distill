#!/usr/bin/env python3
"""Phase C2 Search-1: bind the identities, run the search, resolve the baseline.

Two stages, because a session that trains nothing has no reason to have six.

**A. `bind_identities`** — everything checkable before anything expensive: the
image, the executable digest, the teacher revision, both calibration mixtures by
spec *and* content hash, the configured search space, and the frozen C1 baseline
construction. The last one is the cheapest gate in the session and protects the
most: if the baseline recipe is not the one the preregistration froze, the whole
comparison is against a different B, and it costs `$0` to learn that here
instead of after a 27-minute rebuild.

**B. `search_and_baseline`** — the beam search over the Search-1 space through
the same `run_phase_a_search` Phase A and Phase B ran, which carries the
durability boundary: the ranking is committed to a hash-bound artifact the
instant the expensive part succeeds, because Phase-B attempt 4 completed a
search, measured everything, ranked it, and then raised in a summary dict,
ending eight hours of completed science with no authoritative record of which
leaves had won.

Then the baseline, resolved exactly once by `BaselineFallback` and not by a
judgement call: the search re-derives B when the beam reaches it, and B is
rebuilt through the frozen C1 fixed path when it does not. Never both — their
identities collide by construction, which is the same fact that lets the search
re-derive B at all.

**This driver is standalone.** It does not subclass `PhaseADriver`, which owns
probe training, rungs, batteries, elimination and a six-stage plan. Inheriting
it would mean satisfying every contract that machinery REQUIRES in order to use
none of it, and reasoning from what a session needs rather than from what a base
class demands has cost this project two paid pods.

**Nothing experiment-specific reaches `src/aadistill`.** The geometry, the
teacher, the mixtures, the space, the prices and the baseline identities are all
in `experiments.phase_c2` and `experiments.phase_c1`; this file wires them to
generic machinery and adds no constant of its own.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.governance.authorization import AuthorizationError  # noqa: E402
from aadistill.initialization.planning.ranking import PARETO_V1  # noqa: E402
from aadistill.initialization.planning.search import SearchDeadlineExceeded  # noqa: E402
from experiments.phase_c2 import comparison as C  # noqa: E402

WS = Path("/workspace")
STATUS = WS / "autoinit_phase_c2.status"

#: C2 owns its own roots. The launcher's `ArtifactPolicy` collects
#: `audit/autoinit_phase_c2`, and evidence written anywhere the collector does
#: not walk is evidence the session loses at teardown.
AUDIT = REPO / "artifacts/audit/autoinit_phase_c2"
#: Its own search workdir, never Phase A's or Phase B's: the journals are
#: different runs and a shared path lets one overwrite the other's resume state.
#: A module constant rather than a literal inside the stage, because the
#: artifact specs that COLLECT this journal live in other files and attempt 3 of
#: Phase B proved they can disagree silently — the specs named `phase_a_search`
#: while the driver wrote `phase_b_search`, the collector matched nothing, and
#: the journal went with the pod at the one moment it mattered.
SEARCH_WORKDIR = REPO / "artifacts/autoinit/phase_c2_search"
STATE_EVAL = REPO / "artifacts/stage1/state_eval_v1"


def mark(name: str) -> None:
    line = f"{datetime.now(timezone.utc):%FT%TZ} MARKER:{name}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def say(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


class PhaseC2Driver:
    """The whole session, in two stages."""

    def __init__(self, a) -> None:
        from experiments.phase_c2.session import (
            C2_SESSION_CONTRACT, C2Authorization, c2_plan_hash,
        )

        self.a = a
        self.t0 = time.time()
        self.completed: list[str] = []
        self.baseline = None
        AUDIT.mkdir(parents=True, exist_ok=True)

        self.auth = C2Authorization.load(REPO / a.authorization_path)
        self.ev: dict = {
            "schema": "aadistill.autoinit.c2_evidence/v1",
            "_contract": (
                "The Phase-C2 Search-1 session record, REWRITTEN on every state "
                "change. It is relayed as a `whole_file` spec for exactly that "
                "reason: mirrored by byte offset, a rewritten document comes "
                "home as the head of an early version with the tail of the "
                "final one appended to it."),
            "session_id": C2_SESSION_CONTRACT.session_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "image_digest": a.image_digest,
            "rate_usd_per_hour": a.rate,
            "authorized_usd": a.authorized_usd,
            "soft_stop_usd": a.soft_stop_usd,
            "plan_hash": c2_plan_hash(),
            "session_contract_hash": C2_SESSION_CONTRACT.contract_hash,
            "authorization_id": self.auth.authorization_id,
            "trains_anything": False,
            "stages": {},
        }
        self.save()

    # -- bookkeeping -------------------------------------------------------
    def usd(self) -> float:
        return self.a.spent_usd + (time.time() - self.t0) / 3600.0 * self.a.rate

    def save(self) -> None:
        self.ev["elapsed_min"] = round((time.time() - self.t0) / 60, 2)
        self.ev["spend_usd"] = round(self.usd(), 4)
        self.ev["stages_completed"] = list(self.completed)
        (AUDIT / "c2_evidence.json").write_text(
            json.dumps(self.ev, indent=2, default=str) + "\n")

    def record(self, stage: str, passed: bool, detail, **extra) -> None:
        self.ev["stages"][stage] = {
            "stage_id": stage, "passed": passed,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "spend_usd": round(self.usd(), 4),
            "detail": detail, **extra}
        if passed:
            self.completed.append(stage)
            mark(f"STAGE_PASSED:{stage}")
        else:
            mark(f"STAGE_FAILED:{stage}")
        self.save()

    def afford(self, minutes: float, what: str) -> None:
        """Refuse to START work that would cross the soft stop.

        The check the overrunning session did not have: it re-priced against the
        AUTHORIZATION, so work that would finish just under the cap was allowed
        to start and left nothing for teardown.
        """
        projected = self.usd() + minutes / 60.0 * self.a.rate
        if projected > self.a.soft_stop_usd:
            raise AuthorizationError(
                f"{what} needs {minutes:.1f} min (${projected:.2f} projected) "
                f"and the soft stop is ${self.a.soft_stop_usd:.2f}. Refusing to "
                "start work that would leave nothing for collection.")

    # -- stage A -----------------------------------------------------------
    def bind_identities(self) -> bool:
        """Everything checkable before anything expensive."""
        from aadistill.initialization.calibration.profiles import get_profile
        from experiments.phase_c2 import baseline as B
        from experiments.phase_c2.search_space import (
            C2_ALLOWED_IMPLS, C2_IMPL_PROFILES, C2_PROFILE_IDS,
            TEACHER_GEOMETRY, register_c2_operators,
        )
        from experiments.phase_c2.session import c2_harness_digest
        from phase_a_frozen import SEARCH_SEED, TEACHER_ID, TEACHER_REVISION

        #: Explicit, and FIRST. `attention.activation_importance_v1` is not a
        #: shipped default, and `_allowed_impl_ids` validates `allowed_impls`
        #: against the registry — so an unregistered operator is a refusal, not
        #: a silent omission. C1 attempt 16 died at stage D on an empty adapter
        #: registry, which is the same mistake one layer down.
        register_c2_operators()

        harness = c2_harness_digest(REPO)
        mixtures = {}
        for qualified_id in C2_PROFILE_IDS:
            profile = get_profile(qualified_id)
            #: `resolve` verifies the items file against the profile's recorded
            #: token-content hash and raises if it drifted. Calling it here is
            #: what makes "the mixture the operators will see" a checked fact
            #: rather than a filename.
            items = profile.resolve(REPO)
            mixtures[qualified_id] = {
                "profile_hash": profile.profile_hash,
                "content_sha256": profile.content_sha256,
                "n_items": len(items),
                "materialized": profile.materialized,
            }
            say(f"  {qualified_id}: {len(items)} items, content "
                f"{str(profile.content_sha256)[:12]}…")

        #: The cheapest gate in the session, and the one that protects the most.
        frozen = B.frozen_baseline_spec(device=self.a.device)
        construction = B.assert_frozen_construction(frozen)
        say(f"  frozen baseline construction verified: "
            f"{frozen.spec_hash[:12]}… == {B.B_SPEC_HASH[:12]}…")

        detail = {
            "harness_digest": harness["digest"],
            "harness_files": len(harness["files"]),
            "teacher": {"repo_id": TEACHER_ID, "revision": TEACHER_REVISION},
            "search_seed": SEARCH_SEED,
            "target_geometry": dict(TEACHER_GEOMETRY),
            "space": {
                "allowed_impls": list(C2_ALLOWED_IMPLS),
                "impl_profiles": {k: list(v) for k, v
                                  in sorted(C2_IMPL_PROFILES.items())},
                "profiles": list(C2_PROFILE_IDS),
                "order": "free; no kind is pinned to a position",
            },
            "calibration": mixtures,
            "baseline": {
                "construction": construction,
                "frozen_spec_hash": B.B_SPEC_HASH,
                "expected_artifact_digest": B.B_ARTIFACT_DIGEST,
                "resolution": "decided in stage search_and_baseline",
            },
        }
        self.record("bind_identities", True, detail)
        return True

    # -- stage B -----------------------------------------------------------
    def search_and_baseline(self) -> bool:
        """The search, the durability boundary, and the baseline resolved once."""
        from aadistill.initialization.specs.arch import get_adapter
        from experiments.calibration import register_builtin_profiles
        from experiments.phase_c2 import baseline as B
        from experiments.phase_c2.search_space import (
            C2_ALLOWED_IMPLS, C2_IMPL_PROFILES, C2_PROFILE_IDS,
        )
        from aadistill.initialization.calibration.profiles import get_profile
        from aadistill.initialization.planning.search import Deadline
        from phase_a_search import as_operator_items, run_phase_a_search

        register_builtin_profiles()
        #: The FULL beam envelope, not the expected trajectory. Approving the
        #: beam against 300.16 min would approve work its own deadline permits
        #: — the base plus `beam_composition_risk` — and the soft stop may not
        #: fund. The launcher sends the same figure as the deadline for exactly
        #: that reason.
        self.afford(self.a.search_minutes, "the beam search (full envelope)")

        profiles = tuple(get_profile(q) for q in C2_PROFILE_IDS)
        #: Keyed by qualified id, because a loader that ignores its argument was
        #: the historical shape and is what let a run be LABELLED with one
        #: mixture and FED another.
        items = {p.qualified_id: as_operator_items(p.resolve(REPO))
                 for p in profiles}

        #: The beam's clock, and ONLY the beam's: the launcher derives this as
        #: the DEPTH-early base plus `beam_composition_risk` and deliberately
        #: excludes `baseline_rebuild_reserve`. Before this partition the
        #: deadline was base + every reserve, which let the beam run into the
        #: 27.665 minutes the accounting holds for a missing-B rebuild.
        deadline = Deadline.from_minutes(self.a.search_deadline_minutes)
        #: `root_loader` is supplied by the hook's caller, not here:
        #: `run_phase_a_search` owns the teacher's lifetime and passes the same
        #: object it gave the beam. Loading a second copy would double 8 GiB of
        #: VRAM at the one moment the search has finished and the rebuild is
        #: about to start, and could silently use a different revision than the
        #: search the baseline is being compared against.
        fallback = B.BaselineFallback(
            adapter=get_adapter("qwen3"),
            workdir=SEARCH_WORKDIR,
            repo_root=REPO,
            calibration_items=items,
            device=self.a.device,
            #: A NUMBER, not a `Deadline`. A `Deadline` starts counting when it
            #: is constructed, so handing the beam's clock to the rebuild would
            #: give it whatever the beam left — which after a full-envelope
            #: search is nothing. The fallback builds its own inside
            #: `rebuild()`, the first moment B is known to be absent.
            rebuild_minutes=self.a.baseline_rebuild_minutes,
            #: The same soft-stop check the beam used, re-taken against the
            #: CURRENT spend rather than against a projection made hours ago.
            afford=self.afford,
            say=say)

        found = run_phase_a_search(
            workdir=SEARCH_WORKDIR,
            state_eval=STATE_EVAL,
            top_n=self.a.top_n,
            device=self.a.device,
            repo_root=REPO,
            profiles=profiles,
            calibration_items=items,
            allowed_impls=C2_ALLOWED_IMPLS,
            impl_profiles=C2_IMPL_PROFILES,
            conditional_candidates=fallback,
            run_id="autoinit.v1.phase_c2.search1",
            purpose=("AutoInitializer Phase C2 Search-1: ATTENTION-aware "
                     "composition and order re-search, ATTENTION fixed"),
            search_minutes=self.a.search_deadline_minutes)

        self.baseline = fallback.outcome

        #: THE EVIDENCE BOUNDARY FOR B->C.
        #:
        #: `run_phase_a_search` commits the beam ranking before the baseline
        #: resolves, which is correct — the ranking must not wait on anything.
        #: But its summary serializes an imported candidate as identity and
        #: provenance only, so a REBUILT B's evaluation existed nowhere durable:
        #: the run could rebuild B, measure it correctly, and lose the one
        #: number the whole question is asked against at teardown.
        #:
        #: Written here, after both, and identically in both branches. It cites
        #: `stage1_selection.json` and never rewrites it.
        baseline_state = C.resolve_baseline_state(found=found,
                                                  outcome=self.baseline)
        comparison = C.build(
            baseline=baseline_state,
            baseline_outcome=self.baseline,
            candidates=found.top_n.selected,
            suite=found.result.config.suite,
            policy=PARETO_V1,
            run_id=found.summary["run_id"],
            config_hash=found.summary["config_hash"],
            selection_record=str(
                (SEARCH_WORKDIR / "stage1_selection.json").relative_to(REPO)),
            search_summary=str((AUDIT / "c2_search_summary.json").relative_to(REPO)))
        comparison_path = C.commit(comparison, AUDIT)
        say(f"  B->C comparison: {comparison['comparison']['verdict']} "
            f"({comparison['baseline']['resolution']} B, "
            f"{len(comparison['candidates'])} candidates) -> "
            f"{comparison_path.name}")

        summary = found.summary
        detail = {
            "run_id": summary["run_id"],
            "config_hash": summary["config_hash"],
            "suite": summary["suite"], "suite_hash": summary["suite_hash"],
            "calibration_profiles": summary["calibration_profiles"],
            "n_states": summary["summary"]["n_states"],
            "n_complete_leaves": summary["summary"]["n_complete_leaves"],
            "n_pruned": summary["summary"]["n_pruned"],
            "levels": [{"level": lv["level"],
                        "generated": len(lv["generated"]),
                        "leaves": len(lv["leaves"]),
                        "seconds": lv["seconds"]} for lv in summary["levels"]],
            "top_n": summary["top_n"],
            "control": summary["control"],
            "retained_candidates": summary["retained_candidates"],
            "baseline": self.baseline,
            "baseline_comparison": {
                "path": str(comparison_path.relative_to(REPO)),
                "schema": comparison["schema"],
                "record_sha256": comparison["record_sha256"],
                "resolution": comparison["baseline"]["resolution"],
                "verdict": comparison["comparison"]["verdict"],
                "n_candidates": len(comparison["candidates"]),
                "excluded_as_duplicate": comparison[
                    "excluded_as_duplicate_of_baseline"],
            },
            "_the_metric_is_not_the_result": (
                "the state_eval ranking is a hypothesis generator. A search "
                "winner is not a demonstrated initialization improvement and "
                "must not be reported as one; behavioural confirmation is a "
                "separately authorized paid experiment."),
        }
        (AUDIT / "c2_search_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n")
        self.record("search_and_baseline", True, detail)
        return True

    # -- the loop ----------------------------------------------------------
    def run(self) -> int:
        mark("DRIVER_START")
        stages = (("bind_identities", self.bind_identities),
                  ("search_and_baseline", self.search_and_baseline))
        for name, fn in stages:
            try:
                ok = fn()
            except (AuthorizationError, SearchDeadlineExceeded) as exc:
                #: A refusal whose message IS the explanation.
                self.record(name, False, f"refused: {exc}"[-2000:])
                ok = False
            except Exception as exc:                            # noqa: BLE001
                #: A defect. A defect without a frame has cost this project a
                #: paid session, so the traceback is part of the evidence.
                self.record(name, False, f"{type(exc).__name__}: {exc}"[-2000:],
                            traceback=traceback.format_exc()[-6000:])
                ok = False
            if not ok:
                #: Both stages are blocking. There is nothing after a failed
                #: search that could be a partial result.
                mark("PHASE_C2_FAILED")
                self.finish(False, failed=name)
                return 20 + len(self.completed)
        mark("ALL_DONE")
        say(f"Phase C2 Search-1 complete — ${self.usd():.2f}. STOP for review: "
            "the ranking is a hypothesis, not a recovery result, and no "
            "confirmation is reachable from this driver.")
        self.finish(True, failed=None)
        return 0

    def finish(self, success: bool, *, failed: str | None) -> None:
        self.ev["successful"] = success
        self.ev["failed_stage"] = failed
        self.ev["outcome"] = "SUCCESS" if success else "FAILED"
        self.ev["baseline"] = self.baseline
        self.ev["cleanup_is_not_success"] = (
            "collection and teardown run regardless of the outcome; a clean "
            "cleanup does not make a failed search a successful Search-1")
        self.ev["trained_anything"] = False
        self.ev["followon_reachable_from_this_driver"] = False
        self.save()


def build_parser() -> argparse.ArgumentParser:
    """The real parser, extracted so a test can assert on the namespace it
    produces rather than on a transcription of it."""
    ap = argparse.ArgumentParser(description=__doc__)
    #: There is no stage 2. Search-1 is a terminus: Search-2 is conditional on
    #: its evidence and behavioural confirmation is separately authorized.
    ap.add_argument("--stage", default="all", choices=("all",))
    ap.add_argument("--image-digest", required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--authorized-usd", type=float, required=True)
    #: Both REQUIRED with no defaults. The launcher is the single owner of both
    #: numbers; a default here would be a second copy of a pricing constant that
    #: could drift from the one that books the money.
    ap.add_argument("--search-minutes", type=float, required=True,
                    help="base search allowance; funds the affordability check")
    ap.add_argument("--search-deadline-minutes", type=float, required=True,
                    help="the BEAM's whole envelope: the DEPTH-early base plus "
                         "beam_composition_risk, and nothing else. It excludes "
                         "baseline_rebuild_reserve, which is not the beam's to "
                         "spend")
    ap.add_argument("--baseline-rebuild-minutes", type=float, required=True,
                    help="the conditional baseline_rebuild_reserve, spent only "
                         "if the deterministic rule finds B absent, on a clock "
                         "that starts then")
    #: REQUIRED, with no default. It defaulted to a repository-level path that
    #: no longer exists anywhere: the authorization is owned by the run, so
    #: there is nothing sensible to fall back to, and a default naming an absent
    #: file turns "you did not say which run" into "file not found". The
    #: launcher always passes it, derived from `--run-id`.
    ap.add_argument("--authorization-path", required=True,
                    help="this run's authorization, e.g. logs/stages/stage-1/"
                         "phase_c2/runs/<attempt>/governance/authorization.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-n", type=int, default=5)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    driver = PhaseC2Driver(args)
    driver.auth.require_within_cap(args.authorized_usd,
                                   what="session backstop")
    return driver.run()


if __name__ == "__main__":
    raise SystemExit(main())
