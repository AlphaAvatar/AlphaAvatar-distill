#!/usr/bin/env python3
"""The Phase-C2 FULL JOINT re-search: search, commit a Top-K, and STOP.

    PYTHONPATH=src:scripts python scripts/pod/autoinit_phase_c2_full_search_driver.py \
        --authorization-path <path> --rate 1.09 --authorized-usd ... [--device cuda]

Three stages and no fourth:

    A  bind_identities      nothing expensive; every identity checked first
    B  full_joint_search    the beam over the derived joint space
    C  commit_top_k         freeze the candidate set, then STOP

**It trains nothing and it measures no behaviour.** This is an
initialization-search stage: it generates states and ranks them on the frozen
cheap `state_eval` metrics. It performs no `0.86M` recovery and produces no
`correct_overall`, so nothing it emits can promote a candidate. The behavioural
screening and confirmation that decide the C2 incumbent are a SEPARATE session
under a SEPARATE authorization, and this driver deliberately has no code path
into them.

**Why a second driver rather than a flag on the first.**
`autoinit_phase_c2_driver.py` executes the frozen Search-1 space and couples the
search to a conditional baseline rebuild, because Search-1 had to resolve B when
the beam did not reach it. That coupling is Search-1's, its records are frozen
evidence, and B is now measured — so the full joint search needs neither the
restriction nor the fallback. What the two share is the search itself:
`run_phase_a_search` already takes the space as parameters, and this driver is a
caller of it rather than a copy.

**The space is DERIVED, not declared here.** `full_search_space.full_joint_space()`
enumerates it from the live registry; this file contains no operator list, no
profile list, no leaf count and no geometry.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("AAD_REPO", "/workspace/aad"))
if not (REPO / "src").is_dir():                     # local / toy execution
    REPO = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from aadistill.governance.authorization import AuthorizationError  # noqa: E402
#: The calibration mixtures, registered AT MODULE IMPORT and not inside the
#: stage that needs them. `get_profile` raises on an empty registry, and this
#: project has already lost a paid pod at stage D to exactly that class:
#: registration being explicit is correct, assuming somebody else did it is not.
#: Registering here rather than in `bind_identities` also means stage A cannot
#: be the thing that discovers an empty registry.
from experiments.calibration import register_builtin_profiles  # noqa: E402

register_builtin_profiles()

WS = Path(os.environ.get("AAD_WORKSPACE", "/workspace"))
if not WS.is_dir():
    WS = REPO / ".local"
STATUS = WS / "autoinit_phase_c2_full_search.status"
AUDIT = REPO / "artifacts/audit/autoinit_phase_c2_full_search"
SEARCH_WORKDIR = REPO / "artifacts/autoinit/phase_c2_full_search"
STATE_EVAL = REPO / "artifacts/stage1/state_eval_v1"

SUCCESS_MARKER = "C2_FULL_SEARCH_ALL_DONE"
FAILURE_MARKER = "C2_FULL_SEARCH_FAILED"

#: The run identity the search records into its own artifacts. Distinct from
#: Search-1's, so a reader can never confuse the two searches' journals.
RUN_ID = "autoinit.v1.phase_c2.full_search"
PURPOSE = ("AutoInitializer Phase C2 full joint re-search: implementations, "
           "applicable calibration profiles and operator order all competing in "
           "one beam, with the promoted ATTENTION operator in the library")


def _relative(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise.

    `Path.relative_to` RAISES on a path outside the repository, and a driver
    that crashes while recording where it put its own output is a driver that
    turns a success into a failure at the last stage. This project already
    carries that exact crash as open debt in one issuer; the toy execution of
    this driver reproduced it here, which is the reason the test runs the real
    stages instead of asserting about them.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def mark(name: str) -> None:
    line = f"{datetime.now(timezone.utc):%FT%TZ} MARKER:{name}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def say(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


class FullSearchDriver:
    """The whole session, in three stages, ending at a frozen candidate set."""

    def __init__(self, a) -> None:
        self.a = a
        self.t0 = time.time()
        self.completed: list[str] = []
        self.search = None
        AUDIT.mkdir(parents=True, exist_ok=True)

        self.ev: dict = {
            "schema": "aadistill.autoinit.c2_full_search_evidence/v1",
            "_contract": (
                "The Phase-C2 full-joint-re-search session record, REWRITTEN on "
                "every state change. It is relayed as a `whole_file` spec for "
                "exactly that reason: mirrored by byte offset, a rewritten "
                "document comes home as the head of an early version with the "
                "tail of the final one appended."),
            "run_id": RUN_ID,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "image_digest": a.image_digest,
            "rate_usd_per_hour": a.rate,
            "authorized_usd": a.authorized_usd,
            "soft_stop_usd": a.soft_stop_usd,
            "trains_anything": False,
            "measures_behaviour": False,
            "runs_recovery_probes": False,
            "names_an_incumbent": False,
            "_what_those_flags_mean": (
                "this session is an initialization search. It ranks states on "
                "the frozen cheap state_eval metrics and commits a candidate "
                "SET. It performs no 0.86M recovery, measures no "
                "correct_overall, and has no code path that could promote "
                "anything. The behavioural stages are separately authorized."),
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
        (AUDIT / "c2_full_search_evidence.json").write_text(
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
        """Refuse to START work that would cross the soft stop."""
        projected = self.usd() + minutes / 60.0 * self.a.rate
        if projected > self.a.soft_stop_usd:
            raise AuthorizationError(
                f"{what} needs {minutes:.1f} min (${projected:.2f} projected) "
                f"and the soft stop is ${self.a.soft_stop_usd:.2f}. Refusing to "
                "start work that would leave nothing for collection.")

    # -- stage A -----------------------------------------------------------
    def bind_identities(self) -> bool:
        """Everything checkable before anything expensive. Loads no model."""
        from aadistill.initialization.calibration.profiles import get_profile
        from experiments.phase_c2 import full_search_space as FS
        from phase_a_frozen import SEARCH_SEED, TEACHER_ID, TEACHER_REVISION

        FS.register_c2_operators()
        space = FS.full_joint_space(REPO)
        size = FS.size_report(REPO)
        cost = FS.cost_model(REPO)

        #: Every branch profile must be MATERIALIZED before the beam starts. A
        #: declared-but-unbuilt mixture is a search branch that cannot resolve,
        #: and discovering that hours in is discovering it at the worst moment.
        profiles = tuple(get_profile(q) for q in FS.PROFILE_IDS)
        unbuilt = [p.qualified_id for p in profiles if not p.materialized]
        if unbuilt:
            raise AuthorizationError(
                f"{unbuilt} are declared but not materialized, so they cannot "
                "be search branches. The joint space branches EVERY "
                "calibration-consuming operator over every active profile, so "
                "an unbuilt profile is not a partial loss — it is a space this "
                "session cannot search.")

        detail = {
            "space": {
                "allowed_impls": list(space.allowed_impls),
                "profile_ids": list(space.profile_ids),
                "impl_profiles": space.impl_profiles,
                "required_fields": list(space.required_fields),
                "total_leaves": size["full_joint"]["total_leaves"],
                "decomposed_leaves": size["full_joint"]["decomposed_leaves"],
                "exclusions": FS.EXCLUSIONS,
            },
            "_impl_profiles_is_none": (
                "every operator that consumes calibration branches over every "
                "active profile. Search-1 pinned three of four; that is the "
                "restriction this session reopens."),
            "cost_model_source": cost.source,
            "unmeasured_cost_inputs": list(cost.unmeasured),
            "teacher": {"repo_id": TEACHER_ID, "revision": TEACHER_REVISION},
            "search_seed": SEARCH_SEED,
            "top_n": self.a.top_n,
            "state_eval_root": str(STATE_EVAL),
            "profiles_materialized": [p.qualified_id for p in profiles],
        }
        self.record("bind_identities", True, detail)
        say(f"space: {detail['space']['total_leaves']} reachable leaves "
            f"({detail['space']['decomposed_leaves']} decomposed) over "
            f"{len(space.allowed_impls)} implementations, profiles unpinned")
        return True

    # -- stage B -----------------------------------------------------------
    def full_joint_search(self) -> bool:
        """The beam over the joint space. No baseline fallback, no comparison."""
        from aadistill.initialization.calibration.profiles import get_profile
        from aadistill.initialization.planning.search import Deadline
        from experiments.phase_c2 import full_search_space as FS
        from phase_a_search import as_operator_items, run_phase_a_search

        #: The FULL envelope, not the expected trajectory: approving the beam
        #: against its expected cost would approve work its own deadline permits.
        self.afford(self.a.search_minutes, "the full joint beam (full envelope)")

        space = FS.full_joint_space(REPO)
        profiles = tuple(get_profile(q) for q in FS.PROFILE_IDS)
        #: Keyed by qualified id, because a loader that ignores its argument is
        #: what let a run be LABELLED with one mixture and FED another.
        items = {p.qualified_id: as_operator_items(p.resolve(REPO))
                 for p in profiles}

        self.search = run_phase_a_search(
            workdir=SEARCH_WORKDIR,
            state_eval=STATE_EVAL,
            top_n=self.a.top_n,
            device=self.a.device,
            repo_root=REPO,
            profiles=profiles,
            calibration_items=items,
            allowed_impls=space.allowed_impls,
            #: NONE. The whole point of the re-search.
            impl_profiles=None,
            #: No conditional candidate. Search-1 needed a baseline fallback
            #: because B had to be resolved when the beam missed it; B is now
            #: measured and frozen, and this session compares nothing.
            conditional_candidates=None,
            run_id=RUN_ID,
            purpose=PURPOSE,
            search_minutes=self.a.search_deadline_minutes,
            #: SEARCH ONLY. This session compares nothing -- no conditional
            #: candidate, no baseline, no control -- so it must not trigger the
            #: canonical-control injection, which resolves a 0.6B checkpoint it
            #: deliberately does not stage. Attempt 3 ran the complete beam,
            #: committed its Top-5 and then died there, losing five
            #: checkpoints that existed on the pod at the time.
            include_canonical_control=False,
        )
        deadline = Deadline.from_minutes(self.a.search_deadline_minutes)
        self.record("full_joint_search", True, {
            "run_id": RUN_ID,
            "deadline_minutes": self.a.search_deadline_minutes,
            "deadline_expired": deadline.expired(),
            "_deadline_is_the_beams_only_clock": (
                "derived by the launcher as the costly-early base plus the "
                "beam-composition reserve. There is no baseline rebuild reserve "
                "to exclude, because this session rebuilds nothing."),
        })
        return True

    # -- stage C -----------------------------------------------------------
    def commit_top_k(self) -> bool:
        """Freeze the candidate set. This is the terminus.

        The set is what a LATER, separately authorized behavioural session
        reads. It is committed here, before anything behavioural exists, which
        is what makes it a preregistered set rather than one that could grow
        once probe results are visible.
        """
        selection = SEARCH_WORKDIR / "stage1_selection.json"
        if not selection.is_file():
            raise AuthorizationError(
                f"{selection} was not written, so there is no candidate set to "
                "freeze. The search must commit its ranking before this stage.")
        doc = json.loads(selection.read_text())
        selected = list(doc.get("selected") or ())
        if len(selected) != self.a.top_n:
            raise AuthorizationError(
                f"the selection holds {len(selected)} candidates and {self.a.top_n} "
                "were asked for. A candidate set of the wrong size is not the "
                "preregistered set.")

        self.record("commit_top_k", True, {
            "selection_artifact": _relative(selection),
            "selection_sha256": doc.get("selection_sha256"),
            "n_selected": len(selected),
            "state_ids": [s.get("state_id") for s in selected]
                         if isinstance(selected[0], dict) else selected,
            "search_config_hash": (doc.get("search") or {}).get("config_hash"),
            "_this_is_the_terminus": (
                "the session ends here. The candidate set is cheap-metric "
                "SELECTION evidence and promotes nothing; the behavioural "
                "screening and confirmation that decide the C2 incumbent are a "
                "separate session under a separate authorization."),
        })
        return True

    # -- driving -----------------------------------------------------------
    def run(self) -> int:
        stages = (("bind_identities", self.bind_identities),
                  ("full_joint_search", self.full_joint_search),
                  ("commit_top_k", self.commit_top_k))
        for name, fn in stages:
            say(f"stage {name}")
            try:
                ok = fn()
            except Exception as exc:                            # noqa: BLE001
                #: Broad on purpose: a paid session must record WHY it stopped,
                #: and an uncaught traceback on a pod is evidence nobody can
                #: read from the artifacts that come home.
                self.record(name, False, {"error": f"{type(exc).__name__}: {exc}"})
                self.finish(False, failed=name)
                return 1
            if not ok:
                self.finish(False, failed=name)
                return 1
        self.finish(True, failed=None)
        return 0

    def finish(self, success: bool, *, failed: str | None) -> None:
        self.ev["successful"] = success
        self.ev["failed_stage"] = failed
        self.ev["outcome"] = "ALL_DONE" if success else "FAILED"
        self.ev["followon_reachable_from_this_driver"] = False
        self.ev["_no_followon"] = (
            "there is no behavioural stage in this driver. Screening and "
            "confirmation require a separate authorization and a separate "
            "session; combining them would let one approval buy both a search "
            "and a promotion decision.")
        self.save()
        mark(SUCCESS_MARKER if success else FAILURE_MARKER)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authorization-path", required=True)
    ap.add_argument("--image-digest", default="")
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--authorized-usd", type=float, required=True)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    #: The beam's envelope and its clock, both supplied by the launcher from the
    #: pricing record. Neither is defaulted here: a driver that invents its own
    #: budget is a driver that can outrun the plan that funded it.
    ap.add_argument("--search-minutes", type=float, required=True)
    ap.add_argument("--search-deadline-minutes", type=float, required=True)
    ap.add_argument("--top-n", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    return ap


def main() -> int:
    return FullSearchDriver(build_parser().parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
