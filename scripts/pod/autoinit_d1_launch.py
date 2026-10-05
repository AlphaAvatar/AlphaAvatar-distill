#!/usr/bin/env python3
"""The D1 formal SEARCH launcher. One session, declared completely.

    PYTHONPATH=src:scripts python scripts/pod/autoinit_d1_launch.py \
        --run-id d1_search_001 --gpu "NVIDIA L40S" --arm supervised_target

Everything a previous generation of launcher expressed by overriding a hook is a
FIELD of the `SessionSpec` the shared `SessionRunner` consumes: the authorization
type and path, the setup manifest, the driver command, the budget, the teardown and
artifact policies. The runner owns acquisition, the watchdog, teardown on every exit
path, and collect-before-teardown; this file owns what makes the session D1's.

`SESSION_KIND = "d1"` is DECLARED, and the matching branch in
`autoinit_preflight_setup.sh` loads a `D1Authorization` and asserts its typed
permissions. A missing dispatch entry is not a type error — it falls through to
`spend`, loads the generic artifact, and Phase B's attempt 2 proved what that costs.

**D1 is the first of these sessions that actually runs a beam search.** The
dispatch branch therefore asserts `allows_beam_search is True` where every other
dedicated branch asserts it False, and `allows_recovery`/`allows_behavioural` False:
the search commits a candidate set and stops, because neither behavioural rung can
be bound until that set exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, ExecutionCommands, MarkerPolicy, SessionSpec,
    SetupManifest, TeardownPolicy,
)
from experiments.deployment import deployment_commands  # noqa: E402
from experiments.phase_d1 import d1_session as D1S  # noqa: E402
from experiments.phase_d1.d1_authorization import D1Authorization  # noqa: E402

#: DECLARED, and the shell has a branch for it. See the module docstring.
SESSION_KIND = "d1"

#: THE POD'S BLOCKING TEST GATE runs D1's OWN suite, positively declared. The
#: 2026-10-03 boundary decision made `tests/` reusable-framework behaviour only, so
#: a session that declares nothing gets the framework checked on its machine and no
#: experiment's records. D1 wants its execution contract checked on the billing
#: machine, because that is where a staging difference would show.
TEST_PATHS = ("scripts/experiments/stage-1/phase_d1/tests",)

REPO = "/workspace/aad"


def auth_path_for(run_id: str | None) -> str:
    """Where this run's one-use authorization lives. Per run, never shared."""
    rid = run_id or "unrecorded"
    return (f"logs/stages/stage-1/phase_d1/runs/{rid}/governance/"
            "authorization.json")


def driver_command(ctx: Any, plan: Any) -> str:
    """The exact command the pod runs. A test parses it with the driver's parser.

    C1 attempt 7 cleared the pod test gate for the first time and then died at
    $0.4231 because its launcher emitted a flag the driver's parser has no option
    for: argparse exited 2 before a line of the driver ran.
    """
    arm = getattr(ctx.args, "arm", D1S.TREATMENT_ARM)
    deadline = int(max(60.0, float(plan.soft_stop_seconds))) \
        if getattr(plan, "soft_stop_seconds", None) else 0
    out = f"{REPO}/artifacts/stage1/d1/{getattr(ctx.args, 'run_id', 'unrecorded')}"
    return (f"/opt/train/bin/python {REPO}/scripts/pod/autoinit_d1_driver.py "
            f"--out '{out}' "
            f"--authorization '{REPO}/{auth_path_for(getattr(ctx.args, 'run_id', None))}' "
            f"--arm '{arm}' "
            f"--device cuda"
            + (f" --deadline-s {deadline}" if deadline else ""))


def budget_spec(repo_root: Path) -> BudgetSpec:
    """The `BudgetSpec` the runner plans from, from the SAME ingredients the
    pricing record uses.

    `BudgetSpec` is a PHASE-TIME model, not a dollar cap: the runner derives
    minutes from it and the money comes from the authorization's ceiling. So the
    phases here are the ones `search_cost` already decomposes -- the session
    overhead and the structural search minutes -- read from that priced session
    rather than decomposed a second time. Two decompositions of one cost have to
    be kept agreeing by hand, and this project has paid for that twice.

    `arms=0`: a search trains nothing, so the step term multiplies out. The
    measured floor is passed only so the below-floor guard cannot fire on a figure
    that means nothing here.
    """
    from aadistill.infrastructure.budget import MEASURED_STEP_SECONDS, Phase

    from experiments.phase_d1 import d1_authorization as A

    priced = A.session_ceiling(repo_root)
    session = _priced_session(repo_root)
    overhead = float(session["session_overhead_minutes"])
    expected_search = max(
        0.0, float(session["expected_minutes"]) - overhead)
    worst_search = max(
        0.0, float(session["structural_minutes_max"]))

    return BudgetSpec(
        arms=0, steps_per_arm=0,
        step_seconds=MEASURED_STEP_SECONDS,
        step_source=("unused: a D1 search session trains nothing, so arms=0 and "
                     "the step term is zero. The measured floor is passed so the "
                     "below-floor guard cannot be satisfied by accident"),
        #: The session overhead the cost model already prices, split the way the
        #: runner expects rather than invented: setup and the bundle transfer are
        #: inside `session_overhead_minutes`, so they are taken from it.
        setup_minutes=round(overhead * 0.75, 2),
        transfer_minutes=round(overhead * 0.25, 2),
        other_phases=(
            Phase("beam_search_expected_trajectory", round(expected_search, 2)),
        ),
        contingency_fraction=0.10,
        #: The difference between the expected trajectory and the structural worst
        #: case, NAMED rather than folded in: an identified bounded risk that is
        #: not on the expected path belongs after the contingency multiplier, so it
        #: protects the work instead of moving the watchdog's kill time.
        soft_stop_reserves=(
            Phase("beam_composition_risk",
                  round(worst_search - expected_search, 2)),),
        artifact_recovery_reserve_minutes=30.0,
        below_floor_reason=(
            f"priced at ${priced['hard_ceiling_usd']:.4f} / "
            f"{priced['hard_ceiling_minutes']:.2f} min by "
            "search_space.search_cost; see d1_design.json :: "
            "budget.chain.sessions.search"),
    )


def _priced_session(repo_root: Path) -> dict[str, Any]:
    """The priced search session, measured basis when one exists.

    One owner: the same `topk_search_cost()` the design's chain and
    `search_stage.cost` both read. A launcher that priced it again could disagree
    with the artifact it is about to run under.
    """
    import write_d1_design as w

    priced = w.topk_search_cost()
    if priced is not None:
        return priced["search_session"]
    from experiments.phase_d1 import search_space as d1

    w._ensure_the_frozen_operators_are_registered()
    return d1.search_cost()


def spec(args) -> SessionSpec:
    return SessionSpec(
        session_id="autoinit-d1-search",
        schema="aadistill.autoinit.d1_search_session/v1",
        description=(
            "D1: ONE formal beam search over the frozen four-operator space at "
            "width 6 with one warmup level, under reference_topk_tail_v1 at "
            "K=200, bsz=3 and length_sorted_v1, on the arm's declared position "
            "policy. Commits a candidate set and stops: it trains nothing, "
            "evaluates no behaviour and computes no promotion decision."),
        authorization_path=auth_path_for(getattr(args, "run_id", None)),
        authorization_loader=D1Authorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/pod/watchdog.py",
            setup_script="scripts/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/pod/collect_artifacts.py",
            #: THE DEPLOYMENT'S OWN IMAGE FACTS, read from the shared helper
            #: rather than retyped: the remote interpreter, the workspace and
            #: checkout roots and the minimum CUDA version are properties of the
            #: image, and a launcher that states its own can disagree with the
            #: image it then asks for.
            **deployment_commands()),
        plan_id=f"autoinit.v1.{D1S.EXPERIMENT_ID}",
        #: THE DESIGN HASH is the plan hash: the dispatch branch calls
        #: `require_plan` with it, so an authorization issued against another
        #: design revision is refused at exit 98 before any work.
        plan_hash=D1S.design_hash(REPO_ROOT),
        budget=budget_spec(REPO_ROOT),
        setup=SetupManifest(
            env={"SESSION_KIND": SESSION_KIND},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_KIND"),
            setup_markers=("ENV_READY", "REPO_READY", "TRAIN_ENV", "TESTS_OK",
                           "AUTHORIZATION_OK", "SETUP_DONE"),
            test_paths=TEST_PATHS,
            uv_max_seconds=getattr(args, "uv_max_s", 1500),
            tests_max_seconds=getattr(args, "tests_max_s", 2700)),
        driver_command=driver_command,
        driver_job_id=D1S.DRIVER_JOB_ID,
        status_path=D1S.STATUS_PATH,
        run_log_path=D1S.RUN_LOG_PATH,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("RUN_FAILED",),
            #: WHEN THIS SESSION'S PRODUCTS EXIST. The driver writes
            #: `d1_search.json` on EVERY path out, including a stage-A refusal, so
            #: eligibility is broader than success -- the distinction that cost
            #: $2.82 of verified checkpoints on 2026-08-13, when a collector ran
            #: only on a clean terminal state.
            products_eligible=lambda terminal, stages: True,
            failure_note=("the search stopped -- collecting its evidence, then "
                          "tearing down")),
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_d1",
            evidence_filename="d1_search.json",
            archive_basename="d1_search_artifacts.tar.gz",
            #: JSON AND LOGS ONLY. Attempt a1 of another session pulled
            #: `/workspace/out` wholesale: 9.1 GiB of checkpoints over a
            #: 0.72 MB/s uplink with the pod still billing. Nothing on the dev
            #: box consumes search checkpoints; the decision reads scores,
            #: digests and identities, all of which are JSON.
            spec_success="configs/autoinit/d1_search_artifacts.json",
            #: NOTHING IS `required` ON THE FAILURE PATH. An early refusal may
            #: legitimately have produced none of it, and demanding an artifact
            #: there would block the collection of what the run DOES have.
            spec_failed="configs/autoinit/d1_search_artifacts_failed.json",
            report_names=("d1_search.json", "journal.jsonl")),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
    )


def build_parser() -> argparse.ArgumentParser:
    """The real parser, at module scope so a test can build it.

    `tests/integration/test_session_kind_dispatch.py` enumerates every
    `*_launch.py` that exports `spec` and `build_parser`, fills the required
    arguments and builds the spec, to check each declared `SESSION_KIND` has a
    shell branch that loads ITS OWN authorization type. A launcher whose parser
    lives inside `main` is invisible to that probe — and an unseen launcher is one
    whose missing dispatch entry falls through to `spend`.

    Every required argument is a STRING: the probe fills them with one, and a
    required `type=float` would make it crash rather than report.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    #: THE SHARED SET `SessionRunner` ITSELF READS. Enumerated from the runner's
    #: reads rather than copied from another launcher: a copy keeps the flags that
    #: launcher needs and misses the ones this runner does.
    ap.add_argument("--scr", required=True,
                    help="the session's local scratch directory; "
                         "`SessionRunner` writes its working files there")
    ap.add_argument("--session-commit", required=True,
                    help="the commit the pod checks out. The authorization binds "
                         "it and the lineage gate verifies it")
    ap.add_argument("--bundle", required=True,
                    help="must be the canonical name derived from "
                         "--session-commit; an alias fails at $0")
    ap.add_argument("--runpod-config",
                    default=str(Path("~/.runpod/config.toml").expanduser()))
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--arm", default=D1S.TREATMENT_ARM, choices=list(D1S.ARMS))
    #: NOT `required`, and the reason is mechanical. The dispatch probe fills every
    #: required option with the STRING "kind_probe", so a required `type=float`
    #: raises during conversion, puts this launcher in UNPARSEABLE and removes it
    #: from every check in that module -- fewer tests, all green. A3 was dropped
    #: from all of them exactly this way. The default is the rate the ceiling was
    #: derived at, which is also the better governance: a launch must not be priced
    #: at a rate nothing authorized.
    ap.add_argument("--max-price", type=float, default=None,
                    help="omit to use the rate the derived ceiling was priced at")
    ap.add_argument("--disk-gb", type=int, default=None)
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and validate the spec; create nothing")
    return ap


def main(argv: list[str] | None = None) -> int:
    from aadistill.infrastructure.session_runner import SessionRunner

    args = build_parser().parse_args(argv)
    session = spec(args)
    if args.dry_run:
        print(f"session_id     {session.session_id}")
        print(f"plan_hash      {session.plan_hash[:24]}")
        print(f"authorization  {session.authorization_path}")
        #: The MONEY is the authorization's, not the BudgetSpec's: the spec is a
        #: phase-TIME model the runner plans minutes from, and the dollar ceiling
        #: comes from the one-use artifact. Printing a `hard_cap_usd` off the spec
        #: would be inventing a second price.
        from experiments.phase_d1 import d1_authorization as A
        priced = A.session_ceiling(REPO_ROOT)
        print(f"priced ceiling ${priced['hard_ceiling_usd']:.4f} "
              f"({priced['hard_ceiling_minutes']:.2f} min)")
        print(f"priced expected ${priced['expected_usd']:.4f}")
        print(f"pricing basis  {priced['_basis']}")
        print(f"SESSION_KIND   {session.setup.env['SESSION_KIND']}")
        print(f"test_paths     {list(session.setup.test_paths)}")
        print("CREATED NOTHING: --dry-run validates the declaration only.")
        return 0
    return SessionRunner(session, args).run()


if __name__ == "__main__":
    raise SystemExit(main())
