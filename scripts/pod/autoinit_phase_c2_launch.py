#!/usr/bin/env python3
"""The Phase-C2 Search-1 session, as a specification. NOT AUTHORIZED.

    PYTHONPATH=src setsid nohup python -u scripts/pod/autoinit_phase_c2_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> < /dev/null &

This file declares WHAT the session is. How a session is run — detached start
with a durable descriptor, an independent provider-side watchdog, continuous log
relay, four budget thresholds, the artifact gate and provider-confirmed teardown
— lives once, in `aadistill.infrastructure.session_runner`, and is not
inherited, subclassed or retargeted by anybody.

**It cannot run today, and not because of a missing flag.** `SESSION_AUTH_PATH`
names an artifact that does not exist, and `C2Authorization.load` refuses
anything that is not a C2 grant under its own schema. There is no grant, no
readiness record, no bundle and no provider resource, so the only thing this
launcher can currently do is refuse.

What makes it a search session rather than a copy of another one:

* **Two stages and no training.** `arms=0`; there is no probe, no battery, no
  rung and no elimination, so none of that is declared and none of it can be
  reached.
* **Its own audit root and its own search workdir.** `audit/autoinit_phase_c2`
  and `autoinit/phase_c2_search`, named once here and once in the artifact spec,
  with a test holding the two equal — because Phase-B attempt 3 wrote
  `phase_b_search` while its specs named `phase_a_search`, the collector matched
  nothing, and a deadline failure came home with no per-state timings.
* **The evidence document is a `whole_file` relay spec** by virtue of being the
  `ArtifactPolicy` evidence file: the driver rewrites it on every state change,
  and the runner declares exactly that spec as rewritten-in-place.
* **A disk gate before the provider is contacted.** The search's peak working
  set is derived from the real state geometries, not guessed, and a session
  asking for less disk than the search needs is refused at `$0` rather than
  filling the volume at level 1 with 87 GiB of intermediates.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
#: `scripts` too: the experiment instances live under `experiments.` since the
#: core/application separation, and this file is also run as a subprocess with a
#: caller-set PYTHONPATH.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
#: The sibling science-input declarations. Present when this file is run
#: directly; absent when a test loads it by path, which is how the structural
#: checks load every launcher.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.infrastructure.budget import Phase  # noqa: E402
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, ExecutionCommands, LocalAsset, MarkerPolicy,
    SessionContext, SessionSpec, SessionSpecError, SetupManifest,
    TeardownPolicy,
)
from aadistill.infrastructure.session_runner import run_session  # noqa: E402
from experiments.deployment import POD_IMAGE, deployment_commands  # noqa: E402
from experiments.phase_c2.session import (  # noqa: E402
    C2_PLAN_ID, C2Authorization, c2_budget_spec, c2_hard_ceiling_usd,
    c2_plan_hash, c2_price_per_hour_usd,
)

WS = POD_IMAGE["workspace_root"]
REPO = POD_IMAGE["checkout_root"]

from autoinit_science_inputs import CALIBRATION_V1, CANONICAL_INIT  # noqa: E402

STATUS = f"{WS}/autoinit_phase_c2.status"
RUN_LOG = f"{WS}/autoinit_phase_c2_run.log"

#: The grant this session would consume. It does not exist: creating it is a
#: maintainer decision and is deliberately not part of this change.
AUTH_PATH = "logs/budget/approvals/autoinit_c2_authorization.json"

TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

#: Dev-box-only assets the pod cannot fetch from the relay.
#:
#: `reasoning_heavy_v2` travels this path and not the relay for the reason
#: Phase B recorded: it was built on the dev box at `$0` and has never been
#: uploaded, so declaring it as a `RelayInput` would name an object the pod
#: cannot fetch — which is exactly what killed Phase-A attempt 5 at $0.6426.
#: 780 KB over a measured 0.72 MB/s uplink is about a second.
#:
#: `state_eval_v1` is the suite every candidate is measured on. Without it there
#: is no ranking metric at all.
LOCAL_ASSETS = (
    LocalAsset("artifacts/stage1/reasoning_heavy_v2", "reasoning_heavy_v2",
               "artifacts/stage1"),
    LocalAsset("artifacts/stage1/state_eval_v1", "state_eval_v1",
               "artifacts/stage1"),
)

#: Ignored by the pod's blocking test gate. Held equal to the pod simulator's
#: list by a test, and kept SHORT: the gate has a `timeout 2700` and an exit-90
#: kill, so a slow pre-flight test here ends a paid session.
TEST_IGNORES = ("tests/data/test_recovery_corpus_pipeline.py",
                "tests/pod/test_phase_a_stages1_5_execute.py")

#: Derived, not guessed. Peak resident search states = the kept parents plus the
#: level being generated plus the leaves accumulated so far, over the real
#: geometries: 87.4 GiB, at level 1, where 5 retained level-0 states (still
#: teacher-width, ~6.8 GiB each) are expanded into 18 children. Plus the teacher
#: (7.5 GiB), the repo, the venv and the conditional baseline rebuild's four
#: intermediates. Provision covers that with headroom, because disk is cents and
#: a search that fills the volume at level 1 loses every state it has measured.
C2_PEAK_WORKING_GIB = 87.4
C2_PROVISION_GIB = 200


def storage_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Refuse an under-provisioned volume before a pod exists.

    Mechanically known rather than something to discover at level 1: the peak
    working set is a function of the configured space and the target geometry,
    both frozen.
    """
    requested = int(getattr(ctx.args, "disk_gb", 0) or 0)
    if requested < C2_PROVISION_GIB:
        return False, (
            f"--disk-gb {requested} is below the C2 provision of "
            f"{C2_PROVISION_GIB} GiB. The search's peak working set is "
            f"{C2_PEAK_WORKING_GIB:.1f} GiB of intermediate states, reached at "
            "level 1, plus the teacher, the checkout and the venv. A volume "
            "that fills there loses every state measured up to that point.")
    return True, (f"volume {requested} GiB >= {C2_PROVISION_GIB} for a "
                  f"{C2_PEAK_WORKING_GIB:.1f} GiB peak working set")


def pricing_identity_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The grant's ceiling must be the one the pricing record derives.

    `load_pricing` already refuses a record that does not match its own
    `pricing_sha256`. This is the other half: a grant written against a figure
    the record does not produce is a grant for a different session.
    """
    ceiling = c2_hard_ceiling_usd(REPO_ROOT)
    if abs(float(ctx.auth.hard_cap_usd) - ceiling) > 1e-9:
        return False, (
            f"the grant caps at ${ctx.auth.hard_cap_usd:.4f} and the pricing "
            f"record derives ${ceiling:.4f}. One of them is describing a "
            "different session; refusing to launch either.")
    return True, (f"grant cap ${ceiling:.4f} matches the pricing record, "
                  f"whose own sha256 verified")


def plan_identity_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The grant must bind the space this launcher is about to search.

    `c2_plan_hash` is derived from the session contract AND the configured
    space, so a grant issued against a different order policy, a different
    profile restriction or a different implementation set cannot authorize this
    run.
    """
    live = c2_plan_hash()
    if ctx.auth.plan_hash != live:
        return False, (
            f"the grant binds plan {str(ctx.auth.plan_hash)[:12]}… and the "
            f"configured space hashes to {live[:12]}…. The searched space has "
            "moved since the grant was issued; re-issue it or revert the space.")
    return True, f"grant binds the live plan {live[:12]}…"


def reserve_minutes(plan, name: str) -> float:
    """One named reserve from the plan. By NAME, because the names are the
    partition: summing them would hand the beam a clock the accounting says
    belongs to the baseline rebuild."""
    for reserve in plan.soft_stop_reserves:
        if reserve.name == name:
            return float(reserve.minutes)
    raise SessionSpecError(
        f"the plan declares no {name!r} reserve; the runtime partition cannot "
        f"be derived from it. Declared: "
        f"{[r.name for r in plan.soft_stop_reserves]}")


def beam_envelope_minutes(plan) -> float:
    """What the beam may spend: the DEPTH-early base plus its own risk reserve.

    **Not** the sum of all reserves. `baseline_rebuild_reserve` is excluded on
    purpose: the pricing record separates the two because they buy different
    things, and a deadline of `base + every reserve` let the beam run into the
    27.665 minutes held for a missing-B rebuild. Enforcement now matches the
    accounting, which is a partition repair and adds nothing to the ceiling.
    """
    base = next(p.minutes for p in plan.breakdown
                if p.name == "beam_search_depth_early")
    return base + reserve_minutes(plan, "beam_composition_risk")


def driver_command(ctx: SessionContext, plan) -> str:
    """The pod's command. Every figure comes from the plan, once, by name.

    Three minute numbers with three meanings:

    * `--search-deadline-minutes` bounds the beam at runtime, and is the beam's
      whole envelope — base plus `beam_composition_risk` and nothing else.
    * `--search-minutes` funds the affordability check taken before the beam
      starts. It is the SAME envelope, because approving the beam against its
      expected 300.16-minute trajectory would approve work its own deadline
      permits and the soft stop may not fund.
    * `--baseline-rebuild-minutes` is the conditional reserve, spent only if the
      deterministic rule finds B absent, on a clock that starts then.
    """
    beam = beam_envelope_minutes(plan)
    rebuild = reserve_minutes(plan, "baseline_rebuild_reserve")
    #: The interpreter is a DEPLOYMENT fact, from the same declaration the
    #: runner reads for the collector. Not `ctx.args`, which has no such field:
    #: the first version of this line said `ctx.args.remote_python` and would
    #: have raised AttributeError at driver start, on a billing pod, after setup
    #: had already succeeded.
    #: FLOORED, not rounded. `:.2f` rounds to nearest, so the plan's
    #: $14.499561 reached the driver as `14.50` — three hundredths of a cent
    #: MORE than the plan funds. A limit handed downward rounds DOWN or it is
    #: not a limit, which is the mirror of the rule a ceiling obeys when it is
    #: written: C1's record rounds its ceiling UP for the same reason.
    def floor2(value: float) -> str:
        return f"{math.floor(value * 100) / 100:.2f}"

    return (f"{POD_IMAGE['remote_python']} "
            f"scripts/pod/autoinit_phase_c2_driver.py --stage all "
            f"--image-digest '{ctx.image_digest}' "
            f"--rate {ctx.price} --spent-usd {ctx.spent_usd:.3f} "
            f"--soft-stop-usd {floor2(plan.soft_stop_usd)} "
            f"--authorized-usd {floor2(plan.hard_terminate_usd)} "
            f"--search-minutes {beam:.2f} "
            f"--search-deadline-minutes {beam:.2f} "
            f"--baseline-rebuild-minutes {rebuild:.3f} "
            f"--authorization-path {AUTH_PATH} "
            f"--device cuda")


def spec(args) -> SessionSpec:
    """The whole session, in one object. Nothing about it lives anywhere else."""
    budget = c2_budget_spec(REPO_ROOT)
    return SessionSpec(
        session_id="autoinit-phase-c2-search1",
        schema="aadistill.autoinit.c2_session/v1",
        description=(
            "Phase C2 Search-1: one beam search over four operator kinds, order "
            "free, ATTENTION fixed to activation_importance_v1 and branching "
            "over two calibration mixtures; plus, conditionally, one rebuild of "
            "the frozen C1 baseline B. Trains nothing."),
        authorization_path=AUTH_PATH,
        authorization_loader=C2Authorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/pod/watchdog.py",
            setup_script="scripts/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=C2_PLAN_ID,
        plan_hash=c2_plan_hash(),
        budget=budget,
        setup=SetupManifest(
            #: The canonical init is staged because `run_phase_a_search`
            #: injects it as the measured control, verified against its frozen
            #: single-file sha256. The domain-balanced mixture comes by relay
            #: because it is already there; the reasoning-heavy one is a local
            #: asset above.
            relay_inputs=(*CANONICAL_INIT, *CALIBRATION_V1),
            local_assets=LOCAL_ASSETS,
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "SESSION_RELAY_INPUTS",
                          "SESSION_KIND", "TEACHER_REVISION"),
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "ASSETS_READY", "TEACHER_READY",
                           "ROPE_OK", "TESTS_OK", "AUTHORIZATION_OK",
                           "SETUP_DONE"),
            #: DECLARED. An undeclared kind falls through to
            #: `SESSION_KIND=spend`, whose branch loads a
            #: `PreflightAuthorization` — and a branch that reads the artifact
            #: through the wrong type is worse than a missing one, because it
            #: can SUCCEED while binding this session to another phase's file
            #: list and price. Phase-B attempt 2 proved that at $0.2300, a
            #: KeyError one step after the pod's test gate passed.
            env={"SESSION_KIND": "c2"},
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION, test_ignores=TEST_IGNORES),
        driver_command=driver_command,
        driver_job_id="autoinit_phase_c2_driver",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("PHASE_C2_FAILED",),
            incomplete=(),
            #: Search-1 produces no checkpoint anybody fetches: its leaves are
            #: measured on the pod and their identities come home in the
            #: journal. So there are no products, and saying so is what stops a
            #: collector looking for weights this session never exports.
            products_eligible=lambda terminal, stages: False,
            failure_note=("a blocking stage failed — collecting evidence, then "
                          "tearing down. Nothing was trained and no permanent "
                          "artifact was replaced.")),
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_phase_c2",
            evidence_filename="c2_evidence.json",
            archive_basename="c2_search1_artifacts.tar.gz",
            spec_success="configs/autoinit/c2_artifacts.json",
            spec_failed="configs/autoinit/c2_artifacts_failed.json",
            report_names=("c2_evidence.json", "c2_search_summary.json")),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
        #: Both gates run before a pod exists. A grant whose ceiling or whose
        #: bound space disagrees with this tree is refused at $0.
        precheck=(storage_gate, pricing_identity_gate, plan_identity_gate),
        evidence_fields={
            "c2_plan_hash": c2_plan_hash(),
            "trains_anything": False,
            "peak_working_gib": C2_PEAK_WORKING_GIB,
            "search_is_not_a_recovery_result": (
                "the state_eval ranking is a hypothesis generator. A search "
                "winner is not a demonstrated initialization improvement, and "
                "behavioural confirmation is a separately authorized paid "
                "experiment that this launcher cannot reach."),
            "search2_reachable_from_this_launcher": False,
        })


def build_parser() -> argparse.ArgumentParser:
    """The real parser, extracted so a test can assert on the namespace it
    produces rather than on a transcription of it."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    #: From the pricing record, so there is one hand-maintained price in the
    #: repository and it is the one `pricing_identity_gate` verifies. A stale
    #: value here can only ever refuse a launch, never buy one.
    ap.add_argument("--max-price", type=float,
                    default=c2_price_per_hour_usd(REPO_ROOT))
    ap.add_argument("--disk-gb", type=int, default=C2_PROVISION_GIB)
    ap.add_argument("--token-src",
                    default=os.path.expanduser("~/.cache/huggingface/token"))
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=8)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    #: The hard-terminate envelope plus slack. A poll limit below the priced
    #: ceiling would stop watching a session that is still billing.
    ap.add_argument("--poll-limit-min", type=float, default=900.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    ap.add_argument("--out",
                    default="logs/stages/stage-1/phase_c2/runs/"
                            "autoinit_phase_c2_session.json")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return run_session(spec(args), args, REPO_ROOT,
                       summary=("Search-1 is a terminus: Search-2 is "
                                "conditional on this evidence and behavioural "
                                "confirmation is separately authorized."))


if __name__ == "__main__":
    raise SystemExit(main())
