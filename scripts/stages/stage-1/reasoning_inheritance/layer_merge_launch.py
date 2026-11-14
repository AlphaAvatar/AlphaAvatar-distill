#!/usr/bin/env python3
"""The layer-merge round's sessions (study RI-LM1): grant, one-use artifacts, launch, closeout (dev box).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/reasoning_inheritance/layer_merge_launch.py \\
        {issue|launch|closeout|position|grant|stage-resume|archive-evidence} ...

`shared.study_engineering` is the mechanism; this module is the instance. RI-S1 is CLOSED and its unspent allowance
lapsed: this study has its own proposal (``logs/stages/stage-1/reasoning_inheritance/plans/layer_merge_proposal.json``,
``status: proposed``) and, only once the maintainer grants it, its own grant
(``logs/budget/approvals/stage1_layer_merge_grant.json``). Its envelope counts only its own runs (`SE.spent` filters by
plan id). Dollar amounts live in the proposal / grant, never here.

Beyond the mechanism's gates, two of this round's own, both read at $0 before the create:

* `the_teardown_reserve_stays` -- the proposal's ``teardown_reserve_usd`` is held outside every session's hard cap: the
  artifact's cap plus the reserve must fit what the envelope, the project cap and the provider balance leave;
* `the_relay_has_room` -- the private relay would accept the two candidates and the statistics at their true sizes
  (`aadistill.runtime.hub_capacity.would_accept`; nothing is uploaded).

Finished work leaves the pod as RI-S1's did (`launch.secure_incremental` on every poll; the candidates and statistics
through the relay from the pod). A later attempt resumes from `stage-resume`'s archive.
"""
from __future__ import annotations

import functools
import json
import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from shared import study_engineering as SE  # noqa: E402
from stages.reasoning_inheritance import launch as RIL  # noqa: E402

SESSION_KIND = "stage1_layer_merge"
KIND = "merge_evaluation"
GPU = "NVIDIA L40S"
LOGS = Path("logs/stages/stage-1/reasoning_inheritance")
DESIGN = "configs/stages/stage-1/reasoning_inheritance/layer_merge/design.json"
DIGESTS = "configs/stages/stage-1/reasoning_inheritance/layer_merge/d_tensor_digests.json"
DRIVER = "scripts/stages/stage-1/reasoning_inheritance/layer_merge_driver.py"
RESUME_DEST = "artifacts/stages/stage-1/reasoning_inheritance/layer_merge_resume"
RELAY_TRANSFER = "transfer/stage1_layer_merge"
#: the two candidates' weight files (D's size: the same structure) and the statistics, for the relay-room check
CANDIDATE_BYTES = 6_430_078_848
STATISTICS_BYTES_MAX = 64 * 2**20

STUDY = SE.Study(
    key="RI-LM1", plan_id="stage1.reasoning_inheritance.layer_merge", logs=LOGS,
    grant=Path("logs/budget/approvals/stage1_layer_merge_grant.json"),
    proposal=LOGS / "plans/layer_merge_proposal.json",
    identity_files=(DESIGN, DIGESTS, RIL.DESIGN, RIL.SELECTION),
    scripts=(DRIVER, "scripts/stages/stage-1/reasoning_inheritance/layer_merge.py",
             "scripts/stages/stage-1/reasoning_inheritance/layer_merge_launch.py",
             *RIL.STUDY.scripts),
    wire_schema="aadistill.stage1.reasoning_inheritance.layer_merge.authorization/v1",
    claims=frozenset({"formal", "evaluation"}), kinds=(KIND,))


class LMAuthorization(SE.EngineeringAuthorization):
    """The study's artifact type: setup's ``stage1_layer_merge`` branch loads exactly this."""
    STUDY = STUDY


def resume_input(spec: str | None):
    """``<relay path>@<sha256>`` (what `stage-resume` printed) as the pod's relay input, or nothing."""
    if not spec:
        return ()
    from aadistill.infrastructure.session import RelayInput
    from shared.deployment import MAIN_RELAY
    path, _, sha = spec.partition("@")
    if not path.startswith(RELAY_TRANSFER + "/") or len(sha) != 64:
        raise SE.AuthorizationError(f"--resume-from {spec!r} is not <{RELAY_TRANSFER}/...>@<sha256>")
    return (RelayInput(path=path, repo=MAIN_RELAY, dest=RESUME_DEST, sha256=sha),)


def command(launch, ctx, plan, driver_extra: str = "") -> str:
    ev, status = SE.evidence_dir(launch, ctx.args.run_id), launch.status()
    work = f"/workspace/{SESSION_KIND}_work"
    resume = ""
    if launch.relay_inputs:
        resume = f"--resume {SE.POD_REPO}/{RESUME_DEST}/{Path(launch.relay_inputs[0].path).name}"
    py = (f"cd {SE.POD_REPO} && HF_HUB_OFFLINE=1 PYTHONPATH={SE.POD_REPO}/src:{SE.POD_REPO}/scripts:"
          f"{SE.POD_REPO}/scripts/stages/stage-1 /opt/train/bin/python")
    cmd = (f"{DRIVER} --run-id {shlex.quote(ctx.args.run_id)} --out {ev} --work {work} --status {status} "
           f"--device cuda --vllm-python /opt/vllm/bin/python {driver_extra} {resume}").strip()
    return (f"mkdir -p {ev} {work} && {py} {cmd}; "
            f"grep -qE 'MARKER:(ALL_DONE|RUN_FAILED)' {status} || "
            f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:RUN_FAILED\" >> {status}")


def the_authorization_is_in_the_session_commit(ctx) -> tuple[bool, str]:
    """As RI-S1's (`launch.the_authorization_is_in_the_session_commit`), for this study's artifact."""
    import subprocess
    rel = str(STUDY.authorization_path(ctx.args.run_id))
    blob = subprocess.run(["git", "show", f"{ctx.args.session_commit}:{rel}"], cwd=REPO, capture_output=True)
    here = (REPO / rel).read_bytes() if (REPO / rel).is_file() else None
    if blob.returncode != 0 or blob.stdout != here:
        return False, f"{rel} is not in session commit {ctx.args.session_commit[:12]} as issued; commit it first"
    return True, f"the authorization is in session commit {ctx.args.session_commit[:12]}"


def reserve_problems(hard_cap_usd: float, g: dict, env: dict, pos: dict) -> list[str]:
    """Why the artifact's cap plus the teardown reserve does not fit (empty when it does). Sessions draw only on the
    ceiling (`SE.funding_problems` holds them to it); the reserve is held OUTSIDE the ceiling, so the cap plus the
    reserve must fit what the project cap and the provider balance leave, and no attempt below the smallest useful
    one is issued (the budget's terminal state)."""
    r = float(g.get("teardown_reserve_usd") or 0.0)
    need = hard_cap_usd + r
    bad = []
    if hard_cap_usd > float(env["remaining_usd"]) + 1e-9:
        bad.append(f"${hard_cap_usd} exceeds the ${env['remaining_usd']} the ceiling leaves")
    if need > float(pos["project_remaining_usd"]) + 1e-9:
        bad.append(f"${hard_cap_usd} + the ${r} reserve exceeds the ${pos['project_remaining_usd']} the project cap leaves")
    if need > float(pos.get("balance_usd") or 0.0) + 1e-9:
        bad.append(f"${hard_cap_usd} + the ${r} reserve exceeds the provider balance ${pos.get('balance_usd')}")
    if hard_cap_usd < float(g.get("min_useful_attempt_usd") or 0.0) - 1e-9:
        bad.append(f"${hard_cap_usd} is below the smallest useful attempt ${g['min_useful_attempt_usd']}: the study's "
                   "budget is exhausted (budget_terminal_state)")
    return bad


def the_teardown_reserve_stays(ctx) -> tuple[bool, str]:
    try:
        g = SE.grant(STUDY, REPO, dry_run=getattr(ctx.args, "dry_run", False))
        env = SE.envelope(STUDY, g, REPO)
        pos = ctx.evidence.get("live_position") or SE.live_position(REPO)
    except SE.AuthorizationError as exc:
        return False, str(exc)
    bad = reserve_problems(float(ctx.auth.hard_cap_usd), g, env, pos)
    return (not bad, f"${ctx.auth.hard_cap_usd:.2f} + the ${g.get('teardown_reserve_usd')} teardown reserve fit"
            if not bad else "; ".join(bad))


def the_relay_has_room(ctx) -> tuple[bool, str]:
    from aadistill.runtime.hub_capacity import would_accept
    from shared.deployment import MAIN_RELAY
    tok = Path(os.path.expanduser("~/.cache/huggingface/token"))
    if not tok.is_file():
        return False, "no hub token on the dev box to ask the relay"
    v = would_accept(MAIN_RELAY, [CANDIDATE_BYTES, CANDIDATE_BYTES, STATISTICS_BYTES_MAX], tok.read_text().strip())
    ctx.evidence["relay_room"] = str(v)
    return v.accepted, f"the relay would accept the candidates and statistics: {v}"


def launch_spec(minutes: float = 0.0, resume_from: str | None = None, driver_extra: str = "") -> SE.Launch:
    from aadistill.infrastructure.session import LocalAsset
    d = json.loads((REPO / DESIGN).read_text())

    def asset(path: str) -> LocalAsset:
        return LocalAsset(repo_path=path, dest_name=Path(path).name, install_to=str(Path(path).parent))
    #: RI-S1's three dev-box assets: the calibration items, the state-evaluation suite, the verified TriviaQA copy
    assets = (asset("artifacts/stages/stage-2/teacher_modes/calibration_v3"),
              asset("artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v3"),
              asset("artifacts/stages/stage-3/eval/battery_v2"))
    launch = SE.Launch(
        session_kind=SESSION_KIND, authorization=LMAuthorization, teacher_model=d["teacher"]["model"],
        teacher_revision=d["teacher"]["revision"], kind=KIND, local_assets=assets, disk_gb=150,
        relay_inputs=resume_input(resume_from), driver=DRIVER,
        extra_prechecks=(the_authorization_is_in_the_session_commit, the_teardown_reserve_stays, the_relay_has_room),
        validation_minutes=float(minutes), setup_minutes=25.0, contingency_fraction=0.10,
        description="RI-LM1: M-avg and M-act at D's structure, their checks, diagnostics, state evaluation and eval-set-1",
        artifact_spec="configs/stages/stage-1/reasoning_inheritance/layer_merge/artifacts.json",
        teardown_note=("delete the pod, verify from the provider that it is gone, STOP; every finished file is on the "
                       "dev box, the candidates and statistics on the relay; nothing stays on the provider"))
    import dataclasses
    return dataclasses.replace(launch, command=functools.partial(command, launch, driver_extra=driver_extra),
                               on_poll=RIL.secure_incremental(launch), products_secured=RIL.products_secured(launch))


def _pop(argv: list[str], flag: str) -> tuple[list[str], str | None]:
    if flag in argv:
        i = argv.index(flag)
        return argv[:i] + argv[i + 2:], argv[i + 1]
    return argv, None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("stage-resume", "archive-evidence"):
        import argparse
        ap = argparse.ArgumentParser(prog=f"RI-LM1 {argv[0]}")
        ap.add_argument("--run-id", required=True)
        ap.add_argument("--from", dest="src", required=True)
        a = ap.parse_args(argv[1:])
        if argv[0] == "stage-resume":
            print(RIL.stage_resume(a.run_id, Path(a.src), prefix=RELAY_TRANSFER))
        else:
            d = json.loads((REPO / DESIGN).read_text())
            print(RIL.stage_resume(a.run_id, Path(a.src), prefix=f"{d['products']['relay_prefix']}/evidence",
                                   name="evidence.tgz", verify=True))
        return 0
    argv, minutes = _pop(argv, "--minutes")
    argv, resume = _pop(argv, "--resume-from")
    argv, driver_extra = _pop(argv, "--driver-extra")
    driver_extra = driver_extra or ""
    if any(c in driver_extra for c in ";&|`$<>"):
        raise SystemExit(f"--driver-extra {driver_extra!r}: driver flags only")
    if argv and argv[0] == "issue":
        import argparse
        ap = argparse.ArgumentParser(prog="RI-LM1 issue")
        ap.add_argument("--run-id", required=True)
        ap.add_argument("--expected-usd", type=float, required=True)
        ap.add_argument("--hard-cap-usd", type=float, required=True)
        ap.add_argument("--dry-run", action="store_true")
        ap.add_argument("--out", default=None)
        ap.add_argument("--note", default="")
        x = ap.parse_args(argv[1:])
        try:
            g = SE.grant(STUDY, REPO, dry_run=x.dry_run)
            bad = reserve_problems(x.hard_cap_usd, g, SE.envelope(STUDY, g, REPO), SE.live_position(REPO))
            if bad:
                raise SE.AuthorizationError("; ".join(bad))
            conditions = {"design_sha256": SE.sha256_file(REPO / DESIGN), "digests_sha256": SE.sha256_file(REPO / DIGESTS),
                          "resume_from": resume or "", "driver_extra": driver_extra}
            path = SE.issue(STUDY, x.run_id, expected_usd=x.expected_usd, hard_cap_usd=x.hard_cap_usd,
                            dry_run=x.dry_run, out=Path(x.out) if x.out else None, note=x.note, kind=KIND,
                            conditions=conditions)
        except SE.AuthorizationError as exc:
            print(f"REFUSED at $0: {exc}")
            return 11
        print(path)
        return 0
    return SE.cli(STUDY, lambda: launch_spec(float(minutes or 0.0), resume, driver_extra), argv, gpu=GPU,
                  description=__doc__.split("\n")[0])


if __name__ == "__main__":
    raise SystemExit(main())
