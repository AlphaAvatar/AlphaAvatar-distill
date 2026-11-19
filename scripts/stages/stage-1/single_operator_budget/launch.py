#!/usr/bin/env python3
"""SOB1's sessions (study SOB1): grant, one-use artifacts, launch, closeout, the resume archive and local retention.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/single_operator_budget/launch.py \\
        {issue|launch|closeout|position|grant|stage-resume|retain|budget} --kind {sob1_phase1|sob1_train} [--arm <arm>] ...

    # $0: phase 1's budget check against the live position, then a dry-run artifact outside the run directory
    ... launch.py budget --kind sob1_phase1
    ... launch.py issue --kind sob1_phase1 --run-id <id> --expected-usd <e> --hard-cap-usd <c> --dry-run --out <tmp>

`shared.study_engineering` is the mechanism; this module is the instance (TPI1's launcher, with this study's files).
The study has its own proposal (``plans/proposal.json``) and, only once the maintainer grants it, its own grant
(``logs/budget/approvals/stage1_single_operator_budget_grant.json``); its envelope counts only its own runs. Dollar
amounts live in the proposal / grant, never here.

Two session kinds, one L40S each, on the container disk (no network volume):

* ``sob1_phase1`` -- `phase1_driver.py`: the three initializations (each durable at completion), then each untrained
  arm and the teacher evaluated at 8,192 tokens in both modes. The proposal funds this kind only.
* ``sob1_train`` -- `train_driver.py` (TPI1's, with this study's plan, prefix, authorization and package): one arm's
  two seeds trained and evaluated (``--arm``). Issued only once `plan.py` has derived that arm's plan from a COMMITTED
  init record whose relay copies verify, and only under a grant whose ``kinds`` names a cap for it: the proposal names
  none (phase 2 is the maintainer's decision).

Gates read at $0 before the create, beyond the mechanism's: the authorization is in the session commit; the cap plus
the teardown reserve fit (`layer_merge_launch.reserve_problems`); the relay would accept the kind's bytes at their true
sizes; the artifact's bound records are this tree's; for a training session, the plan is the derivation and the
arm's initialization is on the relay with its hashes.
"""
from __future__ import annotations

import dataclasses
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
from stages.single_operator_budget import common as C  # noqa: E402

SESSION_KIND = "stage1_single_operator"
PHASE1_KIND, TRAIN_KIND = "sob1_phase1", "sob1_train"
GPU = "NVIDIA L40S"
LOGS = Path("logs/stages/stage-1/single_operator_budget")
HERE = "scripts/stages/stage-1/single_operator_budget"
TPI = "scripts/stages/stage-1/teacher_parent_initialization"
DESIGN = "configs/stages/stage-1/single_operator_budget/design.json"
SELECTION = "configs/stages/stage-1/single_operator_budget/selection.json"
STRUCTURES = f"{LOGS}/plans/structures.json"
DRIVERS = {PHASE1_KIND: f"{HERE}/phase1_driver.py", TRAIN_KIND: f"{HERE}/train_driver.py"}
RESUME_DEST = "artifacts/stages/stage-1/single_operator_budget/resume"
RELAY_TRANSFER = "transfer/stage1_single_operator_budget"
LOCAL_RECORD = LOGS / "storage/local_retention.json"
ARMS = ("depth_only", "ffn_attention", "residual_width")

STUDY = SE.Study(
    key="SOB1", plan_id="stage1.single_operator_budget", logs=LOGS,
    grant=Path("logs/budget/approvals/stage1_single_operator_budget_grant.json"),
    proposal=LOGS / "plans/proposal.json",
    identity_files=(DESIGN, SELECTION, STRUCTURES),
    scripts=(f"{HERE}/__init__.py", f"{HERE}/common.py", f"{HERE}/phase1_driver.py", f"{HERE}/train_driver.py",
             f"{HERE}/plan.py", f"{HERE}/selection.py", f"{HERE}/structures.py", f"{HERE}/launch.py",
             f"{TPI}/__init__.py", f"{TPI}/init_driver.py", f"{TPI}/train_driver.py", f"{TPI}/relay.py",
             f"{TPI}/plan.py", f"{TPI}/selection.py", f"{TPI}/launch.py",
             "scripts/stages/stage-1/reasoning_inheritance/driver.py",
             "scripts/stages/stage-1/reasoning_inheritance/selection.py",
             "scripts/stages/stage-1/reasoning_inheritance/launch.py",
             "scripts/stages/stage-1/reasoning_inheritance/layer_merge_launch.py",
             "scripts/stages/stage-2/teacher_modes/a1_units.py",
             "scripts/stages/stage-2/teacher_modes/training_plan.py",
             "scripts/stages/stage-2/teacher_modes/packed_training_plan.py",
             "scripts/stages/stage-2/teacher_modes/packed_full_search.py",
             "scripts/stages/stage-2/teacher_modes/full_search.py",
             "scripts/stages/stage-0/calibration_volume/dose_driver.py",
             "scripts/stages/stage-0/calibration_volume/run_battery.py",
             "scripts/shared/evaluation/uncapped_eval.py",
             "scripts/shared/evaluation/eval_set_scoring.py",
             "scripts/shared/training/train_stage3.py",
             "scripts/shared/training/initialization_provenance.py",
             "scripts/shared/study_engineering.py"),
    wire_schema="aadistill.stage1.single_operator_budget.authorization/v1",
    claims=frozenset({"formal", "training", "evaluation"}), kinds=(PHASE1_KIND, TRAIN_KIND))


class Authorization(SE.EngineeringAuthorization):
    """The study's artifact type: setup's ``stage1_single_operator`` branch loads exactly this."""
    STUDY = STUDY


#: the study's evaluation package (TPI1's train_driver uses it for the recovered arms: one rule for every model)
package = C.package


def proposal() -> dict:
    return json.loads((REPO / STUDY.proposal).read_text())


def plan_path(arm: str) -> str:
    if arm not in ARMS:
        raise SE.AuthorizationError(f"--arm {arm!r} is not one of {list(ARMS)}")
    return f"{LOGS}/plans/training_plan_{arm}.json"


def kind_conditions(kind: str, root: Path = REPO, arm: str | None = None) -> dict:
    """The records an artifact of ``kind`` binds, by sha256 (setup re-derives them from the pod's files)."""
    out = {"design_sha256": SE.sha256_file(root / DESIGN), "selection_sha256": SE.sha256_file(root / SELECTION),
           "structures_sha256": SE.sha256_file(root / STRUCTURES)}
    if kind == TRAIN_KIND:
        if arm is None:
            raise SE.AuthorizationError("a training session names its arm (--arm)")
        plan = json.loads((root / plan_path(arm)).read_text())
        out |= {"arm": arm, "plan": plan_path(arm), "plan_sha256": SE.sha256_file(root / plan_path(arm)),
                "init_record_sha256": SE.sha256_file(root / plan["init_record"]["path"])}
    return out


def relay_object_sizes(kind: str, arm: str | None = None) -> list[int]:
    """What the relay receives at most, per kind (bytes), from structures.json: phase 1 the three bf16 inits and
    their mappings; a training session two save points at once and its two fp32 finals."""
    s = json.loads((REPO / STRUCTURES).read_text())["arms"]
    if kind == PHASE1_KIND:
        return [s[a]["checkpoint_bytes"]["init_bf16"] for a in ARMS] + [64 * 2 ** 20] * len(ARMS)
    b = s[arm]["checkpoint_bytes"]
    return [b["save_point"], b["save_point"], b["final_fp32"], b["final_fp32"]]


def _kind(ctx) -> str:
    return (getattr(ctx.auth, "stage_conditions", None) or {}).get("kind") or PHASE1_KIND


def _arm(ctx) -> str | None:
    return (getattr(ctx.auth, "stage_conditions", None) or {}).get("arm") or None


def the_authorization_is_in_the_session_commit(ctx) -> tuple[bool, str]:
    import subprocess
    rel = str(STUDY.authorization_path(ctx.args.run_id))
    blob = subprocess.run(["git", "show", f"{ctx.args.session_commit}:{rel}"], cwd=REPO, capture_output=True)
    here = (REPO / rel).read_bytes() if (REPO / rel).is_file() else None
    if blob.returncode != 0 or blob.stdout != here:
        return False, f"{rel} is not in session commit {ctx.args.session_commit[:12]} as issued; commit it first"
    return True, f"the authorization is in session commit {ctx.args.session_commit[:12]}"


def the_teardown_reserve_stays(ctx) -> tuple[bool, str]:
    from stages.reasoning_inheritance.layer_merge_launch import reserve_problems
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
    v = would_accept(MAIN_RELAY, relay_object_sizes(_kind(ctx), _arm(ctx)), tok.read_text().strip())
    ctx.evidence["relay_room"] = str(v)
    return v.accepted, f"the relay would accept the {_kind(ctx)} session's bytes: {v}"


def the_kind_records_are_the_bound_ones(ctx) -> tuple[bool, str]:
    kind = _kind(ctx)
    try:
        want = kind_conditions(kind, arm=_arm(ctx))
    except (OSError, KeyError, ValueError, SE.AuthorizationError) as exc:
        return False, f"the {kind} records are not here: {exc}"
    sc = ctx.auth.stage_conditions or {}
    bad = {k: v for k, v in want.items() if sc.get(k) != v}
    return (not bad, f"the artifact binds this tree's {sorted(want)}" if not bad else
            f"the artifact binds other records than this tree holds: {sorted(bad)}")


def the_initialization_is_durable(ctx) -> tuple[bool, str]:
    """A training session: its arm's plan is the derivation, and the arm's relay copy is there with the recorded
    hashes ($0)."""
    if _kind(ctx) != TRAIN_KIND:
        return True, "not a training session"
    from huggingface_hub import HfApi
    from stages.single_operator_budget import plan as PL
    arm = _arm(ctx)
    p = REPO / plan_path(arm)
    try:
        fresh = PL.render(PL.derive(REPO / json.loads(p.read_text())["init_record"]["path"], arm))
    except Exception as exc:  # noqa: BLE001
        return False, f"the {arm} training plan does not derive: {exc}"
    if fresh != p.read_text():
        return False, f"the committed {arm} training plan is not the derivation"
    rec = json.loads(p.read_text())["units"][0]["initialization"]["relay"]
    tok = Path(os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()
    paths = {f"{rec['prefix']}/{k}": v for k, v in rec["files_sha256"].items()}
    info = {i.path: i for i in HfApi(token=tok).get_paths_info(rec["relay"], list(paths), repo_type="model")}
    bad = [q for q, sha in paths.items() if q not in info or (getattr(info[q], "lfs", None) is not None
                                                             and info[q].lfs.sha256 != sha)]
    return (not bad, f"{arm}'s initialization is on the relay with its recorded hashes" if not bad else
            f"{arm}'s initialization is not on the relay as recorded: {bad}")


def resume_input(spec: str | None):
    if not spec:
        return ()
    from aadistill.infrastructure.session import RelayInput
    from shared.deployment import MAIN_RELAY
    path, _, sha = spec.partition("@")
    if not path.startswith(RELAY_TRANSFER + "/") or len(sha) != 64:
        raise SE.AuthorizationError(f"--resume-from {spec!r} is not <{RELAY_TRANSFER}/...>@<sha256>")
    return (RelayInput(path=path, repo=MAIN_RELAY, dest=RESUME_DEST, sha256=sha),)


def command(launch, kind: str, arm: str | None, ctx, plan) -> str:
    ev, status = SE.evidence_dir(launch, ctx.args.run_id), launch.status()
    resume = (f"--resume {SE.POD_REPO}/{RESUME_DEST}/{Path(launch.relay_inputs[-1].path).name}"
              if launch.relay_inputs and launch.relay_inputs[-1].dest == RESUME_DEST else "")
    py = (f"cd {SE.POD_REPO} && HF_HUB_OFFLINE=1 PYTHONPATH={SE.POD_REPO}/src:{SE.POD_REPO}/scripts:"
          f"{SE.POD_REPO}/scripts/stages/stage-1:{SE.POD_REPO}/scripts/stages/stage-0/calibration_volume "
          "/opt/train/bin/python")
    if kind == PHASE1_KIND:
        cmd = (f"{DRIVERS[kind]} --run-id {shlex.quote(ctx.args.run_id)} --out {ev} "
               f"--work {C.pod_root()}/phase1_work --status {status} --device cuda "
               f"--vllm-python /opt/vllm/bin/python {resume}")
    else:
        auth = f"{SE.POD_REPO}/{STUDY.authorization_path(ctx.args.run_id)}"
        deadline = (ctx.evidence or {}).get("billed_window_deadline_utc")
        hours = proposal()["phase_2_draft"]["per_arm_session"][arm]["unit_deadline_hours"]
        cmd = (f"{DRIVERS[kind]} --authorization {auth} --out {ev} --status {status} --plan {plan_path(arm)} "
               f"--prefix-root {C.relay_root()} --unit-deadline-hours {hours} "
               + (f"--session-deadline-utc {deadline} " if deadline else "")
               + f"--vllm-python /opt/vllm/bin/python --attempt {shlex.quote(ctx.args.run_id)} {resume}")
    return (f"mkdir -p {ev} && {py} {cmd.strip()}; "
            f"grep -qE 'MARKER:(ALL_DONE|RUN_FAILED)' {status} || "
            f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:RUN_FAILED\" >> {status}")


def driver_minutes(kind: str, arm: str | None = None) -> float:
    p = proposal()
    if kind == PHASE1_KIND:
        return float(p["kinds"][kind]["driver_minutes"])
    return float(p["phase_2_draft"]["per_arm_session"][arm]["hard_cap"]["driver_minutes"])


def launch_spec(kind: str, minutes: float = 0.0, resume_from: str | None = None, arm: str | None = None) -> SE.Launch:
    from aadistill.infrastructure.session import LocalAsset
    from stages.reasoning_inheritance import launch as RIL
    d = C.design()
    if kind not in STUDY.kinds:
        raise SE.AuthorizationError(f"kind {kind!r} is not one of {list(STUDY.kinds)}")

    def asset(path: str) -> LocalAsset:
        return LocalAsset(repo_path=path, dest_name=Path(path).name, install_to=str(Path(path).parent))
    if kind == PHASE1_KIND:
        assets = (asset("artifacts/stages/stage-2/teacher_modes/calibration_v3"),
                  asset("artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v3"),
                  asset("artifacts/stages/stage-3/eval/battery_v2"))      #: the verified TriviaQA copy
        relay = ()
        desc = ("SOB1 phase 1: three ~2B initializations from the teacher (A's operators), then the untrained arms "
                "and the teacher on eval-set-1 at 8,192 tokens")
    else:
        from stages.teacher_modes import token_matched_phase as TM
        plan_path(arm or "")
        assets = (asset("artifacts/stages/stage-3/eval/battery_v2"),)
        relay = TM.stage3_relay_inputs("train_sr_mix")                     #: the A-MIX pack, at hub speed
        desc = f"SOB1 phase 2: {arm}'s two seeds trained (A-MIX recipe) and evaluated at 8,192 tokens"
    relay += resume_input(resume_from)
    launch = SE.Launch(
        session_kind=SESSION_KIND, authorization=Authorization, teacher_model=d["teacher"]["model"],
        teacher_revision=d["teacher"]["revision"], kind=kind, local_assets=assets, disk_gb=150,
        relay_inputs=relay, driver=DRIVERS[kind],
        extra_prechecks=(the_authorization_is_in_the_session_commit, the_teardown_reserve_stays, the_relay_has_room,
                         the_kind_records_are_the_bound_ones, the_initialization_is_durable),
        validation_minutes=float(minutes or driver_minutes(kind, arm)), setup_minutes=25.0, contingency_fraction=0.05,
        description=desc, artifact_spec="configs/stages/stage-1/single_operator_budget/artifacts.json",
        teardown_note=("delete the pod, verify from the provider that it is gone, STOP; every finished file is on the "
                       "dev box, every weight on the relay; nothing stays on the provider"))
    return dataclasses.replace(launch, command=functools.partial(command, launch, kind, arm),
                               on_poll=RIL.secure_incremental(launch), products_secured=RIL.products_secured(launch))


# ---- the budget check, resume and retention ($0, dev box) ------------------------------------------------------------

def budget(kind: str, arm: str | None = None, *, snapshot: dict | None = None) -> dict:
    """The kind's cap against what funds it NOW: the proposal's (or grant's) cap, the teardown reserve, the study's
    envelope, the project cap's remainder and the provider balance, read live (fail closed)."""
    from stages.reasoning_inheritance.layer_merge_launch import reserve_problems
    g = SE.grant(STUDY, REPO, dry_run=True)
    k = (g.get("kinds") or {}).get(kind) or {}
    cap = k.get("per_session_hard_cap_usd")
    env = SE.envelope(STUDY, g, REPO)
    pos = SE.live_position(REPO, snapshot=snapshot)
    out = {"kind": kind, "arm": arm, "source": g["_source"], "status": g.get("status"),
           "expected_usd": k.get("expected_usd"), "hard_cap_usd": cap, "teardown_reserve_usd": g.get("teardown_reserve_usd"),
           "envelope": env, "position": pos}
    if cap is None:
        out["problems"] = [f"{g['_source']} names no cap for {kind}: {k.get('status') or 'not proposed'}"]
    else:
        out["problems"] = (SE.funding_problems(float(cap), g, env, pos) + reserve_problems(float(cap), g, env, pos))
    out["fundable"] = not out["problems"]
    return out


def resume_archive(src_scr: Path, tgz: Path) -> Path:
    from stages.teacher_parent_initialization import launch as TL
    return TL.resume_archive(src_scr, tgz)


def stage_resume(run_id: str, src_scr: Path) -> str:
    """The resume archive to the relay's transfer area; prints ``<path>@<sha256>`` for ``--resume-from``."""
    from stages.teacher_parent_initialization import relay as R
    tgz = resume_archive(src_scr, Path(src_scr) / f"resume_{run_id}.tgz")
    prefix = f"{RELAY_TRANSFER}/{run_id}"
    res = R.put({tgz.name: tgz}, prefix, message=f"SOB1 resume archive for {run_id}")
    if not res.get("verified"):
        raise SE.AuthorizationError(f"the resume archive did not verify on the relay: {str(res)[:400]}")
    return f"{prefix}/{tgz.name}@{res['files_sha256'][tgz.name]}"


def retain(kind: str, run_id: str, src_scr: Path, *, store: Path | None = None, record: Path = REPO / LOCAL_RECORD) -> dict:
    """Each durable product of a finished session onto the dev box ($0 GPU): fetched from its verified relay record,
    re-hashed, placed in the out-of-tree store and recorded (`relay.get`). Phase 1: each arm's initialization (its
    mapping stays on the relay); a training session: each unit's final weights."""
    from stages.teacher_parent_initialization import relay as R
    store = store or C.local_store()
    out = Path(src_scr) / "store" / "out"
    doc = json.loads(record.read_text()) if record.is_file() else {
        "schema": "aadistill.stage1.single_operator_budget.local/v1", "store": str(store), "products": {}}
    todo = {}
    if kind == PHASE1_KIND:
        for ir in sorted(out.glob("*/init_record.json")):
            rec = json.loads(ir.read_text())["relay"]["init"]
            todo[f"{ir.parent.name}/init"] = rec
    else:
        for fin in sorted(out.glob("*/final.json")):
            todo[f"{fin.parent.name}/final"] = json.loads(fin.read_text())
    for key, rec in todo.items():
        dest = store / key / rec["prefix"].rsplit("/", 1)[-1]
        got = R.get(rec, dest)
        doc["products"][key] = {"path": str(dest), "files_sha256": rec["files_sha256"], "relay": rec["relay"],
                                "prefix": rec["prefix"], "producer_run": run_id, "fetched": got,
                                "checked": "every file re-hashed against the relay record the pod verified"}
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def _pop(argv: list[str], flag: str) -> tuple[list[str], str | None]:
    if flag in argv:
        i = argv.index(flag)
        return argv[:i] + argv[i + 2:], argv[i + 1]
    return argv, None


def main(argv=None) -> int:
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, kind = _pop(argv, "--kind")
    argv, minutes = _pop(argv, "--minutes")
    argv, resume = _pop(argv, "--resume-from")
    argv, arm = _pop(argv, "--arm")
    kind = kind or PHASE1_KIND
    if argv and argv[0] == "budget":
        try:
            out = budget(kind, arm)
        except SE.AuthorizationError as exc:
            print(f"REFUSED at $0: {exc}")
            return 11
        print(json.dumps(out, indent=1, default=str))
        return 0 if out["fundable"] else 11
    if argv and argv[0] in ("retain", "stage-resume"):
        ap = argparse.ArgumentParser(prog=f"SOB1 {argv[0]}")
        ap.add_argument("--run-id", required=True)
        ap.add_argument("--from", dest="src", required=True, help="the run's <scr> directory")
        a = ap.parse_args(argv[1:])
        if argv[0] == "stage-resume":
            print(stage_resume(a.run_id, Path(a.src)))
        else:
            print(json.dumps({k: v["path"] for k, v in retain(kind, a.run_id, Path(a.src))["products"].items()}))
        return 0
    if argv and argv[0] in ("issue", "launch"):
        try:
            SE.require_open(STUDY)
        except SE.AuthorizationError as exc:
            print(f"REFUSED at $0: {exc}")
            return 11
    if argv and argv[0] == "issue":
        from stages.reasoning_inheritance.layer_merge_launch import reserve_problems
        ap = argparse.ArgumentParser(prog="SOB1 issue")
        ap.add_argument("--run-id", required=True)
        ap.add_argument("--expected-usd", type=float, required=True)
        ap.add_argument("--hard-cap-usd", type=float, required=True)
        ap.add_argument("--dry-run", action="store_true")
        ap.add_argument("--out", default=None)
        ap.add_argument("--note", default="")
        x = ap.parse_args(argv[1:])
        try:
            g = SE.grant(STUDY, REPO, dry_run=x.dry_run)
            cap = ((g.get("kinds") or {}).get(kind) or {}).get("per_session_hard_cap_usd")
            if cap is None:
                raise SE.AuthorizationError(f"{g['_source']} names no cap for the {kind} kind")
            if x.hard_cap_usd > float(cap) + 1e-9:
                raise SE.AuthorizationError(f"${x.hard_cap_usd} exceeds the {kind} cap ${cap}")
            bad = reserve_problems(x.hard_cap_usd, g, SE.envelope(STUDY, g, REPO), SE.live_position(REPO))
            if bad:
                raise SE.AuthorizationError("; ".join(bad))
            path = SE.issue(STUDY, x.run_id, expected_usd=x.expected_usd, hard_cap_usd=x.hard_cap_usd,
                            dry_run=x.dry_run, out=Path(x.out) if x.out else None, note=x.note, kind=kind,
                            conditions={**kind_conditions(kind, arm=arm), "resume_from": resume or ""})
        except (SE.AuthorizationError, OSError, KeyError) as exc:
            print(f"REFUSED at $0: {exc}")
            return 11
        print(path)
        return 0
    return SE.cli(STUDY, lambda: launch_spec(kind, float(minutes or 0.0), resume, arm), argv, gpu=GPU,
                  description=__doc__.split("\n")[0])


if __name__ == "__main__":
    raise SystemExit(main())
