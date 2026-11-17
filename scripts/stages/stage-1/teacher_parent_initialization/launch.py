#!/usr/bin/env python3
"""The teacher-parent initialization round's sessions (study TPI1): grant, one-use artifacts, launch, closeout, the
resume archive and local retention (dev box).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/teacher_parent_initialization/launch.py \\
        {issue|launch|closeout|position|grant|stage-resume|retain} --kind {teacher_parent_init|teacher_parent_train} ...

`shared.study_engineering` is the mechanism; this module is the instance. The study has its own proposal
(``plans/proposal.json``) and, only once the maintainer grants it, its own grant
(``logs/budget/approvals/stage1_teacher_parent_init_grant.json``); its envelope counts only its own runs. No earlier
study's unspent allowance funds anything here. Dollar amounts live in the proposal / grant, never here.

Two session kinds, one L40S each, on the container disk (no network volume):

* ``teacher_parent_init`` -- `init_driver.py`: A's initialization, its teacher statistics, state evaluation; the
  products to the relay and the init record;
* ``teacher_parent_train`` -- `train_driver.py`: both seeds trained and evaluated in both modes. Issued only once
  `plan.py` has derived the training plan from a COMMITTED init record whose relay copies verify; the artifact binds
  the plan and that record by sha256 (`kind_conditions`) and setup re-checks them on the pod.

The study identity is the design and the selection (frozen before the grant); the training plan is bound per artifact.
Beyond the mechanism's gates, the round's own, read at $0 before the create: the authorization is in the session
commit; the teardown reserve stays outside every cap (`layer_merge_launch.reserve_problems`); the relay would accept
the kind's bytes at their true sizes; the artifact's bound records are the files here; for the train kind, the plan is
the derivation and A's relay copy is present with its hashes.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import shlex
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from shared import study_engineering as SE  # noqa: E402

SESSION_KIND = "stage1_teacher_parent"
INIT_KIND, TRAIN_KIND = "teacher_parent_init", "teacher_parent_train"
GPU = "NVIDIA L40S"
LOGS = Path("logs/stages/stage-1/teacher_parent_initialization")
HERE = "scripts/stages/stage-1/teacher_parent_initialization"
DESIGN = "configs/stages/stage-1/teacher_parent_initialization/design.json"
SELECTION = "configs/stages/stage-1/teacher_parent_initialization/selection.json"
PLAN = f"{LOGS}/plans/training_plan.json"
DRIVERS = {INIT_KIND: f"{HERE}/init_driver.py", TRAIN_KIND: f"{HERE}/train_driver.py"}
RESUME_DEST = "artifacts/stages/stage-1/teacher_parent_initialization/resume"
RELAY_TRANSFER = "transfer/stage1_teacher_parent"
LOCAL_STORE = Path("~/aad-artifacts/stage1_teacher_parent_init").expanduser()
LOCAL_RECORD = LOGS / "storage/local_retention.json"
#: what the relay receives, at most, per kind (bytes): the init (bf16 596M), the 37-point float64 statistics and the
#: mapping; per unit two save points (fp32 weights + AdamW state) live at once and the final fp32 weights
INIT_BYTES, STATS_BYTES, MAPPING_BYTES = 1_300_000_000, 2_000_000_000, 64 * 2**20
SAVEPOINT_BYTES, FINAL_BYTES = 6_200_000_000, 2_500_000_000

STUDY = SE.Study(
    key="TPI1", plan_id="stage1.teacher_parent_initialization", logs=LOGS,
    grant=Path("logs/budget/approvals/stage1_teacher_parent_init_grant.json"),
    proposal=LOGS / "plans/proposal.json",
    identity_files=(DESIGN, SELECTION),
    scripts=(f"{HERE}/__init__.py", f"{HERE}/init_driver.py", f"{HERE}/train_driver.py", f"{HERE}/relay.py",
             f"{HERE}/plan.py", f"{HERE}/selection.py", f"{HERE}/launch.py",
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
    wire_schema="aadistill.stage1.teacher_parent_initialization.authorization/v1",
    claims=frozenset({"formal", "training", "evaluation"}), kinds=(INIT_KIND, TRAIN_KIND))


class TPAuthorization(SE.EngineeringAuthorization):
    """The study's artifact type: setup's ``stage1_teacher_parent`` branch loads exactly this."""
    STUDY = STUDY


def proposal() -> dict:
    return json.loads((REPO / STUDY.proposal).read_text())


def kind_conditions(kind: str, root: Path = REPO) -> dict:
    """The records an artifact of ``kind`` binds, by sha256 (setup re-derives them from the pod's files)."""
    out = {"design_sha256": SE.sha256_file(root / DESIGN), "selection_sha256": SE.sha256_file(root / SELECTION)}
    if kind == TRAIN_KIND:
        plan = json.loads((root / PLAN).read_text())
        out |= {"plan_sha256": SE.sha256_file(root / PLAN),
                "init_record_sha256": SE.sha256_file(root / plan["init_record"]["path"])}
    return out


def relay_object_sizes(kind: str) -> list[int]:
    if kind == INIT_KIND:
        return [INIT_BYTES, STATS_BYTES, MAPPING_BYTES]
    return [SAVEPOINT_BYTES, SAVEPOINT_BYTES, FINAL_BYTES, FINAL_BYTES]


def _kind(ctx) -> str:
    return (getattr(ctx.auth, "stage_conditions", None) or {}).get("kind") or INIT_KIND


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
    v = would_accept(MAIN_RELAY, relay_object_sizes(_kind(ctx)), tok.read_text().strip())
    ctx.evidence["relay_room"] = str(v)
    return v.accepted, f"the relay would accept the {_kind(ctx)} session's bytes: {v}"


def the_kind_records_are_the_bound_ones(ctx) -> tuple[bool, str]:
    kind = _kind(ctx)
    try:
        want = kind_conditions(kind)
    except (OSError, KeyError, ValueError) as exc:
        return False, f"the {kind} records are not here: {exc}"
    sc = ctx.auth.stage_conditions or {}
    bad = {k: v for k, v in want.items() if sc.get(k) != v}
    return (not bad, f"the artifact binds this tree's {sorted(want)}" if not bad else
            f"the artifact binds other records than this tree holds: {sorted(bad)}")


def the_initialization_is_durable(ctx) -> tuple[bool, str]:
    """The train kind: the plan is the derivation, and A's relay copy is there with the recorded hashes ($0)."""
    if _kind(ctx) != TRAIN_KIND:
        return True, "not a training session"
    from huggingface_hub import HfApi
    from stages.teacher_parent_initialization import plan as PL
    try:
        fresh = json.dumps(PL.derive(REPO / json.loads((REPO / PLAN).read_text())["init_record"]["path"]), indent=1) + "\n"
    except Exception as exc:  # noqa: BLE001
        return False, f"the training plan does not derive: {exc}"
    if fresh != (REPO / PLAN).read_text():
        return False, "the committed training plan is not the derivation"
    rec = json.loads((REPO / PLAN).read_text())["units"][0]["initialization"]["relay"]
    tok = Path(os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()
    paths = {f"{rec['prefix']}/{k}": v for k, v in rec["files_sha256"].items()}
    info = {i.path: i for i in HfApi(token=tok).get_paths_info(rec["relay"], list(paths), repo_type="model")}
    bad = [p for p, sha in paths.items() if p not in info or (getattr(info[p], "lfs", None) is not None
                                                             and info[p].lfs.sha256 != sha)]
    return (not bad, "A's initialization is on the relay with its recorded hashes" if not bad else
            f"A's initialization is not on the relay as recorded: {bad}")


def resume_input(spec: str | None):
    if not spec:
        return ()
    from aadistill.infrastructure.session import RelayInput
    from shared.deployment import MAIN_RELAY
    path, _, sha = spec.partition("@")
    if not path.startswith(RELAY_TRANSFER + "/") or len(sha) != 64:
        raise SE.AuthorizationError(f"--resume-from {spec!r} is not <{RELAY_TRANSFER}/...>@<sha256>")
    return (RelayInput(path=path, repo=MAIN_RELAY, dest=RESUME_DEST, sha256=sha),)


def command(launch, kind: str, ctx, plan) -> str:
    ev, status = SE.evidence_dir(launch, ctx.args.run_id), launch.status()
    resume = (f"--resume {SE.POD_REPO}/{RESUME_DEST}/{Path(launch.relay_inputs[-1].path).name}"
              if launch.relay_inputs and launch.relay_inputs[-1].dest == RESUME_DEST else "")
    py = (f"cd {SE.POD_REPO} && HF_HUB_OFFLINE=1 PYTHONPATH={SE.POD_REPO}/src:{SE.POD_REPO}/scripts:"
          f"{SE.POD_REPO}/scripts/stages/stage-1:{SE.POD_REPO}/scripts/stages/stage-0/calibration_volume "
          "/opt/train/bin/python")
    if kind == INIT_KIND:
        cmd = (f"{DRIVERS[kind]} --run-id {shlex.quote(ctx.args.run_id)} --out {ev} "
               f"--work /workspace/teacher_parent/init_work --status {status} --device cuda {resume}")
    else:
        auth = f"{SE.POD_REPO}/{STUDY.authorization_path(ctx.args.run_id)}"
        cmd = (f"{DRIVERS[kind]} --authorization {auth} --out {ev} --status {status} "
               f"--unit-deadline-hours {proposal()['kinds'][TRAIN_KIND]['unit_deadline_hours']} "
               f"--vllm-python /opt/vllm/bin/python --attempt {shlex.quote(ctx.args.run_id)} {resume}")
    return (f"mkdir -p {ev} && {py} {cmd.strip()}; "
            f"grep -qE 'MARKER:(ALL_DONE|RUN_FAILED)' {status} || "
            f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:RUN_FAILED\" >> {status}")


def launch_spec(kind: str, minutes: float = 0.0, resume_from: str | None = None) -> SE.Launch:
    from aadistill.infrastructure.session import LocalAsset
    from stages.reasoning_inheritance import launch as RIL
    d = json.loads((REPO / DESIGN).read_text())
    if kind not in STUDY.kinds:
        raise SE.AuthorizationError(f"kind {kind!r} is not one of {list(STUDY.kinds)}")

    def asset(path: str) -> LocalAsset:
        return LocalAsset(repo_path=path, dest_name=Path(path).name, install_to=str(Path(path).parent))
    if kind == INIT_KIND:
        assets = (asset("artifacts/stages/stage-2/teacher_modes/calibration_v3"),
                  asset("artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v3"))
        relay = ()
    else:
        from stages.teacher_modes import token_matched_phase as TM
        assets = (asset("artifacts/stages/stage-3/eval/battery_v2"),)       #: the verified TriviaQA copy
        relay = TM.stage3_relay_inputs("train_sr_mix")                      #: the A-MIX pack, at hub speed
    relay += resume_input(resume_from)
    launch = SE.Launch(
        session_kind=SESSION_KIND, authorization=TPAuthorization, teacher_model=d["teacher"]["model"],
        teacher_revision=d["teacher"]["revision"], kind=kind, local_assets=assets, disk_gb=150,
        relay_inputs=relay, driver=DRIVERS[kind],
        extra_prechecks=(the_authorization_is_in_the_session_commit, the_teardown_reserve_stays, the_relay_has_room,
                         the_kind_records_are_the_bound_ones, the_initialization_is_durable),
        validation_minutes=float(minutes), setup_minutes=25.0, contingency_fraction=0.05,
        description=("TPI1: A_teacher_parent's initialization, teacher statistics and state evaluation" if kind == INIT_KIND
                     else "TPI1: A_teacher_parent's two seeds trained (A-MIX recipe) and evaluated in both modes"),
        artifact_spec="configs/stages/stage-1/teacher_parent_initialization/artifacts.json",
        teardown_note=("delete the pod, verify from the provider that it is gone, STOP; every finished file is on the "
                       "dev box, every weight on the relay; nothing stays on the provider"))
    return dataclasses.replace(launch, command=functools.partial(command, launch, kind),
                               on_poll=RIL.secure_incremental(launch), products_secured=RIL.products_secured(launch))


# ---- resume and retention ($0, dev box) -------------------------------------------------------------------------------

def resume_archive(src_scr: Path, tgz: Path) -> Path:
    """The earlier run's secured evidence (``<scr>/store/out``, never a live stream) as one gzip tar."""
    out = Path(src_scr) / "store" / "out"
    if not out.is_dir():
        raise SE.AuthorizationError(f"{out} holds no secured evidence to resume from")
    with tarfile.open(tgz, "w:gz") as t:
        for p in sorted(out.rglob("*")):
            rel = p.relative_to(out)
            if p.is_file() and rel.parts[0] != "live" and not p.name.startswith("."):
                t.add(p, arcname=str(rel))
    return tgz


def stage_resume(run_id: str, src_scr: Path) -> str:
    """The resume archive to the relay's transfer area; prints ``<path>@<sha256>`` for ``--resume-from``."""
    from stages.teacher_parent_initialization import relay as R
    tgz = resume_archive(src_scr, Path(src_scr) / f"resume_{run_id}.tgz")
    prefix = f"{RELAY_TRANSFER}/{run_id}"
    res = R.put({tgz.name: tgz}, prefix, message=f"TPI1 resume archive for {run_id}")
    if not res.get("verified"):
        raise SE.AuthorizationError(f"the resume archive did not verify on the relay: {str(res)[:400]}")
    return f"{prefix}/{tgz.name}@{res['files_sha256'][tgz.name]}"


def retain(kind: str, run_id: str, src_scr: Path, *, store: Path = LOCAL_STORE, record: Path = REPO / LOCAL_RECORD) -> dict:
    """Each durable product of a finished session onto the dev box ($0 GPU): fetched from its verified relay record,
    re-hashed, placed in the out-of-tree store and recorded (`relay.get`). The init: its checkpoint, mapping and
    statistics; a training session: each unit's final weights."""
    from stages.teacher_parent_initialization import relay as R
    out = Path(src_scr) / "store" / "out"
    doc = json.loads(record.read_text()) if record.is_file() else {
        "schema": "aadistill.stage1.teacher_parent_initialization.local/v1", "store": str(store), "products": {}}
    todo = {}
    if kind == INIT_KIND:
        ir = json.loads((out / "init_record.json").read_text())
        for name, rec in ir["relay"].items():
            todo[f"init/{name}"] = rec
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
    kind = kind or INIT_KIND
    if argv and argv[0] in ("retain", "stage-resume"):
        ap = argparse.ArgumentParser(prog=f"TPI1 {argv[0]}")
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
        ap = argparse.ArgumentParser(prog="TPI1 issue")
        ap.add_argument("--run-id", required=True)
        ap.add_argument("--expected-usd", type=float, required=True)
        ap.add_argument("--hard-cap-usd", type=float, required=True)
        ap.add_argument("--dry-run", action="store_true")
        ap.add_argument("--out", default=None)
        ap.add_argument("--note", default="")
        x = ap.parse_args(argv[1:])
        try:
            g = SE.grant(STUDY, REPO, dry_run=x.dry_run)
            cap = float(proposal()["kinds"][kind]["per_session_hard_cap_usd"])
            if x.hard_cap_usd > cap + 1e-9:
                raise SE.AuthorizationError(f"${x.hard_cap_usd} exceeds the {kind} cap ${cap}")
            bad = reserve_problems(x.hard_cap_usd, g, SE.envelope(STUDY, g, REPO), SE.live_position(REPO))
            if bad:
                raise SE.AuthorizationError("; ".join(bad))
            path = SE.issue(STUDY, x.run_id, expected_usd=x.expected_usd, hard_cap_usd=x.hard_cap_usd,
                            dry_run=x.dry_run, out=Path(x.out) if x.out else None, note=x.note, kind=kind,
                            conditions={**kind_conditions(kind), "resume_from": resume or ""})
        except (SE.AuthorizationError, OSError, KeyError) as exc:
            print(f"REFUSED at $0: {exc}")
            return 11
        print(path)
        return 0
    return SE.cli(STUDY, lambda: launch_spec(kind, float(minutes or 0.0), resume), argv, gpu=GPU,
                  description=__doc__.split("\n")[0])


if __name__ == "__main__":
    raise SystemExit(main())
