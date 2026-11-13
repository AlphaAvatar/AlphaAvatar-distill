#!/usr/bin/env python3
"""The Stage-1 reasoning-inheritance study's sessions: grant, one-use artifacts, launch, closeout (dev box).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/reasoning_inheritance/launch.py \\
        {issue|launch|closeout|position|grant|stage-resume} ...

    issue --run-id <run> --expected-usd <x> --hard-cap-usd <y>          the one-use artifact, every funding check
    launch --run-id <run> --scr <dir> --session-commit <sha> --bundle aad_autoinit_<8hex>.bundle --max-price 1.09 \\
        [--data-center-ids A,B] [--resume-from <relay path>@<sha256>]
    closeout --run-id <run> --scr <dir>
    stage-resume --run-id <run> --from <earlier run's scr>                upload its secured evidence to the relay ($0)

`shared.study_engineering` is the mechanism (grant, envelope, live position, one use, the shared runner's session,
closeout); this module is the instance: the study's bound files, its one session kind, its pod driver
(`driver.py`), its inputs and how finished work leaves the pod while it runs. Dollar amounts live in the grant
(``logs/budget/approvals/stage1_reasoning_inheritance_grant.json``), never here.

FINISHED WORK LEAVES THE POD AS IT IS WRITTEN. The driver writes every finished file atomically under its evidence
directory (``live/`` holds the streams still being written and is never pulled); each poll lists the directory, tars
the files the dev box does not yet hold at that size, pulls the archive, checks its sha256 and unpacks it under
``<scr>/store/out``. The checkpoints and the mapping go to the private relay from the pod and are re-hashed there
(`driver.relay_upload`). A later attempt resumes from that copy: `stage-resume` packs the store and uploads it at $0;
the launch declares it as a relay input, pulled by the pod at hub speed.
"""
from __future__ import annotations

import functools
import hashlib
import json
import shlex
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from shared import study_engineering as SE  # noqa: E402

SESSION_KIND = "stage1_reasoning_inheritance"
KIND = "diagnosis"
GPU = "NVIDIA L40S"
LOGS = Path("logs/stages/stage-1/reasoning_inheritance")
DESIGN = "configs/stages/stage-1/reasoning_inheritance/design.json"
SELECTION = "configs/stages/stage-1/reasoning_inheritance/frozen_selection.json"
DRIVER = "scripts/stages/stage-1/reasoning_inheritance/driver.py"
RESUME_DEST = "artifacts/stages/stage-1/reasoning_inheritance/resume"
RELAY_TRANSFER = "transfer/stage1_reasoning_inheritance"

STUDY = SE.Study(
    key="RI-S1", plan_id="stage1.reasoning_inheritance", logs=LOGS,
    grant=Path("logs/budget/approvals/stage1_reasoning_inheritance_grant.json"),
    proposal=LOGS / "plans/proposal.json",
    identity_files=(DESIGN, SELECTION),
    scripts=(DRIVER, "scripts/stages/stage-1/reasoning_inheritance/launch.py",
             "scripts/stages/stage-1/reasoning_inheritance/selection.py",
             "scripts/stages/stage-1/reasoning_inheritance/__init__.py",
             "scripts/stages/stage-2/teacher_modes/packed_full_search.py",
             "scripts/stages/stage-2/teacher_modes/full_search.py",
             "scripts/stages/stage-2/teacher_modes/hybrid_state_eval.py",
             "scripts/stages/stage-2/teacher_modes/packed_inputs.py",
             "scripts/stages/stage-0/calibration_volume/run_battery.py",
             "scripts/stages/stage-0/calibration_volume/dose_driver.py",
             "scripts/shared/evaluation/uncapped_eval.py",
             "scripts/shared/evaluation/decoding.py",
             "scripts/shared/evaluation/eval_set_scoring.py",
             "scripts/shared/eval_sets.py",
             "scripts/shared/forward_baseline.py",
             "scripts/shared/pod/teacher_cache.py"),
    wire_schema="aadistill.stage1.reasoning_inheritance.authorization/v1",
    claims=frozenset({"formal", "evaluation"}), kinds=(KIND,))


class RIAuthorization(SE.EngineeringAuthorization):
    """The study's artifact type: setup's ``stage1_reasoning_inheritance`` branch loads exactly this."""
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


def command(launch, ctx, plan) -> str:
    ev, status = SE.evidence_dir(launch, ctx.args.run_id), launch.status()
    work = f"/workspace/{SESSION_KIND}_work"
    resume = ""
    if launch.relay_inputs:
        resume = f"--resume {SE.POD_REPO}/{RESUME_DEST}/{Path(launch.relay_inputs[0].path).name}"
    py = (f"cd {SE.POD_REPO} && HF_HUB_OFFLINE=1 PYTHONPATH={SE.POD_REPO}/src:{SE.POD_REPO}/scripts:"
          f"{SE.POD_REPO}/scripts/stages/stage-1 /opt/train/bin/python")
    cmd = (f"{DRIVER} --run-id {shlex.quote(ctx.args.run_id)} --out {ev} --work {work} --status {status} "
           f"--device cuda --vllm-python /opt/vllm/bin/python {resume}").strip()
    return (f"mkdir -p {ev} {work} && {py} {cmd}; "
            f"grep -qE 'MARKER:(ALL_DONE|RUN_FAILED)' {status} || "
            f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:RUN_FAILED\" >> {status}")


# ---- finished work leaves the pod while it runs ---------------------------------------------------------------------

def _pulled_index(st: Path) -> dict:
    p = st / "pulled.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def secure_incremental(launch):
    """on_poll (module docstring). Idempotent; never raises."""
    def on_poll(ctx) -> None:
        try:
            if ctx.target is None:
                return
            st = SE.store(ctx)
            (st / "out").mkdir(parents=True, exist_ok=True)
            SE._scp(ctx, launch.status(), st / "driver.status", limit_s=120)
            ev = SE.evidence_dir(launch, ctx.args.run_id)
            res = ctx.target.run(f"test -d {ev} && cd {ev} && find . -type f ! -path './live/*' ! -name '.*' "
                                 f"-printf '%P\\t%s\\n'", timeout=120)
            listing = {}
            for line in (getattr(res, "stdout", "") or "").splitlines():
                name, _, size = line.rpartition("\t")
                if name and size.isdigit():
                    listing[name] = int(size)
            have = _pulled_index(st)
            new = sorted(n for n, s in listing.items() if have.get(n) != s)
            if not new:
                ctx.evidence["secured_files"] = {"files": len(have), "pending": 0}
                return
            n = int(ctx.evidence.get("secure_rounds", 0)) + 1
            lst, tgz = f"/workspace/ri_pull_{n}.list", f"/workspace/ri_pull_{n}.tgz"
            body = "\n".join(new)
            r = ctx.target.run(f"cat > {lst} <<'RIEOF'\n{body}\nRIEOF\ncd {ev} && tar czf {tgz} -T {lst} && "
                               f"sha256sum {tgz} | cut -d' ' -f1", timeout=900)
            digest = ((getattr(r, "stdout", "") or "").strip().splitlines() or [""])[-1]
            dest = st / f"pull_{n}.tgz"
            rc = SE._scp(ctx, tgz, dest, limit_s=1800)
            got = hashlib.sha256(dest.read_bytes()).hexdigest() if dest.is_file() else None
            ok = rc == 0 and got == digest and len(digest) == 64
            if ok:
                with tarfile.open(dest) as t:
                    t.extractall(st / "out", filter="data")
                for name in new:
                    have[name] = listing[name]
                (st / "pulled.json").write_text(json.dumps(have, indent=0, sort_keys=True) + "\n")
                dest.unlink()
                ctx.target.run(f"rm -f {tgz} {lst}", timeout=60)
            ctx.evidence["secure_rounds"] = n
            ctx.evidence["secured_files"] = {"files": len(have), "pending": len(listing) - len(have),
                                             "last_round_ok": ok, "last_round_files": len(new)}
            if ok:
                ctx.say(f"secured {len(new)} file(s) ({len(have)} held)")
        except Exception as exc:  # noqa: BLE001
            ctx.evidence.setdefault("secure_errors", []).append(repr(exc)[:300])
    return on_poll


def products_secured(launch):
    def check(ctx, *_a, **_k) -> tuple[bool, str]:
        import time
        for i in range(3):                     #: a transient transfer failure is retried, never a reason to hold the pod
            secure_incremental(launch)(ctx)
            if (ctx.evidence.get("secured_files") or {}).get("pending") == 0:
                break
            time.sleep(10 * (i + 1))
        st = SE.store(ctx)
        status = (st / "driver.status").read_text() if (st / "driver.status").is_file() else ""
        sec = ctx.evidence.get("secured_files") or {}
        done = "MARKER:ALL_DONE" in status
        ok = done and sec.get("pending") == 0 and (st / "out" / "report.json").is_file()
        return ok, (f"every finished file is on the dev box ({sec.get('files')} files)" if ok else
                    f"not all finished work is secured (driver done: {done}, pending: {sec.get('pending')})")
    return check


def stage_resume(run_id: str, src_scr: Path) -> str:
    """The earlier run's secured evidence as one archive on the relay ($0, the dev box's uplink). Prints the
    ``--resume-from`` value."""
    from huggingface_hub import HfApi
    from shared.deployment import MAIN_RELAY
    out = Path(src_scr) / "store" / "out"
    if not (out / "records").is_dir():
        raise SE.AuthorizationError(f"{out} holds no secured records to resume from")
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "resume.tgz"
        with tarfile.open(tgz, "w:gz") as t:
            for sub in ("mapping", "records", "eval"):
                if (out / sub).is_dir():
                    t.add(out / sub, arcname=sub)
        sha = hashlib.sha256(tgz.read_bytes()).hexdigest()
        path = f"{RELAY_TRANSFER}/{run_id}/{sha[:16]}/resume.tgz"
        HfApi().upload_file(path_or_fileobj=str(tgz), path_in_repo=path, repo_id=MAIN_RELAY, repo_type="model",
                            commit_message=f"stage1 reasoning inheritance: resume input for {run_id}")
    return f"{path}@{sha}"


# ---- the launch ------------------------------------------------------------------------------------------------------

def the_authorization_is_in_the_session_commit(ctx) -> tuple[bool, str]:
    """The pod checks out the session commit, so the artifact its setup loads must be IN that commit, byte for byte.
    ri_diagnosis_20261113_103414 passed every other gate, then died in setup (AUTHORIZATION_MISMATCH, $0.1153): the
    artifact was committed after the commit the bundle carried. Order: commit code, issue, commit the artifact (that
    commit is the session commit), stage its bundle, launch."""
    import subprocess
    rel = str(STUDY.authorization_path(ctx.args.run_id))
    blob = subprocess.run(["git", "show", f"{ctx.args.session_commit}:{rel}"], cwd=REPO, capture_output=True)
    here = (REPO / rel).read_bytes() if (REPO / rel).is_file() else None
    if blob.returncode != 0 or blob.stdout != here:
        return False, f"{rel} is not in session commit {ctx.args.session_commit[:12]} as issued; commit it first"
    return True, f"the authorization is in session commit {ctx.args.session_commit[:12]}"


def launch_spec(minutes: float = 0.0, resume_from: str | None = None) -> SE.Launch:
    from aadistill.infrastructure.session import LocalAsset
    d = json.loads((REPO / DESIGN).read_text())

    def asset(path: str) -> LocalAsset:
        return LocalAsset(repo_path=path, dest_name=Path(path).name, install_to=str(Path(path).parent))
    #: the calibration items and the state-evaluation suite the search read, and the verified TriviaQA copy (with
    #: the battery it sits in), all dev-box-only and small (~24 MB)
    assets = (asset("artifacts/stages/stage-2/teacher_modes/calibration_v3"),
              asset("artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v3"),
              asset("artifacts/stages/stage-3/eval/battery_v2"))
    launch = SE.Launch(
        session_kind=SESSION_KIND, authorization=RIAuthorization, teacher_model=d["teacher"]["model"],
        teacher_revision=d["teacher"]["revision"], kind=KIND, local_assets=assets, disk_gb=150,
        relay_inputs=resume_input(resume_from), driver=DRIVER,
        extra_prechecks=(the_authorization_is_in_the_session_commit,), validation_minutes=float(minutes), setup_minutes=25.0,
        contingency_fraction=0.10, description="RI-S1: the frozen mapping, D/W/DW, state evaluation and eval-set-1",
        artifact_spec="configs/stages/stage-1/reasoning_inheritance/artifacts.json",
        teardown_note=("delete the pod, verify from the provider that it is gone, STOP; every finished file is on the "
                       "dev box and the checkpoints and the mapping on the relay; nothing stays on the provider"))
    import dataclasses
    return dataclasses.replace(launch, command=functools.partial(command, launch),
                               on_poll=secure_incremental(launch), products_secured=products_secured(launch))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "stage-resume":
        import argparse
        ap = argparse.ArgumentParser(prog="RI-S1 stage-resume")
        ap.add_argument("--run-id", required=True)
        ap.add_argument("--from", dest="src", required=True)
        a = ap.parse_args(argv[1:])
        print(stage_resume(a.run_id, Path(a.src)))
        return 0
    minutes, resume = 0.0, None
    if "--minutes" in argv:
        i = argv.index("--minutes")
        minutes = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if "--resume-from" in argv:
        i = argv.index("--resume-from")
        resume = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if argv and argv[0] == "issue":
        import argparse
        ap = argparse.ArgumentParser(prog="RI-S1 issue")
        ap.add_argument("--run-id", required=True)
        ap.add_argument("--expected-usd", type=float, required=True)
        ap.add_argument("--hard-cap-usd", type=float, required=True)
        ap.add_argument("--dry-run", action="store_true")
        ap.add_argument("--out", default=None)
        ap.add_argument("--note", default="")
        x = ap.parse_args(argv[1:])
        conditions = {"design_sha256": SE.sha256_file(REPO / DESIGN), "selection_sha256": SE.sha256_file(REPO / SELECTION),
                      "resume_from": resume or ""}
        try:
            path = SE.issue(STUDY, x.run_id, expected_usd=x.expected_usd, hard_cap_usd=x.hard_cap_usd,
                            dry_run=x.dry_run, out=Path(x.out) if x.out else None, note=x.note, kind=KIND,
                            conditions=conditions)
        except SE.AuthorizationError as exc:
            print(f"REFUSED at $0: {exc}")
            return 11
        print(path)
        return 0
    return SE.cli(STUDY, lambda: launch_spec(minutes, resume), argv, gpu=GPU, description=__doc__.split("\n")[0])


if __name__ == "__main__":
    raise SystemExit(main())
