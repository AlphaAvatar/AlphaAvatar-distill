"""What binds an H1 teacher-generation session (P1), and the one way to issue it.

One authorization artifact, loaded under `POLICY`, binds three things:

* the LAUNCH IDENTITY (`launch_identity`): the sha256 of the final generation
  plan file, of the generation protocol file, and of every run input and task
  pool the plan names, each checked against the bytes on disk. The launcher
  checks it before any pod exists; the driver recomputes it on the pod from the
  very files it then reads, so no protocol or input can be read after the check
  without being bound by it;
* the HARNESS (`HARNESS`): the executable files the session runs -- the pod
  path's whole import closure inside this repository, the launcher, the
  shared setup, watchdog and collector -- digested by
  `aadistill.governance.authorization.harness_source_digest`; checked by the
  launcher before any pod and by the driver on the pod;
* a hard cap the planned hard threshold must fit (`SessionRunner.make_plan`).

`issue` writes an artifact ONLY from a maintainer grant document (`--grant`,
required) that names BOTH the launch identity and the harness digest the
maintainer approved; each is recomputed from disk and must equal the grant's,
so neither the plan nor the code can move between approval and issue. There is
no grant for H1 today, so nothing can be issued by running code. Dollar
amounts live in the grant document, never here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from aadistill.governance.authorization import (ActionPolicy, AuthorizationError,
                                                SpendAuthorization, harness_source_digest)
from aadistill.infrastructure.manifest import sha256_file, sha256_json

REPO = Path(__file__).resolve().parents[4]
EXPERIMENT_LOGS = Path("logs/stages/stage-2/reasoning_data")
PLAN = EXPERIMENT_LOGS / "plans/generation_plan.json"
PROTOCOL = Path("configs/stages/stage-2/reasoning_data/generation_protocol.json")
#: committed beside the plan, and installed at the same path on the pod as a declared local asset
INPUTS_DIR = EXPERIMENT_LOGS / "plans/generation_inputs"
ACTION = "teacher_generation"
POLICY = ActionPolicy(
    policy_id="stage2.reasoning_data.h1_generation", allowed=frozenset({ACTION}),
    wire_claims={"authorizes_teacher_generation": ACTION, "authorizes_training": "training",
                 "authorizes_paid_storage": "paid_storage"},
    wire_schema="aadistill.stage2.reasoning_data.generation_authorization/v2",
    plan_hash_key="launch_identity_sha256",
    enforcement=("the launcher refuses a pod unless this artifact is bound to the launch identity (plan, "
                 "protocol, inputs, pools) and the harness on disk, and its hard cap covers the planned hard "
                 "threshold; the driver re-checks both on the pod; one artifact funds one run"))

#: The pod path's import closure inside the repository (a test recomputes it in
#: a fresh interpreter and refuses a module missing here), plus what the
#: launcher and the pod execute around it.
HARNESS = (
    "scripts/shared/pod/autoinit_preflight_setup.sh",
    "scripts/shared/pod/watchdog.py",
    "scripts/shared/pod/collect_artifacts.py",
    "scripts/shared/evaluation/uncapped_eval.py",
    "scripts/shared/source_sets.py",
    "scripts/stages/stage-0/calibration_volume/run_battery.py",
    "scripts/stages/stage-0/calibration_volume/score_battery.py",
    "scripts/stages/stage-0/calibration_volume/tasks/common.py",
    "scripts/stages/stage-0/calibration_volume/tasks/interactive.py",
    "scripts/stages/stage-2/reasoning_data/assets.py",
    "scripts/stages/stage-2/reasoning_data/checkers.py",
    "scripts/stages/stage-2/reasoning_data/generation_adapter.py",
    "scripts/stages/stage-2/reasoning_data/generation_driver.py",
    "scripts/stages/stage-2/reasoning_data/generation_plan.py",
    "scripts/stages/stage-2/reasoning_data/governance.py",
    "scripts/stages/stage-2/reasoning_data/h1_generation_launch.py",
    "scripts/stages/stage-2/reasoning_data/process_check.py",
    "scripts/stages/stage-2/reasoning_data/source_plan.py",
    "scripts/stages/stage-2/reasoning_data/supporting_verify.py",
    "src/aadistill/data/code_exec.py",
    "src/aadistill/data/target_quality.py",
    "src/aadistill/data/verify.py",
    "src/aadistill/evaluation/behavior.py",
    "src/aadistill/evaluation/degeneration.py",
    "src/aadistill/evaluation/strict_answer.py",
    "src/aadistill/evaluation/usable_rollout.py",
    "src/aadistill/governance/authorization.py",
    "src/aadistill/infrastructure/env.py",
    "src/aadistill/infrastructure/manifest.py",
)
#: Package markers and modules the closure reaches only as namespaces; they
#: carry no behaviour of this session, so they are not digested.
HARNESS_EXEMPT_PREFIXES = ("src/aadistill/initialization/",)


class H1GenerationAuthorization(SpendAuthorization):
    """The P1 generation artifact type: loads only under `POLICY`, refuses a
    missing file and an artifact that does not grant teacher generation. The
    launcher, the pod's setup branch (``SESSION_KIND=stage2_h1_generation``)
    and the driver all read it through this class."""

    POLICY = POLICY

    @classmethod
    def load(cls, path, *, policy=None):
        if not Path(path).is_file():
            raise AuthorizationError(f"no authorization at {path}: nothing was issued for this run")
        auth = super().load(path, policy=policy or POLICY)
        if not auth.allows(ACTION):
            raise AuthorizationError(f"{path} does not authorize {ACTION}")
        return auth


def load(path) -> H1GenerationAuthorization:
    return H1GenerationAuthorization.load(path)


def launch_identity(repo_root: Path = REPO) -> dict:
    """The plan, the protocol and every input and pool the plan names, each by
    the sha256 of its bytes on disk; refuses when a file differs from what the
    plan records."""
    root = Path(repo_root)
    plan = json.loads((root / PLAN).read_text())
    inputs = []
    for f in plan["inputs"]:
        got = sha256_file(root / INPUTS_DIR / f["file"])
        if got != f["sha256"]:
            raise AuthorizationError(f"{INPUTS_DIR / f['file']} hashes to {got[:12]}, not the plan's "
                                     f"{f['sha256'][:12]}")
        inputs.append({"file": str(INPUTS_DIR / f["file"]), "sha256": got})
    pools = []
    for p in plan["pools"] + plan.get("reuse", []):
        got = sha256_file(root / p["path"])
        if got != p["sha256"]:
            raise AuthorizationError(f"{p['path']} hashes to {got[:12]}, not the plan's {p['sha256'][:12]}")
        pools.append({"path": p["path"], "sha256": got})
    return {"generation_plan": {"path": str(PLAN), "sha256": sha256_file(root / PLAN)},
            "generation_protocol": {"path": str(PROTOCOL), "sha256": sha256_file(root / PROTOCOL)},
            "inputs": inputs, "pools": pools}


def launch_hash(repo_root: Path = REPO) -> str:
    return sha256_json(launch_identity(repo_root))


def run_dir(run_id: str) -> Path:
    return EXPERIMENT_LOGS / "runs" / run_id


def authorization_path(run_id: str) -> Path:
    return run_dir(run_id) / "governance" / "authorization.json"


def issue(run_id: str, *, grant_path: Path, repo_root: Path = REPO, out: Path | None = None) -> Path:
    """Write the one-use authorization for `run_id` from a maintainer grant
    document, or refuse. The grant must name the launch identity
    (``launch_identity_sha256``) AND the harness digest (``harness_digest``) it
    approves, who granted it, and the expected and hard-cap amounts. Both are
    recomputed from disk and must equal the grant's: a plan or a harness that
    changed since the grant is refused, never adopted. An existing artifact is
    never re-issued."""
    grant_path = Path(grant_path)
    if not grant_path.is_file():
        raise AuthorizationError(f"no grant document at {grant_path}: an H1 generation authorization is "
                                 "issued only from a maintainer grant")
    g = json.loads(grant_path.read_text())
    for key in ("granted_by", "decision", "launch_identity_sha256", "harness_digest", "expected_usd",
                "hard_cap_usd"):
        if g.get(key) in (None, ""):
            raise AuthorizationError(f"{grant_path} lacks {key!r}")
    if g.get("phase") != "P1":
        raise AuthorizationError(f"{grant_path} grants phase {g.get('phase')!r}, not P1")
    lid = launch_hash(repo_root)
    if g["launch_identity_sha256"] != lid:
        raise AuthorizationError(f"the grant approves launch identity {g['launch_identity_sha256'][:12]}, which "
                                 f"this script does not derive (it derives {lid[:12]}): the plan, protocol, "
                                 "inputs or pools changed since the grant")
    harness = harness_source_digest(repo_root, files=HARNESS)
    if g["harness_digest"] != harness["digest"]:
        raise AuthorizationError(f"the grant approves harness {g['harness_digest'][:12]}, but the code on disk "
                                 f"digests to {harness['digest'][:12]}: the executable changed since the grant "
                                 "(or the grant names another), and a new digest is never adopted at issue")
    path = Path(out) if out is not None else Path(repo_root) / authorization_path(run_id)
    if path.exists():
        raise AuthorizationError(f"{path} exists; one artifact per run, never re-issued")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
                          text=True).stdout.strip() or None
    auth = H1GenerationAuthorization(
        authorization_id=f"stage2.reasoning_data.h1_generation.{run_id}",
        granted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        granted_by=g["granted_by"], plan_id="stage2.reasoning_data.h1_generation",
        plan_hash=lid, expected_usd=round(float(g["expected_usd"]), 4),
        hard_cap_usd=round(float(g["hard_cap_usd"]), 4), authorized_stages=(),
        stage_conditions={"phase": "P1", "decision": g["decision"],
                          "grant_sha256": sha256_file(grant_path),
                          "granted_harness_digest": g["harness_digest"]},
        scope_note=g.get("scope", "P1 teacher generation only: no training, no paid storage"),
        authorized_session_commit=None, harness_source_digest=harness["digest"], harness_source_files=HARNESS,
        per_launch_hard_usd=round(float(g["hard_cap_usd"]), 4), provenance_commit=head, action_policy=POLICY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(auth.as_dict(), indent=1) + "\n")
    load(path).require_plan(lid)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("identity", help="print the launch identity and the harness digest")
    i = sub.add_parser("issue", help="write a run's authorization from a maintainer grant document")
    i.add_argument("--run-id", required=True)
    i.add_argument("--grant", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "identity":
        print(json.dumps({"launch_identity_sha256": launch_hash(), "launch_identity": launch_identity(),
                          "harness_digest": harness_source_digest(REPO, files=HARNESS)["digest"]}, indent=1))
        return 0
    print(issue(args.run_id, grant_path=Path(args.grant)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
