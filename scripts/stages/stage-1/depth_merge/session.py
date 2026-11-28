"""DMR1's sessions, shared by its two launchers (`depth_merge.launch`, variant 1; `depth_recurrence.launch`, variants 2
and 3): one `Variant` says which study, kinds, grants, designs and drivers; everything else is one implementation.

`shared.study_engineering` is the mechanism (grant, one-use artifacts, the runner's gates, closeout); SOB1's launcher
supplies the relay arithmetic this round reuses unchanged (`single_operator_budget.launch`: `rotation_need`,
`relay_state`, `reclaim_superseded`, `retain`, `chunks`), with SOB1's chain bound to the variant's design
(`depth_merge.common.bind_for`). Dollar amounts live in the proposals and grants, never here.

Gates read at $0 before the create, beyond the mechanism's:

* the authorization is in the session commit; the cap plus the teardown reserve fit;
* the artifact binds this tree's records (the designs, the recurrent config, a training arm's plan and its init
  record) and funds the arm this launch runs;
* a training session: its plan is the derivation (`depth_merge.plan`) and its initialization is on the relay with the
  recorded hashes; the relay accepts what the REMAINING rotation adds (`reclaim_superseded`, then `rotation_need`
  from `relay_state`, against the live headroom);
* a build session: the relay accepts what the chain still adds -- its products, less those a takeover holds there
  under this identity, and the training session after them (`build_need`);
* the dev box's filesystem under ``--scr`` holds the session's evidence bound.

Every closeout (the formal launch's after teardown and a manual ``closeout``) runs the variant's completion: a build
session's init record durable on the secured copy; a training session's two finals and four sets complete against them
(`single_operator_budget.completion.session`) and both finals at their placement (`retain`).
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from shared import study_engineering as SE  # noqa: E402
from stages.depth_merge import common as DC  # noqa: E402

GPU = "NVIDIA L40S"
GIB = 2 ** 30
#: the scripts a DMR1 session runs on the pod beyond src/aadistill (study identity): this round's, and the SOB1 /
#: TPI1 / RI / teacher-modes chain it binds
COMMON_SCRIPTS = (
    "scripts/stages/stage-1/depth_merge/__init__.py", "scripts/stages/stage-1/depth_merge/common.py",
    "scripts/stages/stage-1/depth_merge/merge.py", "scripts/stages/stage-1/depth_merge/build_driver.py",
    "scripts/stages/stage-1/depth_merge/train_driver.py", "scripts/stages/stage-1/depth_merge/plan.py",
    "scripts/stages/stage-1/depth_merge/session.py", "scripts/stages/stage-1/depth_merge/launch.py",
    "scripts/stages/stage-1/depth_merge/analysis.py",
    "scripts/stages/stage-1/depth_recurrence/__init__.py", "scripts/stages/stage-1/depth_recurrence/schedule.py",
    "scripts/stages/stage-1/depth_recurrence/launch.py", "scripts/stages/stage-1/depth_recurrence/engine_check.py",
    "scripts/stages/stage-1/single_operator_budget/__init__.py", "scripts/stages/stage-1/single_operator_budget/common.py",
    "scripts/stages/stage-1/single_operator_budget/train_driver.py",
    "scripts/stages/stage-1/single_operator_budget/completion.py", "scripts/stages/stage-1/single_operator_budget/launch.py",
    "scripts/stages/stage-1/single_operator_budget/analysis.py",
    "scripts/stages/stage-1/teacher_parent_initialization/__init__.py",
    "scripts/stages/stage-1/teacher_parent_initialization/init_driver.py",
    "scripts/stages/stage-1/teacher_parent_initialization/train_driver.py",
    "scripts/stages/stage-1/teacher_parent_initialization/relay.py",
    "scripts/stages/stage-1/teacher_parent_initialization/plan.py",
    "scripts/stages/stage-1/teacher_parent_initialization/analysis.py",
    "scripts/stages/stage-1/teacher_parent_initialization/b_evidence.py",
    "scripts/stages/stage-1/reasoning_inheritance/analysis.py",
    "scripts/stages/stage-1/reasoning_inheritance/layer_merge.py",
    "scripts/stages/stage-1/reasoning_inheritance/layer_merge_analysis.py",
    "scripts/stages/stage-1/reasoning_inheritance/driver.py", "scripts/stages/stage-1/reasoning_inheritance/launch.py",
    "scripts/stages/stage-1/reasoning_inheritance/layer_merge_launch.py",
    "scripts/stages/stage-2/teacher_modes/a1_units.py", "scripts/stages/stage-2/teacher_modes/training_plan.py",
    "scripts/stages/stage-2/teacher_modes/packed_training_plan.py",
    "scripts/stages/stage-2/teacher_modes/packed_full_search.py", "scripts/stages/stage-2/teacher_modes/full_search.py",
    "scripts/stages/stage-0/calibration_volume/score_battery.py",
    "scripts/stages/stage-0/calibration_volume/run_battery.py",
    "scripts/shared/evaluation/uncapped_eval.py", "scripts/shared/evaluation/eval_set_scoring.py",
    "scripts/shared/training/train_stage3.py", "scripts/shared/training/initialization_provenance.py",
    "scripts/shared/rollout/vllm_plugin_path.py",
    "scripts/shared/rollout/vllm_plugin/aadistill_vllm_plugin-1.0.dist-info/entry_points.txt",
    "scripts/shared/study_engineering.py")


@dataclass(frozen=True)
class Variant:
    """One DMR1 study (a launcher's instance)."""
    key: str                                   #: the study key, DMR1-M / DMR1-R
    plan_id: str
    logs: Path                                 #: repo-relative
    session_kind: str                          #: setup's authorization branch
    build_kind: str | None                     #: the init build's kind (variant 1 only)
    train_kinds: dict = field(default_factory=dict)   #: kind -> the arms it trains
    identity_files: tuple = ()
    artifact_spec: str = ""
    label: str = ""

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(k for k in ((self.build_kind,) if self.build_kind else ()) + tuple(self.train_kinds))

    def grant_path(self, kind: str) -> Path:
        return Path("logs/budget/approvals") / f"dmr1_{kind}_grant.json"

    def proposal_path(self, kind: str) -> Path:
        return self.logs / "plans" / f"proposal_{kind}.json"

    def study(self) -> SE.Study:
        k0 = self.kinds[0]
        return SE.Study(
            key=self.key, plan_id=self.plan_id, logs=self.logs, grant=self.grant_path(k0),
            proposal=self.proposal_path(k0),
            kind_grants=tuple((k, self.grant_path(k), self.proposal_path(k)) for k in self.kinds),
            identity_files=self.identity_files, scripts=COMMON_SCRIPTS,
            wire_schema=f"aadistill.{self.plan_id}.authorization/v1",
            claims=frozenset({"formal", "training", "evaluation"}), kinds=self.kinds)

    def arm_of(self, kind: str, arm: str | None) -> str | None:
        if kind == self.build_kind:
            return None
        arms = self.train_kinds.get(kind)
        if arms is None:
            raise SE.AuthorizationError(f"kind {kind!r} is not one of {list(self.kinds)}")
        if arm is None and len(arms) == 1:
            return arms[0]
        if arm not in arms:
            raise SE.AuthorizationError(f"--arm {arm!r} is not one of the {kind} arms {list(arms)}")
        return arm


def plan_path(arm: str) -> str:
    from stages.depth_merge import plan as PL
    return str(PL.out_path(arm).relative_to(REPO))


def kind_conditions(v: Variant, kind: str, root: Path = REPO, arm: str | None = None) -> dict:
    """The records an artifact of ``kind`` binds, by sha256 (setup re-derives them from the pod's files)."""
    out = {f"identity:{p}": SE.sha256_file(root / p) for p in v.identity_files}
    if kind == v.build_kind:
        out["digests_sha256"] = SE.sha256_file(root / "configs/stages/stage-1/depth_merge/depth_init_tensor_digests.json")
        return out
    arm = v.arm_of(kind, arm)
    plan = json.loads((root / plan_path(arm)).read_text())
    out |= {"arm": arm, "plan": plan_path(arm), "plan_sha256": SE.sha256_file(root / plan_path(arm)),
            "init_record_sha256": SE.sha256_file(root / plan["init_record"]["path"])}
    ev = plan.get("engine_validation") or {}
    if ev.get("mode") == "reuse":
        #: the record this arm trains under is pinned by the artifact; setup re-derives it from the pod's checkout
        rec = root / ev["record"]
        if not rec.is_file():
            raise SE.AuthorizationError(f"{arm} reuses the engine validation of {ev['from_arm']}, and {ev['record']} "
                                        "does not exist: run and adopt it first (engine_check.py adopt)")
        out["engine_record_sha256"] = SE.sha256_file(rec)
    return out


def engine_validation_problems(arm: str | None, root: Path = REPO) -> list[str]:
    """For an arm whose plan reuses an engine validation: why the canonical record does not stand for THIS plan
    (`engine_check.record_problems`, everything but the pod's environment). Empty for any other arm."""
    if arm is None or not DC.is_recurrent(arm):
        return []
    plan = json.loads((root / plan_path(arm)).read_text())
    ev = plan.get("engine_validation") or {}
    if ev.get("mode") != "reuse":
        return []
    from stages.depth_recurrence import engine_check as EC
    p = root / ev["record"]
    if not p.is_file():
        return [f"{ev['record']} does not exist: {ev['from_arm']}'s session has not produced an adopted PASS"]
    return EC.record_problems(json.loads(p.read_text()), plan, root=root)


def kind_grant(v: Variant, kind: str, *, dry_run: bool = False) -> dict:
    return SE.grant(v.study(), REPO, dry_run=dry_run, kind=kind)


def session_terms(v: Variant, kind: str, arm: str | None, g: dict) -> dict:
    k = (g.get("kinds") or {}).get(kind) or {}
    if kind in v.train_kinds:
        return (k.get("per_arm") or {}).get(v.arm_of(kind, arm)) or {}
    return k


def budget(v: Variant, kind: str, arm: str | None = None, *, snapshot: dict | None = None) -> dict:
    """The kind's cap against what funds it NOW (SOB1's `budget` rule): the grant's cap (its proposal before a grant
    exists), the teardown reserve, the kind's own envelope, the project cap's remainder and the balance, live."""
    from stages.reasoning_inheritance.layer_merge_launch import reserve_problems
    st = v.study()
    pos = SE.live_position(REPO, snapshot=snapshot)
    closed = SE.kind_closure(st, kind, REPO)
    if closed:
        return {"kind": kind, "arm": arm, "closed": closed, "position": pos, "fundable": False,
                "problems": [f"the {kind} kind is CLOSED"]}
    g = kind_grant(v, kind, dry_run=True)
    k = session_terms(v, kind, arm, g)
    cap = k.get("per_session_hard_cap_usd")
    env = SE.envelope(st, g, REPO, kind)
    out = {"kind": kind, "arm": arm, "source": g["_source"], "status": g.get("status"),
           "expected_usd": k.get("expected_usd"), "hard_cap_usd": cap, "teardown_reserve_usd": g.get("teardown_reserve_usd"),
           "envelope": env, "position": pos}
    out["problems"] = ([f"{g['_source']} names no cap for {kind}"] if cap is None else
                       SE.funding_problems(float(cap), g, env, pos) + reserve_problems(float(cap), g, env, pos))
    out["fundable"] = not out["problems"]
    return out


# ---- the gates ----------------------------------------------------------------------------------------------------

def _kind(ctx) -> str:
    return (getattr(ctx.auth, "stage_conditions", None) or {}).get("kind")


def _arm(ctx) -> str | None:
    return (getattr(ctx.auth, "stage_conditions", None) or {}).get("arm") or None


def gates(v: Variant, arm: str | None) -> tuple:
    st = v.study()

    def the_authorization_is_in_the_session_commit(ctx) -> tuple[bool, str]:
        import subprocess
        rel = str(st.authorization_path(ctx.args.run_id))
        blob = subprocess.run(["git", "show", f"{ctx.args.session_commit}:{rel}"], cwd=REPO, capture_output=True)
        here = (REPO / rel).read_bytes() if (REPO / rel).is_file() else None
        if blob.returncode != 0 or blob.stdout != here:
            return False, f"{rel} is not in session commit {ctx.args.session_commit[:12]} as issued; commit it first"
        return True, f"the authorization is in session commit {ctx.args.session_commit[:12]}"

    def the_teardown_reserve_stays(ctx) -> tuple[bool, str]:
        from stages.reasoning_inheritance.layer_merge_launch import reserve_problems
        try:
            g = kind_grant(v, _kind(ctx), dry_run=getattr(ctx.args, "dry_run", False))
            env = SE.envelope(st, g, REPO, _kind(ctx))
            pos = ctx.evidence.get("live_position") or SE.live_position(REPO)
        except SE.AuthorizationError as exc:
            return False, str(exc)
        bad = reserve_problems(float(ctx.auth.hard_cap_usd), g, env, pos)
        return (not bad, "the cap and the teardown reserve fit" if not bad else "; ".join(bad))

    def the_kind_records_are_the_bound_ones(ctx) -> tuple[bool, str]:
        try:
            want = kind_conditions(v, _kind(ctx), arm=_arm(ctx))
        except (OSError, KeyError, ValueError, SE.AuthorizationError) as exc:
            return False, f"the {_kind(ctx)} records are not here: {exc}"
        sc = ctx.auth.stage_conditions or {}
        bad = sorted(k for k, x in want.items() if sc.get(k) != x)
        return (not bad, "the artifact binds this tree's records" if not bad else
                f"the artifact binds other records than this tree holds: {bad}")

    def the_launch_runs_the_authorized_arm_and_plan(ctx) -> tuple[bool, str]:
        bad = arm_binding_problems(v, getattr(ctx.auth, "stage_conditions", None) or {}, arm)
        return (not bad, f"the artifact funds the plan this launch runs ({arm})" if not bad else "; ".join(bad))

    def the_initialization_is_durable(ctx) -> tuple[bool, str]:
        if _kind(ctx) == v.build_kind:
            return True, "a build session"
        return initialization_on_relay(v.arm_of(_kind(ctx), _arm(ctx)))

    def the_engine_validation_applies(ctx) -> tuple[bool, str]:
        a = None if _kind(ctx) == v.build_kind else v.arm_of(_kind(ctx), _arm(ctx))
        bad = engine_validation_problems(a)
        return (not bad, "no engine validation to reuse, or the reused record stands for this plan" if not bad
                else "; ".join(bad))

    def the_relay_has_room(ctx) -> tuple[bool, str]:
        return relay_room(v, _kind(ctx), _arm(ctx), ctx.evidence, resume_from=getattr(ctx.args, "resume_from", None))

    def the_dev_box_has_room(ctx) -> tuple[bool, str]:
        import shutil
        scr = Path(getattr(ctx.args, "scr", None) or DC.local_store(next(iter(DC.ARMS))))
        scr.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(scr).free
        need = int(evidence_bound_gib(_kind(ctx), v) * GIB)
        ctx.evidence["dev_box_room"] = {"free_bytes": free, "bound_bytes": need, "path": str(scr)}
        return free >= need, f"{scr}: {free / GIB:.2f} GiB free for the session's {need / GIB:.2f} GiB evidence bound"

    out = (the_authorization_is_in_the_session_commit, the_teardown_reserve_stays, the_kind_records_are_the_bound_ones,
           the_initialization_is_durable, the_engine_validation_applies, the_relay_has_room, the_dev_box_has_room)
    return out + ((the_launch_runs_the_authorized_arm_and_plan,) if arm is not None else ())


def evidence_bound_gib(kind: str, v: Variant) -> float:
    """What a session brings back to the dev box at most (its evidence, the pull archive and the teardown archive),
    from SOB1's measured sessions (321-348 MB of evidence for two seeds and four sets) with a factor of 3; the weights
    stay on the relay."""
    return 0.5 if kind == v.build_kind else 1.5


def arm_binding_problems(v: Variant, sc: dict, arm: str | None, root: Path = REPO) -> list[str]:
    kind = (sc or {}).get("kind")
    if kind not in v.train_kinds:
        return []
    try:
        want = kind_conditions(v, kind, root, arm)
    except (OSError, KeyError, ValueError, SE.AuthorizationError) as exc:
        return [f"the plan for --arm {arm} is not on this tree: {exc}"]
    return [f"{k}: the artifact binds {sc.get(k)!r}, this launch runs {want[k]!r}"
            for k in ("arm", "plan", "plan_sha256", "init_record_sha256") if sc.get(k) != want[k]]


def initialization_on_relay(arm: str) -> tuple[bool, str]:
    from huggingface_hub import HfApi
    from stages.depth_merge import plan as PL
    p = REPO / plan_path(arm)
    try:
        fresh = PL.render(PL.derive(arm, REPO / json.loads(p.read_text())["init_record"]["path"]))
    except Exception as exc:  # noqa: BLE001
        return False, f"the {arm} training plan does not derive: {exc}"
    if fresh != p.read_text():
        return False, f"the committed {arm} training plan is not the derivation"
    rec = json.loads(p.read_text())["units"][0]["initialization"]["relay"]
    tok = Path(os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()
    paths = {f"{rec['prefix']}/{k}": x for k, x in rec["files_sha256"].items()}
    info = {i.path: i for i in HfApi(token=tok).get_paths_info(rec["relay"], list(paths), repo_type="model")}
    bad = [q for q, sha in paths.items() if q not in info or (getattr(info[q], "lfs", None) is not None
                                                             and info[q].lfs.sha256 != sha)]
    return (not bad, f"{arm}'s initialization is on the relay with its recorded hashes" if not bad else
            f"{arm}'s initialization is not on the relay as recorded: {bad}")


def checkpoint_bytes(root: Path = REPO) -> dict:
    """The depth structure's sizes (every DMR1 arm has SOB1's depth arm's 16 blocks): SOB1's structures.json."""
    s = json.loads((root / "logs/stages/stage-1/single_operator_budget/plans/structures.json").read_text())
    return s["arms"][DC.SOB1_DEPTH]["checkpoint_bytes"]


def resume_records(spec: str | None) -> tuple[dict, dict]:
    """The build record and relay record a build attempt resumes from: ``--resume-from``'s archive on the relay,
    fetched and re-hashed, read as the pod's `take_over` will copy them; ``({}, {})`` for a first run."""
    if not spec:
        return {}, {}
    import tarfile
    import tempfile
    from huggingface_hub import hf_hub_download
    from shared.deployment import MAIN_RELAY
    from stages.depth_merge import build_driver as BD
    path, _, sha = spec.partition("@")
    tok = Path(os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(hf_hub_download(MAIN_RELAY, path, repo_type="model", token=tok, cache_dir=tmp))
        if SE.sha256_file(f) != sha:
            raise SE.AuthorizationError(f"the resume archive {path} is not {sha[:12]}")
        got = []
        with tarfile.open(f) as t:
            names = set(t.getnames())
            for name in (f"{BD.ARM}/records/build.json", f"{BD.ARM}/records/relay.json"):
                got.append(json.loads(t.extractfile(name).read()) if name in names else {})
    return got[0], got[1]


def relay_present(token: str):
    """``present(record)`` for `build_driver.held`: every file the record names at its prefix on the relay, each LFS
    object's sha256 the record's."""
    from huggingface_hub import HfApi
    from shared.deployment import MAIN_RELAY
    api = HfApi(token=token)

    def present(r: dict) -> bool:
        want = {f"{r['prefix']}/{k}": x for k, x in (r.get("files_sha256") or {}).items()}
        info = {i.path: i for i in api.get_paths_info(r.get("relay") or MAIN_RELAY, list(want), repo_type="model")}
        return bool(want) and all(q in info and (getattr(info[q], "lfs", None) is None or info[q].lfs.sha256 == x)
                                  for q, x in want.items())
    return present


def build_need(ident: dict, rec: dict, prev: dict, present) -> dict:
    """What a build session asks of the relay's live headroom: the whole chain (`storage.sequence`) less what the
    attempt takes over there (`build_driver.held` on the records it resumes from: its products under this identity,
    their files present), counted as the chain counts them (`storage.remaining`). A first run holds nothing and asks
    for the whole chain; the live headroom already excludes what is held, so nothing is subtracted twice."""
    from stages.depth_merge import build_driver as BD
    from stages.depth_merge import storage as STO
    return STO.remaining(BD.held(ident, rec, prev, present))


def relay_room(v: Variant, kind: str, arm: str | None, evidence: dict, *,
               resume_from: str | None = None) -> tuple[bool, str]:
    """A build session: what the chain still adds -- its products (less those a takeover holds on the relay) and the
    training session after them (`build_need`). A training session: first the superseded save points a lost pod left
    are removed (SOB1's `reclaim_superseded`, dev-box side), then what the REMAINING rotation adds (`rotation_need`
    from `relay_state`), asked of the relay's live headroom."""
    from aadistill.runtime.hub_capacity import would_accept
    from shared.deployment import MAIN_RELAY
    from stages.single_operator_budget import launch as SL
    tok = Path(os.path.expanduser("~/.cache/huggingface/token"))
    if not tok.is_file():
        return False, "no hub token on the dev box to ask the relay"
    sizes = checkpoint_bytes()
    if kind == v.build_kind:
        #: the WHOLE chain, not the build's products alone: an init built for a training session the relay cannot
        #: hold would wait on storage after its money was spent (storage.py; design storage.chain_rule)
        import tempfile
        from stages.depth_merge import build_driver as BD
        try:
            rec, prev = resume_records(resume_from)
            with tempfile.TemporaryDirectory() as tmp:
                ident = BD.current_identity(Path(tmp))
            need = build_need(ident, rec, prev, relay_present(tok.read_text().strip()))
        except Exception as exc:  # noqa: BLE001
            return False, f"what the relay already holds of this build is unknown: {type(exc).__name__}: {exc}"
        evidence["relay_need"] = need
        res = would_accept(MAIN_RELAY, SL.chunks(need["need_bytes"]), tok.read_text().strip())
        evidence["relay_room"] = str(res)
        return res.accepted, (f"the relay would accept what the chain still adds ({need['need_gib']} GiB: its peak "
                              f"{need['remaining_peak_gib']} GiB at {need['peak_at']} with the margin; held on the "
                              f"relay: {need['held'] or 'nothing'}): {res}")
    arm = v.arm_of(kind, arm)
    before = DC.bind_for(arm)
    try:
        plan = json.loads((REPO / plan_path(arm)).read_text())
        evidence["relay_reclaim"] = SL.reclaim_superseded(arm, plan=plan,
                                                          record=REPO / v.logs / "storage/relay_reclaim.json")
        state = SL.relay_state(arm, plan=plan)
    finally:
        from stages.single_operator_budget import common as SC
        SC.restore(before)
    need = SL.rotation_need(plan, state, sizes)
    evidence["relay_state"], evidence["relay_need"] = state, {k: x for k, x in need.items() if k != "trace"}
    if need["peak_bytes"] <= 0:
        return True, "the remaining rotation adds nothing beyond what the relay holds"
    res = would_accept(MAIN_RELAY, SL.chunks(need["peak_bytes"]), tok.read_text().strip())
    evidence["relay_room"] = str(res)
    return res.accepted, f"the remaining rotation needs {need['peak_gib']} GiB more at its peak: {res}"


# ---- the launch ---------------------------------------------------------------------------------------------------

DRIVERS = {"build": "scripts/stages/stage-1/depth_merge/build_driver.py",
           "train": "scripts/stages/stage-1/depth_merge/train_driver.py"}


def command(v: Variant, launch, kind: str, arm: str | None, ctx, plan) -> str:
    ev, status = SE.evidence_dir(launch, ctx.args.run_id), launch.status()
    resume = (f"--resume {SE.POD_REPO}/{resume_dest(v)}/{Path(launch.relay_inputs[-1].path).name}"
              if launch.relay_inputs and launch.relay_inputs[-1].dest == resume_dest(v) else "")
    py = (f"cd {SE.POD_REPO} && HF_HUB_OFFLINE=1 PYTHONPATH={SE.POD_REPO}/src:{SE.POD_REPO}/scripts:"
          f"{SE.POD_REPO}/scripts/stages/stage-1:{SE.POD_REPO}/scripts/stages/stage-0/calibration_volume "
          "/opt/train/bin/python")
    if kind == v.build_kind:
        from stages.depth_merge import build_driver as BD
        cmd = (f"{DRIVERS['build']} --run-id {shlex.quote(ctx.args.run_id)} --out {ev} "
               f"--work {DC.pod_root('depth_merge')}/build_work --status {status} --device {BD.BUILD_DEVICE} {resume}")
    else:
        arm = v.arm_of(kind, arm)
        auth = f"{SE.POD_REPO}/{v.study().authorization_path(ctx.args.run_id)}"
        deadline = (ctx.evidence or {}).get("billed_window_deadline_utc")
        hours = session_terms(v, kind, arm, kind_grant(v, kind, dry_run=getattr(ctx.args, "dry_run", False)))[
            "unit_deadline_hours"]
        cmd = (f"{DRIVERS['train']} --authorization {auth} --out {ev} --status {status} --plan {plan_path(arm)} "
               f"--prefix-root {DC.relay_root(arm)} --unit-deadline-hours {hours} "
               + (f"--session-deadline-utc {deadline} " if deadline else "")
               + f"--vllm-python /opt/vllm/bin/python --attempt {shlex.quote(ctx.args.run_id)} {resume}")
    return (f"mkdir -p {ev} && {py} {cmd.strip()}; "
            f"grep -qE 'MARKER:(ALL_DONE|RUN_FAILED)' {status} || "
            f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:RUN_FAILED\" >> {status}")


def resume_dest(v: Variant) -> str:
    return f"artifacts/{v.logs.relative_to('logs')}/resume"


def relay_transfer(v: Variant) -> str:
    return f"transfer/{v.plan_id.replace('.', '_')}"


def resume_input(v: Variant, spec: str | None):
    if not spec:
        return ()
    from aadistill.infrastructure.session import RelayInput
    from shared.deployment import MAIN_RELAY
    path, _, sha = spec.partition("@")
    if not path.startswith(relay_transfer(v) + "/") or len(sha) != 64:
        raise SE.AuthorizationError(f"--resume-from {spec!r} is not <{relay_transfer(v)}/...>@<sha256>")
    return (RelayInput(path=path, repo=MAIN_RELAY, dest=resume_dest(v), sha256=sha),)


def driver_minutes(v: Variant, kind: str, arm: str | None) -> float:
    try:
        g = kind_grant(v, kind)
    except SE.AuthorizationError:
        prop = REPO / v.proposal_path(kind)
        if not prop.is_file():
            raise SE.AuthorizationError(f"{kind} has no grant and no proposal ({v.proposal_path(kind)}): nothing to "
                                        "launch") from None
        g = json.loads(prop.read_text())
    return float(session_terms(v, kind, arm, g)["driver_minutes"])


def launch_spec(v: Variant, authorization: type, kind: str, minutes: float = 0.0, resume_from: str | None = None,
                arm: str | None = None) -> SE.Launch:
    from aadistill.infrastructure.session import LocalAsset
    from stages.reasoning_inheritance import launch as RIL
    d = DC.design()
    if kind not in v.kinds:
        raise SE.AuthorizationError(f"kind {kind!r} is not one of {list(v.kinds)}")

    def asset(path: str) -> LocalAsset:
        return LocalAsset(repo_path=path, dest_name=Path(path).name, install_to=str(Path(path).parent))
    if kind == v.build_kind:
        assets = (asset("artifacts/stages/stage-2/teacher_modes/calibration_v3"),
                  asset("artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v3"))
        relay, desc, driver = (), "DMR1-M build: the alpha-merged initialization from the teacher", DRIVERS["build"]
    else:
        from stages.teacher_modes import token_matched_phase as TM
        arm = v.arm_of(kind, arm)
        assets = (asset("artifacts/stages/stage-3/eval/battery_v2"),)
        relay = TM.stage3_relay_inputs("train_sr_mix")                   #: the A-MIX pack, at hub speed
        desc, driver = f"{v.key}: {arm}'s two seeds trained (SOB1's depth recipe) and evaluated at 8,192 tokens", DRIVERS["train"]
    relay += resume_input(v, resume_from)
    launch = SE.Launch(
        session_kind=v.session_kind, authorization=authorization, teacher_model=d["teacher"]["model"],
        teacher_revision=d["teacher"]["revision"], kind=kind, local_assets=assets, disk_gb=150, relay_inputs=relay,
        driver=driver, extra_prechecks=gates(v, arm),
        validation_minutes=float(minutes or driver_minutes(v, kind, arm)), setup_minutes=25.0,
        contingency_fraction=0.05, description=desc, artifact_spec=v.artifact_spec,
        teardown_note=("delete the pod, verify from the provider that it is gone, STOP; every finished file is on the "
                       "dev box, every weight on the relay; nothing stays on the provider"))
    return dataclasses.replace(launch, command=functools.partial(command, v, launch, kind, arm),
                               on_poll=RIL.secure_incremental(launch), products_secured=RIL.products_secured(launch),
                               completion_check=functools.partial(completion, v, kind, arm))


# ---- closeout: the session's completion ($0, dev box) -------------------------------------------------------------

def completion(v: Variant, kind: str, arm: str | None, run_id: str, scr: Path, *, root: Path = REPO,
               lfs_sha=None) -> dict:
    """A build session: the init record durable on the secured copy under this tree's identity of the build's
    products (`build_driver.durable_init`, `build_driver.current_identity`). A training
    session: its two finals and four sets complete against them (`completion.session`, SOB1's chain bound to the
    arm's design) and both finals at their placement (`single_operator_budget.launch.retain`). Writes
    ``runs/<run_id>/closeout/session_completion.json``."""
    from stages.depth_merge import build_driver as BD
    out = Path(scr) / "store" / "out"
    unfinished: dict = {}
    doc: dict = {"schema": "aadistill.dmr1.session_completion/v1", "run_id": run_id, "kind": kind, "arm": arm,
                 "scr": str(scr)}
    if kind == v.build_kind:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:          #: this tree's identity of the build's products ($0, CPU)
            ident = BD.current_identity(Path(tmp))
        rec = BD.durable_init(out, ident) if out.is_dir() else None
        if rec is None:
            p = out / BD.ARM / "init_record.json"
            unfinished["init"] = BD.product_problems(json.loads(p.read_text()) if p.is_file() else None, ident)
        doc["init_record"] = None if rec is None else str((out / BD.ARM / "init_record.json"))
    else:
        from stages.single_operator_budget import common as SC
        from stages.single_operator_budget import completion as CP
        from stages.single_operator_budget import launch as SL
        arm = v.arm_of(kind, arm)
        before = DC.bind_for(arm)
        try:
            plan = json.loads((root / plan_path(arm)).read_text())
            ses = CP.session(out, arm, plan=plan) if out.is_dir() else None
            if ses is None:
                unfinished["evidence"] = f"no secured evidence at {out}"
            else:
                bad_f = {u: f["problems"] for u, f in ses["finals"].items() if f["problems"]}
                bad_s = {k: x["problems"] for k, x in ses["sets"].items() if not x["complete"]}
                unfinished.update({k: x for k, x in (("finals", bad_f), ("sets", bad_s), ("modes", ses["modes"])) if x})
            ret = SL.retain(kind, run_id, Path(scr), arm=arm, record=root / v.logs / f"storage/finals_{arm}.json",
                            lfs_sha=lfs_sha)
            if ret["status"] != "COMPLETE":
                unfinished["final_retention"] = ret["missing"]
            doc.update(session=ses, final_placement=SC.final_placement(arm),
                       final_retention={"status": ret["status"], "missing": ret["missing"]})
        finally:
            from stages.single_operator_budget import common as SC2
            SC2.restore(before)
    doc.update(status="COMPLETE" if not unfinished else "INCOMPLETE", unfinished=unfinished,
               checked_utc=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()))
    dest = Path(root) / v.study().run_dir(run_id) / "closeout/session_completion.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return {"status": doc["status"], "unfinished": sorted(unfinished),
            "record": str(dest.relative_to(root)) if dest.is_relative_to(root) else str(dest)}


def stage_resume(v: Variant, run_id: str, src_scr: Path) -> str:
    """The resume archive to the relay's transfer area; prints ``<path>@<sha256>`` for ``--resume-from``."""
    from stages.teacher_parent_initialization import launch as TL
    from stages.teacher_parent_initialization import relay as R
    tgz = Path(src_scr) / f"resume_{run_id}.tgz"
    TL.resume_archive(Path(src_scr), tgz)
    sha = R.sha256_file(tgz)
    rec = R.run_online("put", {"files": {tgz.name: str(tgz)}, "prefix": f"{relay_transfer(v)}/{run_id}",
                               "message": f"{v.key} {run_id}: resume archive", "scratch": str(src_scr)},
                       timeout_s=3600)
    if not rec.get("verified"):
        raise SE.AuthorizationError(f"the resume archive did not verify on the relay: {str(rec)[:400]}")
    return f"{relay_transfer(v)}/{run_id}/{tgz.name}@{sha}"


# ---- the CLI ------------------------------------------------------------------------------------------------------

def _pop(argv: list[str], flag: str) -> tuple[list[str], str | None]:
    if flag in argv:
        i = argv.index(flag)
        val = argv[i + 1] if i + 1 < len(argv) else None
        return argv[:i] + argv[i + 2:], val
    return argv, None


def main(v: Variant, authorization: type, argv=None, *, description: str = "") -> int:
    import argparse
    st = v.study()
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, kind = _pop(argv, "--kind")
    argv, minutes = _pop(argv, "--minutes")
    argv, resume = _pop(argv, "--resume-from")
    argv, arm = _pop(argv, "--arm")
    kind = kind or v.kinds[0]
    if argv and argv[0] == "budget":
        try:
            out = budget(v, kind, arm)
        except SE.AuthorizationError as exc:
            print(f"REFUSED at $0: {exc}")
            return 11
        print(json.dumps(out, indent=1, default=str))
        return 0 if out["fundable"] else 11
    if argv and argv[0] == "stage-resume":
        ap = argparse.ArgumentParser(prog=f"{v.key} stage-resume")
        ap.add_argument("--run-id", required=True)
        ap.add_argument("--from", dest="src", required=True)
        a = ap.parse_args(argv[1:])
        print(stage_resume(v, a.run_id, Path(a.src)))
        return 0
    if argv and argv[0] in ("issue", "launch"):
        try:
            SE.require_open(st)
            SE.require_kind_open(st, kind)
        except SE.AuthorizationError as exc:
            print(f"REFUSED at $0: {exc}")
            return 11
    if argv and argv[0] == "launch" and kind in v.train_kinds:
        def val(flag):
            return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None
        path = val("--authorization") or (str(REPO / st.authorization_path(val("--run-id"))) if val("--run-id") else None)
        sc = json.loads(Path(path).read_text()).get("stage_conditions") if path and Path(path).is_file() else None
        bad = arm_binding_problems(v, sc or {}, v.arm_of(kind, arm)) if sc is not None else []
        if bad:
            print(f"REFUSED at $0: the artifact does not fund the plan this launch would run: {bad}\n"
                  "No provider resource was created.")
            return 11
    if argv and argv[0] == "grant":
        ap = argparse.ArgumentParser(prog=f"{v.key} grant --kind {kind}")
        ap.add_argument("--decision", required=True)
        ap.add_argument("--granted-by", required=True)
        a = ap.parse_args(argv[1:])
        print(SE.write_grant(st, decision=a.decision, granted_by=a.granted_by, kind=kind))
        return 0
    if argv and argv[0] == "issue":
        from stages.reasoning_inheritance.layer_merge_launch import reserve_problems
        ap = argparse.ArgumentParser(prog=f"{v.key} issue")
        ap.add_argument("--run-id", required=True)
        ap.add_argument("--expected-usd", type=float, required=True)
        ap.add_argument("--hard-cap-usd", type=float, required=True)
        ap.add_argument("--dry-run", action="store_true")
        ap.add_argument("--out", default=None)
        ap.add_argument("--note", default="")
        x = ap.parse_args(argv[1:])
        try:
            a_ = v.arm_of(kind, arm)
            g = kind_grant(v, kind, dry_run=x.dry_run)
            cap = session_terms(v, kind, a_, g).get("per_session_hard_cap_usd")
            if cap is None:
                raise SE.AuthorizationError(f"{g['_source']} names no cap for the {kind} kind")
            if x.hard_cap_usd > float(cap) + 1e-9:
                raise SE.AuthorizationError(f"${x.hard_cap_usd} exceeds the {kind} cap ${cap}")
            bad = reserve_problems(x.hard_cap_usd, g, SE.envelope(st, g, REPO, kind), SE.live_position(REPO))
            if bad:
                raise SE.AuthorizationError("; ".join(bad))
            path = SE.issue(st, x.run_id, expected_usd=x.expected_usd, hard_cap_usd=x.hard_cap_usd, dry_run=x.dry_run,
                            out=Path(x.out) if x.out else None, note=x.note, kind=kind,
                            conditions={**kind_conditions(v, kind, arm=a_), "resume_from": resume or ""})
        except (SE.AuthorizationError, OSError, KeyError) as exc:
            print(f"REFUSED at $0: {exc}")
            return 11
        print(path)
        return 0
    return SE.cli(st, lambda: launch_spec(v, authorization, kind, float(minutes or 0.0), resume, arm), argv, gpu=GPU,
                  description=description)
