#!/usr/bin/env python3
"""SOB1's sessions (study SOB1): grant, one-use artifacts, launch, closeout, the resume archive and local retention.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/single_operator_budget/launch.py \\
        {issue|launch|closeout|position|grant|stage-resume|retain|budget} --kind {sob1_phase1|sob1_train} [--arm <arm>] ...

    # $0: phase 1's budget check against the live position, then a dry-run artifact outside the run directory
    ... launch.py budget --kind sob1_phase1
    ... launch.py issue --kind sob1_phase1 --run-id <id> --expected-usd <e> --hard-cap-usd <c> --dry-run --out <tmp>

`shared.study_engineering` is the mechanism; this module is the instance (TPI1's launcher, with this study's files).
Each phase has its own money (`study_engineering`'s PHASES): phase 1 the study grant
(``logs/budget/approvals/stage1_single_operator_budget_grant.json``, from ``plans/proposal.json``), phase 2 a grant of
its own (``..._phase2_grant.json``, from ``plans/proposal_phase2.json``, only once the maintainer grants it). Each
envelope counts only its own kind's runs, so phase 1's remainder funds nothing in phase 2 and phase 2 -- the Stage-3
recovery training AND its evaluation -- is budgeted once. Dollar amounts live in the proposals / grants, never here.

Two session kinds, one L40S each, on the container disk (no network volume):

* ``sob1_phase1`` -- `phase1_driver.py`: the three initializations (each durable at completion), then each untrained
  arm and the teacher evaluated at 8,192 tokens in both modes. **CLOSED** (the study grant's
  ``kinds.sob1_phase1.closed``, 2026-11-20): `issue` and `launch` refuse it, its remainder is void; `closeout`,
  `retain` and the completion check stay.
* ``sob1_train`` -- `train_driver.py` (this study's state machine on TPI1's durability helpers): one arm's two seeds
  trained and evaluated (``--arm``), each set bound to that seed's final weights. Issued only once `plan.py` has
  derived that arm's plan from a COMMITTED init record whose relay copies verify, and only under phase 2's grant,
  inside its per-arm cap (``kinds.sob1_train.per_arm``).

Gates read at $0 before the create, beyond the mechanism's: the authorization is in the session commit; the cap plus
the teardown reserve fit (`layer_merge_launch.reserve_problems`); the relay would accept the kind's bytes at their true
sizes; the artifact's bound records are this tree's; for a training session, the plan is the derivation and the
arm's initialization is on the relay with its hashes; the dev box's filesystem(s) under ``--scr`` and the local store
hold the session's bound peak (`disk.check_room`).

Finished files leave the pod on every poll (`reasoning_inheritance.launch.secure_incremental`: an unreadable listing is
retried and leaves "pending" unknown, never 0); at teardown every file of the pod's final listing is checked by sha256
against the dev box's copy (`final_hash_check`). Every closeout of a phase-1 session -- the one the formal ``launch``
runs after teardown and a manual ``closeout`` -- then runs `phase1_completion` (`Launch.completion_check`): the eight
sets on that copy (`completion.phase`) and ``retain``, which brings EVERY arm's initialization and mapping home as its
manifest, re-hashed on the disk. Phase 1 is recorded COMPLETE only when both hold
(``runs/<run>/closeout/phase1_completion.json``); otherwise INCOMPLETE with what is unfinished. A training session's
closeouts run `phase2_completion` the same way: the arm's two finals and four sets on the secured copy
(`completion.session`) and both finals at their declared stores (`retain`), then the study's completion
(`study_completion`: six finals, twelve sets, the registered comparison recomputed from the sealed records) once every
arm has a COMPLETE session (``runs/<run>/closeout/phase2_completion.json``, ``results/study_completion.json``).
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

PHASE2_GRANT = Path("logs/budget/approvals/stage1_single_operator_budget_phase2_grant.json")
PHASE2_PROPOSAL = LOGS / "plans/proposal_phase2.json"

STUDY = SE.Study(
    key="SOB1", plan_id="stage1.single_operator_budget", logs=LOGS,
    grant=Path("logs/budget/approvals/stage1_single_operator_budget_grant.json"),
    proposal=LOGS / "plans/proposal.json",
    kind_grants=((TRAIN_KIND, PHASE2_GRANT, PHASE2_PROPOSAL),),
    identity_files=(DESIGN, SELECTION, STRUCTURES),
    scripts=(f"{HERE}/__init__.py", f"{HERE}/common.py", f"{HERE}/phase1_driver.py", f"{HERE}/train_driver.py",
             f"{HERE}/plan.py", f"{HERE}/selection.py", f"{HERE}/structures.py", f"{HERE}/launch.py",
             f"{HERE}/completion.py", f"{HERE}/analysis.py", f"{HERE}/disk.py",
             f"{TPI}/__init__.py", f"{TPI}/init_driver.py", f"{TPI}/train_driver.py", f"{TPI}/relay.py",
             f"{TPI}/plan.py", f"{TPI}/selection.py", f"{TPI}/launch.py", f"{TPI}/analysis.py", f"{TPI}/b_evidence.py",
             "scripts/stages/stage-1/reasoning_inheritance/analysis.py",
             "scripts/stages/stage-1/reasoning_inheritance/layer_merge_analysis.py",
             "scripts/stages/stage-0/calibration_volume/score_battery.py",
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


def kind_grant(kind: str, *, dry_run: bool = False) -> dict:
    """The grant that funds ``kind`` (phase 2's own; on a dry run its proposal), refusing a closed kind."""
    return SE.grant(STUDY, REPO, dry_run=dry_run, kind=kind)


def session_terms(kind: str, arm: str | None, g: dict) -> dict:
    """One session's terms from the grant that funds it: phase 1's kind block, or phase 2's arm block
    (``kinds.sob1_train.per_arm.<arm>``: expected and hard cap in USD, the driver minutes, the unit deadline)."""
    k = (g.get("kinds") or {}).get(kind) or {}
    if kind == TRAIN_KIND:
        if arm is None:
            raise SE.AuthorizationError("a training session names its arm (--arm)")
        return (k.get("per_arm") or {}).get(arm) or {}
    return k


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
    """What the relay must accept on top of what it holds, at the session's peak (bytes, from structures.json):
    phase 1 the three bf16 inits and their mappings. A training session in its real rotation order: a save point is
    removed only after the next one hands off (two at once), a unit's final is uploaded while its last save point is
    still there, and the save points go only after the final hands off. Seed 1 peaks at two save points; seed 2 at
    seed 1's final plus two save points, then at seed 1's final, its last save point and its own final. So the peak
    is ONE final and TWO save points: ``[save_point, save_point, final]`` (`rotation_peaks` shows each step)."""
    s = json.loads((REPO / STRUCTURES).read_text())["arms"]
    if kind == PHASE1_KIND:
        return [s[a]["checkpoint_bytes"]["init_bf16"] for a in ARMS] + [64 * 2 ** 20] * len(ARMS)
    b = s[arm]["checkpoint_bytes"]
    return [b["save_point"], b["save_point"], b["final_fp32"]]


def rotation_need(plan: dict, state: dict, sizes: dict) -> dict:
    """What the relay must still accept, as the remaining rotation of the arm's units would run from ``state`` (what
    the relay already holds, `relay_state`), in the driver's order: each later save point is uploaded while every
    earlier one is still there, then they are removed (a held superseded one is credited only then); the final is
    uploaded while the last save point is there, then the save points go. A unit whose final is held uploads nothing,
    and a save point still held beside it is credited nothing: it is removed before the launch (`reclaim_superseded`)
    or it stays in the quota. Bytes already held are in the quota already: the need is the peak of (bytes uploaded -
    bytes removed) over the sequence, against the relay's LIVE headroom (which already excludes them). The latest
    resumable save point is never removed before a later one is uploaded."""
    sp, fin = int(sizes["save_point"]), int(sizes["final_fp32"])
    net, peak, trace = 0, 0, []

    def step(what: str, delta: int):
        nonlocal net, peak
        net += delta
        peak = max(peak, net)
        trace.append({"step": what, "net_bytes": net})
    for name in plan["unit_order"]:
        u = next(x for x in plan["units"] if x["unit"] == name)
        st = state.get(name) or {}
        held = sorted(st.get("save_points") or [])         #: steps of the valid save points the relay holds
        if st.get("final"):
            continue                    #: uploads nothing; a save point still held is never credited before it goes
        latest = held[-1] if held else 0
        every, total = int(u["config"]["checkpoint"]["save_every"]), int(u["config"]["schedule"]["total_steps"])
        #: every held save point stays until the next upload hands off (the mirror then removes all older ones): the
        #: superseded ones are credited only then, never before -- the need is never under-asked
        current = list(held)
        for k in range(every, total, every):
            if k <= latest:
                continue
            step(f"{name}: save point {k} uploaded", sp)
            for _ in current:
                step(f"{name}: save point before {k} removed", -sp)
            current = [k]
        step(f"{name}: final uploaded", fin)
        for _ in current:
            step(f"{name}: save points retired after the final", -sp)
    return {"peak_bytes": peak, "peak_gib": round(peak / 2 ** 30, 2), "end_bytes": net, "trace": trace,
            "rule": "the peak of bytes uploaded minus bytes removed over the remaining rotation (rotation_need)"}


def relay_state(arm: str, *, list_records=None, lfs_sha=None, plan: dict | None = None) -> dict:
    """What the relay holds for the arm's units NOW ($0, read-only): each unit's save points and final whose record is
    verified, whose files are all listed, and whose LFS objects carry the record's sha256 (a size or hash that differs
    is not reusable and is reported, never counted). ``plan``: the arm's training plan (the committed SOB1 one unless
    given: a study binding this chain passes its own)."""
    from stages.teacher_parent_initialization import train_driver as TD
    plan = plan if plan is not None else json.loads((REPO / plan_path(arm)).read_text())
    list_records = list_records or (lambda prefix: TD.list_records(prefix, TD.durability(plan)))
    lfs_sha = lfs_sha or _lfs_sha
    out = {}
    for name in plan["unit_order"]:
        up = f"{plan['relay_prefix_root']}/{arm}/{name}"
        row = {"save_points": [], "final": False, "not_reusable": []}
        for kind in ("savepoints", "final"):
            for x in list_records(f"{up}/{kind}").get("records", []):
                rec = x["record"]
                big = {f"{rec['prefix']}/{k}": v for k, v in (rec.get("files_sha256") or {}).items()
                       if k.endswith((".safetensors", ".pt"))}
                ok = bool(x.get("present") and rec.get("verified") and big)
                got = lfs_sha(list(big)) if ok else {}
                bad = [q for q, v in big.items() if got.get(q) != v]
                if not ok or bad:
                    row["not_reusable"].append({"prefix": rec.get("prefix"), "why": "not present or not verified"
                                                if not ok else f"LFS sha256 differs at {bad}"})
                elif kind == "final":
                    row["final"] = True
                else:
                    row["save_points"].append(int(rec["step"]))
        out[name] = row
    return out


def _lfs_sha(paths: list[str]) -> dict[str, str]:
    from huggingface_hub import HfApi
    from shared.deployment import MAIN_RELAY
    tok = Path(os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()
    info = HfApi(token=tok).get_paths_info(MAIN_RELAY, paths, repo_type="model")
    return {i.path: getattr(getattr(i, "lfs", None), "sha256", None) for i in info}


RECLAIM_RECORD = LOGS / "storage/relay_reclaim_phase2.json"


def _listed_problem(x: dict, *, up: str, kind: str, every: int, total: int, lfs_sha) -> str | None:
    """Why a listed record is not a valid object of this unit (None when it is): present with every file its pointer
    names, verified, at this unit's path (a save point: a planned step under ``<up>/savepoints/<tag>``), and each weight
    object's LFS sha256 the record's."""
    rec = x["record"]
    big = {f"{rec.get('prefix')}/{k}": v for k, v in (rec.get("files_sha256") or {}).items()
           if k.endswith((".safetensors", ".pt"))}
    if not (x.get("present") and rec.get("verified") and big):
        return "not present with every file, or not verified"
    if kind == "savepoints":
        step = int(rec.get("step", -1))
        if (rec.get("tag") != f"step_{step:06d}" or rec.get("prefix") != f"{up}/savepoints/{rec.get('tag')}"
                or not 0 < step < total or step % every):
            return f"not one of this unit's planned save points ({rec.get('prefix')}, step {step})"
    elif not str(rec.get("prefix", "")).startswith(f"{up}/final/"):
        return f"not under this unit's final path ({rec.get('prefix')})"
    got = lfs_sha(list(big))
    bad = [q for q, v in big.items() if got.get(q) != v]
    return f"LFS sha256 differs at {bad}" if bad else None


def reclaim_superseded(arm: str, *, list_records=None, lfs_sha=None, root: Path = REPO, write: bool = True,
                       plan: dict | None = None, record: Path | None = None) -> dict:
    """Before a training launch, on the dev box ($0): a save point the rotation should already have removed -- the
    driver removes it right after a later one hands off, and a failed removal leaves it PENDING on the relay -- is
    removed here, so the room check reads the relay as the takeover will find it and the pod-side retry is never the
    only way out. Each unit's save points are read from the relay listing (`_listed_problem`); the one a takeover
    restores (`train_driver.restore_savepoint`: the latest listed and present) is checked first: valid, with the same
    resume identity (``resume_identity.json``'s sha256: the same data and initialization) as each one it supersedes.
    Only then is each older valid save point removed (`relay.remove`, permanent, its pointer with it) and confirmed
    absent. Every save point of a unit whose final is listed, valid and this unit's (`completion.final_identity_problems`
    on its archived completion record) is superseded the same way. The latest valid save point of a unit without such a
    final always stays; an invalid listed object is left for diagnosis and never counted. A removal that does not
    confirm stays held and counted. Writes ``storage/relay_reclaim_phase2.json``."""
    import tempfile
    from stages.single_operator_budget import completion as CP
    from stages.teacher_parent_initialization import relay as R
    from stages.teacher_parent_initialization import train_driver as TD
    plan = plan if plan is not None else json.loads((REPO / plan_path(arm)).read_text())
    list_records = list_records or (lambda prefix: TD.list_records(prefix, TD.durability(plan)))
    lfs_sha = lfs_sha or _lfs_sha
    units = {u["unit"]: u for u in plan["units"]}
    rows = {}
    for name in plan["unit_order"]:
        cfg = units[name]["config"]
        every, total = int(cfg["checkpoint"]["save_every"]), int(cfg["schedule"]["total_steps"])
        up = f"{plan['relay_prefix_root']}/{arm}/{name}"
        chk = functools.partial(_listed_problem, up=up, every=every, total=total, lfs_sha=lfs_sha)
        row = {"kept": None, "removed": [], "not_removed": [], "invalid": []}
        sps = []
        for x in list_records(f"{up}/savepoints").get("records", []):
            why = chk(x, kind="savepoints")
            (row["invalid"].append({"prefix": x["record"].get("prefix"), "why": why}) if why else sps.append(x["record"]))
        fins = [x["record"] for x in list_records(f"{up}/final").get("records", []) if not chk(x, kind="final")]
        final_ok = None
        if len(fins) == 1 and sps:
            rec = fins[0]
            names = [k for k in rec.get("files_sha256", {}) if k.startswith("run_completion")]
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / name).mkdir()
                (Path(tmp) / name / "final.json").write_text(json.dumps(rec))
                got = (R.run_online("get", {"record": rec, "dest": f"{tmp}/archived", "only": names}, timeout_s=600)
                       if len(names) == 1 else {"verified": False, "error": f"{len(names)} completion records"})
                bad = (CP.final_identity_problems(Path(tmp), name, plan=plan, archived_record=Path(tmp) / "archived" /
                                                  names[0]) if got.get("verified") else [str(got.get("error"))[:300]])
            final_ok = not bad
            row["final"] = {"prefix": rec.get("prefix"), "identity_problems": bad}
        if final_ok:
            keep, older = None, sorted(sps, key=lambda r: int(r["step"]))
            row["kept"] = {"final": fins[0]["prefix"]}
        elif sps:
            keep = max(sps, key=lambda r: int(r["step"]))
            older = sorted((r for r in sps if int(r["step"]) < int(keep["step"])), key=lambda r: int(r["step"]))
            row["kept"] = {"tag": keep["tag"], "step": int(keep["step"]), "prefix": keep["prefix"],
                           "why": "the latest valid save point: the one a takeover restores"}
        else:
            older = []
        for o in older:
            ident = (o.get("files_sha256") or {}).get("resume_identity.json")
            if keep is not None and (ident is None or ident != (keep.get("files_sha256") or {}).get("resume_identity.json")):
                row["not_removed"].append({"tag": o["tag"], "why": "its resume identity is not the kept save point's"})
                continue
            got = R.run_online("remove", {"record": o, "message": f"{name}: {o['tag']} superseded by "
                                                     f"{keep['tag'] if keep else 'the final'} (dev-box reclaim)"},
                               timeout_s=3600)
            (row["removed"] if got.get("verified_absent") else row["not_removed"]).append(
                {"tag": o["tag"], "step": int(o["step"]), "verified_absent": bool(got.get("verified_absent")),
                 "freed_bytes": got.get("freed_bytes"), "error": got.get("error"), "utc": _utc()})
        rows[name] = row
    doc = {"schema": "aadistill.stage1.single_operator_budget.relay_reclaim/v1", "arm": arm, "units": rows,
           "checked_utc": _utc()}
    if write and any(r["removed"] or r["not_removed"] for r in rows.values()):
        p = Path(record) if record is not None else Path(root) / RECLAIM_RECORD
        hist = json.loads(p.read_text()) if p.is_file() else {"schema": doc["schema"], "reclaims": []}
        hist["reclaims"].append(doc)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(hist, indent=1, default=str) + "\n")
    return doc


def rotation_peaks(arm: str) -> list[dict]:
    """What a training session holds on the relay at each step of its rotation (bytes), seed 1 then seed 2."""
    b = json.loads((REPO / STRUCTURES).read_text())["arms"][arm]["checkpoint_bytes"]
    sp, fin = b["save_point"], b["final_fp32"]
    steps = [("seed 1: a save point uploads while the previous one is still listed", 2 * sp),
             ("seed 1: the final uploads while the last save point is still listed", sp + fin),
             ("seed 1: save points retired after the final hands off", fin),
             ("seed 2: a save point uploads while the previous one is listed, beside seed 1's final", fin + 2 * sp),
             ("seed 2: its final uploads beside its last save point and seed 1's final", 2 * fin + sp),
             ("seed 2: save points retired", 2 * fin)]
    return [{"step": w, "bytes": n, "gib": round(n / 2 ** 30, 2)} for w, n in steps]


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
        g = kind_grant(_kind(ctx), dry_run=getattr(ctx.args, "dry_run", False))
        env = SE.envelope(STUDY, g, REPO, _kind(ctx))
        pos = ctx.evidence.get("live_position") or SE.live_position(REPO)
    except SE.AuthorizationError as exc:
        return False, str(exc)
    bad = reserve_problems(float(ctx.auth.hard_cap_usd), g, env, pos)
    return (not bad, f"${ctx.auth.hard_cap_usd:.2f} + the ${g.get('teardown_reserve_usd')} teardown reserve fit"
            if not bad else "; ".join(bad))


def chunks(n: int, most: int = 10 * 2 ** 30) -> list[int]:
    """``n`` bytes as objects of at most ``most`` (the hub refuses one object of 50 GB or more by its own limit)."""
    k = max(1, -(-int(n) // most))
    return [int(n) // k] * (k - 1) + [int(n) - (int(n) // k) * (k - 1)]


def the_relay_has_room(ctx) -> tuple[bool, str]:
    """Phase 1: its three inits. A training session: first the superseded save points a lost pod left are removed
    (`reclaim_superseded`, real and dry runs alike: it completes the rotation's own step, never touches the latest
    valid save point, and creates nothing); then what the REMAINING rotation will still add (`rotation_need`) from
    what the relay holds now (`relay_state`), asked of the relay's live headroom: a first run asks for two save points
    and a final; a takeover only for what is left, crediting the valid objects already there."""
    from aadistill.runtime.hub_capacity import would_accept
    from shared.deployment import MAIN_RELAY
    tok = Path(os.path.expanduser("~/.cache/huggingface/token"))
    if not tok.is_file():
        return False, "no hub token on the dev box to ask the relay"
    if _kind(ctx) != TRAIN_KIND:
        v = would_accept(MAIN_RELAY, relay_object_sizes(_kind(ctx), _arm(ctx)), tok.read_text().strip())
        ctx.evidence["relay_room"] = str(v)
        return v.accepted, f"the relay would accept the {_kind(ctx)} session's bytes: {v}"
    arm = _arm(ctx)
    plan = json.loads((REPO / plan_path(arm)).read_text())
    sizes = json.loads((REPO / STRUCTURES).read_text())["arms"][arm]["checkpoint_bytes"]
    #: what the rotation should already have removed goes first (a pending removal the lost pod left), so the
    #: remaining need is read from the relay a takeover will find, against its live headroom
    ctx.evidence["relay_reclaim"] = reclaim_superseded(arm)
    state = relay_state(arm)
    need = rotation_need(plan, state, sizes)
    ctx.evidence["relay_state"], ctx.evidence["relay_need"] = state, {k: v for k, v in need.items() if k != "trace"}
    if need["peak_bytes"] <= 0:
        return True, f"the remaining rotation adds nothing beyond what the relay holds ({state})"
    v = would_accept(MAIN_RELAY, chunks(need["peak_bytes"]), tok.read_text().strip())
    ctx.evidence["relay_room"] = str(v)
    return v.accepted, (f"the remaining rotation needs {need['peak_gib']} GiB more at its peak (held now: "
                        f"{ {k: (r['save_points'], r['final']) for k, r in state.items()} }): {v}")


def arm_binding_problems(sc: dict, arm: str | None, root: Path = REPO) -> list[str]:
    """A training artifact funds ONE arm's plan: what this launch would run (``--arm``: its plan path, the plan's and
    its init record's sha256, on this tree) must be what the artifact binds. Empty when they agree."""
    if (sc or {}).get("kind") != TRAIN_KIND:
        return []
    if arm is None:
        return ["the launch names no --arm"]
    try:
        want = kind_conditions(TRAIN_KIND, root, arm)
    except (OSError, KeyError, ValueError, SE.AuthorizationError) as exc:
        return [f"{arm}'s plan is not on this tree: {exc}"]
    return [f"{k}: the artifact binds {sc.get(k)!r}, this launch's --arm {arm} runs {want[k]!r}"
            for k in ("arm", "plan", "plan_sha256", "init_record_sha256") if sc.get(k) != want[k]]


def launch_arm_gate(arm: str | None):
    """The runner's gate for the launch's own arm (bound when the launch is built from ``--arm``)."""
    def the_launch_runs_the_authorized_arm_and_plan(ctx) -> tuple[bool, str]:
        bad = arm_binding_problems(getattr(ctx.auth, "stage_conditions", None) or {}, arm)
        return (not bad, f"the artifact funds the plan this launch runs ({arm})" if not bad else "; ".join(bad))
    return the_launch_runs_the_authorized_arm_and_plan


def artifact_conditions(argv: list[str]) -> dict | None:
    """The stage conditions of the artifact a ``launch`` command line would read: ``--authorization`` (a dry run),
    else the run's own one-use artifact; None when there is none to read (the launcher refuses that itself)."""
    def val(flag):
        return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None
    path = val("--authorization") or (str(REPO / STUDY.authorization_path(val("--run-id"))) if val("--run-id") else None)
    if not path or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text()).get("stage_conditions") or {}


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


def statistics_input():
    """The residual-width build's input: TPI1's sealed teacher statistics, staged by setup and verified there by the
    sha256 `selection.json` derives from TPI1's committed init record (no teacher statistics pass on the pod)."""
    from aadistill.infrastructure.session import RelayInput
    st = C.statistics_input()
    return RelayInput(path=st["path"], repo=st["repo"], dest=st["dest"], sha256=st["sha256"])


def the_dev_box_has_room(ctx) -> tuple[bool, str]:
    """Where this session's bytes land on the dev box (``--scr`` and the local store), against the free bytes of the
    filesystem each is on: the bound peak of `disk.plan` (evidence, archives, then the retained weights)."""
    from stages.single_operator_budget import disk as DK
    scr = getattr(ctx.args, "scr", None) or str(C.local_store() / ctx.args.run_id)
    room = DK.check_room(_kind(ctx), _arm(ctx), scr)
    ctx.evidence["dev_box_room"] = room
    fs = "; ".join(f"{e['read_at']}: {e['free_bytes'] / DK.GIB:.2f} GiB free for {e['bound_bytes'] / DK.GIB:.2f} GiB "
                   f"(margin {e['margin_gib']} GiB)" for e in room["filesystems"])
    return room["fits"], (f"the dev box holds the session's bound peak: {fs}" if room["fits"] else
                          f"the dev box cannot hold what this session brings back: {fs}")


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
        hours = session_terms(kind, arm, kind_grant(kind, dry_run=getattr(ctx.args, "dry_run", False)))[
            "unit_deadline_hours"]
        cmd = (f"{DRIVERS[kind]} --authorization {auth} --out {ev} --status {status} --plan {plan_path(arm)} "
               f"--prefix-root {C.relay_root()} --unit-deadline-hours {hours} "
               + (f"--session-deadline-utc {deadline} " if deadline else "")
               + f"--vllm-python /opt/vllm/bin/python --attempt {shlex.quote(ctx.args.run_id)} {resume}")
    return (f"mkdir -p {ev} && {py} {cmd.strip()}; "
            f"grep -qE 'MARKER:(ALL_DONE|RUN_FAILED)' {status} || "
            f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:RUN_FAILED\" >> {status}")


def driver_minutes(kind: str, arm: str | None = None) -> float:
    """The driver's bound for the session: phase 1's proposal; a training session its arm's terms in phase 2's grant
    (its proposal before a grant exists: a dry run prices the same plan)."""
    if kind == PHASE1_KIND:
        return float(proposal()["kinds"][kind]["driver_minutes"])
    try:
        g = kind_grant(kind)
    except SE.AuthorizationError:
        g = json.loads((REPO / PHASE2_PROPOSAL).read_text())
    return float(session_terms(kind, arm, g)["driver_minutes"])


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
        relay = (statistics_input(),)                     #: TPI1's sealed teacher statistics, at hub speed
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
                         the_kind_records_are_the_bound_ones, the_initialization_is_durable, the_dev_box_has_room,
                         *((launch_arm_gate(arm),) if kind == TRAIN_KIND else ())),
        validation_minutes=float(minutes or driver_minutes(kind, arm)), setup_minutes=25.0, contingency_fraction=0.05,
        description=desc, artifact_spec="configs/stages/stage-1/single_operator_budget/artifacts.json",
        teardown_note=("delete the pod, verify from the provider that it is gone, STOP; every finished file is on the "
                       "dev box, every weight on the relay; nothing stays on the provider"))
    return dataclasses.replace(launch, command=functools.partial(command, launch, kind, arm),
                               on_poll=RIL.secure_incremental(launch), products_secured=RIL.products_secured(launch),
                               completion_check=(phase1_completion if kind == PHASE1_KIND else
                                                 functools.partial(phase2_completion, arm=arm)))


# ---- the budget check, resume and retention ($0, dev box) ------------------------------------------------------------

def budget(kind: str, arm: str | None = None, *, snapshot: dict | None = None) -> dict:
    """The kind's cap against what funds it NOW: the grant's cap for that kind (phase 2's own grant, its proposal
    before one exists), the teardown reserve, that kind's envelope, the project cap's remainder and the provider
    balance, read live (fail closed). A closed kind refuses (`SE.require_kind_open`): its remainder is void."""
    from stages.reasoning_inheritance.layer_merge_launch import reserve_problems
    pos = SE.live_position(REPO, snapshot=snapshot)
    closed = SE.kind_closure(STUDY, kind, REPO)
    if closed:
        return {"kind": kind, "arm": arm, "closed": closed, "position": pos, "fundable": False, "remaining_usd": 0.0,
                "problems": [f"the {kind} kind is CLOSED ({closed.get('utc') if isinstance(closed, dict) else closed}): "
                             "nothing is issued or launched and its remainder is void"]}
    g = kind_grant(kind, dry_run=True)
    k = session_terms(kind, arm, g)
    cap = k.get("per_session_hard_cap_usd")
    env = SE.envelope(STUDY, g, REPO, kind)
    out = {"kind": kind, "arm": arm, "source": g["_source"], "status": g.get("status"),
           "expected_usd": k.get("expected_usd"), "hard_cap_usd": cap, "teardown_reserve_usd": g.get("teardown_reserve_usd"),
           "envelope": env, "position": pos, "project_cap_usd_this_grant_states": g.get("project_cumulative_cap_usd")}
    if cap is None:
        out["problems"] = [f"{g['_source']} names no cap for {kind}{f' {arm}' if arm else ''}: "
                           f"{k.get('status') or 'not proposed'}"]
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


def retain(kind: str, run_id: str, src_scr: Path, *, store: Path | None = None, record: Path | None = None,
           getter=None, arm: str | None = None, lfs_sha=None) -> dict:
    """Each durable product of a finished session onto the dev box ($0 GPU), from its verified relay record, into the
    out-of-tree store (`relay.get`; ``getter`` for a test). Phase 1: EVERY arm's initialization and mapping, each
    exactly its init record's manifest (`common.products`: the checkpoint's two files, ``mapping.json``, and the
    tensor file only for the declared width projection), refused unless the record shows it durable
    (`common.product_problems`); a training session: both seeds' final weights. Each file is hashed again here, on
    the disk, against the manifest -- never taken from the fetch's own answer, and an empty list proves nothing. The
    record is ``COMPLETE`` only when every product is on the dev box with its sha256; an arm without an init record,
    a fetch that fails, a filesystem without room or a file whose bytes differ leaves it ``INCOMPLETE``, naming what
    is missing. Verified products are recorded either way, so a later retain fetches only the rest. An arm whose finals
    are placed on the relay (`common.final_placement`) lands nothing: each final is checked where it stays -- every
    file of its record listed at its relay prefix and each weight object's LFS sha256 the record's (``lfs_sha``)."""
    import shutil
    from stages.teacher_parent_initialization import relay as R
    getter = getter or R.get
    store = Path(store or C.local_store())
    record = Path(record or (REPO / (LOCAL_RECORD if kind == PHASE1_KIND else LOCAL_RECORD_PHASE2)))
    out = Path(src_scr) / "store" / "out"
    doc = json.loads(record.read_text()) if record.is_file() else {
        "schema": "aadistill.stage1.single_operator_budget.local/v1", "store": str(store), "products": {}}
    want, missing = {}, {}                      #: key -> (the relay record, the manifest the disk must hold)
    if kind == PHASE1_KIND:
        for a in C.design()["phase_1"]["build_order"]:
            p = out / a / "init_record.json"
            ir = json.loads(p.read_text()) if p.is_file() else None
            bad = C.product_problems(ir, a)
            missing.update(bad)
            for g, files in (C.products(ir) if ir else {}).items():
                if f"{a}/{g}" not in bad:
                    want[f"{a}/{g}"] = (ir["relay"][g], files)
    else:
        for sd in C.design()["training"]["seeds"]:
            name = f"{arm}-s{int(sd)}"
            fin = out / name / "final.json"
            rec = json.loads(fin.read_text()) if fin.is_file() else None
            if not (rec or {}).get("verified") or not (rec or {}).get("files_sha256"):
                missing[f"{name}/final"] = "no verified final record with its files in the secured evidence"
            else:
                want[f"{name}/final"] = (rec, rec["files_sha256"])
    #: any training kind (SOB1's, or another study's binding this chain): its finals at the design's placement
    on_relay = kind != PHASE1_KIND and C.final_placement(arm) == "relay"
    for key, (rec, files) in (want.items() if on_relay else ()):
        paths = {f"{rec['prefix']}/{k}": v for k, v in files.items()}
        got = (lfs_sha or _lfs_sha)(list(paths))
        bad = sorted(q for q, v in paths.items() if q not in got or (q.endswith((".safetensors", ".pt")) and got[q] != v))
        doc["products"][key] = {"kept_on": "relay", "relay": rec["relay"], "prefix": rec["prefix"],
                                "files_sha256": files, "relay_lfs_sha256": got, "verified": not bad,
                                "producer_run": run_id, "checked": "every file listed at its relay prefix and each "
                                "weight object's LFS sha256 the record's (placement: relay)"}
        if bad:
            missing[key] = f"not on the relay as recorded: {bad}"
    for key, (rec, files) in (() if on_relay else want.items()):
        dest = store / key / rec["prefix"].rsplit("/", 1)[-1]
        paths = {k: dest / k for k in files}
        if files and all(p.is_file() and SE.sha256_file(p) == files[k] for k, p in paths.items()):
            prev = doc["products"].get(key) or {}             #: on the dev box already, re-hashed: nothing fetched
            doc["products"][key] = {**prev, "path": str(dest), "files_sha256": files, "files_on_disk_sha256": files,
                                    "verified": True, "relay": rec["relay"], "prefix": rec["prefix"],
                                    "producer_run": prev.get("producer_run", run_id)}
            continue
        need = sum(int(v) for k, v in (rec.get("bytes") or {}).items() if k in paths and not paths[k].is_file())
        dest.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(dest).free < need:
            missing[key] = f"no room: {need} B to fetch, {shutil.disk_usage(dest).free} B free at {dest}"
            continue
        try:
            got = getter(rec, dest)
        except Exception as exc:  # noqa: BLE001
            got = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        on_disk = {k: (SE.sha256_file(p) if p.is_file() else None) for k, p in paths.items()}
        ok = bool(files) and on_disk == files
        doc["products"][key] = {"path": str(dest), "files_sha256": files, "files_on_disk_sha256": on_disk,
                                "verified": ok, "relay": rec["relay"], "prefix": rec["prefix"], "producer_run": run_id,
                                "fetched": got, "checked": "every file of the manifest re-hashed on the dev box's "
                                                           "disk"}
        if not ok:
            missing[key] = got.get("error") or "the bytes on the disk are not the manifest's"
    for key in missing:                         #: a product no longer shown durable is not verified here either
        if key in doc["products"] and key not in want:
            doc["products"][key]["verified"] = False
    doc.update(status="COMPLETE" if not missing else "INCOMPLETE", missing=missing,
               expected_products=sorted(set(want) | set(missing)), last_run=run_id,
               checked_utc=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()))
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


PHASE1_COMPLETION = "closeout/phase1_completion.json"


def phase1_completion(run_id: str, scr: Path, *, store: Path | None = None, record: Path = REPO / LOCAL_RECORD,
                      getter=None, root: Path = REPO) -> dict:
    """Phase 1's scientific completion, which every closeout of a ``sob1_phase1`` session runs (`Launch.
    completion_check`: the formal launch's and a manual ``closeout`` alike), after the pod is gone and the account
    reconciled. COMPLETE only when the eight sets are complete on the dev box's secured copy (`completion.phase`:
    coverage, records, protocol, the formal report, each part's provenance, one protocol per mode) AND every arm's
    initialization and mapping is on the dev box's disk as its manifest (`retain`). Otherwise INCOMPLETE, every
    unfinished item named; what is durable stays recorded for the next attempt's takeover. Writes
    ``runs/<run_id>/closeout/phase1_completion.json``."""
    from stages.single_operator_budget import completion as CP
    out = Path(scr) / "store" / "out"
    unfinished: dict = {}
    ph = CP.phase(out) if out.is_dir() else None
    if ph is None:
        unfinished["evidence"] = f"no secured evidence at {out}"
    else:
        sets = {k: v["problems"] for k, v in ph["sets"].items() if not v["complete"]}
        if sets:
            unfinished["sets"] = sets
        if ph["inits"]["problems"]:
            unfinished["inits"] = ph["inits"]["problems"]
        if ph["modes"]:
            unfinished["modes"] = ph["modes"]
    ret = retain(PHASE1_KIND, run_id, Path(scr), store=store, record=record, getter=getter)
    if ret["status"] != "COMPLETE":
        unfinished["local_retention"] = ret["missing"]
    doc = {"schema": "aadistill.stage1.single_operator_budget.phase1_completion/v2", "run_id": run_id,
           "status": "COMPLETE" if not unfinished else "INCOMPLETE",
           "rule": "the eight sets complete on the secured copy and every arm's initialization and mapping on the dev "
                   "box's disk as its manifest",
           "sets": None if ph is None else {"n_complete": ph["n_complete"], "n_sets": ph["n_sets"],
                                            "per_set": ph["sets"], "modes": ph["modes"]},
           "inits_durable": None if ph is None else ph["inits"]["durable"],
           "local_retention": {"status": ret["status"], "missing": ret["missing"], "record": str(record),
                               "products": {k: {"path": v["path"], "verified": v["verified"]}
                                            for k, v in ret["products"].items()}},
           "unfinished": unfinished, "checked_on": "the dev box's secured copy and its local store",
           "checked_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())}
    dest = Path(root) / STUDY.run_dir(run_id) / PHASE1_COMPLETION
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return {"status": doc["status"], "unfinished": sorted(unfinished), "record": str(dest.relative_to(root))
            if dest.is_relative_to(root) else str(dest)}


PHASE2_COMPLETION = "closeout/phase2_completion.json"
LOCAL_RECORD_PHASE2 = LOGS / "storage/local_retention_phase2.json"
STUDY_RESULT = LOGS / "results/study_completion.json"
PHASE1_RUN_COMPLETION = "closeout/phase1_completion.json"


def _utc() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def phase2_completion(run_id: str, scr: Path, *, arm: str, store: Path | None = None,
                      record: Path = REPO / LOCAL_RECORD_PHASE2, getter=None, root: Path = REPO,
                      tombstones: Path | None = None, lfs_sha=None) -> dict:
    """A training session's scientific completion, which every closeout of a ``sob1_train`` session runs (the formal
    launch's and a manual ``closeout`` alike), after the pod is gone. COMPLETE only when the arm's two finals are
    durable and its four sets complete against them on the dev box's secured copy (`completion.session`) AND both
    finals are at their placement as their records (`retain`: on the dev box's disk, re-hashed; or, for an arm placed
    on the relay, there by LFS sha256). Then the relay's rotation (`rotate_relay`) and, once every arm has a COMPLETE
    session, the study's completion (`study_completion`). Writes ``runs/<run_id>/closeout/phase2_completion.json``."""
    from stages.single_operator_budget import completion as CP
    out = Path(scr) / "store" / "out"
    unfinished: dict = {}
    ses = CP.session(out, arm) if out.is_dir() else None
    if ses is None:
        unfinished["evidence"] = f"no secured evidence at {out}"
    else:
        bad_f = {u: f["problems"] for u, f in ses["finals"].items() if f["problems"]}
        bad_s = {k: v["problems"] for k, v in ses["sets"].items() if not v["complete"]}
        if bad_f:
            unfinished["finals"] = bad_f
        if bad_s:
            unfinished["sets"] = bad_s
        if ses["modes"]:
            unfinished["modes"] = ses["modes"]
    ret = retain(TRAIN_KIND, run_id, Path(scr), store=store, record=record, getter=getter, arm=arm, lfs_sha=lfs_sha)
    if ret["status"] != "COMPLETE":
        unfinished["local_retention"] = ret["missing"]
    doc = {"schema": "aadistill.stage1.single_operator_budget.phase2_completion/v1", "run_id": run_id, "arm": arm,
           "scr": str(scr), "status": "COMPLETE" if not unfinished else "INCOMPLETE",
           "rule": "both seeds' final weights durable and published, the four sets (two seeds x two modes) complete on "
                   "the secured copy against those finals, and both finals at their placement as their records (the "
                   "dev box's disk, re-hashed; or the relay, by LFS sha256)", "final_placement": C.final_placement(arm),
           "session": ses, "local_retention": {"status": ret["status"], "missing": ret["missing"], "record": str(record)},
           "unfinished": unfinished, "checked_utc": _utc()}
    dest = Path(root) / STUDY.run_dir(run_id) / PHASE2_COMPLETION
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    res = {"status": doc["status"], "arm": arm, "unfinished": sorted(unfinished),
           "record": str(dest.relative_to(root)) if dest.is_relative_to(root) else str(dest)}
    if doc["status"] == "COMPLETE" and C.design()["storage"]["phase_2_finals"].get("relay_rotation") == "retire":
        try:                                #: after the record, so a failing rotation can never lose it
            res["relay_rotation"] = rotate_relay(arm, run_id, Path(scr), json.loads(Path(record).read_text()), root=root,
                                                 tombstones=tombstones)
        except Exception as exc:  # noqa: BLE001
            res["relay_rotation"] = {"status": "FAILED", "error": f"{type(exc).__name__}: {str(exc)[:800]}"}
    if doc["status"] == "COMPLETE":
        st = study_completion(root=root)
        res["study"] = {k: st[k] for k in ("status", "unfinished", "record")}
    return res


TOMBSTONES = Path("logs/maintenance/inventories/checkpoint_tombstones.json")
ROTATION_RECORD = LOGS / "storage/relay_rotation_phase2.json"
ROTATION_DECISION = ("maintainer 2026-11-20 (logs/budget/decisions.md, 'SOB1 phase 2 fixes, storage inventory'): "
                     "「优先采用免费 relay 轮转：每组两个种子的最终权重都已在开发机逐文件核验、该组评测证据已持久化后，"
                     "再退役远端冗余副本。」")


def evidence_archive(src: Path, dest: Path) -> dict:
    """``src`` (a session's secured evidence) as ONE deterministic archive under ``out/`` (sorted paths, mtime 0,
    uid / gid 0, mode 644, gzip mtime 0) with its per-file sha256 manifest beside it."""
    import gzip
    import io
    import tarfile
    files = sorted(q for q in Path(src).rglob("*") if q.is_file())
    man = {str(q.relative_to(src)): {"sha256": SE.sha256_file(q), "bytes": q.stat().st_size} for q in files}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as t:
        for q in files:
            ti = tarfile.TarInfo(f"out/{q.relative_to(src)}")
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = q.stat().st_size, 0, 0o644, 0, 0, "", ""
            with open(q, "rb") as f:
                t.addfile(ti, f)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=9) as g:
        g.write(buf.getvalue())
    mpath = dest.with_name(dest.name.replace(".tar.gz", ".manifest.json"))
    mpath.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    return {"archive": str(dest), "sha256": SE.sha256_file(dest), "bytes": dest.stat().st_size, "files": len(man),
            "manifest": str(mpath), "manifest_sha256": SE.sha256_file(mpath)}


def write_tombstone(entry: dict, path: Path) -> None:
    """Append ``entry`` to the project's tombstones BEFORE the bytes go (retire_artifact.py's schema and totals)."""
    doc = json.loads(path.read_text())
    if any(t["canonical_id"] == entry["canonical_id"] for t in doc["tombstones"]):
        raise SE.AuthorizationError(f"a tombstone for {entry['canonical_id']} already exists")
    doc["fields"] = sorted(set(doc["fields"]) | set(entry))
    doc["tombstones"].append(entry)
    if "totals" in doc:
        t = doc["totals"]
        t.update(tombstones_total=t["tombstones_total"] + 1, active=t["active"] + 1,
                 active_retired_bytes=t["active_retired_bytes"] + int(entry["size_bytes"]))
        t["active_retired_human"] = f"{t['active_retired_bytes'] / 2**30:.3f} GiB"
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


def rotate_relay(arm: str, run_id: str, scr: Path, local: dict, *, root: Path = REPO,
                 tombstones: Path | None = None) -> dict:
    """The free relay's rotation after an arm's session is COMPLETE (`design.json :: storage.phase_2_finals`): (1) the
    session's secured evidence as one archive on the relay, uploaded, downloaded again and re-hashed; then (2) for each
    final whose dev-box copy re-hashes file by file to its record NOW, a tombstone first, then the relay copy removed
    (`relay.remove`, permanent, the pointer with it) and confirmed absent. A final that fails any of these keeps its
    relay copy. Re-running it continues an interrupted rotation: a PENDING tombstone of the same id, relay path,
    weights sha256 and members (written, then a removal that failed, or one that took effect but was never recorded) is
    taken up again -- the local copy re-hashed, the removal retried (an absent copy confirms absent), the tombstone and
    this record completed -- never written twice; a tombstone of that id for other bytes stops that final. Writes
    ``storage/relay_rotation_phase2.json``."""
    from stages.single_operator_budget import completion as CP
    from stages.teacher_parent_initialization import relay as R
    from stages.teacher_parent_initialization import train_driver as TD
    tombstones = Path(tombstones or (root / TOMBSTONES))
    out = Path(scr) / "store" / "out"
    rec_path = Path(root) / ROTATION_RECORD
    doc = json.loads(rec_path.read_text()) if rec_path.is_file() else {
        "schema": "aadistill.stage1.single_operator_budget.relay_rotation/v1", "decision": ROTATION_DECISION, "arms": {}}
    row = doc["arms"].setdefault(arm, {"run_id": run_id, "finals": {}})
    arch = Path(scr) / "archive" / f"{run_id}_evidence.tar.gz"     #: beside the secured copy, on the same disk
    if not (row.get("evidence") or {}).get("verified"):
        a = evidence_archive(out, arch)
        put = R.run_online("put", {"files": {Path(a["archive"]).name: a["archive"], Path(a["manifest"]).name: a["manifest"]},
                                   "prefix": f"{C.relay_root()}/phase2_evidence/{run_id}", "scratch": str(arch.parent),
                                   "message": f"SOB1 phase 2: {run_id}'s secured evidence, one archive"}, timeout_s=7200)
        row["evidence"] = {**a, "relay": {k: put.get(k) for k in ("relay", "prefix", "files_sha256", "error")},
                           "verified": bool(put.get("verified"))}
    if not row["evidence"]["verified"]:
        row["status"] = "EVIDENCE_NOT_DURABLE: no relay copy is retired"
    elif C.final_placement(arm) == "relay":
        row["status"] = ("KEPT_ON_RELAY: this arm's finals are placed on the relay (design.json :: "
                         "storage.phase_2_finals.placement); its evidence archive is there too; nothing is retired")
    else:
        for t in CP.train_targets(arm):
            unit = CP.parse_train_target(t)["unit"]
            fin = json.loads((out / unit / "final.json").read_text())
            if (row["finals"].get(unit) or {}).get("verified_absent"):
                continue
            prod = (local.get("products") or {}).get(f"{unit}/final") or {}
            files = {k: v for k, v in fin["files_sha256"].items() if k != R.POINTER}
            here = {k: (SE.sha256_file(Path(prod["path"]) / k) if prod.get("path") and (Path(prod["path"]) / k).is_file()
                        else None) for k in files}
            if here != files:
                row["finals"][unit] = {"kept": "the dev box's copy does not re-hash to the record", "on_disk": here}
                continue
            size = sum(int(v) for v in (fin.get("bytes") or {}).values())
            tid = f"relay_{run_id}_{unit}_final"
            entry = {
                "canonical_id": tid, "historical_paths": [f"{fin['prefix']}/{k}" for k in sorted(files)],
                "store": "relay", "relay_path": fin["prefix"], "retention_tier": "REDUNDANT_REMOTE_COPY",
                "experiment": "SOB1", "seed": CP.parse_train_target(t)["seed"],
                "scientific_role": f"{unit}'s final weights (fp32): the relay COPY; the dev box keeps the copy",
                "weights_sha256": files.get("model.safetensors"), "config_sha256": None, "architecture": None,
                "size_bytes": size, "size_human": f"{size / 2**30:.3f} GiB", "n_files": len(files),
                "members": [{"path": f"{fin['prefix']}/{k}", "sha256": v} for k, v in sorted(files.items())],
                "deletion_criterion": "byte_identical_duplicate",
                "deletion_criterion_meaning": "byte-identical to a copy that survives",
                "same_hash_copies": [{"location": prod["path"], "kind": "local",
                                      "verified": "every file re-hashed to the record just before the removal"}],
                "canonical_surviving_evidence": prod["path"], "frozen_results_location": str(rec_path.relative_to(root))
                if rec_path.is_relative_to(root) else str(rec_path),
                "reason_physical_weights_deleted": (f"the free relay's rotation: {arm}'s session {run_id} is COMPLETE "
                                                    "(both finals and four sets on their files), its evidence archive "
                                                    "is on the relay, and this final re-hashes on the dev box"),
                "maintainer_decision": ROTATION_DECISION,
                "tombstone_written_utc": _utc(), "deleted_utc": None}
            prior = next((x for x in json.loads(tombstones.read_text())["tombstones"] if x["canonical_id"] == tid), None)
            if prior is not None and any(prior.get(k) != entry[k] for k in ("relay_path", "weights_sha256", "members")):
                row["finals"][unit] = {"kept": f"tombstone {tid} exists for other bytes; nothing is removed",
                                       "tombstone_relay_path": prior.get("relay_path")}
                continue
            if prior is None:               #: first: the tombstone, before any byte goes
                write_tombstone(entry, tombstones)
            #: a pending one (written, then a failed or unrecorded removal) is continued under the same id: the local
            #: copy was re-hashed above; the removal is idempotent (an absent copy confirms absent and frees nothing)
            try:                            #: whether the relay still holds every file (for the record only)
                there = _lfs_sha([f"{fin['prefix']}/{k}" for k in files])
                was = len(there) == len(files)
            except Exception:  # noqa: BLE001
                was = None
            got = R.run_online("remove", {"record": TD.with_pointer(fin), "message": f"{unit}: final weights, a "
                                          f"redundant copy (tombstone {tid})"}, timeout_s=3600)
            ok = bool(got.get("verified_absent"))
            row["finals"][unit] = {"tombstone": tid, "local": prod["path"], "verified_absent": ok,
                                   "freed_bytes": got.get("freed_bytes"), "error": got.get("error"), "utc": _utc(),
                                   "relay_copy_present_before": was,
                                   "continued_pending_tombstone": prior is not None}
            t_doc = json.loads(tombstones.read_text())
            e = next(x for x in t_doc["tombstones"] if x["canonical_id"] == tid)
            att = [*((e.get("deletion") or {}).get("attempts") or []),
                   {"utc": _utc(), "verified_absent": ok, "relay_copy_present_before": was, "error": got.get("error")}]
            e.update(deleted_utc=(e.get("deleted_utc") or _utc()) if ok else None,
                     deletion={"status": "DELETED" if ok else "PENDING", "attempts": att,
                               "relay_remove": {k: got.get(k) for k in ("removed_paths", "lfs_left", "paths_left",
                                                                         "shared_oids_kept", "freed_bytes", "error")}})
            tombstones.write_text(json.dumps(t_doc, indent=1, ensure_ascii=False) + "\n")
        units = [CP.parse_train_target(t)["unit"] for t in CP.train_targets(arm)]
        row["status"] = ("ROTATED: both relay copies retired" if all((row["finals"].get(u) or {}).get("verified_absent")
                                                                     for u in units) else "PARTIAL: see finals")
    row["checked_utc"] = _utc()
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return {k: row.get(k) for k in ("status", "finals")}


def arm_sessions(root: Path = REPO) -> dict[str, dict]:
    """Each arm's latest COMPLETE training session (its ``phase2_completion.json``), by arm."""
    best: dict[str, dict] = {}
    for p in sorted((Path(root) / LOGS / "runs").glob(f"*/{PHASE2_COMPLETION}")):
        doc = json.loads(p.read_text())
        if doc.get("status") == "COMPLETE" and (doc["arm"] not in best or doc["checked_utc"] >= best[doc["arm"]]["checked_utc"]):
            best[doc["arm"]] = doc
    return best


def phase1_evidence(root: Path = REPO) -> Path | None:
    """Phase 1's secured evidence on the dev box (the COMPLETE phase-1 run's ``<local store>/<run>/store/out``)."""
    for p in sorted((Path(root) / LOGS / "runs").glob(f"*/{PHASE1_RUN_COMPLETION}"), reverse=True):
        doc = json.loads(p.read_text())
        out = C.local_store() / doc["run_id"] / "store" / "out"
        if doc.get("status") == "COMPLETE" and out.is_dir():
            return out
    return None


def study_completion(*, root: Path = REPO, sessions: dict | None = None, phase1_out: Path | None = None,
                     write: bool = True) -> dict:
    """The study's scientific completion ($0, dev box): every arm's COMPLETE session's secured evidence
    (`arm_sessions`), so the six final weights and the twelve post-recovery sets checked on their files
    (`completion.study`), and the registered comparison -- three pairs per mode, Holm with m = 3, the second family
    only after the first rejects all three -- recomputed from those sealed records (`analysis.phase2_readout`), with
    each arm's descriptive damage / recovery / gap against phase 1's untrained arm and teacher. Writes
    ``results/study_completion.json``."""
    import tempfile
    from stages.single_operator_budget import analysis as SA
    from stages.single_operator_budget import completion as CP
    sessions = arm_sessions(root) if sessions is None else sessions
    outs = {a: Path(v["scr"]) / "store" / "out" for a, v in sessions.items()}
    phase1_out = phase1_out if phase1_out is not None else phase1_evidence(root)
    st = CP.study(outs, phase1_out=phase1_out)
    unfinished = {}
    if st["arms_missing"]:
        unfinished["arms_without_a_complete_session"] = st["arms_missing"]
    if not st["complete"]:
        unfinished["study"] = {"finals": st["finals"], "sets": st["sets"], "modes": st["modes"]}
    readout = None
    if st["complete"]:
        with tempfile.TemporaryDirectory() as tmp:
            readout = SA.phase2_readout(outs, phase1_out, Path(tmp))
    doc = {"schema": "aadistill.stage1.single_operator_budget.study_completion/v1",
           "status": "COMPLETE" if st["complete"] else "INCOMPLETE", "unfinished": unfinished,
           "rule": "three arms x two seeds: six final weights durable and published, twelve sets complete against "
                   "them, one protocol per mode; the registered comparison recomputed from the sealed records",
           "sessions": {a: {k: v.get(k) for k in ("run_id", "scr", "checked_utc")} for a, v in sessions.items()},
           "phase1_evidence": None if phase1_out is None else str(phase1_out),
           "completion": {k: st[k] for k in ("finals", "sets", "modes", "arms_missing", "phase1")},
           "readout": readout, "checked_utc": _utc()}
    dest = Path(root) / STUDY_RESULT
    if write:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return {**doc, "record": str(dest.relative_to(root)) if dest.is_relative_to(root) else str(dest)}


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
            return 0
        doc = retain(kind, a.run_id, Path(a.src), arm=arm)
        print(json.dumps({"status": doc["status"], "missing": doc["missing"],
                          "products": {k: v["path"] for k, v in doc["products"].items()}}, indent=1))
        return 0 if doc["status"] == "COMPLETE" else 12
    if argv and argv[0] in ("issue", "launch"):
        try:
            SE.require_open(STUDY)
            SE.require_kind_open(STUDY, kind)
        except SE.AuthorizationError as exc:
            print(f"REFUSED at $0: {exc}")
            return 11
    if argv and argv[0] == "launch" and kind == TRAIN_KIND:
        sc = artifact_conditions(argv)              #: before the bundle, the provider or the relay is read
        bad = arm_binding_problems(sc, arm) if sc is not None else []
        if bad:
            print(f"REFUSED at $0: the artifact does not fund the plan this launch would run: {bad}\n"
                  "No provider resource was created.")
            return 11
    if argv and argv[0] == "study-completion":
        doc = study_completion()
        print(json.dumps({k: doc[k] for k in ("status", "unfinished", "record")}, indent=1, default=str))
        return 0 if doc["status"] == "COMPLETE" else 12
    if argv and argv[0] == "grant" and kind == TRAIN_KIND:
        ap = argparse.ArgumentParser(prog="SOB1 grant --kind sob1_train")
        ap.add_argument("--decision", required=True, help="the maintainer's approval, quoted")
        ap.add_argument("--granted-by", required=True)
        a = ap.parse_args(argv[1:])
        print(SE.write_grant(STUDY, decision=a.decision, granted_by=a.granted_by, kind=TRAIN_KIND))
        return 0
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
            g = kind_grant(kind, dry_run=x.dry_run)
            cap = session_terms(kind, arm, g).get("per_session_hard_cap_usd")
            if cap is None:
                raise SE.AuthorizationError(f"{g['_source']} names no cap for the {kind} kind{f' ({arm})' if arm else ''}")
            if x.hard_cap_usd > float(cap) + 1e-9:
                raise SE.AuthorizationError(f"${x.hard_cap_usd} exceeds the {kind}{f' {arm}' if arm else ''} cap ${cap}")
            bad = reserve_problems(x.hard_cap_usd, g, SE.envelope(STUDY, g, REPO, kind), SE.live_position(REPO))
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
