#!/usr/bin/env python3
"""Research A v2 (token-matched): its own study identity, one session kind -- the supplementary teacher generation.

    # a one-use authorization for one supplement session (--dry-run writes outside the run dir)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/token_matched_phase.py issue \\
        --kind supplement --run-id <run> --expected-usd <x> --hard-cap-usd <cap> [--dry-run --out <scratch>/auth.json]
    # the session
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/token_matched_phase.py launch \\
        --kind supplement --minutes <M> --run-id <run> --scr <scratch> --session-commit <sha> --bundle <name> \\
        --max-price <live price> --network-volume-id am5m373kuq --volume-mount-path /runpod-volume \\
        --data-center-ids EUR-IS-2 --volume-gb 55

The study binds token_matched_design.json, its preregistration, the supplement protocol and plan, order_v2 and
the source catalogue extension (`STUDY.identity_files`): Research A v1's frozen identity (phase.py, 848f6508...)
never funds or authorizes anything here, and its grant is closed for new work. The grant
(``logs/budget/approvals/teacher_modes_token_matched_grant.json``) is the maintainer's 2026-10-31 order to
complete the data inside Research A's and the project's existing caps: its ceiling is Research A's $142.19 minus
everything v1 spent (sessions and storage), so the cumulative phase cap is unchanged.

KIND ``supplement`` (one L40S, on am5m373kuq): `supplement_session.py` -- order_v2's rounds in both modes until
each (mode, cell) need is met. Done; the grant closes it for new work (cap 0).

THE A1 PHASE (maintainer 2026-11-01, 'Research A v2: Execute A1 First'; the grant's ``phase_a1``), on the phase's own
350 GB network volume, one L40S, in order:
* ``inputs`` -- `inputs_session.py`: the calibration top-up and the state evaluation's native generation, both modes;
* ``search`` -- the hybrid D4 full search (its driver, below);
* ``train_a1`` -- A1's two seeds, trained and evaluated in turn.
Each kind's hard cap is the grant's ``session_hard_caps_usd[kind]``, enforced at issue; all of them draw on one
cumulative envelope. Sealed shards are pulled on every poll; the volume keeps everything.
"""
from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))

from shared import study_engineering as SE  # noqa: E402
from stages.teacher_modes import phase as P1  # noqa: E402

DESIGN = "configs/stages/stage-2/teacher_modes/token_matched_design.json"
PREREG = "logs/stages/stage-2/teacher_modes/plans/token_matched_preregistration.json"
SUPPLEMENT_PROTOCOL = "configs/stages/stage-2/teacher_modes/supplement_protocol.json"
PLAN = "logs/stages/stage-2/teacher_modes/data/supplement_plan.json"
#: outside git (heavy): bound by its sha256 in order_v2.manifest.json and supplement_protocol.json, staged as an asset
ORDER = "artifacts/stages/stage-2/teacher_modes/tasks/order_v2.jsonl.gz"
GPU = P1.GPU
SESSION_KIND = "stage2_teacher_modes_tm"

STUDY = SE.Study(
    key="A-TM", plan_id="stage2.teacher_modes.token_matched",
    logs=Path("logs/stages/stage-2/teacher_modes"),
    grant=Path("logs/budget/approvals/teacher_modes_token_matched_grant.json"),
    proposal=Path("logs/budget/applications/research_a_tm_2026-10-31/proposal.json"),
    identity_files=(DESIGN, PREREG, SUPPLEMENT_PROTOCOL, PLAN,
                    "data/stages/stage-2/teacher_modes/tasks/order_v2.manifest.json",
                    "data/stages/stage-2/teacher_modes/tasks/calibration_v2.jsonl",
                    "logs/stages/stage-2/teacher_modes/audit/calibration_coverage.json",
                    "configs/stages/stage-2/teacher_modes/token_matched_sources.json", P1.PROTOCOL,
                    "configs/stages/stage-2/teacher_modes/teacher_identity.json",
                    "configs/stages/stage-2/teacher_modes/inputs_protocol.json",
                    "configs/stages/stage-2/teacher_modes/full_search_types.json"),
    scripts=tuple(f"scripts/stages/stage-2/teacher_modes/{n}.py" for n in
                  ("token_matched_phase", "phase", "supplement_session", "teacher_session", "admission",
                   "supply_audit", "build_supplement_tasks", "build_tasks", "build_calibration_supplement",
                   "hybrid_calibration", "inputs_session", "full_search", "hybrid_state_eval",
                   "freeze_state_eval", "a1_units", "training_plan",
                   #: the packed round (packed_round_design.json)
                   "packed_full_search", "packed_inputs", "packed_packs", "packed_training_plan", "packing_audit",
                   "tool_episodes", "tool_admission", "tool_pool", "tool_session", "six_category_selection")) + (
        "scripts/shared/study_engineering.py", "scripts/shared/project_position.py", "scripts/shared/eval_sets.py",
        "scripts/shared/pod/teacher_cache.py", "scripts/stages/stage-0/calibration_volume/run_battery.py",
        "scripts/stages/stage-0/calibration_volume/score_battery.py",
        "scripts/stages/stage-0/calibration_volume/mixed_governance.py",
        "scripts/stages/stage-2/reasoning_data/generation_driver.py",
        "scripts/stages/stage-2/reasoning_data/generation_adapter.py",
        "scripts/stages/stage-2/reasoning_data/generation_plan.py",
        "scripts/stages/stage-2/reasoning_data/supporting_verify.py",
        "scripts/maintenance/consolidation/derive_budget.py", "scripts/maintenance/consolidation/provider_billing.py",
        "scripts/shared/deployment.py", "scripts/shared/pod/watchdog.py",
        "scripts/shared/pod/autoinit_preflight_setup.sh", "scripts/shared/pod/collect_artifacts.py"),
    wire_schema="aadistill.stage2.teacher_modes.tm_authorization/v1",
    claims=frozenset({"formal", "teacher_generation"}),
    kinds=("supplement", "inputs", "search", "train_a1",
           #: the packed round: none runs before the grant names its hard cap (kind_cap_problems)
           "tool_generation", "tool_generation_t2507", "packed_search",
           "train_6cat_a1", "train_6cat_a2", "train_6cat_c1"),
    kind_claims=(("search", frozenset({"formal"})), ("train_a1", frozenset({"formal", "training", "evaluation"})),
                 ("packed_search", frozenset({"formal"})),
                 *((k, frozenset({"formal", "training", "evaluation"}))
                   for k in ("train_6cat_a1", "train_6cat_a2", "train_6cat_c1"))))


class TMAuthorization(SE.EngineeringAuthorization):
    """The study's artifact type: setup's ``stage2_teacher_modes_tm`` branch loads exactly this one."""
    STUDY = STUDY


#: what each kind's artifact binds, beyond the study identity (the records its driver reads, by sha256)
KIND_RECORDS = {
    "supplement": (),
    "inputs": ("configs/stages/stage-2/teacher_modes/inputs_protocol.json",),
    "search": ("configs/stages/stage-2/teacher_modes/hybrid_d4_profiles.json",
               "logs/stages/stage-2/teacher_modes/audit/state_eval_hybrid_freeze.json"),
    "train_a1": ("logs/stages/stage-2/teacher_modes/plans/training_plan_a1.json",),
    "tool_generation": ("configs/stages/stage-2/teacher_modes/tool_generation_protocol.json",
                        "data/stages/stage-2/teacher_modes/tool_pool_v1/pool.json"),
    "tool_generation_t2507": ("configs/stages/stage-2/teacher_modes/tool_generation_protocol.json",
                              "data/stages/stage-2/teacher_modes/tool_pool_v1/pool.json"),
    "packed_search": ("configs/stages/stage-2/teacher_modes/packed_round_design.json",
                      "configs/stages/stage-2/teacher_modes/hybrid_profiles_v2.json",
                      "configs/stages/stage-2/teacher_modes/full_search_types_v2.json",
                      "logs/stages/stage-2/teacher_modes/audit/state_eval_hybrid_v2_freeze.json",
                      "configs/shared/baselines/selection.json",
                      "configs/shared/baselines/initialization_forward_baseline_v2.json"),
    **{k: ("logs/stages/stage-2/teacher_modes/plans/packed_training_plan.json",)
       for k in ("train_6cat_a1", "train_6cat_a2", "train_6cat_c1")},
}
#: the packed training kinds' arms, and the arms trained with Thinking-2507 rather than the hybrid
PACKED_TRAIN_ARMS = {"train_6cat_a1": "A1-6cat", "train_6cat_a2": "A2-6cat", "train_6cat_c1": "C1-6cat"}
THINKING_2507_KINDS = frozenset({"tool_generation_t2507", "train_6cat_c1"})
#: the full search's planned minutes and A1's per-unit deadline (budget_v2.json: the search's deadline-bounded
#: 37.3 h; A1's 6.35 h per seed)
SEARCH_DEADLINE_MINUTES = 37.29 * 60
A1_UNIT_DEADLINE_HOURS = 6.35


def kind_conditions(kind: str, root: Path = REPO_ROOT) -> dict:
    """Generation runs the path the committed L40S check admitted (the same engine, card and protocol as the v1
    teacher session); every kind's artifact binds that record and its own `KIND_RECORDS`."""
    from stages.teacher_modes import l40s_check as LC
    gate = "teacher" if kind in ("supplement", "inputs", "tool_generation") else None
    if gate:
        bad = LC.gate(LC.load_config(Path(root) / P1.CHECK_CONFIG)["gates"][gate], root)
        if bad:
            raise SE.AuthorizationError("the L40S check record does not admit generation: " + "; ".join(bad))
    out = {"l40s_check_record_sha256": P1._sha(P1.CHECK_RECORD, root)}
    for rel in KIND_RECORDS.get(kind, ()):
        out[f"record:{rel}"] = P1._sha(rel, root)
    return out


def kind_cap_problems(kind: str, hard_cap_usd: float, root: Path = REPO_ROOT) -> list[str]:
    """The grant's cap for this kind (``session_hard_caps_usd``): a kind without one, or a cap above it, refuses."""
    import json
    g = json.loads((Path(root) / STUDY.grant).read_text())
    caps = g.get("session_hard_caps_usd") or {}
    if kind not in caps:
        return [f"the grant names no hard cap for kind {kind!r}"]
    if hard_cap_usd > float(caps[kind]) + 1e-9:
        return [f"${hard_cap_usd} exceeds the grant's ${caps[kind]} for kind {kind!r}"]
    return []


def require_on_pod(path: Path, kind: str, root: Path = REPO_ROOT) -> dict:
    auth = TMAuthorization.load(path)
    got = SE.require_on_pod(auth, STUDY, auth.plan_hash, str(root), kind=kind)
    sc = auth.stage_conditions or {}
    bad = {k: (sc.get(k), v) for k, v in kind_conditions(kind, Path(root)).items() if sc.get(k) != v}
    if bad:
        raise SE.AuthorizationError(f"the {kind} artifact binds other records than the pod holds: {bad}")
    return {**got, "harness_digest": got["harness"]}


#: small v1 records left on the volume when the recovery was stopped (unit 6 trained, not scored)
V1_SMALL = tuple(f"units/H-T-C-s20260801/{n}" for n in ("checkpoint.descriptor.json", "training.descriptor.json",
                                                         "attempts.jsonl", "train/a1/train_log.jsonl",
                                                         "train/a1/run_manifest.json", "train/a1/train_inputs.json",
                                                         "train/a1/checkpoints/latest.txt")) + (
    "recovery_a_recovery_20261030_211948.json",)


#: this study's stage labels (supplement_session.py: s<round><t|n>_<mode>_pass<k>_...)
OWN_SHARD = re.compile(r"^s\d\d[tn]_")
#: per kind: (the generation directory under the durable root, its stage labels, the session record)
SHARDS = {"supplement": ("generation", OWN_SHARD, "supplement_session.json"),
          "inputs": ("inputs_generation", re.compile(r"^(c00|e\d\d)[tn]_"), "inputs_session.json")}


def _kind(ctx) -> str:
    return (ctx.auth.stage_conditions or {}).get("kind", "supplement") if getattr(ctx, "auth", None) else "supplement"


def secure(ctx) -> None:
    """on_poll: every sealed shard by its marker, the session record and v1's small records. Never raises."""
    try:
        if ctx.target is None:
            return
        root, st = P1.durable(ctx.args), SE.store(ctx)
        kind = _kind(ctx)
        gdir, own, session_record = SHARDS[kind]
        got = ctx.evidence.setdefault("secured", {})
        lines = P1._listing(ctx, f"cd {root}/{gdir} 2>/dev/null && for f in *.done; do [ -e \"$f\" ] && "
                                 f"echo \"$f $(cat \"$f\")\"; done; ls *.complete 2>/dev/null || true")
        if lines is None:
            ctx.evidence.setdefault("listing_unknown", []).append(gdir)
            return
        for line in lines:
            parts = line.split()
            if not parts or not own.match(parts[0]):
                continue                    #: v1's shards are on the dev box already: artifacts follow consumers
            if len(parts) == 2 and parts[0].endswith(".done") and len(parts[1]) == 64:
                name = parts[0][:-5] + ".jsonl"
                if got.get(name, {}).get("ok"):
                    continue
                ok = P1._pull(ctx, f"{root}/{gdir}/{name}", st / gdir / name, parts[1])
                if ok:
                    (st / gdir / parts[0]).write_text(parts[1] + "\n")
                got[name] = {"ok": ok, "sha256": parts[1]}
            elif len(parts) == 1 and parts[0].endswith(".complete"):
                P1._pull(ctx, f"{root}/{gdir}/{parts[0]}", st / gdir / parts[0], None)
        P1._pull(ctx, f"{root}/{session_record}", st / session_record, None)
        if kind == "supplement" and not ctx.evidence.get("v1_small"):
            ctx.evidence["v1_small"] = {f: P1._pull(ctx, f"{root}/{f}", st / "v1_small" / f, None) for f in V1_SMALL}
        SE._scp(ctx, f"/workspace/{SESSION_KIND}.status", st / "driver.status", limit_s=120)
    except Exception as exc:  # noqa: BLE001
        ctx.evidence.setdefault("secure_errors", []).append(repr(exc)[:300])


def runner_logs(ctx) -> None:
    """At closeout, this study's runner logs (small text): a unit that fails before it seals anything leaves its
    traceback only there (s01n of a_supplement_20261031_092108). Never raises."""
    try:
        if ctx.target is None:
            return
        root, st = P1.durable(ctx.args), SE.store(ctx)
        gdir, own, _ = SHARDS[_kind(ctx)]
        lines = P1._listing(ctx, f"cd {root}/{gdir} 2>/dev/null && ls *.log 2>/dev/null || true") or []
        ctx.evidence["runner_logs"] = {n: P1._pull(ctx, f"{root}/{gdir}/{n}", st / gdir / n, None)
                                       for n in (l.strip() for l in lines) if own.match(n)}
    except Exception as exc:  # noqa: BLE001
        ctx.evidence.setdefault("secure_errors", []).append(repr(exc)[:300])


def products_secured(ctx, *_a, **_k) -> tuple[bool, str]:
    secure(ctx)
    runner_logs(ctx)
    got = ctx.evidence.get("secured", {})
    bad = sorted(k for k, v in got.items() if not v.get("ok"))
    if bad or ctx.evidence.get("listing_unknown"):
        return False, f"not secured off-pod: {bad[:6]} {ctx.evidence.get('listing_unknown') or ''}"
    return True, f"{len(got)} sealed shard(s) secured off-pod; everything is on the network volume"


def the_kind_records_are_the_bound_ones(ctx) -> tuple[bool, str]:
    sc = ctx.auth.stage_conditions or {}
    try:
        want = kind_conditions(sc.get("kind"))
    except SE.AuthorizationError as exc:
        return False, str(exc)
    bad = {k: v for k, v in want.items() if sc.get(k) != v}
    return (not bad, "the kind's bound records are the committed ones" if not bad else f"records changed: {bad}")


def supplement_command(ctx, plan) -> str:
    a, status = ctx.args, f"/workspace/{SESSION_KIND}.status"
    auth = f"{SE.POD_REPO}/{STUDY.authorization_path(a.run_id)}"
    return (f"mkdir -p {P1.durable(a)} && {P1._py()} scripts/stages/stage-2/teacher_modes/supplement_session.py "
            f"--authorization {auth} --durable-root {P1.durable(a)} --status {status}" + P1._tail(status))


def search_command(ctx, plan, planned_minutes: float = SEARCH_DEADLINE_MINUTES) -> str:
    a, status = ctx.args, f"/workspace/{SESSION_KIND}.status"
    auth = f"{SE.POD_REPO}/{STUDY.authorization_path(a.run_id)}"
    #: the driver stops itself 30 minutes before the session's planned minutes, leaving the commit and the copy time
    minutes = max(30.0, float(planned_minutes) - 30.0)
    return (f"mkdir -p {P1.durable(a)} && {P1._py()} scripts/stages/stage-2/teacher_modes/full_search.py "
            f"--authorization {auth} --durable-root {P1.durable(a)} --status {status} --deadline-minutes {minutes:.0f}"
            + P1._tail(status))


def train_a1_command(ctx, plan) -> str:
    a, status = ctx.args, f"/workspace/{SESSION_KIND}.status"
    auth = f"{SE.POD_REPO}/{STUDY.authorization_path(a.run_id)}"
    return (f"mkdir -p {P1.durable(a)} && {P1._py()} scripts/stages/stage-2/teacher_modes/a1_units.py "
            f"--authorization {auth} --durable-root {P1.durable(a)} --status {status} "
            f"--unit-deadline-hours {A1_UNIT_DEADLINE_HOURS}" + P1._tail(status))


def inputs_command(ctx, plan) -> str:
    a, status = ctx.args, f"/workspace/{SESSION_KIND}.status"
    auth = f"{SE.POD_REPO}/{STUDY.authorization_path(a.run_id)}"
    return (f"mkdir -p {P1.durable(a)} && {P1._py()} scripts/stages/stage-2/teacher_modes/inputs_session.py "
            f"--authorization {auth} --durable-root {P1.durable(a)} --status {status}" + P1._tail(status))


def tool_generation_command(ctx, plan, lineage: str = "hybrid") -> str:
    a, status = ctx.args, f"/workspace/{SESSION_KIND}.status"
    auth = f"{SE.POD_REPO}/{STUDY.authorization_path(a.run_id)}"
    return (f"mkdir -p {P1.durable(a)} && {P1._py()} scripts/stages/stage-2/teacher_modes/tool_session.py "
            f"--authorization {auth} --lineage {lineage} --durable-root {P1.durable(a)} --status {status}"
            + P1._tail(status))


def packed_search_command(ctx, plan, planned_minutes: float = SEARCH_DEADLINE_MINUTES) -> str:
    a, status = ctx.args, f"/workspace/{SESSION_KIND}.status"
    auth = f"{SE.POD_REPO}/{STUDY.authorization_path(a.run_id)}"
    minutes = max(30.0, float(planned_minutes) - 30.0)
    return (f"mkdir -p {P1.durable(a)} && {P1._py()} scripts/stages/stage-2/teacher_modes/packed_full_search.py "
            f"--authorization {auth} --durable-root {P1.durable(a)} --status {status} --deadline-minutes {minutes:.0f}"
            + P1._tail(status))


def train_6cat_command(ctx, plan, kind: str) -> str:
    a, status = ctx.args, f"/workspace/{SESSION_KIND}.status"
    auth = f"{SE.POD_REPO}/{STUDY.authorization_path(a.run_id)}"
    return (f"mkdir -p {P1.durable(a)} && {P1._py()} scripts/stages/stage-2/teacher_modes/a1_units.py "
            f"--authorization {auth} --durable-root {P1.durable(a)} --status {status} "
            f"--unit-deadline-hours {PACKED_UNIT_DEADLINE_HOURS[kind]} --kind {kind} "
            f"--plan {KIND_RECORDS[kind][0]} --arm {PACKED_TRAIN_ARMS[kind]} --units-dir packed_units"
            + P1._tail(status))


#: per-unit deadlines of the packed trainings (plans/packed_training_budget.json, upper case per seed, rounded up)
PACKED_UNIT_DEADLINE_HOURS = {"train_6cat_a1": 8.5, "train_6cat_a2": 24.0, "train_6cat_c1": 7.5}

#: per kind: (driver, command, local assets beyond the battery, description)
DRIVERS = {
    "supplement": ("scripts/stages/stage-2/teacher_modes/supplement_session.py", supplement_command, ("order",),
                   "Research A v2: the supplementary teacher generation (order_v2, both modes, token quotas)"),
    "inputs": ("scripts/stages/stage-2/teacher_modes/inputs_session.py", inputs_command, (),
               "Research A v2, A1 phase A: the calibration top-up and the state evaluation's native generation"),
    "search": ("scripts/stages/stage-2/teacher_modes/full_search.py", search_command, ("profiles", "suite"),
               "Research A v2, A1 phase B: the hybrid D4 full search (one L40S, micro-batch 1)"),
    "train_a1": ("scripts/stages/stage-2/teacher_modes/a1_units.py", train_a1_command, ("a1_pack",),
                 "Research A v2, A1 phases C and D: A1's two seeds trained from the Top-1 and formally evaluated"),
    "tool_generation": ("scripts/stages/stage-2/teacher_modes/tool_session.py", tool_generation_command,
                        ("tool_pool",), "Research A packed round: the hybrid's tool-calling episodes, both modes"),
    "tool_generation_t2507": ("scripts/stages/stage-2/teacher_modes/tool_session.py",
                              lambda ctx, plan: tool_generation_command(ctx, plan, "thinking_2507"), ("tool_pool",),
                              "Research A packed round: Thinking-2507's tool-calling episodes"),
    "packed_search": ("scripts/stages/stage-2/teacher_modes/packed_full_search.py", packed_search_command,
                      ("profiles_v2", "suite_v2"),
                      "Research A packed round: the hybrid full search in the packed forward context (one L40S)"),
    **{k: ("scripts/stages/stage-2/teacher_modes/a1_units.py",
           (lambda ctx, plan, _k=k: train_6cat_command(ctx, plan, _k)), (f"pack_{k}",),
           f"Research A packed round: {PACKED_TRAIN_ARMS[k]}'s two seeds trained and formally evaluated")
       for k in PACKED_TRAIN_ARMS},
}
#: the plan's time contingency per kind: the search and A1's units stop themselves at their own deadlines with their
#: state saved (full_search --deadline-minutes, a1_units --unit-deadline-hours), so 5% covers the closeout; the
#: generation kinds keep the runner's 25%
CONTINGENCY = {"search": 0.05, "train_a1": 0.05, "packed_search": 0.05,
               **{k: 0.05 for k in ("train_6cat_a1", "train_6cat_a2", "train_6cat_c1")}}
#: small evidence pulled on every poll, per kind (relative to the durable root)
POLLED = {"search": ("search_a1/search_record.json", "search_a1/first_evaluation_check.json",
                     "search_a1/search/states.jsonl", "search_a1/search/telemetry.jsonl", "search_a1/products.json"),
          "train_a1": ("a1_units.json",),
          "packed_search": ("search_packed/search_record.json", "search_packed/first_evaluation_check.json",
                            "search_packed/search/states.jsonl", "search_packed/search/telemetry.jsonl",
                            "search_packed/products.json"),
          **{k: (f"{PACKED_TRAIN_ARMS[k]}_units.json",) for k in ("train_6cat_a1", "train_6cat_a2", "train_6cat_c1")},
          "tool_generation": ("tool_generation_hybrid/tool_session.json",),
          "tool_generation_t2507": ("tool_generation_thinking_2507/tool_session.json",)}


def secure_files(ctx) -> None:
    """on_poll for the search and A1 kinds: their small evidence, pulled as it stands (never raises)."""
    try:
        if ctx.target is None:
            return
        root, st, kind = P1.durable(ctx.args), SE.store(ctx), _kind(ctx)
        for rel in POLLED.get(kind, ()):
            P1._pull(ctx, f"{root}/{rel}", st / rel, None)
        if kind == "train_a1":
            lines = P1._listing(ctx, f"cd {root}/v2_units 2>/dev/null && ls -d A1-s*/train/train_log.jsonl 2>/dev/null || true") or []
            for rel in (l.strip() for l in lines if l.strip()):
                P1._pull(ctx, f"{root}/v2_units/{rel}", st / "v2_units" / rel, None)
        SE._scp(ctx, f"/workspace/{SESSION_KIND}.status", st / "driver.status", limit_s=120)
    except Exception as exc:  # noqa: BLE001
        ctx.evidence.setdefault("secure_errors", []).append(repr(exc)[:300])


def top1_on_volume(ctx, top_dir: str, files_sha256: dict[str, str]) -> dict[str, bool]:
    """Each Top-1 file hashed where its one consumer reads it -- the network volume, by the train_a1 pod -- against the
    digest the search committed. Its bytes are not carried to the dev box (P8.4): nothing there reads them, and on
    2026-11-02 the 1.19 GB pull moved 5.7 MB in its 1,200 s and held the pod for an hour."""
    names = sorted(files_sha256)
    lines = P1._listing(ctx, f"cd {top_dir} && sha256sum " + " ".join(names)) or []
    seen = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            seen[parts[1]] = parts[0]
    ctx.evidence["top1_on_volume"] = {"path": top_dir, "sha256": seen}
    return {f"top1_on_volume/{n}": seen.get(n) == files_sha256[n] for n in names}


def files_secured(ctx, *_a, **_k) -> tuple[bool, str]:
    """Closeout for the search and A1 kinds: the evidence directories tarred and pulled (weights excluded: they stay
    on the volume, by sha256 in the records), and the search's Top-1 verified by its sha256 on the volume."""
    import json
    secure_files(ctx)
    root, st, kind = P1.durable(ctx.args), SE.store(ctx), _kind(ctx)
    got = {}
    if kind in ("search", "packed_search"):
        sub = "search_a1" if kind == "search" else "search_packed"
        got["search_evidence"] = P1._tar(ctx, f"{root}/{sub}", "search_evidence",
                                         excludes=("./search/states", "./top1", "./repeatability"))[0]
        prod = st / f"{sub}/products.json"
        if prod.is_file():
            got.update(top1_on_volume(ctx, f"{root}/{sub}/top1", json.loads(prod.read_text())["top1"]["files_sha256"]))
    elif kind in ("tool_generation", "tool_generation_t2507"):
        sub = "tool_generation_hybrid" if kind == "tool_generation" else "tool_generation_thinking_2507"
        got["tool_generation"] = P1._tar(ctx, f"{root}/{sub}", "tool_generation")[0]
    elif kind in PACKED_TRAIN_ARMS:
        got["packed_units_evidence"] = P1._tar(ctx, f"{root}/packed_units", "packed_units_evidence",
                                               excludes=("./*/train/checkpoints", "./*/evaluation/package/*.safetensors"))[0]
    else:
        #: the evaluation's model package is the checkpoint's weights again (a_train_a1_20261102_184122 carried two
        #: 2.38 GB copies home that no analysis reads; their sha256 is in a1_units.json) -- they stay on the volume
        got["a1_evidence"] = P1._tar(ctx, f"{root}/v2_units", "a1_evidence",
                                     excludes=("./*/train/checkpoints", "./*/evaluation/package/*.safetensors"))[0]
    ctx.evidence["files_secured"] = got
    bad = sorted(k for k, v in got.items() if not v)
    return (not bad, "secured: " + ", ".join(sorted(got)) if not bad else f"not secured: {bad}")


def launch_spec(kind: str = "supplement", minutes: float = 0.0) -> SE.Launch:
    import json
    from aadistill.infrastructure.session import LocalAsset
    if kind not in DRIVERS:
        raise SE.AuthorizationError(f"kind {kind!r} has no session driver yet")
    driver, command, extra, description = DRIVERS[kind]
    if kind in ("search", "packed_search") and minutes:
        command = functools.partial(command, planned_minutes=float(minutes))
    t = json.loads((REPO_ROOT / P1.PROTOCOL).read_text())["teacher"]
    if kind in THINKING_2507_KINDS:          #: C1-6cat's lineage: Thinking-2507 generates and teaches
        li = json.loads((REPO_ROOT / "configs/stages/stage-2/teacher_modes/tool_generation_protocol.json")
                        .read_text())["lineages"]["thinking_2507"]
        t = {"model": li["model"], "revision": li["revision"]}
    battery = LocalAsset(repo_path="artifacts/stages/stage-3/eval/battery_v2", dest_name="battery_v2",
                         install_to="artifacts/stages/stage-3/eval")         #: the v1 teacher kind's assets, unchanged
    order = LocalAsset(repo_path=ORDER, dest_name=Path(ORDER).name, install_to=str(Path(ORDER).parent))
    cal = "artifacts/stages/stage-2/teacher_modes/calibration_v2"
    suite = "artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v1"
    pack = "artifacts/stages/stage-2/teacher_modes/packs_tm/A1"
    def asset(path: str) -> LocalAsset:
        return LocalAsset(repo_path=path, dest_name=Path(path).name, install_to=str(Path(path).parent))
    packed = {"tool_pool": asset("data/stages/stage-2/teacher_modes/tool_pool_v1"),
              "profiles_v2": asset("artifacts/stages/stage-2/teacher_modes/calibration_v3"),
              "suite_v2": asset("artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v2"),
              **{f"pack_{k}": asset(f"artifacts/stages/stage-2/teacher_modes/packs_6cat/{a}")
                 for k, a in PACKED_TRAIN_ARMS.items()}}
    named = {**packed, "order": order,
             "profiles": LocalAsset(repo_path=cal, dest_name=Path(cal).name, install_to=str(Path(cal).parent)),
             "suite": LocalAsset(repo_path=suite, dest_name=Path(suite).name, install_to=str(Path(suite).parent)),
             "a1_pack": LocalAsset(repo_path=pack, dest_name=Path(pack).name, install_to=str(Path(pack).parent))}
    assets = (battery,) + tuple(named[x] for x in extra)
    polls = (secure, products_secured) if kind in SHARDS else (secure_files, files_secured)
    relay = ()
    if kind == "train_a1" or kind in PACKED_TRAIN_ARMS:   #: the evaluation's relayed inputs (phase.py's recovery)
        from stages.phase_c1.autoinit_c1_launch import C1_EVAL_TOKENIZER
        relay = tuple(C1_EVAL_TOKENIZER)
    return SE.Launch(
        session_kind=SESSION_KIND, authorization=TMAuthorization, teacher_model=t["model"],
        teacher_revision=t["revision"], kind=kind, local_assets=assets, disk_gb=150,
        contingency_fraction=CONTINGENCY.get(kind, 0.25),
        setup_env={"PHASE_KIND": kind},
        extra_prechecks=(P1.the_volume_is_the_planned_one, the_kind_records_are_the_bound_ones),
        on_poll=polls[0], products_secured=polls[1],
        artifact_spec="configs/stages/stage-2/teacher_modes/engineering_artifacts.json",
        teardown_note=("delete the pod, verify from the provider that it is gone, STOP; the network volume is KEPT: it "
                       "holds every shard and the phase's states"),
        driver=driver, validation_minutes=minutes, setup_minutes=20.0, command=command, description=description,
        relay_inputs=relay)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("issue", "launch", "closeout") and "--kind" in argv:
        i = argv.index("--kind")
        kind = argv[i + 1]
        if kind not in STUDY.kinds:
            print(f"REFUSED at $0: {kind!r} is not a kind of {STUDY.key} ({list(STUDY.kinds)})")
            return 11
        rest = argv[:i] + argv[i + 2:]
        minutes = 0.0
        if "--minutes" in rest:
            j = rest.index("--minutes")
            minutes = float(rest[j + 1])
            rest = rest[:j] + rest[j + 2:]
        if argv[0] == "issue":
            import argparse
            ap = argparse.ArgumentParser(prog=f"{STUDY.key} issue")
            ap.add_argument("--run-id", required=True)
            ap.add_argument("--expected-usd", type=float, required=True)
            ap.add_argument("--hard-cap-usd", type=float, required=True)
            ap.add_argument("--dry-run", action="store_true")
            ap.add_argument("--out", default=None)
            ap.add_argument("--note", default="")
            x = ap.parse_args(rest[1:])
            bad = kind_cap_problems(kind, x.hard_cap_usd)
            if bad:
                print(f"REFUSED at $0: {bad}")
                return 11
            try:
                path = SE.issue(STUDY, x.run_id, expected_usd=x.expected_usd, hard_cap_usd=x.hard_cap_usd,
                                dry_run=x.dry_run, out=Path(x.out) if x.out else None, note=x.note, kind=kind,
                                conditions=kind_conditions(kind))
            except SE.AuthorizationError as exc:
                print(f"REFUSED at $0: {exc}")
                return 11
            print(path)
            return 0
        if argv[0] == "launch":
            return SE.launch_main(STUDY, launch_spec(kind, minutes), rest[1:], description=__doc__.split("\n")[0],
                                  gpu=GPU)
        return SE.cli(STUDY, lambda: launch_spec(kind, minutes), rest, gpu=GPU, description=__doc__.split("\n")[0])
    return SE.cli(STUDY, launch_spec, argv, gpu=GPU, description=__doc__.split("\n")[0])


if __name__ == "__main__":
    raise SystemExit(main())
