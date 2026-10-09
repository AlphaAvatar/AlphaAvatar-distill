"""Materialize the behavioural arms ON THE POD, each gated on its identity.

The behavioural rungs train probes FROM five initializations whose bytes exist
nowhere a pod can cheaply reach: a 1.1-1.2 GiB checkpoint fits neither
transport (scp's 600 s per-asset timeout against a ~0.72 MB/s dev-box uplink,
and ~1.756 GiB of relay headroom), so every arm is REBUILT on the pod along a
digest-pinned path -- the same answer the state docs recorded and the same
machinery two closed campaigns already validated:

* the four CANDIDATES replay their search paths through `replay_specs` and
  `materialize_fixed_path`, exactly as `d1_replay_002` reproduced q2 and q4
  byte-identically (`differ: []` on all four identities, both leaves);
* the incumbent B is built from `phase_c2.baseline.frozen_baseline_spec`, the
  one owner of that construction, as C2's and C3's behavioural sessions did.

THE PLAN IS RESOLVED ON THE HOST THAT HAS THE JOURNAL. The search's state
journal is 67 MB in the durable store and does not travel; `write_plan` below
resolves it to a few tens of KB of pinned digests, committed at
`plans/d1_behavioural_replay_plan.json`, so the pod receives pins it must
satisfy rather than evidence it must trust. The pod still re-derives the root
state from the plan's own step-0 identities and re-checks every operator
config hash before an operator runs.

A DIGEST MISMATCH IS TERMINAL FOR THE SESSION, not retried: the paths are
deterministic, so the same inputs diverge the same way, and a probe trained
from a near-equivalent checkpoint measures something nobody asked about.

This module DECIDES NOTHING. The candidate set is the retention decision's,
the incumbent is the design's, every digest is the completed search's (or
C1's, for B), and the only outcomes are "byte-identical" and "STOP".
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[4]

#: The committed, deterministic resolution of the four finalists' pinned paths.
#: Regenerable from the durable-store journal; consumed by the pod from the
#: bundle. The preregistration binds its sha256.
PLAN_REL = "logs/stages/stage-1/phase_d1/plans/d1_behavioural_replay_plan.json"

#: Where arms are materialized on the pod. Execution configuration -- it never
#: enters the hashed session contract -- but named once, here, so the launcher's
#: driver command and the driver's default cannot disagree.
POD_ARM_ROOT = "/workspace/aad_arms"


class D1MaterializeError(RuntimeError):
    """An arm cannot be (or was not) materialized at its recorded identity."""


def write_plan(repo_root: str | Path = REPO_ROOT,
               journal: str | Path | None = None) -> dict[str, Any]:
    """Resolve ALL FOUR finalists' pinned paths from the journal. `$0`.

    `replay_specs.replay_plan` deliberately covers only the NOT-RETAINED
    finalists -- a replay rebuilds what is missing. A behavioural pod starts
    with nothing, so this plan covers the whole frozen field, each leaf built
    by the same `build_leaf` against the same journal and carrying the same
    pins. The root state, the operator-config agreement and the execution
    agreement travel with it, derived here and RE-CHECKED on the pod.
    """
    from stages.phase_d1 import d1_session as D1S
    from stages.phase_d1 import replay_specs as R

    root = Path(repo_root)
    states = R.load_states(journal or R.DEFAULT_JOURNAL)
    leaves = [R.build_leaf(states, m) for m in R.frozen_finalists(root)]
    if len(leaves) != 4:
        raise D1MaterializeError(
            f"the retention decision yields {len(leaves)} leaves, not 4; the "
            "behavioural field is the frozen Top-4 and nothing else")
    teacher = D1S.root_teacher_identity(root)
    root_state = R.derive_root_state(
        leaves, base_config=R.teacher_config(teacher["repo_id"],
                                             teacher["revision"]))
    configs = R.verify_operator_configs(leaves, repo_root=root)
    plan = R.describe(leaves, root_state)
    plan["schema"] = "aadistill.phase_d1.behavioural_replay_plan/v1"
    plan["_what_this_is"] = (
        "a digest-pinned reconstruction plan for the FOUR behavioural "
        "finalists, resolved from the completed search's journal on the host "
        "that holds it. It is NOT a search, NOT a new measurement and NOT a "
        "re-selection: the candidate set is read from the maintainer's "
        "retention decision and every step is pinned to the digest the search "
        "recorded. The behavioural pod consumes it to materialize the arms it "
        "trains probes from; the incumbent B is not in it because B's "
        "construction has one owner, phase_c2.baseline.frozen_baseline_spec.")
    plan["operator_config_agreement"] = configs
    plan["execution_agreement"] = R.verify_execution_config(leaves)
    dest = root / PLAN_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    return plan


def plan_sha256(repo_root: str | Path = REPO_ROOT) -> str:
    """The committed plan's byte hash, for the preregistration to bind."""
    import hashlib

    path = Path(repo_root) / PLAN_REL
    if not path.is_file():
        raise D1MaterializeError(
            f"{PLAN_REL} is missing; run behavioural_materialize.write_plan "
            "on the host that holds the search journal")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(path: str | Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != "aadistill.phase_d1.behavioural_replay_plan/v1":
        raise D1MaterializeError(
            f"{path} declares schema {doc.get('schema')!r}; the replay plan "
            "and the behavioural plan pin different leaf sets and neither may "
            "stand in for the other")
    return doc


def _verify_bytes_at(directory: Path, identities: dict[str, str],
                     *, what: str) -> dict[str, Any]:
    """Hash the bytes that exist and compare ALL FOUR identities. Exactly."""
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.specs.arch import ArchSpec, get_adapter
    from aadistill.initialization.specs.artifact import identify_checkpoint

    register_builtin_adapters()
    adapter = get_adapter("qwen3")
    config = json.loads((directory / "config.json").read_text())
    spec = ArchSpec.of("qwen3", {
        key: config[key] for key in (
            "hidden_size", "num_hidden_layers", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "vocab_size", "tie_word_embeddings")})
    identity = identify_checkpoint(directory, adapter=adapter, spec=spec,
                                   num_parameters=adapter.param_count(spec))
    got = {k: getattr(identity, k, None) for k in (
        "artifact_digest", "weights_digest", "single_shard_sha256",
        "arch_signature")}
    differ = sorted(k for k, want in identities.items()
                    if want and got.get(k) != want)
    if differ:
        raise D1MaterializeError(
            f"{what} at {directory} does not carry its recorded identity: "
            + "; ".join(f"{k}: built {got.get(k)}, recorded {identities[k]}"
                        for k in differ)
            + ". The paths are deterministic, so this is a materialization "
              "discrepancy to diagnose -- never a checkpoint to substitute.")
    return {"verified": sorted(k for k in identities if identities[k]),
            **{k: got[k] for k in sorted(got)}}


def materialize_candidate(leaf: Any, dest: Path, *, device: str,
                          repo_root: str | Path = REPO_ROOT,
                          root_loader: Callable[[], Any] | None = None,
                          execution: Any = None,
                          on_step: Callable[[Any], None] | None = None,
                          ) -> dict[str, Any]:
    """One finalist along its pinned path, into `dest`. Every step gated."""
    from aadistill.initialization.planning.fixed_path import (
        materialize_fixed_path,
    )
    from aadistill.initialization.specs.arch import get_adapter
    from stages.phase_d1 import replay_specs as R

    spec = R.fixed_path_spec(leaf, device=device, repo_root=repo_root)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = materialize_fixed_path(
        spec, adapter=get_adapter(R.FAMILY),
        root_loader=root_loader, workdir=dest,
        repo_root=repo_root, on_step=on_step, execution=execution)
    ok, detail = R.adoption_matches(leaf, results[-1].identity)
    if not ok:
        raise D1MaterializeError(
            f"{leaf.state_id}: reconstruction completed and does not carry the "
            f"recorded identity: {detail['differ']}. Terminal for this "
            "session; diagnose, never substitute.")
    return {"state_id": leaf.state_id, "spec_hash": spec.spec_hash,
            "seconds": round(time.time() - t0, 1),
            "checkpoint_path": results[-1].checkpoint_path,
            "identity": detail, "adopted": True}


def materialize_incumbent(dest: Path, *, device: str,
                          identities: dict[str, str],
                          repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """B, from its ONE construction owner, into `dest`. Gated on all four.

    `frozen_baseline_spec` is C1's own constructor re-exported by C2's
    baseline module -- the same four-step fixed path C2's and C3's behavioural
    sessions rebuilt B from -- and `assert_frozen_construction` refuses a
    recipe drift at `$0` before a tensor moves. The adoption gate here then
    compares what was BUILT against what the design BINDS, which is the half
    no CPU host can claim.
    """
    from aadistill.initialization.planning.fixed_path import (
        materialize_fixed_path,
    )
    from aadistill.initialization.specs.arch import get_adapter
    from stages.phase_c2 import baseline as BL
    from stages.phase_d1 import d1_session as D1S

    D1S._register_frozen_operators()
    spec = BL.frozen_baseline_spec(device=device)
    construction = BL.assert_frozen_construction(spec)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = materialize_fixed_path(
        spec, adapter=get_adapter("qwen3"), workdir=dest,
        repo_root=repo_root)
    built = Path(results[-1].checkpoint_path)
    identity = _verify_bytes_at(built, identities, what="incumbent B")
    return {"arm": "B", "spec_hash": spec.spec_hash,
            "construction": construction,
            "seconds": round(time.time() - t0, 1),
            "checkpoint_path": str(built), "identity": identity,
            "adopted": True}


def materialize_arms(rung: str, *, plan_path: str | Path,
                     arm_root: str | Path, device: str = "cuda",
                     advancing_candidate: str | None = None,
                     repo_root: str | Path = REPO_ROOT,
                     say: Callable[[str], None] = print) -> dict[str, Any]:
    """Every arm THIS rung trains from, present at its recorded identity.

    Arms whose bytes already sit at their resolved directory are verified and
    not rebuilt (idempotent restart); missing candidates replay their pinned
    paths; a missing B is built from its frozen spec. Scoped to the rung's own
    field -- a confirmation session materializes two arms, not five, because
    the priced cell funds two.
    """
    from stages.phase_d1 import behavioural as B
    from stages.phase_d1 import d1_session as D1S
    from stages.phase_d1 import replay_specs as R

    root = Path(repo_root)
    arm_root = Path(arm_root)
    field = B.arms(root, arm_root=str(arm_root))
    if rung == "confirmation":
        if not advancing_candidate:
            raise D1MaterializeError(
                "a confirmation materialization must name the advancing "
                "candidate; the priced cell funds two arms, not five")
        field = tuple(a for a in field
                      if a.arm_id == advancing_candidate or a.is_incumbent)

    plan = load_plan(plan_path)
    leaves = {leaf.state_id: leaf for leaf in R.leaves_from_plan(plan)}

    #: The historical execution state, re-derived ON THE POD from the plan's
    #: own identities before any weights load -- the three agreement checks the
    #: rematerialization campaign's failures bought (root state, operator
    #: configs, execution knobs).
    teacher = D1S.root_teacher_identity(root)
    needed = [leaves[a.state_id] for a in field
              if not a.is_incumbent and a.state_id in leaves]
    missing_plan = [a.state_id for a in field
                    if not a.is_incumbent and a.state_id not in leaves]
    if missing_plan:
        raise D1MaterializeError(
            f"the plan carries no leaf for {missing_plan}; it does not "
            "describe the frozen field and must be regenerated, not worked "
            "around")
    root_state = R.root_state_from_plan(
        plan, needed,
        base_config=R.teacher_config(teacher["repo_id"], teacher["revision"]))
    overrides = dict(root_state["config_overrides"])
    execution = R.replay_execution()
    execution_agreement = R.verify_execution_config(needed,
                                                    execution=execution)

    def load_root(_spec):
        return R.load_root_model(_spec, config_overrides=overrides,
                                 device=device)

    out: dict[str, Any] = {
        "schema": "aadistill.phase_d1.behavioural_arm_materialization/v1",
        "rung": rung, "arm_root": str(arm_root), "device": device,
        "root_state": root_state,
        "execution_agreement": execution_agreement,
        "arms": [],
    }
    #: Cheapest first operator first, the replay driver's own ordering: a
    #: wrong shared input is reported after about a minute rather than after
    #: the DEPTH operator's forty.
    first_cost = {"depth.causal_kl_greedy_v1": 2}
    ordered = sorted(
        (a for a in field if not a.is_incumbent),
        key=lambda a: first_cost.get(
            leaves[a.state_id].steps[0].impl_id, 1))
    for arm in ordered:
        dest = Path(arm.checkpoint_dir)
        if (dest / "config.json").is_file():
            say(f"[{arm.arm_id}] present at {dest}; verifying, not rebuilding")
            identity = _verify_bytes_at(dest, dict(arm.identities),
                                        what=arm.arm_id)
            out["arms"].append({"arm": arm.arm_id, "state_id": arm.state_id,
                                "reused": True, "identity": identity})
            continue
        say(f"[{arm.arm_id}] materializing {arm.state_id} along its pinned "
            "path")
        row = materialize_candidate(
            leaves[arm.state_id], dest, device=device, repo_root=root,
            root_loader=lambda: load_root(None), execution=execution,
            on_step=lambda r: say(
                f"  step {r.index} {r.impl_id} -> "
                f"{r.identity.artifact_digest[:12]} ({r.seconds:.0f}s)"))
        out["arms"].append({"arm": arm.arm_id, "reused": False, **row})

    incumbent = next(a for a in field if a.is_incumbent)
    dest = Path(incumbent.checkpoint_dir)
    if (dest / "config.json").is_file() or any(
            (dest / sub / "config.json").is_file()
            for sub in ([p.name for p in dest.iterdir()] if dest.is_dir()
                        else [])):
        say(f"[B] present under {dest}; verifying, not rebuilding")
        built = (dest if (dest / "config.json").is_file()
                 else next(p.parent for p in sorted(dest.rglob("config.json"))))
        identity = _verify_bytes_at(built, dict(incumbent.identities),
                                    what="incumbent B")
        out["arms"].append({"arm": "B", "reused": True, "identity": identity,
                            "checkpoint_path": str(built)})
    else:
        say("[B] materializing from phase_c2.baseline.frozen_baseline_spec")
        row = materialize_incumbent(dest, device=device,
                                    identities=dict(incumbent.identities),
                                    repo_root=root)
        out["arms"].append({"reused": False, **row})
    out["n_arms"] = len(out["arms"])
    return out


def resolve_arm_checkpoints(rung: str, arm_root: str | Path,
                            materialization: dict[str, Any], *,
                            advancing_candidate: str | None = None,
                            repo_root: str | Path = REPO_ROOT
                            ) -> dict[str, str]:
    """arm_id -> the exact directory a probe's `student_path` must use.

    B's fixed path may write its checkpoint under a step subdirectory of its
    workdir; the materialization row records where the bytes actually landed,
    and the probes must train from THAT, not from a guessed layout.
    """
    from stages.phase_d1 import behavioural as B

    rows = {str(r.get("arm") or r.get("state_id")): r
            for r in materialization["arms"]}
    by_state = {str(r.get("state_id")): r for r in materialization["arms"]
                if r.get("state_id")}
    out: dict[str, str] = {}
    for arm in B.arms(Path(repo_root), arm_root=str(arm_root)):
        if rung == "confirmation" and not (
                arm.is_incumbent or arm.arm_id == advancing_candidate):
            continue
        row = rows.get(arm.arm_id) or by_state.get(arm.state_id or "")
        if row is None:
            raise D1MaterializeError(
                f"{arm.arm_id} has no materialization row; a probe cannot "
                "train from a checkpoint nobody built or verified")
        path = str(row.get("checkpoint_path") or arm.checkpoint_dir)
        if not (Path(path) / "config.json").is_file():
            raise D1MaterializeError(
                f"{arm.arm_id}: no checkpoint at {path} after materialization")
        out[arm.arm_id] = path
    return out


__all__ = ["PLAN_REL", "POD_ARM_ROOT", "D1MaterializeError", "load_plan",
           "materialize_arms", "materialize_candidate",
           "materialize_incumbent", "plan_sha256",
           "resolve_arm_checkpoints", "write_plan"]
