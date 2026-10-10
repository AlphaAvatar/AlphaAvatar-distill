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

#: Where the launcher acknowledges that a scored probe's evidence is durable
#: off-pod, and therefore that the pod may release that probe's local training
#: bytes. Under the audit dir on purpose: the acks are tiny JSON and they are
#: themselves evidence of the release boundary. One constant, imported by the
#: driver (which reads it) and the launcher (which writes it).
RELEASE_ACK_REL = "artifacts/audit/autoinit_d1_behavioural/release_acks"


class D1MaterializeError(RuntimeError):
    """A materialization premise does not hold. ORDINARY ENGINEERING.

    A missing plan leaf, an unreadable journal, a wrong schema, a path that is
    not where a row said it was, an execution-configuration disagreement
    detected BEFORE an operator runs -- every one of these is a harness error:
    diagnose, repair, retry (P12.1). None of them says anything about the
    checkpoints, and none of them may be reported as a scientific finding.
    """


class D1ArmIdentityMismatch(D1MaterializeError):
    """A COMPLETED pinned path does not carry its recorded identity.

    The scientific stop condition, and ONLY this: the root state, the operator
    configs and the execution knobs were all verified against the search's own
    records before the path ran -- the conditions were demonstrably reproduced
    -- and the bytes still differ. That is a deterministic-materialization
    discrepancy to diagnose under review, never an error to retry and never a
    checkpoint to substitute. The driver maps exactly this type to
    DIGEST_MISMATCH; everything else is RUN_FAILED.
    """


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


def final_checkpoint_in(dest: Path, *, n_steps: int) -> tuple[Path | None, bool]:
    """Where a previously materialized path's FINAL checkpoint is, if anywhere.

    `materialize_fixed_path` writes every step's checkpoint under
    `workdir/steps/{i:02d}_{kind}/` -- the final checkpoint of an n-step path
    is the step n-1 directory, NOT `workdir/config.json`. The first version of
    this module verified `dest/config.json` after a full rematerialization,
    which is a directory the producer never writes: the session would have
    paid for every arm and then refused its own output.

    Returns `(path, is_complete)`:

    * `(dest, True)` when the bytes are a FLATTENED checkpoint (`config.json`
      at the top level) -- the shape of a restored durable-store copy;
    * `(step_dir, True)` when the LAST step's checkpoint exists;
    * `(step_dir, False)` when only an EARLIER step exists -- a partial path
      from an interrupted run, which is rebuilt rather than adopted;
    * `(None, False)` when nothing is there.
    """
    if (dest / "config.json").is_file():
        return dest, True
    steps = dest / "steps"
    if not steps.is_dir():
        return None, False
    done = sorted(p for p in steps.iterdir()
                  if p.is_dir() and (p / "config.json").is_file())
    if not done:
        return None, False
    last = done[-1]
    index = int(last.name.split("_", 1)[0])
    return last, index == n_steps - 1


def release_intermediate_steps(label: str, results: Any,
                               workdir: Path) -> dict[str, Any]:
    """Delete an arm's intermediate step checkpoints, keeping the final one.

    C2's production repair, reapplied where the same program shape reappeared:
    `materialize_fixed_path` writes every step of a four-step path under the
    arm's workdir and returns them all, and C2 behavioural attempt3 died at
    `No space left on device` with five full paths resident -- 32.8 minutes of
    completed compute lost to bytes nothing would ever read again. An
    intermediate step's checkpoint has no consumer once the NEXT step's bytes
    exist; the final checkpoint is the arm.

    Same guards as C2's call site, because this deletes: never the final
    checkpoint, and never a path outside this arm's own workdir. NEVER RAISES
    -- a cleanup error must not destroy a verified arm -- and the CALLER fails
    closed on a non-empty `failed`, because the 120 GB provision is derived on
    the assumption these are freed.
    """
    import shutil

    final = Path(results[-1].checkpoint_path).resolve()
    freed, removed, failed = 0, [], []
    for step in results[:-1]:
        path = Path(step.checkpoint_path).resolve()
        if path == final or not path.is_relative_to(Path(workdir).resolve()):
            continue
        try:
            size = sum(f.stat().st_size for f in path.rglob("*")
                       if f.is_file())
            shutil.rmtree(path)
            freed += size
            removed.append(step.impl_id)
        except OSError as exc:                                  # noqa: PERF203
            failed.append(f"{step.impl_id}: {exc}")
    return {"arm": label, "removed_steps": removed,
            "freed_gib": round(freed / 2**30, 3), "failed": failed,
            "kept": str(final)}


def materialize_candidate(leaf: Any, dest: Path, *, device: str,
                          repo_root: str | Path = REPO_ROOT,
                          root_loader: Callable[[], Any] | None = None,
                          execution: Any = None,
                          on_step: Callable[[Any], None] | None = None,
                          ) -> dict[str, Any]:
    """One finalist along its pinned path, into `dest`. Every step gated.

    Raises `D1ArmIdentityMismatch` -- the scientific stop -- only when the
    path COMPLETED under the verified inputs and the bytes still differ: a
    mid-path `FixedPathDigestMismatch` or a failed final adoption. Everything
    else that can go wrong here is a harness error and stays the base type.
    """
    from aadistill.initialization.planning.fixed_path import (
        FixedPathDigestMismatch, materialize_fixed_path,
    )
    from aadistill.initialization.specs.arch import get_adapter
    from stages.phase_d1 import replay_specs as R

    spec = R.fixed_path_spec(leaf, device=device, repo_root=repo_root)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        results = materialize_fixed_path(
            spec, adapter=get_adapter(R.FAMILY),
            root_loader=root_loader, workdir=dest,
            repo_root=repo_root, on_step=on_step, execution=execution)
    except FixedPathDigestMismatch as exc:
        raise D1ArmIdentityMismatch(
            f"{leaf.state_id}: a pinned step diverged after the root state, "
            f"the operator configs and the execution knobs were all verified "
            f"against the search's own records: {exc}. The conditions were "
            "reproduced and the bytes differ; this goes to review, is not "
            "retried, and no near-equivalent checkpoint is substituted."
        ) from exc
    ok, detail = R.adoption_matches(leaf, results[-1].identity)
    if not ok:
        raise D1ArmIdentityMismatch(
            f"{leaf.state_id}: reconstruction completed and does not carry "
            f"the recorded identity: {detail['differ']}. Diagnose, never "
            "substitute.")
    released = release_intermediate_steps(leaf.state_id, results, dest)
    return {"state_id": leaf.state_id, "spec_hash": spec.spec_hash,
            "intermediates_released": released,
            "seconds": round(time.time() - t0, 1),
            #: THE PRODUCER'S OWN ANSWER: the last step's checkpoint
            #: directory. Every consumer -- the identity verification, the
            #: probes' student_path, the contract's byte check -- reads THIS,
            #: never a layout guessed beside it.
            "checkpoint_path": results[-1].checkpoint_path,
            "identity": detail, "adopted": True}


def incumbent_execution():
    """B's NUMERICAL EXECUTION PROTOCOL, read from its authoritative record.

    A3's terminal finding is that `53e30566…` -- canonical B -- is reproduced
    exactly only under `A_bsz1`: `micro_batch_size=1`, unpadded,
    `original_order_v1` (`a3_comparison.json`; A-bsz3 at mbs=3 produced
    `7dd2f6f6…`, a DISTINCT numerical materialization protocol from the same
    path). `stages.phase_a3.a_bsz3.A_BSZ1` is the one place that protocol is
    declared, so it is imported, never retyped -- and never defaulted:
    `materialize_fixed_path` falls back to `DEFAULT_EXECUTION`
    (micro_batch_size=4), which is a third protocol nothing measured B under,
    and the D1 candidates' bsz=3 policy belongs to the candidates' own
    recorded steps, not to B.
    """
    from stages.phase_a3.a_bsz3 import A_BSZ1

    return A_BSZ1


def materialize_incumbent(dest: Path, *, device: str,
                          identities: dict[str, str],
                          repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """B, from its ONE construction owner, into `dest`. Gated on all four.

    `frozen_baseline_spec` is C1's own constructor re-exported by C2's
    baseline module -- the same four-step fixed path C2's and C3's behavioural
    sessions rebuilt B from -- and `assert_frozen_construction` refuses a
    recipe drift at `$0` before a tensor moves. The EXECUTION is A3's
    `A_bsz1`, the protocol canonical B is proven to reproduce under; see
    `incumbent_execution`. The adoption gate then compares what was BUILT
    against what the design BINDS, which is the half no CPU host can claim --
    and because the construction spec and the execution protocol were both
    verified against their authoritative records BEFORE the build, a mismatch
    here is `D1ArmIdentityMismatch`, the scientific stop.
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
    execution = incumbent_execution()
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = materialize_fixed_path(
        spec, adapter=get_adapter("qwen3"), workdir=dest,
        repo_root=repo_root, execution=execution)
    built = Path(results[-1].checkpoint_path)
    try:
        identity = _verify_bytes_at(built, identities, what="incumbent B")
    except D1MaterializeError as exc:
        raise D1ArmIdentityMismatch(str(exc)) from exc
    released = release_intermediate_steps("B", results, dest)
    return {"arm": "B", "spec_hash": spec.spec_hash,
            "construction": construction,
            "execution": execution.as_fingerprint(),
            "intermediates_released": released,
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
    #: All four process-global registries, once, before any spec is built:
    #: `frozen_baseline_spec` and `fixed_path_spec` both construct against
    #: them, and a partial registration has already killed a paid step 0.
    D1S._register_frozen_operators()
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
        leaf = leaves[arm.state_id]
        dest = Path(arm.checkpoint_dir)
        found, complete = final_checkpoint_in(dest, n_steps=len(leaf.steps))
        if found is not None and complete:
            #: A COMPLETED path's bytes at rest: adopt only at the recorded
            #: identity. A mismatch HERE is restored-or-leftover bytes that
            #: are not what they claim -- an engineering state to rebuild
            #: from, not a scientific finding about a path nobody just ran.
            say(f"[{arm.arm_id}] final checkpoint present at {found}; "
                "verifying, not rebuilding")
            try:
                identity = _verify_bytes_at(found, dict(arm.identities),
                                            what=arm.arm_id)
                out["arms"].append({
                    "arm": arm.arm_id, "state_id": arm.state_id,
                    "reused": True, "identity": identity,
                    "checkpoint_path": str(found)})
                continue
            except D1MaterializeError as exc:
                say(f"[{arm.arm_id}] existing bytes fail their identity "
                    f"({str(exc)[:120]}); rebuilding along the pinned path")
        elif found is not None:
            say(f"[{arm.arm_id}] partial path at {found} (interrupted run); "
                "rebuilding along the pinned path")
        say(f"[{arm.arm_id}] materializing {arm.state_id} along its pinned "
            "path")
        row = materialize_candidate(
            leaf, dest, device=device, repo_root=root,
            root_loader=lambda: load_root(None), execution=execution,
            on_step=lambda r: say(
                f"  step {r.index} {r.impl_id} -> "
                f"{r.identity.artifact_digest[:12]} ({r.seconds:.0f}s)"))
        out["arms"].append({"arm": arm.arm_id, "reused": False, **row})
        released = row["intermediates_released"]
        say(f"  released {len(released['removed_steps'])} intermediate(s), "
            f"{released['freed_gib']:.2f} GiB")
        if released["failed"]:
            raise D1MaterializeError(
                f"{arm.arm_id}: intermediate release failed "
                f"({released['failed']}). The 120 GB provision is derived on "
                "the assumption each arm's construction intermediates are "
                "freed when it is verified; that assumption is falsified, so "
                "NO FURTHER ARM MAY BE BUILT under a bound that does not "
                "hold. The verified arm and its evidence are preserved.")

    incumbent = next(a for a in field if a.is_incumbent)
    dest = Path(incumbent.checkpoint_dir)
    from stages.phase_c2 import baseline as BL

    b_steps = len(BL.frozen_baseline_spec(device=device).steps)
    found, complete = final_checkpoint_in(dest, n_steps=b_steps)
    row = None
    if found is not None and complete:
        say(f"[B] final checkpoint present at {found}; verifying, not "
            "rebuilding")
        try:
            identity = _verify_bytes_at(found, dict(incumbent.identities),
                                        what="incumbent B")
            row = {"arm": "B", "reused": True, "identity": identity,
                   "checkpoint_path": str(found)}
        except D1MaterializeError as exc:
            say(f"[B] existing bytes fail their identity "
                f"({str(exc)[:120]}); rebuilding from the frozen spec")
    if row is None:
        say("[B] materializing from phase_c2.baseline.frozen_baseline_spec "
            "under A_bsz1")
        row = {"reused": False,
               **materialize_incumbent(dest, device=device,
                                       identities=dict(incumbent.identities),
                                       repo_root=root)}
        released = row["intermediates_released"]
        if released["failed"]:
            raise D1MaterializeError(
                f"B: intermediate release failed ({released['failed']}); the "
                "storage bound is falsified and the session stops with the "
                "verified arm preserved.")
    out["arms"].append(row)
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


__all__ = ["PLAN_REL", "POD_ARM_ROOT", "RELEASE_ACK_REL", "D1ArmIdentityMismatch",
           "D1MaterializeError", "final_checkpoint_in", "incumbent_execution",
           "load_plan", "materialize_arms", "materialize_candidate",
           "release_intermediate_steps",
           "materialize_incumbent", "plan_sha256",
           "resolve_arm_checkpoints", "write_plan"]
