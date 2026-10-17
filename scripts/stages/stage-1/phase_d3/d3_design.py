"""The new D3: the fixed-q2 DEPTH teacher-reference ablation. One owner.

The question (maintainer decision 2026-10-16): given D1-q2's EXACT structural
path, does choosing the DEPTH layers by minimizing KL against the original
TEACHER, instead of the FFN-compressed PARENT, give a better initialization for
downstream capability recovery?

A single-factor, FIXED-PATH ablation -- not a search:

    control    FFN(domain_balanced@v1) -> DEPTH(reasoning_heavy@v2, KL(P||C(S)))
               -> RESIDUAL_WIDTH(reasoning_heavy@v2) -> ATTENTION(domain_balanced@v1)
    treatment  the same path, the same operators, profiles, positions, support,
               aggregation, greedy removal, geometry and tie-break, with DEPTH's
               KL reference declared as the stored ROOT:  KL(T||C(S))

The control is D1-q2 itself, reconstructed from the committed digest-pinned
replay plan (`plans/d1_behavioural_replay_plan.json`) that reproduced it
byte-exactly twice. The treatment is that spec with exactly these differences,
and `treatment_contract` refuses any other:

* the path id;
* step 1 (DEPTH) declares ``kl_reference: root_reference/v1`` and pins the
  operator-config hash that declaration derives to (registered at $0);
* steps 1-3 carry no expected OUTPUT digest -- their outputs have never existed;
  steps 2-3 keep q2's pinned INPUT config hashes, because their configs do not
  move; step 0, the FFN prefix, keeps q2's pins on both;
* the step labels name the treatment.

Switching the reference from the parent to the teacher also switches which
token ids form the reference's Top-200 support -- support is defined by the
reference. That is the declared reference switch itself, NOT D2's
`reference_student_union_topk_tail_v1`; no D2 weighting, reward, union support
or statistics weighting is used, and every operator other than DEPTH keeps its
own local rule.

Instance facts (state ids, digests, paths) live here and in the registration;
the reusable mechanism is `aadistill.initialization.scoring.kl_reference`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[4]

EXPERIMENT_ID = "phase_d3"
STAGE_ID = "1"
REGISTRATION = "logs/stages/stage-1/phase_d3/plans/d3_registration_v1.json"
CONTROL_RECORD = "logs/stages/stage-1/phase_d3/plans/d3_control_q2.json"
#: The frozen control, as the maintainer named it and the D1 records hold it.
CONTROL_STATE_ID = "afd50c9ff0e3d20542bb5473ea745b55"
CONTROL_ARTIFACT_DIGEST = (
    "3a4b4ede6888e2b0242d59234f1df18f0cac02893617e678992e54eaa243b82b")
CONTROL_PATH_LABEL = (
    "FFN(calib.domain_balanced@v1)->DEPTH(calib.reasoning_heavy@v2)->"
    "RESIDUAL_WIDTH(calib.reasoning_heavy@v2)->ATTENTION(calib.domain_balanced@v1)")
#: The committed digest-pinned plan the control is rebuilt from.
CONTROL_PLAN = "logs/stages/stage-1/phase_d1/plans/d1_behavioural_replay_plan.json"
TREATMENT_PATH_ID = "d3.fixed_q2_depth_teacher_reference"
TREATED_KIND = "DEPTH"

#: Structural outcomes. Exactly one per completed structural session.
NO_STRUCTURAL_CHANGE = "NO_STRUCTURAL_CHANGE"
STRUCTURAL_CHANGE = "STRUCTURAL_CHANGE"
#: The selected layers equal q2's and the bytes do not: a materialization
#: discrepancy to DIAGNOSE before anything is called a scientific effect.
UNEXPLAINED_BYTE_DIFFERENCE = "UNEXPLAINED_BYTE_DIFFERENCE"
STRUCTURAL_OUTCOMES = (NO_STRUCTURAL_CHANGE, STRUCTURAL_CHANGE,
                       UNEXPLAINED_BYTE_DIFFERENCE)
#: NOT an outcome: the driver status of a run whose parent-reference
#: comparability gate failed (`parent_reference_admission`). Its evidence is
#: kept; it is never classified and proposes nothing.
INADMISSIBLE = "INADMISSIBLE"
#: The gate's tolerance and the GPU evidence it was fixed from, before any D3
#: execution (`write_d3_admission.py`).
ADMISSION = ("logs/stages/stage-1/phase_d3/plans/"
             "d3_parent_reference_admission_v1.json")
IDENTITIES = ("artifact_digest", "weights_digest", "single_shard_sha256",
              "arch_signature")


class D3DesignError(RuntimeError):
    """The D3 design cannot be built or does not hold as registered."""


class D3Inadmissible(RuntimeError):
    """The treated DEPTH step's parent-reference diagnostic does not reproduce
    q2's round 0: the run cannot support a structural conclusion."""

    def __init__(self, admission: Mapping[str, Any]):
        self.admission = dict(admission)
        super().__init__("parent-reference comparability gate failed: "
                         + "; ".join(self.admission.get("failures") or []))


def _treatment_config() -> dict[str, str]:
    from aadistill.initialization.scoring.kl_reference import (
        KL_REFERENCE_KEY, ROOT_REFERENCE_V1,
    )

    return {KL_REFERENCE_KEY: ROOT_REFERENCE_V1}


# --- the control ------------------------------------------------------------

def control_plan(repo_root: str | Path = REPO) -> tuple[dict[str, Any], str]:
    """The committed D1 replay plan and its byte hash."""
    from stages.phase_d1 import behavioural_materialize as BM

    path = Path(repo_root) / CONTROL_PLAN
    if not path.is_file():
        raise D3DesignError(f"{CONTROL_PLAN} is missing; the control cannot be "
                            "rebuilt from anything pinned")
    return BM.load_plan(path), hashlib.sha256(path.read_bytes()).hexdigest()


def control_leaf(repo_root: str | Path = REPO):
    """D1-q2 as a fully pinned `ReplayLeaf`, checked against the named control."""
    from stages.phase_d1 import replay_specs as R

    plan, _ = control_plan(repo_root)
    leaves = {leaf.state_id: leaf for leaf in R.leaves_from_plan(plan)}
    leaf = leaves.get(CONTROL_STATE_ID)
    if leaf is None:
        raise D3DesignError(f"the replay plan carries no leaf {CONTROL_STATE_ID}")
    if leaf.expected_artifact_digest != CONTROL_ARTIFACT_DIGEST:
        raise D3DesignError(
            f"the plan's q2 artifact digest is {leaf.expected_artifact_digest}, "
            f"not the frozen control {CONTROL_ARTIFACT_DIGEST}")
    if leaf.path_label != CONTROL_PATH_LABEL:
        raise D3DesignError(f"the plan's q2 path is {leaf.path_label!r}, not "
                            f"{CONTROL_PATH_LABEL!r}")
    return leaf


def treated_index(leaf) -> int:
    """The control path's DEPTH position. Derived, never typed."""
    hits = [s.index for s in leaf.steps if s.kind == TREATED_KIND]
    if len(hits) != 1:
        raise D3DesignError(f"the control path has {len(hits)} {TREATED_KIND} "
                            "steps; a fixed path applies each kind once")
    return hits[0]


def control_spec(leaf, *, device: str = "cuda"):
    from stages.phase_d1 import replay_specs as R

    return R.fixed_path_spec(leaf, device=device, repo_root=REPO)


def root_identity(leaf) -> dict[str, str]:
    """What the teacher reference store declares as its root: the teacher id
    with its revision, and the content hash of its shards."""
    return {"teacher_id": leaf.root_teacher_id,
            "teacher_sha256": leaf.root_teacher_sha256}


def execution():
    """The control's numerical execution: the search's own (bsz 3,
    length-sorted), as every q2 step recorded. Read from its owner."""
    from stages.phase_d1 import replay_specs as R

    return R.replay_execution()


# --- the treatment ------------------------------------------------------------

def treatment_spec(leaf, *, device: str = "cuda",
                   treated_config_hash: str | None = None):
    """The control spec with ONLY the declared DEPTH reference switched.

    `treated_config_hash` is the pin for step 1's derived operator config
    (the registration's); without it the treated step carries no input pin
    (the $0 derivation that produces the pin builds the spec this way).
    """
    from aadistill.initialization.planning.fixed_path import (
        FixedPathSpec, FixedPathStep,
    )

    ctrl = control_spec(leaf, device=device)
    at = treated_index(leaf)
    steps = []
    for i, s in enumerate(ctrl.steps):
        if i < at:
            steps.append(s)
        elif i == at:
            steps.append(FixedPathStep(
                impl_id=s.impl_id, profile_id=s.profile_id,
                config=_treatment_config(),
                expected_config_hash=treated_config_hash,
                label=f"{s.label}+teacher_reference"))
        else:
            steps.append(FixedPathStep(
                impl_id=s.impl_id, profile_id=s.profile_id,
                expected_config_hash=s.expected_config_hash,
                label=f"{s.label}@d3"))
    return FixedPathSpec(
        path_id=TREATMENT_PATH_ID, family=ctrl.family,
        target_spec=ctrl.target_spec, steps=tuple(steps),
        root_repo_id=ctrl.root_repo_id, root_revision=ctrl.root_revision,
        stats_spec=ctrl.stats_spec, device=ctrl.device, seed=ctrl.seed,
        max_shard_size=ctrl.max_shard_size,
        position_policy=ctrl.position_policy,
        distribution_support=ctrl.distribution_support,
        scoring_objective=ctrl.scoring_objective)


def treatment_contract(control, treatment) -> dict[str, Any]:
    """Every difference between the two specs, refused unless declared.

    The single-factor claim is made BY CONSTRUCTION here, and checked: a
    difference outside the declared set means the treatment is not this
    ablation.
    """
    from aadistill.initialization.scoring.kl_reference import KL_REFERENCE_KEY

    c, t = control.as_dict(), treatment.as_dict()
    at = [i for i, s in enumerate(control.steps)
          if s.impl_id.startswith("depth.")]
    if len(at) != 1:
        raise D3DesignError("the control path does not have exactly one DEPTH")
    at = at[0]
    problems: list[str] = []
    for key in sorted(set(c) | set(t)):
        if key in ("path_id", "steps"):
            continue
        if c.get(key) != t.get(key):
            problems.append(f"spec field {key!r} differs")
    if len(c["steps"]) != len(t["steps"]):
        problems.append("the step count differs")
    differences: list[dict[str, Any]] = []
    for i, (cs, ts) in enumerate(zip(c["steps"], t["steps"])):
        for key in sorted(set(cs) | set(ts)):
            if cs.get(key) == ts.get(key):
                continue
            allowed = (
                key == "label"
                or (key == "expected_artifact_digest" and i >= at
                    and ts.get(key) is None)
                or (key == "config" and i == at
                    and ts.get(key) == _treatment_config())
                or (key == "expected_config_hash" and i == at))
            row = {"step": i, "field": key, "control": cs.get(key),
                   "treatment": ts.get(key), "declared": allowed}
            differences.append(row)
            if not allowed:
                problems.append(f"step {i} {key} differs outside the treatment")
    for i, ts in enumerate(t["steps"]):
        if i != at and KL_REFERENCE_KEY in (ts.get("config") or {}):
            problems.append(f"step {i} declares a KL reference")
    if problems:
        raise D3DesignError("the treatment is not the single-factor ablation:\n"
                            "  - " + "\n  - ".join(problems))
    return {"control_path_hash": control.spec_hash,
            "treatment_path_hash": treatment.spec_hash,
            "treated_step": at, "treatment_config": _treatment_config(),
            "differences": differences,
            "unchanged": ["target_spec", "root", "seed", "stats_spec",
                          "position_policy", "distribution_support",
                          "scoring_objective (none)", "every operator id and "
                          "profile", "step 0's output and input pins",
                          "steps 2-3's input pins"],
            "_reference_support": (
                "the Top-K support of a reference-defined sketch is the "
                "REFERENCE's own Top-K, so switching the reference from the "
                "parent to the teacher switches which token ids form the "
                "support. That is the declared reference switch itself, not "
                "D2's reference_student_union_topk_tail_v1")}


def operator_config_hashes(spec, repo_root: str | Path = REPO
                           ) -> list[dict[str, Any]]:
    """Every step's derived operator config hash, at $0 from the mixtures."""
    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )
    from aadistill.initialization.calibration.profiles import get_profile
    from aadistill.initialization.operators.base import get_implementation
    from aadistill.initialization.planning.operator_config import (
        hashed_operator_config, operator_config_hash,
    )

    root = Path(repo_root)
    items: dict[str, Any] = {}
    rows = []
    for i, step in enumerate(spec.steps):
        if step.profile_id not in items:
            profile = get_profile(step.profile_id)
            items[step.profile_id] = prepare_calibration_items(
                profile.resolve(root), profile_id=profile.qualified_id)
        config = hashed_operator_config(
            implementation=get_implementation(step.impl_id),
            policy=spec.position_policy, support=spec.distribution_support,
            items=items[step.profile_id], declared=step.config,
            objective=spec.scoring_objective)
        rows.append({"step": i, "impl_id": step.impl_id,
                     "profile_id": step.profile_id,
                     "config_hash": operator_config_hash(config),
                     "declared": dict(step.config or {}),
                     "pinned": step.expected_config_hash})
    return rows


# --- the registration ------------------------------------------------------------

def registration(repo_root: str | Path = REPO) -> dict[str, Any]:
    path = Path(repo_root) / REGISTRATION
    if not path.is_file():
        raise D3DesignError(f"{REGISTRATION} is missing; D3 has no registered "
                            "design")
    doc = json.loads(path.read_text())
    from aadistill.infrastructure.manifest import sha256_json

    body = {k: v for k, v in doc.items() if k != "registration_hash"}
    if doc.get("registration_hash") != sha256_json(body):
        raise D3DesignError(f"{REGISTRATION} does not match its own "
                            "registration_hash")
    return doc


def registration_hash(repo_root: str | Path = REPO) -> str:
    return str(registration(repo_root)["registration_hash"])


def registered_treatment_spec(repo_root: str | Path = REPO, *,
                              device: str = "cuda"):
    """The treatment exactly as registered: the DEPTH input pin from the
    registration, the spec hash required equal to the registered one."""
    doc = registration(repo_root)
    leaf = control_leaf(repo_root)
    spec = treatment_spec(
        leaf, device=device,
        treated_config_hash=doc["treatment"]["treated_step_config_hash"])
    want = doc["treatment"]["treatment_path_hash"]
    if device == doc["treatment"]["device"] and spec.spec_hash != want:
        raise D3DesignError(
            f"the treatment spec hashes to {spec.spec_hash[:16]} and the "
            f"registration binds {want[:16]}")
    return spec


# --- the structural outcome ----------------------------------------------------------

def classify_structure(*, control: Mapping[str, Any],
                       depth_kept: list[int],
                       step_identities: list[Mapping[str, Any]]
                       ) -> dict[str, Any]:
    """The structural outcome, from the control record and what was built.

    `control` is the control record's ``steps`` / ``depth`` blocks;
    `step_identities` is each treatment step's four identities in path order.
    Unchanged layers must reproduce q2's downstream bytes at EVERY step, not
    just the last -- a coincidence at the leaf would hide a compensating pair.
    """
    same_layers = sorted(depth_kept) == sorted(control["depth"]["kept_layers"])
    per_step = []
    for i, (ctrl, got) in enumerate(zip(control["steps"], step_identities)):
        per_step.append({
            "step": i, "kind": ctrl["kind"],
            "control_artifact_digest": ctrl["artifact_digest"],
            "treatment_artifact_digest": got.get("artifact_digest"),
            "equal": ctrl["artifact_digest"] == got.get("artifact_digest")})
    final_equal = all(
        step_identities[-1].get(k) == control["final"][k] for k in IDENTITIES)
    downstream = [r for r in per_step if r["step"] >= control["depth"]["step"]]
    if same_layers and final_equal and all(r["equal"] for r in downstream):
        outcome = NO_STRUCTURAL_CHANGE
    elif same_layers:
        outcome = UNEXPLAINED_BYTE_DIFFERENCE
    elif final_equal:
        #: Different layers and identical bytes cannot both be true.
        outcome = UNEXPLAINED_BYTE_DIFFERENCE
    else:
        outcome = STRUCTURAL_CHANGE
    return {"outcome": outcome, "same_kept_layers": same_layers,
            "control_kept_layers": list(control["depth"]["kept_layers"]),
            "treatment_kept_layers": sorted(depth_kept),
            "layers_only_in_control": sorted(
                set(control["depth"]["kept_layers"]) - set(depth_kept)),
            "layers_only_in_treatment": sorted(
                set(depth_kept) - set(control["depth"]["kept_layers"])),
            "final_identities_equal": final_equal, "per_step": per_step,
            "next": {
                NO_STRUCTURAL_CHANGE: ("report NO_STRUCTURAL_CHANGE; preserve "
                                       "the evidence; NO recovery training"),
                STRUCTURAL_CHANGE: ("propose the separately authorized "
                                    "three-probe behavioural rung"),
                UNEXPLAINED_BYTE_DIFFERENCE: (
                    "DIAGNOSE the materialization before anything is called a "
                    "scientific effect; no probe is proposed"),
            }[outcome]}


def depth_evidence(depth_step: Mapping[str, Any],
                   control: Mapping[str, Any]) -> dict[str, Any]:
    """The treatment DEPTH's decision record, set beside the control's.

    `depth_step` is the persisted fixed-path step result (`StepResult.as_dict`)
    of the treated DEPTH step; `control` the control record. Per round: the
    candidates tested, the teacher-reference KL each scored (the DECISION),
    the parent-reference KL of the same candidate (the DIAGNOSTIC), the layer
    chosen, and which layer the parent reference would have chosen at that
    same state. Whether round 0's parent diagnostic reproduces q2 is the
    separate, fail-closed `parent_reference_admission`.
    """
    selection = depth_step.get("selection") or {}
    rounds = selection.get("search_rounds") or []
    cands = selection.get("reference_candidates") or {}
    trace = depth_step.get("trace") or {}
    if not rounds or not cands:
        raise D3DesignError("the DEPTH step carries no round tables or "
                            "reference records; it did not run under the "
                            "declared reference")

    def key(removed, c):
        return ",".join(str(x) for x in sorted([*removed, c]))

    table = []
    disagreements = 0
    for r in rounds:
        rows = []
        for e in r["table"]:
            rec = cands[key(r["removed_before"], e["candidate"])]
            if rec["root_reference_kl"] != e["score"]:
                raise D3DesignError(
                    f"round {r['round']} candidate {e['candidate']}: the "
                    "decision table and the reference record disagree")
            rows.append({"candidate": e["candidate"],
                         "teacher_reference_kl": e["score"],
                         "parent_reference_kl": rec["parent_reference_kl"]})
        by_parent = min(rows, key=lambda x: (x["parent_reference_kl"],
                                             x["candidate"]))["candidate"]
        disagreements += by_parent != r["chosen"]
        table.append({"round": r["round"], "removed_before": r["removed_before"],
                      "n_candidates": r["n_candidates"], "chosen": r["chosen"],
                      "chosen_teacher_reference_kl": r["chosen_score"],
                      "parent_reference_would_choose": by_parent,
                      "candidates": rows})
    return {
        "removal_order": trace.get("removal_order"),
        "kept_layers": trace.get("kept_layers"),
        "removed_layers": trace.get("removed_layers"),
        "control_removal_order": control["depth"]["removal_order"],
        "control_kept_layers": control["depth"]["kept_layers"],
        "rounds": table,
        "rounds_where_the_parent_reference_would_choose_differently":
            disagreements,
        "kl_reference": trace.get("kl_reference"),
    }


# --- the parent-reference comparability gate -------------------------------------

def admission_record(repo_root: str | Path = REPO) -> dict[str, Any]:
    """The committed gate, checked against itself, the registration and the
    control record it compares with."""
    from aadistill.infrastructure.manifest import sha256_json

    path = Path(repo_root) / ADMISSION
    if not path.is_file():
        raise D3DesignError(f"{ADMISSION} is missing; the structural session "
                            "has no comparability gate")
    doc = json.loads(path.read_text())
    body = {k: v for k, v in doc.items() if k != "admission_hash"}
    control = Path(repo_root) / CONTROL_RECORD
    problems = [why for ok, why in (
        (doc.get("admission_hash") == sha256_json(body),
         "the record does not match its own admission_hash"),
        (doc.get("registration_hash") == registration_hash(repo_root),
         "it binds another registration"),
        (doc.get("control_record_sha256")
         == hashlib.sha256(control.read_bytes()).hexdigest(),
         "it was fixed against another control record"))
        if not ok]
    if problems:
        raise D3DesignError(f"{ADMISSION}: " + "; ".join(problems))
    return doc


def parent_reference_admission(depth_step: Mapping[str, Any],
                               control: Mapping[str, Any],
                               admission: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the treated DEPTH step's parent is q2's, measured. FAIL-CLOSED.

    Round 0's candidates are single layers of the step's parent under q2's
    reduction, so the parent-reference KL the step records beside each
    teacher-reference decision must reproduce q2's recorded round-0 scores
    within the tolerance `admission` fixed from GPU evidence, and its argmin
    must be q2's first removed layer. An absent, partial or non-finite
    diagnostic is a failure, never a pass. Returns the verdict and the
    per-candidate comparison; it raises only when `admission` and `control`
    disagree with each other, which is a harness defect.
    """
    import math

    expected = admission["expected"]
    tol = admission["tolerance"]
    atol, rtol = float(tol["atol"]), float(tol["rtol"])
    q2 = {int(e["candidate"]): float(e["score"])
          for e in control["depth"]["rounds"][0]["table"]}
    first = int(control["depth"]["removal_order"][0])
    if (len(q2), first) != (int(expected["n_round0_candidates"]),
                            int(expected["first_removed_layer"])):
        raise D3DesignError("the admission record and the control record "
                            "describe different round-0 tables")
    selection = depth_step.get("selection") or {}
    rounds = selection.get("search_rounds") or []
    cands = selection.get("reference_candidates") or {}
    failures: list[str] = []
    r0 = rounds[0] if rounds else None
    if r0 is None or r0.get("round") != 0 or r0.get("removed_before"):
        failures.append("the step carries no round-0 table")
        tested: list[int] = []
    else:
        tested = [int(e["candidate"]) for e in r0.get("table") or []]
    if r0 is not None and len(tested) != len(set(tested)):
        failures.append("round 0 tests a candidate twice")
    if r0 is not None and set(tested) != set(q2):
        failures.append(
            f"round 0 tests {sorted(set(tested) ^ set(q2))} unlike q2")
    rows, missing = [], []
    for c in sorted(q2):
        rec = cands.get(str(c)) or {}
        value = rec.get("parent_reference_kl")
        if rec.get("round") != 0 or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            missing.append(c)
            continue
        bound = atol + rtol * abs(q2[c])
        diff = abs(float(value) - q2[c])
        rows.append({"candidate": c, "q2": q2[c], "d3": float(value),
                     "abs_difference": diff, "tolerance": bound,
                     "within": diff <= bound})
    if missing:
        failures.append(f"no finite round-0 parent-reference KL for "
                        f"{len(missing)} of {len(q2)} candidates: {missing}")
    outside = [r["candidate"] for r in rows if not r["within"]]
    if outside:
        failures.append(f"{len(outside)} candidates differ from q2 beyond "
                        f"tolerance: {outside}")
    argmin = (min(rows, key=lambda r: (r["d3"], r["candidate"]))["candidate"]
              if rows and not missing else None)
    if argmin != first:
        failures.append(f"the parent-reference argmin is {argmin}, q2 "
                        f"removed {first} first")
    return {
        "admissible": not failures, "failures": failures,
        "admission_hash": admission["admission_hash"],
        "atol": atol, "rtol": rtol,
        "candidates_expected": len(q2), "candidates_compared": len(rows),
        "max_abs_difference": max((r["abs_difference"] for r in rows),
                                  default=None),
        "max_fraction_of_tolerance": max(
            (r["abs_difference"] / r["tolerance"] for r in rows), default=None),
        "n_bit_identical": sum(r["abs_difference"] == 0.0 for r in rows),
        "parent_reference_argmin": argmin, "q2_first_removed": first,
        "candidates": rows}


__all__ = ["CONTROL_ARTIFACT_DIGEST", "CONTROL_STATE_ID", "D3DesignError",
           "EXPERIMENT_ID", "NO_STRUCTURAL_CHANGE", "REGISTRATION",
           "STRUCTURAL_CHANGE", "STRUCTURAL_OUTCOMES", "TREATMENT_PATH_ID",
           "UNEXPLAINED_BYTE_DIFFERENCE", "classify_structure", "control_leaf",
           "ADMISSION", "D3Inadmissible", "INADMISSIBLE", "admission_record",
           "parent_reference_admission", "depth_evidence",
           "control_spec", "execution", "operator_config_hashes",
           "registered_treatment_spec", "registration", "registration_hash",
           "root_identity", "treatment_contract", "treatment_spec"]
