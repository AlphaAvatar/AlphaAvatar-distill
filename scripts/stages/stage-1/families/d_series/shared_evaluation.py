#!/usr/bin/env python3
"""The D-series SHARED behavioural evaluation protocol, `d_series_shared_eval@v1`.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/families/d_series/shared_evaluation.py [--write]

**What it is.** A second, explicitly versioned way for D2 and D3 to measure
candidates: on the SAME prompts, seeds, recovery recipe and evaluation protocol
D1 already used, so the incumbent B and the working baseline q2 do not have to
be trained again. Their D1 measurements -- the generations, per-sample rows and
training identities of ten completed units -- are imported by content identity
and consumed as they are.

**What it is not.** It rewrites nothing. The six frozen family roles keep their
bytes, hashes and allocation rule; D1's preregistration, verdict and NO-GO are
cited, never edited; `d2_*` and `d3_*` stay reserved and unconsumed as the
fresh independent batteries a later promotion would need. The protocol has its
own identity (`protocol_hash`) and its own statistical rules, because
re-measuring against evidence that has already been observed is a weaker design
than the one the family was frozen for, and it is described as one.

**Where the weakening is, precisely** (the protocol record carries each):

* both shared rungs have been observed, so neither is a blind holdout. A
  multiplicity correction bounds the error rate of the comparisons it counts;
  it does nothing about choices adapted to what was seen;
* q2 advanced on the screening rung, so its screening rows carry the winner's
  curse and are descriptive only;
* the historical units were trained and generated in D1's sessions and a new
  candidate is trained in another. The family's single-session property does
  not hold across that boundary. The protocol pins every identity that can be
  pinned (image, GPU class, trainer and runtime digests, generation and
  evaluation protocol); the residual host and CUDA nondeterminism is the same
  kind of variation two probes of one session already show, and it is not
  separable from an arm effect;
* D2 and D3 are both tested against the same B rows, so their statistics are
  positively dependent and are controlled jointly, under a rule valid for any
  dependence.

So the shared confirmation can end an experiment (NO_GO, INCONCLUSIVE) and can
nominate a candidate for fresh confirmation (SHARED_GO). It cannot promote.

AUTHORIZES NOTHING.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[5]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _path = str(REPO_ROOT / _extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from shared.evaluation import evidence_reuse as reuse  # noqa: E402

SCHEMA = "aadistill.d_series.shared_evaluation_protocol/v1"
AUDIT_SCHEMA = "aadistill.d_series.shared_evidence_audit/v1"
PROTOCOL_ID = "d_series_shared_eval"
PROTOCOL_VERSION = 1

FAMILY_LOGS = "logs/stages/stage-1/families/d_series"
RECORD = f"{FAMILY_LOGS}/plans/d_series_shared_eval_v1.json"
AUDIT = f"{FAMILY_LOGS}/analyses/d1_shared_evidence_audit.json"
D1_RUNS = "logs/stages/stage-1/phase_d1/runs"
D1_PREREGISTRATION = ("logs/stages/stage-1/phase_d1/plans/"
                      "d1_behavioural_preregistration.json")

#: THE IMPORTED RUNGS. One complete D1 rung per shared rung, named by the run
#: that measured it. These are the protocol's inputs, so they are data here and
#: nowhere in reusable code. Each source run's closeout owns where its bytes
#: live and which file-hash manifest pins them.
RUNGS: dict[str, dict[str, str]] = {
    "shared_screening": {"battery_role": "d1_screening",
                         "source_run": "d1_behavioural_20261011_135910",
                         "d1_rung": "screening",
                         "scorer": "scripts/stages/stage-1/phase_d1/"
                                   "score_d1_screening.py"},
    "shared_confirmation": {"battery_role": "d1_confirmation",
                            "source_run": "d1_behavioural_20261012_064341",
                            "d1_rung": "confirmation",
                            "scorer": "scripts/stages/stage-1/phase_d1/"
                                      "score_d1_confirmation.py"},
}

#: The arms whose D1 measurements are imported, and what each is for. B is the
#: comparator every decision is made against; q2 is reported beside it.
REFERENCE_ARMS: dict[str, str] = {
    "B": "standing_incumbent_and_primary_comparator",
    "q2": "working_initialization_baseline_secondary_comparator",
}

#: The experiments that will test against the shared confirmation rows. The
#: family is fixed BEFORE either has a behavioural result, and its error rate
#: is split equally: Bonferroni is valid under the positive dependence that
#: sharing B's rows creates, and an equal split needs no knowledge of which
#: experiment finishes first.
FAMILY = ("phase_d2", "phase_d3")
FAMILY_ALPHA_ONE_SIDED = 0.05

#: The reserved fresh batteries. Untouched by this protocol.
RESERVED_ROLES = ("d2_screening", "d2_confirmation",
                  "d3_screening", "d3_confirmation")

#: The training-identity fields a reused unit must match. `initialization_digest`
#: is the model-weights coordinate; the rest describe the recovery that turned
#: an initialization into the evaluated checkpoint.
TRAINING_KEYS = ("initialization_digest", "recipe_digest", "data_digest",
                 "seed", "teacher", "trainer_source_digest",
                 "training_runtime_digest", "device_class")
EVALUATION_KEYS = ("evaluation_protocol_hash",)


class SharedEvaluationError(RuntimeError):
    """The shared protocol cannot be built or applied from the evidence."""


def _read(rel: str | Path, repo_root: Path = REPO_ROOT) -> Any:
    return json.loads((repo_root / rel).read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_dir(rung: str, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / D1_RUNS / RUNGS[rung]["source_run"]


def allowed_overrides(repo_root: Path = REPO_ROOT) -> tuple[str, ...]:
    """The probe-config fields allowed to differ, READ from D1's prereg."""
    prereg = _read(D1_PREREGISTRATION, repo_root)
    return tuple(prereg["recovery_recipe"]["allowed_override_set"])


def recipe_digest(config: Mapping[str, Any],
                  overrides: Sequence[str]) -> str:
    """sha256 of a probe config minus its allowed override set.

    Two probes with equal digests ran the same recovery recipe and differ only
    in run identity, paths, seed and starting checkpoint -- which is what the
    frozen recipe says may differ.
    """
    return sha256_json({k: v for k, v in config.items() if k not in overrides})


# --- the historical units, from the committed records ----------------------

@dataclass(frozen=True)
class HistoricalUnit:
    rung: str
    probe_id: str
    arm: str
    seed: int
    training: dict[str, Any]
    evaluation: dict[str, Any]
    trained_sha256: str
    trained_bytes: int
    result: dict[str, Any]
    timing: dict[str, float]
    provenance: dict[str, str]


def _session_training_identity(run_dir: Path, store: Path | None
                               ) -> dict[str, Any]:
    """Trainer and runtime identity of one session, from any manifest it kept.

    One session runs one checkout on one pod, so every probe it trained shares
    these. The screening session kept two of ten manifests (B's, in its final
    pod pull); the confirmation session kept all six in git.
    """
    candidates = sorted((run_dir / "evidence" / "training").glob(
        "*/run_manifest.json"))
    if store is not None:
        candidates += sorted(store.glob(
            "store/pod_final/stages/stage-3/*/*/run_manifest.json"))
    seen: dict[str, dict[str, Any]] = {}
    for path in candidates:
        m = json.loads(path.read_text())
        ex = m["execution"]
        seen[path.parent.name] = {
            "trainer_source_digest": ex["trainer_source"]["digest"],
            "training_runtime_digest": ex["runtime_digest"],
            "training_image_digest": ex["runtime"]["image_digest"],
            "device_class": ",".join(m["hardware"]["cuda_devices"]),
            "teacher": (f"{m['teacher']['model_id']}@"
                        f"{m['teacher']['revision']}"),
            "data_digest": m["ladder"]["blocks_sha256"],
        }
    if not seen:
        return {}
    values = {json.dumps(v, sort_keys=True) for v in seen.values()}
    if len(values) != 1:
        raise SharedEvaluationError(
            f"{run_dir.name}: the session's run manifests disagree on the "
            f"training identity: {sorted(values)}")
    out = dict(next(iter(seen.values())))
    out["_from_manifests"] = sorted(seen)
    return out


def durable_store(rung: str, repo_root: Path = REPO_ROOT,
                  evidence_root: Path | None = None) -> Path:
    """Where a source run's bytes live: its closeout, or a relocated root."""
    closeout = _read(_run_dir(rung, repo_root) / "closeout" / "outcome.json",
                     repo_root)
    recorded = Path(closeout["evidence"]["durable_store"]["path"])
    if evidence_root is not None:
        return evidence_root / recorded.name
    return recorded


def historical_units(repo_root: Path = REPO_ROOT,
                     evidence_root: Path | None = None
                     ) -> list[HistoricalUnit]:
    """The ten imported units, every field read from its owning record."""
    overrides = allowed_overrides(repo_root)
    units: list[HistoricalUnit] = []
    for rung, spec in RUNGS.items():
        run_dir = _run_dir(rung, repo_root)
        relay = json.loads((run_dir / "evidence" /
                            "d1_behavioural.json").read_text())
        attested = json.loads((run_dir / "evidence" /
                               "d1_behavioural_attested_protocol.json"
                               ).read_text())
        store = durable_store(rung, repo_root, evidence_root)
        session = _session_training_identity(
            run_dir, store if store.is_dir() else None)
        if not session:
            raise SharedEvaluationError(
                f"{rung}: no run manifest of {spec['source_run']} is "
                f"reachable. The screening session kept its manifests only in "
                f"its durable store ({store}); pass --evidence-root if it was "
                "relocated. The committed protocol record is the owner "
                "everywhere the store is absent")
        pack = relay["contract"]["recovery_recipe"]["pack_sha256"]
        if session.get("data_digest") not in (None, pack):
            raise SharedEvaluationError(
                f"{rung}: the trained pack {session['data_digest']} is not "
                f"the contract's {pack}")
        by_id = {p["probe_id"]: p for p in relay["probes"]}
        for probe_id, probe in sorted(by_id.items()):
            arm = probe["arm"]
            if arm not in REFERENCE_ARMS:
                continue
            result = json.loads((run_dir / "evidence" / "results" /
                                 f"{probe_id}.json").read_text())
            config = json.loads((run_dir / "evidence" / "configs" /
                                 f"{probe_id}.json").read_text())
            teacher = config["teacher"]
            training = {
                "initialization_digest":
                    result["initialization_artifact_digest"],
                "recipe_digest": recipe_digest(config, overrides),
                "data_digest": pack,
                "seed": int(probe["seed"]),
                "teacher": f"{teacher['model_id']}@{teacher['revision']}",
                "trainer_source_digest": session.get("trainer_source_digest"),
                "training_runtime_digest":
                    session.get("training_runtime_digest"),
                "device_class": session.get("device_class"),
            }
            evaluation = {
                "evaluation_protocol_hash":
                    probe["generation_admission"]["evaluation_protocol_hash"],
            }
            if (evaluation["evaluation_protocol_hash"]
                    != attested["evaluation_protocol_hash"]):
                raise SharedEvaluationError(
                    f"{probe_id}: admitted under a protocol the session did "
                    "not attest")
            units.append(HistoricalUnit(
                rung=rung, probe_id=probe_id, arm=arm,
                seed=int(probe["seed"]), training=training,
                evaluation=evaluation,
                trained_sha256=probe["trained_sha256"],
                trained_bytes=int(probe["trained_bytes"]),
                result=result,
                timing={"train_seconds": float(probe["train_seconds"]),
                        "generate_seconds": float(probe["generate_seconds"])},
                provenance={
                    "trained_in_run": spec["source_run"],
                    "training_identity_from": (
                        "this probe's run manifest"
                        if probe_id in session.get("_from_manifests", ())
                        else "the session's other run manifests (one "
                             "checkout, one pod)"),
                    "training_image_digest":
                        session.get("training_image_digest") or "",
                }))
    return units


# --- the protocol record ---------------------------------------------------

def _rung_record(rung: str, units: Sequence[HistoricalUnit],
                 repo_root: Path) -> dict[str, Any]:
    from stages.phase_d1 import behavioural as D1B

    spec = RUNGS[rung]
    run_dir = _run_dir(rung, repo_root)
    attested = json.loads((run_dir / "evidence" /
                           "d1_behavioural_attested_protocol.json").read_text())
    battery = D1B.battery_role(spec["battery_role"], repo_root)
    seeds = sorted({u.seed for u in units if u.rung == rung})
    declared = (D1B.screening_seeds(repo_root) if spec["d1_rung"] == "screening"
                else D1B.confirmation_seeds(repo_root))
    if seeds != sorted(declared):
        raise SharedEvaluationError(
            f"{rung}: imported seeds {seeds} are not D1's preregistered "
            f"{sorted(declared)}")
    for arm in REFERENCE_ARMS:
        have = sorted(u.seed for u in units if u.rung == rung and u.arm == arm)
        if have != seeds:
            raise SharedEvaluationError(
                f"{rung}: arm {arm} has seeds {have}, not {seeds}; a partial "
                "reference arm cannot be paired")
    scorer = repo_root / spec["scorer"]
    return {
        "reuses_battery": {k: battery[k] for k in (
            "role", "family_content_id", "item_ids_sha256", "n_prompts",
            "n_scorable")},
        "seeds": seeds,
        "seeds_owner": f"{D1_PREREGISTRATION} :: seeds.{spec['d1_rung']}",
        "evaluation_protocol_hash": attested["evaluation_protocol_hash"],
        "generation_protocol_fingerprint":
            attested["generation_protocol_fingerprint"],
        "scoring_digest": attested["scoring_contract"]["digest"],
        "scoring_source_files": {f["path"]: f["sha256"] for f in
                                 attested["scoring_contract"]["files"]},
        "generation_source_digest":
            attested["evaluation_protocol"]["generation"][
                "generation_source_digest"],
        "scorer": {"path": spec["scorer"], "sha256": _sha256(scorer)},
        "generation_runtime_digest":
            attested["evaluation_protocol"]["generation"]["runtime_digest"],
        "source_run": spec["source_run"],
        "reference_units": [{
            "probe_id": u.probe_id, "arm": u.arm, "seed": u.seed,
            "initialization_digest": u.training["initialization_digest"],
            "trained_sha256": u.trained_sha256,
            "result_sha256": u.result["result_sha256"],
            "per_sample_sha256": u.result["per_sample_sha256"],
            "generations_sha256": {s: g["sha256"] for s, g in
                                   sorted(u.result["generations"].items())},
            "correct_overall": u.result["correct_overall"],
            "usable_rollout_rate": u.result["usable_rollout_rate"],
        } for u in units if u.rung == rung],
    }


def _one(values: set, what: str) -> Any:
    if len(values) != 1:
        raise SharedEvaluationError(
            f"the imported units disagree on {what}: {sorted(map(str, values))}")
    return next(iter(values))


def build_protocol(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The protocol, derived from its owners. Deterministic.

    Everything comes from committed records except the screening session's
    trainer and runtime identity, which that session kept only in the run
    manifests of its durable store. Where the store is absent the committed
    record is the owner, and :func:`verify_record` checks it.
    """
    from stages.phase_c1.isolation import BOOTSTRAP_ITERATIONS
    from stages.phase_d1 import behavioural as D1B

    units = historical_units(repo_root)
    bd = D1B.behavioural_design(repo_root)
    prereg = _read(D1_PREREGISTRATION, repo_root)
    d1_verdict = _read(_run_dir("shared_confirmation", repo_root) /
                       "evidence" / "confirmation_verdict.json", repo_root)
    family = _read(f"{FAMILY_LOGS}/current.json", repo_root)

    recovery = {
        "recipe_id": prereg["recovery_recipe"]["recipe_id"],
        "allowed_override_set": list(allowed_overrides(repo_root)),
    }
    for key in ("recipe_digest", "data_digest", "teacher",
                "trainer_source_digest", "training_runtime_digest",
                "device_class"):
        recovery[key] = _one({u.training[key] for u in units}, key)
    recovery["training_image_digest"] = _one(
        {u.provenance["training_image_digest"] for u in units},
        "training image")
    rungs = {r: _rung_record(r, units, repo_root) for r in RUNGS}
    _one({r["generation_protocol_fingerprint"] for r in rungs.values()},
         "generation protocol fingerprint")

    per_experiment_alpha = FAMILY_ALPHA_ONE_SIDED / len(FAMILY)
    science = {
        "protocol_id": PROTOCOL_ID,
        "version": PROTOCOL_VERSION,
        "family_id": family["family_id"],
        "allocation_rule_id": family["allocation_rule_id"],
        "family_content_id": family["family_content_id"],
        "rungs": rungs,
        "reference_arms": {arm: {
            "role": role,
            "initialization_digest": _one(
                {u.training["initialization_digest"] for u in units
                 if u.arm == arm}, f"{arm} initialization"),
        } for arm, role in REFERENCE_ARMS.items()},
        "recovery": recovery,
        "unit_identity": {
            "training_keys": list(TRAINING_KEYS),
            "evaluation_keys": list(EVALUATION_KEYS),
            "classifier": "scripts/shared/evaluation/evidence_reuse.py",
            "_seed_is_one_field": (
                "a matching seed is not a matching unit; the classifier "
                "compares every training field, and a result belongs to the "
                "trained bytes it measured"),
        },
        "new_candidate_requirements": [
            "trained under the recovery block above with every training key "
            "equal except initialization_digest",
            "generated and scored under the rung's evaluation_protocol_hash: "
            "the same generation fingerprint, the same scorer bytes, the same "
            "battery",
            "trained and generated on the pinned image and device class",
            "per-sample rows secured off-pod; the verdict is computed off-pod "
            "from rows, never on the pod",
        ],
        "statistics": {
            "endpoint": bd["decision_rule"]["primary_endpoint"],
            "estimand": bd["decision_rule"]["estimand"],
            "sesoi": float(bd["sesoi"]),
            "inference": ("stratified prompt-cluster bootstrap, seeds as "
                          "fixed blocks (stages.phase_c1.isolation)"),
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed_rule": (
                "SHA256(protocol_hash + ':' + experiment + ':' + rung + "
                "'-bootstrap')[0:4] as uint32 mod 2**31, one per experiment "
                "and rung, never another phase's default"),
            "guardrails": {
                "usable_pooled_min_delta": D1B.GUARDRAIL_POOLED_MIN_DELTA,
                "usable_per_seed_min_delta": D1B.GUARDRAIL_PER_SEED_MIN_DELTA,
                "role": "veto only; never positive credit",
            },
            "seed_robustness_min_positive": 2,
            "comparator": "B",
            "family": {
                "members": list(FAMILY),
                "family_alpha_one_sided": FAMILY_ALPHA_ONE_SIDED,
                "method": "Bonferroni, equal fixed split",
                "per_experiment_alpha_one_sided": per_experiment_alpha,
                "why_this_method": (
                    "D2 and D3 are both compared with the same B rows, so "
                    "their statistics are positively dependent; Bonferroni "
                    "holds under any dependence, and a fixed split needs no "
                    "knowledge of which experiment reports first"),
                "d1_already_spent": {
                    "test": "q2 vs B on these rows, one-sided alpha 0.05",
                    "verdict": d1_verdict["decision"]["verdict"],
                    "union_bound_over_d1_d2_d3": round(
                        0.05 + FAMILY_ALPHA_ONE_SIDED, 4),
                    "_meaning": (
                        "the probability that ANY of the three experiments "
                        "falsely clears this battery is at most this bound. "
                        "It is one reason a shared-confirmation pass "
                        "nominates rather than promotes"),
                },
                "_does_not_correct": (
                    "adaptive reuse. A multiplicity correction bounds the "
                    "error rate of the comparisons it counts; it does not "
                    "remove bias from choices made after seeing these "
                    "prompts' outcomes. That is handled by what is frozen "
                    "before any D2/D3 behavioural result, not by alpha"),
            },
            "screening": {
                "comparator": "B's imported screening rows",
                "statistic": ("pooled seed-mean correct_overall delta vs B, "
                              "C0's guardrail vetoes"),
                "selection": "exactly one candidate advances, or none",
                "tie_break": ("the experiment's frozen quality position, then "
                              "state id"),
                "field": ("frozen by the experiment's own retention decision "
                          "before any behavioural datum; its size is that "
                          "experiment's preregistered choice"),
                "q2_screening_rows": (
                    "DESCRIPTIVE ONLY. q2 advanced on these prompts and "
                    "seeds, so its screening estimate is inflated by the "
                    "winner's curse (screening +0.00825 vs B, confirmation "
                    "-0.000392). Never a ranking comparator"),
            },
            "confirmation_outcomes": {
                "SHARED_GO": (
                    "one-sided (1 - per_experiment_alpha) LCB > 0 AND delta "
                    ">= SESOI AND >= 2 of 3 seed deltas > 0 AND no guardrail "
                    "veto. NOT a promotion: it nominates the candidate for a "
                    "fresh independent confirmation on its reserved battery"),
                "NO_GO": ("one-sided (1 - per_experiment_alpha) UCB < SESOI, "
                          "or a guardrail veto. Terminal for that candidate"),
                "INCONCLUSIVE": "otherwise. Terminal; never rerun on these rows",
            },
            "secondary_comparison": (
                "the candidate vs q2 on the shared confirmation rows, same "
                "bootstrap, reported with its interval. No decision role, so "
                "it consumes no alpha"),
        },
        "claim_boundary": {
            "may_claim": [
                "a paired comparison with B (and q2) on identical prompts, "
                "seeds, recovery recipe, generation and scoring protocol",
                "an experiment-terminal NO_GO or INCONCLUSIVE",
                "a nomination for fresh confirmation (SHARED_GO)",
            ],
            "may_not_claim": [
                "a promotion, or evidence equivalent to a fresh independent "
                "confirmation",
                "a seed-population or prompt-population generalization "
                "beyond these fixed blocks",
                "that the cross-session component is zero: it is pinned, not "
                "measured",
            ],
            "promotion_requires": (
                "a fresh confirmation on the experiment's reserved battery "
                "(d2_confirmation or d3_confirmation) with a common control "
                "measured there. How to obtain that control most cheaply is "
                "decided when a SHARED_GO exists, not before"),
        },
        "session_boundary": {
            "property_lost": (
                "D1's single-session field: every probe of a rung trained and "
                "evaluated in one session on one host"),
            "pinned_instead": [
                "training image digest", "device class",
                "trainer source digest", "training runtime digest",
                "generation protocol fingerprint (engine, sampling, context, "
                "template, tokenizer)", "generation runtime digest",
                "evaluation protocol hash (battery + scorer bytes)"],
            "residual": (
                "host, driver and CUDA reduction-order nondeterminism. "
                "Training is not bit-reproducible even within one session "
                "(logs/stages/stage-1/phase_d1/analyses/"
                "d1_probe_training_is_not_bit_reproducible.json), so this is "
                "the same kind of variation the seed blocks already carry, "
                "but it is not separable from an arm effect and is recorded "
                "per unit"),
        },
        "reserved_independent_batteries": {
            role: family["roles"][role] for role in RESERVED_ROLES},
        "d1_records_unchanged": {
            "preregistration_sha256": prereg["preregistration_sha256"],
            "confirmation_verdict_sha256": d1_verdict["verdict_sha256"],
            "verdict": d1_verdict["decision"]["verdict"],
        },
    }
    protocol_hash = sha256_json(science)
    return {
        "schema": SCHEMA,
        "_contract": (
            "The D-series shared behavioural evaluation protocol. A NEW "
            "identity beside the frozen family, not an edit of it. Derived "
            "from committed records by "
            "scripts/stages/stage-1/families/d_series/shared_evaluation.py. "
            "AUTHORIZES NOTHING: a paid D2/D3 run still needs its own "
            "preregistration and authorization, both bound to protocol_hash."),
        "protocol_hash": protocol_hash,
        "decided_by": ("maintainer decision 2026-10-13, "
                       "logs/budget/decisions.md"),
        **science,
        "errata": [dict(e) for e in ERRATA],
        "authorizes": "nothing",
    }


#: Record keys outside the hashed science: presentation, provenance and errata.
_ENVELOPE = ("schema", "_contract", "protocol_hash", "decided_by", "authorizes",
             "errata")

#: Corrections to how a hashed field READS, which change no rule, number or
#: identity -- so they sit outside the hash and the protocol stays v1.
ERRATA = (
    {"date": "2026-10-13", "source": "independent review",
     "field": "statistics.family.d1_already_spent",
     "correction": (
         "`union_bound_over_d1_d2_d3` = 0.10 is a NOMINAL accounting bound: "
         "the sum of the three tests' nominal one-sided levels. It holds only "
         "if each test attains its nominal level, which adaptive reuse of the "
         "already-observed d1_confirmation rows can violate. It is not an "
         "unconditional guarantee, and its `_meaning` sentence (\"the "
         "probability that ANY of the three experiments falsely clears this "
         "battery is at most this bound\") must be read with that condition. "
         "No rule, alpha, outcome or identity changes.")},
)


def verify_record(record: Mapping[str, Any]) -> bool:
    """A committed protocol record still hashes to its own protocol_hash."""
    science = {k: v for k, v in record.items() if k not in _ENVELOPE}
    return sha256_json(science) == record["protocol_hash"]


def tree_matches(record: Mapping[str, Any], repo_root: Path = REPO_ROOT
                 ) -> dict[str, list[str]]:
    """What in THIS tree differs from the implementation the protocol pins.

    The $0 half of comparability, checkable before any pod exists: a candidate
    trained by a different trainer or scored by different scorer bytes is not
    paired with the imported units, whatever the run is called. The on-pod
    half -- runtime, engine, device -- is the attested generation fingerprint
    and the run manifest, compared against the same record on the pod.
    """
    from shared import source_sets

    out: dict[str, list[str]] = {"trainer": [], "generation": [], "scoring": []}
    if (source_sets.trainer_source_digest(str(repo_root))["digest"]
            != record["recovery"]["trainer_source_digest"]):
        out["trainer"].append("trainer_source_digest")
    for rung, spec in record["rungs"].items():
        if (source_sets.generation_source_digest(str(repo_root))["digest"]
                != spec["generation_source_digest"]):
            out["generation"].append(rung)
        files = dict(spec["scoring_source_files"])
        files[spec["scorer"]["path"]] = spec["scorer"]["sha256"]
        for path, digest in sorted(files.items()):
            target = repo_root / path
            if not target.is_file() or _sha256(target) != digest:
                out["scoring"].append(f"{rung}:{path}")
    return out


def bootstrap_seed(protocol_hash: str, experiment: str, rung: str) -> int:
    digest = hashlib.sha256(
        f"{protocol_hash}:{experiment}:{rung}-bootstrap".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


# --- applying the protocol: ranking and verdict over imported rows ---------

def load_reference_rows(rung: str, arm: str, *, repo_root: Path = REPO_ROOT,
                        evidence_root: Path | None = None
                        ) -> dict[int, list[dict]]:
    """An imported arm's per-sample rows, hash-checked against its result."""
    store = durable_store(rung, repo_root, evidence_root)
    out: dict[int, list[dict]] = {}
    for unit in historical_units(repo_root, evidence_root):
        if unit.rung != rung or unit.arm != arm:
            continue
        path = store / "store" / "probes" / unit.probe_id / "per_sample.jsonl"
        data = path.read_bytes()
        got = hashlib.sha256(data).hexdigest()
        if got != unit.result["per_sample_sha256"]:
            raise SharedEvaluationError(
                f"{path} hashes to {got}, not the recorded "
                f"{unit.result['per_sample_sha256']}")
        out[unit.seed] = [json.loads(line) for line in data.decode().splitlines()
                          if line.strip()]
    if not out:
        raise SharedEvaluationError(f"no imported {arm} rows on {rung}")
    return out


@dataclass(frozen=True)
class SharedConfirmationPlan:
    """The five attributes `isolation.decide` reads, under this protocol."""

    plan_id: str
    seeds: tuple[int, ...]
    sesoi: float
    seed_robustness_min_positive: int
    usable_pooled_min_delta: float
    usable_per_seed_min_delta: float


def confirmation_verdict(candidate: str,
                         candidate_rows: Mapping[int, Sequence[Mapping]],
                         comparator_rows: Mapping[int, Sequence[Mapping]], *,
                         protocol: Mapping[str, Any], alpha: float,
                         bootstrap_seed_value: int,
                         comparator: str = "B") -> dict[str, Any]:
    """C0's three-way rule over candidate rows paired with imported rows.

    Every rule is imported from the modules that own it; this only aligns the
    two sources of rows. `alpha` is the protocol's per-experiment allocation.
    """
    from stages.phase_c1.isolation import (
        BOOTSTRAP_ITERATIONS, decide, paired_differences,
        stratified_cluster_bootstrap)
    from stages.phase_c1.probe_results import decision_inputs

    stats = protocol["statistics"]
    seeds = tuple(protocol["rungs"]["shared_confirmation"]["seeds"])
    if sorted(candidate_rows) != sorted(seeds) or \
            sorted(comparator_rows) != sorted(seeds):
        raise SharedEvaluationError(
            f"both arms need exactly the seeds {list(seeds)}")
    per_sample = {}
    for seed in seeds:
        per_sample[(comparator, seed)] = list(comparator_rows[seed])
        per_sample[(candidate, seed)] = list(candidate_rows[seed])
    inputs = decision_inputs(per_sample, seeds=seeds,
                             arms=(comparator, candidate),
                             control_operand=comparator,
                             candidate_operand=candidate)
    d = paired_differences(inputs.arm(comparator), inputs.arm(candidate))
    boot = stratified_cluster_bootstrap(
        d, inputs.strata, seed=bootstrap_seed_value,
        iterations=int(stats["bootstrap_iterations"]), alpha=alpha)
    per_seed_delta = [
        sum(bool(inputs.correct[candidate][s][j])
            - bool(inputs.correct[comparator][s][j]) for j in d) / len(d)
        for s in seeds]
    plan = SharedConfirmationPlan(
        plan_id=f"{PROTOCOL_ID}@v{PROTOCOL_VERSION}.shared_confirmation",
        seeds=seeds, sesoi=float(stats["sesoi"]),
        seed_robustness_min_positive=int(
            stats["seed_robustness_min_positive"]),
        usable_pooled_min_delta=float(
            stats["guardrails"]["usable_pooled_min_delta"]),
        usable_per_seed_min_delta=float(
            stats["guardrails"]["usable_per_seed_min_delta"]))
    decision = decide(
        plan, boot=boot, per_seed_delta=per_seed_delta,
        usable_pooled_delta=inputs.usable_pooled_delta,
        usable_per_seed_delta=list(inputs.usable_per_seed_delta),
        catastrophic_violations=inputs.catastrophic_violations)
    return {"decision": decision, "per_seed_delta": per_seed_delta,
            "alpha_one_sided": alpha, "bootstrap_seed": bootstrap_seed_value,
            "iterations": int(stats["bootstrap_iterations"]),
            "n_prompts_paired": len(d),
            "_boot_iterations_constant": BOOTSTRAP_ITERATIONS}


def shared_outcome(decision: Mapping[str, Any]) -> str:
    """C0's GO becomes SHARED_GO here: a nomination, never a promotion."""
    return {"GO": "SHARED_GO", "NO-GO": "NO_GO",
            "INCONCLUSIVE": "INCONCLUSIVE"}[decision["verdict"]]


@dataclass(frozen=True)
class FieldArm:
    arm: str
    quality_position: int
    state_id: str


def rank_screening(field: Sequence[FieldArm],
                   scored: Sequence[Mapping[str, Any]],
                   comparator: Sequence[Mapping[str, Any]], *,
                   seeds: Sequence[int]) -> dict[str, Any]:
    """D1's screening rule over a NEW field against imported comparator rows.

    `scored` is one row per candidate probe and `comparator` one per imported
    B probe, each carrying `arm`, `seed`, `correct_overall` and
    `usable_rollout_rate`. Pooling, vetoes and the selection are D1's own
    functions; only the field and the comparator's source change.
    """
    from stages.phase_d1 import behavioural as D1B

    want = sorted(int(s) for s in seeds)
    by_arm: dict[str, list[Mapping[str, Any]]] = {}
    for row in scored:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    for arm in field:
        got = sorted(int(r["seed"]) for r in by_arm.get(arm.arm, ()))
        if got != want:
            raise SharedEvaluationError(
                f"arm {arm.arm} has seeds {got}, not {want}; ranking a partial "
                "field selects on who finished")
    if sorted(int(r["seed"]) for r in comparator) != want:
        raise SharedEvaluationError("the comparator does not cover the seeds")
    b = D1B.pool_arm(comparator)
    rows = []
    for arm in field:
        mine = D1B.pool_arm(by_arm[arm.arm])
        delta_usable = mine["usable_rollout_rate"] - b["usable_rollout_rate"]
        per_seed_usable = {
            s: mine["per_seed"][s]["usable_rollout_rate"]
               - b["per_seed"][s]["usable_rollout_rate"] for s in want}
        vetoes = []
        if delta_usable <= D1B.GUARDRAIL_POOLED_MIN_DELTA:
            vetoes.append(f"pooled usable_rollout delta {delta_usable:+.4f} "
                          f"is not > {D1B.GUARDRAIL_POOLED_MIN_DELTA}")
        for s, v in per_seed_usable.items():
            if v <= D1B.GUARDRAIL_PER_SEED_MIN_DELTA:
                vetoes.append(f"seed {s} usable_rollout delta {v:+.4f} is "
                              f"not > {D1B.GUARDRAIL_PER_SEED_MIN_DELTA}")
        rows.append({
            "arm": arm.arm, "state_id": arm.state_id,
            "quality_position": arm.quality_position,
            "correct_overall": mine["correct_overall"],
            "incumbent_correct_overall": b["correct_overall"],
            "delta_vs_b": round(mine["correct_overall"]
                                - b["correct_overall"], 10),
            "usable_rollout_rate": mine["usable_rollout_rate"],
            "incumbent_usable_rollout_rate": b["usable_rollout_rate"],
            "delta_usable_pooled": round(delta_usable, 10),
            "delta_usable_per_seed": {str(k): round(v, 10)
                                      for k, v in per_seed_usable.items()},
            "vetoed": bool(vetoes), "vetoes": vetoes,
            "per_seed": mine["per_seed"],
        })
    rows.sort(key=lambda r: (r["vetoed"], -r["delta_vs_b"],
                             r["quality_position"], r["state_id"]))
    for position, row in enumerate(rows):
        row["screening_position"] = position
    ranking = {"seeds": want, "incumbent": {"arm": "B", **b}, "ranked": rows}
    return {"ranking": ranking, "selection": D1B.advance_one(ranking)}


# --- the audit -------------------------------------------------------------

def _verify_unit(unit: HistoricalUnit, store: Path, repo_root: Path, *,
                 rescore: bool) -> dict[str, Any]:
    probe_dir = store / "store" / "probes" / unit.probe_id
    committed = (_run_dir(unit.rung, repo_root) / "evidence" / "results" /
                 f"{unit.probe_id}.json")
    checks: dict[str, Any] = {}
    body = {k: v for k, v in unit.result.items() if k != "result_sha256"}
    checks["result_self_hash"] = sha256_json(body) == unit.result["result_sha256"]
    durable_result = probe_dir / "result.json"
    checks["durable_result_equals_committed"] = (
        durable_result.is_file()
        and durable_result.read_bytes() == committed.read_bytes())
    rows_path = probe_dir / "per_sample.jsonl"
    checks["per_sample_present"] = rows_path.is_file()
    if rows_path.is_file():
        data = rows_path.read_bytes()
        checks["per_sample_sha256"] = (hashlib.sha256(data).hexdigest()
                                       == unit.result["per_sample_sha256"])
        rows = [json.loads(x) for x in data.decode().splitlines() if x.strip()]
        checks["per_sample_rows"] = len(rows) == int(unit.result["per_sample_rows"])
        checks["per_sample_pairable"] = (
            len({r["id"] for r in rows}) == len(rows)
            and all({"id", "set", "seed", "correct", "usable", "scorable"}
                    <= set(r) for r in rows))
    gens = {}
    for name, g in sorted(unit.result["generations"].items()):
        path = probe_dir / "generations" / Path(g["path"]).name
        gens[name] = path.is_file() and _sha256(path) == g["sha256"]
    checks["generations_sha256"] = all(gens.values()) and len(gens) == 7
    checks["checkpoint_bytes_present"] = any(
        p.suffix == ".safetensors" for p in probe_dir.rglob("*"))
    if rescore and rows_path.is_file():
        scorer = repo_root / RUNGS[unit.rung]["scorer"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cmd = [sys.executable, str(scorer),
                   "--generations", str(probe_dir / "generations"),
                   "--label", unit.probe_id, "--seed", str(unit.seed),
                   "--out", str(out / "result.json"),
                   "--per-sample", str(out / "rows.jsonl")]
            if RUNGS[unit.rung]["d1_rung"] == "screening":
                cmd += ["--screening-arm", unit.arm]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=repo_root,
                                  env={"PYTHONPATH": "src:scripts",
                                       "PATH": "/usr/bin:/bin"})
            checks["rescore_reproduces_per_sample"] = (
                proc.returncode == 0 and (out / "rows.jsonl").is_file()
                and _sha256(out / "rows.jsonl")
                == unit.result["per_sample_sha256"])
            if proc.returncode != 0:
                checks["rescore_error"] = proc.stderr.strip().splitlines()[-1:]
    verified = all(v for k, v in checks.items()
                   if k not in ("checkpoint_bytes_present", "rescore_error"))
    return {"checks": checks, "result_verified": verified}


def _pooled_seed_sd(repo_root: Path) -> dict[str, Any]:
    """The per-seed spread of D1's paired deltas, pooled within arm.

    The quantity `selection_noise` prices screening breadth with. A3's three
    deltas were its only input before D1; D1 adds eight screening and three
    confirmation deltas, measured on this very battery family.
    """
    ranking = _read(_run_dir("shared_screening", repo_root) / "evidence" /
                    "d1_screening_ranking_offpod.json", repo_root)["ranking"]
    groups = []
    for row in ranking["ranked"]:
        b = ranking["incumbent"]["per_seed"]
        groups.append([row["per_seed"][s]["correct_overall"]
                       - b[s]["correct_overall"] for s in row["per_seed"]])
    verdict = _read(_run_dir("shared_confirmation", repo_root) / "evidence" /
                    "confirmation_verdict.json", repo_root)
    groups.append(list(verdict["per_seed_delta"]))
    ss = sum(sum((x - statistics.fmean(g)) ** 2 for x in g) for g in groups)
    dof = sum(len(g) - 1 for g in groups)
    return {"seed_sd": round(math.sqrt(ss / dof), 6), "dof": dof,
            "groups": [[round(x, 6) for x in g] for g in groups],
            "_from": ("D1's four screening arms (2 seeds each) and its "
                      "confirmation arm (3 seeds), deltas vs B, pooled within "
                      "arm")}


def probe_economy(units: Sequence[HistoricalUnit],
                  repo_root: Path) -> dict[str, Any]:
    """Probes and GPU dollars per D-series experiment, measured not assumed."""
    from stages.phase_d1 import selection_noise as SN

    sessions = {}
    materialize: dict[str, dict[str, float]] = {}
    for rung, spec in RUNGS.items():
        s = _read(_run_dir(rung, repo_root) / "runtime" / "session.json",
                  repo_root)
        sessions[rung] = float(s["actual_price_per_hour"])
        relay = _read(_run_dir(rung, repo_root) / "evidence" /
                      "d1_behavioural.json", repo_root)
        materialize[rung] = {a["arm"]: round(float(a["seconds"]) / 60, 2)
                             for a in relay["arm_materialization"]["arms"]
                             if a["arm"] in REFERENCE_ARMS}
    rate = _one(set(sessions.values()), "billed rate")
    train = [u.timing["train_seconds"] for u in units]
    gen = [u.timing["generate_seconds"] for u in units]
    per_probe_mean = (statistics.fmean(train) + statistics.fmean(gen)) / 60
    per_probe_max = (max(train) + max(gen)) / 60
    spread = _pooled_seed_sd(repo_root)
    table = []
    for k in (1, 2, 3, 4):
        d1_style = (k + 1) * 2 + 2 * 3
        shared = k * 2 + 1 * 3
        table.append({
            "top_k": k,
            "probes_d1_style": d1_style,
            "probes_shared": shared,
            "probes_saved": d1_style - shared,
            "probes_with_q2_retrained_as_second_control": (k + 2) * 2 + 3 * 3,
            "gpu_usd_shared_expected": round(shared * per_probe_mean / 60 * rate, 2),
            "gpu_usd_shared_ceiling": round(shared * per_probe_max / 60 * rate, 2),
            "gpu_usd_saved_vs_d1_style": round(
                (d1_style - shared) * per_probe_mean / 60 * rate, 2),
            "screening_inflation_at_null": SN.screening_estimate_inflation(
                k, 2, seed_sd=spread["seed_sd"]),
            "p_advance_true_sesoi_candidate": SN.advance_probability(
                k, 2, seed_sd=spread["seed_sd"]),
        })
    return {
        "rate_usd_per_hour": rate,
        "rate_owner": "the two source sessions' runtime/session.json",
        "per_probe_minutes": {
            "train_mean": round(statistics.fmean(train) / 60, 2),
            "train_max": round(max(train) / 60, 2),
            "generate_mean": round(statistics.fmean(gen) / 60, 2),
            "generate_max": round(max(gen) / 60, 2),
            "mean_total": round(per_probe_mean, 2),
            "max_total": round(per_probe_max, 2),
            "from": f"{len(units)} imported B/q2 units' relay timings",
        },
        "seed_spread": spread,
        "per_experiment": table,
        "reference_materialization_minutes_per_session": materialize,
        "_excludes": (
            "session setup, candidate materialization on the pod, the engine "
            "probe and teardown: per-session overheads that both designs pay. "
            "The shared design ALSO skips materializing the reference arms on "
            "the pod in every session, which D1 paid "
            "(reference_materialization_minutes_per_session, from the relay)"),
        "_d1_style_means": (
            "K candidates + B on 2 screening seeds, then the advanced "
            "candidate + B on 3 confirmation seeds -- D1's own field shape"),
    }


def audit(repo_root: Path = REPO_ROOT, evidence_root: Path | None = None, *,
          rescore: bool = True, protocol: Mapping[str, Any] | None = None
          ) -> dict[str, Any]:
    """Verify every imported unit's bytes and re-derive D1's decisions."""
    from stages.phase_d1 import behavioural as D1B
    from stages.phase_d1 import behavioural_verdict as D1V

    protocol = protocol or build_protocol(repo_root)
    units = historical_units(repo_root, evidence_root)
    per_unit = []
    evidence: list[reuse.UnitEvidence] = []
    for unit in units:
        store = durable_store(unit.rung, repo_root, evidence_root)
        v = _verify_unit(unit, store, repo_root, rescore=rescore)
        per_unit.append({
            "probe_id": unit.probe_id, "rung": unit.rung, "arm": unit.arm,
            "seed": unit.seed, "training": unit.training,
            "evaluation": unit.evaluation,
            "trained_sha256": unit.trained_sha256,
            "provenance": unit.provenance, **v})
        evidence.append(reuse.UnitEvidence(
            unit_id=unit.probe_id, training=unit.training,
            evaluation=unit.evaluation, trained_sha256=unit.trained_sha256,
            result_verified=v["result_verified"],
            checkpoint_verified=v["checks"]["checkpoint_bytes_present"]))

    def request(arm: str, rung: str, at_seed: int, **change: Any
                ) -> tuple[dict, dict]:
        base = next(u for u in units if u.arm == arm and u.rung == rung
                    and u.seed == at_seed)
        training = {**base.training, **{k: v for k, v in change.items()
                                        if k in TRAINING_KEYS}}
        evaluation = {**base.evaluation, **{k: v for k, v in change.items()
                                            if k in EVALUATION_KEYS}}
        return training, evaluation

    def classify(t: dict, e: dict) -> dict[str, Any]:
        return reuse.classify(t, e, evidence, training_keys=TRAINING_KEYS,
                              evaluation_keys=EVALUATION_KEYS)

    requests = []
    for unit in units:
        t, e = request(unit.arm, unit.rung, unit.seed)
        requests.append({"request": f"{unit.arm} @ {unit.rung} seed "
                                    f"{unit.seed}", **classify(t, e)})
    first = units[0]
    shown = {
        "a new D2/D3 candidate (unseen initialization)": classify(
            *request(first.arm, first.rung, first.seed,
                     initialization_digest="<new candidate digest>")),
        "B at a shared seed, evaluated on a reserved fresh battery": classify(
            *request("B", "shared_confirmation",
                     protocol["rungs"]["shared_confirmation"]["seeds"][0],
                     evaluation_protocol_hash="<d2_confirmation protocol>")),
        "B at a shared seed under a different trainer implementation":
            classify(*request("B", "shared_confirmation",
                              protocol["rungs"]["shared_confirmation"][
                                  "seeds"][0],
                              trainer_source_digest="<changed trainer>")),
        "B at a seed D1 never trained": classify(
            *request("B", "shared_confirmation",
                     protocol["rungs"]["shared_confirmation"]["seeds"][0],
                     seed=1)),
    }

    # D1's decisions, re-derived through THIS module's path from the imported
    # rows. Equality is what shows the shared path reads the evidence the way
    # D1 did; a mismatch would be a defect in this module, not a finding.
    d1_verdict = _read(_run_dir("shared_confirmation", repo_root) /
                       "evidence" / "confirmation_verdict.json", repo_root)
    rows_b = load_reference_rows("shared_confirmation", "B",
                                 repo_root=repo_root,
                                 evidence_root=evidence_root)
    rows_q2 = load_reference_rows("shared_confirmation", "q2",
                                  repo_root=repo_root,
                                  evidence_root=evidence_root)
    replay = confirmation_verdict(
        "q2", rows_q2, rows_b, protocol=protocol, alpha=0.05,
        bootstrap_seed_value=D1V.bootstrap_seed(repo_root))
    keys = ("verdict", "delta", "lcb_one_sided", "ucb_one_sided")
    verdict_equal = (
        all(replay["decision"][k] == d1_verdict["decision"][k] for k in keys)
        and replay["per_seed_delta"] == d1_verdict["per_seed_delta"])

    d1_rank = _read(_run_dir("shared_screening", repo_root) / "evidence" /
                    "d1_screening_ranking_offpod.json", repo_root)
    d1_field = [FieldArm(a.arm_id, a.quality_position, a.state_id)
                for a in D1B.arms(repo_root) if not a.is_incumbent]
    run_dir = _run_dir("shared_screening", repo_root)
    scored, comparator = [], []
    for path in sorted((run_dir / "evidence" / "results").glob("*.json")):
        r = json.loads(path.read_text())
        row = {"arm": r["screening_arm"], "seed": r["seed"],
               "correct_overall": r["correct_overall"],
               "usable_rollout_rate": r["usable_rollout_rate"]}
        (comparator if row["arm"] == "B" else scored).append(row)
    replay_rank = rank_screening(d1_field, scored, comparator,
                                 seeds=protocol["rungs"]["shared_screening"][
                                     "seeds"])
    rank_equal = (
        [(r["arm"], r["delta_vs_b"], r["vetoed"])
         for r in replay_rank["ranking"]["ranked"]]
        == [(r["arm"], r["delta_vs_b"], r["vetoed"])
            for r in d1_rank["ranking"]["ranked"]]
        and replay_rank["selection"].get("arm")
        == d1_rank["selection"].get("arm"))

    # What the shared protocol would report for q2 itself under the D2/D3
    # family alpha -- the working baseline's standing on the shared rows,
    # which every D2/D3 candidate is read beside.
    q2_at_family_alpha = confirmation_verdict(
        "q2", rows_q2, rows_b, protocol=protocol,
        alpha=protocol["statistics"]["family"]["per_experiment_alpha_one_sided"],
        bootstrap_seed_value=bootstrap_seed(protocol["protocol_hash"],
                                            "reference_q2",
                                            "shared_confirmation"))

    drift = tree_matches(protocol, repo_root)
    impl = {p: _sha256(repo_root / p) for p in (
        "scripts/stages/stage-1/families/d_series/shared_evaluation.py",
        "scripts/shared/evaluation/evidence_reuse.py")}
    all_verified = all(u["result_verified"] for u in per_unit)
    return {
        "schema": AUDIT_SCHEMA,
        "_contract": (
            "Whether every B/q2 unit D1 measured can be consumed by the "
            "shared protocol: bytes re-hashed against the committed records, "
            "per-sample rows recomputed from the raw generations by the "
            "rung's own scorer, and D1's two decisions re-derived through the "
            "shared path. $0, CPU. Deterministic: rerun to check."),
        "protocol_hash": protocol["protocol_hash"],
        "implementation_sha256": impl,
        "rescored": rescore,
        "summary": {
            "units": len(per_unit),
            "result_verified": sum(u["result_verified"] for u in per_unit),
            "rescore_reproduces_per_sample": sum(
                bool(u["checks"].get("rescore_reproduces_per_sample"))
                for u in per_unit),
            "checkpoint_bytes_present": sum(
                u["checks"]["checkpoint_bytes_present"] for u in per_unit),
            "all_reusable": all_verified and all(
                r["outcome"] == reuse.REUSE_RESULT for r in requests),
            "d1_verdict_reproduced": verdict_equal,
            "d1_screening_ranking_reproduced": rank_equal,
            "tree_matches_pinned_implementation": not any(drift.values()),
        },
        "tree_drift": drift,
        "units": per_unit,
        "reuse_requests": requests,
        "reuse_outcomes_illustrated": shown,
        "d1_replay": {
            "confirmation": {k: replay["decision"][k] for k in keys}
                            | {"per_seed_delta": replay["per_seed_delta"]},
            "screening_order": [r["arm"] for r in
                                replay_rank["ranking"]["ranked"]],
            "screening_selection": replay_rank["selection"].get("arm"),
        },
        "q2_vs_b_at_family_alpha": {
            "alpha_one_sided": q2_at_family_alpha["alpha_one_sided"],
            **{k: q2_at_family_alpha["decision"][k] for k in keys},
            "shared_outcome": shared_outcome(q2_at_family_alpha["decision"]),
            "_reading": ("descriptive: q2 is not a D2/D3 candidate and this "
                         "consumes none of the family's alpha. It is what a "
                         "candidate equal to q2 would be read as"),
        },
        "probe_economy": probe_economy(units, repo_root),
        "gaps": [
            "trained checkpoints of all ten units are retired: a unit can be "
            "REUSED AS MEASURED, never re-evaluated under another protocol",
            "the screening session kept run manifests for B only; q2's "
            "screening trainer/runtime identity is the session's (one "
            "checkout, one pod), recorded as such per unit",
            "the durable store is a single dev-box copy, pinned by each "
            "source run's files.sha256 manifest and now registered "
            "reproducibility_required in the checkpoint registry",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="write the protocol and audit records")
    ap.add_argument("--evidence-root", type=Path, default=None,
                    help="a relocated root holding the source runs' stores")
    ap.add_argument("--no-rescore", action="store_true")
    args = ap.parse_args(argv)
    protocol = build_protocol()
    report = audit(evidence_root=args.evidence_root,
                   rescore=not args.no_rescore, protocol=protocol)
    bodies = {RECORD: protocol, AUDIT: report}
    stale = []
    for rel, doc in bodies.items():
        body = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
        path = REPO_ROOT / rel
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
            print(f"wrote {rel}")
        elif not path.is_file() or path.read_text() != body:
            stale.append(rel)
    print(json.dumps(report["summary"], indent=1))
    if stale:
        print(f"STALE (rerun with --write): {stale}")
        return 1
    return 0 if report["summary"]["all_reusable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
