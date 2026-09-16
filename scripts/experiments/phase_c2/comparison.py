"""The B→C search-stage comparison, durable and identical in both branches.

**The hole this closes.** `run_phase_a_search` commits the beam ranking before
the baseline is resolved, which is right — the ranking is the durability
boundary and must not wait on anything. But a rebuilt B arrives *after* that
commit, through the retained-candidate path, and the generic search summary
serializes an imported candidate as identity and provenance only. So a run could
rebuild B, evaluate it correctly on the frozen suite, and lose B's actual
cheap-metric result at teardown — leaving a reviewer with a ranking over C and no
number for the baseline the whole question is asked against.

So the comparison is its own record, written after the search and after the
baseline resolves, and **both branches normalize to one schema**:

* **searched B** — B's `StateEvaluation` comes from the matching complete search
  leaf, which the beam already measured;
* **rebuilt B** — it comes from the evaluated retained state the fallback
  produced, which `run_phase_a_search` measured on the same primed evaluator and
  the same suite.

A reviewer answers B→C from committed evidence either way.

**`stage1_selection.json` is not touched.** That artifact is the beam's own
ranking and its durability boundary; rewriting it to insert a state the beam
never generated would make the search's record describe a candidate set the
search did not produce. This is a separate, later record that cites it.

**No second metric.** The comparison is the run's own `PARETO_V1` applied to
`{B} ∪ selected C` through the same public `rank()` the beam used, at
`beam_width=None` so nothing is pruned. The verdict is *computed* from the
resulting fronts and per-objective deltas, not asserted beside them: a prose
conclusion can be the negation of the arithmetic printed next to it and survive
review.

**What it is not.** A cheap-metric hypothesis-generation comparison. Not
behavioural recovery evidence, not a demonstrated initialization improvement,
and not a reason to prefer anything without a separately authorized
confirmation.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
for _extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from aadistill.infrastructure.manifest import sha256_json

SCHEMA = "aadistill.autoinit.c2_baseline_comparison/v1"
FILENAME = "c2_baseline_comparison.json"

NOT_BEHAVIOURAL = (
    "CHEAP-METRIC HYPOTHESIS GENERATION ONLY. Every number here is a step-0 "
    "state_eval measurement on the frozen suite. It is NOT behavioural recovery "
    "evidence, NOT a demonstrated initialization improvement, and NOT a reason "
    "to prefer any candidate. A B->C behavioural result needs three paired "
    "recovery seeds, the frozen 0.86M budget and the frozen battery, and is a "
    "separately authorized paid experiment.")


class ComparisonError(RuntimeError):
    """The comparison cannot be built from what the run produced."""


def _identity(state: Any) -> dict[str, Any]:
    """Everything that pins WHICH checkpoint a metric belongs to."""
    artifact = getattr(state, "artifact", None)
    out: dict[str, Any] = {
        "artifact_digest": state.artifact_digest,
        "num_parameters": state.num_parameters,
    }
    if artifact is not None:
        out.update({
            "weights_digest": artifact.weights_digest,
            "config_sha256": artifact.config_sha256,
            "arch_signature": artifact.arch_signature,
            "tokenizer_sha256": artifact.tokenizer_sha256,
            "index_sha256": artifact.index_sha256,
            "single_shard_sha256": artifact.single_shard_sha256,
        })
    return out


def _measured(state: Any, what: str) -> dict[str, Any]:
    """A state's complete evaluation, or a refusal naming what is missing.

    The refusal is the point of the whole module: an unmeasured B reaching this
    record silently would produce a comparison with a hole exactly where the
    baseline belongs.
    """
    evaluation = getattr(state, "evaluation", None)
    if evaluation is None:
        raise ComparisonError(
            f"{what} ({state.state_id}) carries no state_eval result. It cannot "
            "be compared, and writing a comparison without it would lose the "
            "one number this record exists to preserve.")
    return evaluation.as_dict()


def build(*, baseline: Any, baseline_outcome: dict[str, Any],
          candidates: Any, suite: Any, policy: Any,
          run_id: str, config_hash: str,
          selection_record: str | None = None,
          search_summary: str | None = None,
          frozen_candidate_inputs: str | None = None,
          interpretation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The record. `candidates` is the committed top-N selection, in order.

    B is removed from `candidates` when the beam selected it, so it is never
    compared with itself: an identical pair would land in the same front and
    read as "no candidate beats the baseline" for a reason that is arithmetic
    rather than scientific.

    `selection_record` cites the BEAM RANKING. `frozen_candidate_inputs` is a
    separate citation for the later derived extraction a baseline-completion
    session reads its candidates from -- separate because they are different
    documents making different claims, and overloading one field would let a
    record cite a derived extraction where a reader expects the ranking.

    `interpretation` is merged in BEFORE `record_sha256` is computed. That is
    the whole reason it is a parameter: the completion session has a
    preregistered numerical-sensitivity disclosure to attach, and a caller that
    attached it afterwards would leave the document differing from the object
    its own self-hash describes. Finalization happens once, here, and nothing
    downstream is expected to re-do it.
    """
    baseline_evaluation = _measured(baseline, "the baseline B")

    duplicates = [c.state_id for c in candidates
                  if c.state_id == baseline.state_id
                  or c.artifact_digest == baseline.artifact_digest]
    compared = [c for c in candidates if c.state_id not in set(duplicates)]

    #: The run's OWN policy, through its public entry point, at beam_width=None
    #: so nothing is pruned. Not a reimplementation of epsilon-dominance: a
    #: second implementation of the comparison rule is how a record comes to
    #: disagree with the search that produced it.
    ranked = policy.rank([baseline, *compared], None)
    front_of = {state_id: index for index, front in enumerate(ranked.fronts)
                for state_id in front}
    baseline_front = front_of.get(baseline.state_id)

    per_candidate = []
    for candidate in compared:
        deltas = {}
        for objective in policy.objectives:
            b = float(baseline_evaluation["values"][objective.key])
            c = float(candidate.evaluation.values[objective.key])
            epsilon = float(policy.epsilon.get(objective.key, 0.0))
            deltas[objective.key] = {
                "direction": objective.direction,
                "baseline": b, "candidate": c,
                "delta_candidate_minus_baseline": c - b,
                "candidate_better": objective.better(c, b),
                "within_epsilon": abs(c - b) <= epsilon,
                "epsilon": epsilon,
            }
        candidate_front = front_of.get(candidate.state_id)
        per_candidate.append({
            "state_id": candidate.state_id,
            "path_label": candidate.path_label,
            "identity": _identity(candidate),
            "evaluation": _measured(candidate, "a selected candidate"),
            "front": candidate_front,
            #: Front membership under the run's own epsilon-Pareto policy. A
            #: lower front number is not "better on a scalar" — it means
            #: nothing in the set epsilon-dominates it.
            "in_same_front_as_baseline": candidate_front == baseline_front,
            "front_is_better_than_baseline": (
                candidate_front is not None and baseline_front is not None
                and candidate_front < baseline_front),
            "front_is_worse_than_baseline": (
                candidate_front is not None and baseline_front is not None
                and candidate_front > baseline_front),
            "objectives": deltas,
        })

    better = [c["state_id"] for c in per_candidate
              if c["front_is_better_than_baseline"]]
    worse = [c["state_id"] for c in per_candidate
             if c["front_is_worse_than_baseline"]]
    same = [c["state_id"] for c in per_candidate
            if c["in_same_front_as_baseline"]]

    #: DERIVED from the three counts above, never written beside them. A stated
    #: conclusion can be the negation of the arithmetic it sits next to.
    if not per_candidate:
        verdict = "NO_CANDIDATE_TO_COMPARE"
    elif better:
        verdict = "CANDIDATE_IN_A_BETTER_FRONT_THAN_BASELINE"
    elif same:
        verdict = "BASELINE_AND_CANDIDATES_SHARE_A_FRONT"
    else:
        verdict = "BASELINE_IN_A_BETTER_FRONT_THAN_EVERY_CANDIDATE"

    record = {
        "schema": SCHEMA,
        "_contract": (
            "The Phase-C2 Search-1 B->C comparison on the search-stage metric. "
            "Written AFTER the beam ranking was committed and AFTER the "
            "baseline resolved, so that a rebuilt B's evaluation survives "
            "teardown. It does not modify stage1_selection.json, which remains "
            "the beam's own ranking and durability boundary."),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "search_config_hash": config_hash,
        "cites": {"beam_ranking": selection_record,
                  "search_summary": search_summary,
                  #: A DERIVED extraction of the ranking's measured candidates,
                  #: not the ranking. `None` for a session that read the
                  #: candidates from the live search.
                  "frozen_candidate_inputs": frozen_candidate_inputs,
                  "_not_modified": (
                      "the beam ranking artifact is cited, never rewritten: "
                      "inserting a state the beam did not generate would make "
                      "the search's own record describe a candidate set it "
                      "never produced")},
        "suite": {"id": suite.qualified_id, "hash": suite.suite_hash},
        "policy": {
            "id": policy.qualified_id, "hash": policy.policy_hash,
            "objectives": [o.as_dict() for o in policy.objectives],
            "epsilon": {k: float(v) for k, v in sorted(policy.epsilon.items())},
            "tie_break": list(policy.tie_break),
            "diversity_key": policy.diversity_key,
        },
        "baseline": {
            "resolution": baseline_outcome.get("resolution"),
            "rebuilt": bool(baseline_outcome.get("rebuilt")),
            "state_id": baseline.state_id,
            "provenance": getattr(baseline, "provenance", None),
            "path_label": baseline.path_label,
            "identity": _identity(baseline),
            "evaluation": baseline_evaluation,
            "frozen_identity_matches": baseline_outcome.get("identity_matches"),
            "construction": baseline_outcome.get("construction"),
            "_both_branches_one_schema": (
                "a searched B's evaluation comes from the matching complete "
                "beam leaf; a rebuilt B's comes from the evaluated retained "
                "state the fallback produced, measured on the same primed "
                "evaluator and the same suite. The fields below this line are "
                "identical either way."),
        },
        "candidates": per_candidate,
        "excluded_as_duplicate_of_baseline": sorted(duplicates),
        "_why_excluded": (
            "the beam selected the baseline itself. Comparing B with B would "
            "place an identical pair in one front and read as 'no candidate "
            "beats the baseline' for an arithmetic reason rather than a "
            "scientific one."),
        "comparison": {
            "method": (
                "the run's own policy via its public rank(), at "
                "beam_width=None so nothing is pruned, over {B} union the "
                "committed top-N selection minus any duplicate of B"),
            "fronts": [list(f) for f in ranked.fronts],
            "baseline_front": baseline_front,
            "candidates_in_a_better_front": better,
            "candidates_in_the_same_front": same,
            "candidates_in_a_worse_front": worse,
            "verdict": verdict,
            "_verdict_is_derived": (
                "computed from the three id lists above, not written beside "
                "them"),
        },
        "not_behavioural_evidence": NOT_BEHAVIOURAL,
    }
    if interpretation:
        #: Before the hash, and refusing to shadow a computed field: a
        #: disclosure that could overwrite `comparison` or `candidates` would be
        #: a way to edit the result through the metadata argument.
        collisions = sorted(set(interpretation) & set(record))
        if collisions:
            raise ComparisonError(
                f"the interpretation supplies {collisions}, which the record "
                "already computes. Interpretation is added beside the result, "
                "never over it.")
        record.update(dict(interpretation))
    #: LAST. Every field above is inside the hash, including any
    #: interpretation, so a reader can recompute this over the document minus
    #: this key and get the same value.
    record["record_sha256"] = sha256_json(record)
    return record


def commit(record: dict[str, Any], directory: str | Path) -> Path:
    """Write it atomically. Temp file plus `os.replace`, one filesystem.

    The same rule the relay repair established: a reader never sees a
    half-written document, and a shorter new version cannot leave the tail of a
    longer old one behind it.
    """
    out = Path(directory) / FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    with open(tmp, "w") as handle:
        handle.write(json.dumps(record, indent=2, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, out)
    return out


def resolve_baseline_state(*, found: Any, outcome: dict[str, Any]) -> Any:
    """The live B state, whichever branch produced it. One place decides.

    Refuses rather than guessing: a run that cannot say which state is B has no
    baseline, and a comparison built against the wrong one is worse than none.
    """
    from experiments.phase_c2 import baseline as B

    resolution = (outcome or {}).get("resolution")
    if resolution == "searched":
        state = B.find_searched_baseline(found.result)
        if state is None:
            raise ComparisonError(
                "the fallback recorded resolution='searched' and no complete "
                "leaf carries B's construction. Those cannot both be true.")
        return state
    if resolution == "rebuilt":
        matches = [s for s in found.imported
                   if s.state_id == B.REBUILT_STATE_ID]
        if len(matches) != 1:
            raise ComparisonError(
                f"resolution='rebuilt' but {len(matches)} injected states carry "
                f"{B.REBUILT_STATE_ID!r}. The rebuilt baseline must be exactly "
                "one measured retained state.")
        return matches[0]
    raise ComparisonError(
        f"the baseline fallback recorded resolution={resolution!r}; it must be "
        "'searched' or 'rebuilt' before a comparison can be built.")
