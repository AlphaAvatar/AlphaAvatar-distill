#!/usr/bin/env python3
"""Verify a completed C2 search's evidence, then freeze the candidate side of B->C.

    PYTHONPATH=src:scripts python scripts/autoinit/freeze_c2_comparison_inputs.py \
        --store /path/to/external/search/evidence --run-id attempt4 --stage-id 1

Phase-C2 Search-1 attempt 4 completed its beam search and then failed to rebuild
the frozen baseline B, so the comparison the search exists to feed was never
computed. A later session will measure B once and compute it. That session must
consume the candidate measurements this one produced -- not remeasure them, and
not recompute the ranking -- so the C side has to be independently reviewable
WITHOUT the 28.9 MB journal it currently lives in.

This tool does exactly two things, in order, and refuses to do the second
without the first:

**Verify that the journal reproduces the selection's commitment.** The selection
record binds a journal by sha256. If the file that sha256 names does not hash to
it, or does not contain the states the selection says it selected, or contains
them in a state that was never measured, then the selection is not a description
of that journal and neither may be used. Every check below is a refusal, because
a search result whose journal does not reproduce its own selection cannot be
used for baseline completion.

**Extract, deterministically.** Values are COPIED from the hash-verified journal
under `StateStore`'s own canonical-record rule. Nothing is measured, nothing is
ranked, nothing is recomputed -- and the three ranked objectives are cross-
checked against the values the selection's own decision rows already carry, so
an extraction that read the wrong record cannot pass silently.

The output is small by design: identities, the ranked objectives, and each
candidate's complete `StateEvaluation.as_dict()`. No checkpoint bytes, no
journal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from aadistill.initialization.planning import stage1_selection  # noqa: E402
from aadistill.initialization.planning.ranking import PARETO_V1  # noqa: E402
from aadistill.initialization.specs.state import StateStore  # noqa: E402
from experiments.run_layout import rel_run_dir  # noqa: E402

#: Bumped whenever the extraction changes what it copies or how it chooses a
#: record. A consumer that finds a rule it does not know must refuse rather than
#: assume this document means what the previous version meant.
EXTRACTION_RULE = "c2.frozen_comparison_inputs/v1"

SCHEMA = "aadistill.autoinit.c2_frozen_comparison_inputs/v1"

#: Relative to the external store root. The search writes these three under one
#: workdir; the store preserves that layout.
SELECTION_REL = "autoinit/phase_c2_search/stage1_selection.json"
JOURNAL_REL = "autoinit/phase_c2_search/states.jsonl"
TELEMETRY_REL = "autoinit/phase_c2_search/telemetry.jsonl"
#: The pod-side collection manifest, preserved beside the artifacts it lists.
COLLECTION_REL = "manifest.json"

#: The one lifecycle point at which a candidate carries a measurement.
MEASURED = "measured"


class FreezeError(RuntimeError):
    """The evidence does not support freezing a comparison input."""


def verify(store: Path) -> dict:
    """Every check the extraction is not allowed to proceed without.

    Returns the verified context. Raises `FreezeError` naming the first failure,
    because a partial verification report invites picking the convenient half.
    """
    for rel in (SELECTION_REL, JOURNAL_REL, COLLECTION_REL):
        if not (store / rel).is_file():
            raise FreezeError(f"{rel} is absent from {store}")

    #: 0. The selection still matches its OWN commitment hash, through the
    #: reader that owns that rule. `selection_sha256` covers the record minus
    #: its own hash and `generated_utc`, so an edited selection is refused here
    #: rather than silently re-frozen. Reimplementing the check would be a
    #: second definition of what a selection commits to.
    try:
        selection = stage1_selection.load(store / SELECTION_REL)
    except ValueError as exc:
        raise FreezeError(str(exc)) from exc
    journal_path = store / JOURNAL_REL

    #: 1. The journal IS the one the selection committed to.
    journal_sha256 = sha256_file(journal_path)
    bound = (selection.get("journal") or {}).get("sha256")
    if journal_sha256 != bound:
        raise FreezeError(
            f"the journal at {journal_path} hashes to {journal_sha256}, but the "
            f"selection binds {bound}. This file is not the journal that "
            "selection describes, so neither may be used.")

    #: 2. The selection's own bytes, so a consumer can bind this document to it.
    selection_sha256 = sha256_file(store / SELECTION_REL)

    #: 3. The canonical record per state id, by StateStore's rule and not a
    #: reimplementation of it: the journal is append-only and carries 81 records
    #: for 44 states, so "the record for a state" is a decision, not a lookup.
    latest = StateStore(journal_path).latest_by_state_id()

    required = [objective.key for objective in PARETO_V1.objectives]
    suite_hash = (selection.get("suite") or {}).get("hash")
    decisions = {row["state_id"]: row for row in selection.get("decisions", ())}

    verified = []
    for committed in selection["selected"]:
        state_id = committed["state_id"]
        if state_id not in latest:
            raise FreezeError(
                f"the selection selected {state_id} and the journal has no "
                "record of it")
        record = latest[state_id]
        evaluation = record.get("evaluation") or {}

        if record.get("validity") != MEASURED:
            raise FreezeError(
                f"{state_id} is {record.get('validity')!r} in its canonical "
                f"journal record, not {MEASURED!r}; an unmeasured state cannot "
                "be a comparison input")
        if record.get("artifact_digest") != committed["artifact_digest"]:
            raise FreezeError(
                f"{state_id}: the journal record digests to "
                f"{record.get('artifact_digest')} and the selection committed "
                f"{committed['artifact_digest']}")
        if evaluation.get("artifact_digest") != committed["artifact_digest"]:
            raise FreezeError(
                f"{state_id}: its evaluation measured "
                f"{evaluation.get('artifact_digest')}, which is not the "
                f"artifact the selection committed. A measurement of different "
                "bytes is not this candidate's measurement.")
        if evaluation.get("suite_hash") != suite_hash:
            raise FreezeError(
                f"{state_id}: measured on suite {evaluation.get('suite_hash')} "
                f"and the selection names {suite_hash}; values from two suites "
                "are not comparable")
        missing = [k for k in required if k not in (evaluation.get("values") or {})]
        if missing:
            raise FreezeError(
                f"{state_id}: its evaluation is missing ranked objectives "
                f"{missing}; an incomplete evaluation may not be ranked")

        #: The cross-check that makes a wrong-record read loud. The selection's
        #: decision rows already carry the three ranked values; if the record
        #: this extraction chose disagrees with them, one of the two is not the
        #: measurement the ranking used.
        row = decisions.get(state_id)
        if row is None:
            raise FreezeError(
                f"{state_id} is selected but has no decision row, so its "
                "ranked values cannot be cross-checked")
        for key in required:
            extracted = float(evaluation["values"][key])
            committed_value = float(row["objectives"][key])
            if extracted != committed_value:
                raise FreezeError(
                    f"{state_id}: extracted {key}={extracted!r} and the "
                    f"selection's decision row says {committed_value!r}")

        verified.append({"committed": committed, "record": record,
                         "evaluation": evaluation, "decision": row})

    return {"selection": selection, "selection_sha256": selection_sha256,
            "journal_sha256": journal_sha256,
            "journal_path_in_selection": (selection.get("journal") or {}).get("path"),
            "collection_manifest_sha256": sha256_file(store / COLLECTION_REL),
            "telemetry_sha256": (sha256_file(store / TELEMETRY_REL)
                                 if (store / TELEMETRY_REL).is_file() else None),
            "n_journal_records": len(StateStore(journal_path).records()),
            "n_distinct_states": len(latest),
            "required_objectives": required,
            "verified": verified}


def session_identity(repo: Path, run_dir: str) -> dict:
    """The executable this evidence was produced by, read from the run."""
    record = json.loads((repo / run_dir / "runtime/session.json").read_text())
    return {
        "session_commit": record["session_commit_check"]["session_commit"],
        "executable_closure_digest": record["c2_executable_closure"]["digest"],
        "executable_closure_n_files": record["c2_executable_closure"]["n_files"],
        "terminal": record["terminal"],
        "cost_usd": record["cost"]["actual_usd"],
    }


def freeze(store: Path, repo: Path, run_id: str, stage_id: str,
           experiment_id: str) -> dict:
    context = verify(store)
    selection = context["selection"]
    run_dir = rel_run_dir(experiment_id, run_id, stage_id)

    candidates = []
    for entry in context["verified"]:
        committed, record = entry["committed"], entry["record"]
        candidates.append({
            "state_id": committed["state_id"],
            "path_label": committed["path"],
            "front": entry["decision"]["front"],
            "position_in_front": entry["decision"]["position_in_front"],
            "lineage": entry["decision"].get("lineage"),
            "identity": {
                "artifact_digest": committed["artifact_digest"],
                "weights_digest": committed.get("weights_digest"),
                "single_shard_sha256": committed.get("single_shard_sha256"),
                "arch_signature": committed.get("arch_signature"),
                "num_parameters": committed.get("num_parameters"),
                "impl_ids": committed.get("impl_ids"),
                "calibration_profiles": committed.get("calibration_profiles"),
            },
            "validity": record["validity"],
            #: COMPLETE, so a later session can rehydrate the measurement
            #: without the journal. Copied, not rebuilt.
            "evaluation": entry["evaluation"],
            "ranked_objectives": {
                key: float(entry["evaluation"]["values"][key])
                for key in context["required_objectives"]},
            "cross_checked_against_selection_decision": True,
        })

    doc = {
        "schema": SCHEMA,
        "_what_this_is": (
            "the CANDIDATE side of the Phase-C2 Search-1 B->C comparison, frozen. "
            "The beam search completed and committed this ranking; the conditional "
            "baseline rebuild then failed, so B was never measured and the "
            "comparison was never computed. A later baseline-completion session "
            "measures B once and computes the comparison against exactly these "
            "five measurements."),
        "no_new_measurement": (
            "NO NEW MEASUREMENT -- deterministic extraction from the journal "
            "already hash-bound by stage1_selection.json. No state was "
            "materialized, no checkpoint was loaded, no metric was computed and "
            "no ranking was recomputed."),
        "extraction_rule": EXTRACTION_RULE,
        "run": {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "stage_id": stage_id,
            "run_dir": run_dir,
            **session_identity(repo, run_dir),
        },
        "search": {
            "run_id": selection["search"]["run_id"],
            "config_hash": selection["search"]["config_hash"],
            "seed": selection["search"]["seed"],
            "target_spec_hash": selection["search"]["target_spec_hash"],
            "complete_leaves": len(selection.get("decisions", ())),
            "n_selected": selection["n_selected"],
            "generated_utc": selection["generated_utc"],
        },
        "suite": selection["suite"],
        "policy": selection["policy"],
        "profiles": selection["profiles"],
        "sources": {
            #: Two different hashes, and conflating them is how a pinned
            #: constant comes to name something nobody computed. The FILE hash
            #: is over the raw bytes; `selection_sha256` is the selection's own
            #: COMMITMENT hash over its content minus that field and
            #: `generated_utc`, verified above by its owning reader.
            "selection_file_sha256": context["selection_sha256"],
            "selection_commitment_sha256": selection["selection_sha256"],
            "journal_sha256": context["journal_sha256"],
            "journal_path_in_selection": context["journal_path_in_selection"],
            "journal_records": context["n_journal_records"],
            "journal_distinct_states": context["n_distinct_states"],
            "telemetry_sha256": context["telemetry_sha256"],
            "external_collection_manifest_sha256": context["collection_manifest_sha256"],
            "_journal_is_not_in_git": (
                "28.9 MB. Its durable location is registered in "
                "logs/state/artifact_manifests.md; this document exists so the "
                "candidate side is reviewable without it."),
        },
        "canonical_record_rule": (
            "StateStore.latest_by_state_id -- the last written record per state "
            "id. The journal is append-only and holds more records than states, "
            "so this is a decision and it is the search's own."),
        "ranked_objectives": context["required_objectives"],
        "candidates_in_committed_order": candidates,
        "what_this_does_not_contain": [
            "checkpoint bytes, and no path to any",
            "any measurement of B, which does not exist",
            "any comparison, ranking or verdict",
        ],
        "authorizes": "nothing",
    }
    doc["self_sha256"] = sha256_json({k: v for k, v in doc.items()
                                      if k != "self_sha256"})
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", required=True, type=Path,
                    help=("the external root holding the preserved search "
                          "evidence. Required rather than defaulted: a machine "
                          "path does not belong in the executable."))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage-id", required=True)
    ap.add_argument("--experiment-id", default="phase_c2")
    ap.add_argument("--out", default=None,
                    help="defaults to the run's own evidence area")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    try:
        if args.verify_only:
            context = verify(args.store)
            print(f"VERIFIED: journal {context['journal_sha256'][:16]}… is the "
                  f"one the selection binds; {len(context['verified'])} selected "
                  f"state(s) measured on suite "
                  f"{context['selection']['suite']['hash'][:12]}…")
            return 0
        doc = freeze(args.store, REPO, args.run_id, args.stage_id,
                     args.experiment_id)
    except FreezeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3

    out = Path(args.out) if args.out else (
        REPO / doc["run"]["run_dir"] / "evidence/c2_frozen_comparison_inputs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")

    #: `--out` may legitimately point outside the repository -- a determinism
    #: check writes to a scratch path and compares bytes. Reporting the path
    #: must not be the thing that fails.
    try:
        shown = out.relative_to(REPO)
    except ValueError:
        shown = out
    print(f"wrote {shown}")
    print(f"  self_sha256          {doc['self_sha256']}")
    print(f"  selection file       {doc['sources']['selection_file_sha256']}")
    print(f"  selection commitment {doc['sources']['selection_commitment_sha256']}")
    print(f"  journal_sha256       {doc['sources']['journal_sha256']}")
    print(f"  collection manifest  {doc['sources']['external_collection_manifest_sha256']}")
    print(f"  candidates           {len(doc['candidates_in_committed_order'])} "
          f"of {doc['search']['complete_leaves']} complete leaves")
    print("  NO NEW MEASUREMENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
