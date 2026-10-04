#!/usr/bin/env python3
"""Close a D1 GPU qualification subrun: file its evidence, book its cost, derive
the verdict.

    python scripts/pod/d1_qualification_closeout.py RUN_ID --subrun s2

Three things happen, each to the one record that owns the fact:

* the subrun's small JSON/log evidence is copied from the gitignored artifacts
  tree into ``…/gpu-qualification/v1/runs/<subrun>/`` — it is tens of kilobytes
  of identities and measurements, which is exactly what belongs in git, while
  the checkpoints stay off it;
* the cost is appended to ``v1/campaign.json`` and ``booked_usd`` is RECOMPUTED
  from the components rather than incremented, because an incremented total
  silently drops an entry;
* ``v1/closeout.json`` is written when the qualification is complete. That file
  is the single owner of whether the owed GPU validation ran:
  ``write_d1_design._qualification_state`` reads it, so the D1 design's status
  moves with it and no one has to remember to edit a sentence.

The verdict is DERIVED from the record's own gates, never passed in. A closeout
that let a caller name the verdict would let a failed run be filed as passed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
QUAL = REPO / "logs/stages/stage-1/phase_d1/validations/gpu-qualification/v1"
CAMPAIGN = QUAL / "campaign.json"
CLOSEOUT = QUAL / "closeout.json"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"no record at {path}")
    return json.loads(path.read_text())


#: The run writes TWO records with the same basename: the `--check-only`
#: preflight's and the qualification's. `rglob` returns them in whatever order
#: the filesystem offers, so taking the first would read the preflight on some
#: runs and the qualification on others — reporting a PASSED run as INCOMPLETE
#: because the record it read legitimately contains no measurements. Name the
#: one that is meant, and refuse an ambiguity rather than guess at it.
RECORD = "qualification/qualification.json"
PREFLIGHT_STATUS = "CHECK_ONLY_OK"
#: What `d1_qualification_driver` writes when every stage ran.
COMPLETE_STATUS = "COMPLETE"


def qualification_record(run_dir: Path) -> Path:
    path = run_dir / "out" / RECORD
    if not path.is_file():
        others = sorted(p.relative_to(run_dir)
                        for p in (run_dir / "out").rglob("qualification.json"))
        raise SystemExit(
            f"no qualification record at {path.relative_to(run_dir)}. "
            + (f"Found instead: {others}. The preflight's record is not the "
               "qualification's and must not be closed out as one."
               if others else
               "The subrun produced no record, so there is nothing to close: "
               "book its cost by hand and say why."))
    if json.loads(path.read_text()).get("status") == PREFLIGHT_STATUS:
        raise SystemExit(f"{path.relative_to(run_dir)} carries the preflight "
                         f"status {PREFLIGHT_STATUS!r}. A check-only pass "
                         "measured nothing and cannot be a qualification.")
    return path


def verdict_of(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The verdict and the answers, derived from the record's own gates.

    PASSED requires every gate the qualification exists to answer. Anything
    less is reported as what it is: the point of a qualification is to find out,
    and a partial run filed as passed would be worse than no run at all.
    """
    answers: dict[str, Any] = {}
    unmet: list[str] = []

    for key, name in (("bound_protocol_incumbent", "incumbent"),
                      ("bound_protocol_target_aware", "target-aware")):
        bound = record.get(key) or {}
        suite = bound.get("suite_content_sha256")
        if not suite or suite == "unbound":
            unmet.append(f"the {name} protocol carries no bound suite content")
        else:
            answers[f"protocol_{name.replace('-', '_')}"] = {
                "measurement_protocol_id": bound.get("measurement_protocol_id"),
                "position_policy": bound.get("position_policy"),
                "suite_content_sha256": suite,
                "numerical_environment": bound.get("numerical_environment"),
            }

    a = record.get("A_incumbent") or {}
    if not a:
        unmet.append("A did not run: the incumbent was never reconstructed")
    else:
        matched = bool(a.get("reconstructed"))
        answers["incumbent_reconstruction"] = {
            "matched_the_frozen_incumbent": matched,
            "artifact_digest": a.get("final_artifact_digest"),
            "expected": a.get("expected_final_artifact_digest"),
            "seconds": a.get("seconds"),
            "peak_memory_bytes": a.get("peak_memory_bytes"),
        }
        if not matched:
            unmet.append("A's reconstructed identity does not equal the frozen "
                         "incumbent's")

    b = record.get("B_target_aware") or {}
    if not b:
        unmet.append("B did not run: the target-aware path was never executed")
    else:
        answers["target_aware_execution"] = {
            "artifact_digest": b.get("final_artifact_digest"),
            "batch_size": b.get("batch_size"),
            "seconds": b.get("seconds"),
            "peak_memory_bytes": b.get("peak_memory_bytes"),
        }

    c = record.get("C_selection_comparison") or {}
    if not c:
        unmet.append("C did not run: no selection comparison exists")
    else:
        #: A difference here is EVIDENCE, not a failure. It is reported, with
        #: the steps that moved, and it does not make the verdict anything.
        answers["discrete_decisions"] = {
            "n_moved": c.get("n_moved"),
            "final_artifacts_differ": c.get("final_artifacts_differ"),
            "steps_that_moved": [s for s in (c.get("steps") or [])
                                 if s.get("selection_differs")
                                 or s.get("artifact_digest_differs")],
            "_reading": ("a CPU-vs-GPU or policy-vs-policy selection difference "
                         "is evidence about the operators, not a qualification "
                         "failure. What would fail is an unexplained difference "
                         "between two runs of the SAME environment."),
        }

    d = record.get("D_state_eval") or []
    if not d:
        unmet.append("D did not run: state-eval peak memory was never measured")
    else:
        answers["state_eval_memory"] = [{
            "label": m.get("label"),
            "peak_memory_bytes": m.get("peak_memory_bytes"),
            "evaluation_delta_bytes": m.get("evaluation_delta_bytes"),
            "predicted_peak_logit_bytes": m.get("predicted_peak_logit_bytes"),
            "delta_over_predicted_logits": m.get("delta_over_predicted_logits"),
            "derived_budget_bytes": m.get("derived_budget_bytes"),
            "within_derived_budget": m.get("within_derived_budget"),
            "seconds": m.get("seconds"),
        } for m in d]
        over = [m.get("label") for m in d
                if m.get("within_derived_budget") is False]
        if over:
            unmet.append(f"the derived memory budget does not bound the real "
                         f"peak for {over}")

    if a.get("seconds") and b.get("seconds"):
        answers["timing"] = {
            "incumbent_seconds": a["seconds"],
            "target_aware_seconds": b["seconds"],
            "ratio_target_aware_over_incumbent": round(
                b["seconds"] / a["seconds"], 4),
        }
    else:
        unmet.append("E has no representative timing from both arms, so D1 "
                     "cannot be repriced from measurement")

    #: The DRIVER writes "COMPLETE"; nothing writes "PASSED". Testing for a
    #: string the producer never emits would call every complete run
    #: incomplete -- the same two-sided field contract that has bitten this
    #: repository before, caught here at $0 rather than after a paid run.
    if record.get("status") != COMPLETE_STATUS:
        unmet.append(f"the record's own status is "
                     f"{record.get('status')!r}, not {COMPLETE_STATUS!r}")

    return ("PASSED" if not unmet else "INCOMPLETE"), {
        "answers": answers, "unmet": unmet}


def file_evidence(run_dir: Path, subrun: str) -> dict[str, Any]:
    """Copy the subrun's small evidence into the stage tree, under its own id."""
    dest = QUAL / "runs" / subrun
    src = run_dir / "out"
    copied: list[str] = []
    if src.is_dir():
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in (".json", ".log", ".txt", ".jsonl"):
                continue
            rel = path.relative_to(src)
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            copied.append(str(rel))
    for name in ("cost.json", "watchdog.jsonl"):
        path = run_dir / name
        if path.is_file():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest / name)
            copied.append(name)
    return {"filed_under": str(dest.relative_to(REPO)), "files": copied}


def book(subrun: str, cost: dict[str, Any], outcome: str, record_rel: str,
         notes: str | None, gpu: str) -> dict[str, Any]:
    camp = _load(CAMPAIGN)
    if any(s["subrun_id"] == subrun for s in camp["subruns"]):
        raise SystemExit(f"{subrun} is already booked; it is not booked twice")
    entry = {
        "subrun_id": subrun,
        "pod_id": cost.get("pod_id"),
        "gpu": gpu,
        "rate_usd_per_hour": cost.get("rate_usd_per_hour"),
        "elapsed_minutes": cost.get("elapsed_minutes"),
        "container_disk_gb": cost.get("container_disk_gb"),
        "gpu_usd": cost.get("gpu_usd"),
        "container_disk_usd": cost.get("container_disk_usd"),
        "cost_usd": cost.get("all_in_usd"),
        "outcome": outcome,
        "evidence": record_rel,
    }
    if notes:
        entry["notes"] = notes
    camp["subruns"].append(entry)
    #: RECOMPUTED, never incremented. An incremented total drops an entry
    #: silently and the drop is invisible until someone re-adds by hand.
    camp["booked_usd"] = round(sum(s["cost_usd"] or 0.0
                                   for s in camp["subruns"]), 4)
    camp["_booked_recomputed_from_components"] = (
        " + ".join(str(s["cost_usd"]) for s in camp["subruns"])
        + f" = {camp['booked_usd']}")
    CAMPAIGN.write_text(json.dumps(camp, indent=1, sort_keys=True) + "\n")
    return camp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--subrun", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--notes")
    ap.add_argument("--artifacts", default="artifacts/qualification/d1_gpu")
    args = ap.parse_args(argv)

    run_dir = REPO / args.artifacts / args.run_id
    if not run_dir.is_dir():
        raise SystemExit(f"no run at {run_dir}")
    doc = _load(qualification_record(run_dir))
    verdict, derived = verdict_of(doc)
    filed = file_evidence(run_dir, args.subrun)
    cost = _load(run_dir / "cost.json")
    gpu = (doc.get("environment") or {}).get("gpu_name") or "unrecorded"
    camp = book(args.subrun, cost, verdict, filed["filed_under"],
                args.notes, gpu)

    print(f"subrun {args.subrun}: {verdict}")
    for line in derived["unmet"]:
        print(f"  unmet: {line}")
    print(f"  filed {len(filed['files'])} evidence files under "
          f"{filed['filed_under']}")
    print(f"  campaign booked ${camp['booked_usd']:.4f} of "
          f"${camp['ceiling_usd']:.4f}")

    if verdict != "PASSED":
        print("\nNo closeout.json written: it is the record that says the owed "
              "GPU validation RAN, and an incomplete qualification has not "
              "answered what it was created to answer.")
        return 1

    CLOSEOUT.write_text(json.dumps({
        "schema": "aadistill.d1.gpu_qualification_closeout/v1",
        "verdict": verdict,
        "subrun_id": args.subrun,
        "qualification_commit": args.commit,
        "gpu": gpu,
        "price_per_hour_usd": cost.get("rate_usd_per_hour"),
        "cost_usd": camp["booked_usd"],
        "paid_subruns": len(camp["subruns"]),
        "evidence": filed["filed_under"],
        "answers": derived["answers"],
        "_authorizes": ("nothing. This is ENGINEERING evidence from the "
                        "engineering book, which does not transfer into the "
                        "formal one. It closes none of D1's open blockers and "
                        "it is not D1 authorization."),
        "_owner_of": ("whether the GPU validation D1 owes has run. "
                      "scripts/autoinit/write_d1_design.py reads this file, so "
                      "the design's status derives from it."),
    }, indent=1, sort_keys=True) + "\n")
    print(f"  wrote {CLOSEOUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
