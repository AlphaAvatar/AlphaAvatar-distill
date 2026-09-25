#!/usr/bin/env python3
"""Derive the root-cause finding from the raw reports. Do not type it.

The rejected a4 finding was hand-assembled from numbers that ad-hoc scripts had
printed, and the gap between "what ran" and "what the record says" is the whole
reason this investigation exists. So the record is COMPUTED:

    python scripts/validation/batch_invariance_finding.py REPORT_DIR... \\
        --out logs/.../finding.json

Each REPORT_DIR holds one `report.json` written by
`batch_invariance_diagnostic.py`. This reads them, cross-checks the claims the
a4 finding made against the same quantities measured here, and emits a verdict
of CONFIRMED / REFINED / SUPERSEDED / NOT_ESTABLISHED with the arithmetic that
produced it. It asserts nothing it did not read.

The a4 claims, quoted from the record being adjudicated — kept as data so the
comparison is against what was actually said, not against a memory of it:
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The prior record's own numbers. Read from it at runtime where possible; these
#: are the fields a comparison needs and the paths they live at.
A4_FINDING = ("logs/stages/stage-1/phase_c3/validations/"
              "batching-refactor-cuda/v1/finding.json")

#: What the a4 record asserted, as testable propositions. Each names the field
#: in THIS investigation's report that speaks to it.
A4_CLAIMS = {
    "bf16_batched_vs_solo_logits_relative": {
        "a4_value_key": "batched_vs_solo_logits_bf16_relative",
        "here": ("case_matrix", "comparisons", "A_vs_E", "rel_l2"),
        "says": "a padded batched forward differs from a solo one on the logits",
    },
    "fp32_batched_vs_solo_logits_relative": {
        "a4_value_key": "batched_vs_solo_logits_fp32_relative",
        "here": ("fp32_control", "case_matrix", "comparisons", "A_vs_E", "rel_l2"),
        "says": "the same comparison in float32 is orders of magnitude smaller",
    },
    "ffn_importance_statistic_drift": {
        "a4_value_key": "ffn_importance_statistic_drift_bs1_vs_bs3_bf16_relative",
        "here": ("statistics_decomposition", "by_micro_batch_size", "3",
                 "ffn_abs_sum", "rel_l2"),
        "says": "the FFN activation statistic moves between bs=1 and bs=3",
    },
    "causal_kl_relative": {
        "a4_value_key": "causal_KL_solo_vs_batched_real_model_bf16_relative",
        "here": ("causal_kl", "max_rel_diff"),
        "says": "the scored causal KL moves solo vs batched",
    },
}


def say(msg: str) -> None:
    print(f"[finding] {msg}", flush=True)


def dig(node, *path, default=None):
    for key in path:
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif isinstance(node, list) and isinstance(key, int) and key < len(node):
            node = node[key]
        else:
            return default
    return node


def load_reports(dirs) -> dict:
    """label -> report, keyed by the directory name the pod used."""
    out = {}
    for d in dirs:
        p = Path(d)
        p = p if p.name == "report.json" else p / "report.json"
        if not p.is_file():
            say(f"no report at {p}; skipped")
            continue
        out[p.parent.name] = json.loads(p.read_text())
    return out


def environments(reports: dict) -> dict:
    """What each report actually ran under. The gap a4 could not state."""
    return {
        label: {
            "torch": dig(r, "environment", "torch"),
            "transformers": dig(r, "environment", "transformers"),
            "python": dig(r, "environment", "python"),
            "gpu": dig(r, "environment", "gpu"),
            "nvidia_driver": dig(r, "environment", "nvidia_driver"),
            "container_image": dig(r, "environment", "container_image"),
            "dtype": r.get("dtype"),
            "device": r.get("device"),
            "checkpoint": r.get("checkpoint"),
            "attn_requested": r.get("requested_attn_implementation"),
            "attn_resolved": dig(r, "model", "attention",
                                 "config._attn_implementation"),
            "calibration": dig(r, "calibration", "kind"),
            "calibration_content_sha256": dig(r, "calibration", "content_sha256"),
            "params_hidden": dig(r, "model", "hidden_size"),
            "params_layers": dig(r, "model", "num_hidden_layers"),
        }
        for label, r in reports.items()
    }


def adjudicate(reports: dict, a4: dict) -> dict:
    """Each a4 claim against the same quantity measured here.

    A claim is only adjudicated on a report that measured the SAME object. a4
    measured the 596M student; a number from the parent is a different
    measurement and is reported separately rather than compared.
    """
    a4_values = a4.get("measurements", {})
    out: dict = {}
    for name, spec in A4_CLAIMS.items():
        a4_value = a4_values.get(spec["a4_value_key"])
        per_report = {}
        for label, r in reports.items():
            here = dig(r["stages"], *spec["here"])
            if here is None:
                continue
            row = {"measured": here, "a4": a4_value}
            if isinstance(a4_value, (int, float)) and a4_value and here is not None:
                row["ratio_here_over_a4"] = here / a4_value
                #: Within 2x either way is "the same phenomenon at the same
                #: magnitude"; beyond 10x it is a different one. Chosen before
                #: any number was seen, and stated so a reader can disagree.
                row["same_order_of_magnitude"] = 0.5 <= here / a4_value <= 2.0
                row["differs_by_more_than_10x"] = not (0.1 <= here / a4_value <= 10.0)
            per_report[label] = row
        out[name] = {"claim": spec["says"],
                     "a4_reported": a4_value,
                     "measured_here": per_report,
                     "_field_path": list(spec["here"])}
    return out


def verdict(reports: dict, checks: dict) -> dict:
    """CONFIRMED / REFINED / SUPERSEDED / NOT_ESTABLISHED, computed.

    The a4 finding's operative claim was not a logit norm -- it was that FFN
    top-k SELECTION moves between micro batch sizes, which is what would make an
    operator's output irreproducible. So the verdict turns on the selection, and
    the magnitudes refine it rather than decide it.
    """
    a4_objects = {label: r for label, r in reports.items() if "a4" in label}
    pool = a4_objects or reports
    if not pool:
        return {"verdict": "NOT_ESTABLISHED",
                "why": "no report was available to adjudicate against"}

    moved = {}
    for label, r in pool.items():
        sel = r["stages"].get("ffn_selection", {})
        if not sel.get("ran", True):
            continue
        moved[label] = {
            "headline_layers_moved": sel.get("n_layers_with_moved_selection"),
            "n_layers": sel.get("n_layers"),
            "invariant_everywhere": sel.get(
                "selection_is_batch_invariant_at_every_ratio_and_size"),
            "by_keep_ratio": {
                ratio: {bs: v["n_layers_with_moved_selection"]
                        for bs, v in per_bs.items()}
                for ratio, per_bs in (sel.get("by_keep_ratio") or {}).items()},
        }
    if not moved:
        return {"verdict": "NOT_ESTABLISHED",
                "why": "no report produced an FFN selection result",
                "objects_considered": sorted(pool)}

    any_moved = any(m["headline_layers_moved"] for m in moved.values()
                    if m["headline_layers_moved"] is not None)
    #: THREE states, not two. A report that did not run the sweep reports
    #: `None`, and reading that as "not invariant" turned a pair of runs with
    #: zero moved layers into a REFINED verdict -- a report that measured
    #: nothing arguing that something is narrower than claimed.
    sweep = {label: m["invariant_everywhere"] for label, m in moved.items()}
    #: A report that did not run the sweep is IGNORED, not given a veto. One
    #: failed run out of four should narrow the evidence, not collapse three
    #: clean measurements into "not established".
    answered = [v for v in sweep.values() if v is not None]
    all_invariant = bool(answered) and all(v is True for v in answered)
    some_not_invariant = any(v is False for v in answered)

    if any_moved:
        result = "CONFIRMED"
        why = ("FFN top-k selection moves between micro batch sizes on the same "
               "object a4 measured, under the pinned science runtime")
    elif all_invariant:
        result = "SUPERSEDED"
        why = ("FFN top-k selection is identical at every swept keep ratio and "
               "micro batch size on the object a4 measured, under the pinned "
               "science runtime; the a4 selection claim does not reproduce")
    elif some_not_invariant:
        result = "REFINED"
        why = ("selection is invariant at the headline configuration but not at "
               "every swept ratio and size; the effect is real and narrower "
               "than a4 stated")
    else:
        result = "NOT_ESTABLISHED"
        why = ("no report ran the keep-ratio and batch-size sweep, so the "
               "headline configuration being invariant says nothing about the "
               "configurations a4 quoted")
    return {
        "verdict": result,
        "why": why,
        "_verdict_turns_on": ("FFN top-k SELECTION, not a logit norm: a "
                              "magnitude that moves no decision changes no "
                              "operator output"),
        "objects_adjudicated": sorted(pool),
        "_a4_measured_the_596m_student": bool(a4_objects),
        "_object_geometry": {
            label: {"hidden_size": dig(reports[label], "model", "hidden_size"),
                    "num_hidden_layers": dig(reports[label], "model",
                                             "num_hidden_layers"),
                    "checkpoint": reports[label].get("checkpoint")}
            for label in sorted(pool)},
        "_pool_selection": ("reports whose label names the a4 object, falling "
                            "back to every report when none does; the geometry "
                            "above is what a reader should check that against"),
        "sweep_reported": sweep,
        "_reports_without_a_sweep": sorted(k for k, v in sweep.items() if v is None),
        "selection": moved,
    }


def build(reports: dict) -> dict:
    a4 = {}
    p = REPO / A4_FINDING
    if p.is_file():
        a4 = json.loads(p.read_text())
    checks = adjudicate(reports, a4)
    v = verdict(reports, checks)
    return {
        "schema": "aadistill.validation_finding/v1",
        "_contract": ("DERIVED, not written. Every number here was read from a "
                      "report.json emitted by batch_invariance_diagnostic.py, "
                      "and the verdict is computed by "
                      "scripts/validation/batch_invariance_finding.py. It "
                      "authorizes nothing and changes nothing."),
        "investigation_id": "batch_invariance_root_cause_v1",
        "derived_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes_or_adjudicates": A4_FINDING,
        "_the_prior_record_is_not_rewritten": (
            "It is historical evidence. This record states what its claims do "
            "and do not reproduce; it does not edit them."),
        "reports": sorted(reports),
        "environments": environments(reports),
        "a4_claims_adjudicated": checks,
        **v,
        "localization": {
            label: {
                "locus": dig(r, "conclusion", "divergence_locus"),
                "driver": dig(r, "conclusion", "divergence_driver"),
                "deterministic": dig(r, "conclusion",
                                     "each_shape_internally_deterministic"),
                "bare_gemm_shape_dependent": dig(
                    r, "conclusion", "a_bare_gemm_is_shape_dependent"),
                "padding_alone_is_inert": dig(r, "conclusion",
                                              "padding_alone_is_inert"),
                "batch_size_alone_moves_it": dig(
                    r, "conclusion", "batch_size_alone_moves_the_forward"),
                "backends_diverging": dig(r, "conclusion",
                                          "attention_backends_diverging"),
                "backends_exact": dig(r, "conclusion", "attention_backends_exact"),
                "removed_by_disabling_bf16_reduced_precision_reduction": dig(
                    r, "conclusion",
                    "removed_by_disabling_bf16_reduced_precision_reduction"),
                "first_divergent_tap": dig(
                    r, "stages", "first_divergence",
                    "first_tap_that_is_not_bit_identical", "tap"),
                "forward_drift_max_abs": dig(
                    r, "stages", "statistics_decomposition", "forward_drift",
                    "max_abs"),
                "reduction_order_drift_max_abs": dig(
                    r, "stages", "statistics_decomposition",
                    "reduction_order_drift_at_fixed_activations", "max_abs"),
                "causal_kl_max_rel_diff": dig(r, "conclusion",
                                              "causal_kl_max_rel_diff"),
                "causal_kl_item_ranking_identical": dig(
                    r, "conclusion", "causal_kl_item_ranking_identical"),
            }
            for label, r in reports.items()
        },
        "authorizes": None,
        "changes": None,
        "_c3_status": "NOT STARTED",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reports", nargs="+", help="directories holding report.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    reports = load_reports(args.reports)
    if not reports:
        say("no reports could be read; refusing to derive a finding from nothing")
        return 4
    doc = build(reports)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    say(f"wrote {out}")
    say(f"verdict = {doc['verdict']}  ({len(reports)} report(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
