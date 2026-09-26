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
    "ffn_activation_statistic_drift": {
        "a4_value_key": "ffn_importance_statistic_drift_bs1_vs_bs3_bf16_relative",
        "here": ("statistics_decomposition", "by_micro_batch_size", "3",
                 "ffn_abs_sum", "rel_l2"),
        "says": "the FFN activation statistic moves between bs=1 and bs=3",
    },
    #: The IMPORTANCE drift, which is the quantity a4's key actually names.
    #: `importance = E[|a|] * ||down_proj col||`, and the column norms are a
    #: property of the weights, identical in every arm -- so this and the
    #: statistic above track each other closely and are NOT the same number.
    #: Both are reported rather than one standing in for the other.
    "ffn_importance_drift": {
        "a4_value_key": "ffn_importance_statistic_drift_bs1_vs_bs3_bf16_relative",
        "here": ("__derived__", "max_importance_rel_l2_drift_bs3"),
        "says": "the per-neuron importance the top-k reads moves at bs=3",
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
            "calibration_n_items": dig(r, "calibration", "n_items"),
            #: WHICH executable produced this report. One investigation can
            #: collect reports from two commits, and without this a reader
            #: compares numbers from different code with no way to notice.
            "executable_sha256": dig(r, "executable", "sha256"),
            "executable_git_head": dig(r, "executable", "git_head"),
            "executable_clean_at_head": dig(r, "executable",
                                            "this_file_is_clean_at_git_head"),
            "stages_run": sorted(k for k, v in (r.get("stage_status") or {}).items()
                                 if v == "ok"),
            "params_hidden": dig(r, "model", "hidden_size"),
            "params_layers": dig(r, "model", "num_hidden_layers"),
        }
        for label, r in reports.items()
    }


def _derived(report: dict, name: str):
    """A quantity the report holds per-layer and the comparison needs summarised.

    Computed here rather than added to the diagnostic, because the diagnostic
    that produced these reports is the one a pod already ran: changing its
    output shape now would mean the numbers and the reader disagreed about which
    executable emitted them, which is the defect under investigation.
    """
    if name == "max_importance_rel_l2_drift_bs3":
        sel = report["stages"].get("ffn_selection") or {}
        ratio = f"{float(sel.get('headline_keep_ratio', 0.5)):.2f}"
        per_bs = dig(sel, "by_keep_ratio", ratio) or {}
        rows = dig(per_bs, "3", "per_layer")
        if not rows:
            return None
        return max(r["importance_rel_l2_drift"] for r in rows)
    return None


def _executables(reports: dict) -> dict:
    """Do all the reports come from the same code? Said, not assumed."""
    shas = {label: dig(r, "executable", "sha256") for label, r in reports.items()}
    distinct = sorted({v for v in shas.values() if v})
    return {
        "per_report": shas,
        "distinct_executables": len(distinct),
        "all_reports_share_one_executable": len(distinct) == 1,
        "_absent_means_older": ("a report with no `executable` block predates "
                                "the field; its code identity is the session's "
                                "recorded source sha, not this file"),
        "reports_without_an_executable_identity": sorted(
            k for k, v in shas.items() if not v),
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
            if spec["here"][0] == "__derived__":
                here = _derived(r, spec["here"][1])
            else:
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


def mechanism(reports: dict) -> dict:
    """WHERE the shape dependence lives, derived from the GEMM isolation.

    The interesting structure is that only SOME projections move. Reporting
    which, with their shapes, is what turns "bf16 is imprecise" into a
    mechanism a reader can check -- and the exact-vs-divergent split is data,
    not an interpretation.
    """
    out: dict = {"per_report": {}}
    for label, r in reports.items():
        g = r["stages"].get("gemm_isolation") or {}
        rows = g.get("per_projection") or {}
        if not rows:
            continue
        #: K/N -- the reduction depth over the output width. A GEMM with too
        #: few output tiles to fill the device is the one a library splits
        #: along K, and a split-K reduction combines its partial sums in an
        #: order that depends on the tile grid, which depends on M. So this
        #: ratio, not the dtype and not the layer, is what should separate the
        #: two groups if the cause is split-K. Reported so a reader can check
        #: the separation rather than take the reading on trust.
        def kn(v):
            return v["in_features"] / v["out_features"]

        exact = {n: {"shape": [v["in_features"], v["out_features"]],
                     "reduction_depth_over_output_width": round(kn(v), 4)}
                 for n, v in rows.items() if v["bitwise_identical"]}
        moved = {n: {"shape": [v["in_features"], v["out_features"]],
                     "reduction_depth_over_output_width": round(kn(v), 4),
                     "max_abs": v["max_abs"], "rel_l2": v["rel_l2"]}
                 for n, v in rows.items() if not v["bitwise_identical"]}
        f32 = r["stages"].get("fp32_control") or {}
        rc = r["stages"].get("reduction_control") or {}
        sd = r["stages"].get("statistics_decomposition") or {}
        out["per_report"][label] = {
            "gemm_shape_solo": g.get("shape_solo"),
            "gemm_shape_batched": g.get("shape_batched"),
            "projections_bit_identical": exact,
            "projections_shape_dependent": moved,
            "max_K_over_N_among_exact": (
                max(v["reduction_depth_over_output_width"] for v in exact.values())
                if exact else None),
            "min_K_over_N_among_shape_dependent": (
                max(0.0, min(v["reduction_depth_over_output_width"]
                             for v in moved.values())) if moved else None),
            "K_over_N_separates_the_two_groups": (
                bool(exact) and bool(moved)
                and max(v["reduction_depth_over_output_width"]
                        for v in exact.values())
                <= min(v["reduction_depth_over_output_width"]
                       for v in moved.values())),
            "fp32_gemm_also_shape_dependent": dig(
                f32, "gemm", "a_bare_gemm_is_shape_dependent"),
            "bf16_over_fp32_logit_magnitude": dig(
                f32, "ratio_bf16_over_fp32", "A_vs_E"),
            "logits_max_abs_with_bf16_reduced_precision_reduction_on":
                rc.get("max_abs_on"),
            "logits_max_abs_with_it_off": rc.get("max_abs_off"),
            "knob_removes_the_divergence": rc.get("divergence_removed_by_disabling"),
            "collector_reduction_order_drift_at_fixed_activations": dig(
                sd, "reduction_order_drift_at_fixed_activations", "max_abs"),
            "forward_drift_max_abs": dig(sd, "forward_drift", "max_abs"),
            "solo_unpadded_vs_itself": dig(
                r, "stages", "length_sweep", "vary_padding_at_batch_size_1", 0,
                "max_abs"),
        }

    #: The cross-report facts a single report cannot establish.
    per = out["per_report"]
    runtimes = {label: dig(r, "environment", "torch") for label, r in reports.items()}
    out["holds_under_every_runtime_measured"] = (
        len(set(runtimes.values())) > 1
        and all(v["projections_shape_dependent"] for v in per.values()))
    out["runtimes_measured"] = sorted(set(runtimes.values()))
    out["fp32_removes_the_shape_dependence"] = (
        all(v["fp32_gemm_also_shape_dependent"] is False for v in per.values())
        if per and all(v["fp32_gemm_also_shape_dependent"] is not None
                       for v in per.values()) else None)
    out["collector_reduction_order_contributes"] = any(
        (v["collector_reduction_order_drift_at_fixed_activations"] or 0) > 0
        for v in per.values())
    out["bs1_unpadded_is_exact_everywhere"] = all(
        v["solo_unpadded_vs_itself"] == 0.0 for v in per.values()
        if v["solo_unpadded_vs_itself"] is not None)
    out["_reading"] = (
        "a projection whose GEMM is bit-identical across shapes and one whose "
        "GEMM is not, measured side by side on the same weights, is the "
        "mechanism; `fp32_removes_the_shape_dependence` says whether the dtype "
        "is the CAUSE or the SCALE, and "
        "`collector_reduction_order_contributes` rules the accumulator in or out")
    return out


def subsample_sensitivity(reports: dict) -> dict:
    """Does the DECISION claim survive the full mixture, or was 8 items too few?

    The prediction was recorded before this ran, in `fullmix_prediction.json`:
    the relative statistic drift should fall by roughly sqrt(8) ~ 2.8x because
    the sum grows ~8x while its reordering error grows ~sqrt(8), and the
    selection should still move because at 8 items the drift already exceeds the
    minimum cutoff margin by two to four orders of magnitude.

    Pairs are matched by label: `X` against `X_fullmix`. Nothing is compared
    across objects or runtimes.
    """
    out: dict = {"pairs": {}}
    for label in sorted(reports):
        base = f"{label}_fullmix"
        if base not in reports:
            continue
        small, large = reports[label], reports[base]

        def read(r):
            sd = r["stages"].get("statistics_decomposition") or {}
            sel = r["stages"].get("ffn_selection") or {}
            ratio = f"{float(sel.get('headline_keep_ratio', 0.5)):.2f}"
            per_bs = dig(sel, "by_keep_ratio", ratio) or {}
            return {
                "n_items": dig(r, "calibration", "n_items"),
                "n_tokens": sum(dig(r, "calibration", "lengths") or []),
                "ffn_abs_sum_rel_l2_bs4": dig(
                    sd, "by_micro_batch_size", "4", "ffn_abs_sum", "rel_l2"),
                "layers_moved_at_headline_bs4": dig(
                    per_bs, "4", "n_layers_with_moved_selection"),
                "n_layers": dig(per_bs, "4", "n_layers"),
                "min_cutoff_margin_relative_bs4": dig(
                    per_bs, "4", "min_cutoff_margin_relative"),
                "invariant_at_every_ratio_and_size": sel.get(
                    "selection_is_batch_invariant_at_every_ratio_and_size"),
                "by_keep_ratio_bs4": {
                    ratio_k: dig(per, "4", "n_layers_with_moved_selection")
                    for ratio_k, per in (sel.get("by_keep_ratio") or {}).items()},
            }

        a, b = read(small), read(large)
        drift_ratio = (a["ffn_abs_sum_rel_l2_bs4"] / b["ffn_abs_sum_rel_l2_bs4"]
                       if a["ffn_abs_sum_rel_l2_bs4"] and b["ffn_abs_sum_rel_l2_bs4"]
                       else None)
        out["pairs"][label] = {
            "subsample": a,
            "full_mixture": b,
            "drift_fell_by": drift_ratio,
            #: The prediction, as a checkable proposition rather than prose.
            "prediction_drift_falls_2x_to_4x": (
                2.0 <= drift_ratio <= 4.0 if drift_ratio else None),
            "prediction_selection_still_moves": (
                b["invariant_at_every_ratio_and_size"] is False),
            "decision_claim_survives_the_full_mixture": (
                b["layers_moved_at_headline_bs4"] is not None
                and b["layers_moved_at_headline_bs4"] > 0),
        }
    if out["pairs"]:
        out["decision_claim_survives_everywhere_it_was_tested"] = all(
            v["decision_claim_survives_the_full_mixture"]
            for v in out["pairs"].values())
    out["_prediction_record"] = ("logs/stages/stage-1/phase_c3/investigations/"
                                 "batch-invariance-root-cause/v1/"
                                 "fullmix_prediction.json")
    return out


def cross_session(reports: dict, prior: dict) -> dict:
    """Do two SESSIONS on two pods agree, label for label?

    Within-run repeatability is already measured (each shape repeats itself
    bit-identically). This is the stronger claim: a different pod on a different
    host, hours apart, reproducing the same numbers. If it holds, the effect is
    a property of the computation and not of a run -- which is what lets the
    word "deterministic" be used about a divergence.
    """
    if not prior:
        return {"compared": False,
                "why": "no prior session's reports were supplied"}
    out, shared = {}, sorted(set(reports) & set(prior))
    for label in shared:
        a, b = prior[label], reports[label]
        pairs = {
            "case_matrix_A_vs_E_rel_l2": (
                dig(a, "stages", "case_matrix", "comparisons", "A_vs_E", "rel_l2"),
                dig(b, "stages", "case_matrix", "comparisons", "A_vs_E", "rel_l2")),
            "ffn_layers_with_moved_selection": (
                dig(a, "conclusion", "ffn_layers_with_moved_selection"),
                dig(b, "conclusion", "ffn_layers_with_moved_selection")),
            "ffn_abs_sum_rel_l2_bs4": (
                dig(a, "stages", "statistics_decomposition",
                    "by_micro_batch_size", "4", "ffn_abs_sum", "rel_l2"),
                dig(b, "stages", "statistics_decomposition",
                    "by_micro_batch_size", "4", "ffn_abs_sum", "rel_l2")),
            "divergence_locus": (dig(a, "conclusion", "divergence_locus"),
                                 dig(b, "conclusion", "divergence_locus")),
        }
        out[label] = {
            "prior": {k: v[0] for k, v in pairs.items()},
            "this_session": {k: v[1] for k, v in pairs.items()},
            "identical": {k: v[0] == v[1] for k, v in pairs.items()},
            "all_identical": all(v[0] == v[1] for v in pairs.values()),
        }
    return {
        "compared": True,
        "labels_in_both_sessions": shared,
        "per_label": out,
        "every_shared_label_reproduces_exactly": bool(out) and all(
            v["all_identical"] for v in out.values()),
        "_reading": ("exact equality, not a tolerance. Two pods hours apart "
                     "producing the same float means the divergence is a "
                     "property of the computation and not of a run."),
    }


def build(reports: dict, prior: dict | None = None) -> dict:
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
        "executables": _executables(reports),
        "a4_claims_adjudicated": checks,
        "mechanism": mechanism(reports),
        "subsample_sensitivity": subsample_sensitivity(reports),
        "cross_session_reproducibility": cross_session(reports, prior or {}),
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
    ap.add_argument("--prior", nargs="*", default=(),
                    help="an EARLIER session's report directories; matching "
                         "labels are compared for exact equality")
    args = ap.parse_args(argv)

    reports = load_reports(args.reports)
    if not reports:
        say("no reports could be read; refusing to derive a finding from nothing")
        return 4
    doc = build(reports, load_reports(args.prior) if args.prior else None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    say(f"wrote {out}")
    say(f"verdict = {doc['verdict']}  ({len(reports)} report(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
