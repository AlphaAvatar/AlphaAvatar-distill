"""Mechanically apply the teacher-target 2x2's pre-registered rules.

Pre-registration: logs/proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md §6-7

`verify_and_report.py` is kept for the 2026-07-28 packing control: its rules are
that experiment's (its R1 means "adopt best-fit packing"), and pointing it at
this run would auto-generate a write-up evaluating the wrong hypothesis. Its
`verify` command is generic and is still what checks the uploads.

This reporter reads only artifacts the runs produce and evaluates:

* **R1 — the treatment wins** if it improves *both* p(</think>) and p(<|im_end|>)
  beyond the control's seed spread, *and* holdout NLL stays inside +/-1%.
* **R2 — rejected** if holdout NLL degrades >1%, or `terminated` regresses
  beyond the seed spread. Termination is what the exit gate is blocked on.
* **R3 — inconclusive** if the arms overlap within seed spread on the probes.
* **R4 — abort** is applied per-arm during the run by the orchestrator; this
  reports which arms reached their final step.

Every arm quantity is reported as a **two-seed mean with its spread**, never as a
single run: the measured seed-only floor on `behavior_score_v0` is 0.1290, wider
than any inter-arm difference this project has reported.

Usage:
    uv run python scripts/pod/report_tt2x2.py \
        --control tt2x2_ctrl_a,tt2x2_ctrl_b \
        --treatment tt2x2_treat_a,tt2x2_treat_b \
        --out logs/experiments/stage3/2026-07-30_stage3_teacher_target_2x2.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

GUARD_BAND = 0.01          # R1/R2: +/-1% relative on holdout NLL
BEHAVIOR_NOISE = 0.1290    # measured seed-only floor, 2026-07-28

# Protocol readouts, exactly the pre-registered list.
PROTOCOL_AXES = ("format_ok", "think_closed", "terminated", "empty_answer")


def run_dir(run: str) -> Path:
    return REPO_ROOT / "artifacts" / "stage3" / run


def read_json(run: str, name: str) -> dict | None:
    path = run_dir(run) / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def nll_of(payload: dict | None) -> float | None:
    if not payload:
        return None
    results = payload.get("results") or []
    return results[0].get("mean_nll_nats") if results else None


def protocol_rates(payload: dict | None) -> dict:
    """Per-axis rates over the behavior scorecard's own samples."""
    if not payload or not payload.get("per_sample"):
        return {}
    rows = payload["per_sample"]
    out = {}
    for axis in PROTOCOL_AXES:
        vals = [bool(r[axis]) for r in rows if axis in r]
        if vals:
            out[axis] = sum(vals) / len(vals)
    out["n"] = len(rows)
    return out


def behavior_score_of(payload: dict | None) -> float | None:
    if not payload or not payload.get("per_sample"):
        return None
    from aadistill.evaluation.behavior import behavior_score
    try:
        return float(behavior_score(payload["per_sample"])["score"])
    except (ValueError, KeyError):
        return None


def collect(run: str) -> dict:
    """Every number this report needs from one arm's artifacts."""
    behavior = read_json(run, "eval_behavior_v0.json")
    probe = read_json(run, "probe_think_close.json")
    manifest = read_json(run, "run_manifest.json") or {}
    events = []
    log = run_dir(run) / "train_log.jsonl"
    if log.is_file():
        for line in log.read_text().splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    steps = [e.get("step") for e in events if e.get("event") == "train_step"]
    val = [(e.get("step"), e.get("val_ce")) for e in events
           if e.get("event") == "eval_result" and e.get("val_set", "val") == "val"]
    return {
        "run": run,
        "present": behavior is not None,
        "nll": nll_of(read_json(run, "eval_holdout_v1.json")),
        "nll_int8": nll_of(read_json(run, "eval_holdout_v1_int8.json")),
        "p_close": (probe or {}).get("p_close_mean"),
        "p_im_end": (probe or {}).get("p_im_end_mean"),
        "behavior": behavior_score_of(behavior),
        "protocol": protocol_rates(behavior),
        "last_step": max(steps) if steps else None,
        "val_curve": val,
        "manifest": manifest,
    }


def agg(arms: list[dict], key: str) -> tuple[float | None, float | None, int]:
    """Mean and spread (max-min) over the seeds of one arm."""
    vals = [a[key] for a in arms if a.get(key) is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], None, 1
    return statistics.fmean(vals), max(vals) - min(vals), len(vals)


def fmt(value, spread=None, places=4) -> str:
    if value is None:
        return "—"
    text = f"{value:.{places}f}"
    if spread is not None:
        text += f" ±{spread / 2:.{places}f}"
    return text


def rule_lines(ctrl: list[dict], treat: list[dict]) -> list[str]:
    lines = [
        "Rules registered in "
        "`logs/proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md` §7, "
        "applied mechanically here. Effect sizes are read against the **seed "
        "spread**, not a p-value.",
        "",
    ]

    c_close, c_close_s, _ = agg(ctrl, "p_close")
    t_close, t_close_s, _ = agg(treat, "p_close")
    c_end, c_end_s, _ = agg(ctrl, "p_im_end")
    t_end, t_end_s, _ = agg(treat, "p_im_end")
    c_nll, c_nll_s, _ = agg(ctrl, "nll")
    t_nll, t_nll_s, _ = agg(treat, "nll")

    # The control's own seed spread is the yardstick R1 and R2 are stated against.
    close_band = c_close_s if c_close_s is not None else None
    end_band = c_end_s if c_end_s is not None else None

    def beats(t, c, band):
        if t is None or c is None or band is None:
            return None
        return (t - c) > band

    # --- R1 ------------------------------------------------------------------
    nll_ok = None
    if t_nll is not None and c_nll is not None:
        rel = (t_nll - c_nll) / c_nll
        nll_ok = abs(rel) <= GUARD_BAND
    close_win, end_win = beats(t_close, c_close, close_band), beats(t_end, c_end, end_band)

    if None in (close_win, end_win, nll_ok):
        lines.append(
            "- **R1 not evaluable**: needs both probes and holdout NLL on both "
            f"arms, with >=2 seeds on the control for its spread "
            f"(p_close band={close_band}, p_im_end band={end_band}, "
            f"nll_ok={nll_ok}).")
    else:
        lines.append(
            f"- **R1 — the treatment wins**: p(`</think>`) "
            f"{fmt(t_close)} vs {fmt(c_close)} (control spread "
            f"{fmt(close_band)}) → {'beyond' if close_win else 'inside'} spread; "
            f"p(`<|im_end|>`) {fmt(t_end)} vs {fmt(c_end)} (control spread "
            f"{fmt(end_band)}) → {'beyond' if end_win else 'inside'} spread; "
            f"holdout NLL {fmt(t_nll)} vs {fmt(c_nll)} = "
            f"{(t_nll - c_nll) / c_nll * 100:+.2f}% "
            f"(band ±{GUARD_BAND:.0%}) → {'inside' if nll_ok else 'OUTSIDE'}.")
        if close_win and end_win and nll_ok:
            lines.append("  → **R1 FIRES: the treatment wins.** Both probes "
                         "improve beyond the control's seed spread and the guard "
                         "rail holds. Teacher-native targets are the Stage 3 "
                         "warm-up data path, subject to maintainer approval.")
        else:
            lines.append("  → **R1 does not fire.**")

    # --- R2 ------------------------------------------------------------------
    c_term, c_term_s, _ = agg(
        [{"terminated": a["protocol"].get("terminated")} for a in ctrl], "terminated")
    t_term, _t_term_s, _ = agg(
        [{"terminated": a["protocol"].get("terminated")} for a in treat], "terminated")
    if t_nll is None or c_nll is None:
        lines.append("- **R2 not evaluable**: missing holdout NLL on an arm.")
    else:
        rel = (t_nll - c_nll) / c_nll
        term_regressed = (
            None if (t_term is None or c_term is None or c_term_s is None)
            else (c_term - t_term) > c_term_s
        )
        lines.append(
            f"- **R2 — rejection**: holdout NLL {rel * 100:+.2f}% "
            f"(reject if > +{GUARD_BAND:.0%}); `terminated` {fmt(t_term)} vs "
            f"{fmt(c_term)} (control spread {fmt(c_term_s)}) → "
            f"{'REGRESSED' if term_regressed else 'no regression beyond spread'}.")
        if rel > GUARD_BAND or term_regressed:
            lines.append("  → **R2 FIRES: reject the treatment.** "
                         + ("NLL degraded beyond the band. " if rel > GUARD_BAND else "")
                         + ("`terminated` regressed beyond the control's seed "
                            "spread, which is the metric the exit gate is blocked "
                            "on." if term_regressed else ""))
        else:
            lines.append("  → R2 clear.")

    # --- R3 ------------------------------------------------------------------
    # "Inconclusive" means the arms genuinely OVERLAP within seed spread, so it
    # is |difference| <= spread. Testing `not (treatment > control + spread)`
    # would also fire when the treatment is decisively *worse*, which is a
    # conclusive result reported as an inconclusive one.
    def overlaps(t, c, band):
        if t is None or c is None or band is None:
            return None
        return abs(t - c) <= band

    close_overlap = overlaps(t_close, c_close, close_band)
    end_overlap = overlaps(t_end, c_end, end_band)
    if close_overlap is None or end_overlap is None:
        lines.append("- **R3 not evaluable** (no spread available).")
    elif close_overlap and end_overlap:
        lines.append("- **R3 FIRES: inconclusive.** The arms overlap within seed "
                     "spread on both probes. Recorded as such; per the "
                     "pre-registration a **larger corpus is the lever, not a "
                     "rerun**.")
    else:
        lines.append("- R3 does not fire: the arms are separated by more than the "
                     "seed spread on at least one probe, so the comparison is "
                     "conclusive — in whichever direction R1/R2 report.")

    # --- R4 ------------------------------------------------------------------
    reached = [a["run"] for a in ctrl + treat if a["last_step"] is not None]
    missing = [a["run"] for a in ctrl + treat if a["last_step"] is None]
    lines.append(
        f"- **R4 — abort** is applied per-arm during the run by the orchestrator "
        f"(val CE at the first eval above step 0, or non-finite loss). Arms with "
        f"training logs: {', '.join(reached) or 'none'}"
        + (f"; **no log for {', '.join(missing)}**" if missing else "") + ".")

    # Behavior score is context only, never a primary readout here.
    c_beh, c_beh_s, _ = agg(ctrl, "behavior")
    t_beh, t_beh_s, _ = agg(treat, "behavior")
    if c_beh is not None and t_beh is not None:
        delta = t_beh - c_beh
        lines += [
            "",
            f"**Context, not a decision input:** `behavior_score_v0` "
            f"{fmt(t_beh, t_beh_s)} vs {fmt(c_beh, c_beh_s)}, Δ={delta:+.4f} "
            f"against the measured seed-only noise floor of {BEHAVIOR_NOISE:.4f}. "
            + ("This delta is **inside** the noise floor and supports no ranking."
               if abs(delta) <= BEHAVIOR_NOISE else
               "This delta exceeds the noise floor, but the composite is still "
               "not a pre-registered readout for this experiment."),
        ]
    return lines


def table(ctrl: list[dict], treat: list[dict]) -> list[str]:
    rows = [
        ("holdout NLL (bf16)", "nll", 4),
        ("holdout NLL (INT8)", "nll_int8", 4),
        ("p(`</think>`)", "p_close", 4),
        ("p(`<|im_end|>`)", "p_im_end", 4),
        ("`behavior_score_v0`", "behavior", 4),
    ]
    out = ["| readout | control (2 seeds) | treatment (2 seeds) |",
           "|---|---|---|"]
    for label, key, places in rows:
        c, cs, cn = agg(ctrl, key)
        t, ts, tn = agg(treat, key)
        out.append(f"| {label} | {fmt(c, cs, places)} (n={cn}) | "
                   f"{fmt(t, ts, places)} (n={tn}) |")
    for axis in PROTOCOL_AXES:
        c, cs, cn = agg([{axis: a["protocol"].get(axis)} for a in ctrl], axis)
        t, ts, tn = agg([{axis: a["protocol"].get(axis)} for a in treat], axis)
        out.append(f"| `{axis}` | {fmt(c, cs)} (n={cn}) | {fmt(t, ts)} (n={tn}) |")
    return out


def per_run_table(arms: list[dict]) -> list[str]:
    out = ["| run | last step | holdout NLL | p(</think>) | p(<\\|im_end\\|>) | "
           "behavior | format_ok | terminated |", "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        p = a["protocol"]
        out.append(
            f"| `{a['run']}` | {a['last_step'] or '—'} | {fmt(a['nll'])} | "
            f"{fmt(a['p_close'])} | {fmt(a['p_im_end'])} | {fmt(a['behavior'])} | "
            f"{fmt(p.get('format_ok'))} | {fmt(p.get('terminated'))} |")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, help="comma-separated run names")
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--pilot-manifest", default="data/stage3_pilot/manifest.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ctrl = [collect(r.strip()) for r in args.control.split(",") if r.strip()]
    treat = [collect(r.strip()) for r in args.treatment.split(",") if r.strip()]

    pilot = {}
    pm = REPO_ROOT / args.pilot_manifest
    if pm.is_file():
        pilot = json.loads(pm.read_text())

    body = [
        "# 2026-07-30 — Stage 3 teacher-target SFT warm-up: 2x2 result",
        "",
        "- **Pre-registration:** "
        "[`proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md`]"
        "(../../proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md)",
        "- **Generated mechanically** by `scripts/pod/report_tt2x2.py` from the "
        "runs' own artifacts. Numbers are measured; the rules below are applied "
        "mechanically; the stage verdict is left to review.",
        "",
        "## 1. Corpus",
        "",
    ]
    if pilot:
        body += [
            f"- accepted prompts: **{pilot.get('accepted_prompts')}** "
            f"(dropped {pilot.get('dropped_public_fallback')} public fallbacks)",
            f"- corpus sha256: `{(pilot.get('targets_corpus') or {}).get('sha256', '')[:16]}…`",
            f"- capability scope: {json.dumps(pilot.get('capability_scope', {}))}",
            f"- packing: {json.dumps(pilot.get('packing', {}))}",
            "",
        ]
        for arm in ("control", "treatment"):
            info = (pilot.get("arms") or {}).get(arm, {})
            tok = info.get("tokens") or {}
            body.append(
                f"- **{arm}**: supervised {tok.get('supervised_total')} tokens "
                f"(fraction {tok.get('supervised_fraction')}), lossless="
                f"{tok.get('lossless')}")
        body.append("")
    else:
        body += ["- pilot manifest not found; corpus section unavailable.", ""]

    body += ["## 2. Arm means (two seeds, mean ±half-spread)", ""]
    body += table(ctrl, treat)
    body += ["", "## 3. Per run", ""]
    body += per_run_table(ctrl + treat)
    body += ["", "## 4. Pre-registered decision rules", ""]
    body += rule_lines(ctrl, treat)
    body += [
        "",
        "## 5. Declared asymmetry (P6)",
        "",
        "Total training tokens are **identical** across arms (steps x "
        "blocks_per_step x block_len). Passes over the prompt set and supervised "
        "token counts are **not** equal, because teacher targets are several "
        "times longer than public ones on the same prompts. That is inherent to "
        "the comparison and is reported rather than engineered away (maintainer "
        "decision 2026-07-30). See the corpus section for the measured "
        "supervised-token counts per arm.",
        "",
    ]

    text = "\n".join(body) + "\n"
    print(text)
    if args.out:
        out = REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
