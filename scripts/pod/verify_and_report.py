#!/usr/bin/env python3
"""Verify a run's HF upload and generate the experiment write-up.

Two subcommands, both used by ``scripts/pod/orchestrate.sh``:

``verify --run <name>``
            Independently checks that every expected artifact exists in the
            private HF repo and that its content matches the sha256 computed on
            the pod. Large LFS files are checked via the hub's recorded LFS
            sha256 (no multi-GB download); small files are downloaded and hashed
            locally. Exits non-zero unless every file matches.

``report --run <name>[,<name>...]``
            Builds the experiment write-up from the runs' own logs, including
            the multi-arm comparison table and a mechanical application of the
            ablation's pre-registered decision rules. Measured numbers and rule
            outcomes only; the stage verdict is left for human/agent review.

Nothing here is specific to one run: names, paths and step tags come from the
arguments and from ``run_env.sh``. (The single-run predecessor,
``verify_and_report_s2v1.py``, is in git history at commit f74e5ed.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HF_REPO = "AlphaAvatar/aadistill-artifacts"
# Must match run_env.sh's HF_PREFIX_BASE, which is what post_run.sh uploads
# under. It was a hardcoded "stage3" here while a session could set any prefix,
# so a session that changed it saw all four arms fail verification against a
# path nothing had ever written to — a false negative that, by design, blocks
# teardown and leaves a paid pod running.
HF_PREFIX_BASE = os.environ.get("HF_PREFIX_BASE", "stage3")

RUN_FILES = [
    "train_log.jsonl",
    "run_manifest.json",
    "eval_holdout_v1.json",
    "eval_holdout_v1_int8.json",
    "eval_holdout_v1_int8_decoder.json",
    "eval_behavior_v0.json",
    "eval_behavior_v0.generations.jsonl",
    "gen_smoke.json",
    "console.log",
]
MODEL_FILES = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]

# Reference points, all measured on holdout_v1 (21,080 tokens, bf16).
REF = {
    "teacher": 2.6264,
    "stage1_init": 11.7482,
    "s1@660": 4.2107,
    "ab_arm_A": 4.2747,
    "ab_arm_B": 4.2118,
    "s2_blocks_v1 (A0 chain)": 3.8003,
}
# INT8 deltas measured on s1@660, 2026-07-26 (CPU fake-quant).
REF_INT8 = {"decoder_scope_pct": 0.03, "full_scope_pct": 0.21}

# Pre-registered constants for the CURRENT session
# (retired experiment; see logs/EXPERIMENTS.md).
#
# This block plus `apply_decision_rules` and the doc template in `cmd_report`
# are the session-specific parts of this file; everything else is generic.
A0_CHAIN_NLL = 3.8003          # kept: the REF table is still quoted against it
DECISION_BAND = 0.01           # rule R2 guard rail, 1% relative on holdout NLL
BASELINE_RUN = "s2v1_from_init"          # what the control is measured against
BASELINE_NLL = 3.8285                    # its holdout_v1 bf16 NLL (2026-07-27)
BASELINE_REF_SCORECARD = "s2v1_from_init_step2700"  # re-scored this session
CONTROL_ARM = "s2v1_bl2048"
REPLICATE_ARM = "s2v1_bl2048_seedB"
NOISE_REREAD_THRESHOLD = 0.05  # rule R5
ABLATION_BEHAVIOR = {          # 2026-07-27 ranking, for rule R5's re-read
    "s2v1_from_init": 0.2015, "s1_ffn_norm_v0@660": 0.1290,
    "s2v1_from_s1": 0.0947, "s2_blocks_v1": 0.0891,
}
REPORT_PATH = "logs/experiments/stage3/2026-07-28_stage3_packing_control.md"


def run_dir(run: str) -> Path:
    return REPO / "artifacts/stage3" / run


def hash_file_for(run: str) -> Path:
    matches = sorted((REPO / "artifacts/stage3").glob(f"{run}_artifact_hashes_*.txt"))
    if not matches:
        sys.exit(f"FAIL: no pod hash file for run {run}")
    return matches[-1]


def load_pod_hashes(run: str) -> dict[str, str]:
    """basename -> sha256, from the hash file computed on the pod."""
    path = hash_file_for(run)
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, p = line.partition("  ")
        if not p:
            continue
        out[Path(p.strip()).name] = digest.strip()
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def step_tag_of(run: str) -> str:
    """Final checkpoint tag, read from the run's own manifest/log."""
    ev = read_events(run)
    ends = [e for e in ev if e.get("event") == "run_end"]
    steps = ends[0].get("steps") if ends else None
    if steps is None:
        starts = [e for e in ev if e.get("event") == "run_start"]
        steps = starts[0].get("total_steps") if starts else 0
    return f"step_{int(steps):06d}"


def cmd_verify(run: str) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    pod_hashes = load_pod_hashes(run)
    prefix = f"{HF_PREFIX_BASE}/{run}"
    ckpt_tag = step_tag_of(run)
    hashfile_name = hash_file_for(run).name

    expected = [f"{prefix}/{n}" for n in RUN_FILES]
    expected += [f"{prefix}/{ckpt_tag}/model/{n}" for n in MODEL_FILES]
    expected.append(f"{prefix}/{hashfile_name}")

    tree = {
        item.path: item
        for item in api.list_repo_tree(
            HF_REPO, path_in_repo=prefix, recursive=True,
            expand=True, repo_type="model",
        )
        if getattr(item, "size", None) is not None
    }

    problems: list[str] = []
    checked = 0
    print(f"remote files found under {prefix}/: {len(tree)}")

    for path in expected:
        item = tree.get(path)
        if item is None:
            problems.append(f"MISSING on HF: {path}")
            continue
        name = Path(path).name
        want = pod_hashes.get(name)
        if want is None:
            # The hash file itself is not listed inside itself.
            if name == hashfile_name:
                print(f"  ok (presence+size {item.size}) {path}")
                checked += 1
                continue
            problems.append(f"no pod-side sha256 recorded for {name}")
            continue

        lfs = getattr(item, "lfs", None)
        if isinstance(lfs, dict):
            got = lfs.get("sha256")
        else:
            got = getattr(lfs, "sha256", None) if lfs is not None else None
        how = "lfs-sha256"
        if got is None:
            local = Path(
                hf_hub_download(
                    HF_REPO, path, repo_type="model",
                    cache_dir=str(REPO / "artifacts/stage3/.verify_cache"),
                )
            )
            got = sha256_file(local)
            how = "downloaded"

        if got == want:
            print(f"  ok ({how}, {item.size} B) {path}")
            checked += 1
        else:
            problems.append(f"HASH MISMATCH {path}: hf={got} pod={want}")

    if problems:
        print("\nVERIFICATION FAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    print(f"\nVERIFICATION PASSED: {checked}/{len(expected)} files match pod sha256")
    return 0


def read_events(run: str) -> list[dict]:
    path = run_dir(run) / "train_log.jsonl"
    if not path.exists():
        sys.exit(f"FAIL: {path} missing")
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def read_json(run: str, name: str) -> dict | None:
    path = run_dir(run) / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def read_ref_scorecard(name: str) -> dict | None:
    path = REPO / "artifacts/stage3/reference_scorecards" / f"{name}_behavior_v0.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def nll_of(payload: dict | None) -> float | None:
    if not payload:
        return None
    results = payload.get("results") or []
    if not results:
        return None
    return results[0].get("mean_nll_nats")


def pct(a: float, b: float) -> str:
    return f"{(a - b) / b * 100:+.2f}%"


def arm_section(run: str) -> tuple[str, dict]:
    """Per-arm write-up section plus the numbers the comparison needs."""
    ev = read_events(run)
    by = lambda t: [e for e in ev if e.get("event") == t]  # noqa: E731

    run_start = (by("run_start") or [{}])[0]
    run_end = (by("run_end") or [{}])[0]
    student = (by("student_loaded") or [{}])[0]
    teacher = (by("teacher_loaded") or [{}])[0]
    cfg = (by("config_loaded") or [{}])[0]
    steps = by("train_step")
    evals = by("eval_result")

    manifest = read_json(run, "run_manifest.json") or {}
    bf16 = nll_of(read_json(run, "eval_holdout_v1.json"))
    int8_full = nll_of(read_json(run, "eval_holdout_v1_int8.json"))
    int8_dec = nll_of(read_json(run, "eval_holdout_v1_int8_decoder.json"))
    gen = read_json(run, "gen_smoke.json") or {}
    behavior = read_json(run, "eval_behavior_v0.json") or {}

    sets: dict[str, dict[int, dict]] = {}
    for e in evals:
        sets.setdefault(e.get("val_set", "val"), {})[e.get("step", -1)] = e
    names = sorted(sets, key=lambda n: (n != "val", n))
    all_steps = sorted({s for d in sets.values() for s in d})
    header = "| step | " + " | ".join(f"{n} ce | {n} kd" for n in names) + " |"
    sep = "|---" * (1 + 2 * len(names)) + "|"
    rows = []
    for st in all_steps:
        cells = []
        for n in names:
            e = sets[n].get(st)
            cells.append(
                f"{e.get('val_ce'):.6f} | {e.get('val_kd'):.6f}" if e else "– | –")
        rows.append(f"| {st} | " + " | ".join(cells) + " |")

    losses = [s.get("loss") for s in steps if isinstance(s.get("loss"), (int, float))]
    nan = [s["step"] for s in steps if s.get("loss") != s.get("loss")]
    secs = [s.get("seconds") for s in steps if isinstance(s.get("seconds"), (int, float))]
    mems = [s.get("gpu_mem_gb") for s in steps
            if isinstance(s.get("gpu_mem_gb"), (int, float))]

    int8_lines = []
    if bf16 and int8_dec:
        int8_lines.append(f"- INT8 decoder-scope: {int8_dec:.4f} "
                          f"({pct(int8_dec, bf16)} vs bf16)")
    if bf16 and int8_full:
        int8_lines.append(f"- INT8 full-scope: {int8_full:.4f} "
                          f"({pct(int8_full, bf16)} vs bf16)")

    gen_lines = []
    for prompt, completion in gen.items():
        text = str(completion).replace("\n", "\\n")[:300]
        gen_lines.append(f"- **{prompt}**\n  ```\n  {text}\n  ```")

    o = (behavior.get("aggregate") or {}).get("overall", {})
    code = manifest.get("code_state", {})
    hw = manifest.get("hardware", {})

    section = f"""### Arm `{run}`

- **Start point:** `{student.get('path')}`
  ({student.get('num_parameters')} params, {student.get('dtype')})
- **Config:** `{cfg.get('config')}` sha256 `{str(cfg.get('config_sha256'))[:12]}…`
- **Trainable params:** {run_start.get('trainable_params')}
- **Teacher:** `{teacher.get('model_id')}` @ `{str(teacher.get('revision'))[:8]}`
- **Steps:** {run_end.get('steps')} of {run_start.get('total_steps')};
  wall clock {run_end.get('seconds')} s
  ({(run_end.get('seconds') or 0) / 3600:.2f} h);
  mean {sum(secs) / len(secs):.3f} s/step (n={len(secs)});
  peak GPU mem {max(mems) if mems else 'n/a'} GB
- **Loss first/last:** {losses[0] if losses else 'n/a'} / {losses[-1] if losses else 'n/a'};
  non-finite events: **{len(nan)}**
- **Code state:** git `{str(code.get('git_commit'))[:12]}`, dirty={code.get('dirty')}
- **Hardware:** {hw.get('platform', 'n/a')}, python {hw.get('python', 'n/a')}

#### Validation curve

{header}
{sep}
{chr(10).join(rows)}

#### holdout_v1 (bf16) — **{bf16 if bf16 is not None else 'n/a'}**

{chr(10).join(int8_lines) if int8_lines else '- INT8 not produced'}

#### Generation smoke (greedy, 80 new tokens)

{chr(10).join(gen_lines) if gen_lines else '- not produced'}
"""

    return section, {
        "run": run,
        "bf16": bf16,
        "int8_full": int8_full,
        "int8_dec": int8_dec,
        "behavior": o,
        "steps_done": run_end.get("steps"),
        "total_steps": run_start.get("total_steps"),
        "nan": len(nan),
        "start_point": student.get("path"),
    }


BEHAVIOR_COLS = [
    ("format_ok", "format_ok"),
    ("terminated", "terminated"),
    ("truncated_at_cap", "trunc@cap"),
    ("think_closed", "think_closed"),
    ("empty_answer", "empty"),
    ("answer_is_echo", "echo"),
    ("rep_3gram", "rep3"),
    ("answer_words", "words"),
]


def behavior_table(rows: list[tuple[str, dict]]) -> str:
    head = "| checkpoint | " + " | ".join(lbl for _, lbl in BEHAVIOR_COLS) + " |"
    sep = "|---" * (1 + len(BEHAVIOR_COLS)) + "|"
    out = [head, sep]
    for name, agg in rows:
        cells = []
        for key, _ in BEHAVIOR_COLS:
            v = agg.get(key)
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "–")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def score_of_payload(payload: dict | None) -> float | None:
    """`behavior_score_v0` from a scorecard JSON.

    The headline metric is *derived* from `per_sample`, not stored in the
    scorecard (src/aadistill/evaluation/behavior.py:behavior_score), so it is recomputed
    here rather than read out of `aggregate`.
    """
    if not payload or not payload.get("per_sample"):
        return None
    sys.path.insert(0, str(REPO / "src"))
    from aadistill.evaluation.behavior import behavior_score

    try:
        return float(behavior_score(payload["per_sample"])["score"])
    except (ValueError, KeyError):
        return None


def apply_decision_rules(results: dict[str, dict]) -> list[str]:
    """Mechanically apply the pre-registered rules. Interpretation stays human."""
    A = results.get(CONTROL_ARM) or {}
    B = results.get(REPLICATE_ARM) or {}
    a_beh = score_of_payload(read_json(CONTROL_ARM, "eval_behavior_v0.json"))
    b_beh = score_of_payload(read_json(REPLICATE_ARM, "eval_behavior_v0.json"))
    a_nll, b_nll = A.get("bf16"), B.get("bf16")

    base_beh = score_of_payload(read_ref_scorecard(BASELINE_REF_SCORECARD))

    lines = [
        "Rules registered in "
        "`logs/proposals/stage3/2026-07-28_stage3_packing_blocklen_control.md` §6, "
        "applied mechanically here.",
        "",
        f"- Baseline `{BASELINE_RUN}@2700`: holdout **{BASELINE_NLL:.4f}** "
        f"(2026-07-27), behavior **"
        + (f"{base_beh:.4f}" if base_beh is not None else "not re-scored")
        + "** (re-scored on this pod, same device).",
    ]

    # --- R0: the noise floor -------------------------------------------------
    noise = None
    if a_beh is None or b_beh is None:
        lines.append("- **R0 not evaluable**: both arms must produce a behavior "
                     "scorecard to estimate the noise floor "
                     f"(A={a_beh}, B={b_beh}).")
    else:
        noise = abs(a_beh - b_beh)
        lines.append(f"- **R0 — noise floor**: |A − B| = |{a_beh:.4f} − "
                     f"{b_beh:.4f}| = **{noise:.4f}** on `behavior_score_v0`. "
                     "This is the project's first run-to-run variance estimate; "
                     "the arms differ only in seed (data order, packing order "
                     "and the val subset).")
        if a_nll is not None and b_nll is not None:
            lines.append(f"  Holdout NLL noise: |{a_nll:.4f} − {b_nll:.4f}| = "
                         f"**{abs(a_nll - b_nll):.4f}** "
                         f"({abs(a_nll - b_nll) / b_nll * 100:.2f}% relative).")

    # --- R1: adoption --------------------------------------------------------
    if a_beh is None or base_beh is None or noise is None:
        lines.append("- **R1 not evaluable**: needs the control's behavior score, "
                     "the re-scored baseline, and the R0 noise floor.")
    else:
        delta = a_beh - base_beh
        lines.append(f"- **R1 — adoption**: Δbehavior = {a_beh:.4f} − "
                     f"{base_beh:.4f} = **{delta:+.4f}** against a noise floor of "
                     f"{noise:.4f}.")
        if delta > noise:
            lines.append("  → **R1 fires: adopt** best-fit packing at "
                         "`block_len` 2048 as the default Stage 3 data path. The "
                         "improvement exceeds run-to-run variance.")
        else:
            lines.append("  → **R1 does not fire**: the delta does not exceed the "
                         "noise floor. The baseline data path stands; record the "
                         "control as neutral-or-negative. Note this does *not* "
                         "say packing is harmless — it says the effect is not "
                         "resolvable at this sample size.")

    # --- R2: guard rail ------------------------------------------------------
    if a_nll is None:
        lines.append("- **R2 not evaluable**: the control produced no holdout NLL.")
    else:
        rel = (a_nll - BASELINE_NLL) / BASELINE_NLL
        lines.append(f"- **R2 — guard rail**: holdout {a_nll:.4f} vs baseline "
                     f"{BASELINE_NLL:.4f} = **{rel * 100:+.2f}%** "
                     f"(band ±{DECISION_BAND:.0%}).")
        if rel > DECISION_BAND:
            lines.append("  → **R2 fires**: NLL regressed beyond the band. Do not "
                         "adopt on behavior alone — report the tradeoff and "
                         "escalate the decision to the maintainer.")
        else:
            lines.append("  → R2 clear: inside the guard-rail band.")

    # --- R5: re-read the ablation -------------------------------------------
    if noise is None:
        lines.append("- **R5 not evaluable** (no noise floor).")
    elif noise > NOISE_REREAD_THRESHOLD:
        ranked = " / ".join(f"{k} {v:.4f}" for k, v in ABLATION_BEHAVIOR.items())
        lines.append(f"- **R5 fires**: noise {noise:.4f} exceeds "
                     f"{NOISE_REREAD_THRESHOLD:.2f}. The 2026-07-27 ablation's "
                     f"behavior ranking ({ranked}) must be re-read with this band "
                     "attached, and the \"single-stage is best-behaved\" "
                     "conclusion re-stated with the appropriate confidence.")
    else:
        lines.append(f"- R5 clear: noise {noise:.4f} ≤ "
                     f"{NOISE_REREAD_THRESHOLD:.2f}, so the ablation's behavior "
                     "ranking survives at its reported spacing.")

    lines += [
        "",
        "**R3 (the stated mechanism) and R4 (abort) are judged by a human/agent "
        "from the per-group tables below and the orchestrator log** — R3 asks "
        "whether grounding and multi-hop specifically improved, which the "
        "aggregate cannot answer.",
        "",
        "**Budget check (P6):** both arms ran 2,700 steps × 8 blocks × 2,048 "
        "tokens = 44,236,800 tokens, identical to the baseline's 2,700 × 16 × "
        "1,024. Declared asymmetry: the control sees ~7.7% fewer *supervised* "
        "tokens (10,787,265 vs 11,681,472) because best-fit truncates oversized "
        "samples and pads; the direction is conservative.",
    ]
    return lines


def cmd_report(runs: list[str]) -> int:
    sections, results = [], {}
    for run in runs:
        section, res = arm_section(run)
        sections.append(section)
        results[run] = res

    # holdout comparison across every reference point plus this session's arms.
    cmp_rows = ["| checkpoint | holdout_v1 NLL (bf16) | vs A0 chain (3.8003) |",
                "|---|---:|---:|"]
    for label, ref in REF.items():
        cmp_rows.append(f"| {label} | {ref:.4f} | {pct(ref, A0_CHAIN_NLL)} |")
    for run, res in results.items():
        if res["bf16"] is not None:
            cmp_rows.append(f"| **{run}** | **{res['bf16']:.4f}** | "
                            f"{pct(res['bf16'], A0_CHAIN_NLL)} |")

    # behavior comparison: references (GPU-scored this session) + arms.
    beh_rows = []
    for ref_name, label in [
        (BASELINE_REF_SCORECARD, f"{BASELINE_RUN}@2700 (baseline, re-scored here)"),
    ]:
        card = read_ref_scorecard(ref_name)
        if card:
            beh_rows.append((label, (card.get("aggregate") or {}).get("overall", {})))
    for run, res in results.items():
        if res["behavior"]:
            beh_rows.append((f"**{run}**", res["behavior"]))

    gates = []
    for run, res in results.items():
        gates.append((f"[{run}] final step reached",
                      res["steps_done"] == res["total_steps"],
                      f"{res['steps_done']}/{res['total_steps']}"))
        gates.append((f"[{run}] no NaN/Inf training loss", res["nan"] == 0,
                      f"{res['nan']} non-finite loss events"))
        gates.append((f"[{run}] bf16 holdout produced", res["bf16"] is not None,
                      str(res["bf16"])))
        gates.append((f"[{run}] INT8 both scopes produced",
                      res["int8_full"] is not None and res["int8_dec"] is not None,
                      f"full={res['int8_full']} decoder={res['int8_dec']}"))
        gates.append((f"[{run}] behavior scorecard produced", bool(res["behavior"]),
                      f"{res['behavior'].get('n', 0)} prompts"))
    gate_lines = [f"- {'PASS' if ok else 'CHECK'} — {name}: {detail}"
                  for name, ok, detail in gates]

    doc = f"""# 2026-07-28 — Stage 3 packing / `block_len` control + first run-to-run variance (GPU)

> Auto-generated by `scripts/pod/verify_and_report.py report` from the runs'
> own logs. Numbers are measured and the pre-registered decision rules are
> applied mechanically. **The stage verdict and any README Optim record entry
> are deliberately left for human/agent review.**

- **Agent:** Claude Code (Opus 5), autonomous session `packing_control`.
- **Objective:** Re-run the `{BASELINE_RUN}` recipe with **only the data path
  changed** — best-fit packing at `block_len` 2048 instead of
  concatenate-then-cut at 1024 — so that later experiments (teacher traces
  above all) are not confounded by "samples are no longer torn". The second arm
  repeats the first at a different seed and is the project's **first run-to-run
  variance measurement**; until now no "win" had a noise floor to be read
  against.
- **Proposal:** `logs/proposals/stage3/2026-07-28_stage3_packing_blocklen_control.md`
  (arms, budget and decision rules registered before the run).
- **Arms this session:** {', '.join(f'`{r}`' for r in runs)}.
  `{CONTROL_ARM}` is the control (seed 20260726); `{REPLICATE_ARM}` is the
  identical config at seed 20260728.
- **Budget match (P6):** 2,700 steps × 8 blocks × 2,048 tokens = 44,236,800
  tokens per arm, identical to the baseline's 2,700 × 16 × 1,024, at 2.01 vs
  2.00 epochs over the mixture.
- **Comparability:** the seed differs *between arms by design* (that is the
  variance being measured), so the in-training primary val is not comparable
  across them — `holdout_v1` (per-sample, `--max-seq-len` 1024) and
  `eval_behavior_v0` are. The baseline was re-scored on this pod because
  behavior scorecards are only comparable within one device (decision record
  2026-07-27).

## Holdout comparison (primary metric)

{chr(10).join(cmp_rows)}

## Pre-registered decision rules

{chr(10).join(apply_decision_rules(results))}

## Behavior scorecard (`eval_behavior_v0`, 76 held-out prompts)

All rows below were generated on the **same GPU, dtype and code** in this
session, so they are directly comparable. Mechanical scorers only; see
`src/aadistill/evaluation/behavior.py` for the think-block contract and the echo-credit
rule.

{behavior_table(beh_rows) if beh_rows else '- not produced'}

Lower is better for `trunc@cap`, `empty`, `echo` and `rep3`; higher is better
for `format_ok`, `terminated` and `think_closed`.

## Mechanical gate checks (AGENTS.md 4.5)

{chr(10).join(gate_lines)}

## Per-arm detail

{chr(10).join(sections)}

## Artifacts

Private HF repo `{HF_REPO}` under `{HF_PREFIX_BASE}/<arm>/`: final fp32 model,
`train_log.jsonl`, `run_manifest.json`, the three holdout eval JSONs, the
behavior scorecard + raw generations, `gen_smoke.json`, `console.log`, and the
pod-side sha256 list. Each upload independently verified (LFS sha256 for the
weights, download+hash for small files) by
`scripts/pod/verify_and_report.py verify`. Reference scorecards are under
`{HF_PREFIX_BASE}/reference_scorecards/`.

## Next action

Review the curves, apply the fired decision rule to the recipe, update
`logs/STATE.md`, `logs/supported_models.md` and the perf trend, and decide on
README Optim record entries (maintainer approval required).
"""

    out = REPO / REPORT_PATH
    out.write_text(doc)
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["verify", "report"])
    ap.add_argument("--run", required=True,
                    help="run name; for `report`, a comma-separated list")
    args = ap.parse_args()
    if args.command == "verify":
        return cmd_verify(args.run)
    return cmd_report([r for r in args.run.split(",") if r])


if __name__ == "__main__":
    raise SystemExit(main())
