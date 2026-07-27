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
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HF_REPO = "AlphaAvatar/aadistill-artifacts"
HF_PREFIX_BASE = "stage3"

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

# Pre-registered ablation constants
# (logs/proposals/2026-07-27_stage3_start_point_ablation.md).
A0_CHAIN_NLL = 3.8003
DECISION_BAND = 0.01  # 1% relative
TOTAL_STEPS_TO_ENDPOINT = {
    "s2_blocks_v1": 4020,
    "s2v1_from_s1": 3360,
    "s2v1_from_init": 2700,
}


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


def apply_decision_rules(results: dict[str, dict]) -> list[str]:
    """Mechanically apply the pre-registered rules. Interpretation stays human."""
    C = A0_CHAIN_NLL
    S = (results.get("s2v1_from_s1") or {}).get("bf16")
    I = (results.get("s2v1_from_init") or {}).get("bf16")
    lines = [f"Pre-registered band: **{DECISION_BAND:.0%} relative** on holdout_v1 "
             f"bf16 NLL. C (A0 `s2_blocks_v1`) = {C:.4f}."]

    if S is None:
        lines.append("- Rules 1–3 **not evaluable**: arm `s2v1_from_s1` produced no holdout number.")
    else:
        rel = (S - C) / C
        lines.append(f"- S (`s2v1_from_s1`) = {S:.4f} → S vs C = {rel * 100:+.2f}%")
        if abs(rel) < DECISION_BAND:
            lines.append("  → **Rule 1 fires**: the A/B arm-B leg was *neutral*. "
                         "Adopt `from_s1` as the canonical lineage (shorter, one "
                         "fewer confound); stop chaining through checkpoints that "
                         "overfit their mixture.")
        elif rel < -DECISION_BAND:
            lines.append("  → **Rule 2 fires**: the arm-B leg *hurt*. s1@660 becomes "
                         "the canonical branch point; record \"do not continue from a "
                         "run that exhausted its corpus\" in the recipe.")
        else:
            lines.append("  → **Rule 3 fires**: progressive chaining *helps*; keep the "
                         "ladder and log the measured per-leg benefit.")

    if I is None:
        lines.append("- Rules 4–5 **not evaluable**: arm `s2v1_from_init` produced no holdout number.")
    elif S is not None:
        best = min(C, S)
        rel_i = (I - best) / best
        lines.append(f"- I (`s2v1_from_init`) = {I:.4f} → I vs min(C,S)={best:.4f} "
                     f"= {rel_i * 100:+.2f}%")
        if rel_i <= DECISION_BAND:
            lines.append("  → **Rule 4 fires**: the warm-up ladder is *unnecessary at "
                         "this data scale*. Future recovery runs start from the Stage 1 "
                         "init with the full freeze set, saving 660–1320 steps and a "
                         "session per iteration (P1: this deletes machinery).")
        else:
            lines.append("  → **Rule 5 fires**: the ladder is *justified*; quantify the "
                         "benefit per extra 660/1320 steps and size the next recipe's "
                         "warm-up leg accordingly.")

    lines.append("")
    lines.append("**Total compute to endpoint (deliberately not fixed — this is a "
                 "start-point comparison, so a tie means the cheaper lineage wins):**")
    lines.append("")
    lines.append("| arm | total steps to endpoint |")
    lines.append("|---|---:|")
    for name, steps in TOTAL_STEPS_TO_ENDPOINT.items():
        lines.append(f"| `{name}` | {steps} |")
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
        ("s1_ffn_norm_v0_step660", "s1@660 (reference)"),
        ("s2_blocks_v1_step2700", "s2_blocks_v1 @2700 (A0 chain)"),
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

    doc = f"""# 2026-07-27 — Stage 3 start-point ablation: chain vs s1@660 vs init (GPU)

> Auto-generated by `scripts/pod/verify_and_report.py report` from the runs'
> own logs. Numbers are measured and the pre-registered decision rules are
> applied mechanically. **The stage verdict and any README Optim record entry
> are deliberately left for human/agent review.**

- **Agent:** Claude Code (Opus 5), autonomous session `{"start_point_ablation"}`.
- **Objective:** Re-run the identical 2700-step mixture-v1 leg from different
  start points to (a) remove the lineage confound from the current best
  checkpoint and (b) test whether the FFN-first warm-up ladder is still needed
  at the 22M-token data scale.
- **Proposal:** `logs/proposals/2026-07-27_stage3_start_point_ablation.md`
  (arms, budget and decision rules registered before the run).
- **Arms this session:** {', '.join(f'`{r}`' for r in runs)}.
  The third arm, `s2_blocks_v1` (A0 `chain`), was already paid for on 2026-07-26.
- **Shared seed `20260726`** across all arms — required for comparability, since
  the 64-block val subset is a permutation of `cfg["seed"] + 777`
  (`src/aadistill/train.py:332`; decision record 2026-07-27).

## Holdout comparison (primary metric)

{chr(10).join(cmp_rows)}

## Pre-registered decision rules

{chr(10).join(apply_decision_rules(results))}

## Behavior scorecard (`eval_behavior_v0`, 76 held-out prompts)

All rows below were generated on the **same GPU, dtype and code** in this
session, so they are directly comparable. Mechanical scorers only; see
`src/aadistill/behavior.py` for the think-block contract and the echo-credit
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

    out = REPO / "logs/experiments/2026-07-27_stage3_start_point_ablation.md"
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
