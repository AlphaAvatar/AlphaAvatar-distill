#!/usr/bin/env python3
"""Verify the s2_blocks_v1 HF upload and generate the experiment write-up.

Two subcommands, both used by ``scripts/pod/orchestrate_s2v1.sh``:

``verify``  Independently checks that every expected artifact exists in the
            private HF repo and that its content matches the sha256 computed
            on the pod. Large LFS files are checked via the hub's recorded
            LFS sha256 (no multi-GB download); small files are downloaded and
            hashed locally. Exits non-zero unless every file matches.

``report``  Builds ``logs/experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md``
            from the run's own logs. It reports measured numbers and mechanical
            gate checks only; the stage verdict is left for human/agent review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_REL = "artifacts/stage3/s2_blocks_v1"
LOCAL_RUN = REPO / RUN_REL
HF_REPO = "AlphaAvatar/aadistill-artifacts"
HF_PREFIX = "stage3/s2_blocks_v1"
CKPT_TAG = "step_002700"
HASHFILE = REPO / "artifacts/stage3/s2v1_artifact_hashes_2026-07-26.txt"

RUN_FILES = [
    "train_log.jsonl",
    "run_manifest.json",
    "eval_holdout_v1.json",
    "eval_holdout_v1_int8.json",
    "eval_holdout_v1_int8_decoder.json",
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
    "ab_arm_B (this run's start point)": 4.2118,
}
# INT8 deltas measured on s1@660, 2026-07-26 (CPU fake-quant).
REF_INT8 = {"decoder_scope_pct": 0.03, "full_scope_pct": 0.21}


def load_pod_hashes() -> dict[str, str]:
    """basename -> sha256, from the hash file computed on the pod."""
    if not HASHFILE.exists():
        sys.exit(f"FAIL: pod hash file missing: {HASHFILE}")
    out: dict[str, str] = {}
    for line in HASHFILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, path = line.partition("  ")
        if not path:
            continue
        out[Path(path.strip()).name] = digest.strip()
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_verify() -> int:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    pod_hashes = load_pod_hashes()

    expected = [f"{HF_PREFIX}/{n}" for n in RUN_FILES]
    expected += [f"{HF_PREFIX}/{CKPT_TAG}/model/{n}" for n in MODEL_FILES]
    expected.append(f"{HF_PREFIX}/s2v1_artifact_hashes_2026-07-26.txt")

    tree = {
        item.path: item
        for item in api.list_repo_tree(
            HF_REPO, path_in_repo=HF_PREFIX, recursive=True,
            expand=True, repo_type="model",
        )
        if getattr(item, "size", None) is not None
    }

    problems: list[str] = []
    checked = 0
    print(f"remote files found under {HF_PREFIX}/: {len(tree)}")

    for path in expected:
        item = tree.get(path)
        if item is None:
            problems.append(f"MISSING on HF: {path}")
            continue
        name = Path(path).name
        want = pod_hashes.get(name)
        if want is None:
            # The hash file itself is not listed inside itself.
            if name.startswith("s2v1_artifact_hashes"):
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


def read_events() -> list[dict]:
    path = LOCAL_RUN / "train_log.jsonl"
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


def read_json(name: str) -> dict | None:
    path = LOCAL_RUN / name
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


def cmd_report() -> int:
    ev = read_events()
    by = lambda t: [e for e in ev if e.get("event") == t]  # noqa: E731

    run_start = (by("run_start") or [{}])[0]
    run_end = (by("run_end") or [{}])[0]
    student = (by("student_loaded") or [{}])[0]
    teacher = (by("teacher_loaded") or [{}])[0]
    cfg = (by("config_loaded") or [{}])[0]
    steps = by("train_step")
    evals = by("eval_result")

    manifest = read_json("run_manifest.json") or {}
    bf16 = nll_of(read_json("eval_holdout_v1.json"))
    int8_full = nll_of(read_json("eval_holdout_v1_int8.json"))
    int8_dec = nll_of(read_json("eval_holdout_v1_int8_decoder.json"))
    gen = read_json("gen_smoke.json") or {}

    # Per-step eval table, split by val set (primary is tagged "val").
    sets: dict[str, dict[int, dict]] = {}
    for e in evals:
        sets.setdefault(e.get("val_set", "val"), {})[e.get("step", -1)] = e
    names = sorted(sets, key=lambda n: (n != "val", n))
    all_steps = sorted({s for d in sets.values() for s in d})

    header = "| step | " + " | ".join(
        f"{n} ce | {n} kd" for n in names) + " |"
    sep = "|---" * (1 + 2 * len(names)) + "|"
    rows = []
    for st in all_steps:
        cells = []
        for n in names:
            e = sets[n].get(st)
            cells.append(
                f"{e.get('val_ce'):.6f} | {e.get('val_kd'):.6f}" if e else "– | –"
            )
        rows.append(f"| {st} | " + " | ".join(cells) + " |")

    losses = [s.get("loss") for s in steps if isinstance(s.get("loss"), (int, float))]
    nan = [s["step"] for s in steps if s.get("loss") != s.get("loss")]
    secs = [s.get("seconds") for s in steps if isinstance(s.get("seconds"), (int, float))]
    mems = [
        s.get("gpu_mem_gb") for s in steps
        if isinstance(s.get("gpu_mem_gb"), (int, float))
    ]

    def pct(a: float, b: float) -> str:
        return f"{(a - b) / b * 100:+.2f}%"

    cmp_rows = []
    if bf16 is not None:
        for label, ref in REF.items():
            cmp_rows.append(f"| {label} | {ref:.4f} | {pct(bf16, ref)} |")

    gates = []
    final_ck = run_end.get("steps") == run_start.get("total_steps")
    gates.append(("final step reached", final_ck,
                  f"{run_end.get('steps')}/{run_start.get('total_steps')}"))
    gates.append(("no NaN/Inf training loss", not nan,
                  f"{len(nan)} non-finite loss events"))
    gates.append(("bf16 holdout eval produced", bf16 is not None, str(bf16)))
    gates.append(("INT8 full-scope eval produced", int8_full is not None,
                  str(int8_full)))
    gates.append(("INT8 decoder-scope eval produced", int8_dec is not None,
                  str(int8_dec)))
    gates.append(("generation smoke produced", bool(gen),
                  f"{len(gen)} prompts"))
    if bf16 is not None:
        gates.append(("holdout improves on start point (arm B 4.2118)",
                      bf16 < REF["ab_arm_B (this run's start point)"],
                      f"{bf16:.4f} vs 4.2118 ({pct(bf16, 4.2118)})"))
        gates.append(("holdout improves on s1@660 (4.2107)",
                      bf16 < REF["s1@660"],
                      f"{bf16:.4f} vs 4.2107 ({pct(bf16, 4.2107)})"))

    gate_lines = [
        f"- {'PASS' if ok else 'CHECK'} — {name}: {detail}"
        for name, ok, detail in gates
    ]

    int8_lines = []
    if bf16 and int8_dec:
        int8_lines.append(
            f"- INT8 decoder-scope: {int8_dec:.4f} ({pct(int8_dec, bf16)} vs bf16; "
            f"s1@660 reference was +{REF_INT8['decoder_scope_pct']}%)")
    if bf16 and int8_full:
        int8_lines.append(
            f"- INT8 full-scope: {int8_full:.4f} ({pct(int8_full, bf16)} vs bf16; "
            f"s1@660 reference was +{REF_INT8['full_scope_pct']}%)")

    gen_lines = []
    for prompt, completion in gen.items():
        text = str(completion).replace("\n", "\\n")[:300]
        gen_lines.append(f"- **{prompt}**\n  ```\n  {text}\n  ```")

    hw = manifest.get("hardware", {})
    code = manifest.get("code_state", {})

    doc = f"""# 2026-07-26 — Stage 3 sub-stage 2 on mixture v1: `s2_blocks_v1` (GPU)

> Auto-generated by `scripts/pod/verify_and_report_s2v1.py report` from this
> run's own logs. Numbers are measured. **The stage verdict is deliberately
> left open** — gate interpretation and the decision about README Optim
> record entries need human/agent review.

- **Agent:** Claude Code (Opus 5), autonomous session `stage3-mixture-v1-recovery`.
- **Objective:** First recovery run on the scaled mixture `stage2_offline_v1`,
  using the attention-unfrozen freeze set adopted by the 2026-07-25 A/B, to
  test whether the data scale-up removes the epoch-3–4 overfit signature
  (flat holdout + generation format artifacts).
- **Hypothesis:** With ~2.0 epochs of a 4.11× larger mixture, holdout NLL
  should improve past the A/B plateau (4.2107–4.2118) and the gsm8k/chat-format
  artifacts seen in arm B should recede.
- **Teacher:** `{teacher.get('model_id')}` @ `{str(teacher.get('revision'))[:8]}`,
  {teacher.get('dtype')} ({teacher.get('num_parameters')} params).
- **Student start point:** `{student.get('path')}`
  ({student.get('num_parameters')} params, {student.get('dtype')}) — the
  A/B arm-B final, per the A/B verdict and `logs/STATE.md`.
- **Config:** `{cfg.get('config')}` sha256 `{str(cfg.get('config_sha256'))[:12]}…`
- **Trainable params:** {run_start.get('trainable_params')}
- **Data:** `data/stage2_v1` (mixture v1, 22.13M train tokens), primary val
  = val_v1, `extra_val` val_v0 frozen from mixture v0. Tokenizer sha256
  `{str(manifest.get('tokenizer_sha256'))[:12]}…`
- **Budget (fixed before run):** {run_start.get('total_steps')} steps ×
  16 × 1024-token blocks ≈ 2.0 epochs of v1; eval every 150 steps on 64
  fixed val blocks; single L40S session.
- **Hardware:** {hw.get('platform', 'n/a')}, python {hw.get('python', 'n/a')};
  1× NVIDIA L40S 46 GB (RunPod pod, pod-local disk, no network volume).
- **Code state:** git `{str(code.get('git_commit'))[:12]}`, dirty=
  {code.get('dirty')}, uncommitted_state_sha256
  `{str(code.get('uncommitted_state_sha256'))[:12]}…`

## Training

- Steps completed: **{run_end.get('steps')}** of {run_start.get('total_steps')}
- Wall clock: **{run_end.get('seconds')} s**
  ({(run_end.get('seconds') or 0) / 3600:.2f} h)
- Mean s/step: {sum(secs) / len(secs):.3f} (n={len(secs)})
- Peak GPU memory logged: {max(mems) if mems else 'n/a'} GB
- First / last logged loss: {losses[0] if losses else 'n/a'} /
  {losses[-1] if losses else 'n/a'}
- Non-finite loss events: **{len(nan)}**

### Validation curve

{header}
{sep}
{chr(10).join(rows)}

## holdout_v1 (bf16, 21,080 tokens)

**{bf16 if bf16 is not None else 'n/a'}**

| reference | NLL | this run vs ref |
|---|---|---|
{chr(10).join(cmp_rows)}

## INT8 fake-quant (P9)

{chr(10).join(int8_lines) if int8_lines else '- not produced'}

## Generation smoke (greedy, 80 new tokens, same 3 prompts as s1/A-B)

{chr(10).join(gen_lines) if gen_lines else '- not produced'}

## Mechanical gate checks (AGENTS.md 4.5)

{chr(10).join(gate_lines)}

Checks not mechanically verifiable here and left for review: exact-resume
behaviour (trainer unchanged since the s1 run, where it was verified on GPU),
chat-format discipline in the smoke output above, and whether this result
should become a README Optim record entry.

## Artifacts

- Private HF repo `{HF_REPO}` under `{HF_PREFIX}/`: final fp32 model
  (`{CKPT_TAG}/model/`), `train_log.jsonl`, `run_manifest.json`, the three
  eval JSONs, `gen_smoke.json`, `console.log`, and the pod-side sha256 list.
  Upload independently verified (LFS sha256 for the large weights, download+
  hash for small files) by
  `scripts/pod/verify_and_report_s2v1.py verify`.
- Local (gitignored): same small files under `{RUN_REL}/`. Final weights are
  HF-only by design; optimizer state and rolling checkpoints were not retained.

## Next action

Review the curves and smoke output above, decide the sub-stage 2 verdict,
update `logs/STATE.md` and the perf trend, and decide on README Optim record
entries (maintainer approval required).
"""

    out = REPO / "logs/experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md"
    out.write_text(doc)
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["verify", "report"])
    args = ap.parse_args()
    return cmd_verify() if args.command == "verify" else cmd_report()


if __name__ == "__main__":
    raise SystemExit(main())
