# Current project state

Updated: 2026-07-27 (UTC+8 dev box) — **Stage 3 sub-stage 2 gate passed on
mixture v1.** The `s2_blocks_v1` run (2026-07-26, 1× L40S, $4.49) broke the
two-run holdout plateau: **4.2118 → 3.8003 NLL (−9.8%)**, 26% of the remaining
gap to the teacher closed, INT8 robustness unchanged. It is the project's new
best checkpoint. The data-limited diagnosis is confirmed and retired; the next
constraint is **target style**, and the next two experiments (start-point
ablation, teacher-generated answers) are written up as proposals awaiting
approval. No GPU work is running or billing.

## Status

First dense-model compression experiment, teacher **Qwen/Qwen3-4B-Thinking-2507**
@ `768f209d` (hidden 2560, 36 layers, FFN 9728, 32Q/8KV) → student 0.6B-class
(hidden 1024, 28 layers, FFN 3072, 16Q/8KV, tied emb). BF16 training,
INT8 deployment target.

Pipeline position: **Stage 0 passed → Stage 1 passed → Stage 2 passed
(v0 2026-07-21, v1 scale-up 2026-07-26) → Stage 3 s1 passed (2026-07-22) →
s2 sizing A/B done (2026-07-25, freeze set adopted) → s2 quality gate passed
(2026-07-26 on mixture v1)**. Stage 4 not started.

Repo: branch `main`, working tree clean at the documentation commit on top of
`b9f3958` (run verdict + comparability rules, two proposals, two ablation
configs, manifests, regenerated perf trend, README rework).

## Verified state (all on the real model/data)

- **Stage 3 `s2_blocks_v1` (2026-07-26, 1× L40S, 2700 steps ≈ 2.0 epochs of
  mixture v1):** holdout_v1 **3.8003** (bf16; teacher 2.6264, s1@660 4.2107,
  start point armB 4.2118); val_v1 ce 2.7627 → 1.7898; **val_v0 ce 2.6751 →
  2.4374 on frozen data the start point had already trained on for 3–4 epochs**;
  INT8 +0.08% (decoder) / +0.21% (full scope); 0 non-finite losses; 2.98 s/step,
  peak 37.05 GB. Generation smoke: v0-corpus artifacts gone, **question-echo and
  a stray `</think>` remain**. Log + review:
  `logs/experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md`.
- **Sub-stage 2 A/B (2026-07-25, 1× L40S):** attention-unfrozen freeze set
  adopted (440.5M trainable, beat the FFN-only control by 1.47% holdout).
  Log: `logs/experiments/2026-07-25_stage3_s2_ab_gpu_run.md`.
- **Stage 3 s1 (2026-07-22, 1× L40S):** val_ce 12.009 → 2.1805, holdout 4.2107
  vs init 11.75; exact-resume verified on GPU.
  Log: `logs/experiments/2026-07-22_stage3_s1_gpu_run.md`.
- **Stage 2 mixture v1 (2026-07-26, CPU):** 64,484 train samples / 22,133,631
  train tokens / 21,610 blocks@1024; val_v1 1,916; calib 200. v0 val/calib
  frozen in `data/stage2/`. Manifest committed at
  `data/stage2_v1/stage2_offline_v1.manifest.json`.
- **INT8 fake-quant eval path (2026-07-26, CPU/GPU):** run at every recovery
  gate. Log: `logs/experiments/2026-07-26_int8_fakequant_eval.md`.
- Stage 1 init checkpoint `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`
  (596.0M params, 1.2 GB); Stage 0 stats caches.
- Tests: **75/75** on the dev box (torch 2.13.0+cpu); 63 passed / 6 skipped on
  the 2026-07-26 GPU pod (torch 2.11.0+cu128, Python 3.12.3).
- New configs `stage3_s2v1_from_s1.json` / `stage3_s2v1_from_init.json`
  validated on CPU; each differs from the run-verified `stage3_s2_blocks_v1.json`
  in exactly three fields (`student_path`, `run_name`, `out_dir`).

## Two comparability rules now in force (decision record 2026-07-27)

1. **Pin the seed across compared runs.** The 64-block val subset is a
   permutation of `cfg["seed"] + 777` (`src/aadistill/train.py:332`), so
   *val numbers from runs with different seeds are not comparable* — the A/B
   (seed 20260725) and `s2_blocks_v1` (seed 20260726) evaluated different
   val_v0 subsets. Only within-run deltas were usable there. Seed **20260726**
   is pinned for the whole start-point ablation family.
2. **holdout_v1 is a language-modeling metric, not a behavior metric.** It is
   fineweb-edu web text and is nearly blind to chat format, grounding, refusal,
   and tool-call validity — the defects actually observed. `eval_behavior_v0`
   (next action 1) exists to close that gap.

## Environment

- CPU-only dev box: 16 threads (AMX/AVX-512 BF16), 30 GB RAM, no GPU.
  `uv sync`: Python 3.14, torch 2.13.0+cpu, transformers 5.13.1.
- GPU runs: RunPod (runpodctl 2.7.1 authenticated; skill at
  `.agents/skills/runpodctl`). Balance **$239.02** (~$11.7 total project GPU
  spend), $80 spend limit. **No pods or volumes exist; nothing is billing.**
- Pod playbook: create with `--ports "22/tcp,8888/http"`, judge readiness via
  GraphQL `pod.runtime` (never the CLI `uptimeSeconds`), pod-local disk, L40S,
  venv at `/root/venv`, cu128 torch 2.11.0 (driver 570 ceiling), big files via
  the private HF repo relay with sha256 verification, `--terminate-after` as the
  cost backstop. Durable orchestration: `scripts/pod/orchestrate_s2v1.sh` ran the
  last session end-to-end unattended (train → gates → upload → verify → write-up
  → commit/push → teardown). **Setup was 19% of that session's spend** — reason
  to batch multiple arms onto one pod.
- HF: dev box logged in as `AlphaAvatar` (write token); private artifact repo
  `AlphaAvatar/aadistill-artifacts` holds s1@660, both A/B finals, and the
  `s2_blocks_v1` final (see `logs/artifact_manifests.md`).
- Known CPU nondeterminism (oneDNN/AMX ULP-level, P5-logged) unchanged; GPU
  run-to-run variance at the holdout level remains unmeasured.
- HF cache ~12.3 GB.

## What exists and why

- `src/aadistill/` — env, manifest, teacher, collect (S0), project, sandwich,
  student (S1), data (S2 loader), train (S3 recovery trainer + extra_val),
  quant (INT8 fake-quant eval, P9).
- `scripts/` — stage scripts, `train_stage3.py`, `dry_run_stage2.py`,
  `eval_ppl.py`, `plot_perf_trend.py`, and `scripts/pod/` (GPU session
  scripts — currently **hardcoded to the `s2_blocks_v1` run**; see its
  `AGENTS.md` for the exact lines to parameterize for the next session).
- `configs/` — Stage 0/1; stage3: `s1_ffn_norm`, `s1_ext`, `s2_blocks`,
  `s2_blocks_v1` (all ran), smoke configs, and **`s2v1_from_s1` /
  `s2v1_from_init` (next run, not yet run)**.
- `data/stage2/` — frozen v0 (val + calib + original train); `data/stage2_v1/` —
  v1 mixture (jsonl gitignored, manifests committed).
- `tests/` — 75 tests. `logs/` — decisions (11), experiments (8), proposals (3),
  supported_models, artifact_manifests, this file.
- `artifacts/` (gitignored) — stage0 stats, stage1 checkpoint, stage3 run
  artifacts (small files local; final weights HF-only except s1), stage2
  dry-run reports.
- `assets/` — perf trend json + svg (6 attempt points).

## Latest known working commands

```
uv run pytest tests/ -q                                          # 75 passed
uv run python scripts/dry_run_stage2.py --data-dir data/stage2_v1 \
  --out artifacts/stage2/dry_run_v1_report.json
uv run python scripts/train_stage3.py --config configs/stage3_s2v1_smoke_cpu.json
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
  --model artifacts/stage3/s1_ffn_norm_v0/checkpoints/step_000660/model \
  [--fake-quant int8 [--fake-quant-scope decoder]]
uv run python scripts/plot_perf_trend.py [--print-table]         # figure / README table
```

## Next actions (ordered)

1. **Build `eval_behavior_v0` — CPU, free, no approval, do this first.** It is
   the prerequisite for reading the teacher-generated-answer experiment and the
   only way to see the defects that are currently eyeballed. Spec:
   - ~60–100 fixed prompts sampled deterministically from the **val** splits of
     all 8 groups (never trained on), committed as a small jsonl + manifest.
   - Deterministic greedy generation, fixed `max_new_tokens`, teacher chat
     template, same rendering path as training.
   - **Mechanical scorers only** (no LLM judge, so it stays free and
     reproducible): chat-format validity (terminates with `<|im_end|>`, exactly
     one closed think block, no stray `<|im_start|>`/template markers),
     question-echo rate (n-gram overlap with the prompt), degeneracy (3-gram
     repetition), evidence containment for `rag_evidence`/`multihop_qa` (gold
     span present), refusal rate on `refusal_uncertainty`, tool-call JSON parse
     + schema validity, gsm8k final-answer exact match on a small slice.
   - Output: one scorecard JSON per checkpoint + the raw generations. Run at
     every recovery gate next to holdout_v1 and the INT8 evals.
   - First numbers to produce: `s2_blocks_v1`@2700 (download from HF) and
     `s1_ffn_norm_v0`@660 (local) — a before-picture for everything below.
2. **Decide the start-point ablation** (`logs/proposals/2026-07-27_stage3_start_point_ablation.md`):
   1× L40S, arms `from_s1` and `from_init` at the identical 2700-step budget and
   seed as the completed run, ≈ **$6.0–6.5** for both (≈$3.6–4.0 for `from_s1`
   alone), $9 cap. Answers whether the A/B leg helped and whether the warm-up
   ladder is needed at all at this data scale. Configs are committed and
   CPU-validated; decision rules are pre-registered.
3. **Decide the teacher-generated-answer proposal**
   (`logs/proposals/2026-07-27_stage2_teacher_generated_answers.md`): phase-1
   rewrite of 18.3k train targets in 3 groups, transformers path, **$2–4**
   (cheapest attached to the tail of an approved session), plus a $4–5
   comparison run later. Gated on action 1.
4. Pod-session prep once (2) is approved: upload the Stage 1 init checkpoint to
   the HF relay (~30 min), refresh the repo bundle + transfer hashes, and
   parameterize `scripts/pod/*` by run name/config/start checkpoint.
5. Stage 4 online data collection design (unchanged, still after Stage 3).
6. Optional backlog: Stage 1 ablations; measure GPU run-to-run variance once a
   spare session exists (would justify or shrink the 1% decision band).

## Open decisions for the user

- **GPU session approval for the start-point ablation** (proposal above) — both
  arms recommended; `from_s1` alone is the budget option.
- **Teacher-generated answers, phase 1** (proposal above) — recommended *after*
  `eval_behavior_v0` exists.
- **README "Optim record history" entries.** Three Stage 3 runs now have
  reproducible records (s1, the A/B, `s2_blocks_v1`). AGENTS.md 3.8/P12 require
  maintainer approval before any of them appears as an official record; nothing
  has been added. Say the word and the Stage 3 record entry gets written from
  the existing logs.

## Links

- `logs/experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md` (latest run + review)
- `logs/proposals/2026-07-27_stage3_start_point_ablation.md` (awaiting approval)
- `logs/proposals/2026-07-27_stage2_teacher_generated_answers.md` (awaiting approval)
- `logs/experiments/2026-07-25_stage3_s2_ab_gpu_run.md` ·
  `logs/experiments/2026-07-22_stage3_s1_gpu_run.md`
- `logs/experiments/2026-07-26_stage2_offline_v1.md` ·
  `logs/experiments/2026-07-26_int8_fakequant_eval.md`
- `logs/experiments/2026-07-14_stage1_qwen3_0p6b_init_v0.md` ·
  `logs/experiments/2026-07-13_stage0_qwen3_4b_thinking_v1.md`
- `logs/decisions.md` · `logs/supported_models.md` · `logs/artifact_manifests.md`
- `scripts/pod/AGENTS.md` (GPU session playbook)
