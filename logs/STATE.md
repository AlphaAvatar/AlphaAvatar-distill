# Current project state

Updated: 2026-07-26 (UTC+8 dev box) — **Stage 2 mixture v1 built and gate
passed**: `stage2_offline_v1` = 22.13M train tokens (4.11× v0), approved
same day (incl. the gated xlam add-on, click-through accepted on the
AlphaAvatar HF account). Trainer now supports named secondary val sets
(`extra_val`) so future runs log frozen val_v0 next to val_v1. **Next
recovery run is fully prepared** (`configs/stage3_s2_blocks_v1.json`) and
awaits per-session GPU approval.

## Status

First dense-model compression experiment, teacher **Qwen/Qwen3-4B-Thinking-2507**
@ `768f209d` (hidden 2560, 36 layers, FFN 9728, 32Q/8KV) → student 0.6B-class
(hidden 1024, 28 layers, FFN 3072, 16Q/8KV, tied emb). BF16 training,
INT8 deployment target.

Pipeline position: **Stage 0 passed → Stage 1 passed → Stage 2 passed
(v0 2026-07-21, v1 scale-up 2026-07-26) → Stage 3 s1 passed (2026-07-22) →
s2 sizing A/B done (2026-07-25, attention-unfrozen freeze set adopted)**.
The data bottleneck identified by the A/B is resolved; the next recovery
run is blocked only on GPU-session approval.

## Verified state (all on the real model/data)

- **Stage 2 mixture v1 (2026-07-26, CPU):** 64,484 train samples /
  22,133,631 train tokens / 21,610 blocks@1024 (trainable frac 0.528);
  val_v1 1,916 samples; calib 200 (v0's 120 frozen + 80 new). v0 val/calib
  remain frozen in `data/stage2/`. Gate passed (dry-run report
  `artifacts/stage2/dry_run_v1_report.json`, all checks true; 69/69 tests;
  3-step CPU trainer smoke over v1 + extra_val). Committed manifest:
  `data/stage2_v1/stage2_offline_v1.manifest.json`. Log:
  `logs/experiments/2026-07-26_stage2_offline_v1.md`. Note: train tokens
  landed 8% under the ~24M target (logged deviation) → next run sized
  2,700 steps ≈ 2.0 epochs.
- **Trainer extra_val (2026-07-26):** named secondary val sets, per-set
  `eval_result` events (`val_set` field; primary = "val"); config-validated,
  unit-tested, exercised by the CPU smoke. Old logs/configs unaffected.
- **INT8 fake-quant eval (2026-07-26, CPU):** holdout_v1 on s1@660 — bf16
  4.2111, int8 decoder-only +0.03%, int8 full-scope +0.21%. Run at every
  future recovery gate. Log:
  `logs/experiments/2026-07-26_int8_fakequant_eval.md`.
- **Sub-stage 2 A/B (2026-07-25, 1× L40S):** arm B (attention unfrozen,
  440.5M trainable) beat arm A by 1.47% holdout (4.2118 vs 4.2747; s1@660
  ref 4.2107); pre-registered rule fired → freeze set adopted. Both arms
  data-limited (mixture epochs 3–4). Log:
  `logs/experiments/2026-07-25_stage3_s2_ab_gpu_run.md`.
- **Stage 3 s1 (2026-07-22, 1× L40S):** val_ce 12.009 → 2.1805, holdout
  NLL 4.2107 vs init 11.75; generation smoke passed; exact-resume verified.
  Log: `logs/experiments/2026-07-22_stage3_s1_gpu_run.md`.
- Stage 1 init checkpoint `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`
  (596.0M params); Stage 0 stats caches.
- Tests: **69/69** pass on the dev box (torch 2.13.0+cpu); 43 of the
  pre-quant/pre-v1 subset also passed on the 2026-07-25 GPU pod
  (torch 2.11.0+cu128).

## Environment

- CPU-only dev box: 16 threads (AMX/AVX-512 BF16), 30 GB RAM, no GPU.
  `uv sync`: Python 3.14, torch 2.13.0+cpu, transformers 5.13.1.
- GPU runs: RunPod (runpodctl 2.7.1 authenticated; skill at
  `.agents/skills/runpodctl`). Balance **$244.32** (~$6.4 total project GPU
  spend), $80 spend limit. **No pods or volumes currently exist.**
- HF: dev box logged in as `AlphaAvatar` (write token); private artifact
  repo `AlphaAvatar/aadistill-artifacts` holds s1@660 + both A/B finals
  (see `logs/artifact_manifests.md`). Gated
  `Salesforce/xlam-function-calling-60k` accepted on this account
  (2026-07-26; HF gates need browser click-through, not API tokens).
- Pod playbook (see experiment logs §infrastructure): venv on pod-local
  disk (`UV_PROJECT_ENVIRONMENT=/root/venv`); torch cu128 (2.11.0) max for
  driver 570; big-file transfer via the private HF repo relay; always
  sha256-verify after transfer; `--terminate-after` as cost backstop.
- Known CPU nondeterminism (oneDNN/AMX ULP-level, P5-logged) unchanged.
- HF cache ~12.3 GB (teacher 7.6 GB + Stage 2 sources; v1 build streamed
  the large new sources, so cache growth was only ~0.3 GB).

## What exists and why

- `src/aadistill/` — env, manifest, teacher, collect (S0), project, sandwich,
  student (S1), data (S2 loader), train (S3 recovery trainer + extra_val),
  quant (INT8 fake-quant eval, P9).
- `scripts/` — stage scripts (`build_stage2_v0.py` untouched,
  `build_stage2_v1.py` new), `train_stage3.py` (extra_val wired),
  `dry_run_stage2.py` (any mixture dir), `eval_ppl.py`,
  `plot_perf_trend.py`.
- `configs/` — Stage 0/1 configs; stage3: `s1_ffn_norm` (ran), smoke
  configs (ran), `s1_ext`+`s2_blocks` (A/B, ran),
  `s2v1_smoke_cpu.json` (ran 2026-07-26),
  **`s2_blocks_v1.json` (next run, not yet run)**.
- `data/stage2/` — frozen v0 (val_v0 + calib-120 + original train) +
  manifest; `data/stage2_v1/` — v1 mixture + manifest (jsonl gitignored).
- `tests/` — 69 tests.
- `logs/` — decisions (9), experiments (8), proposals (1, approved),
  supported_models, artifact_manifests, this file.
- `artifacts/` (gitignored) — stage0 stats, stage1 checkpoint, stage3 runs
  (s1 final retained locally + HF; A/B finals HF-only), stage2 dry-run
  reports + v1 build console log, `s2v1_smoke_cpu/` (disposable smoke).
- `assets/` — perf trend json + svg (5 attempt points).

## Latest known working commands

```
uv run pytest tests/ -q                                          # 69 passed
uv run python scripts/build_stage2_v1.py                         # rebuild v1
uv run python scripts/dry_run_stage2.py --data-dir data/stage2_v1 \
  --out artifacts/stage2/dry_run_v1_report.json
uv run python scripts/train_stage3.py --config configs/stage3_s2v1_smoke_cpu.json
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
  --model artifacts/stage3/s1_ffn_norm_v0/checkpoints/step_000660/model \
  [--fake-quant int8 [--fake-quant-scope decoder]]
```

## Not done yet (next, in order)

1. **GPU approval for the next recovery run** (`s2_blocks_v1`, config
   committed): 1× L40S, 2,700 steps × 16×1024 (≈ 2.0 epochs of v1),
   attention-unfrozen freeze set, start from `s2_blocks_v0` final (pull
   from HF, verify vs `ab_artifact_hashes_2026-07-25.txt`), extra_val
   val_v0; est. ~2.5 h train + overhead ≈ **$3–5** (`--terminate-after`
   backstop). Gate: holdout_v1 + val_v0/val_v1 curves + generation smoke
   (expect artifact regressions gone) + INT8 fake-quant eval.
2. After the run: update perf trend, decide whether s1/A-B/this become
   official README Optim record entries (maintainer approval).
3. Stage 4 online data collection design.
4. Optional backlog: Stage 1 ablations; teacher-generated data proposal
   (conversational rewrites for QA groups + reasoning traces).

## Open decisions for the user

- ~~GPU session approval for `s2_blocks_v1`~~ — **approved 2026-07-26**
  ("continue" after the request in-session); session in progress.
- Whether s1 and/or the A/B become official README "Optim record history"
  entries (reproducible records exist; needs maintainer approval).

## Links

- `logs/experiments/2026-07-26_stage2_offline_v1.md` (this session)
- `logs/proposals/2026-07-26_stage2_mixture_v1_scaleup.md` (approved)
- `logs/experiments/2026-07-26_int8_fakequant_eval.md`
- `logs/experiments/2026-07-25_stage3_s2_ab_gpu_run.md`
- `logs/experiments/2026-07-22_stage3_s1_gpu_run.md`
- `logs/experiments/2026-07-21_stage2_offline_v0.md`
- `logs/experiments/2026-07-14_stage1_qwen3_0p6b_init_v0.md`
- `logs/experiments/2026-07-13_stage0_qwen3_4b_thinking_v1.md`
- `logs/decisions.md` · `logs/supported_models.md` ·
  `logs/artifact_manifests.md`
