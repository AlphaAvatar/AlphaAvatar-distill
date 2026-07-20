# Current project state

Updated: 2026-07-21 (UTC+8 dev box) — Stage 2 gate-passed (offline mixture v0
built and dry-run verified).

## Status

First dense-model compression experiment, teacher **Qwen/Qwen3-4B-Thinking-2507**
@ `768f209d` (hidden 2560, 36 layers, FFN 9728, 32Q/8KV).

User decisions (2026-07-13, see decisions.md): student target **0.6B-class**
(hidden 1024, 28 layers, FFN 3072, 16Q/8KV, tied emb — Qwen3-0.6B geometry);
**BF16 training, INT8 deployment target**; warm-up v1 public-corpus download approved.

Pipeline position: **Stage 0 passed → Stage 1 passed → Stage 2 passed → next
is Stage 3** (student recovery). New decisions this session (2026-07-21, see
decisions.md): mixture composition with no teacher-generated data in v0
(on-the-fly KD planned), and assistant-span loss masking with empty-think
targets (the Thinking-2507 chat template is not prefix-stable).

Verified state:

- initialized student checkpoint `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`
  (596.0M params, bf16); holdout NLL: teacher 2.63 / init 11.75 / random 12.13.
- Stage 2 offline mixture `stage2_offline_v0`: 8 groups per AGENTS.md 4.4,
  18,484 train samples / 5.39M tokens (2.39M trainable), 771 val, 120 calib
  (stratified calib = INT8 calibration set). Loader dry run passed
  (deterministic encoding, tool-call rendering, 5,256 blocks @ 1024).

## Environment

- CPU-only dev box: 16 threads (AMX/AVX-512 BF16), 30 GB RAM, no GPU.
- `uv sync`: Python 3.14, torch 2.13.0+cpu, transformers 5.13.1, safetensors 0.8.0,
  datasets, pytest.
- Known nondeterminism (logged per P5): two model instances with bitwise-identical
  bf16 weights can differ by a few ULPs in logits (oneDNN/AMX alignment-dependent);
  each instance is self-deterministic.
- RunPod control plane verified read-only 2026-07-16: `runpodctl` 2.7.1
  authenticated, balance $250 / $80 spend limit, no pods or volumes, SSH keys
  present. Skill at `.agents/skills/runpodctl`. Ready to rent a GPU worker for
  Stage 3 pending user approval (P12).
- HF cache ~12 GB (7.6 GB teacher + Stage 2 source datasets).

## What exists and why

- `src/aadistill/` — `env.py`, `manifest.py`, `teacher.py`, `collect.py`
  (Stage 0), `project.py`, `sandwich.py`, `student.py` (Stage 1), and new
  this session: `data.py` — Stage 2+ loader (schema validation, chat-template
  rendering, assistant-span loss masks via fast-tokenizer offsets, block
  packing). This is the path the Stage 3 trainer will consume.
- `scripts/` — Stage 0/1: `collect_stage0.py`, `build_warmup_v1.py`,
  `build_holdout_v1.py`, `init_stage1.py`, `eval_ppl.py`, `plot_perf_trend.py`.
  New: `build_stage2_v0.py` (mixture builder), `dry_run_stage2.py` (gate check).
- `configs/` — Stage 0 v0/v1 + Stage 1 init configs (Stage 2 sources are
  declared in the builder script and pinned in the mixture manifest).
- `data/warmup/` — Stage 0 corpora + manifests (jsonl gitignored).
- `data/stage2/` — `train/ val/ calib/` per-group jsonl (gitignored,
  rebuildable) + committed `stage2_offline_v0.manifest.json` (revisions,
  licenses, hashes, split rule, dedup rule, holdout exclusion).
- `tests/` — `test_collect_toy.py` (8), `test_stage1_toy.py` (6),
  `test_data_toy.py` (19, incl. real-tokenizer loss-mask checks). All 33 pass.
- `artifacts/` (gitignored) — Stage 0 stats cache, Stage 1 checkpoint +
  eval, `stage2/dry_run_v0_report.json` (gate evidence).
- `logs/` — decisions (6 records), experiments (4), supported_models, this file.

## Latest known working commands

```
uv run pytest tests/ -q                                                # 33 passed
uv run python scripts/build_stage2_v0.py                               # rebuild mixture (network)
uv run python scripts/dry_run_stage2.py                                # Stage 2 gate check (~12 s)
uv run python scripts/collect_stage0.py --config configs/stage0_qwen3_4b_thinking_v1.json
uv run python scripts/init_stage1.py --config configs/stage1_qwen3_0p6b_from_4b_thinking.json
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl --model <dir-or-hf-id> ...
```

## Latest verification

- Stage 2 gate: passed 2026-07-21 — mixture manifest committed, loader dry
  run all-checks-true (report in `artifacts/stage2/`), 33/33 tests pass.
  Caveat logged: "loads in the intended training pipeline" was verified
  against the real loader module; the Stage 3 trainer itself doesn't exist yet.
- Stage 1 gate: passed 2026-07-14. Stage 0 v1 gate: passed 2026-07-13.

## Not done yet (next, in order)

1. **Stage 3 — recovery design + trainer**: recovery sub-stages per AGENTS.md
   4.5 (FFN/norm → block → student-forced span → full offline KD/SFT),
   on-the-fly teacher KD loss (decision 2026-07-21), CPU toy training loop +
   resume tests first. Serious training needs user approval + GPU rental
   (RunPod ready; hardware sizing to be proposed with the Stage 3 design).
2. Optional Stage 2 upgrades (need approval where noted): scaled-up mixture,
   teacher-generated corpora (approval: paid/long teacher inference),
   mixture-ratio tuning once Stage 3 loss mix is defined.
3. Stage 1 ablation backlog (optional): function-aware width subspace,
   per-group P with Procrustes chaining, activation-based head importance.

## Open decisions for the user

- None blocking Stage 3 *implementation* (toy loop is CPU-suitable).
  Blocking Stage 3 *training runs*: approval + GPU plan (a concrete RunPod
  proposal will accompany the trainer once the toy path is verified).
  Blocking Stage 2 scale-up: approval for large downloads / teacher-generated
  corpora.

## Links

- `logs/experiments/2026-07-21_stage2_offline_v0.md` (this session)
- `logs/experiments/2026-07-14_stage1_qwen3_0p6b_init_v0.md`
- `logs/experiments/2026-07-13_stage0_qwen3_4b_thinking_v1.md`
- `logs/decisions.md` (6 records; 2 added 2026-07-21)
- `logs/supported_models.md`
