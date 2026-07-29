# 2026-07-22 — Stage 3: recovery trainer implementation + CPU verification

- **Agent:** Claude Code (Fable 5) session, continuing the first dense-model
  compression experiment (teacher `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d`,
  student 0.6B-class init from Stage 1).
- **Git commit:** built on `19e573a` (dirty during the run — this session's
  Stage 3 trainer; committed immediately after, see repo history).
- **Stage:** Stage 3 — student recovery (AGENTS.md 4.5), implementation +
  toy/smoke verification only. **No serious training has run yet.**
- **Objective:** Implement the config-driven recovery trainer (masked CE +
  on-the-fly teacher KD, freeze policies, exact resume, append-only jsonl
  logging), verify it on CPU at toy scale and on the real models at smoke
  scale, and prepare the GPU execution request for the first real run.
- **Hypothesis:** n/a for the implementation itself; the design hypothesis
  (structural init + plain full-vocab KD recovers the student) is tested by
  the upcoming GPU run, not this session.
- **Hardware:** CPU-only dev box (16 threads AMX/AVX-512 BF16, 30 GB RAM).
  All work this session is CPU-suitable per P8.2 (toy tests seconds-scale;
  real-model smoke ≈ 7–38 s per 1024-token step).
- **Budget:** Smoke fixed at 3 optimizer steps + 2-block evals before/after,
  then a 1-step resume run; toy tests bounded by pytest runtime (~3 s).

## Commands

```
uv run pytest tests/ -q                                                  # 43 passed
uv run python scripts/train_stage3.py --config configs/stage3_smoke_cpu.json
uv run python scripts/train_stage3.py --config configs/stage3_smoke_cpu.json --resume step_000002
```

## What was built

- `src/aadistill/train.py` — Stage 3 trainer: `build_blocks` (per-group
  packing via `aadistill.data`), masked next-token CE (fp32 reduction),
  on-the-fly forward-KL KD on the teacher's full-vocab distribution
  (temperature-scaled, position-chunked fp32 softmaxes), regex freeze policy
  (`select_trainable`), linear-warmup + cosine LR, grad accumulation with
  exact microbatch normalization (normalizers precomputed from masks),
  deterministic infinite block stream (epoch permutation = f(seed, epoch),
  position = step × blocks_per_step ⇒ stateless-exact resume), rolling
  checkpoints (`save_pretrained` + optimizer state + config hash; resume
  refuses a changed config), `JsonlLogger` (append-only, AGENTS.md 3.7).
- `scripts/train_stage3.py` — CLI: config-driven, `--resume [TAG]`, refuses
  to overwrite an out_dir with existing checkpoints, verifies teacher/student
  tokenizer hash equality, writes a full run manifest (config hash, data
  manifest hashes, tokenizer hash, teacher identity, code state, hardware).
- `configs/stage3_s1_ffn_norm.json` — recovery sub-stage 1 (FFN+norm
  trainable, attention/embeddings frozen), KD 1.0 (τ=1, scope "all") +
  CE 0.25, fp32 master + bf16 autocast, 660 steps × 16 KiB-token batches
  (~2 epochs of the v0 mixture). GPU-sized draft — not yet run.
- `configs/stage3_smoke_cpu.json` — same structure at 3 steps, batch 1.
- `tests/test_train_toy.py` — 10 tests; design notes in the 2026-07-22
  decision record.

## Verification results

- **Toy tests (43 total, all pass):** CE matches hand computation; KD is 0
  vs itself, positive vs a different teacher, chunk-size invariant;
  the real s1 config's freeze patterns select exactly FFN+norms on a toy
  Qwen3 (attention incl. q_norm/k_norm and tied embedding frozen); LR
  schedule endpoints/monotonicity; block stream is a per-epoch permutation
  and any restart position reproduces the stream slice; 25 toy steps cut
  val CE by >10×… (>10% asserted, observed ≈2×); **resume is bitwise
  exact** (interrupted-at-3-of-6 == uninterrupted-6, params and AdamW
  moments `torch.equal`); resume under a modified config is refused.
- **Real-path CPU smoke (fresh run):** full pipeline — mixture encoding
  (18,484 train samples → 5,247 blocks @ 1024), teacher + student load,
  step-0 baseline eval `val_ce 11.36` / `val_kd 11.50` (consistent with the
  Stage 1 holdout NLL 11.75), 3 KD+CE steps (train loss 17.64 → 14.85,
  6.8–38 s/step), periodic + final checkpoints, final eval, complete jsonl
  event trail (dataset/teacher/student_loaded, config_loaded, run_start,
  eval_result, train_step ×3, checkpoint_saved ×2, run_end).
- **Real-path resume smoke:** `--resume step_000002` reloaded the step-2
  checkpoint in a fresh process and re-ran step 3, reproducing the original
  run's step-3 metrics **identically to all logged decimals** (train loss
  14.8531 / ce 12.9033 / kd 11.6273; final eval val_ce 12.190326 / val_kd
  11.478813) — despite the known oneDNN/AMX two-instance ULP caveat (P5).
  jsonl recorded `resume_loaded`; a resume manifest was written alongside
  the original run manifest.
- Smoke evals use 2 val blocks and 3 steps — they are pipeline evidence
  only, **not** quality claims (val_ce noise between step 0 and 3 is
  expected at this scale).

## Result / verdict

**Trainer implemented and verified at toy + smoke scale.** The Stage 3
*training-run* gate (AGENTS.md 4.5) is intentionally not claimed: no real
recovery run has happened. Ready for the first GPU run pending user
approval (P12): recovery sub-stage 1 under `configs/stage3_s1_ffn_norm.json`.

## Next action

1. User approval + RunPod GPU rental for the s1 run (proposal in STATE.md).
2. GPU smoke (≈10 steps) before the full 660-step run on the same pod.
3. After s1: evaluate vs init baseline on holdout_v1 + stage2 val, decide
   sub-stage 2 (unfreeze attention / full offline KD) sizing.

## Artifacts

- `artifacts/stage3/smoke_cpu_v0/` — gitignored (train_log.jsonl, run
  manifests, rolling checkpoints ~2.4 GB fp32 + optimizer state).
- Console logs: `artifacts/stage3_smoke_console.log`,
  `artifacts/stage3_smoke_resume_console.log` (gitignored).
