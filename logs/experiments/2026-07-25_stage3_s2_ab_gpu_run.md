# 2026-07-25 — Stage 3 sub-stage 2 sizing A/B: extend-FFN control vs unfreeze-attention (GPU)

- **Agent:** Claude Code (Fable 5), executing the A/B approved by the user on
  2026-07-25 (both arms + private HF artifact repo; see decisions.md
  2026-07-25 and STATE.md).
- **Git commit:** `6230a14` on the pod (A/B configs; logs-only commits
  `1955bf6`+ landed on dev after the pod bundle was cut). Pod tree dirty
  only by the documented cu128 `pyproject.toml` edit
  (uncommitted_state_sha256 `9b8bc39a…3faf7d` in both run manifests).
- **Objective:** Decide the sub-stage 2 freeze set with a controlled
  comparison: does unfreezing attention beat spending the same budget on
  more FFN+norm training?
- **Hypothesis:** s1's un-plateaued val curve is explained by frozen
  attention (sandwich-init, never trained), not by insufficient FFN budget.
- **Teacher:** `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d`, bf16.
- **Student (both arms):** `s1_ffn_norm_v0` step 660 checkpoint
  (`model.safetensors` sha256 `dc64f244…e900`), transferred via the private
  HF artifact repo and **bit-verified on the pod** before training.
- **Stage:** Stage 3, recovery sub-stage 2 (sizing A/B).
- **Hardware:** RunPod secure-cloud pod `simbeepnf8syuu`, 1× L40S 46 GB
  (driver 570.124.06), 16 vCPU, US, $0.99/hr, 120 GB pod-local volume
  (`/dev/md127` — a local md array this run, none of the MooseFS
  stale-read behavior of the previous pod's network volume).
- **Environment:** torch **2.11.0+cu128** (cu128 re-lock as in the s1 run),
  transformers 5.13.1. *Logged deviations:* dev box is torch 2.13.0+cpu /
  Python 3.14; pod resolved **Python 3.12.3**. Bridged as before by running
  the full test suite on the pod: **43/43 passed**. Venv on pod-local disk,
  `HF_HOME=/workspace/hf`.
- **Budget (fixed before run):** per arm 660 steps × 16 × 1024-token blocks
  (≈ 2 epochs of `stage2_offline_v0`, same token budget as s1), single run
  per arm, eval every 110 steps on 64 fixed val blocks, one L40S session,
  cost cap $8. Shared seed 20260725 → identical train-block stream and
  identical eval subset for both arms.

## Arms

| | config (sha256/12) | trainable | peak lr |
|---|---|---|---|
| A control: extend FFN+norm | `stage3_s1_ext.json` (`6f9b17880676`) | 264,299,520 | 2e-4 |
| B treatment: + attention (q/k/v/o + q/k norms) | `stage3_s2_blocks.json` (`762309cbd97a`) | 440,467,456 | 2e-4 |

Identical otherwise: CE 0.25 + full-vocab KD 1.0 (τ=1, scope "all"), warmup
30, cosine→0.1×, fresh AdamW, fp32 master + bf16 autocast.

## Commands (pod, repo at `/workspace/AlphaAvatar-distill`)

```
uv run pytest tests/ -q                                             # 43 passed
uv run python scripts/train_stage3.py --config configs/stage3_s1_ext.json
uv run python scripts/train_stage3.py --config configs/stage3_s2_blocks.json
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
  --model artifacts/stage3/<arm>/checkpoints/step_000660/model --out ...
```

## Results

Stage 2 val (64 blocks, same subset both arms; step 0 identical to all
logged decimals — deterministic eval path confirmed):

| step | A val_ce | A val_kd | B val_ce | B val_kd |
|---|---|---|---|---|
| 0 | 2.602409 | 1.060891 | 2.602409 | 1.060891 |
| 110 | 2.694439 | 1.133627 | 2.817464 | 1.271259 |
| 220 | 2.637685 | 1.083180 | 2.719368 | 1.174381 |
| 330 | 2.619425 | 1.046935 | 2.639717 | 1.086408 |
| 440 | 2.632030 | 1.033165 | 2.595289 | 1.033080 |
| 550 | 2.617960 | 1.013805 | 2.568051 | 0.992625 |
| 660 | 2.633313 | 1.012229 | **2.579108** | **0.986719** |

Both arms show the expected fresh-optimizer/lr-re-warm transient (B's is
larger — 176M never-trained attention params waking); no NaN, no collapse.
A is flat/plateaued; B descends through the whole schedule and separates
cleanly after step 330.

**holdout_v1 NLL (bf16 eval, 21,080 tokens)** vs baselines teacher 2.6264 /
init 11.7482 / s1@660 4.2107:

- arm A: **4.2747** (ppl 71.86) — 1.5% *worse* than s1@660;
- arm B: **4.2118** (ppl 67.48) — flat vs s1@660 (+0.03%).

**Pre-registered decision rule:** B beats A by **1.47% relative ≥ 1%** →
**adopt the attention-unfrozen freeze set** for further Stage 3 recovery.

**Generation smoke** (greedy, 80 new tokens, 3 prompts):

- arm A: s1-like; 2/3 prompts terminate correctly with `<|im_end|>`
  ("Okay, 2+2=<<2+2=4>>4 …"); one degenerate-but-terminating answer.
- arm B: 1/3 clean (water prompt terminates properly; wrong fact, expected
  at this stage); the other two show **format degradation**: stray
  `</think>` openings, gsm8k `<<…>>`/`####` artifacts, and one echo of the
  user turn with a raw `<|im_start|>`. Valid tokens throughout; not a
  collapse, but chat-format discipline regressed vs arm A.

**Interpretation.** This was the model's 3rd–4th epoch over the same 5.39M
train tokens. In-mixture metrics kept improving for B while holdout stayed
flat and format behavior picked up corpus artifacts — a small-corpus
overfit signature. The A/B answered the freeze-set question decisively
(attention-unfrozen strictly dominates under an equal budget: B is better
on val_ce, val_kd, holdout, at +3% step time), but **further recovery is
now data-limited, not freeze-set-limited**. The binding constraint has
moved back to Stage 2 (mixture scale-up, user approval required).

## Throughput / memory

- arm A: 2.878 s/step mean, 1972.6 s train wall-clock, peak 33.69 GB;
- arm B: 2.974 s/step mean (+3.3%), 2039.2 s, peak **36.97 GB** (fits L40S
  46 GB with ≥9 GB headroom; gradient checkpointing unnecessary).

## Gate check (AGENTS.md 4.5)

- reproducible from logged command/config — **yes** (run manifests with
  config/data/tokenizer/teacher hashes + code state, both arms);
- checkpoint resume — not re-tested this session (exact-restore verified on
  GPU in the s1 run; trainer unchanged);
- loss + val proxy logged — **yes** (append-only jsonl per arm);
- no exploding activations / collapse — **yes** (transient bump, monotone
  recovery, no NaN);
- generation smoke valid tokens — **yes**, with arm B format regression
  **documented above**;
- autoregressive behavior improves or failure explained — holdout flat (B)
  / regressed (A); **explained**: mixture exhaustion at epochs 3–4;
- latency/memory — **yes** (s/step + peak VRAM this time);
- quantized eval — still deferred; INT8 eval path is the next CPU-side
  milestone (P9);
- documented — this log.

**Verdict: sizing question ANSWERED (adopt attention-unfrozen freeze set);
sub-stage 2 quality gate NOT advanced — blocked on Stage 2 data budget,
not on the recovery recipe.** Neither arm replaces s1@660 as reference
best; `s2_blocks_v0` final is the preferred starting point for the next
recovery run *after* the mixture is scaled.

## Cost and infrastructure

- Session cost **$2.03** (balance 246.35 → 244.32), ~2.1 h pod time
  including setup and transfers; training 67 min of that. Well under the
  $8 cap.
- **HF-relay transfer worked well**: dev→HF upload of the 2.3 GB fp32
  checkpoint ran at ~680 KB/s (~1 h; ~4× the measured single-stream scp
  rate of ~165 KB/s), pod↔HF both directions fast. All checkpoint moves
  bit-verified by sha256.
- Pod auto-terminate (`--terminate-after` +8 h) used as a cost backstop.

## Artifacts

- HF `AlphaAvatar/aadistill-artifacts` (private), revision `526caa78…db0c`:
  `stage3/s1_ext_v0/` and `stage3/s2_blocks_v0/` (final fp32 model each,
  train_log.jsonl, run_manifest.json, eval_holdout_v1.json, gen_smoke.json,
  console.log) + `stage3/s1_ffn_norm_v0/step_000660/model/` (uploaded
  pre-run, revision `727c837e…14e5`). See `logs/artifact_manifests.md`.
- Local (gitignored): both arms' small files under `artifacts/stage3/<arm>/`
  + `artifacts/stage3/ab_artifact_hashes_2026-07-25.txt` (sha256 of every
  retained file, pod-side). Final model weights are **HF-only** by design
  (2.3 GB each; hashes recorded).
- Not retained: optimizer states (~2.6/3.5 GB), rolling checkpoints
  440/550 (deleted with the pod).

## Next action

1. Propose Stage 2 mixture scale-up (needs user approval: larger download,
   possibly teacher-generated data) — the measured bottleneck.
2. INT8/fake-quant eval path (P9), CPU-suitable, before the next GPU run.
3. Next recovery run: from s1@660 with the adopted freeze set on the scaled
   mixture (fresh data, not more epochs of v0).
