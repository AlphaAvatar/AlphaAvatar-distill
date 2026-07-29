# PROPOSAL (needs user approval) — Stage 3 start-point ablation: chain vs s1@660 vs init

Status: **APPROVED by maintainer 2026-07-27 (both arms, ≤$9 cap) — EXECUTED and
CLOSED the same day.** Result: 1× L40S, pod `ruib84xvfyieqm`, **$5.82**, both
arms verified 16/16. **Rule 1 fired** (A1 `from_s1` 3.8067, +0.17% — the arm-B
leg was neutral) and **Rule 4 fired** (A2 `from_init` 3.8285, +0.74% — the
warm-up ladder is unnecessary; Stage 3 recovery is now single-stage).
Everything below is the proposal **as registered before the run** and is left
unedited so the pre-registration stays auditable. Outcome, review and caveats:
`logs/experiments/2026-07-27_stage3_start_point_ablation.md`; recipe decision:
`logs/decisions.md` (2026-07-27, "Start-point ablation verdict").

Drafted 2026-07-27 from the `s2_blocks_v1` review
(`logs/experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md`).

## The question

`s2_blocks_v1` is the best student so far (holdout_v1 NLL **3.8003**), but its
lineage is confounded:

```
init ──660 steps (v0, lr 3e-4, FFN+norm)──▶ s1@660 ──660 steps (v0 epochs 3–4,
      attention unfrozen)──▶ armB@660 ──2700 steps (v1, 2.0 epochs)──▶ 3.8003
```

The middle leg is *known* to have been overfitting a spent corpus (that is what
the 2026-07-25 A/B measured). Nothing tells us whether it helped, was neutral,
or left damage the 2700 steps had to spend budget undoing. And nothing tells us
whether the FFN-first warm-up leg is needed at all now that a 22M-token mixture
exists — it was designed when the mixture was 5.4M tokens.

Both questions are answered by re-running **the same final leg** from different
start points. **The expensive arm is already paid for.**

## Arms — identical except `student_path`

| arm | start point | config (sha256/12) | prior steps | total steps to endpoint | status |
|---|---|---|---|---:|---:|---|
| **A0 `chain`** | armB@660 (`s2_blocks_v0`) | `stage3_s2_blocks_v1.json` (`5a61689cb9a8`) | 1320 | 4020 | **done — 3.8003** |
| **A1 `from_s1`** | `s1_ffn_norm_v0`@660 | `stage3_s2v1_from_s1.json` (`7e0612ccf3aa`) | 660 | 3360 | proposed |
| **A2 `from_init`** | Stage 1 init checkpoint | `stage3_s2v1_from_init.json` (`b2520a2e0ad8`) | 0 | 2700 | proposed |

The two new configs are committed and CPU-validated; they differ from the
already-run config in **exactly three fields** — `student_path`, `run_name`,
`out_dir` (verified by diff, not by eye). Everything else is bit-identical:
mixture v1, `extra_val` val_v0, **seed 20260726**, 2700 steps × 16 × 1024-token
blocks, attention-unfrozen freeze set (440.5M trainable; tied embedding frozen),
CE 0.25 + full-vocab KD 1.0 at τ=1 scope `all`, lr 2e-4 / warmup 60 / cosine to
0.1×, fp32 master + bf16 autocast, eval every 150 steps on 64 val blocks.

**The shared seed is the point** (decision record 2026-07-27): all three arms
see the same train-block stream in the same order and are evaluated on the same
64-block val_v1 and val_v0 subsets, so A0's logged curves are directly
comparable with the new arms step-for-step — not just at the endpoint.

## Budget (fixed before the run, AGENTS.md P6)

- **Fixed:** the compared leg — 2700 optimizer steps × 16 × 1024 tokens
  (≈ 44.2M block-tokens ≈ 2.0 epochs of mixture v1) per arm, one run per arm.
- **Deliberately not fixed:** total compute to reach the endpoint (4020 / 3360 /
  2700 steps). This is a start-point comparison, so the prior legs *are* the
  variable; every result table must report the total-steps column above, and a
  tie means the cheaper lineage wins.
- Hardware: **1× L40S 46 GB**, both arms sequential on one pod (peak measured
  37.05 GB, ≥9 GB headroom; no gradient checkpointing needed).
- Wall clock: 2 × 2.32 h training + 2 × ~24 min gate evals + ~40 min setup
  ≈ **6.1 h**.
- Cost: **≈ $6.0–6.5** at $0.99/h; request a **$9 cap** with `--terminate-after`
  as the backstop. Current balance $239.02, limit $80/mo, ~$11 project spend to
  date.
- Cheaper alternative if only one question is worth paying for: **A1 only**,
  ≈ 3.6 h ≈ **$3.6–4.0**. (Running both costs only ~$2.5 more than one, because
  setup — 19% of the last session's spend — is amortized.)

## Pre-registered decision rules

Primary metric: **holdout_v1 bf16 NLL** (fixed 21,080-token file, seed
independent). Band: **1% relative**, the same threshold the 2026-07-25 A/B used.
Let C = 3.8003 (A0), S = A1, I = A2.

1. `|S − C| < 1%` → the arm-B leg was **neutral**. Adopt `from_s1` as the
   canonical lineage (shorter, one fewer confound) and stop chaining new runs
   through checkpoints that overfit their mixture.
2. `S < C − 1%` → the arm-B leg **hurt**. s1@660 becomes the canonical branch
   point; record "do not continue from a run that exhausted its corpus" in the
   recipe.
3. `C < S − 1%` → progressive chaining **helps**; keep the ladder and log the
   measured per-leg benefit for sizing future warm-up legs.
4. `I` within 1% of `min(C, S)` → **the warm-up ladder is unnecessary at this
   data scale.** Future recovery runs start from the Stage 1 init with the full
   freeze set, saving 660–1320 steps and a whole session per iteration. This is
   the highest-value outcome available here (P1: it deletes machinery).
5. `I > min(C, S) + 1%` → the ladder is justified; quantify the benefit per
   extra 660/1320 steps and use it to size the warm-up leg of the next recipe.

Secondary readouts, reported for all three arms (comparable by construction):
val_v1 and val_v0 curves, INT8 fake-quant at both scopes, the same 3-prompt
generation smoke, and — if `eval_behavior_v0` exists by then — its scorecard,
which can be computed for A0 retroactively by scoring the HF checkpoint on CPU.

**Abort rule for A2:** a from-init start at lr 2e-4 / warmup 60 is outside the
regime those hyperparameters were tuned for (s1 used 3e-4 from init). If val_v1
ce at step 300 is above its step-0 value, or any non-finite loss appears, stop
A2, keep the logs, and report it as a negative result rather than retuning
mid-session.

## What this does *not* answer (stated, not hidden)

- **A2 is not "is single-stage optimal?"** It is "does single-stage match the
  ladder *under the ladder's own recipe*". A from-init run might do better with
  its own lr/warmup; that is a separate, later question, and A2 losing does not
  settle it.
- One run per arm: no variance estimate. GPU nondeterminism is P5-logged but
  unmeasured at the holdout level; the 1% band is a judgment call carried over
  from the A/B.
- All metrics here are teacher-forced or general-text. The behavioral defects
  seen in the smoke output are **not** what this ablation is testing.

## Already done on CPU (no cost, committed)

- Both configs written and validated through `validate_train_config`, with the
  three-field diff against the reference run verified programmatically.
- Start-point checkpoints located and sized: A1's is already on the private HF
  relay (`stage3/s1_ffn_norm_v0/step_000660/model`, sha256 pinned in
  `logs/artifact_manifests.md`); A2's is local only
  (`artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`, 1.2 GB).
- Mixture v1 + val_v0 + holdout are already staged on the relay
  (`transfer/stage2_data_20260726.tar.zst`).

## On approval, the session will

1. **CPU prep (dev box, before the pod):** upload the Stage 1 init checkpoint to
   the relay (~30 min at the measured ~680 KB/s), record its per-file sha256 in
   `logs/artifact_manifests.md`, regenerate the repo git bundle at the current
   commit, and refresh `scripts/pod/hashes_transfer.txt` / `hashes_ckpt.txt`.
2. Generalize the pod session scripts to take a run name + config + start
   checkpoint (they are currently hardcoded to `s2_blocks_v1` — the exact
   hardcoded lines are listed in `scripts/pod/AGENTS.md`), then run both arms
   sequentially under the same durable orchestrator that ran the last session
   unattended.
3. Run the standard gate for each arm: bf16 holdout, INT8 fake-quant at both
   scopes, generation smoke, `eval_behavior_v0` if it exists.
4. Upload artifacts to the private HF repo, verify the upload independently,
   write one experiment log covering all three arms, apply the decision rule,
   and update `logs/STATE.md`, `logs/supported_models.md`, and the perf trend.

## Recommendation

Run **both arms**. A2 is the one that can delete a stage from the recipe, and it
is only ~$2.5 more than A1 alone. If the budget is tight, A1 alone still removes
the lineage confound from the current best checkpoint.
