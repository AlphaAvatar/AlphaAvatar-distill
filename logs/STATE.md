# Current project state

Updated: 2026-07-28 (UTC+8 dev box) — **Start-point ablation complete; the
Stage 3 recovery recipe is now single-stage.** Both pre-registered rules fired:
the A/B arm-B leg was neutral (+0.17%) and the **FFN-first warm-up ladder is
unnecessary** — a single-stage run from the Stage 1 init reaches the chain's
quality with 33% fewer steps. A new behavior gate, `eval_behavior_v0`, was built
first and **reversed the ranking the primary metric produced**: the cheapest
lineage is the best-behaved one. No GPU work is running or billing.

2026-07-28, maintainer directives (three): **the headline metric is no longer
held-out NLL** — it is `behavior_score_v0` (six credited mechanical axes on 76
held-out prompts; NLL stays as a ±1% guard rail), and the README figure is now
**one point per student at its current best** against its teacher, ARC-style.
Real-world test suites take over as the headline once the student can attempt
them. Current best: `s2v1_from_init@2700` at **20.2%**; the teacher was scored
on 2026-07-28 at **74.4%**, so the figure finally has both ends of the gap
([log](experiments/2026-07-28_teacher_behavior_v0.md)).

Third directive, same day: **the teacher is never forced out of thinking mode.**
Evaluation and generation judge it at its actual capability, so the teacher
scorecard runs with no prefill and a 4096-token cap (≈1 h GPU, ~$1–1.5), and
teacher-generated candidates cost ~1.65k tokens instead of ~150 — which moves the
bulk-corpus estimate from $6–11 to **$25–145**. Scope is cut by lowering n or
dropping slices, never by suppressing reasoning (decision record 2026-07-28).

The next work is a **Stage 3 supplementary
experiment** (recovery run from `s2v1_from_init@2700` on rewritten targets, with
the corpus build as its Stage 2 prerequisite), and its teacher targets must be
**top-n sampled and verified correct** — n candidates per prompt, every
candidate checked against a gold key, one correct candidate selected, otherwise
the public target stays. The proposal is rewritten around that (the two
verifiable math slices move *in*; a ~$2 top-n pilot gates the bulk spend). See
decision record 2026-07-28 and next actions below.

## Status

First dense-model compression experiment, teacher **Qwen/Qwen3-4B-Thinking-2507**
@ `768f209d` (hidden 2560, 36 layers, FFN 9728, 32Q/8KV) → student 0.6B-class
(hidden 1024, 28 layers, FFN 3072, 16Q/8KV, tied emb). BF16 training,
INT8 deployment target.

**This run is a baseline instance, not the target** (decisions 2026-07-28). Two
standing rules follow from that: (a) **no README Optim record entries during
baseline construction** — the first record point is written only after the
baseline reaches Stage 6 with satisfactory results, and it will be the first
entry rather than a backfill; (b) the algorithm core stays
**model-family-agnostic** — the methods are meant to carry to MoE, VLM and
Omni-models, so family specifics belong in recipes and new core code fails
loudly on an unsupported architecture. Neither is a claim of support: no
non-dense, non-text model has been attempted. The porting surface is inventoried
in the 2026-07-28 scope decision.

Pipeline position: **Stage 0 → Stage 1 → Stage 2 (v0 + v1 scale-up) → Stage 3
s1 → s2 sizing A/B → s2 quality gate → start-point ablation (2026-07-27)**, all
passed. Stage 4 not started.

Repo: branch `main`, clean at the 2026-07-28 documentation commits on top of
`30c066c`. Those rebuilt the README figure twice — first as a lineage view, then
(current) as the **leaderboard-style view the maintainer asked for**: one point
per student at its current best against its teacher, y = behavior score, x =
parameters. They also rewrote the teacher-target proposal for the directives
above and recorded the record/scope/metric policies. No experiment data changed;
the behavior scores are recomputed from scorecards that already existed.

## The recipe, as it now stands

**Stage 3 recovery is one run, not a ladder** (decision record 2026-07-27):
`configs/stage3_s2v1_from_init.json` — start from the Stage 1 init, 2700 steps
× 16 × 1024-token blocks on mixture v1, attention-unfrozen freeze set (440.5M
trainable, tied embedding frozen), CE 0.25 + full-vocab KD 1.0 at τ=1 scope
`all`, lr 2e-4 / warmup 60 / cosine to 0.1×, fp32 master + bf16 autocast,
**seed 20260726**, eval every 150 steps on 64 val blocks.
The `s1_ffn_norm` → `s2_blocks` warm-up legs are **retired** for this
architecture and data scale.

## Verified state (all on the real model/data)

- **Start-point ablation (2026-07-27, 1× L40S, $5.82, both arms verified
  16/16):**

  | | s1@660 | A0 `chain` | A1 `from_s1` | A2 `from_init` |
  |---|---|---|---|---|
  | total steps | 660 | 4020 | 3360 | **2700** |
  | holdout_v1 NLL | 4.2107 | **3.8003** | 3.8067 | 3.8285 |
  | INT8 decoder / full | +0.03/+0.21% | +0.08/+0.21% | +0.13/+0.29% | +0.20/+0.33% |
  | format_ok | 0.105 | 0.066 | 0.145 | **0.224** |
  | think_closed | 0.263 | 0.316 | 0.513 | **0.605** |
  | empty_answer | 0.605 | 0.382 | 0.211 | **0.171** |
  | tool_call_parsed | 0.083 | 0.000 | 0.000 | **0.250** |
  | rag evidence_hit *credited* | 0.167 | 0.000 | 0.083 | **0.333** |

  Log + review: `logs/experiments/2026-07-27_stage3_start_point_ablation.md`.
- **`eval_behavior_v0` (2026-07-27, CPU + GPU):** 76 held-out prompts over 7
  chat groups, mechanical scorers only, committed at `data/eval_behavior_v0/`.
  Run at every recovery gate next to holdout_v1 and the INT8 evals. Build log
  (including the echo-credit and truncation findings):
  `logs/experiments/2026-07-27_eval_behavior_v0.md`.
- **Stage 3 `s2_blocks_v1` (2026-07-26):** holdout 3.8003, the best
  language-modeling result; retains that title, but is **last** on every
  behavior axis. Log: `logs/experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md`.
- **Stage 2 mixture v1 (2026-07-26, CPU):** 64,484 train samples / 22,133,631
  train tokens / 21,610 blocks@1024; val_v1 1,916; calib 200.
- Tests: **100/100** on the dev box (torch 2.13.0+cpu); the 2026-07-27 GPU pod ran
  the then-current suite at 85 passed / 6 skipped (torch 2.11.0+cu128,
  Python 3.12.3).

## Best checkpoint — the two metrics disagree, deliberately

- **Best on the headline metric (and the recommended branch point):**
  `stage3/s2v1_from_init/step_002700/model` — `behavior_score_v0` **20.2%**,
  best on every behavior axis, 33% cheaper to produce, holdout 3.8285 (+0.74%,
  inside the guard-rail band).
- **Best held-out NLL:** `stage3/s2_blocks_v1/step_002700/model` (3.8003) —
  and **worst** on behavior (8.9%). NLL is now the guard rail, so this is a
  diagnostic, not the leader.
- Other measured checkpoints: s1@660 **12.9%** / 4.2107 · `s2v1_from_s1`
  **9.5%** / 3.8067. The A/B arms and the two Stage 1 inits were never scored on
  the behavior eval.

## Three comparability rules now in force

1. **Pin the seed across compared runs.** The 64-block val subset is a
   permutation of `cfg["seed"] + 777` (`src/aadistill/train.py:332`). Seed
   **20260726** is pinned for this whole family of runs.
2. **holdout_v1 is a guard rail, not the headline** (decision 2026-07-28). It
   is fineweb-edu web text, nearly blind to chat format, grounding, refusal and
   tool-call validity, and it ranked the ablation arms in the *opposite* order
   to the behavior eval. Band ±1%; a large drop is an abort signal.
3. **Behavior scorecards are only comparable within one device** (decision
   record 2026-07-27). The same checkpoint scored on CPU vs L40S moved
   `format_ok` by 1–3 prompts, though `terminated`/`truncated_at_cap` matched
   exactly. This is why `scripts/pod/score_refs.sh` re-scores reference
   checkpoints on the pod rather than reusing dev-box baselines.

## Environment

- CPU-only dev box: 16 threads (AMX/AVX-512 BF16), 30 GB RAM, no GPU.
  `uv sync`: Python 3.14, torch 2.13.0+cpu, transformers 5.13.1.
  **Note:** a `VIRTUAL_ENV=/home/ecs-user/AlphaAvatar/.venv` (py3.11, torch
  2.9+cu128, transformers 4.57) may be active in the shell; it is a *different
  project's* env and is incompatible with this repo's transformers-v5 API.
  Always run through `uv run`.
- GPU runs: RunPod (runpodctl 2.7.1 authenticated; skill at
  `.agents/skills/runpodctl`). Balance **$233.17** (~$17.5 total project GPU
  spend), $80 spend limit. **No pods or volumes exist; nothing is billing.**
- Pod playbook: `scripts/pod/AGENTS.md`. The scripts are now **parameterized by
  `run_env.sh`** and drive multi-arm sessions unattended; arm-scoped markers,
  sha256 verification at every hop, pre-registered abort checks, and pod
  deletion only after upload verification. Setup was **1.2%** of the last
  session's spend (down from 19%) because two arms shared one pod.
- HF: dev box logged in as `AlphaAvatar`; private artifact repo
  `AlphaAvatar/aadistill-artifacts` holds s1@660, both A/B finals,
  `s2_blocks_v1`, both ablation arms, the Stage 1 init, and the reference
  behavior scorecards (see `logs/artifact_manifests.md`).
- GPU run-to-run variance at the holdout level remains **unmeasured**.

## What exists and why

- `src/aadistill/` — env, manifest, teacher, collect (S0), project, sandwich,
  student (S1), data (S2 loader), train (S3 recovery trainer + extra_val),
  quant (INT8 fake-quant eval, P9), **behavior (eval_behavior_v0 scorers)**.
- `scripts/` — stage scripts, `train_stage3.py`, `eval_ppl.py`,
  **`build_eval_behavior_v0.py`**, **`eval_behavior.py`**, `plot_perf_trend.py`,
  and `scripts/pod/` (parameterized GPU session scripts).
- `configs/` — Stage 0/1; stage3: `s1_ffn_norm`, `s1_ext`, `s2_blocks`,
  `s2_blocks_v1`, `s2v1_from_s1`, `s2v1_from_init` (all ran), smoke configs.
- `data/` — frozen v0 and v1 mixtures (jsonl gitignored, manifests committed);
  **`data/eval_behavior_v0/` (prompts.jsonl + manifest, both committed)**.
- `tests/` — 100 tests. `logs/` — decisions (16), experiments (13), proposals (3),
  supported_models, artifact_manifests, this file.
- `artifacts/` (gitignored) — stage0 stats, stage1 checkpoint, stage3 run
  artifacts and reference scorecards (small files local; final weights HF-only
  except s1).
- `assets/` — perf figure json + svg. The json now carries a `headline` metric
  block, `systems` (one entry per student: current best + previous best +
  params), `references` (teacher size, score `null` until measured), the `guard`
  block (NLL), and the 8-attempt run history that generates the README table.

## Latest known working commands

```
uv run pytest tests/ -q                                          # 100 passed
uv run python scripts/build_eval_behavior_v0.py                  # rebuild prompt set
uv run python scripts/eval_behavior.py \
  --model artifacts/stage3/s1_ffn_norm_v0/checkpoints/step_000660/model \
  --out artifacts/stage3/s1_ffn_norm_v0/eval_behavior_v0.json
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
  --model <ckpt> [--fake-quant int8 [--fake-quant-scope decoder]]
uv run python scripts/plot_perf_trend.py [--print-table]
POD_ID=<id> HOST=<ip> PORT=<port> bash scripts/pod/orchestrate.sh   # GPU session
```

## Next actions (ordered)

Revised 2026-07-28 for the maintainer directives (decision record 2026-07-28):
the next work is a **Stage 3 supplementary experiment**, and its teacher targets
must be **top-n sampled and verified correct**. The experiment is staged behind
a cheap pilot instead of being one bulk spend. Item 2 is the control run that
every trace comparison needs; without it, "traces helped" and "samples are no
longer torn" are the same number.

1. ~~Score the teacher~~ — **done 2026-07-28**, 0.7443 vs the student's 0.2015
   ([log](experiments/2026-07-28_teacher_behavior_v0.md)). Biggest gaps: math
   +0.714, tool_call +0.667, format_ok +0.618. Grounding's ceiling is only
   0.562, so effort there has little headroom. **Carry the caveat:** student
   rows were scored at cap 512 and the teacher at 4096, so form metrics are not
   yet like-for-like — item 2's session re-scores the references at 4096.
2. **Packing / `block_len` control run — the baseline every trace run needs.**
   The data path changed (best-fit packing) and the measured length p90 is
   **1,508 against `block_len` 1024**, so the current baseline trained on torn
   samples. Re-run `s2v1_from_init`'s recipe with **only the data path
   changed**: best-fit packing, `block_len` 2048, `blocks_per_step` 8 —
   identical tokens/step and total token budget (P6), same 2700 steps, same
   seed 20260726. Hypothesis: grounding and multi-hop improve, since those are
   the long slices being cut.
3. **Run-to-run variance** (~$1–2, rides on the same pod as 2). The headline
   metric is a composite of rates over 7–76 prompts; without a noise floor a
   "win" is not readable.
4. **`eval_behavior_v1` prompt-set expansion — free, CPU, no approval needed.**
   ~36 prompts/group from held-out val, v0 kept as an exact subset. Also raise
   the student cap to 4096 for trace-trained checkpoints and re-score the
   references at that cap (a trace student at 512 would score ~0 for protocol
   reasons).
5. **Top-n teacher pilot on vLLM** (~$3–5): accept@1 vs accept@n, reject-reason
   histograms, per-slice trace lengths, engine throughput. Scopes item 6.
6. **Bulk verified generation** ($25–145 depending on n and slice scope; changes
   the official data mixture) — decide with pilot numbers, not before.
7. **The trace-target Stage 3 run (option B)**, compared against (2), not
   against the old torn-sample runs. The kernel benchmark rides on this session
   since it needs the production `block_len`.
8. Stage 4 online data collection design — **now includes real-time teacher
   generation feeding the student** (decision 2026-07-28): that is on-policy
   work, not a Stage 3 optimisation.
8. Optional backlog: Stage 1 ablations; a from-init-tuned lr/warmup sweep (A2
   was run under the ladder's hyperparameters, so single-stage may have more
   headroom than measured).

## Open decisions for the user

- **Score the teacher (~$1–1.5, ≈1 h GPU).** Independent of everything else and
  the fix for the figure's missing ceiling. Recommended first.
- **Online teacher generation feeding the student is Stage 4/5, not Stage 3**
  (decision 2026-07-28). The deciding property is whose distribution the
  training states come from, not whether the teacher runs in real time — the
  teacher already runs inside Stage 3's loop doing full-vocab KD every step.
- **Stage 3 supplementary experiment (top-n verified teacher targets), revised
  2026-07-28.** Next decision after the teacher scorecard is the **~$3–5 top-n
  pilot** (next action 3); the mixture-change approval, the vLLM dependency
  approval and the build spend all follow its measured thinking lengths and
  accept rates. Build estimate **$25–145** with the teacher at full capability
  (was $6–11 when the plan suppressed its reasoning) — scoped by lowering n or
  dropping slices, never by suppressing thinking.
- **Target text for teacher-generated data: (A) answer-only, (B) full trace,
  (C) hybrid.** Recommended (A) for this experiment; B/C change the student's
  output contract and belong in their own proposal. Needed before the bulk build,
  not before the pilot.
*(Closed 2026-07-28: the README Optim-record question. The answer is a standing
rule — no records during baseline construction, first record point after Stage 6
— so it is no longer an open decision. See the Status section and the 2026-07-28
decision record.)*

## Links

- `logs/experiments/2026-07-27_stage3_start_point_ablation.md` (latest run + review)
- `logs/experiments/2026-07-27_eval_behavior_v0.md` (behavior gate: build + baselines)
- `logs/experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md` ·
  `logs/experiments/2026-07-25_stage3_s2_ab_gpu_run.md` ·
  `logs/experiments/2026-07-22_stage3_s1_gpu_run.md`
- `logs/experiments/2026-07-26_stage2_offline_v1.md` ·
  `logs/experiments/2026-07-26_int8_fakequant_eval.md`
- `logs/proposals/2026-07-27_stage2_teacher_generated_answers.md` (awaiting approval)
- `logs/decisions.md` · `logs/supported_models.md` · `logs/artifact_manifests.md`
- `scripts/pod/AGENTS.md` (GPU session playbook)
