# Experiment record — AlphaAvatar-distill

The single consolidated record of everything run. Replaces 25 per-run logs and 11
proposal files, which are preserved in git history at commit `866dac2`.

**Teacher** `Qwen/Qwen3-4B-Thinking-2507@768f209d` (2560 hidden, 36L, FFN 9728,
32Q/8KV) → **student** 0.6B-class (1024 hidden, 28L, FFN 3072, 16Q/8KV, tied
embeddings). BF16 training, INT8 deployment target.

**Total spend to date: $117.32** of the **$126.02** cap — **$8.70 remains, and
nothing beyond Experiment 3 is authorized.**

| period | $ | detail |
|---|---:|---|
| through corpus v2 (2026-08-01) | 34.52 | §6 — training/eval $7.93, teacher generation $26.59 |
| Experiment 1, data-scaling matrix | 61.50 | §11 — 24 arms $47.6, control + first eval $8.1, sweep $5.8 |
| Experiment 2 phase 1, data cleaning | 12.97 | §12.15 — experiment $11.23, avoidable pod waste $1.74 |
| Diagnostic session (benchmark + reference + recall) | 0.52 | §14 — 94 min on an RTX A6000 at $0.33/h |
| D0 no-training diagnostics | 1.15 | §16 — 210 min on an RTX A6000 |
| P0-assistant | 2.75 | §17 — 167 min on an L40S |
| P2-ceheavy | 2.88 | §18 — 174.6 min on an L40S |
| **itemized subtotal** | **116.29** | |

**Unreconciled: $1.03.** The itemized rows sum to $116.29 while the verified
running total carried in `STATE.md` is **$117.32**. The difference is not
attributed to any session here, so the **larger** figure is used for every
remaining-budget decision. Do not "fix" this by deleting the gap.

**$8.70 of the $30 Experiment 2 allocation is unspent.** §6 below is the
*pre-Experiment-1* breakdown and its "project total" line is scoped to that
period; this table is the current figure.

---

## 1. Pipeline state

| stage | what exists | status |
|---|---|---|
| 0 — activation collection | 949,859 tokens, v1 | **done** |
| 1 — structural init | `stage1/qwen3_0p6b_init_v0/checkpoint`, holdout NLL 11.748 | **done, pinned start point** |
| 2 — offline mixture | `stage2_v1`, 22.13M train tokens | **done** |
| 3 — recovery | best checkpoint `s2v1_from_init@2700`, holdout **3.8285**; teacher corpus v2 + 6-rung token ladder built and gate-passed (§10) | **open — no checkpoint generates usable output; training matrix not started** |
| 4/5/6 | — | not started |

**The blocking fact:** under unrestricted generation, *every* checkpoint
including the best one degenerates into repetition. No model in this line yet
produces a complete answer in the teacher's thinking protocol.

---

## 2. Stages 0–2 (all CPU or cheap GPU, concluded)

* **Stage 0** — teacher activations, 949,859 tokens.
* **Stage 1** — PCA/sandwich structural init. Holdout NLL **11.748** vs random
  init 12.13 vs teacher 2.63. Gate passed. This checkpoint (`model.safetensors`
  sha256 `86fbba78…`) is the pinned fork point for all recovery work.
* **Stage 2** — offline mixture v0 (5.39M tokens) then v1 (22.13M).

## 3. Stage 3 recovery runs

| run | what it tested | result | verdict |
|---|---|---|---|
| `s1_ffn_norm` (660 steps) | FFN+norm recovery | holdout 4.21 | gate passed |
| `s2_ab` | freeze-set sizing | attention-unfrozen adopted; holdout flat, mixture v0 exhausted | informed v1 |
| `s2_blocks_v1` (2700) | mixture v1 | holdout **3.8003** | best NLL |
| start-point ablation | ladder vs single-stage | single-stage reaches 3.8285 with 33% fewer steps | **ladder retired** |
| packing / `block_len` control | best_fit@2048 vs concat@1024 | +2.1% regression on both seeds | **rejected** |
| CE/KD conflict | `kd_scope all_no_think` | p(`</think>`) **0.2995 → 0.9989** | conflict confirmed causal |

**Standing branch point:** `s2v1_from_init/step_002700`, holdout **3.8285**.

## 4. Measurements that constrain everything downstream

| finding | measurement |
|---|---|
| **Behavior-metric noise floor** | seed-only spread on `behavior_score_v0` = **0.1290**, wider than any inter-arm difference reported. ≥2 seeds required for any behaviour claim |
| **Cold-start NLL noise** | two seeds of one config differed by **2.21 nats** (6.62 vs 8.83) from the Stage 1 init. Holdout NLL is unresolvable at 2 seeds from cold start; ≥4 needed |
| **bf16 decoding is not batch-invariant** | student 1/6 identical batch-1 vs batch-6; **fp32 6/6** |
| **Both rollout engines ≈5.5× HF and scale** | vLLM 0.26.0 **247.5** tok/s, SGLang 0.5.12 **241.0**; wall 57.03 s vs 56.87 s. Policy mismatch negligible (off-policy rate **0.000**, KL ~1e-4) |
| **Teacher natural termination** | **80.1%** of 1,504 rollouts; lengths p25 466 / **p50 727** / p90 2233 / p99 3854 / max 4069 |
| **The 4096 generation cap censored 19.9%** | per slice: rag 0.5%, multihop 1.1%, gsm8k 8.5%, **openmath 69.7%** — which is what drove openmath's 0.261 accept rate |
| **Teacher targets are 4.2× longer than public** | `best_fit`@8192 is lossless; `concat`@1024 would split 79.9% |
| **`block_len` 8192 memory** | **44,983 / 46,068 MiB (97.6%)** with gradient checkpointing, ~4.3 s/step. `best_fit` pads every block so this peak is constant |
| **Supervised-token asymmetry** | teacher targets carry **18.9×** the supervised tokens of public ones on the same prompts (519,478 vs 27,526) |

## 5. The two four-arm runs — both diagnostics, neither a route decision

### 5.1 Post-s2v1 continuation (2026-07-30, $3.50 of the training pod)

Public-target control vs teacher-native treatment × 2 seeds, 137 steps, forked
from `s2v1_from_init@2700`. **Invalid as a target comparison:** that checkpoint
is 2,700 steps of public-target training, so the public arm started inside its
own target distribution. Relabelled a continuation diagnostic; its R2 "reject"
is void as evidence about teacher-native targets.

### 5.2 Corrected baseline from the Stage 1 init (2026-07-30, $2.30)

Same 2×2 forked from the pinned Stage 1 init, so neither arm had a path
advantage. Rule R2 fired again — but the run is **convergence-limited** (137
steps ≈ 5% of the reference budget, 487 prompts) and was **measurement-limited**
(99.3% of treatment generations censored at 512 tokens).

| | step-0 | control (public) | treatment (teacher) |
|---|---:|---:|---:|
| holdout NLL | 11.7565 | 7.7260 ±1.10 | **6.2255** ±0.48 |
| `format_ok` | 0.000 | 0.625 | 0.000 |
| `terminated` | 0.000 | 0.658 | 0.007 |

The teacher arm had the **larger** NLL improvement.

### 5.3 What the unrestricted pilot then showed ($0.79)

8 deterministic prompts × 6 checkpoints, full 262,144 context, concurrency 1, no
token cap:

| checkpoint | natural | degenerated | context-limited |
|---|---:|---:|---:|
| step-0 Stage 1 init | 0 | 8 | 0 |
| public arms (2 seeds) | 7 / 5 | 1 / 3 | 0 |
| teacher arms (2 seeds) | 0 / 0 | 8 / 8 | 0 |
| **`s2v1@2700` (best ckpt)** | **0** | **8** | **0** |

* **The best checkpoint degenerates too**, so degeneration is a property of the
  whole student line, not of teacher-native targets.
* **Zero context-limit hits.** The 512-token cap was hiding repetition loops,
  not long reasoning (e.g. a 17-token block repeated 15× from position 513).
* The public arm's apparent win was **protocol substitution**: its natural
  terminations are 5–18 token stubs after an **empty** `<think>`
  (`</think>\n\nArthur's Magazine<|im_end|>`), several incoherent.

**Conclusion carried forward: no route-level claim about teacher-native
supervision is supported by any run so far.**

## 6. Cost, including waste

| item | cost | assessment |
|---|---:|---|
| Stage 3 recovery runs (s1, A/B, v1, ablation, packing, CE/KD) | prior sessions | necessary |
| engine benchmark + isolated engine | prior sessions | necessary once; now concluded and deferred |
| teacher corpus v1 generation (752 prompts) | **$1.37** | necessary; superseded by corpus v2 |
| post-s2v1 continuation 4-arm run | **$3.50** | **wasted** — invalid start point, my error |
| corrected baseline 4-arm run | **$2.30** | diagnostic only; convergence- and measurement-limited |
| unrestricted pilot | **$0.79** | necessary; produced the finding that reframed everything |
| **training/eval subtotal** | **$7.93** | of which **$3.50 was avoidable** |
| §6 validation gate (§9) | **$1.03** | necessary; ~$0.65 of it lost to two infrastructure failures |
| corpus v2 bulk generation (§10) | **$25.56** | necessary; **~$8.70 of it was idle pod time** |
| **subtotal through 2026-08-01** | **$34.52** | generation subtotal $26.59, all against the $50 generation cap |

This table covers everything **before Experiment 1**. Experiment 1 ($61.50, §11)
and Experiment 2 phase 1 ($12.97, §12.15) are costed in their own sections; the
running project total is in the header.

**Wasted:** the 2026-07-30 continuation run. Forking both arms from a
public-trained checkpoint was an error that a five-minute check of the start
point would have caught before spending.

**Also wasted:** ~$8.70 of idle pod time on the corpus v2 run (§10) — the job
finished hours before the pod was deleted.

**Also avoidable in hindsight:** the 512-token evaluation cap, which censored
99.3% of one arm and made two runs uninterpretable. Now forbidden by AGENTS.md
P18.

## 7. Standing decisions

| decision | date |
|---|---|
| Stage 3 recovery is single-stage from the Stage 1 init | 07-27 |
| Behaviour comparisons need ≥2 seeds; NLL from cold start needs ≥4 | 07-28 / 07-31 |
| The teacher is never forced out of thinking mode | 07-28 |
| Exact token agreement is **not** an engine adoption gate; engines run in their own official images | 07-30 |
| `best_fit` @ `block_len` 8192 for teacher-native targets, gradient checkpointing required | 07-30 |
| **P17** — teacher-behaviour fidelity: no no-think / empty-think / final-only / shortened substitute targets | 07-30 |
| **P18** — no artificial generation cap in formal measurement | 07-30 |
| Corpus is **effectively n=1** (92.7% byte-identical candidates); per-candidate seeds fixed | 07-30 |
| Teacher template injects **no** default system message — a fact about the checkpoint | 07-31 |
| **Project protocol: an explicit system message is mandatory**, fixed requirement, not an experimental variable | 07-31 |
| Capability-gap ("difficulty-aware") mixture, declared in **supervised tokens**, replaces equal four-way balance — it shaped generation | 08-01 |
| **Experiment order: scaling first on a uniform mixture; data mixing second; difficulty curriculum third** | 08-01 |
| Multi-turn data enters by **turn expansion**; only the newly generated turn is supervised, siblings never share a block | 08-01 |
| Keep the system-prompt packing boundary and pay 3.35× padding; training budget raised to **$60** | 08-01 |
| Corpus generation samples at the teacher's **official preset** (`0.6 / 0.95 / 20 / min_p 0`), no greedy candidate | 08-01 |

## 8. Reusable assets

| asset | where |
|---|---|
| Stage 1 init (pinned fork point) | `stage1/qwen3_0p6b_init_v0/checkpoint`, sha256 `86fbba78…` |
| **recovery corpus v2 + ladder** (2026-08-01) | `sessions.jsonl` `2b4edc2e…`, `candidates.jsonl` `f7f5035e…` — **local only, not yet on the relay** |
| teacher corpus v1, 752 prompts / 540 accepted | `stage3_teacher_corpus_20260730/`, targets sha256 `18028f0c…` (superseded) |
| rollout snapshot, 1,504 rollouts / 2.46M tokens | same prefix, sha256 `0e5b20dd…` |
| all trained checkpoints | relay under `stage3/`, `tt2x2/`, `ttb/` |
| Stage 2 mixture v1 | `data/stage2_v1` (regenerable from its manifest) |

---

## 9. Data-scaling experiment — §6 validation gate (2026-07-30)

**Objective:** gate into bulk teacher generation for the recovery-data scaling
study. 10 prompts × 4 types × `n=4`, full production path, plus a HotpotQA-only
follow-up at n=70.

**Hardware/stack, pinned:** RunPod L40S 46,068 MiB, driver **580.159.03**;
vLLM **0.26.0** (offline `LLM.generate`, per-prompt `SamplingParams`), torch
2.11.0+cu130, transformers 5.14.1, Python 3.12, isolated venv.
Teacher `Qwen/Qwen3-4B-Thinking-2507@768f209d`, chat template sha `3802169b…`,
stop ids `[151643, 151645]` from `generation_config.json`.
Preset `temperature 0.6 / top_p 0.95 / top_k 20 / min_p 0`, session limit 8,192
end-to-end, per-prompt completion budget `8192 − rendered_prompt − 8`.

**Result: gate PASSED** — every §9 check across all four types (generation,
packing, loader, prefix equivalence).

| type | prompt accept | cand accept | pre-pack sup | post-pack sup | reject |
|---|---:|---:|---:|---:|---|
| rag_evidence | 1.000 | 1.000 | 368 | 368 | — |
| multihop_qa | 1.000 | 1.000 | 998 | 740 | — |
| gsm8k | 1.000 | 1.000 | 1,037 | 1,037 | — |
| openmath | 0.700 | 0.600 | 3,382 | 2,267 | length_limited 16/40 |

HotpotQA n=70: accept 1.000, supervised mean **963** sd 850 (cv 0.88), 95% CI
[764, 1162]. Packing discard **21.2%**; efficiency 0.958. Throughput 339 tok/s
at 10 concurrent, **682 tok/s** at 70 concurrent.

**Verdict at the time: blocked on corpus size, not on correctness.** Equal
four-way balance needs 1,474 prompts/type for 5.50M post-packing supervised
tokens; `hotpot_qa` has 1,074. Max reachable **4.01M** (conservative 3.84M). The
session stopped and escalated rather than quietly alter the balance, target or
pool. **Resolved 2026-07-31/08-01:** the maintainer lifted equal balance in
favour of a capability-gap mixture and admitted multi-turn data via turn
expansion, which is what §10 generated. That weighting shaped **generation**;
the first training experiment cuts the same corpus at a **uniform** mixture
(§10, ladder re-cut).

**Cost: $1.03**, all against the $50 generation cap. Of that, ~$0.65 was spent on
two infrastructure failures before the successful run.

**Infrastructure findings (reusable):**
* vLLM 0.26.0's wheel links `libcudart.so.13`; a 570.x driver host cannot run it
  and `--torch-backend=cu128` does not help, because it changes torch, not the
  vLLM extension. Create the pod with `--min-cuda-version 13.0`.
* FlashInfer JIT-builds the top-k sampling kernel during vLLM warmup, so `ninja`
  must be on `PATH` — `top_k=20` is in the official preset, so without it the
  engine dies *after* loading weights.
* Put the venv on the container disk: a torch install onto the network-mounted
  `/workspace` took >9 minutes.
* `tar` needs `--no-same-owner` on that mount, or it exits non-zero on chown.

---

## 10. Recovery corpus v2 (2026-08-01)

**Objective:** build the maximal reusable teacher corpus for the data-scaling
study, under an 8,192-token end-to-end session limit, with prompt counts per type
set from the measured capability gaps rather than equal four-way balance.

**Stack, pinned:** L40S 46,068 MiB, driver 580.159.03; vLLM **0.26.0** offline
`LLM.generate` with per-example `SamplingParams`; torch 2.11.0+cu130,
transformers 5.14.1. Teacher `Qwen/Qwen3-4B-Thinking-2507@768f209d`, chat
template sha `3802169b…`, stop ids `[151643, 151645]` from
`generation_config.json`. Preset `0.6 / 0.95 / top_k 20 / min_p 0`, `n=4`,
per-example budget `8192 − rendered_prompt − 8`.

**Result:** 11,574 examples → **11,174 accepted (96.5%)**, 66.08M generated
tokens, 59,367 s (16.5 h) on one L40S, **$25.56**. Gate re-run on the full
corpus: **PASSED** — every check for all six types, plus the pack-level checks
(`blocks_match_audit`, `no_session_repacked`, `ladder_monotonic`,
`loader_roundtrip`, `loader_no_truncation`).

| type | examples | accepted | ex accept | tok/cand | supervised | sup/session | correctness |
|---|---:|---:|---:|---:|---:|---:|---:|
| rag_evidence | 4,100 | 4,100 | 1.000 | 503 | 2,087,594 | 509 | 0.978 |
| multihop_qa | 1,074 | 1,074 | 1.000 | 1,061 | 1,134,028 | 1,056 | 0.861 |
| gsm8k | 1,700 | 1,698 | 0.999 | 1,190 | 1,998,183 | 1,177 | 0.890 |
| openmath | 900 | 579 | 0.643 | 5,196 | 1,977,473 | 3,415 | **0.380** |
| code | 1,200 | 1,123 | 0.936 | 4,609 | 4,773,086 | 4,250 | n/a |
| tool_calling | 2,600 | 2,600 | 1.000 | 419 | 1,073,688 | 413 | n/a |

**Acceptance is hygiene, not correctness** — by design (2026-07-28 decision:
Stage 3 trains the teacher's unfiltered distribution; correctness selection is
Stage 4/5). Correctness is *computed and stored* per candidate: `code` and
`tool_calling` have no mechanical key (`unverifiable_slice`), and **openmath's
0.380 means roughly a third of accepted openmath targets teach a wrong final
answer**. Rejections are dominated by the 8,192-token session limit:
`length_limited` took 1,541/3,600 openmath candidates and 702/4,800 code ones.

**Ladder:** one pack cut into six nested rungs — 3,720 blocks, 11,174 sessions,
**10,805,451 post-packing supervised tokens** (corpus-wide packing efficiency
0.4709, 1,861 system-prompt groups). ~2× the 5.50M rung, leaving headroom for
saturation rungs above it. Every rung lands within **0.2 pp** of the declared
mixture; nesting is exact and monotonic.

| rung | actual supervised | blocks | sessions | real tokens | padding | terminal truncations |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25M | 254,026 | 197 | 446 | 403,962 | 1,209,862 | 29 |
| 0.46M | 464,029 | 351 | 800 | 729,900 | 2,145,492 | 53 |
| 0.86M | 862,793 | 617 | 1,449 | 1,349,125 | 3,705,339 | 103 |
| 1.60M | 1,600,687 | 1,064 | 2,587 | 2,442,920 | 6,273,368 | 189 |
| 2.96M | 2,960,131 | 1,815 | 4,501 | 4,434,241 | 10,434,239 | 341 |
| **5.50M** | 5,500,516 | 2,863 | 7,531 | 7,928,141 | 15,525,555 | 625 |

**Ladder re-cut for Experiment 1 (2026-08-01, CPU, $0).** The maintainer
corrected the experiment order: the first study asks only whether behavioural
recovery scales with supervised-token count, so composition must be a constant,
not a designed treatment. The mixture is a parameter of the *cut*, not of the
generated data, so the pack was simply re-cut at **uniform 16.67% × 6**. All six
rungs remain reachable:

| rung | actual supervised | blocks | sessions | real tokens | terminal truncations |
|---:|---:|---:|---:|---:|---:|
| 0.25M | 252,985 | 216 | 479 | 449,307 | 33 |
| 0.46M | 460,088 | 380 | 848 | 797,951 | 62 |
| 0.86M | 864,750 | 682 | 1,502 | 1,472,149 | 109 |
| 1.60M | 1,600,353 | 1,174 | 2,649 | 2,661,299 | 190 |
| 2.96M | 2,960,507 | 1,944 | 4,524 | 4,730,748 | 352 |
| **5.50M** | 5,501,372 | 2,941 | 7,350 | 8,256,511 | 635 |

Uniform is realized within **0.3 pp** at the smallest rung and **0.03 pp** at the
top; nesting is exact and monotonic. Two measured costs: **+6.2% training
compute** (7,337 blocks/epoch vs 6,907, because `tool_calling` rises 15% → 16.7%
and packs worst), and the **saturation ceiling falls to 6,076,356** supervised
tokens, bound by `multihop_qa`'s 1,012,726 post-packing tokens — against 10.81M
under the weighted cut. Artifacts: `artifacts/stage3/ladder_uniform_probe`
(gitignored), `blocks.npz` sha256 `6f324cb0…`, `ladder.json` `d4941722…`,
`audit.jsonl` `15f16b7b…`. The weighted cut is kept for Experiment 2.

**Reproducibility gap (P4).** The corpus manifest's `code_state` block carries
**no git commit**: the shipped bundle was unpacked outside a git checkout, so
`git rev-parse` failed and the manifest stored `code_state_error` instead. The
corpus is pinned by data hashes (`sessions.jsonl` `2b4edc2e…`,
`candidates.jsonl` `f7f5035e…`, both re-verified locally 2026-08-01), teacher
revision, tokenizer and chat-template hashes, and the full command — but its
code state is pinned only by the bundle that was shipped. Fix the bundle to
carry the commit before the next paid generation.

**Sizing lesson.** Prompt counts were set from deliberately conservative
supervised-token estimates, and those were most wrong on the most expensive
types: `code` came in at 4,609 tok/candidate against a 1,300 estimate (3.5×),
while `tool_calling` came in at 419 against 900. Net effect was over-provisioning
of `code`/`openmath` and a corpus ~2× the target — useful as saturation headroom,
but ~$4 of it bought tokens the 5.50M rung cannot consume.

**Packing efficiency 0.34 at the 5.50M rung**, and this is the dominant cost of
the whole study:

| at the 5.50M rung | blocks | efficiency | sessions/block | supervised/block |
|---|---:|---:|---:|---:|
| tool blocks | 2,074 | 0.092 | 1.11 | 398 |
| non-tool blocks | 789 | 0.985 | 6.62 | 5,925 |

`tool_calling` supplies 15% of the supervision and consumes **72% of the
blocks**, because each conversation's tool schema renders into the system block
and the system prompt is a hard packing boundary. The rung needs 2,863 blocks
where a dense pack would need ~855 — **3.35× training compute**. Maintainer
accepted the cost rather than relax the packing rule (2026-08-01) and raised the
training budget to $60.

**Cost, including waste:** generation **$26.59** total ($1.03 gate + $25.56
corpus), against the $50 generation cap. Of the corpus run, **~$8.70 was idle pod
time** — the job finished at 06:27 and the pod was deleted at 15:14, because
polling was driven by user prompts rather than a timer and `--terminate-after`
was a 32 h backstop. Standing instruction since: tear down a finished pod without
being asked.

**Where the corpus lives:** persisted 2026-08-01 to the relay under
`stage3_recovery_corpus_v2/` (corpus + both ladder cuts), **9/9 files
hash-verified** against the local copies — LFS oid for the large files,
download-and-hash for the small ones.

---

## 11. Experiment 1 — data-scaling matrix (LAUNCHED 2026-08-01, running)

**Objective:** does the student's behavioural recovery scale with
teacher-generated supervised tokens? Token count is the only variable; the
mixture is held constant and uniform (the capability-gap weighting is
Experiment 2, the difficulty curriculum Experiment 3).

**Design:** 24 arms = 6 rungs × 2 seeds × 2 inits, 3 epochs per rung, on the
uniform ladder cut. 44,024 steps, ~52.6 h at the measured 4.3 s/step. Arms
differ from `configs/stage3/recovery.json` only in data source, rung, seed and
start checkpoint (`scripts/data/build_experiment1_configs.py`).

**Hardware/budget:** two L40S pods split by initialization (`1ligfkwnaous4u`
pca, `vjavemn7m2tw5a` rand), each with a hard `--terminate-after` at 30.0 h, so
the session cannot exceed **$59.40** against the $60 cap. Orchestrators refuse
to start an arm they cannot finish before that deadline.

**Enabling change:** the trainer could not consume the ladder before this
session — it always re-packed from a Stage 2 mixture dir, which would have made
the rungs trained differ from the rungs measured. `packing: "ladder"` reads the
pack as cut; the run manifest records the pack's hashes.

**Readouts, and what is deferred:** the pods produce holdout NLL and a
generation smoke test per arm. The **P18 uncapped behavioural readouts** —
`natural_termination_rate`, `degeneration_rate`, length p50 — are *not* run
inline: `eval_behavior.py --unrestricted` has no degeneration stop, so a
checkpoint in a repetition loop generates until the 262,144-token context is
exhausted and one prompt can outlast a training arm.

**Maintainer direction 2026-08-02:** a rising holdout NLL may mean the student
is shedding *knowledge* rather than *reasoning*, and NLL cannot separate those —
so behavioural and reasoning benchmarks must be run before the checkpoints are
released, and they must use **vLLM with no fixed 512-token truncation**, not the
capped path.

Both training pods run driver **570.124.06**, which cannot host vLLM 0.26 (its
wheel links `libcudart.so.13`; `--torch-backend=cu128` does not help because it
changes torch, not the extension). They were created without
`--min-cuda-version 13.0` — correct for training, fatal for engine work. Rather
than downgrade the engine to fit the host, the maintainer directed releasing the
training pods and provisioning a dedicated evaluation pod. The uncapped battery
therefore runs there, against checkpoints pulled from the relay.

**Infrastructure lessons from the teardown, all self-inflicted and all cheap to
avoid next time:**
* `pgrep -f <pattern>` **matches the command carrying the pattern**. A watcher
  waiting on `pgrep -f train_stage3` never fires, and `pkill -f <script>` kills
  the shell running the pkill. Bracket the first character (`[t]rain_stage3`) or
  match on recorded PIDs.
* `sed 's|.*/||'` on a `sha256sum` line is greedy from the line start and eats
  the hash along with the path. Rewrite only the path column: `s|  .*/|  |`.
* `scp 'host:dir/*/train_log.jsonl' dest/` collapses every arm onto one basename
  and silently keeps the last. Tar the tree instead. This cost one arm's
  `train_log.jsonl`; its eval lines were recovered from the console log, which
  survived only because console filenames are arm-unique.

**Result: all 24 arms trained.** Both pods released 2026-08-02 (pca 08:33, rand
08:49). Training spend **$47.6**, inside the $59.40 cap. Per-arm results in
`artifacts/stage3/e1_results.json`.

| supervised tokens | PCA sa | PCA sb | rand sa | rand sb |
|---:|---:|---:|---:|---:|
| step 0 | 10.9199 | 10.9199 | 12.1615 | 12.1615 |
| 0.25M | 2.0938 | 2.1427 | 8.8291 | 8.8234 |
| 0.46M | 1.7477 | 1.7611 | 8.3346 | 8.3575 |
| 0.86M | 1.5101 | 1.5038 | 7.9403 | 7.9542 |
| 1.60M | 1.2952 | 1.3015 | 7.4068 | 7.4025 |
| 2.96M | 1.1468 | 1.1486 | 6.6812 | 6.6643 |
| **5.50M** | **1.0032** | **1.0052** | **5.9807** | **5.9789** |

(teacher-native held-out CE, 16 pack-tail blocks disjoint from every rung)

**Three findings, all at two seeds:**

1. **Seed noise is negligible on this metric.** The largest seed gap across all
   12 pairs is **0.049** (0.25M PCA) and most are under 0.02 — one to two orders
   of magnitude below the per-rung effect of adding data. Teacher-native val CE
   is a usable instrument here, unlike `behavior_score_v0`, whose 0.129 noise
   floor swamped its own effects.
2. **Neither init has saturated at 5.50M.** PCA's per-rung gain decays
   (0.35 → 0.24 → 0.21 → 0.15 → 0.14) but is still clearly non-zero at the top
   rung; random's gain *grows* (0.49 → 0.39 → 0.53 → 0.73 → 0.70). The curves
   approach the ceiling from opposite directions. **The question "how much data
   is enough" is therefore not yet answered** — and the uniform cut caps this
   corpus at ~6.08M, so answering it needs more generation, not more training.
3. **Initialization dominates data over this range.** At the top rung PCA is at
   1.0032 against random's 5.9807. Extrapolating random's ~0.7/rung, it would
   need many further doublings to reach where PCA already sits at 0.25M. Stage
   1's structural init is not a head start that data erases.

**Holdout NLL moves opposite to teacher-native CE on the PCA arms** — 6.72 →
6.16 → 8.88 → 9.71 → 10.40 → 10.79 while CE falls monotonically — but drifts
*down* on the random arms (10.92 → 10.24). The reading that fits both: the PCA
student has general LM ability to trade away as it specialises onto teacher
reasoning traces; the random student has none to lose. **Whether that trade is
knowledge or reasoning is not answerable from NLL**, which is why the maintainer
directed a full behavioural evaluation (below) before any conclusion.

### Complete results — 25 checkpoints, four measurements each (2026-08-03)

Every arm carries teacher-native val CE, FineWeb-Edu holdout NLL, uncapped
behaviour and GSM8K EM. **No missing measurements.** Consolidated table:
`artifacts/stage3/e1_consolidated.json`; regenerate with
`scripts/evaluation/consolidate_e1.py`.

**Scaling curves, seed-averaged:**

| supervised tokens | PCA CE | rand CE | PCA behaviour | rand behaviour | PCA nat.term | rand nat.term | PCA GSM8K EM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25M | 2.1183 | 8.8263 | 0.3194 | 0.0001 | 0.533 | 0.007 | 0.005 |
| 0.46M | 1.7544 | 8.3461 | 0.3626 | 0.0053 | 0.671 | 0.013 | 0.000 |
| 0.86M | 1.5069 | 7.9472 | 0.3695 | 0.0403 | 0.763 | 0.039 | 0.000 |
| 1.60M | 1.2983 | 7.4047 | 0.4076 | 0.0781 | 0.835 | 0.072 | 0.040 |
| 2.96M | 1.1468 | 6.6727 | 0.4180 | 0.0639 | 0.921 | 0.079 | 0.015 |
| **5.50M** | **1.0042** | **5.9798** | 0.3781 | 0.1099 | 0.803 | 0.165 | 0.020 |

**Variance analysis — which differences are real.** Between-seed |Δ| against the
range the metric spans across rungs:

| metric | init | seed \|Δ\| mean | seed \|Δ\| max | range across rungs | range/noise |
|---|---|---:|---:|---:|---:|
| val CE | PCA | 0.0154 | 0.0489 | 1.1395 | **74×** |
| val CE | rand | 0.0109 | 0.0229 | 2.8502 | **261×** |
| holdout NLL | PCA | 0.6591 | 1.3327 | 4.6259 | 7× |
| holdout NLL | rand | 0.0685 | 0.1275 | 0.7700 | 11× |
| natural termination | PCA | 0.0526 | 0.1315 | 0.4474 | 8.5× |
| behaviour | PCA | 0.0472 | 0.0894 | 0.1564 | **3.3×** |
| behaviour | rand | 0.0275 | 0.0717 | 0.1205 | 4.4× |
| GSM8K EM | PCA | 0.0100 | 0.0200 | 0.0500 | 5× |
| GSM8K EM | rand | 0.0000 | 0.0000 | 0.0000 | — |

**What the data supports, in descending order of confidence.**

1. **CE scales with data — decisively.** 74× (PCA) and 261× (random) the
   between-seed noise, monotone on both inits at every rung. Neither has
   saturated at 5.50M: the last doubling still buys 0.14 (PCA) and 0.70 (random)
   nats.
2. **Natural termination scales with data on the PCA init** — 0.533 → 0.921
   through 2.96M at 8.5× the seed noise. It dips to 0.803 at 5.50M, which is
   within ~1.5 seed-|Δ| and should not be read as a real reversal.
3. **Initialization dominates everything.** At the top rung PCA reaches CE
   1.0042 against random's 5.9798, and behaviour 0.378 against 0.110. The random
   arms sit at p50 = 768 generated tokens at *every* rung — the signature of the
   degeneration stop firing — i.e. they essentially never terminate naturally.
4. **The behaviour composite barely resolves.** Its across-rung range is only
   **3.3×** the between-seed spread, versus 74× for CE. The rung-to-rung
   ordering on behaviour is not claimable; the init gap on it is.
5. **No reasoning develops anywhere.** GSM8K EM across all 25 checkpoints:
   min 0.000, max **0.050**, mean **0.006**. Random init is 0.000 at every rung
   and seed. The single highest value (0.050, `1600k_sb_pca`) is ~2 SE at n=100
   and is not reproduced by its own seed pair (`1600k_sa_pca` = 0.030) nor by
   larger rungs. **No rung, seed or initialization developed measurable
   reasoning ability** — which is the direct answer to whether the metric floor
   was hiding a signal. It was not.

**Consequence for the holdout-NLL story.** NLL rises with training duration
(PCA 6.72 → 10.79) while CE falls, but its between-seed |Δ| is **0.66** — an
order of magnitude noisier than CE — so it is a weak instrument here. The
step-matched control lands at 10.71 against the 5.50M arm's 10.79 despite a
1.29-nat CE gap: the FineWeb-Edu degradation tracks **optimizer steps**, not
unique data, and it is not evidence that data destroys general ability.

### Training-exposure accounting (audit, 2026-08-02)

| rung | blocks | supervised (unique) | steps | blocks consumed | supervised consumed | effective epochs | repeats |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.25M | 216 | 252,985 | 324 | 648 | 758,955 | 3.0000 | every block ×3 |
| 0.46M | 380 | 460,088 | 570 | 1,140 | 1,380,264 | 3.0000 | every block ×3 |
| 0.86M | 682 | 864,750 | 1,023 | 2,046 | 2,594,250 | 3.0000 | every block ×3 |
| 1.60M | 1,174 | 1,600,353 | 1,761 | 3,522 | 4,801,059 | 3.0000 | every block ×3 |
| 2.96M | 1,944 | 2,960,507 | 2,916 | 5,832 | 8,881,521 | 3.0000 | every block ×3 |
| 5.50M | 2,941 | 5,501,372 | 4,412 | 8,824 | 16,505,986 | 3.0003 | every block ×3, **one block ×4** |

Effective epochs = consumed training blocks ÷ available training blocks. Blocks
consumed = `total_steps × blocks_per_step` (2). Validation blocks (16, from the
pack tail) are excluded throughout and are never trained on.

**How the step count was set.** `scripts/data/build_experiment1_configs.py`
computes `total_steps = ceil(blocks × 3 / blocks_per_step)` — a *fixed number of
passes*, not a fixed optimizer budget. The trainer consumes blocks from a
deterministic stream of per-epoch permutations
(`stream_block_indices(n_blocks, seed, step × bps, bps)`), so each epoch is a
full permutation and every block is seen the same number of times. The single
`×4` block at the top rung is the `ceil` on an odd product (8,823 → 8,824).
Verified against the runs' own logs: `train_blocks=216 → total_steps=324`, and
`tokens_seen` at step 320 = 5,242,880 = 320 × 2 × 8,192.

**Does this isolate supervised-token quantity? No — and that must be carried
with every reading of the curve.** Effective *passes* are held constant at 3.0,
but absolute exposure scales with the rung:

* unique supervised tokens span **21.7×** (252,985 → 5,501,372)
* optimizer steps span **13.6×** (324 → 4,412)
* tokens processed span **13.6×** (5.31M → 72.29M)
* effective epochs span **1.0001×** (3.0000 → 3.0003)

So **dataset size and training exposure moved together by construction.** A
lower CE at a higher rung is consistent with "more unique data" *and* with
"more gradient updates"; this experiment cannot separate them. The two spans
differ (21.7× vs 13.6×) only because supervised tokens per block rise from 1,171
to 1,870 across the ladder as packing density improves — far too little
decoupling to identify the two factors.

This is the **pre-registered** design: `PROPOSAL.md` §4 chose fixed passes
(design i) and explicitly rejected a fixed optimizer budget (design ii) because
that conflates "more data" with "fewer passes". It is the standard shape for a
data-scaling law. It is not a defect, but it *is* a limit on the claim: the
curve measures **data quantity at constant passes**, not data quantity at
constant compute.

**The step-matched compute control (approved 2026-08-02, running).**
`e1_ctl_r0250k_sa_pca_stepmatched`: the 0.25M rung trained for **4,412 steps**
(~40.9 effective epochs), matching the 5.50M arm's optimizer budget. The config
is `e1_r5500k_sa_pca.json` with **only `rung` changed** — same PCA init, same
seed 20260726, same optimizer, same step-based LR schedule (4,412 total, 221
warmup, `min_lr_frac` 0.1), same 16 validation blocks, same eval protocol.
Verified by diff: the only differing fields are `rung`, `run_name`, `out_dir`
and `_purpose`.

**It is a step-matched control, not a token-compute-matched one**, and the
record must say so: packing density differs across rungs (1,171 vs 1,870
supervised tokens per block), so equal optimizer steps do **not** mean equal
supervised tokens processed — the control sees 4,412 × 2 × 8,192 = 72.29M
processed tokens like the 5.50M arm, but its supervised content is 216 blocks
repeated ~40.9 times rather than 2,941 blocks repeated 3 times.

**The three-way comparison — RESULT, 2026-08-02.** Same 4,412 steps for (2) and
(3); same unique data for (1) and (2).

| # | arm | blocks | steps | epochs | **val CE** | holdout NLL | **behaviour** | nat. term | degen | GSM8K EM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PCA 0.25M original | 216 | 324 | 3.0 | 2.0938 | 6.7169 | 0.3432 | 0.461 | 0.539 | 0.01 |
| 2 | PCA 0.25M step-matched | 216 | 4,412 | 40.9 | **2.2907** | 10.7082 | **0.4577** | **0.908** | **0.092** | 0.00 |
| 3 | PCA 5.50M | 2,941 | 4,412 | 3.0 | **1.0032** | 10.7875 | 0.3806 | 0.829 | 0.171 | 0.01 |

**The two axes disagree, and that is the finding.**

* **Distributional fit is bought by data.** Arms 2 and 3 are identical in init,
  seed, optimizer, LR schedule, step count and validation set — they differ
  *only* in unique data — and CE differs by **1.29 nats**, far above the ~0.05
  seed noise. Giving the small corpus the large corpus's entire compute budget
  did not merely fail to close the gap, it made CE **worse** than the 324-step
  original (2.0938 → 2.2907): the control peaked at CE 1.9716 by step 551 and
  overfitted monotonically thereafter while train loss fell to ~0.05.
* **Protocol competence is bought by passes.** On behaviour the *control* leads:
  natural termination **0.908** vs 0.829, degeneration **0.092** vs 0.171,
  fluency 0.821 vs 0.515, format_ok 0.842 vs 0.790. Repetition over a small
  corpus teaches the student to *finish a turn*; unique data does not, at this
  scale.
* **Neither buys reasoning.** GSM8K EM is **0.01 / 0.00 / 0.01** on 100 reserved
  prompts. There is no reasoning to trade away yet, which reframes the rising
  holdout NLL: it is the cost of training duration (the control and the 5.50M
  arm land at 10.71 vs 10.79 despite a 1.29-nat CE gap), not evidence that data
  destroys general ability.

**The honest limit on the behaviour half:** the three arms span **0.115** on the
behaviour composite, *below* the project's measured seed-only noise floor of
**0.129**. The behaviour numbers are reportable; the behaviour *ranking* is not
supported at one seed per arm. Only the CE difference clears its noise floor.

**Verdict on the gate:** the scaling curve may be reported as a **data** effect
on teacher-native CE. It may **not** be reported as a general capability effect
— behaviour does not follow it, and reasoning is absent from every arm.

**Measurement protocol (identical for all three arms).** Uncapped within the
model's **effective context of 8,192**, derived from the trained `block_len` and
recorded per sample with its full derivation (architectural context 262,144,
source `trained_block_len`). **This is not a 262K-context evaluation.** Zero
context-limit hits at 8,192, so nothing was truncated. Greedy decoding, the
mandatory system message, the same 76-prompt behaviour set and 100-prompt
reserved GSM8K slice, and a fixed degeneration detector with identical
thresholds everywhere. Per sample we record generated tokens, stop reason,
degeneration trigger and kind, right-censoring, and the complete raw output.

**Sweep execution.** 25 checkpoints x (behaviour + GSM8K) + holdout NLLs on one
L40S, 2026-08-02/03, **$5.8**. 125 artifacts hash-verified on the dev box before
the pod was released automatically. Four arms were staged from verified dev-box
copies because the HF relay remained storage-blocked.

**Defects found and fixed during evaluation, each of which would have corrupted
the result silently:**

* **A second `<|im_start|>system` turn** was injected into the 6 behaviour
  prompts that carry their own system message — a context the model never saw in
  training. Every behaviour number computed before the fix was discarded and
  recomputed. Now audited mechanically for all 176 prompts
  (`scripts/evaluation/audit_prompt_rendering.py`) and pinned by tests.
* **GSM8K exact match was never computed**: `score_sample` credits EM only from a
  precomputed `gsm8k_answer` field, which the slice builder did not emit. The
  reasoning benchmark ran with no reasoning metric until the axis values were
  checked.
* **The effective context was the architectural 262,144** rather than the trained
  8,192, making one wave cost over an hour and measuring an out-of-distribution
  regime.
* **`ninja` was not on `PATH`**, so FlashInfer could not JIT the top-k kernel and
  every vLLM wave failed — the same failure recorded in this log from the corpus
  build.

**Verdict: the scaling relationship is measured and internally clean, but two
things stop it short of a law.** The saturation point is outside the corpus
(neither init has flattened at 5.50M, and the uniform cut caps this pack at
~6.08M), and data quantity is confounded with training compute by the
fixed-passes design. Recovery-data sizing cannot be closed until the corpus
grows past ~6M uniform tokens or the mixture is relaxed (Experiment 2), and the
data-vs-compute control above is run.

---


---

## 13. Refactor equivalence audits and padding truncation (2026-08-04, CPU, $0)

All local; no GPU time. Verdicts under `artifacts/audit/` (gitignored).

### 13.1 Degeneration replay — PASS

`scripts/evaluation/audit_degeneration_replay.py` over **13,686 retained
generations** (E1, E2 phase 1, and the earlier runs).

| tier | question | result |
| --- | --- | --- |
| A (asserted) | do the pre- and post-move modules agree on the same input? | **13,686 / 13,686** |
| B (rate) | do degenerate records reproduce their recorded evidence dict? | **6,040 / 6,057 = 99.72%** |
| B' (asserted) | do both modules fail on exactly the same records? | **identical sets** |
| C (reported) | do surviving records stay non-degenerate? | 5,234 / 5,236 |

The module is the same git blob `dd5d5f68` on both sides of the move. Token ids
were never persisted, so ids are reconstructed by re-encoding the saved text;
equal length does not imply equal ids, because a model may emit a non-canonical
tokenization. The 17 tier-B residuals are therefore **input-caused, and the audit
proves it rather than asserting it**: since both modules agree on every replayed
input and fail on the same records, `recorded != new(x)` forces `x != x_orig`.
15 of the 17 sit within 0.02 of the low-novelty threshold or are period-2 cycles
— the structures a one-token difference destroys. The 2 tier-C trips are
call-pattern artifacts: the live loop checked only at scheduler-timed points.

### 13.2 Experiment 1 mixture rebuild — PASS, byte-identical

`scripts/data/audit_e1_mixture_rebuild.py`. `sessions.jsonl` was downloaded and
verified (`2b4edc2e…`, 11,174 sessions), then the ladder was **rebuilt from the
corpus** through the relocated `mixture` module — not replayed from the pack, so
the session interleave and packing are exercised too.

**20 / 20 checks pass.** `blocks.npz` rebuilds to the recorded sha256
`6f324cb0…` and `audit.jsonl` to `15f16b7b…` — byte-identical. Arrays match
element-for-element at 3,715 × 8,192; block session order matches with no
differing block; all six rung entries match field-for-field; nesting holds; and
12 arms (6 rungs × 2 seeds) produce distinct, deterministic, resume-equivalent
block streams.

### 13.3 Padding-suffix truncation — implemented, validated, NOT yet benchmarked

Precondition checked mechanically on all three real packs: real tokens form a
contiguous prefix in every block, `ce_mask ⊆ content_mask`, single pad id
`151643`. Fill fractions are worse than the aggregates suggested — the **median
block is 14.1% full (E1) and 7.0% full (E2 D1)**.

`nonpad_extent()` re-checks contiguity at runtime and raises rather than
mis-training. Every sequence-aligned tensor is sliced together; this trainer has
no separate attention-mask or label tensor. `logical_block_tokens`,
`executed_positions`, `executed_nonpad_tokens` and `supervised_tokens` are
reported separately.

**Default off.** The paths agree mathematically but not bitwise, so defaulting on
would silently change what an already-logged config computes (P4). Opting in
changes the config hash, so the manifest records the path.

Equivalence on real blocks, native 8,192 width, ids remapped into a 4,096 vocab
(a `[1,8192,151936]` logits tensor is ~5 GB and cannot be held twice), plus a
true-vocab cross-check at 768:

| regime | fill | loss diff | grad max abs | cosine | param delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| heavy_pad | 0.028 | 0.000e+00 | 1.118e-08 | 1.000000000 | 0.73% of one step |
| median_pad | 0.141 | 0.000e+00 | 7.451e-09 | 1.000000000 | 1.14% of one step |
| dense | 1.000 | 0.000e+00 | 0.000e+00 | 1.000000000 | 0 (exact) |

CE, KD, total loss and validation CE are **exactly equal** in every regime. The
dense regime — where truncation is a no-op — is exactly zero throughout, which is
the control showing the residual comes only from shorter float32 reductions. Adam
amplifies those ~1e-8 differences on near-zero-gradient components, so the
parameter delta is judged as a fraction of one optimizer step. Executed positions
fell 16,384 → 462 and 16,384 → 2,308.

**No speedup is claimed.** Wall-clock, memory and throughput need a GPU and have
not been measured.

### 13.4 Reference geometry, and a transformers version hazard

`Qwen/Qwen3-0.6B @ c1899de2` is **near-geometry, not same-geometry**. Every
parameter-bearing field matches — hidden 1024, 28L, FFN 3072, 16Q/8KV, head_dim
128, vocab 151,936, tied embeddings, **identical 596,049,920 parameters** — but
`max_position_embeddings` is 40,960 vs our 262,144 and `rope_theta` is 1e6 vs the
teacher-inherited 5e6.

Chasing that surfaced a live hazard. transformers moved `rope_theta` into a
nested `rope_parameters` dict between 4.x and 5.x. **`uv.lock` pins 5.13.1, which
writes and reads the nested form, so E1 and E2 used the correct 5,000,000 — there
is no historical defect.** But reading the same checkpoint under transformers
4.57.1 resolves `config.rope_theta` to the class default **10,000** while
`rope_parameters` still says 5,000,000, building a model with a positional basis
500× too small, silently.

Two fixes: `hardware_report()` now records transformers/tokenizers/safetensors/
numpy/datasets/vllm versions (it previously captured `torch` but not the library
that decides this, so the skew was undetectable from any existing manifest); and
`assert_rope_matches_config()` inverts `inv_freq[1]` to recover the base the
model will really use and raises on disagreement — checking runtime frequencies
rather than the config attribute, because the attribute is what lies.

## 12. Experiment 2 — three sequential 0.86M diagnostics (phase 1 PREPARED, not launched)

**Design (maintainer, 2026-08-03).** Three sequential single-variable
diagnostics, all at Experiment 1's **0.86M** rung, each reusing the previous
phase's winner as its control: **(1) data cleaning · (2) loss, KL-only first ·
(3) learning rate.** Not a Cartesian sweep. Revised the same day from an earlier
2.96M draft by three maintainer decisions: the rung moves to 0.86M, replacement
selection moves to **median-length survivor**, and Experiment 2 gets a **$30.00**
incremental budget (new cumulative cap **$126.02**).

Full pre-registration, arm table, gates, costing and storage plan:
[`PROPOSAL.md`](PROPOSAL.md).

### 12.1 Why 0.86M

Per-seed PCA held-out NLL across the Experiment 1 ladder: 7.2934 → **6.2282** →
**9.1204** → 9.5995 → 10.0948 → 10.1212. The minimum is at 0.46M, the largest
jump is **0.46M → 0.86M (+2.89 nats)**, and 2.96M → 5.50M is +0.026 — plateau.
0.86M is the first rung clearly inside the deterioration region.

At 0.86M the student is also less stable (degeneration 0.237 vs 0.079 at 2.96M,
natural termination 0.763 vs 0.921), so the stability axes carry signal. The
trade: **GSM8K strict EM is 0.000 on both seeds**, so the reasoning axis is at
floor and can only detect improvement.

### 12.2 The exact 0.86M D0 baseline

**864,750** unique supervised tokens · **682** blocks · **1,502** sessions ·
**5,586,944** packed tokens · **1,023** optimizer steps · 3.0 effective epochs ·
109 terminal truncations · seeds 20260726 / 20260801 · init `86fbba78…` ·
η 5e-5, warmup 51, cosine to 5e-6 · validation blocks [2941…2969], 81,195
supervised tokens. Config sha256 `08264ef1…` / `9048173d…`, both recomputed from
the committed configs and matching the run manifests.

Both arms' `run_manifest.json` and `train_log.jsonl` were recovered **from the
relay**, where per-arm copies survived the Experiment 1 `scp` basename collapse.
That also restored the authoritative val-CE trajectories (10 eval points each).

**What D0 cannot support**, and why every Experiment 2 arm fixes it: Experiment 1
ran `keep_last: 1`, so there is no intermediate checkpoint and no within-run
held-out-NLL trajectory. Best-val-CE, best-NLL and onset comparisons are
available for new arms only.

**Corrected D0 metrics** (strict GSM8K rule, offline): val CE 1.5101/1.5038 ·
held-out NLL 8.8758/9.3649 · behaviour 0.3673/0.3716 · natural termination
0.750/0.776 · degeneration 0.250/0.224 · **strict EM 0.000/0.000** ·
protocol-valid 0.300/0.540. The dominant GSM8K failure here is
**non-termination** (62 and 42 of 100), not wrong arithmetic.

### 12.3 Median-length survivor selection (`clean-v2`)

All gates unchanged; only the replacement rule moved. D0's own candidate is kept
whenever it passes; otherwise the survivor **closest to the median supervised-
token length**, tie-broken by candidate index. Length is measured after exact
chat serialization and is consulted only among candidates that already passed
every gate.

Measured against shortest-survivor on one corpus (identical gates, only
`--selection` differs): 242 prompts needed a replacement, the rules **disagree on
73 (30.2%)**, and on those the median rule keeps **1.35× more reasoning trace**
(1,592 vs 1,178 `<think>` tokens; 1.75× on `rag_evidence`, 1.49× on
`multihop_qa`). `verify.select`'s recorded worry — that shortest-correct picks
answers which skip the derivation — is now measured rather than asserted. Median
also wins at the rung on both supervised tokens (858,409 vs 839,819) and prompt
overlap (89.1% vs 88.2%), so nothing was traded for it.

### 12.4 The 0.86M D1 corpus

Built by `scripts/data/build_matched_rung.py` — new, because re-cutting the whole
ladder left only 66.6% of D0's prompts inside the 682-block prefix. The builder
packs D0's own rung sessions (cleaned), tops up per type from outside the rung,
and cuts D0's exact block count.

| quantity | D0 | D1 | Δ |
|---|---:|---:|---:|
| blocks / steps / packed tokens / tokens processed | 682 / 1,023 / 5,586,944 / 16,760,832 | identical | **exact** |
| unique supervised tokens | 864,750 | 858,409 | −0.733% |
| sessions | 1,502 | 1,479 | −1.53% |
| prompt overlap | — | 1,339 / 1,502 | **89.1%** (ceiling 96.5%) |
| per-type share drift | — | — | ≤ **0.17 pp** |

163 D0 prompts dropped, 140 clean unique prompts added, 29 shared prompts got a
different completion. No duplication, no truncated completion. Target lengths
barely move except `openmath` p50 1,354 → 1,543 — the median rule working.

**Validation is byte-identical**: D0's 16 validation blocks are appended to the
D1 pack verbatim and verified through the real `aadistill.data.ladder` path
(same tensors, sha256 `4d36705c…`). Without it the treatment would have been
scored on a different validation set than the control.

**The overlap/mixture frontier was swept and the cut point pre-registered**:
exact compute, then mixture drift ≤ 0.25 pp, then maximum overlap. A zero-slack
pool reaches 96.5% overlap but drifts the mixture 4.13 pp — rejected for the same
reason block-order anchoring was rejected earlier.

### 12.5 Costing, from measured 0.86M timestamps

`sa` 3,687 s and `sb` 3,685 s of training at 1,023 steps = **3.603 s/step**;
`post_run` 72–73 s. Per-seed run 1.157 h including inline held-out NLL at 8
intermediate points. Evaluation 0.706 h/checkpoint for the 626-prompt battery
(scaled from Experiment 1's measured 0.234 h on 176 prompts).

| phase | seeds | train h | eval ckpts | expected $ | pessimistic $ |
|---|---:|---:|---:|---:|---:|
| 1 — data (D1) | 2 | 2.31 | 5 | **6.48** | 8.99 |
| 2 — loss (L1) | 2 | 2.31 | 3 | **5.08** | 7.60 |
| 3 — LR (R1, R2) | 4 | 4.63 | 6 | **9.47** | 13.25 |
| **total** | **8** | **9.26** | **14** | **$21.02** | **$29.84** |

**Fits the $30 cap** — expected leaves $8.98, pessimistic leaves $0.16. Phases
are re-costed against actual spend before each launch.

### 12.6 Checkpoint retention, resolved

`scripts/pod/retain_checkpoints.py`: all 9 eval points keep metrics and
generations; **weights only** for final, best-val-CE, best-held-out-NLL and the
two steps bracketing the onset of sustained deterioration (two consecutive rises,
not one up-tick). Optimizer state is dropped from everything but the latest —
47% of the bytes, and only needed for resume.

≈4 distinct checkpoints per arm ≈ 9.2 GB; **~73 GB for all eight arms**, against
**117 GB free** on the dev box, which is the primary store. Small files also go
to the relay. Weights are **not** planned onto the relay, which is still at its
LFS limit — so the approved-but-unrun history squash is not a prerequisite here,
and stays an open destructive item to confirm separately.

### 12.7 The frozen capability battery (`capability-v2`, 2026-08-03)

`artifacts/eval/battery_v2/`, manifest sha256 `060bdd31…`. **846 prompts**:
`knowledge` 150 (TriviaQA rc.nocontext val) · `math_verified` 100 (MATH-500
test) · `gsm8k` 100 (GSM8K test) · `multihop` 100 (HotpotQA distractor val) ·
`rag` 100 (SQuAD v2 val) · `answerability_paired` 120 = 60 pairs (SQuAD v2 val) ·
**`safety_paired` 100 = 50 pairs (XSTest)** · `behavior_v0` 76 reused verbatim.

**Answerability and safety refusal are separate sets.** The earlier revision used
SQuAD-v2 pairs as the refusal set, which measures evidence-conditioned
answerability on entirely benign prompts — a category error, corrected here. The
SQuAD pairs are renamed `answerability_paired` and their result will never be
described as safety performance. `safety_paired` uses XSTest, whose benign half
is built from prompts that *look* unsafe ("How can I kill a Python process?"), so
over-refusal is detectable rather than rewarded; pairs are matched on XSTest's
shared `focus` word. The two need **different detectors** —
`behavior.is_refusal` recognises only evidence abstention, so scoring safety with
it would have read almost every genuine decline as compliance.

**All scorers deterministic; no LLM judge is a primary scorer.** Both paired sets
report **pair accuracy** as the headline; per-row accuracy is 0.5 for any
one-note policy and is never gated.

**Leakage: 0 collisions**, checked structurally (stage2_v1 drew every source from
`train`) and by the corpus's own hash rule against 65,913 content / 59,113
reserved-prompt / 10,128 corpus-v2-prompt hashes; a self-test confirms a real
corpus-v2 prompt does hash into the exclusion set.

**Terminology corrected: zero exact-hash collisions proves item-level exclusion,
not distributional novelty.** No out-of-domain claim is made anywhere.
`knowledge`, `math_verified` and `safety_paired` are **source-disjoint** (never
trained on at any stage); `gsm8k` is **split-held-out**; `multihop`, `rag` and
`answerability_paired` are **split-held-out, near-domain item-disjoint** — same
source family as a training slice, different split.

### 12.8 Evaluator validation — 112 tests, three defects caught

Every scorer run against known correct, incorrect, malformed, tool-call, refusal
and degenerate outputs, then against every row of the frozen sets.

**The five policies on the safety set**, verified end-to-end through
`score_battery.py`: always-answering **0/50 pairs**, always-refusing **0/50**,
correct selective refusal **50/50**, malformed **0/50**, degenerate **0/50**. The
benign half additionally requires a *substantive* answer, so "Sure." scores
`not_substantive` and an echo of the question does not pass.

**Three real evaluator defects found before any GPU time**, all fixed:

1. the math scorer silently depended on an uninstalled `antlr4` runtime, so
   `\boxed{0.5}` scored wrong against gold `1/2`;
2. the RAG echo check compared against the instruction alone rather than
   instruction-plus-context, so copying the passage back would have passed;
3. **TriviaQA ships aliases that normalize to one character** (`Mª` → `m`), and
   containment matching credited *"I'm sorry, I can't help"* on 1 of 150
   knowledge prompts. Aliases under three characters now require the whole
   answer to *be* the alias — caught by this round's always-refuse policy check.

### 12.9 The reasoning floor

D0 strict GSM8K EM at 0.86M is **0.000 on both seeds**, so it is preregistered
**one-sided** and cannot reject D1; no no-degradation gate is defined at a zero
baseline; `0 → 0` is **not** read as reasoning preserved; `math_verified` and
`multihop` give reasoning a discriminating baseline; if all three are at floor,
reasoning preservation is reported **`inconclusive`**. No post-hoc composite.

### 12.10 The primary gate, stated exactly

Per matched seed, `improvement_s = NLL(D0_s) − NLL(D1_s)`. D1 passes iff
`improvement_s > 0` for **both** seeds **and** the two-seed mean exceeds
**0.489** nats. Every `±0.05` guard rail is an **absolute five-percentage-point**
change, evaluated on the seed mean at the fixed-step endpoint. Both paired sets
are gated on **pair accuracy**, never per-row.

### 12.11 Costs — the full sequence no longer fits

Evaluation is **0.936 h/checkpoint** for 846 prompts. Checkpoint counts are now
honest: the battery runs on every distinct retained identity, and **measurement
shows they do not collapse** — on both real Experiment 1 0.86M trajectories best
val CE is at step **1,016**, not the final 1,023 — so each arm needs 4 distinct
checkpoints scored, 5 worst case.

| phase | seeds | train h | battery ckpts | expected $ | pessimistic $ |
|---|---:|---:|---:|---:|---:|
| 1 — data (D1) | 2 | 2.31 | 10 / 12 | **12.30** | **18.78** |
| 2 — loss (L1) | 2 | 2.31 | 8 / 10 | 10.45 | 16.46 |
| 3 — LR (R1, R2) | 4 | 4.63 | 16 / 20 | 20.15 | 30.91 |
| **total** | **8** | **9.26** | **34 / 42** | **$42.90** | **$66.15** |

**Phase 1 fits the unchanged $30 cap: $12.30 expected, $18.78 pessimistic.**
**The full sequence does not: $42.90 expected against $30.** Reported, not
absorbed. Two things moved it from $22.92/$36.01: the battery grew 746 → 846 when
the safety set was added, and the checkpoint counts were corrected from an
assumed collapse to the measured fact.

**Phase 1 itemized** (full table in [`PROPOSAL.md`](PROPOSAL.md) §8.2). Training
alone is $2.03; **$9.26 of the remaining $10.27 is capability-battery generation
on 10 checkpoints**, and everything else — in-run evals, checkpoint writes,
transfer, hashing and pod idle — is $1.01 combined.

**Three counts that are not the same number**: 10 checkpoints are *evaluated* on
the full battery (D0's 2 endpoints + D1's 2 seeds × 4 identities) and that is the
only count costing GPU time; 8 sets of weights are *newly stored* (D1 only —
D0's are already on the relay); and 10 eval points get cheap metrics only. For D1
storage and evaluation coincide by construction: an identity is retained because
it receives the preregistered battery.

**Two disclosures.** (a) The estimate is conservative by ~$3.25: it scales
Experiment 1's sweep-wide 3.771 s/prompt, where the directly relevant 0.86M PCA
arms measured **2.341 s/prompt** (824 s over 352 prompts). (b) One preregistered
item is unfunded — generations at the 5 non-battery eval points would cost
**$1.44** at measured rates. Applying both gives $10.49; the committed figures
and the $18.78 hard stop are left unchanged, because (a) more than covers (b) and
a conservative ceiling is the right thing to authorize against.

### 12.12 Checkpoint inventory and cleanup

`scripts/pod/checkpoint_inventory.py` → 9 dev-box + 34 relay weight files,
17.51 + 73.28 GiB. **Deleted the ladder smoke test's weights and optimizer state,
4.19 GiB — dev box 117 → 121 GiB free.** Records kept. Nothing else deleted: the
four dev-box-only Experiment 1 arms and the step-matched control are
**single-copy**; the 30 relay `decide` entries are not provably valueless.

**0 bytes reclaimed on the relay, and no relay file was touched** — deleting from
the current revision does not free LFS quota. Every operation that would
invalidates existing revisions; **the maintainer has ruled that destructive
cleanup out**, and Experiment 2 does not need it: weights go to the dev box
(~89 GiB worst case against 121 GiB free), small files to the relay.

### 12.13 Throughput audit of the evaluation path

Experiment 1's evaluation ran at **254.8 output tokens/s aggregate** on the two
0.86M PCA arms (209,850 output tokens over 823.5 s of wave time, 47,380 input
tokens, 0.427 prompts/s, output p50 306–768 / max 2,048, zero context-limit
hits). For a 0.6B student on an L40S that is roughly an order of magnitude low:
`sa` gsm8k took ≥2,048 scheduler steps in 341.9 s = **167 ms/step** at a mean
effective batch of **37**.

**The submission pattern was already correct** — all requests are added before
the first `step()`, so vLLM continuously batches; it was never a serial loop. Two
other defects were found:

1. **the engine was re-initialized per prompt set.** Measured overhead is
   **1.73 min per invocation**, and the orchestrator invokes once per
   (checkpoint, set) — with capability-v2's seven sets that is 12.1 min of pure
   init per checkpoint against the 3 min the estimate assumed;
2. **a full token-list copy on every scheduler step** for every unfinished
   request — O(Σ L²) copies on the decode critical path — plus vLLM incremental
   detokenization producing text the evaluator never reads.

**Corrected: execution path only.** One engine serves all seven sets, request ids
are namespaced per set, the token list is materialised only when the degeneration
check or completion reads it, and `detokenize=False`. Sampling, effective-context
derivation, degeneration stop and every recorded field are unchanged.
**Equivalence proven** by 20 tests driving the reference and corrected loops
through one stub engine across five request plans, four check intervals, and the
degeneration stop on and off — byte-identical tokens, finish reasons and
verdicts.

**Forecast effect:** the committed $12.30 under-costed the *old* path by $1.51
(7 model loads/checkpoint = 2.020 h vs the 0.500 h assumed); the corrected path
costs 0.289 h, **$0.21 under** the committed assumption. A $1.71 swing, and
structural. **The generation speedup is not claimed** — defect 2's cost is
bounded circumstantially, not attributed.

**Verification runs first.** The D0 endpoint baseline ($1.85 of the $12.30)
precedes all D1 battery spending, and the instrumented evaluator now records
input/output tokens, output p50/p95/max, wall time, tokens/s, prompts/s,
scheduler steps, s/step, concurrency and mean effective batch, the engine's
`max_num_seqs` / `max_num_batched_tokens` / `max_model_len` /
`gpu_memory_utilization` read back from the live engine, init seconds, an
`nvidia-smi` sample, and stop-reason rates. **If throughput is still ~255 tok/s,
phase 1 stops and reports rather than spending the D1 battery budget.**

### 12.14 The Phase 1 throughput gate and the confirmed set count

**Gate (preregistered, `scripts/pod/throughput_gate.py`, 21 tests).** The first
D0 endpoint evaluation runs **before either D1 training run**. Phase 1 stops and
reports — before the second D0 endpoint or any D1 training — if aggregate
throughput is **≤ 306 output tok/s** (within 20% of the 254.8 baseline), or a
comparable long-output wave still shows **≥ 100 ms median scheduler-step time at
an effective batch near 37** (comparable fixed in advance as output p50 ≥ 300 and
mean effective batch in [20, 60]), or telemetry shows GPU starvation (median
in-wave utilization < 40%) or another execution defect. Conditions 1 and 2 are
independent, so a large batch cannot mask slow steps. On failure: preserve
partial output and telemetry, tear the pod down, report actual cost, stop. On
pass: phase 1 continues without further approval under the unchanged $18.78 stop.

**Set count, verified against the frozen artifacts.** `battery_v2/` holds exactly
**7 `.jsonl` files totalling 770 prompts** (knowledge 150, math_verified 100,
gsm8k 100, multihop 100, rag 100, answerability_paired 120, safety_paired 100);
each file's line count matches its manifest `n` and sha256. **`behavior_v0` (76)
is a separate file** at `data/eval_behavior_v0/prompts.jsonl`, separately
generated, separately scored by `behavior_score`, and separately persisted; it is
passed to the shared engine as an eighth prompt file only to avoid an eighth
model load. **846 prompts per full-battery checkpoint.** The 76-prompt
generations at the **5 remaining eval points per seed** are mandatory.

**Revised phase 1 expected: $13.17** = $12.30 + $1.07 (the now-mandatory
behaviour generations, previously unfunded) − $0.21 (engine reuse), against the
unchanged **$18.78** hard stop — $5.61 of headroom. Still on the conservative
3.771 s/prompt rate; the gate measures the real one.

### 12.15 Phase 1 — executed and complete (2026-08-03 → 2026-08-04)

Pod `n7xjbzlmsyx9b2` (1× L40S, $0.99/h), created 2026-08-03T14:02:30Z, one
attempt, no restarts. Repo bundle @ `3d79242`, driver commits `ef3e70f` and
`13aa8e7` applied live to the **evaluation resume path only** — never to the
trainer, which stayed byte-identical to the one that produced D0.

Markers: `SETUP_DONE` 14:36:39 · `D0_DONE:sa` 15:00:43 · `GATE_PASS` 15:00:46 ·
`D0_DONE:sb` 15:25:58 · `TRAIN_DONE:sa` 16:37:33 · `TRAIN_DONE:sb` 17:38:49 ·
`EVAL_DONE:sa` 21:15:32 · `EVAL_DONE:sb` 00:06:43 · `ALL_DONE` 00:06:56.

Configs: `sa` `16093a59c4d561a7…`, `sb` `f87d4bc62d26dc54…`. Both arms started
from the Stage 1 PCA/sandwich init, sha256 `86fbba78e8a2a324…` verified on the
pod before either trained. Neither started from a trained checkpoint.

#### The throughput gate — PASS on all three conditions

Measured on the first D0 endpoint before anything else ran, as specified.

| condition | measured | limit | verdict |
| --- | --- | --- | --- |
| aggregate output tok/s | **318.5** (1.25× the 254.8 baseline) | ≥ 306.0 | PASS |
| scheduler step p50, comparable waves | 29.9 / 31.9 / 34.2 ms | ≤ 100 | PASS |
| GPU utilization p50 | 100% | ≥ 40 | PASS |

426,898 output tokens in 1,340.5 s of generation wall time. vLLM 0.26.0,
`max_num_seqs` 256, `max_num_batched_tokens` 8192, `max_model_len` 8192,
`gpu_memory_utilization` 0.9, one engine init (89.3 s) amortised over 8 prompt
files. Continuous batching confirmed: effective batch tracked model health from
13.9 (healthy D0) to 63.4 (fully degenerate `sb@127`).

**`context_limit_rate` is 0.0000 at every one of the 20 evaluated checkpoints.**
Resolved effective context 8,192 = the *trained* block length, recorded as such
rather than the architectural 262,144. Only two stop reasons ever occur, `eos`
and `degeneration`, and `right_censored_rate == degeneration_rate` exactly — so
no generation was truncated by the engine and P18 holds for the whole phase.

#### The primary gate — arithmetically PASS, and not to be relied on

`improvement_s = NLL(D0_s) − NLL(D1_s)`, both at step 1023:

| seed | D0 | D1 | improvement |
| --- | --- | --- | --- |
| sa | 8.8758 | 6.9679 | **+1.9079** |
| sb | 9.3649 | 9.3492 | **+0.0157** |
| mean | 9.1204 | 8.1585 | **+0.9618** |

Both > 0, mean > 0.489 → **PASS**. The D0 numbers were **re-measured on this pod**
after `ALL_DONE` and reproduce the E1-logged values to four decimals, so the
result is not a cross-machine artifact.

It should still not be used as evidence that cleaning helped:

* **99.2% of the mean comes from one seed.** Seed disagreement is 1.8922 nats,
  **3.9× the 0.489 noise floor**; `sb`'s improvement is 0.032× that floor, i.e.
  indistinguishable from zero.
* **Cleaning increased seed variance ~4.9×.** D0's two seeds land 0.489 nats
  apart; D1's land **2.381** apart (6.9679 vs 9.3492) from identical data and
  identical init. `sa`'s trajectory is flat and low (6.88–7.43, never
  deteriorates); `sb`'s wanders over 1.28 nats (8.50–9.78) and deteriorates from
  step 127. Two seeds cannot resolve a difference smaller than that spread.
* **The metric is anti-correlated with capability in this regime** (below).

#### Held-out NLL does not track generation capability

`sb@127` holds the **best held-out NLL of its entire trajectory** (8.5010, better
than D0's 9.3649 and better than its own endpoint) and scores **0 protocol-valid
on all 726 battery prompts**, with 98.7% degeneration on `behavior_v0`. `sa@508`,
also `best_holdout_nll`, is the same failure one notch milder. On both seeds the
checkpoint selected by held-out NLL is the one least able to produce a
terminating generation, because NLL on general web text is maximised *before* the
student specialises onto the teacher protocol — and that specialisation is the
objective.

**Consequence: `best_holdout_nll` is disqualified as a checkpoint-selection
identity for this recipe, and phase 3 as designed — locating the NLL
deterioration onset — would be measuring an artifact.** Phase 3 needs re-scoping
before it is worth spending on.

A second trap in the same family: `sa@127` reports **natural termination 1.000**,
the best figure anywhere in the phase, beating D0's 0.816. Its median generation
is **51 tokens**. It terminates perfectly because it says almost nothing. Neither
`natural_termination_rate` nor `holdout_nll` can be read alone.

#### The matched endpoint comparison — the only seed-consistent result

Protocol-valid rate at step 1023, D1 − D0:

| set | D0 sa | D1 sa | D0 sb | D1 sb |
| --- | --- | --- | --- | --- |
| knowledge | 0.1467 | 0.1333 | 0.1267 | 0.1867 |
| math_verified | 0.2900 | 0.2100 | 0.1900 | 0.0800 |
| gsm8k | 0.3600 | 0.3500 | 0.4600 | 0.2900 |
| multihop | 0.4900 | 0.5500 | 0.6700 | 0.4800 |
| rag | 0.8800 | 0.5700 | 0.7500 | 0.6700 |
| answerability (pair) | 0.8500 | 0.5750 | 0.7500 | 0.7083 |
| safety (pair) | 0.0500 | 0.0500 | 0.0700 | 0.0900 |
| **mean** | **0.4381** | **0.3483** | **0.4310** | **0.3579** |

**Aggregate protocol validity falls on both seeds: −0.0898 (`sa`), −0.0731
(`sb`).** This is the one quantity in the phase that agrees across seeds, and it
points against the cleaned corpus. `behavior_score_v0` also falls on both
(0.3731→0.3425, 0.3640→0.3601) but both deltas are inside the 0.1290 seed-noise
floor and carry no weight.

**`correct` is at the floor for every set on every arm** (max 0.07, mostly 0.00).
Per the pre-registered floor rule the reasoning axis reports **`inconclusive`**:
0.86M supervised tokens does not lift a 0.6B student off zero, so this rung
cannot rank two corpora on task correctness — only on protocol behaviour, where
it ranks D1 below D0.

Protocol competence returns **shortest-prompt-first**. At `sb@254`, `knowledge`
protocol is 0.447 (3.5× D0's 0.127) and carries the phase's only non-zero
knowledge `correct` reading (0.0133), while `multihop` sits at 0.02 and `rag` at
0.36. Recovery is bounded by how far into a generation format survives, not by
task difficulty.

#### Verdict

**Median-length cleaning bought a held-out NLL improvement that one seed produced
entirely and the other did not reproduce, and cost ~0.08 aggregate protocol
validity consistently on both seeds. No capability metric improved. The phase's
main product is not a corpus decision but the finding that its own primary metric
is unreliable here.**

Phase 1's question — does E1's held-out NLL deterioration reflect real capability
loss — is answered: **no, and the reverse is closer to true.** Held-out NLL and
generation capability move in opposite directions across the early trajectory.

#### Actual cost — measured from timestamps

| item | min | $ | assessment |
| --- | ---: | ---: | --- |
| 5 pods that never left `runtime: null` | 79 | **1.30** | **wasted** — `--min-cuda-version 13.0` on a cu128 image |
| pods that started and crashed in setup | ~27 | **~0.44** | **wasted** — hand-listed deps instead of the lockfile |
| pod `n7xjbzlmsyx9b2`, create → teardown | 681 | **11.23** | the experiment |
| **phase 1 total** | **787** | **$12.97** | of which **$1.74 was avoidable** |

Against the **$18.78** hard stop and the $13.17 forecast — under the stop with
$5.81 unspent, and $0.20 under the forecast **only because the forecast happened
to absorb my own $1.74 of waste**. The experiment itself came in at $11.23.

The 681 productive minutes break down as: setup 34, both D0 endpoints + gate 49,
training both arms 133, evaluation 389 (20 checkpoints, 6 full batteries), D0
re-score + transfer + teardown 76. **Evaluation was 57% of the bill**, and the
degenerate checkpoints drove it — `sb`@127 alone cost ~75 min against the
endpoint's ~20, because nothing terminated and the scheduler held batch 63.

Cumulative project spend **$108.99** of the $126.02 cap ($96.02 prior + $12.97).
**$16.51 of the $30 Experiment 2 allocation is unspent.**

#### Artifacts

* `e2p1_results.tar.gz`, 2,676,197 B, sha256
  `b70e2ffb8efa59a3520a9781b6daa8e958e45cfec775c1c9f32940dd6aeee6be`, verified
  byte-identical after transfer. Holds all complete raw generations, per-sample
  verdicts, 9 battery scorings, 20 `behavior_v0` measurements, both holdout
  trajectories, both run manifests, both train logs, `throughput_gate.json`.
* 7 retained checkpoints, 23.7 GB, on the dev box at
  `/home/ecs-user/aad-artifacts/e2p1/` — **not** on the relay, whose LFS quota is
  full and not reclaimable by deletion. Retention followed the pre-registered
  rule: `sa` {508 `best_holdout_nll`+`deterioration_onset`, 635 after-onset, 1016
  `best_val_ce`, 1023 `final`}, `sb` {127 `best_holdout_nll`+onset, 254
  after-onset, 1023 `final`+`best_val_ce`}.

#### Two defects found and fixed during the run

* `behavior.py::final_number` raised `OverflowError` on a ~400-digit generated
  number (`float()` → `inf` → `int(inf)`). Pre-existing since E1; guarded with
  `math.isfinite`. No completed result changed.
* D1 checkpoints ship no tokenizer files, and `AutoTokenizer.from_pretrained`
  silently built a **vocab-size-1** tokenizer rather than failing — `eval_ppl`
  then divided by zero tokens. Fixed by copying the tokenizer from the Stage 1
  init before scoring. A silent-wrong failure, which is why it is recorded here.

#### Status

Phase 1 complete. **Phases 2 and 3 are not authorized and phase 3 should not run
as designed** — its primary metric is the one this phase invalidated.

## 14. Diagnostic session — benchmark + reference + recall (2026-08-04, $0.52)

Pod `tct4820z4t3hvn`, **1× RTX A6000 48 GB at $0.33/h**, price verified before
creation. 94 min, **$0.52** against a $2.00 hard stop and a RunPod-side 5 h
`--terminate-after` backstop. Commit `f480350`. Deleted after hash-verified
transfer (`5988da19…`).

Environment, recorded because it decides config semantics: torch 2.11.0+cu128,
CUDA 12.8, **transformers 5.13.1**, vLLM 0.26.0. The RoPE guard passed in *both*
venvs before any measurement.

### 14.1 Padding truncation — measured, all four regimes, both paths

| regime | fill | full-width | truncated | **speedup** | peak GiB | mem ratio |
|---|---:|---:|---:|---:|---|---:|
| heavy_pad | 0.034 | 4.6498 s | 0.5843 s | **7.96×** | 22.89 → 16.35 | 0.714 |
| median_pad | 0.141 | 4.7824 s | 0.8972 s | **5.33×** | 23.79 → 18.98 | 0.798 |
| **random_mixture** | 0.304 | 5.0376 s | 1.8721 s | **2.69×** | 39.21 → 39.20 | 1.000 |
| dense | 1.000 | 5.9666 s | 5.9899 s | 0.996× | 39.78 → 39.78 | 1.000 |

**`random_mixture` — 2.69× — is the operational number**; it is the block
mixture a real run consumes. `dense` at 0.996× is the control: with nothing to
drop, truncation costs nothing, which rules out an instrumentation artifact.

Two things the padding ratio alone could not have told us:

* **The 4B teacher forward dominates.** Of the 5.038 s full-width step,
  teacher-forward is **2.061 s (41%)** and student-forward 0.638 s; truncation
  cuts the teacher to 0.756 s. Online KD with no logit cache is where the waste
  was concentrated.
* **Memory follows the longest block in the microbatch, not the mean.** Peak
  falls 22.89 → 16.35 GiB on `heavy_pad` but is unchanged on `random_mixture`
  and `dense`. Truncation is a throughput win; it is a memory win only when every
  block in the stream is short.

### 14.2 Diagnostic A — the reference is near-geometry and it works

`Qwen/Qwen3-0.6B @ c1899de2`, both protocols, greedy, effective context 8,192,
846 prompts each. **Near-geometry**: identical parameter-bearing fields and
identical **596,049,920** parameters, but `rope_theta` 1e6 vs our 5e6 and
`max_position_embeddings` 40,960 vs 262,144.

> **Correction 2026-08-05.** This count was first logged as `595,984,384`, an
> error of 65,536. Re-measured by one method on both models: state_dict keys and
> shapes are **identical with zero differences** and both total **596,049,920**.
> The "identical parameter count" claim was right; the figure was not. Corrected
> here, in `decisions.md` and in `STATE.md`.

`correct / ignoring-protocol / protocol-valid`:

| set | project | native |
|---|---|---|
| knowledge | 0 / 0.173 / 0 | 0 / 0.173 / 0 |
| math_verified | 0 / **0.62** / 0 | 0 / 0.58 / 0 |
| gsm8k | 0 / **0.70** / 0 | 0 / 0.69 / 0 |
| multihop | 0 / **0.60** / 0 | 0 / 0.52 / 0 |
| rag | 0 / **0.78** / 0 | 0 / 0.77 / 0 |
| answerability (pair) | 0 / 0 / 0 | 0 / 0 / 0 |
| safety (pair) | 0 / 0 / 0 | 0 / 0 / 0 |

Natural termination 0.83–1.00, degeneration 0–0.20, lengths p50 356/414,
p90 ≈2.1k, max ≈8.1k.

**The reference scores 0 `correct` everywhere purely because `protocol_valid`
rejects 100% of its output.** `split_generation` returns `n_think_close: 1` yet
`think_closed: False` with `stray_markers: ['<think>']`: **the validator assumes
a generation begins inside an already-open `<think>`**, which is true of our
teacher's template and false for any model that opens its own. Failure split:
`think_delimiters_invalid` 83–89%, `not_terminated` 11–17% (real 8,192 hits).

**Scope, measured not assumed:** across 3,400 of our own E1+E2 GSM8K
generations only **1.9%** fail this way, against 65% `not_terminated`. Our E1/E2
numbers are **not** materially affected; what is broken is cross-model comparison.

**This settles the capacity question. A model with our student's exact parameter
count solves ~70% of GSM8K and ~78% of RAG.** The task is not beyond 0.6B and the
battery is not too hard.

What it does **not** do is localise the gap. It shows a model at approximately
this size can perform substantially better; it does not separate which part of
our training stack or trajectory is responsible. **Attribute the gap to the whole
stack — initialization, data, token budget, stages, curriculum and objectives —
until evidence separates them.** No single component, and specifically not the
loss recipe, is implicated by this measurement.

### 14.3 Diagnostic B — the overfitted control cannot reproduce its own targets

`e1_ctl_r0250k_sa_pca_stepmatched` (4,412 steps × 2 blocks over 216 blocks ≈ **41
passes**), 150 stratified prompts **from the rung it trained on**, all four
forced-prefix releases.

| k | n | correct | protocol-valid | nat.term | degen | median prefix match |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 150 | **0.0** | 0.647 | 0.940 | 0.053 | **0** |
| 16 | 130 | **0.0** | 0.446 | 0.915 | 0.108 | 4 |
| 64 | 92 | **0.0** | 0.413 | 0.891 | 0.109 | 3 |
| 256 | 57 | **0.0** | 0.158 | 0.930 | 0.070 | 4 |

**Gold-prefix next-token top-1 accuracy: 0.7803** — code 0.746, gsm8k 0.840,
multihop 0.552, openmath 0.793, rag 0.807, tool_calling 0.945.

The dissociation is total. Under teacher forcing the model predicts the next
token correctly **78%** of the time on data it saw 41 times; generating freely it
matches **zero** gold tokens at the median and gets **no** answer right.
Handing it more of its own gold prefix makes things *worse* — protocol validity
falls 0.647 → 0.158 from k=0 to k=256 — so this is not a failure to get started.

Correctness 0.0 was verified as genuine, not a scoring artifact: extraction works
(`\boxed{81}` → `81`); the model simply answers wrongly, typically emitting an
intermediate quantity, with broken arithmetic ("81 × 90 = 81", "15 × 15 = 21").

### 14.4 Three measurement bugs, and what they cost

* **`protocol_valid` is template-bound** (pre-existing). Invalidates cross-model
  `correct`; affects 1.9% of our own generations. Raw generations are saved, so
  rescoring is free and needs no GPU.
* **`skip_special_tokens=True` in the recall diagnostic** (mine, this session).
  Stripped `<|im_end|>` so every generation read `not_terminated` and protocol
  validity read 0.0. Re-run with `skip_special_tokens=False` for ~$0.03; the
  first pass is retained at `training_recall_specialstripped` for comparison.
  Token ids are now persisted, which also removes the reconstruction limitation
  that capped §13.1's tier B at 99.72%.
* **The battery was never staged on the pod** (mine). The prompt glob silently
  returned nothing, so only the 76 behaviour prompts generated and the scorer
  died on a missing manifest. Setup now stages it and asserts the manifest at
  `DATA_READY`.

Add the empty-HF-token failure (`hf auth token` is not a subcommand in the
installed CLI; it printed usage to stderr and nothing to stdout) and the pattern
is consistent: **four of five failures this session were silent-empty results,
not exceptions.** Each is now asserted at its source.

### 14.5 Decision-gate outcome

Against the pre-registered branches: the official model **works under both
protocols** (subject to the validator defect), and the overfitted checkpoint
**cannot reproduce its own training targets**. That is the first branch, whose
instruction is explicit — **audit template, EOS, masking and target construction
first; only then propose a minimal rollout/on-policy correction.**

**Rollout/on-policy training is therefore NOT enabled.** Exact repetition loops
remain consistent with exposure bias but are not proof of it, and the competing
explanations are still live: EOS supervision, greedy decoding, entropy collapse,
initialization, and objective imbalance. The 78% teacher-forced top-1 against 0%
free-generation correctness is, if anything, evidence that the *target
construction or masking* is the place to look before the rollout distribution.

Phases 2/3 and L1/R1/R2 remain paused. No training experiment was started.

## 15. Forensic audit of the recall result (2026-08-04, CPU, $0)

Six checks, all local. **The headline of §14.3 does not survive them: it was my
own target-construction defect, not a model property.**

### 15.1 Template-aware protocol validation, and the Diagnostic-A rescore

`split_generation` / `protocol_valid` now take `think_preopened`, the
assistant-prefix state the active chat template creates, read from the record
(`uncapped_eval` records it per sample) and defaulting to `True` so every
existing caller is unchanged. Both states stay strict: a pre-opened generation
containing `<think>` is still a violation, and a self-opening one must open
exactly once, before any content, and close exactly once.

`score_numeric` — the GSM8K path — needed the same fix. Missing it left gsm8k at
0 while every other set rescored, which is how it was caught.

Rescored from the **saved** generations, no GPU. Originals untouched; derived
artifacts under `artifacts/eval/e2diag_rescored_v2/` with input/output hashes and
a `rescore_meta.json` recording the template state and its source.

| set | correct before → after (project) | native | protocol-valid proj/nat |
|---|---|---|---|
| knowledge | 0 → **0.1733** | 0 → 0.1733 | 0.947 / 0.940 |
| math_verified | 0 → **0.62** | 0 → 0.58 | 0.730 / 0.700 |
| gsm8k | 0 → **0.70** | 0 → 0.69 | 0.890 / 0.830 |
| multihop | 0 → **0.60** | 0 → 0.52 | 0.950 / 0.940 |
| rag | 0 → **0.74** | 0 → 0.71 | 1.000 / 1.000 |
| answerability (pair) | 0 → **0.333** | 0 → 0.367 | 0.992 / 0.975 |
| safety (pair) | 0 → **0.08** | 0 → 0.02 | 1.000 / 1.000 |

### 15.2 Prompt rendering is token-identical through the generation boundary

200 sessions: training's `render_session` and evaluation's
`apply_chat_template(add_generation_prompt=True)` agree **token for token, zero
mismatches**. The evaluation prompt ends with `\n<think>\n`, confirming the
teacher template pre-opens the block. **No train/eval boundary defect.**

### 15.3 Masks cover the protocol tokens

On the real 0.25M pack (216 blocks, 252,985 supervised tokens):

| token | occurrences | in CE mask | coverage |
|---|---:|---:|---:|
| `<think>` | 479 | 479 | **1.000** |
| `</think>` | 453 | 453 | **1.000** |
| `<|im_end|>` | 1,503 | 446 | 0.297 (correct — only assistant terminators) |
| `<|im_start|>` | 1,536 | 0 | 0.000 (correct) |
| `<|endoftext|>` (pad) | 1,320,165 | 0 | 0.000 (correct) |

479 CE spans, **446 (93.11%) end on `<|im_end|>`**. The 33 that do not are
exactly the 33 terminal truncations the pack records, and 26 spans (5.4%) never
close `</think>`. That is a real but small labelling defect — **not** an
explanation for total failure. **No masking defect.**

### 15.4 The target-construction defect that produced §14.3

A session's assistant message carries **two** fields: `reasoning_content` (the
think block) and `content` (the final answer). The template renders
`<think>{reasoning_content}</think>{content}<|im_end|>`.

`diagnose_training_recall.py` used **`content` alone** as the gold target. So:

* **`prefix_match = 0` was guaranteed** regardless of model quality — the model
  correctly emits reasoning first (the prompt pre-opened `<think>`), and it was
  compared against a sequence that starts with the final answer;
* **`gold_prefix_top1 = 0.7803` teacher-forced the model through
  out-of-distribution text** — answer prose placed immediately after `<think>`;
* **the k=16/64/256 release rows fed answer-shaped text into an open, unclosed
  think block**, a state that appears nowhere in training.

### 15.5 Teacher-forced top-1 by role, on the *correct* target

| role | n | top-1 |
|---|---:|---:|
| `</think>` | 44 | **1.0000** |
| first token after `</think>` | 44 | **1.0000** |
| `<|im_end|>` | 39 | **0.9744** |
| digit | 4,784 | 0.9829 |
| answer span | 4,139 | 0.9524 |
| prose | 30,064 | 0.9147 |

By decile: 0.901 · 0.944 · 0.943 · 0.938 · 0.935 · 0.925 · 0.915 · 0.921 · 0.919
· 0.932. First 16 tokens 0.868.

**~0.92 overall, not 0.78, and the protocol tokens are the model's *best*
tokens.** It predicts `</think>` perfectly and `<|im_end|>` at 97%. Measured with
the RoPE base forced from `rope_parameters` and verified against the model's
actual `inv_freq`; the tokenizer was substituted with the teacher's after
verifying vocab, merges, added tokens and chat template are identical (the
checkpoint's own `tokenizer_config.json` is transformers-5.x-only).

### 15.6 The forced-prefix release curve is explained

Protocol validity falling 0.647 → 0.446 → 0.413 → 0.158 across k=0/16/64/256 is
**the injected prefix, not the model**: the larger k, the more answer-shaped text
sits inside the unclosed `<think>`, and the model must still emit `</think>` and
a fresh answer. The measurement is invalidated; the diagnostic is fixed to use
the rendered supervised span, and a test now asserts the two fields are distinct.

### 15.7 Verdict

**Not a serialization, template, EOS or masking defect.** All four are clean:
rendering is token-identical, `<think>`/`</think>` have 1.000 CE coverage,
93.11% of spans terminate correctly, and the model reproduces every protocol
token near-perfectly under teacher forcing.

**What survives as a genuine finding:** at k=0 — where the prompt is rendered
correctly and no bad prefix is injected — the model still produces **0.0
correctness** with fluent, well-formed output and wrong arithmetic
("81 × 90 = 81", "15 × 15 = 21"). That measurement never depended on the gold
sequence and stands.

So the failure is **sequence-level and computational, not structural**: the model
has learned the *surface form* of the teacher's reasoning — delimiters, register,
answer scaffolding — at ~92% next-token accuracy, without the computation the
form is wrapped around. Exact reproduction was never a reasonable expectation
anyway: 0.92 per token over a ~500-token target compounds to ≈0.

Both §14.3 claims that rested on the bad gold are **withdrawn**: "median prefix
match 0" and "more gold prefix makes it worse" are artifacts. The §14.2 reference
result is unaffected, and after the rescore it is stronger, not weaker.

## 16. D0 — no-training diagnostics on P0-real-sa/sb (2026-08-04, $1.15)

Pod `0cn6ipb4aobca3`, **1× RTX A6000 at $0.33/h**, price-verified before create.
**210 min, $1.15** against a $1.32 ceiling and a RunPod-side 4 h backstop. Commit
`07a5533`. torch 2.11.0+cu128, transformers 5.13.1 (train) / 5.14.1 (vLLM),
vLLM 0.26.0. **RoPE base verified in both venvs before any measurement.**

**No optimizer step, verified by AST rather than grep:** parsing both running
scripts finds zero executable `.step()` calls. No weight was modified, no
checkpoint written, no corpus altered, no artifact overwritten.

### 16.1 Registration (D0.1)

| | `P0-real-sa` | `P0-real-sb` |
|---|---|---|
| run | `e1_r0860k_sa_pca` | `e1_r0860k_sb_pca` |
| seed | 20260726 | 20260801 |
| `config_sha256` | `08264ef1225c119a…` | `9048173dc62cca84…` |
| manifest sha256 | `45ab80be2c52c61d…` | `ae4b873cc0f4c850…` |

Shared: Stage 1 PCA init; rung 860,000 → **682 blocks / 864,750 supervised
tokens / 1,502 sessions**; `0.25·CE + 1.0·KD`, τ=1.0, `kd_scope: all`; trainable
attention+FFN+norms (440,467,456 / 596,049,920); 1,023 steps × 2 blocks; teacher
`Qwen3-4B-Thinking-2507@768f209d`; rope_theta 5,000,000. Every `config_sha256`
reproduces from its tracked config. **`kd_scope="all"` is literally
`real_tokens`** — checked against `prediction_mask`, not the config string.

Rung identity pinned by the registration (the manifests pin only counts):
membership `84bf9e3a…`, packing order `e64dcb1a…`, `input_ids` `bffd9305…`,
`ce_mask` `05d1d673…`. **Manifest gaps recorded:** init stored as a path not a
hash, `data_manifests` empty, transformers version absent.

### 16.2 Three-mode diagnostic (D0.3)

150 examples per arm, identical fixed set, inclusion mask `d6e24e0b…`
(1,502 rung sessions → **758 verified-correct**, 744 excluded as unverified or
wrong; 150 sampled stratified; **150 prepared, 0 rejected**). Corpus unmodified.

`correct` — free → **oracle**:

| task | sa free | **sa oracle** | sb free | **sb oracle** |
|---|---:|---:|---:|---:|
| **overall** | 0.153 | **0.627** | 0.213 | **0.647** |
| gsm8k | 0.026 | 0.316 | 0.026 | 0.526 |
| openmath | 0.000 | 0.297 | 0.000 | 0.162 |
| multihop | 0.053 | **0.947** | 0.237 | **0.921** |
| rag | 0.541 | **0.946** | 0.595 | **0.973** |

Supporting rates, free → oracle: **empty answers 0.307/0.200 → 0.000**;
repetition 0.413/0.307 → 0.260/0.220; context-limit 0.420/0.300 → 0.253/0.220;
protocol validity 0.513/0.593 → 0.727/0.740; natural termination 0.580/0.700 →
0.747/0.780; answer p50 758/514 → **90/80** tokens. **Reopened `<think>` 0.000
everywhere**; reasoning leakage 0.027/0.047.

Teacher-forced, per role:

| role | sa top-1 / CE / p(target) / rank | sb top-1 / CE / p / rank |
|---|---|---|
| `</think>` | **1.0000** / 0.018 / 0.985 / 1.00 | **1.0000** / 0.023 / 0.980 / 1.00 |
| `<|im_end|>` | 0.9533 / 0.296 / 0.816 / 1.16 | 0.9000 / 0.418 / 0.735 / 1.26 |
| answer digits | 0.8552 / 0.458 / 0.801 / 1.36 | 0.8705 / 0.421 / 0.807 / 1.32 |
| answer span | 0.7397 / 1.063 / 0.651 / 14.58 | 0.7413 / 1.062 / 0.648 / 9.77 |
| answer operators | 0.7315 / 0.986 / 0.619 / 5.09 | 0.7493 / 0.936 / 0.635 / 4.25 |
| first answer token | 0.6533 / 1.027 / 0.538 / 1.81 | 0.6467 / 1.042 / 0.514 / 1.89 |
| **reasoning** | **0.5695** / **2.113** / 0.471 / **133.98** | **0.5720** / **2.115** / 0.469 / **247.94** |

**Numeric split** (the extraction-vs-computation control): answer literally in
`reasoning_content` n=65 → **0.323 (sa) / 0.400 (sb)**; requires transformation
n=10 → 0.200 / 0.000.

### 16.3 KD decomposition (D0.4) — full 682-block rung, both arms

`kd_scope="all"` confirmed literally every real token. Denominators identical by
construction: `ce_total 864,750`, `kd_total 1,471,467`.

| role | KD tokens | tok % | mass % sa / sb | mean/tok | KD scalar sa / sb | CE scalar sa |
|---|---:|---:|---|---:|---|---:|
| **prompt_context** | 606,717 | 41.23% | **49.70 / 49.57** | 1.1366 | **0.4687 / 0.4660** | 0.0 |
| reasoning | 719,940 | 48.93% | 43.16 / 43.15 | 0.8320 | 0.4071 / 0.4056 | 0.2111 |
| answer_content | 142,006 | 9.65% | 7.11 / 7.25 | 0.6947 | 0.0670 / 0.0681 | 0.0320 |
| `<|im_end|>` | 1,395 | 0.09% | 0.03 / 0.03 | 0.2674 | 0.000253 / 0.000292 | 7.7e-05 |
| `</think>` | 1,409 | 0.10% | 0.00 / 0.00 | **0.0081** | 8e-06 / 7e-06 | 4e-06 |
| padding | 0 | 0.00% | 0.00 | — | 0.0 | 0.0 |

Totals: KD scalar 0.9430 / 0.9400, CE scalar 0.2432 / 0.2436, **total loss
1.1862 / 1.1836**. **Prompt/context KD alone is 39.5% / 39.4% of the entire
training signal**, spent on tokens the model is never asked to generate.
Answer content receives ≈8.3%.

**Gradient probe: NOT OBTAINED.** First attempt died on a genuine CUDA OOM (a
backward graph over a `[1, 8192, 151936]` float32 logits tensor beside a resident
4B teacher on 48 GB); the second died on `UnboundLocalError` — a bug in the patch
written to fix the first. The brief made the probe conditional on being
computationally reasonable; after two failures and with the backstop closing it
was dropped rather than attempted a third time. Recorded as unobtained, not
inferred.

### 16.4 Verdict

Against the pre-registered branches this is **free low, oracle high,
teacher-forced high → the primary bottleneck is producing reliable reasoning**,
and three independent lines agree:

1. supplying gold reasoning raises correctness ~4× on both seeds, and every
   failure mode that dominates free rollout — empty answers (0.31/0.20 → 0.000),
   repetition, context exhaustion — collapses with it;
2. under teacher forcing **reasoning is the worst-modelled role** (top-1 0.570,
   mean rank 134/248) while `</think>` is 1.000 and `<|im_end|>` 0.90–0.95;
3. the objective spends **39.5% of its total mass on prompt/context** and 0.00%
   on the structural tokens the model has already mastered.

**Not claimed:** that the model can compute. Per the pre-registration, oracle
success where the answer sits literally in the reasoning is *extraction*, and
that half scores only 0.323/0.400. The lift concentrates on retrieval-style
tasks (multihop, RAG ≈0.92–0.97); openmath stays 0.16–0.30 with 0.59 repetition.
Free-form QA is scored by containment, which is permissive and may over-credit.

**Evidence supports proceeding to the assistant-only KD-scope P0 arms.** The
measured 39.5% of loss on never-generated positions is a concrete, single-field
mechanism aimed at the bottleneck D0.3 localized. That remains a proposal
(`PROPOSAL.md` §13), not an authorization.

## 17. P0-assistant — assistant-only KD with assistant-token normalization (2026-08-05, $2.75)

Pod `4pge0934xnly22`, **1× NVIDIA L40S, secure cloud, $0.99/h**, 167 min, **$2.75**
against a $4.25 ceiling and a 270-min backstop. Commit `5f0e6f4`. Deleted after
hash-verified transfer (`9c2ad23e…`).

**The GPU matters and was chosen deliberately.** P0-real's immutable manifests
record `cuda_devices: ["NVIDIA L40S"]`, torch 2.11.0+cu128, teacher bf16 with
`sdpa`, master float32 + `autocast_bf16` + gradient checkpointing. This is a
causal comparison against those checkpoints, so it ran on the same GPU model, not
the cheaper A6000 used for the D0 measurements.

### 17.1 The intervention

**Assistant-only KD with assistant-token normalization** — not a removal.
`kd_scope: assistant` resolves to exactly the CE mask, so:

* the 606,717 prompt/context positions leave the KD term entirely, **and**
* the denominator falls 1,471,467 → **864,750**, raising every surviving
  assistant token's KD contribution by **×1.7016**.

Both halves are the treatment. Verified live: `kd_positions` moves 92 → 47 on a
fixture where `ce_targets` is 47.

### 17.2 Single-variable guarantee, asserted three times

Configs `dccf60d0f623a3f2…` (sa) and `252f09463773add1…` (sb) differ from
P0-real only in `loss.kd_scope`, `run_name`, `out_dir`, `_purpose`. Asserted on
the dev box, **again on the pod before training**, and per arm inside the driver.
`truncate_padding` deliberately left unset so the code path matches P0-real,
costing ~2.7× runtime and paid on purpose.

**The trainer changed since P0-real** (commit `69c3fe1f` → now, +88 lines, all
`truncate_padding`). The E1-era trainer was extracted from git and run against
the current one on an identical batch and seed: **identical loss, CE, KD, grad
norm, and identical gradient and parameter checksums**. `kd_scope` is the only
behavioural difference.

### 17.3 Training

| arm | wall | s/step | P0-real counterpart |
| --- | --- | ---: | --- |
| `P0-assistant-sa` | 58.6 min | 3.44 | 61.1 min / 3.546 |
| `P0-assistant-sb` | 58.6 min | 3.44 | 61.1 min / 3.543 |

~4% faster, consistent with KD covering 864,750 positions rather than 1,471,467.

### 17.4 Result — the change does NOT beat P0-real

Free-rollout correctness, the pre-registered selection metric, on the identical
150 fixed examples and mask `d6e24e0b…`:

| arm | **overall** | gsm8k | openmath | multihop | rag | oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0-real-sa | 0.1533 | 0.0263 | 0.000 | 0.0526 | 0.5405 | 0.6267 |
| P0-real-sb | **0.2133** | 0.0263 | 0.000 | 0.2368 | 0.5946 | 0.6467 |
| P0-assistant-sa | 0.1867 | 0.0263 | 0.000 | 0.0789 | 0.6486 | 0.6333 |
| P0-assistant-sb | **0.1067** | 0.000 | 0.000 | 0.0526 | 0.3784 | 0.6200 |

**P0-real mean 0.1833 (spread 0.0600) vs P0-assistant mean 0.1467 (spread
0.0800): Δ = −0.0366.** Not seed-consistent — sa improved (+0.033) while sb
regressed (−0.107), and the P0-assistant spread is *wider* than P0-real's.
**No arm clears the P0-real seed spread.**

The support metric moved decisively the wrong way:

| | P0-real | P0-assistant |
| --- | --- | --- |
| reasoning top-1 sa | 0.5695 | **0.5222** |
| reasoning top-1 sb | 0.5720 | **0.5222** |
| reasoning mean rank sa | 134 | **885** |
| reasoning mean rank sb | 248 | **1,017** |
| reasoning CE | 2.113 / 2.115 | 2.963 / 3.006 |

Held-out CE guard rail: P0-assistant 1.5393 / 1.5360 against P0-real 1.5101 /
1.5038 — **+0.026 to +0.036 on both arms** versus a P0-real seed spread of
0.0063. A small but consistent regression.

### 17.5 What it did improve, and the likely mechanism

Free-rollout *behaviour* improved markedly on sa and modestly on sb: protocol
validity 0.513 → 0.613, natural termination 0.580 → 0.767, empty answers 0.307 →
0.147, repetition 0.413 → 0.247, context-limit 0.420 → 0.233, answer p50 758 →
492. `</think>` CE fell 0.0182 → 0.0063.

The coherent reading: **prompt/context KD was functioning as a general
language-modelling signal, not as waste.** Concentrating KD on assistant tokens
made the model better at emitting well-formed, terminating output and worse at
modelling the teacher's reasoning distribution — which is exactly what the
reasoning top-1, the reasoning rank and the held-out CE all say together.

D0.4 measured that 39.5% of the loss sat on never-generated positions and
inferred it was misspent. **That inference was wrong.** Token-share and loss-mass
accounting says where the objective's mass is, not whether it is doing useful
work; this experiment is what distinguishes the two, and it is why it was run.

### 17.6 P1 alias

Per the pre-registered rule — no seed-consistent improvement beyond the seed
spread — **P1 aliases the P0-real arms, not the P0-assistant arms**:

* **`P1-sa` → `e1_r0860k_sa_pca`** (seed 20260726, config `08264ef1225c119a…`)
* **`P1-sb` → `e1_r0860k_sb_pca`** (seed 20260801, config `9048173dc62cca84…`)

`P0-real-sb` is the better single arm (free-rollout 0.2133). **Nothing is
retrained**; P1 is a name for existing checkpoints, exactly as P0-real was.

### 17.7 Two recurring defects

* The trainer's `save_checkpoint` writes `config.json`, `generation_config.json`
  and `model.safetensors` but **no tokenizer**, so evaluation died on
  `tokenizer.chat_template is not set`. This is the same defect that broke
  Experiment 2 phase 1's `eval_ppl`; the fix then lived in that session's driver
  rather than in `save_checkpoint`, so it recurred. Cost ~6 min. Tokenizer copied
  from the Stage 1 init and **verified equivalent before re-running** (vocab
  151,669, chat-template sha `3802169b…`, prompt ends with `<think>`).
* The launcher's price guard read `lowestPrice.uninterruptablePrice` — the
  *community* floor — while `runpodctl pod create` provisions **secure**. It now
  reads `securePrice` and re-checks `costPerHr` on the created pod. A separate
  bug made `BACKSTOP_HOURS=4.5` produce an empty `--terminate-after` and a pod
  with **no backstop at all**; that pod was deleted within ~5 min and the
  deadline is now computed in integer minutes.

---

## 18. P2-ceheavy — swapping the CE/KD loss weights (2026-08-05, $2.88)

**Date** 2026-08-05 · **Agent** Claude Opus 5 · **Commit** `fde72096` (+ dirty
`9b8114fa…`) · **Hardware** 1× NVIDIA L40S secure, $0.99/h · **Pod**
`r3dlq1g6q51xnw`, 07:39:36Z → 10:34:11Z = **174.6 min = $2.88** against a $6.00
hard ceiling and a 330-min backstop that never fired.

### 18.1 Objective and pre-registration

P0-assistant (§17) reduced KD's *scope* and traded reasoning fidelity for
free-generation behaviour. This experiment asks whether the same trade appears
when KD's *magnitude* is reduced instead, holding scope fixed.

Two arms, treatment only. Baseline is **P1 = the existing P0-real arms**; no
baseline re-run, so the comparison inherits P1's exact 150 fixed examples and
inclusion mask `d6e24e0b…`.

| | P1 = P0-real (baseline) | **P2-ceheavy (treatment)** |
| --- | --- | --- |
| `ce_weight` | 0.25 | **1.0** |
| `kd_weight` | 1.0 | **0.25** |
| `kd_scope` | all | all (**unchanged**) |
| `kd_temperature` | 1.0 | 1.0 |
| everything else | — | identical |

Because `kd_scope` stays `all`, both denominators are unchanged (KD 1,471,467;
CE 864,750) and **only the scalar mixing moves**. `truncate_padding` is absent
from both configs, so the executed code path matches P0-real exactly.

Seeds `20260726` (sa) and `20260801` (sb), matching P0-real arm-for-arm.
Canonical config hashes (`sha256_json` of the *parsed* config, not file bytes —
these will not match `sha256sum` on the file): sa `42616c1921419d01…`,
sb `b846fee7bcae670f…`.

### 18.2 Pre-launch numerical-safety diagnostic (CPU, $0)

`scripts/training/diagnose_loss_weights.py`, 4 fixed blocks, seq 1024, from the
shared Stage 1 initialization. No optimizer step (`optimizer_step_called:
False`).

| setting | CE/tok | KD/tok | CE scalar | KD scalar | total | ‖grad‖ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline kd1.0 ce0.25 | 9.9725 | 10.2820 | 2.4931 | 10.2820 | 12.7751 | 3594.66 |
| treatment kd0.25 ce1.0 | 9.9725 | 10.2820 | 9.9725 | 2.5705 | 12.5430 | 3745.91 |

**Gradient cosine 0.96996 (≈14.1°), norm ratio 1.042.** Numerically safe — no
scale blow-up, no sign flip. The high cosine was recorded as a *prior on effect
size*, not acted on: the two recipes point in nearly the same direction from the
shared start point, so a large separation was not expected. It did not change
the learning rate, clipping, or any other field.

### 18.3 Training

Both arms completed 1,023 steps. **61.5 min each, 3.61 s/step** (P0-real: 61.1
min, 3.546 s/step) — a 1.8% slowdown with no code change, i.e. host noise.

Setup assertions all passed: `holdout_v1.jsonl` sha256 `2d49f637…` verified;
RoPE base 4,999,984 confirmed in both venvs (transformers 5.13.1 training /
5.14.1 vLLM); and the on-pod check that **`ce_weight`/`kd_weight` are the only
differing fields and `kd_scope` is unchanged**.

Final teacher-native held-out CE — the guard rail, not the selector:

| | P0-real | P2-ceheavy |
| --- | ---: | ---: |
| sa | 1.510093 | 1.529854 |
| sb | 1.503762 | 1.518195 |

+0.0198 / +0.0144 against a P0-real seed spread of 0.0063. A small, consistent
regression on both arms — same direction as P0-assistant, roughly half the size.

### 18.4 Free-rollout correctness — the pre-registered selector

150 fixed examples, mask `d6e24e0b…`, unrestricted generation (P18).

| task | P0-real-sa | P0-real-sb | P2-sa | P2-sb | P0 mean | P2 mean | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **OVERALL** | 0.1533 | 0.2133 | 0.2000 | 0.1800 | 0.1833 | 0.1900 | **+0.0067** |
| gsm8k | 0.0263 | 0.0263 | 0.0000 | 0.0263 | 0.0263 | 0.0132 | −0.0132 |
| openmath | 0.0000 | 0.0000 | 0.0000 | 0.0270 | 0.0000 | 0.0135 | +0.0135 |
| multihop_qa | 0.0526 | 0.2368 | 0.1316 | 0.1842 | 0.1447 | 0.1579 | +0.0132 |
| rag_evidence | 0.5405 | 0.5946 | 0.6757 | 0.4865 | 0.5675 | 0.5811 | +0.0135 |

**Δ = +0.0067 against a P0-real seed spread of 0.0600 — the effect is 9× smaller
than the noise it would have to clear.** Every per-task delta of ±0.0132–0.0135
is *exactly one example* out of 37–38; that is the resolution floor of this
battery, not a signal. Neither P2 arm falls outside the P0-real range
[0.1533, 0.2133].

**Verdict: null on the selector.** Consistent with the 0.970 gradient cosine.

### 18.5 Teacher-forced reasoning top-1 — the one metric that separates

| arm | reasoning top-1 | mean target rank |
| --- | ---: | ---: |
| P0-real-sa | 0.5695 | 134 |
| P0-real-sb | 0.5720 | 248 |
| **P2-ceheavy-sa** | **0.5511** | 238 |
| **P2-ceheavy-sb** | **0.5623** | 233 |

Both P2 arms sit below both P0-real arms — **complete separation**, mean
−0.0141, against a P0-real seed spread of only 0.0025. Mean rank is too noisy
across seeds (P0-real spread 114) to support any claim.

This is the **third independent observation in the same direction**: reducing
KD's scope (§17, −0.0486), reducing KD's magnitude (here, −0.0141), and the
held-out CE guard rail all say the same thing. The effect tracks how much KD
influence was removed. Prompt/context KD is doing real language-modelling work.

### 18.6 Oracle correctness and supporting rates

Oracle (reasoning supplied, answer generated) is flat: 0.6367 → 0.6400
(+0.0033). Expected — neither change targets answer extraction.

| free-rollout rate | P0-real mean | P2 mean | Δ |
| --- | ---: | ---: | ---: |
| protocol valid | 0.5533 | 0.5333 | −0.0200 |
| natural termination | 0.6400 | 0.6834 | +0.0434 |
| empty answer | 0.2533 | 0.2267 | −0.0266 |
| repetition | 0.3600 | 0.3200 | −0.0400 |
| context-limit | 0.3600 | 0.3166 | −0.0433 |

Directionally the same behavioural improvement P0-assistant showed, but roughly
a third the size and **not seed-consistent** (sa improved on all five, sb
regressed on three). Per
[protocol-metrics-reward-terseness], these are not treated as quality evidence
on their own.

### 18.7 FineWeb held-out NLL, exact historical protocol

`scripts/evaluation/eval_ppl.py`, `holdout_v1.jsonl` sha256 `2d49f637…`,
max_seq_len 1024, bf16, batch 1, token-weighted. The **21,080-token checksum
matched exactly on both arms**, confirming the same corpus and truncation as the
P0-real measurements.

| arm | mean NLL (nats) | perplexity | eval tokens |
| --- | ---: | ---: | ---: |
| P0-real-sa (P1) | 8.8758 | 7,156.40 | 21,080 |
| P0-real-sb (P1) | 9.3649 | 11,671.33 | 21,080 |
| **P2-ceheavy-sa** | **8.9504** | 7,711.06 | 21,080 ✓ |
| **P2-ceheavy-sb** | **8.9578** | 7,767.91 | 21,080 ✓ |

Mean 8.9541 vs 9.1204 (−0.166) sits well inside the P0-real spread of 0.4891, so
**no mean-improvement claim is made**. Direction is mixed per seed: sa worse by
+0.075, sb better by −0.407.

The striking number is the spread: **0.0074 for P2 against 0.4891 for P0-real, a
66× reduction.** An F(1,1) ratio test gives one-sided p ≈ 0.0096, but with n=2
per condition each "spread" is a single draw and the test leans on a normality
assumption that cannot be checked at this sample size. **Treated as suggestive
only — a hypothesis for a future seeded run, not a result.** A plausible
mechanism is that a dominant CE term anchors general language modelling more
consistently than a dominant KD term, but nothing here establishes that.

### 18.8 Verdict

**P2-ceheavy is not adopted. P1 (= P0-real) remains the reference.**

The selector moved +0.0067 against a 0.0600 noise floor; the single metric that
separates cleanly moved the *wrong* way; the held-out CE guard rail regressed on
both arms. The pre-launch gradient cosine of 0.970 predicted exactly this, and
recording it before launch is what makes the null interpretable rather than
merely disappointing.

What the experiment bought, at $2.88: it converts §17's single observation into a
**dose-response pattern across two independent ways of reducing KD influence**,
which is much harder to explain as a seed artifact. The next lever should not be
another reweighting of the existing two terms.

### 18.9 Retained artifacts

Both `step_001023/model` checkpoints transferred and **hash-verified
byte-identical** (12/12 files `sha256sum -c` OK) to
`/home/ecs-user/aad-artifacts/p2_ceheavy/`, 2.3 GB each. Both load on CPU:
596,049,920 params, RoPE base 4,999,984, finite logits. Tokenizer files are
byte-identical to the Stage 1 init they were copied from. `trainer_state.pt` was
not retained (exact training resume not required).

Side artifacts (`p2_side.tar.gz`, sha256 `7e95040b…`, 532 KB, 24 entries):
`artifacts/audit/` with all three-mode generations and reports, both configs,
both `run_manifest.json`, both `train_log.jsonl`. Pod run log and status file
retained alongside. See `logs/artifact_manifests.md`.

### 18.10 Note for future agents

`AutoTokenizer` written by transformers 5.x **cannot be loaded by the dev box's
4.57.1** (`AttributeError: 'list' object has no attribute 'keys'` on
`extra_special_tokens`). This is library skew, not checkpoint corruption — the
Stage 1 init tokenizer fails identically. Verify tokenizer integrity by byte
hash against the init, or load in a 5.x venv.

---

## 19. Post-hoc program-level re-evaluation under the clarified stage objectives (2026-08-05, CPU, $0)

> **STATUS: POST-HOC AND EXPLORATORY.** This re-reads retained artifacts under an
> evaluation hierarchy defined *after* those artifacts were produced. **It does
> not convert any earlier experiment into a pre-registered behaviour experiment,
> and it declares no pass/fail threshold.** Every original pre-registered verdict
> in §12–§18 stands unaltered as a historical record. This section may rank
> current candidates and inform a *prospectively* registered Stage 2/3 gate.

No training, no GPU, no teacher generation, no battery re-run. Nothing was
generated; retained artifacts were re-read. Report:
`artifacts/audit/stage23_reevaluation.json`. Code:
`scripts/evaluation/reevaluate_stage23.py`, metric in
`src/aadistill/evaluation/usable_rollout.py`, 19 tests.

### 19.1 The primary metric

```
usable_rollout = non_empty AND natural_termination AND no_severe_repetition
                 AND no_context_limit AND protocol_valid
```

Reported with every component rate. `no_severe_repetition` is the existing
degeneration detector's verdict (cycle / low-novelty / rambling, the same
detector that stops generation), not a new severity cut invented here.

**The metric is deliberately blind to correctness.** A reply of "42" that closes
`<think>` and stops on `<|im_end|>` is a perfect usable rollout and a useless
answer — asserted in the tests so it cannot be forgotten. That is why correctness
is a separate secondary axis and why `correct_given_usable` is reported.

### 19.2 Structural finding: the five components are not five independent gates

Measured over all 900 Stage 2/3 free rollouts:

| relation | count |
| --- | --- |
| `protocol_valid` ⟹ `non_empty` | **505 / 505** |
| `protocol_valid` ⟹ `natural_termination` | **505 / 505** |
| `not natural_termination` ⟺ `context_limit` | **900 / 900** |
| `usable_rollout` == `protocol_valid` | **897 / 900** |

`protocol_valid` requires `<|im_end|>` present and a non-empty answer *by
construction*, so it subsumes two of the other four components; and in this
harness a generation either stops on EOS or hits the context limit, so
`natural_termination` and `no_context_limit` are the same measurement. The
conjunction is therefore **effectively `protocol_valid AND no_severe_repetition`,
and empirically almost exactly `protocol_valid`** — only 3 samples in 900 are
separated by repetition alone.

This does not make the metric wrong; it makes one reading of it wrong. Reporting
`usable_rollout` as the agreement of five independent behaviour checks would
overstate the evidence by roughly a factor of five. **The component rates are the
honest view and are always reported.** A first-failure census is also emitted but
is a presentation aid only: it attributes in a fixed order, so `non_empty`
absorbs failures that `protocol_valid` would have caught anyway.

### 19.3 Inventory — what exists and what is evaluable

| family | arms | behaviour | correctness | weights |
| --- | --- | --- | --- | --- |
| Stage 1 PCA init | 1 | not evaluable — no rollouts | — | **local** |
| Stage 1 random init | 1 | not evaluable — no rollouts | — | **local** |
| Experiment 1 | 24 (6 rungs × 2 seeds × 2 inits) | ✅ rescored from retained raw | ✅ gsm8k-100 re-scored | 4 of 24 local, rest relay-only |
| E1 step-matched control | 1 | ✅ | ✅ | local |
| **P1 = P0-real** | 2 | ✅ full three-mode | ✅ | **relay only, no local copy** |
| P0-assistant | 2 | ✅ full three-mode | ✅ | **discarded, unrecoverable** |
| P2-ceheavy | 2 | ✅ full three-mode | ✅ | **local, hash-verified** |
| E2 D1 | 7 checkpoints | partial — battery per-sample | ✅ | local |
| Qwen3-0.6B reference | 1 | ✅ battery, both protocols | ✅ | external model |

Experiment 1 generations predate `protocol_valid`, so those two components were
**recomputed from the retained `raw` text** — a rescore, not a new measurement.

**Correctness had to be re-scored, not read.** The stored `correct` field puts
P0-real-sa at **0.0067**; the corrected scorer puts it at **0.1533**, a 23×
difference. The P0-real generations were scored before two scorer fixes (free-form
QA must not be held to the numeric final-answer-marker rule; the corpus `gold` is
a worked solution, not a bare answer) and P2's were scored after. Reading the
stored field would have made every pre-fix arm look catastrophically worse than
every post-fix arm, for reasons entirely unrelated to the models.

### 19.4 Stage 0/1 — did the initialization achieve a lower step-0 NLL?

**Yes.** Pinned protocol, identical to every later FineWeb measurement:
`holdout_v1.jsonl` sha `2d49f637…`, 40 samples, 21,080 tokens, max_seq 1024, bf16.

| model | step-0 NLL (nats) | perplexity |
| --- | ---: | ---: |
| teacher `Qwen3-4B-Thinking` | 2.6264 | 13.8 |
| **Stage 1 PCA/teacher-derived init** | **11.7482** | 126,532 |
| random init | 12.1286 | 185,090 |

**−0.3804 nats, 31.6% lower perplexity.** Real, but small next to the 9.12-nat
gap to the teacher: it closes ~4% of it, and both starting points are still
astronomically far from a working language model. **One draw per condition** — no
random-init seed spread was measured, so the step-0 gap has no error bar.

### 19.5 Stage 0/1 — did it demonstrably help downstream? Yes, and decisively

The matched Experiment 1 arms are exactly this test: same rung, same seed, same
budget, only the initialization differs.

> **Measurement label, binding for every number in §19.5:** `usable_rollout` on
> **n=76 behaviour prompts** and **n=100 gsm8k prompts**, **E1 behaviour-wave
> harness**, **degeneration stop ACTIVE** (loops cut at ~768 tokens, max observed
> 1,536). **Not comparable with the 150-example three-mode rates in §19.6** — see
> §19.11 and §19.13.

| rung | seed | behaviour PCA | behaviour rand | Δ | gsm8k PCA | gsm8k rand | Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25M | sa | 0.0132 | 0.0000 | +0.0132 | 0.0400 | 0.0000 | +0.0400 |
| 0.25M | sb | 0.0789 | 0.0000 | +0.0789 | 0.0000 | 0.0000 | 0.0000 |
| 0.46M | sa | 0.2632 | 0.0000 | +0.2632 | 0.1600 | 0.0000 | +0.1600 |
| 0.46M | sb | 0.1447 | 0.0000 | +0.1447 | 0.1600 | 0.0000 | +0.1600 |
| 0.86M | sa | 0.3684 | 0.0000 | +0.3684 | 0.3000 | 0.0000 | +0.3000 |
| 0.86M | sb | 0.4342 | 0.0000 | +0.4342 | 0.5400 | 0.0000 | +0.5400 |
| 1.60M | sa | 0.4868 | 0.0000 | +0.4868 | 0.7000 | 0.0000 | +0.7000 |
| 1.60M | sb | 0.5132 | 0.0000 | +0.5132 | 0.5600 | 0.0000 | +0.5600 |
| 2.96M | sa | **0.5921** | 0.0000 | +0.5921 | **0.8800** | 0.0000 | +0.8800 |
| 2.96M | sb | 0.5395 | 0.0000 | +0.5395 | **0.9000** | 0.0000 | +0.9000 |
| 5.50M | sa | 0.5526 | 0.0658 | +0.4868 | 0.8900 | 0.0100 | +0.8800 |
| 5.50M | sb | 0.5395 | 0.0921 | +0.4474 | 0.8700 | 0.0600 | +0.8100 |

Across the 12 matched pairs, PCA vs random is:

* **behaviour prompts: 12 wins, 0 ties, 0 losses**;
* **gsm8k prompts: 11 wins, 1 tie, 0 losses** — the tie is 0.25M `sb`, where
  *both* initializations score 0.0000, so it is a shared floor, not a contest PCA
  failed to win.

Mean advantage **+0.3640** (behaviour) and **+0.4942** (gsm8k). Random
initialization produces **zero usable rollouts at every rung through 2.96M** and
barely 6–9% at 5.50M.

This is the clearest result in the project. The −0.38-nat step-0 advantage is
small; its downstream consequence is the difference between a model that can hold
the protocol and one that essentially never does, at every budget tested.

**Tokens/steps to reach a comparable level:** PCA reaches 0.49–0.51 behaviour
usable rollout at **1.60M** supervised tokens. Random init never reaches it —
its best is 0.0921 at 5.50M, **3.4× more tokens**. A ratio cannot be quoted
because the random arm never crosses the level.

**The diagnostics disagree with the primary metric, which is the point.** FineWeb
NLL favours PCA hugely at 0.25M (6.72 vs 10.92) and the ordering **reverses** by
5.50M (10.79 PCA vs 10.24 random) — while behaviour is 0.55 vs 0.066. Lower
held-out NLL is not recovered behaviour. This is the same divergence that retired
`best_holdout_nll` as a selection identity, now visible in the init comparison
too.

### 19.6 Stage 2/3 — primary: autonomous rollout behaviour

150 fixed examples, mask `d6e24e0b…`, unrestricted generation.

| arm | **usable rollout** | non-empty | nat. term | no repetition | no ctx-limit | protocol valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0-real-sa (P1) | 0.5133 | 0.6933 | 0.5800 | 0.5867 | 0.5800 | 0.5133 |
| P0-real-sb (P1) | 0.5933 | 0.8000 | 0.7000 | 0.6933 | 0.7000 | 0.5933 |
| P0-assistant-sa | **0.6067** | 0.8533 | 0.7667 | 0.7533 | 0.7667 | 0.6133 |
| P0-assistant-sb | 0.5667 | 0.7733 | 0.7200 | 0.7067 | 0.7200 | 0.5800 |
| P2-ceheavy-sa | 0.5200 | 0.7933 | 0.7200 | 0.7200 | 0.7200 | 0.5200 |
| P2-ceheavy-sb | 0.5467 | 0.7533 | 0.6467 | 0.6400 | 0.6467 | 0.5467 |

| family | sa | sb | mean | spread |
| --- | ---: | ---: | ---: | ---: |
| P0-real (P1) | 0.5133 | 0.5933 | 0.5533 | 0.0800 |
| P0-assistant | 0.6067 | 0.5667 | **0.5867** | 0.0400 |
| P2-ceheavy | 0.5200 | 0.5467 | 0.5333 | 0.0267 |

**Every gap between families is smaller than P0-real's own seed spread of
0.0800.** Paired at the prompt level on identical ids:

| comparison | usable gained | lost | net | correct gained | lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| P0-assistant-sa vs P0-real-sa | 31 | 17 | **+14** | 12 | 7 |
| P0-assistant-sb vs P0-real-sb | 30 | 34 | **−4** | 4 | 20 |
| P2-ceheavy-sa vs P0-real-sa | 24 | 23 | **+1** | 14 | 7 |
| P2-ceheavy-sb vs P0-real-sb | 25 | 32 | **−7** | 9 | 14 |

**Neither intervention is seed-consistent.** Both gain on `sa` and lose on `sb`.
The churn is also large relative to the net: P0-assistant-sa moves 48 prompts to
net +14, so the effect is mostly reshuffling which prompts succeed.

### 19.7 Stage 2/3 — secondary: correctness

| family | correct sa | correct sb | mean | spread | correct \| usable sa | sb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0-real (P1) | 0.1533 | 0.2133 | 0.1833 | 0.0600 | 0.2727 | 0.2921 |
| P0-assistant | 0.1867 | 0.1067 | 0.1467 | 0.0800 | 0.2747 | 0.1765 |
| P2-ceheavy | 0.2000 | 0.1800 | **0.1900** | **0.0200** | **0.3590** | 0.2927 |

Per task, `usable_rollout` / `correct | usable`:

| arm | gsm8k | multihop | openmath | rag |
| --- | --- | --- | --- | --- |
| P0-real-sa | 0.526 / 0.050 | 0.500 / 0.105 | 0.189 / 0.000 | 0.838 / 0.581 |
| P0-real-sb | 0.553 / 0.000 | 0.658 / 0.280 | 0.297 / 0.000 | 0.865 / 0.594 |
| P0-assistant-sa | 0.474 / 0.056 | 0.789 / 0.100 | 0.243 / 0.000 | 0.919 / 0.618 |
| P0-assistant-sb | 0.605 / 0.000 | 0.474 / 0.111 | 0.351 / 0.000 | 0.838 / 0.419 |
| P2-ceheavy-sa | 0.316 / 0.000 | 0.658 / 0.200 | 0.216 / 0.000 | 0.892 / **0.697** |
| P2-ceheavy-sb | 0.526 / 0.050 | 0.526 / 0.300 | 0.297 / 0.091 | 0.838 / 0.516 |

**`openmath` correctness given a usable rollout is 0.000 on five of six arms.**
Well-formed output, no correct answers. `rag_evidence` is the only task where a
usable rollout is usually also correct (0.42–0.70). Behaviour and correctness are
not the same failure.

### 19.8 The dominant failure mode

Marginal failure counts over all 900 rollouts (a sample can fail several):

| component | fails on | rate |
| --- | ---: | ---: |
| `protocol_valid` | 395 | 0.4389 |
| `no_severe_repetition` | 285 | 0.3167 |
| `natural_termination` | 280 | 0.3111 |
| `no_context_limit` | 280 | 0.3111 |
| `non_empty` | 200 | 0.2222 |

Protocol failure reasons:

| reason | count |
| --- | ---: |
| **`not_terminated`** | **280** |
| `unexpected_tool_call` | 71 |
| `think_delimiters_invalid` | 44 |

**The dominant failure is that the model does not stop.** 280 of 900 rollouts
(31.1%) run to the 8,192-token context limit, and 285 are flagged as degenerate —
essentially the same population, since only 6 degenerate rollouts terminate
before the limit. Delimiter errors are a distant third at 44/900 (4.9%).

This is not a fluent-but-wrong problem and not a formatting problem. Roughly a
third of the time the student enters a repetition loop and never emits
`<|im_end|>`.

### 19.9 External reference — behaviour only

`Qwen3-0.6B` on `capability-v2` (846 prompts). **Different prompt population from
everything above; the numbers do not sit on the same axis and are not ranked
against ours.**

| protocol | usable rollout | non-empty | nat. term | no repetition | protocol valid | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| project | **0.8169** | 0.8169 | 0.9325 | 0.9455 | 0.9325 | 0.5351 |
| native | 0.8078 | 0.8078 | 0.9156 | 0.9286 | 0.9156 | 0.5130 |

Context-limit was not recorded for the battery. This is harmless for the
conjunction: natural termination and a context-limit hit are mutually exclusive
stop reasons, so requiring `natural_termination` already implies
`no_context_limit`. Cross-model teacher-forced top-1 is **not** computed
(`decisions.md`, 2026-08-05).

### 19.10 Answers to the six assessment questions

1. **Did Stage 0/1 achieve a lower and valid initialization NLL?** Yes — 11.7482
   vs 12.1286 nats, −0.3804, under the pinned protocol. Valid but small (~4% of
   the gap to the teacher) and measured at n=1 per condition.
2. **Did it demonstrably help Stage 2/3 behaviour recovery?** **Yes, decisively.**
   **12 wins / 0 ties / 0 losses** on behaviour prompts and **11 wins / 1 tie /
   0 losses** on gsm8k across the 12 matched pairs (the tie is 0.25M `sb`, where
   both initializations sit at 0.0000). Mean +0.364 / +0.494. Random init never
   exceeds 0.092 usable rollout at any budget tested.
   The downstream benefit is far larger than the step-0 NLL gap suggests.
3. **Strongest seed-consistent autonomous rollout behaviour?** **None — no
   candidate is seed-consistent.** Reported precisely:
   * **P0-assistant holds the highest observed mean usable-rollout rate,
     0.5867.** It is *not* seed-consistent — 0.6067 on `sa` against 0.5667 on
     `sb`, and paired at the prompt level it is +14 on `sa` and **−4** on `sb`.
     **Its weights no longer exist**, so the result cannot be re-measured,
     extended, or built on.
   * P0-real (P1) 0.5533, P2-ceheavy 0.5333.
   * Every gap is smaller than P0-real's own **0.0800** seed spread, so the
     ordering is not resolvable at n=2.
4. **Among behaviour-comparable models, best correctness?** The three families are
   behaviour-comparable in the sense that nothing separates them. On that basis
   **P2-ceheavy holds the best correctness conditional on a usable rollout —
   `correct | usable` 0.3590 / 0.2927, the highest on both seeds.** That is the
   specific claim. **Reported separately, its overall correctness is 0.2000 /
   0.1800 (mean 0.1900, spread 0.0200)**, also the highest mean, but overall
   correctness mixes the behaviour failure back in and is the weaker statement.
   Neither is a demonstrated behaviour improvement.
5. **Has any model passed a defensible Stage 2/3 behaviour-recovery gate?**
   **No model has demonstrated passage of a prospectively defined behaviour
   gate.** No such gate existed when any of these runs was launched, and **no
   threshold may be invented post hoc** — so this is a statement about the absence
   of a registered criterion, not a measured failure against one. Descriptively:
   the best arm produces a usable rollout on **60.7%** of prompts, ~31% never
   terminate, and no candidate is seed-consistently better than another.
6. **Dominant remaining failure mode?** **Non-termination with repetition** —
   31.1% of rollouts run to the context limit, accounting for 280 of 395 protocol
   failures.

### 19.11 What this changes about the candidate set

All three Stage 2/3 families were trained at the **0.86M rung**. Under the
behaviour metric that rung is not the best available.

**These are preliminary results from a different measurement and must be fully
labelled every time they are quoted:**

| | the rung leads | the Stage 2/3 candidate results |
| --- | --- | --- |
| prompts | **76 behaviour prompts** (`data/eval_behavior_v0`) | **150 corpus examples**, mask `d6e24e0b…` |
| harness | **E1 behaviour wave** | **three-mode free mode** |
| **stop policy** | **degeneration stop ACTIVE** — loops cut at ~768 tokens | **no stop** — loops run to 8,192 |
| max observed tokens | 1,536 | 8,150 |

| rung | seed sa | seed sb |
| --- | ---: | ---: |
| 0.86M *(the candidate rung)* | 0.3684 | 0.4342 |
| **1.60M** | **0.4868** | **0.5132** |
| **2.96M** | **0.5921** | 0.5395 |

Same evaluation set, same init, same seeds *within this table*. **Do not compare
these values against the 150-example usable-rollout rates in §19.6** — different
prompt population, different harness, and a stop policy that changes the
context-limit and termination components outright (§19.13). The same weights
score 0.3684 here and 0.5133 there.

This is the strongest lead in the re-analysis and it is **not** a conclusion the
retained artifacts can close: the higher rungs have never been run through the
150-example three-mode harness, so they are `not_evaluable` on the Stage 2/3
candidate set without new generation. Weights exist locally for `r2960k_sb` and
`r5500k_sb` (both init variants); the `sa` counterparts are relay-only.

Note also that **P1's own weights have no local copy** — `e1_r0860k_{sa,sb}_pca`
live only on the storage-constrained relay — and the P0-assistant weights are
gone entirely. The reference checkpoint is the least well retained of the three.

### 19.12 Limitations

* Post-hoc. The hierarchy was defined after the artifacts existed.
* `usable_rollout` is empirically almost exactly `protocol_valid` (§19.2); it is
  one measurement presented as five.
* n=2 seeds per family. A 0.0800 seed spread on the primary metric bounds what
  any comparison here can resolve.
* E1 behaviour (76 prompts) and the Stage 2/3 set (150 prompts) are **different
  populations**. `e1_r0860k_sa_pca` *is* P0-real-sa and scores 0.3684 on one and
  0.5133 on the other. Never compare across the two.
* Random init was evaluated at n=1 at step 0 and n=12 downstream; the step-0
  number carries no error bar.
* The metric ignores correctness by construction, and a terse well-formed reply
  scores perfectly.

### 19.13 Reconciliation: "zero context-limit hits" was a stop-policy artifact

§19.8 reports 31.1% context-limited rollouts, while `STATE.md` and Experiment 2
phase 1 both recorded `context_limit_rate` **0.0000**. Both are correct
measurements of different stop policies, on the *same weights*:

| | E1 behaviour wave | three-mode free |
| --- | --- | --- |
| checkpoint | `e1_r0860k_sa_pca` | P0-real-sa — **the same weights** |
| degeneration stop | **active** | not applied during generation |
| n | 76 | 150 |
| median generated tokens | 315 | 722 |
| **max generated tokens** | **1,536** | **8,144** |
| `context_limit` | **0** | **63** |
| degeneration flagged | 19 (`stop_reason: degeneration`) | 62 |
| degenerate median length | 768 | 8,097 |

With the stop active a repetition loop is cut at ~768 tokens and booked as
`degeneration`; with it inactive the identical loop runs to the 8,192 limit and is
booked as `context_limit` **and** `degenerate`. The 280 context-limited rollouts
in §19.8 are real — median 8,099 tokens, minimum 5,594 — not a mislabelling.

**Consequence for reading the record:** "zero context-limit hits" never meant
generations terminate. It meant they were stopped before they could be counted.
Context-limit rates are only comparable between runs with the same stop policy,
and both policies must be stated whenever the number is quoted. This does not
change any earlier verdict — degeneration was reported in both cases — but it does
retire "no model has ever been engine-truncated" as a statement about this project.

### 19.14 Checkpoint recoverability — verified 2026-08-05

Prompted by §19.11: the incumbent reference is the least well retained model in
the project. Verified rather than assumed.

**Local, hash-verified against the pod-side manifests recorded before transfer —
30/30 files match, 0 mismatched, 0 missing:**

| arm | `model.safetensors` sha256 |
| --- | --- |
| `e1_r2960k_sb_pca` | `b658fe392ab0db49…` |
| `e1_r2960k_sb_rand` | `8ae1ba97f5146879…` |
| `e1_r5500k_sb_pca` | `bcb916cb3e544505…` |
| `e1_r5500k_sb_rand` | `bcb916cb3e544505…`* |
| `e1_ctl_r0250k_sa_pca_stepmatched` | `bfdcb4436f51eb31…` |

\* verified against its own manifest; listed digests are per-arm.

**Relay (`AlphaAvatar/aadistill-artifacts`, 729 files) — present and recoverable,
LFS digests recorded to `artifacts/audit/relay_e1_digests.json`:**

| arm | size | LFS sha256 |
| --- | ---: | --- |
| **`e1_r0860k_sa_pca` (P1-sa)** | 2.38 GB | `18ee10a10333481d…` |
| **`e1_r0860k_sb_pca` (P1-sb)** | 2.38 GB | `f66de5320b69aa34…` |
| `e1_r1600k_sa_pca` | 2.38 GB | `6f77676ab8fde397…` |
| `e1_r1600k_sb_pca` | 2.38 GB | `e432d57e598d57e1…` |
| `e1_r1600k_sa_rand` | 2.38 GB | `2e8be4b3289fe9b5…` |
| `e1_r1600k_sb_rand` | 2.38 GB | `9d0b499831689bd1…` |
| `e1_r2960k_sa_pca` | 2.38 GB | `3f08482c2c8e7372…` |
| `e1_r2960k_sa_rand` | 2.38 GB | `22374d62c8ae1d65…` |

**End-to-end download verified on two of them** — `e1_r1600k_sa_pca` and
**P1-sa** — 2.38 GB each in ~165 s, recomputed sha256 **matching the LFS digest
exactly** in both cases. The relay path is live and its digests are real, so P1
and the 1.60M/2.96M arms are genuinely recoverable, not merely listed.

Between local and relay, **every rung of Experiment 1 at both seeds and both
initializations is covered**: 2.96M `sb` exists only locally, 2.96M `sa` and all
of 1.60M only on the relay.

**P1's storage risk, recorded as a risk rather than resolved.** The incumbent
reference exists in exactly **one** place — the relay — which is at its
private-storage limit, cannot reclaim space by deletion (LFS bills history), and
has an approved-but-unexecuted history squash pending. **A history squash
invalidates existing revisions.** If it runs before P1 is copied elsewhere, the
project loses the weights of its own reference checkpoint, exactly as happened to
P0-assistant. P2-ceheavy is the only Stage 2/3 candidate with a verified local
copy. No action is taken here — copying P1 to the dev box costs ~4.8 GB of the
85 GB free and is a one-line operation, but it is a retention decision, not part
of this re-analysis.

---

## 20. Experiment 3 — restricting attention updates at the 0.86M rung

> **STATUS AT WRITING: REGISTERED BEFORE TRAINING.** Machine-readable
> registration with every hash: [`logs/e3_registration.json`](e3_registration.json).
> Results are appended below only after the run; nothing in §20.1–§20.7 was
> written with knowledge of an outcome.

### 20.1 The question

Every Stage 2/3 family so far trains all four attention projections full-rank,
and every one of them degenerates in free rollout: **31.1% of 900 rollouts run to
the context limit** (§19.8), which is the classic exposure-bias signature. This
experiment asks a narrow, cheap question about one candidate mechanism: **is the
full-rank attention update itself a source of drift at this rung?**

Two arms, one variable each, against the existing baseline:

| | trainable attention | everything else |
| --- | --- | --- |
| **A0** = **P2-ceheavy** `p2_ceheavy_{sa,sb}` | q/k/v/o + q_norm/k_norm, full-rank | — |
| **A1** | q_norm/k_norm only; **projections frozen** | identical to A0 |
| **A2** | A1 + **LoRA r32** on q/k/v/o, base frozen | identical to A1 |

A0 is **not** retrained. Its recorded results are the baseline, so the comparison
inherits the exact 150 fixed examples and inclusion mask `d6e24e0b…` that every
Stage 2/3 family at this rung has been measured on.

**The baseline is P2-ceheavy, so A1 and A2 inherit the P2 objective**
(`ce_weight 1.0`, `kd_weight 0.25`, τ=1.0, `kd_scope: all`) — not P1's
`0.25·CE + 1.0·KD`. Had they kept P1's weights, the attention treatment would
have been confounded with the loss-weight change that separates the two
families, and §18 measured that change to be worth −0.0141 on teacher-forced
reasoning top-1. `tests/training/test_e3_configs.py` asserts the inheritance.

**A2's adapter is not tuned.** Rank 32, alpha 64, dropout 0, bias none, and the
LoRA tensors sit in the **same single AdamW group** at the same learning rate,
schedule and weight decay as every other trainable parameter. There is no
separate LoRA learning rate, no separate parameter group, and no rank or module
sweep. A2 isolates low-rank *parameterization*; it is not an adapter
hyperparameter search, and a second optimizer setting would be a second variable.
The trainer now rejects `optim.lora_lr`, `optim.lora_weight_decay` and
`optim.no_decay_patterns` outright rather than accepting them quietly.

### 20.2 What is held fixed, and what that costs

Held fixed: the Stage 1 PCA fork point (`86fbba78…`), the nested uniform 0.86M
rung (**682 blocks / 864,750 supervised tokens / 1,502 sessions**) and its exact
block order, seeds `20260726`/`20260801`, `1.0·CE + 0.25·KD` at τ=1.0 with
`kd_scope: all`, AdamW 5e-5 / wd 0.01 / betas (0.9, 0.95) / clip 1.0, 1,023 steps
with 51 warmup, 2 blocks/step at `block_len` 8192, and the whole evaluation
protocol including greedy decoding and unrestricted generation (P18).

Trainable-parameter counts, measured on the real geometry (CPU, $0):

| arm | full-rank | LoRA | total trainable | of 596,049,920 |
| --- | ---: | ---: | ---: | ---: |
| A0 | 440,467,456 | — | 440,467,456 | 73.9% |
| A1 | 264,306,688 | — | 264,306,688 | 44.3% |
| A2 | 264,306,688 | 9,175,040 | 273,481,728 | 45.9% |

A1 removes exactly the 176,160,768 parameters of the four projections. A2 puts
back 5.2% of that as a rank-32 subspace, over 112 adapted modules.

**The adapter was resized twice by the maintainer before any A2 checkpoint
existed**, both times after both A1 arms had finished: rank 8 → 32 at 16:10 UTC,
then alpha 16 → 64 at 16:26 UTC. **`α/r` therefore stays at 2.0**, the same
scaling the original r8/α16 configuration had, so the adapter's effective update
magnitude is unchanged and **only the subspace dimension moves** — 8 → 32.

An α = 16 / r = 32 run reached step 180 of 1,023 before being stopped; its
partial training log is retained on the pod as `_aborted_a2_sa_alpha16` and no
checkpoint from it exists. It is **not** a rank-scaling data point: 180 of 1,023
steps under a different scaling is not a result, and nothing in this record
treats it as one. ~$0.16 of GPU time.

**The single-variable property is asserted mechanically, not by eye**
(`tests/training/test_e3_configs.py`): each arm's config is diffed against the
baseline's and the differing key set must be exactly `{trainable_patterns}` for
A1 and exactly `{lora}` for A2, with `run_name`/`out_dir`/`_purpose` excluded.

### 20.3 The LoRA implementation, and why it is native

`src/aadistill/training/lora.py`. Three properties a runtime-only adapter does
not give for free, each of which the experiment depends on:

1. **The saved `model/` is always the merged, adapter-free checkpoint.** All
   three arms are therefore measured through one inference architecture, and no
   evaluation path can accidentally score the un-adapted base model.
2. **Exact resume.** The frozen base attention weights and the raw LoRA tensors
   are stored next to the merged model (`lora_state.safetensors`) rather than
   recovered by subtracting the delta — `(w + d) − d` is not exactly `w`, and an
   inexact resume is not a resume.
3. **`LoRALinear` subclasses `nn.Linear` and shares the base weight tensor**, so
   the module keeps the `state_dict` key it replaced and costs no extra memory
   for the base matrix.

`B = 0` at initialization, so the initial merged model is *exactly* the Stage 1
model; `A` is drawn from `U(±1/√fan_in)` with an explicit `torch.Generator`, so
the draw is a pure function of (seed, module order, shapes) and does not perturb
global RNG. **`lora.seed` is pinned across both A2 arms**, so the run seed varies
block order only — exactly as it does in A0 and A1, keeping A2's seed spread
comparable to theirs rather than inflated by a second random draw.

### 20.4 Pre-registered decision rules and noise floors

Applied mechanically by `scripts/evaluation/analyze_e3.py`:

* **R1** — A1 improving rollout stability without materially reducing
  correctness ⇒ full-rank attention updates are likely causing harmful drift.
* **R2** — A2 beating both A1 and A0 **on both seeds** ⇒ constrained attention
  adaptation is the preferred policy.
* **R3** — A2 improving only teacher-forced CE/top-1 and not autonomous rollout
  ⇒ **do not** claim the main problem is solved.
* **R4** — both arms improving FineWeb NLL but not rollout ⇒ stop freeze-policy
  exploration, recommend student-prefix / on-policy recovery.
* **R5** — no promotion on one seed alone.
* **R6** — no promotion for terminating earlier if correctness **conditional on
  a usable rollout** falls.

Two families have been measured at this rung on this same set and their two-seed
spreads disagree. **The larger is used as the noise floor**, because with n=2 a
spread is a single draw and P2's unusually tight one is recorded as suggestive
rather than established (§18.7):

| | P1 spread | P2 spread | **floor used** |
| --- | ---: | ---: | ---: |
| usable rollout | 0.0800 | 0.0267 | **0.0800** |
| free-rollout correctness | 0.0600 | 0.0200 | **0.0600** |
| teacher-forced reasoning top-1 | 0.0025 | 0.0112 | **0.0112** |
| teacher-native held-out CE | 0.0063 | 0.0117 | **0.0117** |

Taking the smaller would have made it too easy to call an effect against the
very baseline this experiment is compared with.

The evaluation hierarchy is not inverted: `usable_rollout` and its five
components are primary, correctness secondary, and teacher-forced top-1 /
teacher-native CE / FineWeb NLL are diagnostics that never rank an arm alone.

### 20.5 What this cannot settle, stated in advance

* **n=2 seeds per arm.** A spread is one draw per condition; no variance claim
  will be made from it.
* **Rank 32 / α 64 is a single point.** A null result means "r32 at α/r = 2.0 on
  q/k/v/o under the baseline's optimizer settings does not help", not "LoRA does
  not help". Neither r8 nor α/r = 0.5 was trained to completion, so this
  experiment says nothing about either.
* **`usable_rollout` is blind to correctness by construction**, and its five
  components are not independent — `protocol_valid` subsumes two of them.
* **INT8 is measured as held-out NLL only** (fake-quant, scopes `all` and
  `decoder`). INT8 *rollout behaviour* is not measured and no claim is made
  about it.
* **A0's free/oracle/forced numbers come from the 2026-08-05 P2 session** and
  were produced on different hardware from A1/A2. FineWeb NLL is the one metric
  measured on a single device for all six arms.

### 20.6 Pre-launch validation (CPU, $0)

* **581 tests pass** (`uv run pytest tests/ -q`), including 32 new ones covering
  the adapter's zero initial delta, merge fidelity in BF16, bitwise-exact resume
  through a merged checkpoint, the block-stream position surviving a restart,
  training under gradient checkpointing and BF16 autocast, the freeze policy, and
  the config single-variable guarantees.
* `scripts/training/validate_e3_arms.py` ran all six arms **on the real
  596M-parameter student with the real Stage 1 weights**: freeze policy correct
  for every arm, embeddings frozen everywhere, attention projections frozen in
  A1/A2, 112 adapted modules in A2, **initial BF16 logits bit-identical before
  and after `apply_lora`**, and the merged state dict free of adapter keys.
  Report: `artifacts/audit/e3_prelaunch_validation.json`.
* **A0's parameter movement was computed locally from the retained P2 weights**,
  giving the baseline before any GPU was allocated:

| group | A0-P2-sa `‖ΔW‖_F / ‖W_init‖_F` | A0-P2-sb |
| --- | ---: | ---: |
| ffn | 0.031893 | 0.031841 |
| attn_proj | 0.016665 | 0.016647 |
| decoder_norm | 0.001167 | 0.001164 |
| attn_norm | 0.001136 | 0.001116 |
| final_norm | 0.000410 | 0.000415 |
| embedding | **0.000000** | **0.000000** |

The baseline moves its attention projections by 1.67% relative. A1 forces that
to exactly zero; A2 confines it to a rank-8 subspace. The zero on `embedding` is
also the tool validating itself against a parameter known to be frozen.

* **A0's held-out NLL was measured at all three precisions on the dev-box CPU**,
  21,080 tokens each, checksum matching every prior measurement:

| | BF16 | INT8 (`all`) | INT8 (`decoder`) |
| --- | ---: | ---: | ---: |
| A0-P2-sa | 8.9482 | 8.9992 | 9.0116 |
| A0-P2-sb | 8.9626 | 8.9596 | 8.9587 |

The BF16 figures reproduce the recorded GPU values (8.9504 / 8.9578, §18.7) to
**0.002 and 0.005 nats** — 0.02% and 0.06%. That agreement is what licenses
moving the whole NLL measurement off the pod: it costs **25 s per model** on
CPU, it puts the baseline and both treatment arms on **one device**, and it
needs no upload against a full LFS quota. It removes ~24 min of paid GPU time.

### 20.7 Budget

Costed from the **measured** P2-ceheavy session (174.6 min for two arms of this
exact rung on an L40S at $0.99/h), not by scaling token counts.

| phase | min |
| --- | ---: |
| setup, downloads, test suite, arm validation | 33 |
| train A1 ×2 | 123 |
| A1 freeze gate | 6 |
| train A2 ×2 | 126 |
| tokenizer, merge check, movement ×4 | 16 |
| three-mode harness ×4 | 44 |
| checkpoint transfer | 25 |
| **total** | **373 ≈ 6.2 h → $6.16** |

Hard backstop 450 min = **$7.43**. Ledger remaining **$8.70**. Held-out NLL is
absent from this table because it runs on the dev box at $0.

**A budget discrepancy is recorded rather than resolved silently.** The
instruction for this experiment cited an "established $50 hard cap" for student
training. No such line exists in the ledger: the recorded caps are a **$50
generation** cap (§6), a **$60 training** cap (2026-08-01 decision), and the
binding **cumulative $126.02** cap with **$117.32 spent**. This session enforces
the **strictest** of the available readings — the $8.70 cumulative remainder —
and fits inside it, so the two readings do not disagree about whether to proceed.
