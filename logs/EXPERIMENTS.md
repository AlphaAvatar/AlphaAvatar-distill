# Experiment record — AlphaAvatar-distill

The single consolidated record of everything run. Replaces 25 per-run logs and 11
proposal files, which are preserved in git history at commit `866dac2`.

**Teacher** `Qwen/Qwen3-4B-Thinking-2507@768f209d` (2560 hidden, 36L, FFN 9728,
32Q/8KV) → **student** 0.6B-class (1024 hidden, 28L, FFN 3072, 16Q/8KV, tied
embeddings). BF16 training, INT8 deployment target.

**Total spend to date: $149.59** of the **$149.03** cap — **the cap is exceeded
by $0.56 and nothing further is authorized.** The cap history: $126.02, then
**+$4.00** for E4 on 2026-08-05 ($130.02), then **+$11.50** for E5 across four
top-ups during its eight attempts ($141.52), **+$1.50** for E6 on 2026-08-08
($143.02), then **+$6.01** for E6b on 2026-08-09 ($149.03). No experiment was
reduced to fit a remainder; each shortfall was reported with its expected and
pessimistic cost and the scientific cost of the cheaper option.

**The $0.56 overrun is E6b's and is not amortized away** (§29.7): a 14% step-time
miss put the session past its backstop, and both automatic stop layers —
RunPod's `--terminate-after` and the launcher's own teardown — were inert at the
same time. Recorded rather than smoothed.

| period | $ | detail |
|---|---:|---|
| through corpus v2 (2026-08-01) | 34.52 | §6 — training/eval $7.93, teacher generation $26.59 |
| Experiment 1, data-scaling matrix | 61.50 | §11 — 24 arms $47.6, control + first eval $8.1, sweep $5.8 |
| Experiment 2 phase 1, data cleaning | 12.97 | §12.15 — experiment $11.23, avoidable pod waste $1.74 |
| Diagnostic session (benchmark + reference + recall) | 0.52 | §14 — 94 min on an RTX A6000 at $0.33/h |
| D0 no-training diagnostics | 1.15 | §16 — 210 min on an RTX A6000 |
| P0-assistant | 2.75 | §17 — 167 min on an L40S |
| P2-ceheavy | 2.88 | §18 — 174.6 min on an L40S |
| Experiment 3, attention restriction | 5.76 | §20 — 349 min on an L40S, four arms |
| Experiment 4, P2 at the 1.60M rung | 4.83 | §21 — 290 min on an L40S + $0.05 failed pod |
| Experiment 5, continuation vs recovery | 11.64 | §22–27 — ten paid events, eight of which produced no result |
| Experiment 6, high-rung normalization | 2.36 | §28 — 130 min on an L40S + $0.22 across three pods killed early |
| Experiment 6b, P2 at the 2.96M rung | 7.68 | §29 — 458 min on an L40S + $0.12 failed pod; **$0.56 over its $7.12 authorization** |
| **itemized subtotal** | **148.56** | |

**Unreconciled: $1.03.** The itemized rows sum to $148.56 while the verified
running total is **$149.59**. The difference is not
attributed to any session here, so the **larger** figure is used for every
remaining-budget decision. Do not "fix" this by deleting the gap.

**The authorized total is exceeded by $0.56.** §6 below is the
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

## 11. Experiment 1 — data-scaling matrix (COMPLETE 2026-08-02, 24 arms, $47.6)

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

> Numbered 13 but filed before §12: the sections are numbered in the order the
> work was *registered*, not the order it appears. Cross-references elsewhere
> use the numbers, so they are left as they are.

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

## 12. Experiment 2 — three sequential 0.86M diagnostics (phase 1 COMPLETE 2026-08-04; phases 2–3 unauthorized)

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

> **STATUS: COMPLETE 2026-08-05, $5.76.** Registered before training in
> [`logs/e3_registration.json`](e3_registration.json); nothing in §20.1–§20.7 was
> written with knowledge of an outcome. Results are §20.8 onward.
> **Verdict: restricting attention updates does not improve autonomous
> generation stability at this rung — it degrades it. Neither arm is adopted.**

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


### 20.8 Execution

Pod `94slla57nnqjqa`, 1× NVIDIA L40S secure at $0.99/h, 14:02 → 19:52 UTC =
**349 min = $5.76**, against a $6.16 projection, a 450-min RunPod-side backstop
that never fired, and $8.70 of ledger headroom. Setup took 7 min against a 33-min
estimate. Training ran at 3.32–3.44 s/step, peak 37.96 GiB of 46 GiB.

The pod deleted itself on `ALL_DONE`. All **28 transferred files hash-verified
byte-identical** against a manifest computed pod-side before transfer.

**Two mid-run reconfigurations by the maintainer, both applied before any A2
checkpoint existed and neither touching A1:** rank 8 → 32 (16:10) and alpha
16 → 64 (16:26). An α=16/r=32 run reached step 180 of 1,023 and was stopped;
its partial log is retained as `artifacts/audit/e3_aborted_a2_sa_alpha16/` and
**is not a data point** — 180 of 1,023 steps under a different scaling is not a
result. Combined cost of both changes: ~$0.30.

The A1 freeze gate passed before A2 was allowed to start: attention-projection
movement exactly `0.000000` on both seeds.

### 20.9 Primary — autonomous rollout behaviour

150 fixed examples, inclusion mask `d6e24e0b…` asserted equal to the baseline's
on every arm, greedy decoding, unrestricted generation (P18).

| metric | A0 | A1 | A2 | A1−A0 | A2−A0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **usable_rollout_rate** | **0.5333** | 0.4467 | 0.4400 | **−0.0866** | **−0.0933** |
| non_empty | 0.7733 | 0.6734 | 0.6666 | −0.0999 | −0.1067 |
| natural_termination | 0.6834 | 0.5400 | 0.5300 | −0.1434 | −0.1534 |
| no_severe_repetition | 0.6800 | 0.5333 | 0.5100 | −0.1467 | −0.1700 |
| no_context_limit | 0.6834 | 0.5400 | 0.5300 | −0.1434 | −0.1534 |
| protocol_valid | 0.5333 | 0.4533 | 0.4534 | −0.0800 | −0.0799 |

Per seed, with no overlap between the baseline and either treatment:

| | sa | sb | mean | spread |
| --- | ---: | ---: | ---: | ---: |
| A0 | 0.5200 | 0.5467 | 0.5333 | 0.0267 |
| A1 | 0.4267 | 0.4667 | 0.4467 | 0.0400 |
| A2 | 0.4533 | 0.4267 | 0.4400 | 0.0266 |

Both treatments are worse **on both seeds, on all five components**, and both
mean deltas clear the registered 0.0800 floor. Paired at the prompt level each
arm loses more prompts than it gains against its own-seed baseline — net −14,
−12 (A1) and −10, −18 (A2) of 150 — so this is not a mixture shift.

### 20.10 Secondary — correctness

| | A0 | A1 | A2 |
| --- | ---: | ---: | ---: |
| correct_overall | 0.1900 | 0.1067 | 0.1366 |
| correct_given_usable | 0.3258 | 0.1737 | **0.2882** |

`correct_given_usable` is the one axis where the adapter clearly earns its
parameters: **0.2882 vs A1's 0.1737**, and seed-stable (0.2794 / 0.2969 against
A1's 0.2188 / 0.1286). Per task, the damage is concentrated in evidence reading —
`rag_evidence` 0.676 → 0.405 (A1) → 0.460 (A2) on `sa` — while `gsm8k` and
`openmath` sit at floor in every arm and cannot register a change.

### 20.11 Diagnostics — reported, never used to rank

| | A0 | A1 | A2 |
| --- | ---: | ---: | ---: |
| teacher-native held-out CE | 1.5240 | 1.6823 | 1.6151 |
| teacher-forced reasoning top-1 | 0.5567 | 0.5193 | 0.5347 |
| FineWeb NLL, BF16 | 8.9554 | 9.9100 | 8.8997 |
| FineWeb NLL, INT8 `all` | 8.9794 | 9.9700 | 8.9439 |
| INT8 penalty | +0.0239 | +0.0600 | +0.0442 |

All six arms were measured on **one device** (dev-box CPU, 21,080 tokens each,
checksum matching every prior measurement), so BF16 and INT8 are comparable
across arms rather than across sessions. INT8 weight fake-quant costs 0.02–0.06
nats everywhere; **no arm is disproportionately quantization-fragile**, and INT8
rollout behaviour was not measured.

**A2's FineWeb mean is the best of any arm (8.8997) and means nothing**: its seed
spread is **0.4668** (8.6663 / 9.1331) against A0's 0.0143 — 33× wider — so the
−0.0557 mean sits deep inside a single draw. Recorded, not claimed.

### 20.12 Parameter movement — the mechanism is exactly as designed

`‖ΔW‖_F / ‖W_init‖_F` against the Stage 1 fork point:

| group | A0-sa | A0-sb | A1-sa | A1-sb | A2-sa | A2-sb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ffn | 0.031893 | 0.031841 | 0.033628 | 0.033569 | 0.033551 | 0.033535 |
| **attn_proj** | 0.016665 | 0.016647 | **0.000000** | **0.000000** | **0.004800** | **0.004771** |
| attn_norm | 0.001136 | 0.001116 | 0.001501 | 0.001493 | 0.001093 | 0.001080 |
| decoder_norm | 0.001167 | 0.001164 | 0.001227 | 0.001223 | 0.001183 | 0.001181 |
| embedding | 0 | 0 | 0 | 0 | 0 | 0 |

Attention displacement is **1.67% → 0% → 0.48%**, exactly the three points the
design asked for. A2 spends 29% of the baseline's attention movement and recovers
~44% of A1's CE loss for 9,175,040 parameters. The adapter is not uniform:
`q_proj` (3.13–3.16) carries ~2× `k_proj` and ~3× `v_proj`. A1's attention norms
strain **32% above** the baseline while A2's fall slightly below it — visible
evidence of norms compensating for frozen projections.

**Merge integrity, verified independently on both A2 arms:** zero LoRA keys in
`model/`, loads through the normal path with no adapter code, finite logits, 112
modules, and the merged-minus-init delta **equals a fresh recomputation of
`scaling · B @ A`**. The evaluated artifact is the trained model.

### 20.13 Verdict — none of the four rules fired

| rule | fired | why |
| --- | --- | --- |
| R1 full-rank attention updates cause harmful drift | **no** | the prediction is *inverted*: A1 is worse on rollout **and** correctness |
| R2 constrained adaptation is preferred | **no** | A2 loses to A0 on both seeds |
| R3 do not claim the problem is solved | **no** | A2's teacher-forced top-1 also fell (−0.022); the guard had nothing to catch |
| R4 stop freeze-policy work, go on-policy | **no** | it required *both* arms to improve FineWeb NLL; A1's rose 0.9546 |

**Both promotion guards return `promotable: false`.** Neither result is a
single-seed artifact, and neither arm terminates earlier — A1 and A2 terminate
*less* often than the baseline, so R6 had no trade-off to weigh.

**What the experiment establishes.** At the 0.86M rung, restricting attention
updates degrades autonomous generation stability, monotonically in how much is
restricted, and costs teacher-distribution fit at the same time. Full-rank
attention updating is **not** the source of the degeneration this project has
been chasing since §19.8 — the leading hypothesis going in is refuted, on both
seeds, on nine independent measurements.

**What it does not establish.** Rank 32 at `α/r = 2.0` is a single point; neither
r8 nor `α/r = 0.5` was trained to completion, so nothing here is a rank-scaling
result. n=2 seeds means every spread is one draw. `usable_rollout` is blind to
correctness by construction and its components are not independent. A0's rollout
numbers come from the earlier P2 session on different hardware; only NLL is
single-device across all six arms.

**Where this leaves the next step.** R4's *recommendation* — student-prefix and
on-policy recovery — is unchanged in attractiveness, but it is now an engineering
judgement rather than a fired rule, and the record must not imply otherwise. What
this experiment adds to the case is negative evidence: the offline-freeze family
of interventions has now been probed three ways (KD scope §17, KD magnitude §18,
attention capacity §20) and none has moved autonomous rollout. `src/aadistill/rollout/`
still holds 2,075 lines of tested infrastructure that no training path consumes.

### 20.14 Retained artifacts

`/home/ecs-user/aad-artifacts/e3/` (external to git), 18.6 GB, **28/28 files
`sha256sum -c` OK** against the pod-side manifest:

| arm | contents | size |
| --- | --- | ---: |
| `e3_a1_frozen_attn_{sa,sb}` | merged `model/` + `trainer_state.pt` | 4.3 GB each |
| `e3_a2_lora_attn_{sa,sb}` | merged `model/` + `lora_state.safetensors` (741 MB) + `trainer_state.pt` | 5.0 GB each |

Every A2 checkpoint is **both** deployable and resumable: `model/` is a plain
Hugging Face checkpoint with the adapter merged, and `lora_state.safetensors`
carries the frozen base and raw LoRA tensors for exact resume.

`e3_side.tar.gz` (1.2 MB) holds all four arms' 150 free + 150 oracle generations,
teacher-forced per-role reports, movement reports, merge check, pre-launch
validation, configs, `run_manifest.json` and `train_log.jsonl`, plus the aborted
α=16 run's log. **Not uploaded to the relay** — its LFS quota is full and no
follow-up arm forks from a trained checkpoint.


---

## 21. Experiment 4 — P2-CE-heavy scaled from the 0.86M to the 1.60M rung

> **STATUS: COMPLETE 2026-08-06, $4.83.** Registered before training in
> [`logs/e4_registration.json`](e4_registration.json).
>
> **Verdict.** Scaling 0.86M → 1.60M **substantially improves autonomous rollout
> stability** (+0.2000 usable rollout, both seeds, repetition and context-limit
> failures each −0.0833) and **does not materially improve autonomous
> correctness** (0.1900 → 0.2000, inside the floor, one seed up and one down,
> with `correct_given_usable` falling). **26.7% of rollouts remain unusable — this
> is a substantial improvement, not a solved problem.** At 1.60M, P1 and P2 are
> **effectively tied on behaviour**; the improvement belongs to **scale**, not to
> the CE/KD weighting change.

### 21.1 The question and the arms

Does the P2-CE-heavy recipe improve autonomous generation when scaled to the
existing nested 1.60M rung? Secondarily: at matched scale, does CE-heavy beat P1?

| arm | rung | objective | status |
| --- | --- | --- | --- |
| **P2-0.86M** `p2_ceheavy_{sa,sb}` | 0.86M | ce 1.0 / kd 0.25 | reference, not retrained |
| **P1-1.60M** `e1_r1600k_{sa,sb}_pca` | 1.60M | ce 0.25 / kd 1.0 | **re-evaluated**, not retrained |
| **P2-1.60M** `e4_p2_r1600k_{sa,sb}` | 1.60M | ce 1.0 / kd 0.25 | **new** |

P1-1.60M was re-evaluated because its recorded numbers came from the 76-prompt
behaviour wave with the degeneration stop **active** — a different measurement
from the 150-example unrestricted harness, and not mixable with it.

**The rung is the only free variable.** Everything it drags with it — 1,174
blocks, 1,600,353 supervised tokens, 1,761 steps (3 passes), warmup 88 (5%, as
at every rung), save/eval cadence — is copied from the tracked E1 config for the
same rung. `tests/training/test_e4_configs.py` asserts the differing key set is
exactly that closed set, that the 1.60M rung is a **strict superset** of 0.86M on
real token ids **and** CE masks, and that both rungs validate on the identical
held-out tail.

### 21.2 Primary — the result

150 fixed examples, mask `d6e24e0b…` **asserted equal on all four arms**, greedy,
unrestricted generation (P18).

| metric | P2-0.86M | P1-1.60M | **P2-1.60M** | scale Δ |
| --- | ---: | ---: | ---: | ---: |
| **usable_rollout_rate** | 0.5333 | 0.7300 | **0.7333** | **+0.2000** |
| natural_termination | 0.6834 | 0.7600 | 0.7667 | +0.0833 |
| no_severe_repetition | 0.6800 | 0.7533 | 0.7633 | +0.0833 |
| no_context_limit | 0.6834 | 0.7600 | 0.7667 | +0.0833 |
| correct_overall | 0.1900 | 0.1867 | 0.2000 | +0.0100 |
| correct_and_naturally_terminated | 0.1834 | 0.1867 | 0.2000 | +0.0166 |
| **correct_given_usable** | **0.3258** | 0.2532 | 0.2695 | **−0.0563** |

Per seed, usable rollout: P2-0.86M 0.5200/0.5467 → P2-1.60M 0.7133/0.7533. **No
overlap.** Paired per prompt: sa **+41/−12** (net +29), sb **+47/−16** (net +31);
bootstrap CIs **[+0.100, +0.280]** and **[+0.107, +0.307]**, both excluding zero.

Correctness does not move. +0.0100 mean is inside the 0.0600 floor and is **not**
seed-consistent: sa +0.0333, sb **−0.0133**. Paired nets are +5 and −2, neither
CI excluding zero. `correct_and_naturally_terminated` behaves identically —
sb's paired net is exactly **0**.

**The mechanism is visible in one number.** `correct_given_usable` *falls*
0.3258 → 0.2695: scaling produced ~30 additional well-formed rollouts per seed
and **most newly completed rollouts remain incorrect**. Stability improved
substantially; reasoning did not.

### 21.3 Objective at matched scale — a tie

| | usable | correct | correct·term | correct \| usable |
| --- | ---: | ---: | ---: | ---: |
| P1-1.60M | 0.7300 | 0.1867 | 0.1867 | 0.2532 |
| P2-1.60M | 0.7333 | 0.2000 | 0.2000 | 0.2695 |
| Δ | +0.0033 | +0.0133 | +0.0133 | +0.0163 |

Every delta is inside its floor, none wins on both seeds, and no CI excludes
zero (usable: sa **−10**, sb **+11**). **At 1.60M the objective barely matters.**

Decisively, **P1 gained from scale too** — 0.5533 at 0.86M → 0.7300 at 1.60M. The
+0.20 belongs to **scale**, not to CE-heavy.

### 21.4 Diagnostics

| | P2-0.86M | P1-1.60M | P2-1.60M |
| --- | ---: | ---: | ---: |
| teacher-native held-out CE | 1.5240 | **1.2983** | 1.3126 |
| FineWeb NLL BF16 | 8.9554 | — | **8.0992** |
| FineWeb NLL INT8 (`all`) | 8.9794 | — | 8.1467 |
| INT8 penalty | +0.0240 | — | +0.0475 |

Teacher-native CE improves **−0.2114** with scale, 18× its 0.0117 floor — while
autonomous correctness does not move at all. That is **R2 firing**, and it is the
sharpest statement this experiment makes.

FineWeb NLL improves −0.856 but its seeds are 8.5645/7.6339, a spread of
**0.9306** against P2-0.86M's 0.0143 — 65× wider. **No mean claim is made.** INT8
weight fake-quant costs 0.024–0.048 nats; neither family is disproportionately
quantization-fragile. All six models were measured on one device (dev-box CPU,
21,080 tokens each), $0.

Parameter movement scales as expected, embeddings exactly 0:

| group | P2-0.86M | P2-1.60M-sa | P2-1.60M-sb |
| --- | ---: | ---: | ---: |
| ffn | 0.031867 | 0.041368 | 0.041420 |
| attn_proj | 0.016656 | 0.021816 | 0.021857 |
| embedding | 0.000000 | **0.000000** | **0.000000** |

### 21.5 Registered rules

| rule | fired | why |
| --- | --- | --- |
| R1 P2 was data-limited at 0.86M | **no** | required correct_overall **and** usable to improve with no serious per-seed regression; correctness is inside noise and regresses on `sb` |
| R2 more teacher-prefix data does not resolve the rollout gap | **YES** | CE −0.2114 (18× floor) while autonomous correctness is flat |
| R3 do not promote P2 on CE/NLL alone | **moot** | P2-1.60M does not underperform P1-1.60M |
| R4 adopt P2-1.60M as the anchor | **no** | required beating **both** references; vs P1-1.60M it is a tie inside every floor |

### 21.5.1 The supported conclusion, stated exactly

1. **P2-0.86M was behaviour/stability-limited.**
2. **Additional teacher-prefix training fixes a substantial part of that
   stability problem** — usable rollout +0.2000 on both seeds, repetition and
   context-limit failures each −0.0833. It does not eliminate it: **26.7% of
   rollouts are still unusable at 1.60M**.
3. **Additional teacher-prefix training does not resolve the remaining
   reasoning/correctness gap.** `correct_overall` 0.1900 → 0.2000 is inside the
   floor, splits by seed, and `correct_given_usable` falls.

Writing "P2 was data-limited" would claim a correctness gain the data does not
show, and "scale fixes behaviour" would overstate a 73.3% usable rate.

**Teacher-native CE and FineWeb NLL are diagnostics, not promotion metrics.**
Both improved substantially here (CE −0.2114, 18× its floor; NLL −0.856) while
the primary axis did not move. That dissociation is the entire reason the
hierarchy keeps them below `correct_overall`, and no arm is promoted on them.

### 21.6 What this cannot settle

* **The 150 prompts are shared *training* prompts for every arm.** Both rungs
  contain all 150, so no arm gains exposure and the comparison is fair — but this
  measures **recall-style autonomous behaviour, not held-out generalization**.
* n=2 seeds; every spread is one draw. What carries weight is that usable rollout
  moves the same way on **both** seeds with non-overlapping ranges.
* Two rungs is not a scaling curve. A gain at 1.60M says nothing about 2.96M.
* `usable_rollout` is blind to correctness by construction — which is precisely
  why the falling `correct_given_usable` matters and is reported beside it.
* The paired bootstrap resamples **prompts at fixed checkpoints**; it is not a
  seed-level inference.

### 21.7 Execution and cost

Pod `qzevis6g43en33`, L40S at $0.99/h, 06:58 → 11:49 = **290 min = $4.78**, plus
**$0.05** for a pod that failed in 3 min on a bad relay path (§21.8) —
**$4.83 total** against $6.94 authorized and a 415-min backstop that never fired.
sa 108.3 min at 3.625 s/step; sb similar. Pod self-deleted; **12/12 files
`sha256sum -c` OK** against a pod-side manifest. Retained:
`/home/ecs-user/aad-artifacts/e4/`, 11.2 GB, plus a 1.1 MB side bundle.

### 21.7.1 Exact reproduction record

Commit **`1dafec29b1637d3e1412be7fcf453640c4cd97d9`** (both arms' manifests agree).

```bash
# preflight, CPU, $0
PYTHONPATH=src python scripts/training/preflight_e4.py \
    --out artifacts/audit/e4_preflight.json
PYTHONPATH=src python scripts/training/register_e4.py --out logs/e4_registration.json

# the pod session (self-tearing-down)
SCR=… SESSION_COMMIT=1dafec29b1637d3e1412be7fcf453640c4cd97d9 \
  BUNDLE_NAME=aad_e4_1dafec29.bundle BACKSTOP_MINUTES=415 \
  nohup bash scripts/pod/e4_launch.sh &
# pod-side, started by the launcher:
#   /opt/train/bin/python scripts/pod/e4_driver.py --stage all

# post-run, CPU, $0
PYTHONPATH=src python scripts/evaluation/eval_ppl.py \
    --data data/warmup/holdout_v1.jsonl \
    --model …/p2_ceheavy_sa --model …/p2_ceheavy_sb \
    --model …/e4_p2_r1600k_sa/model --model …/e4_p2_r1600k_sb/model \
    --max-seq-len 1024 --dtype bfloat16 [--fake-quant int8 --fake-quant-scope all|decoder] \
    --out artifacts/audit/e4_holdout_nll_{bf16,int8_all,int8_decoder}.json
PYTHONPATH=src python scripts/evaluation/analyze_e4.py --bootstrap 10000
```

| identity | value |
| --- | --- |
| config sha256 (`sha256_json`) | sa `8256bfba8b3241a8…` · sb `7c3817a729133dc9…` |
| final `model.safetensors` sha256 | sa `7ee1d9355b97563f…` · sb `98e8c9811414e982…` |
| Stage 1 fork point | `86fbba78e8a2a324…`, asserted on the pod before training |
| ladder pack `blocks.npz` | `6f324cb0f37bc0f0…`, 1,174 blocks / 1,600,353 supervised |
| teacher | `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d9e…` |
| P1-1.60M references | `6f77676ab8fde397…` / `e432d57e598d57e1…`, asserted at download |
| evaluation inclusion mask | `d6e24e0b09da1bcc…`, asserted on all four arms |
| trainable parameters | 440,467,456 of 596,049,920 |

**Teardown and final state:** pod `qzevis6g43en33` deleted by the launcher at
11:49:18Z (`runpodctl pod list` → `[]`); dev-box watchdog confirmed
"already gone"; zero stray launcher/watchdog processes; **12/12 artifacts
`sha256sum -c` OK**; 626 tests pass; working tree clean.

### 21.8 Three infrastructure defects, all mine, all from one blind spot

`sed 's/e3_/e4_/g'` when deriving the E4 pod scripts:

1. **Rewrote the middle of `stag`*`e3_`*`recovery_corpus_v2`** → `stage4_…`. Pod
   downloaded nothing, died on the first `iterdir`, self-deleted. **$0.05.**
2. **Missed `e3.status`** (a dot, not an underscore), so the launcher polled a
   file the driver never wrote. The run finished at 11:17; the launcher would
   have idled to its 400-min timeout — **≈$2.65 of billing for nothing**, against
   a $4.00 authorization. Caught in a routine status check, patched live with a
   symlink, verified by writing a marker through it.
3. **The teardown watchdog was armed on the wrong pid** — `pgrep -f` matched the
   *orchestrating shell's own command line*. It would have deleted a **healthy
   pod mid-training** when that shell exited. Caught before it fired; the
   watchdog now verifies the pid's argv is the launcher and refuses otherwise.

All three are now guarded by `tests/pod/test_pod_script_paths.py`, each verified
by reintroducing the exact defect and confirming the test names it.

---

## 22. Experiment 5 — attempt 1: R corpus generated, then found unpackable (2026-08-07, $1.57)

**Verdict: aborted at the pairing gate. No training ran. No E5 result exists.**
The failure is a defect in this repository's own code, not evidence about
teacher-prefix (C) versus student-prefix (R) continuation.

| | |
| --- | --- |
| Pod | RunPod L40S `n1kavevctllrvi`, $0.99/h securePrice-verified |
| Billed | **$1.57** (95 min), pod self-deleted 09:02 UTC |
| Commit | `91cd1cd` |
| Authorization | $9.12; **$7.55 remains** |
| Reached | setup → validate → benchmark → **gate 1 PASS** → generate sa+sb → **pair FAILED** |

### What worked, and is worth keeping

Setup completed in **5 minutes**, not the 53 minutes measured on 2026-08-06 —
the image was warm. All eleven markers, both venvs, three checkpoint hashes
verified, 689 tests passed on the pod.

The `truncate_padding` benchmark returned a **measured 2.497× wall-clock
speedup** against a 3.245× executed-position reduction, confirming that the
position count was an upper bound rather than a prediction. Full-width 3.222
s/step → truncated 1.290 s/step, on 30 blocks spanning the real E5-C length
distribution (min 362, p50 678, max 8188 real tokens). Gate 1 passed at $7.64
backstop against $8.55 remaining.

R generation itself worked and was cheaper than budgeted: **74 minutes for both
seeds**, against 152 projected. Acceptance was high and the gates fired
sensibly:

| seed | rollouts | recovery requests | accepted | complete bundles | dominant rejection |
| --- | --- | --- | --- | --- | --- |
| sa | 1147 | 2280 | 2116 (92.8%) | 1058 | `natural_termination` 110 |
| sb | 1147 | 2278 | 2080 (91.3%) | 1040 | `natural_termination` 135 |

`natural_termination` dominating is the expected shape given E4: the student
still fails to terminate on roughly a quarter of rollouts.

### The defect

`RecoveryExample.to_record()` emitted only the **summary counts** — token
lengths, fractions, task labels — and dropped `ids` and `mask`. The tokens were
built, gated, and round-trip-checked in memory, then discarded on the way to
disk. `stage_pair` raised on the first record:

```
ValueError: example 'glaive-000008#t1#r0' missing ['ids', 'mask', 'system_key']
```

Arm C's builder wrote its records independently and did include the payload, so
**every offline validation passed**: the registration records the packing path
as "built and validated offline on arm C". The two arms answered to one packing
contract that only one of them had ever been tested against. The full-path pilot
that would have caught this died in setup twice, on 2026-08-06.

4,196 gated R examples were lost. They cannot be recovered from the saved
records because the tokens were never written.

### The second defect, which is the more dangerous one

The driver caught only `(CalledProcessError, AssertionError, OSError)`. A
`ValueError` escaped `main()`, so **no marker was written at all** — not
`STAGE_FAILED`, not `ABORTED_AT_GATE`. The launcher polls for a terminal marker
and would have kept polling for its full 520-minute window with the work already
dead, idle-billing an estimated **$6–7**. It cost ~6 minutes only because the
scheduled status check caught it. A crash on a paid pod is a billing event, not
just a stack trace.

### A third defect, found while fixing the first

`check_gates` computed the context budget as `n_total_tokens + n_system_tokens`,
but `ids` already opens with the system block — the prompt is rendered from the
full message list. The gate was stricter than intended by the system-block
length and rejected valid long samples. Conservative rather than corrupting, but
wrong; a test had codified the double count.

### Fixes (commit `<this>`)

* `system_key` and `n_system_tokens` are now **required fields** on
  `RecoveryExample`, and `to_record()` emits `ids`/`mask`/`system_key`. The
  missing-field class of bug becomes a construction error rather than a
  discovery made after generation is paid for.
* `build_e5_arm_r.py` derives `system_key` from the same `system_group_key`
  helper arm C uses, writes its own `system_ids.json`, asserts the prompt opens
  with exactly the system block it keys, and **runs `example_to_rendered` on
  every record before accepting it**.
* `stage_pair` merges *both* arms' system maps and fails if they disagree about
  the tokens behind a shared key.
* The driver catches `Exception`; the launcher additionally checks driver
  liveness each poll and tears down a pod whose driver has died with no terminal
  marker.
* `check_gates` counts the system block once.

Verified: `test_recovery_record_is_packable` reproduces the exact failing call
and fails against the old `to_record`. An offline rehearsal built R-shaped
records from 400 **real** rescued arm-C examples and ran the complete path —
`to_record` → JSON → `example_to_rendered` → `pack_e5` → `write_pack` →
`verify_pack` — packing 34/34 blocks with `passed: True`. 728 tests pass.

### Artifacts retained

`/home/ecs-user/aad-artifacts/e5/` — pod status, run log, side bundle, and
`rescue/` holding both arms' corpora and manifests pulled before teardown. The
arm C corpora (33 MB per seed) are complete and reusable. The R corpora are
metadata only and must be regenerated.

### Next

Relaunch requires regenerating R (~74 min, ~$1.22). Estimated total for a
complete run from a warm image: ~355 min, ~$5.86 expected, ~$6.56 at the 1.12
backstop, against **$7.55 remaining** — roughly $1.00 of slack. Awaiting a
decision on whether to spend it.

---

## 23. Experiment 5 — attempt 2: stopped by budget gate 1 (2026-08-07, $0.81)

**Verdict: no paid generation, no training, no E5 result.** Gate 1 refused to
proceed and the pod was torn down 11 minutes after setup. This is the budget
layer working as designed — it stopped *before* the expensive stage, not after.

| | |
| --- | --- |
| Pods | `2lz4xqdok1z5s1` (abandoned), `adevfxvuiwu2lw` |
| Billed | **$0.81** total — $0.25 abandoned + $0.56 |
| Commit | `40f8122` |
| Authorization | $7.55; **$6.74 remains** |
| Reached | setup → validate → benchmark → **gate 1 FAIL**, shortfall $0.55 |

### Infrastructure note: a pod that never came up

The first pod never exposed TCP 22 within the 15-minute startup bound. The
launcher deleted it and created a replacement, which was ready in 2 minutes.
That is the bound doing its job, but it exposed a gap: the launcher rewrites
`pod_start_epoch` for the replacement, so the abandoned pod's $0.25 would have
been invisible to the gates, and a fresh 457-minute deadline would have put the
worst case at $7.79 — above the authorization. Corrected by backdating the epoch
by the abandoned pod's exact 913-second lifetime, which flows through both the
reported cost and the driver's starting balance without editing a running
script, and by restarting the watchdog against the corrected origin.

### What the gate found

```
blocks 1123 | sec_per_step 1.4105 | spent $0.52
remaining: r_generation 152, pair_pack 20, evaluate 44, transfer 35, train 158
expected $6.76 | backstop $7.57 | remaining authorization $7.02
covered: false | shortfall $0.55
```

Reused arm C staged and verified clean in the pod's own environment. The
`truncate_padding` benchmark reproduced independently on a second pod:
**2.559×** measured wall-clock speedup (attempt 1: 2.497×), full-width 3.884 →
truncated 1.517 s/step. Two independent measurements 2.5% apart is a usable
figure for the cost model.

### The cost model contained one stale input

`r_generation: 152` was an estimate made before any R generation had ever run.
Attempt 1 then **measured** the identical operation — same hardware, engine,
prompts, student, both seeds — at **74.1 minutes** (sa 34.6, sb 39.5). The gate
was carrying a 78-minute phantom, worth $1.29, and it is the difference between
passing and failing.

Replacing an unmeasured prior with a direct measurement of the same operation is
not moving the goalpost; leaving a known 2× overestimate in place would block a
run that fits. Every other phase estimate is left untouched, and the two other
phases that attempt 2 measured are folded in at their measured values.

Re-projected, with margin over the measurement rather than the measurement
itself:

| phase | min | basis |
| --- | --- | --- |
| pod startup + setup | 9 | measured, attempt 2 |
| validate | 7 | measured, attempt 2 |
| benchmark | 2 | measured, attempt 2 |
| R generation | 90 | **measured 74**, +21% margin |
| verify_records | 2 | new CPU stage |
| pair + pack | 20 | unmeasured, unchanged |
| final benchmark | 5 | unmeasured, unchanged |
| train 4 arms | 158 | 1123 blocks (R = 1.30 × C, unmeasured) |
| evaluate 4 | 44 | unmeasured, unchanged |
| transfer + teardown | 35 | unmeasured, unchanged |
| **total** | **372** | **$6.14 expected** |

Against $6.74 remaining: **fits at expected and at a 1.05 backstop ($6.44), and
is $0.13 short at the registered 1.12 backstop ($6.87).**

The dominant remaining uncertainty is not generation but `train`, which rests on
the unmeasured assumption that R packs to 1.30 × C blocks. That number is
unknowable until R exists, which is exactly what gate 2 is for.

### Next

A relaunch needs either a small additional authorization to restore the 1.12
margin, or an explicit decision to run at a 1.05 backstop with roughly $0.30 of
headroom — thin enough that one more abandoned pod would consume it. No token
target, treatment, seed or arm was altered to fit the budget, and none should be.

---

## 24. Experiment 5 — attempt 3: stopped on a cold host (2026-08-07, $1.45)

**Verdict: terminated during setup, before gate 1. No generation, no training,
no E5 result.** The environment build could not finish inside the budget.

| | |
| --- | --- |
| Pod | `dvoosn07fjs1oh` |
| Billed | **$1.45** (88 min), terminated manually |
| Commit | `917fc34` |
| Authorization | $8.24; **$6.79 remains** |
| Reached | `ENV_READY → REPO_READY → DATA_READY → TRAIN_ENV`, then stopped |

### Setup time is the dominant risk, and it is not under control

| pod | `uv sync` | full setup |
| --- | --- | --- |
| attempt 1 | 44 s | **5 min** |
| attempt 2 (2nd pod) | ~50 s | **8.5 min** |
| attempt 3 | **62 min** | ≥150 min (projected) |

The same script, the same image tag, the same GPU. The difference is whether the
host has the layers and wheels cached. On this host `uv sync` took 62 minutes;
21 minutes into the vLLM install the venv was **13 MB and growing at 0 KB/s**,
and the teacher was 95 MB of ~8 GB. Raw PyPI throughput measured 1.27 MB/s, so
the link was not broken — the volume is simply several GB and nothing was cached.

Gate 1 would have had to run by 12:45 UTC to stay inside the registered 1.12
backstop. Remaining setup was clearly an hour or more, and even ignoring the
backstop, a ~13:40 setup finish plus ~340 min of real work exceeds the $8.23
ceiling at *expected* cost. The run could not complete. It was terminated rather
than left to spend another $0.50–1.00 reaching a foreseeable failure.

### Why this is worth naming as its own failure mode

Setup is pure overhead paid before any science, and it varies **30×** between
hosts — 5 minutes to 150. On a run whose total work is ~370 minutes, an
unpredictable 150-minute tax is not something a cost model can absorb; it is
larger than R generation and training combined at the low end. Three attempts
have now produced no C/R evidence, and the direct causes were different each
time, but the third is the only one that will recur without an infrastructure
change: attempts 1 and 2 fixed code, and code stays fixed.

Observed base rate so far: **1 cold host in 3**. Expected setup cost per attempt
is roughly 55 min ($0.90) with a variance that dominates the budget.

### Next: eliminate the environment build, do not retry into it

A fourth attempt on the current setup path carries roughly a one-in-three chance
of losing another ~$1.50 to the same cause. The environment should be persisted
rather than rebuilt — a RunPod network volume holding `/opt/train`, `/opt/vllm`
and the Hugging Face cache is the natural fit, since it is RunPod-native, avoids
the HF LFS quota problem, and turns a 5-to-150-minute step into a mount. The
tradeoff to weigh is that a network volume pins the pod to one datacenter, which
narrows L40S availability.

No token target, treatment, seed or arm was altered. $6.79 of the authorization
is intact.

---

## 25. Experiment 5 — attempt 4: the feasibility gate found a real conflict (2026-08-07, $1.53)

**Verdict: stopped at the joint feasibility gate. No training, no C/R result —
but the first attempt to produce a genuine measurement about the design.**

| | |
| --- | --- |
| Pod | `8nheigrv25aeq8`, one draw, **no cold-host redraws** |
| Billed | **$1.53** (93 min); cumulative E5 **$5.36** |
| Commit | `a4935c4` |
| Authorization | $8.79; **$7.26 remains** |
| Reached | setup → validate → benchmark → **gate 1 PASS** → generate sa+sb → **`RECORDS_VERIFIED`** → pair → **INFEASIBLE** |

### Two things worked that had never worked before

Setup finished in **4 min 45 s** on the first draw; the tripwire never fired.

**`RECORDS_VERIFIED` passed.** Every accepted R record was re-read from disk and
converted through `example_to_rendered` — 2,102 records for `sa`, 2,056 for `sb`,
zero missing fields, zero unrenderable, zero system-block mismatches. The
attempt-1 data-contract defect is fixed and now confirmed on real generated data
rather than on synthetic examples.

### The conflict the gate found

R's supervised continuation is systematically **1.66–1.76× longer than C's** on
the same bundle at the same relative cut depth:

| seed | C tok/bundle | R tok/bundle | R/C |
| --- | --- | --- | --- |
| sa | 711.6 | 1179.4 | **1.657** |
| sb | 684.2 | 1205.7 | **1.762** |

This is a property of the treatments, not a bug. C's supervised span is the
teacher's *own remaining trajectory* after the cut, which is bounded by the
trajectory that already exists. R's is a *fresh* teacher generation conditioned
on a student prefix, and the teacher handed a partial student reasoning chain
writes a longer completion than its own remainder would have been — it repairs
or restarts rather than continuing.

The consequence is structural: with atomic bundles and identical composition,
**one common bundle count cannot put both arms on the token target.** At the 778
bundles the selector chose, C landed 24.7% under and R 24.7% over (sa); 27.6%
under and 27.5% over (sb). The gate refused, exactly as registered — "if the full
corpus cannot simultaneously satisfy the CE-token target, atomic bundles, the
block budget and the C/R tolerance, STOP before training and report the measured
conflict."

### A resolution exists, and it is within the registered tolerance

Selecting each arm independently to the same token target, with R's bundles
nested inside C's pool:

| seed | C | R |
| --- | --- | --- |
| sa | 1,034 of 1,051 bundles → 735,823 tok (1.000×) | 624 bundles → 735,918 tok (1.000×) |
| sb | **all** 1,028 bundles → 703,337 tok (0.956×) | 610 bundles → 735,495 tok (1.000×) |

Both arms land inside the registered 5% for both seeds. The cost is that
composition is no longer identical: R trains on ~0.60× as many distinct bundles
as C, seeing fewer problems at greater supervised depth each.

This does **not** damage the paired statistics. McNemar and the paired bootstrap
run over the **150 pinned held-out evaluation prompts at fixed checkpoints**, not
over training bundles, so training composition was never what made the
comparison paired.

Two caveats. `sb`'s C arm has no headroom — it needs its entire paired pool and
still lands 4.4% under target, inside tolerance but with nothing spare. And the
per-bundle rates above are **extrapolated** from the 778-bundle stratified
selection; they must be re-measured on the real corpus before being registered,
not assumed.

### The R corpora were lost again — my defect

The launcher's side bundle ships `artifacts/audit`, configs and `manifest.json`
files, but not `examples.jsonl`. So a corpus costing ~$1.20 of GPU time was
generated, verified, and discarded at teardown — for the **second** time
(attempt 1 lost it to the record defect; attempt 4 lost it to the transfer).

Waiting for teardown to preserve an artifact means losing it precisely when the
run stops early, which is when it is most worth keeping. Fixed by pushing the
corpora to the relay inside `verify_records`, the moment they are known good and
before anything downstream can fail, with the side bundle carrying a second copy.
Retention failure warns and never aborts the run.

### Next

The design decision is the user's: independent per-arm selection to the token
target (composition differs), or identical composition with unmatched tokens
(R sees ~69% more supervised signal). No token target, treatment, seed or arm
should be changed to force the other to fit.

---

## 26. Experiment 5 — attempt 5: the selector worked, a vestigial restriction did not (2026-08-07, $1.55)

**Verdict: stopped at the feasibility gate. No training, no C/R result — but the
R corpora survived, and the design is now demonstrably feasible offline.**

| | |
| --- | --- |
| Pod | `m8mmk8f2389f8s`, second create after a capacity drought, no cold redraws |
| Billed | **$1.55** (94 min); cumulative E5 **$6.91** |
| Commit | `c94a96a` |
| Authorization | $7.26; **$5.71 remains** |
| Reached | gate 1 → generate sa+sb → **`RECORDS_VERIFIED`** → `CORPORA_RETAIN_FAILED` → pair → **INFEASIBLE** |

### What worked

Setup 5 min on the first draw. `GENERATED:sa` at 34.9 min, matching the 34.6 and
36.1 of previous attempts. `RECORDS_VERIFIED` passed on 2,098 + 2,042 records.

**The new nested selector passed every one of its own conditions**: arm-to-arm
token delta **0.0000%** on both seeds, R nested in C, atomic bundles intact,
system blocks agreeing, task shares within 2.6 points.

**The R corpora survived.** The relay push failed — the driver is detached with
`setsid` and does not inherit setup's exported `HF_TOKEN`, so it raised
`KeyError` — but the second copy in the side bundle worked, and 32 MB + 30 MB of
real R records came back. The belt-and-braces retention did its job on its first
outing, through the belt failing.

### Why it stopped

T\* came out at **689,621**, 6.25% below the registered 735,603 and outside the
5% ceiling, because `C_sb`'s pool was short. The cause is a **vestigial
restriction**: C was pooling over the *intersection* with R rather than its own
full corpus.

| seed | C full pool | C paired pool | cost of pairing |
| --- | --- | --- | --- |
| sa | 905,488 (1.231×) | 743,088 (1.010×) | 162,400 — 17.9% |
| sb | 905,488 (1.231×) | **689,621 (0.937×)** | 215,867 — **23.8%** |

The intersection was required when both arms had to share a composition. That
requirement was dropped in favour of independent per-arm selection, where only
`R_selected ⊆ C_selected` must hold — and C's full pool satisfies it by
construction, since every R bundle comes from a C prompt.

Keeping it was also not methodologically neutral. The bundles it removed are
exactly the prompts where the teacher's recovery from a student prefix failed a
gate, mostly on natural termination. Restricting C to prompts where R succeeded
conditions C's corpus on R's outcome.

### The corrected design, computed on the real corpora at $0

With C pooling over its full corpus, using attempt 5's actual R:

| | sa | sb |
| --- | --- | --- |
| T\* | **735,603** — unreduced, 0.0000% | same |
| C | 963 bundles → 735,603 (1.0000×) → 882 blocks | 989 → 735,603 → 904 blocks |
| R | 603 bundles → 735,603 (1.0000×) → 591 blocks | 672 → 735,603 → 674 blocks |
| shared / C-only | 603 / 360 | 672 / 317 |
| arm-to-arm delta | 0.0000% | 0.0000% |
| KD-mask = non-padding, C / R | 2,086,364 / 1,534,148 | 2,120,212 / 1,617,726 |

Common block count **904** (even), **1,356** optimizer steps, **117 min** of
training across four arms.

### Budget: the decision the numbers force

| plan | post-gate | ×1.12 | pre-gate allowance |
| --- | --- | --- | --- |
| regenerate R | 313 min | $5.78 | **−$0.07 — SHORT** |
| **reuse the real R corpora** | 223 min | **$4.12** | **+$1.59 = 96 min** |

Regenerating R would reproduce data that already exists, cost $1.48, and put the
run $0.07 past the authorization. The corpora are staged on the relay,
hash-verified (`e2cbbd45…`), and re-checked against the current contract in the
pod's own environment — the same posture arm C has had since attempt 4.

### Fixed

* C pools over its full corpus; only R is intersected.
* `_retain_corpora` reads `/workspace/hf/token` rather than requiring an
  inherited `HF_TOKEN`.
* Setup stages and contract-checks arm R alongside arm C.

---

## 27. Experiment 5 — COMPLETE: teacher-prefix continuation beats student-prefix recovery (2026-08-07, $2.87)

**Verdict: under a matched CE-supervision budget, teacher-prefix continuation (C)
is decisively better than student-prefix recovery (R) on autonomous rollout
behaviour. Correctness is not distinguishable between the arms.**

| | |
| --- | --- |
| Pod | `3wku8lc1yztfbj` (second draw; the first never opened :22) |
| Billed | **$2.87** (174 min); cumulative E5 **$10.65** |
| Commit | `8c85223` |
| Arms | 4 trained, 4 evaluated, `ALL_DONE` 23:44:41 |
| Corpus | T\* = 735,603 CE tokens/pass, all four arms exactly on target, arm-to-arm delta 0.0000% |
| Budget | 904 common blocks, 1,356 steps, 3 passes, 2 blocks/step |

### Primary — autonomous rollout behaviour

| arm | usable | non_empty | natural_term | no_severe_rep | no_context_limit | protocol_valid |
| --- | --- | --- | --- | --- | --- | --- |
| C-sa | 0.7600 | 0.8667 | 0.8333 | 0.8333 | 0.8333 | 0.7600 |
| C-sb | 0.7733 | 0.8600 | 0.8533 | 0.8467 | 0.8533 | 0.7800 |
| R-sa | 0.4600 | 0.5000 | 0.4800 | 0.4733 | 0.4800 | 0.4667 |
| R-sb | 0.4333 | 0.4733 | 0.4467 | 0.4467 | 0.4467 | 0.4400 |
| **C** | **0.7667** | 0.8633 | 0.8433 | 0.8400 | 0.8433 | 0.7700 |
| **R** | **0.4467** | 0.4867 | 0.4633 | 0.4600 | 0.4633 | 0.4533 |

**Δ = −0.3200**, four times the 0.0800 noise floor, in the same direction on both
seeds and on *every* component. Within-arm seed spread is 0.0133 (C) and 0.0267
(R), an order of magnitude smaller than the gap.

Paired, on the 150 pinned prompts at fixed checkpoints:

| seed | metric | C | R | Δ | McNemar (R gained / lost) | bootstrap 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| sa | usable | 0.7600 | 0.4600 | −0.3000 | 11 / 56, net −45 | [−0.3933, −0.2000] **excludes 0** |
| sb | usable | 0.7733 | 0.4333 | −0.3400 | 5 / 56, net −51 | [−0.4267, −0.2533] **excludes 0** |
| sa | correct | 0.1400 | 0.1067 | −0.0333 | 11 / 16, net −5 | [−0.1000, +0.0333] includes 0 |
| sb | correct | 0.1200 | 0.1133 | −0.0067 | 9 / 10, net −1 | [−0.0600, +0.0533] includes 0 |

### Secondary — correctness

C 0.1300, R 0.1100, Δ −0.0200 — inside the 0.0600 floor, with both bootstrap
intervals spanning zero and McNemar nets of −5 and −1 on 27 and 19 discordant
pairs. **The arms are not distinguishable on correctness.**

`correct_given_usable` is C 0.1652 against R 0.2463. **This does not favour R.**
It conditions on 67 usable R rollouts against 115 usable C rollouts, and the
subsets are selected by the very outcome that differs, so the comparison is
selection-biased by construction. It is reported because the registration
requires it, not as evidence.

### The mechanism: R does not stop

The dominant failure is not malformed output — it is that R never terminates.

| arm | median generated tokens | context-limit rate | natural termination |
| --- | --- | --- | --- |
| C | 513–562 | 0.147–0.167 | 0.833–0.853 |
| R | **6,362–6,692** | **0.520–0.553** | 0.447–0.480 |

R's median rollout is **twelve times longer** than C's and runs into the 8,192
context limit on more than half of prompts. Of R's ~77 empty answers per seed,
**1** terminated naturally: the emptiness is a *consequence* of never reaching
`</think>`, not a separate defect. `first_failure` attributes 75/79 of R's
unusable rollouts to `non_empty`, but the census is a presentation aid — the
per-component rates show the whole cluster moving together.

Oracle mode isolates it. Given the reasoning and asked only to write the answer,
R answers correctly 0.5933/0.6067 against C's 0.7400/0.7733, with **zero** empty
answers in either arm. R can extract an answer; it cannot decide to stop
reasoning.

That is what the recipe trains. Every R target is a teacher recovery
*conditioned on an existing student prefix*, so R is supervised only on how to
continue an in-progress trace and never on how to bring one to a close from a
cold start. At inference it must produce the whole trajectory from the prompt,
and it continues.

### Claim boundary, as registered

Evaluation is paired on the fixed 150-prompt battery, but **training composition
was not identical**: C trained on 963/989 bundles and R on 603/672, nested inside
C's selection, because matching the CE-supervision budget forces different bundle
counts when R's continuations run 1.66–1.76× longer. E5 therefore estimates the
performance of the two **complete recipes** under a matched supervised-token
budget. It does not isolate the causal effect of prefix content or state with
composition held fixed, and the paired statistics preserve paired *evaluation*
only — they do not remove the composition difference.

R trained on ~37% fewer distinct prompts, which is a live alternative explanation
for a diversity-sensitive outcome. It is a poor explanation for *this* outcome:
the failure is a specific behavioural one, and the oracle result shows the model
is intact when it does not have to terminate on its own.

### Artifacts and a loss

Evaluation reports, generations, feasibility report, gate records and packing
audits are all retained. **The four trained checkpoints were not**: the
launcher's checkpoint fetch still names `step_000738`, a constant from the
superseded 492-block design, while the real tag is `step_001356`. The weights
died with the pod. The result stands on the retained evaluation artifacts, but
the checkpoints would have to be retrained to re-evaluate.

### 27.0 The complete Experiment 5 ledger

Ten paid events, $11.64. Two attempts (6 and 7) and the two pre-attempt pilots
had no section of their own until this pass; they are recorded here so the ledger
reconciles and so their causes stay useful (P11).

| # | date | event | cost | stopped at | cause |
| ---: | --- | --- | ---: | --- | --- |
| — | 08-06 | pilot 1 | $0.07 | setup | `list_repo_tree(recursive)` 504s on a 700-file relay; replaced with per-file fetch |
| — | 08-06 | pilot 2 | $0.92 | setup | held-out battery read from a gitignored path; pinned to `e5_heldout_eval_ids.json` |
| 1 | 08-07 | [attempt 1](#22-experiment-5--attempt-1-r-corpus-generated-then-found-unpackable-2026-08-07-157) | $1.57 | pairing | `to_record()` dropped `ids`/`mask`; 4,196 gated R examples unusable |
| 2 | 08-07 | [attempt 2](#23-experiment-5--attempt-2-stopped-by-budget-gate-1-2026-08-07-081) | $0.81 | gate 1 | cost model carried a 152-min generation estimate; measured 74.1 |
| 3 | 08-07 | [attempt 3](#24-experiment-5--attempt-3-stopped-on-a-cold-host-2026-08-07-145) | $1.45 | setup | cold host: `uv sync` 62 min against 44 s warm |
| 4 | 08-07 | [attempt 4](#25-experiment-5--attempt-4-the-feasibility-gate-found-a-real-conflict-2026-08-07-153) | $1.53 | feasibility | R/C supervised-token asymmetry 1.66–1.76×; one bundle count cannot serve both |
| 5 | 08-07 | [attempt 5](#26-experiment-5--attempt-5-the-selector-worked-a-vestigial-restriction-did-not-2026-08-07-155) | $1.55 | feasibility | C pooled over the C∩R intersection, costing it 23.8% of its corpus |
| 6 | 08-07 | attempt 6 | $0.33 | gate 1 | gate still reserved 90 min for generation that had become a staged artifact |
| 7 | 08-07 | attempt 7 | $0.54 | training | pack declared one rung over every block, leaving no validation tail for `ladder_blocks` |
| 8 | 08-07 | **attempt 8 — complete** | **$2.87** | — | four arms trained and evaluated |

Attempts 6 and 7 are short because each died at a gate it reached quickly, and
both were caused by a change made between attempts rather than by anything the
experiment was testing: attempt 6 by switching R from generated to staged without
repricing the phase, attempt 7 by a packing contract that had never been exercised
against the trainer's validation-block selection.

Three of the ten events were the *same* class of failure — an artifact or model
that was correct in one respect and unusable in another — which is why the run now
verifies provenance and usability separately (`verify_staged_r.py`) and re-reads
every corpus from disk before pairing.

**The gates worked.** Every failure stopped before the money went into training;
the one attempt that reached training completed. The cost of eight attempts is
the price of finding out that six things were broken, not the price of six wrong
answers.

### 27.1 Post-hoc: C against the P2-1.60M matched-CE anchor

**Not a pre-registered contrast.** E5 registered C vs R. This compares C to the
E4 arms on the *same* 150-prompt battery — inclusion mask `d6e24e0b09da1bcc`
asserted identical across all eight arms — because C's continuation was sized as
the nested-rung increment, so its cumulative CE exposure lands within 0.3% of
P2-1.60M's.

| model | cumulative CE tokens | usable | correct | correct \| usable | seeds (usable) |
| --- | ---: | ---: | ---: | ---: | --- |
| P2-0.86M (E5's start) | 860,000 | 0.5333 | 0.1900 | 0.3258 | 0.5200/0.5467 |
| P1-1.60M | 1,600,353 | 0.7300 | 0.1867 | 0.2511 | 0.7800/0.6800 |
| **P2-1.60M** | 1,600,353 | **0.7333** | **0.2000** | 0.2682 | 0.7133/0.7533 |
| **E5-C** | 1,595,603 | **0.7667** | **0.1300** | 0.1652 | 0.7600/0.7733 |
| **E5-R** | 1,595,603 | **0.4467** | **0.1100** | 0.2463 | 0.4600/0.4333 |

Paired against P2-1.60M on the same prompts:

| seed | metric | P2-1.60M | C | Δ | C gained/lost | bootstrap 95% CI |
| --- | --- | ---: | ---: | ---: | --- | --- |
| sa | usable | 0.7133 | 0.7600 | +0.0467 | 23 / 16 | [−0.0333, +0.1267] |
| sb | usable | 0.7533 | 0.7733 | +0.0200 | 22 / 19 | [−0.0667, +0.1067] |
| sa | correct | 0.2333 | 0.1400 | **−0.0933** | 9 / 23 | **[−0.1667, −0.0200]** excludes 0 |
| sb | correct | 0.1667 | 0.1200 | −0.0467 | 8 / 15 | [−0.1067, +0.0133] |

**On behaviour, C ties the anchor.** +0.0334 pooled, inside the 0.0800 floor,
neither seed's CI excluding zero. Teacher-prefix continuation reaches the same
rollout stability as simply training the ordinary next rung — it does not beat
it. C's seed range does sit entirely above P2-1.60M's (0.7600–0.7733 against
0.7133–0.7533), which is suggestive, but the paired test is the stronger
instrument and it says tied.

**On correctness, C appears to lose.** −0.0700 pooled, above the 0.0600 floor,
same direction on both seeds, and sa's CI excludes zero with C losing 23 prompts
and gaining 9. Against its own starting point C is 0.1900 → 0.1300, a drop right
at the floor. This is one seed significant and one not, so it is a signal to take
seriously rather than a settled result.

**R is worse than not continuing at all.** Against P2-0.86M it loses 0.0866 usable
rollout and 0.0800 correctness, both above their floors. The student would have
been better left alone.

`correct_given_usable` falls monotonically along the whole sequence — 0.3258 →
0.2682 → 0.1652 — which extends E4's finding rather than contradicting it: each
increment of behavioural stability arrives with a *lower* hit rate among the
rollouts it makes usable.

**What this does not establish.** C and P2-1.60M differ in more than the prefix
treatment: different data (prefix/continuation splits with CE on the continuation
only, versus the ordinary next ladder rung), and a different schedule (two
sequential cosine runs against one). Cumulative CE tokens match to 0.3%; nothing
else is controlled. The honest reading is that these are two ways of spending
~1.6M cumulative supervised tokens, and the cheaper, simpler one is at least as
good on behaviour and possibly better on correctness.

---

## 28. Experiment 6 — the E1 PCA scale curve, normalized onto the frozen battery (2026-08-08, $2.36)

> **STATUS: COMPLETE.** Evaluation only — nothing trained, no checkpoint written,
> merged, quantized or overwritten, proven by AST over every executed script
> (`artifacts/audit/e6_notrain_proof.json`). Registered before any GPU existed in
> [`logs/e6_registration.json`](e6_registration.json).
>
> **Verdict. The original PCA lineage improves from 1.60M to 2.96M and then
> plateaus.** Usable rollout +0.1100 at 2.96M and +0.1200 at 5.50M against
> 1.60M — both above the 0.0800 floor, both seed-consistent — while 5.50M against
> 2.96M is +0.0100, inside the floor, with the seeds disagreeing. **Correctness
> never moves**: every correctness comparison in the experiment is inside the
> 0.0600 floor.
>
> **`e1_r2960k_{sa,sb}_pca` is the best checkpoint this project has evaluated.**
> It beats the standing P2-1.60M anchor by **+0.1067** usable rollout, above the
> floor and in the same direction on both seeds, while tying it on correctness
> (+0.0067). **This supersedes the 2026-08-08 decision that P2-1.60M is the best
> checkpoint** — that decision was taken when the high rungs had never been run
> through this harness.

### 28.1 Headline

150 fixed examples, inclusion mask `d6e24e0b…` asserted on all eight arms,
greedy, unrestricted generation (P18). Every arm re-scored from raw generations
with the current scorer; no stored `correct`/`usable`/termination field was
carried in.

| model | CE exposure (unique / cumulative) | seed | usable | correct | correct \| usable |
| --- | ---: | --- | ---: | ---: | ---: |
| E1 PCA 1.60M | 1,600,353 / 4,801,059 | sa | 0.7800 | 0.1733 | 0.2222 |
| E1 PCA 1.60M | 1,600,353 / 4,801,059 | sb | 0.6800 | 0.2000 | 0.2843 |
| **E1 PCA 1.60M** | | **mean** | **0.7300** | **0.1867** | 0.2511 |
| E1 PCA 2.96M | 2,960,507 / 8,881,521 | sa | 0.8533 | 0.2133 | 0.2500 |
| E1 PCA 2.96M | 2,960,507 / 8,881,521 | sb | 0.8267 | 0.2000 | 0.2419 |
| **E1 PCA 2.96M** | | **mean** | **0.8400** | **0.2067** | 0.2460 |
| E1 PCA 5.50M | 5,501,372 / 16,504,116 | sa | 0.8467 | 0.2067 | 0.2362 |
| E1 PCA 5.50M | 5,501,372 / 16,504,116 | sb | 0.8533 | 0.1467 | 0.1719 |
| **E1 PCA 5.50M** | | **mean** | **0.8500** | **0.1767** | 0.2039 |
| P2 1.60M | 1,600,353 / 4,801,059 | sa | 0.7133 | 0.2333 | 0.3178 |
| P2 1.60M | 1,600,353 / 4,801,059 | sb | 0.7533 | 0.1667 | 0.2212 |
| **P2 1.60M** | | **mean** | **0.7333** | **0.2000** | 0.2682 |

"Unique" is the rung's supervised-token count; "cumulative" is 3× that, because
every arm trains 3 epochs. The two must not be confused, and neither is a free
variable separable from optimizer steps — see §28.6.

### 28.2 Paired comparisons on the shared mask

Floors carried unchanged from the E3/E4/E5 registry: usable **0.0800**,
correct_overall **0.0600**. `*` marks a bootstrap interval excluding zero.

| comparison | axis | sa | sb | pooled | verdict |
| --- | --- | ---: | ---: | ---: | --- |
| 2.96M vs 1.60M | usable | +0.0733 | +0.1467 `*` | **+0.1100** | above floor, seed-consistent |
| 2.96M vs 1.60M | correct | +0.0400 | +0.0000 | +0.0200 | inside floor |
| 5.50M vs 1.60M | usable | +0.0667 | +0.1733 `*` | **+0.1200** | above floor, seed-consistent |
| 5.50M vs 1.60M | correct | +0.0333 | −0.0533 | −0.0100 | inside floor, seeds disagree |
| 5.50M vs 2.96M | usable | −0.0067 | +0.0267 | +0.0100 | inside floor, seeds disagree |
| 5.50M vs 2.96M | correct | −0.0067 | −0.0533 | −0.0300 | inside floor |
| 2.96M vs P2-1.60M | usable | +0.1400 `*` | +0.0733 | **+0.1067** | above floor, seed-consistent |
| 2.96M vs P2-1.60M | correct | −0.0200 | +0.0333 | +0.0067 | inside floor |
| 5.50M vs P2-1.60M | usable | +0.1333 `*` | +0.1000 `*` | **+0.1167** | above floor, both CIs exclude 0 |
| 5.50M vs P2-1.60M | correct | −0.0267 | −0.0200 | −0.0234 | inside floor |
| 1.60M vs P2-1.60M | usable | +0.0667 | −0.0733 | −0.0033 | inside floor, seeds disagree |
| 1.60M vs P2-1.60M | correct | −0.0600 | +0.0333 | −0.0133 | inside floor, seeds disagree |

Prompt-level win/tie/loss for the two comparisons that carry the result:
2.96M over 1.60M gains 21 and loses 10 on `sa`, gains 34 and loses 12 on `sb`;
2.96M over P2-1.60M gains 28 and loses 7 on `sa`, 24 and 13 on `sb`.

### 28.3 Components and raw counts

| arm | usable/150 | correct/150 | correct/usable | protocol | nat.term | ctx-limit | repetition | empty | numeric parse-fail | tokens p50 / p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1-1.60M-sa | 117 | 26 | 26/117 | 0.7800 | 0.8133 | 0.1867 | 0.1867 | 0.1400 | 0.1733 | 576 / 8117 |
| E1-1.60M-sb | 102 | 30 | 29/102 | 0.6933 | 0.7067 | 0.2933 | 0.3067 | 0.2200 | 0.2400 | 616 / 8119 |
| E1-2.96M-sa | 128 | 32 | 32/128 | 0.8733 | 0.8733 | 0.1267 | 0.1467 | 0.1200 | 0.0267 | 585 / 8098 |
| E1-2.96M-sb | 124 | 30 | 30/124 | 0.8333 | 0.8467 | 0.1533 | 0.1600 | 0.1267 | 0.1333 | 586 / 8105 |
| E1-5.50M-sa | 127 | 31 | 30/127 | 0.8533 | 0.8533 | 0.1467 | 0.1533 | 0.1333 | 0.0267 | 574 / 8063 |
| E1-5.50M-sb | 128 | 22 | 22/128 | 0.8533 | 0.8533 | 0.1467 | 0.1467 | 0.1133 | 0.1067 | 580 / 8039 |
| P2-1.60M-sa | 107 | 35 | 34/107 | 0.7133 | 0.7467 | 0.2533 | 0.2533 | 0.2333 | 0.1200 | 566 / 8123 |
| P2-1.60M-sb | 113 | 25 | 25/113 | 0.7600 | 0.7867 | 0.2133 | 0.2200 | 0.1933 | 0.1333 | 731 / 8108 |

**What actually changed is termination.** Context-limit hits fall from 28/44
prompts at 1.60M to 19/23 at 2.96M and 22/22 at 5.50M; repetition tracks it
almost exactly, as it has in every prior measurement. Numeric answer-parse
failure falls from 0.1733/0.2400 to 0.0267/0.1333 — the higher rungs emit a
recognisable final answer far more often.

**The residual failure is silence, not malformation.** In the first-failure
census `non_empty` is the largest bucket at every rung and barely moves:
21/33 prompts at 1.60M, 18/19 at 2.96M, 20/17 at 5.50M. Roughly one prompt in
eight still produces no answer at all, and more data does not fix it.

### 28.4 By frozen evaluation subset (family means)

| family | gsm8k usable / correct | multihop usable / correct | openmath usable / correct | rag usable / correct |
| --- | ---: | ---: | ---: | ---: |
| E1-1.60M | 0.7106 / 0.0000 | 0.8158 / 0.2369 | 0.4189 / 0.0135 | 0.9730 / 0.5000 |
| E1-2.96M | 0.9079 / 0.0132 | 0.9343 / 0.2369 | 0.5675 / 0.0135 | 0.9460 / 0.5676 |
| E1-5.50M | 0.9211 / 0.0132 | 0.8552 / 0.1184 | 0.6622 / 0.0000 | 0.9594 / 0.5811 |
| P2-1.60M | 0.7237 / 0.0000 | 0.8158 / 0.1842 | 0.4189 / 0.0135 | 0.9729 / 0.6081 |

**GSM8K usable rollout rises from 0.71 to 0.92 while GSM8K correctness stays at
0.00–0.01.** The same holds on `openmath` (0.42 → 0.66 usable, correctness at
floor). The higher rungs learn to finish a maths problem and not to solve one.

There is **no `behavior` partition** on this battery — that name belongs to the
retired 76-prompt `eval_behavior_v0` wave. It is absent, not omitted.

### 28.5 Diagnostics — they move, and they keep moving after behaviour stops

| arm | teacher-native val CE | teacher-forced reasoning top-1 | FineWeb holdout NLL |
| --- | ---: | ---: | ---: |
| E1-1.60M sa / sb | 1.2952 / 1.3015 | 0.5976 / 0.5940 | 9.7145 / 9.4845 |
| E1-2.96M sa / sb | 1.1468 / 1.1486 | 0.6152 / 0.6213 | 10.4031 / — |
| E1-5.50M sa / sb | 1.0032 / 1.0052 | 0.6403 / 0.6389 | 10.7875 / 9.4548 |
| P2-1.60M sa / sb | 1.3112 / 1.3140 | 0.5815 / 0.5818 | — |

CE falls monotonically (1.298 → 1.148 → 1.004) and teacher-forced reasoning top-1
rises monotonically (0.596 → 0.618 → 0.639) across all three rungs. **Behaviour
stops improving after 2.96M and correctness never improves at all.** From 2.96M
to 5.50M the diagnostics are the *only* thing that moves — which is the sharpest
statement of the CE/behaviour dissociation the project has recorded, and the
reason these metrics may not select a checkpoint. `e1_r2960k_sb_pca`'s FineWeb
NLL was never measured in Experiment 1 and is reported absent rather than
imputed.

### 28.6 Determinism — token streams, not agreeing rates

`e1_r1600k_{sa,sb}_pca` was measured in the E4 session and again in E6, on a
**different physical host two days later**, and reproduced **token for token**:

| arm | E4 token-stream sha256 | E6 token-stream sha256 | identical |
| --- | --- | --- | --- |
| E1-1.60M-sa | `045bc96f93c86cb3…` | `045bc96f93c86cb3…` | **yes** (150/150) |
| E1-1.60M-sb | (matching) | (matching) | **yes** (150/150) |

150/150 identical token-id sequences and identical decoded text on both seeds;
every rate agrees to the digit. This is a stronger claim than matching
percentages, which can coincide while generations differ.

**Consequence:** the cross-session split that shaped this experiment's design is
empirically null, so no comparison here inherits uncertainty from it. The
project's standing "same session, same GPU" rule came from **CPU vs L40S**
differences (2026-07-27); within one GPU model, image and harness, greedy
decoding is bitwise reproducible across pods. The rule should be restated in
those terms rather than dropped.

### 28.7 Does the retired harness's ordering survive?

| rung | prior usable (76-prompt wave, sa / sb) | prior mean | **E6 mean** |
| --- | --- | ---: | ---: |
| 1.60M | 0.4868 / 0.5132 | 0.5000 | **0.7300** |
| 2.96M | 0.5921 / 0.5395 | 0.5658 | **0.8400** |
| 5.50M | 0.5526 / 0.5395 | 0.5461 | **0.8500** |

**Ordering only — never the levels.** The prior wave used 76 behaviour prompts
with the degeneration stop ACTIVE, which cuts a repetition loop before it can be
counted and so changes the termination and context-limit components outright.

The prior ranking was 2.96M > 5.50M > 1.60M. E6 confirms **both high rungs beat
1.60M**, and refines the top: 2.96M and 5.50M are **tied**, not ordered. The
prior ordering's top two were within noise and are still within noise.

### 28.8 The conclusion, stated exactly

1. **Best E1 rung: 2.96M.** It and 5.50M are tied on the primary axis (+0.0100,
   inside the floor, seeds disagreeing), so the registered tie-break applies —
   correctness may break a tie between behaviour-comparable candidates — and
   2.96M leads on `correct_overall` (0.2067 vs 0.1767) and `correct_given_usable`
   (0.2460 vs 0.2039), in the same direction on both seeds. It also costs half
   the tokens and 66% of the optimizer steps.
2. **Best existing evaluated checkpoint: `e1_r2960k_{sa,sb}_pca`.** Above the
   floor and seed-consistent against P2-1.60M on behaviour, tied on correctness.
3. **Scale trend: improve, then plateau.** The gain is spent between 1.60M and
   2.96M; nothing on either primary axis moves after that.
4. **Seed consistency:** every behaviour gain is same-direction on both seeds.
   **No correctness comparison in the entire experiment is both above its floor
   and seed-consistent.**
5. **Higher exposure changes stability, then only diagnostics.**
6. **"1.60M is the practical high point" does not survive.** It was never
   measured on this harness; it does not hold here.

### 28.8.1 Scope clarification, added 2026-08-09 (E6b preregistration)

E6's scale conclusion is about the **E1/P1 KD-heavy lineage**, and nothing in
E6 licenses extending it to other objectives. Stated precisely:

* **Established by E6** — for the E1/P1 objective (ce 0.25 / kd 1.0): 1.60M →
  2.96M improves usable rollout above the floor and seed-consistently; 2.96M →
  5.50M is a tie.
* **NOT established by E6** — that 2.96M is a plateau *for any other objective*.
  P2 CE-heavy has never been trained above 1.60M, so whether the plateau is a
  property of the **data** or of that **particular objective** is unknown.

The two objectives are tied at 1.60M under both registered floors, which is
exactly the situation in which a shared plateau cannot be assumed: tied at one
scale says nothing about behaviour at the next. E6b fills the missing cell
(P2 × 2.96M) and is the only experiment that makes the interaction computable.

Until E6b reports, write "the E1/P1 lineage plateaus after 2.96M", never "2.96M
is a plateau". E6's findings are unchanged and remain valid as recorded.

### 28.9 What E6 cannot settle

* The 150 evaluation prompts are **training** prompts for every arm. The ladder
  is nested and every rung trains 3 epochs, so each evaluated prompt is seen
  exactly 3 times by every arm — exposure to them is identical across rungs and
  the comparison is fair, but this measures recall-style autonomous behaviour,
  **not held-out generalization**.
* Rung and optimizer steps scale together by construction, so "more distinct
  data" and "more steps" are not separated.
* n=2 seeds; every spread is one draw.
* `usable_rollout` is blind to correctness by construction. At 2.96M the model
  finishes 84% of rollouts and answers 21% of them correctly.
* **No prospectively registered Stage 2/3 gate exists**, so no arm here has
  "passed" one. E6 ranks candidates on the registered hierarchy; it does not
  certify one.

### 28.10 Execution and cost

Pod `xikv3u8moofu6c`, L40S at $0.99/h securePrice-verified, 10:59 → 13:10 =
**130 min = $2.14**, plus **$0.22** across three pods killed early (§28.11) —
**$2.36 total** against $3.47 authorized. Pod self-deleted at 13:10:12;
`runpodctl pod list` → `[]`. Artifacts digest-verified before teardown
(`2b757282…`, 1.5 MB) and additionally fetched incrementally per arm.

**The session was transfer-bound, not compute-bound.** The six evaluations took
~36 minutes; the two dev-box-only checkpoints took **117 minutes** to upload at
~0.5 MB/s. The driver evaluated the four relay arms during that upload and waited
on a marker for the rest, so the pod was idle for roughly 50 of 130 minutes
rather than 117.

**Deviation recorded (P4).** `ninja-build` was installed on the running pod at
11:23 UTC, outside the setup script pinned by `SESSION_COMMIT=ad250cd1`. No
source file on the pod differs from that commit; the package is a build tool for
flashinfer's JIT sampling kernel and its absence had killed the driver at its
first `LLM()` call. Installing it restored parity with the E4 session, whose
setup installs ninja and whose arms E6 re-scores. Fixed in the repo at `bf97424`
with a test.

### 28.11 The paid-compute ledger

| # | pod | billed | stopped at | cause |
| ---: | --- | ---: | --- | --- |
| 1 | `qp7xm2k01h6dg4` | $0.03 | before setup | setup would have hash-verified the dev-box checkpoints while the launcher was still uploading them; found by reading the launcher, killed rather than edited while running |
| 2 | `9nrdoydt53nlzc` | $0.09 | during setup | measured the dev-box uplink at 0.72 MB/s; the 4.77 GB transfer could not finish inside the deadline, so the run was doomed as configured |
| 3 | `jusu7nm5e5xcyc` | $0.10 | pod test gate | probe pack fetched without `audit.jsonl`, which `ladder_blocks` reads; and one test read gitignored artifacts a pod never receives |
| 4 | `xikv3u8moofu6c` | **$2.14** | — | **complete: six arms evaluated** |

Every fix carries a test: `--stores relay` staging split, the ninja requirement
across all nine vLLM setups, the ladder-loader file list derived from `ladder.py`
itself, and `scripts/pod/simulate_pod_env.sh`, which runs the pod's exact test
command with every gitignored artifact hidden and would have caught two of the
three for free.

### 28.12 Exact reproduction record

Commit **`ad250cd1dfb5867373a44755113b8fcecb94fbd1`** (evaluation session).

```bash
# preflight and registration, CPU, $0
PYTHONPATH=src python scripts/evaluation/register_e6.py --authorized-usd 3.47
bash scripts/pod/simulate_pod_env.sh

# the pod session (self-tearing-down)
SCR=… SESSION_COMMIT=ad250cd1dfb5867373a44755113b8fcecb94fbd1 \
  BUNDLE_NAME=aad_e6_ad250cd1.bundle AUTHORIZED_USD=3.25 \
  setsid nohup bash scripts/pod/e6_launch.sh &
# pod-side: /opt/train/bin/python scripts/pod/e6_driver.py --stage all

# analysis, CPU, $0, reproducible from the retained generations alone
PYTHONPATH=src python scripts/evaluation/analyze_e6.py --bootstrap 10000
```

| identity | value |
| --- | --- |
| inclusion mask | `d6e24e0b09da1bcc…`, rebuilt on the pod and asserted per arm |
| pack `blocks.npz` | `6f324cb0f37bc0f0…` |
| corpus `sessions.jsonl` | `2b4edc2e2cc16cd5…` |
| Stage 1 init | `86fbba78e8a2a324…` |
| E1 1.60M weights | `6f77676ab8fde397…` / `e432d57e598d57e1…` |
| E1 2.96M weights | `3f08482c2c8e7372…` / `b658fe392ab0db49…` |
| E1 5.50M weights | `3069b329df3edfbd…` / `bcb916cb3e544505…` |
| P2 1.60M weights | `7ee1d9355b97563f…` / `98e8c9811414e982…` |
| engine | vLLM 0.26.0, bf16, greedy, context 8192, no degeneration stop |
| artifacts | `logs/e6_results.json`, `logs/e6_report.md`, `artifacts/audit/e6_per_prompt.jsonl` (1,500 records), `artifacts/audit/three_mode/E1-*` |

---

## 29. Experiment 6b — P2 CE-heavy at 2.96M: the objective interacts with scale (2026-08-09, $7.68)

> **STATUS: COMPLETE.** Both arms trained to completion from the Stage 1 PCA init
> and evaluated on the frozen battery. Registered before training in
> [`logs/e6b_registration.json`](e6b_registration.json).
>
> **Verdict, three separate findings.**
>
> 1. **P2 does not scale.** P2-1.60M → P2-2.96M is **+0.0267** usable rollout —
>    inside the 0.0800 floor, and the seeds disagree in direction (+0.0600 /
>    −0.0067). A tie. Correctness likewise (−0.0100).
> 2. **At 2.96M the KD-heavy objective wins on behaviour.** P2-2.96M against
>    E1-2.96M is **−0.0800** usable — at the floor, **−0.0800 on *both* seeds**,
>    both bootstrap CIs excluding zero. Correctness ties (−0.0167).
> 3. **The objective interacts with data scale.** The
>    difference-in-differences on usable rollout is **−0.0833**, above the floor
>    and consistent in direction across seeds. E1 converts the extra rung into
>    +0.1100 of rollout stability; P2 converts it into +0.0267.
>
> **`e1_r2960k_{sa,sb}_pca` remains the best evaluated checkpoint.** E6b does not
> displace it; it removes the possibility that E6's plateau was a property of the
> data rather than of the E1 objective.

### 29.1 Headline

150 fixed examples, mask `d6e24e0b…` asserted on all eight arms, greedy,
unrestricted (P18). Every arm re-scored from raw generations with the current
scorer. "Unique" is the rung; "cumulative" is 3× it (3 exposures, verified).

| model | unique CE | cumulative CE | seed | usable | correct | correct \| usable |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| E1/P1 1.60M | 1,600,353 | 4,801,059 | sa | 0.7800 | 0.1733 | 0.2222 |
| E1/P1 1.60M | 1,600,353 | 4,801,059 | sb | 0.6800 | 0.2000 | 0.2843 |
| **E1/P1 1.60M** | | | **mean** | **0.7300** | **0.1867** | 0.2511 |
| E1/P1 2.96M | 2,960,507 | 8,881,521 | sa | 0.8533 | 0.2133 | 0.2500 |
| E1/P1 2.96M | 2,960,507 | 8,881,521 | sb | 0.8267 | 0.2000 | 0.2419 |
| **E1/P1 2.96M** | | | **mean** | **0.8400** | **0.2067** | 0.2460 |
| P2 1.60M | 1,600,353 | 4,801,059 | sa | 0.7133 | 0.2333 | 0.3178 |
| P2 1.60M | 1,600,353 | 4,801,059 | sb | 0.7533 | 0.1667 | 0.2212 |
| **P2 1.60M** | | | **mean** | **0.7333** | **0.2000** | 0.2682 |
| P2 2.96M | 2,960,507 | 8,881,521 | sa | 0.7733 | 0.1800 | 0.2241 |
| P2 2.96M | 2,960,507 | 8,881,521 | sb | 0.7467 | 0.2000 | 0.2679 |
| **P2 2.96M** | | | **mean** | **0.7600** | **0.1900** | 0.2456 |

### 29.2 The three comparisons, kept separate

**A — same-scale objective effect: P2-2.96M vs E1-2.96M.**

| seed | usable | correct | usable win/tie/loss | usable 95% CI |
| --- | --- | --- | --- | --- |
| sa | 0.8533 → 0.7733 (**−0.0800**) | 0.2133 → 0.1800 (−0.0333) | 10/118/22 | [−0.1533, −0.0067] **excl. 0** |
| sb | 0.8267 → 0.7467 (**−0.0800**) | 0.2000 → 0.2000 (0.0000) | 11/116/23 | [−0.1533, −0.0067] **excl. 0** |

−0.0800 pooled, **identical on both seeds**, both CIs excluding zero. At 2.96M
the KD-heavy objective is better on the primary axis. Correctness −0.0167, inside
the floor: a tie.

**B — P2 scaling effect: P2-2.96M vs P2-1.60M.**

| seed | usable | correct | usable win/tie/loss | usable 95% CI |
| --- | --- | --- | --- | --- |
| sa | 0.7133 → 0.7733 (+0.0600) | 0.2333 → 0.1800 (−0.0533) | 25/109/16 | [−0.0200, +0.1467] |
| sb | 0.7533 → 0.7467 (−0.0067) | 0.1667 → 0.2000 (+0.0333) | 21/107/22 | [−0.0933, +0.0800] |

+0.0267 pooled, inside the floor, **seeds disagree in direction**. P2 does not
benefit from the rung. Correctness also a tie, also seed-inconsistent.

**C — E1 scaling reference (re-derived, matches E6 §28 exactly).**
+0.1100 usable, above the floor, seed-consistent (+0.0733 / +0.1467).

### 29.3 Objective × scale interaction

`(P2_2.96 − P2_1.60) − (E1_2.96 − E1_1.60)`, per metric:

| metric | better | P2 Δ | E1 Δ | interaction | per seed | consistent | claimable |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| usable_rollout_rate | higher | +0.0267 | +0.1100 | **−0.0833** | −0.0133 / −0.1533 | yes | **yes** |
| correct_overall | higher | −0.0100 | +0.0200 | −0.0300 | −0.0933 / +0.0333 | no | no |
| correct_given_usable | higher | −0.0235 | −0.0073 | −0.0162 | −0.1215 / +0.0891 | no | no |
| natural_termination_rate | higher | +0.0133 | +0.1000 | −0.0867 | −0.0067 / −0.1667 | yes | no floor |
| context_limit_rate | lower | −0.0133 | −0.1000 | +0.0867 | +0.0067 / +0.1667 | yes | no floor |
| severe_repetition_rate | lower | −0.0100 | −0.0933 | +0.0833 | −0.0000 / +0.1667 | no | no |
| empty_output_rate | lower | −0.0166 | −0.0566 | +0.0400 | −0.0333 / +0.1133 | no | no |
| answer_parse_failure (numeric) | lower | −0.0666 | −0.1266 | +0.0600 | +0.0799 / +0.0401 | yes | no floor |

**Only `usable_rollout_rate` clears its floor with a consistent direction, and it
is the primary axis, so the interaction is claimable there.** Every metric
without a registered floor is reported and not claimed, however suggestive.

**The honest caveat on magnitude.** The per-seed interactions are −0.0133 and
−0.1533. They agree in *direction* but differ by an order of magnitude, so the
pooled −0.0833 is carried almost entirely by `sb`. A difference-in-differences
over four two-seed cells stacks four single draws; this one satisfies the
registered rule, and it is the weakest kind of evidence that rule admits. It
should be treated as "P2 does not convert the rung the way E1 does" rather than
as a calibrated effect size.

### 29.4 Components and counts

| arm | usable/150 | correct/150 | c∧u/usable | protocol | nat.term | ctx-limit | repetition | empty | parse-fail | p50 / p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1-1.60M-sa | 117 | 26 | 26/117 | 0.7800 | 0.8133 | 0.1867 | 0.1867 | 0.1400 | 13/75 | 576 / 8117 |
| E1-1.60M-sb | 102 | 30 | 29/102 | 0.6933 | 0.7067 | 0.2933 | 0.3067 | 0.2200 | 18/75 | 616 / 8119 |
| E1-2.96M-sa | 128 | 32 | 32/128 | 0.8733 | 0.8733 | 0.1267 | 0.1467 | 0.1200 | 2/75 | 585 / 8098 |
| E1-2.96M-sb | 124 | 30 | 30/124 | 0.8333 | 0.8467 | 0.1533 | 0.1600 | 0.1267 | 10/75 | 586 / 8105 |
| P2-1.60M-sa | 107 | 35 | 34/107 | 0.7133 | 0.7467 | 0.2533 | 0.2533 | 0.2333 | 9/75 | 566 / 8123 |
| P2-1.60M-sb | 113 | 25 | 25/113 | 0.7600 | 0.7867 | 0.2133 | 0.2200 | 0.1933 | 10/75 | 731 / 8108 |
| **P2-2.96M-sa** | 116 | 27 | 26/116 | 0.7867 | 0.8000 | 0.2000 | 0.2133 | 0.1800 | 4/75 | 688 / 8105 |
| **P2-2.96M-sb** | 112 | 30 | 30/112 | 0.7467 | 0.7600 | 0.2400 | 0.2400 | 0.2133 | 5/75 | 653 / 8117 |

**Where the two objectives diverge is termination.** From 1.60M to 2.96M, E1's
context-limit hits fall 28/44 → 19/23 prompts; P2's fall only 38/32 → 30/36.
E1 converts the extra data into trajectories that stop; P2 largely does not.

**Where they agree is parsing.** Numeric answer-parse failures fall on *both*
objectives (E1 13/18 → 2/10, P2 9/10 → 4/5 of 75 numeric prompts). More data
teaches both to emit a recognisable final answer.

**GSM8K, the sharpest single view:** usable rollout rises for both objectives
(E1 0.71 → 0.91, P2 0.72 → 0.84) while GSM8K correctness stays at **0.00–0.05
everywhere**. Neither objective at any rung learns to solve a grade-school maths
problem; both learn to finish one.

### 29.5 Diagnostics — both objectives improve; only one improves behaviour

| arm | final val CE | teacher-forced reasoning top-1 |
| --- | ---: | ---: |
| E1-2.96M sa / sb | 1.1468 / 1.1486 | 0.6152 / 0.6213 |
| P2-1.60M sa / sb | 1.3112 / 1.3140 | 0.5815 / 0.5818 |
| **P2-2.96M sa / sb** | **1.1694 / 1.1740** | see `logs/e6b_results.json` |

P2's val CE improves with scale exactly as E1's does (1.31 → 1.17 against
1.30 → 1.15), while its rollout behaviour does not. This is the third independent
instance of the CE/behaviour dissociation, and the cleanest: the two objectives
move nearly identically on the diagnostic and differently on the primary axis.
**Diagnostics may not select a checkpoint** — E6b is the strongest evidence yet
for that rule.

### 29.6 Conclusions, against the registered rules

* **Best existing evaluated checkpoint:** `e1_r2960k_{sa,sb}_pca`, unchanged.
* **Best objective at 2.96M:** **E1/P1 KD-heavy**, on the primary axis, at the
  floor, seed-consistent, both CIs excluding zero.
* **P2 scale trend:** **does not improve** — a tie, with the seeds disagreeing.
  Whether it is a true plateau or simply a smaller gain than this design can
  resolve is not separable at n=2.
* **Objective × scale interaction:** **evidence of interaction on the primary
  axis**, with the magnitude caveat in §29.3.
* **Correctness vs stability:** P2-2.96M does neither. It is not the E1 pattern
  of "more usable but not more correct" — it is not materially more of either.
* **P2-5.50M:** **not justified.** P2 did not convert the 1.60M → 2.96M rung, so
  there is no preregistered basis to expect it to convert a larger one. Not
  initiated; a separate decision if ever wanted.

### 29.7 Execution, cost, and an overrun that must be recorded

Pod `luy1txyjro2msz`, L40S at $0.99/h, 17:17 → 00:56 = **458 min = $7.56**, plus
**$0.12** for a pod that died at `INIT_READY` on an unforwarded environment
variable (§29.8) — **$7.68 total against a $7.12 authorization. Overrun $0.56.**

**Root cause of the overrun: the step time.** The cost model used 3.625 s/step,
measured from E4's P2 arms on the same GPU, image, batch and block length. The
run measured **4.148 / 4.110 s/step** — 14% slower for reasons not attributable
to anything in the recipe. That alone is 49 min ≈ $0.81 of unbudgeted time, and
it put the session over the backstop *before* any teardown decision existed.

**Two independent stop mechanisms were inert at the same time, which is the more
important finding.**

1. **The launcher's ssh call to start the driver did not detach.** It used the
   same `setsid nohup … < /dev/null & disown` form that returned in ~74 s in E6,
   and here it blocked for the whole 434-minute run. The launcher therefore never
   reached its polling loop, never saw `ALL_DONE`, and could not tear down at
   completion. Cause not established; the invocation is byte-identical to E6's,
   so it is a property of what the driver runs, not of how it is launched.
2. **RunPod's `--terminate-after` did not fire.** The deadline was set to
   00:28:47 at creation and the pod was still `RUNNING` at 00:34. This flag has
   been the documented last-resort budget layer since E4 and **has never once
   been observed to fire** — every previous session was torn down by its
   launcher first. It was trusted, not tested.

With both inert, nothing would have stopped the pod except the work finishing,
which it did at 00:31:56. The monitoring gap is mine: the watcher tailed the
orchestrator *log*, and a blocked launcher writes no lines, so seven hours of
silence was indistinguishable from seven hours of nothing happening. **Poll the
pod, not the log.**

**Retained:** both checkpoints (5.6 GB each, sha256 `89b14b83…` / `3c4709b5…`,
verified local-vs-pod), all four generation sets, the driver console log, and the
pod-side hash manifests. **Lost:** the structured `train_log.jsonl` and
`run_manifest.json` for both arms. The training curve, per-step timings and final
val CE survive in `e6b_run.log`; the machine-readable event stream (AGENTS.md
3.7) does not. A P4 gap, recorded rather than papered over.

> **Correction, 2026-08-09.** The sentence that stood here attributed the loss to
> "the bundling command's `$(ls -d …)` globs [that] did not expand inside the ssh
> quoting". **That is wrong**, and the corrected cause changes the fix. The E6b
> bundling command at `6375e29` contains no glob at all: it names
> `artifacts/audit/three_mode` and two E6-specific JSON files, inherited verbatim
> from E6 — a session that did not train — so `train_log.jsonl` and
> `run_manifest.json` were **never listed**. Two of the three named paths do not
> exist in an E6b session either; `2>/dev/null` swallowed the error and the
> `;`-chained `sha256sum` ran anyway. `tar tzf` on the retrieved bundle confirms
> it holds `artifacts/audit/three_mode/**` and nothing else. Every downstream
> check then passed *on the incomplete bundle* — tar produced a file, the digests
> matched, the transfer verified — because none of them asked whether everything
> that had to survive was present. The `$(ls -d …)` construct is a real and
> separate fragility, still present in `e3/e4/e5_launch.sh`, and is now banned
> for new launchers by lint. Full record:
> [`e6b_protocol_deviations.md`](e6b_protocol_deviations.md); remediation §30.

**Derived replacement.** [`e6b_reconstructed_training_events.json`](e6b_reconstructed_training_events.json),
parsed from the surviving console log by
`scripts/pod/reconstruct_training_events.py`, carries
`"provenance": "reconstructed_from_driver_console"` and
`"original_event_stream_available": false` plus a per-field provenance block. It
recovers 291 `train_step` and 10 `eval_result` events per arm and is **not** the
original event stream — `grad_norm`, the token accounting, `gpu_mem_gb` and the
`run_start`/`teacher_loaded`/`checkpoint_saved`/`run_end` events were never
printed and are gone.

It also separates two quantities that "4.15 s/step" conflated: the **printed
per-step** timing means 4.1485 (sa) / 4.1099 (sb), while **wall clock per step**
between the driver command and `TRAIN_DONE` is 4.211 / 4.215 — the difference
being the ten periodic evaluations and the checkpoint writes. Future budgets
price the step at 4.15 s and name evaluation and checkpointing as separate
phases.

### 29.8 The $0.12 pod

`e6b_setup.sh` reads `TEACHER_REVISION` because E6b computes KD. Its launcher was
derived from E6's, which had no teacher and never forwarded that variable, so the
pod died with a bare `KeyError` at `INIT_READY` — after both venvs were built and
the Stage 1 init downloaded and hash-verified. No existing guard could see it:
the pod simulation runs the test suite, not the setup script, and `bash -n`
parses fine. `tests/pod/test_launcher_forwards_setup_env.py` now checks, for all
21 launcher/setup pairs, that every variable a setup consumes is forwarded and
defaulted. Verified to bite: with the fix reverted it names `TEACHER_REVISION`.

### 29.9 Exact reproduction record

Commit **`6375e299815416dddc1bd0c12fd6fe273035a9e9`**.

```bash
PYTHONPATH=src python scripts/training/build_e6b_configs.py
PYTHONPATH=src python scripts/training/register_e6b.py --authorized-usd 7.12
bash scripts/pod/simulate_pod_env.sh
SCR=… SESSION_COMMIT=6375e299815416dddc1bd0c12fd6fe273035a9e9 \
  BUNDLE_NAME=aad_e6b_6375e299.bundle AUTHORIZED_USD=7.00 \
  setsid nohup bash scripts/pod/e6b_launch.sh &
PYTHONPATH=src python scripts/evaluation/analyze_e6b.py --bootstrap 10000
```

| identity | value |
| --- | --- |
| config sha256 | sa `963aa00ead167682…` · sb `da71974841a39b96…` |
| final `model.safetensors` | sa `89b14b839ff9b8a2…` · sb `3c4709b51792c7e6…` |
| Stage 1 fork point | `86fbba78e8a2a324…`, asserted on the pod before training |
| teacher | `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d9e…` |
| rung | 1,944 blocks · 2,960,507 unique CE tokens · 3.0000 exposures · 8,881,521 cumulative |
| nesting | strict token-level prefix of 1.60M on `input_ids`, `ce_mask`, `content_mask` |
| objective | ce 1.0 / kd 0.25, τ 1.0, scope all |
| evaluation | mask `d6e24e0b…` on all 8 arms · 150 prompts · greedy · ctx 8192 · vLLM 0.26.0 |
| artifacts | `logs/e6b_results.json`, `logs/e6b_report.md`, `artifacts/audit/e6b_per_prompt.jsonl` (1,200 records) |

---

## 30. Operational hardening after E6b (2026-08-09, CPU, $0)

**Objective.** Fix the orchestration and artifact-retention failures that made
E6b operationally noncompliant, before any further billable run. No GPU: this is
control flow, and control flow does not need to be paid for.

**Scope.** E6b's scientific endpoints are unchanged and neither arm was rerun.
The deviations themselves are recorded permanently in
[`e6b_protocol_deviations.md`](e6b_protocol_deviations.md).

### 30.1 What failed, and what now catches it

| E6b failure | layer that missed it | fix |
| --- | --- | --- |
| driver-start ssh blocked 434 min | none — the launcher *was* the orchestration | `remote.start_detached`: bounded start, durable pod-side job descriptor, out-of-band confirmation |
| launcher never polled, never tore down | — | the launcher returns within `start_timeout + verify` regardless of the channel |
| `--terminate-after` inert | it *was* the last-resort layer | `watchdog.Watchdog`: polls the provider, terminates, **verifies the pod is gone**, retries, journals |
| seven hours of silence read as idle | watcher tailed the orchestrator log | `watchdog.SessionWatcher`: requires provider state; no verdict is reachable from markers alone |
| authorization *was* the kill point | — | `budget.plan_session`: expected / soft stop / recovery reserve / hard terminate |
| priced from a superseded 3.625 s/step | — | 4.15 s/step floor, enforced; a lower estimate needs a recorded reason |
| event stream lived only on the pod | — | `log_relay.LogRelay`: continuous incremental mirroring |
| bundle list inherited from a non-training session | tar exited 0, digests matched | `artifact_gate`: declared spec expanded in Python, archive built from the manifest, ordered teardown gate |

### 30.2 The four budget thresholds

E6b had one number. It now has four, and they answer different questions:

| threshold | meaning |
| --- | --- |
| `expected` | what the session costs if nothing goes wrong — the number in an authorization request |
| `soft_stop` | past this, no **new** phase may start; running work continues |
| `artifact_recovery_reserve` | time held back *after* the soft stop for bundling, hashing, transfer and verification |
| `hard_terminate` | `soft_stop + reserve`; the watchdog's kill point, which must land **at or under** the authorization |

`may_start(elapsed, phase_minutes)` is the gate E6b lacked: its driver re-priced
each arm against the *authorization*, so an arm that would finish just under the
cap was allowed to start, leaving nothing for teardown.

Priced at the measured 4.15 s/step, an E6b-shaped session (2 arms × 2,916 steps,
25 min setup, 10 min/arm evaluation, 20 min transfer, 30 min reserve) terminates
at **$9.00** — so `plan_session` **refuses** it at $7.12 and reports the shortfall
rather than shrinking the run to fit. That refusal is the arithmetic that would
have caught the overrun before launch.

### 30.3 Verification

All local, no GPU. **986 tests pass and 3 are skipped** (879 before this work,
**+110 new**). The 3 skips are the deliberate frozen-record launcher exemptions
in the `$(ls …)` lint.

```bash
PYTHONPATH=src pytest tests/ -q
bash scripts/pod/simulate_pod_env.sh
```

New coverage:

| file | tests | what it holds |
| --- | ---: | --- |
| `tests/infrastructure/test_remote_launch.py` | 9 | a 600-second job must not delay the launcher past 30 s, including when the start channel never closes |
| `tests/infrastructure/test_watchdog.py` | 15 | terminate/verify/retry; an ignored termination is retried; silence never yields an idle verdict |
| `tests/infrastructure/test_budget.py` | 11 | threshold ordering; the E6b plan is refused at its real step time; the superseded figure is refused by name |
| `tests/infrastructure/test_log_relay.py` | 9 | the pod is deleted mid-run and 291 already-synced events remain readable |
| `tests/infrastructure/test_artifact_gate.py` | 19 | a missing event stream blocks teardown; a short archive that self-verifies is detected against the full manifest |
| `tests/infrastructure/test_e6b_failure_simulation.py` | 15 | the whole 2026-08-08 sequence replayed against the hardened stack |
| `tests/pod/test_operational_hardening.py` | 23 | the entry points run; `$(ls …)`-in-ssh is banned for new launchers |
| `tests/pod/test_reconstruct_training_events.py` | 9 | the derived artifact declares its provenance and classifies every original field |

The simulation exercises, in one file: successful detached launch; a remote
driver that dies at startup; blocked SSH; a silent orchestrator over a billing
pod; watchdog termination; a provider that accepts a termination and ignores it;
a provider that cannot terminate at all; partial transfer; hash mismatch; missing
structured logs; safe teardown; and emergency budget teardown.

Two behaviours are asserted by wall clock rather than by inspection, because
those are the ones E6b got wrong: `start_detached` must return in under 30 s
against a 600-second job, and must do so even when the start channel is made to
hang.

### 30.4 What is not covered

* **RunPod's live control plane is untested here.** Polling uses the GraphQL
  query every launcher since E2 has used, so its shape is confirmed by use.
  Termination tries `runpodctl remove pod` first — the call every session in this
  project has actually made — and falls back to a `podTerminate` mutation taken
  from the public schema that **has never been exercised from this repo**; it is
  journalled as `verified_transport: false`. The guarantee that does not depend
  on either is the verification poll: termination means the pod is gone, not that
  a request returned 200.
* **`--terminate-after` is still not known to work.** It is retained as a
  redundant third layer and is no longer counted as a stop mechanism.
* **The E6b blocking cause is still not diagnosed.** The fix removes the
  dependency rather than explaining it: the launcher no longer needs the channel
  to close.

### 30.5 Exact reproduction record

```bash
PYTHONPATH=src pytest tests/infrastructure tests/pod -q
PYTHONPATH=src python scripts/pod/watchdog.py --pod-id sim-1 \
  --session-start-epoch 0 --price-per-hour 0.99 --hard-minutes 1 \
  --authorized-usd 9.00 --journal /tmp/wd.jsonl --once --simulate
PYTHONPATH=src python scripts/pod/reconstruct_training_events.py \
  --run-log /home/ecs-user/aad-artifacts/e6b/e6b_run.log \
  --status  /home/ecs-user/aad-artifacts/e6b/e6b.status \
  --config configs/stage3/e6b/e6b_p2_r2960k_sa.json \
  --config configs/stage3/e6b/e6b_p2_r2960k_sb.json \
  --out logs/e6b_reconstructed_training_events.json
```

**Verdict: complete.** The next billable run may be planned. It must use
`plan_session` for its thresholds, `start_job.py` for its driver,
`watchdog.py` beside the launcher, `LogRelay` for its event streams and
`collect_artifacts.py` for its teardown.

---

## 31. Experiment 7 — design, implementation and preregistration (2026-08-09, CPU, $0)

**Objective.** Prepare E7 — *FineWeb teacher-KD mixture at the fixed 1.60M
rollout rung* — to the point where launching it is a budget decision and nothing
else. No GPU was used. **Nothing has been trained or evaluated, and E7 is not
authorized.**

Full design: [`e7_preregistration.md`](e7_preregistration.md). Decision records:
[`decisions.md`](decisions.md), 2026-08-09.

### 31.1 The question, and the fact behind it

Held-out FineWeb NLL across the E1/P1 KD-heavy lineage improves to **6.16** by
the 0.46M rung and then **gives it back** — 8.88 at 0.86M, **9.71 / 9.48 at the
1.60M rung this experiment trains at**, 10.40 / 9.79 at 2.96M — against a Stage 1
init of 11.75. Over the same range autonomous correctness never leaves 0.11–0.21.
E7 asks whether the two are connected, by adding a strictly additional general
-text KD signal and holding everything else fixed.

**A predicted null is a real outcome.** The design makes "general LM restored,
behaviour unchanged" a clean answer rather than a failed run.

### 31.2 What was verified from the loader, not assumed

Reading `blocks[:1174]` of the canonical pack (`blocks.npz` sha256
`6f324cb0f37bc0f0…`, matching `scripts/pod/hashes_ladder.txt`):

| quantity | value |
| --- | ---: |
| CE targets per exposure | 1,600,353 |
| **cumulative CE exposure (x3)** | **4,801,059** ✓ as specified |
| KD positions per exposure (`scope: all`) | 2,660,125 |
| cumulative rollout KD positions | 7,980,375 |
| packing efficiency at this rung | 0.2767 |
| 1,761 steps x 2 blocks | exactly 3.0 exposures |

### 31.3 The design decisions that mattered

**The extra text is a second stream, not a bigger pack.** Merging it would move
every block boundary and every example's position against the LR schedule, and
the arm could not be compared to the retained baseline at all. Instead a second
cursor is consumed inside the same optimizer steps, with independent
normalizers — which matters concretely, because a pooled mean would make each
stream's effective weight depend on the other's packing efficiency and the
rollout pack is 72% padding here.

**Dense streams make the control exact.** Both extra streams are 1761 x 1024 with
no padding, so KD positions are `n_blocks x (block_len - 1)` = **1,801,503** by
construction, identical for B and C, as are forward tokens, CE positions (zero)
and the microbatch schedule. **Compute matching is exact; there is no mismatch to
report before training.**

**Arm C is in-domain, and that is a stated limitation.** It draws content tokens
from pack blocks `[1174, 1853)` — after the trained rung, before the validation
tail — re-packed densely. E6 showed more in-domain data improves stability, so C
is a *strong* control: if it matches B, the reading is "extra KD positions did
it", not "FineWeb did nothing". Recorded in advance rather than discovered after.

**One λ, no sweep.** `lambda_extra = 0.25`, chosen because rollout KD *falls*
through training (E6b val_kd 10.60 → 1.04) while FineWeb KD stays high, so λ near
1.0 would make general text the dominant late gradient. A **non-training**
preflight measures the gradient-norm ratio against a registered [0.05, 1.00] band
and stops if outside; tuning λ from that measurement is forbidden.

### 31.4 The FineWeb holdout was too small, and is now larger without being replaced

`holdout_v1` is 40 documents (~25k tokens) against between-seed `holdout_nll`
spreads of 0.23 / 0.62 / 1.34 nats at the three top rungs. `e7_fineweb_val` is
512 x 1024 = 524,288 tokens, **20x larger**, disjoint from everything.
`holdout_v1` is preserved and still measured so the historical series stays
continuous; the two are separate columns and are never merged.

### 31.5 Disjointness, and an incidental finding in the frozen battery

The proof covers index ranges **and** content hashes across the three E7 streams,
`holdout_v1`, `warmup_v1`, the behaviour prompts and all seven `capability-v2`
files. **Zero overlaps involving any E7 stream.**

It also surfaced something that is not E7's: `rag.jsonl` and
`answerability_paired.jsonl` share the SQuAD item
`squad-val-57299021af94a219006aa50c` with byte-identical prompt text, ids
differing only by a `pair-0118-safe:` prefix. Exactly 1 of 846 prompts, zero
within-file duplicates. The frozen battery is **not** being rebuilt — the
magnitude is negligible and rebuilding would break comparability with every
result scored on it — but `rag` and `answerability_paired` are not fully
independent subsets and per-subset comparisons must say so.

### 31.6 Verification

**1,029 tests pass, 3 skipped** (+43 over the post-hardening 986). New coverage:

| file | tests | what it holds |
| --- | ---: | --- |
| `tests/training/test_dual_stream.py` | 19 | rollout block order, LR positions, step count, CE/KD terms and normalizers unchanged by the extra stream; zero extra CE; padding refused; B/C budgets matched on the **shipped** configs; both cursors reproduce on resume; planned budget == consumed budget |
| `tests/data/test_e7_streams.py` | 14 | stream identity and pinning; reserved index ranges avoided; control avoids the trained rung and the validation tail; leakage check fails closed and is verified to bite on a planted overlap; a missing `train_log.jsonl` blocks teardown against E7's own artifact spec |
| `tests/evaluation/test_general_text.py` | 10 | known-answer NLL/rank/KL, batching and chunking invariance, empty stream raises |

### 31.7 Costs, and what is open

Priced at 4.60 s/step (E6b's measured 4.15 + 10% for the extra stream), setup
budgeted at 45 min rather than the warm-image 5–8.5:

| | expected | soft stop | reserve | hard terminate |
| --- | ---: | ---: | ---: | ---: |
| live control-plane canary | $0.53 | $0.66 | $0.17 | **$0.82** |
| **E7 full (B×2 + C×2)** | $11.20 | $12.32 | $0.49 | **$12.82** |
| E7 reduced (B×2), attribution-incomplete | $6.07 | $6.68 | $0.49 | **$7.17** |

**Open:** authorization and a cumulative-cap increment above the **$149.59**
actual baseline (**$163.23** full, $157.59 reduced); the live provider control
plane, still unverified ([`e7_canary_proposal.md`](e7_canary_proposal.md)); the λ
preflight, implemented but needing a real teacher/student; arm A's general-text
baseline on the new validation stream, priced into the session.

**Verdict: preparation complete.** Nothing further can be learned about E7
without spending money.

---

## 32. Live RunPod control-plane canary — FAILED 9/10 (2026-08-09, $0.045)

**Objective.** Verify on a real disposable pod the ten control-plane behaviours
the local simulator cannot, before a 776-minute E7 session became their first
live test. Authorized at a $0.82 hard backstop, temporary cumulative cap
$150.41.

**Verdict: FAILED.** Nine criteria passed; artifact manifest + local hash
verification failed on a real defect. Per the authorization, **E7 was not
launched.** Full report: [`e7_canary_report.md`](e7_canary_report.md).

### 32.1 Provider-level record

Pod **`bd83jug4g23qn0`**, NVIDIA RTX 2000 Ada Generation at **$0.240/h** — the
cheapest of six quoted candidates, chosen because the canary needs no GPU
compute. Created **11:35:03 UTC**, provider-confirmed terminated **11:46:10
UTC**: **11.12 min = $0.0445**. Final state `exists=False · TERMINATED ·
billing=False`, `runpodctl pod list` empty. `--terminate-after` was set to
12:25:02 and never reached — still never observed to fire, still not counted.

### 32.2 The GraphQL fallback is no longer unverified

Forced onto it by pointing the watchdog at a binary that does not exist — **no
provider state altered**, only that process's view of the CLI:

```
runpodctl remove pod   ok=false  FileNotFoundError: '/nonexistent/runpodctl-canary-forced-failure'
graphql podTerminate   ok=true   {"data": {"podTerminate": null}}
terminate_verify       exists=false  desired_status=TERMINATED  billing=false
```

The threshold crossing was genuine: polls at 10.40 and 10.74 min under the
limit, the poll at **11.09 min over it**. The *safety* watchdog — separate
process, real CLI, own journal — independently observed the pod go from
`RUNNING/billing` to `TERMINATED/not billing` and exited `pod_gone`.

### 32.3 The failure, and why the fix is not "relax the check"

The pod-side manifest hashed `canary/train_log.jsonl` at 2,166 bytes; `tar` read
it at **2,230** a moment later, because the job was still appending. The gate
blocked at `archive_contents_verified`.

**The gate was right and the workflow was wrong.** A training log is appended for
the whole run, so E7 would have hit this every time with nothing actually
missing — and the tempting fix, tolerating a size difference, would have
re-opened the exact hole E6b fell through. Instead `create_archive` now reads
each file once, capped at the size seen when it opens, hashes the bytes it
writes, and rewrites the manifest to them; growth is recorded in
`appended_during_archive`, and a file that has *shrunk* is a hard error.

### 32.4 Three defects, two caught before any pod existed

1. **RunPod 403s the default `Python-urllib` User-Agent** — found while quoting
   prices, at zero cost. Every watchdog poll would have failed on a live pod, and
   since an unanswered poll counts as billing, every session would have ended in
   `TERMINATION_FAILED` against a pod that had actually died.
2. **Manifest/archive non-atomicity** — found live (§32.3).
3. **`watchdog.py` lacked the `--runpodctl` flag `canary.py` passed it**, so the
   test watchdog died on argparse before its first poll. Undetected, because the
   launch is fire-and-detach. The flag was added and the watchdog relaunched by
   hand against the same live pod inside the same budget: **criteria 5–8 are
   genuine, but the driver did not achieve them unaided.**

### 32.5 Verification

**1,060 tests pass, 3 skipped** (+18). New regression coverage: the User-Agent
and the full provider transport surface (13); the growing-log archive sequence
reproduced end to end; and an assertion that **every flag `canary.py` passes to
`watchdog.py` exists**, which is the class of bug that a detached launch hides
by design.

### 32.6 What is still unknown

Long-session behaviour — 11 minutes exercises nothing like E6b's 434-minute
block. `--terminate-after`, still never observed to fire. And the fixed artifact
path is covered by tests but **not re-verified on a live pod**.

---

## 33. Control-plane canary rerun — PASSED 12/12 (2026-08-09, $0.033)

**Objective.** One run of the complete automated chain with no human repair,
after run 1 (§32) failed 9/10 and needed a hand-launched watchdog to recover
criteria 5–8. Authorized at a **$0.12** hard backstop; no cap increase.

**Verdict: PASSED, all twelve criteria, from one launch command.** Report:
[`e7_canary_rerun_report.md`](e7_canary_rerun_report.md).

### 33.1 Provider record

Pod **`3hvb5d4it6h6pb`**, NVIDIA RTX 2000 Ada at **$0.240/h**, created
**12:09:24**, provider-confirmed terminated **12:17:42** — **8.30 min =
$0.0332**. Final state `exists=False · TERMINATED · billing=False`;
`runpodctl pod list` empty. `--terminate-after` at 12:39:23 was never reached.

**The backstop was derived from the live quote**, not assumed: $0.12 at $0.240/h
buys 30.0 min, which became the backstop and the RunPod deadline. Below 20 min
the driver aborts and reports rather than widening it — the dollar figure is the
ceiling and minutes are whatever fits inside it.

### 33.2 The chain, unaided

Detached launch **5.46 s** · descriptor readable over a later connection · **27
events relayed while the job was still `ALIVE`** · watchdog polled the billing
pod 14 times · the broken primary CLI failed exactly as intended
(`FileNotFoundError`, `verified_transport=true`) · **the same watchdog, on its
first and only launch, crossed its own threshold — five polls under the limit,
then `hard_limit_reached` at 8.27 min — and invoked the GraphQL fallback
itself** · `podTerminate` returned `{"data": {"podTerminate": null}}` · the
first verify poll showed `exists=False · TERMINATED · billing=False` · the
complete sequence `watchdog_start → poll ×6 → hard_limit_reached →
terminate_attempt → terminate_verify → terminated → watchdog_end` is 2,840 bytes
on disk · launches `{safety: 1, test: 1}`, `phase_2_invoked=False`.

### 33.3 Both artifact lifecycles, live, on the same file

| phase | lifecycle | bytes | grew during archive | gate |
| --- | --- | ---: | --- | --- |
| snapshot (writer live) | `mutable_snapshot` | 1,910 | **yes**, 1,846 → 1,910 | allowed |
| final (after `MARKER:ALL_DONE`) | `final_required` | 3,190 | no | allowed |

The snapshot phase **is run 1's failure, now passing**: the log grew by 64 bytes
between manifest and archive, the bounded read captured the boundary, hashed
what it wrote and recorded the growth, and local hashes matched. It proves
durability and claims nothing more. The final phase — 6-second settle window,
terminal-marker verification, no growth, no `still_being_written` — is the shape
a normal E7 teardown uses, and is the only shape that can pass
`final_streams_quiescent`.

### 33.4 The correction that made the distinction safe

The bounded read fixed run 1's hash race but made a growing log *archivable*,
and archivable must not mean finished. Enforcement now sits in three places:
`build_manifest` verifies completion markers and watches each `final_required`
file across a settle window; `create_archive` refuses outright to archive a
`final_required` file that grew under it; and the gate carries
`final_streams_quiescent` separately from `required_files_present`, because
presence and completeness are different questions. Emergency teardown may keep a
snapshot but must **name** the streams it truncates — an unnamed truncation
raises rather than passing quietly.

### 33.5 Four defects across the two runs

| defect | found | cost |
| --- | --- | --- |
| RunPod 403s the `Python-urllib` User-Agent | pre-flight, before any pod | $0 |
| manifest/archive hash race on an appending file | run 1, live | $0.045 (that run) |
| a bounded prefix could pass as a final artifact | the correction | $0 |
| `watchdog.py` lacked a flag `canary.py` passed it | run 1, live | — |

**1,071 tests pass, 3 skipped.** Total canary spend across both runs: **$0.078**.

### 33.6 Deliberate limits

The canary proves the provider/control-plane path and nothing about multi-hour
sessions; long-session watchdog and liveness behaviour is covered by
deterministic local simulation against a fake clock, which is cheaper and
strictly more controllable than approximating E6b's duration live.
`--terminate-after` is still never observed to fire and still does not count.

**E7 B and C are not launched.** They require separate authorization for the
$12.82 hard backstop and the $163.23 cumulative cap.

---

## 34. Experiment 7 — general language modelling restored; behaviour unmoved (2026-08-09, $10.49)

**Objective.** Does adding general-text teacher KD, while preserving the 1.60M
rollout trajectory exactly, restore general language modelling — and does any
restoration transfer to autonomous rollout correctness?

**Answer: yes to the first, no to the second.** This is preregistered outcome 2
(`e7_preregistration.md` §7.4): *FineWeb preserves language modelling but does
not solve reasoning.* Report [`e7_report.md`](e7_report.md), machine-readable
[`e7_results.json`](e7_results.json), session evidence
[`e7_session_evidence.json`](e7_session_evidence.json).

### 34.1 The session

Pod `spa2i4615a10wu`, L40S at $0.99/h, 12:41:50 → 23:17:21 UTC = **635.5 min =
$10.49** against a $12.82 backstop and a $12.32 soft stop. Four arms trained,
six models diagnosed, four evaluated on the frozen battery. The gate passed, the
pod was deleted by the launcher, and the provider confirmed it gone.

Preflight, the registered stop/go gate at the frozen `lambda_extra = 0.25`:

| config | ‖∇(λ·KD_extra)‖ / ‖∇(rollout)‖ | band [0.05, 1.00] |
| --- | ---: | --- |
| B | **0.3613** | IN BAND |
| C | **0.3876** | IN BAND |

Both arms landed at nearly the same gradient scale, which is what makes the
comparison about content rather than about one arm being pushed harder. **All
four arms consumed exactly 1,801,503 extra KD positions** — `planned` at
`run_start`, `extra_kd_positions` at `run_end`, matching to the token.

### 34.2 General language modelling: restored, decisively

On the 512×1024 held-out FineWeb stream, teacher `Qwen3-4B-Thinking-2507`:

| arm | FineWeb NLL | teacher KL | top-1 | mean rank |
| --- | ---: | ---: | ---: | ---: |
| **A** retained baseline | 9.4847 / 9.4541 | 7.350 / 7.320 | 0.032 / 0.033 | 10178 / 7854 |
| **B** FineWeb KD | **4.2664 / 4.2478** | 1.945 / 1.931 | 0.285 / 0.285 | 511 / 502 |
| **C** matched control | 4.7713 / 4.7508 | 2.456 / 2.444 | 0.242 / 0.243 | 710 / 697 |

**B − A = −5.22 nats**, both seeds, against a lineage whose between-seed
`holdout_nll` spread has been 0.23–1.34. Top-1 on general text rises 9×.

**The mechanism is mostly not FineWeb.** C — in-domain rollout text, KD-only —
recovers **−4.71 nats**, 90% of B's gain. FineWeb's *content* adds the remaining
**−0.51 nats** (B − C, seed-consistent). What restores general language
modelling is extra KD signal on unseen text, largely regardless of which text.

### 34.3 Autonomous behaviour: nothing moved

| arm | usable | correct | correct \| usable | nat. term | ctx limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| A-Baseline | 0.7300 | 0.1867 | 0.2511 | 0.7600 | 0.2400 |
| B-FineWeb | **0.7300** | 0.1900 | 0.2603 | 0.7567 | 0.2434 |
| C-Control | 0.7500 | 0.1500 | 0.2000 | 0.7667 | 0.2334 |

Paired on the shared 150-prompt mask, against the registered floors (usable
0.0800, correct 0.0600):

| comparison | usable Δ | verdict | correct Δ | verdict |
| --- | ---: | --- | ---: | --- |
| B vs A | **+0.0000** | tie | +0.0033 | tie |
| C vs A | +0.0200 | tie | −0.0367 | tie |
| B vs C | −0.0200 | tie | +0.0400 | tie |

**Every comparison is inside its floor.** B vs A on usable rollout is +0.0000 —
not "small", zero. GSM8K correctness is 0.0000 on five of six arms; the sixth is
B-sa at 0.0789 (3 of 38) against B-sb's 0.0000, which is one seed of one arm and
is not a signal.

The one comparison that even points somewhere is **B vs C on correctness:
+0.0400 pooled, seed-consistent (+0.06 / +0.02)** — and it is **inside the 0.0600
floor and therefore a tie**. It is recorded, not claimed. `sa`'s CI touches zero.

### 34.4 What this settles

**A −5.22 nat swing in general language modelling produced no measurable change
in autonomous behaviour.** That is the largest diagnostic movement in the
project's history, and it moved the promotion criterion by nothing at all.

This closes the hypothesis that the rollout recipe's destruction of general
language modelling *causes* the correctness ceiling. It does not. The two are
separable, and E7 separated them.

It also strengthens the standing promotion rule past the point of argument. E6b
showed two objectives improving validation CE identically while only one moved
behaviour; E7 shows a diagnostic improving by five nats while behaviour moves by
zero. **Teacher-forced CE, NLL, KL and top-1 are training-health diagnostics.
They may not select a checkpoint.**

**The behavioural anchor is unchanged: E1/P1 KD-heavy 2.96M.** No E7 arm
displaces it, and none was expected to — E7 trained at 1.60M by design.

### 34.5 Exact reproduction

Commit **`97cd963d692ebba628dc8a2d0ef262604a14fc34`**, preregistration sha256
`e1d11b4bea6e31fc…`, inclusion mask `d6e24e0b09da1bcc…` asserted on all four
arms after evaluation.

```bash
PYTHONPATH=src python scripts/pod/e7_launch.py --scr … \
    --session-commit 97cd963d692ebba628dc8a2d0ef262604a14fc34 \
    --bundle aad_e7_97cd963d.bundle --authorized-usd 12.82
PYTHONPATH=src python scripts/evaluation/analyze_e7.py --bootstrap 10000
```

| identity | value |
| --- | --- |
| arms | `e7_{fineweb,control}_r1600k_{sa,sb}`, all from the Stage 1 PCA init |
| objective | ce 0.25 / kd 1.0, τ 1.0, scope all — the E1/P1 KD-heavy lineage |
| rung | 1,600,353 unique CE tokens · 4,801,059 cumulative · 1,761 steps |
| extra stream | 1761 × 1024 dense, λ 0.25, 1 block every step, 1.0 exposures |
| extra KD | 1,801,503 positions per arm, B and C identical |
| streams | fineweb `b70beffac337ee37…`, control `4e54f8e18baf01dc…`, val `e4002bbbbadf1a91…` |
| evaluation | 150 prompts, greedy, ctx 8192, vLLM, mask `d6e24e0b…` |
| artifacts | 38 files, 37 `final_required` + 1 `mutable_snapshot`, all hash-verified |

### 34.6 The infrastructure worked

First full session on the post-E6b stack. The launcher never blocked; the
watchdog polled throughout and was never needed; all four `train_log.jsonl`
streams were relayed continuously **and** passed the `final_required` gate at
teardown (quiescent, marker-backed, six completion markers matched). Setup ran
58 min against 45 budgeted and the contingency absorbed it. Nothing was lost.

---

## 35. Experiment 8 — contribution-guided depth initialization: design and preregistration

**Status: prepared, preregistered, NOT authorized.** No GPU has been used. The
prospective record is [`e8_preregistration.md`](e8_preregistration.md) and is not
to be edited to match any outcome.

**Question.** Does position-based depth compression discard teacher blocks that
are disproportionately important to the teacher's predictive function, and does a
contribution-guided layer map produce a better student under the same downstream
recipe?

### 35.1 What the current map actually does

`depth_span_map(36, 28)` keeps `[0,1,2,3,4,6,8,10,12,14,16,18,20…35]` and drops
`{5,7,9,11,13,15,17,19}`. The band's *position* was chosen by a logged single-axis
ablation (early band → holdout 10.48, middle band → 3.88). Which blocks die inside
it was never measured: they are the odd ones because the merge steps by two.

### 35.2 The selector, frozen before any map was seen

Primary: forward `KL(teacher ‖ teacher-with-S-bypassed)` over all prediction
positions, aggregated as an unweighted mean over 5 domains of the unweighted mean
over each domain's sub-types. Iterative greedy removal, 8 rounds, **260** subset
evaluations, full per-round table saved, ties on the lower layer index, no
positional constraint. Bypass is literal module-list removal — the same operation
the depth map performs — verified against an independent identity-block path.

Diagnostics (`reasoning`, `final_answer`, `think_close`, `eos`, `tool_close`,
`assistant`, CE delta) are recorded per candidate and **preregistered as unable to
select**. The search refuses to run unless `KL(intact‖intact) ≤ 1e-6`, and it
scores the positional map by the same objective for comparison.

### 35.3 Calibration set — `artifacts/stage1/e8_calibration_v1`

67 items, **59,763** prediction positions, 7 sub-types across 5 domains at
8,287–8,749 positions each. Content sha256 `d65c1f40e4837ea1…`. General text is
raw FineWeb-Edu [40000, 40040); the other six are teacher-native renders from the
corpus tail **beyond the 5.50M rung**.

**Two leakage collisions were caught and fixed**, both invisible to a session-id
filter, and both mattering because the frozen 150-prompt battery is sampled from
the 0.86M rung — a prefix of the rung the arms train on:

1. `glaive-000749#t3` — a different source item with byte-identical prompt text to
   a consumed session (tool-calling prompts are formulaic);
2. `openmath-000712#t1` — a session inside the pack's own 16-block validation
   slice, which lives in the same tail the calibration set is drawn from.

Six independent checks now pass clean, plus content-hash and index-range
disjointness against `holdout_v1`, `warmup_v1`, `eval_behavior_v0` and both E7
streams.

### 35.4 Two prerequisite findings, neither of which E8 went looking for

* **The Stage 0 activation cache (1.95 GB, `aaeb2e4c…`) was lost** — not on the
  dev box, never on the relay. Stage 1 cannot construct any initialization without
  it. **Regenerated at $0 and recovered bit-exactly**: 4,972 s of CPU, 949,859
  tokens (the historical count), hashing to `aaeb2e4c…`; rebuilding the positional
  init from it gives `86fbba78…`, byte-identical to the pinned control, with every
  projection diagnostic equal to the last digit. **E8 is a single-variable
  experiment.** See [decisions](decisions.md), 2026-08-10.
* **A silent 500× RoPE misread on the measurement path.** The Stage 1 checkpoint's
  config stores `rope_theta` in the transformers-5 `rope_parameters` dict; a 4.x
  reader falls back to 10,000 and reports holdout NLL 11.3953 instead of 11.7482
  without raising. No trained arm is affected — the pods already asserted
  `ROPE_OK` — but the measurement path did not. Now it does. Adding the assertion
  there also exposed that the guard itself could not survive a bf16 rotary buffer,
  which is what `build_student` produces: it recovered the base by inverting
  `inv_freq[1]`, amplifying the buffer's error 64×. It now inverts the last entry
  (exponent ≈1.016), which is exact on fp32, within 0.3% on bf16, and still four
  orders of magnitude from missing the 500× skew.

### 35.5 The mandatory initialization NLL

**An initialization checkpoint is not complete until its own NLL artifact
exists.** Enforced in `src/aadistill/init/nll_gate.py`: the record is bound to the
checkpoint's recomputed `model.safetensors` hash, every individual series carries
that hash too, self-declared inherited records are rejected, and a missing series
is a failure rather than a shorter report. Three series, never averaged:
`holdout_v1`, `fineweb_val_e7`, `teacher_native_val`. The gate never looks at
whether the NLL is *good* — a record with nll 99.0 passes, by test.

### 35.6 Arms and identity

Control: the retained `e1_r2960k_{sa,sb}_pca` (usable 0.8400, correct 0.2067), not
retrained. Treatment: `e8_contrib_r2960k_{sa,sb}`, whose realized config diff
against the control is exactly `{student_path, run_name, out_dir, _purpose}`.
Loader-verified budget: 1,944 blocks, **2,960,507** unique CE targets, exactly
**3.0** exposures, **8,881,521** cumulative.

### 35.7 Cost and the blocker

$10.38 expected, **$12.41** hard backstop across two pods (search; then init NLL +
2 × 2.96M + evaluation). Actual cumulative spend $160.158 against a $162.49
authorization leaves $2.33, so E8 needs **$10.08** more and a proposed cap of
**$172.57**. Not reduced to one seed: the behaviour seed-noise floor is 0.1290,
wider than any effect E8 could claim.

1,138 tests pass on CPU in both environments, 3 skipped.

---

## 36. Experiment 8 — the contribution-guided map preserves the teacher 3.1× better and initializes 2.8 nats worse

**Status: the search half is COMPLETE and the training half has NOT run**, blocked
on $0.20. Full record: [`e8_step0_report.md`](e8_step0_report.md). Prospective
record: [`e8_preregistration.md`](e8_preregistration.md), frozen before any GPU.

### 36.1 The map

260 subset evaluations, 17,688 forward passes, 1,300 s on one L40S, under the
objective frozen before any map existed. `self_consistency` exactly **0.0**
against a 1e-6 tolerance, so the ranking is not kernel noise.

```
contribution removes [2, 3, 15, 16, 20, 21, 26, 32]   order [2,16,3,32,20,26,15,21]
positional   removes [5, 7, 9, 11, 13, 15, 17, 19]
```

One shared layer. The two blocks the causal measure gives up **first** are layers 2
and 3, which the positional rule explicitly protects.

Primary objective **0.620586 vs 1.932531 — 3.11×** lower. Lower on all five
domains (2.65–3.33×) and on every diagnostic, none of which was allowed to select:
assistant 3.46×, reasoning 3.34×, final answer 3.94×, `</think>` 3.67×, EOS 2.49×,
and `</tool_call>` **0.0241 vs 10.3215, 428×**.

### 36.2 The initialization goes the other way

Both initializations measured on one device by one evaluator, each record
hash-bound to its checkpoint, the control **remeasured** rather than inherited.
Only the depth map differs between them: config hash, parameter count 596,049,920,
projection energy 0.9323228843289764 and final-norm range are identical to the last
digit.

| series | contribution | positional | Δ |
| --- | ---: | ---: | ---: |
| `holdout_v1` NLL | 13.2624 | 11.7565 | **+1.5059** |
| `fineweb_val_e7` NLL | 14.3913 | 11.5749 | **+2.8164** |
| `teacher_native_val` NLL | 11.8027 | 10.9053 | **+0.8974** |

Worse on every diagnostic, all three series, NLL and KL and top-1 and rank alike.

### 36.3 The finding, so far

**Teacher-ablation KL does not predict initialization quality after compression.**
The same map is 3.11× better at preserving the teacher's output distribution when
blocks are bypassed *in the teacher*, and 2.82 nats worse once the kept layers are
projected into a student that also loses 60% of its width, 68% of its FFN and half
its Q heads. The calibration objective never measured that interaction.

Hypothesis for later testing, not a result: the contribution map removes three
**adjacent** pairs (2–3, 15–16, 20–21), so a survivor's input can be two blocks of
transformation away from what it saw in the teacher, while the positional map
removes every *other* layer and is off by at most one. A full-width teacher with 34
blocks left absorbs that; a 0.6B student may not.

### 36.4 What is not decided

None of the registered catastrophic-abort conditions fired. Per the
preregistration a worse initialization NLL may not cancel E8, and two outcomes
remain live: **outcome 3** (init worse, behaviour improves — initialization NLL
again not a sufficient proxy) and **outcome 4** (both regress — reject the map).
Only the two-seed 2.96M training separates them.

E7 is directly relevant: a −5.22-nat NLL swing there moved autonomous behaviour by
exactly +0.0000. A +2.82-nat swing is therefore **not** evidence for outcome 4 on
this project's own record.

### 36.5 Cost, and six defects

$3.7253 total across two paid sessions. Pod A completed at $0.53 after four failed
attempts ($0.9583); pod B's first attempt was stopped at $1.9520 rather than
deliver a one-seed result, and its second reached both initialization measurements
at $0.2850 before failing a gate check on an artifact it does not stage.

Six defects found and fixed: a `meta`-device RoPE check (mine); three latent in the
launcher E8 derived from, none of which had ever fired (a `grep -c … || echo 0`
misparse that read a cold host as a setup failure, a non-starting pod aborting
instead of redrawing, and uv progress measured under `/opt/train` while uv writes to
`/root/.cache/uv`); torch thread oversubscription on a 128-vCPU host making a
70-second test suite run past 66 minutes; and a gate check requiring the
calibration manifest on a pod that does not search. None touched the selector, the
calibration set, the map, the recipe or the evaluation.

---

## 37. E8b — depth-map × compression interaction (preflight; not authorized)

The old E8 2.96M recovery was **cancelled by the maintainer** and replaced by this,
because of E8a's dissociation: the contribution map preserves the full-width teacher
3.11× better and its fully-compressed initialization is 2.82 nats worse. E8b asks
whether the map is good on its own and only breaks when composed with the existing
width/FFN/attention compression.

Full preflight: [`e8b_preregistration.md`](e8b_preregistration.md). **No GPU used.**

2×2 at the **1.60M** rung: depth-only (DP/DC, teacher width, 3,215,021,568 params)
× fully compressed (FP/FC, 596,049,920), positional × contribution, two seeds per
new cell, FP retained or retrained.

**DP and DC are built and verified at $0**, by verbatim `state_dict` copy of the
kept blocks: `d4db65eb8f7ae6d8…` and `eb9e95481988b296…`, one shared config hash
`4e5b7104…`, both **bitwise identical to `bypassed_blocks(teacher, removed)`** —
max logit diff exactly 0.000. So DP and DC *are* the ablated teachers E8a scored,
which means E8a's calibration KLs (1.932531 / 0.620586) are already step-0
statements about them and DC is expected to win the depth-only step-0 comparison.

**The L40S is ruled out.** Under the canonical semantics the depth-only arms peak at
63.41 GB, 72.92 GB with the 15% margin, against 23.09/26.56 for the target-size
cells. Selected from live pricing: A100 SXM 80GB @ $1.59/h, the cheapest ≥70 GB
class at High stock, single device.

**Design revised 2026-08-11 to pair-matched hardware** at the maintainer's
direction: DP/DC on A100 SXM 80 GB, FP retained and FC trained on L40S, so each
depth-map effect is measured within one hardware class. Hardware class is nested with
compression regime; the interaction inherits the nesting and a **conditional bridge**
($14.05, registered but not run) reruns FP/FC on the 80 GB card only if a material
reversal actually occurs.

**Cost:** four restartable sessions — S1 L40S step-0 for all four inits ($3.25 hard),
S2/S3 A100 seed-paired depth-only ($18.76 each), S4 L40S FC ($6.40). Expected $40.54,
hard **$47.18**, 28.5 h with the longest session 10.3 h against E7's proven 10.6 h.
Hardware was chosen on **cost per completed step**: A100 SXM $0.003472/step is
cheapest among High-stock ≥70 GB classes, though the spread to H100 SXM is only 3.5%
and the relative-efficiency figures are assumptions — which is why S2 carries a
registered 20-step gate on s/step, peak VRAM and live $/step that stops and re-prices
rather than widening the budget.

Recorded en route: `Qwen3ForCausalLM(cfg).to(bfloat16)` casts the rotary `inv_freq`
buffer to bf16 while `from_pretrained` recomputes it in fp32 — a 0.78 logit
difference from positional precision alone. The buffer is non-persistent so no saved
checkpoint is affected, including every Stage 1 artifact, but an init-time in-memory
forward is. It looks exactly like a construction bug and is not one.

## 38. E8b S1 — the depth map reverses sign between compression regimes (2026-08-11, $5.21)

**Date:** 2026-08-11 · **Agent:** Claude Opus 5 · **Commit:** `df954db1` (bundle
`aad_e8b_df954db1.bundle`; the measurements themselves ran at `ae6dda30`, dirty,
`uncommitted_state_sha256 9b8114fa…`) · **Hardware:** 1× NVIDIA L40S 48 GB @ $0.99/h

**Objective.** Measure the true step-0 initialization NLL of all four E8b cells
through **one** canonical `from_pretrained` reload path on one device, and run the
step-0 autonomous probe, before any arm trains.

**Result — the step-0 table.** Three held-out series, each measurement bound to the
sha256 of the checkpoint it read:

| cell | regime | depth map | holdout_v1 | fineweb NLL | fw top-1 | teacher-native NLL | tn top-1 | tn mean rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DP | depth-only 3.22B | positional | 4.0049 | 3.9064 | 0.3052 | 1.9360 | 0.5156 | 12.51 |
| DC | depth-only 3.22B | contribution | **3.0407** | **3.0175** | **0.4034** | **0.7596** | **0.7647** | **1.98** |
| FP | fully compressed 596M | positional | **11.7565** | **11.5749** | **0.0227** | **10.9053** | **0.0408** | **6739.9** |
| FC | fully compressed 596M | contribution | 13.2624 | 14.3913 | 0.0075 | 11.8027 | 0.0230 | 19865.6 |

`DC − DP = −0.96 / −0.89 / −1.18` nats; `FC − FP = +1.51 / +2.82 / +0.90` nats. The
contribution map is better at full width and worse fully compressed on **9 of 9
metrics across three independent series** — a unanimous sign reversal. There is no
single number for "is the contribution map better"; the answer depends on what else
is compressed with it.

FP re-measured to **11.756504**, digit-identical to the E8a session, so the reload
path and evaluator are stable across sessions and pods.

**Validity.** One device, one dtype, one environment (`torch 2.11.0+cu128`,
`transformers 5.13.1`), identical source hashes, four distinct checkpoint hashes, and
every series carries `measured_checkpoint_sha256` equal to the checkpoint it read.
`resolved_rope_base` = 5000000.2415 on all four, so the 500× RoPE skew is excluded.
Geometry is matched within each pair (`config_sha256` equal for DP/DC and for FP/FC),
so within a pair only weights differ.

**Interpretation, and what it is not.** This is evidence level 2 of 3. Level 1 (E8a)
found the contribution map preserves the frozen teacher 3.11× better in KL at full
width. Level 2 now confirms that advantage survives into a real reloaded checkpoint —
**at full width** — and inverts under width/FFN/attention compression. Neither level
can promote or cancel an arm: initialization NLL is diagnostic (decision 2026-08-05),
and level 3, matched 1.60M recovery, is the registered endpoint.

The reversal is consistent with the hypothesis that E8a's KL selects blocks whose
*residual-stream geometry* the teacher depends on, and that this geometry is what the
grouped-PCA width projection destroys — the map and the projection are not
independent. It does not establish that; S2–S4 test it.

**Deviations.**

1. **DC's step-0 probe was not run.** DP's took 200 min against a 20 min estimate
   ($3.31 of a $0.33 line item) and the per-probe budget gate refused the second:
   `MARKER:ABORTED_AT_GATE:budget:4.03+0.91`. DP floored on every behaviour axis —
   76/76 truncated at the 2048 cap, `terminated` 0.0, `think_closed` 0.0,
   `format_ok` 0.0, `empty_answer` 1.0 — but **that does not predict DC**, whose
   teacher-native top-1 is 0.76 against DP's 0.52. Recorded as deferred on budget
   grounds, not as uninformative. Re-running it would cost $0.5–3.3 depending on
   whether DC terminates naturally.
2. **`publish_step0` failed** with `RepositoryNotFoundError`. Cause: the launcher
   starts the driver detached with `env={"PYTHONPATH": …}`, so `HF_TOKEN` — exported
   by the setup shell — was never inherited, and `upload_file(token=None)` on a
   private repo cannot see the repo. The same line was in `stage_fetch_step0`, so S2
   and S3 would have failed identically *after* paying A100 setup. Fixed: relay calls
   resolve through `hf_token()`, which reads the staged `/workspace/hf/token` and
   treats the environment as an override (`tests/pod/test_driver_relay_credentials.py`,
   9 tests). The four records were fetched off the pod and published from the dev box,
   so the training sessions still gate on them.
3. **Three unusable host draws** cost $1.07 before the working pod: two cold hosts at
   the 25-min uv ceiling (raised to 3600 s) and one genuine stall at 26.6 min.


**Addendum — what DP's step-0 generations actually contain, and one blind metric.**
The 76 capped generations are not noise. DP opens in the teacher's own reasoning
register and stays on topic — "Okay, the user is asking for a love letter. I need to
respond with a message that…" — then collapses into an exact repetition loop and never
closes `<think>`. Measured on the reasoning text: max-3-gram share averages **0.1420**
and reaches **0.8395**, with **70 of 76** samples above 5%. So a depth-only
initialization with 8 teacher blocks deleted and no training at all retains fluent,
topically-appropriate language modelling and loses only closure and progress. That is
consistent with the reversal above: at full width the damage is behavioural, not
linguistic.

The probe's aggregate reports `rep_3gram 0.0`, and that figure must not be read as
"no degeneration". `behavior.py:350` computes it as `repetition_rate(answer)`, and
`answer` is empty for all 76 samples because `</think>` never closes — so the one
field that could have shown the degeneration is structurally blind to it whenever the
think block does not terminate. **E8b's endpoint is unaffected:**
`run_three_mode_diagnostic.py:231` calls `degeneration.check(gen)` on the full
generated token stream, and `usable_rollout.no_severe_repetition` reads that, so the
frozen battery does see reasoning-block degeneration. The blind field belongs to the
`eval_behavior_v0` diagnostic only. Left in place rather than changed, since
`rep_3gram` feeds only the retired `behavior_score_v0` and E8b does not use it; noted
here so no future reader cites it as evidence of clean output.

**Cost.** $1.07 (three draws) + $4.14 (251 min) = **$5.21** against a $3.25 plan. The
overrun is entirely the DP probe. E8b remaining: $41.97 of $47.18.

**Artifacts.** `logs/e8b_step0_records/{DP,DC,FP,FC}_init_nll.json` and
`DP_step0_probe.json` (tracked); relay `e8b_step0_20260811/` (four records plus the
probe's 1.3 MB generations); `logs/e8b_analysis.json`;
`logs/e8b_s1_session_evidence.json`. Analysis: `scripts/training/analyze_e8b.py`.

**Verdict.** Level 2 complete and internally consistent. The sign reversal is the
finding that reframes E8b: the question is no longer "is the contribution map better"
but "does it only fail once composed with width compression". That is exactly what
the 2×2 was built to answer.

**Next.** S2 (DP-sa + DC-sa, A100, mandatory 20-step gate), S3 (seed sb), S4 (FC both
seeds, L40S) against retained FP.

## 39. E8b S2 attempt 1 — three draws lost to our own gates (2026-08-11, $2.27)

**Date:** 2026-08-11 · **Commit:** `df954db1` · **Hardware:** NVIDIA A100 SXM 80 GB
@ $1.59/h, three consecutive host draws · **Outcome:** ABORTED before any training

No arm trained. All three draws were abandoned with `HOST_COLD`, and **no host was
cold.** Two distinct defects in our own setup gates produce the same marker, which is
why the first draw looked like ordinary bad luck.

**Draw 1 — the uv tripwire kills on a single quiet window.** Killed at 26.4 min. The
tripwire samples 20 s of disk growth across `/opt/train` + `/root/.cache/uv` and kills
if growth is under 20 MB. uv writes nothing while it resolves or builds a wheel, so a
working host is indistinguishable from a hung one when one sample decides. Worse,
`--uv-max-s 3600` could not have protected it: `UV_MAX_S` never extended the deadline,
it only stops the growth check from applying at all. That misreading is what made the
flag look like a fix after S1.

**Draws 2 and 3 — the test gate's 900 s box, and why the suite was slow.** Draw 2
reached `VLLM_READY` in 4 minutes on a warm host, so uv was not involved; it died 24
minutes later. Draw 3 reproduced it, and was caught in the act:

```
3000 1851 206s Rl 1338%CPU  search_depth_map.py --student-layers 4 --dtype float32 --device cpu
1851 1850 742s Sl    2.1%   pytest tests/ -q      (wchan sigsuspend, 227 threads alive)
```

The subprocess that `tests/init/test_depth_search_driver.py` spawns was burning 900+ s
at 1338% CPU on work that takes **7.4 s** on the dev box, while pytest waited. Cause:
a container reports the **host's** cpu count — `nproc` said 128 — while the cgroup
grants a fraction of it. The child sized its thread pools from 128 and thrashed.
`OMP/MKL/OPENBLAS_NUM_THREADS=8` could not bound it because the child re-derives its
own pool. The dev box has 16 real CPUs and no quota, which is exactly why the suite
takes 88 s here and could not reproduce the failure locally.

**Fixes** (`c9517bea`, 20 tests against the deployed script):

* the tripwire needs **3 consecutive** quiet windows; progress resets the counter; the
  marker carries the stall count so a hang is distinguishable from a slow mirror;
* the cpu budget is read from the **cgroup quota** and enforced with `taskset`, whose
  affinity children inherit across fork/exec — no library can ignore it;
* thread caps follow that budget instead of a constant that could exceed it on a
  4-cpu container;
* both the visible and granted cpu counts are logged, since the discrepancy is the bug;
* box raised to 2,700 s, and the suite's elapsed time recorded on success so it is
  calibrated from measurement rather than guessed a third time.

Verified locally against synthetic cgroups (4, 8, capped-32, unlimited, and a bare
host with no cgroup files), and that a quiet-but-finishing host survives while a hung
one still dies.

**Cost.** $2.27. Nothing scientific was produced, and nothing scientific was
corrupted: the abort happened before the throughput gate, so no arm trained from an
unverified state.

**Lesson for the next session.** A gate that fails closed is still a gate that can be
wrong, and ours reported a host defect for a defect of our own. Both misfires were
invisible on the dev box because the dev box has neither 128 visible CPUs nor a
cgroup quota. Setup gates calibrated on one machine class need a synthetic test of the
*other* class, which is what the 20 new tests now provide.

### 39.1 Attempts 2 and 3 — the gate fix worked; the suite then failed on its own terms

**Attempt 2 ($0.42, commit `c9517bea`).** The cgroup fix is confirmed by the pod's own
log: `128 vCPUs visible, cgroup budget 13`, and the suite finished in **94 s** instead
of exceeding 900. Three tests then failed and aborted setup, all three ours:

* two new `cpu_budget` tests asserted the no-quota fallback equals the affinity mask,
  but the suite runs with `OMP_NUM_THREADS` exported and **coreutils `nproc` honours
  `OMP_NUM_THREADS`** — it returned 8 on a 13-cpu set. The production path read the
  quota directly and was unaffected, but the fallback was genuinely fragile: it fed our
  own thread cap back in as the cpu budget. It now reads the affinity mask, and a test
  forbids bare `$(nproc)` inside that function;
* an E8 test asserted one particular absence pattern. Its negative case needs the
  treatment init absent **and** the baseline present; an s2 session stages neither
  compressed init, and with the baseline gone `validate_e8_arms.py` exits 1 on a
  missing checkpoint directory rather than 6 with a report. It now skips on both
  unreachable states.

Reproduced locally first — `taskset -c 0-12` gives the 16-core dev box exactly the
pod's 13 cpus — then verified against a simulated s2 pod (1,247 passed, 19 skipped,
71 s). `simulate_pod_env.sh` no longer claims `artifacts/stage1` is always staged;
that assumption is what let a session-specific staging difference reach a paid pod.

**Attempt 3 ($0.41, commit `ee20fa91`).** Aborted on
`test_timed_returns_result_and_a_duration` with `AcceleratorError` after 1,219 tests
passed. `timed()` calls `torch.cuda.synchronize()` — correctly, since without it a
CUDA-async engine is timed as faster than it is — and `synchronize` re-raises any
*earlier* async CUDA fault in the process as a sticky error. So an assertion about
returning `(value, seconds)` for a pure-CPU lambda could fail for something another
test did 1,200 cases earlier, and passed here only because the dev box has no CUDA. It
passed on attempt 2's host and failed on attempt 3's: same code, different accelerator
state.

The synchronize stays and now has its own test asserting drain-run-drain ordering
through a fake `torch.cuda`, so the behaviour is better covered while neither test
depends on a healthy GPU. Auditing the suite found `test_engines.py` was the **only**
real CUDA interaction — the one other match is a log line inside test data — so the
class is closed, not the instance patched, and an AST test keeps it closed.

**Cumulative S2 setup cost: $3.10 across four attempts** ($2.27 + $0.42 + $0.41 +
attempt 4). No arm has trained, and nothing scientific is affected: every abort
happened before the throughput gate.

## 40. E8b S2 — the registered gate fired: 3.22B does not fit an 80 GB A100 (2026-08-11, $0.55)

**Date:** 2026-08-11 · **Commit:** `ccba0fbf` · **Hardware:** NVIDIA A100 SXM 80 GB
(81,920 MiB, driver 580.159.04) @ $1.59/h · **Outcome:** STOPPED at the registered
20-step gate. No arm trained. **This is the gate working, not failing.**

Setup passed for the first time in five attempts — `TESTS_OK:106s`, `MASK_OK`,
`ARMS_VALIDATED` — and `STEP0_FETCHED` confirms the relay-credential fix: S1's
hash-bound DP and DC records were fetched and re-gated against the checkpoints this
pod had rebuilt from the teacher.

**What the gate measured, on the real trainer and the real DP-sa arm:**

```
eval step 0: val_ce 1.798891  val_ppl 6.0429  val_kd 1.602439
step 1/20  loss 2.4162  ce 2.0587  kd 1.9015  5.39s
step 2/20  loss 3.1120  ce 2.7506  kd 2.4244  4.87s
step 3/20  loss 2.7408  ce 2.1802  kd 2.1957  4.80s
torch.OutOfMemoryError: Tried to allocate 298.00 MiB.
  79.25 GiB capacity, 140.94 MiB free.
  72.44 GiB allocated by PyTorch, 6.16 GiB reserved but unallocated.
  at kd_forward_kl: torch.log_softmax(tp[i:i+chunk].float() / temperature)
```

**Speed and cost passed; memory failed.** Steady step time is ~4.83 s against the
registered ceiling of 7.86 s, so $/step is **$0.00212** against a $0.003472 limit. The
derived figure was conservative rather than wrong: it modelled
`(8·N_student + 2·N_teacher)` FLOPs against E6b's measured 4.15 s/step with a 1.6×
A100 ratio and a 1.15 safety factor. Peak VRAM was never reported because the run died
before the gate could summarise it — the third registered quantity is therefore
**unmeasured**, and the memory sizing (63.41 GB expected, 72.92 GB at +15%) is
falsified: actual in-use was 79.10 GB of 79.25 GB.

**How close it was.** The failing allocation is 298 MiB while **6.16 GiB sat reserved
and unusable**. At vocab 151,936, `chunk=512` makes each fp32 buffer
512 × 151,936 × 4 B = 311 MB, which is exactly the allocation that failed.

**Two memory levers, neither adopted** (a registered gate failed, so this is a
re-pricing decision, not a tuning one — `scripts/training/reprice_e8b_after_gate.py`,
`logs/e8b_reprice_after_gate.json`):

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — allocator segment mapping
   only, changes no numerics, and the torch error recommends it by name. Targets the
   6.16 GiB of fragmentation directly.
2. KD `chunk` 512 → 128 — ~0.9 GB less transient fp32 peak. **Mathematically
   identical but NOT bitwise identical:** the loop accumulates one float32 scalar per
   chunk, so the chunk count changes the summation order. Measured at ~7e-8 relative
   (`tests/training/test_kd_chunk_invariance.py`). A first check appeared to show
   bit-identity and was wrong — with 54 masked positions, chunks 512/256/128/64 were
   all a single chunk and trivially equal. It must therefore be applied to **both arms
   of a pair or neither**, and is **not** proposed for S4, whose retained FP control
   trained at chunk 512 and which needs only 23–27 GB on an L40S.

**Re-pricing, from the measurement.** Because 4.83 s/step is 39% below the derived
7.86 s, each depth-only session drops from **$18.76 hard to $12.91**. Against $38.32
remaining of the $47.18 backstop, re-priced S2 + S3 + S4 comes to **$32.21 hard,
margin $6.11** — so the experiment fits the existing authorization with no further
funds. **The blocker is memory, not money.**

**What is not known.** Whether the two levers are sufficient. They target ~7 GB
against a ~0.3 GB shortfall, so the headroom looks ample, but that is an inference
from one OOM trace, not a measurement. The re-run's own gate would settle it and would
also finally report peak VRAM.

**Cost.** $0.55 for the session; $3.65 for all four S2 attempts. Nothing scientific is
affected: no arm trained, and S1's step-0 table is untouched.

## 41. E8b S2 — DP-sa trained; DC-sa OOM'd; 80 GB is marginal, not sufficient (2026-08-11, $7.21)

**Commit:** `6b9d441` · **A100 SXM 80 GB** · backend: SDPA flash + `kd_chunk 128` +
`expandable_segments:True`, behind the 200-step steady-state gate.

**The 200-step gate passed honestly and was still too short.** It measured 77.31 GiB
with 0.000 GiB drift over its final 50 steps and 1.94 GiB of free margin. The full run
then showed the peak climbing again *after* the gate ended:

| step | peak allocated (GiB) |
| --- | --- |
| 10 | 76.50 |
| 70 | 76.71 |
| 120 | 77.23 |
| 130 | **77.31** — the gate's figure, flat through step 200 |
| **310** | **77.45** — true steady state, held to step 1760 |

**DP-sa completed all 1,761 steps** at that 77.45 GiB peak — the first finished DP/DC
arm. **DC-sa OOM'd at ~step 900** on a **74 MiB** request, with 77.37 GiB allocated and
1.31 GiB reserved-unallocated. Both arms reached the identical 77.45 GiB maximum; one
survived it and one did not.

**What this establishes.** The configuration is *marginal* on an 80 GB card, not
merely un-gated. At 77.45 GiB of 79.25 GiB capacity the nominal margin is 1.80 GiB, of
which ~1.31 GiB is allocator slack — leaving well under a gigabyte usable, and a 74 MiB
request failed. **DP-sa completing was luck, not headroom.** No longer gate can fix
this: the peak is data-dependent, settling only around step 310, and the remaining
margin is smaller than the run-to-run variation between two arms of the same shape.

**Both memory levers are now exhausted.** `expandable_segments` cut fragmentation
6.16 → 1.04 GiB; `kd_chunk 128` bought ~0.7 GB. Together they moved the failure from
step 110 to step 900 without removing it.

**Not a scientific result.** DC-sa has no checkpoint, so `DC − DP` does not exist. No
conclusion about the depth map follows. S1's step-0 table is untouched.

**DP-sa's checkpoint was not staged.** It is 12.9 GB of fp32 weights against an
irreclaimable HF LFS quota, and it is only reusable if the eventual fix preserves both
the backend *and* the hardware — which neither of the remaining options does. Its
`train_log.jsonl` and `run_manifest.json` are kept
(`logs/e8b_s2_dp_sa/`), along with DC-sa's partial trajectory.

**Cost.** $7.21. E8b total **$16.82** of $47.18; **$30.36** remains.

## 42. E8b full-stream shape audit — the OOM is NOT explained by data-dependent shapes (2026-08-12, $0)

Zero GPU cost. `scripts/training/audit_stream_shapes.py` reconstructs the exact
deterministic stream — `stream_block_indices(n_blocks, seed, step*bps, bps)` is a pure
function of seed and step — using the real pack (`blocks.npz`
sha256 `6f324cb0…`, the hash the pod verifies), the real masks and the real seed
derivation. 3,522 microbatches per arm.

**First, a metric correction that reframes everything.** The logged `gpu_mem_gb` is
`torch.cuda.max_memory_allocated()`, a **running maximum that is never reset**. Its rise
from 76.50 to 77.45 is non-decreasing *by construction*. My earlier description of it as
the peak "climbing" implied creep; a running max reaching a plateau is equally
consistent with simply having encountered the worst block.

**Shapes do vary — substantially.**

| quantity | min | mean | p90 | p99 | max | distinct |
| --- | --- | --- | --- | --- | --- | --- |
| `ce_targets` | 71 | 1363.2 | 5166 | 7374 | **7756** | 544 |
| `kd_positions` | 219 | 2265.9 | 8191 | 8191 | **8191** | 561 |
| executed extent | 8192 | 8192 | 8192 | 8192 | 8192 | **1** |

So a fixed `seq_len` really does not imply a fixed memory workload: CE selection varies
by two orders of magnitude. **But that is not what caused the OOM.**

**The worst block is step 133, inside the gate's window.**

| window | max ce | max kd | max estimated transient |
| --- | --- | --- | --- |
| first 20 | 6882 | 8191 | 16,463.7 MB |
| **first 200** | **7756** | **8191** | **17,260.5 MB** |
| first 310 | 7756 | 8191 | 17,260.5 MB |
| around the DC OOM (850–950) | 7732 | 8191 | 17,238.6 MB |
| **full stream** | **7756** | **8191** | **17,260.5 MB** |

`first_200` max **equals** `full_stream` max. The 200-step gate sampled the
worst-case shapes in the entire run, and the region where DC-sa died is *less*
demanding than what the gate had already measured at 77.31 GiB.

**DP-sa and DC-sa are shape-identical** across all 3,522 microbatches — step index,
block indices, `ce_targets`, `kd_positions`, executed extent and non-padding counts all
match exactly. No bug there.

**Two further candidates eliminated at zero cost.** The gate config disables evaluation
(`eval_every: 0`, `eval_blocks: 4`) while the real run evaluates every 220 steps over 16
blocks and checkpoints at 880 — so the gate's workload was genuinely unrepresentative.
But the validation blocks do **not** exceed the training worst case (ce 7,684 vs 7,756;
kd equal at 8,191) and `_eval_blocks` runs under `@torch.no_grad()`, so evaluation
allocates strictly less than a training step. Evaluation does not explain the residual.

**Classification: lifecycle-suspect, not shape-explained.** The rise to 77.31 GiB is
fully accounted for by the worst training block at step 133. The residual **+0.14 GiB**
appearing at step 310, and DC-sa's failure at ~step 900 on shapes smaller than the gate
had already survived, are **not** explained by the workload. That is Branch B.

**The supported statement, and the one to use:** *under the present implementation,
DP/DC operate within a very small A100 memory margin and have produced late-step OOMs
whose relationship to data-dependent tensor shape versus memory lifecycle is now
partially resolved — shapes are excluded as the cause.* It is **not** established that
DP/DC require more than 80 GB, and §41's framing of "80 GB is marginal" should be read
with that correction.

**Recommendation: do not buy a larger GPU yet.** The next step is the Branch B
same-block replay — one exact block repeated past the growth horizon, recording
allocated memory at every phase boundary and reserved memory alongside, with CUDA
allocation snapshots if the boundary value grows. A ≥94 GB card would hide exactly the
defect this would find.
