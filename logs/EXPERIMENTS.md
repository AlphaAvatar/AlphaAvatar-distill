# Experiment record — AlphaAvatar-distill

The single consolidated record of everything run. Replaces 25 per-run logs and 11
proposal files, which are preserved in git history at commit `866dac2`.

**Teacher** `Qwen/Qwen3-4B-Thinking-2507@768f209d` (2560 hidden, 36L, FFN 9728,
32Q/8KV) → **student** 0.6B-class (1024 hidden, 28L, FFN 3072, 16Q/8KV, tied
embeddings). BF16 training, INT8 deployment target.

**Total spend to date: $109.51** of the **$126.02** cap.

| period | $ | detail |
|---|---:|---|
| through corpus v2 (2026-08-01) | 34.52 | §6 — training/eval $7.93, teacher generation $26.59 |
| Experiment 1, data-scaling matrix | 61.50 | §11 — 24 arms $47.6, control + first eval $8.1, sweep $5.8 |
| Experiment 2 phase 1, data cleaning | 12.97 | §12.15 — experiment $11.23, avoidable pod waste $1.74 |
| Diagnostic session (benchmark + reference + recall) | 0.52 | §14 — 94 min on an RTX A6000 at $0.33/h |

**$16.51 of the $30 Experiment 2 allocation is unspent.** §6 below is the
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
128, vocab 151,936, tied embeddings, **identical 595,984,384 parameters** — but
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
identical **595,984,384** parameters, but `rope_theta` 1e6 vs our 5e6 and
`max_position_embeddings` 40,960 vs 262,144.

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
battery is not too hard. The gap is the recipe.

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
