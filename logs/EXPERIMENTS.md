# Experiment record — AlphaAvatar-distill

The single consolidated record of everything run. Replaces 25 per-run logs and 11
proposal files, which are preserved in git history at commit `866dac2`.

**Teacher** `Qwen/Qwen3-4B-Thinking-2507@768f209d` (2560 hidden, 36L, FFN 9728,
32Q/8KV) → **student** 0.6B-class (1024 hidden, 28L, FFN 3072, 16Q/8KV, tied
embeddings). BF16 training, INT8 deployment target.

**Total spend to date: $34.52.** Training and evaluation $7.93 (§6, including
what was wasted); teacher generation $26.59 (§9 gate $1.03 + §10 corpus $25.56).

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
| **project total to date** | **$34.52** | generation subtotal $26.59, all against the $50 generation cap |

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

**The control that would separate them** — train the 0.25M rung for 4,412 steps
(≈40.9 epochs), matching the top rung's compute exactly. If it approaches the
top rung's 1.0032, the curve is largely a compute effect; if it plateaus well
above, the curve is a data effect. One arm, ~3.9 h on one L40S, ~**$3.9**. Not
run; recommended before any scaling-law claim is published.

**Verdict: the scaling relationship is measured and internally clean, but two
things stop it short of a law.** The saturation point is outside the corpus
(neither init has flattened at 5.50M, and the uniform cut caps this pack at
~6.08M), and data quantity is confounded with training compute by the
fixed-passes design. Recovery-data sizing cannot be closed until the corpus
grows past ~6M uniform tokens or the mixture is relaxed (Experiment 2), and the
data-vs-compute control above is run.
