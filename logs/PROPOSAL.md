# Active proposal — Experiment 2: three sequential 0.86M diagnostics

**Status 2026-08-03.** Experiment 1 is complete ([`EXPERIMENTS.md`](EXPERIMENTS.md)
§11). Experiment 2 is **three sequential single-variable diagnostics, all at
Experiment 1's 0.86M rung**, each reusing the previous phase's winner as its
control:

1. **Data cleaning** — did bad teacher targets cause the deterioration?
2. **Loss (KL-only first)** — does removing CE fix it?
3. **Learning rate** — does a lower LR fix it, or only postpone it?

Not a Cartesian sweep. Each phase reports and is approved before the next is
prepared.

**Phase 1 is prepared and fits the budget. Nothing is launched; zero GPU time
has been spent.** Awaiting explicit approval to train D1 at both seeds.

**Three maintainer decisions carried into this revision (2026-08-03):**
the diagnostic rung moves **2.96M → 0.86M**; replacement selection moves
**shortest-survivor → median-length survivor**; Experiment 2 gets an incremental
budget of **$30.00** (new cumulative cap **$126.02**).

---

## 1. Why 0.86M is the right rung

Reconstructed per seed from the arms' own logs — PCA held-out NLL on
`data/warmup/holdout_v1.jsonl` (40 samples / 21,080 tokens):

| rung (unique supervised) | steps | sa | sb | mean | seed \|Δ\| |
|---:|---:|---:|---:|---:|---:|
| 252,985 | 324 | 6.7169 | 7.8699 | 7.2934 | 1.153 |
| 460,088 | 570 | 6.1616 | 6.2948 | **6.2282 ← minimum** | 0.133 |
| **864,750** | **1,023** | **8.8758** | **9.3649** | **9.1204** | **0.489** |
| 1,600,353 | 1,761 | 9.7145 | 9.4845 | 9.5995 | 0.230 |
| 2,960,507 | 2,916 | 10.4031 | 9.7864 | 10.0948 | 0.617 |
| 5,501,372 | 4,412 | 10.7875 | 9.4548 | 10.1212 | 1.333 |

The minimum is at 0.46M and the largest single jump is **0.46M → 0.86M
(+2.89 nats)**. 0.86M is the first rung clearly inside the deterioration region.
2.96M → 5.50M is +0.026, an order of magnitude below the between-seed spread —
the plateau, where the deterioration has already happened and cannot be watched
beginning.

**What the rung change buys beyond that.** At 0.86M the student is measurably
less stable than at 2.96M — degeneration 0.237 vs 0.079, natural termination
0.763 vs 0.921 — so the generation-stability axes carry real signal here rather
than sitting near ceiling. The trade: **GSM8K strict EM is 0.000 on both seeds
at 0.86M** (0.020 / 0.010 at 2.96M), so the reasoning axis is at floor and can
only detect improvement, never degradation. The gate in §7 is written around
that asymmetry rather than pretending it away.

**Two facts that still shape what phase 1 can find.** Held-out NLL tracks
optimizer steps, not unique data: the step-matched control (0.25M of data, 4,412
steps) lands at 10.7082 against the 5.50M/4,412-step arm's 10.7875, a 0.079 gap
across a 21.7× data ratio. And teacher-native val CE falls monotonically the
whole way at 74× the between-seed noise. The two metrics genuinely disagree, and
the Experiment 1 prior is that **phase 3 is where the mechanism most likely
lives.** The maintainer's ordering is kept.

## 2. The exact 0.86M D0 baseline

Parsed from Experiment 1's saved manifests, configs and logs — **not** scaled
from another rung. Both arms' `run_manifest.json`, `train_log.jsonl` and
`eval_holdout_v1.json` were pulled from the relay
(`e1_scaling_20260801/e1_r0860k_s{a,b}_pca/`); the dev-box copies had been lost
to the Experiment 1 `scp` basename collapse.

| quantity | value |
|---|---|
| unique supervised tokens | **864,750** (not 860,000) |
| blocks / sessions | **682** / **1,502** |
| packed tokens | **5,586,944** (682 × 8,192) |
| optimizer steps | **1,023** = `ceil(682 × 3 / 2)`, 3.0 effective epochs |
| tokens processed | 16,760,832 |
| real / padding tokens | 1,472,149 / 4,114,795 |
| terminal truncations | 109, discarding 202,365 supervised tokens |
| per-type token shares | code .1666 · gsm8k .1664 · multihop .1670 · openmath .1673 · rag .1669 · tool .1658 |
| seeds | `sa` **20260726**, `sb` **20260801** (block order only) |
| init | `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`, `model.safetensors` sha256 `86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54` — **re-verified today** |
| pack | `blocks.npz` `6f324cb0…`, `ladder.json` `d4941722…`, `audit.jsonl` `15f16b7b…` |
| loss | CE 0.25 + forward-KL 1.0, τ=1 (τ² present), `kd_scope: "all"` |
| LR / schedule | **η = 5e-5**, warmup **51**, cosine to `min_lr_frac` 0.1 (5e-6) at 1,023 |
| batch / packing | `blocks_per_step` 2, `micro_blocks` 1, `packing: "ladder"`, `block_len` 8192 |
| trainable | 440,467,456 / 596,049,920 (tied embedding frozen) |
| validation | 16 blocks, indices **[2941…2953, 2955, 2957, 2969]**, 81,195 supervised tokens |
| config sha256 | `sa` `08264ef1…`, `sb` `9048173d…` — **both recomputed from the committed configs and matching the run manifests** |
| code state | commit `69c3fe1`, dirty, uncommitted `2e04f683…`; `train.py` / `train_stage3.py` / `ladder.py` byte-identical at HEAD |

**Full val-CE trajectories** (authoritative, from the relay `train_log.jsonl`):

* `sa`: 10.919939 → 2.735890 → 2.166101 → 1.878347 → 1.700951 → 1.628793 →
  1.567471 → 1.528217 → 1.508228 → **1.510093**
* `sb`: 10.919939 → 2.735201 → 2.065509 → 1.825222 → 1.723215 → 1.652858 →
  1.565172 → 1.525525 → 1.502157 → **1.503762**

at steps 0, 127, 254, 381, 508, 635, 762, 889, 1016, 1023.

### What D0 can and cannot support

| comparison | supported? | why |
|---|---|---|
| fixed-step endpoint, val CE | **yes** | full trajectory + final checkpoint |
| fixed-step endpoint, held-out NLL | **yes** | `eval_holdout_v1.json`, and the sweep reproduces it to 0.0000 |
| fixed-step endpoint, behaviour / GSM8K / protocol / termination / degeneration / lengths | **yes** | stored generations, re-scorable offline |
| **best-validation-CE checkpoint** | **no** | `keep_last: 1` — only step 1,023 exists. Val CE is monotone here, so the endpoint *is* the best, but the checkpoint for any other step is gone |
| **best-held-out-NLL checkpoint** | **no** | no intermediate checkpoints, and no within-run NLL trajectory was ever recorded |
| **onset of deterioration within the run** | **no** | same reason |
| new capability sets (knowledge QA, verifier math, multihop, RAG, refusal) | **yes, but needs GPU** | the checkpoints exist on the relay; the stored generations only cover Experiment 1's behaviour and GSM8K prompts |

Every Experiment 2 arm fixes this going forward: eval **and** checkpoint at all
9 points, with retention handled by §8 rather than by keeping everything.

### Corrected D0 metrics at 0.86M

GSM8K re-scored offline under the strict rule (`\boxed{}` first, else an
explicit `Final Answer:`/`Answer:`, tool-call payloads stripped,
protocol-invalid or degenerate output incorrect regardless of content):

| metric | sa | sb | mean | seed \|Δ\| |
|---|---:|---:|---:|---:|
| teacher-native val CE | 1.5101 | 1.5038 | 1.5069 | **0.0063** |
| held-out NLL | 8.8758 | 9.3649 | 9.1204 | **0.4891** |
| behaviour composite | 0.3673 | 0.3716 | 0.3695 | 0.0043 |
| natural termination | 0.750 | 0.776 | 0.763 | 0.0263 |
| degeneration | 0.250 | 0.224 | 0.237 | 0.0263 |
| generated tokens p50 | 320 | 306 | 313 | 14 |
| **GSM8K strict EM** | **0.000** | **0.000** | **0.000** | 0.000 |
| GSM8K protocol-valid | 0.300 | 0.540 | 0.420 | **0.240** |
| GSM8K final-answer present | 0.190 | 0.470 | 0.330 | 0.280 |
| tool_call axis | 0.833 | 0.917 | 0.875 | 0.084 |
| grounding axis | 0.0625 | 0.000 | 0.031 | 0.0625 |

Dominant GSM8K failure at this rung is **non-termination** (62 and 42 of 100
samples), not wrong arithmetic — which is what makes cleaning a plausible lever
here in a way it was not at 2.96M.

## 3. Median-length survivor selection (`clean-v2`)

Every correctness, evidence, protocol, completion and degeneration gate is
**unchanged**. Only what happens after them changed.

1. If D0's own candidate passes every gate, **retain it** — length is never
   consulted.
2. Only a failing original may be replaced.
3. Length is the candidate's **assistant supervised-token count after exact chat
   serialization** (`render_session`), not characters and not raw pre-template
   tokens.
4. Take the **median** of the surviving candidates' lengths.
5. Choose the survivor **closest to that median**.
6. Ties break on the **original candidate index**.

Length is never a filter: it is consulted only among candidates that already
passed every semantic, evidence, protocol, completeness and degeneration check.
Recorded in `RULES_VERSION = "clean-v2"`, in `cleaning_audit.json`'s
`selection_rule` block, and in every emitted manifest.

### Measured against shortest-survivor, on one corpus

Both corpora come from the same run configuration differing only in
`--selection`, so every difference is the rule alone.
`artifacts/stage3/e2_selection_rule_audit.json`.

* 10,778 prompts kept; **242 needed a replacement**; the rules **disagree on 73
  (30.2%)** — gsm8k 29, rag_evidence 24, multihop_qa 12, openmath 8.
* On those 73, median selection keeps materially more of the derivation:

| type | n | median p50 | shortest p50 | median mean | shortest mean |
|---|---:|---:|---:|---:|---:|
| gsm8k | 29 | 1,564 | 1,323 | 2,342 | 1,952 |
| multihop_qa | 12 | 2,597 | 1,119 | 2,287 | 1,557 |
| openmath | 8 | 807 | 740 | 1,579 | 1,374 |
| rag_evidence | 24 | 671 | 458 | 953 | 584 |

* **Derivation length** — `<think>` tokens of the selected candidate, the
  quantity `verify.select`'s docstring was actually worried about:

| type | n | median p50 | shortest p50 | median mean | shortest mean | ratio |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 29 | 1,132 | 1,057 | 2,004 | 1,630 | 1.23× |
| multihop_qa | 12 | 2,590 | 1,109 | 2,224 | 1,489 | 1.49× |
| openmath | 8 | 637 | 598 | 1,325 | 1,122 | 1.18× |
| rag_evidence | 24 | 551 | 367 | 867 | 495 | 1.75× |
| **all** | **73** | **922** | **571** | **1,592** | **1,178** | **1.35×** |

Shortest-survivor was discarding roughly a quarter of the reasoning trace on the
prompts where it had a choice. The recorded concern was real and is now measured.

* Corpus totals (supervised tokens): gsm8k 1,810,430 vs 1,799,132 ·
  multihop_qa 991,280 vs 982,530 · rag_evidence 2,058,723 vs 2,049,852 ·
  openmath 1,272,977 vs 1,271,339 · `code` and `tool_calling` identical
  (0 and 1 replacements).
* At the rung: median 858,409 supervised / 682 blocks / **89.1%** overlap;
  shortest 839,819 / 682 / 88.2%. **Median wins on both**, so nothing was traded
  for the derivation lengths.

Keep/replacement counts are identical under both rules by construction
(the gates are the same): 10,778/11,174 kept (96.5%), `openmath` weakest at
keep 0.713 / replacement 0.099 / 166 prompts with no correct candidate.

## 4. The 0.86M D1 corpus audit

`artifacts/stage3/e2_d1_corpus_audit.json`. D1 is built by
`scripts/data/build_matched_rung.py` — new, because re-cutting the whole ladder
put only **66.6%** of D0's prompts inside the 682-block prefix even with the
session order anchored. The builder instead packs D0's own rung sessions
(cleaned), tops up per type from outside the rung, and cuts D0's exact block
count.

| quantity | D0 | D1 | Δ |
|---|---:|---:|---:|
| packed blocks | 682 | 682 | **0.0000%** |
| optimizer steps | 1,023 | 1,023 | **0.0000%** |
| packed tokens | 5,586,944 | 5,586,944 | **0.0000%** |
| tokens processed | 16,760,832 | 16,760,832 | **0.0000%** |
| effective epochs | 3.0000 | 3.0000 | — |
| **unique supervised tokens** | 864,750 | **858,409** | **−0.733%** |
| sessions | 1,502 | 1,479 | −1.53% |
| terminal truncations | 109 | 104 | — |
| supervised discarded by truncation | 202,365 | 120,476 | — |

**Prompt overlap 1,339 / 1,502 = 89.1%** (Jaccard 81.5%), against a ceiling of
96.5% (1,450 of D0's rung sessions survive cleaning at all). 29 shared prompts
received a different completion. Dropped from D0: 163 (code 8, gsm8k 20,
multihop 36, openmath 24, rag 39, tool 36). Added to D1: 140 clean, unique
prompts from outside D0's rung — **no duplication, no truncated completion.**

Per-type at the rung, and share drift ≤ **0.17 pp**:

| type | D0 tokens | D1 tokens | D0 share | D1 share | D0 sess | D1 sess |
|---|---:|---:|---:|---:|---:|---:|
| code | 144,107 | 143,772 | .1666 | .1675 | 87 | 79 |
| gsm8k | 143,878 | 141,372 | .1664 | .1647 | 166 | 168 |
| multihop_qa | 144,391 | 143,224 | .1670 | .1668 | 215 | 205 |
| openmath | 144,671 | 144,287 | .1673 | .1681 | 95 | 84 |
| rag_evidence | 144,321 | 143,851 | .1669 | .1676 | 366 | 363 |
| tool_calling | 143,382 | 141,903 | .1658 | .1653 | 573 | 580 |

Target lengths (supervised tokens/session, p50): code 1,716→1,609 ·
gsm8k 771→769 · multihop 596→601 · openmath 1,354→**1,543** ·
rag 338→338 · tool 193→193. Cleaning did not shorten targets; on `openmath`
it lengthened them, which is the median rule working as intended.

**Validation is byte-identical.** D0's 16 validation blocks are appended to the
D1 pack verbatim, and the pack was loaded through the real
`aadistill.data.ladder` path to prove it: same `input_ids`, `ce_mask` and
`content_mask` tensors, sha256 `4d36705cfcf414af…`, 81,195 supervised tokens.
Without this the treatment would have been scored on a different validation set
than the control, silently invalidating the sharpest instrument in the
experiment (val CE resolves the Experiment 1 effect at 74× the seed noise).

### The overlap/mixture frontier, and where it was cut

Packing the matched pool with no slack maximises overlap but leaves the
block-level mixture repair nothing to choose between. Measured:

| pool overshoot | prompt overlap | max share drift | supervised | Δ vs D0 |
|---:|---:|---:|---:|---:|
| 1.00 | **96.5%** | **4.13 pp** | 857,271 | −0.86% |
| 1.10 | 90.9% | 0.92 pp | 815,247 | −5.72% |
| 1.15 | 90.1% | 0.37 pp | 831,602 | −3.83% |
| 1.18 | 89.0% | 0.18 pp | 851,714 | −1.51% |
| **1.20** | **89.1%** | **0.20 pp** | **858,409** | **−0.73%** |
| 1.25 | 87.8% | 0.22 pp | 848,022 | −1.93% |
| 1.50 | 82.6% | 0.20 pp | 845,605 | −2.21% |
| 2.00 | 74.1% | 0.17 pp | 855,301 | −1.09% |

Selection is by a pre-registered priority — exact compute, then mixture drift
within 0.25 pp, then maximum prompt overlap — and 1.20 wins it while also
happening to give the best token match. The 4.13 pp option was rejected for the
same reason block-order anchoring was rejected at 2.96M: it would convert a
target-quality experiment into a mixture experiment.

### KD-target alignment: nothing to recompute

Teacher distributions are computed **online** — `Trainer._micro_losses` runs the
teacher on the same packed block as the student every step (`train.py:493`), and
no cached, stored or top-k logits exist anywhere in the repository. A replaced
completion is therefore paired with the teacher distribution of its own
teacher-forced prefix, system message and serialization by construction, and
**teacher-logit recomputation cost is zero** — it is already inside the measured
step time.

## 5. Arms and the reuse chain

| arm | data | loss | LR | trained? |
|---|---|---|---|---|
| **D0** | E1 0.86M, 864,750 sup tok | CE 0.25 + fwd-KL 1.0 | 5e-5 | **exists — reuse** |
| **D1** | clean-v2 0.86M, 858,409 sup tok, 682 blocks | identical | identical | phase 1 |
| **L0** | = D1 if it passes the gate, else = D0 | identical | identical | **reuse** |
| **L1** | = L0's data | **KL-only** (`ce_weight` 0 → CE removed; KD coefficient, direction, τ, τ² scaling, mask, prefix, reduction and normalizer all preserved) | identical | phase 2 |
| **R0** | = L1 if KL-only passes, else = L0 | selected | 5e-5 | **reuse** |
| **R1** | R0's data + objective | selected | **η/2** | phase 3 |
| **R2** | R0's data + objective | selected | **η/4** | phase 3 |

Every new arm starts from the Stage 1 PCA init `86fbba78…` at seeds 20260726 and
20260801, at 682 blocks / 1,023 steps, with the same trainable set, optimizer,
packing, batch construction, scheduler shape, warmup proportion, system-message
policy, serialization contract, validation blocks and evaluation protocol.
`configs/stage3/e2/e2_d1_s{a,b}_pca.json` differ from
`configs/stage3/e1/e1_r0860k_s{a,b}_pca.json` in exactly five fields:
`data_dir`, `rung`, `run_name`, `out_dir`, `_purpose`, plus the checkpoint and
eval intervals that §8 requires. **No control is retrained.**

For R1/R2 the whole schedule scales, not just the peak: `min_lr_frac` is a
fraction of the peak so the floor follows automatically, and warmup stays a fixed
proportion of the run.

## 6. The frozen capability battery (`capability-v1`)

Built and frozen **before** D1 trains: `artifacts/eval/battery_v1/`, manifest
sha256 `a194179a88b2270c7cb05d6528910ee9ab77d2e3a8c751995ebb9c059f1bab46`.
Nothing in it may be tuned after results are seen.

| set | n | source | split | license | sha256 |
|---|---:|---|---|---|---|
| `knowledge` | 150 | `mandarjoshi/trivia_qa` `rc.nocontext` | validation | Apache-2.0 | `2d4420ce…` |
| `math_verified` | 100 | `HuggingFaceH4/MATH-500` | test | MIT | `bf73cb4a…` |
| `gsm8k` | 100 | `openai/gsm8k` `main` | test | MIT | `1ad4ad22…` |
| `multihop` | 100 | `hotpotqa/hotpot_qa` `distractor` | validation | CC-BY-SA-4.0 | `3bd25d89…` |
| `rag` | 100 | `rajpurkar/squad_v2` | validation | CC-BY-SA-4.0 | `1e31c9e0…` |
| `refusal_paired` | 120 (60 pairs) | `rajpurkar/squad_v2` | validation | CC-BY-SA-4.0 | `dfb680fc…` |
| `behavior_v0` | 76 | reused unchanged from Experiment 1 | — | — | — |
| **total** | **746** | | | | |

`behavior_v0` is reused verbatim so D0's stored behaviour generations stay
comparable without a re-run. Every set carries its exact sample ids in the
manifest.

**Scoring is deterministic everywhere; no LLM judge is a primary scorer.**
Alias-set exact match (knowledge); numeric → rational → symbolic → normalized
comparison of a `\boxed{}` answer (math); strict `\boxed{}`-or-explicit-marker
(GSM8K); span containment plus supporting-title recall (multihop); span
containment plus evidence attribution, unsupported-claim rate and echo detection
(RAG); paired must-answer / must-refuse (refusal). Scorer
`src/aadistill/evaluation/capability.py` sha256 `6553fdc7…`,
`strict_answer.py` `ad82892f…`, degeneration detector pinned in the manifest.

**Decoding is identical to Experiment 1's** — greedy, unrestricted within the
8,192 effective context derived from the trained `block_len`, stop ids from the
model's `generation_config`, degeneration stop on, mandatory system message with
a sample's own preserved. D0 and every new arm use the same prompts, decoding
and evaluators.

### Leakage

Two independent guarantees.

1. **Structural.** `build_stage2_v1.py` drew every source from its `train`
   split, so the `validation`/`test` splits used here were never eligible for
   corpus v2. TriviaQA and MATH-500 have never been used by this project at all.
2. **Hash.** Every candidate is checked with the corpus's own `content_key` /
   `prompt_key` rule against 65,913 content hashes and 59,113 reserved prompt
   hashes (`stage2_v1` train+val+calib, `eval_behavior_v0`) plus **10,128
   corpus-v2 prompt hashes**.

**Result: 0 collisions in 15,107 candidates.** The check is not vacuous — a
self-test confirms a real corpus-v2 prompt does hash into the exclusion set, so
zero is a measurement, not a wiring bug.

**Recorded weakness:** `multihop` and `rag` share source *families* (HotpotQA,
SQuAD v2) with corpus v2's training slices — item-disjoint and on a different
split, so near-domain rather than out-of-domain. `knowledge` and `math_verified`
are fully out-of-domain.

### Evaluator validation (CPU, before any model output)

`tests/evaluation/test_capability.py`, 84 tests. Every scorer is run against
known **correct, incorrect, malformed, tool-call, refusal and degenerate**
outputs, and then against every row of the frozen sets:

* gold answers score **100%** on `knowledge` (150/150), `multihop` (100/100),
  `rag` (100/100) and `math_verified` (100/100 with the gold boxed —
  verification paths 63 numeric / 17 rational / 5 symbolic / 15 normalized);
* unrelated answers score **0%** on every set;
* every scorer rejects every malformed mode (degenerate, unterminated, missing
  `</think>`, unexpected tool call, empty) even when the correct answer is
  present;
* **an always-refusing policy wins 0 of 60 pairs** while scoring 0.5 per row —
  verified end-to-end through `score_battery.py` on synthesized generations.

The suite found and fixed two real evaluator defects before any GPU time: the
math scorer scored `\boxed{0.5}` wrong against gold `1/2` (sympy's LaTeX parser
needs an uninstalled `antlr4` runtime, so the symbolic path silently failed — now
a rational path handles it and the symbolic path no longer depends on `antlr4`),
and the RAG echo check compared against the instruction alone instead of the
instruction plus context, so copying the passage back would have passed.

### Where the battery runs

| checkpoint | full battery? |
|---|---|
| all 9 eval points | no — CE, held-out NLL, behaviour metrics and generations only |
| final | **yes** |
| best validation CE | **yes** |
| best held-out NLL | **yes** |
| deterioration bracket (2) | **yes** |
| identical checkpoints | scored **once**; the reasons collapse in practice |
| **D0** | **fixed-step endpoint only** — Experiment 1 kept no other checkpoint |

**Fixed-step D0↔D1 conclusions and within-D1 trajectory conclusions are reported
separately and never merged.** Only the endpoint is a matched D0 comparison;
everything else is a statement about D1's own trajectory.

## 7. The reasoning floor, treated explicitly

Corrected D0 strict GSM8K EM at 0.86M is **0.000 on both seeds**. Therefore:

* **GSM8K strict EM is preregistered as a one-sided improvement metric.** Any
  increase counts for D1; no decrease is possible, so it can never reject D1.
* **No no-degradation gate is defined at a zero baseline** — a gate that cannot
  fire is not a gate.
* **`0 → 0` is not evidence that reasoning was preserved.** It is evidence the
  metric is uninformative at this scale.
* `math_verified` (MATH-500, harder and out-of-domain) and `multihop` exist to
  give reasoning a more discriminating baseline. Their D0 values are unknown
  until D0's endpoint is scored, which happens in phase 1.
* **If every reasoning metric — strict GSM8K, `math_verified`, `multihop` — is at
  floor on both D0 and D1, reasoning preservation is reported as
  `inconclusive`**, not as preserved and not as damaged.
* **No post-hoc composite will be created or retuned** to conceal the floor.

## 8. Cost, from measured 0.86M timestamps and the frozen battery

**Baseline: verified cumulative spend $96.02. Experiment 2 incremental hard cap
$30.00. New cumulative hard cap $126.02.**

Measured, not extrapolated — Experiment 1's own 0.86M PCA orchestrator
timestamps: `sa` 10:01:36 → 11:03:03 (**3,687 s**), `sb` 21:06:33 → 22:07:58
(**3,685 s**), `post_run` 72 s and 73 s. That is **3.603 s/step** at 1,023 steps.

Per-seed run = 1.024 h training + 0.020 h post-run + 0.100 h inline held-out NLL
at the 8 intermediate eval points + 0.013 h checkpoint writes = **1.157 h**.

Evaluation: Experiment 1's sweep measured 0.234 h per checkpoint on 176 prompts;
splitting that into 0.05 h fixed and the rest generation gives **0.831 h per
checkpoint** for the frozen 746-prompt battery.

| phase | seeds | training h | battery ckpts exp/pess | eval h | expected h | expected $ | pessimistic h | pessimistic $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 — data (D1) | 2 | 2.31 | 5 / 6 | 4.16 | 7.22 | **$7.15** | 11.16 | **$11.05** |
| 2 — loss (L1) | 2 | 2.31 | 3 / 4 | 2.49 | 5.56 | **$5.50** | 9.08 | **$8.99** |
| 3 — LR (R1, R2) | 4 | 4.63 | 6 / 8 | 4.99 | 10.37 | **$10.26** | 16.13 | **$15.97** |
| **total** | **8** | **9.26** | **14 / 18** | **11.64** | **23.15** | **$22.92** | **36.37** | **$36.01** |

### What these numbers include — explicitly

| item | included? |
|---|---|
| generation **and** scoring of the complete 746-prompt battery | **yes**, at 0.831 h/checkpoint |
| D0 final-checkpoint capability evaluation, both seeds | **yes**, inside phase 1's 5 checkpoints, charged once for the whole experiment |
| capability evaluation of the selected D1/L1/R1/R2 checkpoints | **yes** — final always, best-val-CE and best-NLL when they differ |
| **online teacher forwards** | **yes** — inside the measured 3.603 s/step; there is no logit cache, so no separate line exists |
| checkpoint transfer and processing | **yes**, in the 0.25 h/phase upload-and-verify allowance, pod still billing |
| pod idle time during evaluation and artifact handling | **yes** — every hour above is billed pod time, including 0.5 h/phase provisioning |
| restart / failure allowance | **pessimistic column only**: one lost run per phase, +25% training rate, +25% evaluation for the degeneration surcharge, and the full best-checkpoint set |
| scoring (`score_battery.py`, `rescore_gsm8k.py`) | **not billed** — CPU, offline, re-runnable |

### The result, and the one thing it changes

**Expected $22.92 — fits, $7.08 of headroom, cumulative $118.94.**

**Pessimistic $36.01 — exceeds the $30 incremental cap by $6.01** (cumulative
$132.03 against $126.02). The previous $21.02 / $29.84 assumed a 626-prompt
battery; freezing it at 746 and adding a degeneration surcharge is what moved it.

Reported rather than absorbed: no seed, evaluation set, training length or
standard has been reduced to make the arithmetic fit.

**Phase 1 by itself is nowhere near the cap** — $7.15 expected, $11.05 worst
case. The pessimistic total compounds a lost run, a slow rate, a degeneration
surcharge and the maximum checkpoint set across all three phases at once, and the
sequential design exists precisely so phases 2 and 3 are re-costed against
measured spend before either launches. Three options, in the order I would take
them:

1. **Approve phase 1 only** ($7.15 / $11.05) and re-cost phases 2–3 from what
   phase 1 actually costs. Most of the pessimistic margin is uncertainty phase 1
   resolves.
2. **Raise the incremental cap to $36** to cover the compounded worst case now.
3. **Reduce coverage** — not recommended, and it needs explicit approval.

### 8.1 Operational constraints at launch

* **One pod, created with `--min-cuda-version 13.0`** so the same host trains and
  runs the vLLM battery. Experiment 1 needed a separate evaluation pod because
  the training driver could not host vLLM 0.26.
* **~100 GB container disk** for 9 checkpoints/arm at 4.3 GB during a run.
* **Pods idle-bill.** Teardown is tied to job completion, not to a generous
  `--terminate-after`, and status polling is set up at launch.

## 9. Checkpoint inventory, cleanup and persistence

### 9.1 Inventory

`scripts/pod/checkpoint_inventory.py` → `artifacts/stage3/checkpoint_inventory.json`.
Every weight and optimizer-state file on both stores, with size, hash,
duplicate status, what needs it and a proposed action. Classification is
declared in the tool, not inferred; anything matching neither the required list
nor a provable-obsolescence rule is `decide` — retained and flagged.

| store | files | size | retain | delete | decide |
|---|---:|---:|---:|---:|---:|
| dev box | 9 → 7 | 17.51 → 13.32 GiB | 3 | 2 | 4 |
| Hugging Face relay | 34 | 73.28 GiB | 4 | 0 | 30 |

**Duplicate detection is real, and it found exactly two.** The local Stage 1
`checkpoint` (`86fbba78…`) and `random_baseline` are byte-identical to their
relay copies, matched on the LFS object sha256 without downloading anything.
**Every other checkpoint on either store is single-copy.**

### 9.2 Deleted

| file | size | why |
|---|---:|---|
| `artifacts/stage3/_smoke_ladder/checkpoints/step_000002/model/model.safetensors` | 2.22 GiB | two-step CPU smoke test of the ladder loader, superseded by 24 real Experiment 1 arms on the same code path |
| `artifacts/stage3/_smoke_ladder/checkpoints/step_000002/trainer_state.pt` | 1.97 GiB | optimizer state of that smoke test; it will never be resumed |

**4.19 GiB reclaimed on the dev box: 117 → 121 GiB free.** Its
`run_manifest.json`, `train_log.jsonl` and model config were kept, so the run
remains reproducible from records.

Nothing else was deleted, and that is the honest outcome rather than a thin one:

* the four dev-box-only Experiment 1 arms (`e1_r2960k_sb_pca`,
  `e1_r2960k_sb_rand`, `e1_r5500k_sb_pca`, `e1_r5500k_sb_rand`) are the **only
  copy** of those weights;
* `e1_ctl_r0250k_sa_pca_stepmatched` carries the single strongest Experiment 1
  finding — that held-out NLL tracks optimizer steps, not data — and is
  single-copy;
* the 30 relay `decide` entries (Experiment 1's other rungs and the pre-E1
  Stage 3 line: `s1_ffn_norm_v0`, `s1_ext_v0`, `s2_blocks_v0/v1`,
  `s2v1_from_init/from_s1`, `s2v1_bl2048{,_seedB}`, `kdconf_ctrl_{a,b}`,
  `kdconf_nothink_{a,b}`) are not required by the D0→D1→L1→R1/R2 chain or the
  frozen battery, but their diagnostic value is not provably zero. The
  `kdconf_*` pair in particular is a **`kd_scope` A/B**, which is adjacent to
  phase 2's loss question — the weights are unlikely to be reused given P17 and
  a different data regime, but that is a judgement, not a proof. Flagged,
  retained.

### 9.3 The Hugging Face boundary — nothing reclaimed, and why

**0 bytes were reclaimed on the relay, and no relay file was touched.**

Removing a file from the current revision **does not reclaim LFS quota**: the
object stays referenced by history. This was measured on 2026-08-02 — deleting
19.07 GB of superseded `tt2x2`/`ttb` weights dropped the working tree to
80.31 GB and reclaimed **nothing**.

So relay reclamation requires one of the operations that are out of scope
without explicit approval. Reported, not performed:

| option | reclaims | risk |
|---|---:|---|
| squash history (approved in principle 2026-08-02, never run) | up to ~73 GiB of superseded LFS objects | **invalidates every existing revision hash**; every artifact manifest entry that pins a revision becomes unresolvable; irreversible |
| `super_squash_history` on the repo | same | same, plus it is a single opaque commit — provenance for 20 Experiment 1 arms collapses |
| delete and recreate the repo | all of it | destroys every revision and every uploaded artifact; unacceptable |
| move superseded prefixes to a second repo, then delete the first | ~26 GiB (`stage3/` pre-E1) | requires re-uploading 12 checkpoints, and the source repo still needs a history operation to actually free the objects |

**Recommendation: do none of them now.** Experiment 2 does not need relay space —
weights go to the dev box (§9.5), and only small files go to the relay, which has
always worked because they are not LFS objects. The squash is a separate
destructive decision to take on its own merits, not folded into a launch.

### 9.4 Retention policy for Experiment 2 arms

`scripts/pod/retain_checkpoints.py` (13 tests), unchanged from the previous
revision and reaffirmed here.

| eval point | metrics + generations | weights | optimizer state |
|---|---|---|---|
| all 9 (0, 127, …, 1016, 1023) | **always** | only if selected below | no |
| final (1,023) | yes | **always** | latest only, then dropped |
| best validation CE | yes | **always** | no |
| best held-out NLL | yes | **always** | no |
| deterioration onset, and the eval after it | yes | **always** | no |

Onset is the first step after which held-out NLL rises for **two consecutive**
evaluations — not the first up-tick, against a 0.489-nat between-seed spread. The
raw trajectory is retained regardless, so a different definition can be applied
later without re-running anything. Optimizer state is discarded except where an
active run must stay resumable.

**Held-out NLL is scored by the orchestrator per checkpoint and merged from
`holdout_trajectory.jsonl`. The trainer is not modified** — byte identity with
D0's trainer is the one thing an A/B against D0 cannot give up.

### 9.5 Storage plan, from the post-cleanup state

Reasons collapse in practice — val CE is monotone at this budget, so
final == best-val-CE — giving **≈4 distinct checkpoints per arm ≈ 9.2 GB**.

| phase | arms | retained | storage |
|---|---:|---:|---:|
| 1 (D1) | 2 | ~4 each | ~18 GB |
| 2 (L1) | 2 | ~4 each | ~18 GB |
| 3 (R1, R2) | 4 | ~4 each | ~37 GB |
| **total** | **8** | | **~73 GB** |

Metrics, trajectories, generations, battery results and manifests: **< 1 GB.**

* **Primary store: this dev box**, `artifacts/stage3/<arm>/`. **121 GiB free**
  after cleanup — enough for all three phases with ~48 GiB spare. Nothing
  required is ever left only on a paid pod.
* **Small files also to the relay** under `e2_0860k_<phase>/` — not LFS objects,
  so the quota does not block them, exactly as Experiment 1's evaluation JSONs.
* **Weights are not planned onto the relay.** The squash is therefore not a
  prerequisite for Experiment 2.

**Verification and recovery.** `retain_checkpoints.py --apply` hashes every
surviving file into `retention.json` *before* deleting anything. Teardown order:
(1) evaluate, (2) retain + hash on the pod, (3) transfer, (4) verify hashes on
the dev box, (5) upload small files to the relay, (6) **delete the pod only after
verification passes**. A dry run prints the plan without deleting, and the final
step can never be pruned.

## 10. Pre-registered per-metric and phase gates

Fixed before training, from the measured 0.86M seed variation in §2. Where a D0
baseline does not exist yet (the new capability sets), the threshold is stated as
a rule against the D0 value that phase 1 measures, not as a number invented now.

### Per-metric

| metric | role | rule |
|---|---|---|
| held-out NLL | **primary** | improve by **> 0.489 nats** on the seed mean — the measured between-seed \|Δ\| at this rung — with the same sign on both seeds |
| teacher-native val CE | reported | seed \|Δ\| is 0.0063, so any change > 0.02 is real; not a gate, because D1 optimises a different target text |
| `knowledge` | guard rail | ≥ D0 − 0.05 |
| `math_verified` | one-sided | see §7 — floor expected; improvement counts, no decrease possible if D0 is 0 |
| `gsm8k` strict EM | one-sided | **cannot reject D1** (D0 = 0.000) |
| `multihop` answer | guard rail | ≥ D0 − 0.05 |
| `multihop` evidence recall | reported | separately, never merged with answer correctness |
| `rag` correct | guard rail | ≥ D0 − 0.05 |
| `rag` attribution / unsupported-claim / echo | reported | four separate numbers |
| `refusal_paired` **pair accuracy** | guard rail | ≥ D0 − 0.05; per-row accuracy is never the headline |
| protocol-valid rate | guard rail | ≥ D0 − 0.05 |
| natural termination | guard rail | ≥ D0 − 0.05 (seed \|Δ\| 0.026) |
| degeneration rate | guard rail | ≤ D0 + 0.05 |

### Phase gate

**Select D1** if held-out NLL passes its primary rule **and** no guard rail is
breached.

**Reject D1** if held-out NLL does not improve beyond noise → cleaning does not
explain the deterioration; D0 stays phase 2's dataset.

**Stop and report** if D1 improves NLL while breaching any guard rail. No
composite will be improvised to break such a tie, and none will be created or
retuned after results are seen.

**Report `inconclusive`** for reasoning preservation if strict GSM8K,
`math_verified` and `multihop` are all at floor on both arms.

Paired bootstrap CIs on the **1,339 shared prompts** for every generation metric;
val CE and held-out NLL per seed, as a mean, and as full trajectories.
**Fixed-step D0↔D1 conclusions are reported separately from within-D1 trajectory
conclusions** — only the endpoint is a matched D0 comparison.

## 11. What will not happen

No Cartesian sweep. No phase launched before the previous one reports and is
approved. No retraining of any valid historical or preceding control. No arm at
any rung other than 0.86M. No one-seed substitution, no shortened training, no
reduced evaluation set to fit a budget.
