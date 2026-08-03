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

## 6. Remaining zero-GPU prerequisite, not yet done

The capability decomposition in the maintainer's brief needs held-out sets that
**do not exist yet**: leakage-safe factual/knowledge QA, verifier-backed math,
multihop, RAG evidence-supported correctness, unsupported-refusal. Experiment 1's
battery covers only behaviour (76 prompts) and GSM8K (100).

They are cheap to build (CPU, no generation) but their design has choices —
sources, leakage rules against corpus v2, size — that should be settled rather
than assumed, and the cost table in §7 already prices the larger battery. **Flag,
not a blocker:** phase 1 can run against the existing battery if you prefer, at
roughly $2 less. Say which and it is built before launch.

## 7. Cost, from measured 0.86M timestamps

**Baseline: verified cumulative spend $96.02. Experiment 2 incremental hard cap
$30.00. New cumulative hard cap $126.02.**

Measured, not extrapolated — Experiment 1's own 0.86M PCA orchestrator
timestamps: `sa` launched 10:01:36 → `TRAIN_DONE` 11:03:03 (**3,687 s**),
`sb` 21:06:33 → 22:07:58 (**3,685 s**); `post_run` 72 s and 73 s. That is
**3.603 s/step** at 1,023 steps.

Per-seed run = 1.024 h training + 0.020 h post-run + 0.100 h inline held-out NLL
at the 8 intermediate eval points + 0.013 h checkpoint writes = **1.157 h**.

Evaluation: Experiment 1's sweep was $5.80 for 25 checkpoints = 0.234 h each on
176 prompts. Splitting that into ~0.05 h fixed and the rest generation, the 626-
prompt Experiment 2 battery costs **0.706 h per checkpoint**.

| phase | seeds | training h | eval ckpts | eval h | expected h | expected $ | pessimistic h | pessimistic $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 — data (D1) | 2 | 2.31 | 5 | 3.53 | 6.54 | **$6.48** | 9.08 | $8.99 |
| 2 — loss (L1) | 2 | 2.31 | 3 | 2.12 | 5.13 | **$5.08** | 7.67 | $7.60 |
| 3 — LR (R1, R2) | 4 | 4.63 | 6 | 4.23 | 9.56 | **$9.47** | 13.39 | $13.25 |
| **total** | **8** | **9.26** | **14** | **9.88** | **21.24** | **$21.02** | **30.15** | **$29.84** |

Each phase includes 0.50 h pod provisioning and 0.20 h upload/hash/teardown;
pessimistic adds 25% to the training rate, the full checkpoint set, and one lost
run per phase.

**Pod occupancy during checkpoint and generation processing** is inside these
numbers: 0.113 h/seed of in-run checkpointing and held-out scoring, 0.706 h per
evaluated checkpoint, and 0.20 h/phase of upload and hash verification with the
pod still billing.

**Verdict: the complete sequential experiment fits.** Expected **$21.02**, which
leaves **$8.98** of the $30 and lands cumulative at **$117.04**. The pessimistic
path is **$29.84** — inside the cap with **$0.16** to spare, i.e. no margin at
all. The sequential structure is the control: after phase 1 the real spend is
known and phases 2–3 are re-costed against what remains before either is
launched. Nothing is shortened, single-seeded or evaluated on a reduced set to
make the arithmetic work; if phase 3 will not fit when its turn comes, that is
reported and paused rather than trimmed.

## 8. Checkpoint retention and persistence

Resolved before launch, by `scripts/pod/retain_checkpoints.py` (13 tests).

**The shape of the problem.** Reducing the rung cut optimizer steps 2,916 →
1,023 but not checkpoint size: each is **4.3 GB** (2.3 GB weights + 2.0 GB
optimizer state). Nine per arm is 39 GB, and eight arms would be 310 GB.

**Retention policy.**

| eval point | metrics + generations | weights | optimizer state |
|---|---|---|---|
| all 9 (0, 127, …, 1016, 1023) | **always** | only if selected below | no |
| final (1,023) | yes | **always** | latest only, then dropped |
| best validation CE | yes | **always** | no |
| best held-out NLL | yes | **always** | no |
| deterioration onset, and the eval after it | yes | **always** | no |

Onset is the first step after which held-out NLL rises for **two consecutive**
evaluations — not the first up-tick, which is meaningless against a 0.489-nat
between-seed spread. The raw trajectory is retained regardless, so a different
definition can be applied later without re-running anything.

The reasons collapse in practice: val CE is monotone at this budget, so
final == best-val-CE, giving **≈4 distinct checkpoints per arm ≈ 9.2 GB**.

| phase | arms | retained | storage |
|---|---:|---:|---:|
| 1 (D1) | 2 | ~4 each | ~18 GB |
| 2 (L1) | 2 | ~4 each | ~18 GB |
| 3 (R1, R2) | 4 | ~4 each | ~37 GB |
| **total** | **8** | | **~73 GB** |

Plus metrics, trajectories, generations and manifests: **< 1 GB total.**

**Where it lives.**

* **Primary: this dev box** under `artifacts/stage3/<arm>/`. **117 GB free**,
  measured — enough for all three phases with headroom. Nothing required is ever
  left only on a paid pod.
* **Small files** (trajectories, metrics, generations, manifests, hashes) also
  go to the relay under `e2_0860k_<phase>/`; these are not LFS objects and upload
  fine, exactly as Experiment 1's evaluation JSONs did.
* **The relay is still at its private-LFS limit** and the approved history squash
  has not run. **Weights are therefore not planned onto the relay**, and the
  squash is not a prerequisite for Experiment 2. It stays an open item, and it is
  a destructive operation on a shared store that will be confirmed separately
  rather than folded into this launch.

**Verification and recoverability.** `retain_checkpoints.py --apply` hashes every
surviving file into `retention.json` before anything is deleted; the transfer to
the dev box is verified against those hashes, and the pod is torn down only after
verification passes. Order at teardown: (1) evaluate, (2) retain + hash on the
pod, (3) transfer, (4) verify hashes on the dev box, (5) upload small files to the
relay, (6) delete the pod. A dry run prints the plan without deleting; the final
step can never be pruned.

## 9. Pre-registered D1 selection gate

Fixed before training, from the measured 0.86M seed variation in §2.

**Select D1** only if **both**:

* held-out NLL improves by **more than 0.489 nats** on the seed mean — the
  measured between-seed |Δ| at this exact rung — **and** the improvement has the
  same sign on both seeds;
* and no material degradation: protocol-valid rate ≥ D0 − 0.05; natural
  termination ≥ D0 − 0.05 (seed |Δ| 0.026); degeneration ≤ D0 + 0.05; RAG /
  grounding and tool-call axes ≥ D0 − 0.10 (both small-n).

**GSM8K strict EM is at floor (0.000 on both seeds), so it is a one-sided
criterion here**: any increase counts in D1's favour, and no decrease is
possible. It cannot be used to reject D1, and that asymmetry is recorded rather
than hidden inside a "no degradation" clause that could never fire.

**Reject D1** if held-out NLL does not improve beyond noise → report that
cleaning does not explain the deterioration; D0 stays phase 2's dataset.

**Stop and report** if D1 improves NLL while degrading any axis past its
threshold. No composite score will be improvised to break such a tie.

Paired bootstrap CIs on the **1,339 shared prompts** for every generation metric;
val CE and held-out NLL reported per seed, as a mean, and as full trajectories.

## 10. What will not happen

No Cartesian sweep. No phase launched before the previous one reports and is
approved. No retraining of any valid historical or preceding control. No arm at
any rung other than 0.86M. No one-seed substitution, no shortened training, no
reduced evaluation set to fit a budget.
