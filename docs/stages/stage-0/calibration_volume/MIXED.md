# Mixed calibration 2×: design recommendation and execution application

**Status (2026-10-26): $0 design complete; nothing paid is authorized.** No pod,
GPU or network volume has been created for it. The execution needs the
maintainer's grant **and** a project-cap increase from $731 to $740 (section 9).

This page owns the reading. The numbers are owned by the records it cites:
- design: [`mixed_design.json`](../../../../configs/stages/stage-0/calibration_volume/mixed_design.json);
- profiles: [`profiles_mixed.json`](../../../../configs/stages/stage-0/calibration_volume/profiles_mixed.json);
- audit (coverage, isolation, compute):
  [`design/mixed_feasibility.json`](../../../../logs/stages/stage-0/calibration_volume/design/mixed_feasibility.json);
- control protocol:
  [`plans/mixed_preregistration.json`](../../../../logs/stages/stage-0/calibration_volume/plans/mixed_preregistration.json);
- application:
  [`plans/mixed_application.json`](../../../../logs/stages/stage-0/calibration_volume/plans/mixed_application.json).

## 1. Recommendation

- **Run 2× as the minimal formal design.**
  - One formal D1 full search on one mixed profile, `calib.s0m_mix_2x@v1`.
  - Its Top-1 is recovered and evaluated exactly as D4-Augmented, on both
    seeds.
  - The result is decided against D4-Augmented's existing two units, which
    are reused rather than retrained or re-evaluated.
- **Do not run 4× now.** It enters only on a preregistered result of the 2×
  comparison (section 6).
- **Money.**
  - Expected cost: $17.95.
  - Hard-capped envelope: $33.05.
  - The project has $24.15 left under its cap, so the cap must rise to $740
    first.

## 2. What D4's calibration already was

Historical D4 was not calibrated without fluid data. Both of its profiles were
mostly static fluid reasoning, by realized scored positions (UESP):

| profile | UESP | static fluid | supporting | interactive | units | rule families |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `calib.s0v_base_d4` | 203,726 | 144,890 (71.1%) | 58,836 | 0 | 85 | 21 |
| `calib.s0v_rh_d4` | 215,935 | 188,503 (87.3%) | 27,432 | 0 | 75 | 24 |

D4 Top-1's path was:

```
ATTENTION(rh_d4) → DEPTH(base_d4) → RESIDUAL_WIDTH(base_d4) → FFN(base_d4)
```

What D4 lacked was interactive data and a supporting share near one half.

**Its calibration also overlapped the recovery training data.**
- The base pool predates E1's 2.96M rung.
- 8 of `base_d4`'s supporting items (7,457 of its 203,726 positions) are
  sessions D4-Augmented later trained on.

## 3. The mixed profile

**Sources (no new teacher data).** Everything comes from the verified
teacher-native `base_pool_v1`, the same pool D4's profiles were drawn from.

- **Fluid half: all five P10.2 axes.**
  - 506 static units over 8 calibration-eligible generators.
  - 13 complete interactive episodes from `hidden_buttons` and
    `probe_function`.
- **Supporting half: knowledge-dependent groups.** gsm8k, openmath,
  multihop_qa and rag_evidence, the four groups D4 drew.
  - Code and tool calling stay out, as they did for D4.
  - No closed-book factual QA source exists among the verified teacher-native
    data.

**Sampling rule** (`build_mixed_profiles.py`, one seed: 179444901, derived
from the declared string and never re-drawn).

- **Fluid quotas.**
  - The fluid half's nominal size is split equally over the five axes.
  - A static axis splits its share equally over its generators.
  - The interactive axis is **one** cell over both generators, so the
    at-least-one rule does not double it. An episode is taken whole, with all
    its turns.
- **Family-interleaved order.** Within a cell, the rule families are
  interleaved in seed-keyed order, so n units span min(n, families) families.
  Nothing reads a length or a result.
- **Supporting quotas.** The supporting half matches the fluid half's
  **realized** positions:
  - four equal cells;
  - each stopped at the prefix nearest its quota.
- **Nesting.** 4× takes a longer prefix of the same orders in every cell, so
  **2× ⊂ 4×**. The builder refuses otherwise.

**Exclusions.**
- Every unit sharing a session id, source id or prompt key with what
  D4-Augmented trained or validated on. That is E1's 2.96M training prefix,
  its 16 validation sequences and the fluid increment, which removes 1,713
  supporting units. No fluid unit was a member.
- eval-set-1's held-out units under the frozen Stage-2 isolation rules.
  - R1 (no shared key) passes.
  - R2 (13-gram near-duplicates) flags 16 SQuAD units against eval_behavior
    prompts, and they are excluded.
  - R3 (no shared rule family), R4 (no evaluation-only generator), R7 (ARC)
    and R9 (mechanism) pass.
  - R0 and R5 are the historical-pool notes the audit explains.
- 4,810 units remain eligible.

**How it enters each operator** (the declared reading).

| operator | what the calibration feeds | effective weight of the mix |
| --- | --- | --- |
| FFN `activation_importance_v0` | E[\|a\|] over scored positions | realized positions: 50/50 |
| RESIDUAL_WIDTH `global_pca_v0` | residual second moments over scored positions | realized positions: 50/50 |
| ATTENTION `activation_importance_v1` | per-head write energy over scored positions | realized positions: 50/50 |
| DEPTH `causal_kl_greedy_v1` | per-item KL → subtype mean → domain mean (`equal_subtype_then_equal_domain/v1`) | **labels**: domain = half, so 1/2 each; each fluid axis 1/10 |

With D4's labels (domain = axis), DEPTH would weight supporting 1/6 whatever
the positions say. The half labels are what make the mix 50/50 in DEPTH.

**Sequence sharing.** Items stay one per forward row.
- Under `length_sorted_v1`, items of different categories may share a
  micro-batch of 3, each padded and reduced separately.
- That breaks no task or episode identity, because an episode is selected
  whole.
- No sequence packing is introduced: without document masking it would change
  every operator's activations.

## 4. Feasibility and coverage audit, 2× and 4×

| | 2× | 4× | base_d4 | rh_d4 |
| --- | ---: | ---: | ---: | ---: |
| nominal UESP (fluid + supporting) | 90,000 | 180,000 | 180,000 | 180,000 |
| realized UESP (= scored positions) | 141,617 | 268,173 | 203,726 | 215,935 |
| fluid / supporting share | 50.3 / 49.7 | 50.1 / 49.9 | 71.1 / 28.9 | 87.3 / 12.7 |
| static fluid UESP | 57,011 | 93,198 | 144,890 | 188,503 |
| interactive UESP | 14,205 | 41,203 | 0 | 0 |
| supporting UESP | 70,401 | 133,772 | 58,836 | 27,432 |
| prediction positions | 169,723 | 320,616 | 228,152 | 236,591 |
| forward tokens | 169,808 | 320,769 | 228,237 | 236,666 |
| padded forward slots (micro-batch 3) | 178,117 | 328,533 | 234,740 | 245,169 |
| items / units | 85 / 81 | 153 / 145 | 85 / 85 | 75 / 75 |
| complete episodes | 1 | 2 | 0 | 0 |
| rule families (of 35 calibration families) | 13 | 20 | 21 | 24 |

Realized exceeds nominal because units are whole and every cell takes at least
one. Historical d1 realized 68,756 against the same 45,000 nominal.

**Per axis** (UESP / units / rule families):

| axis | 2× | 4× | base_d4 | eval-set-1 items / held-out families |
| --- | --- | --- | --- | --- |
| task-contained deduction | 15,005 / 3 / 3 | 23,476 / 6 / 4 | 38,788 / 10 / 4 | 192 / 13 |
| inductive rule discovery | 13,437 / 4 / 4 | 22,861 / 7 / 7 | 37,121 / 11 / 9 | 144 / 30 |
| compositional generalization | 15,618 / 4 / 3 | 25,987 / 6 / 4 | 36,078 / 11 / 4 | 144 / 16 |
| abstract analogical | 12,951 / 2 / 2 | 20,874 / 3 / 3 | 32,903 / 6 / 4 | 96 / 17 |
| interactive adaptation | 14,205 / 1 episode, 5 turns / 1 | 41,203 / 2 episodes / 2 | none | 144 / 10 |
| supporting (gsm8k / multihop / openmath / rag) | 18,092 / 17,910 / 16,574 / 17,825 | 32,166 / 33,568 / 34,294 / 33,744 | 58,836 | not fluid |

**Interactive capacity is limited by supply, not by dose.**
- Every existing teacher-native episode, H1-PA's included, comes from the same
  3 calibration rule families: `act.reverse`, `probe.digit_sum` and
  `probe.threshold`.
- The 13 clean episodes run 8.3k–27.0k scored positions each. One episode
  already exceeds the 2× interactive quota of 9,000.
- 2× carries one complete episode (`act.reverse`) and 4× two (`act.reverse`,
  `probe.threshold`).
- Covering all 3 families would make interactive more than 40% of a 2× fluid
  half. More interactive coverage needs either:
  - new teacher episodes, which are not authorized; or
  - D4-Augmented's 28 trained episodes, which would break the
    calibration/training isolation.

The calibration families are disjoint from eval-set-1's held-out families by
construction (R3). Calibration family coverage therefore measures how varied
the activation statistics are, not overlap with the evaluation.

## 5. The search

**Declared search-space change.**
- The forward baseline's free variable "calibration profile per operator"
  has one value.
- The beam tree is 4 + 12 + 12 + 6 = **34 expansions**, against the
  historical two-profile 8 + 48 + 24 + 12 = 92.
- That count matches the telemetry of all three historical searches.

**Unchanged:**
- operator order (all 24 orders reachable);
- `beam.delayed_prune@v1` (width 6, warmup 1);
- `beam.pareto_multi_objective@v2` and `state_eval@v1`;
- `positions.supervised_target_v1@v1`;
- `reference_topk_tail_v1` (K = 200);
- micro-batch 3, `length_sorted_v1` and `scored_v1`;
- D4's bit-identical execution options (`depth_prefix_reuse`,
  `validate_on_search_device`, `prune_weights=False`).

No other dimension is reduced.

**Compute, from the Stage-0 telemetry.**
- The basis is 276 expansions across d1, d2 and d4. The model reproduces each
  search's telemetry sum to 0.98–1.03×.
- **DEPTH** is token-linear: 226.8, 176.5 and 148.4 minutes per million
  forward tokens at levels 1, 2 and 3.
- **Other operators:** FFN, RESIDUAL_WIDTH and ATTENTION take about 49 + 80 s/Mtok
  (FFN), 38 + 80 s/Mtok (RESIDUAL_WIDTH) and 47 + 110 s/Mtok (ATTENTION) per
  expansion.
- **Overhead:** 14.3 s per expansion.

| | 2× | 4× |
| --- | ---: | ---: |
| DEPTH, one expansion at level 1 / 2 / 3 | 38.5 / 30.0 / 25.2 min | 72.7 / 56.6 / 47.6 min |
| FFN / WIDTH / ATTENTION, one expansion | 63 / 52 / 66 s | 75 / 64 / 82 s |
| expected DEPTH expansions (historical placement) | 4.33 | 4.33 |
| expected search session | 198 min, $5.99 | 326 min, $9.88 |
| bound: DEPTH in all 16 positions it can take | 482 min, $14.59 | 870 min, $26.34 |
| search state storage (34 states) | about 96 GB | about 96 GB |

All figures are on an A100-SXM4-80GB at $1.79/h, with container disk.

## 6. 2× against 4×

**Is 2× enough?**
- **For the question asked, yes.**
  - Every axis, all five, is in every operator's statistics.
  - DEPTH weights the halves 1/2 each.
  - Every static axis has 2–4 units and 2–4 families.
- **Its gaps, stated:**
  - 39% of `base_d4`'s static fluid positions;
  - 12 of 32 static calibration families;
  - one interactive family of three.
- **4× narrows these gaps but does not close them:**
  - 64% of the static fluid positions;
  - 18 static families;
  - two interactive families.
- **Stage-0 pilot evidence on volume** (root level):
  - FFN per-layer Jaccard 0.909 at 1× and 0.944 at 4×;
  - RESIDUAL_WIDTH subspace overlap 0.839 at 1× and 0.896 at 4×;
  - ATTENTION: no volume effect.

  So 2× sits inside the measured range.

**Does 4× have a scientific reason?**
- One: volume. 2×'s realized volume is 0.70× of `base_d4`'s, and 4× is 1.32×.
- A 2× arm that recovers *worse* than D4-Augmented confounds the mixed
  composition with less calibration; 4× separates them.
- A lower state-evaluation KL, more families by themselves, or 4×'s
  affordability are **not** reasons.

**Entry criterion (in the application).** Enter 4× only if the decided
comparison M is:
- (a) **GO**, a dose follow-up; or
- (b) **REVERSED on deduction**, the one axis where D4-Augmented is off the
  floor at 13.5%.

NO_GO, INCONCLUSIVE and NOT_RESOLVED do not enter 4×. An interval that crosses
a SESOI boundary is a power limit, not a volume question.

## 7. Control protocol against D4-Augmented

Owner:
[`plans/mixed_preregistration.json`](../../../../logs/stages/stage-0/calibration_volume/plans/mixed_preregistration.json).
It is registered before any search, Top-1 or recovery exists, and its rule
files are bound by sha256.

**The isolated variable is the initialization the recovery starts from.**
Mix-Augmented's recovery is D4-Augmented's:
- 5,923,021 supervised tokens;
- batch size 2, micro-batch 1, 3 epochs, 3,702 optimizer steps;
- CE 0.25 + KD 1.0, AdamW lr 5e-5, cosine schedule;
- the PCA init's tokenizer and the teacher at `768f209d`;
- seeds 20260726 and 20260801;
- an L40S.

The only overrides are `student_path` and the run identity. Inside the
initialization, composition, volume, interactive data, profile choice and
DEPTH weighting change together. M is never read as any one of them.

**The comparison M = Mix-Augmented − D4-Augmented** is decided. H1's frozen
rule is applied unchanged (`mixed_statistics.py` → `h1_statistics.decide`):
- **Rows:** formal static fluid items, clustered by rule family, both seeds
  as two observations per item.
- **Interval:** 95% cluster bootstrap.
- **SESOI:** 5 pp overall and 8 pp per axis, with a Holm family over the four
  static axes.
- **Floor and resolution:** the 2% floor; resolution needs ≥ 10 discordant
  pairs in ≥ 5 clusters.
- **Seed agreement:** every directional label needs both seeds.
- **E** = Mix-Augmented − PCA-Original is descriptive only.

**Which labels can occur.** D4-Augmented sits at 4.56%.
- **REVERSED** cannot occur overall.
- **GO** needs about 11% or more.
- **NO_GO** needs Mix-Augmented at or above 2% and the interval inside ±5 pp.
- On deduction, every label is reachable.

**Evaluation.** The evaluation is H1-PA's, unchanged: eval-set-1 at
`62a1a5d3`, validity mask v1, scorer v2, `nucleus_v1`, and the same per-item
seeds. The 144 episodes are exploratory.

**Reported beside the decision, every capability on its own:**
- each axis, split into unseen rule and unseen format;
- the interactive episodes;
- every eval-set-1 unit, correct **and** usable side by side (the
  `h1pa_fullset` layout): Knowledge, Multihop, RAG, Instruction, Safety and
  the rest;
- natural termination, degeneration stops and context-limit hits;
- step-0 teacher KL, diagnostic only:
  - the recovery's step-0 `val_kd` / `val_ce` (D4-Augmented s20260726:
    6.966 / 7.205);
  - the initialization's `state_eval@v1` KL.

## 8. Checkpoints and the network volume

Volume `am5m373kuq` (30 GB, EUR-IS-2, $2.10/month, reserved to 2026-12-24
under H1-PA) holds the four D4 checkpoints (9.54 GB). Owner:
[`storage/h1pa_volume_am5m373kuq.json`](../../../../logs/stages/stage-2/reasoning_data/storage/h1pa_volume_am5m373kuq.json),
`declared_future_use`.

- **D4-Augmented × 2: the formal control.**
  - The evidence is reused, never regenerated.
  - The weights are retained as the control anchor until this study's
    analysis and review are complete.
  - Nothing in the design reads the weights (P8.4: evidence, not bytes). They
    are kept for a later authorized consumer, which reads them in place.
- **D4-Original × 2:** retained for H1-PA's independent review.
- **MC2 would write its two new checkpoints here.** That is a peak of about
  24.0 GB of the 30 GB and 14.8 GB after. Treatment and control then share one
  datacenter and GPU class.
- **Retention and cost.**
  - Keep everything in place through both reviews.
  - If a review is still open on 2026-12-24, book one more month ($2.10)
    before that date, by maintainer decision.
  - Retire under AGENTS.md 2.5 only when no declared consumer remains.
- **Migration** (only on a maintainer decision):
  - the destination's capacity is read from its quota endpoint first (the
    relay accepted about 2 GB on 2026-10-22, and the dev box has 3.8 GB free);
  - one pod in EUR-IS-2 does the copy;
  - sha256 is checked before and after;
  - a migration record is written before the volume is deleted.
- **Recommendation: no migration now.**

## 9. Budget application and authorization boundary

Owner:
[`plans/mixed_application.json`](../../../../logs/stages/stage-0/calibration_volume/plans/mixed_application.json).

| | expected | hard cap |
| --- | ---: | ---: |
| MC1: 2× search, A100-80GB, new ≤ 150 GB volume | $5.99 + $1.05 volume | $15.00 + $1.05 |
| MC2: two recoveries + full eval-set-1, L40S, EUR-IS-2, `am5m373kuq` | $10.91 (590 min) | $14.00 |
| engineering reserve (pre-science subruns, P12.1) | — | $3.00 |
| **total** | **$17.95** | **$33.05** |

- **Project:** $706.8547 committed of $731.00, so $24.15 is left.
- **The request:** a $33.05 phase envelope and the project cap raised to
  **$740**.
- **Prices:** re-read live at the grant. Every cap is recomputed by the same
  formula if a rate is higher.
- **Order:**
  1. MC1;
  2. MC2, only after the Top-1 is durable and identified.

  Only one billing resource exists at a time.

**A grant would authorize:**
- MC1: one A100-80GB pod and one volume of at most 150 GB, deleted after the
  Top-1 and the evidence are durable;
- MC2: one L40S pod with `am5m373kuq`;
- the relay transfer of the 1.2 GB Top-1;
- ordinary engineering repair inside the envelope.

**A grant would never authorize:**
- a 4× search or any re-draw of the profile;
- a ratio search;
- new teacher data;
- retraining or re-evaluating any H1-PA arm;
- extra arms or seeds;
- Stage 4/5 work;
- a merge to `main`;
- a cap increase;
- deleting or migrating the D4 checkpoints.

**Engineering owed before the first paid action ($0, inside the grant):**
1. A search-only session for the `2x` treatment. It reuses `search_session`,
   plus dose_driver's search and export stages without its own recovery.
2. A Mix-Augmented arm for the H1-PA driver that changes no H1-PA identity.
3. The analysis: `mixed_statistics.decide` and the full eval-set-1 table.
4. The governance chain, and a $0 dry run of each launcher's success
   closeout.

## 10. Later: training data scale (not authorized)

- **The ladder stays** 5.92M → 11.84M → about 25M → 50M supervised tokens.
- **Variables stay independent:**
  - data quantity;
  - mixture;
  - task quality and trajectory selection;
  - initialization;
  - recovery compute (steps, FLOPs).
- **Cost scales with tokens.** At D4-Augmented's measured rate (225 min of
  L40S training at 5.92M), 50M would take roughly 32 h of training per seed.
- **Data supply.** Every rung above 5.92M needs new teacher data.
- **Nothing is generated now.** This round is calibration only.

## 11. Reproduce ($0)

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/build_mixed_profiles.py \
    --design configs/stages/stage-0/calibration_volume/mixed_design.json
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q scripts/stages/stage-0/calibration_volume/tests/test_mixed_design.py
```

The builder:
- rewrites the profile items under
  `artifacts/stages/stage-0/calibration_volume/profiles_mixed_v1/` (local
  bytes, not tracked);
- rewrites the profiles document and the audit.

It is deterministic given the pool, corpus_v2, the H1-PA packs and eval-set-1.
