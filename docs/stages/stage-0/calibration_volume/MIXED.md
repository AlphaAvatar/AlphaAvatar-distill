# Mixed calibration 2×: design recommendation and execution application

**Status (2026-10-26): GRANTED and EXECUTING.** The maintainer granted MC1 + MC2 with a cumulative $33.05 and the
project cap $731 → $740, 2× only ([grant](../../../../logs/budget/approvals/mixed_calibration_grant.json)). Every
paid session is issued by `mixed_governance.py` against the provider's bill read at that moment (section 12). MC1 (the
search) is running; MC2 follows the Top-1's read-back. 4× is outside this round, and nothing starts it.

This page owns the reading. The numbers are owned by the records it cites:
- design: [`mixed_design_v2.json`](../../../../configs/stages/stage-0/calibration_volume/mixed_design_v2.json);
- profile: [`profiles_mixed_v2.json`](../../../../configs/stages/stage-0/calibration_volume/profiles_mixed_v2.json)
  (`calib.s0m_mix_2x@v2`, content `9a6c2782…`);
- audit (balance, coverage, isolation, compute):
  [`design/mixed_feasibility_v2.json`](../../../../logs/stages/stage-0/calibration_volume/design/mixed_feasibility_v2.json);
- control protocol (version 2):
  [`plans/mixed_preregistration.json`](../../../../logs/stages/stage-0/calibration_volume/plans/mixed_preregistration.json);
- application (version 2):
  [`plans/mixed_application.json`](../../../../logs/stages/stage-0/calibration_volume/plans/mixed_application.json).

**Superseded design v1** (commit `6e2887ca`), kept byte-unchanged and pinned
by sha256 in the v2 design's `supersedes`:
[`mixed_design.json`](../../../../configs/stages/stage-0/calibration_volume/mixed_design.json),
[`profiles_mixed.json`](../../../../configs/stages/stage-0/calibration_volume/profiles_mixed.json)
(`calib.s0m_mix_2x@v1`, `calib.s0m_mix_4x@v1`) and
[`design/mixed_feasibility.json`](../../../../logs/stages/stage-0/calibration_volume/design/mixed_feasibility.json).
They are never searched and never a calibration input.

## 1. Recommendation

- **Run 2× as the minimal formal design.**
  - One formal D1 full search on one mixed profile, `calib.s0m_mix_2x@v2`:
    **71,216 fluid = 71,216 supporting** scored positions, exactly.
  - Its Top-1 is recovered and evaluated exactly as D4-Augmented, on both
    seeds.
  - The result is decided against D4-Augmented's existing two units, which
    are reused rather than retrained or re-evaluated.
- **4× is not part of this round:** no budget, authorization, profile,
  resource or automatic trigger (section 6).
- **Money.**
  - Expected cost: $17.97.
  - Hard-capped envelope: $33.05, unchanged from v1.
  - Provider-verified committed spend at 2026-10-26T09:28Z: $706.85, so the
    envelope needs a project cap of at least $739.91; the application asks
    $740, provisionally.

## 1a. What changed from v1, and why

Maintainer review of `6e2887ca` (2026-10-26):

| | v1 (superseded) | v2 |
| --- | --- | --- |
| fluid / supporting scored positions | 71,216 / 70,401 (50.29 : 49.71) | **71,216 / 71,216** |
| supporting stopping rule | each cell to the prefix nearest its quota (within half a unit per cell) | each cell to the prefix that fits its integer quota, then the residual closed **exactly** by the fewest whole units |
| profiles built | 2× and a nested 4× | 2× only |
| 4× in the application | an entry criterion and a cost | absent |

- **Unchanged:** the pool, the exclusions, the draw (seed 179444901, the same
  seed-keyed orders), the fluid half item for item, the labels, the search,
  the recovery, the evaluation and the decision rule.
- Only the supporting half moved: 64 of its 66 units are v1's prefix units;
  2 units close the residual. v1's three nearest-stop extra units are gone.

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
from v1's declared string and never re-drawn).

- **Fluid quotas.**
  - The fluid half's nominal size is split equally over the five axes.
  - A static axis splits its share equally over its generators.
  - The interactive axis is **one** cell over both generators, so the
    at-least-one rule does not double it. An episode is taken whole, with all
    its turns.
- **Family-interleaved order.** Within a cell, the rule families are
  interleaved in seed-keyed order, so n units span min(n, families) families.
  Nothing reads a length or a result.
- **Supporting: exact balance** (declared in the design before the build).
  1. Target T = the fluid half's **realized** scored positions (71,216).
  2. Four integer cell quotas: T // 4 each, the remainder one position each to
     the first cells (here 17,804 each).
  3. Each cell takes whole units from the front of its seed-keyed order while
     the next unit still fits, and stops at the first that does not. Nothing is
     skipped.
  4. The residual R = T − prefixes is closed by the **fewest** whole units
     (at most 3) whose positions sum to exactly R. Candidates come from the
     cells' remaining orders, interleaved rank by rank. Among the exact sets of
     the fewest units, the one whose last member comes earliest wins, then the
     one whose earlier members come earliest.
  5. If no such set exists, the build writes the closest gaps one and two
     units reach and stops. No profile is written, and the ratio is never
     relaxed. A residual smaller than every remaining unit is one such case,
     and a test pins it.
- **What the rule reads:** scored positions only. It reads no result, score,
  evaluation correctness or other length. Nothing is cut, repeated, copied or
  padded, and every episode keeps all its turns.
- **Realized:** prefixes 67,690, R = 3,526. No single unit has 3,526
  positions, so 2 units close it: `gsm8k-001581` (1,798) and `hotpot-000422`
  (1,728), each at rank 15 of its cell's remaining order.

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

**How it enters each operator.** The builder counts both halves through the
core's own contracts (`operator_view`) and refuses anything but equality
(`require_exact`):
- token and prediction axes: `weights_for_items` under the baseline's
  `positions.supervised_target_v1@v1`;
- DEPTH: `domain_balanced_score` over `causal_kl_greedy._domain_map`.

The baseline declares no confidence weighting, so every supervised position
weighs the same.

| operator | what the calibration feeds | fluid : supporting, as computed |
| --- | --- | --- |
| FFN `activation_importance_v0` | E[\|a\|] over the token-axis positions | 71,216 : 71,216 |
| RESIDUAL_WIDTH `global_pca_v0` | residual second moments over the token-axis positions | 71,216 : 71,216 |
| ATTENTION `activation_importance_v1` | per-head write energy over the token-axis positions | 71,216 : 71,216 |
| DEPTH `causal_kl_greedy_v1` | KL over the prediction-axis positions (71,216 : 71,216), then per-item KL → subtype mean → domain mean (`equal_subtype_then_equal_domain/v1`) | **1/2 : 1/2** by label; each fluid axis 1/10, each supporting source 1/8 |

With D4's labels (domain = axis), DEPTH would weight supporting 1/6 whatever
the positions say. The half labels are what make DEPTH 50/50, and the positions
are what make the other three 50/50.

**Sequence sharing.** Items stay one per forward row.
- Under `length_sorted_v1`, items of different categories may share a
  micro-batch of 3, each padded and reduced separately.
- That breaks no task or episode identity, because an episode is selected
  whole.
- No sequence packing is introduced: without document masking it would change
  every operator's activations.

## 4. Feasibility and coverage audit, 2×

| | 2× v2 | 2× v1 (superseded) | base_d4 | rh_d4 |
| --- | ---: | ---: | ---: | ---: |
| nominal UESP (fluid + supporting) | 90,000 | 90,000 | 180,000 | 180,000 |
| realized UESP (= scored positions) | **142,432** | 141,617 | 203,726 | 215,935 |
| fluid / supporting UESP | **71,216 / 71,216** | 71,216 / 70,401 | 144,890 / 58,836 | 188,503 / 27,432 |
| static fluid UESP | 57,011 | 57,011 | 144,890 | 188,503 |
| interactive UESP | 14,205 | 14,205 | 0 | 0 |
| prediction positions | 170,817 | 169,723 | 228,152 | 236,591 |
| forward tokens | 170,901 | 169,808 | 228,237 | 236,666 |
| padded forward slots (micro-batch 3) | 177,714 | 178,117 | 234,740 | 245,169 |
| items / units | 84 / 80 | 85 / 81 | 85 / 85 | 75 / 75 |
| complete episodes | 1 | 1 | 0 | 0 |
| rule families (of 35 calibration families) | 13 | 13 | 21 | 24 |

**Nominal against realized.**
- The realized dose is 142,432 against the nominal 90,000: 1.58× the
  nominal, 3.17× the 45,000 reference, and 0.70× `base_d4`.
- All of the excess is the fluid half's whole-unit rule. Every fluid cell takes
  whole units until its quota is met, and at least one.
- The supporting half then matches the realized fluid half exactly.
- Historical d1 realized 68,756 against the same 45,000 nominal.

| fluid cell | quota | realized | units |
| --- | ---: | ---: | ---: |
| deduction: entity_tracking / ordering / knights_knaves | 3,000 each | 3,435 / 6,300 / 5,270 | 1 / 1 / 1 |
| induction: list / string | 4,500 each | 5,189 / 8,248 | 2 / 2 |
| composition: defined_ops / scan_language | 4,500 each | 7,882 / 7,736 | 3 / 1 |
| analogy: grid_transform | 9,000 | 12,951 | 2 |
| interactive (one cell) | 9,000 | 14,205 | 1 episode, 5 turns |

**Per axis** (UESP / units / rule families):

| axis | 2× v2 | base_d4 | eval-set-1 items / held-out families |
| --- | --- | --- | --- |
| task-contained deduction | 15,005 / 3 / 3 | 38,788 / 10 / 4 | 192 / 13 |
| inductive rule discovery | 13,437 / 4 / 4 | 37,121 / 11 / 9 | 144 / 30 |
| compositional generalization | 15,618 / 4 / 3 | 36,078 / 11 / 4 | 144 / 16 |
| abstract analogical | 12,951 / 2 / 2 | 32,903 / 6 / 4 | 96 / 17 |
| interactive adaptation | 14,205 / 1 episode, 5 turns / 1 | none | 144 / 10 |
| supporting: gsm8k / multihop / openmath / rag | 18,730 / 18,272 / 16,574 / 17,640 (18 / 10 / 6 / 32 units) | 58,836 | not fluid |

Within the supporting half, the four sources hold 23.3–26.3% each. DEPTH
weights them 1/8 each regardless; the other operators see these shares.

**Interactive capacity is limited by supply, not by dose.**
- Every existing teacher-native episode, H1-PA's included, comes from the same
  3 calibration rule families: `act.reverse`, `probe.digit_sum` and
  `probe.threshold`.
- The 13 clean episodes run 8.3k–27.0k scored positions each. One episode
  already exceeds the 2× interactive quota of 9,000.
- 2× carries one complete episode (`act.reverse`).
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
- **Other operators** per expansion: 49 + 80 s/Mtok (FFN), 38 + 80 s/Mtok
  (RESIDUAL_WIDTH) and 47 + 110 s/Mtok (ATTENTION).
- **Overhead:** 14.3 s per expansion.

| | 2× v2 (170,901 tokens) | 2× v1 (169,808 tokens) |
| --- | ---: | ---: |
| DEPTH, one expansion at level 1 / 2 / 3 | 38.8 / 30.2 / 25.4 min | 38.5 / 30.0 / 25.2 min |
| FFN / WIDTH / ATTENTION, one expansion | 63 / 52 / 66 s | 63 / 52 / 66 s |
| expected DEPTH expansions (historical placement) | 4.33 | 4.33 |
| expected search session | 198.5 min, $6.01 | 197.6 min, $5.99 |
| bound: DEPTH in all 16 positions it can take | 484.5 min, $14.67 | 481.7 min, $14.59 |
| search state storage (34 states) | about 96 GB | about 96 GB |

All figures are on an A100-SXM4-80GB at $1.79/h, with container disk. The
exact balance added 1,093 forward tokens (+0.64%) and $0.02 to the expected
search.

## 6. 2×, and 4× as a future option only

**Is 2× enough?**
- **For the question asked, yes.**
  - Every axis, all five, is in every operator's statistics.
  - The halves are exactly equal, and DEPTH weights them 1/2 each.
  - Every static axis has 2–4 units and 2–4 families.
- **Its gaps, stated:**
  - 39% of `base_d4`'s static fluid positions;
  - 12 of 32 static calibration families;
  - one interactive family of three.
- **Stage-0 pilot evidence on volume** (root level):
  - FFN per-layer Jaccard 0.909 at 1× and 0.944 at 4×;
  - RESIDUAL_WIDTH subspace overlap 0.839 at 1× and 0.896 at 4×;
  - ATTENTION: no volume effect.

  So 2× sits inside the measured range.

**4× is outside this round** (maintainer decision 2026-10-26).
- It has no budget, authorization or implementation scope, and no automatic
  flow.
- No 4× profile is built, no cost is allocated and no paid resource is
  prepared.
- No preregistered result starts it. Version 1's application carried an entry
  criterion; it is removed.
- Whether to run 4× is the maintainer's decision after the formal 2× result.

**As a research note only:**
- 2×'s realized volume is 0.70× of `base_d4`'s.
- A 2× arm that recovers worse than D4-Augmented would confound the mixed
  composition with less calibration; a larger mixed dose would separate them.
- A lower state-evaluation KL, more families by themselves, or affordability
  are not reasons to run one.
- The superseded v1 audit measured a nested 4× profile (268,173 UESP, two
  episodes, 20 families). That is a record of that audit, not a plan.

## 7. Control protocol against D4-Augmented

Owner:
[`plans/mixed_preregistration.json`](../../../../logs/stages/stage-0/calibration_volume/plans/mixed_preregistration.json).
It is registered before any search, Top-1 or recovery exists, and its rule
files are bound by sha256.
- **Version 2** amends version 1 (`6daab43d…`) before any result exists.
- It binds the arm to `calib.s0m_mix_2x@v2` by content hash.
- It states that no label starts a 4× search or any other follow-up.
- Everything else is unchanged, the decision rule's bound bytes included.

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

## 9. Budget application and authorization boundary (2× only)

Owner:
[`plans/mixed_application.json`](../../../../logs/stages/stage-0/calibration_volume/plans/mixed_application.json)
(version 2).

| | expected | hard cap |
| --- | ---: | ---: |
| MC1: 2× search, A100-80GB, new ≤ 150 GB volume | $6.01 + $1.05 volume (199 min) | $15.00 + $1.05 (bound $14.67) |
| MC2: two recoveries + full eval-set-1, L40S, EUR-IS-2, `am5m373kuq` | $10.91 (590 min) | $14.00 (bound $13.63) |
| engineering reserve (pre-science subruns, P12.1) | — | $3.00 |
| **total** | **$17.97** | **$33.05** |

- **MC2 is unchanged by the correction.** The recovery and evaluation are
  D4-Augmented's fixed 3,702 steps and all of eval-set-1, whatever the
  calibration was.
- **The envelope stays $33.05.** Both corrected bounds fit their caps, and no
  4× cost is in it.

**Project position, from the provider's billing**
([snapshot 2026-10-26T09:28Z](../../../../logs/budget/provider/runpod_billing_2026-10-26T0928Z.json),
read-only):

| | USD |
| --- | ---: |
| provider-billed (pods $700.37 + volumes $2.40) | 702.7656 |
| unposted sessions | 0 |
| `am5m373kuq` reservation, net of its posted 2026-10-25/26 rows | 4.20 − 0.1138 |
| **committed** | **706.8518** |
| + envelope | 739.9018 |
| requested cap / headroom | 740 / 0.0982 |

- **Why the figure is netted.** `h1pa_governance.live_position` counts the
  full $4.20 reservation beside the volume's already-posted rows. Its committed
  figure ($706.9656) therefore grows about $0.07/day with no new spend. Under it
  the envelope would sum to $740.0156, $0.0156 over $740.
- **The issuer will net instead.** It counts a reservation net of the rows the
  provider has posted for that volume (owed engineering, item 4 below).
- **$740 is provisional.** At issuance the issuer re-reads the provider's
  billing and recomputes the position against the corrected 2× costs. If
  committed + envelope exceeds the cap the maintainer sets, it issues nothing.
  The agent never raises a cap or trims the envelope to make it fit.
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
- anything 4×: a search, a profile, a cost or a resource;
- any search or recovery started automatically from a result of this study;
- a second 2× draw, or any re-draw or re-balancing of `calib.s0m_mix_2x@v2`;
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
4. The governance chain. It re-reads the provider's billing at issuance,
   nets a storage reservation of its posted rows, and refuses if committed +
   envelope exceeds the maintainer's cap. Plus a $0 dry run of each launcher's
   success closeout.

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
    --design configs/stages/stage-0/calibration_volume/mixed_design_v2.json
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q scripts/stages/stage-0/calibration_volume/tests/test_mixed_design.py
```

The builder:
- rewrites the profile items under
  `artifacts/stages/stage-0/calibration_volume/profiles_mixed_v2/` (local
  bytes, not tracked; items sha256 `d86a4b4c…`);
- rewrites the profiles document and the audit;
- refuses to write a profile unless both halves are equal on both position
  axes and DEPTH weights them 1/2 each.

It is deterministic given the pool, corpus_v2, the H1-PA packs and eval-set-1.
It was checked under two hash seeds, with byte-identical items. The superseded
v1 profiles are rebuilt only from commit `6e2887ca`.

## 12. Execution (2026-10-26)

**Governance.** `scripts/stages/stage-0/calibration_volume/mixed_governance.py` holds:
- one grant, one cumulative envelope, and one-use authorizations of kind `search` (MC1) or `recovery` (MC2);
- the project position, recomputed from the provider's billing, balance and resources at every issue and every launch.

The position counts:
- everything billed;
- unposted sessions inside a 72 h posting window;
- every retained volume's booked reservation, less what the provider has already posted for that volume.

**How posted volume rows are attributed.** The provider's volume rows carry no volume id. A day's row is netted
against a volume only on a UTC day when it was the account's sole recorded volume, and never beyond size × the hours
it existed that day.

**Why so conservatively.** The 2026-10-23..26 rows show billed hours that the volumes' own lifetimes cannot account for:
- H1's `1980hnh75i` existed 15.4 h and was billed 20 h;
- one 30 GB volume can accrue at most 720 GB-h a day, but the 10-25 row is 870 GB-h;
- the 10-24 row mixes both volumes.

Those excess hours stay in the bill and offset nothing.

**Cap checks.**
- At the first funded issue: committed + the whole remaining envelope ≤ $740.
- Afterwards: committed + that session's cap ≤ $740.

A miss is a HARD BOUNDARY, and no cap is raised. At 10:28Z the committed figure was $706.9114, and with the whole
envelope $739.96.

**MC1** (`mixed_search_driver.py`): `search_driver.py` under the pinned search identity (cuda config hash
`7ee23a7d…`) with D4's execution options, mirrored to the 150 GB volume `t6yrsvujha` (EUR-IS-1). It then commits
k = 1, re-identifies the Top-1 and relays it.
- **Attempt 1** (`mixs_20261026_102914`, $0.1903) died at A_contract before any measurement. Its authorization had
  been issued after the session commit, so the pod's checkout never held it.
  - Repaired by committing the authorization as the session commit.
  - A new $0 precheck, `the_authorization_is_in_the_session_commit`, refuses this at launch.
- **Attempt 2** is `mixs_20261026_103634`.

**MC2** (`mixed_recovery_driver.py`): `h1pa_driver.run` with D4-Augmented copied whole and only the initialization
replaced. Each unit's recipe must hash to D4-Augmented's recorded identity for its seed.

The result is decided by `mixed_analysis.py`, which has already been rehearsed on stand-in evidence.
