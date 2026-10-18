# Stage-0 calibration volume: programme report

**Terminal, 2026-10-18. The formal volume study was NOT RUN, for two
independent reasons:**
1. **H2 is not resolvable** under the fixed recovery recipe.
2. **The budget cannot fund the smallest readable formal design.**

The prerequisites and a root-level pilot of H1 ran to completion. Every figure
below is read from
[`programme_closeout.json`](programme_closeout.json), which derives it from the
record that owns it. Where this page and that file disagree, the file is right.

## Verdicts

| hypothesis | verdict | strength |
| --- | --- | --- |
| H1: more calibration volume stabilizes structural selection | FFN and RESIDUAL_WIDTH: **stabilizes with volume**. ATTENTION: **no detectable volume effect**, already near-identical at 1x. DEPTH: unlabeled (one resample pair per dose). | exploratory, root level only |
| H2: downstream reasoning on unseen rule families improves | **EVALUATION NOT RESOLVABLE** under the fixed recovery recipe | no measurement possible |
| H3: the effect saturates | not supported within 1x–4x for FFN/WIDTH, where the gain continues at every doubling; trivially "saturated" for ATTENTION at 1x | exploratory, described only, as the frozen plan says |

## 1. Repository state

- **Branch.** `stage0/calibration-volume-mixture`, cut from `main` at the
  D-series squash `32246d12`.
- **Code.** Experiment code is under
  `scripts/stages/stage-0/calibration_volume/`. The only core addition is
  `src/aadistill/evaluation/isolation.py`:
  - generic key-overlap and disjointness checks;
  - no experiment constants;
  - declared in `tests/architecture/test_cuda_surface_preserved.py`.
- **Instruments.** Explained in
  [`docs/stages/stage-0/calibration_volume/DESIGN.md`](../../../../../docs/stages/stage-0/calibration_volume/DESIGN.md).

## 2. Data audit ($0, [`audit/data_audit.json`](../audit/data_audit.json))

- **`corpus_v2` covers none of the primary axes.**
  - It holds 11,174 accepted sessions in six task families (code, gsm8k,
    multihop_qa, openmath, rag_evidence, tool_calling).
  - It has 0 rule families and 0 items on any primary P10.2 axis.
  - Math and multihop are *mixed* task-contained deduction at best.
- **The D1 calibration profiles.**
  - The two profiles are small: 44,746 and 32,709 UESP.
  - They share 39 of 51 sessions.
  - They draw from the same corpus.
- **D1's length window selects heavily.** `[256, 2048]` admits 14% of code
  sessions, 39% of openmath and 38% of multihop. Teacher-native reasoning on
  procedural tasks is longer still, which is why `base_pool_v1` admits up to
  8,192 tokens.
- **Conclusion.** The existing Stage-0 data could not represent fluid
  reasoning. The programme therefore built its own (section 3).

## 3. Capability-aligned evaluation and calibration data

- **The battery.** Fifteen procedural generators over the five P10.2 axes
  (DESIGN §2), all project-authored, with no ARC-AGI material.
  - Five generators are evaluation-only.
  - Families are split by salted hash.
  - Isolation is enforced by key: family, content and prompt overlap between
    calibration and evaluation are all 0.
- **dev_v1 (720 tasks) on an A100**, greedy and P18-unrestricted
  (`runs/battery_20261018_092345`):
  - **Teacher 638/720 (0.886).**
    - induction 0.972, composition 0.938, deduction 0.984, analogy 0.771,
      interactive 0.694;
    - interactive invalid-action rate 0.0027.
  - **Fixed-recipe students** (0.86M rung, seeds sa and sb): **0/720 and
    0/720**, seed discordance 0, interactive invalid-action rate 0.96 / 0.91.
  - **The 2.96M reference student: 10/720.**
  - The battery is valid for the teacher. It cannot see the students.
- **Hard families for the teacher** (consider for the final battery's
  headroom): `probe.floor_div` 1/12, `grid.gravity_down` 1/6,
  `probe.mod_scale` 3/12. `grid.border`, `grid.count_cells` and
  `probe.quadratic` are each at 0.33.
- **Teacher-native calibration** (`runs/teachercal_20261018_112755`,
  `calibration_v2`, 672 prompts):
  - 519 verified, naturally terminated units: 506 static and 13 interactive
    episodes.
  - Combined with 6,020 verified supporting `corpus_v2` units, these form
    `base_pool_v1` (6,593 items).

## 4. Design: what was frozen and what was proposed

- **The pilot.** Frozen and run.
  - Design: [`pilot_design.json`](../../../../../configs/stages/stage-0/calibration_volume/pilot_design.json).
  - Analysis plan, frozen before any result:
    [`plans/pilot_analysis_plan.json`](../plans/pilot_analysis_plan.json).
  - Profiles: nine, at 1x/2x/4x of 45,000 UESP times three independent
    resamples, nested within a resample.
  - Mixture weights: 0.18 per static primary axis, 0.28 supporting.
  - Interactive is excluded, with the reason recorded (13 episodes).
- **The formal study.** Designed and priced, **not frozen**:
  [`plans/formal_design_proposal.json`](../plans/formal_design_proposal.json).
  - Treatments: one D1 full search per treatment, with operator order and
    profile per operator free, over a base and a reasoning-heavy mixture at
    the treatment's dose.
  - H2 statistics: SESOI 0.05 pooled / 0.08 per axis, a rule-family cluster
    bootstrap, and the GO / NO-GO / INCONCLUSIVE / SATURATION SUPPORTED /
    EVALUATION NOT RESOLVABLE decision rules.
  - Three resamples per dose: the minimum the noise-calibrated H1 reading
    can read.

## 5. Full search results

None was run (see section 10). The pilot measured what a full search on this
pool costs, per the proposal's `pricing.full_search_by_dose`. The figures
scale D2's measured 92-expansion trajectory by this pool's measured root
rates, on an A100-80GB at $1.79/h.

| dose | search minutes | expected | hard |
| --- | --- | --- | --- |
| 1x | 486 | $15.40 | $19.82 |
| 2x | 704 | $21.91 | $27.63 |
| 4x | 1,150 | $35.19 | $43.58 (above the $30 per-session cap) |

DEPTH is 68% (1x) to 85% (4x) of every search. It is token-linear, and the
pool's long teacher-native items make it about 1.5x (1x dose) to 4.4x (4x
dose) D2's root cost.

## 6. Recovery results

None. A recovered student would score 0 by construction (section 3), so no
recovery budget was spent.

## 7. Capability results

Only the dev-battery calibration in section 3. No initialization was compared
on capability.

## 8. The root-level pilot (H1/H3, exploratory)

Source: [`design/pilot_structure_agreement.json`](../design/pilot_structure_agreement.json),
from `runs/pilot_20261018_134131`: 31 of 31 root states on an A100-80GB.

Each value is the mean between-resample agreement over 3 pairs, with the
spread across pairs (max − min) in brackets. The 3 pairs come from the three
independent resamples at that dose.

| operator | 1x | 2x | 4x | chance | 4x − 1x | label |
| --- | --- | --- | --- | --- | --- | --- |
| FFN (per-layer Jaccard) | 0.9085 [0.006] | 0.9289 [0.010] | 0.9435 [0.009] | 0.19 | +0.035 | stabilizes with volume |
| RESIDUAL_WIDTH (subspace overlap) | 0.8391 [0.005] | 0.8645 [0.005] | 0.8956 [0.007] | 0.40 | +0.057 | stabilizes with volume |
| ATTENTION (per-layer Jaccard) | 0.9558 [0.001] | 0.9553 [0.009] | 0.9593 [0.019] | 0.33 | +0.004 | no detectable volume effect |
| DEPTH (removed-set Jaccard, 1 pair) | 0.6 | 1.0 | — | 0.125 | — | none (by design) |

- **DEPTH.**
  - At 1x the two resamples differ in two of eight removed layers:
    {10, 11} vs {8, 16}.
  - At 2x both remove {2, 3, 8, 16, 26, 27, 30, 32}.
  - Three of the four DEPTH runs remove exactly that set: both 2x runs and
    resample 2's 1x run.
- **Saturation, described as the plan says.** Each resample's 1x and 2x
  decisions agree with its own 4x decision:
  - FFN 0.942 / 0.964;
  - WIDTH 0.910 / 0.951;
  - ATTENTION 0.973 / 0.983.

  Two independent 4x resamples agree at FFN 0.944, WIDTH 0.896 and
  ATTENTION 0.959. Nested profiles share units, so nested agreement is an
  upper bound, not a like-for-like comparison.
- **Reference contrast (descriptive).** The historical D1 root decisions sit
  much further from every pilot profile than the resamples sit from each
  other: FFN about 0.72, ATTENTION about 0.80, DEPTH 0.53–0.60. Changing what
  the calibration data *is* moves selection more than resampling it does.
  This is a mixture contrast, not a volume result, and it also crosses
  hardware and prediction-row protocol.
- **Covariate.** Distinct rule families rise with dose (10–12, 15–17,
  18–22). The pilot cannot separate volume from family diversity.
- **Limits.** These are fixed-parent (teacher) decisions, not full-search
  paths. Stability is not quality: whether a more stable selection gives a
  better student is H2, which cannot be measured here.

## 9. Budget

Booked from each run's closeout:

| run | phase | outcome | $ |
| --- | --- | --- | --- |
| battery_20261018_091856 | prerequisites | abandoned before launch | 0 |
| battery_20261018_092345 | prerequisites | battery scored; teacher_cal unit stopped deliberately | 3.6303 |
| teachercal_20261018_112755 | prerequisites | ALL_DONE | 1.6169 |
| pilot_20261018_122600 | pilot | abandoned before launch | 0 |
| pilot_20261018_122725 | pilot | abandoned before launch | 0 |
| pilot_20261018_122808 | pilot | L40S CUDA OOM after 10/31 states (repaired) | 1.2469 |
| pilot_20261018_134131 | pilot | ALL_DONE, 31/31 states | 6.9530 |
| **programme** | | | **13.4471 of 92.00** |

- **Project.** $513.5943 of the $595 cap is spent, leaving $81.4057.
- **Provider at close.** 0 pods, 0 network volumes, $0/h.
- **No D-series grant or allowance was used.**
- **Grant amendment.** $5 moved from `formal_volume` to `pilot` inside the
  unchanged $92 ceiling (`decisions.md`).

## 10. Scientific verdict and what it would take

**The formal study is not run, and running any reduced version would buy an
uninterpretable result.**

- **H2: EVALUATION NOT RESOLVABLE.**
  - Under `e1_p1_kd_heavy@0.86M` both controls score 0/720 and never
    disagree, so there is no minimal detectable effect.
  - Even the 2.96M rung reaches only 10/720.
  - The maintainer's brief says to fix the evaluation before spending recovery
    budget.
  - The fix is a change to the recovery recipe, which AGENTS.md §4.5 reserves
    for the maintainer.
- **Budget: INSUFFICIENT.**

  | design | full searches | probes | expected | gap to the formal phase ($65) |
  | --- | --- | --- | --- | --- |
  | H1 only (1x, 2x) | 6 | 0 | $111.93 | $46.93 |
  | H1 + H2 (1x, 2x) | 6 | 12 | $132.09 | $67.09 |
  | the brief's full study (1x, 2x, 4x) | 9 | 18 | $247.74 | $182.74 |

  - The H1 + H2 design additionally needs a new recovery recipe.
  - The full study's 4x session would also exceed the $30 per-session cap.
  - Even the H1-only design exceeds the whole programme remainder ($78.55).

**The minimum viable plan**, in order. Each step is a maintainer decision:

1. **Make H2 measurable first.** Choose a recovery recipe that transfers
   task-contained reasoning, e.g. verified teacher-native traces on
   calibration-split families, disjoint from every evaluation family and from
   the calibration profiles, under P17.
   - Validate it with one probe of about $2: the control must score above
     floor on dev_v1, and two seeds must disagree on enough tasks to give a
     minimal detectable effect at or below the 0.05 SESOI.
2. **Cut the search's fixed cost.** Investigate the ~1.25 min per-expansion
   overhead (D2: 0.14). It is suspected to be host-side validation sized
   from the host's CPU count rather than the pod's grant. Fixing it saves
   about 115 min per full search.
3. **Then fund MVP-B** (1x/2x, three resamples, two recovery seeds): about
   $132 at today's A100 price, or about $112 for H1 alone. Both are above
   what remains in the project.
4. **If only H1 matters:** the pilot already answers the local question (FFN
   and WIDTH stabilize, ATTENTION does not need more data). A full-search H1
   adds whether path-level choices stabilize too.

## Artifacts

- **Heavy bytes**, gitignored and named by sha256 in each run's closeout:
  - raw generations;
  - `base_pool_v1`;
  - profile items;
  - the pilot's `structure.json`, state journals and nine WIDTH projections.

  They live under `artifacts/stages/stage-0/calibration_volume/`.
- **Relay copies** of the generation records: `relay_copies.json`.
- **D1 reference structure** (built from D1's own journal, $0):
  `/home/ecs-user/aad-artifacts/stage0_calvol_reference/d1_structure.json`,
  sha256 `0e6babb7…`.
- **Checkpoints.** None were produced, and none was duplicated.
