# D2 shared confirmation of c4: result review (2026-10-16 UTC)

The maintainer decided to run c4's shared confirmation (`decisions.md`
2026-10-16, $12.00 cumulative). It ran once, completed, and the frozen verdict
is **NO_GO**. The programme stops here, at the next scientific authorization
boundary.

Owners of every number below:
- the run: `phase_d2/runs/d2_confirmation_20261016_105802/closeout/outcome.json`;
- the verdict: `…/evidence/d2_confirmation_verdict.json`;
- the diagnostics: `…/evidence/d2_confirmation_diagnostics.json`;
- the exploratory analysis:
  `…/evidence/d2_confirmation_rescoring_vs_correctness.json`;
- the programme: `families/d_series/plans/d_series_programme_budget.json`.

## 1. The three probes: endpoint evidence

c4 (`3a4173e4…`) was trained from its INITIAL committed checkpoint at the
three registered seeds under `e1_p1_kd_heavy@0.86M` (about 61 min each).
- **Generation and scoring.** Through the token-id wrapper (950/950 rows exact,
  no fallback), admitted under the shared confirmation rung's protocol, and
  scored by its pinned scorer.
- **Verified off-pod.** Every unit passes the same gate as the screening's
  (`rank_d2_screening.verify_unit`, confirmation rung) and is aligned with B's
  rows.
- **The re-score reproduces the pod.** The pinned confirmation scorer, re-run
  on the stored generations, reproduces every figure of each pod-scored result.

| Seed | c4 correct | B correct | Δ | candidate-only / B-only / both | c4 usable | B usable |
|---|---|---|---|---|---|---|
| 396880939 | 0.0329 | 0.0424 | −0.0094 | 17 / 25 / 11 | 0.6505 | 0.6147 |
| 766527341 | 0.0294 | 0.0471 | −0.0176 | 18 / 33 / 7 | 0.6874 | 0.6000 |
| 1004399678 | 0.0271 | 0.0318 | −0.0047 | 18 / 22 / 5 | 0.6463 | 0.5674 |
| **pooled** | **0.0298** | **0.0404** | **−0.0106** | | 0.6614 | 0.5940 |

Paired counts are over the 850 scorable prompts per seed.

## 2. The frozen verdict

This is the registered rule: `correct_overall`, SESOI 0.01, one-sided α 0.025,
and the stratified prompt-cluster bootstrap (20,000 iterations) at the
registered seed 450870042. It requires ≥ 2 of 3 seed deltas positive and
applies the usable-rollout veto.

| Criterion | Value |
|---|---|
| Δ (c4 − B), prompt-mean of seed-mean paired differences | **−0.0106** |
| one-sided 97.5% LCB / UCB | −0.0200 / **−0.0016** |
| two-sided 95% interval | [−0.0212, 0.0000] |
| UCB < SESOI (0.01) | yes → **NO_GO** |
| LCB > 0 | no |
| seed deltas positive | 0 of 3 (required 2) |
| usable guardrail | passed: pooled +0.067; per seed +0.036, +0.087, +0.079 |

**Verdict: NO_GO.** It is terminal for c4 on these rows: no seed is added and
nothing is re-confirmed.
- The upper bound is below zero. On this battery and these recovery seeds,
  c4's correctness is LOWER than B's, not merely not higher.
- The bound is conditional on the three preregistered recovery seeds; it is
  not an interval over hypothetical future seeds.

**Descriptive comparison with q2** (no decision role, no α). c4 − q2 =
−0.0102, with one-sided UCB −0.0016. q2's own confirmation against B was
−0.0004 (D1, NO-GO).

What must be kept apart:
1. **The screening estimate was optimistic.** Screening's +0.0059 was the
   maximum of four near-sibling estimates (winner's curse), measured on
   different seeds and a different battery.
2. **The confirmation decides.** Its frozen correctness endpoint decides D2's
   statistical result, and that result is NO_GO.
3. **Usable rollout is secondary.** Its improvement (+0.067 here, +0.075 to
   +0.104 in screening) is real secondary behavioural evidence. It does not
   replace the correctness endpoint and earns no credit under this protocol.
4. **No single-variable attribution.** The D2-vs-D1 difference cannot be
   attributed to teacher confidence alone. D2 v3 changed the objective, the
   support and the search together, so no single change is isolated.
5. **Even a pass would only nominate.** A SHARED_GO would have nominated c4
   for a fresh independent confirmation, never promoted it. With NO_GO there
   is nothing to nominate.

## 3. Training and generation identity

| Identity | This session | Shared protocol (B's confirmation units) |
|---|---|---|
| host image | `…torch291-ubuntu2404@595.91.07` (AU) | the same |
| training runtime | `9bd30885…` (torch 2.11.0+cu128, transformers 5.13.1) | the same |
| attested evaluation protocol | `c82260e0…` | the same |
| generation fingerprint | `ceac0947…` | the same |
| scoring digest | `812d3361…` (`score_d1_confirmation.py`) | the same |
| recovery | recipe `a4b0dcab…`, data `6f324cb0…`, trainer `a1807e34…` | the same |
| initialization | c4's committed product, verified at all four identities on the pod; restored from the relay at $0 beforehand (`analyses/d2_candidate_relay_restore.json`) | — |

Draw 1 (AU, driver 580.178.04) was refused by host admission. Draw 2 was
admitted.

## 4. Diagnostics (descriptive only; never read by the rule)

- **Token-id sidecar.** 950/950 exact on every probe.
- **Rescoring.** Complete, 0 retokenized approximations. The mean
  teacher–student KL over positions is 1.148–1.265.
- **Rollout profile:**

| Units | natural termination | degeneration | mean tokens | context-limit hits |
|---|---|---|---|---|
| c4 (3 probes) | 0.759–0.836 | 0.164–0.241 | 478–568 | 0 |
| B (3 units) | 0.697–0.766 | 0.234–0.303 | 614–673 | 0 |
| q2 (3 units) | 0.764–0.807 | 0.193–0.236 | 560–583 | 0 |

- **Exploratory: rescoring against correctness** (2,550 scorable samples, 76
  correct; small and exploratory).
  - On correct samples, the teacher–student KL over the REASONING span is
    lower: 1.14 vs 1.42, point-biserial r = −0.13.
  - Whole-rollout KL, surprisal, confidence and entropy are essentially
    uncorrelated with correctness (|r| ≤ 0.03).
  - Answer-span confidence is weakly higher on correct samples (r = +0.07).
  - High-confidence/high-KL positions are about 0.2%, unrelated to
    correctness.
  - B and q2 have no rescoring (retired checkpoints), so nothing here compares
    arms.

## 5. Cost, teardown, budget

- **Session.** $5.0260 all-in: GPU $4.9513 plus disk $0.0747 over 272.5 min,
  including the refused draw. The provider debit was $5.0198.
- **Envelope.** $5.0260 of the $12.00 confirmation envelope is spent.
- **D2's measured total.** The search ($5.5364), the screening rung over four
  attempts ($12.9730) and the confirmation ($5.0260) sum to **$23.5354**.
- **Teardown.**
  - Pod `tvohx8nubg6kmi` was deleted at 15:32:41Z, and the provider confirms
    it gone and not billing.
  - The refused draw's pod was released and confirmed not billing.
  - The watchdog ended on `pod_gone`; the tmux session and waiter ended with
    the launcher. The provider lists 0 pods.
- **Budget** (`derive_budget.py`): project $100.4494 remaining of $595; formal
  $104.4270; package $107.6074; engineering $3.1804; per-session cap $30.
- **Programme.** The confirmation leaves the plan by the rung completion
  contract, and D2 holds nothing further. D3 plus one fresh confirmation needs
  $66.45 at hard, against $100.45 available, leaving **$34.00** spare.

## 6. Manifest and checkpoint retention

- **Session manifest.** 97 final-required files, none missing, streams
  quiescent.
- **Archive.** `d2_confirmation_artifacts.tar.gz` (sha256 `7e7a82e5…`) is in
  the durable store and has a second copy on the relay, restored and
  re-hashed (`evidence/archive_relay_copy.json`). The screening's archive was
  given the same copy.
- **Per-unit evidence.** Rows, generations and rescoring rows are in the
  durable store, with their sha256 in the closeout. The small evidence
  (results, admissions, configs, training records, verdict, diagnostics) is
  in git.
- **Trained checkpoints.** All three (2.38 GB each) were preserved and
  identity-verified off-pod during the run. They were released only at the
  successful terminal, after each probe's training records were verified
  beside its evidence. Their sha256 are recorded.
- **Initial checkpoints.** The four committed D2 initializations remain on the
  dev host and the relay. The checkpoint registry now records no open
  consumer for them: retained until the maintainer's next D-series decision,
  then retirable under the standing policy after a consumer audit, with a
  tombstone.

## 7. Recommendation for the next step (the maintainer's to decide)

- **No fresh independent confirmation for c4.** It is only defined after a
  SHARED_GO, and the NO_GO is terminal.
- **Before D3 ($50.95 at hard), revisit what it will measure.** Two
  D-series candidates (D1's q2, D2's c4) have now each:
  - raised usable rollout over B under identical recovery;
  - failed to raise correctness, which sits at about 2–6% for every arm, B
    included, at the 0.86M-token recovery budget.

  A D3 run under the same endpoint and recovery budget risks the same
  outcome.
- **Options worth a $0 design pass before D3's preregistration is frozen.**
  Each must be decided before any D3 data, never as a re-reading of D1 or D2:
  - a recovery budget large enough for correctness to separate
    initializations;
  - making usable rollout, or its components, a registered co-primary
    endpoint with its own SESOI;
  - a battery or endpoint with more resolution at this correctness level.
- Nothing in this run authorizes D3, a fresh confirmation, a promotion or a
  cap change.
