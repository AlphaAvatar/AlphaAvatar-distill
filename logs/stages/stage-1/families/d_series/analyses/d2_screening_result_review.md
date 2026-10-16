# D2 behavioural screening: result review (2026-10-16 UTC)

The frozen screening rung of `d_series_shared_eval@v1` ran under the
maintainer's conditional $25.00 cumulative grant (Final Independent Review of
`8e8ae849`). It stops here, at the confirmation authorization boundary.

Owners of every number below:
- the run: `phase_d2/runs/d2_screening_20261015_203346/closeout/outcome.json`;
- the ranking: `…/evidence/d2_screening_ranking.json`;
- the diagnostics: `…/evidence/d2_screening_diagnostics.json`;
- the programme: `families/d_series/plans/d_series_programme_budget.json`.

## 1. The eight probes: training, generation and scoring evidence

- **Session.** Pod `7stah5j2s98e69` (AU, L40S, $1.09/h), 2026-10-15 21:40 to
  2026-10-16 08:59 UTC (679.0 min). Driver status `COMPLETE`; runner terminal
  `ALL_DONE`.
- **Probes.** All 8 (c1–c4 × seeds 516448251, 614923639) trained under
  `e1_p1_kd_heavy@0.86M` (about 61 min each). Each was then generated through
  the token-id wrapper (no fallback), admitted under the shared rung's
  evaluation protocol, scored by the pinned scorer, and rescored as a
  diagnostic.
- **Durability.** Every probe's result, rows, admission and generations, its
  run manifest and completion record, and its trained checkpoint were secured
  off-pod and verified before the pod could release it.
  - The checkpoint was 2.38 GB per probe, identity-verified against the
    driver's digest.
  - The preserved checkpoints were released only at the successful terminal,
    after the training records were verified beside the evidence.
- **Off-pod verification.** Every unit passes the strengthened gate
  (`rank_d2_screening.verify_unit`):
  - its identity is the frozen unit's (label, seed, arm, initialization);
  - its admission and generation fingerprint match;
  - its rows and generations match by the shared protocol's checks;
  - the pinned scorer, re-run on the stored generations, reproduces every
    figure;
  - its training records are the protocol's recovery of the committed
    candidate at the frozen seed;
  - its rows are aligned with B's.

## 2. Candidate vs B: the frozen ranking

Pooled seed-mean `correct_overall`; B is the two imported units.

| Arm | correct | Δ vs B | per-seed Δ | usable | Δ usable (pooled) | veto |
|---|---|---|---|---|---|---|
| **c4** | 0.02765 | **+0.0059** | +0.0035, +0.0083 | 0.679 | +0.092 | no |
| c3 | 0.02705 | +0.0053 | +0.0082, +0.0024 | 0.691 | +0.104 | no |
| c2 | 0.02645 | +0.0047 | −0.0012, +0.0106 | 0.676 | +0.090 | no |
| c1 | 0.01885 | −0.0029 | −0.0047, −0.0011 | 0.662 | +0.075 | no |
| B | 0.02175 | — | — | 0.587 | — | — |

**Paired per-sample correctness vs B**, as candidate-only / B-only / both, of
850 scorable prompts per seed:

| Arm | seed 516448251 | seed 614923639 |
|---|---|---|
| c4 | 23 / 20 / 2 | 19 / 12 / 3 |
| c3 | 27 / 20 / 2 | 17 / 15 / 0 |
| c2 | 18 / 19 / 3 | 20 / 11 / 4 |
| c1 | 16 / 20 / 2 | 13 / 14 / 1 |

q2's rows are descriptive only (winner's curse): pooled 0.030.

## 3. The selection

**`ONE_CANDIDATE_ADVANCES`: c4** (`3a4173e4…`, FFN(dom) × ATTN(dom), quality
position 4 of 4).
- It leads the runner-up c3 by 0.0006; no tie was broken.
- No candidate was vetoed.
- The rule is the frozen one: exactly one candidate advances, or none.

This is a nomination for the shared confirmation, never a promotion.

## 4. Runtime and cross-session comparability

| Identity | This session | Shared protocol (B) |
|---|---|---|
| host image | `…torch291-ubuntu2404@595.91.07` | the same |
| training runtime digest | `9bd30885…` (torch 2.11.0+cu128, transformers 5.13.1, CUDA 12.8) | the same |
| attested evaluation protocol | `e4750826…` | the same |
| generation fingerprint | `ceac0947…` | the same |
| scoring digest | `33b8ff5e…` | the same |
| arms on the pod | all four identities verified | — |

The protocol's driver lives on AU hosts. Attempt 2's five unconstrained draws
landed in SE and US on 580.x and were refused. The record is
`d2_screening_20261015_193502/evidence/host_location_driver_history.json`
(AU 9 of 10 on 595.91.07).

## 5. Diagnostics (descriptive only; never read by the rule)

- **Token-id sidecar.** 950 / 950 rows aligned and exact on every probe; no
  wrapper fallback.
- **Rollout rescoring.** 950 / 950 samples per probe, all on exact engine ids,
  0 retokenized approximations, complete. The mean teacher–student KL over
  positions is 1.19–1.28.
- **Rollout profile** (from the stored generations):

| Units | natural termination | degeneration | mean tokens | context-limit hits |
|---|---|---|---|---|
| D2 c1–c4 (8 probes) | 0.77–0.84 | 0.16–0.23 | 468–544 | 0 |
| B (2 units) | 0.64–0.76 | 0.24–0.36 | 596–640 | 0 |
| q2 (2 units) | 0.80–0.82 | 0.18–0.20 | 532–590 | 0 |

## 6. Durable evidence, teardown, cost

- **Durable store.** `/home/ecs-user/aad-artifacts/phase_d2/d2_screening_20261015_203346/store`
  holds every probe's rows, generations and rescoring rows, with their sha256
  in the closeout. The small evidence is committed in the run directory.
- **Teardown.** The teardown gate allowed it with no emergency. The provider
  confirms the pod gone and not billing. Observers closed in the terminal
  path.
- **Cost, append-only per attempt:**

| Attempt | Outcome | Booked all-in |
|---|---|---|
| `…183412` | no capacity | $0 |
| `…183734` | five SE/US hosts refused | $0.4518 |
| `…193502` | no AU capacity | $0 |
| `…203346` | COMPLETE | **$12.5212** (GPU $12.3351 + disk $0.1860; provider debit $12.4911) |
| **rung** | | **$12.9730 of the $25.00 envelope** ($12.0270 unspent) |

## 7. Budget and programme reservation (re-derived)

- **Project.** $105.4754 remaining of $595.
- **Allowances.** Formal remaining $109.4530; package remaining $112.6334;
  per-session cap $30.
- **The rung leaves the plan.** The completed screening leaves the programme
  plan only through the contract `families/d_series/screening_completion.py`.
  Seven clauses; exactly one attempt satisfies them, and every other attempt's
  cost is known.
- **What D2 still holds.** Only the shared confirmation: $8.51 at hard with
  reserve.
- **After D2.** $96.97 remains against $66.45 for D3 plus a fresh
  confirmation. The reservation holds with $30.52 to spare. The D3 and
  fresh-confirmation reservation was not drawn on.

## 8. Scientific recommendation on the shared confirmation

> **Added 2026-10-16 (UTC).** This section is the EXECUTOR's recommendation,
> not a maintainer decision. The maintainer has since decided to run c4's
> shared confirmation (`decisions.md` 2026-10-16, $12.00 cumulative). The
> screening evidence and the frozen ranking are unchanged.

**Recommendation: do not fund the shared confirmation of c4.** It is the
maintainer's decision, and nothing here starts it.

1. **SESOI.** The registered SHARED_GO needs a confirmation delta ≥ the SESOI
   (0.01) with a one-sided 97.5% LCB > 0. c4's screening estimate is +0.0059,
   already below the SESOI before any shrinkage for having been selected.
2. **Winner's curse.** That estimate is the maximum of four near-sibling
   estimates, the top three of which lie within 0.0012 of each other: c4
   leads c3 by 0.0006, about half a prompt per seed.
3. **Precedent.** On these same rows, D1's q2 screened at +0.00825 vs B and
   confirmed at −0.0004 (NO-GO).
4. **Thin paired evidence.** Net +3 and +7 correct prompts of 850 per seed.
5. **Expected outcome.** INCONCLUSIVE or NO_GO, at about $8.51 hard, spending
   D2's α = 0.025 of the D2/D3 family.

**What the data do show** (descriptive, not a registered endpoint):
- Under identical recovery, all four D2 initializations raise
  `usable_rollout` over B by +7.5 to +10.4 points, positive on all 8 probes.
- They terminate naturally more often and degenerate less.
- That is the Stage 2/3 primary behaviour objective in AGENTS.md. Here,
  however, usable rollout is a veto-only guardrail and cannot be promoted to
  an endpoint after the fact.
- Correctness sits at 2–3% for every arm, B included, at the 0.86M-token
  recovery budget. That limits how well the registered endpoint can separate
  initializations.

Both are inputs to a future preregistration (for D3 or a new rung) chosen
before its data, never a re-reading of this one.

**Not done, and not authorized:** the shared confirmation; D3; a fresh
confirmation; any promotion; a cap change.
