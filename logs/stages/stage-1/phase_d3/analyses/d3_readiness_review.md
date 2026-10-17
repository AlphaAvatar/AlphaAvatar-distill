# D3 readiness review: fixed-q2 DEPTH teacher-reference ablation (2026-10-16 UTC)

This review covers the $0 design, implementation and readiness for the new D3,
under the maintainer decision of 2026-10-16. It stops at the paid
authorization boundary: no grant is issued and no resource is created.

Owners of the numbers below:
- the design: `plans/d3_registration_v1.json`
  (registration `43156f21…`);
- the control evidence: `plans/d3_control_q2.json`;
- the price: `plans/d3_budget_v1.json`;
- the proposed grants: `plans/d3_grant_proposal.json`;
- the decision: `logs/budget/decisions.md`, 2026-10-16.

## 1. The scientific design and the treatment identity

**Question.** Start from D1-q2's exact structural path. Does choosing the
DEPTH layers by minimizing KL against the original teacher, rather than the
FFN-compressed parent, give a better initialization for downstream recovery?

**Control: D1-q2.**
- State `afd50c9f…`, artifact `3a4b4ede…`.
- Path:
  - FFN `calib.domain_balanced@v1`
  - DEPTH `calib.reasoning_heavy@v2`
  - RESIDUAL_WIDTH `calib.reasoning_heavy@v2`
  - ATTENTION `calib.domain_balanced@v1`
- Rebuilt from the committed digest-pinned replay plan, which reproduced q2
  byte-exactly in `d1_replay_002`.
- Its DEPTH removal order was `[26, 32, 18, 17, 11, 10, 30, 16]`. Every greedy
  round's parent-reference table, 260 candidates in all, was extracted from
  the D1 search journal and checked against the plan's pins.

**Treatment.** The control spec with one declared change: DEPTH's config
declares `kl_reference: root_reference/v1`.

| Objective | Formula |
|---|---|
| Control | `KL(P ‖ C(S))`, where P is the 36-layer parent after q2's FFN step |
| Treatment | `KL(T ‖ C(S))`, where T is the original teacher |

Unchanged in the treatment:
- forward KL on the reference-defined Top-200 plus tail (`reference_topk_tail_v1`);
- the supervised-target positions;
- the per-item mean, then the sub-type and domain-balanced aggregation;
- the calibration data;
- the greedy removal, the 36 → 28 geometry and the tie-break;
- FFN, WIDTH and ATTENTION with their own rules.

Switching the reference also switches which token ids form the Top-K support,
because the support is defined by the reference. That is part of the declared
switch. It is not D2's `reference_student_union_topk_tail_v1`.

**Identities.**

| Identity | Hash |
|---|---|
| Control path | `2221484a…` |
| Treatment path | `5948a3c9…` |
| DEPTH operator config, control | `3a0f151e…` |
| DEPTH operator config, treatment | `b5dfa707…` |
| Untreated steps' configs | q2's recorded `464cb782…` / `3a0f151e…` / `464cb782…`, re-derived exactly under the new core |

**The contract.** `d3_design.treatment_contract` refuses any difference other
than:
- the path id and the labels;
- step 1's config and its input pin;
- the output pins of steps 1–3, which have no prior value.

## 2. The generic mechanism (core)

**What was added.**
- **`scoring/kl_reference.py`.** Absent means the historical parent and is
  never written, so no recorded hash moves. The module refuses:
  - an explicit default, or an unknown value;
  - an operator without `accepts_kl_reference`;
  - a declaration beside a consumed objective.
- **The DEPTH operator** reads the stored root through the existing
  `RootReferenceStore`:
  - it uses the same reducer and the same greedy removal;
  - it fails closed on the mixture, the profile, the support and the
    execution protocol;
  - it records every candidate's parent-reference KL as a diagnostic that the
    selection never reads.
- **`materialize_fixed_path`** builds the teacher store from the root before
  step 0, using the search's own builder and identity:
  - the identity is bound to the teacher id and revision, the shard hash, the
    profile, the support and the execution protocol;
  - it checks a store supplied to a suffix against the declared root.
- **Fixed-path step results** now keep DEPTH's round tables (selection
  evidence) and its cache decisions and timing (`operations`).
- **D2's objective path is unchanged.**

**Declared** in `tests/architecture/test_cuda_surface_preserved.py` as one
round from `ee52376c`, seven semantic files.

## 3. Structural outcomes (`d3_design.classify_structure`)

| Outcome | Condition | Next step |
|---|---|---|
| `NO_STRUCTURAL_CHANGE` | Same kept layers, and every downstream step reproduces q2's bytes | No recovery training |
| `STRUCTURAL_CHANGE` | Different kept layers | Propose the three-probe rung |
| `UNEXPLAINED_BYTE_DIFFERENCE` | Same layers with different bytes, or different layers with q2's bytes | Diagnose; nothing is called an effect |

The closeout re-derives the outcome off-pod from the persisted step records
and refuses to close if it disagrees with the pod.

## 4. The conditional behavioural rung (three probes)

**Setup.**
- Seeds 396880939, 766527341 and 1004399678, on the `d1_confirmation`
  battery (950 prompts, 850 scorable), with `e1_p1_kd_heavy@0.86M`.
- Each probe starts from the D3 initial checkpoint.
- Comparators are q2's and B's imported units: q2 is the method comparison, B
  the anchor.
- Same driver, protocol, scorer, token-id sidecar and off-pod verification as
  D2's confirmation (`autoinit_d2_screening_driver.py --rung d3_probes`).

**The comparison is exploratory.**
- Descriptive two-sided 95% intervals.
- Labels: IMPROVEMENT / REGRESSION / INCONCLUSIVE, by the registered interval
  and seed-sign rule.
- No alpha, no SHARED_GO, no promotion. The screening seeds are not added.

**$0 validation.** Fed D2's c4 rows, the analysis reproduces the D2
confirmation's recorded point deltas exactly:
- vs q2: −0.0102; per seed −0.0024, −0.0282 and 0.0; usable −0.0277;
- vs B: −0.0106.

## 5. Price, budget and proposal

| Session | Expected | Hard | Hard + reserve |
|---|---|---|---|
| Structural | $0.86 (46.6 min) | $3.02 (163.5 min) | $3.32 |
| Probes (conditional) | $6.59 | $8.21 | $8.51 |

**What the structural price is built from.**
- q2's replay operator times (DEPTH 843.5 s cached).
- The search's recompute-mode DEPTH (1819.5 s) plus its full distortion
  seconds for the every-round parent diagnostic.
- The teacher store, at 2 expected / 5 bound candidate-equivalents.
- A host bound factor of 1.25.
- The D2 search's measured setup and collection, with its priced 95-minute
  overhead as the bound.
- Four reserved refused host draws.

**The probes price.** It is the D2 confirmation step unchanged, the same
session shape, which measured $5.03.

**Live limits.**

| Limit | Remaining |
|---|---|
| Project | $100.4494 |
| Formal | $104.4270 |
| Package | $107.6074 |
| Engineering | $3.1804 |
| Per session | $30 |

L40S quoted $1.09/h, stock Low. Both sessions at hard with reserves cost
$11.83, which leaves $88.62.

**Proposed envelopes.**
- Structural: $7.00. It holds one session and one corrected pre-science
  attempt.
- Probes: $12.00, conditional on STRUCTURAL_CHANGE.

**Required amendment.** Add `phase_d3` to `funds_formal_sessions_of`. This is
a maintainer amendment; the issuer refuses without it.

The old D3 reservation (the $50.95 search plus its own screening and
confirmation, and a fresh confirmation) is **not** transferred.

## 6. Provider and resource readiness

**Launcher and host.**
- `autoinit_d3_structural_launch.py` (SESSION_KIND `d3_structural`) uses the
  shared `SessionRunner`, with its watchdog, teardown and collection.
- Host admission requires the protocol's image identity
  (`…@595.91.07`, AU).
- `$0` gates:
  - the run id and registration;
  - the treatment re-derived and equal to the bound path hash;
  - the session commit;
  - the same-failure rule;
  - the bundle round trip.

**During and after the run.**
- `steps.jsonl` and the journal relay off the pod on every poll.
- When a new initialization is produced, it is fetched and re-identified at
  all four identities before the pod may go.
- The `$0` closeout uploads it to the relay, checks the LFS oids and restores
  it elsewhere for re-identification.

**Dispatch and checks.**
- The shell dispatch branches `d3_structural` and `d3_probes` load their own
  types and assert every denial.
- The pod preflight re-derives the treatment from the staged mixtures and
  runs a CPU parity check of the declared reference.
- The collector dry run on a driver-shaped tree: manifest (4 of 4 classes),
  archive, verify-archive and verify-local all pass.

## 7. Open items and boundaries

**Open, for the maintainer.**
- The structural grant: envelope, AU constraint, approval.
- The `phase_d3` funding amendment.
- The probes grant only after STRUCTURAL_CHANGE.

**Engineering risk, not blocking.** The teacher store is built before the
pinned FFN step 0. Forward passes are deterministic, so q2's prefix should
still reproduce. A mismatch would stop the session within minutes, at
FixedPathDigestMismatch on step 0.

**Not authorized:** paid execution of either session, a Full Search,
reopening D1/D2, the cancelled D3, data work, promotion, cap changes.

**Interpretation.** Results on the mixed-domain battery are not fluid-reasoning
evidence (AGENTS.md P10.2). If the battery cannot separate the two
references, both are carried to a future capability-aligned evaluation.

## 8. Addendum 2026-10-17 UTC: the review's two repairs and the grant

The maintainer's independent review of `0375d7d8` accepted the design (GO for
the structural ablation), required two `$0` execution-integrity repairs, added
`phase_d3` to the formal book and granted the structural rung `$7.00`
cumulative. Decision: `logs/budget/decisions.md`, 2026-10-17. Grant:
`logs/budget/approvals/autoinit_d3_structural_grant.json`.

**A. The parent-reference comparability gate** (`d3_design.parent_reference_admission`).
The informational `round0_parent_diagnostic_vs_control` is replaced by a
fail-closed admission requirement on the treated DEPTH step:
- round 0 tests exactly q2's 36 candidates, each once;
- every candidate carries a finite round-0 parent-reference KL;
- every one agrees with q2's recorded score within `|d3 − q2| ≤ atol + rtol·|q2|`;
- the parent-reference argmin is q2's first removed layer, 26.

An absent or partial diagnostic fails. The driver checks the moment DEPTH is
persisted and stops an inadmissible run before WIDTH, with status
`INADMISSIBLE`. That is not a fourth outcome: nothing is classified and
nothing is proposed. The closeout re-derives the verdict off-pod and refuses a
disagreement, and the probes contract refuses any run that was not admitted.

The tolerance is fixed in `plans/d3_parent_reference_admission_v1.json`
(`write_d3_admission.py`) from GPU evidence that already exists:

| Evidence | What it shows |
|---|---|
| q2's own DEPTH, run by two D1 search sessions | 260 of 260 candidates bit-identical |
| The root's DEPTH, D1 decision vs D2 diagnostic, same driver and execution shape; code, cache mode, row policy and store residency differ | max abs 4.95e-05, max rel 4.43e-04, argmin unchanged |
| Tolerance: those maxima rounded up at one significant figure | `atol` 5e-05, `rtol` 5e-04; the evidence uses at most 46% of it |
| Every other round-0 table over the same candidates in both journals (29) | all refused, the closest with 34 of 36 candidates outside |

q2's own parent under the other profile chooses layer 26 as well. The argmin
check alone would admit it; the numerical check refuses it.

**Residual, stated in the record.** D3 runs on the protocol's 595.91.07 driver,
and q2's scores came from 580.126.09. No score-level DEPTH measurement exists on
595.91.07. `d1_replay_002` there reproduced all four of q2's step identities,
FFN's included, which is digest-level evidence only. A driver effect beyond the
tolerance makes the run inadmissible; it cannot make one admissible.

**B. A supplied root reference is always checked**
(`fixed_path._acquire_root_references`). With no `root_identity` declared, a
supplied store must declare the spec's own root (`teacher_id` = repo id,
`teacher_revision`). Any other root, or the right teacher declared in another
form, is refused. No production caller supplies stores to the fixed path, so
no historical execution changes. The negative tests include the real suffix
entry point with `root_identity` omitted.
