# D2 behavioural screening: readiness review (2026-10-15)

> **Correction and addendum, 2026-10-15 (UTC), after the maintainer's Final
> Independent Review of `8e8ae849`.** Sections 3 and 5 below used to read as
> if the pod-side checks had happened. No pod has been created, so they are
> **mandatory gates the driver will run, not verification facts**; the text
> now says so. What was done at `$0` keeps its evidence. Section 10 records
> the three `$0` repairs the review required before issuance; section 9 the
> conditional grant.

Prepared under the maintainer's Post-D2-v3-Search Independent Review of
`7dd31165`. That review authorized **$0 work only**: the screening chain, the
verification of its inputs, evidence durability, tests, and a grant proposal.
**No pod was created and no probe ran.** The chain stops at the paid
authorization boundary.

Owners of every number below:
- the contract: `scripts/stages/stage-1/phase_d2/d2_screening.py`;
- the price: `families/d_series/plans/d2_budget_v1.json :: steps.d2_screening`;
- the proposal: `logs/stages/stage-1/phase_d2/plans/d2_screening_grant_proposal.json`.

## 1. Executable commit

The commit that adds this file. The pod would check that commit out from a
canonical bundle. The issuer binds it as the authorized base, and the
lineage gate allows exactly one path to differ: the authorization artifact.

## 2. The frozen eight-probe plan

| Arm | State | Path tail (FFN, ATTN) | Seeds |
|---|---|---|---|
| c1 | `59013099…` | FFN(rh), ATTN(dom) | 516448251, 614923639 |
| c2 | `c2cb27ef…` | FFN(dom), ATTN(rh) | 516448251, 614923639 |
| c3 | `30fe791a…` | FFN(rh), ATTN(rh) | 516448251, 614923639 |
| c4 | `3a4173e4…` | FFN(dom), ATTN(dom) | 516448251, 614923639 |

All four share `RESIDUAL_WIDTH(dom) → DEPTH(dom)`, removing layers
4 9 13 18 30 32 34 35. The arm order is the committed quality order.

Frozen and read from their owners:
- **Field.** The committed selection: Top-4, quality-only retention, K = 4.
  It is not narrowed and nothing is added.
- **Seeds.** Preregistration v3, which equal the shared screening rung's seeds.
- **Recovery.** `e1_p1_kd_heavy@0.86M`.
  - Recipe digest `a4b0dcab…` and data digest `6f324cb0…`, recomputed at $0
    against the shared protocol.
  - The D2 search objective enters nothing in the recovery loss.
- **Evaluation.** The shared rung's evaluation protocol `e4750826…`: the
  generation fingerprint `ceac0947…`, the scorer bytes (`score_d1_screening.py`,
  scoring digest `33b8ff5e…`) and the battery `d1_screening` (950 prompts,
  850 scorable).
- **Comparator.** B's two imported units, `d1_screening_B_s516448251` and
  `…s614923639`. q2's two units are descriptive only.
- **Rule.** Pooled seed-mean `correct_overall` delta vs B. Usable-rollout
  vetoes only. Ties break by quality position, then state id. Exactly one
  candidate advances, or none. It is applied off-pod (`rank_d2_screening.py`).
- **Contract hash** (location-free): `6d4283e7…`.

**Statistical integrity.**
- The four candidates are near-siblings, so eight probes are not eight
  independent architectures.
- No seed is added, no SESOI moves, and nothing is re-screened because
  candidates land close together.
- B's rows were already observed in D1. The shared protocol records the
  adaptivity and multiplicity limits.
- Screening nominates for the shared confirmation at most. It never promotes.

Descriptive structure, which never feeds a rule
(`evidence/candidate_differences.json`):
- **FFN masks.** There are 2 distinct masks, one per FFN profile. They share
  97.49% of kept neurons (83,861 of 86,016).
- **Attention masks.** There are 4 distinct masks. They share 95.98–97.99% of
  kept query heads.
- **Shared units.** Every unit that two candidates both kept has
  bitwise-identical weights. Embeddings, norms and k/v projections are
  identical across all four. The candidates differ only in which units they
  kept.

## 3. Initial checkpoint identity checks

For each arm, all four identities (`artifact_digest`, `weights_digest`,
`single_shard_sha256`, `arch_signature`) were recomputed from bytes and
compared exactly.

**Done at $0 (evidence below):**
1. at rest on the dev host;
2. after a relay round trip into an independent directory.

**Pending, a MANDATORY GATE of the paid session (not yet executed; no pod has
existed):**
3. on the pod, by the driver, on the staged relay copy, before anything
   trains. A mismatch refuses the session before training.

Evidence: `phase_d2/runs/d2_search_20261015_051722/evidence/product_durability.json`
and `phase_d2/analyses/d2_screening_inputs.json`.

## 4. Reuse of B's historical units

This is the shared protocol's own `_verify_unit`, run at $0
(`phase_d2/analyses/d2_screening_inputs.json`). For each of B's two screening
units (and q2's):
- the result self-hash holds;
- the durable result equals the committed result;
- the per-sample rows and generations match their sha256;
- a re-score of the stored generations with the pinned scorer reproduces the
  per-sample rows **byte for byte**;
- the training identity equals the protocol's recovery block;
- the admission matches the rung's evaluation protocol.

Their checkpoints are retired. Nothing about B or q2 is retrained,
regenerated or re-evaluated, and no confidence or logits are computed for
them.

## 5. Session, training and evaluation contract

Everything in this section that runs on the pod is a **mandatory gate still
to be executed**. None of it has happened: no pod has been created. What
has been verified at `$0` is the dev-host side (the contract re-derived, the
arms hashed, the relay round trip, B's units, the CPU launcher rehearsal of
the `$0` gates). The pod-side gates are:
- the arms' four identities on the staged copy (step 3);
- the training runtime digest before the first probe trains (step 3);
- the attested evaluation protocol before any generation (step 5);
- per probe, the trained checkpoint's identity before the pod may release it
  (section 6).

The pod runs these steps in order:
1. Setup on an admitted host only. The image identity must equal the
   protocol's `…@595.91.07`, because the training runtime digest fuses the
   host driver. Up to 5 draws.
2. The arms are staged from the relay as pinned inputs.
3. The driver refuses before anything trains unless all of these hold:
   - the arms verify at all four identities;
   - the contract re-derives to the bound hash;
   - the training runtime equals the protocol's `9bd30885…`.
4. Each probe runs D1's own functions:
   - the frozen recipe, with only C1's override set;
   - the device handoff and the trainer;
   - the evaluation package, built from the trained weights and the frozen
     tokenizer.
5. The protocol is attested from the first package. The run stops before any
   generation unless that hash equals the shared rung's.
6. Generation runs through the read-only token-id sidecar around the
   unchanged generator.
   - If the wrapper fails, its partial output is preserved aside and the plain
     generator runs.
   - Per-stratum row alignment is recorded in each probe's entry.
7. Admission, then the pinned scorer.
8. Rollout rescoring on the candidate's own rollouts, in its own subprocess
   and failure-isolated.
   - It uses engine ids where the sidecar is exact. Otherwise it labels the
     sample `retokenized_text`.
   - Its thresholds are the reference calibration's.
9. The session stops at preservation. The ranking is computed off-pod.

## 6. Durable recovery and failure evidence

D1's verified durability functions are reused, under a D2 layout with no
shared path:
- Every scored probe's result, rows, admission and generations come off the
  pod on each poll.
- Every trained checkpoint is preserved off-pod and identity-verified before
  the pod may release it.
- A replacement pod restores preserved probes and resumes them at evaluation.
  A frozen unit is never retrained.
- Preserved weights are released only at a successful terminal.
- Provider teardown is the runner's gate.
- The ledger is append-only through the closeout and `derive_budget`.

The four initializations are durable twice (the dev host and the relay) and
are restorable byte-exactly.

## 7. Cost and programme reservation (re-derived at the live rate)

**Screening at the live $1.09/h:**
- expected **$15.0704** (817.2 min);
- hard **$19.4558** (1055.0 min);
- plus the $0.30 teardown reserve.

This is below the earlier planning figure ($16.61 / $20.83), because the arms
are now staged from the relay instead of four 22-minute on-pod
re-materializations. Rescoring is inside the ceiling (1.83 min/probe,
measured).

**Money checks:**
- All four conditions pass: per-session $30, project cap $595, formal
  remaining $122.4260, package remaining $125.6064.
- `phase_d2` is funded.

**Programme** (`d_series_programme_budget.json`):
- D2 remaining is $28.27 at hard with reserves, down from $29.98.
- After it, $90.18 remains against $66.45 needed for D3 plus a fresh
  confirmation. The reservation holds with $23.73 to spare.
- Even if the whole proposed $25 screening envelope were spent, $84.94 would
  remain.

**Budget contract.** The completed search now leaves the plan only through a
nine-clause machine-readable contract (`search_completion.py`). Project
remaining is unchanged at **$118.4484**.

## 8. One-use authorization proposal

See `logs/stages/stage-1/phase_d2/plans/d2_screening_grant_proposal.json`.
- It covers one screening session.
- The proposed cumulative envelope is $25.00. That is one session at hard
  plus room for pre-science subruns.
- The issuer derives and binds:
  - the ceiling at the live rate;
  - the contract hash;
  - the seeds and the probe count;
  - the harness;
  - the run id.
- The issuer refuses:
  - a grant that states any derived field;
  - an envelope already narrowed below the session's price.

## 9. Authorization

**At the time of this review (`8e8ae849`), not authorized:** creating any pod;
running any of the eight probes.

**Since then (2026-10-15 UTC):** the maintainer conditionally approved a
**$25.00 cumulative all-in grant** for ONE frozen screening rung
(`logs/budget/approvals/autoinit_d2_screening_grant.json`), to be issued only
after the three `$0` repairs of section 10, their tests, commit and push, with
the executable commit rebound.

**Still not authorized:** the shared confirmation; retraining, regenerating or
re-evaluating B or q2; D3; a fresh confirmation; any promotion; a project-cap
change. After screening the programme stops at the confirmation
authorization boundary.

## 10. The three `$0` repairs required before issuance

**A. The off-pod ranking's evidence gate** (`rank_d2_screening.verify_unit`).
A candidate unit verifies only when, from its secured files:
- the result's self-hash holds and it is the session's record;
- its label, seed, screening arm and initialization digest equal the FROZEN
  unit's (contract), and the session entry names the same probe, arm, seed and
  committed state;
- its admission record names this probe, is comparable, carries the shared
  rung's evaluation protocol hash, and its generation fingerprint is the
  result's and the session's;
- its rows and per-stratum generations match the result by the shared
  protocol's own checks (`shared_evaluation.scored_evidence_checks`, now also
  what B's audit calls), every row carries the frozen label and seed, and the
  counts match;
- the pinned scorer, re-run on the stored generations
  (`shared_evaluation.rescore_unit`), reproduces the rows byte-exactly and every
  non-volatile field of the result (shown on B's real unit: a dev-host re-score
  of a pod-scored result differs only in timestamps, command, paths and the
  hash over them);
- its secured run manifest and completion record equal the digests the driver
  recorded, the completion is the one bound into the result, training completed
  every step, the config hash is bound end to end, and the manifest shows the
  protocol's recovery block, recipe digest and the committed candidate at the
  frozen seed.

A record whose self-hash holds but whose arm, seed, initialization, training
or scores contradict the frozen unit is refused. Mutation tests re-seal the
edited records so only the targeted check can refuse, and a meta-test requires
every check to be falsifiable.

**B. The training records the ranking reads are durable.** D2's layout now
declares `run_manifest.json` and `run_completion.json` as REQUIRED training
records (D1 declares none; its behaviour is unchanged). The driver digests them
at training completion; the launcher secures them into
`store/probes/<id>/training/` against those digests, independent of the
checkpoint copy; they are part of the preserved resume set; and the release
ack, the scored-evidence gate and `probe_fetch_ok` all require them. A
successful terminal no longer prunes a preserved copy while the records beside
the evidence are missing or do not re-hash, and the prune never writes over a
verified copy.

**C. The cumulative retry reserve.** An issuance now requires
`remaining envelope >= derived session hard ceiling + $0.30 teardown reserve`.
An earlier subrun counts as known only when its closeout books
`money.all_in_usd` AND states `provider_teardown_verified: true`, or when a
record shows nothing was ever created at the provider. An issued authorization
with no session record, a session record with a pod id and no closeout, an
unreadable record, or a closeout without a booked cost or verified teardown is
UNKNOWN, and nothing is issued over it.
