# D2 / D3: next-round design on a shared behavioural evaluation

Status: **DESIGN, 2026-10-13.** Authorizes nothing. It follows the maintainer
decision of 2026-10-13 (`logs/budget/decisions.md`). Every number below is
derived by a script and recorded in a machine-readable file; this page explains
them and cites their owners.

| owner | what it holds |
| --- | --- |
| [`d_series_shared_eval_v1.json`](d_series_shared_eval_v1.json) | the shared protocol: rungs, seeds, pins, statistics, claim boundary. `protocol_hash` = `5ed198ae…` |
| [`../analyses/d1_shared_evidence_audit.json`](../analyses/d1_shared_evidence_audit.json) | the $0 evidence audit, the D1 replay and the probe economy |
| [`../validations/prediction-rows-r0/v1/plan.json`](../validations/prediction-rows-r0/v1/plan.json) | R0's tolerances, time and money, and command |
| `scripts/stages/stage-1/families/d_series/shared_evaluation.py` | builds the protocol and audit; check mode by default, `--write` regenerates |
| `scripts/shared/evaluation/evidence_reuse.py` | the identity-based reuse classifier |

## 0. Order and scope

The order is **R0 → D2 → D3 → Stage-0 teacher-native**.

- R1, the $0 corpus audit and profile build, may run in parallel. Its new
  calibration profile does **not** enter D2 or D3.
- D2 and D3 each run a **full search** under their own objective, from the
  same root, operator set, beam, geometry and calibration data as D1 (roadmap:
  "one variable"). Neither re-ranks D1's journal, and neither compresses q2.
- B stays the standing incumbent and the comparator for every decision.
- q2 is the working baseline. It is reported beside B and is never promoted
  by being used.

## 1. What D1 left, unit by unit (audit: 10/10 reusable)

The audit re-reads each of the ten B/q2 units:

- **Inputs:** the committed result, the run relay, the probe config and the
  session's run manifests.
- **Checks against the durable store** (`~/aad-artifacts/phase_d1/<run>`):
  1. the result's self-hash, and its bytes against the durable copy;
  2. the per-sample bytes against `per_sample_sha256` (950 rows, each with
     `id`, `set`, `seed`, `correct`, `usable` and `scorable`, so they pair);
  3. all seven generation files against their recorded sha256;
  4. **a re-score of the raw generations by the rung's own pinned scorer**,
     which reproduces every per-sample file byte for byte, in about 2 s per
     unit on CPU.

**D1's own two decisions, re-derived through the shared code path:**

- **Confirmation:** NO-GO, delta −0.000392, LCB −0.007843, UCB +0.007451, with
  identical per-seed deltas.
- **Screening:** order q2 > q1 > q3 > q4, selection q2.

Both are identical to the committed records. Three injected faults are each
caught: one flipped B row, a moved veto threshold, and a tampered hash.

| unit class | evidence | reuse |
| --- | --- | --- |
| B ×5, q2 ×5 per-sample rows, raw generations, results | present, hash-verified, re-scorable | **unconditional** inside the shared protocol (`REUSE_RESULT`) |
| training identity (init digest, recipe digest `a4b0dcab…`, pack `6f324cb0…`, seed, teacher `768f209d`, trainer source `a1807e34…`, training runtime `9bd30885…`, L40S) | confirmation: per probe in git. Screening: B's two manifests in the store, q2's taken from the same session (one checkout, one pod), and the audit records which | sufficient for identity matching |
| trained checkpoints (`trained_sha256` recorded) | **retired**, 0 of 10 present | never re-evaluable: any other evaluation of B or q2 means a new training draw |
| aggregate scores | in git | recomputable from the rows, so never consumed alone |

Remaining gaps:

- The checkpoints are gone, so B and q2 can be reused **as measured** and in
  no other way.
- The durable store is a single dev-box copy. It is now registered as
  `reproducibility_required` in the checkpoint registry, with this protocol
  as its consumer.

## 2. How a new candidate becomes comparable with the historical rows

A D2/D3 candidate is paired with B and q2 only if **every training key except
the initialization** and the **evaluation protocol hash** match the protocol
record. This is checked twice:

- **$0, before any pod:** `tree_matches()` confirms that this tree's trainer
  source, generation source, scorer bytes and scoring-contract files equal the
  pinned ones. It holds today, and a test fails if any of them changes.
- **On the pod:** the run manifest (runtime digest, L40S) and the attested
  generation fingerprint `ceac0947…` must equal the record's.

Prompts and seeds are D1's own:

- **Shared screening:** `d1_screening` (item ids `d7b95cdd…`), seeds
  516448251 and 614923639.
- **Shared confirmation:** `d1_confirmation` (`7d200ffe…`), seeds 396880939,
  766527341 and 1004399678.
- **Recipe:** `e1_p1_kd_heavy@0.86M`; only the preregistered override set may
  differ.

Scoring runs **off-pod** with the rung's own scorer
(`score_d1_screening.py` `b49ff1b2…`, `score_d1_confirmation.py`
`7481eaf0…`). The audit shows that this reproduces on-pod scoring exactly.

Reuse is decided by `evidence_reuse.classify`, never by a name, a path or a
seed. The audit shows each outcome on real data:

- a new candidate gives `RECOMPUTE_ALL`;
- B on a reserved fresh battery gives `TRAIN_FROM_INIT`, because its
  checkpoint is retired;
- B under a changed trainer, or at a seed D1 never ran, gives
  `TRAIN_FROM_INIT`, naming the one field that differs.

## 3. Versioning: nothing frozen is rewritten

- `d_series_shared_eval@v1` is a **new identity** beside the family. Its hash
  covers the science, not the presentation.
- The six family roles keep their bytes, content hash and allocation rule
  `f6047343…`.
- D1's preregistration and verdict are **cited by hash** in
  `d1_records_unchanged`. Its NO-GO stands.
- `d2_screening`, `d2_confirmation`, `d3_screening` and `d3_confirmation` stay
  **unconsumed**. They are the fresh batteries a later promotion needs, and
  their identities are copied into the record so that any drift is
  detectable.
- A change of trainer, scorer, engine or recipe means a **v2** of this
  protocol, not an edit.

## 4. What sharing costs statistically

1. **Neither rung is blind any more.** Both have been observed.
   - Screening is used only to rank D2/D3's own field against B's imported
     rows.
   - q2's screening rows carry the winner's curse: screening +0.00825,
     confirmation −0.000392. They are **descriptive only**.
2. **Confirmation remains a valid paired comparison.** It uses the same
   prompts, seeds, recipe and protocol, and C0's estimand: the prompt-mean of
   the seed-mean paired difference, under a stratified prompt-cluster
   bootstrap with seeds as fixed blocks. Two limits apply:
   - It is not a fresh test. D2 and D3 are judged against the **same B rows**,
     so their statistics are positively dependent.
   - The family {D2, D3} is controlled by **Bonferroni with a fixed equal
     split, one-sided α = 0.025 each**. That holds under any dependence and
     needs no knowledge of which experiment reports first. D1 already spent
     α = 0.05 on these rows, so the union bound on *any* false pass across
     D1–D3 is **0.10**.
3. **Multiplicity correction does not cover adaptive reuse.** That risk is
   controlled by what is frozen before any D2/D3 behavioural result exists:
   - the objective definitions, already in the 2026-10-01 roadmap;
   - this protocol;
   - each experiment's own retention rule, field size and preregistration.

   No D2/D3 design choice may read the per-prompt outcomes of the
   confirmation rung.
4. **The session boundary is pinned, not measured.** The historical units ran
   in D1's sessions, and a candidate runs in a new one. Everything that can be
   pinned is pinned. The residual is host, driver and CUDA nondeterminism: the
   same kind of variation two probes of one session already show, since
   training is not bit-reproducible, but it is not separable from an arm
   effect.
5. **Outcomes.**

   | outcome | condition | meaning |
   | --- | --- | --- |
   | `SHARED_GO` | C0's GO at α = 0.025 | **nominates, never promotes** |
   | `NO_GO` | — | terminal for the candidate |
   | `INCONCLUSIVE` | — | terminal; never rerun on these rows |

   - Promotion needs a fresh confirmation on the reserved battery, with a
     common control measured there. How to obtain that control cheaply is
     decided when a `SHARED_GO` exists, not before.
   - For calibration: q2 itself, read under the family α, is NO_GO
     (UCB 0.008627 < 0.01).
   - The comparison against q2 is secondary and descriptive, and spends no α.

## 5. Probes and money per experiment

Measured on the ten imported units:

- **Per probe:** training 61.3 min (max 61.5), generation 24.2 min (max 34.6).
- **Rate:** L40S at $1.09/h.
- **Per-seed spread** of paired deltas, pooled over D1's five arms: 0.0107.
- The shared design also skips materializing the reference arms on the pod in
  every session: B 21.1–21.3 min, q2 15.7–15.8 min.

| K candidates | D1-style probes | shared probes | saved | shared GPU $ expected / ceiling | screening inflation at null | P(advance the SESOI-better one) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 5 | 5 | 7.76 / 8.72 | 0 | 1.00 |
| 2 | 12 | 7 | 5 | 10.87 / 12.21 | 0.0043 | 0.825 |
| 3 | 14 | 9 | 5 | 13.97 / 15.70 | 0.0064 | 0.722 |
| 4 | 16 | 11 | 5 | 17.08 / 19.19 | 0.0078 | 0.652 |

The savings in detail:

- **Against D1's field shape:** 5 probes per experiment, $7.76. Across D2 and
  D3 that is 10 probes, about 14 GPU-hours and $15.5.
- **Against retraining both B and q2 as controls**, which is what the decision
  forbids: (K+2)·2 + 9 probes, so 10 probes saved per experiment and 20
  across D2 and D3, about $31.
- The figures exclude session setup, candidate materialization and the D2/D3
  searches themselves. D1's search measured 552 min, $10.03, at full
  vocabulary; Top-K and `scored_v1` should lower it.

K is each experiment's preregistered choice, not this protocol's. At D1's
measured spread, **K = 2** gives the best balance:

- **Probes:** 7 per experiment.
- **Screening inflation:** 0.43 SESOI.
- **Discrimination:** 0.82.

K = 4, D1's choice, inflates the screening estimate by 0.78 SESOI and
discriminates at 0.65.

## 6. Reuse along the whole evidence chain

| stage | artifact | identity key | D2/D3 reuse |
| --- | --- | --- | --- |
| teacher corpus / tokenization | `corpus_v2`, tokenizer `7781771a…` | corpus hash, tokenizer sha | reused as is (R1 only *reads* it) |
| calibration materialization | `calib.domain_balanced@v1`, `calib.reasoning_heavy@v2` items | profile qualified id + items sha | reused; **no new profile in D2/D3** |
| teacher reference sketches | state-eval sketch (Top-200 + tail + exact target log-prob), DEPTH parent sketch | teacher revision, suite/profile content, support, execution fingerprint | rebuilt per session from cached inputs. D2 adds an entropy field computed at build time from the already-normalized row (one extra O(V) reduction); the entropy is not recoverable from Top-K + tail afterwards |
| confidence statistics | D2 `c_ref` per (parent, profile) | parent digest + profile + policy hash (the stats-cache key shape) | cacheable per parent, like activation statistics; D3's candidate confidence is per candidate and never cached |
| D1 search journal, states | journal, Top-4 identities, q2 state `afd50c9f…` | state id + artifact digest | diagnostic only: D1's path re-scored under the D2/D3 objective as a fixed-reference comparison. Never a substitute for the full search |
| B, q2 initialization | `53e30566…`, `3a4b4ede…` | artifact digest | not needed on the pod under the shared protocol; their measured units are imported instead |
| recovery data, recipe, seeds | pack `6f324cb0…`, recipe digest `a4b0dcab…`, D1 seeds | as pinned | reused unchanged; identical by construction |
| trained artifacts | `trained_sha256` of each unit | sha256 | identities only; bytes retired |
| generations, per-sample scores | durable store, hash-pinned | per-sample sha256 | **consumed directly** by ranking and verdict |
| aggregates, bootstrap inputs | rows → `decision_inputs` | per-sample sha256 + bootstrap seed rule | recomputed off-pod at $0; bootstrap seed per (protocol, experiment, rung) |

## 7. D2 and D3: what the objectives need that does not exist yet

**The definitions** come from the roadmap (`phase_c1/plans/phase_c_roadmap.md`
569–632).

- **D2:**
  - weight `w = target_mask · c_ref`, with `c_ref = 1 − H(p_ref)/log|V|`;
  - the reference is the **intact parent** for operator-local statistics and
    the **frozen teacher** for state evaluation.
- **D3:**
  - KL is kept, and `C_student = 1 − H(q)/log|V|` is added as a **separately
    visible** component;
  - both are joined in a preregistered combination whose normalization and
    mixing are frozen from incumbent/reference calibration before any search
    result.

**What already holds in code:**

- weights are float tensors with a `weighted` form;
- the KL/CE/critical-token reducers form Σw denominators;
- `scored_v1`'s row selection is a superset for any w ≥ 0;
- the activation collectors **refuse** continuous weights by name
  (`require_binary_token_weights`), so nothing silently rounds w to a mask.

**D2 must implement:**

1. **Model-dependent weights.** A weight producer that combines the pre-model
   target mask with `c_ref` from a reference forward. Today every policy is
   evaluated from tags alone, before any model runs (`positions.py:257–265`,
   `metrics.py:161–167`).
2. **Entropy in the reference sketch.** `sketch_reference` stores no entropy,
   and the DEPTH parent sketch is built with all-zero targets, so neither can
   provide `c_ref` today. The fix is one reduction over the already-normalized
   row at build time.
3. **Operator-local `c_ref` before the hooks fire.** FFN and attention
   statistics are collected *during* the parent forward, so `c_ref` needs one
   extra parent forward per (parent, profile), cached under the stats-cache
   key.
4. **Weighted collectors.** Weighted numerators, a float Σw `StatsSpec`
   quantity, and divisors at **four** call sites:
   - `uncentered_moment`;
   - `ffn_neuron_importance`;
   - `residual_covariance`;
   - `head_write_energy`, which the roadmap's "three call sites" missed.

   The `n < 2` guards need adjusting, and the binary refusal is relaxed only
   on the new path.
5. **Identity.** A new policy id binding the reference identity (parent digest
   or teacher revision) and the entropy definition into the config hash and
   the measurement protocol id.
6. **Aggregation.** State evaluation pools within a subtype, while DEPTH
   averages per item. The protocol must state which is meant under continuous
   weights. Optionally, move `masked_item_means`' float32 denominator to
   float64.

**D3 must implement:**

1. **Candidate entropy.** A full-vocabulary candidate-entropy reducer in both
   state evaluation and DEPTH. The logits and normalizer already exist there.
2. **New state metrics.** A confidence component plus the combined objective,
   both bound into the protocol id, with a ranking objective and ε.
3. **Frozen combination.** The normalization and mixing coefficient as a
   hashed experiment-layer artifact, never in `src`.
4. **No gaming channel.** `C_student` may never sit in a KL denominator; it
   enters only as its own component.

**Decisions owed before D3 can be preregistered:**

- what "student" means for FFN, WIDTH and ATTENTION, which have no candidate
  forward;
- whether H is taken over the full vocabulary or over the K + 1 partition;
- D3's mixing rule.

On the first: read literally it is the parent's confidence, which makes D3
identical to D2 on those operators.

**For both, under `scored_v1`:** rows are selected by the **pre-model target
mask only**. The confidence factor exists only after logits, so it multiplies
in afterwards. Selecting rows by confidence would be circular, and the
superset property holds because c ≥ 0.

**Diagnostics to keep:**

- Every D2/D3 state also reports D1's fixed-weight (`supervised_target_v1`)
  metrics, so a candidate-dependent weighting can be read against a fixed
  reference.
- For D3, KL, raw confidence and the combined score are kept side by side.

## 8. R0 (see the plan)

**The earlier estimate was wrong.** "About $1.5, about one hour" priced only
the expectation. The work alone takes about 83 min.

**What R0 runs.** One L40S pod runs stages D and P for both modes back to
back, the oracle (`all_v1`) first:

- **Stage P:** 2 real DEPTH invocations per mode, 260 candidates each.
- **Stage D:** 2 state evaluations per mode.

**Cost.**

| case | time | all-in |
| --- | --- | --- |
| expected | 83.4 min | $1.53 |
| ceiling: maxima of the cached path, no saving assumed | 111.0 min | $2.03 |
| **hard ceiling**: all four invocations at the slowest measured one (2484.66 s, the reference-cache recompute fallback) | 196.7 min | **$3.60** |

- Peak memory is 20.5 GiB, so a 24 GB card is too tight.
- **Proposed grant:** all-in $4.00, session cap $3.60 (198 min of useful
  runtime), teardown reserve $0.30, drawn from the GPU engineering book
  ($6.65 left).

**Gates**, unchanged from the declared design:

- identical removal order;
- per-candidate relative difference ≤ 1e-5;
- ranked state objectives (`PARETO_V1`) within an absolute 1e-5;
- the `all_v1` oracle reproduces the historical run's protocol ids and
  removal orders;
- `scored_v1` binds different ids.

Timing and memory are reported and not gated. A failure is diagnosed, never
re-toleranced. Before reading results, note that a bf16 output-head GEMM over
fewer rows may round differently, so the 1e-5 relative bound is a real test.

**Fixed at $0 this round.** None of these is a change of science:

- the launcher resolved the repository two levels short;
- it was funded by a constant that named the **CLOSED** Top-K campaign, and
  the shared budget script never read `status`. It now refuses closed
  campaigns. With the check removed, that campaign still derives a $3.00
  session (checked by mutation); with it, the campaign is refused under test;
- the mode never reached the pod;
- one output directory served every mode;
- the driver's stage P now checks the deadline between invocations (never
  inside one), and the record is flushed after every stage and profile;
- a comparator now reads the plan's tolerances.

**Readiness.** R0 can run as soon as a grant writes `authorization.json`.
Until then the budget script refuses the campaign.

## 9. Generality of `src/aadistill`

This round adds **nothing** to `src/aadistill`.

- The reuse classifier is application-layer (`scripts/shared/evaluation`). It
  takes field names and values as arguments and names no experiment, model or
  store.
- Everything instance-specific lives in this family's script and records:
  - battery roles, seeds, digests and runs;
  - the GPU class, α and the family membership.
- The D2/D3 core work in §7 is expressed through generic contracts:
  - a weight producer that may depend on a model;
  - a Σw statistics quantity;
  - an entropy field on a reference sketch;
  - adapters keep owning family structure.

  None of these names Qwen, D2, D3, a dimension or a ratio, so they serve any
  stage, family, scale or topology.

## 10. R1 (Stage-0 teacher-native), when it is built

R1 runs at $0 from `corpus_v2`:

- It is a versioned, length-stratified profile over complete thinking +
  answer sessions. The old 256–2048 window admitted code 14.4% and openmath
  39.0%; the new profile follows each slice's real length distribution up to
  the 8192 generation cap.
- **Contamination checks** cover:
  - all six family roles, including the unconsumed d2/d3 ones;
  - `c1_confirmation_v1` and `c2_screening_v1`;
  - the recovery training pack and its validation blocks;
  - `state_eval_v1`.

  The checks use the family's canonical problem-content identity
  (`identity.py`), not only native ids.
- The new profile is a Stage-0 variable and never enters a D2/D3 comparison.
