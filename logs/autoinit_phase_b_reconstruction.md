# Phase B — reconstruction and readiness pass

**Status: RECONSTRUCTION, 2026-08-24. Zero cost. Nothing was launched, no pod was
created, no grant or authorization was issued, and no paid work was performed.**

This document answers one question — *what does the repository actually specify
Phase B to be?* — and records what blocks it. It does **not** design Phase B, and
it does not resolve any of the open scientific choices in §6; those are the
reviewer's.

Sources: [`autoinit_pilot_proposal.md`](autoinit_pilot_proposal.md) §4, §5, §6, §9;
[`../docs/AUTOINIT_REFERENCE.md`](../docs/AUTOINIT_REFERENCE.md) §9.4;
[`decisions.md`](decisions.md) 2026-08-12 Decision (1), (7), (8);
[`autoinit_v1_search_space.json`](autoinit_v1_search_space.json);
`src/aadistill/autoinit/calibration.py`.

---

## 1. What Phase B is, according to the repository

**The hypothesis.** Phase A searched operator *order* under one fixed calibration
mixture. Phase B tests a different hypothesis: does the AutoInitializer's
preferred composition change when the **calibration distribution** changes? The
two are recorded as separate hypotheses, and Decision (7) of 2026-08-12 states
that *"no detectable order effect under one profile"* does not imply
*"calibration profile is irrelevant"* — **Phase B's trigger is funding plus a
built `calib.reasoning_heavy@v1`, and is independent of Phase A's outcome.**

**The shape actually specified.** One beam-6, two-profile search:

| element | value | source |
| --- | --- | --- |
| profiles | `calib.domain_balanced@v1` + `calib.reasoning_heavy@v1` | proposal §4 |
| decomposed path count | **288** at P=2 (48 at P=1) | `24 × (1+P) × P × P × 1` |
| beam schedule | `beam.delayed_prune` — no pruning at level 0, width 6 after | `SCHEDULE_V1` |
| ranking policy | `PARETO_V1`: equal-domain mean KL, worst-domain KL, critical-token KL; NLL diagnostic only | proposal §4 |
| suite | `state_eval@v1` | unchanged from Phase A |
| priced cost | **$1.90–7.43**, L40S | proposal §5 |
| working storage | **245 GiB**; provision ≥300 GiB container disk | proposal §6 |

**Phase B as specified is a SEARCH ONLY.** The proposal prices a search and
nothing else. It specifies no recovery rungs, no probe seeds, no selection rule,
and no rule for combining its result with Phase A's. See §6.1 — this is the
largest open question, not an oversight this pass may fill in.

**Which operators consume calibration** (this is what makes a second profile
branch at all):

| implementation | `CalibrationNeed` | branches over profiles? |
| --- | --- | --- |
| `width.global_pca_v0` | `ACTIVATION_STATS` | yes |
| `ffn.activation_importance_v0` | `ACTIVATION_STATS` | yes |
| `depth.causal_kl_greedy_v1` | `FORWARD_LOGITS` | yes |
| `composite.stage1` (the incumbent recipe, whole) | `ACTIVATION_STATS` | yes |
| `depth.positional_v0` | `NONE` | no — one invocation against `calib.none` |
| `attention.weight_proxy_v0` | `NONE` | no — one invocation against `calib.none` |

The activation-statistics cache is keyed on *parent artifact digest + profile hash
+ stat spec + adapter version + numerical config* (Decision 8), so two profiles
cannot share a statistics pass, and cross-parent reuse is impossible by
construction rather than by convention.

---

## 2. The blocker: `calib.reasoning_heavy@v1` cannot be built as specified

`calib.reasoning_heavy@v1` is registered as a **reweighted draw from the
`calib.domain_balanced@v1` item pool** (`sources` name `aadistill/e8_calibration_v1`
rev `2026-08-10`; `metadata.pool` names the domain-balanced profile; `sample_rule`
is *"weighted draw from the domain-balanced pool, deterministic by seed"*). It
declares `token_budget = 59_763` and these domain weights.

The pool is the frozen 67-item mixture, and its per-domain prediction positions
are fixed. Re-derived from `artifacts/stage1/e8_calibration_v1/items.jsonl`
(sha256 `c7202338…`, which matches the profile's pinned `items_file_sha256`):

| domain | pool positions | declared weight | required at budget 59,763 | achievable? |
| --- | ---: | ---: | ---: | --- |
| code | 8,622 | 0.20 | 11,952.60 | **no — short by 3,330.60** |
| math | 16,781 | 0.35 | 20,917.05 | **no — short by 4,136.05** |
| rag_multihop | 17,368 | 0.25 | 14,940.8 | yes |
| general | 8,287 | 0.10 | 5,976.3 | yes |
| tool | 8,705 | 0.10 | 5,976.3 | yes |
| **total** | **59,763** | 1.00 | 59,763 | — |

**Two independent failures.**

1. **Per-domain shortfall.** A draw without replacement cannot satisfy the `code`
   or `math` weights at this budget. The binding constraint is `code`: the largest
   budget for which every weight is achievable is
   `min_d (pool_d / w_d) = 8,622 / 0.20 = 43,110` positions — **72.1% of the pool**.

2. **The budget is the entire pool.** `token_budget` (59,763) *equals* the pool's
   total prediction positions exactly. A without-replacement draw of the whole
   budget is therefore the identity: it selects every item, reproducing
   `calib.domain_balanced@v1`'s tokens. Two profiles whose items are identical
   would differ only by `profile_hash`, and would manufacture byte-identical
   states carrying different labels — precisely the failure Decision (1) of
   2026-08-12 eliminated for `CalibrationNeed.NONE` operators, reintroduced
   through the mixture instead of through the operator.

Both failures are pinned by a test
(`tests/autoinit/test_datasets_and_calibration.py::test_reasoning_heavy_v1_cannot_be_drawn_from_its_declared_pool`)
so this cannot silently rot, and so a future edit to the weights or the budget has
to confront it.

**Every resolution moves `profile_hash` off `f4d4ba673ffe…`,** because the hash
covers `sources`, `domain_weights`, `token_budget`, `sample_rule` and `seed`.
`register_profile` refuses a redefinition at the same version by design ("a
changed mixture needs a new version, not a redefinition"), so any fix is
`calib.reasoning_heavy@v2`, and `autoinit_v1_search_space.json` must be
regenerated. That hash appears in exactly one place today
(`autoinit_v1_search_space.json:322`) and is **not** cited by any consumed
authorization, any preregistration, or the frozen-asset verifier — so this is a
tractable change, not a frozen-science violation.

---

## 3. Implementation status

### Already built

Read §3.5 before treating any of this as *proven*: the engine's P=2 path is
written and its branching rule is asserted, but it has never been executed.

* **The search engine is profile-generic by construction.** `SearchConfig.profiles`
  is a tuple; `BeamSearch._expansions` branches every calibration-consuming
  implementation over `sorted(config.profiles)` and invokes every `NONE`
  implementation once against the sentinel; the per-profile calibration cache is
  keyed on `qualified_id`; the statistics cache key includes `profile_hash`. **No
  engine change appears to be required for P=2** — appears, because no P=2 search
  has ever been run.
* **The cost model prices P=2.** `price_search(..., n_profiles=2)` and
  `branching_estimate` already produce the §5 figures.
* **The two-profile branching *rule* is tested** —
  `test_corrections.py::test_two_profiles_do_not_duplicate_the_weight_proxy_expansion`
  asserts the exact expansion set at P=2: 2 sentinel invocations + 4×2 calibrated
  = 10, not 6×2.
* **All source data is present and hash-verifiable** (§4).

### Missing, and required by the design as written

1. **`calib.reasoning_heavy@v1` is not materialized** — `materialized=False`, no
   `items_path`, no `content_sha256`; `resolve()` refuses it, and a test asserts
   that refusal. **Blocked on §2**, not on effort.
2. **No builder exists for it.** `scripts/data/build_e8_calibration.py` is the
   only calibration builder and it hard-codes `POSITIONS_PER_SUBTYPE = 8192` as a
   module constant with no per-domain override, so it cannot express a reweighted
   mixture without a change. Which change depends on §6.2.
3. **The Phase-A search entry point is single-profile.**
   `scripts/autoinit/phase_a_search.py:158-164` builds `profiles=(active_profile,)`
   and passes `calibration_loader=lambda profile: calibration` — a closure that
   **ignores its `profile` argument** and returns one fixed item list. Both must
   change for P=2.
4. **Latent mislabeling on the same seam, reported and deliberately not fixed
   here.** At `phase_a_search.py:154-156`, `active_profile` and `calibration`
   are resolved independently: a caller passing `profile=X` without
   `calibration_items=` gets a run *labelled* X and *fed* `DOMAIN_BALANCED_V1`
   items. No current caller does this — every call site passes both or neither —
   so it is unreachable today and becomes reachable the moment Phase B wires a
   second profile. It is **not** repaired in this pass: `phase_a_search.py` is
   inside `PHASE_A_HARNESS_SOURCE_FILES_V1`, so editing it moves the Phase-A
   harness digest, and moving a digest bound into a closed record buys nothing
   while no run is authorized. Whoever implements Phase B must fix it in the same
   change that adds the second profile.
5. **A two-profile beam has never been RUN — not once, not even at toy scale.**
   The branching rule above is tested by *enumerating* `_candidate_expansions`;
   it never calls `search.run()`. Every executed search on record — both dry-run
   journals (`autoinit_dryrun_{fresh,resume}.json`) and every `make_search`
   caller in the suite except that one enumeration — uses **P=1**. And every test
   passes `calibration_loader=lambda profile: items`, a closure that ignores its
   argument, so **the per-profile loading path has never run with two distinct
   item sets either**. `dry_run_search.py --profiles 2` exists; no record of it
   having been executed exists. By this project's own standing rule — four paid
   pods have died in never-executed lines — a P=2 toy run through the full
   `run()` cycle, with genuinely different items per profile, is a `$0`
   prerequisite before any Phase-B pod, not an optional extra.
6. **No Phase-B session plan or authorization type.** `PHASE_A_PLAN_V1` stages
   0–5 are Phase-A-shaped (rung 1 on sa, rung 2 on sb, conditional sc, five
   searched leaves plus the canonical control). `PhaseAAuthorization` refuses a
   stage outside its authorized set and cannot express a follow-on.
7. **No Phase-B preregistration.** `autoinit_phase_a_preregistration.json`
   carries `active_calibration_profile` in the **singular** and contains zero
   occurrences of `reasoning_heavy` or `phase_b`. Under the project's own rule a
   ranking policy, halving plan and selection rule must be frozen *before* the run
   they judge — so a preregistration is a prerequisite, and it cannot be written
   until §6 is decided.

---

## 4. Source and asset availability — all present, all $0-verifiable

| asset | path | state |
| --- | --- | --- |
| calibration pool | `artifacts/stage1/e8_calibration_v1/items.jsonl` | present, 67 items; sha256 `c7202338…` **matches** the pinned `items_file_sha256`; token content re-derives to `d65c1f40…` |
| its leakage proof | `.../leakage.json`, `manifest.json` | present |
| session corpus (a re-draw's source) | `artifacts/stage3/corpus_v2/sessions.jsonl` | present, 74 MB |
| rung pack (exclusion source) | `artifacts/stage3/ladder_uniform_probe` | present, 20 MB |
| general-text docs | `artifacts/stage1/e8_calibration_v1/general_docs.jsonl` | present, 204 KB |
| teacher tokenizer | HF cache, `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d…` | present at the pinned revision |
| state-eval suite | `artifacts/stage1/state_eval_v1` | present; `verify_frozen_assets.py` → `passed: true, problems: []` |
| recovery battery | `artifacts/stage3/recovery_search_v2` | present; same verifier |

A re-draw from the session corpus is therefore mechanically possible at zero cost
on CPU — the builder needs only the tokenizer, not teacher weights. Whether it is
*scientifically* the right move is §6.2.

---

## 5. What must remain immutable from Phase A

* **All five searched leaves are retained off-pod**, `1.2 GiB` each, at
  `/home/ecs-user/aad-artifacts/autoinit/phase_a/<canonical_id>/`. Both finalists
  were re-hashed in this pass and match `leaf_retention.json` exactly:
  `cca699c93f34…` → `db056398…`, `85bde4ded2c3…` → `0a9a5a78…`.
* **⚠ None of the five leaf directories contains a tokenizer** — only
  `config.json`, `generation_config.json`, `model.safetensors`. This is the
  failure mode this project has hit twice: `AutoTokenizer.from_pretrained` on such
  a directory returns a 1-token vocabulary instead of raising. Any Phase-B or
  Stage-1-NLL consumer of these leaves must declare its tokenizer source
  explicitly and assert the loaded identity.
* `logs/autoinit_recovery_continuation_attempt7/` — HISTORICAL, including all 11
  probe records. Do not rewrite.
* The frozen identities in `current_state.json.frozen`: science plan
  `02be33b9…`, session plan `9377a2dc…`, Stage-3 evaluation protocol `250f72ef…`,
  equivalence interval `0.011695296982299022`, feasibility floor `0.30`, seeds
  sa/sb/sc, and `calib.domain_balanced@v1` itself.
* The two **historical permanent Stage-3 controls** `preflight_ctl_r0860k_{sa,sb}`
  — imported, never retrained — remain distinct from the **Phase-A canonical
  initialization control** `qwen3_0p6b_init_v0`, which did receive matched fresh
  Phase-A selection probes. Do not collapse the two in Phase-B code or prose.

---

## 6. Open scientific choices — for the reviewer, not for this pass

### 6.1 Does Phase B include recovery probes, and how is the final initialization chosen?

**Nothing in the repository answers this.** The proposal prices Phase B as a
search and stops. But Phase A's selection rule was *behavioural* —
`usable_rollout_rate` as a feasibility gate, then `correct_overall` as the
objective — and the project's own evidence (E8a's 3.11× better operator objective
initializing 2.8 nats worse; E7's −5.22 nat NLL swing moving behaviour by
+0.0000) is the reason step-0 metrics were denied selecting authority.

So a search-only Phase B yields step-0 fidelity numbers for a new profile that
**cannot be compared against Phase A's behavioural result on the same scale**.
Compounding it, Phase A terminated at `unresolved_equivalence` with `winner:
None`, so there is no Phase-A winner for a Phase-B result to be combined *with*.

The reviewer has to decide what Phase B's output is for. Neither reading is free:
search-only is cheap ($1.90–7.43) and produces a quantity that by this project's
own evidence does not settle a selection; search-plus-recovery settles it and
roughly quadruples the cost.

### 6.2 How is the reweighted mixture actually to be drawn?

§2 shows the registered specification is unsatisfiable. Four resolutions exist;
each has a different scientific meaning and each requires `@v2`:

| option | what it means | cost of the choice |
| --- | --- | --- |
| **(a) lower the budget** to ≤ 43,110 positions | a genuine subset of the leakage-checked pool, weights honoured exactly | the two profiles then differ in *size* as well as in *distribution*, confounding the comparison |
| **(b) draw with replacement** to reach 59,763 | duplicated items reweight the mixture | an item counted twice doubles its contribution to `XᵀX`; this is a change to statistics semantics and must be decided, not assumed |
| **(c) re-draw from the session corpus** with per-domain position budgets | a genuinely reasoning-weighted mixture at full size | changes `sources`; **requires a fresh leakage proof** — the current spec declares `leakage_exclusions=()` and no proof path, which is coherent *only* under the subset reading; also needs a builder change |
| **(d) weight at the statistics level**, not by resampling | not a mixture change at all | no such mechanism exists in any operator; would be new science |

Note that (c) is what the declared *weights* suggest and (a)/(b) are what the
declared *sources* suggest. The registered profile is internally inconsistent
about which it is, which is why this needs a decision rather than a patch.

### 6.3 Is Phase A's search reused, or re-run?

At P=2 the beam is joint: `domain_balanced` states compete against
`reasoning_heavy` states for the same six slots, so the domain-balanced arm's
pruning differs from Phase A's and its leaves are **not** a superset of Phase A's
five. Reusing Phase A's leaves and searching only the new profile is a different
experiment from the two-profile search that is priced. Neither is recorded as
chosen.

---

## 7. Cost and storage, if Phase B runs as specified

Mechanically derived from `autoinit_v1_search_space.json` (L40S at $0.99/h,
88.83 TFLOP/s measured from E8a). **These are inputs to an authorization, not an
authorization.**

**Search, P=2, beam 6, warmup 1:**

| quantity | value |
| --- | --- |
| cost | **$1.8950 – $7.4349** |
| wall-clock | 1.91 – 7.51 h |
| states materialized | 74 – 114 |
| leaves | 8 – 20 |
| peak working storage | **244.9 GiB** → provision ≥300 GiB container disk |
| total bytes written | 292.9 GiB |
| retained after pruning | 40.7 GiB |
| peak GPU resident | 14.3 GiB — an L40S is sufficient; no A100 needed |

The range is still the **unmeasured activation-statistics GPU/CPU split**, the
same quantity that has made every AutoInitializer cost a range since 2026-08-12.
It remains unmeasured.

**Recovery probes are not specified.** *If* Phase B were given Phase-A's halving
shape (5 searched leaves + the control on sa; 2 survivors + the control on sb),
the priced anchors give **$14.733** plus a conditional third seed at **$4.911** —
but adopting those numbers would be adopting a design decision that §6.1 says is
open.

**Indicative totals** (search + $3.00 setup/redraw reserve, the reserve Phase A
used; setup time has varied 30× across this project's sessions):

| scenario | expected | hard |
| --- | ---: | ---: |
| search only | ~$4.90 | ~$10.44 |
| search + Phase-A-shaped recovery, no third seed | ~$19.63 | ~$25.17 |
| search + recovery + conditional third seed | ~$24.54 | ~$30.08 |

**The binding per-launch ceiling must be derived by the pricing and authorization
code once a Phase-B plan exists** — as Phase A's $23.0484 was — not from this
table.

### Budget consequence

Actual cumulative spend is **$230.0350** against a **$234.00** cap; **$3.9650**
remains, and **that funds no paid session of any kind.** Against actual spend, the
scenarios above imply a new cumulative cap of roughly:

| scenario | cap implied at the hard figure |
| --- | ---: |
| search only | ~$240.47 |
| search + recovery | ~$255.20 |
| search + recovery + third seed | ~$260.11 |

**A new explicit maintainer cumulative-budget decision is required before any
Phase-B GPU work.** No grant or authorization was issued by this pass, and none
should be until §6 is settled — pricing a design that is not yet chosen would
produce a number that means nothing.

---

## 8. Ordered readiness checklist

| # | item | status |
| --- | --- | --- |
| 1 | resolve §6.2 — how the reweighted mixture is drawn | **blocked on reviewer** |
| 2 | resolve §6.1 — search-only vs search+recovery, and the final-selection rule | **blocked on reviewer** |
| 3 | resolve §6.3 — reuse Phase A's leaves, or re-run jointly | **blocked on reviewer** |
| 4 | build `calib.reasoning_heavy@v2` and pin its content hash | $0, blocked on 1 |
| 5 | fresh leakage proof, if option (c) is chosen | $0, blocked on 1 |
| 6 | two-profile support in `phase_a_search.py`, incl. the §3.4 mislabeling fix | $0, blocked on 2 |
| 7 | **run a full toy P=2 beam through `search.run()`** with genuinely different items per profile — never once executed (§3.5) | $0, **not blocked**, and required before any Phase-B pod |
| 8 | a Phase-B session plan and authorization type | $0, blocked on 2 |
| 9 | Phase-B preregistration, frozen before the run it judges | $0, blocked on 1–3 |
| 10 | measure the activation-statistics GPU/CPU split | collapses the cost range; still unmeasured |
| 11 | new cumulative-budget decision | **maintainer** |
| 12 | reviewer GO | **reviewer** |

Items 1–3 are genuinely upstream: four of the remaining `$0` items cannot be
written without them, and writing them anyway would be inventing Phase-B science
in an implementation pass. **Item 7 is the exception — it is unblocked and can be
done now**, and it should be, because "the engine supports P=2" currently rests
on a branching-rule enumeration rather than on an executed search.
