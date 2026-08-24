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

## 2A. Option (b) resolved mechanically — 2026-08-25

The reviewer selected **option (b)**: a versioned `calib.reasoning_heavy@v2` doing
deterministic **with-replacement** reweighting of the same frozen 67-item pool,
preserving the 59,763-position budget, the declared domain weights, the source
support and the existing leakage-safe support; no silent truncation, no silent
approximation; and — before materializing — establish whether the whole-item
multiset can realize the target allocation, reporting the minimum achievable
deviation and proposing the rounding rule.

**Exact realization is impossible**, for a reason prior to any algorithm: the
targets are not integers. `0.35 × 59,763 = 20,917.05`; four of the five domain
targets have a fractional part. No whole-item multiset can sum to a non-integer.

### The proposed rule

**R1 — domain apportionment.** Largest-remainder (Hamilton) over the declared
weights; remainder ties broken by ascending domain id. This is the standard
integer apportionment and it sums to the budget **exactly**:

| domain | exact target | quota | deviation |
| --- | ---: | ---: | ---: |
| code | 11,952.60 | 11,953 | +0.40 |
| general | 5,976.30 | 5,976 | −0.30 |
| math | 20,917.05 | 20,917 | −0.05 |
| rag_multihop | 14,940.75 | 14,941 | +0.25 |
| tool | 5,976.30 | 5,976 | −0.30 |
| **total** | 59,763 | **59,763** | 0 |

**R2 — unreachable domain quota.** Four of the five quotas are exactly realizable
by a whole-item multiset. **`code` = 11,953 is not**; the nearest reachable values
are 11,952 and 11,954, both at distance 1. Take the nearest, **ties toward the
lower**, and transfer the difference to the reachable domain with the most
negative apportionment remainder (ties by ascending id). That is `general`
(−0.30, tied with `tool`, resolved by name), and 5,977 is reachable.

**Result: `code` 11,952 · `general` 5,977 · `math` 20,917 · `rag_multihop` 14,941
· `tool` 5,976 = 59,763 exactly.** Maximum domain deviation from the exact
fractional target is **0.70 positions — 1.17 × 10⁻⁵ of the budget**. That is the
minimum achievable: it cannot be zero, and no other integer assignment is closer.

**R3 — sub-type apportionment.** Within each domain, split the quota across its
sub-types by largest remainder over **the pool's own sub-type position shares**.
This holds within-domain composition fixed so that only the domain mix — the
thing the hypothesis is about — changes. Without it a plain uniform draw drifts
`math/gsm8k` by **+19.4 points** (0.5049 → 0.6993), which would confound the
comparison with a second, undeclared change.

**R4 — unreachable sub-type quota, repaired INSIDE its domain.** `multihop_qa` =
7,526 is unreachable. The repair transfers the difference to another sub-type of
**the same domain**, so every domain weight stays exactly as R1/R2 set it and the
deviation is confined. `code`, `general`, `tool_calling` are single-sub-type;
`gsm8k`/`openmath` are exact.

**R5 — realization.** Among all exact whole-item multisets summing to a sub-type
quota, take the one **maximizing distinct items used**, then minimizing multiset
size, then lexicographically smallest by sorted `item_id`. Deterministic without a
PRNG, and it maximizes source support by construction.

### The one open choice: `multihop_qa`

`multihop_qa` has 5 items, all 1,675–1,835 positions, so its reachable sums are
sparse and support is tightly coupled to deviation. The full frontier, with
`rag_evidence` absorbing the remainder so `rag_multihop` stays at 14,941:

| multihop_qa | deviation | % of rag_multihop | distinct items used | rag_evidence | its support |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7,340 | −186 | −1.24% | **1 / 5** | 7,601 | 10 / 13 |
| 7,296 | −230 | −1.54% | 2 / 5 | 7,645 | 10 / 13 |
| 7,205 | −321 | −2.15% | 3 / 5 | 7,736 | 11 / 13 |
| **7,074** | **−452** | **−3.03%** | **4 / 5** | **7,867** | 10 / 13 |

**Recommendation: 7,074.** The nearest-reachable rule alone would pick 7,340,
which realizes the sub-type as **four copies of one session** — rank-deficient
input to an activation-statistics operator, and a 5× narrower sample of the
capability the reasoning-heavy profile exists to emphasise. The cost is 452
positions inside one domain, **0.76% of the budget**, with every declared domain
weight still exact to 0.70 positions. A 3% shift inside `rag_multihop` is far
below anything this search can resolve; one session standing in for a sub-type is
not.

**This is the choice to confirm before materialization.** Under it the sub-type
quotas are `code` 11,952 · `general` 5,977 · `gsm8k` 10,560 · `openmath` 10,357 ·
`multihop_qa` 7,074 · `rag_evidence` 7,867 · `tool_calling` 5,976 = **59,763**.

### Why full source support is not attainable, and is not a defect

Requiring every pool item to appear at least once is **impossible for every
down-weighted domain**, because down-weighting *means* dropping sessions:

| sub-type | overshoot if every item must be used |
| --- | ---: |
| tool_calling | +45.67% of its quota |
| general | +38.65% |
| multihop_qa | +19.20% |
| rag_evidence | +13.39% |

Realized support under R5 is `code` 4/6, `general` 12/16, `gsm8k` 9/10,
`openmath` 3/5, `multihop_qa` 4/5, `rag_evidence` 10/13, `tool_calling` 9/12 —
**51 of 67 distinct sessions across 62 draws**, and the realization has been
constructed and checked to sum to 59,763 exactly. "Source support" is preserved in the sense the
profile declares it (all five sources, all seven sub-types, every domain present);
it is not, and cannot be, every individual session.

---

## 2B. v2 materialized — 2026-08-25

The reviewer approved `multihop_qa` = 7,074 and required two corrections before
materialization. Both are in, and the profile is built.

**The approved rationale, stated narrowly.** 7,340 does **not** make the whole
activation-statistics input rank-deficient. It concentrates the `multihop_qa`
capability slice into four copies of one session — an unnecessary item-level
concentration confound. 7,074 removes it for 452 positions absorbed by
`rag_evidence` *inside* `rag_multihop`, leaving every domain quota unchanged.

**Correction 1 — the seed is wired into the tie-break, and its real reach is
recorded.** R5's final tie-break is now the seed-derived order
`sha256(f"{seed}:{item_id}")` ascending, not lexicographic, and the complete R1–R5
semantics are stated verbatim in `sample_rule` **before** the profile hash is
computed — so the hash covers the procedure rather than a label for it.

That is the structural fix. The honest result is that **on this pool the seed
never binds**: every sub-type's optimum is unique — verified by brute-force
enumeration, not merely by the dynamic program that produces it — so no tie is
ever exercised and a different seed yields the same bytes. The seed identifies the
*rule instance*, not the sampled bytes. Asserted as a test so it cannot rot into a
claim, and it is precisely why a preregistration must bind `content_sha256`.

**Correction 2 — the two repair levels are deliberately different rules.**
A domain's deviation distorts the weights the profile exists to declare, so R2
minimizes it (nearest reachable, ties lower). A sub-type's deviation is absorbed
by a sibling and moves no domain weight, so R4 spends it on support (most distinct
sessions at or below the quota, ties by smallest deviation). A single unified rule
was tried and rejected on measurement: it moves quotas that are currently exact —
`gsm8k` 10,560→10,556, `openmath` 10,357→10,347, `code` 11,953→11,939 — trading
real deviation for support where the arithmetic never forced a choice.

### The built mixture

| | |
| --- | --- |
| `profile_hash` | `6c67b8dff0e81a775371a62b1b293677e45c173f52aee119d036c95081782ddd` |
| `content_sha256` | `cdb2838946b9355294406d2bc398bc8390306ced84a35b09900f45a033ccc370` |
| `items_file_sha256` | `a2ff8c92c16aaf5c178db160690430f4972216d45a57ab0025835ecd0ca41fc4` |
| draws / distinct sessions | 62 over **51 of 67** |
| positions | **59,763 exactly** |
| max domain deviation | **0.70 positions** (1.17 × 10⁻⁵ of the budget) |

Domain quotas — `code` 11,952 · `general` 5,977 · `math` 20,917 · `rag_multihop`
14,941 · `tool` 5,976. Sub-type quotas — `code` 11,952 · `general` 5,977 ·
`gsm8k` 10,560 · `openmath` 10,357 · **`multihop_qa` 7,074** · **`rag_evidence`
7,867** · `tool_calling` 5,976.

Leakage is **inherited, not re-proved**: every token is a token of the source
mixture drawn with replacement, and reweighting a leakage-checked pool cannot
introduce a leak the pool does not have.

`manifest_sha256` is deliberately *not* pinned in the profile: it covers
`generated_utc` and so moves on every rebuild. `content_sha256` and
`items_file_sha256` are stable and both reproduce from the pool.

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

### Status against the design, updated 2026-08-25

1. **`calib.reasoning_heavy@v2` is not materialized.** v1 is unbuildable (§2);
   v2's apportionment is derived and verified (§2A) and its realization has been
   constructed and checked to sum to 59,763 exactly. **Held at the reviewer's
   checkpoint**: the `multihop_qa` support/deviation choice is proposed, not
   taken.
2. **No builder exists for it.** `scripts/data/build_e8_calibration.py` is the
   only calibration builder and hard-codes `POSITIONS_PER_SUBTYPE = 8192` with no
   per-domain override. Blocked on 1 only.
3. ~~**The Phase-A search entry point is single-profile.**~~ **DONE.**
   `run_phase_a_search` now takes `profiles=` alongside `profile=`, and
   `resolve_profiles` / `build_calibration_loader` resolve the searched profiles
   and their items **together**. The loader answers for the profile it is *asked*
   about: a mapping keyed by `qualified_id`; a bare sequence only when exactly one
   profile is active, bound to it, raising if asked about another; or — when
   omitted — each profile resolving itself from disk, hash-verified. 8 tests,
   4 mutations.
4. ~~**Latent mislabeling on the same seam.**~~ **FIXED by 3.** `active_profile`
   and `calibration` used to be resolved independently, so `profile=X` without
   `calibration_items=` produced a run *labelled* X and *fed* the domain-balanced
   mixture. There is no longer a path on which the label and the mixture can
   disagree.
   **3 and 4 move the Phase-A harness digest** — `scripts/autoinit/phase_a_search.py`
   is in `PHASE_A_HARNESS_SOURCE_FILES_V1`. Expected, authorized, and reported. No
   frozen *science* identity moves.
5. ~~**A two-profile beam has never been RUN.**~~ **DONE — and it found a
   defect.** `tests/autoinit/test_two_profile_search.py` executes a full P=2
   `search.run()` with genuinely different tokens per profile. On its first
   execution it raised inside the beam: the engine calls `calibration_loader`
   with the `calib.none@v1` sentinel, whose `resolve()` raises by design. Every
   loader ever written here was `lambda profile: items`, which ignores its
   argument and answered anyway, so nothing had ever exercised it; a correct
   Phase-B loader delegating to `profile.resolve()` would have raised
   `CalibrationError` there instead — **on a paid pod, mid-search**. Fixed in
   `search.py` by short-circuiting the sentinel. **No identity moves**:
   `n_calibration_items` is the only thing derived from that list, `_expand_one`
   explicitly excludes it from `config_hash`, and both `CalibrationNeed.NONE`
   implementations ignore `config`. 5 tests, 6 mutations; 1192 autoinit/pod tests
   pass unchanged.
6. **No Phase-B session plan or authorization type.** `PHASE_A_PLAN_V1` stages
   0–5 are Phase-A-shaped and `PhaseAAuthorization` cannot express a follow-on.
   Next in sequence, after 1.
7. **No Phase-B preregistration.** It must bind `calib.reasoning_heavy@v2`'s
   profile hash, so it follows 1 rather than preceding it. The terminal procedure
   it will freeze is the reviewer's cross-phase behavioural selection, already
   encoded mechanically in `scripts/autoinit/price_phase_b.py`.
8. ~~**The paid work is unpriced.**~~ **DONE.** See §7.

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

## 7. Cost and storage — repriced 2026-08-25 against VERIFIED reuse

Regenerable: `PYTHONPATH=src python scripts/autoinit/price_phase_b.py`
→ [`autoinit_phase_b_pricing.json`](autoinit_phase_b_pricing.json). It **fails
closed** if the reuse record is missing, unverified, or describes different probe
bytes than the ones on disk. **These are inputs to an authorization, not one.**

**Search, P=2, beam 6, warmup 1, L40S:** $1.8950–$7.4349, 1.91–7.51 h, 74–114
states, 8–20 leaves, **244.9 GiB** peak working storage (provision ≥300 GiB),
14.3 GiB resident. The spread is the unmeasured activation-statistics GPU/CPU
split, which is **not** to be closed by a separate paid session.

**Probes.** 8 candidates at sa (5 Phase-B leaves + 2 Phase-A finalists + the
control). Reuse is verified for 8 historical probes, so:

| scenario | probes | total |
| --- | ---: | ---: |
| **low** — reuse holds, the priors survive sb, no tie-break | 5 | **$13.0800** |
| **hard, reuse holds** — both survivors new, tie-break fires | 10 | **$26.8049** |
| **hard, no reuse** — Stage 0 finds the runtime not comparable | 14 | **$33.3529** |

**`low` is a floor, not an expectation.** No expected-value assumption over
survivor identity or tie-break probability is defined anywhere, so nothing here is
called "expected".

**`hard_no_reuse` is what an authorization must cover.** The comparability
precondition is only testable at Stage 0 of the run itself, and if it fails
*every* historical probe is lost at once rather than some of them.

Two facts that drive the arithmetic: the control has **no `sc`** on record, which
is why worst-case `sc` costs one probe more than `sb`; and the three unadmitted
Phase-A leaves hold verified `sa` probes and are retained off-pod, so admitting
them would cost nothing at `sa` — surfaced, not acted on.

### Budget consequence

Actual cumulative spend **$230.0350** against a **$234.00** cap; **$3.9650**
remains and funds nothing. Against actual spend, `hard_no_reuse` implies a new
cumulative cap near **$263.39**. **A new explicit maintainer cumulative-budget
decision is required before any Phase-B GPU work.**

## 8. Ordered readiness checklist — updated 2026-08-25 (second pass)

| # | item | status |
| --- | --- | --- |
| 1 | §6.2 — how the reweighted mixture is drawn | **DONE.** Option (b); `multihop_qa` = 7,074 approved; both corrections in (§2B) |
| 2 | §6.1 — the final-selection rule | **DECIDED:** cross-phase behavioural selection |
| 3 | §6.3 — reuse Phase-A leaves, or re-run jointly | **DECIDED:** full fresh joint P=2 beam |
| 4 | a full toy P=2 `search.run()` | **DONE** — found a pod-fatal defect (§3.5) |
| 5 | two-profile entry point + mislabeling fix | **DONE** (§3.3, §3.4) |
| 6 | **strict historical probe reconstruction** | **DONE — all 11 probes verify.** `verify_historical_probe_reuse.py` → `autoinit_historical_probe_reuse.json`. One leg is **not** closable at `$0` and is reported: Phase B's runtime comparability, a Stage-0 precondition |
| 7 | materialize and hash `calib.reasoning_heavy@v2` | **DONE** (§2B) |
| 8 | mechanically reprice the missing paid work | **DONE** (§7), now conditional on the verified reuse and fail-closed without it |
| 9 | a Phase-B session plan and authorization type | `$0`, **next** |
| 10 | Phase-B preregistration | `$0`, follows 9. Must bind **both** v2's `profile_hash` *and* its `content_sha256` (plus `items_file_sha256`) — `profile_hash` does not identify the sampled bytes, and on this pool the seed does not either |
| 11 | final mechanical repricing | `$0`, follows 10 |
| 12 | activation-statistics GPU/CPU split | **not to be measured by a separate paid session.** Keep the conservative bound unless it falls out of the Phase-B run itself |
| 13 | new cumulative-budget decision | **maintainer** |
| 14 | reviewer GO | **reviewer** |

Items 1–8 are complete. 9–11 are the remaining `$0` work and are strictly
ordered: the preregistration binds the plan and the mixture identity, and the
final repricing follows whatever the plan fixes.
