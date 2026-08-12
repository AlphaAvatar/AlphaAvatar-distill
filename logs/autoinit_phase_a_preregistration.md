# Phase A preregistration — AutoInitializer 4B → 596M

**Status: DRAFT, NOT AUTHORIZED. No compute has been launched.** Machine-readable
companion: [`autoinit_phase_a_preregistration.json`](autoinit_phase_a_preregistration.json),
sha256 `e6e674034eb957031584b8701d3721479e49cc5d78e2e18b622b85bb11421668`,
regenerable with `scripts/autoinit/write_preregistration.py`.

Two numbers are deliberately **absent** and marked `PENDING_MICRO_PREFLIGHT`. The
rules that consume them are frozen here; the numbers are measured on the canonical
control before any searched candidate is probed. A rule frozen now with a number
filled later is preregistration. A number chosen after seeing candidates is not.

---

## 1. Question and endpoint

Does conditional operator order plus beam selection produce a better 596M
initialization than the incumbent fixed recipe?

**A legitimate outcome is "the canonical control wins; AutoInitializer v1 did not
improve recovered behaviour."** The final selector treats the control as an
ordinary candidate, so that conclusion is mechanically reachable.

## 2. Fixed inputs

| | |
| --- | --- |
| teacher | `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d`, 4,022,468,096 params |
| target | 1024 hidden, 28L, FFN 3072, 16Q/8KV, tied — 596,049,920 params |
| adapter | `qwen3.dense_v1` |
| operator ledger | `configs/autoinit/operator_ledger.json`, six immutable ids |
| active calibration | `calib.domain_balanced@v1`, content `d65c1f40…` |
| no-calibration sentinel | `calib.none@v1` |

## 3. Search space

```
24 orderings × (1 + P) DEPTH × P WIDTH × P FFN × 1 ATTENTION,  P = 1
```

**48 decomposed paths.** Operators declaring `CalibrationNeed.NONE`
(`depth.positional_v0`, `attention.weight_proxy_v0`) are invoked once against the
sentinel and do not branch over profiles.

`composite.stage1_sandwich_v0` is kept **separate**: it reaches the target in one
step, does not compose with the four structural kinds, and is a searchable leaf.
**It is not the canonical control** — see §6.

Materialized states **39–56**, complete leaves **7–13**.

## 4. Beam

| | |
| --- | --- |
| schedule | `beam.delayed_prune@v1` — `warmup_levels=1`, `width=6` |
| level 0 | **no quality pruning**; all five first-step hypotheses survive |
| policy | `beam.pareto_multi_objective@v2`, ε-Pareto |
| objectives | equal-domain mean teacher KL · **worst-domain** teacher KL · critical-token KL |
| ε | 1e-4 per objective |
| diversity | lineage rotation across fronts, before the state-id tie-break |
| NLL | **diagnostic only** — not an objective, not a tie-break key |

**ε justification.** One unchanged checkpoint scored 12 times through the full
materialize → reload → measure cycle produced an objective range of **exactly
0.0** on CPU. 1e-4 is therefore above the deterministic path's noise. **This does
not establish GPU repeatability**, where reduction order is not guaranteed across
launches; the micro-preflight measures the same quantity on an L40S with the real
teacher and suite, and ε is confirmed or reset from that number *before* any
candidate is ranked.

## 5. State evaluation

`artifacts/stage1/state_eval_v1`, content sha256
`a1197205e43aad0e71c0e1bb436ee7babba3b5d8bb25b9c4d5c464f659db20fc`.

* **80 items, 74,022 prediction positions**, five domains
  (`general`, `reasoning_math`, `rag_multihop`, `code`, `tool`) over seven
  sub-types.
* **Critical-token classes** feeding `state.critical_token_kl`: `think_close`,
  `eos`, `final_answer`, `tool_close` — the narrow, decision-bearing ones.
  `assistant` and `reasoning` are recorded as diagnostics and excluded from the
  aggregate: they cover most supervised positions and would drown out exactly the
  rare tokens the metric exists to watch. Position counts 70 / 70 / 10,058 / 28,
  each above a 25-position floor set before the data was seen.
* **Reference is the original teacher, recomputed per candidate.** Caching the
  reference would be 33.8 GiB at the real vocabulary; recomputing is ~6.9 s per
  candidate on an L40S.
* Tokenizer, chat template and the four special-token ids are pinned by hash.

## 6. Recovery

```
rung 1   5 searched leaves + the canonical control        seed sa   6 probes
rung 2   the control + the best 2 searched leaves         seed sb   3 probes
rung 3   only finalists inside the equivalence interval   seed sc   conditional
```

**The canonical control is `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`,
injected by its frozen sha256 `86fbba78…`.** A composite re-executed inside the
search is built from *this run's* calibration statistics; the canonical checkpoint
was built from the original Stage-0 statistics. Same algorithm, different input,
different weights — and every existing behaviour number in this project belongs to
the latter.

* The control **advances to rung 2 unconditionally** and consumes no survivor slot.
* The control is **exempt from the feasibility gate** — a baseline that fails the
  floor is a finding about the floor or the baseline, not a reason to delete the
  comparison.
* The control is **eligible to win** the final.
* The third seed is offered to **every** tied finalist, control included.

### Selection: constraint, then objective, never a weighted sum

1. `usable_rollout_rate` over **all 190** prompts — a *feasibility* gate. Blind to
   correctness by construction, so it may exclude and never rank.
2. `correct_overall` over the **150 scorable** prompts — the capability objective.
3. `correct_given_usable` — reported to explain a ranking, never to change one.

The plan constructor refuses a configuration where the gate and the objective are
the same metric.

### Seed aggregation — pooled counts

```
correct_overall      = Σ correct_s / Σ n_s
usable_rollout_rate  = Σ usable_s / Σ n_s
correct_given_usable = Σ correct_s / Σ usable_s
```

Never a mean of per-seed rates. For a conditional rate the two differ: a seed with
30 usable rollouts would otherwise weigh the same as one with 90. The definition is
versioned (`pooled_counts@v1`) and participates in the plan hash. The same rule
extends to `sc` across all completed seeds.

### Recovery-search battery

`artifacts/stage3/recovery_search_v1`, content sha256
`a1b22778b00d95b6aba358c14a5af5b559fd807bb371c92131eacca59479f323`.
**190 prompts, 150 scorable.**

| set | domain | n | scorable |
| --- | --- | ---: | --- |
| gsm8k | reasoning_math | 30 | yes |
| math_verified | reasoning_math | 30 | yes |
| multihop | rag_multihop | 30 | yes |
| rag | rag_multihop | 30 | yes |
| knowledge | general | 30 | yes |
| code | code | 20 | **behaviour only** |
| tool | tool | 20 | **behaviour only** |

Code and tool have no frozen scorer, so they contribute stability and failure
diagnostics and are excluded from correctness by construction. Inventing an
unvalidated code executor or call matcher would put an untested scorer on the
selection path. **Recorded limitation: this battery cannot detect a candidate that
trades code or tool capability for math.**

## 7. Preregistered thresholds

Derived in [`autoinit_threshold_characterization.json`](autoinit_threshold_characterization.json).

**Equivalence interval — 0.045.** Two binomial standard errors of
`correct_overall` at the control's historical rate (0.1867) on the pooled scorable
denominator (150 × 2 = 300). One SE would call a difference decisive that a rerun
could reverse; two SE is the smallest interval this denominator supports.

**Feasibility floor — rule frozen, number pending.**
`max(0.30, control_usable_pooled − 3·SE)` on the pooled all-prompt denominator
(190 × 2 = 380). The absolute term guards "cannot hold a rollout at all"; the
relative term guards against a candidate much less stable than the incumbent
without requiring parity, which would make feasibility a second ranking. At the
historical prior (0.7300) the floor would be **0.6617**; the actual control rate on
*this* battery is unmeasured and is a preflight input.

**Catastrophic per-capability floor — rule frozen.** A candidate is excluded if any
scorable set's usable rate is below 0.10 while the control's is above 0.40 on that
set. The pooled rate can hide a total collapse on one capability.

Context: the behaviour metric's seed-only spread is **0.1290**, far larger than the
equivalence interval — which is why seeds are pooled rather than compared, and why
a third seed exists.

## 8. Data hashes and isolation

| role | asset | identity |
| --- | --- | --- |
| OPERATOR_CALIBRATION | `e8_calibration_v1` | content `d65c1f40…` |
| INITIALIZER_STATE_EVAL | `state_eval_v1` | content `a1197205…` |
| RECOVERY_SEARCH | `recovery_search_v1` | content `a1b22778…` |
| FINAL_PROMOTION | `battery_v2` | mask `d6e24e0b…` — **isolated from the search** |
| RECOVERY_TRAINING | `ladder_uniform_probe` @ 0.86M | pack `6f324cb0…` |

[`autoinit_role_isolation.json`](autoinit_role_isolation.json): **passed, complete,
0 exact overlaps** across all ten role pairs, every pair compared on a real shared
identity type. Residual near-duplicates: 17 by a strict shingle rule, **2** under
the rule the builders enforced — both formulaic multi-turn `glaive` tool dialogues.
Recorded rather than chased.

## 9. Artifact identity and resume

Checkpoint identity is an **artifact digest** over every sorted weight shard, the
shard index, the config, the architecture signature and the tokenizer. Metrics bind
to that digest. Frozen single-file hashes stay checkable via `single_shard_sha256`,
and a sharded rebuild reports as a different *layout*, not corruption.

Resume restores a state only when the artifact re-derived from disk matches the
journal **and** the recorded evaluation's `suite_hash` matches this run's suite.
State identity is the path and does not include the suite, so without that second
check a journal measured on different questions would be adopted silently.

## 10. Budget and storage

Search only, L40S at $0.99/h: **$0.93 – $3.57**, 0.94 – 3.60 h. The range is the
unmeasured activation-statistics GPU/CPU split.

Storage: **peak working 105.9 GiB**, total written 135.4 GiB, retained 35.9 GiB,
peak GPU resident 14.3 GiB. **Provision ≥ 150 GiB of container disk.**

Recovery probes are priced separately in
[`autoinit_pilot_proposal.md`](autoinit_pilot_proposal.md). **These search-only
figures are not the final authorization numbers** — those are recomputed after the
micro-preflight.

## 11. Pending before launch

1. canonical control `usable_rollout_rate` on `recovery_search_v1`;
2. canonical control `correct_overall` on `recovery_search_v1`;
3. canonical control per-set usable rates;
4. GPU state-evaluator repeatability — confirms or resets beam ε;
5. activation-statistics GPU/CPU split — collapses the cost range.

All five are micro-preflight outputs. See
[`autoinit_micro_preflight_plan.md`](autoinit_micro_preflight_plan.md).
