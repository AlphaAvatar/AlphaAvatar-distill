# Metric contract

What each reported number IS: where its definition comes from, what it scores,
its numerator and denominator, how it aggregates, and which stage reports it.
Maintainer decision 2026-11-04 (AGENTS.md P10.3): an accuracy, pass rate, task
success, pass@k or pass^k of a public benchmark takes that benchmark's official
definition, and an agent never chooses its meaning, numerator or denominator.
Project-internal metrics are allowed. They are reported under their own names
and never presented as a benchmark score. No two metrics are pooled into one
number under either name.

Frozen scorers and measurement identities are not rewritten by this contract.
Where it found an implementation that reported more than it checked, the fix is
noted under *Changes*. Historical records keep the identity they were scored
under.

## 1. Benchmark-defined metrics

A number below is quoted as the benchmark's ONLY when it comes from the
benchmark's own dataset, task interface, scorer, aggregation, attempt budget,
statistical unit and version. The definitions themselves live with the
benchmark: run its official harness, record its version, and do not re-derive
them here.

| benchmark | official metrics (named by the benchmark) | source |
| --- | --- | --- |
| BFCL V4 (UC Berkeley) | AST accuracy, executable accuracy, relevance / irrelevance detection, multi-turn and agentic categories, the leaderboard's overall accuracy | <https://gorilla.cs.berkeley.edu/leaderboard>, <https://proceedings.mlr.press/v267/patil25a.html> |
| τ-bench / τ²-bench | task success (a trial's reward), pass^k: the chance that all k independent trials of a task succeed | <https://arxiv.org/abs/2406.12045> |
| AgentBench | each environment's own metric, and the benchmark's own overall score | <https://github.com/THUDM/AgentBench> |
| GAIA | final-answer correctness under its scorer, by level | GAIA's official leaderboard and scorer |
| ARC-AGI-1 / -2 / -3 | each version's official task set, interface, attempt budget and scoring; ARC-AGI-3 is interactive and never scored like the static versions (P10.2) | <https://arcprize.org/arc-agi/2>, <https://arcprize.org/arc-agi/3> |

* **pass@k** (Chen et al. 2021) samples n ≥ k candidates for EVERY task and
  estimates, per task, the chance that at least one of k succeeds.
  **pass^k** (τ-bench) estimates the chance that all k succeed. Neither is a
  bounded retry, where a second candidate exists only after the first failed
  (see `admission_yield_up_to_two_candidates` below).
* A result on a modified interface (a textual grid encoding, a custom
  environment adapter, a subset) is a diagnostic. It is never the benchmark's
  official score (P10.2).
* Whether tool calling shows reasoning, planning or generalization is decided by
  the task design and independent evaluation evidence. It is neither ruled out
  in advance nor claimed without that evidence.

## 2. Project metrics

### 2.1 Tool-call verifier levels (`aadistill.data.tool_calls.verify_episode`)

Each level reports exactly what was checked. `None` means unverified: nothing
to check, or no verifier. It is never counted as passed. An episode is admitted
when no level is `False`.

| level | checks | `None` when |
| --- | --- | --- |
| `protocol` | natural termination; the mode's think rule; every tool-call block parses; V1's tail rule (only whitespace after a call turn's last `</tool_call>`) | never |
| `schema_validity` | every call names an offered tool, and its arguments satisfy that tool's JSON Schema; an unchecked keyword fails closed | never |
| `recorded_execution` | **recorded-response replay**: every call the episode executed matched a recorded response in the replay table | the episode executed no call. It never means an executable environment ran the call or a final state was checked: no executable backend exists yet |
| `step_gold` | each gold-carrying turn's call set equals one allowed gold set (name + arguments canonicalized, order-free). The gold is a source's recorded expert call (G2), so this is agreement, not human-verified correctness | the task carries no per-step gold |
| `target_tools_called` | every target tool called at least once: recall of the target set. Extra or wrong calls are NOT checked, so this is not tool-selection accuracy | no such final check |
| `tool_selection` | a complete check of the tool decision: today only `no_call` (no call on any turn of a no-tool task, BFCL's irrelevance criterion) | no such final check |
| `task_success` | a task's registered success criterion, in full | always, today: no source registers one |

### 2.2 Teacher data admission: the tool pilot and `tool_admission`

Stage 2 data construction (`tool_pilot.report`, `tool_admission.admit`). These
numbers measure **whether teacher-native episodes pass the frozen verifier**.
They are not a model-capability evaluation and not a BFCL or τ-bench score.

| metric | numerator / denominator | unit |
| --- | --- | --- |
| first-candidate admission rate (`first_candidate_admission_rate`) | candidate-0 episodes admitted / candidate-0 episodes sealed | episode |
| admission yield of up to two candidates (`admission_yield_up_to_two_candidates`) | tasks with an admitted episode / tasks whose outcome is determined (admitted at candidate 0, or candidate 0 rejected and candidate 1 sealed). Candidate 1 exists only for a task whose candidate 0 was sealed and rejected: a bounded admission retry, not pass@2 | task |
| level pass rate (per level, per candidate) | sealed episodes passing the level / sealed episodes it was checked on (passed + failed); unverified counted apart | episode |
| protocol validity, schema validity, gold tool call match | the level pass rates of `protocol`, `schema_validity`, `step_gold` | episode |
| verification coverage | admitted episodes by the exact set of levels they were verified at, per source and episode shape | episode |
| offered / attempted / sealed / admitted / rejected / incomplete | task counts per mode, split, category, shape and source; incomplete = attempted but unsealed, or unattempted (never a verification failure) | task |
| supervised tokens | generated token ids of admitted training/validation turns, counted once, before packing (P19) | token |
| scored positions | assistant positions of admitted calibration/state-evaluation items | position |
| formal-plan projection | a cell's whole supply × measured yield × measured units per admitted episode, against its target: a projection from a sample, never a measurement, a decision or an approved SHORTFALL | unit |

### 2.3 Batteries and eval-set-1 (Stage 0/2/3 generation, formal measurement)

The scorers are frozen. Their bytes bind `configs/stages/stage-2/teacher_modes/evaluation_identity.json`, a Research
A v1 identity file, so the metrics below are read from the fields the scorers already write. Where a scorer pools
two metrics, read the parts from the rows.

| metric | definition | read from |
| --- | --- | --- |
| answer accuracy | static items whose canonicalized last `\boxed{}` of the final answer equals the canonical gold / static items, whether or not the rollout was usable | row `answer_correct`; summary `answer_correct_rate` (static rows) |
| usable rollout rate | items with `non_empty ∧ natural_termination ∧ no_severe_repetition ∧ no_context_limit ∧ protocol_valid` / items. Its components are reported beside it, and it is blind to correctness (`aadistill.evaluation.usable_rollout`) | row `usable` and its components; summary `usable_rate` (static rows) |
| registered composite correctness | static items with `usable ∧ answer_correct` / static items: the D-series rule, frozen in `score_battery@v1/v2` | row `correct` of static rows |
| environment success rate | interactive episodes whose environment reports success when the recorded replies are REPLAYED through a fresh environment / interactive episodes | row `correct` of interactive rows (`env.success`) |
| battery summary `rate` | the composite over static rows and environment success over interactive rows, POOLED over the cell. Every historical summary reports it under this name, and it is neither of the two metrics alone. Report the two apart from the rows | `score_battery.summarize` |
| eval-set-1 unit `rate` | each component's registered formal protocol. Fluid `dev_v1` pools static composite and interactive success over five axes; read the per-axis capability rows, each of which is one of the two (`interactive_adaptation` is the interactive one). Capability units use each set's scorer `correct` (strict numeric for GSM8K). Paired sets headline pair accuracy. Behaviour groups have no headline. `stage3_legacy_150` is in-sample regression, never generalization. Units and capabilities are never pooled | `eval_set_scoring.report` |

`correct_given_usable` (`usable_rollout`) is correctness among usable rollouts
only. It is reported beside both, never in place of either. Changing a pooled
headline is a new registered scoring protocol and a new identity (AGENTS.md
P10.3), never an edit of the frozen scorer.

### 2.4 Teacher-forced measurement (Stage 1 search, Stage 3 diagnostics)

| metric | what it is | what it is not |
| --- | --- | --- |
| state evaluation: equal-domain mean KL, worst-domain KL, critical-token KL | KL from the root teacher at teacher-forced scored positions of the frozen state-evaluation suite; it ranks and prunes candidates in the structural search | a task accuracy or a generative evaluation |
| scored positions | positions a calibration profile or the state evaluation scores, counted before packing (P19) | a data size in packed sequences |
| teacher agreement / teacher-forced top-1 and rank | agreement with the teacher's next token under teacher forcing | autonomous rollout correctness |
| calibration admission | whether a teacher-native turn enters a calibration profile under its protocol | a correctness judgement |
| CE, KD loss, teacher-native CE, FineWeb NLL, step-0 initialization NLL | optimization and initialization diagnostics (AGENTS.md 4.2, 4.5) | capability |

Each is reported under its true definition, for the purpose it was built for.

## 3. Changes (2026-11-04)

* `verify_episode` reported `recorded_execution: True` for an episode that made
  no call. It is now `None`. Admission is unchanged, since a vacuous `True` and
  `None` are both "not False", so `tool_verification@v2` stands. The pilot's
  frozen task file still carries the earlier `source.verification.levels`
  metadata. The verifier and the report never read it.
* `calls_tools` reported `tool_selection`. It now reports
  `target_tools_called`, because it checks recall, not selection accuracy.
  Admission is unchanged. The frozen six_category_design.json still calls it
  "tool selection only: every target tool called"; its description of the check
  is accurate, and the name is reported as above.
* `contains_all` mapped to `task_success`. A substring of the final answer is a
  task's success only where the task registers it as its whole criterion. No
  source does, so the kind is not implemented, and a task naming it is refused.
* `tool_pilot.report` (schema v3) names its rates as admission rates, gives
  every count, every level's pass / fail / unverified, and the verification
  coverage per source and shape. It also carries `METRICS`, the definitions
  above.
* `score_battery` and `eval_set_scoring` are NOT changed. Their bytes bind
  Research A v1's evaluation identity, so the parts of each pooled `rate` are
  defined in section 2.3 and read from the rows.
