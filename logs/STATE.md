**Updated:** 2026-08-10 · branch `main` · commit `864dd71` + E8 preparation ·
working tree dirty
**No pods running. Nothing billing.** **1,138 CPU tests pass**, 3 skipped.

**Experiment 7 is complete. Experiment 8 is prepared, preregistered and blocked
on authorization** ([`e8_preregistration.md`](e8_preregistration.md)) — it needs
**$10.08** above the current cap. Stage 3 remains open: no model has passed a
prospectively defined behaviour-recovery gate, and **nothing yet moves autonomous
reasoning correctness**.

**One dev-box job is running, free:** Stage 0 activation-statistics regeneration
(`artifacts/stage0/regen_tf513.log`, ~7 h CPU). It is a **prerequisite for E8** —
see §0.5.

This file is a **snapshot of the current state**, not a history. Every completed
experiment lives in [`EXPERIMENTS.md`](EXPERIMENTS.md) and every decision in
[`decisions.md`](decisions.md); this file links to them rather than repeating
them.

---

## 0. Binding rules — read before acting

### 0.1 Budget

```
previous authorized cumulative cap: $149.03   EXCEEDED and CLOSED
temporary canary cap:               $150.41   spent, closed
authorized cumulative cap:          $162.49
ACTUAL CUMULATIVE SPEND:            $160.158
remaining under the cap:              $2.33

authorized cumulative cap (E8):     $172.57   granted 2026-08-10
E8 authorized backstop:              $12.41
E8 spent on failed pod A attempts:    $0.83   4 infrastructure defects, all fixed
E8 remaining:                        $11.58
pod B hard threshold:                 $9.7130
LEFT FOR POD A:                       $1.8670  -> its plan needs $2.7002
E8 SHORTFALL, BLOCKING:               $0.83
```

**E8 is stopped at a $0.83 shortfall.** Pod A has not yet run its search. Four
launcher/setup defects were found and fixed at $0.83 total, each self-terminating
within seconds; none touched the experiment's design. `plan_session` refuses every
pod A authorization below $2.71, so the run cannot proceed without either a $0.83
increment (proposed backstop $13.24, cap $173.40) or an explicit instruction to
re-price pod A's 45-minute setup contingency.

**Plan from actual spend, never from unused room under a previous
authorization.** The historical $149.03 is not a balance; it was exceeded by
$0.56 in E6b and that overrun is recorded, not rewritten. Any new paid execution
needs an explicit increment above **$160.158**.

**Never silently shrink an experiment to fit a shortfall** — report the shortfall
specifically and ask. `budget.plan_session` enforces this: it raises with the
exact figure rather than trimming the run.

### 0.2 Promotion

**Teacher-forced CE, held-out/FineWeb NLL, teacher KL, top-1 and rank, and
training loss are training-health diagnostics only. Checkpoint promotion, arm
selection and stage advancement are decided on the frozen autonomous rollout
evaluation.**

Two experiments now establish this independently. E6b: two objectives improved
validation CE identically (1.30 → 1.15 vs 1.31 → 1.17) and only one moved
behaviour. **E7: a −5.22 nat swing in held-out FineWeb NLL moved autonomous
behaviour by exactly +0.0000.** Do not infer autonomous improvement from a
diagnostic.

### 0.3 Artifact lifecycles

**A bounded prefix of a growing file is a snapshot, never a final artifact.**

* `mutable_snapshot` — the writer may still be active. The archive records the
  captured byte boundary and hashes those bytes. Proves **durability**; claims
  nothing about completeness.
* `final_required` — the producer has finished, its terminal marker exists, and
  the file is quiescent across a settle window. **The default.**

A normal teardown requires every required structured event stream to be
`final_required`. Emergency budget termination may keep a `mutable_snapshot`, but
must **name** the streams it truncates; an unnamed truncation raises.

### 0.4 Session contract for every paid pod

[`scripts/pod/AGENTS.md`](../scripts/pod/AGENTS.md) is binding. Detached start via
`start_job.py`; `watchdog.py` running beside the launcher from pod creation;
`LogRelay` mirroring event streams continuously; `collect_artifacts.py` gating
teardown. **`--terminate-after` is a redundant third layer and is not a stop
mechanism** — it has never been observed to fire. Verified live end-to-end by the
control-plane canary ([`e7_canary_rerun_report.md`](e7_canary_rerun_report.md),
12/12) and exercised for 635 minutes by E7 without incident.

### 0.5 Two prerequisite facts found on 2026-08-10, both binding

**The Stage 0 activation cache was lost — and has been recovered bit-exactly.**
`artifacts/stage0/qwen3_4b_thinking_v1/activation_stats.safetensors` (1.95 GB,
sha256 `aaeb2e4c…`) was not on the dev box and was **never on the relay** — its 780
files contain no `stage0/` path. Stage 1 cannot construct *any* initialization
without it. Regenerated at $0 in 4,972 s of CPU (949,859 tokens, the historical
count) and it hashes to **`aaeb2e4c…`**; rebuilding the **positional** init from it
gives **`86fbba78…`**, byte-identical to the pinned control init, with every
projection diagnostic equal to the last digit. **E8 is therefore a
single-variable experiment.** [decisions](decisions.md) 2026-08-10.

The recovery worked only because the pipeline is deterministic and the hash was
logged. **A 1.95 GB artifact on the critical path of every future initialization
still has no off-box copy** — the relay has no room for it, so any future loss
costs another ~83 min of CPU rather than being unrecoverable.

**The canonical environment for Stage 1 artifacts is transformers 5.13.1 / torch
2.13.0** — the repo `.venv`, not the other venv on this box. The Stage 1
checkpoint's config stores `rope_theta` in the transformers-5 `rope_parameters`
dict; a **4.x** reader silently falls back to 10,000 (500× wrong) and reports
holdout NLL **11.3953** instead of **11.7482**, with nothing raising. The teacher
is immune, which is what makes it easy to miss. No trained arm is affected — the
pods already assert `ROPE_OK` — but the *measurement* path did not, and now does
(`measure_init_nll.py`, `init_stage1.py`). Under 5.13.1 the historical value
reproduces exactly: 11.748248 on 21,080 positions.

---

## 1. Where the project is

Teacher **`Qwen/Qwen3-4B-Thinking-2507` @ `768f209d`** → student **0.6B-class**
(1024 hidden, 28L, FFN 3072, 16Q/8KV, tied emb). BF16 training, INT8 deployment.

Stages 0 → 1 → 2 complete. **Stage 3 recovery is open.**

**Behavioural anchor: `e1_r2960k_{sa,sb}_pca` = E1/P1 KD-heavy 2.96M** — usable
rollout 0.8400, correct 0.2067. Nothing has displaced it.

**Stage 0/1 works and its downstream value is proven.** Step-0 held-out NLL
11.7482 against a random init's 12.1286 (teacher 2.6264). Downstream that
−0.38-nat edge decides everything: across 12 matched pairs the PCA init wins
**12/0/0** on autonomous behaviour and 11/1/0 on GSM8K, and random init produces
**zero usable rollouts at every rung through 2.96M**.

**Stage 3 has not been exited.** No prospectively defined behaviour-recovery gate
has been passed — and no such gate existed when these runs launched, so this is
the absence of a criterion, not a measured failure against one.

**The dominant failure is that the model does not stop.** ~31% of rollouts run to
the 8,192-token context limit; about one prompt in eight is answered with nothing
at all.

## 2. What has been tried, and what it moved

| intervention | behaviour | correctness |
| --- | --- | --- |
| more recovery data, 0.86M → 1.60M | **+0.2000**, both seeds | flat |
| more recovery data, 1.60M → 2.96M (KD-heavy) | **+0.1100**, both seeds | flat |
| more recovery data, 2.96M → 5.50M | saturated (+0.0100) | flat |
| more recovery data, 1.60M → 2.96M (CE-heavy) | a tie (+0.0267) | flat |
| KD scope (assistant-only) | flat | flat |
| CE/KD reweighting | flat | flat |
| restricting attention updates (frozen, LoRA) | **worse** | worse |
| student-prefix continuation | **much worse** | worse |
| teacher-prefix continuation | tie | possibly worse |
| **general-text KD (E7, FineWeb)** | **+0.0000** | flat |
| **matched in-domain extra KD (E7, control)** | tie | tie |

**Eleven interventions. Scale is the only thing that has ever moved behaviour,
and nothing has ever moved reasoning correctness.** It sits in 0.11–0.21 across
every arm ever evaluated, and GSM8K correctness is 0.00–0.08 everywhere.

**Closed by E7:** lost general language modelling is **not** the cause. It can be
restored almost completely (−5.22 nats, top-1 up 9×) with no behavioural effect
whatsoever. See [`EXPERIMENTS.md`](EXPERIMENTS.md) §34.

**Closed earlier:** reweighting the two existing loss terms (both directions
tried, §17/§18); selecting on held-out NLL (retired, §12.15); the capacity
question — the released `Qwen3-0.6B`, the student's exact geometry and parameter
count, answers ~70% of GSM8K and ~74% of RAG on this project's own frozen battery
under this project's own protocol (§14.2, rescored §15.1). **The task is
reachable at 0.6B.** That bounds the task; it does not localise our gap, which
belongs to the whole training stack and trajectory until evidence separates it.

**Binding scope rule.** Teacher-forced reasoning top-1 is a **within-family
controlled-comparison metric** — valid across arms sharing teacher, architecture,
initialization and evaluation set. It **must not** be promoted into a cross-model
capacity scale; scoring a model not trained on this teacher's traces against them
measures style compatibility, not capability.

## 3. Latest result — E7 (2026-08-09, $10.49)

**FineWeb teacher KD restores general language modelling and does not solve
reasoning.** Preregistered outcome 2, fixed before the run.

| arm | FineWeb NLL | usable | correct | correct \| usable |
| --- | ---: | ---: | ---: | ---: |
| A retained baseline | 9.4847 / 9.4541 | 0.7300 | 0.1867 | 0.2511 |
| **B** FineWeb KD | **4.2664 / 4.2478** | **0.7300** | 0.1900 | 0.2603 |
| **C** matched control | 4.7713 / 4.7508 | 0.7500 | 0.1500 | 0.2000 |

Every paired comparison inside its registered floor (usable 0.0800, correct
0.0600). The matched in-domain control recovers **90%** of the same NLL gain, so
what restores general text is *extra KD signal on unseen text*, largely
regardless of which text.

Record [`EXPERIMENTS.md`](EXPERIMENTS.md) §34 · report
[`e7_report.md`](e7_report.md) · preregistration
[`e7_preregistration.md`](e7_preregistration.md) · decision
[`decisions.md`](decisions.md) 2026-08-09.

## 4. Protocol requirements (binding)

* **System message is mandatory** in teacher generation, student training,
  primary evaluation and inference. Fixed project requirement, **not** an
  experimental variable. Default: `You are a helpful Assistant.`
* Thinking mode is never suppressed; `<think>` is opened by the template
  unconditionally on `add_generation_prompt`.
* **No artificial generation cap in formal measurement** (P18). Allowance is
  `context − prompt`. Context resolves to the **trained** `block_len` = **8,192**,
  not the architectural 262,144 the geometry inherits — recorded per measurement
  as `context_resolution.context_source = "trained_block_len"`. Experiment 2
  phase 1 reported `context_limit_rate` **0.0000 at all 20 checkpoints** — but
  only because the **degeneration stop was active**, cutting loops before the
  limit (`stop_reason: degeneration`). With the stop disabled, the same weights
  context-limit on 31.1% of rollouts. When comparing context-limit rates, the
  stop policy must match or the comparison is meaningless.
* Stop ids come from the model's `generation_config` (teacher `[151645, 151643]`),
  not the tokenizer alone.
* No no-think / empty-think / final-only / shortened substitute targets (P17).

## 5. Measurement constraints

| quantity | value |
|---|---|
| behaviour-metric seed noise floor | **0.1290** → ≥2 seeds per arm |
| cold-start holdout-NLL seed spread | **2.21 nats** → ≥4 seeds from the Stage 1 init |
| teacher natural termination | **80.1%**; lengths p50 **727**, p99 3854 |
| `block_len` 8192 memory | 44,983/46,068 MiB with gradient checkpointing, ~4.3 s/step |
| corpus generation throughput | 66.08M tokens in 16.5 h on one L40S (~1,110 tok/s sustained) |
| training throughput | ~21 min per 137-step arm including gate evals |

## 6. Pinned assets

| asset | identity |
|---|---|
| **fork point** — Stage 1 structural init | `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`, `model.safetensors` sha256 `86fbba78…` |
| **recovery corpus v2** (2026-08-01) | `sessions.jsonl` sha256 `2b4edc2e…`, `candidates.jsonl` sha256 `f7f5035e…` |
| **token ladder v2** | `blocks.npz` + `audit.jsonl` + `ladder.json`, 3,720 blocks |
| teacher corpus v1 (752 prompts, 540 accepted) | relay `stage3_teacher_corpus_20260730/`, targets sha256 `18028f0c…` |
| best recovery checkpoint (reference only) | `s2v1_from_init@2700`, holdout 3.8285 |
| relay | `AlphaAvatar/aadistill-artifacts` (private) |

> **Resolved 2026-08-01:** corpus v2 and both ladder cuts are on the relay under
> `stage3_recovery_corpus_v2/`, 9/9 files hash-verified against the local copies.
>
> **Open storage constraint.** The relay hit its private-storage limit during the
> run. Deleting the superseded `tt2x2`/`ttb` weights (19.07 GB, approved) dropped
> the tree to 80.31 GB but reclaimed **nothing** — HF bills LFS storage including
> history. The maintainer approved squashing history to reclaim it (see
> [decisions](decisions.md), 2026-08-02); until that runs, **four arms are
> dev-box-only**: `e1_r2960k_sb_pca`, `e1_r5500k_sb_pca`, `e1_r2960k_sb_rand`,
> `e1_r5500k_sb_rand`, each hash-verified under
> `artifacts/stage3/rescued/`.

## 7. Implementation state (CPU-verified)

**1,084 tests pass on CPU, 3 skipped** (`PYTHONPATH=src pytest tests/ -q`, ~60 s,
no downloads; `uv run pytest tests/ -q` also works). The 3 skips are the
deliberate frozen-record launcher exemptions in the `$(ls …)` lint.

| piece | file | state |
|---|---|---|
| four-threshold session budget | `src/aadistill/infrastructure/budget.py` | expected / soft stop / recovery reserve / hard terminate; 4.15 s/step floor; 11 tests |
| provider control-plane client | `src/aadistill/infrastructure/provider.py` | GraphQL poll (verified by use), CLI-then-GraphQL terminate, in-memory simulator |
| independent cost watchdog | `src/aadistill/infrastructure/watchdog.py` | terminate → **verify gone** → retry → journal; `SessionWatcher` cannot read silence as idle; 15 tests |
| detached remote launch | `src/aadistill/infrastructure/remote.py` | bounded start, durable job descriptor, out-of-band confirmation; 9 tests incl. wall-clock bounds |
| continuous log durability | `src/aadistill/infrastructure/log_relay.py` | incremental offset-resumable mirroring; never raises; 9 tests |
| manifest-driven collection + teardown gate | `src/aadistill/infrastructure/artifact_gate.py` | Python glob expansion, archive from manifest, ordered gate, emergency override; 19 tests |
| E6b failure replay | `tests/infrastructure/test_e6b_failure_simulation.py` | the whole 2026-08-08 sequence, 15 tests, no GPU |
| dual-stream KD trainer (E7) | `src/aadistill/training/train.py` (`extra_stream`) | second cursor inside the same optimizer step, independent normalizers, zero CE, exact budget; 19 tests |
| dense KD-only stream format | `src/aadistill/data/extra_stream.py` | no padding, explicit doc boundaries, `n_blocks x (block_len-1)` KD positions by construction |
| general-text diagnostics | `src/aadistill/evaluation/general_text.py` | NLL / teacher KL / top-1 / rank / confidence; 10 known-answer tests |
| FineWeb + matched-control builders | `scripts/data/build_{fineweb,control}_kd.py` | pinned revision, index ranges, per-doc hashes, exact budget match |
| stream disjointness proof | `scripts/data/check_stream_disjointness.py` | index **and** content-hash separation; fails closed; 14 tests |
| E7 arm guards | `scripts/training/{build_e7_configs,validate_e7_arms}.py` | the diff vs the retained baseline is exactly `{extra_stream, run_name, out_dir, _purpose}` |
| four-threshold E7/canary pricing | `scripts/training/plan_e7_budget.py` | phase-wise, from measured E6b wall clock |
| session rendering + system-grouped packing | `src/aadistill/data/sessions.py` | used in the built corpus |
| shared assistant-mask helper | `src/aadistill/data/dataset.py` | `final_assistant_loss_mask` for turn expansion |
| `min_p` + per-prompt completion budgets | `src/aadistill/rollout/engines.py` | threaded through all 5 adapters |
| corpus builder | `scripts/rollout/build_recovery_corpus.py` | ran the 2026-08-01 bulk build |
| one-pass pack + nested ladder cut | `scripts/data/build_token_ladder.py` | produced the 6-rung ladder |
| §6/§9 gate validator | `scripts/data/validate_corpus_gate.py` | PASS on the full corpus |
| end-to-end CPU dress rehearsal | `tests/data/test_recovery_corpus_pipeline.py` | builder→ladder→gate with a stub engine |
| candidate cleaning rules (`clean-v2`) | `src/aadistill/data/cleaning.py` | median-length survivor; built the D1 corpus; 30 rule tests |
| cleaned-corpus driver | `scripts/data/build_cleaned_corpus.py` | 11,174 sessions screened in 114 s |
| ladder session-order anchor | `scripts/data/build_token_ladder.py --session-order` | keeps a re-cut pack on the anchor's prompts |
| prompt-matched single-rung packer | `scripts/data/build_matched_rung.py` | 89.1% D0 overlap at exact compute; appends the control's validation blocks |
| selection-rule comparison | `scripts/data/audit_selection_rule.py` | median vs shortest on one corpus |
| checkpoint retention policy | `scripts/pod/retain_checkpoints.py` | trajectory-driven keep set; 13 tests |
| frozen capability battery | `scripts/data/build_capability_battery.py` | `capability-v2`, 846 prompts, 7 sets, 0 leakage collisions |
| deterministic capability scorers | `src/aadistill/evaluation/capability.py` | alias EM, symbolic math, evidence recall, paired answerability **and** paired safety; 112 tests |
| battery scoring driver | `scripts/evaluation/score_battery.py` | pair-accuracy headline; offline, re-runnable |
| checkpoint inventory + cleanup | `scripts/pod/checkpoint_inventory.py` | both stores, hash-matched duplicates, declared classification |
| D0↔D1 corpus audit | `scripts/data/audit_d1_corpus.py` | overlap, shares, budget, residual mismatch |
| strict final-answer rule | `src/aadistill/evaluation/strict_answer.py` | replaces last-number GSM8K scoring; 17 tests |
| offline GSM8K re-scoring | `scripts/evaluation/rescore_gsm8k.py` | re-scored all 25 E1 arms, $0 |
| teacher block bypass + greedy contribution search (E8) | `src/aadistill/init/contribution.py` | module-list bypass verified against an identity-block path; 260-evaluation greedy; domain-balanced KL; 23 tests |
| explicit depth map into Stage 1 init | `src/aadistill/init/sandwich.py` (`explicit_depth_map`, `kept_layers=`) | feeding it the positional map's own representatives reproduces that init **bitwise** |
| depth-search driver + resume | `scripts/training/search_depth_map.py` | self-consistency gate, positional-map comparison, full per-round tables, auto cache fallback; 9 end-to-end CPU tests |
| frozen E8 calibration mixture | `scripts/data/build_e8_calibration.py` | 67 items, 59,763 positions, 5 domains, `d65c1f40…`; excludes the rung, the val slice and prompt-content collisions |
| calibration leakage proof | `scripts/data/check_e8_calibration_leakage.py` | six fail-closed checks; caught two real collisions |
| mandatory hash-bound init NLL | `src/aadistill/init/nll_gate.py` + `scripts/evaluation/measure_init_nll.py` | an init is incomplete without its own NLL; inherited records rejected; 11 tests |
| masked teacher-native held-out metrics | `src/aadistill/evaluation/init_nll.py` | assistant-target NLL/KL/top-1/rank on the pack's val slice |
| E8 arm builder + pre-training gate | `scripts/training/{build_e8_configs,validate_e8_arms}.py` | realized diff exactly `{student_path, run_name, out_dir, _purpose}`; gate fails closed |
| E8 four-threshold pricing | `scripts/training/plan_e8_budget.py` | two pods, search cost derived from forward-pass arithmetic |

Chunked CE/KD was assessed and is **not** needed: `block_len` stays 8192, which
the canonical recipe already runs.

**Finding that shaped the design:** the official chat template renders
`<think>…</think>` only for the assistant turn after the *last* user message, so
applying it to a multi-session message list **silently deletes every earlier
trace**. Verified directly. Sessions are therefore rendered independently and
concatenated at token level (asserted exact), with the system block emitted once.

## 8. Known deviations to carry

* Corpus **v1** (2026-07-30) was sampled at temperature 1.0 / top_p 1.0 / top_k
  off, against the model card's preset. Corpus **v2** uses the official preset
  `0.6 / 0.95 / 20 / min_p 0`, so the two corpora are not interchangeable.
* Corpus v1 is **effectively n=1** — 92.7% byte-identical candidate pairs
  because a serving engine seeds per request. Fixed in v2 by per-candidate
  seeds (`seed + batch_index + candidate_index × 1000003`); the v2 gate
  confirms distinct seeds and non-identical candidates for every type.
* The 8,192-token **session limit** censors the long tail of the hardest types:
  `openmath` loses 1,541/3,600 candidates and `code` 702/4,800 to
  `length_limited`, so accepted sessions of those types skew shorter/easier.
  Consequence of the fixed session limit; recorded, not worked around.
* Corpus v2's `code_state` block records **no git commit** — the pod bundle was
  unpacked outside a git checkout, so `git rev-parse` failed and the manifest
  stored `code_state_error` instead of a commit. The corpus is pinned by data
  hashes, teacher revision, chat-template hash and the full command, but its
  code state is pinned only by the bundle that was shipped. A P4 gap; fix the
  bundle to carry the commit before the next paid generation.
* Corpus v2 **computes correctness but does not enforce it** (acceptance is
  hygiene only, by design): `rag_evidence` 0.978, `gsm8k` 0.890, `multihop_qa`
  0.861, **`openmath` 0.380**; `code` and `tool_calling` have no mechanical key
  and score `unverifiable_slice`. Roughly a third of `openmath` targets teach a
  wrong final answer.
* `verify.hygiene_reason`'s `too_long` rule (`MAX_ANSWER_WORDS = 600`) is
  deliberately not applied — a generic word-count gate is forbidden by P3/P10.
  Structural hygiene only. Recorded in the manifest.

## 9. Next actions

**Nothing costing money is authorized.** $2.33 remains under the $162.49 cap;
E8's hard backstop is $12.41, so it needs **$10.08 more** and a cap of **$172.57**.

**E8 is authorized, implemented, staged, and stopped at a $0.83 shortfall.**
Ordered next actions:

1. **Maintainer decision on the $0.83.** Pod A's plan needs $2.71 and $1.87
   remains after pod B's reserved $9.7130. Nothing paid may start before it.
2. `scripts/pod/e8a_launch.py --scr <new> --session-commit <HEAD> --bundle <new>
   --authorized-usd 2.71 --host-draws 4`. All four fixes are in; a cold host or a
   pod that never starts now redraws instead of aborting.
3. Dev box, $0: `scripts/training/build_and_stage_e8_init.py --frozen-map <fetched>`
   builds the treatment init, verifies it against the control, uploads it and
   prints pod B's `--treatment-init-sha256`.
4. `scripts/pod/e8b_launch.py … --treatment-init-sha256 <hash> --authorized-usd
   <12.41 − actual pod A spend, asserted ≥ 9.7130>`.
5. `scripts/evaluation/analyze_e8.py --bootstrap 10000`, then the records.

**Durability is done** (`logs/e8_relay_manifest.json`): 13/13 staged and
roundtrip-verified, including the 1.95 GB Stage 0 cache and `warmup_v1`, its input.

**No other follow-up is running or planned.** A 2.96M + FineWeb confirmation, a
FineWeb-ratio sweep, P2-5.50M, on-policy/GKD, a second contribution map and any E9
combination are **out of scope** and must not be launched.

**The open problem is correctness, and it is now better bounded than before.**
Eleven interventions have moved behaviour or nothing; none has moved reasoning.
E7 removed one candidate explanation (lost general language modelling) without
proposing a replacement. What remains unseparated: initialization, token budget,
curriculum, and the train/inference distribution mismatch.

**The strongest untested lead remains the mismatch.** Teacher-forced reasoning
top-1 is ~0.57 and moves ~0.05 across every recipe tried, while free-rollout
correctness sits at 0.15–0.21 and is dominated by exact repetition loops — the
classic exposure-bias signature. `src/aadistill/rollout/` holds ~2,075 lines of
tested infrastructure that **no training path consumes**. Stage 3 sub-stage 3
and Stages 4–5 are exactly the remedy AGENTS.md specifies. The cheap first step
is CPU-side: report CE under teacher forcing against CE on the student's *own*
prefixes for the same prompts, so the gap is a number before anything is trained
against it.

**Still open, unchanged:** the approved relay history squash (destructive,
confirm separately); the four dev-box-only Experiment 1 arms; Experiment 2 phases
2–3 remain unauthorized and phase 3 should not run as written — it was built
around the retired metric.

## 10. Experiment index

| # | experiment | outcome |
| --- | --- | --- |
| §11 | Experiment 1 — recovery-data scaling, 24 arms, $47.6 | initialization dominates; data drives CE |
| §12 | Experiment 2 phase 1 — corpus cleaning | not adopted; retired `best_holdout_nll` |
| §19 | post-hoc re-evaluation under the clarified Stage 2/3 objective | nothing separates |
| §20 | Experiment 3 — restricting attention updates | rejected; degrades |
| §21 | Experiment 4 — P2 CE-heavy 0.86M → 1.60M | scale gain, not objective |
| §22–27 | Experiment 5 — prefix continuation, eight attempts | teacher-prefix ties, student-prefix much worse |
| §28 | Experiment 6 — the E1 scale curve | improves once, then plateaus |
| §29 | Experiment 6b — objective × scale | KD-heavy converts the rung; CE-heavy does not |
| §30 | operational hardening after E6b | four stop-layer defects fixed |
| §31 | Experiment 7 design and preregistration | — |
| §32–33 | control-plane canary, failed then passed 12/12 | live path verified |
| **§34** | **Experiment 7 — general LM restored, behaviour unmoved** | **the ceiling is not a language-modelling problem** |
| §35 | Experiment 8 design and preregistration — contribution-guided depth | prepared, blocked on $10.08 |

Protocol deviations on record: [`e6b_protocol_deviations.md`](e6b_protocol_deviations.md)
(cost overrun $0.56, lost event streams; scientific endpoint valid, operational
protocol noncompliant).
