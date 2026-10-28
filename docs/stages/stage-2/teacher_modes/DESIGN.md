# Research A — hybrid teacher: thinking against non-thinking distillation

## Current study (2026-10-28): D4-Augmented's training scale, no pilot

**Status: designed at $0; nothing authorized.** Maintainer decision 2026-10-28.
The design below (`hybrid_v1`, T0 scale with a 10% promotion pilot) is
superseded for new work; A-E's PASS and its records stay as they are.

- **Scale.** H-T-S reaches D4-Augmented's 2,468 packed 8,192-token sequences
  (3,702 optimizer steps at batch size 2, 3 epochs), two seeds. Supervised
  tokens follow from the teacher's own documents. Tasks: T0's 1,761 plus an
  extension E from T0's unused pools and the project's generators, built and
  isolation-checked before generation, generated in waves until the target is
  reached. No pilot gate.
- **Arms.** S: H-T-S and H-N-S on the same task ids. C: H-N-C with further
  independent H-N tasks (X) packed to H-T-S's sequence count.
- **Long reasoning.** A turn that reaches its 8,184-minus-prompt allowance is
  right-censored and never admitted or cut; censoring is recorded per cell and
  mode, and a primary cell above 20% is flagged on S.
- **Kept from hybrid_v1:** the official hard switch, token-native documents,
  document isolation, the common hybrid-derived initialization, eval-set-1 under
  nucleus_v1 in each student's own mode, the scoring identity.
- **Objective.** The project's new KD baseline (Top-200 + tail at the target
  positions). A different teacher lineage from Research B: never a controlled
  pair with it.
- **Axes.** AGENTS.md P10.2 has four primary axes; eval-set-1's static axes are
  three of them plus abstract/analogical reasoning, a sub-category reported on
  its own.

| record | owns |
| --- | --- |
| [`d4_scale_design.json`](../../../../configs/stages/stage-2/teacher_modes/d4_scale_design.json) | scale, tasks, stopping rule, long-reasoning rule, arms, objective, execution |
| [`plans/d4_scale_preregistration.json`](../../../../logs/stages/stage-2/teacher_modes/plans/d4_scale_preregistration.json) | S and C, the decision rule, the data-size rules |
| [`logs/budget/applications/research_ab_2026-10-28/`](../../../../logs/budget/applications/research_ab_2026-10-28/) | the phase application and its forecast |

---

## Historical: hybrid_v1 (2026-10-27)

**Status (2026-10-27): $0 READY for an engineering-validation grant; nothing
authorized.**

- The mode protocol, the token-in/token-out data path, both context strategies
  (cross-context and document-isolated) and mode-aware scoring are implemented
  and validated on the host.
- The scientific design and its decision rule are drafted. They freeze only on
  the maintainer's approval, before any hybrid-teacher datum exists.

| record | owns |
| --- | --- |
| [`configs/stages/stage-2/teacher_modes/teacher_identity.json`](../../../../configs/stages/stage-2/teacher_modes/teacher_identity.json) | the teacher's revision, file and weight hashes, chat template, vocabulary, both modes' generation prompts and official sampling |
| [`configs/stages/stage-2/teacher_modes/design.json`](../../../../configs/stages/stage-2/teacher_modes/design.json) | modes, token protocol, packing, supervision, tasks, generation, admission, initialization, arms, recovery, evaluation |
| [`logs/stages/stage-2/teacher_modes/plans/preregistration.json`](../../../../logs/stages/stage-2/teacher_modes/plans/preregistration.json) | comparisons S and C, the decision rule, the data-budget rules |
| [`configs/stages/stage-2/teacher_modes/evaluation_identity.json`](../../../../configs/stages/stage-2/teacher_modes/evaluation_identity.json) | each mode's evaluation identity: template, think state, instrument, scorer sources |
| [`logs/stages/stage-2/teacher_modes/audit/token_identity_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/token_identity_audit.json) | the re-rendered path against the generated tokens on H1-PA's data, classified; both packings on real documents |
| [`logs/stages/stage-2/teacher_modes/plans/cost_forecast.json`](../../../../logs/stages/stage-2/teacher_modes/plans/cost_forecast.json), [`application.json`](../../../../logs/stages/stage-2/teacher_modes/plans/application.json) | phases, caps, the project-cap consequence |

## 1. Question

One hybrid teacher, `Qwen/Qwen3-4B@1cfa9a72`, sampled in its two native modes:
- **H-T**: `enable_thinking=True`;
- **H-N**: `enable_thinking=False`.

The student architecture, its initialization and the recovery recipe are the
same for both. Which mode's native distribution gives a small student better
fluid-reasoning correctness on unseen problems?

The question is asked twice, and the two answers are never merged:
- **S**: on the same tasks;
- **C**: at the same training compute.

**This is not an answer-only cut of Thinking-2507.** Thinking-2507 remains the
historical teacher and is not an arm.

## 2. The teacher and its modes

**Identity** (`teacher_identity.json`):
- revision `1cfa9a72…`;
- three weight shards with LFS sha256 `328a91d3…`, `6cd087b3…`, `e4bf4369…`;
- chat template `a55ee1b1…`;
- vocabulary hash `7781771a…`, identical to Thinking-2507's;
- tokenizer files `0d96ecc9…`.

**Against Thinking-2507:**
- the template differs (`3802169b…`);
- the RoPE base is 1,000,000 against 5,000,000;
- the configured position limit is 40,960 against 262,144.

| | generation prompt | think state | targets |
| --- | --- | --- | --- |
| H-T | `<\|im_start\|>assistant\n` | `self_opening` | `<think>`, the reasoning, `</think>`, the answer, `<\|im_end\|>` |
| H-N | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | `preclosed` | the answer, `<\|im_end\|>` |

**Rules shared by both modes:**
- only the hard switch is used: no `/think` or `/no_think`, no hand-built
  prefix;
- one system message, `You are a helpful Assistant.`, which names no mode;
- sampling follows the model card's preset per mode: H-T 0.6 / 0.95 / 20,
  H-N 0.7 / 0.8 / 20, `min_p` 0.

## 3. Token protocol: prompt plus generation, never re-rendered

**A document is the engine's `prompt_token_ids` + `output_token_ids`**
(`aadistill.data.token_documents`):
- its prompt must end in its mode's generation prompt;
- it must end on a stop id;
- longer than 8,192 tokens, it is refused, never cut;
- each turn of an episode is its own document, with the prompt recorded at that
  turn. Both modes' templates drop earlier turns' reasoning, so that prompt is
  exactly the context the teacher had;
- episodes are admitted whole.

**Why not re-render.** Rendering a finished turn again is not token-identical
to what the teacher emitted. On H1-PA's own fluid data (767 sessions,
`token_identity_audit.json :: target_differences`) the CE targets as trained
differ from the generated tokens in exactly two ways:

| difference | sessions | supervised positions | share | what it is |
| --- | ---: | ---: | ---: | --- |
| the template's prefilled `<think>\n` | 767 | 1,534 | 0.052% | a template control token the deployment template supplies as context: unnecessary to supervise, not wrong |
| re-tokenization inside the generation | 132 | 1,102 (578 spans) | 0.037% | the **same text** in other token ids; no whitespace and no content difference |

- 11.2% of supervised positions follow a re-tokenization: conditioned on the
  same text in canonical ids rather than the sampled ids.
- H1-PA ran the teacher on the same sequence, so KD compared the two models on
  one context everywhere, with KD on every non-padding position (CE 0.25 +
  KD 1.0). The prefill changes only what CE supervised.
- Whether the prefill positions conflict with the teacher depends on the
  teacher's own probability of `<think>` there. That needs the teacher's
  weights and was not measured.
- **These counts do not show that H1-PA's training was invalid, or that they
  caused its low correctness, and nothing here infers it.** They are a
  serialization question, separate from packing; this study removes it by
  taking tokens as generated.

## 4. Context strategy: two supported, one declared

**Both strategies are supported training distributions**
(`batch.context_strategy`, `token_documents.pack_documents`):

| | `cross_context` | `document_isolated` |
| --- | --- | --- |
| context | several documents in one causal context per sequence | each document conditioned only on itself |
| positions | continuous | restarting at every document and at the pad run |
| packer | `pack_cross_context`: the shared system block once, bodies joined by the template's message terminator | `pack_isolated` |
| KD | teacher and student on the same composed sequence | teacher and student on the same isolated sequence |
| status | the trainer's default; H1 and H1-PA trained this way | opt-in |

- **Cross-context is a deliberate design, not an error.** The project wants a
  model that uses, ignores or integrates context already present. Because the
  teacher runs on the same composed sequence, the KL has no conditioning
  mismatch: it is a different training distribution from the teacher's
  generation context, chosen on purpose.
- A composition of independent documents is still not a real conversation. Its
  composition is recorded (`PackedDocuments.composition()`,
  `context_exposure`) and it is never reported as conversational ability.
- Placement is the same seeded best fit for both, so the two differ in the
  conditioning context and not in which documents meet.
- transformers builds the isolated (block-diagonal) mask from position ids only
  with `use_cache=False` and no 2-D mask, which is the trainer's configuration
  and is tested.

**Declared for this study: `document_isolated`, the same for every arm.** The
reason is to keep the context distribution from moving with the mode:

| packed cross-context at 8,192 | documents per sequence | supervised tokens behind foreign context | mean foreign tokens |
| --- | ---: | ---: | ---: |
| H1-PA's as-generated (thinking) documents | 1.87 | 29% | 1,293 |
| length-only proxy, generation × 0.25 | 6.61 | 84% | 3,318 |

- Under cross-context, S and C would compare the modes **and** two different
  context distributions. Isolated, every target is conditioned on exactly its
  own generation context in both modes, which is the teacher-native
  distribution the mode question is about.
- The proxy uses only lengths, with placeholder ids: no trace content is cut
  or used.
- **Not compared in this round:** the strategy itself. Cross-context against
  isolated training is its own experiment, within one mode, later.

**Verified on CPU:**
- isolated packed hidden states equal each document's own on two real
  8,192-token sequences of H1-PA documents (difference 0.0);
- the trainer's CE and KD over an isolated sequence equal the per-document
  sums; under cross-context the first document of a sequence equals its own
  run, later ones are conditioned on their neighbours, and KD is zero for one
  model on one context;
- the hybrid template's message terminator joins two documents exactly as it
  writes consecutive turns.

The real-teacher check in bf16 is the engineering validation's. Isolation also
packs tighter: 410 sequences at 6.1% padding against the earlier pack's 524 at
26.6%.

## 5. Supervision

**Losses:**
- forward KL(teacher ‖ student) at T 1 with `kd_scope="assistant"`;
- CE on exactly the same positions. The weights are H1-PA's: CE 0.25 + KD 1.0.

**The supervised positions are the generated tokens:** the documents' target
mask. System, user, header, prefill and separator positions are never
supervised, and neither are padding or, isolated, a position that predicts
across a document boundary.

**Teacher context:** the teacher runs on the same packed sequence as the
student, so its distribution at every target is conditioned on what the student
sees.

## 6. Data and the two comparisons

**T0** is the frozen H1 task lists:
- P1 + P1t + PA1, 1,761 tasks, isolation R0–R9;
- AGENTS.md P10.2's four primary axes (task-contained deduction, inductive rule discovery, compositional generalization, interactive adaptation and planning), abstract/analogical reasoning as a reported sub-category, plus verifiable supporting cells;
- both modes on the same task ids.

**Sets derived from T0:**
- **I**: the tasks whose every turn is admitted in both modes;
- **X**: H-N extension tasks from T0's unused pools in T0's cell proportions,
  built and isolation-checked at $0.

**Arms**, each on two seeds:

| arm | documents | steps |
| --- | --- | --- |
| H-T-S | H-T on I | ceil(3 × sequences / 2) |
| H-N-S | H-N on I | the same rule (fewer: H-N documents are shorter) |
| H-N-C | H-N on I + X | H-T-S's sequence count |

**Comparisons:**
- **S** = H-T-S − H-N-S: same tasks, different compute.
- **C** = H-T-S − H-N-C: same compute, more tasks for H-N.
- **C's fallback**, if X is infeasible: a seeded prefix of H-T on I, matched to
  H-N-S's sequence count.

**Generation:**
- A1's first 10% slice is generated first and reprices the rest;
- the slice is part of the data, not extra.

## 7. The common initialization

**One initialization for every arm**, compressed from the hybrid teacher itself:
- the forward-baseline operators on D4 Top-1's path
  (ATTENTION → DEPTH → RESIDUAL_WIDTH → FFN);
- a fixed path: no search, one materialization;
- calibrated on the hybrid's own H-T and H-N trajectories over
  `calib.s0v_base_d4@v1`'s prompts, with equal scored positions from each mode.

**Why not D4 Top-1:** it was compressed from a thinking-only teacher with a
different RoPE base, which would give H-T a starting advantage.

**Declared prior:** the path itself came from a Thinking-2507 search. That is a
structural prior, not a mode prior.

## 8. Evaluation and decision

**Evaluation:**
- eval-set-1, every item, with the mask and formal protocol v2 unchanged;
- nucleus_v1 for every student and for the teacher;
- each student is measured in its own mode, and the teacher in both;
- generations record their `think_state`; `preclosed` ones are scored with no
  think delimiter allowed (approved by the maintainer 2026-10-27 as the $0
  scoring design; no paid evaluation is authorized by it);
- each mode has an evaluation identity
  ([`evaluation_identity.json`](../../../../configs/stages/stage-2/teacher_modes/evaluation_identity.json)):
  template, think state, eval-set-1 identity, formal protocols, conformance
  vectors, nucleus_v1 and scorer sources, by sha256;
- every registered conformance vector rewritten into the thinking and
  non-thinking forms scores exactly as it did historically, and a record that
  does not state `think_state` keeps its historical rule.

**Decision:**
- the H1/mixed rule (`h1_statistics`): SESOI 5 pp, 8 pp per axis, 2% floor,
  rule-family cluster bootstrap;
- S and C each at 97.5% (Bonferroni over two);
- labels: GO_THINKING / GO_NON_THINKING / NO_GO / INCONCLUSIVE / NOT_RESOLVED.

**Reported beside the decision:**
- the teacher's correctness per mode and the teacher-student gap;
- reasoning and answer lengths, supervised tokens and training FLOPs.

**Floor risk:** H1-PA's students sit at 3–5% static fluid. A floor result is a
real possibility and is labelled as such. The
[reasoning difficulty ladder](../../../shared/REASONING_DIFFICULTY_LADDER.md) is
the planned instrument below the floor.

## 9. Cost and authorization

From [`cost_forecast.json`](../../../../logs/stages/stage-2/teacher_modes/plans/cost_forecast.json):
- **A-E** (engineering validation): $1.81 expected, $3.50 cap. It fits the
  current $740 project cap.
- **A1–A3** (generation, initialization, recovery and evaluation): about $31
  expected, $50 in caps.
- With storage and the reserve, **total $60.00**.

The formal phases need a project cap near $790. That decision is the
maintainer's.

## 10. Owed before the first paid action ($0, inside a grant)

The application lists the full set. In short:
- governance that counts engineering-campaign subruns;
- the per-mode generation and admission chain;
- the mode-balanced calibration and fixed-path materialization;
- the recovery and evaluation drivers;
- the analysis.

## 11. A-E: the engineering validation

A-E checks, on one A100-SXM4-80GB, what a host cannot: the real hybrid teacher
in both official modes, the engine, real memory and real step time. It is not
a scientific run. No document it builds is training data, no correctness is
reported, and nothing it measures changes this design. A-E uses
`document_isolated` only and adds no cross-context comparison.

- **Driver:** [`engineering_validation.py`](../../../../scripts/stages/stage-2/teacher_modes/engineering_validation.py).
- **Check config:** [`engineering_validation.json`](../../../../configs/stages/stage-2/teacher_modes/engineering_validation.json).
  It fixes every task id, threshold and numeric setting before the run, and
  the issuer binds it into the study identity.

| check | what runs | passes when |
|---|---|---|
| prompt identity | four T0 tasks (one per axis) and four eval-set-1 items, rendered with the official hard switch only | the chat template is teacher_identity.json's, every prompt ends in its mode's generation prompt and leaves its mode's think state, the teacher's files and weight shards match the identity |
| generation | the teacher under vLLM 0.27.1 in both modes: token ids in, each mode's official sampling preset, the same seed per task in both modes. T0 allowance: A1's data-construction limit (8,184 minus the prompt). Eval items: the full remaining context (P18) | the engine echoed every prompt's ids, every turn ended on a stop id or is recorded at its limit, no non-thinking output holds a think delimiter, each mode yields documents |
| teacher forward | the documents in an isolated pack at 8,192, against each document alone, at every target position, in fp32 (TF32 off) and bf16, both modes; and the engine's teacher-forced log-probabilities of the same ids | bf16 packed-vs-alone ≤ 2× the documents' own bf16-vs-fp32 difference; fp32 packed-vs-alone ≤ 0.1× it; the engine's mean log-probability difference from the HF fp32 forward ≤ 4× the HF bf16 one |
| assistant-target KL | the trainer's `kd_scope` `assistant` mask on the pack | the KD positions are exactly the generated tokens (no system, user, header, prefill, padding or cross-document position); positions restart at every document; KL(teacher ‖ teacher) = 0; the stand-in student's KD is finite and counts the target tokens |
| trainer smoke | 20 steps of H1-PA's recipe with this design's declared changes (`kd_scope` `assistant`, `batch.context_strategy` `document_isolated`, the hybrid teacher on the fly) on H1-PA's PCA-Original seed-20260726 checkpoint as a stand-in student | finite losses; every step isolated, with KD positions equal to CE targets; the config differs from E1's only in the declared keys; the student's bytes are the declared ones. Step time and peak memory are recorded |
| non-thinking scoring | the eval-item turns through eval-set-1's frozen scorers under their recorded think state | every record is read in its mode's state; no well-formed record is a protocol violation; a delimiter in a non-thinking reply is one; without `think_state` the same reply keeps the historical rule; evaluation_identity.json regenerates unchanged |

**Process isolation.** The parent process never initializes CUDA. The PyTorch
work runs in child processes that exit before the next GPU user starts. vLLM
runs through `shared.validation.vllm_check.run_isolated`, which reads the GPU
first and refuses while any compute process holds it. The engine runs before
the PyTorch phases, so the GPU is read and recorded again before each phase.

**Verdict.** `verdict(out, config)` re-derives every status from the evidence
files and the config alone. The pod writes it as `report.json`; the dev box
re-runs it on the secured copy.

The driver also runs end to end on a host (`--tiny`): random small models at
the real vocabulary, and the engine NOT_RUN, or a stand-in engine in the tests.

## Reproduce ($0)

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/teacher_identity.py \
    --write configs/stages/stage-2/teacher_modes/teacher_identity.json
PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/token_identity_audit.py \
    --write logs/stages/stage-2/teacher_modes/audit/token_identity_audit.json
PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/evaluation_identity.py \
    --write configs/stages/stage-2/teacher_modes/evaluation_identity.json
PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/forecast_cost.py \
    --write logs/stages/stage-2/teacher_modes/plans/cost_forecast.json
PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/engineering_validation.py \
    --run-id host --out /tmp/ae/out --work /tmp/ae/work --status /tmp/ae/status --tiny
PYTHONPATH=src:scripts:tests .venv/bin/python -m pytest -q tests/data/test_token_documents.py \
    scripts/stages/stage-2/teacher_modes/tests
```
