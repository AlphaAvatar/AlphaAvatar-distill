# PROPOSAL (deferred to Stage 4/5) — recovery on **top-n sampled, verified-correct** teacher traces

Status: **DEFERRED to Stage 4/5 on 2026-07-28 — no spend committed.**

Reclassified by the maintainer the same day it was scoped: large-scale teacher
generation belongs to Stage 4/5, because **(a)** a corpus of one teacher's traces
cannot be applied to another model, so it is a per-teacher artifact rather than a
reusable Stage 2 mixture (the pipeline is model-family-agnostic by decision), and
**(b)** generation and training can run in parallel, which is an online data loop
of the same shape as Stage 4/5 even though the states still come from corpus
prompts rather than the student. **Nothing below is discarded** — the design,
costs, verification rules and option-B target format stand as written for when
this work is picked up. What changed is *when*, and that no money is committed
now. Decision record: `logs/decisions.md`, 2026-07-28.

Read before reviving this: the public mixture already contains worked-solution
targets (all 7,149 gsm8k targets carry step-by-step arithmetic, all 4,344
OpenMathInstruct targets carry full derivations), and the student still scores
**0.000** gsm8k EM. The case for buying teacher traces should be re-argued after
the packing control run shows whether the student can use the reasoning
supervision it already has.

Original status line: **REVISED 2026-07-28 (2nd pass) — awaiting maintainer approval.**

*Filename note:* the path still says `stage2` and is kept that way for link
stability (two historical experiment logs point at it). The staging below is
what governs.

**Three maintainer directives, 2026-07-28:**

1. ~~This is the next supplementary experiment in Stage 3~~ — **superseded the
   same day**: reclassified to Stage 4/5 (see the status block above). The
   run design below is unchanged; only its stage and timing moved.
2. **The targets must be teacher-generated *and* correct**, produced by
   **sampling n candidates per prompt and selecting a verified-correct one**
   (top-n / rejection sampling), not by a single greedy pass.
3. **The teacher is never forced out of thinking mode.** Evaluation and
   generation both judge it at its actual capability. This costs roughly 10×
   more decode per candidate and is the reason the cost table below moved by an
   order of magnitude; it also means the teacher can finally be *measured*,
   which is what the README figure is missing today.

Decision record: `logs/decisions.md`, 2026-07-28.

## What changed across the two revisions

| | 2026-07-27 draft | 1st pass (correctness) | now (top-n + Stage 3 framing) |
|---|---|---|---|
| framing | "Stage 2 v2 corpus" | Stage 3 supplementary experiment | **Stage 4/5, deferred** — a one-teacher corpus is not a reusable Stage 2 mixture |
| accept rule | grounding filter | **mechanical correctness check** vs a gold key; else keep the v1 target | unchanged |
| generation | 1 greedy pass | 1 greedy pass, bounded retry on rejects | **n sampled candidates per prompt, verify all, select one** |
| scope | 3 groups, 18,314 targets | 5 slices, 29,807 candidates (math slices move in) | unchanged |
| coverage | — | items the teacher fails keep public targets → bimodal corpus, selection bias | **top-n is the direct fix**: accept@n > accept@1, so fewer items fall back |
| engine | HF transformers | HF transformers | **decided by the pilot**: HF adaptive, or vLLM (shared prefill across the n samples) — a heavy-dependency decision |
| teacher mode | empty-think prefill (cheap) | same | **native thinking, always** — the framework distills the teacher's real capability, not a suppressed mode |
| cost | $6–9 | $11–17 | **$1.20 spent** (teacher scorecard only, done); the pilot and the $25–145 build are **not committed** |

## The argument (unchanged)

After the mixture-v1 recovery run the student's language modeling is
substantially better (holdout NLL 4.21 → 3.80, 26% of the teacher gap closed)
while its *answers* are still wrong-shaped. All three start-point ablation arms
share the defect — `rep_3gram` 0.35–0.41, `answer_words` 199–231,
`truncated_at_cap` 0.58–0.67, `format_ok` ≤ 0.224, and **gsm8k exact match 0.000
on every arm** (n=12).

More data of the same kind will not fix this, and neither will more KD:

> On-the-fly full-vocab KD distills the teacher's distribution **over the
> dataset's own target tokens**. If the target is a 3-word extractive span from
> SQuAD, KD teaches the teacher's *uncertainty about that span* — it never
> teaches the teacher's answering behavior.

Replacing the target text with a **verified-correct teacher answer** is
rejection-sampled sequence-level KD (STaR / RFT family): the student is trained
on teacher behavior only where that behavior is demonstrably right.

## The rule

> **A v1 target is replaced only by a teacher answer that passes a mechanical
> correctness check against a gold key. Every other sample keeps its v1 public
> target, unchanged.**

Consequences: no unverified teacher text enters training; no sample is dropped
and no group is resized, so `stage2_offline_v2` stays sample-for-sample
comparable with v1 and with the four logged runs; and the **accept rate is a
headline result**, because it measures the teacher on our own prompts.

**The gold key needs no schema change: the v1 target *is* the key.** The v1
builders wrote the reference answer as the assistant message
(`scripts/build_stage2_v1.py`) — a SQuAD/HotpotQA span, a gsm8k `The answer is
N.`, an OpenMathInstruct `\boxed{…}`. The checkers already exist in
`src/aadistill/behavior.py` (`normalize_text`, `contains_gold`, `is_refusal`,
`final_number`); the only new code is a ~6-line `boxed_answer()` (P1).

## Scope — what gets rewritten, and how correctness is checked

| slice | group | train n | gold key (from the v1 target) | accept rule | checker |
|---|---|---:|---|---|---|
| squad_v2 answerable | `rag_evidence` | 9,635 | extractive span | normalized span appears in the answer | `contains_gold` |
| hotpot_qa | `multihop_qa` | 1,074 | short answer | containment; **yes/no golds must be the leading token** | `contains_gold` + lead-token rule |
| squad_v2 unanswerable | `refusal_uncertainty` | 7,605 | "not answerable from this context" | refusal, ≤ 60 words, terminated, asserts no span | `is_refusal` |
| gsm8k | `code_math` | 7,149 | final number of the v1 target | **exact match** on the final number | `final_number` |
| openmath_instruct_2 | `code_math` | 4,344 | `\boxed{…}` content | normalized boxed-answer match | `boxed_answer` (new) |
| **total candidates** | | **29,807** | | | |

Plus the existing hygiene rejections on every slice: empty, non-terminating,
over-length, language-mismatched, template-marker contaminated (`<|im_start|>`,
`####`, `<<…>>`).

**Out of scope, with reasons:** mbpp / magicoder_oss (359 / 1,392 — correctness
means executing tests in a sandbox); `tool_calling` (7,127 — gold calls are
already schema-validated, rewriting risks the format we are training);
`instruction` / `short_realtime` (10,752 / 14,251 — **no mechanical correctness
key exists**, and a judge was rejected on 2026-07-27); `long_context` (796 — no
targets). Candidates are 46% of the 64,484-sample train split.

## Generation algorithm — top-n sample, verify all, select one

1. **Candidates per prompt: `n = 4` (recommended), candidate 0 greedy.**
   Candidate 0 is the deterministic greedy answer; candidates 1…n-1 are sampled
   (temp 0.7, top_p 0.95, per-candidate seeds logged). Sampling, not beam search:
   beams return near-duplicates, so they raise cost without raising accept@n,
   which is the entire point of generating more than one.
2. **Every candidate is verified independently** against the slice's gold key.
   Reported per slice: **accept@1** (greedy alone) and **accept@n** — the gap is
   what the extra sampling buys.
3. **Selection rule among verified-correct candidates (default):** take the
   greedy answer if it verified; otherwise the **median-length** correct
   candidate, tie-broken by lowest candidate index.
   *Why not "shortest correct":* on the math slices that systematically selects
   answers that skip the derivation (`The answer is 42.`), training the student
   to state answers without working them out — the opposite of what the slice is
   for. The pilot reports the length distribution of correct candidates per
   slice, and the final rule is confirmed from that data before the bulk run.
   Only one target per prompt is kept; see Alternatives for why correct
   candidates are not shipped as extra samples.
4. **The teacher runs in its native thinking mode — always** (maintainer
   directive, 2026-07-28; decision record same date). Earlier drafts prefilled a
   closed `</think>` so the Thinking-2507 teacher would answer directly and
   cheaply. That is now ruled out: this is a distillation framework, so the
   teacher must be used and judged at its **actual capability**, not in a mode we
   suppressed to save money. Suppressing the think block was also the single most
   likely explanation for a poor accept rate on the math slices — the plan would
   have measured our own prompt convention and blamed the teacher.
   **Consequence, stated plainly:** each candidate now costs ~1k–3k reasoning
   tokens on top of its answer (~10× decode, ~40× once multiplied by n=4), which
   is the dominant cost of this experiment and the reason the pilot now also
   measures the *thinking-length distribution* per slice. Verification is
   unaffected: the correctness check reads the answer after `</think>`.
5. **Reproducibility (P5).** Batched sampled generation is not bitwise
   reproducible, so the corpus is the artifact and its hash pins the experiment;
   seeds, engine, engine version, batch layout and decode config are logged, and
   **all n candidates plus their verdicts are written to a sidecar file** so the
   selection can be re-derived without regenerating.
6. **Splits stay frozen.** val and calib keep their public targets — rewriting
   them would silently change what `val_v0`/`val_v1` measure and destroy
   comparability with all four logged runs.
7. **Provenance per sample:** `target_source` (`teacher_verified` | `v1_public`),
   candidate index chosen, accept@1/accept@n for its slice, teacher id +
   revision, decode config, thinking mode + token budget, thinking length,
   prompt-template hash, source sample id, reject reasons for the discarded
   candidates.
8. **Open question — what becomes the target text.** The teacher now produces a
   reasoning trace *and* an answer, and the student's current convention is an
   empty think block (decision 2026-07-21, tied to the realtime latency target).
   Three options, and this needs a maintainer decision before phase 1b:
   **(A) answer-only** — keep the trace out of the corpus; the student keeps its
   direct-answer convention and inherits answers that were *produced* by
   reasoning. No training-side change, no latency change. **Recommended for this
   experiment.**
   **(B) full trace** — train the student to reason. Targets grow ~10×, so the
   same 2700-step budget covers roughly a tenth of the samples (blocks are packed
   at `block_len` 1024, so a 1.5k-token trace simply spans blocks); it also
   changes the student's inference latency, its chat convention, and every
   behavior baseline measured so far.
   **(C) hybrid** — traces only on the reasoning-heavy slices (gsm8k, openmath,
   multihop), answers elsewhere.
   B and C are a different experiment, not a variant of this one: they change the
   student's output contract. They should be their own proposal, informed by
   whether (A) moves the behavior scorecard at all.

## Execution plan and cost (AGENTS.md P8, P8.2)

**Prerequisite A — `eval_behavior_v1` prompt-set expansion. Free, CPU, no
approval needed.** The Stage 3 verdict is read on the behavior scorecard, whose
noise floor is ±0.11 at n=76 and ±0.25 at n=12 (95% Wilson, p≈0.5): a
0.000 → 0.083 move in `answer_em_credited` is **one prompt**. Expand to ~36
prompts/group from held-out val, keeping the v0 prompts as an exact subset so the
four logged scorecards stay comparable (report both rows). Floor per group
becomes ±0.155.

**Prerequisite A2 — score the teacher on `eval_behavior_v0`, in thinking mode.
≈1 h on 1× L40S, ~$1–1.5.** *The single cheapest thing on this list, and the one
the figure is currently missing:* the project has never measured its own teacher,
so the chart has a student point and no ceiling. Run the standard harness with
**no prefill and `--max-new-tokens 4096`** so the reasoning fits; truncation rate
is reported, not hidden. Not possible on the dev box: 76 prompts × ~1.5k tokens
at the measured 1–3 tok/s is 10–40 h of a machine that is also the development
environment. Record mode and cap in the scorecard — a thinking row and a
direct-answer row are never mixed silently.

**Prerequisite B — top-n pilot. ~$3–5 with thinking on** (was ~$2 with the
suppressed think block), attachable to any approved session. 200 prompts ×
5 slices × n=4 = 4,000 candidates × ~1.65k tokens ≈ **6.6M output tokens**
(2–5 h on 1× L40S). Deliverables: per-slice **accept@1 and accept@n**,
reject-reason histograms, correct-candidate length distributions (which fix the
selection rule), **the thinking-length distribution per slice — now the dominant
cost driver and the biggest unknown in C**, and the engine decision.
Pre-registered gates: a slice with **accept@n < 0.5** is not rewritten in bulk
without an explicit decision (a corpus half teacher-styled and half
annotator-styled may train worse than either); **< 0.2** is dropped. If
accept@n ≈ accept@1, drop top-n and run the cheap n=1 path.

**Prerequisite C — bulk verified generation. Needs the mixture-change approval
(AGENTS.md 4.4) and a dependency approval — and its cost is now the open
question.** With thinking on, 29,807 prompts × n=4 = 119,228 candidates ×
~1.65k tokens ≈ **197M output tokens**, versus 15.5M under the suppressed-think
plan. At the repo's own logged vLLM throughput estimate (390–780 tok/s for this
teacher) that is **70–140 h ≈ $70–140** — an order of magnitude more than the
$6–11 the previous draft quoted, entirely because the teacher is no longer
crippled to make it cheap.

Ways to bring that down **without** touching the thinking rule, in the order the
pilot should decide them:

| lever | effect | note |
|---|---|---|
| **n=1 or 2 instead of 4** | ÷4 or ÷2 → **$18–70** | thinking should raise accept@1 substantially; if it does, top-n buys little. The pilot measures exactly this. |
| **reasoning-heavy slices only** (gsm8k, openmath, multihop = 12,567) | ≈83M tokens → **$30–58** | the slices where a reasoning teacher plausibly beats the public target; rag/refusal targets are short spans where thinking may add nothing |
| **cap the prompts per slice** (e.g. 3,000) | scales linearly | keeps every slice represented; the accept rate, not the volume, is what the experiment tests |
| faster accelerator (H100) | ~2–3× faster at ~2–3× the price | roughly a wash; not recommended |

**Recommendation: do not approve C as a blank cheque.** Approve the pilot, then
size C from measured thinking lengths and accept@1 — the pilot converts a
$70–140 unknown into a scoped number. On the CPU dev box C is years; CPU is good
for a 2-sample correctness smoke, not for the build.

**The Stage 3 experiment itself — $4–5.** 2700 steps from
`s2v1_from_init@2700`, seed 20260726, config identical to
`configs/stage3_s2v1_from_init.json` except the data root — one meaningful
field, verified by diff (P6: budget fixed before the run). Gates as usual:
holdout_v1 bf16, INT8 at both scopes, behavior scorecard, no non-finite losses,
resume path. Attach the **variance measurement** (STATE next action) to the same
session — one extra leg on a pod already paid for.

End to end, with the teacher at full capability: **$5–7 committed now** (the
teacher scorecard + the pilot), and **$25–145 for the build + comparison run**,
scoped by what the pilot measures. The previous $12–22 assumed a teacher with its
reasoning switched off, which is no longer on the table.

## Pre-registered reading of the results

- **`eval_behavior_v0/v1` (credited variants) is the primary metric**:
  `format_ok`, `empty_answer`, `think_closed`, `rep_3gram`, `answer_words`,
  `evidence_hit_credited`, `refusal_credited`, `answer_em_credited`.
- **val_v0 / val_v1 CE is expected to rise. That is not a regression** — it
  measures a target distribution we chose to leave behind.
- **holdout_v1 is expected to be flat (±1%).** A large drop is the abort signal.
- **Noise floor, pre-registered.** 95% Wilson half-widths at p≈0.5: n=76 →
  **±0.11** (≈8 prompts), n=36 → ±0.155, n=12 → **±0.25** (≈3 prompts), n=4 →
  ±0.35. The verdict rests on aggregate rows; per-group rows may only support it
  when the move exceeds the group's floor.
- **Adopt v2 only if** the behavior scorecard improves on the aggregate rows
  while holdout_v1 stays within ±1%.
- **Report accept@1, accept@n and the fallback fraction next to the result.** A
  behavior gain on a corpus where only 40% of targets were rewritten is a
  different claim from one where 90% were.

## Risks

- **Verification-induced selection bias.** Teacher style appears only where the
  teacher was right, i.e. on easier items; the student may learn "teacher voice
  on easy questions, annotator voice on hard ones". Top-n *reduces* this (higher
  accept@n → fewer fallbacks) but cannot remove it. Measurable after the fact via
  `target_source` against a difficulty proxy; gated by the accept-rate rule.
- **Selection rule can corrupt the training signal.** Choosing among n correct
  candidates is itself a design choice with teeth — "shortest correct" would
  strip derivations on math. Hence the median-length default and the pilot's
  length data before the bulk run.
- **Sampled targets are lower-quality than greedy on average.** Verification
  filters wrongness, not sloppiness: a sampled candidate can be correct and
  badly written. Mitigation: greedy is preferred whenever it verifies, so
  sampling only supplies targets the greedy pass could not.
- **The correctness key is a proxy.** Span containment is not truthfulness;
  final-number EM does not check the reasoning that produced it. Standard RFT
  caveats, and they bound how strongly "correct" can be read.
- **A bimodal corpus** (two target styles inside one group) may be worse than
  either alone — what the accept-rate gate is for.
- **Heavy dependency.** vLLM is GPU-only and large; installing it changes the
  reproducibility story for generation (P12 approval, pod-only, version pinned
  and logged in the corpus manifest). The HF path keeps the current story at 3–4×
  the cost.
- **Distribution narrowing / teacher error inheritance / corpus staleness /
  scope creep into Stage 4** — as before: greedy-modal targets are lower-entropy
  than human data (monitored by degeneracy scoring); accepted answers can still
  carry wrong side-claims; targets are pinned to one teacher revision; student
  rollouts and preference data remain Stage 4.
- **License / share-alike.** Recorded in the v1 manifest: SQuAD v2 and HotpotQA
  are **CC-BY-SA-4.0** → share-alike propagates to derived data; gsm8k is MIT;
  OpenMathInstruct-2 is **CC-BY-4.0 "Built with Llama"**, so the math slice adds
  a second synthetic-data lineage (Llama-derived prompts, Qwen-derived targets)
  whose attribution obligations need checking before any release. jsonl stays
  gitignored; flagged for release-time review.

## Alternatives considered

- **Ship every correct candidate as its own training sample** (n-fold
  augmentation on the items the teacher answers well). Rejected: it resizes
  groups by the teacher's own competence, breaking mixture balance and
  like-for-like comparison with v1 — and it amplifies selection bias instead of
  reducing it.
- **Beam search instead of sampling.** Rejected: near-duplicate beams cost n×
  without raising accept@n.
- **Drop rejected samples** instead of keeping the v1 target. Rejected for now:
  uneven group shrinkage, same comparability problem. Revisit if the pilot shows
  a bimodal-corpus effect.
- **Grounding filters only** (the 2026-07-27 draft): trains on answers that are
  well-formed and wrong — ruled out by the directive.
- **An LLM judge** to extend correctness to the chat groups: rejected 2026-07-27
  (per-gate cost, not reproducible from stored artifacts).

## Decision requested

1. ~~Approve building `eval_behavior_v0` first~~ — **done 2026-07-27** (free, CPU).
2. **Approve the teacher scorecard (prerequisite A2) — ~$1–1.5, ≈1 h.** Thinking
   mode, no prefill, cap 4096, same 76 prompts. This is what puts the teacher
   into the comparison and gives the figure its ceiling; it needs no other
   decision and nothing downstream depends on it. **Recommended: yes, first.**
3. **Approve the top-n pilot (prerequisite B) — ~$3–5.** 1,000 prompts × n=4 in
   thinking mode. Returns accept@1/accept@n, selection-rule data, the
   thinking-length distribution, and the engine decision. Can share the pod with
   (2). **Recommended: yes.**
4. **Decide the target text: (A) answer-only, (B) full trace, (C) hybrid** — see
   Generation design §8. **Recommended: (A) for this experiment**; B/C change the
   student's output contract and its latency budget, so they belong in their own
   proposal. *Needed before the bulk build, not before the pilot.*
5. Approve bulk generation (prerequisite C) — **changes the official data
   mixture** (AGENTS.md 4.4), **$25–145** depending on n and slice scope.
   *Decide with the pilot's measured thinking lengths and accept rates in hand;
   not a blank cheque.*
6. Approve **vLLM as a pod-only dependency** if the pilot points that way
   (AGENTS.md P12 heavy-dependency approval). *Decide with the pilot's throughput
   numbers.*
7. Approve the Stage 3 comparison run — **$4–5**, sharing a session with the
   variance measurement. *Decide after C.*

## References

- Kim & Rush, *Sequence-Level Knowledge Distillation*, EMNLP 2016
  ([arXiv:1606.07947](https://arxiv.org/abs/1606.07947)) — training the student
  on the teacher's generated targets rather than on gold targets reweighted by
  the teacher. The method this experiment instantiates.
- Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning*, NeurIPS 2022
  ([arXiv:2203.14465](https://arxiv.org/abs/2203.14465)) — keep only generations
  whose final answer matches the reference. The correctness gate is this filter
  applied to a teacher rather than to the model itself.
- Yuan et al., *Scaling Relationship on Learning Mathematical Reasoning with
  Large Language Models*, 2023 ([arXiv:2308.01825](https://arxiv.org/abs/2308.01825))
  — rejection-sampling fine-tuning with k samples per prompt: the source of both
  the top-n recipe and the selection-bias/diversity caveats recorded above.
