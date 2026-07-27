# PROPOSAL (needs user approval) — Stage 2 v2: teacher-generated answers (sequence-level KD)

Status: **UNBLOCKED — awaiting maintainer approval.** Updated 2026-07-27.
The prerequisite below (`eval_behavior_v0`) is **built, committed and run**
(`logs/experiments/2026-07-27_eval_behavior_v0.md`), so this experiment is now
measurable and the "before" numbers exist for every candidate branch point.
The start-point ablation also settled which checkpoint to branch from:
**`s2v1_from_init@2700`** (`logs/decisions.md`, 2026-07-27). Two approvals are
still needed and are *not* implied by any earlier approval: this changes the
**official data mixture** (AGENTS.md 4.4) and it needs **$2–4 of GPU** for
phase-1 generation plus **$4–5** for the comparison run.

The behavior numbers strengthen the argument below rather than weaken it: all
three ablation arms share one defect the mixture cannot fix by adding volume —
verbose, repetitive, non-terminating answers (`rep_3gram` 0.35–0.41,
`answer_words` 199–231, `truncated_at_cap` 0.58–0.67), with `format_ok` at
most 0.224 and gsm8k credited EM at 0.000 across the board.

Drafted 2026-07-27 from
the `s2_blocks_v1` review
(`logs/experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md`); deferred once
already by the 2026-07-26 mixture-v1 decision ("teacher-generated data — a
separate future proposal").

## The argument

After the mixture-v1 recovery run, the student's language modeling is
substantially better (holdout NLL 4.21 → 3.80, 26% of the teacher gap closed)
while its *answers* are still wrong-shaped: it restates the question instead of
answering, emits a stray `</think>`, and states a confidently wrong fact.

More data of the same kind will not fix this, and neither will more KD, because
of what the loss actually does:

> On-the-fly full-vocab KD distills the teacher's distribution **over the
> dataset's own target tokens**. If the target is a 3-word extractive span from
> SQuAD, KD teaches the teacher's *uncertainty about that span* — it never
> teaches the teacher's answering behavior. The student is optimizing to imitate
> squad/hotpot/dolly annotators, with the teacher only reweighting them.

The mixture's answer text is public-SFT-grade by construction: terse extractive
spans (`rag_evidence`, `multihop_qa`), 12 cycled hand-written strings
(`refusal_uncertainty`), and noisy crowd prose (dolly/oasst2). Replacing the
*target text* with the teacher's own answer — classic sequence-level KD — is the
lever that matches the training target to the model we are distilling from and
to the format we intend to deploy.

This is Stage 2 work (off-policy prompts, teacher answers), deliberately not
Stage 4 (student rollouts + verifier/teacher correction). It is much cheaper,
needs no rollout or reward machinery, and fixes a defect that Stage 4 would
otherwise inherit and amplify.

## Prerequisite (free, no approval needed): `eval_behavior_v0` — **DONE 2026-07-27**

**This experiment was unmeasurable when the proposal was written, and that was
the main reason it was not just started.** holdout_v1 is fineweb-edu general web
text; teacher-style answers are expected to leave it **flat by construction**.
The only behavior signal in the project was three eyeballed prompts.

`eval_behavior_v0` now exists: 76 held-out prompts over 7 chat groups, mechanical
scorers only, committed at `data/eval_behavior_v0/` with a manifest, run at every
recovery gate (`scripts/eval_behavior.py`; build log
`logs/experiments/2026-07-27_eval_behavior_v0.md`, design decisions in
`logs/decisions.md`). Before-numbers exist for s1@660, `s2_blocks_v1@2700` and
both ablation arms, all scored on one GPU in one session.

Two rules from building it apply directly to reading this experiment:

- **Compare on the `_credited` metrics.** These prompts embed their own answer
  material, so a parroting student scores `evidence_hit` and `refusal` for free.
  Teacher-written targets are *expected* to reduce echo, which will move raw and
  credited metrics in different directions — only the credited ones are evidence.
- **Score every arm on the same device in the same session** (CPU vs GPU greedy
  decoding differs by 1–4 prompts on a damaged student).

## Scope — phase 1 (recommended)

Rewrite **train targets only**, for the three groups whose targets are furthest
from deployment behavior:

| group | train samples | why rewrite | expected output len |
|---|---:|---|---:|
| `rag_evidence` | 9,635 | SQuAD spans → grounded conversational answers | ~80 tok |
| `multihop_qa` | 1,074 | HotpotQA spans → answers that show the hop | ~120 tok |
| `refusal_uncertainty` | 7,605 | 12 cycled templates → varied natural refusals | ~60 tok |
| **total** | **18,314** | | ≈1.4M output tokens (plus ≈5M prefill) |

Left unchanged in phase 1: `code_math` (OpenMathInstruct/Magicoder targets are
already conversational, and gsm8k answers are verifiable references we would
rather not launder through the teacher), `tool_calling` (schema-validated JSON
targets — teacher rewriting risks breaking the very format we are training),
`long_context` (plain LM text, no targets), `instruction` + `short_realtime`
(largest volume, 33M chars, and smoltalk is already well-formed chat — phase 2
if phase 1 pays off).

## Generation design

1. **Prefill a closed, empty think block.** The Thinking-2507 generation prompt
   ends with `…assistant\n<think>\n`; continuing it with the closing `</think>`
   makes the teacher answer directly, exactly matching the empty-think target
   convention the student is already trained on (decision 2026-07-21). Without
   this, a Thinking teacher emits 1k–3k reasoning tokens per sample — ~10× the
   cost, in a style we deliberately do not train. A CPU assertion will verify
   that the teacher's rendered prompt + prefill is a **prefix-exact match** of
   the training-time rendering of that sample, so the teacher answers in the
   same position the student is trained to answer.
2. **Greedy decoding** (`do_sample=false`), fixed `max_new_tokens` per group,
   logged batch size. Chosen for reproducibility over diversity (P5); note that
   batched padded generation is not bitwise-identical to unbatched, so the
   corpus itself is the artifact and its hash is what pins the experiment.
3. **Grounding filters — the safety net that makes this honest.** The public
   gold answer is kept as a verification key, not as a target:
   - `rag_evidence` / `multihop_qa`: reject the teacher answer unless the
     normalized gold span appears in it. This is what stops the teacher from
     hallucinating away from the evidence and us from training that in.
   - `refusal_uncertainty`: the source items are SQuAD-unanswerable, so reject
     any teacher answer that *does* answer; keep the template target for
     rejected items rather than dropping the sample.
   - All groups: reject empty, non-terminating, over-length, language-mismatched,
     or template-marker-contaminated outputs (the existing builder hygiene rule).
   - Per-group **accept rates are a reported gate metric**, not a footnote; a low
     accept rate is itself a finding about the teacher on that group.
4. **Splits stay frozen.** val and calib keep their public targets. This is a
   hard invariant: rewriting val targets would silently change what val_v0 and
   val_v1 measure and destroy comparability with all four logged runs. An
   optional `val_teacher_v2` slice can be added as a *named extra_val set* for a
   like-for-like teacher-target curve.
5. **Provenance recorded per sample**: teacher id + revision, decode config,
   prefill string, prompt-template hash, source sample id, accept/reject reason.

## Pre-registered reading of the results (important)

Training on teacher-style targets will make the student **worse at predicting
public-style targets**. Therefore:

- **val_v0 / val_v1 CE is expected to rise under v2 training. That is not a
  regression** — it is the metric measuring a target distribution we chose to
  leave behind. Pre-registering this now prevents a correct result from being
  read as a failure later.
- **holdout_v1 is expected to be flat** (±1%). A large *drop* would be evidence
  of damage to general language modeling and is the abort signal.
- **`eval_behavior_v0` is the primary metric**: question-echo rate, chat-format
  validity, evidence containment, refusal rate on unanswerable prompts. Adopt v2
  only if it improves the behavior scorecard while holdout_v1 stays within ±1%.

## Cost and hardware (AGENTS.md P8.2)

**GPU-required in practice.** Generating ≈1.4M answer tokens (plus ≈5M tokens of
prefill, much of it long RAG/multi-hop context) on the CPU dev box at the
measured ~1–3 tok/s decode of a 4B bf16 model is ≥8 days of wall clock; CPU is
only good for a 2-sample correctness smoke, not for the build.

| path | dependency | est. runtime (1× L40S) | est. cost |
|---|---|---|---|
| **HF transformers batched generate (recommended)** | none — reuses the proven pod env | 2–4 h | **$2–4** |
| vLLM serving path | new heavy GPU-only dep | 0.5–1 h | $0.5–1 |

Recommendation: **transformers, no new dependency.** vLLM only pays for itself
at phase-2 volume (25k longer prompts), and installing it on the pod only
(never in the dev-box lockfile) keeps CPU reproducibility intact if we get
there.

**Cheapest sequencing:** attach generation to the tail of an already-approved
GPU session (e.g. the start-point ablation pod, after the training arms finish)
— setup is already paid for, so the marginal cost is ~$2–3.

Full experiment cost if run end to end: generation $2–4 + one 2700-step
comparison run $4–5 ≈ **$6–9**, plus the free CPU work (v2 build, gates,
behavior eval).

## Risks

- **Teacher error inheritance.** The teacher is wrong sometimes; grounding
  filters catch unfaithful RAG answers but not confidently wrong world facts on
  other groups. Phase 1's groups are all evidence-grounded or refusal, which is
  the deliberate mitigation.
- **Distribution narrowing.** Greedy single-sample targets are lower-entropy
  than human data; combined with KD this can make the student bland. Monitored
  via degeneracy scoring in the behavior eval.
- **License/share-alike.** Qwen3 is Apache-2.0, so its outputs are usable, but
  the prompts/contexts carry their source licenses (SQuAD v2 and HotpotQA are
  CC-BY-SA-4.0 → share-alike propagates to derived data). Recorded in the
  manifest, jsonl stays gitignored, flagged for release-time review.
- **Corpus staleness.** The targets are pinned to one teacher revision; changing
  teachers later invalidates them (unlike the on-the-fly KD path, which is why
  v0/v1 avoided cached teacher outputs).
- **Scope creep into Stage 4.** This proposal stops at off-policy prompts.
  Student rollouts, verifiers, and preference data remain Stage 4.

## Decision requested

1. ~~Approve building `eval_behavior_v0` first~~ — **done 2026-07-27**, no
   approval was required (free, CPU). Both blockers named in item 2 are now
   cleared: the eval exists, and the ablation fixed the branch point.
2. Approve phase-1 teacher generation (18.3k prompts, 3 groups, transformers
   path, $2–4, ideally attached to an approved session)? **Recommended: yes.**
   Note this is the approval that **changes the official data mixture**
   (AGENTS.md 4.4) — the generated targets replace public-SFT targets in three
   groups, creating `stage2_offline_v2`. v1 stays frozen and comparable.
3. Approve the follow-up comparison run (2700 steps on v2 vs the logged v1
   result, $4–5)? Can be decided later, once phase-1 accept rates are known.

## References

- Kim & Rush, *Sequence-Level Knowledge Distillation*, EMNLP 2016
  ([arXiv:1606.07947](https://arxiv.org/abs/1606.07947)) — the method this
  proposal is an instance of (train the student on the teacher's own generated
  targets rather than on gold targets reweighted by the teacher). Added to the
  README reference table as `queued`.
