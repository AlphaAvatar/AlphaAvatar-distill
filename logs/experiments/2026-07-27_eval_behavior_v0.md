# 2026-07-27 — `eval_behavior_v0`: mechanical behavior eval + first baselines

- **Agent:** Claude Code (Opus 5), autonomous session; built before the
  start-point ablation so that session had a behavior gate to run.
- **Git commit:** implemented and committed as `f3d7547`; design decisions and
  the device-comparability rule in `79f32df`.
- **Stage:** Stage 3 support milestone (AGENTS.md P10 — evaluation must
  eventually cover RAG reading, tool calling, refusal/uncertainty, code/math and
  short realtime responses). Queued as next-action 1 in `logs/STATE.md`.
- **Objective:** Give the project a behavior signal. Until now the only one was
  three eyeballed generation-smoke prompts, while the gate metric (`holdout_v1`,
  fineweb-edu NLL) is nearly blind to chat-format discipline, grounding, refusal
  and tool-call validity — the defects actually observed.
- **Hypothesis:** The recovery runs' NLL gains overstate their behavioral
  progress, and a mechanical eval will show it.
- **Hardware:** CPU-only dev box for the build and first baselines (~5.5 min per
  checkpoint at the 200-token cap, ~13 min at 512); GPU baselines produced in
  the 2026-07-27 ablation session. CPU-suitable per P8.2.
- **Budget:** free (CPU only; no GPU spend attributable to this milestone —
  the reference scorecards rode along on an already-approved session).

## What was built

- `src/aadistill/behavior.py` — mechanical scorers, no LLM judge, so a scorecard
  is free, deterministic, and reproducible from the stored raw generations:
  chat-format validity against the think-block contract the training data
  teaches (the prompt opens `<think>`, so a generation must close it exactly
  once), question-echo (prompt 4-gram overlap), degeneracy (1 − distinct-3),
  evidence containment, refusal detection, tool-call JSON + schema validity, and
  gsm8k final-answer exact match.
- `scripts/build_eval_behavior_v0.py` — deterministic prompt-set builder.
- `scripts/eval_behavior.py` — scores a checkpoint (greedy, batch 1, fixed
  `max_new_tokens`), writes a scorecard JSON + the raw generations.
- `data/eval_behavior_v0/` — **76 prompts + manifest, both committed** (140 KB).
- `tests/test_behavior_toy.py` — 16 tests; suite went 75 → 91.

## Prompt set

76 prompts over 7 chat groups, sampled deterministically from the **val** splits
of mixture v1 (never trained on): 12 each for `instruction`, `rag_evidence`,
`tool_calling`, `refusal_uncertainty`, `code_math`, `short_realtime`, and 4 for
`multihop_qa`. Each prompt is the conversation prefix up to the first assistant
turn; that turn is the gold.

- `long_context` is **excluded**: it is `format=="text"` fineweb-edu continuation
  data with no conversation to prompt from.
- `tool_calling` is restricted to samples whose first assistant turn is a tool
  call, so the tool scorers apply to all 12.
- Prompt cap **1024 tokens = `block_len`**, so no prompt sits outside the
  contiguous-context regime the student was trained in. Cost: only 4 of 25
  `multihop_qa` val samples fit (hotpot p50 is 1515 tokens), so that group's row
  is indicative only. Recorded in the manifest.

## Three findings from building it

These are the reason the eval is worth its complexity; two are measurement bugs
that would have corrupted every future comparison.

1. **Echo credit (load-bearing).** The rag/refusal prompts embed both the gold
   span *and* the instruction "say you cannot answer from the context", so a
   student that parrots the prompt scores `evidence_hit` and `refusal` for free.
   This is not hypothetical — it is what s1@660 does. Every content metric now
   ships as a raw check plus a `_credited` variant requiring a non-empty,
   non-echoed answer (4-gram overlap < 0.5). On s1@660 the difference is
   `evidence_hit` **0.667 → 0.167 credited**, and `refusal` **1.000 → 0.250**.
   **Comparisons use the credited variants.**
2. **Truncation ≠ non-termination.** At the initial 200-token cap, **100%** of
   `s2_blocks_v1`'s non-terminations were cap hits, which made `terminated` a
   verbosity measure rather than a format one. The cap is now 512 and
   `truncated_at_cap` is recorded separately; at 512, s1@660 still truncates on
   **54%** of prompts — genuine runaway.
3. **Scorecards are only comparable within one device.** Scoring the same
   checkpoints on the CPU dev box and on an L40S — identical prompts, cap and
   code — moved `format_ok` by 1–3 prompts (s1@660 0.079 vs 0.105;
   `s2_blocks_v1` 0.053 vs 0.066) and `tool_call_parsed` 0.000 vs 0.083, while
   `terminated`/`truncated_at_cap` matched **to the digit on both checkpoints**.
   Greedy decoding is deterministic per device, but bf16 kernel differences flip
   tokens on a model this damaged. Whether it stops is device-stable; what it
   says is not. This is why `scripts/pod/score_refs.sh` re-scores references on
   the pod instead of reusing dev-box baselines (P5).

The refusal detector is calibrated at build time: the builder fails loudly
unless it fires on ≥95% of the split's gold refusals (measured **1.000**).

## First baselines (GPU, L40S, 2026-07-27 session — comparable rows)

| | s1@660 | s2_blocks_v1@2700 |
|---|---|---|
| holdout_v1 NLL | 4.2107 | **3.8003** (−9.8%) |
| format_ok | **0.105** | 0.066 |
| terminated | **0.461** | 0.329 |
| think_closed | 0.263 | **0.316** |
| empty_answer | 0.605 | **0.382** |
| answer_is_echo | **0.171** | 0.263 |
| rep_3gram | **0.208** | 0.351 |
| answer_words | 111.3 | 199.2 |
| tool_call_parsed | **0.083** | 0.000 |
| rag evidence_hit *credited* | **0.167** | 0.000 |

**Verdict: the hypothesis held.** The checkpoint that improved `holdout_v1` by
9.8% is *worse* on most behavior axes — it writes nearly twice as much, repeats
more, terminates less, and emits no parseable tool call. The NLL gain bought
knowledge, not answering behavior. This result is what made the target-style
diagnosis measurable rather than eyeballed, and it directly motivated leaving
the teacher-generated-answer proposal as the next lever.

## Caveats

- Mechanical scorers measure **form and grounding, not answer quality**. A model
  can score well here and still be unhelpful: a high score is weak evidence, a
  low score is strong evidence of a defect.
- n=76 (12 per group, 4 for `multihop_qa`). A 0.08 delta in an overall metric is
  ~6 prompts. Group rows are noisy; drive decisions from overall rows and from
  agreement across metrics.
- The refusal detector is a regex with 1.000 recall on gold refusals but
  **unmeasured precision** on student text.
- No pre-registered decision band exists for these metrics, unlike `holdout_v1`'s
  1%. Deltas reported anywhere are exploratory until one is set.

## Reproduce

```
uv run python scripts/build_eval_behavior_v0.py          # rebuilds prompts.jsonl + manifest
uv run python scripts/eval_behavior.py --model <ckpt> --out <scorecard.json>
uv run pytest tests/test_behavior_toy.py -q              # 16 tests
```

## Next action

Run it at every recovery gate (already wired into `scripts/pod/post_run.sh`).
Set a decision band for the format metrics once GPU run-to-run variance is
measured. Related: `logs/decisions.md` (2026-07-27, `eval_behavior_v0` record),
`logs/experiments/2026-07-27_stage3_start_point_ablation.md` (first use as a
gate, where it reversed the holdout ranking).
