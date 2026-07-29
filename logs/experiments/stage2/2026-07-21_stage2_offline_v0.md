# 2026-07-21 — Stage 2: offline warm-up mixture v0 (`stage2_offline_v0`)

- **Agent:** Claude Code (Fable 5) session, continuing the first dense-model
  compression experiment (teacher `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d`,
  student target 0.6B-class per 2026-07-13 decision).
- **Git commit:** built on `f3c7a24` (dirty during the run — this session's
  Stage 2 implementation; committed immediately after, see repo history for
  the commit containing this log).
- **Stage:** Stage 2 — offline warm-up data collection (AGENTS.md 4.4).
- **Objective:** Create the grouped offline training mixture that Stage 3
  recovery will consume, plus the loader path (rendering, loss masks,
  packing) the trainer will use, and pass the Stage 2 validation gate.
- **Hypothesis:** n/a (infrastructure/data milestone, no model claim).
- **Hardware:** CPU-only dev box (16 threads, 30 GB RAM, no GPU). Everything
  in this stage is CPU-suitable per AGENTS.md P8.2.
- **Budget:** ~21M chars (~5M tokens) across eight groups; actual download
  footprint modest (streaming for hotpot/glaive/fineweb; HF cache grew to
  12 GB total including the pre-existing 7.6 GB teacher).

## Commands

```
uv run python scripts/build_stage2_v0.py        # build mixture (network)
uv run python scripts/dry_run_stage2.py         # gate: loader dry run (12.1 s)
uv run pytest tests/ -q                         # 33 passed
```

## What was built

- `src/aadistill/data.py` — loader for Stage 3+: schema validation, chat
  rendering via the teacher chat template, **assistant-span loss masking by
  character offsets** (the Thinking-2507 template is not prefix-stable — see
  the 2026-07-21 rendering decision record), fixed-length block packing.
- `scripts/build_stage2_v0.py` — deterministic revision-pinned mixture
  builder; global content dedup; holdout_v1 leakage exclusion (fineweb
  stream offset ≥10000 + first-1000-char match); per-group train/val/calib
  modular split (calib across groups = INT8 calibration set).
- `scripts/dry_run_stage2.py` — gate script (schema, encoding, packing,
  determinism, tool-format checks) writing
  `artifacts/stage2/dry_run_v0_report.json`.
- `tests/test_data_toy.py` — 19 tests (validation, packing, loss-mask
  correctness against the real tokenizer, determinism, truncation).

## Data result

Mixture manifest: `data/stage2/stage2_offline_v0.manifest.json`
(sha256 `7b86d9178f0f2c1d…`); jsonl files gitignored, rebuildable.
Tokenizer `7781771acc3798ee…` (teacher tokenizer, matches Stage 0 manifest).

| group | source(s) | train samples | train tokens | trainable frac |
| --- | --- | ---: | ---: | ---: |
| instruction | dolly (long), oasst2 en threads | 3,716 | 1,258,802 | 0.61 |
| code_math | gsm8k, mbpp | 4,950 | 959,117 | 0.63 |
| tool_calling | glaive-fc-v2 (Qwen3 tool schema), v0 | 2,321 | 894,393 | 0.26 |
| multihop_qa | hotpot_qa distractor | 475 | 689,407 | 0.006 |
| long_context | fineweb-edu 8k–24k-char docs | 188 | 609,465 | 1.00 |
| rag_evidence | squad_v2 answerable | 2,394 | 571,615 | 0.04 |
| short_realtime | dolly (short, no-context) | 3,583 | 232,402 | 0.62 |
| refusal_uncertainty | squad_v2 unanswerable, v0 | 857 | 170,438 | 0.11 |
| **train total** | | **18,484** | **5,385,639** | 0.44 |

Val: 771 samples / 223,375 tokens. Calib: 120 samples / 65,657 tokens
(stratified across groups). Packing at block 1024: 5,256 train blocks.

Source hygiene: glaive 5 parse failures + 7 schema rejects out of ~2,500
consumed rows; 71 exact duplicates removed globally; 1 hotpot row over the
12k-char cap. All eight groups non-empty (builder fails loudly otherwise).

## Gate checklist (AGENTS.md 4.4)

- data manifests exist — **pass** (committed mixture manifest with per-file
  sha256/bytes/counts).
- dataset names, revisions, licenses, hashes logged — **pass** (all sources
  pinned to exact dataset revisions; licenses recorded per source).
- filtering and dedup rules logged — **pass** (caps, marker hygiene, global
  content-sha dedup, split rule, holdout exclusion — all in the manifest).
- teacher-generated data provenance — **n/a, documented**: v0 contains no
  teacher-generated data (see 2026-07-21 mixture decision record; KD targets
  will be computed on-the-fly in Stage 3).
- data loads in the intended training pipeline — **pass** with scope caveat:
  the Stage 3 trainer does not exist yet; the dry run exercised the actual
  loader module (`aadistill.data`: validate → render → mask → pack) that the
  trainer will consume.
- small-batch dry run passes — **pass** (`dry_run_v0_report.json`; all
  checks true; deterministic re-encoding verified).
- data mixture reproducible — **pass** (pinned revisions, native-order
  first-N selection, fixed split rule, logged command).
- known risks / license constraints recorded — **pass**, see below.

**Verdict: Stage 2 v0 gate passed.**

## Risks and notes

- **License note:** dolly (CC-BY-SA 3.0), squad_v2 and hotpot_qa
  (CC-BY-SA 4.0) are share-alike; the jsonl mixture is not redistributed
  (gitignored) and any future released dataset artifact containing them must
  honor share-alike + attribution. gsm8k MIT, mbpp CC-BY-4.0, glaive/oasst2
  Apache-2.0, fineweb-edu ODC-By 1.0.
- **Low trainable fraction in extractive-QA groups** (multihop 0.6%,
  rag_evidence 3.9%): long evidence contexts with terse span answers. Cheap
  as context for KD (teacher logits can be distilled on context tokens too —
  a Stage 3 loss-design choice), expensive per SFT-trainable token. Revisit
  mixture ratios once Stage 3 defines its loss mix.
- **Quality grade:** public-SFT-grade data; answers are terse for QA groups;
  no reasoning traces (empty-think targets by design, see rendering decision
  record). Teacher-generated corpora remain the main upgrade path and need
  user approval.
- ~5M tokens is warm-up scale for early recovery sub-stages (FFN/norm,
  block recovery), not full offline-KD scale. Scaling up is a budget change
  in `build_stage2_v0.py` + user approval.

## Next action

Stage 3 recovery design: trainer skeleton (data loader is ready), recovery
sub-stage order per AGENTS.md 4.5, on-the-fly KD loss, CPU toy loop first,
then a GPU execution request for the user (RunPod control plane verified
2026-07-16).
