# 2026-07-26 — Stage 2: offline mixture v1 scale-up (`stage2_offline_v1`)

- **Agent:** Claude Code (Fable 5) session, first dense-model compression
  experiment (teacher `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d`, student
  0.6B-class).
- **Git commit:** built dirty on top of `b0155af` (this session's v1
  implementation; committed immediately after, see the commit containing
  this log). Builder `code_state` is embedded in the mixture manifest.
- **Stage:** Stage 2 — offline warm-up data collection (AGENTS.md 4.4),
  second pass (scale-up), executed under the approved proposal
  `logs/proposals/2026-07-26_stage2_mixture_v1_scaleup.md`.
- **Objective:** Scale train data 5.39M → ~24M tokens to un-block
  data-limited Stage 3 recovery (2026-07-25 A/B verdict), fixing the
  observed corpus artifacts at the data level.
- **Hypothesis:** n/a (data milestone; the recovery hypothesis is tested by
  the next Stage 3 run).
- **Hardware:** CPU-only dev box (16 threads, 30 GB RAM). CPU-suitable per
  P8.2 (streaming downloads + tokenizer encoding only).
- **Budget:** approved ≤ 30M train tokens, ≤ 3 GB downloads, ≤ 1 h build.
  Actual: 22.13M train tokens; HF cache grew only ~0.3 GB (12 GB total —
  the large sources were streamed, not cached); build ~16 min wall.

## Commands

```
uv run python scripts/build_stage2_v1.py        # build (network); console log
                                                # in artifacts/stage2/build_v1_console.log
uv run python scripts/dry_run_stage2.py --data-dir data/stage2_v1 \
  --out artifacts/stage2/dry_run_v1_report.json # gate: loader dry run (~54 s)
uv run pytest tests/ -q                         # 69 passed
uv run python scripts/train_stage3.py --config configs/stage3_s2v1_smoke_cpu.json
                                                # 3-step CPU smoke incl. extra_val
```

## What was built

- `scripts/build_stage2_v1.py` — v1 builder: v0-carry + fresh offsets + five
  new sources; reuses the v0 builders (oasst2 threading, glaive parsing,
  fineweb filtering) through an offset-skipping sink; v0 script untouched.
  gsm8k normalization (strip `<<…>>`, `#### N` → "The answer is N.") applied
  to fresh rows and carried v0 train rows (4,591 normalized); refusal
  template pool widened 3 → 12; smol-smoltalk short-dialogue filter routes
  tiny conversations to short_realtime; xlam converted to the Qwen3 tool
  schema (query → parallel tool_calls; no tool responses in that source).
- `src/aadistill/train.py` + `scripts/train_stage3.py` — optional
  `extra_val` config: named secondary val sets, one `eval_result` event per
  set tagged `val_set` (primary = "val"); run_manifest now hashes the
  extra-val dirs' mixture manifests too.
- `scripts/dry_run_stage2.py` — now takes any mixture dir (single-manifest
  glob instead of the hardcoded v0 name).
- `tests/test_build_stage2_v1.py` (16) + 2 extra-val trainer tests → suite
  69 passed.
- `configs/stage3_s2v1_smoke_cpu.json` — 3-step CPU smoke of the new data
  path (v1 train/val + val_v0 extra_val), result below.

## Data result

Mixture manifest: `data/stage2_v1/stage2_offline_v1.manifest.json`
(committed; jsonl gitignored, rebuildable). Tokenizer = teacher tokenizer
(hash in the dry-run report, matches Stage 0/2 manifests). All sources
revision-pinned; access to the gated xlam repo verified before the build.

Train split (tokens measured by the gate dry run, block 1024):

| group | train samples | train tokens | trainable frac | ×v0 tokens |
| --- | ---: | ---: | ---: | ---: |
| instruction | 10,752 | 6,315,196 | 0.771 | 5.0 |
| code_math | 13,244 | 4,025,027 | 0.685 | 4.2 |
| tool_calling | 7,127 | 2,981,696 | 0.212 | 3.3 |
| long_context | 796 | 2,480,059 | 1.000 | 4.1 |
| rag_evidence | 9,635 | 1,908,192 | 0.047 | 3.3 |
| multihop_qa | 1,074 | 1,553,126 | 0.006 | 2.3 |
| refusal_uncertainty | 7,605 | 1,492,301 | 0.121 | 8.8 |
| short_realtime | 14,251 | 1,378,034 | 0.485 | 5.9 |
| **total** | **64,484** | **22,133,631** | **0.528** | **4.11** |

val_v1: 1,916 samples / 705,114 tokens. calib: 200 samples / 119,144 tokens
(v0's 120 frozen + 80 new stratified). Train blocks at 1024: **21,610**
(→ 2 epochs = 2,700 steps at 16 blocks/step). v0 val (771) and calib (120)
remain untouched in `data/stage2/`.

Per-source intake (details + counters in the manifest): oasst2 +1,552
threads; smol-smoltalk 5,787 instruction + 9,117 short; everyday-
conversations 2,005 (source fully consumed); squad_v2 +7,553 rag / +7,039
refusal; hotpot +634; glaive +3,272; xlam 1,744; gsm8k +2,675 (all
remaining rows, normalized); OpenMathInstruct-2 (`train_1M`) 4,524
(deduped by problem); Magicoder 1,451; fineweb +643 long docs.

Deviations from the proposal targets: train tokens 22.13M vs ~24M (−8%:
char→token ratio drift plus three logged source exhaustions — smoltalk-short
0.34M chars, everyday 0.07M, gsm8k 0.18M). Judged not worth a top-up
rebuild: the ≤2-epoch regime is preserved by sizing the next run at 2,700
steps instead of 3,000. Trainable fraction 0.528 vs ~0.52 estimated.

## Gate checklist (AGENTS.md 4.4)

- data manifests exist — **pass** (committed v1 manifest: per-file
  sha256/bytes/counts, per-source counters, carry stats).
- dataset names, revisions, licenses, hashes logged — **pass** (all eleven
  sources pinned to exact revisions with licenses; xlam recorded as
  CC-BY-4.0 auto-gated, accepted on the AlphaAvatar account 2026-07-26).
- filtering and dedup rules logged — **pass** (caps, marker hygiene,
  v0-seeded global content dedup = frozen-val/calib leakage guard, fresh
  offsets, split rule, holdout exclusion, gsm8k normalization, short filter).
- teacher-generated data provenance — **n/a, documented**: none in v1;
  third-party synthetic provenance (Llama/GPT-3.5 sources) recorded in the
  manifest with release-time license review flagged.
- data loads in the intended training pipeline — **pass**: 3-step CPU smoke
  of the actual Stage 3 trainer over v1 data incl. the new extra_val path
  (see `artifacts/stage3/s2v1_smoke_cpu/`).
- small-batch dry run passes — **pass**
  (`artifacts/stage2/dry_run_v1_report.json`: schema, rendering, masking,
  packing, determinism, tool-format checks all true).
- data mixture reproducible — **pass** (pinned revisions, deterministic
  first-N selection + offset skipping, fixed split rule, logged command;
  same-seed carry from committed v0 manifest state).
- known risks / license constraints recorded — **pass** (share-alike notes
  unchanged; synthetic provenance above; jsonl not redistributed).

**Verdict: Stage 2 v1 gate passed.**

## Risks and notes

- The v1 train split is not a byte-superset of v0 train: carried gsm8k rows
  are normalized (intended; both forms digest-seeded against re-entry).
- multihop_qa stayed terse-extractive (trainable frac 0.006, grew only
  2.3×) by design — conversational rewrite is the teacher-gen upgrade path.
- xlam contributes call-emission-only samples (no tool responses); glaive
  still provides full call/response loops (counters in manifest keep the
  two distinguishable via `source`).
- Killed the builder process after the manifest write: it hung in
  fineweb stream-teardown retries (`Errno 9` on a closed connection during
  generator cleanup); all outputs were flushed and hash-verified by the
  dry run before the kill. Harmless, but worth knowing for future builds.

## Next action

Stage 3 next recovery run on v1 data (attention-unfrozen freeze set,
start from `s2_blocks_v0` final per the A/B verdict): config
`configs/stage3_s2_blocks_v1.json` drafted; needs the usual per-session
GPU approval (~1× L40S, est. ≤ $5) plus HF pull + hash-verify of the
s2_blocks_v0 final weights (HF-only). Gate: holdout_v1 + val_v0/val_v1
curves + generation smoke + INT8 fake-quant eval.
