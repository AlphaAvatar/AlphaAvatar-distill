# PROPOSAL (needs user approval) — Stage 2 mixture v1 scale-up

Status: **awaiting user approval** (P12: larger public downloads). Drafted
2026-07-26 from the 2026-07-25 A/B finding that Stage 3 recovery is now
**data-limited**: both A/B arms ran mixture-v0 epochs 3–4, holdout went
flat/regressed, and generation smoke picked up corpus artifacts (gsm8k
`<<…>>`/`####`, refusal-template echo, chat-format slips).

## What is requested

Approval to build `stage2_offline_v1`: scale train data **5.39M → ~24M
tokens (~4.5×)**, all public permissively-licensed sources, no paid APIs,
no teacher-generated data (that is a separate future proposal). One CPU
build session on the dev box; est. HF cache growth ≤ 3 GB (124 GB free),
build ≤ 1 h. Approval covers builder budgets up to 30M train tokens so a
minor top-up does not need a new round-trip.

## Sizing rationale

The next recovery run (attention-unfrozen recipe adopted 2026-07-25) should
train ~3000 steps × 16×1024 ≈ 49M block-tokens. Keeping it at **≤ 2 epochs**
— the regime where s1 improved monotonically, vs. the epoch-3–4 overfit —
needs ≥ 24M fresh-ish train tokens. ~24M also covers a second follow-up run
at ≤ 3 total epochs.

## Composition (train tokens, targets)

| group | v0 | v1 target | fresh sources (license, status) |
| --- | ---: | ---: | --- |
| instruction | 1.26M | 7.0M | **HuggingFaceTB/smol-smoltalk** (Apache-2.0, ungated) + oasst2 headroom; dolly kept, not grown |
| code_math | 0.96M | 4.5M | **nvidia/OpenMathInstruct-2** (CC-BY-4.0) ~2M; **ise-uiuc/Magicoder-OSS-Instruct-75K** (MIT) ~1M; gsm8k remainder (~2.6k rows), all gsm8k format-normalized (below) |
| tool_calling | 0.89M | 3.2M | glaive fresh offset (~110k rows unused); optional: Salesforce/xlam-function-calling-60k (CC-BY-4.0, **auto-gated** — needs a click-through accept; skipped unless approved) |
| multihop_qa | 0.69M | 1.6M | hotpot fresh offset (kept modest — terse extractive targets) |
| rag_evidence | 0.57M | 2.0M | squad_v2 fresh offset (modest; conversational rewrite is the future teacher-gen upgrade) |
| long_context | 0.61M | 2.6M | fineweb-edu fresh stream offsets (holdout exclusion unchanged) |
| short_realtime | 0.23M | 1.6M | **HuggingFaceTB/everyday-conversations-llama3.1-2k** (Apache-2.0) + smol-smoltalk short-dialogue filter |
| refusal_uncertainty | 0.17M | 1.5M | squad_v2 unanswerable fresh; refusal template pool widened 3 → ~12 (anti-echo) |
| **total** | **5.39M** | **~24M** | licenses verified via HF API 2026-07-26 |

Estimated overall trainable fraction rises 0.44 → ~0.52 (conversational
groups grow faster than terse-extractive ones).

## Design changes vs v0 (each maps to an observed failure)

1. **Fresh data, not more epochs** (holdout-flat at epochs 3–4): all new
   samples come from unconsumed row/stream offsets of pinned revisions or
   from new sources; global content dedup as in v0.
2. **gsm8k normalization** (generation smoke emitted `<<…>>` and `####`):
   strip `<<…>>` calculator annotations; rewrite the trailing `#### N` line
   as a natural "The answer is N." — applied to every gsm8k sample in v1.
3. **Chat-format discipline** (one `<|im_start|>` echo, stray `</think>`
   openings): the largest new mass is well-formed multi-turn chat
   (smol-smoltalk — the SmolLM2 SFT mixture), not single-turn extractive QA.
4. **Refusal-template echo risk**: response templates widened from 3 cycled
   strings to ~12 varied ones.

## Split & continuity invariants

- v0 `val` (771) and `calib` (120) stay **frozen** — val_v0 keeps the s1/A-B
  learning curves comparable across mixture versions; calib remains the INT8
  activation-calibration reserve (topped up to ~200 stratified samples
  including new sources, old 120 unchanged).
- New sources contribute train + a fresh `val_v1` slice (same modular rule);
  the trainer logs val_v0 and val_v1 separately.
- v0 val/calib content hashes are excluded from v1 train (leakage guard);
  holdout_v1 exclusion and revision pinning carry over unchanged.

## Risks / notes

- **Synthetic provenance:** smol-smoltalk and everyday-conversations are
  largely Llama-3.1-405B-generated (attribution noted at build);
  OpenMathInstruct-2 is Llama-generated (CC-BY-4.0, "Built with Llama"
  attribution recorded in the manifest); Magicoder is GPT-3.5-generated
  (MIT; flag for release-time license review). None are redistributed —
  jsonl stays gitignored, as in v0.
- Existing share-alike notes (dolly/squad/hotpot CC-BY-SA) unchanged.
- Mixture ratios remain judgment calls; val_v0 continuity + holdout_v1 keep
  regressions measurable.
- Not included (future, separate approvals): teacher-generated corpora
  (grounded conversational rewrites of the QA groups + reasoning traces;
  est. 1–3 h L40S with a vLLM serving path, ~$2–4 + new dependency) and the
  next GPU recovery run itself (own request once v1 exists, est. ~$4–5
  L40S session, consistent with prior per-session approvals).

## On approval, the build session will

1. add `scripts/build_stage2_v1.py` (v0 builders + fresh-offset logic + new
   source builders + normalizations; v0 script untouched);
2. build, validate (dry-run gate script as v0), and commit the v1 manifest;
3. run the Stage 2 gate checklist (AGENTS.md 4.4) and log the experiment;
4. then prepare the next recovery run request (Stage 3, adopted freeze set,
   start-point comparison s2_blocks_v0-final vs s1@660 per the A/B verdict).
