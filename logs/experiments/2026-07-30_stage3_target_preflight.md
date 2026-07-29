# 2026-07-30 — Stage 3 teacher-target preflight: the current data path would shred them

- **Agent:** Claude, dev box, **CPU only, $0**
- **Command:** `uv run python scripts/preflight_stage3_targets.py --out artifacts/stage3_preflight_lengths.json`
- **Input:** the 2026-07-29 pilot corpus, `artifacts/stage2_v2/pilot/targets.jsonl`
  (33 `teacher_verified` + 17 `v1_public` targets), sha256 `2a598649…`
- **Purpose:** the four preflight items the maintainer set before a Stage 3
  teacher-target warm-up can be designed.

## 1. Capability scope — frozen, unchanged

Primary target: the teacher's **reasoning, problem-solving and agent-decision**
capability (AGENTS.md P3/P10.1). In-scope slices for generation and for this
experiment:

**`rag_evidence`, `multihop_qa`, `gsm8k`, `openmath`.**

`refusal_uncertainty` is **evaluation-only** and generates nothing (decision
2026-07-30). `short_realtime` remains provisionally evaluation-only and is not
touched here. Both generation scripts already default to exactly this list, so
the scope is enforced by default rather than by a flag anyone must remember.

## 2. The finding: `concat` @ 1024 cannot carry a teacher-native target

Rendered token lengths, measured through the **real** loader (`encode_sample`,
the same call the trainer makes):

| target kind | n | rendered tokens (min / p50 / p90 / max / mean) | supervised span (p50 / max / mean) |
|---|---:|---|---|
| **teacher_verified** | 33 | 371 / **997** / 3536 / **5193** / 1657 | 586 / 3878 / **1139** |
| v1_public | 17 | 152 / **245** / 950 / 1046 / 398 | 23 / 965 / **218** |

Teacher-native targets are **4.2× longer** overall and carry **5.2× the
supervised span**. Against the current Stage 3 recipe (`concat`, `block_len`
1024):

* **48.5% of teacher targets exceed a whole block**, versus 5.9% of public ones;
* expected split rate is **79.9%** versus 38.7%.

`pack_blocks` documents that "a sample may straddle a block boundary". For short
public targets that is a rounding error. For reasoning traces it is a
correctness problem, and `best_fit_blocks`'s own docstring already says so,
citing Ding et al., *Fewer Truncations Improve Language Modeling* (ICML 2024,
arXiv:2404.10830): the second half of a split trace trains as a sequence whose
premises are not in context, teaching the student to continue reasoning it
cannot see.

**Nothing crashes and nothing is logged.** The loss simply supervises
continuations of invisible premises. This is exactly the silent failure the
preflight was asked to rule out, and it is present in the recipe as it stands.

## 3. Proof, on the real corpus (50 samples, 61,454 tokens, 41,276 supervised)

| packing | block_len | truncated samples | supervised tokens kept | verdict |
|---|---:|---:|---|---|
| `best_fit` | 1024 | 17 / 50 | 18,161 / 41,276 | **loses 56% of supervision** |
| `best_fit` | 4096 | 2 / 50 | 39,782 / 41,276 | loses 3.6% |
| **`best_fit`** | **8192** | **0** | **41,276 / 41,276** | **lossless** |
| `concat` | 1024 | — | 41,263 / 41,276 | no token loss, but **samples straddle** |
| `concat` | 8192 | — | 37,474 / 41,276 | loses 3,802 to the dropped tail |

Two things worth stating plainly:

* **The naive fix is the worst option.** Switching to `best_fit` at the current
  `block_len` 1024 — "the packing that never splits a sample" — would silently
  discard **56% of the supervised tokens**, because a sample longer than a block
  cannot be kept whole by any packing and is truncated instead.
* **Only `best_fit` @ 8192 is lossless**, at 0.938 packing efficiency.

## 4. The bound is constructive, not just empirical

§3 rests on 33 teacher targets. The requirement can be bounded for *any* target
generated from these slices at cap 4096. Worst-case prompt length over 400
samples per in-scope slice:

| slice | p50 | p90 | max |
|---|---:|---:|---:|
| rag_evidence | 266 | 392 | 567 |
| **multihop_qa** | 1425 | 1930 | **2765** |
| gsm8k | 67 | 99 | 149 |
| openmath | 69 | 123 | 739 |

**max prompt 2765 + generation cap 4096 + template overhead ≈ 64 = 6925**, so
`block_len` **8192 is sufficient by construction** with 1,267 tokens of
headroom. If the generation cap is ever raised (it will not be for openmath —
decision 2026-07-30 keeps it at 4096), this bound must be recomputed.

## 5. What this forces in the experiment design

* **Both arms must use identical packing**: `best_fit` @ `block_len` 8192.
  Running the treatment on shredded blocks and the control on intact ones would
  measure fragmentation damage, not teacher targets. The packing choice is a
  shared setting, not an arm variable.
* This **supersedes the packing decision of 2026-07-28 for this experiment
  only**. That decision rejected `best_fit`@2048 because it regressed holdout
  +2.1% on *public* targets, where splitting is a rounding error. It was correct
  for its data and is not evidence about a corpus whose median sample exceeds
  the block.
* **8× the sequence length is not free.** Attention cost is quadratic in
  sequence length, so batch size must fall and a throughput smoke test at 8192
  is a prerequisite before sizing the run. Not yet measured.

## 6. The budget tension the maintainer should settle

The pre-registration asks for an identical **prompt set** *and* an identical
**training-token budget**. Those cannot both hold naively, because teacher
targets are 4.2× longer: at equal tokens the treatment arm makes fewer passes
over the prompt set.

Recommended reading, and the one the pilot will use unless overridden: hold
**total training tokens identical** (steps × batch × block_len), which is what a
compute budget means, and let the number of passes over the prompt set differ.
Report the supervised-token count per arm as a secondary, because it will *not*
be equal — the supervised fraction is **0.687** for teacher targets against
**0.547** for public ones, so at equal total tokens the treatment arm receives
roughly **26% more supervised tokens**. That asymmetry is inherent to the
comparison and is reported rather than engineered away.

## 7. Claim strength

* **Measured:** every number above, through the real loader, on a hashed corpus.
* **Bounded by construction:** §4's 8192 sufficiency, given cap 4096 and these
  slices.
* **Not measured:** training throughput and memory at `block_len` 8192 on the
  0.6B student; whether `best_fit`'s 0.938 efficiency at 8192 costs anything in
  final quality; and the length distribution of a 500–1000 prompt corpus, which
  is assumed to resemble the 33-sample pilot.
