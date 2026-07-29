# 2026-07-29 — Pilot slice analysis: one "failure" is the guard working

- **Agent:** Claude, dev box, **CPU only, $0** — read-only analysis of an
  artifact already on disk (P8)
- **Input:** `artifacts/stage2_v2/pilot/candidates.jsonl`, sha256
  `169cece8d02bbad469aa435161942af1f9316012276de03aa80422cc5b6bd821`
  (relay: `engine_bench_20260729/pilot/`)
- **Command:** `uv run python scripts/analyze_pilot.py`
- **Objective:** decide what to do about the two slices the
  [pilot](2026-07-29_engine_benchmark_gpu.md) flagged — `refusal_uncertainty`
  (accept@n 0.100) and `openmath` (0.300)

## 1. `refusal_uncertainty` is not failing. It is refusing to make things worse.

The summary invited an obvious fix: `REFUSAL_MAX_WORDS` is 60, and 29 of 40
candidates were rejected only for exceeding it, so raise the threshold. The
sensitivity curve makes that look free:

| threshold | candidates passing | accept@n would be |
|---:|---:|---:|
| **60** (current) | 1/40 | **0.100** |
| 80 | 8/40 | 0.600 |
| 100 | 21/40 | 0.900 |
| 150 | 28/40 | 1.000 |

**Do not do it.** The comparison that matters is not the threshold, it is the
target being replaced:

| | public v1 target | teacher candidate |
|---|---:|---:|
| refusal length (words) | 13–16, **median 15** | 66–160, **median 87** |

The teacher's refusals are **~6× longer** than the ones already in the mixture.
Raising the threshold to 100 would swap a 15-word refusal for an 87-word one on
9 of 10 prompts — a direct regression against P10 (short realtime responses,
refusal behavior), bought by relaxing a rule until a metric moved. That is the
failure mode P7 and P14 exist to prevent.

What the pipeline actually did: **9 of 10 refusal prompts kept `v1_public`**.
The fallback is designed to preserve the public target when nothing verifies,
and here it preserved a *better* target. accept@n 0.100 is the guard reporting
success, not a yield problem.

**Recommendation:** leave `REFUSAL_MAX_WORDS` at 60, and **exclude
`refusal_uncertainty` from teacher-target generation entirely**. It is the
second most expensive slice to generate (median 1,628 think tokens per
candidate) and its measured yield is ~0 by design. Dropping it is a pure budget
saving with no data loss.

## 2. Independent finding: the teacher hallucinates on unanswerable questions

The other 10 rejections are `not_a_refusal`, and inspection shows they are **not**
detector misses — the teacher answers questions squad_v2 marks unanswerable:

```
[  1w] Hyrule                                   (x2)
[  3w] GameCube and Wii.
[105w] Based solely on the provided context, the answer is December 2006. …
[ 34w] Based on the provided context, the game takes place "in an alternate timeline…
```

So **10/40 candidates (25%) assert an answer where the gold is "not answerable
from this context"**. This is a teacher capability limit, and it is consistent
with the previously measured grounding ceiling of 0.562 — the lowest of the
teacher's behavior axes.

Two consequences:

* It is a second, independent reason not to source refusal targets from this
  teacher: a quarter of its candidates are exactly the behavior the slice exists
  to train *against*.
* It validates the verifier split. The length rule and the hallucination rule do
  different jobs, and only the length rule looked miscalibrated. Had the
  threshold been relaxed, the hallucination rule would still have caught these —
  which is why the relaxation *looked* safe. It was safe and still wrong.

## 3. `openmath` is genuinely cap-bound, and the yield is unmeasurable from this data

| | value |
|---|---|
| truncated at cap | **28/40** candidates |
| think tokens, candidates that **finished** | median 1,177, **max 2,970** |
| accuracy among finished candidates | **9/12 = 0.750** |
| prompts where **no** candidate finished | **6/10** |

The distribution is effectively bimodal: a candidate either closes its reasoning
well under the cap (max 2,970 of 4,096) or blows through it entirely. Among
those that finish, accuracy is 0.750 — respectable — so the 28 truncations are a
**budget** failure, not a correctness one. Most never reached an answer to be
wrong about.

**But the yield of a higher cap cannot be estimated from this run.** The
truncated candidates are censored at exactly 4,096; nothing in the data says
whether they needed 5k tokens or 50k. Raising the cap is a hypothesis with an
unknown payoff, and the only way to price it is to measure it.

## 4. What changes, and what does not

| slice | accept@n | verdict | action |
|---|---:|---|---|
| `rag_evidence` | 1.000 | works | keep |
| `gsm8k` | 1.000 | works | keep |
| `multihop_qa` | 0.900 | works | keep |
| `openmath` | 0.300 | cap-bound, fixable-in-principle | measure a higher cap before deciding |
| `refusal_uncertainty` | 0.100 | **not a failure** | **drop from teacher generation** |

No code is changed by this analysis. `REFUSAL_MAX_WORDS` stays at 60 and the
verifier is untouched; the recommended change is to the *slice list* a
generation run is given, which is already a command-line argument
(`--slices`) and needs no code.

## 5. Claim strength

* **Measured:** every number above, from a hashed artifact, reproducible with
  the one command at the top.
* **Inferred:** that the 28 openmath truncations would mostly succeed at a
  higher cap. The finished-candidate accuracy of 0.750 makes it plausible; the
  censoring makes it unproven.
* **Not measured:** whether a higher cap changes accuracy (longer reasoning is
  not automatically better reasoning), and whether the hallucination rate on
  unanswerable questions holds beyond n=40 candidates on 10 prompts.

## 6. Next action

The openmath cap question is the only one that needs GPU, and it is small: the
same 10 openmath prompts at a raised cap, comparing how many close their
reasoning and at what accuracy. It is worth bundling with the isolated-venv
engine test rather than buying a pod for it alone — and it should be, because at
the in-stack throughput measured on 2026-07-29 a cap of 16,384 would cost ~4×
per candidate, which is precisely the kind of spend that wants the engine
question settled first.
