# 2026-07-30 — Isolated-venv engine test + openmath cap measurement (pre-registration)

**Status:** pre-registered, **not run, no spend committed.** Written before the
session so the rules cannot be chosen after seeing the numbers (P6). Revised
2026-07-30 for the capability-scope / alignment-tax principle (P3, P10, P10.1).

**Capability scope of this session.** Both arms serve the recipe's **primary
capability target** — reasoning and problem-solving — and nothing else. The
engine arm is infrastructure for transferring it; the cap arm is about
`openmath`, a primary-transfer slice. **No refusal data is generated, and no
alignment-oriented slice is added.** `refusal_uncertainty` is evaluation-only
for this recipe (decision 2026-07-30), so it is absent from the corpus by scope,
not by a length filter.

Successor to [`2026-07-29_engine_benchmark.md`](2026-07-29_engine_benchmark.md),
which measured the in-stack path and found both serving engines unusable
*in-process*. This proposal tests the one path that remains.

## 1. Why this is the next spend

The 2026-07-29 session produced a bind:

* in-stack decoding runs at **~44 tok/s and does not improve with batch size**
  (37.5 / 43.9 / 39.3 at batch 2 / 4 / 8), which prices a 1,000-prompt × n=4
  corpus at **~40 GPU-hours (~$40)**;
* both serving engines are **incompatible with the pinned stack in-process** —
  vLLM 0.26 needs CUDA 13 against cu128, SGLang downgrades transformers by a
  major version.

So the corpus is unaffordable on the only engine that works. The untested escape
is to stop trying to share a process: run vLLM in **its own venv with its own
torch**, and talk to it over HTTP. That is how it is actually deployed, and it
makes the engine's dependency tree stop being the trainer's problem.

## 2. Hypotheses

* **H1. An isolated vLLM server is ≥3× faster than in-stack at this job shape.**
  If it is not, the teacher-target direction needs re-scoping rather than a
  faster engine, because nothing else on the table changes the arithmetic.
* **H2. Its greedy tokens will not exactly match the in-stack reference.**
  Pre-registered in the previous proposal and still unmeasured — no engine ran.
  This is the property Stage 4/5 needs and no vendor claims across stacks.
* **H3. Raising the openmath cap recovers most of the 28/40 truncated
  candidates without lowering accuracy.** Grounded in the finished candidates
  being 0.750 accurate at ≤2,970 think tokens, but the truncated ones are
  censored at exactly 4,096, so the payoff is genuinely unknown
  ([analysis](../experiments/2026-07-29_pilot_slice_analysis.md)).

## 3. Fixed budget

| item | value |
|---|---|
| hardware | 1× L40S (48 GB), pod-local disk |
| price basis | **$0.99/h** (measured 2026-07-29, not the $0.86 list estimate) |
| setup incl. isolated venv + vLLM install | ≤ 1.0 h |
| engine benchmark (10 prompts, cap 4096) | ≤ 0.5 h |
| openmath cap measurement | ≤ 0.7 h |
| **ceiling** | **≤ 2.5 h ≈ $2.50**, `--terminate-after` +5 h |

No corpus is built this session. Sizing a bulk build is the *output* of H1, not
an activity to run alongside it — the previous session's lesson is that chaining
a long job behind an unmeasured one risks spending the budget before the
measurement lands.

## 4. Method

**Engine arm.** Create `/opt/vllm-venv` with vLLM and whatever torch it wants —
explicitly *not* the project venv. Start `vllm serve Qwen/Qwen3-4B-Thinking-2507`
with the pinned revision. Drive it with `VLLMServerEngine`
(`src/aadistill/engines.py`), which sends **token ids** and requires
`return_token_ids`; a server that cannot return token ids fails the arm loudly
rather than round-tripping through text and reintroducing retokenization drift.

Same job shape as 2026-07-29 so the numbers are comparable: 10 slice-balanced
prompts, mean ~374 prompt tokens, cap 4096, greedy, `$/1k prompts` at $0.99/h.
Measure throughput, batch invariance, and **agreement against the in-stack
completions already recorded** in `engine_bench_20260729/bench/report.json`.

**Cap arm.** The same 10 openmath prompts, n=4, at caps **4096 (control,
already measured) and 16384**. Report, per cap: fraction closing their reasoning,
accuracy among those that close, and think-token distribution.

Raising a cap is the opposite of a terseness constraint and is consistent with
P10: it lets the teacher reason to its natural length instead of truncating a
higher-quality answer to fit a budget. The extra tokens are logged as a workload
characteristic (they set the corpus's cost), not treated as a quality signal in
either direction.

**Slices used anywhere in this session:** `rag_evidence`, `multihop_qa`,
`gsm8k`, `openmath`. The benchmark arm's prompt sampler is slice-balanced over
that list. `refusal_uncertainty` is excluded by capability scope, which also
removes the most expensive slice per accepted sample from the job — a budget
saving that falls out of the scope decision rather than motivating it.

## 5. Pre-registered rules

* **R1 — agreement is a gate.** The server arm is adoptable only if greedy
  agreement with the in-stack reference is **≥0.90**. Below that it is a
  different policy from the trainer's.
* **R2 — speed must clear the bind.** Adopt only at **≥3× in-stack throughput**.
  Below that the corpus is still unaffordable and a second stack has been bought
  for nothing (P1). This is deliberately stricter than the 1.5× of the previous
  proposal, because the process boundary is real integration surface and a
  marginal win no longer pays for it.
* **R3 — the cap is raised only if it pays.** Raise openmath's cap only if the
  16,384 arm both (a) closes ≥2× as many candidates and (b) does **not** reduce
  accuracy among closing candidates. Longer reasoning is not automatically
  better reasoning.
* **R4 — a failed arm is reported, not retried.** Each arm is independent; each
  sweep point within an arm is independently guarded (fixed 2026-07-29 after one
  OOM discarded a 45-minute arm).

## 6. Abort rules

* **A1.** If the isolated venv or `vllm serve` fails to start within the setup
  budget, stop and report. That outcome answers the question — it means vLLM is
  not cheaply deployable here — and burning an hour on dependency archaeology
  does not improve the answer.
* **A2.** If the server cannot return token ids, the engine arm ends
  immediately (see §4); the cap arm still runs, since it uses the in-stack path.
* **A3.** Hard stop at the 2.5 h ceiling regardless of arm state.

## 7. Verified on CPU before any spend

- `VLLMServerEngine` exists and is tested against a stub server (**198 tests
  pass**): it sends token ids rather than text, requires `return_token_ids`,
  **orders choices by the server's `index` rather than arrival** — the OpenAI
  schema carries an index because order is not contractual, and mispairing
  completions with prompts is silently wrong — and rejects choice-count
  mismatches and out-of-range indices.
- It adds **no dependency**: the client is `urllib` from the standard library.
- Unverified, and only testable on the pod: that a real vLLM build accepts
  token-id prompts on `/v1/completions`, honours `return_token_ids`, and starts
  at all in an isolated venv.

## 8. What each outcome means

| H1 | H2 | consequence |
|---|---|---|
| ≥3× | ≥0.90 agreement | adopt the server for corpus builds; re-price the bulk build and proceed |
| ≥3× | <0.90 | **the interesting case.** Fast but off-policy: usable for Stage 3 offline targets, *not* for Stage 4/5 rollouts. Record the split explicitly. |
| <3× | any | the teacher-target direction is re-scoped, not re-engined. Options then: fewer prompts, fewer candidates, or dropping the slices with the worst cost-per-accepted-sample. |

The third row is a real possibility and is why this is worth $2.50: it would
redirect a plan that currently assumes a corpus is affordable.
