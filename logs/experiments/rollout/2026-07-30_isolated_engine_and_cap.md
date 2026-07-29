# 2026-07-30 — Isolated-venv vLLM + openmath cap: both hypotheses answered

> **Conclusion corrected 2026-07-30 (maintainer).** Every measurement in this
> log stands. **§2's recommendation does not** and has been rewritten in place.
>
> The error was in the pre-registered gate, not the arithmetic: R1 made exact
> greedy token agreement with the in-stack path a prerequisite, and §2 then
> assigned HF to Stage 4/5 *permanently* on the strength of one measured
> alternative. Token equality is not a prerequisite for on-policy training —
> production RL systems pair an inference-optimized rollout engine with a
> separate trainer and correct the mismatch explicitly — and one comparison
> cannot select a standing backend. **vLLM 0.11.0 is the first measured engine,
> not the chosen one.** See the 2026-07-30 decision record "Rollout engine
> selection is reopened". §4 (openmath cap) is unaffected.

- **Agent:** Claude, pods `wdpyamp2pp5v8t` (dead on arrival) and `w86xu7t78y571h`
  (1× L40S, $0.99/h)
- **Pre-registration:** [`proposals/2026-07-30_isolated_engine_and_cap.md`](../proposals/2026-07-30_isolated_engine_and_cap.md)
- **Cost:** **$1.85** against a $2.50 ceiling. No corpus built, no refusal data
  generated, no alignment-oriented slice added.
- **Artifacts:** relay `isolated_engine_20260730/`; `report.json` sha256
  `209f50db…`, `openmath_cap16384/candidates.jsonl` `e875f5e8…`

## 1. Headline

**An isolated-venv vLLM server is 5.3× faster and 5.4× cheaper than the
in-stack path — and agrees with it on 0 of 8 prompts.** Both pre-registered
hypotheses fired, in the combination §8 of the proposal called "the interesting
case".

| engine | tok/s | $/1k prompts | mean new tok | batch-invariant | agreement vs in-stack |
|---|---:|---:|---:|---:|---:|
| `hf` (in-stack) | 40.4 | $12.33 | 1811 | 7/8 identical | — (reference) |
| `vllm_server` (isolated venv) | **213.9** | **$2.27** | 1764 | 4/8 identical | **0/8 = 0.000** |

* **R2 (≥3× throughput): fires.** 5.29×.
* **R1 (≥0.90 agreement): fails.** 0.000. `decision.json` therefore records
  `winner: hf` with `vllm_server` rejected under R1 — the mechanical rule did
  its job without an agent interpreting the numbers.

Divergence is not immediate garbage: first-divergence tokens are
`[438, 260, 34, 238, 914, 98, 0, 269]`, median **260**. The two stacks track each
other for a few hundred tokens of reasoning and then take different trajectories.
That is the expected consequence of different kernels and reduction orders, now
measured across stacks rather than assumed.

Note also that **vLLM is itself less batch-invariant than the in-stack path**
(4/8 vs 7/8 identical). Neither is invariant; the faster one is worse.

## 2. What this means (rewritten 2026-07-30 after the maintainer correction)

**vLLM 0.11.0 is the first rollout engine this project has measured. It is not
the engine choice.** It stays a live candidate on its 5.29× throughput and 5.4×
cost advantage, and it is not adopted until compared against at least one other
serious candidate — SGLang deterministic mode above all, which was never reached
because 2026-07-29 mis-attributed its failure to a Python conflict when the real
constraint is the host's CUDA-12.8 driver.

**The 0/8 agreement figure does not disqualify anything.** The original §2 read
it as a gate and concluded HF should own Stage 4/5 permanently. That reasoning
is retired for two independent reasons:

* **Token equality is not a prerequisite for on-policy training.** Production RL
  systems routinely run an inference-optimized rollout engine separate from the
  trainer, asynchronously, and handle the resulting mismatch explicitly — with
  rollout and trainer log probabilities, policy/checkpoint versioning, bounded
  staleness, token- or sequence-level importance sampling, clipping or rejection
  of excessively off-policy samples, and hashed snapshots of the exact rollout
  tokens. A different engine creates a *measurable* mismatch, not an impossible
  one. The real question is whether that mismatch can be quantified and
  corrected inside a pre-registered stability bound.
* **The gate is incoherent with this project's own measurements anyway.**
  Decoding is not batch-invariant *within* a single stack — 7/8 for in-stack,
  4/8 for vLLM in this very session. A criterion demanding cross-stack token
  identity is asking for a property the trainer does not have against itself.

What the 0/8 figure *is* worth: a mismatch signal. Median first divergence at
token **260** says the stacks share a few hundred tokens of reasoning and then
separate, which is a useful prior for how large a correction term will need to
be. It is a diagnostic, and it gates nothing.

**HF `model.generate` is retired as the planned production rollout path**
(decision 2026-07-30). It remains a reference implementation, a debugging path,
a small-scale correctness oracle, and a fallback when no efficient engine is
available. The production direction is an efficient, isolated rollout service
reusable across Stages 3, 4 and 5 — the same service, not a per-stage split.

**The adoption criteria are replaced.** Instead of exact agreement: correct
token-in/token-out transport; exact recording of rollout token IDs; rollout
log-prob availability; measured KL / importance-ratio distribution against the
trainer policy; bounded off-policy rate and staleness; stable corrected training
in a small pilot; and throughput, cost and operational reliability. **This
project satisfies the first two today and none of the rest** — `aadistill.engines`
has no log-prob path at all, which is now the gating piece of work.

**No bulk corpus is built on this result.** The benchmark proposal is revised
first (see `proposals/2026-07-30_rollout_engine_comparison.md`).

## 3. The isolated venv works, but only when pinned — three failures deep

2026-07-29 concluded "vLLM is incompatible with this project's stack". That was
**version-specific, not fundamental**, and this session found the real
constraint stack:

1. **Latest vLLM (0.26.0) is blocked by the host driver, not by Python.**
   `RuntimeError: The NVIDIA driver on your system is too old (found version
   12080)`. Driver 570.124.06 is CUDA 12.8-era. **Venv isolation cannot fix
   this** — it is a property of the machine. This supersedes the earlier
   `libcudart.so.13` reading, which was the same wall seen from inside Python.
2. **vLLM 0.11.0 (torch 2.8.0+cu128) imports and runs on this driver** — but
   `uv` resolved **transformers 5.14.1** into its venv, and vLLM 0.11.0 calls
   `all_special_tokens_extended`, removed in v5. Pinning `transformers==4.57.1`
   *inside the isolated venv* fixed it, and crucially **without touching the
   project env**, which is the entire point of the isolation.
3. **`ninja` was installed but not on the spawned subprocess's PATH**, so the
   engine core died with `FileNotFoundError: 'ninja'`. Launching with
   `/opt/vllm-venv/bin` on PATH fixed it.

So the honest integration cost is: **a pinned vLLM version, a pinned
transformers inside its venv, a PATH fix, and a separate process to supervise** —
about 25 minutes of pod time to discover, and now written down. That is real but
bounded, and it buys 5.4×.

## 4. openmath cap: R3 does not fire — longer reasoning is worse reasoning here

10 openmath prompts at cap 16,384 (n=2, batch 2, in-stack for comparability),
against the pilot's cap-4096 control (n=4).

| | cap 4096 (n=4) | cap 16384 (n=2) |
|---|---:|---:|
| candidates closing their reasoning | 12/40 = **0.300** | 17/20 = **0.850** |
| accuracy **among closing candidates** | 9/12 = **0.750** | 5/17 = **0.294** |
| accepted per candidate | 9/40 = 0.225 | 5/20 = 0.250 |
| think tokens, median | 4096 (censored) | **6487** |
| **tokens per accepted target** | **14,931** | **29,707** |

* **R3(a) — closes ≥2× as many candidates: fires.** 0.300 → 0.850, a 2.83×
  improvement. The cap genuinely was binding, and the censored median was real:
  the true median trace is **6,487 tokens**, well beyond 4,096.
* **R3(b) — must not reduce accuracy among closing candidates: fails badly.**
  0.750 → 0.294.

**R3 therefore does not fire and the cap stays at 4,096.**

The interpretation is the pre-registered risk, now measured: *the candidates that
need more than 4,096 tokens are the ones the teacher gets wrong.* Long reasoning
on this slice is a symptom of floundering, not of care. Raising the cap converts
`truncated_at_cap` rejections into `answer_mismatch` rejections (3 → 10) rather
than into accepted targets.

The economics say the same thing more bluntly: **cost per accepted target
doubles** (14,931 → 29,707 tokens) to buy a +2.5 percentage-point rise in
per-candidate accept rate. That is a worse deal, not a better one.

## 5. Deviations from the pre-registration

* **Cap arm scoped `n=4` → `n=2`, batch 2, before the run** and recorded in the
  proposal at the time. The original was arithmetically impossible (~4 h inside a
  0.7 h budget) and 4 × 16,384-token sequences need ~48 GB of KV cache on a 48 GB
  card. Completion rate and accuracy are per-candidate rates, so the halved `n`
  widens error bars without changing what is measured.
* **The benchmark ran with `--min-speedup` at its default 1.5, not the 3.0 in
  the proposal.** This did not change the outcome — `vllm_server` was already
  ineligible under R1, so the speedup threshold was never the binding
  constraint — but the config did not match the plan and that is recorded rather
  than glossed. The 3.0 judgement is applied in §2 by hand.
* **First pod never started.** `wdpyamp2pp5v8t` sat at `runtime: null` for 25
  minutes with correct ports and `desiredStatus: RUNNING`; deleted and recreated
  with a 150 GB container disk instead of 200 GB, and the replacement was ready
  in 2 minutes. **$0.42 wasted.** Unclear whether the disk size or the host was
  at fault; 150 GB is the size every prior successful session used.

## 6. Process failure worth recording (P11)

`pkill -f <pattern>` **matches the remote shell's own command line** when the
pattern appears in the SSH command. This killed the launcher twice in this
session (once for `generate_teacher_answers`, once for `vllm serve`) and cost
~10 minutes of pod time before it was diagnosed. It was already recorded on
2026-07-29 and repeated anyway. **Rule: never combine a `pkill` and a launch in
one SSH invocation; kill by PID, or use separate calls.**

## 7. Claim strength

* **Measured:** every number in §1 and §4, from hashed artifacts, on one L40S,
  one session, `n=8` prompts (engine) and `n=10` prompts (cap).
* **Not measured:** whether the 5.3× holds at larger batch/concurrency (vLLM's
  advantage should *grow*, since continuous batching is what it is for); whether
  vLLM's accepted-target quality differs from in-stack's after verification —
  only token agreement was compared, not accept rates per engine.
* **Single run per arm.** The behaviour noise floor (0.1290) does not apply here
  — these are throughput and token-identity measurements, not behaviour scores —
  but the accuracy figures in §4 rest on 12 and 17 candidates respectively and
  have wide error bars. The *direction* (0.750 → 0.294) is far larger than that
  uncertainty; the exact values are not.

## 8. Next actions (revised 2026-07-30)

1. **Revise the engine benchmark into a rollout-engine comparison** and define
   the importance-sampling/correction experiment:
   `proposals/2026-07-30_rollout_engine_comparison.md`. At minimum vLLM 0.11.0
   versus SGLang deterministic mode, on a driver that supports both.
2. **Build the log-prob path.** `aadistill.engines` returns tokens only; rollout
   log probabilities, policy/checkpoint version stamping and a hashed
   rollout-snapshot format are prerequisites for any Stage 4/5 pilot and for
   measuring the KL / importance-ratio distribution.
3. **Do not build the bulk corpus yet** (maintainer, 2026-07-30). The $2.27/1k
   figure is real but the engine is not chosen, and a corpus is not worth
   building twice.
4. **Leave the openmath cap at 4,096.** If openmath yield matters later, the
   lever is prompt selection or a different teacher — not more tokens.
