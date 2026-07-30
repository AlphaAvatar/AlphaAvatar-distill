# 2026-07-30 — Unrestricted (P18) generation pilot: the degeneration is the whole line, not the targets

> **Scope note added 2026-07-31.** Every generation here was **no-system**. Under
> the project protocol (proposal §13) that is the **auxiliary/control**
> distribution, not the primary one. The findings below stand as measured — most
> importantly that `s2v1@2700` degenerates 8/8 — but they characterise
> user-only inference. The primary, system-conditioned distribution has never
> been generated, trained or measured.

- **Cost:** **$0.79** (pilot) · session total **$7.93** of the $15.00 authorization
- **Pod:** `2dbp44q6jpp5jk` (vLLM 0.26.0 official image, L40S), deleted; nothing billing
- **Artifacts:** `artifacts/pilot/*.json` — complete raw output, exact token ids,
  stop reason and degeneration evidence for all 48 generations

## 1. Method

8 deterministic prompts (the audited stratified set) × 6 checkpoints, paired by
prompt. Greedy, **concurrency 1**, `max_tokens = 262,144 − prompt_len` per
sample. No token cap, no wall-clock cutoff.

One addition, on maintainer direction: generation is streamed and stopped when
the output **stops producing new information** — a tail block of period *p*
repeated ≥4 times, or a trailing-window distinct-token ratio <6%. This is a
*semantic* stop, recorded as its own class `degeneration` with the period,
repeat count and start index as evidence. It is never merged into natural
termination or into a context-limit hit, and both non-EOS outcomes stay
right-censored. Nothing is judged before 600 tokens — below the teacher's own
median natural completion (727) — so long-but-progressing reasoning is not cut.

Justification: a model in a repetition loop never emits EOS. The first wave ran
**31 minutes without completing one sample**; with detection the same wave took
**35 seconds**. The full 6-wave pilot cost $0.79 instead of an estimated ~$26.

## 2. Result

| checkpoint | natural | degenerated | context-limited | gen tok p50 / max | TTFT | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| step-0 Stage 1 init | 0 | **8** | 0 | 768 / 1024 | 27 ms | 228 |
| `ttb_ctrl_a` (public) | **7** | 1 | 0 | 16 / 768 | 18 ms | 271 |
| `ttb_ctrl_b` (public) | **5** | 3 | 0 | 34 / 768 | 18 ms | 263 |
| `ttb_treat_a` (teacher) | 0 | **8** | 0 | 768 / 1536 | 26 ms | 214 |
| `ttb_treat_b` (teacher) | 0 | **8** | 0 | 768 / 1024 | 24 ms | 229 |
| **`s2v1@2700` (best ckpt)** | **0** | **8** | 0 | 896 / 1024 | 26 ms | 221 |

**Zero context-limit hits anywhere.** The 262,144 window was never the binding
constraint: every non-terminating generation degenerated well before 1,536
tokens. The former 512-token cap was therefore not hiding legitimate long
reasoning — it was hiding a repetition loop.

## 3. The three findings that change the picture

### 3.1 The project's best checkpoint degenerates too

`s2v1@2700` — 2,700 steps, 22.1M tokens, holdout NLL 3.8285, the standing branch
point — produces **0/8 natural terminations** and degenerates on every prompt.
Degeneration is therefore **not caused by teacher-native targets**. It is the
state of this student line at every checkpoint measured, including the best one.
Any conclusion that blamed the teacher targets was comparing two degenerate
models and calling one of them better.

### 3.2 The public arm "terminates" by abandoning the teacher protocol

Its natural terminations are 5–18 tokens that close an empty think block and
emit a stub:

```
</think>\n\nArthur's Magazine<|im_end|>          (6 tokens)
</think>\n\nCarlsberg Laboratory<|im_end|>       (7 tokens)
</think>\n\nin the late 1990s<|im_end|>          (12 tokens)
```

This is the empty-`<think>` protocol substitution, now confirmed at token level.
`format_ok` 0.625 and `terminated` 0.658 were measuring exactly this. It is not
the thinking-only teacher's behaviour, and P17 forbids treating it as progress.

Several "natural" terminations are also incoherent — `ttb_ctrl_b` on
`openmath-000119` emits *"The answer is 124. The answer is 125. The answer is
126."* and then stops. Terminating is not the same as answering.

### 3.3 The teacher arm's failure mode is repetition, not long reasoning

`ttb_treat_a` on `gsm8k-000000`: a **17-token block repeated 15×** starting at
position 513. `ttb_treat_b`: period 51 × 6 from position 718. These are loops,
not extended reasoning — so "it needs a bigger budget" is refuted, and the
correct diagnosis is under-optimization at 137 steps / 3 corpus passes.

## 4. Diagnosed failure modes (§8 of the directive)

Ruled **out** by direct measurement:

* target rendering, delimiter duplication, loss-mask errors, unsupervised final
  answer or stop token, packing loss — all clean, 8/8 (structural audit);
* legitimate long reasoning — refuted, the tails are periodic loops;
* evaluation censoring as the *cause* — zero context-limit hits;
* teacher-native targets as the cause — the best public-trained checkpoint
  degenerates identically.

Remaining live causes, in order of evidential support:

1. **Insufficient optimization / non-convergence** — strongest. 137 steps is 5%
   of the reference budget, and the reference itself is not converged for free
   generation either.
2. **Insufficient data coverage** — 487 prompts, 0.71M real tokens; the arms make
   3.0 (treatment) and 7.6 (control) passes, so repetition is being memorised.
3. **Teacher-forcing vs free-generation exposure bias** — teacher-forced NLL
   improves monotonically (11.76 → 6.23) while free generation degenerates. This
   is the classic signature and is currently uncorrected: there is no on-policy
   term anywhere in the recipe.
4. **Missing representation-level distillation** — the objective is CE + full-vocab
   logit KD only; no hidden-state or sequence-level term.
5. **No system-conditioned data exists.** The *template* requires no system
   message ([contract](2026-07-31_system_prompt_contract.md)) — but the *project
   protocol* mandates one (proposal §13), so 100% of training and all six
   checkpoints sit on the auxiliary distribution. Not a cause of the degeneration
   measured here, which is no-system throughout, but it means the primary
   distribution is entirely unmeasured.

## 5. Effect of the former 4096-token teacher cap

Classified over all 1,504 rollouts from the snapshot's exact token ids:

| class | n | share | accepted |
|---|---:|---:|---:|
| `natural_termination` | 1204 | **80.1%** | 1079 |
| `generation_cap_reached` | 300 | 19.9% | 0 |
| `termination_unknown` | 0 | 0% | — |

Natural-completion lengths: min 87, p10 353, **p25 466**, **p50 727**, p75 1284,
p90 2233, p95 2909, p99 3854, max 4069.

Per slice capped: rag_evidence 0.5%, multihop_qa 1.1%, gsm8k 8.5%,
**openmath 69.7%** — which fully explains openmath's 0.261 accept rate. Every
accepted target came from a natural termination, because `truncated_at_cap` was
an automatic rejection, so **the corpus is unbiased by the cap in content but
biased in composition**: openmath is under-represented (45 of 487) and its hard,
long-reasoning instances are systematically absent.

The teacher's p25 is 466 tokens, so the 512-token evaluation cap sat inside the
teacher's first quartile.

## 6. What this does and does not license

* **Does not** license rejecting teacher-native supervision — the best
  public-trained checkpoint fails the same way.
* **Does not** license a public/no-think curriculum — that arm's apparent
  success is protocol substitution plus stub answers.
* **Does** establish that no checkpoint in this line has yet reached usable free
  generation in teacher mode, and that convergence, coverage and exposure bias
  are the live variables.
