# 2026-07-31 — Current-engine benchmark: vLLM 0.26.0 vs SGLang 0.5.12

- **Pre-registration:** [`proposals/2026-07-31_current_engine_benchmark.md`](../../proposals/rollout/2026-07-30_current_engine_benchmark.md)
- **Cost:** **$0.93** against a $3.00 ceiling (+$1 contingency unused)
- **Pods:** `ibht1nb8o0nhyh` (failed, deleted), `lwf2th777hri8c` (Arm A),
  `oy00tooo4ckkv9` (Arm B). All deleted; nothing billing.
- **Artifacts:** `artifacts/bench/current_engines/` — three engine reports plus
  `importance_stats.json`
- **Scope kept:** no corpus built, no refusal data, no corrected-training pilot.

## 1. The correction was right, and one flag was the whole story

Both engines' **current official releases run fine on an L40S** — the same GPU
type the previous session called incompatible. The only change was passing
**`--min-cuda-version 13.0`** at pod creation:

| arm | GPU | driver | CUDA | engine | torch | transformers | arch |
|---|---|---|---|---|---|---|---|
| A | L40S 46 GB | **580.126.09** | **13.0** | **vLLM 0.26.0** | 2.11.0+cu130 | 5.14.1 | sm_89 |
| B | L40S 46 GB | **580.159.03** | **13.0** | **SGLang 0.5.12** | 2.11.0+cu130 | 5.6.0 | sm_89 |

The 2026-07-30 conclusion that vLLM 0.26.0 was unusable, and the resulting
downgrade to 0.11.0, were **artifacts of unconstrained host selection**. Nothing
about the engine or the GPU required either.

Images: `vllm/vllm-openai:v0.26.0`, `lmsysorg/sglang:v0.5.12`, each run as its
own pod via a template that overrides the entrypoint to give shell access
(`--docker-entrypoint /bin/bash,-lc`). The benchmark client ran **on-pod**, so no
network latency enters the throughput numbers.

## 2. Throughput, cost and batch sensitivity

8 slice-balanced prompts (mean 410.6 prompt tokens, sha256 `93b1b825…`), greedy,
$0.99/h.

| arm | cell | tok/s | wall | mean new tok | hit cap | $/1k prompts |
|---|---|---:|---:|---:|---:|---:|
| **vLLM 0.26.0** | cap 4096, conc 2 | 119.4 | 10.83 s | 646.5 | 0/2 | $1.49 |
| **vLLM 0.26.0** | **cap 4096, conc 8** | **247.5** | **57.03 s** | 1764.1 | 2/8 | **$1.96** |
| vLLM 0.26.0 | cap 8192 | — | — | — | — | **not measured** (§5) |
| **SGLang 0.5.12** | cap 4096, conc 2 | 143.5 | 6.81 s | 488.5 | 0/2 | $0.94 |
| **SGLang 0.5.12** | **cap 4096, conc 8** | **241.0** | **56.87 s** | 1713.0 | 2/8 | **$1.95** |
| SGLang 0.5.12 | cap 8192, conc 2 | 143.2 | 6.77 s | 485.0 | 0/2 | $0.93 |
| SGLang 0.5.12 | cap 8192, conc 8 | 162.3 | 110.58 s | 2243.0 | 1/8 | $3.80 |
| SGLang **deterministic** | cap 4096, conc 2 | 64.5 | 14.94 s | 482.0 | 0/2 | $2.05 |
| SGLang **deterministic** | cap 4096, conc 8 | 108.6 | 125.00 s | 1696.1 | 2/8 | $4.30 |

**At the comparable cell the two engines are tied.** 247.5 vs 241.0 tok/s is
2.7% apart, and the cleaner statistic — wall-clock for the identical 8-prompt
workload — is **57.03 s vs 56.87 s, 0.3% apart**. Tokens/s differs slightly only
because the engines emit different completions (mean 1764 vs 1713 tokens), so
wall-clock is the fairer comparison and it says: no difference.

**Batch sensitivity differs, and favours vLLM.** Going from 2 to 8 concurrent
prompts, vLLM scales 119.4 → 247.5 (**2.07×**) while SGLang scales 143.5 → 241.0
(**1.68×**). SGLang is ~20% faster at low concurrency; vLLM overtakes it under
load. For a corpus build, which runs at high concurrency, that favours vLLM
slightly — but not by enough to decide anything on its own.

For scale against the retired in-stack path: HF `model.generate` measured **~44
tok/s and flat in batch size**. Both current engines are **~5.5×** that and both
actually scale.

**The deterministic-mode tax is much larger than advertised.** SGLang's
deterministic inference costs **55% of throughput** (241.0 → 108.6 tok/s; wall
56.87 s → 125.00 s, a 2.2× slowdown), against the ~34% the vendor documentation
cites. Backends it selects: sampling `pytorch`, attention `fa3`.

## 3. Token and log-prob transport: both engines pass

| | vLLM 0.26.0 | SGLang 0.5.12 | SGLang deterministic |
|---|---|---|---|
| token-in / token-out | ✅ | ✅ | ✅ |
| per-token rollout log-probs | ✅ | ✅ | ✅ |
| log-probs aligned 1:1 with tokens | ✅ | ✅ | ✅ |
| masked (unreported) positions | **0** | **0** | **0** |

vLLM via `/v1/completions` with `return_token_ids` + `logprobs`; SGLang via its
**native** `/generate` with `return_logprob`, because its OpenAI surface is
text-oriented. SGLang returns `output_token_logprobs` as (logprob, token_id,
text) triples, and the adapter **verifies those token ids against `output_ids`**
rather than zipping positionally — no misalignment was observed, but the check is
what makes that a fact rather than an assumption.

Adoption criteria 1–3 are therefore **satisfied by both engines**.

## 4. Importance ratios: the mismatch is negligible, and this settles the gate

Rollouts recorded per engine, then scored against the trainer policy in the
project environment (fp32, CPU). Criteria 4–5.

| engine | tokens | median ratio | max ratio | **off-policy rate** (band 2.0) | KL |
|---|---:|---:|---:|---:|---:|
| vLLM 0.26.0 | 288 | 1.000 | 1.072 | **0.000** | 7.1e-05 |
| SGLang 0.5.12 | 288 | 1.000 | 1.035 | **0.000** | 4.6e-05 |
| SGLang deterministic | 288 | 1.000 | 1.083 | **0.000** | 1.15e-04 |

**Not one token in 864 fell outside the [0.5, 2.0] band, on any engine.** The
worst single ratio anywhere was 1.083 — an 8% deviation — and typical KL is
~1e-4.

This is the empirical vindication of retiring exact token agreement as a gate.
The 2026-07-30 session measured **0/8 greedy token agreement** between stacks and
concluded the engine was unusable for on-policy work. Both statements are true at
once, and they are not in tension: greedy decoding takes an **argmax**, so a
logit difference of one part in a thousand flips a token whenever two candidates
are near-tied — while the *distributions* remain almost identical. **Token
divergence is not policy divergence.** Correcting this mismatch is nearly free;
gating on token identity would have discarded a 5.5× speedup to avoid a KL of
0.0001.

*Claim strength.* Measured on 3 sequences × 96 tokens per engine, greedy, at cap
256. The trainer scored in **fp32 on CPU** while rollouts were generated in
**bf16 on GPU**, so part of the residual deviation is dtype, not engine — which
makes these an **upper bound** on the engine-attributable mismatch, and the
conclusion only gets stronger. **Not measured:** ratios under *sampling* (rollouts
here were greedy), longer sequences, and staleness against later checkpoints.

## 5. Failures and what they cost

1. **First pod looped and was deleted (~$0.15).** `vllm/vllm-openai` sets an
   ENTRYPOINT, so `--docker-args` is appended as *arguments to the vLLM server*
   rather than run as a shell; the container crash-looped at
   `uptimeInSeconds: 1`. Fixed with a template using `--docker-entrypoint`.
   **Reusable lesson:** for third-party engine images, override the entrypoint via
   a template; `--docker-args` alone is not an entrypoint override.
2. **vLLM's cap-8192 cells were never measured — my error, not the engine's.**
   I launched the server with `--max-model-len 8192`, leaving no room for a
   ~410-token prompt, so the request was correctly rejected (`maximum context
   length is 8192 … total 8195`). Two restart attempts then failed: the first
   because `pkill` did not kill the old server and the port stayed bound, the
   second because the killed server had not released its 39.7 GB and engine init
   OOMed. I stopped rather than keep spending — a long-context cell is only
   meaningful on **both** arms, and Arm A was already deleted by the time SGLang
   produced its own. **SGLang's cap-8192 numbers stand alone and must not be read
   as a comparison.**
3. **`pkill -f <pattern>` matched the remote shell again.** Third occurrence in
   this project. Kill by PID.

## 6. Verdict against the pre-registered criteria

| # | criterion | vLLM 0.26.0 | SGLang 0.5.12 |
|---|---|---|---|
| 1 | token-in/token-out | ✅ | ✅ |
| 2 | exact token-ID recording | ✅ | ✅ |
| 3 | rollout log-probs | ✅ | ✅ |
| 4 | KL / importance-ratio measured | ✅ 7.1e-05 | ✅ 4.6e-05 |
| 5 | bounded off-policy rate | ✅ 0.000 | ✅ 0.000 |
| 6 | stable corrected training | not run (out of scope) | not run |
| 7 | throughput / cost / reliability | 247.5 tok/s, $1.96/1k | 241.0 tok/s, $1.95/1k |

**Both engines pass criteria 1–5 and are indistinguishable on 7.** Criterion 6 is
the remaining discriminator and was explicitly excluded from this session.

## 7. Recommendation

**Do not adopt yet — but the field is now two viable candidates, not zero.**

* **Neither engine is disqualified**, and neither has a throughput case against
  the other: 0.3% apart on wall-clock at the workload that matters.
* **Tie-breakers as measured:** vLLM scales better under concurrency (2.07× vs
  1.68×), which suits corpus builds; SGLang is faster at low concurrency and is
  the only one offering deterministic inference — at a **55% throughput cost**,
  which is steep enough that it should be a per-run option, never a default.
* **The decision belongs to criterion 6.** With off-policy rates at 0.000 and KL
  ~1e-4, the correction machinery has almost nothing to correct, so a
  corrected-training pilot is likely to be uneventful — which is itself the point
  worth confirming cheaply before committing.
* **Provisional lean: vLLM 0.26.0**, on concurrency scaling and the broader
  ecosystem, with SGLang retained as a live alternative specifically for any
  workload where reproducible rollouts justify the 55% tax.

## 8. Next actions

1. Re-run the **long-context cell on both arms in one session** (cap 8192/16384,
   servers sized with headroom for the prompt). It is the one comparison this
   session owes.
2. **Corrected-training pilot** (criterion 6) — the actual discriminator.
   Separate request; not started here.
3. Re-price the bulk corpus at **~$1.95/1k prompts** at n=1, and note the earlier
   $2.27 figure came from an obsolete build and should be replaced by this one.
