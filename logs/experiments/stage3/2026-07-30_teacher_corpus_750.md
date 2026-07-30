# 2026-07-30 — Teacher-target corpus build (750 prompts) on vLLM 0.26.0

- **Agent:** Claude, dev box + one L40S pod
- **Pre-registration:** [`proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md`](../../proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md) §3
- **Cost:** **$1.37** of the gate's $7.00 ceiling (includes ~$0.10 for one pod
  deleted before use, §5.1)
- **Pods:** `vuf5sxxnil7drh` (no sshd, deleted), `fg3nr3ibef553d` (the build,
  deleted). Nothing billing.
- **Corpus:** `artifacts/stage2_v2/teacher_corpus_750/`, targets sha256
  `18028f0c…`, on the private relay under `stage3_teacher_corpus_20260730/`

## 1. Result

752 prompts, slice-balanced across the in-scope four, nominal n=2 but
**effectively n=1** (§2), cap 4096, sampled at temperature 1.0 / top_p 1.0 /
top_k off. **540 targets accepted (71.8%)**, inside the pre-registered 400–550
expectation.

Both columns below are the same measurement: the two candidates were not
independent draws, so accept@n carries no information beyond accept@1.

| slice | prompts | accept@1 | accept@n *(= accept@1, §2)* |
|---|---:|---:|---:|
| rag_evidence | 188 | 0.979 | 0.979 |
| multihop_qa | 188 | 0.825 | 0.825 |
| gsm8k | 188 | 0.808 | 0.808 |
| **openmath** | 188 | **0.261** | **0.261** |

openmath's low acceptance is consistent with the 2026-07-29 finding that it is
cap-bound; the cap stayed 4096 by decision, and it contributes 45 of 487 training
samples rather than its balanced share.

Wall clock 4,347 s for 752 prompts (5.8 s/prompt at n=2). The server sustained
**~600 tok/s at 16 concurrent requests**, against the 247.5 tok/s the 2026-07-29
benchmark measured at concurrency 8 — the benchmark's own caveat that 8 prompts
do not saturate the engine was correct.

Environment, pinned: `vllm/vllm-openai:v0.26.0`, vLLM 0.26.0, torch 2.11.0+cu130,
transformers 5.14.1, L40S 46 GB, driver 580.159.03, CUDA 13, sm_89.

## 2. The corpus must be recorded as effectively n=1

**697 of 752 candidate pairs (92.7%) are byte-identical, and candidate 2 rescued
0 of 212 first-candidate failures.** accept@n therefore equals accept@1 in every
slice above — by construction, not by coincidence.

**This is a statement about the implementation, not about sampling.** At
temperature 1.0 / top_p 1.0 with independent RNG streams, two draws from a
thinking teacher would not agree byte-for-byte 92.7% of the time; that rate is
the signature of draws that were never independent. It is **not** evidence that
nucleus sampling lacks diversity, and must not be cited as such.

Cause: `generate_candidates` replicated each prompt `n` times inside **one**
engine request under **one** seed. A serving engine seeds per *request*, so the
replicas decoded identically. The replication trick only ever worked for
in-process HF `generate`, which draws every sequence from one rolling RNG; it was
carried over to the server path unexamined.

**Recorded status: this corpus is effectively n=1** — 540 targets each accepted
on a single draw, at 2× the token cost. Every accept rate above is an accept@1.
It is **not** regenerated for now (maintainer direction 2026-07-30): 540 verified
single-draw targets are sufficient for the corrected baseline, and GPU time is
not spent on diversity that baseline does not depend on.

Fixed in code: each candidate index now draws under its own seed
(`seed + index * 1_000_003`), with a regression test asserting distinct seeds per
call. **The fix is untested against real acceptance** — no regenerated corpus
exists to confirm that independent draws raise accept@n. The measured
non-independence is recorded in the corpus manifest so nobody later reads
accept@n as evidence that n candidates helped.

**This matters more for Stage 4/5 than for here.** GRPO-style on-policy methods
depend on genuine sample diversity per prompt; had this reached rollout
collection unnoticed, every group would have been near-degenerate.

## 3. Both 2×2 arms build losslessly

540 accepted prompts → **487 train / 53 val**, identical prompt sets and identical
split membership in both arms (`sha256(id)`-derived, seed-free).

| | control (public) | treatment (teacher-native) |
|---|---|---|
| rendered tokens p50 / max | 213 / 2,384 | **1,149 / 5,457** |
| supervised tokens | **27,526** | **519,478** |
| supervised fraction | 0.0929 | 0.6591 |
| train blocks @ best_fit 8192 | 36 | 91 |
| packing efficiency | 0.895 | 0.959 |
| truncated samples | **0** | **0** |

**The pre-registration's supervised-token estimate was wrong by an order of
magnitude.** It predicted the treatment arm would carry ~26% more supervised
tokens (fraction 0.687 vs 0.547). On the actual shared accepted subset it carries
**18.9×** more. The 0.547 figure came from the 17 public-*fallback* rows of the
July pilot — a different, gsm8k-heavy population. The accepted subset is
dominated by rag_evidence and multihop_qa, where the prompt is long and the
public answer is a few tokens.

This is not a fixable confound: a public target cannot be given a reasoning
trace, so the extra supervision *is* the intervention. It does mean the
experiment can support "teacher-native targets improve protocol competence" but
**not** "because they are teacher-native rather than because they supervise more
tokens" — the two are not separable by this design.

## 4. Reproducibility

Recorded in the corpus manifest: teacher id + revision, tokenizer sha256 and
transformers version, engine name/version/image, full decoding parameters,
prompt-selection mode (`stride`, 188/slice) and the sha256 of both output files.
Exact sampled token ids and the rollout policy's per-token log-probs are in a
hashed snapshot (1,504 rollouts, 2,460,814 tokens, sha256 `0e5b20dd…`).

The manifest is **reconstructed**, and says so — see §5.2.

## 5. Failures and what they cost

### 5.1 The engine image has no sshd (~$0.10)

`vllm/vllm-openai` ships no sshd, and RunPod's TCP 22 mapping only works if the
container runs one. The first pod was healthy and unreachable. Fixed with a
template whose start command installs `openssh-server`, writes `$PUBLIC_KEY` to
`authorized_keys` and execs `sshd -D`. **Reusable:** for third-party engine
images, expect to provide both the entrypoint override *and* sshd.

### 5.2 `code_state()` destroyed a manifest after 72 minutes of paid generation

The run generated all 752 prompts, wrote candidates, targets and the hashed
snapshot — then died on its last line, because `code_state()` shells out to
`git`, which the vLLM image does not ship (P8.1: do not assume a tool exists).

Nothing was lost but the manifest itself: the session script still hashed and
uploaded every artifact. It was reconstructed by
`scripts/rollout/rebuild_corpus_manifest.py`, which recomputes the per-slice
statistics from `candidates.jsonl` and takes the commit and hardware from the
caller, and declares in the manifest that it was rebuilt, by what, and why.

`code_state()` now degrades instead of raising: it records whether the commit
came from git, from `AADISTILL_CODE_COMMIT`, or from nowhere, and **never guesses
one**. The general lesson is worth keeping: *reproducibility metadata is
collected at the end of expensive work, so it must never be able to fail in a way
that destroys the record of the work it describes.*

## 6. Claim strength

* **Measured:** every number in §1 and §3, on the real corpus through the real
  loader; the diversity figures in §2 over all 752 rows.
* **Not measured:** whether distinct per-candidate seeds would raise accept@n
  (the fix is tested for seed distinctness, not yet for its effect on
  acceptance); whether a higher openmath cap with better prompt selection would
  help (rejected as a blanket setting on 2026-07-30).
* **Superseded:** the pre-registration's "~26% more supervised tokens" (§3).
