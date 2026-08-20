# Phase-A attempt 11 — 2026-08-20, $3.2101, **Stage 1 PASSED**, Stage 2 failed closed

**The beam search ran to completion for the first time.** Stage 0 attested, Stage
1 searched 43 states over 4 levels and selected 5 leaves, and Stage 2 then
refused to train the first probe on a fail-closed guard. Pod deleted with
provider confirmation. **Authorization CONSUMED.**

| | |
| --- | --- |
| authorization | `autoinit.phase_a.2026-08-20T0940Z`, sha256 `8f7d9979c51dae4589217d4ee7e0ed34b1c92e2413389211eb7d7c40ced377ed` |
| grant | [`../autoinit_phase_a_attempt11_grant.json`](../autoinit_phase_a_attempt11_grant.json), sha256 `6a6251f7a534a1dd…` |
| authorized base | `6702fc74e6e6581ce10b140fd05f5ac929dcdcdb` |
| session commit | `ac3c895f6d80291a1007ddada84f2bf7cde7abe3` (authorization-only, one path) |
| harness digest | `91fd94f4683e3be8f98f7d2cc88db8108302291cb3d9a0cc56d84a0357f6373b` (16 files) |
| bundle | `aad_autoinit_ac3c895f.bundle`, sha256 `66ad8d1df656b94e26bc4dd22b391ce60a21fc19eeaeb0c11ed8b43aaa7791c7` |
| pod | `y1wmjhemi50jkw`, 1×L40S @ $0.99/h, first draw |
| lifetime | 10:03:46 → 13:18:19 UTC, **194.55 min** |
| cost | **$3.2101** of a $23.0484 ceiling |
| terminal | `PHASE_A_FAILED` at stage 2, `passed: false`, no launcher error |

## Stage by stage

| stage | result | elapsed |
| --- | --- | ---: |
| setup | `SETUP_RC=0`, first draw | 6.0 min |
| **0 — attestation and binding** | **PASSED** | 2.0 min |
| **1 — beam search** | **PASSED**, `SEARCH_DONE:5` | **180.3 min** |
| **2 — rung-1 probes** | **FAILED** on the first probe | 0.1 min |
| 3, 4, 5 | not reached | — |

## Stage 1: the first completed AutoInit search

```
43 states · 4 levels · 7 complete leaves · 18 pruned · 0 resumed
beam.delayed_prune, width 6, no quality pruning at level 0
policy beam.pareto_multi_objective v2, epsilon-Pareto over teacher fidelity
calibration calib.domain_balanced@v1 · seed 20260815
config_hash 567d32789ba6dcef…
```

Five leaves selected, every one **596,049,920 parameters**, each a distinct
four-operator composition:

| # | state | path |
| --- | --- | --- |
| 0 | `cca699c93f34` | FFN → DEPTH → RESIDUAL_WIDTH → ATTENTION |
| 1 | `85bde4ded2c3` | DEPTH(none) → FFN → RESIDUAL_WIDTH → ATTENTION |
| 2 | `158b96cf651f` | DEPTH → FFN → RESIDUAL_WIDTH → ATTENTION |
| 3 | `4e429f7ed722` | ATTENTION → DEPTH → RESIDUAL_WIDTH → FFN |
| 4 | `281a02c3ac18` | FFN → DEPTH(none) → RESIDUAL_WIDTH → ATTENTION |

**The existing composite initialization did not make the cut.**
`COMPOSITE_STAGE1(calib.domain_balanced@v1)` — state `0a61c14f8c2f`, the
single-shot recipe this search was built to test against — lands on **front 4**
and is `selected: False`. Four search-discovered orderings dominate it on the
Pareto front. That is the first evidence this project has that operator
*ordering* carries signal, and it is exactly the question Phase A exists to ask.

The canonical control was injected and verified against its frozen hash:
`control-qwen3_0p6b_init_v0`, `frozen_sha256_verified: true`.

**The leaf weights are gone.** `finalists_to_fetch` pulls finalists after Stage 5
selection, which never ran, so `checkpoints_fetched` is empty and the five
checkpoints died with the pod. The *record* survives — `search_result.json`,
`search_states_reduced.jsonl` (77 states, all traces) and the out-of-tree full
journal — but regenerating the weights costs another ~180 min of search.

## The reference cache fell back, all four times

```
reference cache 16.9 GiB does not fit in 66% of 20.3 GiB
  -> recomputing the reference per candidate: identical numbers, ~2x forward passes
```

Measurement Attempt 3 saw **36.42 GiB** free running standalone. Inside the
search only **20.3 GiB** is free, because the beam holds the parent teacher and
candidate state, so the cache never fits. Four `depth.causal_kl_greedy_v1`
invocations ran, each a full 260 evaluations:

| invocation | minutes | eval/min |
| ---: | ---: | ---: |
| 1 | 37.3 | 6.96 |
| 2 | 27.0 | 9.63 |
| 3 | 33.7 | 7.72 |
| 4 | 24.1 | 10.79 |
| **total** | **122.1** | — |

That is **68% of the whole search**. Against the standalone measurement of 21.53
min / 12.07 eval/min, the in-situ fallback path costs **1.6–1.7×** — consistent
with ~2× forward passes where forwards dominate but are not the whole
per-evaluation cost. Later invocations are faster because the candidates are
progressively more compressed.

**The measurement was not wrong; it measured a different memory regime.** A
standalone process has the device to itself. This is worth carrying forward: the
cached path may never be reachable inside the search at this teacher size.

## The deadline fix was load-bearing by 17 seconds

```
Stage 1 elapsed       180.283 min
old deadline (base)   180.0000 min      → exceeded by 0.283 min = 17 s
new deadline          363.9841 min      → 183.7 min unused
```

The search finished **17 seconds past the bound it would have been killed at
yesterday.** The `$0` alignment committed in `16e382f` — deriving the Stage-1
deadline from the priced envelope instead of the base allowance — is the only
reason this search produced a result at all. Its own justification predicted the
reference-cache fallback as the case that would need the reserve; the fallback
engaged on the very next run, four times out of four.

That is a margin, not a comfort. The next search that is 0.2% slower still fits
easily under 363.98 min, but nothing about 180.28 was designed — it landed there.

## Stage 2: a fail-closed guard, working

```
File "/workspace/aad/scripts/training/train_stage3.py", line 173, in main
    raise ValueError("teacher and student tokenizers differ; refusing to train")
```

**Diagnosed at $0 and reproduced exactly.** The chain:

1. `Qwen3Adapter.save()` calls `model.save_pretrained(path)`, which writes
   weights, `config.json` and `generation_config.json` — and **no tokenizer
   files**. Correct for the search, which never needs a tokenizer: it consumes
   pre-tokenized calibration items.
2. Stage 2 points a probe's `student_path` at that leaf directory.
3. `train_stage3.py` does `AutoTokenizer.from_pretrained(student_path)`.

Step 3 **does not raise**. On a directory holding only `config.json`, it returns
a tokenizer with a **vocabulary of one token**:

```
AutoTokenizer.from_pretrained(leaf) SUCCEEDED
  vocab size: 1
  hash      : 42d8c56b2d86cf7b…      teacher: 7781771acc3798ee…
```

The teacher-vs-student equality guard caught it. Without that guard the probe
would have trained against a one-token vocabulary and produced numbers.

The canonical init's own tokenizer is **not** the problem — it hashes
`7781771acc3798ee…`, identical to the teacher's, 151,669 tokens, zero
differences. What is missing is any step that carries those files into a
*searched* leaf.

**This is the `control_sb` class again**, recorded on 2026-08-16: identity gates
all pass while the checkpoint cannot be used, because the gates check what the
producer needs rather than what the consumer needs. Every leaf here passed
`materialize → reload → hash → validate` and reached `MEASURED`; none of those
asks whether a downstream trainer can load a tokenizer.

**No fix is applied here.** The failure is recorded and the run stopped for
review, per the grant.

## What worked

Everything in the chain, and every gate that was supposed to fire, fired. Setup
passed on the first draw. Stage 0 attested. The cgroup correction held (13
threads from `cgroup.v2` against 128 visible). The search's own progress logging
— added in the runtime repair — made the fallback and the per-invocation rates
visible while the run was live, which is how the 1.7× was known at 60 minutes
rather than at the postmortem. `manifest rc=0`, 13 files across 7 classes, all
`final_required` present and quiescent. The teardown gate allowed on
`training_complete` failing, deleted the pod, and the provider confirmed it gone
at 194.5 min against a 1397-minute bound.

## Disposition

Recorded as **consumed, Stage-1 PASSED, Stage-2 fail-closed stop**. The grant and
the authorization are spent and must not be reused. Cumulative spend $206.4741 →
**$209.6842** of $231.00, leaving **$21.3158** — which is **less than the
$23.0484 per-launch ceiling**, so a further full-ceiling attempt does not fit
without another cap decision.

**No Attempt 12 is prepared, granted, funded or implied.** The maintainer's cap
approval says in terms that it "does not authorize any subsequent attempt".
