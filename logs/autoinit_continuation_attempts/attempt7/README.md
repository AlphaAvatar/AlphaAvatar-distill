# Continuation attempt 7 — INCOMPLETE, but the driver RAN. $0.4500, 27.0 min.

First attempt to reach a driver stage. Authorization `c398850b…` (**consumed**),
checkout `05157d29…`, bundle `aad_autoinit_05157d29.bundle` (`778eb055…`),
harness `a00dce7f…`, plan `79da6d7a…`, transport relay. Pod deleted, provider
confirms `TERMINATED`, account has no pods.

## Stages 0, 1 and 2 PASSED

| stage | wall | cumulative | verdict |
| --- | --- | --- | --- |
| setup (whole script) | 6.4 min | $0.15 | complete |
| 0 — strict import of both controls | 0.07 min | $0.1609 | **passed** |
| 1 — generation/evaluation attestation | 2.80 min | $0.2070 | **passed** |
| 2 — real v2 tool+RAG smoke | 1.63 min | $0.2339 | **passed** |
| 3 — sa/sb characterization | 9.45 min | $0.3899 | **FAILED on sb** |

Stage 0 verified both controls strictly: weights sha256, probe id, observed
protocol fingerprint, seed, initialization digest, and a re-run of the strict
reconstruction reproducing `aad75fee8a897d9c…` with `completed_all_steps: true`
at step 1023. Both controls are untouched inputs; nothing was trained.

Engine actually observed: `vllm 0.27.1`, `torch 2.13.0+cu130`,
`transformers 5.15.0`, bfloat16, resolved context 8192 from `trained_block_len`,
runtime digest `85a14f8b…`.

## What failed: sb's checkpoint has no tokenizer

```
preflight_ctl_r0860k_sb generation rc=1
ValueError: Cannot use chat template functions because tokenizer.chat_template
is not set and no template argument was passed!
```

`preflight_ctl_r0860k_sa` was characterized. `sb` could not render a single
prompt, because its stored checkpoint is missing three files that `sa` has:

| file | sa | sb |
| --- | --- | --- |
| `chat_template.jinja` | present | **MISSING** |
| `tokenizer.json` | present | **MISSING** |
| `tokenizer_config.json` | present | **MISSING** |
| `config.json`, `generation_config.json`, `model.safetensors` | present | present |

This is an **artifact defect in the permanent control**, not orchestration and
not a Stage-3 scientific result. The Stage-0 import gate passed because it
verifies weights, protocol and probe identity — not tokenizer assets.

The three missing files are **byte-identical between `sa` and the canonical
initialization** (`chat_template.jinja` `3802169b…`, `tokenizer.json`
`be756060…`, `tokenizer_config.json` `8fa82a4b…`), so they are tokenizer assets
of the shared initialization, not seed-dependent. The engine probe independently
recorded `chat_template_sha256: 3802169b…` for the run that worked.

Repairing `sb` therefore needs **no retraining and no weight change** — but it
does modify a permanent control artifact, so it is the maintainer's call.

## sa's characterization is REAL and complete

Scored on the dev box by `scripts/autoinit/score_recovery_search.py` against the
frozen `recovery_search_v2` battery, `pooled_counts@v2` denominators:

```
n 190 · usable 79 · usable_rollout_rate 0.4158
n_scorable 170 · correct 3 · correct_overall 0.0176 · correct_given_usable 0.0380
per-capability usable: rag 0.8667 · gsm8k 0.5000 · multihop 0.5000
                       tool 0.4500 · math_verified 0.4000 · knowledge 0.0667
```

`code` (20 items) has no oracle, which is the 190 vs 170 difference. This is a
**single seed** and is not a Stage-3 verdict: the design is sa+sb, and any pass
threshold was registered against the pair.

## Battery wall time — a bound, not a measurement

Stage 3's 9.45 min contains sa's engine load, sa's full 190-prompt battery, sb's
engine load and sb's failure. So one control's battery is **under ~8 min**,
against a 24 min/control allowance. Useful for repricing Phase A, but it is an
upper bound on sa alone, not a clean per-control measurement, because sb's
failure shares the window.

## Not retried

Per the maintainer's instruction. `c398850b…` is consumed.
