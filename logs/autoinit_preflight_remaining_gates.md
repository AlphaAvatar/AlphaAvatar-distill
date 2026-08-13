# Execution-verification gates before the paid preflight

**Status (2026-08-13, third pass): all five items are implemented and rehearsed.
The harness is complete; the authorization was re-issued against it.**

The principle, in one sentence: *the harness proved what the driver intended to
run, not what actually ran.* That is now closed on both sides.

| # | gate | state |
| --- | --- | --- |
| 1 | Stage 2 — strict observed `RecoveryProtocolFingerprint` | **DONE** |
| 2 | Stage 3 — strict observed `RecoveryGenerationProtocolFingerprint` | **DONE** |
| 3 | engine-observed generation fields fail closed (14 fields) | DONE (2nd pass) |
| 4 | setup pins the frozen assets to preregistered constants | DONE (2nd pass) |
| 5 | enforced harness/session identity | DONE (2nd pass) |

## 1. Stage 2 — the protocol is reconstructed from the run's own artifacts

`RecoveryProtocolFingerprint.from_run_artifacts(run_dir, repo_root=..., strict=True)`
and the richer `observe_recovery_protocol()`, which also returns the evidence.

`historical_protocol()` was **not** promoted. It stays where it is, permissive, for
runs whose evidence no longer exists. The strict path shares none of its defaults:

    kd_chunk        resolved by the trainer through KD_CHUNK_DEFAULT and recorded
    optimizer       type(self.opt).__name__, off the constructed object
    lr / betas /    self.opt.defaults, cross-checked against the config
      eps / decay
    lr_schedule     LR_SCHEDULE_ID, defined beside lr_factor
    block_ordering  BLOCK_ORDERING_ID, corrected: the stream is a per-epoch
                    torch.randperm seeded from the run seed, not "sequential,
                    no shuffle"
    pack hash       RECOMPUTED from the pack the run named, and required to equal
                    what the run recorded
    teacher / tok   the identity `load_teacher` resolved, and the tokenizer hash
    trainer digest  written by the run, compared against the attestation
    runtime digest  written by the run, image digest from AADISTILL_IMAGE_DIGEST
    accounting      run_completion.json: every declared step ran, and consumed
                    blocks == step x blocks_per_step

A material field with no evidence raises, listing all of them at once. `strict=False`
keeps the forensic behaviour: the gap becomes `unverifiable`, and `compare` reports
it as unknown, never as matched.

`Driver.verify_control` now builds the observed protocol, compares it to the
Stage-0 attested fingerprint, independently hashes the initialization the run
actually read, and only then constructs `RecoveryProbeIdentity(protocol=observed,
…)`. The permanent control is bound by three hashes: observed protocol
fingerprint, probe id, checkpoint weights sha256. A rejected control is bound by
none.

## 2. Stage 3 — the generation protocol is reconstructed from the rollouts

`RecoveryGenerationProtocolFingerprint.from_run_summaries(summaries)` /
`observe_generation_protocol()`. Every material field is read from the stored
summaries via `SUMMARY_FIELD_PATHS`; all sets of one evaluation must agree; the
result is compared to the attested fingerprint **before scoring**, and each
`sa`/`sb` result binds to an observed `RecoveryEvaluationProtocol` that must be
comparable with the attestation.

`uncapped_eval.py` emits the identity it did not carry — tokenizer hash, the
generation and degeneration source digests, the generation runtime and its digest,
the tokenizer-source *rule* (the raw path names the arm and cannot be a shared
identity), and the four rule strings — all from constants shared with
`generation.py`. Generation semantics are unchanged.

Two things this closed that were not on the list:

* Stage 0 filled the generation protocol's `torch`/`transformers`/`runtime_digest`
  from the **training** venv. Rollouts run in `/opt/vllm`; the attested protocol
  described a stack that never generates a token, and no observed reconstruction
  could ever have matched it.
* The declared `max_tokens_rule` and the generator's own prose had already drifted
  into two sentences for one behaviour.

## Rehearsal — `tests/pod/test_observed_protocol_rehearsal.py`

    1. observed Stage-2 protocol mismatch          -> control rejected, no binding
    2. missing Stage-2 material field              -> fail closed (8 fields, each)
    3. observed Stage-3 generation mismatch        -> characterization rejected
    4. missing Stage-3 material field              -> fail closed (11 fields, each,
                                                      absent AND null)
    5. state_eval frozen identity mismatch         -> blocked before measurement
    6. recovery_search identity / scoring drift    -> blocked before it

Scenarios 1 and 2 drive the **real `train_stage3.py`** on a toy ladder and then the
**real `Driver.verify_control`** over its artifacts, because the thing under test is
a contract between two programs and a fixture written by the test would prove
nothing about the writer. The one field the offline path cannot produce is the
teacher identity (a Hub call); scenario 2 uses that gap rather than simulating one.

The 21 existing rehearsal scenarios still pass, including the success path and its
proof that Phase A is unreachable.

## Two paid-path defects this work found

Neither was reachable by the previous rehearsal, and both would have fired on the
pod after money had been spent:

* the driver invoked `train_stage3.py --out-dir`, an argument it does not accept —
  Stage 2 would have failed immediately, on the first control;
* `save_pretrained` writes no tokenizer, and `AutoTokenizer.from_pretrained` on
  such a directory returns a **vocab-size-1** tokenizer instead of raising, so
  Stage 3 would have generated from a degenerate tokenizer.

A third came from the pod-environment simulation: four tests read gitignored
artifacts the preflight does not stage, so the setup test gate would have failed
after `uv sync`, the vLLM install and the teacher download. They now skip.

## Pins re-issued against the final harness

    scoring contract   recovery_search_scoring@v2  f76008d5… -> 69591aab…
                       (recovery.py gained the reconstruction; the metric did not
                        move — nine policies x 190 prompts reproduce every number)
    preregistration    9b4229c8… -> 1d70a91a…  (15 leaves, all identity digests)
    trainer source     054dd6d6… -> 729bc9e6…
    harness digest     re-issued by scripts/autoinit/issue_authorization.py
    declared generation fingerprint  f4ac7448…  UNCHANGED
