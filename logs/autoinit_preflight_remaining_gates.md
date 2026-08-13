# Remaining execution-verification gates before the paid preflight

**Status: items 3 and 5 are implemented and rehearsed. Items 1, 2 and 4 are NOT
implemented. The preflight must not launch until they are.**

The principle behind all three is one sentence: *the harness currently proves what
the driver intended to run, not what actually ran.* Stage 2 builds a
`RecoveryProbeIdentity` from the already-attested protocol and compares it back to
that same object, which is a tautology; Stage 3 does not check the rollout
metadata at all; and setup verifies the frozen assets against hashes read out of
their own manifests, which proves self-consistency rather than identity with the
preregistered asset.

## 1. Stage 2 — reconstruct the protocol from the run's own artifacts

Add `RecoveryProtocolFingerprint.from_run_artifacts(run_manifest, *, runtime,
trainer_source)`. **The reconstruction already exists**: `historical_protocol()`
in `scripts/autoinit/compare_recovery_fingerprints.py` builds a fingerprint from a
`run_manifest.json` and is the function to promote into the class. It reads
`config.{data_dir,rung,block_len,packing,val_blocks,dtype,autocast_bf16,
gradient_checkpointing,trainable_patterns}`, `config.{loss,optim,schedule,batch}`,
`ladder.{train_blocks,train_supervised_tokens}`, `teacher.*`, `tokenizer_sha256`
and `trainable_params`.

Then in `Driver.verify_control`, replace the current construction with:

    observed = RecoveryProtocolFingerprint.from_run_artifacts(...)
    comparison = observed.compare(self.attested)
    if not comparison["protocol_identical"]: reject the control
    probe = RecoveryProbeIdentity(protocol=observed, ...)

Fail closed when the manifest lacks a required field: a value filled from the
preregistered object would make the comparison a tautology again. Bind the
checkpoint to (observed protocol hash, probe id, weights sha256).

## 2. Stage 3 — reconstruct the generation protocol from the rollout summaries

`uncapped_eval.py` already emits everything needed in each `<set>.json` summary:
`libraries`, `chat_template_sha256`, `tokenizer_source`, `system_message`,
`protocol`, `chat_template_kwargs`, `thinking_mode`, `context_len`,
`context_resolution`, `stop_ids`, `degeneration_stop`, and `engine` (which carries
`vllm_version`, `max_num_seqs`, `max_num_batched_tokens`, `enforce_eager`). Do not
change the generator.

Add `RecoveryGenerationProtocolFingerprint.from_run_summaries(summaries, ...)`
that requires all summaries of one evaluation run to agree field by field, then
compare against the Stage-0 attested generation fingerprint before scoring is
accepted, and build the observed `RecoveryEvaluationProtocol` from it. Bind each
sa/sb result to that hash. Do not backfill a missing observed value from the
expected fingerprint.

Note: `temperature`/`top_p`/`top_k`/`detokenize` are *not* in the summary today.
Either record them in the summary (a purely additive change to what is written,
not to generation semantics) or declare them non-reconstructable and fail closed —
do not assume them.

## 3. DONE — engine-observed fields fail closed

`MATERIALIZATION_REQUIRED` now includes `max_num_seqs`,
`max_num_batched_tokens`, `enforce_eager` and `context_source`, 14 fields in all.

## 4. Setup — pin the frozen assets to preregistered constants

`autoinit_preflight_setup.sh` currently reads `manifest["content_sha256"]` and
checks it against itself. Replace with explicit constants bound to the
preregistration:

    recovery_search_v1  content  a1b22778b00d95b6aba358c14a5af5b559fd807bb371c92131eacca59479f323
    recovery_search_v1  manifest 72d8c0535e7752faf704d9075b7835a47610fd3cd26866cf5be7d48eb7b40ad1
                                 (canonicalized manifest_sha256 convention, NOT
                                  sha256_file of the raw bytes -- both exist in
                                  this repo and comparing across them looks like
                                  corruption)
    state_eval_v1       content  read from logs/autoinit_phase_a_preregistration.json
                                 -> state_evaluation.content_sha256, and pin it as
                                 a constant here rather than from the live file

A mismatch is a Stage-0/setup blocker: record evidence, do not enter Stage 1,
teardown, stop.

## 5. DONE — enforced harness/session identity

`consuming_commit` is replaced by enforced `authorized_session_commit` +
`harness_source_digest` over eight declared files, with `provenance_commit` kept
separate and never enforced. `SpendAuthorization.require_harness()` is called in
the launcher's `__init__`, before a pod can exist. Re-issue with
`scripts/autoinit/issue_authorization.py` after any harness edit -- and re-rehearse
first, which is the point of the pin.

## Rehearsal still owed

    Stage-2 run manifest differs from attested        -> control rejected
    Stage-2 required protocol field missing           -> fail closed
    Stage-3 rollout metadata differs from attested    -> characterization rejected
    Stage-3 material generation field missing         -> fail closed
    state_eval manifest/content mismatch              -> blocked before measurement
    recovery_search manifest/content mismatch         -> blocked before characterization

The existing 21 scenarios, including the success and teardown paths, must keep
passing.
