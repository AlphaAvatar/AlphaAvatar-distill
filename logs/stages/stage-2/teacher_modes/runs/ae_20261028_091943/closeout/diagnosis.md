# A-E attempt 1 (ae_20261028_091943): engineering failure before any check ran on the teacher

- Pod `oodcj3rhp5mkgp`, A100-SXM4-80GB at $1.79/h, 7.55 min, **$0.2275** all in
  (`outcome.json`). Deleted at 09:30:03Z; the provider confirms it is gone, and the
  account holds only `am5m373kuq`.
- Setup took 3.7 min from create to `SETUP_DONE`: 1.6 min to SSH, then 2.1 min of setup.
  The driver ended `RUN_FAILED` 20 s after it started.
- Evidence secured off-pod (`evidence.tgz` sha256 `a38ce977…`, verified). The small
  files are in `../evidence/`, and `report.json` there is byte-identical to the archive's.

## What failed

The checks `prompt_identity` and `preclosed_scoring` ran and PASSED. Every check that
loads the teacher failed in the same call:
- `generation`
- `teacher_forward`
- `assistant_target_kl`
- `trainer_smoke`

The call is `teacher_source` → `snapshot_download(model, revision, local_files_only=True)`.
It raised `IncompleteSnapshotError` because 3 files are missing: .gitattributes, LICENSE
and README.md.

Setup's TEACHER_READY step downloads only `*.json *.safetensors *.jinja *.txt`.
huggingface_hub 1.x refuses an offline whole-repository lookup when the cached tree
listing names a file that the snapshot lacks. This is a wrong assumption in the
harness (P12.1). It is not a finding about the teacher or the protocol. No measurement
was taken.

The same failure reproduces on the dev box at $0 against the real repository. With the
fetch's own patterns, the lookup returns the snapshot directory.

## Repair

- `shared.pod.teacher_cache.local_snapshot` asks for exactly the files setup fetched.
- A test keeps those patterns equal to the setup script's.
- The A-E driver resolves the teacher through it.
- B-E's vLLM request now names the same local directory as its tokenizer instead of the
  repository id, so the offline engine makes no hub lookup.
- The study identity is unchanged. The harness changed, so the retry needs a new run and
  a new one-use artifact.

## Re-price

The envelope left $3.5 − $0.2275 = $3.2725. At the $1.79 quote, the 110-minute plan
(25 min setup) bounds at $3.3496 all in, which no longer fits. Setup is re-priced to
20 min, still more than five times the measured 3.7 min. The plan becomes:
- 20 setup + 5 transfer + 40 validation, ×1.25, plus reserves;
- **103.75 min** hard deadline;
- $3.1613 all in at $1.79, and $3.2663 at $1.85, the highest rate inside $3.2725.

The validation phase (40 min ×1.25) and the checks are unchanged. The granted rate
ceiling stays $1.87/h.
