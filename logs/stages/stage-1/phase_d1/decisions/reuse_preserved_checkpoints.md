# 2026-10-11 — A new attempt REUSES a preserved checkpoint; it does not retrain it

- **Context:** D1 behavioural screening has failed six times before producing a
  measurement, each time on a seam that had never executed. Three of those
  attempts trained a probe for ~61 minutes and preserved it. Attempt 7 is
  retraining all ten from scratch. The maintainer states the project principle:

  > 对于同一个实验，新的 attempt 必须要能重用之前失败保存的 checkpoint，而不是
  > 重新语料化或者训练，从而减少无意义浪费钱的工作。

- **Decision:** Adopted as the standing rule for this repository. A replacement
  attempt of the same experiment **resumes every preserved, identity-verified
  checkpoint at the next unfinished step** and retrains none of them. Where a
  probe's measurement is already complete and durable, it is not recomputed
  either.

## Why this does not conflict with the frozen design

`d1_design.json :: behavioural_design.protocol_uniformity` has two sentences
that say different things, and only one of them is the criterion:

- `requirement` — *"every probe in one comparison field is GENERATED and
  EVALUATED under one protocol identity"*
- `how` — *"all probes of a rung are trained and evaluated in ONE session on
  one host, under one image digest, one engine version and one scoring
  contract"*

The **requirement** binds generation and evaluation, and it is enforced in
code, per probe: the engine probe attests the live generation protocol, and
`admit_generation` refuses any probe whose rollouts were not produced under
that attested identity before a scorer may touch them. A resumed probe is
generated and evaluated in the resuming session, under that session's single
attested protocol, so the requirement holds for the whole field.

The `how` sentence describes the normal case, in which nothing was preserved.
It is not the discriminating condition the design's own `requirement` names.
An earlier round of this campaign read `how` as the criterion and concluded
that reuse was forbidden; that reading was stricter than the design's binding
clause, and it is superseded here.

## What is therefore recorded rather than forbidden

Cross-session **training** remains a real difference — a different physical
host, possibly a different NVIDIA driver (580.126.09 and 580.159.04 have both
appeared in this campaign), and a different floating-point reduction schedule,
since CUDA recovery training is reproducible in distribution and not
bit-for-bit. The honest treatment is to make it visible, not to pretend it is
absent:

- `preserved_identity.json` now carries `trained_in_run`, `trained_on_pod` and
  `trained_under_image` beside the digest.
- `trained_checkpoint_state` carries those onto the probe record and sets
  `_training_session_differs` when the resuming run is not the training run.
- So a field that mixes training sessions **says so** in the record a reader
  consults, instead of looking single-session by default.

## What is still not reusable

A probe that was never trained has no checkpoint to restore, so it must be
trained in whichever session reaches it. That is not a policy choice. The
useful consequence is that the reuse rule saves the most where it matters:

| attempt 7 fails at | preserved | attempt 8 |
| --- | --- | --- |
| before all ten trained | the probes trained so far | resumes those at scoring, trains only the ones never started |
| all ten trained, some unscored | every unscored probe | scores only; **no training at all** |
| all ten scored | every measurement | **no pod at all** — `rank_screening_offpod.py` computes the ranking and selection at `$0` |

- **Alternatives considered:** forbidding reuse and retraining the field each
  time, which is what the previous round did. It costs ~61 minutes and ~$1.1
  per probe to recreate bytes that already exist, and it does not even
  reproduce the same checkpoint — so it buys no identity it did not already
  have, while spending money the principle exists to save.
- **Expected upside:** a late failure stops costing a full retrain. Attempt 7's
  worst realistic case drops from ~13 hours of recomputation to scoring only.
- **Risks:** a mixed-training-session field carries an unquantified host/driver
  difference on the reused arms. Mitigated by recording it per probe rather
  than averaging over it, so any analysis can condition on it.
- **Revisit when:** a rung's reused fraction is large enough that the
  host/driver residual could plausibly rival the SESOI, or when a future design
  makes training-session identity part of its binding requirement.
