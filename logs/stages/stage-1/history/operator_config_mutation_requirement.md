# Required before the next Stage-1 initialization or search experiment

**An operator must not leave persistent config mutation visible to sibling
expansions.**

## What happened

`DepthCausalKLGreedyV1.apply` sets `model.config.use_cache = False` on the
parent model it is handed. It needs the cache off — `bypassed_blocks` scores a
block by skipping it, and a KV cache is indexed by layer — but it sets the flag
on the caller's object and never restores it.

`build_config` derives a child config by copying the parent's dict, so every
descendant inherits the flag. `use_cache` is serialized into `config.json`,
which lands in `config_sha256`, which lands in `artifact_digest`.

`run_phase_a_search` expands from **one** teacher object
(`root_loader=lambda: teacher`). Root implementations are expanded
deterministically and causal-KL DEPTH runs before positional DEPTH, FFN and
WIDTH. So from the first causal-KL DEPTH expansion onward, every later expansion
of **any** path began from a mutated root.

**A state's recorded identity therefore depended on beam order, not on its
path.** Two states with identical operator sequences and identical calibration
would carry different `artifact_digest`s according to whether a DEPTH sibling
had already run.

It cost `$1.13`: Phase-C2 replay attempt 8 reconstructed two of five leaves
byte-for-byte and diverged on a third whose weights were **identical** and whose
config was not. The full diagnosis is
[`config_lineage_forensic.json`](../phase_c2_replay/results/config_lineage_forensic.json).

## What is required

For any future initialization or search experiment — C3, C4, or any Stage-1
search — an operator must leave its parent's persistent config as it found it.

The scoped fix is small and generic: disable the cache only while the scoring
path needs it and restore the original value in a `finally`, or pass the setting
to the forward call rather than onto the config. No model, experiment, stage or
geometry may appear in it, and **one boolean does not need an abstraction**.

`tests/autoinit/test_use_cache_propagates_into_identity.py` pins the current
behaviour, including a check that greps the initialization tree so a second
mutation site cannot appear unnoticed. When the operator is repaired, that test
is the one to update.

## What this does NOT block

The C2 behavioural-selection session. It consumes the now-frozen checkpoints and
runs no initialization beam, so no operator mutates anything it reads.

The historical compatibility for the replay is **not** this repair: it lives in
the Phase-C2 replay application layer as an evidence-bound root pin, deriving
each path's historical root state from attempt 3's own recorded step-0 config.
That reproduces history and must not be moved into `src/aadistill`.
