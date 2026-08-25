# Post-Phase-B generalization backlog

**Status: BACKLOG, recorded 2026-08-25. Nothing here is implemented, and none of
it may be started before Phase B completes.** Phase B must run against the
current, reviewed implementation; mixing a software-architecture rewrite into the
calibration-distribution experiment would make its result unattributable.

## The intended boundary

> **Core** describes mechanisms.
> **Adapters** describe model families.
> **Presets** describe data and experiment choices.
> **Session plans** describe execution.

The target is that a future experiment — *different teacher family → different
target size → different dataset mixture* — needs only:

* another architecture adapter, if the family is new;
* a new target `ArchSpec`;
* dataset and calibration presets;
* an experiment configuration and session plan;

while `search.py`, `ranking.py`, `state.py` and the operator framework stay
unchanged.

## What is already right, and must not be redesigned

Absent a real defect, leave these alone. They are the abstractions that make the
rest of this list small:

* **`ArchSpec` as an open structural field map**, not a fixed Qwen schema.
* **`ArchitectureAdapter` as the model-family boundary.**
* **Capability-driven operator dispatch** — the dispatcher already refuses to
  offer a GQA operator to a non-GQA adapter.
* **The open operator registry** — `register_kind` accepts kinds this module has
  never heard of; `MOE_EXPERT_SET`, `MOE_ROUTER` and `MOE_SHARED_EXPERT` are
  accepted without a core edit.
* **Generic search over `target_spec`.**

## Priorities, in order

### A. Move model-specific validation out of `search.py` — highest priority

The search's materialize → reload → validate → measure round-trip still
implicitly assumes a causal LM: it constructs token IDs, calls the model
positionally, and compares `.logits`.

`search.py` should not know that a model accepts token IDs, supports a positional
forward call, exposes `.logits`, or uses a particular vocabulary or input shape.
Move it behind an adapter-level round-trip / comparable-output contract.

This is the single largest obstacle to a non-causal-LM or non-text family.

### B. Separate generic calibration machinery from E8 experiment presets

**Stays in core:** `CalibrationProfile`; source and materialization identities;
the registry and resolution; role isolation; content verification.

**Moves to a thin `presets/` or experiment-configuration layer:** the E8 domain
names; the 59,763-position budget; `calib.domain_balanced@v1`;
`calib.reasoning_heavy@v2`; the R1–R5 reweighting rule; the project's dataset
registrations.

The generic calibration core should not permanently know about E8.

### C. Make data loading and content identity format-agnostic

Core still assumes JSONL items with `item_id` / `ids` fields and local-file
loading. Introduce a **small** data-adapter/artifact protocol with three
responsibilities — load, content identity, item identity — so core calibration
logic does not depend on the container format, the token field names, or the
files being local.

Not a dataset framework. A protocol.

### D. Generalize `domain` into a grouping/stratum concept

Future calibration distributions may vary by domain, task, language, difficulty,
modality, source or sequence-length bucket. Do not build a taxonomy engine —
simply stop baking `domain` into the generic abstraction as the only axis a
mixture can be stratified along.

### E. Make calibration budget units explicit

`59_763` is **prediction positions**, not generically "tokens", and the field is
called `token_budget`. Future profiles may be budgeted in positions, tokens,
samples, frames or seconds. A small typed budget/unit object would remove the
assumption.

**Do not retroactively change the frozen v2 profile identity** to fix the name:
`token_budget` is inside `profile_hash`, and `calib.reasoning_heavy@v2` is bound
by the Phase-B preregistration.

### F. Separate scientific search identity from execution identity

Scientific identity: `target_spec`, profiles, operators, ranking policy,
evaluation suite, search schedule. Execution identity: workdir, retention,
sharding, operational placement.

**Careful:** some runtime choices change numerical semantics — device, dtype,
accumulation dtype, attention backend. Those need their own *runtime* identity;
they must not simply become "unhashed" because they were reclassified as
execution.

### G. Parameterize resource planning

Search should not remain wired to a Qwen3 4B teacher, a 0.6B student, an L40S at
$0.99/h and a fixed storage estimate. Derive resource planning from model
footprint, data budget, the operator work plan and a hardware profile, rather
than accumulating phase-specific constants in the cost model.

## The simplicity constraint

**Do not turn every concept into a new interface, dataclass or protocol.** The
goal is not abstraction coverage.

Introduce an abstraction only when it removes a real hard-coded assumption about
one of: model family, data representation, experiment preset, or
hardware/resource environment. Prefer the smallest readable boundary that lets a
new model or data configuration work without changing generic core logic.

A refactor that adds five protocols and removes no hard-coded assumption has made
the codebase worse, and this project's standing rule (AGENTS.md P1) says so.
