# `data/eval-set/` — permanent evaluation sets

The project's evaluation data for **every stage** (maintainer decision
2026-10-18, `logs/budget/decisions.md`; the rule is AGENTS.md P10.3). Stage data
trees (`data/stages/stage-<n>/`) hold what a stage trains or calibrates on; what
a model is *judged* on lives here, once, and does not change underneath the
results that cite it.

| file | what it is |
| --- | --- |
| [`registry.json`](registry.json) | every set: status, the history of its authorized identities (manifest sha256 + the authorization that admitted each), open decisions, and `default_formal_measurement` |
| `eval-set-<n>/manifest.json` | every content file with its sha256 and size, the withheld files' identities, components, roles, capabilities, scorer protocols, exposure and licenses |
| `eval-set-<n>/README.md`, `NOTICE.md`, `audit/` | documentation and registration evidence; not identity-bearing |

## Rules

1. **Formal capability measurement reads the registered default set**
   (`default_formal_measurement`, today `eval-set-1`) unless a registered
   protocol names another registered set. No experiment re-samples or rebuilds
   an internal evaluation set for itself.
2. **A protected set does not change.** For a `candidate` or `frozen` set,
   items, gold answers, weights, scoring semantics and identity change only by
   appending a new identity to `identity_history` that cites an explicit
   **maintainer authorization**. An agent never does this on its own.
3. **New needs are new sets.** `eval-set-2`, `eval-set-3`, ... are added beside
   the old ones; an old set is never silently rewritten.
4. **Every read is verified.** Readers go through `shared.eval_sets`, which
   calls `aadistill.evaluation.eval_registry.require_intact`: a set whose bytes
   differ from the authorized identity is refused before anything is scored.
5. **Scoring semantics are pinned by conformance vectors**, not by freezing
   scorer bytes: `eval-set-<n>/conformance/*.jsonl` re-scored through the
   current scorers must reproduce the registered verdicts exactly.
6. **Report every unit and capability separately.** No set defines a pooled
   score. A unit the model cannot take is `unsupported`, a withheld unit
   without its verified local copy is `withheld`, an unrun unit is `not_run` —
   none of them is ever counted as zero.
7. **Isolation.** Training, calibration and search data must share no prompt or
   content key with a held-out unit: `shared.eval_sets.require_disjoint`, and
   builders reserve `shared.eval_sets.held_out_paths()`. An
   `in_sample_regression` unit is training data by definition and is never
   reported as generalization.

## Statuses

| status | meaning |
| --- | --- |
| `draft` | under construction; not protected |
| `candidate` | identity pinned and protected; open review items block freezing; formal measurement may read it and must carry its identity and status |
| `frozen` | declared frozen by the maintainer; protected |
| `retired` | kept for history; not read by new work |

## Checks

```bash
PYTHONPATH=src:scripts .venv/bin/python -m shared.eval_sets                       # bytes vs identity
PYTHONPATH=src:scripts .venv/bin/python scripts/shared/evaluation/eval_set_scoring.py conformance
.venv/bin/python -m pytest tests/data/test_eval_set_registry.py                    # both, in the core suite
```

The same checks run in CI on every change under `data/eval-set/`
(`.github/workflows/eval-set-integrity.yml`), and `.github/CODEOWNERS` routes
such changes to the maintainer.

## Sets

| set | status | components |
| --- | --- | --- |
| [`eval-set-1`](eval-set-1/README.md) | `candidate` | `fluid_reasoning_720` (held out, development-exposed) · `stage3_capability_846` (held out) · `stage3_legacy_150` (in-sample regression) |
