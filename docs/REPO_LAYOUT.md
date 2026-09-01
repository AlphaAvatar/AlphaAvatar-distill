# Repository layout

Where things live and which rule governs each area. This is **reference**: it
changes when the structure changes, not when a run finishes. No live facts —
those are in [`logs/current_state.json`](../logs/current_state.json).

Every path named here must exist; `tests/docs/test_repository_structure.py`
fails otherwise, so this page cannot rot into a description of a repository that
no longer exists.

## Top level

| path | what it is |
| --- | --- |
| `AGENTS.md` | the binding working contract for coding agents. Read first. |
| `CLAUDE.md` | includes `AGENTS.md`; there is no second instruction file |
| `README.md` | timeless public overview. Owns **no** live facts |
| `src/aadistill/` | the algorithm core — reusable, model-recipe-agnostic |
| `scripts/` | executables: data, training, evaluation, rollout, pod sessions |
| `tests/` | mirrors `src/` and `scripts/` |
| `configs/` | frozen run configurations and artifact specifications |
| `logs/` | project memory. See [`logs/CATALOG.md`](../logs/CATALOG.md) |
| `docs/` | durable reference (this file, and its siblings) |
| `data/` | corpora; the large files are gitignored |
| `artifacts/` | generated locally, gitignored, never committed |

## Storage that is not in the tree

Most of the project's bytes are not under this directory, and a reader who
assumes otherwise will mis-plan a cleanup. There are four areas; sizes are
measured into [`logs/storage_measurements.json`](../logs/storage_measurements.json)
and their contents inventoried in
[`logs/checkpoint_registry.json`](../logs/checkpoint_registry.json).

| area | where | what it holds |
| --- | --- | --- |
| repository working tree | this directory | code, configs, logs, manifests |
| local artifact storage | `artifacts/` **and `/home/ecs-user/aad-artifacts/`** | run outputs; the out-of-tree store is where pod sessions collect checkpoints and optimizer states, and it is the larger of the two by two orders of magnitude |
| relay / LFS | the **AlphaAvatar/aadistill-artifacts** repository | the artifact store pods fetch from and push to; manifests in [`logs/artifact_manifests.md`](../logs/artifact_manifests.md) |
| scratch / session | `/home/ecs-user/aad-scratch/` | per-session working directories, bundles, poller output, pod-simulator quarantine |

## `src/aadistill/` — the algorithm core

| path | responsibility |
| --- | --- |
| `src/aadistill/autoinit/` | the AutoInitializer: search, operators, ranking, recovery, authorization |
| `src/aadistill/autoinit/device.py` | **the Stage-1 device contract**, `autoinit.stage1_device_contract@v1` |
| `src/aadistill/autoinit/search.py` | the beam engine and the materialize → reload → validate → measure cycle |
| `src/aadistill/autoinit/operators/` | the five frozen operator kinds and their implementations |
| `src/aadistill/autoinit/recovery.py` | the successive-halving plan, pooling, selection rules |
| `src/aadistill/autoinit/generation.py`, `src/aadistill/autoinit/generation_compat.py` | evaluation-protocol identity and `generation_runtime_comparability@v2` |
| `src/aadistill/autoinit/authorization.py` | `SpendAuthorization` — the type whose `allows_phase_a` is always `False` |
| `src/aadistill/autoinit/phase_a.py` | the Phase-A **schema and frozen plan**. Carries no grant prose |
| `src/aadistill/autoinit/phase_b.py` | the Phase-B session plan, its own executable-source identity, and the `PhaseBAuthorization` type whose `allows_phase_a` is False by type |
| `src/aadistill/autoinit/reweight.py` | the R1–R5 calibration reweighting rule. Build-time only; the pod never runs it |
| `src/aadistill/autoinit/fixed_path.py` | **Phase C1**: replays one frozen operator sequence with a fail-stop artifact-digest gate. Not a search — no enumeration, ranking, pruning or profile branching |
| `src/aadistill/autoinit/c1_session.py` | **Phase C1**: the ten-stage session contract, the two fail-stop replay gates, and `build_arm_specs`, which refuses to construct the treatment arm until the operator has been explicitly registered |
| `src/aadistill/autoinit/c1_isolation.py` | **Phase C1**: the two-arm isolation plan (no rungs, no survivors, no tie-break), the seed-derivation rule, the paired prompt-cluster bootstrap and the frozen three-way decision |
| `src/aadistill/autoinit/operators/attention_activation.py` | `attention.activation_importance_v1`. A **separate module** because `src/aadistill/autoinit/operators/attention.py` and its package `__init__` are members of `CONTINUATION_SOURCE_FILES_V2`; editing either moves a frozen Phase-B digest. **Import is inert** — a consumer calls `register()` explicitly. Staying outside `V1_IMPLEMENTATIONS` is not sufficient on its own, because an unrestricted `BeamSearch` enumerates the whole registry |
| `src/aadistill/init/attention_stats.py` | streaming per-head second moments of the attention output — the exact sufficient statistic for the C1 head score |
| `src/aadistill/init/` | Stage-0/1 primitives: activation statistics, contribution, sandwich init |
| `src/aadistill/data/`, `src/aadistill/evaluation/`, `src/aadistill/models/`, `src/aadistill/rollout/` | corpora, scorers, student construction, rollout |
| `src/aadistill/infrastructure/` | provider, budget, watchdog, artifact gate, log relay, manifests |
| `src/aadistill/infrastructure/session.py` | the typed, immutable `SessionSpec` a paid session IS |
| `src/aadistill/infrastructure/session_runner.py` | the one runner that executes a spec. Never subclassed |
| `src/aadistill/infrastructure/session_prechecks.py` | the shared $0 gates a session lists in `precheck` |

## `scripts/`

| path | responsibility |
| --- | --- |
| `scripts/autoinit/` | AutoInitializer tooling: preregistration, search, scoring, issuers, audits |
| `scripts/pod/` | paid-session executables. Catalogued in [`docs/POD_SCRIPTS.md`](POD_SCRIPTS.md) |
| `scripts/training/`, `scripts/evaluation/`, `scripts/data/`, `scripts/rollout/` | stage tooling |
| `scripts/consolidate/` | result consolidation |

## `configs/`

| path | contents |
| --- | --- |
| `configs/autoinit/` | artifact specifications and the operator ledger |
| `configs/stage3/` | frozen recovery recipes; `configs/stage3/e1/e1_r0860k_sa_pca.json` is the Phase-A probe base |

## `docs/`

| file | contents |
| --- | --- |
| `docs/REPO_LAYOUT.md` | this file |
| `docs/SESSION_ARCHITECTURE.md` | how a paid session is specified and run: the inheritance problem, and the IMPLEMENTED replacement |
| `docs/POD_SCRIPTS.md` | every pod script, classified |
| `docs/AUTOINIT_REFERENCE.md` | AutoInitializer binding rules, pinned assets, protocol requirements |
| `docs/POST_PHASE_B_GENERALIZATION.md` | the generalization backlog — **not to be started before Phase B completes** |
| `docs/archive/` | superseded documents, kept for provenance and clearly bannered |

## Rules that govern the layout

1. **The algorithm core holds no model-recipe constants.** Teacher ids, target
   geometry and frozen hashes live in the scripts and configs that own them.
2. **One owner per fact** (`logs/CATALOG.md`). A file that does not own a fact
   links to the one that does.
3. **Nothing is created before a milestone needs it** (AGENTS.md P2). Empty
   scaffolding is a defect, not preparation.
4. **Generated artifacts are never committed** — code, configs, manifests, small
   metadata and logs only (AGENTS.md §2.5).
5. **Historical implementations are not deleted for tidiness.** Reproducing a
   recorded result means reproducing the implementation that produced it
   (AGENTS.md P4), so retired experiment machinery stays and is classified
   instead.
