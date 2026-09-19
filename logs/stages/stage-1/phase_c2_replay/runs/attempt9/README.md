# phase_c2_replay / attempt9

One run of `phase_c2_replay`, at `logs/stages/stage-1/phase_c2_replay/runs/attempt9/`.

**Canonical index: `manifest.json` in this directory.** It names every role this run recorded, its status and its cost. Read it rather than this file for anything factual: this README describes the layout and is not evidence that the run executed, succeeded or was authorized.

| area | what is in it |
| --- | --- |
| `governance/` | what permitted this run: the maintainer grant, the one-use authorization, the readiness record and the bundle record. INPUTS and one-use snapshots, not products. |
| `runtime/` | how it executed: the session record the runner writes on every path, the launcher console, the watchdog journal. |
| `evidence/` | what it observed: driver evidence, replay records, the marker stream. Observations, not conclusions. |
| `artifacts/` | what it produced or brought home. Large objects stay in the session scratch and are referenced by hash from `artifacts/manifest.json`; this directory holds reviewable text. |
| `closeout/` | how it ended: the outcome classification, the measured cost and the provider teardown confirmation. |

The repository-wide index of every run is `logs/index.json`.
