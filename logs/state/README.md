# logs/state

What is true **now**. Nothing here is history; nothing here authorizes anything.

| file | owns |
| --- | --- |
| [`current.json`](current.json) | the machine-readable snapshot |
| [`current.md`](current.md) | the same, for a human |
| [`ownership.md`](ownership.md) | which file owns which fact, and the class of every `logs/` entry |
| [`phase_index.md`](phase_index.md) | the scientific history by phase — status labels and links, no restated facts |
| [`experiment_index.md`](experiment_index.md) | what each experiment proved, what it does **not** support, and which conclusions are still in the current lineage |
| [`supported_models.md`](supported_models.md) | the supported-model log |
| [`artifact_manifests.md`](artifact_manifests.md) | external artifacts and their manifests |

Two of these index past work, which is not a contradiction: what a completed
experiment *established* is a current fact about what the project knows, and
"which of those conclusions still bind" is exactly the question a new session
opens with. The chronology behind them is not here — it belongs to the stage
that owns it, in [`../stages/stage-3/history/`](../stages/stage-3/history/).

`current.json` and `current.md` are **replaced, not appended**. A narrative of
what happened belongs to the experiment that owns it, under
[`../stages/`](../stages/); the money belongs to
[`../budget/`](../budget/).

The readiness block in `current.md` and `latest_verification` in `current.json`
are **generated** from the readiness record by
[`render_log_navigation.py`](../../scripts/consolidate/render_log_navigation.py).
Do not hand-edit them: they went stale within hours when they were prose.
