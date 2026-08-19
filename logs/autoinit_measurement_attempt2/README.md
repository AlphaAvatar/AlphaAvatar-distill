# Bounded measurement, attempt 2 — 2026-08-19, $0.1834, setup complete, driver died on its first import

**No measurement ran.** Setup passed end to end — `SETUP_RC=0`, all eleven
markers, both frozen assets present — and the driver then exited 1 in the first
statement of `main()` that touches the repository. **Authorization CONSUMED — it
covered one launch and must not be reused.**

| | |
| --- | --- |
| authorization | `autoinit.measurement.2026-08-19T1738Z`, sha256 `9cc117f64f8aadf1436e76946c680f80f5cb73ef6de58493114e87668bac0205` |
| grant | [`../autoinit_measurement_grant2.json`](../autoinit_measurement_grant2.json), sha256 `82f5104d49e4231f…` |
| authorized base | `dfa7bbd9fccdb0f6679605545ac806c178671194` |
| session commit | `9ffa9f498332a18e9c48d250739e1a0b773982ff` (authorization-only, one path) |
| harness digest | `abaf1460bcfbd6f3dbb313826fff916019366e2e375d6f52a22cbb3111ad3b83` (13 files) |
| bundle | `aad_autoinit_9ffa9f49.bundle` |
| pod | `1dxw5aw2d112jx`, 1×L40S @ $0.99/h |
| lifetime | 17:40:21 → 17:51:28 UTC, **11.1 min** |
| cost | **$0.1834** of a $1.6294 ceiling |
| terminal | `DRIVER_EXITED:1` |

## The cause

```
2026-08-19T17:49:15Z MARKER:MEASUREMENT_START
Traceback (most recent call last):
  File ".../measure_causal_depth_runtime.py", line 488, in <module>
    main()
  File ".../measure_causal_depth_runtime.py", line 370, in main
    from aadistill.autoinit.datasets import as_operator_items
ImportError: cannot import name 'as_operator_items' from
             'aadistill.autoinit.datasets'
```

`as_operator_items` lives in `scripts/autoinit/phase_a_search.py`. It has always
lived there. The import named a plausible owner — `datasets.py` is where the
calibration mixture is defined — and was never executed, because it sat inside
`main()`.

## Why $0 could not see it

Twenty-two tests covered this job. Every one of them called `run_measurement`,
`skip_set`, `GpuSampler` or the stop conditions **directly**. Not one called
`main()`, so the whole of the production entrypoint — argument defaults, the
pinned revision, model loading, calibration resolution, identity assembly, the
report, the stop conditions, the artifact write — was reachable only from a paid
pod. The tested surface and the executed surface were different surfaces, and
the untested one was the one with the bug in it.

That is the real finding. The import was a typo; the reason a typo survived to a
pod is that `main()` was structurally untestable and nobody had noticed.

## The repair

`run_entrypoint(args, *, hardware, teacher_loader, calibration, repo_root,
n_layers, n_remove)` now holds **all** of it, and `main()` is three lines:

```python
def main() -> None:
    args = build_parser().parse_args()
    report = run_entrypoint(args)
    print(json.dumps(report, indent=2))
```

The CPU test drives that same function with a toy loader, a fake hardware
object and a two-layer model, so the dev box executes the production path rather
than a parallel imitation of it. There is no second `main()`.

Mutation-verified — each of these was applied to the job and the $0 suite
re-run:

| mutation | result |
| --- | --- |
| the $0.18 bug restored (`from aadistill.autoinit.datasets import …`) | **1 failed** |
| the seam bypassed: orchestration back inside `main()` | **1 failed** |
| the unpinned-revision guard moved after loading | **2 failed** |
| the artifact is written but the report is not returned | **1 failed** |
| the loader stops passing the pinned revision | **passed — a real hole** |

The fifth is the interesting one. `load_teacher` is the single function the
entrypoint test *injects past*, so its body is the one place the seam cannot
cover, and a mutation dropping `revision=` from `from_pretrained` — which would
have measured against whatever the Hub published that morning — sailed through
every other test in the file. `test_the_real_loader_passes_the_pinned_revision_
through` now stubs `AutoModelForCausalLM` and asserts on the call, and both that
mutation and `use_cache` no longer being disabled fail it.

## The seam found a second defect before it ever ran

`resolve_calibration` is the *other* function the entrypoint test injects past,
and it had no test of its own. Executing it for real:

`CalibrationProfile.resolve()` returns the loaded **rows**, not a filename. The
job passed that straight into the report as `identities.calibration_path`, so
`str(calib_path)` would have written **734,042 characters** of serialized
mixture — every item, every token id — into a field labelled with a path. The
real path is 81 characters.

The measurement itself was unaffected: 67 items and 59,763 positions either way,
and `as_operator_items` takes rows. It was an artifact defect, and no recorded
result is touched, because this job has never completed a run. But it is the same
shape as the import — a line only a pod would execute — and it would have been
found by attempt 3 rather than at $0.

The path now comes from `profile.items_path`, and a test runs the real resolver
against the real frozen asset. Restoring the defect fails 2 tests; **the bad
import now fails 3 instead of 1**, because the resolver is executed rather than
matched as a string.

### A $0 test that was one broken guard away from real work

Mutation-testing the hardware seam found this. The refusal test is the only one
that calls `run_entrypoint` with nothing injected, so when the mutation stopped
the CUDA guard firing, it fell through to the real loader, loaded the cached
7.6 GB teacher and started a full CPU measurement — 900 s before the harness
killed it. Nothing was downloaded and nothing was damaged. The test now injects a
loader that raises and fails in 2.9 s. **A $0 test must stay $0 even when the
thing it guards is broken.**

## The second defect: the teardown gate

The launcher did not finish cleanly either:

```
[17:51:02]   manifest rc=5
  final_required: 3 · mutable_snapshot: 0 · final_streams_quiescent: False
  MISSING measurement_report [final_required]: audit/.../result.json
[17:51:15] LAUNCHER ERROR: ArtifactError: an emergency teardown over a
           non-quiescent event stream must name the streams it is truncating
```

The measurement declares **no event streams** — `event_streams=lambda ctx: ()`,
by design, because it produces one report and nothing incremental. It was still
asked to name the streams it was truncating.

`final_streams_quiescent` is one name over three different failures: a producer
never signalled completion; a file is still being written; or a `final_required`
class is simply **missing**. The first two are a tail being cut off. The third is
an artifact that was never created — which is what happened here, and it is not
a truncation at all.

`evaluate_teardown` now takes `streams_at_risk`, the manifest's own evidence
(`completion_marker_failures + still_being_written`), and requires naming only
when there is something to name. A session with no streams and a missing report
gets the honest note instead of an exception:

> No event stream was truncated: this session declares none, and quiescence
> failed because a `final_required` artifact is missing rather than because a
> producer was mid-write.

`streams_at_risk=None` — a caller offering no evidence — keeps the strict rule
unchanged. **Fail-closed behaviour for training sessions with incomplete event
streams is preserved**: those sessions do have streams, the manifest does name
them, and they still must be named in the record. Mutation-verified three ways
(restore the conflation → 1 failed; drop the naming requirement → 3 failed; give
no-evidence callers the weak rule → 2 failed).

The cost of this defect was $0.0034 — the 13 seconds between the ArtifactError
and the delete. It is worth fixing because the *next* session to hit it might not
have had a launcher that deletes the pod on error.

## What worked

Everything that failed in attempt 1. Setup reached `SETUP_RC=0` in 6.5 minutes
with both frozen assets staged and verified, which is exactly what
`test_session_setup_contract.py` was written to guarantee — the derived-roots
contract closed the class and the closure held on hardware. The authorization
chain was again sound: a new one-use `SpendAuthorization`, `phase_a_authorized:
false` by type, its own 13-file harness digest, an authorization-only commit
differing in exactly one path, byte-identical relay bundle. The watchdog
detached, the pod was deleted, and the provider confirmed it gone at 11.1
minutes against a 54-minute hard bound.

Two attempts, $0.2534 total, and each one bought exactly one contract that no
CPU could have told us was missing. This one bought the more valuable of the
two: the entrypoint is now executable at $0.

## Disposition

Recorded as **consumed, setup-complete, driver-entrypoint failure**. The grant
and the authorization are spent and must not be reused; a third measurement
needs a new grant and a new artifact, and **is not being prepared**. Cumulative
spend $206.0830 → **$206.2664** of $219.00, leaving **$12.7336**.
