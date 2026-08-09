# Live control-plane canary — clean rerun, 2026-08-09 — **PASSED 12/12**

**One launch command. No manual watchdog restart, no SSH repair, no
substitution.** The complete automated chain succeeded unaided.

Nothing is left running. Cost **$0.033** against a $0.12 backstop.

---

## 1. Provider-level facts

| | |
| --- | --- |
| pod id | **`3hvb5d4it6h6pb`** |
| GPU | **NVIDIA RTX 2000 Ada Generation** — cheapest of six quoted; no GPU compute used |
| quoted / provisioned `securePrice` | **$0.240/h** |
| created | **2026-08-09 12:09:24 UTC** |
| terminated (provider-confirmed) | **2026-08-09 12:17:42 UTC** |
| pod lifetime | **8.30 min** |
| **actual cost** | **8.30 / 60 × $0.240 = $0.0332 ≈ $0.033** |
| authorized backstop | **$0.12** — derived to 30.0 min at the live quote |
| `--terminate-after` (redundant) | 12:39:23 UTC — never reached, did not fire, **not counted** |
| **final pod state** | `exists=False · desired_status=TERMINATED · billing=False` |
| `runpodctl pod list` after the run | `[]` |

Quotes before creation: RTX 2000 Ada **$0.240** (Low), A4000 $0.250 (Low), A4500
$0.250 (Low), A5000 $0.270 (Low), A6000 $0.530 (Medium), RTX 4090 $0.740 (High).

**The backstop is derived, not assumed.** $0.12 at $0.240/h buys 30.0 minutes;
that became the backstop and `--terminate-after`, with the safety watchdog at
25.0. Had the quote bought fewer than 20 minutes, the driver would have aborted
and reported rather than widening it.

Machine-readable: [`e7_canary_rerun_evidence.json`](e7_canary_rerun_evidence.json).
Journals and manifests: [`e7_canary_rerun/`](e7_canary_rerun/).

---

## 2. All twelve criteria

| # | criterion | result | evidence |
| --- | --- | --- | --- |
| 1 | detached launch returns promptly | **PASS** | **5.46 s**; `start_channel_closed=True`, pid 297 |
| 2 | durable job descriptor created automatically | **PASS** | `canary.job.json` read back over a later ssh connection; liveness `ALIVE` |
| 3 | events relayed while the remote process is active | **PASS** | 4 cycles, **27 events**, job liveness **`ALIVE`** at the check |
| 4 | provider-only watchdog starts and polls the live billing pod | **PASS** | 14 polls, `pod_billing=true`, `RUNNING`, accruing $0.0179 |
| 5 | the deliberately broken primary `runpodctl` path fails as intended | **PASS** | `ok=false`, `FileNotFoundError`, `verified_transport=true` |
| 6 | GraphQL `podTerminate` invoked **automatically by the same watchdog** | **PASS** | `ok=true`, `{"data": {"podTerminate": null}}`; **test watchdog launches = 1**; 5 polls under the limit, then `hard_limit_reached` at 8.27 min |
| 7 | provider polling verifies `exists=false` / terminated / non-billing | **PASS** | `pod_exists=False · TERMINATED · billing=False` on the first verify poll |
| 8 | journal durably records the complete termination sequence | **PASS** | `watchdog_start → poll ×6 → hard_limit_reached → terminate_attempt → terminate_verify → terminated → watchdog_end`, 2,840 bytes on disk |
| 9 | mutable live-snapshot semantics without hash races | **PASS** | see §3 |
| 10 | final artifact/hash verification under declared semantics | **PASS** | see §3 |
| 11 | orchestration returns PASS with no human repair | **PASS** | launches `{safety: 1, test: 1}`, `phase_2_invoked=False` |
| 12 | no pod remains running | **PASS** | independent post-run query and `runpodctl pod list` both empty |

---

## 3. Both artifact lifecycles, exercised live in one session

The same file, twice, under the two claims a manifest can make about it:

| phase | lifecycle | bytes | grew during archive | `final_streams_quiescent` | gate |
| --- | --- | ---: | --- | --- | --- |
| **snapshot** (job writing) | `mutable_snapshot` | 1,910 | **yes**, 1,846 → 1,910 | true¹ | allowed |
| **final** (after `MARKER:ALL_DONE`) | `final_required` | 3,190 | no | true | allowed |

¹ no `final_required` *event stream* was declared in the snapshot phase, so there
was nothing to be non-quiescent about — the snapshot makes no completeness claim
by construction.

**The snapshot phase is the 2026-08-09 failure, now passing.** The log grew by
64 bytes between manifest and archive — exactly the race that failed the first
canary — and the bounded read captured the boundary, hashed the bytes it wrote,
rewrote the entry, and recorded the growth in `appended_during_archive`. Local
hash verification matched. **This proves already-emitted data is durable; it
claims nothing about completeness.**

**The final phase is what a normal E7 teardown will do.** After the job's
terminal marker, the same stream was re-manifested as `final_required` with a
6-second settle window and marker verification: 3,190 bytes, no growth, no
`still_being_written`, no `completion_marker_failures`, and the full gate
allowed with `emergency=False` and `incomplete_event_streams=()`.

A bounded prefix of a growing file **cannot** reach that state: `create_archive`
refuses a `final_required` file that grew, and `build_manifest` records one that
moves in the settle window as still being written. Covered by
`tests/infrastructure/test_artifact_gate.py`.

---

## 4. What changed since the failed run

| defect | where found | fix |
| --- | --- | --- |
| RunPod 403s the `Python-urllib` User-Agent | pre-flight, $0 | explicit UA; `tests/infrastructure/test_provider.py` |
| manifest/archive hash race on an appending file | first canary, live | bounded read; the manifest describes what was archived |
| **a bounded prefix could pass as a final artifact** | this correction, $0 | `mutable_snapshot` vs `final_required`, markers, settle window, `final_streams_quiescent` gate check |
| `watchdog.py` lacked the `--runpodctl` flag `canary.py` passed | first canary, live | flag added; a test asserts **every** flag one script passes another exists |

The first run's GraphQL observation stands as the **first live observation** of
that transport. This rerun is the second, and it is the one that proves the
*chain*: the same watchdog that crossed its own threshold invoked the fallback
itself, on its first and only launch.

---

## 5. Standing limits

The canary proves the provider/control-plane path. It does **not** approximate a
multi-hour session, and deliberately does not try to: long-session
watchdog/liveness behaviour is covered by deterministic local simulation
(`test_e6b_failure_simulation.py`, 15 tests including the 434-minute overrun
shape against a fake clock). `--terminate-after` remains never observed to fire
and is still not counted as a stop mechanism.

---

## 6. Accounting

```
cumulative spend before rerun:  $149.635
rerun cost:                     $  0.033
actual cumulative spend:        $149.668
temporary cumulative cap:       $150.41
remaining under the cap:        $  0.742
```

Both canary runs together cost **$0.078** of the $0.12 + $0.82 authorized across
them.

**E7 B and C are not launched.** E7 still requires separate authorization for its
$12.82 hard backstop and the $163.23 cumulative cap.
