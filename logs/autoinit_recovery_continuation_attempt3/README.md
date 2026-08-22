# Recovery continuation attempt 3 — no stage ran, $0.2011

**Verdict: fail-closed in the setup test gate, 12.2 minutes into the pod. The
transport worked; a dev-box path inside a publishing tool did not.** No stage
executed, no science changed, the five leaves are untouched.

| | |
| --- | --- |
| authorization | `autoinit.recovery_continuation.2026-08-22T1311Z`, sha256 `021d8830…` |
| grant | `logs/autoinit_recovery_continuation_attempt3_grant.json` |
| base commit | `7368568` |
| session commit | `ad73e05a34c5144a5a16e2b85622637cf45ed610` |
| harness digest | `162c09ed7d6fb2da28ecac98c326a8144f3f4d66f5af6b1c6e5282ecd91fcfe7`, 22 files, search excluded |
| bundle | `aad_autoinit_ad73e05a.bundle`, sha256 `d2ea1df7…` |
| pod | `ku8vcn5mu8hp9i`, L40S $0.99/h, **12.19 min, $0.2011**, provider confirms gone |
| terminal | `INCOMPLETE`, `setup_failed` after draw 1 |

## The transport premise is now PROVEN on a paid pod

This is the result worth keeping. Attempt 2 died pushing the first 1.110 GiB leaf
by scp. Attempt 3's pre-provider gate read:

```
precheck OK: 25 relay inputs (10 from AlphaAvatar/aadistill-artifacts,
             15 from AlphaAvatar/aadistill-transport), 2 local assets
```

and the pod then **reached `MARKER:VLLM_READY`, `MARKER:TEACHER_READY` and
`MARKER:ROPE_OK`**. The setup script runs under `set -euo pipefail` and marks
strictly in order, and `ASSETS_STAGED` (line 193) and `ASSETS_READY` (line 313)
both precede `VLLM_READY` (line 404). Reaching those markers therefore proves the
staging block ran to completion: **all 25 declared inputs were fetched from their
own declared repositories and every declared sha256 was verified at every landing
site**, including 5.5513 GiB of Stage-1 leaves pulled from the transport repo at
hub speed.

The multi-repo relay contract and the five-leaf mirror are no longer a `$0`
claim. They are a paid-pod observation.

## What failed

```
[13:42:19] CPU test suite (128 vCPUs visible, cgroup budget 13; cpu set 0-12)
FAILED tests/autoinit/test_leaf_transport_publish.py::test_a_size_mismatch_at_the_far_end_is_caught
FAILED tests/autoinit/test_leaf_transport_publish.py::test_an_lfs_oid_that_disagrees_is_caught_without_downloading
FAILED tests/autoinit/test_leaf_transport_publish.py::test_a_file_absent_from_the_far_end_is_caught
5 failed, 2053 passed, 84 skipped in 219.10s (0:03:39)
[13:45:59] test suite failed rc=1
SETUP_RC=1
```

The launcher captures only a 40-line tail of `setup.log`, and the pod is gone, so
two of the five failure names were not transported. **They were recovered by
reproducing the failure locally at $0** — see below — and are
`test_the_before_manifest_refuses_a_drifted_canonical_shard` and
`test_a_corrupted_remote_file_is_caught_by_the_round_trip`.

### Root cause, reproduced at $0 rather than inferred

`scripts/autoinit/publish_selected_leaves.py:199`:

```python
tmp = Path(tempfile.mkdtemp(prefix="leaf-roundtrip-", dir="/home/ecs-user/aad-scratch"))
```

`verify()` hard-codes a **dev-box absolute path** as the parent of its round-trip
scratch directory. On a pod that directory does not exist and `mkdtemp` raises.
Four tests call `pub.verify()` and a fifth reaches the same line.

Reproduced by running the real test module in a mount namespace holding the repo
and the interpreter but **no `/home/ecs-user/aad-scratch`** — the pod's condition:

```
aad-scratch exists? NO
FileNotFoundError: [Errno 2] No such file or directory:
  '/home/ecs-user/aad-scratch/leaf-roundtrip-p4qyecmz'
5 failed, 4 passed in 0.25s
```

**5 failed**, matching the pod's count exactly, with the three transported names
among them.

## Why no $0 gate caught it — and it is the attempt-8 class again

Each gate was true and none was sufficient:

* **the layout test** partitions its references: a *declared* host-local storage
  root is verified where present and **skipped where absent**. `/home/ecs-user/
  aad-scratch/` is declared in `docs/REPO_LAYOUT.md`, so on a pod the layout test
  correctly skips it — and nothing connects "this path is host-local" to "code
  requiring it must not execute on a pod";
* **the pod simulator** simulates the pod's *repository tree*, not the pod's
  *host filesystem*. It runs on the dev box, where the directory exists;
* **the continuation harness gate** digests 22 files, and
  `publish_selected_leaves.py` is **not one of them** — correctly, since it is a
  dev-box publishing tool the session never executes. But its *test module* is
  not in `TEST_IGNORES` (which holds only two entries), so the pod ran it;
* **the dev-box suite** passed 2160/12 twice, at the pre-authorization baseline
  and again at the session commit, because the directory is right there.

Attempt 8 was *"a `$0` test asserting dev-box filesystem state"* and cost $0.1900.
This is the same class one step further out: a `$0` test **executing production
code that requires** dev-box filesystem state. The transport work added both the
tool and its tests on 2026-08-22, and the tests entered the pod's gate without
anyone asking whether the tool they exercise can run there.

## Not attempted, and why

No repair on the live pod and no relaunch. The grant makes a failed setup gate a
fail-closed stop, and the authorization is spent. A fix to
`publish_selected_leaves.py` would not move the harness digest — the file is not
in the harness set — but it would move the session commit, and the lineage gate
requires the session commit to differ from its authorized base in exactly one
path. **A retry needs a new grant and a new authorization.**

## State

* pod deleted, **provider confirms gone**; watchdog ended `pod_gone` after 13
  ticks; the independent poller's last observation was `RUNNING` at 13:44:37Z and
  the provider now returns zero pods; nothing is billing;
* the five Attempt-12 leaves are **untouched**, canonically in
  `/home/ecs-user/aad-artifacts/autoinit/phase_a/` and mirrored in the transport
  repo; frozen science untouched;
* `$213.9214` cumulative against the `$234.00` cap.
