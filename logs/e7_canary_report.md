# Live RunPod control-plane canary — 2026-08-09 — **FAILED**

**Verdict: FAILED. 9 of 10 required criteria passed; one failed on a real
defect. Per the authorization, E7 is not launched.**

Nothing is left running. Total spend **$0.045** against a $0.82 backstop.

---

## 1. Provider-level facts

| | |
| --- | --- |
| pod id | **`bd83jug4g23qn0`** |
| GPU | **NVIDIA RTX 2000 Ada Generation** (the canary needs no GPU compute) |
| quoted / provisioned `securePrice` | **$0.240/h** — cheapest of six candidates |
| created | **2026-08-09 11:35:03 UTC** |
| terminated (provider-confirmed) | **2026-08-09 11:46:10 UTC** |
| pod lifetime | **11.12 min** |
| **actual cost** | **11.12 / 60 × $0.240 = $0.0445 ≈ $0.045** |
| authorized backstop | $0.82 |
| `--terminate-after` (redundant layer) | 12:25:02 UTC — **never reached, did not fire, does not count** |
| **final pod state** | `exists=False · desired_status=TERMINATED · billing=False` |
| `runpodctl pod list` after the run | `[]` |

Quotes taken before creation: RTX 2000 Ada $0.240 (Low), A4000 $0.250 (Low),
A4500 $0.250 (Low), A5000 $0.270 (Low), A6000 $0.530 (Medium), RTX 4090 $0.740
(High). Worst case at the 50-minute backstop was computed as $0.200 and checked
against the authorization **before** anything was created.

Machine-readable: [`e7_canary_evidence.json`](e7_canary_evidence.json).
Journals: [`e7_canary/`](e7_canary/).

---

## 2. Criteria

| # | criterion | result | evidence |
| --- | --- | --- | --- |
| 1 | detached launch returns promptly | **PASS** | returned in **5.58 s** against a 30-minute job; `start_channel_closed=True`, pid 231 |
| 2 | durable job descriptor created | **PASS** | `/workspace/jobs/canary.job.json` read back over a **later** ssh connection; liveness `ALIVE` |
| 3 | structured logs off the pod before teardown | **PASS** | 4 relay cycles, **33 events** (steps 0–32) durable locally; all `canary_tick`; the job was still running when the pod died |
| 4 | provider-only watchdog sees the live billing pod | **PASS** | safety watchdog, 8 polls by that point: `pod_billing=true`, `desired_status=RUNNING`, accruing $0.0097 |
| 5 | watchdog crosses its threshold without launcher assistance | **PASS** | polls at 10.40 and 10.74 min `over_hard_limit=false`, poll at **11.09 min `over_hard_limit=true`** → `hard_limit_reached` |
| 6 | **GraphQL fallback transport actually exercised** | **PASS** | see §3 |
| 7 | termination issued and journalled | **PASS** | 1 round, both attempts recorded with methods, ok flags and errors |
| 8 | provider polling confirms disappearance | **PASS** | verify poll: `exists=False, desired_status=TERMINATED, billing=False` → `terminated`, `watchdog_end reason=pod_gone` |
| 9 | **artifact manifest + local hash verification** | **FAIL** | see §4 |
| 10 | no pod remains running | **PASS** | independent post-run query and `runpodctl pod list` both empty |

---

## 3. The GraphQL fallback works — first time ever exercised

The primary path was forced to fail by pointing the watchdog at a binary that
does not exist. **No provider state was altered to induce the failure**; only
that process's view of the CLI changed.

```json
{"event": "terminate_attempt", "round": 1, "any_ok": true, "attempts": [
  {"method": "runpodctl remove pod", "verified_transport": true, "ok": false,
   "error": "FileNotFoundError: [Errno 2] No such file or directory: '/nonexistent/runpodctl-canary-forced-failure'"},
  {"method": "graphql podTerminate", "verified_transport": false, "ok": true,
   "response": "{\"data\": {\"podTerminate\": null}}"}]}
{"event": "terminate_verify", "round": 1, "poll": 1,
 "pod_exists": false, "desired_status": "TERMINATED", "billing": false}
{"event": "terminated", "round": 1, "polls": 1}
```

**`podTerminate` is no longer an unverified transport.** It accepted the
mutation and the pod was gone on the very next verification poll. It remains
marked `verified_transport: false` in the journal schema — that flag records
"this transport has no CLI-level track record in this project", and one
observation does not retire it.

**Independent confirmation.** The *safety* watchdog — a separate process, real
CLI, its own journal — polled at 11:37 with `pod_billing=true, RUNNING` and at
11:46:26 with `pod_exists=false, TERMINATED, billing=false`, then exited
`pod_gone`. Two processes, two transports, same conclusion.

---

## 4. Why it failed: manifest and archive were not atomic

The pod-side manifest hashed `canary/train_log.jsonl` at **2,166 bytes**. `tar`
read it a moment later at **2,230 bytes** — the job had written one more event
in the gap. `verify-archive` then reported a size mismatch, the local hash
verification failed, and the teardown gate blocked at
`archive_contents_verified`.

| file | manifest | archive |
| --- | ---: | ---: |
| `canary/train_log.jsonl` | 2,166 | **2,230** |
| `jobs/canary.job.json` | 172 | 172 |

**The gate was right and the workflow was wrong.** `train_log.jsonl` is appended
by the trainer for the whole run, so an E7 session would have hit this on every
attempt, and the gate would have refused teardown forever with nothing actually
missing. Worse, the failure mode is one that invites the wrong fix — relaxing
the size check — which would have re-opened the exact hole E6b fell through.

**Fixed** (`artifact_gate.create_archive`): each file is read **once**, capped at
the size observed when it is opened, hashed as it streams into the tar, and its
manifest entry rewritten to those exact bytes. The manifest now describes what
was archived rather than what was on disk a moment earlier. A size-capped read
of an append-only jsonl is a valid prefix; a file that has **shrunk** is a hard
error, because that is truncation rather than appending. Files that grew are
listed in `appended_during_archive` — normal for a training log, a red flag for
anything else. `collect_artifacts.py archive` now persists the rewritten
manifest, so the dev-box check compares against the archived bytes.

Regression test: `test_a_log_that_grows_between_manifest_and_archive_still_verifies`
reproduces the exact sequence and asserts the whole gate passes.

---

## 5. Two other defects, and where they were caught

**Before any pod existed — free.** RunPod's GraphQL edge returns **403
Forbidden** to the default `Python-urllib/3.x` User-Agent, on every query,
including ones that succeed byte-for-byte from curl. The launchers never hit it
because they shell out to curl; the hardened watchdog talks to the API directly,
so **every poll would have failed on a live pod**. Because `PodState.billing`
treats an unanswered poll as "still billing", the watchdog would still have
*tried* to terminate — but its verification poll could never have confirmed
success, so every session would have ended in `TERMINATION_FAILED` against a pod
that had actually died. Fixed by sending an explicit User-Agent; regression test
in `tests/infrastructure/test_provider.py`.

**During the run — a driver bug, not a control-plane finding.** `canary.py`
launched `watchdog.py --runpodctl <nonexistent>`; **`watchdog.py` had no such
flag**, argparse exited 2, and the test watchdog died before its first poll.
Nothing detected it, because the launch is fire-and-detach — which is the point
of detaching.

**Disclosure, because it bounds what criteria 5–8 prove.** The flag was added
and the test watchdog **relaunched by hand** against the same live pod, inside
the same session and the same budget. The evidence in §3 is genuine — a real
billing pod, a real threshold crossing, a real GraphQL mutation, a real
provider-polled disappearance — but **the driver did not achieve it unaided**.
A clean end-to-end pass has not been demonstrated. Regression tests now assert
that every flag `canary.py` passes to `watchdog.py` exists.

---

## 6. What this canary does and does not establish

**Established, on live infrastructure:** detached launch returns in seconds
while the job runs on; the descriptor survives and is readable from a later
connection; the relay makes events durable off-pod while the job is still
writing; a provider-only watchdog observes a billing pod, crosses its own
threshold, terminates, and confirms disappearance by polling; the GraphQL
fallback works; a forced primary-path failure does not strand a pod.

**Not established:** long-session behaviour. An 11-minute pod does not exercise
what produced E6b's 434-minute block. `--terminate-after` was never reached and
is still unobserved. And the artifact path has been fixed but **not re-verified
on a live pod**.

---

## 7. Consequence

Per the authorization — *"if any required criterion fails, do not launch E7 and
do not spend beyond the canary hard backstop"* — **E7 is not launched.** No B or
C arm has been started.

Both defects are fixed at zero cost with regression tests. Demonstrating a clean
end-to-end canary would need a re-run (~$0.05, same shape), which is a separate
authorization and is **not** being requested here as a foregone conclusion — the
maintainer may reasonably judge the evidence above sufficient for criteria 1–8
and the fixed artifact path adequately covered by tests.

```
authorized temporary cumulative cap: $150.41
actual cumulative spend after canary: $149.635   ($149.59 + $0.045)
remaining under the temporary cap:      $0.775
```
