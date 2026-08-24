# Recovery continuation attempt 7 — PHASE A COMPLETE, $12.8587

**Verdict: `ALL_DONE`. All six stages ran, all eleven probes trained and scored,
and the frozen selection produced a result.** The launcher then crashed *after*
collection on a contract mismatch, mislabelling the session `INCOMPLETE`. No
science, evidence or artifact was lost.

| | |
| --- | --- |
| authorization | `autoinit.recovery_continuation.2026-08-23T1314Z`, sha256 `d354557e…` |
| grant | `logs/autoinit_recovery_continuation_attempt7_grant.json` |
| base commit | `7e1d429` |
| session commit | `d968b20102bb8d40de54f840047a809c4536d3ca` |
| harness digest | `b824441c…`, 22 files, search excluded |
| bundle | `aad_autoinit_d968b201.bundle`, sha256 `c0d63dc6…` |
| pod | `3c1g6e01kdu1ya`, L40S $0.99/h, **779.32 min, $12.8587**, provider confirms gone |
| driver terminal | **`DRIVER_EXITED:0` / `ALL_DONE`** |
| session terminal | `INCOMPLETE` — see *The launcher defect* below |

## The result

```
decision_status   unresolved_equivalence
winner            None
winner_is_control False
tie_break_ran     True
```

Pooled over seeds, primary metric `correct_overall`, equivalence interval
**0.011695296982299022**, feasibility floor **0.3000**:

| candidate | correct_overall | usable_rollout_rate |
| --- | ---: | ---: |
| `cca699c93f34` | **0.029412** | 0.6561 |
| `85bde4ded2c3` | 0.019608 | 0.5456 |
| `control-qwen` (canonical init) | 0.008824 | 0.4947 |

Pairwise against the frozen interval:

| pair | Δ correct_overall | vs 0.011695 |
| --- | ---: | --- |
| `cca699c9` − `85bde4de` | 0.009804 | **inside** — tied, unresolved after seed sc |
| `cca699c9` − control | 0.020588 | **outside** — separated, leaf better |
| `85bde4de` − control | 0.010784 | **inside** — not separated |

**What this does and does not say.** The best searched initialization is
separated from the canonical one on the primary metric and is also ahead on the
behaviour axis (0.656 vs 0.495 usable rollout). The two best searched
initializations cannot be told apart from each other, even after the conditional
third seed. Per the frozen rules that is a **result**, not a condition to be
resolved: `unresolved_equivalence` stands and **no fourth seed follows**. Whether
the leaf-over-control separation is scientifically actionable is a maintainer
judgement; this record does not claim it as a win, because the plan's own winner
rule returned `None`.

The absolute correctness numbers are small in every arm, control included. That
is a property of the 0.86M-token recovery rung, not of the comparison.

## Comparability

* `comparable_identity` **`70a26e0b…`** — live equals historical, `identities_equal: true`
* `bound_to_stage3_thresholds: true`, `stage3_evaluation_protocol_hash 250f72ef…`
* the run's own `evaluation_protocol_hash` is `7327e880…`, which differs from
  `250f72ef` **only** in the driver patch (`580.178.04` vs `580.159.03`).
  `generation_runtime_comparability@v2` declares that non-material — the rule
  exists precisely so comparability is not a host lottery.

## Execution

```
13:46:25 STAGE_PASSED:0     attested 0.011695 / 0.3000 / 02be33b9
13:49:20 STAGE_PASSED:1     5 leaves re-identified from bytes; control measured
13:49:20 → 20:45:11  rung 1, seed sa: 6 probes trained and SCORED
20:45:11 → 00:11:19  rung 2, seed sb: 3 probes trained and SCORED
00:11:19 → 02:30:05  rung 3, seed sc: 2 tie-break probes trained and SCORED
02:30:05 STAGE_PASSED:5 · ALL_DONE
```

**Eleven probes, every one trained and scored.** Training time was remarkably
stable at **61.0–61.1 min** against the **61.55 min** the envelope is priced
from — so attempt 6's 71.9 min was host variance, not a systematic
underestimate. Batteries ran 6.4–10.3 min against 9.82 priced.

Both repairs carried into this run held on hardware: the device handoff freed to
`0.01 GiB` with `live_retention: false`, and the evaluation tokenizer was
materialized (`copied, copied, copied`) before every one of the eleven batteries.

## The launcher defect — post-collection, nothing lost

`session_runner.py`:

```python
"checkpoint_hashes_matched": all(f.get("rc") == 0 for f in fetched),
```

`fetched` is the return of `art.fetch_products(...)`, which for Phase A is
`finalists_to_fetch` — and that returns **a list of `canonical_id` strings**, not
transfer-result dicts. `f.get("rc")` therefore raises
`AttributeError: 'str' object has no attribute 'get'`. A contract mismatch
between two halves of the supported path, on a line only a **successful** Phase A
reaches: every previous attempt failed before `fetch_products` was ever called
with a non-empty list.

**It fired after everything was already retrieved**, and nothing was owed off-pod:

* all **9 reports** fetched, `local_hash_problems: []`;
* the artifact archive fetched and extracted — **all 11 probe trees**, configs,
  the eval outputs;
* `finalists_to_fetch` returns the two **initializations** that earn permanent
  retention, and both are already preserved canonically at 1.2 GiB each in
  `/home/ecs-user/aad-artifacts/autoinit/phase_a/` and mirrored in the transport
  repo. Verified present. Nothing needed transferring.

Consequences, all cosmetic: the session is recorded `passed: false` /
`INCOMPLETE` despite `DRIVER_EXITED:0`, `teardown_gate` is absent, and teardown
ran as an emergency delete rather than through the gate. The pod was deleted with
provider confirmation either way.

**This is a real defect and should be fixed before any future session** — a
successful Phase A currently cannot be recorded as successful. It is not fixed
here: nothing further is authorized.

## State

* pod deleted, **provider confirms gone**; watchdog ended `pod_gone` after 91
  ticks; poller stopped; provider returns zero pods; nothing billing;
* the five Attempt-12 leaves are untouched; permanent controls not retrained;
  frozen science untouched;
* `$230.0350` cumulative against the `$234.00` cap — **`$3.9650` remains**, which
  funds no further paid session of any kind.
