# Phase-A searched-leaf durability. Measured 2026-08-15, before any authorization.

Resolves the item left open in
[`autoinit_relay_capacity.md`](autoinit_relay_capacity.md): *"Phase-A's relay
requirement is not established, and I decline to invent it."* It is established
here, from measurement, and the answer is that **Phase A needs no relay growth at
all**.

Settled before the harness digest is computed, because the retention rule is
implemented in the launcher and the driver — deciding it afterwards would
invalidate a digest already issued.

## What was measured

| quantity | value | how |
| --- | ---: | --- |
| relay `usedStorage` | **91.54 GiB** | `GET /api/models/{repo}?expand[]=usedStorage` → 98,287,134,179 B |
| relay LFS inventory sum | 92.03 GiB | `HfApi().repo_info(files_metadata=True)`, 1,143 files |
| inferred limit (100 GB decimal) | 93.13 GiB | **still inferred**, unchanged from the 2026-08-14 analysis |
| **headroom** | **1.60 GiB** | limit − usedStorage |
| one 596M bf16 initialization | 1.121 GiB | `stage1/qwen3_0p6b_init_v0/checkpoint` on the relay; corroborated by the 1.12 GiB `size_human` in `checkpoint_tombstones.json` |
| one 596M fp32 *trained* checkpoint | 2.231 GiB | `permanent_controls/preflight_ctl_r0860k_sa/model` |
| **five searched leaves** | **5.61 GiB** | 5 × 1.121 |

The relay grew 81.85 → 91.54 GiB since 2026-08-14, mostly the two permanent
controls (4.46 GiB) and `transfer/` (0.17 → 7.66 GiB of git bundles).

**Five leaves do not fit in 1.60 GiB. Neither do two (2.24 GiB). One barely
does.** Relay staging is therefore not the mechanism, and no amount of arranging
makes it one without deleting something.

## The retention rule

The requirement is that the five leaves stay recoverable through rung 1 and the
materialization of the sa survivor decision, and the two survivors then stay
durable through sb/sc. That is satisfied on the **pod's container disk**, which
is provisioned at 200 GB against a priced peak working set of ~106 GiB:

| phase | what is materialized | where | size |
| --- | --- | --- | ---: |
| search → rung 1 → sa selection | all 5 leaves + control | pod disk | 5.61 GiB |
| rung 2 → conditional rung 3 | 2 survivors + control | pod disk | 2.24 GiB |
| teardown | finalists only | dev box, by scp | ≤ 2.24 GiB |
| permanently | rejected leaves | **tombstone only** | 0 |

**This is a property of the frozen search code, not something added for it.**
`BeamSearch._release_weights` is called only for states pruned from the beam,
inside the `if partial:` branch; a complete leaf is appended to `self.leaves` and
its weights are never released (`keep_leaf_weights=True`). So every searched leaf
survives on disk from materialization to teardown without any retention logic at
all. Verified by reading the call site rather than asserted.

## What survives a rejected leaf

`audit/autoinit_phase_a/leaf_retention.json`, written the moment the rung-1
selection is materialized, in the shape `checkpoint_tombstones.json` already
uses — for **all five** leaves and the control:

* `artifact_digest` and `weights_sha256` — exactly what it was;
* `search_lineage` (`path_label`) — how it was built;
* `sa_probe_id`, `sa_result`, `sa_evaluation_protocol_hash` — the evidence;
* `selection_rule` and `rejected_reason` — why it did not advance;
* `retention_tier` and `permanent_checkpoint_retained`.

A rejected leaf loses its bytes and keeps its accountability. That is the
maintainer's stated allowance, and nothing broader is taken from it.

## What is NOT done here

* **`transfer/` is not the reclaim opportunity it was in August.** The
  2026-08-14 analysis recorded it as 0.17 GiB of superseded git bundles. It is
  now 7.66 GiB, and the growth is **not** bundles:

  | group | size | reclaimable? |
  | --- | ---: | --- |
  | `transfer/wheelhouse_cu128_cp312` | 3.821 GiB | **no** — the offline train environment |
  | `transfer/wheelhouse_vllm_cp312` | 3.620 GiB | **no** — the offline vLLM environment |
  | loose files (85, mostly `*.bundle`) | 0.219 GiB | mostly yes, ~0.19 GiB |

  The two wheelhouses are what keep PyPI off the paid setup's critical path —
  four of five host draws once died resolving and downloading from it — so
  deleting them would break every future pod and cost dev-box uplink to restore.
  **The genuinely reclaimable amount is ~0.19 GiB of superseded bundles, not
  7.66 GiB.** That does not change the conclusion, it removes a false escape
  hatch: even reclaiming all of it leaves ~1.79 GiB, still far short of 5.61.

* **Nothing is deleted.** Rejected leaves are not fetched; the pod is destroyed
  at teardown either way. No relay artifact, checkpoint, log or record is
  removed, and the experiment is not shrunk — it is still 5 searched leaves, the
  canonical control, sa → best 2 + control → sb, and the conditional sc.

* **The session bundle does fit.** `aad_autoinit_*.bundle` runs ~4 MiB against
  1.60 GiB of headroom, so staging one for a launch is not in question.
* **The 100 GB limit is still inferred.** Phase A no longer depends on it — the
  plan adds 0 GiB to the relay — so confirming it is no longer blocking. It
  would still be worth one look at the billing page before any future staging.

## The exposure this accepts, stated plainly

Off-pod durability during the run is **not** provided. If the pod is lost before
teardown, the searched leaves are lost with it, and because weight hashes are not
reproducible across sessions in this project, a relaunch re-searches rather than
resumes: the completed probes' journal entries would no longer bind to any
available checkpoint.

That is a **cost** exposure, not a scientific-integrity one — a relaunch is a
fresh authorization and a fresh search — and it is bounded by the session's own
hard threshold. The alternative costs 5.61 GiB of permanent, irreclaimable relay
quota that does not exist. Buying pod-loss insurance by deleting scientific
evidence would be the worse trade.
