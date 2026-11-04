---
name: aadistill-gpu-experiments
description: How AlphaAvatar-distill runs paid GPU work on RunPod -- choosing the card and the datacenter, storage and data identity, resuming search, training and generation, preserving expensive artifacts, acquisition windows and budgets, autonomous repair and stop conditions, validation cadence and where records go. Use before planning, pricing, launching, retrying or closing out any pod session, and before creating or deleting provider storage.
metadata:
  owner: maintainer
  source: maintainer 2026-11-04 ('GPU/RunPod 操作知识更新、Pilot 跨数据中心调度与恢复修正', section 5); AGENTS.md P8.2-P8.4, P12.1, P12.3, 2.5
---

# GPU experiments on RunPod

AGENTS.md is the contract; this skill says how its clauses are executed here, and names the code that executes
them. The `runpodctl` skill (vendored from `runpod/runpod-plugins-official`, see `skills-lock.json`) is the
provider's own reference; where it and this project differ, this project's records and code win.

## 1. Card and datacenter are separate constraints

- **The card** is decided by measured memory and cost (AGENTS.md P8.2; the L40S 48 GB by default). An A100 needs a
  measured L40S out-of-memory that survives protocol-neutral memory management, for that operation alone, under its
  own authorization. **No L40S stock is never a reason to change the card**: it is a capacity problem, solved by
  waiting or by more datacenters.
- **The datacenter** is a draw constraint only. Without a network volume a session may be drawn in any datacenter
  that meets its runtime, price and host requirements: `--data-center-ids A,B` lets the provider choose in one
  create, `A,B;C` rotates groups across attempts (`SessionRunner.placement`); none means any. A network volume pins
  the datacenter (a volume attaches only in its own), so **choose the card before the volume** and do not create a
  volume a session does not read.
- Stock (`runpodctl gpu list`, GraphQL `dataCenters.gpuAvailability`, `tool_pilot placement`) is a hint. The
  create's own answer decides, and it is classified (`SessionRunner.create`): an id; the provider's capacity refusal
  (`CAPACITY_REFUSAL`, retried on the window); or unknown -- a timeout, a transport error, `create_uncertain`, an
  answer with no id. "No pod id parsed" never means "no pod":
  - every create request carries a UNIQUE name (`pod_name(request)`) and its start time, recorded before the call;
  - an unknown answer is settled by reading the account over a window (`reconcile_create`,
    `--reconcile-window-seconds`, default 180): any pod of this session's requests is owned at once; nothing counts
    as uncreated until two listings read empty and the last one was read; until then no other create is made;
  - before each further create the account is read again: a pod an earlier request made late is owned, and no
    second create is sent;
  - every pod of this session's requests is registered: one is owned, each further one is released with the
    provider's confirmation and billed from its own request's start (`release_extra`); an unconfirmed release tears
    down and stops.
- Right after a pod exists, its REST v2 record is read (`RunPodProvider.pod_record`) and kept in the session record
  (`pod_record`: ``dataCenterId``, ``createdAt``, cost): the datacenter a provider-chosen draw landed in is recorded.

## 2. Content identity is not location

- An artifact is its identity and content hash, wherever it lives. Moving a session to another datacenter never
  re-initializes, re-generates or re-trains what already exists with a valid identity: copy or re-download the
  identical bytes and verify the hash. Copying identical content is not regeneration.
- Move bytes only for a consumer that reads them (P8.4): print the transfer preflight
  (`N objects · X GiB required · skipped · avoided`), materialize the minimum working set.
- Never attach, overwrite or use for convenience a volume holding an irreplaceable artifact (today `4rhubprswo`, the
  v2 Top-1's only copy). Storage created for an experiment is registered with its purpose, cost and retention in the
  existing records (`logs/stages/<stage>/<experiment>/storage/*.json`, which `mixed_governance.volume_records`
  reads); the live position refuses any volume no record owns.
- Global Volumes (beta) are not POSIX: no locking, no atomic rename, last write wins. Never put atomic sealing,
  locks or checkpoint writes on one without a measured equivalence. S3 access exists for network volumes in some
  datacenters only (not EUR-IS-2); check before planning a transfer through it.

## 3. Resume what finished; redo nothing

| work | what is restored | what runs |
| --- | --- | --- |
| structural search | the completed candidates and the search state (beam, journal, statistics caches) | the remaining rounds |
| training | the checkpoint with optimizer, scheduler and RNG state and the data position | the remaining steps |
| teacher generation | every sealed shard (task, mode, candidate), verified by its marker and launch identity | the unsealed tasks only |

A session on a network volume resumes from the volume. A session on the container disk resumes from the dev box:
the poll pulls each sealed shard as it is sealed and writes its RECEIPT (`token_matched_phase.receipt`:
`receipts.json`, the shard's sha256 and when it was received and verified); the replacement pod receives that copy
(`--resume-from`, `tool_session.bring_sealed`); only receipted shards count as done (`tool_pilot.resume_problems`),
and `resumable` / `generation_driver.sealed` re-verify identity, round input, protocol and runner. A shard from
another plan, protocol or implementation refuses; nothing sealed is overwritten. Never select or discard sealed
results by their outcome.

**Reuse across runs is an explicit import, never a resample.** Episodes an earlier run sealed under another identity
(the tool pilot's) enter a formal generation only through an import record (`tool_generation_import`): every
candidate's shard, sha256, original identity, candidate index and verdict, after checking the same input (each
task's sha256), teacher lineage and template, generation code, execution terms and seeds. An imported task is DONE in
its mode -- both candidates spent, never attempted again; top-up rounds take NEW tasks of the frozen order. The plan
binds the import record and the measured priors by sha256 and carries each cell's sealed state.

## 4. Save expensive work at completion; retire only dead weights

- A finished checkpoint, cache or rollout set is persisted with its identity and hash when it completes, not at
  closeout (P8.2.1). Confirm the durable backend has room first.
- Historical weights are retired only under AGENTS.md 2.5's standing rule: terminal, valid evidence durable, hashes
  kept, no consumer, not an anchor; keep the manifest and a tombstone.

## 5. Windows and budgets

- Three different budgets: the **acquisition window** at $0 (`--create-attempts` x `--create-retry-seconds`); the
  **billed run window** from the resource's start to the plan's hard deadline, enforced by the **independent
  watchdog** (`scripts/shared/pod/watchdog.py`, started the moment an id exists); and the **cumulative cost cap**,
  which no window extends. The start is the provider's verifiable ``createdAt`` (`pod_record`) when it reads one,
  else the request's own start -- never the later moment the answer arrived; cost, deadline and watchdog all count
  from it (`SessionRunner.own`). Queue time never shortens a run. `--terminate-after` was never enforced by RunPod
  and runpodctl >= 2.12 has no such flag; the runner passes it only to a CLI that takes it (`provider_cli_facts`),
  and never relies on it.
- A large input reaches the pod by the relay at hub speed, staged at $0 before the launch (`RelayInput` with its
  sha256; the bounded generation's task file, `token_matched_phase.stage_tasks`), never pushed from the dev box
  (0.7 MB/s) while a GPU bills.
- Each attempt is a new run with a new one-use artifact; its cap is what the cumulative cap leaves after every
  earlier attempt's closeout (e.g. `token_matched_phase.kind_cap_problems` for the tool pilot). Never start a run
  that cannot reach a saved, valid result inside what is left.
- At most one billing compute resource. Unknown ownership or billing state: create nothing, reconcile first.

## 6. Hardware and validation cadence

- Pick CPU or GPU by the question (P8.2, P8.3): go straight to the smallest GPU check when the uncertainty is CUDA,
  memory, throughput or integration; do not manufacture a CPU canary.
- Finish the whole round, then one convergence validation, then fix only the failures (P8.3). No repository suite
  on a paid pod; the paid pod runs the experiment's own preflight.

## 7. Repair and stop

- Ordinary failures inside an authorized envelope are repaired and retried without asking (P12.1): preserve
  evidence, tear down and reconcile, diagnose, minimal fix, minimal regression, fresh run identity, continue. A
  capacity refusal is not a failure to diagnose; a deterministic failure is never retried unchanged
  (`same_failure_gate`).
- Stop and report only for: the budget, frozen science, a hardware or rate boundary, unresolvable resource or
  billing state, or a complete valid result. Close every observer in the same terminal path (P12.3).

## 8. The provider's API and CLI

- REST v1 (`rest.runpod.io/v1`) retires 2026-11-15; GraphQL in early 2027. Billing reads REST v2
  (`provider_billing.fetch`: `/v2/billing/pods`, `/v2/billing/network-volumes`; v2 counts CPU pods and bills
  volumes per volume id; `provider_billing compare-v1-v2` reconciles one window item by item). runpodctl >= 3.0 is
  REST v2 throughout; its legacy `remove pod` and `pod delete` call the same v2 delete. Each session records the
  CLI's version (`session.json :: provider_cli`).
- GraphQL calls still on the active launch path, to move to REST v2 in one scoped change before GraphQL's
  retirement (no fixed date yet; target: before 2026-12-31, after the bounded tool generation, with recorded v2
  responses as test fixtures): `RunPodProvider.account_balance` (`myself`), `.account_resources` (`myself.pods` /
  `networkVolumes` -> `GET /v2/pods`, `/v2/network-volumes`), `.get` (`pod` -> `GET /v2/pods/{id}`), the
  `podTerminate` fallback (-> `DELETE /v2/pods/{id}`); `SessionRunner.check_gpu_offered` (`gpuTypes` ->
  `/v2/catalog/gpus`), `.wait_endpoint` and `.read_image_digest` (`pod` -> `GET /v2/pods/{id}`); `tool_pilot.placement`
  (`dataCenters` -> `/v2/catalog/datacenters`). Historical launchers are not migrated.
- Upgrading the CLI or the skill is provider support, never a change of the frozen experiment environment (vLLM,
  PyTorch, teacher revision, tokenizer, chat template stay pinned by the experiment's identity). Upgrade
  deliberately between sessions, never during one.

## 9. Records

Logs stay `logs/stages/stage-<n>/<experiment>/runs/<run_id>/{governance,runtime,evidence,closeout}` with the stage
and experiment READMEs maintained (AGENTS.md 3.2; regenerate the indexes after a new run). Use the field's
terminology (`aadistill-terminology`, AGENTS.md 2.10). One fact, one owner.
