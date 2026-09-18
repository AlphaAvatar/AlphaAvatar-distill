## 2026-09-19 — Reconstructed Top-5 leaves go to the dev-box store, not the hub

- **Context:** the authorized replay-only session reconstructs attempt 3's five
  selected leaves (1,192,135,096 bytes each, 5.55 GiB total) and must
  "persist and destination-verify each successfully reconstructed leaf
  immediately". The project's existing leaf transport,
  `scripts/autoinit/publish_selected_leaves.py`, uploads to the private hub repo
  `AlphaAvatar/aadistill-transport`.

  AGENTS.md requires confirming the backend has room *before* the run: "A
  durability mechanism needs a backend with the capacity to hold what it is
  protecting." C1 attempt 18 exercised a correct durability mechanism on six
  2.22 GiB probes and preserved none of them, because every upload was refused
  for private storage quota.

- **Measurement (2026-09-19, $0, no bytes transferred, nothing stored).** The
  quota is enforced at the LFS batch endpoint, so it can be asked directly:

      POST https://huggingface.co/<repo>.git/info/lfs/objects/batch
      {"operation":"upload","objects":[{"oid":…,"size":…}]}

  Fresh random oids were used so a dedup hit could not produce a false pass.

  | request | result |
  | --- | --- |
  | 1 leaf (1.11 GiB) | `200`, `actions: [upload]` — **granted** |
  | 5 leaves (5.55 GiB) | `403 Private repository storage limit reached` |
  | 30 × leaf (33.31 GiB) | `403`, same message |

  Bisecting a single object between 0 and 8 GiB: largest admitted
  **1,885,609,984 bytes (1.756 GiB)**; smallest refused 1.756 GiB. Account
  occupancy at the time was 93.08 GiB across 6 repos, 91.92 GiB of it private
  (85.26 GiB in `aadistill-artifacts`, 6.66 GiB in `aadistill-transport`).

  **One leaf fits. The second would be refused.**

- **Decision:** the five reconstructed leaves are persisted to the established
  out-of-tree store on the development host,
  `/home/ecs-user/aad-artifacts/phase_c2_full_search/attempt3_replay/leaves/<state_id>/`,
  pulled from the pod immediately as each leaf completes and verified at the
  destination with `verify_transferred_leaf()` before the next path starts. The
  hub and Git carry identities, digests and locations only — which is what
  AGENTS.md prescribes for heavy bytes, and all that would fit regardless.

  The store already holds 47 GiB of project artifacts and `/` has 55 GiB free,
  so 5.55 GiB is unremarkable for it.

- **Alternatives considered:**
  * *Upload to the hub anyway* — measured to fail on leaf 2. This is the
    attempt-18 failure repeated with prior knowledge.
  * *Split: one leaf to the hub, four to the store* — incoherent provenance for
    one selection, and consumes the last of the headroom for no durability gain.
  * *Purge hub LFS objects to make room* — `permanently_delete_lfs_files` on an
    exact `file_oid` does reclaim space and is proven in this project, but
    AGENTS.md is explicit that deleting historical objects or buying storage is
    a maintainer decision, never an autonomous repair. Not done.

- **Expected upside:** the replay's durability requirement is satisfiable at
  all, and satisfiable immediately per leaf, so a failure at leaf N cannot
  destroy leaves 1..N-1 — the property the whole session exists to restore.
  Transfer cost is small: the dev-box downlink measured 11.5 MB/s
  (209,715,200 bytes in 18.28 s), so 5.55 GiB moves in roughly 9 minutes of pod
  time against roughly 118 minutes of compute.

- **Risks:** the dev-box store is a single host and is not replicated. That is
  weaker than a hub repo and is stated plainly rather than implied: it protects
  against the failure that actually occurred (pod teardown destroying the only
  copy), not against loss of the development host. The hub's private tier is
  effectively full, which will block the next large-artifact session too, and is
  referred to the maintainer as a storage decision rather than repaired here.

- **Revisit when:** hub private storage is purged or upgraded, a second durable
  backend is available, or a session needs to preserve artifacts larger than the
  development host can hold.
