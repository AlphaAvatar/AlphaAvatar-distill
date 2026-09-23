# 2026-09-23 — The durable backend is a provider network volume, not an object store

- **Context:**

  A Phase-C2 continuation must put this campaign's ten completed probes on the
  replacement pod and re-identify them there (resume rules R2 and R6). That is
  22.21 GiB. The bytes live on the launcher host, whose uplink measures
  0.49–0.91 MB/s — a **host** cap, not a per-connection one: three parallel
  streams measured ~1.3×, not 3×. Copying them during the session therefore
  charged roughly nine hours of L40S time to every attempt, which is how a
  continuation came to cost more than the science it was continuing.

  The previously approved direction was an S3-compatible object store: upload
  once from the host while nothing bills, then let each pod fetch with
  presigned URLs. The interface, the backends and the validator were built and
  tested (`src/aadistill/runtime/durable_store.py`,
  `scripts/experiments/durable_stores.py`, 34 passing tests).

  It was never run. Every such store reachable from this environment —
  Cloudflare R2, Backblaze B2, a second cloud account — requires account
  creation and a payment method, which only a maintainer can complete. The
  Hugging Face account that already exists has **2.62 GiB** of private
  headroom, bisected at `$0` through the LFS batch endpoint on 2026-09-23; one
  probe is 2.22 GiB, so it can hold exactly one of the ten. RunPod's own
  S3-compatible API needs an S3 key that is issued only through its console.

- **Decision:**

  Pre-stage the probes onto a **RunPod network volume** and attach it to every
  pod of the campaign.

  * volume `59qt99zeg5`, 40 GB, `EU-NL-1`, mounted at `/durable`;
  * written once by `scripts/autoinit/stage_c2_probes_to_volume.py` through a
    CPU pod at `$0.06/h`, with every file re-hashed from the bytes that landed;
  * attached by `SessionRunner.attached_volume()`, which is additive and
    optional: a launcher naming no volume produces the command line it always
    produced;
  * `restore_campaign_probes` becomes a presence-and-identity check instead of
    a transfer, and the driver's on-pod re-identification is unchanged.

- **Alternatives considered:**

  * **Object store (R2/B2/S3).** Blocked at the human-authentication boundary,
    not at the code. Kept, annotated as having no production caller.
  * **Hugging Face private repo.** 2.62 GiB of headroom against 22.21 GiB
    needed. Freeing ~20 GiB means permanently deleting historical LFS objects,
    some of which — `e1_scaling_20260801`, `stage3` — have no local copy and
    are the provenance of this campaign's own recovery recipe.
  * **Hugging Face public repo.** Free and large, but every relay repo this
    project uses is private, and publishing 22 GiB of intermediate checkpoints
    is an outward-facing, hard-to-reverse act that is a maintainer's to take.
  * **Keep copying during the session.** ~9 hours of L40S per attempt against a
    continuation whose science is 174 minutes.

- **Expected upside:**

  * the transfer leaves the experiment entirely. The reserve it is priced under
    fell from 90 minutes to 45 — and what those 45 minutes now buy is honest:
    re-reading and re-hashing 22.2 GiB, not moving it;
  * the continuation's hard all-in fell from `$9.8956` to `$9.0656`;
  * no credential of any kind reaches a pod. A volume is attached by the
    provider; there is nothing to leak, expire or rotate;
  * it removes a latent accounting hole. Restored probes used to land under
    `WORKDIR` on the container disk, and `container_gate`'s peak-residency
    model never counted them — 22.2 GiB unaccounted against a 120 GB
    provision, which happened to fit. On the volume they are not the container
    disk's problem at all.

- **Risks:**

  * **Datacenter concentration.** A volume lives in one datacenter and a pod
    can attach it only from there, so attempt6 must draw an L40S in `EU-NL-1`.
    Stock there is reported *Low*, as it is in `US-TX-4` and `US-MO-1` — the
    only other datacenters carrying the card at all. If `EU-NL-1` dries up the
    fallback is a second volume elsewhere and a re-stage, which costs hours of
    wall clock and about `$0.50`, and no new code.
  * **A volume is not a backup.** It is one provider's storage, in one region,
    holding the only copy that is not on the launcher host. The host keeps its
    copy; neither is deleted.
  * **`df` cannot see the quota.** The mount reports the backing cluster —
    165,732 GiB free on a 40 GB volume — so a free-space check built on it can
    never fail. Capacity is taken from the provisioned size instead, which is
    the correction ENOSPC taught this campaign once already.

- **Revisit when:**

  a campaign needs pods in more than one datacenter, or needs bytes on a
  machine this provider does not run, or a maintainer creates an object-store
  account. The generic interface is still there and still names no vendor.
