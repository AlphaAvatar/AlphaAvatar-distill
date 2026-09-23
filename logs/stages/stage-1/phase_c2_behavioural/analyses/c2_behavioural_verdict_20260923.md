# 2026-09-23 — C2 behavioural selection: NO_GO

- **Verdict:**

  ```text
  terminal_state   NO_GO
  delta            -0.008235294117647058   (exactly -7/850)
  lcb              -0.016862745098039214   (-14.33/850)
  SESOI            +0.010
  bootstrap_seed   834816710               (pre-registered)
  ```

  The advanced candidate `1a2b5b030e7e4e202fda3a810ed53c5f` does **not**
  displace incumbent **B**. It is *worse* by 0.82 points of `correct_overall`
  across the three frozen confirmation seeds, and the lower confidence bound is
  nowhere near the +0.010 SESOI.

  This is not a marginal call. GO was already arithmetically unreachable after
  the first two seeds (−0.00353 each); the third did not change the direction.

  **NO_GO is a complete terminal result of the frozen rule**, not a failure of
  the experiment. C2 names no new incumbent. B stands. C3 is separately
  authorized and unreachable from here.

- **What was executed:** twelve probes — six screening, one mechanical ranking,
  six confirmation — then one verdict. All five driver stages passed.
  `probes_trained 12`, `probes_scored 12`. The plan hash
  `31088b981499777cede9822adf6d011494f2de7fec29e59906eea147a8c9d705` never
  moved across thirteen launch chains.

- **How the last probe pair was obtained.** attempt12 trained the eleventh
  probe, announced it durable, and then failed in `attest` before scoring it.
  Its 2.22 GiB were fetched off-pod and re-identified from the delivered bytes
  before the pod was deleted, so 62 minutes of formal training outlived the
  failure of a later stage. attempt13 restored it from the network volume,
  re-identified it at the new local path, and scored it with `score_existing`
  in 28 minutes — **it was not retrained**, its initialization was not rebuilt,
  and it was priced as score-only. That is R2's mechanism and R3's prohibition,
  both pre-registered before any candidate existed, executed as written.

---

## The reproducibility limitation, stated plainly

**Probe 11's 950 per-sample rows and its `result.json` were never collected.**

Eleven of twelve probes hold complete evidence — 950 rows and a result each.
The twelfth does not. Its `probe_record.json` at the durable destination is the
pre-scoring copy from attempt12 and carries no score.

`secure_probe_evidence` iterated the session's **announced** durable units. A
probe is announced by the attempt that *trains* it, so a probe attempt13
*restored* and then *scored* never entered that list, and the evidence it
produced had no collector. The pod was then deleted by its own teardown gate,
correctly and automatically, and the rows went with it.

**Consequence.** The verdict was validly produced under the frozen rule and is
recorded above with its delta, its bound and its seed. It is **not
independently recomputable from the archive**, because one of the twelve
probes' rows are absent. AGENTS.md P4 says a result that cannot be reproduced
from its logged state is not a valid result; that sentence bears on the
recomputation, not on the execution, and the distinction is the maintainer's to
weigh rather than mine to dissolve.

The gap is visible in the bookkeeping: `campaign_state` reads the durable
destination and therefore reports **11 complete and 1 trained-but-unscored**,
while the pod scored twelve and the verdict consumed twelve. The archive cannot
show that probe 11 was scored.

**Repaired so it cannot recur.** `probes_owed_evidence` unions the announced
units with the probes a session restored, and a restored probe's evidence lands
beside the copy it was restored from, at its own `source_attempt`, so one
directory describes one probe.

**Not acted on beyond that.** Nothing was re-run. The campaign has a complete
valid verdict and `what_a_resume_may_never_do` forbids continuing after one.
Re-scoring probe 11 from its preserved, re-identified weights would restore
archive-reproducibility and is technically available — the checkpoint and its
descriptor are intact — but it would be a **new measurement taken after a
terminal result**. That is a maintainer's decision, not an engineering repair,
and it is left open rather than taken.

---

## Money

```text
formal behavioural campaign   $28.5891 of $42.0000   (headroom $13.4109)
  attempt3   $2.5425    attempt10  $0.0915
  attempt5  $19.7041    attempt11  $1.2180
  attempt9   $0.1146    attempt12  $2.2614
                        attempt13  $2.6570
  attempts 1, 2, 4, 6, 7, 8: $0.0000

$25 remaining-C2 stage envelope   $7.1647 spent   (unused $17.8353)
  measured as the provider balance delta across the whole window,
  $80.0378314134 -> $72.8731645539; $7.1216 attributable to named
  attempts and staging rounds, $0.0431 to volume storage and
  container-disk rounding no per-attempt derivation captures

project cumulative   $339.7876 of $370.0000   (remaining $30.2124)
```

## Provider state

Zero pods, zero network volumes. Network volume `59qt99zeg5` was deleted at
closeout after verifying every probe it held had its original on the launcher
host, so nothing unique was destroyed.

## Thirteen chains, and what each cost

Four were consumed at `$0` before any provider contact, five on real pods
before any science, and the last two carried the science:

| chain | ended at | cost |
| --- | --- | --- |
| 6, 7 | defects in the `$0` dry-run mechanism itself | `$0` |
| 8 | eight create calls refused: pinned to one datacenter for a volume nothing read | `$0` |
| 9 | the pod's own test gate: 27 cases read a host store no pod has, and the sweep could not see it | `$0.1146` |
| 10 | `scp` does not create its target's parent, and the test fake did | `$0.0915` |
| 11 | the storage bound asked a host-only manifest for a parameter count | `$1.2180` |
| 12 | `attest` took the first journal entry, which is now evidence-only — **trained probe 11** | `$2.2614` |
| 13 | **NO_GO** | `$2.6570` |

Every one of those five pod failures was a consequence of the consumer-derived
working set (P8.4) meeting a consumer nobody had enumerated. The rule was
right and cheap; tracing it to every reader was neither.
