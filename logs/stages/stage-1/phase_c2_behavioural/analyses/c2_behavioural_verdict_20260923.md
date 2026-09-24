# 2026-09-23 — C2 behavioural selection: NO_GO

> **Superseded in two ways by the P4 repair of 2026-09-24.** This document
> explained NO_GO through the **LCB**, which is a *GO* criterion; the rule
> produces NO_GO from `ucb_one_sided < SESOI`. And probe 11's rows, absent when
> this was written, were reconstructed — giving a slightly different aggregate.
> The complete record is
> [`c2_behavioural_decision_recomputed.json`](c2_behavioural_decision_recomputed.json)
> and [`attempt14/closeout/outcome.json`](../runs/attempt14/closeout/outcome.json).
> What is *unchanged* is the terminal state and every qualitative conclusion below.

- **Verdict, as attempt13's runtime observed it:**

  ```text
  terminal_state   NO_GO
  delta            -0.008235294117647058   (exactly -7/850)
  lcb              -0.016862745098039214   (-14.33/850)
  SESOI            +0.010
  bootstrap_seed   834816710               (pre-registered)
  ```

  The runtime recorded only delta, lcb and the seed — a truncated projection of
  `decide()`'s output, which also returns `ucb_one_sided`. **The bound that
  actually produces NO_GO was therefore not in the record at all**, and citing
  the LCB for it was wrong:

  ```text
  GO    : lcb_one_sided > 0 AND delta >= SESOI AND >=2/3 seed deltas > 0
          AND no veto
  NO-GO : ucb_one_sided < SESOI OR a behavioural veto fires
  otherwise INCONCLUSIVE
  ```

  Reconstructed, the full criteria read:

  ```text
  terminal_state   NO_GO
  delta            -0.009019607843137253
  lcb_one_sided    -0.017647058823529408
  ucb_one_sided    -0.0003921568627450984   <  SESOI 0.010   -> NO_GO
  seed_robustness  0 positive of 2 required
  guardrails       passed · catastrophic_violations []
  ```

  So NO_GO comes purely from the interval excluding the smallest effect worth
  having. No veto fired.

  **The reconstruction differs from the runtime by 2 prompts of 850** on probe
  11: its seed delta is −17/850 where attempt13 measured −15/850, and the
  pooled delta −23/2550 against −21/2550. The terminal state, the criterion,
  the bootstrap seed, the guardrail outcome and the sign of every per-seed
  delta are identical. The likely cause is serving-engine non-determinism under
  greedy decoding — this repository's own Stage-4 notes record that decoding is
  not batch-invariant even within one stack, and the two runs used different
  physical L40S instances. It is recorded, not averaged.

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

## The reproducibility limitation — RESOLVED 2026-09-24

**Probe 11's 950 per-sample rows and its `result.json` were never collected**,
and for one day the archive could not recompute this verdict. The reviewer
withheld C2's final closure on exactly that ground. What follows is the gap
as it stood; the repair is at the end of the section.

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

**And then repaired, on the reviewer's authorization.** Re-scoring probe 11
from its preserved weights is a new measurement after a terminal result, which
is a maintainer's decision and not an engineering repair — so it was left open
here, and the reviewer authorized it as a reproduction-only P4 repair on
2026-09-24. attempt14 restored the same identity-verified checkpoint, re-ran
its evaluation under the frozen protocol, and the archive now recomputes the
verdict independently. `campaign_state` reports twelve complete probes.

The cost of that repair is honest about what it is: the rows in the archive are
the **reconstruction's**, and they differ from attempt13's by 2 prompts in 850.
See the banner at the top and
[`attempt14/closeout/outcome.json`](../runs/attempt14/closeout/outcome.json).

---

## Money

```text
formal behavioural campaign   $30.6561 of $42.0000   (headroom $11.3439)
  attempt3   $2.5425    attempt10  $0.0915    attempt13  $2.6570
  attempt5  $19.7041    attempt11  $1.2180    attempt14  $2.0670
  attempt9   $0.1146    attempt12  $2.2614
  attempts 1, 2, 4, 6, 7, 8: $0.0000

$25 remaining-C2 stage envelope   $9.3042 spent   (unused $15.6958)
  the provider balance delta across the whole window,
  $80.0378314134 -> $70.7336584514. Named attempts and staging rounds
  account for $9.2611; the rest is network-volume storage and
  container-disk rounding that no per-attempt derivation captures.

project cumulative   $341.9271 of $370.0000   (remaining $28.0729)
```

## Provider state

Zero pods, zero network volumes. `59qt99zeg5` was deleted at the first
closeout and `a0zqgxsm7p` — the 10 GB volume the P4 repair used — after it, in
both cases having verified that every probe they held had its original on the
launcher host, so nothing unique was destroyed.

## Fourteen chains, and what each cost

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
| 13 | **NO_GO** — and it trained B, scored the restored probe 11, and decided | `$2.6570` |
| 14 | **NO_GO reproduced** from the reconstructed evidence; the teardown gate blocked on a required artifact and the pod was removed by hand | `$2.0670` |

Every one of those five pod failures was a consequence of the consumer-derived
working set (P8.4) meeting a consumer nobody had enumerated. The rule was
right and cheap; tracing it to every reader was neither.
