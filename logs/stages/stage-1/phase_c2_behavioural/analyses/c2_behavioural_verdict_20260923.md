# 2026-09-23 — C2 behavioural selection: NO_GO

> ## C2 CLOSED WITHOUT PROMOTION — maintainer decision, 2026-09-24
>
> **This document must not be read as "C2 scientifically proved NO_GO."** The
> maintainer closed C2 without promotion and explicitly declined to elevate the
> NO_GO estimate into a canonical scientific result, because the confirmation
> field mixed evaluation-protocol identities.
>
> ```text
> new C2 incumbent            NONE
> accepted incumbent after C2 B = frozen C1 treatment
> why B stands                no valid C2 challenger displaced it
> ```
>
> B stands **by absence of valid promotion evidence**, not because a clean
> canonical NO_GO experiment ruled against the candidate.
>
> The full joint search and screening did produce a candidate. The behavioural
> confirmation did not produce a sufficiently clean, protocol-consistent result
> that could replace the accepted incumbent.
>
> The attempt13 and attempt14 observations below remain historical evidence and
> are deliberately left intact. Both carry the limitation that the final
> confirmation evidence does not form one uniform evaluation-protocol field.
> Further C2 scientific spend is not authorized; no re-evaluation, no uniform
> six-probe replay, no new seeds, no attempt15.
>
> Two corrections this document originally got wrong, kept visible rather than
> silently edited:
>
> 1. it explained NO_GO through the **LCB**, which is a *GO* criterion. The rule
>    produces NO_GO from `ucb_one_sided < SESOI`.
> 2. it attributed the attempt13/attempt14 difference to **greedy-decoding
>    nondeterminism**. That was asserted, not established. See
>    "The confound, established" below.

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
  delta are identical.

  Both figures stand. Neither is averaged with the other and neither is
  selected as more correct — see "The confound, established".

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

## The confound, established

The six confirmation probes do **not** form one uniform evaluation-protocol
field. Read off each probe's own `result.json`:

| attempt | arm | seed | `generation_protocol_fingerprint` |
| --- | --- | --- | --- |
| 5 | candidate | 1916380711 | `e9d8da97…` |
| 5 | candidate | 1936324010 | `e9d8da97…` |
| 5 | B | 1916380711 | `e9d8da97…` |
| 5 | B | 1936324010 | `e9d8da97…` |
| 13 | **B** | **1523147638** | **`af7beb55…`** |
| 12 → 14 | **candidate** | **1523147638** | **`bbba93df…`** |

Battery content identity, scoring contract and metric contract are **uniform
across all six**. The generation protocol is not: three distinct fingerprints,
and the two seeds measured wholly within attempt5 are internally consistent
while **the third seed's pair spans two different protocols** — B under one,
the candidate under another.

So the confound is *inside the pair*, on the seed with the largest magnitude
(−0.02) and the one whose delta moved by two prompts between attempt13 and
attempt14. Protocol/runtime drift is therefore a live alternative explanation
for that difference, and it cannot be separated from any candidate-versus-B
effect at that seed using this evidence.

**What is not claimed.** That greedy-decoding nondeterminism caused the two
changed prompts. That was the first explanation offered here and it was
asserted rather than established. The runs did differ in NVIDIA driver patch
(`@580.173.02` for attempt13, `@580.126.20` for attempt14), and
`generation_compat` v2 deliberately treats a patch move *within one branch* as
recorded-not-material — so the driver alone does not establish incomparability
either. What is established is that the recorded generation protocol identities
differ and that the archive does not carry the expanded protocol and runtime
blocks needed to judge them under that rule. Unjudgeable is not comparable, and
it is not incomparable; it is unjudgeable, and that is why no canonical estimate
is claimed.

**Now enforced.** `behavioural_decision.assert_one_measurement_protocol`
refuses a confirmation field that cannot be shown to share one measurement
protocol, and `confirm` requires the protocol identities rather than accepting
them optionally. Run against this archive today it exits 3 and writes nothing:

```text
REFUSING: the confirmation field spans 3 distinct generation protocol
fingerprints, and 6 of 6 probes did not record the expanded protocol and
runtime blocks that `generation_compat` v2 needs to judge comparability.
Unjudgeable is refused, not assumed comparable.
```

C3 and C4 inherit the gate. It compares battery, scoring contract and metric
contract by identity, and the generation protocol through the project's own
comparability rule rather than by fingerprint equality — because that
fingerprint transitively contains the fused `imageName@driver` field, and exact
equality over it would make every comparison a host lottery.

## Money

```text
formal behavioural campaign   $30.6561 of $42.0000   (headroom $11.3439)
  attempt3   $2.5425    attempt10  $0.0915    attempt13  $2.6570
  attempt5  $19.7041    attempt11  $1.2180    attempt14  $2.0670
  attempt9   $0.1146    attempt12  $2.2614
  attempts 1, 2, 4, 6, 7, 8: $0.0000

$25 remaining-C2 stage envelope   $9.3042 spent   (unused $15.6958)
  RECONCILED TO THE PROVIDER, not to the sum of the parts. The account
  balance delta across the whole window is
  $80.0378314134 -> $70.7336584514 = $9.3042, and the ledger now sums
  to exactly that: attempts 9-14 $8.4095 + the staging campaign $0.8947.
  The staging figure includes $0.0431 of NETWORK-VOLUME STORAGE that no
  per-attempt derivation captures -- previously a silent hole, now booked
  to the campaign that owns the volumes.

project cumulative   $341.9702 of $370.0000   (remaining $28.0298)
  derived by `derive_budget.py`, not transcribed. $0.0431 higher than the
  figure quoted before the volume storage was attributed.
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
