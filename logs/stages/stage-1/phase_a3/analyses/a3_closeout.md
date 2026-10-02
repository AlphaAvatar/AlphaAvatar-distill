# A3 closeout — the ATTENTION calibration batching protocol, end to end

**Terminal 2026-10-03.** Run `a3_attempt38`, comparison
`488f9dd2c81b1107`, design `8d390eed92c5`.

Two owners, and this document restates rather than derives:
[`a3_comparison.json`](a3_comparison.json) owns the behavioural figures and
was computed **off pod at `$0`**;
[`a3_diagnostics.json`](../runs/a3_attempt38/evidence/a3_diagnostics.json) owns
the structural and runtime figures and was written on a pod by stage C.

**The structural half was measured three times, on three different pods.**
`a3_attempt34`, `a3_attempt35` and `a3_attempt36` each ran it end to end and
each wrote its own diagnostics; attempts 37 and 38 resumed at scoring and carry
attempt35's file byte-identically, which
[`a3_cited_evidence.json`](../runs/a3_attempt38/evidence/a3_cited_evidence.json)
attributes rather than reprints. So every structural figure below is a
three-session replicate, which is more than the design asked for and is the
reason the finding can be stated plainly.

A3 asked one practical question: does running the incumbent
`attention.activation_importance_v1` at `calibration_forward_batch_size=3`
with `calibration_batch_packing=length_sorted_v1` build the same artifact as
the bsz=1 incumbent, and if not, does what it *does* build recover to the
same behaviour?

---

## 1. The structural answer: a distinct materialization, not a distinct operator

```text
A_bsz1   53e30566c5f7   == the frozen C3 incumbent
A_bsz3   7dd2f6f6980b   DIFFERENT
         -> DISTINCT_NUMERICAL_MATERIALIZATION_PROTOCOL
```

* **Both digests are identical on all three pods**, and each protocol also
  reproduces its own digest across the four interleaved rounds *within* each
  session. So the difference between them is deterministic in the strong sense:
  reproducible within a session and across machines.
* **A-bsz1 rebuilt `53e30566c5f7` on each of the three pods**, which is also an
  independent reproduction of the frozen C3 incumbent identity on fresh
  hardware — and it is what makes attempt75's controls describe what this
  session built.
* **`result_spec_hash` is identical.** Same operator, same hashed config, same
  semantics — different bytes. That distinction is the whole finding.
* **The masking invariant holds** and the `$0` prediction matched observation
  exactly for both protocols:

  | | forwards | executed positions | valid | padded |
  | --- | --- | --- | --- | --- |
  | A-bsz1 | 67 | 59,830 | 59,830 | 0 |
  | A-bsz3 | 23 | 62,228 | 59,830 | 2,398 |

* **Head scores barely move** across all `896` heads: rank correlation
  `0.9999997`, max relative drift `0.0034`, median `0.0002` — the same figures
  to seven decimal places on all three pods.
* **Exactly one kept-head slot of 448 differs** (0.22%), in 1 of 28 layers:
  layer 7, where bsz1 keeps head 13 and bsz3 keeps head 12, and the margin is
  `0.00047` — a near-tie. Same slot, same layer, all three times.

## 2. The runtime answer: bsz=3 is SLOWER, on every pod that measured it

Scorer seconds, mean of 3 timed rounds per session with the warm-up excluded.
The scorer is the statistics pass alone, CUDA-synchronized at both ends — the
only part of the suffix a batching protocol touches.

| session | A_bsz1 | A_bsz3 | bsz3 slower by |
| --- | --- | --- | --- |
| `a3_attempt34` | 2.8073 | 3.0807 | **9.7%** |
| `a3_attempt35` | 2.8548 | 3.0892 | **8.2%** |
| `a3_attempt36` | 2.7868 | 3.0647 | **10.0%** |

**Mean 9.3%, range 8.2–10.0%, and the sign never flips.** Peak VRAM
`2.4480 → 3.1737 GiB` (+29.6%), identical on all three. Within-session spread
is tiny (`sd 0.0043` and `0.0030` on attempt35), so the between-session spread
is machine-to-machine variation, not measurement noise.

**And it is a modest slowdown in a small part of the work.** The scorer is
about a fifth of the ATTENTION suffix, and the suffix itself is essentially
unchanged: `12.861 s` vs `12.915 s` on attempt35, a `0.4%` difference well
inside each arm's own round-to-round spread (`sd 0.104` and `0.220`). Nothing
here would be worth adopting even if the sign were the other way.

**No runtime threshold was inherited**, by maintainer instruction. The `1.25x`
bar came from the causal-KL packing pilot, and the two workloads are not the
same question. Fitting `T = N·F + P·c` to each operator's two batch points
shows why:

| | per-invocation fixed cost `F` | per-position `c` |
| --- | --- | --- |
| `causal_kl_v1` | **+12.91 ms** | 36.24 µs |
| `activation_importance_v1`, attempt34 | **−3.45 ms** | 50.78 µs |
| `activation_importance_v1`, attempt35 | **−2.57 ms** | 50.59 µs |
| `activation_importance_v1`, attempt36 | **−3.56 ms** | 50.57 µs |

A negative fixed cost is impossible, which is the diagnostic: no positive `F`
fits this operator's data — on any of the three pods. Note also how tightly
`c` clusters (`50.57–50.78 µs`, a 0.4% spread) while `F` is nonsense in all
three fits: the per-position term is real and stable, the per-invocation term
is the model being wrong.

causal-KL does `60,099 = 67 × 897` forwards — one per item per head ablation —
and each carries a length-independent ablation setup worth amortizing, so
batching won it `1.1884x`. activation-importance does **67** forwards, one per
item, with nothing to amortize; batching only adds `2,398` padded positions
(+4.0% of real compute, which masking removes from the statistics but not from
the FLOPs) and widens every activation tensor.

*Caveat:* two batch points per operator is an exactly-determined fit, so `F`
and `c` are not independently identified within one session. The negative `F`
refutes the amortization model; drawing the real curve needs a third batch
size, not a third session.

## 3. The behavioural answer: no detectable correctness effect

Paired over the three frozen C3 seeds, `n_scorable = 850` each, controls
reused from attempt75 and **not retrained**.

| seed | control | treatment | Δ correct | McNemar b/d | Δ usable |
| --- | --- | --- | --- | --- | --- |
| 217230555 | 0.034118 (29) | 0.047059 (40) | **+0.012941** | 29 / 18 | +0.0811 |
| 1151307191 | 0.043529 (37) | 0.031765 (27) | **−0.011765** | 16 / 26 | −0.0094 |
| 2045359208 | 0.043529 (37) | 0.050588 (43) | **+0.007059** | 29 / 23 | +0.0674 |

```text
pooled correctness delta   +0.002745
descriptive 95% CI         [-0.006275, +0.012157]   one-sided LCB -0.005098
pooled usable delta        +0.046367
guardrails fired           none
```

**Mixed signs and a pooled delta of `+0.0027`** — an order of magnitude below
the `±0.022353` seed-level spread C3 measured on arms whose pooled
differences were within `0.001961` of zero. This is the signature of no
effect, which is what a single near-tie head-slot flip would predict.

The **usable-rollout** axis moved more (`+0.046367` pooled) and is reported
with every component rather than folded into an average:

| seed | non-empty | natural term. | no severe rep. | protocol valid | context limit |
| --- | --- | --- | --- | --- | --- |
| 217230555 | +0.0663 | +0.0705 | +0.0705 | +0.0832 | 0.0 |
| 1151307191 | −0.0390 | −0.0389 | −0.0389 | −0.0127 | 0.0 |
| 2045359208 | +0.1642 | +0.1726 | +0.1726 | +0.0789 | 0.0 |

No sample hit the context limit on either side. `usable_rollout` is blind to
correctness by construction and its components are not independent —
`protocol_valid` subsumes two of them — so the conjunction is not five
agreeing checks.

**Read the usable-rollout movement with the session caveat in front of it, not
behind it.** The three *treatment* non-empty rates cluster tightly
(`0.8526 / 0.7726 / 0.8463`) while the three *control* rates spread much wider
(`0.7863 / 0.8116 / 0.6821`), and the largest positive delta is the one whose
control is the outlier. That pattern is more consistent with between-session
variation on the control side than with a batching-protocol effect — and A3
cannot separate the two, because treatment and control were trained and
evaluated in different sessions. The correctness axis, which is the one the
design is about, shows no such movement in either direction.

**Per capability, the correctness signs are mixed there too** — which is what
no effect looks like when it is cut six ways:

| capability | Δ correct | Δ usable |
| --- | --- | --- |
| gsm8k | +0.0111 | +0.1356 |
| knowledge | +0.0089 | +0.0111 |
| math_verified | **−0.0089** | +0.0155 |
| multihop | 0.0000 | +0.0867 |
| rag | +0.0044 | +0.0200 |
| tool | 0.0000 | **−0.0233** |

Two capabilities are exactly unchanged, two move up by about the SESOI, one
moves down, and the largest correctness cell (`gsm8k +0.0111`) sits on a
control base rate of `0.0067` — a handful of correct answers either way. These
are descriptive slices of an 850-prompt battery, not powered subgroup tests,
and no capability-level claim is made from them. Per-domain and per-set axes are in
the comparison artifact; correctness is `null` for the `code` domain, where the
battery scores no correctness, and the usable axis is reported there instead.

**The comparability premise, verified from the records the pod wrote.** Each
probe's own admission record is required to assert comparability against
attempt75's controls under `generation_runtime_comparability@v2`, and all three
report the material identity **byte-equal on both sides**
(`f9d5bc543e49c451…`) with only the driver patch differing — `580.126.09` live
against `580.159.03` historical, the same branch, which the rule calls
provenance. That is the strongest form the statement can take: not "no
material field differed", but "the material identity is the same hash".
Serialized into the artifact under `admitted_against_the_controls`, because a
check whose result is not written down is not evidence that it ran.

## 4. What this evidence may NOT be used to claim

* **Not a non-inferiority result.** Three seeds cannot support a
  population-level claim and A3 claims none. The interval is **descriptive**:
  prompt-level uncertainty conditional on these three checkpoint pairs, with
  seeds as fixed blocks. The SESOI `0.01` is reported as the SCALE of a
  material loss; no threshold here decided anything. The withdrawn
  `−0.030` fail-fast is the measured seed-level noise band, never an
  equivalence margin.
* **Session is an unquantified alternative explanation.** Treatment and
  control were measured in different sessions on different physical hardware;
  C3 trained all nine of its probes in one session precisely to avoid that.
  What partly bounds it: A3 re-derived the incumbent initialization on the new
  hardware and gated it against `53e30566c5f7`, which confines the unverified
  session effect to training and evaluation.
* **A-bsz3 may NOT enter D1/D2/D3 execution** until the repository binds the
  numerical execution fingerprint to materialization/resume identity.
  `compute_state_id` binds neither the `ExecutionConfig` nor the artifact
  digest, so two differing artifacts would collide on one resumable,
  deduplicable state id. A passing behavioural result does **not** clear this.

## 5. Adoption

On this operator and this workload, the batching protocol **is not an
optimization**: it is 8–10% slower on the scorer, it produces a different
artifact, and it shows no detectable correctness effect. There is nothing to
adopt and nothing that needs rejecting on quality grounds.

**The cheap test the next operator should get, before any of this is paid
for.** Measure per-invocation scorer time against sequence length at batch 1
and read the intercept. A positive intercept means there is a
length-independent fixed cost per forward, and batching can amortize it — that
is causal-KL, with `60,099` forwards each carrying an ablation setup, and
length-sorted packing won it `1.1884x`. An intercept at zero means batching can
only add padded positions and widen tensors, which is what
activation-importance's `67` forwards gave us. That is a single-session,
single-GPU measurement. A3 reached the same conclusion across 34 pods and
`$13.46`, and the knob was an `ExecutionConfig` field the whole time.

## 6. Execution record

| | |
| --- | --- |
| terminal run | `a3_attempt38`, 78.9 min, `$1.43`, clean `DRIVER_EXITED:0` |
| probes | restored from `a3_attempt35` (AGENTS.md P8.4 state 2), not retrained |
| ladder | `B, C, F(restore), G, H` — D and E cited from the preserving attempt |
| structural | measured by `a3_attempt34`, `35` and `36`; identical all three |
| A3 total | **`$13.4600`** across 34 pods, every one provider-confirmed deleted |
| provider | 0 pods, 0 network volumes |
| formal remaining | `$7.2431` of `$76.6523` |
| GPU engineering | `$1.9395` of `$10.0000` (untouched by A3) |
| package remaining | `$9.1826` of `$86.6523` |
| project remaining | `$13.1777` of `$410.0000` |

**`$2.7400` of that total was one missing dictionary entry, rediscovered 21
times.** The launcher did not put `SESSION_FROZEN_EXPECT` in the pod
environment and the shared setup script requires it with `${VAR:?}`, so each
attempt created a provider resource and died during setup. The maintainer's
round-4 judgement is that this violated the autonomous-repair workflow, and the
two mechanisms that now exist are deliberately small: a contract regression that
parses the shell's own `${VAR:?}` requirements and checks every session launcher
supplies them, and `same_failure_gate`, which refuses a paid retry of an
identical deterministic failure signature without a corrective change.
Transients — capacity, cold host, unreachable — are excluded by design so the
existing backoff policy still applies to them.

Per-attempt costs and causes are in each run's `closeout/outcome.json`; the
spend narrative is in [`logs/budget/ledger.md`](../../../../budget/ledger.md).

**One recorded gap, left open deliberately.** A3's launcher calls `open_run`
and never `record_run`, so none of its 38 runs wrote a `manifest.json` and the
run index classifies every one as *unrecorded*. C3's launcher does this in
`close_c1_run`; A3's equivalent was never written. Nothing downstream depends
on it — `derive_budget` reads each attempt's own closeout, and the comparison
artifact binds every file it consumed by `sha256` — so reproducibility is
intact and the index states the gap rather than hiding it. It is not repaired
here because A3 is terminal and the launcher will not run again; a future
session launcher should close its runs the way C3's does.

**Checkpoint retention.** The three A-bsz3 probes remain durable at
`a3_preserved_probes/a3_attempt35/` with per-file digests. They are now
*completed and validly scored* — AGENTS.md P8.4 state 1 — so their weights
are archival evidence rather than an execution dependency, and the standing
retirement policy applies once no declared consumer needs the bytes. The
comparison consumed per-sample rows and scored records, never weights.

**Authorizes nothing.** No release, no promotion, no D-series execution.
