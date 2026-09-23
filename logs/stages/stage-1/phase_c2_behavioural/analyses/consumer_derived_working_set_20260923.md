# 2026-09-23 — Artifacts follow consumers, not campaigns

- **Context:**

  A continuation restored every probe its campaign held, because the probe
  existed and belonged to the campaign. For C2 that is ten completed probes and
  **22.21 GiB** of model weights.

  All ten are trained **and validly scored**, each with 950 per-sample rows.
  Nothing remaining reads their weights: `run_rung` skips a probe that has a
  score, and the verdict consumes rows, scores, descriptors, seeds, battery
  identities and hashes. The weights were being moved as a **preregistered
  admission token**, not as a computation input — R2's
  `_a_record_is_not_a_checkpoint` clause, which asks that reuse re-derive
  identity from bytes on the consuming pod.

  That cost was invisible while it was ~9 hours of L40S per attempt and looked
  like the price of the rule. It became visible once a network volume made the
  transfer cheap: sixteen hours of unattended wall clock, four pod draws, two
  stall aborts and `$0.7148` — to satisfy no consumer.

- **Decision:**

  The maintainer adopted a standing rule, now **AGENTS.md P8.4**: never move a
  large artifact onto an execution resource merely because it exists or belongs
  to the same campaign. Before any substantial transfer, classify each artifact
  by lifecycle state and ask what exact downstream operation will read the
  bytes. If there is none, the bytes do not move.

  Applied to probes:

  | state | on a continuation resource | weights |
  | --- | --- | --- |
  | completed + validly scored | evidence only | **not restored** |
  | trained + not validly scored | weights + descriptor + identity | restored, resumed at scoring, never retrained |
  | not trained | nothing to restore | initialization materialized, trained normally |
  | invalid measurement | decided by whether the checkpoint is still valid | not inferred from the bytes existing |

  For this campaign the working set is **8.1 MiB against 22.21 GiB — 0.035% of
  the bytes.**

- **What still binds a probe admitted without its weights:**

  * its descriptor — rung, arm, seed, initialization digest, config hash — is
    checked against the schedule's own by `assert_reuse_matches`;
  * the rows the verdict reads are hash-checked against the hashes the score
    record itself carries, by `_restore_score_evidence`;
  * `campaign_state` still admits only probes whose `durable_ack.json` records
    a destination-side re-identification that matched.

  What is **not** re-derived is the artifact identity of a checkpoint nothing
  will execute. That is the narrowing, it is the maintainer's, and it is
  recorded here rather than buried: a completed and scored checkpoint is
  archival evidence, not an execution dependency.

- **Alternatives considered:**

  * **Finish the transfer, then optimize.** Rejected: paying for bytes with no
    consumer is an engineering defect whether or not the payment has started.
  * **Undo the eight already transferred.** Rejected by the maintainer as sunk
    operational cost. They may remain until teardown and are explicitly **not**
    a required input by virtue of being present — `volume_gate` asks only for
    probes whose weights a remaining operation reads, and this campaign has
    none.

- **Expected upside:**

  * the continuation's hard all-in fell from `$9.0656` to `$8.4201`, and its
    availability reserve from 45 minutes to 10;
  * fourteen remaining hours of transfer, two pod draws and ~`$0.5` avoided;
  * the rule is generic. Every later stage derives its working set from
    consumers, and a `transfer_plan` preflight records artifact, state, size,
    destination, consumer, why and when before anything moves.

- **Risks:**

  * **A future operation may want those weights.** Then it restores them: the
    volume keeps the eight until C2 closeout, and the durable store keeps all
    ten. Preservation is unchanged; only default transport moved.
  * **The lifecycle classification is now load-bearing.** A probe misclassified
    as completed-and-scored would have its weights withheld from a scorer that
    needs them. `campaign_state` derives `scored` from the score record AND the
    presence of its rows, an unscored probe is refused admission as evidence at
    two separate points, and both refusals are mutation-tested.

- **Revisit when:**

  a stage has an authorized operation that genuinely consumes a completed
  probe's weights — a re-scoring under a changed scorer, a distillation from a
  probe, a diagnostic that loads one. The state machine already has a route for
  that; it is the explicit-authorization branch.
