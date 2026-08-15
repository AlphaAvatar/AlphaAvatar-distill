# Handoff — Teacher-Adaptive AutoInitializer

**Self-contained.** Read this and you do not need any conversation history. Written
2026-08-12 at the close of E8, for the session that implements AutoInitializer.

---

## 1. Objective

Compress an **arbitrary larger teacher** into a **requested smaller target
architecture** while preserving reasoning and agent capability. Not a Qwen-only
compressor.

* current study: **~4B → ~596M**
* planned later study: **~30B → ~4.xB**

The second is why nothing may hard-code layer counts, hidden or FFN sizes, head counts,
or a target parameter count.

## 2. Scientific state — what E1–E8 established

1. **PCA/structural initialization decisively beats random initialization** (E1).
2. **Same-distribution scaling improved autonomous stability, not reasoning
   correctness** (E1, E6). Behaviour rises to ~0.73 usable rollout; correctness does not
   move.
3. **KD-heavy scales better than CE-heavy on autonomous stability** (E4, E6b).
4. **Extra unseen-text KD (FineWeb-Edu) strongly recovers general language modelling
   without solving autonomous reasoning** (E7).
5. **General-language NLL is not a reliable promotion criterion** (E7). The checkpoint
   with the best held-out NLL of its trajectory produced *zero* protocol-valid
   generations.
6. **A full-width depth-ablation proxy does not predict the fully-compressed step-0
   initializer** (E8a). The contribution map preserves the teacher **3.11×** better at
   full width (KL 0.620586 vs 1.932531) and initializes **2.8 nats worse** once composed
   with width/FFN/attention compression. **This mismatch is the reason AutoInitializer
   exists.**
7. **E8b did not complete recovered behaviour.** No DP-vs-DC or FP-vs-FC result exists
   and **no depth × compression interaction may be claimed.**

**The open problem is correctness.** Eleven interventions moved behaviour or nothing;
none moved reasoning.

## 3. Canonical assets

| asset | identity |
| --- | --- |
| teacher | `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d9ea81521153ed38c47d515654e938aea` — 36 layers, hidden 2560, FFN 9728, 32Q/8KV, head_dim 128, vocab 151936, rope_theta 5,000,000, 4,022,468,096 params |
| Stage-1 init (canonical) | `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint` — sha256 `86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54`, 596,049,920 params (1024 hidden, 28L, FFN 3072, 16Q/8KV, tied emb) |
| E8a contribution init | `artifacts/stage1/e8_contribution_init_v1/checkpoint` — sha256 `7a0694a5d5c59f8e0b0ebc9ac8648b1ec026bf93cab026d33c61ca8fc85d1edb` |
| E8a frozen depth map | removes teacher layers **[2, 3, 15, 16, 20, 21, 26, 32]** |
| behavioral anchor | `artifacts/stage3/rescued/e1_r2960k_sb_pca` — E1/P1 KD-heavy 2.96M, seed sb. **SOLE COPY, not on the relay** |
| Stage-0 activation stats | `artifacts/stage0/qwen3_4b_thinking_v1/activation_stats.safetensors` — sha256 `aaeb2e4c…`, also on the relay at `e8_inputs_20260810/stage0/qwen3_4b_thinking_v1/` |
| recovery recipe | `configs/stage3/e1/e1_r*_pca.json` — `ce_weight 0.25, kd_weight 1.0, T 1.0, kd_scope all`; seeds `20260726` (sa), `20260801` (sb) |
| training pack | `artifacts/stage3/ladder_uniform_probe` — `blocks.npz` sha256 `6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c`, block_len 8192 |
| frozen promotion battery | 150 prompts, inclusion mask `d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba`, sampled from the 0.86M rung |
| retained reference on it | usable_rollout **0.7300** · correct_overall **0.1867** · correct_given_usable **0.2511** |
| E8a calibration mixture | 67 items, content sha256 `d65c1f40e4837ea1bd5bcc33c68041a13b797c68f5be3c0686e0142ed761028f`, leakage-checked |
| **permanent controls** | `preflight_ctl_r0860k_{sa,sb}` — weights `573847a730c1a499…` / `4c6adcf861871690…`, shared observed protocol `aad75fee8a897d9c…`, probe ids `799bd5ac…` / `793f786a…`. On the relay at `permanent_controls/`, and in `~/aad-artifacts/autoinit/`. **Never retrain these.** |
| **recovery-search battery** | `artifacts/stage3/recovery_search_v2` — 190 prompts, 170 scorable, content `a1b22778…`, scoring contract `recovery_search_scoring@v2` digest `808080a7…`. v1 is INVALID (rendered 0/20 tool prompts) and is kept only with a `SUPERSEDED.md`. |
| **materialized thresholds** | `logs/autoinit_stage3_complete/materialized_thresholds.json` — equivalence interval **0.011695**, feasibility floor **0.3000**, from pooled control rates. Phase A cannot start without these. |
| control reference on `recovery_search_v2` | pooled `usable_rollout_rate` **0.3711** · `correct_overall` **0.0118** · `correct_given_usable` **0.0286** (380 prompts, seeds sa+sb) |

Every deleted checkpoint has a tombstone in
[`../logs/checkpoint_tombstones.json`](../logs/checkpoint_tombstones.json) with hash,
reason, reconstruction recipe and cost. **A scientific conclusion never depends on a
file still existing at a path** — resolve through
[`../logs/checkpoint_registry.json`](../logs/checkpoint_registry.json).

## 4. Closed paths — do not reopen without a new hypothesis

* **E8b completion** (DP/DC/FC recovery, the S5 lifecycle diagnostic, the ≥94 GB
  hardware question). Strategically terminated.
* **P2 / CE-heavy expansion**, including P2-5.50M.
* **FineWeb ratio sweeps** — E7 answered the question.
* **Selecting on held-out NLL** — retired.
* **Restricting attention updates** (freezing, LoRA) — degraded stability, both seeds.
* **Student-prefix recovery** — teacher-prefix wins, 0.77 vs 0.45 usable rollout.
* **E2 phases 2–3** — never authorized; phase 3 was built around a retired metric and
  must not run as written.

## 5. Next work

**AutoInitializer Phase 22 step 8 onward.** The architecture below is already decided;
implement it, do not redesign it.

### 5.1 Decided constraints

1. **OperatorKind and OperatorImplementation are distinct.** Initial kinds: `DEPTH`,
   `RESIDUAL_WIDTH`, `FFN`, `ATTENTION`. The engine must accept **new kinds** without
   core changes.
2. **Implementations are versioned algorithms:** `depth.positional_v0`,
   `depth.causal_kl_greedy_v1`, `width.global_pca_v0`,
   `ffn.activation_importance_v0`, `attention.weight_proxy_v0`. A new algorithm
   registers a **new ID**; it never changes an old ID's semantics. Historical
   implementations are immutable.
3. **Architecture extensibility.** Future kinds may include `MOE_EXPERT_SET`,
   `MOE_ROUTER`, `MOE_SHARED_EXPERT`; ATTENTION needs family-specific implementations
   for MHA, GQA, MLA, linear attention and successors. **The core must not assume
   exactly four permanent kinds** — dispatch on capabilities and preconditions.
4. **Operator order is searchable and non-commutative.** `D→A→F→W` and `A→F→W→D` are
   different paths.
5. **Each operator acts on the checkpoint the previous operators produced**, and that
   state is remeasured. The operator's *local* reference is its parent state; the
   *global* reference for state evaluation remains the original teacher. Represent the
   distinction explicitly.
6. **Intermediate checkpoints are search states only.** They may have intermediate
   parameter counts. They **never** enter recovery-probe Top-N.
7. **Every Beam Top-N leaf must exactly match the requested target architecture** —
   ~596M in the pilot, ~4.xB in the 30B study.
8. **Calibration profile is per operator and searchable.** No single global mixture.
9. **Every generated checkpoint independently measures and hash-binds its own
   diagnostics. No inherited NLL.**
10. **Metric hierarchy, not interchangeable:** operator-local objective → global
    intermediate-state evaluation → beam-ranking policy → 0.86M recovery search battery
    → final frozen promotion battery.
11. **Operator-local metrics are algorithm-specific** (depth causal search may use
    parent→candidate KL; PCA captured covariance/energy; FFN activation importance) and
    **do not automatically become beam metrics**.
12. **Global state metrics include original-teacher fidelity by domain:** general KL,
    reasoning/math KL, RAG/multihop KL, code KL, tool KL, critical-token fidelity, and
    general NLL as a diagnostic guardrail.
13. **Beam ranking is multi-objective/Pareto-like, not minimum NLL** — E7 is why. NLL
    alone may not eliminate a state that remains competitive on reasoning/domain
    fidelity. The policy must be config-driven, versioned, hashed and frozen before any
    paid search.
14. **Top-N target-size leaves receive identical ~0.86M recovery probes**, likely
    successive-halving: Top-N on seed sa → survivors on seed sb → Top-1. **Exact N and
    thresholds must be preregistered**, not chosen post hoc.
15. **The final promotion battery is isolated from the entire search.**
16. **Long-term KD direction is sparse Top-K KD**, candidate support
    `TopK_teacher ∪ TopK_student` with explicit tail/residual treatment. Full-vocabulary
    KL is **not** the permanent interface. This is future work, **not** v1.
17. **PyTorch SDPA using its flash backend and separately packaged FlashAttention/FA2
    are distinct runtimes** and must be named separately. The current path is SDPA's
    flash backend; no attention change is pending.
18. **v1 reuses the existing frozen recovery KD** so initialization stays the
    experimental variable.

### 5.2 Build status — updated 2026-08-12

**The framework is built and CPU-verified.** `src/aadistill/autoinit/` implements
every constraint in §5.1: the architecture-adapter layer (`arch.py`,
`adapters/qwen3.py`), the operator kind/implementation registry with immutable
versioned ids and a committed ledger (`operators/base.py`,
`configs/autoinit/operator_ledger.json`), the five v1 operators wrapping the
existing algorithms plus the incumbent recipe as a `COMPOSITE_STAGE1` operator,
versioned calibration profiles (`calibration.py`), dataset-role isolation
(`datasets.py`), the versioned search state with hash-bound metrics (`state.py`),
the four-level metric taxonomy (`metrics.py`), the Pareto beam-ranking policy
(`ranking.py`), the deterministic resumable beam search (`search.py`), the search
manifest (`manifest.py`), the recovery orchestration interface (`recovery.py`) and
the cost model (`cost.py`). 112 tests; an end-to-end dry run on real tiny
checkpoints; the search core proven family-agnostic against a non-transformers MoE
fixture.

**Corrected 2026-08-12** after a maintainer review of the implementation: the
calibration branch identity (a `CalibrationNeed.NONE` operator no longer branches
over profiles), shard-safe checkpoint identity, the canonical control injected by
frozen hash rather than re-executed, `state.nll.general` computed from the general
domain alone and demoted out of the beam objectives, delayed pruning with
epsilon-dominance and lineage diversity, a constraint-then-objective recovery
selector, an activation-statistics cache keyed on the parent artifact, and
recompute-per-candidate reference logits. Nine items, each pinned by a test in
`tests/autoinit/test_corrections.py`; see [`../logs/decisions.md`](../logs/decisions.md)
2026-08-12.

**What is still missing, and blocks a paid run:** the initializer-state evaluation
suite, the recovery search battery, the `calib.reasoning_heavy@v1` mixture, a
frozen halving preregistration, and one measurement of the statistics-pass GPU/CPU
split. All zero cost. See [`../logs/autoinit_pilot_proposal.md`](../logs/autoinit_pilot_proposal.md) §3.

## 6. Budget

```
actual cumulative spend        $191.5462
authorized cumulative cap      $211.07
unused, uncommitted            $ 19.5238
paid compute running           NONE
```

**Phase A does not fit.** Repriced from the measured battery on 2026-08-15 it is
**$12.36 expected / $20.13 hard**; the hard bound exceeds the $19.5238 remaining
by $0.61. That is a maintainer decision, not an accounting one — the options and
what each costs are in
[`../logs/autoinit_phase_a_repricing.md`](../logs/autoinit_phase_a_repricing.md).

E8b's termination released its earmark. Full reconciliation, including the
E6b $0.56 overrun and the limit of the evidence-file record, is in
[`../logs/BUDGET_LEDGER.md`](../logs/BUDGET_LEDGER.md). **Plan from actual spend, never
from unused room under a previous authorization.**

## 7. Read these, in order

1. [`../logs/STATE.md`](../logs/STATE.md) — current state, a few minutes
2. **this file**
3. [`../logs/EXPERIMENT_INDEX.md`](../logs/EXPERIMENT_INDEX.md) — what each of E1–E8
   proved and what it does *not* support
4. [`../logs/decisions.md`](../logs/decisions.md) — decision records, including the
   AutoInitializer constraints and the backend freeze
5. [`../logs/checkpoint_registry.json`](../logs/checkpoint_registry.json) +
   [`../logs/checkpoint_tombstones.json`](../logs/checkpoint_tombstones.json)
6. [`../logs/e8b_backend_audit.md`](../logs/e8b_backend_audit.md) — the runtime findings
   that constrain any 3B+ recovery
7. [`../AGENTS.md`](../AGENTS.md) — the working contract; P17/P18 and the promotion rules
   are binding
8. [`../logs/current_state.json`](../logs/current_state.json) — machine-readable version
   of the above

## 8. Runtime facts that will bite you

* **A container reports the host's CPU count.** `nproc` said 128 while the cgroup granted
  13; torch sized its pools from 128 and a test subprocess burned 900 s at 1338% CPU on
  work that takes 7.4 s. Read `/sys/fs/cgroup/cpu.max` and enforce with `taskset`. Also
  **`nproc` honours `OMP_NUM_THREADS`**, so it is unusable as a fallback once you set
  thread caps.
* **A detached job inherits only the env you pass it.** `HF_TOKEN` absent →
  `RepositoryNotFoundError`, which reads as a missing repo rather than a missing
  credential.
* **`max_memory_allocated()` is a running maximum, never reset.** It cannot distinguish
  growth from having met a worse workload. Use instantaneous
  `memory_allocated()` at fixed lifecycle boundaries.
* **Recovery at 3.2B does not fit an 80 GB A100 with margin.** Fixed cost is 54.82 GB
  (fp32 master weights + fp32 grads + AdamW m,v + bf16 teacher) and the
  `[sequence, vocabulary]` transients add ~10.9 GiB concurrent. `masked_ce`'s unchunked
  `sel.float()` is the single largest buffer at 4,978 MB. **CE/KD memory is unresolved**
  and will matter again for ~4.xB targets.
* **Validate before spending.** Run the suite pinned to the pod's CPU budget
  (`taskset -c 0-12`) and under `scripts/pod/simulate_pod_env.sh` with the session's
  hide-set. Those two checks would have caught four of five defects that reached paid
  pods.

## 9. First task for the next session

**Updated 2026-08-15.** The five zero-cost prerequisites are done, and Stage 3 is
**complete**: both permanent controls are characterized and the thresholds are
materialized (§3). The framework is built and the orchestration path works end to
end — attempt 8 ran setup, Stages 0–3, artifact collection and provider-confirmed
teardown without an infrastructure defect.

> **Nothing is authorized.** The cap was raised to **$213.00** on 2026-08-15 and
> the Phase-A harness — which did not exist until then — is now built and
> rehearsed, so Phase A now *fits* ($20.0126 hard against $21.4538 remaining).
> **Fitting is not authorization.** No `PhaseAAuthorization` artifact has been
> issued, and `scripts/pod/autoinit_phase_a_launch.py` refuses to create a pod
> without one. Do not launch paid compute without issuing one. Do not retrain the
> permanent controls. Do not reopen the frozen AutoInitializer search/recovery
> design, or E8b.
>
> Two things are still open: the searched-leaf **storage plan** (weight hashes
> are not reproducible across sessions, so a lost leaf can only be re-searched;
> relay capacity for 5 × ~2.3 GB is unconfirmed), and the **operator-build
> compute**, the one term in the search still covered by an allowance rather than
> a measurement.

What is worth reading first, in this order: [`../logs/STATE.md`](../logs/STATE.md)
for the position, [`../logs/autoinit_stage3_complete/`](../logs/autoinit_stage3_complete/)
for what Stage 3 produced, and
[`../logs/autoinit_continuation_attempts/`](../logs/autoinit_continuation_attempts/)
for what eight attempts cost and why.

### 9.0 The infrastructure lesson, because it cost $2.3 of the $4.1

Four of the seven failed attempts died on lines in
`scripts/pod/autoinit_preflight_setup.sh` that no rehearsal had ever executed —
an offline-install flag, an unpinned `pip install`, a cross-session authorization
binding, and a status filename the launcher did not probe. Each was found by
paying for the next one.

`tests/pod/test_setup_end_to_end.py` now runs the **real** setup script from entry
to `SETUP_DONE` inside bubblewrap and reads the result back through the launcher's
own `PROBE_COMMAND` and `parse_setup_probe`, stubbing only the expensive external
operations. It needs the 196-wheel vLLM wheelhouse at
`~/aad-artifacts/wheelhouse_vllm_cp312` (or `AAD_VLLM_WHEELHOUSE`); without it the
test **skips**, so check it ran. Rebuild with
`scripts/pod/build_wheelhouse.py --from-pins --requirements requirements-vllm.txt`.

Before any paid run: the full suite, `scripts/pod/simulate_pod_env.sh`, and that
end-to-end rehearsal.

### 9.1 One correction to carry

An earlier reading of the record treated `d65c1f40…` as the sha256 of
`artifacts/stage1/e8_calibration_v1/items.jsonl`. It is not. It is the mixture's
**token-level content hash** — sha256 over `item_id:sha256(ids)[:16]` lines, as
`scripts/data/build_e8_calibration.py` computes it. The items *file* hashes to
`c7202338…`. Both are now pinned separately and the token-level one is re-derived
from the loaded items rather than trusted.
