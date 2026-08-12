# Canonical experiment index — E1 to E8

One place to answer: **what has been run, what did each one prove, and which
checkpoints still matter?** Detail lives in [`EXPERIMENTS.md`](EXPERIMENTS.md) (numbered
sections, cited per row); this file is the index, not the chronology.

**Kinds.** `experiment` — a complete scientific comparison with an accepted conclusion.
`diagnostic` — run to answer a question, never promotable. `infrastructure` — produced
artifacts or verified machinery, no scientific claim. `cancelled` — withdrawn before
execution. `terminated` — started, stopped deliberately without a valid endpoint.

**Promotion rule that governs every row:** checkpoint promotion, arm selection and
stage advancement are decided on the **frozen autonomous rollout evaluation**. NLL, CE,
teacher KL, top-1 and rank are diagnostics (decision 2026-08-05 / 2026-08-09).

---

## E1 — data-scaling matrix

| field | value |
| --- | --- |
| kind | **experiment**, COMPLETE 2026-08-02 |
| question | how does recovery scale with supervised token budget, and does structural initialization beat random? |
| arms | 24 (rungs 0.25M–5.50M × PCA/random init × 2 seeds) |
| init lineage | `qwen3_0p6b_init_v0` (positional Stage-1 PCA) and `random_baseline` |
| recipe | E1/P1 KD-heavy — `ce_weight 0.25, kd_weight 1.0, T 1.0, kd_scope all` |
| data scale | 0.25M → 5.50M unique CE tokens |
| seeds | `20260726` (sa), `20260801` (sb) |
| primary result | monotone improvement with scale; PCA init decisively beats random at every rung |
| accepted | **PCA/structural initialization beats random initialization.** Scale improves autonomous *stability*. |
| **not** supported | that scale improves reasoning *correctness* — it does not move |
| checkpoints | `e1_r0860k_{sa,sb}_pca` (behavioral anchor), `e1_r2960k_sb_pca`, `e1_r5500k_sb_pca`, `e1_r2960k_sb_rand`, `e1_r5500k_sb_rand` |
| artifacts | `EXPERIMENTS.md` §11, §28; `logs/e1_test_cases.jsonl` |
| cost | $47.60 |
| in current lineage | **yes** — E1/P1 KD-heavy is still the recovery recipe; 0.86M is still the probe rung |

## E2 — three sequential 0.86M diagnostics

| field | value |
| --- | --- |
| kind | **diagnostic**; phase 1 complete 2026-08-04, phases 2–3 never authorized |
| question | do targeted recipe variations at 0.86M move behaviour? |
| accepted | phase 1's result only |
| **not** supported | anything from phases 2–3; phase 3 was built around a since-retired metric and **must not run as written** |
| artifacts | `EXPERIMENTS.md` §12 |
| in current lineage | **no** |

## E3 — restricting attention updates at 0.86M

| field | value |
| --- | --- |
| kind | **experiment**, complete |
| question | does freezing or LoRA-restricting attention help recovery? |
| accepted | restriction did not help; full-parameter recovery retained |
| checkpoints | deleted 2026-08-09, hashes retained |
| artifacts | `EXPERIMENTS.md` §20; `logs/e3_registration.json` |
| cost | see §20 |
| in current lineage | **no** — its negative result closed the question |

## E4 — P2 CE-heavy scaled 0.86M → 1.60M

| field | value |
| --- | --- |
| kind | **experiment**, complete |
| question | does the CE-heavy objective scale as well as KD-heavy? |
| accepted | **KD-heavy scales better than CE-heavy on autonomous stability** |
| artifacts | `EXPERIMENTS.md` §21; `logs/e4_registration.json` |
| in current lineage | **yes** as a closed comparison — it is why the recipe is KD-heavy |

## E5 — teacher-prefix vs student-prefix recovery

| field | value |
| --- | --- |
| kind | **experiment**, COMPLETE after five attempts (four infrastructure failures) |
| question | does conditioning recovery on student-generated prefixes beat teacher prefixes? |
| accepted | **teacher-prefix continuation beats student-prefix recovery** — usable_rollout 0.77 vs 0.45. Prefix-conditioned targets teach continuation, not closure |
| artifacts | `EXPERIMENTS.md` §22–27; `logs/e5_registration.json` |
| cost | $9.78 across five attempts |
| in current lineage | **yes** as a closed comparison |

## E6 / E6b — scale curve on the frozen battery; objective × scale

| field | value |
| --- | --- |
| kind | **experiment**, complete |
| question | E6: normalize the E1 rungs onto the frozen 150-prompt battery. E6b: does P2 CE-heavy behave differently at 2.96M? |
| accepted | the scale curve holds on the frozen battery; **the objective interacts with scale** — CE-heavy degrades where KD-heavy does not |
| binding asset | frozen 150-prompt battery, inclusion mask `d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba`, sampled from the 0.86M rung |
| artifacts | `EXPERIMENTS.md` §28–29; `logs/e6_report.md`, `e6_results.json`, `e6b_report.md`, `e6b_results.json` |
| cost | $2.36 + $7.68 |
| in current lineage | **yes** — the battery and the retained FP behaviour numbers come from here |

## E7 — FineWeb-Edu KD: does general language modelling reach behaviour?

| field | value |
| --- | --- |
| kind | **experiment**, COMPLETE 2026-08-09 |
| question | does restoring general language modelling improve autonomous reasoning? |
| arms | A retained baseline, B FineWeb-Edu KD, C matched in-domain KD-only control |
| accepted | **general language modelling was strongly restored and behaviour did not move.** Therefore **general-language NLL is not a reliable promotion criterion** |
| **not** supported | that FineWeb solved the reasoning bottleneck |
| artifacts | `EXPERIMENTS.md` §34; `logs/e7_preregistration.md`, `e7_canary_report.md` |
| cost | $10.49 |
| in current lineage | **yes** — it is the direct basis for the beam-ranking rule that NLL alone may not prune a state |

## E8a — contribution-guided depth initialization

| field | value |
| --- | --- |
| kind | **experiment** — **COMPLETE** |
| question | does position-based depth compression discard teacher blocks disproportionately important to the teacher's predictive function? |
| method | frozen teacher → ablated-teacher forward KL over real prediction positions, **iterative greedy** removal (260 subset evaluations for 36→28), domain-balanced over 5 domains |
| frozen result | contribution map removes teacher layers **[2, 3, 15, 16, 20, 21, 26, 32]** (one shared with the positional map) |
| primary result | full-width teacher-ablation KL **0.620586 vs 1.932531 = 3.11× lower**, lower on all 5 domains and every diagnostic (`tool_close` 428×); `self_consistency` exactly 0.0 |
| step-0 result | the fully-compressed contribution init is **2.8 nats worse** than the control on every held-out series — hash-bound, one device, one reload path |
| accepted | (1) the contribution map preserves the full-width teacher output distribution **substantially better** than the positional heuristic; (2) **a full-width depth-ablation proxy does not predict the fully-compressed step-0 initializer.** That mismatch is what motivates conditional, operator-order-aware search |
| **not** supported | any claim about recovered behaviour |
| checkpoints | `e8_contribution_init_v1` (`7a0694a5d5c59f8e0b0ebc9ac8648b1ec026bf93cab026d33c61ca8fc85d1edb`, canonical); control `qwen3_0p6b_init_v0` (`86fbba78…`) |
| calibration | frozen 67-item mixture, content sha256 `d65c1f40e4837ea1bd5bcc33c68041a13b797c68f5be3c0686e0142ed761028f`, leakage-checked |
| artifacts | `EXPERIMENTS.md` §35–36; `logs/e8_preregistration.md`, `e8_step0_report.md` |
| cost | $3.7253 |
| in current lineage | **yes** — `depth.causal_kl_greedy_v1` is the AutoInitializer's second DEPTH implementation |

## E8b — depth-map × compression interaction

| field | value |
| --- | --- |
| kind | **TERMINATED** — strategically, 2026-08-12. **NO VALID RECOVERED-BEHAVIOR COMPARISON.** |
| question | is the contribution map good on its own, or does it only fail once composed with width/FFN/attention compression? |
| design | 2×2 factorial at the 1.60M rung — DP/DC depth-only full-width × FP/FC fully compressed, positional × contribution, two seeds, hardware pair-matched |
| **valid result — step-0 only** | all four initializations measured on one device through one canonical `from_pretrained` reload path, each series hash-bound. DC beats DP by **0.89–1.18 nats** at full width; FC loses to FP by **0.90–2.82 nats** — a unanimous reversal on **9 of 9 metrics across three held-out series**. `resolved_rope_base` 5000000.2415 on all four. FP re-measured digit-identical to the E8a session |
| what did **not** complete | DP-sa trained 1,761 steps and is an **infrastructure artifact, not a promotable scientific arm**. DC-sa **did not complete** (OOM ~step 900). **DP-sb, DC-sb, FC-sa, FC-sb were never run.** |
| **no conclusion exists** | there is **no DP-vs-DC and no FP-vs-FC recovered-behaviour result**, and therefore **no depth × compression interaction claim**. Behaviour must not be inferred from partial training |
| preserved | four step-0 records (`logs/e8b_step0_records/`), all manifests and configs (`configs/stage3/e8b/`), DP-sa logs and manifest (`logs/e8b_s2_dp_sa/`), DC-sa partial log, the memory profile, both gate artifacts, the backend audit, the shape audit, and every failure analysis |
| infrastructure findings | five self-inflicted defects fixed with tests (setup tripwire, cgroup CPU misread, `nproc` honouring `OMP_NUM_THREADS`, an artifact-absence assumption, a CPU test coupled to GPU state); two OOMs; and the **shape audit result that the worst block in the stream is at step 133, inside the 200-step gate's window**, so the late OOM is *not* explained by data-dependent shapes |
| unresolved | whether the late OOM is pure-loop retention, lifecycle retention, or a narrow-margin allocator event. The Branch-B replay was designed and priced, **not run** |
| artifacts | `EXPERIMENTS.md` §37–42; `logs/e8b_preregistration.md`, `e8b_backend_audit.md`, `e8b_stream_shape_audit.json`, `e8b_reprice_after_gate.json` |
| cost | $16.82 |
| in current lineage | **step-0 results yes** (they motivate AutoInitializer); **recovery design no** |

---

## Project-level conclusions carried forward

1. **PCA/structural initialization decisively beats random initialization** (E1).
2. **Same-distribution scaling improved autonomous stability, not reasoning correctness** (E1, E6).
3. **KD-heavy scales better than CE-heavy on autonomous stability** (E4, E6b).
4. **FineWeb/extra unseen-text KD strongly recovers general language modelling without solving autonomous reasoning** (E7).
5. **General-language NLL is not a reliable promotion criterion** (E7) — hence NLL alone may not prune a search state.
6. **E8a demonstrated a strong mismatch between a full-width depth-ablation proxy and the fully-compressed step-0 initializer**, motivating conditional / operator-order search.
7. **E8b did not complete recovered behaviour and must not be used to claim a depth × compression interaction.**

## Current best behavioural checkpoint

`e1_r2960k_sb_pca` lineage — E1/P1 KD-heavy at the 2.96M rung. The 0.86M rung
(`e1_r0860k_{sa,sb}_pca`) remains the low-budget probe rung and the source of the frozen
battery. Retained FP behaviour on that battery: `usable_rollout_rate` 0.7300,
`correct_overall` 0.1867, `correct_given_usable` 0.2511.

**The open problem is correctness.** Eleven interventions have moved behaviour or
nothing; none has moved reasoning.
