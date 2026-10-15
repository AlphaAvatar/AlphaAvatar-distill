# D2 v3 formal search: evidence review (2026-10-15)

**Status:**

- The D2 v3 formal search `d2_search_20261015_051722` is **COMPLETE**. It ran
  92/92 expansions and produced 12 complete leaves.
- The preregistered Top-4 is committed, fetched off-pod and identity-verified.
- The pod is deleted and the provider reports 0 pods. Nothing is billing.
- **$5.5364 all-in** was spent, against $11.14 expected and a $23.69 hard
  ceiling.
- This record authorizes nothing. D2 recovery screening, confirmation, D3, a
  fresh confirmation and any promotion are **not** granted.

| | |
| --- | --- |
| run | `d2_search_20261015_051722`, the grant's one use |
| preregistration | v3 `b52810921a4c…`, APPROVED 2026-10-14, `prediction_rows = scored_v1` |
| pod | `ricuwbh64pqlwd`, NVIDIA L40S, $1.09/h; image `runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404@580.126.09` |
| runtime | torch 2.11.0+cu128, CUDA 12.8, 44.4 GiB |
| closeout | [`../../../phase_d2/runs/d2_search_20261015_051722/closeout/outcome.json`](../../../phase_d2/runs/d2_search_20261015_051722/closeout/outcome.json) |
| derived evidence | `evidence/{complete_leaf_ranking,depth_choices,search_analysis}.json`, produced by `scripts/stages/stage-1/phase_d2/analyze_d2_search.py` from the archived journal |

## 1. Executable commit and one-use grant

| | |
| --- | --- |
| session commit | `d64490eeab7541eb20259dd541d84af344922dd4`: the authorized base `e0a7443e` plus only this run's `governance/authorization.json` (lineage gate ok) |
| bundle | `aad_autoinit_d64490ee.bundle`, sha256 `b9437033…`, round-trip verified before the provider was called |
| harness digest | `4829cf21…` at the commit, equal to the authorized value |
| authorization | `authorization_sha256` `987bf0a1…` (file sha256 `9e5931d0…`), granted 2026-10-15T05:17:22Z at a live $1.09/h; expected $11.141, hard cap $23.6923 over 1240.86 min, pricing `cache_fix_validated`, rows `scored_v1` |
| identity on the device | config `5dd1ee15…`, protocol `4fecc463…`, objective `90e851fd…`, ranking policy `836ae1d3…`; all equal to the grant |
| grant | **consumed** (`one_use_grant_consumed: true`); one provider resource was created |

## 2. Search completion

**COMPLETE FORMAL SEARCH.** Every condition the maintainer set for reporting
COMPLETE holds:

- The search ended normally. Stage C `beam_search` took 16 170.8 s and
  ended ok with 93 states (the root plus 92). The soft deadline (1205 min)
  did not fire, the cache guard was armed, and no cache regression occurred.
- The Top-4 was committed by the driver. Stage D `commit_top_k` was ok, and
  `stage1_selection.json` (sha256 `5a264763…`) loads hash-verified.
- All four checkpoints were retrieved before the pod was deleted. Each one
  was `rc=0, identity MATCHED (re-identified)`, and the teardown gate's
  `required_products_secured` was true.
- Their identity was verified independently: each `model.safetensors` was
  re-hashed on the dev box and equals the selection's `single_shard_sha256`.

| UTC | event |
| --- | --- |
| 05:19:24 | pod created; account balance $142.30 against $30 required |
| 05:25:15 | setup complete; pod test gate `scripts/stages/stage-1/phase_d2/tests` passed in 149 s |
| 05:25:18 | driver start |
| 09:55:06 | `ALL_DONE` |
| 09:58:01 | manifest: 5/5 required, final streams quiescent |
| 10:09:11 | teardown gate allowed (12/12 checks) |
| 10:09:22 | pod deleted; provider confirms gone |

## 3. Top-4 structures and ranking evidence

The committed order is the objective ranking policy's quality order
(`beam.pareto_scoring_objective@v1`, retention `quality_only`). Lineage
diversity is **not** applied, as preregistration v3 requires.

| # | quality order (front) | state | path | objective | raw union KL | root teacher-top-k KL | removed layers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 (0) | `59013099` | WIDTH(dom) → DEPTH(dom) → FFN(rh) → ATTN(dom) | 6.6132 | 7.2054 | 7.2009 | 4 9 13 18 30 32 34 35 |
| 2 | 2 (0) | `c2cb27ef` | WIDTH(dom) → DEPTH(dom) → FFN(dom) → ATTN(rh) | 6.6578 | 7.2651 | 7.2603 | same |
| 3 | 3 (0) | `30fe791a` | WIDTH(dom) → DEPTH(dom) → FFN(rh) → ATTN(rh) | 6.6710 | 7.2726 | 7.2680 | same |
| 4 | 4 (1) | `3a4173e4` | WIDTH(dom) → DEPTH(dom) → FFN(dom) → ATTN(dom) | 6.6610 | 7.2677 | 7.2627 | same |

`dom` = `calib.domain_balanced@v1` and `rh` = `calib.reasoning_heavy@v2`. The
objective and KL columns are equal-domain means.

All four have 596 049 920 parameters, arch signature `09147a5c…`, and
1 192 135 096 bytes. Their artifact and weights digests are in the closeout.

**The Top-4 is a single lineage.**
- All four share `RESIDUAL_WIDTH(dom) → DEPTH(dom)` and the same eight removed
  layers. They differ only in the FFN and ATTENTION calibration profiles.
- The next four leaves (quality order 5–8, objective 8.18–8.25) are one other
  lineage, `ATTENTION(rh) → DEPTH(dom)`.
- The gap between the two lineages is large: 6.61–6.67 against 8.18.

Inside the Top-4 the spread is small. Objective, raw KL and the frozen
ranking agree on the first two positions. Orders 3 and 4 swap between the
Pareto order and the equal-domain mean, because `30fe791a` sits on front 0 by
its critical-token and worst-domain values.

**The winning lineage was kept by search-time diversity, not by its score.**
- At depth 2, `RESIDUAL_WIDTH(dom) → DEPTH(dom)` was the sixth of the six
  states kept, with objective 5.437. Several better-scoring depth-2 states
  were pruned, among them `FFN(rh) → DEPTH(dom)` (1.72) and
  `DEPTH → FFN` (2.44–2.45).
- It then produced the best leaf. Along that path the objective was 11.74
  after WIDTH, fell to 5.44 after DEPTH, and rose only to 6.18–6.28 after
  FFN and 6.61–6.67 at the leaf.
- The proxy is not monotone along a path, and the intermediate objective did
  not predict the final leaf ranking.

## 4. Raw KL against the teacher-weighted union KL

**Teacher confidence (the root stores).** Both profiles' stores were built in
`root_reference_store` mode, in 6.8 s and 6.0 s:

| profile | positions | mean | median | q05 | effective fraction |
| --- | --- | --- | --- | --- | --- |
| domain_balanced | 59 763 | 0.8957 | 0.9864 | 0.532 | 0.969 |
| reasoning_heavy | 44 324 | 0.8960 | 0.9872 | 0.527 | 0.969 |

At the state level, every state is weighted by the same root teacher's
confidence on the fixed `state_eval@v1` suite (mean weight 0.9117). So the
objective and the raw union KL differ by a per-domain reweighting and never
reorder much. The objective/raw ratio over the 12 leaves is 0.908–0.919.

**Final ranking: the weighting changed nothing.**
- Over the 12 leaves, Spearman(objective, raw union KL) = 0.993, and
  Spearman(objective, root teacher-top-k KL) = 0.993.
- The top four by the objective, by the raw union KL, and by D1's
  PARETO_V1 policy on the raw KL (the KL-only control) are the same four
  states: overlap **4/4**.
- The control was reported only and was never used for selection.

**DEPTH: the reference, not the weighting, is where D2 departs.** The table
covers all 14 DEPTH expansions. Two are root children, where the parent *is*
the root; 12 are deeper.

| comparison | result |
| --- | --- |
| round-0 first layer: objective = raw union KL = root teacher-top-k KL | **14/14** |
| round-0 first layer: objective = parent-reference KL, at the root | 2/2 |
| round-0 first layer: objective = parent-reference KL, off the root | **0/12** |
| round-0 Spearman(objective, raw union KL) | ≥ 0.9858 (all 14) |
| round-0 Spearman(objective, parent KL) off the root | 0.06–0.80 (WIDTH-first parents 0.06–0.12, ATTN-first 0.43–0.63, FFN-first 0.75–0.80) |
| greedy rounds where the objective's choice was also the raw-KL argmin | **99 / 112** |

What this means:
- The root reference rewrites which layer DEPTH removes first in every
  non-root expansion.
- The confidence weighting changed 13 of 112 greedy choices relative to the
  unweighted root-referenced KL. Three of those 13 fall in
  `RESIDUAL_WIDTH(dom) → DEPTH(dom)`, the expansion every Top-4 member
  descends from (5/8 rounds agree there).
- The parent-KL comparison is the operator's registered round-0 diagnostic
  only. The alternative greedy trajectories were not replayed, so whether
  raw KL would have removed a different final set is **not** claimed.

This agrees with the Stage W attribution: the single-round difference came
mainly from the root reference. It does not establish what determines
behaviour.

**Proxy comparison with D1, descriptive only.**
- On the two states D1 and D2 share by artifact digest (the root-child DEPTH
  states), D2's root teacher-top-k KL reproduces D1's `state.teacher_kl`:
  0.511197 against 0.511240.
- On that metric, D2's best leaf scores **7.2009** and its fourth 7.2680.
  D1's best leaf scored 7.4341, and D1's committed, lineage-rotated Top-4
  spanned 7.4341–15.2268.
- D1's winning path, `FFN(dom) → DEPTH(rh) → WIDTH(rh) → ATTN(rh)`, ends at
  10.2958 in D2 (quality order 11). D2 also weights the FFN and ATTENTION
  statistics, so that difference is not DEPTH alone.
- A lower step-0 proxy is **not** recovered behaviour (AGENTS.md 4.2). The
  behavioural question is screening's.

## 5. Beam trajectory and operator timing

| depth | states | expanded | pruned | complete leaves | invalid |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 8 (warm-up) | 0 | 0 | 0 |
| 2 | 48 | 6 | 42 | 0 | 0 |
| 3 | 24 | 6 | 18 | 0 | 0 |
| 4 | 12 | — | 0 | 12 | 0 |

The schedule was `beam.delayed_prune`, width 6, warm-up 1. Every pruned state
was evaluated first; the reason recorded for each is "epsilon-dominated
(front n) and the beam of 6 was already full". The 92 states break down as
92 measured, 60 pruned and 0 invalid.

The six states kept at each level are in `search_analysis.json :: beam`:

- **Depth 2:** FFN(dom)→DEPTH(dom) 1.694, FFN(dom)→DEPTH(rh) 1.706,
  DEPTH(rh)→FFN(rh) 2.442, ATTN(dom)→DEPTH(dom) 2.852,
  ATTN(rh)→DEPTH(dom) 2.916, WIDTH(dom)→DEPTH(dom) 5.437.
- **Depth 3:** ATTN(rh)→DEPTH(dom)→FFN(dom|rh) 4.379/4.381,
  DEPTH(rh)→FFN(rh)→ATTN(dom) 6.180, WIDTH(dom)→DEPTH(dom)→FFN(dom|rh)
  6.184/6.278, FFN(dom)→DEPTH(rh)→WIDTH(rh) 6.269.

| operator | expansions | mean min | min–max | total min |
| --- | --- | --- | --- | --- |
| DEPTH `causal_kl_greedy_v1` | 14 | 16.05 | 13.79–20.11 | 224.68 |
| FFN `activation_importance_v0` | 20 | 0.50 | 0.19–0.78 | 10.08 |
| ATTENTION `activation_importance_v1` | 28 | 0.30 | 0.11–0.68 | 8.40 |
| RESIDUAL_WIDTH `global_pca_v0` | 30 | 0.42 | 0.24–0.81 | 12.59 |

**Cost by component.**
- Operators total 255.8 min, plus 12.8 min of per-expansion
  identify/materialize/evaluate/validate.
- State evaluation took 329 s over 92 evaluations.
- Inside DEPTH: ablated forwards 8 881 s, distortion 4 202 s, the
  objective 214.7 s, and the round-0 parent-KL diagnostic 339.4 s.

**DEPTH reference caching.**
- Every one of the 14 expansions ran `root_reference_store` with the parent
  reference `cached`.
- 939 120 reference hits, 0 fills, 0 recomputes.

**Against the price.**
- DEPTH averaged 16.05 min. The priced cell was 25.73 min at the root and
  25.39 min deeper; D1's DEPTH averaged 31.1 min.
- The plan assumed D1's 16 DEPTH expansions. This search ran 14, because the
  beam kept only states that already had DEPTH from depth 3 on.
- The session took 289.98 min against 583.5 expected and 1240.86 hard
  (0.23 of hard).

## 6. The four budget limits, after this session

Derived by `scripts/maintenance/consolidation/derive_budget.py` from the
recorded closeouts. This session books **$5.5364** under `phase_d2`.

| limit | allowance | spent | remaining |
| --- | --- | --- | --- |
| formal | $256.6523 | $134.2263 | **$122.4260** |
| engineering | $25.0741 | $21.8937 | **$3.1804** |
| package | $281.7264 | $156.1200 | **$125.6064** |
| project | $595.00 cap | $476.5516 cumulative | **$118.4484** |
| per session | $30.00 | — | unchanged |

**The programme, re-derived from actual spend**
(`plans/d_series_programme_budget.json`, cache-fix pricing):
- The D2 search is now booked at its actual cost and is no longer planned at
  its ceiling (`d2_search_completed`).
- What remains of D2 is screening ($20.83 hard, $16.61 expected) plus
  confirmation ($8.55 hard, $6.97 expected). With teardown reserves that is
  **$29.98** at hard.
- After D2 at hard, **$88.47** remains, against **$68.16** needed for D3
  ($52.66) plus one fresh confirmation ($15.50). The reservation holds with
  $20.31 to spare.
- Scenario C (D2 + D3 + fresh confirmation) needs $98.14 at hard against
  $118.45 remaining.
- The extra $5.00 granted on 2026-10-14 was not spent as free budget: the
  search needed $5.54 of its $23.69 ceiling.

## 7. Artifact and checkpoint integrity

| artifact | where | identity |
| --- | --- | --- |
| `d2_search.json` | in tree, `evidence/` | sha256 `c99b801e…`, equal to the manifest |
| `stage1_selection.json` | in tree | `5a264763…` (selection), file `71a530b8…`; loads hash-verified |
| `journal.jsonl`, `telemetry.jsonl` | in tree | `d54722fa…`, `59ca70af…` |
| `states.jsonl` (92 states, 118 MB) | out of tree, `<scr>/store/d2_search_artifacts.tar.gz` (sha256 `d8d95700…`) | `1a91541d…`; the selection's journal hash names the same file |
| Top-4 checkpoints | out of tree, `<scr>/products/<state_id>/`, 4 × 1.19 GB | launcher re-identified (MATCHED) and dev-box sha256 equal to the selection's `single_shard_sha256` |

`<scr>` is `/home/ecs-user/aad-artifacts/phase_d2/d2_search_20261015_051722`.
These checkpoints are a dev-box copy, as D1's Top-4 are. They travel to a pod
only when an authorized screening session actually reads them (P8.4). Keeping
them grants no permission to use them.

## 8. Provider teardown and cost closeout

- The teardown gate allowed deletion with all 12 checks true. These include
  `archive_contents_verified`, `checkpoint_hashes_matched` and
  `required_products_secured`.
- Final pod state: `exists: false`, `TERMINATED`, `billing: false`.
- The watchdog ended itself on `pod_gone` at 10:09:38Z, after 287 ticks.
- An independent provider query at about 10:12Z returned no pods, no network
  volumes and a current spend of $0/h.
- **Cost:** GPU $5.2679 (289.98 min at $1.09/h) plus container disk $0.2685
  (400 GB) gives **$5.5364 all-in**. That is the figure booked.
  - The provider's balance moved $142.2981 → $136.7927, a debit of $5.5053.
  - The larger derived figure is booked, so a late settlement cannot make the
    ledger understate spend.
- Observers: the launcher's waiters ended on `LAUNCH_EXIT=0`, the tmux session
  exited, and the watchdog ended. No observer remains on this run.

## 9. Readiness for behavioural screening

**Scientifically and financially ready. Not authorized, and its execution
chain is not built.**

Ready:
- The field is frozen: preregistration v3 says "the 4 retained D2
  candidates", and they are committed and secured.
- The comparator is frozen: B's two imported screening units, not retrained
  and not regenerated. q2's rows are descriptive only.
- The recovery recipe (`e1_p1_kd_heavy@0.86M`) and the battery are frozen.
- Seeds `516448251` and `614923639` are frozen, which makes **8 probes**.
- The decision rule is frozen: exactly one candidate advances, or none.
- Screening fits one session: $20.83 hard plus the $0.30 reserve is under the
  $30 per-session cap. The formal allowance ($122.43) and the programme
  reservation both cover it.

Not ready:
- **No maintainer grant.** Screening and confirmation each need a separate
  authorization.
- **No D2 behavioural session chain.** D1's behavioural launcher and driver
  (`scripts/stages/stage-1/phase_d1/autoinit_d1_behavioural_*`) are bound to
  D1's experiment and field. D2 screening needs its own driver, authorization
  type and launch chain, presumably by generalizing D1's over the shared
  protocol (`families/d_series/shared_evaluation.py`). That is $0 engineering,
  and it is not started.

**For the maintainer's screening decision:** the four D2 candidates are one
lineage.
- They share the width/depth structure and differ only in FFN/ATTENTION
  calibration profiles, so 8 screening probes will measure close siblings.
- The preregistered field (k = 4) is frozen science, so this review neither
  narrows nor changes it.
- If the maintainer wants a different breadth, that is a protocol decision to
  make before any behavioural datum exists.

**A screening authorization would state:**
- one formal session of 8 probes (4 candidates × 2 seeds);
- the price at the live rate when issued (today's derivation: $20.83 hard,
  $16.61 expected, at $1.09/h);
- the per-session cap ($30) and the project cap ($595);
- the cumulative formal allowance;
- that the screening rule advances exactly one candidate or none;
- that confirmation is a further separate grant.
