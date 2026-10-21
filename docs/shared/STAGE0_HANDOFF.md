# Stage-0 handoff: calibration data volume and mixture, on the adopted baseline

> **Taken up 2026-10-18; the volume programme CLOSED 2026-10-21.** The formal
> study (1x/2x/4x, one D1 full search and one fixed recovery each) and a Top-4
> extension ran: EXECUTION COMPLETE / DOWNSTREAM REASONING FLOOR / VOLUME
> EFFECT ON REASONING NOT RESOLVED. The mixture ratio is DEFERRED, never run.
> - Results and closeout:
>   [`logs/stages/stage-0/calibration_volume/closeout/REPORT.md`](../../logs/stages/stage-0/calibration_volume/closeout/REPORT.md).
> - The next round's facts: [`STAGE2_3_HANDOFF.md`](STAGE2_3_HANDOFF.md).
> - The page below is the handoff as written before that work.

**Status: handoff only (2026-10-17 UTC).** It authorizes nothing. No Stage-0
experiment is designed, no dataset is built, downloaded or generated, and no
GPU resource or grant exists for this work. Design and authorization happen in
new conversations. Every number below names the record that owns it; where
this page derived a figure itself, it says so.

## 0. Where to start reading

- **Branch:** `main` carries everything below. The D-series branch
  `exec/d1-behavioural-readiness` was squash-merged on 2026-10-17 (P12.2) and
  is preserved: records cite its commit hashes.
  - New work starts on `stage0/calibration-volume-mixture`, cut from that squash commit.
  - The tested tree is `27c279b6`. Run `git log -1 main` for the squash
    commit.
- **Inspect, in this order:**

| what | where | sha256 |
|---|---|---|
| the agent contract; the target is P10.2, Stage 0 is §4.2 | `AGENTS.md` | — |
| the adopted baseline (the owner) | `configs/shared/baselines/initialization_forward_baseline_v1.json` | `333af13d9f2441fc658ee064f1a6ec6b15ebf5be5ee8ab31320763bbe41ce266` |
| which baseline is in force | `configs/shared/baselines/selection.json` | `7217d7e49a694b3fe3e17c3b3d567fa9987cf111b89229e76489639dc1eb7184` |
| the generic resolver | `src/aadistill/initialization/baseline.py` | — |
| this project's entry point | `scripts/shared/forward_baseline.py` (`python -m shared.forward_baseline`) | — |
| the D-series terminal record | `logs/stages/stage-1/families/d_series/closeout/programme_closeout.json` | `723d66d6ec45fcb0e3d98fb107fa670ef1a19917bb99d89eb3784002797bc484` |
| the decision | `logs/budget/decisions.md`, last entry of 2026-10-17 | — |
| the forward design for Stage 0, 2 and 4 | `docs/shared/REASONING_TRANSFER_DESIGN.md` | — |
| the live state | `logs/state/current.md`, `logs/state/current.json` | — |
| scored-position counts | `logs/stages/stage-1/phase_d1/analyses/d1_scored_position_profile.json` | — |
| D1's own critique of its calibration data | `logs/stages/stage-1/phase_d1/analyses/d1_followup_technical_review_and_next_round.md` | — |

## 1. The adopted forward baseline

The baseline is `stage1_init.d1_target_aware.scored_v1@v1`. The maintainer
chose it on 2026-10-17 as a selection, not a verdict: D1's confirmation stays
NO-GO.
- **Resolve it in code** with `shared.forward_baseline.selected()`. It checks
  every term against the core and refuses a record edited without
  re-selection.
- **Operators,** one per kind:
  - FFN `ffn.activation_importance_v0`;
  - DEPTH `depth.causal_kl_greedy_v1`, scored by KL(P_parent ‖ C(S)) against
    the intact current parent;
  - RESIDUAL_WIDTH `width.global_pca_v0`;
  - ATTENTION `attention.activation_importance_v1`.
- **Scoring positions.** `positions.supervised_target_v1@v1`, in every
  operator and in state evaluation.
  - FFN, WIDTH and ATTENTION apply it to token-axis statistics. DEPTH and
    state evaluation apply it to prediction positions.
  - `scored_v1` is a separate mechanism. It reduces output-head work, and only
    paths that produce vocabulary logits have that work.
- **Support.** `reference_topk_tail_v1` at K=200: the reference's Top-200
  entries plus one aggregate tail bucket. It is not renormalized.
- **State evaluation** compares against the original teacher (the lineage
  root).
- **Search.** `beam.delayed_prune@v1` (width 6, warm-up 1) with
  `beam.pareto_multi_objective@v2`. Operator order and the calibration profile
  per operator stay free variables of each registered protocol.
- **Execution.** Micro-batch 3, `length_sorted_v1`, `prediction_rows=scored_v1`.
  The fingerprint under cuda/bf16/fp32 is `e627f461…`.

**The historical reference initialization is D1-q2.**
- **Identities:** state `afd50c9ff0e3d20542bb5473ea745b55`, artifact digest
  `3a4b4ede6888e2b0242d59234f1df18f0cac02893617e678992e54eaa243b82b`.
- **Path:** `FFN(domain_balanced@v1)` → `DEPTH(reasoning_heavy@v2)` →
  `RESIDUAL_WIDTH(reasoning_heavy@v2)` → `ATTENTION(domain_balanced@v1)`.
- **Execution profile:** `all_v1`, fingerprint `956a6951…`.
- **Reconstruction:** `scripts/stages/stage-1/phase_d1/autoinit_d1_replay_driver.py`.
  The finalist-rematerialization campaign reproduced it exactly.
- **Copies:** one local copy, protected in the checkpoint registry, and no
  relay copy.
- **No `scored_v1` materialization exists.** A Stage-0 study that materializes
  under the forward profile establishes new identities, and q2's digest is not
  its expectation. The resolver returns q2 only for q2's own fingerprint.

## 2. The calibration data that exists

Two profiles are the D1 search's free per-operator choice. Their owners are
`configs/stages/stage-1/calibration/profiles.json` and each profile's
manifest.

| | `calib.domain_balanced@v1` | `calib.reasoning_heavy@v2` |
|---|---|---|
| items | 67 unique | 62 draws, 51 distinct (with replacement) |
| prediction positions | 59,763 | 59,763 |
| scored under supervised_target | 44,746 (0.7487) | 46,825 (0.7835) |
| reasoning / final-answer positions | 27,797 / 8,560 | 30,225 / 10,523 |
| manifest | `artifacts/stages/stage-1/e8_calibration_v1/manifest.json` | `artifacts/stages/stage-1/reasoning_heavy_v2/manifest.json` |

**Per-domain scored fraction** (domain_balanced / reasoning_heavy):

| domain | domain_balanced | reasoning_heavy |
|---|---|---|
| code | 0.925 | 0.945 |
| general | 1.0 | 1.0 |
| math | 0.926 | 0.923 |
| rag_multihop | 0.431 | 0.461 |
| tool | 0.626 | 0.562 |

All the counts above are owned by `d1_scored_position_profile.json`. The
following are not stated in it:
- **Unique scored positions in reasoning_heavy@v2:** 32,709. This page derived
  it from `items.jsonl`; duplicates inflate the 46,825 above.
- **General documents.** "general" is untemplated FineWeb text. With no
  assistant tag, every one of its positions is scored by the fallback rule.

**Other assets:**
- **Global state evaluation** uses `state_eval@v1`
  (`artifacts/stages/stage-1/state_eval_v1`, content
  `a1197205e43aad0e71c0e1bb436ee7babba3b5d8bb25b9c4d5c464f659db20fc`): 80
  items, 74,022 positions, 54,014 scored (0.7297).
- **The historical Stage-0 warm-up pool** (`warmup_v1`, 3,202 samples,
  `data/stages/stage-0/warmup/warmup_v1.manifest.json`) feeds the old
  sandwich-init pipeline. D1 does not use it.

## 3. Limitations the next design has to face

- **Length window.** Sessions are admitted whole inside [256, 2048] tokens;
  out-of-window sessions are dropped, never cut (e8 manifest `design`).
  - The window keeps only a short tail of the reasoning-heavy slices. Code
    admits 14.4% at a median of 4,431 tokens. openmath admits 39.0% and
    multihop_qa 37.8%. Owner: `d1_scored_position_profile.json :: corpus.by_type`.
  - The long reasoning traces Stage 0 most needs are the ones excluded.
- **Truncation.** General documents are cut at 1,024 tokens. The old Stage-0
  warm-up cuts every sample at 1,024 tokens: 61 of 3,202 hit the cap (derived
  from that cache's manifest).
- **Repeats.** reasoning_heavy@v2 draws with replacement. Eight items repeat:
  code is 8 draws of 4 sessions, openmath 6 draws of 3. It adds no item that
  domain_balanced@v1 lacks.
- **Thin sub-types.** multihop 5 sessions, openmath 5, code 6, so a mixture
  axis rests on very few task instances.
- **Reasoning-token coverage.** Every templated trace closes `</think>` and
  terminates (`tag_positions`).
  - Reasoning positions are 76% (27,797 of 36,459) and 74% (30,225 of
    40,848) of the supervised assistant positions, concentrated in math and
    code (derived).
  - RAG and tool items are mostly prompt, and only 43–63% of their positions
    are scored.
- **Unfiltered traces.** Corpus openmath correctness is 0.380. Code and tool
  traces are unverifiable (`logs/stages/stage-3/history/EXPERIMENTS.md`).
- **Mixed-domain, not fluid reasoning.** The sources are MBPP, GSM8K,
  OpenMathInstruct-2, HotpotQA, SQuAD v2, Glaive and FineWeb.
  - The 2026-10-16 scope declaration classes `code_math` and `multihop_qa` as
    mixed.
  - Rule-induction and compositional tasks are in no mixture yet.
- **Not held out.** The state-evaluation suite is isolated from the 67
  calibration items but drawn from the same corpus and FineWeb fetch.

## 4. The target and the evaluation that must come first

The target is fluid reasoning and adaptive problem-solving (AGENTS.md P10.2).
Report the axes separately: task-contained deduction, inductive rule
discovery, compositional generalization, and interactive adaptation. The
supporting-knowledge and protocol-integrity guard rails are reported apart
from them.

**No claim about retaining fluid reasoning may be made** until a
capability-aligned evaluation exists. It needs:
- per-axis batteries;
- held-out **rule and task families**, isolated by content, transformation,
  generator and lineage;
- difficulty off the floor (current mixed-domain correctness is 2–6%);
- a registered SESOI per battery.

The design is in `REASONING_TRANSFER_DESIGN.md` §2, and its order of work in
§7 puts the evaluation design and batteries before the §3 Stage-0 study.
Official ARC-AGI evaluation material never enters any data.

## 5. What the closed D-series does and does not establish

Owner: `programme_closeout.json`.

| | outcome |
|---|---|
| D1 | search complete; q2 advanced from screening; confirmation **NO-GO** (q2 102/2550 vs B 103/2550 correct; usable about 68.9% vs 59.4%) |
| D2 | v3 search complete; c4 advanced from screening; shared confirmation **NO_GO** |
| D3 | structural **STRUCTURAL_CHANGE** (the teacher-referenced DEPTH keeps layers 26, 30 and drops 8, 31); three probes **INCONCLUSIVE** vs q2 (−0.0067) and vs B (−0.0071), exploratory |
| student-confidence D3 | cancelled, never executed |

Every behavioural number above has the same limits:
- measured on one mixed-domain battery (`d1_confirmation`, 850 scorable
  prompts, code unscorable);
- three seeds and a 0.86M recovery probe;
- correctness near the floor.

None of it is fluid-reasoning evidence. Usable rollout is a guard rail, not
reasoning evidence.

## 6. Money, storage, resources (at closure)

- **Budget,** derived by `scripts/maintenance/consolidation/derive_budget.py`:
  - project $94.8528 of the $595 cap;
  - formal $98.8304;
  - package $102.0108;
  - engineering $3.1804.
- **What can fund Stage 0.** The formal allowance funds only the experiment
  ids in `funds_formal_sessions_of`, and the D-series authorities are closed.
  A Stage-0 programme needs its own scope and grant.
- **Planning reference, not a price.** One complete D2 search session (92
  expansions, scored_v1, a different KL objective) cost $5.5364.
- **Storage:**
  - development host: 26 GB free of 197 GB;
  - local weight units: 74.86 GiB;
  - relay: 71.81 GiB in 1,397 files. Private headroom was last measured on
    2026-10-01 at 32–33 GiB and has shrunk since, so re-measure it before a
    large upload.
- **Retirement-eligible.** 14 D-series weight units (16.69 GB) have no
  consumer under AGENTS.md 2.5; none has been deleted (checkpoint registry).
- **Resources:** 0 pods, 0 network volumes, $0/h, and no watcher
  (2026-10-17 17:47 UTC).

## 7. Open Stage-0 design decisions

None of these is decided or authorized:
1. **The capability-aligned evaluation:** per-axis batteries, family splits,
   SESOI and difficulty strata. Its sources and licenses need authorization
   before any download or generation.
2. **The volume unit and levels for the dose-response:** effective scored
   positions, unique or with replacement, reasoning positions or all scored
   positions.
3. **Mixture axes and proportions** at fixed effective positions, including
   whether untemplated general text stays.
4. **The length window and truncation policy for long reasoning traces,**
   against the state-evaluation memory bound (12 GiB batch budget,
   `d1_design.json :: execution_protocol.state_eval_bound`).
5. **Sampling with or without replacement,** and the minimum distinct
   sessions per sub-type.
6. **Whether to filter teacher traces** by verified correctness, kept apart
   from length.
7. **The fixed controls:** geometry, operators, recovery recipe and seeds; and
   whether the endpoint is per-axis correctness after a fixed recovery.
8. **The study's anchor:** whether it starts by materializing q2's path under
   the forward profile, which creates a new identity and does not reproduce
   q2's digest.
9. **A held-out state-evaluation suite.** The D1 review asks for one isolated
   from the calibration sources.
10. **Hardware, a cost model and a grant** for the whole programme.
