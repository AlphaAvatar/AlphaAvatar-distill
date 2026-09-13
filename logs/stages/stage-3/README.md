# stage-3 — Student recovery

Pipeline stage **3**: recover the initialized student offline, before on-policy training (AGENTS.md 4.5).

Generated from [`../index.json`](../index.json). It records the
evidence behind every claim below; a path in a code span is a
pointer to its canonical owner, which is where the revisions,
counts, hashes and configs live. Stage membership is read from
repository facts — a config's stage field, the
`configs/stage<N>/` directory it lives in, a declared `stage_id`,
or an experiment's own subject and product — never guessed from a
name.

## Purpose

Recover the structurally initialized student offline — repair the damage compression did — until it can produce usable autonomous rollouts, which is what Stage 4/5 on-policy work needs from it.

## Inputs

What this stage receives from the pipeline before it.

* the pinned Stage-1 init checkpoint `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`, which every arm forks from so that arms differ only in what is under test
* the same pinned teacher, loaded to produce KD targets rather than to be imitated wholesale
* Stage-2 data: today the teacher-target pilot corpus, with the public mixtures behind it

## Data

* `data/stage3_pilot/manifest.json` — the recovery corpus in use: control and treatment arms over one shared accepted prompt subset, grouped code_math / multihop_qa / rag_evidence, with per-arm token and packing statistics
* `data/stage3_pilot/treatment` — the treatment arm the canonical config points at by default
* `data/eval_behavior_v0` — the behaviour evaluation set the stage is judged on — autonomous rollout, not held-out loss

## How it runs

Only what helps to understand the stage; the config is the
source of truth.

* a run differs from the canonical config only in `data_dir` — the token-ladder subset it trains on — and in `schedule.total_steps`. That is what makes the rungs comparable
* best-fit packing at an 8192-token block length with gradient checkpointing; FP32 master weights under BF16 autocast
* a combined CE + top-k KD objective; the weights, the KD scope and the trainable-parameter patterns are declared per config, and swapping them is what several experiments below are about

## Outputs

* recovered checkpoints under `artifacts/stage3/<run_name>/`, each with its own run manifest. Outside git; the checkpoint registry and the tombstones record which still exist and what was frozen before any were deleted

## This stage's own areas

Material that belongs to the stage rather than to one of its
experiments.

| area | what it holds |
| --- | --- |
| [`history/`](history/) | records spanning this stage's experiments, kept verbatim |

## Pipeline activity

The stage's own work, and measurements supporting it. Not
studies *of* it.

| what | kind | logs | status |
| --- | --- | --- | --- |
| Stage-3 sub-stage 1 — FFN + norm recovery (`s1_ffn_norm`) | `pipeline-activity` | none | complete |
| Stage-3 sub-stage 2 sizing — freeze-set A/B (`s1_ext_v0` vs `s2_blocks_v0`) | `pipeline-activity` | none | complete |
| Stage-3 sub-stage 2 on mixture v1 — the standing branch point | `pipeline-activity` | none | complete |

## Experiments

What each one asked of this stage.

| experiment | what it asked | logs | status |
| --- | --- | --- | --- |
| `ttb` — Teacher-native vs public target, 2x2 from the Stage-1 init | Teacher-native targets or public targets — which recovers better from the Stage-1 init? | none | complete — diagnostic, no route claim |
| `p0_real` — P0-real — the first full-scope KD recovery arms | What does full-scope KD recovery reach from the pinned init? | none | complete |
| `d0` — D0 — no-training diagnostics on P0-real-sa/sb | What can the P0-real checkpoints be shown to do without training anything further? | none | complete |
| `p0` — P0-assistant — assistant-only KD with assistant-token normalization | Does restricting the KD scope to assistant tokens, normalized over assistant tokens, help? | none | complete — does not beat P0-real |
| `p2` — P2-ceheavy — swapping the CE/KD loss weights | Does swapping the CE/KD loss weights toward CE improve recovery? | none | complete — adopted as a protocol |
| `e1` — Experiment 1 — data-scaling matrix, 24 arms | How does recovery scale with the supervised token budget, and does structural initialization beat random? | [`e1/`](e1/) | complete |
| `e2` — Experiment 2 — three sequential 0.86M diagnostics | Do targeted recipe variations at the 0.86M rung move behaviour? | [`e2/`](e2/) | phase 1 complete; phases 2-3 never authorized |
| `e3` — Experiment 3 — restricting attention updates at the 0.86M rung | Does freezing or LoRA-restricting attention help recovery? | [`e3/`](e3/) | complete |
| `e4` — Experiment 4 — P2 CE-heavy scaled from the 0.86M to the 1.60M rung | Does the CE-heavy objective scale as well as KD-heavy from the 0.86M rung to the 1.60M rung? | [`e4/`](e4/) | complete |
| `e5` — Experiment 5 — teacher-prefix continuation vs student-prefix recovery | Does conditioning recovery on student-generated prefixes beat teacher prefixes? | [`e5/`](e5/) | complete — 5 aborted attempts, then a result |
| `e6` — Experiment 6 — the E1 PCA scale curve normalized onto the frozen battery | What does the E1 scale curve look like once every rung is normalized onto the frozen battery? | [`e6/`](e6/) | complete |
| `e6b` — Experiment 6b — P2 CE-heavy at the 2.96M rung | Does the CE-heavy objective behave differently at the 2.96M rung? | [`e6b/`](e6b/) | complete |
| `e7` — Experiment 7 — general language modelling restored; behaviour unmoved | Does restoring general language modelling improve autonomous reasoning? | [`e7/`](e7/) | complete |
| `e8` — Experiment 8 — contribution-guided depth initialization | Does position-based depth compression discard teacher blocks that are disproportionately important to the teacher's predictions? | [`e8/`](e8/) | complete |
| `e8b` — E8b — depth-map x compression interaction | Is the contribution depth map good on its own, or does it only fail once composed with width/FFN/attention compression? | [`e8b/`](e8b/) | strategically terminated — no valid comparison |

An experiment with no `logs` directory RAN, and produced no
log files of its own — its evidence is its configs, its
artifacts and the historical index. It is listed here
because omitting it would misreport the stage; it gets no
directory because an empty one would assert material that
does not exist. Its evidence is in the stage index:

* `ttb` — artifacts/stage3/ttb_ctrl_a/run_manifest.json; artifacts/stage3/ttb_treat_a/run_manifest.json; logs/stages/stage-3/history/EXPERIMENTS.md
* `p0_real` — artifacts/audit/three_mode/P0-real-sa; artifacts/audit/three_mode/P0-real-sb; logs/stages/stage-3/history/EXPERIMENTS.md
* `d0` — logs/stages/stage-3/history/EXPERIMENTS.md; artifacts/audit/three_mode/P0-real-sa
* `p0` — configs/stage3/p0/p0_assistant_sa.json; artifacts/stage3/p0_assistant_sa/run_manifest.json; artifacts/stage3/p0_assistant_sb/run_manifest.json; logs/stages/stage-3/history/EXPERIMENTS.md
* `p2` — configs/stage3/p2/p2_ceheavy_sa.json; artifacts/stage3/p2_ceheavy_sa/run_manifest.json; configs/stage3/e4/e4_p2_r1600k_sa.json; logs/stages/stage-3/history/EXPERIMENTS.md

### Arms and aliases, filed with their experiment

| name | is | filed under |
| --- | --- | --- |
| `e8a` | arm — E8 arm A | `logs/stages/stage-3/e8` |
| `p1` | alias — P1 — a name for existing checkpoints | `—` |

Not counted as experiments. See the stage index for why.

## Runs

**0** run(s) are registered for this stage's
experiments. An experiment's plans, analyses, results, history,
validations and runs are all inside its own directory; the
canonical run list, across every stage, is
[`../../index.json`](../../index.json).

## Canonical configs

`configs/` is the source of truth. A run's manifest records
the config path and hash it ran under.

* `configs/stage3/e1/e1_r1600k_sa_pca.json`
* `configs/stage3/e2/e2_d1_sa_pca.json`
* `configs/stage3/e3/e3_a1_frozen_attn_sa.json`
* `configs/stage3/e4/e4_p2_r1600k_sa.json`
* `configs/stage3/e5/e5_c_sa.json`
* `configs/stage3/e6b/e6b_p2_r2960k_sa.json`
* `configs/stage3/e7/e7_control_r1600k_sa.json`
* `configs/stage3/e8/e8_contrib_r2960k_sa.json`
* `configs/stage3/e8b/e8b_dc_r1600k_sa.json`
* `configs/stage3/p0/p0_assistant_sa.json`
* `configs/stage3/p2/p2_ceheavy_sa.json`
* `configs/stage3/recovery.json`
* `configs/stage3/s2v1_from_init.json`

## Canonical data and artifact manifests

A dataset manifest lives beside the data it describes, and
an artifact lives outside git with its manifest. Neither is
copied here.

* `data/stage3_pilot/manifest.json`

## Decisions

* logs/budget/decisions.md — 2026-07-25: Stage 3 sub-stage 2 sizing, fixed-budget A/B from s1@660
* logs/budget/decisions.md — 2026-07-27: Stage 3 sub-stage 2 verdict on mixture v1
* logs/budget/decisions.md — 2026-08-03: Experiment 2 is three sequential 2.96M diagnostics, not a mixture study
* logs/budget/decisions.md — e2_d1_sb_pca@127 holds the best held-out NLL of its lineage with 98.7% degeneration

## Current status

**The stage where the open problem lives.** Recovery reliably restores autonomous *stability* — the student terminates, stays in protocol and stops degenerating — and eleven interventions have not moved reasoning *correctness*. Scale, objective weights, attention restriction, prefix conditioning and extra general-text KD have all been tested; the conclusions and the ones they do **not** support are in `logs/state/experiment_index.md`.

Money, authorizations and what is running right now are not
here: they belong to [`../../budget/`](../../budget/) and
[`../../state/`](../../state/).
