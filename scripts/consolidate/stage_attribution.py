#!/usr/bin/env python3
"""Which pipeline stage each experiment belongs to, and the evidence for it.

    PYTHONPATH=src python scripts/consolidate/stage_attribution.py
    PYTHONPATH=src python scripts/consolidate/stage_attribution.py --json
    PYTHONPATH=src python scripts/consolidate/stage_attribution.py --write

`--write` regenerates `logs/stages/index.json`, the machine-readable stage index
every stage README and `logs/README.md` is rendered from. There is no second,
hand-maintained stage table anywhere.

It states a **current** fact — which stage an experiment belongs to — not the
history of how the tree came to hold it. Where an object used to live is git's
to answer, and the few old paths current tooling must still resolve are a flat
table in `logs/index.json`.

WHAT THIS FILE IS
-----------------
A curated inventory whose every claim is **checked**, not a heuristic. Each row
names concrete repository objects, and `verify()` asserts that each one exists
and that each quoted field really says what the row claims. A row that drifts
from the repository fails loudly instead of quietly becoming fiction.

THE CLASSIFICATION RULES, applied uniformly
-------------------------------------------
Not every directory under `configs/stage<N>/` is an experiment, and not every
experiment has a directory. The kinds are:

* ``experiment`` — its own question, its own result, its own record. This is
  what a stage README lists.
* ``arm`` — a configured variant *inside* one experiment: `e8a` is arm A of
  `e8` (`configs/stage3/e8/arms.json`), and `sa`/`sb` are seed arms. Filed with
  its experiment; never counted as an experiment.
* ``protocol`` — a named training recipe established by one experiment and
  reused by later ones. `p2` is both: the experiment P2-ceheavy established it,
  and `e4`/`e6b` are *different* experiments that reuse it at other rungs.
* ``pipeline-activity`` — the stage's own production work, not a study of it:
  Stage-0 activation collection, the Stage-1 structural init, the Stage-2
  mixture builds, the Stage-3 baseline recovery runs.
* ``engineering-measurement`` — a paid runtime/cost measurement supporting a
  stage's work, which registers `scientific_use: false`.
* ``alias`` — a name for checkpoints that already existed. `P1` retrains
  nothing, so it is not an experiment.
* ``stage-neutral`` — infrastructure owned by no stage.

SUBJECT vs INSTRUMENT — why every autoinit experiment is Stage 1
-----------------------------------------------------------------
Every autoinit experiment touches two stages: it reads a Stage-1 checkpoint and
measures candidates with a Stage-3 recovery battery. Read naively that is
"cross-stage", and it would be wrong — the repository has already settled the
question in a declaration. **`phase_c1` declares `stage_id: "1"`** while doing
exactly that: it selects a Stage-1 initialization operator and measures it with
Stage-3 recovery probes.

The probe is the *instrument*, not the subject. An experiment whose product is a
Stage-1 initialization decision is Stage 1 however it measures; an experiment
whose product is a recovered checkpoint is Stage 3 however it was initialized.
The rule is applied to the one experiment that declares its own stage too, so
declaration and lineage agree on C1 — which is what makes the rule checkable
rather than convenient.

That agreement calibrates the rule; it does **not** attribute the others. Each
autoinit experiment carries its own evidence chain below, naming its own
preregistration, its own subject and its own product. None of them is filed as
Stage 1 because it belongs to the autoinit family.

`cross-stage` MEANS SOMETHING
------------------------------
It means *this experiment genuinely takes several pipeline stages as its
subject*. It does not mean "no `stage_id` field was found". Using it for the
second is how six historical experiments once landed in a bucket that explained
nothing. When no repository fact decides an experiment, this file reports it
``unresolved`` — a finding to go and settle, never a shelf.

SCOPE
-----
Rows cover named experiments, pipeline activities and measurements that exist as
repository objects at HEAD — a config, a log directory, an artifact run
directory, or a numbered section of the historical experiment index. Supporting
assets (evaluation batteries, corpora, calibration sets, smoke runs, probes) are
inputs to experiments, not experiments, and are not enumerated here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

STAGE_INDEX = "logs/stages/index.json"
SCHEMA = "aadistill.logs.stage_index/v1"

#: Stage 3's own historical experiment record. Its numbered sections are the
#: repository's record of what was run before the stage-first layout existed,
#: and several experiments have no other log material — which is why it is
#: retained under the stage that owns it rather than shelved.
INDEX = "logs/stages/stage-3/history/EXPERIMENTS.md"

#: Kinds that count as an experiment in the totals.
EXPERIMENT_KINDS = ("experiment",)


def E(path: str, says: str, **check) -> dict:
    """One piece of evidence: a repository path, and what it says.

    `check` may carry `field` (a dotted path into a JSON document) and `equals`
    (what it must read), which `verify()` enforces.
    """
    return {"path": path, "says": says, **check}


#: ===========================================================================
#: WHAT EACH STAGE IS
#: ===========================================================================
#: `logs/stages/` is the pipeline's stage-level entry point, not a container for
#: run logs. A stage that produced no experiment-run logs is not an empty stage:
#: Stage 0 collected the statistics Stage 1 initializes from, and Stage 2 built
#: the mixtures Stage 3 trains on. A README saying only "no logs yet" reports
#: the absence of one kind of material as the absence of the stage.
#:
#: These rows are the NAVIGATION layer. They say what a stage is for, what it
#: consumes, what it produces, and where its canonical material lives. They do
#: NOT restate revisions, sample counts, hashes, frozen identities or configs:
#: those belong to `configs/`, to the dataset manifests beside their data, and
#: to the artifact manifests, and a second copy here would be one more thing to
#: keep in sync. Every path cited is checked by `verify()`.
STAGES = [
    dict(
        stage_id="0",
        purpose=(
            "Collect the teacher signals a teacher-aware student "
            "initialization needs. Stage 0 trains nothing and produces no "
            "model — its product is statistics."),
        inputs=[
            "the teacher checkpoint `Qwen/Qwen3-4B-Thinking-2507`, pinned by "
            "revision in the collection config",
            "a small, deliberately diverse warm-up corpus — general text, "
            "instructions, grade-school math and short programs — so the "
            "statistics are not fitted to one distribution",
        ],
        data=[
            E("data/warmup/warmup_v1.jsonl",
              "the warm-up corpus itself: five families — FineWeb-Edu, "
              "Dolly-15k, GSM8K, MBPP and a small project-authored set — "
              "deduplicated on content hash"),
            E("data/warmup/warmup_v1.manifest.json",
              "owns the per-source dataset id, revision, license, sample count "
              "and output hash. Read it rather than this page for any number"),
            E("data/warmup/holdout_v1.jsonl",
              "held-out FineWeb-Edu documents, disjoint from the warm-up "
              "corpus, kept for the Stage-1 perplexity gate"),
        ],
        execution=[
            "the teacher runs in BF16 at a 1024-token sequence length, under a "
            "logged activation-cache budget",
            "the collector caches streaming **sufficient statistics**, not raw "
            "activations (decision 2026-07-12) — which is why a cache of this "
            "size can stand in for a corpus pass",
        ],
        outputs=[
            E("artifacts/stage0/qwen3_4b_thinking_v1",
              "the statistics cache Stage 1 initializes from. Outside git; "
              "regenerable from the config and the corpus"),
        ],
        configs=[
            E("configs/stage0/qwen3_4b_thinking.json",
              "the collection recipe: teacher id and revision, dtype, sequence "
              "length, corpus and cache budget"),
            E("configs/stage0/qwen3_4b_thinking_v1.json",
              "the v1 recipe, which produced the cache now in use"),
        ],
        code=[
            E("scripts/training/collect_stage0.py", "the collector"),
        ],
        status=(
            "**Complete.** The v0 cache was lost and v1 was regenerated; the "
            "initialization hash gate then passed, so the Stage-1 result is "
            "unchanged by the regeneration (decisions 2026-08-10). No "
            "experiment ran on top of the collection, so there are no "
            "experiment-run logs — and none are invented."),
    ),
    dict(
        stage_id="1",
        purpose=(
            "Build a complete student checkpoint from teacher structure and "
            "teacher activations instead of random weights, and decide which "
            "initialization operators and operator order to use."),
        inputs=[
            "Stage 0's statistics cache, named by the recipe as `stats_dir`",
            "the same pinned teacher checkpoint",
            "a declared student geometry — 1024 hidden, 28 layers, FFN 3072, "
            "16 query and 8 key/value heads, tied embeddings — which the "
            "recipe states and this page does not own",
        ],
        data=[
            E("data/warmup/holdout_v1.manifest.json",
              'purpose: "Held-out perplexity eval for Stage 1 gate"; the gate '
              "compares the initialized student against a random baseline"),
        ],
        execution=[
            "embedding and lm-head by activation PCA, attention by sandwich "
            "initialization, FFN width by activation-importance selection and "
            "depth by a teacher-span map — the recipe is the source of truth "
            "for which operators a given checkpoint used",
            "the initialization is deterministic from its config and seed, "
            "which is what lets a regenerated Stage-0 cache be checked by hash "
            "rather than by rerunning the comparison",
        ],
        outputs=[
            E("artifacts/stage1/qwen3_0p6b_init_v0",
              "the pinned init checkpoint every Stage-3 recovery run forks "
              "from, plus the random baseline saved beside it for comparison"),
            E("artifacts/stage1/state_eval_v1",
              "the state-evaluation suite the AutoInitializer search scores "
              "candidate initializations with"),
            E("artifacts/stage1/e8_contribution_init_v1",
              "the contribution-guided depth variant, built for E8"),
        ],
        configs=[
            E("configs/stage1/qwen3_0p6b_from_4b_thinking.json",
              "the canonical initialization recipe"),
            E("configs/stage1/qwen3_0p6b_from_4b_thinking_contribution.json",
              "the contribution-guided depth variant E8 initialized from"),
        ],
        code=[],
        status=(
            "**Complete for the canonical recipe; the operator search is the "
            "open work.** `qwen3_0p6b_init_v0` exists and is pinned. Phase A "
            "ended `unresolved_equivalence` with no winner; Phase B resolved "
            "one, but by a margin its own record puts at about 3.6% of a "
            "single correct answer, and its winner is not distinguishable from "
            "Phase A's leader. Phase C1 — the ATTENTION isolation that would "
            "give the first operator-level causal claim — is authorized and "
            "has never been executed. The figures are in "
            "`logs/state/experiment_index.md`, which owns them."),
    ),
    dict(
        stage_id="2",
        purpose=(
            "Prepare the offline data the initialized student is taught with. "
            "**The product of this stage is data.** It trains nothing, and no "
            "training run is invented to make the log tree look symmetrical."),
        inputs=[
            "public instruction, QA, tool-calling, code, math and long-context "
            "datasets, each pinned by revision in the manifest",
            "no teacher-generated targets in v0 or v1 — the manifest records "
            "that explicitly; the teacher-generated corpus is a separate pilot",
        ],
        data=[
            E("data/stage2/stage2_offline_v0.manifest.json",
              "mixture v0: the first grouped build, from public sources only"),
            E("data/stage2_v1/stage2_offline_v1.manifest.json",
              "mixture v1: the approved ~4.5x train scale-up for data-limited "
              "Stage-3 recovery. The current mixture"),
            E("data/stage3_pilot/manifest.json",
              "the teacher-generated pilot: a control/treatment 2x2 over one "
              "shared accepted prompt subset, which is what Stage 3 actually "
              "trains on today"),
        ],
        groups=[
            ("instruction", "general instruction following"),
            ("rag_evidence", "answering from supplied evidence"),
            ("multihop_qa", "questions needing more than one hop"),
            ("tool_calling", "structured / function-call outputs"),
            ("refusal_uncertainty", "unanswerable and under-specified prompts"),
            ("code_math", "programs and grade-school to competition math"),
            ("short_realtime", "short conversational turns"),
            ("long_context", "long documents"),
        ],
        splits=[
            "**train** — what recovery optimizes on",
            "**val** — held out from train, for in-training validation",
            "**calib** — a small per-group sample reserved for calibration and "
            "quantization-sensitivity work, never trained on",
        ],
        sources=[
            "OpenAssistant oasst2, smol-smoltalk, everyday-conversations "
            "(instruction, short_realtime)",
            "SQuAD v2 (rag_evidence, refusal_uncertainty), HotpotQA "
            "(multihop_qa)",
            "Glaive function-calling v2 and xLAM function-calling 60k "
            "(tool_calling)",
            "GSM8K, OpenMathInstruct-2, Magicoder-OSS-Instruct (code_math)",
            "FineWeb-Edu (long_context)",
        ],
        execution=[],
        outputs=[
            E("data/stage2_v1",
              "the v1 mixture on disk: `train/`, `val/` and `calib/` per group. "
              "Untracked; rebuildable from the manifest and its builder"),
        ],
        code=[
            E("scripts/data/build_stage2_v0.py", "the v0 builder"),
            E("scripts/data/build_stage2_v1.py", "the v1 builder"),
            E("scripts/data/build_stage3_pilot.py",
              "the teacher-target pilot builder"),
        ],
        status=(
            "**Complete — v0 and v1 built, and the teacher-target pilot "
            "exists.** Every group's dataset revision, license, sample count "
            "and content hash is owned by the manifests above; the split rule "
            "and the deduplication and holdout-exclusion rules are recorded "
            "there too. Which groups a given recipe should actually train on "
            "is a capability-scope decision (AGENTS.md P3, P10.1), not a "
            "completeness checklist — a group existing here does not mean it "
            "enters a mixture."),
    ),
    dict(
        stage_id="3",
        purpose=(
            "Recover the structurally initialized student offline — repair the "
            "damage compression did — until it can produce usable autonomous "
            "rollouts, which is what Stage 4/5 on-policy work needs from it."),
        inputs=[
            "the pinned Stage-1 init checkpoint "
            "`artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`, which every arm "
            "forks from so that arms differ only in what is under test",
            "the same pinned teacher, loaded to produce KD targets rather than "
            "to be imitated wholesale",
            "Stage-2 data: today the teacher-target pilot corpus, with the "
            "public mixtures behind it",
        ],
        data=[
            E("data/stage3_pilot/manifest.json",
              "the recovery corpus in use: control and treatment arms over one "
              "shared accepted prompt subset, grouped code_math / multihop_qa "
              "/ rag_evidence, with per-arm token and packing statistics"),
            E("data/stage3_pilot/treatment",
              "the treatment arm the canonical config points at by default"),
            E("data/eval_behavior_v0",
              "the behaviour evaluation set the stage is judged on — "
              "autonomous rollout, not held-out loss"),
        ],
        execution=[
            "a run differs from the canonical config only in `data_dir` — the "
            "token-ladder subset it trains on — and in `schedule.total_steps`. "
            "That is what makes the rungs comparable",
            "best-fit packing at an 8192-token block length with gradient "
            "checkpointing; FP32 master weights under BF16 autocast",
            "a combined CE + top-k KD objective; the weights, the KD scope and "
            "the trainable-parameter patterns are declared per config, and "
            "swapping them is what several experiments below are about",
        ],
        outputs=[
            "recovered checkpoints under `artifacts/stage3/<run_name>/`, each "
            "with its own run manifest. Outside git; the checkpoint registry "
            "and the tombstones record which still exist and what was frozen "
            "before any were deleted",
        ],
        configs=[
            E("configs/stage3/recovery.json",
              "the canonical recovery config, runnable as committed. Its "
              "`_purpose` states the contract every arm keeps: a run differs "
              "from it only in `data_dir` and `schedule.total_steps`",
              field="stage", equals="stage3_recovery"),
        ],
        code=[],
        status=(
            "**The stage where the open problem lives.** Recovery reliably "
            "restores autonomous *stability* — the student terminates, stays "
            "in protocol and stops degenerating — and eleven interventions "
            "have not moved reasoning *correctness*. Scale, objective weights, "
            "attention restriction, prefix conditioning and extra general-text "
            "KD have all been tested; the conclusions and the ones they do "
            "**not** support are in `logs/state/experiment_index.md`."),
    ),
]


#: ---------------------------------------------------------------------------
#: STAGE 0 -- initialization warm-up collection (AGENTS.md 4.2)
#: ---------------------------------------------------------------------------
STAGE_0 = [
    dict(
        id="stage0_activation_collection",
        kind="pipeline-activity",
        stage_id="0",
        status="complete",
        title="Teacher activation / statistics collection for student initialization",
        canonical_config="configs/stage0/qwen3_4b_thinking.json",
        evidence=[
            E("configs/stage0/qwen3_4b_thinking.json",
              'declares stage "stage0_init_warmup_collection"',
              field="stage", equals="stage0_init_warmup_collection"),
            E("configs/stage0/qwen3_4b_thinking_v1.json",
              "the v1 collection config, after the cache-loss regeneration"),
            E("scripts/training/collect_stage0.py", "the collector"),
            E("data/warmup/warmup_v1.manifest.json",
              'purpose: "Stage 0 initialization warm-up statistics (not training data)"'),
            E("artifacts/stage0/qwen3_4b_thinking_v1",
              "the regenerated cache this stage produced"),
        ],
        external_material=[
            "configs/stage0/ — collection configs (source of truth; not copied here)",
            "data/warmup/ — dataset and manifests, beside the data they describe",
            "artifacts/stage0/ — the activation cache itself, outside git",
        ],
        decisions=[
            "logs/budget/decisions.md — 2026-07-12: Stage 0 caches streaming "
            "sufficient statistics, not raw activations",
            "logs/budget/decisions.md — 2026-08-10: the Stage 0 activation cache "
            "was lost; regenerate it and prove the initialization is unchanged",
            "logs/budget/decisions.md — 2026-08-10: RESOLVED, the Stage 0 / "
            "initialization hash gate passed",
        ],
        canonical_log_destination=None,
        classification_reason=(
            "Pipeline production work, not a study: it collects the teacher "
            "signals Stage 1 initializes from. It has configs, a collector, a "
            "dataset manifest and an artifact, and no experiment-run logs — so "
            "the stage is represented by a README, and no run, experiment or "
            "manifest is invented to fill it."),
    ),
]

#: ---------------------------------------------------------------------------
#: STAGE 1 -- projection and structural initialization (AGENTS.md 4.3)
#: ---------------------------------------------------------------------------
STAGE_1 = [
    dict(
        id="stage1_structural_init",
        kind="pipeline-activity",
        stage_id="1",
        status="complete",
        title="PCA / sandwich structural initialization of the 0.6B student",
        canonical_config="configs/stage1/qwen3_0p6b_from_4b_thinking.json",
        evidence=[
            E("configs/stage1/qwen3_0p6b_from_4b_thinking.json",
              "the initialization recipe"),
            E("configs/stage1/qwen3_0p6b_from_4b_thinking_contribution.json",
              "the contribution-guided depth variant E8 initialized from"),
            E("artifacts/stage1/qwen3_0p6b_init_v0",
              "the pinned init checkpoint every recovery run forks from"),
            E("data/warmup/holdout_v1.manifest.json",
              'purpose: "Held-out perplexity eval for Stage 1 gate"'),
        ],
        external_material=[
            "configs/stage1/ — initialization recipes",
            "artifacts/stage1/ — init checkpoints and the state-eval suite",
        ],
        decisions=[],
        canonical_log_destination=None,
        classification_reason=(
            "The stage's own production work: it builds the student checkpoint. "
            "Its gate result (holdout NLL 11.748 vs 12.13 random) is recorded in "
            f"{INDEX} §2. No experiment-run logs of its own."),
    ),
    dict(
        id="phase_a",
        kind="experiment",
        stage_id="1",
        status="complete",
        title="AutoInitializer Phase A — greedy search over initialization operator paths",
        canonical_config="configs/experiments/phase_a/source_sets.json",
        evidence=[
            E("logs/stages/stage-1/phase_a/plans/autoinit_phase_a_preregistration.json",
              "its subject is `target_architecture` — the student spec being "
              "initialized — searched over a `search_space` of DEPTH, WIDTH, FFN "
              "and ATTENTION structural operators plus "
              "`composite.stage1_sandwich_v0`. Those are the Stage-1 operators of "
              "AGENTS.md 4.3, and its product is which operator path to use"),
            E("configs/experiments/phase_a/recovery_policy.json",
              "the recovery battery it scores candidates with — the INSTRUMENT; "
              "the search does not produce a recovered checkpoint"),
            E("artifacts/stage1/state_eval_v1",
              "the Stage-1 state-evaluation suite named by the preregistration"),
        ],
        external_material=[],
        decisions=[],
        canonical_log_destination="logs/stages/stage-1/phase_a",
        classification_reason=(
            "Its own preregistration names a student architecture and a space of "
            "structural initialization operators. Subject and product are both "
            "Stage 1; the stage-3 battery is how it measures."),
    ),
    dict(
        id="phase_b",
        kind="experiment",
        stage_id="1",
        status="complete",
        title="AutoInitializer Phase B — does the preferred composition change with the calibration distribution?",
        canonical_config=None,
        evidence=[
            E("logs/stages/stage-1/phase_b/plans/autoinit_phase_b_preregistration.json",
              'hypothesis: "Does the AutoInitializer\'s preferred composition '
              'change when the calibration distribution changes? Phase A searched '
              'one distribution, so its ranking is conditional on it." The '
              'subject is the initialization composition'),
            E("logs/stages/stage-1/phase_b/analyses/autoinit_phase_b_session.json",
              "its own paid sessions, separate from Phase A's"),
        ],
        external_material=[],
        decisions=[],
        canonical_log_destination="logs/stages/stage-1/phase_b",
        classification_reason=(
            "Its registered hypothesis is about the initialization composition's "
            "sensitivity to calibration data. A composition of initialization "
            "operators is a Stage-1 object."),
    ),
    dict(
        id="continuation_b",
        kind="experiment",
        stage_id="1",
        status="complete",
        title="Continuation B — resolving the Phase-B behavioural selection at rung 2",
        canonical_config=None,
        evidence=[
            E("logs/stages/stage-1/continuation_b/plans/autoinit_continuation_b_preregistration.json",
              "`stage1_evidence` is imported COMPLETE with a pinned selection "
              "digest, `no_search_guarantee` is registered, and the session "
              "resolves which initialization candidate wins"),
            E("logs/stages/stage-1/continuation_b/analyses/autoinit_continuation_b_session.json",
              "its own sessions"),
        ],
        external_material=[],
        decisions=[],
        canonical_log_destination="logs/stages/stage-1/continuation_b",
        classification_reason=(
            "It imports Phase B's Stage-1 selection artifact and finishes that "
            "selection. Its product is a Stage-1 decision; it writes no recovered "
            "checkpoint."),
    ),
    dict(
        id="recovery_continuation",
        kind="experiment",
        stage_id="1",
        status="complete",
        title="Recovery continuation — finishing Phase A's search under the repaired harness",
        canonical_config=None,
        evidence=[
            E("logs/stages/stage-1/recovery_continuation/analyses/autoinit_recovery_continuation_session.json",
              "`stage1_source` and `selected_state_ids`: it continues the search "
              "over initialization states and selects among them"),
            E("logs/stages/stage-1/recovery_continuation/analyses/autoinit_recovery_search_v2_build.json",
              "the recovery battery build — the instrument it scores with"),
        ],
        external_material=[],
        decisions=[],
        canonical_log_destination="logs/stages/stage-1/recovery_continuation",
        classification_reason=(
            "It continues Phase A's operator-path search and selects state ids. "
            "`trains_anything: true` describes its probes, which are the "
            "instrument; the product is the Stage-1 path selection."),
    ),
    dict(
        id="phase_c1",
        kind="experiment",
        stage_id="1",
        status="authorized — not launched",
        title="Phase C1 — fixed-path ATTENTION isolation",
        canonical_config="configs/experiments/phase_c1/authorization.json",
        evidence=[
            E("configs/experiments/phase_c1/authorization.json",
              'declares stage_id "1"', field="stage_id", equals="1"),
            E("logs/stages/stage-1/phase_c1/validations/cuda-stage-f",
              "its engineering validation, filed under the experiment it serves"),
        ],
        external_material=[],
        decisions=[],
        canonical_log_destination="logs/stages/stage-1/phase_c1",
        classification_reason=(
            "The only experiment in the repository that declares its own stage. "
            "It also measures with stage-3 confirmation probes, which is what "
            "calibrates the subject-vs-instrument rule for the others."),
    ),
    dict(
        id="measurement",
        kind="engineering-measurement",
        stage_id="1",
        status="complete",
        title="Causal-depth runtime measurement — pricing a Stage-1 operator",
        canonical_config=None,
        evidence=[
            E("logs/stages/stage-1/measurement/analyses/autoinit_measurement_session.json",
              'plan_id "autoinit.causal_depth_measurement"; `scientific_use` is '
              'false, `trains_anything` false, `selects_a_depth_map` false. Its '
              'harness includes the depth operator, recorded under its '
              'then-current path src/aadistill/autoinit/operators/depth.py',
              field="scientific_use", equals=False),
            E("scripts/autoinit/measure_causal_depth_runtime.py", "the measurement"),
            E("src/aadistill/initialization/operators/depth.py",
              "the Stage-1 structural operator whose runtime it prices — the "
              "same module, at the path the initialization-core migration moved "
              "it to"),
        ],
        external_material=[],
        decisions=[],
        canonical_log_destination="logs/stages/stage-1/measurement",
        classification_reason=(
            "Not a scientific experiment — it registers `scientific_use: false` "
            "and answers a cost question. It is Stage 1 because the operator it "
            "prices is a Stage-1 depth-compression operator and the search it "
            "prices for is the Stage-1 autoinit search."),
    ),
    dict(
        id="autoinit_program_material",
        kind="experiment-spanning",
        stage_id="1",
        status="pinned in place",
        title="AutoInit program analyses and harness validations, spanning the Stage-1 experiments",
        canonical_config=None,
        evidence=[
            E("logs/shared/analyses/autoinit_v1_search_space.json",
              "the Stage-1 operator search space, shared by Phase A and Phase B"),
            E("logs/shared/analyses/autoinit_historical_probe_reuse.json",
              "probe reuse across Phase-B and continuation-B attempts"),
            E("logs/shared/validations/depth-backend/autoinit_depth_backend_equivalence.json",
              "equivalence of the depth operator's backends"),
            E("scripts/pod/autoinit_phase_b_driver.py",
              "reads logs/shared/analyses/autoinit_historical_probe_reuse.json at "
              "a hard-coded path, and is named with a digest by consumed Phase-B "
              "authorizations"),
        ],
        external_material=[],
        decisions=[],
        canonical_log_destination="logs/shared/ (BLOCKED — pinned by frozen sources)",
        blocked_by=(
            "scripts/pod/autoinit_phase_b_driver.py and "
            "scripts/pod/autoinit_continuation_b_driver.py read these exact "
            "paths and are named with a digest by consumed Phase-B and "
            "continuation-B authorizations, now under each phase's own "
            "history/superseded_authorizations/"),
        classification_reason=(
            "Stage-1 AutoInit program material that belongs to no single "
            "experiment, and it is NOT stage-neutral. It stays under "
            "`logs/shared/` because roughly twenty-five scripts read these exact "
            "paths, including `autoinit_phase_b_driver.py` and "
            "`autoinit_continuation_b_driver.py`, which consumed Phase-B and "
            "continuation-B authorizations name by digest. Moving the files means "
            "editing frozen-set members to tidy a directory, which is not a trade "
            "this project makes. Declared here as a known exception with its "
            "blocker rather than left looking stage-neutral."),
    ),
]

#: ---------------------------------------------------------------------------
#: STAGE 2 -- offline warm-up data (AGENTS.md 4.4)
#: ---------------------------------------------------------------------------
STAGE_2 = [
    dict(
        id="stage2_offline_mixture",
        kind="pipeline-activity",
        stage_id="2",
        status="complete — v0 and v1 built",
        title="Offline warm-up / distillation mixtures v0 and v1",
        canonical_config="data/stage2_v1/stage2_offline_v1.manifest.json",
        evidence=[
            E("data/stage2/stage2_offline_v0.manifest.json",
              'purpose: "Stage 2 offline warm-up/distillation data for post-init '
              'student recovery (Stage 3+); grouped by training use"'),
            E("data/stage2_v1/stage2_offline_v1.manifest.json",
              'purpose: "Stage 2 offline mixture v1: approved ~4.5x train '
              'scale-up for data-limited Stage 3 recovery"'),
            E("scripts/data/build_stage2_v0.py", "the v0 builder"),
            E("scripts/data/build_stage2_v1.py", "the v1 builder"),
            E("artifacts/stage2/dry_run_v1_report.json", "its build gate"),
        ],
        external_material=[
            "data/stage2/, data/stage2_v1/ — mixtures and manifests, beside the "
            "data they describe",
            "data/stage3_pilot/, artifacts/stage2_v2/ — the teacher-generated "
            "corpus pilot",
            "artifacts/stage2/ — build console and dry-run reports",
        ],
        decisions=[
            "logs/budget/decisions.md — 2026-07-21: Stage 2 offline mixture v0, "
            "grouped public sources, no teacher-generated data",
            "logs/budget/decisions.md — 2026-07-21: offline data rendering, "
            "assistant-span masking, empty-think targets",
            "logs/budget/decisions.md — 2026-07-26: Stage 2 mixture v1, approved "
            "4x scale-up, carry-plus-fresh design, named val sets",
            "logs/budget/decisions.md — 2026-08-01: difficulty-aware mixture "
            "replaces equal four-way balance",
        ],
        canonical_log_destination=None,
        classification_reason=(
            "Data production, not a study of it. Its records are the dataset "
            "manifests, which live beside the data by the same rule that keeps "
            "configs in `configs/`, plus four decision records. The two "
            "2026-07 build notes were folded into the historical index and the "
            "decision log; nothing is reconstructed to give the stage a run."),
    ),
]

#: ---------------------------------------------------------------------------
#: STAGE 3 -- student recovery (AGENTS.md 4.5)
#: ---------------------------------------------------------------------------
def _s3(id, kind, status, title, section, **kw):
    d = dict(id=id, kind=kind, stage_id="3", status=status, title=title,
             canonical_config=kw.pop("canonical_config", None),
             evidence=kw.pop("evidence", []),
             external_material=kw.pop("external_material", []),
             decisions=kw.pop("decisions", []),
             canonical_log_destination=kw.pop("canonical_log_destination", None),
             classification_reason=kw.pop("classification_reason", ""))
    d["index_section"] = section
    assert not kw, kw
    return d


#: Every config directly under `configs/stage3/` declares `stage3_recovery`, so
#: config location settles the stage for everything below it. What each row adds
#: is the *kind*: which of these is an experiment, and which is not.
STAGE_3 = [
    _s3("recovery", "pipeline-activity", "complete",
        "Stage-3 sub-stage 1 — FFN + norm recovery (`s1_ffn_norm`)", "§3",
        canonical_config="configs/stage3/recovery.json",
        evidence=[E("configs/stage3/recovery.json",
                    'declares stage "stage3_recovery", run_name "recovery"',
                    field="stage", equals="stage3_recovery"),
                  E(INDEX, "§3 lists `s1_ffn_norm` (660 steps), holdout 4.21, "
                           "gate passed")],
        classification_reason=(
            "The recovery recipe the stage exists to run, not an experiment "
            "about it. Every E-series experiment forks from its line.")),
    _s3("s2_ab", "pipeline-activity", "complete",
        "Stage-3 sub-stage 2 sizing — freeze-set A/B (`s1_ext_v0` vs `s2_blocks_v0`)", "§3",
        evidence=[E("artifacts/stage3/s1_ext_v0/run_manifest.json", "arm A, the continuation control"),
                  E("artifacts/stage3/s2_blocks_v0/run_manifest.json", "arm B, attention unfrozen"),
                  E(INDEX, "§3: freeze-set sizing; attention-unfrozen adopted")],
        decisions=["logs/budget/decisions.md — 2026-07-25: Stage 3 sub-stage 2 "
                   "sizing, fixed-budget A/B from s1@660"],
        classification_reason=(
            "A sizing decision for the stage's own recovery recipe — which "
            "tensors to unfreeze — taken before the E-series existed. Recorded "
            "as pipeline activity, with its decision record.")),
    _s3("s2v1_from_init", "pipeline-activity", "complete",
        "Stage-3 sub-stage 2 on mixture v1 — the standing branch point", "§3",
        canonical_config="configs/stage3/s2v1_from_init.json",
        evidence=[E("configs/stage3/s2v1_from_init.json",
                    'declares stage "stage3_recovery"',
                    field="stage", equals="stage3_recovery"),
                  E("artifacts/stage3/s2v1_from_init/run_manifest.json", "its run"),
                  E(INDEX, "§3: holdout 3.8285 at step_002700, the standing "
                           "branch point")],
        decisions=["logs/budget/decisions.md — 2026-07-27: Stage 3 sub-stage 2 "
                   "verdict on mixture v1"],
        classification_reason=(
            "The stage's own recovery run on mixture v1. Its checkpoint is an "
            "input to later experiments, which does not make it one.")),
    _s3("ttb", "experiment", "complete — diagnostic, no route claim",
        "Teacher-native vs public target, 2x2 from the Stage-1 init", "§5",
        evidence=[E("artifacts/stage3/ttb_ctrl_a/run_manifest.json", "control arm, seed a"),
                  E("artifacts/stage3/ttb_treat_a/run_manifest.json", "treatment arm, seed a"),
                  E(INDEX, "§5.2: the corrected baseline forked from the pinned "
                           "Stage-1 init, and §5.3 the unrestricted pilot that "
                           "voided its apparent result")],
        classification_reason=(
            "Its own question (which target style), its own arms, its own "
            "verdict. Four run directories, no config directory and no log "
            "directory — it predates both.")),
    _s3("p0_real", "experiment", "complete",
        "P0-real — the first full-scope KD recovery arms", "§16, §17",
        evidence=[E("artifacts/audit/three_mode/P0-real-sa", "its retained evaluation"),
                  E("artifacts/audit/three_mode/P0-real-sb", "second seed"),
                  E(INDEX, "§17.1: \"P0-real's immutable manifests\"; P0-assistant "
                           "differs from it only in `loss.kd_scope`, `run_name`, "
                           "`out_dir`, `_purpose`")],
        classification_reason=(
            "A real training experiment whose checkpoints were later reclaimed. "
            "Its evaluation survives under artifacts/audit/three_mode/, and the "
            "index treats it as the baseline P0-assistant and D0 are measured "
            "against.")),
    _s3("d0", "experiment", "complete",
        "D0 — no-training diagnostics on P0-real-sa/sb", "§16",
        evidence=[E(INDEX, "§16: D0, 2026-08-04, $1.15"),
                  E("artifacts/audit/three_mode/P0-real-sa",
                    "the checkpoints it diagnosed")],
        classification_reason=(
            "Its own section, its own budget line and its own verdict "
            "(\"evidence supports proceeding to the assistant-only KD-scope P0 "
            "arms\"). It trains nothing — a diagnostic experiment is still an "
            "experiment, as E6 also is.")),
    _s3("p0", "experiment", "complete — does not beat P0-real",
        "P0-assistant — assistant-only KD with assistant-token normalization", "§17",
        canonical_config="configs/stage3/p0/p0_assistant_sa.json",
        evidence=[E("configs/stage3/p0/p0_assistant_sa.json",
                    'declares stage "stage3_recovery", run_name "p0_assistant_sa"',
                    field="stage", equals="stage3_recovery"),
                  E("artifacts/stage3/p0_assistant_sa/run_manifest.json", "seed a run"),
                  E("artifacts/stage3/p0_assistant_sb/run_manifest.json", "seed b run"),
                  E(INDEX, "§17: 2026-08-05, $2.75, no arm clears the P0-real "
                           "seed spread")],
        classification_reason=(
            "An experiment: one question (does assistant-only KD scope help), "
            "two seed arms, a registered comparison and a verdict. `sa`/`sb` are "
            "its arms, not two experiments.")),
    _s3("p2", "experiment", "complete — adopted as a protocol",
        "P2-ceheavy — swapping the CE/KD loss weights", "§18",
        canonical_config="configs/stage3/p2/p2_ceheavy_sa.json",
        evidence=[E("configs/stage3/p2/p2_ceheavy_sa.json",
                    'declares stage "stage3_recovery", run_name "p2_ceheavy_sa"',
                    field="stage", equals="stage3_recovery"),
                  E("artifacts/stage3/p2_ceheavy_sa/run_manifest.json", "seed a run"),
                  E("configs/stage3/e4/e4_p2_r1600k_sa.json",
                    "E4 reuses the protocol this experiment established, at "
                    "another rung — a different experiment, same recipe"),
                  E(INDEX, "§18: 2026-08-05, $2.88")],
        classification_reason=(
            "Both an experiment and a protocol. The experiment P2-ceheavy "
            "established the CE-heavy recipe; `e4` and `e6b` are separate "
            "experiments that reuse it, which is why `p2` appearing inside "
            "`e4_p2_r1600k_sa` does not make E4 part of P2.")),
    _s3("e1", "experiment", "complete",
        "Experiment 1 — data-scaling matrix, 24 arms", "§11",
        canonical_config="configs/stage3/e1/e1_r1600k_sa_pca.json",
        evidence=[E("configs/stage3/e1/e1_r1600k_sa_pca.json",
                    'declares stage "stage3_recovery"',
                    field="stage", equals="stage3_recovery"),
                  E("logs/stages/stage-3/e1/analyses/e1_test_cases.md", "its analyses"),
                  E(INDEX, "§11: COMPLETE 2026-08-02, 24 arms, $47.6")],
        canonical_log_destination="logs/stages/stage-3/e1",
        classification_reason="Config location; 24 arms across 6 rungs x 2 inits x 2 seeds."),
    _s3("e2", "experiment", "phase 1 complete; phases 2-3 never authorized",
        "Experiment 2 — three sequential 0.86M diagnostics", "§12",
        canonical_config="configs/stage3/e2/e2_d1_sa_pca.json",
        evidence=[E("configs/stage3/e2/e2_d1_sa_pca.json",
                    'declares stage "stage3_recovery", run_name "e2_d1_sa_pca"',
                    field="stage", equals="stage3_recovery"),
                  E("configs/stage3/e2/e2_d1_sb_pca.json", "the second seed arm"),
                  E("artifacts/stage3/e2_d1_sa_pca/run_manifest.json",
                    "D1 ran: its run manifest and train log are on disk"),
                  E("artifacts/stage3/e2_d1_corpus_audit.json", "its corpus audit"),
                  E("artifacts/stage3/e2_selection_rule_audit.json", "its selection-rule audit"),
                  E("logs/stages/stage-3/e2/plans/PROPOSAL.md",
                    "its phases 2-3 proposal, registered and never authorized; "
                    "two builders name it as their pre-registration"),
                  E(INDEX, "§12: phase 1 COMPLETE 2026-08-04; phases 2-3 unauthorized")],
        decisions=["logs/budget/decisions.md — 2026-08-03: Experiment 2 is three "
                   "sequential 2.96M diagnostics, not a mixture study",
                   "logs/budget/decisions.md — e2_d1_sb_pca@127 holds the best "
                   "held-out NLL of its lineage with 98.7% degeneration"],
        canonical_log_destination="logs/stages/stage-3/e2",
        classification_reason=(
            "A real experiment that RAN — two D1 arms, two audits, a decision "
            "record — and wrote no run logs of its own. Its one surviving "
            "document is the phases 2-3 proposal, which is a plan for this "
            "experiment and now sits in `plans/` under it; the directory holds "
            "that and nothing invented. Its runs remain absent because they "
            "were never recorded, not because they are filed elsewhere."),
    ),
    _s3("e3", "experiment", "complete",
        "Experiment 3 — restricting attention updates at the 0.86M rung", "§20",
        canonical_config="configs/stage3/e3/e3_a1_frozen_attn_sa.json",
        evidence=[E("configs/stage3/e3/e3_a1_frozen_attn_sa.json",
                    'declares stage "stage3_recovery"',
                    field="stage", equals="stage3_recovery"),
                  E("logs/stages/stage-3/e3/analyses/e3_registration.json", "its registration"),
                  E("artifacts/stage3/e3_a1_frozen_attn_sa/run_manifest.json", "A1 seed a")],
        canonical_log_destination="logs/stages/stage-3/e3",
        classification_reason="Config location; `a1`/`a2` are its arms."),
    _s3("e4", "experiment", "complete",
        "Experiment 4 — P2 CE-heavy scaled from the 0.86M to the 1.60M rung", "§21",
        canonical_config="configs/stage3/e4/e4_p2_r1600k_sa.json",
        evidence=[E("configs/stage3/e4/e4_p2_r1600k_sa.json",
                    'declares stage "stage3_recovery"',
                    field="stage", equals="stage3_recovery"),
                  E("logs/stages/stage-3/e4/analyses/e4_registration.json", "its registration"),
                  E("artifacts/stage3/e4_p2_r1600k_sa/run_manifest.json", "seed a run")],
        canonical_log_destination="logs/stages/stage-3/e4",
        classification_reason=(
            "Config location. It reuses the P2 protocol, which is a recipe it "
            "borrows, not the experiment it belongs to.")),
    _s3("e5", "experiment", "complete — 5 aborted attempts, then a result",
        "Experiment 5 — teacher-prefix continuation vs student-prefix recovery", "§22-§27",
        canonical_config="configs/stage3/e5/e5_c_sa.json",
        evidence=[E("configs/stage3/e5/e5_c_sa.json",
                    'declares stage "stage3_recovery"',
                    field="stage", equals="stage3_recovery"),
                  E("configs/experiments/e5/nested_rung.json",
                    "a second config location, added when the rung shape changed"),
                  E("logs/stages/stage-3/e5/analyses/e5_registration.json", "its registration"),
                  E(INDEX, "§27: COMPLETE 2026-08-07")],
        canonical_log_destination="logs/stages/stage-3/e5",
        classification_reason=(
            "Config location. `c`/`r` are its arms and attempts 1-5 are its "
            "runs, not six experiments.")),
    _s3("e6", "experiment", "complete",
        "Experiment 6 — the E1 PCA scale curve normalized onto the frozen battery", "§28",
        evidence=[E("logs/stages/stage-3/e6/analyses/e6_registration.json",
                    '`kind: "evaluation-only"`, `trains_anything: false`, and '
                    'five registered questions. Its arms cite E1\'s configs under '
                    'configs/stage3/e1/, so its subjects are Stage-3 recovery '
                    'checkpoints',
                    field="trains_anything", equals=False),
                  E("logs/stages/stage-3/e6/analyses/e6_results.json", "its results"),
                  E("logs/stages/stage-3/e6/analyses/e6_report.md", "its report"),
                  E(INDEX, "§28: COMPLETE 2026-08-08, $2.36")],
        canonical_log_destination="logs/stages/stage-3/e6",
        classification_reason=(
            "An experiment in its own right, NOT merely the predecessor named in "
            "e6b's provenance: it has its own registration, results and report, "
            "and its own section and budget line. It has no "
            "`configs/stage3/e6/` because it trained nothing and so needed no "
            "training config — its registration names the E1 configs it "
            "re-scored. Stage 3 by the subjects it evaluates."),
    ),
    _s3("e6b", "experiment", "complete",
        "Experiment 6b — P2 CE-heavy at the 2.96M rung", "§29",
        canonical_config="configs/stage3/e6b/e6b_p2_r2960k_sa.json",
        evidence=[E("configs/stage3/e6b/e6b_p2_r2960k_sa.json",
                    'declares stage "stage3_recovery"',
                    field="stage", equals="stage3_recovery"),
                  E("configs/stage3/e6b/provenance.json",
                    "names its parents under configs/stage3/e4/ and configs/stage3/e1/"),
                  E("logs/stages/stage-3/e6b/analyses/e6b_results.json", "its results"),
                  E(INDEX, "§29: 2026-08-09, $7.68, $0.56 over its authorization")],
        canonical_log_destination="logs/stages/stage-3/e6b",
        classification_reason=(
            "Config location. A separate experiment from E6 — it trains at a new "
            "rung, where E6 trained nothing — and separate from P2, whose "
            "protocol it reuses.")),
    _s3("e7", "experiment", "complete",
        "Experiment 7 — general language modelling restored; behaviour unmoved", "§31, §34",
        canonical_config="configs/stage3/e7/e7_control_r1600k_sa.json",
        evidence=[E("configs/stage3/e7/e7_control_r1600k_sa.json",
                    'declares stage "stage3_recovery"',
                    field="stage", equals="stage3_recovery"),
                  E("logs/stages/stage-3/e7/analyses/e7_preregistration.md", "its preregistration"),
                  E("logs/stages/stage-3/e7/history/e7_canary", "its canary session"),
                  E(INDEX, "§34: 2026-08-09, $10.49")],
        canonical_log_destination="logs/stages/stage-3/e7",
        classification_reason="Config location; control/fineweb are its arms."),
    _s3("e8", "experiment", "complete",
        "Experiment 8 — contribution-guided depth initialization", "§35, §36",
        canonical_config="configs/stage3/e8/e8_contrib_r2960k_sa.json",
        evidence=[E("configs/stage3/e8/e8_contrib_r2960k_sa.json",
                    'declares stage "stage3_recovery"',
                    field="stage", equals="stage3_recovery"),
                  E("configs/stage3/e8/arms.json", "defines its arms"),
                  E("logs/stages/stage-3/e8/analyses/e8_step0_report.md", "its step-0 report"),
                  E("logs/stages/stage-3/e8/plans/e8_preregistration.md",
                    "the original 2.96M preregistration, cancelled before execution")],
        canonical_log_destination="logs/stages/stage-3/e8",
        classification_reason=(
            "Config location. It initializes from a contribution-guided depth "
            "map — a Stage-1 operator — and its subject is whether that map "
            "recovers better, which is a Stage-3 question about recovery.")),
    _s3("e8a", "arm", "complete",
        "E8 arm A", "§36",
        evidence=[E("configs/stage3/e8/arms.json", "defines the arms; A is one of them"),
                  E("configs/stage3/e8/artifacts_a.json", "arm A's artifact list"),
                  E("configs/stage3/e8/completion_markers_a.json", "arm A's markers"),
                  E("logs/stages/stage-3/e8/analyses/e8a_session_evidence.json",
                    "its session evidence, filed with E8")],
        canonical_log_destination="logs/stages/stage-3/e8",
        classification_reason=(
            "An ARM of E8, not an experiment: it has no arms file, no "
            "registration and no verdict of its own, and its files are E8's "
            "`*_a.json` half. Not counted in the experiment totals."),
    ),
    _s3("e8b", "experiment", "strategically terminated — no valid comparison",
        "E8b — depth-map x compression interaction", "§37-§42",
        canonical_config="configs/stage3/e8b/e8b_dc_r1600k_sa.json",
        evidence=[E("configs/stage3/e8b/e8b_dc_r1600k_sa.json",
                    'declares stage "stage3_recovery"',
                    field="stage", equals="stage3_recovery"),
                  E("configs/stage3/e8b/arms.json", "its four arms DC/DP/FC/FP"),
                  E("logs/stages/stage-3/e8b/analyses/e8b_analysis.json", "its analysis"),
                  E("logs/stages/stage-3/e8b/plans/e8b_preregistration.md", "its preregistration"),
                  E(INDEX, "§41: DP-sa trained, DC-sa OOM'd, 80 GB is marginal")],
        canonical_log_destination="logs/stages/stage-3/e8b",
        classification_reason=(
            "Config location. `s1`-`s4` are its paid sessions and DC/DP/FC/FP "
            "its arms; a terminated experiment is still one experiment.")),
    _s3("p1", "alias", "n/a",
        "P1 — a name for existing checkpoints", "§17.5",
        evidence=[E("artifacts/audit/three_mode/P1-1600k-sa",
                    "retained under the P1 name"),
                  E(INDEX, "§17.5: \"P1 aliases the P0-real arms, not the "
                           "P0-assistant arms... Nothing is retrained; P1 is a "
                           "name for existing checkpoints\"")],
        classification_reason=(
            "Not an experiment: nothing was trained and no question was asked. "
            "A label over checkpoints that already existed. Not counted."),
    ),
]

#: ---------------------------------------------------------------------------
#: Genuinely stage-neutral: infrastructure whose subject is the machine, the
#: provider or the transport, not any pipeline stage.
#: ---------------------------------------------------------------------------
SHARED = [
    dict(
        id="device_canary",
        kind="stage-neutral",
        stage_id=None,
        status="terminated — two authorized sessions, zero canary runs",
        title="Provider / device canary",
        canonical_config=None,
        evidence=[E("logs/shared/validations/device-canary/autoinit_device_canary_session.json",
                    "its session record")],
        external_material=[],
        decisions=[],
        canonical_log_destination="logs/shared/validations/device-canary",
        classification_reason=(
            "Its subject is whether a rented machine can run the stack at all. "
            "No stage owns it, and any session may need it."),
    ),
    dict(
        id="relay_mirror",
        kind="stage-neutral",
        stage_id=None,
        status="complete",
        title="Artifact relay mirror verification",
        canonical_config=None,
        evidence=[E("logs/shared/validations/relay-mirror/relay_mirror_verification.json",
                    "the verification record")],
        external_material=[],
        decisions=[],
        canonical_log_destination="logs/shared/validations/relay-mirror",
        classification_reason=(
            "Its subject is the artifact transport between the dev box, the "
            "relay and paid pods. Stage 1 and Stage 3 both used it; neither "
            "owns it."),
    ),
]

#: What each experiment asked OF ITS STAGE, in one line. A stage README that
#: lists fifteen identifiers and their status tells a first-time reader how much
#: was done and nothing about what was being found out. Every line below is the
#: question recorded for that experiment in `logs/state/experiment_index.md`;
#: `verify()` requires one for every row classified `experiment`, so a new
#: experiment cannot be added without saying what it is for.
QUESTIONS = {
    "phase_a": "Under one fixed calibration distribution, which initialization "
               "operator path gives the best behavioural starting point?",
    "phase_b": "Does the preferred composition change when the calibration "
               "distribution is allowed to vary?",
    "continuation_b": "Which Phase-B candidate wins once the behavioural "
                      "comparison is carried to rung 2?",
    "recovery_continuation": "Finished under the repaired harness, which of "
                             "Phase A's searched states survive?",
    "phase_c1": "Does a replacement ATTENTION operator beat the frozen "
                "incumbent with every other operator held on a fixed path?",
    "ttb": "Teacher-native targets or public targets — which recovers better "
           "from the Stage-1 init?",
    "p0_real": "What does full-scope KD recovery reach from the pinned init?",
    "d0": "What can the P0-real checkpoints be shown to do without training "
          "anything further?",
    "p0": "Does restricting the KD scope to assistant tokens, normalized over "
          "assistant tokens, help?",
    "p2": "Does swapping the CE/KD loss weights toward CE improve recovery?",
    "e1": "How does recovery scale with the supervised token budget, and does "
          "structural initialization beat random?",
    "e2": "Do targeted recipe variations at the 0.86M rung move behaviour?",
    "e3": "Does freezing or LoRA-restricting attention help recovery?",
    "e4": "Does the CE-heavy objective scale as well as KD-heavy from the "
          "0.86M rung to the 1.60M rung?",
    "e5": "Does conditioning recovery on student-generated prefixes beat "
          "teacher prefixes?",
    "e6": "What does the E1 scale curve look like once every rung is "
          "normalized onto the frozen battery?",
    "e6b": "Does the CE-heavy objective behave differently at the 2.96M rung?",
    "e7": "Does restoring general language modelling improve autonomous "
          "reasoning?",
    "e8": "Does position-based depth compression discard teacher blocks that "
          "are disproportionately important to the teacher's predictions?",
    "e8b": "Is the contribution depth map good on its own, or does it only "
           "fail once composed with width/FFN/attention compression?",
}

#: The field names the stage index publishes. The rows above are authored with
#: short keys because they are read as source; one normalizer turns them into
#: the published shape, so there is exactly one place where a field is named
#: and no consumer has to know both spellings.
def _row(rec: dict) -> dict:
    r = dict(rec)
    out = {
        "experiment_id": r.pop("id"),
        "stage_id": r.pop("stage_id"),
        "classification": r.pop("kind"),
        "status": r.pop("status"),
        "title": r.pop("title"),
        "question": QUESTIONS.get(rec["id"]),
        "canonical_path": r.pop("canonical_log_destination", None),
        "canonical_config": r.pop("canonical_config", None),
        "evidence": r.pop("evidence"),
        "external_material": r.pop("external_material", []),
        "decisions": r.pop("decisions", []),
        "classification_reason": r.pop("classification_reason", ""),
    }
    #: `blocked_by` and `index_section` only appear on the rows that have them.
    out.update(r)
    return out


INVENTORY = [_row(r) for r in (*STAGE_0, *STAGE_1, *STAGE_2, *STAGE_3, *SHARED)]


def _dig(doc, dotted: str):
    for part in dotted.split("."):
        if not isinstance(doc, dict) or part not in doc:
            return KeyError
        doc = doc[part]
    return doc


def _check(root: Path, where: str, evidence) -> list[str]:
    """Every cited path exists; every quoted field reads what is claimed."""
    problems: list[str] = []
    for ev in evidence:
        p = root / ev["path"]
        if not p.exists():
            problems.append(f"{where}: cited path missing: {ev['path']}")
            continue
        if "field" not in ev:
            continue
        if p.suffix != ".json":
            problems.append(f"{where}: field check on non-JSON {ev['path']}")
            continue
        got = _dig(json.loads(p.read_text()), ev["field"])
        if got != ev["equals"]:
            problems.append(f"{where}: {ev['path']}:{ev['field']} is {got!r}, "
                            f"claimed {ev['equals']!r}")
    return problems


def verify(root: Path = REPO_ROOT) -> list[str]:
    """Every cited path exists, and every quoted field says what is claimed.

    This is what separates an inventory from an assertion. A row whose evidence
    has moved, been deleted or changed its mind fails here.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for rec in INVENTORY:
        if rec["experiment_id"] in seen:
            problems.append(f"{rec['experiment_id']}: duplicate id")
        seen.add(rec["experiment_id"])
        if not rec["evidence"]:
            problems.append(f"{rec['experiment_id']}: no evidence cited")
        if not rec["classification_reason"]:
            problems.append(f"{rec['experiment_id']}: no classification reason")
        problems += _check(root, rec["experiment_id"], rec["evidence"])
        cfg = rec.get("canonical_config")
        if cfg and not (root / cfg).exists():
            problems.append(f"{rec['experiment_id']}: canonical_config missing: {cfg}")
        dest = rec.get("canonical_path")
        if dest and "BLOCKED" not in dest and not (root / dest).exists():
            problems.append(f"{rec['experiment_id']}: destination missing: {dest}")
        #: An experiment with no stated question produces a stage README that
        #: lists what ran and not what was being found out.
        if rec["classification"] in EXPERIMENT_KINDS and not rec.get("question"):
            problems.append(f"{rec['experiment_id']}: no question stated")

    #: The stage-level rows carry the same burden as the experiment rows: every
    #: path they point a reader at has to be there, and every quoted field has
    #: to say what is claimed. A navigation layer that has quietly stopped
    #: describing the tree is worse than none, because it is still read.
    stage_ids = {s["stage_id"] for s in STAGES}
    for sid in sorted({r["stage_id"] for r in INVENTORY if r["stage_id"]}):
        if sid not in stage_ids:
            problems.append(f"stage-{sid}: experiments are filed under it and "
                            "it has no stage description")
    for s in STAGES:
        where = f"stage-{s['stage_id']}"
        if not s.get("purpose") or not s.get("status"):
            problems.append(f"{where}: no purpose or no status")
        for key in ("data", "outputs", "code", "configs"):
            problems += _check(root, f"{where}.{key}",
                               [i for i in (s.get(key) or [])
                                if isinstance(i, dict)])
    return problems


def totals() -> dict:
    """Counts that must be mechanically consistent with the rows themselves."""
    exps = [r for r in INVENTORY if r["classification"] in EXPERIMENT_KINDS]
    single = [r for r in exps if r["stage_id"] is not None]
    cross = [r for r in exps if r["stage_id"] is None
             and r.get("stages_involved")]
    neutral = [r for r in INVENTORY if r["classification"] == "stage-neutral"]
    unresolved = [r for r in exps
                  if r["stage_id"] is None and not r.get("stages_involved")]
    return {
        "experiment_total": len(exps),
        "single_stage": len(single),
        "true_cross_stage": len(cross),
        "stage_neutral": len(neutral),
        "unresolved": len(unresolved),
        "identity": ("experiment_total = single_stage + true_cross_stage; "
                     "stage_neutral is counted separately because it is not an "
                     "experiment"),
        "holds": len(exps) == len(single) + len(cross) and not unresolved,
        "unresolved_ids": [r["experiment_id"] for r in unresolved],
        "non_experiment_rows": {
            k: len([r for r in INVENTORY if r["classification"] == k])
            for k in sorted({r["classification"] for r in INVENTORY}
                            - set(EXPERIMENT_KINDS))},
    }


def by_stage() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for rec in INVENTORY:
        key = f"stage-{rec['stage_id']}" if rec["stage_id"] else "shared"
        out.setdefault(key, []).append(rec)
    return {k: out[k] for k in sorted(out)}


def stages_present() -> list[str]:
    """Stages with repository evidence. Never a hardcoded 0-6 range."""
    return [k for k in by_stage() if k.startswith("stage-")]


#: Classifications that are the stage's own work rather than a study of it.
ACTIVITY_KINDS = ("pipeline-activity", "engineering-measurement",
                  "experiment-spanning")


def _runs_by_experiment(root: Path) -> dict[str, int]:
    """How many runs the run index holds per experiment, recorded or not."""
    p = root / "logs/index.json"
    if not p.is_file():
        return {}
    idx = json.loads(p.read_text())
    out: dict[str, int] = {}
    for e in [*idx.get("runs", []), *idx.get("unrecorded", [])]:
        out[e["experiment_id"]] = out.get(e["experiment_id"], 0) + 1
    return out


def stage_rows(root: Path = REPO_ROOT) -> list[dict]:
    """The stage-level view: what each stage is, plus what it actually holds.

    The counts exist because "experiments: 0" reads as an empty stage, and
    Stage 0 and Stage 2 are not empty — they produced the statistics cache and
    the mixtures the later stages consume. Reporting pipeline activity,
    canonical configs and canonical data manifests separately says which kind
    of material a stage produced rather than implying it produced none.
    """
    per_exp = _runs_by_experiment(root)
    out = []
    for s in STAGES:
        sid = s["stage_id"]
        rows = [r for r in INVENTORY if r["stage_id"] == sid]
        exps = [r for r in rows if r["classification"] in EXPERIMENT_KINDS]
        stage_dir = root / "logs/stages" / f"stage-{sid}"
        with_logs = [r for r in exps
                     if (stage_dir / r["experiment_id"]).is_dir()]
        #: The stage's own declared configs, plus whatever its rows name. A
        #: stage may have a config no single experiment claims -- Stage 0's v1
        #: collection config is the one that produced the live cache, and no
        #: experiment row owns it.
        configs = sorted({r["canonical_config"] for r in rows
                          if r.get("canonical_config")}
                         | {c["path"] for c in (s.get("configs") or [])})
        manifests = [d["path"] for d in (s.get("data") or [])
                     if isinstance(d, dict) and d["path"].endswith(".json")]
        rec = dict(s)
        #: JSON has no tuple. A `(group, what)` pair authored as one comes back
        #: from the committed file as a list, so the regeneration check would
        #: report permanent drift against a document that is byte-identical.
        rec["groups"] = [list(g) for g in s.get("groups") or []]
        rec["activity"] = {
            "pipeline_activity": len([r for r in rows
                                      if r["classification"] in ACTIVITY_KINDS]),
            "experiments": len(exps),
            "experiments_with_logs": len(with_logs),
            "runs": sum(per_exp.get(r["experiment_id"], 0) for r in rows),
            "canonical_configs": len(configs),
            "canonical_data_manifests": len(manifests),
            "_note": ("a stage with 0 experiments is not an empty stage: its "
                      "material may be configs, datasets and artifacts, whose "
                      "canonical owners are outside logs/"),
        }
        rec["canonical_configs"] = configs
        rec["canonical_data_manifests"] = manifests
        out.append(rec)
    return out


def document(root: Path = REPO_ROOT) -> dict:
    return {
        "schema": SCHEMA,
        "_what_this_is": (
            "The stage index: which pipeline stage each experiment belongs to "
            "and the evidence for it, generated by "
            "scripts/consolidate/stage_attribution.py. Every stage README and "
            "the logs/README.md stage section is rendered from this file; there "
            "is no second hand-maintained stage table. It states where an "
            "experiment belongs NOW; git history holds where its files used to "
            "be."),
        "_fields": (
            "stage_id | experiment_id | classification | evidence | "
            "canonical_path. `evidence` is the repository objects the "
            "assignment rests on, each with what it says and, where it is "
            "checkable, the field and value verify() enforces. `canonical_path` "
            "is where this experiment's logs live, or null when it produced "
            "none."),
        "_rules": {
            "subject_vs_instrument": (
                "An experiment's stage is decided by its subject and product, "
                "not by the batteries it measures with. phase_c1 DECLARES "
                "stage_id 1 while measuring with stage-3 recovery probes, which "
                "calibrates the rule; each other experiment still carries its "
                "own evidence chain."),
            "cross_stage": (
                "An experiment that genuinely takes several pipeline stages as "
                "its subject would be reported here with stages_involved, and "
                "would get its own directory then. No such experiment exists, "
                "so no such directory does either. It is never a home for an "
                "experiment whose stage has not been established -- that is "
                "reported unresolved."),
            "classifications": (
                "experiment | arm | protocol | pipeline-activity | "
                "engineering-measurement | experiment-spanning | alias | "
                "stage-neutral. Only 'experiment' is counted in the totals."),
            "empty_stages": (
                "A stage appears in logs/stages/ when the repository holds "
                "evidence that it happened, not because 0-6 exist in "
                "AGENTS.md. A stage with no experiment-run logs gets a README "
                "and no fabricated run."),
        },
        "verified": not verify(root),
        "problems": verify(root),
        "totals": totals(),
        "stages_present": stages_present(),
        "stages": stage_rows(root),
        "experiments": INVENTORY,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", action="store_true",
                    help=f"regenerate {STAGE_INDEX}")
    a = ap.parse_args()
    doc = document()

    if a.write:
        p = REPO_ROOT / STAGE_INDEX
        p.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(doc, indent=1) + "\n"
        changed = not p.exists() or p.read_text() != body
        p.write_text(body)
        print(f"{'rewrote' if changed else 'unchanged'} {STAGE_INDEX}")
        return 1 if doc["problems"] else 0
    if a.json:
        print(json.dumps(doc, indent=1))
        return 1 if doc["problems"] else 0

    for stage, rows in by_stage().items():
        print(f"\n=== {stage}")
        for r in rows:
            print(f"  {r['experiment_id']:28s} {r['classification']:24s} {r['status']}")
    t = doc["totals"]
    print(f"\nexperiment total   {t['experiment_total']}")
    print(f"  single-stage     {t['single_stage']}")
    print(f"  true cross-stage {t['true_cross_stage']}")
    print(f"stage-neutral      {t['stage_neutral']}")
    print(f"unresolved         {t['unresolved']} {t['unresolved_ids']}")
    print(f"identity holds     {t['holds']}")
    print(f"other rows         {t['non_experiment_rows']}")
    if doc["problems"]:
        print("\nPROBLEMS:")
        for p in doc["problems"]:
            print("  ", p)
        return 1
    print("\nevidence verified: every cited path exists and every quoted field matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
