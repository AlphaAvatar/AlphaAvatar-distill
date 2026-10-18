# eval-set-1

**Status: `candidate`** — registered 2026-10-18 (maintainer decision of that
date, `logs/budget/decisions.md`). Its identity is pinned and protected: no
item, gold answer, weight, scoring semantic or file changes without a new
identity authorized by the maintainer. It is **not frozen**: the open decisions
below must be settled first, and no agent may settle them by editing items.

The default set for formal capability measurement in every stage
(`data/eval-set/registry.json :: default_formal_measurement`, AGENTS.md P10.3).

## Components

| component | units | items | role | exposure |
| --- | --- | --- | --- | --- |
| `fluid_reasoning_720` | `dev_v1` | 720 (576 static, 144 interactive episodes) | held out: evaluation-split rule families and evaluation-only generators | **development-exposed** (see below) |
| `stage3_capability_846` | `math_verified` 100, `gsm8k` 100, `multihop` 100, `rag` 100, `answerability_paired` 120 (60 pairs), `safety_paired` 100 (50 pairs), `knowledge` 150 (**withheld**), `eval_behavior_v0` 76 | 846 | held out: validation/test splits never eligible for the training corpus; the 76 from the stage2_v1 val split | scored throughout Stage 3; never trained on |
| `stage3_legacy_150` | `items` | 150 | **in-sample regression** | **training prompts**: every item is in the 0.86M rung |

### fluid_reasoning_720

Project-generated procedural tasks (`scripts/stages/stage-0/calibration_volume/tasks/`)
over five axes: inductive rule discovery 144, compositional generalization 144,
task-contained deduction 192, abstract/analogical 96, interactive adaptation 144.
Built 2026-10-18 as Stage-0 `dev_v1`; moved here byte-identically.

- **Development exposure.** It was built to calibrate difficulty and estimate
  variance, and it was scored for the teacher (638/720) and three recovered
  students (0, 0 and 10/720) in `runs/battery_20261018_092345`; the developing
  agent read its items and per-item results. It is held out of every training,
  calibration and search set, but it is **not a never-seen blind set**. A claim
  of untouched-test generalization needs fresh items from the generators' held-out
  seeds, registered as a new set.
- **Reproducible from code.** The generators rebuild the file byte for byte under
  `PYTHONHASHSEED=14`. Under other hash seeds 4 `graph.shortest` items differ,
  because `graph_paths@v1` iterates a set of edges; every frozen instance is still
  self-consistent (`audit/provenance.json`).
- **Quality review.** An independent audit is in
  `audit/fluid_reasoning_720_quality.json`. It re-solved every static item
  from the prompt alone and simulated every episode. 573 of 576 static golds
  were confirmed. Its findings are open decisions, and **no item was changed**:

  | item | defect |
  | --- | --- |
  | `82b5c880000f9eaa2c4e` (order.seating) | gold `Nia` never appears in the prompt; the teacher's correct-from-the-prompt `Ava` scored wrong |
  | `83f5811a86b339da4a0c` (order.seating) | seat count unstated: underdetermined |
  | `f9010f2dd60a31946a20` (graph.unreachable) | gold lists `Elm`, a town on no road |
  | `a68fc81fffe597f45d54` (letter.double_all) | query `ijk` is a worked example (`ijk -> iijjkk`) |
  | `9fba439d2d0b78476103`, `d36815ef540363149b6c` (str.remove_vowels) | "remove only that one vowel" fits every example and gives another answer |

  - 6 of the teacher's 82 failures come from item defects.
  - 0 exact duplicates; about 705 independent items under the audit's
    semantic equivalence.
  - Every episode is solvable within its budget.
  - The scorer rejects some equivalent renderings, for example `\emptyset`
    for "none". That is pinned scoring semantics, recorded rather than
    changed.
- **Scoring.** Static: correct = usable rollout AND the canonicalized last
  `\boxed{}` of the final answer equals the gold. Interactive: correct = the
  environment's own success, re-derived by replaying the recorded answers.

### stage3_capability_846

The frozen `capability-v2` battery (2026-08-03, manifest sha256 `060bdd31…`,
recorded in `logs/state/artifact_manifests.md`) plus `eval_behavior_v0`.
Seven files were copied byte-identically from the durable artifact
`artifacts/stages/stage-3/eval/battery_v2/`; `eval_behavior_v0` moved here from
`data/stages/stage-3/`.

- **Withheld:** `knowledge` (TriviaQA). No redistribution basis — see
  [`NOTICE.md`](NOTICE.md). Its identity is in `manifest.json :: withheld`, and
  readers use the verified local copy when it exists, or report it `withheld`.
- **Known dependence:** `rag` and `answerability_paired` share one SQuAD item.
- **Scoring:** `aadistill.evaluation.capability` per set, `strict_answer.score_numeric`
  for gsm8k, `aadistill.evaluation.behavior.score_sample` for the 76. Both paired
  sets are headlined by **pair** accuracy. Numbers scored before the 2026-08-04
  rescoring used an earlier `capability.py`: compare only rescored numbers.

### stage3_legacy_150

The fixed 150 prompts (inclusion mask `d6e24e0b…`) every Stage-3 arm from E3 to
E8 was scored on. Re-derived here from `corpus_v2` by the original rule and checked
against the binding mask (`stage3_legacy_150/derivation.json`). Only the prompts
and the source gold are stored; the teacher targets are not.

**These are training prompts.** The historical ~0.19–0.21 correctness on this set
(0.1867 the retained reference) is in-sample recall. Use the set only for
regression comparisons between models trained on that rung, and never report it
beside held-out numbers as one score.

## Capabilities and reporting

`manifest.json :: capabilities` maps each capability to the units that measure it.
`scripts/shared/evaluation/eval_set_scoring.py report` writes one row per unit and
per capability and **no pooled number**. A unit the model cannot take is
`unsupported`, a withheld unit without its verified copy `withheld`, an unrun one
`not_run`. None of them counts as zero.

## Verification

- `audit/provenance.json` proves each asset against the record that froze it:
  - file, set and manifest hashes;
  - the 150's mask, rebuilt;
  - every scored run's prompt ids equal to this set's;
  - every historical per-sample verdict reproduced by the current scorers:
    - 144/144 teacher interactive episodes;
    - 620/620 capability rows (the 150 knowledge rows are withheld);
    - 300/300 legacy rows.
- `audit/duplicates.json`: prompt-level duplicates within and across units.
- `audit/isolation.json`: every training, calibration and search source compared
  with every unit.
- `conformance/*.jsonl`: the scoring semantics this set is registered under,
  re-checked by `tests/data/test_eval_set_registry.py` and CI.

## Isolation

`audit/isolation.json` compares every training, calibration and search source
with every unit, on prompt text and content hash.

- **No overlap with any held-out unit:**
  - the Stage-3 recovery corpus (`corpus_v2`, every rung);
  - the Stage-0 calibration sets, base pool and pilot profiles;
  - the D-series state-evaluation suite and calibration profile.
- **`corpus_v2` overlaps exactly the 150 in-sample prompts**, by construction.
- **The Stage-2 `stage2_v1` train split shares 4 prompt texts with
  `eval_behavior_v0`:** two GSM8K problems OpenMathInstruct-2 reproduces,
  `Hi`, and one short smoltalk prompt.
  - This concerns only models trained on `stage2_v1`.
  - Data builders now reserve every held-out unit
    (`shared.eval_sets.held_out_paths()`).

## Open decisions (block freezing)

`registry.json :: open_decisions`, with the context in
`logs/budget/decisions.md` (2026-10-18):

1. the six item defects above: fix, drop, or keep annotated, each a new identity;
2. the generator defects behind them, which must be fixed before any new set
   is drawn from these generators;
3. whether semantically equivalent items count once;
4. the scorer's normalization gaps;
5. TriviaQA's license;
6. the behaviour set's two duplicate pairs and four Stage-2-train overlaps, and
   its seven gated-source prompts;
7. making `.github/CODEOWNERS` binding on `main`.
