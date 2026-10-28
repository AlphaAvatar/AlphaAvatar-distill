# Reasoning difficulty ladder (PLANNED; not implemented, not registered)

**Why.** On eval-set-1, every student this programme has trained sits at or near
the floor on three of the four static axes:

| arm | induction | composition | analogy | deduction |
| --- | ---: | ---: | ---: | ---: |
| D4-Augmented | 0.00% | 0.35% | 0% | 13.5% |
| Mix-Augmented | 0.35% | 0.35% | 0% | 9.3% |
| PCA-Original | 0.35% | 0.70% | 0% | 3.2% |

At those rates two arms cannot be told apart. The fault is not eval-set-1's,
which measures held-out rule families at a fixed difficulty. A graded instrument
can say where along a difficulty axis a student stops succeeding.

## What it would be

**Levels:** for each of five reasoning axes -- AGENTS.md P10.2's four primary axes and abstract/analogical reasoning, a sub-category reported on its own -- five levels L0–L4. Each
level is set by a declared generator parameter:

| axis | the parameter |
| --- | --- |
| deduction | number of premises and chain length |
| induction | examples shown, rule complexity |
| composition | composition depth, number of interacting rules |
| analogy | grid size, number of transformation components |
| interactive | state-space size, hidden-rule complexity, turn budget |

**Sources:** the project's own generators (`s0_generators_v2` families) and the
external modules that expose difficulty parameters (Reasoning Gym, Reasoning
Core).

**Isolation, enforced rather than assumed:**
- only evaluation-only rule families (eval-set-1's generator family split);
- a seed block disjoint from every training, calibration and evaluation seed
  range;
- `shared.eval_sets.require_disjoint` against every training and calibration
  set;
- its paths reserved in `shared.eval_sets.held_out_paths()`, so no data builder
  can admit it;
- never trained or calibrated on, by any arm.

**Calibration:** before any student is measured, the teacher's correctness per
level is measured once, a small paid generation. Levels are tuned so the
teacher spans roughly 95% down to 30%.
- The tuning is frozen before any student is measured.
- It is never re-tuned on student results.

**Size:** about 50 items per axis and level, about 1,250 items in all. That is
under eval-set-1's size, so a measurement costs about 0.7× an eval-set-1 pass.

## What it would not be

- **Not a formal measurement.** It is a diagnostic beside eval-set-1, reported
  per axis and level with no pooled score. It becomes `eval-set-2` only by the
  maintainer's decision (AGENTS.md P10.3).
- **Not retroactive.** Earlier results stay bound to eval-set-1, and no verdict
  is re-read through it.
- **Not a decision input** in Research A or B unless registered, with its
  identity, before their first result.

## Order of work

**$0 steps:**
1. the builder: levels, families, seed block and identity;
2. its isolation audit;
3. the scorers, reusing the generators' own gold and the formal v2
   canonicalization.

**Paid steps:**
4. the teacher's per-level pilot;
5. the maintainer's registration decision.

None is started. The time and cost for each step will be stated when it is
proposed.
