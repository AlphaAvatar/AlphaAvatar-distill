# The operator promotion cycle

A reusable research pattern for improving an initialization operator and finding
out what the improvement is worth. It is written family-neutrally on purpose:
Phase C1–C4 are the first four turns of this cycle on one teacher/student pair,
and nothing in the shape depends on that pair.

```text
operator R&D
  → isolation              (does the new operator beat the incumbent one,
                            with everything else held?)
  → promotion              (a verdict, not a preference)
  → full joint re-search   (given the promotion, what composition is best now?)
  → behavioural selection  (which candidate is actually better, on behaviour?)
  → new incumbent
  → (next operator) ───────┘
```

Each arrow is a decision with its own evidence. The cycle exists because skipping
any one of them produces a claim the project cannot support.

---

## Why each step is there

**Operator R&D** is unpaid engineering. A new formulation is implemented,
registered, and exercised on toy geometry until its shapes, dtypes and device
placement are right. It is not an experiment and it earns nothing.

**Isolation** is the only step that produces an attributable *causal-ish*
claim. One operator varies; the path, the geometry, the teacher, the tokenizer,
the recipe, the battery, the seeds and the decision rule are all held. Paired
fresh seeds, because the incumbent's retained result was produced under the
seeds that selected it and cannot stand in for a fresh arm.

> A joint search can never do this job. It varies everything at once, so a
> better leaf is evidence about a *composition*, never about an operator.

**Promotion** is what an isolation verdict is *for*. Once a verdict promotes an
operator, the loser leaves the searched library — and that exclusion is
recorded, with the verdict as its reason. Re-entering a decided loser as a
search branch spends budget re-deciding a closed question and makes the new
result depend on it.

> If a promotion is ever withdrawn, every exclusion it justified is withdrawn
> with it. An exclusion is only as alive as the verdict behind it.

**Full joint re-search** exists because a promotion invalidates its
neighbourhood. The other operators' calibration assignments were selected in a
world where the promoted operator behaved differently — in the C1→C2 turn, the
replaced ATTENTION operator declared `CalibrationNeed.NONE` and so was offered
*once*, against the no-calibration sentinel, however many mixtures were active.
Assignments chosen under that constraint cannot be assumed optimal after it
lifts.

So the re-search exposes implementations, applicable calibration profiles and
operator **order** jointly, and lets calibration choices affect pruning. It is
not exhaustive enumeration: a beam is fine. What matters is that every
admissible alternative *competes inside the same search*.

**Behavioural selection** is the only step that may name an incumbent. A search
ranks states on a cheap metric, and cheap-metric order is not behavioural order
— this project measured a `-5.22` nat NLL swing that moved behaviour by
`+0.0000`. So the search commits a closed Top-K set and behaviour decides among
them, under a frozen recipe, battery, feasibility floor and equivalence
interval.

> "No winner" is a result. An equivalence interval that nothing clears is a
> finding about the candidates, not a licence to add a seed.

---

## Rules that make the cycle safe to repeat

1. **Freeze what the previous turn measured.** A later turn cites earlier
   evidence; it never rewrites, remeasures or re-ranks it. The previous
   incumbent becomes an *anchor* that advances unconditionally, because the
   question is whether the new candidates beat it.
2. **Derive the space, never state it.** Enumerate from the live registry
   through the same functions the search calls. A product formula silently
   assumes every kind is required and every operator addresses one field, and
   both fail once a composite operator can reach the target in a single step.
   Report the total and the decomposed count separately — they are different
   claims about different sets.
3. **Derive the cost from every run you have.** Pool the per-expansion telemetry
   of every completed search on the same hardware and take the per-cell
   *maximum*. A ceiling built on the cheaper of two observations is not a
   ceiling. An operator that has never run inside a search is priced by a
   stated proxy and named as unmeasured — and the proxy is retired the moment
   it runs.
4. **Bound over beams, not averages.** `children × mean node cost` is what
   under-priced a search here by more than 4×. Enumerate the beam compositions
   the ranking policy could return and take the extremum; optimise minutes and
   expansion counts *separately*, because deferring the expensive operator is
   cheap now and expensive later.
5. **The beam width is part of the design, not a discount.** Policy, objectives
   and ε stay frozen, and so does the standing width. A narrower beam leaves the
   space intact but explores less of it and can return a different front, so
   adopting one is a changed experiment to be registered deliberately —
   **never a way to make an authorization number fit an existing cap.**
   Narrowing the *space* instead turns a joint search back into a restricted
   one, which is a different experiment again.
6. **Do not assume you can resume across sessions.** Content-derived state ids
   are an identity, not the bytes; a multi-gigabyte search workdir that cannot
   be relayed means a fresh resource re-derives what was lost. Unless durable
   cross-session staging has been *implemented and validated*, price a search as
   one session and treat "split it up" as a design project rather than a
   contingency.
7. **Register the candidate rule before the results.** Top-K, the anchors and
   the rung schedule are fixed in advance. A set that can grow once results are
   visible is not a preregistered set.
8. **Price the whole chain before funding the first half.** A search that fits
   the budget and a behavioural stage that does not is a chain that cannot reach
   a verdict. When it does not fit, say so — do not shrink the run to fit.
9. **Every exclusion is a recorded scientific claim.** Cost is not a reason.

---

## What has to be an instance, and what must not be

The arithmetic is generic and lives in
`scripts/experiments/search_cost_model.py`: the space, the branching model, the
leaf enumeration, the bound, the nominated trajectory and the session price. It
names no operator, no profile, no family, no card, no price and no path. Which
operator is "the expensive one" is *derived* from the cost model rather than
written down, so a different family's search nominates its own trajectory
correctly.

An instance supplies exactly the things that are its own:

| the instance owns | example |
| --- | --- |
| the allowed implementations and the exclusions, with reasons | `full_search_space.py` |
| the applicable calibration profiles | which mixtures are materialized |
| the teacher and target geometry | resolved from the frozen spec |
| the cost table's telemetry sources | which committed runs to pool |
| the session phase minutes | setup, transfer, gates, commit, sync |
| the price basis and the budget position | quoted rate, derived cumulative |

**Nothing in this cycle belongs in `src/aadistill`.** The core owns operators,
adapters, calibration, ranking and the search engine — mechanism that a new
family extends by registering, not by editing. Planning and pricing an
*experiment* is an application concern, and a core module that knew a beam width,
a dollar ceiling, a stage number or a repository path would make the framework
depend on the tree it is meant to be reusable inside.

---

## Applying it to a new family, geometry or ratio

Nothing above changes. In practice:

1. register the family's architecture adapter and its operator implementations;
2. declare which calibration profiles are materialized for it;
3. point a `SearchSpace` at that family's teacher and target geometry;
4. pool whatever search telemetry exists for the target hardware into a
   `CostModel` — and if none exists, say the cost is unmeasured and price a
   small pilot rather than a full search;
5. reuse the isolation → promotion → re-search → selection order unchanged.

The first turn on a new family will have no telemetry and therefore no honest
ceiling. That is a reason to run a *pilot*, not a reason to guess: this project
authorized a search at a 1.91–7.51 h projection that then ran 9.08 h without
finishing, and the correction was to stop multiplying averages.
