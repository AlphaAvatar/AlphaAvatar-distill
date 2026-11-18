# single_operator_budget configs (Stage 1)

| file | what it is | identity |
| --- | --- | --- |
| `design.json` | SOB1: the question and budget point (N ≈ 2), A's method as the baseline, the three arms' geometries and the operators, profiles, reused selections and gates that build them, the calibration counts, the recovery (A's units), evaluation before and after recovery, the registered analysis (three pairs, Holm, serial gatekeeping), the cost model's constants | `status: proposed`; a grant will bind its sha256 as study identity |

Instance facts (geometries, seeds, prices, prefixes) live here and in `scripts/stages/stage-1/single_operator_budget/`,
never in `src/aadistill`. The derived structures and the budget draft are generated into
`logs/stages/stage-1/single_operator_budget/plans/` (`structures.py`, `proposal.py`, each with `--check`). Nothing
here authorizes anything.
