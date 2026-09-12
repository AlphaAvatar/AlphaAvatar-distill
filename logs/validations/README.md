# logs/validations

Engineering validation: CUDA integration, device placement, dtype, kernels, a
canary, a transport check. Each answers a **hardware or integration question**
and produces no scientific measurement.

One directory per validation subject, versioned inside it.

**This is deliberately not `runs/`.** A formal run executes a frozen scientific
protocol and its result can change a decision; a validation cannot. Mixing them
would let engineering evidence be read as a result — which is why every
validation record here says `authorizes: nothing` and says what it is not.

Costs incurred by a validation are attributed to an approval package by
[`derive_budget.py`](../../scripts/consolidate/derive_budget.py), from an
explicit `package_id` or from the authorization date, and are kept apart from
formal-session spend.
