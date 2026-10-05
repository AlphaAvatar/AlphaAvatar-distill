"""Phase A3: the incumbent ATTENTION operator under bsz3 + length_sorted_v1.

A package marker, which most siblings already have and this one did not. It
matters beyond tidiness: `aadistill.governance.closure` resolves an import by
looking for a file, so without this `experiments.phase_a3` was UNRESOLVABLE to the
closure walk — and D1's session legitimately imports `a3_session.path_spec()` for
the frozen target spec and seed. An executable closure that cannot see a real
dependency describes a smaller set than runs, which is exactly what it refuses to
do: deriving D1's closure failed with `unresolved internal import(s):
['experiments.phase_a3']` until this file existed.

Nothing is re-exported. Importing this package has no side effects, which is the
property that lets a closure walk read it safely.
"""
