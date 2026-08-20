# The full search state journal is out of tree

`search_states_reduced.jsonl` (committed) carries all 77 states with every
field intact **except** oversized `steps[].artifacts` blobs — the FFN
activation-importance per-neuron arrays, up to 628 KB in a single step.
Each elided blob is replaced by its byte count and sha256, so the reduction
is checkable rather than merely asserted. Everything that carries scientific
meaning is kept, including each DEPTH step's `trace` with `kept_layers`,
`removal_order` and `removed_layers`.

The complete 25.0 MB journal, sha256 `1ccacb6f3f2ff15f31507270f71782e0e947d58e14258a56da9b01e754c4fe5a`, is at
`/home/ecs-user/aad-artifacts/phase_a_attempt11/search_states.jsonl`.
It is out of tree because AGENTS.md 2.5 admits only artifacts that are
small and reviewable, and 25 MB of per-neuron arrays is neither.
