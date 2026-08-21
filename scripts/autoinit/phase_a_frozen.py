"""The frozen Phase-A identities, with no search code in the module.

These constants were defined in `phase_a_search.py`, which also defines
`run_phase_a_search`. That is fine for the search itself and wrong for anything
that must be **structurally unable** to run one: a recovery continuation that
imports a completed Stage-1 result cannot be allowed to reach the search, and
"we simply do not call it" is a convention, not a guarantee.

So the identities live here and both sides import them. Nothing moved but the
location — every value below is byte-identical to what attempts 11 and 12 ran,
and `phase_a_search` re-exports them so no existing caller changes.
"""

from __future__ import annotations

#: The teacher, pinned. A paid run against an unpinned Hub HEAD measures
#: whatever was published that morning.
TEACHER_ID = "Qwen/Qwen3-4B-Thinking-2507"
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

#: The student geometry every searched leaf must reach exactly. A state that is
#: merely *close* is an intermediate: it usually scores better on teacher KL and
#: cannot be deployed, which is why `is_complete_leaf` compares field for field.
TARGET_GEOMETRY = dict(hidden_size=1024, num_hidden_layers=28, intermediate_size=3072,
                       num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                       vocab_size=151936, tie_word_embeddings=True)

#: The retained canonical initialization, injected as the recovery control by
#: hash. A re-executed composite is not the historical incumbent.
CANONICAL_INIT = "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
CANONICAL_INIT_SHA256 = (
    "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54")

SEARCH_SEED = 20260815

__all__ = ["TEACHER_ID", "TEACHER_REVISION", "TARGET_GEOMETRY",
           "CANONICAL_INIT", "CANONICAL_INIT_SHA256", "SEARCH_SEED"]
