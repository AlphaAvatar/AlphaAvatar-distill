# Decision records

## 2026-07-12 — Teacher model: Qwen3-4B-Thinking-2507

- **Context:** First dense-model compression experiment needs a teacher. User directed the choice explicitly.
- **Decision:** Use `Qwen/Qwen3-4B-Thinking-2507`, pinned to revision `768f209d9ea81521153ed38c47d515654e938aea`, as the Stage 0 teacher.
- **Alternatives considered:** None weighed; teacher was user-specified.
- **Expected upside:** A capable 4B reasoning ("thinking") teacher that fits in 30 GB RAM on the CPU dev box and downloads to local HF cache (~7.6 GB), enabling CPU-only Stage 0 activation collection before any GPU spend.
- **Risks:** Student architecture target not yet chosen, so compression ratio and deployment precision are undefined. Thinking-style teacher may have long reasoning traces that differ from realtime deployment distribution.
- **Revisit when:** Choosing the student architecture target, or if the teacher proves mismatched to the realtime/quantized deployment goal.

## 2026-07-12 — Stage 0 caches streaming sufficient statistics, not raw activations

- **Context:** Stage 0 must supply the signals for Stage 1 teacher-aware init (grouped activation-PCA hidden projection, activation-importance FFN neuron selection, frequency-weighted embedding PCA). Raw activation dumps over many tokens would be large and grow unboundedly.
- **Decision:** Accumulate streaming sufficient statistics in float64: per residual collection point, token count + sum vector + uncentered second moment `X^T X`; per FFN layer, per-neuron `sum|a|` and `sum a^2`; global token-frequency counts. These are exact sufficient statistics for the intended Stage 1 consumers.
- **Alternatives considered:** (a) Cache raw hidden states — rejected: unbounded size, needs a cap/sampling policy, and Stage 1 only needs second-order stats. (b) float32 accumulation — rejected: residual streams have large-magnitude outlier dims; float32 loses precision in the `E[xx^T] − μμ^T` centering step.
- **Expected upside:** Fixed, small cache (1.95 GB for 36 layers at hidden 2560), O(1) memory in token count, directly consumable by a projection dry run.
- **Risks:** `X^T X` is O(d²) per point (37 × 2560² × 8 B ≈ 1.9 GB); scales quadratically with hidden size and could become large for wider teachers. Second-moment stats are lossy relative to raw activations if a future stage needs higher-order structure.
- **Revisit when:** Moving to a much wider teacher, or if a Stage 1/3 method needs signals beyond second-order statistics.
