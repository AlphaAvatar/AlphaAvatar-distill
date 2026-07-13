# Decision records

## 2026-07-14 — Stage 1 recipe fixes: middle-band depth merging, end-weighted projection fit

- **Context:** The first full Stage 1 init (early-band depth merge, unweighted P over pre-norm points) evaluated *worse than uniform* on held-out text (NLL 17.80 vs random-init 12.13, teacher 2.63). Temperature sweep showed it was not a logit-scale problem (top-1 0.12%; best temperature only flattens to uniform). A single-axis ablation isolated the damage: width-only NLL 11.19, **depth-only (early merge) 10.48**, heads-only 5.47, FFN-only 4.64.
- **Decision:** (1) `depth_span_map` merges pairs in the **middle band** with ~1/5 of surviving 1:1 layers before the band, and uses the **first** layer of each span as representative — depth-only NLL improves 10.48 → **3.88** (late-rep: 7.54). (2) `stream_projection` includes the post-final-norm point and upweights the two end points (weights 9/8 vs 1) because the tied embedding/head interfaces were the worst-captured (0.74/0.75 energy vs >0.94 mid-stream) — width-only top-1 improves 0.036 → 0.082 (NLL 11.19 → 10.83).
- **Alternatives considered:** Late representative for merged spans (worse, 7.54); keeping the early-band merge per a literal reading of "compress late layers less" (collapsed); leaving P unweighted (worse head capture 0.75 vs 0.92); per-group projections (breaks residual-skip consistency; deferred as ablation).
- **Expected upside:** Init checkpoint carries real teacher signal into Stage 3 recovery instead of confident noise.
- **Risks:** End-point weights (9/8) are tuned on a single 10-doc probe; width remains the dominant zero-shot bottleneck (2560→1024 is a 2.5× cut) and PCA is only L2-optimal, not function-aware. Zero-shot init quality is still far from teacher — expected per Minitron-style pruning literature; recovery is Stage 3's job.
- **Revisit when:** Stage 3 recovery underperforms, or a function-aware subspace method (e.g., logit-gradient-weighted PCA) is tested under a fixed budget.

## 2026-07-13 — Student architecture target: ~0.6B-class, Qwen3-0.6B geometry

- **Context:** Stage 1 was blocked on the student architecture decision. User chose among ~1.7B / ~1B / ~0.6B options.
- **Decision:** Student target is **hidden 1024, 28 layers, FFN intermediate 3072, 16 Q heads / 8 KV heads, head_dim 128, tied embeddings, vocab 151936** — the same geometry as Qwen3-0.6B (~0.6B params, ~6.7x compression of the 4B teacher).
- **Alternatives considered:** ~1.7B-class (Qwen3-1.7B geometry, recommended for pipeline de-risking) and ~1B-class (hidden 1536, 24 layers). User selected the aggressive target.
- **Expected upside:** Best realtime latency/memory for the AlphaAvatar deployment goal; Qwen3-0.6B is an exact-geometry open baseline for comparison; small enough to train/recover cheaply.
- **Risks:** 6.7x compression is aggressive — a weak first result may reflect the compression ratio rather than pipeline bugs. Depth 36→28 and width 2560→1024 both compress simultaneously.
- **Revisit when:** Stage 3 recovery plateaus below a useful quality bar, suggesting the ratio (not the method) is the bottleneck.

## 2026-07-13 — Precision policy: BF16 training, INT8 deployment target

- **Context:** AGENTS.md P9 requires choosing deployment numerics before serious training.
- **Decision:** Initialize and recover in BF16; record INT8 (weight, with activation quantization to be decided at Stage 6) as the deployment target. Add fake-quant/INT8 evaluation from Stage 3 onward; quantization-sensitive calibration samples go into Stage 2 data groups.
- **Alternatives considered:** INT4-first (more aggressive, more complexity before the pipeline is proven); defer decision (risks Stage 3 rework).
- **Expected upside:** Simplest proven-numerics path for the first experiment while keeping deployment awareness on the roadmap.
- **Risks:** If the final runtime demands INT4, INT8-oriented recovery may need a QAT follow-up stage.
- **Revisit when:** Stage 6 target runtime/hardware is fixed, or INT8 eval shows unacceptable degradation.

## 2026-07-13 — Warm-up v1: small public-corpus download approved

- **Context:** Warm-up v0 (47 handcrafted samples, 4,068 tokens) supports only ~191 of 2560 covariance directions — statistically thin for PCA-based Stage 1 init. User approved a small public download.
- **Decision:** Build `warmup_v1` (~1M tokens): fineweb-edu sample stream (ODC-By 1.0, general/edu text), databricks-dolly-15k slice (CC-BY-SA 3.0, instruction chat), gsm8k slice (MIT, math reasoning), mbpp slice (CC-BY-4.0, code), plus all 47 handcrafted v0 samples (tool/refusal/RAG coverage). The jsonl is gitignored; the builder script and a manifest with source revisions, licenses, and hashes are committed. Adds the `datasets` library as a dependency (needed anyway for Stage 2 data work).
- **Alternatives considered:** Proceed on thin v0 stats (rank-deficient PCA); defer download and only build Stage 1 code.
- **Expected upside:** ~1M tokens ≫ 2560 dims gives well-conditioned second moments for all residual points and FFN neuron importances.
- **Risks:** fineweb-edu skews educational web prose; mixture is not the deployment distribution. Acceptable for init statistics (not for training data, which is Stage 2's job).
- **Revisit when:** Stage 1 projections look distribution-sensitive, or Stage 2 defines the official data mixture.

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
