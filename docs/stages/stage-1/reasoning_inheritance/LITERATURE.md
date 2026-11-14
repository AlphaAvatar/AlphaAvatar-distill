# Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers* -- what it shows, and what it does not

Read 2026-11-13 from arXiv:2403.17887 **v2** (3 March 2025), HTML text. Status: `used` (it frames the questions of
`reasoning_inheritance`); topic `depth-compression`.

## What the paper directly shows

* **Method.** For a block of `n` layers, the angular distance `d = arccos(cos(x_T^(l), x_T^(l+n))) / pi` between the
  inputs of layer `l` and layer `l+n`, **at the final token of each sequence only**, averaged over a neutral dataset
  (10k C4 validation samples in their analysis); the **contiguous** block starting at the minimizing `l` is dropped.
  A simpler heuristic -- drop the deepest layers, never the final one -- is studied too. Optional "healing": QLoRA
  (4-bit weights, LoRA adapters) on 164M or 328M C4 tokens.
* **Models.** Base (pretrained, not instruction-tuned) Llama-2 7B/13B/70B, Qwen-7B and Qwen-14B (the 2023 Qwen 1
  family), Mistral-7B, Phi-2 (2.7B): 32 to 80 layers.
* **QA accuracy.** MMLU (5-shot) and BoolQ stay near the unpruned score up to a model-dependent fraction, then fall to
  chance sharply: about 45-55% of layers for Llama-2, 35% for Mistral-7B, 25% for Phi-2 and **20% for the Qwen family**,
  whose similarity map has "islands" of high similarity among shallow blocks and a "shorter region of robust
  performance". Healing improves QA modestly.
* **Next-token loss.** C4 validation loss rises smoothly with the pruned fraction after healing; without healing it
  reaches chance at about the same fraction as QA. Healing "significantly restores" next-token prediction.
* **Simple heuristic.** Without healing it degrades QA and loss quickly; with healing it nearly matches the
  similarity-informed choice.
* **Reasoning (v2 only, Llama-2-70B only).** GSM8K (CoT, pass@1) and HellaSwag "exhibit immediate degradation in
  performance with any amount of pruning", tracking the loss; CoT-MMLU stays flat like MMLU. The authors read this as
  initial evidence that deeper layers matter for reasoning rather than for knowledge-heavy QA, and list "a pruning
  strategy that is robust for reasoning tasks" as an open question.
* **Not studied:** width or head compression, depth and width combined, matched-parameter comparisons of depth against
  width, instruction-tuned or thinking models, long generated reasoning beyond CoT benchmarks, untrained (unhealed)
  generation quality on reasoning tasks other than the three above.

## Where this round's setting differs

| | the paper | this round |
| --- | --- | --- |
| model | base Llama-2 / Qwen 1 / Mistral / Phi-2 | hybrid thinking model Qwen3-4B (`1cfa9a72`), 36 layers |
| depth selection | contiguous block, final-token angular distance on C4 | **non-contiguous** greedy removal by causal KL on the packed hybrid calibration profile (removed 3, 11, 17, 18, 25, 26, 30, 32) |
| fraction | swept | 8/36 = 22%, near the paper's Qwen-family transition (a different Qwen generation) |
| other compression | none | FFN 9728 -> 3072, hidden 2560 -> 1024, query heads 32 -> 16 (W, DW) |
| healing | QLoRA on C4 | none: Stage 1 models are measured untrained |
| measurement | MMLU / BoolQ log-likelihood; GSM8K and HellaSwag on Llama-2-70B | free generation to 40,960 tokens on eval-set-1 (fluid reasoning, interactive episodes, supporting units), both modes |

## What this round may and may not infer from it

* The paper is evidence that **reasoning-heavy generation is more sensitive to depth than knowledge QA** in one large
  base model. It is not evidence for a 4B hybrid thinking model, for a non-contiguous causal-KL selection, or for depth
  relative to width at any parameter budget; D vs W here answers that for this mapping only.
* Its Qwen observation (a short robust region) makes a large depth loss in D **plausible**, not established; the
  hypothesis "depth is the main cause" is tested by the D/W/DW contrasts, never assumed.
* Healing restoring next-token loss but not QA accuracy past the transition is a caution against reading state
  KL/NLL as reasoning retention -- which P10.2 already requires.

# Layer- and sublayer-merging methods read for the next round's plan

LaCo and MKA read 2026-11-14 from their arXiv HTML (the merge equations verbatim). Verma et al. read again in full on
2026-11-14 (UTC) from the arXiv HTML of **v2** (sections 3-7, Appendix A-C), after review found that the first reading
(2026-11-14, partly from a page summary) misstated what the method executes and what it already covers. Status: `used`
(they define the candidates of REPORT section 11); topic `depth-compression`; Verma et al. also `queued` for Stage 6
(below).

| reference | what was used | what differs here |
| --- | --- | --- |
| Yang, Cao, Zhao, *LaCo: Large Language Model Pruning via Layer Collapse*, arXiv:2402.11187v2 | Eq. 1, `theta*_l = theta_l + sum_{k=1..m} (theta_{l+k} - theta_l)`: the m layers after l are merged into the earlier layer l, attention and MLP alike, no scaling; layers chosen by a cosine-similarity threshold on calibration data; the merged layer replaces the span, so fewer layers execute | the merge direction (into the earlier kept block) and the removal of the absorbed blocks from execution are kept; the additive formula is not registered: on a two-block span it equals keeping the removed block, and it gives no scaling rule |
| Verma, Murray, Duh, *Merging Feed-Forward Sublayers for Compressed Transformers*, arXiv:2501.06126v2 | the alignment of section 3.2 and Appendix A: per window of k adjacent FF sublayers, features from one forward pass over example data (about 10k tokens), the first sublayer of the window the anchor, each other sublayer's neurons permuted onto the anchor's by the linear assignment that maximizes the summed Pearson cross-correlation (Eqs. 1-2, Jonker-Volgenant via scipy); for SwiGLU (Appendix A, used for OLMo) the feature is the output of the component-wise product, so one permutation moves the up and gate rows (`P_i W_i`, `P_i V_i`) and the down columns (`W_i^down P_i^T`) together; then a uniform `1/k` average (Eqs. 3-10) | **what is executed.** Section 3.3 averages the k aligned sublayers, "replace[s] each of them with their average", and ties the weights: the model still executes k FF sublayers, one at each original position between that position's own attention sublayer, with one parameter set in memory. It saves parameters, not depth or FLOPs (section 7: no inference speed-up targeted). This round removes the absorbed blocks from execution (28 blocks run, D's depth), so it borrows the alignment, not the execution scheme. **What is merged.** FF sublayers only; attention sublayers are not merged (section 5.6, Appendix G: attention outputs do not show the FF outputs' similarity). Merging whole blocks, and aligning attention heads by GQA group, is this round's own extension and an untested risk. **Other differences**: activation-importance weights instead of the uniform average in M-act (M-avg is the unaligned uniform average, the paper's "vanilla" baseline); the RMSNorm gains folded first; the spans fixed by D's DEPTH selection instead of a sliding window chosen on validation data; no recovery fine-tuning (Stage 1 measures the untrained model) |
| Liu et al., *Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging* (MKA), arXiv:2406.16330v2 | `theta~_c = alpha * theta_l + (1 - alpha) * theta_m`, alpha from a normalized mutual information of diffusion embeddings (Algorithm 1, verbatim); pairs need not be adjacent; the merged layer replaces the pair | the equal-weight case is M-avg; alpha here is per neuron / head from activation importance, not one similarity per pair |

What Verma et al. report, as written in v2 (not re-derived here): across GPT-2 Large, ViT and an OPUS-MT model the
aligned merge ("Permute FF Merge") is consistently better than the unaligned one ("Vanilla FF Merge"), more so as more
FF sublayers are removed, and close to the uncompressed model at one third of the FF sublayers removed, after recovery
fine-tuning (section 5.1). The layer-pruning baseline (section 4.4) drops a window of whole layers chosen the same way
and gets the same fine-tuning. OLMo-7B (32 layers, SwiGLU, section 4.5 / 5.5, 4-bit QLoRA on SamSum, features on ~10k
Dolma tokens): merging FF sublayers 9-20 gives ROUGE-1/2/Lsum 51.61 / 26.95 / 47.41 against 50.95 / 26.82 / 47.28 for
dropping layers 10-16 and 53.68 / 29.34 / 49.83 uncompressed -- a small gain over dropping, at equal parameter
reduction, after recovery training.

None of the three measures free-generation reasoning of an untrained merged model; their results are perplexity,
MMLU-style or recovered-model numbers and are not carried over as a prediction for this round.

## Stage 6 related work (recorded here; not this round's method)

Verma et al.'s scheme -- align, average, tie, and keep executing the shared parameters at every original position --
is post-training parameter sharing across depth: k executions of one parameter set. That is the object of Stage 6
(AGENTS.md 4.8: shared parameters executed repeatedly), not of Stage 1. For Stage 6 it is a reference for initializing
a shared block from several teacher blocks (alignment before averaging) and a reminder that its gains were measured
after recovery training, at equal parameters but not equal compute: k executions keep the FLOPs of the k sublayers.
A Stage-6 design records its own parameter-, FLOPs- and latency-matched comparisons (AGENTS.md 4.8); nothing here
starts one.
