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
