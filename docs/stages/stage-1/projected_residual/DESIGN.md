# Research B — full-width residual stream with per-layer trainable PCA (900M)

**Status (2026-10-27, revised the same day): $0 READY for an engineering-validation
grant; nothing authorized.**

- The architecture, its initializer, the exact counters and the evaluation
  route are implemented and validated on the host.
- The design and decision rule are drafted. They freeze only on the
  maintainer's approval.
- **900M comes first.** 600M is a reserved design, studied only after 900M's
  result and the maintainer's review.

| record | owns |
| --- | --- |
| [`configs/stages/stage-1/projected_residual/design.json`](../../../../configs/stages/stage-1/projected_residual/design.json) | architecture, selection, calibration, initialization, recovery, evaluation route, what is not matched, the reserved 600M |
| [`configs/stages/stage-1/projected_residual/d4_top1_selection.json`](../../../../configs/stages/stage-1/projected_residual/d4_top1_selection.json) | D4 Top-1's kept layers, heads and FFN neurons, bound to the search journal |
| [`logs/stages/stage-1/projected_residual/analyses/architecture_report.json`](../../../../logs/stages/stage-1/projected_residual/analyses/architecture_report.json) | parameters, FLOPs, KV cache, memory |
| [`logs/stages/stage-1/projected_residual/plans/preregistration.json`](../../../../logs/stages/stage-1/projected_residual/plans/preregistration.json) | comparison P, the decision rule, what would recommend 600M |
| [`plans/cost_forecast.json`](../../../../logs/stages/stage-1/projected_residual/plans/cost_forecast.json), [`plans/application.json`](../../../../logs/stages/stage-1/projected_residual/plans/application.json) | caps and the project-cap consequence |

## 1. Question

**What stays at full width:**
- the teacher's 2,560-wide residual stream;
- its token embedding and tied output head, both frozen;
- every RMSNorm, which works on the full stream.

**What is compressed:** depth, attention and FFN. Each layer `l` reads through
one trainable projection `P_l` [2560, 1024] and writes back through `P_l^T`:

```
h = x + ( attn( RMSNorm_attn(x) @ P_l ) ) @ P_l^T
y = h + ( mlp ( RMSNorm_ffn(h)  @ P_l ) ) @ P_l^T
```

**The question:** does this lose less information than compressing the stream
itself to 1,024, and does it recover better fluid-reasoning correctness than
D4-Augmented under D4-Augmented's teacher, data, recipe and evaluation?

## 2. Architecture (`aadistill.initialization.adapters.projected_residual`)

**Model type:** its own, `aad_projected_residual`:
- the config is Qwen3's plus `internal_size` and `projection_sharing`;
- the model saves and reloads through the Auto classes;
- attention and MLP reuse transformers' Qwen3 modules at the inner width, so
  SDPA, flash and flex attention, the KV cache and packed-sequence masks are
  transformers' own.

**Geometry:**
- 28 layers, inner width 1,024, 16 query / 8 KV heads × 128, FFN 3,072;
- the KV cache is the same 114,688 bytes per token as the 596M student's;
- q/k norms and RoPE are the teacher's.

**Parameters**, counted exactly and equal to a meta-device build:
**902,911,488** total, of which 513,955,328 are trainable and 388,956,160 are
the frozen tied embedding. The 596M student trains 440,467,456.

**Folding:**
- `fold_to_qwen3` multiplies each `P_l` into the sub-layer weights;
- the result is a stock `Qwen3ForCausalLM` of width 2,560 computing the same
  function: 1,490,114,048 parameters, rank ≤ 1,024 per sub-layer;
- that is the evaluation route under vLLM. It is exact in float64;
- in bf16 its difference from the factored model equals bf16's own rounding,
  0.0039 against 0.0042 on the smoke test.

**Orthogonality:**
- `P_l` starts orthonormal and trains freely: no constraint and no regularizer
  in v1;
- `projection_orthogonality` (‖PᵀP − I‖ / √r and the singular values) is logged
  and reported;
- a constraint would be its own declared arm.

## 3. Initialization (`aadistill.initialization.transforms.projected_residual`)

1. **W0** = the teacher with D4 Top-1's selections:
   - kept layers (removed 8, 11, 12, 15, 18, 21, 25, 32);
   - 16 query heads per layer, two per KV group;
   - 3,072 FFN neurons;
   - width unchanged. The selections come from `d4_top1_selection.json`,
     exported from the d4 search journal (`c162a362…`).
2. **Statistics.** Per-layer float64 second moments of the two normalized
   sub-layer inputs, over every real token of `calib.s0v_base_d4@v1`: the
   profile D4's own width step used.
3. **Bases.** `P_l` = the top-1,024 eigenbasis of their trace-normalized mean,
   deterministic, with per-role energy captured recorded.
   - The alternatives, `residual_input` and `read_write`, are implemented and
     computed as diagnostics.
   - The choice is declared now and is not switched on a diagnostic.
4. **Fold into W0.** Readers become `W P_l` and writers `P_l^T W`. Norms, q/k
   norms, embedding and head are W0's.

**Exactness anchors (tests):**
- `P_l = I` at full rank reproduces W0 exactly;
- inputs confined to an r-dimensional subspace give bases that reproduce W0
  exactly at rank r;
- W0 equals the teacher with the unkept heads, neurons and layers deleted.

**One declared bias:** FFN's neurons were chosen on D4's 1,024-wide state.

## 4. Recovery and evaluation

**Recovery** is D4-Augmented's exactly: the merged set (5,923,021 supervised
tokens), 3,702 steps, the recipe, both seeds, KD over every non-padding position.
- **The single change:** the projections are trainable, `\.projection$` added
  to E1's patterns.
- E1's patterns never matched the embedding, so D4-Augmented's tied embedding
  is frozen too.

**Evaluation** is H1-PA's: eval-set-1, every item, nucleus_v1, trained
context 8,192.
- It runs through the folded checkpoint.
- Before any evaluation, a bf16 fold check on eval-set-1 prompts must pass: no
  more than twice the model's own bf16-vs-fp32 difference.

## 5. Not matched, and said so

| | PW-900 | D4 (596M) | ratio |
| --- | ---: | ---: | ---: |
| parameters | 902.9M | 596.0M | 1.515 |
| trainable | 514.0M | 440.5M | 1.167 |
| training FLOPs / supervised position, with the 4B teacher's forward | 21.65 G | 18.37 G | 1.179 |
| the same, student only | 11.19 G | 7.90 G | 1.415 |
| inference FLOPs / token, context 1 | 2.25 G | 1.19 G | 1.884 |
| inference FLOPs / token, context 4,096 | 3.19 G | 2.13 G | 1.494 |
| KV cache / token (bf16) | 114,688 B | 114,688 B | 1.000 |
| weights (bf16) | 1.81 GB | 1.19 GB | 1.52 |

- **The output head dominates** the decode-time difference: 0.78 of 2.25
  GFLOPs per token.
- **P measures the architecture's feasibility and effect at this size.** It is
  never a same-size claim.

## 6. Decision (preregistration)

**P** = PW-Augmented − D4-Augmented, decided. D4-Augmented's two units are
reused: not retrained, not re-evaluated.
- The rule is the H1/mixed one: 570 formal static items × 2 seeds, SESOI 5 pp
  (8 pp per axis), 2% floor, rule-family cluster bootstrap, Holm over the axes.
- **Reachable labels:** D4-Augmented sits at 4.56%, so a pooled GO needs about
  11%. Deduction (13.5%) can reach every label.

**Recommending 600M needs all three:**
- P labelled GO, or deduction GO surviving Holm with P not REVERSED;
- the guard rails held: integrity and per-unit correctness;
- clean training evidence.

The maintainer decides in every case.

## 7. Reserved 600M (design only)

**Geometry:** inner width 512, FFN 1,910, giving 596,035,072 parameters. That is
parameter-matched to 596,049,920 within 14,848.

**Costs that remain:**
- its forward FLOPs are still 1.10× at context 4,096, because the full-width
  head remains;
- 16 × 128-wide query heads read a 512-wide space, a rank bottleneck the 900M
  design does not have.

**Status:** not priced for execution and not authorized.

## 8. Cost and authorization

From [`cost_forecast.json`](../../../../logs/stages/stage-1/projected_residual/plans/cost_forecast.json),
scaled from D4-Augmented's measured sessions:

**B-E** (engineering validation): $1.11 expected, $3.00 cap, on one L40S. It
fits the current $740 cap. One driver,
[`engineering_validation.py`](../../../../scripts/stages/stage-1/projected_residual/engineering_validation.py),
runs every check and writes one record; it trains no student that is kept and
scores nothing. Every threshold, the probe and the numeric identity are fixed
in advance in
[`engineering_validation.json`](../../../../configs/stages/stage-1/projected_residual/engineering_validation.json),
and `verdict` re-derives every status from the evidence and that file alone.

| check | measures | passes when |
| --- | --- | --- |
| initialization | the real materialization; the declared basis's captured energy per layer on the input roles, and on `residual_in` and the sub-layers' outputs (`attn_out`, `ffn_out`) | counted parameters equal the cost model's |
| structural_fold | factored against folded stock Qwen3 on the CPU in float32 (is the fold the same function) | max \|diff\| / max \|logit\| ≤ 1e-3 |
| fold_cuda | both forms ON the L40S, float32 and bfloat16, over 7 fixed eval-set-1 prompts (4 fluid static items, one per axis; 2 gsm8k; 1 math), rendered as the evaluation renders them and hash-checked; TF32 off, sdpa | max \|folded_bf16 − factored_bf16\| ≤ 2 × the factored model's own bf16-vs-fp32, on the L40S; mean \|diff\| and argmax agreement reported beside it |
| trainer_smoke | 20 steps at the recipe's shape (8,192, batch 2, micro-batch 1, bf16 autocast, gradient checkpointing, the teacher on the fly): step time, peak memory, orthogonality drift | finite losses, a gradient on every `P_l`, the tied embedding unchanged |
| resume | uninterrupted against half + checkpoint + reload + half, on CUDA | bit-identical; otherwise no further apart than two uninterrupted runs are (the kernel nondeterminism logged, P5) |
| checkpoint_identity | save → reload → save; the folded model reloaded as stock Qwen3 | identical shard bytes; identical logits |
| gpu_isolation | the PyTorch checks' process exited before vLLM started; nvidia-smi read before the engine | no compute process and ≤ 1 GiB in use; the parent never imported torch |
| vllm | the saved folded checkpoint in vLLM 0.27.1 at the evaluation's resolved context (8,192), greedy, 128 tokens per prompt (an engineering load check, not a P18 measurement); its teacher-forced log-probabilities against HF bf16 on the same ids and card | loads; echoed prompt ids; finite log-probabilities; finish reasons; median \|Δ realized log-prob\| ≤ 0.1 nat and ≥ 98% top-1 agreement where the reference leads by ≥ 1 nat |

- **One process boundary.** The driver's parent never imports torch. The
  teacher, the students and the trainer live in a child process that exits
  before vLLM starts in its own interpreter, so the engine sizes its KV cache
  on an empty card.
- **Every check runs** even after one fails, so one session reports all of them.
- **On the host** (`--tiny`, a random small teacher) the same orchestration runs
  on the CPU. `fold_cuda` keeps its numbers as a rehearsal and is NOT_RUN;
  `gpu_isolation` and `vllm` are NOT_RUN. That dry run is a test
  (`tests/test_be_driver.py`).
- **No result and no GO.** B-E does not settle whether trainable projections
  help reasoning; that is B1's preregistered comparison.

**B1** (formal): $14.53 expected (13.1 L40S hours), $18.52 conservative,
$21.00 cap.
- It runs on am5m373kuq, with no new storage.
- It needs the project cap at $749 or more.

## Reproduce ($0)

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/projected_residual/export_selection.py \
    --write configs/stages/stage-1/projected_residual/d4_top1_selection.json
PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/projected_residual/architecture_report.py \
    --write logs/stages/stage-1/projected_residual/analyses/architecture_report.json
PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/projected_residual/materialize.py --tiny --out /tmp/pw
PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/projected_residual/engineering_validation.py --tiny --out /tmp/pw-be
PYTHONPATH=src:scripts:tests .venv/bin/python -m pytest -q tests/models/test_projected_residual.py \
    tests/initialization/test_projected_residual_init.py scripts/stages/stage-1/projected_residual/tests
```
