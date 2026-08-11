# E8b execution-backend audit — DP/DC, 2026-08-11

Commissioned after S2's registered throughput gate OOM'd on an 80 GB A100
([`EXPERIMENTS.md` §40](EXPERIMENTS.md)). Purpose: select and freeze an efficient,
numerically acceptable execution backend for the DP/DC pair before committing to the
long 1.60M runs. **The scientific experiment is unchanged** — no change to
architecture, loss definition, data, optimizer hyperparameters, trainable parameters,
batch semantics, sequence length, or any comparison.

**Outcome: two changes adopted — the allocator, then the preregistered KD chunk
fallback. The 20-step gate is FALSIFIED and has been replaced.**

The allocator alone was not sufficient. It passed a 20-step gate at 77.15 GiB and the
real run then OOM'd at **step ~110 of 1,761**. `expandable_segments` did what it was
adopted for — fragmentation fell from 6.16 GiB to **1.04 GiB** — but allocated memory
itself kept climbing past the gate's measurement, to 77.37 GiB by step 70 and 77.60 GiB
at the failure. **The 20-step horizon, not the allocator, was the defective part.** That
earlier pass must not be read as a success: it measured a transient, not a steady state.

## 1. What the runtime actually uses

Established from the code and from the pod, not from config or imports.

| component | actual implementation | evidence |
| --- | --- | --- |
| attention | **PyTorch SDPA, dispatching to its flash backend** | both models resolve `_attn_implementation: sdpa` (neither loader passes an override); the failed run's backward emitted `attention_backward.cu:124` *"Flash Attention defaults to a non-deterministic algorithm"*, which only the flash kernel produces. Re-confirmed on the pod: `{"student": "sdpa", "teacher": "sdpa"}` |
| RMSNorm | `Qwen3RMSNorm`, unfused | fp32 upcast, `pow(2).mean`, `rsqrt`, multiply |
| RoPE | `Qwen3RotaryEmbedding`, native | |
| SwiGLU / MLP | `Qwen3MLP` + `SiLUActivation`, unfused | |
| AdamW | `torch.optim.AdamW`, `foreach` | `foreach=None`, `fused=None` → foreach resolves True for CUDA params. Already the multi-tensor path, **not** a per-tensor loop. 309 trainable tensors, 2,826,065,408 parameters |
| KD | `kd_forward_kl`, `chunk=512` | largest temporaries in §4 |
| allocator | **default** before this audit | no `PYTORCH_CUDA_ALLOC_CONF` anywhere in the run path |
| teacher | frozen, no gradient retention | measured on the pod: `training_mode false`, `any_param_requires_grad false`, `use_cache false`; forward under `torch.no_grad()` + autocast |
| student | fp32 weights, bf16 autocast, gradient checkpointing on, `use_cache false` | measured: `weight_dtype torch.float32` |

## 2. Where the memory actually goes — measured

`scripts/training/profile_dp_memory.py`, two real steps of the real DP-sa config on the
A100, with `expandable_segments:True`:

| phase | allocated (GiB) | reserved (GiB) |
| --- | --- | --- |
| teacher + student loaded | 7.49 | 7.50 |
| trainer built (weights only) | 19.47 | 19.48 |
| after step 1 | 51.07 | 62.14 |
| after step 2 | 51.07 | 67.31 |
| **peak** | **62.00** | **67.31** |

Two things this settles:

* **the fixed cost is 51.07 GiB = 54.8 GB**, matching the arithmetic exactly (fp32
  master weights 12.86 + fp32 grads 11.30 + AdamW m,v 22.61 + teacher bf16 8.04 =
  54.82 GB). Grads and optimizer state are **lazily allocated**, which is why
  `trainer_built` shows only 19.47 GiB — weights alone;
* **the concurrent transient peak is ~10.9 GiB, not the ~17.6 GB inferred from the
  OOM trace.** Summing every large buffer overcounts: CE's fp32 upcast is freed before
  KD allocates its copies, so those never coexist. The corrected figure is the one to
  use.

The failed 298 MiB allocation belongs to **KD**, not attention, optimizer, or
anything else: at vocab 151,936 a `chunk=512` fp32 buffer is
512 × 151,936 × 4 B = **311 MB**, and the traceback frame is
`kd_forward_kl → torch.log_softmax(tp[i:i+chunk].float() / temperature)`.

## 3. Candidates — two resolved without spending

| candidate | verdict |
| --- | --- |
| **K0** current backend | reference; OOM'd |
| **K1** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **SELECTED.** The allocator was genuinely at its default, so this is a real change |
| **K2** optimized attention | **already in use** — not tested, per the instruction to test only if the audit proves the current path is not already flash. Precisely: `torch.nn.functional.scaled_dot_product_attention` selecting its **flash backend**. This is *not* the separately packaged FlashAttention/FA2 library, and the two must not be equated; no attention-backend change is requested for E8b |
| **K3** fused RMSNorm / RoPE / SwiGLU | **unavailable in the pinned runtime.** `kernels`, `flash_attn`, `apex`, `liger_kernel` all absent; `use_kernels` requires the hub-`kernels` package. Adding one is a dependency decision (P12), not a low-risk swap, so nothing was installed |
| optimizer | unchanged. Already `foreach`; fused AdamW **not** adopted — it would change the update trajectory for speed alone |

## 4. The vocabulary-KD path dominates, and the chunk was the wrong first lever

At `block_len 8192` × vocab 151,936, every large transient is `[tokens, vocab]`:

| buffer | size | note |
| --- | --- | --- |
| `masked_ce` `sel.float()` | **4,978 MB** | fp32, **unchunked** |
| student logits bf16 | 2,489 MB | autograd-saved |
| teacher logits bf16 | 2,489 MB | |
| `masked_ce` `sel` copy bf16 | 2,489 MB | full copy before the upcast |
| KD `sp` copy bf16 | 2,489 MB | full copy, autograd-saved |
| KD `tp` copy bf16 | 2,489 MB | full copy |
| KD chunk concurrent peak | 933 MB | 3 × 311 MB fp32 |

So reducing the KD chunk 512 → 128 recovers ~0.7 GB — **the smallest of the large
contributors**. `masked_ce` (`train.py:370-372`) makes the same
full-copy-then-fp32-upcast pattern as KD with no chunking at all, and is the single
largest buffer.

**The registered chunk-128 fallback has now been adopted**, after the allocator alone
proved insufficient at step 110. It is applied to the whole depth-only regime — DP-sa,
DC-sa, DP-sb, DC-sb — via `loss.kd_chunk`, never to one arm or one seed, and **not** to
FC. It preserves the KD objective exactly; it is **not bit-identical**, because the loop
accumulates one float32 scalar per chunk so the chunk count changes the reduction order
(~7e-8 relative, `tests/training/test_kd_chunk_invariance.py`). Its ~0.7 GB is a
stopgap, not a fix: the full `[sequence, vocabulary]` tensors remain.

## 5. Numerical equivalence

`expandable_segments` changes how the caching allocator maps segments and nothing
else — no kernel, no reduction order, no dtype. Equivalence is exact by construction,
so the equivalence battery has nothing to measure for K1. It becomes substantive only
if a kernel or the KD chunk is ever changed, and for the chunk the property is already
pinned: `tests/training/test_kd_chunk_invariance.py` records that the objective is
identical while float32 accumulation order is **not** bit-identical (~7e-8 relative),
so any chunk change must apply to both arms of a pair or neither.

## 6. Selected and frozen backend

Chosen on execution properties only — no validation loss or model behaviour was
consulted.

```
attention            PyTorch SDPA, flash backend (transformers default, UNCHANGED)
                     NOT the separately packaged FlashAttention/FA2 — that is a
                     different implementation and no attention change is requested
RMSNorm / RoPE / MLP  Qwen3 native, unfused (no fused package in the pinned runtime)
optimizer            torch.optim.AdamW, foreach (unchanged)
KD                   kd_forward_kl, chunk 128   <- change 2, regime-wide (was 512)
allocator            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   <- change 1
gate                 200 real steps + no-upward-trend + free-margin (was 20 steps)
```

Set once in `scripts/pod/e8b_driver.py` as `ALLOC_CONF`, so the profiler, the gate and
the training run share one allocator configuration. **DP and DC, both seeds, use
exactly this backend.**

### The 20-step gate, and why it is falsified

| quantity | measured at 20 steps | registered limit | |
| --- | --- | --- | --- |
| seconds/step (median of 11) | 4.815 | ≤ 7.86 | pass |
| peak VRAM | 77.15 | ≤ 78.0 | pass, margin 0.85 |
| $/step | 0.002127 | ≤ 0.003472 | pass |

`violations: []` — and the arm then died at step ~110. Trajectory from the real run
(`logs/e8b_dp_sa_train_log_oom.jsonl`, logged every 10 steps):

| steps | `gpu_mem_gb` (GiB, `max_memory_allocated`) |
| --- | --- |
| 10–60 | 77.15 |
| 70–110 | **77.37** |
| ~110–120 | OOM: 77.60 allocated, 1.04 reserved-unallocated, 100.94 MiB free |

The peak was still rising when the gate stopped looking. **The thresholds were not
wrong and have not been relaxed; the horizon was too short and the trend was not
checked at all.**

### Replacement gate — registered before it runs

* **200 real steps**, mechanically chosen to outlast the observed step-110 failure;
* optimizer state and gradients fully materialized by construction (both appear on the
  first step — the profiler shows 19.47 GiB at `trainer_built` rising to 51.07 GiB after
  step 1);
* `log_every: 1`, so the trend has per-step resolution rather than the coarse
  ten-step jumps that hid the climb;
* `max_memory_allocated` **and** `max_memory_reserved` tracked across the whole gate;
* **no upward drift**: the maximum over the final 50 steps may exceed the maximum over
  the preceding 50 by at most **0.05 GiB**;
* **free margin**: the peak must sit at least **1.5 GiB** below the card's real
  capacity, not merely below a constant;
* seconds/step and $/step still inside the unchanged registered budget.

The three registered thresholds are untouched. Two checks are added, which is the
opposite of widening.

### Result — 200 steps, chunk 128, `expandable_segments:True` (PASSED)

| quantity | measured | registered | |
| --- | --- | --- | --- |
| steps run | **200** | 200 | outlasts the step-110 failure |
| seconds/step (median from step 101) | **4.975** | ≤ 7.86 | pass |
| peak VRAM | **77.31 GiB** | ≤ 78.0 | pass |
| $/step | **0.002197** | ≤ 0.003472 | pass |
| VRAM drift over the final 50 steps | **0.000 GiB** | ≤ 0.05 | pass |
| free margin vs real capacity 79.25 | **1.94 GiB** | ≥ 1.5 | pass |

`violations: []`. The series is what a settled peak looks like, and why 20 steps could
not have established it:

```
first: 61.75, 62.00, 62.00, 76.28, 76.28   <- lazy allocation still landing
last:  77.31, 77.31, 77.31, 77.31, 77.31   <- flat
```

Steady state is **77.31 GiB**, held flat, against the failed run's 77.15 -> 77.37 ->
77.60 climb. Chunk 128 moved the peak 0.29 GiB below the failure point and left
**1.94 GiB of real headroom** where the failed run had ~0.3 GiB usable.

**Remaining caveat:** steady state is demonstrated to step 200, not to 1,761.
`save_every: 880` writes a checkpoint mid-run; that happens between steps at the
54.8 GB steady state with ~24 GiB free, so it should be uneventful, but it lies outside
what the gate measured.

## 7. The compressed pair is untouched

FP/FC stay on their hardware-matched L40S path with the unmodified backend. The
allocator setting is applied through the driver, which S4 also uses, but S4 needs only
23–27 GB and its retained FP control trained without it; the setting cannot change FC's
numerics, so within-regime matching holds either way. **No DP/DC-only kernel or chunk
change is applied to FC.**

## 8. For a future standardized training backend

Recorded from this audit, not implemented:

* **the intended long-term direction is sparse Top-K distillation, not full-vocabulary
  KL.** The measured ~10.9 GiB concurrent transient is almost entirely
  `[sequence, vocabulary]` materialization, and the answer is to stop needing the full
  vocabulary at all: distil on a sparse support such as
  `TopK_teacher ∪ TopK_student`, with explicit residual/tail probability handling so
  the mass outside the support is accounted for rather than dropped. Streaming or fused
  computation is an **implementation technique** that may be used to obtain and process
  those sparse logits without retaining full `[sequence, vocabulary]` tensors — it is
  not itself the objective. **Full-vocabulary streaming KL is not the intended default.**
  Either way, shrinking the Python-level chunk is a stopgap: it trades ~0.7 GB for loop
  overhead while leaving the full-vocabulary tensors in place.
* **`masked_ce` should be chunked like KD.** Its unchunked `sel.float()` at 4,978 MB is
  the largest single buffer in the step. Chunking it is the same transformation already
  applied to KD, with the same accumulation-order caveat.
* **fused RMSNorm / RoPE / SwiGLU** would help throughput but not this bottleneck; the
  norms and MLP activations are small beside the vocabulary traffic. Worth considering
  only alongside a dependency decision, since nothing fused is installed.
* **PyTorch SDPA's flash backend is already in use** and needs no work. That is not the
  same thing as the packaged FlashAttention/FA2 library; a future proposal to add FA2
  would be a new dependency with its own numerics to gate, not the enabling of something
  currently switched off. Check any "we should add FlashAttention" claim against this
  audit first.
* **fused optimizer** is a trajectory-level change, not a free speedup; the current
  `foreach` path is already the multi-tensor implementation.
* **autocast caches a bf16 copy of each weight** — the embedding appears as both a
  1,556 MiB fp32 tensor and a 778 MiB bf16 tensor. Across all weights this is a real
  cost worth quantifying before scaling further.
* **FSDP/ZeRO** would address the 54.8 GB fixed cost — fp32 master weights plus fp32
  Adam states for a 3.2B model are 46.8 GB of it. That is the lever for larger students,
  and it changes nothing numerically if sharding is exact.

## 9. Cost

The audit's static half cost **$0**. The measurement was folded into the resumed S2
session so setup was paid once: memory profile ~2.5 min, gate ~3 min. It did not
materially change the E8b projection, and no cap increase was requested or needed.
