# B-E attempt 1 (be_20261028_095332): a complete measurement, verdict FAIL

## Session

- Pod `42dgaejv0tke69`, NVIDIA L40S (driver 595.91.07; 44.39 GiB usable) at $1.09/h.
- 23.68 min, **$0.4366** all in. The pod was deleted at 10:18:31Z, and the provider confirms it is
  gone; the account holds only `am5m373kuq`.
- Setup took 7.3 min from create to `SETUP_DONE`. The PyTorch child ran 714.5 s, then the vLLM
  child ran 131.5 s.
- Evidence was secured off-pod (`evidence.tgz` sha256 `b0c80dab…`, verified). The closeout
  re-derived the verdict from those files, and it agrees with the pod's.

## Checks

| check | status | measured | criterion (fixed in advance) |
| --- | --- | --- | --- |
| initialization | PASS | 902,911,488 parameters (equal to the cost model); 228,237 calibration positions; statistics 421 s, bases 134 s | real materialization on the L40S |
| structural_fold | PASS | max \|Δlogit\| / max \|logit\| = 3.2e-5 (CPU float32, 256 tokens) | ≤ 1e-3 |
| fold_cuda | PASS | folded vs factored, both bf16: max \|Δlogit\| 12.03. Reference, the factored model bf16 vs fp32: 11.22. fp32 vs fp32: 0.0061. 768 positions, 7 eval-set-1 prompts, sdpa, TF32 off | bf16 difference ≤ 2 × the reference error (22.44) |
| trainer_smoke | **FAIL** | `torch.OutOfMemoryError` in the KD loss (`kd_forward_kl`): 44.31 GiB in use (42.56 allocated by PyTorch) of 44.39 GiB. No step time or peak was recorded | 20 steps at the frozen recipe (E1 seed-20260726 config: sequence length 8192, its batch size, micro-batch, bf16 autocast and gradient checkpointing, on-the-fly Thinking-2507 teacher) on the requested device |
| resume | **FAIL** (not measured) | it needs the uninterrupted smoke run to compare with | bitwise resume vs the uninterrupted run |
| checkpoint_identity | PASS | re-saved shards identical; the folded Qwen3ForCausalLM reloads with max \|Δ\| 0.0 | identical bytes and reload |
| gpu_isolation | PASS | the parent never imported torch; the GPU showed 0 MiB and no compute process before and after the engine | ≤ 1024 MiB, no process |
| vllm | **FAIL** | median \|Δ realized log-prob\| vs the HF bf16 reference 0.234 nat (p99 1.90, max 3.17) over 761 prompt positions. Top-1 agreement 0.986 on 295 confident positions. vLLM 0.27.1 loaded the folded checkpoint in 116.6 s | median ≤ 0.1 nat **and** top-1 agreement ≥ 0.98 where the reference leads by ≥ 1 nat (≥ 20 positions) |

## Why this is a result, not a harness failure (P12.1)

**The OOM.** The smoke runs the frozen recipe unchanged on the authorized L40S:
- the trainer, the KD objective and the teacher kept on the same GPU are the recipe's own;
- materialization released its fp32 teacher before the smoke;
- `fold_cuda` deleted its models, and the CUDA cache was emptied after every check.

The allocation that failed belongs to the KD loss over the full vocabulary at sequence length
8192. The $0 estimate (a 42–46 GB peak on a 48 GB card) had flagged exactly this.

**vLLM.** Both sides compute the same function:
- they read the same saved folded bytes and the same token ids;
- both run in bf16;
- the HF reference applies log-softmax to logits upcast to fp32;
- vLLM's own prompt log-probs are compared token by token, after a length check.

The 98.6% confident top-1 agreement confirms they compute one function. The disagreement is
numeric.

`fold_cuda` measures how sensitive this un-recovered initialization is to bf16. Two bf16
implementations of the same function agree on the argmax at only 53–78% of positions (mean
0.656), and the mean \|Δlogit\| is 0.52. **This is a diagnostic only; the frozen 0.1-nat
criterion stands and is not changed.**

## Not retried

Under the frozen protocol on the authorized GPU, both FAILs are deterministic properties of this
validation. A rerun would reproduce them. Every change that might reach a PASS is excluded by the
maintainer's authorization:
- a shorter sequence;
- another batch shape;
- another GPU;
- another threshold;
- another architecture.

Envelope: $0.4366 of $3.00 spent, $2.5634 unused, and no other study may draw on it.
