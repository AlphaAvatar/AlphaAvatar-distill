# E8b — depth-map × compression interaction: preflight and paid-compute proposal

**Status: PREFLIGHT COMPLETE, NOT AUTHORIZED. No GPU has been used for E8b.**
Both depth-only initializations are built and verified at $0; the cost, the GPU
class and the memory sizing are below. Nothing paid launches without a separate
authorization.

**E8a is closed and is not reopened.** Its frozen map, its 3.11× result and its
step-0 dissociation stand as recorded in [`e8_step0_report.md`](e8_step0_report.md)
and [`EXPERIMENTS.md`](EXPERIMENTS.md) §36. E8b does not alter, rerun or reinterpret
it.

**Question.** Is the contribution-guided depth map better when depth is the only
compression, and does it become worse only when composed with the existing
width/FFN/attention compression?

---

## 1. The factorial

```
                            DEPTH MAP
                    positional        contribution
                  ┌───────────────┬───────────────┐
depth-only        │  DP           │  DC           │  3,215,021,568 params
(teacher width)   │  2 seeds      │  2 seeds      │
                  ├───────────────┼───────────────┤
fully compressed  │  FP           │  FC           │    596,049,920 params
(target student)  │  retained     │  2 seeds      │
                  └───────────────┴───────────────┘
```

Every cell trains the canonical **E1/P1 KD-heavy 1.60M** recipe — 1,600,353 unique
supervised CE tokens, three exposures, 1,761 steps, CE 0.25 / KD 1.0, τ 1.0, scope
all, lr 5e-5, warmup 88, 2 blocks/step, `block_len` 8192, seeds `sa` 20260726 and
`sb` 20260801, identical data, block order, optimizer, scheduler, trainable-pattern
set and frozen evaluation protocol. **Not 2.96M.**

## 2. DP and DC are built, verified, and hashed — at $0

`scripts/training/build_depth_only_init.py`. Depth is the only compression: teacher
hidden 2560, FFN 9728, 32 Q / 8 KV heads, head_dim 128, embeddings, tied lm head,
norms, vocabulary and tokenizer all carried over untouched. 36 → 28 layers, by
verbatim `state_dict` copy of the kept blocks — no projection, no norm folding, no
head or neuron selection.

| | DP | DC |
| --- | --- | --- |
| removed teacher layers | `[5,7,9,11,13,15,17,19]` | `[2,3,15,16,20,21,26,32]` |
| parameters | 3,215,021,568 | 3,215,021,568 |
| `model.safetensors` sha256 | `d4db65eb8f7ae6d8a847c2db9a9e5e307e449f50f3bd129e07a1b20f6ec5f3cd` | `eb9e95481988b296a77c30d7b4754069f1874330fca9ad198f4457029e11e182` |
| config sha256 | `4e5b71040b0badb8e9b3f1c58f99ef6d3e69723612ab69ee783a8ba56887ad82` | **identical** |
| resolved RoPE base | 5,000,000 | 5,000,000 |
| bf16 checkpoint | 6.43 GB | 6.43 GB |

**Both verified bitwise identical to the ablated teacher** — max absolute logit
difference exactly `0.000e+00` against `bypassed_blocks(teacher, removed)`, checked
on the reloaded checkpoint.

### 2.1 What that identity means, and it sharpens the design

`bypassed_blocks` is the exact operation E8a's search objective measured. **DP and
DC therefore *are* the ablated teachers E8a scored, materialized as checkpoints.**
E8a's own numbers — calibration KL 1.932531 for DP's map, 0.620586 for DC's — are
already step-0 statements about these two models on the calibration set, and DC
starts 3.11× closer to the teacher there.

E8b's step-0 measurements extend that from the calibration set to the *evaluation*
streams, and add autonomous behaviour, which E8a never measured. So the depth-only
half of §7's case analysis has a strong prior: **DC is expected to beat DP at step
0.** The informative unknown is whether that survives 1.60M recovery, and whether
the sign flips under full compression.

### 2.2 A verification finding worth recording

`Qwen3ForCausalLM(cfg).to(bfloat16)` casts the rotary `inv_freq` **buffer** to
bf16, while `from_pretrained` recomputes it in fp32. The in-memory model therefore
ran a lower-precision positional basis and differed from the ablated teacher by
**0.78** in logits; the reloaded checkpoint is identical. `inv_freq` is
non-persistent, so no saved checkpoint is affected — including every Stage 1
artifact built through `build_student`, whose blanket `.to(dtype)` does the same
thing. Only an init-time in-memory forward is affected. Recorded because it looks
exactly like a construction bug and is not one.

## 3. FP — the retained control identity

```
FP = E1/P1 KD-heavy 1.60M from the canonical positional PCA init
     artifacts/stage1/qwen3_0p6b_init_v0/checkpoint, model.safetensors 86fbba78…
     arms e1_r1600k_{sa,sb}_pca @ step_001761
     relay e1_scaling_20260801/, sha256 6f77676ab8fde397… / e432d57e598d57e1…
     frozen battery artifacts exist from E6, mask d6e24e0b09da1bcc…
     behaviour: usable 0.7300, correct 0.1867, correct|usable 0.2511
```

**Compatibility is only partly proven, and that is a decision for you.** FP's
weights were trained on an **L40S**; E8b's arms would train on an **A100**. The
interaction term subtracts `(FC − FP)` from `(DC − DP)`, so a hardware confound in
FP propagates into the headline number. Two options are priced in §6:

* **RETAINED-FP** — reuse FP as-is, document the hardware difference as a declared
  limitation of the interaction estimate. $42.26 backstop.
* **FULL** — retrain FP on the same A100, making all four cells hardware-matched,
  and keep the retained L40S arms as a free cross-check on how much hardware moves
  a matched recipe (itself a useful measurement). $48.11 backstop.

I recommend **FULL**: $5.85 more buys the interaction term its cleanest reading,
and the retained-vs-retrained comparison is worth having on record.

## 4. Mandatory initialization measurements

The project rule is unchanged and is enforced by `nll_gate.require_init_nll`: an
initialization checkpoint is not complete until its own NLL artifact exists, bound
to that checkpoint's recomputed hash, with every required series present. **All four
initializations are measured in one session, on one device, by one evaluator** —
DP, DC, FP and FC — so the step-0 table is directly comparable. FP's NLL is
**remeasured** even though the checkpoint is pinned and was measured on 2026-08-11;
that measurement was on a different card.

Per initialization: `holdout_v1` NLL · `fineweb_val_e7` NLL, teacher→student KL,
top-1, mean rank, entropy · `teacher_native_val` NLL, KL, top-1, mean rank ·
config hash · parameter count · resolved RoPE base · environment.

FC's and FP's numbers from the E8a session are retained for continuity but are
**not** substituted: `fineweb_val_e7` NLL 14.3913 (FC) and 11.5749 (FP) on an
L40S, records `50863410fa170683…` and `a40feef0dd1535aa…`.

## 5. Step-0 autonomous probe — on a separate battery

DP and DC are 5.39× the target student and much closer to the teacher, so they may
already generate usefully at step 0. Both are probed before any training.

**Battery: `data/eval_behavior_v0`** — 76 prompts across 7 groups (instruction 12,
rag_evidence 12, tool_calling 12, refusal_uncertainty 12, code_math 12,
short_realtime 12, multihop_qa 4), an existing frozen mechanically-scored
diagnostic. Deliberately **not** the 150-prompt promotion battery, which is sampled
from the 0.86M rung and is the E8b endpoint; using it as a step-0 search signal is
what §5 of the instruction warns against. Generation settings identical to the
promotion protocol: greedy, system message mandatory, thinking never suppressed,
no artificial cap (allowance = trained `block_len` 8192 − prompt), complete raw
generations saved.

Three levels of evidence, kept separate:

```
full-width teacher-ablation KL (E8a)   DP 1.932531   DC 0.620586
        ↓
depth-only checkpoint at step 0        measured in E8b, diagnostic
        ↓
depth-only after 1.60M recovery        the E8b conclusion
```

Step-0 behaviour is **diagnostic**. Formal E8b conclusions come only from the
matched recovered models on the frozen promotion battery.

## 6. Hardware, memory, and cost

### 6.1 Memory sizing — the L40S is ruled out

`scripts/training/size_e8b_memory.py`, under the canonical semantics (float32
master weights, bf16 autocast, gradient checkpointing on, embeddings and tied lm
head frozen, one microbatch of one 8,192-token block, SDPA attention):

| term | DP/DC (3.215B) | FP/FC (596M) |
| --- | ---: | ---: |
| params, float32 | 12.86 GB | 2.38 GB |
| gradients, float32 (trainable only) | 11.30 GB | 1.76 GB |
| Adam states, float32 ×2 | 22.61 GB | 3.52 GB |
| **student state** | **46.77 GB** | **7.67 GB** |
| teacher, bf16, resident | 8.04 GB | 8.04 GB |
| student + teacher logits | 4.98 GB | 4.98 GB |
| KD/CE float32 reduction | 0.93 GB | 0.93 GB |
| activations (checkpointed + recompute) | 2.68 GB | 1.47 GB |
| **expected peak** | **63.41 GB** | **23.09 GB** |
| **+15% allocator/workspace margin** | **72.92 GB** | **26.56 GB** |

| card | VRAM | DP/DC | FP/FC |
| --- | ---: | --- | --- |
| L40S | 48 GB | **NO** | yes |
| A100 80GB | 80 GB | yes | yes |
| H100 80GB | 80 GB | yes | yes |
| H200 | 141 GB | yes | yes |

Assumption stated because it is load-bearing: SDPA/flash means the O(L²) score
matrix is never materialized. An eager fallback would add ~4,295 GB and is
impossible, so the setup must assert the attention implementation.

### 6.2 GPU selection — live RunPod secure pricing, 2026-08-11

| GPU | VRAM | secure $/h | stock |
| --- | ---: | ---: | --- |
| RTX PRO 6000 MaxQ | 96 | 0.50 | Low |
| A100 PCIe | 80 | 1.39 | Low |
| **A100 SXM** | **80** | **1.59** | **High** |
| RTX PRO 6000 | 96 | 2.09 | High |
| H100 PCIe | 80 | 2.89 | Low |
| H100 SXM | 80 | 3.29 | High |
| H200 SXM | 141 | 4.59 | High |
| *L40S (E8a's card)* | *48* | *0.99* | *Low* |

**Selected: A100 SXM 80GB @ $1.59/h.** Cheapest ≥70 GB class at High stock. The
$0.50 MaxQ is power-limited and Low stock; A100 PCIe is $0.20 cheaper but Low stock
and slower interconnect. No semantics are changed to fit a cheaper card — no LoRA,
no extra freezing, no quantized optimizer states, no objective change.

Single-GPU, no distributed configuration: 72.9 GB fits one 80 GB card, so DP and DC
share an identical single-device setup by construction.

### 6.3 Step time — derived, then gated

No 3.2B step has ever been measured in this project. Derivation:

```
step FLOPs = 8·N_student·tokens + 2·N_teacher·tokens      tokens = 2 × 8192
  depth-only  5.532e14        596M reference  2.099e14
L40S effective  50.6 TFLOPS   (from E6b's measured 4.15 s/step)
A100 assumed    80.9 TFLOPS   (1.6×, deliberately under the 1.72× peak ratio)
  raw 6.83 s/step  →  priced at 7.86 s/step with a 1.15× safety factor
  target-size cells: 2.98 s/step
```

**Blocking throughput + memory gate on the first arm**, before the remaining arms
are paid for: measure actual s/step and peak VRAM over the first ~20 steps and abort
if s/step exceeds the priced 7.86 or peak exceeds 78 GB. Two unmeasured quantities
become measurements at the cost of 20 steps.

### 6.4 Cost

**RETAINED-FP** (DP×2, DC×2, FC×2; FP reused):

| phase | min | $ |
| --- | ---: | ---: |
| setup | 45.0 | 1.19 |
| init NLL, DP + DC | 20.0 | 0.53 |
| init NLL, FP + FC | 12.0 | 0.32 |
| step-0 autonomous probe, DP + DC | 40.0 | 1.06 |
| pre-training gates | 6.0 | 0.16 |
| train depth-only, 4 arms × 1,761 steps | 922.8 | 24.45 |
| train target-size, 2 arms | 175.1 | 4.64 |
| evaluate depth-only, 4 arms | 120.0 | 3.18 |
| evaluate target-size, 2 arms | 16.5 | 0.44 |
| general-text diagnostics | 18.0 | 0.48 |
| artifact manifest + verify | 8.0 | 0.21 |
| artifact synchronization (28.3 GB) | 39.0 | 1.03 |
| **expected completion** | **1422.4** | **$37.69** |
| soft stop | 1564.6 | $41.46 |
| artifact-recovery reserve | 30.0 | $0.80 |
| **absolute termination** | **1594.6** | **$42.26** |

**FULL** (adds FP×2, hardware-matched): expected **$43.02**, soft stop $47.32,
**absolute termination $48.11**, 1,815.6 min ≈ 30.3 h.

### 6.5 Authorization

```
actual cumulative spend to date         $163.8833
  of which E8/E8a already incurred        $3.7253
E8's unspent backstop                     $9.5147   NOT carried over — E8b is a
                                                    new design and needs its own

E8b RETAINED-FP  expected $37.69  hard backstop $42.26
E8b FULL         expected $43.02  hard backstop $48.11

ADDITIONAL AUTHORIZATION REQUIRED   $42.26  (retained)  or  $48.11  (full)
PROPOSED NEW CUMULATIVE CAP        $206.14  (retained)  or  $212.00  (full)
GPU                                A100 SXM 80GB @ $1.59/h, single device
EXPECTED WALL TIME                 23.7 h  (retained)  or  30.3 h  (full)
```

Not reduced to one seed and no training semantics changed to fit the previous cap.
The behaviour-metric seed noise floor is 0.1290 and the interaction term compounds
four two-seed cells, so single-seed cells would make it unreadable.

**Session-length caveat, stated rather than discovered.** 24–30 h is 2.5–3× the
longest session this project has run (E7, 635 min). Across today's six draws the
session-level hazards were cold hosts, non-starting pods and one pathological host
shape; all are now fixed and every one is a redraw rather than a session loss. Even
so, a 30-hour single session is a new operational regime. Splitting DP/DC from
FC/FP into two pods would halve the exposure but would put the four step-0
measurements on two devices, which §4 forbids — so the recommendation is one
session with the throughput gate, per-arm budget gates, and continuous relay.

## 7. Comparisons and preregistered interpretation

Absolute DP/DC versus FP/FC is **not** a comparison — the depth-only models are
5.39× larger. The effects live inside each compression regime:

```
Δ_depth_only  = DC − DP
Δ_compressed  = FC − FP
interaction   = (FC − FP) − (DC − DP)
```

Computed per seed and pooled, for `usable_rollout_rate`, `correct_overall`,
`correct_given_usable`, `natural_termination_rate`, `context_limit_rate`,
`severe_repetition_rate`, `empty_output_rate`,
`answer_parse_failure_rate_numeric`, GSM8K and every frozen capability subset.
Floors reused unchanged: usable **0.0800**, correct **0.0600**, each requiring the
same sign on both seeds. The interaction compounds four two-seed cells, so direction
agreement across seeds carries more weight than the point estimate and a nonzero
value is not by itself evidence of interaction — the same caution E6b's interaction
table carries.

| case | reading |
| --- | --- |
| **DC > DP and FC < FP** | contribution-guided depth selection is itself useful but interacts negatively with the current width/FFN/attention compression. Direct motivation for a recursive/conditional initializer that remeasures each operator on the checkpoint the previous operator produced |
| **DC < DP and FC < FP** | the full-width teacher-ablation KL objective is not a good selector for a recoverable depth initialization |
| **DC > DP and FC > FP** | contribution-guided depth is beneficial despite its worse fully-compressed step-0 NLL — further weakening NLL as an initializer-ranking proxy |
| **DC ≈ DP but FC differs** | the interaction with downstream structural compression dominates the standalone depth-map effect |

**The depth selector will not be redesigned inside E8b after seeing which case
occurs.** Any adjacency penalty, compression-aware objective or operator-order
search is a separate, separately-registered experiment.

## 8. Comparability checks, mechanical

Before training:

1. every arm's config differs from its cell's control in only `{student_path,
   run_name, out_dir, _purpose}`;
2. DP and DC configs are byte-identical except `student_path` — same for FP and FC;
3. DP and DC share one config hash `4e5b7104…`; FP and FC share the control's
   `15e63575…`;
4. the token budget is re-derived from the pack: 1,174 blocks, 1,600,353 unique CE
   targets, 3.0 exposures, 4,801,059 cumulative;
5. all four initializations pass `require_init_nll` against their own recomputed
   hashes;
6. the frozen depth map's hash matches E8a's search report, and the calibration
   leakage report is clean;
7. RoPE resolves to 5,000,000 for all four checkpoints in both venvs;
8. DP and DC re-verify bitwise against `bypassed_blocks(teacher, removed)` on the
   pod, so a corrupted transfer cannot pass;
9. the inclusion mask rebuilds to `d6e24e0b09da1bcc…`;
10. the attention implementation is SDPA, asserted, because the memory sizing
    depends on it;
11. one distributed configuration for DP and DC: single device, recorded.

## 9. What E8b preserves for a possible later AutoInitializer

Not built, not started, and out of scope here. E8b's artifacts are kept in a form
that can test whether

```
D(W/F/A(T))  ≠  W/F/A(D(T))
```

— i.e. whether compression operators commute. If DC beats DP while FC loses to FP,
that is direct empirical motivation for sequential conditional compression, and the
four cells plus their step-0 tables are the evidence. **No E9, no operator-order
search, no beam search, no 0.86M candidate probing under this instruction.**

## 10. Status

* **tests: 1,221 pass, 6 skipped** locally; 1,192 pass, 22 skipped under a pod-env
  simulation of E8a's search-only staged set.
* **repository: clean and pushed**, `15cf9dc` plus this preflight.
* **no pods running, nothing billing**, verified by `runpodctl pod list` → `[]`.
* DP and DC checkpoints are built; DP's weights were deleted locally after hashing
  because the dev box has 12 GB free and each is 6.43 GB. Both are reconstructible
  deterministically from the teacher, and the pod builds them itself and asserts
  the hashes above — no multi-GB upload crosses the dev-box uplink.

## 11. Reproduction of everything above, all $0

```bash
PYTHONPATH=src python scripts/training/build_depth_only_init.py \
    --map positional   --out artifacts/stage1/e8b_dp_init
PYTHONPATH=src python scripts/training/build_depth_only_init.py \
    --map contribution --out artifacts/stage1/e8b_dc_init
PYTHONPATH=src python scripts/training/size_e8b_memory.py
PYTHONPATH=src python scripts/training/plan_e8b_budget.py
PYTHONPATH=src pytest tests/ -q
```
