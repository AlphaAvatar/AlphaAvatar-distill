# E8b — depth-map × compression interaction: pair-matched preflight and proposal

**Status: PREFLIGHT COMPLETE, NOT AUTHORIZED. No GPU has been used for E8b.** Both
depth-only initializations are built and verified, all six arm configs are generated
and identity-checked, hardware is selected on cost per completed step, and the
sessions are priced. Nothing paid launches without separate authorization.

**E8a is closed and not reopened** — its frozen map, its 3.11× result and its step-0
dissociation stand as recorded in [`e8_step0_report.md`](e8_step0_report.md) and
[`EXPERIMENTS.md`](EXPERIMENTS.md) §36. **The old E8 2.96M recovery is cancelled and
must not be launched.**

**Question.** Is the contribution-guided depth map better when depth is the only
compression, and does it become worse only when composed with the existing
width/FFN/attention compression?

---

## 1. The factorial, and the hardware it is measured on

```
                            DEPTH MAP
                    positional        contribution        hardware
                  ┌───────────────┬───────────────┐
depth-only        │  DP  2 seeds  │  DC  2 seeds  │   A100 SXM 80 GB
3,215,021,568     │               │               │   (needs 72.9 GB)
                  ├───────────────┼───────────────┤
fully compressed  │  FP retained  │  FC  2 seeds  │   L40S 48 GB
596,049,920       │               │               │   (FP's own hardware)
                  └───────────────┴───────────────┘
```

Hardware is **pair-matched inside each causal comparison**, not uniform across the
matrix, because the two primary effects are `DC − DP` and `FC − FP` and each is
measured within one class. FP is *not* retrained on the A100 to homogenise the
matrix at this stage.

**Hardware class is therefore nested with compression regime, and the interaction
inherits that nesting.** `(FC − FP) − (DC − DP)` is sufficient for the first
mechanism test but **cannot by itself exclude a hardware × depth-map interaction**.
§9's conditional bridge resolves that only if a material reversal actually occurs.

Every cell trains the canonical **E1/P1 KD-heavy 1.60M** recipe: 1,600,353 unique
supervised CE tokens, three exposures, 1,761 steps, CE 0.25 / KD 1.0, τ 1.0, scope
all, lr 5e-5, warmup 88, 2 blocks/step, `block_len` 8192, `save_every` 880, seeds
`sa` 20260726 / `sb` 20260801, identical data, block-order derivation, optimizer,
scheduler, trainable-pattern set and frozen evaluation protocol. **Not 2.96M, and not
reduced to 0.86M** — 0.86M may later serve an AutoInitializer candidate search, but
E8b is a mechanism experiment and keeps the stronger 1.60M behavioural signal.

## 2. DP and DC — built, verified, hashed, at $0

`scripts/training/build_depth_only_init.py`. Depth is the only compression: teacher
hidden 2560, FFN 9728, 32 Q / 8 KV heads, head_dim 128, embeddings, tied lm head,
norms, vocabulary and tokenizer carried over untouched. 36 → 28 layers by verbatim
`state_dict` copy — no projection, no norm folding, no head or neuron selection.

| | DP | DC |
| --- | --- | --- |
| removed teacher layers | `[5,7,9,11,13,15,17,19]` | `[2,3,15,16,20,21,26,32]` |
| parameters | 3,215,021,568 | 3,215,021,568 |
| `model.safetensors` sha256 | `d4db65eb8f7ae6d8a847c2db9a9e5e307e449f50f3bd129e07a1b20f6ec5f3cd` | `eb9e95481988b296a77c30d7b4754069f1874330fca9ad198f4457029e11e182` |
| config sha256 | `4e5b71040b0badb8e9b3f1c58f99ef6d3e69723612ab69ee783a8ba56887ad82` | **identical** |
| resolved RoPE base | 5,000,000 | 5,000,000 |
| bf16 checkpoint | 6.43 GB | 6.43 GB |

**Both verified bitwise identical to `bypassed_blocks(teacher, removed)`** — max
absolute logit difference exactly `0.000e+00`, checked on the reloaded checkpoint.

Neither is uploaded: they are deterministic functions of the pinned teacher, so each
pod rebuilds them in ~6 minutes and asserts the hashes above. Nothing multi-GB
crosses the dev-box uplink, and a corrupted transfer cannot pass.

### 2.1 The identity sharpens the design

`bypassed_blocks` is the exact operation E8a's objective measured, so **DP and DC
*are* the ablated teachers E8a scored.** E8a's calibration KLs — 1.932531 for DP's
map, 0.620586 for DC's — are already step-0 statements about these two models, and
DC starts 3.11× closer to the teacher there. E8b extends that to the evaluation
streams and adds autonomous behaviour, which E8a never measured. So §8's depth-only
cases carry a strong prior that **DC beats DP at step 0**; the informative unknowns
are whether that survives recovery and whether the sign flips under compression.

### 2.2 A precision finding that constrains the step-0 protocol

`Qwen3ForCausalLM(cfg).to(bfloat16)` casts the rotary `inv_freq` **buffer** to bf16;
`from_pretrained` recomputes it in fp32. The in-memory model therefore ran a
lower-precision positional basis and differed from the ablated teacher by **0.78** in
logits, while the reloaded checkpoint is identical. `inv_freq` is non-persistent, so
no saved checkpoint is affected — including every Stage 1 artifact built through
`build_student`, whose blanket `.to(dtype)` does the same thing.

**Consequence, binding on §4:** no step-0 number may come from a directly-constructed
in-memory model. All four initializations are measured by loading the **saved
checkpoint** through one canonical `from_pretrained` path with the corrected
`assert_rope_from_config` check.

## 3. FP — retained control, and exactly what is proven

```
FP = E1/P1 KD-heavy 1.60M from the canonical positional PCA init
     init      artifacts/stage1/qwen3_0p6b_init_v0/checkpoint, 86fbba78e8a2a324…
     arms      e1_r1600k_{sa,sb}_pca @ step_001761
     relay     e1_scaling_20260801/e1_r1600k_{seed}_pca/step_001761
     weights   sa 6f77676ab8fde397…   sb e432d57e598d57e1…
     battery   retained from E6, inclusion mask d6e24e0b09da1bcc…
     behaviour usable 0.7300 · correct 0.1867 · correct|usable 0.2511
     hardware  L40S 48 GB @ $0.99/h
```

**Proven compatible:** identical rung, recipe, objective, optimizer, scheduler,
seeds, data, block order and trainable-parameter set — FC's config is FP's control
config with only `student_path` changed, asserted mechanically (§7). Same GPU class.
Frozen battery artifacts exist and re-score under the current scorer on the binding
mask.

**Not proven, and stated as a limitation:** FP's pod is not FC's pod, and FP's
initialization NLL has never been measured on the device that will measure FC's — so
it is **remeasured** in S1 alongside the other three. Its trained weights are what
they are; that is the nesting §1 records.

## 4. Step-0 measurement — one device, one loader, all four

Session **S1** measures **DP, DC, FP and FC** on one L40S, in one environment, each
through the same canonical saved-checkpoint reload path with the corrected RoPE
assertion. Inference needs only ~20.5 GB (6.43 GB student + 8.04 GB teacher + logits),
so the 3.2B checkpoints fit the cheap card comfortably — the "one evaluator"
requirement costs $2.51 rather than an 80 GB session.

Enforced by `nll_gate.require_init_nll`: an initialization is incomplete until its
own NLL artifact exists, bound to its recomputed hash, with every series present.
Per checkpoint: `holdout_v1` NLL · `fineweb_val_e7` NLL, teacher→student KL, top-1,
mean rank, entropy · `teacher_native_val` NLL, KL, top-1, mean rank · config hash ·
parameter count · resolved RoPE base · environment.

The E8a-session numbers for FP and FC (`fineweb_val_e7` 11.5749 and 14.3913, records
`a40feef0dd1535aa…` and `50863410fa170683…`) are retained for continuity and **not
substituted** into the E8b table.

### 4.1 Step-0 autonomous probe, on a separate battery

DP and DC are probed before training on **`data/eval_behavior_v0`** — 76 prompts, 7
groups, an existing frozen mechanically-scored diagnostic. Deliberately **not** the
150-prompt promotion battery, which is sampled from the 0.86M rung and is E8b's
endpoint; using it as a step-0 signal is the contamination to avoid. Generation
settings identical to the promotion protocol: greedy, system message mandatory,
thinking never suppressed, no artificial cap, complete raw generations saved.

Three levels of evidence, kept separate:

```
full-width teacher-ablation KL (E8a)   DP 1.932531   DC 0.620586
        ↓
depth-only checkpoint at step 0        S1, diagnostic
        ↓
depth-only after 1.60M recovery        S2/S3, the conclusion
```

Step-0 behaviour is diagnostic. Formal conclusions come only from the matched
recovered models on the frozen promotion battery.

## 5. Memory sizing — why the depth-only cells cannot use the L40S

`scripts/training/size_e8b_memory.py`, canonical semantics (float32 master weights,
bf16 autocast, gradient checkpointing on, embeddings and tied lm head frozen, one
microbatch of one 8,192-token block, SDPA attention):

| term | DP/DC (3.215B) | FP/FC (596M) |
| --- | ---: | ---: |
| params, float32 | 12.86 GB | 2.38 GB |
| gradients, float32, trainable only | 11.30 GB | 1.76 GB |
| Adam states, float32 ×2 | 22.61 GB | 3.52 GB |
| **student state** | **46.77 GB** | **7.67 GB** |
| teacher, bf16, resident | 8.04 GB | 8.04 GB |
| student + teacher logits | 4.98 GB | 4.98 GB |
| KD/CE float32 reduction | 0.93 GB | 0.93 GB |
| activations | 2.68 GB | 1.47 GB |
| **expected peak** | **63.41 GB** | **23.09 GB** |
| **+15% allocator/workspace margin** | **72.92 GB** | **26.56 GB** |

L40S 48 GB holds FP/FC and **not** DP/DC. Load-bearing assumption, asserted at
setup: SDPA/flash attention, so the O(L²) score matrix is never materialized — an
eager fallback would add ~4,295 GB.

## 6. Hardware chosen on cost per completed step

Live RunPod secure quotes, 2026-08-11, ≥70 GB classes. `s/step` is derived from
`8·N_student·tokens + 2·N_teacher·tokens` = 5.532e14 FLOPs against the 596M
reference's 2.099e14 at E6b's measured 4.15 s/step (50.6 effective L40S TFLOPS),
with an assumed relative bf16 training efficiency and a 1.15× safety factor:

| GPU | VRAM | $/h | stock | s/step | **$/step** | 4 arms |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| RTX PRO 6000 MaxQ | 96 | 0.50 | Low | 11.23 | **0.001560** | $10.99 |
| A100 PCIe | 80 | 1.39 | Low | 8.73 | 0.003372 | $23.75 |
| RTX PRO 6000 WK | 96 | 1.89 | Low | 6.55 | 0.003439 | $24.22 |
| **A100 SXM** | **80** | **1.59** | **High** | **7.86** | **0.003472** | **$24.45** |
| H100 SXM | 80 | 3.29 | High | 3.93 | 0.003592 | $25.30 |
| RTX PRO 6000 | 96 | 2.09 | High | 6.55 | 0.003803 | $26.79 |
| H100 PCIe | 80 | 2.89 | Low | 5.07 | 0.004071 | $28.68 |
| H200 SXM | 141 | 4.59 | High | 3.93 | 0.005011 | $35.30 |

**Selected: A100 SXM 80 GB @ $1.59/h** — cheapest per step among High-stock classes,
single device, no distributed configuration (72.9 GB fits one card, so DP and DC
share an identical setup by construction). No semantics changed to fit a card: no
LoRA, no extra freezing, no quantized optimizer states, no objective change.

**Two honest caveats about this table.** The relative-efficiency figures are
assumptions, not measurements, and the High-stock spread is only 3.5% — so if H100
SXM were ≥2.5× the A100 rather than the assumed 2.0×, it would become cheaper per
step. And RTX PRO 6000 MaxQ is 2.2× cheaper per step on paper but is Low stock,
power-limited and unproven here. **Neither is benchmarked prospectively.** The gate
below settles A100's real number; if A100 misses its registered assumption the run
stops and re-prices, with H100 SXM and MaxQ as the named alternatives.

### 6.1 Registered first-arm gate — 20 steps, three quantities

On the first depth-only arm, before any further arm is paid for, measure and record:

| quantity | registered assumption | action if violated |
| --- | --- | --- |
| wall-clock s/step | ≤ **7.86** | stop, re-price |
| peak VRAM | ≤ **78 GB** | stop, re-price or re-plan |
| effective $/step at the live pod quote | ≤ **0.003472** | stop, re-price |

Not a budget widener and not a hardware switch: on violation the session stops and
returns a revised proposal.

## 7. Session layout — four restartable sessions, none over 10.3 h

| | GPU | contents | expected | hard | wall |
| --- | --- | --- | ---: | ---: | ---: |
| **S1** | L40S $0.99 | build DP/DC + assert hashes; step-0 NLL for **all four**; step-0 probes DP/DC | $2.51 | $3.25 | 2.5 h |
| **S2** | A100 $1.59 | DP-`sa` + DC-`sa`, gate, evaluate | $16.33 | $18.76 | 10.3 h |
| **S3** | A100 $1.59 | DP-`sb` + DC-`sb`, evaluate | $16.33 | $18.76 | 10.3 h |
| **S4** | L40S $0.99 | FC-`sa` + FC-`sb`, evaluate | $5.37 | $6.40 | 5.4 h |
| **total** | | | **$40.54** | **$47.18** | **28.5 h** |

**Seed-paired, not cell-paired**: each A100 session holds one DP and one DC arm, so a
lost session costs a seed rather than a cell and the surviving session still carries a
complete matched pair. The longest session is 10.3 h against E7's proven 10.6 h.

Every session runs the post-E6b contract: detached start, provider-only watchdog from
pod creation, continuous `LogRelay` of structured event streams, `final_required`
artifact gate before teardown, hash verification, per-arm budget gates, and exact
mid-arm resume via `save_every` 880. **A split cannot alter science**: all six arms
read generated configs whose diff against their controls is exactly
`{student_path, run_name, out_dir, _purpose}`, so token exposure, schedule, seeds and
evaluator semantics are identical however the arms are distributed.

The hard total exceeds the expected by $6.64 mostly because four sessions carry four
30-minute recovery reserves rather than one; the expected-cost penalty for splitting
is small and the exposure reduction is large.

### 7.1 The six arms, generated and identity-checked

`scripts/training/build_e8b_configs.py`:

| arm | cell | seed | GPU | config sha256 |
| --- | --- | --- | --- | --- |
| `e8b_dp_r1600k_sa` | DP | 20260726 | A100 | `7c682389a204acdb…` |
| `e8b_dp_r1600k_sb` | DP | 20260801 | A100 | `5a44ccc135321ba9…` |
| `e8b_dc_r1600k_sa` | DC | 20260726 | A100 | `ba91edaaa17b1956…` |
| `e8b_dc_r1600k_sb` | DC | 20260801 | A100 | `5a617bccefb3f606…` |
| `e8b_fc_r1600k_sa` | FC | 20260726 | L40S | `77b3da5f56fc4f6d…` |
| `e8b_fc_r1600k_sb` | FC | 20260801 | L40S | `b220726c1fb94cc9…` |

Verified: `DP-vs-DC-sa`, `DP-vs-DC-sb`, `FP-vs-FC-sa`, `FP-vs-FC-sb` each differ in
exactly `{student_path, run_name, out_dir, _purpose}`. Each depth-map effect is a
single-variable comparison inside one hardware class.

## 8. Comparisons and preregistered interpretation

Absolute DP/DC versus FP/FC is **not** a comparison — 5.39× the parameters. The
effects live inside each regime:

```
Δ_depth_only  = DC − DP        both on A100
Δ_compressed  = FC − FP        both on L40S
interaction   = (FC − FP) − (DC − DP)      hardware nested with regime
```

Per seed and pooled, for `usable_rollout_rate`, `correct_overall`,
`correct_given_usable`, `natural_termination_rate`, `context_limit_rate`,
`severe_repetition_rate`, `empty_output_rate`, `answer_parse_failure_rate_numeric`,
GSM8K and every frozen capability subset. Floors unchanged: usable **0.0800**,
correct **0.0600**, each requiring the same sign on both seeds. The interaction
compounds four two-seed cells, so direction agreement across seeds carries more
weight than the point estimate and a nonzero value is not by itself evidence of
interaction — the caution E6b's interaction table already carries.

| case | reading |
| --- | --- |
| **DC > DP and FC < FP** | contribution-guided depth selection is itself useful but interacts negatively with the current width/FFN/attention compression. **Triggers the §9 bridge before any attribution is made.** Direct motivation for a recursive/conditional initializer |
| **DC < DP and FC < FP** | the full-width teacher-ablation KL objective is not a good selector for a recoverable depth initialization |
| **DC > DP and FC > FP** | contribution-guided depth is beneficial despite its worse fully-compressed step-0 NLL — further weakening NLL as an initializer-ranking proxy |
| **DC ≈ DP but FC differs** | interaction with downstream structural compression dominates the standalone depth-map effect |

**The depth selector will not be redesigned inside E8b after seeing which case
occurs.** Adjacency penalties, compression-aware objectives and operator-order search
are separate, separately-registered experiments.

## 8.1 The seven questions the final report must answer

Registered verbatim by the maintainer in the 2026-08-11 authorization, recorded here
because chat history is not project memory (§3.1). The report answers these in order
and does not substitute a different framing:

1. Does DC beat DP after matched recovery?
2. Does FC beat FP after matched recovery?
3. Are the two depth-map effects in the same direction?
4. Is there evidence of a depth-map × downstream-compression interaction?
5. Does step-0 NLL predict either recovered comparison?
6. Does the E8a full-width causal KL predict either recovered comparison?
7. Is the hardware bridge trigger satisfied?

Questions 5 and 6 are the ones the experiment is most likely to answer cleanly and
are the reason the three evidence levels are kept separate rather than pooled. Both
are answerable in the negative by a single regime — a proxy that gets the *sign*
wrong in either regime has failed as a predictor there, whatever its magnitude.

The interaction is reported as `(FC − FP) − (DC − DP)`, the maintainer's stated
direction: negative means the depth map does worse once compression is applied than
it does at full width.

Absolute DP/DC performance is not compared with FP/FC as though model size were
controlled (5.39× the parameters).


## 9. Conditional hardware-bridge rule — registered, not run

**Trigger:** E8b produces a scientifically material sign reversal or interaction —
specifically `DC > DP` while `FC < FP`, with each effect above its registered floor
and seed-consistent.

**On trigger: do not attribute the reversal to compression interaction.** Stop and
propose a bridge that reruns the compressed pair, **FP and FC, both seeds**, on the
same ≥80 GB hardware used for DP/DC, so the compressed effect is measured free of the
hardware nesting.

**Bridge cost, priced now so the decision is informed, and not spent now:**
$12.05 expected, **$14.05 hard**, 7.6 h on A100 SXM, 4 arms × 1,761 steps at a
derived 2.98 s/step with its own first-arm gate. If it ran, the cumulative cap would
become **$225.11**.

**Do not run the bridge prospectively.** This avoids paying now to remove a confound
that may never matter.

## 10. Authorization

```
actual cumulative spend to date        $163.8833
  of which E8/E8a already incurred       $3.7253
E8's unspent backstop                    $9.5147   NOT carried over — E8b is a new
                                                    design and carries its own

E8b expected completion                 $40.54
E8b hard backstop (sum of four sessions) $47.18
expected wall time                        28.5 h across 4 sessions, longest 10.3 h

ADDITIONAL AUTHORIZATION REQUIRED        $47.18
PROPOSED NEW CUMULATIVE CAP             $211.06

CONDITIONAL bridge, only on trigger      $14.05  ->  cap would become $225.11
```

Not reduced to one seed; no training semantics changed to fit a previous cap. The
behaviour seed-noise floor is 0.1290 and the interaction compounds four two-seed
cells, so single-seed cells would be unreadable.

## 11. Comparability checks, mechanical, before any training

1. every arm's config diff against its cell's control is exactly
   `{student_path, run_name, out_dir, _purpose}`;
2. DP/DC configs are byte-identical but for `student_path`; likewise FP/FC;
3. DP and DC share config hash `4e5b7104…`; FP and FC share `15e63575…`;
4. token budget re-derived from the pack: 1,174 blocks, 1,600,353 unique CE targets,
   3.0 exposures, 4,801,059 cumulative;
5. all four initializations pass `require_init_nll` against their own recomputed
   hashes, measured in S1 on one device through one canonical reload path;
6. the frozen depth map's hash matches E8a's search report; calibration leakage clean;
7. RoPE resolves to 5,000,000 for all four checkpoints in every venv used;
8. DP and DC re-verify bitwise against `bypassed_blocks(teacher, removed)` on each
   pod that builds them;
9. the inclusion mask rebuilds to `d6e24e0b09da1bcc…` in every session that evaluates;
10. SDPA attention asserted, because the memory sizing depends on it;
11. single-device configuration for DP and DC, recorded;
12. **hardware nesting recorded in the arms manifest**, so no later reader can take
    the interaction as hardware-controlled.

## 12. What E8b preserves for a possible later AutoInitializer

Not built, not started, out of scope. The artifacts are kept in a form that can test
whether compression operators commute:

```
D(W/F/A(T))  ≠  W/F/A(D(T))
```

If DC beats DP while FC loses to FP, the four cells plus their step-0 tables are the
empirical motivation for sequential conditional compression. **No E9, no
operator-order search, no beam search, no 0.86M candidate probing under this
instruction.**

## 13. Status

* **tests: 1,221 pass, 6 skipped**; 1,192 pass / 22 skipped under a pod-env
  simulation of a search-only staged set.
* **repository clean and pushed.**
* **no pods running, nothing billing** — `runpodctl pod list` → `[]`.
* DP/DC built and hashed; DP's weights deleted locally after hashing (12 GB free, 6.43
  GB each). Both rebuild deterministically on any pod and assert their hashes.

## 14. Reproduction, all $0

```bash
PYTHONPATH=src python scripts/training/build_depth_only_init.py \
    --map positional   --out artifacts/stage1/e8b_dp_init
PYTHONPATH=src python scripts/training/build_depth_only_init.py \
    --map contribution --out artifacts/stage1/e8b_dc_init
PYTHONPATH=src python scripts/training/build_e8b_configs.py
PYTHONPATH=src python scripts/training/size_e8b_memory.py
PYTHONPATH=src python scripts/training/plan_e8b_budget.py
PYTHONPATH=src pytest tests/ -q
```
