# E8 step 0 — the contribution-guided depth map, and what it did to the initialization

**Status: the search half of E8 is COMPLETE. The training half has not run.** The
map is frozen, both initializations are built and measured, and the formal
two-seed comparison is blocked on $0.20 of authorization (§5).

Preregistration: [`e8_preregistration.md`](archive/e8_preregistration.md). Every threshold,
the selector, the calibration set and the four outcome readings were fixed before
any GPU was used.

---

## 1. The map

The search ran the preregistered objective — forward
`KL(teacher ‖ teacher-with-S-bypassed)` over all prediction positions, aggregated
as an unweighted mean over 5 domains of the unweighted mean over their sub-types —
as an iterative greedy removal, 8 rounds, **260 subset evaluations**, 17,688
forward passes, 1,300 s on one L40S.

```
contribution-guided  keeps   [0,1,4,5,6,7,8,9,10,11,12,13,14,17,18,19,
                              22,23,24,25,27,28,29,30,31,33,34,35]
                     removes [2, 3, 15, 16, 20, 21, 26, 32]
positional           removes [5, 7, 9, 11, 13, 15, 17, 19]
```

They share exactly one layer. The removal order was
`[2, 16, 3, 32, 20, 26, 15, 21]`: the two blocks the causal measure gives up first
are **layers 2 and 3**, which the positional rule explicitly protects.

| | contribution | positional | ratio |
| --- | ---: | ---: | ---: |
| **primary objective** | **0.620586** | **1.932531** | **3.11×** |
| general | 0.598103 | 1.587545 | 2.65× |
| math | 0.563415 | 1.878962 | 3.33× |
| rag_multihop | 0.633695 | 1.857232 | 2.93× |
| code | 0.615500 | 2.050719 | 3.33× |
| tool | 0.692215 | 2.288195 | 3.31× |

Diagnostics — recorded per candidate per round and preregistered as **unable to
select**. Every one of them agrees with the primary, so no tie-break was needed
and none was used:

| diagnostic | contribution | positional | ratio |
| --- | ---: | ---: | ---: |
| assistant positions | 0.568161 | 1.966484 | 3.46× |
| reasoning (inside `<think>`) | 0.599376 | 2.002778 | 3.34× |
| final answer (after `</think>`) | 0.460475 | 1.814546 | 3.94× |
| `</think>` itself | 0.594824 | 2.183240 | 3.67× |
| assistant `<|im_end|>` | 2.829312 | 7.045699 | 2.49× |
| `</tool_call>` | **0.024094** | **10.321481** | **428×** |

**Instrument validity.** `self_consistency` — the intact reference against a fresh
pass of the same model — was **exactly 0.0** against a 1e-6 tolerance, so the
candidate ranking is not measuring kernel noise. The reference logit cache was
kept (18.16 GB estimate against 39.0 GB free).

## 2. The treatment initialization is a single-variable change, measured

| quantity | control | treatment |
| --- | --- | --- |
| `model.safetensors` sha256 | `86fbba78e8a2a324…` | `7a0694a5d5c59f8e…` |
| student config sha256 | `15e63575bd90e5f4…` | `15e63575bd90e5f4…` **identical** |
| parameters | 596,049,920 | 596,049,920 **identical** |
| projection energy captured | 0.9323228843289764 | 0.9323228843289764 **identical** |
| final-norm weight range | [-0.03870667333325841, 7.125069193436976] | **identical** |
| resolved RoPE base | 5,000,000 | 5,000,000 |
| depth map source | `derived_from_recorded_spans` | `explicit_kept_layers` |

Only the depth map differs. This is measured, not asserted: the projection is a
function of the Stage 0 statistics alone, and it reproduces to the last digit.

## 3. Step-0 diagnostics — and they go the other way

Both initializations measured **on one device, by one evaluator, in one
environment** (L40S, bf16, transformers 5.13.1, torch 2.11.0+cu128), each record
hash-bound to its own checkpoint. The control was **remeasured**, not inherited —
its historical 11.7482 came from a different reader path and is not comparable
(see [`decisions.md`](decisions.md), 2026-08-10).

| series · metric | contribution | positional | Δ |
| --- | ---: | ---: | ---: |
| `holdout_v1` NLL | 13.2624 | 11.7565 | **+1.5059** |
| `fineweb_val_e7` NLL | 14.3913 | 11.5749 | **+2.8164** |
| `fineweb_val_e7` teacher KL | 12.5598 | 9.4187 | +3.1411 |
| `fineweb_val_e7` top-1 | 0.0075 | 0.0227 | −0.0151 |
| `fineweb_val_e7` mean rank | 51,875.3 | 18,144.3 | +33,731.0 |
| `teacher_native_val` NLL | 11.8027 | 10.9053 | **+0.8974** |
| `teacher_native_val` teacher KL | 11.5075 | 10.6332 | +0.8743 |
| `teacher_native_val` top-1 | 0.0230 | 0.0408 | −0.0177 |
| `teacher_native_val` mean rank | 19,865.6 | 6,739.9 | +13,125.8 |

Positions: 21,080 · 523,776 · 81,195. Records `50863410fa170683…` (treatment) and
`a40feef0dd1535aa…` (control), both `complete: true`.

**The contribution-guided initialization is worse on every diagnostic, on all
three series, on NLL and teacher KL and top-1 and rank alike.**

### 3.1 This is a dissociation, and it is the finding

The same map is **3.11× better** at preserving the teacher's output distribution
when blocks are bypassed *in the teacher*, and **substantially worse** once those
kept layers are projected into the compressed student. Teacher-ablation KL did not
predict initialization quality after compression.

The two operations are not the same operation, which is the likely explanation and
is stated as a hypothesis, not a result:

* the calibration measured the teacher at **full width** — 2560 hidden, 9728 FFN,
  32 Q heads — with blocks bypassed;
* the initialization additionally compresses hidden width 2560 → 1024 by activation
  PCA, FFN 9728 → 3072 by top-k neuron selection, and 32 → 16 Q heads. Depth choice
  interacts with width and FFN compression, and the objective never measured that
  interaction.

A specific mechanism worth testing later: the contribution map removes three
**adjacent** pairs (2–3, 15–16, 20–21), so a surviving layer's input can be two
blocks' worth of transformation away from what it saw in the teacher. The
positional map removes every *other* layer in its band, so each survivor's input
is off by at most one block. Sandwich initialization applies each representative's
own weights to the incoming stream state, so a one-block gap preserves that state
better. The teacher, at full width with 34 blocks remaining, absorbs a two-block
gap; a 0.6B student that also lost 60% of its width may not.

### 3.2 What this does **not** decide

Per the preregistration (§7.1, §8), a worse initialization NLL may not cancel E8.
It is diagnostic. The registered catastrophic-abort conditions were all checked and
**none fired**: logits finite, save/reload round-trip clean, RoPE base 5,000,000,
parameter count and config hash identical to the control's, 28 strictly increasing
teacher layers, `self_consistency` 0.0, calibration leakage clean.

Two preregistered outcomes remain live and only the training separates them:

* **outcome 3** — initialization NLL worse, autonomous behaviour *improves*:
  contribution-aware structure preserves reasoning-relevant computation that
  general-LM NLL does not see, and initialization NLL is again not a sufficient
  proxy. This is the result that would matter most.
* **outcome 4** — both regress: reject the contribution-guided map.

Given E7 already showed a −5.22-nat NLL swing moving autonomous behaviour by
exactly +0.0000, a +2.82-nat swing in the other direction is **not** evidence for
outcome 4. The project's own record says this diagnostic does not predict the
endpoint. That is precisely why the preregistration forbade cancelling on it.

## 4. What the two paid sessions cost

| session | outcome | cost |
| --- | --- | ---: |
| pod A, attempts 1–4 | five infrastructure defects, each self-terminating | $0.9583 |
| **pod A, attempt 5** | **search COMPLETE**, gate 8/8, teardown clean | **$0.53** |
| pod B, attempt 1 | 128-vCPU host: torch oversubscription made the test gate consume the session's second-arm budget; stopped rather than deliver one seed | $1.9520 |
| pod B, attempt 2 | both initializations measured; gate failed on a calibration artifact pod B does not stage | $0.2850 |
| **total** | | **$3.7253** |

Six defects were found and fixed. One was mine (`meta`-device RoPE check); three
were latent in the launcher E8 was derived from and had never fired (cold-host
misparse, no-endpoint abort, uv progress measured in the wrong tree); one was a
host-shape hazard (thread oversubscription on a wide host); one was a check
requiring an artifact its session does not stage. None touched the selector, the
calibration set, the depth map, the recipe or the evaluation protocol.

## 5. What completing E8 needs

```
E8 authorized backstop   $13.2400
spent                     $3.7253
available                 $9.5147
pod B hard threshold      $9.7130
SHORTFALL                 $0.1983
```

The training half is one pod: measure both initializations (already done once, ~3
min), pass the now-complete 18-check gate, train `sa` and `sb` at 2.96M
(403 min), run the frozen battery, sync. Everything it needs is built, staged and
verified; `validate_e8_arms.py --require-init` passes all 18 checks locally.

**Proposed:** an increment of **$0.20** — backstop $13.44, cumulative cap
$173.60. A larger increment would buy margin against a further bad draw; at
$9.7130 exactly there is none.

## 6. Reproduction

```bash
# search (pod A), complete
scripts/pod/e8a_launch.py --scr <dir> --session-commit 6539d6b8… \
    --bundle <bundle> --authorized-usd 2.54 --host-draws 5

# treatment init + staging (dev box, $0), complete
PYTHONPATH=src python scripts/training/build_and_stage_e8_init.py \
    --frozen-map <fetched e8_frozen_depth_map.json>

# training (pod B), NOT RUN
scripts/pod/e8b_launch.py --scr <dir> --session-commit <HEAD> --bundle <bundle> \
    --treatment-init-sha256 7a0694a5d5c59f8e0b0ebc9ac8648b1ec026bf93cab026d33c61ca8fc85d1edb \
    --authorized-usd <available> --host-draws 5

# analysis, once the treatment arms exist
PYTHONPATH=src python scripts/evaluation/analyze_e8.py --bootstrap 10000
```
