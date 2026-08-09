# Supported / in-progress models

Status values follow AGENTS.md 3.4. A model is listed here only once real work
exists for it; nothing below is a released or validated result.

Only the dense text baseline below has been attempted. The methods are *intended*
to generalize to other families (MoE, VLM, Omni-models) — that is intent, not
support, and no such model gets a row here until real work exists for it
([decision 2026-07-28](decisions.md)). No entry becomes a README Optim record
during baseline construction (same date).

| Model | Teacher | Student target | Status | Stages passed | Best checkpoint | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-4b-thinking-distill (working name) | `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d` | 0.6B-class: hidden 1024, 28L, FFN 3072, 16Q/8KV, tied emb (user-chosen 2026-07-13) | **stage3-running** | Stage 0, Stage 1, Stage 2 (v0+v1); Stage 3 sub-stages 1–2; Experiments 1–6b complete | **`e1_r2960k_{sa,sb}_pca`** = **E1/P1 KD-heavy 2.96M**, the default behavioural anchor — usable rollout **0.8400**, correct **0.2067**. Held against P2-2.96M (E6b): KD-heavy wins the primary axis by **0.0800 on both seeds**, both paired CIs excluding zero | E6b settles that the post-2.96M plateau is the **E1 objective's**, not the corpus's: P2 CE-heavy gains only +0.0267 from the same rung (a tie). Evidence of **objective-dependent scaling** — pooled interaction −0.0833, but per-seed −0.0133 / −0.1533, so the magnitude is **seed-sensitive and not an effect-size estimate**. The advantage is **stability, not reasoning**: `correct_given_usable` is essentially identical (0.2460 vs ≈0.2460) and **GSM8K correctness is 0.00–0.05 in every cell** while GSM8K usable rollout climbs for both. **P2-5.50M is not justified and must not be launched**; the P2 lineage is no longer the preferred basis for scaling. Stage 3 still not exited; no prospectively registered gate exists. |

---

## qwen3-4b-thinking-distill — current position

Everything below is recorded in the consolidated
[experiment record](EXPERIMENTS.md); per-run logs were merged into it on
2026-07-31 and remain in git history at commit `866dac2`.

**What passed.** Stage 0 collected 949,859 tokens of teacher sufficient
statistics. Stage 1's PCA/sandwich init reached holdout NLL 11.748 against a
random-init 12.129 and a teacher 2.6264, and its checkpoint
(`model.safetensors` sha256 `86fbba78…`) is the pinned fork point for all
recovery work. Stage 2 built offline mixtures v0 (5.39M train tokens) and v1
(22.13M). Stage 3 sub-stage 1 (FFN+norm, 660 steps) passed its gate at holdout
4.21; sub-stage 2 on mixture v1 reached **3.8003**; the start-point ablation
then retired the warm-up ladder — single-stage from the Stage 1 init reaches
3.8285 (+0.74%, inside the pre-registered 1% band) with 33% fewer steps, and is
best on every behaviour axis.

**Why the recommended checkpoint is not the best-NLL one.**
`s2v1_from_init@2700` (3.8285) is recommended over `s2_blocks_v1@2700` (3.8003)
because it is cheaper to produce and scored better on `eval_behavior_v0`.
**Carry the error bar:** the seed-only noise floor on that metric is **0.1290**,
so behaviour *orderings* between this project's checkpoints are not supported at
one run per arm. The scores are measurements, not a ranking. Behaviour
comparisons now require ≥2 seeds; cold-start holdout-NLL comparisons need ≥4
(two seeds of one config differed by 2.21 nats).

**Where Stage 3 stands after Experiment 1.** The degeneration blocker is
substantially resolved: on teacher-native recovery data the best arm terminates
naturally on **93.4%** of held-out prompts (`e1_r2960k_sa_pca`), against 0/8 for
every checkpoint measured before. Teacher-native held-out CE scales cleanly with
data — 74× the between-seed noise — and reaches **1.0032** at 5.50M supervised
tokens.

**Why it still has not exited.** Two things block it. First, **no reasoning**:
GSM8K exact match is ≤0.05 across all 25 checkpoints (mean 0.006) and exactly
0.000 at every rung and seed on the random init, so the student can hold the
protocol without solving anything. Second, the exit gate is format competence
measured at ≥2 seeds, and the behaviour composite resolves at only **3.3×** the
seed spread — it cannot rank the rungs, only separate the initializations.

**Largest measured capability gaps** against the teacher's `behavior_score_v0`
ceiling of 0.7443: math EM +0.714, tool_call +0.667, format_ok +0.618. These
gaps set how many prompts of each type the recovery corpus was generated from.

**What Experiment 1 also established, and what it cost.** Data quantity — not
optimizer updates — drives the CE gain (a step-matched control at 4,412 steps on
the smallest rung came out *worse* than the same rung at 324 steps), and
initialization dominates data over the whole measured range. $61.5 across 25
checkpoints.

**What Experiment 2 phase 1 then established (2026-08-04).** Median-length
corpus cleaning was tested at the 0.86M rung, two seeds, matched tokens, both
arms forked from the pinned Stage 1 init. **The cleaned corpus is not adopted.**
Its held-out-NLL gate passed arithmetically (mean +0.9618 > the 0.489 floor) but
99.2% of the margin came from one seed, and cleaning *raised* the between-seed
spread from 0.489 to 2.381 nats on identical data and init. The only
seed-consistent capability reading went the other way: aggregate protocol
validity fell 0.090 and 0.073 at the matched endpoint.

The phase's durable output is a **measurement finding, not a data decision**:
`best_holdout_nll` is retired as a checkpoint-selection identity. The checkpoint
holding the best held-out NLL of its trajectory (`sb`@127) produces **zero
protocol-valid generations across all 726 battery prompts** — general-text
perplexity peaks *before* the student specialises onto the teacher protocol, so
the metric and the objective diverge by construction
([decision](decisions.md), [record](EXPERIMENTS.md) §12.15).

**What exists for the next step.** Corpus v2 (11,174 accepted sessions, 66.08M
generated tokens, gate-passed) cut into the six-rung nested token ladder, plus a
frozen 846-prompt capability battery (`capability-v2`) with deterministic
scorers and no LLM judge, validated at 112 evaluator tests. Experiment 2 phases
2 (loss) and 3 (learning rate) are **specified but unauthorized**, and phase 3
should not run as written — it was built around the metric phase 1 retired
([`PROPOSAL.md`](PROPOSAL.md) §12).

**Deployment target:** INT8. Every recovery gate already re-evaluates under INT8
weight fake-quantization at two scopes.

## Stage 3 recovery arms — 2026-08-05, under the clarified Stage 2/3 objective

**No arm has completed the Stage 2/3 objective.** Primary metric is
`usable_rollout`; correctness is secondary; teacher-forced top-1 is a diagnostic
only ([`decisions.md`](decisions.md) 2026-08-05, [`EXPERIMENTS.md`](EXPERIMENTS.md) §19).

150 fixed examples, mask `d6e24e0b…`, unrestricted generation.

| alias | run | seed | loss | `kd_scope` | **usable rollout** | correct | correct \| usable | *fwd top-1* | status |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| **P1-sa** | `e1_r0860k_sa_pca` | 20260726 | ce .25 / kd 1.0 | all | 0.5133 | 0.1533 | 0.2727 | *0.5695* | **working baseline** |
| **P1-sb** | `e1_r0860k_sb_pca` | 20260801 | ce .25 / kd 1.0 | all | 0.5933 | 0.2133 | 0.2921 | *0.5720* | **working baseline** |
| P0-assistant-sa | `p0_assistant_sa` | 20260726 | ce .25 / kd 1.0 | assistant | **0.6067** | 0.1867 | 0.2747 | *0.5222* | not adopted; weights gone |
| P0-assistant-sb | `p0_assistant_sb` | 20260801 | ce .25 / kd 1.0 | assistant | 0.5667 | 0.1067 | 0.1765 | *0.5222* | not adopted; weights gone |
| P2-ceheavy-sa | `p2_ceheavy_sa` | 20260726 | **ce 1.0 / kd .25** | all | 0.5200 | 0.2000 | **0.3590** | *0.5511* | not adopted; weights retained |
| P2-ceheavy-sb | `p2_ceheavy_sb` | 20260801 | **ce 1.0 / kd .25** | all | 0.5467 | 0.1800 | 0.2927 | *0.5623* | not adopted; weights retained |

Family means — usable rollout: P0-assistant **0.5867**, P1 0.5533, P2 0.5333.
**Every gap is smaller than P1's own 0.0800 seed spread**, and paired at the
prompt level both interventions gain on `sa` and lose on `sb`.

* **P1 is the incumbent reference checkpoint**, retained for continuity of
  comparison — **not the best checkpoint**, and not confirmed by behaviour.
* **P0-assistant holds the highest observed mean usable-rollout rate (0.5867)** —
  not seed-consistent, and **its weights no longer exist**.
* **P2-ceheavy holds the best correctness conditional on a usable rollout**
  (0.3590 / 0.2927). Its overall correctness (0.2000 / 0.1800, mean 0.1900) is a
  separate and weaker statement. Not promoted on either.

### Beyond the 0.86M rung — Experiments 3, 4 and 5

Same 150 examples, same mask `d6e24e0b…`, unrestricted generation, so every row
below is directly comparable to the table above.

| alias | rung / continuation | seeds | **usable rollout** | correct | correct \| usable | status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| E3-A1 | 0.86M, attention frozen | sa/sb | 0.4467 | — | — | **rejected** — −0.087 vs P2 |
| E3-A2 | 0.86M, LoRA r32 α64 on q/k/v/o | sa/sb | 0.4400 | — | — | **rejected** — −0.093 vs P2 |
| P1-1.60M | 1.60M rung | sa/sb | 0.7300 | 0.1867 | 0.2511 | re-evaluated reference |
| **P2-1.60M** | 1.60M rung | sa/sb | **0.7333** | **0.2000** | 0.2682 | **best behaviour+correctness so far**; weights retained |
| E5-C | +735,603 teacher-prefix continuation from P2-0.86M | sa/sb | 0.7667 | 0.1300 | 0.1652 | not adopted — ties P2-1.60M on behaviour, appears to cost correctness; **weights lost** |
| E5-R | +735,603 student-prefix recovery from P2-0.86M | sa/sb | 0.4467 | 0.1100 | 0.2463 | **rejected** — worse than its own start point; **weights lost** |

**P2-1.60M is the current best checkpoint on both axes** and the reference any
Stage 4/5 work should start from. E5's two arms cost $11.64 across ten paid
events and neither beat it; the C and R weights were lost to a stale checkpoint
tag in the transfer path ([EXPERIMENTS.md](EXPERIMENTS.md) §27), so re-evaluating
either would require retraining.

All six trained at the 0.86M rung, 1,023 steps, η 5e-5, warmup 51, same Stage 1
init (`86fbba78…`), teacher `Qwen/Qwen3-4B-Thinking-2507@768f209d`.

**Weight retention differs sharply:** P2 is local and hash-verified; **P1 exists
only on the storage-constrained relay**; P0-assistant is gone.

## Experiment 1 initialization arms — the Stage 0/1 downstream test

| | behaviour usable rollout, PCA vs random |
| --- | --- |
| 0.25M | 0.0132 / 0.0789 vs **0.0000 / 0.0000** |
| 0.46M | 0.2632 / 0.1447 vs **0.0000 / 0.0000** |
| 0.86M | 0.3684 / 0.4342 vs **0.0000 / 0.0000** |
| 1.60M | 0.4868 / 0.5132 vs **0.0000 / 0.0000** |
| **2.96M** | **0.5921 / 0.5395** vs **0.0000 / 0.0000** |
| 5.50M | 0.5526 / 0.5395 vs 0.0658 / 0.0921 |

**PCA initialization: 12 wins / 0 ties / 0 losses on behaviour prompts, and
11 wins / 1 tie / 0 losses on gsm8k** — the tie is 0.25M `sb`, where both
initializations score 0.0000 (a shared floor, not a contest PCA failed to win).
Random init produces zero usable rollouts at every rung through 2.96M. This is the
strongest result in the project.

> All values in this section: **n=76 behaviour prompts / n=100 gsm8k prompts,
> E1 behaviour-wave harness, degeneration stop ACTIVE.** They are **not
> comparable** with the 150-example three-mode rates in the table above — the same
> weights score 0.3684 here and 0.5133 there.

**2.96M and 1.60M behave better than the 0.86M rung** every Stage 2/3 candidate
was trained at. Those rungs have never been run through the 150-example harness
and are **not evaluable** on the Stage 2/3 candidate set without new generation.

**Recoverability verified 2026-08-05:** 30/30 local files match their pod-side
manifests; relay LFS digests recorded in `artifacts/audit/relay_e1_digests.json`,
with `e1_r1600k_sa_pca` and P1-sa downloaded and recomputed byte-exact. Every rung
is covered across local + relay. **P1's weights exist only on the relay** — see
the storage risk in [`STATE.md`](STATE.md).
