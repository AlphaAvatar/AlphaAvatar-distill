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
| qwen3-4b-thinking-distill (working name) | `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d` | 0.6B-class: hidden 1024, 28L, FFN 3072, 16Q/8KV, tied emb (user-chosen 2026-07-13) | **stage3-running** | Stage 0, Stage 1, Stage 2 (v0+v1); Stage 3 sub-stages 1–2; Experiment 1 data-scaling matrix complete; Experiment 2 phase 1 (data cleaning) complete | `e1_r5500k_sa_pca` — teacher-native held-out CE **1.0032**; best behaviour/termination `e1_r2960k_sa_pca` (0.4413 / **0.934**). 20 arms on `AlphaAvatar/aadistill-artifacts`, 5 held verified on the dev box | Natural termination rose from 0/8 to **0.934** on teacher-native recovery data. Stage 3 still not exited: **no reasoning at any rung** (GSM8K EM ≤0.05). See below. |

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
