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
| qwen3-4b-thinking-distill (working name) | `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d` | 0.6B-class: hidden 1024, 28L, FFN 3072, 16Q/8KV, tied emb (user-chosen 2026-07-13) | **stage3-running** | Stage 0, Stage 1, Stage 2 (v0+v1); Stage 3 sub-stages 1–2 | `stage3/s2v1_from_init/step_002700/model` — holdout NLL **3.8285**, `behavior_score_v0` **0.2015**; fp32, HF-only on `AlphaAvatar/aadistill-artifacts` | Stage 3 has **not** exited: no checkpoint generates usable output under unrestricted generation. See below. |

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

**Why Stage 3 has not exited.** Under unrestricted generation (P18, full 262,144
context, no token cap) **every** checkpoint in this line — including
`s2v1_from_init@2700` — degenerates into repetition, with **zero** context-limit
hits. The 512-token evaluation cap used before 2026-07-30 was hiding repetition
loops, not long reasoning. Neither 2026-07-30 four-arm run supports a
route-level claim about teacher-native supervision: one forked from a
public-trained checkpoint (invalid), and both were convergence- and
measurement-limited.

**Largest measured capability gaps** against the teacher's `behavior_score_v0`
ceiling of 0.7443: math EM +0.714, tool_call +0.667, format_ok +0.618. These
gaps set how many prompts of each type the recovery corpus was generated from.

**What exists for the next step.** A teacher corpus (corpus v2, 2026-08-01):
11,174 accepted sessions, 66.08M generated tokens, gate-passed, cut into a
six-rung nested token ladder. The first experiment over that ladder asks only
**whether behavioural recovery scales with supervised-token count**, so it runs
on a neutral uniform type mixture; data mixing and difficulty curriculum are
separate later experiments (maintainer, 2026-08-01). The training matrix is
specified and costed but **not started** — it awaits a go-ahead against the $60
training budget ([`PROPOSAL.md`](PROPOSAL.md)).

**Deployment target:** INT8. Every recovery gate already re-evaluates under INT8
weight fake-quantization at two scopes.
