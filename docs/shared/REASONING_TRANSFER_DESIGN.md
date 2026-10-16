# Reasoning transfer: the forward design for Stage 0, Stage 2 and Stage 4

**Status: forward-looking design.** It authorizes nothing. No dataset is
downloaded or generated, and no experiment starts, because of this page. Each
experiment below needs its own authorization, budget and preregistration.

Owners:
- the objective and the capability taxonomy: [`AGENTS.md`](../../AGENTS.md)
  P10.2, with the stage clauses in §4.2 (Stage 0), §4.4 (Stage 2), §4.5
  (Stage 3) and §4.6 (Stage 4);
- the recipe's capability scope: [`decisions.md`](../../logs/budget/decisions.md),
  2026-10-16, "Maintainer clarification: the primary target is fluid reasoning".

This page uses the axis names that P10.2 defines and does not restate them.

## 1. What the existing evidence can and cannot say

Every behavioural result so far was measured on mixed-domain batteries. The
D-series battery (`d_series_shared_eval@v1`) has seven sets. Its scores stay
valid as what they measured; none of them isolates a primary axis.

| set | mainly measures | axes mixed in it |
| --- | --- | --- |
| `gsm8k`, `math_verified` | GSM8K and verified MATH problems | task-contained deduction + supporting knowledge (arithmetic, conventions, learned procedures) |
| `knowledge` | TriviaQA | supporting knowledge |
| `rag` | SQuAD: answering from a supplied passage | reading comprehension + light task-contained deduction |
| `multihop` | HotpotQA | task-contained deduction over evidence + reading and retrieval |
| `code` | MBPP program synthesis | learned domain skill + reasoning |
| `tool` | xLAM tool-call decisions | protocol integrity + agent decision |

- **No measurement exists** of inductive rule discovery, compositional
  generalization to unseen rules, or interactive adaptation.
- **Low correctness here does not measure fluid reasoning.** c4's 3 correct
  answers out of 900 GSM8K + MATH prompt-seed observations (D2 confirmation)
  is a fact about those sets, not about fluid reasoning.
- **Neither does a usable-rollout gain.** The usable-rollout gains of q2 and
  c4 over B are readiness evidence only.

## 2. Prerequisite: a capability-aligned evaluation

The data experiments below need an endpoint that measures the primary axes, so
the evaluation is designed first. It is a $0 design pass; building it needs its
own authorization.
- **Per-axis batteries.**
  - One battery per primary axis, plus a supporting-knowledge battery and the
    usable-rollout guard rail. Each is reported separately.
  - No weighted scalar combines them.
- **Unseen families.** Every task carries a rule family, a transformation
  family, a generator identity and a source lineage. Evaluation families are
  disjoint from every family used in calibration or training, and the
  isolation is checked by those keys, not assumed.
- **Resolution.**
  - The difficulty strata must keep the student off the floor. At 2–6%
    correctness, the current battery cannot separate initializations.
  - Each battery is sized for a registered SESOI before any data experiment
    reads it.
- **ARC-AGI.**
  - Static ARC-AGI-1/2-like tasks and interactive ARC-AGI-3-like environments
    are separate batteries with separate protocols.
  - A textual grid encoding is a research interface. Its scores are never
    presented as official ARC-AGI scores.
  - No official ARC-AGI evaluation task, solution, trajectory or trivial
    derivative enters any training, calibration or development data.
- **Measurement rules.** Generation is unrestricted (P18), answers are
  verifier-scored, and the complete generations are kept.
- **History is not imported.** The retired B/q2 checkpoints have no scores on
  a new battery, and none are imputed.

## 3. Stage 0: calibration volume and mixture

**Question.** How do calibration-set size and composition change which
structure the AutoInitializer keeps, and what that structure retains on the
primary axes?

**Factors**, varied one at a time:
- **Volume.** Counted in effective scored token positions, i.e. the positions
  a selection objective actually reads, not raw tokens.
- **Mixture proportion by axis.** Rule-induction, compositional,
  task-contained-deduction and supporting-knowledge content.
- **Reasoning length and diversity of task structure.**

**Controls.** The geometry, the operators and their search budget, the
recovery recipe, and the seeds.

**Measurements:**
1. **Signal preservation.** The fraction of scored positions inside reasoning
   spans, per axis. This checks whether sampling or truncation removed the
   positions the mixture was meant to supply.
2. **Structural selection.**
   - The agreement of the DEPTH, FFN, WIDTH and ATTENTION choices across
     volumes, across mixtures, and across resamples of the same mixture.
   - Selection that moves under resampling is noise, not a mixture effect.
3. **Initialization stability.** Step-0 NLL on per-axis held-out sets. This is
   kept separate from every other metric (AGENTS.md §4.2).
4. **Downstream retention.** Per-axis correctness after a fixed recovery, on
   the §2 batteries.

**Design.**
- First, dose-response in volume at a fixed mixture.
- Then mixture comparisons at fixed effective positions.
- Teacher KL and NLL stay diagnostics; the endpoint is measurement 4.

## 4. Stage 2: offline transfer of generalizable reasoning

**Question.** Which offline data transfers the teacher's problem-solving
process to unseen rule families?

**Data:**
- self-contained tasks with verifiable solutions;
- generated rule-induction, symbolic and spatial transformation, compositional
  and counterfactual tasks, at controlled difficulty;
- teacher-native trajectories, kept only when the result is verified correct;
- the supporting language and conceptual data these need.

**Factors**, each varied with the others held fixed:
1. raw quantity;
2. effective reasoning-token count;
3. difficulty;
4. rule-family diversity, i.e. families against examples per family;
5. trajectory quality, i.e. verified-correct against unverified trajectories;
6. mixture proportion, including the supporting-knowledge share.

**Two confounds are controlled by design:**
- **Mixture and diversity comparisons hold the effective reasoning-token count
  fixed.** Otherwise they would measure quantity, not composition.
- **Trajectory length is controlled or stratified.** A longer chain of thought
  is not better by default.

**Endpoint and guard rails.**
- The endpoint is per-axis correctness on held-out families.
- Supporting knowledge is a separate metric.
- Usable rollout remains the Stage 2/3 recovery-readiness guard rail, never
  the endpoint.

## 5. Stage 4: student on-policy reasoning data

**Question.** Which on-policy data teaches the student to reason and adapt
during its own execution?

**Data:**
- exploration of unfamiliar tasks;
- reasoning from observations and feedback;
- corrective attempts after failure;
- hypothesis formation and testing;
- multi-step planning and action selection;
- interactive environments with verifiable outcomes.

Successful and failed trajectories are both kept, each with its feedback,
verifier identity, policy identity and token-level rollout record (AGENTS.md
§4.6).

**Metrics:**
- task success;
- generalization to unseen environments and rule families;
- improvement through feedback, i.e. success as a function of the attempt
  index;
- action and sample efficiency;
- for interactive environments, completion and adaptation efficiency.

**Not rewarded by itself:**
- valid formatting;
- natural termination;
- short answers;
- confident predictions.

## 6. Relation to the D-series

The new D3 (fixed-q2 DEPTH teacher-reference ablation, decision 2026-10-16) is
measured on D1's historical confirmation battery.
- Its comparisons with q2 and B are exploratory measurements of the
  mixed-domain protocol.
- An improvement there in correctness or usable rollout is not a
  demonstration of fluid reasoning.

## 7. Order of work and core boundary

**Order.** Each step needs its own authorization:
1. §2 evaluation design;
2. building the §2 batteries;
3. the §3 Stage-0 study;
4. the §4 Stage-2 study;
5. the §5 Stage-4 study.

**Core boundary.**
- **Generic mechanisms belong in `src/aadistill`:**
  - effective-position accounting;
  - isolation-key checks;
  - verifier interfaces;
  - per-axis reporting.
- **Instance material belongs in configs and scripts:** task families,
  generators, rule-family splits, batteries and benchmark adapters.
- Nothing ARC-specific or model-specific enters the core (AGENTS.md P3,
  P10.2).
