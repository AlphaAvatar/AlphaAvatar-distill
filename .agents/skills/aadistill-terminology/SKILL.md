---
name: aadistill-terminology
description: The maintainer's corrected field terminology for AlphaAvatar-distill reports, replies and docs -- how to count data (examples, turns, packed sequences, supervised tokens, task categories), name capabilities, terminations, comparisons and full-search data roles. Use before writing any status report, data audit, plan or reply that states data volume, training scale, evaluation results or experiment arms.
metadata:
  owner: maintainer
  source: corrections made in sessions 2026-10-27 to 2026-10-31; AGENTS.md 2.10
---

# AlphaAvatar-distill terminology

AGENTS.md 2.10 is the rule: call every concept by the name the field uses. This skill lists the definitions the
maintainer has had to correct, so they are not repeated. Code identifiers keep their spelling (frozen records bind
them); prose uses the standard term, with the identifier in a code span only when the reader needs the key.
Replies to the maintainer are in Chinese; dates are UTC.

## 1. Counting data

| say (中文 / English) | it means | never |
| --- | --- | --- |
| **样本 / 条数据** (examples) | one problem instance: one prompt, or one whole interactive episode | "任务" for an instance |
| **轮 / 轮数** (turns) | one assistant reply with its context; a static example is 1 turn, a tool-calling conversation or an interactive episode is several | "文档 / 文档数" (`document` is a code identifier) |
| **任务类别 / 任务数** (task categories) | how many KINDS of task: GSM8K, HotpotQA, grid_transform, a rule family... | a count of examples |
| **打包序列** (packed sequences) | the 8,192-token training sequences turns are packed into | "block" |
| **监督 token** (unique supervised tokens) | the generated tokens a turn trains on (CE/KD positions), counted once | "tokens" alone, or packed sequences as a data size |
| **监督 token 曝光量** (supervised-token exposure) | unique supervised tokens x epochs | the unique count |
| **非 padding token** (non-padding tokens) | prompt/context plus supervised tokens actually in the sequences | "real / executed positions" |
| **提示 / 上下文 token** (prompt / context tokens) | non-padding tokens that are not supervised | |
| **推理 token / 答案 token** | supervised tokens through `</think>` / after it | |

Report a data set as **样本数 / 轮数 / 打包序列数 / 监督 token (/ 任务类别)**, e.g. D4-Augmented's training set:
4,996 examples, 5,291 turns, 2,468 packed sequences, 5,923,021 supervised tokens. A size target is stated in
supervised tokens, never in packed sequences.

## 2. Training terms (AGENTS.md 2.10)

| in code | say |
| --- | --- |
| block, `block_len` | packed sequence (打包序列), sequence length (序列长度) |
| `blocks_per_step` | batch size (批大小), sequences per optimizer step |
| `micro_blocks` | micro-batch; gradient accumulation steps = `blocks_per_step / micro_blocks` |
| passes | epochs (轮次 / epoch) |
| pack | packed training set |
| `kd_scope` `assistant` | KD (and CE) on the assistant target positions only |
| `reference_topk_tail_v1`, K = 200 | Top-200 + tail KD support |

## 3. Comparisons and compute

- **Token-matched (监督 token 匹配), step-matched (步数匹配) and FLOPs-matched (计算量匹配) are three different
  comparisons.** Equal supervised tokens never imply equal compute: report non-padding tokens, packed sequences,
  optimizer steps, exposure and training FLOPs beside it.
- A **same-tasks** comparison holds the examples fixed and lets supervised tokens differ; never pooled with a
  token-matched one.
- **Arms are described in words** (思考模式、非思考模式、同一批题...), not with letter labels the design records
  coined (H-T-S, H-N-C, sets I / X, comparisons S / C). Names the maintainer assigned (A1, A2, C1) may be used; say
  what each is the first time.
- Modes: **思考模式 / 非思考模式** (thinking / non-thinking: `enable_thinking=True/False`); `H-T` / `H-N` are record
  keys, not prose.

## 4. Capabilities and data categories (P10.2)

- Primary: **fluid reasoning** axes -- task-contained deduction (任务内演绎), inductive rule discovery (归纳规则发现),
  compositional generalization (组合泛化), interactive adaptation and planning (交互适应与规划); abstract/analogical
  (抽象类比) is a reported sub-category.
- **Supporting** (支持类能力): math and code are learned procedures; SQuAD/HotpotQA-style reading answers from
  evidence SUPPLIED in the prompt (基于给定证据的阅读 / 多跳阅读).
- **None of these is "knowledge reasoning" (知识推理)**: that would mean closed-book recall. E1's corpus_v2 is the
  "base-corpus" category (基础语料对应类别), supporting types, not knowledge reasoning, and Research A v1's
  supporting strata are not E1.
- A mixed-domain score (math, RAG, code...) is never read as fluid reasoning.

## 5. Terminations and limits

| say | it is |
| --- | --- |
| **自然终止** (natural termination) | the model emitted its EOS / `<|im_end|>` |
| **重复 / 退化终止** (repetition / degeneration stop) | the engine's degeneration check stopped it |
| **超出训练长度** (training-length-limited; right-censored at 8,184 minus the prompt) | a training-data turn that reached its allowance; refused, never cut |
| **达到上下文上限** (context-limit hit) | evaluation reached the model's supported context (40,960 for the hybrid) |

`truncated_at_cap` counts EVERY non-EOS stop, most of them degeneration stops: never call it "truncated at the
context limit". Right-censored observations are reported as censored, not as wrong answers.

## 6. Full-search data roles (three different sets)

| set | role |
| --- | --- |
| **校准数据** (calibration profiles) | read INSIDE the operators: activation statistics, PCA, DEPTH's KL against the parent -- decides HOW to compress; the profile per operator is a searched variable |
| **状态评估数据** (state evaluation suite) | scores every candidate against the root teacher (equal-domain mean KL, worst-domain KL, critical-token KL) -- drives beam pruning and the Pareto ranking, so it picks the Top-1 |
| **eval-set-1** | the frozen formal capability evaluation AFTER training; never used to select a candidate |

Calibration and state evaluation share one type definition and are disjoint in examples; neither is eval-set-1.

## 7. Data provenance words

- **Teacher-native / token-native**: the engine's own prompt and output token ids, as generated; **re-rendered**:
  decoded and re-tokenized (not identical).
- **Teacher lineage** (教师谱系): root teacher -> calibration -> search -> initialization -> training data -> online KD
  teacher -> checkpoint, one teacher identity end to end.
- **Verified correct** (验证正确) is an answer checked against a gold; "hygiene acceptance" is not correctness.

## 8. Reporting order

Lead with the work (done / running / blocked, the numbers above), then decisions owed to the maintainer, then the
evidence; test counts come last and are never the body of a report.
