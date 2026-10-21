# eval-set-1 — third-party content and licenses

This set redistributes items drawn from public datasets. Each file below keeps
the license of its source; files derived from CC-BY-SA-4.0 sources are
themselves CC-BY-SA-4.0 (ShareAlike). License bases were checked on
2026-10-18 against each dataset card (Hugging Face API `cardData.license`) and,
where the card is silent, the upstream repository's license.

| file(s) | source (revision) | license | basis |
| --- | --- | --- | --- |
| `fluid_reasoning_720/*` | project-generated procedural tasks | Apache-2.0 | repository license; no external data |
| `stage3_capability_846/battery_v2/math_verified.jsonl` | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) test, from [openai/prm800k](https://github.com/openai/prm800k) / [hendrycks/math](https://github.com/hendrycks/math) | MIT | the card carries no license field; both upstream repositories are MIT |
| `stage3_capability_846/battery_v2/gsm8k.jsonl` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` test | MIT | dataset card |
| `stage3_capability_846/battery_v2/multihop.jsonl` | [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa) `distractor` validation | CC-BY-SA-4.0 | dataset card; this file is CC-BY-SA-4.0 |
| `stage3_capability_846/battery_v2/rag.jsonl`, `answerability_paired.jsonl` | [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2) validation | CC-BY-SA-4.0 | dataset card; these files are CC-BY-SA-4.0 |
| `stage3_capability_846/battery_v2/safety_paired.jsonl` | [Paul/XSTest](https://huggingface.co/datasets/Paul/XSTest) (Röttger et al., 2024) | CC-BY-4.0 | dataset card; upstream [paul-rottger/exaggerated-safety](https://github.com/paul-rottger/exaggerated-safety) |
| `stage3_capability_846/eval_behavior_v0/prompts.jsonl` | `stage2_v1` val split: OpenAssistant/oasst2, HuggingFaceTB/smol-smoltalk, HuggingFaceTB/everyday-conversations-llama3.1-2k, glaiveai/glaive-function-calling-v2 (Apache-2.0); Salesforce/xlam-function-calling-60k (CC-BY-4.0, access-gated); rajpurkar/squad_v2, hotpotqa/hotpot_qa (CC-BY-SA-4.0); openai/gsm8k (MIT); nvidia/OpenMathInstruct-2 (CC-BY-4.0, built with Llama) | per source | dataset cards; tracked in this repository since 2026-07-27 |
| `stage3_legacy_150/items.jsonl` | `corpus_v2` sessions: openai/gsm8k train (MIT); nvidia/OpenMathInstruct-2 `train_1M` (CC-BY-4.0; generated with Llama 3.1 — "Built with Llama"); hotpotqa/hotpot_qa train, rajpurkar/squad_v2 train (CC-BY-SA-4.0) | per source | dataset cards; prompts and source gold only — no teacher output is included |
| `conformance/*` | model outputs of this project's own runs (Qwen3 family models, Apache-2.0) on the items above, plus synthetic strings | same as the item they answer | — |

## Withheld

| file | source | why it is not in the repository |
| --- | --- | --- |
| `stage3_capability_846/battery_v2/knowledge.jsonl` (150 prompts) | [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa) `rc.nocontext` validation | the dataset card's license is `unknown`, and it states that the University of Washington does not own the copyright of the questions and documents. The Apache-2.0 the battery manifest records is the license of the TriviaQA *code* repository, not of the question text. No basis to republish in a public repository without maintainer review. |

Its identity (sha256, size) is in `manifest.json :: withheld`, and the verified
local copy is the durable artifact `artifacts/stages/stage-3/eval/battery_v2/knowledge.jsonl`.
The exact sample ids are public in `battery_v2/manifest.json`, so the file can
be rebuilt from the source by `scripts/stages/stage-3/e2/build_capability_battery.py`.

## Outside the formal core (still in the repository)

| items | source | why |
| --- | --- | --- |
| 7 `eval_behavior_v0` prompts `xlam-*` | [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (CC-BY-4.0, access-gated) | the dataset is gated behind an access agreement whose redistribution terms have not been confirmed; maintainer decision 2026-10-19: they stay outside the public frozen core (`stage3_capability_846/core_v1.json`) until they are. They have been in this repository since 2026-07-27. |

## Attribution

- HotpotQA: Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP 2018.
- SQuAD 2.0: Rajpurkar, Jia and Liang, *Know What You Don't Know: Unanswerable Questions for SQuAD*, ACL 2018.
- XSTest: Röttger et al., *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*, NAACL 2024.
- GSM8K: Cobbe et al., *Training Verifiers to Solve Math Word Problems*, 2021.
- MATH: Hendrycks et al., *Measuring Mathematical Problem Solving With the MATH Dataset*, NeurIPS 2021; MATH-500 subset from Lightman et al., *Let's Verify Step by Step*, 2023.
- OpenMathInstruct-2: Toshniwal et al., NVIDIA, 2024. Built with Llama.
