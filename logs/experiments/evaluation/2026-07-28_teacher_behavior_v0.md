# 2026-07-28 — Teacher scorecard on `eval_behavior_v0` (GPU): the project's first ceiling

- **Agent:** Claude Code (Opus 5), session `aadistill-teacher-pilot`.
- **Git commit:** `a58a1ca` (pod clone verified against the dev-box bundle hash).
- **Objective:** Measure the teacher on the same behavior eval the students are
  scored on. Until now the figure had a student point and **no ceiling**, so
  every "the teacher answers better" statement in the proposals was an
  assumption.
- **Teacher:** `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d`, bf16, **native
  thinking mode, no prefill** (decision record 2026-07-28), cap 4096.
- **Hardware / budget:** 1× L40S (RunPod, $0.99/h), 70 min of generation, pod
  deleted by the fetch driver on completion. **~$1.20**, inside the approved
  ~$1–1.5.
- **Command:**
  `uv run python scripts/eval_behavior.py --model Qwen/Qwen3-4B-Thinking-2507@768f209d --max-new-tokens 4096 --out artifacts/teacher/eval_behavior_v0.json`

## Result

**`behavior_score_v0` = 0.7443** (student `s2v1_from_init@2700`: 0.2015).

| axis | student | teacher | gap |
|---|---:|---:|---:|
| math (gsm8k EM, n=7) | 0.000 | 0.714 | **+0.714** |
| tool_call (n=12) | 0.250 | 0.917 | **+0.667** |
| format_ok (n=76) | 0.224 | 0.842 | **+0.618** |
| refusal (n=12) | 0.167 | 0.667 | +0.500 |
| fluency (n=76) | 0.319 | 0.764 | +0.445 |
| grounding (n=16) | 0.250 | 0.562 | +0.312 |

Form metrics: `terminated` 0.842, `truncated_at_cap` 0.158, `think_closed`
0.882, `empty_answer` 0.118, `rep_3gram` **0.043**, `answer_words` 146.

Trace + answer length: **median 1,162 tokens, p90 4,096 (the cap), max 4,096.**

Per group, `format_ok`: rag 1.000 · multihop 1.000 · tool_calling 1.000 ·
code_math 0.917 · refusal 0.833 · short_realtime 0.833 · **instruction 0.417**
(truncated 0.583 — open-ended instruction prompts are where the teacher thinks
past 4096).

## What this establishes

1. **The eval is not the problem.** A competent model scores 0.74 on it, so the
   student's 0.20 is the student's, not the harness's. That is the single most
   useful thing a ceiling row buys, and it could not be asserted before today.
2. **Degeneracy is a student pathology, not something inherited.** Teacher
   `rep_3gram` 0.043 vs student 0.408. The teacher also writes *shorter* final
   answers (146 words vs 217) while thinking longer — so training on traces does
   not imply verbose answers.
3. **Grounding has a low ceiling: 0.562.** The credited-grounding metric is
   strict (echo credit), so chasing student grounding much past ~0.56 is
   chasing a ceiling that is not there. Effort belongs on math, tools and form.
4. **`block_len` evidence.** Median trace+answer is 1,162 tokens — comfortably
   inside a 2048 block — but the p90 is at the cap. A trace corpus will have a
   long tail no block size absorbs; best-fit packing plus a counted truncation
   rate is the right handling, not a bigger block.
5. **The 4096 cap is not generous enough for `instruction`.** 58% of those
   prompts hit it. The teacher's score is therefore a *lower bound*.

## Caveat that must be carried forward

**The student rows were scored at cap 512, the teacher at 4096, so the form
metrics are not strictly comparable.** The student's `truncated_at_cap` 0.632
is partly the smaller cap, which means its `format_ok` may be understated. Every
student checkpoint must be re-scored at 4096 before this table is quoted as a
like-for-like gap; that is folded into the next session. The content axes
(math, tool, grounding, refusal credited) are less affected — they are about
what the answer says, not where it stopped.

## Verdict / next

Ceiling established and written into `assets/perf_trend.json`, so the README
figure now shows both ends of the gap. Next actions in `logs/STATE.md`; the
immediate one is the packing/`block_len` control run, which re-scores the
references at 4096 in the same session and removes this caveat.
