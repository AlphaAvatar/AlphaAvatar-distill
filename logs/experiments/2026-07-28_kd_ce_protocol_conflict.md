# 2026-07-28 — CE and KD are in direct opposition at `</think>`; the student equilibrates between them

- **Agent:** Claude Code (Opus 5). **Hardware:** CPU dev box only. **Cost: $0.**
- **Git commit:** `014eba0`.
- **Trigger:** maintainer hypothesis that the seed-driven behavior variance
  (noise floor 0.1290, measured earlier the same day) comes from open-source
  target diversity plus a small student's parameter sensitivity, and that
  teacher-generated, more consistent answers would reduce it.
- **Objective:** test that hypothesis before spending on it.

## Result: the hypothesis is right, for a sharper reason than target diversity

**Target diversity is not the cause of the protocol instability, because the
protocol in the targets is perfectly uniform.** Sampling 3,200 Stage 2 v1 rows:
**zero** assistant messages contain a `<think>` tag in their content. The
protocol comes entirely from the chat template's empty-think rendering, so every
single training example carries an identical `<think>\n\n</think>` prefix. The
student is failing to reproduce a signal present identically in 100% of examples.

The cause is a **direct conflict between the two training objectives at exactly
that token.**

At the position immediately after `<think>\n\n`, context
`'\n<|im_start|>assistant\n<think>\n\n'`:

| signal | wants | weight |
|---|---|---|
| **CE** (loss mask confirmed `True`) | `</think>`, one-hot | 0.25 |
| **KD** (`kd_scope: all`, τ=1) | `Okay` / `Hmm` at p≈1.0; **p(`</think>`) = 0.000000** | 1.0 |

Measured teacher distribution at that position (bf16, 8 samples across 4 groups):

| group | p(`</think>`) | teacher's top token |
|---|---:|---|
| instruction | 0.000000 | `Okay` 1.000 |
| instruction | 0.000000 | `Okay` 0.938 |
| tool_calling | 0.000000 | `Okay` 1.000 |
| tool_calling | 0.000000 | `Okay` 1.000 |
| rag_evidence | 0.000000 | `Hmm` 0.949 |
| rag_evidence | 0.000000 | `Hmm` 0.915 |
| code_math | 0.000000 | `Okay` 0.992 |
| code_math | 0.000000 | `Okay` 0.996 |

**Per-position, KD pulls 2.0× harder than CE.** The loss is
`0.25·ce_sum/ce_total + 1.0·kd_sum/kd_total`, and measured over both control
arms `ce_targets`≈7,950–8,060 against `kd_positions`≈15,720–15,760, so the
per-position ratio is `(1.0/kd_total)/(0.25/ce_total)` = **2.02× and 2.05×**.
The weaker signal is the one that wants a well-formed answer.

## The student sits exactly where that predicts

p(`</think>`) at the contested position, measured on local checkpoints (fp32):

| checkpoint | p(`</think>`) | argmax |
|---|---:|---|
| Stage 1 init | 0.0000 | `.` (0.560) — protocol not learned at all |
| s1_ffn_norm@660 | **0.3342** | **`Okay` ≈0.63** |
| teacher | 0.0000 | `Okay` ≈1.0 |
| CE target | 1.0000 | `</think>` |

Per group: instruction 0.3528 · tool_calling 0.3306 · rag_evidence 0.3236 ·
code_math 0.3299. **A spread of 0.029 across four unrelated groups** is the
signature of a systematic force balance, not of noise.

Recovery *is* teaching the protocol — the init is at 0.0000 and 660 steps move
it to 0.33 — but KD holds it there instead of letting it go to 1.0.

## Why this explains what was measured earlier today

1. **Why the student fails a perfectly consistent signal.** It is being taught
   the opposite by a 2× stronger one.
2. **Why behavior is unstable across seeds while loss is not.** The loss is a
   weighted sum of two opposed terms and its *value* is stable — arms A and B
   differ by 0.028 in mean loss against a within-run sd of 0.81, and by 0.09% on
   holdout. What is balanced on a knife edge is the **argmax** at this position.
   Greedy decoding turns a near-tie into a discrete, correlated flip across many
   prompts at once — which is exactly the observed signature: `think_closed`
   0.5000 vs 0.8684, with the largest spreads on the n=76 axes rather than the
   small ones.
3. **Why more data or steps would not fix it.** The conflict is structural, not
   a sample-size problem. It has been present in every Stage 3 run since
   2026-07-22.

## Claim strength

**Measured:** the teacher's p(`</think>`)≈0; that the token is CE-supervised;
the 2.0× per-position ratio; the student's ~0.33 equilibrium; the uniformity of
the protocol in the targets.

**Inferred, well-supported but not proven by intervention:** that this conflict
*causes* the protocol failure and the seed instability. The decisive test is an
intervention — remove the conflict and see whether `think_closed` rises and
stabilizes. Two mechanism claims were already retracted today
(`2026-07-28_stage3_packing_control.md` §4), so this one is stated as a
hypothesis with a named test rather than a conclusion.

## Candidate fixes, cheapest first

1. **Exclude the template-inserted think block from KD.** A targeted mask: KD
   already accepts a content mask (added today for padding), so this is the same
   mechanism applied to a different span. Cheap to implement, short run to test,
   and it leaves the data alone.
2. **Drop the empty-think rendering** so targets do not contain a protocol the
   teacher contradicts. Changes every logged run's data path.
3. **Teacher-generated targets** (the maintainer's Stage 3 warm-up direction).
   Fixes it at the root: if the target *is* the teacher's own generation, the
   teacher is on its own manifold, and CE and KD agree everywhere instead of
   fighting. This finding is an independent argument for that plan — and it
   raises its priority, because (1) and (2) treat a symptom of training the
   teacher off-manifold while (3) removes the cause.
4. **Reweight** (raise CE, lower KD, or τ). Cheapest of all, but it trades one
   arbitrary balance for another and does not remove the contradiction.

Any of these must be evaluated with **≥2 seeds per arm** (standing rule from
the same day), and the protocol metrics — `think_closed`, `format_ok`,
`empty_answer` — are the readout, not the composite score.

## Reproduce

```
uv run python - <<'PY'   # teacher's distribution at the contested position
import sys, torch; sys.path.insert(0,"src")
from aadistill.data import load_split, render_chat
from transformers import AutoTokenizer, AutoModelForCausalLM
T="Qwen/Qwen3-4B-Thinking-2507"; REV="768f209d9ea81521153ed38c47d515654e938aea"
tok=AutoTokenizer.from_pretrained(T,revision=REV,local_files_only=True)
close=tok.encode("</think>",add_special_tokens=False)[0]
s=load_split("data/stage2_v1","train")["instruction"][0]
ids=tok(render_chat(tok,s),add_special_tokens=False).input_ids
i=ids.index(close)
m=AutoModelForCausalLM.from_pretrained(T,revision=REV,dtype=torch.bfloat16,local_files_only=True).eval()
with torch.no_grad(): lg=m(torch.tensor([ids[:i+1]])).logits[0,i-1].float()
print("p(</think>) =", torch.softmax(lg,-1)[close].item())
PY
```

## Links

- `logs/experiments/2026-07-28_stage3_packing_control.md` (the noise floor this explains)
- `logs/decisions.md` 2026-07-21 (empty-think targets), 2026-07-22 (KD design)
- `logs/decisions.md` 2026-07-28 (the Stage 3 SFT warm-up direction this supports)
