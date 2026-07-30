# 2026-07-31 — Teacher prompt contract: the system-coverage defect is RETRACTED

- **CPU only, $0.** Verified against the pinned revision
  `Qwen/Qwen3-4B-Thinking-2507@768f209d`, not against inference or defaults.
- **Supersedes** the "confirmed coverage + train/inference mismatch" claim in the
  2026-07-30 audit. That claim was **wrong as stated**.

## 1. The exact prompt contract

Authoritative source: `tokenizer_config.json` → `chat_template`, sha256
`3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7`
(4,049 chars; there is no standalone `chat_template.jinja` at this revision).

The template's entire system-message logic is:

```jinja
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}{{- messages[0].content + '\n\n' }}{%- endif %}
    {{- "# Tools\n\n…" }}                    {# tool spec forces a system block #}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}                              {# …and NOTHING otherwise #}
{%- endif %}
```

There is **no `{%- else %}` fallback and no default system string anywhere**.
Contrast Qwen2.5, whose template hard-codes *"You are Qwen, created by Alibaba
Cloud…"*. This model has no such default.

Measured on the pinned tokenizer:

| render | system block emitted? | opens `<think>`? |
|---|---|---|
| user only, `add_generation_prompt=True` | **False** | True |
| system + user | True (verbatim, `SYS` preserved) | True |
| user + `tools` | True (tool spec; user system prepended if present) | True |

**`<think>` injection is fully independent.** It comes from the unconditional
tail `{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n<think>\n' }}`,
which has no relationship to system or tools — confirmed opening in all three
renders above.

Model card at the pinned revision: **zero occurrences of "system"**. It states
*"This model supports only thinking mode. Meanwhile, specifying
`enable_thinking=True` is no longer required."* — so thinking is unconditional
and our not passing `enable_thinking` was correct.

## 2. Verdict — retraction

**The teacher does not require a native system message.** Supporting the
`system` role is not the same as requiring a default system prompt: this
template renders a supplied system turn verbatim, forces a system block only to
carry a tool spec, and otherwise emits none.

Therefore **generating the corpus without a system message is native-correct**.

* The corpus is **not misconditioned**.
* **No affected scope**, no correction pilot, no regeneration on these grounds.
* The "confirmed coverage + train/inference mismatch" finding is **retracted**;
  it conflated *"the data has no system turns"* with *"the protocol demands
  one"*.

## 3. All four in-repo paths share one rendered protocol

Sample `hotpot-000000`, prompt-prefix sha256 (through
`<|im_start|>assistant\n<think>\n`):

| path | system block | prefix sha256 |
|---|---|---|
| 1 teacher-corpus generation | False | `b19e629cfb95c566` |
| 2 public-target training | False | `b19e629cfb95c566` |
| 3 teacher-target training | False | `b19e629cfb95c566` |
| 4 student evaluation | False | `b19e629cfb95c566` |

**Byte-identical.** Teacher generation, both training arms and evaluation used
the same rendered protocol.

Assistant-turn rendering differs only where it should:

* public → `…assistant\n<think>\n\n</think>\n\nArthur's Magazine<|im_end|>`
  (**empty** think block — the protocol substitution, as previously found);
* teacher → `<think>\n{trace}\n</think>\n\n{answer}<|im_end|>`.

## 4. SUPERSEDED 2026-07-31 by the project protocol

Sections 1-3 are **template facts** and stand. The framing below — "absent
capability, not a defect", "much smaller" — was written before the maintainer
set the project protocol, under which **an explicit system message is
mandatory** for teacher generation, training, primary evaluation and inference
(proposal §13). Under that protocol the missing system conditioning IS a
training-coverage problem covering 100% of the primary target distribution, and
the 6 system-conditioned eval prompts are retained precisely because their
mismatch reveals it. Retained verbatim below for the record.

## 4a. What genuinely remains (restated, much smaller) — SUPERSEDED

1. **6 of 76 evaluation prompts carry a system turn** the student has never seen
   in training. That is a **7.9% evaluation-side inconsistency**, not a corpus
   defect. It affects `instruction` (1) and `short_realtime` (5) only — neither
   is in the trained in-scope four.
2. **Deployment system-prompt capability is absent, not broken.** If AlphaAvatar
   will condition on personas/tool contracts, that is a *new capability to add*,
   scoped as its own objective — and any such data must be **generated with the
   teacher conditioned on the system prompt**, never retrofitted to existing
   targets. It is not a defect in the current corpus.

## 5. Two real deviations this recheck did surface

### 5.1 Evaluation used a narrower stop set than the teacher's native one

The teacher's `generation_config.json` lists **`eos_token_id: [151645, 151643]`**
(`<|im_end|>` *and* `<|endoftext|>`). The unrestricted pilot derived its stop set
from the tokenizer alone and ran with `stop_token_ids: [151645]`.

Impact on the pilot: **none** — every checkpoint it ran (Stage 1 init and its
descendants) has `eos_token_id: 151645` alone, so 151645 was their native stop
set. But the code would have been wrong for the teacher. Fixed: stop ids are now
unioned from the model's `generation_config` and config, not the tokenizer.

### 5.2 Teacher-corpus sampling deviates from the model card

| param | Qwen official | teacher corpus | pilot (student) |
|---|---|---|---|
| temperature | **0.6** | **1.0** | 0.0 (greedy) |
| top_p | **0.95** | **1.0** | 1.0 |
| top_k | **20** | **0 (disabled)** | -1 (disabled) |
| min_p | 0 | unset | unset |
| presence_penalty | 0–2 suggested to reduce endless repetition | unset (0.0) | unset (0.0) |

The temperature/top_p/top_k deviation is **deliberate and logged** (decision
2026-07-29: untruncated sampling, DAPO/GRPO practice) — but it is a deviation
from the model card and belongs in the prompt contract, which it was not.

Worth flagging next to the degeneration finding: the model card explicitly
suggests `presence_penalty` 0–2 *"to reduce endless repetitions"*. That is about
the teacher's own decoding, not the student's training, so it does not explain
the student's loops — but it means our student evaluation runs with no
repetition control while the vendor recommends one be available.

## 6. Scope and next step

Nothing here changes the unrestricted-pilot conclusion: `s2v1@2700` degenerates
8/8, and degeneration is a property of the whole student line rather than of
teacher-native targets. The prompt contract is clean; the open problems remain
convergence, coverage and exposure bias
([proposal §12](../../proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md)).
