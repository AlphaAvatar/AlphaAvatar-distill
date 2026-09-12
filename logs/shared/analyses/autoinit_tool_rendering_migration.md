# Tool rendering: an evaluation-protocol migration question, not a bug fix

**Status: STOPPED for a maintainer decision.** The audit was run at $0 over the
**full** frozen tool subset (all 20 items, not a sample) with
`scripts/autoinit/audit_tool_rendering.py`; the artifact is
[`autoinit_tool_rendering_audit_tf5.json`](autoinit_tool_rendering_audit_tf5.json).

## What was asked, and why arm (1) could not be constructed

The audit was to compare the historical transformers-4.x rendering of the frozen
`tools` value against the 5.x rendering of that same value strictly parsed. **The
historical arm does not exist**, for three independent reasons, each verified:

1. **transformers 4.57.1 cannot load this checkpoint's tokenizer at all.**
   `AutoTokenizer.from_pretrained(artifacts/stage1/qwen3_0p6b_init_v0/checkpoint)`
   raises `AttributeError: 'list' object has no attribute 'keys'` in
   `_set_model_specific_special_tokens`. This is a *second* 4.x incompatibility,
   independent of the known RoPE misreading (STATE §0.5). No rendering of any
   kind can be produced under 4.x for this tokenizer.
2. **No historical evaluation ran under 4.x.** `logs/e6_results.json` records
   `transformers 5.14.1` / `vllm 0.26.0` for every E6 arm. My earlier claim that
   "historical E1–E7 tool evaluations ran under transformers 4.x, which accepted
   it" was an inference from the fact that older runs worked, and it is **wrong**.
3. **This battery's tool prompts have never been rendered by anything.**
   `recovery_search_v1` was built 2026-08-12; its first evaluation attempt was the
   2026-08-13 micro-preflight, which crashed on the first tool prompt. There are
   no historical tool scores from this asset to preserve comparability with.

## What the frozen asset actually contains

`recovery_search_v1/tool.jsonl` is the **only** tool asset in this repository that
stores `tools` as a **JSON string** of xLAM-style objects. Every other one —
`data/stage2{,_v1}/{train,val,calib}/tool_calling.jsonl` — stores a **list** of
OpenAI-style `{"type": "function", "function": {...}}` entries. The battery
builder appears to have copied the upstream xLAM `tools` column verbatim.

## The measured result, all 20 items

| form passed to `apply_chat_template` | renders under transformers 5.13.1 |
| --- | --- |
| the stored value, untouched | **0 / 20** — `ValueError: Tools should either be a JSON schema, or a callable function …` |
| `json.loads(stored)` → list of xLAM dicts | 20 / 20 |
| parsed, then converted to the project's OpenAI-style form | 20 / 20 |
| no tools at all | 20 / 20 |

The same `ValueError` was what stopped Stage 3 on the pod under 5.15.0, so the
failure is not version-specific within 5.x.

**The two renderable forms are not equivalent to each other: 0 / 20 items produce
the same token ids.** On the first item, 272 tokens versus 296, and the `<tools>`
block differs in structure:

    parsed_list    {"name": "live_giveaways_by_type", "description": "...",
                    "parameters": {"type": {"description": "...", "type": "str",
                                            "default": "game"}}}

    openai_schema  {"type": "function", "function": {"name": "...",
                    "description": "...", "parameters": {"type": "object",
                    "properties": {...}, "required": []}}}

## Why this is a decision and not a fix

There is no historical rendering to be exactly equivalent to, so the equivalence
test that would have authorised a silent compatibility adapter **cannot be
satisfied by any option**. What remains is a choice about *what the model is
shown in every tool prompt*, and that choice moves the `tool` capability —
one of the six the catastrophic rule ranks on, and the capability that
`recovery_search_scoring@v2` exists to have measured correctly.

Three options, none of them adopted here:

* **A — parse only.** Minimal, literal to the asset: the model sees xLAM function
  descriptions. Diverges from the tool format the student was *trained* on
  (`data/stage2*` is OpenAI-style), so the battery would test tool use under a
  schema the training data never used.
* **B — parse and convert to the project's OpenAI-style form.** Matches the
  training distribution and every other asset, at the cost of a documented
  transformation between the frozen bytes and the rendered prompt. The conversion
  rule itself has a judgement in it — which parameters are `required` — that
  `audit_tool_scoring.py` already had to make once for scoring.
* **C — rebuild the asset.** Cleanest to read afterwards, but it changes
  `content_sha256 a1b22778…`, which is pinned in the preregistration, the setup
  gate and the scoring contract, and would need those re-emitted.

**Not options:** pinning the evaluator to transformers 4.x (it cannot load the
tokenizer, it misreads RoPE by 500×, and `transformers_version` is itself material
generation-protocol identity), or editing the frozen asset in place to satisfy an
API.

## What this does not touch

Nothing here affects the two permanent Stage-2 controls. Recovery identity and
generation/evaluation identity are separate by construction: the controls are
bound to `RecoveryProtocolFingerprint aad75fee8a897d9c…` and their own weights,
and no tool prompt was involved in producing them. **They must not be retrained.**

Whichever option is chosen, it changes `scripts/evaluation/uncapped_eval.py` and
therefore `generation_source_digest`, so the generation protocol must be
re-attested at Stage 0 before Stage 3 runs. That is a Stage-0/3 re-attestation,
not a Stage-2 invalidation.
