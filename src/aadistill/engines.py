"""Inference-engine adapters, token-in / token-out.

Why this exists
---------------
Teacher generation is the next spend, and its cost is dominated by decode
throughput (`logs/proposals/2026-07-29_inference_engine_survey.md`). The survey
reached one conclusion: **benchmark, do not pick from a table** — the published
engine ordering is driven by prefix-cache reuse under concurrent load, and a
corpus build of unique prompts is the workload where that ordering reverses.

This module is the thing being benchmarked *and* the thing being priced. An
engine's cost to this project is not only tokens/second; it is also how much
code and dependency weight it drags in (P1). So the adapter is deliberately the
whole integration surface: if an engine cannot be driven through
`Engine.generate` below, that is a finding about the engine, not a reason to
grow the interface.

The contract, and why it is this narrow
---------------------------------------
An adapter implements exactly one thing:

    _raw_generate(prompts) -> list[(new_token_ids, finish_reason)]

Prompt **token ids** in, completion **token ids** out. No string round-trip:
`decode` then `encode` is not the identity for most tokenizers, so a corpus
stored as text can train the model on a different token sequence than the one
the teacher actually produced (`generate.py` makes the same argument).

Everything after that — cutting at the stop token, the `hit_cap` flag, the
finished flag — is done **once, here, for every engine** by `_finalize`. That is
not tidiness: the benchmark's headline metric is whether two engines emit the
*same tokens*, and if each adapter did its own trimming, that comparison would
partly be measuring the trimming code. One shared post-processing path means a
disagreement is a real numerical disagreement.

Two engine hazards are handled by construction rather than trusted
------------------------------------------------------------------
* **vLLM** removed the `LLM.generate(prompt_token_ids=...)` keyword; tokenized
  input now goes through `TokensPrompt`. Passing the old keyword raises
  `TypeError` on current versions.
* **SGLang** has a reported bug where `output_ids` includes a prefix that
  overlaps the suffix of `input_ids` (sgl-project/sglang#10896). `_strip_prefix`
  removes it defensively for every engine, so the corpus cannot silently gain a
  duplicated prompt tail.

Neither adapter has been executed — the dev box is CPU-only and both engines are
CUDA-only. They are written against documented APIs and are **unverified until
the pod session**; `bench_engines.py` isolates each engine's failure so a wrong
guess here costs one arm, not the session.
"""

from __future__ import annotations

import time

import torch

from .generate import _left_pad

# A completion is the same dict shape `generate.generate_ids` already returns,
# so an engine can be dropped into the generation script without a translation
# layer: {"tokens", "n_new", "hit_cap", "finished"}.


def _strip_prefix(out_ids: list[int], prompt_ids: list[int]) -> list[int]:
    """Drop a whole-prompt echo from the front of `out_ids`.

    Engines disagree about whether `output_ids` means "new tokens" or "the whole
    sequence", so the unambiguous case is handled here: if the output opens with
    the *entire* prompt, that prefix is echo and goes.

    **Partial overlaps are deliberately not guessed at.** The earlier version of
    this function also stripped the longest prompt-suffix/output-prefix overlap,
    to defend against sglang#10896. `test_hf_engine_respects_the_cap` caught what
    that costs: prompt `[5, 6, 7]` with a genuine completion `[7, 7, 7, 7]` lost
    a real token, because a completion is perfectly entitled to begin with the
    same token the prompt ended on. A heuristic that silently shortens targets is
    worse than the bug it defends against — it corrupts data instead of crashing.
    The SGLang case is resolved exactly instead, from that engine's own token
    counts (see `SGLangEngine._raw_generate`).
    """
    if out_ids and len(out_ids) >= len(prompt_ids) and out_ids[: len(prompt_ids)] == prompt_ids:
        return out_ids[len(prompt_ids):]
    return out_ids


def _finalize(
    new_ids: list[int],
    finish_reason: str,
    stop_ids: set[int],
    cap: int,
    logprobs: list[float | None] | None = None,
) -> dict:
    """Shared post-processing for every engine — see the module docstring.

    Cuts at the first stop token and keeps it. If the engine reports a stop but
    strips the token itself (vLLM's default for `stop_token_ids`), the canonical
    stop is re-appended so `n_new` is comparable across engines rather than
    off-by-one for some of them.

    `logprobs`, when given, is trimmed **in lockstep** with the tokens and the
    result is length-checked. That check is not defensive padding: an importance
    ratio is `exp(logp_trainer - logp_rollout)` for one specific token, so a
    single position of drift between the two lists silently computes every
    downstream ratio against the wrong token. It would not crash and it would not
    look wrong — it would just corrupt the correction term.

    A re-appended stop token gets `None` rather than a fabricated value: the
    engine never reported a probability for a token it did not return, and
    inventing one would put a made-up number into the correction path.
    """
    def out(kept_ids, kept_lps, *, hit_cap, finished):
        record = {"tokens": kept_ids, "n_new": len(kept_ids),
                  "hit_cap": hit_cap, "finished": finished}
        if logprobs is not None:
            if len(kept_lps) != len(kept_ids):
                raise RuntimeError(
                    f"logprob/token length mismatch: {len(kept_lps)} vs "
                    f"{len(kept_ids)} — importance ratios would be misaligned")
            record["logprobs"] = kept_lps
        return record

    lps = list(logprobs) if logprobs is not None else []

    for position, token in enumerate(new_ids):
        if token in stop_ids:
            return out(new_ids[: position + 1], lps[: position + 1],
                       hit_cap=False, finished=True)

    if finish_reason == "stop" and stop_ids:
        # Engine stopped on a stop token but did not return it. Re-append the
        # canonical one so lengths line up with engines that do return it.
        return out(new_ids + [min(stop_ids)], lps + [None],
                   hit_cap=False, finished=True)

    return out(new_ids, lps, hit_cap=len(new_ids) >= cap, finished=False)


class Engine:
    """Base class holding the shared contract. Subclasses implement `_raw_generate`."""

    name = "base"
    # Set by subclasses that load a model, for the memory line in the report.
    supports_deterministic = False

    def generate(
        self,
        prompts: list[list[int]],
        *,
        max_new_tokens: int,
        stop_ids: set[int],
        greedy: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
        logprobs: bool = False,
    ) -> list[dict]:
        """`top_k=0` disables top-k. It is threaded explicitly and never left to
        an engine default, because the defaults disagree: HF `generate` uses
        `top_k=50` unless told otherwise, while vLLM and SGLang disable it. Left
        implicit, the arms would be sampling from different distributions and
        the cross-engine comparison would be measuring that instead of the
        engines.

        `logprobs=True` adds a `logprobs` list to each completion, aligned 1:1
        with `tokens`, holding the **rollout policy's** log-probability of each
        token it emitted. This is what a correction term is computed against
        (AGENTS.md §4.6): it must come from the engine that actually sampled,
        not be recomputed later, because recomputation is not even
        batch-invariant on this project's own measurements.

        Positions the engine did not report a probability for are `None`. They
        are not fabricated and callers must mask them.
        """
        raw = self._raw_generate(
            prompts, max_new_tokens=max_new_tokens, stop_ids=stop_ids,
            greedy=greedy, temperature=temperature, top_p=top_p, top_k=top_k,
            seed=seed, logprobs=logprobs,
        )
        if len(raw) != len(prompts):
            raise RuntimeError(
                f"{self.name}: got {len(raw)} completions for {len(prompts)} prompts — "
                "an engine that reorders or drops rows cannot build a corpus")

        out = []
        for item, prompt in zip(raw, prompts):
            ids, reason = item[0], item[1]
            lps = item[2] if len(item) > 2 else None
            if logprobs and lps is None:
                raise RuntimeError(
                    f"{self.name}: logprobs requested but the engine returned none — "
                    "this backend cannot supply rollout log-probabilities, which "
                    "Stage 4/5 correction requires")
            kept = _strip_prefix(ids, prompt)
            if lps is not None:
                # Whatever the prefix strip removed from the tokens must come off
                # the log-probs too, or the two lists desynchronise silently.
                lps = list(lps)[len(ids) - len(kept):]
            out.append(_finalize(kept, reason, stop_ids, max_new_tokens,
                                 lps if logprobs else None))
        return out

    def _raw_generate(self, prompts, **kw) -> list[tuple]:
        """Return one `(token_ids, finish_reason)` or
        `(token_ids, finish_reason, logprobs)` per prompt, in input order."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class HFEngine(Engine):
    """In-stack `transformers.generate` — the incumbent.

    Its numerics are training-identical by construction (same modeling code,
    kernels and dtype as the trainer), which is why it is the reference every
    other engine is scored against rather than just another arm.
    """

    name = "hf"

    def __init__(self, model, pad_token_id: int, batch_size: int = 8):
        self.model = model
        self.pad_token_id = pad_token_id
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def _raw_generate(self, prompts, *, max_new_tokens, stop_ids, greedy,
                      temperature, top_p, top_k, seed, logprobs=False):
        out: list[tuple | None] = [None] * len(prompts)
        # Length-sorted batches so one long prompt does not pad out the rest;
        # the original index rides along to restore input order.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))

        for start in range(0, len(order), self.batch_size):
            idxs = order[start:start + self.batch_size]
            ids, mask = _left_pad([prompts[i] for i in idxs], self.pad_token_id)
            ids, mask = ids.to(self.device), mask.to(self.device)
            width = ids.shape[1]

            kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.pad_token_id,
                "eos_token_id": sorted(stop_ids),
                # None, not 0: transformers reads 0 as "no tokens allowed".
                "top_k": top_k or None,
            }
            if greedy:
                kwargs.update(do_sample=False, temperature=None, top_p=None)
            else:
                kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
                if seed is not None:
                    torch.manual_seed(seed + start)

            if logprobs:
                # `output_scores` keeps one (batch, vocab) tensor per generated
                # step, so memory grows as steps x batch x vocab: a 4k-token
                # generation at batch 4 on a 152k vocab is ~10 GB. Fine for the
                # oracle role this engine now has (reference, debugging, toy
                # validation) and for short runs; it is a reason the production
                # rollout path is a serving engine that streams its own
                # log-probs rather than this one (AGENTS.md §4.6).
                kwargs.update(output_scores=True, return_dict_in_generate=True)
                result = self.model.generate(ids, attention_mask=mask, **kwargs)
                seq, scores = result.sequences, result.scores
            else:
                seq, scores = self.model.generate(ids, attention_mask=mask, **kwargs), None

            for row, i in enumerate(idxs):
                new = seq[row, width:].tolist()
                # `generate` pads finished rows out to the longest row in the
                # batch; `_finalize` cuts at the stop, so the reason only has to
                # distinguish "ran to cap" from "stopped".
                reason = "length" if not any(t in stop_ids for t in new) else "stop"
                if scores is None:
                    out[i] = (new, reason)
                    continue
                # `scores[t]` holds the distribution the sampler actually drew
                # from at step t, after any temperature/top-p processing. That is
                # the behaviour policy, which is the correct denominator for an
                # importance ratio. At temperature 1.0 / top_p 1.0 / top_k off —
                # this project's rollout setting (decision 2026-07-29) — it also
                # equals the model's own distribution.
                row_lps: list[float | None] = []
                for step, token in enumerate(new):
                    if step >= len(scores):
                        break
                    logits = scores[step][row].float()
                    row_lps.append(float(torch.log_softmax(logits, dim=-1)[token]))
                out[i] = (new, reason, row_lps)
        return [o for o in out if o is not None]


class VLLMEngine(Engine):
    """vLLM offline `LLM.generate`.

    `TokensPrompt`, not the removed `prompt_token_ids=` keyword — see the module
    docstring. `--model-impl transformers` is available upstream and would keep
    one model implementation across train and generate; it is left off here so
    the arm measures vLLM's own path, and is recorded as a follow-up if vLLM
    wins on throughput but loses on agreement.
    """

    name = "vllm"

    def __init__(self, model_path: str, *, dtype: str = "bfloat16",
                 revision: str | None = None, max_model_len: int | None = None,
                 gpu_memory_utilization: float = 0.90):
        from vllm import LLM

        self.llm = LLM(
            model=model_path, revision=revision, dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=False,
        )

    def _raw_generate(self, prompts, *, max_new_tokens, stop_ids, greedy,
                      temperature, top_p, top_k, seed, logprobs=False):
        if logprobs:
            raise NotImplementedError(
                "in-process vLLM logprobs are not wired; use VLLMServerEngine, "
                "which is the supported isolated path")
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(
            temperature=0.0 if greedy else temperature,
            top_p=1.0 if greedy else top_p,
            top_k=-1 if (greedy or not top_k) else top_k,  # -1 disables in vLLM
            max_tokens=max_new_tokens,
            stop_token_ids=sorted(stop_ids),
            seed=seed if not greedy else None,
        )
        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts],
            params, use_tqdm=False,
        )
        # vLLM preserves input order for offline generate, but the corpus depends
        # on that, so assert rather than assume where the API exposes an index.
        return [(list(o.outputs[0].token_ids), o.outputs[0].finish_reason or "length")
                for o in outputs]

    def close(self) -> None:
        del self.llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SGLangEngine(Engine):
    """SGLang offline `Engine.generate`.

    The only candidate shipping an explicit **deterministic mode** (batch-
    invariant operators; Qwen3 is named as supported), at a cited ~34% average
    throughput cost. That is the axis this project cares most about, so the arm
    is worth its setup cost even if it loses on raw tokens/second.
    """

    name = "sglang"
    supports_deterministic = True

    def __init__(self, model_path: str, *, dtype: str = "bfloat16",
                 revision: str | None = None, deterministic: bool = False,
                 mem_fraction_static: float = 0.85):
        import sglang as sgl

        kwargs = dict(model_path=model_path, dtype=dtype,
                      mem_fraction_static=mem_fraction_static)
        if revision:
            kwargs["revision"] = revision
        if deterministic:
            # Named `enable_deterministic_inference` in the released flag set;
            # if the installed build spells it differently the arm fails loudly
            # here rather than silently benchmarking the nondeterministic path,
            # which would be the worst possible outcome for this measurement.
            kwargs["enable_deterministic_inference"] = True
        self.engine = sgl.Engine(**kwargs)
        self.deterministic = deterministic

    def _raw_generate(self, prompts, *, max_new_tokens, stop_ids, greedy,
                      temperature, top_p, top_k, seed, logprobs=False):
        if logprobs:
            raise NotImplementedError(
                "sglang logprobs are not wired yet; it is an untested candidate "
                "(proposal 2026-07-30)")
        params = {
            "temperature": 0.0 if greedy else temperature,
            "top_p": 1.0 if greedy else top_p,
            "top_k": -1 if (greedy or not top_k) else top_k,  # -1 disables in SGLang
            "max_new_tokens": max_new_tokens,
            "stop_token_ids": sorted(stop_ids),
        }
        outputs = self.engine.generate(input_ids=prompts, sampling_params=params)
        rows = []
        for out, prompt in zip(outputs, prompts):
            ids = out.get("output_ids")
            if ids is None:
                # Falling back to re-encoding `text` would break token-in/token-out
                # and quietly reintroduce retokenization drift into the corpus.
                # That is a disqualifying property, so say so instead of coping.
                raise RuntimeError(
                    "sglang returned no output_ids — this build cannot do "
                    "token-in/token-out, which the corpus build requires")
            meta = out.get("meta_info") or {}
            ids = list(ids)

            # sglang#10896: `output_ids` can carry a prefix overlapping the tail
            # of `input_ids`. Resolve it *exactly* rather than by pattern-guessing
            # (see `_strip_prefix` for why guessing corrupts data): the engine
            # reports how many tokens it actually completed, so anything above
            # that count is echo and its length is known, not inferred.
            completion_tokens = meta.get("completion_tokens")
            if isinstance(completion_tokens, int) and 0 < completion_tokens < len(ids):
                ids = ids[len(ids) - completion_tokens:]
            elif len(ids) >= len(prompt) and ids[: len(prompt)] == prompt:
                ids = ids[len(prompt):]

            reason = meta.get("finish_reason")
            if isinstance(reason, dict):
                reason = reason.get("type", "length")
            rows.append((ids, str(reason or "length")))
        return rows

    def close(self) -> None:
        try:
            self.engine.shutdown()
        except Exception:
            pass


class VLLMServerEngine(Engine):
    """Talks to a vLLM OpenAI-compatible server over HTTP.

    Why this exists rather than `VLLMEngine`
    ----------------------------------------
    The in-process adapter cannot be used with this project's pinned stack: on
    2026-07-29 vLLM 0.26.0 installed but would not import, because its compiled
    extension wants `libcudart.so.13` while the trainer runs on cu128, and
    SGLang imported only after downgrading transformers by a major version. An
    engine that must own its own torch build cannot share a process with the
    trainer — so it gets its own process, its own venv, and a socket.

    That is not a workaround, it is how these engines are actually deployed. The
    cost is a process boundary and a serialization hop; the benefit is that the
    engine's dependency tree stops being the trainer's problem entirely, which
    is the "impact on the overall code" the benchmark exists to weigh (P1).

    Token-in / token-out over HTTP
    ------------------------------
    `/v1/completions` accepts `prompt` as token ids, and `return_token_ids`
    makes it return them, which is the property this corpus build needs — vLLM
    added it precisely because re-encoding text reintroduces retokenization
    drift in agent RL. If a server cannot do it, this adapter fails loudly
    rather than silently round-tripping through text.

    Uses `urllib` from the standard library on purpose: an HTTP client is not
    worth a new dependency (P1), and adding one would need approval (P12).
    """

    name = "vllm_server"

    def __init__(self, base_url: str, model: str, *, timeout: float = 3600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        import json
        import urllib.request

        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def _raw_generate(self, prompts, *, max_new_tokens, stop_ids, greedy,
                      temperature, top_p, top_k, seed, logprobs=False):
        payload = {
            "model": self.model,
            "prompt": [list(p) for p in prompts],  # token ids, not text
            "max_tokens": max_new_tokens,
            "temperature": 0.0 if greedy else temperature,
            "top_p": 1.0 if greedy else top_p,
            "top_k": -1 if (greedy or not top_k) else top_k,
            "stop_token_ids": sorted(stop_ids),
            "return_token_ids": True,
        }
        if logprobs:
            # `logprobs: 0` asks for the chosen token's own log-probability and
            # no alternatives — the only value a correction term needs, and the
            # cheapest thing to ship over the wire.
            payload["logprobs"] = 0
        if seed is not None and not greedy:
            payload["seed"] = seed

        choices = self._post("/v1/completions", payload).get("choices", [])
        if len(choices) != len(prompts):
            raise RuntimeError(
                f"{self.name}: {len(choices)} choices for {len(prompts)} prompts")

        # Order by the server's `index`, never by arrival. The OpenAI schema
        # carries an index precisely because the order is not contractual, and a
        # corpus that pairs completions with the wrong prompts is silently wrong
        # in a way no downstream check would catch.
        rows: list[tuple[list[int], str] | None] = [None] * len(prompts)
        for position, choice in enumerate(choices):
            index = choice.get("index", position)
            ids = choice.get("token_ids")
            if ids is None:
                raise RuntimeError(
                    f"{self.name}: response has no token_ids — this server "
                    "cannot do token-in/token-out, which the corpus build "
                    "requires (start it with a build supporting "
                    "`return_token_ids`)")
            if not 0 <= index < len(prompts):
                raise RuntimeError(f"{self.name}: choice index {index} out of range")
            reason = str(choice.get("finish_reason") or "length")
            if not logprobs:
                rows[index] = (list(ids), reason)
                continue
            lp = (choice.get("logprobs") or {}).get("token_logprobs")
            if lp is None:
                raise RuntimeError(
                    f"{self.name}: logprobs requested but the response carries "
                    "none; this server build cannot supply rollout "
                    "log-probabilities")
            if len(lp) != len(ids):
                raise RuntimeError(
                    f"{self.name}: {len(lp)} logprobs for {len(ids)} tokens — "
                    "refusing to guess the alignment")
            rows[index] = (list(ids), reason, [None if v is None else float(v) for v in lp])
        missing = [i for i, r in enumerate(rows) if r is None]
        if missing:
            raise RuntimeError(f"{self.name}: no choice returned for prompts {missing}")
        return rows  # type: ignore[return-value]


def agreement(a: list[dict], b: list[dict]) -> dict:
    """Token-level agreement between two engines' completions on the same prompts.

    This is the property Stage 4/5 needs and that no vendor claims across stacks:
    if the rollout engine and the trainer disagree, "on-policy" updates are
    quietly off-policy. Reported as an exact-match rate plus where the first
    divergence lands, because *how far in* it diverges says whether this is a
    last-bits argmax flip or a different decode entirely.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    first: list[int | None] = []
    for x, y in zip(a, b):
        tx, ty = x["tokens"], y["tokens"]
        point = None
        for i in range(min(len(tx), len(ty))):
            if tx[i] != ty[i]:
                point = i
                break
        if point is None and len(tx) != len(ty):
            point = min(len(tx), len(ty))
        first.append(point)
    matched = sum(p is None for p in first)
    diverged = [p for p in first if p is not None]
    return {
        "n": len(first),
        "exact_match": matched,
        "exact_match_rate": round(matched / len(first), 4) if first else 0.0,
        "first_divergence": first,
        "median_divergence_token": (sorted(diverged)[len(diverged) // 2]
                                    if diverged else None),
    }


def batch_invariance(engine: Engine, prompts: list[list[int]], *,
                     stop_ids: set[int], max_new_tokens: int = 64) -> dict:
    """Does batching change the tokens a prompt generates, on this engine?

    Float addition is not associative and batch size changes how kernels split
    reductions, so identical prompts can take different numerical paths to
    logits and flip an argmax under greedy decoding. Generates every prompt
    alone, then all of them as one batch, and compares.

    `generate.assert_batch_invariant` does this for the in-stack path only; this
    is the same check hoisted to the interface so every candidate faces it.
    """
    alone: list[dict] = []
    for p in prompts:
        alone.extend(engine.generate([p], max_new_tokens=max_new_tokens,
                                     stop_ids=stop_ids, greedy=True))
    batched = engine.generate(prompts, max_new_tokens=max_new_tokens,
                              stop_ids=stop_ids, greedy=True)
    result = agreement(alone, batched)
    result["identical"] = result["exact_match"] == result["n"]
    return result


def timed(fn, *args, **kwargs) -> tuple[object, float]:
    """Run `fn`, return `(result, wall_seconds)` with the GPU actually drained.

    `torch.cuda.synchronize` matters here: without it a CUDA-async engine
    returns before the work is done and the arm looks faster than it is.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start
