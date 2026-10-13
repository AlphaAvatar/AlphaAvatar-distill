#!/usr/bin/env python3
"""Run `uncapped_eval.py` UNCHANGED and keep the engine's exact token ids.

    /opt/vllm/bin/python scripts/shared/evaluation/uncapped_eval_token_ids.py \
        --expect-generation-source-digest <digest> <uncapped_eval arguments...>

The generator decodes each rollout once, from the engine's final token ids, and
records only the text. The ids are what a per-position rescoring of the
student's own trajectory needs, and re-tokenizing the text is not unique.
Recording them inside the generator would change a file of the generation
source set, hence the generation fingerprint and the shared evaluation
protocol's identity. This wrapper keeps them WITHOUT touching that file:

* it imports the in-tree generator and runs its `main()` with the same argv,
  the same interpreter and the same environment -- every summary field, and so
  the reconstructed generation fingerprint, is the plain run's;
* the generator's tokenizer gets a read-only `decode` hook. It fires at the
  one post-wave `tok.decode(gen)` per record, after the engine loop has
  finished, copies the ids it was handed, and returns the original text. It
  never touches sampling, request order, `step` or `abort`, and it never
  raises into the generator;
* the hook is installed only when the generator's source digest equals the
  one the caller expects, because it reads the generator's local names, and
  those are pinned by exactly that digest. Otherwise the run is plain and the
  sidecar says why it is absent.

Output: `<stem>.token_ids.jsonl` beside each `<stem>.generations.jsonl` -- a
header line, then one row per record. Never `*.json`: admission reads every
`*.json` in the directory as a summary. Each row is verified against its
record before the header calls it exact: the id count equals
`generated_tokens`, the prompt count equals `prompt_tokens`, the decoded
text's sha256 equals the record's `raw`, and EOS agrees with `stop_reason`.

A DIAGNOSTIC sidecar: nothing admits, scores or gates on it, and the wrapper
never changes the generator's exit status.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "aadistill.engine_token_ids/v1"
SIDECAR_SUFFIX = ".token_ids.jsonl"
FLAG = "--expect-generation-source-digest"
GENERATOR = Path(__file__).resolve().with_name("uncapped_eval.py")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _split_argv(argv: list[str]) -> tuple[str | None, list[str]]:
    """Pop this wrapper's own flag; forward everything else verbatim."""
    expected, rest, i = None, [], 0
    while i < len(argv):
        if argv[i] == FLAG and i + 1 < len(argv):
            expected, i = argv[i + 1], i + 2
        elif argv[i].startswith(FLAG + "="):
            expected, i = argv[i].split("=", 1)[1], i + 1
        else:
            rest.append(argv[i])
            i += 1
    return expected, rest


def _out_paths(argv: list[str]) -> list[tuple[str, Path]]:
    """(stem, summary path) per prompt file, by the generator's own rule."""
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--prompts", nargs="+", default=[])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out", default=None)
    a, _ = ap.parse_known_args(argv)
    return [(Path(p).stem,
             Path(a.out) if a.out else Path(a.out_dir or ".") / f"{Path(p).stem}.json")
            for p in a.prompts]


class _Hook:
    """Collects (stem, record id) -> ids at the generator's decode site."""

    def __init__(self, main_code) -> None:
        self.main_code = main_code
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.calls = 0
        self.errors: list[str] = []
        self.installed = False

    def wrap_tokenizer(self, tok) -> None:
        real = tok.decode

        def decode(ids, *args, **kwargs):
            text = real(ids, *args, **kwargs)
            try:
                frame = sys._getframe(1)
                local = frame.f_locals
                if frame.f_code is self.main_code and local.get("gen") is ids:
                    self.calls += 1
                    stem = Path(local["prompts_path"]).stem
                    key = (stem, str(local["s_"]["id"]))
                    if key in self.rows:
                        self.errors.append(f"{key}: decoded twice")
                    self.rows[key] = {
                        "id": key[1], "request_id": local["rid"],
                        "prompt_token_ids": [int(t) for t in local["st"]["p_ids"]],
                        "output_token_ids": [int(t) for t in ids],
                        "raw_sha256": _sha(text)}
            except Exception as exc:               # noqa: BLE001 -- never raise
                self.errors.append(f"{type(exc).__name__}: {exc}")
            return text

        tok.decode = decode
        self.installed = True


def _install(hook: _Hook, auto_tokenizer, generator_main) -> None:
    """One-shot: hook the tokenizer the generator loads, then restore.

    The generator's first `AutoTokenizer.from_pretrained` call is its own, and
    it precedes the engine; restoring there means the engine's tokenizer and
    any engine process are never touched. The class attribute is restored from
    the class's own `__dict__`, so no shadowing attribute is left behind.
    """
    had = "from_pretrained" in auto_tokenizer.__dict__
    original = auto_tokenizer.__dict__.get("from_pretrained")
    real = auto_tokenizer.from_pretrained

    def restore() -> None:
        if had:
            setattr(auto_tokenizer, "from_pretrained", original)
        elif "from_pretrained" in auto_tokenizer.__dict__:
            delattr(auto_tokenizer, "from_pretrained")

    def once(*args, **kwargs):
        restore()
        tok = real(*args, **kwargs)
        if sys._getframe(1).f_code is generator_main.__code__:
            hook.wrap_tokenizer(tok)
        else:
            hook.errors.append("first tokenizer load was not the generator's")
        return tok

    auto_tokenizer.from_pretrained = once
    hook.restore = restore                       # type: ignore[attr-defined]


def verify_rows(records: list[dict], rows: dict[str, dict],
                stop_ids: list[int]) -> dict[str, Any]:
    """Per record: are these the ids the record's text was decoded from?"""
    problems = []
    for rec in records:
        row = rows.get(str(rec["id"]))
        if row is None:
            problems.append(f"{rec['id']}: no ids")
            continue
        out, prompt = row["output_token_ids"], row["prompt_token_ids"]
        if len(out) != int(rec["generated_tokens"]):
            problems.append(f"{rec['id']}: {len(out)} ids vs "
                            f"{rec['generated_tokens']} generated")
        if len(prompt) != int(rec["prompt_tokens"]):
            problems.append(f"{rec['id']}: prompt {len(prompt)} vs "
                            f"{rec['prompt_tokens']}")
        if row["raw_sha256"] != _sha(rec["raw"]):
            problems.append(f"{rec['id']}: decoded text differs from the record")
        if (bool(out) and out[-1] in stop_ids) != (rec["stop_reason"] == "eos"):
            problems.append(f"{rec['id']}: EOS disagrees with stop_reason")
    extra = sorted(set(rows) - {str(r["id"]) for r in records})
    problems += [f"{i}: ids without a record" for i in extra]
    return {"records": len(records), "rows": len(rows),
            "exact": not problems, "problems": problems[:50],
            "n_problems": len(problems)}


def write_sidecars(hook: _Hook | None, outputs: list[tuple[str, Path]], *,
                   expected: str | None, observed: str | None,
                   reason: str | None) -> None:
    """One sidecar per stratum whose generations exist. Never raises."""
    wrapper_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for stem, summary_path in outputs:
        try:
            gens = summary_path.with_suffix(".generations.jsonl")
            if not gens.is_file():
                continue
            records = [json.loads(l) for l in gens.read_text().splitlines()
                       if l.strip()]
            summary = (json.loads(summary_path.read_text())
                       if summary_path.is_file() else {})
            rows = ({i: r for (s, i), r in hook.rows.items() if s == stem}
                    if hook and hook.installed else {})
            check = (verify_rows(records, rows, list(summary.get("stop_ids") or []))
                     if rows else None)
            header = {
                "schema": SCHEMA, "diagnostic_only": True,
                "available": bool(rows),
                "reason_unavailable": None if rows else (reason or "hook captured nothing"),
                "exact": bool(check and check["exact"]),
                "wrapper_sha256": wrapper_sha,
                "generator": "scripts/shared/evaluation/uncapped_eval.py",
                "generator_sha256": hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
                "generation_source_digest": observed,
                "expected_generation_source_digest": expected,
                "generations_file": gens.name,
                "generations_sha256": hashlib.sha256(gens.read_bytes()).hexdigest(),
                "verification": check,
                "hook_errors": (hook.errors[:20] if hook else []),
            }
            path = summary_path.with_name(f"{stem}{SIDECAR_SUFFIX}")
            with open(path, "w") as f:
                f.write(json.dumps(header) + "\n")
                for rec in records:
                    row = rows.get(str(rec["id"]))
                    if row is not None:
                        f.write(json.dumps(row) + "\n")
        except Exception as exc:                   # noqa: BLE001
            print(f"[token-ids] sidecar for {stem} not written: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    expected, forwarded = _split_argv(list(sys.argv[1:] if argv is None else argv))
    sys.argv = [str(GENERATOR), *forwarded]
    sys.path.insert(0, str(GENERATOR.parent))
    import uncapped_eval as UE                     # noqa: E402

    outputs = _out_paths(forwarded)
    observed = UE.generation_source_digest(UE.REPO_ROOT)["digest"]
    hook, reason = None, None
    if expected is None:
        reason = f"no {FLAG} given"
    elif observed != expected:
        reason = (f"generator source digest {observed} is not the expected "
                  f"{expected}; ran unhooked")
    else:
        import transformers
        hook = _Hook(UE.main.__code__)
        _install(hook, transformers.AutoTokenizer, UE.main)
    if reason:
        print(f"[token-ids] {reason}", file=sys.stderr, flush=True)
    try:
        return UE.main()
    finally:
        if hook is not None:
            hook.restore()                         # type: ignore[attr-defined]
        write_sidecars(hook, outputs, expected=expected, observed=observed,
                       reason=reason)


def load_sidecar(summary_path: Path) -> tuple[dict, dict[str, dict]] | None:
    """(header, id -> row) for a stratum, or None when there is no sidecar."""
    path = Path(summary_path).with_name(
        f"{Path(summary_path).stem}{SIDECAR_SUFFIX}")
    if not path.is_file():
        return None
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return lines[0], {str(r["id"]): r for r in lines[1:]}


if __name__ == "__main__":
    raise SystemExit(main())
