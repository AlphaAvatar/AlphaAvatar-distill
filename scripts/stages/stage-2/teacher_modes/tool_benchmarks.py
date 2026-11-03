#!/usr/bin/env python3
"""The public tool-calling benchmarks `tool_pool@v1` is decontaminated against: pinned copies, prompts and tools. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_benchmarks fetch

BFCL v3, When2Call (test), API-Bank (test), tau-bench and ToolSandbox, each pinned by revision and fetched once
into ``artifacts/stages/stage-2/teacher_modes/tool_sources/benchmarks/<name>@<rev8>/`` (gitignored; only these
small test files, P8.4). Nothing here is training data: the pool reads `prompts` (each item's user text) for the
13-gram screen and `tools` (name and parameter schema) for the tool-identity screen. The downloaded files are
untrusted data: Python sources are parsed with `ast`, never imported or run.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ROOT = REPO / "artifacts/stages/stage-2/teacher_modes/tool_sources/benchmarks"

#: name -> (host, repository, revision, the files read: an HF path glob or a path prefix inside a GitHub tarball)
BENCHMARKS = {
    "bfcl_v3": ("hf", "gorilla-llm/Berkeley-Function-Calling-Leaderboard", "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
                ("BFCL_v3_*.json", "multi_turn_func_doc/*.json")),
    "when2call": ("hf", "nvidia/When2Call", "0582f7749df63a96fdc3070932e83e72396ace53", ("test/*.jsonl",)),
    "api_bank": ("hf", "liminghao1630/API-Bank", "12e8158b7628c168f07e8f31fbbe3445e99f44cf", ("test-data/*.json",)),
    "tau_bench": ("github", "sierra-research/tau-bench", "59a200c6d575d595120f1cb70fea53cef0632f6b",
                  ("tau_bench/envs/",)),
    "toolsandbox": ("github", "apple-aiml-research/ToolSandbox", "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1",
                    ("tool_sandbox/scenarios/", "tool_sandbox/tools/")),
}


def path_of(name: str, root: Path = ROOT) -> Path:
    return Path(root) / f"{name}@{BENCHMARKS[name][2][:8]}"


def fetch(name: str, root: Path = ROOT) -> dict:
    """Download one benchmark's files at its pinned revision (idempotent); returns its manifest."""
    host, repo, rev, files = BENCHMARKS[name]
    out = path_of(name, root)
    man = out / "manifest.json"
    if man.is_file():
        return json.loads(man.read_text())
    out.mkdir(parents=True, exist_ok=True)
    got = {}
    if host == "hf":
        from huggingface_hub import snapshot_download
        snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=list(files), local_dir=out)
        for p in sorted(out.rglob("*")):
            if p.is_file() and ".cache" not in p.parts and p.name != "manifest.json":
                got[str(p.relative_to(out))] = hashlib.sha256(p.read_bytes()).hexdigest()
    else:
        url = f"https://codeload.github.com/{repo}/tar.gz/{rev}"
        with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 (a pinned public tarball)
            blob = r.read()
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            for m in tar.getmembers():
                rel = m.name.split("/", 1)[1] if "/" in m.name else ""
                if not m.isfile() or not rel.endswith((".py", ".json", ".jsonl")) or not rel.startswith(files):
                    continue
                data = tar.extractfile(m).read()
                dest = out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                got[rel] = hashlib.sha256(data).hexdigest()
        got["_tarball_sha256"] = hashlib.sha256(blob).hexdigest()
    rec = {"benchmark": name, "host": host, "repository": repo, "revision": rev, "files": got}
    man.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def _jsonl(p: Path) -> Iterator[dict]:
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _json_rows(p: Path) -> Iterator[dict]:
    """A JSON array, an object of rows, or JSON lines."""
    text = p.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except ValueError:
        yield from _jsonl(p)
        return
    rows = obj if isinstance(obj, list) else list(obj.values()) if isinstance(obj, dict) else []
    for r in rows:
        if isinstance(r, dict):
            yield r


def _user_text(messages) -> str:
    """The user turns of a message list (or of a list of turns, BFCL's multi-turn shape)."""
    out = []
    for m in messages or []:
        if isinstance(m, list):
            out.append(_user_text(m))
        elif isinstance(m, dict) and m.get("role") == "user":
            out.append(str(m.get("content") or ""))
    return "\n".join(x for x in out if x)


def _py_strings(p: Path, keywords: tuple[str, ...]) -> Iterator[str]:
    """String constants passed as one of ``keywords`` in any call of a Python source (parsed, never run)."""
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in keywords and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    yield kw.value.value


def _py_dict_strings(p: Path, key: str) -> Iterator[str]:
    """String constants stored under ``key`` in any dict literal of a Python source (parsed, never run)."""
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and k.value == key and isinstance(v, ast.Constant)
                        and isinstance(v.value, str)):
                    yield v.value


def _py_tool_dicts(p: Path) -> Iterator[dict]:
    """tau-bench's tool schemas: the literal dicts its ``get_info`` methods return (parsed, never run)."""
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_info":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and sub.value is not None:
                    try:
                        v = ast.literal_eval(sub.value)
                    except ValueError:
                        continue
                    if isinstance(v, dict):
                        yield v


def prompts(name: str, root: Path = ROOT) -> list[tuple[str, str]]:
    """``(item id, user text)`` of every test item."""
    d = path_of(name, root)
    out = []
    if name == "bfcl_v3":
        for p in sorted(d.glob("BFCL_v3_*.json")):
            for r in _jsonl(p):
                out.append((f"{p.stem}:{r.get('id')}", _user_text(r.get("question"))))
    elif name == "when2call":
        for p in sorted((d / "test").glob("*.jsonl")):
            for k, r in enumerate(_jsonl(p)):
                q = r.get("question") or _user_text(r.get("messages"))
                out.append((f"{p.stem}:{r.get('uuid', k)}", str(q or "")))
    elif name == "api_bank":
        for p in sorted((d / "test-data").glob("*.json")):
            for k, r in enumerate(_json_rows(p)):
                #: levels 1-2: the dialogue's user lines; level 3: the requirement
                text = "\n".join(x[len("User:"):].strip() for x in str(r.get("input") or "").splitlines()
                                 if x.startswith("User:")) or str(r.get("requirement") or "")
                out.append((f"{p.stem}:{k}", text))
    elif name == "tau_bench":
        for p in sorted(d.rglob("tasks*.py")):
            for k, s in enumerate(_py_strings(p, ("instruction",))):
                out.append((f"{p.relative_to(d)}:{k}", s))
    elif name == "toolsandbox":
        for p in sorted((d / "tool_sandbox/scenarios").rglob("*.py")):
            for k, s in enumerate(list(_py_strings(p, ("content",))) + list(_py_dict_strings(p, "content"))):
                out.append((f"{p.relative_to(d)}:{k}", s))
    return [(i, t) for i, t in out if t and t.strip()]


def tools(name: str, root: Path = ROOT) -> list[dict]:
    """Every tool the benchmark offers, as ``{"name", "parameters"}`` (ToolSandbox's Python tools: name only)."""
    d = path_of(name, root)
    out = []

    def add(t):
        f = t.get("function", t) if isinstance(t, dict) else None
        if isinstance(f, dict) and isinstance(f.get("name"), str):
            out.append({"name": f["name"], "parameters": f.get("parameters")})
    if name == "bfcl_v3":
        for p in sorted(d.glob("BFCL_v3_*.json")):
            for r in _jsonl(p):
                for t in r.get("function") or []:
                    add(t)
        for p in sorted((d / "multi_turn_func_doc").glob("*.json")):
            for r in _jsonl(p):
                add(r)
    elif name == "when2call":
        for p in sorted((d / "test").glob("*.jsonl")):
            for r in _jsonl(p):
                for t in r.get("tools") or []:
                    add(json.loads(t) if isinstance(t, str) else t)
    elif name == "api_bank":
        #: levels 1-2 carry their API descriptions inside the instruction, one JSON object per line
        for p in sorted((d / "test-data").glob("level-[12]-api.json")):
            for r in _json_rows(p):
                for line in str(r.get("instruction") or "").splitlines():
                    if line.startswith('{"name"'):
                        try:
                            api = json.loads(line)
                        except ValueError:
                            continue
                        out.append({"name": api["name"], "parameters": api.get("input_parameters")})
    elif name == "tau_bench":
        for p in sorted(d.rglob("tools/*.py")):
            for t in _py_tool_dicts(p):
                add(t)
    elif name == "toolsandbox":
        for p in sorted((d / "tool_sandbox/tools").rglob("*.py")):
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            out += [{"name": n.name, "parameters": None} for n in tree.body
                    if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ["fetch"]:
        print(__doc__)
        return 2
    for name in (argv[1:] or sorted(BENCHMARKS)):
        rec = fetch(name)
        print(name, rec["revision"][:8], len(rec["files"]), "files,", len(prompts(name)), "prompts,",
              len(tools(name)), "tools")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
