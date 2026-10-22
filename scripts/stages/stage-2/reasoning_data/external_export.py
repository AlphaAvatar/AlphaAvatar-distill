"""Export WHITELISTED third-party generator modules to JSONL, inside an isolated env.

    PYTHONHASHSEED=0 <isolated venv>/bin/python -I external_export.py --whitelist <modules.json> \
        --n <per module and level> --seed-base <int> --out <export.jsonl> [--prontoqa-repo <checkout>]

Runs in its own pinned virtual environment (the third-party packages and
their dependencies -- including data they bundle, such as arckit's ARC-AGI
copies -- live there and nowhere else), with ``-I``, and imports nothing from
this repository. ``--whitelist`` is the only input: a JSON list of
``{"source", "module", "levels"}`` written by `external_tasks.py` from the
candidate modules of `external_modules.json`; a module not on it is never
imported or instantiated. The output is prompts, gold answers and the
generator's own reference trace when it has one -- the trace is exported for
the STEP check only and must never reach a prompt.

Instances are generated once and stored (`external_modules.json ::
verification_policy`): every entry runs in a FRESH process (several generators
depend on process history and global RNG state), the hash seed is pinned and
recorded per row, and each entry's seed derives from its own (source, module)
key and level -- never from its position in the whitelist -- so adding a
module changes no other module's instances. ``-I`` implies ``-E``, which
IGNORES ``PYTHONHASHSEED``; so each child runs with ``-s -P`` (no user site, no
script directory on the path) and an environment this process builds -- every
``PYTHON*`` variable removed, then ``PYTHONHASHSEED`` set -- and refuses to run
unless the interpreter reads its environment and the seed is the one asked
for.

The training build reads only this file (by its sha256), never the library.
Supported sources and what a level means:

* reasoning_gym, reasoning_core -- the library's difficulty level;
* prontoqa -- the hop count, by the repository's own CLI
  (``run_experiment.py --model-name json``) run from a pinned checkout
  (``--prontoqa-repo``) in a scratch working directory; the zero-shot prompt is
  the test example's question and query, and its chain of thought is exported
  as the reference (``reference_chain``); the in-context examples the CLI also
  draws are discarded;
* mathematics_dataset -- the train entropy level (1 train-medium, 2
  train-hard), sampled as its ``generate.sample_from_module`` does (questions
  over 160 or answers over 30 characters are redrawn). It needs numpy < 2, so it
  has its own venv and its own export file.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from importlib.metadata import distribution, version

SUPPORTED = ("reasoning_gym", "reasoning_core", "prontoqa", "mathematics_dataset")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    return x if isinstance(x, (str, int, float, bool)) or x is None else repr(x)


def _gold_in_prompt(gold, prompt: str) -> bool:
    g = str(gold).strip()
    return bool(g) and re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", prompt) is not None


def entry_seed(base: int, source: str, module: str, level: int) -> int:
    """Stable per (source, module, level); 100,000 instance seeds per level."""
    key = int(hashlib.sha256(f"{source}/{module}".encode()).hexdigest()[:8], 16) % 1000
    return base + 1_000_000 * key + 100_000 * level


#: A library generator may not return: reasoning_gym's bf at difficulty 3 hangs
#: on some seeds (a program that does not halt). One item gets this long; an
#: item that does not return is skipped and reported, never retried.
ITEM_SECONDS = 60


class _ItemTimeout(Exception):
    pass


def _bounded(fn, *a):
    def _alarm(signum, frame):
        raise _ItemTimeout()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(ITEM_SECONDS)
    try:
        return fn(*a)
    except _ItemTimeout:
        return None
    finally:
        signal.alarm(0)


def rg_rows(module: str, level: int, n: int, seed: int, _args):
    from reasoning_gym.coaching.base_curriculum import DefaultCurriculumContext, RangeAttributeMode
    from reasoning_gym.factory import CURRICULA, DATASETS, create_curriculum
    ds_cls, cfg_cls = DATASETS[module]
    if module in CURRICULA:
        cur = create_curriculum(module)
        top = max(len(a.levels) - 1 for a in cur.attributes.values())
        cur.set_global_level(min(level, top))
        cfg = cur.generate_configuration(defaults={"seed": seed, "size": n},
                                         context=DefaultCurriculumContext(mode=RangeAttributeMode.UPPER_BOUND))
    else:
        cfg = cfg_cls(seed=seed, size=n)
    ds = ds_cls(config=cfg)
    for i in range(n):
        e = _bounded(ds.__getitem__, i)
        if e is None:
            yield i, None, None, None, {"timed_out": ITEM_SECONDS}
            continue
        gold = e["answer"] if e["answer"] is not None else (e.get("metadata") or {}).get("example_answer")
        yield i, e["question"], gold, None, {}


def rc_rows(module: str, level: int, n: int, seed: int, _args):
    import numpy as np
    import reasoning_core as rc
    for i in range(n):
        random.seed(seed + i)
        np.random.seed((seed + i) % (2 ** 32))
        ex = _bounded(lambda: rc.get_task(module).generate_example(level=level))
        if ex is None:
            yield i, None, None, None, {"timed_out": ITEM_SECONDS}
            continue
        cot = dict(ex.metadata).get("cot")
        yield i, ex.prompt, ex.answer, cot if isinstance(cot, str) else None, {}


def pq_rows(module: str, level: int, n: int, seed: int, args):
    import runpy
    repo = os.path.abspath(args.prontoqa_repo)
    work = tempfile.mkdtemp(prefix="prontoqa_")
    try:
        shutil.copy(os.path.join(repo, "bad_patterns.txt"), work)
        sys.path.insert(0, repo)                  # run_experiment.py imports its siblings
        cwd, argv = os.getcwd(), sys.argv
        os.chdir(work)
        sys.argv = ["run_experiment.py", "--model-name", "json", "--model-size", "none",
                    "--num-trials", str(n), "--min-hops", str(level), "--max-hops", str(level),
                    "--seed", str(seed)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(os.path.join(repo, "run_experiment.py"), run_name="__main__")
        finally:
            os.chdir(cwd)
            sys.argv = argv
        files = [f for f in os.listdir(work) if f.endswith(".json")]
        if len(files) != 1:
            raise RuntimeError(f"prontoqa wrote {files}")
        with open(os.path.join(work, files[0])) as f:
            data = json.load(f)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    for i, (_, ex) in enumerate(sorted(data.items(), key=lambda kv: int(kv[0][len("example"):]))):
        test = ex["test_example"]
        chain = list(test["chain_of_thought"])
        yield i, test["question"] + " " + test["query"], test["answer"], " ".join(chain), \
            {"reference_chain": chain}


def _md_entropy(level: int, num_levels: int = 3):
    """`mathematics_dataset.generate._make_entropy_fn`, without its absl flags."""
    lower, upper = level / num_levels, (level + 1) / num_levels

    def modify(range_):
        length = range_[1] - range_[0]
        return (range_[0] + lower * length, range_[0] + upper * length)
    return modify


def md_rows(module: str, level: int, n: int, seed: int, _args):
    import numpy as np
    from mathematics_dataset import generate_settings
    from mathematics_dataset.modules import modules
    group, name = module.split("__", 1)
    fn = modules.train(_md_entropy(level))[group][name]
    for i in range(n):
        random.seed(seed + i)
        np.random.seed(seed + i)
        redrawn = 0
        while True:
            p = fn()
            q, a = str(p.question), str(p.answer)
            if len(q) <= generate_settings.MAX_QUESTION_LENGTH and len(a) <= generate_settings.MAX_ANSWER_LENGTH:
                break
            redrawn += 1
        yield i, q, a, None, {"answer_class": type(p.answer).__name__, "redrawn_overlong": redrawn}


ROWS = {"reasoning_gym": rg_rows, "reasoning_core": rc_rows, "prontoqa": pq_rows, "mathematics_dataset": md_rows}


def source_version(source: str, args) -> str:
    if source == "prontoqa":
        sha = subprocess.run(["git", "-C", args.prontoqa_repo, "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        return f"git:{sha}"
    dist = {"reasoning_gym": "reasoning-gym", "reasoning_core": "reasoning-core",
            "mathematics_dataset": "mathematics-dataset"}[source]
    v = version(dist)
    url = distribution(dist).read_text("direct_url.json")
    commit = (json.loads(url).get("vcs_info") or {}).get("commit_id") if url else None
    return f"{v}+git:{commit}" if commit else v


def export_entry(e: dict, args) -> tuple[list[dict], int]:
    """One whitelist entry, every level: (rows, levels that failed)."""
    ver = source_version(e["source"], args)
    rows, errors = [], 0
    for level in e["levels"]:
        seed = entry_seed(args.seed_base, e["source"], e["module"], level)
        try:
            for i, prompt, gold, trace, extra in ROWS[e["source"]](e["module"], level, args.n, seed, args):
                if extra.get("timed_out"):
                    print(f"{e['source']}/{e['module']} level {level} item {i}: no return within "
                          f"{extra['timed_out']} s, skipped", file=sys.stderr)
                    continue
                rows.append(_jsonable({
                    "source": e["source"], "module": e["module"], "version": ver, "level": level,
                    "seed": seed, "index": i, "python_hash_seed": os.environ["PYTHONHASHSEED"],
                    "prompt": prompt, "gold": gold, "gold_in_prompt": _gold_in_prompt(gold, prompt),
                    "reference_trace": trace, **extra}))
        except Exception as exc:          # a module that cannot generate at this level
            errors += 1
            print(f"{e['source']}/{e['module']} level {level}: {type(exc).__name__}: {str(exc)[:160]}",
                  file=sys.stderr)
    return rows, errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whitelist", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prontoqa-repo", default=None)
    ap.add_argument("--entry", type=int, default=None, help=argparse.SUPPRESS)  # the per-entry child
    args = ap.parse_args(argv)
    hash_seed = os.environ.get("PYTHONHASHSEED", "")
    if not re.fullmatch(r"\d+", hash_seed):
        raise SystemExit("PYTHONHASHSEED must be pinned to an integer: instances are generated once, reproducibly")
    entries = json.load(open(args.whitelist))
    bad = sorted({e["source"] for e in entries} - set(SUPPORTED))
    if bad:
        raise SystemExit(f"no exporter for {bad} here")
    if any(e["source"] == "prontoqa" for e in entries) and not args.prontoqa_repo:
        raise SystemExit("--prontoqa-repo is required for prontoqa")
    if args.entry is not None:                     # child: one entry, rows to --out
        if sys.flags.ignore_environment or not (sys.flags.no_user_site and sys.flags.safe_path):
            raise SystemExit("the per-entry child must run with -s -P and its environment read (not -I/-E)")
        rows, errors = export_entry(entries[args.entry], args)
        with open(args.out, "w") as out:
            out.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        with open(args.out + ".summary", "w") as f:    # libraries print to stdout too
            json.dump({"rows": len(rows), "errors": errors}, f)
        return 0
    made = errors = 0
    versions = {}
    with open(args.out, "w") as out, tempfile.TemporaryDirectory() as tmp:
        for k, e in enumerate(entries):
            part = os.path.join(tmp, f"{k}.jsonl")
            cmd = [sys.executable, "-s", "-P", os.path.abspath(__file__), "--whitelist", args.whitelist,
                   "--n", str(args.n), "--seed-base", str(args.seed_base), "--out", part, "--entry", str(k)]
            if args.prontoqa_repo:
                cmd += ["--prontoqa-repo", args.prontoqa_repo]
            env = {k2: v for k2, v in os.environ.items() if not k2.startswith("PYTHON")}
            res = subprocess.run(cmd, capture_output=True, text=True, env={**env, "PYTHONHASHSEED": hash_seed})
            sys.stderr.write(res.stderr)
            if res.returncode != 0:
                raise SystemExit(f"{e['source']}/{e['module']}: exporter exited {res.returncode}")
            with open(part + ".summary") as f:
                got = json.load(f)
            errors += got["errors"]
            with open(part) as f:
                for line in f:
                    out.write(line)
                    made += 1
                    versions.setdefault(e["source"], json.loads(line)["version"])
    print(json.dumps({"rows": made, "module_level_errors": errors, "versions": versions}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
