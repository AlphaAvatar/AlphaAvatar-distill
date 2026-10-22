"""Export WHITELISTED third-party generator modules to JSONL, inside an isolated env.

    <isolated venv>/bin/python -I external_export.py --whitelist <modules.json> \
        --n <per module and level> --seed-base <int> --out <export.jsonl>

Runs in its own pinned virtual environment (the third-party packages and
their dependencies -- including data they bundle, such as arckit's ARC-AGI
copies -- live there and nowhere else), with ``-I``, and imports nothing from
this repository. ``--whitelist`` is the only input: a JSON list of
``{"source", "module", "levels"}`` written by `external_tasks.py` from the
candidate modules of `external_modules.json`; a module not on it is never
imported or instantiated. The output is prompts, gold answers and the
generator's own reference trace when it has one -- the trace is exported for
the STEP check only and must never reach a prompt.

The training build reads only this file (by its sha256), never the library.
Supported sources: reasoning_gym and reasoning_core. Any other source on the
whitelist is refused here: PrOntoQA and DeepMind Mathematics have their own
command-line generators and get their own exporter when they are approved.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from importlib.metadata import version

SUPPORTED = ("reasoning_gym", "reasoning_core")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    return x if isinstance(x, (str, int, float, bool)) or x is None else repr(x)


def _gold_in_prompt(gold, prompt: str) -> bool:
    g = str(gold).strip()
    return bool(g) and re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", prompt) is not None


def rg_rows(module: str, level: int, n: int, seed: int):
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
        e = ds[i]
        gold = e["answer"] if e["answer"] is not None else (e.get("metadata") or {}).get("example_answer")
        yield i, e["question"], gold, None


def rc_rows(module: str, level: int, n: int, seed: int):
    import numpy as np
    import reasoning_core as rc
    for i in range(n):
        random.seed(seed + i)
        np.random.seed((seed + i) % (2 ** 32))
        ex = rc.get_task(module).generate_example(level=level)
        cot = dict(ex.metadata).get("cot")
        yield i, ex.prompt, ex.answer, cot if isinstance(cot, str) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whitelist", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    entries = json.load(open(args.whitelist))
    bad = sorted({e["source"] for e in entries} - set(SUPPORTED))
    if bad:
        raise SystemExit(f"no exporter for {bad} here")
    versions = {"reasoning_gym": version("reasoning-gym"), "reasoning_core": version("reasoning-core")}
    made = errors = 0
    with open(args.out, "w") as out:
        for k, e in enumerate(entries):
            for level in e["levels"]:
                seed = args.seed_base + 1000 * k + 100 * level
                rows = (rg_rows if e["source"] == "reasoning_gym" else rc_rows)(e["module"], level, args.n, seed)
                try:
                    for i, prompt, gold, trace in rows:
                        out.write(json.dumps(_jsonable({
                            "source": e["source"], "module": e["module"], "version": versions[e["source"]],
                            "level": level, "seed": seed, "index": i, "prompt": prompt, "gold": gold,
                            "gold_in_prompt": _gold_in_prompt(gold, prompt),
                            "reference_trace": trace}), ensure_ascii=False) + "\n")
                        made += 1
                except Exception as exc:          # a module that cannot generate at this level
                    errors += 1
                    print(f"{e['source']}/{e['module']} level {level}: {type(exc).__name__}: {str(exc)[:160]}",
                          file=sys.stderr)
    print(json.dumps({"rows": made, "module_level_errors": errors, "versions": versions}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
