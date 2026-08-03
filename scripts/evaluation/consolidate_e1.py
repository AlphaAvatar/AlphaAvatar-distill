"""Consolidate every Experiment 1 measurement into one table and one report.

Pulls four measurements per arm — teacher-native val CE (from the run's own
train_log), FineWeb-Edu holdout NLL, uncapped behaviour, and GSM8K EM — and
reports the PCA-vs-random curves across six rungs and two seeds with the
between-seed spread, so the reader can see which differences clear the noise
and which do not.

    uv run python scripts/evaluation/consolidate_e1.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from pathlib import Path

RUNGS = [("0250k", 250_000), ("0460k", 460_000), ("0860k", 860_000),
         ("1600k", 1_600_000), ("2960k", 2_960_000), ("5500k", 5_500_000)]


_CONSOLE_EVAL = re.compile(r"eval step (\d+)(?: \(final\))?: (\{.*?\})")


def val_ce(paths: list[Path]) -> float | None:
    for p in paths:
        if not p.is_file():
            continue
        last = None
        for line in p.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "eval_result" and e.get("val_set", "val") == "val":
                last = e.get("val_ce")
        if last is not None:
            return last
    return None


def val_ce_from_console(path: Path) -> float | None:
    """Recover the final val CE from an arm's console log.

    Needed because the pca arms' per-arm `train_log.jsonl` were collapsed onto
    one basename by `scp 'host:dir/*/train_log.jsonl'` during the Experiment 1
    teardown; only the console logs are arm-unique, and they carry the same
    `eval step N: {...}` lines. Without this the pca arms read `val_ce: null`
    and the strongest instrument in the experiment goes missing from the
    consolidated table.
    """
    if not path.is_file():
        return None
    last = None
    for line in path.read_text(errors="replace").splitlines():
        m = _CONSOLE_EVAL.search(line)
        if m:
            try:
                last = ast.literal_eval(m.group(2)).get("val_ce", last)
            except (ValueError, SyntaxError):
                continue
    return last


def jload(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def collect(repo: Path, eval_dir: Path) -> list[dict]:
    rows = []
    arms = [(f"e1_r{tag}_{seed}_{init}", init, seed, toks)
            for init in ("pca", "rand") for seed in ("sa", "sb")
            for tag, toks in RUNGS]
    arms.append(("e1_ctl_r0250k_sa_pca_stepmatched", "pca", "sa", 250_000))
    for arm, init, seed, toks in arms:
        beh = jload(eval_dir / f"{arm}_behavior.json")
        gsm = jload(eval_dir / f"{arm}_gsm8k.json")
        nll_j = jload(eval_dir / f"{arm}_holdout.json")
        nll = None
        if nll_j and nll_j.get("results"):
            nll = nll_j["results"][0]["mean_nll_nats"]
        if nll is None:  # from the training pods
            for cand in (repo / f"artifacts/stage3/rescued/_relay/{arm}/eval_holdout_v1.json",
                         repo / f"artifacts/stage3/rescued/{arm}/eval_holdout_v1.json",
                         repo / f"artifacts/stage3/rescued/_logs_rand/{arm}/eval_holdout_v1.json"):
                j = jload(cand)
                if j and j.get("results"):
                    nll = j["results"][0]["mean_nll_nats"]
                    break
        rows.append({
            "arm": arm, "init": init, "seed": seed, "tokens": toks,
            "control": arm.startswith("e1_ctl"),
            "val_ce": val_ce([
                repo / f"artifacts/stage3/rescued/_relay/{arm}/train_log.jsonl",
                repo / f"artifacts/stage3/rescued/_logs_rand/{arm}/train_log.jsonl",
                repo / f"artifacts/stage3/rescued/{arm}/train_log.jsonl",
            ]) or val_ce_from_console(
                repo / f"artifacts/stage3/rescued/_logs_{init}/console_{arm}.log"),
            "holdout_nll": nll,
            "behavior": (beh or {}).get("behavior", {}).get("score"),
            "nat_term": (beh or {}).get("natural_termination_rate"),
            "degen": (beh or {}).get("degeneration_rate"),
            "degen_kinds": (beh or {}).get("degeneration_kinds"),
            "tokens_p50": (beh or {}).get("generated_tokens_p50"),
            "axes": (beh or {}).get("behavior", {}).get("axes"),
            "gsm8k_em": (gsm or {}).get("behavior", {}).get("axes", {}).get("math"),
            "gsm8k_nat_term": (gsm or {}).get("natural_termination_rate"),
        })
    return rows


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "--"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="artifacts/eval/e1")
    ap.add_argument("--out", default="artifacts/stage3/e1_consolidated.json")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    rows = collect(repo, Path(args.eval_dir))
    Path(args.out).write_text(json.dumps(rows, indent=1))

    by = {(r["init"], r["seed"], r["tokens"]): r for r in rows if not r["control"]}
    print(f"{'rung':>7s} | {'PCA sa':>8s} {'PCA sb':>8s} {'|Δ|':>6s} | "
          f"{'rnd sa':>8s} {'rnd sb':>8s} {'|Δ|':>6s}   (teacher-native val CE)")
    spreads = {"pca": [], "rand": []}
    for tag, toks in RUNGS:
        cells = []
        for init in ("pca", "rand"):
            a = by.get((init, "sa", toks), {}).get("val_ce")
            b = by.get((init, "sb", toks), {}).get("val_ce")
            d = abs(a - b) if isinstance(a, float) and isinstance(b, float) else None
            if d is not None:
                spreads[init].append(d)
            cells += [fmt(a), fmt(b), fmt(d) if d is not None else "--"]
        print(f"{tag:>7s} | {cells[0]:>8s} {cells[1]:>8s} {cells[2]:>6s} | "
              f"{cells[3]:>8s} {cells[4]:>8s} {cells[5]:>6s}")
    for init in ("pca", "rand"):
        s = spreads[init]
        if s:
            print(f"  {init}: between-seed |Δ| mean {statistics.mean(s):.4f} "
                  f"max {max(s):.4f} over {len(s)} rungs")

    print()
    print(f"{'arm':34s} {'CE':>8s} {'NLL':>8s} {'behav':>7s} {'nat':>6s} "
          f"{'degen':>6s} {'p50':>6s} {'gsmEM':>6s}")
    for r in sorted(rows, key=lambda r: (r["init"], r["seed"], r["tokens"], r["control"])):
        print(f"{r['arm']:34s} {fmt(r['val_ce']):>8s} {fmt(r['holdout_nll']):>8s} "
              f"{fmt(r['behavior']):>7s} {fmt(r['nat_term'],3):>6s} "
              f"{fmt(r['degen'],3):>6s} "
              f"{(str(r['tokens_p50']) if r['tokens_p50'] is not None else '--'):>6s} "
              f"{fmt(r['gsm8k_em'],3):>6s}")

    ems = [r["gsm8k_em"] for r in rows if isinstance(r["gsm8k_em"], float)]
    if ems:
        print(f"\nGSM8K EM across {len(ems)} checkpoints: "
              f"min {min(ems):.3f} max {max(ems):.3f} mean {statistics.mean(ems):.3f}")
    miss = [r["arm"] for r in rows if r["behavior"] is None or r["gsm8k_em"] is None]
    print(f"missing measurements: {miss or 'none'}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
