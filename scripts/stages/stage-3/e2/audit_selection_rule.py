#!/usr/bin/env python
"""Compare two candidate-selection rules on one cleaned corpus.

The gates are identical by construction — both corpora come from the same
`build_cleaned_corpus.py` run configuration differing only in `--selection` — so
every difference this reports is attributable to the selection rule alone, on
exactly the prompts where the corpus's own candidate failed and a replacement
had to be chosen.

The question it answers: does median-length selection actually keep the
derivations that shortest-length selection throws away? `verify.select` claimed
it does; this measures it.

Usage:
    scripts/data/audit_selection_rule.py \
        --a artifacts/stage3/corpus_v2_clean          --a-name median \
        --b artifacts/stage3/corpus_v2_clean_shortest --b-name shortest \
        --a-rung artifacts/stage3/rung_0860k_clean_median \
        --b-rung <shortest rung dir> \
        --out artifacts/stage3/e2_selection_rule_audit.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.env import code_state  # noqa: E402


def load_sessions(path: Path) -> dict:
    out = {}
    with path.open() as f:
        for line in f:
            s = json.loads(line)
            out[s["id"]] = s
    return out


def load_per_example(path: Path) -> dict:
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def dist(values):
    if not values:
        return None
    v = sorted(values)
    return {"n": len(v), "mean": round(statistics.mean(v), 1),
            "p10": v[int(len(v) * 0.10)], "p50": v[len(v) // 2],
            "p90": v[int(len(v) * 0.90)], "max": v[-1], "sum": sum(v)}


def rung_facts(path: Path | None) -> dict | None:
    if path is None:
        return None
    ladder = json.loads((path / "ladder.json").read_text())
    rung = ladder["rungs"][0]
    ids = set()
    with (path / "audit.jsonl").open() as f:
        for line in f:
            for s in json.loads(line)["sessions"]:
                ids.add(s["session_id"])
    return {"n_blocks": rung["n_blocks"],
            "supervised_tokens": rung["actual_supervised_tokens"],
            "n_sessions": rung["n_sessions"],
            "token_mix": rung["token_mix"],
            "prompt_overlap_rate": ladder["matched_to"]["prompt_overlap_rate"],
            "session_ids": ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--a-name", default="a")
    ap.add_argument("--b-name", default="b")
    ap.add_argument("--a-rung", type=Path, default=None)
    ap.add_argument("--b-rung", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    sa, sb = load_sessions(args.a / "sessions_clean.jsonl"), \
        load_sessions(args.b / "sessions_clean.jsonl")
    pa, pb = load_per_example(args.a / "cleaning_per_example.jsonl"), \
        load_per_example(args.b / "cleaning_per_example.jsonl")

    if set(sa) != set(sb):
        raise SystemExit("the two corpora hold different prompts — the gates "
                         "must be identical for this comparison to be valid")

    replaced = [i for i in sa if not pa[i]["retained_original"]]
    changed = [i for i in replaced
               if sa[i]["candidate_index"] != sb[i]["candidate_index"]]

    by_type_changed = Counter(sa[i]["data_type"] for i in changed)
    by_type_replaced = Counter(sa[i]["data_type"] for i in replaced)

    lens_a = defaultdict(list)
    lens_b = defaultdict(list)
    for i in sa:
        lens_a[sa[i]["data_type"]].append(sa[i]["n_supervised_tokens"])
        lens_b[sb[i]["data_type"]].append(sb[i]["n_supervised_tokens"])

    # The claim under test, restricted to the prompts the rule actually decides.
    decided_a = defaultdict(list)
    decided_b = defaultdict(list)
    for i in changed:
        decided_a[sa[i]["data_type"]].append(sa[i]["n_supervised_tokens"])
        decided_b[sb[i]["data_type"]].append(sb[i]["n_supervised_tokens"])

    examples = []
    for i in sorted(changed)[:200]:
        examples.append({
            "id": i, "data_type": sa[i]["data_type"],
            "original_index": pa[i]["original_index"],
            "survivor_lengths": pa[i]["survivor_lengths"],
            f"{args.a_name}_index": sa[i]["candidate_index"],
            f"{args.a_name}_supervised": sa[i]["n_supervised_tokens"],
            f"{args.b_name}_index": sb[i]["candidate_index"],
            f"{args.b_name}_supervised": sb[i]["n_supervised_tokens"],
        })

    ra, rb = rung_facts(args.a_rung), rung_facts(args.b_rung)
    rung_cmp = None
    if ra and rb:
        shared = ra["session_ids"] & rb["session_ids"]
        rung_cmp = {
            args.a_name: {k: v for k, v in ra.items() if k != "session_ids"},
            args.b_name: {k: v for k, v in rb.items() if k != "session_ids"},
            "rung_prompt_agreement": round(len(shared) / len(ra["session_ids"]), 4),
            "supervised_delta": ra["supervised_tokens"] - rb["supervised_tokens"],
        }

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "rules": {args.a_name: json.loads(
            (args.a / "cleaning_audit.json").read_text())["selection_rule"],
            args.b_name: json.loads(
            (args.b / "cleaning_audit.json").read_text())["selection_rule"]},
        "code_state": code_state(REPO_ROOT),
        "prompts": {
            "total_kept": len(sa),
            "original_retained": len(sa) - len(replaced),
            "replaced_by_a_rule": len(replaced),
            "rules_disagree_on": len(changed),
            "disagreement_rate_among_replacements": round(
                len(changed) / len(replaced), 4) if replaced else 0.0,
            "replaced_by_type": dict(sorted(by_type_replaced.items())),
            "disagree_by_type": dict(sorted(by_type_changed.items())),
        },
        "corpus_target_lengths_supervised_tokens": {
            args.a_name: {t: dist(v) for t, v in sorted(lens_a.items())},
            args.b_name: {t: dist(v) for t, v in sorted(lens_b.items())},
        },
        "decided_prompts_only": {
            args.a_name: {t: dist(v) for t, v in sorted(decided_a.items())},
            args.b_name: {t: dist(v) for t, v in sorted(decided_b.items())},
        },
        "rung": rung_cmp,
        "changed_examples": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"kept {len(sa)} prompts; {len(replaced)} needed a replacement; the "
          f"rules disagree on {len(changed)} "
          f"({len(changed) / max(len(replaced), 1):.1%} of replacements)")
    print(f"  disagreements by type: {dict(sorted(by_type_changed.items()))}")
    print(f"\n{'type':14s} {'n':>4} {args.a_name + ' p50':>12} "
          f"{args.b_name + ' p50':>12} {args.a_name + ' mean':>12} "
          f"{args.b_name + ' mean':>12}   (decided prompts only)")
    for t in sorted(decided_a):
        a, b = dist(decided_a[t]), dist(decided_b[t])
        print(f"{t:14s} {a['n']:>4} {a['p50']:>12,} {b['p50']:>12,} "
              f"{a['mean']:>12,.0f} {b['mean']:>12,.0f}")
    if rung_cmp:
        print(f"\nrung: {args.a_name} {ra['supervised_tokens']:,} sup / "
              f"{ra['n_blocks']} blocks / overlap {ra['prompt_overlap_rate']:.1%}"
              f"   vs   {args.b_name} {rb['supervised_tokens']:,} sup / "
              f"{rb['n_blocks']} blocks / overlap {rb['prompt_overlap_rate']:.1%}")


if __name__ == "__main__":
    main()
