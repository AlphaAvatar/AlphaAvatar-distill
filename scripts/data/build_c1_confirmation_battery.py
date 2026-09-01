"""Build the frozen Phase-C1 confirmation battery (role C1_CONFIRMATION).

    PYTHONPATH=src .venv/bin/python scripts/data/build_c1_confirmation_battery.py \
        --out artifacts/stage3/c1_confirmation_v1

Its only job is the Phase-C1 two-arm ATTENTION isolation. It is not the
recovery-search battery, not the promotion battery, and never training data.

Three properties are load-bearing.

**The mixture is the frozen historical one, exactly.** 3:3:3:3:3:2 over the
scorable sets plus the same relative behaviour-only `code` component, scaled to
850 scorable / 950 total. It was *not* reweighted toward the sets that happened
to yield more correct answers historically: `correct_overall` and its SESOI are
defined **on this mixture**, so changing the mixture would change what
`+0.010 absolute` means. That ~142 of 170 historical prompts were never answered
correctly is a measurement result, not permission to reshape the distribution.

**Selection is by cryptographic rank, not iteration order.** Every eligible
example gets `SHA256(C0_digest : phase-c1-battery : stratum : stable_id)` and the
lowest ranks win. The key is fixed by a document frozen before any C1 candidate
existed, so the sample cannot have been chosen — by a person or by a loader's
row order — to favour an outcome. No model output touches it.

**Isolation is by stable id AND normalized prompt content**, against
`OPERATOR_CALIBRATION`, `STATE_EVALUATION`, `RECOVERY_SEARCH`, the entire
recovery-training corpus, and `FINAL_PROMOTION`. `FINAL_PROMOTION` is read only
to compute those exclusions — it is never sampled from and remains unconsumed as
an evaluation asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/data"))

from aadistill.data.extra_stream import content_sha256  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from battery_render import (  # noqa: E402
    RENDERERS,
    norm,
    rank_take,
    read_rows,
    source_digest,
)

BATTERY_ID = "c1_confirmation"
BATTERY_VERSION = 1
ROLE = "C1_CONFIRMATION"

#: The pushed Phase-C0 preregistration digest. The selection key, and nothing
#: else, is derived from it.
C0_DIGEST = "fb2eeea531f9f0d11f84b77cd47dff30697122de90a072a7a80c3a7535e89280"

#: set -> (domain, n, scorable). The frozen C0 mixture, verbatim.
SETS = {
    "gsm8k":         ("reasoning_math", 150, True),
    "math_verified": ("reasoning_math", 150, True),
    "multihop":      ("rag_multihop", 150, True),
    "rag":           ("rag_multihop", 150, True),
    "knowledge":     ("general", 150, True),
    "tool":          ("tool", 100, True),
    "code":          ("code", 100, False),
}

#: Exact pinned revisions — the same snapshots `recovery_search_v2` was built
#: from, so the two batteries differ in their sample, never in their source.
SOURCES = {
    "gsm8k":         ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
                      "main/test-00000-of-00001.parquet"),
    "math_verified": ("HuggingFaceH4/MATH-500",
                      "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test.jsonl"),
    "multihop":      ("hotpotqa/hotpot_qa", "1908d6afbbead072334abe2965f91bd2709910ab",
                      "distractor/validation-00000-of-00001.parquet"),
    "rag":           ("rajpurkar/squad_v2", "3ffb306f725f7d2ce8394bc1873b24868140c412",
                      "squad_v2/validation-00000-of-00001.parquet"),
    "knowledge":     ("mandarjoshi/trivia_qa", "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
                      "rc.nocontext/validation-00000-of-00001.parquet"),
    "tool":          ("Salesforce/xlam-function-calling-60k",
                      "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
                      "xlam_function_calling_60k.json"),
    "code":          ("google-research-datasets/mbpp",
                      "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
                      "full/test-00000-of-00001.parquet"),
}


def excluded_identities(args) -> tuple[set[str], set[str], dict]:
    """Ids and normalized prompt hashes this battery must avoid."""
    source_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    provenance: dict[str, dict] = {}

    # FINAL_PROMOTION — read for exclusion ONLY. Never sampled from.
    n = 0
    for path in sorted((REPO_ROOT / args.battery).glob("*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            source_ids.add(str(row["id"]))
            prompt_hashes.add(content_sha256(norm(row.get("prompt_text", ""))))
            n += 1
    provenance["final_promotion"] = {
        "asset": args.battery, "n_prompts": n,
        "note": "read for exclusion only; not sampled from, remains unconsumed"}

    # RECOVERY_SEARCH — the development battery. C1 confirmation must not reuse it.
    n = 0
    for path in sorted((REPO_ROOT / args.recovery_search).glob("*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            source_ids.add(str(row["id"]))
            if row.get("source_key"):
                source_ids.add(str(row["source_key"]))
            prompt_hashes.add(content_sha256(norm(row.get("prompt_text", ""))))
            n += 1
    provenance["recovery_search"] = {
        "asset": args.recovery_search, "n_prompts": n,
        "note": "historical/development evidence; excluded from C1 confirmation"}

    # Recovery training — the whole corpus, not just the 0.86M rung.
    corpus_ids, corpus_prompts = set(), 0
    for line in (REPO_ROOT / args.sessions).open():
        d = json.loads(line)
        corpus_ids.add(str(d["source_id"]))
        text = "\n".join(str(m.get("content", "")) for m in d["messages"]
                         if m.get("role") != "assistant")
        prompt_hashes.add(content_sha256(norm(text)))
        corpus_prompts += 1
    source_ids |= corpus_ids
    provenance["recovery_training"] = {
        "asset": args.sessions, "n_sessions": corpus_prompts,
        "distinct_source_ids": len(corpus_ids),
        "note": "the whole recovery corpus, not just the 0.86M rung"}

    for role, rel in (("initializer_state_eval", args.state_eval),
                      ("operator_calibration", args.calibration)):
        items = [json.loads(l) for l in (REPO_ROOT / rel / "items.jsonl").open()
                 if l.strip()]
        ids = {str(i.get("source_id")) for i in items if i.get("source_id")}
        source_ids |= ids
        provenance[role] = {"asset": rel, "n_items": len(items),
                            "excluded_source_ids": len(ids)}
    return source_ids, prompt_hashes, provenance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/stage3/c1_confirmation_v1")
    ap.add_argument("--battery", default="artifacts/eval/battery_v2")
    ap.add_argument("--recovery-search", default="artifacts/stage3/recovery_search_v2")
    ap.add_argument("--sessions", default="artifacts/stage3/corpus_v2/sessions.jsonl")
    ap.add_argument("--state-eval", default="artifacts/stage1/state_eval_v1")
    ap.add_argument("--calibration", default="artifacts/stage1/e8_calibration_v1")
    args = ap.parse_args()

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    exclude_ids, exclude_hashes, provenance = excluded_identities(args)

    built: dict[str, list[dict]] = {}
    sources: dict[str, dict] = {}
    for name, (_domain, want, _scorable) in SETS.items():
        repo, rev, rel = SOURCES[name]
        rows = read_rows(repo, rev, rel)
        if name == "gsm8k":          # index-based ids, as the v1 convention requires
            rows = [dict(r, _index=i) for i, r in enumerate(rows)]
        sources[name] = {**source_digest(repo, rev, rel), "n_rows": len(rows)}
        built[name] = rank_take(
            rows, want, stratum=name, base_digest=C0_DIGEST,
            exclude_ids=exclude_ids, exclude_hashes=exclude_hashes,
            make=RENDERERS[name])

    short = {k: (len(v), SETS[k][1]) for k, v in built.items() if len(v) < SETS[k][1]}
    if short:
        raise SystemExit(f"sets short of their target (got, want): {short}")

    all_hashes = [i["prompt_sha256"] for v in built.values() for i in v]
    if len(set(all_hashes)) != len(all_hashes):
        raise SystemExit("duplicate prompt content inside the battery")
    leaked = sorted(set(all_hashes) & exclude_hashes)
    if leaked:
        raise SystemExit(f"{len(leaked)} prompts collide with an excluded role")

    outputs = {}
    for name, items in built.items():
        items.sort(key=lambda i: str(i["id"]))
        path = out / f"{name}.jsonl"
        with path.open("w") as fh:
            for item in items:
                fh.write(json.dumps(item, sort_keys=True) + "\n")
        # Repo-relative when it can be, absolute otherwise. `relative_to` raises
        # on an --out outside the repo, and this project already carries that
        # exact crash as open debt in the authorization issuer; there is no
        # reason to reproduce it in new code.
        try:
            rel = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(path)
        outputs[name] = {"path": rel, "n": len(items),
                         "sha256": sha256_file(path), "domain": SETS[name][0],
                         "scorable": SETS[name][2]}

    scorable = [n for n in built if SETS[n][2]]
    # Same convention as recovery_search: sha256 over sorted `id:prompt_sha256`
    # pairs, so equality proves membership, ordering, ids and prompts are fixed.
    pairs = sorted(f"{i['id']}:{i['prompt_sha256']}"
                   for v in built.values() for i in v)
    content = hashlib.sha256("\n".join(pairs).encode()).hexdigest()

    manifest = {
        "artifact": f"{BATTERY_ID}_v{BATTERY_VERSION}",
        "role": ROLE,
        "battery_id": BATTERY_ID, "version": BATTERY_VERSION,
        "purpose": ("the Phase-C1 two-arm ATTENTION isolation confirmation battery; "
                    "never a promotion asset, never training data, never a "
                    "beam-ranking input"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "scripts/data/build_c1_confirmation_battery.py",
        "n_prompts": sum(len(v) for v in built.values()),
        "n_scorable_prompts": sum(len(built[n]) for n in scorable),
        "sets": outputs,
        "scorable_sets": sorted(scorable),
        "behaviour_only_sets": sorted(n for n in built if not SETS[n][2]),
        "domains": sorted({v[0] for v in SETS.values()}),
        "mixture": {k: v[1] for k, v in SETS.items()},
        "mixture_rule": ("the frozen historical recovery-search mixture 3:3:3:3:3:2 "
                         "over the scorable sets plus the same relative "
                         "behaviour-only code component, scaled to 850/950. NOT "
                         "reweighted by historical correctness."),
        "sampling_rule": {
            "order": "ascending cryptographic rank, ties by ascending stable id",
            "rank": ("SHA256(C0_preregistration_digest + ':phase-c1-battery:' + "
                     "stratum + ':' + stable_source_id)"),
            "base_digest": C0_DIGEST,
            "base_digest_source": "logs/phase_c0_preregistration.json, commit be2ab08",
            "outcome_dependence": "NONE — no model output of any kind is consulted",
            "difficulty_reweighting": ("none. Source-native metadata (MATH level, "
                                       "SQuAD answerability) is retained on the items "
                                       "and may be reported, but does not stratify."),
            "filter": "skip excluded ids and excluded normalized prompt hashes",
            "deterministic": True,
            "normalization": "whitespace-collapsed, lowercased, for cross-role hashing",
        },
        "isolation": provenance,
        "sources": sources,
        "rendering": ("scripts/data/battery_render.py — the same renderers, "
                      "instructions and id conventions as recovery_search, asserted "
                      "against the frozen artifact in tests/data/test_c1_battery.py"),
        "content_sha256": content,
        "content_sha256_convention": ("sha256 over newline-joined sorted "
                                      "'id:prompt_sha256' pairs"),
        "code_state": code_state(REPO_ROOT),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"{manifest['n_prompts']} prompts "
          f"({manifest['n_scorable_prompts']} scorable) -> {out}")
    for name, o in sorted(outputs.items()):
        print(f"  {name:16s} {o['n']:4d}  scorable={o['scorable']}")
    print(f"content_sha256  {content}")
    print(f"manifest_sha256 {manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
