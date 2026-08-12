"""Fail-closed isolation check across every AutoInitializer data role.

    PYTHONPATH=src .venv/bin/python scripts/data/check_autoinit_role_isolation.py \
        --out logs/autoinit_role_isolation.json

Five roles, and no prompt may appear under two of them:

    OPERATOR_CALIBRATION    what an operator measures its own decision on
    INITIALIZER_STATE_EVAL  what the beam ranks states on
    RECOVERY_SEARCH         what the 0.86M probes are selected on
    FINAL_PROMOTION         what a result is finally reported on
    RECOVERY_TRAINING       what the probes are trained on

The check is **typed**, and that is the part that matters. The assets are stored
in genuinely different representations — the calibration mixture and the state-eval
suite are token-id sequences, the batteries are rendered prompts, the recovery
corpus is chat sessions. Reducing all of them to "the prompt hash" is impossible,
and silently comparing a set of text hashes against a set of token hashes returns
"no overlap" for *every* input, including one that leaks. That is the worst
possible failure for a check like this: it always passes.

So identities are typed, compared only within a type, and a role pair sharing no
type at all is reported as **uncomparable** and fails closed. Near-duplicate
detection runs where a normalized text form exists on both sides.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/data"))

from aadistill.data.extra_stream import content_sha256  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from build_e8_calibration import sha_ids  # noqa: E402

ROLES = ("OPERATOR_CALIBRATION", "INITIALIZER_STATE_EVAL", "RECOVERY_SEARCH",
         "FINAL_PROMOTION", "RECOVERY_TRAINING")

#: Role pairs where sharing is expected and legitimate, with the reason. Nothing
#: is on this list today; it exists so that a future exemption has to be written
#: down rather than argued in a review.
ALLOWED_SHARING: dict[tuple[str, str], str] = {}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def shingles(text: str, k: int = 8) -> set[str]:
    words = norm(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


class RoleAssets:
    """Typed identities for one role."""

    def __init__(self, role: str) -> None:
        self.role = role
        self.ids: dict[str, set[str]] = defaultdict(set)   # type -> values
        self.texts: dict[str, str] = {}                     # item id -> normalized
        self.assets: list[dict] = []

    def add(self, *, item_id: str, source_id=None, token_ids=None, text=None,
            source_revision=None) -> None:
        self.ids["item_id"].add(str(item_id))
        if source_id:
            self.ids["source_id"].add(str(source_id))
        if token_ids:
            self.ids["token_content"].add(sha_ids(list(token_ids)))
        if text:
            self.ids["normalized_text"].add(content_sha256(norm(text)))
            self.texts[str(item_id)] = norm(text)
        if source_revision:
            self.ids["source_revision"].add(str(source_revision))


def load_roles(args) -> dict[str, RoleAssets]:
    roles = {r: RoleAssets(r) for r in ROLES}

    # OPERATOR_CALIBRATION and INITIALIZER_STATE_EVAL: token-id items.
    for role, rel in (("OPERATOR_CALIBRATION", args.calibration),
                      ("INITIALIZER_STATE_EVAL", args.state_eval)):
        directory = REPO_ROOT / rel
        items = [json.loads(l) for l in (directory / "items.jsonl").open() if l.strip()]
        for item in items:
            roles[role].add(item_id=item["item_id"], source_id=item.get("source_id"),
                            token_ids=item["ids"])
        roles[role].assets.append({
            "path": rel, "n_items": len(items),
            "items_sha256": sha256_file(directory / "items.jsonl"),
            "manifest_sha256": json.loads(
                (directory / "manifest.json").read_text()).get("manifest_sha256"),
            "representation": "token ids (+ source ids)"})

    # RECOVERY_SEARCH and FINAL_PROMOTION: rendered prompts.
    for role, rel in (("RECOVERY_SEARCH", args.recovery_search),
                      ("FINAL_PROMOTION", args.battery)):
        directory = REPO_ROOT / rel
        n = 0
        for path in sorted(directory.glob("*.jsonl")):
            for line in path.open():
                if not line.strip():
                    continue
                row = json.loads(line)
                roles[role].add(item_id=row["id"], source_id=row.get("source_key"),
                                text=row.get("prompt_text", ""))
                n += 1
        roles[role].assets.append({
            "path": rel, "n_items": n, "representation": "rendered prompt text"})

    # RECOVERY_TRAINING: what the frozen probe plan actually trains on — the
    # 0.86M rung, plus the pack's validation slice. Deliberately *not* the whole
    # corpus: sessions past the rung are never seen by a probe, and treating the
    # entire pool as training data would report a leak where there is none while
    # saying nothing about the rung that matters.
    pack = REPO_ROOT / args.pack
    ladder = json.loads((pack / "ladder.json").read_text())
    rung = next(r for r in ladder["rungs"]
                if r["target_supervised_tokens"] == args.probe_rung)
    n_blocks = int(rung["n_blocks"])
    trained_sessions: set[str] = set()
    with (pack / "audit.jsonl").open() as fh:
        for i, line in enumerate(fh):
            if i >= n_blocks + args.val_blocks:
                break
            for entry in json.loads(line)["sessions"]:
                trained_sessions.add(str(entry["session_id"]))

    by_id, pool = {}, 0
    for line in (REPO_ROOT / args.sessions).open():
        d = json.loads(line)
        by_id[str(d["id"])] = d
        pool += 1
    n = 0
    for session_id in sorted(trained_sessions):
        d = by_id.get(session_id)
        if d is None:
            continue
        text = "\n".join(str(m.get("content", "")) for m in d["messages"]
                         if m.get("role") != "assistant")
        roles["RECOVERY_TRAINING"].add(item_id=d["id"], source_id=d.get("source_id"),
                                       text=text)
        n += 1
    roles["RECOVERY_TRAINING"].assets.append({
        "path": args.sessions, "pack": args.pack,
        "probe_rung": args.probe_rung, "val_blocks": args.val_blocks,
        "blocks": n_blocks, "n_items": n, "corpus_pool": pool,
        "sha256": sha256_file(REPO_ROOT / args.sessions),
        "representation": "chat sessions (prompt text + source ids)",
        "note": ("the sessions the 0.86M probes actually train on; the surrounding "
                 "corpus pool is reported separately as an advisory")})

    # Token-id roles carry no text, so a comparison against a prompt battery would
    # be uncomparable and fail closed. Resolve their source ids back to the corpus
    # to recover a normalized-text identity. The check reconstructs the comparable
    # form rather than requiring the frozen artifacts to have stored it.
    for role in ("OPERATOR_CALIBRATION", "INITIALIZER_STATE_EVAL"):
        resolved = 0
        for source_id in list(roles[role].ids["source_id"]):
            for session_id, d in by_id.items():
                if str(d.get("source_id")) != source_id:
                    continue
                text = "\n".join(str(m.get("content", "")) for m in d["messages"]
                                 if m.get("role") != "assistant")
                roles[role].ids["normalized_text"].add(content_sha256(norm(text)))
                roles[role].texts[session_id] = norm(text)
                resolved += 1
                break
        roles[role].assets.append({
            "derived": "normalized_text resolved from source_id via the corpus",
            "resolved_items": resolved,
            "why": ("without this the pair against a prompt battery shares no "
                    "identity type and the check cannot see a leak either way")})
    return roles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_role_isolation.json")
    ap.add_argument("--calibration", default="artifacts/stage1/e8_calibration_v1")
    ap.add_argument("--state-eval", default="artifacts/stage1/state_eval_v1")
    ap.add_argument("--recovery-search", default="artifacts/stage3/recovery_search_v1")
    ap.add_argument("--battery", default="artifacts/eval/battery_v2")
    ap.add_argument("--sessions", default="artifacts/stage3/corpus_v2/sessions.jsonl")
    ap.add_argument("--pack", default="artifacts/stage3/ladder_uniform_probe")
    ap.add_argument("--probe-rung", type=int, default=860_000)
    ap.add_argument("--val-blocks", type=int, default=16)
    ap.add_argument("--near-duplicate-shingles", type=int, default=8)
    ap.add_argument("--near-duplicate-min-words", type=int, default=40)
    args = ap.parse_args()

    roles = load_roles(args)
    names = list(ROLES)
    overlaps, uncomparable, near_duplicates, pairs = [], [], [], []

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # `item_id` is excluded from the comparison: two roles can legitimately
            # use the same id scheme for different content, and an id collision
            # without a content collision is a naming coincidence, not a leak.
            types_a = {t for t in roles[a].ids if t != "item_id"}
            types_b = {t for t in roles[b].ids if t != "item_id"}
            shared_types = sorted(types_a & types_b)
            if not shared_types:
                uncomparable.append({
                    "role_a": a, "role_b": b,
                    "types_a": sorted(types_a), "types_b": sorted(types_b),
                    "reason": ("no identity type in common, so no overlap could have "
                               "been detected either way")})
                continue
            found = []
            for kind in shared_types:
                shared = roles[a].ids[kind] & roles[b].ids[kind]
                if shared:
                    found.append({"identity_kind": kind, "n_shared": len(shared),
                                  "examples": sorted(shared)[:5]})
            pairs.append({"role_a": a, "role_b": b, "compared_on": shared_types,
                          "exact_overlaps": found})
            if found:
                overlaps.append({"role_a": a, "role_b": b, "detail": found})

            # Near duplicates, where both sides carry text. Reported two ways.
            #
            # `strict` applies the shingle rule to every prompt regardless of
            # length. `scoped` applies the word floor the builders enforce, and is
            # the rule an asset was actually built against. Both are reported: the
            # strict count alone would flag prompts the builder deliberately
            # allowed, and the scoped count alone would hide short formulaic
            # collisions that a reader should still know about.
            if roles[a].texts and roles[b].texts:
                b_shingles = set()
                b_shingles_scoped = set()
                for text in roles[b].texts.values():
                    sh = shingles(text, args.near_duplicate_shingles)
                    b_shingles |= sh
                    if len(text.split()) >= args.near_duplicate_min_words:
                        b_shingles_scoped |= sh
                for item_id, text in roles[a].texts.items():
                    sh = shingles(text, args.near_duplicate_shingles)
                    if not sh:
                        continue
                    long_enough = len(text.split()) >= args.near_duplicate_min_words
                    strict = len(sh & b_shingles) / len(sh)
                    scoped = (len(sh & b_shingles_scoped) / len(sh)
                              if long_enough else 0.0)
                    if strict >= 0.5:
                        near_duplicates.append({
                            "role_a": a, "role_b": b, "item_id": item_id,
                            "shingle_overlap_strict": round(strict, 4),
                            "shingle_overlap_scoped": round(scoped, 4),
                            "long_enough_for_the_enforced_rule": long_enough,
                            "flagged_by_enforced_rule": scoped >= 0.5})

    complete = not uncomparable
    passed = not overlaps and complete
    report = {
        "schema": "aadistill.autoinit.role_isolation/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "roles": {r: {"assets": roles[r].assets,
                      "identity_kinds": {k: len(v) for k, v in sorted(roles[r].ids.items())}}
                  for r in names},
        "pairs_compared": pairs,
        "exact_overlaps": overlaps,
        "uncomparable_role_pairs": uncomparable,
        "near_duplicates": near_duplicates,
        "near_duplicate_counts": {
            "strict": len(near_duplicates),
            "flagged_by_enforced_rule": sum(
                1 for n in near_duplicates if n["flagged_by_enforced_rule"]),
        },
        "near_duplicate_rule": {
            "method": f"{args.near_duplicate_shingles}-word shingle Jaccard-on-A",
            "threshold": 0.5,
            "min_words_for_enforced_rule": args.near_duplicate_min_words,
            "scope": "role pairs where both sides carry prompt text",
            "note": ("strict ignores the word floor and will flag short formulaic "
                     "prompts; the enforced rule is what the builders applied"),
        },
        "allowed_sharing": {f"{a}|{b}": why for (a, b), why in ALLOWED_SHARING.items()},
        "complete": complete,
        "passed": passed,
    }
    report["report_sha256"] = sha256_json(report)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({
        "passed": passed, "complete": complete,
        "n_exact_overlaps": len(overlaps),
        "n_near_duplicates": len(near_duplicates),
        "uncomparable": [(u["role_a"], u["role_b"]) for u in uncomparable],
        "pairs_compared": [(p["role_a"], p["role_b"], p["compared_on"]) for p in pairs],
    }, indent=2))
    if not passed:
        raise SystemExit(
            "role isolation FAILED. A prompt that both steers the search and grades "
            "it makes the final number in-sample.")


if __name__ == "__main__":
    main()
