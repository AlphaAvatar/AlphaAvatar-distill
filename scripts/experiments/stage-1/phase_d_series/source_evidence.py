#!/usr/bin/env python3
"""Source-extension evidence for the three short D-series strata.

    PYTHONPATH=src:scripts:scripts/data python -m \
        experiments.phase_d_series.source_evidence --write

**What this is for.** `battery_family.py` says the six-role family is short in
three strata and names a candidate extension for each. This measures those
candidates against what the repository has actually consumed, so the maintainer
source decision rests on counts rather than on row totals quoted from a dataset
card.

**What it deliberately does NOT do.** It does not call any row *eligible*. The
live exclusion chain compares a stable id and a **normalized rendered prompt**,
exactly (`battery_render.norm` is whitespace-collapse plus lower-case). Running
it is a separate step that happens after a source is pinned, and only its output
is an eligible count. Everything here is either an upstream row count or a
measured overlap, labelled as such.

**It also does not download anything.** Two of the three candidates live in
repositories already pinned at a frozen revision whose other files are already
in the local hub cache, so their counts are measurable at `$0` offline. The
third is a different repository, is not cached, and is reported as needing a
pinning decision before it can be measured at all. A candidate nobody can
measure yet is a finding, not a gap to paper over.

**Why the near-duplicate screen is here and labelled a screen.** The live chain
is an exact comparison, so a restatement of a consumed problem passes it. For
MBPP that is not hypothetical: `full/train` task 602 is the same problem as the
already-consumed `full/test` task 217, with a different `task_id` and different
test asserts, so both of the chain's keys differ. The screen reports what the
chain would miss; it is not a replacement for the chain and its thresholds are
not acceptance criteria.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/data"):
    path = str(REPO_ROOT / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from battery_render import (  # noqa: E402
    FROZEN_SOURCES,
    RENDERERS,
    norm,
    read_rows,
    snapshot_path,
)

SCHEMA = "aadistill.autoinit.d_series_source_evidence/v1"
RECORD = "logs/shared/analyses/autoinit_d_series_source_evidence.json"

#: Evaluation pools already drawn from these strata. Read rather than assumed:
#: each one's `source_key` is what the exclusion chain keys on.
CONSUMED_POOLS = ("c1_confirmation_v1", "c2_screening_v1", "recovery_search_v2")

#: The recovery TRAINING corpus. Reserved for a different reason from the pools
#: above -- a behavioural prompt the student trained on is not a measurement.
TRAINING_CORPUS = "artifacts/stage3/corpus_v2/sessions.jsonl"

#: Stage-1 operator-calibration and state-evaluation assets. Checked for
#: provenance rather than content: they carry token ids, not prompt text.
STAGE1_ASSETS = ("e8_calibration_v1", "state_eval_v1", "reasoning_heavy_v2")

#: Candidate files, per stratum, inside the ALREADY-PINNED repository.
CANDIDATE_FILES: dict[str, tuple[str, ...]] = {
    "gsm8k": ("main/train-00000-of-00001.parquet",),
    "code": ("full/train-00000-of-00001.parquet",
             "full/validation-00000-of-00001.parquet",
             "full/prompt-00000-of-00001.parquet"),
}

#: Candidates that are a DIFFERENT repository and therefore a pinning decision.
#: Row counts are deliberately absent: the snapshot is not cached, and quoting a
#: dataset card's total here would be the transcription this module exists to
#: avoid.
UNPINNED_CANDIDATES: dict[str, dict[str, Any]] = {
    "math_verified": {
        "candidate_repo": "the upstream Hendrycks MATH release",
        "currently_pinned": "HuggingFaceH4/MATH-500, test.jsonl",
        "relationship": (
            "MATH-500 is a 500-item subset of the MATH test split, so the "
            "larger source is the SAME population sampled less sparsely -- a "
            "population change, not a task change. The subset was itself "
            "selected, so its difficulty distribution is not the parent's."),
        "status": "NOT CACHED, NOT PINNED, NOT MEASURED",
        "why_not_measured": (
            "a different repository from the pinned one, absent from the local "
            "hub cache, so no row count or overlap can be derived offline. "
            "Pinning it is a maintainer data decision (licence, revision, "
            "redistribution) and downloading a new dataset needs approval per "
            "AGENTS.md P15/4.4."),
        "owed_after_pinning": [
            "a revision pin and per-file digest",
            "the row count of the test split actually drawn from",
            "overlap against the consumed pools, the training corpus and "
            "MATH-500's own 500 rows",
            "renderer parity: `make_math_verified` reads `unique_id`, "
            "`subject`, `level`, `problem`, `answer`. An upstream release with "
            "different field names needs a renderer, and a renderer change is "
            "a new id scheme",
            "whether the 500 pinned rows are a subset, which the exclusion "
            "chain handles either way but which changes what the remainder is",
        ],
    },
}

#: Per-stratum six-role shortfall, taken from the family rather than restated.
def _shortfall() -> dict[str, int]:
    from experiments.phase_d_series.battery_family import requirement

    return {name: int(row["short_by"]) for name, row in requirement().items()
            if int(row["short_by"]) > 0}


# --- reserved populations ---------------------------------------------------

def _pool_rows(pool: str, group: str) -> list[dict]:
    p = REPO_ROOT / "artifacts/stage3" / pool / f"{group}.jsonl"
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def consumed_prompts(group: str) -> dict[str, Any]:
    """Normalized rendered prompts already drawn from this stratum, by pool."""
    per_pool, every = {}, set()
    keys: set[str] = set()
    for pool in CONSUMED_POOLS:
        rows = _pool_rows(pool, group)
        if not rows:
            continue
        got = {norm(r["prompt_text"]) for r in rows if r.get("prompt_text")}
        per_pool[pool] = len(rows)
        every |= got
        keys |= {str(r["source_key"]) for r in rows if r.get("source_key") is not None}
    return {"items_per_pool": per_pool, "normalized": every, "source_keys": keys}


def training_prompts() -> set[str]:
    """First user turn of every recovery-training session, normalized.

    Streamed: the corpus is ~76 MB and this runs on a dev box.
    """
    out: set[str] = set()
    p = REPO_ROOT / TRAINING_CORPUS
    if not p.is_file():
        return out
    with p.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            for m in (json.loads(line).get("messages") or []):
                if m.get("role") == "user":
                    text = str(m.get("content") or "")
                    if text:
                        out.add(norm(text))
                    break
    return out


def stage1_provenance() -> dict[str, Any]:
    """Whether the Stage-1 assets' items of these strata come from the corpus.

    They carry token ids rather than prompt text, so they cannot be compared by
    content. What can be checked is where they came from -- and if every one is
    traceable into `corpus_v2`, excluding the training corpus already excludes
    them, and the chain has one fewer independent population than it appears to.
    """
    corpus_ids: set[str] = set()
    p = REPO_ROOT / TRAINING_CORPUS
    if p.is_file():
        with p.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                for key in ("session_id", "id", "source_id"):
                    if d.get(key):
                        corpus_ids.add(str(d[key]))
    out = {}
    for asset in STAGE1_ASSETS:
        f = REPO_ROOT / "artifacts/stage1" / asset / "items.jsonl"
        if not f.is_file():
            out[asset] = {"status": "ABSENT"}
            continue
        total = traced = 0
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("subtype") not in ("gsm8k", "code"):
                continue
            total += 1
            sid = str(d.get("source_id") or d.get("session_id") or "")
            if sid in corpus_ids or sid.split("#")[0] in corpus_ids:
                traced += 1
        out[asset] = {"items_in_these_strata": total,
                      "traceable_into_corpus_v2": traced}
    return out


# --- near-duplicate screen --------------------------------------------------

def near_duplicate_screen(candidates: list[str], reserved: set[str],
                          *, rare_tokens: int = 4) -> dict[str, Any]:
    """Max token-set Jaccard against any reserved prompt, per candidate.

    A rare-token inverted index rather than an all-pairs comparison: 7,473
    candidates against 10,000 reserved prompts is 75M set operations, which is
    not a dev-box measurement. Each candidate is compared only to reserved
    prompts sharing one of its rarest tokens, which is where a restatement of
    the same problem will agree.

    **A SCREEN, with a stated blind spot.** It finds a near-duplicate that
    shares rare vocabulary. A paraphrase that replaces the rare words -- the
    numbers and names in a word problem, say -- is invisible to it. The
    thresholds below are descriptive and are NOT acceptance criteria.
    """
    df: collections.Counter = collections.Counter()
    tokens = {}
    for s in reserved:
        t = frozenset(s.split())
        tokens[s] = t
        df.update(t)
    index: dict[str, list[str]] = collections.defaultdict(list)
    for s, t in tokens.items():
        for w in sorted(t, key=lambda w: df[w])[:rare_tokens]:
            index[w].append(s)

    exact = 0
    maxima: list[float] = []
    for q in candidates:
        if q in reserved:
            exact += 1
            continue
        tq = frozenset(q.split())
        seen: set[str] = set()
        best = 0.0
        for w in sorted(tq, key=lambda w: df.get(w, 0))[:rare_tokens]:
            for c in index.get(w, ()):
                if c in seen:
                    continue
                seen.add(c)
                j = len(tq & tokens[c]) / max(1, len(tq | tokens[c]))
                best = max(best, j)
        maxima.append(best)
    return {
        "_what": ("max token-set Jaccard of each non-exact candidate against "
                  "any reserved prompt, via a rare-token index"),
        "_not_an_acceptance_criterion": (
            "a screen of what the exact-match chain would miss. Thresholds are "
            "descriptive. A paraphrase that replaces the rare tokens is "
            "invisible to it."),
        "exact_matches": exact,
        "compared": len(maxima),
        "with_any_neighbour": sum(1 for j in maxima if j > 0),
        "at_or_above": {str(t): sum(1 for j in maxima if j >= t)
                        for t in (0.9, 0.8, 0.7, 0.5)},
        "highest": round(max(maxima), 4) if maxima else None,
    }


# --- per-stratum evidence ---------------------------------------------------

#: The field holding the PROBLEM, where the rendered prompt wraps it in
#: boilerplate. `make_code` prepends a fixed 12-word instruction and appends the
#: test list, so two unrelated MBPP prompts already share vocabulary and the
#: screen's Jaccard is inflated toward agreement. Screening the bare field as
#: well separates "the same problem" from "the same wrapper".
BARE_PROBLEM_FIELD: dict[str, str] = {"code": "text"}


def bare_problem_screen(group: str, cand_rows: list[dict]) -> dict[str, Any] | None:
    """The near-duplicate screen on the problem alone, for wrapped prompts.

    The reserved side is rebuilt from the pinned file's own rows for the
    consumed `source_key`s, because the committed pools store the RENDERED
    prompt and the bare problem is not recoverable from it.
    """
    field = BARE_PROBLEM_FIELD.get(group)
    if not field:
        return None
    repo, revision, pinned_rel = FROZEN_SOURCES[group]
    pinned = {str(r["task_id"]): r for r in read_rows(repo, revision, pinned_rel)}
    keys = consumed_prompts(group)["source_keys"]
    reserved = {norm(pinned[k][field]) for k in keys if k in pinned}
    unresolved = sorted(k for k in keys if k not in pinned)
    screen = near_duplicate_screen([norm(r[field]) for r in cand_rows], reserved)
    screen["_scope"] = (
        f"on `{field}` only -- the problem without the instruction preamble or "
        "the test list. The reserved side is the pinned file's rows for the "
        f"{len(reserved)} consumed source_keys.")
    screen["_why_separately"] = (
        "the rendered-prompt screen above shares a fixed instruction across "
        "every item, which inflates agreement. This one answers whether the "
        "same PROBLEM reappears, which is what disjointness means.")
    if unresolved:
        screen["consumed_keys_absent_from_the_pinned_file"] = unresolved
    return screen


def pinned_stratum(group: str) -> dict[str, Any]:
    """Evidence for a candidate inside the already-pinned repository."""
    repo, revision, pinned_rel = FROZEN_SOURCES[group]
    snap = snapshot_path(repo, revision)
    make = RENDERERS[group]

    out: dict[str, Any] = {
        "upstream_identity": {"repo_id": repo, "revision": revision},
        "pinned_file_in_use": pinned_rel,
        "candidate_files": list(CANDIDATE_FILES[group]),
        "same_repository_and_revision": True,
        "_why_that_matters": (
            "licence, redistribution terms and the revision pin are INHERITED. "
            "No new data decision is needed about provenance -- only about "
            "whether a different split of the same release is an acceptable "
            "population for this stratum."),
        "snapshot_cached": snap.is_dir(),
    }
    if not snap.is_dir():
        out["status"] = "SNAPSHOT ABSENT -- nothing measurable offline"
        return out

    missing = [rel for rel in CANDIDATE_FILES[group] if not (snap / rel).is_file()]
    if missing:
        out["status"] = f"CANDIDATE FILES ABSENT FROM THE SNAPSHOT: {missing}"
        return out

    pinned_rows = read_rows(repo, revision, pinned_rel)
    rows: list[dict] = []
    per_file = {}
    for rel in CANDIDATE_FILES[group]:
        got = read_rows(repo, revision, rel)
        per_file[rel] = len(got)
        rows += got
    out["upstream_rows"] = {
        "_label": "UPSTREAM ROW COUNTS BEFORE ANY EXCLUSION -- not eligible counts",
        "pinned_file_in_use": len(pinned_rows),
        "candidate_files": per_file,
        "candidate_total": len(rows),
    }

    #: identity behaviour of the EXISTING renderer on candidate rows
    out["renderer"] = _renderer_evidence(group, make, pinned_rows, rows)

    consumed = consumed_prompts(group)
    training = training_prompts()
    cand_norm = [norm(make(r)["prompt_text"]) for r in _indexed(group, rows)]
    distinct = set(cand_norm)
    reserved = consumed["normalized"] | training
    out["exclusion_measurement"] = {
        "_label": ("overlap under the LIVE chain's comparison -- "
                   "`norm`, exact. Not an eligible count."),
        "consumed_pools": consumed["items_per_pool"],
        "consumed_distinct_normalized": len(consumed["normalized"]),
        "training_corpus_distinct_normalized": len(training),
        "candidate_distinct_normalized": len(distinct),
        "duplicates_within_candidates": len(cand_norm) - len(distinct),
        "overlap_consumed_pools": len(distinct & consumed["normalized"]),
        "overlap_training_corpus": len(distinct & training),
        "remaining_after_exact_exclusions": len(distinct - reserved),
    }
    out["near_duplicate_screen"] = near_duplicate_screen(cand_norm, reserved)
    bare = bare_problem_screen(group, rows)
    if bare is not None:
        out["bare_problem_screen"] = bare
    short = _shortfall().get(group)
    if short is not None:
        rest = len(distinct - reserved)
        out["against_the_family_shortfall"] = {
            "six_role_shortfall": short,
            "remaining_after_exact_exclusions": rest,
            "headroom": rest - short,
            "_not_a_sufficiency_claim": (
                "the remainder is measured under exact matching only. Whether "
                "it covers the shortfall depends on the full chain, which has "
                "not run, and on how many near-duplicates are rejected."),
        }
    return out


def _indexed(group: str, rows: list[dict]) -> list[dict]:
    """`make_gsm8k` needs `_index`; `check_group_parity` injects it the same way."""
    if group == "gsm8k":
        return [dict(r, _index=i) for i, r in enumerate(rows)]
    return rows


def _renderer_evidence(group: str, make, pinned_rows: list[dict],
                       cand_rows: list[dict]) -> dict[str, Any]:
    """Whether the EXISTING renderer can render the candidate rows safely.

    Field compatibility is necessary and not sufficient: the id scheme is what
    the exclusion chain and every frozen battery key on, so a renderer whose id
    is positional or split-named produces ids that are wrong for a new file even
    when every field is present.
    """
    pinned_fields = set(pinned_rows[0]) if pinned_rows else set()
    cand_fields = set(cand_rows[0]) if cand_rows else set()
    pinned_ids = {make(r)["id"] for r in _indexed(group, pinned_rows)}
    cand_ids = {make(r)["id"] for r in _indexed(group, cand_rows)}
    collide = sorted(pinned_ids & cand_ids)

    out: dict[str, Any] = {
        "fields_match_the_pinned_file": sorted(pinned_fields) == sorted(cand_fields),
        "pinned_fields": sorted(pinned_fields),
        "candidate_fields": sorted(cand_fields),
        "rendered_ids_pinned": len(pinned_ids),
        "rendered_ids_candidate": len(cand_ids),
        "id_collisions_with_the_pinned_file": len(collide),
        "example_collisions": collide[:3],
    }
    if collide:
        #: a collision is only benign if it is the same item
        pinned_text = {make(r)["id"]: make(r)["prompt_text"]
                       for r in _indexed(group, pinned_rows)}
        cand_text = {make(r)["id"]: make(r)["prompt_text"]
                     for r in _indexed(group, cand_rows)}
        same = sum(1 for i in collide
                   if norm(pinned_text[i]) == norm(cand_text[i]))
        out["colliding_ids_that_are_the_same_prompt"] = same
        out["BLOCKER"] = (
            f"{len(collide)} rendered ids collide with the pinned file and "
            f"{same} of them are the same prompt. The existing renderer CANNOT "
            "be used on this candidate: every colliding id would name a "
            "different problem than the already-consumed row holding that id, "
            "and ids are what the exclusion chain, the frozen batteries and "
            "every score record key on. A split-aware id scheme is required, "
            "and changing an id scheme is a data decision rather than a "
            "refactor.")
    else:
        out["id_scheme_note"] = _id_scheme_note(group)
    return out


def _id_scheme_note(group: str) -> str:
    if group == "code":
        return (
            "no collision: `task_id` partitions across the splits, so "
            "`mbpp-test-<task_id>` stays unique. The literal `test` in the id "
            "would nonetheless be WRONG for rows from another split -- "
            "cosmetic for uniqueness, misleading in provenance, and worth "
            "correcting when the source is pinned rather than after a battery "
            "is frozen on it.")
    return "no collision between the pinned file and the candidate files."


# --- the record -------------------------------------------------------------

def report() -> dict[str, Any]:
    strata = {g: pinned_stratum(g) for g in sorted(CANDIDATE_FILES)}
    for g, spec in UNPINNED_CANDIDATES.items():
        strata[g] = dict(spec)
    blockers = {g: v["renderer"]["BLOCKER"] for g, v in strata.items()
                if isinstance(v.get("renderer"), dict) and "BLOCKER" in v["renderer"]}
    return {
        "schema": SCHEMA,
        "_contract": (
            "Evidence for the maintainer source decision. AUTHORIZES NOTHING "
            "and MATERIALIZES NOTHING. No row here is called eligible: the live "
            "exclusion chain has not been run against any candidate, and only "
            "its output is an eligible count. $0, offline, no download."),
        "status": "EVIDENCE ONLY -- no source pinned, no battery materialized",
        "six_role_shortfall": _shortfall(),
        "strata": strata,
        "renderer_blockers": blockers or "none",
        "chain_gap": {
            "_what": (
                "the live chain isolates on a stable id AND a normalized "
                "rendered prompt, compared EXACTLY. It has no near-duplicate "
                "stage."),
            "demonstrated": (
                "MBPP `full/train` task 602 is the same problem as the "
                "already-consumed `full/test` task 217. Different `task_id` "
                "and different test asserts, so the rendered id and the "
                "rendered prompt both differ: the duplicate passes the chain. "
                "Found by the screen, not by the chain."),
            "consequence": (
                "cross-split expansion of a stratum needs a problem-text key "
                "added to the chain, or an explicit review of the screen's "
                "high-Jaccard candidates. Which, and at what threshold, is a "
                "maintainer decision -- a threshold invented here would be a "
                "guess wearing a number."),
        },
        "stage1_provenance": stage1_provenance(),
        "_stage1_provenance_meaning": (
            "the Stage-1 calibration and state-evaluation items of these "
            "strata carry token ids rather than text, so they cannot be "
            "compared by content. Every one is traceable into `corpus_v2`, so "
            "excluding the training corpus already excludes them and the chain "
            "has one fewer independent population than its documentation "
            "suggests."),
        "traps": [
            {"trap": "gsm8k `socratic` config",
             "why_not": (
                 "the same problems with a different answer rendering. It is "
                 "not additional capacity, and treating it as such would put "
                 "one problem into two roles of a family whose whole point is "
                 "disjointness.")},
            {"trap": "mbpp `sanitized` config",
             "why_not": (
                 "a hand-verified SUBSET of `full`, so it adds no problems "
                 "that `full` does not already contain.")},
            {"trap": "reading an upstream row total as capacity",
             "why_not": (
                 "every count in this record is upstream-rows or "
                 "measured-overlap. The eligible count is the output of the "
                 "exclusion chain, which has not been run.")},
        ],
        "still_owed_before_any_materialization": [
            "the maintainer source decision, per stratum",
            "a revision pin and per-file digest for anything newly pinned",
            "the live exclusion/contamination chain, run against the pinned "
            "candidate -- its output is the first eligible count",
            "a maintainer ruling on near-duplicate handling and its threshold",
            "a split-aware id scheme wherever `renderer_blockers` is non-empty",
            "renderer parity against the existing stratum, re-checked after any "
            "id or field change",
        ],
        "_authorizes": "nothing",
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help=f"regenerate {RECORD}")
    args = parser.parse_args(argv)

    doc = report()
    if args.write:
        out = REPO_ROOT / RECORD
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"wrote {RECORD}\n")

    print("D-series source evidence — EVIDENCE ONLY, AUTHORIZES NOTHING\n")
    print(f"  six-role shortfall : {doc['six_role_shortfall']}\n")
    for name, s in doc["strata"].items():
        print(f"  {name}")
        if "upstream_rows" not in s:
            print(f"      {s.get('status', 'no measurement')}")
            continue
        rows = s["upstream_rows"]
        ex = s["exclusion_measurement"]
        print(f"      upstream candidate rows        : {rows['candidate_total']}")
        print(f"      overlap: consumed / training   : "
              f"{ex['overlap_consumed_pools']} / {ex['overlap_training_corpus']}")
        print(f"      remaining (exact-match only)   : "
              f"{ex['remaining_after_exact_exclusions']}")
        scr = s["near_duplicate_screen"]
        print(f"      screen, rendered >=0.8 / >=0.7 : "
              f"{scr['at_or_above']['0.8']} / {scr['at_or_above']['0.7']}"
              f"   (highest {scr['highest']})")
        bare = s.get("bare_problem_screen")
        if bare:
            print(f"      screen, PROBLEM  >=0.8 / >=0.7 : "
                  f"{bare['at_or_above']['0.8']} / {bare['at_or_above']['0.7']}"
                  f"   (highest {bare['highest']}, exact {bare['exact_matches']})")
        if "BLOCKER" in s["renderer"]:
            print(f"      RENDERER BLOCKER               : "
                  f"{s['renderer']['id_collisions_with_the_pinned_file']} id collisions")
    print(f"\n  renderer blockers : {list(doc['renderer_blockers']) if isinstance(doc['renderer_blockers'], dict) else doc['renderer_blockers']}")
    print("\n  NO SOURCE PINNED. NO BATTERY MATERIALIZED. NO ROW CALLED ELIGIBLE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
