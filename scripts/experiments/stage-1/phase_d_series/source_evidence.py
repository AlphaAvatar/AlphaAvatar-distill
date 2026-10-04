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
    rank_take,
    read_rows,
    snapshot_path,
)

#: THE LIVE CHAIN, IMPORTED RATHER THAN REIMPLEMENTED. `excluded_identities` is
#: the function the frozen C1 confirmation battery was actually built with, and
#: `rank_take` is what APPLIES the exclusion -- stable id OR source_key against
#: the id set, `content_sha256(norm(prompt_text))` against the hash set, then a
#: within-battery dedup on that same hash. A second implementation of either
#: would be a second definition of eligibility, and the first time they drifted
#: the record would report a capacity the builder would not honour.
from build_c1_confirmation_battery import (  # noqa: E402
    C0_DIGEST,
    excluded_identities,
)

SCHEMA = "aadistill.autoinit.d_series_source_evidence/v1"
RECORD = "logs/shared/analyses/autoinit_d_series_source_evidence.json"

#: THE BASELINE CHAIN'S INPUTS, exactly the defaults `build_c1_confirmation_battery`
#: runs with. Passed to the imported `excluded_identities` as an argument object,
#: so the five populations it reads -- FINAL_PROMOTION, RECOVERY_SEARCH, the whole
#: recovery-training corpus, operator calibration and state evaluation -- and HOW
#: it reads each one are the builder's, not this module's.
#:
#: The first version of this module got two of those wrong: it omitted
#: FINAL_PROMOTION entirely, and it hashed only each training session's FIRST
#: USER TURN where the chain joins EVERY non-assistant message and also excludes
#: the corpus `source_id`. Importing the function is what makes that class of
#: error impossible rather than merely fixed.
BASELINE_INPUTS = {
    "battery": "artifacts/eval/battery_v2",
    "recovery_search": "artifacts/stage3/recovery_search_v2",
    "sessions": "artifacts/stage3/corpus_v2/sessions.jsonl",
    "state_eval": "artifacts/stage1/state_eval_v1",
    "calibration": "artifacts/stage1/e8_calibration_v1",
}

#: What the D-SERIES owes beyond the baseline chain. These post-date the C1
#: builder, so its defaults do not know about them, and a D-series battery that
#: reused any of them would be measuring a population its own design was tuned
#: against.
D_SERIES_ADDITIONAL_POOLS = ("c1_confirmation_v1", "c2_screening_v1")

#: Other active calibration profiles. `reasoning_heavy_v2` is the second mixture
#: D1's scoring policy is measured over, so its items are consumed material too.
D_SERIES_ADDITIONAL_CALIBRATION = ("reasoning_heavy_v2",)

#: Prior D-series roles, as they are allocated. Empty because none exists: the
#: family is DESIGNED and NOT MATERIALIZED. Listed so the next round adds a
#: directory name here rather than rediscovering the obligation.
D_SERIES_PRIOR_ROLES: tuple[str, ...] = ()

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


# --- the exclusion chain, reused ---------------------------------------------

def _namespace(**over: Any):
    """An argument object shaped like `build_c1_confirmation_battery`'s own."""
    from types import SimpleNamespace

    return SimpleNamespace(**{**BASELINE_INPUTS, **over})


def baseline_chain() -> tuple[set[str], set[str], dict]:
    """`(exclude_ids, exclude_hashes, provenance)` from the LIVE chain.

    A direct call into the builder's `excluded_identities`. Everything about what
    is excluded and how -- FINAL_PROMOTION's ids and prompt hashes, recovery
    search's ids AND source_keys, the whole training corpus's `source_id`s plus a
    hash of every session's non-assistant messages JOINED, and the two Stage-1
    assets' `source_id`s -- is the builder's definition, read from the same
    assets at the same default paths.
    """
    return excluded_identities(_namespace())


def _pool_rows(pool: str, group: str) -> list[dict]:
    p = REPO_ROOT / "artifacts/stage3" / pool / f"{group}.jsonl"
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def d_series_additional(group: str) -> tuple[set[str], set[str], dict]:
    """What the D-series owes on top of the baseline chain.

    Built with the same two keys the baseline uses and applied by the same
    `rank_take`, so "additional" means more populations and never different
    semantics.
    """
    from aadistill.data.extra_stream import content_sha256

    ids: set[str] = set()
    hashes: set[str] = set()
    provenance: dict[str, Any] = {}

    for pool in D_SERIES_ADDITIONAL_POOLS:
        rows = _pool_rows(pool, group)
        if not rows:
            provenance[pool] = {"status": "ABSENT or no rows in this stratum"}
            continue
        for r in rows:
            ids.add(str(r["id"]))
            if r.get("source_key"):
                ids.add(str(r["source_key"]))
            if r.get("prompt_text"):
                hashes.add(content_sha256(norm(r["prompt_text"])))
        provenance[pool] = {"n_items": len(rows),
                            "note": "post-dates the C1 builder, so its defaults "
                                    "do not know about it"}

    for asset in D_SERIES_ADDITIONAL_CALIBRATION:
        f = REPO_ROOT / "artifacts/stage1" / asset / "items.jsonl"
        if not f.is_file():
            provenance[asset] = {"status": "ABSENT"}
            continue
        items = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        got = {str(i["source_id"]) for i in items if i.get("source_id")}
        ids |= got
        provenance[asset] = {
            "n_items": len(items), "excluded_source_ids": len(got),
            "note": "the second calibration mixture D1's scoring policy is "
                    "measured over; items carry token ids, so only source_ids "
                    "can be excluded"}

    for role in D_SERIES_PRIOR_ROLES:          # none yet, by construction
        rows = _pool_rows(role, group)
        for r in rows:
            ids.add(str(r["id"]))
            if r.get("source_key"):
                ids.add(str(r["source_key"]))
            if r.get("prompt_text"):
                hashes.add(content_sha256(norm(r["prompt_text"])))
        provenance[role] = {"n_items": len(rows), "note": "prior D-series role"}
    if not D_SERIES_PRIOR_ROLES:
        provenance["prior_d_series_roles"] = {
            "n_roles": 0,
            "note": "none allocated: the family is DESIGNED and NOT "
                    "MATERIALIZED. Each role freezes into this list as it is "
                    "drawn, so round n+1 isolates against rounds 1..n."}
    return ids, hashes, provenance


def _reserved_text(group: str) -> set[str]:
    """Normalized reserved PROMPT TEXT, for the screen only.

    The chain carries hashes, which cannot be compared for similarity, so the
    screen needs the text back. Only the populations that actually store prompt
    text contribute: the committed pools and FINAL_PROMOTION. The training
    corpus's joined non-assistant text is included too, because a restatement of
    a trained prompt is exactly what the screen is for.

    Deliberately NOT used for any exclusion -- `eligible_rows` comes from the
    chain. This set exists so a similarity number can be reported beside it.
    """
    out: set[str] = set()
    for pool in (*D_SERIES_ADDITIONAL_POOLS, "recovery_search_v2"):
        for r in _pool_rows(pool, group):
            if r.get("prompt_text"):
                out.add(norm(r["prompt_text"]))
    battery = REPO_ROOT / BASELINE_INPUTS["battery"] / f"{group}.jsonl"
    if battery.is_file():
        for line in battery.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("prompt_text"):
                    out.add(norm(r["prompt_text"]))
    sessions = REPO_ROOT / BASELINE_INPUTS["sessions"]
    if sessions.is_file():
        with sessions.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                text = "\n".join(str(m.get("content", ""))
                                 for m in d.get("messages", [])
                                 if m.get("role") != "assistant")
                if text:
                    out.add(norm(text))
    return out


def training_corpus_content_gap(group: str, cand_rows: list[dict],
                                make) -> dict[str, Any]:
    """What the live chain's training-corpus CONTENT exclusion actually catches.

    The chain hashes each session's non-assistant messages JOINED. Every one of
    the corpus's sessions carries a system message, so that hash is
    `system + "\n" + user` and can never equal a bare rendered question --
    measured, not argued: the joined-hash set and the first-user-turn-hash set
    are DISJOINT.

    So for a candidate rendered as its bare prompt, the training-corpus
    protection runs entirely through `source_id`, and a NEW upstream source has
    no corpus source_ids to match. There is effectively no content protection.

    **This was latent and never bit, for a reason that the proposed gsm8k
    extension removes.** The corpus drew its gsm8k material from `main/train`
    and every historical battery drew from `main/test`, so split separation did
    the protecting and the inert hash was never load-bearing. Extending INTO
    `main/train` is precisely the case where it becomes load-bearing.
    """
    from aadistill.data.extra_stream import content_sha256

    sessions = REPO_ROOT / BASELINE_INPUTS["sessions"]
    if not sessions.is_file():
        return {"status": "TRAINING CORPUS ABSENT -- not measurable"}
    joined: set[str] = set()
    first_turn: set[str] = set()
    with sessions.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            msgs = d.get("messages") or []
            text = "\n".join(str(m.get("content", "")) for m in msgs
                             if m.get("role") != "assistant")
            if text:
                joined.add(content_sha256(norm(text)))
            for m in msgs:
                if m.get("role") == "user":
                    q = str(m.get("content") or "")
                    if q:
                        first_turn.add(content_sha256(norm(q)))
                    break

    hashes = [content_sha256(norm(make(r)["prompt_text"]))
              for r in _indexed(group, cand_rows)]
    caught = sum(1 for h in hashes if h in joined)
    present = sum(1 for h in hashes if h in first_turn)
    return {
        "_what": ("whether the chain's training-corpus content exclusion can "
                  "see a candidate that IS in the training corpus"),
        "joined_hashes": len(joined),
        "first_user_turn_hashes": len(first_turn),
        "the_two_sets_overlap_by": len(joined & first_turn),
        "candidates_caught_by_the_chain": caught,
        "candidates_present_as_a_corpus_first_user_turn": present,
        "IN_THE_TRAINING_CORPUS_BUT_NOT_CAUGHT": present - caught,
        "_verdict": (
            "INERT for bare-prompt candidates" if present and not caught
            else "no training-corpus content overlap in this candidate set"),
        "_why_it_never_bit_historically": (
            "the corpus drew this stratum from one split and every historical "
            "battery drew from another, so split separation protected them and "
            "this hash was never load-bearing. An extension INTO the corpus's "
            "own split removes that protection."),
    }


def math_pinning_readiness() -> dict[str, Any]:
    """What can be established about the MATH candidate WITHOUT downloading it.

    The upstream release is not cached, so its rows cannot be counted here. What
    can be measured is the side that already exists: the stratum the new source
    must be parity-compatible with. Doing that now means the pinning decision
    is not blocked on a download, and the adapter rule is verified before any
    bytes move.
    """
    from aadistill.data.verify import boxed_answer

    repo, revision, rel = FROZEN_SOURCES["math_verified"]
    snap = snapshot_path(repo, revision)
    if not (snap / rel).is_file():
        return {"status": "the pinned MATH-500 snapshot is absent"}
    rows = read_rows(repo, revision, rel)

    #: THE ADAPTER RULE, VERIFIED. Upstream Hendrycks MATH carries `solution` and
    #: no `answer`, so the adapter must derive the gold answer. This checks the
    #: derivation against the existing stratum's own gold field: if
    #: `boxed_answer(solution)` reproduces `answer` on every pinned row, the same
    #: rule applied upstream is the SAME correctness semantics rather than a new
    #: one invented for a new source.
    agree = sum(1 for r in rows if boxed_answer(r["solution"]) == r["answer"])

    subjects: dict[str, int] = {}
    levels: dict[str, int] = {}
    for r in rows:
        subjects[str(r["subject"])] = subjects.get(str(r["subject"]), 0) + 1
        levels[str(r["level"])] = levels.get(str(r["level"]), 0) + 1
    return {
        "_what": ("measurable readiness for the MATH pinning decision, derived "
                  "from the ALREADY PINNED MATH-500 without downloading the "
                  "candidate"),
        "parity_baseline": {"repo_id": repo, "revision": revision, "file": rel,
                            "rows": len(rows)},
        "answer_derivation_rule": {
            "rule": "aadistill.data.verify.boxed_answer(solution)",
            "reproduces_the_pinned_gold_answer": f"{agree}/{len(rows)}",
            "_why_this_settles_the_adapter": (
                "upstream carries `solution` and no `answer`, so the adapter "
                "must derive the gold. This rule reproduces the existing "
                "stratum's gold field exactly on every pinned row, so applying "
                "it upstream is the SAME correctness semantics rather than a "
                "new one introduced with a new source. Verified at $0, before "
                "any download."),
            "status": ("VERIFIED against the existing stratum" if agree == len(rows)
                       else "DOES NOT REPRODUCE THE GOLD -- the adapter needs a "
                            "different rule and this must be resolved first"),
        },
        "field_mapping_required": {
            "upstream_to_existing": {"type": "subject", "problem": "problem",
                                     "level": "level", "solution": "answer "
                                     "(via the derivation rule above)"},
            "_unique_id": (
                "the existing renderer builds `math500-<unique_id>`, and upstream "
                "has no `unique_id`. A split-aware D-series id is required, with "
                "the problem-content hash as a SEPARATE coordinate -- the same "
                "separation the other two strata owe."),
        },
        "distribution_baseline": {
            "_what": ("the existing stratum's subject and level distribution. "
                      "The owed shift measurement compares the pinned "
                      "candidate against THIS, and it is recorded now so the "
                      "comparison has a baseline that predates the decision."),
            "subjects": dict(sorted(subjects.items())),
            "levels": dict(sorted(levels.items())),
        },
        "still_requires_a_download_decision": (
            "row counts, overlap and the eligible count cannot be derived "
            "offline. Pinning and fetching the candidate is a maintainer data "
            "decision (AGENTS.md P15, 4.4) and nothing here presumes it."),
    }


def survivors(group: str, rows: list[dict], exclude_ids: set[str],
              exclude_hashes: set[str]) -> list[dict]:
    """Rows surviving the chain, counted by the function that APPLIES it.

    `rank_take` with `want` above the row count returns every eligible item in
    rank order, so `len()` is the survivor count under exactly the builder's
    predicate -- id-or-source_key exclusion, prompt-hash exclusion, and the
    within-battery dedup on that same hash. The rank itself cannot change the
    set, only its order, which is why a count may borrow it.
    """
    return rank_take(_indexed(group, rows), want=len(rows) + 1, stratum=group,
                     base_digest=C0_DIGEST, exclude_ids=exclude_ids,
                     exclude_hashes=exclude_hashes, make=RENDERERS[group])


def stage1_provenance() -> dict[str, Any]:
    """Whether the Stage-1 assets' items of these strata come from the corpus.

    They carry token ids rather than prompt text, so they cannot be compared by
    content -- which is exactly why the baseline chain excludes them by
    `source_id` and not by hash. If every one traces into `corpus_v2`, the
    corpus's own `source_id` exclusion already covers them.
    """
    corpus_ids: set[str] = set()
    p = REPO_ROOT / BASELINE_INPUTS["sessions"]
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
    for asset in (Path(BASELINE_INPUTS["calibration"]).name,
                  Path(BASELINE_INPUTS["state_eval"]).name,
                  *D_SERIES_ADDITIONAL_CALIBRATION):
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

#: What gets LOOKED AT, not what gets excluded. A number here is a review
#: trigger: the maintainer ruling on each pair is the decision, and the ruling
#: list becomes part of the family's content identity. Automatically excluding
#: everything above a similarity number would be inventing a semantic
#: equivalence threshold, which this round explicitly does not do.
REVIEW_TRIGGER_JACCARD = 0.8


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
    keys = {str(r["source_key"])
            for pool in (*D_SERIES_ADDITIONAL_POOLS, "recovery_search_v2")
            for r in _pool_rows(pool, group) if r.get("source_key") is not None}
    reserved = {norm(pinned[k][field]) for k in keys if k in pinned}
    unresolved = sorted(k for k in keys if k not in pinned)
    cand_norm = [norm(r[field]) for r in cand_rows]
    screen = near_duplicate_screen(cand_norm, reserved)
    #: THE REVIEW LIST, not a threshold. Each entry is a candidate a human has
    #: to rule on before the source is frozen, with the reserved problem it
    #: resembles and the native ids of both so the pair can be read side by side.
    #: `REVIEW_TRIGGER_JACCARD` selects what gets looked at; it decides nothing.
    by_hash = {norm(pinned[k][field]): k for k in keys if k in pinned}
    pairs = []
    for row, q in zip(cand_rows, cand_norm):
        hi, partner = 0.0, None
        tq = frozenset(q.split())
        for text, key in by_hash.items():
            tr = frozenset(text.split())
            j = len(tq & tr) / max(1, len(tq | tr))
            if j > hi:
                hi, partner = j, key
        if hi >= REVIEW_TRIGGER_JACCARD:
            pairs.append({
                "candidate_task_id": int(row["task_id"]),
                "candidate_problem": row[field],
                "resembles_consumed_task_id": int(partner),
                "consumed_problem": pinned[partner][field],
                "jaccard": round(hi, 4),
                "identical_problem_text": hi == 1.0,
            })
    pairs.sort(key=lambda d: -d["jaccard"])
    screen["review_list"] = {
        "_what": (
            f"candidates at Jaccard >= {REVIEW_TRIGGER_JACCARD} against a "
            "consumed problem. A REVIEW SURFACE: each pair needs a human ruling "
            "before the source is frozen, and the decisions become part of the "
            "family's content identity. The threshold is the review trigger and "
            "is NOT an equivalence criterion -- nothing here is excluded by it."),
        "_when": ("frozen BEFORE any D1 outcome exists, so a later result cannot "
                  "have informed which duplicates were dropped"),
        "_why_this_is_one_more_than_the_band_above": (
            "`at_or_above['0.8']` excludes exact matches -- "
            "`near_duplicate_screen` counts those separately as "
            "`exact_matches`, because a Jaccard of 1.0 is a different finding "
            "from a near miss. The review list includes them: an identical "
            "problem is the first thing a reviewer must see."),
        "n_to_review": len(pairs),
        "identical_problem_text": sum(1 for d in pairs
                                      if d["identical_problem_text"]),
        "pairs": pairs,
    }
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

    #: THE FOUR LEVELS. Each is produced by the function that APPLIES the
    #: chain, so "survivor" means the same thing at every level and only the
    #: populations differ.
    base_ids, base_hashes, base_prov = baseline_chain()
    add_ids, add_hashes, add_prov = d_series_additional(group)

    baseline_survivors = survivors(group, rows, base_ids, base_hashes)
    full_survivors = survivors(group, rows, base_ids | add_ids,
                               base_hashes | add_hashes)

    out["baseline_chain"] = {
        "_what": ("the LIVE C1 exclusion chain, by calling "
                  "`build_c1_confirmation_battery.excluded_identities` -- not a "
                  "reimplementation"),
        "populations": base_prov,
        "exclude_ids": len(base_ids),
        "exclude_prompt_hashes": len(base_hashes),
        "survivors": len(baseline_survivors),
        "_survivors_label": (
            "EXACT BASELINE-CHAIN SURVIVORS. Not yet eligible: the D-series "
            "owes further isolation below."),
    }
    out["d_series_isolation"] = {
        "_what": ("what the D-series owes beyond the baseline chain, applied by "
                  "the same `rank_take` predicate"),
        "populations": add_prov,
        "additional_exclude_ids": len(add_ids),
        "additional_exclude_prompt_hashes": len(add_hashes),
        "survivors": len(full_survivors),
        "removed_beyond_baseline": len(baseline_survivors) - len(full_survivors),
        "_survivors_label": "D-SERIES ADDITIONAL ISOLATION SURVIVORS",
    }
    out["eligible_rows"] = {
        "count": len(full_survivors),
        "_this_is_the_eligible_count": (
            "the output of the complete live chain -- baseline plus D-series "
            "isolation -- applied by the builder's own `rank_take`. This is the "
            "ONLY number in this record that may be called eligible, and it is "
            "eligible under EXACT identity only: the problem-content key and "
            "the pre-freeze duplicate review below are still owed."),
    }

    cand_norm = [norm(make(r)["prompt_text"]) for r in _indexed(group, rows)]
    reserved_for_screen = set(base_hashes) | set(add_hashes)
    #: the screen compares TEXT, so it needs the reserved text rather than its
    #: hashes -- rebuilt from the populations that carry prompt text.
    out["training_corpus_content_gap"] = training_corpus_content_gap(
        group, rows, make)
    out["near_duplicate_screen"] = near_duplicate_screen(
        cand_norm, _reserved_text(group))
    bare = bare_problem_screen(group, rows)
    if bare is not None:
        out["bare_problem_screen"] = bare
    short = _shortfall().get(group)
    if short is not None:
        out["against_the_family_shortfall"] = {
            "six_role_shortfall": short,
            "eligible_rows": len(full_survivors),
            "headroom": len(full_survivors) - short,
            "_not_a_sufficiency_claim": (
                "eligible under EXACT identity. The problem-content key is not "
                "yet in the chain, and the pre-freeze duplicate review has not "
                "run, so this headroom can only shrink."),
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
                "cross-split expansion of a stratum needs an exact "
                "problem-content key added to the chain, plus a pre-freeze "
                "review of the high-Jaccard pairs. The threshold selects what "
                "is reviewed and decides nothing -- one invented here would be "
                "a guess wearing a number."),
            "SECOND_GAP_FOUND_BY_CHAIN_PARITY": (
                "getting parity right revealed a larger hole than the one "
                "above. The chain hashes each training session's non-assistant "
                "messages JOINED, and every session carries a system message, "
                "so that hash can never equal a bare rendered question: the "
                "joined-hash set and the first-user-turn-hash set are measurably "
                "DISJOINT. For a new upstream source there are no corpus "
                "`source_id`s either, so the training-corpus protection is "
                "INERT. It caught 0 of the 1,708 gsm8k train rows that are "
                "literally in the corpus. See "
                "`strata.gsm8k.training_corpus_content_gap`."),
            "_why_the_second_gap_never_bit": (
                "the corpus drew each stratum from one split and every "
                "historical battery drew from another, so split separation did "
                "the protecting and the inert hash was never load-bearing. NO "
                "historical battery is contaminated -- measured, not assumed. "
                "The gsm8k extension is the first case that would rely on the "
                "hash, because it extends INTO the corpus's own split."),
            "_how_the_first_version_of_this_record_got_it_backwards": (
                "it compared each session's FIRST USER TURN, which is not what "
                "the chain does, and so reported 1,708 exclusions the chain "
                "would not make -- a remainder of 5,765 against the chain's "
                "actual 7,043. A non-parity measurement was accidentally "
                "stricter than the contract it claimed to describe, which hid "
                "the defect instead of finding it."),
        },
        "math_pinning_readiness": math_pinning_readiness(),
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
        "proposed_source_decision": {
            "_status": (
                "A PROPOSAL recorded for maintainer decision, from the review of "
                "2026-10-03. NOT a decision, NOT an authorization, and nothing "
                "is pinned or materialized by it. Each entry states what would "
                "be pinned and what must be built before it could be."),
            "code": {
                "preferred": ("google-research-datasets/mbpp, the SAME pinned "
                              "revision, full/train + full/validation + "
                              "full/prompt"),
                "second_dataset": (
                    "NOT to be introduced unless the complete chain proves this "
                    "source insufficient. Eligible under the chain as it is: "
                    "see `eligible_rows`, against a shortfall of 321."),
                "identity_requirements": [
                    "preserve the native `task_id` as provenance / source "
                    "identity",
                    "do NOT label a non-test row `mbpp-test-*`",
                    "use a split-aware D-series item id",
                    "add a canonical hash of the BARE PROBLEM TEXT as an "
                    "exclusion identity -- required by task 602 == task 217, "
                    "which differ in `task_id` and in rendered tests and so "
                    "pass both of the chain's current keys",
                ],
                "pre_freeze_review": (
                    "the `bare_problem_screen.review_list` pairs, ruled on and "
                    "frozen BEFORE any D1 outcome exists. The trigger threshold "
                    "selects what is looked at and decides nothing."),
            },
            "gsm8k": {
                "preferred": ("openai/gsm8k, the SAME pinned revision, "
                              "main/train"),
                "BLOCKED_ON": [
                    "the positional renderer may NOT be reused: it collides "
                    "with the consumed test split on every id it emits",
                    "the training-corpus content gap -- see "
                    "`training_corpus_content_gap`. The chain catches 0 of the "
                    "rows that ARE in the recovery-training corpus, because it "
                    "hashes joined non-assistant text and every session carries "
                    "a system message. This stratum's extension IS the split "
                    "the corpus drew from, so split separation no longer "
                    "protects it.",
                ],
                "identity_requirements": [
                    "historical test ids stay untouched",
                    "new train rows get a split-aware identity, e.g. "
                    "`gsm8k-main-train-00000`",
                    "bind a canonical normalized-question hash SEPARATELY as "
                    "the semantic exclusion identity -- the provenance "
                    "coordinate and the problem-content coordinate must not be "
                    "conflated",
                ],
                "not_capacity": "the `socratic` configuration",
                "_the_count_is_preliminary": (
                    "`eligible_rows` is eligible under the chain AS IT IS, and "
                    "the chain is measurably blind to this stratum's training "
                    "overlap. The usable figure is not known until the renderer "
                    "is corrected and a content key is added."),
            },
            "math_verified": {
                "preferred_population": (
                    "the canonical Hendrycks MATH TEST split -- not MATH train, "
                    "and not another rendering of MATH-500"),
                "status": ("APPROVED AS A CANDIDATE TO PIN AND MEASURE, not as "
                           "a materialized D-series source"),
                "pin_requirements": [
                    "an immutable revision of the canonical dataset "
                    "corresponding to the upstream Hendrycks MATH release",
                    "the exact test-file SHA256",
                    "the licence and provenance record",
                ],
                "measure_after_pinning": [
                    "actual test rows",
                    "overlap with MATH-500's 500 rows",
                    "overlap with every historical isolation role",
                    "recovery-training overlap",
                    "the eligible count the same exclusion contract would "
                    "produce -- not known for this stratum until it runs",
                    "subject/level distribution shift relative to the existing "
                    "`math_verified` stratum",
                    "renderer and scorer parity",
                ],
                "adapter_requirements": [
                    "the upstream representation uses fields equivalent to "
                    "`problem`, `level`, `type`, `solution`, so it is NOT "
                    "directly compatible with `make_math_verified`, which reads "
                    "`unique_id`, `subject`, `level`, `problem`, `answer`",
                    "map `type` -> `subject`",
                    "derive the final scored answer from `solution` under "
                    "EXACTLY the correctness semantics the existing "
                    "`math_verified` stratum uses",
                    "parity must pass before the source can be accepted",
                ],
                "scope": (
                    "a NEW D-series behavioural population. Historical "
                    "C1/C2/C3/A3 scores are NOT imported into it."),
            },
            "duplicate_policy_this_round": {
                "add": ("the exact PROBLEM-CONTENT identity described above, "
                        "which is principled and decidable"),
                "do_not_add": (
                    "automatic exclusion above a similarity number. Fuzzy "
                    "similarity stays a pre-freeze audit surface."),
                "freeze": (
                    "any semantic-duplicate exclusion list is frozen before "
                    "outcomes and becomes part of the family content identity"),
                "stop_condition": (
                    "if the review shows the exact-problem key is materially "
                    "insufficient, STOP and report rather than inventing a "
                    "general semantic-dedup system"),
            },
        },
        "still_owed_before_any_materialization": [
            "the maintainer source decision, per stratum",
            "for `math_verified`: a revision pin, a per-file digest and a "
            "licence/provenance record -- the chain has NOT been run against "
            "it, and its output would be that stratum's first eligible count",
            "an EXACT problem-content exclusion key added to the chain. The "
            "`eligible_rows` counts here are eligible under the chain AS IT IS, "
            "and the chain cannot see a restated problem",
            "for `gsm8k`: the training-corpus content gap must be closed before "
            "the train split can be used at all -- the chain catches 0 of the "
            "rows that are literally in the recovery-training corpus",
            "a split-aware id scheme wherever `renderer_blockers` is non-empty, "
            "with provenance and problem-content kept as SEPARATE coordinates",
            "a pre-freeze review list of the high-Jaccard bare-problem "
            "candidates, decided and frozen BEFORE any D1 outcome exists",
            "renderer and scorer parity against the existing stratum, "
            "re-checked after any id, field or adapter change",
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
        base, add = s["baseline_chain"], s["d_series_isolation"]
        print(f"      upstream rows                  : {rows['candidate_total']}")
        print(f"      baseline-chain survivors       : {base['survivors']}")
        print(f"      D-series isolation survivors   : {add['survivors']}"
              f"   (-{add['removed_beyond_baseline']} beyond baseline)")
        print(f"      ELIGIBLE (exact identity)      : "
              f"{s['eligible_rows']['count']}")
        gap = s.get("training_corpus_content_gap", {})
        missed = gap.get("IN_THE_TRAINING_CORPUS_BUT_NOT_CAUGHT")
        if missed:
            print(f"      !! IN TRAINING CORPUS, UNCAUGHT : {missed}"
                  f"   ({gap['_verdict']})")
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
