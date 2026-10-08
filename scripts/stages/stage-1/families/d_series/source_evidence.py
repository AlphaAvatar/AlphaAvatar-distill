#!/usr/bin/env python3
"""Source-extension evidence for the three short D-series strata.

    PYTHONPATH=src:scripts:scripts/data python -m \
        stages.d_series.source_evidence --write

**What this is for.** `battery_family.py` says the six-role family is short in
three strata and names a candidate extension for each. This measures those
candidates against what the repository has actually consumed, so the maintainer
source decision rests on counts rather than on row totals quoted from a dataset
card.

**Counts are reported at four named levels, and `eligible` means one of them.**
Upstream rows; survivors of the historical chain; survivors of that plus the
D-series pool/calibration isolation (`current_exact_chain_survivors`); and
survivors of the **strengthened** contract, which adds canonical problem-content
identity. Only the last is called eligible, at exactly one place per stratum —
`eligible_rows.count`. The level below it is blind to recovery-training problem
content, which is why it is not an eligible count.

**Two of the three sources were measurable offline** — already-pinned
repositories whose other files were already in the local hub cache. The third,
the canonical MATH test population, was pinned and fetched under the maintainer
source decision of 2026-10-03: an immutable revision, test splits only, a SHA256
per file.

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
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/data"):
    path = str(REPO_ROOT / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

#: the ONE new concept this round adds, owned by the D-series application layer
from stages.d_series.identity import (  # noqa: E402
    RENDERED_PROMPT_IS_THE_PROBLEM,
    d_series_item_id,
    excluded_by_review,
    problem_content_id,
    problem_content_id_from_prompt,
    review_decision,
)

from shared.data.battery_render import (  # noqa: E402
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
from stages.phase_c1.build_c1_confirmation_battery import (  # noqa: E402
    C0_DIGEST,
    excluded_identities,
)

SCHEMA = "aadistill.autoinit.d_series_source_evidence/v1"
RECORD = "logs/stages/stage-1/families/d_series/analyses/autoinit_d_series_source_evidence.json"

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
    "battery": "artifacts/stages/stage-3/eval/battery_v2",
    "recovery_search": "artifacts/stages/stage-1/batteries/recovery_search_v2",
    "sessions": "artifacts/stages/stage-3/corpus_v2/sessions.jsonl",
    "state_eval": "artifacts/stages/stage-1/state_eval_v1",
    "calibration": "artifacts/stages/stage-1/e8_calibration_v1",
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
    from stages.d_series.battery_family import requirement

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
    #: pools are addressed by their frozen spelling (the path each had when it
    #: was drawn); the 2026-10-08 migration moved the bytes, so access resolves
    #: through the historical-path table.
    from shared.run_layout import resolve_historical
    p = REPO_ROOT / resolve_historical(f"artifacts/stage3/{pool}") / f"{group}.jsonl"
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
        f = REPO_ROOT / "artifacts/stages/stage-1" / asset / "items.jsonl"
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


def math_stratum() -> dict[str, Any]:
    """The pinned canonical MATH test population, measured.

    Reported through the SAME four levels as the other two strata, but the
    historical chain cannot be run on it directly: its rows are not in
    `FROZEN_SOURCES` and `make_math_verified` reads fields this source does not
    have. So the contract is applied by its coordinates rather than by
    `rank_take` -- the rendered-prompt hash the historical chain compares, then
    the problem-content coordinate -- and that difference is stated rather than
    hidden behind a shared number.
    """
    from aadistill.data.verify import boxed_answer
    from stages.d_series import math_source as ms

    out: dict[str, Any] = {"pin": ms.file_manifest()}
    if not ms.is_fetched():
        out["status"] = "PINNED BUT NOT FETCHED -- nothing measurable"
        return out
    rows = ms.rows()
    out["upstream_rows"] = {
        "_label": "UPSTREAM TEST ROWS BEFORE ANY EXCLUSION -- not eligible counts",
        "total": len(rows),
        "per_subject": dict(sorted(collections.Counter(
            r["subject"] for r in rows).items())),
    }

    #: adapter + scorer parity against the already-frozen stratum
    repo, revision, rel = FROZEN_SOURCES["math_verified"]
    pinned = read_rows(repo, revision, rel)
    pinned_problems = {norm(r["problem"]): r for r in pinned}
    derivable = sum(1 for r in pinned if boxed_answer(r["solution"]) == r["answer"])
    shared = [r for r in rows if norm(r["problem"]) in pinned_problems]
    gold_agrees = sum(1 for r in shared
                      if r["answer"] == pinned_problems[norm(r["problem"])]["answer"])
    out["adapter_parity"] = {
        "_what": ("the adapter must produce, for the rows the two populations "
                  "share, exactly what the frozen stratum already holds"),
        "rule": "aadistill.data.verify.boxed_answer(solution)",
        "reproduces_pinned_gold_on_the_pinned_file": f"{derivable}/{len(pinned)}",
        "shared_problems": len(shared),
        "gold_agrees_on_shared_problems": f"{gold_agrees}/{len(shared)}",
        "level_mapping": "upstream 'Level N' -> N; the frozen stratum stores N",
        "subject_mapping": "config name -> the frozen stratum's subject value",
        "status": ("PARITY HOLDS" if derivable == len(pinned)
                   and gold_agrees == len(shared)
                   else "PARITY FAILS -- the adapter must not be accepted"),
    }

    #: level/subject distribution shift from the frozen baseline
    def dist(rs, key):
        c = collections.Counter(str(r[key]) for r in rs)
        n = sum(c.values()) or 1
        return {k: round(v / n, 4) for k, v in sorted(c.items())}
    out["distribution_shift"] = {
        "_what": ("the candidate population against the frozen MATH-500 "
                  "baseline. A population change, which the maintainer decision "
                  "accepts explicitly -- recorded so it is not discovered later"),
        "subject": {"candidate": dist(rows, "subject"),
                    "frozen_math500": dist(pinned, "subject")},
        "level": {"candidate": dist(rows, "level"),
                  "frozen_math500": dist(pinned, "level")},
    }

    #: the contract, by coordinate
    base_ids, base_hashes, base_prov = baseline_chain()
    add_ids, add_hashes, add_prov = d_series_additional("math_verified")
    rendered_excluded = sum(
        1 for r in rows
        if _sha(norm(r["problem"])) in (base_hashes | add_hashes))
    out["baseline_chain"] = {
        "_what": ("the historical chain's rendered-prompt hash, applied by "
                  "coordinate. `make_math_verified` reads `unique_id`, which "
                  "this source lacks, so `rank_take` cannot be called on it."),
        "populations": base_prov,
        "removed_by_rendered_prompt_hash": rendered_excluded,
        "survivors": len(rows) - rendered_excluded,
        "_survivors_label": "EXACT BASELINE-CHAIN SURVIVORS. Not yet eligible.",
    }
    out["d_series_isolation"] = {
        "populations": add_prov,
        "survivors": len(rows) - rendered_excluded,
        "removed_beyond_baseline": 0,
        "_note": ("the D-series pools' math prompts are rendered as the bare "
                  "problem, so they are already inside the rendered-prompt "
                  "hash above; the problem-content coordinate below is what "
                  "adds isolation for this stratum"),
        "_survivors_label": "D-SERIES ADDITIONAL ISOLATION SURVIVORS",
    }
    out["current_exact_chain_survivors"] = {
        "count": len(rows) - rendered_excluded,
        "_what_this_is_NOT": (
            "NOT an eligible count. The contract as it exists is blind to "
            "recovery-training problem content, exactly as for the other two "
            "strata."),
    }

    #: Applied to the rows the CURRENT contract already admits, so the number
    #: reported is what this coordinate ADDS. Measuring it against all 5,000
    #: would re-count the 330 the rendered-prompt hash already removed -- and for
    #: this source the two coincide, because the rendered prompt IS the problem.
    reserved = reserved_problem_content("math_verified")
    admitted = [r for r in rows
                if _sha(norm(r["problem"])) not in (base_hashes | add_hashes)]
    kept, by_content, seen = 0, 0, set()
    for r in admitted:
        cid = problem_content_id("math_verified", r)
        if cid in reserved["ids"] or cid in seen:
            by_content += 1
            continue
        seen.add(cid)
        kept += 1
    out["strengthened_contract"] = {
        "_what": ("the contract plus canonical problem-content identity, "
                  "applied to what the contract already admits so the figure is "
                  "what this coordinate ADDS. For this source the rendered "
                  "prompt IS the problem, so the two coordinates largely "
                  "coincide and the addition is small by construction -- which "
                  "is a property of the source, not a weaker contract."),
        "survivors": kept,
        "removed_by_problem_content": by_content,
        "_removed_beyond_the_current_contract": by_content,
        "removed_by_frozen_review": 0,
        "unhashable_rows": 0,
        "reserved_problem_content_ids": len(reserved["ids"]),
        "reserved_provenance": reserved["provenance"],
        "reserved_unrecoverable": reserved["n_unrecoverable"],
        "frozen_review": review_decision("math_verified"),
    }
    out["eligible_rows"] = {
        "count": kept,
        "_this_is_the_eligible_count": (
            "the output of the complete strengthened chain for this stratum."),
    }
    short = _shortfall().get("math_verified")
    if short is not None:
        out["against_the_family_shortfall"] = {
            "six_role_shortfall": short,
            #: A REFERENCE, not a copy. This block previously carried its own
            #: numeric `eligible_rows`, which went stale the moment the
            #: strengthened contract changed the authoritative count -- it read
            #: 464/7043 against the real 463/5433, with prose saying the
            #: problem-content key was still owed. Two machine-readable values
            #: called eligible is one too many.
            "eligible_ref": "eligible_rows.count",
            "headroom": out["eligible_rows"]["count"] - short,
        }
    out["scope"] = (
        "a NEW D-series behavioural population. Historical C1/C2/C3/A3 scores "
        "are measured on a different population and are NOT imported or "
        "compared to anything measured here.")
    return out


def _sha(text: str) -> str:
    from aadistill.data.extra_stream import content_sha256

    return content_sha256(text)


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
        "_download_decision": (
            "MADE and EXECUTED 2026-10-03. This block was written while the "
            "candidate was unpinned, to show the pinning decision was not "
            "blocked on a download for the part that did not need one. The "
            "measured counts now live in `strata.math_verified`; this stays as "
            "the parity baseline those counts were checked against."),
    }


# --- the strengthened D-series contract -------------------------------------

def _recover(group: str, rows: list[dict], by_key: dict[str, dict],
             into: set[str]) -> tuple[int, int, str]:
    """Add each reserved item's problem-content id, by whichever route works.

    Two routes, and which one applies is a DECLARED property of the source: where
    the rendered prompt IS the problem, hash it directly; otherwise recover the
    problem through the pinned source file by native key. A source with neither
    is a real hole, reported rather than counted as zero.
    """
    direct = RENDERED_PROMPT_IS_THE_PROBLEM.get(group, False)
    got = miss = 0
    for r in rows:
        if direct and r.get("prompt_text"):
            into.add(problem_content_id_from_prompt(group, r["prompt_text"]))
            got += 1
            continue
        src = by_key.get(str(r.get("source_key") or ""))
        if src is not None:
            into.add(problem_content_id(group, src))
            got += 1
        else:
            miss += 1
    how = ("from `prompt_text` directly -- this source renders the bare problem"
           if direct else
           "through the pinned source file by native key" if by_key else
           "NO ROUTE: no native key and the rendering wraps the problem")
    return got, miss, how


def reserved_problem_content(group: str) -> dict[str, Any]:
    """Problem-content ids of every reserved problem that can be reconstructed.

    The historical chain carries hashes of RENDERED prompts and of joined
    training text, neither of which is a problem-content id, so this set has to
    be rebuilt from the sources. Each contributor says what it can and cannot
    answer, because a population whose problem text is unrecoverable is a
    coverage limit and not a zero.

    `n_unrecoverable` is the number that matters on review: it is how many
    reserved problems this contract is blind to.
    """
    from shared.data.battery_render import FROZEN_SOURCES, read_rows

    ids: set[str] = set()
    provenance: dict[str, Any] = {}
    unrecoverable = 0

    #: 1. the consumed evaluation pools and FINAL_PROMOTION store the RENDERED
    #: prompt, so the bare problem is recovered through the pinned source file by
    #: `source_key` where the source has a native key.
    repo, revision, pinned_rel = FROZEN_SOURCES[group]
    by_key: dict[str, dict] = {}
    if group == "code":
        by_key = {str(r["task_id"]): r
                  for r in read_rows(repo, revision, pinned_rel)}

    for pool in (*D_SERIES_ADDITIONAL_POOLS, "recovery_search_v2"):
        rows = _pool_rows(pool, group)
        if not rows:
            provenance[pool] = {"status": "no rows in this stratum"}
            continue
        got, miss, how = _recover(group, rows, by_key, ids)
        unrecoverable += miss
        provenance[pool] = {"items": len(rows), "problem_content_recovered": got,
                            "unrecoverable": miss, "_how": how}

    battery = REPO_ROOT / BASELINE_INPUTS["battery"] / f"{group}.jsonl"
    if battery.is_file():
        rows = [json.loads(l) for l in battery.read_text().splitlines() if l.strip()]
        got, miss, how = _recover(group, rows, by_key, ids)
        unrecoverable += miss
        provenance["final_promotion"] = {
            "items": len(rows), "problem_content_recovered": got,
            "unrecoverable": miss, "_how": how}

    #: 2. THE RECOVERY-TRAINING CORPUS, by its FIRST USER PROBLEM -- not the
    #: joined system+user text the historical chain hashes. This is the gap the
    #: parity round measured: the historical hash catches 0 of the GSM8K train
    #: rows that are in the corpus, because no bare question can ever equal
    #: `system + "\n" + user`.
    sessions = REPO_ROOT / BASELINE_INPUTS["sessions"]
    if sessions.is_file():
        n = 0
        with sessions.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                for m in (json.loads(line).get("messages") or []):
                    if m.get("role") == "user":
                        q = str(m.get("content") or "")
                        if q:
                            ids.add(hashlib.sha256(norm(q).encode()).hexdigest())
                            n += 1
                        break
        provenance["recovery_training"] = {
            "first_user_problems_hashed": n,
            "_why_first_user_and_not_joined": (
                "the historical chain hashes non-assistant messages JOINED, and "
                "every session carries a system message, so that hash can never "
                "equal a bare problem. Hashing the first user problem is what "
                "closes the gap -- and it does NOT change the historical chain, "
                "which keeps its own semantics so frozen battery membership "
                "stays reproducible."),
        }
    return {"ids": ids, "provenance": provenance,
            "n_unrecoverable": unrecoverable}


def strengthened_survivors(group: str, rows: list[dict],
                           baseline_ids: set[str], baseline_hashes: set[str],
                           add_ids: set[str], add_hashes: set[str]
                           ) -> dict[str, Any]:
    """Survivors of the historical chain PLUS problem-content isolation.

    Runs the historical contract first, by the historical code, and then applies
    the one new coordinate. The order matters for reporting, not for the result:
    it keeps "what the old contract admitted" separately visible from "what the
    new one removes".
    """
    survived = survivors(group, rows, baseline_ids | add_ids,
                         baseline_hashes | add_hashes)
    by_item = {str(i["id"]): i for i in survived}

    reserved = reserved_problem_content(group)
    review_excluded = excluded_by_review(group)

    #: map each surviving rendered item back to its source row, to hash the
    #: problem rather than the rendering
    source_of: dict[str, dict] = {}
    for index, row in enumerate(rows):
        item = RENDERERS[group](dict(row, _index=index)
                               if group == "gsm8k" else row)
        if item is not None:
            source_of[str(item["id"])] = dict(row, _index=index)

    kept, by_content, by_review, unhashable = [], 0, 0, 0
    seen: set[str] = set()
    for item_id in by_item:
        src = source_of.get(item_id)
        if src is None:
            unhashable += 1
            continue
        try:
            content = problem_content_id(group, src)
        except ValueError:
            unhashable += 1
            continue
        if content in reserved["ids"]:
            by_content += 1
            continue
        if content in seen:              # two candidate rows, one problem
            by_content += 1
            continue
        seen.add(content)
        kept.append(item_id)
    #: the frozen review decision, applied by D-SERIES id
    d_ids = {}
    for index, row in enumerate(rows):
        try:
            d_ids[str(RENDERERS[group](dict(row, _index=index) if group == "gsm8k"
                                       else row)["id"])] = d_series_item_id(
                group, *_config_split(group), row, index=index)
        except Exception:
            continue
    #: the frozen review's exclusions, applied by D-SERIES id. Written plainly:
    #: an earlier version had `not in review_excluded or not review_excluded`,
    #: which is true for every item when the set is empty AND when it is not --
    #: a filter that filtered nothing.
    final = [i for i in kept if d_ids.get(i) not in review_excluded]
    by_review = len(kept) - len(final)
    return {
        "survivors": len(final),
        "removed_by_problem_content": by_content,
        "removed_by_frozen_review": by_review,
        "unhashable_rows": unhashable,
        "reserved_problem_content_ids": len(reserved["ids"]),
        "reserved_provenance": reserved["provenance"],
        "reserved_unrecoverable": reserved["n_unrecoverable"],
        "frozen_review": review_decision(group),
    }


def _config_split(group: str) -> tuple[str, str]:
    """The config/split label a D-series id carries, from the candidate file."""
    rels = CANDIDATE_FILES.get(group, ())
    if not rels:
        return ("unknown", "unknown")
    head = rels[0]
    config = head.split("/")[0]
    split = head.split("/")[-1].split("-")[0]
    return (config, split)


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
        f = REPO_ROOT / "artifacts/stages/stage-1" / asset / "items.jsonl"
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
    #: SORTED, and the keys below are PAIRS, because this screen was
    #: hash-seed dependent and therefore unregenerable.
    #:
    #: `reserved` is a set and `t` a frozenset, so `sorted(t, key=df.__getitem__)`
    #: breaks ties in document frequency by the set's ITERATION order -- which
    #: PYTHONHASHSEED decides. Different seeds indexed and queried different
    #: "rarest" tokens, so different candidates were compared at all:
    #: `with_any_neighbour` measured 375 at one seed and 377 at another, and
    #: `test_it_regenerates_identically` could only pass when the ambient seed
    #: happened to match the one in use when the record was last written.
    #:
    #: `(df[w], w)` is a total order over the tokens, and iterating `reserved`
    #: in sorted order makes `tokens` insertion-stable, so the index is built
    #: the same way every time. The screen's VALUES move once, to the ones a
    #: total order produces; it is descriptive and not an acceptance criterion,
    #: and a descriptive number that changes with an environment variable is
    #: not a measurement of anything.
    df: collections.Counter = collections.Counter()
    tokens = {}
    for s in sorted(reserved):
        t = frozenset(s.split())
        tokens[s] = t
        df.update(t)
    index: dict[str, list[str]] = collections.defaultdict(list)
    for s, t in tokens.items():
        for w in sorted(t, key=lambda w: (df[w], w))[:rare_tokens]:
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
        for w in sorted(tq, key=lambda w: (df.get(w, 0), w))[:rare_tokens]:
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
    #: SORTED, twice, and for two distinct reasons -- this list was hash-seed
    #: dependent and a review surface that names a different partner on every
    #: run is not reviewable.
    #:
    #: `keys` is a set, so when two consumed task ids normalise to the SAME
    #: problem text the last writer won and WHICH one won was decided by
    #: PYTHONHASHSEED. And the scan below takes the first strict maximum, so
    #: among equally-similar partners it returned whichever the dict happened
    #: to yield first: `resembles_consumed_task_id` for one candidate came back
    #: as 17 at one seed and 458 at another, naming two different prompts as
    #: "the problem it resembles".
    #:
    #: Building from `sorted(keys)` makes the collision resolution defined, and
    #: scanning `sorted(by_hash.items())` makes the tie go to the
    #: lexicographically first problem text. Neither is more correct as a
    #: similarity judgement -- the point is that there IS an answer, and that a
    #: human ruling recorded against this list still describes it tomorrow.
    by_hash = {norm(pinned[k][field]): k for k in sorted(keys) if k in pinned}
    pairs = []
    for row, q in zip(cand_rows, cand_norm):
        hi, partner = 0.0, None
        tq = frozenset(q.split())
        for text, key in sorted(by_hash.items()):
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
    #: A TOTAL ORDER. `-jaccard` alone leaves ties in the input order, which is
    #: `cand_rows` and is deterministic today -- but the sort key is where a
    #: future unordered input would surface as a reordered record, so the tie
    #: is broken on the candidate's own id rather than on how it arrived.
    pairs.sort(key=lambda d: (-d["jaccard"], d["candidate_task_id"]))
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
    out["current_exact_chain_survivors"] = {
        "count": len(full_survivors),
        "_what_this_is_NOT": (
            "NOT an eligible count and NOT final D-series admissibility. These "
            "are survivors under the contract AS IT EXISTS TODAY -- the "
            "historical chain plus the D-series pool/calibration isolation -- "
            "and that contract is measurably blind to recovery-training problem "
            "CONTENT. The word `eligible` is reserved for "
            "`strengthened_contract` below, which adds the problem-content "
            "coordinate."),
    }

    strengthened = strengthened_survivors(
        group, rows, base_ids, base_hashes, add_ids, add_hashes)
    out["strengthened_contract"] = {
        "_what": ("the current chain PLUS canonical problem-content identity "
                  "and the frozen duplicate-review decision. The historical "
                  "chain is unchanged -- this is a D-series layer on top, so "
                  "frozen C1/C2/C3 battery membership stays reproducible."),
        **strengthened,
    }
    out["eligible_rows"] = {
        "count": strengthened["survivors"],
        "_this_is_the_eligible_count": (
            "the output of the complete strengthened chain. The only number in "
            "this stratum that may be called eligible without qualification."),
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
            #: A REFERENCE, not a copy. This block previously carried its own
            #: numeric `eligible_rows`, which went stale the moment the
            #: strengthened contract changed the authoritative count -- it read
            #: 464/7043 against the real 463/5433, with prose saying the
            #: problem-content key was still owed. Two machine-readable values
            #: called eligible is one too many.
            "eligible_ref": "eligible_rows.count",
            "headroom": out["eligible_rows"]["count"] - short,
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
    strata["math_verified"] = math_stratum()
    blockers = {g: v["renderer"]["BLOCKER"] for g, v in strata.items()
                if isinstance(v.get("renderer"), dict) and "BLOCKER" in v["renderer"]}
    return {
        "schema": SCHEMA,
        "_contract": (
            "Evidence for the maintainer source decision. AUTHORIZES NOTHING "
            "and MATERIALIZES NOTHING -- no battery is built and no row is "
            "admitted anywhere by it. Counts are reported at four named levels "
            "and `eligible` means ONE of them: the output of the STRENGTHENED "
            "chain, which is the historical contract plus canonical "
            "problem-content identity. Survivors of the contract as it exists "
            "today are reported as `current_exact_chain_survivors` and are NOT "
            "eligible counts, because that contract is measurably blind to "
            "recovery-training problem content."),
        "status": ("EVIDENCE ONLY. MBPP and GSM8K use already-pinned "
                   "revisions; the canonical MATH test population is NOW "
                   "PINNED AND FETCHED. NO BATTERY MATERIALIZED."),
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
                 "upstream rows, current-contract survivors and eligible rows "
                 "are three different numbers and the record reports all three. "
                 "`eligible_rows.count` is the strengthened chain's output and "
                 "the only one that answers the capacity question.")},
            {"trap": "reading GSM8K's eligible count as rendered membership",
             "why_not": (
                 "it was derived with the HISTORICAL positional renderer "
                 "participating in the chain. The D-series renderer uses a "
                 "split-aware identity, so the positional false collisions "
                 "disappear and the realized pool may differ. 5,433 is "
                 "conservative capacity evidence; realized membership is frozen "
                 "when the renderer exists. A change there needs no new source "
                 "decision.")},
        ],
        "source_decision": {
            "_status": (
                "DECIDED by the maintainer on 2026-10-03 and IMPLEMENTED. This "
                "is current state, not a proposal: the proposal language and the "
                "reasoning that led to it live in logs/budget/decisions.md. It "
                "still authorizes no battery and admits no row."),
            "code": {
                "source": ("google-research-datasets/mbpp, the already-pinned "
                           "revision, full/train + full/validation + "
                           "full/prompt"),
                "second_dataset": "NOT introduced; this source is sufficient",
                "identity": ("native task_id as provenance, split-aware D-series "
                             "item id, canonical problem-content id"),
            },
            "gsm8k": {
                "source": ("openai/gsm8k, the already-pinned revision, "
                           "main/train"),
                "identity": ("split-aware D-series item id "
                             "(gsm8k-main-train-NNNNN), canonical "
                             "problem-content id as a SEPARATE coordinate"),
                "socratic": "NOT used as capacity",
                "REMAINING_BLOCKER": (
                    "the split-aware RENDERER does not exist yet. The historical "
                    "positional renderer may not be reused. Capacity is "
                    "established; rendered membership is not."),
            },
            "math_verified": {
                "source": ("the canonical Hendrycks MATH TEST population, "
                           "PINNED AND FETCHED -- see `strata.math_verified.pin`"),
                "train_split": "NOT used for this behavioural stratum",
                "adapter": ("problem->problem, type->subject, level 'Level N'->N, "
                            "solution->gold via boxed_answer; parity verified "
                            "against the frozen stratum"),
                "scope": ("a NEW behavioural population; historical C1/C2/C3/A3 "
                          "scores are not imported or compared"),
            },
            "duplicate_policy": {
                "added": "the exact canonical problem-content identity",
                "not_added": ("any automatic similarity cutoff. Fuzzy similarity "
                              "remains a pre-freeze review surface."),
                "frozen_review": ("MBPP 602 excluded as an exact duplicate of "
                                  "consumed 217; the other 43 reviewed pairs "
                                  "explicitly retained. Frozen before any D1 "
                                  "outcome. Owner: identity.py"),
            },
        },
        "still_owed_before_any_materialization": [
            "the GSM8K split-aware RENDERER, and renderer/scorer parity for it. "
            "Until it exists the 5,433 figure is conservative capacity evidence "
            "and not final rendered membership",
            "the explicit maintainer MATERIALIZATION decision for the six-role "
            "family",
        ],
        "_what_is_no_longer_owed": (
            "the source decision (made and implemented 2026-10-03), the MATH pin "
            "and fetch (done -- immutable revision, per-file SHA256, test only), "
            "the canonical problem-content key (implemented in identity.py and "
            "load-bearing in every count above), the GSM8K training-content "
            "repair (0 recovery-training rows survive), and the duplicate review "
            "(frozen before any outcome). Each was listed here while open; "
            "leaving them would make the record describe a state it is no longer "
            "in."),
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
        #: `candidate_total` for an already-pinned repo's extra files; `total`
        #: for a newly pinned source whose whole test population is the candidate
        upstream = rows.get("candidate_total", rows.get("total"))
        print(f"      upstream rows                  : {upstream}")
        print(f"      baseline-chain survivors       : {base['survivors']}")
        print(f"      D-series isolation survivors   : {add['survivors']}"
              f"   (-{add['removed_beyond_baseline']} beyond baseline)")
        print(f"      current-contract survivors     : "
              f"{s['current_exact_chain_survivors']['count']}")
        st = s.get("strengthened_contract", {})
        print(f"      ELIGIBLE (strengthened)        : "
              f"{s['eligible_rows']['count']}"
              f"   (-{st.get('removed_by_problem_content', 0)} by problem "
              f"content, -{st.get('removed_by_frozen_review', 0)} by review)")
        gap = s.get("training_corpus_content_gap", {})
        missed = gap.get("IN_THE_TRAINING_CORPUS_BUT_NOT_CAUGHT")
        if missed:
            print(f"      !! IN TRAINING CORPUS, UNCAUGHT : {missed}"
                  f"   ({gap['_verdict']})")
        scr = s.get("near_duplicate_screen")
        if scr:
            print(f"      screen, rendered >=0.8 / >=0.7 : "
                  f"{scr['at_or_above']['0.8']} / {scr['at_or_above']['0.7']}"
                  f"   (highest {scr['highest']})")
        bare = s.get("bare_problem_screen")
        if bare:
            print(f"      screen, PROBLEM  >=0.8 / >=0.7 : "
                  f"{bare['at_or_above']['0.8']} / {bare['at_or_above']['0.7']}"
                  f"   (highest {bare['highest']}, exact {bare['exact_matches']})")
        if "BLOCKER" in s.get("renderer", {}):
            print(f"      RENDERER BLOCKER               : "
                  f"{s['renderer']['id_collisions_with_the_pinned_file']} id collisions")
    print(f"\n  renderer blockers : "
          f"{list(doc['renderer_blockers']) if isinstance(doc['renderer_blockers'], dict) else doc['renderer_blockers']}")
    print("\n  `eligible` above = survivors of the STRENGTHENED chain "
          "(historical contract + problem-content identity).")
    print("  NO BATTERY MATERIALIZED. NOTHING ADMITTED. AUTHORIZES NOTHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
