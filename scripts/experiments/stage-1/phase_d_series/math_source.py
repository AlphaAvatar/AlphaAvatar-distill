#!/usr/bin/env python3
"""The canonical Hendrycks MATH test population, pinned, with its adapter.

Authorized for pinning and fetching by the maintainer source decision of
2026-10-03. **TEST splits only** — the train splits are not fetched and not used
for this behavioural stratum.

**Why this needs an adapter at all.** The pinned `math_verified` stratum is
`HuggingFaceH4/MATH-500`, whose rows carry `unique_id`, `subject`, `level`,
`problem`, `answer`. The canonical release carries `problem`, `level`, `type`,
`solution` — no `answer` and no `unique_id`. So the gold must be DERIVED and the
field names MAPPED, and neither may be done in a way that changes what
"correct" means for this stratum.

```text
problem   -> problem        verbatim
type      -> subject        verbatim; the seven config names map onto the seven
                            existing subject values exactly
level     -> level          "Level 3" -> 3, because upstream is a STRING and the
                            existing stratum is an INTEGER
solution  -> gold           boxed_answer(solution)
```

**The derivation is verified against the existing stratum, not asserted.**
`boxed_answer(solution)` reproduces MATH-500's own `answer` field on 500/500
pinned rows, so applying it upstream is the same correctness semantics rather
than a new one arriving with a new source. That check runs in
`source_evidence.math_pinning_readiness` and is the gate on this adapter.

**The `level` mapping is the detail the contract did not name.** Upstream emits
`"Level 3"`; MATH-500 emits `3`. Passing it through would have put a string where
every consumer expects an integer, and a stratification by level would have
silently produced one bucket per string.

**This is a NEW behavioural population.** MATH-500's 500 rows are a measured
subset of these 5,000, so the overlap is real and is excluded by content like
any other reserved material. Historical C1/C2/C3/A3 scores are not comparable to
anything measured on it and are not imported.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/data"):
    path = str(REPO_ROOT / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

#: THE PIN. An immutable revision sha, never a floating branch: `main` would make
#: every measurement below describe whatever the mirror happened to hold.
REPO_ID = "EleutherAI/hendrycks_math"
REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
LICENCE = "mit"

#: The seven configs, each one subject of the canonical MATH test population.
#: TEST only. Config name -> the existing stratum's `subject` value, so the
#: mapping is explicit rather than a title-casing rule that would silently
#: mis-handle `counting_and_probability`.
CONFIG_TO_SUBJECT: dict[str, str] = {
    "algebra": "Algebra",
    "counting_and_probability": "Counting & Probability",
    "geometry": "Geometry",
    "intermediate_algebra": "Intermediate Algebra",
    "number_theory": "Number Theory",
    "prealgebra": "Prealgebra",
    "precalculus": "Precalculus",
}

TEST_FILE = "test-00000-of-00001.parquet"


def test_files() -> tuple[str, ...]:
    return tuple(f"{c}/{TEST_FILE}" for c in sorted(CONFIG_TO_SUBJECT))


def local_path(relpath: str) -> Path:
    """The cached path of one pinned file. Raises if it is not fetched."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, relpath, repo_type="dataset",
                                revision=REVISION, local_files_only=True))


def is_fetched() -> bool:
    try:
        return all(local_path(r).is_file() for r in test_files())
    except Exception:
        return False


def file_manifest() -> dict[str, Any]:
    """Revision, every file, and each file's SHA256 — the provenance record."""
    files = {}
    for rel in test_files():
        try:
            p = local_path(rel)
        except Exception as exc:
            files[rel] = {"status": f"NOT FETCHED ({type(exc).__name__})"}
            continue
        raw = p.read_bytes()
        files[rel] = {"size_bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest()}
    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "_revision_is_immutable": (
            "a commit sha, not a branch. `main` would make every count below "
            "describe whatever the mirror held at read time."),
        "licence": LICENCE,
        "splits_fetched": "TEST ONLY -- no train file is fetched or used",
        "configs": dict(sorted(CONFIG_TO_SUBJECT.items())),
        "files": files,
    }


def rows() -> list[dict[str, Any]]:
    """Every pinned test row, adapted to the existing stratum's field names."""
    import pandas as pd

    out: list[dict[str, Any]] = []
    for config in sorted(CONFIG_TO_SUBJECT):
        frame = pd.read_parquet(local_path(f"{config}/{TEST_FILE}"))
        for index, raw in enumerate(frame.to_dict("records")):
            out.append(adapt(raw, config=config, index=index))
    return out


def level_to_int(level: Any) -> int:
    """`"Level 3"` -> `3`. Upstream is a string; the existing stratum is an int.

    Refuses anything it cannot read rather than defaulting, because a silent 0
    would collapse a stratification into one bucket.
    """
    text = str(level).strip()
    digits = "".join(c for c in text if c.isdigit())
    if not digits:
        raise ValueError(
            f"cannot read a level from {level!r}; upstream emits 'Level N' and "
            "the existing stratum stores N")
    return int(digits)


def adapt(raw: dict[str, Any], *, config: str, index: int) -> dict[str, Any]:
    """One upstream row in the existing stratum's shape, plus its provenance.

    The gold is DERIVED by the verified rule. A row whose solution carries no
    boxed answer is refused rather than admitted with an empty gold: it would
    score as wrong for every model forever.
    """
    from aadistill.data.verify import boxed_answer

    gold = boxed_answer(raw["solution"])
    if gold is None:
        raise ValueError(
            f"{config}[{index}]: no boxed answer in the solution, so no gold can "
            "be derived. Admitting it would create an item nothing can answer "
            "correctly.")
    subject = CONFIG_TO_SUBJECT[config]
    if str(raw.get("type")) != subject:
        raise ValueError(
            f"{config}[{index}]: upstream `type` is {raw.get('type')!r} but the "
            f"config maps to {subject!r}. The mapping and the data disagree, "
            "which is a source change rather than a rendering detail.")
    return {
        "problem": raw["problem"],
        "subject": subject,
        "level": level_to_int(raw["level"]),
        "answer": gold,
        "solution": raw["solution"],
        #: provenance, kept separate from content by construction
        "_config": config,
        "_split": "test",
        "_index": index,
        "_upstream_level": str(raw["level"]),
    }
