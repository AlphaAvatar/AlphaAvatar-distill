"""Round 3 of the information architecture: the permanent evaluation registry.

Maintainer decision 2026-10-18: `data/eval-set/` is the authoritative,
cross-stage home of permanent evaluation sets, starting with `eval-set-1`.
Two tracked evaluation files that lived under a stage's data tree move into it,
and the frozen capability battery that lived only as a gitignored artifact is
copied in. This module is the single declaration of those relocations; the
builder (`scripts/shared/data/build_eval_set_1.py`) and the historical path
map (`logs/index.json :: historical_paths.map`) read the pairs from here.

Doctrine, unchanged from rounds 1 and 2:

* **Frozen records keep their freeze-time spellings.** A run's evidence, a
  consumed authorization, a closed session plan or a frozen manifest that
  names `data/stages/stage-0/calibration_volume/tasks/dev_v1.jsonl` keeps
  saying so; ACCESS resolves through the historical map.
* **Live readers move with the file.** Current loaders, builders, tests and
  defaults name the new address.
* **Content is never rewritten by a move.** Every pair relocates bytes
  verbatim; the builder verifies sha256 before and after.

Ownership calls:

* `eval_behavior_v0` was stage-3 material because stage 3 built and used it.
  The maintainer has now designated it part of eval-set-1's
  `stage3_capability_846` component, so its owner is the registry; the leaf
  directory name is kept because manifests and run records spell it.
* `battery_v2` stays where it is as a durable artifact (frozen records and
  the D-series name it there) and is COPIED, not moved: deleting an artifact is
  a maintainer decision (AGENTS.md P12), and the two copies are proven
  byte-identical by sha256. The registry names the eval-set copy canonical.
  `knowledge.jsonl` is not copied (license basis unknown; see the eval-set
  NOTICE).
"""
from __future__ import annotations

ES1 = "data/eval-set/eval-set-1"

#: Whole directories, verbatim (old prefix -> new prefix).
DATA_DIR_MOVES: tuple[tuple[str, str], ...] = (
    ("data/stages/stage-3/eval_behavior_v0", f"{ES1}/stage3_capability_846/eval_behavior_v0"),
)

#: Single files, basenames preserved.
DATA_FILE_MOVES: tuple[tuple[str, str], ...] = (
    ("data/stages/stage-0/calibration_volume/tasks/dev_v1.jsonl",
     f"{ES1}/fluid_reasoning_720/dev_v1.jsonl"),
    ("data/stages/stage-0/calibration_volume/tasks/dev_v1.manifest.json",
     f"{ES1}/fluid_reasoning_720/dev_v1.manifest.json"),
)

#: Copies out of the durable artifact store (source stays; bytes identical).
BATTERY_SOURCE = "artifacts/stages/stage-3/eval/battery_v2"
BATTERY_DEST = f"{ES1}/stage3_capability_846/battery_v2"
BATTERY_COPIED = ("manifest.json", "math_verified.jsonl", "gsm8k.jsonl", "multihop.jsonl",
                  "rag.jsonl", "answerability_paired.jsonl", "safety_paired.jsonl")
BATTERY_WITHHELD = ("knowledge.jsonl",)

#: Historical-map pairs this round adds or RETARGETS. The map holds "where the
#: object is now" and the resolver does not chain, so the round-2 key for the
#: pre-2026-10-08 spelling is retargeted rather than left pointing at an
#: address that no longer exists.
HISTORICAL_PAIRS: tuple[tuple[str, str], ...] = (
    *DATA_DIR_MOVES,
    *DATA_FILE_MOVES,
    ("data/eval_behavior_v0", f"{ES1}/stage3_capability_846/eval_behavior_v0"),
)


def all_pairs() -> list[tuple[str, str]]:
    return list(HISTORICAL_PAIRS)


if __name__ == "__main__":
    for a, b in all_pairs():
        print(f"{a} -> {b}")
