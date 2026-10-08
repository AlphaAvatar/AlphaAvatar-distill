"""Where Phase C2's frozen assets are, resolved from the document that declares them.

`configs/experiments/phase_c2/frozen_assets.json` is the expectation the pod's
setup verifies before any measurement, and it already states the one asset C2
consumes and the root it must live at. This module reads that declaration so a
consumer does not restate the path.

**Why this exists rather than another constant.** The literal
`artifacts/stage1/state_eval_v1` appears in seven pod scripts, each of which
predates the expectation document. The baseline-completion path does not add an
eighth: it asks the declaration. The older copies are left alone deliberately --
they belong to consumed or closed experiments whose executable identity is
recorded in grants that have already been spent, and retargeting them would move
those digests for no scientific gain.

The baseline-completion driver reached here after passing the REPOSITORY ROOT to
`load_state_eval.load`, which reads `root/manifest.json` and `root/items.jsonl`.
The repository root has neither, so the suite would have failed to load on the
pod, after setup, inside the one stage that spends money.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: The declaration. Named ONCE, here.
EXPECTATION_DOCUMENT = "configs/experiments/phase_c2/frozen_assets.json"

#: The asset C2 consumes. There is exactly one.
STATE_EVAL_ASSET = "state_eval_v1"


class FrozenAssetError(RuntimeError):
    """The frozen-asset declaration does not say what a consumer needs."""


def expectation(repo_root: str | Path = ".") -> dict[str, Any]:
    path = Path(repo_root) / EXPECTATION_DOCUMENT
    if not path.is_file():
        raise FrozenAssetError(f"{EXPECTATION_DOCUMENT} is absent from {repo_root}")
    return json.loads(path.read_text())


def asset(name: str, repo_root: str | Path = ".") -> dict[str, Any]:
    """One declared asset's entry, or a refusal naming what is declared."""
    assets = expectation(repo_root).get("assets") or {}
    if name not in assets:
        raise FrozenAssetError(
            f"{EXPECTATION_DOCUMENT} declares {sorted(assets)} and not {name!r}")
    return assets[name]


def state_eval_root(repo_root: str | Path = ".") -> Path:
    """The absolute root of the frozen `state_eval_v1` suite.

    `load_state_eval.load` reads `manifest.json` and `items.jsonl` directly
    beneath this, which is checked here: a root that resolves but holds neither
    is a staging failure, and discovering it inside the measuring stage is the
    expensive way to learn it.
    """
    entry = asset(STATE_EVAL_ASSET, repo_root)
    declared = entry.get("root")
    if not declared:
        raise FrozenAssetError(
            f"{EXPECTATION_DOCUMENT} declares {STATE_EVAL_ASSET} with no root")
    root = Path(repo_root) / declared
    missing = [name for name in ("manifest.json", "items.jsonl")
               if not (root / name).is_file()]
    if missing:
        raise FrozenAssetError(
            f"{STATE_EVAL_ASSET} is declared at {declared} and {missing} are not "
            f"there. `load_state_eval.load` reads exactly those two files, so the "
            "suite cannot be loaded from this root.")
    return root


def declared_identities(repo_root: str | Path = ".") -> dict[str, str]:
    """The suite's pinned hashes, for a record that wants to cite them."""
    entry = asset(STATE_EVAL_ASSET, repo_root)
    return {key: entry[key] for key in
            ("content_sha256", "manifest_sha256", "items_sha256") if key in entry}
