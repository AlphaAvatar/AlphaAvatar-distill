"""Dataset-role isolation.

The AutoInitializer consumes data at four points, and they must not be the same
data:

1. ``OPERATOR_CALIBRATION`` — what an operator measures its own decision on
   (E8a's 67-item mixture).
2. ``STATE_EVALUATION`` — the frozen suite every produced checkpoint is scored on
   for beam ranking.
3. ``RECOVERY_BATTERY`` — what the fixed low-budget recovery probes are selected
   on.
4. ``FINAL_PROMOTION`` — the frozen 150-prompt battery, and the 846-prompt
   capability battery it is drawn from. **Isolated from the entire search.**

The failure this module exists to prevent is quiet: nothing crashes when a
calibration mixture happens to contain three promotion prompts. The search simply
selects, a little, for the thing it will later be graded on, and the final number
is no longer an out-of-sample number. This project has already had a near-miss
here: its leakage proof caught two real collisions.

So the check is content-based and fail-closed. Prompt *content* hashes are
compared, not ids: the same question arriving through two dataset paths under two
ids is exactly the leak, and an id comparison cannot see it. The convention
matches ``scripts/data/check_e8_calibration_leakage.py`` so a role check and the
existing leakage proof mean the same thing by the same rule.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from aadistill.data.extra_stream import content_sha256


class DatasetRole(Enum):
    OPERATOR_CALIBRATION = "operator_calibration"
    STATE_EVALUATION = "state_evaluation"
    RECOVERY_BATTERY = "recovery_battery"
    FINAL_PROMOTION = "final_promotion"


#: Roles the search loop is allowed to read at all. `FINAL_PROMOTION` is absent
#: by construction, which is the mechanical form of "the final battery is
#: isolated from the search".
SEARCH_VISIBLE_ROLES = frozenset({
    DatasetRole.OPERATOR_CALIBRATION,
    DatasetRole.STATE_EVALUATION,
})


class DatasetRoleViolation(RuntimeError):
    """An asset was used outside its declared role, or two roles overlap."""


def prompt_text(item: Mapping[str, Any]) -> str | None:
    """Everything the model is shown, excluding any assistant output.

    Accepts a chat-style ``{"messages": [...]}`` record or a flat record carrying
    ``prompt``/``question``/``text``. Returns ``None`` when the item carries no
    raw text at all — a pre-tokenized mixture, for instance — so the caller can
    fall back to another identity rather than hashing an empty string, which
    would make every unreadable item collide with every other one.
    """
    if "messages" in item:
        parts = [str(m.get("content", "")) for m in item["messages"]
                 if m.get("role") != "assistant"]
        return "\n".join(parts)
    for key in ("prompt", "question", "text", "input"):
        if key in item:
            return str(item[key])
    return None


def item_identities(item: Mapping[str, Any]) -> dict[str, str]:
    """Every comparable identity an item carries, by kind.

    Assets are stored in different forms. E8a's calibration mixture is
    pre-tokenized — ``item_id``, ``ids``, ``doc_sha256``, no raw text — while a
    battery is prompts. Reducing both to "the prompt hash" is not possible, and
    silently comparing a set of text hashes against a set of token hashes returns
    "no overlap" for every input, which is the worst possible failure for a check
    like this: it always passes.

    So identities are typed, and ``check_role_isolation`` compares only within a
    kind and *reports* role pairs that share no kind at all.
    """
    out: dict[str, str] = {}
    text = prompt_text(item)
    if text is not None:
        out["prompt_content"] = content_sha256(text)
    for key in ("doc_sha256", "prompt_sha256", "candidate_sha256", "source_id"):
        if item.get(key):
            out[key] = str(item[key])
    ids = item.get("ids") or item.get("input_ids")
    if ids is not None:
        out["token_ids"] = content_sha256(",".join(map(str, ids)))
    if not out:
        raise DatasetRoleViolation(
            f"cannot derive any comparable identity from an item with keys "
            f"{sorted(item)}; a role check that cannot read an item must fail, "
            "not skip it")
    return out


@dataclass(frozen=True)
class DatasetAsset:
    """A data artifact bound to exactly one role."""

    asset_id: str
    role: DatasetRole
    path: str | None
    description: str
    n_items: int | None = None
    content_sha256: str | None = None
    protected: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "role": self.role.value,
            "path": self.path,
            "description": self.description,
            "n_items": self.n_items,
            "content_sha256": self.content_sha256,
            "protected": self.protected,
            "metadata": dict(self.metadata),
        }

    def load_items(self, repo_root: str | Path = ".") -> list[dict[str, Any]]:
        if not self.path:
            raise DatasetRoleViolation(f"{self.asset_id} has no path to load")
        p = Path(repo_root) / self.path
        if not p.is_file():
            raise DatasetRoleViolation(f"{self.asset_id}: {p} is missing")
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def identity_sets(self, repo_root: str | Path = ".") -> dict[str, set[str]]:
        """Comparable identities this asset carries, grouped by kind."""
        out: dict[str, set[str]] = {}
        for item in self.load_items(repo_root):
            for kind, value in item_identities(item).items():
                out.setdefault(kind, set()).add(value)
        return out

    def prompt_hashes(self, repo_root: str | Path = ".") -> set[str]:
        """Prompt-content hashes, when the asset stores raw text."""
        return self.identity_sets(repo_root).get("prompt_content", set())


_ASSETS: dict[str, DatasetAsset] = {}


def register_asset(asset: DatasetAsset, *, replace: bool = False) -> DatasetAsset:
    existing = _ASSETS.get(asset.asset_id)
    if existing is not None and not replace:
        if existing != asset:
            raise DatasetRoleViolation(
                f"{asset.asset_id} is already registered in role {existing.role.value!r}; "
                "re-registering it would move an asset between roles silently")
        return existing
    _ASSETS[asset.asset_id] = asset
    return asset


def get_asset(asset_id: str) -> DatasetAsset:
    if asset_id not in _ASSETS:
        raise KeyError(f"no dataset asset {asset_id!r}; registered: {sorted(_ASSETS)}")
    return _ASSETS[asset_id]


def registered_assets() -> list[DatasetAsset]:
    return [_ASSETS[k] for k in sorted(_ASSETS)]


def unregister_asset(asset_id: str) -> None:
    """Test-only."""
    _ASSETS.pop(asset_id, None)


def assert_usable_for(asset_id: str, role: DatasetRole) -> DatasetAsset:
    """Fail unless ``asset_id`` was declared for exactly ``role``."""
    asset = get_asset(asset_id)
    if asset.role is not role:
        raise DatasetRoleViolation(
            f"{asset_id} is declared for {asset.role.value!r} and cannot be used as "
            f"{role.value!r}")
    return asset


def assert_search_visible(asset_id: str) -> DatasetAsset:
    """Fail unless the search loop may read this asset at all."""
    asset = get_asset(asset_id)
    if asset.role not in SEARCH_VISIBLE_ROLES:
        raise DatasetRoleViolation(
            f"{asset_id} is a {asset.role.value!r} asset; the search loop may not "
            "read it. Selecting on the data a result is later reported on is not "
            "an out-of-sample result")
    return asset


def check_role_isolation(
    asset_ids: Sequence[str] | None = None,
    *,
    repo_root: str | Path = ".",
    require_loadable: bool = True,
) -> dict[str, Any]:
    """Prove that no prompt appears under two roles.

    Returns a report and raises ``DatasetRoleViolation`` on any overlap. Assets
    whose files are absent are reported as ``unloadable``; with
    ``require_loadable`` that is itself a failure, because a check that silently
    skips the asset it could not open proves nothing.
    """
    assets = [get_asset(a) for a in asset_ids] if asset_ids else registered_assets()
    by_role: dict[DatasetRole, dict[str, set[str]]] = {}
    unloadable: list[dict[str, str]] = []
    counted: list[dict[str, Any]] = []

    for asset in assets:
        try:
            identities = asset.identity_sets(repo_root)
        except DatasetRoleViolation as exc:
            unloadable.append({"asset_id": asset.asset_id, "reason": str(exc)})
            continue
        role_map = by_role.setdefault(asset.role, {})
        for kind, values in identities.items():
            role_map.setdefault(kind, set()).update(values)
        counted.append({"asset_id": asset.asset_id, "role": asset.role.value,
                        "identity_kinds": {k: len(v) for k, v in sorted(identities.items())}})

    overlaps, uncomparable = [], []
    roles = sorted(by_role, key=lambda r: r.value)
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            shared_kinds = sorted(set(by_role[a]) & set(by_role[b]))
            if not shared_kinds:
                uncomparable.append({
                    "role_a": a.value, "role_b": b.value,
                    "kinds_a": sorted(by_role[a]), "kinds_b": sorted(by_role[b]),
                    "reason": ("the two roles store no identity kind in common, so "
                               "no overlap could have been detected either way"),
                })
                continue
            for kind in shared_kinds:
                shared = by_role[a][kind] & by_role[b][kind]
                if shared:
                    overlaps.append({
                        "role_a": a.value, "role_b": b.value, "identity_kind": kind,
                        "n_shared": len(shared), "examples": sorted(shared)[:5],
                    })

    report = {
        "checked_assets": counted,
        "unloadable": unloadable,
        "overlaps": overlaps,
        "uncomparable_role_pairs": uncomparable,
        "roles_present": [r.value for r in roles],
        # `complete` and `passed` are separate on purpose: "no overlap among what
        # I could compare" and "I compared everything" are different claims, and a
        # report that merged them would let an unreadable or incomparable asset
        # read as clean.
        "complete": not unloadable and not uncomparable,
        "passed": not overlaps and not (unloadable and require_loadable),
    }
    if overlaps:
        raise DatasetRoleViolation(
            f"dataset roles overlap: {overlaps}. A prompt used both to steer the "
            "search and to grade it makes the final number in-sample.")
    if unloadable and require_loadable:
        raise DatasetRoleViolation(
            f"could not load {[u['asset_id'] for u in unloadable]}; a role check "
            "that skips an asset proves nothing about it")
    if uncomparable and require_loadable:
        raise DatasetRoleViolation(
            f"role pairs {[(u['role_a'], u['role_b']) for u in uncomparable]} share "
            "no identity kind, so this check could not have found a leak between "
            "them. Render one side into the other's form before relying on it.")
    return report


# --- loading concrete assets ------------------------------------------------
#
# The three assets this project froze used to be written out here and registered
# at import. That put experiment DATA -- prompt counts, artifact paths, content
# hashes, leakage proofs -- inside the reusable core, and made merely importing
# this module change global state.
#
# They now live in `configs/datasets/assets.json`, loaded by
# `scripts/experiments/datasets.py`. This module keeps only the mechanism.

ASSET_DOCUMENT_SCHEMA = "aadistill.dataset_assets/v1"


def asset_from_dict(doc: Mapping[str, Any]) -> DatasetAsset:
    """One asset from its serialized form. Field names are the dataclass's."""
    fields = dict(doc)
    if fields.get("role") is not None:
        fields["role"] = DatasetRole(fields["role"])
    if fields.get("metadata") is not None:
        fields["metadata"] = dict(fields["metadata"])
    return DatasetAsset(**fields)


def load_assets(document: Mapping[str, Any], *,
                register: bool = True) -> tuple[DatasetAsset, ...]:
    """Build the assets a document declares, and register them by default.

    Registration stays refusable: re-registering an id whose specification
    differs raises, so a document that redefines a frozen asset is an error
    rather than a silent swap.
    """
    if document.get("schema") != ASSET_DOCUMENT_SCHEMA:
        raise DatasetRoleViolation(
            f"expected a {ASSET_DOCUMENT_SCHEMA} document, got "
            f"{document.get('schema')!r}")
    built = tuple(asset_from_dict(d) for d in document["assets"])
    if register:
        built = tuple(register_asset(a) for a in built)
    return built


def protected_assets() -> list[DatasetAsset]:
    return [a for a in registered_assets() if a.protected]


def role_summary() -> list[dict[str, Any]]:
    return [a.as_dict() for a in registered_assets()]
