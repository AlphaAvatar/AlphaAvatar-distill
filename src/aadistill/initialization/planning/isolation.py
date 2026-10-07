"""One expansion must not change what a sibling expansion produces.

THE INVARIANT this module exists to hold:

    same parent artifact
      + same operator
      + same profile
      + same execution protocol
    = same child artifact

**independent of which sibling expansion ran before it.**

WHY IT WAS NOT HELD. A beam caches its root model and hands the SAME object to
every level-0 expansion, because reloading a teacher per expansion is minutes of
GPU time each. An operator that mutates the model it is handed therefore leaves
that mutation visible to every later sibling — and a mutated *config* is not a
transient: `ArchitectureAdapter.build_config` builds a child's config from its
parent's whole `to_dict()`, so a field assigned on the parent reaches the
child's `config.json`, its `config_sha256`, its `artifact_digest`, and every
identity a replay or a resume is pinned to.

The consequence is an identity that depends on **expansion order**. Two
searches with identical science, identical operators and identical profiles
could produce different child digests purely because one enumerated its
operators in a different order, and the second one's records would be
unreproducible by the first one's path. A replay of such a search cannot start
from the published teacher; it has to reconstruct what the root had become, and
reconstructing in-memory state from a journal is possible only because the
mutation happened to be one documented boolean.

WHAT THIS DOES. `isolate_parent_config` snapshots the parent's config before an
operator runs and restores whatever the operator changed afterwards. It does
not prevent the mutation — an operator may legitimately need
`use_cache=False` for its own forwards, and a frozen operator whose historical
identities depend on its behaviour must not be altered to satisfy a later
rule. It contains the mutation to the expansion that made it.

WHAT IT IS NOT. Not a correctness check on the operator, not a validator, and
not a place for a policy. It reports what it restored so a caller can record
it, because a mutation that is silently undone is a mutation nobody knows
about, and the next person to wonder why a digest moved deserves the evidence.

It knows no field names, no model family and no operator. The quantity it
protects is "whatever `config.to_dict()` said before", because that is exactly
the quantity a child inherits.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any


def config_snapshot(model: Any) -> dict[str, Any] | None:
    """What this model's config would contribute to a child, right now.

    `None` when the object has no config, or one that cannot be serialized:
    there is then nothing a child could inherit through it and nothing to
    protect. Returning `None` rather than raising keeps this usable from a
    search that may be handed a bare module in a test.
    """
    config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if config is None or not callable(to_dict):
        return None
    try:
        snapshot = to_dict()
    except Exception:                                             # noqa: BLE001
        return None
    return dict(snapshot) if isinstance(snapshot, Mapping) else None


def restore_config(model: Any, snapshot: Mapping[str, Any] | None
                   ) -> dict[str, dict[str, Any]]:
    """Put back every top-level config field that changed. Returns the diff.

    The diff maps a field name to `{"was": ..., "became": ...}`, in the
    direction the operator moved it, so a record reads as what happened rather
    than as what was undone.

    TOP-LEVEL ONLY, deliberately. The fields an operator assigns are attributes
    of the config object, and that is what `setattr` reaches; a nested value
    that changed without its top-level key changing identity would be a mutation
    *inside* a structure the config exposes, which is a different problem and
    not one this should pretend to solve. A key that appears or disappears is
    reported and the appearing one is not removed: deleting an attribute a
    config added for its own reasons is more dangerous than recording it.
    """
    if snapshot is None:
        return {}
    current = config_snapshot(model)
    if current is None:
        return {}
    config = model.config
    diff: dict[str, dict[str, Any]] = {}
    for key, was in snapshot.items():
        became = current.get(key, _MISSING)
        if became is _MISSING or became == was:
            continue
        diff[key] = {"was": was, "became": became}
        try:
            setattr(config, key, was)
        except Exception as exc:                                  # noqa: BLE001
            diff[key]["restore_failed"] = f"{type(exc).__name__}: {exc}"
    for key in current:
        if key not in snapshot:
            diff[key] = {"was": None, "became": current[key], "added": True}
    return diff


class _Missing:
    def __repr__(self) -> str:                          # pragma: no cover
        return "<missing>"


_MISSING = _Missing()


@contextmanager
def isolate_parent_config(model: Any, record: dict[str, Any] | None = None
                          ) -> Iterator[dict[str, Any]]:
    """Run an expansion, then undo whatever it did to the parent's config.

        with isolate_parent_config(parent_model) as touched:
            outcome = impl.execute(ctx)
        # `touched` now holds the diff, and the parent is as it was

    The yielded dict is filled on exit, so a caller reads it after the block.
    When `record` is given it is the dict that gets filled, which lets a
    telemetry payload be built without a second copy.

    Restoration runs in a `finally`: an operator that raises part-way through
    its own mutations is exactly when a shared parent is most likely to be left
    in a state no later expansion expects.
    """
    touched = record if record is not None else {}
    snapshot = config_snapshot(model)
    try:
        yield touched
    finally:
        touched.update(restore_config(model, snapshot))
