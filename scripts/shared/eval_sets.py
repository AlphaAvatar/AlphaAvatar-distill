"""The project's permanent evaluation registry (`data/eval-set/`), resolved.

    from shared import eval_sets as ES
    es = ES.default_formal()                    # "eval-set-1", hash-verified
    rows = ES.items(es, "fluid_reasoning_720", "dev_v1")
    report = ES.overlap_report(training_items)  # vs every held-out component
    ES.require_disjoint(training_items)         # refuses a held-out overlap

Formal capability measurement in any stage reads the registered set by default
(AGENTS.md P10.3). The generic mechanism is `aadistill.evaluation.eval_registry`,
which knows no set and no path; this module only knows where this project keeps
its registry and how its components' items are keyed. Every read goes through
`require_intact`, so a modified set is refused before anything is scored on it.

    PYTHONPATH=src:scripts python -m shared.eval_sets      # verify every set
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.evaluation import eval_registry as ER  # noqa: E402
from aadistill.evaluation.isolation import key_overlaps  # noqa: E402
from aadistill.evaluation.isolation import require_disjoint as iso_require_disjoint  # noqa: E402

REGISTRY = "data/eval-set/registry.json"

#: Component roles whose items must never appear in training, calibration or
#: search data. An in-sample regression component is, by definition, made of
#: training prompts; overlap with it is reported, never refused.
HELD_OUT_ROLES = ("held_out", "development_exposed_held_out")
IN_SAMPLE_ROLES = ("in_sample_regression",)


def registry_path(repo_root: str | Path = REPO_ROOT) -> Path:
    return Path(repo_root) / REGISTRY


def registry(repo_root: str | Path = REPO_ROOT) -> dict:
    return ER.load_registry(registry_path(repo_root))


def default_formal(repo_root: str | Path = REPO_ROOT) -> str:
    """The eval set formal capability measurement reads unless a protocol says otherwise."""
    return registry(repo_root)["default_formal_measurement"]


def manifest(eval_set_id: str | None = None, repo_root: str | Path = REPO_ROOT) -> dict:
    """The set's manifest, after its bytes are proven to match the authorized identity."""
    eval_set_id = eval_set_id or default_formal(repo_root)
    return ER.require_intact(registry_path(repo_root), eval_set_id)


def set_root(eval_set_id: str, repo_root: str | Path = REPO_ROOT) -> Path:
    e = ER.entry(registry(repo_root), eval_set_id)
    return registry_path(repo_root).parent / e["root"]


def unit_path(eval_set_id: str, component: str, unit: str,
              repo_root: str | Path = REPO_ROOT, *, verify: bool = True) -> Path | None:
    """Where a unit's items are. A withheld unit resolves to its out-of-repo copy
    only when that copy still has the recorded identity; otherwise None."""
    m = manifest(eval_set_id, repo_root) if verify else json.loads(
        (set_root(eval_set_id, repo_root) / "manifest.json").read_text())
    spec = m["components"][component]["units"][unit]
    if spec.get("withheld"):
        ident = m["withheld"][spec["file"]]
        local = Path(repo_root) / ident["local_copy"]
        return local if ER.withheld_intact(local, ident) else None
    return set_root(eval_set_id, repo_root) / spec["file"]


def items(eval_set_id: str, component: str, unit: str,
          repo_root: str | Path = REPO_ROOT) -> list[dict]:
    path = unit_path(eval_set_id, component, unit, repo_root)
    if path is None:
        raise FileNotFoundError(f"{eval_set_id}/{component}/{unit} is withheld and its "
                                "recorded local copy is absent or altered")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def prompt_text(item: Mapping) -> str:
    """The prompt a model is shown, minus any system message: every non-system,
    non-assistant message's text, in order. One spelling for every component."""
    msgs = item.get("messages") or item.get("prompt_messages") or []
    parts = [m.get("content") or "" for m in msgs if m.get("role") not in ("system", "assistant")]
    if not parts and item.get("prompt_text"):
        parts = [item["prompt_text"]]
    return "\n".join(parts)


def prompt_key(text: str) -> str:
    """Case- and whitespace-insensitive identity of a prompt's text."""
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


#: An interactive episode's opening message shows only its start state, so two
#: episodes with different hidden rules may open with the same words; their
#: identity is the content hash (which includes the hidden environment).
KEYS: dict[str, Callable[[Mapping], str | None]] = {
    "prompt": lambda r: (None if r.get("interactive") or not prompt_text(r).strip()
                         else prompt_key(prompt_text(r))),
    "content": lambda r: r.get("content_sha256"),
}


def component_items(eval_set_id: str | None = None, *, roles: Iterable[str] | None = None,
                    repo_root: str | Path = REPO_ROOT) -> dict[str, list[dict]]:
    """Every readable unit's items, keyed `component/unit`, optionally by role."""
    eval_set_id = eval_set_id or default_formal(repo_root)
    m = manifest(eval_set_id, repo_root)
    want = set(roles) if roles is not None else None
    out: dict[str, list[dict]] = {}
    for cname, comp in m["components"].items():
        if want is not None and comp["role"] not in want:
            continue
        for uname in comp["units"]:
            try:
                out[f"{cname}/{uname}"] = items(eval_set_id, cname, uname, repo_root)
            except FileNotFoundError:
                continue
    return out


def overlap_report(candidate: Iterable[Mapping], eval_set_id: str | None = None, *,
                   repo_root: str | Path = REPO_ROOT, keys=None) -> dict:
    """Key overlaps between candidate data (training, calibration, search) and
    every readable unit of the set, held-out and in-sample alike."""
    roles = {"candidate": list(candidate),
             **component_items(eval_set_id, repo_root=repo_root)}
    return key_overlaps(roles, keys or KEYS)


def require_disjoint(candidate: Iterable[Mapping], eval_set_id: str | None = None, *,
                     repo_root: str | Path = REPO_ROOT, keys=None) -> dict:
    """Refuse candidate data that shares a key with any HELD-OUT unit."""
    eval_set_id = eval_set_id or default_formal(repo_root)
    keys = keys or KEYS
    held = component_items(eval_set_id, roles=HELD_OUT_ROLES, repo_root=repo_root)
    report = key_overlaps({"candidate": list(candidate), **held}, keys)
    report["checked"] = iso_require_disjoint(
        report, [("candidate", unit, k) for unit in held for k in keys])
    return report


def held_out_paths(eval_set_id: str | None = None,
                   repo_root: str | Path = REPO_ROOT) -> list[str]:
    """Repository-relative files of every in-repository HELD-OUT unit: what a
    training, calibration or search data builder must reserve."""
    eval_set_id = eval_set_id or default_formal(repo_root)
    m = manifest(eval_set_id, repo_root)
    root = set_root(eval_set_id, repo_root)
    return sorted(str((root / spec["file"]).relative_to(Path(repo_root)))
                  for comp in m["components"].values() if comp["role"] in HELD_OUT_ROLES
                  for spec in comp["units"].values() if not spec.get("withheld"))


def registered_files(schema: str, repo_root: str | Path = REPO_ROOT) -> list[Path]:
    """Content files of every registered set whose sidecar manifest has `schema`
    (how a producer finds the registered sets it must stay isolated from)."""
    out = []
    reg = registry(repo_root)
    for e in reg["eval_sets"]:
        manifest(e["eval_set_id"], repo_root)          # refuses a modified set
        root = registry_path(repo_root).parent / e["root"]
        for side in sorted(root.rglob("*.manifest.json")):
            try:
                if json.loads(side.read_text()).get("schema") == schema:
                    out.append(side.with_name(side.name.replace(".manifest.json", ".jsonl")))
            except (OSError, ValueError):
                continue
    return out


def main() -> int:
    reg = registry()
    bad = 0
    for e in reg["eval_sets"]:
        problems = ER.verify(registry_path(), e["eval_set_id"])
        print(f"{e['eval_set_id']:<14} {e['status']:<10} "
              f"{'INTACT' if not problems else 'MODIFIED'}")
        for p in problems:
            print(f"    {p}")
        bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
