"""Battery identity for the C2 screening rung.

C1's scorer refuses any battery but `c1_confirmation_v1`, and that refusal is
deliberate: "The C1 pins stay on `main()`, so the production path cannot be
aimed anywhere else." The screening rung needs the same *metric semantics* on a
different *battery*, and the way to get that is another pinned entry point — not
a `--battery` escape hatch through C1's, which would weaken a guard protecting
the rung that actually names an incumbent.

So the split is:

* the metric contract `c1_confirmation_scoring@v1` is REUSED, unchanged. The
  frozen protocol requires screening and confirmation to share the mixture
  exactly, "because correct_overall and its +0.010 SESOI are defined ON the
  mixture, so screening and confirmation must share it or a screening delta says
  nothing about a confirmation delta". Same semantics is the requirement, not an
  economy.
* the battery IDENTITY is C2's, and is pinned here.

The pins are READ from `c2_screening_battery.json`, the identity record that
owns them, and checked against the manifest on disk. They are not transcribed
into module constants: a constant copied out of a record is a second owner of
the same fact, and the two disagree silently the first time either moves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

BATTERY_PATH = "artifacts/stage3/c2_screening_v1"
IDENTITY_RECORD = "logs/stages/stage-1/phase_c2/plans/c2_screening_battery.json"

#: What the screening rung may never do, from the frozen protocol. Carried into
#: every screening result so a reader of one result file alone still sees it.
SCREENING_MAY_NOT = (
    "produce a GO / NO-GO / INCONCLUSIVE verdict",
    "promote any candidate",
    "serve as a promotion or final-promotion asset",
    "become training data",
    "be used as a beam-ranking input",
)


class C2ScoringError(RuntimeError):
    """The screening battery is not the one the protocol froze."""


def screening_identity(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The identity record, verified against its own `record_sha256`."""
    from aadistill.infrastructure.manifest import sha256_json

    path = Path(repo_root) / IDENTITY_RECORD
    doc = json.loads(path.read_text())
    stated = doc.get("record_sha256")
    recomputed = sha256_json({k: v for k, v in doc.items()
                              if k != "record_sha256"})
    if stated != recomputed:
        raise C2ScoringError(
            f"{IDENTITY_RECORD} does not match its own record_sha256 "
            f"({recomputed} vs {stated}); it has been edited since it was "
            "frozen, and the screening rung pins to it")
    return doc


def validate_screening_battery(manifest: dict[str, Any], *,
                               repo_root: str | Path = REPO_ROOT,
                               ) -> dict[str, Any]:
    """Refuse anything that is not exactly the frozen C2 screening battery.

    Mirrors `validate_c1_battery`'s shape and failure mode — every check an
    equality, all problems collected before raising — but against C2's record
    rather than C1's module constants. A battery that was rebuilt, half-staged,
    re-mixed or relabelled fails here instead of producing a number that looks
    like a ranking.
    """
    from aadistill.infrastructure.manifest import sha256_json

    rec = screening_identity(repo_root)
    problems: list[str] = []

    #: The manifest carries no `manifest_sha256` of its own, so the convention
    #: is the canonicalized hash of the whole document. Both readings coincide
    #: while the key is absent; excluding it keeps that true if it is ever added.
    manifest_sha = sha256_json({k: v for k, v in manifest.items()
                                if k != "manifest_sha256"})
    if manifest_sha != rec["manifest_sha256"]:
        problems.append(
            f"battery manifest is {manifest_sha} but the identity record pins "
            f"{rec['manifest_sha256']}")
    if manifest.get("content_sha256") != rec["content_sha256"]:
        problems.append(
            f"battery content is {manifest.get('content_sha256')} but the "
            f"identity record pins {rec['content_sha256']}")
    if manifest.get("artifact") != rec["asset_id"]:
        problems.append(
            f"battery artifact is {manifest.get('artifact')!r}, pinned "
            f"{rec['asset_id']!r}")

    sets = manifest.get("sets") or {}
    if sorted(sets) != sorted(rec["set_counts"]):
        problems.append(
            f"battery sets are {sorted(sets)}, pinned {sorted(rec['set_counts'])}")
    else:
        for name, n in rec["set_counts"].items():
            spec = sets[name]
            if spec.get("n") != n:
                problems.append(f"{name}: manifest says n={spec.get('n')}, pinned {n}")
            if spec.get("sha256") != rec["set_sha256"][name]:
                problems.append(
                    f"{name}: content {spec.get('sha256')}, pinned "
                    f"{rec['set_sha256'][name]}")

    if set(manifest.get("scorable_sets") or ()) != set(rec["scorable_sets"]):
        problems.append(
            f"scorable sets are {sorted(manifest.get('scorable_sets') or ())}, "
            f"pinned {sorted(rec['scorable_sets'])}")
    if set(manifest.get("behaviour_only_sets") or ()) != set(
            rec["behaviour_only_sets"]):
        problems.append(
            f"behaviour-only sets are "
            f"{sorted(manifest.get('behaviour_only_sets') or ())}, pinned "
            f"{sorted(rec['behaviour_only_sets'])}")
    if manifest.get("n_prompts") != rec["n_prompts"]:
        problems.append(
            f"n_prompts {manifest.get('n_prompts')}, pinned {rec['n_prompts']}")
    if manifest.get("n_scorable_prompts") != rec["n_scorable_prompts"]:
        problems.append(
            f"n_scorable_prompts {manifest.get('n_scorable_prompts')}, pinned "
            f"{rec['n_scorable_prompts']}")

    if problems:
        raise C2ScoringError(
            "the battery is not the frozen C2 screening battery: "
            + "; ".join(problems))

    return {
        "artifact": rec["asset_id"],
        "role": rec["role"],
        "content_sha256": rec["content_sha256"],
        "manifest_sha256": manifest_sha,
        "n_prompts": rec["n_prompts"],
        "n_scorable_prompts": rec["n_scorable_prompts"],
        "identity_record": IDENTITY_RECORD,
        "record_sha256": rec["record_sha256"],
        "may_not": list(SCREENING_MAY_NOT),
    }
