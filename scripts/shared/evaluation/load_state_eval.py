"""Load the frozen `state_eval_v1` asset into the objects the evaluator takes.

One loader, shared by the preflight's repeatability gate and any later search, so
the suite a threshold was characterized on and the suite a candidate is ranked on
cannot drift apart by way of two transcriptions.
"""
from __future__ import annotations

import json
from pathlib import Path


def load(root: str | Path):
    import torch

    from aadistill.initialization.specs.metrics import StateEvalSuite, SuiteItem

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    #: THE CONTENT HASH IS REQUIRED TO EXIST — and deliberately NOT folded into
    #: `suite_hash`.
    #:
    #: `suite_hash` is the STRUCTURAL identity by this project's own design:
    #: `autoinit_phase_a_driver` pins it as "the STRUCTURAL suite hash —
    #: suite_id/version/domains/subtypes/critical_tags, which is what
    #: `StateEvalSuite.required_metrics()` and therefore the beam ranking read",
    #: and pins `STATE_EVAL_CONTENT_SHA256` beside it as a separate field that
    #: `verify_frozen_assets.py` re-derives from the loaded items. Forty-five
    #: committed records bind the structural value, and the C2 baseline driver
    #: refuses a measurement whose staged suite hash differs from the one its
    #: frozen candidates were measured under.
    #:
    #: So passing the content hash into the suite was the wrong repair: it moved
    #: the structural identity of an unchanged asset and silently reinterpreted
    #: every record that had pinned it. The real hole — that a measurement's
    #: identity did not bind the suite's CONTENT — is closed where it belongs,
    #: in `measurement_protocol_id`, which is new and pins nothing historical;
    #: `StateEvaluator` takes this value as `suite_content_sha256` and binds it
    #: there beside the structural hash.
    #:
    #: What this loader owes is that the value EXISTS and comes from the record
    #: the asset ships with rather than being recomputed here, so a caller
    #: cannot end up binding `None`.
    content = manifest.get("content_sha256")
    if not content:
        raise ValueError(
            f"{root}/manifest.json carries no `content_sha256`, so a "
            "measurement taken on this asset could not bind its content and "
            "could not be told apart from one taken on different prompts. The "
            "builder writes this field; an asset without it is incomplete.")
    suite = StateEvalSuite(
        suite_id=manifest["suite_id"], version=manifest["version"],
        domains=tuple(manifest["domains"]),
        subtypes={d: tuple(s) for d, s in manifest["domains"].items()}
        if isinstance(manifest["domains"], dict) else
        {d: tuple(s) for d, s in manifest["subtypes"].items()},
        critical_tags=tuple(manifest["critical_tags"]),
        general_domain=manifest.get("general_domain", "general"))
    items = []
    for line in (root / "items.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids = torch.tensor([row["ids"]], dtype=torch.long)
        targets = ids[0, 1:]
        # Tags are stored as prediction-position INDICES and expanded to boolean
        # masks here. Read as masks they would be the wrong length and silently
        # reweight the critical-token metric, so the bound is checked.
        n_pred = targets.shape[0]
        if n_pred != row["n_prediction_positions"]:
            raise ValueError(
                f"{row['item_id']}: {n_pred} prediction positions but the manifest "
                f"says {row['n_prediction_positions']}")
        tags = {}
        for name, positions in row["tags"].items():
            mask = torch.zeros(n_pred, dtype=torch.bool)
            index = torch.tensor(positions, dtype=torch.long)
            if index.numel() and int(index.max()) >= n_pred:
                raise ValueError(
                    f"{row['item_id']}: tag {name!r} indexes position "
                    f"{int(index.max())} beyond {n_pred}")
            mask[index] = True
            tags[name] = mask
        items.append(SuiteItem(item_id=row["item_id"], input_ids=ids,
                               domain=row["domain"], subtype=row["subtype"],
                               tags=tags))
    return suite, items, manifest
