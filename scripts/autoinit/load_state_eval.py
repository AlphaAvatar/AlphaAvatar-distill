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

    from aadistill.autoinit.metrics import StateEvalSuite, SuiteItem

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
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
