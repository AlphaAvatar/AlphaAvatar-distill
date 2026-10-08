#!/usr/bin/env python3
"""Prove an E7 stream shares no content with anything it must not.

Two independent checks, because either alone can be fooled.

**Index separation** says which rows of a streaming dataset each artifact read.
It is cheap and it is what the builders enforce up front — but two disjoint
index ranges can still deliver the same document, because FineWeb-Edu contains
near-duplicates and because a re-tagged revision renumbers everything.

**Content-hash separation** is the real proof: sha256 over document text alone,
excluding ids and metadata, compared across every artifact.

Fails closed. A missing input file is an error, not a skipped check — a
disjointness proof that silently proves nothing is worse than no proof, and this
runs immediately before paid training.

    python3 scripts/data/check_stream_disjointness.py \\
        --stream artifacts/stage3/e7_fineweb_kd \\
        --stream artifacts/stage3/e7_fineweb_val \\
        --stream artifacts/stage3/e7_control_kd \\
        --reserved data/warmup/holdout_v1.jsonl \\
        --reserved data/warmup/warmup_v1.jsonl \\
        --reserved data/eval_behavior_v0/prompts.jsonl \\
        --out artifacts/stage3/e7_disjointness.json

Exit codes: 0 disjoint; 6 an overlap or a missing input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.extra_stream import content_sha256  # noqa: E402


def row_text(row: dict) -> str:
    """The comparable content of a jsonl row, whatever schema it uses."""
    if isinstance(row.get("text"), str):
        return row["text"]
    if isinstance(row.get("messages"), list):
        return "\n".join(str(m.get("content", "")) for m in row["messages"])
    for key in ("prompt", "question", "content", "input"):
        if isinstance(row.get(key), str):
            return row[key]
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def hashes_from_jsonl(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; refusing to report disjointness against a "
            "file that was not read")
    out = set()
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            out.add(hashlib.sha256(line.encode()).hexdigest())
            continue
        if isinstance(row, dict) and isinstance(row.get("sha256"), str):
            out.add(row["sha256"])
        out.add(content_sha256(row_text(row) if isinstance(row, dict) else str(row)))
    return out


def stream_record(d: Path) -> dict:
    manifest = json.loads((d / "manifest.json").read_text())
    docs = d / "docs.jsonl"
    if not docs.is_file():
        raise FileNotFoundError(f"{docs} is missing; a stream must list its "
                                "documents to be checkable")
    hashes = {json.loads(l)["sha256"]
              for l in docs.read_text().splitlines() if l.strip()}
    return {"label": d.name, "manifest": manifest, "hashes": hashes}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream", action="append", default=[])
    ap.add_argument("--reserved", action="append", default=[])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.stream:
        raise SystemExit("no --stream given; nothing would be checked")

    groups: dict[str, set[str]] = {}
    manifests: dict[str, dict] = {}
    for s in args.stream:
        p = Path(s) if Path(s).is_absolute() else REPO_ROOT / s
        rec = stream_record(p)
        groups[rec["label"]] = rec["hashes"]
        manifests[rec["label"]] = rec["manifest"]
    for r in args.reserved:
        p = Path(r) if Path(r).is_absolute() else REPO_ROOT / r
        groups[f"reserved:{p.name}"] = hashes_from_jsonl(p)

    # An overlap involving an E7 stream is leakage and blocks the experiment.
    # An overlap between two *reserved* artifacts is a property of those
    # artifacts — real, worth reporting, and not something E7 can or should fix
    # by refusing to run. The two are reported separately so neither hides the
    # other; only the first is fatal.
    stream_labels = {Path(s).name for s in args.stream}
    labels = sorted(groups)
    overlaps = []
    reserved_overlaps = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            shared = groups[a] & groups[b]
            if not shared:
                continue
            row = {"a": a, "b": b, "n_shared": len(shared),
                   "examples": sorted(shared)[:5]}
            if a in stream_labels or b in stream_labels:
                overlaps.append(row)
            else:
                reserved_overlaps.append(row)

    # Index separation, for the streams that came from an indexed dataset.
    ranges = {}
    for label, m in manifests.items():
        src = m.get("source", {})
        if "index_range" in src:
            ranges[label] = (src["dataset"], tuple(src["index_range"]))
    index_overlaps = []
    keys = sorted(ranges)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            (da, ra), (db, rb) = ranges[a], ranges[b]
            if da == db and ra[0] < rb[1] and rb[0] < ra[1]:
                index_overlaps.append({"a": a, "b": b, "dataset": da,
                                       "range_a": list(ra), "range_b": list(rb)})

    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "groups": {k: len(v) for k, v in groups.items()},
        "index_ranges": {k: {"dataset": v[0], "range": list(v[1])}
                         for k, v in ranges.items()},
        "content_hash_overlaps": overlaps,
        "index_range_overlaps": index_overlaps,
        "reserved_vs_reserved_overlaps": reserved_overlaps,
        "disjoint": not overlaps and not index_overlaps,
    }
    if args.out:
        out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if reserved_overlaps:
        print(f"NOTE: {len(reserved_overlaps)} overlap(s) between reserved "
              "artifacts, not involving any E7 stream. Recorded, not fatal — "
              "but two evaluation subsets that share an item are not "
              "independent, and per-subset comparisons should say so.",
              file=sys.stderr)
    if not report["disjoint"]:
        print("LEAKAGE: the streams above share content or index ranges. "
              "E7 must not train on anything it validates on, and must not "
              "train on the Stage 0 statistics or the historical holdout.",
              file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
