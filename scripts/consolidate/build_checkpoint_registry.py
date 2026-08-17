#!/usr/bin/env python3
"""Canonical inventory of every checkpoint and weight artifact, wherever it lives.

    PYTHONPATH=src python scripts/consolidate/build_checkpoint_registry.py \
        --hash --relay --out logs/checkpoint_registry.json

Nothing is deleted here. This produces the registry a deletion pass may act on,
and the rule is that no file is removed before it appears in this file with a
class, a hash and a justification.

Three storage areas, because "is there another copy?" cannot be answered from one:

    local        artifacts/ in the repo AND /home/ecs-user/aad-artifacts, the
                 out-of-tree store every pod session collects into. The second one
                 holds the overwhelming majority of the bytes and was invisible to
                 the first version of this script, which is why it is named here
    relay        AlphaAvatar/aadistill-artifacts — the artifact store pods fetch from
    repository   anything git actually tracks. Expected to be empty (AGENTS.md 2.5)

Retention classes, from the consolidation brief:

    canonical                the accepted result of a completed experiment
    control                  a baseline another accepted result is measured against
    behavioral_anchor        the current best behaviour, or a reference for it
    diagnostic               produced to answer a question, not to be promoted
    reproducibility_required needed to reproduce an accepted endpoint
    superseded               a later checkpoint replaced its role
    duplicate                byte-identical or trivially reconstructable copy
    failed_or_partial        an interrupted or OOM-killed run
    deletable                safe to remove once a tombstone records it

A checkpoint is `deletable` only when it is none of the protective classes, its
scientific status is captured elsewhere, and its hash survives in this registry.

**Deleting from the relay is not a way to reclaim relay storage**, and this
script never proposes it. On a Hugging Face repository at its LFS limit, removing
a file from the current revision frees nothing — the object stays referenced by
history — and the operations that would free it are exactly the ones that must
not run unattended. `by_prefix` in the relay section is therefore a map for a
human decision, not a deletion plan.

**A hash-verified relay copy does not by itself make a local copy deletable.** A
canonical initialization, a permanent control, a finalist, an artifact the
experiment index or a living recovery path names, and anything unique stays local
whatever the relay holds. The relay column exists so that fact is *checked*
rather than assumed; the `relay_verified` flag says a second copy exists, and
`disposition` says whether that matters.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

RELAY_REPO = "AlphaAvatar/aadistill-artifacts"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt")

#: Local storage roots. The second is out of tree: pod sessions collect into it,
#: nothing in the repository points at it, and it is where the bytes actually are.
LOCAL_ROOTS = ((REPO_ROOT / "artifacts", "repo_artifacts"),
               (Path("/home/ecs-user/aad-artifacts"), "external_store"))

#: path suffix -> classification. `disposition` is the proposal a deletion pass
#: reads; `never_delete` records which clause of the retention rule protects it.
CLASSIFY: dict[str, dict] = {
    "stage1/qwen3_0p6b_init_v0/checkpoint": dict(
        experiment="E1-E8", role="positional Stage-1 initialization (FP lineage)",
        retention="canonical", status="active",
        why="the pinned control initialization every E1/E4/E6/E7/E8 arm descends "
            "from; reproducing any accepted result needs it, and it is the "
            "Phase-A probe base",
        never_delete="canonical initialization; named by the experiment index and "
                     "by a living recovery path",
        reconstruction="PYTHONPATH=src python scripts/init/build_stage1_init.py "
                       "against the pinned Stage-0 activation cache aaeb2e4c…",
        reconstruction_cost="$0 (CPU)", disposition="keep"),
    "stage1/e8_contribution_init_v1/checkpoint": dict(
        experiment="E8a",
        role="contribution-guided fully-compressed initialization (FC)",
        retention="canonical", status="active",
        why="E8a's accepted treatment initialization, single-variable against the "
            "control",
        never_delete="canonical initialization; named by the experiment index",
        reconstruction="the E8a initialization recipe against the same Stage-0 "
                       "cache and the frozen depth map",
        reconstruction_cost="$0 (CPU)", disposition="keep"),
    "stage3/rescued/e1_r2960k_sb_pca": dict(
        experiment="E1/E6", role="2.96M rung, seed sb",
        retention="canonical", status="active",
        why="part of the accepted E1 scale curve",
        never_delete="accepted experiment checkpoint; the relay holds its "
                     "evaluations but NOT its weights, so this is the only copy",
        reconstruction="a paid recovery run of configs/stage3/e1/e1_r2960k_sb_pca.json",
        reconstruction_cost="paid GPU (a full 2.96M-token recovery)",
        disposition="keep"),
    "stage0/qwen3_4b_thinking_v1": dict(
        experiment="Stage 0", role="teacher activation statistics cache",
        retention="reproducibility_required", status="active",
        why="every structural initialization in the project is a deterministic "
            "function of this cache; it is the input the tombstoned "
            "initializations name as their reconstruction recipe",
        never_delete="living recovery path: the reconstruction recipe of four "
                     "tombstoned checkpoints reads it",
        reconstruction="a full teacher statistics pass over the pinned corpus",
        reconstruction_cost="paid GPU (teacher inference)", disposition="keep"),
    "autoinit/dryrun/canonical_control": dict(
        experiment="AutoInitializer dry run", role="toy control materialization",
        retention="duplicate", status="superseded",
        why="a 32-wide 6-layer toy model built by a $0 CPU dry run; its identity "
            "is recorded in artifacts/autoinit/dryrun/search/states.jsonl and in "
            "logs/autoinit_dryrun_fresh.json",
        never_delete=None,
        reconstruction="PYTHONPATH=src python scripts/autoinit/dry_run_search.py "
                       "--out artifacts/autoinit/dryrun",
        reconstruction_cost="$0 (CPU, minutes)", disposition="delete"),
    "autoinit/dryrun/search/states": dict(
        experiment="AutoInitializer dry run", role="rejected/searched toy leaf",
        retention="duplicate", status="superseded",
        why="a materialized beam leaf of the $0 toy search. Its complete lineage — "
            "arch spec hash, artifact digest, config hash, parameter count, score "
            "and prune decision — is in search/states.jsonl and in "
            "logs/autoinit_dryrun_fresh.json / _resume.json",
        never_delete=None,
        reconstruction="PYTHONPATH=src python scripts/autoinit/dry_run_search.py "
                       "--out artifacts/autoinit/dryrun",
        reconstruction_cost="$0 (CPU, minutes)", disposition="delete"),

    # ---- the out-of-tree store -----------------------------------------
    "aad-artifacts/wheelhouse_vllm_cp312": dict(
        experiment="infrastructure", role="vLLM wheelhouse build cache",
        retention="duplicate", status="superseded",
        why="196 wheels, every one hash-verified against the tracked manifest "
            "wheelhouse_vllm_sha256.json AND against the relay copy at "
            "transfer/wheelhouse_vllm_cp312 (123 by LFS oid, 73 downloaded and "
            "hashed). Pods fetch from the relay, never from this machine",
        never_delete=None,
        reconstruction="hf download AlphaAvatar/aadistill-artifacts "
                       "transfer/wheelhouse_vllm_cp312, or rebuild with "
                       "scripts/pod/build_wheelhouse.py --from-pins "
                       "--requirements requirements-vllm.txt",
        reconstruction_cost="$0 (download)", disposition="delete"),
    "aad-artifacts/autoinit/preflight_ctl_r0860k": dict(
        experiment="Stage 3", role="permanent control (0.86M rung)",
        retention="control", status="active",
        why="one of the two permanent controls Stage 3 completed on; the frozen "
            "thresholds were materialized against them",
        never_delete="permanent control — named explicitly in the retention rule",
        reconstruction="a paid recovery run; the model is also on the relay at "
                       "permanent_controls/, the optimizer state is not",
        reconstruction_cost="paid GPU", disposition="keep"),
    "aad-artifacts/p2_ceheavy": dict(
        experiment="P2 / E4 / E5", role="CE-heavy 0.86M start checkpoint",
        retention="reproducibility_required", status="active",
        why="the initialization E4's 1.60M arms and E5's arm-C/arm-R recoveries "
            "start from; configs/stage3/{e4,e5}/*.json name it",
        never_delete="living recovery path: downstream experiment configs start "
                     "from it. Its relay copy at e5_start/ is hash-verified, which "
                     "makes it eligible under the stale-cache clause and the "
                     "never-delete clause overrides that",
        reconstruction="hf download AlphaAvatar/aadistill-artifacts e5_start/",
        reconstruction_cost="$0 (download)", disposition="keep"),
    "aad-artifacts/e7/": dict(
        experiment="E7", role="FineWeb-Edu KD arm or its matched control",
        retention="canonical", status="active",
        why="E7's four accepted arms. The relay holds no copy of these weights",
        never_delete="accepted experiment checkpoint, and the only copy anywhere",
        reconstruction="a paid recovery run of the E7 arm config",
        reconstruction_cost="paid GPU", disposition="keep"),
    "aad-artifacts/e6b/": dict(
        experiment="E6b", role="P2 CE-heavy 2.96M arm",
        retention="canonical", status="active",
        why="the evidence behind the accepted objective x scale interaction. The "
            "relay holds no copy of these weights",
        never_delete="accepted experiment checkpoint, and the only copy anywhere",
        reconstruction="a paid recovery run of the E6b arm config",
        reconstruction_cost="paid GPU", disposition="keep"),
    "aad-artifacts/e4/": dict(
        experiment="E4", role="P2 CE-heavy 1.60M arm",
        retention="canonical", status="active",
        why="the CE-heavy side of the accepted KD-vs-CE comparison. The relay "
            "holds no copy of these weights",
        never_delete="accepted experiment checkpoint, and the only copy anywhere",
        reconstruction="a paid recovery run of configs/stage3/e4/*.json",
        reconstruction_cost="paid GPU", disposition="keep"),
    "aad-artifacts/e2p1/": dict(
        experiment="E2 phase 1", role="0.86M diagnostic arm",
        retention="diagnostic", status="closed",
        why="E2 is a diagnostic whose conclusion is recorded in EXPERIMENTS.md 12 "
            "and does not depend on the weights; phases 2-3 were never authorized. "
            "Four step checkpoints are kept per arm where one is the endpoint",
        never_delete=None,
        reconstruction="a paid recovery run of the E2 phase-1 arm config",
        reconstruction_cost="paid GPU", disposition="review"),
    "aad-artifacts/e5/": dict(
        experiment="E5", role="per-attempt side artifacts",
        retention="diagnostic", status="closed",
        why="side bundles collected from five E5 attempts, four of which failed on "
            "infrastructure. No weights: evaluation output, configs and logs",
        never_delete=None, reconstruction=None, reconstruction_cost=None,
        disposition="review"),
}

PROTECTED = {"canonical", "control", "behavioral_anchor", "reproducibility_required"}

#: Bulk non-weight artifacts that are byte-identical to a copy that survives.
#: Weights are the subject of this registry; these are inventoried alongside
#: because they occupy the same local storage and need the same discipline.
BULK_DUPLICATE_NOTES = {
    "artifacts/audit/ladder_uniform_rebuild": (
        "the rebuilt ladder from scripts/data/audit_e1_mixture_rebuild.py. Its "
        "blocks.npz and audit.jsonl are byte-identical to the historical pack it "
        "was compared against; the audit's value is the recorded match "
        "(artifacts/audit/e1_mixture_rebuild.json, logs/EXPERIMENTS.md), not the "
        "second copy of the bytes"),
    "artifacts/stage3/ladder_uniform": (
        "the trainer-side name of the frozen training pack. Byte-identical to "
        "artifacts/stage3/ladder_uniform_probe, which the test suite reads, and "
        "refetched from the relay at stage3_recovery_corpus_v2/ladder_uniform by "
        "every pod setup script against the pinned hash 6f324cb0…"),
    "artifacts/_audit_nested_bak": (
        "residue of the 2026-08-15 pod-simulator nesting defect: two copies of "
        "artifacts/audit/autoinit_preflight/frozen_asset_verification.json saved "
        "while the real audit tree was being recovered. The verifier regenerates "
        "that file on every run"),
}


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def classify(rel: str) -> dict:
    for suffix, meta in CLASSIFY.items():
        if rel.endswith(suffix) or suffix in rel:
            return meta
    return dict(experiment="unknown", role="unclassified", retention="diagnostic",
                status="unreviewed",
                why="not matched by the classification table — review before any "
                    "deletion",
                never_delete="unclassified", reconstruction=None,
                reconstruction_cost=None, disposition="review")


def load_corpus() -> dict[str, str]:
    corpus = {}
    for r in ("logs", "configs", "tests", "scripts", "src", "docs", "README.md"):
        base = REPO_ROOT / r
        files = [base] if base.is_file() else [
            f for f in base.rglob("*") if f.is_file()
            and "__pycache__" not in f.parts
            and f.suffix in (".md", ".json", ".py", ".txt", ".sh")]
        for f in files:
            try:
                corpus[str(f.relative_to(REPO_ROOT))] = f.read_text(errors="ignore")
            except Exception:
                continue
    return corpus


def referenced_by(corpus: dict[str, str], rel: str, name: str) -> list[str]:
    hits = [k for k, text in corpus.items() if rel in text or name in text]
    return sorted(set(hits))[:12]


def discover_units() -> list[tuple[str, str, Path, list[Path]]]:
    """(store, kind, unit path, weight files). A checkpoint unit is a directory
    with a config beside its weights; anything else weight-shaped stands alone —
    which is how a bare trainer_state.pt gets counted rather than hidden inside
    the checkpoint it sits next to."""
    units, claimed = [], set()
    for root, store in LOCAL_ROOTS:
        if not root.is_dir():
            continue
        weights = sorted(p for p in root.rglob("*")
                         if p.is_file() and p.suffix in WEIGHT_SUFFIXES)
        for w in weights:
            if w in claimed:
                continue
            d = w.parent
            if (d / "config.json").is_file():
                group = sorted(x for x in d.iterdir()
                               if x.is_file() and x.suffix in WEIGHT_SUFFIXES)
                claimed.update(group)
                units.append((store, "checkpoint", d, group))
            else:
                claimed.add(w)
                units.append((store, "tensor_file", w, [w]))
    return units


def rel_to_repo(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def declared_trees(matched: set[str]) -> list[tuple[str, str, Path, list[Path]]]:
    """Trees named in CLASSIFY that hold no weight file — the vLLM wheelhouse is
    3.6 GiB of local storage under the same retention rules, and a registry that
    only sees `.safetensors` would report it as absent rather than as kept."""
    out = []
    for suffix in CLASSIFY:
        for root, store in LOCAL_ROOTS:
            cand = root.parent / suffix if suffix.startswith("aad-artifacts/") \
                else root / suffix
            if not cand.is_dir() or rel_to_repo(cand) in matched:
                continue
            if any(p.suffix in WEIGHT_SUFFIXES for p in cand.rglob("*")
                   if p.is_file()):
                continue
            out.append((store, "artifact_tree", cand, []))
    return out


def relay_listing(use_network: bool, previous: dict | None) -> dict:
    if not use_network:
        if previous and previous.get("relay", {}).get("files"):
            r = dict(previous["relay"])
            r["source"] = "carried forward from the previous registry (--relay not given)"
            return r
        return {"repo": RELAY_REPO, "queried": False, "files": [],
                "source": "not queried; rerun with --relay"}
    from huggingface_hub import HfApi                      # noqa: PLC0415
    info = HfApi().repo_info(RELAY_REPO, repo_type="model", files_metadata=True)
    files = []
    for f in info.siblings or []:
        lfs = f.lfs
        files.append({"path": f.rfilename,
                      "size_bytes": (lfs.size if lfs else f.size) or 0,
                      "sha256": lfs.sha256 if lfs else None})
    by_prefix = collections.Counter()
    bytes_prefix = collections.Counter()
    for f in files:
        k = f["path"].split("/")[0]
        by_prefix[k] += 1
        bytes_prefix[k] += f["size_bytes"]
    return {"repo": RELAY_REPO, "queried": True,
            "queried_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_files": len(files),
            "total_bytes": sum(f["size_bytes"] for f in files),
            "by_prefix": {k: {"files": by_prefix[k], "bytes": bytes_prefix[k]}
                          for k in sorted(bytes_prefix, key=bytes_prefix.get,
                                          reverse=True)},
            "note": ("an LFS object's oid IS the sha256 of its content, so a local "
                     "copy can be verified against the relay without downloading it"),
            "files": files,
            "source": "huggingface_hub repo_info(files_metadata=True)"}


def mirror_verifications() -> dict[str, dict]:
    """What scripts/consolidate/verify_relay_mirror.py has actually proved, keyed
    by local tree. A `verified_stale_cache` deletion cites this, so the registry
    reads the evidence rather than repeating a claim about it."""
    p = REPO_ROOT / "logs/relay_mirror_verification.json"
    if not p.is_file():
        return {}
    out = {}
    for v in json.loads(p.read_text()).get("verifications", []):
        out[v["local_tree"].rstrip("/")] = {
            "verified": v["verified"],
            "relay_prefix": v["relay_prefix"],
            "n_files": v["n_local_files"],
            "by_lfs_oid": v["verified_by_lfs_oid"],
            "by_download": v["verified_by_download"],
            "generated_utc": v["generated_utc"],
            "evidence": "logs/relay_mirror_verification.json",
        }
    return out


def repository_visible() -> dict:
    """Weight artifacts git actually tracks. AGENTS.md 2.5 says: none."""
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True,
                         text=True, check=False).stdout.splitlines()
    tracked = [p for p in out if Path(p).suffix in WEIGHT_SUFFIXES or
               Path(p).suffix in (".npz", ".npy")]
    return {"rule": "AGENTS.md 2.5 — checkpoints, optimizer states, activation "
                    "caches and large datasets are never committed",
            "tracked_weight_artifacts": tracked,
            "violations": len(tracked)}


def bulk_duplicates() -> list[dict]:
    """Byte-identical non-weight artifacts under artifacts/, largest first."""
    art = REPO_ROOT / "artifacts"
    if not art.is_dir():
        return []
    by_size = collections.defaultdict(list)
    for p in art.rglob("*"):
        if p.is_file() and p.suffix not in WEIGHT_SUFFIXES and p.stat().st_size > 4096:
            by_size[p.stat().st_size].append(p)
    groups = collections.defaultdict(list)
    for size, ps in by_size.items():
        if len(ps) > 1:
            for p in ps:
                groups[(size, sha256_file(p))].append(str(p.relative_to(REPO_ROOT)))
    out = []
    for (size, digest), members in groups.items():
        if len(members) < 2:
            continue
        note = next((v for k, v in BULK_DUPLICATE_NOTES.items()
                     if any(m.startswith(k) for m in members)), None)
        out.append({"sha256": digest, "size_bytes": size,
                    "members": sorted(members),
                    "reclaimable_bytes": size * (len(members) - 1),
                    "note": note})
    return sorted(out, key=lambda g: -g["reclaimable_bytes"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/checkpoint_registry.json")
    ap.add_argument("--hash", action="store_true",
                    help="compute weight sha256 (slow; the registry is not "
                         "authoritative for deletion without it)")
    ap.add_argument("--relay", action="store_true",
                    help="query the artifact store; without it the previous "
                         "listing is carried forward")
    args = ap.parse_args()

    out_path = REPO_ROOT / args.out
    previous = json.loads(out_path.read_text()) if out_path.is_file() else None
    relay = relay_listing(args.relay, previous)
    relay_by_sha = {f["sha256"]: f["path"] for f in relay["files"] if f["sha256"]}
    relay_by_size = collections.defaultdict(list)
    for f in relay["files"]:
        relay_by_size[f["size_bytes"]].append(f["path"])

    corpus = load_corpus()
    mirrors = mirror_verifications()
    entries = []
    units = discover_units()
    units += declared_trees({rel_to_repo(u) for _, _, u, _ in units})
    for store, kind, unit, weights in units:
        rel = rel_to_repo(unit)
        meta = classify(rel)
        cfg = (unit / "config.json") if kind == "checkpoint" else None
        arch = {}
        if cfg and cfg.is_file():
            c = json.loads(cfg.read_text())
            arch = {k: c.get(k) for k in
                    ("num_hidden_layers", "hidden_size", "intermediate_size",
                     "num_attention_heads", "num_key_value_heads", "vocab_size")}
        name = unit.name if unit.name != "checkpoint" else unit.parent.name
        primary = weights[0] if weights else None
        wsha = sha256_file(primary) if (args.hash and primary) else None
        size = (primary.stat().st_size if kind == "tensor_file"
                else dir_size(unit))
        relay_path = relay_by_sha.get(wsha) if wsha else None
        relay_size_only = (None if (relay_path or not primary) else
                           next(iter(relay_by_size.get(primary.stat().st_size, [])),
                                None))
        entries.append({
            "canonical_name": name,
            "kind": kind,
            "store": store,
            "path_local": rel,
            "weight_files": [rel_to_repo(w) for w in weights],
            "experiment": meta["experiment"],
            "role": meta["role"],
            "retention": meta["retention"],
            "status": meta["status"],
            "retention_reason": meta["why"],
            "architecture": arch,
            "size_bytes": size,
            "size_human": f"{size / 2**30:.3f} GiB",
            "weights_sha256": wsha,
            "config_sha256": (hashlib.sha256(cfg.read_bytes()).hexdigest()
                              if cfg and cfg.is_file() else None),
            "referenced_by": referenced_by(corpus, rel, name),
            "relay_path": relay_path,
            "relay_verified": bool(relay_path),
            "relay_same_size_unverified": relay_size_only,
            "relay_mirror_verification": mirrors.get(str(unit)),
            "canonical_location": (
                "relay + local" if (relay_path or mirrors.get(str(unit), {}).get("verified"))
                else "local only (no second copy)"),
            "reconstructable": meta["reconstruction"] is not None,
            "reconstruction_recipe": meta["reconstruction"],
            "reconstruction_cost": meta["reconstruction_cost"],
            "never_delete_clause": meta["never_delete"],
            "protected": meta["retention"] in PROTECTED or bool(meta["never_delete"]),
            "disposition": meta["disposition"],
        })

    entries.sort(key=lambda e: -e["size_bytes"])
    total = sum(e["size_bytes"] for e in entries)
    proposed = collections.Counter(e["disposition"] for e in entries)
    bulk = bulk_duplicates()
    by_store = collections.defaultdict(lambda: {"units": 0, "bytes": 0})
    for e in entries:
        by_store[e["store"]]["units"] += 1
        by_store[e["store"]]["bytes"] += e["size_bytes"]
    proposed_bytes = collections.Counter()
    for e in entries:
        proposed_bytes[e["disposition"]] += e["size_bytes"]

    report = {
        "schema": "aadistill.checkpoint_registry/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "hashed": bool(args.hash),
        "scope": ("every checkpoint and weight artifact in the three storage areas: "
                  "local artifacts/, the relay, and anything git tracks"),
        "deletion_rule": ("a checkpoint may be deleted only if it is not protected, "
                          "no never_delete clause applies, its scientific status is "
                          "recorded elsewhere, its hash is in this registry, and a "
                          "tombstone is written to logs/checkpoint_tombstones.json"),
        "retention_classes": sorted({e["retention"] for e in entries}),
        "protected_classes": sorted(PROTECTED),
        "local": {
            "n_units": len(entries),
            "n_weight_files": sum(len(e["weight_files"]) for e in entries),
            "total_bytes": total,
            "total_human": f"{total / 2**30:.3f} GiB",
            "artifacts_tree_bytes": dir_size(REPO_ROOT / "artifacts"),
            "external_store_tree_bytes": dir_size(Path("/home/ecs-user/aad-artifacts")),
            "by_store": {k: dict(v) for k, v in sorted(by_store.items())},
            "proposed_disposition": dict(sorted(proposed.items())),
            "proposed_disposition_bytes": dict(sorted(proposed_bytes.items())),
        },
        "relay": relay,
        "repository_visible": repository_visible(),
        "non_weight_artifact_duplicates": bulk,
        "checkpoints": entries,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{len(entries)} weight units, {total / 2**30:.3f} GiB local; "
          f"artifacts/ tree {report['local']['artifacts_tree_bytes'] / 2**30:.3f} GiB")
    for e in entries[:12]:
        flag = "PROTECTED" if e["protected"] else e["disposition"].upper()
        print(f"  {flag:10s} {e['retention']:24s} {e['size_human']:>11s}  "
              f"{e['path_local']}")
    if len(entries) > 12:
        print(f"  … {len(entries) - 12} more (see the file)")
    print(f"  relay: {relay.get('n_files', len(relay['files']))} files, "
          f"{relay.get('total_bytes', 0) / 2**30:.2f} GiB")
    print(f"  git-tracked weight artifacts: "
          f"{report['repository_visible']['violations']}")
    print(f"  non-weight byte-identical groups: {len(bulk)}, reclaimable "
          f"{sum(g['reclaimable_bytes'] for g in bulk) / 2**20:.2f} MiB")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
