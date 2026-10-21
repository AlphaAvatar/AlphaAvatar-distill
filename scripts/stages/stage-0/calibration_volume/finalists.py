#!/usr/bin/env python3
"""Stage-0 volume study, Top-4 extension: which leaves, and how a retired one is rebuilt.

    # $0, on the host that holds the dose's journal: write the d1 rebuild plan
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/finalists.py \
        --treatment d1 --journal <states.jsonl> --selection <stage1_selection.json> \
        --tombstone <retired_states_d1.json> --out <plan.json>

Maintainer decision 2026-10-20 (`plans/top4_extension.json`): every dose's Top-4
complete leaves from its own full search are recovered and evaluated.

* `distinct_finalists` -- the candidate rule. The search's own quality order
  (`stage1_selection.json :: decisions`, the policy's ordering with no lineage
  rotation), byte-identical leaves counted once by checkpoint sha256, the first
  `k`. d2's quality positions 1 and 2 are the same weights: DEPTH removed the
  same eight layers under both profiles, so everything built on the two DEPTH
  states is identical.
* `rebuild_plan` -- d1's Top-2..4 were retired with d1's other states
  (2026-10-19). The tombstone kept their digests and the journal kept their
  ancestry, so each is rebuilt as a digest-pinned fixed path: the D1 replay's
  mechanism (`stages.phase_d1.replay_specs`, which rebuilt two D1 finalists
  byte-exactly on 2026-10-07), with the spec built from Stage-0's owners --
  the forward baseline's scoring terms and execution, A3's target geometry, the
  D1 teacher binding -- because those are what the Stage-0 search ran under.
  The three $0 agreement checks (root state, operator configs, execution
  knobs) run here, before any pod. It is a REPLAY: nothing ranks, measures or
  re-selects, and the only outcomes are "byte-identical to the search" and STOP.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

PREREG = REPO / "logs/stages/stage-0/calibration_volume/plans/volume_preregistration.json"


class FinalistError(RuntimeError):
    pass


def journal_index(journal: Path) -> dict[str, dict[str, Any]]:
    """state_id -> its rows merged, an empty value never replacing a recorded one.

    A resumed search journals every restored state again with an empty decision
    and empty step traces; the identities agree, so the merge keeps them."""
    out: dict[str, dict[str, Any]] = {}
    for line in Path(journal).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prior = out.setdefault(row["state_id"], {})
        for k, v in row.items():
            if v not in (None, "", [], {}) or k not in prior:
                prior[k] = v
    return out


def distinct_finalists(selection: Mapping[str, Any], states: Mapping[str, Mapping[str, Any]],
                       k: int) -> list[dict[str, Any]]:
    """The first `k` complete leaves by the policy's quality order, byte-identical
    leaves counted once (the better-ranked one stands for both)."""
    out: list[dict[str, Any]] = []
    by_sha: dict[str, dict[str, Any]] = {}
    for d in sorted(selection["decisions"], key=lambda d: int(d["quality_order"])):
        row = states.get(d["state_id"]) or {}
        sha = row.get("checkpoint_sha256")
        if not sha:
            raise FinalistError(f"{d['state_id']} has no checkpoint_sha256 in the journal")
        if sha in by_sha:
            by_sha[sha]["identical_leaves"].append(
                {"state_id": d["state_id"], "quality_order": int(d["quality_order"])})
            continue
        if len(out) == k:
            break
        art = row.get("artifact") or {}
        entry = {"rank": len(out) + 1, "state_id": d["state_id"],
                 "quality_order": int(d["quality_order"]), "path": d.get("path"),
                 "checkpoint_sha256": sha,
                 "artifact_digest": row.get("artifact_digest") or art.get("artifact_digest"),
                 "weights_digest": art.get("weights_digest"),
                 "num_parameters": row.get("num_parameters"),
                 "identical_leaves": []}
        by_sha[sha] = entry
        out.append(entry)
    if len(out) < k:
        raise FinalistError(f"only {len(out)} distinct complete leaves; the rule needs {k}")
    return out


# --- the d1 rebuild ----------------------------------------------------------------

def _registries(repo_root: Path = REPO):
    """Every process-global registry the Stage-0 search filled, in its order."""
    from shared import forward_baseline as FB
    from stages.calibration_volume.search_session import register_stage0_profiles
    from stages.phase_d1 import d1_session as D1S

    prereg = json.loads(PREREG.read_text())
    baseline = FB.selected(repo_root)
    D1S._register_frozen_operators()
    FB.register_operators(baseline)
    register_stage0_profiles(repo_root / prereg["profiles_doc"])
    return FB.session_terms(baseline), prereg


def search_execution(terms: Mapping[str, Any], prereg: Mapping[str, Any]):
    """The execution the dose searches ran: the baseline's, with DEPTH prefix
    reuse exactly when the preregistration ran it (`dose_driver` S stage)."""
    ex = terms["execution"]
    if prereg["execution"]["depth_prefix_reuse"]:
        ex = dataclasses.replace(ex, depth_prefix_reuse=True)
    return ex


def fixed_path_spec(leaf, *, device: str = "cuda", repo_root: Path = REPO):
    """`leaf` as a FixedPathSpec under the Stage-0 search's own owners."""
    from aadistill.initialization.planning.fixed_path import FixedPathSpec, FixedPathStep
    from stages.phase_a3 import a3_session as A3S
    from stages.phase_d1 import d1_session as D1S

    terms, _ = _registries(repo_root)
    target = A3S.path_spec(workdir_device=device).target_spec
    if target.spec_hash != leaf.target_spec_hash:
        raise FinalistError(f"{leaf.state_id}: the target geometry hashes to "
                            f"{target.spec_hash[:12]}, the search recorded "
                            f"{leaf.target_spec_hash[:12]}")
    root = D1S.root_teacher_identity(repo_root)
    for key in ("root_teacher_id", "root_teacher_sha256"):
        if root[key] != getattr(leaf, key):
            raise FinalistError(f"{leaf.state_id}: the teacher binding gives {key}="
                                f"{root[key]}, the search recorded {getattr(leaf, key)}")
    steps = tuple(FixedPathStep(impl_id=s.impl_id, profile_id=s.profile_id,
                                expected_artifact_digest=s.expected_artifact_digest,
                                expected_config_hash=s.expected_config_hash,
                                label=f"{s.kind}({s.profile_id})") for s in leaf.steps)
    return FixedPathSpec(
        path_id=f"s0v_rebuild.{leaf.state_id}", family=target.family, target_spec=target,
        steps=steps, root_repo_id=root["repo_id"], root_revision=root["revision"],
        device=device, seed=leaf.seed,
        position_policy=terms["position_policy"],
        distribution_support=terms["distribution_support"])


def verify_operator_configs(leaves: Sequence[Any], repo_root: Path = REPO) -> dict[str, Any]:
    """Every step's operator inputs against the config_hash the search recorded. $0."""
    from aadistill.initialization.calibration.items import prepare_calibration_items
    from aadistill.initialization.calibration.profiles import get_profile
    from aadistill.initialization.operators.base import get_implementation
    from aadistill.initialization.planning.operator_config import (hashed_operator_config,
                                                                   operator_config_hash)
    checked, items = [], {}
    for leaf in leaves:
        spec = fixed_path_spec(leaf, device="cpu", repo_root=repo_root)
        for rec, step in zip(leaf.steps, spec.steps):
            if step.profile_id not in items:
                prof = get_profile(step.profile_id)
                items[step.profile_id] = prepare_calibration_items(
                    prof.resolve(repo_root), profile_id=prof.qualified_id)
            got = operator_config_hash(hashed_operator_config(
                implementation=get_implementation(step.impl_id),
                policy=spec.position_policy, support=spec.distribution_support,
                items=items[step.profile_id], declared=step.config))
            checked.append({"leaf": leaf.state_id, "index": rec.index, "impl_id": step.impl_id,
                            "profile_id": step.profile_id,
                            "recorded": rec.expected_config_hash, "derived": got,
                            "matches": got == rec.expected_config_hash})
    wrong = [c for c in checked if not c["matches"]]
    if wrong:
        raise FinalistError(f"{len(wrong)} of {len(checked)} steps would run under an "
                            f"operator config the search did not record: {json.dumps(wrong)}")
    return {"n_steps": len(checked), "all_match": True, "steps": checked}


def rebuild_plan(*, journal: Path, selection: Mapping[str, Any],
                 tombstone: Mapping[str, Any], state_ids: Sequence[str],
                 repo_root: Path = REPO) -> dict[str, Any]:
    """The pinned rebuild of `state_ids`, with the three agreement checks. $0."""
    from stages.phase_d1 import replay_specs as R

    states = R.load_states(journal)
    retired = {r["state_id"]: r for r in tombstone["retired"]}
    quality = {d["state_id"]: int(d["quality_order"]) for d in selection["decisions"]}
    leaves = []
    for sid in state_ids:
        row, gone = states.get(sid), retired.get(sid)
        if row is None or gone is None:
            raise FinalistError(f"{sid} is {'not journaled' if row is None else 'not retired'}")
        art = row.get("artifact") or {}
        #: the tombstone is the independent record of what was deleted
        for key in ("artifact_digest", "checkpoint_sha256"):
            if gone.get(key) != row.get(key):
                raise FinalistError(f"{sid}: the tombstone's {key} {gone.get(key)} is not "
                                    f"the journal's {row.get(key)}")
        leaves.append(R.build_leaf(states, {
            "state_id": sid, "quality_position": quality[sid],
            "artifact_digest": row["artifact_digest"],
            "weights_digest": art["weights_digest"],
            "single_shard_sha256": art["single_shard_sha256"],
            "arch_signature": art["arch_signature"]}))
    terms, prereg = _registries(repo_root)
    from stages.phase_d1 import d1_session as D1S
    root = D1S.root_teacher_identity(repo_root)
    root_state = R.derive_root_state(
        leaves, base_config=R.teacher_config(root["repo_id"], root["revision"]),
        family=fixed_path_spec(leaves[0], device="cpu", repo_root=repo_root).family)
    doc = R.describe(leaves, root_state)
    doc.update({
        "schema": "aadistill.stage0.calibration_volume.rebuild_plan/v1",
        "source_run_id": None, "source_session_commit": None,
        "source": {"journal_rows": str(journal), "selection_sha256": selection.get("selection_sha256"),
                   "tombstone_utc": tombstone.get("utc"), "treatment": tombstone.get("treatment")},
        "agreement": {
            "operator_configs": verify_operator_configs(leaves, repo_root),
            "execution": R.verify_execution_config(
                leaves, execution=search_execution(terms, prereg)),
            "root_state": root_state["chosen_candidate"]},
        "spec_hashes": {leaf.state_id: fixed_path_spec(leaf, device="cuda",
                                                       repo_root=repo_root).spec_hash
                        for leaf in leaves}})
    return doc


# --- export to the relay ----------------------------------------------------------

RELAY_PREFIX = "stage0_calvol/finalists"


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    import hashlib
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _stored(info) -> dict[str, Any]:
    lfs = getattr(info, "lfs", None)
    return {"sha256": getattr(lfs, "sha256", None) if lfs else None,
            "blob_id": getattr(info, "blob_id", None), "size": getattr(info, "size", None)}


def export_candidates(candidates: Sequence[Mapping[str, Any]], *, study: str, treatment: str,
                      repo: str, token: str, api=None) -> list[dict[str, Any]]:
    """Upload each candidate's checkpoint files UNCHANGED to `repo` and verify what
    the hub STORED: its LFS sha256 for the shard, its git blob id for small files.

    `candidates`: dicts with `state_id`, `checkpoint_sha256` and `dir` (the
    checkpoint directory). A file already stored with the same content is not
    re-sent, so a replacement session repeats this for free."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    rows = []
    for c in candidates:
        d = Path(c["dir"])
        shards = sorted(d.glob("*.safetensors"))
        if len(shards) != 1 or _sha256(shards[0]) != c["checkpoint_sha256"]:
            raise FinalistError(f"{c['state_id']}: {d} does not hold the shard "
                                f"{c['checkpoint_sha256'][:12]}")
        for f in sorted(p for p in d.iterdir() if p.is_file()):
            path = f"{RELAY_PREFIX}/{study}/{treatment}/{c['state_id']}/{f.name}"
            sha, blob = _sha256(f), _git_blob_sha1(f)

            def stored():
                got = list(api.get_paths_info(repo, [path], repo_type="model"))
                return _stored(got[0]) if got else None

            have = stored()
            if not have or (have["sha256"] or have["blob_id"]) not in (sha, blob):
                api.upload_file(path_or_fileobj=str(f), path_in_repo=path, repo_id=repo,
                                repo_type="model",
                                commit_message=f"stage-0 top-4 candidate {treatment} {c['state_id']}")
                have = stored()
            if not have or (have["sha256"] or have["blob_id"]) not in (sha, blob):
                raise FinalistError(f"{path}: the relay holds {have}, not sha256 {sha}")
            rows.append({"treatment": treatment, "state_id": c["state_id"], "path": path,
                         "name": f.name, "sha256": sha, "bytes": f.stat().st_size,
                         "is_shard": f.suffix == ".safetensors"})
    return rows


# --- the probe sessions' units ----------------------------------------------------

CANDIDATES_DIR = "artifacts/stages/stage-0/calibration_volume/candidates"


def probe_units(manifest: Mapping[str, Any], treatment: str,
                seeds: Mapping[str, Sequence[int]]) -> list[dict[str, Any]]:
    """One unit per (candidate, seed) the extension registers for `treatment`:
    `seeds["top1"]` for the dose's Top-1 (its A100 unit already holds the
    preregistered seed) and `seeds["others"]` for ranks 2..k."""
    files: dict[str, list] = {}
    for f in manifest["files"]:
        if f["treatment"] == treatment:
            files.setdefault(f["state_id"], []).append(
                {"path": f["path"], "name": f["name"], "sha256": f["sha256"], "bytes": f["bytes"]})
    units = []
    for c in sorted(manifest["candidates"][treatment], key=lambda c: c["rank"]):
        got = files.get(c["state_id"]) or []
        shard = [f for f in got if f["name"].endswith(".safetensors")]
        if len(shard) != 1 or shard[0]["sha256"] != c["checkpoint_sha256"]:
            raise FinalistError(f"{treatment} {c['state_id']}: the manifest holds no relay copy "
                                f"of shard {c['checkpoint_sha256'][:12]}")
        for s in seeds["top1"] if c["rank"] == 1 else seeds["others"]:
            units.append({
                "label": f"s0v_top4_{treatment}_r{c['rank']}_{c['state_id'][:8]}_s{s}",
                "treatment": treatment, "rank": c["rank"], "quality_order": c["quality_order"],
                "state_id": c["state_id"], "checkpoint_sha256": c["checkpoint_sha256"],
                "seed": int(s), "path": c.get("path"),
                "candidate_dir": f"{CANDIDATES_DIR}/{treatment}/{c['state_id']}",
                "candidate_files": sorted(got, key=lambda f: f["name"])})
    return units


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--journal", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--tombstone", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    sel = json.loads(args.selection.read_text())
    tomb = json.loads(args.tombstone.read_text())
    fin = distinct_finalists(sel, journal_index(args.journal), args.k)
    kept = set(tomb["kept"])
    todo = [f["state_id"] for f in fin if f["state_id"] not in kept]
    doc = rebuild_plan(journal=args.journal, selection=sel, tombstone=tomb, state_ids=todo)
    doc["treatment"] = args.treatment
    doc["finalists"] = fin
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({"finalists": [(f["rank"], f["quality_order"], f["state_id"][:8]) for f in fin],
                      "rebuild": [s[:8] for s in todo],
                      "agreement": {k: (v if isinstance(v, str) else v.get("all_match"))
                                    for k, v in doc["agreement"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
