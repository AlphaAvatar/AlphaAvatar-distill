#!/usr/bin/env python3
"""Rebuild attempt 3's five selected leaves. Replay only — nothing is decided here.

    PYTHONPATH=src:scripts python scripts/pod/autoinit_c2_replay_driver.py \
        --workdir /workspace/aad/artifacts/autoinit/c2_replay \
        --rate 1.09 --authorized-usd 5.00 --soft-stop-usd 4.00

The Full Joint Search committed its Top-5 and then lost the weights. The
selection is frozen and accepted; this session restores the artifacts behind it
and does nothing else. There is no beam, no ranking, no selection-bearing
evaluation, no control comparison and no behavioural work in this file, and the
evidence record says so in fields a reader can check rather than in prose.

Every step of every path is pinned to the artifact digest attempt 3 recorded for
it — all four steps, not just the leaf. `materialize_fixed_path` raises
`FixedPathDigestMismatch` the moment a realized digest differs, and this driver
treats that as terminal: a deterministic replay that diverges is a scientific
finding, not something to retry.

**Each leaf is secured before the next path starts.** The driver announces a
finished leaf with a marker and a sidecar the moment it exists, and the launcher
pulls it to the durable store while the next path computes. That ordering is the
whole point of the session: attempt 3 did ten hours of work and lost all of it
because nothing left the pod until closeout, and a failure at leaf N here must
leave leaves 1..N-1 already safe.

The leaves go to the development host's out-of-tree store, not to the hub: the
hub's private tier was measured on 2026-09-19 to admit one 1.11 GiB leaf and
refuse the second (largest admitted object 1.756 GiB against 5.55 GiB needed).
See `logs/stages/stage-1/phase_c2_replay/plans/leaf_destination_decision.md`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

RUN_ID = "autoinit.v1.phase_c2.replay"
SUCCESS_MARKER = "C2_REPLAY_ALL_DONE"
FAILURE_MARKER = "C2_REPLAY_FAILED"
LEAF_MARKER = "C2_REPLAY_LEAF_READY"
MISMATCH_MARKER = "C2_REPLAY_DIGEST_MISMATCH"

#: Attempt 3's leaves are single-shard and 1,192,135,096 bytes each. The largest
#: single path's intermediates measured 16.12 GiB; the teacher is 7.29 GiB. The
#: driver refuses to start without room for the worst path plus every leaf it
#: will retain, because discovering that at path 4 wastes the first three.
LEAF_BYTES = 1_192_135_096
WORST_PATH_BYTES = int(16.12 * 2**30)
N_LEAVES = 5


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


#: Where the LAUNCHER reads this driver's markers. The runner decides a
#: session's terminal by tailing the status file, not the driver's stdout: a
#: marker printed only to stdout is invisible to it, and attempt 9 was recorded
#: INCOMPLETE after reconstructing all five leaves exactly, because the runner
#: never saw C2_REPLAY_ALL_DONE and fell back to the exit code. The setup
#: script appends to this same file, which is why SETUP_DONE was visible and
#: nothing after it was.
STATUS_PATH: Path | None = None


def mark(name: str, detail: str = "") -> None:
    line = f"{datetime.now(timezone.utc):%FT%TZ} MARKER:{name}"
    if detail:
        line += f":{detail}"
    print(line, flush=True)
    if STATUS_PATH is not None:
        try:
            with open(STATUS_PATH, "a") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            #: Never fatal. A driver that died because it could not append to a
            #: status file would lose the work the file exists to report.
            print(f"{datetime.now(timezone.utc):%FT%TZ} "
                  f"could not write the marker to {STATUS_PATH}: {exc}",
                  flush=True)


def say(msg: str) -> None:
    print(f"{datetime.now(timezone.utc):%FT%TZ} {msg}", flush=True)


def load_root(spec, device: str, config_overrides=None):
    """The teacher THIS path was pinned to, in the ROOT STATE it was expanded from.

    `FixedPathSpec` records `root_repo_id` and `root_revision` because the root
    is part of the path's identity. Reading the teacher from a module constant
    instead would let a spec pinned to one checkpoint be replayed from another
    whenever the constant moved, and the digests would diverge four steps later
    with nothing in the record pointing at the cause.

    `config_overrides` reproduces the historical root state. The search reused
    ONE teacher object across all 108 expansions, and `DepthCausalKLGreedyV1`
    sets `use_cache = False` on the model it is handed — so once any causal-KL
    DEPTH expansion had run, every later expansion of any path began from a
    mutated root, and that flag is serialized into every descendant's config and
    therefore into its `artifact_digest`. A replay loading a fresh teacher gets
    the hub default and cannot reproduce those paths.

    The overrides are DERIVED per path from attempt 3's own recorded step-0
    config hash (`replay_specs.derive_root_overrides`), not keyed on state ids
    and not asserted here. They reproduce a historical state; they change
    nothing about what any operator computes, and every intermediate and final
    digest is still gated against attempt 3's record.

    Deliberately small and deliberately named: it is the one function a toy
    rehearsal must replace, so everything around it can be executed for real.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        spec.root_repo_id, dtype=torch.bfloat16,
        revision=spec.root_revision).to(device).eval()
    for key, value in (config_overrides or {}).items():
        setattr(model.config, key, value)
    return model


class ReplayDriver:
    """Five fixed paths, each pinned at every step, each secured when it lands."""

    def __init__(self, a) -> None:
        self.a = a
        self.t0 = time.time()
        self.leaves = []
        #: The authoritative set of finished leaves, keyed by state id. It is
        #: NOT derived from `self.ev["leaves"]`: that field is rebuilt from this
        #: one on every save, and reading it back to discover what had completed
        #: made the two mutually dependent — the first version reported 0/2 after
        #: reconstructing and verifying both.
        self.secured: dict[str, dict] = {}
        self.workdir = Path(a.workdir)
        self.leaf_out = Path(a.leaf_dir)
        #: The evidence goes where the COLLECTOR looks, which is not the
        #: workdir. `c2_replay_artifacts.json` patterns it at
        #: `audit/<audit_dirname>/` under the pod's `artifacts/` root, and it is
        #: the session's one REQUIRED artifact — written into the workdir it
        #: would simply not be collected, and a torn-down session would come
        #: home with no record of itself.
        self.audit = Path(a.audit_dir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.leaf_out.mkdir(parents=True, exist_ok=True)
        self.audit.mkdir(parents=True, exist_ok=True)

        self.ev: dict = {
            "schema": "aadistill.autoinit.c2_replay_evidence/v1",
            "_contract": (
                "The replay-only artifact-reconstruction record, REWRITTEN on "
                "every state change and relayed as a whole_file spec for that "
                "reason. It reconstructs artifacts behind a frozen selection; it "
                "produces no scientific decision of its own."),
            "run_id": RUN_ID,
            "started_utc": now(),
            "image_digest": a.image_digest,
            "rate_usd_per_hour": a.rate,
            "authorized_usd": a.authorized_usd,
            "soft_stop_usd": a.soft_stop_usd,
            "already_spent_usd_at_driver_start": a.already_spent_usd,
            #: Checkable claims, not assurances. Every one of these is false for
            #: a session that would need a different authorization.
            "is_full_search_attempt": False,
            "runs_beam_search": False,
            "ranks_or_reranks": False,
            "produces_a_selection": False,
            "runs_selection_bearing_evaluation": False,
            "injects_canonical_control": False,
            "trains_anything": False,
            "measures_behaviour": False,
            "substitutes_candidates": False,
            "reconstructs_selection_sha256": None,
            "source_session_commit": None,
            "stages": [],
            "leaves": [],
        }

    # -- records ------------------------------------------------------------
    def save(self) -> None:
        self.ev["elapsed_minutes"] = round((time.time() - self.t0) / 60.0, 3)
        self.ev["cost_so_far_usd"] = round(
            self.ev["elapsed_minutes"] / 60.0 * self.a.rate, 4)
        out = self.audit / "c2_replay_evidence.json"
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(self.ev, indent=1) + "\n")
        os.replace(tmp, out)

    def record(self, stage: str, ok: bool, detail: dict | None = None) -> None:
        self.ev["stages"].append({
            "stage": stage, "ok": ok, "at": now(),
            "elapsed_minutes": round((time.time() - self.t0) / 60.0, 3),
            **(detail or {})})
        self.save()

    def spend(self) -> float:
        """Everything this session has cost, including the setup it did not run.

        `t0` is when the DRIVER started, which on a pod is well after billing
        began: image pull, environment build, teacher download and the test gate
        all precede it, and setup time in this programme has varied by a factor
        of thirty. A soft stop measured from the driver's own clock would think
        it had the whole budget left while most of it was already spent.
        """
        return (self.a.already_spent_usd
                + (time.time() - self.t0) / 3600.0 * self.a.rate)

    # -- stage A ------------------------------------------------------------
    def bind_identities(self) -> bool:
        """Everything the replay is pinned to, before a single byte is computed."""
        from experiments.calibration import register_builtin_profiles
        from experiments.phase_c2 import replay_specs

        #: EXPLICIT, and first, exactly as every other pod driver does it. The
        #: driver passes no `calibration_items`, so `materialize_fixed_path`
        #: resolves each profile from the registry — and this module's own import
        #: chain registers none. Without this line the session completes setup,
        #: passes every gate, and dies at path 1 step 1 with `no calibration
        #: profile 'calib.domain_balanced@v1'`, on a billing machine. The toy
        #: rehearsal could not see it: its fixture registers a toy profile of its
        #: own, which is what made the branch look covered.
        registered = register_builtin_profiles()
        say(f"  {len(registered)} calibration profile(s) registered")

        binding = replay_specs.source_binding(REPO_ROOT)
        self.ev["source_binding"] = binding
        self.ev["source_session_commit"] = binding["source_session_commit"]
        self.ev["reconstructs_selection_sha256"] = binding["selection_sha256"]
        say(f"  source commit {binding['source_session_commit'][:12]}…, "
            f"selection {binding['selection_sha256'][:12]}…")
        say(f"  operators: {binding['operators']['verdict']}")

        self.leaves = replay_specs.build_replay_leaves(
            REPO_ROOT, device=self.a.device)
        if len(self.leaves) != N_LEAVES:
            raise RuntimeError(
                f"expected {N_LEAVES} leaves to reconstruct, built "
                f"{len(self.leaves)}")
        for leaf in self.leaves:
            pins = len([s for s in leaf.spec.steps
                        if s.expected_artifact_digest])
            if pins != len(leaf.spec.steps):
                raise RuntimeError(
                    f"{leaf.state_id}: {pins} of {len(leaf.spec.steps)} steps "
                    "pinned. Every intermediate must be pinned, not just the "
                    "leaf; a path that agrees only at its end leaves the "
                    "operators behind it unverified.")
            root = dict(leaf.root_config_overrides)
            prov = leaf.root_override_provenance
            say(f"  {leaf.state_id[:12]}… {len(leaf.spec.steps)} steps, all "
                f"pinned → {leaf.artifact_digest[:12]}…; root "
                f"{root or 'hub default'}"
                + (" (ambiguous, either reproduces)"
                   if prov.get("root_state_is_ambiguous") else ""))

        #: Headroom for the worst single path plus every leaf retained on the
        #: pod. Measured against the workdir's filesystem, which is where both
        #: land.
        free = shutil.disk_usage(self.workdir).free
        need = WORST_PATH_BYTES + N_LEAVES * LEAF_BYTES
        say(f"  disk: {free / 2**30:.1f} GiB free, need {need / 2**30:.1f} GiB "
            f"(worst path {WORST_PATH_BYTES / 2**30:.1f} + {N_LEAVES} leaves)")
        if free < need:
            raise RuntimeError(
                f"{free / 2**30:.1f} GiB free at {self.workdir} but the worst "
                f"path plus {N_LEAVES} retained leaves needs "
                f"{need / 2**30:.1f} GiB. Provision the disk rather than "
                "discovering this at path 4.")
        self.record("bind_identities", True, {
            "leaves": len(self.leaves),
            "free_gib": round(free / 2**30, 2),
            "required_gib": round(need / 2**30, 2)})
        return True

    # -- stage B ------------------------------------------------------------
    def reconstruct(self) -> bool:
        """Materialize each pinned path; secure each leaf the moment it exists."""
        from aadistill.initialization.planning.fixed_path import (
            FixedPathDigestMismatch, materialize_fixed_path,
        )
        from aadistill.initialization.specs.arch import get_adapter

        adapter = get_adapter("qwen3")

        for index, leaf in enumerate(self.leaves, start=1):
            budget_left = self.a.soft_stop_usd - self.spend()
            #: ADMISSION CONTROL, not a trip-wire. The path's own bounded cost
            #: comes from attempt 3's telemetry, priced at this session's rate,
            #: and a path is started only if the budget can see it FINISH.
            #: Checking merely that something remains would let a 30-minute path
            #: begin on two minutes' worth of budget and overshoot by the whole
            #: difference — which is how a soft stop becomes advisory.
            path_cost = leaf.bounded_minutes / 60.0 * self.a.rate
            say(f"path {index}/{N_LEAVES} {leaf.state_id[:12]}… "
                f"${self.spend():.2f} spent, ${budget_left:.2f} to the soft stop, "
                f"this path bounded at {leaf.bounded_minutes:.1f} min "
                f"(${path_cost:.2f})")
            if path_cost > budget_left:
                #: Not a failure: leaves 1..index-1 are reconstructed and
                #: secured. Stopping here preserves them and the budget.
                say(f"  cannot afford to FINISH this path (${path_cost:.2f} "
                    f"needed, ${budget_left:.2f} left); stopping with "
                    f"{len(self.secured)} leaves already secured")
                self.record("soft_stop", True, {
                    "reconstructed": index - 1,
                    "remaining": N_LEAVES - index + 1,
                    "path_bounded_usd": round(path_cost, 4),
                    "budget_left_usd": round(budget_left, 4)})
                self.ev["stopped_at_soft_stop"] = True
                return False

            #: Reloaded per path, not shared. Operators mutate the module they
            #: are given, so a second path starting from the first path's root
            #: would begin from an already-compressed model and diverge at step
            #: one. After the first load the weights are in the local cache, so
            #: this is a disk read; the alternative is wrong.
            def root_loader(_spec=leaf.spec, _ov=dict(leaf.root_config_overrides)):
                return load_root(_spec, self.a.device, _ov)

            path_dir = self.workdir / "paths" / leaf.state_id
            path_dir.mkdir(parents=True, exist_ok=True)
            steps_seen: list[dict] = []

            def on_step(result, _seen=steps_seen, _leaf=leaf):
                _seen.append(result.as_dict())
                say(f"    step {result.index} {result.impl_id} "
                    f"→ {result.identity.artifact_digest[:12]}… "
                    f"(pinned {str(result.digest_expected)[:12]}…, "
                    f"{result.seconds:.0f}s)")
                self.ev["leaves"] = self._leaf_records(partial=(_leaf, _seen))
                self.save()

            t0 = time.time()
            try:
                results = materialize_fixed_path(
                    leaf.spec, adapter=adapter, root_loader=root_loader,
                    workdir=path_dir, repo_root=REPO_ROOT, on_step=on_step)
            except FixedPathDigestMismatch as exc:
                #: TERMINAL. A deterministic replay that diverges is a finding.
                mark(MISMATCH_MARKER, leaf.state_id)
                self.record("digest_mismatch", False, {
                    "state_id": leaf.state_id,
                    "step_index": exc.step_index, "label": exc.label,
                    "expected": exc.expected, "actual": exc.actual,
                    "evidence": exc.evidence,
                    "steps_completed": steps_seen,
                    "verdict": (
                        "replay mismatch. Not retried and not substituted: the "
                        "path is deterministic, so the same inputs would "
                        "diverge the same way. Refer to review.")})
                self.ev["digest_mismatch"] = {
                    "state_id": leaf.state_id, "step_index": exc.step_index,
                    "expected": exc.expected, "actual": exc.actual}
                return False

            final = results[-1]
            self._assert_identity(leaf, final)
            secured = self._announce(leaf, final, path_dir,
                                     seconds=time.time() - t0, steps=steps_seen)
            self.secured[leaf.state_id] = secured
            self.ev["leaves"] = self._leaf_records()
            self.record("leaf_reconstructed", True, secured)

            #: The intermediates are worth nothing once the leaf matches: they
            #: are pinned to digests already recorded, and the next path needs
            #: the room. The leaf itself is NOT here — it was copied out first.
            shutil.rmtree(path_dir, ignore_errors=True)

        return True

    # -- helpers ------------------------------------------------------------
    def _assert_identity(self, leaf, final) -> None:
        """Attempt 3's identity, field by field. The digest gate is not enough.

        `materialize_fixed_path` already refused anything whose artifact digest
        differed, so this cannot fail on that field. It is here for the fields
        the gate does not cover — the shard hash, the parameter count, the
        architecture signature, and the absence of a tokenizer — because a
        checkpoint that matches a digest and ships a tokenizer is a different
        artifact from the one attempt 3 recorded, and this programme has twice
        shipped checkpoints whose identity gates all passed and which could not
        be used.
        """
        got = final.identity
        mismatches = []
        for field, expected, actual in (
                ("artifact_digest", leaf.artifact_digest, got.artifact_digest),
                ("weights_digest", leaf.weights_digest, got.weights_digest),
                ("single_shard_sha256", leaf.single_shard_sha256,
                 got.single_shard_sha256),
                ("arch_signature", leaf.arch_signature, got.arch_signature),
                ("num_parameters", leaf.num_parameters, got.num_parameters)):
            if expected != actual:
                mismatches.append(f"{field}: expected {expected!r}, got {actual!r}")
        if got.tokenizer_sha256 is not None:
            mismatches.append(
                f"tokenizer_sha256: expected None (weight-only), got "
                f"{got.tokenizer_sha256!r}. Attempt 3's leaves carry weights "
                "only; a tokenizer here is contamination, not a bonus.")
        if mismatches:
            raise RuntimeError(
                f"{leaf.state_id}: reconstructed leaf does not carry attempt 3's "
                "identity — " + "; ".join(mismatches) + ". STOP.")

    def _announce(self, leaf, final, path_dir: Path, *, seconds: float,
                  steps: list[dict]) -> dict:
        """Copy the leaf clear of the path, write its sidecar, raise the marker.

        The copy happens BEFORE the marker and before the intermediates are
        deleted, so the object the launcher is told about is already at a stable
        path that the cleanup below cannot touch.
        """
        dest = self.leaf_out / leaf.state_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(final.checkpoint_path, dest)
        sidecar = {
            "schema": "aadistill.autoinit.c2_replay_leaf/v1",
            "state_id": leaf.state_id,
            "path": leaf.path_label,
            "lineage": leaf.lineage,
            "reconstructed_utc": now(),
            "seconds": round(seconds, 1),
            "source_session_commit": self.ev["source_session_commit"],
            "reconstructs_selection_sha256":
                self.ev["reconstructs_selection_sha256"],
            "identity": {
                "artifact_digest": leaf.artifact_digest,
                "weights_digest": leaf.weights_digest,
                "single_shard_sha256": leaf.single_shard_sha256,
                "arch_signature": leaf.arch_signature,
                "num_parameters": leaf.num_parameters,
                "tokenizer_sha256": None,
            },
            "step_digests": list(leaf.step_digests),
            "steps": steps,
            "verified_against": "attempt 3 stage1_selection.json",
        }
        (dest / "replay_leaf.json").write_text(json.dumps(sidecar, indent=1) + "\n")
        say(f"  leaf {leaf.state_id[:12]}… reconstructed and identical to "
            f"attempt 3 ({seconds / 60:.1f} min)")
        #: The launcher watches for this and pulls while the next path runs.
        mark(LEAF_MARKER, leaf.state_id)
        return {"state_id": leaf.state_id, "seconds": round(seconds, 1),
                "dest": str(dest), "identity_matches_attempt3": True}

    def _leaf_records(self, partial=None) -> list[dict]:
        out = []
        for leaf in self.leaves:
            if leaf.state_id in self.secured:
                out.append(self.secured[leaf.state_id])
                continue
            row = {"state_id": leaf.state_id, "path": leaf.path_label,
                   "identity_matches_attempt3": False}
            if partial and partial[0].state_id == leaf.state_id:
                row["steps_completed"] = len(partial[1])
                row["steps_total"] = len(leaf.spec.steps)
            out.append(row)
        return out

    # -- run ----------------------------------------------------------------
    def run(self) -> int:
        stages = (("bind_identities", self.bind_identities),
                  ("reconstruct", self.reconstruct))
        for name, fn in stages:
            say(f"stage {name}")
            try:
                ok = fn()
            except Exception as exc:                            # noqa: BLE001
                #: Broad on purpose: a paid session must record WHY it stopped,
                #: and an uncaught traceback on a pod is evidence nobody can read
                #: from the artifacts that come home.
                self.record(name, False,
                            {"error": f"{type(exc).__name__}: {exc}"})
                self.finish(False, failed=name)
                return 1
            if not ok:
                self.finish(False, failed=name)
                return 1
        self.finish(True, failed=None)
        return 0

    def finish(self, success: bool, *, failed: str | None) -> None:
        reconstructed = list(self.secured.values())
        self.ev["leaves"] = self._leaf_records() if self.leaves else []
        self.ev["successful"] = success
        self.ev["failed_stage"] = failed
        self.ev["n_reconstructed"] = len(reconstructed)
        self.ev["outcome"] = "ALL_DONE" if success else "FAILED"
        self.ev["_partial_is_not_waste"] = (
            "each leaf is copied clear and announced before the next path "
            "starts, so a failure at leaf N leaves 1..N-1 reconstructed and "
            "already pulled. A FAILED outcome here does not invalidate them.")
        self.ev["followon_reachable_from_this_driver"] = False
        self.save()
        say(f"{len(reconstructed)}/{N_LEAVES} leaves reconstructed, "
            f"${self.spend():.2f} spent")
        mark(SUCCESS_MARKER if success else FAILURE_MARKER)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--audit-dir", default=None,
                    help="where the session evidence is written; must be the "
                         "directory the artifact spec patterns. Defaults to "
                         "<repo>/artifacts/audit/autoinit_c2_replay.")
    ap.add_argument("--leaf-dir", default=None,
                    help="where finished leaves are copied for collection; "
                         "defaults to <workdir>/leaves")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rate", type=float, required=True,
                    help="the accepted provider rate, USD/hour")
    ap.add_argument("--authorized-usd", type=float, required=True)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--status-path", default=None,
                    help="the session status file the LAUNCHER tails. Markers "
                         "are appended here as well as printed; the runner "
                         "reads this file and nothing else to decide a "
                         "session's terminal.")
    ap.add_argument("--image-digest", default="")
    ap.add_argument("--already-spent-usd", type=float, default=0.0,
                    help="what this pod had already billed when the driver "
                         "started -- setup, image pull, teacher download. The "
                         "launcher knows it; the driver cannot observe it.")
    return ap


def main() -> int:
    global STATUS_PATH

    args = build_parser().parse_args()
    if args.status_path:
        STATUS_PATH = Path(args.status_path)
    if args.leaf_dir is None:
        args.leaf_dir = str(Path(args.workdir) / "leaves")
    if args.audit_dir is None:
        args.audit_dir = str(REPO_ROOT / "artifacts/audit/autoinit_c2_replay")
    if args.soft_stop_usd >= args.authorized_usd:
        raise SystemExit(
            f"soft stop ${args.soft_stop_usd:.2f} must leave room under the "
            f"authorized ${args.authorized_usd:.2f} for teardown")
    return ReplayDriver(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
