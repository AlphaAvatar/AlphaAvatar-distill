#!/usr/bin/env python3
"""Dev-box orchestrator for AutoInitializer Phase A.

    PYTHONPATH=src setsid nohup python -u \
        scripts/pod/autoinit_phase_a_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> < /dev/null &

A **subclass** of the micro-preflight launcher, like the continuation before it.
Everything verified live — the detached start, the independent watchdog, the log
relay, the four-threshold budget, the artifact gate and the provider-confirmed
teardown — is inherited unchanged. What differs is what this session *is*.

**This is the long one.** Nine recovery probes at ~62 min each, plus a beam
search, is 12-17 GPU-hours against a project whose longest successful session so
far was 3.6 hours and four of whose last eight attempts hit an infrastructure
event. Two consequences are designed in rather than hoped for:

*Resume is per probe.* The driver journals each probe the moment it is scored and
restores it only when the student digest, seed and evaluation protocol still
match. A redrawn pod re-runs the search cheaply from its own journal and repeats
no completed probe.

*Retention is decided by measurement, not by preference.* The relay reports
`usedStorage` **91.54 GiB** against an inferred 100 GB (93.13 GiB) limit — 1.60
GiB of headroom — and five bf16 leaves are 5.61 GiB, so relay staging of the
searched leaves is **off**. All five leaves stay materialized on the pod's
container disk through rung 1 and the materialization of the sa survivor
decision (`BeamSearch` releases weights only for pruned *intermediates*, never
for a complete leaf), the survivors and the control stay through rung 2 and any
conditional rung 3, and only the finalists are pulled to the dev box. Rejected
leaves are neither fetched nor deleted: `leaf_retention.json` carries their
digest, lineage, sa evidence and rejection reason. Full accounting in
`logs/autoinit_phase_a_storage.md`.

Phase A is a terminus. This launcher starts one driver, that driver has no stage
6, and the authorization has no code path to a follow-on experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.phase_a import (  # noqa: E402
    PHASE_A_PLAN_V1, PHASE_A_SCOPE, PhaseAAuthorization,
)
from aadistill.infrastructure.budget import Phase, StepTime, plan_session  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "autoinit_preflight_launch",
    REPO_ROOT / "scripts/pod/autoinit_preflight_launch.py")
_preflight = importlib.util.module_from_spec(_spec)
sys.modules["autoinit_preflight_launch"] = _preflight
_spec.loader.exec_module(_preflight)

WS = _preflight.WS
REPO = _preflight.REPO
STATUS = f"{WS}/autoinit_phase_a.status"
RUN_LOG = f"{WS}/autoinit_phase_a_run.log"
AUTH_PATH = "logs/autoinit_phase_a_authorization.json"
#: Dev-box-only inputs the pod cannot fetch from git: the two frozen assets and
#: the frozen science plan the driver binds against.
LOCAL_ASSETS = ("artifacts/stage1/state_eval_v1",
                "artifacts/stage3/recovery_search_v2")
#: Measured, not estimated. 61.55 min end-to-end for an 0.86M probe (attempt 4,
#: two arms) and 9.82 min conservative for one battery (half of the 19.65 min
#: Stage 3 took for two, so it carries engine load and scoring).
PROBE_TRAIN_MINUTES = 61.55
#: The CONSERVATIVE battery figure — half of the 19.65 min Stage 3 took for two
#: controls, so it carries engine load, scoring and materialization. The marginal
#: 8.43 min is what one more battery costs once the engine is warm, and pricing
#: from the marginal figure would under-book the first one.
PROBE_BATTERY_MINUTES = 9.82
#: The search: 39-56 states at ~2.6 min evaluation + ~8 s statistics each, plus
#: the operator build, which is the one term still unmeasured and is why this is
#: generous rather than tight.
SEARCH_MINUTES = 180.0
#: Setup allowance, measured across the 2026-08-14/15 runs at ~10.4 min and
#: priced at 11 so the number is not the optimistic one.
SETUP_MINUTES = 11.0


def lineage_from_authorized_base(repo_root, base: str | None, commit: str,
                                 auth_path: str) -> dict:
    """Is `commit` the authorized base plus the authorization artifact, and nothing else?

    The harness digest proves the *declared harness files* did not move, and the
    auth-blob check proves the driver will load this exact grant. Neither says
    anything about the rest of the tree. `authorized_session_commit` is
    necessarily issued against the clean PRE-authorization HEAD — the artifact
    cannot be committed before it exists — so the commit the pod actually checks
    out is always a later one, and until now nothing constrained what else rode
    along in that gap.

    Closing it needs two facts, both from git:

    * the final commit **descends from** the authorized base, so it is not some
      unrelated tip that happens to carry a matching harness;
    * the only path that differs between them is the authorization artifact.

    Returns a record rather than a bool so the launcher can log exactly what it
    saw, including on the paths that refuse.
    """
    out = {"authorized_base": base, "session_commit": commit,
           "descends_from_base": None, "changed_paths": None,
           "unexpected_paths": None, "ok": False, "reason": ""}
    if not base:
        out["reason"] = ("the authorization declares no authorized_session_commit, "
                         "so there is no base to constrain the tree against")
        return out
    known = subprocess.run(["git", "cat-file", "-e", f"{base}^{{commit}}"],
                           capture_output=True, cwd=repo_root)
    if known.returncode != 0:
        out["reason"] = f"the authorized base {base} is not a commit in this repository"
        return out
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", base, commit],
                         capture_output=True, cwd=repo_root)
    out["descends_from_base"] = anc.returncode == 0
    if not out["descends_from_base"]:
        out["reason"] = (f"{commit} does not descend from the authorized base "
                         f"{base}; it is a different line of history")
        return out
    diff = subprocess.run(["git", "diff", "--name-only", base, commit],
                          capture_output=True, text=True, cwd=repo_root)
    if diff.returncode != 0:
        out["reason"] = f"could not diff {base}..{commit}: {diff.stderr.strip()[:200]}"
        return out
    changed = [p for p in diff.stdout.split("\n") if p.strip()]
    out["changed_paths"] = changed
    out["unexpected_paths"] = [p for p in changed if p != auth_path]
    if out["unexpected_paths"]:
        out["reason"] = (
            f"{len(out['unexpected_paths'])} path(s) other than {auth_path} "
            f"changed between the authorized base and the session commit: "
            f"{out['unexpected_paths'][:8]}")
        return out
    out["ok"] = True
    out["reason"] = (f"only {auth_path} differs from the authorized base"
                     if changed else "identical to the authorized base")
    return out


class PhaseA(_preflight.Preflight):
    """The preflight launcher, retargeted at the search it was built to precede."""

    def __init__(self, a):
        super().__init__(a)
        self.ev["schema"] = "aadistill.autoinit.phase_a_session/v1"
        self.ev["phase_a_session_plan_hash"] = PHASE_A_PLAN_V1.plan_hash
        self.ev["scope"] = PHASE_A_SCOPE.as_dict()
        self.ev["retrains_permanent_controls"] = False
        # Inherited from the preflight, where they mean "Phase A did not start".
        # Here Phase A *is* the session, so the honest fields are about what
        # comes after it — which is nothing.
        self.ev.pop("phase_a_launched", None)
        self.ev.pop("phase_a_reachable_from_this_launcher", None)
        self.ev["followon_started"] = False
        self.ev["followon_reachable_from_this_launcher"] = False

    # -- what this session owns -------------------------------------------
    audit_dirname = "autoinit_phase_a"
    evidence_filename = "phase_a_evidence.json"
    archive_basename = "phase_a_artifacts.tar.gz"
    spec_success = "configs/autoinit/phase_a_artifacts.json"
    spec_failed = "configs/autoinit/phase_a_artifacts_failed.json"
    failure_markers = ("PHASE_A_FAILED", "PHASE_A_INCOMPLETE")
    incomplete_markers = ("PHASE_A_INCOMPLETE",)
    failure_note = ("a blocking stage failed — collecting evidence, then tearing "
                    "down. Completed probes are journalled and the permanent "
                    "controls are inputs here and untouched.")
    #: `leaf_retention.json` is fetched BEFORE fetch_products runs, because
    #: `finalists_to_fetch` reads it to decide which initializations come home.
    report_names = ("phase_a_evidence.json", "attested_evaluation_protocol.json",
                    "search_result.json", "rung1_selection.json",
                    "leaf_retention.json",
                    "rung2_selection.json", "phase_a_result.json")
    job_id = "autoinit_phase_a_driver"

    def event_streams(self) -> tuple[str, ...]:
        """The probe training streams a torn-down session may have left mid-write.

        Nine probes, so this is derived rather than listed: naming a fixed set
        would either miss a rung-2 probe or demand a rung-3 one that correctly
        never ran.
        """
        journalled = sorted((self.scr / "relay").glob("*.train_log.jsonl"))
        return tuple(f"artifacts/stage3/phase_a/{p.name.split('.')[0]}/train_log.jsonl"
                     for p in journalled)

    # -- this session's own authorization ----------------------------------
    def session_auth_path(self) -> str:
        return AUTH_PATH

    def session_plan_hash(self) -> str:
        return PHASE_A_PLAN_V1.plan_hash

    def setup_env(self) -> dict[str, str]:
        """Setup must load the Phase-A authorization TYPE, not the narrow one.

        The science plan is not passed: setup has no executing plan to compare
        it against, so it is bound in the driver's Stage 0 against the rebuilt
        object instead.
        """
        return {"SESSION_KIND": "phase_a"}

    # -- budget: a search, then nine probes --------------------------------
    def make_plan(self) -> bool:
        # The conditional tie-break rung IS given headroom, even though it
        # usually does not run. If it were left out, a legitimately-triggered
        # seed-sc rung would be killed by the watchdog at the threshold — and an
        # unused leash costs nothing, because the pod is torn down on completion
        # rather than at the threshold. Same reasoning as the continuation's
        # 24-minute characterization allowance.
        priced = self.a.rung1_probes + self.a.rung2_probes + self.a.tie_break_probes
        phases = (Phase("stage0_attestation_and_binding", 8.0),
                  Phase("stage1_beam_search", self.a.search_minutes),
                  Phase("stage5_selection_and_report", 5.0),
                  Phase("artifact_manifest_and_verify", 8.0),
                  Phase("artifact_synchronization", 10.0))
        self.plan = plan_session(
            price_per_hour=self.a.max_price,
            authorized_usd=self.auth.hard_cap_usd,
            # The probes ARE the training: each is one arm of 1023 steps. Priced
            # through the step-time model so the budget moves with the measured
            # step time rather than with a transcribed total.
            arms=priced, steps_per_arm=1023,
            step_time=StepTime(
                self.a.probe_train_minutes * 60 / 1023,
                "measured end-to-end: 61.55 min per 1023-step probe, on two arms"),
            below_floor_reason=(
                "the 4.15 s/step floor is E6b's IN-LOOP figure. Attempt 4 "
                "measured 3.15 s/step in-loop for this exact model, rung and "
                "card, and 61.55 min end-to-end for the whole 1023-step probe "
                "including its periodic evaluation and checkpointing, on two "
                "arms. The rate used here is that end-to-end figure divided by "
                "the step count, so it is slower than the in-loop measurement "
                "rather than an optimistic version of it."),
            setup_minutes=self.a.setup_minutes, other_phases=phases,
            eval_minutes_per_arm=self.a.probe_battery_minutes,
            transfer_minutes=self.a.transfer_minutes,
            contingency_fraction=0.10, artifact_recovery_reserve_minutes=20.0)
        self.ev["budget_plan"] = self.plan.as_dict()
        self.ev["priced_probes"] = {
            "rung1": self.a.rung1_probes, "rung2": self.a.rung2_probes,
            "tie_break_conditional": self.a.tie_break_probes,
            "total_priced": priced,
            "expected_without_tie_break_usd": round(
                self.plan.expected_usd
                - self.a.tie_break_probes
                * (self.a.probe_train_minutes + self.a.probe_battery_minutes)
                / 60 * self.a.max_price, 4),
            "note": (
                "the tie-break rung is priced so it CAN run, not because it is "
                "expected to. It runs only for finalists inside the "
                "preregistered equivalence interval."),
            "why_this_exceeds_the_repricing_doc": (
                "logs/autoinit_phase_a_repricing.md priced search + probes only. "
                "This plan additionally carries setup, Stage-0 attestation, "
                "selection, artifact manifest/verify, synchronization, transfer, "
                "a 10% contingency and a 20-minute artifact-recovery reserve — "
                "all of which are session time that is really spent."),
        }
        try:
            self.auth.require_within_cap(self.plan.hard_terminate_usd,
                                         what="planned hard threshold")
            self.auth.require_within_launch_limit(self.plan.hard_terminate_usd,
                                                  what="planned hard threshold")
        except _preflight.AuthorizationError as exc:
            self.say(f"ABORT: {exc}")
            return False
        self.say(f"budget: expected {self.plan.expected_minutes:.0f} min "
                 f"${self.plan.expected_usd:.2f} · soft "
                 f"${self.plan.soft_stop_usd:.2f} · hard "
                 f"${self.plan.hard_terminate_usd:.2f} "
                 f"(authorized ${self.auth.hard_cap_usd:.2f})")
        return self.check_gpu_offered()

    # -- the commit the pod will actually run ------------------------------
    def verify_session_commit(self) -> bool:
        """The harness at `--session-commit` must be the authorized one.

        Same check the continuation added after attempt 5 died on a stale
        binding: the pod does not run the dev box's working tree, it clones a
        bundle and checks out this commit.
        """
        commit = self.a.session_commit
        entries, missing = [], []
        for rel in sorted(self.auth.harness_source_files):
            blob = subprocess.run(["git", "show", f"{commit}:{rel}"],
                                  capture_output=True, cwd=REPO_ROOT)
            if blob.returncode != 0:
                missing.append(rel)
                continue
            entries.append({"path": rel,
                            "sha256": hashlib.sha256(blob.stdout).hexdigest()})
        if missing:
            self.say(f"ABORT at $0: {commit} does not contain {missing}")
            return False
        digest = hashlib.sha256(
            "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()
        ).hexdigest()
        auth_blob = subprocess.run(["git", "show", f"{commit}:{AUTH_PATH}"],
                                   capture_output=True, cwd=REPO_ROOT)
        carries_auth = (auth_blob.returncode == 0
                        and auth_blob.stdout == (REPO_ROOT / AUTH_PATH).read_bytes())
        lineage = lineage_from_authorized_base(
            REPO_ROOT, self.auth.authorized_session_commit, commit, AUTH_PATH)
        self.ev["session_commit_check"] = {
            "session_commit": commit,
            "authorized_session_commit": self.auth.authorized_session_commit,
            "harness_digest_at_commit": digest,
            "authorized_harness_digest": self.auth.harness_source_digest,
            "harness_matches": digest == self.auth.harness_source_digest,
            "commit_carries_this_authorization": carries_auth,
            "lineage": lineage,
        }
        if digest != self.auth.harness_source_digest:
            self.say(f"ABORT at $0: the harness at {commit} digests to {digest}, "
                     f"authorized {self.auth.harness_source_digest}. The pod "
                     "would run code this authorization was not granted against.")
            return False
        if not carries_auth:
            self.say(f"ABORT at $0: {commit} does not carry this exact "
                     f"{AUTH_PATH}; the driver would load a different "
                     "authorization, or none.")
            return False
        # The two checks above cover the declared harness and the grant itself.
        # Neither constrains the REST of the tree between the authorized base
        # and the commit the pod checks out, and that gap is real: the base is
        # necessarily the clean pre-authorization HEAD, so the bundle commit is
        # always later. Anything else that rode along would run unreviewed.
        if not lineage["ok"]:
            self.say(f"ABORT at $0: {lineage['reason']}")
            return False
        self.say(f"session commit {commit[:12]} verified: harness digests to "
                 f"{digest[:12]}…, carries the authorization, and {lineage['reason']}")
        return True

    # -- precheck: everything this session reads, checked at $0 ------------
    def relay_precheck(self) -> bool:
        if not self.verify_session_commit():
            return False
        # The frozen science plan must exist and must be the one the
        # authorization names, or the driver's Stage 0 fails after setup is paid.
        frozen = REPO_ROOT / "logs/autoinit_phase_a_recovery_plan_frozen.json"
        if not frozen.is_file():
            self.say(f"ABORT at $0: no frozen science plan at {frozen}; "
                     "assert_preregistered would have nothing to bind against")
            return False
        frozen_hash = json.loads(frozen.read_text()).get("plan_hash")
        if frozen_hash != self.auth.science_plan_hash:
            self.say(f"ABORT at $0: the frozen plan hashes to {frozen_hash} but "
                     f"the authorization names {self.auth.science_plan_hash}")
            return False
        try:
            from huggingface_hub import HfApi
            present = set(HfApi().list_repo_files(self.a.relay_repo,
                                                  repo_type="model"))
        except Exception as exc:                                  # noqa: BLE001
            self.say(f"ABORT: cannot list the relay: {exc!r}"[:200])
            return False
        need = ["stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors",
                "stage3_recovery_corpus_v2/ladder_uniform/blocks.npz"]
        missing = [f for f in need if f not in present]
        local_missing = [p for p in LOCAL_ASSETS if not (REPO_ROOT / p).is_dir()]
        self.ev["precheck"] = {"relay_needed": need, "relay_missing": missing,
                               "local_assets": list(LOCAL_ASSETS),
                               "local_missing": local_missing,
                               "frozen_plan_hash": frozen_hash}
        if missing or local_missing:
            self.say(f"ABORT at $0: relay missing {missing}, "
                     f"local missing {local_missing}")
            return False
        self.say(f"precheck OK: {len(need)} relay inputs, {len(LOCAL_ASSETS)} "
                 f"local assets, frozen plan {frozen_hash[:12]}…")
        return True

    # -- products: push from the pod, do not pull across the uplink --------
    def fetch_products(self, host: str, target, stage2_passed: bool) -> list:
        """Get the irreplaceable artifacts off the pod before it is deleted.

        **The finalists, not the winner.** If the run ends in
        `unresolved_equivalence` there is no winner and BOTH tied candidates are
        the result; fetching only a winner would throw away the finding. So this
        fetches whatever advanced past rung 1, which is the survivors plus the
        control, and falls back to the winner alone only if the retention record
        is missing.

        **Rejected leaves are not fetched, and not deleted either.** They stay on
        the pod until it is destroyed, and what survives them is
        `leaf_retention.json`: digest, lineage, sa evidence and rejection reason.
        See `logs/autoinit_phase_a_storage.md` for why permanent retention is
        limited to the finalists — the relay has 1.60 GiB of headroom and five
        leaves are 5.61 GiB.
        """
        fetched: list = []
        if not stage2_passed:
            return fetched
        if self.a.stage_leaves_to_relay:
            rc = target.run(
                f"cd {REPO} && HF_TOKEN=\"$(cat {WS}/hf/token)\" "
                f"PYTHONPATH={REPO}/src /opt/train/bin/python "
                f"{REPO}/scripts/pod/collect_artifacts.py stage-leaves "
                f"--search-dir {REPO}/artifacts/autoinit/phase_a_search "
                f"--repo {self.a.relay_repo} --prefix phase_a_leaves",
                timeout=5400)
            fetched.append({"artifact": "searched_leaves", "route": "relay",
                            "rc": rc.returncode,
                            "tail": (rc.stdout or "")[-400:]})
            self.say(f"  searched leaves -> relay: rc={rc.returncode}")

        for state_id in self.finalists_to_fetch():
            dest = Path(self.a.ckpt_store) / "phase_a" / state_id
            dest.parent.mkdir(parents=True, exist_ok=True)
            rc = subprocess.run(
                ["timeout", f"{self.a.ckpt_fetch_limit_min}m", "scp", "-r",
                 "-P", str(target.port), "-o", "StrictHostKeyChecking=no",
                 "-o", "UserKnownHostsFile=/dev/null",
                 f"root@{host}:{REPO}/artifacts/autoinit/phase_a_search/states/{state_id}",
                 str(dest)], capture_output=True, timeout=None)
            size = sum(f.stat().st_size for f in dest.rglob("*")
                       if f.is_file()) if dest.exists() else 0
            fetched.append({"artifact": "finalist_initialization",
                            "state_id": state_id, "rc": rc.returncode,
                            "bytes": size, "dest": str(dest)})
            self.say(f"  finalist {state_id}: rc={rc.returncode}, "
                     f"{size / 2**30:.2f} GiB -> {dest}")
        if not fetched:
            self.say("  nothing to fetch: rung 1 produced no retention record "
                     "and no winner")
        return fetched

    def finalists_to_fetch(self) -> list[str]:
        """Which initializations earn permanent off-pod retention.

        The control is excluded: it is the retained canonical checkpoint, it
        already exists at `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`, and
        re-fetching it would spend transfer time on a byte we already hold.
        """
        if not self.a.fetch_finalists:
            return []
        store = self.scr / "store"
        retention = store / "leaf_retention.json"
        if retention.is_file():
            entries = json.loads(retention.read_text())["entries"]
            return [e["canonical_id"] for e in entries
                    if e.get("permanent_checkpoint_retained")
                    and not e.get("is_control")]
        result = store / "phase_a_result.json"
        if result.is_file():
            winner = json.loads(result.read_text()).get("winner")
            if winner:
                self.say("  no retention record; falling back to the winner alone")
                return [winner]
        return []

    # -- the driver this session runs --------------------------------------
    def driver_command(self) -> str:
        return (f"/opt/train/bin/python "
                f"{REPO}/scripts/pod/autoinit_phase_a_driver.py "
                f"--stage all --image-digest '{self.image_digest}' "
                f"--rate {self.price or self.a.max_price} "
                f"--spent-usd {self.usd():.4f} "
                f"--soft-stop-usd {self.plan.soft_stop_usd:.4f} "
                f"--authorized-usd {self.auth.hard_cap_usd:.4f} "
                f"--search-minutes {self.a.search_minutes} "
                f"--probe-train-minutes {self.a.probe_train_minutes} "
                f"--probe-battery-minutes {self.a.probe_battery_minutes}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=0.99)
    #: 106 GiB peak working storage from the priced search, plus the probe
    #: checkpoints. The preregistration says provision at least 150.
    ap.add_argument("--disk-gb", type=int, default=200)
    ap.add_argument("--rung1-probes", type=int, default=6)
    ap.add_argument("--rung2-probes", type=int, default=3)
    #: Conditional by construction; priced so it can run, not because it will.
    ap.add_argument("--tie-break-probes", type=int, default=3)
    ap.add_argument("--search-minutes", type=float, default=SEARCH_MINUTES)
    ap.add_argument("--probe-train-minutes", type=float, default=PROBE_TRAIN_MINUTES)
    ap.add_argument("--probe-battery-minutes", type=float,
                    default=PROBE_BATTERY_MINUTES)
    ap.add_argument("--setup-minutes", type=float, default=SETUP_MINUTES)
    ap.add_argument("--transfer-minutes", type=float, default=6.0)
    #: OFF, and measured rather than assumed. The relay reports usedStorage
    #: 91.54 GiB against an inferred 100 GB (93.13 GiB) limit — 1.60 GiB of
    #: headroom — and five bf16 leaves are 5.61 GiB. Staging them would fail on
    #: quota partway through, and deletion on this relay is a maintainer
    #: decision, not a launcher's. See logs/autoinit_phase_a_storage.md.
    ap.add_argument("--stage-leaves-to-relay", action="store_true", default=False)
    ap.add_argument("--no-stage-leaves-to-relay", dest="stage_leaves_to_relay",
                    action="store_false")
    ap.add_argument("--fetch-finalists", action="store_true", default=True)
    ap.add_argument("--no-fetch-finalists", dest="fetch_finalists",
                    action="store_false")
    ap.add_argument("--teacher-revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--token-src",
                    default=os.path.expanduser("~/.cache/huggingface/token"))
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--ckpt-store", default="/home/ecs-user/aad-artifacts/autoinit")
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=30)
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=8)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    #: The session is priced at 12-17 GPU-hours; the poll limit must outlast the
    #: hard threshold or the launcher would stop watching a pod that is still
    #: billing and still working.
    ap.add_argument("--poll-limit-min", type=float, default=1320.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    ap.add_argument("--out", default="logs/autoinit_phase_a_session.json")
    args = ap.parse_args()

    # Retarget the inherited module-level constants before anything runs.
    _preflight.AUTH_PATH = AUTH_PATH
    _preflight.STATUS = STATUS
    _preflight.RUN_LOG = RUN_LOG
    _preflight.LOCAL_ASSETS = LOCAL_ASSETS
    _preflight.PREFLIGHT_PLAN_V1 = PHASE_A_PLAN_V1
    # The base class loads a `SpendAuthorization`, which by design can never
    # grant Phase A. This session loads the one artifact type that can.
    _preflight.SpendAuthorization = PhaseAAuthorization

    session = PhaseA(args)
    ok = False
    try:
        ok = session.run()
    except Exception as exc:                                      # noqa: BLE001
        session.ev["launcher_error"] = f"{type(exc).__name__}: {exc}"
        session.say(f"LAUNCHER ERROR: {type(exc).__name__}: {exc}")
        if session.pod_id:
            session.teardown_now("launcher error")
    session.ev["passed"] = bool(ok)
    session.ev["phase_a_successful"] = bool(ok)
    session.ev["cleanup_is_not_success"] = (
        "artifacts are collected and the pod is torn down on every path; the "
        "session outcome is decided by the driver's terminal marker alone")
    session.ev["followon_started"] = False
    session.ev["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    session.save()
    print(f"\nPhase A {'COMPLETE' if ok else 'INCOMPLETE'} — "
          f"{REPO_ROOT / args.out}. STOP for review; no follow-on experiment "
          "was started.")
    return 0 if ok else 11


if __name__ == "__main__":
    raise SystemExit(main())
