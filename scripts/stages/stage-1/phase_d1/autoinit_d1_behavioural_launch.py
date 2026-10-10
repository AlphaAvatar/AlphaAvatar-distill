#!/usr/bin/env python3
"""The D1 BEHAVIOURAL launcher. One rung's session, declared completely.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d1/autoinit_d1_behavioural_launch.py \
        --scr <scr> --session-commit <sha> --bundle <canonical> \
        --run-id d1_behavioural_<utc> --rung screening

Everything is a FIELD of the `SessionSpec` the shared `SessionRunner` consumes:
the authorization type and path, the setup manifest, the driver command, the
budget, the teardown and artifact policies. The runner owns acquisition, the
watchdog, teardown on every exit path and collect-before-teardown; this file
owns what makes the session D1-behavioural's.

`SESSION_KIND = "d1_behavioural"` is DECLARED, and the matching branch in
`autoinit_preflight_setup.sh` loads a `D1BehaviouralAuthorization` and asserts
its typed permissions -- `allows_recovery_training` TRUE where the search's
branch asserts recovery False, and `allows_beam_search` False where the
search's asserts it True. A missing dispatch entry is not a type error: it
falls through to `spend`, and Phase B's attempt 2 proved what that costs.

**THE SESSION'S PRODUCTS ARE EVIDENCE, NOT CHECKPOINTS.** Once a probe is
scored its weights have no downstream consumer (P8.4): the ranking reads
scores, the verdict reads per-sample rows, and the confirmation rung retrains
from the ARM initializations at fresh seeds. What must come home -- and comes
home DURING the run, on every poll -- is each finished probe's result, its
per-sample rows and its raw generations, plus the session record carrying the
contract, the materialization gates, the ranking and the selection.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/pod", "scripts/autoinit"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, ExecutionCommands, LocalAsset, MarkerPolicy,
    SessionContext, SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    same_failure_gate, session_commit_gate,
)
from shared.deployment import deployment_commands  # noqa: E402
from shared.pod.autoinit_science_inputs import (  # noqa: E402
    CALIBRATION_V1, RECOVERY_LADDER,
)
from stages.phase_c1.autoinit_c1_launch import C1_EVAL_TOKENIZER  # noqa: E402
from shared.run_layout import rel_run_dir  # noqa: E402
from stages.phase_d1 import behavioural as D1B  # noqa: E402
from stages.phase_d1 import behavioural_authorization as BA  # noqa: E402
from stages.phase_d1 import behavioural_materialize as M  # noqa: E402
from stages.phase_d1 import d1_session as D1S  # noqa: E402
from stages.phase_d1.behavioural_governance import (  # noqa: E402
    D1BehaviouralAuthorization,
)

#: DECLARED, and the shell has a branch for it. See the module docstring.
SESSION_KIND = "d1_behavioural"

#: THE OPERATIONAL RESERVE in this session's provider-account requirement.
#: Small, and deliberately not a second budget: a pod stopped for an exhausted
#: ACCOUNT balance spends money and produces no endpoint -- the 2026-10-06
#: search lost $8.1716 to exactly that.
ACCOUNT_OPERATIONAL_RESERVE_USD = 5.0

#: THE POD'S BLOCKING TEST GATE runs D1's OWN suite, positively declared.
TEST_PATHS = ("scripts/stages/stage-1/phase_d1/tests",)

REPO = "/workspace/aad"

#: THE ONE AUTHORITATIVE EVIDENCE LAYOUT, repository-relative. The driver's
#: `--out`, the relay's evidence spec, the report fetch, the poll hook and both
#: artifact specs all name this directory.
AUDIT_DIRNAME = "autoinit_d1_behavioural"
EVIDENCE_DIR = f"artifacts/audit/{AUDIT_DIRNAME}"

#: WHERE THE TERMINAL MARKER GOES. Named ONCE: the driver command passes it and
#: `SessionSpec.status_path` carries the same value, so the file the driver
#: appends to is the file the runner polls. A driver writing markers to a path
#: its launcher does not poll made every marker invisible on A3's attempt75.
STATUS = "/workspace/autoinit_d1_behavioural.status"
RUN_LOG = "/workspace/autoinit_d1_behavioural_run.log"

#: EVERY NON-SOURCE INPUT THE DRIVER READS THAT GIT DOES NOT CARRY, and how
#: each travels. The pack, the evaluation tokenizer and the C1-era calibration
#: mixture are PINNED RELAY OBJECTS -- the pod pulls them at hub speed, the
#: same staging C2-behavioural and A3 ran under. The rest go by scp because
#: they are not on the relay and total ~11 MiB at the measured 0.72 MB/s
#: uplink: the second calibration mixture and BOTH battery roles -- both,
#: because `roles_are_disjoint` reads the realized item ids of each, on the
#: pod as on this host.
BATTERY_INSTALL = ("artifacts/stages/stage-1/families/d_series/batteries/"
                   "d_series_behavioural_v1")
SCIENCE_ASSETS: tuple[LocalAsset, ...] = (
    LocalAsset(repo_path="artifacts/stages/stage-1/reasoning_heavy_v2",
               dest_name="reasoning_heavy_v2",
               install_to="artifacts/stages/stage-1"),
    LocalAsset(repo_path=f"{BATTERY_INSTALL}/d1_screening",
               dest_name="d1_screening",
               install_to=BATTERY_INSTALL),
    LocalAsset(repo_path=f"{BATTERY_INSTALL}/d1_confirmation",
               dest_name="d1_confirmation",
               install_to=BATTERY_INSTALL),
)

RELAY_INPUTS = (*RECOVERY_LADDER, *CALIBRATION_V1, *C1_EVAL_TOKENIZER)


def run_dir_for(run_id: str | None) -> str:
    return rel_run_dir(D1S.EXPERIMENT_ID, run_id or "unrecorded", D1S.STAGE_ID)


def auth_path_for(run_id: str | None) -> str:
    """Where this run's one-use authorization lives. Per run, never shared."""
    return f"{run_dir_for(run_id)}/governance/authorization.json"


def readiness_path_for(run_id: str | None) -> str:
    """This run's launch-bound readiness record. ADMISSION, never authorization."""
    return f"{run_dir_for(run_id)}/governance/launch_readiness.json"


def bundle_record_for(run_id: str | None) -> str:
    return f"{run_dir_for(run_id)}/governance/bundle.json"


def driver_command(ctx: Any, plan: Any) -> str:
    """The exact command the pod runs. A test parses it with the driver's parser.

    The rung and -- on confirmation -- the advancing candidate come from the
    AUTHORIZATION, not from this launcher's arguments, so an artifact that does
    not name them cannot start a session that uses them.
    """
    run_id = getattr(ctx.args, "run_id", "unrecorded")
    rung = ctx.auth.rung
    out = f"{REPO}/{EVIDENCE_DIR}"
    cmd = (f"/opt/train/bin/python "
           f"{REPO}/scripts/stages/stage-1/phase_d1/autoinit_d1_behavioural_driver.py "
           f"--out '{out}' "
           f"--run-id '{run_id}' "
           f"--rung '{rung}' "
           f"--authorization '{REPO}/{auth_path_for(run_id)}' "
           f"--arm-root '{M.POD_ARM_ROOT}' "
           f"--replay-plan '{REPO}/{M.PLAN_REL}' "
           f"--status '{STATUS}' "
           f"--device cuda "
           f"--image-digest '{ctx.image_digest}'")
    if rung == "confirmation":
        cmd += f" --advancing-candidate '{ctx.auth.advancing_candidate}'"
    return cmd


# ---------------------------------------------------------------------------
# per-probe durability, during the run
# ---------------------------------------------------------------------------

STORE_SUBDIR = "store"


def evidence_locations(ctx: SessionContext) -> tuple[Path, ...]:
    """Where the driver's live evidence can be read off-pod, newest first.

    The runner mirrors the driver's evidence to `<scr>/relay/` during the run
    and fetches reports to `<scr>/store/` at closeout. A sibling launcher
    GUESSED this path, found no evidence, and reported no finished work on a
    pod holding two completed leaves.
    """
    scr = Path(ctx.args.scr)
    return (scr / "relay" / "d1_behavioural.json",
            scr / STORE_SUBDIR / "d1_behavioural.json",
            scr / STORE_SUBDIR / "extracted" / EVIDENCE_DIR
                / "d1_behavioural.json")


def finished_probes(ctx: SessionContext) -> list[dict] | None:
    """What the driver says it SCORED, or None when that is UNKNOWN.

    `[]` means the driver ran and finished nothing; `None` means its evidence
    could not be read -- different claims, never reported as one.
    """
    for path in evidence_locations(ctx):
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        ctx.evidence["probe_evidence_read_from"] = str(path)
        return [p for p in record.get("probes", []) if p.get("scored")]
    ctx.evidence["probe_evidence_unreadable"] = [
        str(p) for p in evidence_locations(ctx)]
    return None


def _scp_from_pod(ctx: SessionContext, remote: str, dest: Path, *,
                  recursive: bool = False, limit_min: int = 5) -> int:
    """One object off the pod. Returns the return code; never raises."""
    cmd = ["timeout", f"{limit_min}m", "scp"]
    if recursive:
        cmd.append("-r")
    cmd += ["-P", str(ctx.target.port), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"root@{ctx.host}:{remote}", str(dest)]
    return subprocess.run(cmd, capture_output=True, timeout=None).returncode


def probe_store(ctx: SessionContext) -> Path:
    return Path(ctx.args.scr) / STORE_SUBDIR / "probes"


def secure_probe_evidence(ctx: SessionContext) -> None:
    """Pull each SCORED probe's evidence off-pod, while the next trains.

    Idempotent: a probe whose result, per-sample rows and generations all
    arrived is skipped on later polls. The files are small (a result JSON and
    a <1 MiB per-sample JSONL) except the raw generations, which P18 requires
    saved and which travel once, recursively, with their own timeout.

    MUST NOT raise: a durability helper that throws into a paid session's poll
    loop is a defect regardless of who catches it.

    THE SCORED COUNT GATES ONLY THE SCORED-EVIDENCE LOOP. The first version
    returned on `not units`, so during the FIRST probe's trained-but-unscored
    interval -- zero probes scored, one completed checkpoint on the pod, the
    exact window the C1 failure lived in -- the trained-unscored preservation
    below was never reached, and the provider-disappearance repair protected
    every window except the first one it was written for.
    """
    try:
        units = finished_probes(ctx) or []
        store = probe_store(ctx)
        secured = ctx.evidence.setdefault("probe_evidence_secured", {})
        for unit in units:
            probe_id = str(unit.get("probe_id") or "")
            if not probe_id:
                continue
            state = secured.setdefault(probe_id, {})
            dest = store / probe_id
            dest.mkdir(parents=True, exist_ok=True)
            want = {
                "result.json": (f"{REPO}/{EVIDENCE_DIR}/results/"
                                f"{probe_id}.json", False),
                "per_sample.jsonl": (f"{REPO}/{EVIDENCE_DIR}/per_sample/"
                                     f"{probe_id}.jsonl", False),
                #: The probe's generation-admission record: the raw evidence
                #: that THIS probe's protocol was observed comparable, not
                #: just the hash the result carries.
                "admission.json": (f"{REPO}/{EVIDENCE_DIR}/"
                                   f"{probe_id}_generation_admission.json",
                                   False),
                "generations": (f"{REPO}/{EVIDENCE_DIR}/generations/"
                                f"{probe_id}", True),
            }
            for name, (remote, recursive) in want.items():
                if state.get(name):
                    continue
                target = dest / name
                rc = _scp_from_pod(ctx, remote, target, recursive=recursive,
                                   limit_min=12 if recursive else 5)
                ok = rc == 0 and (target.is_dir() if recursive
                                  else (target.is_file()
                                        and target.stat().st_size > 0))
                state[name] = bool(ok)
                if ok:
                    ctx.say(f"  probe {probe_id}: secured {name}")
            #: EVIDENCE DURABLE -> THE POD MAY RELEASE. The ack is written
            #: only when every strict piece arrived, which is what makes the
            #: driver's workdir release never race durability.
            if (state.get("result.json") and state.get("per_sample.jsonl")
                    and state.get("admission.json")
                    and not state.get("release_acked")):
                state["release_acked"] = write_release_ack(ctx, probe_id)
        #: The session-level protocol evidence, once: the attestation and the
        #: engine probe are what an auditor reconstructs the protocol identity
        #: from, and the recorded hash alone is not that.
        store = Path(ctx.args.scr) / STORE_SUBDIR
        store.mkdir(parents=True, exist_ok=True)
        for name in ("d1_behavioural_attested_protocol.json",
                     "engine_probe.json"):
            target = store / name
            if not target.is_file():
                rc = _scp_from_pod(ctx, f"{REPO}/{EVIDENCE_DIR}/{name}",
                                   target)
                if rc == 0 and target.is_file():
                    ctx.say(f"  secured {name}")
        #: Trained-but-unscored durability DURING the run (P8.2.1: persist at
        #: the moment of completion, not at a closeout that may never come) --
        #: and prune preserved copies whose probe has since been validly
        #: scored, because those bytes then have no remaining consumer.
        preserve_trained_unscored(ctx)
        prune_preserved_scored(ctx)
    except Exception as exc:                                    # noqa: BLE001
        ctx.evidence.setdefault("on_poll_errors", []).append(
            f"secure_probe_evidence: {type(exc).__name__}: {exc}")


def write_release_ack(ctx: SessionContext, probe_id: str) -> bool:
    """Tell the pod this probe's evidence is durable, so it may release.

    C2's mechanism, under D1's ack path. MUST NOT raise (poll loop); a failure
    costs pod disk, which the driver's fail-closed release then reports.
    """
    import tempfile

    ack_dir = f"{REPO}/{M.RELEASE_ACK_REL}"
    payload = json.dumps({
        "schema": "aadistill.phase_d1.behavioural_release_ack/v1",
        "probe_id": probe_id,
        "evidence": ["result.json", "per_sample.jsonl", "admission.json"],
        "_what_this_permits": (
            "releasing this probe's LOCAL training workdir and evaluation "
            "package on the pod. The probe is validly scored and its "
            "scientific evidence is off-pod; its weights have no remaining "
            "consumer (P8.4 state 1)."),
    })
    try:
        ctx.target.run(f"mkdir -p {ack_dir}", timeout=60)
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / f"{probe_id}.json"
            local.write_text(payload + "\n")
            rc = subprocess.run(
                ["scp", "-P", str(ctx.target.port),
                 "-o", "StrictHostKeyChecking=no",
                 "-o", "UserKnownHostsFile=/dev/null",
                 str(local), f"root@{ctx.host}:{ack_dir}/{probe_id}.json"],
                capture_output=True, timeout=120).returncode
        ok = rc == 0
    except Exception as exc:                                    # noqa: BLE001
        ctx.evidence.setdefault("release_ack_errors", []).append(
            f"{probe_id}: {type(exc).__name__}: {exc}")
        return False
    ctx.evidence.setdefault("release_acks", []).append(
        {"probe_id": probe_id, "delivered": ok})
    if ok:
        ctx.say(f"  probe {probe_id}: release ack delivered")
    return ok


def prune_preserved_scored(ctx: SessionContext) -> None:
    """Delete preserved trained-unscored copies whose probe is now scored.

    P8.4: preserve only bytes with an actual remaining scoring consumer. A
    probe preserved during its scoring window loses that consumer the moment
    its score and evidence are durable, so the local copy is removed --
    containment-checked under the preservation root -- and the row says so.
    MUST NOT raise.
    """
    import shutil

    try:
        preserved = ctx.evidence.get("trained_unscored_preserved") or {}
        if not preserved:
            return
        secured = ctx.evidence.get("probe_evidence_secured") or {}
        store = (Path(getattr(ctx.args, "ckpt_store", None)
                      or Path(ctx.args.scr) / "products")
                 / "trained_unscored").resolve()
        for probe_id, row in preserved.items():
            if row.get("pruned") or not row.get("verified"):
                continue
            state = secured.get(probe_id) or {}
            if not (state.get("result.json") and state.get("per_sample.jsonl")
                    and state.get("admission.json")):
                continue
            dest = Path(row.get("dest") or "")
            if not dest.is_dir() or not dest.resolve().is_relative_to(store):
                continue
            shutil.rmtree(dest)
            row["pruned"] = True
            row["_why"] = ("validly scored and its evidence durable; the "
                           "weights have no remaining consumer (P8.4)")
            ctx.say(f"  probe {probe_id}: pruned preserved trained copy "
                    "(now scored)")
    except Exception as exc:                                    # noqa: BLE001
        ctx.evidence.setdefault("on_poll_errors", []).append(
            f"prune_preserved_scored: {type(exc).__name__}: {exc}")


#: Where a probe's training writes on the pod, and where a restore puts a
#: preserved probe back. Named once; the driver's `model_root` is the same
#: expression over its own REPO_ROOT.
POD_PROBE_ROOT = f"{REPO}/artifacts/stages/stage-3/d1_behavioural"


def trained_unscored_probes(ctx: SessionContext) -> list[dict] | None:
    """Probes that FINISHED TRAINING and were not scored, or None if unknown.

    P8.4 state 2: trained + not validly scored means the WEIGHTS are owed --
    scoring genuinely consumes them, and the protocol forbids retraining a
    frozen unit for a different outcome. A probe that scored is state 1 and
    its weights are deliberately NOT fetched.
    """
    for path in evidence_locations(ctx):
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        return [p for p in record.get("probes", [])
                if p.get("trained") and not p.get("scored")
                and p.get("model_dir")]
    return None


def preserve_trained_unscored(ctx: SessionContext) -> None:
    """Fetch every trained-but-unscored probe's checkpoint off-pod. Verified.

    DOWNLOAD, which is the direction that works: the search secured four
    1.11 GiB products this way, and it is the dev-box UPLINK that cannot carry
    a checkpoint. Each arrival is re-hashed against the `trained_sha256` the
    driver recorded the moment training completed, and the identity travels
    beside the bytes as `preserved_identity.json` -- the sidecar a replacement
    pod's resume re-checks before scoring them.

    Idempotent and MUST NOT raise into the poll/teardown path.
    """
    import hashlib
    import shutil

    try:
        units = trained_unscored_probes(ctx)
        if not units:
            return
        store = Path(getattr(ctx.args, "ckpt_store", None)
                     or Path(ctx.args.scr) / "products") / "trained_unscored"
        preserved = ctx.evidence.setdefault("trained_unscored_preserved", {})
        for unit in units:
            probe_id = str(unit.get("probe_id") or "")
            if not probe_id or preserved.get(probe_id, {}).get("verified"):
                continue
            remote_model = str(unit["model_dir"])
            tag = Path(remote_model).parent.name
            remote_root = f"{POD_PROBE_ROOT}/{probe_id}"
            dest = store / probe_id
            (dest / "checkpoints" / tag).mkdir(parents=True, exist_ok=True)
            rcs = {
                "run_completion.json": _scp_from_pod(
                    ctx, f"{remote_root}/run_completion.json",
                    dest / "run_completion.json"),
                "latest.txt": _scp_from_pod(
                    ctx, f"{remote_root}/checkpoints/latest.txt",
                    dest / "checkpoints" / "latest.txt"),
                "model": _scp_from_pod(
                    ctx, remote_model, dest / "checkpoints" / tag,
                    recursive=True,
                    limit_min=int(getattr(ctx.args, "ckpt_fetch_limit_min",
                                          45))),
            }
            shard = dest / "checkpoints" / tag / "model" / "model.safetensors"
            #: scp -r of `.../model` INTO `checkpoints/<tag>/` lands at
            #: `checkpoints/<tag>/model/` -- create the tag dir first so the
            #: copy nests rather than renames.
            if not shard.is_file():
                flat = dest / "checkpoints" / tag / "model.safetensors"
                if flat.is_file():
                    (dest / "checkpoints" / tag / "model").mkdir(
                        parents=True, exist_ok=True)
                    for item in list((dest / "checkpoints" / tag).iterdir()):
                        if item.name != "model":
                            shutil.move(str(item),
                                        str(dest / "checkpoints" / tag
                                            / "model" / item.name))
            want = str(unit.get("trained_sha256") or "")
            got = (hashlib.sha256(shard.read_bytes()).hexdigest()
                   if shard.is_file() else None)
            #: VERIFIED means THE WHOLE RESUME SET arrived, not that one shard
            #: hashes right. The driver's evaluation-only resume consumes four
            #: things -- the training-completion record, the checkpoint tag,
            #: the model directory with its config, and digest-matching
            #: weights -- and a preservation marked durable on the shard alone
            #: would restore an incomplete set that `trained_checkpoint_state`
            #: then reads as "never finished training" and RETRAINS, which is
            #: exactly what P8.4 state 2 forbids.
            latest = dest / "checkpoints" / "latest.txt"
            completeness = {
                "transfers_ok": all(rc == 0 for rc in rcs.values()),
                "run_completion": (dest / "run_completion.json").is_file(),
                "latest_tag": (latest.is_file()
                               and latest.read_text().strip() == tag),
                "model_config": (dest / "checkpoints" / tag / "model"
                                 / "config.json").is_file(),
                "weights_digest": bool(want) and got == want,
            }
            verified = all(completeness.values())
            row = {"rcs": rcs, "trained_sha256": want,
                   "arrived_sha256": got, "completeness": completeness,
                   "verified": verified,
                   "dest": str(dest)}
            preserved[probe_id] = row
            if verified:
                (dest / "preserved_identity.json").write_text(json.dumps({
                    "schema": "aadistill.phase_d1.preserved_trained_probe/v1",
                    "probe_id": probe_id,
                    "trained_sha256": want,
                    "tag": tag,
                    "_reuse_is_not_authorized_by_preservation": (
                        "P8.2.1: saving this checkpoint authorizes nothing "
                        "about reusing it. Whether a replacement session may "
                        "resume it at evaluation is the frozen protocol's "
                        "rule (P8.4 state 2), enforced by the driver's "
                        "identity-checked resume."),
                }, indent=1) + "\n")
                ctx.say(f"  trained-unscored {probe_id}: preserved and "
                        f"verified ({row['arrived_sha256'][:12]})")
            else:
                ctx.say(f"  trained-unscored {probe_id}: NOT verified "
                        f"(rcs={rcs}, want={want[:12] if want else None}, "
                        f"got={str(got)[:12]})")
    except Exception as exc:                                    # noqa: BLE001
        ctx.evidence.setdefault("on_poll_errors", []).append(
            f"preserve_trained_unscored: {type(exc).__name__}: {exc}")


def restore_trained_probes(ctx: SessionContext) -> bool:
    """Put preserved trained-unscored probes back on a REPLACEMENT pod.

    The other half of P8.4 state 2. `--restore-trained` names a local
    directory of preserved probes (the layout `preserve_trained_unscored`
    writes); each is pushed to the pod's probe root before the driver starts,
    sidecar included, and the driver's `trained_checkpoint_state` then
    re-hashes the shard against the sidecar before resuming at evaluation.
    `False` aborts the session -- a restore that half-arrived must not
    silently retrain the probe it was supposed to resume.
    """
    root = str(getattr(ctx.args, "restore_trained", "") or "").strip()
    if not root:
        return True
    source = Path(root)
    probes = sorted(p for p in source.iterdir() if p.is_dir()) \
        if source.is_dir() else []
    if not probes:
        ctx.say(f"  --restore-trained {root}: nothing to restore")
        return True
    import subprocess as sp

    for probe_dir in probes:
        sidecar = probe_dir / "preserved_identity.json"
        if not sidecar.is_file():
            ctx.say(f"  restore {probe_dir.name}: no preserved_identity.json; "
                    "refusing to push unverifiable bytes")
            return False
        remote_root = f"{POD_PROBE_ROOT}/{probe_dir.name}"
        sp.run(["ssh", "-p", str(ctx.target.port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                f"root@{ctx.host}", f"mkdir -p {remote_root}"],
               capture_output=True, timeout=120)
        #: PUSH, deliberately without the LocalAsset machinery: its 600 s
        #: per-asset cap exists for small science inputs, and a 1.2 GiB
        #: checkpoint at the measured ~0.72 MB/s uplink is ~28 minutes --
        #: a priced contingency, not a routine transfer.
        rc = sp.run(["scp", "-r", "-P", str(ctx.target.port),
                     "-o", "StrictHostKeyChecking=no",
                     "-o", "UserKnownHostsFile=/dev/null",
                     *[str(p) for p in sorted(probe_dir.iterdir())],
                     f"root@{ctx.host}:{remote_root}/"],
                    capture_output=True, timeout=None).returncode
        ctx.evidence.setdefault("trained_unscored_restored", {})[
            probe_dir.name] = {"rc": rc, "from": str(probe_dir)}
        if rc != 0:
            ctx.say(f"  restore {probe_dir.name}: scp rc={rc}; aborting "
                    "before the driver can retrain a preserved probe")
            return False
        ctx.say(f"  restored trained-unscored probe {probe_dir.name}")
    return True


def fetch_probe_evidence(ctx: SessionContext) -> list:
    """Closeout sweep: anything the poll loop did not already secure."""
    secure_probe_evidence(ctx)
    preserve_trained_unscored(ctx)
    return [
        {"probe_id": probe_id, **state}
        for probe_id, state in sorted(
            (ctx.evidence.get("probe_evidence_secured") or {}).items())]


def probes_evidence_secured(ctx: SessionContext,
                            fetched: list) -> tuple[bool, str]:
    """Teardown may not proceed while a scored probe's evidence is only on
    the pod.

    The RESULT and the PER-SAMPLE rows are strict: the ranking reads the
    first and the verdict reads the second, and a probe is thirty-plus paid
    minutes the protocol forbids repeating for a different outcome. The raw
    generations are also owed (P18), but they additionally travel in the
    required final archive, so a generations directory the poll hook missed is
    reported rather than blocking alone.
    """
    units = finished_probes(ctx)
    if units is None:
        return False, (
            "the driver evidence could not be read, so what it finished is "
            "UNKNOWN. Refusing teardown: 'I found no probes' and 'no probes "
            "were scored' are different findings, and treating the first as "
            "the second is how a pod holding finished work gets deleted with "
            f"every check green. Looked in: "
            f"{[str(p) for p in evidence_locations(ctx)]}")
    #: THE TRAINED-BUT-UNSCORED WEIGHTS FIRST (P8.4 state 2), and BEFORE the
    #: "nothing scored" early return: the session that dies during its FIRST
    #: probe's generation has zero scored probes and one trained checkpoint,
    #: and that is precisely the pod that must not be deleted early. Scoring
    #: genuinely consumes these weights and the protocol forbids retraining a
    #: frozen unit for a different outcome.
    unscored = trained_unscored_probes(ctx)
    if unscored is None:
        return False, (
            "the driver evidence could not be read while deciding whether any "
            "trained-but-unscored checkpoint is owed; refusing teardown on an "
            "unknown")
    preserved = ctx.evidence.get("trained_unscored_preserved") or {}
    unpreserved = sorted(
        str(u.get("probe_id")) for u in unscored
        if not preserved.get(str(u.get("probe_id")), {}).get("verified"))
    if unpreserved:
        return False, (
            f"{len(unscored)} probe(s) finished TRAINING without a valid "
            f"score and {unpreserved} are not preserved off-pod. P8.4 state "
            "2: their weights are what a resumed scoring consumes, and "
            "retraining them for a different outcome is forbidden -- the pod "
            "holds the only copy.")

    want = {str(u.get("probe_id")) for u in units if u.get("probe_id")}
    if not want:
        note = "the driver scored no probe, so no scored evidence is owed"
        if unscored:
            note += (f"; {len(unscored)} trained-unscored checkpoint(s) "
                     "preserved and identity-verified")
        return True, note
    secured = ctx.evidence.get("probe_evidence_secured") or {}
    #: The ADMISSION record is strict beside the result and the rows: it is
    #: the raw evidence that this probe's generations were produced under the
    #: attested protocol, and a result whose admission did not come home
    #: carries a hash nobody can audit.
    missing = sorted(
        probe_id for probe_id in want
        if not (secured.get(probe_id, {}).get("result.json")
                and secured.get(probe_id, {}).get("per_sample.jsonl")
                and secured.get(probe_id, {}).get("admission.json")))
    if missing:
        return False, (
            f"{len(want)} probes scored and {len(want) - len(missing)} have "
            f"their result, per-sample rows and generation admission "
            f"off-pod; missing {missing}. Deleting the pod now would destroy "
            "measurements this session already paid for.")
    #: And the session-level protocol evidence itself: the attestation every
    #: admission was compared against.
    attested = Path(ctx.args.scr) / STORE_SUBDIR / \
        "d1_behavioural_attested_protocol.json"
    if not attested.is_file():
        return False, (
            f"{len(want)} probes scored and the attested evaluation protocol "
            "record is not off-pod; the per-probe admissions compare against "
            "it and without it the protocol identity cannot be audited.")

    no_generations = sorted(
        probe_id for probe_id in want
        if not secured.get(probe_id, {}).get("generations"))
    note = (f"all {len(want)} scored probes' results and per-sample rows "
            "off-pod")
    if unscored:
        note += (f"; {len(unscored)} trained-unscored checkpoint(s) preserved "
                 "and identity-verified")
    if no_generations:
        note += (f"; generations for {no_generations} rely on the required "
                 "final archive")
    return True, note


# ---------------------------------------------------------------------------
# the gates that run at $0, before a provider is contacted
# ---------------------------------------------------------------------------


def run_identity_gate(ctx) -> tuple[bool, str]:
    """The rung, the run id, the design revision and -- on confirmation --
    the advancing candidate this grant was issued for."""
    rung = getattr(ctx.args, "rung", "screening")
    try:
        ctx.auth.require_run_id(getattr(ctx.args, "run_id", ""))
        ctx.auth.require_rung(rung)
    except Exception as exc:                                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    design_hash = D1B.design(REPO_ROOT)["design_hash"]
    if ctx.auth.design_hash != design_hash:
        return False, (f"the authorization binds design "
                       f"{ctx.auth.design_hash[:12]} and the tree carries "
                       f"{design_hash[:12]}")
    blockers = list(D1B.design(REPO_ROOT)["open_blockers"])
    if blockers:
        return False, (f"the design reports open blockers {blockers}; a "
                       "formal session may not run over one")
    advancing = None
    if rung == "confirmation":
        try:
            advancing = ctx.auth.require_advancing_candidate()
        except Exception as exc:                                  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
    ctx.evidence["d1_behavioural_run_identity"] = {
        "run_id": ctx.auth.run_id, "rung": ctx.auth.rung,
        "advancing_candidate": advancing,
        "design_hash": design_hash,
        "contract_hash": ctx.auth.contract_hash,
        "open_blockers": blockers,
    }
    return True, (f"run {ctx.auth.run_id}, {ctx.auth.rung} rung"
                  + (f" confirming {advancing}" if advancing else "")
                  + f", design {design_hash[:12]}, no open blockers")


def behavioural_contract_gate(ctx) -> tuple[bool, str]:
    """Derive the REAL contract here and require the bound hash. `$0`.

    The driver asks the same equality on the pod, which is correct and is also
    the most expensive place to discover a mismatch: building the contract
    verifies the candidates' secured bytes, the battery bytes and the
    schedule, all of which exist on this host.
    """
    from aadistill.infrastructure.manifest import sha256_json

    rung = ctx.auth.rung
    advancing = (ctx.auth.advancing_candidate
                 if rung == "confirmation" else None)
    try:
        contract = D1B.session_contract(rung, REPO_ROOT,
                                        advancing_candidate=advancing)
    except Exception as exc:                                      # noqa: BLE001
        return False, (f"the behavioural contract does not derive on this "
                       f"tree: {type(exc).__name__}: {exc}")
    got = sha256_json(contract)
    ctx.evidence["d1_behavioural_contract_check"] = {
        "contract_hash": got,
        "authorized": ctx.auth.contract_hash,
        "n_probes": contract["n_probes"],
        "battery_role": contract["battery"]["role"],
        "battery_content_id": contract["battery"]["family_content_id"],
        "seeds": contract["seeds"],
    }
    if got != ctx.auth.contract_hash:
        return False, (
            f"the contract this tree derives hashes to {got[:16]} and the "
            f"authorization binds {ctx.auth.contract_hash[:16]}. Something in "
            "the arms, the battery, the recipe, the seeds or the schedule is "
            "not what was authorized.")
    if contract["battery"]["family_content_id"] != ctx.auth.battery_content_id:
        return False, "the battery content id is not the authorized one"
    if tuple(contract["seeds"]) != tuple(ctx.auth.seeds):
        return False, (f"the derived seeds {contract['seeds']} are not the "
                       f"authorized {list(ctx.auth.seeds)}")
    return True, (f"contract verified at $0: {got[:16]}, "
                  f"{contract['n_probes']} probes on "
                  f"{contract['battery']['role']}")


def staged_science_inputs_gate(ctx) -> tuple[bool, str]:
    """Every input the pod consumes exists here and matches the pin it ships.

    The batteries re-hash against the family manifest (`battery_role`), both
    calibration mixtures resolve through their own `items_file_sha256`, the
    recovery pack's digest is checked against the frozen recipe inside the
    contract derivation, the committed replay plan matches the hash the
    authorization binds, and the preregistration matches both its own hash and
    this tree's derivation.
    """
    checked: dict[str, Any] = {}
    try:
        for role in ("d1_screening", "d1_confirmation"):
            identity = D1B.battery_role(role, REPO_ROOT)
            checked[role] = {"n_prompts": identity["n_prompts"],
                             "item_ids_sha256": identity["item_ids_sha256"]}
    except Exception as exc:                                      # noqa: BLE001
        return False, f"a battery role is not usable: {exc}"

    try:
        from aadistill.initialization.calibration.profiles import get_profile
        from shared.calibration import register_builtin_profiles

        register_builtin_profiles()
        for pid in ("calib.domain_balanced@v1", "calib.reasoning_heavy@v2"):
            profile = get_profile(pid)
            items = profile.resolve(REPO_ROOT)
            checked[pid] = {"n_items": len(items),
                            "items_file_sha256": profile.items_file_sha256}
    except Exception as exc:                                      # noqa: BLE001
        return False, (f"a calibration profile does not resolve on this tree: "
                       f"{type(exc).__name__}: {exc}")

    try:
        got = M.plan_sha256(REPO_ROOT)
    except Exception as exc:                                      # noqa: BLE001
        return False, f"the replay plan is not usable: {exc}"
    #: The bound values live beside the typed fields in the raw artifact; the
    #: type does not carry them, so they are read from the document the
    #: self-hash covers.
    auth_doc = json.loads(
        (REPO_ROOT / auth_path_for(getattr(ctx.args, "run_id", None)))
        .read_text())
    if got != auth_doc.get("replay_plan_sha256"):
        return False, (
            f"the committed replay plan hashes to {got[:12]} and the "
            f"authorization binds {str(auth_doc.get('replay_plan_sha256'))[:12]}; "
            "the pinned paths the pod would materialize along are not the "
            "ones that were authorized")
    checked["replay_plan_sha256"] = got

    try:
        prereg = BA.preregistration(REPO_ROOT)
    except Exception as exc:                                      # noqa: BLE001
        return False, str(exc)
    if prereg["preregistration_sha256"] != auth_doc.get(
            "preregistration_sha256"):
        return False, (
            "the committed preregistration is not the one the authorization "
            "binds; regenerate, re-review and re-issue rather than launching "
            "over the disagreement")
    checked["preregistration_sha256"] = prereg["preregistration_sha256"]

    missing = [a.repo_path for a in SCIENCE_ASSETS
               if not (REPO_ROOT / a.repo_path).exists()]
    if missing:
        return False, f"declared local assets absent from this tree: {missing}"
    ctx.evidence["d1_behavioural_staged_inputs"] = {
        "local_assets": [a.as_env_entry() for a in SCIENCE_ASSETS],
        "relay_inputs": [r.as_record() for r in RELAY_INPUTS],
        "verified": checked,
        "teacher_revision": D1S.root_teacher_identity(REPO_ROOT)["revision"],
    }
    return True, ("science inputs verified against their own pins: "
                  + ", ".join(sorted(checked)))


def launch_readiness_gate(ctx) -> tuple[bool, str]:
    """The launch-bound readiness record must describe THIS invocation.

    AUTHORIZES NOTHING; see the search launcher's gate, whose shape this is.
    """
    rel = readiness_path_for(getattr(ctx.args, "run_id", None))
    path = REPO_ROOT / rel
    if not path.is_file():
        return False, (f"no launch-readiness record at {rel}; run "
                       "scripts/stages/stage-1/phase_d1/"
                       "write_d1_behavioural_launch_readiness.py immediately "
                       "before launching")
    try:
        doc = json.loads(path.read_text())
    except ValueError as exc:
        return False, f"{rel} is not parseable JSON: {exc}"

    auth_rel = auth_path_for(getattr(ctx.args, "run_id", None))
    closure = BA.behavioural_current_executable(REPO_ROOT)
    want = {
        "run_id": getattr(ctx.args, "run_id", None),
        "session_commit": getattr(ctx.args, "session_commit", None),
        "rung": ctx.auth.rung,
        "design_hash": D1B.design(REPO_ROOT)["design_hash"],
        "harness_digest": closure["digest"],
        "authorization_sha256": _authorization_self_hash(auth_rel),
        "bundle_name": D1S.canonical_bundle_name(
            getattr(ctx.args, "session_commit", "") or ""),
    }
    got = {
        "run_id": doc.get("run_id"),
        "session_commit": doc.get("session_commit"),
        "rung": doc.get("rung"),
        "design_hash": doc.get("design_hash"),
        "harness_digest": (doc.get("harness") or {}).get("digest"),
        "authorization_sha256": (doc.get("authorization") or {}).get(
            "authorization_sha256"),
        "bundle_name": (doc.get("bundle") or {}).get("name"),
    }
    disagree = sorted(k for k in want if want[k] != got[k])
    ceiling_in_record = (doc.get("derived_session") or {}).get(
        "hard_ceiling_usd")
    ceiling_disagrees = (
        ceiling_in_record is None
        or abs(float(ceiling_in_record) - float(ctx.auth.hard_cap_usd)) > 5e-4)
    blockers = list(doc.get("open_blockers") or [])
    ctx.evidence["d1_behavioural_launch_readiness_check"] = {
        "path": rel, "expected": want, "recorded": got, "disagree": disagree,
        "recorded_ceiling_usd": ceiling_in_record,
        "authorized_ceiling_usd": float(ctx.auth.hard_cap_usd),
        "open_blockers": blockers,
        "authorizes": "nothing; this is an admission/consistency gate",
    }
    if disagree:
        return False, (
            f"{rel} does not describe this launch: {disagree} disagree. A "
            "stale readiness record is refused at $0 rather than read as "
            "reassurance.")
    if ceiling_disagrees:
        return False, (f"{rel} records a derived ceiling of "
                       f"{ceiling_in_record} and the authorization carries "
                       f"${float(ctx.auth.hard_cap_usd):.4f}")
    if blockers:
        return False, f"{rel} reports open blockers {blockers}"
    return True, (f"launch-readiness {rel} matches this invocation on "
                  f"{len(want)} bound facts and reports no open blockers")


def _authorization_self_hash(auth_rel: str) -> str | None:
    path = REPO_ROOT / auth_rel
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text()).get("authorization_sha256")
    except ValueError:
        return None


def bundle_staged_gate(ctx) -> tuple[bool, str]:
    """Can a pod, RIGHT NOW, obtain the exact authorized code? Read-only."""
    import tempfile

    from aadistill.infrastructure.bundle_transport import (
        BundleTransportError, hf_download, roundtrip,
    )

    commit = getattr(ctx.args, "session_commit", "")
    try:
        D1S.require_canonical_bundle(getattr(ctx.args, "bundle", ""), commit)
    except BundleTransportError as exc:
        return False, str(exc)

    rel = bundle_record_for(getattr(ctx.args, "run_id", None))
    staged = REPO_ROOT / rel
    if not staged.is_file():
        return False, (
            f"{rel} is missing; run scripts/stages/stage-1/phase_d1/"
            f"stage_d1_bundle.py --session-commit {commit} --run-id "
            f"{getattr(ctx.args, 'run_id', '')} first.")
    record = json.loads(staged.read_text())
    if record.get("session_commit") != commit:
        return False, (f"{rel} describes a bundle for "
                       f"{str(record.get('session_commit'))[:12]}, not the "
                       f"session commit {commit[:12]}")

    auth_rel = auth_path_for(getattr(ctx.args, "run_id", None))
    auth_bytes = (REPO_ROOT / auth_rel).read_bytes()
    try:
        with tempfile.TemporaryDirectory(prefix="d1b-bundle-") as tmp:
            evidence = roundtrip(
                D1S.transport(),
                session_commit=commit,
                local_bundle_sha256=record["sha256"],
                authorization_bytes=auth_bytes,
                authorization_path=auth_rel,
                expected_harness_digest=ctx.auth.harness_source_digest,
                harness_files=tuple(ctx.auth.harness_source_files),
                download=hf_download, workdir=Path(tmp))
    except Exception as exc:                                      # noqa: BLE001
        return False, f"the pod could not obtain the authorized commit: {exc}"

    ctx.evidence["bundle_staged_check"] = evidence
    return True, (f"{evidence['canonical_bundle_name']} ({evidence['bytes']} "
                  f"bytes) round-trips to {evidence['roundtrip_head'][:12]} "
                  "carrying this authorization and harness "
                  f"{evidence['roundtrip_harness_digest'][:12]}")


#: Where this experiment's prior behavioural attempts leave their launcher
#: logs -- an INSTANCE fact the reusable same-failure gate takes as a callable.
RUNS_ROOT = REPO_ROOT / "logs/stages/stage-1/phase_d1/runs"


def _prior_attempt_logs() -> list[tuple[str, str]]:
    """(attempt, launcher log) for every prior behavioural attempt, oldest
    first. Behavioural run ids carry a UTC stamp, so lexical order is time
    order within the prefix."""
    if not RUNS_ROOT.is_dir():
        return []
    out = []
    for d in sorted(p for p in RUNS_ROOT.iterdir()
                    if p.is_dir() and p.name.startswith("d1_behavioural")):
        log = d / "runtime" / "launcher.log"
        if log.is_file():
            out.append((d.name, log.read_text(errors="replace")))
    return out


def _corrective_change_since() -> tuple[bool, str]:
    """Has anything that could plausibly address the last failure changed?

    The commit the previous attempt ran, against this launch's HEAD, with the
    run directories themselves excluded: a tracked change outside them is a
    code, config or environment change; a change inside one is the chain
    rebuilding itself, which is not a repair.
    """
    logs = _prior_attempt_logs()
    if not logs:
        return True, "no prior behavioural attempt exists"
    prior_commit, prior_run = None, None
    for name, _ in reversed(logs):
        record = (REPO_ROOT / run_dir_for(name) / "runtime" / "session.json")
        if not record.is_file():
            continue
        try:
            doc = json.loads(record.read_text())
        except json.JSONDecodeError:
            continue
        commit = (doc.get("session_commit_check") or {}).get("session_commit")
        if commit:
            prior_commit, prior_run = commit, name
            break
    if not prior_commit:
        return False, (
            "a prior behavioural attempt exists and its session commit cannot "
            "be read, so whether anything changed since it cannot be "
            "established -- and this gate's whole purpose is to stop a "
            "repetition it cannot rule out")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=REPO_ROOT).stdout.strip()
    if head == prior_commit:
        return False, (f"HEAD is still {head[:12]}, the commit "
                       f"{prior_run} ran; nothing tracked has changed")
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{prior_commit}..{head}"],
        capture_output=True, text=True, cwd=REPO_ROOT).stdout.splitlines()
    runs_prefix = "logs/stages/stage-1/phase_d1/runs/"
    outside = [p for p in diff if not p.startswith(runs_prefix)]
    if outside:
        return True, (f"{len(outside)} tracked path(s) changed since "
                      f"{prior_run} ({prior_commit[:12]}..{head[:12]}), e.g. "
                      f"{outside[:3]}")
    return False, (f"only run-directory records changed since {prior_run}; "
                   "the chain rebuilding itself is not a repair")


def prechecks(args) -> tuple:
    """Every `$0` gate, in the order a refusal is cheapest."""
    return (
        run_identity_gate,
        behavioural_contract_gate,
        staged_science_inputs_gate,
        launch_readiness_gate,
        #: THE HARNESS AT `--session-commit` must be the authorized one, AND
        #: nothing other than the authorization artifact may differ from the
        #: authorized base. The readiness and bundle records stay UNCOMMITTED
        #: inside a launch window for exactly this reason.
        session_commit_gate(REPO_ROOT,
                            auth_path_for(getattr(args, "run_id", None)),
                            check_lineage=True),
        #: A deterministic paid failure may not be retried unchanged.
        same_failure_gate(_prior_attempt_logs,
                          corrective_change=_corrective_change_since),
        #: LAST, because it is the only gate that downloads anything.
        bundle_staged_gate,
    )


def budget_spec(repo_root: Path, rung: str) -> BudgetSpec:
    """The `BudgetSpec` the runner plans from, DECOMPOSING the accepted bound.

    The accepted behavioural bound is the design's priced cell, which already
    carries its own overrun factor -- so the reserve is DERIVED as whatever
    makes the plan land exactly on that bound, exactly as the search launcher
    does. Two models of one session is the defect; there is one total.
    """
    from aadistill.infrastructure.budget import MEASURED_STEP_SECONDS, Phase

    cell = BA.session_cell(rung, repo_root)
    overhead = cell["session_overhead_minutes"]
    accepted_hard = cell["hard_ceiling_minutes"]
    expected_total = cell["expected_minutes"]
    contingency = 0.10
    recovery_reserve = 30.0
    reserve = round(accepted_hard - expected_total * (1.0 + contingency)
                    - recovery_reserve, 2)
    if reserve < 0:
        raise SystemExit(
            f"the accepted {accepted_hard:.2f}-minute bound cannot hold the "
            f"expected {expected_total:.2f} min plus a {contingency:.0%} "
            f"contingency and a {recovery_reserve:.0f}-minute recovery "
            "reserve. Obtain a larger authorization or a smaller session; do "
            "not shrink the reserve to fit.")

    return BudgetSpec(
        arms=0, steps_per_arm=0,
        step_seconds=MEASURED_STEP_SECONDS,
        step_source=("unused: the generic arms x steps shape does not "
                     "describe arm materializations followed by probes; every "
                     "phase is named below, from the design's priced cell"),
        setup_minutes=round(overhead * 0.75, 2),
        transfer_minutes=round(overhead * 0.25, 2),
        other_phases=(
            Phase("arm_materialization",
                  round(cell["arm_materialization_minutes"], 2)),
            Phase("probe_train_eval_score_expected",
                  round(cell["probe_minutes"], 2)),
        ),
        contingency_fraction=contingency,
        soft_stop_reserves=(Phase("probe_overrun_risk", reserve),),
        artifact_recovery_reserve_minutes=recovery_reserve,
        below_floor_reason=(
            f"priced at ${cell['hard_ceiling_usd']:.4f} / "
            f"{accepted_hard:.2f} min by the design's budget.chain.sessions."
            f"{rung}; this spec DECOMPOSES that bound and does not add to it"),
        #: WHAT THE PROVIDER ACCOUNT MUST HOLD: the full session ceiling plus
        #: a small operational reserve, floored at the package's per-attempt
        #: envelope -- the same two-term rule the search derived after losing
        #: $8.1716 to an exhausted account balance.
        account_balance_required_usd=round(max(
            float(BA.live_money(repo_root)["per_session_envelope_usd"]),
            float(cell["hard_ceiling_usd"])
            + ACCOUNT_OPERATIONAL_RESERVE_USD), 4),
    )


def spec(args) -> SessionSpec:
    rung = getattr(args, "rung", "screening")
    return SessionSpec(
        session_id="autoinit-d1-behavioural",
        schema="aadistill.autoinit.d1_behavioural_session/v1",
        description=(
            "One D1 behavioural rung under the frozen design: the arms "
            "materialized on the pod along digest-pinned paths, the probes "
            "trained under the frozen recipe at the preregistered seeds, "
            "evaluated uncapped on the rung's own realized battery, scored by "
            "the rung's pinned scorer, and -- for screening -- mechanically "
            "ranked with exactly one candidate advanced, or none. It runs no "
            "beam, re-selects nothing, re-measures no frozen result, and "
            "computes no promotion."),
        authorization_path=auth_path_for(getattr(args, "run_id", None)),
        authorization_loader=D1BehaviouralAuthorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/shared/pod/watchdog.py",
            setup_script="scripts/shared/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/shared/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id="autoinit.v1.phase_d1_behavioural",
        #: THE DESIGN HASH is the plan hash: the dispatch branch calls
        #: `require_plan` with it, so an authorization issued against another
        #: design revision is refused at exit 98 before any work.
        plan_hash=D1B.design(REPO_ROOT)["design_hash"],
        budget=budget_spec(REPO_ROOT, rung),
        setup=SetupManifest(
            env={"SESSION_KIND": SESSION_KIND},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_KIND", "TEACHER_REVISION"),
            #: EVERY STEP THIS SESSION NEEDS, and deliberately not one more.
            #: `VLLM_READY` is PRESENT -- both rungs generate through vLLM.
            #: `TEACHER_READY` is PRESENT -- the candidate replays start from
            #: the pinned teacher and the KD trainer reads it.
            #: `ROPE_OK` stays undeclared: it globs a staged checkpoint config
            #: and this session stages no checkpoint -- the arms are
            #: materialized during the driver, after setup.
            #: `ASSETS_READY` stays undeclared, as the search left it: every
            #: input is digest-verified against the pin it ships, at $0 by
            #: `staged_science_inputs_gate` and on the pod by `battery_role`,
            #: `profile.resolve()`, the recipe's pack digest and
            #: `verify_staged_teacher`. A second sweep over the same question
            #: is the duplicate validation layer AGENTS.md P8.2.1 forbids.
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "TEACHER_READY", "VLLM_READY",
                           "TESTS_OK", "AUTHORIZATION_OK", "SETUP_DONE"),
            local_assets=SCIENCE_ASSETS,
            relay_inputs=RELAY_INPUTS,
            teacher_revision=D1S.root_teacher_identity(REPO_ROOT)["revision"],
            test_paths=TEST_PATHS,
            uv_max_seconds=getattr(args, "uv_max_s", 1500),
            tests_max_seconds=getattr(args, "tests_max_s", 2700)),
        driver_command=driver_command,
        #: The replacement-resource handoff for P8.4 state 2, AFTER setup and
        #: BEFORE the driver: preserved trained-unscored probes are pushed
        #: back so the driver resumes them at evaluation instead of
        #: retraining. Returns False -- and the runner tears down -- on a
        #: half-arrived restore.
        materialize_inputs=restore_trained_probes,
        driver_job_id="autoinit_d1_behavioural_driver",
        status_path=STATUS,
        run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("RUN_FAILED", "DIGEST_MISMATCH"),
            #: WHEN THIS SESSION'S PRODUCTS EXIST: the driver writes its
            #: record on every path out, and any SCORED probe is evidence
            #: whether or not the probes after it ran.
            products_eligible=lambda terminal, stages: True,
            failure_note=(
                "the session stopped before completing its rung. Whatever "
                "probes SCORED are secured first. A DIGEST_MISMATCH on an arm "
                "is a scientific finding, not a retryable engineering "
                "failure, and goes to review unchanged.")),
        precheck=prechecks(args),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT_DIRNAME,
            evidence_filename="d1_behavioural.json",
            archive_basename="d1_behavioural_artifacts.tar.gz",
            spec_success=("configs/stages/stage-1/phase_d1/"
                          "d1_behavioural_artifacts.json"),
            spec_failed=("configs/stages/stage-1/phase_d1/"
                         "d1_behavioural_artifacts_failed.json"),
            report_names=("d1_behavioural.json",),
            on_poll=secure_probe_evidence,
            fetch_products=fetch_probe_evidence,
            products_secured=probes_evidence_secured),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, "
                 "STOP"),
    )


def build_parser() -> argparse.ArgumentParser:
    """The real parser, at module scope so a test can build it. Every required
    argument is a STRING: the dispatch probe fills them with one."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scr", required=True,
                    help="the session's local scratch directory")
    ap.add_argument("--session-commit", required=True,
                    help="the commit the pod checks out; the authorization "
                         "binds it and the lineage gate verifies it")
    ap.add_argument("--bundle", required=True,
                    help="must be the canonical name derived from "
                         "--session-commit; an alias fails at $0")
    ap.add_argument("--runpod-config",
                    default=str(Path("~/.runpod/config.toml").expanduser()))
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rung", default="screening",
                    choices=("screening", "confirmation"))
    ap.add_argument("--gpu", default="NVIDIA L40S")
    #: NOT `required` and NOT typed-required: the dispatch probe fills every
    #: required option with a string. Defaults resolve from the priced cell.
    ap.add_argument("--max-price", type=float, default=None,
                    help="omit to use the rate the derived ceiling was priced "
                         "at")
    ap.add_argument("--disk-gb", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--token-src",
                    default=str(Path("~/.cache/huggingface/token").expanduser()))
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    #: ONE create call per acquisition invocation, by the package's own rule.
    ap.add_argument("--create-attempts", type=int, default=1)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    #: MUST OUTLAST THE HARD THRESHOLD. Screening's own bound is 1355.71 min.
    ap.add_argument("--poll-limit-min", type=float, default=1600.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--ckpt-store", default=None,
                    help="where trained-but-unscored probes are preserved on "
                         "a failure path; defaults to <scr>/products")
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=45,
                    help="per-checkpoint preservation transfer timeout")
    ap.add_argument("--restore-trained", default=None,
                    help="a directory of preserved trained-unscored probes "
                         "(the layout preserve_trained_unscored writes) to "
                         "push back onto a REPLACEMENT pod before the driver "
                         "starts; the driver then resumes each at evaluation "
                         "after re-hashing it against its sidecar")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every $0 gate through the real SessionRunner "
                         "and stop before provider creation; create nothing")
    return ap


def resolve_operational_defaults(args) -> dict[str, Any]:
    """Fill the arguments that default to `None` and that the runner READS.

    All from the priced cell rather than constants, so a launch is priced at
    the rate its authorization was derived at and carries the disk the cost
    model charged for.
    """
    cell = BA.session_cell(getattr(args, "rung", "screening"), REPO_ROOT)
    resolved: dict[str, Any] = {}
    if getattr(args, "max_price", None) is None:
        args.max_price = float(cell["price_per_hour"])
        resolved["max_price"] = args.max_price
    if getattr(args, "disk_gb", None) is None:
        args.disk_gb = int(cell["container_disk_gb"])
        resolved["disk_gb"] = args.disk_gb
    if getattr(args, "out", None) is None:
        args.out = (f"{run_dir_for(getattr(args, 'run_id', None))}"
                    "/runtime/session.json")
        resolved["out"] = args.out
    if not args.disk_gb or args.disk_gb < 1:
        raise SystemExit(
            f"the priced cell carries container_disk_gb={args.disk_gb!r}; a "
            "pod cannot be created without a disk size")
    return resolved


def main(argv: list[str] | None = None) -> int:
    from aadistill.governance.authorization import AuthorizationError
    from aadistill.infrastructure.session_runner import (
        SessionRunner, run_session,
    )

    args = build_parser().parse_args(argv)
    resolved = resolve_operational_defaults(args)
    try:
        session = spec(args)
    except AuthorizationError as exc:
        print(f"REFUSED at $0: {exc}")
        print("No provider resource was created and nothing bills.")
        return 11

    if args.dry_run:
        cell = BA.session_cell(args.rung, REPO_ROOT)
        print(f"session_id      {session.session_id}")
        print(f"plan_hash       {session.plan_hash[:24]}")
        print(f"authorization   {session.authorization_path}")
        print(f"priced ceiling  ${cell['hard_ceiling_usd']:.4f} "
              f"({cell['hard_ceiling_minutes']:.2f} min, {args.rung})")
        print(f"SESSION_KIND    {session.setup.env['SESSION_KIND']}")
        print(f"test_paths      {list(session.setup.test_paths)}")
        print(f"local_assets    "
              f"{[a.as_env_entry() for a in session.setup.local_assets]}")
        print(f"setup_markers   {list(session.setup.setup_markers)}")
        print(f"prechecks       "
              f"{[getattr(c, '__name__', str(c)) for c in session.precheck]}")
        print(f"resolved        {resolved}")
        print()
        try:
            runner = SessionRunner(session, args, REPO_ROOT)
        except AuthorizationError as exc:
            print(f"REFUSED at $0: {exc}")
            print("No provider resource was created and nothing bills.")
            return 11
        runner.run()
        terminal = runner.ev.get("terminal")
        passed = terminal == "DRY_RUN_GATES_PASSED"
        runner.ev["passed"] = False
        runner.save()
        print(f"\nterminal        {terminal}")
        print(f"record          {args.out}")
        print("CREATED NOTHING: every pre-provider gate "
              f"{'PASSED' if passed else 'did NOT pass'}; "
              "no provider resource was created and nothing bills.")
        if passed:
            print("A DRY RUN IS NOT A PASS: no pod existed, no stage ran and "
                  "nothing was measured. It is evidence the chain is "
                  "launchable.")
        return 0 if passed else 11

    try:
        return run_session(
            session, args, REPO_ROOT,
            summary=("D1 behavioural rung: every scored probe's evidence and "
                     "the session record are what this session owes."))
    except AuthorizationError as exc:
        print(f"REFUSED at $0: {exc}")
        print("No provider resource was created and nothing bills.")
        return 11


if __name__ == "__main__":
    raise SystemExit(main())
