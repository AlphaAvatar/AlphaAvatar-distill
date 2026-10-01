#!/usr/bin/env python3
"""Launch the complete A3 chain. Every refusal it can make costs `$0`.

    PYTHONPATH=src:scripts python scripts/pod/autoinit_a3_launch.py \
        --run-id a3-attempt1 --gpu "NVIDIA L40S" --max-price 1.09

One session, one experiment: the frozen parent under its digest gate, the
A-bsz1 incumbent gate, interleaved A-bsz1/A-bsz3 diagnostics, three A-bsz3
recovery probes, three evaluations, preservation. **No decision is computed on
the pod** — `scripts/autoinit/aggregate_a3.py` runs the comparison off pod at
`$0`, which is why attempt75's failure mode is absent rather than guarded.

**One status path, named once.** `A3S.STATUS_PATH` is read by the driver and by
this launcher from the same owner. The C3 pair hardcoded two different files,
so every driver marker — `ALL_DONE` included — was invisible, and the
acquisition loop would have read a completed formal run as "no measurement
began" and launched a second paid attempt.

**The gates in cost order.** Local refusals first, the one network gate last,
and the account-wide billing check immediately before any create call.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pod"))

from aadistill.infrastructure.manifest import sha256_file  # noqa: E402
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, ExecutionCommands, LocalAsset, MarkerPolicy, RelayInput,
    SessionContext, SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    local_files_gate, session_commit_gate,
)
from aadistill.infrastructure.session_runner import run_session  # noqa: E402
from autoinit_science_inputs import (  # noqa: E402
    CALIBRATION_V1, RECOVERY_LADDER,
)
from experiments.deployment import (  # noqa: E402
    MAIN_RELAY, POD_IMAGE, deployment_commands,
)
from experiments.phase_c1.authorization_payload import load_config  # noqa: E402
from experiments.run_layout import rel_run_dir  # noqa: E402
from experiments.phase_c3 import a3_session as A3S  # noqa: E402
from experiments.phase_c3.a3_authorization import (  # noqa: E402
    A3Authorization, a3_budget_spec, a3_harness_digest, a3_hard_ceiling_usd,
    load_live_pricing,
)

WS = POD_IMAGE["workspace_root"]
REPO = POD_IMAGE["checkout_root"]

RUN_EXPERIMENT_ID = "phase_a3"
#: The pipeline stage A3's runs EXECUTE, READ from the experiment's own
#: configuration rather than decided here. A3 trains and evaluates Stage-3
#: recovery probes to answer a Stage-1 question; `a3_chain` is a phase name,
#: not a stage id, and `rel_run_dir` correctly refused it.
RUN_STAGE_ID = load_config(REPO_ROOT)["stage_id"]
SPEC_SUCCESS = "configs/autoinit/a3_artifacts.json"
SPEC_FAILED = "configs/autoinit/a3_artifacts_failed.json"

#: attempt75's control evidence, which the OFF-POD comparison reads. Checked
#: here at $0 because a session that trains three probes and then cannot be
#: compared to anything has produced half an experiment.
STAGE_I = REPO_ROOT / "logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i"

#: The evaluation tokenizer, pinned by file hash. These exact bytes are what
#: the Stage-3 attestation's `chat_template_sha256` was observed over; the
#: teacher's own tokenizer is a DIFFERENT artifact and cannot substitute.
A3_EVAL_TOKENIZER: tuple[RelayInput, ...] = tuple(
    RelayInput(f"stage1/qwen3_0p6b_init_v0/checkpoint/{name}",
               dest="artifacts/stage1/qwen3_0p6b_init_v0/checkpoint",
               sha256=sha, repo=MAIN_RELAY)
    for name, sha in (
        ("tokenizer.json",
         "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"),
        ("tokenizer_config.json",
         "8fa82a4ba512c8bee7c1c5e82b9a71ddbef362e4665be5c8f7ce0afd78af129a"),
        ("chat_template.jinja",
         "3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7"),
    )
)

#: SETUP READINESS, not a measurement input. The shared setup's `ROPE_OK` step
#: globs `artifacts/stage1/*/checkpoint/config.json` and loads each match in
#: BOTH venvs. C1 attempt 2 staged the three sidecars above and nothing else,
#: so the glob was empty and setup exited `no staged checkpoint to check`
#: AFTER the teacher had been fetched and verified -- $0.1013. 1,418 bytes,
#: against the 1.19 GiB of weights the full group would pull for a check that
#: reads none of them.
A3_ROPE_INPUT: tuple[RelayInput, ...] = (
    RelayInput("stage1/qwen3_0p6b_init_v0/checkpoint/config.json",
               dest="artifacts/stage1/qwen3_0p6b_init_v0/checkpoint",
               sha256="a7131bb092b38a078edc213961f0eb57eaead24f1396e25741f4887b1a694054",
               repo=MAIN_RELAY),
)

#: Declared because the SHARED setup requires them, not because A3 reads them:
#: `verify_frozen_assets.py` runs unconditionally at ASSETS_READY and checks
#: all four. Declaring only what a session reads is what cost two earlier
#: sessions their setup.
A3_LOCAL_ASSETS = (
    LocalAsset("artifacts/stage3/c1_confirmation_v1", "c1_confirmation_v1",
               "artifacts/stage3"),
    LocalAsset("artifacts/stage1/reasoning_heavy_v2", "reasoning_heavy_v2",
               "artifacts/stage1"),
    LocalAsset("artifacts/stage1/state_eval_v1", "state_eval_v1",
               "artifacts/stage1"),
    LocalAsset("artifacts/stage3/recovery_search_v2", "recovery_search_v2",
               "artifacts/stage3"),
)


def auth_path_for(run_id: str | None) -> str:
    return (f"{rel_run_dir(RUN_EXPERIMENT_ID, run_id or '_', RUN_STAGE_ID)}"
            "/governance/authorization.json")


def grant_path_for(run_id: str | None) -> str:
    return (f"{rel_run_dir(RUN_EXPERIMENT_ID, run_id or '_', RUN_STAGE_ID)}"
            "/governance/grant.json")


def readiness_path_for(run_id: str | None) -> str:
    return (f"{rel_run_dir(RUN_EXPERIMENT_ID, run_id or '_', RUN_STAGE_ID)}"
            "/governance/readiness.json")


def bundle_record_for(run_id: str | None) -> str:
    return (f"{rel_run_dir(RUN_EXPERIMENT_ID, run_id or '_', RUN_STAGE_ID)}"
            "/governance/bundle.json")


# ---------------------------------------------------------------------------
# prechecks — everything that can refuse before a pod exists
# ---------------------------------------------------------------------------

def a3_harness_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The grant measured a harness; this tree must still be that harness."""
    want = ctx.auth.harness_source_digest
    if not want:
        return False, ("the authorization names no harness digest, so nothing "
                       "binds the executable this session would run")
    got = a3_harness_digest(REPO_ROOT)
    ctx.evidence["a3_harness"] = {"expected": want, "observed": got["digest"],
                                  "n_files": len(got["files"])}
    if got["digest"] != want:
        moved = [f["path"] for f in got["files"]
                 if f.get("sha256") != (ctx.auth.harness_source_files and None)]
        return False, (
            f"harness digest {got['digest'][:12]} != granted {want[:12]}. The "
            "grant authorizes the launcher, driver and comparison it measured; "
            "a repair moves the closure and owes a fresh chain. "
            f"{len(got['files'])} files declared.")
    return True, f"harness {got['digest'][:12]} matches the grant"


def grant_provenance_gate(ctx: SessionContext) -> tuple[bool, str]:
    """A grant exists, is recorded, and is this run's."""
    p = REPO_ROOT / grant_path_for(getattr(ctx.args, "run_id", None))
    if not p.is_file():
        return False, (f"no grant at {p.relative_to(REPO_ROOT)}; a session is "
                       "not issuable without one")
    doc = json.loads(p.read_text())
    ctx.evidence["grant_provenance"] = {
        "path": str(p.relative_to(REPO_ROOT)), "sha256": sha256_file(p),
        "granted_utc": doc.get("granted_utc"),
        "run_id": doc.get("run_id")}
    if doc.get("run_id") != getattr(ctx.args, "run_id", None):
        return False, (f"the grant names run {doc.get('run_id')!r} and this "
                       f"session is {getattr(ctx.args, 'run_id', None)!r}; a "
                       "one-use chain is not transferable")
    if doc.get("experiment_id") != RUN_EXPERIMENT_ID:
        return False, (f"the grant names experiment "
                       f"{doc.get('experiment_id')!r}, not {RUN_EXPERIMENT_ID}")
    return True, f"grant {sha256_file(p)[:12]} recorded for this run"


def pricing_identity_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The LIVE price funds this session, under EVERY applicable limit.

    `load_live_pricing` refuses an absent or unfundable record AND refuses one
    that did not evaluate the package total — which equals formal plus
    engineering, and is exactly why nothing used to check it.
    """
    try:
        doc = load_live_pricing(REPO_ROOT)
    except Exception as exc:                                  # noqa: BLE001
        return False, str(exc)
    ceiling = a3_hard_ceiling_usd(REPO_ROOT)
    ctx.evidence["pricing"] = {
        "gpu": doc.get("gpu"),
        "rate_usd_per_hour": doc.get("queried_rate_usd_per_hour"),
        "hard_ceiling_usd": ceiling,
        "expected_usd": doc["price"]["expected"]["usd"],
        "container_disk_gb": doc["price"]["container_disk_gb"],
        "conditions": doc["conditions"],
        "limits_checked": doc["_every_applicable_limit_is_checked"]}
    if abs(float(ctx.auth.hard_cap_usd) - ceiling) > 1e-4:
        return False, (
            f"the authorization carries ${ctx.auth.hard_cap_usd:.4f} and the "
            f"live price derives ${ceiling:.4f}. The grant receives the "
            "DERIVED ceiling, never an envelope, and a re-quote that moved the "
            "price owes a re-issued authorization.")
    rate = float(getattr(ctx.args, "max_price", 0) or 0)
    quoted = float(doc.get("queried_rate_usd_per_hour") or 0)
    if rate and quoted and rate > quoted + 1e-9:
        return False, (f"--max-price {rate} exceeds the priced rate {quoted}; "
                       "a session may not be authorized at one rate and run "
                       "at a higher one")
    return True, (f"live ${quoted}/h -> hard ${ceiling:.4f}, "
                  f"{len(doc['_every_applicable_limit_is_checked'])} limits "
                  "checked including the package total")


def a3_design_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The frozen design hashes to itself, and the grant binds that hash."""
    try:
        design = A3S.design(REPO_ROOT)
    except Exception as exc:                                  # noqa: BLE001
        return False, str(exc)
    contract = A3S.A3_SESSION_CONTRACT
    ctx.evidence["a3_design"] = {
        "design_sha256": design["design_sha256"],
        "session_contract_hash": contract.contract_hash,
        "probes": contract.n_probes, "arms": contract.n_arms,
        "seeds": list(A3S.recovery_seeds()),
        "aggregation_off_pod": contract.aggregation_off_pod}
    if design["recovery"]["probes"] != contract.n_probes:
        return False, (f"the design declares {design['recovery']['probes']} "
                       f"probes and the session contract {contract.n_probes}")
    if not contract.aggregation_off_pod:
        return False, ("the session contract claims an on-pod aggregation; A3's "
                       "comparison runs off pod and the driver has no stage "
                       "for a decision")
    if design["control"]["retrained"]:
        return False, ("the design claims the controls are retrained; A3 "
                       "reuses attempt75's three and the grant funds three "
                       "probes, not six")
    return True, (f"design {design['design_sha256'][:12]}, "
                  f"{contract.n_arms} arm x {len(A3S.recovery_seeds())} seeds")


def controls_evidence_gate(ctx: SessionContext) -> tuple[bool, str]:
    """attempt75's control evidence is present and still hashes correctly.

    Checked BEFORE a pod exists. A3's whole saving is that it does not retrain
    the controls, so a session that trains three probes and then finds the
    comparison has nothing to compare against has produced half an experiment
    at full price. The weights were retired on 2026-10-01 and are NOT what is
    checked here: a control contributes per-sample rows and a scored record.
    """
    probes = json.loads((STAGE_I / "c3_probe_results.json").read_text())
    controls = [p for p in probes["probes"] if p["arm"] == "A_incumbent"]
    if len(controls) != 3:
        return False, (f"expected three A_incumbent controls in attempt75, "
                       f"found {len(controls)}")
    bad, checked = [], []
    for probe in controls:
        for key, digest_key in (("per_sample_path", "per_sample_sha256"),
                                ("result_path", "result_sha256")):
            p = Path(probe[key])
            if not p.is_file():
                bad.append(f"seed {probe['seed']}: {key} absent at {p}")
                continue
            got = sha256_file(p)
            if got != probe[digest_key]:
                bad.append(f"seed {probe['seed']}: {key} hashes {got[:12]} "
                           f"and the record says {probe[digest_key][:12]}")
            else:
                checked.append(f"{probe['seed']}:{key}")
    ctx.evidence["controls_evidence"] = {
        "seeds": sorted(p["seed"] for p in controls),
        "files_verified": len(checked), "problems": bad,
        "_weights_are_not_checked": (
            "retired 2026-10-01 under the consumer rule; the comparison reads "
            "evidence, never bytes")}
    if bad:
        return False, ("attempt75's control evidence cannot be used: "
                       + "; ".join(bad))
    return True, (f"three controls verified, {len(checked)} files by content "
                  "hash")


def teacher_binding_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The teacher binding pins the revision this session declares."""
    from experiments.phase_c3 import session as CS

    p = REPO_ROOT / "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"
    if not p.is_file():
        return False, f"no teacher binding at {p.relative_to(REPO_ROOT)}"
    doc = json.loads(p.read_text())
    ctx.evidence["teacher_binding"] = {
        "revision": doc.get("revision"), "shards": len(
            doc.get("expected_shard_sha256") or {})}
    if doc.get("revision") != CS.TEACHER_REVISION:
        return False, (f"the binding pins {doc.get('revision')} and the "
                       f"session declares {CS.TEACHER_REVISION}")
    if not doc.get("expected_shard_sha256"):
        return False, "the binding names no shard hashes, so stage B verifies nothing"
    return True, (f"teacher {CS.TEACHER_REVISION[:12]} bound over "
                  f"{len(doc['expected_shard_sha256'])} shards")


def battery_staged_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The confirmation battery is staged and is the one attempt75 used."""
    battery = REPO_ROOT / "artifacts/stage3/c1_confirmation_v1"
    manifest = battery / "manifest.json"
    if not manifest.is_file():
        return False, (f"the confirmation battery is not staged at "
                       f"{battery.relative_to(REPO_ROOT)}")
    doc = json.loads(manifest.read_text())
    probes = json.loads((STAGE_I / "c3_probe_results.json").read_text())
    want = probes["battery"]
    ctx.evidence["battery"] = {
        "artifact": doc.get("artifact"),
        "content_sha256": doc.get("content_sha256"),
        "expected_content_sha256": want["content_sha256"]}
    if doc.get("content_sha256") != want["content_sha256"]:
        return False, (
            f"the staged battery content hashes {str(doc.get('content_sha256'))[:12]} "
            f"and attempt75's controls were scored on {want['content_sha256'][:12]}. "
            "A3 pools its probes with those controls, so a different battery "
            "would pool two fields as one.")
    return True, (f"battery {doc.get('artifact')} staged, content "
                  f"{want['content_sha256'][:12]} matches the controls")


def artifact_spec_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Both specs load, and the success spec's counts match the design.

    A spec that required nine probes of a three-probe session would classify a
    complete run as incomplete; one that required three of a nine-probe session
    would call a third of an experiment finished.
    """
    from collect_artifacts import load_specs

    problems, counts = [], {}
    #: `load_specs` takes ONE path and returns that file's entries. Reading
    #: both separately is what the signature actually offers; guessing a
    #: two-argument form is the shape of defect this gate exists to catch
    #: before a pod reads these files at teardown.
    for name, rel in (("success", SPEC_SUCCESS), ("failed", SPEC_FAILED)):
        path = REPO_ROOT / rel
        if not path.is_file():
            problems.append(f"{name}: {rel} is absent, and the pod first reads "
                            "it at teardown, after the money is spent")
            continue
        try:
            counts[name] = len(load_specs(str(path)))
        except Exception as exc:                              # noqa: BLE001
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
    n_probes = A3S.A3_SESSION_CONTRACT.n_probes
    success = json.loads((REPO_ROOT / SPEC_SUCCESS).read_text())
    probe_entries = [e for e in success["entries"]
                     if e.get("required") and "A_bsz3.*" in e["pattern"]
                     and e.get("min_matches") not in (None, 0)]
    wrong = [e["pattern"] for e in probe_entries
             if e["min_matches"] % n_probes != 0]
    if wrong:
        problems.append(f"per-probe min_matches not a multiple of {n_probes}: "
                        f"{wrong}")
    failed = json.loads((REPO_ROOT / SPEC_FAILED).read_text())
    if any(e.get("required") for e in failed["entries"]):
        problems.append("the FAILED spec requires an artifact; an early "
                        "failure may legitimately have produced none")
    if any("decision" in e["pattern"] for e in success["entries"]):
        problems.append("the success spec requires a decision artifact, which "
                        "A3's driver does not produce: the comparison runs off "
                        "pod")
    ctx.evidence["artifact_specs"] = {"entries": counts, "problems": problems}
    if problems:
        return False, "; ".join(problems)
    return True, (f"both specs load ({counts}); per-probe counts are multiples "
                  f"of {n_probes}; the failed spec requires nothing")


def readiness_gate(ctx: SessionContext) -> tuple[bool, str]:
    """A launch-bound readiness record exists for the commit about to run."""
    p = REPO_ROOT / readiness_path_for(getattr(ctx.args, "run_id", None))
    if not p.is_file():
        return False, (f"no readiness record at {p.relative_to(REPO_ROOT)}; "
                       "the chain is grant -> launch-bound readiness -> "
                       "authorization -> bundle -> gates -> provider")
    doc = json.loads(p.read_text())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    ctx.evidence["readiness"] = {
        "record_kind": doc.get("record_kind"), "verdict": doc.get("verdict"),
        "swept_base_commit": doc.get("swept_base_commit"), "head": head}
    #: `record_kind`, not `kind`. A gate asked for `kind` once, no record this
    #: repository writes carries that field, and it could not have passed for
    #: any record.
    if doc.get("record_kind") != "launch_bound":
        return False, (f"the readiness record is "
                       f"{doc.get('record_kind')!r}, not launch_bound")
    if doc.get("verdict") != "PASS":
        return False, (f"the readiness verdict is {doc.get('verdict')!r}; a "
                       "FAILED sweep does not satisfy this gate")
    if doc.get("swept_base_commit") != head:
        return False, (f"the sweep describes {str(doc.get('swept_base_commit'))[:12]} "
                       f"and HEAD is {head[:12]}; a launch-bound record "
                       "describes the tree a launch will use")
    return True, f"launch-bound readiness PASS at {head[:12]}"


def bundle_staged_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The exact-session bundle is on the relay and is fetchable.

    Eight gates once verified the commit and none verified that a bundle could
    be FETCHED; contents are not obtainability.
    """
    p = REPO_ROOT / bundle_record_for(getattr(ctx.args, "run_id", None))
    if not p.is_file():
        return False, f"no bundle record at {p.relative_to(REPO_ROOT)}"
    doc = json.loads(p.read_text())
    ctx.evidence["bundle"] = doc
    for key in ("bundle_name", "relay_repo", "relay_path", "sha256",
                "session_commit"):
        if not doc.get(key):
            return False, f"the bundle record names no {key}"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    if doc["session_commit"] != head:
        return False, (f"the bundle was staged for {doc['session_commit'][:12]} "
                       f"and HEAD is {head[:12]}")
    if not doc.get("fetch_verified"):
        return False, ("the bundle record does not assert that the object was "
                       "fetched back and re-identified; a staged-but-"
                       "unfetchable bundle fails on the pod at full price")
    return True, (f"bundle {doc['bundle_name']} staged and fetch-verified at "
                  f"{head[:12]}")


def durable_capacity_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Would the relay accept this session's probes? Asked before spending.

    AGENTS.md names this failure twice: C1 attempt 18 lost six probes and C3
    attempt66 lost nine, both to a private-storage limit, with the durability
    mechanism behaving perfectly and nowhere to write. Asked at the real
    per-probe size and the real probe count, with FRESH random oids, because
    an oid that already exists dedups and returns a false pass.
    """
    from aadistill.runtime.hub_capacity import would_accept

    n = A3S.A3_SESSION_CONTRACT.n_probes
    per_probe = 2_384_236_592        # measured, attempt66's preservation payload
    try:
        token = Path(
            "~/.cache/huggingface/token").expanduser().read_text().strip()
        verdict = would_accept("AlphaAvatar/aadistill-artifacts",
                               [per_probe] * n, token)
    except Exception as exc:                                  # noqa: BLE001
        return False, (f"could not ask the durable backend whether it has "
                       f"room: {type(exc).__name__}: {exc}. 'Could not ask' "
                       "and 'was refused' must not look alike to a caller "
                       "deciding whether to spend.")
    ctx.evidence["durable_capacity"] = {
        "n_probes": n, "bytes_each": per_probe,
        "gib_total": round(n * per_probe / 2 ** 30, 4),
        "accepted": verdict.accepted}
    if not verdict.accepted:
        return False, (
            f"the relay would refuse {n} probes "
            f"({n * per_probe / 2 ** 30:.2f} GiB). A session that trains them "
            "and cannot preserve them has produced unpreservable work.")
    return True, (f"the relay accepts {n} probes "
                  f"({n * per_probe / 2 ** 30:.2f} GiB)")


def no_billing_resource_gate(ctx: SessionContext) -> tuple[bool, str]:
    """At most ONE billing resource ever, and right now there must be none.

    The last gate before a create call. Account-wide, not scoped to this
    session: the rule is about the account's exposure, and a resource another
    chain left running is exactly what this must catch.
    """
    import urllib.request

    from aadistill.infrastructure.provider import USER_AGENT, read_api_key

    key = read_api_key(str(Path("~/.runpod/config.toml").expanduser()))
    query = "query { myself { pods { id desiredStatus costPerHr } } }"
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={key}",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as fh:
            doc = json.loads(fh.read().decode())
    except Exception as exc:                                  # noqa: BLE001
        return False, (f"could not establish whether anything is billing: "
                       f"{type(exc).__name__}: {exc}. Unknown ownership is a "
                       "refusal, never a $0.0000.")
    pods = ((doc.get("data") or {}).get("myself") or {}).get("pods") or []
    ctx.evidence["pre_create_billing_check"] = {
        "pods": [{"id": p.get("id"), "desiredStatus": p.get("desiredStatus"),
                  "costPerHr": p.get("costPerHr")} for p in pods]}
    if pods:
        return False, (f"{len(pods)} provider resource(s) exist: "
                       f"{[p.get('id') for p in pods]}. At most one billing "
                       "resource at any instant, and this session would be a "
                       "second.")
    return True, "no provider resource exists; nothing is billing"


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------

def driver_command(ctx: SessionContext, plan: Any) -> str:
    """The exact command the pod runs, from the context and the plan.

    No `--stage`. `run()` executes the complete fixed sequence B..H, and a
    stage-selection surface would make partial A3 execution expressible when
    it is not an authorized control. C1 attempt 7 cleared the pod test gate
    for the first time ever and then died at $0.4231 because this function
    emitted a flag the driver's parser has no option for: argparse exited 2
    before a line of the driver ran. A test parses this exact string with the
    driver's OWN parser, which is the authority on what it accepts.
    """
    return (f"/opt/train/bin/python {REPO}/scripts/pod/autoinit_a3_driver.py "
            f"--image-digest '{ctx.image_digest}' "
            f"--run-id '{getattr(ctx.args, 'run_id', None) or 'unrecorded'}' "
            f"--rate {ctx.price or ctx.args.max_price} "
            f"--spent-usd {ctx.spent_usd:.4f} "
            f"--soft-stop-usd {plan.soft_stop_usd:.4f} "
            f"--authorized-usd {ctx.auth.hard_cap_usd:.4f}")


def probe_streams(ctx: SessionContext) -> tuple[str, ...]:
    return tuple(
        f"artifacts/stage3/a3/{pid}/train_log.jsonl"
        for pid in A3S.probe_ids())


def spec(args) -> SessionSpec:
    from experiments.phase_c3 import session as CS
    from experiments.phase_c3.hardware import require_approved

    require_approved(args.gpu)
    return SessionSpec(
        session_id="autoinit-a3",
        schema="aadistill.autoinit.a3_session/v1",
        description=(
            "A3: the incumbent ATTENTION operator under bsz3 + "
            "length_sorted_v1, end to end. Replays the frozen parent under "
            "its digest gate, verifies A-bsz1 rebuilds the incumbent, "
            "collects interleaved diagnostics, trains THREE A-bsz3 probes at "
            "the frozen seeds and evaluates each once. Computes no decision: "
            "the comparison runs off pod at $0. Runs no search."),
        authorization_path=auth_path_for(getattr(args, "run_id", None)),
        authorization_loader=A3Authorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/pod/watchdog.py",
            setup_script="scripts/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=f"autoinit.v1.{A3S.EXPERIMENT_ID}",
        plan_hash=A3S.A3_SESSION_CONTRACT.contract_hash,
        budget=a3_budget_spec(REPO_ROOT),
        setup=SetupManifest(
            relay_inputs=(*RECOVERY_LADDER, *CALIBRATION_V1,
                          *A3_EVAL_TOKENIZER, *A3_ROPE_INPUT),
            local_assets=A3_LOCAL_ASSETS,
            #: DECLARED. Without it setup falls through to SESSION_KIND=spend,
            #: loads a SpendAuthorization and refuses this artifact at exit 98
            #: before any work. A missing dispatch entry is not a type error.
            env={"SESSION_KIND": "a3"},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "SESSION_RELAY_INPUTS",
                          "SESSION_KIND", "TEACHER_REVISION"),
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "ASSETS_READY", "VLLM_READY",
                           "TEACHER_READY", "ROPE_OK", "TESTS_OK",
                           "AUTHORIZATION_OK", "SETUP_DONE"),
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=CS.TEACHER_REVISION),
        driver_command=driver_command,
        #: FROM THE ONE OWNER, both of them. The C3 pair hardcoded a driver
        #: job id and a status path that disagreed with the driver's, and
        #: every marker was invisible.
        driver_job_id=A3S.DRIVER_JOB_ID,
        status_path=A3S.STATUS_PATH,
        run_log_path=A3S.RUN_LOG_PATH,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("A3_FAILED", "A3_REPLAY_MISMATCH",
                     "A3_INTEGRITY_FAILURE"),
            incomplete=(),
            failure_note=(
                "a blocking A3 stage failed — collecting evidence, then "
                "tearing down. Classify from the explicit terminal marker and "
                "the collected driver evidence; this line asserts nothing "
                "about which stage failed or why, and in particular asserts "
                "NOTHING about whether the frozen path reproduced.")),
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_a3",
            evidence_filename="a3_evidence.json",
            archive_basename="a3_artifacts.tar.gz",
            spec_success=SPEC_SUCCESS,
            spec_failed=SPEC_FAILED,
            report_names=("a3_evidence.json", "a3_diagnostics.json",
                          "a3_arm_identities.json", "a3_replay.json",
                          "a3_probe_inventory.json",
                          "a3_attested_evaluation_protocol.json"),
            event_streams=probe_streams),
        teardown=TeardownPolicy(
            note=("nothing chains off A3. The comparison runs off pod at $0 "
                  "and the FFN experiment and D-series are not started.")),
        setup_failure_files=("/workspace/pytest_outcomes.json",
                             "/workspace/pytest_junit.xml",
                             "/workspace/pytest.log"),
        precheck=(
            session_commit_gate(REPO_ROOT,
                                auth_path_for(getattr(args, "run_id", None)),
                                check_lineage=True),
            grant_provenance_gate,
            a3_harness_gate,
            a3_design_gate,
            pricing_identity_gate,
            teacher_binding_gate,
            battery_staged_gate,
            controls_evidence_gate,
            artifact_spec_gate,
            local_files_gate(REPO_ROOT,
                             ("scripts/pod/autoinit_a3_driver.py",
                              "scripts/autoinit/aggregate_a3.py",
                              "scripts/experiments/phase_c3/a3_session.py"),
                             what="the A3 executable"),
            readiness_gate,
            bundle_staged_gate,
            #: The two that talk to a network service, last, so a cheap local
            #: refusal still costs nothing. The billing check is the final one
            #: before any create call.
            durable_capacity_gate,
            no_billing_resource_gate,
        ),
        evidence_fields={
            "a3_session_contract_hash": A3S.A3_SESSION_CONTRACT.contract_hash,
            "a3_design_sha256": A3S.design_hash(),
            "runs_a_search": False,
            "eliminates_arms": False,
            "retrains_controls": False,
            "computes_a_decision_on_pod": False,
            #: DERIVED from the frozen design, never transcribed. A session
            #: record that stated two arms for a run that trained nine across
            #: three is a record contradicting its own measurement.
            "arms": A3S.A3_SESSION_CONTRACT.n_arms,
            "seeds": len(A3S.recovery_seeds()),
            "probes": A3S.A3_SESSION_CONTRACT.n_probes,
            "diagnostic_rounds": A3S.DIAGNOSTIC_ROUNDS,
            "expected_parent_digest": A3S.expected_parent_digest(),
            "expected_incumbent_digest": A3S.expected_incumbent_digest(),
            "measured_protocol": A3S.TREATMENT_PROTOCOL,
            "gated_protocol": A3S.REFERENCE_PROTOCOL,
            "followon_started": False,
            "followon_reachable_from_this_launcher": False},
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, required=True)
    ap.add_argument("--disk-gb", type=int, default=None,
                    help="omit to use the DERIVED provision")
    ap.add_argument("--uv-max-s", type=int, default=2700)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--max-draws", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="run every $0 gate and create nothing")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.disk_gb is None:
        args.disk_gb = int(
            load_live_pricing(REPO_ROOT)["price"]["container_disk_gb"])
    if args.out is None:
        args.out = (f"{rel_run_dir(RUN_EXPERIMENT_ID, args.run_id, RUN_STAGE_ID)}"
                    "/runtime/session.json")
    return run_session(
        spec(args), args, REPO_ROOT,
        summary=("STOP for review. A3 ran one experiment end to end and "
                 "computed no decision on the pod; the comparison is "
                 "scripts/autoinit/aggregate_a3.py, off pod, at $0."))


if __name__ == "__main__":
    raise SystemExit(main())
