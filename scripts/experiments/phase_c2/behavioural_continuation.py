"""Same-campaign continuation across a REPLACEMENT resource. Decides nothing.

A campaign is one 12-probe behavioural experiment. A run attempt is one
launcher invocation and the provider resource it draws. When a resource is
replaced, the replacement gets a **fresh filesystem**: the previous pod's
`audit/probes/*.json` is gone and so is every absolute `model_dir` it recorded.
The only thing that survived is the durable destination on the launcher host.

So this module answers three questions, all on the launcher host, all at `$0`:

1. **What has this campaign actually verified?** Read from the durable
   destination and nowhere else. A pod's own journal is not evidence about a
   campaign, because it dies with the pod.
2. **What does the campaign still owe?** Mechanically, from that state: which
   probes remain, which arms their remaining probes need rebuilt, and how long
   restoring the verified ones costs.
3. **What travels to the replacement pod?** A small manifest plus the verified
   bytes and evidence it names.

What it does NOT do is decide whether a continuation may run. That is
`campaign_continuation_gate`'s, and the answer depends on money and on whether
the previous resource is provider-confirmed released — neither of which is a
property of the probes.

**The bytes are not ceremony.** A restored probe that was trained but not
validly scored resumes AT SCORING under the preregistered policy, and scoring
reads the weights. That is why a manifest of identities would not be enough and
the checkpoint itself has to land on the replacement pod, at a new path, and be
re-identified there.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

SCHEMA = "aadistill.autoinit.c2_behavioural_continuation_manifest/v1"

#: The file the launcher's destination-side verification writes beside a
#: probe's bytes, and the ONLY admission token for reuse. Written only when
#: `verify_transferred_leaf` matched every identity field of the arrival.
ACK_NAME = "durable_ack.json"

#: The probe's science evidence, secured beside its bytes during the run. The
#: names are the DESTINATION's, not the pod's: the pod names these per probe
#: (`<probe_id>_result.json`) and a destination directory is already per probe,
#: so a second copy of the probe id in the filename is one more thing to keep
#: consistent.
RECORD_NAME = "probe_record.json"
RESULT_NAME = "result.json"
PER_SAMPLE_NAME = "per_sample.jsonl"

#: The campaign's screening commitment, at the campaign root rather than under
#: an attempt: the commitment belongs to the experiment, and a later attempt
#: must honour the one its campaign made.
RANKING_NAME = "c2_screening_ranking.json"

#: MB/s used to BOUND the restore of verified probes to a replacement pod.
#:
#: The project's recorded dev-box uplink figure is 0.72 MB/s, and observations
#: span 0.23-0.79 MB/s — the 0.23 came from a real 3.7 MB bundle upload. A
#: bound takes the SLOWEST recorded observation, because an average is not a
#: bound and this number decides whether a continuation is affordable: pricing
#: the restore at the mean and then running at the floor is how continuation
#: attempt 2 died mid-transfer, "arithmetic rather than luck".
#:
#: The honest consequence is stated rather than softened: at this rate one
#: 1.11 GiB probe is ~80 minutes of BILLED pod time, so a continuation late in
#: a campaign will not fit a ceiling sized for one session and the gate will
#: refuse it. A maintainer freeing Hugging Face private storage would move this
#: transfer to the `$0` pre-pod relay and remove the minutes entirely; that is
#: a maintainer decision, never an autonomous repair.
RESTORE_MB_PER_SECOND = 0.23
RESTORE_RATE_BASIS = (
    "the slowest recorded dev-box uplink observation (0.23 MB/s, from a real "
    "3.7 MB bundle upload; the project's recorded figure is 0.72 MB/s and "
    "observations span 0.23-0.79). A bound takes the slowest, not the mean. "
    "logs/shared/analyses/autoinit_relay_capacity.md")


class ContinuationError(RuntimeError):
    """The campaign's state cannot be read, or cannot be continued from."""


# ---------------------------------------------------------------------------
# what the campaign has verified, read from the durable destination
# ---------------------------------------------------------------------------

def _probe_dirs(campaign_root: Path, *, exclude_attempt: str = "") -> list[Path]:
    if not campaign_root.is_dir():
        return []
    out: list[Path] = []
    for attempt in sorted(p for p in campaign_root.iterdir() if p.is_dir()):
        if attempt.name == exclude_attempt:
            continue
        out.extend(sorted(p for p in attempt.iterdir() if p.is_dir()))
    return out


def _shard_bytes(directory: Path) -> int:
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def campaign_state(campaign_root: str | Path, *,
                   exclude_attempt: str = "") -> dict[str, Any]:
    """What this campaign has verified off-pod, and what it committed.

    Every admitted probe satisfies all of:

    * it lives under this CAMPAIGN's durable root — a different campaign is a
      different experiment and is not visible from here at all;
    * its `durable_ack.json` records a destination-side re-identification that
      MATCHED. The ack is written only when every identity field of the arrival
      matched what the driver announced, so an unverifiable or mismatched
      arrival is preserved without becoming reusable;
    * it carries the identity a replacement pod will re-check the bytes
      against.

    A probe with bytes but no score is admitted and flagged `scored: False`. It
    is not a completed measurement — under the preregistered policy it resumes
    at scoring — and it is emphatically not retrained.
    """
    root = Path(campaign_root)
    probes: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for d in _probe_dirs(root, exclude_attempt=exclude_attempt):
        unit = d.name
        ack_path = d / ACK_NAME
        if not ack_path.is_file():
            rejected.append({"probe_id": unit, "path": str(d), "why": (
                "no durable_ack.json: the destination never re-identified these "
                "bytes, so they are preserved and not reusable")})
            continue
        try:
            ack = json.loads(ack_path.read_text())
        except json.JSONDecodeError as exc:
            rejected.append({"probe_id": unit, "path": str(d),
                             "why": f"unreadable ack: {exc}"})
            continue
        if not ack.get("re_identified_from_delivered_bytes"):
            rejected.append({"probe_id": unit, "path": str(d), "why": (
                "the ack does not claim destination-side re-identification")})
            continue
        identity = ack.get("identity") or {}
        missing = [f for f in ("artifact_digest", "arch_signature",
                               "num_parameters") if not identity.get(f)]
        if missing:
            rejected.append({"probe_id": unit, "path": str(d), "why": (
                f"the ack's identity is missing {missing}, so the bytes cannot "
                "be re-checked on a replacement pod")})
            continue

        record_path, result_path = d / RECORD_NAME, d / RESULT_NAME
        per_sample = d / PER_SAMPLE_NAME
        record = None
        if record_path.is_file():
            try:
                record = json.loads(record_path.read_text())
            except json.JSONDecodeError:
                record = None
        score = (record or {}).get("score") or None
        #: A score whose evidence did not survive is not a usable score: the
        #: verdict reads the per-sample rows, not the summary.
        if score and not (result_path.is_file() and per_sample.is_file()):
            score = None

        probes[unit] = {
            "probe_id": unit,
            "attempt": d.parent.name,
            "durable_path": str(d),
            "identity": identity,
            "bytes": _shard_bytes(d),
            "scored": bool(score),
            "record": record,
            "score": score,
            "has_result": result_path.is_file(),
            "has_per_sample": per_sample.is_file(),
        }

    ranking = None
    ranking_path = root / RANKING_NAME
    if ranking_path.is_file():
        ranking = json.loads(ranking_path.read_text())
        if not (ranking.get("advanced") or {}).get("state_id"):
            raise ContinuationError(
                f"{ranking_path} exists but names no advanced candidate. A "
                "ranking record that cannot say what advanced is not a "
                "commitment, and a continuation cannot be bound by it.")

    return {
        "campaign_root": str(root),
        "probes": probes,
        "rejected": rejected,
        "committed_ranking": ranking,
        "committed_candidate": ((ranking or {}).get("advanced") or {}).get(
            "state_id"),
        "_source_is_the_destination": (
            "read from the durable destination on the launcher host, never from "
            "a pod's own journal. A pod's journal dies with the pod, and a "
            "replacement resource has a fresh filesystem."),
    }


# ---------------------------------------------------------------------------
# what the campaign still owes
# ---------------------------------------------------------------------------

def _expected_probe_ids(repo_root: str | Path,
                        committed_candidate: str | None) -> dict[str, str]:
    """`probe_id -> arm` for every probe the frozen protocol expects.

    Built from the protocol's own seeds and the frozen Top-5 rather than from a
    pattern typed here, and it is the same `probe_id` spelling the schedule
    produces because it comes FROM the schedule.

    Before screening commits, the confirmation rung's candidate arm is not yet
    known and is named `<unranked>`; the caller resolves it against the most
    expensive admissible candidate when it needs a materialization bound.
    """
    from experiments.phase_c2 import behavioural as BH
    from experiments.phase_c2 import behavioural_governance as BG
    from experiments.phase_c2 import behavioural_schedule as SCH

    proto = BH.protocol(repo_root)["behavioural_selection"]
    #: From `candidate_leaves`, which resolves the five candidates out of the
    #: FROZEN RECORD in the repository — not from `candidate_manifest`, which
    #: joins them to the dev box's durable products at
    #: `/home/ecs-user/aad-artifacts/...`. Only the state ids are needed to
    #: derive probe ids, and reading them from the repo keeps this module usable
    #: wherever the repository is, instead of only on the launcher host.
    candidates = [{"state_id": leaf.state_id, "artifact_digest": "",
                   "durable_path": ""}
                  for leaf in BG.candidate_leaves(repo_root)]
    anchor = {"state_id": SCH.ANCHOR, "artifact_digest": "", "durable_path": ""}
    out: dict[str, str] = {}
    for probe in SCH.screening_probes(candidates, anchor,
                                      proto["seeds"]["screening"]):
        out[probe.probe_id] = probe.arm
    advanced = {"state_id": committed_candidate or "<unranked>",
                "artifact_digest": "", "durable_path": ""}
    for probe in SCH.confirmation_probes(advanced, anchor,
                                         proto["seeds"]["confirmation"]):
        out[probe.probe_id] = probe.arm
    return out


def arm_minutes(repo_root: str | Path = REPO_ROOT) -> dict[str, float]:
    """Bounded build minutes PER ARM, from the owners of each bound.

    The candidates' bounds come from the replay's leaves and B's from its own
    deriver, so this is a join rather than a third derivation. The sum over all
    six is `materialization_minutes()['total_minutes']`, which the full-session
    ceiling already uses.
    """
    from experiments.phase_c2 import behavioural as BH
    from experiments.phase_c2 import behavioural_governance as BG
    from experiments.phase_c2 import behavioural_schedule as SCH

    out = {leaf.state_id: float(leaf.bounded_minutes)
           for leaf in BG.candidate_leaves(repo_root)}
    out[SCH.ANCHOR] = float(BH.b_preparation_minutes(repo_root)["bounded_minutes"])
    return out


def restore_minutes(total_bytes: int,
                    rate_mb_per_second: float = RESTORE_MB_PER_SECOND) -> float:
    """Billed minutes to push `total_bytes` to a replacement pod. A BOUND."""
    if rate_mb_per_second <= 0:
        raise ContinuationError("a restore rate must be positive")
    return round(total_bytes / (rate_mb_per_second * 1e6) / 60.0, 2)


def remaining_work(repo_root: str | Path = REPO_ROOT, *,
                   state: dict[str, Any]) -> dict[str, Any]:
    """What a replacement resource still owes, derived mechanically.

    The rules are the frozen protocol's, applied to what the destination
    actually holds:

    * a probe verified off-pod is NOT owed — a completed probe is never
      retrained, for any outcome;
    * a probe verified but not validly scored IS owed its scoring, not its
      training. It costs the battery, not the trainer;
    * the arms of the probes that remain ARE owed, because a replacement
      resource has a fresh filesystem. That is a runtime necessity of
      replacement, not completed science charged twice;
    * once screening has committed a candidate, confirmation is owed for THAT
      candidate and screening is not owed at all;
    * before screening commits, the confirmation candidate is unknown, so the
      materialization bound takes the most expensive admissible candidate —
      bounding by an average is how a ceiling comes to be exceeded by the arm
      that was actually chosen.

    Returns `n_probes_remaining == 0` when the campaign owes nothing. That is
    not permission to do anything: a complete verdict is terminal, and NO_GO and
    INCONCLUSIVE are complete results.
    """
    from experiments.phase_c2 import behavioural as BH
    from experiments.phase_c2 import behavioural_schedule as SCH

    committed = state.get("committed_candidate")
    expected = _expected_probe_ids(repo_root, committed)
    held = state.get("probes") or {}

    complete = sorted(pid for pid in expected
                      if held.get(pid, {}).get("scored"))
    #: Trained and verified but not validly scored: owed its scoring only.
    unscored = sorted(pid for pid in expected
                      if pid in held and not held[pid].get("scored"))
    untrained = sorted(pid for pid in expected if pid not in held)
    remaining = sorted(set(unscored) | set(untrained))

    per_arm = arm_minutes(repo_root)
    arms_needed = sorted({expected[pid] for pid in remaining})
    #: The unknown confirmation candidate, bounded by the dearest admissible
    #: one rather than by a mean.
    if "<unranked>" in arms_needed:
        arms_needed.remove("<unranked>")
        dearest = max((a for a in per_arm if a != SCH.ANCHOR),
                      key=lambda a: per_arm[a])
        if dearest not in arms_needed:
            arms_needed.append(dearest)
        arms_needed.sort()
    unknown_arms = [a for a in arms_needed if a not in per_arm]
    if unknown_arms:
        raise ContinuationError(
            f"no bounded build minutes for arm(s) {unknown_arms}; a "
            "materialization cannot be priced from a guess")
    materialization = round(sum(per_arm[a] for a in arms_needed), 2)

    restorable = sorted(pid for pid in expected if pid in held)
    restore_bytes = sum(int(held[pid]["bytes"]) for pid in restorable)
    minutes = restore_minutes(restore_bytes)

    decomposition = BH.session_decomposition(
        repo_root, materialization_minutes=materialization,
        probes_remaining=len(remaining), restore_minutes=minutes)
    return {
        "probes_expected": len(expected),
        "probes_complete": complete,
        "probes_trained_not_scored": unscored,
        "probes_untrained": untrained,
        "probes_remaining": remaining,
        "n_probes_remaining": len(remaining),
        "committed_candidate": committed,
        "screening_committed": bool(committed),
        "arms_needed": arms_needed,
        "materialization_minutes": materialization,
        "restore": {
            "probes": restorable,
            "n": len(restorable),
            "bytes": restore_bytes,
            "gib": round(restore_bytes / 2**30, 3),
            "minutes": minutes,
            "rate_mb_per_second": RESTORE_MB_PER_SECOND,
            "_basis": RESTORE_RATE_BASIS,
        },
        "decomposition": decomposition,
        "_a_completed_probe_is_never_retrained": (
            "probes_complete is excluded from the remaining probe count and "
            "from the arms priced. The protocol forbids retraining a finished "
            "probe for a different outcome, so pricing one again would reserve "
            "money for work that may not be done."),
        "_zero_remaining_is_not_permission": (
            "a campaign that owes nothing has a verdict or has nothing left to "
            "run. GO, NO_GO and INCONCLUSIVE are all terminal, and none of them "
            "is a reason to start another attempt."),
    }


# ---------------------------------------------------------------------------
# what travels
# ---------------------------------------------------------------------------

def build_manifest(repo_root: str | Path = REPO_ROOT, *,
                   campaign_id: str, run_attempt: str,
                   state: dict[str, Any], work: dict[str, Any],
                   pod_root: str) -> dict[str, Any]:
    """The small document a replacement pod reads to continue this campaign.

    Small on purpose: identities, scores and destinations. The bytes travel
    beside it and the pod re-identifies them from the path the manifest names —
    the old pod's absolute `model_dir` appears NOWHERE in here, because it does
    not exist on a replacement resource and must not be the truth about one.
    """
    entries = []
    for pid in work["restore"]["probes"]:
        held = state["probes"][pid]
        record = held.get("record") or {}
        entries.append({
            "probe_id": pid,
            "source_attempt": held["attempt"],
            "durable_path": held["durable_path"],
            "pod_path": f"{pod_root}/{pid}",
            "identity": held["identity"],
            "bytes": held["bytes"],
            "scored": held["scored"],
            #: The descriptor fields the driver re-checks against the rung's
            #: own descriptor. A continuation may consume; it may not
            #: substitute.
            "rung": record.get("rung"),
            "arm": record.get("arm"),
            "seed": record.get("seed"),
            "initialization_artifact_digest":
                record.get("initialization_artifact_digest"),
            "config_sha256": record.get("config_sha256"),
            "score": held.get("score"),
            "result_sha256": (held.get("score") or {}).get("result_sha256"),
            "per_sample_sha256": (held.get("score") or {}).get(
                "per_sample_sha256"),
        })
    return {
        "schema": SCHEMA,
        "authorizes": "nothing",
        "campaign_id": campaign_id,
        "run_attempt": run_attempt,
        "pod_root": pod_root,
        "committed_ranking": state.get("committed_ranking"),
        "committed_candidate": state.get("committed_candidate"),
        "probes": entries,
        "remaining": {
            "probes": work["probes_remaining"],
            "arms_needed": work["arms_needed"],
        },
        "_what_the_pod_must_do": (
            "restore each probe's bytes to pod_path, re-identify them THERE "
            "against identity, and only then admit the probe to the campaign "
            "journal. An identity that does not reproduce from the delivered "
            "bytes is a refusal, not a warning."),
        "_no_source_model_dir": (
            "no entry carries the producing pod's model_dir. It does not exist "
            "on a replacement resource, and a path that cannot be checked is "
            "not evidence: the driver's truth about where a probe lives is "
            "pod_path, which it re-identifies before admitting anything."),
        "_identity_travels_verbatim": (
            "each identity is the one the producing driver ANNOUNCED, copied "
            "unchanged. Its `path` field records where those bytes were first "
            "hashed and is provenance, not a location on any live machine — "
            "verification reads the digests, the arch signature and the "
            "parameter count, and never that path. It is not stripped, because "
            "re-identifying against an edited identity would verify the bytes "
            "against something other than what was announced."),
    }
