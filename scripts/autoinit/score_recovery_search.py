"""Score stored generations against the frozen `recovery_search_v1` battery.

    PYTHONPATH=src python scripts/autoinit/score_recovery_search.py \
        --generations artifacts/eval/preflight/<label> --label <label> \
        --seed 20260726 --out artifacts/audit/<label>_recovery_search.json

CPU only, and deliberately separate from generation: generation is the paid part
and scoring is free and re-runnable, which is what let the Experiment 1 GSM8K
evaluator be corrected after the fact without re-running a checkpoint.

**Why this exists rather than `scripts/evaluation/score_battery.py`.** That script
requires `manifest["battery_version"] == capability-v2` and iterates *its* set
list. `recovery_search_v1` is a different asset with a different manifest schema
(`battery_id` / `version`), a different set list (it adds `gsm8k`, `code` and
`tool` and drops the paired refusal sets), and a different metric contract —
`usable_rollout_rate` over ALL prompts, `correct_overall` over SCORABLE prompts
only, never combined into a weighted scalar. Running it through the promotion
scorer would raise on the first line; quietly adding `battery_version` to a frozen
manifest to make it pass would change the asset's hash to satisfy a consumer.

The **scoring rules are not reimplemented**: every verdict comes from
`aadistill.evaluation.capability` and `aadistill.evaluation.behavior`, the same
frozen deterministic functions the promotion battery uses, and their source hash
is recorded in the output. What this script adds is the recovery-search contract:

* `usable_rollout` from `aadistill/evaluation/usable_rollout.py`, all five
  components reported alongside the conjunction;
* `score_recovery_row()`, so `correct => usable` holds **by construction** — a
  rollout that answers correctly and then loops forever cannot produce a
  trajectory for Stage 5 and is not counted correct;
* `code` scored for behaviour only (20 prompts), never for correctness, so
  `correct_overall` is over the 170 scorable prompts and says so;
* the `CAPABILITY_SCHEMA_V1` breakdown, validated fail-closed — a scoring bug
  that drops a capability must raise, not read as a clean bill of health.

Counts, not rates, are what leave this script for the pooled seed aggregation:
`pooled_counts@v1` refuses a float outright.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/autoinit"))

from audit_tool_scoring import as_openai_calls, as_openai_tools  # noqa: E402

from aadistill.autoinit.recovery import (  # noqa: E402
    CAPABILITY_SCHEMA_V1,
    recovery_scoring_contract,
    score_recovery_row,
    validate_scored_rows,
)
from aadistill.evaluation import usable_rollout  # noqa: E402
from aadistill.evaluation.behavior import score_tool_call, split_generation  # noqa: E402
from aadistill.evaluation.capability import SCORERS  # noqa: E402
from aadistill.evaluation.strict_answer import score_numeric  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

BATTERY = "artifacts/stage3/recovery_search_v1"
#: Frozen pins. The battery this scores is an immutable asset; if either moves,
#: the result is not comparable to anything else scored under this name.
#:
#: **Two conventions are in play in this repository and they are not
#: interchangeable.** `manifest_sha256` is `sha256_json` over the manifest with
#: that field removed — self-referential, so it can live inside the file it
#: describes — while `sha256_file` hashes the raw bytes. Comparing across the two
#: produces a mismatch that looks like corruption and is not; it has already cost
#: this project a false alarm once. This pins the canonicalized value, verifies it
#: self-consistently, and records the file hash separately as provenance.
BATTERY_MANIFEST_SHA256 = "72d8c0535e7752faf704d9075b7835a47610fd3cd26866cf5be7d48eb7b40ad1"
BATTERY_CONTENT_SHA256 = "a1b22778b00d95b6aba358c14a5af5b559fd807bb371c92131eacca59479f323"


#: The tool-call diagnostics that must hold for a generation to be **structurally
#: executable** by an agent runtime. This is the `tool` capability's extra usable
#: condition, on top of the generic five-component rollout check.
#:
#: `tools_offered=True` only says the tool-call *envelope* is allowed here — it
#: deliberately does not inspect the call. A reply containing `<tool_call>` and
#: then unparseable bytes satisfies generic protocol validity and is not something
#: any runtime can execute, so counting it as a usable rollout would say Stage 5
#: could collect a trajectory from it. It could not.
#:
#: Deliberately **excluded**:
#:
#: * `tool_args_schema_ok` — the xLAM `required` list is reconstructed from
#:   missing defaults, which is an interpretive step; the audit
#:   (logs/autoinit_tool_scoring_audit.json) showed no verdict depends on that
#:   interpretation, and the field stays diagnostic rather than becoming a gate;
#: * `tool_call_exact_match` — that is *correctness*, and folding it into
#:   usability would collapse the two axes this battery exists to keep apart.
#:
#: So a well-formed call on a declared tool with the wrong argument values is
#: **usable and incorrect** — the model can drive the runtime, it just chose
#: wrongly, which is exactly the distinction between stability and capability.
#:
#: Multi-call samples take the frozen scorer's own semantics: `parsed` and
#: `tool_name_valid` are `all(...)` over the emitted calls, so one bad call in a
#: pair makes the sample unusable. No second interpretation is invented here.
TOOL_STRUCTURAL_GATE = ("tool_call_emitted", "tool_call_parsed", "tool_name_valid")


def scorer_correct(set_name: str, record: dict, sample: dict) -> tuple[bool, dict]:
    """One verdict from the frozen scorers. Never a rule written here."""
    if set_name == "gsm8k":
        verdict = score_numeric(record, sample["gsm8k_answer"])
        return bool(verdict["correct"]), verdict
    if set_name == "tool":
        # The battery stores tools/reference_calls in the corpus envelope; the
        # scorer takes the OpenAI envelope. The translation is mechanical and was
        # audited over 20 items x 6 adversarial cases
        # (logs/autoinit_tool_scoring_audit.json): no verdict depends on how the
        # `required` list is interpreted.
        tools = sample["tools"]
        gold = sample["reference_calls"]
        if isinstance(tools, str):
            tools = json.loads(tools)
        if isinstance(gold, str):
            gold = json.loads(gold)
        think_preopened = bool(record.get("think_preopened", True))
        answer = split_generation(record.get("raw") or "",
                                  think_preopened=think_preopened)["answer"]
        verdict = score_tool_call(answer, as_openai_tools(tools, True),
                                  as_openai_calls(gold))
        # Correctness is exact match, as the battery manifest declares. The
        # weaker rungs (emitted / parsed / name valid / args schema ok) are kept
        # as diagnostics so a partial failure is visible rather than inferred.
        return bool(verdict["tool_call_exact_match"]), verdict
    if set_name in SCORERS:
        verdict = SCORERS[set_name](record, sample)
        return bool(verdict["correct"]), verdict
    raise SystemExit(f"no frozen scorer for set {set_name!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True, type=Path,
                    help="uncapped_eval.py --out-dir for this checkpoint")
    ap.add_argument("--battery", default=BATTERY, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="the TRAINING seed of the scored checkpoint; pooling "
                         "refuses duplicates, so this must be the real one")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-sample", type=Path, default=None)
    ap.add_argument("--allow-missing-sets", action="store_true",
                    help="score what is present; only for local dry runs")
    args = ap.parse_args()

    battery = (REPO_ROOT / args.battery) if not args.battery.is_absolute() \
        else args.battery
    manifest = json.loads((battery / "manifest.json").read_text())
    manifest_sha = sha256_json(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if manifest_sha != manifest.get("manifest_sha256"):
        raise SystemExit(
            f"the battery manifest does not match its own manifest_sha256 "
            f"({manifest_sha} vs {manifest.get('manifest_sha256')}); it has been "
            "edited since it was frozen")
    if manifest_sha != BATTERY_MANIFEST_SHA256:
        raise SystemExit(
            f"battery manifest is {manifest_sha} but this scorer pins "
            f"{BATTERY_MANIFEST_SHA256}. Two runs scored against different "
            "batteries are not comparable. (This is the canonicalized "
            "manifest_sha256 convention, not sha256_file of the raw bytes.)")
    if manifest.get("content_sha256") != BATTERY_CONTENT_SHA256:
        raise SystemExit("battery content hash moved; the asset is not frozen")

    gen_dir = (REPO_ROOT / args.generations) if not args.generations.is_absolute() \
        else args.generations
    sets = manifest["sets"]
    scorable_sets = set(manifest["scorable_sets"])
    behaviour_only = set(manifest["behaviour_only_sets"])

    rows, per_sample, per_set = [], [], {}
    missing_sets = []
    for name, spec in sets.items():
        gen_path = gen_dir / f"{name}.generations.jsonl"
        if not gen_path.is_file():
            missing_sets.append(name)
            continue
        samples = {json.loads(line)["id"]: json.loads(line)
                   for line in (battery / f"{name}.jsonl").open() if line.strip()}
        scorable = name in scorable_sets
        if scorable == (name in behaviour_only):
            raise SystemExit(f"{name}: the manifest lists it as both or neither "
                             "scorable and behaviour-only")
        records = [json.loads(line) for line in gen_path.open() if line.strip()]
        seen = set()
        set_rows = []
        for record in records:
            sample = samples.get(record["id"])
            if sample is None:
                raise SystemExit(
                    f"{name}: generation {record['id']!r} is not in the frozen "
                    "set — the battery and the generations disagree")
            if record["id"] in seen:
                raise SystemExit(f"{name}: duplicate generation {record['id']!r}")
            seen.add(record["id"])
            think_preopened = bool(record.get("think_preopened", True))
            # Whether the prompt declared tools is a property of the frozen
            # sample, read here rather than guessed from the generation. Without
            # it a correct, well-terminated tool call is `unexpected_tool_call`
            # -> not protocol_valid -> not usable -> not correct, and the tool
            # capability is structurally zero for every arm.
            tools_offered = bool(sample.get("tools"))
            components = usable_rollout.components(
                record, think_preopened=think_preopened,
                tools_offered=tools_offered)
            is_usable = all(components.values())
            correct, verdict = ((False, {"scorable": False}) if not scorable
                                else scorer_correct(name, record, sample))
            # The tool capability's extra usable condition: structurally
            # executable, not merely "a tool call was allowed here". Applied in
            # this scoring path only — `protocol_valid` stays generic.
            structural = None
            if name == "tool":
                structural = {k: bool(verdict.get(k)) for k in TOOL_STRUCTURAL_GATE}
                is_usable = is_usable and all(structural.values())
            row = score_recovery_row(usable=is_usable, scorer_correct=correct,
                                     scorable=scorable)
            row.update({"set": name, "id": record["id"],
                        "domain": spec["domain"], **components})
            if structural is not None:
                row["tool_structurally_executable"] = all(structural.values())
                row.update(structural)
            set_rows.append(row)
            rows.append(row)
            if args.per_sample is not None:
                per_sample.append({"label": args.label, "seed": args.seed,
                                   **row, "verdict": verdict})
        absent = set(samples) - seen
        if absent and not args.allow_missing_sets:
            raise SystemExit(
                f"{name}: {len(absent)} of {len(samples)} prompts have no "
                f"generation, e.g. {sorted(absent)[:3]}. A silently short set "
                "would inflate every rate computed over it.")
        per_set[name] = summarize(set_rows, scorable=scorable)
        per_set[name]["missing_generations"] = len(absent)
        if len(samples) != spec["n"]:
            raise SystemExit(f"{name}: frozen set has {len(samples)} items, "
                             f"manifest says {spec['n']}")

    if missing_sets and not args.allow_missing_sets:
        raise SystemExit(
            f"no generations for {sorted(missing_sets)}; refusing to report a "
            "rate over a subset of the battery as if it were the battery")

    contract = validate_scored_rows(rows)
    result = {
        "schema": "aadistill.autoinit.recovery_search_result/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "label": args.label,
        "seed": args.seed,
        "battery": {"artifact": manifest["artifact"], "role": manifest["role"],
                    "version": manifest["version"],
                    "manifest_sha256": manifest_sha,
                    "manifest_sha256_convention": (
                        "sha256_json over the manifest with manifest_sha256 "
                        "removed; NOT sha256_file of the raw bytes"),
                    "manifest_file_sha256": sha256_file(battery / "manifest.json"),
                    "content_sha256": manifest["content_sha256"],
                    "n_prompts": manifest["n_prompts"],
                    "n_scorable_prompts": manifest["n_scorable_prompts"]},
        # One aggregate contract digest, which is what a control and a candidate
        # bind to. Per-file hashes are inside it as provenance; a single scorer
        # file's hash is NOT the metric identity — that is precisely how the
        # composition defect stayed invisible under v1.
        "scoring_contract": recovery_scoring_contract(REPO_ROOT),
        "tool_usable_gate": {
            "definition": ("generic usable_rollout AND " +
                           " AND ".join(TOOL_STRUCTURAL_GATE)),
            "excluded": {
                "tool_args_schema_ok": ("the xLAM `required` reconstruction is "
                                        "interpretive; kept diagnostic"),
                "tool_call_exact_match": ("that is correctness; folding it in "
                                          "would collapse the two axes"),
            },
            "multi_call": ("the frozen scorer's own all-calls semantics; no "
                           "second interpretation is defined here"),
        },
        "missing_sets": sorted(missing_sets),
        # Per-row enforcement counts from validate_scored_rows: distinct from
        # `scoring_contract`, which is the identity of the code that produced them.
        "row_contract": contract,
        **summarize(rows, scorable=None),
        "per_set": per_set,
        "per_domain": group(rows, "domain"),
        "per_capability": {name: per_set[name] for name in
                           CAPABILITY_SCHEMA_V1.expected if name in per_set},
        "metric_contract": manifest["metrics"],
        "no_weighted_scalar": (
            "usable_rollout_rate and correct_overall are reported separately and "
            "are never combined; usable_rollout is blind to correctness by "
            "construction, which is why correctness is a separate axis"),
    }
    # Fail closed: a dropped capability must raise here rather than reading as a
    # pass when the catastrophic rule later cannot see it.
    if not missing_sets:
        CAPABILITY_SCHEMA_V1.validate(result, label=args.label)
        result["capability_schema_enforced"] = True
    else:
        result["capability_schema_enforced"] = False
    result["result_sha256"] = sha256_json(result)

    out = (REPO_ROOT / args.out) if not args.out.is_absolute() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if args.per_sample is not None:
        ps = (REPO_ROOT / args.per_sample) if not args.per_sample.is_absolute() \
            else args.per_sample
        ps.parent.mkdir(parents=True, exist_ok=True)
        with ps.open("w") as f:
            for row in per_sample:
                f.write(json.dumps(row) + "\n")

    print(json.dumps({
        "label": args.label, "seed": args.seed,
        "n": result["n"], "usable": result["usable"], "correct": result["correct"],
        "usable_rollout_rate": result["usable_rollout_rate"],
        "correct_overall": result["correct_overall"],
        "correct_given_usable": result["correct_given_usable"],
        "n_scorable": result["n_scorable"],
        "per_capability_usable": {k: v["usable_rollout_rate"]
                                  for k, v in result["per_capability"].items()},
    }, indent=2))


def summarize(rows: list[dict], *, scorable: bool | None) -> dict:
    """Counts first, rates derived. Pooling consumes the counts, never the rates."""
    n = len(rows)
    usable = sum(r["usable"] for r in rows)
    scorable_rows = [r for r in rows if r["scorable"]]
    n_scorable = len(scorable_rows)
    correct = sum(r["correct"] for r in scorable_rows)
    usable_scorable = sum(r["usable"] for r in scorable_rows)
    out = {
        "n": n, "usable": usable, "correct": correct,
        "n_scorable": n_scorable,
        # `usable_rollout_rate` is over ALL prompts (behaviour is measurable
        # everywhere); `correct_overall` is over SCORABLE prompts only, because
        # the 20 `code` prompts have no correctness oracle and counting them as
        # wrong would depress every candidate identically and mean nothing.
        "usable_rollout_rate": round(usable / n, 4) if n else None,
        "correct_overall": round(correct / n_scorable, 4) if n_scorable else None,
        "correct_given_usable": (round(correct / usable_scorable, 4)
                                 if usable_scorable else None),
        "scorable": scorable,
    }
    for component in usable_rollout.COMPONENTS:
        out[component] = round(sum(r[component] for r in rows) / n, 4) if n else None
    census = {}
    for row in rows:
        if row["usable"]:
            continue
        # A tool row can fail only the structural gate, with all five generic
        # components passing. Attribute it rather than letting `next()` raise.
        first = next((c for c in usable_rollout.COMPONENTS if not row[c]),
                     "tool_not_structurally_executable")
        census[first] = census.get(first, 0) + 1
    out["first_failure"] = dict(sorted(census.items(), key=lambda kv: -kv[1]))
    gated = [r for r in rows if "tool_structurally_executable" in r]
    if gated:
        out["tool_structurally_executable"] = round(
            sum(r["tool_structurally_executable"] for r in gated) / len(gated), 4)
        for field in TOOL_STRUCTURAL_GATE:
            out[field] = round(sum(r[field] for r in gated) / len(gated), 4)
    return out


def group(rows: list[dict], key: str) -> dict:
    out = {}
    for value in sorted({r[key] for r in rows}):
        out[value] = summarize([r for r in rows if r[key] == value], scorable=None)
    return out


if __name__ == "__main__":
    main()
