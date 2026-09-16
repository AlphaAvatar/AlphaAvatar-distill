"""Build the frozen Phase-C2 SCREENING battery (role C2_SCREENING).

    PYTHONPATH=src .venv/bin/python scripts/data/build_c2_screening_battery.py \
        --out artifacts/stage3/c2_screening_v1

Its only job is to RANK the Top-K full-search candidates against the incumbent B
on one screening recovery seed, so that one candidate can advance to
confirmation. It produces no verdict, it may not promote anything, and it is
never training data.

**Why a second battery exists at all.** C2's behavioural stage selects one of K
candidates and then tests it. Disjoint recovery *seeds* are not enough to make
that confirmation interpretable, because C0's inferential unit is the **prompt**
and C0 measured substantial same-prompt cross-seed dependence — ICC `0.25 ±
0.095`, and `P(correct on a second seed | correct on the first) = 0.257` against
a `0.022` marginal, an `11.7x` lift. Screening and confirming on the same prompts
would let the selection leak into the confirmation through that dependence. So
the screening rung gets its own prompts as well as its own seed, and the
confirmation evidence is genuinely held out.

**Everything else is C1's, deliberately.** The mixture, the sources and their
pinned revisions, the renderers, the id conventions and the exclusion contract
are imported from `build_c1_confirmation_battery` rather than restated, so the
two batteries differ in their *sample* and nothing else. `correct_overall` means
the same thing on both.

Three properties are load-bearing.

**Disjoint from `c1_confirmation_v1`, by id AND normalized prompt content.** That
asset is added to C1's own five exclusion roles, so the screening prompts cannot
overlap the confirmation prompts in either identity or text.

**An independent ordering, not the next ranks after C1's.** The rank key uses a
distinct domain, so this is a fresh cryptographic sample of the eligible pool
rather than C1's ranks 151-300. The key still derives only from a digest frozen
before any C2 candidate existed, the stratum and the example's stable id.

**No outcome touches it.** No model output of any kind — not a Search-1 ranking,
not a state_eval value, not a C1 probe result — is consulted anywhere in the
selection.
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
sys.path.insert(0, str(REPO_ROOT / "scripts/data"))

from aadistill.data.extra_stream import content_sha256  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from battery_render import (  # noqa: E402
    RENDERERS, norm, rank_take, read_rows, source_digest,
)
#: The mixture, the pinned sources and the five-role exclusion contract are C1's.
#: Imported, never restated: a second copy is how two batteries come to mean
#: different things by `correct_overall`.
from build_c1_confirmation_battery import (  # noqa: E402
    C0_DIGEST, SETS, SOURCES, excluded_identities,
)

BATTERY_ID = "c2_screening"
BATTERY_VERSION = 1
ROLE = "C2_SCREENING"

#: The rank-key domain. Distinct from C1's `phase-c1-battery`, so the sample is
#: independent rather than the continuation of C1's ordering.
RANK_DOMAIN = "phase-c2-screening-battery"

#: The asset this battery must be disjoint from, beyond C1's five roles. It is
#: read for exclusion ONLY and is not consumed.
C1_CONFIRMATION = "artifacts/stage3/c1_confirmation_v1"


def exclude_c1_confirmation(rel: str, source_ids: set[str],
                            prompt_hashes: set[str]) -> dict:
    """Add the C1 confirmation battery to the exclusion sets."""
    directory = REPO_ROOT / rel
    if not directory.is_dir():
        raise SystemExit(
            f"{rel} is missing, so the screening battery cannot be proven "
            "disjoint from the confirmation battery. Refusing to build a "
            "screening asset whose independence is unverifiable.")
    n = 0
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            source_ids.add(str(row["id"]))
            if row.get("source_key"):
                source_ids.add(str(row["source_key"]))
            prompt_hashes.add(content_sha256(norm(row.get("prompt_text", ""))))
            n += 1
    if not n:
        raise SystemExit(f"{rel} holds no prompts; refusing a vacuous exclusion")
    return {"asset": rel, "n_prompts": n,
            "note": ("the C1 confirmation battery, read for exclusion only. "
                     "Screening must not share a prompt with the rung that "
                     "confirms, by id or by normalized content, because C0's "
                     "inferential unit is the prompt and same-prompt cross-seed "
                     "dependence is substantial (ICC 0.25 +/- 0.095).")}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="artifacts/stage3/c2_screening_v1")
    ap.add_argument("--battery", default="artifacts/eval/battery_v2")
    ap.add_argument("--recovery-search",
                    default="artifacts/stage3/recovery_search_v2")
    ap.add_argument("--sessions",
                    default="artifacts/stage3/corpus_v2/sessions.jsonl")
    ap.add_argument("--state-eval", default="artifacts/stage1/state_eval_v1")
    ap.add_argument("--calibration",
                    default="artifacts/stage1/e8_calibration_v1")
    ap.add_argument("--c1-confirmation", default=C1_CONFIRMATION)
    args = ap.parse_args()

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    exclude_ids, exclude_hashes, provenance = excluded_identities(args)
    provenance["c1_confirmation"] = exclude_c1_confirmation(
        args.c1_confirmation, exclude_ids, exclude_hashes)

    built: dict[str, list[dict]] = {}
    sources: dict[str, dict] = {}
    for name, (_domain, want, _scorable) in SETS.items():
        repo, rev, rel = SOURCES[name]
        rows = read_rows(repo, rev, rel)
        if name == "gsm8k":      # index-based ids, as the v1 convention requires
            rows = [dict(r, _index=i) for i, r in enumerate(rows)]
        sources[name] = {**source_digest(repo, rev, rel), "n_rows": len(rows)}
        built[name] = rank_take(
            rows, want, stratum=name, base_digest=C0_DIGEST,
            exclude_ids=exclude_ids, exclude_hashes=exclude_hashes,
            make=RENDERERS[name], domain=RANK_DOMAIN)

    short = {k: (len(v), SETS[k][1]) for k, v in built.items()
             if len(v) < SETS[k][1]}
    if short:
        raise SystemExit(
            f"sets short of their target (got, want): {short}. The mixture is "
            "C1's and is not silently rescaled: a short set is a scientific "
            "finding about pool exhaustion and needs a decision.")

    all_hashes = [i["prompt_sha256"] for v in built.values() for i in v]
    if len(set(all_hashes)) != len(all_hashes):
        raise SystemExit("duplicate prompt content inside the battery")
    leaked = sorted(set(all_hashes) & exclude_hashes)
    if leaked:
        raise SystemExit(f"{len(leaked)} prompts collide with an excluded role")

    outputs = {}
    for name, items in built.items():
        items.sort(key=lambda i: str(i["id"]))
        path = out / f"{name}.jsonl"
        with path.open("w") as fh:
            for item in items:
                fh.write(json.dumps(item, sort_keys=True) + "\n")
        try:
            rel_path = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel_path = str(path)
        outputs[name] = {"path": rel_path, "n": len(items),
                         "sha256": sha256_file(path), "domain": SETS[name][0],
                         "scorable": SETS[name][2]}

    scorable = [n for n in built if SETS[n][2]]
    #: Same convention as every other battery here: sha256 over sorted
    #: `id:prompt_sha256` pairs, so equality proves membership, ordering, ids and
    #: prompts are all fixed.
    pairs = sorted(f"{i['id']}:{i['prompt_sha256']}"
                   for v in built.values() for i in v)
    content = hashlib.sha256("\n".join(pairs).encode()).hexdigest()

    manifest = {
        "artifact": f"{BATTERY_ID}_v{BATTERY_VERSION}",
        "role": ROLE,
        "battery_id": BATTERY_ID, "version": BATTERY_VERSION,
        "purpose": (
            "the Phase-C2 behavioural SCREENING battery: it ranks the Top-K "
            "full-search candidates against the incumbent B on one screening "
            "seed so that one candidate advances. It produces no GO/NO-GO "
            "verdict, may not promote anything, is never a promotion asset and "
            "is never training data."),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "scripts/data/build_c2_screening_battery.py",
        "n_prompts": sum(len(v) for v in built.values()),
        "n_scorable_prompts": sum(len(built[n]) for n in scorable),
        "sets": outputs,
        "scorable_sets": sorted(scorable),
        "behaviour_only_sets": sorted(n for n in built if not SETS[n][2]),
        "domains": sorted({v[0] for v in SETS.values()}),
        "mixture": {k: v[1] for k, v in SETS.items()},
        "mixture_rule": (
            "IDENTICAL to c1_confirmation_v1's, imported from that builder "
            "rather than restated: the frozen historical 3:3:3:3:3:2 over the "
            "scorable sets plus the same relative behaviour-only code "
            "component, scaled to 850/950. Screening and confirmation therefore "
            "measure correct_overall on the same distribution, which is what "
            "makes a screening delta informative about a confirmation delta."),
        "sampling_rule": {
            "order": "ascending cryptographic rank, ties by ascending stable id",
            "rank": (f"SHA256(C0_preregistration_digest + ':{RANK_DOMAIN}:' + "
                     "stratum + ':' + stable_source_id)"),
            "rank_domain": RANK_DOMAIN,
            "_why_a_distinct_domain": (
                "so this is an INDEPENDENT cryptographic sample of the eligible "
                "pool rather than C1's next ranks. Reusing C1's domain would "
                "still be deterministic and disjoint, but the screening sample "
                "would be the tail of the confirmation sample's ordering."),
            "base_digest": C0_DIGEST,
            "base_digest_source": (
                "logs/stages/stage-1/phase_c1/plans/phase_c0_preregistration.json"
                ", commit be2ab08 — frozen long before any C2 candidate existed"),
            "outcome_dependence": (
                "NONE — no model output of any kind is consulted. Not a Search-1 "
                "ranking, not a state_eval value, not a C1 probe result."),
            "difficulty_reweighting": (
                "none. Source-native metadata is retained on the items and may "
                "be reported, but does not stratify."),
            "filter": "skip excluded ids and excluded normalized prompt hashes",
            "deterministic": True,
            "normalization": (
                "whitespace-collapsed, lowercased, for cross-role hashing"),
        },
        "isolation": provenance,
        "isolation_roles": sorted(provenance),
        "disjoint_from_confirmation": {
            "asset": args.c1_confirmation,
            "by": ["stable source id", "normalized prompt content hash"],
            "why": (
                "C0's inferential unit is the prompt and same-prompt cross-seed "
                "dependence is substantial, so seed-disjointness alone would let "
                "the screening selection leak into the confirmation. Disjoint "
                "prompts AND disjoint seeds make the confirmation genuinely "
                "held out."),
        },
        "sources": sources,
        "rendering": (
            "scripts/data/battery_render.py — the same renderers, instructions "
            "and id conventions as c1_confirmation_v1 and recovery_search"),
        "content_sha256": content,
        "content_sha256_convention": (
            "sha256 over newline-joined sorted 'id:prompt_sha256' pairs"),
        "no_model_has_been_evaluated_on_it": True,
        "authorizes": "nothing",
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"{manifest['n_prompts']} prompts / "
          f"{manifest['n_scorable_prompts']} scorable -> {args.out}")
    for name in sorted(outputs):
        print(f"  {name:14s} {outputs[name]['n']:5d}  "
              f"scorable={outputs[name]['scorable']}")
    print(f"content_sha256  {content}")
    print(f"manifest_sha256 {sha256_json(manifest)}")
    print("AUTHORIZES NOTHING; produces no verdict.")


if __name__ == "__main__":
    main()
