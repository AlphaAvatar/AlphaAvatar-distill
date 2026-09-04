"""The Phase-C1 confirmation battery: counts, isolation, and rendering parity.

The parity test is the important one. C1 renders prompts through
`scripts/data/battery_render.py` while the frozen `recovery_search_v2` was built
by closures inside `build_recovery_search_battery.py`. Two copies of a rendering
convention drift; this asserts they have not, by re-rendering the frozen
artifact's own source rows and requiring byte equality.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/data"))

from aadistill.data.extra_stream import content_sha256  # noqa: E402
from battery_render import (FROZEN_SOURCES, RENDERERS, check_group_parity,  # noqa: E402
                            norm, rank_key, rank_take)

BATTERY = REPO / "artifacts/stage3/c1_confirmation_v1"
FROZEN = REPO / "artifacts/stage3/recovery_search_v2"
C0_DIGEST = "fb2eeea531f9f0d11f84b77cd47dff30697122de90a072a7a80c3a7535e89280"

MIXTURE = {"gsm8k": 150, "math_verified": 150, "multihop": 150, "rag": 150,
           "knowledge": 150, "tool": 100, "code": 100}


def load(directory: Path) -> list[dict]:
    out = []
    for p in sorted(directory.glob("*.jsonl")):
        out.extend(json.loads(l) for l in p.open() if l.strip())
    return out


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((BATTERY / "manifest.json").read_text())


@pytest.fixture(scope="module")
def items() -> list[dict]:
    return load(BATTERY)


# --- the frozen shape -------------------------------------------------------

def test_the_counts_are_exactly_the_frozen_mixture(items, manifest):
    counts = {g: sum(1 for i in items if i["group"] == g) for g in MIXTURE}
    assert counts == MIXTURE
    assert len(items) == 950
    scorable = sum(1 for i in items if manifest["sets"][i["group"]]["scorable"])
    assert scorable == 850
    assert manifest["n_prompts"] == 950 and manifest["n_scorable_prompts"] == 850


def test_code_is_the_only_behaviour_only_set(manifest):
    assert manifest["behaviour_only_sets"] == ["code"]
    assert sorted(manifest["scorable_sets"]) == [
        "gsm8k", "knowledge", "math_verified", "multihop", "rag", "tool"]


def test_the_content_hash_still_describes_the_artifact(items, manifest):
    pairs = sorted(f"{i['id']}:{i['prompt_sha256']}" for i in items)
    assert hashlib.sha256("\n".join(pairs).encode()).hexdigest() == \
        manifest["content_sha256"]


def test_each_stored_prompt_hash_is_its_prompt(items):
    for i in items:
        assert content_sha256(norm(i["prompt_text"])) == i["prompt_sha256"], i["id"]


def test_there_are_no_duplicates_inside_the_battery(items):
    assert len({i["id"] for i in items}) == len(items)
    assert len({i["prompt_sha256"] for i in items}) == len(items)


# --- isolation --------------------------------------------------------------

@pytest.mark.parametrize("role_dir", ["artifacts/eval/battery_v2",
                                      "artifacts/stage3/recovery_search_v2"])
def test_it_is_disjoint_from_each_jsonl_role_by_id_and_by_content(items, role_dir):
    rows = load(REPO / role_dir)
    role_ids = {str(r["id"]) for r in rows} | {
        str(r["source_key"]) for r in rows if r.get("source_key")}
    role_hashes = {content_sha256(norm(r.get("prompt_text", ""))) for r in rows}
    own_ids = {str(i["id"]) for i in items} | {
        str(i["source_key"]) for i in items if i.get("source_key")}
    assert not own_ids & role_ids
    assert not {i["prompt_sha256"] for i in items} & role_hashes


def test_it_is_disjoint_from_the_recovery_training_corpus(items):
    ids, hashes = set(), set()
    for line in (REPO / "artifacts/stage3/corpus_v2/sessions.jsonl").open():
        d = json.loads(line)
        ids.add(str(d["source_id"]))
        hashes.add(content_sha256(norm("\n".join(
            str(m.get("content", "")) for m in d["messages"]
            if m.get("role") != "assistant"))))
    own_ids = {str(i["id"]) for i in items} | {
        str(i["source_key"]) for i in items if i.get("source_key")}
    assert not own_ids & ids
    assert not {i["prompt_sha256"] for i in items} & hashes


@pytest.mark.parametrize("rel", ["artifacts/stage1/state_eval_v1/items.jsonl",
                                 "artifacts/stage1/e8_calibration_v1/items.jsonl"])
def test_it_is_disjoint_from_the_token_id_roles(items, rel):
    ids = set()
    for line in (REPO / rel).open():
        if line.strip():
            d = json.loads(line)
            if d.get("source_id"):
                ids.add(str(d["source_id"]))
    own = {str(i["id"]) for i in items} | {
        str(i["source_key"]) for i in items if i.get("source_key")}
    assert not own & ids


def test_final_promotion_is_still_intact_and_was_only_read(manifest):
    """It is an exclusion source, never a sample source."""
    assert manifest["isolation"]["final_promotion"]["n_prompts"] == 770
    assert "not sampled from" in manifest["isolation"]["final_promotion"]["note"]
    assert (REPO / "artifacts/eval/battery_v2/manifest.json").is_file()


# --- selection is deterministic and outcome-independent ---------------------

def test_selection_is_by_cryptographic_rank_keyed_by_the_c0_digest(items, manifest):
    assert manifest["sampling_rule"]["base_digest"] == C0_DIGEST
    assert manifest["sampling_rule"]["outcome_dependence"] == (
        "NONE — no model output of any kind is consulted")
    # Every chosen item must be among the lowest-ranked of its stratum: no
    # chosen rank may exceed the smallest rank that was left out.
    for group in ("gsm8k", "code"):                 # one scorable, one behaviour-only
        chosen = [i for i in items if i["group"] == group]
        ranks = sorted(rank_key(C0_DIGEST, group, str(i["id"])) for i in chosen)
        assert ranks == sorted(ranks)
        assert len(set(ranks)) == len(ranks)


def test_rank_take_is_deterministic_and_ignores_row_order():
    rows = [{"_index": i, "question": f"q{i}", "answer": f"x #### {i}"}
            for i in range(50)]
    a = rank_take(rows, 10, stratum="gsm8k", base_digest=C0_DIGEST,
                  exclude_ids=set(), exclude_hashes=set(),
                  make=RENDERERS["gsm8k"])
    b = rank_take(list(reversed(rows)), 10, stratum="gsm8k", base_digest=C0_DIGEST,
                  exclude_ids=set(), exclude_hashes=set(),
                  make=RENDERERS["gsm8k"])
    assert [i["id"] for i in a] == [i["id"] for i in b]
    assert len(a) == 10


def test_rank_take_honours_exclusions_before_ranking():
    rows = [{"_index": i, "question": f"q{i}", "answer": f"x #### {i}"}
            for i in range(50)]
    full = rank_take(rows, 5, stratum="gsm8k", base_digest=C0_DIGEST,
                     exclude_ids=set(), exclude_hashes=set(),
                     make=RENDERERS["gsm8k"])
    banned = str(full[0]["id"])
    without = rank_take(rows, 5, stratum="gsm8k", base_digest=C0_DIGEST,
                        exclude_ids={banned}, exclude_hashes=set(),
                        make=RENDERERS["gsm8k"])
    assert banned not in {str(i["id"]) for i in without}
    assert [str(i["id"]) for i in without[:4]] == [str(i["id"]) for i in full[1:5]]


def test_a_different_base_digest_would_select_a_different_sample():
    """The key really is doing the work — otherwise it is decoration."""
    rows = [{"_index": i, "question": f"q{i}", "answer": f"x #### {i}"}
            for i in range(200)]
    a = rank_take(rows, 20, stratum="gsm8k", base_digest=C0_DIGEST,
                  exclude_ids=set(), exclude_hashes=set(), make=RENDERERS["gsm8k"])
    b = rank_take(rows, 20, stratum="gsm8k", base_digest="0" * 64,
                  exclude_ids=set(), exclude_hashes=set(), make=RENDERERS["gsm8k"])
    assert [i["id"] for i in a] != [i["id"] for i in b]


def test_the_stratum_is_part_of_the_key():
    assert rank_key(C0_DIGEST, "gsm8k", "x") != rank_key(C0_DIGEST, "rag", "x")


# --- rendering parity with the frozen recovery-search battery ---------------

@pytest.mark.parametrize("group", sorted(FROZEN_SOURCES))
def test_the_shared_renderers_reproduce_the_frozen_battery_byte_for_byte(group):
    """Re-render `recovery_search_v2`'s own items and require exact equality.

    This is what stops the C1 copy of the rendering convention from drifting
    away from the one every historical measurement used.

    These seven cases — and only these seven — need the pinned source snapshots
    themselves, which are ~4 GiB of Hugging Face datasets that the C1 pod is not
    given and does not need: they are a readiness input, never a C1 runtime or
    scientific one. On a host without them the case skips rather than fails,
    which is what aborted attempt 3R at the pod's setup test gate.

    Skipping does **not** retire the guarantee, it relocates it: the dev-box
    `renderer_parity_gate` requires all seven present and all seven PASS, with
    zero skips, before a C1 provider may be created.
    """
    result = check_group_parity(group)
    if result["status"] == "SOURCE_ABSENT":
        pytest.skip(
            f"pinned source snapshot absent: {result['repo_id']}@{result['revision']} "
            f"(file {result['file']}) expected at {result['resolved_snapshot']} — "
            "renderer parity is enforced at $0 by scripts/autoinit/renderer_parity_gate.py")
    assert result["mismatches"] == [], f"{group}: {result['mismatches'][:5]}"
    assert not result["missing"], (
        f"{group}: re-rendered {result['n_checked']} of {result['n_frozen']} frozen "
        f"items; missing {result['missing'][:5]}")
    assert result["status"] == "PASS"


def test_the_sources_are_the_same_pinned_revisions(manifest):
    for group, (repo, rev, rel) in FROZEN_SOURCES.items():
        got = manifest["sources"][group]
        assert got["repo_id"] == repo and got["revision"] == rev and got["file"] == rel


def test_the_mixture_ratio_matches_the_historical_battery(manifest):
    """3:3:3:3:3:2 scorable, plus the same relative behaviour-only code share."""
    hist = json.loads((FROZEN / "manifest.json").read_text())
    hist_counts = {k: v["n"] for k, v in hist["sets"].items()}
    scale = 5.0
    for group, n in MIXTURE.items():
        assert n == hist_counts[group] * scale, group
