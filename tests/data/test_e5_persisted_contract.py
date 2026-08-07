"""Both E5 producers must satisfy one packing contract *through disk*.

Attempt 1 on 2026-08-07 generated, gated and wrote 4,196 arm-R examples and then
died at the pairing gate: `to_record()` emitted the summary counts and dropped
the token payload. Every in-memory check had passed, and arm C -- which writes
its records through a different code path -- was unaffected, so the whole
offline suite was green while the corpus that cost GPU time was unusable.

The lesson is about *where* the assertion sits. Checking the in-memory
`RecoveryExample` proves nothing, because the object that crashed the pod was
the one reconstructed from JSON. So these tests run the complete persisted path

    RecoveryExample -> to_record -> JSON file -> reload
                    -> example_to_rendered -> pack_e5 -> write_pack -> verify_pack

for an R-shaped example carrying the real structure: a student-generated prefix,
a supervised teacher recovery, the system mapping, the source seed and the
truncation metadata. `test_both_producers_survive_the_same_persisted_path` runs
C-shaped and R-shaped records through the identical assertions, which is the
specific gap that let one producer drift.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.data.e5_pack import (  # noqa: E402
    REQUIRED_FIELDS, example_to_rendered, pack_e5, verify_pack, write_pack,
)
from aadistill.data.recovery import build_example  # noqa: E402

PAD = 151643
STOP = 151645
SYSTEM_KEY = "a" * 64
SYSTEM_IDS = [1001, 1002, 1003, 1004, 1005]      # the rendered system block


def r_example(session: str, truncation_index: int, *, prompt_len=9,
              prefix_len=17, cont_len=23, seed="sa"):
    """One arm-R example with the real prefix/recovery structure.

    `prompt_ids` opens with the system block, exactly as the builder renders it
    from the full message list -- that is what makes `n_system_tokens` a slice
    index rather than an extra quantity to add on.
    """
    prompt = SYSTEM_IDS + list(range(2000, 2000 + prompt_len - len(SYSTEM_IDS)))
    student_prefix = list(range(3000, 3000 + prefix_len))       # student rollout
    recovery = list(range(4000, 4000 + cont_len - 1)) + [STOP]  # teacher recovery
    return build_example(
        prompt_ids=prompt, student_prefix_ids=student_prefix,
        teacher_continuation_ids=recovery, source_session_id=session,
        source_seed=seed, truncation_index=truncation_index,
        truncation_fraction=0.25 + 0.3 * truncation_index, data_type="gsm8k",
        system_key=SYSTEM_KEY, n_system_tokens=len(SYSTEM_IDS))


def c_record(session: str, truncation_index: int, *, prefix_len=26, cont_len=23,
             seed="sa"):
    """An arm-C record in the shape its own builder writes.

    Written here as a literal rather than by importing the builder, so the test
    fails if the two producers' on-disk shapes ever diverge again.
    """
    body = list(range(5000, 5000 + prefix_len)) + \
        list(range(6000, 6000 + cont_len - 1)) + [STOP]
    return {
        "id": f"{session}#t{truncation_index}",
        "ids": SYSTEM_IDS + body,
        "mask": [False] * (len(SYSTEM_IDS) + prefix_len) + [True] * cont_len,
        "system_key": SYSTEM_KEY, "n_system_tokens": len(SYSTEM_IDS),
        "source_session_id": session, "source_seed": seed,
        "truncation_index": truncation_index,
        "truncation_fraction": 0.25 + 0.3 * truncation_index,
        "data_type": "gsm8k", "arm": "C", "prefix_source": "teacher_forced",
        "n_prefix_tokens": len(SYSTEM_IDS) + prefix_len,
        "n_continuation_tokens": cont_len,
        "n_total_tokens": len(SYSTEM_IDS) + prefix_len + cont_len,
    }


def write_and_reload(records, tmp_path: Path) -> list[dict]:
    """Persist exactly as the builder does, then read back exactly as the driver
    does. Neither side may keep the in-memory object."""
    path = tmp_path / "examples.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return [json.loads(line) for line in path.open() if line.strip()]


# ---------------------------------------------------- the persisted R contract

def test_r_record_survives_disk_and_packs(tmp_path):
    """The full path, on an R example, from construction to a verified pack."""
    examples = [r_example(f"gsm8k-{i}", j)
                for i in range(6) for j in range(2)]     # atomic two-cut bundles
    reloaded = write_and_reload([e.to_record() for e in examples], tmp_path)

    assert len(reloaded) == 12
    for rec in reloaded:
        assert not [f for f in REQUIRED_FIELDS if f not in rec]

    rendered = [example_to_rendered(r) for r in reloaded]
    for rec, rend in zip(reloaded, rendered):
        assert rend.body_ids == rec["ids"][len(SYSTEM_IDS):]
        assert sum(rend.body_mask) == rec["n_continuation_tokens"]
        assert rend.n_system_tokens == len(SYSTEM_IDS)

    blocks = pack_e5(reloaded, {SYSTEM_KEY: SYSTEM_IDS}, block_len=256, pad_id=PAD)
    out = tmp_path / "pack"
    ce_total = sum(r["n_continuation_tokens"] for r in reloaded)
    write_pack(blocks, out, arm="r", seed="sa", block_len=256, pad_id=PAD,
               target_ce_tokens=ce_total)
    report = verify_pack(out, reloaded, expected_blocks=len(blocks),
                         target_ce_tokens=ce_total, tolerance=0.05,
                         steps=max(1, len(blocks) * 3 // 2), blocks_per_step=2)
    assert report["passed"], report["failures"]


def test_the_supervised_tokens_that_land_in_the_pack_are_the_recovery_tokens(tmp_path):
    """Not just the right *count* -- the right tokens.

    A pack can carry the correct number of CE positions while supervising the
    prefix instead of the recovery, which would train R on the student's own
    output and quietly answer a different question than the registration asks.
    """
    ex = r_example("gsm8k-1", 0, prompt_len=9, prefix_len=17, cont_len=23)
    reloaded = write_and_reload([ex.to_record()], tmp_path)
    blocks = pack_e5(reloaded, {SYSTEM_KEY: SYSTEM_IDS}, block_len=256, pad_id=PAD)
    out = tmp_path / "pack"
    write_pack(blocks, out, arm="r", seed="sa", block_len=256, pad_id=PAD,
               target_ce_tokens=23)

    arrays = np.load(out / "blocks.npz")
    ids, ce = arrays["input_ids"][0], arrays["ce_mask"][0]
    supervised = [int(t) for t, m in zip(ids, ce) if m]
    assert supervised == ex.ids[-23:], "the CE mask must cover the teacher recovery"
    assert supervised[-1] == STOP
    # The student prefix is present as context and is never supervised.
    assert all(t not in supervised for t in range(3000, 3017))


def test_truncation_metadata_and_seed_survive_to_the_audit(tmp_path):
    ex = r_example("gsm8k-7", 1, seed="sb")
    reloaded = write_and_reload([ex.to_record()], tmp_path)
    assert reloaded[0]["source_seed"] == "sb"
    assert reloaded[0]["truncation_index"] == 1
    assert reloaded[0]["truncation_fraction"] == pytest.approx(0.55)

    blocks = pack_e5(reloaded, {SYSTEM_KEY: SYSTEM_IDS}, block_len=256, pad_id=PAD)
    out = tmp_path / "pack"
    write_pack(blocks, out, arm="r", seed="sb", block_len=256, pad_id=PAD,
               target_ce_tokens=23)
    audit = [json.loads(line) for line in (out / "audit.jsonl").open() if line.strip()]
    members = [m for row in audit for m in row.get("sessions", [])]
    assert members, "the audit must name the examples in each block"
    # `bundle_id` is the trajectory, which is how a rejected R bundle removes its
    # C twin later; the per-truncation session id keeps the two cuts distinct.
    assert {m["bundle_id"] for m in members} == {"gsm8k-7"}
    assert any(m["session_id"].endswith("#r1") for m in members)


def test_bundle_siblings_are_never_co_packed(tmp_path):
    """Two cuts of one trajectory are prefixes of each other; sharing a block
    would leak one into the other's context."""
    examples = [r_example("gsm8k-1", 0), r_example("gsm8k-1", 1)]
    reloaded = write_and_reload([e.to_record() for e in examples], tmp_path)
    blocks = pack_e5(reloaded, {SYSTEM_KEY: SYSTEM_IDS}, block_len=8192, pad_id=PAD)
    assert len(blocks) == 2, "siblings must be deferred to separate blocks"


# ------------------------------------------------ one contract, two producers

def test_both_producers_survive_the_same_persisted_path(tmp_path):
    """The gap that cost attempt 1: only one producer was ever exercised."""
    r_records = [r_example(f"gsm8k-{i}", j).to_record()
                 for i in range(4) for j in range(2)]
    c_records = [c_record(f"gsm8k-{i}", j) for i in range(4) for j in range(2)]

    packs = {}
    for arm, records in (("r", r_records), ("c", c_records)):
        d = tmp_path / arm
        d.mkdir()
        reloaded = write_and_reload(records, d)
        for rec in reloaded:
            missing = [f for f in REQUIRED_FIELDS if f not in rec]
            assert not missing, f"arm {arm.upper()} record missing {missing}"
            example_to_rendered(rec)
        blocks = pack_e5(reloaded, {SYSTEM_KEY: SYSTEM_IDS}, block_len=256,
                         pad_id=PAD)
        ce_total = sum(sum(r["mask"]) for r in reloaded)
        write_pack(blocks, d / "pack", arm=arm, seed="sa", block_len=256,
                   pad_id=PAD, target_ce_tokens=ce_total)
        report = verify_pack(d / "pack", reloaded, expected_blocks=len(blocks),
                             target_ce_tokens=ce_total, tolerance=0.05,
                             steps=max(1, len(blocks) * 3 // 2), blocks_per_step=2)
        assert report["passed"], f"arm {arm.upper()}: {report['failures']}"
        packs[arm] = len(blocks)

    # Both arms hold 8 examples of the same length here, so they must pack alike.
    assert packs["c"] == packs["r"]


def test_a_producer_that_drops_the_payload_is_caught_before_packing(tmp_path):
    """The 2026-08-07 failure, reproduced: a record with only the counts."""
    rec = r_example("gsm8k-1", 0).to_record()
    for dropped in ("ids", "mask", "system_key", "n_system_tokens"):
        crippled = {k: v for k, v in rec.items() if k != dropped}
        reloaded = write_and_reload([crippled], tmp_path)
        with pytest.raises(ValueError, match=f"missing.*{dropped}"):
            example_to_rendered(reloaded[0])


# --- the validation tail (2026-08-07, attempt 7) ------------------------------
# Attempt 7 passed every gate, including gate 2, and then died on the first
# training call: `ladder_blocks` takes its validation set from the blocks PAST
# the largest rung, and an E5 pack whose single rung covered every block left an
# empty tail. The pack was correct as a corpus and unloadable as an artifact.

def test_the_pack_loads_through_the_production_ladder_loader(tmp_path):
    """The exact call that failed: pack -> ladder_blocks."""
    from aadistill.data.ladder import ladder_blocks

    train = [r_example(f"gsm8k-{i}", j).to_record()
             for i in range(40) for j in range(2)]
    held = [r_example(f"held-{i}", j).to_record() for i in range(40) for j in range(2)]
    reloaded = write_and_reload(train, tmp_path)
    d = tmp_path / "held"
    d.mkdir()
    held_reloaded = write_and_reload(held, d)

    sysmap = {SYSTEM_KEY: SYSTEM_IDS}
    blocks = pack_e5(reloaded, sysmap, block_len=256, pad_id=PAD)
    val = pack_e5(held_reloaded, sysmap, block_len=256, pad_id=PAD)[:4]
    out = tmp_path / "pack"
    ce_total = sum(r["n_continuation_tokens"] for r in reloaded)
    write_pack(blocks, out, arm="r", seed="sa", block_len=256, pad_id=PAD,
               target_ce_tokens=ce_total, val_blocks=val)

    tr, va, _ = ladder_blocks(out, ce_total, n_val=4)
    assert tr[0].shape[0] == len(blocks), "training must be the rung, not the file"
    assert va[0].shape[0] == 4
    # The budget is the rung. Counting the tail would overstate what was trained on.
    assert int(tr[1].sum()) == ce_total


def test_a_pack_with_no_validation_tail_is_rejected(tmp_path):
    """Reproduces attempt 7 directly: no tail, and the loader cannot proceed."""
    from aadistill.data.ladder import ladder_blocks

    reloaded = write_and_reload(
        [r_example(f"gsm8k-{i}", j).to_record() for i in range(20) for j in range(2)],
        tmp_path)
    blocks = pack_e5(reloaded, {SYSTEM_KEY: SYSTEM_IDS}, block_len=256, pad_id=PAD)
    out = tmp_path / "pack"
    ce_total = sum(r["n_continuation_tokens"] for r in reloaded)
    write_pack(blocks, out, arm="r", seed="sa", block_len=256, pad_id=PAD,
               target_ce_tokens=ce_total)
    with pytest.raises(ValueError, match="blocks past the largest rung"):
        ladder_blocks(out, ce_total, n_val=4)


def test_validation_blocks_share_no_bundle_with_training(tmp_path):
    """A tail drawn from training bundles would leak, and `verify_pack` must say so."""
    reloaded = write_and_reload(
        [r_example(f"gsm8k-{i}", j).to_record() for i in range(30) for j in range(2)],
        tmp_path)
    sysmap = {SYSTEM_KEY: SYSTEM_IDS}
    blocks = pack_e5(reloaded, sysmap, block_len=256, pad_id=PAD)
    out = tmp_path / "pack"
    ce_total = sum(r["n_continuation_tokens"] for r in reloaded)
    # Deliberately reuse training blocks as the "validation" tail.
    write_pack(blocks[:20], out, arm="r", seed="sa", block_len=256, pad_id=PAD,
               target_ce_tokens=ce_total, val_blocks=blocks[:4])
    rep = verify_pack(out, reloaded, expected_blocks=20, target_ce_tokens=ce_total,
                      tolerance=1.0, steps=30, blocks_per_step=2,
                      expected_val_blocks=4)
    assert "validation_is_held_out" in rep["failures"]
