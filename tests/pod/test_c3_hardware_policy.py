"""The approved GPU tier, and the one thing that may choose within it.

The 2026-09-29 amendment replaced `secure L40S only` with a fixed ordered
availability policy over three 48 GB-class Ada parts. Two ways that goes
wrong, and both are cheap to prevent:

* **selecting on anything but availability.** Price, a benchmark result or
  `communityPrice` would each be a different policy wearing this one's name —
  and selecting on a measured result would let the hardware be chosen by the
  thing it is meant to measure.
* **the tier quietly widening.** A100/H100/H200/4090/A6000 are outside this
  authorization; crossing architecture needs a maintainer decision, not a
  flag.

Everything is pure: the inventory is a parameter, so the whole decision
surface is testable without touching the provider or spending anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from experiments.phase_c3 import hardware as H  # noqa: E402


def python_code(text: str) -> str:
    """Python with comments and ALL string literals removed.

    `split('\"\"\"', 2)[2]` only drops the MODULE docstring -- every function
    docstring after it survives, so a check for "does this name appear in
    executable code" reports the prose explaining why it does not.
    """
    import io
    import tokenize

    out = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            continue
        out.append('""' if tok.type == tokenize.STRING else tok.string)
    return " ".join(out)


def shell_code(text: str) -> str:
    """Shell with `#` comment lines removed. Same reason."""
    return "\n".join(l for l in text.splitlines()
                      if not l.lstrip().startswith("#"))


def offer(gpu: str, price: float | None, stock: str | None) -> H.GpuOffer:
    return H.GpuOffer(gpu, price, stock)


def inventory(**by_short: tuple[float | None, str | None]) -> dict[str, H.GpuOffer]:
    """Build a tier inventory from short names, in the approved order."""
    short = {"l40s": "NVIDIA L40S",
             "ada6000": "NVIDIA RTX 6000 Ada Generation",
             "l40": "NVIDIA L40"}
    return {short[k]: offer(short[k], p, s) for k, (p, s) in by_short.items()}


def test_the_order_is_the_policy_and_is_not_sorted():
    assert H.APPROVED_GPU_ORDER == (
        "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40")
    #: Not alphabetical, not by price. If it were either, the order would be
    #: a consequence of something else and could drift when that changed.
    assert list(H.APPROVED_GPU_ORDER) != sorted(H.APPROVED_GPU_ORDER)


def test_selection_takes_the_first_usable_in_order_not_the_cheapest():
    """The whole point. L40S is the DEAREST and still wins when usable."""
    inv = inventory(l40s=(1.09, "Low"), ada6000=(0.84, "High"), l40=(0.82, "High"))
    chosen = H.select(inv)
    assert chosen is not None and chosen.gpu_type_id == "NVIDIA L40S", (
        "selection picked a cheaper part over an available earlier one; that "
        "is price chasing, which this policy forbids")


def test_selection_falls_through_in_order_when_earlier_types_are_dry():
    inv = inventory(l40s=(1.09, "None"), ada6000=(0.84, "Medium"), l40=(0.82, "High"))
    assert H.select(inv).gpu_type_id == "NVIDIA RTX 6000 Ada Generation"
    inv = inventory(l40s=(1.09, "None"), ada6000=(0.84, "None"), l40=(0.82, "High"))
    assert H.select(inv).gpu_type_id == "NVIDIA L40"


def test_a_dry_tier_is_none_rather_than_an_error():
    """An empty tier is a fact about the provider, not a fault: it is the
    `$0` watch's continue condition."""
    inv = inventory(l40s=(1.09, "None"), ada6000=(0.84, "None"), l40=(0.82, "None"))
    assert H.select(inv) is None


def test_low_stock_counts_as_usable():
    """Both pods this project ever created for C3 were created at `Low`.

    I had excluded `Low` after six consecutive refusals -- over-fitting to a
    streak. Excluding it makes the tier permanently dry on the only signal
    that has ever actually produced a pod.
    """
    assert "Low" not in H.UNUSABLE_STOCK
    assert offer("NVIDIA L40S", 1.09, "Low").usable is True


@pytest.mark.parametrize("stock", ["None", "null", "", "ABSENT", None])
def test_absence_is_not_usable(stock):
    assert offer("NVIDIA L40S", 1.09, stock).usable is False


def test_a_type_with_no_secure_price_is_not_usable():
    """securePrice is what the launcher prices on. No secure price, no run."""
    assert offer("NVIDIA L40", None, "High").usable is False


def test_a_transport_failure_is_not_read_as_absence():
    """A network blip must not silently walk past an available type."""
    o = offer("NVIDIA L40S", 1.09, "QUERY_FAILED: timed out")
    assert o.usable is False
    #: and it is distinguishable from real absence, which is the point.
    assert "QUERY_FAILED" in str(o.stock_status)
    assert str(o.stock_status) not in H.UNUSABLE_STOCK


@pytest.mark.parametrize("gpu", [
    "NVIDIA A100", "NVIDIA H100", "NVIDIA H200",
    "NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000",
])
def test_excluded_architectures_are_refused_by_name(gpu):
    with pytest.raises(H.C3HardwareError, match="outside this authorization"):
        H.require_approved(gpu)


def test_an_unknown_type_is_refused_too():
    with pytest.raises(H.C3HardwareError, match="not in the approved tier"):
        H.require_approved("NVIDIA A40")


def test_every_approved_type_is_accepted():
    for gpu in H.APPROVED_GPU_ORDER:
        assert H.require_approved(gpu) == gpu


def test_selection_never_reads_community_price():
    """The behaviour, not a substring sweep.

    Two lenses collide here: blanking string literals hides the GraphQL query
    (where `securePrice` legitimately lives), while keeping them surfaces the
    module docstring (where `communityPrice` legitimately lives, explaining
    that it is never used). So this asserts what the code DOES: it requests
    and reads securePrice, and nowhere reads communityPrice.
    """
    src = (REPO / "scripts/experiments/phase_c3/hardware.py").read_text()
    assert 'row.get("securePrice")' in src, (
        "the tier does not read securePrice from the response")
    assert "securePrice" in src
    for forbidden in ('get("communityPrice")', "get('communityPrice')",
                      '["communityPrice"]'):
        assert forbidden not in src, (
            f"the tier reads {forbidden}; the launcher prices on securePrice "
            "and must not fall back")


def test_the_tier_records_what_membership_requires():
    """A future addition must satisfy something, not merely be appended."""
    r = H.TIER_REQUIREMENTS
    assert r["vram_class_gb"] == 48
    assert r["architecture"] == "Ada"
    assert r["native_bf16"] is True
    assert r["secure_instances_only"] is True
    assert r["single_gpu_execution"] is True


def test_the_launcher_refuses_an_unapproved_device_before_anything_exists():
    """At $0, in spec(), before a SessionSpec can be built."""
    import importlib.util

    for p in ("scripts/pod", "scripts/autoinit"):
        sys.path.insert(0, str(REPO / p))
    spec = importlib.util.spec_from_file_location(
        "c3launch_gpu", REPO / "scripts/pod/autoinit_c3_launch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Args:
        gpu = "NVIDIA H100"
        create_attempts = 1
        host_draws = 1

    with pytest.raises(H.C3HardwareError):
        mod.spec(Args())


def test_the_acquisition_loop_creates_no_run_directory_while_polling():
    """Section 11: stock polling must not explode the run tree.

    The watch reads inventory and writes nothing; a chain is built only after
    a usable type has been seen. Asserted structurally, because the failure
    mode is a `mkdir` that drifts above the capacity check.
    """
    raw = (REPO / "scripts/pod/c3_acquire.sh").read_text()
    watch = raw[raw.index("# ---- $0 capacity watch"):raw.index("# ---- re-price LIVE")]
    watch = shell_code(watch)
    for forbidden in ("mkdir", "issue_c3_authorization", "stage_c1_bundle",
                      "record_pod_environment", "git commit"):
        assert forbidden not in watch, (
            f"the $0 capacity watch runs {forbidden!r}; polling must create "
            f"nothing")
    #: and the chain is built strictly AFTER a usable type has been seen.
    code = shell_code(raw)
    assert code.index('PICK="$(usable_now)"') < code.index('mkdir -p "logs/stages')


def test_the_loop_reprices_on_the_device_it_will_use():
    """$1.09/h must not be carried from L40S onto another part."""
    raw = (REPO / "scripts/pod/c3_acquire.sh").read_text()
    code = shell_code(raw)
    assert "query_live_secure_price(gpu)" in code
    assert "--gpu" in code and '--max-price "$PRICE"' in code
    assert "1.09" not in code, "a rate is hard-coded in the acquisition loop"
