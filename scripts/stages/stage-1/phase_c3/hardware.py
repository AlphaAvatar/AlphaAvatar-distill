"""The approved GPU tier for formal C3, and the rule for choosing within it.

Maintainer decision 2026-09-29, after `63fe722a`: the `secure L40S only` rule
became a **fixed ordered availability policy** over three interchangeable
48 GB-class Ada parts. L40S was unavailable for eighteen acquisition attempts
and the design should not wait indefinitely on one SKU.

    1. NVIDIA L40S
    2. NVIDIA RTX 6000 Ada Generation   (display name "RTX 6000 Ada")
    3. NVIDIA L40

**Availability is the selection criterion, and the only one.** The first type
in this fixed order with usable secure capacity wins. Not the cheapest, not
the fastest, not the one a benchmark preferred, and never `communityPrice`.
Choosing on price would be price chasing; choosing on a measured result would
let the hardware be selected by the thing it is meant to measure.

**This is a resource-policy decision, not a claim of numerical equivalence.**
Nothing here asserts that a kernel is bitwise identical across these parts.
What decides whether a device may execute this C3 lineage is the pair of
exact frozen replay gates — the pre-ATTENTION parent and incumbent B — which
the session already runs before any recovery training. If they pass on the
selected device, that fact is recorded and execution continues; if either
fails, the device is not qualified, the mismatch evidence is preserved, and
the next type in the order gets a fresh one-use chain. A qualification
failure is a *pre-treatment hardware* result, never a C3 scientific result.

**The tier does not widen.** A100, H100, H200, 4090, A6000 and every other
architecture are outside this authorization. If none of the three can be
acquired, the `$0` capacity watch continues rather than the tier growing.

**One formal result uses one GPU type.** Once recovery training starts, the
type and the pinned runtime are frozen for all three arms, all nine probes
and all nine evaluations. Probes are never spliced across types and evidence
is never pooled across them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The approved tier, in the fixed order the selector walks. Order is the
#: policy; it is not sorted, scored or re-derived anywhere.
APPROVED_GPU_ORDER: tuple[str, ...] = (
    "NVIDIA L40S",
    #: The provider's id, not the display name. `NVIDIA RTX 6000 Ada` returns
    #: no such type and would have read as "permanently unavailable" while
    #: the part sat there at $0.84/h.
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA L40",
)

#: Display names, for evidence a human reads. Recorded beside the ids so a
#: report can say "RTX 6000 Ada" without anything selecting on that string.
GPU_DISPLAY_NAMES: dict[str, str] = {
    "NVIDIA L40S": "L40S",
    "NVIDIA RTX 6000 Ada Generation": "RTX 6000 Ada",
    "NVIDIA L40": "L40",
}

#: What every member of the tier must be. Recorded so a future addition has
#: to satisfy something rather than merely being appended to the tuple.
TIER_REQUIREMENTS: dict[str, Any] = {
    "secure_instances_only": True,
    "vram_class_gb": 48,
    "architecture": "Ada",
    "native_bf16": True,
    "single_gpu_execution": True,
}

#: Explicitly outside this authorization. Named rather than implied, because
#: "not in the approved list" and "deliberately excluded" read differently to
#: someone adding a line in a hurry.
NOT_AUTHORIZED: tuple[str, ...] = (
    "NVIDIA A100", "NVIDIA H100", "NVIDIA H200",
    "NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000",
)

#: Stock values that cannot yield a pod. `Low` is deliberately NOT here.
#:
#: I excluded it after six consecutive capacity refusals at `Low` on
#: 2026-09-28 -- and that was over-fitting to a streak. Both pods this
#: project has actually created for C3 were created while stock read `Low`:
#: attempt 7 (`3r1vgzkoa3855r`) and attempt 10 (`m82qrw2rzfzaos`). `Low`
#: means scarce and racy, not impossible, and excluding it would have made
#: the tier permanently dry on the only signal that has ever worked.
#:
#: `QUERY_FAILED:` is treated as unusable here but is NOT absence -- see
#: `query_offers`, which records the transport error rather than silently
#: reporting the type as unavailable.
UNUSABLE_STOCK: frozenset[str] = frozenset({"None", "null", "", "ABSENT"})


class C3HardwareError(RuntimeError):
    """The requested device is outside the approved tier, or the tier is dry."""


@dataclass(frozen=True)
class GpuOffer:
    """One approved type's live availability and price."""

    gpu_type_id: str
    secure_price_usd_per_hour: float | None
    stock_status: str | None

    @property
    def usable(self) -> bool:
        stock = str(self.stock_status)
        return (self.secure_price_usd_per_hour is not None
                and stock not in UNUSABLE_STOCK
                and not stock.startswith("QUERY_FAILED"))

    def as_dict(self) -> dict[str, Any]:
        return {"gpu_type_id": self.gpu_type_id,
                "display_name": GPU_DISPLAY_NAMES.get(self.gpu_type_id, ""),
                "secure_price_usd_per_hour": self.secure_price_usd_per_hour,
                "stock_status": self.stock_status,
                "usable": self.usable}


def require_approved(gpu_type_id: str) -> str:
    """Refuse anything outside the tier, by name."""
    if gpu_type_id in APPROVED_GPU_ORDER:
        return gpu_type_id
    if gpu_type_id in NOT_AUTHORIZED:
        raise C3HardwareError(
            f"{gpu_type_id!r} is explicitly outside this authorization. The "
            f"approved tier is {list(APPROVED_GPU_ORDER)}; crossing GPU "
            "architecture needs another maintainer decision.")
    raise C3HardwareError(
        f"{gpu_type_id!r} is not in the approved tier {list(APPROVED_GPU_ORDER)}")


def select(offers: dict[str, GpuOffer]) -> GpuOffer | None:
    """The first approved type, IN ORDER, with usable secure capacity.

    Pure: the caller supplies the live inventory. Returns None when the whole
    tier is dry, which is the `$0` watch's condition rather than an error --
    an empty tier is a fact about the provider, not a fault.
    """
    for gpu_type_id in APPROVED_GPU_ORDER:
        offer = offers.get(gpu_type_id)
        if offer is not None and offer.usable:
            return offer
    return None


def query_offers(api_key: str | None = None,
                 timeout: float = 30.0) -> dict[str, GpuOffer]:
    """Live secure price and stock for every approved type.

    `securePrice` only. Reporting `communityPrice` once produced a number the
    launcher could not act on, because the launcher prices on secure.
    """
    import json
    import os
    import sys
    import urllib.error
    import urllib.request
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo / "src"))
    from aadistill.infrastructure.provider import USER_AGENT, read_api_key

    key = api_key or os.environ.get("RUNPOD_API_KEY")
    if not key:
        cfg = os.path.expanduser("~/.runpod/config.toml")
        if not os.path.isfile(cfg):
            raise C3HardwareError(
                f"no RUNPOD_API_KEY and no {cfg}; cannot read live capacity")
        key = read_api_key(cfg)

    out: dict[str, GpuOffer] = {}
    for gpu_type_id in APPROVED_GPU_ORDER:
        query = ('query { gpuTypes(input:{id:"%s"}) { id securePrice '
                 'lowestPrice(input:{gpuCount:1}) { stockStatus } } }'
                 % gpu_type_id)
        req = urllib.request.Request(
            f"https://api.runpod.io/graphql?api_key={key}",
            data=json.dumps({"query": query}).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                doc = json.loads(fh.read().decode())
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            #: A transport failure is "unknown", never "unavailable": treating
            #: it as absence would silently walk past an available type.
            out[gpu_type_id] = GpuOffer(gpu_type_id, None, f"QUERY_FAILED: {exc}")
            continue
        rows = ((doc.get("data") or {}).get("gpuTypes") or [])
        if not rows:
            out[gpu_type_id] = GpuOffer(gpu_type_id, None, "ABSENT")
            continue
        row = rows[0]
        price = row.get("securePrice")
        stock = ((row.get("lowestPrice") or {}).get("stockStatus"))
        out[gpu_type_id] = GpuOffer(
            gpu_type_id,
            float(price) if price is not None else None,
            None if stock is None else str(stock))
    return out


def main(argv=None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    offers = query_offers()
    chosen = select(offers)
    if args.json:
        print(json.dumps(
            {"approved_order": list(APPROVED_GPU_ORDER),
             "offers": {k: v.as_dict() for k, v in offers.items()},
             "selected": chosen.as_dict() if chosen else None,
             "_rule": "first approved type IN ORDER with usable secure capacity"},
            indent=1, sort_keys=True))
        return 0 if chosen else 1
    for gpu_type_id in APPROVED_GPU_ORDER:
        o = offers[gpu_type_id]
        price = f"${o.secure_price_usd_per_hour:.4f}/h" if \
            o.secure_price_usd_per_hour is not None else "no securePrice"
        mark = "USABLE" if o.usable else "      "
        print(f"  [{mark}] {gpu_type_id:22} {price:16} stock={o.stock_status}")
    print()
    if chosen:
        print(f"SELECTED: {chosen.gpu_type_id} at "
              f"${chosen.secure_price_usd_per_hour:.4f}/h "
              f"(stock {chosen.stock_status})")
        return 0
    print("no approved GPU has usable secure capacity; the $0 watch continues")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
