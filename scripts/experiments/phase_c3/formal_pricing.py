"""What formal C3 costs at a LIVE provider rate, and whether it is fundable.

The 2026-09-28 amendment made the provider price a live pricing *input* rather
than a frozen bound. `$1.09/h` is a historical observation; a different live
`securePrice` is re-priced, not refused. So nothing here hard-codes a rate:
`price_c3(rate)` is a pure function of the rate and the measured component
table, and `fundable(...)` answers the package question against the amended
envelopes.

**The envelope is not the grant.** `$30.00` is the per-session ceiling the
package allows; C3's one-use authorization receives the ceiling *derived*
here. A session may not be issued at the envelope merely because the envelope
exists.

Three conditions, all of which must hold (maintainer decision 2026-09-28):

    derived hard ceiling                        <= per-session envelope
    project cumulative spend + derived ceiling  <= project cap
    remaining formal allowance                  >= derived ceiling

The component minutes live in `c3_pricing_9probe.json` and are read, never
restated — that document is where each one records which session measured it.
Container disk is priced separately because the provider bills it separately:
a GPU-hour rate is not the bill.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
PRICING_PATH = REPO / "logs/stages/stage-1/phase_c3/plans/c3_pricing_9probe.json"

#: RunPod bills container disk separately from the GPU, at $0.10/GB/month.
#: Omitting it understated an earlier estimate by $0.33 on the hard bound.
DISK_USD_PER_GB_MONTH = 0.10
HOURS_PER_MONTH = 30 * 24


class C3PricingError(RuntimeError):
    """The pricing inputs are missing, malformed or not fundable as read."""


def _pricing_doc() -> dict[str, Any]:
    if not PRICING_PATH.is_file():
        raise C3PricingError(f"no C3 pricing table at {PRICING_PATH}")
    return json.loads(PRICING_PATH.read_text())


def component_minutes() -> dict[str, float]:
    """The measured component table. One owner, read not restated."""
    return {k: float(v["minutes"])
            for k, v in _pricing_doc()["components"].items()}


def allowances() -> dict[str, float]:
    """The hard-ceiling allowances: worst-case setup, overrun factor."""
    a = _pricing_doc()["hard_ceiling"]["_allowances"]
    return {"setup_worst_case_minutes": float(a["setup_worst_case_minutes"]),
            "train_and_eval_overrun_factor": float(
                a["train_and_eval_overrun_factor"])}


def container_disk_gb() -> int:
    return int(_pricing_doc()["container_disk_gb"])


def billed_rate(gpu_rate_usd_per_hour: float, disk_gb: int | None = None) -> float:
    """GPU price plus container disk, per hour. The GPU price is not the bill."""
    gb = container_disk_gb() if disk_gb is None else disk_gb
    return gpu_rate_usd_per_hour + gb * DISK_USD_PER_GB_MONTH / HOURS_PER_MONTH


def _minutes(worst: bool) -> float:
    comps = component_minutes()
    a = allowances()
    total = 0.0
    for name, minutes in comps.items():
        m = minutes
        if worst and name == "setup":
            m = a["setup_worst_case_minutes"]
        elif worst and name.startswith(("recovery_", "evaluation_")):
            m *= a["train_and_eval_overrun_factor"]
        total += m
    return total


@dataclass(frozen=True)
class C3Price:
    gpu_rate_usd_per_hour: float
    billed_rate_usd_per_hour: float
    container_disk_gb: int
    expected_minutes: float
    expected_usd: float
    hard_minutes: float
    hard_usd: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "gpu_rate_usd_per_hour": self.gpu_rate_usd_per_hour,
            "billed_rate_usd_per_hour": round(self.billed_rate_usd_per_hour, 6),
            "container_disk_gb": self.container_disk_gb,
            "expected": {"minutes": round(self.expected_minutes, 2),
                         "hours": round(self.expected_minutes / 60.0, 4),
                         "usd": self.expected_usd},
            "hard_ceiling": {"minutes": round(self.hard_minutes, 2),
                             "hours": round(self.hard_minutes / 60.0, 4),
                             "usd": self.hard_usd},
            "_disk_is_billed_separately": (
                f"{self.container_disk_gb} GB x ${DISK_USD_PER_GB_MONTH}"
                f"/GB/month = ${self.billed_rate_usd_per_hour - self.gpu_rate_usd_per_hour:.6f}/h"),
        }


def price_c3(gpu_rate_usd_per_hour: float) -> C3Price:
    """Expected and hard-ceiling cost at a given live rate. Pure."""
    if not isinstance(gpu_rate_usd_per_hour, (int, float)) or \
            isinstance(gpu_rate_usd_per_hour, bool):
        raise C3PricingError(
            f"rate must be a number, got {type(gpu_rate_usd_per_hour).__name__}")
    rate = float(gpu_rate_usd_per_hour)
    if not (0.0 < rate < 100.0):
        raise C3PricingError(f"implausible live rate ${rate}/h; refusing to price")
    billed = billed_rate(rate)
    exp_m, hard_m = _minutes(False), _minutes(True)
    return C3Price(
        gpu_rate_usd_per_hour=rate,
        billed_rate_usd_per_hour=billed,
        container_disk_gb=container_disk_gb(),
        expected_minutes=exp_m, expected_usd=round(exp_m / 60.0 * billed, 4),
        hard_minutes=hard_m, hard_usd=round(hard_m / 60.0 * billed, 4))


@dataclass(frozen=True)
class Fundability:
    price: C3Price
    per_session_envelope_usd: float
    project_cap_usd: float
    cumulative_spend_usd: float
    formal_remaining_usd: float

    @property
    def fits_envelope(self) -> bool:
        return self.price.hard_usd <= self.per_session_envelope_usd

    @property
    def fits_project_cap(self) -> bool:
        return round(self.cumulative_spend_usd + self.price.hard_usd, 4) \
            <= self.project_cap_usd

    @property
    def fits_formal_allowance(self) -> bool:
        return self.formal_remaining_usd >= self.price.hard_usd

    @property
    def fundable(self) -> bool:
        return (self.fits_envelope and self.fits_project_cap
                and self.fits_formal_allowance)

    def shortfalls(self) -> dict[str, float]:
        """Exactly how far short each failing condition is. Empty when fundable."""
        out: dict[str, float] = {}
        if not self.fits_envelope:
            out["per_session_envelope"] = round(
                self.price.hard_usd - self.per_session_envelope_usd, 4)
        if not self.fits_project_cap:
            out["project_cap"] = round(
                self.cumulative_spend_usd + self.price.hard_usd
                - self.project_cap_usd, 4)
        if not self.fits_formal_allowance:
            out["formal_allowance"] = round(
                self.price.hard_usd - self.formal_remaining_usd, 4)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "price": self.price.as_dict(),
            "envelopes": {
                "per_session_envelope_usd": self.per_session_envelope_usd,
                "project_cap_usd": self.project_cap_usd,
                "cumulative_spend_usd": self.cumulative_spend_usd,
                "formal_remaining_usd": self.formal_remaining_usd},
            "conditions": {
                "derived_ceiling_within_envelope": self.fits_envelope,
                "cumulative_plus_derived_within_cap": self.fits_project_cap,
                "formal_allowance_covers_derived": self.fits_formal_allowance},
            "FUNDABLE": self.fundable,
            "shortfalls_usd": self.shortfalls(),
            "_the_envelope_is_not_the_grant": (
                "the authorization receives hard_ceiling.usd derived above, "
                "never the envelope"),
        }


def live_envelopes() -> dict[str, float]:
    """The amended envelopes and the live spend, from their canonical owners."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/consolidate/derive_budget.py"), "--json"],
        capture_output=True, text=True, cwd=str(REPO))
    if out.returncode != 0:
        raise C3PricingError(f"derive_budget failed: {out.stderr[-400:]}")
    d = json.loads(out.stdout)
    return {
        "per_session_envelope_usd": float(d["per_session_ceiling_usd"]),
        "project_cap_usd": float(d["project"]["cap_usd"]),
        "cumulative_spend_usd": float(d["project"]["cumulative_spend_usd"]),
        "formal_remaining_usd": float(d["formal"]["remaining_usd"]),
    }


def assess(gpu_rate_usd_per_hour: float,
           envelopes: dict[str, float] | None = None) -> Fundability:
    env = envelopes if envelopes is not None else live_envelopes()
    return Fundability(price=price_c3(gpu_rate_usd_per_hour), **env)


# ---------------------------------------------------------------------------
# Live rate
# ---------------------------------------------------------------------------

RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"


def query_live_secure_price(gpu_type_id: str = "NVIDIA L40S",
                            api_key: str | None = None,
                            timeout: float = 30.0) -> float:
    """Live `securePrice` for one GPU type. NEVER `communityPrice`.

    Reporting `communityPrice` once produced a number the launcher could not
    act on, because the launcher prices on `securePrice`: read the field the
    consumer reads.
    """
    key = api_key or os.environ.get("RUNPOD_API_KEY")
    if not key:
        #: The repository's own convention: the RunPod CLI config. Its values
        #: are single-quoted, and stripping only `"` yields a key with a
        #: leading quote that GraphQL answers with `{"error":{}}` -- a parsing
        #: failure that reads like an auth failure. `read_api_key` handles it.
        import sys as _sys
        _sys.path.insert(0, str(REPO / "src"))
        from aadistill.infrastructure.provider import read_api_key

        cfg = os.path.expanduser("~/.runpod/config.toml")
        if not os.path.isfile(cfg):
            raise C3PricingError(
                f"no RUNPOD_API_KEY in the environment and no {cfg}; "
                "cannot re-price live")
        key = read_api_key(cfg)
    query = ('query { gpuTypes(input:{id:"%s"}) { id securePrice '
             'communityPrice } }' % gpu_type_id)
    #: RunPod's edge answers the default `Python-urllib/3.x` User-Agent with
    #: **403 Forbidden** on every query, including ones that succeed
    #: byte-for-byte from curl. The launchers never hit it because they shell
    #: out to curl; `provider.USER_AGENT` exists for exactly this.
    import sys as _sys
    _sys.path.insert(0, str(REPO / "src"))
    from aadistill.infrastructure.provider import USER_AGENT

    req = urllib.request.Request(
        f"{RUNPOD_GRAPHQL}?api_key={key}",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            doc = json.loads(fh.read().decode())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise C3PricingError(f"live price query failed: {exc}") from exc
    rows = ((doc.get("data") or {}).get("gpuTypes") or [])
    if not rows:
        raise C3PricingError(f"provider returned no gpuTypes for {gpu_type_id!r}")
    price = rows[0].get("securePrice")
    if price is None:
        raise C3PricingError(
            f"{gpu_type_id} has no securePrice (communityPrice "
            f"{rows[0].get('communityPrice')}); the launcher prices on "
            f"securePrice and must not fall back")
    return float(price)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rate", type=float, default=None,
                    help="GPU $/h; omit to query the provider live")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    rate = args.rate if args.rate is not None else query_live_secure_price(args.gpu)
    f = assess(rate)
    if args.json:
        print(json.dumps(f.as_dict(), indent=1, sort_keys=True))
        return 0 if f.fundable else 1

    p = f.price
    print(f"live {args.gpu} securePrice  ${p.gpu_rate_usd_per_hour:.4f}/h")
    print(f"  + {p.container_disk_gb} GB container disk "
          f"-> billed ${p.billed_rate_usd_per_hour:.6f}/h")
    print(f"  expected  {p.expected_minutes:8.2f} min = "
          f"{p.expected_minutes/60:6.2f} h  ${p.expected_usd:8.4f}")
    print(f"  HARD      {p.hard_minutes:8.2f} min = "
          f"{p.hard_minutes/60:6.2f} h  ${p.hard_usd:8.4f}")
    print()
    for label, ok in (("derived <= envelope "
                       f"(${f.per_session_envelope_usd:.4f})", f.fits_envelope),
                      (f"cumulative + derived <= cap (${f.project_cap_usd:.2f})",
                       f.fits_project_cap),
                      (f"formal remaining >= derived "
                       f"(${f.formal_remaining_usd:.4f})",
                       f.fits_formal_allowance)):
        print(f"  [{'OK ' if ok else 'NO '}] {label}")
    print(f"\n  FUNDABLE: {f.fundable}")
    if not f.fundable:
        for k, v in f.shortfalls().items():
            print(f"    short on {k}: ${v:.4f}")
    return 0 if f.fundable else 1


if __name__ == "__main__":
    raise SystemExit(main())
