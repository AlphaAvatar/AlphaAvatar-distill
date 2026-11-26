"""The hardware profiles historical plans were priced with (`configs/shared/infrastructure/hardware_profiles.json`),
as `aadistill.runtime.cost.HardwareProfile` objects for the application layer. The core holds the type and the
formulas only; these instances, their prices, throughputs and statistics rates live in the record with their
sources. `L40S_MEASURED`, `A100_80GB_ESTIMATED` and `DEVBOX_CPU_MEASURED` keep the names the historical planners
import; a new plan builds its own profile from the live rate (`profile(name, price_per_hour_usd=...)`)."""
from __future__ import annotations

import json
from pathlib import Path

from aadistill.runtime.cost import HardwareProfile

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "configs/shared/infrastructure/hardware_profiles.json"


def profile(name: str, **override) -> HardwareProfile:
    doc = json.loads(RECORD.read_text())["profiles"]
    if name not in doc:
        raise KeyError(f"no hardware profile {name!r} in {RECORD.relative_to(REPO)}; known: {sorted(doc)}")
    r = dict(doc[name])
    rate = r.pop("stats_seconds_per_token_upper", None)
    fields = {k: r[k] for k in ("name", "price_per_hour_usd", "effective_tflops", "vram_gb", "measured", "source")}
    if rate is not None:
        fields["stats_seconds_per_token_upper"] = float(rate["numerator_seconds"]) / float(rate["denominator_tokens"])
        fields["stats_rate_source"] = rate["source"]
    return HardwareProfile(**{**fields, **override})


L40S_MEASURED = profile("L40S_MEASURED")
A100_80GB_ESTIMATED = profile("A100_80GB_ESTIMATED")
DEVBOX_CPU_MEASURED = profile("DEVBOX_CPU_MEASURED")
#: the dev-box CPU statistics rate the historical plans used (seconds per token), from the record
CPU_STATS_SECONDS_PER_TOKEN = L40S_MEASURED.stats_seconds_per_token_upper
