"""Where the wall clock went. Operational only — never scientific identity.

Phase-B attempt 3 spent 544.7 min in Stage 1 and stopped on its deadline. What
was known afterwards: `depth.causal_kl_greedy_v1` ran 12 expansions totalling
388.2 min. What was *not* known, and could not be recovered because the run log
is the only surviving record: how much of that was reference forwards versus
ablated forwards, how much went to materialize/reload/measure, how many
activation-statistics passes were collected twice, and how much of it the GPU
spent idle. Repricing a nine-hour search from "DEPTH was 71% of it" is guesswork.

This module fixes that at `$0`. It records timings per expansion and per
operator-internal phase, appends them to a JSONL stream beside the search
journal, and stops there.

**The boundary that must not be crossed.** A timing is a property of the machine
and the moment, not of the initialization. `state_id`, `config_hash`,
`profile_hash`, `spec_hash` and every operator metric must be byte-identical
whether this module is recording or absent — otherwise the same search on a
faster host would produce different states, and every recorded result would be
unreproducible. So nothing here is ever returned into a state, a trace, a metric
or a hash input: the sink writes to its own file, and `TelemetrySink.disabled()`
is a first-class mode that changes nothing but the absence of a file.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class _Accumulator:
    """Summed elapsed seconds and call count for one named phase."""

    seconds: float = 0.0
    calls: int = 0

    def add(self, seconds: float) -> None:
        self.seconds += seconds
        self.calls += 1

    def as_dict(self) -> dict[str, Any]:
        return {"seconds": round(self.seconds, 4), "calls": self.calls}


@dataclass
class TelemetrySink:
    """Append-only operational record. Safe to call from anywhere, including
    inside an operator's innermost loop.

    Every method is failure-tolerant on purpose: telemetry that can raise turns a
    diagnostic into an outage, and this project has already lost a paid pod to a
    metadata collector that could throw.
    """

    path: Path | None = None
    _events: list[dict[str, Any]] = field(default_factory=list)
    _phases: dict[str, _Accumulator] = field(default_factory=dict)

    @classmethod
    def disabled(cls) -> "TelemetrySink":
        """A sink that records in memory and writes nothing."""
        return cls(path=None)

    # --- recording ---------------------------------------------------------

    def record(self, event: str, **fields: Any) -> None:
        entry = {"event": event, **fields}
        self._events.append(entry)
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except Exception:                          # noqa: BLE001 - never fatal
            pass

    def add(self, phase: str, seconds: float) -> None:
        """Accumulate into a named phase without emitting an event.

        For the inner loops — a per-candidate forward pass emits nothing, because
        260 candidates x 67 items would drown the stream it is meant to explain.
        """
        self._phases.setdefault(phase, _Accumulator()).add(seconds)

    @contextmanager
    def timed(self, phase: str):
        """Accumulate the elapsed time of a block into `phase`."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(phase, time.perf_counter() - started)

    @contextmanager
    def span(self, event: str, **fields: Any):
        """Time a block and emit ONE event carrying its duration."""
        started = time.perf_counter()
        extra: dict[str, Any] = {}
        try:
            yield extra
        finally:
            self.record(event, seconds=round(time.perf_counter() - started, 4),
                        **fields, **extra)

    # --- reading -----------------------------------------------------------

    def phases(self) -> dict[str, dict[str, Any]]:
        return {k: v.as_dict() for k, v in sorted(self._phases.items())}

    def drain_phases(self) -> dict[str, dict[str, Any]]:
        """Read the accumulators and reset them, for per-expansion reporting."""
        out = self.phases()
        self._phases.clear()
        return out

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def summary(self) -> dict[str, Any]:
        """Aggregate by event type. Diagnostic only."""
        by_event: dict[str, _Accumulator] = {}
        for e in self._events:
            seconds = e.get("seconds")
            if isinstance(seconds, (int, float)):
                by_event.setdefault(e["event"], _Accumulator()).add(float(seconds))
        return {"events": len(self._events),
                "by_event": {k: v.as_dict() for k, v in sorted(by_event.items())},
                "open_phases": self.phases()}


def stopwatch() -> Any:
    """A monotonic elapsed-seconds callable, for hot paths that cannot afford a
    context manager's bookkeeping."""
    started = time.perf_counter()
    return lambda: time.perf_counter() - started
