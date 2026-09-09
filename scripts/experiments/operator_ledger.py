"""Where this project keeps the operator immutability ledger.

The application layer. `aadistill.initialization.operators.base` owns what a
ledger IS and how it is verified; it used to also name
`configs/autoinit/operator_ledger.json`, which made a reusable core depend on
this repository's directory layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.operators.base import (  # noqa: E402
    verify_ledger as _verify, write_ledger as _write)

#: Repository-relative, so it reads the same on a pod checkout.
LEDGER_PATH = "configs/autoinit/operator_ledger.json"


def verify_ledger(path: str | Path = LEDGER_PATH, **kw):
    """Verify this project's operator ledger."""
    return _verify(path, **kw)


def write_ledger(path: str | Path = LEDGER_PATH, **kw) -> Path:
    """Re-emit this project's operator ledger."""
    return _write(path, **kw)


__all__ = ["LEDGER_PATH", "verify_ledger", "write_ledger"]
