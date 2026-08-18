"""The frozen science inputs a pod session stages from the relay.

These lived inside `autoinit_preflight_setup.sh` until 2026-08-18: three relay
prefixes, ten filenames, four sha256 pins and a probe-to-ladder copy, written
into the shared shell and executed unconditionally for every session. The
sessions' own `relay_inputs` declarations named at most three of the ten files,
so the declaration was an existence assertion for the $0 precheck and the
staging was hidden — the relay-side twin of the `LOCAL_ASSETS` defect that cost
the device-canary retry $0.0637.

They are here, and not in `src/aadistill/`, because `docs/REPO_LAYOUT.md` rule 1
is that the algorithm core holds no model-recipe constants: teacher ids, target
geometry and frozen hashes live in the scripts and configs that own them.

A session composes what it needs by naming these groups, so "the micro-preflight
stages the calibration mixture" is a line a reader can see rather than a line
buried in a shell heredoc. Nothing here is inherited: a session that names no
group stages nothing.

**These are frozen identities.** The four digests are the ones the shared setup
verified before this file existed, unchanged. `CALIBRATION_V1` restates the file
hash that `aadistill.autoinit.datasets.E8A_CALIBRATION` already carries, and
`tests/pod/test_session_architecture.py` pins the two equal so they cannot
drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.session import RelayInput  # noqa: E402

#: Where the pack is read from. `scripts/pod/p2_driver.py` reads
#: `ladder_uniform_probe`; the recovery corpus loader reads `ladder_uniform`.
#: The shell satisfied both with an undeclared directory walk after the fetch.
_LADDER_PROBE = "artifacts/stage3/ladder_uniform_probe"
_LADDER_MIRROR = "artifacts/stage3/ladder_uniform"

#: The canonical Stage-1 student. `model.safetensors` is the pinned one; the
#: other five are the companion files a checkpoint is unloadable without — and
#: which no session declared, on the reasoning that the weights were the
#: artifact. A control that shipped without its tokenizer has already been
#: written up once (`logs/autoinit_control_sb_packaging_repair.json`).
CANONICAL_INIT: tuple[RelayInput, ...] = tuple(
    RelayInput(f"stage1/qwen3_0p6b_init_v0/checkpoint/{name}",
               dest="artifacts/stage1/qwen3_0p6b_init_v0/checkpoint",
               sha256=sha)
    for name, sha in (
        ("model.safetensors",
         "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"),
        ("config.json", None),
        ("generation_config.json", None),
        ("tokenizer.json", None),
        ("tokenizer_config.json", None),
        ("chat_template.jinja", None),
    )
)

#: The Stage-3 recovery pack, staged under both names it is read by.
RECOVERY_LADDER: tuple[RelayInput, ...] = tuple(
    RelayInput(f"stage3_recovery_corpus_v2/ladder_uniform/{name}",
               dest=_LADDER_PROBE, also_stage_to=_LADDER_MIRROR, sha256=sha)
    for name, sha in (
        ("blocks.npz",
         "6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c"),
        ("ladder.json", None),
        ("audit.jsonl", None),
    )
)

#: The operator calibration mixture. Phase-A stage 1 calls
#: `DOMAIN_BALANCED_V1.resolve()`, which reads this exact file and verifies both
#: its byte hash and its derived token-content hash. Attempt 5 died on it at
#: $0.6426 because nothing staged it — and after the fix that staged it, three
#: of the four sessions still did not declare it.
CALIBRATION_V1: tuple[RelayInput, ...] = (
    RelayInput("e8_inputs_20260810/calibration_v1/items.jsonl",
               dest="artifacts/stage1/e8_calibration_v1",
               sha256="c7202338109e459b17b70456461e8f304fadea"
                      "7929ea547accee21adbbe7fd0b"),
)
