# Stage-3 continuation — the attempts that did not complete

Eight attempts, **$4.1060**. Attempt 8 completed and its products are in
[`../autoinit_stage3_complete/`](../autoinit_stage3_complete/). This directory
holds the ones that failed, because a failed run has to stay useful (AGENTS.md
P11).

| # | cost | died in | cause |
| --- | ---: | --- | --- |
| 1 | $0.6312 | setup | cold host, and a test gate reading a battery that was never staged |
| 2 | $0.6367 | setup | three consecutive cold hosts, all inside the `uv sync` window |
| 3 | $0.0700 | setup | `uv sync --frozen` cannot install a registry-pinned wheel offline |
| 4 | $1.3672 | setup | unpinned `pip install vllm` hung 76 min on the paid critical path |
| [5](attempt5/) | $0.1369 | setup, last line | asserted an unrelated session's authorization, whose plan hash had moved |
| [6](attempt6/) | $0.1324 | launcher readback | setup succeeded (`SETUP_RC=0`); markers went to a filename the launcher does not probe |
| [7](attempt7/) | $0.4500 | Stage 3 | `sb`'s package had no tokenizer or chat template, so no prompt could render |
| 8 | $0.6816 | — | **completed**; see `../autoinit_stage3_complete/` |

Attempts 1–4 have no directory here: they produced no driver artifacts, and their
records live in [`../BUDGET_LEDGER.md`](../BUDGET_LEDGER.md) and
[`../decisions.md`](../decisions.md).

**What the sequence cost and what it bought.** Four of the seven failures were
lines in `scripts/pod/autoinit_preflight_setup.sh` that no rehearsal had ever
executed, each found by paying for the next one. That is what
`tests/pod/test_setup_end_to_end.py` now prevents: it runs the real script end to
end under bubblewrap and reads the result back through the launcher's own probe.
Attempt 8's setup worked first time.

Attempt 7's `sa` result is valid partial evidence and is **not** combined with
attempt 8's `sb`: attempt 8 re-ran both controls on the complete path.
