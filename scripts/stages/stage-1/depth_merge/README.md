# depth_merge (Stage 1): DMR1 variant 1, and the round's shared chain

DMR1 variant 1 (`depth_merge`: the alpha-merged initialization at SOB1's depth structure, then SOB1's depth recovery)
and the execution chain all three DMR1 variants share. Design: `configs/stages/stage-1/depth_merge/design.json`;
evidence: `logs/stages/stage-1/depth_merge/`; the round's review report:
`docs/stages/stage-1/reasoning_inheritance/DMR1_REVIEW.md`. Status: PREPARED at $0, READY_FOR_REVIEW (2026-11-26);
nothing granted, issued or run.

| file | role |
| --- | --- |
| `common.py` | the round's designs, arms and the binding of SOB1's chain to an arm's design (`bind_for`) |
| `merge.py` | the alpha merge's instance: spans from SOB1's mapping, the base digests, the calibration call, per-pair persisted alignments (RI-LM1's M-act code by identity) |
| `build_driver.py` | pod driver of `dmr1_merge_build`: the init from the teacher, durable with its statistics |
| `plan.py` | any DMR1 arm's two units from SOB1's committed depth plan, every other field compared |
| `train_driver.py` | pod driver of every DMR1 training kind: SOB1's `run_unit` bound to the arm's design; a recurrent arm's init staged with its config; the vLLM plugin on the runners' path |
| `session.py` | the round's launcher machinery (one `Variant` per study): gates, issue, launch spec, closeout completion |
| `launch.py` | variant 1's launcher (kinds `dmr1_merge_build`, `dmr1_merge_train`) |
| `analysis.py` | the registered comparisons C1-C3 from the sealed sets, and the Holm sensitivity line |
| `proposal.py` | variant 1's two proposals and variants 2 and 3's estimates |
| `tests/` | `pytest scripts/stages/stage-1/depth_merge/tests` |

Every command runs as `PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/<file>`.
