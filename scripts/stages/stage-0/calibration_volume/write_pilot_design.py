#!/usr/bin/env python3
"""Write the root-level selection-stability pilot's design, from the built pool.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/write_pilot_design.py \
        --pool artifacts/stages/stage-0/calibration_volume/base_pool_v1

Every choice is stated here once, with its reason, BEFORE any pilot result
exists; the design file's hash is what the profiles document and the plan bind.

* Base mixture (UESP weights): the four static P10.2 primary axes carry 0.72
  (0.18 each for induction, composition, deduction and analogy) and supporting
  knowledge 0.28 (verified corpus_v2 math, multihop and RAG), so reasoning is
  primary without starving its supports.
* The INTERACTIVE axis is excluded from this pilot, and the exclusion is a
  recorded gap, not a weighting choice: base_pool_v1 holds 13 admissible
  successful teacher episodes (17 of 72 sampled calibration episodes
  succeeded; 4 more exceed the length bound), each ~16k scored positions over
  its turns. One episode would overshoot a 1x interactive cell quota several
  times over, and 13 episodes cannot give three independent resamples.
* Cells: every calibration-eligible generator (families pooled within it) and
  every supporting subtype. The number of distinct rule families per profile
  is recorded as a covariate of the dose.
* Doses: 1x, 2x, 4x of 45,000 UESP, the forward baseline's calibration scale
  (calib.domain_balanced@v1: 44,746). With teacher-native units of several
  thousand scored positions each, 1x already holds about one unit per cell; a
  smaller dose would drop cells rather than shrink them.
* Resamples: three sampling seeds per dose, derived from a fixed string.
* DEPTH (token-linear, ~20 min per 60k tokens) runs on resamples 1-2 at 1x and
  2x only; FFN, WIDTH and ATTENTION (about a minute each) run on all nine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "configs/stages/stage-0/calibration_volume/pilot_design.json"

WEIGHTS = {"inductive_rule_discovery": 0.18, "compositional_generalization": 0.18,
           "task_contained_deduction": 0.18, "abstract_analogical": 0.18,
           "supporting": 0.28}
EXCLUDED_DOMAINS = {"interactive_adaptation": (
    "13 admissible successful teacher episodes in base_pool_v1, ~16k scored positions "
    "each: too few for independent resampling and each larger than a 1x cell quota")}
DOSES = (1, 2, 4)
RESAMPLES = (1, 2, 3)
DEPTH_RESAMPLES, DEPTH_DOSES = (1, 2), (1, 2)
REFERENCE_UESP = 45000


def seed_for(k: int) -> int:
    h = hashlib.sha256(f"stage0.calibration_volume.pilot_v1|resample|{k}".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2 ** 31)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    args = ap.parse_args(argv)
    pool = REPO / args.pool
    man = json.loads((pool / "manifest.json").read_text())
    cells: dict[str, dict[str, list[str]]] = {}
    for key in man["by_stratum"]:
        domain, subtype = key.split("|", 1)
        if domain in EXCLUDED_DOMAINS:
            continue
        cells.setdefault(domain, {})[subtype] = ["*"]
    if set(cells) != set(WEIGHTS):
        raise SystemExit(f"pool domains {sorted(cells)} != weighted {sorted(WEIGHTS)}")
    treatments = {f"d{d}_r{r}": {"dose": d, "sampling_seed": seed_for(r), "resample": r}
                  for d in DOSES for r in RESAMPLES}
    design = {
        "schema": "aadistill.stage0.calibration_volume.design/v1",
        "name": "pilot_v1",
        "question": ("At a fixed parent (the teacher), how much does each forward-baseline "
                     "operator's selection vary across independent resamples of the Stage-0 "
                     "base pool, and does that variation shrink from 1x to 4x calibration "
                     "volume? Exploratory H1/H3 evidence; not the formal full search."),
        "pool": {"dir": args.pool, "items_sha256": man["items_sha256"],
                 "content_sha256": man["content_sha256"]},
        "reference_uesp": REFERENCE_UESP,
        "sample_rule": ("stratified nested sampling: generator/subtype cells, seed-keyed "
                        "per-cell shuffles, whole units, quota = dose x reference x domain "
                        "weight split equally over the domain's cells (build_volume_profiles.py)"),
        "profiles": {"base": {"domain_weights": WEIGHTS}},
        "excluded_domains": EXCLUDED_DOMAINS,
        "cells": cells,
        "treatments": treatments,
        "seed_derivation": "uint32_be(sha256('stage0.calibration_volume.pilot_v1|resample|k')[:4]) mod 2**31",
        "search_treatments": {"pilot_root": sorted(treatments)},
        "depth_profiles_rule": {"resamples": list(DEPTH_RESAMPLES), "doses": list(DEPTH_DOSES)},
        "leakage_exclusions": ["dev_v1 and final evaluation families (generator family split)",
                               "recovery rung 0.86M sessions", "state_eval@v1 sessions",
                               "d_series_behavioural_v1 prompts"],
        "output": {"items_dir": "artifacts/stages/stage-0/calibration_volume/profiles_pilot_v1",
                   "profiles_doc": "configs/stages/stage-0/calibration_volume/profiles.json"},
        "authorizes": "nothing",
    }
    OUT.write_text(json.dumps(design, indent=1) + "\n")
    print(OUT.relative_to(REPO), {k: v["sampling_seed"] for k, v in treatments.items() if k.startswith("d1")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
