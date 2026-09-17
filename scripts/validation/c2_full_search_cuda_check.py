#!/usr/bin/env python3
"""Real-CUDA engineering check of the Phase-C2 full-joint-search DRIVER.

    PYTHONPATH=src:scripts python scripts/validation/c2_full_search_cuda_check.py \
        --config configs/validation/c2_full_search_cuda.json --run-id <id>

**What is already verified, and therefore not re-run here.** All six
implementations the joint space searches have real-L40S evidence from paid
sessions -- 14/2/19/6/29/38 expansions across Phase-B attempt 5 and C2
attempt 4 -- each a real plan/apply/materialize/canonical-reload/identity/
state_eval against the real 4B teacher. Phase B also ran with
`impl_profiles=None` over two mixtures, which is the joint search's own unpinned
branching. Re-running those at real scale would cost hours and answer a question
already answered.

**What this checks.** The DRIVER, which has never run on a GPU at all. Its stage
sequencing, its `afford` check, the arguments it forwards to
`run_phase_a_search`, its selection parsing and its path handling are exercised
only by a CPU toy execution -- and that execution passes `device="cpu"`, which
is precisely the substitution that has already hidden a device-placement bug in
this repository. So:

    D  driver      the REAL driver, all three stages, over the REAL derived
                   joint space, at toy geometry, with `--device cuda` PASSED
                   THROUGH UNCHANGED and bf16. Every materialized state's
                   parameter device and dtype is asserted.
    M  geometry    one real-target-geometry (1024x28) student materialization,
                   canonical `from_pretrained` reload, rope-base read and
                   identity check, in bf16 on CUDA.

Stage M needs the teacher's CONFIG at the pinned revision and NOT its weights,
which is what keeps this cheap: a 4B download would dominate a validation whose
question is whether a 596M student materializes and reloads on a real device.

**Assertions are about PLACEMENT and DTYPE, not arithmetic.** CPU and CUDA do
not agree bit-for-bit and are not required to; demanding numerical equality
would fail for a reason that is not a defect. What a GPU can prove and a CPU
cannot is that tensors are on the device the caller asked for, in the dtype the
caller asked for, and that no operator silently relocates them.

**No CPU fallback.** With CUDA absent this reports NOT RUN and exits 3. A check
that quietly ran on CPU would report a pass for the one thing it cannot see.

AUTHORIZES NOTHING. It trains nothing, measures no behaviour, produces no
`correct_overall` and ranks no candidate for promotion. Its verdict is
engineering evidence for a later launch review.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("AAD_REPO", "/workspace/aad"))
if not (REPO / "src").is_dir():                     # local / toy execution
    REPO = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

#: Registered AT MODULE IMPORT, not inside whichever stage happens to need one.
#: Both registries are explicit by design, and the $0 execution of this file
#: found the failure that makes the placement matter: stage D reached its
#: adapter through the driver's own import chain and passed, while stage M --
#: which resolves the same adapter directly -- died in `get_adapter` before it
#: built anything. With `stages.driver.enabled: false` that is the whole check
#: gone, and the reason would look like a missing model rather than a missing
#: call. This project has already lost a paid pod to an empty registry.
from aadistill.initialization.adapters import register_builtin_adapters  # noqa: E402
from experiments.calibration import register_builtin_profiles  # noqa: E402
from experiments.phase_c2 import full_search_space as _FS  # noqa: E402

register_builtin_adapters()
register_builtin_profiles()
#: The promoted ATTENTION operator is deliberately not a shipped default, so the
#: joint space cannot even be enumerated without this.
_FS.register_c2_operators()

#: Exit codes. 3 is NOT RUN, which the launcher maps to a NOT RUN verdict rather
#: than a failure: an environment that cannot host the check has not failed it.
OK, FAILED, NOT_RUN = 0, 1, 3


def say(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}",
          flush=True)


def require_cuda(requested: str) -> dict:
    """CUDA or nothing. Reports what it found either way."""
    import torch

    if requested != "cuda":
        raise SystemExit(f"this check is only meaningful on cuda, got {requested!r}")
    if not torch.cuda.is_available():
        say("NOT RUN: torch reports no CUDA device")
        raise SystemExit(NOT_RUN)
    index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    free, total = torch.cuda.mem_get_info(index)
    info = {
        "device": torch.cuda.get_device_name(index),
        "capability": f"{major}.{minor}",
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "free_gib": round(free / 1024 ** 3, 2),
        "total_gib": round(total / 1024 ** 3, 2),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    say(f"CUDA {info['device']} cc{info['capability']} "
        f"bf16={info['bf16_supported']} free={info['free_gib']}GiB "
        f"torch={info['torch']} cuda={info['cuda_runtime']}")
    return info


def require_capability(info: dict, need: dict) -> None:
    """The authorization's capability floor, enforced before anything runs.

    The dtype and the workload are NOT lowered to make an unsupported device
    pass -- that would validate a configuration the formal search will not use.
    """
    have = tuple(int(x) for x in info["capability"].split("."))
    want = tuple(int(x) for x in str(need["compute_capability_min"]).split("."))
    if have < want:
        raise AssertionError(
            f"compute capability {info['capability']} is below the required "
            f"{need['compute_capability_min']}")
    if need.get("native_bf16") and not info["bf16_supported"]:
        raise AssertionError("native bf16 is required and unsupported here")
    if info["free_gib"] < float(need.get("free_vram_gib_min", 0)):
        raise AssertionError(
            f"{info['free_gib']} GiB free is below the required "
            f"{need['free_vram_gib_min']} GiB")
    say("capability floor satisfied")


# --------------------------------------------------------------------------
# stage D: the driver, on a real device
# --------------------------------------------------------------------------

def placement_of(model) -> dict:
    """Every distinct (device type, dtype) among a model's parameters.

    Distinct SETS, not a spot check on one tensor: a partially-relocated model
    is the defect this looks for, and reading `next(model.parameters())` would
    miss exactly that.
    """
    devices, dtypes = set(), set()
    for p in model.parameters():
        devices.add(p.device.type)
        dtypes.add(str(p.dtype))
    for b in model.buffers():
        devices.add(b.device.type)
    return {"devices": sorted(devices), "dtypes": sorted(dtypes)}


def toy_world(cfg: dict, device: str, dtype):
    """A toy teacher, target geometry and suite, built on the real device."""
    import torch
    from aadistill.initialization.specs.arch import ArchSpec, get_adapter
    from aadistill.initialization.specs.metrics import StateEvalSuite, SuiteItem
    from transformers import Qwen3Config, Qwen3ForCausalLM

    m = cfg["toy"]
    teacher_geometry = dict(m["teacher"])
    target_geometry = dict(m["target"])
    vocab = teacher_geometry["vocab_size"]

    torch.manual_seed(cfg["seed"])
    teacher = Qwen3ForCausalLM(Qwen3Config(
        max_position_embeddings=m["max_position_embeddings"],
        rope_theta=m["rope_theta"], **teacher_geometry))
    #: ON THE DEVICE, in the dtype under test. The CPU execution of this same
    #: driver builds the teacher with `.float()` on the host, so this line is
    #: the first time the search sees a teacher that is not host-resident.
    teacher = teacher.to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        for module in teacher.modules():
            if module.__class__.__name__ == "Qwen3RMSNorm":
                module.weight.uniform_(0.5, 1.5)

    def items(seed, n=None, seq_len=None):
        n = m["items_per_profile"] if n is None else n
        seq_len = m["seq_len"] if seq_len is None else seq_len
        torch.manual_seed(seed)
        out = []
        for k in range(n):
            ids = torch.randint(0, vocab, (1, seq_len))
            targets = ids[0, 1:]
            #: The tags must MATCH something, or every state has no value for
            #: `critical_token_kl`, the policy drops them all as ineligible,
            #: and the search ends with no leaf -- which reads exactly like a
            #: broken operator rather than like an empty suite.
            out.append({"item_id": f"text-{k}", "domain": "general",
                        "subtype": "text", "input_ids": ids,
                        "tags": {"eos_like": targets == 0,
                                 "answer_like": targets % 17 == 0}})
        return out

    suite = StateEvalSuite(
        suite_id="c2.full_search.cuda_check", version=1, domains=("general",),
        subtypes={"general": ("text",)},
        critical_tags=("eos_like", "answer_like"),
        description="engineering check; authorizes nothing")
    suite_items = [SuiteItem(item_id=i["item_id"], input_ids=i["input_ids"],
                             domain=i["domain"], subtype=i["subtype"],
                             tags=i["tags"]) for i in items(cfg["seed"] + 1)]

    #: A canonical control on disk, because `run_phase_a_search` measures one.
    adapter = get_adapter(cfg["adapter"])
    control = Path(cfg["work"]) / "control"
    adapter.save(adapter.build_model(
        adapter.build_config(teacher.config,
                             ArchSpec.of(cfg["family"], target_geometry)),
        dtype, cfg["seed"]), str(control))

    return {"teacher": teacher, "target_geometry": target_geometry,
            "suite_bundle": (suite, suite_items, {"teacher_sha256": "0" * 64}),
            "items": items}


def stage_driver(cfg: dict, device: str, dtype_name: str, work: Path) -> dict:
    """Drive the REAL driver, all three stages, on a real CUDA device."""
    import torch
    import autoinit_phase_c2_full_search_driver as D
    import phase_a_search
    from aadistill.initialization.calibration.profiles import get_profile
    from experiments.phase_c2 import full_search_space as FS

    dtype = getattr(torch, dtype_name)
    world = toy_world({**cfg["driver"], "work": str(work),
                       "adapter": cfg["adapter"], "family": cfg["family"],
                       "seed": cfg["seed"]}, device, dtype)

    #: The driver's own workspace, redirected into scratch. Its AUDIT, STATUS
    #: and SEARCH_WORKDIR constants are pod paths; everything else about the
    #: driver -- every stage, every guard, every forwarded argument -- is real.
    D.AUDIT = work / "audit"
    D.SEARCH_WORKDIR = work / "search"
    D.STATUS = work / "status.txt"
    D.STATE_EVAL = work / "unused_state_eval"

    observed: dict = {"forwarded": {}, "placements": [], "state_dirs": 0}
    real_search = phase_a_search.run_phase_a_search

    def wrapped(**kwargs):
        """Substitute the toy MODEL, never the device.

        `run_phase_a_search` accepts the teacher, geometry, suite and hashes
        precisely so it can execute at toy scale, and its docstring says so.
        What is NOT substituted is `device`: the CPU execution of this driver
        overrides it to `"cpu"`, and that override is the entire reason this
        check exists. Asserting it here means a future edit that reintroduced a
        host default would fail rather than pass quietly.
        """
        observed["forwarded"] = {
            "device": kwargs.get("device"),
            "impl_profiles": kwargs.get("impl_profiles"),
            "conditional_candidates": kwargs.get("conditional_candidates"),
            "allowed_impls": list(kwargs.get("allowed_impls") or ()),
            "top_n": kwargs.get("top_n"),
            "run_id": kwargs.get("run_id"),
            "search_minutes": kwargs.get("search_minutes"),
            "profiles": [p.qualified_id for p in (kwargs.get("profiles") or ())],
        }
        assert kwargs.get("device") == device, (
            f"the driver forwarded device={kwargs.get('device')!r}; this check "
            f"is worthless unless it is {device!r}")
        toy_items = {p.qualified_id: [
            {k: v for k, v in item.items() if k != "tags"}
            for item in world["items"](abs(hash(p.qualified_id)) % 10_000)]
            for p in (kwargs.get("profiles") or ())}
        return real_search(**{**kwargs,
                              "repo_root": work,
                              "teacher_id": "toy",
                              "canonical_init": "control",
                              "canonical_sha256": None,
                              "teacher_loader": lambda: world["teacher"],
                              "target_geometry": world["target_geometry"],
                              "suite_bundle": world["suite_bundle"],
                              "calibration_items": toy_items})

    phase_a_search.run_phase_a_search = wrapped
    try:
        argv = ["--authorization-path", str(work / "unused_authorization.json"),
                "--rate", str(cfg["driver"]["rate"]),
                "--authorized-usd", str(cfg["driver"]["authorized_usd"]),
                "--device", device,
                "--top-n", str(cfg["driver"]["top_n"]),
                "--search-minutes", str(cfg["driver"]["search_minutes"]),
                "--search-deadline-minutes",
                str(cfg["driver"]["search_deadline_minutes"]),
                "--soft-stop-usd", str(cfg["driver"]["soft_stop_usd"])]
        parser = D.build_parser()
        args = parser.parse_args(argv)
        #: `--authorization-path` is RECORDED by the driver, not loaded, so a
        #: scratch path is honest here: the authorization is not the subject of
        #: this check. The `afford` guard itself still executes, against the
        #: rate and soft stop passed above.
        started = time.time()
        rc = D.FullSearchDriver(args).run()
        elapsed = round(time.time() - started, 2)
    finally:
        phase_a_search.run_phase_a_search = real_search

    #: Placement is read from what the search actually WROTE: every state
    #: directory is reloaded through the canonical loader and inspected. A
    #: search that produced states on the host would be invisible to any
    #: assertion made only about the teacher.
    from aadistill.initialization.specs.arch import get_adapter
    adapter = get_adapter(cfg["adapter"])
    seen_devices, seen_dtypes = set(), set()
    for cand in sorted((D.SEARCH_WORKDIR).rglob("config.json")):
        root = cand.parent
        try:
            model = adapter.load(str(root), dtype=dtype, device=device)
        except Exception as exc:                                  # noqa: BLE001
            observed["placements"].append(
                {"root": root.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        place = placement_of(model)
        observed["placements"].append({"root": root.name, **place})
        seen_devices.update(place["devices"])
        seen_dtypes.update(place["dtypes"])
        observed["state_dirs"] += 1
        del model
        torch.cuda.empty_cache()

    observed["driver_returncode"] = rc
    observed["elapsed_seconds"] = elapsed
    observed["distinct_devices"] = sorted(seen_devices)
    observed["distinct_dtypes"] = sorted(seen_dtypes)
    #: `total_leaves`, from the nested report. The first version of this
    #: line read a key that does not exist -- in the LAST statement of
    #: stage D, after the whole search had succeeded, which is precisely
    #: the shape of failure that has turned completed paid work into a
    #: failed session in this repository before.
    observed["space_leaves"] = FS.size_report(REPO)["full_joint"]["total_leaves"]
    observed["profile_ids"] = [get_profile(q).qualified_id
                               for q in FS.PROFILE_IDS]

    assert rc == 0, f"the driver returned {rc}"
    assert observed["state_dirs"] > 0, (
        "the search wrote no reloadable state, so nothing was placed anywhere "
        "-- a pass here would be vacuous")
    assert seen_devices == {device}, (
        f"states materialized on {sorted(seen_devices)}, expected [{device!r}]")
    assert seen_dtypes == {f"torch.{dtype_name}"}, (
        f"states carry {sorted(seen_dtypes)}, expected torch.{dtype_name}")
    assert observed["forwarded"]["impl_profiles"] is None, (
        "the joint search must pass impl_profiles=None; a pinned mapping is "
        "Search-1's restriction, not this one's")
    assert observed["forwarded"]["conditional_candidates"] is None, (
        "this session rebuilds no baseline")
    say(f"stage D passed in {elapsed}s — {observed['state_dirs']} state(s), "
        f"all on {device} in {dtype_name}")
    return observed


# --------------------------------------------------------------------------
# stage M: one real-geometry materialization on a real device
# --------------------------------------------------------------------------

def stage_geometry(cfg: dict, device: str, dtype_name: str, work: Path) -> dict:
    """Build, save, reload and identity-check the REAL student geometry.

    The teacher's CONFIG at the pinned revision, never its weights: the
    question is whether a 596M student materializes, saves and reloads on a
    real device in bf16 -- and a 4B download would dominate a validation that
    does not need it.

    `AutoTokenizer.from_pretrained` on a config-only directory returns a
    1-token vocab rather than raising, so identity is asserted on the loaded
    VALUES and not on the call succeeding.
    """
    import torch
    from aadistill.initialization.specs.arch import ArchSpec, get_adapter
    from aadistill.models.student import stored_rope_base
    from phase_a_frozen import TARGET_GEOMETRY, TEACHER_ID, TEACHER_REVISION
    from transformers import AutoConfig

    dtype = getattr(torch, dtype_name)
    teacher_config = AutoConfig.from_pretrained(
        TEACHER_ID, revision=TEACHER_REVISION)
    adapter = get_adapter(cfg["adapter"])
    spec = ArchSpec.of(cfg["family"], TARGET_GEOMETRY)
    config = adapter.build_config(teacher_config, spec)

    started = time.time()
    model = adapter.build_model(config, dtype, cfg["seed"])
    built = placement_of(model)
    params = sum(p.numel() for p in model.parameters())
    root = work / "real_geometry"
    adapter.save(model, str(root))
    del model
    torch.cuda.empty_cache()

    #: The CANONICAL loader, the same call a session uses, on the device.
    reloaded = adapter.load(str(root), dtype=dtype, device=device)
    place = placement_of(reloaded)
    reloaded_params = sum(p.numel() for p in reloaded.parameters())
    #: The rope base, read from the LOADED model through the repo's own
    #: reader. transformers moved `rope_theta` from a flat config key into a
    #: nested `rope_parameters` dict between 4.x and 5.x: on 5.x, reading
    #: `config.rope_theta` RAISES, and on a mixed pair it silently returns the
    #: 4.x class default -- the 500x misread that reports a wrong NLL instead
    #: of failing. `stored_rope_base` handles both layouts and already exists;
    #: a second local read of the same field is how the two drift apart.
    rope = stored_rope_base(reloaded.config)
    tied = bool(getattr(reloaded.config, "tie_word_embeddings", False))
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    elapsed = round(time.time() - started, 2)

    out = {
        "teacher_id": TEACHER_ID,
        "teacher_revision": TEACHER_REVISION,
        "_weights_not_downloaded": "the pinned CONFIG only",
        "target_geometry": dict(TARGET_GEOMETRY),
        "parameters": params,
        "parameters_after_reload": reloaded_params,
        "built_placement": built,
        "reloaded_placement": place,
        "rope_theta_from_the_loaded_model": rope,
        "rope_theta_from_the_teacher_config": stored_rope_base(teacher_config),
        "transformers_version": __import__("transformers").__version__,
        "tie_word_embeddings": tied,
        "peak_cuda_gib": round(peak, 3),
        "elapsed_seconds": elapsed,
    }
    del reloaded
    torch.cuda.empty_cache()

    assert place["devices"] == [device], (
        f"the reloaded student sits on {place['devices']}, expected [{device!r}]")
    assert place["dtypes"] == [f"torch.{dtype_name}"], (
        f"the reloaded student carries {place['dtypes']}, "
        f"expected torch.{dtype_name}")
    assert reloaded_params == params, (
        f"reload changed the parameter count: {params} -> {reloaded_params}")
    assert rope is not None, (
        "the reloaded student records no rope base under EITHER field layout")
    assert rope == out["rope_theta_from_the_teacher_config"], (
        f"rope base moved through build/save/reload: "
        f"{out['rope_theta_from_the_teacher_config']} -> {rope}")
    assert tied, (
        "the real students tie the lm head to the embedding, and the operators "
        "rely on it")
    say(f"stage M passed in {elapsed}s — {params:,} parameters, reloaded on "
        f"{device} in {dtype_name}, rope {rope}, peak {out['peak_cuda_gib']} GiB")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="configs/validation/c2_full_search_cuda.json")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts/validation")
    a = ap.parse_args()

    cfg = json.loads((REPO / a.config).read_text()
                     if not Path(a.config).is_absolute()
                     else Path(a.config).read_text())
    out_dir = REPO / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "aadistill.c2_full_search_cuda_check/v1",
        "run_id": a.run_id,
        "validation_id": cfg["validation_id"],
        "scientific_use": False,
        "authorizes": "nothing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }

    try:
        device_info = require_cuda(a.device)
    except SystemExit as exc:
        #: Two different refusals arrive here and they are NOT the same verdict:
        #: an integer code means "no CUDA device" -- an environment that cannot
        #: host the check, NOT RUN. A string means the CALLER asked for a device
        #: this check refuses to certify on, which is a failure of the request.
        #:
        #: The first version ran `int(exc.code)` on both and raised ValueError
        #: on the string, so the line meant to report the refusal crashed
        #: instead -- and the verdict a launcher reads would have been a
        #: traceback. Found by executing this path at $0.
        if isinstance(exc.code, int):
            report["verdict"] = "NOT RUN"
            report["reason"] = "no CUDA device"
            code = exc.code
        else:
            report["verdict"] = "FAIL"
            report["reason"] = str(exc.code)
            code = FAILED
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (out_dir / "c2_full_search_cuda_report.json").write_text(
            json.dumps(report, indent=1) + "\n")
        say(f"verdict: {report['verdict']} — {report['reason']}")
        return code

    report["device"] = device_info
    dtype_name = cfg["dtype"]
    verdict = "PASS"
    try:
        require_capability(device_info, cfg["capability_requirement"])
        for name, fn in (("driver", stage_driver), ("geometry", stage_geometry)):
            if not cfg["stages"].get(name, {}).get("enabled", True):
                report["stages"][name] = {"skipped": "disabled in the config"}
                continue
            say(f"--- stage {name} ---")
            report["stages"][name] = fn(cfg, a.device, dtype_name, work)
    except AssertionError as exc:
        verdict, report["failure"] = "FAIL", str(exc)
        report["traceback"] = traceback.format_exc()[-4000:]
        say(f"FAIL: {exc}")
    except Exception as exc:                                       # noqa: BLE001
        verdict = "FAIL"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-6000:]
        say(f"FAIL: {report['failure']}")

    report["verdict"] = verdict
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "c2_full_search_cuda_report.json").write_text(
        json.dumps(report, indent=1, default=str) + "\n")
    say(f"verdict: {verdict}")
    return OK if verdict == "PASS" else FAILED


if __name__ == "__main__":
    raise SystemExit(main())
