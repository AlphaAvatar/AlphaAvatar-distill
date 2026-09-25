"""`use_cache` is part of a checkpoint's identity, and one operator mutates it.

This is the mechanism that made three of attempt 3's five selected leaves
irreproducible by a path-pinned replay, and it is subtle enough to deserve a
test that states it outright:

* `DepthCausalKLGreedyV1.apply` sets `model.config.use_cache = False` on the
  PARENT MODEL OBJECT, because `bypassed_blocks` cannot work with a
  layer-indexed KV cache;
* `build_config` copies the parent's dict, so every descendant inherits it;
* `use_cache` is serialized into `config.json`, so it lands in `config_sha256`
  and therefore in `artifact_digest`.

A search that reuses one teacher model across expansions therefore records
artifacts whose identity depends on BEAM ORDER: once any causal-KL DEPTH
expansion has run, every later expansion of any path starts from a contaminated
root. A replay that loads a fresh teacher per path cannot reproduce those
artifacts unless the root's cache flag is pinned as an input.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit"):
    if str(ROOT / extra) not in sys.path:
        sys.path.insert(0, str(ROOT / extra))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from aadistill.initialization.adapters import (  # noqa: E402
    register_builtin_adapters,
)
from aadistill.initialization.specs.arch import ArchSpec, get_adapter  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "autoinit_conftest", ROOT / "tests/autoinit/conftest.py")
_fx = importlib.util.module_from_spec(_spec)
sys.modules["autoinit_conftest"] = _fx
_spec.loader.exec_module(_fx)

TEACHER_GEOMETRY = _fx.TEACHER_GEOMETRY
build_tiny_model = _fx.build_tiny_model


def _adapter():
    register_builtin_adapters()
    return get_adapter("qwen3")


def _config_sha(config) -> str:
    """Exactly as `CheckpointIdentity` computes it: write, read back, hash."""
    with tempfile.TemporaryDirectory() as tmp:
        config.save_pretrained(tmp)
        return sha256_json(json.loads((Path(tmp) / "config.json").read_text()))


def test_use_cache_changes_the_config_sha_and_therefore_the_identity():
    """The premise. If this ever stops being true the rest is moot."""
    adapter = _adapter()
    model = build_tiny_model(TEACHER_GEOMETRY)
    spec = ArchSpec.of("qwen3", dict(TEACHER_GEOMETRY,
                                     num_attention_heads=2))

    model.config.use_cache = True
    on = _config_sha(adapter.build_config(model.config, spec))
    model.config.use_cache = False
    off = _config_sha(adapter.build_config(model.config, spec))

    assert on != off, (
        "use_cache no longer reaches config.json; the identity consequence "
        "this whole test file describes would be gone")


def test_build_config_inherits_use_cache_from_the_parent():
    """The propagation. One flipped bit on a shared parent reaches every child."""
    adapter = _adapter()
    spec = ArchSpec.of("qwen3", dict(TEACHER_GEOMETRY, num_attention_heads=2))

    parent = build_tiny_model(TEACHER_GEOMETRY)
    parent.config.use_cache = False
    child = adapter.build_config(parent.config, spec)
    assert child.use_cache is False

    grandchild = adapter.build_config(
        child, ArchSpec.of("qwen3", dict(TEACHER_GEOMETRY,
                                         num_attention_heads=2,
                                         intermediate_size=24)))
    assert grandchild.use_cache is False, (
        "the flag stopped propagating; a descendant would silently regain a "
        "KV-cache default its ancestor had turned off")


def test_the_causal_kl_depth_operator_is_the_one_that_mutates_it():
    """Named from the SOURCE, so a second mutation site cannot appear unnoticed.

    Grepping the initialization tree rather than asserting a remembered list:
    if another operator starts flipping this bit, the set changes and this
    fails, which is the point.
    """
    import re

    tree = ROOT / "src/aadistill/initialization"
    sites = []
    for path in sorted(tree.rglob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            #: An ASSIGNMENT, not a mention. `contribution.py` names the flag
            #: inside the error it raises when the caller has not turned it off;
            #: that is a requirement, not a mutation, and counting it here would
            #: make this test assert the wrong thing.
            stripped = line.strip()
            if not re.match(r"^[A-Za-z_][\w.]*\.use_cache\s*=\s*False\s*(#.*)?$",
                            stripped):
                continue
            sites.append(str(path.relative_to(ROOT)))
    #: The FILE, not the line. A line number pins nothing this test is about --
    #: it moves when a docstring above the assignment is edited, which is a
    #: false alarm, while a SECOND assignment in the same file changes the list
    #: either way. What must not change silently is WHICH modules flip the bit.
    assert sites == [
        "src/aadistill/initialization/operators/depth/causal_kl_greedy.py"], (
        f"the set of use_cache mutation sites changed: {sites}. Each one makes "
        "a checkpoint's identity depend on whether that operator has run, so a "
        "new site needs a deliberate decision, not a silent addition.")


def test_the_mutation_lands_on_the_parent_not_a_copy():
    """Why beam order leaks into identity.

    The operator mutates the model it was handed. In a search that reuses one
    teacher object, that object is shared with every later expansion.
    """
    source = (ROOT / "src/aadistill/initialization/operators/depth"
                    "/causal_kl_greedy.py").read_text()
    apply_body = source[source.index("class DepthCausalKLGreedyV1"):]
    apply_body = apply_body[apply_body.index("def apply"):]
    head = apply_body[:apply_body.index("domains = ")]
    assert "model = ctx.model" in head
    assert "model.config.use_cache = False" in head, (
        "the mutation moved; if it now targets a copy, the beam-order "
        "contamination described in the replay forensic no longer applies and "
        "that document needs revisiting")
