"""Real model geometries, shared because three suites assert against them.

The 36-layer teacher and the 596M target are facts about the models this project
compresses, not about any one experiment. They lived in
`scripts/experiments/stage-3/tests/test_frozen_records.py`, so a test in another suite could only
reach them by importing that test module — which is how a historical E-series
test ended up imported by the shared cost-model tests, and why moving either one
broke the other.
"""

from __future__ import annotations

from aadistill.initialization.specs.arch import ArchSpec

#: The Qwen3 4B teacher, as its config declares it.
TEACHER_36 = ArchSpec.of("qwen3", dict(
    hidden_size=2560, num_hidden_layers=36, intermediate_size=9728,
    num_attention_heads=32, num_key_value_heads=8, head_dim=128,
    vocab_size=151936, tie_word_embeddings=True))

#: The 596M student target.
TARGET_596M = ArchSpec.of("qwen3", dict(
    hidden_size=1024, num_hidden_layers=28, intermediate_size=3072,
    num_attention_heads=16, num_key_value_heads=8, head_dim=128,
    vocab_size=151936, tie_word_embeddings=True))
