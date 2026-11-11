"""The teacher snapshot setup fetched, as a local directory, on a pod that runs offline.

Setup's TEACHER_READY step (`autoinit_preflight_setup.sh`) downloads only `TEACHER_FILES` of the
teacher repository: weights, configs, tokenizer, chat template. huggingface_hub (1.x) refuses an
offline ``snapshot_download(local_files_only=True)`` whose cached tree listing names a file the
snapshot lacks, so asking for the whole repository fails on README.md, LICENSE and .gitattributes
(A-E ae_20261028_091943, $0.23). The offline lookup asks for exactly the files setup fetched; one of
those missing still refuses where the cached listing names it.
"""

from __future__ import annotations

#: setup's `allow_patterns`, which `scripts/shared/tests/test_teacher_cache.py` keeps equal to the script's
TEACHER_FILES = ("*.json", "*.safetensors", "*.jinja", "*.txt")


def local_snapshot(model: str, revision: str) -> str:
    """The cached snapshot directory of `model` at `revision` (the path names the commit)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(model, revision=revision, local_files_only=True, allow_patterns=list(TEACHER_FILES))
