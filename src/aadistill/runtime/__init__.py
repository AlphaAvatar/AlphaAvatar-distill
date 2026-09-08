"""Generic run-time contracts: where a run's files live, for any stage.

Nothing here names a repository directory, a stage, an experiment or a role
vocabulary. The caller supplies the run root and its own roles; this package
validates structure, containment, uniqueness and hashes.
"""
