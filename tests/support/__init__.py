"""Helpers shared by the core suite and the experiment suites.

Imported as ``support.<module>`` — the repository-root conftest puts ``tests/``
on ``sys.path`` so one spelling works from ``tests/`` and from
``scripts/experiments/*/tests/`` alike.

**No module here contains a test.** That is the point: a helper that lived
beside the tests of one subsystem could only be imported by them, so a test
moving to its owning experiment had to leave its fixtures behind or drag a
`tests/`-relative import with it.
"""
