"""A single residual stream every block reads from and writes to.

Operators here may assume one stream of one width, so a projection of that
width is a projection of the whole inter-block interface. A Hyper-Connection
style architecture carries several parallel streams and would not satisfy that;
whether it becomes a topology beside this one or a distinct kind with different
``ArchSpec`` fields is deliberately left open — see the kind note in
:mod:`.global_pca`.

Importing this package registers nothing.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
