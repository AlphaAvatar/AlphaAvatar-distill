"""Artifacts shared by the D-series experiments (D1, D2, D3).

Lives beside `phase_d1`, `phase_c1`, ... rather than inside any one of them: a
family of six behavioural batteries, three of whose roles belong to experiments
that have not been designed yet, cannot be owned by the first experiment that
consumes it. D2 importing from `phase_d1` would make D1's module the authority
on D2's evidence.
"""
