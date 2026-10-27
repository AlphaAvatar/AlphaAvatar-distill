"""Step times measured on earlier sessions, for the launchers that price from them.

These are experiment facts, not framework facts: `aadistill.infrastructure.budget`
prices whatever step time, floor and superseded figures its caller supplies, and
names none itself. Provenance: `docs/maintenance/core-provenance.md`.
"""

#: What a Stage-3 ladder arm actually sustained on an L40S: 2916 steps, bf16,
#: teacher-in-memory KD, both arms, mean of the per-10-step console timings
#: (E6b, 2026-08-08). Pass it as `step_time_floor` to refuse a Stage-3 L40S
#: estimate below it that gives no reason.
MEASURED_STEP_SECONDS = 4.15

#: Derived from E4's comparable arms; it underpriced E6b by 14%. Pass it in
#: `superseded_step_seconds` so a plan reaching for it is named in its notes.
SUPERSEDED_STEP_SECONDS = 3.625
