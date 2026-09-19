# `.scratch/` — ephemeral per-run working directories

```
.scratch/stage-<N>/<experiment>/<run>/
```

Gitignored. Nothing durable lives here: a launcher's scratch root holds the
relay mirror, the fetched archive and the run transcript while a session is
live, and every one of those is either collected into
`logs/stages/<stage>/<experiment>/runs/<run>/` or is a copy of something that
was.

It exists because per-attempt roots were being created at the top of `$HOME` —
`aad-scratch-c2replay-a2` through `a8`, `c1_scr` through `c1_scr4` — one per
launcher invocation, none of them removed. They were inventoried on 2026-09-19
and removed in one pass after their unique content (launcher transcripts,
watchdog journals, two price-refusal logs) was preserved into the run tree.

Heavy artifacts still go to `/home/ecs-user/aad-artifacts`, which remains the
single intentional out-of-repo durable store.
