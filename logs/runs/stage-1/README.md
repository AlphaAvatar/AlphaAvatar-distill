# stage-1 — projection and structural initialization

Runs whose subject is **Stage 1** of `AGENTS.md` §4.3: building a student
checkpoint from teacher structure and teacher activations rather than from
random weights — embedding and lm-head projection, grouped activation PCA for
hidden width, sandwich initialization for attention, activation-importance
neuron selection for FFN, and teacher-span-to-student-layer depth mapping.

A run appears here because its experiment **declares** `stage_id: "1"`, not
because of anything in its name.

## What "subject" means here, and why it is not stage 3

An experiment is filed by the question it answers, not by the machinery it runs
to answer it. `phase_c1` compares two Stage-1 initializations that differ in
exactly one operator of `DEPTH → FFN → RESIDUAL_WIDTH → ATTENTION`. To measure
them it trains short 0.86M SFT recovery probes — which is Stage-3 *work* — but
the probes are the instrument. Filing the experiment under stage 3 would put an
initialization study next to the student-recovery experiments and lose what it
is for.

Experiments are listed by `../index.json`, which is canonical.
