# teacher_parent_initialization (Stage 1)

Compares two initializations of the fixed 596M student after the same Stage-3 recovery:

* `A_teacher_parent` makes every selection on the original teacher. The residual PCA is re-aggregated from the
  teacher's own statistics at this round's kept layers.
* `B_packed_top1` is the packed Top-1, already trained as A-MIX-6cat-sr. Its sealed evaluation is reused, and it is
  never retrained.

The experiment is `configs/stages/stage-1/teacher_parent_initialization/design.json`. Evidence is in
`logs/stages/stage-1/teacher_parent_initialization/`, and the design note is
`docs/stages/stage-1/teacher_parent_initialization/README.md`. The training execution stage stays `stage3_recovery`.

| file | runs where | what it does |
| --- | --- | --- |
| `selection.py` | dev box, $0 | reads A's FFN(base), DEPTH(base) and ATTENTION(rh) selections, plus the RESIDUAL_WIDTH(rh) gate, from the packed search's teacher-level states into `selection.json`; checks the DEPTH record round by round (`--write` / `--check`) |
| `init_driver.py` | pod (`teacher_parent_init`), or dev box `--check-only` at $0 | builds A behind the gates below, collects the teacher statistics, runs the state evaluation, writes the relay copies and `init_record.json` |
| `plan.py` | dev box, $0 | derives the two training units from the A-MIX plan and a committed init record (`--write` / `--check`) |
| `train_driver.py` | pod (`teacher_parent_train`; SOB1's `sob1_train` through `--plan`, `--prefix-root` and its study module) | trains a plan's units with the A-MIX recipe, makes save points and final weights durable on the relay, then evaluates every eval-set-1 unit in both modes under the plan's protocol. Since 2026-11-19: an unreadable relay listing is retried and never read as empty; a verified upload whose record did not publish has only its record retried; copies get their time from their bytes at the floor rates, and training stops early enough to secure its last save point before the session's hard deadline; the first optimizer steps' memory is logged (`units.json :: memory`); `take_over` (and `launch.resume_archive`) never carry a stream under a `live/` directory at any depth (`is_live`) |
| `relay.py` | both | put (quota-checked, downloaded again and re-hashed), get (re-hashed), and permanent removal of a superseded save point |
| `launch.py` | dev box | `grant`, `issue`, `launch`, `closeout`, `position`, `stage-resume`, `retain` for study TPI1 (`--kind teacher_parent_init` or `--kind teacher_parent_train`) |
| `b_evidence.py` | dev box, $0 | audits B's sealed A-MIX evidence: completeness, protocol fields and identities (`audit/b_evidence.json`) |
| `analysis.py` | dev box, $0 | the registered paired A − B analysis (`analyses/teacher_parent.json`) |
| `tests/` | dev box | this experiment's own tests: `pytest scripts/stages/stage-1/teacher_parent_initialization/tests` |

`init_driver.py` checks these gates in order; each must hold before the next step builds on it:

1. the teacher's shards hash to the recorded values;
2. the FFN, DEPTH and ATTENTION children rebuilt on the teacher each equal their journal state bit for bit;
3. the replayed root width state equals its journal digest;
4. P over all 37 points reproduces the journal's diagnostics.

**Inputs.** The pod receives:
* the teacher snapshot, through setup;
* the calibration items and the state-evaluation suite (init kind), and the verified TriviaQA copy (train kind), as
  local assets;
* the A-MIX pack, from the relay;
* A's initialization, from its relay record (train kind).

**Resume.**
* **Init:** a later attempt given `--resume` takes a verified init record over, and nothing is recomputed. The
  products are cheap to rebuild, so a failed attempt rebuilds them.
* **Training, a unit with no local state:** it gets back its final weights if they are durable, otherwise its latest
  save point. That save point carries the weights, the AdamW state, the step, the consumed sequences and the RNG.
  The trainer then continues with `--resume latest`.
* **Evaluation:** a later attempt generates only the samples that have no sealed record, each under the seed its
  item derives.

To resume after a pod loss, stage the secured evidence with
`launch.py stage-resume --run-id <new> --from <old scr>` and pass the result as `--resume-from` at issue and launch.
