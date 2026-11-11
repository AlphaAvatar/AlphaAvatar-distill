# Semantic review: `schema_synth@v1` schema_arguments tasks

Reviewer: independent, read-only. Repository unchanged.

Inputs: `scripts/stages/stage-2/teacher_modes/schema_synth.py` (working tree, sha256 `00c6eaad…b5b2`, the same hash as the manifest's `generator_sha256`, so the file that was read is the one that wrote the tasks) and `artifacts/stages/stage-2/teacher_modes/schema_synth_v1/tasks.jsonl` (5,070 tasks, 30 domains).

## Method

1. **Read every sampler and tool schema.** I read all 30 `Domain(...)` definitions and the shared phrase toolkit (`date_phrase`, `time_phrase`, `duration_phrase`, `count_phrase`, `money_text`, `phone`).
2. **Re-derived every gold call independently, all 5,070 tasks.** I wrote new parsers in `scratchpad/review/lib.py` and `derive.py`; nothing is imported from the generator. They take only what the teacher sees, the system-prompt date and `messages[0].content`, and rebuild the full argument object:
   - relative dates and weekdays;
   - 12-hour, 24-hour and spoken times;
   - durations, money, E.164 phone numbers;
   - dozens and packs;
   - unit conversions (kg→g, mL→µL, GB→MiB);
   - totals, scaling, ages in months, and the threshold, plan and class rules;
   - sorting rules, and whether each optional field is present.

   The rebuilt object is compared with gold using exact canonical JSON. **Result: 5,070 / 5,070 match.** Where the request has more than one reasonable reading, the parser had to pick a convention, so "match" means the gold is consistent with its own convention. Those cases are measured separately below.
3. **Ran the repository's validator on every gold call.** All gold calls pass `aadistill.data.tool_calls.call_problems`, and every replay table has exactly one key, `canonical_call(gold)`.
4. **Checked for answer leakage.**
   - The system prompt follows its template in all 5,070 tasks.
   - Each domain uses one fixed tool set: target plus 2 distractors, 3 tools per task, and no tool name is shared across domains.
   - No visible field contains `gold`, `"arguments"`, `step_gold` or `canonical`.
   - No derived value (E.164 number, cents total) appears verbatim in the request.
   - Only one user message is present.
5. **Read tasks by hand.** I read 4 to 6 tasks per domain: at least one row ≥160 in every training domain, and every difficulty level, with level-3 rows in all held-out domains. I also listed each phrase class that admits two readings and counted the tasks where the readings give different calls (`ambig.py`, `appendix.txt`).

## Domain table

| # | domain (split) | tasks inspected (manual / systematic) | verdict | note |
|---|---|---|---|---|
| 1 | parcel_shipment (training) | 5 / 220 | AMBIGUOUS-RISK | R1: 10 tasks. Low risks: "A x B x C cm" read as L/W/H, US numbers read as +1 (L1, L2) |
| 2 | calendar_event (reserve) | 5 / 10 | OK | corrections, reminders and attendee order all unambiguous |
| 3 | restaurant_booking (training) | 4 / 220 | AMBIGUOUS-RISK | R1: 11. Cosmetic: 5 "corrections" restate the same time (C2) |
| 4 | flight_search (calibration) | 5 / 30 | AMBIGUOUS-RISK | R1: 2/30 (rows 17, 18). The age-band rule is stated and correct |
| 5 | hotel_booking (state_eval) | 5 / 40 | AMBIGUOUS-RISK | D2: "one big bed for one" could be SINGLE or DOUBLE, 7/40. R1: 4 |
| 6 | grocery_order (training) | 4 / 220 | AMBIGUOUS-RISK | R1: 15. Low: "N packs of g" read as N×g units, 70 tasks (L4) |
| 7 | medication_reminders (training) | 4 / 220 | AMBIGUOUS-RISK | R1: 4 only. Schedule, halving and sorting all correct |
| 8 | expense_report (training) | 4 / 220 | OK | stable date sort, categories and totals correct |
| 9 | thermostat_program (state_eval) | 5 / 40 | OK | Low: 6 tasks repeat the same setpoint in consecutive periods; the literal reading is gold (L6) |
| 10 | vm_provisioning (training) | 4 / 220 | OK | GB→MiB and GB→GiB conventions are stated |
| 11 | orders_report (training) | 4 / 220 | OK | "last N days (created after that date)" with an exclusive bound gives today−N, which is unique |
| 12 | ticket_purchase (state_eval) | 5 / 40 | OK | |
| 13 | car_rental (training) | 4 / 220 | AMBIGUOUS-RISK | R1: 16. Low: time described only as "HH:MM." (L3) |
| 14 | recipe_scaling (training) | 4 / 220 | OK | every scaled amount is an exact integer |
| 15 | invoicing (calibration) | 5 / 30 | OK | |
| 16 | pet_grooming (training) | 4 / 220 | AMBIGUOUS-RISK | R1: 16. Low: L3 |
| 17 | subscription_change (training) | 4 / 220 | OK | the "cheapest plan" rule is fully specified |
| 18 | timesheet (training) | 4 / 220 | AMBIGUOUS-RISK | D3: "last week" said on a Sunday, 39/220 |
| 19 | lab_samples (training) | 4 / 220 | OK | |
| 20 | translation_job (training) | 4 / 220 | AMBIGUOUS-RISK | R1: 18 |
| 21 | parking_reservation (training) | 4 / 220 | **DEFECT** | D1: "My motorcycle/van … I need a charger for my electric car", 59/220. Also R1: 7, C3 |
| 22 | warehouse_transfer (training) | 4 / 220 | AMBIGUOUS-RISK | R1: 11. Date corrections are fine |
| 23 | library_holds (validation) | 4 / 20 | OK | |
| 24 | utility_payment (training) | 4 / 220 | AMBIGUOUS-RISK | R1: 9. Amount corrections are fine |
| 25 | conference_registration (training) | 4 / 220 | OK | the cutoff is never equal to today |
| 26 | insurance_claim (training) | 4 / 220 | OK | Low: L3 |
| 27 | bug_tracker (training) | 4 / 220 | OK | lexical sort equals semver sort for this version set |
| 28 | email_campaign (validation) | 5 / 20 | OK | |
| 29 | workout_log (training) | 4 / 220 | OK | |
| 30 | crm_update (training) | 4 / 220 | OK | |

Totals: 17 OK, 12 AMBIGUOUS-RISK, 1 DEFECT. All 5,070 tasks were re-derived.

## Defects and risks in detail

### D1. parking_reservation: the request contradicts itself (DEFECT, 59 / 220 tasks)

- **Code:** `_parking` draws the vehicle type from CAR, MOTORCYCLE or VAN (line 1071). Independently, the EV clause is always the fixed text "I need a charger for my electric car" (line 1077).
- **Effect:** when the type is MOTORCYCLE (35 tasks) or VAN (24), the request names one vehicle as the one to park and a different vehicle as needing the charger.
- **Example:** `tc1-d87a579d05d2e5207306` (row 7), "Reserve a parking spot. Lot P3. My van, plate VGC 524. From a week from today at 16:45. For an hour and a half. I need a charger for my electric car."
  - Gold: `vehicle.type = "VAN"`, `ev_charging = true`.
- **Other examples:** row 10, `tc1-4daa78232fcc6178a64a` (motorcycle); row 180, `tc1-dc70fb65cb9298a5420f` (motorcycle).
- **Why this is a defect:** the gold takes the most literal reading (park the van, give it a charging space). Two other readings are reasonable: the van is not the electric car, so `ev_charging=false`; or the vehicle is really a car, so `type=CAR`. A teacher that notices the contradiction may also ask for clarification instead of calling. The fault is in the request text itself, not in the gold arithmetic.
- **Affected rows (59):** 7, 10, 16, 17, 18, 21, 25, 30, 35, 37, 38, 43, 44, 47, 55, 66, 70, 78, 81, 83, 84, 88, 93, 99, 102, 104, 108, 109, 111, 114, 116, 120, 121, 127, 128, 136, 137, 139, 140, 143, 145, 146, 147, 148, 163, 167, 169, 173, 180, 184, 185, 191, 193, 200, 201, 204, 207, 209, 217.

### D2. hotel_booking: "a room with one big bed for one" fits two enum values (AMBIGUOUS-RISK, 7 / 40 state_eval tasks)

- **Tool enum:** `SINGLE (1 bed, 1 guest), DOUBLE (1 large bed), TWIN (2 beds), SUITE`.
- **Code:** the sampler allows a DOUBLE with 1 guest (line 530) and words it "a room with one big bed for 1/one".
- **The two readings:** that request matches SINGLE's description exactly ("1 bed, 1 guest"), and it also matches DOUBLE's ("1 large bed"). Gold is always DOUBLE. "Big bed" pointing to "large bed" favours DOUBLE, but SINGLE is a reasonable call.
- **Affected tasks (7):**
  - row 1 `tc1-24ef5398e3d3320aab4e`
  - row 2 `tc1-84b32cc91fbccdea3ab3`
  - row 11 `tc1-294da303c2ea8c5caf8f`
  - row 24 `tc1-767ec004a567f09d8772`
  - row 28 `tc1-a1b14f3cab7da8dbebad`
  - row 30 `tc1-2f4d46a66cf1430f5a46`
  - row 36 `tc1-ff14b5e4a781cf556ebe`
- **Why it matters more here:** this is a state_eval domain, so these tasks are 17.5% of the domain.

### D3. timesheet: "Log my hours from last week" said on a Sunday (AMBIGUOUS-RISK, 39 / 220)

- **Gold:** `week_start = today − (weekday + 7)`, the Monday of the previous Monday–Sunday week (line 971).
- **Where the readings agree:** Monday to Saturday, the Monday–Sunday and Sunday–Saturday week conventions give the same "last week".
- **Where they split, on a Sunday:**
  - Monday–Sunday weeks: today is the last day of the current week, so gold's Monday is 13 days back.
  - Sunday–Saturday weeks, the usual US convention: today starts a new week, so "last week" is the one that just ended and its Monday is 6 days back. That is also the work week the user has just finished.
- **Example:** `tc1-6e2bc457a2b18ec949d5` (row 6), current date Sunday 2026-10-18: "Monday: an eight-hour day on PRJ-DELTA; Thursday: 4 hours …".
  - Gold: `week_start 2026-10-05`, entries dated 10-05, 10-08, 10-09.
  - Alternative: `2026-10-12`, entries dated 10-12, 10-15, 10-16.
- **Second example:** row 10, `tc1-00c5aa6a34c9c52b0f64`.
- **Field description:** it says only "Monday of the week" and does not define "last week".
- **Affected rows (39):** 6, 10, 13, 16, 23, 37, 38, 39, 42, 45, 47, 48, 53, 63, 64, 80, 86, 96, 106, 109, 111, 120, 121, 123, 126, 127, 133, 143, 160, 162, 163, 164, 172, 175, 179, 197, 198, 202, 205.

### R1. "the <Weekday> after next" (AMBIGUOUS-RISK, shared by 12 domains, 123 tasks)

- **Gold:** `coming(today, w, 1)` (lines 155–157), meaning the X after the first X strictly after today.
- **The competing reading:** take "next X" to mean the X in next calendar week. Then "the X after next" is the X in the week after next, 7 days later than gold.
- **When the readings split:** only when the first coming X still falls in the current week.
  - Under Monday–Sunday weeks: 101 tasks.
  - Under Sunday–Saturday weeks: 89 tasks.
  - Under either: **123** tasks out of the 212 that use the phrase.
- **Hardest subset (35 tasks):** the coming X is tomorrow.
- **Examples:**
  - `tc1-51fc3987e83219231fbc` (parcel_shipment row 18), Tuesday 2026-07-14, "picked up the Wednesday after next". Gold `2026-07-22`; the alternative is `2026-07-29`.
  - `tc1-ed3028f02570576fa8d0` (flight_search row 17), Wednesday 2026-07-01, "Leaving the Thursday after next". Gold `2026-07-09` (the coming Thursday is tomorrow); the alternative is `2026-07-16`.
- **Assessment:** gold follows what is probably the majority reading. Still, this is a well-known English ambiguity, inherited from "next Friday", and no description states a convention.
- **By split:** training 117, calibration 2, state_eval 4.
- **Per domain:**

  | domain | tasks | rows |
  |---|---|---|
  | car_rental | 16 | 23, 29, 53, 66, 71, 81, 93, 98, 115, 121, 125, 129, 136, 174, 183, 195 |
  | flight_search | 2 | 17, 18 |
  | grocery_order | 15 | 17, 21, 36, 38, 70, 85, 94, 109, 111, 129, 131, 141, 173, 193, 206 |
  | hotel_booking | 4 | 15, 18, 28, 33 |
  | medication_reminders | 4 | 21, 85, 128, 135 |
  | parcel_shipment | 10 | 1, 18, 57, 90, 115, 145, 147, 148, 185, 214 |
  | parking_reservation | 7 | 3, 57, 58, 106, 130, 163, 182 |
  | pet_grooming | 16 | 4, 29, 39, 54, 57, 87, 99, 106, 140, 143, 149, 163, 182, 183, 189, 198 |
  | restaurant_booking | 11 | 6, 35, 42, 126, 145, 149, 174, 177, 187, 206, 208 |
  | translation_job | 18 | 5, 6, 9, 36, 49, 76, 83, 84, 99, 148, 150, 168, 192, 198, 204, 206, 214, 219 |
  | utility_payment | 9 | 0, 42, 52, 131, 147, 175, 187, 188, 205 |
  | warehouse_transfer | 11 | 11, 32, 40, 81, 115, 118, 120, 129, 137, 154, 214 |

  calendar_event and email_campaign never use the phrase in a case where the readings differ.

### Low risks (each has one clearly better reading; not counted as defects)

- **L1. parcel_shipment, box dimensions (220 tasks).** "35 x 50 x 30 cm" is mapped to `length_cm`, `width_cm`, `height_cm` in that order. The order is conventional but not stated in the description.
- **L2. US-format numbers without a country code.** Examples: "(945) 697-8469" and "550-484-1635", mapped to `+1…`; 138 of the parcel tasks, plus restaurant, pet and crm tasks. UK "020 xxxx xxxx" maps to `+4420…`. Both are standard, and the description's example is a +1 number.
- **L3. Time fields described only as "HH:MM."** In car_rental, pet_grooming, parking_reservation, insurance_claim, email_campaign and lab_samples, the description does not say "24-hour" (the other domains do). PM times occur in 329 of these tasks. `02:15` for "2:15 pm" would pass the pattern but lose the information, so 24-hour is the only sensible reading.
- **L4. grocery_order packs (70 tasks).** "three packs of 5 bagels" gives gold `quantity = 15`; the field says "Number of units" and the catalog lists items, not packs. warehouse_transfer spells out its equivalent rule ("cases x units per case"); grocery does not.
- **L5. "this coming X" said on day X (28 tasks).** Gold is today + 7, which is the standard reading.
  - car_rental 68, 102, 152
  - grocery_order 172, 185
  - medication_reminders 79, 173
  - parcel_shipment 14, 71, 117, 122, 201
  - parking_reservation 118, 162, 194
  - pet_grooming 20, 55, 141
  - restaurant_booking 87, 102, 170
  - translation_job 4, 60, 104, 213
  - utility_payment 60, 73
  - warehouse_transfer 23
- **L6. thermostat_program, repeated setpoints (6 tasks: rows 5, 16, 17, 25, 29, 38).** Two consecutive periods have the same temperature, for example "20°C from 3 pm, 20°C from 4 pm". Gold keeps both periods, which is the literal reading; a teacher that merges them would fail.
- **L7. "12:xx pm" (36 tasks).** Gold treats it as the noon hour, which is standard.

### Cosmetic (gold unaffected)

- **C1. Ungrammatical date phrasing in about 145 tasks.** Phrases such as "From in eleven days" (39), "On in nineteen days" (42), "Needed by in 20 days" (32) and "On tomorrow" (32). The meaning is intact.
- **C2. restaurant_booking corrections that change nothing (5 tasks: rows 26, 30, 102, 126, 214).** The "correction" restates the same time in another spelling, e.g. "at 21:00 -- actually no, make that 9:00 pm". These rows are still tagged with the `correction` operator.
- **C3. parking_reservation "to be safe" with a shorter duration (20 tasks).** Example: "For 2 hours, or rather 105 minutes to be safe". "Or rather" still clearly selects the second value.
- **C4. Notes-style closings run into the last bullet,** e.g. "- Text me Thanks!".
- **Metadata (not visible to the teacher).** grocery_order `source.generator.operators` always includes `count:dozen`, including 146 tasks with no dozen phrase. Operator-level statistics built from that field will be inaccurate.

## Overall verdict

**No computational or consistency defects.**
- Every one of the 5,070 gold calls can be rebuilt exactly from the stated current date and the user text.
- Each call is schema-valid and is the only key in its replay table.
- Each field follows the convention its tool description states: units, sort order, when optional fields are present, the enum meanings, and the threshold, plan, class and early-bird rules.
- Nothing visible to the teacher reveals the answer, and distractor sentences (old ids, retracted values) are unambiguous.

**Interpretive problems in 226 of 5,070 tasks (4.5%):**
- training 214 / 4,840 (4.4%)
- state_eval 10 / 120 (8.3%, all hotel_booking)
- calibration 2 / 60
- validation and reserve 0

They fall into one defect class and three risk classes:

| class | domain(s) | tasks |
|---|---|---|
| D1, self-contradictory request (defect) | parking_reservation | 59 |
| R1, "X after next" | 12 domains | 123 |
| D3, "last week" on a Sunday | timesheet | 39 |
| D2, DOUBLE for one guest | hotel_booking | 7 |

The classes overlap in 2 tasks.

**What they cost.** Exact-match admission against an internally consistent gold means none of these can admit a wrong call. The cost is that a teacher taking the other reasonable reading is rejected: admission yield drops in those rows, and admission is quietly biased toward the generator's own conventions. That matters most for the 10 state_eval hotel tasks, where it shifts a measurement rather than a yield.

**Smallest fixes, for a new version since v1's bytes are frozen:**
- D1: make the EV clause vehicle-neutral ("it needs a charging space") or draw EV only for CAR.
- D2: drop DOUBLE with 1 guest, or describe SINGLE as "1 single bed".
- D3: state the week convention in the request or description ("the previous Monday–Sunday week"), or avoid Sunday current dates for this domain.
- R1: replace "the X after next" with an unambiguous phrase ("a week after this coming X"), or state the convention in the date field descriptions.

Scripts and row lists are in `/tmp/claude-1000/-home-ecs-user-AlphaAvatar-distill/40abda2f-efac-4413-af84-45085d7650d4/scratchpad/review/` (`lib.py`, `derive.py`, `ambig.py`, `leak.py`, `appendix.txt`).
