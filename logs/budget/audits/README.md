# Accounting audits

An audit sets the provider's own billing beside every record in this
repository. It is read-only: no record, grant, authorization or cap is edited
by one. Each audit is regenerated from a committed provider snapshot by
[`accounting_audit.py`](../../../scripts/maintenance/consolidation/accounting_audit.py).

| audit | snapshot | verdict |
| --- | --- | --- |
| [`accounting_audit_2026-10-27.json`](accounting_audit_2026-10-27.json) | [`runpod_billing_2026-10-27T1011Z.json`](../provider/runpod_billing_2026-10-27T1011Z.json) | CLOSED, with four findings |

## 2026-10-27: closing the mixed-calibration accounting

The maintainer required the mixed-calibration budget attribution and the audit
of historical unposted sessions to be finished, with an independent conclusion,
before any new paid experiment.

**Conclusion: CLOSED.**

- The provider-based position is complete and conservative.
- No unposted session is outstanding.
- Every billed pod since the 2026-09-11 anchor is attributed.
- The remaining difference lies before that anchor, which the maintainer
  stated. Its attribution is the maintainer's call.

**Position at 2026-10-27T10:11Z** (`mixed_governance.position`, every billed row
counted):

| | USD |
| --- | ---: |
| provider-billed (pods 717.6444 + volumes 2.5501) | 720.1945 |
| unposted sessions | 0 |
| `am5m373kuq` reservation, net of its posted rows | 4.0979 |
| MC1 volume `t6yrsvujha`, deleted inside the posting window | 0.0526 |
| **committed** | **724.3450** |
| project cap (maintainer, 2026-10-26) | 740.00 |
| **headroom** | **15.655** |

The account balance ($139.24) is not a grant.

**The mixed-calibration study** reconciles item by item:

| item | booked | provider billed |
| --- | ---: | ---: |
| MC1 attempt 1 (`6cy5fhf4q1tru0`) | 0.1903 | 0.1847 |
| MC1 attempt 2 (`z90fahqwqrlgb0`) | 6.1817 | 6.1777 |
| MC2 (`ntluvkf9fod0ec`) | 10.9120 | 10.9119 |
| MC1 volume (`t6yrsvujha`) | 0.0526 | inside the 2026-10-26 volume row |
| **total** | **17.3366** | equals the grant's closeout |

**Provider against records** ($720.19 − $687.40 = **+$32.79**):

| | provider | records | provider − records |
| --- | ---: | ---: | ---: |
| before the 2026-09-11 anchor | 305.43 | 267.90 (stated anchor) | +37.53 |
| since the anchor | 414.76 | 419.50 | −4.74 |

- **Before the anchor:** 55 billed pods ($77.33) are named by no record. The
  maintainer's stated anchor is $37.53 below what the provider billed before
  it. Whether all of that is project spend is unverified.
- **Since the anchor:** the records run $4.74 above the provider. Five
  recorded pods were never charged (a3_attempt20 ×3, a3_attempt37,
  `volaudit`), and sessions are booked at elapsed time.

**Findings and what was done:**

1. **Stale project cap in the records derivation.**
   - `derive_budget.py` read the cap from the C1 package config, which no grant
     after 2026-10-23 updated: $712.17 against the $740 every issuer used.
   - Fixed in code: `derive_budget.latest_project_cap` reads the newest grant
     that states a cap. No record was edited.
2. **One billed pod (`xrksmenhg7dlps`, $2.63) named by no record.**
   - It is the C3 packing pilot's subrun v6, which its campaign recorded under
     v5's pod id. The minutes and dollars agree to 0.1 min / $0.003.
   - Attributed by [`attribution_evidence.json`](attribution_evidence.json),
     whose numbers the audit re-checks. Its cost was already booked.
3. **Engineering-campaign subruns are not scanned for unposted pods.**
   - The current issuers look only in `runs/*/runtime/session.json`.
   - None is inside the posting window today, so today's position is complete.
   - The next study's issuer must count campaign subruns. This is owed before
     any paid engineering validation.
4. **Volume rows bill more GB-hours than any recorded lifetime can accrue.**
   - The excess is $0.34 of $2.55, including three September rows that belong
     to the C2 durable-staging volumes.
   - It stays in the bill and is never netted against a reservation.

**Unchanged:**
- No historical conclusion, authorization, grant or run record.
- No cap.
- The pre-anchor difference is reported, not booked.
