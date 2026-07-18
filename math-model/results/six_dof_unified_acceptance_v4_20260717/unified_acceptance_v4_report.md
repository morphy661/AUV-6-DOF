# Six-DOF unified acceptance V4

Decision: **accepted**

This is a simulation evidence aggregation, not a real-sea result or a new independent blind test.

## Acceptance categories

| Category | Passed | Total | Result |
|---|---:|---:|---|
| source_integrity | 5 | 5 | PASS |
| sensor_diagnosis | 7 | 7 | PASS |
| esc_telemetry | 6 | 6 | PASS |
| thruster_ftc | 5 | 5 | PASS |
| unified_operations | 8 | 8 | PASS |
| maintenance_model | 5 | 5 | PASS |

## Non-gating model information

- Weak thrust-loss Top-2 location accuracy in unified random missions: `0.5555555555555556`
- Weak thrust-loss formal maintenance ticket recall: `0.06666666666666667`
- Frozen blind temporal fault-mode macro F1: `0.834989149545995`
- Frozen blind temporal exact-location macro F1: `0.5808177326156122`
- Raw temporal normal-window advisory false-alarm rate before maintenance policy: `0.29193109700815956`
- Formal ticket Top-2 probable-location hit rate: `0.96875`
- Formal maintenance ticket median delay in seconds: `4.92500114440918`

## Diagnostic boundary

- Confirmed sensor and complete thruster failures may trigger FTC.
- Weak thrust loss remains a recorded probability-based maintenance clue.
- Exact weak-fault thruster location is not an automatic safety claim.
