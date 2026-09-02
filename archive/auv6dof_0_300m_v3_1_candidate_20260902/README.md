# AUV 6-DOF 0–300 m V3.1 Candidate

This candidate repairs the V3 mass-mismatch false-alarm mechanism without
changing the rule-based FTC. Rigid-body mass uncertainty is limited to ±2%
around 50 kg, while the expected-response and motion-loss features use an
explicit onboard configured generalized-mass diagonal.

The candidate is not a final frozen release. It passed the current independent
simulation gates, but the remaining 150–300 m record-level false alarms still
require a separate cause audit.

Selected model: training seed 42, chosen by validation joint macro-F1.

Deployment must pass the calibrated vehicle generalized-mass diagonal to
`SixDOFModelBridge`. Omitting it uses the legacy nominal diagonal and is valid
only for the nominal 50 kg / nominal added-mass configuration.

| Independent-test metric | V3 | V3.1 candidate |
|---|---:|---:|
| Raw normal-window FAR | 12.61% | 7.86% |
| Temporal normal-window FAR | 12.39% | 7.79% |
| Joint macro-F1 | 0.7981 | 0.8237 |
| False formal tickets | 0 | 0 |
| No-output ticket recall | 100% | 100% |
| Thrust-loss ticket recall | 77.78% | 94.44% |

The 813+ MB generated dataset is intentionally excluded from Git. Its identity
and reproduction command are recorded in `source/dataset_record.json`.
