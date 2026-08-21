# AUV 6-DOF Stage Archive: 0-300 m Focus V3

Archive ID: `auv6dof_0_300m_v3_20260821`  
Created: `2026-08-21T17:56:44+09:00`  
Base commit: `4867c2ce41095c87cace45ac55d32e30b8f4a159`  
Branch/upstream: `main` / `auv6dof/main`

## Status

This is the current stage freeze for the six-DOF, six-thruster AUV simulation,
multi-sensor diagnosis, rule-based FTC, and BiLSTM-attention maintenance
advisory. It is suitable for professor review and abstract preparation.

It is **not** a locked final acceptance release, a hardware depth
certification, or sea-trial evidence. The historical V4 36/36 acceptance result
predates this V3 checkpoint and must not be attached to it as a current result.

## Frozen scope

- Primary operating range: 0-300 m; 300-500 m is 10% deep-stress coverage.
- Six thrusters: H1-H4 and V1-V2.
- Active control: surge, sway, heave, pitch, yaw; roll is passively stabilized.
- Sensors: depth, IMU, DVL, and ESC telemetry.
- Diagnosis grades: normal, log-only, possible, and confirmed.
- Only confirmed rule-based safety evidence can command FTC.
- BiLSTM-attention output remains maintenance advice.

## Key independent-test results

| Metric | Result |
|---|---:|
| Mode macro F1 | 0.8882 |
| Location macro F1 | 0.8052 |
| Joint macro F1 | 0.7981 |
| Raw normal-window false-alarm rate | 12.61% |
| Fault missions detected | 36/36 |
| Exact mode and location missions | 35/36 |
| Mean raw mode-detection delay | 0.85 s |
| Formal maintenance ticket precision | 100% |
| False formal maintenance tickets | 0 |
| No-output ticket recall | 100% |
| Thrust-loss ticket recall | 77.78% |
| Top-2 ticket location hit rate | 93.75% |

Regression check at archive time: **213 tests passed and 5 subtests passed**.

## Contents

- `model/`: deployable checkpoint and normalization arrays.
- `evaluation/`: independent-test metrics, class reports, confusion arrays,
  training history, and the result summary.
- `policy/`: selected temporal and formal-maintenance policy plus audit log.
- `demo/`: representative 200 m video, two static views, flat time series,
  and a compact causal event summary.
- `source/`: dataset identity, Git state, and the uncommitted code patch needed
  to reconstruct the exact archived working state from the base commit.
- `manifest.json` and `SHA256SUMS.txt`: integrity and provenance records.

The 813,858,561-byte training dataset and the 3.4 MB full per-frame replay JSON
are intentionally not duplicated. Their paths and SHA-256 hashes are recorded
in `source/dataset_record.json` and `demo/run_summary.json`.

## Reproduction

Configured Python interpreter:

```text
D:\Anaconda_envs\envs\auv_gpu\python.exe
```

Run the full regression suite:

```powershell
cd C:\Users\Administrator\PycharmProjects\AUV-Project\math-model
D:\Anaconda_envs\envs\auv_gpu\python.exe -m pytest tests
```

Regenerate the focused dataset, retrain, and render the current demo using the
commands documented in the repository root `README.md`. The source working tree
was not clean at archive time, so apply `source/working_tree_code_changes.patch`
to base commit `4867c2ce41095c87cace45ac55d32e30b8f4a159` when exact code reconstruction is required.

## Claim boundary

The archived scores measure independent missions sampled from the modeled,
randomized deployment domain. They do not establish generalization to arbitrary
out-of-distribution hydrodynamics or real ocean operation. Weak or intermittent
thrust loss is intentionally allowed to remain a logged hypothesis when safety
evidence is insufficient for intervention.
