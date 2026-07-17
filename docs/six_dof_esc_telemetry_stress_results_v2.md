# Six-DOF ESC telemetry stress validation V2

Date: 2026-07-17
Status: accepted for the simulation deployment baseline
Claim boundary: locked simulation validation only; this is not a real-sea or independent blind-test result.

## Purpose

The FTC no-output rule previously treated zero-filled ESC current and RPM data as physical measurements. A sustained packet loss could therefore look identical to a real thruster no-output fault and cause an unsafe targeted isolation.

The deployed rule now requires each ESC channel to be both valid and fresh before its current/RPM residual can qualify for targeted reallocation. Invalid or stale channels are logged as communication anomalies and cannot become isolation candidates.

## Locked V2 design

- Six thrusters: H1, H2, H3, H4, V1, V2.
- Two contexts: clean and randomized sensor stress.
- Five new replicates per context, using seeds starting at 20261900.
- 300 paired scenarios per strategy:
  - 240 healthy missions with ESC telemetry stress.
  - 60 real no-output fault missions.
- Stronger ESC noise than V1.
- Stress types: continuous packet loss, communication freeze, bus-voltage dip, and current/RPM quantization.
- Paired comparison: historical zero-fill behavior versus the deployed freshness-guarded default.

## V2 results

| Scenario | Missions | Legacy false isolation | Guarded false isolation | Guarded communication record | Guarded real-fault recall |
|---|---:|---:|---:|---:|---:|
| Continuous packet loss | 60 | 66.67% | 0% | 100% | N/A |
| Communication freeze | 60 | 0% | 0% | 100% | N/A |
| Bus-voltage dip | 60 | 0% | 0% | N/A | N/A |
| Quantization | 60 | 0% | 0% | N/A | N/A |
| Real no-output | 60 | N/A | N/A | N/A | 100% |

Additional deployed metrics:

- Overall telemetry-stress false isolation: 0/240.
- Real no-output correct target: 60/60.
- Real no-output wrong target: 0/60.
- Median real no-output target delay: 0.55 s.
- All seven V2 predeclared acceptance checks passed.
- Full regression: 183 tests passed plus 5 subtests.

The first locked V1 run also produced 0/240 guarded false isolations and 60/60 real-fault recall. Its overall acceptance flag was false only because the predeclared expectation that the legacy logic would fail at least 90% of continuous-loss cases was too aggressive; the observed legacy rate was 68.33%. The V1 artifact was retained and hash-locked into V2 rather than overwritten.

## Implemented interface contract

Every six-thruster log now carries:

- `thruster_telemetry_valid`: six Boolean channel-validity flags.
- `thruster_telemetry_age_s`: six non-negative ages measured from the last fresh packet.

The deployed default accepts direct no-output evidence only while a channel is valid and its age is at most 0.20 s. Invalid or stale channels receive a zero no-output score and an `esc_telemetry_guard` evidence marker. The supervisor returns `log_only` for an isolated ESC communication anomaly unless an independent, higher-priority safety condition is active.

For compatibility with older simulation logs, missing validity and age fields default to valid and fresh. A future CAN/serial/ROS hardware adapter must therefore populate both fields explicitly from packet status and timestamps; it must not represent packet loss only by filling current and RPM with zeros.

## Reproducible artifacts

- Protocol: `docs/six_dof_esc_telemetry_stress_protocol_v2.json`
- Evaluator: `math-model/examples/evaluate_six_dof_esc_telemetry_stress.py`
- Summary: `math-model/results/six_dof_esc_telemetry_stress_v2_20260717/esc_telemetry_stress_summary.json`
- Per-scenario rows: `math-model/results/six_dof_esc_telemetry_stress_v2_20260717/esc_telemetry_stress_rows.csv`
- Plot: `math-model/results/six_dof_esc_telemetry_stress_v2_20260717/esc_telemetry_stress.png`

## Unified operator demonstration integration

The final operator demonstration now includes a separate ESC-link event before the physical thruster fault:

- 15.35-16.35 s: V2 telemetry is unavailable for 21 simulator frames. The display marks `ESC telemetry unavailable`, the FTC action remains `log_only`, and no thruster target is selected.
- ESC onset and recovery both reset the learned-advice context. Model output remains withheld until a fresh post-transition window is available, so zero-filled communication data is not presented as a physical thruster diagnosis.
- 17.00 s: the independent V1 physical no-output fault starts.
- 17.55 s: direct valid current/RPM evidence confirms V1 and the FTC enters `targeted_reallocation`.

The generated H.264 demonstration is 1280 x 720, 12 fps, and 15 seconds long. Its source simulation contains 440 causal frames over 22 seconds and the rendered video uses 180 evenly sampled frames.

- Video: `math-model/results/six_dof_unified_diagnostics_esc_link_v3/six_dof_unified_diagnostics.mp4`
- Physical-fault snapshot: `math-model/results/six_dof_unified_diagnostics_esc_link_v3/six_dof_unified_diagnostics.png`
- ESC-link snapshot: `math-model/results/six_dof_unified_diagnostics_esc_link_v3/six_dof_unified_diagnostics_esc_link.png`
- Presentation JSON: `math-model/results/six_dof_unified_diagnostics_esc_link_v3/six_dof_unified_diagnostics.json`
- Presentation CSV: `math-model/results/six_dof_unified_diagnostics_esc_link_v3/six_dof_unified_diagnostics.csv`
- Final regression: 186 tests passed plus 5 subtests.
