# Six-DOF Stratified Thruster FTC Results V2

## Scope

This report records the paired, balanced development benchmark used to reduce
vertical no-output FTC localization delay. It evaluates direct ESC current/RPM
evidence only; learned model output remains excluded from FTC.

The result is simulation-development evidence. It is not a hardware-in-the-loop
test, independent blind test, or real-sea reliability claim.

## Root cause

The original FTC gate required every thruster command to remain above a `0.20`
excitation ratio for the complete `0.50 s` no-output confirmation interval.
V1/V2 commands frequently crossed above `0.20` briefly and then dropped below
it. Direct no-output evidence therefore appeared quickly, but the confirmation
timer repeatedly reset. Horizontal commands were normally well above the same
gate and did not show this behaviour.

The deployed change keeps the horizontal threshold at `0.20` and sets only the
two vertical thresholds to `0.08`. Current and RPM must still both show at least
the configured no-output deficit, the score margin remains unchanged, and the
evidence must still persist for `0.50 s`.

## Candidate benchmark V1

The first locked benchmark used equal samples for H1--H4 and V1--V2, five paired
replicates, clean and sensor-stress contexts, ESC telemetry noise, and healthy
controls. It compared vertical thresholds `0.20`, `0.12`, and `0.08`.

| Metric | Baseline 0.20 | Vertical 0.08 |
| --- | ---: | ---: |
| Fault missions | 60 | 60 |
| Correct FTC target recall | 100.00% | 100.00% |
| Wrong-target missions | 0 | 0 |
| Healthy false-target missions | 0/10 | 0/10 |
| Vertical median target delay | 3.40 s | 0.55 s |
| Vertical P90 target delay | 3.82 s | 0.565 s |
| Horizontal median target delay | 0.55 s | 0.55 s |

The paired median vertical improvement was `2.85 s`; the maximum horizontal
delay change was exactly `0.0 s`.

## Deployed-default validation V2

After `0.08` became the default V1/V2 threshold, a second locked batch used new
seeds, different fault start times, and stronger current/RPM/voltage/temperature
noise. The evaluator verifies that its candidate configuration exactly equals
`FTCSupervisorConfig()` before running.

| Thruster | Baseline median | Deployed median | Deployed P90 |
| --- | ---: | ---: | ---: |
| H1 | 0.55 s | 0.55 s | 0.55 s |
| H2 | 0.55 s | 0.55 s | 0.61 s |
| H3 | 0.75 s | 0.75 s | 0.96 s |
| H4 | 0.55 s | 0.55 s | 0.60 s |
| V1 | 3.35 s | 0.55 s | 0.55 s |
| V2 | 3.35 s | 0.55 s | 0.55 s |

The deployed configuration again achieved:

- 60/60 correct no-output targets;
- 0 wrong-target missions;
- 10/10 healthy controls without isolation;
- 100% target recall in sensor-stress missions;
- no change to horizontal target delay.

## Safety interpretation

Lowering the vertical excitation gate does not remove the direct-evidence
requirements. A short vertical telemetry dropout still clears before the
`0.50 s` confirmation timer and cannot isolate a thruster. The change also does
not affect partial-thrust-loss policy: ambiguous thrust loss remains log/advice
only and cannot force component isolation.

The benchmark currently models independent Gaussian ESC telemetry noise. A
future hardware-in-the-loop test should add persistent communication faults,
shared power-bus disturbances, calibration errors, and real ESC quantization
before treating the threshold as hardware validated.

## Artifacts

- Candidate protocol: `docs/six_dof_thruster_stratified_protocol_v1.json`
- Candidate summary: `math-model/results/six_dof_thruster_stratified_v1_20260717/thruster_stratified_summary.json`
- Deployment protocol: `docs/six_dof_thruster_stratified_protocol_v2.json`
- Deployment summary: `math-model/results/six_dof_thruster_stratified_v2_20260717/thruster_stratified_v2_summary.json`
- Deployment rows: `math-model/results/six_dof_thruster_stratified_v2_20260717/thruster_stratified_v2_rows.csv`
- Delay comparison: `math-model/results/six_dof_thruster_stratified_v2_20260717/thruster_stratified_v2_delay.png`
