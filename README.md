# AUV 6-DOF

This repository contains the current six-degree-of-freedom AUV simulator,
causal fault diagnosis, fault-tolerant control (FTC), and the supporting
BiLSTM-attention maintenance model.

## Current configuration

- State: 3D NED position, scalar-first quaternion attitude, and body-frame
  linear/angular velocity.
- Actuation: six thrusters in the order `H1, H2, H3, H4, V1, V2`.
- Control: `H1-H4` control surge, sway, and yaw; `V1-V2` control heave and
  pitch; roll is passively stabilised.
- Primary simulation range: 0-300 m, with 1.2-1.5 m/s horizontal cruise and
  0.15-0.35 m/s vertical motion.
- Deep-stress range: 300-500 m, limited to 10% of generated missions.
- Nominal simulated thrust limit: 40 N per thruster, with training-time domain
  randomisation.

The 500 m value is a simulation envelope, not a hardware depth certification.
Vehicle pressure-hull, seals, penetrators, sensors, batteries, and thrusters
must all be independently rated before a real deployment.

## Repository layout

```text
math-model/
  examples/        runnable simulation, dataset, benchmark, and video commands
  src/             dynamics, control, sensors, diagnosis, FTC, and rendering
  tests/           automated regression suite
  results/         selected frozen evidence and generated demo outputs
depth-sensor-fault-detection/depth_fault_detection/
  data/            generated datasets (not stored in Git)
  results/         checkpoints, calibration, and evaluation artifacts
docs/              locked protocols, final reports, and historical evidence
archive/           curated stage freezes with integrity and provenance records
```

## Environment and regression test

The configured interpreter on the development machine is:

```text
D:\Anaconda_envs\envs\auv_gpu\python.exe
```

Run the complete test suite from `math-model`:

```powershell
cd C:\Users\Administrator\PycharmProjects\AUV-Project\math-model
D:\Anaconda_envs\envs\auv_gpu\python.exe -m pytest tests
```

## Run the current diagnosis and FTC video

```powershell
cd C:\Users\Administrator\PycharmProjects\AUV-Project\math-model
D:\Anaconda_envs\envs\auv_gpu\python.exe examples\demo_six_dof_unified_diagnostics.py
```

The default output is:

```text
math-model/results/six_dof_unified_diagnostics_demo/
```

It contains the MP4 dashboard, a static image, causal JSON, and a flat CSV.
The fixed scenario is the reproducible default. A seeded random injection run
is available with:

```powershell
D:\Anaconda_envs\envs\auv_gpu\python.exe examples\demo_six_dof_unified_diagnostics.py `
  --injection-mode random --seed 20260718 `
  --output-dir results\six_dof_unified_diagnostics_random_seed_20260718
```

The online dashboard does not read injected truth. Strong directly observable
sensor failures and complete thruster no-output faults can be confirmed;
ambiguous sensor behavior and weak/intermittent thrust loss remain possible
maintenance hypotheses. Only the rule-based safety supervisor can command FTC.

The default demo now loads the `0_300m_focus_v3` checkpoint and its independently
calibrated temporal configuration, and the fixed presentation mission runs near
200 m depth. The frozen V4 `36/36` badge belongs to the older baseline, so it is
hidden by default; add `--show-acceptance-badge` only when intentionally
presenting that historical evidence.

The frozen V4 demonstration and its explanation are retained at:

- `math-model/results/six_dof_unified_diagnostics_final_v4_20260718/`
- `docs/six_dof_unified_diagnostics_final_v4.md`

## Generate 0-300 m focused training data and retrain

Generate the new leakage-safe dataset:

```powershell
cd C:\Users\Administrator\PycharmProjects\AUV-Project\math-model
D:\Anaconda_envs\envs\auv_gpu\python.exe examples\generate_six_dof_fault_dataset.py
```

The default output is:

```text
depth-sensor-fault-detection/depth_fault_detection/data/
simulation_dataset_six_dof_hybrid_telemetry_0_300m_focus_v3.pth
```

Then retrain the multi-task BiLSTM-attention model by supplying that dataset to
the trainer:

```powershell
cd C:\Users\Administrator\PycharmProjects\AUV-Project
D:\Anaconda_envs\envs\auv_gpu\python.exe `
  depth-sensor-fault-detection\depth_fault_detection\train_six_dof_multitask.py `
  --dataset depth-sensor-fault-detection\depth_fault_detection\data\simulation_dataset_six_dof_hybrid_telemetry_0_300m_focus_v3.pth `
  --results-dir depth-sensor-fault-detection\depth_fault_detection\results\six_dof_hybrid_telemetry_0_300m_focus_v3
```

The existing `six_dof_hybrid_telemetry` checkpoint and frozen V4 evidence were
trained/evaluated before the new depth-weighted mission update. They remain a
historical baseline and must not be presented as validated deep-sea performance.

The primary accuracy test uses independent mission seeds drawn from the same
broad randomized deployment domain as training. Compound parameter ranges that
are entirely outside training are treated as a separate OOD stress test, not as
the main accuracy score.

## Current validation and historical evidence

The deployment-oriented simulation acceptance aggregator is:

```powershell
cd C:\Users\Administrator\PycharmProjects\AUV-Project\math-model
D:\Anaconda_envs\envs\auv_gpu\python.exe examples\evaluate_six_dof_unified_acceptance_v4.py
```

Its locked historical baseline passed 36/36 checks. It aggregates existing
sensor-observer, ESC-telemetry, thruster-FTC, random-batch, and blind-test
artifacts; it is not a new independent sea trial.

Key documents:

- `docs/six_dof_0_300m_focus_v3_results.md` - current dataset, model, depth,
  and maintenance metrics.
- `docs/six_dof_unified_acceptance_protocol_v4.json` - locked acceptance rules.
- `docs/six_dof_hybrid_telemetry_results.md` - model and temporal-policy results.
- `docs/six_dof_sensor_fault_observer_results_v3.md` - three-tier sensor output.
- `docs/six_dof_esc_telemetry_stress_results_v2.md` - ESC evidence limits.
- `docs/code_subtraction_audit_v1.md` - why the remaining runtime layers stay
  separate and which historical runners are archival.

Older protocol/result files under `docs/` are retained only for traceability.
They are not alternative current entry points.

## Current stage archive

The current 0-300 m Focus V3 checkpoint, evaluation evidence, temporal policy,
representative 200 m presentation, and source-state record are frozen at:

- `archive/auv6dof_0_300m_v3_20260821/`

This is a stage freeze for professor review and abstract preparation, not a new
locked acceptance release or sea-trial claim. Raw generated runs remain in their
original result directories and are excluded from the curated archive unless
needed as representative evidence.

## Repository hygiene

Generated `.pth` datasets are intentionally excluded from Git because they are
large and reproducible from the generators. Python caches, Pytest caches, IDE
metadata, and local plots/videos should not be committed. Keep only the current
working checkpoint, locked acceptance evidence, and presentation output needed
for the research record.
