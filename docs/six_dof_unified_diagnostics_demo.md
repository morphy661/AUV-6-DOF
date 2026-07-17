# Six-DOF unified diagnosis demonstration

## Purpose

This stage closes the loop from six-DOF simulation logs to an operator-facing
video. The demonstration does not add a new classifier or change FTC rules. It
adapts the already validated sensor monitor, long-horizon observer, state
estimator, ESC evidence, and FTC supervisor into one synchronized presentation.
The frozen BiLSTM-Attention checkpoint is now connected as a separate causal
maintenance-advice channel.

## Causal display contract

The presentation tiers are:

| Tier | Meaning | Typical output |
|---|---|---|
| `confirmed` | Directly observable fault signature or completed FTC isolation | Sensor unavailable/stuck/strong spike; targeted thruster reallocation |
| `possible` | Persistent but ambiguous evidence | Bias/drift, partial stuck, intermittent root cause, pre-confirmation no-output evidence |
| `log_only` | Weak or transient evidence retained for later inspection | Single weak spike and low-strength dropout evidence |
| `normal` | No active diagnostic evidence | Normal operation |

Sensor tiers use only `sensor_health` and `sensor_fault_observations`. Thruster
tiers use commanded force, force limits, expected/measured current and RPM,
`ftc_no_output_scores`, and the FTC decision. The adapter never reads
`sensor_fault_truth`, `thruster_fault_modes`, `faulted_thruster_index`, actual
thrust, or injected effectiveness when producing a diagnosis. Simulation truth
is used only to draw the vehicle motion.

The learned model consumes 50 past/current samples of the established 109-field
observable feature vector plus temporal differences. It updates every ten
simulation samples and holds the last result between updates. Its fault mode,
group, and Top-2 candidates are always rendered as `possible`; even an internal
100% probability does not become a confirmed fault and cannot drive FTC.

## Demonstration sequence

The deterministic 22-second mission contains:

1. a weak depth spike, retained as a background log;
2. a DVL channel bias, displayed as a possible bias/drift diagnosis;
3. three short IMU-unavailable intervals, each directly confirmed while the
   invalid sample is present and grouped as a possible intermittent root cause;
4. a V1 no-output fault, first shown as causal ESC evidence and then as a
   confirmed target after the FTC supervisor isolates V1.

The target changes in north, east, depth, and yaw so the video shows coupled
six-DOF motion while preserving the six-thruster architecture.

This fixed sequence is the regression reference, not the only input mode.
`--injection-mode random` uses the supplied seed to randomize which sensor gets
the weak, ambiguous, and intermittent event roles; bias versus drift; event
times; failed thruster; no-output versus thrust-loss mode; and thrust-loss
severity. The full generated schedule is saved as `injection_manifest`, so a
random video is reproducible without exposing truth to online diagnosis.

The presentation demo deliberately keeps nominal vehicle dynamics. The broader
training and stress generators continue to randomize vehicle, sensor, actuator,
noise, and disturbance parameters for robustness evaluation.

## Run command

```powershell
cd C:\Users\Administrator\PycharmProjects\AUV-Project\math-model
D:\Anaconda_envs\envs\auv_gpu\python.exe examples\demo_six_dof_unified_diagnostics.py
```

Randomized but reproducible example:

```powershell
D:\Anaconda_envs\envs\auv_gpu\python.exe examples\demo_six_dof_unified_diagnostics.py `
  --injection-mode random --seed 20260718 `
  --output-dir results\six_dof_unified_diagnostics_random_seed_20260718
```

If FFmpeg is not already installed:

```powershell
D:\Anaconda_envs\envs\auv_gpu\python.exe -m pip install imageio-ffmpeg
```

Use `--skip-video` for a fast JSON/CSV/static-image validation,
`--disable-model` for the rule/FTC-only view, or `--max-video-frames` and
`--fps` to change encoding cost.

## Outputs and current result

The generated directory is
`math-model/results/six_dof_unified_diagnostics_demo/`:

- `six_dof_unified_diagnostics.mp4`: 1280x720 H.264, 12 fps, 20 seconds;
- `six_dof_unified_diagnostics.png`: the first confirmed V1 isolation frame;
- `six_dof_unified_diagnostics.json`: 440 adapted frames, grouped events, and
  summary;
- `six_dof_unified_diagnostics.csv`: flat pose, tier, FTC, and per-thruster
  score trace plus model fault probability and Top-2 candidates.

The current deterministic run targets V1 correctly. It includes confirmed IMU
sample-unavailability events, possible DVL and IMU root-cause messages,
background weak-event logs, and FTC actions spanning normal control, log-only,
safe hold/abort, and targeted reallocation. The MP4 was fully decoded after
generation to verify that all 240 frames are readable.

The recorded random example uses seed `20260718`: DVL weak spike, depth drift,
three IMU dropouts, and H3 no-output starting at 16.806 s. ESC evidence and FTC
correctly target H3 at 17.65 s. The model simultaneously reports possible
no-output and a horizontal group; its Top-2 probabilities remain advisory.
The random MP4 is under
`math-model/results/six_dof_unified_diagnostics_random_seed_20260718/`.

## Verification

`math-model/tests/test_six_dof_demo_adapter.py` checks tier priority, direct
confirmation precedence, truth-field isolation, causal thruster targeting,
event recovery, and grouping of intermittent no-output candidates. These tests
protect the presentation layer from reintroducing label leakage or converting
brief evidence into repeated operator alarms.

`math-model/tests/test_six_dof_demo_scenario.py` additionally verifies that a
random seed reproduces the exact injection manifest, different seeds change
the schedule, and the fixed V1 sequence remains unchanged. The current full
regression is 166 tests plus five subtests.
