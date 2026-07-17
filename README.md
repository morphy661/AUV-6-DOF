# AUV Project

This repository contains the related AUV modeling, simulation, and depth sensor fault detection work.

## Project Structure

- `math-model/` - AUV mathematical model, simulation utilities, sensor/fault modules, and demo scripts.
- `depth-sensor-fault-detection/` - Depth sensor fault detection model training, inference, and generated model outputs.

## Six-thruster 6-DOF model

The default actuator layout follows the KYUBIC/Tuna-Sand2 architecture:

- four horizontal thrusters (`H1`-`H4`) actively control surge, sway, and yaw;
- two vertical thrusters (`V1`-`V2`) actively control heave;
- the vehicle dynamics retain all six degrees of freedom;
- roll and pitch are passively stabilised by hydrostatic restoring moments.

Thruster commands, forces, currents, efficiencies, and fault modes therefore
use six-element arrays in the order `H1, H2, H3, H4, V1, V2`.

The complete single-thruster validation covers 13 scenarios: one nominal run
plus no-output and thrust-loss faults for each of the six thrusters. Run it with:

```powershell
cd math-model
python examples/demo_six_dof_fault_coverage.py
```

It produces a summary table, a six-axis residual table, per-scenario source
logs, and a comparison figure under `math-model/results/six_dof_fault_coverage/`.

An oracle FTC benchmark can then compare the baseline with instantaneous,
perfectly known thruster-effectiveness reallocation:

```powershell
python examples/demo_six_dof_ideal_ftc.py
```

This benchmark is a control-performance upper bound, not a deployable fault
diagnoser: the rule-based or learned detector must supply the health estimate.

`math-model/src/ftc/safety_supervisor.py` provides the deployable safety gate
between health evidence and the allocator. It never reads injected fault labels,
actual thrust, or true effectiveness. Compensated thrust loss is log-only;
targeted reallocation requires persistent per-thruster current-and-RPM dropout
under sufficient command excitation. Uncertain faults can request degraded
operation, mission abort, or controlled ascent only after independent control
stress persists. The oracle FTC remains available solely as an upper bound.

## Six-DOF sensor health and FTC guard

The synchronized depth, IMU, and DVL interface now supports deterministic
`unavailable`, `stuck`, and `spike` fault schedules. Injected fault truth is
kept in a simulation-only field; the online monitor classifies each sensor
from measured values, validity flags, controller motion demand, and independent
onboard evidence. Its per-step output includes the fault type, confidence,
trust level, confirmation state, and recommended response.

The FTC supervisor consumes only the observable health summary:

- a one-sample spike is rejected and logged without guessing a thruster;
- confirmed depth or DVL loss requests degraded navigation;
- confirmed IMU loss or stuck attitude data requests safe hold or abort;
- a specific thruster is isolated only when independent ESC current and RPM
  evidence directly confirms no output.

These outputs are stored in every six-DOF simulation log together with pose,
thruster telemetry, and FTC decisions. The unified MP4 dashboard now shows the
3D trajectory and attitude, live three-tier sensor diagnosis, six-thruster ESC
evidence, estimator state, FTC actions, and the event timeline from the same
causal log stream.

When a synchronized sensor suite is present, the controller now uses the
previous causal depth/IMU/DVL state estimate by default. Spikes are rejected,
depth or DVL loss activates dead reckoning, and IMU loss holds the last
attitude while the FTC requests a safe response. Simulator truth remains in
separate evaluation fields and can be selected explicitly only by setting
`use_sensor_feedback=False`. The lightweight estimator establishes the causal
closed-loop interface; it is not yet a calibrated production INS/EKF. See
`docs/six_dof_sensor_fault_and_video_roadmap.md` for the staged integration.

Run the independent development matrix for normal, unavailable, stuck, and
spike behavior across depth, IMU, and DVL with:

```powershell
python math-model/examples/evaluate_six_dof_sensor_faults.py --strict
```

The current 50-mission development run achieved 100% event precision, event
recall, FTC action matching, and sensor-health recovery, with no normal-mission
protective action and no thruster-location guess. Absolute trajectory recovery
was 73.33%, because persistent IMU/DVL outages can leave an unobservable
horizontal dead-reckoning offset. The estimator therefore keeps navigation
degraded until an external horizontal position fix is supplied. These are
development results, not a frozen blind-test or real-sea accuracy claim. See
`docs/six_dof_sensor_fault_benchmark_results.md` for the full breakdown.

## Leakage-safe six-DOF diagnosis data

The next-generation diagnosis path uses 109 raw inference-time observable
features (218 after first differences): depth, DVL, IMU, six commanded forces,
ESC current/RPM/voltage/temperature telemetry, allocator saturation flags,
commanded body wrench, six local thruster-health scores, and six projected
command-to-motion loss scores. Simulator truth such as actual thrust,
effectiveness, and fault state is isolated in the label path and cannot enter
the feature vector.

`math-model/src/utils/six_dof_dataset_builder.py` creates causal windows that
never cross mission boundaries and provides scenario-stratified mission splits.
Generate the first six-thruster dataset with:

```powershell
python math-model/examples/generate_six_dof_fault_dataset.py
```

The default generator creates 20 missions per scenario, uses five-second
windows, randomizes vehicle/sensor/actuator parameters, and reserves fixed
held-out seeds with out-of-domain physical parameters for the test split.

The current hybrid-telemetry ablation and pressure-test results are documented
in `docs/six_dof_hybrid_telemetry_results.md`.

The depth/IMU/DVL diagnostic boundary was also evaluated in a hash-locked
75-mission sensor stress benchmark. Strong unavailability, full-channel stuck,
and strong spike faults achieved 100% event recall and precision with zero
normal false confirmations. Weak spikes, bias, drift, and partial-channel
faults remain possible/log-only diagnoses; brief unavailability separates the
certain current-sample state from the uncertain hardware root cause. The
immutable protocol and results are documented in
`docs/six_dof_sensor_fault_stress_results_v1.md` and preserved under
`math-model/results/six_dof_sensor_fault_stress_v1_20260717`.

The follow-up causal observation layer adds log-only weak-jump evidence,
multi-second bias/drift consistency residuals, per-channel partial-stuck
tracking, recovery rebaselining, and repeated-unavailability grouping. It is
strictly separated from FTC and cannot confirm a hardware failure or select a
thruster. The frozen 75-mission V2 retained 100% strong-fault recall/precision,
zero normal operator prompts, 100% coverage of policy-required possible
scenarios, and zero FTC leakage. V2 remains formally failed because one real
depth-drift mission was displayed as possible although the frozen protocol had
required all depth drift to stay log-only. The unmodified outcome and revised
policy rationale are documented in
`docs/six_dof_sensor_fault_observer_results_v2.md`.

V3 kept the detector unchanged and froze a corrected three-tier presentation
policy: nine ambiguous scenarios require a possible message, depth/DVL slow
drift may be possible or log-only depending on evidence, and single weak
spikes must remain background logs. A new 75-mission one-shot run passed all
13 checks: 100% strong-fault recall/precision, zero normal operator prompts,
100% required-possible coverage, zero weak-spike overpromotion, and no FTC or
thruster-target leakage. See `docs/six_dof_sensor_fault_observer_results_v3.md`
and `math-model/results/six_dof_sensor_fault_observer_v3_20260717`.

## Unified six-DOF diagnosis video

Generate the deterministic end-to-end demonstration with the configured AUV
environment:

```powershell
cd math-model
D:\Anaconda_envs\envs\auv_gpu\python.exe examples\demo_six_dof_unified_diagnostics.py
```

The fixed schedule is the default because it gives a reproducible regression
story. A seed-reproducible randomized demonstration is also available:

```powershell
D:\Anaconda_envs\envs\auv_gpu\python.exe examples\demo_six_dof_unified_diagnostics.py `
  --injection-mode random --seed 20260718 `
  --output-dir results\six_dof_unified_diagnostics_random_seed_20260718
```

Random mode changes the weak/ambiguous/intermittent sensor assignments,
ambiguous mode, event timing, failed thruster, actuator fault mode, and thrust
loss severity. The exact truth schedule is written only to the offline
`injection_manifest`; online diagnosis never reads it. Full vehicle-parameter
domain randomization remains in the dataset and stress generators rather than
the presentation demo.

The renderer accepts a system `ffmpeg`, `FFMPEG_PATH`, or the binary supplied
by `imageio-ffmpeg` (`python -m pip install imageio-ffmpeg`). It writes a
1280x720 MP4, a static dashboard image, causal per-frame JSON, and a flat CSV
under `math-model/results/six_dof_unified_diagnostics_demo/`.

The demonstration sequences a weak depth spike (background log), DVL bias
(possible diagnosis), repeated IMU loss (certain sample unavailability plus
possible intermittent root cause), and a V1 no-output fault. V1 is shown as a
candidate only from observable ESC evidence and is marked confirmed only after
the FTC supervisor performs targeted reallocation. Injected sensor truth,
actual thrust, true effectiveness, and simulator fault labels are excluded from
the diagnostic adapter. See `docs/six_dof_unified_diagnostics_demo.md`.

The frozen BiLSTM-Attention checkpoint now runs causally on 50-sample windows
and adds fault-mode probability plus Top-2 inspection candidates. Its output is
always labelled `possible`, even when the internal probability is high; it is
maintenance advice and cannot command FTC or isolate a thruster. Use
`--disable-model` to render the rule/FTC-only view.

`math-model/src/diagnosis/temporal_fault_decision.py` adds a causal
normal/suspected/confirmed/recovering state machine. Calibrate its thresholds
on validation missions with:

```powershell
python depth-sensor-fault-detection/depth_fault_detection/calibrate_six_dof_temporal_decision.py
```

Evaluate recovered short-current and turbulence pulses with no injected
thruster fault:

```powershell
python depth-sensor-fault-detection/depth_fault_detection/evaluate_six_dof_transient_recovery.py --strict
```

The transient benchmark permits raw health observations but fails if the
vehicle does not recover, a formal maintenance ticket is opened, or the FTC
supervisor requests reallocation, isolation, abort, or controlled ascent.

Find the discrete no-intervention boundary for 1/2/4-second weak, medium, and
strong multi-axis disturbances plus scheduled DVL loss:

```powershell
python depth-sensor-fault-detection/depth_fault_detection/evaluate_six_dof_safety_boundary.py
```

The boundary report separates recovered log-only cases, false maintenance
tickets, FTC interventions, and genuinely unrecovered missions instead of
assuming every strong disturbance should pass without a protective action.
The FTC supervisor uses separate critical timers: 2 seconds when critical
control stress is corroborated by thruster-fault evidence, and 5 seconds for
stress alone. A clearly decaying stress-only excursion receives another
recovery interval; direct ESC current/RPM no-output isolation remains at
0.5 seconds.

`math-model/src/diagnosis/maintenance_health_decision.py` converts the
temporal output into four operational levels: normal, transient observation,
persistent degradation, and critical fault. Persistent but compensated thrust
loss and uncertain anomalies are retained in the raw health log. The separate
`maintenance_ticket_policy.py` opens a formal inspection ticket only when the
diagnosis has sufficient command excitation and independent motion or local
ESC evidence. Direct no-output evidence uses the fast path. Thrust-loss
evidence stays pending until it accumulates 8 qualified seconds inside one
stable guidance context. Target/waypoint transitions split pending evidence,
unstable-context windows do not confirm a ticket, and recovered recurrences
remain intermittent advisories instead of becoming formal tickets. A 3.75
second recovery still closes the pending episode while preserving its
observation. Location is advisory and reported as a
horizontal/vertical/uncertain group plus Top-2 inspection candidates. Short
same-mode ticket segments are merged into one incident. The calibration
command selects both layers on validation missions and writes raw events,
pending observations, advisories, and formal tickets to
`maintenance_event_log.json` beside the temporal summary.

The detector and decision policy were then locked in the one-shot protocol
`docs/six_dof_final_blind_protocol.json` before generating 65 new OOD missions.
That V1 audit retained 60/60 fault missions in the raw log and opened 30/30
no-output tickets, but failed its strict zero-false-ticket criterion because
one normal mission produced a recurrent thrust-loss ticket. The failed result
is preserved under `results/six_dof_final_blind_20260716`.

The context-aware revision was frozen separately in
`docs/six_dof_final_blind_protocol_v2.json` and evaluated once with a new seed
namespace. V2 again logged 60/60 fault missions and opened 30/30 no-output
tickets, while reducing formal false maintenance tickets to 0 and passing all
three predeclared acceptance checks. Compensated thrust loss remains log-first:
2/30 missions opened a formal ticket, while the remaining evidence stayed in
the raw log or observation records. The immutable V2 result is under
`results/six_dof_final_blind_v2_20260717`.

The post-V2 presentation layer in `maintenance_log_policy.py` does not change
the detector, tickets, or FTC. It retains every raw event, merges separated
same-mode episodes only within the same guidance context and a five-second
gap, then grades them as background trace, collapsed observation, maintenance
advisory, or safety alert. A retrospective replay reproduced the archived V2
metrics before applying this layer: 109 raw events became 102 grouped events
(28 traces, 42 observations, 2 maintenance advisories, and 30 safety alerts).
Only the final 32 require operator attention, with zero false attention events;
the replay is a presentation audit, not a new blind-test claim.

Train the multi-task BiLSTM-attention detector after generation with:

```powershell
python depth-sensor-fault-detection/depth_fault_detection/train_six_dof_multitask.py
```

The shared temporal encoder has separate attention pooling and heads for three
fault modes and six thruster locations. The location head is trained only on
fault windows; normal/fault gating supplies the seventh `none` state. Exact
six-way location and the derived 13-class label are retained for research
comparison, but they are no longer deployment acceptance criteria.

The staged research plan and readiness gates are documented in
`docs/six_dof_diagnosis_optimization_strategy.md`.

## Notes

The generated dataset `depth-sensor-fault-detection/depth_fault_detection/data/simulation_dataset.pth` is not tracked because it is larger than GitHub's normal 100 MB file limit.
