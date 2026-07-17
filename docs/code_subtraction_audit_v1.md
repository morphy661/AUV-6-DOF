# AUV six-DOF code subtraction audit V1

Date: 2026-07-17
Scope: structural simplification without changing diagnosis or FTC behavior

## Current runtime path

1. `SixDOFSimulator` produces vehicle, sensor, actuator, and ESC telemetry.
2. `SensorHealthMonitor` provides the fast direct sensor-safety path.
3. `SensorFaultObserver` provides long-horizon possible-fault hypotheses.
4. `SixDOFStateEstimator` rejects or degrades untrusted sensor sources.
5. `build_rule_based_ftc_evidence` and `FTCSafetySupervisor` own intervention decisions.
6. `SixDOFModelBridge`, `MaintenanceHealthDecision`, and `AdvisoryContextGate` provide non-authoritative learned maintenance advice.
7. The demo adapter and renderer expose the causal results to the operator.

This separation is intentional. The learned model must not be merged into the FTC intervention path, and the fast sensor monitor must not be replaced by the slower ambiguity observer.

## Keep as separate responsibilities

| Module or layer | Decision | Reason |
|---|---|---|
| Sensor health monitor | Keep | Fast confirmed safety action. |
| Sensor fault observer | Keep | Long-horizon possible classification; does not command FTC. |
| State estimator | Keep | Owns sensor exclusion and degraded navigation state. |
| FTC supervisor | Keep | Sole owner of automatic control intervention. |
| Advisory context gate | Keep | Prevents stale or context-contaminated model advice. |
| Maintenance health decision | Keep | Online learned advice contract. |
| Maintenance ticket/log policies | Keep, but classify as offline | They are used by calibration, blind evaluation, replay, and maintenance reporting rather than being extra runtime FTC layers. |

## Confirmed merge candidates

### Completed: ESC telemetry fault injection

Before this audit, the demonstration and ESC stress evaluator each implemented packet loss, communication freeze, telemetry age, and held-sample state independently.

The behavior now lives in one reusable module:

- `math-model/src/actuators/esc_telemetry_faults.py`

Both callers delegate to `ESCTelemetryFaultInjector`:

- `math-model/examples/demo_six_dof_unified_diagnostics.py`
- `math-model/examples/evaluate_six_dof_esc_telemetry_stress.py`

Call-site reduction:

- Demo script: 480 to 443 lines (-37).
- ESC stress evaluator: 539 to 490 lines (-49).
- Duplicate telemetry state machines: two to one.

The shared module adds validation for event IDs, thruster names, fault modes, time windows, telemetry vector sizes, finite values, and quantization steps. The first pass therefore reduces behavioral duplication rather than total repository line count.

### Next: versioned evaluation runners

The following versioned runners contain repeated protocol loading, hashing, iteration, CSV, plotting, and summary code:

- Sensor observer development/V2/V3.
- Unified random batch V1/V2/V3.
- Thruster stratified V1/V2.

They should eventually become parameterized runners with small protocol-specific summary functions. They are not changed in V1 because locked JSON protocols contain exact file hashes. Removing or rewriting them before creating a repository release tag would damage historical reproducibility.

### Completed: common locked-protocol utilities

The pre-refactor baseline was committed as `da60db4`. Nine evaluation runners now delegate their shared immutable-protocol preflight to:

- `math-model/src/evaluation/protocol.py`

The shared module owns stable JSON/file hashing, protocol identity and lock checks, code/artifact manifest verification, and output overwrite prevention. Mission-count, scenario-matrix, thruster-layout, timing, and acceptance rules remain in their experiment-specific runners.

Reduction in the active production code:

- Four local SHA-256 implementations reduced to one.
- Nine repeated hash/output preflight paths reduced to one.
- Nine runners reduced from 2,602 to 2,515 physical lines (-87).
- The common module is 70 physical lines, giving a net production reduction of 17 lines.

The existing locked JSON protocols and recorded outputs were not rewritten. A future execution after this refactor must use a new protocol version with fresh hashes and a new output directory; the pre-refactor runner sources remain available from commit `da60db4`.

## Archive candidates, not immediate deletions

- Intermediate result directories superseded by V2 or V3 locked results.
- Development-only observer and random-batch outputs.
- Old single-depth demonstrations after the six-DOF release is tagged.

Only generated artifacts should be archived first. Source files referenced by a locked protocol must remain available in the tagged baseline.

## Simplification constraints

Every subtraction must preserve:

- Zero targeted isolation for invalid or stale ESC telemetry.
- 60/60 real no-output recall in the locked simulation benchmark.
- The fixed demo sequence: V2 communication anomaly is log-only, followed by V1 targeted reallocation for the real no-output fault.
- Separation of privileged simulation truth from diagnostic presentation.
- Full automated regression success.

## Recommended order

1. Commit and tag the current reproducible baseline.
2. Introduce shared protocol/evaluation utilities.
3. Replace V1/V2/V3 runners with one current runner while moving historical scripts under a clearly marked locked archive.
4. Re-run the locked behavioral baselines under a new protocol version.
5. Only then review whether legacy single-depth runtime code can be moved out of the primary six-DOF package.
