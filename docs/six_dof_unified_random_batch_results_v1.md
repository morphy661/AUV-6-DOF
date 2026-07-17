# Six-DOF Unified Random Batch V1 Results

## Scope

This report records the predefined 30-mission development batch for the unified
six-DOF diagnostic and FTC demonstration. It evaluates randomly injected sensor
and thruster faults using seeds `20260800` through `20260829`.

This is a reproducible simulation-development result. It is not an independent
blind test, hardware-in-the-loop validation, or evidence of real-sea reliability.

The frozen protocol is stored in
`docs/six_dof_unified_random_batch_protocol_v1.json`. The protocol hashes the
evaluator, diagnostic/FTC implementation, feature adapter, simulator, and frozen
BiLSTM-Attention artifacts used for the run.

## Main result

All predefined V1 acceptance checks passed. The safety-critical direct-evidence
and rule-based FTC path performed reliably in this batch, while the learned
thruster advisory remained suitable only as a non-authoritative operator hint.

### Sensor diagnostics and FTC

| Metric | Result |
| --- | ---: |
| Weak-spike over-promotion rate | 0.00% |
| Ambiguous-fault record rate | 96.67% (29/30) |
| Ambiguous operator-possible rate | 86.67% |
| Intermittent sample confirmation rate | 100.00% |
| Intermittent root-possible rate | 100.00% |
| Sensor FTC action-match rate | 100.00% |

One depth-drift case was not retained as an ambiguous record. This is a weak,
non-safety-critical miss under the current policy, but it should remain visible
in later regression comparisons.

### Thruster evidence and FTC

| Metric | Result |
| --- | ---: |
| No-output direct-evidence recall | 100.00% (18/18) |
| No-output target recall | 100.00% (18/18) |
| Wrong-thruster target mission rate | 0.00% |
| Thrust-loss wrong-isolation rate | 0.00% |
| Safe FTC behaviour rate | 100.00% (30/30) |
| Median no-output evidence delay | 0.095 s |
| Median no-output target delay | 0.757 s |

Direct no-output evidence can therefore drive FTC in this development batch.
Weak thrust loss is intentionally not force-isolated by the safety path when the
evidence is ambiguous.

### Frozen BiLSTM-Attention advisory

| Metric | All missions | No-output subset | Thrust-loss subset |
| --- | ---: | ---: | ---: |
| Peak fault-mode accuracy | 100.00% | 100.00% | 100.00% |
| Peak group accuracy | 60.00% | 100.00% | 0.00% |
| Peak Top-1 thruster accuracy | 60.00% | 100.00% | 0.00% |
| Peak Top-2 thruster accuracy | 76.67% | 100.00% | 41.67% |

The learned model produced an advisory before the injected thruster fault in
30/30 missions. Earlier sensor faults and normal manoeuvre context can therefore
activate the raw learned advisory. This does not cause an unsafe FTC action,
because learned localization is displayed only as `possible` and is not allowed
to command FTC. It does mean that peak post-fault accuracy alone overstates the
model's practical maintenance-alert reliability.

## Interpretation

The current system has two distinct performance levels:

1. Direct sensor/no-output evidence plus rule-based FTC is the reliable safety
   layer in this simulation batch.
2. BiLSTM-Attention localization is strong for obvious no-output faults but weak
   for partial thrust loss. It should continue to provide a ranked possibility
   and supporting log, not a confirmed component diagnosis.

Consequently, the next optimization should not loosen FTC thresholds merely to
raise a model accuracy number. It should add context gating to the learned
advisory so that active sensor faults and expected waypoint/manoeuvre transients
do not create premature thruster-maintenance prompts. A V2 protocol should then
use new frozen random seeds and explicitly require a low pre-thruster advisory
rate while preserving the V1 no-output and FTC safety results.

## Artifacts

- Summary: `math-model/results/six_dof_unified_random_batch_v1_20260717/unified_random_batch_summary.json`
- Per-mission data: `math-model/results/six_dof_unified_random_batch_v1_20260717/unified_random_batch_missions.csv`
- Summary chart: `math-model/results/six_dof_unified_random_batch_v1_20260717/unified_random_batch_summary.png`
- Locked protocol: `docs/six_dof_unified_random_batch_protocol_v1.json`
- Evaluator: `math-model/examples/evaluate_six_dof_unified_random_batch.py`
