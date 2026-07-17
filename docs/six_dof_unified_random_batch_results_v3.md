# Six-DOF Unified Random Batch V3 Results

## Scope

This report records the development-only evaluation of causal context gating
for learned thruster-maintenance advice. Each version used a separately locked
30-mission random seed batch:

- V1: no learned-advice context gate;
- V2: three-second stabilization after sensor/guidance transitions;
- V3: the V2 gate plus a required fresh model inference before release.

These results are reproducible simulation evidence. They are not an independent
blind test, hardware-in-the-loop validation, or real-sea reliability evidence.

## Why V2 did not pass

V2 reduced pre-thruster advice from 30/30 missions to 2/30 missions, but its
predeclared maximum was 1/30. Both failures occurred at 12.10--12.15 s. The
three-second timer had expired, so a model result computed while the gate was
still active became visible as a held result immediately before the first
intermittent sensor dropout.

V3 fixes the state-machine boundary: an expired timer remains gated until the
model performs a new inference on post-transition data. The two V2 failure seeds
were checked as development regressions before V3 was locked; both changed from
pre-fault advice to no pre-fault advice while preserving post-fault records.

## Locked V3 result

V3 used seeds `20261000` through `20261029`. All predefined acceptance checks
passed.

| Metric | V1 | V2 | V3 |
| --- | ---: | ---: | ---: |
| Pre-thruster advisory rate | 100.00% | 6.67% | 0.00% |
| Post-fault advisory rate | 100.00% | 100.00% | 100.00% |
| No-output FTC target recall | 100.00% | 100.00% | 100.00% |
| Safe FTC behaviour | 100.00% | 100.00% | 100.00% |
| Model Top-2 location accuracy | 76.67% | 83.33% | 86.67% |

V3 additional results:

| Metric | Result |
| --- | ---: |
| Weak-spike over-promotion | 0.00% |
| Ambiguous sensor-fault record rate | 93.33% |
| Intermittent sample confirmation | 100.00% |
| Sensor FTC action match | 100.00% |
| Wrong thruster target missions | 0/30 |
| Thrust-loss wrong isolation missions | 0/9 |
| No-output Top-2 location accuracy | 100.00% |
| Thrust-loss Top-2 location accuracy | 55.56% |
| Median first post-fault model-advice delay | 1.57 s |

The learned output therefore remains a ranked possibility for partial thrust
loss. The result does not justify confirmed component localization for that
fault mode.

## No-output FTC delay interpretation

The aggregate V3 median FTC target delay was 3.16 s, compared with 0.60--0.76 s
in the earlier aggregate batches. This is a seed-composition effect rather than
an interaction with the learned-advice gate: model enrichment occurs only after
the simulator and FTC logs have already been produced.

Per-group delays are stable across versions:

| Batch | Horizontal count | Horizontal median | Vertical count | Vertical median |
| --- | ---: | ---: | ---: | ---: |
| V1 | 13 | 0.74 s | 5 | 3.62 s |
| V2 | 15 | 0.59 s | 4 | 3.45 s |
| V3 | 9 | 0.80 s | 12 | 3.40 s |

V3 randomly contained many more vertical no-output missions, which moved the
aggregate median. All vertical cases were still targeted correctly. The slower
vertical detection is now the clearest FTC performance improvement candidate;
it should be evaluated with a stratified per-thruster protocol rather than by
tuning against another unbalanced random batch.

## Artifacts

- V3 protocol: `docs/six_dof_unified_random_batch_protocol_v3.json`
- V3 summary: `math-model/results/six_dof_unified_random_batch_v3_20260717/unified_random_batch_v3_summary.json`
- V3 per-mission data: `math-model/results/six_dof_unified_random_batch_v3_20260717/unified_random_batch_v3_missions.csv`
- V1/V2/V3 comparison: `math-model/results/six_dof_unified_random_batch_v3_20260717/unified_random_batch_v1_v2_v3_comparison.png`
- V2 retained result: `math-model/results/six_dof_unified_random_batch_v2_20260717/unified_random_batch_v2_summary.json`
