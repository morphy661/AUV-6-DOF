# Six-DOF false-alarm audit V1

Audit date: 2026-08-21
Scope: frozen `six_dof_hybrid_telemetry_0_300m_focus_v3` diagnosis stack
Policy: descriptive audit only; no threshold, model, or test-set tuning

## Reproducibility

- Dataset SHA-256: `209517014A9E9AA192EFEB92633FA7EEB1DF7382F58ECA899ECD1A4749549398`
- Checkpoint SHA-256: `26C88C14F4E8D5E9C93ACD7383A50EC78244CD5B150D9208C6953173D843D667`
- Runner: `depth-sensor-fault-detection/depth_fault_detection/audit_six_dof_false_alarms.py`
- Outputs: `math-model/results/six_dof_false_alarm_audit_v1_20260821/`

The audit reconstructs the already selected temporal, maintenance-health,
graded-log, and formal-ticket policies. It analyses only windows whose ground
truth mode is normal. Validation and test remain separate.

## Main result

| Decision layer | Validation false windows | Validation FAR | Test false windows | Test FAR |
|---|---:|---:|---:|---:|
| Raw neural classifier | 112 / 1,287 | 8.70% | 170 / 1,348 | 12.61% |
| Temporal decision | 99 / 1,287 | 7.69% | 167 / 1,348 | 12.39% |
| Graded operator-attention coverage | 1 / 1,287 | 0.08% | 18 / 1,348 | 1.34% |
| Formal maintenance ticket | 0 / 1,287 | 0.00% | 0 / 1,348 | 0.00% |

On test, the 170 raw false windows form 49 episodes. Median episode duration is
3.75 s, the maximum is 13.75 s, and 8 episodes last at least 8 s. The temporal
layer reduces 49 raw episodes to 26 latched episodes, but does not materially
reduce normal-window coverage. This means persistence alone is not a sufficient
bottom-layer fix.

All 32 formal test tickets start after the corresponding injected fault. There
are no false tickets, no tickets starting before a fault, and no ticket mode
mismatches. Ticket event precision is 100%; event recall is 100% for no-output
faults and 77.78% for thrust-loss faults.

## Where the raw false alarms occur

| Test subset | False / normal windows | FAR |
|---|---:|---:|
| Wholly normal missions | 66 / 207 | 31.88% |
| Pre-fault segments of fault missions | 104 / 1,141 | 9.11% |
| 0-50 m | 57 / 350 | 16.29% |
| 50-150 m | 37 / 444 | 8.33% |
| 150-300 m | 75 / 425 | 17.65% |
| 300-500 m secondary coverage | 1 / 129 | 0.78% |
| Within 2.5 s of a context transition | 43 / 389 | 11.05% |
| Outside the transition neighbourhood | 127 / 959 | 13.24% |

Only three wholly normal missions are present in each of validation and test.
Their raw test FARs are 11.59%, 40.58%, and 43.48%. The corresponding validation
missions are also unstable across random deployments: two have 0% raw FAR and
one has 55.07%. Therefore the current normal-only mission sample is too small to
claim a stable deployment false-alarm rate.

The dataset contains 14 wholly normal training missions (966 windows). The
remaining 5,077 normal-labelled training windows are pre-fault portions of
fault missions. This is enough to learn a normal class, but it does not provide
broad coverage of complete normal missions under independent parameter,
disturbance, sensor, and guidance randomisation.

## Evidence pattern

Of 170 raw test false windows, 168 are classified as `thrust_loss` and only 2 as
`no_output`. The false fault-probability median is 98.68% and its 90th percentile
is 99.995%, so the problem is not merely a cluster of predictions just above a
decision threshold.

False windows also have larger average attitude error (0.565 rad versus 0.325
rad), depth tracking error (0.147 m versus 0.102 m), maximum excitation ratio
(0.403 versus 0.346), and maximum motion-loss evidence (0.394 versus 0.299) than
correct normal windows. False-alarm rates increase rather than decrease in the
higher-excitation bins. Local ESC anomaly evidence is almost always zero.

These observations indicate overlap between difficult but healthy motion and
the learned thrust-loss signature. They do not support depth, low excitation,
or context transition as a single root cause. Threshold-only tuning would hide
some symptoms while leaving the model's overconfident normal/thrust-loss
separation unresolved.

## Important metric interpretation

The graded log merges adjacent event segments and assigns one final level to the
whole merged event. Consequently, a merged event that later contains a true
fault can backfill an operator-attention level onto a few preceding normal
windows. The 1.34% test `normal_window_operator_attention_rate` therefore must
not be read as 18 real pre-fault prompts. Event-level metrics show zero false
operator-attention events, and formal ticket start times confirm zero early
tickets.

## Decision

V3 is suitable to freeze as a reproducible archived baseline, and its formal
ticket/FTC gate should remain unchanged for now. It should not yet be declared
the final accepted diagnosis model because the bottom classifier is genuinely
over-sensitive and the normal-only evaluation contains only three missions per
held-out split.

The next controlled version should be V3.1, limited to diagnosis data and
calibration work:

1. Generate a separate normal-only development set covering 0-50, 50-150, and
   150-300 m, with high-excitation turns, depth changes, disturbances, sensor
   variation, and dynamics randomisation.
2. Use training/validation hard negatives from persistent false episodes;
   neither the present test labels nor its false episodes may be used for model
   or threshold selection.
3. Retrain the same BiLSTM-attention architecture first, then calibrate its mode
   probabilities on validation data. Architecture replacement is not justified
   by this audit.
4. Keep the current formal-ticket and FTC logic fixed while diagnosing the raw
   model, so changes remain attributable.
5. After V3.1 is locked, generate a new untouched mixed normal/fault acceptance
   set. Require zero false formal tickets, no loss of current no-output recall,
   no material degradation of thrust-loss recall or delay, and a clear reduction
   in persistent raw/temporal false episodes.
