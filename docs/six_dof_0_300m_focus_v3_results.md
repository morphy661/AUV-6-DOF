# Six-DOF 0-300 m Focus V3 Results

Date: 2026-08-05

## Scope

V3 is the current primary simulation dataset and BiLSTM-attention checkpoint.
It separates two questions that had previously been mixed together:

1. Main accuracy is measured on independent mission seeds from one broad,
   randomized deployment domain.
2. Compound parameter ranges entirely outside training remain a separate OOD
   stress test and are not reported as the main accuracy score.

The depth distribution is 25% at 0-50 m, 35% at 50-150 m, 30% at 150-300 m,
and 10% at 300-500 m. Thus, 90% of missions focus on 0-300 m while retaining
limited deep-stress coverage.

## Dataset

| Item | Value |
|---|---:|
| Missions | 260 |
| Windows | 18,633 |
| Window shape | 100 x 109 raw features |
| Train / validation / test missions | 182 / 39 / 39 |
| Train / validation / test windows | 13,050 / 2,793 / 2,790 |
| File size | 776.16 MB |
| SHA-256 | `209517014A9E9AA192EFEB92633FA7EEB1DF7382F58ECA899ECD1A4749549398` |

All features were finite, all three mode labels, seven location labels, and 13
joint labels were present, and mission IDs were disjoint across splits.

## Model comparison

| Metric | Previous baseline | V2 compound OOD | V3 current |
|---|---:|---:|---:|
| Mode macro F1 | 0.819 | 0.687 | **0.888** |
| Location macro F1 | 0.557 | 0.619 | **0.805** |
| Joint macro F1 | 0.563 | 0.651 | **0.798** |
| Normal-window false alarm rate | 28.48% | 64.01% | **12.61%** |
| Fault-mission detection | 36/36 | 36/36 | **36/36** |
| Exact mode and location missions | 33/36 | 35/36 | **35/36** |
| Mean mode-detection delay | 1.48 s | 1.05 s | **0.85 s** |

The V2 result showed that placing every test parameter outside training at the
same time caused severe normal-to-thrust-loss confusion. V3 broadens training
randomization and uses new, disjoint test missions inside that deployment
domain. The change reduces train-to-test feature shift rather than hiding deep
samples or selecting a threshold on test data.

## Depth-stratified test mode F1

| Depth | Mode macro F1 | Normal-window false alarm rate |
|---|---:|---:|
| 0-50 m | 0.888 | 16.29% |
| 50-150 m | 0.920 | 8.33% |
| 150-300 m | 0.865 | 17.65% |
| 300-500 m | 0.865 | 0.78% |

The remaining false alarms are not concentrated in the 300-500 m band.

## Temporal and maintenance layer

Temporal and ticket parameters were selected on validation missions before the
independent test metrics were read.

| Operational metric | Test result |
|---|---:|
| Review-log fault mission recall | 100% |
| Formal ticket precision | 100% |
| False formal maintenance tickets | 0 |
| No-output ticket recall | 100% |
| Thrust-loss ticket recall | 77.78% |
| Operator-attention precision | 100% |
| False operator-attention events | 0 |
| Top-2 ticket location hit rate | 93.75% |

Raw possible events remain in the research log, but the graded policy prevents
them from becoming false operator maintenance prompts. The learned model stays
advisory and cannot command FTC or isolate a thruster.

## Limitations

- These are simulation results, not sea-trial accuracy.
- The V3 main test covers independent missions in a broad modeled deployment
  domain; the failed compound OOD V2 run remains evidence of extrapolation
  limits.
- Only 10% of missions are in 300-500 m, so conclusions in that band have lower
  statistical support.
- The frozen V4 `36/36` acceptance badge predates V3. A new locked acceptance
  version is required before displaying a current acceptance badge by default.
