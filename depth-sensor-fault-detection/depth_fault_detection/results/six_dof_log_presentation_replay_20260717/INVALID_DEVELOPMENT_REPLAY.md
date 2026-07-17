# Invalid development replay

This first presentation-layer replay exposed an implementation error: a
continuously active anomaly was split at every guidance-context transition,
increasing 109 raw events to 224 display events. The detector, frozen V2
maintenance tickets, and FTC outputs were not changed.

Do not use this directory as the graded-log result. The logic was corrected so
that context only blocks merging between already-separated episodes. The valid
retrospective result is stored in
`results/six_dof_log_presentation_replay_v2_20260717`.
