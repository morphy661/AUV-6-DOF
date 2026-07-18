# Six-DOF unified diagnostics final V4 presentation

Date: 2026-07-18

## Outcome

The final presentation integrates the unified V4 acceptance status into the existing causal six-DOF dashboard without changing diagnosis, estimator, model, or FTC behavior.

- Offline simulation baseline: 36/36 checks passed.
- Source simulation: 440 frames over 22 seconds.
- Rendered video: 240 uniformly sampled frames, H.264, 1280 x 720, 12 fps, 20 seconds.
- Full diagnostic regression after the presentation change: 207 tests plus 5 subtests.

The top-right badge is deliberately labelled `OFFLINE V4 BASELINE`. It is loaded from the frozen unified acceptance summary and is not evidence about the current frame. All sensor, thruster, model, estimator, timeline, and FTC fields remain causal replay values produced by the onboard-facing presentation adapter.

## Fixed causal story

1. A short depth spike is retained as a background log.
2. DVL bias becomes a possible diagnosis rather than a confirmed hardware failure.
3. Repeated IMU unavailability is confirmed per sample and requests the configured safe-hold-or-abort action; the longer-term root cause remains possible.
4. V2 ESC packet loss starts at 15.35 s, is shown as log-only, and produces no thruster target.
5. V2 communication evidence clears after recovery.
6. Real V1 no-output evidence appears at 17.05 s.
7. The FTC supervisor performs targeted reallocation to V1 at 17.55 s.
8. Learned maintenance output remains possible/advisory and never commands FTC.

## Verification

- ESC communication-anomaly frames: 21 in the simulation summary.
- Thruster targets during the V2 communication-fault interval: 0.
- First targeted V1 reallocation: 17.55 s.
- Targeted thrusters in the complete replay: V1 only.
- Offline acceptance badge embedded in JSON: 36/36 accepted.

This is a deterministic simulation presentation. It is not a real-sea trial, and the offline badge is not a new independent blind-test claim.

## Artifacts

- Video: `math-model/results/six_dof_unified_diagnostics_final_v4_20260718/six_dof_unified_diagnostics.mp4`
- Causal frames and offline injection manifest: `math-model/results/six_dof_unified_diagnostics_final_v4_20260718/six_dof_unified_diagnostics.json`
- Targeted-FTC snapshot: `math-model/results/six_dof_unified_diagnostics_final_v4_20260718/six_dof_unified_diagnostics.png`
- ESC-link log-only snapshot: `math-model/results/six_dof_unified_diagnostics_final_v4_20260718/six_dof_unified_diagnostics_esc_link.png`
