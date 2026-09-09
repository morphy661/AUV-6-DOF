"""Matplotlib video renderer for unified six-DOF diagnostics."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


TIER_COLORS = {
    "normal": "#31c7b5",
    "log_only": "#8ea1b2",
    "possible": "#f1b94b",
    "confirmed": "#ff5c67",
}
BACKGROUND = "#07131f"
PANEL = "#0d2031"
FOREGROUND = "#eef6fb"
MUTED = "#91a8ba"
GRID = "#294256"
TARGET = "#d8e060"
ESTIMATE = "#4db5ff"
FTC_COLOR = "#b28cff"
TRUTH_COLOR = "#ff8a65"


def active_injected_truth(manifest, time_s):
    """Return display-only injected events active at ``time_s``.

    The returned truth is presentation/evaluation metadata.  It is deliberately
    kept outside adapted diagnosis frames so it cannot become FTC or model
    evidence.
    """

    if not manifest:
        return []
    active = []
    for event in manifest.get("sensor_events", ()):
        if event["start_time_s"] <= time_s <= event["end_time_s"]:
            active.append({
                "kind": "sensor",
                "source": str(event["sensor"]).upper(),
                "mode": str(event["mode"]),
                "start_time_s": float(event["start_time_s"]),
                "end_time_s": float(event["end_time_s"]),
                "magnitude": float(event.get("magnitude", 0.0)),
            })
    for event in manifest.get("esc_telemetry_events", ()):
        if event["start_time_s"] <= time_s <= event["end_time_s"]:
            active.append({
                "kind": "esc_link",
                "source": f"{event['thruster_name']} ESC",
                "mode": str(event["mode"]),
                "start_time_s": float(event["start_time_s"]),
                "end_time_s": float(event["end_time_s"]),
            })
    fault = manifest.get("thruster_fault")
    if fault and time_s >= float(fault["start_time_s"]):
        active.append({
            "kind": "thruster",
            "source": str(fault["thruster_name"]),
            "mode": str(fault["mode"]),
            "start_time_s": float(fault["start_time_s"]),
            "end_time_s": None,
        })
    return active


def _configure_axis(axis, title=None):
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    if title:
        axis.set_title(title, color=FOREGROUND, fontsize=11, loc="left", pad=7)


def _set_3d_style(axis):
    _configure_axis(axis, "6-DOF motion and target")
    axis.xaxis.pane.set_facecolor(PANEL)
    axis.yaxis.pane.set_facecolor(PANEL)
    axis.zaxis.pane.set_facecolor(PANEL)
    axis.xaxis.pane.set_edgecolor(GRID)
    axis.yaxis.pane.set_edgecolor(GRID)
    axis.zaxis.pane.set_edgecolor(GRID)
    axis.set_xlabel("North (m)", color=MUTED, labelpad=7)
    axis.set_ylabel("East (m)", color=MUTED, labelpad=7)
    axis.set_zlabel("Depth (m)", color=MUTED, labelpad=7)
    axis.view_init(elev=24, azim=-58)


def _pose_array(frames, key):
    return np.asarray([frame["pose"][key] for frame in frames], dtype=float)


def _limits(values, padding=0.12, minimum_span=1.0):
    low = np.min(values, axis=0)
    high = np.max(values, axis=0)
    span = np.maximum(high - low, minimum_span)
    return low - padding * span, high + padding * span


class SixDOFDemoRenderer:
    """Render adapted frames as a 16:9 operator dashboard."""

    def __init__(
        self,
        frames: Sequence[Mapping],
        events=(),
        *,
        injection_manifest: Mapping | None = None,
        view_mode: str = "summary",
        acceptance_badge: Mapping | None = None,
    ):
        self.frames = list(frames)
        if not self.frames:
            raise ValueError("frames cannot be empty")
        if view_mode not in {"summary", "technical"}:
            raise ValueError("view_mode must be summary or technical")
        self.events = list(events)
        self.injection_manifest = (
            {} if injection_manifest is None else dict(injection_manifest)
        )
        self.view_mode = view_mode
        self.acceptance_badge = (
            None if acceptance_badge is None else dict(acceptance_badge)
        )
        self.times = np.asarray([frame["time_s"] for frame in self.frames])
        self.positions = _pose_array(self.frames, "position_ned_m")
        self.estimates = _pose_array(self.frames, "estimated_position_ned_m")
        self.control_targets = _pose_array(self.frames, "target_position_ned_m")
        self.targets = np.asarray([
            frame["pose"].get(
                "mission_target_position_ned_m",
                frame["pose"]["target_position_ned_m"],
            )
            for frame in self.frames
        ], dtype=float)
        combined = np.vstack((
            self.positions,
            self.estimates,
            self.control_targets,
            self.targets,
        ))
        self.spatial_low, self.spatial_high = _limits(combined)
        self.figure = plt.figure(figsize=(12.8, 7.2), facecolor=BACKGROUND)
        grid = self.figure.add_gridspec(
            12, 16, left=0.070, right=0.975, bottom=0.07, top=0.93,
            hspace=1.15, wspace=1.0,
        )
        self.motion_axis = self.figure.add_subplot(
            grid[:9, :10], projection="3d"
        )
        self.sensor_axis = self.figure.add_subplot(grid[:4, 10:])
        self.thruster_axis = self.figure.add_subplot(grid[4:8, 10:])
        self.status_axis = self.figure.add_subplot(grid[9:, 10:])
        self.timeline_axis = self.figure.add_subplot(grid[9:, :10])
        self.figure.suptitle(
            "AUV 6-DOF  |  Causal diagnosis and fault-tolerant control replay",
            color=FOREGROUND, fontsize=14, x=0.045, ha="left",
        )
        if self.acceptance_badge is not None:
            passed = int(self.acceptance_badge["passed_count"])
            total = int(self.acceptance_badge["check_count"])
            accepted = bool(self.acceptance_badge["accepted"])
            badge_color = (
                TIER_COLORS["normal"]
                if accepted
                else TIER_COLORS["confirmed"]
            )
            result = "PASS" if accepted else "NOT ACCEPTED"
            self.figure.text(
                0.975,
                0.988,
                f"OFFLINE V4 BASELINE  {passed}/{total} {result}",
                color=badge_color,
                fontsize=8.0,
                ha="right",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.32",
                    "facecolor": PANEL,
                    "edgecolor": badge_color,
                    "linewidth": 0.9,
                },
            )

    def _draw_motion(self, index):
        axis = self.motion_axis
        axis.clear()
        _set_3d_style(axis)
        axis.plot(
            self.targets[: index + 1, 0], self.targets[: index + 1, 1],
            self.targets[: index + 1, 2], color=TARGET, linestyle="--",
            linewidth=1.5, label="Mission target",
        )
        target_diverged = bool(np.any(np.linalg.norm(
            self.control_targets[: index + 1] - self.targets[: index + 1],
            axis=1,
        ) > 1e-6))
        if target_diverged:
            axis.plot(
                self.control_targets[: index + 1, 0],
                self.control_targets[: index + 1, 1],
                self.control_targets[: index + 1, 2],
                color=FTC_COLOR,
                linestyle=":",
                linewidth=1.8,
                label="FTC hold target",
            )
        axis.plot(
            self.positions[: index + 1, 0], self.positions[: index + 1, 1],
            self.positions[: index + 1, 2], color=FOREGROUND, linewidth=2.0,
            label=(
                "Simulation truth"
                if self.view_mode == "summary"
                else "Vehicle"
            ),
        )
        axis.plot(
            self.estimates[: index + 1, 0], self.estimates[: index + 1, 1],
            self.estimates[: index + 1, 2], color=ESTIMATE, linewidth=1.2,
            alpha=0.8, label="Estimate",
        )
        pose = self.frames[index]["pose"]
        current = np.asarray(pose["position_ned_m"], dtype=float)
        target = np.asarray(pose["target_position_ned_m"], dtype=float)
        mission_target = np.asarray(
            pose.get("mission_target_position_ned_m", target), dtype=float
        )
        rpy = np.asarray(pose["euler_rpy_rad"], dtype=float)
        axis.scatter(*current, s=65, color=FOREGROUND, edgecolor=BACKGROUND)
        axis.scatter(*mission_target, s=70, marker="*", color=TARGET)
        if target_diverged:
            axis.scatter(*target, s=50, marker="s", color=FTC_COLOR)
        yaw = rpy[2]
        direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        axis.quiver(
            *current, *direction, length=0.8, normalize=True,
            color=FTC_COLOR, linewidth=2.0,
        )
        axis.set_xlim(self.spatial_low[0], self.spatial_high[0])
        axis.set_ylim(self.spatial_low[1], self.spatial_high[1])
        axis.set_zlim(self.spatial_high[2], self.spatial_low[2])
        legend = axis.legend(loc="upper right", fontsize=8, frameon=False)
        for text in legend.get_texts():
            text.set_color(FOREGROUND)
        axis.text2D(
            0.02, 0.02,
            "Roll {:+5.1f}°   Pitch {:+5.1f}°   Yaw {:+5.1f}°".format(
                *np.rad2deg(rpy)
            ),
            transform=axis.transAxes, color=FOREGROUND, fontsize=9,
        )

    def _truth_text(self, index):
        active = active_injected_truth(
            self.injection_manifest, float(self.times[index])
        )
        if not active:
            return active, "NORMAL | no active injected anomaly"
        labels = [
            (
                f"{item['source']}: {item['mode'].replace('_', ' ').upper()} "
                f"@ {item['start_time_s']:.2f} s"
            )
            for item in active
        ]
        return active, " | ".join(labels)

    def _maintenance_records(self, index, *, min_time_s=None):
        records = []
        for candidate_index in range(int(index) + 1):
            frame = self.frames[candidate_index]
            time_s = float(frame.get("time_s", self.times[candidate_index]))
            if min_time_s is not None and time_s < float(min_time_s):
                continue
            maintenance = frame.get("maintenance", {})
            if maintenance.get("tier", "normal") not in {
                "log_only", "possible"
            }:
                continue
            if maintenance.get("probable_mode", "normal") == "normal":
                continue
            records.append((time_s, maintenance))
        return records

    @staticmethod
    def _maintenance_text(time_s, maintenance, *, historical):
        mode = str(
            maintenance.get("probable_mode", "unknown")
        ).replace("_", " ").upper()
        group = str(
            maintenance.get("suspected_group", "unknown")
        ).replace("_", " ").upper()
        candidates = list(maintenance.get("candidates", ()))[:2]
        candidate_text = ", ".join(
            f"{candidate.get('name', '?')} "
            f"{100.0 * float(candidate.get('probability', 0.0)):.0f}%"
            for candidate in candidates
        ) or "none"
        if historical:
            return (
                f"RECORDED MODEL ADVICE @ {float(time_s):.2f} s\n"
                f"{mode} / {group} | Last Top-2: {candidate_text}"
            )
        return (
            f"MODEL ADVISORY | {mode} / {group}\n"
            f"Top-2: {candidate_text}"
        )

    def _primary_diagnosis(self, index):
        frame = self.frames[index]
        ftc = frame["ftc"]
        target = ftc.get("target_thruster")
        if target:
            return (
                f"{target}: PHYSICAL FAULT CONFIRMED",
                TIER_COLORS["confirmed"],
            )
        tier_rank = {
            "normal": 0, "log_only": 1, "possible": 2, "confirmed": 3,
        }
        sensor_name, sensor_card = max(
            frame["sensors"].items(),
            key=lambda item: tier_rank.get(item[1]["tier"], 0),
        )
        tier = sensor_card["tier"]
        if tier == "confirmed":
            return (
                f"{sensor_name.upper()}: CONFIRMED",
                TIER_COLORS["confirmed"],
            )
        maintenance = frame.get("maintenance", {})
        if (
            maintenance.get("tier", "normal") in {"log_only", "possible"}
            and maintenance.get("probable_mode", "normal") != "normal"
        ):
            return (
                self._maintenance_text(
                    float(frame.get("time_s", self.times[index])),
                    maintenance,
                    historical=False,
                ),
                TIER_COLORS[maintenance.get("tier", "possible")],
            )
        untrusted = tuple(ftc.get("untrusted_esc_channels", ()))
        if untrusted:
            return (
                f"{', '.join(untrusted)} ESC: LINK ANOMALY / LOG ONLY",
                TIER_COLORS["log_only"],
            )
        if tier != "normal":
            return (
                f"{sensor_name.upper()}: {tier.replace('_', ' ').upper()}",
                TIER_COLORS[tier],
            )
        maintenance_records = self._maintenance_records(index)
        if maintenance_records:
            time_s, maintenance = maintenance_records[-1]
            return (
                self._maintenance_text(
                    time_s, maintenance, historical=True
                ),
                TIER_COLORS["log_only"],
            )
        return "NORMAL | no actionable fault", TIER_COLORS["normal"]

    def _ai_summary(self, index):
        maintenance = self.frames[index].get("maintenance", {})
        if not maintenance.get("available", False):
            now_s = float(self.frames[index].get("time_s", 0.0))
            for candidate_index in range(index - 1, -1, -1):
                candidate_frame = self.frames[candidate_index]
                candidate_time_s = float(candidate_frame.get("time_s", 0.0))
                if now_s - candidate_time_s > 3.0:
                    break
                candidate = candidate_frame.get("maintenance", {})
                if (
                    candidate.get("available", False)
                    and float(candidate.get("raw_fault_probability", 0.0)) >= 0.5
                ):
                    mode = str(
                        candidate.get("raw_probable_mode", "unknown")
                    ).replace("_", " ").upper()
                    probability = 100.0 * float(
                        candidate.get("raw_fault_probability", 0.0)
                    )
                    group = str(
                        candidate.get("raw_suspected_group", "unknown")
                    ).replace("_", " ").upper()
                    return (
                        f"RECENT {mode} | P {probability:.0f}% @ "
                        f"{candidate_time_s:.2f} s | {group}",
                        TIER_COLORS["possible"],
                    )
            return "WARMING UP CAUSAL WINDOW", MUTED
        mode = str(
            maintenance.get("raw_probable_mode", "normal")
        ).replace("_", " ").upper()
        probability = 100.0 * float(
            maintenance.get("raw_fault_probability", 0.0)
        )
        candidates = list(maintenance.get("candidates", ()))[:2]
        top = ", ".join(
            f"{candidate.get('name', '?')} "
            f"{100.0 * float(candidate.get('probability', 0.0)):.0f}%"
            for candidate in candidates
        )
        if not top:
            group = str(
                maintenance.get("raw_suspected_group", "unknown")
            ).replace("_", " ")
            top = f"group {group}" if group != "unknown" else "location uncertain"
        prefix = "NEW" if maintenance.get("updated", False) else "HELD"
        color = (
            TIER_COLORS["normal"]
            if mode == "NORMAL"
            else TIER_COLORS["possible"]
        )
        return (
            f"{prefix} {mode} | fault P {probability:.0f}% | {top}",
            color,
        )

    def _rule_summary(self, index):
        frame = self.frames[index]
        ftc = frame["ftc"]
        target = ftc.get("target_thruster")
        cards = list(frame.get("thrusters", ()))
        strongest = max(
            cards,
            key=lambda card: float(card.get("no_output_score", 0.0)),
            default=None,
        )
        strongest_name = "none" if strongest is None else strongest.get("name", "?")
        strongest_score = (
            0.0 if strongest is None
            else 100.0 * float(strongest.get("no_output_score", 0.0))
        )
        if target:
            text = f"ESC CONFIRMED {target} | LATCHED"
            color = TIER_COLORS["confirmed"]
        elif ftc.get("untrusted_esc_channels"):
            text = "ESC LINK UNTRUSTED | LOG ONLY"
            color = TIER_COLORS["log_only"]
        elif strongest_score >= 80.0:
            text = (
                f"{strongest_name} SCORE {strongest_score:.0f}% | "
                "CONFIRMATION TIMER"
            )
            color = TIER_COLORS["possible"]
        elif strongest_score >= 35.0:
            text = f"{strongest_name} WEAK EVIDENCE {strongest_score:.0f}% | LOG"
            color = TIER_COLORS["log_only"]
        else:
            text = "NO DIRECT ESC FAULT CONFIRMATION"
            color = TIER_COLORS["normal"]
        fusion = ftc.get("ai_fusion", {})
        if fusion:
            raw_verdict = str(fusion.get("verdict", "unavailable"))
            verdict = {
                "direct_rule_priority": "DIRECT RULE",
                "context_blocked": "CONTEXT BLOCK",
                "suspected": "AI SUSPECTED",
                "normal": "NO AI ESCALATION",
                "unavailable": "UNAVAILABLE",
            }.get(raw_verdict, raw_verdict.replace("_", " ").upper())
            if (
                raw_verdict == "unavailable"
                and target
                and ftc.get("guidance_mode_applied") != "normal"
            ):
                verdict = "RESET AFTER ACTION"
            text += f" | GATE {verdict}"
        return text, color

    def _comparison_text(self, index, active):
        frame = self.frames[index]
        if not active:
            if frame["overall_tier"] == "normal":
                return "NO ACTIVE INJECTION", TIER_COLORS["normal"]
            return "POST-EVENT MONITORING", TIER_COLORS["log_only"]
        event = active[-1]
        if event["kind"] == "thruster":
            if frame["ftc"].get("target_thruster") == event["source"]:
                detected = [
                    float(candidate["time_s"])
                    for candidate in self.frames
                    if candidate["ftc"].get("target_thruster")
                    == event["source"]
                ]
                delay = min(detected) - float(event["start_time_s"])
                return (
                    f"MATCH / CORRECT | confirmation delay {delay:.2f} s",
                    TIER_COLORS["normal"],
                )
            records = self._maintenance_records(
                index, min_time_s=float(event["start_time_s"])
            )
            expected_mode = str(event["mode"])
            matching = [
                record for record in records
                if str(record[1].get("probable_mode", "normal"))
                == expected_mode
            ]
            if matching:
                first_time = matching[0][0]
                _, latest = matching[-1]
                candidate_names = {
                    str(candidate.get("name", ""))
                    for candidate in latest.get("candidates", ())[:2]
                }
                expected_group = (
                    "horizontal"
                    if str(event["source"]).startswith("H")
                    else "vertical"
                )
                delay = first_time - float(event["start_time_s"])
                if str(event["source"]) in candidate_names:
                    detail = "mode + Top-2"
                    color = TIER_COLORS["normal"]
                    outcome = "RECORDED MATCH"
                elif latest.get("suspected_group") == expected_group:
                    detail = "mode + group; exact source uncertain"
                    color = TIER_COLORS["possible"]
                    outcome = "RECORDED PARTIAL"
                else:
                    detail = "mode only; location uncertain"
                    color = TIER_COLORS["possible"]
                    outcome = "RECORDED PARTIAL"
                return (
                    f"{outcome} | {detail} | first delay {delay:.2f} s",
                    color,
                )
            if records:
                return (
                    "MODEL DISAGREES | recorded mode differs from truth",
                    TIER_COLORS["possible"],
                )
            return "PENDING | collecting causal evidence", TIER_COLORS["possible"]
        if event["kind"] == "esc_link":
            thruster = event["source"].split()[0]
            untrusted = frame["ftc"].get("untrusted_esc_channels", ())
            if thruster in untrusted and not frame["ftc"].get("target_thruster"):
                return (
                    "MATCH / CORRECT | link only, no physical isolation",
                    TIER_COLORS["normal"],
                )
            return "PENDING | checking link state", TIER_COLORS["possible"]
        sensor = str(event["source"]).lower()
        tier = frame["sensors"][sensor]["tier"]
        if tier != "normal":
            return (
                f"OBSERVED | response tier {tier.replace('_', ' ')}",
                TIER_COLORS.get(tier, MUTED),
            )
        return "MONITORING | no decision yet", TIER_COLORS["possible"]

    def _draw_summary_scenario(self, index):
        axis = self.sensor_axis
        axis.clear()
        _configure_axis(axis, "Scenario")
        axis.axis("off")
        frame = self.frames[index]
        depth = float(frame["pose"]["position_ned_m"][2])
        control_depth = float(frame["pose"]["target_position_ned_m"][2])
        mission_depth = float(
            frame["pose"].get(
                "mission_target_position_ned_m",
                frame["pose"]["target_position_ned_m"],
            )[2]
        )
        initial_depth = float(self.positions[0, 2])
        guidance = str(
            frame["ftc"].get("guidance_mode_applied", "normal")
        )
        mission_state = (
            "PAUSED / LOCAL HOLD"
            if guidance == "vertical_thruster_hold"
            else "ACTIVE"
        )
        mode = str(
            self.injection_manifest.get("injection_mode", "unspecified")
        ).upper()
        lines = (
            f"{initial_depth:.0f} m local-depth replay",
            f"Mission time   {self.times[index]:05.2f} / {self.times[-1]:.2f} s",
            (
                f"Depth          {depth:7.2f} m | control {control_depth:7.2f} m "
                f"| mission {mission_depth:7.2f} m"
            ),
            f"Mission        {mission_state} | guidance {guidance.replace('_', ' ')}",
            f"Injection      {mode}   | environmental disturbance ON",
        )
        for row, line in enumerate(lines):
            axis.text(
                0.02, 0.88 - 0.18 * row, line,
                color=FOREGROUND if row < 4 else MUTED,
                fontsize=9.4 if row == 0 else 8.3,
                fontweight="bold" if row == 0 else "normal",
                transform=axis.transAxes, va="top",
            )

    def _draw_summary_comparison(self, index):
        axis = self.thruster_axis
        axis.clear()
        _configure_axis(axis, "Causal diagnosis chain (truth is display-only)")
        axis.axis("off")
        active, truth_text = self._truth_text(index)
        ai_text, ai_color = self._ai_summary(index)
        rule_text, rule_color = self._rule_summary(index)
        comparison_text, comparison_color = self._comparison_text(
            index, active
        )
        rows = (
            ("GROUND TRUTH", truth_text, TRUTH_COLOR, 0.88),
            ("AI MODEL", ai_text, ai_color, 0.66),
            ("RULE / GATE", rule_text, rule_color, 0.44),
            ("OUTCOME", comparison_text, comparison_color, 0.22),
        )
        for label, value, color, y in rows:
            axis.text(
                0.02, y, label, color=MUTED, fontsize=7.2,
                transform=axis.transAxes, va="top",
            )
            axis.text(
                0.25, y, str(value)[:78], color=color, fontsize=7.7,
                fontweight="bold", transform=axis.transAxes, va="top",
            )

    def _draw_summary_ftc(self, index):
        axis = self.status_axis
        axis.clear()
        _configure_axis(axis, "Fault-tolerant control")
        axis.axis("off")
        frame = self.frames[index]
        ftc = frame["ftc"]
        color = TIER_COLORS[ftc["tier"]]
        target = ftc.get("target_thruster") or "none"
        axis.text(0.02, 0.84, "Action", color=MUTED, fontsize=8,
                  transform=axis.transAxes)
        axis.text(
            0.22, 0.84, ftc["action"].replace("_", " ").upper(),
            color=color, fontsize=9.5, fontweight="bold",
            transform=axis.transAxes,
        )
        axis.text(0.02, 0.62, "Target", color=MUTED, fontsize=8,
                  transform=axis.transAxes)
        axis.text(0.22, 0.62, target, color=FOREGROUND, fontsize=9.5,
                  fontweight="bold", transform=axis.transAxes)
        guidance = str(ftc.get("guidance_mode_applied", "normal"))
        guidance_text = guidance.replace("_", " ").upper()
        if guidance == "vertical_thruster_hold":
            guidance_text += " | MISSION PAUSED"
        axis.text(0.02, 0.40, "Guidance", color=MUTED, fontsize=8,
                  transform=axis.transAxes)
        axis.text(
            0.22, 0.40, guidance_text,
            color=FTC_COLOR if guidance != "normal" else FOREGROUND,
            fontsize=8.4, fontweight="bold", transform=axis.transAxes,
        )
        tracking_available = bool(ftc.get("tracking_error_available", False))
        ratio = float(ftc.get("tracking_error_ratio", 0.0))
        if not tracking_available:
            safety_text = "UNAVAILABLE"
            safety_color = TIER_COLORS["log_only"]
        else:
            tilt = float(ftc.get("tilt_angle_deg", 0.0))
            depth_error = float(ftc.get("depth_tracking_error_m", 0.0))
            horizontal = float(ftc.get("horizontal_tracking_error_m", 0.0))
            safety_text = (
                f"{'SAFE' if ratio < 1.0 else 'LIMIT EXCEEDED'} | "
                f"tilt {tilt:.1f}/35 deg | dz {depth_error:.2f}/3 m | "
                f"dxy {horizontal:.2f}/8 m"
            )
            safety_color = (
                TIER_COLORS["normal"]
                if ratio < 0.60
                else (
                    TIER_COLORS["possible"]
                    if ratio < 1.0
                    else TIER_COLORS["confirmed"]
                )
            )
        axis.text(0.02, 0.18, "Safety", color=MUTED, fontsize=8,
                  transform=axis.transAxes)
        axis.text(
            0.22, 0.18, safety_text,
            color=safety_color, fontsize=7.6, transform=axis.transAxes,
        )

    def _draw_sensors(self, index):
        axis = self.sensor_axis
        axis.clear()
        _configure_axis(axis, "Current-frame sensor diagnosis")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 3)
        axis.axis("off")
        cards = self.frames[index]["sensors"]
        for row, name in enumerate(("depth", "imu", "dvl")):
            card = cards[name]
            y = 2.45 - row
            color = TIER_COLORS[card["tier"]]
            patch = FancyBboxPatch(
                (0.01, y - 0.32), 0.98, 0.64,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                facecolor=color + "24", edgecolor=color, linewidth=1.2,
            )
            axis.add_patch(patch)
            axis.text(0.04, y + 0.10, name.upper(), color=FOREGROUND,
                      fontsize=10, fontweight="bold", va="center")
            axis.text(0.25, y + 0.10, card["tier"].replace("_", " "),
                      color=color, fontsize=9, va="center")
            axis.text(0.04, y - 0.14, card["label"][:42], color=MUTED,
                      fontsize=8, va="center")
            axis.text(0.96, y, f"{100 * card['confidence']:.0f}%",
                      color=FOREGROUND, fontsize=9, ha="right", va="center")

    def _draw_thrusters(self, index):
        axis = self.thruster_axis
        axis.clear()
        _configure_axis(
            axis,
            "Current-frame thrusters: command / ESC / link / model P",
        )
        cards = self.frames[index]["thrusters"]
        maintenance = self.frames[index].get("maintenance", {})
        names = [card["name"] for card in cards]
        excitation = np.asarray([card["excitation_ratio"] for card in cards])
        scores = np.asarray([card["no_output_score"] for card in cards])
        location = np.asarray(
            maintenance.get("location_probabilities", np.zeros(6)),
            dtype=float,
        )
        if location.shape != (6,):
            location = np.zeros(6)
        y = np.arange(6)
        axis.barh(y, excitation, height=0.54, color=ESTIMATE, alpha=0.34,
                  label="command excitation")
        colors = [TIER_COLORS[card["tier"]] for card in cards]
        axis.barh(y, scores, height=0.24, color=colors, alpha=0.95,
                  label="no-output score")
        axis.scatter(
            location, y + 0.18, marker="D", s=22, color=FTC_COLOR,
            edgecolor=BACKGROUND, linewidth=0.5, zorder=4,
            label="model location probability",
        )
        for row, card in enumerate(cards):
            if card["tier"] != "normal":
                axis.text(
                    1.06, row,
                    card["label"], color=colors[row],
                    fontsize=7, va="center", ha="right",
                )
        axis.axvline(0.80, color=TIER_COLORS["confirmed"], linestyle="--",
                     linewidth=1.0, alpha=0.7)
        axis.set_yticks(y, names)
        axis.set_xlim(0, 1.08)
        axis.grid(axis="x", color=GRID, alpha=0.55, linewidth=0.6)
        axis.invert_yaxis()

    def _draw_status(self, index):
        axis = self.status_axis
        axis.clear()
        _configure_axis(axis, "Current-frame FTC, model advice and safety")
        axis.axis("off")
        frame = self.frames[index]
        ftc = frame["ftc"]
        estimator = frame["estimator"]
        maintenance = frame.get("maintenance", {})
        color = TIER_COLORS[ftc["tier"]]
        target = ftc["target_thruster"] or "none"
        axis.text(0.01, 0.91, "FTC", color=MUTED, fontsize=8, va="top")
        axis.text(0.16, 0.91, ftc["action"].replace("_", " "),
                  color=color, fontsize=10, va="top")
        axis.text(0.01, 0.72, "Target", color=MUTED, fontsize=8, va="top")
        axis.text(0.16, 0.72, target, color=FOREGROUND, fontsize=9, va="top")
        axis.text(0.50, 0.72, "Estimator", color=MUTED, fontsize=8, va="top")
        axis.text(0.68, 0.72, estimator["quality"].replace("_", " "),
                   color=FOREGROUND, fontsize=9, va="top")
        untrusted_esc = ftc.get("untrusted_esc_channels", ())
        esc_text = (
            "all channels fresh"
            if not untrusted_esc
            else f"{', '.join(untrusted_esc)} unavailable/stale | log only"
        )
        esc_color = MUTED if not untrusted_esc else TIER_COLORS["log_only"]
        axis.text(0.01, 0.56, "ESC link", color=MUTED, fontsize=8, va="top")
        axis.text(0.16, 0.56, esc_text, color=esc_color, fontsize=8.5, va="top")
        model_tier = maintenance.get("tier", "normal")
        model_color = TIER_COLORS.get(model_tier, MUTED)
        mode = maintenance.get("probable_mode", "normal").replace("_", " ")
        probability = 100.0 * float(maintenance.get("fault_probability", 0.0))
        if maintenance.get("advisory_gate_active", False):
            model_text = "advice withheld | context stabilizing"
        elif not maintenance.get("available", False):
            model_text = "warming up causal window"
        elif model_tier == "normal":
            model_text = f"normal | fault probability {probability:.0f}%"
        else:
            model_text = f"possible {mode} | probability {probability:.0f}%"
        axis.text(0.01, 0.42, "Model", color=MUTED, fontsize=8, va="top")
        axis.text(
            0.16, 0.42, model_text,
            color=model_color, fontsize=8.5, va="top",
        )
        candidates = maintenance.get("candidates", ())
        candidate_text = ", ".join(
            f"{candidate['name']} {100 * candidate['probability']:.0f}%"
            for candidate in candidates[:2]
        ) or "none"
        group = str(maintenance.get("suspected_group", "none"))
        if maintenance.get("advisory_gate_active", False):
            reasons = maintenance.get("advisory_gate_reasons", ())
            candidate_text = ", ".join(reasons[:2]) or "context transition"
            group = "withheld"
        axis.text(0.01, 0.25, "Advice", color=MUTED, fontsize=8, va="top")
        axis.text(
            0.16, 0.25, f"{group} | Top-2 {candidate_text}",
            color=FOREGROUND, fontsize=8.2, va="top",
        )
        guidance = str(ftc.get("guidance_mode_applied", "normal")).replace(
            "_", " "
        )
        ratio = float(ftc.get("tracking_error_ratio", 0.0))
        safety_text = (
            f"{guidance} | ratio {ratio:.2f} | "
            f"tilt {float(ftc.get('tilt_angle_deg', 0.0)):.1f} deg | "
            f"dz {float(ftc.get('depth_tracking_error_m', 0.0)):.2f} m"
        )
        safety_color = (
            TIER_COLORS["normal"]
            if ftc.get("tracking_error_available", False) and ratio < 0.60
            else TIER_COLORS["possible"]
        )
        axis.text(0.01, 0.07, "Safety", color=MUTED, fontsize=8, va="top")
        axis.text(
            0.16, 0.07, safety_text,
            color=safety_color, fontsize=7.5, va="top",
        )

    def _truth_intervals(self):
        intervals = []
        for event in self.injection_manifest.get("sensor_events", ()):
            intervals.append((
                float(event["start_time_s"]),
                float(event["end_time_s"]),
                str(event["sensor"]).upper(),
            ))
        for event in self.injection_manifest.get("esc_telemetry_events", ()):
            intervals.append((
                float(event["start_time_s"]),
                float(event["end_time_s"]),
                f"{event['thruster_name']} link",
            ))
        fault = self.injection_manifest.get("thruster_fault")
        if fault:
            intervals.append((
                float(fault["start_time_s"]),
                float(self.times[-1]),
                f"{fault['thruster_name']} {str(fault['mode']).replace('_', ' ')}",
            ))
        return intervals

    def _draw_summary_timeline(self, index):
        axis = self.timeline_axis
        axis.clear()
        _configure_axis(axis, "Injected truth strip and causal diagnosis tier")
        tier_value = {"normal": 0, "log_only": 1, "possible": 2, "confirmed": 3}
        values = [tier_value[frame["overall_tier"]] for frame in self.frames]
        axis.step(self.times, values, where="post", color=FTC_COLOR, linewidth=1.8)
        for start, end, label in self._truth_intervals():
            width = max(end - start, 0.03)
            axis.broken_barh(
                [(start, width)], (-0.86, 0.42),
                facecolors=TRUTH_COLOR, alpha=0.75,
            )
            rotation = 90 if width < 0.6 else 0
            x = start + width / 2.0
            axis.text(
                x, -0.65, label, color=FOREGROUND, fontsize=6.1,
                ha="center", va="center", rotation=rotation,
            )
        axis.axvline(self.times[index], color=FOREGROUND, linewidth=1.3)
        axis.set_xlim(self.times[0], self.times[-1])
        axis.set_ylim(-1.0, 3.2)
        axis.set_yticks((0, 1, 2, 3), ("normal", "log", "possible", "confirmed"))
        axis.set_xlabel("Mission time (s)  |  orange strip = injected truth",
                        color=MUTED, fontsize=8)
        axis.grid(color=GRID, alpha=0.45, linewidth=0.6)
        axis.text(
            0.99, 0.92, f"t = {self.times[index]:05.2f} s",
            transform=axis.transAxes, ha="right", va="top",
            color=FOREGROUND, fontsize=10,
        )

    def _draw_timeline(self, index):
        axis = self.timeline_axis
        axis.clear()
        _configure_axis(axis, "Causal event timeline")
        tier_value = {"normal": 0, "log_only": 1, "possible": 2, "confirmed": 3}
        values = [tier_value[frame["overall_tier"]] for frame in self.frames]
        axis.step(self.times, values, where="post", color=FTC_COLOR, linewidth=1.8)
        for event in self.events:
            if event["time_s"] <= self.times[index]:
                axis.axvline(
                    event["time_s"],
                    color=TIER_COLORS.get(event["level"], MUTED),
                    alpha=0.35, linewidth=0.8,
                )
        axis.axvline(self.times[index], color=FOREGROUND, linewidth=1.3)
        axis.set_xlim(self.times[0], self.times[-1])
        axis.set_ylim(-0.15, 3.2)
        axis.set_yticks((0, 1, 2, 3), ("normal", "log", "possible", "confirmed"))
        axis.set_xlabel("Mission time (s)", color=MUTED, fontsize=8)
        axis.grid(color=GRID, alpha=0.45, linewidth=0.6)
        axis.text(
            0.99, 0.92, f"t = {self.times[index]:05.2f} s",
            transform=axis.transAxes, ha="right", va="top",
            color=FOREGROUND, fontsize=10,
        )

    def draw(self, index):
        self._draw_motion(index)
        if self.view_mode == "summary":
            self._draw_summary_scenario(index)
            self._draw_summary_comparison(index)
            self._draw_summary_ftc(index)
            self._draw_summary_timeline(index)
        else:
            self._draw_sensors(index)
            self._draw_thrusters(index)
            self._draw_status(index)
            self._draw_timeline(index)
        return []

    @staticmethod
    def _ffmpeg_path():
        configured = os.environ.get("FFMPEG_PATH")
        if configured and Path(configured).exists():
            return configured
        found = shutil.which("ffmpeg")
        if found:
            return found
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            return None

    def save_snapshot(self, path, index=None, dpi=150):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if index is None:
            recovery = [
                frame_index
                for frame_index, frame in enumerate(self.frames)
                if frame["ftc"].get("guidance_mode_applied")
                == "vertical_thruster_hold"
            ]
            targeted = [
                frame_index
                for frame_index, frame in enumerate(self.frames)
                if frame["ftc"]["target_thruster"] is not None
            ]
            index = (
                recovery[0]
                if recovery
                else (targeted[0] if targeted else len(self.frames) - 1)
            )
        self.draw(int(index))
        self.figure.savefig(path, dpi=dpi, facecolor=BACKGROUND)
        return path

    def save_video(self, path, fps=12, max_frames=240, dpi=100):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_path = self._ffmpeg_path()
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg is required for MP4 output; set FFMPEG_PATH or install imageio-ffmpeg"
            )
        matplotlib.rcParams["animation.ffmpeg_path"] = str(ffmpeg_path)
        frame_count = min(int(max_frames), len(self.frames))
        indices = np.unique(np.linspace(
            0, len(self.frames) - 1, frame_count, dtype=int
        ))
        movie = animation.FuncAnimation(
            self.figure,
            lambda sequence_index: self.draw(int(indices[sequence_index])),
            frames=len(indices), interval=1000.0 / float(fps), blit=False,
        )
        writer = animation.FFMpegWriter(
            fps=fps, codec="libx264", bitrate=2400,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        movie.save(path, writer=writer, dpi=dpi)
        return path

    def close(self):
        plt.close(self.figure)
