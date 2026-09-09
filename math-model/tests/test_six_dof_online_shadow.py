"""Streaming order and isolation contracts for the online shadow observer."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from environment.six_dof_simulator import SixDOFSimulator
from ftc.safety_supervisor import FTCSafetySupervisor
from simple_control.six_dof_controller import PoseTarget


def test_observer_runs_before_step_returns_and_cannot_mutate_control_log():
    class Observer:
        def reset(self):
            self.times = []

        def update(self, log):
            assert len(simulator.logs) == len(self.times)
            assert "ftc_action" in log
            self.times.append(log["time"])
            log["commanded_thruster_forces"][:] = 1e9
            log["ftc_action"] = "observer_attempted_override"
            return {"time_s": log["time"], "available": True, "updated": True}

    observer = Observer()
    simulator = SixDOFSimulator(
        ftc_supervisor=FTCSafetySupervisor(), diagnostic_observer=observer
    )
    baseline = SixDOFSimulator(ftc_supervisor=FTCSafetySupervisor())
    simulator.reset()
    baseline.reset()
    target = PoseTarget(np.array([0., 0., 2.]), np.zeros(3))
    for _ in range(4):
        actual = simulator.step(target, dt=0.05)
        expected = baseline.step(target, dt=0.05)
        assert actual["maintenance_diagnosis"]["time_s"] == actual["time"]
        assert actual["maintenance_diagnosis"]["affects_ftc"] is False
        assert actual["ftc_action"] == expected["ftc_action"]
        np.testing.assert_array_equal(actual["position_ned"], expected["position_ned"])
        np.testing.assert_array_equal(actual["commanded_thruster_forces"],
                                      expected["commanded_thruster_forces"])
    assert len(observer.times) == 4
    simulator.reset()
    assert observer.times == []
    assert simulator.logs == []


def test_shadow_operates_without_ftc_supervisor():
    class Observer:
        def reset(self):
            pass

        def update(self, log):
            return {"time_s": log["time"]}

    simulator = SixDOFSimulator(diagnostic_observer=Observer())
    simulator.reset()
    log = simulator.step(PoseTarget(np.zeros(3), np.zeros(3)), dt=0.05)
    assert log["maintenance_diagnosis"]["execution_mode"] == "online_shadow"
    assert log["step_wall_ms"] >= log["maintenance_diagnosis"]["update_wall_ms"]
