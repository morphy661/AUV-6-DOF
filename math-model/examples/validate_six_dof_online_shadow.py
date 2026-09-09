"""Paired real-checkpoint validation of streaming shadow inference."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "examples"):
    sys.path.insert(0, str(root))

from demo_six_dof_unified_diagnostics import (
    DEFAULT_CHECKPOINT, DEFAULT_TEMPORAL_CONFIG, MODEL_ROOT, MISSION_TARGETS,
    run_demo, target_provider, disturbance_provider, summarize_online_diagnosis,
    save_online_trace, _sha256,
)
from environment.six_dof_simulator import SixDOFSimulator
from environment.six_dof_dynamics import SixDOFState
from ftc.safety_supervisor import FTCSafetySupervisor
from sensors.six_dof_sensor_suite import SixDOFSensorSuite
from presentation.six_dof_model_bridge import SixDOFModelBridge
from utils.six_dof_feature_extractor import PRIVILEGED_SIMULATOR_FIELDS

sys.path.insert(0, str(MODEL_ROOT))
from model_six_dof_multitask import AUVSixDOFMultiTaskDetector


def normal_run(duration, dt, seed, bridge):
    simulator = SixDOFSimulator(
        sensor_suite=SixDOFSensorSuite(seed=seed),
        ftc_supervisor=FTCSafetySupervisor(), diagnostic_observer=bridge,
    )
    simulator.reset(SixDOFState(position_ned=np.array(MISSION_TARGETS[0][1])))
    return simulator.run(duration, dt, target_provider, disturbance_provider)


def assert_tree_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        np.testing.assert_equal(left, right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30.)
    parser.add_argument("--dt", type=float, default=.05)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bridge = SixDOFModelBridge(DEFAULT_CHECKPOINT, DEFAULT_TEMPORAL_CONFIG,
                              AUVSixDOFMultiTaskDetector, device=args.device)
    cases = [("normal", 20260907), ("fixed", 20260907),
             ("random", 20260911), ("random", 20260922)]
    results = []
    for case, seed in cases:
        print(f"Validating {case} seed {seed}", flush=True)
        if case == "normal":
            baseline = normal_run(args.duration, args.dt, seed, None)
            online = normal_run(args.duration, args.dt, seed, bridge)
            manifest = None
        else:
            baseline, _, _, manifest = run_demo(args.duration, args.dt, seed,
                                               injection_mode=case)
            online, _, _, _ = run_demo(args.duration, args.dt, seed,
                                       injection_mode=case, model_bridge=bridge)
        assert len(baseline) == len(online)
        for a, b in zip(baseline, online):
            assert_tree_equal(
                {k: v for k, v in a.items() if k != "step_wall_ms"},
                {k: v for k, v in b.items()
                 if k not in ("step_wall_ms", "maintenance_diagnosis")},
            )
        bridge.enrich_logs(baseline)
        for a, b in zip(baseline, online):
            expected = a["maintenance_diagnosis"]
            actual = {k: v for k, v in b["maintenance_diagnosis"].items()
                      if k not in ("execution_mode", "affects_ftc", "update_wall_ms")}
            assert_tree_equal(expected, actual)
            if actual["available"]:
                assert actual["window_end_time_s"] <= b["time"]
                assert actual["inference_time_s"] <= b["time"]
        # Replaying a prefix with ALL privileged truth removed must reproduce
        # the online prefix; this also verifies reset and future independence.
        prefix = deepcopy(baseline[:len(baseline) // 2])
        for log in prefix:
            for field in PRIVILEGED_SIMULATOR_FIELDS:
                log.pop(field, None)
            log.pop("maintenance_diagnosis", None)
        bridge.enrich_logs(prefix)
        for a, b in zip(prefix, baseline):
            assert_tree_equal(a["maintenance_diagnosis"], b["maintenance_diagnosis"])
        timing = summarize_online_diagnosis(online, args.dt)
        assert timing["inference_count"] > 0
        item = {
            "case": case, "seed": seed, "steps": len(online),
            "physical_and_ftc_logs_exactly_equal": True,
            "online_and_replay_diagnoses_exactly_equal": True,
            "truth_removed_prefix_exactly_equal": True,
            "manifest": manifest, "timing": timing,
        }
        results.append(item)
        save_online_trace(online, args.output_dir / f"{case}_{seed}_trace.jsonl")
        print(json.dumps(item), flush=True)
    report = {
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "checkpoint_sha256": _sha256(DEFAULT_CHECKPOINT),
        "temporal_config_sha256": _sha256(DEFAULT_TEMPORAL_CONFIG),
        "device": str(bridge.device), "dt_s": args.dt,
        "runtime_preparation_ms": bridge.runtime_preparation_ms,
        "sequence_length": bridge.sequence_length,
        "inference_stride": bridge.inference_stride,
        "functional_checks_passed": True, "cases": results,
        "note": "Synchronous simulation timing is measured, not a hard real-time guarantee. "
                "Device preparation is reported separately before mission start; "
                "all measured mission updates include OS scheduling delays.",
    }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("All paired functional checks passed.", flush=True)


if __name__ == "__main__":
    main()
