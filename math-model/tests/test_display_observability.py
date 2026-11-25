from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from presentation.six_dof_observability import display_observability, safety_display_lines
from presentation.six_dof_demo_adapter import adapt_log
from environment.six_dof_simulator import SixDOFSimulator
from sensors.six_dof_sensor_suite import SixDOFSensorSuite
from simple_control.six_dof_controller import PoseTarget
import numpy as np


def healthy():
    metadata = dict(time_s=1., quality='nominal', sources=dict(attitude='imu', angular_velocity='imu',
        linear_velocity='dvl', position_xy='dvl_dead_reckoning', depth='depth_sensor'),
        sensor_health={k: dict(trust_level='trusted', fault_type='normal') for k in ('imu', 'dvl', 'depth')},
        horizontal_reference='initial_dead_reckoning', excluded_sensors=(), rejected_sensors=())
    log = dict(time=1.05, sensor_feedback_enabled=True, controller_state_source='sensor_estimate_previous_step',
        controller_estimate_metadata=deepcopy(metadata), sensor_health=deepcopy(metadata['sensor_health']),
        state_estimate_sources=deepcopy(metadata['sources']), horizontal_position_reference='initial_dead_reckoning')
    for key in ('tilt_angle_deg', 'yaw_tracking_error_deg', 'yaw_rate_tracking_error_deg_s',
                'depth_tracking_error_m', 'horizontal_tracking_error_m', 'cross_track_error_m'):
        log['ftc_' + key] = 0.
    return log


def lose_imu(log):
    log['state_estimate_sources'].update(attitude='held_last_attitude', angular_velocity='zero_rate_safe_hold')
    log['sensor_health']['imu']['trust_level'] = 'untrusted'
    log['horizontal_position_reference'] = 'degraded_without_absolute_fix'
    log['ftc_guidance_mode_applied'] = 'safety_pause_untrusted_attitude'


def test_finite_zero_is_available_with_valid_provenance():
    obs = display_observability(healthy())
    assert obs['metrics']['tilt_angle_deg'] == 0.
    assert obs['controller_input']['time_s'] == 1.
    assert obs['current_estimate']['time_s'] == 1.05


def test_held_attitude_and_zero_rate_are_not_observations():
    log = healthy(); lose_imu(log)
    obs = display_observability(log)
    assert obs['metrics']['tilt_angle_deg'] is None
    assert obs['metrics']['yaw_rate_tracking_error_deg_s'] is None
    assert obs['current_estimate']['horizontal_velocity_body']['status'] == 'measured'
    first, second, tier = safety_display_lines({'observability': obs})
    assert 'PAUSED' in first and 'tilt --' in first
    assert 'Att LOST' in second and 'rate LOST' in second and 'V(body) OK' in second
    assert tier != 'normal'


def test_first_recovery_frame_keeps_controller_unavailable():
    log = healthy()
    log['controller_estimate_metadata']['sources'].update(attitude='held_last_attitude', angular_velocity='zero_rate_safe_hold')
    log['controller_estimate_metadata']['sensor_health']['imu']['trust_level'] = 'untrusted'
    obs = display_observability(log)
    assert obs['metrics']['yaw_tracking_error_deg'] is None
    assert 'Att recovering' in safety_display_lines({'observability': obs})[1]


def test_dvl_recovery_does_not_restore_xy_reference():
    log = healthy()
    log['horizontal_position_reference'] = 'degraded_without_absolute_fix'
    log['controller_estimate_metadata']['horizontal_reference'] = 'degraded_without_absolute_fix'
    obs = display_observability(log)
    assert obs['current_estimate']['horizontal_velocity_body']['status'] == 'measured'
    assert obs['metrics']['horizontal_tracking_error_m'] is None
    assert 'XY DEG (no abs fix)' in safety_display_lines({'observability': obs})[1]


def test_legacy_missing_metadata_remains_unknown():
    log = healthy(); del log['controller_estimate_metadata']
    obs = display_observability(log)
    assert obs['legacy_provenance_missing']
    assert all(value is None for value in obs['metrics'].values())
    assert 'Att ?' in safety_display_lines({'observability': obs})[1]


def test_truth_never_changes_observability():
    log = healthy(); before = display_observability(log)
    log.update(position_ned=[999, 999, 999], euler_rpy=[2, 2, 2],
               sensor_fault_truth={'imu': 'normal'}, actual_thruster_forces=[100]*6)
    assert display_observability(log) == before


def test_nonfinite_or_missing_metrics_not_zero():
    log = healthy(); log['ftc_tilt_angle_deg'] = float('nan'); del log['ftc_yaw_tracking_error_deg']
    obs = display_observability(log)
    assert obs['metrics']['tilt_angle_deg'] is None and obs['metrics']['yaw_tracking_error_deg'] is None


def test_prior_and_truth_feedback_not_measured():
    for source in ('initial_pose_prior', 'simulator_truth'):
        log = healthy(); log['controller_state_source'] = source
        assert display_observability(log)['metrics']['tilt_angle_deg'] is None


def test_summary_and_technical_render_share_unavailability():
    from presentation.six_dof_demo_renderer import SixDOFDemoRenderer
    log = healthy(); lose_imu(log); frame = adapt_log(log)
    for mode in ('summary', 'technical'):
        renderer = SixDOFDemoRenderer([frame, dict(frame, time_s=1.1)], view_mode=mode)
        try:
            renderer.draw(0)
            text = '\n'.join(t.get_text() for a in renderer.figure.axes for t in a.texts)
            assert 'tilt --' in text and 'Att LOST' in text and 'SIM TRUE' in text
            assert 'SAFE |' not in text
            if mode == 'summary':
                assert 'Mission        Paused | attitude untrusted' in text
                assert 'PAUSED / LOCAL HOLD' not in text and 'Mission        Active' not in text
        finally:
            renderer.close()
