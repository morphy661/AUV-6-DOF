"""Operator-contract tests; no control code or injection truth is consumed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from presentation.six_dof_demo_adapter import adapt_log, extract_demo_events


def frame(t, mode, request=False, phase="inactive"):
    trusted = not ("untrusted" in mode or "guard" in mode)
    return adapt_log(dict(time=t, ftc_guidance_mode_applied=mode,
        ftc_mission_abort_requested=request, mission_recovery=dict(phase=phase,
        external_recovery_required=phase=="abort_external_recovery"),
        state_estimate_sources={"attitude":"imu", "angular_velocity":"imu"},
        sensor_health={"imu":{"trust_level":"trusted" if trusted else "untrusted"}}))


def test_recurrence_stays_in_same_locked_episode_and_request_is_not_confirmation():
    modes=["normal","attitude_feedback_guard","safety_pause_untrusted_attitude",
           "safety_pause","safety_pause_untrusted_attitude","safety_pause"]
    frames=[frame(i,mode,i>=2) for i,mode in enumerate(modes)]
    events=[e for e in extract_demo_events(frames) if e["category"]=="mission"]
    assert [e["state"] for e in events]==["temporary_guard","paused_untrusted","paused_trusted",
                                         "paused_untrusted","paused_trusted"]
    assert {e["episode"] for e in events}=={1}
    assert all(e["external_takeover_confirmed"] is None and e["external_request_sent"] is None for e in events)


def test_navigation_relocalization_is_not_resume_or_external_takeover():
    for phase,state in (("pause","navigation_pause"),("relocalized_wait","relocalized_wait"),
                        ("abort_external_recovery","external_recovery")):
        h=frame(1,"safety_pause",True,phase)["mission_handoff"]
        assert h["state"]==state and h["external_recovery_required"]==(state=="external_recovery")
        assert h["external_takeover_confirmed"] is None
    assert adapt_log({})["mission_handoff"]["state"]=="unknown"


def test_requested_pause_and_its_application_share_one_episode():
    frames = [frame(13.95, "normal"), frame(14.0, "normal", True),
              frame(14.05, "safety_pause", True, "pause")]
    events = [e for e in extract_demo_events(frames) if e["category"] == "mission"]
    assert [(e["time_s"], e["state"], e["episode"]) for e in events] == [
        (14.0, "pause_pending", 1), (14.05, "navigation_pause", 1)]
    assert frames[1]["ftc"]["guidance_mode_applied"] == "normal"
    assert events[0]["level"] != "normal" and events[0]["external_request_sent"] is None
