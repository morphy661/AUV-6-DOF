"""Presentation adapters and renderers for six-DOF diagnostics."""

from presentation.six_dof_demo_adapter import (
    adapt_log,
    adapt_logs,
    extract_demo_events,
    summarize_demo,
)
from presentation.six_dof_model_bridge import SixDOFModelBridge
from presentation.advisory_context_gate import (
    AdvisoryContextGate,
    AdvisoryGateConfig,
    AdvisoryGateDecision,
)

__all__ = [
    "adapt_log",
    "adapt_logs",
    "extract_demo_events",
    "summarize_demo",
    "SixDOFModelBridge",
    "AdvisoryContextGate",
    "AdvisoryGateConfig",
    "AdvisoryGateDecision",
]
