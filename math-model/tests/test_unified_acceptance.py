"""Tests for the unified six-DOF acceptance aggregator."""

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.unified_acceptance import (
    evaluate_unified_acceptance,
    nested_value,
)


def protocol(checks=None, categories=("safety",)):
    return {
        "configuration": {
            "required_categories": list(categories),
            "diagnostic_policy": {"weak_fault": "advisory"},
        },
        "acceptance_checks": checks or [{
            "id": "recall",
            "category": "safety",
            "description": "recall gate",
            "source": "run",
            "value_path": "summary.recall",
            "operator": ">=",
            "threshold": 0.95,
        }],
        "informational_metrics": [{
            "id": "location",
            "description": "location information",
            "source": "run",
            "value_path": "summary.location",
        }],
    }


def test_nested_value_resolves_mapping_path():
    assert nested_value({"a": {"b": 3}}, "a.b") == 3


def test_unified_acceptance_separates_gates_from_information():
    result = evaluate_unified_acceptance(
        protocol(),
        {"run": {"summary": {"recall": 1.0, "location": 0.55}}},
    )

    assert result["decision"] == "accepted"
    assert result["all_acceptance_checks_passed"]
    assert result["categories"]["safety"]["all_passed"]
    assert result["informational_metrics"][0]["observed"] == 0.55
    assert not result["informational_metrics"][0]["gating"]


def test_unified_acceptance_reports_failed_gate():
    result = evaluate_unified_acceptance(
        protocol(),
        {"run": {"summary": {"recall": 0.80, "location": 0.55}}},
    )

    assert result["decision"] == "not_accepted"
    assert not result["checks"][0]["passed"]


def test_unified_acceptance_rejects_duplicate_check_ids():
    check = protocol()["acceptance_checks"][0]
    with pytest.raises(ValueError, match="duplicate acceptance check IDs"):
        evaluate_unified_acceptance(
            protocol([check, dict(check)]),
            {"run": {"summary": {"recall": 1.0, "location": 0.55}}},
        )


def test_unified_acceptance_requires_every_declared_category():
    with pytest.raises(ValueError, match="required category has no checks"):
        evaluate_unified_acceptance(
            protocol(categories=("safety", "missing")),
            {"run": {"summary": {"recall": 1.0, "location": 0.55}}},
        )


def test_unified_acceptance_rejects_missing_evidence_path():
    with pytest.raises(KeyError, match="missing evidence path"):
        evaluate_unified_acceptance(
            protocol(),
            {"run": {"summary": {"location": 0.55}}},
        )
