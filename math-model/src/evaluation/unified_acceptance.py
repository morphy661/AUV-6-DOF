"""Protocol-driven aggregation for the final six-DOF acceptance report."""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


_OPERATORS = {
    "==": operator.eq,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


def nested_value(value: Mapping[str, Any], path: str) -> Any:
    """Resolve a dot-separated key path from nested mappings."""

    current: Any = value
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"missing evidence path: {path}")
        current = current[key]
    return current


def _evaluate_spec(
    spec: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_name = str(spec["source"])
    if source_name not in sources:
        raise KeyError(f"unknown evidence source: {source_name}")
    comparison = str(spec["operator"])
    if comparison not in _OPERATORS:
        raise ValueError(f"unsupported comparison operator: {comparison}")
    observed = nested_value(sources[source_name], str(spec["value_path"]))
    threshold = spec["threshold"]
    if isinstance(observed, float) and not math.isfinite(observed):
        raise ValueError(f"non-finite evidence value: {spec['id']}")
    return {
        "id": str(spec["id"]),
        "category": str(spec["category"]),
        "description": str(spec["description"]),
        "source": source_name,
        "value_path": str(spec["value_path"]),
        "observed": observed,
        "operator": comparison,
        "threshold": threshold,
        "passed": bool(_OPERATORS[comparison](observed, threshold)),
    }


def evaluate_unified_acceptance(
    protocol: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate all safety gates and collect non-gating model information."""

    specs = protocol.get("acceptance_checks")
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)):
        raise TypeError("acceptance_checks must be a sequence")
    ids = [str(spec["id"]) for spec in specs]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate acceptance check IDs: {duplicates}")

    checks = [_evaluate_spec(spec, sources) for spec in specs]
    required_categories = tuple(
        protocol["configuration"]["required_categories"]
    )
    categories = {}
    for category in required_categories:
        selected = [check for check in checks if check["category"] == category]
        if not selected:
            raise ValueError(f"required category has no checks: {category}")
        categories[category] = {
            "check_count": len(selected),
            "passed_count": sum(check["passed"] for check in selected),
            "all_passed": all(check["passed"] for check in selected),
        }

    information = []
    for spec in protocol.get("informational_metrics", []):
        source_name = str(spec["source"])
        information.append({
            "id": str(spec["id"]),
            "description": str(spec["description"]),
            "source": source_name,
            "value_path": str(spec["value_path"]),
            "observed": nested_value(
                sources[source_name], str(spec["value_path"])
            ),
            "gating": False,
        })

    all_passed = all(item["all_passed"] for item in categories.values())
    return {
        "decision": "accepted" if all_passed else "not_accepted",
        "all_acceptance_checks_passed": all_passed,
        "check_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "categories": categories,
        "checks": checks,
        "informational_metrics": information,
        "diagnostic_policy": protocol["configuration"]["diagnostic_policy"],
    }
