from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from typing import Any

_DIFF_CATEGORIES = (
    "missing_from_prediction",
    "extra_in_prediction",
    "changed_value",
    "type_or_shape_mismatch",
)
_AUDIT_STATUSES = ("supported", "unsupported", "ambiguous", "possible_omission")


def diff_json(prediction: Any, gold: Any) -> dict[str, list[str]]:
    """Return a stable, JSONPath-indexed structural comparison with Gold."""
    result = {category: [] for category in _DIFF_CATEGORIES}
    _diff_at_path(prediction, gold, "$", result)
    return {category: sorted(paths) for category, paths in result.items()}


def diff_summary(diff: dict[str, list[str]], sample_limit: int = 12) -> dict[str, dict[str, Any]]:
    return {
        category: {"count": len(diff.get(category, [])), "sample_paths": list(diff.get(category, []))[:sample_limit]}
        for category in _DIFF_CATEGORIES
    }


def analyze_blind_document(
    *,
    document_id: str,
    baseline: dict[str, Any],
    optimized: dict[str, Any],
    gold: Any,
    audit_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Create a no-model explanation record from saved comparison evidence."""
    baseline_prediction = baseline.get("prediction")
    optimized_prediction = optimized.get("prediction")
    baseline_score = _number_or_none(baseline.get("score"))
    optimized_score = _number_or_none(optimized.get("score"))
    audit_counts = {status: int((audit_counts or {}).get(status, 0)) for status in _AUDIT_STATUSES}

    has_baseline = baseline_prediction is not None and baseline_score is not None
    has_optimized = optimized_prediction is not None and optimized_score is not None
    if not has_baseline or not has_optimized:
        return {
            "version": 1,
            "document_id": document_id,
            "scores": {"baseline": baseline_score, "optimized": optimized_score, "delta": None},
            "arms": {"baseline": _arm_evidence(baseline), "optimized": _arm_evidence(optimized)},
            "classification": "execution_failure",
            "comparison": None,
            "judge_feedback": _feedback_evidence(baseline, optimized),
            "gold_audit_context": audit_counts,
            "limitations": _limitations(),
        }

    baseline_to_gold = diff_json(baseline_prediction, gold)
    optimized_to_gold = diff_json(optimized_prediction, gold)
    baseline_to_optimized = diff_json(baseline_prediction, optimized_prediction)
    delta = optimized_score - baseline_score
    baseline_error_total = _gold_error_total(baseline_to_gold)
    optimized_error_total = _gold_error_total(optimized_to_gold)
    classification = _classify(delta, baseline_error_total, optimized_error_total)
    return {
        "version": 1,
        "document_id": document_id,
        "scores": {"baseline": baseline_score, "optimized": optimized_score, "delta": delta},
        "arms": {"baseline": _arm_evidence(baseline), "optimized": _arm_evidence(optimized)},
        "classification": classification,
        "comparison": {
            "baseline_to_gold": {"full": baseline_to_gold, "counts": diff_summary(baseline_to_gold)},
            "optimized_to_gold": {"full": optimized_to_gold, "counts": diff_summary(optimized_to_gold)},
            "baseline_to_optimized": {"full": baseline_to_optimized, "counts": diff_summary(baseline_to_optimized)},
        },
        "judge_feedback": _feedback_evidence(baseline, optimized),
        "gold_audit_context": audit_counts,
        "limitations": _limitations(),
    }


def summarize_blind_analysis(documents: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [document for document in documents if isinstance(document.get("scores", {}).get("delta"), (int, float))]
    deltas = [(str(document["document_id"]), float(document["scores"]["delta"])) for document in valid]
    paired_sum = sum(delta for _document_id, delta in deltas)
    top_positive = sorted(((document_id, delta) for document_id, delta in deltas if delta > 0), key=lambda item: (-item[1], item[0]))
    negative = sorted(((document_id, delta) for document_id, delta in deltas if delta < 0), key=lambda item: (item[1], item[0]))
    aggregate = {
        arm: {category: 0 for category in _DIFF_CATEGORIES}
        for arm in ("baseline_to_gold", "optimized_to_gold")
    }
    for document in valid:
        comparison = document.get("comparison") or {}
        for arm in aggregate:
            for category in _DIFF_CATEGORIES:
                aggregate[arm][category] += int(comparison.get(arm, {}).get("counts", {}).get(category, {}).get("count", 0))
    classifications = Counter(str(document.get("classification", "unavailable")) for document in documents)
    return {
        "version": 1,
        "document_count": len(documents),
        "paired_valid_count": len(valid),
        "mean_delta": paired_sum / len(valid) if valid else None,
        "median_delta": statistics.median(delta for _document_id, delta in deltas) if deltas else None,
        "paired_delta_sum": paired_sum,
        "largest_positive_deltas": [{"document_id": document_id, "delta": delta, "contribution": delta / paired_sum if paired_sum else None} for document_id, delta in top_positive[:3]],
        "largest_negative_deltas": [{"document_id": document_id, "delta": delta, "contribution": delta / paired_sum if paired_sum else None} for document_id, delta in negative[:3]],
        "top_positive_contribution_share": sum(delta for _document_id, delta in top_positive[:3]) / paired_sum if paired_sum > 0 else None,
        "aggregate_gold_diff_counts": aggregate,
        "classification_counts": dict(sorted(classifications.items())),
    }


def _diff_at_path(prediction: Any, gold: Any, path: str, result: dict[str, list[str]]) -> None:
    prediction_kind = _kind(prediction)
    gold_kind = _kind(gold)
    if prediction_kind != gold_kind:
        result["type_or_shape_mismatch"].append(path)
        return
    if isinstance(gold, dict):
        for key in sorted(set(gold) | set(prediction), key=str):
            child_path = _object_path(path, str(key))
            if key not in prediction:
                result["missing_from_prediction"].append(child_path)
            elif key not in gold:
                result["extra_in_prediction"].append(child_path)
            else:
                _diff_at_path(prediction[key], gold[key], child_path, result)
        return
    if isinstance(gold, list):
        for index in range(max(len(gold), len(prediction))):
            child_path = f"{path}[{index}]"
            if index >= len(prediction):
                result["missing_from_prediction"].append(child_path)
            elif index >= len(gold):
                result["extra_in_prediction"].append(child_path)
            else:
                _diff_at_path(prediction[index], gold[index], child_path, result)
        return
    if prediction != gold:
        result["changed_value"].append(path)


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _object_path(parent: str, key: str) -> str:
    return f"{parent}.{key}" if key.isidentifier() else f"{parent}[{json.dumps(key, ensure_ascii=False)}]"


def _number_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _arm_evidence(arm: dict[str, Any]) -> dict[str, Any]:
    validation_errors = arm.get("validation_errors")
    return {
        "status": arm.get("status", "unknown"),
        "prediction_available": arm.get("prediction") is not None,
        "validation_error_count": len(validation_errors) if isinstance(validation_errors, list) else 0,
        "repair_attempts": arm.get("repair_attempts", 0),
        "failure": arm.get("failure"),
        "artifact_paths": arm.get("artifact_paths", {}),
    }


def _feedback_evidence(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, str | None]:
    return {"baseline": _text_or_none(baseline.get("feedback")), "optimized": _text_or_none(optimized.get("feedback"))}


def _text_or_none(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _gold_error_total(diff: dict[str, list[str]]) -> int:
    return sum(len(diff[category]) for category in _DIFF_CATEGORIES)


def _classify(delta: float, baseline_errors: int, optimized_errors: int) -> str:
    if delta > 0 and optimized_errors < baseline_errors:
        return "observable_improvement"
    if delta < 0 and optimized_errors > baseline_errors:
        return "observable_regression"
    return "mixed_or_inconclusive"


def _limitations() -> list[str]:
    return [
        "Deterministic JSON differences and judge feedback are co-occurring evidence, not proof that a field change caused the score change.",
        "Gold-audit findings are advisory PDF-versus-Gold context and cannot establish which prediction is better.",
    ]
