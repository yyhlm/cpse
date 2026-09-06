from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator

from .models import PredictionArtifact


def validate_prediction(raw_response: str, json_schema: dict[str, Any]) -> PredictionArtifact:
    """Parse exactly what the model returned; never repair malformed output."""
    try:
        prediction = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return PredictionArtifact(
            raw_response=raw_response,
            parsed_prediction=None,
            validation_errors=(
                {
                    "path": "$",
                    "message": f"Invalid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}",
                },
            ),
        )

    validator = Draft202012Validator(json_schema)
    errors = sorted(validator.iter_errors(prediction), key=lambda error: list(error.absolute_path))
    if errors:
        return PredictionArtifact(
            raw_response=raw_response,
            parsed_prediction=None,
            validation_errors=tuple(
                {
                    "path": _format_path(error.absolute_path),
                    "message": error.message,
                }
                for error in errors
            ),
        )
    return PredictionArtifact(raw_response=raw_response, parsed_prediction=prediction, validation_errors=())


def prediction_or_failure_payload(artifact: PredictionArtifact) -> Any:
    if artifact.is_valid:
        return artifact.parsed_prediction
    # A schema mismatch is still useful evidence for the judge and optimizer if
    # the model returned parseable JSON. Preserve it verbatim rather than
    # replacing it with an invalid_prediction envelope; malformed JSON remains
    # an explicit failure payload.
    try:
        return json.loads(artifact.raw_response)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return {
        "status": "invalid_prediction",
        "raw_response": artifact.raw_response,
        "validation_errors": list(artifact.validation_errors),
    }


def _format_path(parts: Any) -> str:
    result = "$"
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def _empty_value_for(schema_node: dict[str, Any]) -> Any:
    """Return a schema-compatible empty placeholder for an absent required key.

    Never invents content: only structural empty values (string "", object {},
    array []). Nested-object empties are filled recursively so every required
    descendant exists, which is what turns a "missing required property" failure
    into a valid (if sparse) record.
    """
    if not isinstance(schema_node, dict):
        return ""
    types = schema_node.get("type")
    if isinstance(types, list):
        types_set = set(types)
    elif isinstance(types, str):
        types_set = {types}
    else:
        types_set = set()
    if "array" in types_set:
        return []
    if "object" in types_set or "properties" in schema_node:
        empty: dict[str, Any] = {}
        props = schema_node.get("properties") or {}
        for key in schema_node.get("required", []) or []:
            if key in props and isinstance(props[key], dict):
                empty[key] = _empty_value_for(props[key])
        return empty
    return ""


def fill_missing_required(prediction: Any, json_schema: dict[str, Any]) -> Any:
    """Deterministically add schema-required keys the prediction omitted.

    Walks the prediction against the JSON Schema and, for every object node,
    inserts a schema-compatible empty placeholder for each declared ``required``
    key that is absent. Existing values are never touched and no content is
    invented — only structural empties (string "", object {}, array []) are
    added so a "missing required property" validation failure resolves without
    another model call. Arrays recurse into their item schema; nested object
    required keys are filled recursively so a multi-level omission (e.g.
    测试速率 missing 单值 inside 性质.测试条件) is fixed in one pass.
    """

    def fill(node: Any, schema_node: dict[str, Any] | None) -> Any:
        if not isinstance(schema_node, dict):
            return node
        types = schema_node.get("type")
        types_set = set(types) if isinstance(types, list) else ({types} if isinstance(types, str) else set())
        if isinstance(node, dict) and ("object" in types_set or "properties" in schema_node):
            props = schema_node.get("properties") or {}
            required = schema_node.get("required") or []
            result = dict(node)
            for key in required:
                if key not in result and key in props:
                    result[key] = _empty_value_for(props[key])
            for key, value in list(result.items()):
                if key in props and isinstance(props[key], dict):
                    result[key] = fill(value, props[key])
            return result
        if isinstance(node, list) and "array" in types_set:
            items_schema = schema_node.get("items")
            if isinstance(items_schema, dict):
                return [fill(item, items_schema) for item in node]
        return node

    if not isinstance(json_schema, dict):
        return prediction
    return fill(prediction, json_schema)
