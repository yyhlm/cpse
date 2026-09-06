from __future__ import annotations

from typing import Any


def dsl_to_json_schema(dsl: Any) -> dict[str, Any]:
    """Convert the repository's field-oriented schema DSL to JSON Schema."""
    if not isinstance(dsl, dict):
        raise ValueError("Schema DSL root must be an object mapping field names to nodes.")
    return _object_schema(dsl, root=True)


def _object_schema(properties: dict[str, Any], root: bool = False) -> dict[str, Any]:
    required: list[str] = []
    converted: dict[str, Any] = {}
    for name, node in properties.items():
        if root and name == "description" and isinstance(node, str):
            continue
        if not isinstance(node, dict):
            raise ValueError(f"Schema DSL node {name!r} must be an object.")
        converted[str(name)] = _node_schema(node, str(name))
        if node.get("required") is True and not node.get("gold_optional"):
            required.append(str(name))
    result: dict[str, Any] = {
        "type": "object",
        "properties": converted,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    if root:
        result["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return result


def _node_schema(node: dict[str, Any], path: str) -> dict[str, Any]:
    raw_type = node.get("type")
    if raw_type is None:
        raise ValueError(f"Schema DSL node {path!r} has no type.")
    types = raw_type if isinstance(raw_type, list) else [raw_type]
    if not types or any(not isinstance(item, str) for item in types):
        raise ValueError(f"Schema DSL node {path!r} has an invalid type.")

    if "object" in types:
        properties = node.get("properties")
        if properties is None and len(types) > 1:
            # Empty arrays in the gold corpus can only reveal that arbitrary JSON
            # values are allowed; no object-field contract can be inferred.
            return {}
        if not isinstance(properties, dict):
            raise ValueError(f"Object schema node {path!r} has no properties object.")
        result = _object_schema(properties)
        if len(types) > 1:
            result["type"] = types
        return result

    result: dict[str, Any] = {"type": types[0] if len(types) == 1 else types}
    if "array" in types:
        items = node.get("items")
        if items is None:
            raise ValueError(f"Array schema node {path!r} has no items definition.")
        result["items"] = _items_schema(items, path)
    return result


def _items_schema(items: Any, path: str) -> dict[str, Any]:
    if isinstance(items, str):
        return {"type": items}
    if not isinstance(items, dict):
        raise ValueError(f"Array items for {path!r} must be a type string or node object.")
    if "type" in items:
        return _node_schema(items, f"{path}[]")
    return _object_schema(items)
