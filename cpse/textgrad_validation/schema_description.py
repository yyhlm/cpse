"""Deterministic description-only patches for field-oriented schema DSLs."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from .schema_contract import dsl_to_json_schema

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_JSONPATH_RE = re.compile(r"\$(?:\.|\[)")
_MEASUREMENT_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?|\.\d+)\s*(?:wt\s*%|vol\s*%|mol\s*%|%|°\s*[CF]|K|Pa|kPa|MPa|bar|atm|mL|µL|uL|L|mg|g|kg|mm|cm|nm|µm|um|mol|mmol|M|mM|µM|h|min|s|rpm|eV)(?![A-Za-z])",
    re.IGNORECASE,
)
_IDENTIFIER_RE = re.compile(r"\b(?:PMID|PMCID|arXiv|ISBN|ISSN)\s*[:#]?\s*[A-Z0-9._/-]+", re.IGNORECASE)
_EXAMPLE_RE = re.compile(r"\b(?:e\.g\.?|i\.e\.?|for example|example\s*:|sample\s*\d+|table\s*\d+|figure\s*\d+)\b", re.IGNORECASE)
_DOCUMENT_FACT_RE = re.compile(r"\b(?:this|the)\s+(?:paper|article|study|document)\b", re.IGNORECASE)
_CONDITION_VALUE_RE = re.compile(
    r"\b(?:p\s*h\s*\d+(?:\.\d+)?|(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+(?:\.\d+)?)\s+(?:equiv(?:alent)?s?|eq\.?|fold|hours?|minutes?|seconds?|days?)|room\s+temperature|ambient\s+temperature)\b",
    re.IGNORECASE,
)


class SchemaDescriptionPatchError(ValueError):
    """Raised when a description patch cannot safely apply to a schema DSL."""


# Upper bound on patches per proposal. A runaway patch that rewrites every
# description with boilerplate exceeds output budgets and truncates; reject it
# deterministically so the round degrades to proposal_failed with a clear reason.
_MAX_PATCHES = 20


def canonical_structure(dsl: Any) -> Any:
    """Return a deep-copied DSL view with every ``description`` key removed."""
    if isinstance(dsl, dict):
        return {str(key): canonical_structure(value) for key, value in dsl.items() if key != "description"}
    if isinstance(dsl, list):
        return [canonical_structure(value) for value in dsl]
    return copy.deepcopy(dsl)


def structural_fingerprint(dsl: Any) -> str:
    """SHA-256 of canonical DSL structure, deliberately ignoring descriptions."""
    payload = json.dumps(canonical_structure(dsl), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Markers that count as "the required key must still be emitted". TextGrad
# sometimes replaces a required field's original "always emit" wording with
# "preserve explicitly reported…" language, which makes the model omit the key
# when the value is absent (e.g. 0b30cafa/206bd242 dropping 红外_FTIR/紫外_UVVis).
_REQUIRED_EMISSION_MARKERS = ("固定输出", "必须输出", "必填键", "不得省略")
_REQUIRED_EMISSION_REINFORCEMENT = (
    "该键为必填键：来源未报告对应值时，输出该键类型的空值结构（空字符串/空对象/空数组），不得省略该键。"
)


def _ensure_emission_guarantee(description: str) -> str:
    if any(marker in description for marker in _REQUIRED_EMISSION_MARKERS):
        return description
    return description + " " + _REQUIRED_EMISSION_REINFORCEMENT


def reinforce_required_emission(dsl: Any) -> Any:
    """Deterministically append a required-key emission guarantee to every required
    description that lacks one, without touching anything else.

    Walks the DSL and, for every node flagged ``required: True`` whose
    ``description`` no longer carries an emission marker, appends a canonical
    sentence so the materialized schema always instructs the model to emit the
    key (using an empty-value structure when the source reports nothing). This is
    description-only, so ``structural_fingerprint`` is preserved. Applied to the
    base schema at load time and again after every description patch, it cannot be
    defeated by an aggressive optimization step.
    """
    if isinstance(dsl, dict):
        required = dsl.get("required") is True
        result: dict[str, Any] = {}
        for key, value in dsl.items():
            if key == "description" and isinstance(value, str) and required:
                value = _ensure_emission_guarantee(value)
            result[key] = reinforce_required_emission(value)
        return result
    if isinstance(dsl, list):
        return [reinforce_required_emission(value) for value in dsl]
    return copy.deepcopy(dsl)


def parse_description_patch_text(text: str) -> dict[str, Any]:
    """Accept only one raw JSON object; Markdown fences and prose are invalid."""
    if not isinstance(text, str) or not text or text != text.strip():
        raise SchemaDescriptionPatchError("Patch response must be a non-empty, whitespace-trimmed JSON object.")
    try:
        document = json.loads(text, parse_constant=_reject_nonstandard_json_constant, object_pairs_hook=_reject_duplicate_json_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SchemaDescriptionPatchError("Patch response must be strict JSON without duplicate keys, non-finite constants, Markdown, or preamble.") from exc
    if not isinstance(document, dict):
        raise SchemaDescriptionPatchError("Patch response root must be a JSON object.")
    return document


def _reject_nonstandard_json_constant(value: str) -> Any:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is not allowed: {key}")
        result[key] = value
    return result


def validate_patch_document(dsl: Any, document: Any) -> list[dict[str, str]]:
    """Validate a direct, generic, description-only patch document against ``dsl``."""
    if not isinstance(dsl, dict):
        raise SchemaDescriptionPatchError("Schema DSL root must be an object.")
    if not isinstance(document, dict) or set(document) != {"patches"}:
        raise SchemaDescriptionPatchError("Patch document must contain exactly one 'patches' array.")
    patches = document["patches"]
    if not isinstance(patches, list):
        raise SchemaDescriptionPatchError("Patch document 'patches' must be an array.")
    if len(patches) > _MAX_PATCHES:
        raise SchemaDescriptionPatchError(
            f"Patch document has {len(patches)} patches; maximum is {_MAX_PATCHES}. "
            "Keep the patch minimal and focused on feedback-implicated descriptions."
        )

    validated: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, patch in enumerate(patches):
        if not isinstance(patch, dict) or set(patch) != {"path", "description"}:
            raise SchemaDescriptionPatchError(f"Patch {index} must contain exactly 'path' and 'description'.")
        path, description = patch["path"], patch["description"]
        if not isinstance(path, str):
            raise SchemaDescriptionPatchError(f"Patch {index} path must be a string.")
        if not isinstance(description, str) or not description.strip():
            raise SchemaDescriptionPatchError(f"Patch {index} description must be a non-blank string.")
        _validate_generic_description(description, index)
        canonical_path = _canonical_description_path(dsl, path)
        if canonical_path in seen:
            raise SchemaDescriptionPatchError(f"Duplicate patch path: {canonical_path}")
        seen.add(canonical_path)
        current = _resolve_description(dsl, canonical_path)
        if not isinstance(current, str):
            raise SchemaDescriptionPatchError(f"Patch {index} target must be an existing string description: {path}")
        validated.append({"path": canonical_path, "description": description})
    return validated


def retain_valid_description_patches(dsl: Any, document: Any) -> tuple[dict[str, Any], list[str]]:
    """Keep independently valid patches and report why invalid siblings were dropped."""
    if not isinstance(document, dict) or set(document) != {"patches"} or not isinstance(document["patches"], list):
        validate_patch_document(dsl, document)
    valid: list[dict[str, Any]] = []
    dropped: list[str] = []
    for patch in document["patches"]:
        try:
            validated = validate_patch_document(dsl, {"patches": [patch]})
        except SchemaDescriptionPatchError as exc:
            dropped.append(str(exc))
        else:
            valid.append(validated[0])
    return {"patches": valid}, dropped


def apply_description_patch_document(dsl: Any, document: Any) -> dict[str, Any]:
    """Return a patched DSL without mutating ``dsl``, preserving all structure."""
    patches = validate_patch_document(dsl, document)
    result = copy.deepcopy(dsl)
    before = structural_fingerprint(result)
    for patch in patches:
        _set_description(result, patch["path"], patch["description"])
    if structural_fingerprint(result) != before:
        raise AssertionError("Description-only patch unexpectedly changed DSL structure.")
    # Deterministic guard rail: whatever the patch did, every required key's
    # description must still instruct the model to emit the key. This keeps the
    # patch's intent while restoring any emission language it removed.
    result = reinforce_required_emission(result)
    try:
        dsl_to_json_schema(result)
    except (TypeError, ValueError) as exc:
        raise SchemaDescriptionPatchError("Patched schema DSL does not materialize to JSON Schema.") from exc
    return result


def _validate_generic_description(description: str, index: int) -> None:
    """Reject identifiers and concrete experimental values; allow generic field guidance."""
    checks = (
        (_DOI_RE, "DOI-style identifier"),
        (_URL_RE, "URL or document identifier"),
        (_JSONPATH_RE, "JSONPath reference"),
        (_MEASUREMENT_RE, "literal measurement or unit value"),
        (_IDENTIFIER_RE, "bibliographic identifier"),
        (_EXAMPLE_RE, "sample-specific example"),
        (_DOCUMENT_FACT_RE, "document-specific fact"),
        (_CONDITION_VALUE_RE, "literal experimental condition"),
    )
    for pattern, label in checks:
        if pattern.search(description):
            raise SchemaDescriptionPatchError(f"Patch {index} description contains prohibited {label}.")


_WRAPPER_PREFIXES = ("schema_dsl.", "schema.", "root.")


def _normalize_dsl_path(path: str) -> str:
    """Strip a gateway wrapper key some patch models prepend to DSL paths.

    The patch model receives the DSL wrapped as ``{"schema_dsl": {...}}`` and may
    anchor every path on that wrapper (e.g. ``$.schema_dsl.聚合物.items...``). DSL
    paths are rooted at the DSL itself, so a leading wrapper segment is dropped.
    No top-level DSL key collides with these prefixes (top-level keys are
    文献信息 / 聚合物 / description).
    """
    if isinstance(path, str) and path.startswith("$."):
        for prefix in _WRAPPER_PREFIXES:
            marker = "$." + prefix
            if path.startswith(marker):
                return "$." + path[len(marker):]
    return path


def _resolve_description(dsl: dict[str, Any], path: str) -> Any:
    parent, key = _resolve_parent_with_fallback(dsl, path)
    if key != "description" or not isinstance(parent, dict) or key not in parent:
        raise SchemaDescriptionPatchError(f"Unknown description path: {path}")
    return parent[key]


def _canonical_description_path(dsl: dict[str, Any], path: str) -> str:
    """Resolve a patch path to the one concrete description path in the DSL.

    Validation may recover an abbreviated path by field-name fallback.  Returning
    the recovered path here prevents a later write from reusing the abbreviated
    spelling and failing under the stricter direct traversal used by ``_set_description``.
    """
    normalized = _normalize_dsl_path(path)
    if normalized == "$.description" and isinstance(dsl.get("description"), str):
        return normalized
    parent, key = _resolve_parent_with_fallback(dsl, normalized)
    if key != "description" or not isinstance(parent, dict) or key not in parent:
        raise SchemaDescriptionPatchError(f"Unknown description path: {path}")
    field = normalized[2:].split(".")[-2] if normalized.startswith("$.") else ""
    matches = _paths_by_field_name(dsl, field)
    canonical = [candidate for candidate, _ in matches if _resolve_parent(dsl, candidate)[0] is parent]
    if len(canonical) == 1:
        return canonical[0]
    raise SchemaDescriptionPatchError(f"Unknown description path: {path}")


def _resolve_parent_with_fallback(dsl: dict[str, Any], path: str) -> tuple[Any, str]:
    """Resolve ``path``; if it fails, fall back to locating by trailing field name.

    LLMs frequently write DSL paths with a correct trailing field but a broken or
    abbreviated prefix (e.g. ``$.性质.description`` instead of the full key chain).
    If the trailing field name is unique among description-bearing nodes, resolve
    to that node. If the name is ambiguous, narrow by the parent field name (the
    non-meta segment before the field) — ``$.性质.items.properties.测试条件`` narrows
    to the 测试条件 whose parent is 性质. Still ambiguous: keep it a failure.
    """
    path = _normalize_dsl_path(path)
    try:
        return _resolve_parent(dsl, path)
    except SchemaDescriptionPatchError:
        pass
    segments = path[2:].split(".")
    if segments[-1] != "description":
        raise SchemaDescriptionPatchError(f"Unknown description path: {path}")
    non_meta = [s for s in segments if s not in ("items", "properties", "description")]
    if not non_meta:
        raise SchemaDescriptionPatchError(f"Unknown description path: {path}")
    field = non_meta[-1]
    matches = _paths_by_field_name(dsl, field)
    if len(matches) == 1:
        return _resolve_parent(dsl, matches[0][0])
    if len(non_meta) >= 2:
        parent = non_meta[-2]
        narrowed = [m for m in matches if m[1] == parent]
        if len(narrowed) == 1:
            return _resolve_parent(dsl, narrowed[0][0])
    # Items-shape hint: a path containing ".items." (e.g. ...反应条件.items.description)
    # targets an array's items sub-object, so prefer candidates whose node is an
    # array (has "items") over plain-object/string candidates of the same name.
    # This resolves the 反应条件/后处理步骤 same-name ambiguity where the outer
    # field is an array and the inner同名 string is not.
    if any(seg == "items" for seg in segments):
        array_matches = [m for m in matches if _node_is_array(dsl, m[0])]
        if len(array_matches) == 1:
            return _resolve_parent(dsl, array_matches[0][0])
    raise SchemaDescriptionPatchError(f"Unknown description path: {path}")


def _node_is_array(dsl: Any, desc_path: str) -> bool:
    """Return True if the node owning ``desc_path`` is an array (has ``items``)."""
    try:
        parent, _ = _resolve_parent(dsl, desc_path)
    except SchemaDescriptionPatchError:
        return False
    return isinstance(parent, dict) and isinstance(parent.get("items"), (dict, list))


def _parent_field_of_segments(segments: list[str]) -> str | None:
    """Return the second-to-last non-meta field name, or None if absent."""
    non_meta = [s for s in segments if s not in ("items", "properties")]
    return non_meta[-2] if len(non_meta) >= 2 else None


def _paths_by_field_name(dsl: Any, field: str) -> list[tuple[str, str]]:
    """Return ``(full_path, parent_field)`` for every description-bearing node named ``field``."""
    paths: list[tuple[str, str]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("description"), str):
                leaf = path.rsplit(".", 1)[-1]
                if leaf == field:
                    parent = _parent_field_of_segments([s for s in path.split(".") if s != "$"])
                    paths.append((path + ".description", parent))
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    walk(value, path + "." + key)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(dsl, "$")
    return paths


def ambiguous_candidates(dsl: Any, path: str) -> list[str]:
    """Return full description paths a broken ``path`` may refer to.

    Used as a closed-set fallback: when ``path`` does not resolve but its trailing
    field name matches description-bearing nodes, return all such full paths so an
    LLM can choose the intended one. Empty list means no candidates (unknown field
    or resolvable path).
    """
    # Use _resolve_description (with the key-existence check) so the "is this
    # resolvable" verdict matches validate_patch_document. _resolve_parent alone
    # returns the parent node even when that node lacks a "description" key
    # (e.g. an array's items object), so it would report "resolvable" and skip
    # the LLM closed-set fallback while validate still rejects the path.
    try:
        _resolve_description(dsl, path)
        return []
    except SchemaDescriptionPatchError:
        pass
    segments = path[2:].split(".")
    if segments[-1] != "description":
        return []
    non_meta = [s for s in segments if s not in ("items", "properties", "description")]
    if not non_meta:
        return []
    return [full for full, _ in _paths_by_field_name(dsl, non_meta[-1])]


def _set_description(dsl: dict[str, Any], path: str, description: str) -> None:
    parent, key = _resolve_parent(dsl, path)
    parent[key] = description


def _resolve_parent(dsl: dict[str, Any], path: str) -> tuple[Any, str]:
    path = _normalize_dsl_path(path)
    if not isinstance(path, str) or not path.startswith("$."):
        raise SchemaDescriptionPatchError(f"Patch path must start with '$.': {path!r}")
    segments = path[2:].split(".")
    if not segments or any(not segment for segment in segments) or segments[-1] != "description":
        raise SchemaDescriptionPatchError(f"Patch path must end in '.description': {path!r}")
    current: Any = dsl
    for segment in segments[:-1]:
        if not isinstance(current, dict):
            raise SchemaDescriptionPatchError(f"Unknown description path: {path}")
        if segment == "properties" and segment not in current:
            # A redundant ".properties" at a level that has none (e.g. the root,
            # which already is the object) is safe to skip — "properties" is a
            # structural meta key and never collides with a field name.
            continue
        if segment in current:
            current = current[segment]
            continue
        # Lenient fallback: DSL fields live under "properties"; the schema's
        # meta keys (type/required/description/items/properties) never collide
        # with field names, so auto-filling ".properties" is unambiguous.
        props = current.get("properties")
        if isinstance(props, dict) and segment in props:
            current = props[segment]
            continue
        raise SchemaDescriptionPatchError(f"Unknown description path: {path}")
    return current, segments[-1]
