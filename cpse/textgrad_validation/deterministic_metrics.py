from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any

from .artifacts import write_csv, write_json
from .existing_evaluation import load_arm_schema, load_existing_prediction


_IDENTITY_KEYS = ("身份标识", "身份标识符", "唯一标识", "sample_id", "identity_id", "id")
_UNIT_FACTORS = {
    "hz": ("frequency_hz", 1.0), "khz": ("frequency_hz", 1e3), "mhz": ("frequency_hz", 1e6), "ghz": ("frequency_hz", 1e9),
    "s": ("time_s", 1.0), "sec": ("time_s", 1.0), "min": ("time_s", 60.0), "h": ("time_s", 3600.0), "hr": ("time_s", 3600.0),
    "pa": ("pressure_pa", 1.0), "kpa": ("pressure_pa", 1e3), "mpa": ("pressure_pa", 1e6), "gpa": ("pressure_pa", 1e9),
    "m": ("length_m", 1.0), "cm": ("length_m", 1e-2), "mm": ("length_m", 1e-3), "μm": ("length_m", 1e-6), "µm": ("length_m", 1e-6), "um": ("length_m", 1e-6), "nm": ("length_m", 1e-9),
    "l": ("volume_l", 1.0), "ml": ("volume_l", 1e-3), "μl": ("volume_l", 1e-6), "µl": ("volume_l", 1e-6), "ul": ("volume_l", 1e-6),
    "kg": ("mass_g", 1e3), "g": ("mass_g", 1.0), "mg": ("mass_g", 1e-3),
}
_NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_PROPERTY_NAME_ALIASES = {
    "密度": "density", "孔隙率": "porosity", "平均泡孔直径": "average cellular diameter",
    "拉伸强度": "tensile strength", "拉伸模量": "tensile modulus", "断裂伸长率": "elongation",
    "伸长率": "elongation", "介电常数": "dielectric constant", "介电损耗": "dielectric loss",
    "玻璃化转变温度": "glass transition temperature", "5%失重分解温度": "degradation temperature at 5% weight loss",
    "热分解温度": "decomposition temperature", "吸水率": "water absorption", "接触角": "contact angle",
}


def _normalize_string(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _scalar_token(value: Any) -> str:
    if isinstance(value, str):
        return "str:" + _normalize_string(value)
    return type(value).__name__ + ":" + json.dumps(value, ensure_ascii=False, sort_keys=True)


def _leaf_facts(value: Any, path: str = "$") -> Counter[tuple[str, str]]:
    facts: Counter[tuple[str, str]] = Counter()
    if isinstance(value, dict):
        if not value:
            facts[(path, "empty_object")] += 1
        for key, child in value.items():
            facts.update(_leaf_facts(child, f"{path}.{key}"))
    elif isinstance(value, list):
        if not value:
            facts[(path + "[]", "empty_array")] += 1
        for child in value:
            facts.update(_leaf_facts(child, path + "[]"))
    else:
        facts[(path, _scalar_token(value))] += 1
    return facts


def _entity_identities(value: Any) -> Counter[str]:
    identities: Counter[str] = Counter()
    if not isinstance(value, dict) or not isinstance(value.get("聚合物"), list):
        return identities
    for record in value["聚合物"]:
        if not isinstance(record, dict):
            continue
        for key in _IDENTITY_KEYS:
            identity = record.get(key)
            if isinstance(identity, str) and identity.strip():
                identities[_normalize_string(identity)] += 1
                break
    return identities


def _prf(predicted: Counter[Any], gold: Counter[Any]) -> tuple[float, float, float]:
    true_positive = sum((predicted & gold).values())
    predicted_count = sum(predicted.values())
    gold_count = sum(gold.values())
    precision = true_positive / predicted_count if predicted_count else (1.0 if not gold_count else 0.0)
    recall = true_positive / gold_count if gold_count else (1.0 if not predicted_count else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _entity_key(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    for key in _IDENTITY_KEYS + ("名称", "样品名称", "name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            normalized = _normalize_string(value).casefold()
            match = re.fullmatch(r"doi:[^_]+_(.+)_\d+", normalized)
            return match.group(1) if match else normalized.removeprefix("doi:")
    return None


def _property_name(record: dict[str, Any]) -> str:
    abbreviation = _normalize_string(str(record.get("缩写", ""))).casefold()
    if abbreviation:
        return "abbr:" + abbreviation
    name = _normalize_string(str(record.get("名称", ""))).casefold()
    return _PROPERTY_NAME_ALIASES.get(name, name)


def _quantity(value: Any, unit: Any) -> tuple[str, Any]:
    text = _normalize_string(str(value)) if value is not None else ""
    unit_text = _normalize_string(str(unit)).replace("·", "").replace("⁻", "-").casefold() if unit is not None else ""
    if not text:
        return "missing", ""
    if text and _NUMERIC.fullmatch(text):
        number = float(text)
        dimension, factor = _UNIT_FACTORS.get(unit_text, ("unit:" + unit_text, 1.0))
        return dimension, number * factor
    return "text:" + unit_text, text.casefold()


def _key_conditions(record: Any) -> tuple[tuple[str, tuple[str, Any]], ...]:
    if not isinstance(record, dict):
        return ()
    conditions = record.get("测试条件")
    if not isinstance(conditions, dict):
        return ()
    selected: list[tuple[str, tuple[str, Any]]] = []
    for key, value in conditions.items():
        if not any(marker in str(key) for marker in ("温度", "频率", "速率", "气氛", "压力", "厚度")):
            continue
        if isinstance(value, dict) and "单值" in value:
            selected.append((str(key), _quantity(value.get("单值"), value.get("单位"))))
        elif not isinstance(value, (dict, list)):
            selected.append((str(key), _quantity(value, "")))
    return tuple(sorted(selected))


def _property_records(value: Any) -> list[dict[str, Any]]:
    properties: list[dict[str, Any]] = []
    if not isinstance(value, dict) or not isinstance(value.get("聚合物"), list):
        return properties
    for entity in value["聚合物"]:
        identity = _entity_key(entity)
        if identity is None or not isinstance(entity, dict) or not isinstance(entity.get("性质"), list):
            continue
        for prop in entity["性质"]:
            if not isinstance(prop, dict):
                continue
            raw_value = prop.get("值") if isinstance(prop.get("值"), dict) else {}
            unit = raw_value.get("单位", "")
            properties.append({
                "entity": identity,
                "name": _property_name(prop),
                "minimum": _quantity(raw_value.get("最小", ""), unit),
                "maximum": _quantity(raw_value.get("最大", ""), unit),
                "single": _quantity(raw_value.get("单值", ""), unit),
                "conditions": _key_conditions(prop),
            })
    return properties


def _normalized_equal(left: Any, right: Any) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            return False
        return all(_normalized_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=0.01, abs_tol=1e-12)
    return left == right


def _property_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(_normalized_equal(left[key], right[key]) for key in ("entity", "name", "minimum", "maximum", "single", "conditions"))


def _property_prf(prediction: Any, gold: Any) -> tuple[float, float, float, int, int, int]:
    predicted = _property_records(prediction)
    expected = _property_records(gold)
    available = set(range(len(expected)))
    matched = 0
    for item in predicted:
        match = next((index for index in sorted(available) if _property_matches(item, expected[index])), None)
        if match is not None:
            available.remove(match)
            matched += 1
    precision = matched / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = matched / len(expected) if expected else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, matched, len(predicted), len(expected)


def compare_prediction_to_gold(prediction, gold):
    predicted_facts = _leaf_facts(prediction)
    gold_facts = _leaf_facts(gold)
    matched_leaf_count = sum((predicted_facts & gold_facts).values())
    leaf_precision, leaf_recall, leaf_f1 = _prf(predicted_facts, gold_facts)
    predicted_entities = _entity_identities(prediction)
    gold_entities = _entity_identities(gold)
    entity_precision, entity_recall, entity_f1 = _prf(predicted_entities, gold_entities)
    property_precision, property_recall, property_f1, property_matched, property_predicted, property_gold = _property_prf(prediction, gold)
    predicted_keys = {_entity_key(item) for item in prediction.get("聚合物", [])} if isinstance(prediction, dict) and isinstance(prediction.get("聚合物"), list) else set()
    gold_keys = {_entity_key(item) for item in gold.get("聚合物", [])} if isinstance(gold, dict) and isinstance(gold.get("聚合物"), list) else set()
    predicted_keys.discard(None)
    gold_keys.discard(None)
    return {
        "strict_leaf_precision": leaf_precision,
        "strict_leaf_recall": leaf_recall,
        "strict_leaf_f1": leaf_f1,
        "matched_leaf_count": matched_leaf_count,
        "entity_identity_precision": entity_precision,
        "entity_identity_recall": entity_recall,
        "entity_identity_f1": entity_f1,
        "predicted_leaf_count": sum(predicted_facts.values()),
        "gold_leaf_count": sum(gold_facts.values()),
        "predicted_entity_count": sum(predicted_entities.values()),
        "gold_entity_count": sum(gold_entities.values()),
        "aligned_entity_count": len(predicted_keys & gold_keys),
        "unaligned_predicted_entity_count": len(predicted_keys - gold_keys),
        "unaligned_gold_entity_count": len(gold_keys - predicted_keys),
        "property_tuple_precision": property_precision,
        "property_tuple_recall": property_recall,
        "property_tuple_f1": property_f1,
        "property_tuple_matched_count": property_matched,
        "property_tuple_predicted_count": property_predicted,
        "property_tuple_gold_count": property_gold,
        "normalized_metric_scope": "entity_aligned_canonicalizable_properties_only",
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, bool))]
    return sum(values) / len(values) if values else None


def _micro_leaf_prf(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    matched = sum(int(row["matched_leaf_count"]) for row in rows)
    predicted = sum(int(row["predicted_leaf_count"]) for row in rows)
    gold = sum(int(row["gold_leaf_count"]) for row in rows)
    precision = matched / predicted if predicted else (1.0 if not gold else 0.0)
    recall = matched / gold if gold else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "strict_leaf_matched_count": matched,
        "strict_leaf_predicted_count": predicted,
        "strict_leaf_gold_count": gold,
        "strict_leaf_micro_precision": precision,
        "strict_leaf_micro_recall": recall,
        "strict_leaf_micro_f1": f1,
    }


def _micro_property_prf(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    matched = sum(int(row["property_tuple_matched_count"]) for row in rows)
    predicted = sum(int(row["property_tuple_predicted_count"]) for row in rows)
    gold = sum(int(row["property_tuple_gold_count"]) for row in rows)
    precision = matched / predicted if predicted else (1.0 if not gold else 0.0)
    recall = matched / gold if gold else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "property_tuple_matched_count": matched,
        "property_tuple_predicted_count": predicted,
        "property_tuple_gold_count": gold,
        "property_tuple_micro_precision": precision,
        "property_tuple_micro_recall": recall,
        "property_tuple_micro_f1": f1,
    }


def evaluate_existing_run(config, run_id):
    """Compute strict, model-free metrics over frozen blind predictions."""
    run_dir = config.output_root / run_id
    output_dir = run_dir / "deterministic_metrics"
    rows: list[dict[str, Any]] = []
    for arm in ("baseline", "optimized"):
        documents_root = run_dir / "blind_test" / arm / "documents"
        if not documents_root.is_dir():
            continue
        schema = load_arm_schema(config, run_dir, arm)
        for document_dir in sorted(path for path in documents_root.iterdir() if path.is_dir()):
            document_id = document_dir.name
            gold_path = config.data_dir / f"{document_id}.json"
            if not gold_path.is_file():
                continue
            artifact = load_existing_prediction(document_dir, schema)
            if artifact is None:
                continue
            try:
                prediction = json.loads(artifact.raw_response)
            except (json.JSONDecodeError, TypeError, ValueError):
                prediction = None
            gold = json.loads(gold_path.read_text(encoding="utf-8"))
            metrics = compare_prediction_to_gold(prediction, gold) if prediction is not None else compare_prediction_to_gold({}, gold)
            row = {"document_id": document_id, "arm": arm, "schema_valid": artifact.is_valid, **metrics}
            rows.append(row)
            write_json(output_dir / "documents" / document_id / f"{arm}.json", row)
    arm_summaries: dict[str, Any] = {}
    for arm in ("baseline", "optimized"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        arm_summaries[arm] = {
            "document_count": len(arm_rows),
            "schema_valid_rate": _mean(arm_rows, "schema_valid"),
            "strict_leaf_precision_mean": _mean(arm_rows, "strict_leaf_precision"),
            "strict_leaf_recall_mean": _mean(arm_rows, "strict_leaf_recall"),
            "strict_leaf_f1_mean": _mean(arm_rows, "strict_leaf_f1"),
            "entity_identity_precision_mean": _mean(arm_rows, "entity_identity_precision"),
            "entity_identity_recall_mean": _mean(arm_rows, "entity_identity_recall"),
            "entity_identity_f1_mean": _mean(arm_rows, "entity_identity_f1"),
            "property_tuple_precision_mean": _mean(arm_rows, "property_tuple_precision"),
            "property_tuple_recall_mean": _mean(arm_rows, "property_tuple_recall"),
            "property_tuple_f1_mean": _mean(arm_rows, "property_tuple_f1"),
            **_micro_leaf_prf(arm_rows),
            **_micro_property_prf(arm_rows),
        }
    by_document: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_document.setdefault(str(row["document_id"]), {})[str(row["arm"])] = row
    paired_rows = [arms for arms in by_document.values() if "baseline" in arms and "optimized" in arms]
    paired_summary = {"document_count": len(paired_rows)}
    for metric in ("strict_leaf_precision", "strict_leaf_recall", "strict_leaf_f1", "entity_identity_f1", "property_tuple_precision", "property_tuple_recall", "property_tuple_f1"):
        deltas = [float(arms["optimized"][metric]) - float(arms["baseline"][metric]) for arms in paired_rows]
        paired_summary[f"{metric}_delta_mean"] = sum(deltas) / len(deltas) if deltas else None
    summary = {
        "metric_scope": "strict_deterministic_auxiliary_only",
        "semantic_equivalence_handled": False,
        "normalized_metric_scope": "entity_aligned_canonicalizable_properties_only",
        "free_text_semantic_equivalence_handled": False,
        "document_count": len({row["document_id"] for row in rows}),
        "paired": paired_summary,
        **arm_summaries,
    }
    columns = list(rows[0]) if rows else ["document_id", "arm", "schema_valid"]
    write_csv(output_dir / "documents.csv", rows, columns)
    write_json(output_dir / "summary.json", summary)
    return output_dir
