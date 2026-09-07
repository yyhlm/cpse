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
_CONDITION_NAME_ALIASES = {
    "温度": "temperature", "测试温度": "temperature",
    "频率": "frequency", "测试频率": "frequency",
    "气氛": "atmosphere", "测试气氛": "atmosphere",
    "压力": "pressure", "测试压力": "pressure",
    "厚度": "thickness", "样品厚度": "thickness",
    "升温速率": "heating_rate", "温度扫描速率": "temperature_scan_rate",
    "扫描速率": "scan_rate", "测试速率": "test_rate", "拉伸速率": "tensile_rate",
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
        if unit_text in {"°c", "℃", "celsius", "degc"}:
            return "temperature_k", number + 273.15
        if unit_text in {"k", "kelvin"}:
            return "temperature_k", number
        if unit_text in {"%", "percent", "wt%", "vol%"}:
            return "ratio_percent", number
        if unit_text in {"cm-1", "cm^-1", "1/cm"}:
            return "wavenumber_cm-1", number
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


def _normalized_label(value: Any) -> str:
    text = _normalize_string(str(value)).casefold()
    return re.sub(r"[\s_\-–—:/]+", "", text)


def _canonical_identity(value: Any) -> str:
    text = _normalize_string(str(value)).casefold()
    match = re.fullmatch(r"doi:[^_]+_(.+)_\d+", text)
    if match:
        text = match.group(1)
    return _normalized_label(text.removeprefix("doi:"))


def _identity_tokens(record: Any) -> tuple[set[str], set[str]]:
    if not isinstance(record, dict):
        return set(), set()
    identities = {
        _canonical_identity(record[key])
        for key in _IDENTITY_KEYS
        if isinstance(record.get(key), str) and str(record[key]).strip()
    }
    names = {
        _normalized_label(record[key])
        for key in ("名称", "样品名称", "缩写", "name")
        if isinstance(record.get(key), str) and str(record[key]).strip()
    }
    return {item for item in identities if item}, {item for item in names if item}


def _record_similarity(left: Any, right: Any) -> float:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return 0.0
    shared = 0
    compared = 0
    for key in ("聚合物分类名称", "聚合物分类编码", "样本形态", "结构特征_L1", "结构特征_L2"):
        left_value = _normalized_label(left.get(key, ""))
        right_value = _normalized_label(right.get(key, ""))
        if not left_value or not right_value:
            continue
        compared += 1
        shared += int(left_value == right_value)
    left_props = {_normalized_label(item.get("名称", "")) for item in left.get("性质", []) if isinstance(item, dict)}
    right_props = {_normalized_label(item.get("名称", "")) for item in right.get("性质", []) if isinstance(item, dict)}
    left_props.discard("")
    right_props.discard("")
    if left_props and right_props:
        compared += 1
        shared += len(left_props & right_props) / len(left_props | right_props)
    return shared / compared if compared else 0.0


def _align_entities(prediction: Any, gold: Any) -> tuple[list[tuple[int, int]], int, int]:
    predicted = prediction.get("聚合物", []) if isinstance(prediction, dict) else []
    expected = gold.get("聚合物", []) if isinstance(gold, dict) else []
    predicted = predicted if isinstance(predicted, list) else []
    expected = expected if isinstance(expected, list) else []
    available_pred = set(range(len(predicted)))
    available_gold = set(range(len(expected)))
    pairs: list[tuple[int, int]] = []

    def match_tier(score_fn) -> None:
        candidates = []
        for pred_index in available_pred:
            for gold_index in available_gold:
                score = score_fn(predicted[pred_index], expected[gold_index])
                if score > 0:
                    candidates.append((score, pred_index, gold_index))
        for _, pred_index, gold_index in sorted(candidates, key=lambda row: (-row[0], row[1], row[2])):
            if pred_index in available_pred and gold_index in available_gold:
                pairs.append((pred_index, gold_index))
                available_pred.remove(pred_index)
                available_gold.remove(gold_index)

    match_tier(lambda left, right: 1.0 if _identity_tokens(left)[0] & _identity_tokens(right)[0] else 0.0)
    match_tier(lambda left, right: 1.0 if _identity_tokens(left)[1] & _identity_tokens(right)[1] else 0.0)
    match_tier(lambda left, right: _record_similarity(left, right) if _record_similarity(left, right) >= 0.5 else 0.0)
    return pairs, len(predicted), len(expected)


def _property_aliases(record: Any) -> set[str]:
    if not isinstance(record, dict):
        return set()
    aliases = set()
    for key in ("名称", "缩写", "类别"):
        value = _normalize_string(str(record.get(key, ""))).casefold()
        if value:
            aliases.add(_normalized_label(_PROPERTY_NAME_ALIASES.get(value, value)))
    return aliases


def _value_slots(record: Any) -> dict[str, tuple[str, Any]]:
    if not isinstance(record, dict) or not isinstance(record.get("值"), dict):
        return {}
    value = record["值"]
    unit = value.get("单位", "")
    slots = {}
    for key in ("最小", "最大", "单值"):
        raw = value.get(key, "")
        if raw not in (None, ""):
            slots[key] = _quantity(raw, unit)
    return slots


def _condition_slots(record: Any) -> dict[str, tuple[str, Any]]:
    if not isinstance(record, dict) or not isinstance(record.get("测试条件"), dict):
        return {}
    slots = {}
    for key, raw in record["测试条件"].items():
        canonical_key = _CONDITION_NAME_ALIASES.get(str(key), _normalized_label(key))
        if isinstance(raw, dict):
            if raw.get("单值", "") not in (None, ""):
                slots[canonical_key] = _quantity(raw.get("单值"), raw.get("单位", ""))
            for bound in ("最小", "最大"):
                if raw.get(bound, "") not in (None, ""):
                    slots[f"{canonical_key}:{bound}"] = _quantity(raw.get(bound), raw.get("单位", ""))
        elif raw not in (None, "", [], {}):
            slots[canonical_key] = _quantity(raw, "")
    return slots


def _slot_match_count(predicted: dict[str, Any], expected: dict[str, Any]) -> int:
    return sum(
        1 for key, value in predicted.items()
        if key in expected and _normalized_equal(value, expected[key])
    )


def _align_properties(predicted: Any, expected: Any) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], int, int]:
    predicted_props = predicted.get("性质", []) if isinstance(predicted, dict) else []
    expected_props = expected.get("性质", []) if isinstance(expected, dict) else []
    predicted_props = [item for item in predicted_props if isinstance(item, dict)] if isinstance(predicted_props, list) else []
    expected_props = [item for item in expected_props if isinstance(item, dict)] if isinstance(expected_props, list) else []
    available = set(range(len(expected_props)))
    pairs = []
    for predicted_prop in predicted_props:
        candidates = []
        for index in available:
            if not (_property_aliases(predicted_prop) & _property_aliases(expected_props[index])):
                continue
            value_overlap = _slot_match_count(_value_slots(predicted_prop), _value_slots(expected_props[index]))
            condition_overlap = _slot_match_count(_condition_slots(predicted_prop), _condition_slots(expected_props[index]))
            candidates.append((value_overlap + condition_overlap, index))
        if candidates:
            _, index = max(candidates, key=lambda row: (row[0], -row[1]))
            available.remove(index)
            pairs.append((predicted_prop, expected_props[index]))
    return pairs, len(predicted_props), len(expected_props)


def _schema_aware_metrics(prediction: Any, gold: Any) -> dict[str, Any]:
    predicted_entities = prediction.get("聚合物", []) if isinstance(prediction, dict) else []
    gold_entities = gold.get("聚合物", []) if isinstance(gold, dict) else []
    predicted_entities = predicted_entities if isinstance(predicted_entities, list) else []
    gold_entities = gold_entities if isinstance(gold_entities, list) else []
    entity_pairs, predicted_entity_count, gold_entity_count = _align_entities(prediction, gold)
    entity_precision, entity_recall, entity_f1 = _prf(Counter({index: 1 for index in range(len(entity_pairs))}), Counter({index: 1 for index in range(gold_entity_count)}))
    entity_precision = len(entity_pairs) / predicted_entity_count if predicted_entity_count else (1.0 if not gold_entity_count else 0.0)
    entity_recall = len(entity_pairs) / gold_entity_count if gold_entity_count else (1.0 if not predicted_entity_count else 0.0)
    entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if entity_precision + entity_recall else 0.0

    property_pairs = []
    predicted_property_count = 0
    gold_property_count = 0
    for pred_index, gold_index in entity_pairs:
        pairs, pred_count, gold_count = _align_properties(predicted_entities[pred_index], gold_entities[gold_index])
        property_pairs.extend(pairs)
        predicted_property_count += pred_count
        gold_property_count += gold_count
    for index, entity in enumerate(predicted_entities):
        if index not in {pair[0] for pair in entity_pairs} and isinstance(entity, dict) and isinstance(entity.get("性质"), list):
            predicted_property_count += len(entity["性质"])
    for index, entity in enumerate(gold_entities):
        if index not in {pair[1] for pair in entity_pairs} and isinstance(entity, dict) and isinstance(entity.get("性质"), list):
            gold_property_count += len(entity["性质"])
    property_matched = len(property_pairs)
    property_precision = property_matched / predicted_property_count if predicted_property_count else (1.0 if not gold_property_count else 0.0)
    property_recall = property_matched / gold_property_count if gold_property_count else (1.0 if not predicted_property_count else 0.0)
    property_f1 = 2 * property_precision * property_recall / (property_precision + property_recall) if property_precision + property_recall else 0.0

    all_predicted_properties = [
        prop for entity in predicted_entities if isinstance(entity, dict)
        for prop in (entity.get("性质", []) if isinstance(entity.get("性质"), list) else [])
        if isinstance(prop, dict)
    ]
    all_gold_properties = [
        prop for entity in gold_entities if isinstance(entity, dict)
        for prop in (entity.get("性质", []) if isinstance(entity.get("性质"), list) else [])
        if isinstance(prop, dict)
    ]
    value_matched = 0
    value_predicted = sum(len(_value_slots(prop)) for prop in all_predicted_properties)
    value_gold = sum(len(_value_slots(prop)) for prop in all_gold_properties)
    condition_matched = 0
    condition_predicted = sum(len(_condition_slots(prop)) for prop in all_predicted_properties)
    condition_gold = sum(len(_condition_slots(prop)) for prop in all_gold_properties)
    for predicted_prop, gold_prop in property_pairs:
        predicted_values = _value_slots(predicted_prop)
        gold_values = _value_slots(gold_prop)
        value_matched += _slot_match_count(predicted_values, gold_values)
        predicted_conditions = _condition_slots(predicted_prop)
        gold_conditions = _condition_slots(gold_prop)
        condition_matched += _slot_match_count(predicted_conditions, gold_conditions)

    def scores(matched: int, predicted_count: int, gold_count: int) -> tuple[float, float, float]:
        precision = matched / predicted_count if predicted_count else (1.0 if not gold_count else 0.0)
        recall = matched / gold_count if gold_count else (1.0 if not predicted_count else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    value_precision, value_recall, value_f1 = scores(value_matched, value_predicted, value_gold)
    condition_precision, condition_recall, condition_f1 = scores(condition_matched, condition_predicted, condition_gold)
    total_matched = len(entity_pairs) + property_matched + value_matched + condition_matched
    total_predicted = predicted_entity_count + predicted_property_count + value_predicted + condition_predicted
    total_gold = gold_entity_count + gold_property_count + value_gold + condition_gold
    slot_precision, slot_recall, slot_f1 = scores(total_matched, total_predicted, total_gold)
    return {
        "aligned_entity_precision": entity_precision,
        "aligned_entity_recall": entity_recall,
        "aligned_entity_f1": entity_f1,
        "aligned_entity_matched_count": len(entity_pairs),
        "aligned_entity_predicted_count": predicted_entity_count,
        "aligned_entity_gold_count": gold_entity_count,
        "property_detection_precision": property_precision,
        "property_detection_recall": property_recall,
        "property_detection_f1": property_f1,
        "property_detection_matched_count": property_matched,
        "property_detection_predicted_count": predicted_property_count,
        "property_detection_gold_count": gold_property_count,
        "value_unit_precision": value_precision,
        "value_unit_recall": value_recall,
        "value_unit_f1": value_f1,
        "value_unit_matched_count": value_matched,
        "value_unit_predicted_count": value_predicted,
        "value_unit_gold_count": value_gold,
        "condition_slot_precision": condition_precision,
        "condition_slot_recall": condition_recall,
        "condition_slot_f1": condition_f1,
        "condition_slot_matched_count": condition_matched,
        "condition_slot_predicted_count": condition_predicted,
        "condition_slot_gold_count": condition_gold,
        "schema_aware_slot_precision": slot_precision,
        "schema_aware_slot_recall": slot_recall,
        "schema_aware_slot_f1": slot_f1,
        "schema_aware_slot_matched_count": total_matched,
        "schema_aware_slot_predicted_count": total_predicted,
        "schema_aware_slot_gold_count": total_gold,
    }


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
        **_schema_aware_metrics(prediction, gold),
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


def _micro_schema_aware_prf(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    matched = sum(int(row["schema_aware_slot_matched_count"]) for row in rows)
    predicted = sum(int(row["schema_aware_slot_predicted_count"]) for row in rows)
    gold = sum(int(row["schema_aware_slot_gold_count"]) for row in rows)
    precision = matched / predicted if predicted else (1.0 if not gold else 0.0)
    recall = matched / gold if gold else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "schema_aware_slot_matched_count": matched,
        "schema_aware_slot_predicted_count": predicted,
        "schema_aware_slot_gold_count": gold,
        "schema_aware_slot_micro_precision": precision,
        "schema_aware_slot_micro_recall": recall,
        "schema_aware_slot_micro_f1": f1,
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
            "aligned_entity_f1_mean": _mean(arm_rows, "aligned_entity_f1"),
            "property_detection_f1_mean": _mean(arm_rows, "property_detection_f1"),
            "value_unit_f1_mean": _mean(arm_rows, "value_unit_f1"),
            "condition_slot_f1_mean": _mean(arm_rows, "condition_slot_f1"),
            "schema_aware_slot_precision_mean": _mean(arm_rows, "schema_aware_slot_precision"),
            "schema_aware_slot_recall_mean": _mean(arm_rows, "schema_aware_slot_recall"),
            "schema_aware_slot_f1_mean": _mean(arm_rows, "schema_aware_slot_f1"),
            **_micro_leaf_prf(arm_rows),
            **_micro_property_prf(arm_rows),
            **_micro_schema_aware_prf(arm_rows),
        }
    by_document: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_document.setdefault(str(row["document_id"]), {})[str(row["arm"])] = row
    paired_rows = [arms for arms in by_document.values() if "baseline" in arms and "optimized" in arms]
    paired_summary = {"document_count": len(paired_rows)}
    for metric in (
        "strict_leaf_precision", "strict_leaf_recall", "strict_leaf_f1",
        "entity_identity_f1", "property_tuple_precision", "property_tuple_recall", "property_tuple_f1",
        "aligned_entity_f1", "property_detection_f1", "value_unit_f1",
        "condition_slot_f1", "schema_aware_slot_precision", "schema_aware_slot_recall",
        "schema_aware_slot_f1",
    ):
        deltas = [float(arms["optimized"][metric]) - float(arms["baseline"][metric]) for arms in paired_rows]
        paired_summary[f"{metric}_delta_mean"] = sum(deltas) / len(deltas) if deltas else None
    summary = {
        "metric_scope": "strict_deterministic_auxiliary_only",
        "semantic_equivalence_handled": False,
        "normalized_metric_scope": "entity_aligned_canonicalizable_properties_only",
        "free_text_semantic_equivalence_handled": False,
        "schema_aware_metric_scope": "entity_aligned_unit_normalized_schema_slots",
        "document_count": len({row["document_id"] for row in rows}),
        "paired": paired_summary,
        **arm_summaries,
    }
    columns = list(rows[0]) if rows else ["document_id", "arm", "schema_valid"]
    write_csv(output_dir / "documents.csv", rows, columns)
    write_json(output_dir / "summary.json", summary)
    return output_dir
