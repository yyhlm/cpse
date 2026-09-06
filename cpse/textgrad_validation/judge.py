from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .models import JudgeResult, PathError, PredictionArtifact
from .validator import prediction_or_failure_payload

_CATEGORIES = {
    "omission",
    "hallucination",
    "incorrect_value",
    "incorrect_unit",
    "incorrect_condition",
    "schema_error",
}
_FORBIDDEN_INPUT_KEYS = {"pdf", "file", "file_data", "base64", "pdf_path"}
_GOLD_MATCH_BREAKDOWN_LIMITS = {
    "coverage": 45,
    "accuracy": 55,
}
_PDF_BREAKDOWN_LIMITS = {
    "document_sample": 10,
    "process": 30,
    "properties": 50,
    "characterization": 10,
}


class TextTransport(Protocol):
    def complete_json(self, *, system_prompt: str, payload: dict[str, Any]) -> str: ...

    def complete_pdf_json(
        self, *, pdf_path: Path, system_prompt: str, payload: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]: ...


class GoldJudge:
    """Scores a prediction solely against the fixed human gold annotation."""

    def __init__(
        self,
        transport: TextTransport,
        system_prompt: str,
        include_error_locations: bool = False,
        include_pdf: bool = False,
        include_schema: bool = True,
    ):
        self._transport = transport
        self._system_prompt = system_prompt
        self._include_error_locations = include_error_locations
        self._include_pdf = include_pdf
        self._include_schema = include_schema

    def judge(
        self,
        *,
        schema: dict[str, Any],
        prediction: PredictionArtifact,
        gold: Any,
        pdf_path: Path | None = None,
    ) -> JudgeResult:
        payload = {
            "prediction": prediction_or_failure_payload(prediction),
            "gold": gold,
        }
        if self._include_schema:
            payload = {"schema": schema, **payload}
        if self._include_pdf:
            if pdf_path is None or not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
                raise ValueError("PDF-aware judge requires a valid PDF path.")
            raw, _metadata = self._transport.complete_pdf_json(
                pdf_path=pdf_path, system_prompt=self._system_prompt, payload=payload
            )
        else:
            _assert_no_pdf_input(payload)
            raw = self._transport.complete_json(system_prompt=self._system_prompt, payload=payload)
        return parse_judge_result(raw, include_error_locations=self._include_error_locations, include_pdf=self._include_pdf)


def parse_judge_result(raw_response: str, *, include_error_locations: bool = False, include_pdf: bool = False) -> JudgeResult:
    try:
        value = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("Judge result must be a JSON object.")
    expected = {"score", "score_breakdown", "optimization_feedback"}
    if include_error_locations:
        expected.add("path_errors")
    missing = expected - set(value)
    if missing:
        raise ValueError(f"Judge result is missing required keys: {sorted(missing)}.")
    score = value["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= float(score) <= 100:
        raise ValueError("Judge score must be a number between 0 and 100.")
    breakdown = _parse_score_breakdown(value["score_breakdown"], include_pdf=include_pdf)
    if abs(float(score) - sum(breakdown.values())) > 1e-6:
        raise ValueError("Judge score must equal the sum of score_breakdown.")
    feedback = value["optimization_feedback"]
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("Judge optimization_feedback must be non-empty text.")
    errors = _parse_path_errors(value.get("path_errors", [])) if include_error_locations else ()
    return JudgeResult(
        score=float(score),
        optimization_feedback=feedback.strip(),
        path_errors=errors,
        score_breakdown=breakdown,
    )


def _parse_score_breakdown(raw_breakdown: Any, *, include_pdf: bool) -> dict[str, float]:
    limits = _PDF_BREAKDOWN_LIMITS if include_pdf else _GOLD_MATCH_BREAKDOWN_LIMITS
    if not isinstance(raw_breakdown, dict) or set(raw_breakdown) != set(limits):
        raise ValueError(f"Judge score_breakdown must contain exactly: {sorted(limits)}.")
    breakdown: dict[str, float] = {}
    for name, maximum in limits.items():
        value = raw_breakdown[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= maximum:
            raise ValueError(f"Judge score_breakdown.{name} must be a number between 0 and {maximum}.")
        if not include_pdf and not isinstance(value, int):
            raise ValueError(f"Judge score_breakdown.{name} must be an integer in non-PDF mode.")
        breakdown[name] = float(value)
    return breakdown


def _parse_path_errors(raw_errors: Any) -> tuple[PathError, ...]:
    if not isinstance(raw_errors, list):
        raise ValueError("Judge path_errors must be an array.")
    errors: list[PathError] = []
    for item in raw_errors:
        if not isinstance(item, dict) or {"path", "category", "evidence"} - set(item):
            raise ValueError("Judge path error has an invalid shape.")
        if not isinstance(item["path"], str) or not item["path"].startswith("$"):
            raise ValueError("Judge path error path must start with $.")
        if item["category"] not in _CATEGORIES:
            raise ValueError("Judge path error category is invalid.")
        if not isinstance(item["evidence"], str) or not item["evidence"].strip():
            raise ValueError("Judge path error evidence must be non-empty text.")
        errors.append(PathError(path=item["path"], category=item["category"], evidence=item["evidence"]))
    return tuple(errors)


def _assert_no_pdf_input(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = _FORBIDDEN_INPUT_KEYS & {str(key).lower() for key in value}
        if forbidden:
            raise ValueError(f"Gold judge payload cannot contain PDF input fields: {sorted(forbidden)}")
        for child in value.values():
            _assert_no_pdf_input(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_pdf_input(child)
