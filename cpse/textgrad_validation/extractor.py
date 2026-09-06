from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import PredictionArtifact
from .responses_client import ResponsesPdfClient
from .validator import fill_missing_required, validate_prediction


def _has_missing_required(artifact: PredictionArtifact) -> bool:
    """True if any validation error is a missing-required-property failure."""
    for error in artifact.validation_errors or ():
        if "is a required property" in error.get("message", ""):
            return True
    return False


def _parsed_or_raw(artifact: PredictionArtifact) -> Any:
    """Return the parsed prediction if available, else best-effort parse of raw."""
    if artifact.parsed_prediction is not None:
        return artifact.parsed_prediction
    try:
        return json.loads(artifact.raw_response)
    except (json.JSONDecodeError, ValueError):
        return {}


class PdfExtractor:
    def __init__(
        self,
        client: ResponsesPdfClient,
        *,
        schema: dict[str, Any],
        schema_text: str,
        system_prompt: str,
        max_repair_attempts: int = 1,
        schema_free: bool = False,
        demonstrations: tuple[dict[str, str], ...] = (),
    ):
        self._client = client
        self._schema = schema
        self._schema_text = schema_text
        self._system_prompt = system_prompt
        self._max_repair_attempts = max_repair_attempts
        self._schema_free = schema_free
        self._demonstrations = tuple(dict(item) for item in demonstrations)

    def extract(self, pdf_path: Path, prompt: str, *, schema_text: str | None = None, schema: dict[str, Any] | None = None) -> tuple[PredictionArtifact, dict[str, Any]]:
        # Per-call schema override lets alternating runs feed a candidate schema
        # DSL to the model (descriptions included) instead of the base schema.
        schema_text = schema_text or self._schema_text
        schema = schema if schema is not None else self._schema
        payload: dict[str, Any] = {"extraction_prompt": prompt}
        if not self._schema_free:
            payload["schema"] = schema_text
        if self._demonstrations:
            payload["demonstrations"] = [dict(item) for item in self._demonstrations]
        raw, metadata = self._client.complete_pdf_json(
            pdf_path=pdf_path,
            system_prompt=self._system_prompt,
            payload=payload,
        )
        validation_schema = {} if self._schema_free else schema
        artifact = validate_prediction(raw, validation_schema)
        attempts = 0
        # Deterministic repair first: fill absent required keys with structural
        # empties (no content invented). Resolves nested omissions like
        # 性质.测试条件.测试速率 missing 单值 without a model call.
        if not self._schema_free and not artifact.is_valid and _has_missing_required(artifact):
            filled = fill_missing_required(_parsed_or_raw(artifact), schema)
            filled_artifact = validate_prediction(json.dumps(filled, ensure_ascii=False), schema)
            if filled_artifact.is_valid:
                metadata = {**metadata, "repair_attempts": attempts, "deterministic_fill": True}
                return filled_artifact, metadata
            if filled_artifact.validation_errors and len(filled_artifact.validation_errors) < len(artifact.validation_errors or ()):
                artifact = filled_artifact
        while not artifact.is_valid and attempts < self._max_repair_attempts:
            attempts += 1
            repair_prompt = _build_repair_prompt(artifact)
            repair_payload = dict(payload)
            repair_payload["repair"] = repair_prompt
            raw, repair_meta = self._client.complete_pdf_json(
                pdf_path=pdf_path,
                system_prompt=self._system_prompt,
                payload=repair_payload,
            )
            metadata = {**metadata, "repair_attempts": attempts, "last_repair_metadata": repair_meta}
            artifact = validate_prediction(raw, validation_schema)
            # After each LLM repair, also try deterministic fill: the model may
            # have fixed the original error but introduced a new missing-required
            # key (e.g. dropped a 值.单值 while editing). Filling those structural
            # gaps avoids leaving a near-valid output invalid.
            if not self._schema_free and not artifact.is_valid and _has_missing_required(artifact):
                filled = fill_missing_required(_parsed_or_raw(artifact), schema)
                filled_artifact = validate_prediction(json.dumps(filled, ensure_ascii=False), schema)
                if filled_artifact.is_valid:
                    metadata["deterministic_fill"] = True
                    return filled_artifact, metadata
                if filled_artifact.validation_errors and len(filled_artifact.validation_errors) < len(artifact.validation_errors or ()):
                    artifact = filled_artifact
        if attempts:
            metadata.setdefault("repair_attempts", attempts)
        return artifact, metadata


def _build_repair_prompt(artifact: PredictionArtifact) -> str:
    """Tell the model where its previous output failed and ask for a corrected copy.

    Failure modes get targeted instructions:
    - JSON parse error: fix the syntax, keep all content.
    - Missing required fields: add them as empty string/object; keep all other content,
      never invent values.
    - Other schema violation: conform to the schema without changing content.
    """
    errors = artifact.validation_errors or ()
    snippet = artifact.raw_response[-1200:] if artifact.raw_response else ""
    first_error = errors[0] if errors else None
    is_parse = bool(first_error) and first_error["message"].startswith("Invalid JSON:")
    if is_parse:
        hint = f"Your previous output failed to parse: {first_error['message']}."
        instruction = "Fix the syntax error and preserve all previously extracted fields."
    else:
        missing = [error for error in errors if "is a required property" in error["message"]]
        if missing:
            listing = "\n".join(f"  - {error['path']}: {error['message']}" for error in missing)
            hint = (
                "Your previous output is missing required fields:\n"
                f"{listing}\n"
                "Add every missing required field, using an empty string/object when the value is absent. "
                "Never omit a required key."
            )
            instruction = (
                "Add only the listed missing required fields; preserve all previously extracted fields "
                "and do not invent content."
            )
        elif first_error:
            hint = f"Your previous output does not conform to the schema: {first_error['message']}."
            instruction = "Correct the output to conform to the schema without changing other content."
        else:
            hint = "Your previous output was not valid JSON."
            instruction = "Fix the syntax error and preserve all previously extracted fields."
    return (
        f"{hint}\n\n"
        "Return the SAME extraction content as a single valid JSON value. "
        "Do not add prose, markdown fences, or commentary — only the JSON object. "
        f"{instruction}\n\n"
        f"Tail of your previous output (for context):\n{snippet}"
    )
