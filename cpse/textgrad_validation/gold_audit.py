from __future__ import annotations

import json
from typing import Any, Protocol

from .models import GoldAuditFinding, GoldAuditResult

_AUDIT_STATUSES = {"supported", "unsupported", "ambiguous", "possible_omission"}


class PdfAuditTransport(Protocol):
    def complete_pdf_json(self, *, pdf_path: Any, system_prompt: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]: ...


class GoldAuditor:
    """PDF-backed audit of gold evidence that never sees a model prediction."""

    def __init__(self, transport: PdfAuditTransport, system_prompt: str):
        self._transport = transport
        self._system_prompt = system_prompt

    def audit(self, *, pdf_path: Any, schema: dict[str, Any], gold: Any) -> GoldAuditResult:
        payload = {"schema": schema, "gold": gold, "audit_mode": "gold_evidence_only"}
        raw, _metadata = self._transport.complete_pdf_json(
            pdf_path=pdf_path,
            system_prompt=self._system_prompt,
            payload=payload,
        )
        try:
            return parse_gold_audit_result(raw)
        except ValueError as exc:
            correction_payload = {
                **payload,
                "audit_mode": "gold_evidence_format_correction",
                "previous_response": raw,
                "format_error": str(exc),
            }
            corrected_raw, _corrected_metadata = self._transport.complete_pdf_json(
                pdf_path=pdf_path,
                system_prompt=self._system_prompt,
                payload=correction_payload,
            )
            return parse_gold_audit_result(corrected_raw)


def parse_gold_audit_result(raw_response: str) -> GoldAuditResult:
    try:
        value = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gold audit returned invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict) or "findings" not in value or not isinstance(value["findings"], list):
        raise ValueError("Gold audit result must be an object with a findings array.")
    findings: list[GoldAuditFinding] = []
    for item in value["findings"]:
        if not isinstance(item, dict) or {"path", "status", "page", "evidence"} - set(item):
            raise ValueError("Gold audit finding has an invalid shape.")
        path, status, page, evidence = item["path"], item["status"], item["page"], item["evidence"]
        path = _normalize_path(path)
        if status not in _AUDIT_STATUSES:
            raise ValueError("Gold audit finding status is invalid.")
        page = _normalize_page(page)
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Gold audit finding evidence must be non-empty text.")
        findings.append(GoldAuditFinding(path=path, status=status, page=page, evidence=evidence.strip()))
    return GoldAuditResult(findings=tuple(findings))


def _normalize_path(value: object) -> str:
    if not isinstance(value, str) or not (path := value.strip()):
        raise ValueError("Gold audit finding path must be non-empty text.")
    if path.startswith("$"):
        return path
    if path.startswith(".") or path.startswith("["):
        return f"${path}"
    if path[0].isalpha() or path[0] == "_":
        return f"$.{path}"
    raise ValueError("Gold audit finding path must be a JSONPath-like field path.")


def _normalize_page(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError("Gold audit finding page must be text, an integer, or null.")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and (page := value.strip()):
        return page
    raise ValueError("Gold audit finding page must be non-empty text, an integer, or null.")
